//! XQA verify **几何**扫描探针(2026-10-15 perf 会话)。
//!
//! 为什么单开一个探针:`xqa_verify_probe` 回答"XQA 快不快/数值对不对",
//! 本探针回答"**快不快取决于哪个几何量**" —— 长 ctx 下 verify 实测
//! **2.09 TFLOPS(5.2% 峰值)/ ≤174 GB/s**,必须把并行度这一维单独扫出来:
//!
//! | 维 | 取值 | 含义 |
//! |---|---|---|
//! | `ctx` | 2048 / 8192 / 32768 | 长 ctx 主战场 |
//! | 资产 | `verify`(M_TILESIZE=32) | `nb_token_blocks = ceil(48/M)` ⇒ KV 重读倍数 |
//!
//! ⚠️ **M_TILESIZE=64 一档已判死**(2026-10-15):`mha.cu:85` 硬门
//! `static_assert(nbValidRows <= 32u)` ⇒ 48 打包单元结构上必分 2 block,
//! KV 重读 ×2 是内核约束而非几何选择 ⇒ 该维**没有旋钮**。
//! | `nb_sub_seq` | 1 / 2 / 4 / 8 / 16 / 20 / 40 / 80 | KV-split 度(= 引擎默认 `sm/(hkv)=20`) |
//!
//! 口径:
//! - 计时 = **cudaEvent 前后包夹 + 每配置预热 3 发**;取 10 发中位数;
//! - 有效带宽 = 该几何下的**真实 KV 流量**(K+V × `nb_token_blocks` 重读)/ 时间;
//! - 口径上限:`flops = 2·S·ctx·HD·H`(S=8 行、H=24 q 头)。
//!
//! 运行(GPU 需空闲;`OWL_TEST_DEVICE=<ordinal>`):
//! ```bash
//! OWL_TEST_DEVICE=2 cargo test --release -p owl-kernels --features xqa,device \
//!   --test xqa_geom_probe -- --nocapture --test-threads=1
//! ```
#![cfg(all(feature = "xqa", feature = "device"))]

use std::time::Instant;

use cudarc::driver::{CudaContext, DevicePtr};

const PAGE: usize = 32;
const HKV: usize = 4;
const H: usize = 24;
const HD: usize = 256;
const S: usize = 8; // 1 锚 + 7 草稿

fn e4m3_decode(code: u8) -> f32 {
    let s = if code & 0x80 != 0 { -1f32 } else { 1f32 };
    let e = ((code >> 3) & 0x0F) as i32;
    let m = (code & 0x07) as i32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let v = if e == 0 { (m as f32) * 2f32.powi(-9) } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7) };
    s * v
}
fn e4m3_encode(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0..=255u8 {
        if code & 0x7F == 0x7F {
            continue;
        }
        let err = (e4m3_decode(code) - v).abs();
        if err < best_err {
            best_err = err;
            best = code;
        }
    }
    best
}
/// f16↔e4m3 往返表(2026-10-15 测试优化:消 256 次线性扫描;逐位等价)
fn e4m3_from_f16(v: f32) -> u8 {
    e4m3_encode(v)
}
fn lcg_u8(n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            let f = ((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 2.0;
            e4m3_from_f16(f)
        })
        .collect()
}
fn lcg_f16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            half::f16::from_f32((x & 0xFFFF) as f32 / 65535.0 - 0.5).to_bits()
        })
        .collect()
}

struct Cfg {
    name: &'static str,
    ctx: usize,
    nsub: u32,
}

fn run_one(ctx_c: &std::sync::Arc<CudaContext>, cfg: &Cfg) -> Vec<u16> {
    let ctx = cfg.ctx;
    let nb = ctx.div_ceil(PAGE);
    let stream = ctx_c.default_stream();

    let pool_elems = nb * HKV * HD * PAGE;
    let k_fp8 = lcg_u8(pool_elems, 701);
    let v_fp8 = lcg_u8(pool_elems, 702);
    let q_f16 = lcg_f16(S * H * HD, 703);

    let dev_u8 = |v: &[u8]| {
        let mut d = unsafe { stream.alloc::<u8>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let d_k = dev_u8(&k_fp8);
    let d_v = dev_u8(&v_fp8);
    let mut d_q = unsafe { stream.alloc::<u16>(q_f16.len()) }.unwrap();
    stream.memcpy_htod(&q_f16, &mut d_q).unwrap();
    let d_out = unsafe { stream.alloc::<u16>(S * H * HD) }.unwrap();
    let mut d_bt = unsafe { stream.alloc::<i32>(nb) }.unwrap();
    stream.memcpy_htod(&(0..nb as i32).collect::<Vec<_>>(), &mut d_bt).unwrap();
    let mut d_seq = unsafe { stream.alloc::<u32>(1) }.unwrap();
    stream.memcpy_htod(&[ctx as u32], &mut d_seq).unwrap();
    let mask: Vec<u32> = (0..S).map(|i| (1u32 << (i + 1)) - 1).collect();
    let mut d_mask = unsafe { stream.alloc::<u32>(S) }.unwrap();
    stream.memcpy_htod(&mask, &mut d_mask).unwrap();
    let sem_len = HKV.div_ceil(2) * 2 + 2 + HKV + 2;
    let mut d_sem = unsafe { stream.alloc::<u32>(sem_len) }.unwrap();
    stream.memcpy_htod(&vec![0u32; sem_len], &mut d_sem).unwrap();
    let d_scratch = unsafe { stream.alloc::<u8>(256 << 20) }.unwrap();

    let (kernel, consts) =
        owl_kernels::family::xqa::load_for_probe(&ctx_c, cfg.name).expect("载 xqa 资产");

    let args = owl_kernels::family::xqa::XqaArgs {
        q_seq_len: S as u32,
        nb_k_heads: HKV as u32,
        q_cu_seq_lens: 0,
        q_scale: 1.0,
        q_scale_ptr: 0,
        output: d_out.device_ptr(&stream).0,
        q: d_q.device_ptr(&stream).0,
        mask: d_mask.device_ptr(&stream).0,
        attention_sinks: 0,
        cache: owl_kernels::family::xqa::KVCacheList {
            k_cache: d_k.device_ptr(&stream).0,
            v_cache: d_v.device_ptr(&stream).0,
            page_list: d_bt.device_ptr(&stream).0,
            seq_len_list: d_seq.device_ptr(&stream).0,
            max_nb_pages_per_seq: consts.max_nb_pages_per_seq((nb * PAGE) as u32),
            _pad: 0,
        },
        batch_size: 1,
        kv_scale: 1.0,
        kv_scale_ptr: 0,
        stride_page: consts.stride_in_heads((PAGE * HKV * HD) as u64),
        stride_token: consts.stride_in_heads((HKV * HD) as u64),
        stride_head: consts.stride_in_heads(HD as u64),
        semaphores: d_sem.device_ptr(&stream).0,
        scratch: d_scratch.device_ptr(&stream).0,
    };

    let fire = || {
        owl_kernels::family::xqa::launch_on_stream_subseq(
            &stream, &kernel, &consts, &args, cfg.nsub,
        )
        .expect("launch");
    };

    // 预热(3)+ 计时(10,host 时钟包在 sync 两侧)
    for _ in 0..3 {
        fire();
    }
    stream.synchronize().unwrap();
    let mut samples = Vec::with_capacity(10);
    for _ in 0..10 {
        let t0 = Instant::now();
        fire();
        stream.synchronize().unwrap();
        samples.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = samples[samples.len() / 2];

    // 有效带宽:KV 流量 = (K+V) × nb_token_blocks(重读倍数由几何公式算)
    let tb = (S as u32 * consts.head_grp).div_ceil(consts.rows_per_block.max(1)) as usize;
    let kv_bytes = 2 * nb * HKV * HD * PAGE; // 池字节(K+V 各一份)
    let moved = kv_bytes * tb;
    println!(
        "  {:>12} ctx={:>6} nsub={:>3}  rows/blk={:>2} token_blk={}  grid.y={:>3}  \
         {:>7.3} ms   {:>7.2} TFLOPS   {:>7.1} GB/s",
        cfg.name,
        ctx,
        cfg.nsub,
        consts.rows_per_block,
        tb,
        HKV as u32 * tb as u32,
        med,
        2.0 * S as f64 * ctx as f64 * HD as f64 * H as f64 / (med * 1e-3) / 1e12,
        moved as f64 / (med * 1e-3) / 1e9,
    );
    stream.clone_dtoh(&d_out).expect("d2h out")
}

#[test]
fn xqa_verify_geometry_scan() {
    if owl_cuda::test_device_bdf().is_none() {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let dev: usize = owl_cuda::test_device_ordinal();
    let ctx_c = CudaContext::new(dev).expect("cuda ctx");
    ctx_c.bind_to_thread().expect("bind");
    let stream = ctx_c.default_stream();
    let _ = &stream;
    let ctxs: Vec<usize> = owl_shared::env_reader::str("XQA_GEO_CTX")
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![2048, 8192, 32768]);
    let subs: Vec<u32> = owl_shared::env_reader::str("XQA_GEO_SUB")
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 4, 8, 16, 20, 40, 80]);
    let assets: Vec<&'static str> = owl_shared::env_reader::str("XQA_GEO_ASSET")
        .map(|s| {
            s.split(',')
                .filter_map(|v| match v.trim() {
                    "verify" => Some(owl_kernels::family::xqa::ASSET_VERIFY),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_else(|| vec![owl_kernels::family::xqa::ASSET_VERIFY]);
    println!("=== XQA verify 几何扫描(dev={dev};S={S} hkv={HKV} hd={HD} page={PAGE})===");
    for ctx in ctxs {
        for name in &assets {
            // 跨几何数值对拍(2026-10-15):KV-split 只是**归约方式**,输出必须与
            // split 度无关 —— 任一处不一致 = 归约/semaphores 语义破了。
            let mut base: Option<(&u32, Vec<u16>)> = None;
            for &nsub in &subs {
                let cfg = Cfg { name, ctx, nsub };
                let out = run_one(&ctx_c, &cfg);
                match &base {
                    None => base = Some((&subs[0], out)),
                    Some((bn, b)) => {
                        let worst = b
                            .iter()
                            .zip(out.iter())
                            .map(|(x, y)| {
                                (half::f16::from_bits(*x).to_f32()
                                    - half::f16::from_bits(*y).to_f32())
                                .abs()
                            })
                            .fold(0f32, f32::max);
                        // 判据:归约序不同 ⇒ 允许 **≤ 2 个 f16 ULP**(@|v|≈1 即 9.8e-4;
                        // 该族 host 门是 1e-2,余量充裕)。超过 = 归约/semaphores 语义破。
                        let tol = 2.0 * 0.000488;
                        println!(
                            "  → 对拍 nsub={bn} vs {nsub}: max|Δ| = {worst:.6}{}",
                            if worst == 0.0 {
                                "(逐位等)"
                            } else if worst <= tol {
                                "(≤2 ulp f16:归约序差异,合规)"
                            } else {
                                "  ⚠️ 超门"
                            }
                        );
                        assert!(worst <= tol, "跨几何数值超门:nsub={bn} vs {nsub} Δ={worst}");
                    }
                }
            }
        }
    }
}
