//! **FI plan 键 × ctx 扫描门**(2026-10-19;设计卷 `docs/arch/kvmem-选择层-设计与施工.md` §8.30 的 F2-a)。
//!
//! # 门是什么
//!
//! **同一个 `FiRuntime`(同一 plan 缓存)里,固定 T、逐档加大 ctx** —— 输出
//! 必须与**另一个缓存干净的 runtime 在同一形状上**的输出一致(≤5e-4)。
//!
//! **为什么这是不变量**:plan 是**性能**事实(把请求怎么切瓦/切 KV 铺满 SM),
//! 不是数值事实 ⇒ 缓存里**先前存过什么形状**,结构上不允许改变本次的输出。
//!
//! # 为什么既有门看不见它
//!
//! `fi_prefill_probe` 的每个形状都**新起 registry**(= 新 `FiRuntime` ⇒ 空缓存)
//! ⇒ plan 必然与形状自洽。真实引擎里缓存**长期在场**,而切块 prefill 的节奏是
//! "同一 T、ctx 逐块变大" ⇒ 一旦 plan 键漏掉 `ctx`(而 plan 的调度表/掩码其实
//! 依赖 ctx),**第 2 块起就在用首块的 plan** —— 数值静默错(A2 实测定谳:
//! `FI_SPLIT_KV` 两侧缺省相反,见 §8.30 的"机制"段)。
//!
//! # 分工
//!
//! - **本门(核级)**:`同形状 × 缓存状态` 的关系不变量 —— 便宜(无 host 参考);
//! - **`fi_prefill_probe`(核级)**:`核 vs host` 的绝对值门(小形状背);
//! - **`gpu_08b_chunk_invariance_gate`(引擎级)**:真 prompt × 两种切块的读数不变量。
//!
//! # 跑法(要卡;单线程)
//!
//! ```bash
//! OWL_TEST_DEVICE=1 cargo test -p owl-kernels --features flashinfer --test fi_plan_key_sweep \
//!   -- --test-threads=1 --nocapture
//! # 旋钮:FI_SWEEP_TS=192,256,384,512(缺省)/ FI_SWEEP_STEPS=1,2,4,8(缺省)
//! ```

#![cfg(feature = "device")]

mod fi_support;
use fi_support::{f16_to_f32, TestRes, PAGE};

use owl_kernels::contract::{Arg, Dtype, KernelSource, LaunchMsg};
use owl_kernels::device::Exec;

// ============================================================================
// 形状与数据(纯测试面;确定性生成,无外部依赖)
// ============================================================================
/// `(hq, hkv)` 两套几何:**引擎 0.8B 的错带几何** 与探针的 GQA24/4 几何。
const GEOMS: [(usize, usize); 2] = [(8, 2), (24, 4)];
const HD: usize = 256;

/// 确定性字节流(LCG;与探针同法,给 pool 灌**有限**位型)。
fn lcg_bytes(n: usize, seed: u32) -> Vec<u8> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        v.push((s >> 16) as u8);
    }
    v
}

/// e4m3 位型(**避开 NaN**:`exp=1111 & mant=111` 是 NaN ⇒ 落下来一位)。
fn e4m3_bytes(n: usize, seed: u32) -> Vec<u8> {
    lcg_bytes(n, seed)
        .into_iter()
        .map(|b| if b & 0x7F == 0x7F { b & 0x7E } else { b })
        .collect()
}

/// f16 位型(**避开 NaN/Inf**:指数全 1 就清掉最高位)。
fn f16_bits(n: usize, seed: u32) -> Vec<u16> {
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let mut x = (s >> 16) as u16;
        if x & 0x7C00 == 0x7C00 {
            x &= 0x7BFF;
        }
        v.push(x);
    }
    v
}

// ============================================================================
// 一次"会话" = 一个 registry(= 一个 `FiRuntime` = 一份 plan 缓存)
// ============================================================================
struct Session {
    res: TestRes,
    reg: owl_kernels::registry::OpRegistry,
    exec: Exec,
}

impl Session {
    fn new(dev: usize) -> Self {
        let mut res = TestRes::new(dev);
        let mut exec = Exec::new();
        // 激活集 = 本门要发射的那一个算子(白名单外的声明不审孤儿);
        // 写者算子在场只为池审计 E10(读该池实例必须有 ≥1 个写者)。
        let active: &[&str] = &[
            owl_kernels::family::flashinfer::PREFILL_FI_FP8KV,
            "owl_reshape_and_cache_fp8kv_knhd",
        ];
        let reg = owl_kernels::registry::OpRegistry::init_all_with(
            owl_kernels::server::family_factories(),
            &mut res,
            &mut exec,
            Some(active),
        )
        .expect("注册表装配");
        Self { res, reg, exec }
    }

    /// 在 `(t, ctx)` 形状上跑一发 FI(fp8 池),回读输出(host f32)。
    ///
    /// **同会话多次调用 = 共享 plan 缓存**(这正是本门要压的那条路径)。
    fn run(&mut self, hq: usize, hkv: usize, t: usize, ctx: usize) -> Vec<f32> {
        let res = &mut self.res;
        let nb = ctx.div_ceil(PAGE);
        let q_bits = f16_bits(t * hq * HD, 0x51ED + t as u32);
        let k = e4m3_bytes(nb * PAGE * hkv * HD, 0x9E37 + ctx as u32);
        let v = e4m3_bytes(nb * PAGE * hkv * HD, 0x1F3D + ctx as u32);

        let q = res.upload(&q_bits.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let k_b = res.upload(&k);
        let v_b = res.upload(&v);
        let q_cu = res.upload(&[0u32, t as u32].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let indices: Vec<u32> = (0..nb as u32).collect();
        let idx_b = res.upload(&indices.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let indptr =
            res.upload(&[0u32, nb as u32].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let last_len = ctx - (nb - 1) * PAGE;
        let ll_b = res.upload(&(last_len as u32).to_le_bytes());
        let wr_b = res.zeros(8);
        let out_b = res.zeros(t * hq * HD * 2);

        let msg = LaunchMsg {
            kernel: KernelSource {
                name: owl_kernels::client::flashinfer::PREFILL_FI_FP8KV.to_string(),
                source: String::new(),
            },
            args: vec![
                Arg::Block { id: q.id, dtype: Dtype::F16, rank: 2 },
                Arg::Block { id: k_b.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: v_b.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: q_cu.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: idx_b.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: indptr.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: ll_b.id, dtype: Dtype::U32, rank: 1 },
                Arg::Block { id: wr_b.id, dtype: Dtype::F16, rank: 1 },
                Arg::Block { id: out_b.id, dtype: Dtype::F16, rank: 2 },
                Arg::U64(t as u64),    // total_rows
                Arg::U64(ctx as u64),  // ctx_total(plan 键里被抹掉的那一维)
                Arg::U64(t as u64),    // T
                Arg::U64(hq as u64),
                Arg::U64(hkv as u64),
                Arg::U64(HD as u64),
                Arg::U64(PAGE as u64),
                Arg::U64(((1.0f32 / (HD as f32).sqrt()).to_bits()) as u64), // sm_scale 位型
            ],
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: t * hq * HD,
        };
        self.reg.execute(&msg, res, &mut self.exec).expect("FI run");
        res.stream.synchronize().expect("sync");
        res.take_u16(out_b.id).iter().map(|&h| f16_to_f32(h)).collect()
    }
}

/// 两路输出的最大绝对差(NaN 视为超门 —— 不许静默通过)。
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "输出长度必须相同");
    a.iter()
        .zip(b)
        .map(|(x, y)| if x.is_nan() || y.is_nan() { f32::INFINITY } else { (x - y).abs() })
        .fold(0.0f32, f32::max)
}

// ============================================================================
// 门
// ============================================================================
/// **plan 是性能事实,不许改数值**:同形状 + 缓存状态不同 ⇒ 输出必须一致。
///
/// 序列照**引擎的真实节奏**排:同一 `T`,ctx 从 `T` 起逐块长大(1×、2×、4×、8×)。
#[test]
fn fi_plan_key_ctx_sweep() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设(哪张卡)");
        return;
    }
    let dev: usize = owl_cuda::test_device_ordinal();
    // 错带(0.8B 几何实测;§8.30)与带外各留几个 —— 带外档必须也过(回归面)
    let ts: Vec<usize> = owl_shared::env_reader::str("FI_SWEEP_TS")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![192, 256, 384, 512, 1024]);
    let steps: Vec<usize> = owl_shared::env_reader::str("FI_SWEEP_STEPS")
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 2, 4, 8]);

    let mut bad: Vec<String> = Vec::new();
    for (hq, hkv) in GEOMS {
        for &t in &ts {
            // ① 缓存"脏"路:先跑 ctx=t(把首块 plan 钉进缓存),再逐档加大
            let mut dirty = Session::new(dev);
            let _ = dirty.run(hq, hkv, t, t);
            let stale: Vec<(usize, Vec<f32>)> =
                steps.iter().map(|s| (t * s, dirty.run(hq, hkv, t, t * s))).collect();
            // ② 缓存干净路:同形状,直接跑(plan 与形状自洽)
            let mut clean = Session::new(dev);
            let fresh: Vec<(usize, Vec<f32>)> =
                steps.iter().map(|s| (t * s, clean.run(hq, hkv, t, t * s))).collect();
            for ((ctx, a), (_, b)) in stale.iter().zip(fresh.iter()) {
                let d = max_abs_diff(a, b);
                let tag = if d <= 5e-4 { "ok  " } else { "**✗**" };
                println!(
                    "[plan-key] hq={hq:2} hkv={hkv} T={t:4} ctx={ctx:5} · max|脏−净| = {d:.6}  {tag}"
                );
                if d > 5e-4 {
                    bad.push(format!("hq={hq} hkv={hkv} T={t} ctx={ctx} Δ={d:.6}"));
                }
            }
        }
    }
    assert!(
        bad.is_empty(),
        "plan 缓存状态改变了数值(plan 必须只影响调度)⇒ plan 键漏维度(§8.30):\n  {}",
        bad.join("\n  ")
    );
}
