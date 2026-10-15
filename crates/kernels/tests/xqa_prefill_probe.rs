//! XQA **prefill 形状**(chunked prefill, q=T 行因果块)速度/数值探针。
//!
//! 来源与契约:`crates/kernels/cu/xqa/README.md`(Ampere 闭包;SPEC_DEC 位掩码
//! 语义定谳 = 行 i 看块内 bit 0..i ⇒ 因果)。
//!
//! 形状 = owl prefill 实况(chunked prefill 一个 chunk):q = T 行
//! (全局位置 `[ctx−T, ctx)`),GQA 24q/4kv,hd256,page 32,kNHD fp8 e4m3 池;
//! `seq_lens = ctx`(整请求 KV 长),`mask[i] = (1<<(i+1))-1` per 32-bit word。
//!
//! 三条判据:
//!   1) 数值:XQA vs host 参考(行 i 可见 `[0, ctx−T+i]` 闭区间)误差 < 1e-2;
//!   2) 交叉:XQA vs **现役 prefill 注意力核**(`vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256`
//!      = `cu/attn/prefill_paged_attn_f16.cu`,xinfer 血统 chunked prefill 主核);
//!   3) 速度:同 (T, ctx) 头对头 per-call ms。
//!
//! 运行:`CUDA_VISIBLE_DEVICES=<空闲卡 UUID> OWL_TEST_DEVICE=0 cargo test --release \
//!        -p owl-kernels --features xqa --test xqa_prefill_probe -- --test-threads=1 --nocapture`
//! 可选:`XQA_PF_ONLY=4096` 只跑该 ctx;`XQA_PF_HOST=1` 大形状也跑 host 参考(慢)。
//!
//! ⚠️ **跑法**:本探针是基准 —— host 参考/建池是纯 CPU 大循环,debug 慢
//! 10–30×(2026-10-15 gdb 定谳:卡点在 `lcg_u8` 的逐元素 e4m3 编码与 host 参考)。
//! 建议:`cargo test --release -p owl-kernels --features device,<族> --test <本文件>
//!       -- --test-threads=1 --nocapture`;debug 下只用 `*_PF_ONLY=<ctx>` 跳 host 检查。
#![cfg(feature = "xqa")]

use std::time::Instant;

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const PAGE: usize = 32;
const HKV: usize = 4;
const H: usize = 24;
const HD: usize = 256;
const SM86_SM_COUNT: u32 = 84; // 3090 Ti

fn half_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f16_to_f32(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}
// ---------------------------------------------------------------------------
// e4m3 编解码:表化 + 二分(2026-10-15 **测试用例优化**)
//
// 病根(实测,见 gdb 栈):原 `e4m3_encode` 是**每元素 256 次线性扫**、`e4m3_decode`
// 每元素调 `powi`;而本探针建池要 8.4M(16384 档 16.8M)个元素、host 参考里 decode
// 又要被调用十亿次 ⇒ **debug 构建下单核跑分钟级**(表现为"卡住",GPU util 0%)。
// 修法(语义不变):
//   ① `e4m3_decode` → 256 项**一次性表**(数值与原式逐位相同);
//   ② `e4m3_encode` → 按值排序表 + 二分 + 小窗取最小 (err, code)
//      —— 与原 256 线性扫**逐位等价**(含"同误差取最小 code"的 tie 规则),
//      由本文件单测 `e4m3_encode_matches_bruteforce` 机器锁死;
//   ③ host 参考**预解码**一次(池 → f32),内层不再逐元素 decode。
// ---------------------------------------------------------------------------
fn e4m3_table() -> &'static [f32; 256] {
    static T: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut t = [0f32; 256];
        for (code, slot) in t.iter_mut().enumerate() {
            *slot = e4m3_decode_exact(code as u8);
        }
        t
    })
}
/// 原始解码式(仅建表时用一次/项;NaN 槽保留原样)
fn e4m3_decode_exact(code: u8) -> f32 {
    let s = if code & 0x80 != 0 { -1f32 } else { 1f32 };
    let e = ((code >> 3) & 0x0F) as i32;
    let m = (code & 0x07) as i32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let v = if e == 0 {
        (m as f32) * 2f32.powi(-9)
    } else {
        (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7)
    };
    s * v
}
fn e4m3_decode(code: u8) -> f32 {
    e4m3_table()[code as usize]
}
/// 编码表:按值升序,同值按 code 升序(⇒ tie 自动取最小 code)
fn e4m3_sorted() -> &'static [(f32, u8)] {
    static T: std::sync::OnceLock<Vec<(f32, u8)>> = std::sync::OnceLock::new();
    T.get_or_init(|| {
        let mut v: Vec<(f32, u8)> = (0..=255u8)
            .filter(|c| c & 0x7F != 0x7F) // NaN 两种位型不参与编码(原式同)
            .map(|c| (e4m3_decode_exact(c), c))
            .collect();
        v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
        v
    })
}
/// 最近码编码(与原 256 线性扫逐位等价;见文件头注 ②)
fn e4m3_encode(v: f32) -> u8 {
    let tbl = e4m3_sorted();
    let pos = tbl.partition_point(|&(x, _)| x < v);
    let lo = pos.saturating_sub(2);
    let hi = (pos + 2).min(tbl.len());
    let mut best_err = f32::INFINITY;
    let mut best = 0u8;
    for &(x, code) in &tbl[lo..hi] {
        let err = (x - v).abs();
        if err < best_err {
            best_err = err;
            best = code;
        }
    }
    best
}
/// 暴力参照(仅单测用;原实现原样保留)
#[cfg(test)]
fn e4m3_encode_bruteforce(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0..=255u8 {
        if code & 0x7F == 0x7F {
            continue;
        }
        let err = (e4m3_decode_exact(code) - v).abs();
        if err < best_err {
            best_err = err;
            best = code;
        }
    }
    best
}

#[cfg(test)]
mod encode_opt_tests {
    use super::*;
    /// 二分编码 ≡ 原 256 线性扫(含 tie/±0/饱和)
    #[test]
    fn e4m3_encode_matches_bruteforce() {
        let mut cases: Vec<f32> = Vec::new();
        for code in 0..=255u8 {
            let v = e4m3_decode_exact(code);
            if v.is_finite() {
                cases.push(v);
                cases.push(v + 1e-3);
                cases.push(v - 1e-3);
            }
        }
        cases.extend([0.0f32, -0.0, 1e-9, -1e-9, 448.0, -448.0, 1e9, -1e9, 1.0, -1.0]);
        for i in 0..20_000u32 {
            let x = i.wrapping_mul(0x9E37_79B9).wrapping_add(0x85EB_CA6B);
            cases.push(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 2.0);
        }
        for v in cases {
            if v.is_nan() {
                continue;
            }
            assert_eq!(
                e4m3_encode(v),
                e4m3_encode_bruteforce(v),
                "v={v} 二分编码与原线性扫不等价"
            );
        }
    }
}

fn lcg_u8(n: usize, seed: u32) -> Vec<u8> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            let f = ((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 2.0;
            e4m3_encode(f)
        })
        .collect()
}
fn lcg_f16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            half_bits(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0)
        })
        .collect()
}

/// host 参考:kNHD + fp8 池,chunked prefill 行 i 看 `[0, ctx−T+i]`(闭区间)。
/// 成本 = O(T·ctx·H·HD);只在 `t·ctx·H·HD ≤ 4e8` 或 `XQA_PF_HOST=1` 时调用。
fn host_ref_knhd(q: &[u16], kc: &[u8], vc: &[u8], t: usize, ctx: usize) -> Vec<f32> {
    // **预解码一次**(2026-10-15 测试用例优化):原实现在最内层逐元素
    // `e4m3_decode`,T=512/ctx=512 单 case 就是 ~1.6e9 次调用(debug 分钟级);
    // 预解码后内层只剩 f32 FMA —— 语义不变(同一批值,同一归约序)。
    let kf: Vec<f32> = kc.iter().map(|&c| e4m3_decode(c)).collect();
    let vf: Vec<f32> = vc.iter().map(|&c| e4m3_decode(c)).collect();
    let qf: Vec<f32> = q.iter().map(|&b| f16_to_f32(b)).collect();
    let mut out = vec![0f32; t * H * HD];
    let kbs = HKV * HD * PAGE;
    let scale = 1.0 / (HD as f32).sqrt();
    for s in 0..t {
        let len = ctx - t + s + 1;
        for h in 0..H {
            let kvh = h / (H / HKV);
            let mut qk = vec![0f32; len];
            #[allow(clippy::needless_range_loop)]
            for b in 0..len {
                let (pb, bb) = (b / PAGE, b % PAGE);
                let mut acc = 0f32;
                for d in 0..HD {
                    let kidx = pb * kbs + bb * (HKV * HD) + kvh * HD + d;
                    acc += qf[(s * H + h) * HD + d] * kf[kidx];
                }
                qk[b] = acc * scale;
            }
            let m = qk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l: f32 = qk.iter().map(|&x| (x - m).exp()).sum();
            for d in 0..HD {
                let mut acc = 0f32;
                #[allow(clippy::needless_range_loop)]
                for b in 0..len {
                    let (pb, bb) = (b / PAGE, b % PAGE);
                    let vidx = pb * kbs + bb * (HKV * HD) + kvh * HD + d;
                    acc += (qk[b] - m).exp() * vf[vidx];
                }
                out[(s * H + h) * HD + d] = acc / l;
            }
        }
    }
    out
}

/// XQA SPEC_DEC 位掩码:`[t, ceil(t/32)] u32`,行 i 的位 0..i 置 1。
fn causal_mask_words(t: usize) -> Vec<u32> {
    let wpr = t.div_ceil(32);
    let mut v = vec![0u32; t * wpr];
    for i in 0..t {
        for w in 0..wpr {
            let lo = w * 32;
            let hi = (i + 1).min(lo + 32);
            v[i * wpr + w] = if hi <= lo { 0 } else if hi - lo == 32 { u32::MAX } else { (1u32 << (hi - lo)) - 1 };
        }
    }
    v
}

/// 一次 (T, ctx) 全量测量:XQA 单发 vs chunked prefill 核同池同形状。
fn run_case(t: usize, ctx: usize, host_check: bool, chunked: bool) {
    let nb = ctx.div_ceil(PAGE);
    let ctx_c = CudaContext::new(owl_shared::env_reader::parse_or("OWL_TEST_DEVICE", 0)).expect("ctx");
    ctx_c.bind_to_thread().expect("bind");
    let stream = ctx_c.default_stream();

    // ---- 数据(kNHD fp8 池 + f16 q) ----
    let pool_elems = nb * HKV * HD * PAGE;
    let k_fp8 = lcg_u8(pool_elems, 701);
    let v_fp8 = lcg_u8(pool_elems, 702);
    let q_f16 = lcg_f16(t * H * HD, 703);

    let dev_u8 = |v: &[u8]| {
        let mut d = unsafe { stream.alloc::<u8>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let dev_u16 = |v: &[u16]| {
        let mut d = unsafe { stream.alloc::<u16>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let dev_f32 = |v: &[f32]| {
        let mut d = unsafe { stream.alloc::<f32>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let d_k = dev_u8(&k_fp8);
    let d_v = dev_u8(&v_fp8);
    let d_q = dev_u16(&q_f16);
    let d_out_xqa = dev_u16(&vec![0u16; t * H * HD]);
    let d_out_ch = dev_u16(&vec![0u16; t * H * HD]);

    // ---- XQA 参数 ----
    let bt_i32: Vec<i32> = (0..nb as i32).collect();
    let d_bt_i32 = {
        let mut d = unsafe { stream.alloc::<i32>(bt_i32.len()) }.unwrap();
        stream.memcpy_htod(&bt_i32, &mut d).unwrap();
        d
    };
    let seq_u32: Vec<u32> = vec![ctx as u32];
    let d_seq_u32 = {
        let mut d = unsafe { stream.alloc::<u32>(seq_u32.len()) }.unwrap();
        stream.memcpy_htod(&seq_u32, &mut d).unwrap();
        d
    };
    let mask = causal_mask_words(t);
    let d_mask = {
        let mut d = unsafe { stream.alloc::<u32>(mask.len()) }.unwrap();
        stream.memcpy_htod(&mask, &mut d).unwrap();
        d
    };
    let (xqa_kernel, xqa_consts) =
        owl_kernels::family::xqa::load_for_probe(&ctx_c, owl_kernels::family::xqa::ASSET_VERIFY)
            .expect("载 xqa_verify 资产");
    // ⚠️ semaphores 数组按 **gridDim.y × batch**(KV-split 归约计数,idxSeq =
    // gridDim.y·idxReq + blockIdx.y;参考实现只按 hkv 给,SPEC_DEC 下不够!
    // t=512 时 grid.y = hkv·ceil(t·6/32) = 384 ⇒ 按 12 给 = 越界写)
    let xqa_grid_y =
        xqa_consts.launch_geometry(SM86_SM_COUNT, 1, HKV as u32, (nb * PAGE) as u32, t as u32).0 .1;
    let sem_len = xqa_grid_y as usize + 8;
    let d_sem = {
        let mut d = unsafe { stream.alloc::<u32>(sem_len) }.unwrap();
        stream.memcpy_htod(&vec![0u32; sem_len], &mut d).unwrap();
        d
    };
    let d_scratch = unsafe { stream.alloc::<u8>(256 << 20) }.unwrap();

    let (q_p, k_p, v_p) = (
        d_q.device_ptr(&stream).0,
        d_k.device_ptr(&stream).0,
        d_v.device_ptr(&stream).0,
    );
    let (o_p, bt_p, sq_p, mk_p, sem_p, sc_p) = (
        d_out_xqa.device_ptr(&stream).0,
        d_bt_i32.device_ptr(&stream).0,
        d_seq_u32.device_ptr(&stream).0,
        d_mask.device_ptr(&stream).0,
        d_sem.device_ptr(&stream).0,
        d_scratch.device_ptr(&stream).0,
    );
    let mk_args = |out: u64| owl_kernels::family::xqa::XqaArgs {
        q_seq_len: t as u32,
        nb_k_heads: HKV as u32,
        q_cu_seq_lens: 0,
        q_scale: 1.0, // = attn_scale(1/sqrt(256)) * sqrt(256)
        q_scale_ptr: 0,
        output: out,
        q: q_p,
        mask: mk_p,
        attention_sinks: 0,
        cache: owl_kernels::family::xqa::KVCacheList {
            k_cache: k_p,
            v_cache: v_p,
            page_list: bt_p,
            seq_len_list: sq_p,
            max_nb_pages_per_seq: xqa_consts.max_nb_pages_per_seq((nb * PAGE) as u32),
            _pad: 0,
        },
        batch_size: 1,
        kv_scale: 1.0,
        kv_scale_ptr: 0,
        stride_page: xqa_consts.stride_in_heads((PAGE * HKV * HD) as u64),
        stride_token: xqa_consts.stride_in_heads((HKV * HD) as u64),
        stride_head: xqa_consts.stride_in_heads(HD as u64),
        semaphores: sem_p,
        scratch: sc_p,
    };
    // KV-split 关闭实验(XQA_PF_NOSPLIT=1):几何上的 max_seq_len 传 ctaTile
    // ⇒ nbSubSeq=1(sscratch/semaphores 全省);页表 stride 仍按 nb(batch=1 无影响)
    let no_split = owl_shared::env_reader::flag("XQA_PF_NOSPLIT");
    let xqa_call = |out: u64| {
        let mut a = mk_args(out);
        if no_split {
            a.cache.max_nb_pages_per_seq = xqa_consts.cta_tile_x / xqa_consts.tokens_per_page;
        }
        owl_kernels::family::xqa::launch_on_stream(
            &stream,
            &xqa_kernel,
            &xqa_consts,
            &a,
            SM86_SM_COUNT,
        )
        .expect("xqa launch");
    };
    xqa_call(o_p);
    stream.synchronize().expect("xqa sync");

    // ---- 现役 prefill 注意力核(xinfer chunked prefill 主核,kNHD fp8) ----
    let ch_func = if chunked {
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(
            owl_kernels::cu_sig::cu_text(
                owl_kernels::ops::source("vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256"),
            )
            .as_str(),
            cudarc::nvrtc::CompileOptions {
                arch: Some("compute_86"),
                include_paths: vec!["/usr/local/cuda/include".into()],
                ..Default::default()
            },
        )
        .expect("nvrtc compile chunked prefill");
        let m = ctx_c.load_module(ptx).expect("module");
        Some(
            m.load_function("vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256")
                .expect("load chunked knhd fp8"),
        )
    } else {
        None
    };
    // block_tables [num_seqs=1, block_table_stride] f32(恒等链)
    let bt_f32: Vec<f32> = (0..nb).map(|b| b as f32).collect();
    let d_bt_f32 = dev_f32(&bt_f32);
    let d_seq_f32 = dev_f32(&[ctx as f32]);
    let d_qsl_f32 = dev_f32(&[0.0, t as f32]);
    let d_null_f32 = dev_f32(&[0.0f32; 1]);
    let o_ch_p = d_out_ch.device_ptr(&stream).0;
    let (btf_p, sqf_p, qsl_p, null_p) = (
        d_bt_f32.device_ptr(&stream).0,
        d_seq_f32.device_ptr(&stream).0,
        d_qsl_f32.device_ptr(&stream).0,
        d_null_f32.device_ptr(&stream).0,
    );
    let ch_call = |f: &cudarc::driver::CudaFunction| {
        let hkv_i = HKV as i32;
        let scale = 1f32 / (HD as f32).sqrt();
        let bts_i = 1i32;
        let nseq_i = 1i32;
        let hq_i = H as i32;
        let qn_i = t as i32;
        let soft = 1f32;
        let ostr_i = (H * HD) as i32;
        let sw_i = -1i32;
        let nb_i = nb as i32;
        let kvbs_i = (HKV * HD * PAGE) as i32;
        let kvhs_i = HD as i32;
        let zero_i = 0i32;
        let mut b = stream.launch_builder(f);
        b.arg(&q_p)
            .arg(&k_p)
            .arg(&v_p)
            .arg(&btf_p)
            .arg(&sqf_p)
            .arg(&qsl_p)
            .arg(&null_p) // alibi
            .arg(&null_p) // sinks
            .arg(&hkv_i)
            .arg(&scale)
            .arg(&bts_i)
            .arg(&nseq_i)
            .arg(&hq_i)
            .arg(&qn_i)
            .arg(&soft)
            .arg(&ostr_i)
            .arg(&sw_i)
            .arg(&nb_i)
            .arg(&kvbs_i)
            .arg(&kvhs_i)
            .arg(&zero_i) // use_alibi
            .arg(&zero_i) // use_sinks
            .arg(&o_ch_p);
        let cfg = LaunchConfig {
            grid_dim: ((H / HKV) as u32, HKV as u32, t.div_ceil(256) as u32),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (64 + 2 * HD * PAGE * 2) as u32,
        };
        unsafe { b.launch(cfg) }.expect("chunked prefill");
    };
    if let Some(f) = ch_func.as_ref() {
        ch_call(f);
        stream.synchronize().expect("chunked sync");
    }

    // ---- 计时 ----
    let bench = |n: usize, mut f: Box<dyn FnMut()>| -> f64 {
        for _ in 0..3 {
            f();
        }
        stream.synchronize().unwrap();
        let t0 = Instant::now();
        for _ in 0..n {
            f();
        }
        stream.synchronize().unwrap();
        t0.elapsed().as_secs_f64() * 1e3 / n as f64
    };
    let xqa_ms = bench(20, Box::new(|| xqa_call(o_p)));
    let ch_ms = ch_func.as_ref().map(|f| bench(20, Box::new(|| ch_call(f))));

    // ---- 数值 ----
    let mut o_xqa = vec![0u16; t * H * HD];
    stream.memcpy_dtoh(&d_out_xqa, &mut o_xqa).unwrap();
    let fx: Vec<f32> = o_xqa.iter().map(|&h| f16_to_f32(h)).collect();
    let mut o_ch: Option<Vec<f32>> = None;
    if chunked {
        let mut och = vec![0u16; t * H * HD];
        stream.memcpy_dtoh(&d_out_ch, &mut och).unwrap();
        o_ch = Some(och.iter().map(|&h| f16_to_f32(h)).collect());
    }
    let host = host_check.then(|| host_ref_knhd(&q_f16, &k_fp8, &v_fp8, t, ctx));
    let dmax = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let nz = fx.iter().filter(|&&x| x == 0.0).count();
    // 逐 token 行诊断:哪一行开始分叉(掩码字/行映射嫌疑定位)
    let row_diag = |a: &[f32], b: &[f32]| -> String {
        let mut bad = Vec::new();
        for s in 0..t {
            let d = dmax(&a[s * H * HD..(s + 1) * H * HD], &b[s * H * HD..(s + 1) * H * HD]);
            if d > 1e-2 {
                bad.push(s);
            }
        }
        if bad.is_empty() {
            "all rows ok".to_string()
        } else {
            let head: Vec<String> = bad.iter().take(8).map(|r| r.to_string()).collect();
            format!("{} 行超门: {:?}{}", bad.len(), head, if bad.len() > 8 { " …" } else { "" })
        }
    };

    let (grid, block, smem) =
        xqa_consts.launch_geometry(SM86_SM_COUNT, 1, HKV as u32, (nb * PAGE) as u32, t as u32);
    println!(
        "=== T={t} ctx={ctx} (pages={nb}) ===\n  \
         consts: ctaTile=({},{}) headGrp={} rowsPerBlock={} multiBlock={} smem={}\n  \
         XQA grid={grid:?} block={block:?} smem={smem}\n  \
         XQA    : {xqa_ms:7.3} ms/call\n  \
         chunked: {}\n  \
         max|XQA−chunked| = {:?}\n  \
         max|XQA−host| = {:?}",
        xqa_consts.cta_tile_x,
        xqa_consts.cta_tile_y,
        xqa_consts.head_grp,
        xqa_consts.rows_per_block,
        xqa_consts.allow_multi_block,
        xqa_consts.smem,
        ch_ms.map(|m| format!("{m:7.3} ms/call")).unwrap_or_else(|| "(skip)".into()),
        o_ch.as_ref().map(|c| dmax(&fx, c)),
        host.as_ref().map(|h| dmax(&fx, h)),
    );
    let _ = nz;
    if let (Some(h), Some(c)) = (host.as_ref(), o_ch.as_ref()) {
        println!("  max|chunked−host| = {:.6}", dmax(c, h));
    }
    if let Some(c) = o_ch.as_ref() {
        println!("  XQA vs chunked 逐行: {}", row_diag(&fx, c));
    }
    if let Some(h) = host.as_ref() {
        println!("  XQA vs host    逐行: {}", row_diag(&fx, h));
    }
    let mut failed = false;
    if let Some(h) = host.as_ref() {
        let dh = dmax(&fx, h);
        println!("  max|XQA−host| = {dh:.6}  (XQA 零值 {nz}/{})", fx.len());
        if dh >= 1e-2 {
            println!("  ❌ T={t} ctx={ctx}: XQA vs host 超门 {dh:.6}");
            failed = true;
        }
    }
    if let Some(c) = o_ch.as_ref() {
        let dc = dmax(&fx, c);
        if dc >= 2e-2 {
            println!("  ❌ T={t} ctx={ctx}: XQA vs chunked 超门 {dc:.6}");
            failed = true;
        }
    }
    assert!(!failed, "T={t} ctx={ctx} 数值门未过(详见上文)");
}

#[test]
fn xqa_prefill_speed_and_parity() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let only = owl_shared::env_reader::str("XQA_PF_ONLY").unwrap_or_default();
    let host_all = owl_shared::env_reader::flag("XQA_PF_HOST");
    let only_t = owl_shared::env_reader::parse_or("XQA_PF_T", 0usize);
    // (T, ctx, host_check)
    let cases: &[(usize, usize, bool)] = &[
        (128, 512, true),
        (128, 8192, false),
        (512, 512, true),
        (512, 2048, false),
        (512, 4096, false),
        (512, 8192, false),
        (512, 16384, false),
    ];
    let mut failed: Vec<String> = Vec::new();
    for &(t, ctx, hc) in cases {
        if (!only.is_empty() && only != ctx.to_string()) || (only_t != 0 && only_t != t) {
            continue;
        }
        let host_check = hc || (host_all && t.saturating_mul(ctx) * H * HD <= 8_000_000_000);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_case(t, ctx, host_check, true)));
        if r.is_err() {
            failed.push(format!("T={t} ctx={ctx}"));
        }
    }
    assert!(failed.is_empty(), "数值门未过: {failed:?}");
}

// ============================================================================
// 小形状诊断:定位"1/3 非零"的真凶(掩码字 / 行映射 / 头组映射)
// 运行:XQA_PF_DIAG=1 ... --test xqa_prefill_diag --nocapture
// ============================================================================
#[test]
fn xqa_prefill_diag() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let (t, ctx) = (
        owl_shared::env_reader::parse_or("XQA_PF_DIAG_T", 64usize),
        owl_shared::env_reader::parse_or("XQA_PF_DIAG_CTX", 64usize),
    );
    let nb = ctx.div_ceil(PAGE);
    let ctx_c = CudaContext::new(owl_shared::env_reader::parse_or("OWL_TEST_DEVICE", 0)).expect("ctx");
    ctx_c.bind_to_thread().expect("bind");
    let stream = ctx_c.default_stream();
    let pool_elems = nb * HKV * HD * PAGE;
    let k_fp8 = lcg_u8(pool_elems, 701);
    let v_fp8 = lcg_u8(pool_elems, 702);
    let q_f16 = lcg_f16(t * H * HD, 703);
    let dev_u8 = |v: &[u8]| {
        let mut d = unsafe { stream.alloc::<u8>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let dev_u16 = |v: &[u16]| {
        let mut d = unsafe { stream.alloc::<u16>(v.len()) }.unwrap();
        stream.memcpy_htod(v, &mut d).unwrap();
        d
    };
    let d_k = dev_u8(&k_fp8);
    let d_v = dev_u8(&v_fp8);
    let d_q = dev_u16(&q_f16);
    let d_out = dev_u16(&vec![0u16; t * H * HD]);
    let bt_i32: Vec<i32> = (0..nb as i32).collect();
    let d_bt = {
        let mut d = unsafe { stream.alloc::<i32>(nb) }.unwrap();
        stream.memcpy_htod(&bt_i32, &mut d).unwrap();
        d
    };
    let d_seq = {
        let mut d = unsafe { stream.alloc::<u32>(1) }.unwrap();
        stream.memcpy_htod(&[ctx as u32], &mut d).unwrap();
        d
    };
    let mask = causal_mask_words(t);
    let wpr = t.div_ceil(32);
    let d_mask = {
        let mut d = unsafe { stream.alloc::<u32>(mask.len()) }.unwrap();
        stream.memcpy_htod(&mask, &mut d).unwrap();
        d
    };
    let (xqa_kernel, xqa_consts) =
        owl_kernels::family::xqa::load_for_probe(&ctx_c, owl_kernels::family::xqa::ASSET_VERIFY)
            .expect("载资产");
    let diag_grid_y =
        xqa_consts.launch_geometry(SM86_SM_COUNT, 1, HKV as u32, (nb * PAGE) as u32, t as u32).0 .1;
    let sem_len = diag_grid_y as usize + 8;
    let d_sem = {
        let mut d = unsafe { stream.alloc::<u32>(sem_len) }.unwrap();
        stream.memcpy_htod(&vec![0u32; sem_len], &mut d).unwrap();
        d
    };
    let d_scratch = unsafe { stream.alloc::<u8>(256 << 20) }.unwrap();
    let (q_p, k_p, v_p) = (
        d_q.device_ptr(&stream).0,
        d_k.device_ptr(&stream).0,
        d_v.device_ptr(&stream).0,
    );
    let (o_p, bt_p, sq_p, mk_p, sem_p, sc_p) = (
        d_out.device_ptr(&stream).0,
        d_bt.device_ptr(&stream).0,
        d_seq.device_ptr(&stream).0,
        d_mask.device_ptr(&stream).0,
        d_sem.device_ptr(&stream).0,
        d_scratch.device_ptr(&stream).0,
    );
    let args = owl_kernels::family::xqa::XqaArgs {
        q_seq_len: t as u32,
        nb_k_heads: HKV as u32,
        q_cu_seq_lens: 0,
        q_scale: 1.0,
        q_scale_ptr: 0,
        output: o_p,
        q: q_p,
        mask: mk_p,
        attention_sinks: 0,
        cache: owl_kernels::family::xqa::KVCacheList {
            k_cache: k_p,
            v_cache: v_p,
            page_list: bt_p,
            seq_len_list: sq_p,
            max_nb_pages_per_seq: xqa_consts.max_nb_pages_per_seq((nb * PAGE) as u32),
            _pad: 0,
        },
        batch_size: 1,
        kv_scale: 1.0,
        kv_scale_ptr: 0,
        stride_page: xqa_consts.stride_in_heads((PAGE * HKV * HD) as u64),
        stride_token: xqa_consts.stride_in_heads((HKV * HD) as u64),
        stride_head: xqa_consts.stride_in_heads(HD as u64),
        semaphores: sem_p,
        scratch: sc_p,
    };
    let (grid, block, smem) = xqa_consts.launch_geometry(
        SM86_SM_COUNT, 1, HKV as u32, (nb * PAGE) as u32, t as u32,
    );
    println!("T={t} ctx={ctx} grid={grid:?} block={block:?} smem={smem} consts={xqa_consts:?}");
    owl_kernels::family::xqa::launch_on_stream(&stream, &xqa_kernel, &xqa_consts, &args, SM86_SM_COUNT)
        .expect("launch");
    stream.synchronize().unwrap();
    let mut o = vec![0u16; t * H * HD];
    stream.memcpy_dtoh(&d_out, &mut o).unwrap();
    let fx: Vec<f32> = o.iter().map(|&h| f16_to_f32(h)).collect();
    let host = host_ref_knhd(&q_f16, &k_fp8, &v_fp8, t, ctx);
    println!("mask words/row={wpr}");
    for i in 0..t.min(4) {
        println!("  mask[{i}] = {:?}", &mask[i * wpr..(i + 1) * wpr]);
    }
    let mut nz_rows = 0;
    for s in 0..t {
        let row = &fx[s * H * HD..(s + 1) * H * HD];
        let nz = row.iter().filter(|&&x| x != 0.0).count();
        if nz > 0 {
            nz_rows += 1;
        }
        let d = row.iter().zip(&host[s * H * HD..(s + 1) * H * HD]).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        if s < 6 || (nz > 0 && nz < H * HD) {
            // 每个 token 内按 head 统计非零个数
            let per_head: Vec<usize> = (0..H)
                .map(|h| row[h * HD..(h + 1) * HD].iter().filter(|&&x| x != 0.0).count())
                .collect();
            println!("  row {s}: nonzero={nz}/{} maxdiff={d:.4} per-head-nz[0..8]={:?}", H * HD, &per_head[..8]);
        }
    }
    println!("非零行 {nz_rows}/{t}");
}
