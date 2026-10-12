//! XQA(Ampere spec-verify 核)速度/数值探针 —— 与现役 v2 伪序列臂同机同 ctx 头对头。
//!
//! 来源与契约:`crates/kernels/cu/xqa/README.md`(移植自
//! repos/vllm-ampere-optimized/flashampere,Ampere 闭包)。
//!
//! 形状 = owl verify 实况:q_seq_len=8 行(行 i 看 [0, CTX−8+i)),GQA 24q/4kv,
//! hd256,page 32,kNHD 布局 [page, token, hkv, hd] 的 **fp8 e4m3** 池。
//!
//! 三条判据:
//!   1) 数值:XQA vs host 参考(每行独立全局 softmax,与
//!      `attention_v2_multiseq_probe` 同一 oracle 语义)误差 < 1e-2;
//!   2) 速度:XQA 单发 vs v2(主核 + reduce 两发)per-call ms,同 ctx;
//!   3) 交叉:XQA vs v2 输出 max diff(链式传递:v2 ≡ host 已由另一探针证明)。
//!
//! 运行:`OWL_TEST_DEVICE=1 cargo test --release -p owl-kernels --features xqa \
//!        --test xqa_verify_probe -- --test-threads=1 --nocapture`
#![cfg(feature = "xqa")]

use std::time::Instant;

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const PAGE: usize = 32;
const HKV: usize = 4;
const H: usize = 24;
const HD: usize = 256;
const S: usize = 8; // q_seq_len(verify = 1 锚 + 7 草稿)
const SM86_SM_COUNT: u32 = 84; // 3090 Ti

fn half_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}
fn f16_to_f32(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}
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

/// host 参考:kNHD + fp8 池,行 s 看 [0, lens[s])。
fn host_ref_knhd(q: &[u16], kc: &[u8], vc: &[u8], lens: &[usize]) -> Vec<f32> {
    let mut out = vec![0f32; S * H * HD];
    let kbs = HKV * HD * PAGE;
    let scale = 1.0 / (HD as f32).sqrt();
    for s in 0..S {
        let len = lens[s];
        for h in 0..H {
            let kvh = h / (H / HKV);
            let mut qk = vec![0f32; len];
            #[allow(clippy::needless_range_loop)]
            for b in 0..len {
                let (pb, bb) = (b / PAGE, b % PAGE);
                let mut acc = 0f32;
                for d in 0..HD {
                    let kidx = pb * kbs + bb * (HKV * HD) + kvh * HD + d;
                    acc += f16_to_f32(q[(s * H + h) * HD + d]) * e4m3_decode(kc[kidx]);
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
                    acc += (qk[b] - m).exp() * e4m3_decode(vc[vidx]);
                }
                out[(s * H + h) * HD + d] = acc / l;
            }
        }
    }
    out
}


/// 一次全量测量:XQA 单发 + v2(主核+reduce)两发,同 ctx、同池。
fn run_case(ctx: usize, check: bool) { run_case_mm(ctx, check, false) }

/// 掩码变体 A/B:v=0 因果(bit k 允许 k≤row) / 1 位序反 / 2 极性反 / 3 行序反
fn mask_variant(v: u32) -> Vec<u32> {
    (0..S)
        .map(|i| {
            let i2 = match v {
                3 => S - 1 - i,
                _ => i,
            };
            let m = (1u32 << (i2 + 1)) - 1;
            match v {
                1 => {
                    // 位序反:bit k 表示块内第 (S-1-k) 个 key
                    let mut r = 0u32;
                    for k in 0..S {
                        if m & (1 << k) != 0 {
                            r |= 1 << (S - 1 - k);
                        }
                    }
                    r
                }
                2 => !m,
                _ => m,
            }
        })
        .collect()
}

/// `mask_all=true` → 掩码全开(块内 8 行互相全可见)+ host 同口径(lens 全 = ctx)
/// 用途:区分"掩码语义/位打包复刻错"(全开后误差应塌)与"数值精度"(不塌)。
fn run_case_mm(ctx: usize, check: bool, mask_all: bool) { run_case_var(ctx, check, mask_all, 0) }

fn run_case_var(ctx: usize, check: bool, mask_all: bool, mask_var: u32) {
    let nb = ctx.div_ceil(PAGE);
    let nparts = (nb * PAGE).div_ceil(512);
    let ctx_c = CudaContext::new(1).expect("ctx");
    ctx_c.bind_to_thread().expect("bind");
    let stream = ctx_c.default_stream();

    // ---- 数据(kNHD fp8 池 + f16 q) ----
    let pool_elems = nb * HKV * HD * PAGE;
    let k_fp8 = lcg_u8(pool_elems, 701);
    let v_fp8 = lcg_u8(pool_elems, 702);
    let q_f16 = lcg_f16(S * H * HD, 703);
    // 行 i 的可见长 = ctx − S + i(因果块 = 最后 S 个位置)
    // host 参考口径:mask_all → 全可见;否则 **因果含自身位置**(行 i 看 [0, ctx-S+i] 闭区间
    // ⇒ key 数 = ctx-S+i+1)。首版按"不含自身"算 → 每行少 1 个 key = 假误差来源。
    let lens: Vec<usize> = if mask_all {
        vec![ctx; S]
    } else {
        (0..S).map(|i| ctx - S + i + 1).collect()
    };

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
    let d_out_xqa = dev_u16(&vec![0u16; S * H * HD]);
    let d_out_v2 = dev_u16(&vec![0u16; S * H * HD]);

    // XQA 参数:i32 页表(恒等链)、u32 seq_lens、spec-dec 位掩码、semaphore、scratch
    let bt_i32: Vec<i32> = (0..nb as i32).collect();
    let d_bt_i32 = {
        let mut d = unsafe { stream.alloc::<i32>(bt_i32.len()) }.unwrap();
        stream.memcpy_htod(&bt_i32, &mut d).unwrap();
        d
    };
    // ⚠️ XQA 的 seqLen 是**整请求总 KV 长**(batch 维,不是逐行!):
    // 块内 8 行的因果结构由 mask 位掩码表达,行 i 的可见上界 = seq_len-(S-1-i)。
    // 首版误传逐行 lens → 行 1..7 各少 1..7 个 key → 0.0146 假误差(掩码全开对照
    // 一开即塌到 3.1e-4,定谳 = 参数语义错,非精度)。
    let seq_u32: Vec<u32> = vec![ctx as u32];
    let d_seq_u32 = {
        let mut d = unsafe { stream.alloc::<u32>(seq_u32.len()) }.unwrap();
        stream.memcpy_htod(&seq_u32, &mut d).unwrap();
        d
    };
    // 下三角位掩码:行 i 的 bit j = 可看块内第 j 个 key ⇒ 0x1,0x3,…,0xFF
    let mask: Vec<u32> = if mask_all { vec![u32::MAX; S] } else { mask_variant(mask_var) };
    let d_mask = {
        let mut d = unsafe { stream.alloc::<u32>(mask.len()) }.unwrap();
        stream.memcpy_htod(&mask, &mut d).unwrap();
        d
    };
    let sem_len = HKV.div_ceil(2) * 2 + 2 + HKV + 2; // round_up(nb_seq,2)+2+nb_seq+2
    let d_sem = {
        let mut d = unsafe { stream.alloc::<u32>(sem_len) }.unwrap();
        stream.memcpy_htod(&vec![0u32; sem_len], &mut d).unwrap();
        d
    };
    let d_scratch = unsafe { stream.alloc::<u8>(256 << 20) }.unwrap();

    // v2 基线参数(f32 页表/可见长,与生产同式)
    let bt_f32: Vec<f32> = (0..S).flat_map(|_| (0..nb).map(|b| b as f32)).collect();
    let d_bt_f32 = {
        let mut d = unsafe { stream.alloc::<f32>(bt_f32.len()) }.unwrap();
        stream.memcpy_htod(&bt_f32, &mut d).unwrap();
        d
    };
    let d_lens_f32 = {
        let mut d = unsafe { stream.alloc::<f32>(lens.len()) }.unwrap();
        stream.memcpy_htod(&lens.iter().map(|&l| l as f32).collect::<Vec<_>>(), &mut d).unwrap();
        d
    };
    let d_es = {
        let mut d = unsafe { stream.alloc::<f32>(S * H * nparts) }.unwrap();
        stream.memcpy_htod(&vec![0f32; S * H * nparts], &mut d).unwrap();
        d
    };
    let d_ml = {
        let mut d = unsafe { stream.alloc::<f32>(S * H * nparts) }.unwrap();
        stream.memcpy_htod(&vec![0f32; S * H * nparts], &mut d).unwrap();
        d
    };
    let d_partial = dev_u16(&vec![0u16; S * H * nparts * HD]);

    // ---- XQA 单发 ----
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
    // ---- XQA 单发(2026-10-13 C3a:构建期资产 + Rust 发射器) ----
    let (xqa_kernel, xqa_consts) =
        owl_kernels::family::xqa::load_for_probe(&ctx_c, owl_kernels::family::xqa::ASSET_VERIFY)
            .expect("载 xqa_verify 资产");
    let mk_args = |out: u64| owl_kernels::family::xqa::XqaArgs {
        q_seq_len: S as u32,
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
    let xqa_call = |out: u64| {
        owl_kernels::family::xqa::launch_on_stream(
            &stream,
            &xqa_kernel,
            &xqa_consts,
            &mk_args(out),
            SM86_SM_COUNT,
        )
        .expect("xqa launch");
    };
    xqa_call(o_p);
    stream.synchronize().expect("xqa sync");

    // ---- v2 基线(主核 + reduce;与生产同两发) ----
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(
        owl_kernels::sources::attention::PAGED_ATTENTION_F16,
        cudarc::nvrtc::CompileOptions {
            arch: Some("compute_86"),
            include_paths: vec!["/usr/local/cuda/include".into()],
            ..Default::default()
        },
    )
    .expect("nvrtc compile paged v2");
    let m = ctx_c.load_module(ptx).expect("module");
    let f_main = m
        .load_function("vllm_paged_attention_v2_fp8_knhd_hd256bs32")
        .expect("load v2 fp8 knhd hd256bs32");
    let f_red = m
        .load_function("vllm_paged_attention_v2_reduce_f16_hd256")
        .expect("load v2 reduce hd256");
    let (es_p, ml_p, pa_p, ln_p) = (
        d_es.device_ptr(&stream).0,
        d_ml.device_ptr(&stream).0,
        d_partial.device_ptr(&stream).0,
        d_lens_f32.device_ptr(&stream).0,
    );
    let o2_p = d_out_v2.device_ptr(&stream).0;
    let scale = 1f32 / (HD as f32).sqrt();
    let bt_v2 = d_bt_f32.device_ptr(&stream).0;
    // 标量实参需具名绑定(launch_builder 借用临时量 → E0716)
    let hkv_i = HKV as i32;
    let nb_i = nb as i32;
    let qstride_i = (H * HD) as i32;
    let kvbs_i = (HKV * HD * PAGE) as i32;
    let kvhs_i = HD as i32;
    let one_f = 1f32;
    let neg1_i = -1i32;
    let zero_i = 0i32;
    let v2_call = || {
        let mut b = stream.launch_builder(&f_main);
        b.arg(&q_p)
            .arg(&k_p)
            .arg(&v_p)
            .arg(&bt_v2)
            .arg(&ln_p)
            .arg(&es_p)
            .arg(&es_p)
            .arg(&ml_p)
            .arg(&hkv_i)
            .arg(&scale)
            .arg(&nb_i)
            .arg(&qstride_i)
            .arg(&kvbs_i)
            .arg(&kvhs_i)
            .arg(&one_f)
            .arg(&neg1_i)
            .arg(&zero_i)
            .arg(&pa_p);
        let cfg = LaunchConfig {
            grid_dim: (H as u32, S as u32, nparts as u32),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 2048,
        };
        unsafe { b.launch(cfg) }.expect("v2 main");
        let mut b2 = stream.launch_builder(&f_red);
        let np_i = nparts as i32;
        b2.arg(&es_p).arg(&ml_p).arg(&pa_p).arg(&ln_p).arg(&np_i).arg(&o2_p);
        let cfg2 = LaunchConfig {
            grid_dim: (H as u32, S as u32, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 2048,
        };
        unsafe { b2.launch(cfg2) }.expect("v2 reduce");
    };
    v2_call();
    stream.synchronize().expect("v2 sync");

    // ---- 计时 ----
    let bench = |n: usize, mut f: Box<dyn FnMut()>| -> f64 {
        for _ in 0..3 {
            f();
        }
        stream.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        stream.synchronize().unwrap();
        t.elapsed().as_secs_f64() * 1e3 / n as f64
    };
    let xqa_ms = bench(20, Box::new(|| { xqa_call(o_p); }));
    let v2_ms = bench(20, Box::new(|| { v2_call(); }));

    // ---- 数值 ----
    let mut o_xqa = vec![0u16; S * H * HD];
    stream.memcpy_dtoh(&d_out_xqa, &mut o_xqa).unwrap();
    let mut o_v2 = vec![0u16; S * H * HD];
    stream.memcpy_dtoh(&d_out_v2, &mut o_v2).unwrap();
    let fx: Vec<f32> = o_xqa.iter().map(|&h| f16_to_f32(h)).collect();
    let fv: Vec<f32> = o_v2.iter().map(|&h| f16_to_f32(h)).collect();
    let (mut d_xv, mut n_xv) = (0f32, 0usize);
    for i in 0..fx.len() {
        let e = (fx[i] - fv[i]).abs();
        if e > d_xv {
            d_xv = e;
        }
        if fx[i] == 0.0 {
            n_xv += 1;
        }
    }
    let host = if check { Some(host_ref_knhd(&q_f16, &k_fp8, &v_fp8, &lens)) } else { None };
    let d_xh = host.as_ref().map(|h| {
        fx.iter().zip(h.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
    });
    let d_vh = host.as_ref().map(|h| {
        fv.iter().zip(h.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max)
    });

    println!(
        "=== ctx={ctx} (pages={nb}, nparts={nparts}, q_seq_len={S}) ===\n  \
         XQA 单发      : {xqa_ms:7.3} ms/call\n  \
         v2 主核+reduce: {v2_ms:7.3} ms/call\n  \
         加速比        : {:5.2}×\n  \
         数值 max|XQA−v2| = {d_xv:.6}   (XQA 零值 {n_xv}/{})\n  \
         数值 max|XQA−host| = {:?}   max|v2−host| = {:?}",
        v2_ms / xqa_ms,
        fx.len(),
        d_xh,
        d_vh
    );
    if let Some(dh) = d_xh {
        println!("   [v={mask_var} mask_all={mask_all}] max|XQA−host| = {dh:.6}");
        assert!(dh < 1e-2, "XQA vs host 超门:{dh:.6}");
    }
}

#[test]
fn xqa_verify_speed_and_parity() {
    if !owl_shared::env_reader::flag("OWL_TEST_DEVICE") {
        eprintln!("skip: OWL_TEST_DEVICE 未设");
        return;
    }
    // 判据分离:掩码全开(块内全可见)+ host 同口径
    run_case_mm(2048, true, true);
    // 掩码语义 A/B(定谳留档:仅 v=0 因果口径贴 host;v=1 位序反 / v=2 极性反 /
    // v=3 行序反 全部远离 —— 由 XQA_MASK_AB=1 复跑)
    if owl_shared::env_reader::flag("XQA_MASK_AB") {
        for v in 0..4u32 {
            println!("--- 掩码变体 v={v} ---");
            run_case_var(2048, true, false, v);
        }
    }
    let only = owl_shared::env_reader::str("XQA_ONLY").unwrap_or_default();
    if only != "8192" {
        run_case(2048, true);
    }
    if only != "2048" {
        run_case(8192, false); // host 参考在 8192 太慢;该点只测速 + 与 v2 交叉
    }
}
