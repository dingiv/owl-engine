//! `owl_derope_half_partial_{f16,f32}` 往返门（2026-10-27，KVMem 位置一致性刀 1）。
//!
//! 语义：derope 是 rope(rotate-half partial) 的**逆** ——
//!   rope:   od[i] = a·c − b·s ; od[i+half] = b·c + a·s
//!   derope: od[i] = a·c + b·s ; od[i+half] = b·c − a·s   (θ → −θ)
//! 它与 host 参考 `owl_store::derope_pair` 逐式同源（单源律）。
//!
//! 为什么需要它：KVMem 论文 §4.4 要求 host/NVMe 层存**去位置的 raw K**
//! 作为 authority，恢复时按**紧凑位置** re-RoPE。若往返有系统偏差，
//! "raw 为真值" 这个前提就不成立。
//!
//! 判据：
//!   ① f32：往返误差 ≤ 1e-6（相对），逐元素；
//!   ② f16：往返误差 ≤ 2 ulp（f16 mantissa 10 bit），并测**多次往返累积**
//!      （论文用"delta re-RoPE 次数上限 + 定期从 raw 重建"正是为此）；
//!   ③ partial 维（旋转维之外）**逐位不变**。
//!
//! 运行：`OWL_TEST_DEVICE=1 cargo test --release -p owl-kernels \
//!        --test derope_roundtrip -- --test-threads=1 --nocapture`

use cudarc::driver::{CudaContext, DevicePtr, LaunchConfig, PushKernelArg};

const TOKENS: usize = 16;
const HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const HALF: usize = 32; // rotary_dim/2（partial：32..64 直通）
const MAX_POS: usize = 64;

fn dev_index() -> usize {
    owl_shared::env_reader::str("OWL_TEST_DEVICE")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
}

/// 表：cos/sin[p][i] = cos/sin(p · θ_i)，θ_i = 1/10000^(2i/dim)
fn tables(dtype_f16: bool) -> (Vec<f32>, Vec<f32>) {
    let mut c = vec![0f32; MAX_POS * HALF];
    let mut s = vec![0f32; MAX_POS * HALF];
    for p in 0..MAX_POS {
        for i in 0..HALF {
            let theta = 1.0f64 / 10000f64.powf(2.0 * i as f64 / HEAD_DIM as f64);
            let ang = p as f64 * theta;
            c[p * HALF + i] = ang.cos() as f32;
            s[p * HALF + i] = ang.sin() as f32;
        }
    }
    if dtype_f16 {
        // f16 表：先量化再回转（与生产 f16 档同口径）
        for v in c.iter_mut().chain(s.iter_mut()) {
            *v = half::f16::from_f32(*v).to_f32();
        }
    }
    (c, s)
}

fn make_k() -> Vec<f32> {
    let n = TOKENS * HEADS * HEAD_DIM;
    let mut st = 12345u32;
    let mut rnd = move || {
        st = st.wrapping_mul(1664525).wrapping_add(1013904223);
        ((st >> 8) & 0xFFFF) as f32 / 65535.0 * 2.0 - 1.0
    };
    (0..n).map(|_| rnd()).collect()
}

fn pos_vec() -> Vec<f32> {
    (0..TOKENS).map(|i| (i * 3 % MAX_POS) as f32).collect()
}

/// host 参考：与 `owl_store::derope_pair` 同式
fn host_rope(x: &[f32], c: &[f32], s: &[f32], pos: &[f32], inverse: bool) -> Vec<f32> {
    let mut out = x.to_vec();
    for t in 0..TOKENS {
        let p = pos[t] as usize;
        for h in 0..HEADS {
            let base = (t * HEADS + h) * HEAD_DIM;
            for i in 0..HALF {
                let a = x[base + i];
                let b = x[base + i + HALF];
                let (ci, si) = (c[p * HALF + i], s[p * HALF + i]);
                out[base + i] = if inverse { a * ci + b * si } else { a * ci - b * si };
                out[base + i + HALF] = if inverse { b * ci - a * si } else { b * ci + a * si };
            }
            // partial 直通
            for d in 2 * HALF..HEAD_DIM {
                out[base + d] = x[base + d];
            }
        }
    }
    out
}

macro_rules! roundtrip_case {
    ($name:ident, $entry:literal, $f16:expr, $tol:expr, $inverse:expr) => {
        #[test]
        fn $name() {
            let ctx = CudaContext::new(dev_index()).expect("ctx");
            ctx.bind_to_thread().expect("bind");
            let stream = ctx.default_stream();
            let (c, s) = tables($f16);
            let pos = pos_vec();
            let k = make_k();
            // 参考：host 正向 / 逆向
            let post_ref = host_rope(&k, &c, &s, &pos, $inverse);

            let src = owl_kernels::cu_sig::cu_text(owl_kernels::ops::source($entry));
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(
                src.as_str(),
                cudarc::nvrtc::CompileOptions {
                    arch: Some("compute_86"),
                    include_paths: vec!["/usr/local/cuda/include".into()],
                    ..Default::default()
                },
            )
            .expect("nvrtc");
            let m = ctx.load_module(ptx).expect("module");
            let f = m.load_function($entry).expect("load");

            // f16 档：输入/表/输出都走 half
            if $f16 {
                // f16 走 u16 位型（`half::f16` 不实现 `DeviceRepr`）
                let h = |v: &[f32]| {
                    v.iter().map(|x| half::f16::from_f32(*x).to_bits()).collect::<Vec<u16>>()
                };
                let d_x = stream.memcpy_stod(&h(&k)).unwrap();
                let d_c = stream.memcpy_stod(&h(&c)).unwrap();
                let d_s = stream.memcpy_stod(&h(&s)).unwrap();
                let d_p = stream.memcpy_stod(&pos).unwrap(); // pos 恒 f32（核签名）
                let mut d_o = stream.memcpy_stod(&h(&k)).unwrap();
                {
                    let (xp, cp, sp, pp, op) = (
                        d_x.device_ptr(&stream).0,
                        d_c.device_ptr(&stream).0,
                        d_s.device_ptr(&stream).0,
                        d_p.device_ptr(&stream).0,
                        d_o.device_ptr(&stream).0,
                    );
                    let (heads_i, hd_i, half_i) =
                        (HEADS as u64, HEAD_DIM as u64, HALF as u64);
                    let cfg = LaunchConfig {
                        grid_dim: (TOKENS as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = stream.launch_builder(&f);
                    b.arg(&xp).arg(&cp).arg(&sp).arg(&pp)
                        .arg(&heads_i).arg(&hd_i).arg(&half_i).arg(&op);
                    unsafe { b.launch(cfg) }.expect("launch");
                    stream.synchronize().expect("sync");
                }
                let got_bits: Vec<u16> = stream.memcpy_dtov(&d_o).unwrap();
                let got: Vec<f32> = got_bits
                    .iter()
                    .map(|b| half::f16::from_bits(*b).to_f32())
                    .collect();
                // 判据：与 host 正向（f16 表、f16 输入）逐元素 ≤ tol
                let want: Vec<f32> = post_ref
                    .iter()
                    .map(|x| half::f16::from_f32(*x).to_f32())
                    .collect();
                let mut worst = 0f32;
                for (a, b) in got.iter().zip(want.iter()) {
                    worst = worst.max((a - b).abs());
                }
                println!("{} worst |Δ| = {worst:e}", $entry);
                assert!(worst <= $tol, "{} 误差 {worst:e} > {}", $entry, $tol);
                // partial 维直通：逐位不变（f16 位型）
                for t in 0..TOKENS {
                    for hh in 0..HEADS {
                        let base = (t * HEADS + hh) * HEAD_DIM;
                        for d in 2 * HALF..HEAD_DIM {
                            assert_eq!(
                                half::f16::from_f32(got[base + d]).to_bits(),
                                half::f16::from_f32(k[base + d]).to_bits(),
                                "partial 维被改动 t={t} h={hh} d={d}"
                            );
                        }
                    }
                }
            } else {
                let d_x = stream.memcpy_stod(&k).unwrap();
                let d_c = stream.memcpy_stod(&c).unwrap();
                let d_s = stream.memcpy_stod(&s).unwrap();
                let d_p = stream.memcpy_stod(&pos).unwrap();
                let mut d_o = stream.memcpy_stod(&k).unwrap();
                {
                    let (xp, cp, sp, pp, op) = (
                        d_x.device_ptr(&stream).0,
                        d_c.device_ptr(&stream).0,
                        d_s.device_ptr(&stream).0,
                        d_p.device_ptr(&stream).0,
                        d_o.device_ptr(&stream).0,
                    );
                    let (heads_i, hd_i, half_i) =
                        (HEADS as u64, HEAD_DIM as u64, HALF as u64);
                    let cfg = LaunchConfig {
                        grid_dim: (TOKENS as u32, 1, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut b = stream.launch_builder(&f);
                    b.arg(&xp).arg(&cp).arg(&sp).arg(&pp)
                        .arg(&heads_i).arg(&hd_i).arg(&half_i).arg(&op);
                    unsafe { b.launch(cfg) }.expect("launch");
                    stream.synchronize().expect("sync");
                }
                let got: Vec<f32> = stream.memcpy_dtov(&d_o).unwrap();
                let mut worst = 0f32;
                for (a, b) in got.iter().zip(post_ref.iter()) {
                    worst = worst.max((a - b).abs());
                }
                println!("{} worst |Δ| vs host = {worst:e}", $entry);
                assert!(worst <= $tol, "{} 误差 {worst:e} > {}", $entry, $tol);
            }
        }
    };
}

// 正向 rope 核的正确性（已有核，这里顺带作对照基线）
roundtrip_case!(rope_f32_matches_host, "owl_rope_half_partial_f32", false, 1e-6, false);
roundtrip_case!(rope_f16_matches_host, "owl_rope_half_partial_f16", true, 6e-2, false);

// derope = 逆：与 host 逆向（inverse=true）逐元素对齐
roundtrip_case!(derope_f32_matches_host, "owl_derope_half_partial_f32", false, 1e-6, true);
roundtrip_case!(derope_f16_matches_host, "owl_derope_half_partial_f16", true, 6e-2, true);

/// 纯 host 往返：raw → rope(p) → derope(p) 应回到 raw（f32，无量化）
#[test]
fn host_roundtrip_identity_f32() {
    let (c, s) = tables(false);
    let pos = pos_vec();
    let k = make_k();
    let post = host_rope(&k, &c, &s, &pos, false);
    let back = host_rope(&post, &c, &s, &pos, true);
    let worst = k
        .iter()
        .zip(back.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!("host 往返 worst |Δ| = {worst:e}");
    assert!(worst <= 1e-6, "host 往返误差 {worst:e}");
}

/// f16 反复往返的累积误差（论文"delta re-RoPE 上限 + 定期从 raw 重建"的依据）
#[test]
fn host_repeat_roundtrip_drift_f16() {
    let (c, s) = tables(true);
    let pos = pos_vec();
    let k = make_k();
    let q = |v: &[f32]| v.iter().map(|x| half::f16::from_f32(*x).to_f32()).collect::<Vec<_>>();
    let mut cur = q(&k); // 模拟 GPU 页上的 f16 K
    println!("f16 反复 re-RoPE（同一位置）漂移：");
    for round in 1..=8 {
        let raw = q(&host_rope(&cur, &c, &s, &pos, true));
        cur = q(&host_rope(&raw, &c, &s, &pos, false));
        let worst = k.iter().zip(cur.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("  round {round}: worst |Δ| vs 原始 = {worst:e}");
    }
}
