//! 示范用例:owl-nn gdn_kernels 三例迁移到 testkit 框架(与手写对拍并存)。
//!
//! 输入公式与 `crates/nn/src/kernels/gdn_kernels.rs` 手写对拍逐字一致
//! (交叉验证:框架用例与原用例应产出相同数值);host 参考公式出处:
//! - fused_gating:g = -exp(A_log)·softplus(a + dt_bias),beta = sigmoid(b)
//!   (Qwen3.5 GDN gating 定义;原实现 xinfer compute_gating 直译)
//! - l2_norm:out = x / sqrt(Σx² + eps)(数学定义;eps 内层)
//! - rmsnorm_act:y = (x·rmsnorm(x)·gamma + bias) · act(z),act=silu|sigmoid
//!   (deltanet 调用点定谳语义,per-group gamma/bias,恒 f32)

use std::collections::HashMap;

use owl_nn::kernels::gdn_kernels::GdnKernels;
use owl_shared::testkit::{Case, Gen, GenU32, Rig};

fn launch_kernel(rig: &Rig) -> GdnKernels {
    GdnKernels::new(rig.device().ctx()).expect("nvrtc gdn")
}

/// 用例一:fused_gating(g/beta 双输出;原测试 total=64, heads=8)
#[test]
fn ex_gdn_fused_gating_f32() {
    let rig = Rig::acquire();
    let case = Case::new("gdn_fused_gating_f32", 1)
        .in_f32("a_log", &[8], Gen::Formula(|i| -0.1 * i as f32))
        .in_f32("dt_bias", &[8], Gen::Formula(|i| 0.05 * i as f32))
        .in_f32("a", &[64], Gen::Formula(|i| (i % 17) as f32 * 0.3 - 8.0 * 0.3))
        .in_f32("b", &[64], Gen::Formula(|i| (i % 23) as f32 * 0.2 - 11.0 * 0.2))
        .out_f32("g", &[64])
        .out_f32("beta", &[64]);
    let (total, heads) = (64usize, 8usize);
    case.run(
        &rig,
        0.0,
        1e-5,
        |b| {
            let mut k = launch_kernel(b.rig);
            k.fused_gating(
                b.rig.stream(),
                "f32",
                b.in_f32("a_log").f32_ptr(),
                b.in_f32("a").u8_ptr(),
                b.in_f32("b").u8_ptr(),
                b.in_f32("dt_bias").f32_ptr(),
                b.out_f32_mut("g"),
                b.out_f32_mut("beta"),
                total as i32,
                heads as i32,
            )
            .map_err(|e| format!("fused_gating: {e}"))
        },
        |h| {
            let (a, b, a_log, dt_bias) = (
                h.host_f32("a"),
                h.host_f32("b"),
                h.host_f32("a_log"),
                h.host_f32("dt_bias"),
            );
            let mut g = Vec::with_capacity(total);
            let mut beta = Vec::with_capacity(total);
            for i in 0..total {
                let hd = i % heads;
                let x = a[i] + dt_bias[hd];
                let sp = if x <= 20.0 { x.exp().ln_1p() } else { x }; // softplus
                g.push(-a_log[hd].exp() * sp);
                beta.push(1.0 / (1.0 + (-b[i]).exp()));
            }
            HashMap::from([("g".into(), g), ("beta".into(), beta)])
        },
    )
    .expect("fused_gating 框架对拍全绿");
}

/// 用例二:l2_norm(dim 64 走 warp 变体 / 512 走 block 变体,双档覆盖)
#[test]
fn ex_gdn_l2_norm_f32() {
    let rig = Rig::acquire();
    for dim in [64usize, 512usize] {
        let rows = 3usize;
        let case = Case::new("gdn_l2_norm_f32", dim as u64)
            .in_f32(
                "input",
                &[rows, dim],
                Gen::Formula(|i| (i % 37) as f32 * 0.1 - 18.0 * 0.1),
            )
            .out_f32("out", &[rows, dim]);
        case.run(
            &rig,
            0.0,
            1e-4,
            |b| {
                let mut k = launch_kernel(b.rig);
                k.l2_norm(
                    b.rig.stream(),
                    "f32",
                    b.in_f32("input").u8_ptr(),
                    b.out_u8_mut("out"),
                    rows as i32,
                    dim as i32,
                    1e-6,
                )
                .map_err(|e| format!("l2_norm: {e}"))
            },
            |h| {
                let input = h.host_f32("input");
                let mut out = vec![0f32; rows * dim];
                for r in 0..rows {
                    let sumsq: f32 = input[r * dim..(r + 1) * dim].iter().map(|v| v * v).sum();
                    let inv = 1.0 / (sumsq.max(0.0) + 1e-6).sqrt();
                    for c in 0..dim {
                        out[r * dim + c] = input[r * dim + c] * inv;
                    }
                }
                HashMap::from([("out".into(), out)])
            },
        )
        .unwrap_or_else(|e| panic!("l2_norm dim={dim}: {e}"));
    }
}

/// 用例三:rmsnorm_act(per-group gamma/bias + silu/sigmoid 双激活;
/// 原 test rows=2, heads=4, group=32)
#[test]
fn ex_gdn_rmsnorm_act_f32() {
    let rig = Rig::acquire();
    let (rows, heads, group) = (2usize, 4usize, 32usize);
    let vdim = heads * group;
    for act in [0i32, 1i32] {
        let case = Case::new("gdn_rmsnorm_act_f32", 100 + act as u64)
            .in_f32(
                "x",
                &[rows, vdim],
                Gen::Formula(|i| (i % 29) as f32 * 0.2 - 14.0 * 0.2),
            )
            .in_f32(
                "z",
                &[rows, vdim],
                Gen::Formula(|i| (i % 13) as f32 * 0.4 - 6.0 * 0.4),
            )
            .in_f32("gamma", &[group], Gen::Formula(|i| 0.9 + 0.01 * i as f32))
            .in_f32("bias", &[group], Gen::Formula(|i| -0.02 * i as f32))
            .out_f32("out", &[rows, vdim]);
        case.run(
            &rig,
            0.0,
            1e-4,
            |b| {
                let mut k = launch_kernel(b.rig);
                k.rmsnorm_act(
                    b.rig.stream(),
                    "f32",
                    b.in_f32("x").u8_ptr(),
                    b.in_f32("z").u8_ptr(),
                    b.in_f32("gamma").f32_ptr(),
                    b.in_f32("bias").f32_ptr(),
                    b.out_u8_mut("out"),
                    rows as i32,
                    vdim as i32,
                    group as i32,
                    1e-6,
                    true, // per_group_weights(生产形态)
                    true, // has_bias
                    act,
                )
                .map_err(|e| format!("rmsnorm_act: {e}"))
            },
            |h| {
                let (x, z, gamma, bias) = (
                    h.host_f32("x"),
                    h.host_f32("z"),
                    h.host_f32("gamma"),
                    h.host_f32("bias"),
                );
                let mut out = vec![0f32; rows * vdim];
                for r in 0..rows {
                    for grp in 0..heads {
                        let off = r * vdim + grp * group;
                        let sumsq: f32 = x[off..off + group].iter().map(|v| v * v).sum();
                        let inv = (sumsq / group as f32 + 1e-6).sqrt().powi(-1);
                        for c in 0..group {
                            let y = x[off + c] * inv * gamma[c] + bias[c];
                            let zv = z[off + c];
                            let gate = if act == 1 {
                                1.0 / (1.0 + (-zv).exp()) // sigmoid
                            } else {
                                zv / (1.0 + (-zv).exp()) // silu
                            };
                            out[off + c] = y * gate;
                        }
                    }
                }
                HashMap::from([("out".into(), out)])
            },
        )
        .unwrap_or_else(|e| panic!("rmsnorm_act act={act}: {e}"));
    }
}

/// 附加一:GenU32 通道(framework 面:u32 输入声明/绑定/host 副本一致);
/// 被测算子面用一个「把 u32 输入求和写进 out[0]」的假核表达(不发真核)。
#[test]
fn ex_u32_channel_binding() {
    let rig = Rig::acquire();
    let case = Case::new("u32_channel", 11)
        .in_u32("slots", &[4], GenU32::Iota(100))
        .in_f32("x", &[8], Gen::Const(0.5))
        .out_f32("sum", &[1]);
    case.run(
        &rig,
        0.0,
        1e-6,
        |b| {
            assert_eq!(b.host_u32("slots"), &[100, 101, 102, 103]);
            let s = b.host_u32("slots").iter().sum::<u32>() as f32;
            // 假核的设备写路径:HtoD 进输出缓冲(与真核写 out 等价)
            use owl_cuda::ffi::sys;
            unsafe {
                sys::cuMemcpyHtoD_v2(
                    b.out_f32_mut("sum") as sys::CUdeviceptr,
                    &[s][0] as *const f32 as *const std::ffi::c_void,
                    4,
                )
                .result()
                .map_err(|e| format!("htod {e:?}"))?;
            }
            Ok(())
        },
        |_h| HashMap::from([("sum".into(), vec![406.0])]),
    )
    .expect("u32 通道 + 输出对比全绿");
}

/// 附加二:就地核(exp)的正用法——输入即输出缓冲,核外对拍。
/// (就地核与框架「输出预清零」语义天然冲突,(inplace) 形态走手写闭包)
#[test]
fn ex_gdn_exp_inplace_inplace() {
    let rig = Rig::acquire();
    let n = 64usize;
    let g: Vec<f32> = (0..n).map(|i| -0.01 * (i % 50) as f32).collect();
    let buf = rig.htod_f32(&g);
    let mut k = launch_kernel(&rig);
    k.exp_inplace_f32(rig.stream(), buf.f32_ptr_mut(), n as i32)
        .unwrap();
    let got = rig.dtoh_f32(&buf);
    let want: Vec<f32> = g.iter().map(|v| v.exp()).collect();
    owl_shared::testkit::allclose("gdn_exp_inplace_f32", &got, &want, 0.0, 1e-5, false)
        .expect("exp 就地对拍全绿");
}
