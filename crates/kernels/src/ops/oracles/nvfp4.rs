//! `nvfp4_gemm` 装配期自检(**NVFP4 第一道常驻数值门**;2026-10-28)。
//!
//! 经注册表 `dispatch`(与生产同一条路径)跑一发真 GEMM,对 **host f32 参考**判相对偏差。
//! 参考 = 用**同一套量化规则**在 host 上重建(`g = amax/(6·448)`、每 16 个 k 一个
//! e4m3 局部缩放、`e2m1` 取最近码点),再在 f32 里做内积 —— 与 CUTLASS 输出比
//! **相对偏差**(判据 = 1e-2:FP4 量化的相对误差在 4×4 切片上实测 2.2e-4,
//! 留一个数量级余量,能抓住"布局/步长/alpha 错"却不被量化噪声卡住)。
//!
//! ## 为什么不止一档几何(M5,2026-10-28)
//!
//! 原来只跑 `m=n=k=128`(tile 整块)—— 那恰好是**不触发任何 pad 谓词**的最省事形状,
//! 而本轮两次踩到的口径错都长在边上:
//!
//! | 档 | 覆盖的事实 |
//! |---|---|
//! | `m=1, n=256, k=256` | **decode 形状**;A 侧 SF 按 128 行 pad(SFA 账长 = 128·K/16)而数据只有 1 行 |
//! | `m=5, n=128, k=256` | 不满一个 tile 的**余数行**(spec/fold 的奇数 m) |
//! | `m=128, n=128, k=128` | 控制组(tile 整块,零 pad 路径 —— 旧档) |
//! | `m=1, n=9216, k=256` | 真实 decode 的 N(gate_proj 量级) |
//! | `m=1, n=1000, k=256` | **N 非 128 对齐**(只 8 对齐):B 侧 SF 账长 `> N·K/16` ⇒ 装载期必须拒;这一档同时钉住"库能跑、装载面不许"的边界 |
//!
//! 每档都自证 SF 账长口径(`sfa = ceil(m/128)·128·k/16`、`sfb = N·K/16`(128 对齐时)),
//! 口径不符 = 结构化拒 —— 这比数值门更早、更确定。

use crate::bootcheck::{BootCheck, CheckReport, CheckRunner};
use crate::contract::{Arg, KernelSource, LaunchMsg, OpError};

/// 自检几何表:`(m, n, k, 说明)`。
///
/// `k` 取小值(256/128)是为了把 host 侧参考的内积成本压住(自检跑在 boot 路径上);
/// 覆盖点是 **m 的 pad 行为** 与 **N 的对齐面**,不是大 K。
const CASES: &[(usize, usize, usize, &str)] = &[
    (1, 256, 256, "decode(m=1)"),
    (5, 128, 256, "余数 m=5"),
    (128, 128, 128, "tile 整块(控制组)"),
    (1, 9216, 256, "decode 真实 N=9216"),
    (1, 1000, 256, "N 非 128 对齐(N=1000)"),
];

const TOL: f32 = 1e-2;

const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// f16 位型 → f32(自足;不引外来依赖)
fn f16_to_f32(bits: u16) -> f32 {
    let s = (bits >> 15) as u32;
    let e = ((bits >> 10) & 0x1f) as i32;
    let m = (bits & 0x3ff) as f32;
    let v = if e == 0 {
        m * 2f32.powi(-24)
    } else if e == 0x1f {
        f32::INFINITY
    } else {
        (1.0 + m / 1024.0) * 2f32.powi(e - 15)
    };
    if s != 0 { -v } else { v }
}

/// e4m3(不带符号扩展的 8 位浮点)反解 —— 只覆盖量化器会产出的码点
fn e4m3_dec(c: u8) -> f32 {
    let sign = if c & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((c >> 3) & 0xf) as i32;
    let mant = (c & 7) as f32;
    let mag = if e == 0 {
        (mant / 8.0) * 2f32.powi(-6)
    } else {
        (1.0 + mant / 8.0) * 2f32.powi(e - 7)
    };
    sign * mag
}

/// 最接近的 e2m1 码点(与硬件 RNE 在非 tie 上一致;tie 归偶数索引 —— 参考侧
/// 只用它估期望值,允许 1 码点差由 TOL 吸收)
fn e2m1_enc(x: f32) -> u8 {
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for (i, v) in E2M1.iter().enumerate() {
        let d = (x.abs() - v).abs();
        if d < bd {
            bd = d;
            best = i;
        }
    }
    (if x < 0.0 { 8 } else { 0 }) | best as u8
}

/// 按量化器同一规则把 host f16 `[rows, K]` 量化成 (每元素值, 全局缩放)
fn quant_rows(vals: &[f32], rows: usize, k: usize) -> (Vec<f32>, f32) {
    let mut amax = 0f32;
    for &v in vals {
        amax = amax.max(v.abs());
    }
    let g = if amax > 0.0 { amax / (6.0 * 448.0) } else { 1.0 };
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        for kg in 0..k / 16 {
            let base = r * k + kg * 16;
            let mx = (0..16).fold(0f32, |a, i| a.max(vals[base + i].abs()));
            let want = if mx > 0.0 { mx / (6.0 * g) } else { 1.0 };
            let mut sc = 1u8;
            let mut bd = f32::INFINITY;
            for code in 1..127u8 {
                let d = (e4m3_dec(code) - want).abs();
                if d < bd {
                    bd = d;
                    sc = code;
                }
            }
            let sd = e4m3_dec(sc) * g;
            for i in 0..16 {
                let q = e2m1_enc(if sd > 0.0 { vals[base + i] / sd } else { 0.0 });
                let mag = E2M1[(q & 7) as usize];
                out[base + i] = if q & 8 != 0 { -mag } else { mag } * sd;
            }
        }
    }
    (out, g)
}

pub struct NvFp4GemmOracle;

impl BootCheck for NvFp4GemmOracle {
    fn name(&self) -> &'static str {
        "host f32 参考(同量化规则反量化后内积;经注册表 dispatch;5 档几何)"
    }

    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError> {
        let mut worst = 0f32;
        let mut first_bad: Option<(usize, f32, f32)> = None;
        let mut worst_case = "";
        let mut detail: Vec<String> = Vec::new();

        for (ci, &(m, n, k, what)) in CASES.iter().enumerate() {
            // ---- host 输入(每档独立种子;任意 f16 位型,量化前不要求可解释) ----
            let mut seed = 0xC0FFEEu32 ^ (ci as u32).wrapping_mul(0x9E37_79B9);
            let mut rnd = || {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) & 0xffff) as u16
            };
            let a: Vec<u16> = (0..m * k).map(|_| rnd()).collect();
            let b: Vec<u16> = (0..n * k).map(|_| rnd()).collect();
            let a_f32: Vec<f32> = a.iter().map(|&x| f16_to_f32(x)).collect();
            let b_f32: Vec<f32> = b.iter().map(|&x| f16_to_f32(x)).collect();
            let (a_ref, _) = quant_rows(&a_f32, m, k);
            let (b_ref, _) = quant_rows(&b_f32, n, k);

            let plan = crate::nvfp4::Nvfp4Plan::new(m, n, k).ok_or_else(|| OpError::Asset {
                op: crate::family::nvfp4::GEMM.into(),
                detail: format!("Nvfp4Plan::new 失败 m={m} n={n} k={k}({what})"),
            })?;
            // ---- **SF 账长口径自证**(比数值门更早、更确定) ----
            let want_sfa = m.div_ceil(128).max(1) * 128 * k / 16;
            if plan.sfa_bytes() != want_sfa {
                return Err(OpError::Contract {
                    op: crate::family::nvfp4::GEMM.into(),
                    field: "sfa_bytes",
                    expect: format!("ceil(m/128)·128·k/16 = {want_sfa}(m={m} k={k},{what})"),
                    got: format!("{}", plan.sfa_bytes()),
                });
            }
            let flat_sfb = n * k / 16;
            if n % 128 == 0 {
                if plan.sfb_bytes() != flat_sfb {
                    return Err(OpError::Contract {
                        op: crate::family::nvfp4::GEMM.into(),
                        field: "sfb_bytes",
                        expect: format!("N·K/16 = {flat_sfb}(n={n} k={k},{what})"),
                        got: format!("{}", plan.sfb_bytes()),
                    });
                }
            } else if plan.sfb_bytes() <= flat_sfb {
                // N 只 8 对齐时 SFB 必然**大于** N·K/16(128 行窗口 pad)——
                // 这正是装载面 `N%128` 门的依据;哪天变了,这里先叫
                return Err(OpError::Contract {
                    op: crate::family::nvfp4::GEMM.into(),
                    field: "sfb_bytes",
                    expect: format!("N 非 128 对齐 ⇒ sfb_bytes > N·K/16 = {flat_sfb}({what})"),
                    got: format!("{}", plan.sfb_bytes()),
                });
            }

            let mut aq = vec![0u8; plan.a_q_bytes()];
            let mut asf = vec![0u8; plan.sfa_bytes()];
            let mut bq = vec![0u8; plan.b_q_bytes()];
            let mut bsf = vec![0u8; plan.sfb_bytes()];
            let ab: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
            let bb: Vec<u8> = b.iter().flat_map(|x| x.to_le_bytes()).collect();
            // SAFETY: 长度按 plan 的 `*_bytes()` 精确分配
            let (ga, gb) = unsafe {
                (
                    plan.pack_a(&ab, &mut aq, &mut asf),
                    plan.pack_b(&bb, &mut bq, &mut bsf),
                )
            };

            // ---- 账本块 + 上传 ----
            let ba = r.alloc_block(ab.len(), "selfcheck.nvfp4.a")?;
            let bbk = r.alloc_block(bb.len(), "selfcheck.nvfp4.b")?;
            let bq_b = r.alloc_block(bq.len(), "selfcheck.nvfp4.bq")?;
            let bsf_b = r.alloc_block(bsf.len(), "selfcheck.nvfp4.bsf")?;
            let aq_b = r.alloc_block(aq.len(), "selfcheck.nvfp4.aq")?;
            let asf_b = r.alloc_block(asf.len(), "selfcheck.nvfp4.asf")?;
            // `dg` = **f32[2]**:`[0]` = g(量化核读),`[1]` = alpha = g·gb(GEMM 的 alpha_ptr)
            let dg_b = r.alloc_block(8, "selfcheck.nvfp4.dg")?;
            let out_b = r.alloc_block(m * n * 2, "selfcheck.nvfp4.out")?;
            r.upload_block(&ba, &ab)?;
            r.upload_block(&bbk, &bb)?;
            r.upload_block(&bq_b, &bq)?;
            r.upload_block(&bsf_b, &bsf)?;

            let msg = LaunchMsg {
                kernel: KernelSource {
                    name: crate::family::nvfp4::GEMM.to_string(),
                    source: String::new(),
                },
                args: vec![
                    Arg::block(ba.id, ba.dtype, ba.rank),
                    Arg::block(bq_b.id, bq_b.dtype, bq_b.rank),
                    Arg::block(bsf_b.id, bsf_b.dtype, bsf_b.rank),
                    Arg::block(out_b.id, out_b.dtype, out_b.rank),
                    Arg::block(aq_b.id, aq_b.dtype, aq_b.rank),
                    Arg::block(asf_b.id, asf_b.dtype, asf_b.rank),
                    Arg::block(dg_b.id, dg_b.dtype, dg_b.rank),
                    Arg::U64(m as u64),
                    Arg::U64(k as u64),
                    Arg::U64(n as u64),
                    Arg::F32(gb),
                ],
                grid: (0, 0, 0),
                block: (0, 0, 0),
                shared_mem: 0,
                out_elems: m * n,
            };
            r.dispatch(msg)?;
            r.sync()?;
            let mut raw = vec![0u8; m * n * 2];
            r.download_block(&out_b, &mut raw)?;

            // ---- 参考:量化后数据在 f32 里做内积 ----
            let mut case_worst = 0f32;
            for i in 0..m {
                for j in 0..n {
                    let mut want = 0f32;
                    for kk in 0..k {
                        want += a_ref[i * k + kk] * b_ref[j * k + kk];
                    }
                    want *= ga * gb;
                    let got = f16_to_f32(u16::from_le_bytes([
                        raw[(i * n + j) * 2],
                        raw[(i * n + j) * 2 + 1],
                    ]));
                    let err = (got - want).abs();
                    if err > case_worst {
                        case_worst = err;
                    }
                    let rel = err / want.abs().max(1e-3);
                    if rel > TOL && first_bad.is_none() {
                        first_bad = Some((ci * 1_000_000 + i * n + j, got, want));
                    }
                }
            }
            if case_worst > worst {
                worst = case_worst;
                worst_case = what;
            }
            detail.push(format!(
                "{m}x{n}x{k} {what}: max|err|={case_worst:.3e} sfa={} sfb={}",
                plan.sfa_bytes(),
                plan.sfb_bytes()
            ));
        }

        if let Some((idx, got, want)) = first_bad {
            return Err(OpError::Contract {
                op: crate::family::nvfp4::GEMM.into(),
                field: "out",
                expect: format!("相对偏差 ≤ {TOL:.1e}(host 同规则反量化参考)"),
                got: format!(
                    "首越界 idx={idx} gpu={got} host={want}(max|err|={worst:.3e} @ {worst_case})\n{}",
                    detail.join("\n")
                ),
            });
        }
        let mut rep = CheckReport::numeric(crate::family::nvfp4::GEMM, worst, TOL);
        rep.first_bad = None;
        // 逐档账长/误差上屏(诊断面:哪一档的 pad 口径变了,一行看出来)
        crate::klog_info!("nvfp4-oracle", "{}", detail.join(" ｜ "));
        Ok(rep)
    }
}
