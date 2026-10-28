//! `nvfp4_gemm` 装配期自检(**NVFP4 第一道常驻数值门**;2026-10-28)。
//!
//! 经注册表 `dispatch`(与生产同一条路径)跑一发真 GEMM,对 **host f32 参考**判相对偏差。
//! 参考 = 用**同一套量化规则**在 host 上重建(`g = amax/(6·448)`、每 16 个 k 一个
//! e4m3 局部缩放、`e2m1` 取最近码点),再在 f32 里做内积 —— 与 CUTLASS 输出比
//! **相对偏差**(判据 = 1e-2:FP4 量化的相对误差在 4×4 切片上实测 2.2e-4,
//! 留一个数量级余量,能抓住"布局/步长/alpha 错"却不被量化噪声卡住)。
//!
//! 几何取**最小可用**:`m=n=k=128/128/128`(tile 128 整块,不触发 pad 谓词的
//! 边缘路径;SF 账长 = `128*128/16`)。

use crate::contract::{Arg, KernelSource, LaunchMsg, OpError};
use crate::bootcheck::{BootCheck, CheckReport, CheckRunner};

/// 自检几何(全 128 对齐 ⇒ SF 账长 = M*K/16 无 pad)
const MM: usize = 128;
const NN: usize = 128;
const KK: usize = 128;
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
    for &v in vals { amax = amax.max(v.abs()); }
    let g = if amax > 0.0 { amax / (6.0 * 448.0) } else { 1.0 };
    let mut out = vec![0f32; rows * k];
    for r in 0..rows {
        for kg in 0..k / 16 {
            let base = r * k + kg * 16;
            let mx = (0..16).fold(0f32, |a, i| a.max(vals[base + i].abs()));
            // e4m3 量化局部缩放(取最近码点)
            let want = if mx > 0.0 { mx / (6.0 * g) } else { 1.0 };
            let mut sc = 1u8;
            let mut bd = f32::INFINITY;
            for code in 1..127u8 {
                let d = (e4m3_dec(code) - want).abs();
                if d < bd { bd = d; sc = code; }
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
        "host f32 参考(同量化规则反量化后内积;经注册表 dispatch)"
    }

    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError> {
        // ---- host 输入 + 打包(与装载期同一条 FFI:`pack_a` / `pack_b`) ----
        let mut seed = 0xC0FFEEu32;
        let mut rnd = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 8) & 0xffff) as u16 // 任意 f16 位型(量化前不要求可解释)
        };
        let a: Vec<u16> = (0..MM * KK).map(|_| rnd()).collect();
        let b: Vec<u16> = (0..NN * KK).map(|_| rnd()).collect();
        let a_f32: Vec<f32> = a.iter().map(|&x| f16_to_f32(x)).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| f16_to_f32(x)).collect();
        let (a_ref, _ga_ref) = quant_rows(&a_f32, MM, KK);
        let (b_ref, _gb_ref) = quant_rows(&b_f32, NN, KK);

        let plan = crate::nvfp4::Nvfp4Plan::new(MM, NN, KK).ok_or_else(|| OpError::Asset {
            op: crate::family::nvfp4::GEMM.into(),
            detail: "Nvfp4Plan::new 失败(SF 仿射布局自校验或形态门)".into(),
        })?;
        let mut aq = vec![0u8; plan.a_q_bytes()];
        let mut asf = vec![0u8; plan.sfa_bytes()];
        let mut bq = vec![0u8; plan.b_q_bytes()];
        let mut bsf = vec![0u8; plan.sfb_bytes()];
        let ab: Vec<u8> = a.iter().flat_map(|x| x.to_le_bytes()).collect();
        let bb: Vec<u8> = b.iter().flat_map(|x| x.to_le_bytes()).collect();
        // SAFETY: 长度按 plan 的 `*_bytes()` 精确分配(函数内断言过)
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
        // —— 家族用它把 alpha 留在设备侧(零 D2H ⇒ 可捕获)
        let dg_b = r.alloc_block(8, "selfcheck.nvfp4.dg")?;
        let out_b = r.alloc_block(MM * NN * 2, "selfcheck.nvfp4.out")?;
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
                Arg::U64(MM as u64),
                Arg::U64(KK as u64),
                Arg::U64(NN as u64),
                Arg::F32(gb),
            ],
            grid: (0, 0, 0),
            block: (0, 0, 0),
            shared_mem: 0,
            out_elems: MM * NN,
        };
        r.dispatch(msg)?;
        r.sync()?;
        let mut raw = vec![0u8; MM * NN * 2];
        r.download_block(&out_b, &mut raw)?;

        // ---- 参考:量化后数据在 f32 里做内积 ----
        let mut worst = 0f32;
        let mut first_bad = None;
        for i in 0..MM {
            for j in 0..NN {
                let mut want = 0f32;
                for k in 0..KK {
                    want += a_ref[i * KK + k] * b_ref[j * KK + k];
                }
                want *= ga * gb;
                let got = f16_to_f32(u16::from_le_bytes([raw[(i * NN + j) * 2], raw[(i * NN + j) * 2 + 1]]));
                let err = (got - want).abs();
                if err > worst {
                    worst = err;
                }
                let rel = err / want.abs().max(1e-3);
                if rel > TOL && first_bad.is_none() {
                    first_bad = Some((i * NN + j, got, want));
                }
            }
        }
        if let Some((i, got, want)) = first_bad {
            return Err(OpError::Contract {
                op: crate::family::nvfp4::GEMM.into(),
                field: "out",
                expect: format!("相对偏差 ≤ {TOL:.1e}(host 同规则反量化参考)"),
                got: format!("首越界 idx={i} gpu={got} host={want}(max|err|={worst:.3e})"),
            });
        }
        let mut rep = CheckReport::numeric(crate::family::nvfp4::GEMM, worst, TOL);
        rep.first_bad = first_bad;
        Ok(rep)
    }
}
