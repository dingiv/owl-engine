//! `cublas_gemm_f16` 装配期自检(**第一个 foreign oracle**;2026-10-13 P5c)。
//!
//! foreign 算子(marlin/xqa/FI/cublas)的参数是**账本块句柄**,发射走
//! 家族 runtime —— 所以 oracle 用 [`CheckRunner::dispatch`](经注册表,与生产
//! 同一路径),而不是 `launch_aot`。本 oracle 同时是 P5c 运行面的**端到端
//! 验证件**:`alloc_block` + `dispatch` + `download_block` 三件都走到。
//!
//! 语义(`client::cublas` 线格式 + `family::cublas::gemm_f16`,nt=true =
//! owl Linear 惯例):`out[T, n_out] = a[T, k] · b[n_out, k]ᵀ`,f32 累加、
//! f16 入出(`CUBLAS_COMPUTE_32F`)。
//!
//! 参考 = host f32 三重循环;判据 = **相对偏差**(f16 输出舍入 + 累加序)。

use crate::contract::{Arg, KernelSource, LaunchMsg, OpError};
use crate::spec::{BootCheck, CheckReport, CheckRunner};

use super::{f16_bits, f16_to_f32, lcg_f16};

// 自检几何(小规模;含 k 非 4 对齐与 T>1,防"历届测试都单行"类掩盖)
const TT: usize = 4;
const K: usize = 8;
const N_OUT: usize = 6;
const TOL: f32 = 1e-2; // 相对

pub struct Case {
    pub a: Vec<u16>, // [T, K] f16 位型
    pub b: Vec<u16>, // [N_OUT, K]
}

pub fn case() -> Case {
    Case { a: lcg_f16(TT * K, 0xC1), b: lcg_f16(N_OUT * K, 0xC2) }
}

/// host 参考:`out[t, o] = Σ_i a[t,i]·b[o,i]`(f32 累加)。
/// `transposed: true` = **故意做错**(按 `a·b` 而非 `a·bᵀ`)—— 敏感性负例用。
pub fn reference(c: &Case, transposed: bool) -> Vec<f32> {
    let mut out = vec![0f32; TT * N_OUT];
    for t in 0..TT {
        for o in 0..N_OUT {
            let mut acc = 0f32;
            for i in 0..K {
                // transposed: 取 b[i, o](即把 b 当 [K, N_OUT] 读)—— 布局错
                let bi = if transposed { i * N_OUT + o } else { o * K + i };
                acc += f16_to_f32(c.a[t * K + i]) * f16_to_f32(c.b[bi % c.b.len()]);
            }
            out[t * N_OUT + o] = acc;
        }
    }
    out
}

/// 经注册表发射 cublas GEMM(与生产同一路径),返回原始输出(f16 位型)
pub fn run_case(r: &mut dyn CheckRunner, c: &Case) -> Result<Vec<u16>, OpError> {
    let hb = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|h| h.to_le_bytes()).collect() };
    let da = r.alloc_block(c.a.len() * 2, "selfcheck.gemm.a")?;
    let db = r.alloc_block(c.b.len() * 2, "selfcheck.gemm.b")?;
    let dout = r.alloc_block(TT * N_OUT * 2, "selfcheck.gemm.out")?;
    r.upload_block(&da, &hb(&c.a))?;
    r.upload_block(&db, &hb(&c.b))?;

    // 线槽序(权威 = `client::cublas` 文档 / ops::cublas 表):
    // [a, b, out(O), m, k, n, nt];nt=true ⇒ m = n_out、n = T(owl Linear 惯例)
    let msg = LaunchMsg {
        kernel: KernelSource {
            name: crate::contract::names::GEMM_F16.to_string(),
            source: String::new(),
        },
        args: vec![
            Arg::Block { id: da.id },
            Arg::Block { id: db.id },
            Arg::Block { id: dout.id },
            Arg::U64(N_OUT as u64),
            Arg::U64(K as u64),
            Arg::U64(TT as u64),
            Arg::U64(1),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: TT * N_OUT,
    };
    r.dispatch(msg)?;
    r.sync()?;
    let mut raw = vec![0u8; TT * N_OUT * 2];
    r.download_block(&dout, &mut raw)?;
    Ok(raw.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect())
}

/// 相对偏差
pub fn rel_worst(got: &[u16], want: &[f32]) -> (f32, Option<(usize, f32, f32)>) {
    let mut worst = 0f32;
    let mut first = None;
    for i in 0..got.len().min(want.len()) {
        let g = f16_to_f32(got[i]);
        let d = (g - want[i]).abs();
        let rel = d / want[i].abs().max(1e-3);
        if d > worst {
            worst = d;
        }
        if rel > TOL && first.is_none() {
            first = Some((i, g, want[i]));
        }
    }
    (worst, first)
}

pub struct CublasGemmOracle;

impl BootCheck for CublasGemmOracle {
    fn name(&self) -> &'static str {
        "host f32 三重循环参考(a·bᵀ;经注册表 dispatch)"
    }

    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError> {
        let c = case();
        let got = run_case(r, &c)?;
        let want = reference(&c, false);
        let (worst, first_bad) = rel_worst(&got, &want);
        if let Some((i, g, w)) = first_bad {
            return Err(OpError::Contract {
                op: crate::contract::names::GEMM_F16.into(),
                field: "out",
                expect: format!("相对偏差 ≤ {TOL:.1e}(host f32 参考 a·bᵀ)"),
                got: format!("首越界 idx={i} gpu={g} host={w}(max|err|={worst:.3e})"),
            });
        }
        let mut rep = CheckReport::numeric(crate::contract::names::GEMM_F16, worst, TOL);
        rep.first_bad = first_bad;
        let _ = f16_bits(0.0);
        Ok(rep)
    }
}
