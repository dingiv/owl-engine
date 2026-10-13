//! `marlin_gemm_w4a16_awq` 装配期自检(**生产发射量第一**:spec 档 2822 发/轮)。
//!
//! 语义(kU4 非对称 AWQ):`out[m, n] = Σ_k a[m,k]·(q[n,k] − zp[n, k/g])·s[n, k/g]`,
//! f16 入出、f32 累加;权重以 marlin 交错布局打包(`pack_marlin_b/s/z`)。
//!
//! **参考复用自家 packer**(布局知识只住 packer),参考数学 = 原始 q/s/zp 上的
//! 显式反量化(与 `marlin_parity.rs::host_dequant_gemm_awq` 同一套独立实现 ——
//! 该实现曾与 vLLM golden 对拍过)。
//!
//! 门控:oracle 住 `#[cfg(feature = "marlin")]`(要用 packer),`SELFCHECK`
//! 在 `ops/marlin.rs` 用一对互斥 cfg 常量挂 —— 表(零门)与 oracle(门内)
//! 的接触点只有这一个。

use crate::contract::{Arg, KernelSource, LaunchMsg, OpError};
use crate::family::marlin::{c_tmp_float_len_v2, repack, v2_workspace_len};
use crate::bootcheck::{BootCheck, CheckReport, CheckRunner};

use super::{f16_bits, f16_to_f32};

// 自检几何(**必须满足 marlin 打包约束** `k%128==0 && n%256==0` ——
// 2026-10-13 实测:小形状会被 packer 直接 assert 掉):
// m=4(多行)、n=256(16 个 word 行)、k=128(1 个 k-tile)、g=32 ⇒ 4 组
const M: usize = 4;
const N: usize = 256;
const K: usize = 128;
const G: usize = 32;
const TOL: f32 = 2e-2; // 相对(噪声/信号 RMS 口径同 parity 门)

pub struct Case {
    /// [m, k] f16 位型(激活)
    pub a: Vec<u16>,
    /// [n, k] 的 int4 值(0..15;逐元素一字节)
    pub q: Vec<u8>,
    /// [n, groups] 尺度
    pub s: Vec<f32>,
    /// [n, groups] 零偏(0..15)
    pub zp: Vec<u8>,
}

pub fn case() -> Case {
    let groups = K / G;
    // 确定性伪随机(与 parity 用例同族口径;幅值小 ⇒ f16 域信噪比可判)
    let a_f: Vec<f32> = (0..M * K).map(|i| ((i as f32 * 0.31) - 4.0).sin() * 0.5).collect();
    Case {
        a: a_f.iter().map(|f| f16_bits(*f)).collect(),
        q: (0..N * K).map(|i| ((i * 7 + 3) % 16) as u8).collect(),
        s: (0..N * groups).map(|i| 0.02 + ((i * 13) % 13) as f32 * 0.004).collect(),
        zp: (0..N * groups).map(|i| ((i * 5 + 2) % 16) as u8).collect(),
    }
}

/// **独立 host 参考**(显式反量化 + f32 累加)。
///
/// `symmetric: true` = **故意做错**(忽略零偏,按 zp≡8 对称域算)——
/// 敏感性负例用;`transposed: true` = 把权重当 [k, n] 读(布局错)。
pub fn reference(c: &Case, symmetric: bool, transposed: bool) -> Vec<f32> {
    let groups = K / G;
    let mut out = vec![0f32; M * N];
    for r in 0..M {
        for nn in 0..N {
            let mut acc = 0f32;
            for kk in 0..K {
                let (qi, si, zi) = if transposed {
                    // 故意错:按 [k, n] 读
                    (kk * N + nn, (kk / G) * N + nn, (kk / G) * N + nn)
                } else {
                    (nn * K + kk, nn * groups + kk / G, nn * groups + kk / G)
                };
                let q = c.q[qi % c.q.len()] as f32;
                let z = if symmetric { 8.0 } else { c.zp[zi % c.zp.len()] as f32 };
                let s = c.s[si % c.s.len()];
                acc += f16_to_f32(c.a[r * K + kk]) * (q - z) * s;
            }
            out[r * N + nn] = acc;
        }
    }
    out
}

/// 打包并发射(经注册表 dispatch),返回原始输出(f16 位型)
pub fn run_case(r: &mut dyn CheckRunner, c: &Case) -> Result<Vec<u16>, OpError> {
    let b_packed = repack::pack_marlin_b(&c.q, K, N);
    let s_packed = repack::pack_marlin_s(&c.s, N, K / G);
    let z_packed = repack::pack_marlin_z(&c.zp, N, K / G);
    let ws_words = v2_workspace_len(N).max(N / 128 * 16);
    // c_tmp:parity 用例取 1 word(内核只把它当 FFI 透传/规约暂存)
    let c_tmp_words = c_tmp_float_len_v2(M, 1).max(1);

    let da = r.alloc_block(M * K * 2, "selfcheck.marlin.a")?;
    let db = r.alloc_block(b_packed.len() * 4, "selfcheck.marlin.b")?;
    let dout = r.alloc_block(M * N * 2, "selfcheck.marlin.out")?;
    let ds = r.alloc_block(s_packed.len() * 2, "selfcheck.marlin.s")?;
    let dz = r.alloc_block(z_packed.len() * 4, "selfcheck.marlin.z")?;
    let dws = r.alloc_block(ws_words * 4, "selfcheck.marlin.ws")?;
    let dct = r.alloc_block(c_tmp_words * 4, "selfcheck.marlin.ctmp")?;
    let le16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let le32 = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    r.upload_block(&da, &le16(&c.a))?;
    r.upload_block(&db, &le32(&b_packed))?;
    r.upload_block(&ds, &le16(&s_packed))?;
    r.upload_block(&dz, &le32(&z_packed))?;

    // 线槽序(AWQ 臂;权威 = ops::marlin 表 / client::marlin::SIG_AWQ):
    // [a, b, out(O), scales, zeros, ws, c_tmp, m, k, n, groupsize]
    let msg = LaunchMsg {
        kernel: KernelSource {
            name: crate::contract::names::GEMM_W4A16_AWQ.to_string(),
            source: String::new(),
        },
        args: vec![
            Arg::Block { id: da.id },
            Arg::Block { id: db.id },
            Arg::Block { id: dout.id },
            Arg::Block { id: ds.id },
            Arg::Block { id: dz.id },
            Arg::Block { id: dws.id },
            Arg::Block { id: dct.id },
            Arg::U64(M as u64),
            Arg::U64(K as u64),
            Arg::U64(N as u64),
            Arg::U64(G as u64),
        ],
        grid: (0, 0, 0),
        block: (0, 0, 0),
        shared_mem: 0,
        out_elems: M * N,
    };
    r.dispatch(msg)?;
    r.sync()?;
    let mut raw = vec![0u8; M * N * 2];
    r.download_block(&dout, &mut raw)?;
    Ok(raw.chunks_exact(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect())
}

/// (噪声/信号 RMS, max|err|, 首个越界)——与 parity 门同口径
pub fn noise_metrics(got: &[u16], want: &[f32]) -> (f64, f32, Option<(usize, f32, f32)>) {
    let (mut sum_abs, mut sum_sq, mut max_abs) = (0f64, 0f64, 0f32);
    let mut first = None;
    for i in 0..got.len().min(want.len()) {
        let g = f16_to_f32(got[i]);
        let d = (g - want[i]).abs();
        sum_abs += d as f64;
        sum_sq += (want[i] as f64) * (want[i] as f64);
        max_abs = max_abs.max(d);
        if first.is_none() && d > TOL * want[i].abs().max(1e-3) {
            first = Some((i, g, want[i]));
        }
    }
    let rms = (sum_sq / got.len().max(1) as f64).sqrt().max(1e-6);
    (sum_abs / got.len().max(1) as f64 / rms, max_abs, first)
}

pub struct MarlinAwqOracle;

impl BootCheck for MarlinAwqOracle {
    fn name(&self) -> &'static str {
        "host 显式反量化参考(q−zp)·s + 自家 packer(经注册表 dispatch)"
    }

    fn run(&self, r: &mut dyn CheckRunner) -> Result<CheckReport, OpError> {
        let c = case();
        let got = run_case(r, &c)?;
        let want = reference(&c, false, false);
        let (noise, max_abs, first_bad) = noise_metrics(&got, &want);
        if let Some((i, g, w)) = first_bad {
            return Err(OpError::Contract {
                op: crate::contract::names::GEMM_W4A16_AWQ.into(),
                field: "out",
                expect: format!("逐元素相对偏差 ≤ {TOL:.1e}(host 显式反量化参考)"),
                got: format!(
                    "首越界 idx={i} gpu={g} host={w}(噪声/信号 RMS={noise:.4}, max|err|={max_abs:.4})"
                ),
            });
        }
        if noise > TOL as f64 {
            return Err(OpError::Contract {
                op: crate::contract::names::GEMM_W4A16_AWQ.into(),
                field: "out",
                expect: format!("噪声/信号 RMS ≤ {TOL:.1e}"),
                got: format!("{noise:.4}(max|err|={max_abs:.4})"),
            });
        }
        Ok(CheckReport::numeric(crate::contract::names::GEMM_W4A16_AWQ, max_abs, TOL))
    }
}
