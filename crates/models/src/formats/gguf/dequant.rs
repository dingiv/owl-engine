//! **GGUF 块 → f32 参考解量化**(CPU;2026-10-19 刀 5)。
//!
//! 定位:本模块**不进生产热路径**(生产走 MMQ/dequant-in-kernel),它是
//! **正确性金标 + 装载期变换的执行面**:
//!
//! 1. **金标**:CUDA 内核(刀 6)与它的对拍基准;
//! 2. **装载期** 需要逐元素变换的小件(F32 的 `ssm_a`/`dt_bias`/`conv1d` 等)。
//!
//! # 位级权威
//!
//! 逐循环**翻译自 `llama.cpp/ggml/src/ggml-quants.c`**(`dequantize_row_q*` +
//! `get_scale_min_k4`,2026-10-19 本机树);对拍锚 = **`gguf-py/gguf/quants.py`**
//! (同一规格的**独立实现**,Python)⇒ 两者一致 = 规格无歧义,谁抄错都会被抓。
//!
//! 块布局(**字节序**;来源 `ggml-common.h` 的结构体定义 + `GgmlTy::block`):
//!
//! ```text
//! Q8_0 (32 元素/块,34 B): d:f16 | qs:i8 × 32
//! Q4_K (256/144):         d:f16 | dmin:f16 | scales:u8×12 | qs:u8×128
//! Q5_K (256/176):         d:f16 | dmin:f16 | scales:u8×12 | qh:u8×32 | qs:u8×128
//! Q6_K (256/210):         ql:u8×128 | qh:u8×64 | scales:i8×16 | d:f16
//! ```
//!
//! ⚠️ **Q6_K 的 `d` 在块尾**(其余在块首)—— 抄错 = 全错且看起来"有值"。

use crate::contract::ModelError;
use owl_kernels::family::gguf::GgmlTy;

/// f16 位型 → f32(权重表读 f16 尺度)
#[inline]
fn f16_at(b: &[u8], off: usize) -> f32 {
    half::f16::from_le_bytes([b[off], b[off + 1]]).to_f32()
}

/// K-quant 6bit 尺度/最小值的解包(逐字译自 `ggml-quants.c::get_scale_min_k4`)
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// **解量化一行**(`row` = 该行的块字节,长度 = 行字节数;`out` = 该行元素数)。
///
/// 不变量(违反 = 调用方 bug ⇒ 响亮拒,不静默算半截):
/// `row.len() == out.len() / block.0 * block.1`。
pub fn dequant_row(ty: GgmlTy, row: &[u8], out: &mut [f32]) -> Result<(), ModelError> {
    let (be, bb) = ty.block();
    let want_row = out.len() / be * bb;
    if row.len() != want_row {
        return Err(ModelError::Msg(format!(
            "gguf dequant: {} 行字节数 {} ≠ 应 {} ({} 元素 / 块 {}×{}B)",
            ty.name(),
            row.len(),
            want_row,
            out.len(),
            be,
            bb
        )));
    }
    if out.len() % be != 0 {
        return Err(ModelError::Msg(format!(
            "gguf dequant: {} 元素数 {} 不是块大小 {} 的倍数",
            ty.name(),
            out.len(),
            be
        )));
    }
    match ty {
        GgmlTy::F32 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        GgmlTy::F16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = f16_at(row, i * 2);
            }
        }
        GgmlTy::Bf16 => {
            for (i, o) in out.iter_mut().enumerate() {
                *o = half::bf16::from_le_bytes([row[i * 2], row[i * 2 + 1]]).to_f32();
            }
        }
        GgmlTy::Q8_0 => {
            for (blk, chunk) in row.chunks_exact(34).enumerate() {
                let d = f16_at(chunk, 0);
                let y = &mut out[blk * 32..blk * 32 + 32];
                for (j, o) in y.iter_mut().enumerate() {
                    *o = chunk[2 + j] as i8 as f32 * d;
                }
            }
        }
        GgmlTy::Q4K => {
            for (blk, chunk) in row.chunks_exact(144).enumerate() {
                let d = f16_at(chunk, 0);
                let dmin = f16_at(chunk, 2);
                let scales = &chunk[4..16];
                let qs = &chunk[16..144];
                let y = &mut out[blk * 256..blk * 256 + 256];
                let mut is = 0usize;
                let mut qoff = 0usize;
                let mut yoff = 0usize;
                while yoff < 256 {
                    let (sc1, m1) = scale_min_k4(is, scales);
                    let (sc2, m2) = scale_min_k4(is + 1, scales);
                    let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
                    let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
                    for l in 0..32 {
                        y[yoff + l] = d1 * (qs[qoff + l] & 0xF) as f32 - mm1;
                    }
                    for l in 0..32 {
                        y[yoff + 32 + l] = d2 * (qs[qoff + l] >> 4) as f32 - mm2;
                    }
                    qoff += 32;
                    yoff += 64;
                    is += 2;
                }
            }
        }
        GgmlTy::Q5K => {
            for (blk, chunk) in row.chunks_exact(176).enumerate() {
                let d = f16_at(chunk, 0);
                let dmin = f16_at(chunk, 2);
                let scales = &chunk[4..16];
                let qh = &chunk[16..48];
                let ql = &chunk[48..176];
                let y = &mut out[blk * 256..blk * 256 + 256];
                let mut is = 0usize;
                let mut qoff = 0usize;
                let mut yoff = 0usize;
                let (mut u1, mut u2) = (1u8, 2u8);
                while yoff < 256 {
                    let (sc1, m1) = scale_min_k4(is, scales);
                    let (sc2, m2) = scale_min_k4(is + 1, scales);
                    let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
                    let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
                    for l in 0..32 {
                        let hi = if qh[l] & u1 != 0 { 16 } else { 0 };
                        y[yoff + l] = d1 * ((ql[qoff + l] & 0xF) as f32 + hi as f32) - mm1;
                    }
                    for l in 0..32 {
                        let hi = if qh[l] & u2 != 0 { 16 } else { 0 };
                        y[yoff + 32 + l] = d2 * ((ql[qoff + l] >> 4) as f32 + hi as f32) - mm2;
                    }
                    qoff += 32;
                    yoff += 64;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        GgmlTy::Q6K => {
            for (blk, chunk) in row.chunks_exact(210).enumerate() {
                let ql = &chunk[0..128];
                let qh = &chunk[128..192];
                let sc = &chunk[192..208];
                let d = f16_at(chunk, 208);
                let y = &mut out[blk * 256..blk * 256 + 256];
                let (mut qo, mut qho, mut sco, mut yo) = (0usize, 0usize, 0usize, 0usize);
                while yo < 256 {
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[qo + l] & 0xF) as i16 | (((qh[qho + l] >> 0) & 3) as i16) << 4) - 32;
                        let q2 = ((ql[qo + l + 32] & 0xF) as i16 | (((qh[qho + l] >> 2) & 3) as i16) << 4) - 32;
                        let q3 = ((ql[qo + l] >> 4) as i16 | (((qh[qho + l] >> 4) & 3) as i16) << 4) - 32;
                        let q4 = ((ql[qo + l + 32] >> 4) as i16 | (((qh[qho + l] >> 6) & 3) as i16) << 4) - 32;
                        y[yo + l] = d * sc[sco + is] as i8 as f32 * q1 as f32;
                        y[yo + l + 32] = d * sc[sco + is + 2] as i8 as f32 * q2 as f32;
                        y[yo + l + 64] = d * sc[sco + is + 4] as i8 as f32 * q3 as f32;
                        y[yo + l + 96] = d * sc[sco + is + 6] as i8 as f32 * q4 as f32;
                    }
                    yo += 128;
                    qo += 64;
                    qho += 32;
                    sco += 8;
                }
            }
        }
        other => {
            return Err(ModelError::Msg(format!(
                "gguf dequant: {} 未接解量化臂(本刀只接 F32/F16/BF16/Q8_0/Q4_K/Q5_K/Q6_K;\
                 IQ 族 = 设计 R3 挂账 — 样本到位后按同一模式加)",
                other.name()
            )))
        }
    }
    Ok(())
}

/// 整体解量化(`bytes` = 整张量块字节,行主序;`rows` = 外维连乘)。
/// 返回 `rows × inner` 的 f32(行主序 = GGML 序,**未**翻转)。
pub fn dequant_all(
    ty: GgmlTy,
    bytes: &[u8],
    rows: usize,
    inner: usize,
) -> Result<Vec<f32>, ModelError> {
    let (be, bb) = ty.block();
    let row_bytes = inner / be * bb;
    if bytes.len() != row_bytes * rows {
        return Err(ModelError::Msg(format!(
            "gguf dequant: {} 张量字节 {} ≠ {} 行 × {}B/行",
            ty.name(),
            bytes.len(),
            rows,
            row_bytes
        )));
    }
    let mut out = vec![0f32; rows * inner];
    for r in 0..rows {
        dequant_row(ty, &bytes[r * row_bytes..(r + 1) * row_bytes], &mut out[r * inner..(r + 1) * inner])?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::gguf::reader::Gguf;
    use owl_shared::testkit::{resource, resource_ok};
    use std::path::PathBuf;
    use std::process::Command;

    fn res(key: &str) -> Option<PathBuf> {
        resource_ok(key).then(|| PathBuf::from(resource(key).expect("列了就有")))
    }

    /// **位级对拍门**:本模块 vs `gguf-py/gguf/quants.py`(独立实现)。
    ///
    /// 为什么这个锚够硬:两边都是"同一规格的两个独立实现"(C 循环 port vs
    /// numpy 位运算 port)⇒ 任何一处抄错(位移量、块内偏移、Q6_K 的 `d` 在块尾、
    /// Q5_K 的 `u1/u2` 递推)都会在**成千上万个元素**上分叉。
    #[test]
    fn dequant_matches_gguf_py() {
        let Some(py) = owl_shared::env_reader::str("OWL_GGUF_PY")
            .or_else(|| Some("/home/div/Documents/codes/repos/llama.cpp/gguf-py".to_string()))
            .filter(|p| std::path::Path::new(p).is_dir())
        else {
            eprintln!("skip: gguf-py 源码树不在");
            return;
        };
        let script = crate::testkit::manifest_dir().join("src/formats/gguf_dequant.py");
        // 每型挑一张代表张量(4B:五型俱全;27B:补 Q5_K 无 ⇒ 只跑交集)
        let cases: &[(&str, &str)] = &[
            ("gguf:4b", "blk.0.attn_qkv.weight"),   // Q6_K
            ("gguf:4b", "blk.0.attn_gate.weight"),  // Q4_K
            ("gguf:4b", "blk.0.ffn_up.weight"),     // Q4_K(大张量)
            ("gguf:4b", "blk.0.ssm_out.weight"),    // Q8_0
            ("gguf:4b", "blk.0.ssm_a"),             // F32
            ("gguf:4b", "blk.3.attn_q.weight"),     // Q6_K(full 层)
            ("gguf:27b", "blk.1.ffn_down.weight"),  // Q4_K
            ("gguf:27b", "blk.1.attn_qkv.weight"),  // Q8_0
            // Q5_K:4B 的 full 层 k_proj(27B 无 Q5_K 存量)
            ("gguf:4b", "blk.3.attn_k.weight"),
            ("gguf:27b", "blk.3.attn_v.weight"),    // Q8_0(full 层)
            ("gguf:27b", "blk.3.attn_q_norm.weight"), // F32(小张量)
        ];
        for (key, tname) in cases {
            let Some(path) = res(key) else {
                eprintln!("skip: 资源清单缺 \"{key}\"");
                continue;
            };
            let g = Gguf::open(&path).expect("open");
            let ti = g.get_tensor(tname).unwrap_or_else(|| panic!("[{key}] 缺张量 {tname}"));
            let ix = g.index_of(tname).unwrap();
            let bytes = g.bytes(ix);
            let rows = ti.rows() as usize;
            let inner = ti.inner() as usize;
            let mine = dequant_all(ti.ty, bytes, rows, inner).expect("dequant");

            // python 锚:导出同一张量的 f32 到临时文件
            let outp = std::env::temp_dir().join(format!(
                "owl_gguf_dq_{}_{}.bin",
                std::process::id(),
                tname.replace('.', "_")
            ));
            let out = Command::new("uv")
                .arg("run")
                .arg("python")
                .arg(&script)
                .arg(&path)
                .arg(tname)
                .arg(&outp)
                .env("PYTHONPATH", &py)
                .current_dir(crate::testkit::manifest_dir())
                .output()
                .expect("spawn uv(python 环境未就绪?先在 crates/models 下 `uv sync`)");
            assert!(
                out.status.success(),
                "[{key}] {tname} python 导出失败:\n{}",
                String::from_utf8_lossy(&out.stderr)
            );
            let raw = std::fs::read(&outp).expect("读锚文件");
            let theirs: Vec<f32> = raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            assert_eq!(mine.len(), theirs.len(), "[{key}] {tname} 元素数");
            let mut maxd = 0f32;
            let mut worst = 0usize;
            for (i, (a, b)) in mine.iter().zip(&theirs).enumerate() {
                let d = (a - b).abs();
                if d > maxd {
                    maxd = d;
                    worst = i;
                }
            }
            assert!(
                maxd == 0.0,
                "[{key}] {tname}({}) 位级不等:max|Δ|={maxd:e} @ 元素 {worst}(我方 {} vs gguf-py {})",
                ti.ty.name(),
                mine[worst],
                theirs[worst]
            );
            eprintln!(
                "[{key}] {} {tname} — {}×{} 位级一致 ✓",
                ti.ty.name(),
                rows,
                inner
            );
            let _ = std::fs::remove_file(&outp);
        }
    }

    /// 不变量:行字节数/元素数不符 = 响亮拒(不静默算半截)
    #[test]
    fn rejects_bad_geometry() {
        let mut out = vec![0f32; 256];
        assert!(dequant_row(GgmlTy::Q4K, &[0u8; 144], &mut out).is_ok());
        assert!(dequant_row(GgmlTy::Q4K, &[0u8; 143], &mut out).is_err(), "短一字节必须拒");
        assert!(dequant_row(GgmlTy::Q4K, &[0u8; 145], &mut out).is_err(), "多一字节必须拒");
        let mut bad = vec![0f32; 250];
        assert!(dequant_row(GgmlTy::Q4K, &[0u8; 144], &mut bad).is_err(), "元素数非块倍数必须拒");
        assert!(dequant_row(GgmlTy::Iq4Xs, &[0u8; 136], &mut out).is_err(), "未接臂必须拒");
    }

    /// 零块 = 全零(Q4_K 的 d=0 时:`d*q - dmin*m`,若 dmin 也 0 ⇒ 全 0)
    #[test]
    fn zero_block_is_zero() {
        for ty in [GgmlTy::Q8_0, GgmlTy::Q4K, GgmlTy::Q5K, GgmlTy::Q6K] {
            let (be, bb) = ty.block();
            let mut out = vec![0f32; be];
            dequant_row(ty, &vec![0u8; bb], &mut out).unwrap();
            assert!(out.iter().all(|x| *x == 0.0), "{} 零块应为全零", ty.name());
        }
    }
}
