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

/// IQ 符号掩码位 → ±1(`kmask_iq2xs[j] = 1 << j`)。
#[inline]
fn iq_sign(bit: u8) -> f32 {
    if bit != 0 {
        -1.0
    } else {
        1.0
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
        GgmlTy::Iq4Nl => {
            // 32/18: d:f16 | qs:u8×16(低 nibble = 元素 j,高 nibble = 元素 j+16)
            for (blk, chunk) in row.chunks_exact(18).enumerate() {
                let d = f16_at(chunk, 0);
                let qs = &chunk[2..18];
                let y = &mut out[blk * 32..blk * 32 + 32];
                for j in 0..16 {
                    y[j] = d * super::iq_tables::KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32;
                    y[j + 16] =
                        d * super::iq_tables::KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }
            }
        }
        GgmlTy::Iq4Xs => {
            // 256/136: d:f16 | scales_h:u16 | scales_l:u8×4 | qs:u8×128
            for (blk, chunk) in row.chunks_exact(136).enumerate() {
                let d = f16_at(chunk, 0);
                let scales_h = u16::from_le_bytes([chunk[2], chunk[3]]);
                let scales_l = &chunk[4..8];
                let qs = &chunk[8..136];
                let y = &mut out[blk * 256..blk * 256 + 256];
                for ib in 0..8usize {
                    let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32
                        | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
                    let dl = d * (ls - 32) as f32;
                    for j in 0..16 {
                        let q = qs[ib * 16 + j];
                        y[ib * 32 + j] =
                            dl * super::iq_tables::KVALUES_IQ4NL[(q & 0xF) as usize] as f32;
                        y[ib * 32 + 16 + j] =
                            dl * super::iq_tables::KVALUES_IQ4NL[(q >> 4) as usize] as f32;
                    }
                }
            }
        }
        GgmlTy::Iq3Xxs => {
            // 256/98: d:f16 | qs:u8×64(格索引) | scales_and_signs:u8×32(每 4B 一个 ib32)
            for (blk, chunk) in row.chunks_exact(98).enumerate() {
                let d = f16_at(chunk, 0);
                let qs = &chunk[2..98];
                let y = &mut out[blk * 256..blk * 256 + 256];
                for ib32 in 0..8usize {
                    let aux = u32::from_le_bytes(
                        qs[64 + 4 * ib32..68 + 4 * ib32].try_into().unwrap(),
                    );
                    let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
                    for l in 0..4usize {
                        let signs =
                            super::iq_tables::KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
                        let g1 = super::iq_tables::IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize]
                            .to_le_bytes();
                        let g2 = super::iq_tables::IQ3XXS_GRID
                            [qs[8 * ib32 + 2 * l + 1] as usize]
                            .to_le_bytes();
                        let o = ib32 * 32 + l * 8;
                        for j in 0..4usize {
                            y[o + j] = db * g1[j] as f32 * iq_sign(signs & (1 << j));
                            y[o + 4 + j] =
                                db * g2[j] as f32 * iq_sign(signs & (1 << (j + 4)));
                        }
                    }
                }
            }
        }
        GgmlTy::Iq3S => {
            // 256/110: d:f16 | qs:u8×64 | qh:u8×8 | signs:u8×32 | scales:u8×4
            // 一个 ib32 对共享一个 scale 字节(低/高 nibble),qh/signs/qs 逐半推进。
            for (blk, chunk) in row.chunks_exact(110).enumerate() {
                let d = f16_at(chunk, 0);
                let qs = &chunk[2..66];
                let qh = &chunk[66..74];
                let signs = &chunk[74..106];
                let sc = &chunk[106..110];
                let y = &mut out[blk * 256..blk * 256 + 256];
                for pair in 0..4usize {
                    let s = sc[pair];
                    let db = [
                        d * (1.0 + 2.0 * (s & 0xF) as f32),
                        d * (1.0 + 2.0 * (s >> 4) as f32),
                    ];
                    for half in 0..2usize {
                        let qhv = qh[pair * 2 + half] as u32;
                        for l in 0..4usize {
                            let i1 = (qs[pair * 16 + half * 8 + 2 * l] as u32)
                                | ((qhv << (8 - 2 * l)) & 256);
                            let i2 = (qs[pair * 16 + half * 8 + 2 * l + 1] as u32)
                                | ((qhv << (7 - 2 * l)) & 256);
                            let g1 = super::iq_tables::IQ3S_GRID[i1 as usize].to_le_bytes();
                            let g2 = super::iq_tables::IQ3S_GRID[i2 as usize].to_le_bytes();
                            let sg = signs[pair * 8 + half * 4 + l];
                            let o = pair * 64 + half * 32 + l * 8;
                            for j in 0..4usize {
                                y[o + j] = db[half] * g1[j] as f32 * iq_sign(sg & (1 << j));
                                y[o + 4 + j] =
                                    db[half] * g2[j] as f32 * iq_sign(sg & (1 << (j + 4)));
                            }
                        }
                    }
                }
            }
        }
        GgmlTy::Q2K => {
            // 256/84: scales:u8×16(4bit 尺/最小) | qs:u8×64(2bit) | d:f16 | dmin:f16
            // ⚠️ **d/dmin 在块尾**(与 Q6_K 同型;ggml-common.h 的 Q2_K 结构体
            // 是 `scales, qs, union{d,dmin}`,不是"头部两个 half")
            for (blk, c) in row.chunks_exact(84).enumerate() {
                let scales = &c[0..16];
                let qs = &c[16..80];
                let d = f16_at(c, 80);
                let dmin = f16_at(c, 82);
                let y = &mut out[blk * 256..blk * 256 + 256];
                let mut is = 0usize;
                let mut yo = 0usize;
                for n in 0..2usize {
                    let mut shift = 0u32;
                    for _j in 0..4 {
                        for half in 0..2usize {
                            let sc = scales[is];
                            is += 1;
                            let dl = d * (sc & 0xF) as f32;
                            let ml = dmin * (sc >> 4) as f32;
                            let base = n * 32 + half * 16;
                            for l in 0..16 {
                                let q = ((qs[base + l] >> shift) & 3) as f32;
                                y[yo + l] = dl * q - ml;
                            }
                            yo += 16;
                        }
                        shift += 2;
                    }
                }
            }
        }
        GgmlTy::Q3K => {
            // 256/110: hmask:u8×32 | qs:u8×64(2bit) | scales:u8×12(6bit) | d:f16
            for (blk, c) in row.chunks_exact(110).enumerate() {
                let hm = &c[0..32];
                let qs = &c[32..96];
                let sc12 = &c[96..108];
                let d = f16_at(c, 108);
                // 12 字节 → 16 个 6bit 尺度(逐字译自 ggml-quants.c 的 kmask 拼法)
                let mut aux = [0u32; 4];
                for w in 0..3 {
                    aux[w] = u32::from_le_bytes(
                        sc12[w * 4..w * 4 + 4].try_into().unwrap(),
                    );
                }
                let tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & 0x0F0F_0F0F) | (((tmp >> 4) & 0x0303_0303) << 4);
                aux[3] = ((aux[1] >> 4) & 0x0F0F_0F0F) | (((tmp >> 6) & 0x0303_0303) << 4);
                aux[0] = (aux[0] & 0x0F0F_0F0F) | (((tmp >> 0) & 0x0303_0303) << 4);
                aux[1] = (aux[1] & 0x0F0F_0F0F) | (((tmp >> 2) & 0x0303_0303) << 4);
                let mut scales = [0i8; 16];
                for w in 0..4 {
                    // 字节面直接抄(有符号解释在读取时做:`scales[is] as f32`)
                    let b = aux[w].to_le_bytes();
                    for k in 0..4 {
                        scales[w * 4 + k] = b[k] as i8;
                    }
                }
                let y = &mut out[blk * 256..blk * 256 + 256];
                let mut is = 0usize;
                let mut yo = 0usize;
                let mut m = 1u8;
                for n in 0..2usize {
                    let mut shift = 0u32;
                    for _j in 0..4 {
                        for half in 0..2usize {
                            let dl = d * (scales[is] as f32 - 32.0);
                            is += 1;
                            let base = n * 32 + half * 16;
                            for l in 0..16 {
                                let q = ((qs[base + l] >> shift) & 3) as i8;
                                let hi = if hm[half * 16 + l] & m != 0 { 0 } else { -4 };
                                y[yo + l] = dl * (q + hi) as f32;
                            }
                            yo += 16;
                        }
                        shift += 2;
                        m <<= 1;
                    }
                }
            }
        }
        GgmlTy::Iq2Xxs => {
            // 256/66: d:f16 | qs:u16×32(每 ib32: 4B 格索引 + 4B scale/signs)
            for (blk, c) in row.chunks_exact(66).enumerate() {
                let d = f16_at(c, 0);
                let qs = &c[2..66];
                let y = &mut out[blk * 256..blk * 256 + 256];
                for ib32 in 0..8usize {
                    // ⚠️ 结构体里 `qs` 是 **uint16_t[32]**:C 的 `qs + 4*ib32`
                    // 按 u16 步进 ⇒ 字节偏移 = **8*ib32**(踩过:按 4 字节步进
                    // 会让后 7 个 ib32 全部错位,首块却恰好正确)
                    let byte = 8 * ib32;
                    let aux0 = u32::from_le_bytes(
                        qs[byte..byte + 4].try_into().unwrap(),
                    );
                    let aux1 = u32::from_le_bytes(
                        qs[byte + 4..byte + 8].try_into().unwrap(),
                    );
                    let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
                    for l in 0..4usize {
                        let grid = super::iq_tables::IQ2XXS_GRID[((aux0 >> (8 * l)) & 0xFF) as usize]
                            .to_le_bytes();
                        let signs =
                            super::iq_tables::KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                        let o = ib32 * 32 + l * 8;
                        for j in 0..8usize {
                            y[o + j] = db * grid[j] as f32 * iq_sign(signs & (1 << j));
                        }
                    }
                }
            }
        }
        GgmlTy::Iq2Xs => {
            // 256/74: d:f16 | qs:u16×32(9bit 格索引 + 7bit 符号) | scales:u8×8
            for (blk, c) in row.chunks_exact(74).enumerate() {
                let d = f16_at(c, 0);
                let qs = &c[2..66];
                let scales = &c[66..74];
                let y = &mut out[blk * 256..blk * 256 + 256];
                for ib32 in 0..8usize {
                    let db = [
                        d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25,
                        d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4usize {
                        let q = u16::from_le_bytes(
                            qs[2 * (4 * ib32 + l)..2 * (4 * ib32 + l) + 2]
                                .try_into()
                                .unwrap(),
                        );
                        let grid = super::iq_tables::IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                        let signs = super::iq_tables::KSIGNS_IQ2XS[(q >> 9) as usize];
                        let o = ib32 * 32 + l * 8;
                        for j in 0..8usize {
                            y[o + j] = db[l / 2] * grid[j] as f32 * iq_sign(signs & (1 << j));
                        }
                    }
                }
            }
        }
        GgmlTy::Iq2S => {
            // 256/82: d:f16 | qs:u8×64(前 32 索引 + 后 32 符号) | qh:u8×8 | scales:u8×8
            for (blk, c) in row.chunks_exact(82).enumerate() {
                let d = f16_at(c, 0);
                let qs = &c[2..66];
                let qh = &c[66..74];
                let scales = &c[74..82];
                let signs = &qs[32..64];
                let y = &mut out[blk * 256..blk * 256 + 256];
                for ib32 in 0..8usize {
                    let db = [
                        d * (0.5 + (scales[ib32] & 0xF) as f32) * 0.25,
                        d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
                    ];
                    for l in 0..4usize {
                        let idx = qs[4 * ib32 + l] as u32
                            | (((qh[ib32] as u32) << (8 - 2 * l)) & 0x300);
                        let grid = super::iq_tables::IQ2S_GRID[idx as usize].to_le_bytes();
                        let sg = signs[4 * ib32 + l];
                        let o = ib32 * 32 + l * 8;
                        for j in 0..8usize {
                            y[o + j] = db[l / 2] * grid[j] as f32 * iq_sign(sg & (1 << j));
                        }
                    }
                }
            }
        }
        GgmlTy::Iq1S => {
            // 256/50: d:f16 | qs:u8×32(8bit 索引) | qh:u16×8(3bit 高索引 + 3bit 尺度 + 符号位)
            for (blk, c) in row.chunks_exact(50).enumerate() {
                let d = f16_at(c, 0);
                let qs = &c[2..34];
                let qh = &c[34..50];
                let y = &mut out[blk * 256..blk * 256 + 256];
                for ib in 0..8usize {
                    let h = u16::from_le_bytes(qh[2 * ib..2 * ib + 2].try_into().unwrap());
                    let dl = d * (2.0 * ((h >> 12) & 7) as f32 + 1.0);
                    let delta = if h & 0x8000 != 0 { -0.125 } else { 0.125 };
                    for l in 0..4usize {
                        let idx = qs[4 * ib + l] as usize | ((((h >> (3 * l)) & 7) as usize) << 8);
                        let grid = super::iq_tables::IQ1S_GRID[idx].to_le_bytes();
                        let o = ib * 32 + l * 8;
                        for j in 0..8usize {
                            y[o + j] = dl * (grid[j] as i8 as f32 + delta);
                        }
                    }
                }
            }
        }
        other => {
            return Err(ModelError::Msg(format!(
                "gguf dequant: {} 未接解量化臂(已接 F32/F16/BF16/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K + \
                 IQ1_S/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_NL/IQ4_XS;余下 IQ1_M/TQ 等长尾)",
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


    /// gguf-py 源码树(对拍锚的另一端);不在 = 调用方 skip。
    fn py_root() -> Option<String> {
        owl_shared::env_reader::str("OWL_GGUF_PY")
            .or_else(|| Some("/home/div/Documents/codes/repos/llama.cpp/gguf-py".to_string()))
            .filter(|p| std::path::Path::new(p).is_dir())
    }

    /// **单张量位级对拍**:`dequant_all` vs `gguf-py`(独立实现),逐元素 max|Δ| == 0。
    ///
    /// 为什么用"位级"而不是 allclose:两边是同一规格的两种独立写法(C 循环 port vs
    /// numpy 位运算 port)⇒ 任何一处抄错都会在成千上万个元素上分叉;`== 0` 没有容差
    /// 可调,也就没有"调容差蒙过去"的余地。
    fn parity_case(py: &str, key: &str, tname: &str) {
        let Some(path) = res(key) else {
            eprintln!("skip: 资源清单缺 \"{key}\"");
            return;
        };
        let script = crate::testkit::manifest_dir().join("src/formats/gguf_dequant.py");
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
            .env("PYTHONPATH", py)
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

    /// **位级对拍门**:本模块 vs `gguf-py/gguf/quants.py`(独立实现)。
    ///
    /// 为什么这个锚够硬:两边都是"同一规格的两个独立实现"⇒ 任何一处抄错(位移量、
    /// 块内偏移、Q6_K 的 `d` 在块尾、Q5_K 的 `u1/u2` 递推)都会在成千上万个元素上分叉。
    #[test]
    fn dequant_matches_gguf_py() {
        let Some(py) = py_root() else {
            eprintln!("skip: gguf-py 源码树不在");
            return;
        };
        // 每型挑一张代表张量(4B:五型俱全;27B:补 Q5_K 无 ⇒ 只跑交集)
        let cases: &[(&str, &str)] = &[
            ("gguf:4b", "blk.0.attn_qkv.weight"),    // Q6_K
            ("gguf:4b", "blk.0.attn_gate.weight"),   // Q4_K
            ("gguf:4b", "blk.0.ffn_up.weight"),      // Q4_K(大张量)
            ("gguf:4b", "blk.0.ssm_out.weight"),     // Q8_0
            ("gguf:4b", "blk.0.ssm_a"),              // F32
            ("gguf:4b", "blk.3.attn_q.weight"),      // Q6_K(full 层)
            ("gguf:27b", "blk.1.ffn_down.weight"),   // Q4_K
            ("gguf:27b", "blk.1.attn_qkv.weight"),   // Q8_0
            ("gguf:4b", "blk.3.attn_k.weight"),      // Q5_K(full 层)
            ("gguf:27b", "blk.3.attn_v.weight"),     // Q8_0(full 层)
            ("gguf:27b", "blk.3.attn_q_norm.weight"), // F32(小张量)
        ];
        for (k, t) in cases {
            parity_case(&py, k, t);
        }
    }

    /// **IQ 族对拍**(2026-10-20;REQ-GGUF-04)—— 单开一门:IQ 张量在 27B 文件里,
    /// 混进主门会让每次"只改 k-quant"的回归都去解 27B 大张量。
    #[test]
    fn dequant_iq_matches_gguf_py() {
        let Some(py) = py_root() else {
            eprintln!("skip: gguf-py 源码树不在");
            return;
        };
        let cases: &[(&str, &str)] = &[
            ("gguf:0.8b-iq4xs", "blk.0.attn_gate.weight"), // IQ4_XS
            ("gguf:27b-iq3s", "blk.1.attn_gate.weight"),   // IQ3_XXS
            ("gguf:27b-iq3s", "blk.4.attn_gate.weight"),   // IQ3_S
            ("gguf:27b-iq3s", "blk.11.attn_k.weight"),     // IQ4_NL
            // ---- 27B 的 k-quant / IQ2 / IQ1(2026-10-20;UD 混型档)----
            ("gguf:27b-iq4xs", "blk.15.attn_q.weight"),    // Q2_K
            ("gguf:27b-iq4xs", "blk.12.ssm_out.weight"),   // Q3_K
            ("gguf:27b-iq4xs", "blk.0.ffn_gate.weight"),   // IQ2_XS
            ("gguf:27b-iq4xs", "blk.0.ffn_up.weight"),     // IQ2_S
            ("gguf:27b-iq3s", "blk.19.attn_q.weight"),     // IQ2_XXS
            ("gguf:27b-iq3s", "blk.0.ffn_up.weight"),      // IQ1_S
        ];
        for (k, t) in cases {
            parity_case(&py, k, t);
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
        // IQ3/IQ4 已接;未接臂(IQ2)必须照旧响亮拒
        assert!(dequant_row(GgmlTy::Iq2Xxs, &[0u8; 66], &mut out).is_err(), "未接臂必须拒");
    }

    /// 零块 = 全零(Q4_K 的 d=0 时:`d*q - dmin*m`,若 dmin 也 0 ⇒ 全 0)
    #[test]
    fn zero_block_is_zero() {
        for ty in [
            GgmlTy::Q8_0,
            GgmlTy::Q4K,
            GgmlTy::Q5K,
            GgmlTy::Q6K,
            GgmlTy::Iq4Nl,
            GgmlTy::Iq4Xs,
            GgmlTy::Iq3Xxs,
            GgmlTy::Iq3S,
        ] {
            let (be, bb) = ty.block();
            let mut out = vec![0f32; be];
            dequant_row(ty, &vec![0u8; bb], &mut out).unwrap();
            assert!(out.iter().all(|x| *x == 0.0), "{} 零块应为全零", ty.name());
        }
    }
}
