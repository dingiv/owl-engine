//! **KV 池页的宿主解码面**(单源)。
//!
//! 为什么要有这一件:凡是"把池里的 K/V 页搬回 host 再算"的家(索引采集的
//! Mean-K、覆盖率诊断的真实质量)都必须按**池的实际字节格式**解码。此前这些
//! 家各自写死 `elem = 2`(f16)并在 fp8 档上**响亮拒** ⇒ 27B 的 fp8 生产档
//! 根本开不了选择层。
//!
//! # 布局与格式(单源出处:池写核)
//!
//! `owl_qknorm_rope_kv_insert_f16_{knhd,fp8kv_knhd}`
//! (`crates/kernels/cu/fused/qknorm_insert.cu`):
//!
//! ```text
//! key_cache / value_cache: [nb, page, Hkv, hd]      (kNHD 单布局,K/V 同址)
//! 偏移  kk = ((block·page + off_in_page)·Hkv + kh)·hd + d
//! f16 : __half           2B/elem
//! fp8 : unsigned char    1B/elem,值 = __nv_fp8_e4m3(n).__x —— **裸 e4m3 转换,无 scale**
//! ```
//!
//! 两侧**寻址式逐式相同**,只有元素宽与转换律不同 ⇒ 分支点只有两个:
//! 每页字节数、以及「字节 → f32」的转换。转换律走
//! [`owl_models::kernel::e4m3_decode`](独立手写、已被 GPU 逐字节验证的参考),
//! **本模块不写第二份 e4m3**。
//!
//! # 格式从哪来
//!
//! 一律从**池实况**取(`StatePool::kv_fp8`),不从第二个配置项猜 ——
//! 配置项只决定池怎么建,建完之后的真相在池里。两个声明源 = 迟早对不上。

use owl_iface::contract::ModelError;

/// 池页的宿主解码格式。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvPageFormat {
    /// f16(2B/elem;`Dtype::F16` 池)。
    F16,
    /// fp8 e4m3(1B/elem;池以 `Dtype::U32` 承载,元素数 ÷4;值 = 裸 e4m3)。
    Fp8E4M3,
}

impl KvPageFormat {
    /// **从池实况定格式**(唯一产地;第二处出现即违反单源)。
    pub(crate) fn of_pool(kv_fp8: bool) -> Self {
        if kv_fp8 {
            Self::Fp8E4M3
        } else {
            Self::F16
        }
    }

    /// 元素宽(字节)。
    pub(crate) fn elem_bytes(self) -> usize {
        match self {
            Self::F16 => 2,
            Self::Fp8E4M3 => 1,
        }
    }

    /// 单页字节数(K 或 V 之一;几何 = `[page, Hkv, hd]`)。
    pub(crate) fn page_bytes(self, page: usize, hkv: usize, hd: usize) -> usize {
        page * hkv * hd * self.elem_bytes()
    }

    /// 把一整页的某个 `(token, head)` 行解成 f32(长度 = `hd`)。
    ///
    /// 页内序 = `[token][head][dim]`(kNHD 的 `off_in_page` 段)——
    /// 行首 = `(t·hkv + h)·hd`(**元素**偏移,不是字节)。
    pub(crate) fn decode_row(
        self,
        page_bytes: &[u8],
        t: usize,
        h: usize,
        hkv: usize,
        hd: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        let need = page_bytes.len() / self.elem_bytes();
        let base = (t * hkv + h) * hd;
        if out.len() != hd || base + hd > need {
            return Err(ModelError::Msg(format!(
                "KV 页解码越界:{:?} 片 {}B ⇒ {need} 元素,要 [{base}, {}) 行(长 {hd})",
                self,
                page_bytes.len(),
                base + hd
            )));
        }
        // **格式分支在循环外 + e4m3 表一次取**(2026-10-19 A3 热路修正):
        // 原式逐元素 `elem_at`(match + `OnceLock` 原子读)⇒ 11 µs/行;
        // 本式逐元素只有一次查表/转换(单块 52 万元素的采集热路上是数量级差)。
        match self {
            Self::F16 => {
                let lo = base * 2;
                let raw = &page_bytes[lo..lo + out.len() * 2];
                for (o, c) in out.iter_mut().zip(raw.chunks_exact(2)) {
                    *o = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
                }
            }
            Self::Fp8E4M3 => {
                let lut = owl_models::kernel::e4m3_lut();
                let n = out.len();
                for (o, b) in out.iter_mut().zip(&page_bytes[base..base + n]) {
                    *o = lut[*b as usize];
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 格式选择单源:池实况 → 格式(第二处若各写一份 `if kv_fp8` 即在此失守)。
    #[test]
    fn 格式从池实况定() {
        assert_eq!(KvPageFormat::of_pool(false), KvPageFormat::F16);
        assert_eq!(KvPageFormat::of_pool(true), KvPageFormat::Fp8E4M3);
        assert_eq!(KvPageFormat::F16.elem_bytes(), 2);
        assert_eq!(KvPageFormat::Fp8E4M3.elem_bytes(), 1);
    }

    #[test]
    fn 页字节数按元素宽() {
        // 27B 几何:page 32 × hkv 4 × hd 256
        let (page, hkv, hd) = (32, 4, 256);
        assert_eq!(KvPageFormat::F16.page_bytes(page, hkv, hd), 65536); // 64 KiB
        assert_eq!(KvPageFormat::Fp8E4M3.page_bytes(page, hkv, hd), 32768); // 32 KiB
    }

    /// f16 侧:解码与手写 f16 → f32 逐位同。
    #[test]
    fn f16行解码逐位同() {
        let (hkv, hd) = (2, 4); // 1 token = 2×4 = 8 元素 ⇒ 要读 token 1 需 2 token 数据
        let vals: [f32; 16] = [
            0.0, 1.0, -2.5, 0.125, 100.0, -0.0, 3.0, -7.5, // token 0
            9.0, -1.5, 0.25, -64.0, 2.0, 0.5, -0.125, 7.0, // token 1
        ];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        let mut out = [0.0f32; 4];
        KvPageFormat::F16.decode_row(&bytes, 1, 0, hkv, hd, &mut out).expect("行 1/头 0");
        assert_eq!(out, [9.0, -1.5, 0.25, -64.0]);
        KvPageFormat::F16.decode_row(&bytes, 0, 1, hkv, hd, &mut out).expect("行 0/头 1");
        assert_eq!(out, [100.0, -0.0, 3.0, -7.5]);
    }

    /// fp8 侧:**与 kernels 的单源参考逐位同**(不是"差不多")。
    /// 若将来有人在本文件里手写第二份 e4m3,这条会因两式分歧而红。
    #[test]
    fn fp8行解码与kernels参考同源() {
        let (hkv, hd) = (2, 4);
        // 2 token × 2 头 × hd4 = 16 码(含 0x7F = e4m3 NaN,判据取位型 §
        // 比 f32::NAN 相等更严)
        let codes: [u8; 16] = [
            0x00, 0x38, 0xB8, 0x30, 0x7E, 0x80, 0x40, 0xC1, // token 0
            0x01, 0x7F, 0xFF, 0x08, 0x48, 0x88, 0x20, 0x60, // token 1
        ];
        let mut out = [0.0f32; 4];
        KvPageFormat::Fp8E4M3.decode_row(&codes, 1, 0, hkv, hd, &mut out).expect("行 1/头 0");
        for (i, c) in codes[8..12].iter().enumerate() {
            assert_eq!(
                out[i].to_bits(),
                owl_models::kernel::e4m3_decode(*c).to_bits(),
                "第 {i} 个元素与 kernels 参考不一致"
            );
        }
        KvPageFormat::Fp8E4M3.decode_row(&codes, 0, 1, hkv, hd, &mut out).expect("行 0/头 1");
        for (i, c) in codes[4..8].iter().enumerate() {
            assert_eq!(out[i].to_bits(), owl_models::kernel::e4m3_decode(*c).to_bits());
        }
    }

    /// 越界 = 响亮拒(不许静默读片外)。
    #[test]
    fn 越界响亮拒() {
        let bytes = [0u8; 32];
        let mut out = [0.0f32; 4];
        assert!(KvPageFormat::F16.decode_row(&bytes, 2, 0, 2, 4, &mut out).is_err());
        assert!(KvPageFormat::F16.decode_row(&bytes, 0, 0, 2, 8, &mut out).is_err());
    }
}
