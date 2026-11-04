//! **KV 池页的宿主解码面**(单源)。
//!
//! 索引采集的 Mean-K、覆盖率诊断的真实质量等"把池页搬回 host 再算"的家,
//! 必须按池的实际字节格式解码。K 与 V 的布局在 INT8 档**不同**：
//!
//! ```text
//! f16 / fp8 : K/V 都是 [nb, page, Hkv, hd]（kNHD）
//! INT8 K    : [nb, page, Hkv, hd] int8 + Kscale [nb, page, Hkv, 8] f32
//! INT8 V    : [nb, page, Hkv, hd] int8 + Vscale [nb, page, Hkv] f32
//! ```
//!
//! 因此这里提供 **`KvPageCodec` trait**，把"元素宽 / K 行 / V 行 / scale 槽"
//! 收进格式对象；调用点只拿 `&dyn KvPageCodec`，不再各自写
//! `if kv_fp8 { ... } else if kv_int8 { ... }`。格式从池实况取
//! (`KvPageFormat::of_pool`)，不从第二个配置项猜。

use owl_iface::contract::ModelError;

/// 池页的宿主解码格式。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KvPageFormat {
    F16,
    Fp8E4M3,
    Int8,
}

impl KvPageFormat {
    /// **从池实况定格式**(唯一产地;第二处出现即违反单源)。
    pub(crate) fn of_pool(kv_fp8: bool, kv_int8: bool) -> Self {
        if kv_int8 {
            Self::Int8
        } else if kv_fp8 {
            Self::Fp8E4M3
        } else {
            Self::F16
        }
    }

    /// 格式对象。分支只在这里发生一次；行解码走 trait。
    pub(crate) fn codec(self) -> &'static dyn KvPageCodec {
        match self {
            Self::F16 => &F16_CODEC,
            Self::Fp8E4M3 => &FP8_CODEC,
            Self::Int8 => &INT8_CODEC,
        }
    }

    #[cfg(test)]
    pub(crate) fn elem_bytes(self) -> usize {
        self.codec().elem_bytes()
    }

    #[cfg(test)]
    pub(crate) fn page_bytes(self, page: usize, hkv: usize, hd: usize) -> usize {
        self.codec().page_bytes(page, hkv, hd)
    }
}

/// 一页的 scale 视图。f16/fp8 用空切片；INT8 用 page-local 的
/// `[page,Hkv,8]` / `[page,Hkv]` f32。
pub(crate) struct KvRowScale<'a> {
    pub(crate) k: &'a [f32],
    pub(crate) v: &'a [f32],
}

impl<'a> KvRowScale<'a> {
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self { k: &[], v: &[] }
    }
}

/// 池页宿主解码契约。**K 与 V 分开**：INT8 V 是 token-major
/// `[page,Hkv,hd]`，与 K 的 kNHD 行式同形但字节语义不同（K 有 8 组 scale）。
pub(crate) trait KvPageCodec: Sync {
    fn elem_bytes(&self) -> usize;

    fn page_bytes(&self, page: usize, hkv: usize, hd: usize) -> usize {
        page * hkv * hd * self.elem_bytes()
    }

    /// 每页 K scale 的 f32 元素数(0 = 该格式无 scale)。
    fn k_scale_elems(&self, _page: usize, _hkv: usize) -> usize {
        0
    }

    /// 每页 V scale 的 f32 元素数(0 = 该格式无 scale)。
    fn v_scale_elems(&self, _page: usize, _hkv: usize) -> usize {
        0
    }

    fn k_row(
        &self,
        page: &[u8],
        scale: &KvRowScale<'_>,
        t: usize,
        h: usize,
        hkv: usize,
        hd: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError>;

    fn v_row(
        &self,
        page: &[u8],
        scale: &KvRowScale<'_>,
        t: usize,
        h: usize,
        hkv: usize,
        hd: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError>;
}

fn check_row(
    what: &str,
    page_bytes: usize,
    elem_bytes: usize,
    base: usize,
    hd: usize,
    out_len: usize,
) -> Result<(), ModelError> {
    let need = page_bytes / elem_bytes;
    if out_len != hd || base + hd > need {
        return Err(ModelError::Msg(format!(
            "KV 页解码越界:{what} 片 {page_bytes}B ⇒ {need} 元素,要 [{base}, {}) 行(长 {hd})",
            base + hd
        )));
    }
    Ok(())
}

struct F16Codec;
static F16_CODEC: F16Codec = F16Codec;

impl KvPageCodec for F16Codec {
    fn elem_bytes(&self) -> usize {
        2
    }

    fn k_row(
        &self,
        page: &[u8],
        _scale: &KvRowScale<'_>,
        t: usize,
        h: usize,
        hkv: usize,
        hd: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        let base = (t * hkv + h) * hd;
        check_row("f16", page.len(), 2, base, hd, out.len())?;
        let lo = base * 2;
        let raw = &page[lo..lo + out.len() * 2];
        for (o, c) in out.iter_mut().zip(raw.chunks_exact(2)) {
            *o = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
        }
        Ok(())
    }

    fn v_row(
        &self,
        page: &[u8],
        scale: &KvRowScale<'_>,
        t: usize,
        h: usize,
        hkv: usize,
        hd: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        // f16 的 V 与 K 同布局。
        self.k_row(page, scale, t, h, hkv, hd, out)
    }
}

struct Fp8Codec;
static FP8_CODEC: Fp8Codec = Fp8Codec;

impl KvPageCodec for Fp8Codec {
    fn elem_bytes(&self) -> usize {
        1
    }

    fn k_row(
        &self,
        page: &[u8],
        _scale: &KvRowScale<'_>,
        t: usize,
        h: usize,
        hkv: usize,
        hd: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        let base = (t * hkv + h) * hd;
        check_row("fp8", page.len(), 1, base, hd, out.len())?;
        let lut = owl_models::kernel::e4m3_lut();
        let n = out.len();
        for (o, b) in out.iter_mut().zip(&page[base..base + n]) {
            *o = lut[*b as usize];
        }
        Ok(())
    }

    fn v_row(
        &self,
        page: &[u8],
        scale: &KvRowScale<'_>,
        t: usize,
        h: usize,
        hkv: usize,
        hd: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        // fp8 的 V 与 K 同布局。
        self.k_row(page, scale, t, h, hkv, hd, out)
    }
}

struct Int8Codec;
static INT8_CODEC: Int8Codec = Int8Codec;

impl KvPageCodec for Int8Codec {
    fn elem_bytes(&self) -> usize {
        1
    }

    fn k_scale_elems(&self, page: usize, hkv: usize) -> usize {
        page * hkv * 8
    }

    fn v_scale_elems(&self, page: usize, hkv: usize) -> usize {
        page * hkv
    }

    fn k_row(
        &self,
        page: &[u8],
        scale: &KvRowScale<'_>,
        t: usize,
        h: usize,
        hkv: usize,
        hd: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        let base = (t * hkv + h) * hd;
        check_row("int8-K", page.len(), 1, base, hd, out.len())?;
        let row = (t * hkv + h) * 8;
        if scale.k.len() < (t + 1) * hkv * 8 {
            return Err(ModelError::Msg(format!(
                "INT8 K scale 片太小:{} < {}",
                scale.k.len(),
                (t + 1) * hkv * 8
            )));
        }
        for (d, o) in out.iter_mut().enumerate() {
            *o = (page[base + d] as i8) as f32 * scale.k[row + d / 32];
        }
        Ok(())
    }

    fn v_row(
        &self,
        page: &[u8],
        scale: &KvRowScale<'_>,
        t: usize,
        h: usize,
        hkv: usize,
        hd: usize,
        out: &mut [f32],
    ) -> Result<(), ModelError> {
        if t >= page.len() / (hkv * hd).max(1) || h >= hkv || out.len() != hd {
            return Err(ModelError::Msg(format!(
                "INT8 V 行越界:t={t} h={h} hkv={hkv} hd={hd} 片 {}B",
                page.len()
            )));
        }
        let vs = t * hkv + h;
        if scale.v.len() <= vs {
            return Err(ModelError::Msg(format!(
                "INT8 V scale 片太小:{} <= {vs}",
                scale.v.len()
            )));
        }
        let base = (t * hkv + h) * hd;
        check_row("int8-V", page.len(), 1, base, hd, out.len())?;
        for (d, o) in out.iter_mut().enumerate() {
            *o = (page[base + d] as i8) as f32 * scale.v[vs];
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 格式从池实况定() {
        assert_eq!(KvPageFormat::of_pool(false, false), KvPageFormat::F16);
        assert_eq!(KvPageFormat::of_pool(true, false), KvPageFormat::Fp8E4M3);
        assert_eq!(KvPageFormat::of_pool(false, true), KvPageFormat::Int8);
        assert_eq!(KvPageFormat::F16.elem_bytes(), 2);
        assert_eq!(KvPageFormat::Fp8E4M3.elem_bytes(), 1);
        assert_eq!(KvPageFormat::Int8.elem_bytes(), 1);
    }

    #[test]
    fn 页字节数按元素宽() {
        let (page, hkv, hd) = (32, 4, 256);
        assert_eq!(KvPageFormat::F16.page_bytes(page, hkv, hd), 65536);
        assert_eq!(KvPageFormat::Fp8E4M3.page_bytes(page, hkv, hd), 32768);
        assert_eq!(KvPageFormat::Int8.page_bytes(page, hkv, hd), 32768);
    }

    #[test]
    fn f16行解码逐位同() {
        let (hkv, hd) = (2, 4);
        let vals: [f32; 16] = [
            0.0, 1.0, -2.5, 0.125, 100.0, -0.0, 3.0, -7.5, 9.0, -1.5, 0.25, -64.0, 2.0, 0.5,
            -0.125, 7.0,
        ];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
        let codec = KvPageFormat::F16.codec();
        let scale = KvRowScale::empty();
        let mut out = [0.0f32; 4];
        codec.k_row(&bytes, &scale, 1, 0, hkv, hd, &mut out).unwrap();
        assert_eq!(out, [9.0, -1.5, 0.25, -64.0]);
        codec.k_row(&bytes, &scale, 0, 1, hkv, hd, &mut out).unwrap();
        assert_eq!(out, [100.0, -0.0, 3.0, -7.5]);
    }

    #[test]
    fn fp8行解码与kernels参考同源() {
        let (hkv, hd) = (2, 4);
        let codes: [u8; 16] = [
            0x00, 0x38, 0xB8, 0x30, 0x7E, 0x80, 0x40, 0xC1, 0x01, 0x7F, 0xFF, 0x08, 0x48,
            0x88, 0x20, 0x60,
        ];
        let codec = KvPageFormat::Fp8E4M3.codec();
        let scale = KvRowScale::empty();
        let mut out = [0.0f32; 4];
        codec.k_row(&codes, &scale, 1, 0, hkv, hd, &mut out).unwrap();
        for (i, c) in codes[8..12].iter().enumerate() {
            assert_eq!(
                out[i].to_bits(),
                owl_models::kernel::e4m3_decode(*c).to_bits()
            );
        }
        codec.k_row(&codes, &scale, 0, 1, hkv, hd, &mut out).unwrap();
        for (i, c) in codes[4..8].iter().enumerate() {
            assert_eq!(
                out[i].to_bits(),
                owl_models::kernel::e4m3_decode(*c).to_bits()
            );
        }
    }

    #[test]
    fn int8_k行乘scale() {
        let (page, hkv, hd) = (2, 1, 4);
        let codes: [i8; 8] = [1, -2, 3, -4, 5, -6, 7, -8];
        let mut bytes = Vec::new();
        for c in codes {
            bytes.push(c as u8);
        }
        let ks = [
            0.5f32, 0.25, 2.0, 4.0, 1.0, 1.5, 2.5, 3.5, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0,
            9.0,
        ];
        let scale = KvRowScale {
            k: &ks,
            v: &[],
        };
        let codec = KvPageFormat::Int8.codec();
        assert_eq!(codec.k_scale_elems(page, hkv), 16);
        let mut out = [0.0f32; 4];
        codec.k_row(&bytes, &scale, 1, 0, hkv, hd, &mut out).unwrap();
        assert_eq!(out, [45.0, -54.0, 63.0, -72.0]);
    }

    #[test]
    fn int8_v行解token_major() {
        let (page, hkv, hd) = (4, 1, 2);
        // [page][Hkv][D] int8:t0=(1,5), t1=(2,6), t2=(3,7), t3=(4,8)。
        let vals: [[i8; 2]; 4] = [[1, 5], [2, 6], [3, 7], [4, 8]];
        let mut page_bytes = Vec::new();
        for row in vals {
            for v in row {
                page_bytes.push(v as u8);
            }
        }
        let vs = [0.5f32, 1.0, 2.0, 4.0];
        let scale = KvRowScale {
            k: &[],
            v: &vs,
        };
        let codec = KvPageFormat::Int8.codec();
        assert_eq!(codec.v_scale_elems(page, hkv), 4);
        let mut out = [0.0f32; 2];
        for t in 0..page {
            codec
                .v_row(&page_bytes, &scale, t, 0, hkv, hd, &mut out)
                .unwrap();
            let d0 = (t + 1) as f32 * vs[t];
            let d1 = (t + 5) as f32 * vs[t];
            assert_eq!(out, [d0, d1], "t={t}");
        }
    }

    #[test]
    fn 越界响亮拒() {
        let bytes = [0u8; 32];
        let mut out = [0.0f32; 4];
        let codec = KvPageFormat::F16.codec();
        let scale = KvRowScale::empty();
        assert!(codec.k_row(&bytes, &scale, 2, 0, 2, 4, &mut out).is_err());
        assert!(codec.k_row(&bytes, &scale, 0, 0, 2, 8, &mut out).is_err());
    }
}
