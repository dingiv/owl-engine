//! **前缀无感删除**的 KV 侧基元(2026-10-20 起于 image-rotation 实验档;
//! 2026-10-23 随「前缀无感删除」方案收编 —— 窗口/滑窗形态已否决,这枚核
//! 与 Δ 表算式原样保留)。
//!
//! 一件数学 + 一件线段:
//!
//! 1. [`delta_table`] —— 位置整体前移 `Δ` 后 K 需要的**增量旋转表**。
//!    因为 m-RoPE 图像段的三个分量都以 `cur` 为锚(`pos = [cur, cur+hi, cur+wi]`),
//!    整段平移 `Δ` ⇒ 三个分量同加 `Δ` ⇒ **一个标量就能表达全部频率**:
//!    `cosΔ_j = cos(Δ·θ_j)`。这里**不重抄** `inv_freq` 公式,直接问
//!    [`MropeCfg::cos_sin`](crate::layers::mrope::MropeCfg::cos_sin) 要一行
//!    —— 与上一轮 prefill 真正用的表**同源**(theta / rotary_dim / sections
//!    全由同一份配置产出 ⇒ 不会"公式抄差了半格"静默位移)。
//! 2. [`k_shift`] —— 把**池内一段物理页**的 K 就地乘 `R(Δ)`(`Op::CallDst`;
//!    不分配、不搬页、不动 V)。
//!
//! 池几何(`[nb, hkv, hd/x, page, x]`)是引擎侧事实(`StatePool::k_shape_kv`),
//! 这里只按 [`KvShiftGeom`] 的值算网格 —— **不探池、不读环境**(被动律)。

use crate::layers::mrope::MropeCfg;
use crate::ops::SemanticKernel;
use crate::tensor::TensorOps;

/// 池内 K 的几何(值来自引擎 `StatePool`;本模块不推断)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvShiftGeom {
    /// kv 头数(分片后)
    pub hkv: usize,
    /// `head_dim`
    pub hd: usize,
    /// 页 token 数(32)
    pub page: usize,
    /// 末维向量宽(8)
    pub x: usize,
    /// `rotary_dim / 2`
    pub half: usize,
}

impl KvShiftGeom {
    /// 契约校验(结构非法 = 调用方接线错,响亮拒)
    pub fn check(&self) -> Result<(), String> {
        if self.hkv == 0 || self.hd == 0 || self.page == 0 {
            return Err(format!("KvShiftGeom: 零维几何 {self:?}"));
        }
        if self.x == 0 || !self.half.is_multiple_of(self.x) {
            return Err(format!(
                "KvShiftGeom: 要求 half % x == 0(得 half={} x={})",
                self.half, self.x
            ));
        }
        if 2 * self.half > self.hd {
            return Err(format!(
                "KvShiftGeom: 2·half = {} 超 head_dim {}",
                2 * self.half,
                self.hd
            ));
        }
        if self.page * self.x > 1024 {
            return Err(format!(
                "KvShiftGeom: page·x = {} 超 1024(block 尺寸)",
                self.page * self.x
            ));
        }
        Ok(())
    }
}

/// **增量旋转表**(单行 `[half]` f32;`Δ` = 位置前移量,正 = 往右).
///
/// 与上一轮 prefill 用的 m-RoPE 表**同源**:三网格同值(`[Δ, Δ, Δ]`)代入
/// `cos_sin` —— 图像段三分量同加 `Δ`,所以三网格同值就是"整段平移 `Δ`"。
pub fn delta_table(mrope: &MropeCfg, delta: i64) -> (Vec<f32>, Vec<f32>) {
    let d = [[delta, delta, delta]];
    mrope.cos_sin(&d)
}

/// **池内 K 的 Δ 旋转**(原地;`blocks` = 物理页号列表,f32 过线)。
///
/// 返回 = `k` 同块别名(`Op::CallDst` 语义),调用方按普通节点消费;
/// 该节点必须在读这些页的注意力**之前**求值(同 COMPUTE 流天然有序)。
pub fn k_shift(
    k: &TensorOps,
    cos1: &TensorOps,
    sin1: &TensorOps,
    blocks: &TensorOps,
    geom: KvShiftGeom,
) -> TensorOps {
    TensorOps::call_dst(k, SemanticKernel::KvRopeShift)
        .arg(cos1)
        .arg(sin1)
        .arg(blocks)
        .arg_usize(geom.hkv)
        .arg_usize(geom.hd)
        .arg_usize(geom.page)
        .arg_usize(geom.x)
        .arg_usize(geom.half)
}

/// **host 参考实现**(FP8 e4m3;对拍/单测用,生产零调用)。
///
/// 与 `cu/elems/kv_rope_shift.cu::owl_kv_rope_k_shift_fp8` 同式:
/// `e4m3 → f32 → R(−S) → e4m3`。每轮一次量化舍入是 FP8 存储的固有代价;
/// `image_evict_delta_limit` 超限时仍走整段重建。
pub fn k_shift_host_fp8(
    k: &mut [u8],
    cos1: &[f32],
    sin1: &[f32],
    blocks: &[u32],
    geom: KvShiftGeom,
) -> Result<(), String> {
    geom.check()?;
    let KvShiftGeom { hkv, hd, page, x, half } = geom;
    let dg_per_half = half / x;
    let page_elems = hkv * hd * page;
    if cos1.len() < half || sin1.len() < half {
        return Err(format!(
            "k_shift_host_fp8: 表宽 {}/{} 短于 half={half}",
            cos1.len(),
            sin1.len()
        ));
    }
    for &b in blocks {
        let base = (b as usize) * page_elems;
        if base + page_elems > k.len() {
            return Err(format!(
                "k_shift_host_fp8: 页 {b} 越出池(需 {} elems,池 {} elems)",
                base + page_elems,
                k.len()
            ));
        }
        for h in 0..hkv {
            for dg in 0..dg_per_half {
                for t in 0..page {
                    for j in 0..x {
                        let d = dg * x + j;
                        let lo = base + h * (hd * page) + dg * (page * x) + t * x + j;
                        let hi = lo + (half / x) * (page * x);
                        let c = cos1[d];
                        let s = sin1[d];
                        let a = crate::kernel::e4m3_decode(k[lo]);
                        let bb = crate::kernel::e4m3_decode(k[hi]);
                        k[lo] = crate::kernel::e4m3_encode(a * c - bb * s);
                        k[hi] = crate::kernel::e4m3_encode(bb * c + a * s);
                    }
                }
            }
        }
    }
    Ok(())
}

/// **host 参考实现**(f16;对拍/单测用,生产零调用)。
///
/// 与 `cu/elems/kv_rope_shift.cu` 逐式同源:先 `__half2float`,f32 内乘加,
/// 再 `__float2half` 回写。
///
/// ⚠️ **不是逐位锚**:设备侧 `a*c − b*s` 会被 nvcc 收缩成 FMA(默认
/// `-fmad=true`),本函数是两次独立舍入 ⇒ 末位可差 1 ulp;抵销情形按**配对
/// 量级**取 ulp(2026-10-20 真机实测)。判据见
/// `crates/kernels/tests/kv_rope_shift_parity.rs`。
///
/// `k` = 池的 f16 元素数组(整个池,扁平);半径 = `nb_kv·hkv·hd·page`。
pub fn k_shift_host_f16(
    k: &mut [u16],
    cos1: &[f32],
    sin1: &[f32],
    blocks: &[u32],
    geom: KvShiftGeom,
) -> Result<(), String> {
    geom.check()?;
    let KvShiftGeom { hkv, hd, page, x, half } = geom;
    let dg_per_half = half / x;
    let page_elems = hkv * hd * page;
    if cos1.len() < half || sin1.len() < half {
        return Err(format!(
            "k_shift_host_f16: 表宽 {}/{} 短于 half={half}",
            cos1.len(),
            sin1.len()
        ));
    }
    for &b in blocks {
        let base = (b as usize) * page_elems;
        if base + page_elems > k.len() {
            return Err(format!(
                "k_shift_host_f16: 页 {b} 越出池(需 {} elems,池 {} elems)",
                base + page_elems,
                k.len()
            ));
        }
        for h in 0..hkv {
            for dg in 0..dg_per_half {
                for t in 0..page {
                    for j in 0..x {
                        let d = dg * x + j;
                        let lo = base + h * (hd * page) + dg * (page * x) + t * x + j;
                        let hi = lo + (half / x) * (page * x);
                        let c = cos1[d];
                        let s = sin1[d];
                        let a = f32::from(half_f32(k[lo]));
                        let bb = f32::from(half_f32(k[hi]));
                        k[lo] = half_bits(a * c - bb * s);
                        k[hi] = half_bits(bb * c + a * s);
                    }
                }
            }
        }
    }
    Ok(())
}

/// `u16`(f16 位型)→ f32(与设备 `__half2float` 同值)
pub fn half_f32(bits: u16) -> half::f16 {
    half::f16::from_bits(bits)
}

/// f32 → f16 位型(**唯一舍入点**;与设备 `__float2half` 同为 RN-even)
pub fn half_bits(v: f32) -> u16 {
    half::f16::from_f32(v).to_bits()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> KvShiftGeom {
        // 0.8B 档:hkv=2 / hd=256 / page=32 / x=8 / half=32
        KvShiftGeom { hkv: 2, hd: 256, page: 32, x: 8, half: 32 }
    }

    fn cfg() -> MropeCfg {
        MropeCfg::new([11, 11, 10], 64, 1e7, 2).expect("4B/0.8B mrope")
    }

    /// Δ = 0 ⇒ 表恒等(cos=1, sin=0)⇒ 核是**逐字节空操作**
    #[test]
    fn zero_delta_table_is_identity() {
        let (c, s) = delta_table(&cfg(), 0);
        assert_eq!(c.len(), 32);
        assert_eq!(s.len(), 32);
        for (j, v) in c.iter().enumerate() {
            assert_eq!(*v, 1.0f32, "cosΔ[{j}] 应恰为 1");
        }
        for (j, v) in s.iter().enumerate() {
            assert_eq!(*v, 0.0f32, "sinΔ[{j}] 应恰为 0");
        }
    }

    /// Δ 表与"全位置表"逐行同源:同 Δ 下 `R(p0+Δ) = R(Δ)·R(p0)`
    /// (用旋转矩阵乘法的数值判据 —— 防止频率序/网格交错抄错)
    #[test]
    fn delta_table_is_a_real_rotation() {
        let m = cfg();
        let half = m.half();
        for &(p0, dp) in &[(0i64, 5i64), (37, 11), (128, 64), (1000, -333)] {
            let (c0, s0) = m.cos_sin(&[[p0, p0, p0]]);
            let (c1, s1) = m.cos_sin(&[[p0 + dp, p0 + dp, p0 + dp]]);
            let (cd, sd) = delta_table(&m, dp);
            for j in 0..half {
                // R(p1) = R(Δ)·R(p0)
                let ec = cd[j] * c0[j] - sd[j] * s0[j];
                let es = sd[j] * c0[j] + cd[j] * s0[j];
                assert!(
                    (ec - c1[j]).abs() < 1e-4 && (es - s1[j]).abs() < 1e-4,
                    "Δ 表与全位置表不同源:j={j} p0={p0} Δ={dp} 期望 ({}, {}) 得 ({ec}, {es})",
                    c1[j],
                    s1[j]
                );
            }
        }
    }

    /// 几何契约响亮拒
    #[test]
    fn geom_contract_is_loud() {
        assert!(geom().check().is_ok());
        assert!(KvShiftGeom { x: 3, ..geom() }.check().is_err(), "half % x ≠ 0 应拒");
        assert!(KvShiftGeom { half: 200, ..geom() }.check().is_err(), "2·half > hd 应拒");
        assert!(KvShiftGeom { page: 256, ..geom() }.check().is_err(), "page·x > 1024 应拒");
    }

    /// host 参考:Δ=0 空操作 / R(Δ)∘R(−Δ) 回到原值 / 只碰 K 不越界
    #[test]
    fn host_reference_rotations_round_trip() {
        let g = geom();
        let page_elems = g.hkv * g.hd * g.page;
        let mut k: Vec<u16> = (0..page_elems * 4).map(|i| half_bits(i as f32 * 1e-3 - 0.5)).collect();
        let k0 = k.clone();
        let orig = k.clone();
        let blocks = [2u32, 0u32];
        // Δ=0:恒等(表 1/0;f16 乘法逐位保真)
        let (c0, s0) = delta_table(&cfg(), 0);
        k_shift_host_f16(&mut k, &c0, &s0, &blocks, g).expect("Δ=0");
        assert_eq!(k, orig, "Δ=0 必须是逐字节空操作");
        // R(Δ)∘R(−Δ):回到原值。**误差尺度 = 配对量级**(f16 旋转对在
        // 大角度下会抵销 —— 小值元素的相对误差可以很大,绝对误差仍受
        // 配对|a|,|b| 的 ulp 支配;这是 f16 存储的固有代价,不是核错)
        let m = cfg();
        let (cp, sp) = delta_table(&m, 977);
        let (cm, sm) = delta_table(&m, -977);
        k_shift_host_f16(&mut k, &cp, &sp, &blocks, g).expect("+Δ");
        assert_ne!(k, orig, "+Δ 应真的改了字节(否则核没落到目标页)");
        k_shift_host_f16(&mut k, &cm, &sm, &blocks, g).expect("−Δ");
        let dg_per_half = g.half / g.x;
        let out_of_tol = |k: &[u16]| -> f32 {
            let mut worst = 0f32;
            for &b in &blocks {
                let base = (b as usize) * page_elems;
                for h in 0..g.hkv {
                    for dg in 0..dg_per_half {
                        for t in 0..g.page {
                            for j in 0..g.x {
                                let lo = base + h * (g.hd * g.page) + dg * (g.page * g.x) + t * g.x + j;
                                let hi = lo + (g.half / g.x) * (g.page * g.x);
                                let (a0, b0) = (k0[lo], k0[hi]);
                                let (a1, b1) = (k[lo], k[hi]);
                                let scale = [a0, b0, a1, b1]
                                    .iter()
                                    .map(|v| f32::from(half_f32(*v)).abs())
                                    .fold(0f32, f32::max);
                                let tol = 8.0 * 2f32.powi(-10) * scale.max(1e-3);
                                worst = worst.max((r(a1) - r(a0)).abs() / tol);
                                worst = worst.max((r(b1) - r(b0)).abs() / tol);
                            }
                        }
                    }
                }
            }
            worst
        };
        fn r(bits: u16) -> f32 {
            f32::from(half_f32(bits))
        }
        let worst = out_of_tol(&k);
        assert!(worst <= 1.0, "往返误差 {worst:.3}× 容差超 8 ulp(配对量级)");
        // 未列出的页一字未动
        for (i, (a, b)) in k.iter().zip(&orig).enumerate() {
            let pg = i / page_elems;
            if pg != 0 && pg != 2 {
                assert_eq!(a, b, "页 {pg} 元素 {i} 被误改(应只碰 blocks 列出的页)");
            }
        }
    }
}
