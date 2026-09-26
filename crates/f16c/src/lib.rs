//! f16 装载转换热路径。
//!
//! 为什么独立成 crate:std::arch intrinsic 是 `#[inline]` 小包装,debug 档
//! 不内联 —— 每 4 元素 5 次调用税,裸循环也只剩 ~80MB/s(微基准定谳,
//! 2026-09-26);而装载测试跑 debug。workspace 对本 crate 单独
//! `opt-level=3`(见根 Cargo.toml profile 覆盖),调试构建下转换段回到
//! GB/s 级,其余 crate 保持快速编译迭代。
//!
//! 舍入语义:vcvtps2ph imm=0(RNE)与 `half::f16::from_f32` 一致,
//! checksum 逐位门可证;bf16→f32 为零舍入精确展宽。

fn has_f16c() -> bool {
    static DET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DET.get_or_init(|| std::arch::is_x86_feature_detected!("f16c"))
}

/// bf16 字节 → f16 字节(4 宽 F16C;非 x86 / 无 f16c 回退标量 half)
pub fn bf16_bytes_to_f16_bytes(src: &[u8], dst: &mut [u8]) {
    debug_assert!(dst.len() >= src.len());
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16c() {
            // SAFETY:运行时已探明 f16c;src/dst 不重叠,按 2 字节配对
            unsafe { bf16_to_f16_f16c(src, dst) };
            return;
        }
    }
    for (i, c) in src.chunks_exact(2).enumerate() {
        let f = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
        dst[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(f).to_le_bytes());
    }
}

/// f32 切片 → f16 字节(4 宽 F16C;回退标量 half)
pub fn f32_slice_to_f16_bytes(src: &[f32], dst: &mut [u8]) {
    debug_assert!(dst.len() >= src.len() * 2);
    #[cfg(target_arch = "x86_64")]
    {
        if has_f16c() {
            // SAFETY:运行时已探明 f16c;src/dst 不重叠
            unsafe { f32_to_f16_f16c(src, dst) };
            return;
        }
    }
    for (i, f) in src.iter().enumerate() {
        dst[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(*f).to_le_bytes());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn bf16_to_f16_f16c(src: &[u8], dst: &mut [u8]) {
    use std::arch::x86_64::*;
    let n = src.len() / 2;
    let sp = src.as_ptr() as *const u16;
    let dp = dst.as_mut_ptr();
    let mut i = 0;
    while i + 4 <= n {
        let u = _mm_loadl_epi64(sp.add(i) as *const __m128i); // 4×u16
        let w = _mm_cvtepu16_epi32(u); // 4×i32(零扩展)
        let f = _mm_castsi128_ps(_mm_slli_epi32(w, 16)); // bf16 → f32(精确)
        let h = _mm_cvtps_ph::<0>(f); // 4×f16(imm 0 = RNE)
        _mm_storel_epi64(dp.add(i * 2) as *mut __m128i, h); // 低 64b = 8B
        i += 4;
    }
    for k in i..n {
        let f = f32::from_bits((*sp.add(k) as u32) << 16);
        let b = half::f16::from_f32(f).to_le_bytes();
        std::ptr::copy_nonoverlapping(b.as_ptr(), dp.add(k * 2), 2);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn f32_to_f16_f16c(src: &[f32], dst: &mut [u8]) {
    use std::arch::x86_64::*;
    let n = src.len();
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    let mut i = 0;
    while i + 4 <= n {
        let f = _mm_loadu_ps(sp.add(i));
        let h = _mm_cvtps_ph::<0>(f);
        _mm_storel_epi64(dp.add(i * 2) as *mut __m128i, h);
        i += 4;
    }
    for k in i..n {
        let b = half::f16::from_f32(*sp.add(k)).to_le_bytes();
        std::ptr::copy_nonoverlapping(b.as_ptr(), dp.add(k * 2), 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_matches_half_scalar() {
        // 位型直喂:偶数 u16 当 bf16 高位(f32 = bits << 16)
        let bit_patterns: [u16; 14] = [
            0x0000, // +0
            0x8000, // -0
            0x3f80, // 1.0
            0xbf80, // -1.0
            0x7f80, // +inf
            0xff80, // -inf
            0x7fc0, // NaN
            0x42c8, // 100.0
            0x3e00, // 0.125(可精确表示)
            0x3d8f, // 0.0859…(需 RNE 舍入)
            0x0001, // 最小非规格(bf16)→ f16 下溢路径
            0x0080, // 非规格中段
            0x477f, // 接近 f16 上溢边界(65504 侧)
            0x4780, // bf16 65536 → f16 inf(溢出)
        ];
        let src: Vec<u8> = bit_patterns.iter().flat_map(|b| b.to_le_bytes()).collect();
        let mut got = vec![0u8; src.len()];
        bf16_bytes_to_f16_bytes(&src, &mut got);
        for (i, &b) in bit_patterns.iter().enumerate() {
            let f = f32::from_bits((b as u32) << 16);
            let want = half::f16::from_f32(f).to_le_bytes();
            assert_eq!(&got[i * 2..i * 2 + 2], &want, "elem {i} (bf16 {b:#06x})");
        }
    }

    #[test]
    fn f32_matches_half_scalar() {
        let src: Vec<f32> = [0.0, -0.0, 1.0, -1.5, 65504.0, 65520.0, 1e-8, 5.9e-8, f32::INFINITY]
            .into_iter()
            .chain((0..1000).map(|i| i as f32 * 0.3717))
            .collect();
        let mut got = vec![0u8; src.len() * 2];
        f32_slice_to_f16_bytes(&src, &mut got);
        for (i, f) in src.iter().enumerate() {
            let want = half::f16::from_f32(*f).to_le_bytes();
            assert_eq!(&got[i * 2..i * 2 + 2], &want, "elem {i} ({f})");
        }
    }
}
