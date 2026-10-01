//! f16 装载转换热路径(2026-10-01 自 owl-f16c 独立 crate 收编入 models;
//! 唯一消费者本就是本 crate 的 formats 装载链)。
//!
//! 历史与补偿(原 crate 头注释):std::arch intrinsic 是 `#[inline]`
//! 小包装,debug 档不内联 —— 每 4 元素 5 次调用税,裸循环也只剩
//! ~80MB/s(微基准定谳,2026-09-26);而装载测试跑 debug。原方案 =
//! 独立 crate + workspace `opt-level=3` 覆盖;并入 models 后覆盖失效,
//! 补偿 = 外层入口每 buffer 仅一次调用(无调用税)+ std arch
//! intrinsic 自带 `#[inline(always)]` 在 O0 就地展开,**吞吐由微基准
//! 门验收**(`tests::bench_throughput_note`;若回退 ~80MB/s 则重启
//! 独立 crate 方案再议)。
//!
//! 舍入语义:vcvtps2ph imm=0(RNE)与 `half::f16::from_f32` 一致,
//! checksum 逐位门可证;bf16→f32 为零舍入精确展宽。

/// f16c 运行时探测(OnceLock 缓存)
fn has_f16c() -> bool {
    static DET: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *DET.get_or_init(|| std::arch::is_x86_feature_detected!("f16c"))
}

/// bf16 字节 → f16 字节(4 宽 F16C;非 x86 / 无 f16c 回退标量 half)
#[inline]
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

/// U4 仿射反量化 → f16 字节(U4B8 语义:w = (q-8) × scale)。
/// 行主序 [out, k];packed i32 [out, k/8](LSB-first 每 i32 8 nibble);
/// scales f32 [out, k/128]。128 % 8 == 0 ⇒ 每个 i32 的 8 个 nibble
/// 必落在同一 group(scale 每 i32 取一次,免逐元素除法)。
/// rayon 按行组并行;吞吐依赖下方 f32_slice_to_f16_bytes 的 SIMD 路径
/// (原 opt 覆盖 crate 时代治本案 2026-09-28 的放置先例)。
#[inline]
pub fn dequant_u4_affine_f16_bytes(
    packed: &[i32],
    scales: &[f32],
    out: usize,
    k: usize,
    dst: &mut [u8],
) {
    use rayon::prelude::*;
    debug_assert_eq!(packed.len(), out * (k / 8));
    debug_assert_eq!(scales.len(), out * (k / 128));
    debug_assert!(dst.len() >= out * k * 2);
    let groups = k / 128;
    let kpr = k / 8;
    const ROWS_PER_TASK: usize = 32;
    dst.par_chunks_mut(k * 2 * ROWS_PER_TASK)
        .enumerate()
        .for_each(|(task, block)| {
            // 行组级一次分配,组内各行复用(免逐行堆分配)
            let mut tmp = vec![0f32; k];
            for (ri, row) in block.chunks_mut(k * 2).enumerate() {
                let r = task * ROWS_PER_TASK + ri;
                let sc_row = &scales[r * groups..(r + 1) * groups];
                let pk_row = &packed[r * kpr..(r + 1) * kpr];
                for (ci, v) in pk_row.iter().enumerate() {
                    let col0 = ci * 8;
                    let sc = sc_row[col0 / 128];
                    let v = *v as u32;
                    for j in 0..8 {
                        let q = ((v >> (4 * j)) & 0xF) as i32 - 8;
                        tmp[col0 + j] = (q as f32) * sc;
                    }
                }
                f32_slice_to_f16_bytes(&tmp, row);
            }
        });
}

/// f32 切片 → f16 字节(4 宽 F16C;回退标量 half)
#[inline]
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

// ⚠️ 下方两个 SIMD 环不能 #[inline(always)] —— E0755:inline(always)
// 与 #[target_feature] 互斥。调用税已被「每 buffer 一次调用」摊薄,
// 环体性能 = std arch intrinsic 的 O0 就地展开效果,微基准门验收。
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

    /// 微基准门(并入 models 的 inline 保速验收;头注释「历史与补偿」):
    /// debug 档下 bf16/f32 转换吞吐须 ≫ 80MB/s 原病线(目标数百 MB/s 级;
    /// 原 opt-3 独立 crate 时代 = GB/s 级,本门是回退线不是冲刺线)。
    /// 恒打印;吞掉断言风险为零(只测吞吐,数值正确性归上两用例)。
    #[test]
    fn bench_throughput_note() {
        let n = 32 << 20; // 32M 元素(f32 128MB / bf16 64MB)
        let src: Vec<f32> = (0..n).map(|i| (i % 65536) as f32).collect();
        let mut dst = vec![0u8; n * 2];

        let t = std::time::Instant::now();
        f32_slice_to_f16_bytes(&src, &mut dst);
        let f32_gbps = (n * 4) as f64 / t.elapsed().as_secs_f64() / 1e9;
        eprintln!("[f16c bench] f32→f16 debug 吞吐 = {f32_gbps:.2} GB/s");

        let src_bf16: Vec<u8> = dst.clone(); // 产物回喂(bf16 位型)
        let t = std::time::Instant::now();
        bf16_bytes_to_f16_bytes(&src_bf16, &mut dst);
        let bf16_gbps = src_bf16.len() as f64 / t.elapsed().as_secs_f64() / 1e9;
        eprintln!("[f16c bench] bf16→f16 debug 吞吐 = {bf16_gbps:.2} GB/s");

        // 门:≥ 0.4 GB/s(原病线 80MB/s 的 5×;原 opt-3 时代 GB/s 级。
        // CI 机差异敏感 → 门放宽到「量级正确」,数值细节看打印趋势)
        assert!(f32_gbps > 0.4, "f32→f16 debug 吞吐 {f32_gbps:.2} GB/s 过病线");
        assert!(bf16_gbps > 0.4, "bf16→f16 debug 吞吐 {bf16_gbps:.2} GB/s 过病线");
    }
}
