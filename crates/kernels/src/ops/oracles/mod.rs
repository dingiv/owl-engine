//! 装配期自检 oracle(设计 §5.5)—— **独立参考实现**住这里。
//!
//! 三条纪律:
//! 1. **参考必须独立**:不许复用被测核的任何代码路径(同源副本对不出双错,
//!    坑册 §18);本模块的量化表/寻址/softmax 全部按 `.cu` 语义**手写**;
//! 2. **装配期默认执行**(`OpRegistry::boot_with` → `server::selfcheck`),
//!    失败 = 拒启;
//! 3. 报告含逐元素 `max|err|` + 首个越界元素 + 毒值律结论。
//!
//! 目前覆盖(按"曾出过事"优先级):
//! - [`k0_fp8_knhd`]:池写核(fp8/e4m3 + 写集穷举毒值自检)—— E10 案现场;
//! - [`nc_bf16_v2`]:草稿非因果块注意力(滑窗 + 单头 softmax)—— ③ 重写核。

pub mod cublas_gemm;
pub mod k0_fp8_knhd;
/// marlin oracle(需自家 packer ⇒ 随 feature 门;`SELFCHECK` 用互斥 cfg 常量挂)
#[cfg(feature = "marlin")]
pub mod marlin;
pub mod nc_bf16_v2;
pub mod qknorm_insert;

// ============================================================================
// 共享 host 参考件(与 `.cu` 语义手写对齐;零复用被测代码)
// ============================================================================

/// e4m3 码 → f32(OCP e4m3;0x7F = NaN)。
///
/// **实现 = 256 项查表**(2026-10-19,A3 实测):解析式每元素一次
/// `2f32.powi(e−7)`(libm 调用,≈50 ns/elem)。在“逐 (块,层,头,token,维)”
/// 的采集热路上那是致命开销 —— 27B fp8 池单块 **52 万元素 ⇒ 31 ms/块**
/// (实测:`[idx]` 相位拆分 D2H 0.5 ms + host 30.9 ms),而每 512 tok 就要采
/// 16 块 ⇒ **500 ms/边界**(超卖档 prefill 的头号开销)。
///
/// 表由 [`e4m3_decode_exact`](本模块的单源解析式)构造一次 ⇒ **语义逐位不变**
/// (NaN 位型不保证,但仍是 NaN)。保留解析式 = 参考可读、可对照。
#[inline]
pub fn e4m3_decode(b: u8) -> f32 {
    e4m3_lut()[b as usize]
}

/// **e4m3 全码表**(256 项)。热路应当**一次取引用**再逐元素查
/// (`e4m3_decode` 单元素版每元素一次 `OnceLock` 原子读,在逐元素循环里不划算)。
pub fn e4m3_lut() -> &'static [f32; 256] {
    static LUT: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0.0f32; 256];
        for (c, v) in t.iter_mut().enumerate() {
            *v = e4m3_decode_exact(c as u8);
        }
        t
    })
}

/// e4m3 码 → f32 的**解析式**(单源参考;[`e4m3_decode`] 的表就是它建的)。
pub fn e4m3_decode_exact(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as i32;
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let v = if e == 0 { (m as f32) * 2f32.powi(-9) } else { (1.0 + m as f32 / 8.0) * 2f32.powi(e - 7) };
    s * v
}

/// f32 → e4m3 码(**RNE 语义**,与核内 `__nv_fp8_e4m3` 同式;暴力搜最近码,
/// 平局取偶数尾码 —— 与探针口径一致,已被 GPU 逐字节验证过)
pub fn e4m3_encode(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7F;
    }
    // 零的符号必须保留(`__nv_fp8_e4m3(-0.0f)` = 0x80)—— 2026-10-13 实测:
    // 忽略这一点会在输入恰好为 ±0.0 时逐字节判据误报(1 处 0x00 vs 0x80)
    if v == 0.0 {
        return if v.is_sign_negative() { 0x80 } else { 0x00 };
    }
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for code in 0..=255u8 {
        if code & 0x7F == 0x7F {
            continue;
        }
        let err = (e4m3_decode(code) - v).abs();
        if err < best_err || (err == best_err && code & 1 == 0 && best & 1 == 1) {
            best_err = err;
            best = code;
        }
    }
    // **下溢到零的符号必须保留**(硬件 RNE:负微小量 → -0.0 = 0x80)。
    // 2026-10-13 实测:不处理会在负次正规输入上误报 1 处(0x00 vs 0x80)——
    // 这类边界正是"逐字节判据"的价值:探针输入刻意避开微小量,从没碰到过。
    if best & 0x7F == 0 {
        return if v.is_sign_negative() { 0x80 } else { 0x00 };
    }
    best
}

/// f32 → f16 位型(host;仅用于构造测试输入)
pub fn f16_bits(f: f32) -> u16 {
    half_bits(f)
}

/// f16 位型 → f32
pub fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1F) as i32;
    let m = (h & 0x3FF) as f32;
    if e == 0 {
        return s * m * 2f32.powi(-24);
    }
    s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
}

fn half_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let man = (bits >> 13) & 0x3FF;
    if exp <= 0 {
        return sign;
    }
    if exp >= 31 {
        return sign | 0x7C00;
    }
    sign | ((exp as u16) << 10) | man as u16
}

/// bf16 位型 → f32(纯位运算:bf16 = f32 高 16 位 —— oracle 零依赖律,
/// 不引 `half`(那是 marlin feature 的 optional 依赖))
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f32 → bf16 位型(RNE 截断)
pub fn f32_to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    let round = ((bits >> 16) & 1) + 0x7FFF;
    (bits.wrapping_add(round) >> 16) as u16
}

/// 确定性伪随机(LCG;幅值 ±2)—— 固定种子 ⇒ 指纹可复现
pub fn lcg_bf16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            f32_to_bf16(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 4.0)
        })
        .collect()
}

/// f16 输入(LCG;幅值 ±4)
pub fn lcg_f16(n: usize, seed: u32) -> Vec<u16> {
    (0..n)
        .map(|i| {
            let x = seed
                .wrapping_mul(0x9E37_79B9)
                .wrapping_add((i as u32).wrapping_mul(0x85EB_CA6B));
            f16_bits(((x & 0xFFFF) as f32 / 65535.0 - 0.5) * 8.0)
        })
        .collect()
}
