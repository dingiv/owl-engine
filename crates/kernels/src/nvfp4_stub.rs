//! NVFP4 面的**未编入替身**(`feature = "nvfp4"` 关时)。
//!
//! 为什么要有这个文件:`cfg(feature)` 是**本 crate** 的 feature,上层
//! (`models`/`engine`/`server`)没法自己 cfg 它 —— 让每个上层各写一遍
//! 同一个开关,就会出现"三处 cfg 不同步"的静默路径。故把门控收在
//! `lib.rs` 一处:模块**恒存在**,编入与否只改实现。
//!
//! 纪律:**没有一条路径能靠本替身跑出结果**。装载期 `pack_weight_b` 响亮
//! 拒;发射面本就不存在(算子族的准入表 `admit.rs` 先拒 `NvFp4Cutlass`)。
//! 即"能构造、不能运行" —— 与 `DesktopPlatform` 之类的显式桩同律。

/// fp4 数据字节数(`k` 个半字节 = `k/2` 字节)
#[must_use]
pub const fn q_bytes(rows: usize, k: usize) -> usize {
    rows * k / 2
}

/// 缩放字节数(与编入版同式;纯算术,替身也照实给)
#[must_use]
pub const fn sf_bytes(n: usize, k: usize) -> usize {
    n * k / 16
}

/// 库自述(替身:说明未编入)
#[must_use]
pub fn banner() -> &'static str {
    "nvfp4(未编入:cargo feature nvfp4 / arch-sm120 关)"
}

/// host 侧权重打包(**未编入 ⇒ 一律拒**)。
///
/// # Errors
/// 恒 `Err`(`feature = "nvfp4"` 未开)。
pub fn pack_weight_b(
    _n: usize,
    _k: usize,
    _b_f16: &[u8],
    _q: &mut [u8],
    _sf: &mut [u8],
) -> Result<f32, String> {
    Err(format!(
        "NVFP4 host 打包不可用:本构建未编入 libnvfp4(需 --features arch-sm120;\n\
         当前自述 = {})",
        banner()
    ))
}
