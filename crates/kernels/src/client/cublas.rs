//! cublas GEMM 解析面(registry 服务端面;models 调用点暂走 lower_kernel
//! sig 路径——cublas 家族零事故面,面成熟度按需补)。
//!
//! 线格式:`[T a, T b, T out, sz m, sz k, sz n, sz nt]`(3 Block + 4 sz;
//! a/b/out 允许 BlockSlice —— linear 行切片直喂)。nt 非零 = bf16 三参。

use crate::contract::{LaunchMsg, OpError};

pub use super::BlockRef;

pub const GEMM_F16: &str = crate::family::cublas::GEMM_F16;
pub const GEMM_BF16: &str = crate::family::cublas::GEMM_BF16;
pub const GEMM_F16_DST: &str = crate::family::cublas::GEMM_F16_DST;

/// `parse_gemm` 取用的槽名(声明序)。
///
/// **这是与声明表的接口面**(F3 成对律):改名/插槽必须本表与
/// `ops::cublas::PARAMS` 一起改;`client::tests::parse_slots_are_exactly_declared`
/// 按集合逐字锁死(少一个/多一个都红)。同型槽**对调**由
/// `ops::tests::slot_order_lock_family_ops` 的名字序列锁拦(集合相等拦不住)。
pub const PARSE_SLOTS: &[&str] = &["a", "b", "out", "m", "k", "n", "nt"];
/// **写列窗变体**的槽名(与 `GEMM_F16` 的差别 = 末两位 `ldc`/`off`)
pub const PARSE_SLOTS_DST: &[&str] = &["a", "b", "out", "m", "k", "n", "nt", "ldc", "off"];

/// 写列窗 GEMM 调用(槽序契约见 `PARSE_SLOTS_DST`)
#[derive(Clone, Copy, Debug)]
pub struct GemmDstCall {
    pub a: BlockRef,
    pub b: BlockRef,
    pub out: BlockRef,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub nt: bool,
    /// C 的行宽(= 输出张量的行宽;`out` 指针按 `off` 元素偏移)
    pub ldc: usize,
    /// 列窗起点(元素;`C = out + off*2B`)
    pub off: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct GemmCall {
    pub a: BlockRef,
    pub b: BlockRef,
    pub out: BlockRef,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// cublas 转置标志(gemm_f16/bf16 的 nt 形参;非变体选择!)
    pub nt: bool,
}

/// 解析(3 Block + 4 sz;F16/BF16 两名同构)。
///
/// **按名取址**(F3):槽名/槽序全取自声明表 `ops::cublas`(此处只写名字),
/// 名字打错或槽移走 ⇒ 结构化拒(不再静默错位)。
pub fn parse_gemm_dst(msg: &LaunchMsg) -> Result<GemmDstCall, OpError> {
    let s = crate::client::NamedSlots::new(msg)?;
    Ok(GemmDstCall {
        a: s.block("a")?,
        b: s.block("b")?,
        out: s.block("out")?,
        m: s.u64("m")? as usize,
        k: s.u64("k")? as usize,
        n: s.u64("n")? as usize,
        nt: s.u64("nt")? != 0,
        ldc: s.u64("ldc")? as usize,
        off: s.u64("off")? as usize,
    })
}

pub fn parse_gemm(msg: &LaunchMsg) -> Result<GemmCall, OpError> {
    let s = crate::client::NamedSlots::new(msg)?;
    Ok(GemmCall {
        a: s.block("a")?,
        b: s.block("b")?,
        out: s.block("out")?,
        m: s.u64("m")? as usize,
        k: s.u64("k")? as usize,
        n: s.u64("n")? as usize,
        nt: s.u64("nt")? != 0,
    })
}
