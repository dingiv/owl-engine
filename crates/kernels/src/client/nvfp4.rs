//! NVFP4 GEMM 解析面(registry 服务端面)。
//!
//! 槽序:见 [`crate::family::nvfp4::GEMM_SLOTS`];名字/顺序的唯一真源是
//! `ops::nvfp4::PARAMS`(本文件只写名字,按名取址 ⇒ 槽移走/改名 = 结构化拒)。
//!
//! 语义:`out[m,n] = (ga·gb) · A[m,k] × B[n,k]ᵀ`,其中 `ga` = 激活 per-tensor
//! 全局缩放(**设备算、家族回读**),`gb` = 权重全局缩放(host 标量)。

use crate::contract::{LaunchMsg, OpError};

pub use super::BlockRef;

pub use crate::family::nvfp4::GEMM;

/// `parse_gemm` 取用的槽名(声明序)
pub const PARSE_SLOTS: &[&str] =
    &["a", "bq", "bsf", "out", "aq", "asf", "dg", "m", "k", "n", "gb"];

/// 一次 NVFP4 GEMM 调用(解析结果)
#[derive(Clone, Copy, Debug)]
pub struct GemmCall {
    /// 激活 f16 `[m, k]`
    pub a: BlockRef,
    /// 权重 fp4 `[n, k/2]`(u8)
    pub bq: BlockRef,
    /// 权重块缩放(u8;CUTLASS 交错布局)
    pub bsf: BlockRef,
    /// 输出 f16 `[m, n]`
    pub out: BlockRef,
    /// 激活 fp4 `[m, k/2]`(瞬态)
    pub aq: BlockRef,
    /// 激活块缩放(瞬态)
    pub asf: BlockRef,
    /// 激活全局缩放 f32[1](设备;家族回读后当 alpha 的一半)
    pub dg: BlockRef,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    /// 权重全局缩放(host)
    pub gb: f32,
}

pub fn parse_gemm(msg: &LaunchMsg) -> Result<GemmCall, OpError> {
    let s = crate::client::NamedSlots::new(msg)?;
    Ok(GemmCall {
        a: s.block("a")?,
        bq: s.block("bq")?,
        bsf: s.block("bsf")?,
        out: s.block("out")?,
        aq: s.block("aq")?,
        asf: s.block("asf")?,
        dg: s.block("dg")?,
        m: s.u64("m")? as usize,
        k: s.u64("k")? as usize,
        n: s.u64("n")? as usize,
        gb: s.f32("gb")?,
    })
}
