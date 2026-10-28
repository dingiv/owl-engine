//! NVFP4 GEMM 解析面(registry 服务端面)。
//!
//! 槽序:见 [`crate::family::nvfp4::GEMM_SLOTS`];名字/顺序的唯一真源是
//! `ops::nvfp4::PARAMS`(本文件只写名字,按名取址 ⇒ 槽移走/改名 = 结构化拒)。
//!
//! 语义:`out[m,n] = (ga·gb) · A[m,k] × B[n,k]ᵀ`,其中 `ga` = 激活 per-tensor
//! 全局缩放(**设备算、家族回读**),`gb` = 权重全局缩放(host 标量)。

use crate::contract::{LaunchMsg, OpError};

pub use super::BlockRef;

pub use crate::family::nvfp4::{GEMM, PACK_WEIGHT};

/// `parse_gemm` 取用的槽名(声明序)
pub const PARSE_SLOTS: &[&str] =
    &["a", "bq", "bsf", "out", "aq", "asf", "dg", "m", "k", "n", "gb"];

/// `parse_pack` 取用的槽名(声明序;见表 `ops::nvfp4::PACK_WEIGHT`)
pub const PARSE_SLOTS_PACK: &[&str] = &["src", "q", "sf", "g", "n", "k", "src_bf16"];

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

/// 一次**装载期权重打包**调用(解析结果)
#[derive(Clone, Copy, Debug)]
pub struct PackCall {
    /// 源权重字节(**不透明**:f16 或 bf16,由 `src_bf16` 解释;`[n, k]` 行主序)
    pub src: BlockRef,
    /// 产物 fp4 `[n, k/2]`
    pub q: BlockRef,
    /// 产物块缩放(SFB 交错)
    pub sf: BlockRef,
    /// 产物 per-tensor 全局缩放 `f32[1]`(装载面回读)
    pub g: BlockRef,
    pub n: usize,
    pub k: usize,
    /// 源精度开关(0 = f16,非 0 = bf16)
    pub src_bf16: bool,
}

pub fn parse_pack(msg: &LaunchMsg) -> Result<PackCall, OpError> {
    let s = crate::client::NamedSlots::new(msg)?;
    Ok(PackCall {
        src: s.block("src")?,
        q: s.block("q")?,
        sf: s.block("sf")?,
        g: s.block("g")?,
        n: s.u64("n")? as usize,
        k: s.u64("k")? as usize,
        src_bf16: s.u64("src_bf16")? != 0,
    })
}
