//! Marlin W4A16 解析面(registry 服务端面;三名同构:f16 / AWQ / bf16)。
//!
//! 线格式:`[T a, T b(qpack), T out, T scales, T ws, T c_tmp,
//! sz m, sz k, sz n, sz groupsize]`(6 Block + 4 sz;BlockSlice 兼容)。

use crate::contract::{LaunchMsg, OpError};

pub use super::BlockRef;
pub use crate::family::marlin::GEMM_W4A16;
pub use crate::family::marlin::GEMM_W4A16_AWQ;
pub use crate::family::marlin::GEMM_W4A16_BF16;

/// 槽序签名(f16/bf16 通用臂;6 Block + 4 sz,O = 输出槽。
/// 槽序单源 —— models 调用点经本常量声明,禁止手撸 —— 2026-10-12 review H 案)
pub const SIG: &str = "T,T,O,T,T,T,sz,sz,sz,sz";

/// AWQ 臂签名(kU4 has_zp:scales 后插 zeros,7 Block + 4 sz)
pub const SIG_AWQ: &str = "T,T,O,T,T,T,T,sz,sz,sz,sz";

/// `parse_gemm`(f16/bf16 臂)取用的槽名(声明序) —— F3 成对律;
/// `client::tests::parse_slots_are_exactly_declared` 逐字锁死。
pub const PARSE_SLOTS: &[&str] =
    &["a", "b", "out", "scales", "ws", "c_tmp", "m", "k", "n", "groupsize"];

/// `parse_gemm_awq` 取用的槽名 —— 同律(AWQ 臂比上面多 `zeros`)。
pub const PARSE_SLOTS_AWQ: &[&str] = &[
    "a", "b", "out", "scales", "zeros", "ws", "c_tmp", "m", "k", "n", "groupsize",
];

/// 块引用(id + 字节偏移;BlockSlice 兼容;与 cublas 面同构)
#[derive(Clone, Copy, Debug)]
pub struct MarlinCall {
    pub a: BlockRef,
    pub b: BlockRef,
    pub out: BlockRef,
    pub scales: BlockRef,
    /// AWQ zeros 槽(kU4 has_zp;仅 AWQ 臂)
    pub zeros: Option<BlockRef>,
    /// workspace(i32;marlin 排队工作区)
    pub ws: BlockRef,
    /// c_tmp(FFI 透传,内核不解引用的依赖边)
    pub c_tmp: BlockRef,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub groupsize: i32,
    /// bf16 三族实例(E5-DF3)
    pub bf16: bool,
}

/// 解析(f16/bf16:6 Block + 4 sz;bf16 标志由 registry 按名字路由)。
/// **按名取址**(F3):槽名/槽序均取自声明表(`ops::marlin` 三名各自的 PARAMS)
/// —— 拿错了名字或槽位漂移 ⇒ 结构化拒;同型槽(`ws`↔`c_tmp`↔`zeros`)对调
/// 由声明侧的 `slot_order_lock_family_ops` 双锁拦。
pub fn parse_gemm(msg: &LaunchMsg, bf16: bool) -> Result<MarlinCall, OpError> {
    let s = crate::client::NamedSlots::new(msg)?;
    Ok(MarlinCall {
        a: s.block("a")?,
        b: s.block("b")?,
        out: s.block("out")?,
        scales: s.block("scales")?,
        zeros: None,
        ws: s.block("ws")?,
        c_tmp: s.block("c_tmp")?,
        m: s.u64("m")? as usize,
        k: s.u64("k")? as usize,
        n: s.u64("n")? as usize,
        groupsize: s.u64("groupsize")? as i32,
        bf16,
    })
}

/// AWQ 臂解析(kU4 has_zp:**7 Block** = scales 后插 zeros 槽 + 4 sz;
/// FFI 实参序 = scales(3), zeros(4), ws(5), c_tmp(6);按名取址同 f16 臂)
pub fn parse_gemm_awq(msg: &LaunchMsg) -> Result<MarlinCall, OpError> {
    let s = crate::client::NamedSlots::new(msg)?;
    let zeros = s.block("zeros").map_err(|e| OpError::Contract {
        op: msg.kernel.name.clone(),
        field: "zeros",
        expect: "AWQ 臂必带 zeros 槽".into(),
        got: format!("{e}"),
    })?;
    Ok(MarlinCall {
        a: s.block("a")?,
        b: s.block("b")?,
        out: s.block("out")?,
        scales: s.block("scales")?,
        zeros: Some(zeros),
        ws: s.block("ws")?,
        c_tmp: s.block("c_tmp")?,
        m: s.u64("m")? as usize,
        k: s.u64("k")? as usize,
        n: s.u64("n")? as usize,
        groupsize: s.u64("groupsize")? as i32,
        bf16: false,
    })
}
