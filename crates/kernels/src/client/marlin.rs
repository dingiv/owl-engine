//! Marlin W4A16 解析面(registry 服务端面;三名同构:f16 / AWQ / bf16)。
//!
//! 线格式:`[T a, T b(qpack), T out, T scales, T ws, T c_tmp,
//! sz m, sz k, sz n, sz groupsize]`(6 Block + 4 sz;BlockSlice 兼容)。

use crate::contract::{LaunchMsg, OpError};

pub use super::BlockRef;
pub use crate::family::marlin::GEMM_W4A16;
pub use crate::family::marlin::GEMM_W4A16_AWQ;
pub use crate::family::marlin::GEMM_W4A16_BF16;
pub use crate::family::marlin::GEMM_W4A8_AWQ;

/// 槽序签名(f16/bf16 通用臂;6 Block + 4 sz,O = 输出槽。
/// 槽序单源 —— models 调用点经本常量声明,禁止手撸 —— 2026-10-12 review H 案)
pub const SIG: &str = "T,T,O,T,T,T,sz,sz,sz,sz";

/// AWQ 臂签名(kU4 has_zp:scales 后插 zeros,7 Block + 4 sz)
pub const SIG_AWQ: &str = "T,T,O,T,T,T,T,sz,sz,sz,sz";

/// W4A8 × AWQ 臂签名(kU4 has_zp + int8 激活:zeros 后插 `a_scales`,
/// 8 Block + 4 sz)。A 槽 = 字节承载(U8 线事实;int8 语义在核侧)
pub const SIG_W4A8: &str = "T,T,O,T,T,T,T,T,sz,sz,sz,sz";

/// `parse_gemm`(f16/bf16 臂)取用的槽名(声明序) —— F3 成对律;
/// `client::tests::parse_slots_are_exactly_declared` 逐字锁死。
pub const PARSE_SLOTS: &[&str] =
    &["a", "b", "out", "scales", "ws", "c_tmp", "m", "k", "n", "groupsize"];

/// `parse_gemm_awq` 取用的槽名 —— 同律(AWQ 臂比上面多 `zeros`)。
pub const PARSE_SLOTS_AWQ: &[&str] = &[
    "a", "b", "out", "scales", "zeros", "ws", "c_tmp", "m", "k", "n", "groupsize",
];

/// `parse_gemm_w4a8_awq` 取用的槽名 —— 同律(W4A8 臂再插 `a_scales`)。
pub const PARSE_SLOTS_W4A8: &[&str] = &[
    "a", "b", "out", "scales", "zeros", "a_scales", "ws", "c_tmp", "m", "k", "n",
    "groupsize",
];

/// 块引用(id + 字节偏移;BlockSlice 兼容;与 cublas 面同构)
#[derive(Clone, Copy, Debug)]
pub struct MarlinCall {
    pub a: BlockRef,
    pub b: BlockRef,
    pub out: BlockRef,
    pub scales: BlockRef,
    /// AWQ zeros 槽(kU4 has_zp;仅 AWQ / W4A8 臂)
    pub zeros: Option<BlockRef>,
    /// per-token int8 激活尺度槽(fp32 `[M,1]`;仅 W4A8 臂)
    pub a_scales: Option<BlockRef>,
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

impl MarlinCall {
    /// **槽序 (m,k,n) → C 形参序 (m,n,k) 的换位点**(2026-10-16;唯一)。
    ///
    /// 线槽序 = `m, k, n`(声明表/`PARSE_SLOTS*` 序),而 `marlin_gemm_v2_*_ffi`
    /// 的 C 签名是 `(prob_m, prob_n, prob_k)` —— **两者序不同**。各家自写
    /// `call.m, call.n, call.k` 时把后两位写反 = n/k 静默对调:形状检查多半
    /// 看不出(两者常同为 2 的幂),结果全错而不报错。故所有 arms 一律经本
    /// 方法取序(单测用互不相同值锁住)。
    pub fn ffi_mnk(&self) -> (i32, i32, i32) {
        (self.m as i32, self.n as i32, self.k as i32)
    }
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
        a_scales: None,
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
        a_scales: None,
        ws: s.block("ws")?,
        c_tmp: s.block("c_tmp")?,
        m: s.u64("m")? as usize,
        k: s.u64("k")? as usize,
        n: s.u64("n")? as usize,
        groupsize: s.u64("groupsize")? as i32,
        bf16: false,
    })
}

/// W4A8 × AWQ 臂解析(kU4 has_zp **且** int8 激活:**7 Block** = AWQ 槽表后
/// 插 `a_scales` + 4 sz;FFI 实参序 = scales(3), zeros(4), a_scales(5),
/// ws(6), c_tmp(7);按名取址同前两臂)。
pub fn parse_gemm_w4a8_awq(msg: &LaunchMsg) -> Result<MarlinCall, OpError> {
    let s = crate::client::NamedSlots::new(msg)?;
    let zeros = s.block("zeros").map_err(|e| OpError::Contract {
        op: msg.kernel.name.clone(),
        field: "zeros",
        expect: "W4A8×AWQ 臂必带 zeros 槽(kU4 has_zp)".into(),
        got: format!("{e}"),
    })?;
    let a_scales = s.block("a_scales").map_err(|e| OpError::Contract {
        op: msg.kernel.name.clone(),
        field: "a_scales",
        expect: "W4A8×AWQ 臂必带 a_scales 槽(per-token int8 激活尺度 fp32[M,1])".into(),
        got: format!("{e}"),
    })?;
    Ok(MarlinCall {
        a: s.block("a")?,
        b: s.block("b")?,
        out: s.block("out")?,
        scales: s.block("scales")?,
        zeros: Some(zeros),
        a_scales: Some(a_scales),
        ws: s.block("ws")?,
        c_tmp: s.block("c_tmp")?,
        m: s.u64("m")? as usize,
        k: s.u64("k")? as usize,
        n: s.u64("n")? as usize,
        groupsize: s.u64("groupsize")? as i32,
        bf16: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blk(id: u64) -> BlockRef {
        BlockRef { id, byte_offset: 0 }
    }

    /// **n/k 换位锁**(第一雷):三槽取互不相同值 ⇒ 换位写错当场红。
    /// C 签名 = `(prob_m, prob_n, prob_k)`,线槽序 = `m, k, n`。
    #[test]
    fn ffi_mnk_reorders_slots_to_c_params() {
        let call = MarlinCall {
            a: blk(1), b: blk(2), out: blk(3), scales: blk(4),
            zeros: None, a_scales: None, ws: blk(5), c_tmp: blk(6),
            m: 8, k: 64, n: 256, groupsize: 32, bf16: false,
        };
        assert_eq!(
            call.ffi_mnk(),
            (8, 256, 64),
            "C 形参序 = (m, n, k);换位点错 = n/k 静默对调"
        );
    }
}
