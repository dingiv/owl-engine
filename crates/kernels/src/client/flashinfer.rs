//! FlashInfer paged prefill 解析面(registry 服务端面)。
//!
//! 线格式(handler 权威;8 sz):`[T q, T kc_fi, T vc, T q_cu, T indices,
//! T indptr, T last_len, T wr, O out, sz total_rows, sz ctx_total, sz T,
//! sz hq, sz hkv, sz hd, sz page, sz sm_scale_bits]`(9 Block + 8 sz;
//! wr = 依赖边;sm_scale 以 f32 位型过线,server 面 from_bits 还原;
//! nb 由页表推导,非线格式字段)。

use crate::contract::{Bytes, LaunchMsg, OpError};

pub use super::BlockRef;

pub const PREFILL_FI: &str = crate::family::flashinfer::PREFILL_FI;
pub const PREFILL_FI_FP8KV: &str = crate::family::flashinfer::PREFILL_FI_FP8KV;

pub use super::fi_sig::{scale_bits, SIG};

/// `parse_prefill` 取用的槽名(声明序;两臂同名 —— F3 成对律,
/// `client::tests::parse_slots_are_exactly_declared` 逐字锁死)。
pub const PARSE_SLOTS: &[&str] = &[
    "q", "k_cache", "v_cache", "q_cu", "indices", "indptr", "last_len", "wr", "out",
    "total_rows", "ctx_total", "t", "hq", "hkv", "hd", "page", "sm_scale_bits",
];

#[derive(Clone, Debug)]
pub struct FiPrefillCall {
    pub q: BlockRef,
    pub kc_fi: BlockRef,
    pub vc: BlockRef,
    pub q_cu: BlockRef,
    pub indices: BlockRef,
    pub indptr: BlockRef,
    pub last_len: BlockRef,
    /// 依赖边(树序;FI 不解引用)
    pub wr: BlockRef,
    pub out: Bytes,
    pub total_rows: usize,
    pub ctx_total: usize,
    pub t: usize,
    pub hq: usize,
    pub hkv: usize,
    pub hd: usize,
    pub page: usize,
    pub sm_scale: f32,
    /// fp8kv 变体(按名字路由)
    pub fp8kv: bool,
}

/// 解析(9 Block + 8 sz)。**按名取址**(F3):槽名/槽序取自声明表
/// (`ops::flashinfer` 两臂各自的 PARAMS —— 同型槽 `indices`↔`indptr`、
/// `kc_fi`↔`vc` 对调 ⇒ 结构化拒而非静默错值)。
pub fn parse_prefill(msg: &LaunchMsg) -> Result<FiPrefillCall, OpError> {
    let s = crate::client::NamedSlots::new(msg)?;
    Ok(FiPrefillCall {
        q: s.block("q")?,
        kc_fi: s.block("k_cache")?,
        vc: s.block("v_cache")?,
        q_cu: s.block("q_cu")?,
        indices: s.block("indices")?,
        indptr: s.block("indptr")?,
        last_len: s.block("last_len")?,
        wr: s.block("wr")?,
        out: s.out_block("out", 0)?,
        total_rows: s.u64("total_rows")? as usize,
        ctx_total: s.u64("ctx_total")? as usize,
        t: s.u64("t")? as usize,
        hq: s.u64("hq")? as usize,
        hkv: s.u64("hkv")? as usize,
        hd: s.u64("hd")? as usize,
        page: s.u64("page")? as usize,
        sm_scale: f32::from_bits(s.u64("sm_scale_bits")? as u32),        fp8kv: msg.kernel.name == PREFILL_FI_FP8KV,
    })
}
