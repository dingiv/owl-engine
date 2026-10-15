//! XQA verify/decode 解析面(registry 服务端面)。
//!
//! 线格式(handler 权威;8 Block + 10 sz,见 family::xqa::VERIFY_XQA_SLOTS):
//! `[T q, T k_cache, T v_cache, T page_table(i32), T seq_lens(u32),
//!   T mask(u32), T semaphores(u32), T scratch, T wr, O out, sz q_seq_len, sz hkv, sz hd,
//!   sz page, sz max_seq_len, sz q_scale_bits, sz sm_count, sz stride_page,
//!   sz stride_token, sz stride_head]`
//!
//! ⚠️ semaphores 必须**每次发射前为零**(KV-split 归约计数;参考实现
//! `sem.zero_()`)。图内路线 = 图里挂一个零写节点(引擎接线时落;本步
//! 层测试由调用方直接给零张量)。

use crate::contract::{Bytes, LaunchMsg, OpError};

pub const VERIFY_XQA: &str = crate::family::xqa::VERIFY_XQA;
/// 裸 decode 臂(同名解析口径 —— 槽序与 verify 同型,解析函数复用)
pub const DECODE_XQA: &str = crate::family::xqa::DECODE_XQA;

pub use super::BlockRef;

/// `parse_verify` 取用的槽名(声明序;两臂共用一张表 —— F3 成对律,
/// `client::tests::parse_slots_are_exactly_declared` 逐字锁死)。
pub const PARSE_SLOTS: &[&str] = &[
    "q", "k_cache", "v_cache", "page_table", "seq_lens", "mask", "semaphores", "scratch",
    "wr", "out", "q_seq_len", "hkv", "hd", "page", "max_seq_len", "q_scale_bits", "sm_count",
    "stride_page", "stride_token", "stride_head",
];

#[derive(Clone, Debug)]
pub struct XqaVerifyCall {
    pub q: BlockRef,
    pub out: Bytes,
    pub k_cache: BlockRef,
    pub v_cache: BlockRef,
    pub page_table: BlockRef,
    pub seq_lens: BlockRef,
    pub mask: BlockRef,
    pub semaphores: BlockRef,
    pub scratch: BlockRef,
    /// 依赖边(树序;K0 写池先于 XQA 读池 —— XQA 不解引用,仅保序)
    pub wr: BlockRef,
    pub q_seq_len: usize,
    pub hkv: usize,
    pub hd: usize,
    pub page: usize,
    pub max_seq_len: usize,
    pub q_scale: f32,
    pub sm_count: usize,
    pub stride_page: usize,
    pub stride_token: usize,
    pub stride_head: usize,
}

/// 解析(9 Block + 10 sz;两臂同型)。**按名取址**(F3):槽名/槽序取自
/// 声明表(`ops::xqa` 双臂共一张表);槽位漂移/名字打错 ⇒ 结构化拒。
pub fn parse_verify(msg: &LaunchMsg) -> Result<XqaVerifyCall, OpError> {
    let s = crate::client::NamedSlots::new(msg)?;
    Ok(XqaVerifyCall {
        q: s.block("q")?,
        // 槽序:[q, k_cache, v_cache, page_table, seq_lens, mask, semaphores,
        //       scratch, wr] + out(解释器追加)
        out: s.out_block("out", 0)?,
        k_cache: s.block("k_cache")?,
        v_cache: s.block("v_cache")?,
        page_table: s.block("page_table")?,
        seq_lens: s.block("seq_lens")?,
        mask: s.block("mask")?,
        semaphores: s.block("semaphores")?,
        scratch: s.block("scratch")?,
        wr: s.block("wr")?,
        q_seq_len: s.u64("q_seq_len")? as usize,
        hkv: s.u64("hkv")? as usize,
        hd: s.u64("hd")? as usize,
        page: s.u64("page")? as usize,
        max_seq_len: s.u64("max_seq_len")? as usize,
        q_scale: f32::from_bits(s.u64("q_scale_bits")? as u32),
        sm_count: s.u64("sm_count")? as usize,
        stride_page: s.u64("stride_page")? as usize,
        stride_token: s.u64("stride_token")? as usize,
        stride_head: s.u64("stride_head")? as usize,
    })
}

/// q_scale 位型过线(层侧 builder 与 server 侧还原共用一个口径)
pub fn q_scale_bits(attn_scale: f32, hd: usize) -> u64 {
    (attn_scale * (hd as f32).sqrt()).to_bits() as u64
}
