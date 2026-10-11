//! GDN chunked 家族元数据(foreign:fork-bf16 六核编排;无单 `.cu` 入口)。
//!
//! 槽序权威历史 = `client::gdn_chunked::SIG`;现在唯一真源 = 本表。
//! **g 语义律(E7)**:本面只产 **raw gate**;cumsum 产物是核内私有态 ——
//! 客户端不可达(类型面 `GateRaw` 堵死 raw/cumsum 混喂)。

use crate::contract::{Dtype, Law};
use crate::spec::{
    meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param, ScalarTy,
};

/// `gdn_chunked_delta_rule_fwd` —— GDN chunked 前向(六核编排入口)
pub struct GdnChunkedDeltaRuleFwd;

impl KernelSpec for GdnChunkedDeltaRuleFwd {
    const ENTRY: &'static str = crate::contract::names::GDN_CHUNKED;
    /// foreign(预编 cubin 编排):无单一 `.cu` 源
    const SOURCE: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 3),
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        Param::t_in("beta", Dtype::F16, 3),
        // raw gate(语义律:cumsum 由 handler 内部产生,不跨面)
        Param::t_in("gate", Dtype::F16, 3),
        // GDN 状态槽 [slots, HV, KD, VD](f32)
        Param::t_inout("state", Dtype::F32, 4),
        Param::t_out("out", Dtype::F16, 3),
        Param::sz("t"),
        Param::sz("slot"),
        Param::sz("nv"),
        Param::sz("nk"),
        Param::sz("kd"),
        // kd^-0.5 位型走私(显式 token)
        Param::s("scale_bits", ScalarTy::F32Bits),
    ];
    const POOL: &'static [crate::spec::PoolAccess] =
        &[crate::spec::PoolAccess::rw(crate::spec::pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static GDN_CHUNKED_FWD: KernelMeta = meta_of::<GdnChunkedDeltaRuleFwd>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[&GDN_CHUNKED_FWD];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::meta_of;

    /// 与旧权威串(`client::gdn_chunked::SIG`)逐字一致
    #[test]
    fn chunked_slot_order_matches_legacy_sig() {
        assert_eq!(
            meta_of::<GdnChunkedDeltaRuleFwd>().args_str(),
            "T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz"
        );
    }
}
