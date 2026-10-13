//! GDN scalar 家族元数据(foreign:单核 AOT cubin + cast 编排;无单 `.cu` 入口)。
//!
//! 与 [`crate::ops::gdn_chunked`] 同形:两者都是 **GDN 前向的编排入口**,
//! 共享 `GDN_STATE` 状态池(rw / f32)。槽序权威历史 = `client::gdn_scalar::SIG`。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param, ScalarTy};

/// `gdn_scalar_delta_rule_fwd` —— GDN scalar 前向(单核 + cast 编排入口)
pub struct GdnScalarDeltaRuleFwd;

impl KernelSpec for GdnScalarDeltaRuleFwd {
    const ENTRY: &'static str = crate::client::gdn_scalar::GDN_SCALAR;
    /// foreign(AOT cubin + cast 编排):无单一 `.cu` 源
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 3),
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        Param::t_in("beta", Dtype::F16, 3),
        // raw gate(语义律同 chunked:cumsum 是核内私有态,不跨面)
        Param::t_in("gate", Dtype::F16, 3),
        // GDN 状态槽 [slots, HV, KD, VD](f32)
        Param::t_inout("state", Dtype::F32, 4),
        Param::t_out("out", Dtype::F16, 3),
        Param::sz("t"),
        Param::sz("slot"),
        Param::sz("nv"),
        Param::sz("nk"),
        Param::sz("kd"),
        Param::sz("vd"),
        // kd^-0.5 位型走私(显式 token)
        Param::s("scale_bits", ScalarTy::F32Bits),
    ];
    const POOL: &'static [crate::spec::PoolAccess] =
        &[crate::spec::PoolAccess::rw(crate::spec::pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static GDN_SCALAR_FWD: KernelMeta = meta_of::<GdnScalarDeltaRuleFwd>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[&GDN_SCALAR_FWD];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::meta_of;

    /// 与旧权威串(`client::gdn_scalar::SIG`)逐字一致
    #[test]
    fn scalar_slot_order_matches_legacy_sig() {
        assert_eq!(
            meta_of::<GdnScalarDeltaRuleFwd>().args_str(),
            crate::client::gdn_scalar::SIG
        );
    }
}
