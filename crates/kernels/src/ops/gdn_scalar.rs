//! GDN scalar 家族元数据(foreign:单核 AOT cubin + cast 编排;无单 `.cu` 入口)。
//!
//! 与 [`crate::ops::gdn_chunked`] 同形:两者都是 **GDN 前向的编排入口**,
//! 共享 `GDN_STATE` 状态池(rw / f32)。槽序权威历史 = `client::gdn_scalar::SIG`。

use crate::contract::{Arg, Dtype, Law, OpError};
use crate::bootcheck::CheckCtx;
use crate::spec::check_from_params;
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param, ScalarTy};

/// `gdn_scalar_delta_rule_fwd` —— GDN scalar 前向(单核 + cast 编排入口)
pub struct GdnScalarDeltaRuleFwd;

impl KernelSpec for GdnScalarDeltaRuleFwd {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::gdn_scalar::GDN_SCALAR_FWD);
    const ENTRY: &'static str = crate::family::gdn_scalar::GDN_SCALAR_FWD;
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

    /// **跨参校验(2026-10-15 审查补)**:`slot` 是 host 标量,会乘进**状态块
    /// 指针**(`state.pool + slot × nv·kd·kd·4`,与 handler 同式)⇒ 必须验
    /// "落在状态块账内"。单参 `RangeRule` 表达不了(依赖 shape 与账长),故覆写
    /// `check`(先例 = `ops/xqa.rs::check_xqa`)。
    ///
    /// 为什么要紧:同族**已有实发史** —— `gdn_slot_host` 恒 0 曾让核读写别人的
    /// 会话格(长 prompt 塌陷);`slot` 越界则是它的"另一侧"。
    fn check(ctx: &CheckCtx<'_>) -> Result<(), OpError> {
        check_from_params(Self::ENTRY, Self::PARAMS, ctx)?;
        let idx = |name: &'static str| -> Result<usize, OpError> {
            Self::PARAMS.iter().position(|p| p.name == name).ok_or_else(|| {
                OpError::Contract {
                    op: Self::ENTRY.into(),
                    field: "PARAMS",
                    expect: "槽名在声明表".into(),
                    got: name.to_string(),
                }
            })
        };
        let sz = |name: &'static str| -> Result<u64, OpError> {
            let i = idx(name)?;
            match ctx.args.get(i) {
                Some(Arg::U64(v)) => Ok(*v),
                Some(Arg::I32(v)) => Ok(*v as u64),
                other => Err(OpError::Contract {
                    op: Self::ENTRY.into(),
                    field: name,
                    expect: "sz 标量".into(),
                    got: format!("{other:?}"),
                }),
            }
        };
        let n = ctx.arg_len(idx("state")?).ok_or_else(|| OpError::Contract {
            op: Self::ENTRY.into(),
            field: "state",
            expect: "状态块在账本(可查账长)".into(),
            got: "未知块/不在账本".into(),
        })?;
        let (nv, kd) = (sz("nv")?, sz("kd")?);
        let stride = nv
            .checked_mul(kd)
            .and_then(|x| x.checked_mul(kd))
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| OpError::Contract {
                op: Self::ENTRY.into(),
                field: "nv*kd*kd*4",
                expect: "不溢出".into(),
                got: "checked_mul 溢出".into(),
            })?;
        let slot = sz("slot")?;
        let end = slot
            .checked_add(1)
            .and_then(|s| s.checked_mul(stride))
            .ok_or_else(|| OpError::Contract {
                op: Self::ENTRY.into(),
                field: "slot",
                expect: "slot×stride 不溢出".into(),
                got: format!("slot={slot} stride={stride}"),
            })?;
        if end > n as u64 {
            return Err(OpError::Contract {
                op: Self::ENTRY.into(),
                field: "slot",
                expect: format!("(slot+1)×stride ≤ 状态块账长({n}B)"),
                got: format!("slot={slot} stride={stride} ⇒ end={end}"),
            });
        }
        Ok(())
    }
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
