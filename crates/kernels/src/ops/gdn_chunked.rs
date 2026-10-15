//! GDN chunked 家族元数据(foreign:fork-bf16 六核编排;无单 `.cu` 入口)。
//!
//! 槽序权威历史 = `client::gdn_chunked::SIG`;现在唯一真源 = 本表。
//! **g 语义律(E7)**:本面只产 **raw gate**;cumsum 产物是核内私有态 ——
//! 客户端不可达(类型面 `GateRaw` 堵死 raw/cumsum 混喂)。

use crate::contract::{Arg, Dtype, Law, OpError};
use crate::bootcheck::CheckCtx;
use crate::spec::check_from_params;
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param, ScalarTy};

/// `gdn_chunked_delta_rule_fwd` —— GDN chunked 前向(六核编排入口)
pub struct GdnChunkedDeltaRuleFwd;

impl KernelSpec for GdnChunkedDeltaRuleFwd {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::gdn_chunked::GDN_CHUNKED_FWD);
    const ENTRY: &'static str = crate::family::gdn_chunked::GDN_CHUNKED_FWD;
    /// foreign(预编 cubin 编排):无单一 `.cu` 源
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 3),
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        Param::t_in("beta", Dtype::F16, 2),
        // raw gate(语义律:cumsum 由 handler 内部产生,不跨面;[T, HV] 二继)
        Param::t_in("gate", Dtype::F16, 2),
        // GDN 状态槽 [slots, HV, KD, VD](f32)
        Param::t_inout("state", Dtype::F32, 4),
        // 序列表(P-6b):cu|coff|idx 三件同块(字节口径 U32x1,同 marlin ws 记法)
        Param::t_in("tables", Dtype::U32, 1),
        Param::t_out("out", Dtype::F16, 3),
        Param::sz("t"),
        Param::sz("slot"),
        Param::sz("nv"),
        Param::sz("nk"),
        Param::sz("kd"),
        // kd^-0.5 位型走私(显式 token)
        Param::s("scale_bits", ScalarTy::F32Bits),
        // 块数 nt = ceil(t_real / BT)(P-6b;网格与 scratch 定尺都吃它)
        Param::sz("nt"),
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

pub static GDN_CHUNKED_FWD: KernelMeta = meta_of::<GdnChunkedDeltaRuleFwd>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[&GDN_CHUNKED_FWD];

#[cfg(test)]
mod tests {
    use crate::family::gdn_chunked::GDN_CHUNKED_FWD as ENTRY;
    use super::*;
    use crate::spec::meta_of;

    /// 与权威串(`client::gdn_chunked::SIG`)逐字一致 —— **P-6b 后 +1 表块(T)
    /// +1 标量(nt)**:声明序 = q,k,v,beta,gate,state,tables | out | t,slot,nv,nk,kd,scale_bits,nt
    #[test]
    fn chunked_slot_order_matches_legacy_sig() {
        assert_eq!(
            meta_of::<GdnChunkedDeltaRuleFwd>().args_str(),
            "T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz"
        );
    }

    // ── slot 跨参门(2026-10-15 审查补)的正/负例 ──────────────────────
    const GEOM: crate::bootcheck::Geom = crate::bootcheck::Geom {
        grid: (1, 1, 1),
        block: (128, 1, 1),
        smem: 0,
        out_elems: 128,
    };
    /// 状态块 id=6;`slots` 槽数;stride = nv*kd*kd*4 = 32*128*128*4 = 2MiB
    const STRIDE: usize = 32 * 128 * 128 * 4;

    /// 测试参数:事实按 PARAMS 声明填(q/k/v F16x3,beta/gate F16x2,state F32x4,
    /// **tables U32x1**,out F16x3);块 id 1..8(state 恒 id=6 —— slot 门按它查账长)
    fn args_with_slot(slot: u64) -> Vec<Arg> {
        let facts = [
            (Dtype::F16, 3),
            (Dtype::F16, 3),
            (Dtype::F16, 3),
            (Dtype::F16, 2),
            (Dtype::F16, 2),
            (Dtype::F32, 4),
            (Dtype::U32, 1),
            (Dtype::F16, 3),
        ];
        let mut v: Vec<Arg> = facts
            .iter()
            .enumerate()
            .map(|(i, (d, r))| Arg::block(i as u64 + 1, *d, *r))
            .collect();
        // 标量:t, slot, nv, nk, kd, scale_bits, nt
        for s in [512u64, slot, 32, 16, 128, 0x3f00_0000, 8] {
            v.push(Arg::U64(s));
        }
        v
    }

    /// 正例:slot 落在状态块内(含最后一格边界 `(slot+1)*stride == 账长`)⇒ 过
    #[test]
    fn slot_check_accepts_in_range() {
        let probe = |id: u64| if id == 6 { Some(8 * STRIDE) } else { Some(64) };
        for slot in [0u64, 7] {
            let args = args_with_slot(slot);
            let c = CheckCtx { op: ENTRY, args: &args, geom: &GEOM, len: &probe };
            let r = GdnChunkedDeltaRuleFwd::check(&c);
            assert!(r.is_ok(), "slot={slot} 应过:{:?}", r.unwrap_err());
        }
    }

    /// 负例:slot 越界(读写别人的会话格/邻块)⇒ 结构化拒,消息带 slot/stride/账长
    #[test]
    fn slot_check_rejects_out_of_range() {
        let probe = |id: u64| if id == 6 { Some(8 * STRIDE) } else { Some(64) };
        let args = args_with_slot(8);
        let c = CheckCtx { op: ENTRY, args: &args, geom: &GEOM, len: &probe };
        let e = GdnChunkedDeltaRuleFwd::check(&c).unwrap_err().to_string();
        assert!(e.contains("slot") && e.contains("stride"), "{e}");
    }

    /// 负例:状态块不在账本(查不到账长)⇒ 拒(不许静默跳过)
    #[test]
    fn slot_check_rejects_unknown_state_block() {
        let probe = |id: u64| if id == 6 { None } else { Some(64) };
        let args = args_with_slot(0);
        let c = CheckCtx { op: ENTRY, args: &args, geom: &GEOM, len: &probe };
        let e = GdnChunkedDeltaRuleFwd::check(&c).unwrap_err().to_string();
        assert!(e.contains("state"), "{e}");
    }
}
