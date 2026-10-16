//! XQA 双臂元数据(foreign:预编 `.a` + host 发射函数;**无 .cu 源**)。
//!
//! 槽序契约曾三处各写一份(`family/xqa.rs` 的 `*_SIG` 常量、models 的
//! `foreign_sig`、client face 的 parse 断言)—— 现在**唯一真源 = 本表**,
//! 由 `PARAMS` 派生;parse 断言与 driver 拾取都从 [`KernelMeta`] 取。
//!
//! 跨参耦合(表表达不了的那类)在 `check()` 里覆写:`mask` 的元素数必须
//! 覆盖 `q_seq_len`(SPEC_DEC 掩码按 `ceil(q/32)` 行组织)。

use crate::contract::{Dtype, Law, OpError};
use crate::bootcheck::{CheckCtx};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param, RangeRule, ShapeFamily};
use crate::family::xqa;

/// 双臂共用的 20 槽参数表(两份实例化签名逐字同型,故共用一张表)。
///
/// `[q, k_cache, v_cache, page_table, seq_lens, mask, semaphores, scratch,
///   wr, out, q_seq_len, hkv, hd, page, max_seq_len, q_scale_bits, sm_count,
///   stride_page, stride_token, stride_head]`
const XQA_PARAMS: &[Param] = &[
    Param::t_in("q", Dtype::F16, 2),
    // fp8 主池句柄:引擎叶子口径 = U32 扁平字节账(`bytes/4` 字;2026-10-15
    // L3 事实面对齐 —— 池的 e4m3 字节语义由 `POOL` 的 `Dtype::F8E4M3` 承担)。
    // 旧声明 `U8 rank4` 描述的是核内 kNHD 字节视图,与喂进来的句柄不是同一层。
    Param::t_in("k_cache", Dtype::U32, 1),
    Param::t_in("v_cache", Dtype::U32, 1),
    // 页表/序列长/掩码/信号量/scratch 都是 u32/i32 承载块(位型见 .cu)
    // 页表/序列长/信号量/掩码 = **图输入槽**(每步由引擎写零/写值):
    // 写入面是 F32(数值口径),核按 u32 读同一 4 字节 —— 位型走私,
    // 与标量槽 `q_scale_bits` 同律(2026-10-15 事实面对齐:声明 = 喂进来的视图)
    Param::t_in("page_table", Dtype::F32, 2),
    Param::t_in("seq_lens", Dtype::F32, 2),
    // mask / wr:两臂喂的阶数不同(decode rank1 / verify rank2;wr 是纯依赖边)
    // ⇒ 显式 `0` = 阶数无关(两臂共用本表,不拆表)
    Param::t_in("mask", Dtype::F32, 0),
    Param::t_in("semaphores", Dtype::F32, 2),
    // 工作区:引擎以 F16 口径分配/注解(`(scr_mb<<20)/2` 元素)
    Param::t_in("scratch", Dtype::F16, 1),
    // 依赖边(K0 写池先于 XQA 读池;XQA 不解引用,仅保序)
    Param::t_in("wr", Dtype::F16, 0),
    Param::t_out("out", Dtype::F16, 2),
    // ---- 10 个 sz 标量 ----
    Param::sz("q_seq_len").range(RangeRule { min: Some(1.0), max: None, set: None }),
    Param::sz("hkv").range(RangeRule { min: Some(1.0), max: None, set: None }),
    Param::sz("hd").range(RangeRule { min: None, max: None, set: Some(&[128, 256]) }),
    Param::sz("page").range(RangeRule { min: None, max: None, set: Some(&[16, 32, 64, 128]) }),
    Param::sz("max_seq_len").range(RangeRule { min: Some(1.0), max: None, set: None }),
    // f32 位型走私(显式 token):= attn_scale·sqrt(hd)
    Param::s("q_scale_bits", crate::spec::ScalarTy::F32Bits),
    Param::sz("sm_count").range(RangeRule { min: Some(1.0), max: None, set: None }),
    Param::sz("stride_page"),
    Param::sz("stride_token"),
    Param::sz("stride_head"),
];

/// kNHD fp8 池 + hd256 + GQA 6:1 的形状族(两臂同)
const XQA_SHAPE: ShapeFamily = ShapeFamily {
    kv_dtype: Some(Dtype::F8E4M3),
    head_dim: &[256],
    page: &[16, 32, 64, 128],
    grp: Some(6),
    // 实例化行数上界:`M_TILESIZE=32`(verify 臂)/ 16(decode 臂)——
    // q 超过上界 = 走错实例化(§二十八 幽灵宏名案的反面:声明化)
    tile: Some(32),
};

/// 跨参耦合检查(表表达不了):`mask` 元素数 ≥ `ceil(q_seq_len/32)`,
/// 且 `q_seq_len ≤ max_seq_len`。
fn check_xqa(ctx: &CheckCtx<'_>) -> Result<(), OpError> {
    crate::spec::check_from_params("xqa", XQA_PARAMS, ctx)?;
    let sc = |i: usize| -> Option<u64> {
        match ctx.args.get(i) {
            Some(crate::contract::Arg::U64(v)) => Some(*v),
            _ => None,
        }
    };
    let (q_seq_len, max_seq) = (sc(10).unwrap_or(0), sc(14).unwrap_or(0));
    if q_seq_len > max_seq {
        return Err(OpError::Contract {
            op: ctx.op.to_string(),
            field: "q_seq_len",
            expect: format!("≤ max_seq_len({max_seq})"),
            got: format!("{q_seq_len}"),
        });
    }
    if let Some(bytes) = ctx.arg_len(5) {
        // mask 是 u32 承载:每行 ceil(q/32) 个 4B 字
        let need_bytes = (q_seq_len as usize).div_ceil(32) * 4;
        if bytes < need_bytes {
            return Err(OpError::Contract {
                op: ctx.op.to_string(),
                field: "mask",
                expect: format!("字节数 ≥ ceil(q_seq_len/32)·4 = {need_bytes}"),
                got: format!("{bytes}B"),
            });
        }
    }
    Ok(())
}

/// `owl_xqa_verify` —— spec-verify 注意力(`SPEC_DEC=1`, `M_TILESIZE=32`)
pub struct XqaVerify;

impl KernelSpec for XqaVerify {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(xqa::VERIFY_XQA);
    const ENTRY: &'static str = xqa::VERIFY_XQA;
    /// foreign(预编 `.a`):无 `.cu` 源 —— 源码/符号在 vendor 归档,
    /// 装配门由族的 manifest/sha256 面承担
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = XQA_PARAMS;
    const SHAPE: ShapeFamily = XQA_SHAPE;
    const POOL: &'static [crate::spec::PoolAccess] =
        &[crate::spec::PoolAccess::read(
            crate::spec::pools::KV_MAIN,
            crate::spec::LayoutTag::Knhd,
            Dtype::F8E4M3,
        )];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    fn check(ctx: &CheckCtx<'_>) -> Result<(), OpError> {
        check_xqa(ctx)
    }
}

/// `owl_xqa_decode` —— 裸 decode 注意力(第二份实例化 `M_TILESIZE=16`,q=1)
pub struct XqaDecode;

impl KernelSpec for XqaDecode {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(xqa::DECODE_XQA);
    const ENTRY: &'static str = xqa::DECODE_XQA;
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = XQA_PARAMS;
    const SHAPE: ShapeFamily = XQA_SHAPE;
    const POOL: &'static [crate::spec::PoolAccess] =
        &[crate::spec::PoolAccess::read(
            crate::spec::pools::KV_MAIN,
            crate::spec::LayoutTag::Knhd,
            Dtype::F8E4M3,
        )];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    fn check(ctx: &CheckCtx<'_>) -> Result<(), OpError> {
        check_xqa(ctx)
    }
}


pub static XQA_VERIFY: KernelMeta = meta_of::<XqaVerify>();
pub static XQA_DECODE: KernelMeta = meta_of::<XqaDecode>();

/// `owl_xqa_prefill` —— **chunked prefill** 注意力(q=T 行因果块)。
///
/// 与 [`XqaVerify`] **同资产同槽序**(同一份 `M_TILESIZE=32` 实例化;
/// SPEC_DEC 单遍核在 q=T 时就是 chunked prefill 的因果块:行 i 看
/// `[0, ctx_base+i]`,掩码 `mask[i] = (1<<(i+1))-1` 逐字同式)。
/// 分开建表的两个理由:① 审计面能分辨臂(白名单/日志/性能相位);
/// ② **几何契约不同** —— 本臂强制单块(`launch_geometry_single_block`),
/// 该不变量住 server emit 一处(见 `server::xqa::XqaPrefillRuntime`)。
///
/// ⚠️ 生产主路径**不用它**(CUPTI 同请求 A/B:XQA 258.5ms vs FI 63.0ms = 慢 4.1×;
/// XQA 是 decode 形状核);它只接 **FI 的缺口**(`t < fi_min_tokens` 尾块 /
/// `ctx.fi` 缺席),那里 9.4× 于回落核。`dispatch.xqa_prefill` 缺省 false。
pub struct XqaPrefill;

impl KernelSpec for XqaPrefill {
    /// 执行家族(声明驱动路由;**声明面与执行面同一字面量** —— A18 律)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(xqa::PREFILL_XQA);
    const ENTRY: &'static str = xqa::PREFILL_XQA;
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = XQA_PARAMS;
    const SHAPE: ShapeFamily = ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3),
        head_dim: &[256],
        page: &[16, 32, 64, 128],
        grp: Some(6),
        tile: None,
    };
    const POOL: &'static [crate::spec::PoolAccess] =
        &[crate::spec::PoolAccess::read(
            crate::spec::pools::KV_MAIN,
            crate::spec::LayoutTag::Knhd,
            Dtype::F8E4M3,
        )];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    fn check(ctx: &CheckCtx<'_>) -> Result<(), OpError> {
        check_xqa(ctx)
    }
}

pub static XQA_PREFILL: KernelMeta = meta_of::<XqaPrefill>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[&XQA_VERIFY, &XQA_DECODE, &XQA_PREFILL];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::Arg;
    use crate::bootcheck::{Geom};
use crate::spec::{meta_of};

    const GEOM: Geom = Geom { grid: (1, 1, 1), block: (128, 1, 1), smem: 0, out_elems: 256 };

    /// 槽序与 `family::xqa::VERIFY_XQA_SIG`(历史权威串)**逐字一致** ——
    /// 表接管后旧常量仍作对拍锚,不允许悄悄漂移。
    #[test]
    fn xqa_slot_order_matches_legacy_sig() {
        let m = meta_of::<XqaVerify>();
        assert_eq!(m.args_str(), xqa::VERIFY_XQA_SIG);
        assert_eq!(meta_of::<XqaDecode>().args_str(), xqa::DECODE_XQA_SIG);
        // prefill 臂:同一张槽序表(逐字同型)⇒ 三臂互证
        assert_eq!(meta_of::<XqaPrefill>().args_str(), xqa::PREFILL_XQA_SIG);
        assert_eq!(meta_of::<XqaPrefill>().args_str(), m.args_str());
    }

    /// 测试参数:事实按 XQA_PARAMS 声明填(事实面门现在会逐槽比对)
    fn xqa_args(q_seq_len: u64) -> Vec<Arg> {
        let facts = [
            (Dtype::F16, 2), // q
            (Dtype::U32, 1), // k_cache
            (Dtype::U32, 1), // v_cache
            (Dtype::F32, 2), // page_table
            (Dtype::F32, 2), // seq_lens
            (Dtype::F32, 1), // mask(声明 rank=0 阶数无关;发送方仍须给真实阶数 ≥1)
            (Dtype::F32, 2), // semaphores
            (Dtype::F16, 1), // scratch
            (Dtype::F16, 1), // wr(依赖边)
        ];
        let mut v: Vec<Arg> = facts
            .iter()
            .enumerate()
            .map(|(i, (d, r))| Arg::block(i as u64 + 1, *d, *r))
            .collect();
        v.push(Arg::block(100, Dtype::F16, 2)); // out
        for s in [q_seq_len, 1, 256, 32, 4096, 1, 84, 1, 1, 1] {
            v.push(Arg::U64(s));
        }
        v
    }

    /// 正例:mask 元素数足够 ⇒ 过
    #[test]
    fn xqa_check_accepts_wellformed() {
        let probe = |id: u64| if id == 6 { Some(32usize) } else { Some(64) };
        let args = xqa_args(8);
        let c = CheckCtx { op: "xqa", args: &args, geom: &GEOM, len: &probe };
        assert!(meta_of::<XqaVerify>().check(&c).is_ok(), "{}", meta_of::<XqaVerify>().check(&c).unwrap_err());
    }

    /// 负例:mask 太小 ⇒ 跨参耦合报错(带算子名 + mask + 期望元素数)
    #[test]
    fn xqa_check_rejects_short_mask() {
        let probe = |id: u64| if id == 6 { Some(4usize) } else { Some(64) };
        let args = xqa_args(128);
        let c = CheckCtx { op: "xqa", args: &args, geom: &GEOM, len: &probe };
        let e = meta_of::<XqaVerify>().check(&c).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("mask") && s.contains("ceil(q_seq_len/32)"), "{s}");
    }

    /// 负例:hd 不在声明集 ⇒ 表驱动的集合检查拦下
    #[test]
    fn xqa_check_rejects_unknown_head_dim() {
        let probe = |_id: u64| Some(64usize);
        let mut args = xqa_args(8);
        args[12] = Arg::U64(192); // hd = 192(不在 {128,256})
        let c = CheckCtx { op: "xqa", args: &args, geom: &GEOM, len: &probe };
        let e = meta_of::<XqaVerify>().check(&c).unwrap_err();
        assert!(e.to_string().contains("hd"), "{e}");
    }
}
