//! FlashInfer prefill 元数据(foreign:预编 `.a` + adapter;无 `.cu` 源)。
//!
//! 生产档 **FI 恒启用**(tokens ≥ 128 接管 prefill),是主池的**主读者**
//! 之一 —— 池访问/布局声明必须与 K0 写者、chunked 读者对齐(同一份
//! `ShapeFamily`),否则"写者/读者布局不一致"这类案(E14)没有机器面。
//!
//! 槽序权威历史 = `client::fi_sig::SIG`;现在唯一真源 = 本表。

use owl_macros::kernel_tags;
use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param, ShapeFamily};

/// `flashinfer_prefill_paged_fp8kv` —— paged prefill(fp8 e4m3 影子池)。
pub struct FiPrefillFp8kv;

#[kernel_tags("dom.flashinfer", "role.attn", "phase.prefill", "kv.main.fp8")]
impl KernelSpec for FiPrefillFp8kv {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::flashinfer::PREFILL_FI);
    const ENTRY: &'static str = crate::family::flashinfer::PREFILL_FI_FP8KV;
    /// foreign(预编 `.a`):无 `.cu` 源
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        // fp8 主池句柄:引擎叶子口径 = U32 扁平字节账(与 XQA 同律;
        // 池的 e4m3 字节语义由 `POOL` 的 `Dtype::F8E4M3` 承担)
        Param::t_in("k_cache", Dtype::U32, 1),
        Param::t_in("v_cache", Dtype::U32, 1),
        // FI plan 的 host 侧索引表(设备块承载;dtype 由 adapter 定义)
        Param::t_in("q_cu", Dtype::U32, 1),
        Param::t_in("indices", Dtype::U32, 1),
        Param::t_in("indptr", Dtype::U32, 1),
        Param::t_in("last_len", Dtype::U32, 1),
        // 依赖边(K0 写池先于 FI 读池;FI 不解引用,仅保序;假输出块 F16 [1])
        Param::t_in("wr", Dtype::F16, 1),
        Param::t_out("out", Dtype::F16, 2),
        Param::sz("total_rows"),
        Param::sz("ctx_total"),
        Param::sz("t"),
        Param::sz("hq"),
        Param::sz("hkv"),
        Param::sz("hd"),
        Param::sz("page"),
        // f32 位型走私(显式 token):sm_scale
        Param::s("sm_scale_bits", crate::spec::ScalarTy::F32Bits),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const POOL: &'static [crate::spec::PoolAccess] =
        &[crate::spec::PoolAccess::read(
            crate::spec::pools::KV_MAIN,
            crate::spec::LayoutTag::Knhd,
            Dtype::F8E4M3,
        )];
    const SHAPE: ShapeFamily = ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3),
        head_dim: &[256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}

/// `flashinfer_prefill_paged_f16` —— paged prefill(**f16 主池**;fp8kv 孪生)。
///
/// 槽序与 fp8kv 版逐位同形(适配器同一 plan;只有池元素宽不同)——
/// 两者共享 `client::fi_sig::SIG`,故本表 PARAMS 除池 dtype 外逐字一致。
pub struct FiPrefillF16;

#[kernel_tags("dom.flashinfer", "role.attn", "phase.prefill", "kv.main.f16")]
impl KernelSpec for FiPrefillF16 {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::flashinfer::PREFILL_FI);
    const ENTRY: &'static str = crate::family::flashinfer::PREFILL_FI;
    /// foreign(预编 `.a`):无 `.cu` 源
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        // f16 主池:kNHD 分页视图(引擎叶子 = [nb, hkv, hd/x, page, x] / v [nb,hkv,hd,page])
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("q_cu", Dtype::U32, 1),
        Param::t_in("indices", Dtype::U32, 1),
        Param::t_in("indptr", Dtype::U32, 1),
        Param::t_in("last_len", Dtype::U32, 1),
        Param::t_in("wr", Dtype::F16, 1),
        Param::t_out("out", Dtype::F16, 2),
        Param::sz("total_rows"),
        Param::sz("ctx_total"),
        Param::sz("t"),
        Param::sz("hq"),
        Param::sz("hkv"),
        Param::sz("hd"),
        Param::sz("page"),
        Param::s("sm_scale_bits", crate::spec::ScalarTy::F32Bits),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const POOL: &'static [crate::spec::PoolAccess] =
        &[crate::spec::PoolAccess::read(
            crate::spec::pools::KV_MAIN,
            crate::spec::LayoutTag::Knhd,
            Dtype::F16,
        )];
    const SHAPE: ShapeFamily = ShapeFamily {
        kv_dtype: Some(Dtype::F16),
        head_dim: &[128, 256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}

pub static PREFILL_FI_F16: KernelMeta = meta_of::<FiPrefillF16>();

pub static PREFILL_FI_FP8KV: KernelMeta = meta_of::<FiPrefillFp8kv>();

// ============================================================================
// 视觉塔注意力:ragged **非因果** prefill(2026-10-15)
//   —— 复用 vendored FlashInfer FA2 家族核(`MaskMode::kNone`);
//      槽位与 native `vis::AttnFullF16` **逐字同构**(driver 按 dtype/图数选臂)。
// ============================================================================

/// `flashinfer_attn_ragged_f16` —— 非因果 varlen 自注意力(单图;dense K/V;hd ∈ {64})
pub struct FiAttnRaggedF16;

#[kernel_tags("dom.flashinfer", "role.attn", "phase.decode")]
impl KernelSpec for FiAttnRaggedF16 {
    /// 家族 id = **FI 族**(= `FiRuntime::ID`,其值就是 paged f16 的入口名 ——
    /// 族 id 与"首个入口名"同源的既定语汇,勿把它当算子名读)
    const FAMILY: crate::contract::OpId =
        crate::contract::OpId(crate::family::flashinfer::PREFILL_FI);
    const ENTRY: &'static str = crate::family::flashinfer::ATTN_RAGGED_F16;
    /// foreign(预编 `.a` + 适配器胶水):无 `.cu` 资产
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 3),
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        Param::t_in("cu_seqlens", Dtype::F32, 1),
        Param::t_out("out", Dtype::F16, 3),
        Param::sz("rows"),
        Param::sz("heads"),
        Param::sz("head_dim"),
        Param::i32("n_seq"),
        Param::f32("scale"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static FI_ATTN_RAGGED_F16: KernelMeta = meta_of::<FiAttnRaggedF16>();

/// **vLLM 原版 FA2 varlen forward**（2026-10-23；hd=72→kHeadDim 96）。
/// 与 `FiAttnRaggedF16` **同槽位**（q/k/v/cu/out/rows/heads/head_dim/n_seq/scale）；
/// server 按 kernel 名分流到 `owl_fa2_varlen_fwd_f16`。
pub struct Fa2VarlenF16;
#[kernel_tags("dom.flashinfer", "role.attn", "phase.prefill")]
impl KernelSpec for Fa2VarlenF16 {
    const FAMILY: crate::contract::OpId =
        crate::contract::OpId(crate::family::flashinfer::PREFILL_FI);
    const ENTRY: &'static str = crate::family::flashinfer::FA2_VARLEN_F16;
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 3),
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        Param::t_in("cu_seqlens", Dtype::F32, 1),
        Param::t_out("out", Dtype::F16, 3),
        Param::sz("rows"),
        Param::sz("heads"),
        Param::sz("head_dim"),
        Param::i32("n_seq"),
        Param::f32("scale"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
pub static FA2_VARLEN_F16: KernelMeta = meta_of::<Fa2VarlenF16>();



#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::meta_of;

    /// f16 孪生与 fp8kv 版**逐位同形**(同一 plan,只差池元素宽)
    #[test]
    fn fi_f16_mirrors_fp8kv_slots() {
        let a = meta_of::<FiPrefillFp8kv>();
        let b = meta_of::<FiPrefillF16>();
        assert_eq!(a.args_str(), b.args_str(), "FI f16 与 fp8kv 槽序必须同形");
    }

    /// 与旧权威串(`client::fi_sig::SIG`)逐字一致
    #[test]
    fn fi_slot_order_matches_legacy_sig() {
        assert_eq!(
            meta_of::<FiPrefillFp8kv>().args_str(),
            "T,T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz,sz"
        );
    }
}
