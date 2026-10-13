//! FlashInfer prefill 元数据(foreign:预编 `.a` + adapter;无 `.cu` 源)。
//!
//! 生产档 **FI 恒启用**(tokens ≥ 128 接管 prefill),是主池的**主读者**
//! 之一 —— 池访问/布局声明必须与 K0 写者、chunked 读者对齐(同一份
//! `ShapeFamily`),否则"写者/读者布局不一致"这类案(E14)没有机器面。
//!
//! 槽序权威历史 = `client::fi_sig::SIG`;现在唯一真源 = 本表。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param, ShapeFamily};

/// `flashinfer_prefill_paged_fp8kv` —— paged prefill(fp8 e4m3 影子池)。
pub struct FiPrefillFp8kv;

impl KernelSpec for FiPrefillFp8kv {
    const ENTRY: &'static str = crate::contract::names::PREFILL_FI_FP8KV;
    /// foreign(预编 `.a`):无 `.cu` 源
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        // fp8 影子池(adapter 内部布局;1B/elem)
        Param::t_in("k_cache", Dtype::U8, 4),
        Param::t_in("v_cache", Dtype::U8, 4),
        // FI plan 的 host 侧索引表(设备块承载;dtype 由 adapter 定义)
        Param::t_in("q_cu", Dtype::U32, 1),
        Param::t_in("indices", Dtype::U32, 1),
        Param::t_in("indptr", Dtype::U32, 1),
        Param::t_in("last_len", Dtype::U32, 1),
        // 依赖边(K0 写池先于 FI 读池;FI 不解引用,仅保序)
        Param::t_in("wr", Dtype::U8, 1),
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
            Dtype::U8,
        )];
    const SHAPE: ShapeFamily = ShapeFamily {
        kv_dtype: Some(Dtype::U8),
        head_dim: &[256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}


pub static PREFILL_FI_FP8KV: KernelMeta = meta_of::<FiPrefillFp8kv>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[&PREFILL_FI_FP8KV];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::meta_of;

    /// 与旧权威串(`client::fi_sig::SIG`)逐字一致
    #[test]
    fn fi_slot_order_matches_legacy_sig() {
        assert_eq!(
            meta_of::<FiPrefillFp8kv>().args_str(),
            "T,T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz,sz"
        );
    }
}
