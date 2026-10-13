//! 融合核域生产算子(norm/rope/池插三发合一)。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, pools, CaptureEvidence, Determinism, KernelMeta, KernelSpec, LayoutTag, Param, PoolAccess};

/// `owl_qknorm_rope_kv_insert_f16_fp8kv_knhd` —— qk-norm + rotate-half rope
/// + KV 散写(主池 e4m3/kNHD)三发合一。
///
/// **E10 案现场**:池 dtype 改造时本核的 fp8 臂曾缺失 ⇒ f16 写落在 1B/elem
/// 池的 2× 偏移 = 槽区永不落笔全零。本表把"写哪两块池、什么 dtype"写进
/// 声明(装配期写集穷举审计,P4)。
pub struct QknormRopeKvInsertF16Fp8Knhd;

impl KernelSpec for QknormRopeKvInsertF16Fp8Knhd {
    const ENTRY: &'static str = "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd";
    const SRC: &'static str = "cu/fused/qknorm_insert.cu";
    const PARAMS: &'static [Param] = &[
        // q_raw 是 per-head [value|gate] 半段拼接(strided 读)
        Param::t_in("q_raw", Dtype::F16, 2),
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        // 主池 fp8(e4m3,1B/elem)
        Param::t_in("key_cache", Dtype::U8, 4),
        Param::t_in("value_cache", Dtype::U8, 4),
        Param::t_in("slots", Dtype::F32, 1),
        Param::t_in("q_w", Dtype::F16, 1),
        Param::t_in("k_w", Dtype::F16, 1),
        Param::t_in("cos_t", Dtype::F16, 2),
        Param::t_in("sin_t", Dtype::F16, 2),
        Param::t_in("pos", Dtype::F32, 1),
        Param::f32("eps"),
        Param::i32("hkv"),
        Param::i32("half"),
        Param::i32("page"),
        Param::i32("w_off"),
        Param::t_out("q_out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_MAIN, LayoutTag::Knhd, Dtype::U8)];
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> =
        Some(&crate::ops::oracles::qknorm_insert::QknormInsertOracle);
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8),
        head_dim: &[128, 256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}

/// `owl_qknorm_rope_kv_insert_f16_fp8kv` —— **classic 寻址**变体(与 kNHD
/// 变体同 ABI,只有池寻址不同)。classic 池档的融合插池写者。
pub struct QknormRopeKvInsertF16Fp8;

impl KernelSpec for QknormRopeKvInsertF16Fp8 {
    const ENTRY: &'static str = "owl_qknorm_rope_kv_insert_f16_fp8kv";
    const SRC: &'static str = "cu/fused/qknorm_insert.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q_raw", Dtype::F16, 2),
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        Param::t_in("key_cache", Dtype::U8, 4),
        Param::t_in("value_cache", Dtype::U8, 4),
        Param::t_in("slots", Dtype::F32, 1),
        Param::t_in("q_w", Dtype::F16, 1),
        Param::t_in("k_w", Dtype::F16, 1),
        Param::t_in("cos_t", Dtype::F16, 2),
        Param::t_in("sin_t", Dtype::F16, 2),
        Param::t_in("pos", Dtype::F32, 1),
        Param::f32("eps"),
        Param::i32("hkv"),
        Param::i32("half"),
        Param::i32("page"),
        Param::i32("w_off"),
        Param::t_out("q_out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[
        PoolAccess::write(pools::KV_MAIN, LayoutTag::Classic, Dtype::U8),
        PoolAccess::write(pools::KV_DRAFT, LayoutTag::Classic, Dtype::U8),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8),
        head_dim: &[128, 256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}


pub static QKNORM_ROPE_KV_INSERT_F16_FP8_KNHD: KernelMeta = meta_of::<QknormRopeKvInsertF16Fp8Knhd>();
pub static QKNORM_ROPE_KV_INSERT_F16_FP8: KernelMeta = meta_of::<QknormRopeKvInsertF16Fp8>();

// ----------------------------------------------------------------------------
// R1 补表(2026-10-13):f16 主池两支 —— 声明从登记表派生,签名由 `op_tables` 门互证
// ----------------------------------------------------------------------------

/// `owl_qknorm_rope_kv_insert_f16` —— 经典池 f16 插池(classic 布局,主池+草稿池双写)
pub struct QknormRopeKvInsertF16;

impl KernelSpec for QknormRopeKvInsertF16 {
    const ENTRY: &'static str = "owl_qknorm_rope_kv_insert_f16";
    const SRC: &'static str = "cu/fused/qknorm_insert.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("q_raw", Dtype::F16, 2),
        Param::t_in("kw", Dtype::F16, 2),
        Param::t_in("beta", Dtype::F16, 2),
        Param::t_in("cos", Dtype::F16, 2),
        Param::t_in("sin", Dtype::F16, 2),
        Param::t_in("k", Dtype::F16, 2),
        Param::t_in("v", Dtype::F16, 2),
        Param::t_in("kc", Dtype::F16, 3),
        Param::t_in("vc", Dtype::F16, 3),
        Param::t_in("slots", Dtype::F32, 1),
        Param::f32("eps"),
        Param::i32("tokens"),
        Param::i32("heads"),
        Param::i32("head_dim"),
        Param::i32("w_off"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[
        PoolAccess::write(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16),
        PoolAccess::write(pools::KV_DRAFT, LayoutTag::Classic, Dtype::F16),
    ];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16),
        head_dim: &[128],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_qknorm_rope_kv_insert_f16_knhd` —— kNHD 单布局 f16 插池(主池 f16)
pub struct QknormRopeKvInsertF16Knhd;

impl KernelSpec for QknormRopeKvInsertF16Knhd {
    const ENTRY: &'static str = "owl_qknorm_rope_kv_insert_f16_knhd";
    const SRC: &'static str = "cu/fused/qknorm_insert.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("q_raw", Dtype::F16, 2),
        Param::t_in("kw", Dtype::F16, 2),
        Param::t_in("beta", Dtype::F16, 2),
        Param::t_in("cos", Dtype::F16, 2),
        Param::t_in("sin", Dtype::F16, 2),
        Param::t_in("k", Dtype::F16, 2),
        Param::t_in("v", Dtype::F16, 2),
        Param::t_in("kc", Dtype::F16, 3),
        Param::t_in("vc", Dtype::F16, 3),
        Param::t_in("slots", Dtype::F32, 1),
        Param::f32("eps"),
        Param::i32("tokens"),
        Param::i32("heads"),
        Param::i32("head_dim"),
        Param::i32("w_off"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] =
        &[PoolAccess::write(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static QKNORM_ROPE_KV_INSERT_F16: KernelMeta = meta_of::<QknormRopeKvInsertF16>();
pub static QKNORM_ROPE_KV_INSERT_F16_KNHD: KernelMeta = meta_of::<QknormRopeKvInsertF16Knhd>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[
    &QKNORM_ROPE_KV_INSERT_F16_FP8_KNHD,
    &QKNORM_ROPE_KV_INSERT_F16_FP8,
    &QKNORM_ROPE_KV_INSERT_F16,
    &QKNORM_ROPE_KV_INSERT_F16_KNHD,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::meta_of;

    #[test]
    fn qknorm_insert_slot_order_derives_from_params() {
        assert_eq!(
            meta_of::<QknormRopeKvInsertF16Fp8Knhd>().args_str(),
            "T,T,T,T,T,T,T,T,T,T,T,f32,i32,i32,i32,i32,O"
        );
    }

    /// 输出槽位置由 `ParamRole::Out` 决定,不靠"末位约定"猜
    #[test]
    fn out_pos_comes_from_role() {
        assert_eq!(meta_of::<QknormRopeKvInsertF16Fp8Knhd>().out_pos(), Some(16));
    }
}
