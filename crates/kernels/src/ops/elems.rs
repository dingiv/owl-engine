//! 逐元素 / 视图 / 采样 / 铸边界 生产算子元数据。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};

/// `owl_narrow_strided_f16` —— 窄切物化(逐 token 视图;`outer/src_dim/start/out_dim` 四标量)
pub struct NarrowStridedF16;

impl KernelSpec for NarrowStridedF16 {
    const ENTRY: &'static str = "owl_narrow_strided_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/attention.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("src", Dtype::F16, 2),
        Param::sz("outer"),
        Param::sz("src_dim"),
        Param::sz("start"),
        Param::sz("out_dim"),
        Param::t_out("dst", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_gemv_dual_f16` —— 双权 GEMV(b/a 投影单发)
pub struct GemvDualF16;

impl KernelSpec for GemvDualF16 {
    const ENTRY: &'static str = "owl_gemv_dual_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/gemv_dual_f16.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("wb", Dtype::F16, 2),
        Param::t_in("wa", Dtype::F16, 2),
        Param::t_in("x", Dtype::F16, 2),
        Param::i32("rows_b"),
        Param::i32("rows_a"),
        Param::i32("cols"),
        Param::t_out("y", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_add_f16` —— 逐元素加(宏实例化;fc 部分和累加)
pub struct AddF16;

impl KernelSpec for AddF16 {
    const ENTRY: &'static str = "owl_add_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/ops_pair.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F16, 1),
        Param::t_in("b", Dtype::F16, 1),
        Param::t_out("out", Dtype::F16, 1),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_add_bf16` —— 同上(BF16 域;草稿 fc 累加)
pub struct AddBf16;

impl KernelSpec for AddBf16 {
    const ENTRY: &'static str = "owl_add_bf16";
    const SOURCE: &'static str = include_str!("../../cu/owl/ops_pair.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::BF16, 1),
        Param::t_in("b", Dtype::BF16, 1),
        Param::t_out("out", Dtype::BF16, 1),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_sigmoid_f16` —— 逐元素 sigmoid
pub struct SigmoidF16;

impl KernelSpec for SigmoidF16 {
    const ENTRY: &'static str = "owl_sigmoid_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/ops_pair.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 1),
        Param::t_out("out", Dtype::F16, 1),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_sigmoid_gate_mul_f16` —— attention 门融合(gate 先 sigmoid 再乘 x)
pub struct SigmoidGateMulF16;

impl KernelSpec for SigmoidGateMulF16 {
    const ENTRY: &'static str = "owl_sigmoid_gate_mul_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/sigmoid_gate_mul_f16.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("gate", Dtype::F16, 1),
        Param::t_in("x", Dtype::F16, 1),
        Param::sz("n"),
        Param::t_out("out", Dtype::F16, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_cast_f16_bf16` —— 铸边界(f16 → bf16;入草稿)
pub struct CastF16Bf16;

impl KernelSpec for CastF16Bf16 {
    const ENTRY: &'static str = "owl_cast_f16_bf16";
    const SOURCE: &'static str = include_str!("../../cu/owl/cast.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 1),
        Param::i32("n"),
        Param::t_out("out", Dtype::BF16, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_cast_bf16_f16` —— 铸边界(bf16 → f16;出草稿)
pub struct CastBf16F16;

impl KernelSpec for CastBf16F16 {
    const ENTRY: &'static str = "owl_cast_bf16_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/cast.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 1),
        Param::i32("n"),
        Param::t_out("out", Dtype::F16, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_concat_rows_f16` —— 8 路行拼接(arity 8 = 固定)
pub struct ConcatRowsF16;

impl KernelSpec for ConcatRowsF16 {
    const ENTRY: &'static str = "owl_concat_rows_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/concat.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("i0", Dtype::F16, 2),
        Param::t_in("i1", Dtype::F16, 2),
        Param::t_in("i2", Dtype::F16, 2),
        Param::t_in("i3", Dtype::F16, 2),
        Param::t_in("i4", Dtype::F16, 2),
        Param::t_in("i5", Dtype::F16, 2),
        Param::t_in("i6", Dtype::F16, 2),
        Param::t_in("i7", Dtype::F16, 2),
        Param::sz("n"),
        Param::sz("r"),
        Param::sz("d"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_concat_rows_f32` —— 同上(f32;首轮拼接)
pub struct ConcatRowsF32;

impl KernelSpec for ConcatRowsF32 {
    const ENTRY: &'static str = "owl_concat_rows_f32";
    const SOURCE: &'static str = include_str!("../../cu/owl/concat.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("i0", Dtype::F32, 2),
        Param::t_in("i1", Dtype::F32, 2),
        Param::t_in("i2", Dtype::F32, 2),
        Param::t_in("i3", Dtype::F32, 2),
        Param::t_in("i4", Dtype::F32, 2),
        Param::t_in("i5", Dtype::F32, 2),
        Param::t_in("i6", Dtype::F32, 2),
        Param::t_in("i7", Dtype::F32, 2),
        Param::sz("n"),
        Param::sz("r"),
        Param::sz("d"),
        Param::t_out("out", Dtype::F32, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_embed_f16` —— embedding 查表(ids 为 f32 张量:设备侧索引)
pub struct EmbedF16;

impl KernelSpec for EmbedF16 {
    const ENTRY: &'static str = "owl_embed_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/embed_f32.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::F16, 2),
        Param::t_in("ids", Dtype::F32, 1),
        Param::sz("d_dim"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_argmax_f32idx_f16` —— 设备侧贪心采样(argmax;输出 f32 承载索引)
pub struct ArgmaxF32IdxF16;

impl KernelSpec for ArgmaxF32IdxF16 {
    const ENTRY: &'static str = "owl_argmax_f32idx_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/argmax_f16.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 1),
        Param::i32("n"),
        Param::i32("offset"),
        Param::t_out("out", Dtype::F32, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_topk16_f16` —— 逐行 top-16(DFlash2 草稿;单缓冲 [rows,32])
pub struct Topk16F16;

impl KernelSpec for Topk16F16 {
    const ENTRY: &'static str = "owl_topk16_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/dflash2.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::i32("vocab"),
        Param::t_out("out", Dtype::F32, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static NARROW_STRIDED_F16: KernelMeta = meta_of::<NarrowStridedF16>();
pub static GEMV_DUAL_F16: KernelMeta = meta_of::<GemvDualF16>();
pub static ADD_F16: KernelMeta = meta_of::<AddF16>();
pub static ADD_BF16: KernelMeta = meta_of::<AddBf16>();
pub static SIGMOID_F16: KernelMeta = meta_of::<SigmoidF16>();
pub static SIGMOID_GATE_MUL_F16: KernelMeta = meta_of::<SigmoidGateMulF16>();
pub static CAST_F16_BF16: KernelMeta = meta_of::<CastF16Bf16>();
pub static CAST_BF16_F16: KernelMeta = meta_of::<CastBf16F16>();
pub static CONCAT_ROWS_F16: KernelMeta = meta_of::<ConcatRowsF16>();
pub static CONCAT_ROWS_F32: KernelMeta = meta_of::<ConcatRowsF32>();
pub static EMBED_F16: KernelMeta = meta_of::<EmbedF16>();
pub static ARGMAX_F32IDX_F16: KernelMeta = meta_of::<ArgmaxF32IdxF16>();
pub static TOPK16_F16: KernelMeta = meta_of::<Topk16F16>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[
    &NARROW_STRIDED_F16,
    &GEMV_DUAL_F16,
    &ADD_F16,
    &ADD_BF16,
    &SIGMOID_F16,
    &SIGMOID_GATE_MUL_F16,
    &CAST_F16_BF16,
    &CAST_BF16_F16,
    &CONCAT_ROWS_F16,
    &CONCAT_ROWS_F32,
    &EMBED_F16,
    &ARGMAX_F32IDX_F16,
    &TOPK16_F16,
];
