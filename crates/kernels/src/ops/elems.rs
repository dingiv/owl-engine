//! 逐元素 / 视图 / 采样 / 铸边界 生产算子元数据。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};

/// `owl_narrow_strided_f16` —— 窄切物化(逐 token 视图;`outer/src_dim/start/out_dim` 四标量)
pub struct NarrowStridedF16;

impl KernelSpec for NarrowStridedF16 {
    const ENTRY: &'static str = "owl_narrow_strided_f16";
    const SRC: &'static str = "cu/elems/narrow_strided.cu";
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
    const SRC: &'static str = "cu/elems/gemv_dual_f16.cu";
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
    const SRC: &'static str = "cu/elems/add.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F16, 0),
        Param::t_in("b", Dtype::F16, 0),
        Param::t_out("out", Dtype::F16, 0),
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
    const SRC: &'static str = "cu/elems/add.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::BF16, 0),
        Param::t_in("b", Dtype::BF16, 0),
        Param::t_out("out", Dtype::BF16, 0),
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
    const SRC: &'static str = "cu/elems/sigmoid.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 0),
        Param::t_out("out", Dtype::F16, 0),
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
    const SRC: &'static str = "cu/elems/sigmoid_gate_mul_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("gate", Dtype::F16, 0),
        Param::t_in("x", Dtype::F16, 0),
        Param::sz("n"),
        Param::t_out("out", Dtype::F16, 0),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_cast_f16_bf16` —— 铸边界(f16 → bf16;入草稿)
pub struct CastF16Bf16;

impl KernelSpec for CastF16Bf16 {
    const ENTRY: &'static str = "owl_cast_f16_bf16";
    const SRC: &'static str = "cu/elems/cast.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 0),
        Param::i32("n"),
        Param::t_out("out", Dtype::BF16, 0),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_cast_bf16_f16` —— 铸边界(bf16 → f16;出草稿)
pub struct CastBf16F16;

impl KernelSpec for CastBf16F16 {
    const ENTRY: &'static str = "owl_cast_bf16_f16";
    const SRC: &'static str = "cu/elems/cast.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 0),
        Param::i32("n"),
        Param::t_out("out", Dtype::F16, 0),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// **首轮拼接的输入槽数**(`owl_concat_rows_{f16,f32}` = `i0..i{7}`;arity 固定)。
///
/// 8 = **核声明面的事实**(两张表都恰好 8 槽)–—— 调用方(engine 的“固定槽树”
/// 把 t 行 pad 到 8 槽)与装配期门**都取本常量**,不许各写一个字面量;
/// 改核时本处跟着改,`concat_rows_slots_lock` 拿两张表逐槽核。
pub const CONCAT_ROWS_SLOTS: usize = 8;

/// `owl_concat_rows_f16` —— 8 路行拼接(arity 8 = 固定)
pub struct ConcatRowsF16;

impl KernelSpec for ConcatRowsF16 {
    const ENTRY: &'static str = "owl_concat_rows_f16";
    const SRC: &'static str = "cu/elems/concat.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("i0", Dtype::F16, 0),
        Param::t_in("i1", Dtype::F16, 0),
        Param::t_in("i2", Dtype::F16, 0),
        Param::t_in("i3", Dtype::F16, 0),
        Param::t_in("i4", Dtype::F16, 0),
        Param::t_in("i5", Dtype::F16, 0),
        Param::t_in("i6", Dtype::F16, 0),
        Param::t_in("i7", Dtype::F16, 0),
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
    const SRC: &'static str = "cu/elems/concat.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("i0", Dtype::F32, 0),
        Param::t_in("i1", Dtype::F32, 0),
        Param::t_in("i2", Dtype::F32, 0),
        Param::t_in("i3", Dtype::F32, 0),
        Param::t_in("i4", Dtype::F32, 0),
        Param::t_in("i5", Dtype::F32, 0),
        Param::t_in("i6", Dtype::F32, 0),
        Param::t_in("i7", Dtype::F32, 0),
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
    const SRC: &'static str = "cu/elems/embed_f32.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::F16, 2),
        Param::t_in("ids", Dtype::F32, 0),
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
    const SRC: &'static str = "cu/elems/argmax_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 0),
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
    const SRC: &'static str = "cu/elems/dflash_topk.cu";
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

/// `owl_topk16_split_f16` —— split-K 段一(grid = rows×nparts):
/// 每块扫 vocab/nparts,uint4 向量载入 + 寄存器全展开插入 + warp 锦标赛归并
/// (2026-10-14 D2:旧单块核 grid=(rows,1,1) 只用 7/132 SM)
pub struct Topk16SplitF16;

impl KernelSpec for Topk16SplitF16 {
    const ENTRY: &'static str = "owl_topk16_split_f16";
    const SRC: &'static str = "cu/elems/dflash_topk.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::i32("vocab"),
        Param::i32("nparts"),
        Param::i32("rows"),
        // 单输出槽:值区 [rows*nparts*K] + 索引区(int32 位型)[rows*nparts*K]
        Param::t_out("tmp", Dtype::F32, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static TOPK16_SPLIT_F16: KernelMeta = meta_of::<Topk16SplitF16>();

/// `owl_topk16_merge_f16` —— split-K 段二(一行一 warp,零同步):
/// nparts×16 候选 → top-16,输出布局与卷 `owl_topk16_f16` 逐位一致
pub struct Topk16MergeF16;

impl KernelSpec for Topk16MergeF16 {
    const ENTRY: &'static str = "owl_topk16_merge_f16";
    const SRC: &'static str = "cu/elems/dflash_topk.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("tmp", Dtype::F32, 1),
        Param::i32("nparts"),
        Param::i32("rows"),
        Param::i32("vocab"),
        Param::t_out("out", Dtype::F32, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static TOPK16_MERGE_F16: KernelMeta = meta_of::<Topk16MergeF16>();


// ============================================================================
// `elems.rs` 域 R1 补表
// (R1 补表 2026-10-13:声明从登记表派生,签名由 `op_tables` 门逐参互证)
// ============================================================================


/// `owl_add_f32` —— add.cu(R1 补表)
pub struct AddF32;

impl KernelSpec for AddF32 {
    const ENTRY: &'static str = "owl_add_f32";
    const SRC: &'static str = "cu/elems/add.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F32, 0),
        Param::t_in("b", Dtype::F32, 0),
        Param::t_out("out", Dtype::F32, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_mul_f32` —— mul.cu(R1 补表)
pub struct MulF32;

impl KernelSpec for MulF32 {
    const ENTRY: &'static str = "owl_mul_f32";
    const SRC: &'static str = "cu/elems/mul.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F32, 0),
        Param::t_in("b", Dtype::F32, 0),
        Param::t_out("out", Dtype::F32, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_mul_f16` —— mul.cu(R1 补表)
pub struct MulF16;

impl KernelSpec for MulF16 {
    const ENTRY: &'static str = "owl_mul_f16";
    const SRC: &'static str = "cu/elems/mul.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F16, 0),
        Param::t_in("b", Dtype::F16, 0),
        Param::t_out("out", Dtype::F16, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_mul_bf16` —— mul.cu(R1 补表)
pub struct MulBf16;

impl KernelSpec for MulBf16 {
    const ENTRY: &'static str = "owl_mul_bf16";
    const SRC: &'static str = "cu/elems/mul.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::BF16, 0),
        Param::t_in("b", Dtype::BF16, 0),
        Param::t_out("out", Dtype::BF16, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_sigmoid_f32` —— sigmoid.cu(R1 补表)
pub struct SigmoidF32;

impl KernelSpec for SigmoidF32 {
    const ENTRY: &'static str = "owl_sigmoid_f32";
    const SRC: &'static str = "cu/elems/sigmoid.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 0),
        Param::t_out("out", Dtype::F32, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_silu_f32` —— silu.cu(R1 补表)
pub struct SiluF32;

impl KernelSpec for SiluF32 {
    const ENTRY: &'static str = "owl_silu_f32";
    const SRC: &'static str = "cu/elems/silu.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 0),
        Param::t_out("out", Dtype::F32, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_silu_f16` —— silu.cu(R1 补表)
pub struct SiluF16;

impl KernelSpec for SiluF16 {
    const ENTRY: &'static str = "owl_silu_f16";
    const SRC: &'static str = "cu/elems/silu.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 0),
        Param::t_out("out", Dtype::F16, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_matmul_f32` —— matmul.cu(R1 补表)
pub struct MatmulF32;

impl KernelSpec for MatmulF32 {
    const ENTRY: &'static str = "owl_matmul_f32";
    const SRC: &'static str = "cu/elems/matmul.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F32, 2),
        Param::t_in("b", Dtype::F32, 2),
        Param::t_out("out", Dtype::F32, 2),
        Param::i32("m"),
        Param::i32("k"),
        Param::i32("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_matmul_nt_f32` —— matmul.cu(R1 补表)
pub struct MatmulNtF32;

impl KernelSpec for MatmulNtF32 {
    const ENTRY: &'static str = "owl_matmul_nt_f32";
    const SRC: &'static str = "cu/elems/matmul.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F32, 2),
        Param::t_in("b", Dtype::F32, 2),
        Param::t_out("out", Dtype::F32, 2),
        Param::i32("m"),
        Param::i32("k"),
        Param::i32("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_cast_f16_f32` —— cast.cu(R1 补表)
pub struct CastF16F32;

impl KernelSpec for CastF16F32 {
    const ENTRY: &'static str = "owl_cast_f16_f32";
    const SRC: &'static str = "cu/elems/cast.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 0),
        Param::i32("n"),
        Param::t_out("out", Dtype::F32, 0),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_cast_f32_f16` —— cast.cu(R1 补表)
pub struct CastF32F16;

impl KernelSpec for CastF32F16 {
    const ENTRY: &'static str = "owl_cast_f32_f16";
    const SRC: &'static str = "cu/elems/cast.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 0),
        Param::i32("n"),
        Param::t_out("out", Dtype::F16, 0),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_embed_f32` —— embed_f32.cu(R1 补表)
pub struct EmbedF32;

impl KernelSpec for EmbedF32 {
    const ENTRY: &'static str = "owl_embed_f32";
    const SRC: &'static str = "cu/elems/embed_f32.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::F32, 2),
        Param::t_in("ids", Dtype::F32, 0),
        Param::sz("d_dim"),
        Param::t_out("out", Dtype::F32, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_narrow_strided_f32` —— narrow_strided.cu(R1 补表)
pub struct NarrowStridedF32;

impl KernelSpec for NarrowStridedF32 {
    const ENTRY: &'static str = "owl_narrow_strided_f32";
    const SRC: &'static str = "cu/elems/narrow_strided.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("src", Dtype::F32, 2),
        Param::sz("outer"),
        Param::sz("src_dim"),
        Param::sz("start"),
        Param::sz("out_dim"),
        Param::t_out("dst", Dtype::F32, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_rope_half_partial_f16` —— rope_f32.cu(R1 补表)
pub struct RopeHalfPartialF16;

impl KernelSpec for RopeHalfPartialF16 {
    const ENTRY: &'static str = "owl_rope_half_partial_f16";
    const SRC: &'static str = "cu/elems/rope_f32.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("cos_t", Dtype::F16, 2),
        Param::t_in("sin_t", Dtype::F16, 2),
        Param::t_in("pos", Dtype::F32, 2),
        Param::sz("heads"),
        Param::sz("head_dim"),
        Param::sz("half"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_rope_half_partial_f32` —— rope_f32.cu(R1 补表)
pub struct RopeHalfPartialF32;

impl KernelSpec for RopeHalfPartialF32 {
    const ENTRY: &'static str = "owl_rope_half_partial_f32";
    const SRC: &'static str = "cu/elems/rope_f32.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 2),
        Param::t_in("cos_t", Dtype::F32, 2),
        Param::t_in("sin_t", Dtype::F32, 2),
        Param::t_in("pos", Dtype::F32, 2),
        Param::sz("heads"),
        Param::sz("head_dim"),
        Param::sz("half"),
        Param::t_out("out", Dtype::F32, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `probe_ts_f16` —— ts_probe.cu(R1 补表)
pub struct ProbeTsF16;

impl KernelSpec for ProbeTsF16 {
    const ENTRY: &'static str = "probe_ts_f16";
    const SRC: &'static str = "cu/probe/ts_probe.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("in", Dtype::F16, 2),
        Param::t_in("ts", Dtype::U32, 2),
        Param::sz("idx"),
        Param::sz("n"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static OWL_ADD_F32: KernelMeta = meta_of::<AddF32>();
pub static OWL_MUL_F32: KernelMeta = meta_of::<MulF32>();
pub static OWL_MUL_F16: KernelMeta = meta_of::<MulF16>();
pub static OWL_MUL_BF16: KernelMeta = meta_of::<MulBf16>();
pub static OWL_SIGMOID_F32: KernelMeta = meta_of::<SigmoidF32>();
pub static OWL_SILU_F32: KernelMeta = meta_of::<SiluF32>();
pub static OWL_SILU_F16: KernelMeta = meta_of::<SiluF16>();
pub static OWL_MATMUL_F32: KernelMeta = meta_of::<MatmulF32>();
pub static OWL_MATMUL_NT_F32: KernelMeta = meta_of::<MatmulNtF32>();
pub static OWL_CAST_F16_F32: KernelMeta = meta_of::<CastF16F32>();
pub static OWL_CAST_F32_F16: KernelMeta = meta_of::<CastF32F16>();
pub static OWL_EMBED_F32: KernelMeta = meta_of::<EmbedF32>();
pub static OWL_NARROW_STRIDED_F32: KernelMeta = meta_of::<NarrowStridedF32>();
pub static OWL_ROPE_HALF_PARTIAL_F16: KernelMeta = meta_of::<RopeHalfPartialF16>();
pub static OWL_ROPE_HALF_PARTIAL_F32: KernelMeta = meta_of::<RopeHalfPartialF32>();
pub static PROBE_TS_F16: KernelMeta = meta_of::<ProbeTsF16>();

pub static QUANT_INT8_ROW_F16: KernelMeta = meta_of::<QuantInt8RowF16>();
pub static QUANT_INT8_ROW_BF16: KernelMeta = meta_of::<QuantInt8RowBf16>();

/// 本域全部已建表算子
// ============================================================================
// W4A8 前置:激活 per-token int8 量化(S1c;2026-10-16)
// ============================================================================

/// `owl_quant_int8_row_f16` —— 激活 per-token int8 量化(**W4A8 的喂料手**;
/// 上游 `dynamic_scaled_int8_quant_kernel` 的逐点 port)。
///
/// 语义:`scales[row] = max|x[row,:]|/127 · mul`;`q = cvt.rni.sat.s8(x · 127/absmax)`
/// (round-nearest-even + 饱和;零行 ⇒ 整行 0)。`mul` = **权重侧全局因子**
/// (`pack_marlin_s_a8` 第二返回值 `max/4096`;它是**每层常量**,由调用方折入,
/// 这样运行时零额外发射 —— 见设计卷 §9.2③)。
///
/// 槽:`x(f16,2) / rows(i32) / cols(i32) / mul(f32) / scales(f32,2,写) / q(u8,2,出)`
/// —— `scales` 声明为 `t_in`(由调用方备块,同 xqa 的 `wr`/`scratch` 先例),
/// `q` 才是本算子回传的输出槽(字节承载 int8,与 W4A8 GEMM 的 `a` 槽同惯例:
/// 同一块要能同时满足两算子的 dtype/rank 事实门)。
pub struct QuantInt8RowF16;

impl KernelSpec for QuantInt8RowF16 {
    const ENTRY: &'static str = "owl_quant_int8_row_f16";
    const SRC: &'static str = "cu/elems/quant_int8_row.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        // **设备标量块** f32[1]:权重侧全局因子(装载期由尺度数据算出 ⇒
        // 走 host 标量要给装载器开旁路;设备块与 vLLM `input_global_scale` 同构)
        Param::t_in("mul", Dtype::F32, 1),
        // 写出槽:per-token 尺度 [rows,1](调用方备块)
        Param::t_in("scales", Dtype::F32, 2),
        Param::t_out("q", Dtype::I8, 2),
    ];
    // 出参是 int8:有限性律不适用(饱和已由 `cvt.rni.sat` 保证)—— 故 LAWS 空,
    // 靠**核内饱和 + 回读对拍**（tests/quant_int8_row_probe.rs）守门
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_quant_int8_row_bf16` —— 同上,入 f16 → bf16(草稿/DFlash2 路备用)
pub struct QuantInt8RowBf16;

impl KernelSpec for QuantInt8RowBf16 {
    const ENTRY: &'static str = "owl_quant_int8_row_bf16";
    const SRC: &'static str = "cu/elems/quant_int8_row.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::t_in("mul", Dtype::F32, 1),
        Param::t_in("scales", Dtype::F32, 2),
        Param::t_out("q", Dtype::I8, 2),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

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
    &TOPK16_SPLIT_F16,
    &TOPK16_MERGE_F16,
    &OWL_ADD_F32,
    &OWL_MUL_F32,
    &OWL_MUL_F16,
    &OWL_MUL_BF16,
    &OWL_SIGMOID_F32,
    &OWL_SILU_F32,
    &OWL_SILU_F16,
    &OWL_MATMUL_F32,
    &OWL_MATMUL_NT_F32,
    &OWL_CAST_F16_F32,
    &OWL_CAST_F32_F16,
    &OWL_EMBED_F32,
    &OWL_NARROW_STRIDED_F32,
    &OWL_ROPE_HALF_PARTIAL_F16,
    &OWL_ROPE_HALF_PARTIAL_F32,
    &PROBE_TS_F16,
    &QUANT_INT8_ROW_F16,
    &QUANT_INT8_ROW_BF16,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// **常量 ↔ 声明面锁**(2026-10-16):`CONCAT_ROWS_SLOTS` 必须等于两张表里
    /// `i0..i{n-1}` 的个数 —— 改核时忘改常量 = 本门红(否则 engine 的
    /// 固定槽树会静默 pad 到错槽数)。
    #[test]
    fn concat_rows_slots_lock() {
        for m in [&CONCAT_ROWS_F16, &CONCAT_ROWS_F32] {
            let slots = m
                .params
                .iter()
                .filter(|p| {
                    p.name
                        .strip_prefix('i')
                        .is_some_and(|n| n.parse::<usize>().is_ok())
                })
                .count();
            assert_eq!(
                slots, CONCAT_ROWS_SLOTS,
                "{}: 输入槽数 {slots} ≠ CONCAT_ROWS_SLOTS {CONCAT_ROWS_SLOTS}",
                m.entry
            );
        }
    }
}
