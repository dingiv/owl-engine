//! 逐元素 / 视图 / 采样 / 铸边界 生产算子元数据。

use owl_macros::kernel_tags;
use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};

/// `owl_narrow_strided_f16` —— 窄切物化(逐 token 视图;`outer/src_dim/start/out_dim` 四标量)
pub struct NarrowStridedF16;

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.gemv", "phase.decode")]
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

/// `owl_gemv_i8w_f16` —— **int8 权重 × f16 激活的 GEMV**(m=1 decode;P1.5-a)
///
/// 语义见 `.cu` 头注:`y[token,r] = scale[r] · Σ_k w_i8[r,k] · x[token,k]`;
/// per-row 对称量化(装载期现算),`cols % 16 == 0`(几何侧断言)。
pub struct GemvI8wF16;

#[kernel_tags("dom.elem", "role.gemv", "quant.int8", "feat.head", "phase.decode")]
impl KernelSpec for GemvI8wF16 {
    const ENTRY: &'static str = "owl_gemv_i8w_f16";
    const SRC: &'static str = "cu/elems/gemv_i8w_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::I8, 2),
        Param::t_in("scale", Dtype::F32, 1),
        Param::t_in("x", Dtype::F16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::t_out("y", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_add_f16` —— 逐元素加(宏实例化;fc 部分和累加)
pub struct AddF16;

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

/// **首轮拼接的输入槽数**(`owl_concat_rows_{f16,f32}` = `i0..i{15}`;arity 固定)。
///
/// 16 = **核声明面的事实**(两张表都恰好 16 槽)—— DFlash2 K=1..15
/// 单 block 的 verify 行数 = K+1 最大 16。调用方(engine 的“固定槽树”
/// 把 t 行 pad 到 16 槽)与装配期门**都取本常量**,不许各写一个字面量;
/// 改核时本处跟着改,`concat_rows_slots_lock` 拿两张表逐槽核。
pub const CONCAT_ROWS_SLOTS: usize = 16;

/// `owl_concat_rows_f16` —— 16 路行拼接(arity 16 = 固定)
pub struct ConcatRowsF16;

#[kernel_tags("dom.elem", "role.elem")]
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
        Param::t_in("i8", Dtype::F16, 0),
        Param::t_in("i9", Dtype::F16, 0),
        Param::t_in("i10", Dtype::F16, 0),
        Param::t_in("i11", Dtype::F16, 0),
        Param::t_in("i12", Dtype::F16, 0),
        Param::t_in("i13", Dtype::F16, 0),
        Param::t_in("i14", Dtype::F16, 0),
        Param::t_in("i15", Dtype::F16, 0),
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

#[kernel_tags("dom.elem", "role.elem")]
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
        Param::t_in("i8", Dtype::F32, 0),
        Param::t_in("i9", Dtype::F32, 0),
        Param::t_in("i10", Dtype::F32, 0),
        Param::t_in("i11", Dtype::F32, 0),
        Param::t_in("i12", Dtype::F32, 0),
        Param::t_in("i13", Dtype::F32, 0),
        Param::t_in("i14", Dtype::F32, 0),
        Param::t_in("i15", Dtype::F32, 0),
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

#[kernel_tags("dom.elem", "role.embed")]
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

#[kernel_tags("dom.elem", "role.sample", "phase.decode")]
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

#[kernel_tags("dom.elem", "role.sample", "phase.decode")]
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
pub static GEMV_I8W_F16: KernelMeta = meta_of::<GemvI8wF16>();

/// `owl_gemv_i8w_bf16` —— 同上的 **bf16 激活**变体(生产 lm_head 走这条:
/// 模型激活是 bf16,省掉两次 cast 节点;语义/几何与 f16 版逐字相同)
pub struct GemvI8wBf16;

#[kernel_tags("dom.elem", "role.gemv", "quant.int8", "feat.head", "phase.decode")]
impl KernelSpec for GemvI8wBf16 {
    const ENTRY: &'static str = "owl_gemv_i8w_bf16";
    const SRC: &'static str = "cu/elems/gemv_i8w_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::I8, 2),
        Param::t_in("scale", Dtype::F32, 1),
        Param::t_in("x", Dtype::BF16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::t_out("y", Dtype::BF16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static GEMV_I8W_BF16: KernelMeta = meta_of::<GemvI8wBf16>();

/// `owl_gemv_i8w_mt_{f16,bf16}` —— **多 token int8 权重 GEMV**
/// (2026-10-22;spec verify M=7/8 主路)。
///
/// 与单 token 版**语义逐位相同**;差别只在调度:一个 block 一次读权重、
/// 同时算 `TOK_TILE` 个 token,`ROWS_PER_WARP=4` 让一个权重字节复用到
/// 4 行 × 8 token。`tokens` 是新增的显式标量(token tile 的边界)。
pub struct GemvI8wMtF16;

#[kernel_tags("dom.elem", "role.gemv", "quant.int8", "feat.head", "phase.decode")]
impl KernelSpec for GemvI8wMtF16 {
    const ENTRY: &'static str = "owl_gemv_i8w_mt_f16";
    const SRC: &'static str = "cu/elems/gemv_i8w_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::I8, 2),
        Param::t_in("scale", Dtype::F32, 1),
        Param::t_in("x", Dtype::F16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::i32("tokens"),
        Param::t_out("y", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static GEMV_I8W_MT_F16: KernelMeta = meta_of::<GemvI8wMtF16>();

/// 同上,bf16 激活变体(生产 target verify / lm_head 走这条)。
pub struct GemvI8wMtBf16;

#[kernel_tags("dom.elem", "role.gemv", "quant.int8", "feat.head", "phase.decode")]
impl KernelSpec for GemvI8wMtBf16 {
    const ENTRY: &'static str = "owl_gemv_i8w_mt_bf16";
    const SRC: &'static str = "cu/elems/gemv_i8w_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::I8, 2),
        Param::t_in("scale", Dtype::F32, 1),
        Param::t_in("x", Dtype::BF16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::i32("tokens"),
        Param::t_out("y", Dtype::BF16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static GEMV_I8W_MT_BF16: KernelMeta = meta_of::<GemvI8wMtBf16>();

/// `owl_gemv_i8w_mt7_{f16,bf16}` —— **7-token 专用实例**（DFlash2 propose
/// 固定 7 行；2026-10-22）。语义/参数与 8-token 版逐字相同，仅少算第 8 行。
pub struct GemvI8wMt7F16;

#[kernel_tags("dom.elem", "role.gemv", "quant.int8", "feat.head", "phase.decode")]
impl KernelSpec for GemvI8wMt7F16 {
    const ENTRY: &'static str = "owl_gemv_i8w_mt7_f16";
    const SRC: &'static str = "cu/elems/gemv_i8w_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::I8, 2),
        Param::t_in("scale", Dtype::F32, 1),
        Param::t_in("x", Dtype::F16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::i32("tokens"),
        Param::t_out("y", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static GEMV_I8W_MT7_F16: KernelMeta = meta_of::<GemvI8wMt7F16>();

/// 同上，bf16 激活变体。
pub struct GemvI8wMt7Bf16;

#[kernel_tags("dom.elem", "role.gemv", "quant.int8", "feat.head", "phase.decode")]
impl KernelSpec for GemvI8wMt7Bf16 {
    const ENTRY: &'static str = "owl_gemv_i8w_mt7_bf16";
    const SRC: &'static str = "cu/elems/gemv_i8w_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::I8, 2),
        Param::t_in("scale", Dtype::F32, 1),
        Param::t_in("x", Dtype::BF16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::i32("tokens"),
        Param::t_out("y", Dtype::BF16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static GEMV_I8W_MT7_BF16: KernelMeta = meta_of::<GemvI8wMt7Bf16>();

// ============================================================================
// W8A8 head:per-token 激活量化 + dp4a 多 token GEMV(2026-10-22)
// ============================================================================

/// `owl_quant_i8_act_scale_{f16,bf16}` —— 每 token absmax → f32 scale。
pub struct QuantI8ActScaleF16;

#[kernel_tags("dom.elem", "role.elem", "quant.int8", "feat.head", "phase.decode")]
impl KernelSpec for QuantI8ActScaleF16 {
    const ENTRY: &'static str = "owl_quant_i8_act_scale_f16";
    const SRC: &'static str = "cu/elems/quant_i8_act.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::t_out("scale", Dtype::F32, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static QUANT_I8_ACT_SCALE_F16: KernelMeta = meta_of::<QuantI8ActScaleF16>();

pub struct QuantI8ActScaleBf16;

#[kernel_tags("dom.elem", "role.elem", "quant.int8", "feat.head", "phase.decode")]
impl KernelSpec for QuantI8ActScaleBf16 {
    const ENTRY: &'static str = "owl_quant_i8_act_scale_bf16";
    const SRC: &'static str = "cu/elems/quant_i8_act.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::t_out("scale", Dtype::F32, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static QUANT_I8_ACT_SCALE_BF16: KernelMeta = meta_of::<QuantI8ActScaleBf16>();

/// `owl_quant_i8_act_q_{f16,bf16}` —— 吃 scale,写 RN-even i8。
pub struct QuantI8ActQF16;

#[kernel_tags("dom.elem", "role.elem", "quant.int8", "feat.head", "phase.decode")]
impl KernelSpec for QuantI8ActQF16 {
    const ENTRY: &'static str = "owl_quant_i8_act_q_f16";
    const SRC: &'static str = "cu/elems/quant_i8_act.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("scale", Dtype::F32, 1),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::t_out("q", Dtype::I8, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static QUANT_I8_ACT_Q_F16: KernelMeta = meta_of::<QuantI8ActQF16>();

pub struct QuantI8ActQBf16;

#[kernel_tags("dom.elem", "role.elem", "quant.int8", "feat.head", "phase.decode")]
impl KernelSpec for QuantI8ActQBf16 {
    const ENTRY: &'static str = "owl_quant_i8_act_q_bf16";
    const SRC: &'static str = "cu/elems/quant_i8_act.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::t_in("scale", Dtype::F32, 1),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::t_out("q", Dtype::I8, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static QUANT_I8_ACT_Q_BF16: KernelMeta = meta_of::<QuantI8ActQBf16>();

/// `owl_gemv_i8w_a8_mt{,_7}_{f16,bf16}` —— W8A8 dp4a 多头 GEMV。
macro_rules! gemv_i8w_a8_spec {
    ($name:ident, $entry:literal, $src_ty:ident) => {
        pub struct $name;
        #[kernel_tags("dom.elem", "role.gemv", "quant.int8", "feat.head", "phase.decode")]
        impl KernelSpec for $name {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = "cu/elems/gemv_i8w_a8.cu";
            const PARAMS: &'static [Param] = &[
                Param::t_in("w", Dtype::I8, 2),
                Param::t_in("sw", Dtype::F32, 1),
                Param::t_in("xq", Dtype::I8, 2),
                Param::t_in("sx", Dtype::F32, 1),
                Param::i32("rows"),
                Param::i32("cols"),
                Param::i32("tokens"),
                Param::t_out("y", Dtype::$src_ty, 2),
            ];
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

gemv_i8w_a8_spec!(GemvI8wA8MtF16, "owl_gemv_i8w_a8_mt_f16", F16);
gemv_i8w_a8_spec!(GemvI8wA8MtBf16, "owl_gemv_i8w_a8_mt_bf16", BF16);
gemv_i8w_a8_spec!(GemvI8wA8Mt7F16, "owl_gemv_i8w_a8_mt7_f16", F16);
gemv_i8w_a8_spec!(GemvI8wA8Mt7Bf16, "owl_gemv_i8w_a8_mt7_bf16", BF16);

pub static GEMV_I8W_A8_MT_F16: KernelMeta = meta_of::<GemvI8wA8MtF16>();
pub static GEMV_I8W_A8_MT_BF16: KernelMeta = meta_of::<GemvI8wA8MtBf16>();
pub static GEMV_I8W_A8_MT7_F16: KernelMeta = meta_of::<GemvI8wA8Mt7F16>();
pub static GEMV_I8W_A8_MT7_BF16: KernelMeta = meta_of::<GemvI8wA8Mt7Bf16>();

/// `owl_gemm_i8w_a8_tc_{f16,bf16}` —— **W8A8 多头 int8 Tensor Core**
/// (`mma.sync.m16n8k32.s8s8s32`;SM86 有 int8 TC、没有 fp8 TC)。
///
/// 语义与 dp4a 版**逐位相同**(同一个 int8 点积、同一个 per-token scale);
/// 区别是权重走 64 行/CTA 的 shared 暂存 + 16B 连续装载,不再每 warp 散读。
/// `tokens` 仍为显式标量(2..=8;M 固定补到 16)。
macro_rules! gemm_i8w_a8_tc_spec {
    ($name:ident, $entry:literal, $src_ty:ident) => {
        pub struct $name;
        #[kernel_tags("dom.elem", "role.gemv", "quant.int8", "feat.head", "phase.decode")]
        impl KernelSpec for $name {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = "cu/elems/gemm_i8w_a8_tc.cu";
            const PARAMS: &'static [Param] = &[
                Param::t_in("w", Dtype::I8, 2),
                Param::t_in("sw", Dtype::F32, 1),
                Param::t_in("q", Dtype::I8, 2),
                Param::t_in("sx", Dtype::F32, 1),
                Param::i32("rows"),
                Param::i32("cols"),
                Param::i32("tokens"),
                Param::t_out("y", Dtype::$src_ty, 2),
            ];
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

gemm_i8w_a8_tc_spec!(GemmI8wA8TcF16, "owl_gemm_i8w_a8_tc_f16", F16);
gemm_i8w_a8_tc_spec!(GemmI8wA8TcBf16, "owl_gemm_i8w_a8_tc_bf16", BF16);

pub static GEMM_I8W_A8_TC_F16: KernelMeta = meta_of::<GemmI8wA8TcF16>();
pub static GEMM_I8W_A8_TC_BF16: KernelMeta = meta_of::<GemmI8wA8TcBf16>();

/// `owl_gemm_i8w_a16_tc_{f16,bf16}` —— **W8A16 多头 f16/bf16 Tensor Core**
///
/// HBM 读 int8 权重;B 在 shared 仍是 int8,MMA fragment 在寄存器里
/// `int8 → half/bf16`。激活保持 f16/bf16,不做 W8A8 的激活量化。
/// 目标 = Marlin W4A16 同款“压缩读 + 近端反量化”流水,但权重是 8-bit。
macro_rules! gemm_i8w_a16_tc_spec {
    ($name:ident, $entry:literal, $src_ty:ident) => {
        pub struct $name;
        #[kernel_tags("dom.elem", "role.gemv", "quant.int8", "feat.head", "phase.decode")]
        impl KernelSpec for $name {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = "cu/elems/gemm_i8w_a16_tc.cu";
            const PARAMS: &'static [Param] = &[
                Param::t_in("w", Dtype::I8, 2),
                Param::t_in("sw", Dtype::F32, 1),
                Param::t_in("x", Dtype::$src_ty, 2),
                Param::i32("rows"),
                Param::i32("cols"),
                Param::i32("tokens"),
                Param::t_out("y", Dtype::$src_ty, 2),
            ];
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

gemm_i8w_a16_tc_spec!(GemmI8wA16TcF16, "owl_gemm_i8w_a16_tc_f16", F16);
gemm_i8w_a16_tc_spec!(GemmI8wA16TcBf16, "owl_gemm_i8w_a16_tc_bf16", BF16);

pub static GEMM_I8W_A16_TC_F16: KernelMeta = meta_of::<GemmI8wA16TcF16>();
pub static GEMM_I8W_A16_TC_BF16: KernelMeta = meta_of::<GemmI8wA16TcBf16>();
pub static ADD_F16: KernelMeta = meta_of::<AddF16>();
pub static ADD_BF16: KernelMeta = meta_of::<AddBf16>();
pub static SIGMOID_F16: KernelMeta = meta_of::<SigmoidF16>();
pub static SIGMOID_GATE_MUL_F16: KernelMeta = meta_of::<SigmoidGateMulF16>();
/// `owl_scale_{f16,bf16}` —— 逐元素乘常数(成对指数移位;见 `cu/elems/scale.cu`
/// 头注:GGUF 块型 GEMM 的 F16 输出上限 65504 × BF16 草稿域 = 必须缩放)
pub struct ScaleF16;

#[kernel_tags("dom.elem", "role.elem")]
impl KernelSpec for ScaleF16 {
    const ENTRY: &'static str = "owl_scale_f16";
    const SRC: &'static str = "cu/elems/scale.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 0),
        Param::i32("n"),
        Param::f32("scale"),
        Param::t_out("out", Dtype::F16, 0),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub struct ScaleBf16;

#[kernel_tags("dom.elem", "role.elem")]
impl KernelSpec for ScaleBf16 {
    const ENTRY: &'static str = "owl_scale_bf16";
    const SRC: &'static str = "cu/elems/scale.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 0),
        Param::i32("n"),
        Param::f32("scale"),
        Param::t_out("out", Dtype::BF16, 0),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static SCALE_F16: KernelMeta = meta_of::<ScaleF16>();
pub static SCALE_BF16: KernelMeta = meta_of::<ScaleBf16>();
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

#[kernel_tags("dom.elem", "role.sample", "phase.decode")]
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

#[kernel_tags("dom.elem", "role.sample", "phase.decode")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.gemm")]
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

#[kernel_tags("dom.elem", "role.gemm")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.embed")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.rope", "feat.vision")]
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

#[kernel_tags("dom.elem", "role.rope", "feat.vision")]
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

#[kernel_tags("dom.elem", "role.elem")]
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

#[kernel_tags("dom.elem", "role.elem", "quant.w4a8", "phase.decode", "phase.load")]
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

#[kernel_tags("dom.elem", "role.elem", "quant.w4a8", "phase.decode", "phase.load")]
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
