//! 装载域生产算子元数据(ct packed → marlin B 设备重排)。

use owl_macros::kernel_tags;
use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};

/// `owl_ct_repack_u32` —— AWQ 装载线的设备侧权重重排(**非轮内**;装载期一次)
pub struct CtRepackU32;

#[kernel_tags("dom.load", "phase.load", "role.elem")]
impl KernelSpec for CtRepackU32 {
    const ENTRY: &'static str = "owl_ct_repack_u32";
    const SRC: &'static str = "cu/load/marlin_repack_ct.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("in", Dtype::U32, 2),
        Param::sz("rows"),
        Param::sz("cols"),
        Param::t_out("out", Dtype::U32, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static CT_REPACK_U32: KernelMeta = meta_of::<CtRepackU32>();

/// `owl_ct_repack_a8_u32` —— W4A8 的 B 重排(S2 补)。
///
/// **同输入契约、同输出字数**于 a16 版(`k·n/8` u32),但 tile 几何与 nibble
/// 排布不同(核侧 a8 的 B 读法与 a16 不同)⇒ 两个算子**不可互换**
/// (判别探针实测:a8 布局喂 a16 表错列,反之亦然;见设计卷 §14.3)。
/// 参照实现 = `family::marlin::repack::pack_marlin_b_a8`(真机对拍验证过)。
pub struct CtRepackA8U32;

#[kernel_tags("dom.load", "phase.load", "role.elem", "quant.w4a8")]
impl KernelSpec for CtRepackA8U32 {
    const ENTRY: &'static str = "owl_ct_repack_a8_u32";
    const SRC: &'static str = "cu/load/marlin_repack_ct_a8.cu";
    const PARAMS: &'static [Param] = &K0_REPACK_PARAMS;
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// 两版重排共用参数面(in: U32[rows,cols] / rows / cols / out: U32;逐字 = .cu 签名)
const K0_REPACK_PARAMS: &[Param] = &[
    Param::t_in("in", Dtype::U32, 2),
    Param::sz("rows"),
    Param::sz("cols"),
    Param::t_out("out", Dtype::U32, 2),
];

pub static CT_REPACK_A8_U32: KernelMeta = meta_of::<CtRepackA8U32>();

/// `owl_quant_i8_row_w_scale_bf16` —— lm_head 逐行对称 int8 量化的**第一步**
/// (行内 absmax → `scale[row]`;P1.5-a,装载期一次)
///
/// 与第二步拆成两发的原因 = **Out 槽必须末位**(`op_tables::
/// non_family_out_slot_is_last_tensor`);语义与 host 参考逐元素相同(台子背书)。
pub struct QuantI8RowWScaleBf16;

#[kernel_tags("dom.load", "phase.load", "role.elem", "quant.int8")]
impl KernelSpec for QuantI8RowWScaleBf16 {
    const ENTRY: &'static str = "owl_quant_i8_row_w_scale_bf16";
    const SRC: &'static str = "cu/load/quant_i8_row_w.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::BF16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::t_out("scale", Dtype::F32, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_quant_i8_row_w_q_bf16` —— 第二步:吃 `scale`,写 `q = RN(w/s)` 截到 ±127
pub struct QuantI8RowWQBf16;

#[kernel_tags("dom.load", "phase.load", "role.elem", "quant.int8")]
impl KernelSpec for QuantI8RowWQBf16 {
    const ENTRY: &'static str = "owl_quant_i8_row_w_q_bf16";
    const SRC: &'static str = "cu/load/quant_i8_row_w.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("w", Dtype::BF16, 2),
        Param::t_in("scale", Dtype::F32, 1),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::t_out("q", Dtype::I8, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static QUANT_I8_ROW_W_SCALE_BF16: KernelMeta = meta_of::<QuantI8RowWScaleBf16>();
pub static QUANT_I8_ROW_W_Q_BF16: KernelMeta = meta_of::<QuantI8RowWQBf16>();
