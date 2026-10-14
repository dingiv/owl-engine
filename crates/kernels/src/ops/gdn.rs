//! GDN(线性注意力)生产算子元数据。
//!
//! 参数名/序/类型 = `.cu` 签名的逐字抄录(机器互证 `tests/op_tables.rs`)。
//! GDN 族是"曾出过事"的重灾区(raw g 语义律 E7、同源对拍双错 E8)——本表的
//! `LAWS` 声明是装配期不变量门的挂点(设计 §5.5)。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, pools, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param, PoolAccess};

// ============================================================================
// 门控 / 归一(前置琐核)
// ============================================================================

/// `owl_gdn_gating_g_f16` —— g = softplus(a_log) + dt_bias 门
pub struct GatingGF16;

impl KernelSpec for GatingGF16 {
    const ENTRY: &'static str = "owl_gdn_gating_g_f16";
    const SRC: &'static str = "cu/gdn/gdn_gating.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a_log", Dtype::F16, 1),
        Param::t_in("a", Dtype::F16, 2),
        Param::t_in("dt_bias", Dtype::F16, 1),
        Param::sz("total"),
        Param::sz("heads"),
        Param::t_out("g", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_gdn_l2norm_f16` —— 行末维 L2 归一(一 block 一行)
pub struct L2NormF16;

impl KernelSpec for L2NormF16 {
    const ENTRY: &'static str = "owl_gdn_l2norm_f16";
    const SRC: &'static str = "cu/gdn/gdn_l2norm.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::sz("rows"),
        Param::sz("dim"),
        Param::f32("eps"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_gdn_norm_act_f16` —— 分组 RMSNorm + 激活(门控)
pub struct NormActF16;

impl KernelSpec for NormActF16 {
    const ENTRY: &'static str = "owl_gdn_norm_act_f16";
    const SRC: &'static str = "cu/gdn/gdn_norm_act.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("z", Dtype::F16, 2),
        Param::t_in("gamma", Dtype::F16, 1),
        Param::sz("rows"),
        Param::sz("value_dim"),
        Param::sz("group_size"),
        Param::f32("eps"),
        Param::i32("act"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

// ============================================================================
// 卷积(短卷积状态更新 / 批式前向)
// ============================================================================

/// `owl_gdn_conv_fwd_f16` —— chunked prefill 批式短卷积(state 原地更新)
pub struct ConvFwdF16;

impl KernelSpec for ConvFwdF16 {
    const ENTRY: &'static str = "owl_gdn_conv_fwd_f16";
    const SRC: &'static str = "cu/gdn/gdn_conv_fwd.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("w", Dtype::F16, 2),
        // conv_state 原地更新(副作用律;单流保序)
        Param::t_inout("conv_state", Dtype::F32, 0),
        Param::t_in("slots", Dtype::F32, 0),
        Param::t_in("cu_seqlens", Dtype::F32, 0),
        Param::i32("batch"),
        Param::i32("d"),
        Param::i32("silu"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::rw(pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_gdn_conv_upd_dual_f16` —— q/k 双段 conv 单发(输出单块 [batch, dq+dk])
pub struct ConvUpdDualF16;

impl KernelSpec for ConvUpdDualF16 {
    const ENTRY: &'static str = "owl_gdn_conv_upd_dual_f16";
    const SRC: &'static str = "cu/gdn/gdn_conv_upd.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x_q", Dtype::F16, 2),
        Param::t_in("x_k", Dtype::F16, 2),
        Param::t_in("w_q", Dtype::F16, 2),
        Param::t_in("w_k", Dtype::F16, 2),
        Param::t_inout("state_q", Dtype::F32, 3),
        Param::t_inout("state_k", Dtype::F32, 3),
        Param::t_in("slots", Dtype::F32, 0),
        Param::i32("dq"),
        Param::i32("dk"),
        Param::i32("batch"),
        Param::i32("silu"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::rw(pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

// ============================================================================
// 递推(批式 varlen / decode 融合)
// ============================================================================

/// `owl_gdn_recurrence_varlen_gqa_f16` —— 批式 varlen GQA 递推(chunked 路径)
pub struct RecurrenceVarlenGqaF16;

impl KernelSpec for RecurrenceVarlenGqaF16 {
    const ENTRY: &'static str = "owl_gdn_recurrence_varlen_gqa_f16";
    const SRC: &'static str = "cu/gdn/gdn_recurrence.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 3),
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        // g 是 raw 门(语义律 E7:cumsum 是核内私有态,客户端不可达)
        Param::t_in("g", Dtype::F16, 2),
        Param::t_in("beta", Dtype::F16, 2),
        Param::t_inout("state", Dtype::F32, 0),
        Param::t_in("slots", Dtype::F32, 0),
        Param::t_in("cu_seqlens", Dtype::F32, 0),
        Param::sz("batch"),
        Param::sz("nv"),
        Param::sz("nk"),
        Param::sz("kd"),
        Param::sz("vd"),
        Param::f32("q_scale"),
        Param::t_out("out", Dtype::F16, 3),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::rw(pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_gdn_decode_step_v2_f16` —— decode 整链融合(v-conv + l2norm×2 + gating
/// + sigmoid + delta + norm_act;22 参 = 13 块 + 5 sz + 3 f32 + 输出)
pub struct DecodeStepV2F16;

impl KernelSpec for DecodeStepV2F16 {
    const ENTRY: &'static str = "owl_gdn_decode_step_v2_f16";
    const SRC: &'static str = "cu/gdn/gdn_decode_step.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q_c", Dtype::F16, 2),
        Param::t_in("k_c", Dtype::F16, 2),
        Param::t_in("v_raw", Dtype::F16, 2),
        Param::t_in("z", Dtype::F16, 2),
        Param::t_in("b_gate", Dtype::F16, 2),
        Param::t_in("a_gate", Dtype::F16, 2),
        Param::t_in("w", Dtype::F16, 2),
        Param::t_inout("conv_v_state", Dtype::F32, 3),
        Param::t_in("a_log", Dtype::F16, 1),
        Param::t_in("dt_bias", Dtype::F16, 1),
        Param::t_inout("rec_state", Dtype::F32, 4),
        Param::t_in("slots", Dtype::F32, 0),
        Param::t_in("norm_w", Dtype::F16, 1),
        Param::sz("batch"),
        Param::sz("nv"),
        Param::sz("nk"),
        Param::sz("kd"),
        Param::sz("vd"),
        Param::f32("eps_l2"),
        Param::f32("eps_norm"),
        Param::f32("q_scale"),
        Param::t_out("out", Dtype::F16, 3),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::rw(pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

// ============================================================================
// 命名 static(**单一提升点**):域表与族表都引用它 —— 同一算子不会出现两份
// `KernelMeta`(重复提升会让"元数据同一性"比较失效)
// ============================================================================

pub static GATING_G_F16: KernelMeta = meta_of::<GatingGF16>();
pub static L2NORM_F16: KernelMeta = meta_of::<L2NormF16>();
pub static NORM_ACT_F16: KernelMeta = meta_of::<NormActF16>();
pub static CONV_FWD_F16: KernelMeta = meta_of::<ConvFwdF16>();
pub static CONV_UPD_DUAL_F16: KernelMeta = meta_of::<ConvUpdDualF16>();
pub static RECURRENCE_VARLEN_GQA_F16: KernelMeta = meta_of::<RecurrenceVarlenGqaF16>();
pub static DECODE_STEP_V2_F16: KernelMeta = meta_of::<DecodeStepV2F16>();


// ============================================================================
// `gdn.rs` 域 R1 补表
// (R1 补表 2026-10-13:声明从登记表派生,签名由 `op_tables` 门逐参互证)
// ============================================================================


/// `owl_gdn_conv_upd_f16` —— gdn_conv_upd.cu(R1 补表)
pub struct GdnConvUpdF16;

impl KernelSpec for GdnConvUpdF16 {
    const ENTRY: &'static str = "owl_gdn_conv_upd_f16";
    const SRC: &'static str = "cu/gdn/gdn_conv_upd.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("w", Dtype::F16, 2),
        Param::t_in("conv_state", Dtype::F32, 0),
        Param::t_in("slots", Dtype::F32, 0),
        Param::sz("total"),
        Param::sz("d"),
        Param::sz("w_offset"),
        Param::i32("silu"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::rw(pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_gdn_conv_upd_f32` —— gdn_conv_upd.cu(R1 补表)
pub struct GdnConvUpdF32;

impl KernelSpec for GdnConvUpdF32 {
    const ENTRY: &'static str = "owl_gdn_conv_upd_f32";
    const SRC: &'static str = "cu/gdn/gdn_conv_upd.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 2),
        Param::t_in("w", Dtype::F32, 2),
        Param::t_in("conv_state", Dtype::F32, 0),
        Param::t_in("slots", Dtype::F32, 0),
        Param::sz("total"),
        Param::sz("d"),
        Param::sz("w_offset"),
        Param::i32("silu"),
        Param::t_out("out", Dtype::F32, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::rw(pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_gdn_delta_dec_f16` —— gdn_delta_dec.cu(R1 补表)
pub struct GdnDeltaDecF16;

impl KernelSpec for GdnDeltaDecF16 {
    const ENTRY: &'static str = "owl_gdn_delta_dec_f16";
    const SRC: &'static str = "cu/gdn/gdn_delta_dec.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 3),
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        Param::t_in("g", Dtype::F16, 2),
        Param::t_in("beta", Dtype::F16, 2),
        Param::t_in("state", Dtype::F32, 4),
        Param::t_in("slots", Dtype::F32, 0),
        Param::sz("batch"),
        Param::sz("nv"),
        Param::sz("nk"),
        Param::sz("kd"),
        Param::sz("vd"),
        Param::f32("q_scale"),
        Param::t_out("out", Dtype::F16, 3),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::rw(pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_gdn_delta_dec_f32` —— gdn_delta_dec.cu(R1 补表)
pub struct GdnDeltaDecF32;

impl KernelSpec for GdnDeltaDecF32 {
    const ENTRY: &'static str = "owl_gdn_delta_dec_f32";
    const SRC: &'static str = "cu/gdn/gdn_delta_dec.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F32, 3),
        Param::t_in("k", Dtype::F32, 3),
        Param::t_in("v", Dtype::F32, 3),
        Param::t_in("g", Dtype::F32, 2),
        Param::t_in("beta", Dtype::F32, 2),
        Param::t_in("state", Dtype::F32, 4),
        Param::t_in("slots", Dtype::F32, 0),
        Param::sz("batch"),
        Param::sz("nv"),
        Param::sz("nk"),
        Param::sz("kd"),
        Param::sz("vd"),
        Param::f32("q_scale"),
        Param::t_out("out", Dtype::F32, 3),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::rw(pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_gdn_decode_step_f16` —— gdn_decode_step.cu(R1 补表)
pub struct GdnDecodeStepF16;

impl KernelSpec for GdnDecodeStepF16 {
    const ENTRY: &'static str = "owl_gdn_decode_step_f16";
    const SRC: &'static str = "cu/gdn/gdn_decode_step.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q_c", Dtype::F16, 2),
        Param::t_in("k_c", Dtype::F16, 2),
        Param::t_in("v_raw", Dtype::F16, 2),
        Param::t_in("z", Dtype::F16, 2),
        Param::t_in("b_gate", Dtype::F16, 2),
        Param::t_in("a_gate", Dtype::F16, 2),
        Param::t_in("w", Dtype::F16, 2),
        Param::t_in("conv_v_state", Dtype::F32, 2),
        Param::t_in("a_log", Dtype::F16, 2),
        Param::t_in("dt_bias", Dtype::F16, 2),
        Param::t_in("rec_state", Dtype::F32, 2),
        Param::t_in("slots", Dtype::F32, 2),
        Param::t_in("norm_w", Dtype::F16, 2),
        Param::sz("batch"),
        Param::sz("nv"),
        Param::sz("nk"),
        Param::sz("kd"),
        Param::sz("vd"),
        Param::f32("eps_l2"),
        Param::f32("eps_norm"),
        Param::f32("q_scale"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::rw(pools::GDN_STATE, Dtype::F32)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_gdn_gating_g_f32` —— gdn_gating.cu(R1 补表)
pub struct GdnGatingGF32;

impl KernelSpec for GdnGatingGF32 {
    const ENTRY: &'static str = "owl_gdn_gating_g_f32";
    const SRC: &'static str = "cu/gdn/gdn_gating.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a_log", Dtype::F32, 1),
        Param::t_in("a", Dtype::F32, 2),
        Param::t_in("dt_bias", Dtype::F32, 1),
        Param::sz("total"),
        Param::sz("heads"),
        Param::t_out("g", Dtype::F32, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_gdn_l2norm_f32` —— gdn_l2norm.cu(R1 补表)
pub struct GdnL2normF32;

impl KernelSpec for GdnL2normF32 {
    const ENTRY: &'static str = "owl_gdn_l2norm_f32";
    const SRC: &'static str = "cu/gdn/gdn_l2norm.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 2),
        Param::sz("rows"),
        Param::sz("dim"),
        Param::f32("eps"),
        Param::t_out("out", Dtype::F32, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_gdn_norm_act_f32` —— gdn_norm_act.cu(R1 补表)
pub struct GdnNormActF32;

impl KernelSpec for GdnNormActF32 {
    const ENTRY: &'static str = "owl_gdn_norm_act_f32";
    const SRC: &'static str = "cu/gdn/gdn_norm_act.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 2),
        Param::t_in("z", Dtype::F32, 2),
        Param::t_in("gamma", Dtype::F32, 1),
        Param::sz("rows"),
        Param::sz("value_dim"),
        Param::sz("group_size"),
        Param::f32("eps"),
        Param::i32("act"),
        Param::t_out("out", Dtype::F32, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static OWL_GDN_CONV_UPD_F16: KernelMeta = meta_of::<GdnConvUpdF16>();
pub static OWL_GDN_CONV_UPD_F32: KernelMeta = meta_of::<GdnConvUpdF32>();
pub static OWL_GDN_DELTA_DEC_F16: KernelMeta = meta_of::<GdnDeltaDecF16>();
pub static OWL_GDN_DELTA_DEC_F32: KernelMeta = meta_of::<GdnDeltaDecF32>();
pub static OWL_GDN_DECODE_STEP_F16: KernelMeta = meta_of::<GdnDecodeStepF16>();
pub static OWL_GDN_GATING_G_F32: KernelMeta = meta_of::<GdnGatingGF32>();
pub static OWL_GDN_L2NORM_F32: KernelMeta = meta_of::<GdnL2normF32>();
pub static OWL_GDN_NORM_ACT_F32: KernelMeta = meta_of::<GdnNormActF32>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[
    &GATING_G_F16,
    &L2NORM_F16,
    &NORM_ACT_F16,
    &CONV_FWD_F16,
    &CONV_UPD_DUAL_F16,
    &RECURRENCE_VARLEN_GQA_F16,
    &DECODE_STEP_V2_F16,
    &OWL_GDN_CONV_UPD_F16,
    &OWL_GDN_CONV_UPD_F32,
    &OWL_GDN_DELTA_DEC_F16,
    &OWL_GDN_DELTA_DEC_F32,
    &OWL_GDN_DECODE_STEP_F16,
    &OWL_GDN_GATING_G_F32,
    &OWL_GDN_L2NORM_F32,
    &OWL_GDN_NORM_ACT_F32,
];
