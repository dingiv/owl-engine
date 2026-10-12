//! Norm / MLP 门控 生产算子元数据。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};

/// `owl_rmsnorm_f16` —— RMSNorm(宏实例化;alpha 权重 + `w_off` 半段偏移)
pub struct RmsnormF16;

impl KernelSpec for RmsnormF16 {
    const ENTRY: &'static str = "owl_rmsnorm_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/ops_pair.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("alpha", Dtype::F16, 1),
        Param::t_out("out", Dtype::F16, 2),
        Param::i32("n"),
        Param::f32("eps"),
        Param::i32("w_off"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_rmsnorm_bf16` —— 同上(BF16 域)
pub struct RmsnormBf16;

impl KernelSpec for RmsnormBf16 {
    const ENTRY: &'static str = "owl_rmsnorm_bf16";
    const SOURCE: &'static str = include_str!("../../cu/owl/ops_pair.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::t_in("alpha", Dtype::BF16, 1),
        Param::t_out("out", Dtype::BF16, 2),
        Param::i32("n"),
        Param::f32("eps"),
        Param::i32("w_off"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_fused_add_rmsnorm_f16` —— residual 原地 += mixed(副作用律)+ rmsnorm·w
pub struct FusedAddRmsnormF16;

impl KernelSpec for FusedAddRmsnormF16 {
    const ENTRY: &'static str = "owl_fused_add_rmsnorm_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/fused.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("mixed", Dtype::F16, 2),
        Param::t_inout("residual", Dtype::F16, 2),
        Param::t_in("w", Dtype::F16, 1),
        Param::f32("eps"),
        Param::sz("n"),
        Param::i32("w_off"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_fused_add_rmsnorm_bf16` —— 同上(BF16 域)
pub struct FusedAddRmsnormBf16;

impl KernelSpec for FusedAddRmsnormBf16 {
    const ENTRY: &'static str = "owl_fused_add_rmsnorm_bf16";
    const SOURCE: &'static str = include_str!("../../cu/owl/fused.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("mixed", Dtype::BF16, 2),
        Param::t_inout("residual", Dtype::BF16, 2),
        Param::t_in("w", Dtype::BF16, 1),
        Param::f32("eps"),
        Param::sz("n"),
        Param::i32("w_off"),
        Param::t_out("out", Dtype::BF16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_silu_and_mul_f16` —— SwiGLU 门控 silu(g)⊙u(双输入单输出)
pub struct SiluAndMulF16;

impl KernelSpec for SiluAndMulF16 {
    const ENTRY: &'static str = "owl_silu_and_mul_f16";
    const SOURCE: &'static str = include_str!("../../cu/owl/fused.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("g", Dtype::F16, 2),
        Param::t_in("u", Dtype::F16, 2),
        Param::sz("n"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_silu_and_mul_bf16` —— 同上(BF16 域)
pub struct SiluAndMulBf16;

impl KernelSpec for SiluAndMulBf16 {
    const ENTRY: &'static str = "owl_silu_and_mul_bf16";
    const SOURCE: &'static str = include_str!("../../cu/owl/fused.cu");
    const PARAMS: &'static [Param] = &[
        Param::t_in("g", Dtype::BF16, 2),
        Param::t_in("u", Dtype::BF16, 2),
        Param::sz("n"),
        Param::t_out("out", Dtype::BF16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static RMSNORM_F16: KernelMeta = meta_of::<RmsnormF16>();
pub static RMSNORM_BF16: KernelMeta = meta_of::<RmsnormBf16>();
pub static FUSED_ADD_RMSNORM_F16: KernelMeta = meta_of::<FusedAddRmsnormF16>();
pub static FUSED_ADD_RMSNORM_BF16: KernelMeta = meta_of::<FusedAddRmsnormBf16>();
pub static SILU_AND_MUL_F16: KernelMeta = meta_of::<SiluAndMulF16>();
pub static SILU_AND_MUL_BF16: KernelMeta = meta_of::<SiluAndMulBf16>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[
    &RMSNORM_F16,
    &RMSNORM_BF16,
    &FUSED_ADD_RMSNORM_F16,
    &FUSED_ADD_RMSNORM_BF16,
    &SILU_AND_MUL_F16,
    &SILU_AND_MUL_BF16,
];
