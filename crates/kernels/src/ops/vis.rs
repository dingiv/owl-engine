//! 视觉塔(Qwen3.5-VL ViT)生产算子元数据(2026-10-15 vision-tower 立项)。
//!
//! 一族对应 `cu/vis/`,`tests/op_tables.rs` 逐参互证,`driver::vis` 拾取。
//! 设计文档:`docs/arch/vision-tower-设计与CPU对拍.md`(§3 差分表)。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};
use owl_macros::kernel_tags;

/// `owl_layernorm_f32` —— LayerNorm(affine;two-pass mean/var,eps 入参)
pub struct LayernormF32;

#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.norm")]
impl KernelSpec for LayernormF32 {
    const ENTRY: &'static str = "owl_layernorm_f32";
    const SRC: &'static str = "cu/vis/layernorm.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 2),
        Param::t_in("weight", Dtype::F32, 1),
        Param::t_in("bias", Dtype::F32, 1),
        Param::t_out("out", Dtype::F32, 2),
        Param::i32("n"),
        Param::f32("eps"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_layernorm_f16` —— 同上(F16 域)
pub struct LayernormF16;

#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.norm")]
impl KernelSpec for LayernormF16 {
    const ENTRY: &'static str = "owl_layernorm_f16";
    const SRC: &'static str = "cu/vis/layernorm.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("weight", Dtype::F16, 1),
        Param::t_in("bias", Dtype::F16, 1),
        Param::t_out("out", Dtype::F16, 2),
        Param::i32("n"),
        Param::f32("eps"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_layernorm_bf16` —— 同上(BF16 域)
pub struct LayernormBf16;

#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.norm")]
impl KernelSpec for LayernormBf16 {
    const ENTRY: &'static str = "owl_layernorm_bf16";
    const SRC: &'static str = "cu/vis/layernorm.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::t_in("weight", Dtype::BF16, 1),
        Param::t_in("bias", Dtype::BF16, 1),
        Param::t_out("out", Dtype::BF16, 2),
        Param::i32("n"),
        Param::f32("eps"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static LAYERNORM_F32: KernelMeta = meta_of::<LayernormF32>();
pub static LAYERNORM_F16: KernelMeta = meta_of::<LayernormF16>();
pub static LAYERNORM_BF16: KernelMeta = meta_of::<LayernormBf16>();

// ============================================================================
// 批二(2026-10-15):Gelu / BiasAdd / PosEmbedBilinear / AttnFull
// 全部住 `cu/vis/vision_ops.cu`(Layernorm 单开 `cu/vis/layernorm.cu`)
// ============================================================================

/// `owl_gelu_tanh_*` —— GELU(tanh 近似;塔 MLP)
pub struct GeluTanhF32;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for GeluTanhF32 {
    const ENTRY: &'static str = "owl_gelu_tanh_f32";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 0),
        Param::t_out("out", Dtype::F32, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
pub struct GeluTanhF16;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for GeluTanhF16 {
    const ENTRY: &'static str = "owl_gelu_tanh_f16";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 0),
        Param::t_out("out", Dtype::F16, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
pub struct GeluTanhBf16;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for GeluTanhBf16 {
    const ENTRY: &'static str = "owl_gelu_tanh_bf16";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 0),
        Param::t_out("out", Dtype::BF16, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_gelu_erf_*` —— GELU(精确 erf;**merger 专用**,与 tanh 版不可互替)
pub struct GeluErfF32;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for GeluErfF32 {
    const ENTRY: &'static str = "owl_gelu_erf_f32";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 0),
        Param::t_out("out", Dtype::F32, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
pub struct GeluErfF16;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for GeluErfF16 {
    const ENTRY: &'static str = "owl_gelu_erf_f16";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 0),
        Param::t_out("out", Dtype::F16, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
pub struct GeluErfBf16;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for GeluErfBf16 {
    const ENTRY: &'static str = "owl_gelu_erf_bf16";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 0),
        Param::t_out("out", Dtype::BF16, 0),
        Param::sz("n"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_bias_add_*` —— 行广播加(grid = (rows,1,1);cols 为标量)
pub struct BiasAddF32;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for BiasAddF32 {
    const ENTRY: &'static str = "owl_bias_add_f32";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F32, 2),
        Param::t_in("bias", Dtype::F32, 1),
        Param::t_out("out", Dtype::F32, 2),
        Param::sz("cols"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
pub struct BiasAddF16;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for BiasAddF16 {
    const ENTRY: &'static str = "owl_bias_add_f16";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("bias", Dtype::F16, 1),
        Param::t_out("out", Dtype::F16, 2),
        Param::sz("cols"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
pub struct BiasAddBf16;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for BiasAddBf16 {
    const ENTRY: &'static str = "owl_bias_add_bf16";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::t_in("bias", Dtype::BF16, 1),
        Param::t_out("out", Dtype::BF16, 2),
        Param::sz("cols"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_pos_embed_bilinear_*` —— 4-tap 重采样(out[n,d] = Σ_k wts·table[idx])
pub struct PosEmbedF32;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.rope")]
impl KernelSpec for PosEmbedF32 {
    const ENTRY: &'static str = "owl_pos_embed_bilinear_f32";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("table", Dtype::F32, 2),
        Param::t_in("idx", Dtype::F32, 2),
        Param::t_in("wts", Dtype::F32, 2),
        Param::t_out("out", Dtype::F32, 2),
        Param::sz("n_taps"),
        Param::sz("dim"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
pub struct PosEmbedF16;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.rope")]
impl KernelSpec for PosEmbedF16 {
    const ENTRY: &'static str = "owl_pos_embed_bilinear_f16";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("table", Dtype::F16, 2),
        Param::t_in("idx", Dtype::F32, 2),
        Param::t_in("wts", Dtype::F32, 2),
        Param::t_out("out", Dtype::F16, 2),
        Param::sz("n_taps"),
        Param::sz("dim"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_vis_attn_full_*` —— 非因果 varlen 全注意力(逐图;hd=72;无 GQA)
pub struct AttnFullF32;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.attn")]
impl KernelSpec for AttnFullF32 {
    const ENTRY: &'static str = "owl_vis_attn_full_f32";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F32, 3),
        Param::t_in("k", Dtype::F32, 3),
        Param::t_in("v", Dtype::F32, 3),
        Param::t_in("cu_seqlens", Dtype::F32, 1),
        Param::sz("rows"),
        Param::sz("heads"),
        Param::sz("head_dim"),
        Param::i32("n_seq"),
        Param::f32("scale"),
        Param::t_out("out", Dtype::F32, 3),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
pub struct AttnFullF16;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.attn")]
impl KernelSpec for AttnFullF16 {
    const ENTRY: &'static str = "owl_vis_attn_full_f16";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 3),
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        Param::t_in("cu_seqlens", Dtype::F32, 1),
        Param::sz("rows"),
        Param::sz("heads"),
        Param::sz("head_dim"),
        Param::i32("n_seq"),
        Param::f32("scale"),
        Param::t_out("out", Dtype::F16, 3),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_image_scatter_*` —— VLM 图像 token 注入(out[t] = idx[t] ≥ 0 ? merged[idx[t]] : h[t])
pub struct ImageScatterF32;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for ImageScatterF32 {
    const ENTRY: &'static str = "owl_image_scatter_f32";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("h", Dtype::F32, 2),
        Param::t_in("merged", Dtype::F32, 2),
        Param::t_in("idx", Dtype::F32, 1),
        Param::t_out("out", Dtype::F32, 2),
        Param::sz("dim"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
pub struct ImageScatterF16;
#[kernel_tags("dom.vis", "feat.vision", "phase.prefill", "role.elem")]
impl KernelSpec for ImageScatterF16 {
    const ENTRY: &'static str = "owl_image_scatter_f16";
    const SRC: &'static str = "cu/vis/vision_ops.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("h", Dtype::F16, 2),
        Param::t_in("merged", Dtype::F16, 2),
        Param::t_in("idx", Dtype::F32, 1),
        Param::t_out("out", Dtype::F16, 2),
        Param::sz("dim"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static GELU_TANH_F32: KernelMeta = meta_of::<GeluTanhF32>();
pub static GELU_TANH_F16: KernelMeta = meta_of::<GeluTanhF16>();
pub static GELU_TANH_BF16: KernelMeta = meta_of::<GeluTanhBf16>();
pub static GELU_ERF_F32: KernelMeta = meta_of::<GeluErfF32>();
pub static GELU_ERF_F16: KernelMeta = meta_of::<GeluErfF16>();
pub static GELU_ERF_BF16: KernelMeta = meta_of::<GeluErfBf16>();
pub static BIAS_ADD_F32: KernelMeta = meta_of::<BiasAddF32>();
pub static BIAS_ADD_F16: KernelMeta = meta_of::<BiasAddF16>();
pub static BIAS_ADD_BF16: KernelMeta = meta_of::<BiasAddBf16>();
pub static POS_EMBED_F32: KernelMeta = meta_of::<PosEmbedF32>();
pub static POS_EMBED_F16: KernelMeta = meta_of::<PosEmbedF16>();
pub static ATTN_FULL_F32: KernelMeta = meta_of::<AttnFullF32>();
pub static ATTN_FULL_F16: KernelMeta = meta_of::<AttnFullF16>();
pub static IMAGE_SCATTER_F32: KernelMeta = meta_of::<ImageScatterF32>();
pub static IMAGE_SCATTER_F16: KernelMeta = meta_of::<ImageScatterF16>();
