//! kernel 源码之家(纯 include_str;零依赖,零逻辑)。
//!
//! 纪律:**.cu 源码只允许住在本 crate**(`cu/` 目录)——models/layers、
//! TensorOps 组合面一律经 `owl_models::kernels` 注册表按名取用,禁止
//! 直接内嵌/直引源码(2026-09-25 用户裁决:kernel 源与业务层解耦,
//! 中间垫 = models::kernels 注册表)。
//!
//! 后端语义:cu/text/ = 文本主干(Qwen3.5 mini-demo);cu/ops.cu = 语义
//! 算子动作表母本;cu/gdn/ 等随各域立项迁入。launcher(nvrtc 编译/发射)
//! 在后端(owl-cuda server),本 crate 的 cuda feature(build.rs nvcc 预编
//! PTX 面)与源码之家无关。

/// 语义算子动作表母本(F5 模板统一:单源双 dtype 宏展开;matmul 除外
/// —— f16 走 cuBLAS foreign 通道,f32 手写核保留单元锚)
pub const OPS_F32: &str = include_str!("../cu/ops_pair.cu");
pub const OPS_F16: &str = include_str!("../cu/ops_pair.cu");

/// owl 移植位(工单 N;NInfer 等外部引擎核的 owl 契约改写)
pub mod owl {
    pub const SIGMOID_GATE_MUL_F16: &str = include_str!("../cu/owl/sigmoid_gate_mul_f16.cu");
}

/// 文本主干 kernel(models layers 消费)
pub mod text {
    /// embedding 查表
    pub const EMBED_F32: &str = include_str!("../cu/text/embed_f32.cu");
    /// rope(interleaved partial)
    pub const ROPE_HALF_PARTIAL_F32: &str = include_str!("../cu/text/rope_f32.cu");
    /// full-attention(narrow 窄切物化 + naive decode slot 直排)
    pub const ATTENTION_F32: &str = include_str!("../cu/text/attention.cu");
    /// GDN 线性注意力(gating g 臂;后续批:l2norm/conv_upd/delta_dec/norm_act)
    pub const GDN_F32: &str = include_str!("../cu/text/gdn.cu");
}
