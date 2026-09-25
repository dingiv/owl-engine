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

/// 语义算子动作表母本(add/mul/silu/matmul/rmsnorm;与 lower_* 一一对应)
pub const OPS_F32: &str = include_str!("../cu/ops.cu");

/// 文本主干 kernel(models layers 消费)
pub mod text {
    /// embedding 查表
    pub const EMBED_F32: &str = include_str!("../cu/text/embed_f32.cu");
    /// rope(interleaved partial)
    pub const ROPE_INTERLEAVED_F32: &str = include_str!("../cu/text/rope_f32.cu");
}
