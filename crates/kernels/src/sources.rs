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
    /// 刀3a'(2026-10-04):双权 GEMV 单发(b/a 投影;杀 cublas gemvx+splitK)
    pub const GEMV_DUAL_F16: &str = include_str!("../cu/owl/gemv_dual_f16.cu");
    /// 设备侧贪心采样(E3;REQ-DEC-04)
    pub const ARGMAX_F16: &str = include_str!("../cu/owl/argmax_f16.cu");
    // ---- 融合核族(C1;2026-10-01;Ampere-first 准则见 .cu 头注)----
    pub const NORM_ROPE_F16: &str = include_str!("../cu/owl/fused.cu");
    pub const SILU_AND_MUL_F16: &str = include_str!("../cu/owl/fused.cu");
    pub const FUSED_ADD_RMSNORM_F16: &str = include_str!("../cu/owl/fused.cu");
    pub const QKNORM_ROPE_KV_INSERT_F16: &str = include_str!("../cu/owl/fused.cu");
    /// ct packed → marlin B 设备重排(2026-10-01 装载提速;主源 =
    /// attention.rs marlin_repack.cu gptq_repack_kernel,输入侧转置适配)
    pub const CT_REPACK_U32: &str = include_str!("../cu/marlin_repack_ct.cu");
}

/// attention port 家族(K0 起步;vendor attention.rs rev c0f19f2,Apache-2.0,
/// 文件级复用 + f16 特化;出处与适配认领见各 .cu 头注)
pub mod attention {
    /// KV cache 散写(vLLM classic 布局;K1/K2 paged_attention 同款布局)
    pub const RESHAPE_AND_CACHE_F16: &str = include_str!("../cu/attention/reshape_and_cache.cu");
    pub const RESHAPE_AND_CACHE_DUAL_F16: &str = include_str!("../cu/attention/reshape_and_cache_dual.cu");
    /// f16<->f32 设备 cast(GDN chunked 编排配套)
    pub const CAST: &str = include_str!("../cu/owl/cast.cu");
    /// fp8 KV 变体(nvrtc 独立核文件;FI adapter 本体含 flashinfer 头不可 nvrtc)
    pub const RESHAPE_AND_CACHE_DUAL_F16_FP8KV: &str = include_str!("../cu/flashinfer/reshape_and_cache_dual_fp8kv.cu");
    /// paged attention decode 家族(v1 / v2 分片 / v2 reduce;K1/K2)
    pub const PAGED_ATTENTION_F16: &str = include_str!("../cu/attention/pagedattention_f16.cu");
    /// chunked prefill paged attention(在线 softmax + 滑窗;smem tile)
    pub const PREFILL_PAGED_ATTN_F16: &str = include_str!("../cu/attention/prefill_paged_attn_f16.cu");
    /// prefill split attention(flash-decoding 式 context 分块 + reduce;
    /// 2026-10-02 C1 自研序,长 ctx prefill 主案)
    pub const PREFILL_SPLIT_F16: &str = include_str!("../cu/attention/prefill_split_f16.cu");
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
    /// PF1a 栈核(concat_rows;arity 8,展开路径测试锚专用)
    pub const CONCAT_F32: &str = include_str!("../cu/text/concat.cu");
}
