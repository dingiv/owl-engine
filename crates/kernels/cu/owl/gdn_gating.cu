// ============================================================================
// GDN 核族:门控 g 臂(g[:, i] = -exp(A_log[h])·softplus(a+dt_bias))(2026-10-13 C 轮自 cu/owl/gdn.cu 按**算子族**拆出 ——
// 一算子一 `.cu`,与 ops/gdn.rs 的声明对望)。
//
// 溯源:repos/attention.rs/src/kernels/src/gdn.cu 模板体显式展开(nvrtc 禁
// host launcher;旧世界 GDN_GATING_KERNEL 宏模式同款)。
//
// dtype 律:激活 half → float 计算 → half 写;**rec/conv state 恒 f32**
// (上游同款 + HF/vLLM/xinfer 三方先例);slots 恒 f32 数值过线(契约 5,
// 上游 i64 已适配);输出块固定末参(槽序契约 4);标量 i32 与 Arg::I32
// 严格对位(禁止 unsigned int)。数学与 f32 版逐式同源(移植不失真由
// 金标对拍背书)。
// ============================================================================

#include <cuda_fp16.h>

__device__ __forceinline__ float gdn16_to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ __half gdn16_from_float(float x) { return __float2half(x); }
__device__ __forceinline__ float gdn16_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

// ---- GDN 门控 g 臂(Qwen3_5GatedDeltaNet 融合门控同式)----
//   g[i] = -exp(A_log[h]) · softplus(a[i] + dt_bias[h]),h = i % heads
// softplus 分支与 torch 一致(x < 20 走 log1p(exp x),否则直通)。
// a_log/dt_bias [heads] 恒 f32(checkpoint 直存);a [total] = [T, H] 展平。
// (beta 臂 = sigmoid(b),复用 owl_sigmoid_f32,不在本文件。)

extern "C" __global__ void owl_gdn_gating_g_f32(
    const float *__restrict__ a_log,     // [heads]
    const float *__restrict__ a,         // [total](= [T, H] 展平)
    const float *__restrict__ dt_bias,   // [heads]
    size_t total, size_t heads,
    float *__restrict__ g) {             // [total](末参 = 输出)
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t h = (size_t)(idx % heads);
    const float x = a[idx] + dt_bias[h];
    float sp = (x < 20.0f) ? log1pf(expf(x)) : x;
    g[idx] = -__expf(a_log[h]) * sp;
}
// ---- 门控 g 臂 f16(上游 fused_gdn_gating_kernel/compute_gating 的 g 臂;
// owl 单输出契约:beta = sigmoid(b) 走语义算子,M-b 改判)----
//   g[i] = -exp(A_log[h]) · softplus(a[i] + dt_bias[h]),h = i % heads
// softplus 分支沿上游(x <= 20 走 log1pf(exp x),否则直通)。
extern "C" __global__ void owl_gdn_gating_g_f16(
    const __half *__restrict__ a_log,    // [heads]
    const __half *__restrict__ a,        // [total](= [T, H] 展平)
    const __half *__restrict__ dt_bias,  // [heads]
    size_t total, size_t heads,
    __half *__restrict__ g) {            // [total](末参 = 输出)
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t h = (size_t)(idx % heads);
    const float x = gdn16_to_float(a[idx]) + gdn16_to_float(dt_bias[h]);
    const float sp = (x <= 20.0f) ? log1pf(expf(x)) : x;
    g[idx] = gdn16_from_float(-expf(gdn16_to_float(a_log[h])) * sp);
}
