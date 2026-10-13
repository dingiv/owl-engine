// ============================================================================
// GDN 核族:causal conv1d prefill varlen(k=4)(2026-10-13 C 轮自 cu/owl/gdn.cu 按**算子族**拆出 ——
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

// ---- causal conv1d prefill varlen f16(上游 causal_conv1d_fwd_varlen_
// kernel<T,4> 同体;owl 适配:state 槽寻址(上游 seq 直索,免 gather/scatter
// 垫)、去 state_snapshots(MTP 域外)、cu_seqlens [batch+1] u32 保真)----
// grid = (batch, ceil(d/256)) × (256,1,1);t 循环核内顺序扫描。
extern "C" __global__ void owl_gdn_conv_fwd_f16(
    const __half *__restrict__ x,            // [total_tokens, d]
    const __half *__restrict__ w,            // [d, 4]
    float *__restrict__ conv_state,          // [max_slots, d, 3](in/out 恒 f32)
    const float *__restrict__ slots,         // [batch](f32 数值;负 = 全跳)
    const float *__restrict__ cu_seqlens,    // [batch + 1](f32 数值过线,契约 5)
    int batch, int d, int silu,
    __half *__restrict__ out) {              // [total_tokens, d](末参 = 输出)
    const int seq_idx = blockIdx.x;
    int channel_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (seq_idx >= batch || channel_idx >= d) return;
    const int slot = (int)slots[seq_idx];
    const int start = (int)cu_seqlens[seq_idx];
    const int end = (int)cu_seqlens[seq_idx + 1];
    const int seq_len = end - start;
    const __half *w_ptr = w + channel_idx * 4;
    float *state_ptr = (slot >= 0)
        ? conv_state + ((size_t)slot * d + channel_idx) * 3
        : nullptr;
    float w_reg[4];
    for (int k = 0; k < 4; ++k) w_reg[k] = gdn16_to_float(w_ptr[k]);
    float hist[3] = {0.0f, 0.0f, 0.0f};
    if (state_ptr) {
        for (int i = 0; i < 3; ++i) hist[i] = state_ptr[i];
    }
    for (int t = 0; t < seq_len; ++t) {
        const float x_t = gdn16_to_float(x[(size_t)(start + t) * d + channel_idx]);
        float sum = x_t * w_reg[3];
        for (int k = 0; k < 3; ++k) sum = __fmaf_rn(hist[k], w_reg[k], sum);
        if (silu) sum /= (1.0f + __expf(-sum));
        out[(size_t)(start + t) * d + channel_idx] = gdn16_from_float(sum);
        hist[0] = hist[1];
        hist[1] = hist[2];
        hist[2] = x_t;
    }
    if (state_ptr) {
        for (int i = 0; i < 3; ++i) state_ptr[i] = hist[i];
    }
}
