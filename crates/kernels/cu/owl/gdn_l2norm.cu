// ============================================================================
// GDN 核族:末维 L2 归一(block256 变体;一 block 一行)(2026-10-13 C 轮自 cu/owl/gdn.cu 按**算子族**拆出 ——
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

// ---- 末维 L2 归一(block256 变体;一 block 一行)----
//   y[r][i] = x[r][i] · rsqrt(sum(x[r][·]²) + eps)
// 与 HF l2norm(qwen3_5 modeling 内嵌定义,对齐 FLA)同式。
// 旧世界 warp/block 双变体合并:mini-demo dim(128/256) ≤ 256 走块内归约;
// 发射 = (rows,1,1) × (256,1,1)(行核,非哨兵)。

extern "C" __global__ void owl_gdn_l2norm_f32(
    const float *__restrict__ x,         // [rows, dim]
    size_t rows, size_t dim, float eps,
    float *__restrict__ out) {           // [rows, dim](末参 = 输出)
    const size_t row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const float *in_row = x + row * dim;
    float *out_row = out + row * dim;
    float sumsq = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        sumsq = __fmaf_rn(in_row[i], in_row[i], sumsq);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    __shared__ float warp_sums[8];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sumsq;
    __syncthreads();
    float total = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 4; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
    }
    if (tid == 0) warp_sums[0] = total;
    __syncthreads();
    const float inv = rsqrtf(fmaxf(warp_sums[0], 0.0f) + eps);
    for (size_t i = tid; i < dim; i += blockDim.x)
        out_row[i] = in_row[i] * inv;
}
// ---- 末维 L2 归一 f16(上游 l2_norm_last_dim_block_kernel<T,256> 同体)----
extern "C" __global__ void owl_gdn_l2norm_f16(
    const __half *__restrict__ x,        // [rows, dim]
    size_t rows, size_t dim, float eps,
    __half *__restrict__ out) {          // [rows, dim](末参 = 输出)
    const size_t row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const __half *in_row = x + row * dim;
    __half *out_row = out + row * dim;
    float sumsq = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        const float v = gdn16_to_float(in_row[i]);
        sumsq = __fmaf_rn(v, v, sumsq);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    __shared__ float warp_sums[8];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sumsq;
    __syncthreads();
    float total = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 4; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
    }
    if (tid == 0) warp_sums[0] = total;
    __syncthreads();
    const float inv = rsqrtf(fmaxf(warp_sums[0], 0.0f) + eps);
    for (size_t i = tid; i < dim; i += blockDim.x)
        out_row[i] = gdn16_from_float(gdn16_to_float(in_row[i]) * inv);
}
