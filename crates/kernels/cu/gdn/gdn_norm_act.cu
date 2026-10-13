// ============================================================================
// GDN 核族:门控 RMSNorm × act(z)(Qwen3_5RMSNormGated 同式)(2026-10-13 C 轮自 cu/owl/gdn.cu 按**算子族**拆出 ——
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

// ---- 门控 RMSNorm × act(z)(批5;Qwen3_5RMSNormGated 同式)----
//   y[i] = rmsnorm(x 组内)·gamma[i] · act(z[i])
// 一 block 一 (row, group);gamma [group_size] 组内共享(×w 非零中心:
// RMSNormGated weight=ones 初始化,与 qk-norm 零中心相反 —— 已 HF 实证);
// act:0 = silu(Qwen3.5),1 = sigmoid(Qwen4 预留);无 bias。
// 发射 = (rows·value_dim/group_size,1,1) × (256,1,1)。

extern "C" __global__ void owl_gdn_norm_act_f32(
    const float *__restrict__ x,         // [rows, value_dim]
    const float *__restrict__ z,         // [rows, value_dim]
    const float *__restrict__ gamma,     // [group_size](×w 语义)
    size_t rows, size_t value_dim, size_t group_size,
    float eps,
    int act,
    float *__restrict__ out) {           // [rows, value_dim](末参 = 输出)
    const size_t row_group = blockIdx.x;
    const size_t num_groups = value_dim / group_size;
    const size_t row = row_group / num_groups;
    const size_t group = row_group % num_groups;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) return;
    const size_t group_offset = row * value_dim + group * group_size;
    const float *x_row = x + group_offset;
    const float *z_row = z + group_offset;
    float *out_row = out + group_offset;
    float sumsq = 0.0f;
    for (size_t i = tid; i < group_size; i += blockDim.x) {
        sumsq = __fmaf_rn(x_row[i], x_row[i], sumsq);
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
    const float inv = rsqrtf(fmaxf(warp_sums[0] / (float)group_size, 0.0f) + eps);
    for (size_t i = tid; i < group_size; i += blockDim.x) {
        const float nx = x_row[i] * inv * gamma[i];
        const float zv = z_row[i];
        const float actv = (act == 0) ? (zv / (1.0f + __expf(-zv)))
                                      : (1.0f / (1.0f + __expf(-zv)));
        out_row[i] = nx * actv;
    }
}
// ---- 门控 RMSNorm × act(z) f16(上游 gated_rmsnorm_silu_mul_kernel
// <T,W,256> 同体;owl 简化:per_group_weights=true、无 bias)----
extern "C" __global__ void owl_gdn_norm_act_f16(
    const __half *__restrict__ x,        // [rows, value_dim]
    const __half *__restrict__ z,        // [rows, value_dim]
    const __half *__restrict__ gamma,    // [group_size](×w 非零中心)
    size_t rows, size_t value_dim, size_t group_size,
    float eps,
    int act,
    __half *__restrict__ out) {          // [rows, value_dim](末参 = 输出)
    const size_t row_group = blockIdx.x;
    const size_t num_groups = value_dim / group_size;
    const size_t row = row_group / num_groups;
    const size_t group = row_group % num_groups;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) return;
    const size_t group_offset = row * value_dim + group * group_size;
    const __half *x_row = x + group_offset;
    const __half *z_row = z + group_offset;
    __half *out_row = out + group_offset;
    float sumsq = 0.0f;
    for (size_t i = tid; i < group_size; i += blockDim.x) {
        const float v = gdn16_to_float(x_row[i]);
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
    const float inv = rsqrtf(fmaxf(warp_sums[0] / (float)group_size, 0.0f) + eps);
    for (size_t i = tid; i < group_size; i += blockDim.x) {
        const float nx = gdn16_to_float(x_row[i]) * inv * gdn16_to_float(gamma[i]);
        const float zv = gdn16_to_float(z_row[i]);
        const float actv = (act == 0) ? (zv / (1.0f + __expf(-zv)))
                                      : (1.0f / (1.0f + __expf(-zv)));
        out_row[i] = gdn16_from_float(nx * actv);
    }
    // (2026-10-04 清撤:D1 取证期残留的无条件设备端 printf —— 每次
    // norm_act 发射 3 行 trap+flush 串行化,基线测速含此税;核内调试
    // 打印纪律:取证完必须撤,不留无门控残留)
}
