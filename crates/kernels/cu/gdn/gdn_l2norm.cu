// ============================================================================
// GDN 核族:末维 L2 归一(2026-10-13 C 轮自 cu/owl/gdn.cu 按**算子族**拆出 ——
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
//
// ---- **2026-10-19 工作映射重排:一 warp 一行,8 行/块** --------------------
// 旧形 = **一 block 一行** + 256 线程 + 共享归约(两次 `__syncthreads`):
//   `dim = 128`(GDN 的 hk/hv)时一半线程闲置,块级同步只为 128 个元素。
//   decode 实测该核 **15.5 µs/发**(864 发/窗口)而工作量只有 128 个元素
//   ⇒ 纯块调度/同步开销;与 `gdn_norm_act` 同款病灶(那里实测 239 GB/s = 带宽 1/4)。
// 新形:一 warp 一行、一 block 8 行(仍 256 线程)、`__shfl` 归约、零同步、
// 无共享内存 ⇒ 块数 ÷8、线程 ÷4。数学逐式不变(归约仍在 float 域,
// 仅求和顺序变 ⇒ 末位差 ~1e-7;金标门与 Determinism 律不受影响)。
// 发射维度见 driver.rs `KernelPick::l2norm`(grid = ceil(rows / 8))。
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

extern "C" __global__ void owl_gdn_l2norm_f32(
    const float *__restrict__ x,         // [rows, dim]
    size_t rows, size_t dim, float eps,
    float *__restrict__ out) {           // [rows, dim](末参 = 输出)
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int warp = threadIdx.x >> 5;
    const size_t lane = threadIdx.x & 31u;
    const size_t row = (size_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const float *in_row = x + row * dim;
    float *out_row = out + row * dim;
    float sumsq = 0.0f;
    for (size_t i = lane; i < dim; i += 32) {
        sumsq = __fmaf_rn(in_row[i], in_row[i], sumsq);
    }
    const float total = gdn16_warp_reduce_sum(sumsq);   // warp 内归约(无块级同步)
    const float inv = rsqrtf(fmaxf(total, 0.0f) + eps);
    for (size_t i = lane; i < dim; i += 32)
        out_row[i] = in_row[i] * inv;
}

// ---- 末维 L2 归一 f16(上游 l2_norm_last_dim_block_kernel<T,256> 同体)----
extern "C" __global__ void owl_gdn_l2norm_f16(
    const __half *__restrict__ x,        // [rows, dim]
    size_t rows, size_t dim, float eps,
    __half *__restrict__ out) {          // [rows, dim](末参 = 输出)
    const unsigned int warps_per_block = blockDim.x >> 5;
    const unsigned int warp = threadIdx.x >> 5;
    const size_t lane = threadIdx.x & 31u;
    const size_t row = (size_t)blockIdx.x * warps_per_block + warp;
    if (row >= rows) return;
    const __half *in_row = x + row * dim;
    __half *out_row = out + row * dim;
    float sumsq = 0.0f;
    for (size_t i = lane; i < dim; i += 32) {
        const float v = gdn16_to_float(in_row[i]);
        sumsq = __fmaf_rn(v, v, sumsq);
    }
    const float total = gdn16_warp_reduce_sum(sumsq);   // warp 内归约(无块级同步)
    const float inv = rsqrtf(fmaxf(total, 0.0f) + eps);
    for (size_t i = lane; i < dim; i += 32)
        out_row[i] = gdn16_from_float(gdn16_to_float(in_row[i]) * inv);
}
