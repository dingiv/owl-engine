// owl gdn kernels —— K3:GDN 线性注意力配套核(第一刀)
// 出处: vendor/attention.rs rev c0f19f2 src/kernels/src/gdn.cu
//       (fused_gdn_gating / l2_norm_last_dim / gated_rmsnorm_act_mul;
//        port 专项见 docs/arch/attention-kernel-port.md)
// owl 改编(2026-09-22, A4 所有权转移):
//   - nvrtc JIT 禁 host launcher(vLLM 原生 DISPATCH 宏是 host 端
//     <<<>>> 包装)——按 dtype 宏展开完整 __global__,launch 归 Rust(K1 同款);
//   - l2norm warp/block 双变体合并为单核,按 blockDim.x 分派
//     (Rust 侧已按 dim 选 grid/block);
//   - FP8 分支不搬(marlin-ffi 归属)。

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---- dtype 桥 ----
template <typename T> __device__ __forceinline__ float gdn_to_float(T x);
template <> __device__ __forceinline__ float gdn_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float gdn_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float gdn_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T gdn_from_float(float x);
template <> __device__ __forceinline__ float gdn_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half gdn_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 gdn_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ float gdn_silu(float x) { return x / (1.0f + __expf(-x)); }

// ---- 三核 × 三 dtype 宏展开(完整 __global__,无 launcher、无模板)----

#define GDN_GATING_KERNEL(T, SUFFIX)                                                   \
extern "C" __global__ void gdn_fused_gating_##SUFFIX(                                  \
    const float *__restrict__ a_log,                                                   \
    const T *__restrict__ a, const T *__restrict__ b,                                  \
    const float *__restrict__ dt_bias,                                                 \
    float *__restrict__ g, float *__restrict__ beta,                                   \
    int total_elements, int num_heads) {                                               \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                   \
    if (idx >= total_elements) return;                                                 \
    int h_idx = idx % num_heads;                                                       \
    const float a_f = gdn_to_float(a[idx]) + dt_bias[h_idx];                           \
    const float b_f = gdn_to_float(b[idx]);                                            \
    float sp = a_f;                                                                    \
    if (sp < 20.0f) sp = log1pf(expf(sp));                                             \
    g[idx] = -__expf(a_log[h_idx]) * sp;                                               \
    beta[idx] = 1.0f / (1.0f + __expf(-b_f));                                          \
}

#define GDN_L2NORM_WARP_KERNEL(T, SUFFIX)                                              \
extern "C" __global__ void gdn_l2norm_warp_##SUFFIX(                                   \
    const T *__restrict__ input, T *__restrict__ output,                               \
    int rows, int dim, float eps) {                                                    \
    const int row = blockIdx.x * 8 + (threadIdx.x / 32);                               \
    if (row >= rows) return;                                                           \
    const int lane_id = threadIdx.x % 32;                                              \
    const T *in_row = input + row * dim;                                               \
    T *out_row = output + row * dim;                                                   \
    float sumsq = 0.0f;                                                                \
    for (int i = lane_id; i < dim; i += 32) {                                          \
        const float v = gdn_to_float(in_row[i]);                                       \
        sumsq = __fmaf_rn(v, v, sumsq);                                                \
    }                                                                                  \
    for (int offset = 16; offset > 0; offset >>= 1)                                    \
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);                          \
    const float row_sumsq = __shfl_sync(0xffffffff, sumsq, 0);                         \
    const float inv = rsqrtf(fmaxf(row_sumsq, 0.0f) + eps);                            \
    for (int i = lane_id; i < dim; i += 32)                                            \
        out_row[i] = gdn_from_float<T>(gdn_to_float(in_row[i]) * inv);                 \
}

#define GDN_L2NORM_BLOCK_KERNEL(T, SUFFIX)                                             \
extern "C" __global__ void gdn_l2norm_block256_##SUFFIX(                               \
    const T *__restrict__ input, T *__restrict__ output,                               \
    int rows, int dim, float eps) {                                                    \
    const int row = blockIdx.x;                                                        \
    if (row >= rows) return;                                                           \
    const int tid = threadIdx.x;                                                       \
    const T *in_row = input + row * dim;                                               \
    T *out_row = output + row * dim;                                                   \
    float sumsq = 0.0f;                                                                \
    for (int i = tid; i < dim; i += 256) {                                             \
        const float v = gdn_to_float(in_row[i]);                                       \
        sumsq = __fmaf_rn(v, v, sumsq);                                                \
    }                                                                                  \
    for (int offset = 16; offset > 0; offset >>= 1)                                    \
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);                          \
    __shared__ float warp_sums[8];                                                     \
    const int warp_id = tid / 32;                                                      \
    const int lane_id = tid % 32;                                                      \
    if (lane_id == 0) warp_sums[warp_id] = sumsq;                                      \
    __syncthreads();                                                                   \
    float total = (tid < 8) ? warp_sums[tid] : 0.0f;                                   \
    if (warp_id == 0) {                                                                \
        for (int offset = 4; offset > 0; offset >>= 1)                                 \
            total += __shfl_down_sync(0xffffffff, total, offset);                      \
    }                                                                                  \
    if (tid == 0) warp_sums[0] = total;                                                \
    __syncthreads();                                                                   \
    const float inv = rsqrtf(fmaxf(warp_sums[0], 0.0f) + eps);                         \
    for (int i = tid; i < dim; i += 256)                                               \
        out_row[i] = gdn_from_float<T>(gdn_to_float(in_row[i]) * inv);                 \
}

#define GDN_RMSNORM_ACT_KERNEL(T, SUFFIX)                                              \
extern "C" __global__ void gdn_rmsnorm_act_##SUFFIX(                                   \
    const T *__restrict__ x, const T *__restrict__ z,                                  \
    const T *__restrict__ gamma, const T *__restrict__ bias,                           \
    T *__restrict__ out,                                                               \
    int rows, int value_dim, int group_size, float eps,                                \
    int per_group_weights, int has_bias, int act) {                                    \
    const int row_group = blockIdx.x;                                                  \
    const int num_groups = value_dim / group_size;                                     \
    const int row = row_group / num_groups;                                            \
    const int group = row_group % num_groups;                                          \
    const int tid = threadIdx.x;                                                       \
    if (row >= rows) return;                                                           \
    const T *x_row = x + row * value_dim;                                              \
    const T *z_row = z + row * value_dim;                                              \
    T *out_row = out + row * value_dim;                                                \
    const T *g_gamma = per_group_weights ? gamma + group * group_size : gamma;         \
    const T *g_bias = has_bias ? (per_group_weights ? bias + group * group_size : bias) \
                               : nullptr;                                              \
    float sumsq = 0.0f;                                                                \
    for (int i = tid; i < group_size; i += 256) {                                      \
        const float v = gdn_to_float(x_row[group * group_size + i]);                   \
        sumsq = __fmaf_rn(v, v, sumsq);                                                \
    }                                                                                  \
    for (int offset = 16; offset > 0; offset >>= 1)                                    \
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);                          \
    __shared__ float warp_sums[8];                                                     \
    const int warp_id = tid / 32;                                                      \
    const int lane_id = tid % 32;                                                      \
    if (lane_id == 0) warp_sums[warp_id] = sumsq;                                      \
    __syncthreads();                                                                   \
    float total = (tid < 8) ? warp_sums[tid] : 0.0f;                                   \
    if (warp_id == 0) {                                                                \
        for (int offset = 4; offset > 0; offset >>= 1)                                 \
            total += __shfl_down_sync(0xffffffff, total, offset);                      \
    }                                                                                  \
    if (tid == 0) warp_sums[0] = total;                                                \
    __syncthreads();                                                                   \
    const float inv = rsqrtf(fmaxf(warp_sums[0] / group_size, 0.0f) + eps);            \
    for (int i = tid; i < group_size; i += 256) {                                      \
        const int col = group * group_size + i;                                        \
        float nx = gdn_to_float(x_row[col]) * inv * gdn_to_float(g_gamma[i]);          \
        if (has_bias) nx += gdn_to_float(g_bias[i]);                                   \
        const float zv = gdn_to_float(z_row[col]);                                     \
        const float actv = (act == 0) ? gdn_silu(zv)                                   \
                                      : 1.0f / (1.0f + __expf(-zv));                   \
        out_row[col] = gdn_from_float<T>(nx * actv);                                   \
    }                                                                                  \
}

GDN_GATING_KERNEL(float, f32)
GDN_GATING_KERNEL(__half, f16)
GDN_GATING_KERNEL(__nv_bfloat16, bf16)

GDN_L2NORM_WARP_KERNEL(float, f32)
GDN_L2NORM_WARP_KERNEL(__half, f16)
GDN_L2NORM_WARP_KERNEL(__nv_bfloat16, bf16)
GDN_L2NORM_BLOCK_KERNEL(float, f32)
GDN_L2NORM_BLOCK_KERNEL(__half, f16)
GDN_L2NORM_BLOCK_KERNEL(__nv_bfloat16, bf16)

GDN_RMSNORM_ACT_KERNEL(float, f32)
GDN_RMSNORM_ACT_KERNEL(__half, f16)
GDN_RMSNORM_ACT_KERNEL(__nv_bfloat16, bf16)
