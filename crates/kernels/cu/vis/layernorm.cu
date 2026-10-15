// owl 视觉塔算子族:LayerNorm(affine;f32/f16/bf16)(2026-10-15 vision-tower 第一刀)。
//
// 与 `ops/vis.rs` 的 `owl_layernorm_*` 声明对望(参数序逐字对齐,机器门 =
// `tests/op_tables.rs`)。
//
// 为什么单开一族:主干只有 RMSNorm(`owl_rmsnorm_*`,×(1+w) 语义、无均值、无
// bias);Qwen3.5-VL 视觉塔 27 层 × 2 个 norm 全是 **LayerNorm(affine, eps=1e-6)**
// —— 去均值 + 有 bias,不能靠喂参绕过(见 docs/arch/vision-tower-设计与CPU对拍.md §1.2 #5)。
//
// 数值口径(two-pass,与 torch.nn.LayerNorm 对齐):
//   ① mean = Σx / n(先归约一次)
//   ② var  = Σ(x-mean)² / n(biased;第二遍归约)⇒ 不做 E[x²]-mean²(抵消风险)
//   ③ y = (x-mean)·rsqrt(var+eps)·weight + bias
// 全链 float 累计(读 T → float → 写 T);**输出块固定末参**(槽序契约 4);
// 标量 i32/f32 与 Arg::I32/F32 严格对位。

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// dtype 桥(本 TU 私有;`owl_to_float` 家族名已被 rmsnorm.cu 占用 ⇒ 前缀分域)
template <typename T> __device__ __forceinline__ float vis_to_float(T x);
template <> __device__ __forceinline__ float vis_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float vis_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float vis_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T vis_from_float(float x);
template <> __device__ __forceinline__ float vis_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half vis_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 vis_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// 块内求和归约(结果落 smem[0];两次调用之间调用方自行 __syncthreads)
#define VIS_ROW_REDUCE(LOAD_EXPR)                                                  \
    {                                                                              \
        float local = 0.0f;                                                        \
        for (int c = threadIdx.x; c < n; c += blockDim.x) { local += (LOAD_EXPR); } \
        smem[threadIdx.x] = local;                                                 \
        __syncthreads();                                                           \
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {                             \
            if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }    \
            __syncthreads();                                                       \
        }                                                                          \
    }

#define OWL_LAYERNORM_KERNEL(NAME, T, PARAMS)                                              \
extern "C" __global__ void NAME PARAMS {                                                   \
    extern __shared__ float smem[];                                                        \
    const int row = blockIdx.x;                                                            \
    const T *xin = x + (size_t)row * n;                                                    \
    T *y = out + (size_t)row * n;                                                          \
    if (n <= 0) { return; }                                                                \
    VIS_ROW_REDUCE(vis_to_float(xin[c]))                                                   \
    const float mean = smem[0] / (float)n;                                                 \
    __syncthreads();                                                                       \
    VIS_ROW_REDUCE((vis_to_float(xin[c]) - mean) * (vis_to_float(xin[c]) - mean))           \
    const float inv = rsqrtf(smem[0] / (float)n + eps);                                     \
    for (int c = threadIdx.x; c < n; c += blockDim.x) {                                     \
        const float xn = (vis_to_float(xin[c]) - mean) * inv;                               \
        y[c] = vis_from_float<T>(xn * vis_to_float(weight[c]) + vis_to_float(bias[c]));     \
    }                                                                                      \
}

// ⚠️ 实例行**必须单行**(签名互证门 `cu_sig::cu_params_of` 逐行解析,见 op_tables.rs):
OWL_LAYERNORM_KERNEL(owl_layernorm_f32, float, (const float* x, const float* weight, const float* bias, float* out, const int n, const float eps))
OWL_LAYERNORM_KERNEL(owl_layernorm_f16, __half, (const __half* x, const __half* weight, const __half* bias, __half* out, const int n, const float eps))
OWL_LAYERNORM_KERNEL(owl_layernorm_bf16, __nv_bfloat16, (const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias, __nv_bfloat16* out, const int n, const float eps))
