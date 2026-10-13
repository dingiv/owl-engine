// owl 语义算子族:RMSNorm(f32/f16/bf16)(2026-10-13 C 轮自 cu/owl/ops_pair.cu 按**算子族**拆出
// —— 一算子一 `.cu`,与 ops/ln.rs 的 `owl_rmsnorm_*` 声明对望)。
//
// 宏展开模式(nvrtc 铁律:禁 host launcher/禁模板具名实例 —— 模板同单元
// 双实例会撞 PTX 入口名,server 按 load_function(name) 找核)。
// 实例行**自带类型化形参**(宏第三参 PARAMS)—— C2 签名互证按实例行解析
// (宏定义不携带类型,机器对账以实例为准)。
//
// 规则:读 T → to_float → float 计算 → from_float 写 T(存储换档,计算
// 精度不换档 —— float 累计)。命名统一前缀 owl_;**输出块固定末参**
// (槽序契约 4);标量 size_t/i32 与 Arg::U64/I32 严格对位。

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---- dtype 桥(三 dtype;float 为恒等)----
template <typename T> __device__ __forceinline__ float owl_to_float(T x);
template <> __device__ __forceinline__ float owl_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float owl_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float owl_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ float owl_from_float(float x);
template <> __device__ __forceinline__ float owl_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float owl_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ float owl_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// ---- rmsnorm:x / rms(x) × (alpha + w_off)(块内归约,float 累计)----
#define OWL_RMSNORM_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    extern __shared__ float smem[]; \
    const int row = blockIdx.x; \
    const T *xin = x + (size_t)row * n; \
    T *y = out + (size_t)row * n; \
    float local = 0.0f; \
    for (int c = threadIdx.x; c < n; c += blockDim.x) { \
        const float v = owl_to_float(xin[c]); \
        local += v * v; \
    } \
    smem[threadIdx.x] = local; \
    __syncthreads(); \
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { \
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; } \
        __syncthreads(); \
    } \
    const float inv = rsqrtf(smem[0] / (float)n + eps); \
    for (int c = threadIdx.x; c < n; c += blockDim.x) { \
        const float a = w_off ? (owl_to_float(alpha[c]) + 1.0f) : owl_to_float(alpha[c]); \
        y[c] = owl_from_float<T>(owl_to_float(xin[c]) * inv * a); \
    } \
}


OWL_RMSNORM_KERNEL(owl_rmsnorm_f32, float, (const float* x, const float* alpha, float* out, const int n, const float eps, const int w_off))
OWL_RMSNORM_KERNEL(owl_rmsnorm_f16, __half, (const __half* x, const __half* alpha, __half* out, const int n, const float eps, const int w_off))
OWL_RMSNORM_KERNEL(owl_rmsnorm_bf16, __nv_bfloat16, (const __nv_bfloat16* x, const __nv_bfloat16* alpha, __nv_bfloat16* out, const int n, const float eps, const int w_off))
