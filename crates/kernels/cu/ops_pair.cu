// owl kernels:语义算子族(模板统一,F5 工单①;用户裁决"算子共用一个")
//
// **宏展开模式**(nvrtc 铁律:禁 host launcher/禁模板具名实例 —— 模板
// 同单元双实例会撞 PTX 入口名,server 按 load_function(name) 找核;
// 旧世界 GDN_GATING_KERNEL(T, SUFFIX) 同款,数学逐式同源)。
//
// 实例行**自带类型化形参**(第三参 PARAMS)—— C2 签名互证测试按实例行
// 解析(宏定义不携带具体类型,机器对账以实例为准)。
//
// 规则:读 T → to_float → float 计算 → from_float 写 T(存储换档,
// 计算精度不换档 —— float 累计)。to_float/from_float 三 dtype 桥
// (f32/f16/bf16;bf16 免费预留,模型 dtype 仍 f16,战役 §二)。
//
// matmul f32 在文件尾保留手写核(单元锚);f16 matmul 走 cuBLAS foreign
// 通道(eval dtype 路由),无手写变体。
//
// 命名统一前缀 owl_;输出块固定末参(槽序契约 4);标量 size_t/i32
// 与 Arg::U64/I32 严格对位(禁止 unsigned int 形参)。

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

__device__ __forceinline__ float owl_silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float owl_sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---- 同形逐元素二元(残差 add / 门控 mul)----
#define OWL_ADD_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) { out[i] = owl_from_float<T>(owl_to_float(a[i]) + owl_to_float(b[i])); } \
}
#define OWL_MUL_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) { out[i] = owl_from_float<T>(owl_to_float(a[i]) * owl_to_float(b[i])); } \
}

// ---- silu / sigmoid(逐元素一元)----
#define OWL_SILU_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) { out[i] = owl_from_float<T>(owl_silu_f(owl_to_float(x[i]))); } \
}
#define OWL_SIGMOID_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) { out[i] = owl_from_float<T>(owl_sigmoid_f(owl_to_float(x[i]))); } \
}

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

// ---- 实例化(f32 / f16;实例行第三参 = 类型化形参,C2 互证按行解析)----
OWL_ADD_KERNEL(owl_add_f32, float, (const float* a, const float* b, float* out, const size_t n))
OWL_ADD_KERNEL(owl_add_f16, __half, (const __half* a, const __half* b, __half* out, const size_t n))
OWL_MUL_KERNEL(owl_mul_f32, float, (const float* a, const float* b, float* out, const size_t n))
OWL_MUL_KERNEL(owl_mul_f16, __half, (const __half* a, const __half* b, __half* out, const size_t n))
OWL_SILU_KERNEL(owl_silu_f32, float, (const float* x, float* out, const size_t n))
OWL_SILU_KERNEL(owl_silu_f16, __half, (const __half* x, __half* out, const size_t n))
OWL_SIGMOID_KERNEL(owl_sigmoid_f32, float, (const float* x, float* out, const size_t n))
OWL_SIGMOID_KERNEL(owl_sigmoid_f16, __half, (const __half* x, __half* out, const size_t n))
OWL_RMSNORM_KERNEL(owl_rmsnorm_f32, float, (const float* x, const float* alpha, float* out, const int n, const float eps, const int w_off))
OWL_RMSNORM_KERNEL(owl_rmsnorm_f16, __half, (const __half* x, const __half* alpha, __half* out, const int n, const float eps, const int w_off))

// ---- matmul f32 手写核(单元锚保留;[m,k]×[k,n] 行主序,grid 二维)----
// f16 基线 matmul 走 cuBLAS foreign 通道(eval dtype 路由),不养手写变体。
extern "C" __global__ void owl_matmul_f32(
    const float* a, const float* b, float* out,
    const int m, const int k, const int n) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < m && c < n) {
        float acc = 0.0f;
        for (int p = 0; p < k; p++) {
            acc += a[(size_t)r * k + p] * b[(size_t)p * n + c];
        }
        out[(size_t)r * n + c] = acc;
    }
}

// nt 变体:B 按 [n, k] 行主序直读(lm_head/tied embedding 形态;
// 权重保持 checkpoint 原布局,免 host 转置与第二份显存)。
extern "C" __global__ void owl_matmul_nt_f32(
    const float* a, const float* b, float* out,
    const int m, const int k, const int n) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (r < m && c < n) {
        const float* brow = b + (size_t)c * k;
        float acc = 0.0f;
        for (int p = 0; p < k; p++) {
            acc += a[(size_t)r * k + p] * brow[p];
        }
        out[(size_t)r * n + c] = acc;
    }
}
