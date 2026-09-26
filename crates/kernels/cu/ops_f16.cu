// owl kernels:f16 基础算子(f16 基线战役 F2;ops.cu 的 f16 对偶面)
//
// dtype 桥模式(照 nn-bak gdn_kernels 先例):读 __half → 算 float →
// 写 __half —— 存储 f16、计算 f32,不引入 f16 累计精度损失。
// nvrtc + cuda_fp16.h:nn-bak 同款先例(工具链自带 include 路径)。
//
// 纪律(A1.5):形状参数只来自声明;无同步、无 D2H、无 host 分支。
// matmul 无 f16 变体:f16 基线下 matmul 走 cuBLAS(eval dtype 路由,
// 见 eval.rs Op::Matmul 分支),不再养手写 GEMM。

#include <cuda_fp16.h>

extern "C" __global__ void owl_add_f16(
    const __half* a, const __half* b, __half* out, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __float2half(__half2float(a[i]) + __half2float(b[i])); }
}

extern "C" __global__ void owl_silu_f16(
    const __half* x, __half* out, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = __half2float(x[i]);
        out[i] = __float2half(v / (1.0f + expf(-v)));
    }
}

// 同形逐元素乘(MLP 门控 / 注意力输出门)
extern "C" __global__ void owl_mul_f16(
    const __half* a, const __half* b, __half* out, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = __float2half(__half2float(a[i]) * __half2float(b[i])); }
}

// 逐元素 sigmoid(attn_output_gate 门;GDN beta 同族)
extern "C" __global__ void owl_sigmoid_f16(
    const __half* x, __half* out, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(1.0f / (1.0f + expf(-__half2float(x[i]))));
    }
}

// rmsnorm(f16):读 half 转 float 累计/归约,写 half
// 语义与 ops.cu owl_rmsnorm_f32 逐式同源(x / rms(x) × (alpha + w_off))
extern "C" __global__ void owl_rmsnorm_f16(
    const __half* x, const __half* alpha, __half* out,
    const int n, const float eps, const int w_off) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const __half* xin = x + (size_t)row * n;
    __half* y = out + (size_t)row * n;

    float local = 0.0f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float v = __half2float(xin[c]);
        local += v * v;
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)n + eps);
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float a = w_off ? (__half2float(alpha[c]) + 1.0f) : __half2float(alpha[c]);
        y[c] = __float2half(__half2float(xin[c]) * inv * a);
    }
}
