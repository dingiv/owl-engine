// owl nn kernels —— ported from candle-kernels (repos/candle-gb/candle-kernels/src)
// 出处:
//   - binary: binary.cu + binary_op_macros.cuh(BINARY_OP 宏的 contiguous 路径)
//   - unary:  unary.cu(UNARY_OP 宏的 contiguous 路径)
//   - softmax: reduce.cu SOFTMAX_OP(ggml softmax 改编,块内归约)
//   - rmsnorm: reduce.cu RMSNORM_OP(ggml rmsnorm 改编,块内归约)
// owl 改编(charter 裁决 3①,2026-09-22):
//   - 参数只允许标量与裸指针,禁止 candle 式 `const size_t *info` 设备侧
//     形状数组(每次 launch htod 新地址,图捕获后悬空);
//   - contiguous-only;非连续输入由上层先 .contiguous()。
//   - softmax/rmsnorm 采用一 block 一行的朴素实现(正确性优先,
//     warp shuffle 优化留 M1)。

#include <cuda_fp16.h>
#include <math_constants.h>

// ---- binary (port: binary.cu BINARY_OP, contiguous) ----
#define OWL_BINARY_F32(FN_NAME, FUNC)                          \
extern "C" __global__ void FN_NAME(                            \
    const unsigned long long n,                                \
    const float *a, const float *b, float *out) {              \
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) { out[i] = FUNC; }                              \
}

OWL_BINARY_F32(owl_add_f32, a[i] + b[i])
OWL_BINARY_F32(owl_mul_f32, a[i] * b[i])

// ---- unary (port: unary.cu UNARY_OP, contiguous) ----
// silu: x * sigmoid(x)  (candle unary.cu silu_fwd)
#define OWL_UNARY_F32(FN_NAME, FUNC)                           \
extern "C" __global__ void FN_NAME(                            \
    const unsigned long long n,                                \
    const float *inp, float *out) {                            \
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) { float x = inp[i]; out[i] = FUNC; }            \
}

OWL_UNARY_F32(owl_silu_f32, x / (1.0f + expf(-x)))
OWL_UNARY_F32(owl_exp_f32, expf(x))
// gelu tanh 近似(candle unary.cu gelu_fwd)
OWL_UNARY_F32(owl_gelu_f32, 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x))))

// ---- softmax last-dim (port: reduce.cu SOFTMAX_OP, ggml softmax 改编) ----
// 一 block 一行:block_size = blockDim.x;两遍(max + sum)+ 共享内存归约。
extern "C" __global__ void owl_softmax_f32(
    const float *src, float *dst, const int n_cols) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const float *x = src + (size_t)row * n_cols;
    float *y = dst + (size_t)row * n_cols;

    // pass 1: row max
    float local_max = -CUDART_INF_F;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        local_max = fmaxf(local_max, x[col]);
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + s]); }
        __syncthreads();
    }
    const float max_val = smem[0];
    __syncthreads();

    // pass 2: exp + sum
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        const float v = expf(x[col] - max_val);
        y[col] = v;
        local_sum += v;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    const float sum = smem[0];

    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        y[col] /= sum;
    }
}

// ---- rmsnorm (port: reduce.cu RMSNORM_OP, ggml rmsnorm 改编) ----
// 一 block 一行:先算均方根,再缩放乘 alpha。
extern "C" __global__ void owl_rmsnorm_f32(
    const float *src, float *dst, const float *alpha,
    const int n_cols, const float eps) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const float *x = src + (size_t)row * n_cols;
    float *y = dst + (size_t)row * n_cols;

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        local_sum += x[col] * x[col];
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    const float rms = rsqrtf(smem[0] / (float)n_cols + eps);

    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        y[col] = x[col] * rms * alpha[col];
    }
}

// ---- f16 unary/binary(candle f16 路径,__half;容差放宽) ----
#define OWL_BINARY_F16(FN_NAME, FUNC)                          \
extern "C" __global__ void FN_NAME(                            \
    const unsigned long long n,                                \
    const __half *a, const __half *b, __half *out) {           \
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) { out[i] = FUNC; }                              \
}

OWL_BINARY_F16(owl_add_f16, __hadd(a[i], b[i]))
OWL_BINARY_F16(owl_mul_f16, __hmul(a[i], b[i]))

#define OWL_UNARY_F16(FN_NAME, FUNC)                           \
extern "C" __global__ void FN_NAME(                            \
    const unsigned long long n,                                \
    const __half *inp, __half *out) {                          \
    const unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) { __half x = inp[i]; out[i] = FUNC; }           \
}

OWL_UNARY_F16(owl_silu_f16, __hdiv(x, __float2half(1.0f + expf(-__half2float(x)))))

// ---- rope rotate-half f32(port 语义:llama/llama.cpp rotate-half 风格)----
// x: [rows, n_cols];pos: [rows](i64 token 位置);theta_base 标量。
// 对每行:half = n_cols/2;对 i < half:
//   inv_freq_i = theta^(-2i/n_cols);ang = pos * inv_freq_i
//   out[i]      = x[i]*cos(ang) - x[i+half]*sin(ang)
//   out[i+half] = x[i]*sin(ang) + x[i+half]*cos(ang)
// 动态量(pos)走 device 指针,标量只有 theta_base 与 n_cols(裁决 3①)。
extern "C" __global__ void owl_rope_f32(
    const float *x, const long long *pos, float *out,
    const int n_cols, const float theta_base) {
    const int half = n_cols / 2;
    const int row = blockIdx.x;
    const long long p = pos[row];
    const float *xr = x + (size_t)row * n_cols;
    float *our = out + (size_t)row * n_cols;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float inv_freq = powf(theta_base, -(2.0f * (float)i / (float)n_cols));
        const float ang = (float)p * inv_freq;
        const float c = cosf(ang), s = sinf(ang);
        const float x1 = xr[i], x2 = xr[i + half];
        our[i] = x1 * c - x2 * s;
        our[i + half] = x1 * s + x2 * c;
    }
}

// ---- embedding lookup(port 语义:table[ids[i]] 行拷贝)----
// table: [vocab, n_cols];ids: [rows](i32);out: [rows, n_cols]。
// 一 block 一行,行内并行拷贝。
extern "C" __global__ void owl_embedding_f32(
    const float *table, const int *ids, float *out, const int n_cols) {
    const int row = blockIdx.x;
    const int id = ids[row];
    const float *src = table + (size_t)id * n_cols;
    float *dst = out + (size_t)row * n_cols;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        dst[col] = src[col];
    }
}
