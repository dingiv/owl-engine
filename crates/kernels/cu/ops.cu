// owl kernels:f32 基础算子(声明式管线的 CPU/GPU 对偶面,GPU 侧)
//
// 溯源:与 playground/mlp 管线同族;host launcher 见 src/cuda_ops.rs。
// 纪律(A1.5 捕获契约):形状参数只来自声明,动态量从 device 读;
// 无同步、无 D2H、无 host 分支。
//
// 命名统一前缀 owl_。

extern "C" __global__ void owl_add_f32(
    const float* a, const float* b, float* out, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = a[i] + b[i]; }
}

extern "C" __global__ void owl_silu_f32(
    const float* x, float* out, const size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { out[i] = x[i] / (1.0f + expf(-x[i])); }
}

// [m,k] × [k,n] → [m,n](行主序;grid 二维:行 × 列)
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

// rmsnorm:x / rms(x) × (alpha + w_off)
// (w_off = ×(1+w) 语义,use_norm_offset;块内归约,candle reduce.cu 同构)
extern "C" __global__ void owl_rmsnorm_f32(
    const float* x, const float* alpha, float* out,
    const int n, const float eps, const int w_off) {
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    const float* xin = x + (size_t)row * n;
    float* y = out + (size_t)row * n;

    float local = 0.0f;
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        local += xin[c] * xin[c];
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) { smem[threadIdx.x] += smem[threadIdx.x + s]; }
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)n + eps);
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        const float a = w_off ? (alpha[c] + 1.0f) : alpha[c];
        y[c] = xin[c] * inv * a;
    }
}
