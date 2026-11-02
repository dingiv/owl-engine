// INT8 V prepack for 4-warp int8 PV.
//   v:      [T, Hkv, D] f16
//   vscale: [Hkv, D/64] f32, per (head, 64-dim group), constant along K
//   vp:     [Hkv, D, ceil(T/2)] b16, low byte = key0, high byte = key1
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kThreads = 256;

__device__ __forceinline__ int8_t quant_i8(float x, float inv) {
    float q = x * inv;
    q = (q >= 0.0f) ? (q + 0.5f) : (q - 0.5f);
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    return static_cast<int8_t>(q);
}

__device__ __forceinline__ float block_max(float x, float* smem) {
    smem[threadIdx.x] = x;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
        __syncthreads();
    }
    return smem[0];
}
}  // namespace

extern "C" __global__ void owl_v_scale_int8(const __half* __restrict__ v,
                                             int T, int Hkv, int D,
                                             float* __restrict__ vscale) {
    __shared__ float smem[kThreads];
    const int unit = blockIdx.x;
    const int chunks = gridDim.y;
    const int chunk = blockIdx.y;
    const int h = unit / 4;
    const int g = unit % 4;
    const int G = 64;
    const long long n = (long long)T * G;
    const long long stride = (long long)chunks * blockDim.x;
    float amax = 0.0f;
    for (long long i = (long long)chunk * blockDim.x + threadIdx.x; i < n; i += stride) {
        const int t = (int)(i / G);
        const int d = g * G + (int)(i % G);
        amax = fmaxf(amax, fabsf(__half2float(v[((long long)t * Hkv + h) * D + d])));
    }
    amax = block_max(amax, smem);
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<int*>(&vscale[unit]), __float_as_int(amax));
    }
}

extern "C" __global__ void owl_v_scale_finalize(float* __restrict__ vscale, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vscale[i] = fmaxf(vscale[i] / 127.0f, 1e-6f);
}

extern "C" __global__ void owl_pack_v_int8(const __half* __restrict__ v,
                                            int T, int Hkv, int D, int page,
                                            const float* __restrict__ vscale,
                                            uint16_t* __restrict__ vp) {
    const int PP = page / 2;
    const int npages = (T + page - 1) / page;
    const long long total = (long long)npages * Hkv * D * PP;
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int pair = (int)(idx % PP);
    const int d = (int)((idx / PP) % D);
    const int h = (int)((idx / ((long long)PP * D)) % Hkv);
    const int pg = (int)(idx / ((long long)PP * D * Hkv));
    const int key0 = pg * page + pair * 2;
    const int key1 = key0 + 1;
    const float inv = 1.0f / vscale[h * 4 + d / 64];
    const float x0 = (key0 < T) ? __half2float(v[((long long)key0 * Hkv + h) * D + d]) : 0.0f;
    const float x1 = (key1 < T) ? __half2float(v[((long long)key1 * Hkv + h) * D + d]) : 0.0f;
    const uint16_t a = static_cast<uint8_t>(quant_i8(x0, inv));
    const uint16_t b = static_cast<uint8_t>(quant_i8(x1, inv));
    vp[(((long long)pg * Hkv + h) * D + d) * PP + pair] = a | (b << 8);
}
