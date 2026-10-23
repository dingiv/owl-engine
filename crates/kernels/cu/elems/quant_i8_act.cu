// ============================================================================
// owl 基础算子族 —— **per-token int8 激活量化**(W8A8 喂料手;2026-10-22)
//
// 语义:输入 x [rows=tokens, cols],每行独立:
//   sx = max|row| / 127(全零行 = 1)
//   q[i] = RN-even(x[i] / sx),clamp [-127,127]
//
// 与 `cu/load/quant_i8_row_w.cu` 的权重版**同式**,差别只有:
// - 本文件是 decode 热路径(tag phase.decode),输入 dtype 有 f16/bf16 两档;
// - 拆成 scale / q 两发,输出槽各自末位(与权重版同纪律)。
//
// 为什么需要:W8A8 head 的权重已经是 per-row int8;激活也按 token 量化后,
// 内积可走 `__dp4a`(int8×int8 → int32),权重流量不变、FMA 指令数 ÷4。
// 代价 = 每 token 一层 absmax 量化误差;质量门见 `tp2-decode-fast` §10.12。
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define OWL_QUANT_I8_ACT_WARPS 8

template <typename T> __device__ __forceinline__ float owl_qi8a_to_f(T x);
template <> __device__ __forceinline__ float owl_qi8a_to_f<__half>(__half x) {
    return __half2float(x);
}
template <> __device__ __forceinline__ float owl_qi8a_to_f<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename T>
__device__ __forceinline__ void owl_quant_i8_act_scale_body(
    const T* __restrict__ x, const int rows, const int cols,
    float* __restrict__ scale) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const T* xr = x + (size_t)row * cols;

    float amax = 0.0f;
    for (int i = tid; i < cols; i += (int)blockDim.x) {
        amax = fmaxf(amax, fabsf(owl_qi8a_to_f<T>(xr[i])));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, off));
    }
    __shared__ float wmax[OWL_QUANT_I8_ACT_WARPS];
    if (lane == 0) wmax[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        amax = (lane < OWL_QUANT_I8_ACT_WARPS) ? wmax[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, off));
        }
        if (lane == 0) scale[row] = amax > 0.0f ? amax / 127.0f : 1.0f;
    }
}

template <typename T>
__device__ __forceinline__ void owl_quant_i8_act_q_body(
    const T* __restrict__ x, const float* __restrict__ scale,
    const int rows, const int cols, signed char* __restrict__ q) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const T* xr = x + (size_t)row * cols;
    signed char* qr = q + (size_t)row * cols;
    const float s = scale[row];
    for (int i = tid; i < cols; i += (int)blockDim.x) {
        const float v = owl_qi8a_to_f<T>(xr[i]) / s;
        int qi = __float2int_rn(v);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        qr[i] = (signed char)qi;
    }
}

extern "C" __global__ void owl_quant_i8_act_scale_f16(
    const __half* __restrict__ x, const int rows, const int cols,
    float* __restrict__ scale) {
    owl_quant_i8_act_scale_body<__half>(x, rows, cols, scale);
}

extern "C" __global__ void owl_quant_i8_act_scale_bf16(
    const __nv_bfloat16* __restrict__ x, const int rows, const int cols,
    float* __restrict__ scale) {
    owl_quant_i8_act_scale_body<__nv_bfloat16>(x, rows, cols, scale);
}

extern "C" __global__ void owl_quant_i8_act_q_f16(
    const __half* __restrict__ x, const float* __restrict__ scale,
    const int rows, const int cols, signed char* __restrict__ q) {
    owl_quant_i8_act_q_body<__half>(x, scale, rows, cols, q);
}

extern "C" __global__ void owl_quant_i8_act_q_bf16(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ scale,
    const int rows, const int cols, signed char* __restrict__ q) {
    owl_quant_i8_act_q_body<__nv_bfloat16>(x, scale, rows, cols, q);
}
