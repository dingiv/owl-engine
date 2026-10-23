//! KVMem Mean-K 索引的**设备块级归约**（2026-10-23；262K prefill 的 index 刀）。
//!
//! 取代 host 路径的「整页 K D2H → 逐 token 逆 RoPE → MeanKCollector」：
//! 每块只回读 `[layers][hkv][hd]` f16（27B 几何 = 32 KiB/块），池内 K
//! **不回读**。语义与 `MeanKCollector::push_post_rope_halves` 严格配对：
//!
//! ```text
//! 正变换: x' = x·c − y·s,  y' = x·s + y·c       (rotate-half, y = k[d+half])
//! 逆变换: x  = x'·c + y'·s, y = −x'·s + y'·c
//! 均值   : 逐 (layer, kv_head, dim) 对 page 个 token 求和 ÷ page
//! ```
//!
//! 配对 = **rotate-half**（`(d, d+half)`），不是 interleaved `(2d,2d+1)`；
//! 与 `owl_qknorm_rope_kv_insert_*_knhd` 的写侧同律。尾维（`d ≥ 2·half`）
//! 原样累加（partial rotary 直通）。
//!
//! 池布局 = kNHD（`[nb, page, hkv, hd]`；f16/e4m3 元素序相同）。核按
//! `phys[block]` 直接寻址主池页；`out` 布局 = `[block][layer][hkv][hd]`。
//!
//! grid = `(blocks, hkv)`，block = `(hd)`；每线程负责一个（或一对）dim，
//! 块内无归约、无共享内存。累加用 double（与 host 参考同序），出口 f16
//! 只用于排序。

#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" __global__ void owl_mean_k_block_f16(
    const __half* __restrict__ key_cache,   // [nb, page, hkv, hd]
    const float* __restrict__ cos_t,        // [blocks*page, half]
    const float* __restrict__ sin_t,        // [blocks*page, half]
    const unsigned int* __restrict__ phys,  // [blocks]
    int page, int hkv, int hd, int half, int layer, int layers,
    __half* __restrict__ out)               // [blocks, layers, hkv, hd]
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= hd) return;

    const size_t row = (size_t)hkv * (size_t)hd;
    const __half* src =
        key_cache + (size_t)phys[b] * (size_t)page * row + (size_t)h * (size_t)hd;
    const size_t ob =
        (((size_t)b * (size_t)layers + (size_t)layer) * (size_t)hkv + (size_t)h) *
        (size_t)hd;
    const size_t cstride = (size_t)half;

    if (d < half) {
        double acc0 = 0.0;
        double acc1 = 0.0;
        for (int t = 0; t < page; ++t) {
            const size_t off = (size_t)t * row + (size_t)d;
            const float x = __half2float(src[off]);
            const float y = __half2float(src[off + half]);
            const size_t ci = (size_t)b * (size_t)page + (size_t)t;
            const float c = cos_t[ci * cstride + (size_t)d];
            const float s = sin_t[ci * cstride + (size_t)d];
            // 显式非 FMA（与 host MeanKCollector 的 f32 `x*c + y*s` 逐位配对；
            // nvcc 缺省会缩成 fma，差 1 ulp f16）
            acc0 += (double)__fadd_rn(__fmul_rn(x, c), __fmul_rn(y, s));
            acc1 += (double)__fadd_rn(__fmul_rn(-x, s), __fmul_rn(y, c));
        }
        const double n = (double)page;
        out[ob + (size_t)d] = __float2half((float)(acc0 / n));
        out[ob + (size_t)d + half] = __float2half((float)(acc1 / n));
    } else if (d >= 2 * half) {
        double acc = 0.0;
        for (int t = 0; t < page; ++t) {
            acc += (double)__half2float(src[(size_t)t * row + (size_t)d]);
        }
        out[ob + (size_t)d] = __float2half((float)(acc / (double)page));
    }
    // d ∈ [half, 2·half) 是逆旋转的伙伴位，由 d−half 线程一并写出。
}

extern "C" __global__ void owl_mean_k_block_fp8(
    const unsigned char* __restrict__ key_cache,  // e4m3 [nb, page, hkv, hd]
    const float* __restrict__ cos_t,
    const float* __restrict__ sin_t,
    const unsigned int* __restrict__ phys,
    int page, int hkv, int hd, int half, int layer, int layers,
    __half* __restrict__ out)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (d >= hd) return;

    const size_t row = (size_t)hkv * (size_t)hd;
    const unsigned char* src =
        key_cache + (size_t)phys[b] * (size_t)page * row + (size_t)h * (size_t)hd;
    const size_t ob =
        (((size_t)b * (size_t)layers + (size_t)layer) * (size_t)hkv + (size_t)h) *
        (size_t)hd;
    const size_t cstride = (size_t)half;

    if (d < half) {
        double acc0 = 0.0;
        double acc1 = 0.0;
        for (int t = 0; t < page; ++t) {
            const size_t off = (size_t)t * row + (size_t)d;
            __nv_fp8_e4m3 xe;
            xe.__x = src[off];
            __nv_fp8_e4m3 ye;
            ye.__x = src[off + half];
            const float x = __half2float(__half(xe));
            const float y = __half2float(__half(ye));
            const size_t ci = (size_t)b * (size_t)page + (size_t)t;
            const float c = cos_t[ci * cstride + (size_t)d];
            const float s = sin_t[ci * cstride + (size_t)d];
            // 显式非 FMA（同 f16 核：与 host f32 运算逐位配对）
            acc0 += (double)__fadd_rn(__fmul_rn(x, c), __fmul_rn(y, s));
            acc1 += (double)__fadd_rn(__fmul_rn(-x, s), __fmul_rn(y, c));
        }
        const double n = (double)page;
        out[ob + (size_t)d] = __float2half((float)(acc0 / n));
        out[ob + (size_t)d + half] = __float2half((float)(acc1 / n));
    } else if (d >= 2 * half) {
        double acc = 0.0;
        for (int t = 0; t < page; ++t) {
            __nv_fp8_e4m3 xe;
            xe.__x = src[(size_t)t * row + (size_t)d];
            acc += (double)__half2float(__half(xe));
        }
        out[ob + (size_t)d] = __float2half((float)(acc / (double)page));
    }
}
