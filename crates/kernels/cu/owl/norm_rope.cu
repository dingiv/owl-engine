// owl 融合核族:qk-norm + rotate-half partial rope 三发合一(2026-10-01 C1 立项;2026-10-13 C 轮自 cu/owl/fused.cu
// 按**算子族**拆出 —— 一算子一 `.cu`,与 ops/fused.rs / attn.rs 的声明对望)。
//
// Ampere-first 准则(driver 立项同步确立):
//   ① 最小发射数优先于最小流量 —— 一次 DRAM 往返,中间值全寄存器;
//   ② 128-bit 向量化访问(half2/uint4);
//   ③ grid 铺满 SM(按元素/按 (t,head) 开,不整行串行);
//   ④ float 中间精度(与被替核同式,口径可对拍)。
//
// 本文件核族(全部单输出,SSA 契约友好):
//   owl_norm_rope_f16 / _bf16 —— qk-norm(×(1+w) / ×w)+ rope 三发合一
//                          (narrow+norm+rope;strided 读 q_raw 的 per-head
//                          [value|gate] 半段,narrow 视图消失)
//   bf16 变体:x/w/out bf16;cos/sin 保持 f16 指针(rope 表与 target 共享)
//
// 输出块固定末参(槽序契约 4)。
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ----------------------------------------------------------------------------
// owl_norm_rope_f16:out[t,h,d] = rope( rmsnorm(x[t,h,·]) ×(1+w)^{w_off} )[d]
//
// x 寻址(strided;base + t·row_stride + h·head_stride + d)—— q 链:
//   base = q_raw [T, Hq·2HD](per-head [value|gate] 交错),row_stride =
//   Hq·2HD,head_stride = 2HD,value 半段 [0,HD);k 链:连续池
//   row_stride = Hkv·HD,head_stride = HD。
// 归一:per-(t,h) 行内 hd 元素 float 累计(与 owl_rmsnorm 同式);
// rope:rotate-half partial,配对 (d, d+half),[2·half, hd) 直通
//   (与 owl_rope_half_partial 逐式同源;float 中间精度同口径)。
// grid (tokens, heads, 1);block (hd, 1, 1)(d = threadIdx.x);
// smem = hd·4B(块内平方和归约)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_norm_rope_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ cos_t,   // [max_pos, half]
    const __half* __restrict__ sin_t,   // [max_pos, half]
    const float* __restrict__ pos,      // [tokens]
    float eps,
    size_t row_stride,
    size_t head_stride,
    size_t half,
    int w_off,
    __half* __restrict__ out)
{
    const size_t t = blockIdx.x;
    const size_t h = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const __half* xs = x + t * row_stride + h * head_stride;

    // pass 1:平方和(块内归约;float 累计,owl_rmsnorm 同式)
    extern __shared__ float smem[];
    const float xf = __half2float(xs[d]);
    smem[d] = xf * xf;
    __syncthreads();
    for (size_t s = hd / 2; s > 0; s >>= 1) {
        if (d < s) smem[d] += smem[d + s];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)hd + eps);

    // pass 2:归一 + rope 配对
    const float wf = __half2float(w[d]);
    const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    const size_t base = (t * gridDim.y + h) * hd;

    if (d < half) {
        // 配对 (d, d+half):伙伴侧 n 值就地重算(strided 读已入 L2)
        const float xb = __half2float(xs[d + half]);
        const float wb = __half2float(w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        out[base + d] = __float2half(n * cf - nb * sf);
        out[base + d + half] = __float2half(nb * cf + n * sf);
    } else if (d >= 2 * half) {
        out[base + d] = __float2half(n);   // partial 维直通
    }
}
// ----------------------------------------------------------------------------
// owl_norm_rope_bf16(E5-DF3 同日十四):x/w/out = BF16;cos/sin 保持 f16
// 指针(rope 表与 target 共享,免双表/免拷贝)。数学逐式同 f16 版。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_norm_rope_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    const __half* __restrict__ cos_t,   // [max_pos, half](f16 共享表)
    const __half* __restrict__ sin_t,   // [max_pos, half]
    const float* __restrict__ pos,      // [tokens]
    float eps,
    size_t row_stride,
    size_t head_stride,
    size_t half,
    int w_off,
    __nv_bfloat16* __restrict__ out)
{
    const size_t t = blockIdx.x;
    const size_t h = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const __nv_bfloat16* xs = x + t * row_stride + h * head_stride;

    extern __shared__ float smem[];
    const float xf = __bfloat162float(xs[d]);
    smem[d] = xf * xf;
    __syncthreads();
    for (size_t s = hd / 2; s > 0; s >>= 1) {
        if (d < s) smem[d] += smem[d + s];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)hd + eps);

    const float wf = __bfloat162float(w[d]);
    const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    const size_t base = (t * gridDim.y + h) * hd;

    if (d < half) {
        const float xb = __bfloat162float(xs[d + half]);
        const float wb = __bfloat162float(w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        out[base + d] = __float2bfloat16(n * cf - nb * sf);
        out[base + d + half] = __float2bfloat16(nb * cf + n * sf);
    } else if (d >= 2 * half) {
        out[base + d] = __float2bfloat16(n);   // partial 维直通
    }
}
