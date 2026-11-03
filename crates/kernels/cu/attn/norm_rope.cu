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

// ----------------------------------------------------------------------------
// INT8 KV 专用:norm+rope 后直接量化,去掉 F16 中间写读。
//   Q: 每 (t,h) 块算 q8 + qscale,布局 [tokens*heads, hd](行连续)。
//   K: 每 (t,h) 块算完直接写 paged K int8 池 + k_scale(槽映射内联)。
// 量化约定与 quant_int8_qk.cu / reshape_and_cache_int8.cu 逐式一致:
//   Q clamp [-127,127],K clamp [-128,127];scale=max(absmax/127,1e-6)。
// ----------------------------------------------------------------------------
__device__ __forceinline__ int8_t q_i8_q(float x, float inv) {
    float q = x * inv;
    q = (q >= 0.0f) ? (q + 0.5f) : (q - 0.5f);
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    return static_cast<int8_t>(q);
}
__device__ __forceinline__ int8_t q_i8_k(float x, float inv) {
    float q = x * inv;
    q = (q >= 0.0f) ? (q + 0.5f) : (q - 0.5f);
    q = fminf(fmaxf(q, -128.0f), 127.0f);
    return static_cast<int8_t>(q);
}

extern "C" __global__ void owl_norm_rope_q_int8_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ cos_t,
    const __half* __restrict__ sin_t,
    const float* __restrict__ pos,
    float eps,
    size_t row_stride,
    size_t head_stride,
    size_t half,
    int w_off,
    float* __restrict__ qscale,
    int8_t* __restrict__ qout)
{
    const size_t t = blockIdx.x;
    const size_t h = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const __half* xs = x + t * row_stride + h * head_stride;

    extern __shared__ float smem[];
    const int warp = static_cast<int>(d >> 5);
    const int lane = static_cast<int>(d & 31);
    const int nwarp = static_cast<int>((hd + 31) >> 5);
    const float xf = __half2float(xs[d]);
    float ss = xf * xf;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    if (lane == 0) smem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < nwarp) ? smem[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) smem[32] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(smem[32] / (float)hd + eps);

    const float wf = __half2float(w[d]);
    const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    float v0 = 0.0f, v1 = 0.0f, amax = 0.0f;
    if (d < half) {
        const float xb = __half2float(xs[d + half]);
        const float wb = __half2float(w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        v0 = n * cf - nb * sf;
        v1 = nb * cf + n * sf;
        // 与旧链逐位一致:先落 F16(NormRope 输出),再统计/量化。
        v0 = __half2float(__float2half_rn(v0));
        v1 = __half2float(__float2half_rn(v1));
        amax = fmaxf(fabsf(v0), fabsf(v1));
    } else if (d >= 2 * half) {
        v0 = __half2float(__float2half_rn(n));
        amax = fabsf(v0);
    }

    float gmax[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    gmax[d / 32] = fmaxf(gmax[d / 32], fabsf(v0));
    if (d < half) gmax[(d + half) / 32] = fmaxf(gmax[(d + half) / 32], fabsf(v1));
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        float mm = gmax[g];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, off));
        if (lane == 0) smem[64 + warp * 8 + g] = mm;
    }
    __syncthreads();
    if (d < 8) {
        float mm = 0.0f;
        for (int w = 0; w < nwarp; ++w) mm = fmaxf(mm, smem[64 + w * 8 + d]);
        smem[192 + d] = mm;
    }
    __syncthreads();
    const float scale0 = fmaxf(smem[192 + d / 32] / 127.0f, 1e-6f);
    const float qinv0 = 1.0f / scale0;
    const float scale1 = fmaxf(smem[192 + (d + half) / 32] / 127.0f, 1e-6f);
    const float qinv1 = 1.0f / scale1;

    const size_t base = (t * gridDim.y + h) * hd;
    if (d < half) {
        qout[base + d] = q_i8_q(v0, qinv0);
        qout[base + d + half] = q_i8_q(v1, qinv1);
    } else if (d >= 2 * half) {
        qout[base + d] = q_i8_q(v0, qinv0);
    }
    if (d < 8) {
        qscale[(t * gridDim.y + h) * 8 + d] = fmaxf(smem[192 + d] / 127.0f, 1e-6f);
    }
}

extern "C" __global__ void owl_norm_rope_k_int8_write_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ w,
    const __half* __restrict__ cos_t,
    const __half* __restrict__ sin_t,
    const float* __restrict__ pos,
    float eps,
    size_t row_stride,
    size_t head_stride,
    size_t half,
    int w_off,
    const float* __restrict__ slots,
    int8_t* __restrict__ k_cache,
    float* __restrict__ k_scale,
    int num_heads,
    int head_size,
    int block_size,
    __half* __restrict__ out)
{
    const size_t t = blockIdx.x;
    const size_t h = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const int slot = static_cast<int>(slots[t]);
    if (slot < 0) {
        if (threadIdx.x == 0) out[0] = __float2half(0.0f);
        return;
    }
    const __half* xs = x + t * row_stride + h * head_stride;

    extern __shared__ float smem[];
    const int warp = static_cast<int>(d >> 5);
    const int lane = static_cast<int>(d & 31);
    const int nwarp = static_cast<int>((hd + 31) >> 5);
    const float xf = __half2float(xs[d]);
    float ss = xf * xf;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, off);
    if (lane == 0) smem[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < nwarp) ? smem[lane] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) smem[32] = v;
    }
    __syncthreads();
    const float inv = rsqrtf(smem[32] / (float)hd + eps);

    const float wf = __half2float(w[d]);
    const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    float v0 = 0.0f, v1 = 0.0f;
    float gmax[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (d < half) {
        const float xb = __half2float(xs[d + half]);
        const float wb = __half2float(w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        v0 = n * cf - nb * sf;
        v1 = nb * cf + n * sf;
        // 与旧链逐位一致:先落 F16(NormRope 输出),再统计/量化。
        v0 = __half2float(__float2half_rn(v0));
        v1 = __half2float(__float2half_rn(v1));
        gmax[(d) / 32] = fmaxf(gmax[(d) / 32], fabsf(v0));
        gmax[(d + half) / 32] = fmaxf(gmax[(d + half) / 32], fabsf(v1));
    } else if (d >= 2 * half) {
        v0 = __half2float(__float2half_rn(n));
        gmax[d / 32] = fmaxf(gmax[d / 32], fabsf(v0));
    }

#pragma unroll
    for (int g = 0; g < 8; ++g) {
        float mm = gmax[g];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            mm = fmaxf(mm, __shfl_xor_sync(0xffffffffu, mm, off));
        if (lane == 0) smem[64 + warp * 8 + g] = mm;
    }
    __syncthreads();
    if (d < 8) {
        float mm = 0.0f;
        for (int w = 0; w < nwarp; ++w) mm = fmaxf(mm, smem[64 + w * 8 + d]);
        smem[192 + d] = mm;
    }
    __syncthreads();
    const float scale0 = fmaxf(smem[192 + d / 32] / 127.0f, 1e-6f);
    const float qinv0 = 1.0f / scale0;
    // rotate-half 的 v1 落在 d+half,属于另一个 32-dim group,必须用自己的 scale。
    const float scale1 = fmaxf(smem[192 + (d + half) / 32] / 127.0f, 1e-6f);
    const float qinv1 = 1.0f / scale1;

    const int block = slot / block_size;
    const int offset = slot % block_size;
    const long long row_base =
        (static_cast<long long>(block) * block_size + offset) * num_heads * head_size +
        static_cast<long long>(h) * head_size;
    if (d < half) {
        k_cache[row_base + d] = q_i8_k(v0, qinv0);
        k_cache[row_base + d + half] = q_i8_k(v1, qinv1);
    } else if (d >= 2 * half) {
        k_cache[row_base + d] = q_i8_k(v0, qinv0);
    }
    if (d < 8) {
        k_scale[((static_cast<long long>(block) * block_size + offset) * num_heads + h) * 8 + d] =
            fmaxf(smem[192 + d] / 127.0f, 1e-6f);
    }
    if (d == 0) out[0] = __float2half(0.0f);
}
