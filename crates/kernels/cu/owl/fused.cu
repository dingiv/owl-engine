// ============================================================================
// owl 融合核(C1;2026-10-01 立项)
//
// Ampere-first 准则(driver 立项同步确立):
//   ① 最小发射数优先于最小流量 —— 一次 DRAM 往返,中间值全寄存器;
//   ② 128-bit 向量化访问(half2/uint4);
//   ③ grid 铺满 SM(按元素/按 (t,head) 开,不整行串行);
//   ④ nvrtc compute_86 + float 中间精度(与被替核同式,口径可对拍)。
//
// 本文件核族(全部单输出,SSA 契约友好):
//   owl_norm_rope_f16     —— qk-norm(×(1+w) / ×w)+ rotate-half partial rope
//                            三发合一(narrow+norm+rope;strided 读 q_raw 的
//                            per-head [value|gate] 半段,narrow 视图消失)
//   owl_silu_and_mul_f16  —— SwiGLU 门控 silu(g)⊙u 双输入单输出
//                            (port 语义 = vLLM silu_and_mul;dry_kernels
//                            挂账兑现,mlp.rs 头注 2026-09-26 立项)
// ============================================================================

// ---- nvrtc 序言(同 ops_pair 族:cuda_fp16 提供 __half;float 中间)----
#include <cuda_fp16.h>

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
// owl_silu_and_mul_f16:out = silu(g) ⊙ u(逐元素;SwiGLU 门控)。
// half2 向量化(n 偶数;inter 全偶保证);float 中间 —— 被替两发核
// (silu 写 half → mul 读 half)的中间量化在此消除(精度红利)。
// grid = ceil(n / 512);block 256(每线程 2 元素)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_silu_and_mul_f16(
    const __half* __restrict__ g,
    const __half* __restrict__ u,
    size_t n,
    __half* __restrict__ out)
{
    const size_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 < n) {
        const __half2 gv = *reinterpret_cast<const __half2*>(g + i);
        const __half2 uv = *reinterpret_cast<const __half2*>(u + i);
        const float2 gf = __half22float2(gv);
        const float2 uf = __half22float2(uv);
        float2 r;
        r.x = (gf.x / (1.0f + expf(-gf.x))) * uf.x;
        r.y = (gf.y / (1.0f + expf(-gf.y))) * uf.y;
        *reinterpret_cast<__half2*>(out + i) = __floats2half2_rn(r.x, r.y);
    }
}
