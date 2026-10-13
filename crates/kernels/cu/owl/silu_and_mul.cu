// owl 融合核族:SwiGLU 门控 silu(g)⊙u(2026-10-01 C1 立项;2026-10-13 C 轮自 cu/owl/fused.cu
// 按**算子族**拆出 —— 一算子一 `.cu`,与 ops/fused.rs / elems.rs 的声明对望)。
//
// Ampere-first 准则(driver 立项同步确立):
//   ① 最小发射数优先于最小流量 —— 一次 DRAM 往返,中间值全寄存器;
//   ② 128-bit 向量化访问(half2/uint4);
//   ③ grid 铺满 SM(按元素/按 (t,head) 开,不整行串行);
//   ④ float 中间精度(与被替核同式,口径可对拍)。
//
// 本文件核族(全部单输出,SSA 契约友好):
//   owl_silu_and_mul_f16 / _bf16 —— silu(g)⊙u 双输入单输出
//                          (port 语义 = vLLM silu_and_mul)
//
// 输出块固定末参(槽序契约 4)。
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

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
// ----------------------------------------------------------------------------
// owl_silu_and_mul_bf16:全 bf16,逐式同 f16 版(float 中间;bf16 无 half2
// 便捷对 —— 逐元素标量处理,带宽非瓶颈位)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_silu_and_mul_bf16(
    const __nv_bfloat16* __restrict__ g,
    const __nv_bfloat16* __restrict__ u,
    size_t n,
    __nv_bfloat16* __restrict__ out)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float gf = __bfloat162float(g[i]);
        const float uf = __bfloat162float(u[i]);
        out[i] = __float2bfloat16((gf / (1.0f + expf(-gf))) * uf);
    }
}
