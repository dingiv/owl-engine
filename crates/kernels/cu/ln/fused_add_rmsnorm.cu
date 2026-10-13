// owl 融合核族:residual 原地加 + RMSNorm 融合(2026-10-01 C1 立项;2026-10-13 C 轮自 cu/owl/fused.cu
// 按**算子族**拆出 —— 一算子一 `.cu`,与 ops/fused.rs / ln.rs 的声明对望)。
//
// Ampere-first 准则(driver 立项同步确立):
//   ① 最小发射数优先于最小流量 —— 一次 DRAM 往返,中间值全寄存器;
//   ② 128-bit 向量化访问(half2/uint4);
//   ③ grid 铺满 SM(按元素/按 (t,head) 开,不整行串行);
//   ④ float 中间精度(与被替核同式,口径可对拍)。
//
// 本文件核族(全部单输出,SSA 契约友好):
//   owl_fused_add_rmsnorm_f16 / _bf16 —— residual 原地 += mixed + rmsnorm·w
//                          (vLLM port;residual 为 InOut 槽)
//
// 输出块固定末参(槽序契约 4)。
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ----------------------------------------------------------------------------
// owl_fused_add_rmsnorm_f16(vLLM layernorm_kernels.cu fused_add_rms_norm port;
// 双原地语义 —— residual 块原地 += mixed(已知副作用,GDN conv_upd 同款
// 单流保序律),out = rmsnorm(residual)·w^{w_off} 单输出)。
// 替 decoder 层 [残差 add + post_ln rmsnorm] 两发;grid (rows,1,1),
// block (256,1,1),smem 256·4B(行归约;n = hidden 5120,行内分段循环)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_fused_add_rmsnorm_f16(
    const __half* __restrict__ mixed,       // mixer 输出(只读)
    __half* __restrict__ residual,          // in/out:+= mixed(原地副作用)
    const __half* __restrict__ w,           // [n]
    float eps,
    size_t n,
    int w_off,
    __half* __restrict__ out)               // [rows, n] = rmsnorm(residual)·w
{
    const size_t row = blockIdx.x;
    const size_t base = row * n;
    __half* r = residual + base;
    const __half* m = mixed + base;
    __shared__ float smem[256];

    float local = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(m[i]) + __half2float(r[i]);
        r[i] = __float2half(v);            // 原地写(回收前根已收割,单流序)
        local += v * v;
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (size_t s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) smem[threadIdx.x] += smem[threadIdx.x + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)n + eps);
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __half2float(r[i]);
        const float wf = __half2float(w[i]);
        out[base + i] = __float2half(v * inv * (w_off ? (wf + 1.0f) : wf));
    }
}
// ----------------------------------------------------------------------------
// owl_fused_add_rmsnorm_bf16:全 bf16(含 gamma;检查点原生 BF16)。
// residual 原地 += mixed + rmsnorm·w,逐式同 f16 版。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_fused_add_rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ mixed,       // mixer 输出(只读)
    __nv_bfloat16* __restrict__ residual,          // in/out:+= mixed(原地副作用)
    const __nv_bfloat16* __restrict__ w,           // [n]
    float eps,
    size_t n,
    int w_off,
    __nv_bfloat16* __restrict__ out)               // [rows, n] = rmsnorm(residual)·w
{
    const size_t row = blockIdx.x;
    const size_t base = row * n;
    __nv_bfloat16* r = residual + base;
    const __nv_bfloat16* m = mixed + base;
    __shared__ float smem[256];

    float local = 0.0f;
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(m[i]) + __bfloat162float(r[i]);
        r[i] = __float2bfloat16(v);            // 原地写(回收前根已收割,单流序)
        local += v * v;
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (size_t s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) smem[threadIdx.x] += smem[threadIdx.x + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)n + eps);
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(r[i]);
        const float wf = __bfloat162float(w[i]);
        out[base + i] = __float2bfloat16(v * inv * (w_off ? (wf + 1.0f) : wf));
    }
}
