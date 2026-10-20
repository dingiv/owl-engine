// ============================================================================
// owl 逐元素乘常数(2026-10-20;GGUF 草稿 GEMM 的动态范围护栏)
//
// 为什么需要它:GGUF 块型 GEMM 的输出核写 `half*`(**F16 上限 65504**),
// 而 DFlash2 草稿的执行域是 BF16(BF16 才装得下残差流的动态范围)。草稿的
// GEMM 输出实测可达 ~6e4 ⇒ 直接 F16 落盘会饱和成 inf(静默 NaN 塌陷)。
//
// 用法 = **成对**的纯指数移位(2⁻ᵏ / 2ᵏ,k = 6):
//   x' = x · 2⁻⁶   (F16 输入侧,使 GEMM 输出 ≤ 65504)
//   y' = y_f16 → bf16 · 2⁶  (输出侧;2 的幂在 bf16 上**精确**,零舍入)
// 语义与"BF16 输出 GEMM"等价(相对精度仍由 F16 的 10-bit 尾数决定)。
//
// 槽序契约(注册表 params 同序;输出块固定末参):
//   T x, i32 n, f32 scale, T out
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void owl_scale_f16(const __half* __restrict__ x,
                                         int n,
                                         float scale,
                                         __half* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(__half2float(x[i]) * scale);
}

extern "C" __global__ void owl_scale_bf16(const __nv_bfloat16* __restrict__ x,
                                          int n,
                                          float scale,
                                          __nv_bfloat16* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(__bfloat162float(x[i]) * scale);
}
