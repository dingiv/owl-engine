// ============================================================================
// DFlash2 草稿核族:分组动态深度 K-tap 卷积(_grouped_conv 逐式)(2026-10-13 C 轮自 cu/owl/dflash2.cu 按**算子族**
// 拆出 —— 一算子一 `.cu`,与 ops/dflash.rs / elems.rs 的声明对望)。
//
// 上游语义锚 = packages/sglang python/sglang/srt/models/dflash.py +
// kernels/ops/speculative/dflash.py(sglang-dflash-patch 分支,逐式对拍)。
//
// 数值律:输入 f16/bf16,累计 f32,输出随槽(契约 5:索引量 f32 过线;词表
// id < 2²⁴ 精确)。平局取最小索引(与 owl_argmax / torch.topk sorted 语义
// 一致,host 对拍同律)。输出块固定末参(槽序契约 4)。
//
//   blocks = x.unflatten(-1, (G, group))          # [T, G, group]
//   coeff[t,tap,g,s] = base[side,tap,g*group+s] + delta[t,side,tap,g]
//   out[t,g,s] = Σ_tap coeff[t,tap,g,s] · x[t-tap,g,s] · [t%block ≥ tap]
//
// taps=2 / group=16 由检查点家族定形(conv_kernel_size=2, conv_group_size=16)。
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

// ---------------------------------------------------------------------------
// 1) 分组动态深度 K-tap 卷积(sglang _grouped_conv 逐式)
//
//   blocks = x.unflatten(-1, (G, group))          # [T, G, group]
//   coeff[t,tap,g,s] = base[side,tap,g*group+s] + delta[t,side,tap,g]
//   out[t,g,s] = Σ_tap coeff[t,tap,g,s] · x[t-tap,g,s] · [t%block ≥ tap]
//
// delta = kernel_projection(x) 布局 [T, side(2), tap(2), G](side-major,
// sglang reshape 序);t<tap 时 x[t-tap] 记 0(且 t%block 掩码同清)。
// taps=2 / group=16 由检查点家族定形(conv_kernel_size=2, conv_group_size=16),
// block(T 行内块宽)/ G / H 参数化。
// ---------------------------------------------------------------------------
extern "C" __global__ void owl_dflash_conv_f16(
    const __half* __restrict__ x,      // [T, H]
    const __half* __restrict__ delta,  // [T, 2, 2, G]
    const __half* __restrict__ base,   // [2, 2, H]
    int side,                          // 0 = input 子层前,1 = output 子层后
    int block_size,                    // 位置掩码模数(T=8 时掩码 [0..7])
    int hidden,                        // H
    int groups,                        // G = H / group, group 烘焙 16
    int rows,                          // T(哨兵哨界 = T*H)
    __half* __restrict__ out) {        // [T, H]
    const int group = 16;
    const int total = blockIdx.x * blockDim.x + threadIdx.x;
    if (total >= rows * hidden) return;

    const int t = total / hidden;
    const int d = total % hidden;
    const int g = d / group;
    const int s = d % group;
    const int G = groups;

    const float d0 = __half2float(delta[((long long)t * 2 + side) * 2 * G + 0 * G + g]);
    const float d1 = __half2float(delta[((long long)t * 2 + side) * 2 * G + 1 * G + g]);
    const float b0 = __half2float(base[((long long)side * 2 + 0) * hidden + d]);
    const float b1 = __half2float(base[((long long)side * 2 + 1) * hidden + d]);
    const float c0 = d0 + b0;
    const float c1 = d1 + b1;

    const float x0 = __half2float(x[(long long)t * hidden + d]);
    float x1 = 0.0f;
    if (t >= 1) x1 = __half2float(x[(long long)(t - 1) * hidden + d]);
    const float m1 = ((t % block_size) >= 1) ? 1.0f : 0.0f;

    out[total] = __float2half(c0 * x0 + c1 * x1 * m1);
}

// ---- BF16 变体(全 bf16;数学逐式同 f16 版,float 中间)----
extern "C" __global__ void owl_dflash_conv_bf16(
    const __nv_bfloat16* __restrict__ x,      // [T, H]
    const __nv_bfloat16* __restrict__ delta,  // [T, 2, 2, G]
    const __nv_bfloat16* __restrict__ base,   // [2, 2, H]
    int side,
    int block_size,
    int hidden,
    int groups,
    int rows,
    __nv_bfloat16* __restrict__ out) {        // [T, H]
    const int group = 16;
    const int total = blockIdx.x * blockDim.x + threadIdx.x;
    if (total >= rows * hidden) return;

    const int t = total / hidden;
    const int d = total % hidden;
    const int g = d / group;
    const int G = groups;

    const float d0 = __bfloat162float(delta[((long long)t * 2 + side) * 2 * G + 0 * G + g]);
    const float d1 = __bfloat162float(delta[((long long)t * 2 + side) * 2 * G + 1 * G + g]);
    const float b0 = __bfloat162float(base[((long long)side * 2 + 0) * hidden + d]);
    const float b1 = __bfloat162float(base[((long long)side * 2 + 1) * hidden + d]);
    const float c0 = d0 + b0;
    const float c1 = d1 + b1;

    const float x0 = __bfloat162float(x[(long long)t * hidden + d]);
    float x1 = 0.0f;
    if (t >= 1) x1 = __bfloat162float(x[(long long)(t - 1) * hidden + d]);
    const float m1 = ((t % block_size) >= 1) ? 1.0f : 0.0f;

    out[total] = __float2bfloat16(c0 * x0 + c1 * x1 * m1);
}
