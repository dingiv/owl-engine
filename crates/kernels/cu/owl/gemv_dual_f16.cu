// kernels/cu/owl —— 刀3a'(2026-10-04,E-decode 冲 45 t/s)
// 双权 GEMV 单发:y = [Wb·x; Wa·x](b/a 投影 concat 语义,零装载手术)
// —— 替 cublas gemvx + splitKreduce 双发射对(×48 GDN 层 = 96 节点 → 48)。
// 溯源:朴素行 GEMV(warp-per-row,fp32 累加,half2 主路);decode m=1
// 专用(prefill 臂保持 cublas gemm,m>1 本核无效率)。
// 契约:wb [rows_b, cols] / wa [rows_a, cols] / x [m, cols] |
//       i32 rows_b, rows_a, cols | y [m, rows_b+rows_a](末参输出)

#include <cuda_fp16.h>

extern "C" __global__ void owl_gemv_dual_f16(
    const __half* __restrict__ wb,    // [rows_b, cols]
    const __half* __restrict__ wa,    // [rows_a, cols]
    const __half* __restrict__ x,     // [m, cols]
    const int rows_b,
    const int rows_a,
    const int cols,
    __half* __restrict__ y) {         // [m, rows_b + rows_a](末参 = 输出)
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int rows_total = rows_b + rows_a;
    if (row >= rows_total) return;
    const int token = blockIdx.y;
    const __half* wrow = (row < rows_b) ? (wb + (size_t)row * cols)
                                        : (wa + (size_t)(row - rows_b) * cols);
    const __half* xrow = x + (size_t)token * cols;
    float acc = 0.0f;
    const __half2* w2 = reinterpret_cast<const __half2*>(wrow);
    const __half2* x2 = reinterpret_cast<const __half2*>(xrow);
    const int c2 = cols >> 1;  // cols 偶数(hidden 全偶;奇数尾不支持)
    for (int c = lane; c < c2; c += 32) {
        const float2 wf = __half22float2(w2[c]);
        const float2 xf = __half22float2(x2[c]);
        acc += wf.x * xf.x + wf.y * xf.y;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0)
        y[(size_t)token * rows_total + row] = __float2half(acc);
}
