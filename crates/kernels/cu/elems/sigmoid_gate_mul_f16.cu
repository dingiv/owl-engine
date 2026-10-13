// owl kernels/cu/owl —— NInfer 移植位(f16 基线战役;工单 N)
// 溯源:repos/ninfer_duo_3080/src/ops/kernel/sigmoid_gate_mul.cuh(bf16x8
// 向量包 + fp32 sigmoid)—— owl 契约改写:单输出树语义(非原地)、
// extern "C"(nvrtc 无 namespace/launcher)、__half2 主路。
// 语义:out[i] = x[i] · sigmoid(gate[i])(attention 输出门融合:
// owl 现为 sigmoid + mul 两发射,本核一次完成 —— F2 语义链融合首例)。
// n 偶数由声明侧保证(hidden/head_dim 全偶);grid-stride 兜底任意规模。

#include <cuda_fp16.h>

extern "C" __global__ void owl_sigmoid_gate_mul_f16(
    const __half* __restrict__ gate,  // [n]
    const __half* __restrict__ x,     // [n]
    size_t n,
    __half* __restrict__ out) {       // [n](末参 = 输出,槽序契约 4)
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const __half2* g2 = reinterpret_cast<const __half2*>(gate);
    const __half2* x2 = reinterpret_cast<const __half2*>(x);
    __half2* o2 = reinterpret_cast<__half2*>(out);
    for (size_t p = i; p < n / 2; p += stride) {
        const float2 gf = __half22float2(g2[p]);
        const float2 xf = __half22float2(x2[p]);
        float2 r;
        r.x = xf.x / (1.0f + __expf(-gf.x));
        r.y = xf.y / (1.0f + __expf(-gf.y));
        o2[p] = __floats2half2_rn(r.x, r.y);
    }
}
