// f16 <-> f32 设备 cast(2026-10-03;GDN chunked 编排配套:FLA cubin 吃/吐 f32,
// owl 图面为 f16 —— handler 在发射前后各 cast 一次)。输出末参(契约 4)。
// nvrtc 极简头集需自备 typedef 与 half 原语(prefill_split_f16.cu 同款守卫)。
#ifdef __CUDACC_RTC__
typedef unsigned short ushort;
#endif
#include <cuda_fp16.h>

extern "C" __global__ void owl_cast_f16_f32(const __half *__restrict__ x, int n, float *__restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(x[i]);
}

extern "C" __global__ void owl_cast_f32_f16(const float *__restrict__ x, int n, __half *__restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}
