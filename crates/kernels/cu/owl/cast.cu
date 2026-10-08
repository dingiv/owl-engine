// f16 <-> f32 设备 cast(2026-10-03;GDN chunked 编排配套:FLA cubin 吃/吐 f32,
// owl 图面为 f16 —— handler 在发射前后各 cast 一次)。输出末参(契约 4)。
// nvrtc 极简头集需自备 typedef 与 half 原语(prefill_split_f16.cu 同款守卫)。
//
// f16 <-> bf16 cast(E5-DF3 同日十四;DFlash2 草稿路径 BF16 化铸边界):
//   owl_cast_f16_bf16   embed 查表后(f16 表)→ 草稿主干入口(bf16)
//   owl_cast_bf16_f16   草稿 hidden(bf16)→ lm_head/cublas 入口(f16)
#ifdef __CUDACC_RTC__
typedef unsigned short ushort;
#endif
#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" __global__ void owl_cast_f16_f32(const __half *__restrict__ x, int n, float *__restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(x[i]);
}

extern "C" __global__ void owl_cast_f32_f16(const float *__restrict__ x, int n, __half *__restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

extern "C" __global__ void owl_cast_f16_bf16(const __half *__restrict__ x, int n, __nv_bfloat16 *__restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__half2float(x[i]));
}

extern "C" __global__ void owl_cast_bf16_f16(const __nv_bfloat16 *__restrict__ x, int n, __half *__restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(__bfloat162float(x[i]));
}

// GDN 状态转置(2026-10-11 fork-bf16 全家桶配套):owl 池 [HV,K,V] ↔
// fork h0/ht [HV,V,K](f32;KD=VD=128)。grid (v_dim, hv),block (k_dim):
// 写侧合并，读侧列距 —— 半保守转置，3.1MB 双向 ~30µs 量级(672 次/请求
// ≈ 20ms,对应省下的 5.5× GDN 核时间可忽略)。
extern "C" __global__ void owl_state_kv_to_vk(const float *__restrict__ in,
                                            float *__restrict__ out,
                                            int k_dim, int v_dim) {
    // out[(hv*V + v)*K + k] = in[(hv*K + k)*V + v]
    out[(blockIdx.y * v_dim + blockIdx.x) * k_dim + threadIdx.x] =
        in[(blockIdx.y * k_dim + threadIdx.x) * v_dim + blockIdx.x];
}

extern "C" __global__ void owl_state_vk_to_kv(const float *__restrict__ in,
                                            float *__restrict__ out,
                                            int k_dim, int v_dim) {
    // 逆:in [HV,V,K] → out [HV,K,V]
    out[(blockIdx.y * k_dim + threadIdx.x) * v_dim + blockIdx.x] =
        in[(blockIdx.y * v_dim + blockIdx.x) * k_dim + threadIdx.x];
}
