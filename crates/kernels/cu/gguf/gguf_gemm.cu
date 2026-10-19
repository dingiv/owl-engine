// ============================================================================
// owl GGUF 族:dequant-in-kernel GEMM(**MMVQ**:一个 warp 一行输出)。
//
// 来源与改动登记:移植自 `repos/attention.rs/.../gguf_gemm.cu` 的
// `gguf_gemm_kernel`(其头注:"adapted from llama.cpp ggml-cuda")。本仓改动:
//
//   **H2 零运行期设备查询**:上游 `gguf_gemm()` 用 `cudaGetDeviceProperties`
//   在发射期查 CC,且核内 `if (size_m >= 8) 走 prefill tile` 分派 —— 本仓
//   全部去掉:arch 由 boot 期定(编译期 gencode + 调用方选入口),prefill
//   tile 臂**尚未移植**(设计 R2 挂账:prefill 优化二期,先保正确性)。
//   **H1 零分配**:上游核内 `cudaMallocAsync` 装 q8_1 中间缓冲 ⇒ 本仓由调用方
//   传入(见 `gguf_quant.cu`),本核**纯读**。
//   **H3 逐型入口**:上游 `quant_type` 是裸 int switch(0..14)⇒ 本仓**每型一个
//   `extern "C"` 入口**(名字 = `family::gguf::entry_for(ty)`,准入表逐型点名)。
//   `out_off`/`out_ld` 新增:支持**分段合并写**(GDN 的 qkv⊕z 两段异型,各写
//   输出张量的不同列窗 —— 设计 §十一 11.5)。
//
// 写法:**共享 `__device__` 模板体 + 逐型字面 wrapper**(体只写一次;wrapper
// 是字面核形态 ⇒ `cu_sig` 取证面可直接互证,不用宏实例行)。
//
// # 发射契约(声明面 `ops/gguf.rs` 逐参对位)
//
//   grid  = (ceil(n / n_warps), m);block = (32, n_warps, 1)
//   lane = threadIdx.x;warp = threadIdx.y;n_warps = blockDim.y
//   行 = blockIdx.x * n_warps + warp(权重的**外维**);m 索引 = blockIdx.y
//
// 权重按 GGUF 块布局**原样**读(`dequant 在核内`),激活必须先过
// `owl_gguf_quant_act_*`(q8_1 块,`k_padded` 为 32 的倍数)。
// 每次 `vec_dot` 调用的归约序**与形状无关**(设计 TP 纪律:归约块数不随
// batch 变)⇒ 同权重同输入 = 逐位可复现。
// ============================================================================
#include "gguf.cuh"

template <int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
__device__ __forceinline__ void owl_gguf_mmvq_body(
    const void * __restrict__ weights, const void * __restrict__ xq,
    __half * __restrict__ out, int m, int n, int k, int k_padded, int out_off,
    int out_ld) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int n_warps = blockDim.y;
    const int row = blockIdx.x * n_warps + warp;
    const int m_idx = blockIdx.y;
    if (row >= n || m_idx >= m) {
        return;
    }
    const int weight_blocks_per_row = k / qk;
    const int input_blocks_per_row = k_padded / QK8_1;
    const block_q_t * __restrict__ weight_row =
        reinterpret_cast<const block_q_t *>(weights) + (size_t) row * weight_blocks_per_row;
    const block_q8_1 * __restrict__ input_row =
        reinterpret_cast<const block_q8_1 *>(xq) + (size_t) m_idx * input_blocks_per_row;
    float acc = 0.0f;
    const int blocks_per_iter = vdr * WARP_SIZE / qi;
    for (int k_block = lane / (qi / vdr); k_block < weight_blocks_per_row; k_block += blocks_per_iter) {
        const int input_block = k_block * (qk / QK8_1);
        const int quant_index = vdr * (lane % (qi / vdr));
        acc += vec_dot_q_cuda(&weight_row[k_block], &input_row[input_block], quant_index);
    }
    const float value = warp_reduce_sum(acc);
    if (lane == 0) {
        out[(size_t) m_idx * out_ld + out_off + row] = __float2half(value);
    }
}

// ---- M1 实例集(四型;设计 REQ-GGUF-04);加型 = 加一个 wrapper + 加一条声明 ----

extern "C" __global__ void owl_gguf_gemm_q8_0(
    const void * __restrict__ weights, const void * __restrict__ xq,
    __half * __restrict__ out, int m, int n, int k, int k_padded, int out_off,
    int out_ld) {
    owl_gguf_mmvq_body<QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(
        weights, xq, out, m, n, k, k_padded, out_off, out_ld);
}

extern "C" __global__ void owl_gguf_gemm_q4k(
    const void * __restrict__ weights, const void * __restrict__ xq,
    __half * __restrict__ out, int m, int n, int k, int k_padded, int out_off,
    int out_ld) {
    owl_gguf_mmvq_body<QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(
        weights, xq, out, m, n, k, k_padded, out_off, out_ld);
}

extern "C" __global__ void owl_gguf_gemm_q5k(
    const void * __restrict__ weights, const void * __restrict__ xq,
    __half * __restrict__ out, int m, int n, int k, int k_padded, int out_off,
    int out_ld) {
    owl_gguf_mmvq_body<QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(
        weights, xq, out, m, n, k, k_padded, out_off, out_ld);
}

extern "C" __global__ void owl_gguf_gemm_q6k(
    const void * __restrict__ weights, const void * __restrict__ xq,
    __half * __restrict__ out, int m, int n, int k, int k_padded, int out_off,
    int out_ld) {
    owl_gguf_mmvq_body<QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(
        weights, xq, out, m, n, k, k_padded, out_off, out_ld);
}
