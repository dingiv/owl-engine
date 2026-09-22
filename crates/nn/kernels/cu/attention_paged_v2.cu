// owl attention kernels —— K2: paged_attention_v2(大 bs 分片归约变体)
// 出处: vendor/attention.rs rev c0f19f2 src/kernels/src/paged_attention_v2.cu
//       (vLLM/FasterTransformer 系;port 专项见 docs/arch/attention-kernel-port.md)
// owl 改编(2026-09-22, A4 所有权转移;方法论同 K1 attention_paged_v1.cu):
//   - 手工解模板:显式 extern "C" __global__ 实例化 ×4 主核(f16/bf16 × BLOCK_SIZE 32/64)
//     + reduce 核 ×2(f16/bf16;reduce 不依赖 cache_T/BLOCK_SIZE);
//     head_size 只实例化 128;PARTITION_SIZE = 512(vLLM 默认),NUM_THREADS = 128;
//   - exp_sums/max_logits/tmp_out 为**调用方 scratch 缓冲**(A5.2:显式参数,
//     kernel 内零分配);布局 [num_seqs, num_heads, max_num_partitions(, head_size)];
//   - grid(main) = (num_heads, num_seqs, max_num_partitions);grid(reduce) = (num_heads, num_seqs);
//     smem(main) = max(PARTITION_SIZE*4, NUM_WARPS/2*head_size*4);smem(reduce) = 2*parts*4;
//   - FP8/量化分支裁掉(k_scales/v_scales 恒 null);alibi nullptr;
//   - 布局契约同 v1: key_cache [num_blocks, num_kv_heads, head_size/x, block_size, x]
//                  value_cache [num_blocks, num_kv_heads, head_size, block_size]
//   - 捕获安全: 纯指针算术 + shfl/syncthreads,无分配/无同步/无 D2H(A1.5)。

#include "attention/pagedattention.cuh"

using namespace vllm;

// ---- 主核(分片:每个 partition 一个 blockIdx.z)----

#define OWL_PA_V2_MAIN(FN_NAME, T, cache_T, BLOCK_SIZE)                        \
extern "C" __global__ void FN_NAME(                                            \
    float* __restrict__ exp_sums,                                              \
    float* __restrict__ max_logits,                                            \
    T* __restrict__ tmp_out,                                                   \
    const T* __restrict__ query,                                               \
    const cache_T* __restrict__ key_cache,                                     \
    const cache_T* __restrict__ value_cache,                                   \
    const float* __restrict__ k_scales, const float* __restrict__ v_scales,    \
    const int num_kv_heads, const float scale,                                 \
    const int32_t* __restrict__ block_tables,                                  \
    const int32_t* __restrict__ context_lens,                                  \
    const int max_num_blocks_per_seq,                                          \
    const float* __restrict__ alibi_slopes,                                    \
    const int q_stride, const int kv_block_stride, const int kv_head_stride,   \
    const float softscapping, const int sliding_window) {                      \
  vllm::paged_attention_v2_kernel<T, cache_T, 128, BLOCK_SIZE, 128, 512>(      \
      exp_sums, max_logits, tmp_out, query, key_cache, value_cache,            \
      k_scales, v_scales, num_kv_heads, scale, block_tables, context_lens,     \
      max_num_blocks_per_seq, alibi_slopes, q_stride, kv_block_stride,         \
      kv_head_stride, softscapping, sliding_window);                           \
}

// ---- reduce 核(跨 partition log-sum-exp 合并)----

#define OWL_PA_V2_REDUCE(FN_NAME, T)                                           \
extern "C" __global__ void FN_NAME(                                            \
    T* __restrict__ out,                                                       \
    const float* __restrict__ exp_sums,                                        \
    const float* __restrict__ max_logits,                                      \
    const T* __restrict__ tmp_out,                                             \
    const int32_t* __restrict__ context_lens,                                  \
    const int max_num_partitions) {                                            \
  vllm::paged_attention_v2_reduce_kernel<T, 128, 128, 512>(                    \
      out, exp_sums, max_logits, tmp_out, context_lens, max_num_partitions);   \
}

// ---- f16 ----
OWL_PA_V2_MAIN(owl_pa_v2_main_f16_bs32, uint16_t, uint16_t, 32)
OWL_PA_V2_MAIN(owl_pa_v2_main_f16_bs64, uint16_t, uint16_t, 64)
OWL_PA_V2_REDUCE(owl_pa_v2_reduce_f16, uint16_t)

// ---- bf16 ----
OWL_PA_V2_MAIN(owl_pa_v2_main_bf16_bs32, __nv_bfloat16, __nv_bfloat16, 32)
OWL_PA_V2_MAIN(owl_pa_v2_main_bf16_bs64, __nv_bfloat16, __nv_bfloat16, 64)
OWL_PA_V2_REDUCE(owl_pa_v2_reduce_bf16, __nv_bfloat16)
