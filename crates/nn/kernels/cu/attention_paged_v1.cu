// owl attention kernels —— K1: paged_attention_v1(f16/bf16 × block 32/64)
// 出处: vendor/attention.rs rev c0f19f2 src/kernels/src/paged_attention_v1.cu
//       (vLLM 系;port 专项见 docs/arch/attention-kernel-port.md)
// owl 改编(2026-09-22, A4 所有权转移):
//   - 手工解模板:vLLM 的 C 级 dispatch switch(dtype/block_size 运行时分支)
//     改为显式 extern "C" 实例化 ×4(f16/bf16 × BLOCK_SIZE 32/64);
//     head_size 只实例化 128(qwen3.5 档,nvrtc 编译时长控制,立项文档 §二·2);
//   - NUM_THREADS = 128(vLLM 默认);alibi 暂不启用(nullptr 路径);
//   - softscapping / sliding_window 参数面保留(qwen3.5 不用,传默认值);
//   - k_scales/v_scales 恒 nullptr(非量化;FP8 KV 归 marlin-ffi 路线);
//   - 布局契约: key_cache [num_blocks, num_kv_heads, head_size/x, block_size, x]
//               value_cache [num_blocks, num_kv_heads, head_size, block_size]
//   - 捕获安全: 纯指针算术 + shfl/syncthreads,无分配/无同步/无 D2H(A1.5)。

#include "attention/pagedattention.cuh"

using namespace vllm;

// ---- f16(vLLM 惯例:scalar_t = uint16_t,PTX 位运算数学,dtype_float16.cuh)----

extern "C" __global__ void owl_pa_v1_f16_bs32(
    uint16_t* __restrict__ out, const uint16_t* __restrict__ query,
    const uint16_t* __restrict__ key_cache, const uint16_t* __restrict__ value_cache,
    const float* __restrict__ k_scales, const float* __restrict__ v_scales,
    const int num_kv_heads, const float scale, const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens, const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window) {
  vllm::paged_attention_v1_kernel<uint16_t, uint16_t, 128, 32, 128>(
      out, query, key_cache, value_cache, k_scales, v_scales,
      num_kv_heads, scale, block_tables, context_lens, max_num_blocks_per_seq,
      alibi_slopes, q_stride, kv_block_stride, kv_head_stride, softscapping,
      sliding_window);
}

extern "C" __global__ void owl_pa_v1_f16_bs64(
    uint16_t* __restrict__ out, const uint16_t* __restrict__ query,
    const uint16_t* __restrict__ key_cache, const uint16_t* __restrict__ value_cache,
    const float* __restrict__ k_scales, const float* __restrict__ v_scales,
    const int num_kv_heads, const float scale, const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens, const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window) {
  vllm::paged_attention_v1_kernel<uint16_t, uint16_t, 128, 64, 128>(
      out, query, key_cache, value_cache, k_scales, v_scales,
      num_kv_heads, scale, block_tables, context_lens, max_num_blocks_per_seq,
      alibi_slopes, q_stride, kv_block_stride, kv_head_stride, softscapping,
      sliding_window);
}

// ---- bf16(__nv_bfloat16;sm86 ≥ 800 档可用)----

extern "C" __global__ void owl_pa_v1_bf16_bs32(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache, const __nv_bfloat16* __restrict__ value_cache,
    const float* __restrict__ k_scales, const float* __restrict__ v_scales,
    const int num_kv_heads, const float scale, const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens, const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window) {
  vllm::paged_attention_v1_kernel<__nv_bfloat16, __nv_bfloat16, 128, 32, 128>(
      out, query, key_cache, value_cache, k_scales, v_scales,
      num_kv_heads, scale, block_tables, context_lens, max_num_blocks_per_seq,
      alibi_slopes, q_stride, kv_block_stride, kv_head_stride, softscapping,
      sliding_window);
}

extern "C" __global__ void owl_pa_v1_bf16_bs64(
    __nv_bfloat16* __restrict__ out, const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key_cache, const __nv_bfloat16* __restrict__ value_cache,
    const float* __restrict__ k_scales, const float* __restrict__ v_scales,
    const int num_kv_heads, const float scale, const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens, const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window) {
  vllm::paged_attention_v1_kernel<__nv_bfloat16, __nv_bfloat16, 128, 64, 128>(
      out, query, key_cache, value_cache, k_scales, v_scales,
      num_kv_heads, scale, block_tables, context_lens, max_num_blocks_per_seq,
      alibi_slopes, q_stride, kv_block_stride, kv_head_stride, softscapping,
      sliding_window);
}
