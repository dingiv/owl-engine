// owl attention kernels —— K0: reshape_and_cache
// 出处: vendor/attention.rs rev c0f19f2 src/kernels/src/reshape_and_cache_kernel.cu
//       (vLLM 系 paged attention 配套;port 专项见 docs/arch/attention-kernel-port.md)
// owl 改编(2026-09-22, A4 所有权转移):
//   - K0 仅实例化 scalar_t = cache_t = float(非量化路径;f16/bf16 档 K1 收窄实例化);
//   - FP8/E4M3 量化分支不搬(marlin-ffi 归属);
//   - 自包含:去掉 attention_dtypes.h / cuda_compat.h 依赖(f32 路径不需要);
//   - 布局: key_cache [num_blocks, num_heads, head_size/x, block_size, x]
//           value_cache [num_blocks, num_heads, head_size, block_size]
//   - 捕获安全: 纯指针算术,无分配/无同步/无 D2H(A1.5 契约,形式化留证)。



extern "C" __global__ void owl_reshape_and_cache_f32(
    const float* __restrict__ key,           // [num_tokens, num_heads * head_size]
    const float* __restrict__ value,         // [num_tokens, num_heads * head_size]
    float* __restrict__ key_cache,           // [num_blocks, num_heads, head_size/x, block_size, x]
    float* __restrict__ value_cache,         // [num_blocks, num_heads, head_size, block_size]
    const long long* __restrict__ slot_mapping, // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    const int num_tokens) {
  const long long token_idx = blockIdx.x;
  if (token_idx >= num_tokens) return;
  const long long slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    // Padding token that should be ignored.
    return;
  }

  const long long block_idx = slot_idx / block_size;
  const long long block_offset = slot_idx % block_size;

  const int n = num_heads * head_size;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const long long src_key_idx = token_idx * key_stride + i;
    const long long src_value_idx = token_idx * value_stride + i;

    const int head_idx = i / head_size;
    const int head_offset = i % head_size;
    const int x_idx = head_offset / x;
    const int x_offset = head_offset % x;

    const long long tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                + head_idx * (head_size / x) * block_size * x
                                + x_idx * block_size * x
                                + block_offset * x
                                + x_offset;
    const long long tgt_value_idx = block_idx * num_heads * head_size * block_size
                                  + head_idx * head_size * block_size
                                  + head_offset * block_size
                                  + block_offset;
    key_cache[tgt_key_idx] = key[src_key_idx];
    value_cache[tgt_value_idx] = value[src_value_idx];
  }
}
