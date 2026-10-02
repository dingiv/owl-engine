// ============================================================================
// reshape_and_cache dual(K0-dual;2026-10-03 FlashInfer 接入配套)
//
// 与 vllm_reshape_and_cache_f16(同目录,K0)同语义,增写第三/四目标:
//   FlashInfer paged_kv_t QKVLayout::kNHD 页布局(per page =
//   [page_size, Hkv, head_dim],entry 在外 —— page.cuh get_elem_offset:
//   page·stride_page + h·hd + entry·(Hkv·hd) + d)。owl classic K(x-
//   interleave)与 V([nb,Hkv,hd,page] dim-major)均不匹配 → K/V 双影子
//   池由此核同发射写出(k_fi/v_fi);FI prefill 吃影子池;v1 decode 继续
//   吃 classic K/V,互不干扰。
//
// 头注纪律同 K0(槽契约 5:f32 过线;grid 显式 (num_tokens,1,1);哑尾参)。
// ============================================================================
#include <cuda_fp16.h>

extern "C" __global__ void owl_reshape_and_cache_dual_f16(
    const __half* __restrict__ key,            // [num_tokens, num_heads, head_size]
    const __half* __restrict__ value,          // 同上
    __half* __restrict__ key_cache,            // classic [nb, Hkv, hd/x, bs, x]
    __half* __restrict__ value_cache,          // classic [nb, Hkv, hd, bs]
    __half* __restrict__ key_fi,               // kNHD [nb, bs, Hkv, hd]
    __half* __restrict__ value_fi,             // kNHD [nb, bs, Hkv, hd]
    const float* __restrict__ slot_mapping,    // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    __half* __restrict__ out) {   // 哑输出(契约 4 尾位;不写)
    (void)out;
    const long long token_idx = blockIdx.x;
    const long long slot_idx = (long long)slot_mapping[token_idx];
    if (slot_idx < 0) {
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

        // classic K:vLLM x-interleave(同 K0)
        const long long tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                    + head_idx * (head_size / x) * block_size * x
                                    + x_idx * block_size * x
                                    + block_offset * x
                                    + x_offset;
        // classic V:[nb, Hkv, hd, bs](= kHND;同 K0)
        const long long tgt_value_idx = block_idx * num_heads * head_size * block_size
                                      + head_idx * head_size * block_size
                                      + head_offset * block_size
                                      + block_offset;
        key_cache[tgt_key_idx] = key[src_key_idx];
        value_cache[tgt_value_idx] = value[src_value_idx];
        // kNHD 影子:[nb, bs, Hkv, hd](entry-major;per token 连续 h·hd)
        const long long tgt_fi_idx = block_idx * block_size * num_heads * head_size
                                   + block_offset * num_heads * head_size
                                   + head_idx * head_size
                                   + head_offset;
        key_fi[tgt_fi_idx] = key[src_key_idx];
        value_fi[tgt_fi_idx] = value[src_value_idx];
    }
}
