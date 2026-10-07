// ============================================================================
// owl_reshape_and_cache_fp8kv(B6.2;2026-10-10)
//
// K0 批量写池的 fp8 e4m3 变体:输入 f16 K/V(投影产物),写池时转
// e4m3(1B/elem)。布局与 f16 版同逻辑([num_blocks, Hkv, D/x, bs, x] /
// [num_blocks, Hkv, D, bs]),仅元素宽 2B→1B;索引公式逐式同源。
//
// port 适配(承 reshape_and_cache.cu 头注同款):
//   - slots f32 过线(契约 5;负 = padding 跳写);
//   - grid = (num_tokens,1,1) 显式发射;尾参哑输出(契约 4);
//   - 转换 = __nv_fp8_e4m3(__half)(SATFINITE:饱和不溢,有限输入
//     不产 NaN);sm_86 走软件转换路径(cuda_fp8.hpp 内建回退)。
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" __global__ void owl_reshape_and_cache_fp8kv(
    const __half* __restrict__ key,            // [num_tokens, num_heads, head_size]
    const __half* __restrict__ value,          // [num_tokens, num_heads, head_size]
    unsigned char* __restrict__ key_cache,     // e4m3 [num_blocks, num_heads, head_size/x, block_size, x]
    unsigned char* __restrict__ value_cache,   // e4m3 [num_blocks, num_heads, head_size, block_size]
    const float* __restrict__ slot_mapping,    // [num_tokens](物理槽;负 = padding 跳写)
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    __half* __restrict__ out) {   // 哑输出(契约 4 尾位)
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

        const long long tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                    + head_idx * (head_size / x) * block_size * x
                                    + x_idx * block_size * x
                                    + block_offset * x
                                    + x_offset;
        const long long tgt_value_idx = block_idx * num_heads * head_size * block_size
                                      + head_idx * head_size * block_size
                                      + head_offset * block_size
                                      + block_offset;
        key_cache[tgt_key_idx] =
            __nv_fp8_e4m3(__half2float(key[src_key_idx])).__x;
        value_cache[tgt_value_idx] =
            __nv_fp8_e4m3(__half2float(value[src_value_idx])).__x;
    }
}

// ---- BF16 输入变体(草稿侧 k/v 为 BF16;池仍 e4m3)----
extern "C" __global__ void owl_reshape_and_cache_fp8kv_bf16(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    unsigned char* __restrict__ key_cache,
    unsigned char* __restrict__ value_cache,
    const float* __restrict__ slot_mapping,
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    __nv_bfloat16* __restrict__ out) {
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
        const long long tgt_key_idx = block_idx * num_heads * (head_size / x) * block_size * x
                                    + head_idx * (head_size / x) * block_size * x
                                    + x_idx * block_size * x
                                    + block_offset * x
                                    + x_offset;
        const long long tgt_value_idx = block_idx * num_heads * head_size * block_size
                                      + head_idx * head_size * block_size
                                      + head_offset * block_size
                                      + block_offset;
        key_cache[tgt_key_idx] =
            __nv_fp8_e4m3(__bfloat162float(key[src_key_idx])).__x;
        value_cache[tgt_value_idx] =
            __nv_fp8_e4m3(__bfloat162float(value[src_value_idx])).__x;
    }
}
