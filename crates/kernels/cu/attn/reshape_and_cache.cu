// ============================================================================
// reshape_and_cache(fatt/paged 布局 K0;2026-09-27 port)
//
// 出处(vendor file-level 复用,Apache-2.0):
//   packages/xinfer/vendor/attention.rs/src/kernels/src/reshape_and_cache_kernel.cu
//   (rev c0f19f2,冻结只读;上游 = vLLM csrc;A4 所有权纪律:头标注 +
//    语义保持,dtype 收窄见 attention-kernel-port.md §二.2)
//
// port 适配(显式认领,均为实例化特化而非语义改动):
//   1. 模板 <scalar_t, cache_t> → f16/f16 单实例(scalar_t == cache_t ⇒
//      is_quantized ≡ false,FP8 臂编译期剥除,k_scales/v_scales 形参随删);
//   2. extern "C" 入口(vnvrtc 按名取核;模板实例化名不可寻址);
//   3. grid 契约 = (num_tokens, 1, 1) 显式发射(核内 blockIdx.x = token_idx;
//      哨兵自动 1D 会越界读 slot_mapping,禁用);
//   4. slot_mapping long long* → float*(owl 契约 5:索引/位置量 f32 数值过线,
//      核内 cast;同 naive decode 核 slots 同款;|slot| < 2²⁴ 内精确);
//   5. 布局契约(vLLM classic,K1/K2 paged_attention_v1/v2 同款):
//      key_cache   [num_blocks, Hkv, D/x, block_size, x](x = 16/sizeof(f16) = 8)
//      value_cache [num_blocks, Hkv, D, block_size]
//      slot_mapping [num_tokens](物理槽 = block_idx * block_size + offset;
//      单序列连续分配时 block_table 为恒等,物理槽 = 逻辑 pos)。
//   6. 尾参 `out`(哑输出):owl 图契约末位 T = 解释器新分配输出块(槽序契约
//      4),本核无单一输出(双 cache 就地写)—— 尾参占位不写;发射 grid
//      必须显式(out 元素数 = 1,哨兵自动 grid 会错)。
//   7. nvrtc 头约束:stdint.h 不可开(nvrtc 默认头集极简)→ int64_t 以内建
//      long long 同宽替(其余内建面与 ops_pair.cu 同口径)。
// ============================================================================
#include <cuda_fp16.h>

extern "C" __global__ void vllm_reshape_and_cache_f16(
    const __half* __restrict__ key,            // [num_tokens, num_heads, head_size]
    const __half* __restrict__ value,          // [num_tokens, num_heads, head_size]
    __half* __restrict__ key_cache,            // [num_blocks, num_heads, head_size/x, block_size, x]
    __half* __restrict__ value_cache,          // [num_blocks, num_heads, head_size, block_size]
    const float* __restrict__ slot_mapping,    // [num_tokens](物理槽;负 = padding 跳写;契约 5)
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    __half* __restrict__ out) {   // 哑输出(契约 4 尾位;核不写,见头注 6)
    (void)out;
    const long long token_idx = blockIdx.x;
    const long long slot_idx = (long long)slot_mapping[token_idx];
    if (slot_idx < 0) {
        // Padding token that should be ignored.(原注释保留)
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

// ---- BF16 变体(E5-DF3 同日十四;DFlash2 草稿池 BF16)----
// 全 bf16(key/value/cache 同型;slots f32 / 标量契约不变)。数学无算术,
// 纯搬运 —— 与 f16 版仅指针类型差。
#include <cuda_bf16.h>

extern "C" __global__ void vllm_reshape_and_cache_bf16(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
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
        key_cache[tgt_key_idx] = key[src_key_idx];
        value_cache[tgt_value_idx] = value[src_value_idx];
    }
}

// ---- kNHD 变体(kv布局统一契约 P1;roadmap.local/kv布局统一契约-立项设计与施工.md)----
// 布局:K/V 同构 [num_blocks, block_size, num_heads, head_size](页内 (head,d) 连续;
// FlashInfer 原生)。寻址与 classic 的 x 交错不同,槽序契约/x 形参保持不变
// (x 不再参与寻址,保留占位 = 发射面零改动;P6 随 Classic 臂一并退役)。
// 纯搬运零算术 → 与 classic 版写值逐位一致(parity 门 = bitwise)。

extern "C" __global__ void vllm_reshape_and_cache_f16_knhd(
    const __half* __restrict__ key,
    const __half* __restrict__ value,
    __half* __restrict__ key_cache,            // [nb, block_size, num_heads, head_size]
    __half* __restrict__ value_cache,          // [nb, block_size, num_heads, head_size]
    const float* __restrict__ slot_mapping,
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    __half* __restrict__ out) {
    (void)out; (void)x;
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
        const long long tgt = block_idx * num_heads * head_size * block_size
                            + block_offset * num_heads * head_size
                            + head_idx * head_size
                            + head_offset;
        key_cache[tgt] = key[src_key_idx];
        value_cache[tgt] = value[src_value_idx];
    }
}

extern "C" __global__ void vllm_reshape_and_cache_bf16_knhd(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_bfloat16* __restrict__ key_cache,
    __nv_bfloat16* __restrict__ value_cache,
    const float* __restrict__ slot_mapping,
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    __nv_bfloat16* __restrict__ out) {
    (void)out; (void)x;
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
        const long long tgt = block_idx * num_heads * head_size * block_size
                            + block_offset * num_heads * head_size
                            + head_idx * head_size
                            + head_offset;
        key_cache[tgt] = key[src_key_idx];
        value_cache[tgt] = value[src_value_idx];
    }
}
