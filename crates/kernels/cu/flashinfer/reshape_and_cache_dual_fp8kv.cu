// ============================================================================
// reshape_and_cache dual fp8 KV(2026-10-03;REQ-CTX-03 fp8 KV 线)
//
// B6.3 收口(2026-10-10):**双臂全 e4m3**。本核仅在主池=fp8 时被选中
// (attention.rs fp8kv 分派;f16 主池走 K0_DUAL 原版),故 classic 臂
// 必须 e4m3 1B/elem —— 原版 classic f16 2B 存储落在 1B 池 = 真槽区零写
// + 毒液洒 2s 槽区(与融合插池同型;FI-decode 案真凶,探针同族补)
//
// owl_reshape_and_cache_dual_f16(同族)的 fp8 KV 变体:classic e4m3
// (v2 fp8 decode 读),K/V 影子写 e4m3 字节(kNHD [nb,page,Hkv,hd])。
// FI fp8 prefill(DTypeKV=__nv_fp8_e4m3,scale 隐式 1.0)消费影子池。
// e4m3 转换 = cuda_fp8.h __nv_fp8_e4m3(sem86 软件,无硬件 fp8;写侧一次性
// 摊销)。nvrtc 独立编译(FI adapter 本体含 flashinfer 头,nvrtc 啃不动)。
// 头注纪律同 K0(槽契约 5;grid 显式 (num_tokens,1,1);哑尾参)。
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_fp8.h>

extern "C" __global__ void owl_reshape_and_cache_dual_f16_fp8kv(
    const __half* __restrict__ key,            // [num_tokens, num_heads, head_size]
    const __half* __restrict__ value,          // 同上
    unsigned char* __restrict__ key_cache,     // classic e4m3 [nb, Hkv, hd/x, bs, x]
    unsigned char* __restrict__ value_cache,   // classic e4m3 [nb, Hkv, hd, bs]
    __nv_fp8_e4m3* __restrict__ key_fi,        // kNHD fp8 [nb, bs, Hkv, hd]
    __nv_fp8_e4m3* __restrict__ value_fi,      // kNHD fp8 [nb, bs, Hkv, hd]
    const float* __restrict__ slot_mapping,    // [num_tokens]
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int x,
    __half* __restrict__ out) {                // 哑输出(契约 4 尾位;不写)
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
        key_cache[tgt_key_idx] = __nv_fp8_e4m3(__half2float(key[src_key_idx])).__x;
        value_cache[tgt_value_idx] = __nv_fp8_e4m3(__half2float(value[src_value_idx])).__x;
        const long long tgt_fi_idx = block_idx * block_size * num_heads * head_size
                                   + block_offset * num_heads * head_size
                                   + head_idx * head_size
                                   + head_offset;
        key_fi[tgt_fi_idx] = __nv_fp8_e4m3(key[src_key_idx]);
        value_fi[tgt_fi_idx] = __nv_fp8_e4m3(value[src_value_idx]);
    }
}
