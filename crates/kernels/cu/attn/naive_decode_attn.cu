// ============================================================================
// owl naive decode attention(2026-10-13 C 轮自 cu/owl/attention.cu 按**算子族**
// 拆出 —— 与 narrow_strided 分家,各自成文件)。
//
// 一线程一 (t, q_head)。写 cache(slot>=0)→ 对以 slot 结尾的连续 kv_len 行
// 打分 softmax → 加权和出 out。索引/位置量按形态契约 5 走 f32 数值过线。
// 语义与 f32 版逐式同源(窗 = [slot-kv_len+1, slot];scale = 1/sqrt(D))。
// OWL_MAX_KV = 256(单步寄存器分数上限;扩容/分页归真分页 kernel 立项)。
// ============================================================================
#include <cuda_fp16.h>

// decode(seq=1 per t)naive attention:一线程一 (t, q_head)。
// 写 cache(slot>=0)→ 对以 slot 结尾的连续 kv_len 行打分 softmax → 加权和出 out。
// 新世界契约改造(相对旧世界 dry 版):
//   - slots/kv_lens 以 f32 数值过线(契约 5;核内 cast,非负整数值 f32 精确);
//   - 加 bs 上界 guard(发射 grid 走哨兵自动 1D,多余线程安全退出);
//   - 打分窗口 = [slot-kv_len+1, slot](序列连续槽窗;旧版从行 0 扫,
//     仅 bs=1 / slot=kv_len-1 自洽 —— 本版为其严格泛化,等价条件下结果一致)。
// caller 契约:kv_len 含本步(先写 cache 后打分,kv_lens = past + 1);
// 序列 KV 占连续槽段且结尾在 slot(padding 惰形 slot<0 只跳过写,打分照扫)。
// OWL_MAX_KV = 256(单步寄存器分数上限;扩容/分页归真分页 kernel 立项)。
#define OWL_MAX_KV 256

extern "C" __global__ void owl_naive_decode_attn_f32(
    const float* q,            // [bs, Hq, D]
    const float* k,            // [bs, Hkv, D]
    const float* v,            // [bs, Hkv, D]
    float* kc,                 // [max_slots, Hkv, D](slot 直排)
    float* vc,                 // [max_slots, Hkv, D]
    const float* slots,        // [bs](f32 数值;负数 = padding)
    const float* kv_lens,      // [bs](f32 数值)
    size_t bs, size_t q_heads, size_t kv_heads, size_t d_dim,
    float* out) {              // [bs, Hq*D](末参 = 槽序契约)
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned int)(bs * q_heads)) return;
    unsigned int t = idx / (unsigned int)q_heads;
    unsigned int h = idx % (unsigned int)q_heads;
    unsigned int kvh = h / ((unsigned int)q_heads / (unsigned int)kv_heads);
    const int slot = (int)slots[t];
    int kv_len = (int)kv_lens[t];
    if (kv_len > OWL_MAX_KV) kv_len = OWL_MAX_KV;
    const int base = slot - kv_len + 1;   // 序列连续槽窗起点(padding slot<0 时打分窗由 caller 保证合法)

    const float* qs = q + ((unsigned long long)t * q_heads + h) * d_dim;

    // 1) 写 cache(slot >= 0)
    if (slot >= 0) {
        float* kcd = kc + (unsigned long long)slot * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        float* vcd = vc + (unsigned long long)slot * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        const float* ks = k + ((unsigned long long)t * kv_heads + kvh) * d_dim;
        const float* vs = v + ((unsigned long long)t * kv_heads + kvh) * d_dim;
        for (size_t d = 0; d < d_dim; ++d) {
            kcd[d] = ks[d];
            vcd[d] = vs[d];
        }
    }

    // 2) 打分(全行扫描;scale = 1/sqrt(D))
    float scores[OWL_MAX_KV];
    float maxs = -3.0e38f;
    float scale = rsqrtf((float)d_dim);
    for (int s = 0; s < kv_len; ++s) {
        const float* kc_row = kc + (unsigned long long)(base + s) * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        float acc = 0.0f;
        for (size_t d = 0; d < d_dim; ++d) acc += qs[d] * kc_row[d];
        acc *= scale;
        scores[s] = acc;
        if (acc > maxs) maxs = acc;
    }

    // 3) softmax(kv_len 掩码 = 循环上界本身)
    float denom = 0.0f;
    for (int s = 0; s < kv_len; ++s) {
        scores[s] = expf(scores[s] - maxs);
        denom += scores[s];
    }

    // 4) 加权和
    float* od = out + ((unsigned long long)t * q_heads + h) * d_dim;
    for (size_t d = 0; d < d_dim; ++d) od[d] = 0.0f;
    for (int s = 0; s < kv_len; ++s) {
        float wgt = scores[s] / denom;
        const float* vc_row = vc + (unsigned long long)(base + s) * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        for (size_t d = 0; d < d_dim; ++d) od[d] += wgt * vc_row[d];
    }
}

// f16 基线变体(F4):q/k/v half,KV cache half(先写后打分同款),
// slots/kv_lens 恒 f32(契约 5);打分/softmax/加权和全 float。
// 语义与 f32 版逐式同源(窗 = [slot-kv_len+1, slot];scale = 1/sqrt(D))。
extern "C" __global__ void owl_naive_decode_attn_f16(
    const __half* q,           // [bs, Hq, D]
    const __half* k,           // [bs, Hkv, D]
    const __half* v,           // [bs, Hkv, D]
    __half* kc,                // [max_slots, Hkv, D](slot 直排)
    __half* vc,                // [max_slots, Hkv, D]
    const float* slots,        // [bs](f32 数值;负 = padding)
    const float* kv_lens,      // [bs](f32 数值)
    size_t bs, size_t q_heads, size_t kv_heads, size_t d_dim,
    __half* out) {             // [bs, Hq*D](末参 = 槽序契约)
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (unsigned int)(bs * q_heads)) return;
    unsigned int t = idx / (unsigned int)q_heads;
    unsigned int h = idx % (unsigned int)q_heads;
    unsigned int kvh = h / ((unsigned int)q_heads / (unsigned int)kv_heads);
    const int slot = (int)slots[t];
    int kv_len = (int)kv_lens[t];
    if (kv_len > OWL_MAX_KV) kv_len = OWL_MAX_KV;
    const int base = slot - kv_len + 1;

    const __half* qs = q + ((unsigned long long)t * q_heads + h) * d_dim;

    // 1) 写 cache(slot >= 0;half 直存)
    if (slot >= 0) {
        __half* kcd = kc + (unsigned long long)slot * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        __half* vcd = vc + (unsigned long long)slot * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        const __half* ks = k + ((unsigned long long)t * kv_heads + kvh) * d_dim;
        const __half* vs = v + ((unsigned long long)t * kv_heads + kvh) * d_dim;
        for (size_t d = 0; d < d_dim; ++d) {
            kcd[d] = ks[d];
            vcd[d] = vs[d];
        }
    }

    // 2) 打分(float;q·k 逐元素 half2float)
    float scores[OWL_MAX_KV];
    float maxs = -3.0e38f;
    float scale = rsqrtf((float)d_dim);
    for (int s = 0; s < kv_len; ++s) {
        const __half* kc_row = kc + (unsigned long long)(base + s) * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        float acc = 0.0f;
        for (size_t d = 0; d < d_dim; ++d) acc += __half2float(qs[d]) * __half2float(kc_row[d]);
        acc *= scale;
        scores[s] = acc;
        if (acc > maxs) maxs = acc;
    }

    // 3) softmax
    float denom = 0.0f;
    for (int s = 0; s < kv_len; ++s) {
        scores[s] = expf(scores[s] - maxs);
        denom += scores[s];
    }

    // 4) 加权和(half2float 读 v,float 累计,half 写)
    __half* od = out + ((unsigned long long)t * q_heads + h) * d_dim;
    for (size_t d = 0; d < d_dim; ++d) od[d] = __float2half(0.0f);
    for (int s = 0; s < kv_len; ++s) {
        float wgt = scores[s] / denom;
        const __half* vc_row = vc + (unsigned long long)(base + s) * kv_heads * d_dim + (unsigned long long)kvh * d_dim;
        for (size_t d = 0; d < d_dim; ++d) {
            od[d] = __float2half(__half2float(od[d]) + wgt * __half2float(vc_row[d]));
        }
    }
}
