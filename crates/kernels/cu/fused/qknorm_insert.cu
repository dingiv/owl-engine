// owl 融合核族:qk-norm + rope + KV 插池融合(2026-10-01 C1 立项;2026-10-13 C 轮自 cu/owl/fused.cu
// 按**算子族**拆出 —— 一算子一 `.cu`,与 ops/attn.rs 的声明对望)。
//
// Ampere-first 准则(driver 立项同步确立):
//   ① 最小发射数优先于最小流量 —— 一次 DRAM 往返,中间值全寄存器;
//   ② 128-bit 向量化访问(half2/uint4);
//   ③ grid 铺满 SM(按元素/按 (t,head) 开,不整行串行);
//   ④ float 中间精度(与被替核同式,口径可对拍)。
//
// 本文件核族(全部单输出,SSA 契约友好):
//   owl_qknorm_rope_kv_insert_f16          经典池(classic 布局)
//   owl_qknorm_rope_kv_insert_f16_fp8kv    e4m3 主池(classic)
//   owl_qknorm_rope_kv_insert_f16_knhd     f16 池(kNHD 单布局)
//   owl_qknorm_rope_kv_insert_f16_fp8kv_knhd  e4m3 池(kNHD 单布局)
//   —— decode 融合插池:norm+rope+scatter 一发(池布局由后缀决定)
//
// 输出块固定末参(槽序契约 4)。
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// ----------------------------------------------------------------------------
// owl_qknorm_rope_kv_insert_f16(Wave-2 头号;port 源 = vLLM
// fused_minimax_m3_qknorm_rope_kv_insert_kernel.cu 的 (token, head-slot)
// warp 结构,适配 owl gated 布局与 classic cache 寻址):
//
//   q 头:blockIdx.y ∈ [0, Hq)     —— q_raw 的 value 半段 strided 读 →
//                                    qk-norm(×(1+w)^{w_off})+ rope → q_out
//   k 头:blockIdx.y ∈ [Hq, Hq+Hkv) —— k 的 qk-norm + rope → **key_cache
//                                    slot 散写**;同块捎带 v → value_cache
//
// 替三发:norm_rope(q)+ norm_rope(k)+ reshape_and_cache(k,v)。
// q_out = 尾参(v1 注意力输入;cache 写 = 原地副作用,单流保序 ——
// v1 为本核输出块的消费者,树序天然后置)。
// cache 寻址 = reshape_and_cache 逐字(classic:kc [nb,Hkv,D/x,page,x],
// vc [nb,Hkv,D,page];slot<0 = padding 跳写,k 头仍出 q…… k 头 return
// 前 q_out 与本核无关,直接 return 安全)。
// grid (T, Hq+Hkv, 1);block (hd, 1, 1);smem = hd·4B(行归约)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_qknorm_rope_kv_insert_f16(
    const __half* __restrict__ q_raw,   // [T, Hq·2HD](per-head [value|gate])
    const __half* __restrict__ k,       // [T, Hkv·HD]
    const __half* __restrict__ v,       // [T, Hkv·HD]
    __half* __restrict__ key_cache,     // [nb, Hkv, hd/x, page, x]
    __half* __restrict__ value_cache,   // [nb, Hkv, hd, page]
    const float* __restrict__ slots,    // [T]
    const __half* __restrict__ q_w,     // [hd]
    const __half* __restrict__ k_w,     // [hd]
    const __half* __restrict__ cos_t,   // [max_pos, half]
    const __half* __restrict__ sin_t,   // [max_pos, half]
    const float* __restrict__ pos,      // [T]
    float eps,
    int hkv, int half, int page, int w_off,
    __half* __restrict__ q_out)         // [T, Hq·hd](尾参契约 4)
{
    const size_t t = blockIdx.x;
    const size_t head = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const size_t hq = gridDim.y - (size_t)hkv;
    const long long slot = (long long)slots[t];

    if (head < hq) {
        // ---- q:value 半段 strided 读 + norm + rope → q_out ----
        const size_t row_stride = hq * 2 * hd;
        const __half* xs = q_raw + t * row_stride + head * 2 * hd;
        extern __shared__ float smem[];
        const float xf = __half2float(xs[d]);
        smem[d] = xf * xf;
        __syncthreads();
        for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) smem[d] += smem[d + s2];
            __syncthreads();
        }
        const float inv = rsqrtf(smem[0] / (float)hd + eps);
        const float wf = __half2float(q_w[d]);
        const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
        const size_t p = (size_t)pos[t];
        const size_t base = (t * hq + head) * hd;
        if (d < (size_t)half) {
            const float xb = __half2float(xs[d + half]);
            const float wb = __half2float(q_w[d + half]);
            const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
            const float cf = __half2float(cos_t[p * half + d]);
            const float sf = __half2float(sin_t[p * half + d]);
            q_out[base + d] = __float2half(n * cf - nb * sf);
            q_out[base + d + half] = __float2half(nb * cf + n * sf);
        } else if (d >= 2 * (size_t)half) {
            q_out[base + d] = __float2half(n);
        }
            return;
    }

    // ---- k 头:qk-norm + rope → key_cache;捎带 v → value_cache ----
    if (slot < 0) return;   // padding 跳写(K0 契约)
    const size_t kh = head - hq;
    const __half* ks = k + t * (size_t)hkv * hd + kh * hd;
    extern __shared__ float smem[];
    const float kf = __half2float(ks[d]);
    smem[d] = kf * kf;
    __syncthreads();
    for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
        if (d < s2) smem[d] += smem[d + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)hd + eps);
    const float wf = __half2float(k_w[d]);
    float n = kf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    if (d < (size_t)half) {
        // 配对 (d, d+half):每 thread 只写自己的 d 位(伙伴位由 d+half 写)
        const float xb = __half2float(ks[d + half]);
        const float wb = __half2float(k_w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        n = n * cf - nb * sf;
    } else if (d >= 2 * (size_t)half) {
        // partial 维直通(n 已是 norm 值)
    } else {
        // d ∈ [half, 2half):伙伴 = d - half(基值 = 自己 n[d],旋转伙伴 n[d-half]
        // —— 2026-10-02 转置案修复:曾写 nb*cf + n*sf(基/伙伴互换,pos=0 时
        // 直接写 n[d-half],引擎 kv-hex 3-4% 偏差真凶)
        const float xb = __half2float(ks[d - half]);
        const float wb = __half2float(k_w[d - half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d - half]);
        const float sf = __half2float(sin_t[p * half + d - half]);
        n = n * cf + nb * sf;
    }
    // 诊断(隔离 hd128 档 slot 2 取证):分段值打印
    const int block_idx = (int)(slot / page);
    const int off = (int)(slot % page);
    const size_t kk = ((block_idx * (size_t)hkv + kh) * (hd / 8) + d / 8) * page * 8
                    + off * 8 + d % 8;
    key_cache[kk] = __float2half(n);
    // v 捎带拷贝(线性寻址)
    const size_t vi = ((block_idx * (size_t)hkv + kh) * hd + d) * page + off;
    value_cache[vi] = v[t * (size_t)hkv * hd + kh * hd + d];
}

// ----------------------------------------------------------------------------
// owl_qknorm_rope_kv_insert_f16_fp8kv(B6.3 收口;2026-10-10):fp8 e4m3
// 主池变体。数学逐式同 f16 版(同一 norm/rope);差异仅池写 ——
//   key_cache/value_cache = e4m3 1B/elem(u8 寻址,元素序不变),
//   池写 = __nv_fp8_e4m3(value) SATFINITE(与 K0 写核同转换)。
// 背景:fp8 池下 f16 版 2B 存储落在 2× 字节偏移 —— 真槽区永不落笔
// (kv-dump 实证 decode 槽全零)+ f16 位型毒液洒 2s 槽区 = decode 崩坏
// 真凶(B6.2 曾覆盖 K0/chunked/v2 读三路,独漏此融合插池路)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_qknorm_rope_kv_insert_f16_fp8kv(
    const __half* __restrict__ q_raw,   // [T, Hq·2HD](per-head [value|gate])
    const __half* __restrict__ k,       // [T, Hkv·HD]
    const __half* __restrict__ v,       // [T, Hkv·HD]
    unsigned char* __restrict__ key_cache,   // e4m3 [nb, Hkv, hd/x, page, x]
    unsigned char* __restrict__ value_cache, // e4m3 [nb, Hkv, hd, page]
    const float* __restrict__ slots,    // [T]
    const __half* __restrict__ q_w,     // [hd]
    const __half* __restrict__ k_w,     // [hd]
    const __half* __restrict__ cos_t,   // [max_pos, half]
    const __half* __restrict__ sin_t,   // [max_pos, half]
    const float* __restrict__ pos,      // [T]
    float eps,
    int hkv, int half, int page, int w_off,
    __half* __restrict__ q_out)         // [T, Hq·hd](尾参契约 4)
{
    const size_t t = blockIdx.x;
    const size_t head = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const size_t hq = gridDim.y - (size_t)hkv;
    const long long slot = (long long)slots[t];

    if (head < hq) {
        // ---- q:value 半段 strided 读 + norm + rope → q_out ----
        const size_t row_stride = hq * 2 * hd;
        const __half* xs = q_raw + t * row_stride + head * 2 * hd;
        extern __shared__ float smem[];
        const float xf = __half2float(xs[d]);
        smem[d] = xf * xf;
        __syncthreads();
        for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) smem[d] += smem[d + s2];
            __syncthreads();
        }
        const float inv = rsqrtf(smem[0] / (float)hd + eps);
        const float wf = __half2float(q_w[d]);
        const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
        const size_t p = (size_t)pos[t];
        const size_t base = (t * hq + head) * hd;
        if (d < (size_t)half) {
            const float xb = __half2float(xs[d + half]);
            const float wb = __half2float(q_w[d + half]);
            const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
            const float cf = __half2float(cos_t[p * half + d]);
            const float sf = __half2float(sin_t[p * half + d]);
            q_out[base + d] = __float2half(n * cf - nb * sf);
            q_out[base + d + half] = __float2half(nb * cf + n * sf);
        } else if (d >= 2 * (size_t)half) {
            q_out[base + d] = __float2half(n);
        }
            return;
    }

    // ---- k 头:qk-norm + rope → key_cache;捎带 v → value_cache ----
    if (slot < 0) return;   // padding 跳写(K0 契约)
    const size_t kh = head - hq;
    const __half* ks = k + t * (size_t)hkv * hd + kh * hd;
    extern __shared__ float smem[];
    const float kf = __half2float(ks[d]);
    smem[d] = kf * kf;
    __syncthreads();
    for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
        if (d < s2) smem[d] += smem[d + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)hd + eps);
    const float wf = __half2float(k_w[d]);
    float n = kf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    if (d < (size_t)half) {
        // 配对 (d, d+half):每 thread 只写自己的 d 位(伙伴位由 d+half 写)
        const float xb = __half2float(ks[d + half]);
        const float wb = __half2float(k_w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        n = n * cf - nb * sf;
    } else if (d >= 2 * (size_t)half) {
        // partial 维直通(n 已是 norm 值)
    } else {
        // d ∈ [half, 2half):伙伴 = d - half(基值 = 自己 n[d],旋转伙伴 n[d-half]
        // —— 2026-10-02 转置案修复:曾写 nb*cf + n*sf(基/伙伴互换,pos=0 时
        // 直接写 n[d-half],引擎 kv-hex 3-4% 偏差真凶)
        const float xb = __half2float(ks[d - half]);
        const float wb = __half2float(k_w[d - half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d - half]);
        const float sf = __half2float(sin_t[p * half + d - half]);
        n = n * cf + nb * sf;
    }
    const int block_idx = (int)(slot / page);
    const int off = (int)(slot % page);
    const size_t kk = ((block_idx * (size_t)hkv + kh) * (hd / 8) + d / 8) * page * 8
                    + off * 8 + d % 8;
    key_cache[kk] = __nv_fp8_e4m3(n).__x;
    // v 捎带拷贝(线性寻址;e4m3 转换)
    const size_t vi = ((block_idx * (size_t)hkv + kh) * hd + d) * page + off;
    value_cache[vi] = __nv_fp8_e4m3(__half2float(v[t * (size_t)hkv * hd + kh * hd + d])).__x;
}
// ----------------------------------------------------------------------------
// owl_qknorm_rope_kv_insert_f16_knhd / _f16_fp8kv_knhd(kv布局统一契约 P1)。
// 数学逐式同 classic 版(同一 norm/rope/捎带 v);唯一差异 = 池写寻址:
//   kNHD:K/V 同构 [nb, page, Hkv, hd] → off = (block·page + off_in_page)
//         ·Hkv·hd + kh·hd + d(K/V 同址,classic 的双式寻址合一线)。
// x 交错退役;parity 门 = q_out 与两池映射回读逐位一致。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_qknorm_rope_kv_insert_f16_knhd(
    const __half* __restrict__ q_raw,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ key_cache,     // [nb, page, Hkv, hd]
    __half* __restrict__ value_cache,   // [nb, page, Hkv, hd]
    const float* __restrict__ slots,
    const __half* __restrict__ q_w,
    const __half* __restrict__ k_w,
    const __half* __restrict__ cos_t,
    const __half* __restrict__ sin_t,
    const float* __restrict__ pos,
    float eps,
    int hkv, int half, int page, int w_off,
    __half* __restrict__ q_out)
{
    const size_t t = blockIdx.x;
    const size_t head = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const size_t hq = gridDim.y - (size_t)hkv;
    const long long slot = (long long)slots[t];

    if (head < hq) {
        const size_t row_stride = hq * 2 * hd;
        const __half* xs = q_raw + t * row_stride + head * 2 * hd;
        extern __shared__ float smem[];
        const float xf = __half2float(xs[d]);
        smem[d] = xf * xf;
        __syncthreads();
        for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) smem[d] += smem[d + s2];
            __syncthreads();
        }
        const float inv = rsqrtf(smem[0] / (float)hd + eps);
        const float wf = __half2float(q_w[d]);
        const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
        const size_t p = (size_t)pos[t];
        const size_t base = (t * hq + head) * hd;
        if (d < (size_t)half) {
            const float xb = __half2float(xs[d + half]);
            const float wb = __half2float(q_w[d + half]);
            const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
            const float cf = __half2float(cos_t[p * half + d]);
            const float sf = __half2float(sin_t[p * half + d]);
            q_out[base + d] = __float2half(n * cf - nb * sf);
            q_out[base + d + half] = __float2half(nb * cf + n * sf);
        } else if (d >= 2 * (size_t)half) {
            q_out[base + d] = __float2half(n);
        }
            return;
    }

    if (slot < 0) return;
    const size_t kh = head - hq;
    const __half* ks = k + t * (size_t)hkv * hd + kh * hd;
    extern __shared__ float smem[];
    const float kf = __half2float(ks[d]);
    smem[d] = kf * kf;
    __syncthreads();
    for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
        if (d < s2) smem[d] += smem[d + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)hd + eps);
    const float wf = __half2float(k_w[d]);
    float n = kf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    if (d < (size_t)half) {
        const float xb = __half2float(ks[d + half]);
        const float wb = __half2float(k_w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        n = n * cf - nb * sf;
    } else if (d >= 2 * (size_t)half) {
    } else {
        const float xb = __half2float(ks[d - half]);
        const float wb = __half2float(k_w[d - half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d - half]);
        const float sf = __half2float(sin_t[p * half + d - half]);
        n = n * cf + nb * sf;
    }
    const int block_idx = (int)(slot / page);
    const int off = (int)(slot % page);
    const size_t kk = ((block_idx * (size_t)page + off) * (size_t)hkv + kh) * hd + d;
    key_cache[kk] = __float2half(n);
    value_cache[kk] = v[t * (size_t)hkv * hd + kh * hd + d];
}

extern "C" __global__ void owl_qknorm_rope_kv_insert_f16_fp8kv_knhd(
    const __half* __restrict__ q_raw,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    unsigned char* __restrict__ key_cache,     // e4m3 [nb, page, Hkv, hd]
    unsigned char* __restrict__ value_cache,
    const float* __restrict__ slots,
    const __half* __restrict__ q_w,
    const __half* __restrict__ k_w,
    const __half* __restrict__ cos_t,
    const __half* __restrict__ sin_t,
    const float* __restrict__ pos,
    float eps,
    int hkv, int half, int page, int w_off,
    __half* __restrict__ q_out)
{
    const size_t t = blockIdx.x;
    const size_t head = blockIdx.y;
    const size_t d = threadIdx.x;
    const size_t hd = blockDim.x;
    const size_t hq = gridDim.y - (size_t)hkv;
    const long long slot = (long long)slots[t];

    if (head < hq) {
        const size_t row_stride = hq * 2 * hd;
        const __half* xs = q_raw + t * row_stride + head * 2 * hd;
        extern __shared__ float smem[];
        const float xf = __half2float(xs[d]);
        smem[d] = xf * xf;
        __syncthreads();
        for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) smem[d] += smem[d + s2];
            __syncthreads();
        }
        const float inv = rsqrtf(smem[0] / (float)hd + eps);
        const float wf = __half2float(q_w[d]);
        const float n = xf * inv * (w_off ? (wf + 1.0f) : wf);
        const size_t p = (size_t)pos[t];
        const size_t base = (t * hq + head) * hd;
        if (d < (size_t)half) {
            const float xb = __half2float(xs[d + half]);
            const float wb = __half2float(q_w[d + half]);
            const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
            const float cf = __half2float(cos_t[p * half + d]);
            const float sf = __half2float(sin_t[p * half + d]);
            q_out[base + d] = __float2half(n * cf - nb * sf);
            q_out[base + d + half] = __float2half(nb * cf + n * sf);
        } else if (d >= 2 * (size_t)half) {
            q_out[base + d] = __float2half(n);
        }
            return;
    }

    if (slot < 0) return;
    const size_t kh = head - hq;
    const __half* ks = k + t * (size_t)hkv * hd + kh * hd;
    extern __shared__ float smem[];
    const float kf = __half2float(ks[d]);
    smem[d] = kf * kf;
    __syncthreads();
    for (size_t s2 = hd / 2; s2 > 0; s2 >>= 1) {
        if (d < s2) smem[d] += smem[d + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(smem[0] / (float)hd + eps);
    const float wf = __half2float(k_w[d]);
    float n = kf * inv * (w_off ? (wf + 1.0f) : wf);
    const size_t p = (size_t)pos[t];
    if (d < (size_t)half) {
        const float xb = __half2float(ks[d + half]);
        const float wb = __half2float(k_w[d + half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d]);
        const float sf = __half2float(sin_t[p * half + d]);
        n = n * cf - nb * sf;
    } else if (d >= 2 * (size_t)half) {
    } else {
        const float xb = __half2float(ks[d - half]);
        const float wb = __half2float(k_w[d - half]);
        const float nb = xb * inv * (w_off ? (wb + 1.0f) : wb);
        const float cf = __half2float(cos_t[p * half + d - half]);
        const float sf = __half2float(sin_t[p * half + d - half]);
        n = n * cf + nb * sf;
    }
    const int block_idx = (int)(slot / page);
    const int off = (int)(slot % page);
    const size_t kk = ((block_idx * (size_t)page + off) * (size_t)hkv + kh) * hd + d;
    key_cache[kk] = __nv_fp8_e4m3(n).__x;
    value_cache[kk] = __nv_fp8_e4m3(__half2float(v[t * (size_t)hkv * hd + kh * hd + d])).__x;
}
