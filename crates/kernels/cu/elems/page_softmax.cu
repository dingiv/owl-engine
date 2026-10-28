//! **page-softmax 召回打分**（2026-10-28；参照实现 `page_softmax_scores` 的设备版）。
//!
//! 语义（与 `owl_store::score_blocks(ScoreMode::SoftmaxMean)` 逐式同源）：
//!
//! ```text
//! z[t, h, b] = dot(q[t, l, h], mk[b, l, g(h)]) · scale        // g(h) = GQA 映射
//! p[t, h, b] = exp(z − max_b z) / Σ_b' exp(z[t,h,b'] − max_b z)
//! out[b]    += (1/(L·hq)) · Σ_t Σ_h p[t, h, b]
//! ```
//!
//! **为什么必须上 GPU**：softmax 非线性 ⇒ 不能像 cosine 那样把 token 与 GQA 头
//! 折叠成一次点积；host 侧代价 = `W × L × hq × blocks × hd`
//! （72k/2240 块/W=64 ≈ 28 GFLOP，实测 4.4 s/边界）。设备侧同量 = 几 ms。
//!
//! 三段发射（每层一遍，scratch = `[W, hq, nb]` f16，逐层复用）：
//!
//! | # | 核 | grid | 作用 |
//! |---|---|---|---|
//! | ① | `owl_kvscore_logits_f16` | `(ceil(nb/BT), W)` | 块瓦片 × 单 token：mk 瓦片进 shared，逐 (h, b) 出 z |
//! | ② | `owl_kvscore_maxsum` | `(W, hq)` | 逐 (t,h) 扫整行 z ⇒ `(m, s)` |
//! | ③ | `owl_kvscore_acc` | `(ceil(nb/BT))` | 逐块累加 `exp(z−m)/s · 1/(L·hq)`（**无原子**：每块只属一个 CTA） |
//!
//! 分块而不是「一 CTA 一 (t,l) 扫全部块」的原因：后者会让每层 mk 被 `W×L` 个
//! CTA 各读一遍（8192 × 9 MB ≈ 74 GB）；块瓦片化后 mk 只按 token-tile 数
//! （`W/WT` 次）重读。
//!
//! 精度口径 = 参照实现的**显式 FP16 操作数边界**：q 取 f16，mk 取 f16，
//! 点积在 f32 里累加（与 host 参考的差异只有 q 的 f16 取整）。

#include <cuda_fp16.h>

#ifndef OWL_KVSCORE_BT
#define OWL_KVSCORE_BT 32
#endif

/// ① logits：z[t][h][b] = (q[t][h] · mk[b][g(h)]) · scale
///
/// grid = (ceil(nb/BT), hkv)；block = 256。
///
/// **瓦片取 (块, KV 头)**：mk 瓦片（`BT×hd` f16）进 shared **只读一次**，
/// 再在 CTA 内对所有 token × 该组查询头复用；q 从 global 读（`W×hq×hd`
/// 整族 <1 MB，天然驻 L2）。前一版按 (块, 单 token) 切 ⇒ 每层 mk 被 4480 个
/// CTA 各读一遍（≈9 GB/层，实测 104 ms/边界）；改完 ≈10 ms。
extern "C" __global__ void owl_kvscore_logits_f16(
    const __half* __restrict__ q,      // [W, hq, hd]（**已去位置**的 q̃，f16）
    const __half* __restrict__ mk,     // [nb, hkv, hd]（本层 mean-K，f16）
    const int* __restrict__ gqa,       // [hq] ⇒ kv 头
    int W, int hq, int hkv, int hd, int nb,
    float scale,
    __half* __restrict__ z)            // [W, hq, nb]
{
    const int b0 = blockIdx.x * OWL_KVSCORE_BT;
    const int g = blockIdx.y;
    if (b0 >= nb || g >= hkv) return;
    const int bt = min(OWL_KVSCORE_BT, nb - b0);

    extern __shared__ __half smem[];
    // mk 瓦片 → shared（每块一行，行宽 hd）
    for (int i = threadIdx.x; i < bt * hd; i += blockDim.x) {
        const int b = i / hd, d = i % hd;
        smem[i] = mk[((size_t)(b0 + b) * hkv + g) * hd + d];
    }
    __syncthreads();

    // 该 kv 头对应哪些查询头（GQA 比 = hq/hkv）
    const int grp = hq / hkv;
    const int h_lo = g * grp, h_hi = min(hq, h_lo + grp);
    const int hd2 = hd >> 1; // half2 元素对
    // 逐 (t, h) 点积：**每个块 8 线程协作**（tid = b*8 + s，块内 8-lane 归约）
    // —— 前一版每轮只有 `bt` 个线程干活（256 线程里 32 个）⇒ 76 ms；
    // 满线程 + half2 之后落到 ~10 ms 量级。
    const __half2* sm2 = reinterpret_cast<const __half2*>(smem);
    const int b = threadIdx.x >> 3;   // 0..31（= BT/1，8 线程一组）
    const int sub = threadIdx.x & 7;  // 组内 lane
    if (b >= bt) return;
    for (int t = 0; t < W; ++t) {
        for (int h = h_lo; h < h_hi; ++h) {
            const __half2* q2 = reinterpret_cast<const __half2*>(q + ((size_t)t * hq + h) * hd);
            const __half2* k2 = sm2 + (size_t)b * hd2;
            float2 acc = make_float2(0.0f, 0.0f);
            for (int i = sub; i < hd2; i += 8) {
                const float2 qv = __half22float2(q2[i]);
                const float2 kv = __half22float2(k2[i]);
                acc.x += qv.x * kv.x;
                acc.y += qv.y * kv.y;
            }
            float v = acc.x + acc.y;
            v += __shfl_down_sync(0xffffffffu, v, 4);
            v += __shfl_down_sync(0xffffffffu, v, 2);
            v += __shfl_down_sync(0xffffffffu, v, 1);
            if (sub == 0) {
                z[((size_t)t * hq + h) * nb + (b0 + b)] = __float2half(v * scale);
            }
        }
    }
}

/// ② 逐 (t,h) 的 max / Σexp：`ms_out[t][h][0]=m, [1]=s`
///
/// grid = (W, hq)；block = 256。两趟扫行（行 = nb 个 f16，几 KB，但全局读两次
/// 只有 55 MB/层 ≪ 计算量）。
extern "C" __global__ void owl_kvscore_maxsum(
    const __half* __restrict__ z, // [W, hq, nb]
    int W, int hq, int nb,
    float* __restrict__ ms_out)   // [W, hq, 2]
{
    const int t = blockIdx.x, h = blockIdx.y;
    if (t >= W || h >= hq) return;
    const __half* row = z + ((size_t)t * hq + h) * nb;
    __shared__ float sred[32];
    float m = -3.4e38f; // 不用 INFINITY/FLT_MAX:本仓 NVRTC 环境不带头文件常量
    for (int i = threadIdx.x; i < nb; i += blockDim.x) {
        const float v = __half2float(row[i]);
        m = fmaxf(m, v);
    }
    // 块内归约 max
    for (int off = 16; off > 0; off >>= 1) {
        const float other = __shfl_down_sync(0xffffffffu, m, off);
        m = fmaxf(m, other);
    }
    if ((threadIdx.x & 31) == 0) sred[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = -3.4e38f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; ++i) r = fmaxf(r, sred[i]);
        sred[0] = r;
    }
    __syncthreads();
    m = sred[0];
    if (!isfinite(m)) {
        if (threadIdx.x == 0) {
            ms_out[((size_t)t * hq + h) * 2 + 0] = 0.0f;
            ms_out[((size_t)t * hq + h) * 2 + 1] = 0.0f; // s = 0 ⇒ 下游 p = 0
        }
        return;
    }
    float s = 0.0f;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) {
        s += __expf(__half2float(row[i]) - m);
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
    }
    if ((threadIdx.x & 31) == 0) sred[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        float r = 0.0f;
        const int nw = (blockDim.x + 31) >> 5;
        for (int i = 0; i < nw; ++i) r += sred[i];
        ms_out[((size_t)t * hq + h) * 2 + 0] = m;
        ms_out[((size_t)t * hq + h) * 2 + 1] = r;
    }
}

/// ③ 逐块累加：`out[b] += Σ_{t,h} exp(z−m)/s · (1/(L·hq))`
///
/// grid = (ceil(nb/BT))；block = 128。**无原子**：每个 b 只被一个 CTA 写。
extern "C" __global__ void owl_kvscore_acc(
    const __half* __restrict__ z,  // [W, hq, nb]
    const float* __restrict__ ms,  // [W, hq, 2]
    int W, int hq, int nb, int layers,
    float* __restrict__ out)       // [nb]（跨层累加）
{
    const int b0 = blockIdx.x * OWL_KVSCORE_BT;
    // **t 维二分 × 网格**（grid.y = 分片数）：单 CTA 只做 `W/分片` 行 ⇒
    // 并行度从 70 CTA 提到几百，且每 CTA 只摊一次局部归约。
    const int nsplit = gridDim.y;
    const int t_lo = (int)((long long)W * blockIdx.y / nsplit);
    const int t_hi = (int)((long long)W * (blockIdx.y + 1) / nsplit);
    if (b0 >= nb) return;
    const int bt = min(OWL_KVSCORE_BT, nb - b0);
    __shared__ float acc[OWL_KVSCORE_BT];
    if (threadIdx.x < bt) acc[threadIdx.x] = 0.0f;
    __syncthreads();
    const float norm = 1.0f / ((float)layers * (float)hq);
    // 每个线程负责一个 / 多个块，沿 (t,h) 扫
    for (int i = threadIdx.x; i < bt; i += blockDim.x) {
        float local = 0.0f;
        for (int t = t_lo; t < t_hi; ++t) {
            for (int h = 0; h < hq; ++h) {
                const float s = ms[((size_t)t * hq + h) * 2 + 1];
                if (s <= 0.0f) continue;
                const float m = ms[((size_t)t * hq + h) * 2 + 0];
                const float v = __half2float(z[((size_t)t * hq + h) * nb + (b0 + i)]);
                local += __expf(v - m) / s;
            }
        }
        acc[i] = local;
    }
    __syncthreads();
    __syncthreads();
    for (int i = threadIdx.x; i < bt; i += blockDim.x) {
        // 逐层累加 + 跨 t 分片合并（atomic:分片数 × 块数 级,可忽略）
        atomicAdd(out + b0 + i, acc[i] * norm);
    }
}
