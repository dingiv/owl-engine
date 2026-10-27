// ============================================================================
// **top-32 split-K 逐行候选**(2026-10-26 分布式采样用;K=32 twin)。
//
// 与 `dflash_topk.cu` 的 K=16 版**同体同算术序** —— 那份是权威实现与注释
// 出处(全序律/锦标赛/uint4 载入/平局取小索引),本文件只把常量改成 32 并
// 换名字。为什么需要 32:host 采样器的 `topk` 缺省 20 > 16;候选集必须
// **包含全局 top-k**,否则采样分布不等价。
//
// 入口契约:
//   owl_topk32_split_f16(x[vocab] f16, hist[H] u32, rep_penalty[1] f32,
//                        vocab, nparts, rows, hist_cap, vocab_offset,
//                        tmp[2*rows*nparts*K])
//   owl_topk32_merge_f16(tmp, nparts, rows, vocab, out[2*rows*K])
// 输出:值区 [rows*K) + 索引区 [rows*K, 2*rows*K)(f32 过线)。
//
// **设备端重复惩罚(2026-10-26)**:必须在 top-K 选择**之前**施加 —— 被惩罚
// 的重复 token 可能从第 1 名掉到第 33 名,原第 33 名必须补进候选。hist 是
// **全局 token id**(升序去重,尾部 u32::MAX);本 rank 的 local index 经
// `vocab_offset` 映射成全局 id。rep_penalty <= 1.0 时整段跳过(旧路径逐位
// 不变)。rep_penalty 是**设备指针**(图输入槽内容可逐轮变,不重捕获)。
// ============================================================================
#include <cuda_fp16.h>

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef unsigned short uint16_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

#define TOPK32_BLOCK 256
#define TOPK32_K 32
#define TOPK32_NWARP (TOPK32_BLOCK / 32)
#define TOPK32_HIST 256

// ---------------------------------------------------------------------------
// 2b) split-K 逐行 top-K(K=16 烘焙)—— **段一**(2026-10-14 D2 性能战线)
//
// 动机:上面 `owl_topk16_f16` 的 grid=(rows,1,1) 结构性问题 —— rows = DEPTH = 7
// ⇒ 只用 7/132 SM;实测(CUPTI 图内逐核)1067.9 µs/次,而 **纯扫描地板仅 16.5 µs**
// (3.48 MB @ 224 GB/s)。三段分解定谳(详见 crates/kernels/tests/topk16_probe.rs):
//   ① **旧插入法动态索引** ⇒ ptxas 落 128 B 栈帧(局部内存),比寄存器展开慢 6.7×;
//   ② **每线程 1 个在飞载入** ⇒ 970 元素/线程全被访存延迟串死(纯扫描地板同结构
//      却快 100×:小循环体能被 ptxas 软件流水);
//   ③ **16 轮 smem argmax 归并 = 144 次 __syncthreads ≈ 20 µs/块**(同步墙)。
// 对策 = `rows×nparts` 网格(每块只扫 vocab/nparts)+ uint4 向量载入 +
//       寄存器全展开插入 + **warp shuffle 锦标赛**(零 syncthreads)。
//
// 全序律(必须):插入按 **(值降, 索引升) 严格全序**。只比值会让同值不同索引
// 乱序 ⇒ 锦标赛"winner 恒在持方 lane 的 slot 0"前提破裂 ⇒ 同值被重复选中
// (实测 112 槽中差 1)。平局取小索引,与 owl_argmax / torch.topk(sorted) 同律。
//
// 输出单缓冲(`tmp`;解释器单输出槽约定):
//   tmp[0, N)          值区   f32, N = rows*nparts*K,行主 [row*nparts+part]*K + k
//   tmp[N, 2N)         索引区 **int32 位型**(零转换;契约 5 的 f32 化在段二做)
// ---------------------------------------------------------------------------
#define TOPK32_NWARP (TOPK32_BLOCK / 32)

/// (值降, 索引升)严格全序比较:真 = (v,iv) 优于 (w,iw)
__device__ __forceinline__ bool topk32_gt(float v, int iv, float w, int iw) {
    return (v > w) || (v == w && iv < iw);
}

/// hist 已升序去重;哨兵 u32::MAX 在尾部,不会与真实 id 相等。
__device__ __forceinline__ bool topk32_hist_has(
    const unsigned int* __restrict__ hist, int n, unsigned int gid) {
    int lo = 0, hi = n;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (hist[mid] < gid) lo = mid + 1; else hi = mid;
    }
    return lo < n && hist[lo] == gid;
}

/// llama.cpp 同语义:正 logit 除,负 logit 乘;每个不同 token 只罚一次
/// (host 的 `penalty_history` 已去重)。
__device__ __forceinline__ float topk32_rep_penalty(
    float v, unsigned int gid, const unsigned int* __restrict__ hist, int n, float rep) {
    if (!topk32_hist_has(hist, n, gid)) return v;
    return v > 0.0f ? v / rep : v * rep;
}

/// 寄存器 top-16 插入(全展开常量索引;tk_tour 的"winner 在 slot 0"前提靠它)
__device__ __forceinline__ void topk32_push(float v, int ii, float* tv, int* ti) {
    if (topk32_gt(v, ii, tv[TOPK32_K - 1], ti[TOPK32_K - 1])) {
        #pragma unroll
        for (int k = 0; k < TOPK32_K; ++k) {
            if (topk32_gt(v, ii, tv[k], ti[k])) {
                const float t = tv[k];
                const int t2 = ti[k];
                tv[k] = v; ti[k] = ii; v = t; ii = t2;
            }
        }
    }
}

/// warp 内 16 轮锦标赛:第 k 名由 lane k 写入 out_v/out_i(**零 syncthreads**;
/// 全序律保证持方 winner 恒在 slot 0 ⇒ 弹出只需左移)
__device__ __forceinline__ void topk32_tour(float* tv, int* ti, int lane, int sid,
                                              float* out_v, int* out_i) {
    #pragma unroll 1
    for (int k = 0; k < TOPK32_K; ++k) {
        float bv = tv[0];
        int bi = ti[0];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int   oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (topk32_gt(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == k) { out_v[k] = bv; out_i[k] = bi; }
        if (ti[0] == bi) {   // 本 lane 持 winner(索引唯一;哨兵可多 lane 同持)
            #pragma unroll
            for (int j = 0; j < TOPK32_K - 1; ++j) { tv[j] = tv[j + 1]; ti[j] = ti[j + 1]; }
            tv[TOPK32_K - 1] = -3.4e38f; ti[TOPK32_K - 1] = sid;
        }
    }
}

extern "C" __global__ void owl_topk32_split_f16(
    const __half* __restrict__ x,            // [rows, vocab](本 rank 半表)
    const unsigned int* __restrict__ hist,       // [hist_cap] 全局 id,升序去重
    const float* __restrict__ rep_penalty,   // [1](设备指针;内容逐轮可换)
    int vocab,
    int nparts,
    int rows,
    int hist_cap,
    int vocab_offset,
    float* __restrict__ tmp) {               // [2*rows*nparts*K]:值区 + 索引区
    const int row  = blockIdx.x / nparts;
    const int part = blockIdx.x % nparts;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int chunk = (vocab + nparts - 1) / nparts;
    const int lo = part * chunk;
    const int hi = min(vocab, lo + chunk);
    const int n = hi > lo ? (hi - lo) : 0;   // nparts > vocab 时尾部空段(测试小词表)
    const __half* base = x + (unsigned long long)row * vocab + (lo < vocab ? lo : vocab);

    // 惩罚集:每 block 一次载入 shared(1 KiB);rep<=1 时连载入/同步都跳过。
    const float rp = rep_penalty[0];
    const bool penalize = rp > 1.0f;
    const int hn = hist_cap < 0 ? 0 : (hist_cap > TOPK32_HIST ? TOPK32_HIST : hist_cap);
    __shared__ unsigned int hist_s[TOPK32_HIST];
    if (penalize) {
        for (int i = tid; i < hn; i += TOPK32_BLOCK) hist_s[i] = hist[i];
        __syncthreads();
    }

    float tv[TOPK32_K];
    int   ti[TOPK32_K];
    #pragma unroll
    for (int k = 0; k < TOPK32_K; ++k) { tv[k] = -3.4e38f; ti[k] = vocab; }

    // 向量段:uint4 = 8 half/线程/迭代(16B 对齐时;否则全程标量兜底)
    const int nvec = (((unsigned long long)base & 15ull) == 0ull) ? (n >> 3) : 0;
    const uint4* vb = (const uint4*)base;
    #pragma unroll 1
    for (int v = tid; v < nvec; v += TOPK32_BLOCK) {
        const uint4 u = vb[v];
        const int i0 = lo + v * 8;
        float f[8];
        f[0] = __half2float(__ushort_as_half((unsigned short)(u.x & 0xffffu)));
        f[1] = __half2float(__ushort_as_half((unsigned short)(u.x >> 16)));
        f[2] = __half2float(__ushort_as_half((unsigned short)(u.y & 0xffffu)));
        f[3] = __half2float(__ushort_as_half((unsigned short)(u.y >> 16)));
        f[4] = __half2float(__ushort_as_half((unsigned short)(u.z & 0xffffu)));
        f[5] = __half2float(__ushort_as_half((unsigned short)(u.z >> 16)));
        f[6] = __half2float(__ushort_as_half((unsigned short)(u.w & 0xffffu)));
        f[7] = __half2float(__ushort_as_half((unsigned short)(u.w >> 16)));
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int ii = i0 + j;
            float v = f[j];
            if (penalize)
                v = topk32_rep_penalty(v, (unsigned int)(vocab_offset + ii), hist_s, hn, rp);
            topk32_push(v, ii, tv, ti);
        }
    }
    // 标量尾(含未对齐兜底)
    for (int i = lo + nvec * 8 + tid; i < hi; i += TOPK32_BLOCK) {
        float v = __half2float(base[i - lo]);
        if (penalize)
            v = topk32_rep_penalty(v, (unsigned int)(vocab_offset + i), hist_s, hn, rp);
        topk32_push(v, i, tv, ti);
    }

    // 块内归并:各 warp 锦标赛 → smem 候选 → warp0 再锦标赛 → 直写 tmp
    // (块内跨 warp 只 1 次同步;旧 16 轮 smem argmax 是 144 次)
    __shared__ float sw[TOPK32_NWARP * TOPK32_K];
    __shared__ int   swi[TOPK32_NWARP * TOPK32_K];
    const int warp = tid >> 5;
    topk32_tour(tv, ti, lane, vocab, sw + warp * TOPK32_K, swi + warp * TOPK32_K);
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int k = 0; k < TOPK32_K; ++k) { tv[k] = -3.4e38f; ti[k] = vocab; }
        for (int i = lane; i < TOPK32_NWARP * TOPK32_K; i += 32)
            topk32_push(sw[i], swi[i], tv, ti);
        const long long bv = (long long)(row * nparts + part) * TOPK32_K;
        const long long bi = (long long)rows * nparts * TOPK32_K + bv;
        topk32_tour(tv, ti, lane, vocab, tmp + bv, (int*)(tmp + bi));
    }
}

// ---------------------------------------------------------------------------
// 2c) split-K 段二:**一行一 warp**(block=32),把 nparts×16 候选归并成 top-16
//      —— 零 __syncthreads(warp 内 shuffle 锦标赛)。grid = rows。
// 契约与 2) 同:输出 [2*rows*K](值区行主 + 索引区,索引 **f32**,供 SliceView
// 行内对齐切分);平局取小索引。
// ---------------------------------------------------------------------------
extern "C" __global__ void owl_topk32_merge_f16(
    const float* __restrict__ tmp,  // 段一输出 [2*rows*nparts*K]
    int nparts,
    int rows,
    int vocab,
    float* __restrict__ out) {      // [2*rows*K]
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int n = nparts * TOPK32_K;
    const float* pvs = tmp + (long long)row * n;
    const int*   pis = (const int*)(tmp + (long long)rows * n + (long long)row * n);

    float tv[TOPK32_K];
    int   ti[TOPK32_K];
    #pragma unroll
    for (int k = 0; k < TOPK32_K; ++k) { tv[k] = -3.4e38f; ti[k] = vocab; }
    for (int i = lane; i < n; i += 32) topk32_push(pvs[i], pis[i], tv, ti);

    // 值区直写(out 值半区本就是 f32);索引经 smem 转 f32(契约 5)
    __shared__ int sidx[TOPK32_K];
    topk32_tour(tv, ti, lane, vocab, out + row * TOPK32_K, sidx);
    __syncwarp();
    if (lane < TOPK32_K)
        out[rows * TOPK32_K + row * TOPK32_K + lane] = (float)sidx[lane];
}
