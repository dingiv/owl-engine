// ============================================================================
// DFlash2 草稿核族:逐行 top-K(K=16 烘焙;flashinfer topk 同位)(2026-10-13 C 轮自 cu/owl/dflash2.cu 按**算子族**
// 拆出 —— 一算子一 `.cu`,与 ops/elems.rs 的声明对望)。
//
// 上游语义锚 = packages/sglang python/sglang/srt/models/dflash.py +
// kernels/ops/speculative/dflash.py(sglang-dflash-patch 分支,逐式对拍)。
//
// 数值律:输入 f16/bf16,累计 f32,输出随槽(契约 5:索引量 f32 过线;词表
// id < 2²⁴ 精确)。平局取最小索引(与 owl_argmax / torch.topk sorted 语义
// 一致,host 对拍同律)。输出块固定末参(槽序契约 4)。
//
//   x [rows, vocab] f16 → vals [rows, 16] f32 + idx [rows, 16] f32
//
// 结构:块 = 一行;线程内 strided 扫描保本地 top-16;smem 汇 256×16 候选;
// 16 轮块内 argmax 归约选全局 top-16。topk16 保持 f16 唯一(lm_head 铸
// 边界前 logits 恒 f16,cast 保 cublas/topk)。
// ============================================================================
#include <cuda_fp16.h>

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

// ---------------------------------------------------------------------------
// 2) 逐行 top-K(K=16 烘焙)—— sglang _radix_topk(flashinfer topk)同位
//
//   x [rows, vocab] f16 → vals [rows, 16] f32 + idx [rows, 16] f32
//
// 结构:块 = 一行;线程内 strided 扫描保本地 top-16(插入,降序,严格 >
// 才替换 → 平局保低索引);smem 汇 256×16 候选;16 轮块内 argmax 归约
// 选全局 top-16(选中置 -inf 剔除)。smem = 16KB(值)+ 8KB(索引)。
// ---------------------------------------------------------------------------
#define DFLASH_TOPK_BLOCK 256
#define DFLASH_TOPK_K 16

extern "C" __global__ void owl_topk16_f16(
    const __half* __restrict__ x,   // [rows, vocab]
    int vocab,
    float* __restrict__ out) {      // [2*rows*K]:值区 [rows*K) 行主 + 索引区(平铺连续,供 SliceView 行内对齐切分)
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* xr = x + (long long)row * vocab;

    // 线程内 top-16(降序插入;严格 > 替换 = 平局保低索引)
    float tv[DFLASH_TOPK_K];
    int ti[DFLASH_TOPK_K];
    int tn = 0;
    for (int i = tid; i < vocab; i += DFLASH_TOPK_BLOCK) {
        const float v = __half2float(xr[i]);
        if (tn < DFLASH_TOPK_K || v > tv[tn - 1]) {
            // 插入位(二分;tv 全序 = (值降,索引升)——平局插等值段后,
            // 先见低索引占先)
            int lo = 0, hi = tn;
            while (lo < hi) { const int mid = (lo + hi) >> 1; if (tv[mid] >= v) lo = mid + 1; else hi = mid; }
            if (tn < DFLASH_TOPK_K) ++tn;
            for (int j = tn - 1; j > lo; --j) { tv[j] = tv[j - 1]; ti[j] = ti[j - 1]; }
            tv[lo] = v; ti[lo] = i;
        }
    }

    // smem 汇聚:256 × 16 候选
    __shared__ float sv[DFLASH_TOPK_BLOCK * DFLASH_TOPK_K];
    __shared__ int   si[DFLASH_TOPK_BLOCK * DFLASH_TOPK_K];
    #pragma unroll
    for (int k = 0; k < DFLASH_TOPK_K; ++k) {
        const int slot = tid * DFLASH_TOPK_K + k;
        if (k < tn) { sv[slot] = tv[k]; si[slot] = ti[k]; }
        else        { sv[slot] = -3.4e38f; si[slot] = vocab; }
    }
    __shared__ float rs_val[DFLASH_TOPK_BLOCK];
    __shared__ int   rs_idx[DFLASH_TOPK_BLOCK];
    __shared__ float s_wv;
    __shared__ int   s_wi;
    __syncthreads();

    // 16 轮:块内 argmax(平局取小索引)→ 选中置 -inf
    for (int k = 0; k < DFLASH_TOPK_K; ++k) {
        float best = -3.4e38f; int bi = vocab;
        for (int i = tid; i < DFLASH_TOPK_BLOCK * DFLASH_TOPK_K; i += DFLASH_TOPK_BLOCK) {
            const float v = sv[i];
            if (v > best || (v == best && si[i] < bi)) { best = v; bi = si[i]; }
        }
        rs_val[tid] = best; rs_idx[tid] = bi;
        __syncthreads();
        for (int off = DFLASH_TOPK_BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) {
                const float ov = rs_val[tid + off]; const int oi = rs_idx[tid + off];
                if (ov > rs_val[tid] || (ov == rs_val[tid] && oi < rs_idx[tid])) {
                    rs_val[tid] = ov; rs_idx[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            // 两段连续布局:值区 [rows*K) + 索引区 [rows*K, 2*rows*K)
            // (行主 [r*K+k] —— SliceView 平铺偏移切片方可行内对齐)
            out[row * DFLASH_TOPK_K + k] = rs_val[0];
            out[gridDim.x * DFLASH_TOPK_K + row * DFLASH_TOPK_K + k] = (float)rs_idx[0];
            s_wv = rs_val[0]; s_wi = rs_idx[0];
        }
        __syncthreads();
        // 剔除选中:全线程分片扫(曾 thread 0 串行 4096 扫 ×16 轮 = 
        // 单线程 ~130µs/行 病理;E5-DF4 并行化)
        {
            const float wv = s_wv; const int wi = s_wi;
            for (int i = tid; i < DFLASH_TOPK_BLOCK * DFLASH_TOPK_K; i += DFLASH_TOPK_BLOCK) {
                if (sv[i] == wv && si[i] == wi) { sv[i] = -3.4e38f; si[i] = vocab; }
            }
        }
        __syncthreads();
    }
}

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
#define DFLASH_TOPK_NWARP (DFLASH_TOPK_BLOCK / 32)

/// (值降, 索引升)严格全序比较:真 = (v,iv) 优于 (w,iw)
__device__ __forceinline__ bool owl_topk_gt(float v, int iv, float w, int iw) {
    return (v > w) || (v == w && iv < iw);
}

/// 寄存器 top-16 插入(全展开常量索引;tk_tour 的"winner 在 slot 0"前提靠它)
__device__ __forceinline__ void owl_topk_push(float v, int ii, float* tv, int* ti) {
    if (owl_topk_gt(v, ii, tv[DFLASH_TOPK_K - 1], ti[DFLASH_TOPK_K - 1])) {
        #pragma unroll
        for (int k = 0; k < DFLASH_TOPK_K; ++k) {
            if (owl_topk_gt(v, ii, tv[k], ti[k])) {
                const float t = tv[k];
                const int t2 = ti[k];
                tv[k] = v; ti[k] = ii; v = t; ii = t2;
            }
        }
    }
}

/// warp 内 16 轮锦标赛:第 k 名由 lane k 写入 out_v/out_i(**零 syncthreads**;
/// 全序律保证持方 winner 恒在 slot 0 ⇒ 弹出只需左移)
__device__ __forceinline__ void owl_topk_tour(float* tv, int* ti, int lane, int sid,
                                              float* out_v, int* out_i) {
    #pragma unroll 1
    for (int k = 0; k < DFLASH_TOPK_K; ++k) {
        float bv = tv[0];
        int bi = ti[0];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int   oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (owl_topk_gt(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == k) { out_v[k] = bv; out_i[k] = bi; }
        if (ti[0] == bi) {   // 本 lane 持 winner(索引唯一;哨兵可多 lane 同持)
            #pragma unroll
            for (int j = 0; j < DFLASH_TOPK_K - 1; ++j) { tv[j] = tv[j + 1]; ti[j] = ti[j + 1]; }
            tv[DFLASH_TOPK_K - 1] = -3.4e38f; ti[DFLASH_TOPK_K - 1] = sid;
        }
    }
}

extern "C" __global__ void owl_topk16_split_f16(
    const __half* __restrict__ x,   // [rows, vocab]
    int vocab,
    int nparts,
    int rows,
    float* __restrict__ tmp) {      // [2*rows*nparts*K]:值区 + 索引区(见头注)
    const int row  = blockIdx.x / nparts;
    const int part = blockIdx.x % nparts;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int chunk = (vocab + nparts - 1) / nparts;
    const int lo = part * chunk;
    const int hi = min(vocab, lo + chunk);
    const int n = hi > lo ? (hi - lo) : 0;   // nparts > vocab 时尾部空段(测试小词表)
    const __half* base = x + (unsigned long long)row * vocab + (lo < vocab ? lo : vocab);

    float tv[DFLASH_TOPK_K];
    int   ti[DFLASH_TOPK_K];
    #pragma unroll
    for (int k = 0; k < DFLASH_TOPK_K; ++k) { tv[k] = -3.4e38f; ti[k] = vocab; }

    // 向量段:uint4 = 8 half/线程/迭代(16B 对齐时;否则全程标量兜底)
    const int nvec = (((unsigned long long)base & 15ull) == 0ull) ? (n >> 3) : 0;
    const uint4* vb = (const uint4*)base;
    #pragma unroll 1
    for (int v = tid; v < nvec; v += DFLASH_TOPK_BLOCK) {
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
        for (int j = 0; j < 8; ++j) owl_topk_push(f[j], i0 + j, tv, ti);
    }
    // 标量尾(含未对齐兜底)
    for (int i = lo + nvec * 8 + tid; i < hi; i += DFLASH_TOPK_BLOCK)
        owl_topk_push(__half2float(base[i - lo]), i, tv, ti);

    // 块内归并:各 warp 锦标赛 → smem 候选 → warp0 再锦标赛 → 直写 tmp
    // (块内跨 warp 只 1 次同步;旧 16 轮 smem argmax 是 144 次)
    __shared__ float sw[DFLASH_TOPK_NWARP * DFLASH_TOPK_K];
    __shared__ int   swi[DFLASH_TOPK_NWARP * DFLASH_TOPK_K];
    const int warp = tid >> 5;
    owl_topk_tour(tv, ti, lane, vocab, sw + warp * DFLASH_TOPK_K, swi + warp * DFLASH_TOPK_K);
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int k = 0; k < DFLASH_TOPK_K; ++k) { tv[k] = -3.4e38f; ti[k] = vocab; }
        for (int i = lane; i < DFLASH_TOPK_NWARP * DFLASH_TOPK_K; i += 32)
            owl_topk_push(sw[i], swi[i], tv, ti);
        const long long bv = (long long)(row * nparts + part) * DFLASH_TOPK_K;
        const long long bi = (long long)rows * nparts * DFLASH_TOPK_K + bv;
        owl_topk_tour(tv, ti, lane, vocab, tmp + bv, (int*)(tmp + bi));
    }
}

// ---------------------------------------------------------------------------
// 2c) split-K 段二:**一行一 warp**(block=32),把 nparts×16 候选归并成 top-16
//      —— 零 __syncthreads(warp 内 shuffle 锦标赛)。grid = rows。
// 契约与 2) 同:输出 [2*rows*K](值区行主 + 索引区,索引 **f32**,供 SliceView
// 行内对齐切分);平局取小索引。
// ---------------------------------------------------------------------------
extern "C" __global__ void owl_topk16_merge_f16(
    const float* __restrict__ tmp,  // 段一输出 [2*rows*nparts*K]
    int nparts,
    int rows,
    int vocab,
    float* __restrict__ out) {      // [2*rows*K]
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int n = nparts * DFLASH_TOPK_K;
    const float* pvs = tmp + (long long)row * n;
    const int*   pis = (const int*)(tmp + (long long)rows * n + (long long)row * n);

    float tv[DFLASH_TOPK_K];
    int   ti[DFLASH_TOPK_K];
    #pragma unroll
    for (int k = 0; k < DFLASH_TOPK_K; ++k) { tv[k] = -3.4e38f; ti[k] = vocab; }
    for (int i = lane; i < n; i += 32) owl_topk_push(pvs[i], pis[i], tv, ti);

    // 值区直写(out 值半区本就是 f32);索引经 smem 转 f32(契约 5)
    __shared__ int sidx[DFLASH_TOPK_K];
    owl_topk_tour(tv, ti, lane, vocab, out + row * DFLASH_TOPK_K, sidx);
    __syncwarp();
    if (lane < DFLASH_TOPK_K)
        out[rows * DFLASH_TOPK_K + row * DFLASH_TOPK_K + lane] = (float)sidx[lane];
}
