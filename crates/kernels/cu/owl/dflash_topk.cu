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
