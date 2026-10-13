// ============================================================================
// DFlash2 草稿核族:selector 格打分 + 贪心 walk(_score_edges + sample_path 融合)(2026-10-13 C 轮自 cu/owl/dflash2.cu 按**算子族**
// 拆出 —— 一算子一 `.cu`,与 ops/dflash.rs 的声明对望)。
//
// 上游语义锚 = packages/sglang python/sglang/srt/models/dflash.py +
// kernels/ops/speculative/dflash.py(sglang-dflash-patch 分支,逐式对拍)。
//
// 数值律:输入 f16/bf16,累计 f32,输出随槽(契约 5:索引量 f32 过线;词表
// id < 2²⁴ 精确)。平局取最小索引(与 owl_argmax / torch.topk sorted 语义
// 一致,host 对拍同律)。输出块固定末参(槽序契约 4)。
//
//   score[e,p,c] = unary[e,c] + Σ_r A[pred(e,p)][r] · proj[e][r] · B[cand[e,c]][r]
//   pred(e=0,p)  = anchor(所有 p 同值);pred(e>0,p) = cand[e-1][p]
//   walk: idx0 = argmax_c score[0,0,c];idx_e = argmax_c score[e,idx_{e-1},c]
//
// bf16 变体:proj/a_tab/b_tab bf16(码本/投影原生 BF16);cand/unary/
// anchor/out 仍 f32(契约 5 不变)。
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifdef __CUDACC_RTC__
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#endif

// ---------------------------------------------------------------------------
// 3) selector:格打分 + 贪心 walk(sglang _score_edges + sample_path
//    greedy 臂融合;单块)
//
//   score[e,p,c] = unary[e,c] + Σ_r A[pred(e,p)][r] · proj[e][r] · B[cand[e,c]][r]
//   pred(e=0,p)  = anchor(所有 p 同值);pred(e>0,p) = cand[e-1][p]
//   walk: idx0 = argmax_c score[0,0,c];idx_e = argmax_c score[e,idx_{e-1},c]
//         tok[e] = cand[e][idx_e]
//
// proj = hidden_projection(h)[S, R] f16(声明侧 Linear 产出);cand/unary
// = topk 输出 f32。A/B = 前驱/后继码本 [vocab, R] f16。输出 scores(对拍
// 观测面)+ toks [S] f32。平局取小索引(host 对拍同律)。
// ---------------------------------------------------------------------------
extern "C" __global__ void owl_dflash_select_f16(
    const float* __restrict__ cand,   // [S, K]
    const float* __restrict__ unary,  // [S, K](topk 值半区)
    const __half* __restrict__ proj,  // [S, R]
    const float* __restrict__ anchor, // [1]
    const __half* __restrict__ a_tab, // [vocab, R](predecessor_codebook)
    const __half* __restrict__ b_tab, // [vocab, R](successor_codebook)
    int slots,                        // S = K_draft(7);打分含 slot 0
    int k_top,                        // K = 16
    int rank,                         // R = 256
    float* __restrict__ out) {        // [S·(K·K+1)]:[0..S) = toks(f32),[S..S+S·K·K) = scores(对拍观测面)
    const int S = slots, K = k_top, R = rank;
    const int items = S * K * K;
    const float anc = anchor[0];
    float* __restrict__ toks = out;             // [S]
    float* __restrict__ scores = out + S;       // [S, K, K]

    for (int it = threadIdx.x; it < items; it += blockDim.x) {
        const int e = it / (K * K);
        const int pc = it % (K * K);
        const int p = pc / K;
        const int c = pc % K;
        const float pred_id = (e == 0) ? anc : cand[(e - 1) * K + p];
        const float cand_id = cand[e * K + c];
        const __half* arow = a_tab + (long long)pred_id * R;
        const __half* brow = b_tab + (long long)cand_id * R;
        const __half* prow = proj + (long long)e * R;
        float dot = 0.0f;
        for (int r = 0; r < R; ++r) {
            dot += __half2float(arow[r]) * __half2float(prow[r]) * __half2float(brow[r]);
        }
        scores[it] = unary[e * K + c] + dot;
    }
    __syncthreads();

    // 贪心 walk(单线程;S×K = 112 步,µs 级)
    if (threadIdx.x == 0) {
        int idx = 0;
        for (int e = 0; e < S; ++e) {
            const float* row = scores + (long long)e * K * K + (long long)idx * K;
            float best = row[0]; int bi = 0;
            for (int c = 1; c < K; ++c) {
                if (row[c] > best) { best = row[c]; bi = c; }
            }
            idx = bi;
            toks[e] = cand[e * K + idx];
        }
    }
}

// ---- BF16 变体(proj/a_tab/b_tab bf16,码本/投影原生 BF16;
//      cand/unary/anchor/out 仍 f32,契约 5 不变)----
extern "C" __global__ void owl_dflash_select_bf16(
    const float* __restrict__ cand,   // [S, K]
    const float* __restrict__ unary,  // [S, K](topk 值半区)
    const __nv_bfloat16* __restrict__ proj,  // [S, R]
    const float* __restrict__ anchor, // [1]
    const __nv_bfloat16* __restrict__ a_tab, // [vocab, R](predecessor_codebook)
    const __nv_bfloat16* __restrict__ b_tab, // [vocab, R](successor_codebook)
    int slots,                        // S = K_draft(7);打分含 slot 0
    int k_top,                        // K = 16
    int rank,                         // R = 256
    float* __restrict__ out) {        // [S·(K·K+1)]:[0..S) = toks,[S..) = scores
    const int S = slots, K = k_top, R = rank;
    const int items = S * K * K;
    const float anc = anchor[0];
    float* __restrict__ toks = out;             // [S]
    float* __restrict__ scores = out + S;       // [S, K, K]

    for (int it = threadIdx.x; it < items; it += blockDim.x) {
        const int e = it / (K * K);
        const int pc = it % (K * K);
        const int p = pc / K;
        const int c = pc % K;
        const float pred_id = (e == 0) ? anc : cand[(e - 1) * K + p];
        const float cand_id = cand[e * K + c];
        const __nv_bfloat16* arow = a_tab + (long long)pred_id * R;
        const __nv_bfloat16* brow = b_tab + (long long)cand_id * R;
        const __nv_bfloat16* prow = proj + (long long)e * R;
        float dot = 0.0f;
        for (int r = 0; r < R; ++r) {
            dot += __bfloat162float(arow[r]) * __bfloat162float(prow[r]) * __bfloat162float(brow[r]);
        }
        scores[it] = unary[e * K + c] + dot;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int idx = 0;
        for (int e = 0; e < S; ++e) {
            const float* row = scores + (long long)e * K * K + (long long)idx * K;
            float best = row[0]; int bi = 0;
            for (int c = 1; c < K; ++c) {
                if (row[c] > best) { best = row[c]; bi = c; }
            }
            idx = bi;
            toks[e] = cand[e * K + idx];
        }
    }
}
