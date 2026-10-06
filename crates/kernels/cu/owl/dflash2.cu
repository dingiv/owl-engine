// ============================================================================
// dflash2.cu —— DFlash2 草稿特有核族(E5-DF1;2026-10-07)
//
// 上游语义锚 = packages/sglang python/sglang/srt/models/dflash.py +
// kernels/ops/speculative/dflash.py(sglang-dflash-patch 分支,逐式对拍)。
//
//  1) owl_dflash_conv_f16     分组动态深度卷积(DFlashGroupedConv._convolve)
//  2) owl_topk16_f16          逐行 top-K=16(candidate top-k;flashinfer topk 同位)
//  3) owl_dflash_select_f16   selector 格打分 + 贪心 walk(_score_edges +
//                             sample_path greedy 臂,单块融合)
//  4) owl_naive_attn_nc_f16   非因果块 attention(propose 8 行双向;测试/
//                             eager 回退臂;生产臂 = FI kNonCausal 变体)
//
// 数值律:输入 f16,累计 f32,输出随槽(契约 5:索引量 f32 过线;词表 id
// < 2²⁴ 精确)。平局取最小索引(与 owl_argmax / torch.topksorted 语义一致,
// host 对拍同律)。
//
// BF16 变体(E5-DF3 同日十四;sglang 对齐 —— 草稿路径 BF16 全程,真实
// 权重激活超 f16 范围,见施工方案 §6.14-6.16):
//   owl_dflash_conv_bf16      全 bf16(x/delta/base/out;权重原生 BF16)
//   owl_dflash_select_bf16    proj/a_tab/b_tab bf16(码本原生 BF16);
//                             cand/unary/anchor/out 仍 f32(契约 5 不变)
// topk16 保持 f16 唯一(lm_head 铸边界前 logits 恒 f16,cast 保 cublas/topk)。
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
// 1) 分组动态深度 K-tap 卷积(sglang _grouped_conv 逐式)
//
//   blocks = x.unflatten(-1, (G, group))          # [T, G, group]
//   coeff[t,tap,g,s] = base[side,tap,g*group+s] + delta[t,side,tap,g]
//   out[t,g,s] = Σ_tap coeff[t,tap,g,s] · x[t-tap,g,s] · [t%block ≥ tap]
//
// delta = kernel_projection(x) 布局 [T, side(2), tap(2), G](side-major,
// sglang reshape 序);t<tap 时 x[t-tap] 记 0(且 t%block 掩码同清)。
// taps=2 / group=16 由检查点家族定形(conv_kernel_size=2, conv_group_size=16),
// block(T 行内块宽)/ G / H 参数化。
// ---------------------------------------------------------------------------
extern "C" __global__ void owl_dflash_conv_f16(
    const __half* __restrict__ x,      // [T, H]
    const __half* __restrict__ delta,  // [T, 2, 2, G]
    const __half* __restrict__ base,   // [2, 2, H]
    int side,                          // 0 = input 子层前,1 = output 子层后
    int block_size,                    // 位置掩码模数(T=8 时掩码 [0..7])
    int hidden,                        // H
    int groups,                        // G = H / group, group 烘焙 16
    int rows,                          // T(哨兵哨界 = T*H)
    __half* __restrict__ out) {        // [T, H]
    const int group = 16;
    const int total = blockIdx.x * blockDim.x + threadIdx.x;
    if (total >= rows * hidden) return;

    const int t = total / hidden;
    const int d = total % hidden;
    const int g = d / group;
    const int s = d % group;
    const int G = groups;

    const float d0 = __half2float(delta[((long long)t * 2 + side) * 2 * G + 0 * G + g]);
    const float d1 = __half2float(delta[((long long)t * 2 + side) * 2 * G + 1 * G + g]);
    const float b0 = __half2float(base[((long long)side * 2 + 0) * hidden + d]);
    const float b1 = __half2float(base[((long long)side * 2 + 1) * hidden + d]);
    const float c0 = d0 + b0;
    const float c1 = d1 + b1;

    const float x0 = __half2float(x[(long long)t * hidden + d]);
    float x1 = 0.0f;
    if (t >= 1) x1 = __half2float(x[(long long)(t - 1) * hidden + d]);
    const float m1 = ((t % block_size) >= 1) ? 1.0f : 0.0f;

    out[total] = __float2half(c0 * x0 + c1 * x1 * m1);
}

// ---- BF16 变体(全 bf16;数学逐式同 f16 版,float 中间)----
extern "C" __global__ void owl_dflash_conv_bf16(
    const __nv_bfloat16* __restrict__ x,      // [T, H]
    const __nv_bfloat16* __restrict__ delta,  // [T, 2, 2, G]
    const __nv_bfloat16* __restrict__ base,   // [2, 2, H]
    int side,
    int block_size,
    int hidden,
    int groups,
    int rows,
    __nv_bfloat16* __restrict__ out) {        // [T, H]
    const int group = 16;
    const int total = blockIdx.x * blockDim.x + threadIdx.x;
    if (total >= rows * hidden) return;

    const int t = total / hidden;
    const int d = total % hidden;
    const int g = d / group;
    const int G = groups;

    const float d0 = __bfloat162float(delta[((long long)t * 2 + side) * 2 * G + 0 * G + g]);
    const float d1 = __bfloat162float(delta[((long long)t * 2 + side) * 2 * G + 1 * G + g]);
    const float b0 = __bfloat162float(base[((long long)side * 2 + 0) * hidden + d]);
    const float b1 = __bfloat162float(base[((long long)side * 2 + 1) * hidden + d]);
    const float c0 = d0 + b0;
    const float c1 = d1 + b1;

    const float x0 = __bfloat162float(x[(long long)t * hidden + d]);
    float x1 = 0.0f;
    if (t >= 1) x1 = __bfloat162float(x[(long long)(t - 1) * hidden + d]);
    const float m1 = ((t % block_size) >= 1) ? 1.0f : 0.0f;

    out[total] = __float2bfloat16(c0 * x0 + c1 * x1 * m1);
}
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
            // 剔除选中(扫描置 -inf;每轮一次 4096 扫由下轮线程自扫完成)
            const float wv = rs_val[0]; const int wi = rs_idx[0];
            for (int i = 0; i < DFLASH_TOPK_BLOCK * DFLASH_TOPK_K; ++i) {
                if (sv[i] == wv && si[i] == wi) { sv[i] = -3.4e38f; si[i] = vocab; }
            }
        }
        __syncthreads();
    }
}

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

// ---------------------------------------------------------------------------
// 4) 非因果块 attention(naive;测试 + eager 回退臂)
//
//   q [T, Hq, D];keys/values = 前缀池行 [0, prefix_len)(classic 布局,
//   寻址与 reshape_and_cache 同式)+ **自块核输入直读** k/v_self
//   [T, Hkv, D](行 s-prefix;写-读同核有跨 block 竞态,池回读免谈,
//   直读 = 零 launch 内依赖)。每查询行可见全部 kv_len = prefix+T 行
//   (ENCODER_ONLY,sglang dflash is_causal=false 语义)。
//
// 在线 softmax 单趟流式(零分数组,零窗上限);f16 读,f32 算,f16 写。
// grid = ceil(T*Hq/256),block 256。
// ---------------------------------------------------------------------------
extern "C" __global__ void owl_naive_attn_nc_f16(
    const __half* __restrict__ q,      // [T, Hq, D]
    const __half* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __half* __restrict__ v_self, // 同上
    const __half* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x]
    const __half* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page]
    int q_tokens,                      // T(哨兵哨界 = T*Hq)
    int prefix_len,                    // 池内前缀行数
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= q_tokens * q_heads) return;
    const int t = idx / q_heads;
    const int h = idx % q_heads;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = prefix_len + q_tokens;

    const __half* qs = q + (long long)t * q_heads * d_dim + (long long)h * d_dim;
    const float scale = rsqrtf((float)d_dim);

    // 在线 softmax 单趟(m/l/acc 流式;s < prefix 走池,s >= prefix 直读自块)
    float m = -3.0e38f, l = 0.0f;
    float acc[512]; // hd <= 512(draft 128 / target 256)
    for (int d = 0; d < d_dim; ++d) acc[d] = 0.0f;
    for (int s = 0; s < kv_len; ++s) {
        float dot = 0.0f;
        float vr[512];
        if (s < prefix_len) {
            const int b = s / page;
            const int off = s % page;
            const long long kbase = (long long)b * kv_heads * d_dim * page
                                  + (long long)kvh * hd_x * page * x + (long long)off * x;
            const long long vbase = (long long)b * kv_heads * d_dim * page
                                  + (long long)kvh * d_dim * page + (long long)off;
            for (int d = 0; d < d_dim; ++d) {
                const float kv = __half2float(kc[kbase + (long long)(d / x) * page * x + (d % x)]);
                dot += __half2float(qs[d]) * kv;
                vr[d] = __half2float(vc[vbase + (long long)d * page]);
            }
        } else {
            const int st = s - prefix_len;
            const __half* kr = k_self + (long long)st * kv_heads * d_dim + (long long)kvh * d_dim;
            const __half* vrow = v_self + (long long)st * kv_heads * d_dim + (long long)kvh * d_dim;
            for (int d = 0; d < d_dim; ++d) {
                dot += __half2float(qs[d]) * __half2float(kr[d]);
                vr[d] = __half2float(vrow[d]);
            }
        }
        dot *= scale;
        float w;
        if (dot <= m) { w = __expf(dot - m); l += w; }
        else { const float wold = __expf(m - dot); for (int d = 0; d < d_dim; ++d) acc[d] *= wold; l = l * wold + 1.0f; m = dot; w = 1.0f; }
        for (int d = 0; d < d_dim; ++d) acc[d] += w * vr[d];
    }
    __half* od = out + (long long)t * q_heads * d_dim + (long long)h * d_dim;
    const float inv = 1.0f / l;
    for (int d = 0; d < d_dim; ++d) od[d] = __float2half(acc[d] * inv);
}

// ---- BF16 变体(q/k/v/池/出全 bf16;在线 softmax f32 中间逐式)----
extern "C" __global__ void owl_naive_attn_nc_bf16(
    const __nv_bfloat16* __restrict__ q,      // [T, Hq, D]
    const __nv_bfloat16* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __nv_bfloat16* __restrict__ v_self, // 同上
    const __nv_bfloat16* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x]
    const __nv_bfloat16* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page]
    int q_tokens,
    int prefix_len,
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __nv_bfloat16* __restrict__ out) {        // [T, Hq, D]
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= q_tokens * q_heads) return;
    const int t = idx / q_heads;
    const int h = idx % q_heads;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = prefix_len + q_tokens;

    const __nv_bfloat16* qs = q + (long long)t * q_heads * d_dim + (long long)h * d_dim;
    const float scale = rsqrtf((float)d_dim);

    float m = -3.0e38f, l = 0.0f;
    float acc[512]; // hd <= 512(draft 128 / target 256)
    for (int d = 0; d < d_dim; ++d) acc[d] = 0.0f;
    for (int s = 0; s < kv_len; ++s) {
        float dot = 0.0f;
        float vr[512];
        if (s < prefix_len) {
            const int b = s / page;
            const int off = s % page;
            const long long kbase = (long long)b * kv_heads * d_dim * page
                                  + (long long)kvh * hd_x * page * x + (long long)off * x;
            const long long vbase = (long long)b * kv_heads * d_dim * page
                                  + (long long)kvh * d_dim * page + (long long)off;
            for (int d = 0; d < d_dim; ++d) {
                const float kv = __bfloat162float(kc[kbase + (long long)(d / x) * page * x + (d % x)]);
                dot += __bfloat162float(qs[d]) * kv;
                vr[d] = __bfloat162float(vc[vbase + (long long)d * page]);
            }
        } else {
            const int st = s - prefix_len;
            const __nv_bfloat16* kr = k_self + (long long)st * kv_heads * d_dim + (long long)kvh * d_dim;
            const __nv_bfloat16* vrow = v_self + (long long)st * kv_heads * d_dim + (long long)kvh * d_dim;
            for (int d = 0; d < d_dim; ++d) {
                dot += __bfloat162float(qs[d]) * __bfloat162float(kr[d]);
                vr[d] = __bfloat162float(vrow[d]);
            }
        }
        dot *= scale;
        float w;
        if (dot <= m) { w = __expf(dot - m); l += w; }
        else { const float wold = __expf(m - dot); for (int d = 0; d < d_dim; ++d) acc[d] *= wold; l = l * wold + 1.0f; m = dot; w = 1.0f; }
        for (int d = 0; d < d_dim; ++d) acc[d] += w * vr[d];
    }
    __nv_bfloat16* od = out + (long long)t * q_heads * d_dim + (long long)h * d_dim;
    const float inv = 1.0f / l;
    for (int d = 0; d < d_dim; ++d) od[d] = __float2bfloat16(acc[d] * inv);
}
