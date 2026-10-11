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
// ⭐ 草稿自注意力滑窗(2026-10-13,D2 修正):检查点 config.json
//   `sliding_window = 2048` + `use_sliding_window = true`(5 层全滑窗),
//   参考研究原文:"the drafter reads 5 of the target's 64 layers through a
//   **2 048-token sliding window** so its work grows roughly linearly"。
//   → 非因果块注意力的**前缀**只取最近 2048 行(自块 8 行恒全可见)。
//   修正前:前缀全扫(0..prefix_len)→ ①长 ctx 白烧 (prefix/2048)× 时间,
//   ②attention 图案超出训练窗。仅当前缀 > 2048 时生效(短 ctx 逐位不变)。
#define OWL_NC_SLIDING_WINDOW 2048

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
#include <cuda_fp8.h>
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
// v2(E5-DF4 性能:block-per-head flash 式在线 softmax):
//   grid (T, Hq) × block (hd) —— 一 block 一 (t,h) 查询行,thread d 独占
//   一维(acc 单寄存器)。v1 一 thread 一行 + acc[512] 溢出 local memory +
//   全程单 block(82 SM 占用 1.2%),实测 804µs/次。每 kv 行一次块内
//   归约(q·k);m/l/p 行标量住 shared,thread-0 独写。
// ---- B6 偷显存:fp8kv 变体(前缀池 e4m3 读入即转;自块直读保持原 dtype)----
// B6.3 收口:恢复 f16 前缀版(fb1e453 曾直接改名删体 → (F16,false)
// 悬空 = spec+f16 草稿池 propose 首轮 CUDA_ERROR_NOT_FOUND;本函数 =
// pre-fb1e453 原体逐字回植,前缀池 f16 读)
extern "C" __global__ void owl_naive_attn_nc_f16(
    const __half* __restrict__ q,      // [T, Hq, D]
    const __half* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __half* __restrict__ v_self, // 同上
    const __half* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x]
    const __half* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page]
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __half* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __half2float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            kv = __half2float(kc[((long long)b * kv_heads + kvh) * hd_x * page * x
                                 + (long long)(d / x) * page * x + off * x + d % x]);
        } else {
            const int st = srow - prefix_len;
            kv = __half2float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            p *= __half2float(vc[((long long)b * kv_heads + kvh) * d_dim * page
                                 + (long long)d * page + off]);
        } else {
            const int st = srow - prefix_len;
            p *= __half2float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2half(acc / s_l);
}

extern "C" __global__ void owl_naive_attn_nc_fp8kv_f16(
    const __half* __restrict__ q,      // [T, Hq, D]
    const __half* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __half* __restrict__ v_self, // 同上
    const unsigned char* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x](e4m3)
    const unsigned char* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page](e4m3)
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __half* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __half2float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            // B6 偷显存:前缀池 e4m3 读入即转(元素序 = 字节序)
            __nv_fp8_e4m3 ke; ke.__x = kc[((long long)b * kv_heads + kvh) * hd_x * page * x
                                 + (long long)(d / x) * page * x + off * x + d % x];
            kv = __half2float(__half(ke));
        } else {
            const int st = srow - prefix_len;
            kv = __half2float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            __nv_fp8_e4m3 ve; ve.__x = vc[((long long)b * kv_heads + kvh) * d_dim * page
                                 + (long long)d * page + off];
            p *= __half2float(__half(ve));
        } else {
            const int st = srow - prefix_len;
            p *= __half2float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2half(acc / s_l);
}

// ---- BF16 v2(同款 block-per-head flash 式;池/出 bf16)----
extern "C" __global__ void owl_naive_attn_nc_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_self,
    const __nv_bfloat16* __restrict__ v_self,
    const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc,
    const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __nv_bfloat16* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __bfloat162float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            kv = __bfloat162float(kc[((long long)b * kv_heads + kvh) * hd_x * page * x
                                     + (long long)(d / x) * page * x + off * x + d % x]);
        } else {
            const int st = srow - prefix_len;
            kv = __bfloat162float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            p *= __bfloat162float(vc[((long long)b * kv_heads + kvh) * d_dim * page
                                     + (long long)d * page + off]);
        } else {
            const int st = srow - prefix_len;
            p *= __bfloat162float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2bfloat16(acc / s_l);
}

extern "C" __global__ void owl_naive_attn_nc_fp8kv_bf16(
    const __nv_bfloat16* __restrict__ q,      // [T, Hq, D]
    const __nv_bfloat16* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __nv_bfloat16* __restrict__ v_self, // 同上
    const unsigned char* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x](e4m3)
    const unsigned char* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page](e4m3)
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __nv_bfloat16* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __nv_bfloat16* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __bfloat162float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            // B6 偷显存:前缀池 e4m3 读入即转(元素序 = 字节序)
            __nv_fp8_e4m3 ke; ke.__x = kc[((long long)b * kv_heads + kvh) * hd_x * page * x
                                 + (long long)(d / x) * page * x + off * x + d % x];
            kv = __bfloat162float(__float2bfloat16_rn(float(ke)));
        } else {
            const int st = srow - prefix_len;
            kv = __bfloat162float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            __nv_fp8_e4m3 ve; ve.__x = vc[((long long)b * kv_heads + kvh) * d_dim * page
                                 + (long long)d * page + off];
            p *= __bfloat162float(__float2bfloat16_rn(float(ve)));
        } else {
            const int st = srow - prefix_len;
            p *= __bfloat162float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2bfloat16_rn(acc / s_l);
}

// ---- kNHD 变体(kv布局统一契约 P2):前缀池 [nb,page,hkv,hd],K/V 同式;自块直读不变 ----
extern "C" __global__ void owl_naive_attn_nc_f16_knhd(
    const __half* __restrict__ q,      // [T, Hq, D]
    const __half* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __half* __restrict__ v_self, // 同上
    const __half* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x]
    const __half* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page]
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __half* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __half2float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            kv = __half2float(kc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d]);
        } else {
            const int st = srow - prefix_len;
            kv = __half2float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            p *= __half2float(vc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d]);
        } else {
            const int st = srow - prefix_len;
            p *= __half2float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2half(acc / s_l);
}


extern "C" __global__ void owl_naive_attn_nc_fp8kv_f16_knhd(
    const __half* __restrict__ q,      // [T, Hq, D]
    const __half* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __half* __restrict__ v_self, // 同上
    const unsigned char* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x](e4m3)
    const unsigned char* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page](e4m3)
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __half* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __half* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __half2float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            // B6 偷显存:前缀池 e4m3 读入即转(元素序 = 字节序)
            __nv_fp8_e4m3 ke; ke.__x = kc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d];
            kv = __half2float(__half(ke));
        } else {
            const int st = srow - prefix_len;
            kv = __half2float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            __nv_fp8_e4m3 ve; ve.__x = vc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d];
            p *= __half2float(__half(ve));
        } else {
            const int st = srow - prefix_len;
            p *= __half2float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2half(acc / s_l);
}

// ---- BF16 v2(同款 block-per-head flash 式;池/出 bf16)----

extern "C" __global__ void owl_naive_attn_nc_bf16_knhd(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_self,
    const __nv_bfloat16* __restrict__ v_self,
    const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc,
    const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __nv_bfloat16* __restrict__ out) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __nv_bfloat16* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __bfloat162float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            kv = __bfloat162float(kc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d]);
        } else {
            const int st = srow - prefix_len;
            kv = __bfloat162float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            p *= __bfloat162float(vc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d]);
        } else {
            const int st = srow - prefix_len;
            p *= __bfloat162float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2bfloat16(acc / s_l);
}


extern "C" __global__ void owl_naive_attn_nc_fp8kv_bf16_knhd(
    const __nv_bfloat16* __restrict__ q,      // [T, Hq, D]
    const __nv_bfloat16* __restrict__ k_self, // [T, Hkv, D](自块直读)
    const __nv_bfloat16* __restrict__ v_self, // 同上
    const unsigned char* __restrict__ kc,     // classic 前缀 [nb, Hkv, hd/x, page, x](e4m3)
    const unsigned char* __restrict__ vc,     // classic 前缀 [nb, Hkv, hd, page](e4m3)
    const float* __restrict__ kv_len_p, // [1] 全窗 = 前缀 + T(契约 5;
                                       // E5-DF4 图化:运行时读,曾宿主烘焙)
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    __nv_bfloat16* __restrict__ out) {        // [T, Hq, D]
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int hd = blockDim.x;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;

    const __nv_bfloat16* qs = q + ((long long)t * q_heads + h) * d_dim;
    const float scale = rsqrtf((float)d_dim);

    __shared__ float red[512];  // 块内归约(hd ≤ 512:draft 128 / target 256)
    __shared__ float s_m, s_l, s_resc, s_p;

    const float qd = __bfloat162float(qs[d]);
    float acc = 0.0f;
    if (d == 0) { s_m = -3.0e38f; s_l = 0.0f; }
    __syncthreads();

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    for (int srow = srow_begin; srow < kv_len; ++srow) {
        float kv;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            // B6 偷显存:前缀池 e4m3 读入即转(元素序 = 字节序)
            __nv_fp8_e4m3 ke; ke.__x = kc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d];
            kv = __bfloat162float(__float2bfloat16_rn(float(ke)));
        } else {
            const int st = srow - prefix_len;
            kv = __bfloat162float(k_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        red[d] = qd * kv;
        __syncthreads();
        for (int s2 = hd / 2; s2 > 0; s2 >>= 1) {
            if (d < s2) red[d] += red[d + s2];
            __syncthreads();
        }
        if (d == 0) {
            const float dot = red[0] * scale;
            const float m_new = fmaxf(s_m, dot);
            const float resc = __expf(s_m - m_new);
            const float p = __expf(dot - m_new);
            s_l = s_l * resc + p;
            s_resc = resc;
            s_p = p;
            s_m = m_new;
        }
        __syncthreads();
        float p = s_p;
        if (srow < prefix_len) {
            const int b = srow / page;
            const int off = srow % page;
            __nv_fp8_e4m3 ve; ve.__x = vc[((long long)(b * page + off) * kv_heads + kvh) * d_dim + d];
            p *= __bfloat162float(__float2bfloat16_rn(float(ve)));
        } else {
            const int st = srow - prefix_len;
            p *= __bfloat162float(v_self[((long long)st * kv_heads + kvh) * d_dim + d]);
        }
        acc = acc * s_resc + p;
        __syncthreads();
    }
    out[((long long)t * q_heads + h) * d_dim + d] = __float2bfloat16_rn(acc / s_l);
}

// ============================================================================
// NC 注意力 **v2 核**(2026-10-13 ③;只换并行结构,**语义与上面 `owl_naive_attn_nc_*`
// 逐字同一套**:非因果自块 + 前缀滑窗 2048 + 经典池布局 + kv_len_p 设备读)。
//
// 为什么:旧核每 KV 行一次 128 宽 block 归约(7 步树 + 共 11 次 __syncthreads)。
// 拆分实测(prefix 2048/窗口 2048,测试文件里的诊断核 diag_*):
//   纯同步链+softmax = **1.295ms** / +K 读 0.48 / +V 读 0.36 / 合计 2.237ms
//   ⇒ **同步链占 58%**,是唯一值得砍的大项(设计稿 D2 判断正确;V 的 sector
//   放大只值 0.36ms —— L2 把它兜住了,原估的"16× 放大"不成立)。
//
// 结构(lane→连续 dim + warp butterfly + split-K + 收尾一次归并):
//   ① 每 lane 持 DPL = d_dim/32 个**连续** dim(sr 与 q 同布局);
//   ② 逐行:lane 内点积 → **warp butterfly(__shfl_xor_sync × 5)** ⇒ 每个 lane
//      都拿到完整点积 ⇒ online softmax 的 m/l/resc 在 lane 间**逐位一致**
//      (无需共享、无需同步);
//   ③ 行的遍历区间按 warp 切成 split-K(floating 均分),各 warp 独立
//      online softmax;
//   ④ **每 block 只归并一次**(flash-decoding 式:smem 存各 warp 的 m/l/acc,
//      1 次 __syncthreads,再按本 lane 的 dim 合并 + 归一)。
//   ⇒ 内层循环零同步;全局同步从"每行 11 次"降到"每 block 1 次"。
//
// ⚠️ 数值序变化(点积改 butterfly、softmax 改 split-K 归并)⇒ 必须过
//   `dflash2_nc_probe` 的 host 参考门(1e-2)与引擎恒等门/AL 红线。
//
// ⚠️ 约束:d_dim 必须 32 整除且 DPL ∈ {1..8};x 取 dtype 的向量宽(bf16/f16 → 8)。
//   池 K 的 4/8 连续 dim 落在同一个 x 组内(对齐前提见工单 §P1)。
// ============================================================================

#define OWL_NC_DPL_MAX 8

/// 4 个连续元素 → float[4](8B 合曲读;bf16 = 2×bfloat162 / f16 = 2×half2)
template <typename T>
struct NcVec4;
template <>
struct NcVec4<__nv_bfloat16> {
    static __device__ __forceinline__ void load(const __nv_bfloat16* p, float* o) {
        uint2 v = *reinterpret_cast<const uint2*>(p);
        const __nv_bfloat162 a = *reinterpret_cast<const __nv_bfloat162*>(&v.x);
        const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&v.y);
        o[0] = __bfloat162float(a.x);
        o[1] = __bfloat162float(a.y);
        o[2] = __bfloat162float(b.x);
        o[3] = __bfloat162float(b.y);
    }
};
template <>
struct NcVec4<__half> {
    static __device__ __forceinline__ void load(const __half* p, float* o) {
        uint2 v = *reinterpret_cast<const uint2*>(p);
        const __half2 a = *reinterpret_cast<const __half2*>(&v.x);
        const __half2 b = *reinterpret_cast<const __half2*>(&v.y);
        o[0] = __low2float(a);
        o[1] = __high2float(a);
        o[2] = __low2float(b);
        o[3] = __high2float(b);
    }
};

template <typename T>
struct NcConv;
template <>
struct NcConv<__nv_bfloat16> {
    static __device__ __forceinline__ float tof(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __device__ __forceinline__ __nv_bfloat16 fromf(float v) { return __float2bfloat16(v); }
};
template <>
struct NcConv<__half> {
    static __device__ __forceinline__ float tof(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half fromf(float v) { return __float2half(v); }
};

template <typename T, int DPL>
__device__ __forceinline__ void nc_attn_loop(
    const T* __restrict__ q,
    const T* __restrict__ k_self,
    const T* __restrict__ v_self,
    const T* __restrict__ kc,
    const T* __restrict__ vc,
    const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim,
    int page, int x,
    T* __restrict__ out) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nwarp = blockDim.x >> 5;
    // DPL>0 = 编译期定死(可矢量读);DPL==0 = 运行时 dpl(标量兜底)
    const int dpl = (DPL > 0) ? DPL : (d_dim >> 5);
    const int d0 = lane * dpl;
    const int kvh = h / (q_heads / kv_heads);
    const int hd_x = d_dim / x;
    const int kv_len = (int)kv_len_p[0];
    const int prefix_len = kv_len - gridDim.x;
    const float scale = rsqrtf((float)d_dim);

    const T* qs = q + ((long long)t * q_heads + h) * d_dim;
    float qd[OWL_NC_DPL_MAX];
#pragma unroll
    for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
        qd[j] = (j < dpl) ? NcConv<T>::tof(qs[d0 + j]) : 0.0f;
    }

    // D2 滑窗:前缀只取最近 OWL_NC_SLIDING_WINDOW 行(自块恒全可见)
    const int srow_begin =
        (prefix_len > OWL_NC_SLIDING_WINDOW) ? (prefix_len - OWL_NC_SLIDING_WINDOW) : 0;
    // 本 warp 的行切片(连续均分;余数归前几片)
    const int nrow = kv_len - srow_begin;
    const int per = (nrow + nwarp - 1) / nwarp;
    const int lo = srow_begin + warp * per;
    const int hi = (lo + per < kv_len) ? (lo + per) : kv_len;

    float m = -3.0e38f;
    float l = 0.0f;
    float acc[OWL_NC_DPL_MAX];
#pragma unroll
    for (int j = 0; j < OWL_NC_DPL_MAX; ++j) acc[j] = 0.0f;

    for (int srow = lo; srow < hi; ++srow) {
        const bool pre = (srow < prefix_len);
        const int b = pre ? (srow / page) : 0;
        const int off = pre ? (srow % page) : 0;
        float dot = 0.0f;
        if (DPL == 4 && (x % 4) == 0) {
            // 合曲路径:4 个连续 dim 落在同一 x 组(池)/ 本身就连续(自块)
            float kvf[4];
            if (pre) {
                const long long base = ((long long)b * kv_heads + kvh) * hd_x * page * x
                                       + (long long)off * x
                                       + (long long)(d0 / x) * page * x + (d0 % x);
                NcVec4<T>::load(kc + base, kvf);
            } else {
                const int st = srow - prefix_len;
                NcVec4<T>::load(k_self + ((long long)st * kv_heads + kvh) * d_dim + d0, kvf);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) dot += qd[j] * kvf[j];
        } else if (pre) {
            const long long base = ((long long)b * kv_heads + kvh) * hd_x * page * x
                                   + (long long)off * x;
#pragma unroll
            for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
                if (j < dpl) {
                    const int dim = d0 + j;
                    dot += qd[j]
                           * NcConv<T>::tof(kc[base + (long long)(dim / x) * page * x
                                               + (dim % x)]);
                }
            }
        } else {
            const int st = srow - prefix_len;
            const T* ks = k_self + ((long long)st * kv_heads + kvh) * d_dim;
#pragma unroll
            for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
                if (j < dpl) dot += qd[j] * NcConv<T>::tof(ks[d0 + j]);
            }
        }
        // warp butterfly:5 步后**每个 lane 都持完整点积**(故 softmax 态逐位一致)
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            dot += __shfl_xor_sync(0xffffffffu, dot, o);
        }
        const float dot_s = dot * scale;
        const float m_new = fmaxf(m, dot_s);
        const float resc = __expf(m - m_new);
        const float p = __expf(dot_s - m_new);
        l = l * resc + p;
        m = m_new;
        if (pre) {
            const long long base = ((long long)b * kv_heads + kvh) * (long long)d_dim * page;
#pragma unroll
            for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
                if (j < dpl) {
                    acc[j] = acc[j] * resc
                             + p * NcConv<T>::tof(vc[base + (long long)(d0 + j) * page + off]);
                }
            }
        } else {
            const int st = srow - prefix_len;
            const T* vs = v_self + ((long long)st * kv_heads + kvh) * d_dim;
#pragma unroll
            for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
                if (j < dpl) acc[j] = acc[j] * resc + p * NcConv<T>::tof(vs[d0 + j]);
            }
        }
    }

    // ---- 收尾:每 block 一次跨 warp 归并(flash-decoding 式;1 次 __syncthreads)----
    __shared__ float s_acc[8][256];
    __shared__ float s_m[8];
    __shared__ float s_l[8];
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
#pragma unroll
    for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
        if (j < dpl) s_acc[warp][d0 + j] = acc[j];
    }
    __syncthreads();
    float mg = -3.0e38f;
    for (int w = 0; w < nwarp; ++w) mg = fmaxf(mg, s_m[w]);
    float lg = 0.0f;
    float ag[OWL_NC_DPL_MAX];
#pragma unroll
    for (int j = 0; j < OWL_NC_DPL_MAX; ++j) ag[j] = 0.0f;
    for (int w = 0; w < nwarp; ++w) {
        const float e = __expf(s_m[w] - mg);
        lg += s_l[w] * e;
#pragma unroll
        for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
            if (j < dpl) ag[j] += s_acc[w][d0 + j] * e;
        }
    }
    T* o = out + ((long long)t * q_heads + h) * d_dim;
#pragma unroll
    for (int j = 0; j < OWL_NC_DPL_MAX; ++j) {
        if (j < dpl) o[d0 + j] = NcConv<T>::fromf(ag[j] / lg);
    }
}

extern "C" __global__ void owl_naive_attn_nc_bf16_v2(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k_self,
    const __nv_bfloat16* __restrict__ v_self, const __nv_bfloat16* __restrict__ kc,
    const __nv_bfloat16* __restrict__ vc, const float* __restrict__ kv_len_p,
    int q_heads, int kv_heads, int d_dim, int page, int x,
    __nv_bfloat16* __restrict__ out) {
    const int dpl = d_dim >> 5;
    if (dpl == 4) {
        nc_attn_loop<__nv_bfloat16, 4>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,
                                       d_dim, page, x, out);
    } else if (dpl == 8) {
        nc_attn_loop<__nv_bfloat16, 8>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,
                                       d_dim, page, x, out);
    } else {
        nc_attn_loop<__nv_bfloat16, 0>(q, k_self, v_self, kc, vc, kv_len_p, q_heads, kv_heads,
                                       d_dim, page, x, out);
    }
}

