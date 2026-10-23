// owl 视觉塔算子族(**批二**):Gelu / BiasAdd / PosEmbedBilinear / AttnFull
// (2026-10-15 vision-tower;设计文档 `docs/arch/vision-tower-设计与CPU对拍.md` §1.2/§3)。
//
// 与 `ops/vis.rs` 的声明对望(参数序逐字对齐;门 = `tests/op_tables.rs`)。
// Layernorm 住在 `cu/vis/layernorm.cu`(首件,勿并入本文件)。
//
// 规则(照 elems 族先例):读 T → to_float → float 计算 → from_float 写 T;
// 命名统一前缀 `owl_`;**输出块固定末参**(槽序契约 4);标量 size_t/i32/f32
// 与 Arg::U64/I32/F32 严格对位。
//
// 语义锚(HF `modeling_qwen3_5.py` 逐条,行号见设计文档 §1.2):
//   Gelu(tanh) = 0.5x(1+tanh(√(2/π)(x+0.044715x³)))  —— 视觉塔 MLP
//   Gelu(erf)  = 0.5x(1+erf(x/√2))                    —— **merger 专用**(精确版)
//   BiasAdd    = x[rows,cols] + bias[cols](行广播;塔每条 Linear 都有 bias)
//   PosEmbed   = Σ_k wts[n,k]·table[idx[n,k]]          —— 48² 网格双线性 4-tap 重采样
//   AttnFull   = 非因果 varlen 全注意力(逐图;hd=72;一遍在线 softmax)
//                (rope 不在本文件:**复用** `owl_rope_half_partial_*` 表驱动核,
//                 half = head_dim/2 时 partial 直通段空 ⇒ 与 HF 2D axial 逐式同构)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---- dtype 桥(本 TU 私有;与 layernorm.cu 的 vis_to_float 同源同前缀)----
template <typename T> __device__ __forceinline__ float vis_to_float(T x);
template <> __device__ __forceinline__ float vis_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float vis_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float vis_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T vis_from_float(float x);
template <> __device__ __forceinline__ float vis_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half vis_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 vis_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// ============================================================================
// ① Gelu(tanh 近似)
// ============================================================================
#define OWL_GELU_TANH_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) { \
        const float v = vis_to_float(x[i]); \
        const float c = 0.7978845608028654f; /* √(2/π) */ \
        out[i] = vis_from_float<T>(0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)))); \
    } \
}

// ⚠️ 实例行**必须单行**(签名互证门逐行解析)
OWL_GELU_TANH_KERNEL(owl_gelu_tanh_f32, float, (const float* x, float* out, const size_t n))
OWL_GELU_TANH_KERNEL(owl_gelu_tanh_f16, __half, (const __half* x, __half* out, const size_t n))
OWL_GELU_TANH_KERNEL(owl_gelu_tanh_bf16, __nv_bfloat16, (const __nv_bfloat16* x, __nv_bfloat16* out, const size_t n))

// ============================================================================
// ② Gelu(精确 erf)—— **merger 专用**;与 tanh 近似**不可互替**(差 ~1e-3)
// ============================================================================
#define OWL_GELU_ERF_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i < n) { \
        const float v = vis_to_float(x[i]); \
        out[i] = vis_from_float<T>(0.5f * v * (1.0f + erff(v * 0.7071067811865476f))); \
    } \
}

OWL_GELU_ERF_KERNEL(owl_gelu_erf_f32, float, (const float* x, float* out, const size_t n))
OWL_GELU_ERF_KERNEL(owl_gelu_erf_f16, __half, (const __half* x, __half* out, const size_t n))
OWL_GELU_ERF_KERNEL(owl_gelu_erf_bf16, __nv_bfloat16, (const __nv_bfloat16* x, __nv_bfloat16* out, const size_t n))

// ============================================================================
// ③ BiasAdd(行广播):grid = (rows,1,1);block 256;cols 为标量
// ============================================================================
#define OWL_BIAS_ADD_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    const size_t row = blockIdx.x; \
    const T* xr = x + row * cols; \
    T* orow = out + row * cols; \
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) { \
        orow[c] = vis_from_float<T>(vis_to_float(xr[c]) + vis_to_float(bias[c])); \
    } \
}

OWL_BIAS_ADD_KERNEL(owl_bias_add_f32, float, (const float* x, const float* bias, float* out, const size_t cols))
OWL_BIAS_ADD_KERNEL(owl_bias_add_f16, __half, (const __half* x, const __half* bias, __half* out, const size_t cols))
OWL_BIAS_ADD_KERNEL(owl_bias_add_bf16, __nv_bfloat16, (const __nv_bfloat16* x, const __nv_bfloat16* bias, __nv_bfloat16* out, const size_t cols))

// ============================================================================
// ④ PosEmbedBilinear(4-tap 重采样):out[n,d] = Σ_k wts[n,k]·table[idx[n,k],d]
//    grid = (n,1,1);block 256;idx/wts 恒 f32(索引量 f32 过线;契约 5)
// ============================================================================
#define OWL_POS_EMBED_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    const size_t row = blockIdx.x; \
    const float* wi = wts + row * n_taps; \
    const float* ii = idx + row * n_taps; \
    T* orow = out + row * dim; \
    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) { \
        float acc = 0.0f; \
        for (size_t k = 0; k < n_taps; ++k) { \
            const size_t r = (size_t)ii[k]; \
            acc += wi[k] * vis_to_float(table[r * dim + d]); \
        } \
        orow[d] = vis_from_float<T>(acc); \
    } \
}

OWL_POS_EMBED_KERNEL(owl_pos_embed_bilinear_f32, float, (const float* table, const float* idx, const float* wts, float* out, const size_t n_taps, const size_t dim))
OWL_POS_EMBED_KERNEL(owl_pos_embed_bilinear_f16, __half, (const __half* table, const float* idx, const float* wts, __half* out, const size_t n_taps, const size_t dim))

// ============================================================================
// ⑤ AttnFull:非因果 varlen 全注意力(逐图;**无 GQA**(塔 hq == hkv))
//
// 版一(2026-10-15 首版):一 warp 一行 × 一头 ⇒ 每行独立扫全 KV:
//   实测 1280×736(3680 patch)⇒ **50.7 ms/层**(3680² × 72 × 16 头 × 2(K+V) ≈ 62 GB
//   的 L2/DRAM 重读)⇒ 占整塔 92% ⇒ 必须共享 K/V。
// 版二(2026-10-15):一 block 8 行(8 warp)× 一头,K/V 按 TILE=64 key 协作装 smem。
// 版三(2026-10-23):**一 block 32 行(8 warp × 4 行)× 一头**,每 warp 将同一 K/V
//   行复用于 4 个 q 行 ⇒ K/V 重读量再 ÷4;K/V tile 用 16B 向量装载。
//   行区间按 `cu_seqlens` 逐子行裁(块内可跨图;越界 key 跳过)。
//
//   q [rows, heads, head_dim] / k,v 同形 / cu_seqlens [n_seq+1](f32 段界)
//   grid = (ceil(rows/32), heads);block 256;动态 smem = 2·TILE·head_dim·sizeof(T)
//   在线 softmax(一遍);lane 持 dim = lane, lane+32, lane+64(f32 累计)
// ============================================================================

#define OWL_VIS_ATTN_TILE 64

#define OWL_VIS_ATTN_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    constexpr int TILE = OWL_VIS_ATTN_TILE; \
    constexpr int WARPS = 8; \
    constexpr int RPW = 4; /* 每 warp 4 行:32 行/block,复写 K/V smem */ \
    extern __shared__ char vsm[]; \
    T* ks = reinterpret_cast<T*>(vsm); \
    T* vv = ks + (size_t)TILE * head_dim; \
    const int warp = (int)(threadIdx.x >> 5); \
    const int lane = (int)(threadIdx.x & 31u); \
    const size_t row0 = (size_t)blockIdx.x * (WARPS * RPW) + (size_t)warp * RPW; \
    const int head = blockIdx.y; \
    bool valid[RPW]; \
    int s[RPW], e[RPW]; \
    /* 每个子行独立查 cu_seqlens(块内可跨图) */ \
    _Pragma("unroll") \
    for (int rr = 0; rr < RPW; ++rr) { \
        const size_t row = row0 + (size_t)rr; \
        valid[rr] = (row < rows); \
        s[rr] = 0; e[rr] = 0; \
        if (valid[rr]) { \
            for (int t = 0; t < n_seq; ++t) { \
                const int a = (int)cu_seqlens[t], b = (int)cu_seqlens[t + 1]; \
                if ((int)row >= a && (int)row < b) { s[rr] = a; e[rr] = b; break; } \
            } \
        } \
    } \
    /* 块级 K/V 扫描区间 = 首行与末行各自所属图的并(段界单调) */ \
    const size_t row_lo = (size_t)blockIdx.x * (WARPS * RPW); \
    const size_t row_hi = row_lo + (WARPS * RPW) - 1; \
    int block_lo = 0, block_hi = (int)rows; \
    for (int t = 0; t < n_seq; ++t) { \
        const int a = (int)cu_seqlens[t], b = (int)cu_seqlens[t + 1]; \
        if ((int)row_lo >= a && (int)row_lo < b) { block_lo = a; } \
        if ((int)row_hi >= a && (int)row_hi < b) { block_hi = b; } \
    } \
    if (block_lo > block_hi) { block_hi = block_lo; } \
    \
    float qv[RPW][3]; \
    _Pragma("unroll") \
    for (int rr = 0; rr < RPW; ++rr) { \
        const size_t row = row0 + (size_t)rr; \
        _Pragma("unroll") \
        for (int r = 0; r < 3; ++r) { \
            const int d = lane + 32 * r; \
            qv[rr][r] = (valid[rr] && d < (int)head_dim) \
                        ? vis_to_float(q[((size_t)row * heads + (size_t)head) * head_dim + d]) \
                        : 0.0f; \
        } \
    } \
    float m[RPW], l[RPW], acc[RPW][3]; \
    _Pragma("unroll") \
    for (int rr = 0; rr < RPW; ++rr) { \
        m[rr] = -3.0e38f; \
        l[rr] = 0.0f; \
        _Pragma("unroll") \
        for (int r = 0; r < 3; ++r) { acc[rr][r] = 0.0f; } \
    } \
    /* 16B 向量装载 K/V tile(2026-10-23):native 慢核的主因之一是 \
       每元素 2B scalar load;hd=72 正好 9×16B,可整行向量化。 */ \
    constexpr int VEC = 16 / (int)sizeof(T); \
    const int vecs = (int)head_dim / VEC; \
    const bool vec_ok = ((int)head_dim % VEC) == 0; \
    for (int t0 = block_lo; t0 < block_hi; t0 += TILE) { \
        __syncthreads(); \
        /* 协作装载 tile(越块尾补 0;补 0 的 key 被行区间裁掉) */ \
        if (vec_ok) { \
            for (int idx = (int)threadIdx.x; idx < TILE * vecs; idx += (int)blockDim.x) { \
                const int j = idx / vecs; \
                const int vcol = idx - j * vecs; \
                const int tg = t0 + j; \
                uint4 kk = make_uint4(0u, 0u, 0u, 0u); \
                uint4 vv4 = make_uint4(0u, 0u, 0u, 0u); \
                if (tg < block_hi) { \
                    const size_t off = ((size_t)tg * heads + (size_t)head) * head_dim + (size_t)vcol * VEC; \
                    kk = *reinterpret_cast<const uint4*>(k + off); \
                    vv4 = *reinterpret_cast<const uint4*>(v + off); \
                } \
                const size_t so = (size_t)j * head_dim + (size_t)vcol * VEC; \
                *reinterpret_cast<uint4*>(ks + so) = kk; \
                *reinterpret_cast<uint4*>(vv + so) = vv4; \
            } \
        } else { \
            for (int idx = (int)threadIdx.x; idx < TILE * (int)head_dim; idx += (int)blockDim.x) { \
                const int j = idx / (int)head_dim; \
                const int d = idx - j * (int)head_dim; \
                const int tg = t0 + j; \
                if (tg < block_hi) { \
                    const size_t off = ((size_t)tg * heads + (size_t)head) * head_dim + (size_t)d; \
                    ks[idx] = k[off]; \
                    vv[idx] = v[off]; \
                } else { \
                    ks[idx] = vis_from_float<T>(0.0f); \
                    vv[idx] = vis_from_float<T>(0.0f); \
                } \
            } \
        } \
        __syncthreads(); \
        for (int j = 0; j < TILE; ++j) { \
            const int tg = t0 + j; \
            float kf[3], vf[3]; \
            _Pragma("unroll") \
            for (int r = 0; r < 3; ++r) { \
                const int d = lane + 32 * r; \
                const bool in = d < (int)head_dim; \
                kf[r] = in ? vis_to_float(ks[(size_t)j * head_dim + (size_t)d]) : 0.0f; \
                vf[r] = in ? vis_to_float(vv[(size_t)j * head_dim + (size_t)d]) : 0.0f; \
            } \
            _Pragma("unroll") \
            for (int rr = 0; rr < RPW; ++rr) { \
                if (!valid[rr] || tg < s[rr] || tg >= e[rr]) { continue; } \
                float part = 0.0f; \
                _Pragma("unroll") \
                for (int r = 0; r < 3; ++r) { part += qv[rr][r] * kf[r]; } \
                for (int off = 16; off > 0; off >>= 1) { \
                    part += __shfl_xor_sync(0xffffffffu, part, off); \
                } \
                const float sc = part * scale; \
                const float m_new = fmaxf(m[rr], sc); \
                const float corr = __expf(m[rr] - m_new); \
                const float e_w = __expf(sc - m_new); \
                l[rr] = l[rr] * corr + e_w; \
                _Pragma("unroll") \
                for (int r = 0; r < 3; ++r) { \
                    acc[rr][r] = acc[rr][r] * corr + e_w * vf[r]; \
                } \
                m[rr] = m_new; \
            } \
        } \
    } \
    _Pragma("unroll") \
    for (int rr = 0; rr < RPW; ++rr) { \
        if (!valid[rr]) { continue; } \
        const size_t row = row0 + (size_t)rr; \
        const float inv = (l[rr] > 0.0f) ? (1.0f / l[rr]) : 0.0f; \
        _Pragma("unroll") \
        for (int r = 0; r < 3; ++r) { \
            const int d = lane + 32 * r; \
            if (d < (int)head_dim) { \
                out[((size_t)row * heads + (size_t)head) * head_dim + (size_t)d] = \
                    vis_from_float<T>(acc[rr][r] * inv); \
            } \
        } \
    } \
}

OWL_VIS_ATTN_KERNEL(owl_vis_attn_full_f32, float, (const float* q, const float* k, const float* v, const float* cu_seqlens, const size_t rows, const size_t heads, const size_t head_dim, const int n_seq, const float scale, float* out))
OWL_VIS_ATTN_KERNEL(owl_vis_attn_full_f16, __half, (const __half* q, const __half* k, const __half* v, const float* cu_seqlens, const size_t rows, const size_t heads, const size_t head_dim, const int n_seq, const float scale, __half* out))

// ============================================================================
// ⑥ ImageScatter(VLM 图像 token 注入):out[t] = idx[t] ≥ 0 ? merged[idx[t]] : h[t]
//
//   文本侧接线算子(非塔内算子):embed 出来的 h [L, D] 按占位 token 换成塔输出
//   merged [N, D] 的对应行;idx[t] < 0 = 该位是文本 token,原样保留。
//   落本族因同属 VLM 面(与塔同批交付)。
//
//   grid = (rows,1,1);block 256;idx 恒 f32(索引量 f32 过线,契约 5)。
//   ⚠️ 行号越界 = 未定义(照 rope 的 pos 先例:**host 侧构造时**已按
//   N/占位 token 数互证,核内不查 —— 见 `TensorOps::image_scatter`)。
// ============================================================================
#define OWL_IMAGE_SCATTER_KERNEL(NAME, T, PARAMS) \
extern "C" __global__ void NAME PARAMS { \
    const size_t row = blockIdx.x; \
    const float r = idx[row]; \
    const T* src = (r < 0.0f) ? (h + row * dim) : (merged + (size_t)r * dim); \
    T* orow = out + row * dim; \
    for (size_t c = threadIdx.x; c < dim; c += blockDim.x) { \
        orow[c] = src[c]; \
    } \
}

OWL_IMAGE_SCATTER_KERNEL(owl_image_scatter_f32, float, (const float* h, const float* merged, const float* idx, float* out, const size_t dim))
OWL_IMAGE_SCATTER_KERNEL(owl_image_scatter_f16, __half, (const __half* h, const __half* merged, const float* idx, __half* out, const size_t dim))
