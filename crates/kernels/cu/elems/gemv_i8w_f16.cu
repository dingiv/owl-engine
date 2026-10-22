// ============================================================================
// owl 基础算子族 —— **int8 权重 × f16 激活的 GEMV**(m=1 decode 专用;P1.5-a)
//
// # 为什么有它
//
// §71 的账:decode 每卡每 token 读 **8.29 GB**,其中 lm_head 占 **1.27 GB(bf16)**;
// lm_head 的 gemv 已经跑到 **724 GB/s(95% 峰值)** ⇒ 只剩"减字节"一条路。
// 本核 = **int8 权重版**(0.635 GB)⇒ 目标 1.76 ms → ~0.88 ms(**+4.3% decode**)。
//
// # 与 lm_head 现役路的差别(只有两处)
//
// | 面 | 现役(cublas gemv, bf16) | 本核 |
// |---|---|---|
// | 权重 | bf16,2 B/参数 | **int8 + per-row f32 尺度**(1 B/参数)|
// | 累加 | cublas 内部 | fp32 逐 lane → `__shfl_down_sync` 5 步 warp 归约 |
//
// **语义**:`y[token, r] = scale[r] · Σ_k ( int8)w[r,k] · x[token,k]`
// (per-row 对称量化:装载期 `s_r = max|w_r|/127`,`q = round(w/s)`)。
//
// # 几何(带宽受限的形态,不是张量核形态)
//
// - **一 warp 一行**(与 `gemv_dual` 同族):行内按 16 B(`uint4` = 16 个 int8)切,
//   32 lane 一次覆盖 512 元素 ⇒ 一个 warp 的 16 B 请求在 lane 间**连成 512 B**;
// - 每个 block 先把 x 那一行(≤ 5120 f16 = 10 KB)搬进 shared,**块内 8 个 warp 共用**
//   (否则每行都要从 L2 重读 10 KB ⇒ 1.24 GB 的 L2 流量,反而把带宽吃掉);
// - fp32 累加 + 5 步归约 + 尾乘 `scale[row]`。
//
// **为什么不是 `dp4a`**:激活是 f16(W8A16),没有 int8×int8 的算术;把激活也量化成
// int8(W8A8)会再引入一层 per-token 量化误差,而收益只在本已不是瓶颈的算力上。
//
// 契约:`w` [rows, cols] int8 行主 · `scale` [rows] f32 · `x` [tokens, cols] f16 ·
//       i32 rows, cols · `y` [tokens, rows] f16(末参 = 输出)。
// 约束:`cols % 16 == 0`(几何侧断言;5120/4096/2048 均满足)。
//
// # 多 token 快路(2026-10-22;spec verify 的 M=7/8 主路)
//
// 上面的 m=1 核把 `grid.y` 当 token:每个 token 的 block 都从头读一遍权重
// ⇒ M=8 时权重流量 ×8(0.635 GB → 5.08 GB)。spec verify 恰好是 8 行、
// DFlash2 propose 是 7 行,这就是 int8 头比 int4 慢 ~150% 的根因。
//
// 本快路把 **token 也分块**:一个 block 一次读权重,同时算 `TOK_TILE` 个 token。
// `ROWS_PER_WARP=4` ⇒ 每读一个权重字节,复用到 4 行 × 8 token = 32 个累加器;
// 另有 `<T,7>` 专用实例(DFlash2 propose 固定 7 行,少算 1 行)。
// 权重只读一遍,x 先落 shared(按 32 位字做 `k + (k>>5)` swizzle,消除
// lane 间 16-float 步长的 bank conflict)。m=1 仍走上面的线性核(prefill 末行)。
//
// 几何:TOK_TILE=8 · ROWS_PER_WARP=4 · WARPS=8(32 行/块) · K_TILE=512。
// shared = 8 × (K_TILE+32) × 4 B = 17 KiB(固定 8 行, nt<8 的尾部补 0;
// 按 TOK_TILE 而不是 nt 分配 —— 按 nt 会让第 8 行写穿动态 shared)。
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define OWL_GEMV_I8W_WARPS 8
#define OWL_GEMV_I8W_CHUNK 16  // 每个 lane 一次读 16 B = 16 个 int8

// ---- 激活 dtype 桥(两个入口共用同一体;只换"激活怎么转 float / 怎么写回")----
template <typename T> __device__ __forceinline__ float owl_i8w_to_f(T x);
template <> __device__ __forceinline__ float owl_i8w_to_f<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float owl_i8w_to_f<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <typename T> __device__ __forceinline__ T owl_i8w_from_f(float x);
template <> __device__ __forceinline__ __half owl_i8w_from_f<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 owl_i8w_from_f<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <typename T>
__device__ __forceinline__ void owl_gemv_i8w_body(
    const signed char* __restrict__ w,  // [rows, cols] int8(行主)
    const float* __restrict__ scale,    // [rows] per-row 对称量化尺度
    const T* __restrict__ x,            // [tokens, cols]
    const int rows,
    const int cols,
    T* __restrict__ y) {                // [tokens, rows](末参 = 输出)
    extern __shared__ unsigned char xs_raw[];  // cols 个 T(块内共用)
    T* xs = reinterpret_cast<T*>(xs_raw);
    const int token = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row = blockIdx.x * OWL_GEMV_I8W_WARPS + warp;

    const T* xrow = x + (size_t)token * cols;
    for (int j = tid; j < cols; j += (int)blockDim.x) xs[j] = xrow[j];
    __syncthreads();

    if (row >= rows) return;

    const signed char* wrow = w + (size_t)row * cols;
    const float s = scale[row];
    float acc = 0.0f;
    for (int base = lane * OWL_GEMV_I8W_CHUNK; base < cols;
         base += 32 * OWL_GEMV_I8W_CHUNK) {
        const uint4 wv = *reinterpret_cast<const uint4*>(wrow + base);
        const signed char* b = reinterpret_cast<const signed char*>(&wv);
#pragma unroll
        for (int i = 0; i < OWL_GEMV_I8W_CHUNK; ++i) {
            const float xv = owl_i8w_to_f<T>(xs[base + i]);
            acc = __fmaf_rn((float)b[i], xv, acc);
        }
    }
    acc *= s;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[(size_t)token * rows + row] = owl_i8w_from_f<T>(acc);
}

extern "C" __global__ void owl_gemv_i8w_f16(
    const signed char* __restrict__ w, const float* __restrict__ scale,
    const __half* __restrict__ x, const int rows, const int cols,
    __half* __restrict__ y) {
    owl_gemv_i8w_body<__half>(w, scale, x, rows, cols, y);
}

// bf16 激活变体(**生产 lm_head 走这条**:模型激活是 bf16,避免两次 cast 节点)
extern "C" __global__ void owl_gemv_i8w_bf16(
    const signed char* __restrict__ w, const float* __restrict__ scale,
    const __nv_bfloat16* __restrict__ x, const int rows, const int cols,
    __nv_bfloat16* __restrict__ y) {
    owl_gemv_i8w_body<__nv_bfloat16>(w, scale, x, rows, cols, y);
}

// ---------------------------------------------------------------------------
// 多 token 快路(cu 侧常量必须与 Rust driver 的 GEMV_I8W_MT_* 同步)
// ---------------------------------------------------------------------------
#define OWL_GEMV_I8W_MT_TOK_TILE 8
#define OWL_GEMV_I8W_MT_ROWS_PER_WARP 4
#define OWL_GEMV_I8W_MT_WARPS 8
#define OWL_GEMV_I8W_MT_K_TILE 512
// 尾部留 32 个 float 给 `k + (k>>5)` swizzle(k=511 → 542)
#define OWL_GEMV_I8W_MT_XSTRIDE (OWL_GEMV_I8W_MT_K_TILE + 32)

template <typename T, int TOK_N>
__device__ __forceinline__ void owl_gemv_i8w_mt_body(
    const signed char* __restrict__ w,  // [rows, cols] int8(行主)
    const float* __restrict__ scale,    // [rows]
    const T* __restrict__ x,            // [tokens, cols]
    const int rows,
    const int cols,
    const int tokens,
    T* __restrict__ y) {                // [tokens, rows]
    extern __shared__ float xs[];       // [TOK_TILE][XSTRIDE]
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int token0 = (int)blockIdx.y * TOK_N;
    int nt = tokens - token0;
    if (nt > TOK_N) nt = TOK_N;
    // driver 保证 rows % (WARPS*ROWS_PER_WARP) == 0 ⇒ 本 block 32 行全有效。
    const int row0 = ((int)blockIdx.x * OWL_GEMV_I8W_MT_WARPS + warp)
                     * OWL_GEMV_I8W_MT_ROWS_PER_WARP;

    float acc[OWL_GEMV_I8W_MT_ROWS_PER_WARP][TOK_N];
#pragma unroll
    for (int r = 0; r < OWL_GEMV_I8W_MT_ROWS_PER_WARP; ++r) {
#pragma unroll
        for (int t = 0; t < TOK_N; ++t) acc[r][t] = 0.0f;
    }

    for (int k0 = 0; k0 < cols; k0 += OWL_GEMV_I8W_MT_K_TILE) {
        int kt = cols - k0;
        if (kt > OWL_GEMV_I8W_MT_K_TILE) kt = OWL_GEMV_I8W_MT_K_TILE;
        // **固定装 8 个 token**:nt<8 的尾部 token 补 0,内层就不必逐 t 判 nt。
        // 逐 token 的 2D 循环(不除 kt;曾用 idx/kt,ptxas 会展开成整段除法)。
#pragma unroll
        for (int t = 0; t < TOK_N; ++t) {
            const bool has = (t < nt);
            // t>=nt 时把行钳到 0:即使 ptxas 把 LDG 提到 select 之前,
            // 地址也仍在 x[0] 内(实测 t=7 的越界指针在 capture 里炸过)。
            const int tt = has ? t : 0;
            const T* xrow = x + (size_t)(token0 + tt) * cols + k0;
            float* srow = xs + t * OWL_GEMV_I8W_MT_XSTRIDE;
            for (int k = tid; k < kt; k += (int)blockDim.x) {
                srow[k + (k >> 5)] = has ? owl_i8w_to_f<T>(xrow[k]) : 0.0f;
            }
        }
        __syncthreads();

        if (row0 < rows) {
            for (int base = lane * OWL_GEMV_I8W_CHUNK; base < kt;
                 base += 32 * OWL_GEMV_I8W_CHUNK) {
                uint4 wv[OWL_GEMV_I8W_MT_ROWS_PER_WARP];
#pragma unroll
                for (int r = 0; r < OWL_GEMV_I8W_MT_ROWS_PER_WARP; ++r) {
                    wv[r] = *reinterpret_cast<const uint4*>(
                        w + (size_t)(row0 + r) * cols + k0 + base);
                }
#pragma unroll
                for (int i = 0; i < OWL_GEMV_I8W_CHUNK; ++i) {
                    const int k = base + i;
                    const int xoff = k + (k >> 5);
                    float xv[TOK_N];
#pragma unroll
                    for (int t = 0; t < TOK_N; ++t) {
                        xv[t] = xs[t * OWL_GEMV_I8W_MT_XSTRIDE + xoff];
                    }
#pragma unroll
                    for (int r = 0; r < OWL_GEMV_I8W_MT_ROWS_PER_WARP; ++r) {
                        const signed char* b =
                            reinterpret_cast<const signed char*>(&wv[r]);
                        const float wf = (float)b[i];
#pragma unroll
                        for (int t = 0; t < TOK_N; ++t) {
                            acc[r][t] = __fmaf_rn(wf, xv[t], acc[r][t]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row0 < rows) {
        // 与 m=1 核逐位同序:先乘本行 scale,再做 lane 归约。
#pragma unroll
        for (int r = 0; r < OWL_GEMV_I8W_MT_ROWS_PER_WARP; ++r) {
            const float s = scale[row0 + r];
#pragma unroll
            for (int t = 0; t < TOK_N; ++t) acc[r][t] *= s;
        }
#pragma unroll
        for (int r = 0; r < OWL_GEMV_I8W_MT_ROWS_PER_WARP; ++r) {
#pragma unroll
            for (int t = 0; t < TOK_N; ++t) {
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    acc[r][t] += __shfl_down_sync(0xffffffff, acc[r][t], off);
                }
            }
        }
        if (lane == 0) {
#pragma unroll
            for (int r = 0; r < OWL_GEMV_I8W_MT_ROWS_PER_WARP; ++r) {
#pragma unroll
                for (int t = 0; t < TOK_N; ++t) {
                    if (t < nt) {
                        y[(size_t)(token0 + t) * rows + row0 + r] =
                            owl_i8w_from_f<T>(acc[r][t]);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void owl_gemv_i8w_mt_f16(
    const signed char* __restrict__ w, const float* __restrict__ scale,
    const __half* __restrict__ x, const int rows, const int cols,
    const int tokens, __half* __restrict__ y) {
    owl_gemv_i8w_mt_body<__half, 8>(w, scale, x, rows, cols, tokens, y);
}

extern "C" __global__ void owl_gemv_i8w_mt_bf16(
    const signed char* __restrict__ w, const float* __restrict__ scale,
    const __nv_bfloat16* __restrict__ x, const int rows, const int cols,
    const int tokens, __nv_bfloat16* __restrict__ y) {
    owl_gemv_i8w_mt_body<__nv_bfloat16, 8>(w, scale, x, rows, cols, tokens, y);
}

// DFlash2 propose 固定 7 行:专用 7-token 实例,少一个累加器/一行 shared。
extern "C" __global__ void owl_gemv_i8w_mt7_f16(
    const signed char* __restrict__ w, const float* __restrict__ scale,
    const __half* __restrict__ x, const int rows, const int cols,
    const int tokens, __half* __restrict__ y) {
    owl_gemv_i8w_mt_body<__half, 7>(w, scale, x, rows, cols, tokens, y);
}

extern "C" __global__ void owl_gemv_i8w_mt7_bf16(
    const signed char* __restrict__ w, const float* __restrict__ scale,
    const __nv_bfloat16* __restrict__ x, const int rows, const int cols,
    const int tokens, __nv_bfloat16* __restrict__ y) {
    owl_gemv_i8w_mt_body<__nv_bfloat16, 7>(w, scale, x, rows, cols, tokens, y);
}
