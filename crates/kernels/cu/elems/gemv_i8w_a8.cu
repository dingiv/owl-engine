// ============================================================================
// owl 基础算子族 —— **W8A8 多 token 头**(int8 权重 × int8 激活;dp4a)
//
// 语义:`y[t, r] = (sw[r] · sx[t]) · Σ_k w_i8[r,k] · q_i8[t,k]`
// - `w`  per-row 对称 int8 + `sw` f32 [rows](现有 int8 头权重,零改动)
// - `q`  per-token 对称 int8 + `sx` f32 [tokens](`quant_i8_act_*` 产出)
// - 内积走 `__dp4a`(signed int8×int8 → int32),每 4 个 k 一条指令。
//
// 与 W8A16 mt 核同几何:TOK_TILE=8、ROWS_PER_WARP=4、WARPS=8、K_TILE=512;
// 一个权重字节复用到 4 行 × 8 token。区别只在 x 是 int8 且内层是 dp4a。
//
// # shared 布局(4-byte group 的 bank swizzle)
//
// 一个 group = 4 个连续 int8 = 1 word。group j 存到 word `j + (j>>2)`;
// 这样 lane l 在 base=l*16 的 group j=4l+g 落到 word 5l+g,固定 g 下
// 32 lane 命中 32 个 bank(无冲突)。XSTRIDE = K_TILE/4 + K_TILE/16 + 1
// (K=512 ⇒ 161 words;最大用到 158)。
//
// 契约:`w [rows, cols] i8` / `sw [rows] f32` / `xq [tokens, cols] i8` /
//       `sx [tokens] f32` / i32 rows, cols, tokens / `y [tokens, rows]` T。
// 约束:`cols % 16 == 0`(权重 uint4 / group 对齐)。
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define OWL_GEMV_I8W_A8_TOK_TILE 8
#define OWL_GEMV_I8W_A8_ROWS_PER_WARP 4
#define OWL_GEMV_I8W_A8_WARPS 8
#define OWL_GEMV_I8W_A8_K_TILE 512
#define OWL_GEMV_I8W_A8_XSTRIDE \
    (OWL_GEMV_I8W_A8_K_TILE / 4 + OWL_GEMV_I8W_A8_K_TILE / 16 + 1)

template <typename T> __device__ __forceinline__ T owl_i8a8_from_f(float x);
template <> __device__ __forceinline__ __half owl_i8a8_from_f<__half>(float x) {
    return __float2half(x);
}
template <> __device__ __forceinline__ __nv_bfloat16 owl_i8a8_from_f<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <typename T, int TOK_N>
__device__ __forceinline__ void owl_gemv_i8w_a8_mt_body(
    const signed char* __restrict__ w,  // [rows, cols]
    const float* __restrict__ sw,       // [rows]
    const signed char* __restrict__ xq, // [tokens, cols]
    const float* __restrict__ sx,       // [tokens]
    const int rows,
    const int cols,
    const int tokens,
    T* __restrict__ y) {                // [tokens, rows]
    extern __shared__ int32_t xs[];     // [TOK_N][XSTRIDE] words
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int token0 = (int)blockIdx.y * TOK_N;
    int nt = tokens - token0;
    if (nt > TOK_N) nt = TOK_N;
    const int row0 = ((int)blockIdx.x * OWL_GEMV_I8W_A8_WARPS + warp)
                     * OWL_GEMV_I8W_A8_ROWS_PER_WARP;

    int acc[OWL_GEMV_I8W_A8_ROWS_PER_WARP][TOK_N];
#pragma unroll
    for (int r = 0; r < OWL_GEMV_I8W_A8_ROWS_PER_WARP; ++r) {
#pragma unroll
        for (int t = 0; t < TOK_N; ++t) acc[r][t] = 0;
    }

    const int groups_per_row = cols >> 2;  // 4 个 int8/word
    for (int k0 = 0; k0 < cols; k0 += OWL_GEMV_I8W_A8_K_TILE) {
        int kt = cols - k0;
        if (kt > OWL_GEMV_I8W_A8_K_TILE) kt = OWL_GEMV_I8W_A8_K_TILE;
        const int gt = kt >> 2;  // 本 tile 的 group 数

        // 固定装 TOK_N 行:nt<TOK_N 的尾部 token 补 0(写共享之前钳行,
        // 避免 t>=nt 的越界指针被 ptxas 提前成 LDG)。
#pragma unroll
        for (int t = 0; t < TOK_N; ++t) {
            const bool has = (t < nt);
            const int tt = has ? t : 0;
            const int32_t* xrow = reinterpret_cast<const int32_t*>(xq)
                + (size_t)(token0 + tt) * groups_per_row + (k0 >> 2);
            int32_t* srow = xs + t * OWL_GEMV_I8W_A8_XSTRIDE;
            for (int g = tid; g < gt; g += (int)blockDim.x) {
                srow[g + (g >> 2)] = has ? xrow[g] : 0;
            }
        }
        __syncthreads();

        if (row0 < rows) {
            for (int base = lane * 16; base < kt; base += 32 * 16) {
                uint4 wv[OWL_GEMV_I8W_A8_ROWS_PER_WARP];
#pragma unroll
                for (int r = 0; r < OWL_GEMV_I8W_A8_ROWS_PER_WARP; ++r) {
                    wv[r] = *reinterpret_cast<const uint4*>(
                        w + (size_t)(row0 + r) * cols + k0 + base);
                }
                // base=lane*16 ⇒ 本 lane 的 4 个 group 的共享 word = 5*lane+g
#pragma unroll
                for (int g = 0; g < 4; ++g) {
                    int32_t xv[TOK_N];
#pragma unroll
                    for (int t = 0; t < TOK_N; ++t) {
                        xv[t] = xs[t * OWL_GEMV_I8W_A8_XSTRIDE + 5 * lane + g];
                    }
#pragma unroll
                    for (int r = 0; r < OWL_GEMV_I8W_A8_ROWS_PER_WARP; ++r) {
                        const int32_t w4 =
                            reinterpret_cast<const int32_t*>(&wv[r])[g];
#pragma unroll
                        for (int t = 0; t < TOK_N; ++t) {
                            acc[r][t] = __dp4a(w4, xv[t], acc[r][t]);
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (row0 < rows) {
        float sxt[TOK_N];
#pragma unroll
        for (int t = 0; t < TOK_N; ++t) {
            sxt[t] = (t < nt) ? sx[token0 + t] : 0.0f;
        }
        // int32 求和是结合的;跨 lane 归约不影响精确性。
#pragma unroll
        for (int r = 0; r < OWL_GEMV_I8W_A8_ROWS_PER_WARP; ++r) {
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
            for (int r = 0; r < OWL_GEMV_I8W_A8_ROWS_PER_WARP; ++r) {
                const float wscale = sw[row0 + r];
#pragma unroll
                for (int t = 0; t < TOK_N; ++t) {
                    if (t < nt) {
                        const float s = wscale * sxt[t];
                        y[(size_t)(token0 + t) * rows + row0 + r] =
                            owl_i8a8_from_f<T>((float)acc[r][t] * s);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void owl_gemv_i8w_a8_mt_f16(
    const signed char* __restrict__ w, const float* __restrict__ sw,
    const signed char* __restrict__ xq, const float* __restrict__ sx,
    const int rows, const int cols, const int tokens, __half* __restrict__ y) {
    owl_gemv_i8w_a8_mt_body<__half, 8>(w, sw, xq, sx, rows, cols, tokens, y);
}

extern "C" __global__ void owl_gemv_i8w_a8_mt_bf16(
    const signed char* __restrict__ w, const float* __restrict__ sw,
    const signed char* __restrict__ xq, const float* __restrict__ sx,
    const int rows, const int cols, const int tokens,
    __nv_bfloat16* __restrict__ y) {
    owl_gemv_i8w_a8_mt_body<__nv_bfloat16, 8>(w, sw, xq, sx, rows, cols, tokens, y);
}

extern "C" __global__ void owl_gemv_i8w_a8_mt7_f16(
    const signed char* __restrict__ w, const float* __restrict__ sw,
    const signed char* __restrict__ xq, const float* __restrict__ sx,
    const int rows, const int cols, const int tokens, __half* __restrict__ y) {
    owl_gemv_i8w_a8_mt_body<__half, 7>(w, sw, xq, sx, rows, cols, tokens, y);
}

extern "C" __global__ void owl_gemv_i8w_a8_mt7_bf16(
    const signed char* __restrict__ w, const float* __restrict__ sw,
    const signed char* __restrict__ xq, const float* __restrict__ sx,
    const int rows, const int cols, const int tokens,
    __nv_bfloat16* __restrict__ y) {
    owl_gemv_i8w_a8_mt_body<__nv_bfloat16, 7>(w, sw, xq, sx, rows, cols, tokens, y);
}
