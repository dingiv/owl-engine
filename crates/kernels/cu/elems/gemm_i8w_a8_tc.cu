// ============================================================================
// owl 基础算子族 —— **W8A8 多头 GEMM(int8 Tensor Core)**
//
// 语义与 `gemv_i8w_a8_mt*` 逐位相同:
//   y[t, r] = (sw[r] · sx[t]) · Σ_k w_i8[r,k] · q_i8[t,k]
// 区别只在算术:sm80+ 的 `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
// (int8 IMMA)代替 `__dp4a`。**SM86 有 int8 Tensor Core、没有 fp8 TC** ——
// 这条是本机唯一可用的 8-bit 矩阵乘硬件路径;不要把它写成 fp8。
//
// # 形状/Tile
//
// - M = 16(固定):spec 只有 7/8 个 token,其余行补 0;A = 激活 [tokens, K]。
// - N = 8 权重行/MMA;一个 warp 一条 N=8 的竖条。
// - K = 32/MMA;block 先按 K_TILE=512 把 **B 的 64 行**搬进 shared,
//   再让 8 warp 各自从 shared 取 B fragment —— 关键是 B 的 global 读
//   一次 16B、按 row 连续(8 行 × 512B),而不是每 warp 散读 16B。
// - 一个 block 覆盖 64 个权重行;grid.x = ceil(rows/64)。
// - A 也按 K_TILE 搬进 shared;A 的 16 行在 block 内共享。
// - 输出先落 shared [8,64],再按 token 行连续写 y([tokens, rows])。
//
// # 为什么不用 8 行 MMA
//
// Ampere 的 int8 IMMA 档位是 m16n8k32;M=7/8 只能补到 16(50% 算力空转)。
// 这条路的墙是**权重带宽**,不是算力,所以补行不改变量级;实测 3080
// 124160×5120/tokens=8 ≈ 1.07 ms(0.635 GB,~590 GB/s),dp4a 版 ≈ 1.33 ms。
//
// 契约:`w [rows,cols] I8` / `sw [rows] f32` / `q [tokens,cols] I8` /
//       `sx [tokens] f32` / i32 rows,cols,tokens / `y [tokens,rows]` T。
// 约束:`cols % 32 == 0`;`2 <= tokens <= 8`;`rows > 0`。
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#define OWL_TC_TOK_PAD 16
#define OWL_TC_K_TILE 512
#define OWL_TC_A_STRIDE (OWL_TC_K_TILE + 16)
#define OWL_TC_B_STRIDE (OWL_TC_K_TILE + 16)
#define OWL_TC_ROWS_PER_BLOCK 64
#define OWL_TC_WARPS 8

template <typename T> __device__ __forceinline__ T owl_tc_from_f(float x);
template <> __device__ __forceinline__ __half owl_tc_from_f<__half>(float x) {
    return __float2half(x);
}
template <> __device__ __forceinline__ __nv_bfloat16 owl_tc_from_f<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ uint32_t owl_tc_ld4(const signed char* p) {
    return *reinterpret_cast<const uint32_t*>(p);
}

template <typename T>
__device__ __forceinline__ void owl_gemm_i8w_a8_tc_body(
    const signed char* __restrict__ w,  // [rows, cols]
    const float* __restrict__ sw,       // [rows]
    const signed char* __restrict__ q,  // [tokens, cols]
    const float* __restrict__ sx,       // [tokens]
    int rows, int cols, int tokens,
    T* __restrict__ y) {                // [tokens, rows]
    __shared__ signed char as[OWL_TC_TOK_PAD][OWL_TC_A_STRIDE];
    __shared__ signed char bs[OWL_TC_ROWS_PER_BLOCK][OWL_TC_B_STRIDE];
    __shared__ T os[8][OWL_TC_ROWS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;      // 0..7:MMA 的 row group
    const int t4 = lane & 3;      // 0..3:4-byte K group
    const int block_n = (int)blockIdx.x * OWL_TC_ROWS_PER_BLOCK;
    const int warp_n = block_n + warp * 8;

    int c[4] = {0, 0, 0, 0};
    for (int k0 = 0; k0 < cols; k0 += OWL_TC_K_TILE) {
        int kt = cols - k0;
        if (kt > OWL_TC_K_TILE) kt = OWL_TC_K_TILE;

        // A tile:16 行 × 512B;行 >= tokens 或 k >= kt 写 0。
        for (int idx = tid; idx < OWL_TC_TOK_PAD * (OWL_TC_K_TILE / 4);
             idx += (int)blockDim.x) {
            const int r = idx / (OWL_TC_K_TILE / 4);
            const int j = idx % (OWL_TC_K_TILE / 4);
            uint32_t v = 0;
            if (r < tokens && j * 4 + 3 < kt) {
                v = owl_tc_ld4(q + (size_t)r * cols + k0 + j * 4);
            }
            *reinterpret_cast<uint32_t*>(&as[r][j * 4]) = v;
        }
        // B tile:64 行 × 512B,global 侧按 row 连续 16B 搬。
        for (int idx = tid; idx < OWL_TC_ROWS_PER_BLOCK * (OWL_TC_K_TILE / 16);
             idx += (int)blockDim.x) {
            const int r = idx / (OWL_TC_K_TILE / 16);
            const int j = idx % (OWL_TC_K_TILE / 16);
            const int n = block_n + r;
            uint4 v = {0, 0, 0, 0};
            if (n < rows && j * 16 + 15 < kt) {
                v = *reinterpret_cast<const uint4*>(w + (size_t)n * cols + k0 + j * 16);
            }
            *reinterpret_cast<uint4*>(&bs[r][j * 16]) = v;
        }
        __syncthreads();

        for (int kk = 0; kk < kt; kk += 32) {
            uint32_t a[4], b[2];
            a[0] = owl_tc_ld4(&as[g][kk + t4 * 4]);
            a[1] = owl_tc_ld4(&as[g + 8][kk + t4 * 4]);
            a[2] = owl_tc_ld4(&as[g][kk + 16 + t4 * 4]);
            a[3] = owl_tc_ld4(&as[g + 8][kk + 16 + t4 * 4]);
            const int n = warp * 8 + g;
            b[0] = owl_tc_ld4(&bs[n][kk + t4 * 4]);
            b[1] = owl_tc_ld4(&bs[n][kk + 16 + t4 * 4]);
            // int8×int8 → int32;不启用 satfinite(和 dp4a 一样精确整数和)。
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                  "r"(b[0]), "r"(b[1]));
        }
        __syncthreads();
    }

    // 只消费 MMA 的 row 0..7(c0/c1);row 8..15 对 tokens<=8 全是补零。
    if (g < tokens) {
        const int n = warp_n + t4 * 2;
        const float s0 = (n < rows) ? sw[n] * sx[g] : 0.0f;
        const float s1 = (n + 1 < rows) ? sw[n + 1] * sx[g] : 0.0f;
        os[g][warp * 8 + t4 * 2] = owl_tc_from_f<T>((float)c[0] * s0);
        os[g][warp * 8 + t4 * 2 + 1] = owl_tc_from_f<T>((float)c[1] * s1);
    }
    __syncthreads();
    // 按 token 行连续写;每 block 覆盖 64 个权重列。
    for (int idx = tid; idx < 8 * OWL_TC_ROWS_PER_BLOCK; idx += (int)blockDim.x) {
        const int t = idx / OWL_TC_ROWS_PER_BLOCK;
        const int col = idx % OWL_TC_ROWS_PER_BLOCK;
        const int n = block_n + col;
        if (t < tokens && n < rows) {
            y[(size_t)t * rows + n] = os[t][col];
        }
    }
}

extern "C" __global__ void owl_gemm_i8w_a8_tc_f16(
    const signed char* __restrict__ w, const float* __restrict__ sw,
    const signed char* __restrict__ q, const float* __restrict__ sx,
    const int rows, const int cols, const int tokens, __half* __restrict__ y) {
    owl_gemm_i8w_a8_tc_body<__half>(w, sw, q, sx, rows, cols, tokens, y);
}

extern "C" __global__ void owl_gemm_i8w_a8_tc_bf16(
    const signed char* __restrict__ w, const float* __restrict__ sw,
    const signed char* __restrict__ q, const float* __restrict__ sx,
    const int rows, const int cols, const int tokens,
    __nv_bfloat16* __restrict__ y) {
    owl_gemm_i8w_a8_tc_body<__nv_bfloat16>(w, sw, q, sx, rows, cols, tokens, y);
}
