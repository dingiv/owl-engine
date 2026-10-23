// ============================================================================
// owl 基础算子族 —— **W8A16 多头 GEMM(f16 Tensor Core + 在线反量化)**
//
// 语义与 `gemv_i8w_mt*` 相同:
//   y[t, r] = sw[r] · Σ_k w_i8[r,k] · x_t[t,k]
// 但算术走 **f16/bf16 Tensor Core**:
//   - HBM 读 int8 权重(1 B/参数;不落 f16 副本);
//   - B tile 进 shared 仍是 int8;
//   - 每个 MMA fragment 在寄存器里 `int8 → half/bf16`(Marlin 的同款
//     “压缩读 + 近端反量化”流水);
//   - 激活保持 f16/bf16,不做 W8A8 的激活量化 ⇒ 质量仍是 W8A16。
//
// 为什么不是 int8 MMA:W8A16 的 A 是 f16,`mma.s8s8s32` 吃不了;把 A 量化
// 成 int8 就是 W8A8(会掉质量)。所以这里用 f16/bf16 MMA,只把 B 侧的
// int8 权重在线展开。
//
// # Tile
//   M=16(7/8 token 补零);N=8 权重行/warp;CTA 8 warp = 64 行;
//   K_TILE=256;A shared `16×264`(f16/bf16);B shared `64×272`(int8);
//   out shared `8×64`;静态 shared ≈ 26.3 KB ⇒ 3 block/SM。
//   B 的 global 读 16B/row 连续;A 也按 16B 搬。
//
// 契约:`w [rows,cols] I8` / `sw [rows] f32` / `x [tokens,cols] T` /
//       i32 rows, cols, tokens / `y [tokens,rows] T`。
// 约束:`cols % 16 == 0`;`2 <= tokens <= 8`;`rows > 0`。
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#define OWL_A16TC_TOK_PAD 16
#define OWL_A16TC_K_TILE 256
#define OWL_A16TC_A_STRIDE (OWL_A16TC_K_TILE + 8)
#define OWL_A16TC_B_STRIDE (OWL_A16TC_K_TILE + 16)
#define OWL_A16TC_ROWS_PER_BLOCK 64
#define OWL_A16TC_WARPS 8

template <typename T> __device__ __forceinline__ T owl_a16tc_from_int(int v);
template <> __device__ __forceinline__ __half owl_a16tc_from_int<__half>(int v) {
    return __int2half_rn(v);
}
template <> __device__ __forceinline__ __nv_bfloat16 owl_a16tc_from_int<__nv_bfloat16>(int v) {
    return __int2bfloat16_rn(v);
}

template <typename T> __device__ __forceinline__ uint32_t owl_a16tc_pack2(T a, T b);
template <> __device__ __forceinline__ uint32_t owl_a16tc_pack2<__half>(__half a, __half b) {
    const __half2 h = __halves2half2(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}
template <> __device__ __forceinline__ uint32_t owl_a16tc_pack2<__nv_bfloat16>(
    __nv_bfloat16 a, __nv_bfloat16 b) {
    const __nv_bfloat162 h = __halves2bfloat162(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

template <typename T> __device__ __forceinline__ T owl_a16tc_from_f(float x);
template <> __device__ __forceinline__ __half owl_a16tc_from_f<__half>(float x) {
    return __float2half(x);
}
template <> __device__ __forceinline__ __nv_bfloat16 owl_a16tc_from_f<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <typename T, bool BF16>
__device__ __forceinline__ void owl_gemm_i8w_a16_tc_body(
    const signed char* __restrict__ w,  // [rows, cols] int8
    const float* __restrict__ sw,       // [rows]
    const T* __restrict__ x,            // [tokens, cols] f16/bf16
    int rows, int cols, int tokens,
    T* __restrict__ y) {                // [tokens, rows]
    __shared__ T as[OWL_A16TC_TOK_PAD][OWL_A16TC_A_STRIDE];
    __shared__ signed char bs[OWL_A16TC_ROWS_PER_BLOCK][OWL_A16TC_B_STRIDE];
    __shared__ T os[8][OWL_A16TC_ROWS_PER_BLOCK];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int block_n = (int)blockIdx.x * OWL_A16TC_ROWS_PER_BLOCK;
    const int warp_n = block_n + warp * 8;

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int k0 = 0; k0 < cols; k0 += OWL_A16TC_K_TILE) {
        int kt = cols - k0;
        if (kt > OWL_A16TC_K_TILE) kt = OWL_A16TC_K_TILE;

        // A:f16/bf16 tile(行 >= tokens 或 k >= kt 补 0)
        for (int idx = tid; idx < OWL_A16TC_TOK_PAD * (OWL_A16TC_K_TILE / 8);
             idx += (int)blockDim.x) {
            const int r = idx / (OWL_A16TC_K_TILE / 8);
            const int j = idx % (OWL_A16TC_K_TILE / 8);
            uint4 v = {0, 0, 0, 0};
            if (r < tokens && j * 8 + 7 < kt) {
                v = *reinterpret_cast<const uint4*>(x + (size_t)r * cols + k0 + j * 8);
            }
            *reinterpret_cast<uint4*>(&as[r][j * 8]) = v;
        }
        // B:int8 tile(HBM 16B/row 连续)
        for (int idx = tid; idx < OWL_A16TC_ROWS_PER_BLOCK * (OWL_A16TC_K_TILE / 16);
             idx += (int)blockDim.x) {
            const int r = idx / (OWL_A16TC_K_TILE / 16);
            const int j = idx % (OWL_A16TC_K_TILE / 16);
            const int n = block_n + r;
            uint4 v = {0, 0, 0, 0};
            if (n < rows && j * 16 + 15 < kt) {
                v = *reinterpret_cast<const uint4*>(w + (size_t)n * cols + k0 + j * 16);
            }
            *reinterpret_cast<uint4*>(&bs[r][j * 16]) = v;
        }
        __syncthreads();

        for (int kk = 0; kk < kt; kk += 16) {
            uint32_t a[4], b[2];
            a[0] = *reinterpret_cast<const uint32_t*>(&as[g][kk + 2 * t4]);
            a[1] = *reinterpret_cast<const uint32_t*>(&as[g + 8][kk + 2 * t4]);
            a[2] = *reinterpret_cast<const uint32_t*>(&as[g][kk + 2 * t4 + 8]);
            a[3] = *reinterpret_cast<const uint32_t*>(&as[g + 8][kk + 2 * t4 + 8]);
            const int n = warp * 8 + g;
            const signed char* bp0 = &bs[n][kk + 2 * t4];
            const signed char* bp1 = &bs[n][kk + 2 * t4 + 8];
            b[0] = owl_a16tc_pack2<T>(
                owl_a16tc_from_int<T>((int)bp0[0]),
                owl_a16tc_from_int<T>((int)bp0[1]));
            b[1] = owl_a16tc_pack2<T>(
                owl_a16tc_from_int<T>((int)bp1[0]),
                owl_a16tc_from_int<T>((int)bp1[1]));
            if constexpr (BF16) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(b[0]), "r"(b[1]));
            } else {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(b[0]), "r"(b[1]));
            }
        }
        __syncthreads();
    }

    // 只消费 row 0..7(c0/c1);row 8..15 对 tokens<=8 全补零。
    if (g < tokens) {
        const int n = warp_n + t4 * 2;
        const float s0 = (n < rows) ? sw[n] : 0.0f;
        const float s1 = (n + 1 < rows) ? sw[n + 1] : 0.0f;
        os[g][warp * 8 + t4 * 2] = owl_a16tc_from_f<T>(c[0] * s0);
        os[g][warp * 8 + t4 * 2 + 1] = owl_a16tc_from_f<T>(c[1] * s1);
    }
    __syncthreads();
    for (int idx = tid; idx < 8 * OWL_A16TC_ROWS_PER_BLOCK; idx += (int)blockDim.x) {
        const int t = idx / OWL_A16TC_ROWS_PER_BLOCK;
        const int col = idx % OWL_A16TC_ROWS_PER_BLOCK;
        const int n = block_n + col;
        if (t < tokens && n < rows) {
            y[(size_t)t * rows + n] = os[t][col];
        }
    }
}

extern "C" __global__ void owl_gemm_i8w_a16_tc_f16(
    const signed char* __restrict__ w, const float* __restrict__ sw,
    const __half* __restrict__ x, const int rows, const int cols, const int tokens,
    __half* __restrict__ y) {
    owl_gemm_i8w_a16_tc_body<__half, false>(w, sw, x, rows, cols, tokens, y);
}

extern "C" __global__ void owl_gemm_i8w_a16_tc_bf16(
    const signed char* __restrict__ w, const float* __restrict__ sw,
    const __nv_bfloat16* __restrict__ x, const int rows, const int cols, const int tokens,
    __nv_bfloat16* __restrict__ y) {
    owl_gemm_i8w_a16_tc_body<__nv_bfloat16, true>(w, sw, x, rows, cols, tokens, y);
}
