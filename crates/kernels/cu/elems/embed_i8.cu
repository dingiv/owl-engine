// ============================================================================
// Embedding int8 查表 + per-row 反量化(2026-10-22;P8 首刀)
//
// 表布局 = `awq_build::rtn_i8_row_sym` 的产物:
//   q  [vocab, D] int8(行主;每行独立 scale)
//   s  [vocab]    f32 (s_o = row_absmax / 127; 0 行 = 1)
// 语义:out[t, d] = f16(q[id_t, d] * s[id_t])
//
// 为什么不在 host 反量化:`embed_tokens` 2,425 MiB(248k×5120 f16),
// int8 后 1,212 MiB;查表核直接读 int8 + scale,零第二份表。
//
// 向量支:每线程一次 8 个 int8(8B 载入)→ 4 个 half2(16B 写出)。
// 前提 `D % 8 == 0`(27B D=5120 ✓);否则标量支。
// grid = (tokens,1,1),block = (512,1,1)(与 `owl_embed_f16` 同几何)。
// ============================================================================
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void owl_embed_i8_f16(
    const int8_t *__restrict__ q,     // [vocab, D] int8(row-major)
    const float *__restrict__ s,      // [vocab] f32 per-row scale
    const float *__restrict__ ids,    // [tokens](f32 数值形态;kernel 内 cast)
    size_t d_dim,
    __half *__restrict__ out) {       // [tokens, D](末参 = 输出)
    const unsigned int t = blockIdx.x;
    const unsigned int id = (unsigned int)ids[t];
    const int8_t *src = q + (unsigned long long)id * d_dim;
    __half *dst = out + (unsigned long long)t * d_dim;
    const float scale = s[id];
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    if ((d_dim & 7) == 0) {
        for (size_t i = (size_t)tid * 8; i < d_dim; i += (size_t)stride * 8) {
            // 8B 载入(行基址按 8 对齐:D % 8 == 0)
            const int2 raw = *reinterpret_cast<const int2 *>(src + i);
            const int8_t *b = reinterpret_cast<const int8_t *>(&raw);
            __half2 o[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                o[j] = __floats2half2_rn((float)b[2 * j] * scale,
                                         (float)b[2 * j + 1] * scale);
            }
            // 16B 写出(输出行基址按 16 对齐:D*2 % 16 == 0)
            *reinterpret_cast<uint4 *>(dst + i) =
                *reinterpret_cast<const uint4 *>(o);
        }
    } else {
        for (size_t i = tid; i < d_dim; i += stride) {
            dst[i] = __float2half((float)src[i] * scale);
        }
    }
}
