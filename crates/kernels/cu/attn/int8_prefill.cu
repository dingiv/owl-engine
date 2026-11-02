// owl native INT8 paged prefill (K int8 + V int8 prepacked + P uint8), no V shadow.
// 4-warp / Bc=32 / Br=64 / D=256. Ported from the validated fi-int8 experiment.
#include "int8_attn_helpers.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

using namespace owl_int8_attn;

namespace {
constexpr int D = 256;
constexpr int Br = 64;
constexpr int Bc = 32;
constexpr int DB16 = D / 2;
constexpr int Threads = 128;
constexpr int QBytes = Br * D;
constexpr int KVBytes = 16 * 1024;
constexpr int PBytes = Br * (Bc / 2) * 2;
constexpr int ScaleBytes = Bc * sizeof(float);
constexpr int SmemBytes = QBytes + KVBytes + PBytes + ScaleBytes;

__device__ __forceinline__ int swz(int row, int col) { return (((col >> 3) ^ (row & 7)) << 3) | (col & 7); }
__device__ __forceinline__ int swz_p(int row, int col) { return (((col >> 3) ^ (row & 1)) << 3) | (col & 7); }
__device__ __forceinline__ int swz_v(int row, int col) { return (((col >> 3) ^ (row & 1)) << 3) | (col & 7); }
__device__ __forceinline__ uint16_t p_code8(float p, float inv) {
    uint16_t r;
    asm("cvt.rni.sat.u8.f32 %0, %1;" : "=h"(r) : "f"(p * inv));
    return r;
}
__device__ __forceinline__ float ex2f_fast(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
}  // namespace

extern "C" __global__ void owl_int8_prefill_paged(const int8_t* __restrict__ q,
                              const float* __restrict__ qscale,
                              const int8_t* __restrict__ k,
                              const float* __restrict__ kscale,
                              const uint16_t* __restrict__ vp,
                              const float* __restrict__ vscale,
                              const float* __restrict__ block_table,
                              __half* __restrict__ out, int T, int L, int Hq, int Hkv,
                              int page, int base_pos,
                              const __half* __restrict__ dep_k,
                              const __half* __restrict__ dep_v) {
    (void)dep_k;
    (void)dep_v;
    extern __shared__ __align__(16) int8_t smem[];
    int8_t* qs = smem;
    int8_t* ks = smem + QBytes;
    uint16_t* v_b16 = reinterpret_cast<uint16_t*>(smem + QBytes + 8 * 1024);
    uint16_t* p_b16 = reinterpret_cast<uint16_t*>(smem + QBytes + KVBytes);
    float* kscale_s = reinterpret_cast<float*>(smem + QBytes + KVBytes + PBytes);
    __half* q_b16 = reinterpret_cast<__half*>(qs);
    __half* k_b16 = reinterpret_cast<__half*>(ks);

    const int q_block = blockIdx.x;
    const int q_head = blockIdx.y;
    const int q0 = q_block * Br;
    const int kv_head = q_head / (Hq / Hkv);
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (q0 >= T) return;

    for (int chunk = tid; chunk < Br * (D / 16); chunk += Threads) {
        const int row = chunk / (D / 16);
        const int d = (chunk % (D / 16)) * 16;
        if (q0 + row >= T) continue;
        cp_async<16, Cache::cg>(&qs[row * D + swz(row, d / 2) * 2],
                                &q[((size_t)(q0 + row) * Hq + q_head) * D + d]);
    }
    cp_commit(); cp_wait<0>(); __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin = lane & 7;
    const int b_koff = ((lane >> 3) & 1) << 3;
    const int row_base = warp * 16;
    const float scale_l2 = 1.4426950408889634f;

    const int max_query_abs = base_pos + q0 + Br - 1;
    const int key_blocks = (max_query_abs < L ? max_query_abs : L - 1) / Bc + 1;

    const int row0 = row_base + gid;
    const int row1 = row0 + 8;
    const int tile_rows = min(Br, T - q0);
    const float qs0 = (row0 < tile_rows) ? qscale[(size_t)(q0 + row0) * Hq + q_head] : 0.0f;
    const float qs1 = (row1 < tile_rows) ? qscale[(size_t)(q0 + row1) * Hq + q_head] : 0.0f;
    const int qabs0 = (row0 < tile_rows) ? base_pos + q0 + row0 : -1;
    const int qabs1 = (row1 < tile_rows) ? base_pos + q0 + row1 : -1;
    float run_m0 = -CUDART_INF_F, run_m1 = -CUDART_INF_F;
    float run_l0 = 0.0f, run_l1 = 0.0f;

    float acc[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[n][j] = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int tile_k0 = kb * Bc;
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key_l = chunk / (D / 16);
            const int d = (chunk % (D / 16)) * 16;
            const int key = tile_k0 + key_l;
            int8_t* dst = &ks[key_l * D + swz(key_l, d / 2) * 2];
            if (key < L) {
                const int page_idx = static_cast<int>(block_table[tile_k0 / page]);
                cp_async<16, Cache::cg>(
                    dst, &k[((size_t)page_idx * page + key_l) * Hkv * D +
                            (size_t)kv_head * D + d]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        if (tid < Bc) {
            const int key = tile_k0 + tid;
            kscale_s[tid] = (key < L) ? kscale[(size_t)key * Hkv + kv_head] : 0.0f;
        }
        cp_commit(); cp_wait<0>(); __syncthreads();

        float score[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int j = 0; j < 4; ++j) score[nt][j] = 0.0f;

#pragma unroll
        for (int grp = 0; grp < 4; ++grp) {
            unsigned af[2][4];
#pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                const int kk_abs = grp * 2 + kk;
                const int acol = kk_abs * 16 + a_coloff;
                ldmatrix_x4(af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                            smem_addr(&q_b16[(row_base + a_rowoff) * DB16 +
                                             swz(row_base + a_rowoff, acol)]));
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int kk_abs = grp * 2 + kk;
                    const int brow = nt * 8 + b_rin;
                    const int bcol = kk_abs * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + swz(brow, bcol)]));
                    mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                           bf[0], bf[1]);
                }
                score[nt][0] += static_cast<float>(c0);
                score[nt][1] += static_cast<float>(c1);
                score[nt][2] += static_cast<float>(c2);
                score[nt][3] += static_cast<float>(c3);
            }
        }

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int key0 = tile_k0 + nt * 8 + 2 * lid;
            const int key1 = key0 + 1;
            const float ks0 = kscale_s[nt * 8 + 2 * lid];
            const float ks1 = kscale_s[nt * 8 + 2 * lid + 1];
            score[nt][0] *= qs0 * ks0;
            score[nt][1] *= qs0 * ks1;
            score[nt][2] *= qs1 * ks0;
            score[nt][3] *= qs1 * ks1;
            if (key0 > qabs0) score[nt][0] = -CUDART_INF_F;
            if (key1 > qabs0) score[nt][1] = -CUDART_INF_F;
            if (key0 > qabs1) score[nt][2] = -CUDART_INF_F;
            if (key1 > qabs1) score[nt][3] = -CUDART_INF_F;
            bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
            bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
        }
        bm0 = warp_max<4>(bm0, 0xffffffffu);
        bm1 = warp_max<4>(bm1, 0xffffffffu);

        const float nm0 = fmaxf(run_m0, bm0);
        const float nm1 = fmaxf(run_m1, bm1);
        const float alpha0 = (run_m0 == -CUDART_INF_F)
                                 ? 0.0f
                                 : ex2f_fast(__fmaf_rn(run_m0, scale_l2, -nm0 * scale_l2));
        const float alpha1 = (run_m1 == -CUDART_INF_F)
                                 ? 0.0f
                                 : ex2f_fast(__fmaf_rn(run_m1, scale_l2, -nm1 * scale_l2));
        const float pmax0 = bm0 > -CUDART_INF_F
                                ? ex2f_fast(__fmaf_rn(bm0, scale_l2, -nm0 * scale_l2)) : 0.0f;
        const float pmax1 = bm1 > -CUDART_INF_F
                                ? ex2f_fast(__fmaf_rn(bm1, scale_l2, -nm1 * scale_l2)) : 0.0f;
        const float ps0 = pmax0 > 0.0f ? pmax0 * (1.0f / 255.0f) : (1.0f / 255.0f);
        const float ps1 = pmax1 > 0.0f ? pmax1 * (1.0f / 255.0f) : (1.0f / 255.0f);
        const float inv0 = pmax0 > 0.0f ? 255.0f / pmax0 : 0.0f;
        const float inv1 = pmax1 > 0.0f ? 255.0f / pmax1 : 0.0f;
        const float nm0s = nm0 * scale_l2;
        const float nm1s = nm1 * scale_l2;
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float p00 = score[nt][0] > -CUDART_INF_F
                                  ? ex2f_fast(__fmaf_rn(score[nt][0], scale_l2, -nm0s)) : 0.0f;
            const float p01 = score[nt][1] > -CUDART_INF_F
                                  ? ex2f_fast(__fmaf_rn(score[nt][1], scale_l2, -nm0s)) : 0.0f;
            const float p10 = score[nt][2] > -CUDART_INF_F
                                  ? ex2f_fast(__fmaf_rn(score[nt][2], scale_l2, -nm1s)) : 0.0f;
            const float p11 = score[nt][3] > -CUDART_INF_F
                                  ? ex2f_fast(__fmaf_rn(score[nt][3], scale_l2, -nm1s)) : 0.0f;
            sum0 += p00 + p01;
            sum1 += p10 + p11;
            const int col0 = nt * 8 + 2 * lid;
            const uint16_t c00 = p_code8(p00, inv0);
            const uint16_t c01 = p_code8(p01, inv0);
            const uint16_t c10 = p_code8(p10, inv1);
            const uint16_t c11 = p_code8(p11, inv1);
            p_b16[row0 * (Bc / 2) + swz_p(row0, col0 / 2)] = c00 | (c01 << 8);
            p_b16[row1 * (Bc / 2) + swz_p(row1, col0 / 2)] = c10 | (c11 << 8);
        }
        sum0 = warp_sum<4>(sum0, 0xffffffffu);
        sum1 = warp_sum<4>(sum1, 0xffffffffu);
        run_l0 = __fmaf_rn(run_l0, alpha0, sum0);
        run_l1 = __fmaf_rn(run_l1, alpha1, sum1);
        run_m0 = nm0;
        run_m1 = nm1;
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        __syncthreads();  // all QK done; K is free, V can overwrite

        const int v_page = static_cast<int>(block_table[tile_k0 / page]);
        // V int8 prepacked [Hkv][D][L/2] b16; Bc/2 = 16 b16 per dim.
        for (int chunk = tid; chunk < D * (Bc / 2 / 8); chunk += Threads) {
            const int d = chunk / (Bc / 2 / 8);
            const int pair0 = (chunk % (Bc / 2 / 8)) * 8;
            const uint16_t* src = &vp[((size_t)v_page * Hkv + kv_head) * D *
                                          (page / 2) +
                                      (size_t)d * (page / 2) + pair0];
            uint16_t* dst = &v_b16[d * (Bc / 2) + swz_v(d, pair0)];
            cp_async<16, Cache::cg>(dst, src);
        }
        cp_commit(); cp_wait<0>(); __syncthreads();

        uint32_t a[4];
        ldmatrix_x4(a[0], a[1], a[2], a[3],
                    smem_addr(&p_b16[(row_base + a_rowoff) * (Bc / 2) +
                                     swz_p(row_base + a_rowoff, a_coloff)]));
#pragma unroll
        for (int n = 0; n < 32; ++n) {
            uint32_t b[2];
            const int brow = n * 8 + b_rin;
            ldmatrix_x2(b[0], b[1],
                        smem_addr(&v_b16[brow * (Bc / 2) + swz_v(brow, b_koff)]));
            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
            mma_u8s8(c0, c1, c2, c3, a[0], a[1], a[2], a[3], b[0], b[1]);
            const float vv = vscale[(size_t)v_page * Hkv * 4 + kv_head * 4 + (n * 8) / 64];
            const float s0 = ps0 * vv;
            const float s1 = ps1 * vv;
            acc[n][0] += static_cast<float>(c0) * s0;
            acc[n][1] += static_cast<float>(c1) * s0;
            acc[n][2] += static_cast<float>(c2) * s1;
            acc[n][3] += static_cast<float>(c3) * s1;
        }
        __syncthreads();
    }

    const float inv_l0 = run_l0 > 0.0f ? 1.0f / run_l0 : 0.0f;
    const float inv_l1 = run_l1 > 0.0f ? 1.0f / run_l1 : 0.0f;
#pragma unroll
    for (int n = 0; n < 32; ++n) {
        const int d0 = n * 8 + 2 * lid;
        if (row0 < tile_rows) {
            __half* p0 = &out[((size_t)(q0 + row0) * Hq + q_head) * D + d0];
            *reinterpret_cast<__half2*>(p0) = __floats2half2_rn(
                acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            __half* p1 = &out[((size_t)(q0 + row1) * Hq + q_head) * D + d0];
            *reinterpret_cast<__half2*>(p1) = __floats2half2_rn(
                acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
}
