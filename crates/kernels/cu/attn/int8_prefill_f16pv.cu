// RESEARCH ONLY — not the production path.
// owl INT8 QK + f16 PV probe. 4-warp / Bc=32 / Br=64 / D=256.
//
// 契约:v16 必须是**已反量化的 V shadow**(f16(code * vscale)),本核不再乘
// vscale。生产 P8 核(`int8_prefill.cu`)读 V int8 packed + vscale,不要把这个
// 研究核直接接进池;要接必须先补 vscale 参数或改 writer 契约。
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
// 标准 attention logits = q·k / sqrt(D);D=256 ⇒ 1/16。
constexpr float AttnScale = 1.0f / 16.0f;
constexpr int QBytes = Br * D;
constexpr int KBytes = 8 * 1024;
constexpr int VBytes = 16 * 1024;  // V f16 shadow (token-major)
constexpr int KVBytes = KBytes + VBytes;  // 单 K 缓冲 + V f16
constexpr int PBytes = Br * Bc * sizeof(__half);  // P f16, 16 key x 16 row
constexpr int ScaleBytes = 8 * Bc * sizeof(float);  // K per-32 scale
constexpr int SmemBytes = QBytes + KVBytes + PBytes + ScaleBytes;
constexpr float kP15 = 32768.0f;

__device__ __forceinline__ int swz(int row, int col) { return (((col >> 3) ^ (row & 7)) << 3) | (col & 7); }
__device__ __forceinline__ int swz_p(int row, int col) { return (((col >> 3) ^ (row & 1)) << 3) | (col & 7); }
__device__ __forceinline__ int swz_v(int row, int col) { return (((col >> 3) ^ (row & 1)) << 3) | (col & 7); }
__device__ __forceinline__ int swz32(int row, int col) { return (((col >> 3) ^ (row & 3)) << 3) | (col & 7); }
__device__ __forceinline__ __half2 u2h(uint32_t x) { return *reinterpret_cast<__half2*>(&x); }
__device__ __forceinline__ uint32_t h2u(__half2 x) { return *reinterpret_cast<uint32_t*>(&x); }
__device__ __forceinline__ void mma_f16_f32(float& d0, float& d1, float& d2, float& d3,
                                            uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                            uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t p_code32(float p, float inv) {
    uint32_t r;
    asm("cvt.rni.sat.u32.f32 %0, %1;" : "=r"(r) : "f"(p * inv));
    return r;
}
__device__ __forceinline__ float ex2f_fast(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
}  // namespace

extern "C" __global__ void owl_int8_prefill_paged_f16pv(const int8_t* __restrict__ q,
                              const float* __restrict__ qscale,
                              const int8_t* __restrict__ k,
                              const float* __restrict__ kscale,
                              const __half* __restrict__ v16,
                              const float* __restrict__ block_table,
                              __half* __restrict__ out, int T, int L, int Hq, int Hkv,
                              int page, int base_pos,
                              const __half* __restrict__ dep_k,
                              const __half* __restrict__ dep_v) {
    (void)dep_k;
    (void)dep_v;
    extern __shared__ __align__(16) int8_t smem[];
    int8_t* qs = smem;
    int8_t* ks0 = smem + QBytes;
    __half* vf = reinterpret_cast<__half*>(smem + QBytes + KBytes);
    __half* ph = reinterpret_cast<__half*>(smem + QBytes + KVBytes);
    float* kscale0 = reinterpret_cast<float*>(smem + QBytes + KVBytes + PBytes);
    __half* q_b16 = reinterpret_cast<__half*>(qs);

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
    // Prologue K0 into buffer 0.
    for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
        const int key = chunk / (D / 16);
        const int d = (chunk % (D / 16)) * 16;
        int8_t* dst = &ks0[key * D + swz(key, d / 2) * 2];
        if (key < L) {
            const int page_idx = static_cast<int>(block_table[0]);
            cp_async<16, Cache::cg>(dst, &k[((size_t)page_idx * page + key) * Hkv * D +
                                           (size_t)kv_head * D + d]);
        } else {
            store_vec(dst, make_int4(0, 0, 0, 0));
        }
    }
    for (int i = tid; i < 8 * Bc; i += Threads) {
        const int g = i / Bc;
        const int key = i % Bc;
        const int phys = static_cast<int>(block_table[0]);
        kscale0[i] = (key < L)
                         ? kscale[((size_t)(phys * page + key) * Hkv + kv_head) * 8 + g]
                         : 0.0f;
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
    float qsg0[8], qsg1[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        qsg0[g] = (row0 < tile_rows)
                      ? qscale[((size_t)(q0 + row0) * Hq + q_head) * 8 + g] * AttnScale
                      : 0.0f;
        qsg1[g] = (row1 < tile_rows)
                      ? qscale[((size_t)(q0 + row1) * Hq + q_head) * 8 + g] * AttnScale
                      : 0.0f;
    }
    const int qabs0 = (row0 < tile_rows) ? base_pos + q0 + row0 : -1;
    const int qabs1 = (row1 < tile_rows) ? base_pos + q0 + row1 : -1;
    float run_m0 = -CUDART_INF_F, run_m1 = -CUDART_INF_F;
    float run_l0 = 0.0f, run_l1 = 0.0f;

    float acc[16][8];
#pragma unroll
    for (int n = 0; n < 16; ++n)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[n][j] = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int tile_k0 = kb * Bc;
        __half* k_b16 = reinterpret_cast<__half*>(smem + QBytes);
        float* kscale_s = kscale0;
        // V f16 shadow 预取,与 QK 重叠;shared 为 token-major [key][D]。
        const int v_page = static_cast<int>(block_table[tile_k0 / page]);
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
            const int key0 = chunk / (D / 8);
            const int d = (chunk % (D / 8)) * 8;
            const size_t slot = (size_t)v_page * page + key0;
            const __half* src = &v16[((slot * Hkv + kv_head) * D) + d];
            __half* dst = &vf[key0 * D + swz(key0, d)];
            cp_async<16, Cache::cg>(dst, src);
        }
        cp_commit();  // V group;K[buf] already resident

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
                const int kk0 = nt * 8 + 2 * lid;
#pragma unroll
                for (int kk = 0; kk < 2; ++kk) {
                    const int kk_abs = grp * 2 + kk;
                    const int brow = nt * 8 + b_rin;
                    const int bcol = kk_abs * 16 + b_koff;
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&k_b16[brow * DB16 + swz(brow, bcol)]));
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    mma_s8(c0, c1, c2, c3, af[kk][0], af[kk][1], af[kk][2], af[kk][3],
                           bf[0], bf[1]);
                    const int g = grp * 2 + kk;
                    const float gs0 = kscale_s[g * Bc + kk0];
                    const float gs1 = kscale_s[g * Bc + kk0 + 1];
                    score[nt][0] += static_cast<float>(c0) * (qsg0[g] * gs0);
                    score[nt][1] += static_cast<float>(c1) * (qsg0[g] * gs1);
                    score[nt][2] += static_cast<float>(c2) * (qsg1[g] * gs0);
                    score[nt][3] += static_cast<float>(c3) * (qsg1[g] * gs1);
                }
            }
        }

        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int key0 = tile_k0 + nt * 8 + 2 * lid;
            const int key1 = key0 + 1;
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
            ph[row0 * Bc + swz32(row0, col0)] = __float2half(p00 * kP15);
            ph[row0 * Bc + swz32(row0, col0 + 1)] = __float2half(p01 * kP15);
            ph[row1 * Bc + swz32(row1, col0)] = __float2half(p10 * kP15);
            ph[row1 * Bc + swz32(row1, col0 + 1)] = __float2half(p11 * kP15);
        }
        sum0 = warp_sum<4>(sum0, 0xffffffffu);
        sum1 = warp_sum<4>(sum1, 0xffffffffu);
        run_l0 = __fmaf_rn(run_l0, alpha0, sum0);
        run_l1 = __fmaf_rn(run_l1, alpha1, sum1);
        run_m0 = nm0;
        run_m1 = nm1;

#pragma unroll
        for (int n = 0; n < 16; ++n) {
            acc[n][0] *= alpha0; acc[n][1] *= alpha0;
            acc[n][2] *= alpha1; acc[n][3] *= alpha1;
            acc[n][4] *= alpha0; acc[n][5] *= alpha0;
            acc[n][6] *= alpha1; acc[n][7] *= alpha1;
        }

        cp_wait<0>(); // V 预取完成
        __syncthreads();
        // K(kb+1) 预取,与 PV(kb) 重叠;写另一 K 缓冲。
        if (kb + 1 < key_blocks) {
            const int nk = kb + 1;
            int8_t* nks = smem + QBytes;
            for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
                const int key_l = chunk / (D / 16);
                const int d = (chunk % (D / 16)) * 16;
                const int key = nk * Bc + key_l;
                int8_t* dst = &nks[key_l * D + swz(key_l, d / 2) * 2];
                if (key < L) {
                    const int page_idx = static_cast<int>(block_table[nk]);
                    cp_async<16, Cache::cg>(
                        dst, &k[((size_t)page_idx * page + key_l) * Hkv * D +
                                (size_t)kv_head * D + d]);
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
            for (int i = tid; i < 8 * Bc; i += Threads) {
                const int g = i / Bc;
                const int key_l = i % Bc;
                const int key = nk * Bc + key_l;
                const int phys = static_cast<int>(block_table[nk]);
                kscale0[i] = (key < L)
                                 ? kscale[((size_t)(phys * page + key_l) * Hkv + kv_head) * 8 + g]
                                 : 0.0f;
            }
            cp_commit();
        }

#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t a[4];
            ldmatrix_x4(a[0], a[1], a[2], a[3],
                        smem_addr(&ph[(row_base + a_rowoff) * Bc +
                                      swz32(row_base + a_rowoff, kk * 16 + a_coloff)]));
#pragma unroll
            for (int n = 0; n < 16; ++n) {
                uint32_t b[4];
                const int brow = kk * 16 + a_rowoff;
                const int bcol = n * 16 + a_coloff;
                ldmatrix_x4_t(b[0], b[1], b[2], b[3],
                              smem_addr(&vf[brow * D + swz(brow, bcol)]));
                mma_f16_f32(acc[n][0], acc[n][1], acc[n][2], acc[n][3],
                            a[0], a[1], a[2], a[3], b[0], b[1]);
                mma_f16_f32(acc[n][4], acc[n][5], acc[n][6], acc[n][7],
                            a[0], a[1], a[2], a[3], b[2], b[3]);
            }
        }
        if (kb + 1 < key_blocks) cp_wait<0>();
        __syncthreads();
    }

    const float inv_l0 = run_l0 > 0.0f ? (1.0f / run_l0) / kP15 : 0.0f;
    const float inv_l1 = run_l1 > 0.0f ? (1.0f / run_l1) / kP15 : 0.0f;
#pragma unroll
    for (int n = 0; n < 16; ++n) {
        const int d0 = n * 16 + 2 * lid;
        if (row0 < tile_rows) {
            __half* p0 = &out[((size_t)(q0 + row0) * Hq + q_head) * D + d0];
            *reinterpret_cast<__half2*>(p0) =
                __floats2half2_rn(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
            *reinterpret_cast<__half2*>(p0 + 8) =
                __floats2half2_rn(acc[n][4] * inv_l0, acc[n][5] * inv_l0);
        }
        if (row1 < tile_rows) {
            __half* p1 = &out[((size_t)(q0 + row1) * Hq + q_head) * D + d0];
            *reinterpret_cast<__half2*>(p1) =
                __floats2half2_rn(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
            *reinterpret_cast<__half2*>(p1 + 8) =
                __floats2half2_rn(acc[n][6] * inv_l1, acc[n][7] * inv_l1);
        }
    }
}
