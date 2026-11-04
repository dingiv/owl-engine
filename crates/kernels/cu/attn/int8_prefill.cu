// owl native INT8 paged prefill (K int8 + V token-major int8 + P uint8), no V shadow.
// 4-warp / Bc=32 / Br=64 / D=256. Ported from the validated fi-int8 experiment.
//
// 2026-11-03 质量复测:此前 P8/P24 的 32k 失败根因是 norm_rope 归约/K scale/
// V scale 的旧 bug,不是 P 精度。修复后 P8 在 27B 32k/64k/100k needle 全过,
// 因此生产 PV 回到单平面 P8(1 个 mma / 32 keys),不再付 P24/P32 的 3~4 倍税。
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
constexpr int VBytes = 8 * 1024;
constexpr int KVBytes = KBytes + VBytes;  // 单 K 缓冲 + V
constexpr int PBytes = Br * (Bc / 2) * 2;
constexpr int ScaleBytes = 8 * Bc * sizeof(float);
constexpr int VSBytes = Bc * sizeof(float);
// token-major V 的 8KB staging:cp.async 先按 [token][dim] 连续读,
// 再在 shared 内转成 PV 所需的 dim-major b16。池布局只此一处换序。
constexpr int VTBytes = Bc * D;
constexpr int SmemBytes = QBytes + KVBytes + PBytes + ScaleBytes + VSBytes + VTBytes;

__device__ __forceinline__ int swz(int row, int col) { return (((col >> 3) ^ (row & 7)) << 3) | (col & 7); }
__device__ __forceinline__ int swz_p(int row, int col) { return (((col >> 3) ^ (row & 1)) << 3) | (col & 7); }
__device__ __forceinline__ int swz_v(int row, int col) { return (((col >> 3) ^ (row & 1)) << 3) | (col & 7); }
__device__ __forceinline__ uint32_t p_code8(float p, float inv) {
    uint32_t r;
    asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(r) : "f"(p * inv));
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
                              const int8_t* __restrict__ v,
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
    int8_t* ks0 = smem + QBytes;
    uint16_t* v_b16 = reinterpret_cast<uint16_t*>(smem + QBytes + KBytes);
    uint16_t* p_b16 = reinterpret_cast<uint16_t*>(smem + QBytes + KVBytes);
    float* kscale0 = reinterpret_cast<float*>(smem + QBytes + KVBytes + PBytes);
    float* vscale_s = reinterpret_cast<float*>(reinterpret_cast<int8_t*>(kscale0) + ScaleBytes);
    int8_t* s_vt = reinterpret_cast<int8_t*>(smem + QBytes + KVBytes + PBytes +
                                            ScaleBytes + VSBytes);
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
    for (int i = tid; i < Bc; i += Threads) {
        const int phys = static_cast<int>(block_table[0]);
        vscale_s[i] = (i < L)
                          ? vscale[((size_t)(phys * page + i) * Hkv + kv_head)]
                          : 1.0f;
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

    float acc[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n)
#pragma unroll
        for (int j = 0; j < 4; ++j) acc[n][j] = 0.0f;

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int tile_k0 = kb * Bc;
        __half* k_b16 = reinterpret_cast<__half*>(smem + QBytes);
        float* kscale_s = kscale0;
        // V 预取,与 QK 重叠;K 单缓冲在 QK 完成后由下一轮 prefetch 覆盖。
        // token-major 源:[key, Hkv, D],每个 key 的 D 字节连续。
        const int v_page = static_cast<int>(block_table[tile_k0 / page]);
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
            const int key = chunk / (D / 16);
            const int off = (chunk % (D / 16)) * 16;
            const int8_t* src = v + ((size_t)(v_page * page + key) * Hkv + kv_head) * D + off;
            int8_t* dst = &s_vt[key * D + off];
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
        float pmaxv0 = 0.0f, pmaxv1 = 0.0f;
        float inv0 = 0.0f, inv1 = 0.0f;
        float ps0 = (1.0f / 255.0f), ps1 = (1.0f / 255.0f);
        const float nm0s = nm0 * scale_l2;
        const float nm1s = nm1 * scale_l2;
        float sum0 = 0.0f, sum1 = 0.0f;
        float pv[4][4];
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
            pv[nt][0] = p00 * vscale_s[col0];
            pv[nt][1] = p01 * vscale_s[col0 + 1];
            pv[nt][2] = p10 * vscale_s[col0];
            pv[nt][3] = p11 * vscale_s[col0 + 1];
            pmaxv0 = fmaxf(pmaxv0, fmaxf(pv[nt][0], pv[nt][1]));
            pmaxv1 = fmaxf(pmaxv1, fmaxf(pv[nt][2], pv[nt][3]));
        }
        pmaxv0 = warp_max<4>(pmaxv0, 0xffffffffu);
        pmaxv1 = warp_max<4>(pmaxv1, 0xffffffffu);
        ps0 = pmaxv0 > 0.0f ? pmaxv0 * (1.0f / 255.0f) : (1.0f / 255.0f);
        ps1 = pmaxv1 > 0.0f ? pmaxv1 * (1.0f / 255.0f) : (1.0f / 255.0f);
        inv0 = pmaxv0 > 0.0f ? 255.0f / pmaxv0 : 0.0f;
        inv1 = pmaxv1 > 0.0f ? 255.0f / pmaxv1 : 0.0f;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const int col0 = nt * 8 + 2 * lid;
            const uint32_t c00 = p_code8(pv[nt][0], inv0);
            const uint32_t c01 = p_code8(pv[nt][1], inv0);
            const uint32_t c10 = p_code8(pv[nt][2], inv1);
            const uint32_t c11 = p_code8(pv[nt][3], inv1);
            const int p0 = row0 * (Bc / 2) + swz_p(row0, col0 / 2);
            const int p1 = row1 * (Bc / 2) + swz_p(row1, col0 / 2);
            p_b16[p0] = static_cast<uint16_t>((c00 & 0xff) | ((c01 & 0xff) << 8));
            p_b16[p1] = static_cast<uint16_t>((c10 & 0xff) | ((c11 & 0xff) << 8));
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
            for (int i = tid; i < Bc; i += Threads) {
                const int phys = static_cast<int>(block_table[nk]);
                const int key = nk * Bc + i;
                vscale_s[i] = (key < L)
                                  ? vscale[((size_t)(phys * page + i) * Hkv + kv_head)]
                                  : 1.0f;
            }
            cp_commit();
        }

        // s_vt[key][d] -> v_b16[d][pair](pair = key/2,低/高字节 = key&1)。
        // 仍走 P8 的 u8s8 PV,不改变 P 量化/softmax 数值路径。
        for (int d = tid; d < D; d += Threads) {
            // thread 独占一个 dim:32 个 key 行各读 1B(跨 lane 为连续 d),
            // 16 个 pair 连续存;比“1 word/线程、16 线程读同一字节”少 8x store。
            uint16_t w[16];
#pragma unroll
            for (int p = 0; p < 16; ++p) {
                const int k0 = 2 * p;
                const uint32_t b0 = static_cast<uint8_t>(s_vt[k0 * D + d]);
                const uint32_t b1 = static_cast<uint8_t>(s_vt[(k0 + 1) * D + d]);
                w[p] = static_cast<uint16_t>(b0 | (b1 << 8));
            }
#pragma unroll
            for (int g = 0; g < 2; ++g) {
                uint4 v;
                v.x = static_cast<uint32_t>(w[g * 8]) |
                      (static_cast<uint32_t>(w[g * 8 + 1]) << 16);
                v.y = static_cast<uint32_t>(w[g * 8 + 2]) |
                      (static_cast<uint32_t>(w[g * 8 + 3]) << 16);
                v.z = static_cast<uint32_t>(w[g * 8 + 4]) |
                      (static_cast<uint32_t>(w[g * 8 + 5]) << 16);
                v.w = static_cast<uint32_t>(w[g * 8 + 6]) |
                      (static_cast<uint32_t>(w[g * 8 + 7]) << 16);
                *reinterpret_cast<uint4*>(&v_b16[d * (Bc / 2) + swz_v(d, g * 8)]) = v;
            }
        }
        __syncthreads();

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
            acc[n][0] = __fmaf_rn(static_cast<float>(c0), ps0, acc[n][0]);
            acc[n][1] = __fmaf_rn(static_cast<float>(c1), ps0, acc[n][1]);
            acc[n][2] = __fmaf_rn(static_cast<float>(c2), ps1, acc[n][2]);
            acc[n][3] = __fmaf_rn(static_cast<float>(c3), ps1, acc[n][3]);
        }
        if (kb + 1 < key_blocks) cp_wait<0>();
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
