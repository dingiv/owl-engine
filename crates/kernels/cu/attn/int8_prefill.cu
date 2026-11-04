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
// ---------------------------------------------------------------------------
// staging(K/scales 装载)的**块不变量**
//
// K/scale 两处装载都是"线程 t 固定负责若干 chunk、跨块只换页基址",
// 同一线程的各 chunk 只差**常数步长**(key_l += 8 / d += 64 / g += 4),
// 并且 swizzle 里用到的低位(key_l&7 / d&1)在块间、chunk 间都不变。原实现把
// `chunk/16`、`i/Bc` 这类**有符号除法**(SASS 上是 SHF+LEA.HI+LOP3 四条一套)
// 与 swizzle 每 chunk 重算一遍,还把 `block_table[nk]` 每 chunk 重载一次
// (LDG + F2I)—— 每块 ~230 条整数指令(占循环体 12%)。
// 这里把地址数学整体提出块循环:块内只剩 1 次页号读取 + 常数步长加法。
// ---------------------------------------------------------------------------
struct StageGeom {
    // 每线程迭代数
    int k_iters;
    int s_iters;
    // 迭代步长(= 跨线程覆盖同一 tile 的"跳跃")
    int k_step;
    int s_step;
    // 本线程首个元素
    int k_smem0;
    int k_off0;
    int k_key0;
    int s_smem0;
    int s_off0;
    int s_key;
};

__device__ __forceinline__ StageGeom stage_geom(int tid, int Hkv, int page) {
    StageGeom g;
    // 线程 t 负责 chunk = t + Threads*j；chunk → (key_l, d) 是 chunk/16, chunk%16
    // ⇒ key_l = t/16 + (Threads/16)j,d 不变;scale 侧 g = t/32 +
    // (Threads/32)j,key 不变。**步长按 Threads 算,不是按迭代数。**
    constexpr int KIters = (Bc * (D / 16)) / Threads;
    constexpr int SIters = (8 * Bc) / Threads;
    constexpr int KStep = Threads / (D / 16);
    constexpr int SStep = Threads / Bc;
    g.k_iters = KIters;
    g.s_iters = SIters;
    g.k_step = KStep;
    g.s_step = SStep;
    const int k_key_l0 = tid / (D / 16);
    const int k_d = (tid % (D / 16)) * 16;
    g.k_key0 = k_key_l0;
    g.k_smem0 = k_key_l0 * D + swz(k_key_l0, k_d / 2) * 2;
    g.k_off0 = k_key_l0 * Hkv * D + k_d;
    g.s_key = tid % Bc;
    g.s_smem0 = tid;
    g.s_off0 = (tid % Bc) * Hkv * 8 + (tid / Bc);
    return g;
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
    // 本核 K 预取按 `block_table[nk]`(每块一页)寻址 ⇒ **要求 page == Bc**。
    // 形状族已强约束(page=32),这里显式拒一次,不留"两种寻址并存"的暗门。
    if (page != Bc) return;

    for (int chunk = tid; chunk < Br * (D / 16); chunk += Threads) {
        const int row = chunk / (D / 16);
        const int d = (chunk % (D / 16)) * 16;
        if (q0 + row >= T) continue;
        cp_async<16, Cache::cg>(&qs[row * D + swz(row, d / 2) * 2],
                                &q[((size_t)(q0 + row) * Hq + q_head) * D + d]);
    }
    const StageGeom sg = stage_geom(tid, Hkv, page);
    // Prologue K0 + scales(块不变量 + 常数步长,见 StageGeom)。
    {
        const int phys0 = static_cast<int>(block_table[0]);
        const int8_t* kpage = k + (size_t)phys0 * page * Hkv * D + (size_t)kv_head * D;
#pragma unroll
        for (int j = 0; j < sg.k_iters; ++j) {
            int8_t* dst = ks0 + sg.k_smem0 + j * (sg.k_step * D);
            const int key = sg.k_key0 + j * sg.k_step;
            if (key < L) {
                cp_async<16, Cache::cg>(dst, kpage + sg.k_off0 + j * (sg.k_step * Hkv * D));
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        const float* sbase =
            kscale + (size_t)phys0 * page * Hkv * 8 + (size_t)kv_head * 8;
#pragma unroll
        for (int j = 0; j < sg.s_iters; ++j) {
            // 无条件读(页已分配,槽必在池内)+ FSEL:免掉 BSSY/条件 LDG 分支
            const float raw = sbase[sg.s_off0 + j * sg.s_step];
            kscale0[sg.s_smem0 + j * (sg.s_step * Bc)] = (sg.s_key < L) ? raw : 0.0f;
        }
        if (tid < Bc) {
            const float raw = vscale[(size_t)(phys0 * page + tid) * Hkv + kv_head];
            vscale_s[tid] = (tid < L) ? raw : 1.0f;
        }
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
        {
            const int v_page = static_cast<int>(block_table[kb]);  // page==Bc ⇒ 块号即页号
            for (int chunk = tid; chunk < Bc * (D / 16); chunk += Threads) {
                const int key = chunk / (D / 16);
                const int off = (chunk % (D / 16)) * 16;
                const int8_t* src =
                    v + ((size_t)(v_page * page + key) * Hkv + kv_head) * D + off;
                int8_t* dst = &s_vt[key * D + off];
                cp_async<16, Cache::cg>(dst, src);
            }
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
        // 因果掩码只在**对角块**做:块尾 key ≤ 本 tile 最小 query 绝对位置时,
        // 掩码对所有行恒不触发(严格因果)。`key_blocks` 是按**最大** query 圈定的,
        // 所以早行所在的块仍需要对角逐块掩码 —— 判据必须用 min(base_pos+q0)。
        // 越界行(row>=tile_rows)结果不写出 ⇒ 它们不需要掩码。
        // 指令账(issue-bound 核):非对角块每块省 ~40 条 ISETP/FSEL。
        const bool tail_mask = (tile_k0 + Bc - 1) > (base_pos + q0);
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            if (tail_mask) {
                const int key0 = tile_k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (key0 > qabs0) score[nt][0] = -CUDART_INF_F;
                if (key1 > qabs0) score[nt][1] = -CUDART_INF_F;
                if (key0 > qabs1) score[nt][2] = -CUDART_INF_F;
                if (key1 > qabs1) score[nt][3] = -CUDART_INF_F;
            }
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
            const int phys = static_cast<int>(block_table[nk]);   // 每块只读一次
            int8_t* nks = smem + QBytes;
            const int8_t* kpage = k + (size_t)phys * page * Hkv * D + (size_t)kv_head * D;
            // 整块都在界内(常态,>99%)走无分支快路径;只有尾块逐 chunk 判。
            if (nk * Bc + Bc <= L) {
#pragma unroll
                for (int j = 0; j < sg.k_iters; ++j) {
                    cp_async<16, Cache::cg>(nks + sg.k_smem0 + j * (sg.k_step * D),
                                            kpage + sg.k_off0 + j * (sg.k_step * Hkv * D));
                }
            } else {
#pragma unroll
                for (int j = 0; j < sg.k_iters; ++j) {
                    int8_t* dst = nks + sg.k_smem0 + j * (sg.k_step * D);
                    const int key = sg.k_key0 + j * sg.k_step;
                    if (nk * Bc + key < L) {
                        cp_async<16, Cache::cg>(dst,
                                                kpage + sg.k_off0 + j * (sg.k_step * Hkv * D));
                    } else {
                        store_vec(dst, make_int4(0, 0, 0, 0));
                    }
                }
            }
            const float* sbase =
                kscale + (size_t)phys * page * Hkv * 8 + (size_t)kv_head * 8;
#pragma unroll
            for (int j = 0; j < sg.s_iters; ++j) {
                const float raw = sbase[sg.s_off0 + j * sg.s_step];
                kscale0[sg.s_smem0 + j * (sg.s_step * Bc)] =
                    (nk * Bc + sg.s_key < L) ? raw : 0.0f;
            }
            if (tid < Bc) {
                const float raw = vscale[(size_t)(phys * page + tid) * Hkv + kv_head];
                vscale_s[tid] = (nk * Bc + tid < L) ? raw : 1.0f;
            }
            cp_commit();
        }

        // s_vt[key][d](token-major int8 staging)→ v_b16[d][pair](2 keys/16bit),
        // 供 P8 的 u8s8 PV 使用(不改 P 量化/softmax 数值路径)。
        //
        // **4 维 × 8 key 一块**:每 key 一次 LDS.32(4 个连续 dim)+ 6 条 PRMT
        // 拼 2 个 32 位字 + 1 条 STS.64。原实现按 dim 做 32 条**字节** LDS
        // (64 条/线程)+ 16 个 uint16 临时量(压出 STACK 96 溢出);本版
        // 16 条 LDS + 48 条 PRMT,零临时数组。
        {
            constexpr int Units = (Bc / 8) * (D / 4);   // 4 组 key-pair × 64 组 dim
            constexpr int Per = Units / Threads;        // 2
            static_assert(Units % Threads == 0);
#pragma unroll
            for (int u0 = 0; u0 < Per; ++u0) {
                const int u = tid + u0 * Threads;
                const int kq = u & 3;                   // 4 个 key-pair 一组(key 8 个)
                const int dq = u >> 2;                  // 4 个 dim 一组
                const int k0 = kq * 8;
                const int d0 = dq * 4;
                uint32_t b[8];
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    b[j] = *reinterpret_cast<const uint32_t*>(&s_vt[(k0 + j) * D + d0]);
#pragma unroll
                for (int dd = 0; dd < 4; ++dd) {
                    // sel:byte0 = x[dd], byte1 = y[dd](高两字节不用)
                    const uint32_t sel = static_cast<uint32_t>(dd) |
                                         (static_cast<uint32_t>(dd + 4) << 4);
                    const uint32_t w0 = __byte_perm(b[0], b[1], sel);
                    const uint32_t w1 = __byte_perm(b[2], b[3], sel);
                    const uint32_t w2 = __byte_perm(b[4], b[5], sel);
                    const uint32_t w3 = __byte_perm(b[6], b[7], sel);
                    // lo = 4 字节 = pair(k0,k0+1) + pair(k0+2,k0+3)
                    const uint32_t lo = __byte_perm(w0, w1, 0x5410);
                    const uint32_t hi = __byte_perm(w2, w3, 0x5410);
                    const int d = d0 + dd;
                    uint32_t* dst = reinterpret_cast<uint32_t*>(
                        &v_b16[d * (Bc / 2) + swz_v(d, kq * 4)]);
                    dst[0] = lo;
                    dst[1] = hi;
                }
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
