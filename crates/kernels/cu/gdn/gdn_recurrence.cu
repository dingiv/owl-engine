// ============================================================================
// GDN 核族:gated delta rule prefill varlen gqa(<T,128,8> 同体)(2026-10-13 C 轮自 cu/owl/gdn.cu 按**算子族**拆出 ——
// 一算子一 `.cu`,与 ops/gdn.rs 的声明对望)。
//
// 溯源:repos/attention.rs/src/kernels/src/gdn.cu 模板体显式展开(nvrtc 禁
// host launcher;旧世界 GDN_GATING_KERNEL 宏模式同款)。
//
// dtype 律:激活 half → float 计算 → half 写;**rec/conv state 恒 f32**
// (上游同款 + HF/vLLM/xinfer 三方先例);slots 恒 f32 数值过线(契约 5,
// 上游 i64 已适配);输出块固定末参(槽序契约 4);标量 i32 与 Arg::I32
// 严格对位(禁止 unsigned int)。数学与 f32 版逐式同源(移植不失真由
// 金标对拍背书)。
//
// ============================================================================
// ★ 2026-10-19 布局重写(`_cols2` 快路;P1.4)—— 逐位相同,1.61×
//
// 判据来源:`tools/gdn-rec-probe.cu` 的归因阶梯(`docs/arch/TP2-实录卷.md` §68)。
// 现役形(V0)1.74 µs/token 里 **81% 是布局税,不是算术**:
//
// | 项 | µs/token | 病根 |
// |---|---:|---|
// | 两次 32-lane 归约 | 0.42 | 一列一 warp ⇒ 归约开销按列重复 |
// | q/k 标量装载 | 0.45 | 块内 8 warp 各读一遍同一份 256B |
// | 单 lane 2B 写 out | 0.18 | 每 token 一条 2B,32B sector 摊薄 |
// | `__syncthreads()` | 0.24 | warp 之间**零数据依赖**,barrier 纯税 |
//
// 新形(记 `_cols2`):**一个 warp 管 2 个相邻 v 列**
//   · 归约:2 列共用一次 32-lane 归约树(每列 2.5 次 shfl vs 10);
//   · q/k:同一份 q/k 摊给 2 列(块内冗余 ÷2);**无 shared / 无 barrier**;
//   · g/beta 是**每 head 一个标量** ⇒ 每 token 只取 1 次(原来每列都取);
//   · out:2 列合成 **4B** 一条写;寄存器预取深度 2(q/k/v/g/beta 全提前)。
//
// **算术序逐位不变**:每列的 r 循环序、`__shfl_down_sync` 的 5 步树、
// fma 次序全都没动 ⇒ 判据是 `out`/`state` 与旧形**逐字节相同**
// (`tools/gdn-rec-probe.cu` 的 `check` 臂 + 生产 `gdn_*_golden` 全绿)。
// 实测(3080 单卡,nv=24/nk=8,T=1024):1.749 → **1.087 µs/token**。
//
// 形状门:**kd == 128 且 vd 为偶数** 才走快路(4 行/线 + 2 列/块的索引前提),
// 其余形状回落到原形(逐字保留,行为零变化);grid 由 `driver::gdn::
// recurrence_varlen_gqa` 按同一个谓词选(快路 `ceil(vd/16)`,原形 `ceil(vd/8)`)。
// ============================================================================

#include <cuda_fp16.h>

__device__ __forceinline__ float gdn16_to_float(__half x) { return __half2float(x); }
__device__ __forceinline__ __half gdn16_from_float(float x) { return __float2half(x); }
__device__ __forceinline__ float gdn16_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return __shfl_sync(0xffffffff, val, 0);
}

#define OWL_GDN16_WARPS_PER_BLOCK 8
/// 快路:一个 warp 管几个相邻 v 列(见文件头归因表;C=2 为实测最优:
/// C=1 块内 q/k 冗余 ×2 且 warp 数翻倍,C=4 并行 warp 数掉到 1/4 反而更慢)
#define OWL_GDN16_COLS_PER_WARP 2
/// 快路:寄存器预取深度(2/3/4 实测 1.087/1.096/1.155 ⇒ 取 2)
#define OWL_GDN16_PREFETCH 2

// ============================================================================
// 快路体(2026-10-19;逐位等价于原形)
// ============================================================================
__device__ __forceinline__ void gdn_rec_cols2(
    const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v,
    const __half *__restrict__ g, const __half *__restrict__ beta, float *__restrict__ state,
    const float *__restrict__ slots, const float *__restrict__ cu_seqlens, size_t batch,
    size_t nv, size_t nk, size_t kd, size_t vd, float q_scale, __half *__restrict__ out) {
    constexpr int C = OWL_GDN16_COLS_PER_WARP;
    constexpr int D = OWL_GDN16_PREFETCH;
    constexpr int ROWS_PER_LANE = 4;  // kd == 128 且 32 lane ⇒ 4 行/线(快路前提)
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int vcol0 = (blockIdx.x * OWL_GDN16_WARPS_PER_BLOCK + warp_id) * C;
    if (vcol0 >= (int)vd) return;
    const int seq_head = blockIdx.y;
    if (seq_head >= batch * nv) return;
    const int seq_idx = seq_head / nv;
    const int v_head_idx = seq_head % nv;
    const int k_head_idx = v_head_idx / (int)(nv / nk);
    const int slot = (int)slots[seq_idx];
    if (slot < 0) return;
    const int start = (int)cu_seqlens[seq_idx];
    const int end = (int)cu_seqlens[seq_idx + 1];
    const int seq_len = end - start;
    if (seq_len <= 0) return;

    const int ts_qk = (int)(nk * kd);
    const int ts_v = (int)(nv * vd);
    const __half *q_base = q + (size_t)start * ts_qk + (size_t)k_head_idx * kd + (size_t)lane;
    const __half *k_base = k + (size_t)start * ts_qk + (size_t)k_head_idx * kd + (size_t)lane;
    const __half *v_base = v + (size_t)start * ts_v + (size_t)v_head_idx * vd + vcol0;
    const __half *g_base = g + (size_t)start * nv + v_head_idx;
    const __half *bt_base = beta + (size_t)start * nv + v_head_idx;
    __half *out_base = out + (size_t)start * ts_v + (size_t)v_head_idx * vd + vcol0;
    float *st = state + ((size_t)slot * nv + (size_t)v_head_idx) * kd * vd + vcol0;

    // 状态分片:lane 持 4 个 k 行 × C 列(**寻址与原形逐字同**:r*32+lane 行)
    float s[ROWS_PER_LANE][C];
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) s[r][c] = st[(size_t)(r * 32 + lane) * vd + c];

    __half qr[D][ROWS_PER_LANE], kr[D][ROWS_PER_LANE], vr[D][C], gr[D], br[D];

#define OWL_GDN16_LOAD(stage, tok)                                                              \
    do {                                                                                        \
        if ((tok) < seq_len) {                                                                  \
            const size_t o = (size_t)(tok) * ts_qk;                                             \
            _Pragma("unroll") for (int r = 0; r < ROWS_PER_LANE; ++r) {                         \
                qr[stage][r] = q_base[o + (size_t)r * 32];                                      \
                kr[stage][r] = k_base[o + (size_t)r * 32];                                      \
            }                                                                                   \
            const size_t ov = (size_t)(tok) * ts_v;                                             \
            _Pragma("unroll") for (int c = 0; c < C; ++c) vr[stage][c] = v_base[ov + c];        \
            gr[stage] = g_base[(size_t)(tok) * nv];                                             \
            br[stage] = bt_base[(size_t)(tok) * nv];                                            \
        }                                                                                       \
    } while (0)

#define OWL_GDN16_COMPUTE(stage, tok)                                                           \
    do {                                                                                        \
        const float decay = expf(gdn16_to_float(gr[stage]));                                    \
        const float beta_t = gdn16_to_float(br[stage]);                                         \
        float kv[C];                                                                            \
        _Pragma("unroll") for (int c = 0; c < C; ++c) kv[c] = 0.0f;                             \
        _Pragma("unroll") for (int r = 0; r < ROWS_PER_LANE; ++r) {                             \
            const float k_v = gdn16_to_float(kr[stage][r]);                                     \
            _Pragma("unroll") for (int c = 0; c < C; ++c) {                                     \
                s[r][c] *= decay;                                                               \
                kv[c] = __fmaf_rn(s[r][c], k_v, kv[c]);                                         \
            }                                                                                   \
        }                                                                                       \
        _Pragma("unroll") for (int c = 0; c < C; ++c) kv[c] = gdn16_warp_reduce_sum(kv[c]);     \
        float delta[C], y[C];                                                                   \
        _Pragma("unroll") for (int c = 0; c < C; ++c) {                                         \
            delta[c] = (gdn16_to_float(vr[stage][c]) - kv[c]) * beta_t;                         \
            y[c] = 0.0f;                                                                        \
        }                                                                                       \
        _Pragma("unroll") for (int r = 0; r < ROWS_PER_LANE; ++r) {                             \
            const float k_v = gdn16_to_float(kr[stage][r]);                                     \
            const float q_v = gdn16_to_float(qr[stage][r]) * q_scale;                           \
            _Pragma("unroll") for (int c = 0; c < C; ++c) {                                     \
                s[r][c] = __fmaf_rn(k_v, delta[c], s[r][c]);                                    \
                y[c] = __fmaf_rn(s[r][c], q_v, y[c]);                                           \
            }                                                                                   \
        }                                                                                       \
        _Pragma("unroll") for (int c = 0; c < C; ++c) y[c] = gdn16_warp_reduce_sum(y[c]);       \
        if (lane == 0) {                                                                        \
            _Pragma("unroll") for (int c = 0; c < C; ++c)                                       \
                out_base[(size_t)(tok) * ts_v + c] = gdn16_from_float(y[c]);                    \
        }                                                                                       \
    } while (0)

#pragma unroll
    for (int d = 0; d < D; ++d) OWL_GDN16_LOAD(d, d);
    int t0 = 0;
    for (; t0 + D <= seq_len; t0 += D) {
#pragma unroll
        for (int stg = 0; stg < D; ++stg) {
            OWL_GDN16_COMPUTE(stg, t0 + stg);
            OWL_GDN16_LOAD(stg, t0 + stg + D);
        }
    }
    for (int t = t0; t < seq_len; ++t) {  // 尾 ≤ D-1 个 token(内联装载,可忽略)
        OWL_GDN16_LOAD(0, t);
        OWL_GDN16_COMPUTE(0, t);
    }
#undef OWL_GDN16_LOAD
#undef OWL_GDN16_COMPUTE

#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; ++r)
#pragma unroll
        for (int c = 0; c < C; ++c) st[(size_t)(r * 32 + lane) * vd + c] = s[r][c];
}

// ============================================================================
// 原形体(**逐字保留**;非快路形状回落用 —— 行为与 2026-10-13 版完全相同)
// grid = (ceil(vd/8), batch·HV) × (32,8);t 循环核内,q/k 双缓冲 shared,
// s_shard 每线寄存器分片(KD≤128 → 4 行/线);warp_reduce 归约。
// ============================================================================
__device__ __forceinline__ void gdn_rec_legacy(
    const __half *__restrict__ q, const __half *__restrict__ k, const __half *__restrict__ v,
    const __half *__restrict__ g, const __half *__restrict__ beta, float *__restrict__ state,
    const float *__restrict__ slots, const float *__restrict__ cu_seqlens, size_t batch,
    size_t nv, size_t nk, size_t kd, size_t vd, float q_scale, __half *__restrict__ out) {
    constexpr int BK = 128;
    constexpr int WARPS_PER_BLOCK = OWL_GDN16_WARPS_PER_BLOCK;
    constexpr int GDN_WARP_SIZE = 32;
    constexpr int ROWS_PER_LANE = (BK + GDN_WARP_SIZE - 1) / GDN_WARP_SIZE;

    const int seq_head = blockIdx.y;
    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int v_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (seq_head >= batch * nv) return;

    const int seq_idx = seq_head / nv;
    const int v_head_idx = seq_head % nv;
    const int kv_group = nv / nk;
    const int k_head_idx = v_head_idx / kv_group;
    const int slot = (int)slots[seq_idx];
    if (slot < 0) return;

    const int start = (int)cu_seqlens[seq_idx];
    const int end = (int)cu_seqlens[seq_idx + 1];
    const int seq_len = end - start;
    if (seq_len <= 0) return;

    const int token_stride_qk = nk * kd;
    const int token_stride_v = nv * vd;
    const int token_stride_g = nv;

    const __half *q_base = q + (size_t)start * token_stride_qk + (size_t)k_head_idx * kd;
    const __half *k_base = k + (size_t)start * token_stride_qk + (size_t)k_head_idx * kd;
    const __half *v_base = v + (size_t)start * token_stride_v + (size_t)v_head_idx * vd;
    const __half *g_base = g + (size_t)start * token_stride_g + v_head_idx;
    const __half *beta_base = beta + (size_t)start * token_stride_g + v_head_idx;
    __half *out_base = out + (size_t)start * token_stride_v + (size_t)v_head_idx * vd;

    __shared__ float q_buf[2][BK];
    __shared__ float k_buf[2][BK];
    __shared__ float scalars[2][2];

    const bool v_valid = (v_idx < vd);
    float *state_head = v_valid
        ? state + ((size_t)slot * nv + v_head_idx) * kd * vd
        : nullptr;

    float s_shard[ROWS_PER_LANE];
    if (v_valid) {
        for (int r = 0; r < ROWS_PER_LANE; ++r) {
            const int k_idx = r * GDN_WARP_SIZE + lane;
            s_shard[r] = ((size_t)k_idx < kd) ? state_head[(size_t)k_idx * vd + v_idx] : 0.0f;
        }
    }

    const int total_threads = WARPS_PER_BLOCK * GDN_WARP_SIZE;
    const int tid = warp_id * GDN_WARP_SIZE + lane;

    if (seq_len > 0) {
        for (int j = tid; j < BK; j += total_threads) {
            if (j < kd) {
                q_buf[0][j] = gdn16_to_float(q_base[j]) * q_scale;
                k_buf[0][j] = gdn16_to_float(k_base[j]);
            } else {
                q_buf[0][j] = 0.0f;
                k_buf[0][j] = 0.0f;
            }
        }
        if (tid == 0) {
            scalars[0][0] = expf(gdn16_to_float(g_base[0]));
            scalars[0][1] = gdn16_to_float(beta_base[0]);
        }
        __syncthreads();
    }

    for (int t = 0; t < seq_len; ++t) {
        const int cur = t & 1;
        const int nxt = 1 - cur;
        if (t + 1 < seq_len) {
            const __half *q_next = q_base + (size_t)(t + 1) * token_stride_qk;
            const __half *k_next = k_base + (size_t)(t + 1) * token_stride_qk;
            for (int j = tid; j < BK; j += total_threads) {
                if ((size_t)j < kd) {
                    q_buf[nxt][j] = gdn16_to_float(q_next[j]) * q_scale;
                    k_buf[nxt][j] = gdn16_to_float(k_next[j]);
                } else {
                    q_buf[nxt][j] = 0.0f;
                    k_buf[nxt][j] = 0.0f;
                }
            }
            if (tid == 0) {
                scalars[nxt][0] = expf(gdn16_to_float(g_base[(size_t)(t + 1) * token_stride_g]));
                scalars[nxt][1] = gdn16_to_float(beta_base[(size_t)(t + 1) * token_stride_g]);
            }
        }

        if (v_valid) {
            const float decay = scalars[cur][0];
            const float beta_t = scalars[cur][1];
            float kv_partial = 0.0f;
            for (int r = 0; r < ROWS_PER_LANE; ++r) {
                const int k_idx = r * GDN_WARP_SIZE + lane;
                s_shard[r] *= decay;
                kv_partial = __fmaf_rn(s_shard[r], k_buf[cur][(size_t)k_idx], kv_partial);
            }
            const float kv_mem = gdn16_warp_reduce_sum(kv_partial);
            const float delta = (gdn16_to_float(v_base[(size_t)t * token_stride_v + v_idx]) - kv_mem) * beta_t;
            float y_partial = 0.0f;
            for (int r = 0; r < ROWS_PER_LANE; ++r) {
                const int k_idx = r * GDN_WARP_SIZE + lane;
                s_shard[r] = __fmaf_rn(k_buf[cur][(size_t)k_idx], delta, s_shard[r]);
                y_partial = __fmaf_rn(s_shard[r], q_buf[cur][(size_t)k_idx], y_partial);
            }
            const float y_t = gdn16_warp_reduce_sum(y_partial);
            if (lane == 0) {
                out_base[(size_t)t * token_stride_v + (size_t)v_idx] = gdn16_from_float(y_t);
            }
        }
        __syncthreads();
    }

    if (v_valid) {
        for (int r = 0; r < ROWS_PER_LANE; ++r) {
            const int k_idx = r * GDN_WARP_SIZE + lane;
            if ((size_t)k_idx < kd) {
                state_head[(size_t)k_idx * vd + (size_t)v_idx] = s_shard[r];
            }
        }
    }
}

// ============================================================================
// 入口(入口名/签名/槽序 = 契约,不改;快慢路由 = `gdn_rec_fast_shape`)
// ============================================================================

/// 快路形状判据 —— **必须与 `driver::gdn::recurrence_varlen_gqa_fast` 逐字同源**
/// (device 侧调不到 host fn;两边都由 `gdn_rec_geometry_matches_route` 单测钉住:
/// 一致时只是快慢,**不一致时是错的覆盖**(legacy 的 v_idx 步长 = 8,快路的 = 16))
__device__ __forceinline__ bool gdn_rec_fast_shape(size_t kd, size_t vd) {
    return kd == 128 && (vd % OWL_GDN16_COLS_PER_WARP) == 0;
}

extern "C" __global__ void owl_gdn_recurrence_varlen_gqa_f16(
    const __half *__restrict__ q,        // [total, NK, K]
    const __half *__restrict__ k,        // [total, NK, K]
    const __half *__restrict__ v,        // [total, NV, V]
    const __half *__restrict__ g,        // [total, NV](log 空间)
    const __half *__restrict__ beta,     // [total, NV]
    float *__restrict__ state,           // [max_slots, NV, K, V](in/out)
    const float *__restrict__ slots,     // [batch](负 = 全跳)
    const float *__restrict__ cu_seqlens, // [batch + 1](f32 数值过线,契约 5)
    size_t batch, size_t nv, size_t nk,
    size_t kd, size_t vd,
    float q_scale,
    __half *__restrict__ out) {          // [total, NV, V](末参 = 输出)
    if (gdn_rec_fast_shape(kd, vd)) {
        gdn_rec_cols2(q, k, v, g, beta, state, slots, cu_seqlens, batch, nv, nk, kd, vd, q_scale,
                      out);
    } else {
        gdn_rec_legacy(q, k, v, g, beta, state, slots, cu_seqlens, batch, nv, nk, kd, vd, q_scale,
                       out);
    }
}
