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

// ---- gated delta rule prefill varlen gqa f16(上游 recurrence_varlen_
// gqa_kernel<T,128,8> 同体;g/beta half 入参、slots f32、out 重排末参、
// 去 state_snapshots;state 槽寻址恒 f32;cu_seqlens u32)----
// grid = (ceil(vd/8), batch·HV) × (32,8);t 循环核内,q/k 双缓冲 shared,
// s_shard 每线寄存器分片(KD≤128 → 4 行/线);warp_reduce 归约。
#define OWL_GDN16_WARPS_PER_BLOCK 8

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
