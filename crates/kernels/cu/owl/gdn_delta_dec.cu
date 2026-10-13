// ============================================================================
// GDN 核族:gated delta rule decode(单步,slot 寻址,GQA)(2026-10-13 C 轮自 cu/owl/gdn.cu 按**算子族**拆出 ——
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

// ---- gated delta rule decode(批4;单步,slot 寻址,GQA)----
// 一 block 一 (batch·v_head, v 64 切片);kd ≤ 128(寄存器 s_buf 硬上界,
// 0.8B hd_k=128 恰满)。g 为 log 空间(核内 exp);q 在核内乘 q_scale。
//   state *= exp(g);  kv_mem = Σ_k state[k]·k[k]
//   delta = (v - kv_mem)·beta;  state += k⊗delta;  out = Σ_k state[k]·(q[k]·scale)
// 布局 q/k [B,HK,K]、v/out [B,HV,V]、state [max_slots,HV,K,V] 恒 f32、
// g/beta [B,HV];slots f32 数值过线(负 = padding:跳写 state 与 out)。
// 发射 = (ceil(vd/64), B·HV) × (64,1,1),shared = (128+128+2)·4 字节。

#define OWL_GDN_MAX_KD 128

extern "C" __global__ void owl_gdn_delta_dec_f32(
    const float *__restrict__ q,         // [B, HK, K]
    const float *__restrict__ k,         // [B, HK, K]
    const float *__restrict__ v,         // [B, HV, V]
    const float *__restrict__ g,         // [B, HV](log 空间)
    const float *__restrict__ beta,      // [B, HV]
    float *__restrict__ state,           // [max_slots, HV, K, V](in/out)
    const float *__restrict__ slots,     // [B](f32 数值;负 = padding)
    size_t batch, size_t nv, size_t nk,
    size_t kd, size_t vd,
    float q_scale,
    float *__restrict__ out) {           // [B, HV, V](末参 = 输出)
    const unsigned int v_tile = blockIdx.x;
    const unsigned int bh = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int v_idx = v_tile * 64u + tid;
    if (bh >= batch * nv) return;
    const bool v_valid = v_idx < vd;
    const unsigned int b = bh / nv;
    const unsigned int v_head = bh % nv;
    const unsigned int kv_group = nv / nk;
    const unsigned int k_head = v_head / kv_group;
    const int slot = (int)slots[b];
    const bool slot_valid = slot >= 0;
    extern __shared__ float smem[];
    float *q_smem = smem;
    float *k_smem = smem + OWL_GDN_MAX_KD;
    float *scalars = smem + 2 * OWL_GDN_MAX_KD;
    if (tid == 0) {
        scalars[0] = __expf(g[b * nv + v_head]);
        scalars[1] = beta[b * nv + v_head];
    }
    const float *q_bh = q + (b * nk + k_head) * kd;
    const float *k_bh = k + (b * nk + k_head) * kd;
    for (size_t j = tid; j < kd; j += blockDim.x)
        k_smem[j] = k_bh[j];
    __syncthreads();
    const float decay = scalars[0];
    const float beta_t = scalars[1];
    float *state_head = slot_valid
        ? state + ((size_t)slot * nv + v_head) * kd * vd
        : nullptr;
    float s_buf[OWL_GDN_MAX_KD];
    for (size_t j = 0; j < kd; ++j)
        s_buf[j] = (v_valid && slot_valid) ? state_head[j * vd + v_idx] : 0.0f;
    float kv_mem = 0.0f;
    for (size_t j = 0; j < kd; ++j) {
        if (v_valid && slot_valid) {
            s_buf[j] *= decay;
            kv_mem = __fmaf_rn(s_buf[j], k_smem[j], kv_mem);
        }
    }
    const float *v_bh = v + (b * nv + v_head) * vd;
    const float delta = (v_valid && slot_valid)
        ? (v_bh[v_idx] - kv_mem) * beta_t : 0.0f;
    __syncthreads();
    for (size_t j = tid; j < kd; j += blockDim.x)
        q_smem[j] = q_bh[j] * q_scale;
    __syncthreads();
    float y = 0.0f;
    for (size_t j = 0; j < kd; ++j) {
        if (v_valid && slot_valid) {
            s_buf[j] = __fmaf_rn(k_smem[j], delta, s_buf[j]);
            y = __fmaf_rn(s_buf[j], q_smem[j], y);
        }
    }
    for (size_t j = 0; j < kd; ++j) {
        if (v_valid && slot_valid)
            state_head[j * vd + v_idx] = s_buf[j];
    }
    if (v_valid && slot_valid)
        out[(b * nv + v_head) * vd + v_idx] = y;
}
// ---- gated delta rule decode f16(上游 decode_slots_gqa_kernel<T,BV,BK>
// BK=128 档展开;g/beta half 入参(上游 float,适配 owl 块词汇);slots f32;
// out 重排至末参(槽序契约 4));state 恒 f32;s_buf float 寄存器分片)----
// 发射 = (ceil(vd/64), B·HV) × (64,1,1),shared = (2·128+2)·4 字节。
#define OWL_GDN16_MAX_KD 128

extern "C" __global__ void owl_gdn_delta_dec_f16(
    const __half *__restrict__ q,        // [B, HK, K]
    const __half *__restrict__ k,        // [B, HK, K]
    const __half *__restrict__ v,        // [B, HV, V]
    const __half *__restrict__ g,        // [B, HV](log 空间)
    const __half *__restrict__ beta,     // [B, HV]
    float *__restrict__ state,           // [max_slots, HV, K, V](in/out)
    const float *__restrict__ slots,     // [B](负 = padding)
    size_t batch, size_t nv, size_t nk,
    size_t kd, size_t vd,
    float q_scale,
    __half *__restrict__ out) {          // [B, HV, V](末参 = 输出)
    const unsigned int v_tile = blockIdx.x;
    const unsigned int bh = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int v_idx = v_tile * 64u + tid;
    if (bh >= batch * nv) return;
    const bool v_valid = v_idx < vd;
    const unsigned int b = bh / nv;
    const unsigned int v_head = bh % nv;
    const unsigned int kv_group = nv / nk;
    const unsigned int k_head = v_head / kv_group;
    const int slot = (int)slots[b];
    const bool slot_valid = slot >= 0;
    extern __shared__ float smem[];
    float *q_smem = smem;
    float *k_smem = smem + OWL_GDN16_MAX_KD;
    float *scalars = smem + 2 * OWL_GDN16_MAX_KD;
    if (tid == 0) {
        scalars[0] = expf(gdn16_to_float(g[b * nv + v_head]));
        scalars[1] = gdn16_to_float(beta[b * nv + v_head]);
    }
    const __half *q_bh = q + (b * nk + k_head) * kd;
    const __half *k_bh = k + (b * nk + k_head) * kd;
    for (size_t j = tid; j < kd; j += blockDim.x)
        k_smem[j] = gdn16_to_float(k_bh[j]);
    __syncthreads();
    const float decay = scalars[0];
    const float beta_t = scalars[1];
    float *state_head = slot_valid
        ? state + ((size_t)slot * nv + v_head) * kd * vd
        : nullptr;
    float s_buf[OWL_GDN16_MAX_KD];
    for (size_t j = 0; j < kd; ++j)
        s_buf[j] = (v_valid && slot_valid) ? state_head[j * vd + v_idx] : 0.0f;
    float kv_mem = 0.0f;
    for (size_t j = 0; j < kd; ++j) {
        if (v_valid && slot_valid) {
            s_buf[j] *= decay;
            kv_mem = __fmaf_rn(s_buf[j], k_smem[j], kv_mem);
        }
    }
    const __half *v_bh = v + (b * nv + v_head) * vd;
    const float delta = (v_valid && slot_valid)
        ? (gdn16_to_float(v_bh[v_idx]) - kv_mem) * beta_t : 0.0f;
    __syncthreads();
    for (size_t j = tid; j < kd; j += blockDim.x)
        q_smem[j] = gdn16_to_float(q_bh[j]) * q_scale;
    __syncthreads();
    float y = 0.0f;
    for (size_t j = 0; j < kd; ++j) {
        if (v_valid && slot_valid) {
            s_buf[j] = __fmaf_rn(k_smem[j], delta, s_buf[j]);
            y = __fmaf_rn(s_buf[j], q_smem[j], y);
        }
    }
    for (size_t j = 0; j < kd; ++j) {
        if (v_valid && slot_valid)
            state_head[j * vd + v_idx] = s_buf[j];
    }
    if (v_valid && slot_valid)
        out[(b * nv + v_head) * vd + v_idx] = gdn16_from_float(y);
}
