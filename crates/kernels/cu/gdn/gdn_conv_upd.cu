// ============================================================================
// GDN 核族:causal conv1d decode 槽更新(k=4 收窄,无 bias)(2026-10-13 C 轮自 cu/owl/gdn.cu 按**算子族**拆出 ——
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

// ---- causal conv1d decode 槽更新(批3;k=4 收窄,无 bias;批6 修订)----
// 每线程一 (b, channel):读 state[slot] 历史窗 → 卷积 → silu → 写 out
// 并滑动 state。三段 q/k/v 各自独立发射(段内通道连续,state 亦按段
// 建块),零拼接算子 —— **段基址走 w_offset 标量**(共享单权重块,
// 勿用 narrow 行切:narrow 的 start 是行内列偏移,做不了行块偏移)。
// Qwen3.5 conv1d bias=False,无 bias 形参。
//   out[b][ch] = silu(x[b][ch]·w[w_off+ch][3] + Σ_{k<3} hist[k]·w[w_off+ch][k])
//   state[slot][ch] = [hist1, hist2, x[b][ch]](原地,单流保序)
// slots f32 数值过线;slot < 0 = padding:跳写 state 也跳写 out(行不落地)。

extern "C" __global__ void owl_gdn_conv_upd_f32(
    const float *__restrict__ x,         // [batch, d]
    const float *__restrict__ w,         // [conv_dim_total, 4](w_off 起为本段)
    float *__restrict__ conv_state,      // [max_slots, d, 3](in/out 恒 f32)
    const float *__restrict__ slots,     // [batch](f32 数值;负 = padding)
    size_t total, size_t d, size_t w_offset,
    int silu,
    float *__restrict__ out) {           // [batch, d](末参 = 输出)
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t b = idx / d;
    const size_t ch = idx % d;
    const int slot = (int)slots[b];
    if (slot < 0) return;                              // padding:全跳
    const float *w_ptr = w + (w_offset + ch) * 4;
    float *state_ptr = conv_state + ((size_t)slot * d + ch) * 3;
    float hist[3] = {state_ptr[0], state_ptr[1], state_ptr[2]};
    const float x_t = x[idx];
    float sum = x_t * w_ptr[3];
    for (int k = 0; k < 3; ++k) sum = __fmaf_rn(hist[k], w_ptr[k], sum);
    if (silu) sum /= (1.0f + __expf(-sum));
    state_ptr[0] = hist[1];
    state_ptr[1] = hist[2];
    state_ptr[2] = x_t;
    out[idx] = sum;
}
// ---- causal conv1d decode 槽更新 f16(上游 causal_conv1d_update_slots_
// kernel<T> 同体;KERNEL_SIZE=4 收窄,无 bias;w_offset 段基址 = owl 契约)----
extern "C" __global__ void owl_gdn_conv_upd_f16(
    const __half *__restrict__ x,        // [batch, d]
    const __half *__restrict__ w,        // [conv_dim_total, 4](w_off 起为本段)
    float *__restrict__ conv_state,      // [max_slots, d, 3](in/out 恒 f32)
    const float *__restrict__ slots,     // [batch](负 = padding)
    size_t total, size_t d, size_t w_offset,
    int silu,
    __half *__restrict__ out) {          // [batch, d](末参 = 输出)
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const size_t b = idx / d;
    const size_t ch = idx % d;
    const int slot = (int)slots[b];
    if (slot < 0) return;
    const __half *w_ptr = w + (w_offset + ch) * 4;
    float *state_ptr = conv_state + ((size_t)slot * d + ch) * 3;
    float hist[3] = {state_ptr[0], state_ptr[1], state_ptr[2]};
    const float x_t = gdn16_to_float(x[idx]);
    float sum = x_t * gdn16_to_float(w_ptr[3]);
    for (int k = 0; k < 3; ++k) sum = __fmaf_rn(hist[k], gdn16_to_float(w_ptr[k]), sum);
    if (silu) sum /= (1.0f + __expf(-sum));
    state_ptr[0] = hist[1];
    state_ptr[1] = hist[2];
    state_ptr[2] = x_t;
    out[idx] = gdn16_from_float(sum);
}

// ---- 刀3b(2026-10-04,E-decode 冲 45):q/k 双段 conv 槽更新单发 ——
// 替 owl_gdn_conv_upd_f16 ×2(q/k 各一; cublas 无涉,纯发射收敛 -48 节点/步)。
// 两段各自独立 state/权重,逻辑平面 [batch, dq+dk];输出单块 [batch, dq+dk]
// (SSA 单出;消费侧 T=1 时列切 = 连续 BlockSlice 免费视图)。契约同族:
// 7 Block(x_q,x_k,w_q,w_k,state_q,state_k,slots)+ 4 i32(dq,dk,batch,silu)+ out。
extern "C" __global__ void owl_gdn_conv_upd_dual_f16(
    const __half *__restrict__ x_q,      // [batch, dq]
    const __half *__restrict__ x_k,      // [batch, dk]
    const __half *__restrict__ w_q,      // [dq, 4]
    const __half *__restrict__ w_k,      // [dk, 4]
    float *__restrict__ state_q,         // [max_slots, dq, 3]
    float *__restrict__ state_k,         // [max_slots, dk, 3]
    const float *__restrict__ slots,     // [batch](负 = padding)
    const int dq, const int dk, const int batch, const int silu,
    __half *__restrict__ out) {          // [batch, dq + dk](末参 = 输出)
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int seg = dq + dk;
    if ((size_t)idx >= (size_t)batch * seg) return;
    const size_t b = idx / seg;
    const size_t i = idx % seg;
    const int slot = (int)slots[b];
    if (slot < 0) return;
    const bool is_q = i < (size_t)dq;
    const size_t ch = is_q ? i : i - dq;
    const __half *x_ptr = (is_q ? x_q : x_k);
    const __half *w_ptr = (is_q ? w_q : w_k) + ch * 4;
    float *state_ptr = (is_q ? state_q : state_k) + ((size_t)slot * (is_q ? dq : dk) + ch) * 3;
    const float x_t = gdn16_to_float(x_ptr[b * (is_q ? dq : dk) + ch]);
    float sum = x_t * gdn16_to_float(w_ptr[3]);
    for (int k = 0; k < 3; ++k) sum = __fmaf_rn(state_ptr[k], gdn16_to_float(w_ptr[k]), sum);
    if (silu) sum /= (1.0f + __expf(-sum));
    state_ptr[0] = state_ptr[1];
    state_ptr[1] = state_ptr[2];
    state_ptr[2] = x_t;
    out[idx] = gdn16_from_float(sum);
}
