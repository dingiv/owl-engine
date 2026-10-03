// gdn kernels(Qwen3.5 GatedDeltaNet 线性注意力;port 自 nn/kernels/cu/gdn
// 旧世界 gdn_kernels.cu,出处 vendor/attention.rs rev c0f19f2 src/kernels/src/gdn.cu;
// A4 所有权:owl-kernels 本文件为新世界唯一源码之家,f32 单 dtype)。
//
// 新世界契约改造(相对旧世界):
//   - 单输出契约:旧 fused_gating 双输出(g/beta)拆臂 —— g 走新核
//     (softplus 无语义等价),beta = sigmoid(b) 复用 ops.cu owl_sigmoid_f32
//     (M-b qk-norm/sigmoid-gate 同款改判,不新写);
//   - 标量形参 = size_t(与 Arg::U64/arg_usize 8 字节严格对位);
//     flag 形参 = int(Arg::I32;禁止 unsigned int);
//   - 输出块固定末参(槽序契约 4);
//   - 槽/索引量以 f32 数值过线(契约 5;核内 cast;负值 = padding 跳过);
//   - nvrtc 懒编译:无 host launcher、无模板宏展开(f32 单档)。

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

// ---- GDN 门控 g 臂(Qwen3_5GatedDeltaNet 融合门控同式)----
//   g[i] = -exp(A_log[h]) · softplus(a[i] + dt_bias[h]),h = i % heads
// softplus 分支与 torch 一致(x < 20 走 log1p(exp x),否则直通)。
// a_log/dt_bias [heads] 恒 f32(checkpoint 直存);a [total] = [T, H] 展平。
// (beta 臂 = sigmoid(b),复用 owl_sigmoid_f32,不在本文件。)

extern "C" __global__ void owl_gdn_gating_g_f32(
    const float *__restrict__ a_log,     // [heads]
    const float *__restrict__ a,         // [total](= [T, H] 展平)
    const float *__restrict__ dt_bias,   // [heads]
    size_t total, size_t heads,
    float *__restrict__ g) {             // [total](末参 = 输出)
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t h = (size_t)(idx % heads);
    const float x = a[idx] + dt_bias[h];
    float sp = (x < 20.0f) ? log1pf(expf(x)) : x;
    g[idx] = -__expf(a_log[h]) * sp;
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

// ---- 门控 RMSNorm × act(z)(批5;Qwen3_5RMSNormGated 同式)----
//   y[i] = rmsnorm(x 组内)·gamma[i] · act(z[i])
// 一 block 一 (row, group);gamma [group_size] 组内共享(×w 非零中心:
// RMSNormGated weight=ones 初始化,与 qk-norm 零中心相反 —— 已 HF 实证);
// act:0 = silu(Qwen3.5),1 = sigmoid(Qwen4 预留);无 bias。
// 发射 = (rows·value_dim/group_size,1,1) × (256,1,1)。

extern "C" __global__ void owl_gdn_norm_act_f32(
    const float *__restrict__ x,         // [rows, value_dim]
    const float *__restrict__ z,         // [rows, value_dim]
    const float *__restrict__ gamma,     // [group_size](×w 语义)
    size_t rows, size_t value_dim, size_t group_size,
    float eps,
    int act,
    float *__restrict__ out) {           // [rows, value_dim](末参 = 输出)
    const size_t row_group = blockIdx.x;
    const size_t num_groups = value_dim / group_size;
    const size_t row = row_group / num_groups;
    const size_t group = row_group % num_groups;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) return;
    const size_t group_offset = row * value_dim + group * group_size;
    const float *x_row = x + group_offset;
    const float *z_row = z + group_offset;
    float *out_row = out + group_offset;
    float sumsq = 0.0f;
    for (size_t i = tid; i < group_size; i += blockDim.x) {
        sumsq = __fmaf_rn(x_row[i], x_row[i], sumsq);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    __shared__ float warp_sums[8];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sumsq;
    __syncthreads();
    float total = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 4; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
    }
    if (tid == 0) warp_sums[0] = total;
    __syncthreads();
    const float inv = rsqrtf(fmaxf(warp_sums[0] / (float)group_size, 0.0f) + eps);
    for (size_t i = tid; i < group_size; i += blockDim.x) {
        const float nx = x_row[i] * inv * gamma[i];
        const float zv = z_row[i];
        const float actv = (act == 0) ? (zv / (1.0f + __expf(-zv)))
                                      : (1.0f / (1.0f + __expf(-zv)));
        out_row[i] = nx * actv;
    }
}

// ---- 末维 L2 归一(block256 变体;一 block 一行)----
//   y[r][i] = x[r][i] · rsqrt(sum(x[r][·]²) + eps)
// 与 HF l2norm(qwen3_5 modeling 内嵌定义,对齐 FLA)同式。
// 旧世界 warp/block 双变体合并:mini-demo dim(128/256) ≤ 256 走块内归约;
// 发射 = (rows,1,1) × (256,1,1)(行核,非哨兵)。

extern "C" __global__ void owl_gdn_l2norm_f32(
    const float *__restrict__ x,         // [rows, dim]
    size_t rows, size_t dim, float eps,
    float *__restrict__ out) {           // [rows, dim](末参 = 输出)
    const size_t row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const float *in_row = x + row * dim;
    float *out_row = out + row * dim;
    float sumsq = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        sumsq = __fmaf_rn(in_row[i], in_row[i], sumsq);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    __shared__ float warp_sums[8];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sumsq;
    __syncthreads();
    float total = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 4; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
    }
    if (tid == 0) warp_sums[0] = total;
    __syncthreads();
    const float inv = rsqrtf(fmaxf(warp_sums[0], 0.0f) + eps);
    for (size_t i = tid; i < dim; i += blockDim.x)
        out_row[i] = in_row[i] * inv;
}


// ============================================================================
// f16 基线变体(F4 换源版;2026-09-26 工单 G)
// 溯源:repos/attention.rs/src/kernels/src/gdn.cu 模板体显式展开(nvrtc 禁
// host launcher,<<<>>> 剥离;旧世界 GDN_GATING_KERNEL 宏模式同款)。
// dtype 律:激活 half → float 计算 → half 写;**rec/conv state 恒 f32**
// (上游同款 + HF/vLLM/xinfer 三方先例);slots 恒 f32 数值过线(契约 5,
// 上游 i64 已适配);输出块固定末参(槽序契约 4,上游 gqa 核已重排)。
// 数学与上方 f32 版逐式同源(移植不失真由 f16_tests 对拍背书)。
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

// ---- 门控 g 臂 f16(上游 fused_gdn_gating_kernel/compute_gating 的 g 臂;
// owl 单输出契约:beta = sigmoid(b) 走语义算子,M-b 改判)----
//   g[i] = -exp(A_log[h]) · softplus(a[i] + dt_bias[h]),h = i % heads
// softplus 分支沿上游(x <= 20 走 log1pf(exp x),否则直通)。
extern "C" __global__ void owl_gdn_gating_g_f16(
    const __half *__restrict__ a_log,    // [heads]
    const __half *__restrict__ a,        // [total](= [T, H] 展平)
    const __half *__restrict__ dt_bias,  // [heads]
    size_t total, size_t heads,
    __half *__restrict__ g) {            // [total](末参 = 输出)
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t h = (size_t)(idx % heads);
    const float x = gdn16_to_float(a[idx]) + gdn16_to_float(dt_bias[h]);
    const float sp = (x <= 20.0f) ? log1pf(expf(x)) : x;
    g[idx] = gdn16_from_float(-expf(gdn16_to_float(a_log[h])) * sp);
}

// ---- 末维 L2 归一 f16(上游 l2_norm_last_dim_block_kernel<T,256> 同体)----
extern "C" __global__ void owl_gdn_l2norm_f16(
    const __half *__restrict__ x,        // [rows, dim]
    size_t rows, size_t dim, float eps,
    __half *__restrict__ out) {          // [rows, dim](末参 = 输出)
    const size_t row = blockIdx.x;
    if (row >= rows) return;
    const unsigned int tid = threadIdx.x;
    const __half *in_row = x + row * dim;
    __half *out_row = out + row * dim;
    float sumsq = 0.0f;
    for (size_t i = tid; i < dim; i += blockDim.x) {
        const float v = gdn16_to_float(in_row[i]);
        sumsq = __fmaf_rn(v, v, sumsq);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    __shared__ float warp_sums[8];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sumsq;
    __syncthreads();
    float total = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 4; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
    }
    if (tid == 0) warp_sums[0] = total;
    __syncthreads();
    const float inv = rsqrtf(fmaxf(warp_sums[0], 0.0f) + eps);
    for (size_t i = tid; i < dim; i += blockDim.x)
        out_row[i] = gdn16_from_float(gdn16_to_float(in_row[i]) * inv);
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

// ---- 门控 RMSNorm × act(z) f16(上游 gated_rmsnorm_silu_mul_kernel
// <T,W,256> 同体;owl 简化:per_group_weights=true、无 bias)----
extern "C" __global__ void owl_gdn_norm_act_f16(
    const __half *__restrict__ x,        // [rows, value_dim]
    const __half *__restrict__ z,        // [rows, value_dim]
    const __half *__restrict__ gamma,    // [group_size](×w 非零中心)
    size_t rows, size_t value_dim, size_t group_size,
    float eps,
    int act,
    __half *__restrict__ out) {          // [rows, value_dim](末参 = 输出)
    const size_t row_group = blockIdx.x;
    const size_t num_groups = value_dim / group_size;
    const size_t row = row_group / num_groups;
    const size_t group = row_group % num_groups;
    const unsigned int tid = threadIdx.x;
    if (row >= rows) return;
    const size_t group_offset = row * value_dim + group * group_size;
    const __half *x_row = x + group_offset;
    const __half *z_row = z + group_offset;
    __half *out_row = out + group_offset;
    float sumsq = 0.0f;
    for (size_t i = tid; i < group_size; i += blockDim.x) {
        const float v = gdn16_to_float(x_row[i]);
        sumsq = __fmaf_rn(v, v, sumsq);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);
    __shared__ float warp_sums[8];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) warp_sums[warp_id] = sumsq;
    __syncthreads();
    float total = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (warp_id == 0) {
        for (int offset = 4; offset > 0; offset >>= 1)
            total += __shfl_down_sync(0xffffffff, total, offset);
    }
    if (tid == 0) warp_sums[0] = total;
    __syncthreads();
    const float inv = rsqrtf(fmaxf(warp_sums[0] / (float)group_size, 0.0f) + eps);
    for (size_t i = tid; i < group_size; i += blockDim.x) {
        const float nx = gdn16_to_float(x_row[i]) * inv * gdn16_to_float(gamma[i]);
        const float zv = gdn16_to_float(z_row[i]);
        const float actv = (act == 0) ? (zv / (1.0f + __expf(-zv)))
                                      : (1.0f / (1.0f + __expf(-zv)));
        out_row[i] = gdn16_from_float(nx * actv);
    }
    // (2026-10-04 清撤:D1 取证期残留的无条件设备端 printf —— 每次
    // norm_act 发射 3 行 trap+flush 串行化,基线测速含此税;核内调试
    // 打印纪律:取证完必须撤,不留无门控残留)
}

// ============================================================================
// PF1b 批核(chunked prefill;T 循环在核内,一次发射吃全块;同源上游)
// ============================================================================

// ---- causal conv1d prefill varlen f16(上游 causal_conv1d_fwd_varlen_
// kernel<T,4> 同体;owl 适配:state 槽寻址(上游 seq 直索,免 gather/scatter
// 垫)、去 state_snapshots(MTP 域外)、cu_seqlens [batch+1] u32 保真)----
// grid = (batch, ceil(d/256)) × (256,1,1);t 循环核内顺序扫描。
extern "C" __global__ void owl_gdn_conv_fwd_f16(
    const __half *__restrict__ x,            // [total_tokens, d]
    const __half *__restrict__ w,            // [d, 4]
    float *__restrict__ conv_state,          // [max_slots, d, 3](in/out 恒 f32)
    const float *__restrict__ slots,         // [batch](f32 数值;负 = 全跳)
    const float *__restrict__ cu_seqlens,    // [batch + 1](f32 数值过线,契约 5)
    int batch, int d, int silu,
    __half *__restrict__ out) {              // [total_tokens, d](末参 = 输出)
    const int seq_idx = blockIdx.x;
    int channel_idx = blockIdx.y * blockDim.x + threadIdx.x;
    if (seq_idx >= batch || channel_idx >= d) return;
    const int slot = (int)slots[seq_idx];
    const int start = (int)cu_seqlens[seq_idx];
    const int end = (int)cu_seqlens[seq_idx + 1];
    const int seq_len = end - start;
    const __half *w_ptr = w + channel_idx * 4;
    float *state_ptr = (slot >= 0)
        ? conv_state + ((size_t)slot * d + channel_idx) * 3
        : nullptr;
    float w_reg[4];
    for (int k = 0; k < 4; ++k) w_reg[k] = gdn16_to_float(w_ptr[k]);
    float hist[3] = {0.0f, 0.0f, 0.0f};
    if (state_ptr) {
        for (int i = 0; i < 3; ++i) hist[i] = state_ptr[i];
    }
    for (int t = 0; t < seq_len; ++t) {
        const float x_t = gdn16_to_float(x[(size_t)(start + t) * d + channel_idx]);
        float sum = x_t * w_reg[3];
        for (int k = 0; k < 3; ++k) sum = __fmaf_rn(hist[k], w_reg[k], sum);
        if (silu) sum /= (1.0f + __expf(-sum));
        out[(size_t)(start + t) * d + channel_idx] = gdn16_from_float(sum);
        hist[0] = hist[1];
        hist[1] = hist[2];
        hist[2] = x_t;
    }
    if (state_ptr) {
        for (int i = 0; i < 3; ++i) state_ptr[i] = hist[i];
    }
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

// ============================================================================
// D1:decode 整链融合核(v-conv + l2norm×2 + gating + sigmoid + delta + norm_act
// 六算子一发;2026-10-03 decode 主攻 C1 波次)
// ----------------------------------------------------------------------------
// 动机:decode 每 GDN 层 17 发(4 marlin + narrow×3 + conv×3 + l2norm×2 +
// gating + sigmoid + delta_dec + norm_act + out_proj),48 层 ≈ 816 发/步 ——
// 发射开销与延迟链主导。本核把 v 段之后的六个小核合一,层发射 17 → 11。
//
// **为什么 q/k conv_upd 不并入(GVA 跨块竞态定谳)**:grid = (B·NV) 时同一
// k_head 的 conv state 被 kv_group=3 个块读写 —— 三块写值相同(幂等)但
// 「读 hist」与「写新值」无跨块序,后写先读即得移位 hist → 组内 q_c/k_c
// 分叉。故 q/k conv_upd 保留独立发射(2 发),本核吃其产物;v 段 state 按
// v-head 严格分块,无跨块触碰,安全并入。
//
// 结构:grid (B·NV,1,1) × block (kd,1,1)(kd==vd==blockDim,≤128 守卫;
// 每块独占一个 (b, v_head):v-conv → l2norm q/k(纯 shuffle 归约)→
// gating/beta(寄存器冗余)→ delta 步(s_buf[kd] 寄存器,列并行)→
// norm_act(块归约)。数学逐式镜像 owl_gdn_{conv_upd,l2norm,gating_g,
// delta_dec,norm_act}_f16,公式零改动。
// slot<0(padding):state 跳写,y=0,出 = norm_act(0) = 0(确定性;
// 旧链出 = 陈值未定义,本核更严,decode 单槽恒 ≥0 不涉)。
// ----
extern "C" __global__ void owl_gdn_decode_step_f16(
    const __half *__restrict__ q_c,      // [B, K](conv_upd q 产物,含 silu)
    const __half *__restrict__ k_c,      // [B, K]
    const __half *__restrict__ v_raw,    // [B, V](投影段,未 conv)
    const __half *__restrict__ z,        // [B, V]
    const __half *__restrict__ b_gate,   // [B, HV]
    const __half *__restrict__ a_gate,   // [B, HV]
    const __half *__restrict__ w,        // [conv_dim_total, 4]
    float *__restrict__ conv_v_state,    // [max_slots, V, 3](in/out)
    const __half *__restrict__ a_log,    // [HV]
    const __half *__restrict__ dt_bias,  // [HV]
    float *__restrict__ rec_state,       // [max_slots, HV, KD, VD](in/out)
    const float *__restrict__ slots,     // [B](负 = padding)
    const __half *__restrict__ norm_w,   // [VD](×w 非零中心)
    size_t batch, size_t nv, size_t nk, size_t kd, size_t vd,
    float eps_l2, float eps_norm, float q_scale,
    __half *__restrict__ out) {          // [B, HV·VD](末参 = 输出)
    const unsigned int d = (unsigned int)kd;
    if (d != blockDim.x || kd != vd || d == 0 || d > 128) return;
    const size_t bh = blockIdx.x;
    const size_t b = bh / nv;
    const size_t h = bh % nv;
    const size_t kv_group = nv / nk;
    const size_t kh = h / kv_group;
    const int slot = (int)slots[b];
    const bool valid = slot >= 0;
    const unsigned int tid = threadIdx.x;
    const size_t key_dim = kd * nk;
    const size_t value_dim = vd * nv;
    const size_t conv_dim = 2 * key_dim + value_dim;

    __shared__ float warp_sums[8];
    __shared__ float q_smem[128];   // 归一化后 q 向量(delta 全维读)
    __shared__ float k_smem[128];   // 归一化后 k 向量

    // ---- v 段 conv(state 滑窗 + silu;线程 tid = 头内通道 tid)----
    float v_c = 0.0f;
    if (valid) {
        const size_t ch = h * vd + tid;
        float *sp = conv_v_state + ((size_t)slot * value_dim + ch) * 3;
        const float hist[3] = {sp[0], sp[1], sp[2]};
        const __half *wp = w + (2 * key_dim + ch) * 4;
        const float xt = gdn16_to_float(v_raw[b * value_dim + ch]);
        float sum = xt * gdn16_to_float(wp[3]);
        for (int k = 0; k < 3; ++k) sum = __fmaf_rn(hist[k], gdn16_to_float(wp[k]), sum);
        sum /= (1.0f + __expf(-sum));
        sp[0] = hist[1];
        sp[1] = hist[2];
        sp[2] = xt;
        v_c = sum;
    }

    // ---- q/k l2norm(每线程一元素;两行独立归约,一趟 sync 双结果)----
    const float qv = gdn16_to_float(q_c[b * key_dim + kh * kd + tid]);
    const float kv_ = gdn16_to_float(k_c[b * key_dim + kh * kd + tid]);
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    float sum_q = qv * qv;
    float sum_k = kv_ * kv_;
    for (int off = 16; off > 0; off >>= 1) {
        sum_q += __shfl_down_sync(0xffffffffu, sum_q, off);
        sum_k += __shfl_down_sync(0xffffffffu, sum_k, off);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_q;
        warp_sums[warp_id + 4] = sum_k;
    }
    __syncthreads();
    if (tid == 0) {
        float tq = 0.0f, tk = 0.0f;
        for (unsigned int wI = 0; wI < blockDim.x / 32; ++wI) {
            tq += warp_sums[wI];
            tk += warp_sums[wI + 4];
        }
        warp_sums[0] = tq;
        warp_sums[1] = tk;
    }
    __syncthreads();
    const float inv_q = rsqrtf(fmaxf(warp_sums[0], 0.0f) + eps_l2);
    const float inv_k = rsqrtf(fmaxf(warp_sums[1], 0.0f) + eps_l2);
    q_smem[tid] = qv * inv_q;
    k_smem[tid] = kv_ * inv_k;
    __syncthreads();

    // ---- gating g + beta(寄存器冗余,免同步;公式 = gating_g_f16)----
    const size_t gi = b * nv + h;
    const float gx = gdn16_to_float(a_gate[gi]) + gdn16_to_float(dt_bias[h]);
    const float sp = (gx <= 20.0f) ? log1pf(expf(gx)) : gx;
    const float decay = expf(-expf(gdn16_to_float(a_log[h])) * sp);
    const float bv = gdn16_to_float(b_gate[gi]);
    const float beta_h = 1.0f / (1.0f + __expf(-bv));

    // ---- delta 步(线程 = v 列 tid;s_buf[kd] 寄存器;公式 = delta_dec_f16)----
    float *state_head = valid ? rec_state + ((size_t)slot * nv + h) * kd * vd : nullptr;
    float s_buf[128];
    for (size_t j = 0; j < kd; ++j)
        s_buf[j] = valid ? state_head[j * vd + tid] : 0.0f;
    float kv_mem = 0.0f;
    for (size_t j = 0; j < kd; ++j) {
        s_buf[j] *= decay;
        kv_mem = __fmaf_rn(s_buf[j], k_smem[j], kv_mem);
    }
    const float delta = (v_c - kv_mem) * beta_h;
    float y_reg = 0.0f;
    for (size_t j = 0; j < kd; ++j) {
        s_buf[j] = __fmaf_rn(k_smem[j], delta, s_buf[j]);
        y_reg = __fmaf_rn(s_buf[j], q_smem[j] * q_scale, y_reg);
    }
    if (valid) {
        for (size_t j = 0; j < kd; ++j)
            state_head[j * vd + tid] = s_buf[j];
    }
    // ---- norm_act(块归一 + ×gamma × silu(z);公式 = norm_act_f16 act=0 支)----
    // ⚠️ 激活 = silu(旧链 norm_act act_silu=true);2026-10-03 立案定谳:
    // 初版误写 sigmoid(单测 host 参考同错相消全绿 → 引擎域 y 逐通道错
    // 1~10 倍,状态写全对而文本乱码——「状态对而 y 错」定位于此)
    // 归约按实际 warp 数(blockDim 可为 32/64 —— warp_sums 高槽残留 l2norm
    // 的 k 行平方和,读满 8 槽即污染;教训:nwarps 数进公式,勿抄 8 槽常量)
    float nsq = y_reg * y_reg;
    for (int off = 16; off > 0; off >>= 1)
        nsq += __shfl_down_sync(0xffffffffu, nsq, off);
    const unsigned int nwarps = blockDim.x / 32;
    if (lane_id == 0) warp_sums[warp_id] = nsq;
    __syncthreads();
    if (tid == 0) {
        float ntotal = 0.0f;
        for (unsigned int wI = 0; wI < nwarps; ++wI) ntotal += warp_sums[wI];
        warp_sums[0] = ntotal;
    }
    __syncthreads();
    const float ninv = rsqrtf(fmaxf(warp_sums[0] / (float)vd, 0.0f) + eps_norm);
    const float zv = gdn16_to_float(z[b * value_dim + h * vd + tid]);
    const float actv = zv / (1.0f + __expf(-zv));
    out[(b * nv + h) * vd + tid] =
        gdn16_from_float(y_reg * ninv * gdn16_to_float(norm_w[tid]) * actv);
}

// ============================================================================
// D1-v2:decode 融合核 v2(2026-10-04 sglang 刺探产物;delta 相重写)
// ----------------------------------------------------------------------------
// v1 病灶(实测 28.4µs = 221 GB/s):状态列映射「线程 owning VD 列」,
// 每线程 128 次串行 4B 装载(列步进 512B),装载发射率受限 → 延迟饥饿。
// sglang Triton 同形核实测 7.5µs(844 GB/s)→ 3.8× 差距 × 48 层 = 1.0ms/步。
//
// v2 映射:warp w = 行组 {w, w+4, ..., w+124}(32 行),lane = float4 列组
// (4 列;32 lane × 4 = vd)—— 装载 16B/lane 合并 512B/warp,装载指令数 /4;
// 跨 warp 的 (kS)_c 部分和经 shared [4][vd] 归并(delta 需全行和,数学同
// delta_dec:kv_mem = Σ_j S[j][c]·k[j] 列点积,decay 先行);y 同法归并。
// v-conv / qk-l2norm / gating / norm_act 数学逐式镜像 v1(公式零改动)。
// 契约与 v1 完全一致(13 Block + 5 sz + 3 f32 + out);smem ≈ 5KB 免 opt-in。
// ----
extern "C" __global__ void owl_gdn_decode_step_v2_f16(
    const __half *__restrict__ q_c,
    const __half *__restrict__ k_c,
    const __half *__restrict__ v_raw,
    const __half *__restrict__ z,
    const __half *__restrict__ b_gate,
    const __half *__restrict__ a_gate,
    const __half *__restrict__ w,
    float *__restrict__ conv_v_state,
    const __half *__restrict__ a_log,
    const __half *__restrict__ dt_bias,
    float *__restrict__ rec_state,
    const float *__restrict__ slots,
    const __half *__restrict__ norm_w,
    size_t batch, size_t nv, size_t nk, size_t kd, size_t vd,
    float eps_l2, float eps_norm, float q_scale,
    __half *__restrict__ out) {
    const unsigned int d = (unsigned int)vd;
    if (d != blockDim.x || kd != vd || d == 0 || d > 128 || (d & 3)) return;
    const size_t bh = blockIdx.x;
    const size_t b = bh / nv;
    const size_t h = bh % nv;
    const size_t kv_group = nv / nk;
    const size_t kh = h / kv_group;
    const int slot = (int)slots[b];
    const bool valid = slot >= 0;
    const unsigned int tid = threadIdx.x;
    const size_t key_dim = kd * nk;
    const size_t value_dim = vd * nv;
    const size_t conv_dim = 2 * key_dim + value_dim;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    const size_t c0 = (size_t)lane_id * 4;   // 本 lane 的 4 连列

    __shared__ float warp_sums[8];
    __shared__ float q_smem[128];
    __shared__ float k_smem[128];
    __shared__ float part[4][128];  // (kS)_c 跨 warp 部分和
    __shared__ float v_smem[128];   // v-conv 产物(delta 相按列组读)

    // ---- v 段 conv(线程 tid = 头内通道;与 v1 逐式同源)----
    {
        const size_t ch = h * vd + tid;
        float vc = 0.0f;
        if (valid) {
            float *sp = conv_v_state + ((size_t)slot * value_dim + ch) * 3;
            const float hist[3] = {sp[0], sp[1], sp[2]};
            const __half *wp = w + (2 * key_dim + ch) * 4;
            const float xt = gdn16_to_float(v_raw[b * value_dim + ch]);
            float sum = xt * gdn16_to_float(wp[3]);
            for (int k = 0; k < 3; ++k) sum = __fmaf_rn(hist[k], gdn16_to_float(wp[k]), sum);
            sum /= (1.0f + __expf(-sum));
            sp[0] = hist[1];
            sp[1] = hist[2];
            sp[2] = xt;
            vc = sum;
        }
        v_smem[tid] = vc;
    }

    // ---- q/k l2norm(与 v1 同式;两行独立归约一趟 sync)----
    const float qv = gdn16_to_float(q_c[b * key_dim + kh * kd + tid]);
    const float kv_ = gdn16_to_float(k_c[b * key_dim + kh * kd + tid]);
    float sum_q = qv * qv;
    float sum_k = kv_ * kv_;
    for (int off = 16; off > 0; off >>= 1) {
        sum_q += __shfl_down_sync(0xffffffffu, sum_q, off);
        sum_k += __shfl_down_sync(0xffffffffu, sum_k, off);
    }
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_q;
        warp_sums[warp_id + 4] = sum_k;
    }
    __syncthreads();
    if (tid == 0) {
        float tq = 0.0f, tk = 0.0f;
        for (unsigned int wI = 0; wI < blockDim.x / 32; ++wI) {
            tq += warp_sums[wI];
            tk += warp_sums[wI + 4];
        }
        warp_sums[0] = tq;
        warp_sums[1] = tk;
    }
    __syncthreads();
    const float inv_q = rsqrtf(fmaxf(warp_sums[0], 0.0f) + eps_l2);
    const float inv_k = rsqrtf(fmaxf(warp_sums[1], 0.0f) + eps_l2);
    q_smem[tid] = qv * inv_q;
    k_smem[tid] = kv_ * inv_k;
    __syncthreads();

    // ---- gating g + beta(寄存器冗余;公式 = gating_g_f16)----
    const size_t gi = b * nv + h;
    const float gx = gdn16_to_float(a_gate[gi]) + gdn16_to_float(dt_bias[h]);
    const float sp = (gx <= 20.0f) ? log1pf(expf(gx)) : gx;
    const float decay = expf(-expf(gdn16_to_float(a_log[h])) * sp);
    const float bv = gdn16_to_float(b_gate[gi]);
    const float beta_h = 1.0f / (1.0f + __expf(-bv));

    // ---- delta 相 v2:行组 warp × float4 列组 ----
    float *state_head = valid ? rec_state + ((size_t)slot * nv + h) * kd * vd : nullptr;
    float4 s_reg[32];
    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
    if (valid) {
        const float4 *sp4 = reinterpret_cast<const float4 *>(state_head + c0);
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            const size_t j = (size_t)(warp_id + r * 4);
            float4 s = *(sp4 + (size_t)j * (vd >> 2));
            s.x *= decay; s.y *= decay; s.z *= decay; s.w *= decay;
            s_reg[r] = s;
            const float kj = k_smem[j];
            p0 = __fmaf_rn(s.x, kj, p0);
            p1 = __fmaf_rn(s.y, kj, p1);
            p2 = __fmaf_rn(s.z, kj, p2);
            p3 = __fmaf_rn(s.w, kj, p3);
        }
    }
    part[warp_id][c0 + 0] = p0;
    part[warp_id][c0 + 1] = p1;
    part[warp_id][c0 + 2] = p2;
    part[warp_id][c0 + 3] = p3;
    __syncthreads();
    const float kv0 = part[0][c0 + 0] + part[1][c0 + 0] + part[2][c0 + 0] + part[3][c0 + 0];
    const float kv1 = part[0][c0 + 1] + part[1][c0 + 1] + part[2][c0 + 1] + part[3][c0 + 1];
    const float kv2 = part[0][c0 + 2] + part[1][c0 + 2] + part[2][c0 + 2] + part[3][c0 + 2];
    const float kv3 = part[0][c0 + 3] + part[1][c0 + 3] + part[2][c0 + 3] + part[3][c0 + 3];
    const float d0 = (v_smem[c0 + 0] - kv0) * beta_h;
    const float d1 = (v_smem[c0 + 1] - kv1) * beta_h;
    const float d2 = (v_smem[c0 + 2] - kv2) * beta_h;
    const float d3 = (v_smem[c0 + 3] - kv3) * beta_h;

    // 更新 + 写回 + y 累加(寄存器;行组内 y 部分和)
    float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
    __shared__ float y_part[4][128];
    if (valid) {
        const float4 *sp4 = reinterpret_cast<const float4 *>(state_head + c0);
        float4 *dp4 = reinterpret_cast<float4 *>(state_head + c0);
#pragma unroll
        for (int r = 0; r < 32; ++r) {
            const size_t j = (size_t)(warp_id + r * 4);
            const float kj = k_smem[j];
            const float qj = q_smem[j] * q_scale;
            float4 s = s_reg[r];
            s.x = __fmaf_rn(kj, d0, s.x);
            s.y = __fmaf_rn(kj, d1, s.y);
            s.z = __fmaf_rn(kj, d2, s.z);
            s.w = __fmaf_rn(kj, d3, s.w);
            s_reg[r] = s;
            y0 = __fmaf_rn(s.x, qj, y0);
            y1 = __fmaf_rn(s.y, qj, y1);
            y2 = __fmaf_rn(s.z, qj, y2);
            y3 = __fmaf_rn(s.w, qj, y3);
            *(dp4 + (size_t)j * (vd >> 2)) = s;
        }
    }
    y_part[warp_id][c0 + 0] = valid ? y0 : 0.0f;
    y_part[warp_id][c0 + 1] = valid ? y1 : 0.0f;
    y_part[warp_id][c0 + 2] = valid ? y2 : 0.0f;
    y_part[warp_id][c0 + 3] = valid ? y3 : 0.0f;
    __syncthreads();

    // ---- norm_act(列线程 c 读全和 y;归约/ninv/门控与 v1 同式)----
    const float y_reg = y_part[0][tid] + y_part[1][tid] + y_part[2][tid] + y_part[3][tid];
    float nsq = y_reg * y_reg;
    for (int off = 16; off > 0; off >>= 1)
        nsq += __shfl_down_sync(0xffffffffu, nsq, off);
    const unsigned int nwarps = blockDim.x / 32;
    if (lane_id == 0) warp_sums[warp_id] = nsq;
    __syncthreads();
    if (tid == 0) {
        float ntotal = 0.0f;
        for (unsigned int wI = 0; wI < nwarps; ++wI) ntotal += warp_sums[wI];
        warp_sums[0] = ntotal;
    }
    __syncthreads();
    const float ninv = rsqrtf(fmaxf(warp_sums[0] / (float)vd, 0.0f) + eps_norm);
    const float zv = gdn16_to_float(z[b * value_dim + h * vd + tid]);
    const float actv = zv / (1.0f + __expf(-zv));
    out[(b * nv + h) * vd + tid] =
        gdn16_from_float(y_reg * ninv * gdn16_to_float(norm_w[tid]) * actv);
}
