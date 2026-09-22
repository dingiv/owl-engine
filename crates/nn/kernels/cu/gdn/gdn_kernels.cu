// owl gdn kernels —— K3:GDN 线性注意力配套核(第一刀)
// 出处: vendor/attention.rs rev c0f19f2 src/kernels/src/gdn.cu
//       (fused_gdn_gating / l2_norm_last_dim / gated_rmsnorm_act_mul;
//        port 专项见 docs/arch/attention-kernel-port.md)
// owl 改编(2026-09-22, A4 所有权转移):
//   - nvrtc JIT 禁 host launcher(vLLM 原生 DISPATCH 宏是 host 端
//     <<<>>> 包装)——按 dtype 宏展开完整 __global__,launch 归 Rust(K1 同款);
//   - l2norm warp/block 双变体合并为单核,按 blockDim.x 分派
//     (Rust 侧已按 dim 选 grid/block);
//   - FP8 分支不搬(marlin-ffi 归属)。

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---- dtype 桥 ----
template <typename T> __device__ __forceinline__ float gdn_to_float(T x);
template <> __device__ __forceinline__ float gdn_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float gdn_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float gdn_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T gdn_from_float(float x);
template <> __device__ __forceinline__ float gdn_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half gdn_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 gdn_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

__device__ __forceinline__ float gdn_silu(float x) { return x / (1.0f + __expf(-x)); }

// ---- 三核 × 三 dtype 宏展开(完整 __global__,无 launcher、无模板)----

#define GDN_GATING_KERNEL(T, SUFFIX)                                                   \
extern "C" __global__ void gdn_fused_gating_##SUFFIX(                                  \
    const float *__restrict__ a_log,                                                   \
    const T *__restrict__ a, const T *__restrict__ b,                                  \
    const float *__restrict__ dt_bias,                                                 \
    float *__restrict__ g, float *__restrict__ beta,                                   \
    int total_elements, int num_heads) {                                               \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                                   \
    if (idx >= total_elements) return;                                                 \
    int h_idx = idx % num_heads;                                                       \
    const float a_f = gdn_to_float(a[idx]) + dt_bias[h_idx];                           \
    const float b_f = gdn_to_float(b[idx]);                                            \
    float sp = a_f;                                                                    \
    if (sp < 20.0f) sp = log1pf(expf(sp));                                             \
    g[idx] = -__expf(a_log[h_idx]) * sp;                                               \
    beta[idx] = 1.0f / (1.0f + __expf(-b_f));                                          \
}

#define GDN_L2NORM_WARP_KERNEL(T, SUFFIX)                                              \
extern "C" __global__ void gdn_l2norm_warp_##SUFFIX(                                   \
    const T *__restrict__ input, T *__restrict__ output,                               \
    int rows, int dim, float eps) {                                                    \
    const int row = blockIdx.x * 8 + (threadIdx.x / 32);                               \
    if (row >= rows) return;                                                           \
    const int lane_id = threadIdx.x % 32;                                              \
    const T *in_row = input + row * dim;                                               \
    T *out_row = output + row * dim;                                                   \
    float sumsq = 0.0f;                                                                \
    for (int i = lane_id; i < dim; i += 32) {                                          \
        const float v = gdn_to_float(in_row[i]);                                       \
        sumsq = __fmaf_rn(v, v, sumsq);                                                \
    }                                                                                  \
    for (int offset = 16; offset > 0; offset >>= 1)                                    \
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);                          \
    const float row_sumsq = __shfl_sync(0xffffffff, sumsq, 0);                         \
    const float inv = rsqrtf(fmaxf(row_sumsq, 0.0f) + eps);                            \
    for (int i = lane_id; i < dim; i += 32)                                            \
        out_row[i] = gdn_from_float<T>(gdn_to_float(in_row[i]) * inv);                 \
}

#define GDN_L2NORM_BLOCK_KERNEL(T, SUFFIX)                                             \
extern "C" __global__ void gdn_l2norm_block256_##SUFFIX(                               \
    const T *__restrict__ input, T *__restrict__ output,                               \
    int rows, int dim, float eps) {                                                    \
    const int row = blockIdx.x;                                                        \
    if (row >= rows) return;                                                           \
    const int tid = threadIdx.x;                                                       \
    const T *in_row = input + row * dim;                                               \
    T *out_row = output + row * dim;                                                   \
    float sumsq = 0.0f;                                                                \
    for (int i = tid; i < dim; i += 256) {                                             \
        const float v = gdn_to_float(in_row[i]);                                       \
        sumsq = __fmaf_rn(v, v, sumsq);                                                \
    }                                                                                  \
    for (int offset = 16; offset > 0; offset >>= 1)                                    \
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);                          \
    __shared__ float warp_sums[8];                                                     \
    const int warp_id = tid / 32;                                                      \
    const int lane_id = tid % 32;                                                      \
    if (lane_id == 0) warp_sums[warp_id] = sumsq;                                      \
    __syncthreads();                                                                   \
    float total = (tid < 8) ? warp_sums[tid] : 0.0f;                                   \
    if (warp_id == 0) {                                                                \
        for (int offset = 4; offset > 0; offset >>= 1)                                 \
            total += __shfl_down_sync(0xffffffff, total, offset);                      \
    }                                                                                  \
    if (tid == 0) warp_sums[0] = total;                                                \
    __syncthreads();                                                                   \
    const float inv = rsqrtf(fmaxf(warp_sums[0], 0.0f) + eps);                         \
    for (int i = tid; i < dim; i += 256)                                               \
        out_row[i] = gdn_from_float<T>(gdn_to_float(in_row[i]) * inv);                 \
}

#define GDN_RMSNORM_ACT_KERNEL(T, SUFFIX)                                              \
extern "C" __global__ void gdn_rmsnorm_act_##SUFFIX(                                   \
    const T *__restrict__ x, const T *__restrict__ z,                                  \
    const float *__restrict__ gamma, const float *__restrict__ bias,                   \
    T *__restrict__ out,                                                               \
    int rows, int value_dim, int group_size, float eps,                                \
    int per_group_weights, int has_bias, int act) {                                    \
    /* 二刀语义定谳(对齐 deltanet.rs 调用点):per_group_weights=true 表示*/              \
    /* gamma/bias 为 [group_size] 组内共享(deltanet 载入 (head_v_dim,));  */          \
    /* false = 全长 [value_dim] 按 group*group_size 偏移。第一刀三元反了,*/           \
    /* 本版已纠(参考 gdn.cu: wb_idx = per_group ? i : group*group_size+i)。*/        \
    /* 权重恒 float(生产加载恒 F32;数据 T 可为 f16/bf16)。                  */       \
    const int row_group = blockIdx.x;                                                  \
    const int num_groups = value_dim / group_size;                                     \
    const int row = row_group / num_groups;                                            \
    const int group = row_group % num_groups;                                          \
    const int tid = threadIdx.x;                                                       \
    if (row >= rows) return;                                                           \
    const int group_offset = row * value_dim + group * group_size;                     \
    const T *x_row = x + group_offset;                                                 \
    const T *z_row = z + group_offset;                                                 \
    T *out_row = out + group_offset;                                                   \
    const int wb_idx_base = per_group_weights ? 0 : group * group_size;                \
    float sumsq = 0.0f;                                                                \
    for (int i = tid; i < group_size; i += 256) {                                      \
        const float v = gdn_to_float(x_row[i]);                                        \
        sumsq = __fmaf_rn(v, v, sumsq);                                                \
    }                                                                                  \
    for (int offset = 16; offset > 0; offset >>= 1)                                    \
        sumsq += __shfl_down_sync(0xffffffff, sumsq, offset);                          \
    __shared__ float warp_sums[8];                                                     \
    const int warp_id = tid / 32;                                                      \
    const int lane_id = tid % 32;                                                      \
    if (lane_id == 0) warp_sums[warp_id] = sumsq;                                      \
    __syncthreads();                                                                   \
    float total = (tid < 8) ? warp_sums[tid] : 0.0f;                                   \
    if (warp_id == 0) {                                                                \
        for (int offset = 4; offset > 0; offset >>= 1)                                 \
            total += __shfl_down_sync(0xffffffff, total, offset);                      \
    }                                                                                  \
    if (tid == 0) warp_sums[0] = total;                                                \
    __syncthreads();                                                                   \
    const float inv = rsqrtf(fmaxf(warp_sums[0] / group_size, 0.0f) + eps);            \
    for (int i = tid; i < group_size; i += 256) {                                      \
        const int wb = wb_idx_base + i;                                                \
        float nx = gdn_to_float(x_row[i]) * inv * gamma[wb];                           \
        if (has_bias) nx += bias[wb];                                                  \
        const float zv = gdn_to_float(z_row[i]);                                       \
        const float actv = (act == 0) ? gdn_silu(zv)                                   \
                                      : 1.0f / (1.0f + __expf(-zv));                   \
        out_row[i] = gdn_from_float<T>(nx * actv);                                     \
    }                                                                                  \
}

GDN_GATING_KERNEL(float, f32)
GDN_GATING_KERNEL(__half, f16)
GDN_GATING_KERNEL(__nv_bfloat16, bf16)

// ---- K3 第二刀:causal_conv1d(k=4 收窄)/ gated_delta_rule(k=128 收窄)----
// 出处: vendor/attention.rs rev c0f19f2 src/kernels/src/gdn.cu
//   (causal_conv1d_fwd_varlen_kernel / causal_conv1d_update_slots_kernel /
//    gated_delta_rule_recurrence_kernel_fallback /
//    gated_delta_rule_decode_slots_gqa_kernel)
// owl 改编:
//   - conv kernel_size 恒 4、gqa decode BK 恒 128(0.8B 档:linear_conv_kernel_dim=4,
//     linear_key_head_dim=128);BV=64 与参考一致;
//   - conv_state / recurrent state 恒 F32(与参考同);
//   - int64_t → long long(nvrtc 无 stdint);slots 统一 U32(S4 owl 槽语义,
//     哨兵 0xFFFFFFFFu=跳过;原 i64 负数语义作废);state_snapshots(MTP)未搬;
//   - g 空间约定:recurrence fallback 吃实空间 decay(已 exp);gqa decode 吃
//     log 空间核内自 exp(与参考两核各自约定一致,Rust 侧注释已标明)。

#define GDN_CONV1D_FWD_K4_KERNEL(T, SUFFIX)                                            \
extern "C" __global__ void gdn_conv1d_fwd_k4_##SUFFIX(                                 \
    const T *__restrict__ x,           /* [total_tokens, d_conv] */                    \
    const T *__restrict__ weight,      /* [d_conv, 4] */                               \
    const T *__restrict__ bias,        /* [d_conv] nullable */                         \
    float *__restrict__ conv_state,    /* [batch, d_conv, 3] in/out 恒F32 */           \
    T *__restrict__ out,               /* [total_tokens, d_conv] */                    \
    const unsigned int *__restrict__ cu_seqlens, /* [batch + 1] */                     \
    int batch_size, int d_conv, int silu) {                                            \
    const int seq_idx = blockIdx.x;                                                    \
    const int channel_idx = blockIdx.y * blockDim.x + threadIdx.x;                     \
    if (seq_idx >= batch_size || channel_idx >= d_conv) return;                        \
    const int start = (int)cu_seqlens[seq_idx];                                        \
    const int end = (int)cu_seqlens[seq_idx + 1];                                      \
    const int seq_len = end - start;                                                   \
    const T *w_ptr = weight + channel_idx * 4;                                         \
    float *state_ptr = conv_state + (seq_idx * d_conv + channel_idx) * 3;              \
    float w_reg[4];                                                                    \
    for (int k = 0; k < 4; ++k) w_reg[k] = gdn_to_float(w_ptr[k]);                     \
    float history[4];                                                                  \
    for (int i = 0; i < 4; ++i) history[i] = 0.0f;                                     \
    for (int i = 0; i < 3; ++i) history[i] = state_ptr[i];                             \
    const float bias_val = (bias != nullptr) ? gdn_to_float(bias[channel_idx]) : 0.0f; \
    for (int t = 0; t < seq_len; ++t) {                                                \
        const float x_t = gdn_to_float(x[(start + t) * d_conv + channel_idx]);         \
        float sum = x_t * w_reg[3];                                                    \
        for (int k = 0; k < 3; ++k) sum = __fmaf_rn(history[k], w_reg[k], sum);        \
        if (bias != nullptr) sum += bias_val;                                          \
        if (silu) sum = gdn_silu(sum);                                                 \
        out[(start + t) * d_conv + channel_idx] = gdn_from_float<T>(sum);              \
        history[0] = history[1]; history[1] = history[2]; history[2] = x_t;            \
    }                                                                                  \
    for (int i = 0; i < 3; ++i) state_ptr[i] = history[i];                             \
}

#define GDN_CONV1D_UPD_K4_KERNEL(T, SUFFIX)                                            \
extern "C" __global__ void gdn_conv1d_upd_k4_##SUFFIX(                                 \
    const T *__restrict__ x,           /* [batch, d_conv] */                           \
    const T *__restrict__ weight,      /* [d_conv, 4] */                               \
    const T *__restrict__ bias,        /* [d_conv] nullable */                         \
    float *__restrict__ conv_state,    /* [max_batch, d_conv, 3] in/out 恒F32 */       \
    const unsigned int *__restrict__ slots, /* [batch], 0xFFFFFFFFu=跳过(S4 U32 槽语义) */                         \
    T *__restrict__ out,               /* [batch, d_conv] */                           \
    int total, int d_conv, int silu) {                                                 \
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (idx >= total) return;                                                          \
    const int batch_idx = idx / d_conv;                                                \
    const int channel_idx = idx % d_conv;                                              \
    const unsigned int slot = slots[batch_idx];                                           \
    if (slot == 0xFFFFFFFFu) return;                                                              \
    const T *w_ptr = weight + channel_idx * 4;                                         \
    float *state_ptr =                                                                 \
        conv_state + ((size_t)slot * d_conv + channel_idx) * 3;                        \
    float w_reg[4];                                                                    \
    for (int k = 0; k < 4; ++k) w_reg[k] = gdn_to_float(w_ptr[k]);                     \
    float history[4];                                                                  \
    for (int i = 0; i < 4; ++i) history[i] = 0.0f;                                     \
    for (int i = 0; i < 3; ++i) history[i] = state_ptr[i];                             \
    const float x_t = gdn_to_float(x[idx]);                                            \
    float sum = x_t * w_reg[3];                                                        \
    for (int k = 0; k < 3; ++k) sum = __fmaf_rn(history[k], w_reg[k], sum);            \
    if (bias != nullptr) sum += gdn_to_float(bias[channel_idx]);                       \
    if (silu) sum = gdn_silu(sum);                                                     \
    state_ptr[0] = history[1]; state_ptr[1] = history[2]; state_ptr[2] = x_t;          \
    out[idx] = gdn_from_float<T>(sum);                                                 \
}

/* gated_delta_rule prefill(fallback 变体):任意 k_dim <= 256;
 *   g 必须已是实空间 decay(调用方 exp 过),beta in (0,1)。
 * 布局 q/k [BH,S,K]、v/out [BH,S,V]、state [BH,K,V] 恒F32。 */
#define GDN_DELTA_REC_FB_KERNEL(T, SUFFIX)                                             \
extern "C" __global__ void gdn_delta_rec_fb_##SUFFIX(                                  \
    const T *__restrict__ q, const T *__restrict__ k,                                  \
    const T *__restrict__ v,                                                           \
    const float *__restrict__ g, const float *__restrict__ beta,                       \
    float *__restrict__ state, float *__restrict__ out,                                \
    int seq_len, int k_dim, int v_dim) {                                               \
    const int v_tile = blockIdx.x;                                                     \
    const int bh = blockIdx.y;                                                         \
    const int tid = threadIdx.x;                                                       \
    const int v_idx = v_tile * 64 + tid;                                               \
    const bool v_valid = v_idx < v_dim;                                                \
    const T *q_bh = q + bh * seq_len * k_dim;                                          \
    const T *k_bh = k + bh * seq_len * k_dim;                                          \
    const T *v_bh = v + bh * seq_len * v_dim;                                          \
    const float *g_bh = g + bh * seq_len;                                              \
    const float *beta_bh = beta + bh * seq_len;                                        \
    float *state_base = state + bh * k_dim * v_dim;                                    \
    float *out_bh = out + bh * seq_len * v_dim;                                        \
    extern __shared__ float shared[];                                                  \
    float *k_buf = shared;                                                             \
    float *q_buf = shared + k_dim;                                                     \
    float *scalars_buf = shared + 2 * k_dim;                                           \
    float s[256];                                                                      \
    for (int j = 0; j < k_dim; ++j)                                                    \
        s[j] = v_valid ? state_base[j * v_dim + v_idx] : 0.0f;                         \
    for (int t = 0; t < seq_len; ++t) {                                                \
        for (int j = tid; j < k_dim; j += 64)                                          \
            k_buf[j] = gdn_to_float(k_bh[t * k_dim + j]);                              \
        if (tid == 0) {                                                                \
            scalars_buf[0] = g_bh[t];                                                  \
            scalars_buf[1] = beta_bh[t];                                               \
        }                                                                              \
        __syncthreads();                                                               \
        const float decay = scalars_buf[0];                                            \
        const float beta_t = scalars_buf[1];                                           \
        const float v_t = v_valid ? gdn_to_float(v_bh[t * v_dim + v_idx]) : 0.0f;      \
        float kv_mem = 0.0f;                                                           \
        for (int j = 0; j < k_dim; ++j) {                                              \
            s[j] *= decay;                                                             \
            kv_mem = __fmaf_rn(s[j], k_buf[j], kv_mem);                                \
        }                                                                              \
        const float delta = (v_t - kv_mem) * beta_t;                                   \
        __syncthreads();                                                               \
        for (int j = tid; j < k_dim; j += 64)                                          \
            q_buf[j] = gdn_to_float(q_bh[t * k_dim + j]);                              \
        __syncthreads();                                                               \
        float y_t = 0.0f;                                                              \
        for (int j = 0; j < k_dim; ++j) {                                              \
            s[j] = __fmaf_rn(k_buf[j], delta, s[j]);                                   \
            y_t = __fmaf_rn(s[j], q_buf[j], y_t);                                      \
        }                                                                              \
        if (v_valid) out_bh[t * v_dim + v_idx] = y_t;                                  \
    }                                                                                  \
    for (int j = 0; j < k_dim; ++j) {                                                  \
        if (v_valid) state_base[j * v_dim + v_idx] = s[j];                             \
    }                                                                                  \
}

/* gated_delta_rule decode(单步,slot 寻址):k_dim <= 128(BK=128);
 *   g 为 log 空间(核内 exp),q 在核内乘 q_scale。
 * 布局 q/k [B,num_k_heads,K]、v/out [B,num_v_heads,V]、
 * state [max_batch,num_v_heads,K,V] 恒F32、slots [B] U32(0xFFFFFFFF=跳过)。 */
#define GDN_DELTA_DEC_GQA_KERNEL(T, SUFFIX)                                            \
extern "C" __global__ void gdn_delta_dec_gqa_##SUFFIX(                                 \
    const T *__restrict__ q, const T *__restrict__ k,                                  \
    const T *__restrict__ v,                                                           \
    const float *__restrict__ g, const float *__restrict__ beta,                       \
    float *__restrict__ state,                                                         \
    const unsigned int *__restrict__ slots,                                            \
    T *__restrict__ out,                                                               \
    int batch, int num_v_heads, int num_k_heads,                                       \
    int k_dim, int v_dim, float q_scale) {                                             \
    const int v_tile = blockIdx.x;                                                     \
    const int bh = blockIdx.y;                                                         \
    const int tid = threadIdx.x;                                                       \
    const int v_idx = v_tile * 64 + tid;                                               \
    if (bh >= batch * num_v_heads) return;                                             \
    const bool v_valid = v_idx < v_dim;                                                \
    const int b = bh / num_v_heads;                                                    \
    const int v_head_idx = bh % num_v_heads;                                           \
    const int kv_group = num_v_heads / num_k_heads;                                    \
    const int k_head_idx = v_head_idx / kv_group;                                      \
    const unsigned int slot = slots[b];                                                   \
    const bool slot_valid = slot != 0xFFFFFFFFu;                                                 \
    extern __shared__ float smem[];                                                    \
    float *q_smem = smem;                                                              \
    float *k_smem = smem + 128;                                                        \
    float *scalars = smem + 2 * 128;                                                   \
    if (tid == 0) {                                                                    \
        scalars[0] = __expf(g[b * num_v_heads + v_head_idx]);                          \
        scalars[1] = beta[b * num_v_heads + v_head_idx];                               \
    }                                                                                  \
    const T *q_bh = q + (b * num_k_heads + k_head_idx) * k_dim;                        \
    const T *k_bh = k + (b * num_k_heads + k_head_idx) * k_dim;                        \
    for (int j = tid; j < k_dim; j += 64)                                              \
        k_smem[j] = gdn_to_float(k_bh[j]);                                             \
    __syncthreads();                                                                   \
    const float decay = scalars[0];                                                    \
    const float beta_t = scalars[1];                                                   \
    float *state_head = slot_valid                                                     \
        ? state + ((size_t)slot * num_v_heads + v_head_idx) * k_dim * v_dim            \
        : nullptr;                                                                     \
    float s_buf[128];                                                                  \
    for (int j = 0; j < 128; ++j)                                                      \
        s_buf[j] = (v_valid && slot_valid && j < k_dim)                                \
            ? state_head[j * v_dim + v_idx] : 0.0f;                                    \
    float kv_mem = 0.0f;                                                               \
    for (int j = 0; j < 128; ++j) {                                                    \
        if (v_valid && slot_valid && j < k_dim) {                                      \
            s_buf[j] *= decay;                                                         \
            kv_mem = __fmaf_rn(s_buf[j], k_smem[j], kv_mem);                           \
        }                                                                              \
    }                                                                                  \
    const T *v_bh = v + (b * num_v_heads + v_head_idx) * v_dim;                        \
    const float delta = (v_valid && slot_valid)                                        \
        ? (gdn_to_float(v_bh[v_idx]) - kv_mem) * beta_t : 0.0f;                        \
    __syncthreads();                                                                   \
    for (int j = tid; j < k_dim; j += 64)                                              \
        q_smem[j] = gdn_to_float(q_bh[j]) * q_scale;                                   \
    __syncthreads();                                                                   \
    float y = 0.0f;                                                                    \
    for (int j = 0; j < 128; ++j) {                                                    \
        if (v_valid && slot_valid && j < k_dim) {                                      \
            s_buf[j] = __fmaf_rn(k_smem[j], delta, s_buf[j]);                          \
            y = __fmaf_rn(s_buf[j], q_smem[j], y);                                     \
        }                                                                              \
    }                                                                                  \
    for (int j = 0; j < 128; ++j) {                                                    \
        if (v_valid && slot_valid && j < k_dim)                                        \
            state_head[j * v_dim + v_idx] = s_buf[j];                                  \
    }                                                                                  \
    if (v_valid && slot_valid)                                                         \
        out[(b * num_v_heads + v_head_idx) * v_dim + v_idx] = gdn_from_float<T>(y);    \
}

GDN_L2NORM_WARP_KERNEL(float, f32)
GDN_L2NORM_WARP_KERNEL(__half, f16)
GDN_L2NORM_WARP_KERNEL(__nv_bfloat16, bf16)
GDN_L2NORM_BLOCK_KERNEL(float, f32)
GDN_L2NORM_BLOCK_KERNEL(__half, f16)
GDN_L2NORM_BLOCK_KERNEL(__nv_bfloat16, bf16)

GDN_RMSNORM_ACT_KERNEL(float, f32)
GDN_RMSNORM_ACT_KERNEL(__half, f16)
GDN_RMSNORM_ACT_KERNEL(__nv_bfloat16, bf16)

GDN_CONV1D_FWD_K4_KERNEL(float, f32)
GDN_CONV1D_FWD_K4_KERNEL(__half, f16)
GDN_CONV1D_FWD_K4_KERNEL(__nv_bfloat16, bf16)

GDN_CONV1D_UPD_K4_KERNEL(float, f32)
GDN_CONV1D_UPD_K4_KERNEL(__half, f16)
GDN_CONV1D_UPD_K4_KERNEL(__nv_bfloat16, bf16)

GDN_DELTA_REC_FB_KERNEL(float, f32)
GDN_DELTA_REC_FB_KERNEL(__half, f16)
GDN_DELTA_REC_FB_KERNEL(__nv_bfloat16, bf16)

GDN_DELTA_DEC_GQA_KERNEL(float, f32)
GDN_DELTA_DEC_GQA_KERNEL(__half, f16)
GDN_DELTA_DEC_GQA_KERNEL(__nv_bfloat16, bf16)

// ---- exp 就地(f32):prefill fallback 的 g log→实空间 decay 转换 ----
// (dec_gqa 核内自 exp,fb 核要求实空间入参;避免引擎侧 D2H 同步)
extern "C" __global__ void gdn_exp_inplace_f32(float *__restrict__ v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    v[i] = __expf(v[i]);
}
