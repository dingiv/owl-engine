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
