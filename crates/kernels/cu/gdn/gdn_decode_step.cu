// ============================================================================
// GDN 核族:decode 整链融合 v1/v2(v-conv+l2norm×2+gating+sigmoid+delta+norm_act 六算子一发)(2026-10-13 C 轮自 cu/owl/gdn.cu 按**算子族**拆出 ——
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
