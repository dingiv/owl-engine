// owl GDN 标量门 chunked delta-rule 前向(port 自 lmdeploy turbomind pre_sm90)
// ============================================================================
// 溯源:lmdeploy `src/turbomind/kernels/linear_attn/kernel/pre_sm90/chunked.cu`
// ChunkedGdrKernel(2026-10 调研定案:sm86 全社区唯一完整 CUDA 源码 GDN 前向;
// lmdeploy Qwen3.5/3.6 turbomind 在 Ampere/3090 上的现役路径,Apache-2.0)。
//
// 相对原版的简化(数学逐位同构,工程面剥除):
// 1. 状态不经 swizzled smem 转置 —— 寄存器直读直写(状态 I/O 每核一次,
//    非热路径;ThreadMap_V2/SmemLayoutV2/Swizzle 依赖随之全剥);
// 2. beta/g 固定 f32(FLA 约定,引擎 gating 产物即 f32);
// 3. out 固定 f32(引擎侧由 handler 的 cast 核转 bf16);
// 4. lmdeploy 的 physical_batch/token 拆分与 finished 短路剥除 —— owl 态
//    为扁平 token 轴 + 恒写回终态;
// 5. varlen(seq_off)保留 —— 并发批在内核层就位。
//
// 数学(逐 token,与 FLA gated_delta_rule / owl 金标同构;g = log-e 负域
// per v-head 标量门,expf 直接吃,无需 RCP_LN2 换算):
//   S ← exp(g_t)·S
//   kv = k_t·S            (k 维 warp 归约)
//   δ  = β_t·(v_t − kv)
//   S += k_t ⊗ δ
//   o_t = scale·q_t·S
// 终态 ht = S_T 原地写回 state(初态 h0 同址读 —— block 间状态片互斥,安全)。
//
// 分工:block 256 = 16(k 分组)×16(v 分组),每线程持 8×8 状态片;
// grid = (num_seqs, HV),每 block 独占一条序列一个 v-head 的 [KD×VD] 状态。
// chunk = 16(lmdeploy 原定;非 FLA 的 64 —— 本核不产 WY 中间量,chunk 仅
// 是 smem 分代粒度)。
//
// dtype 实例化:f32 金标通道(OWL_GDN_CHUNK_DEFINE_F32);f16/bf16 引擎
// 通道接线时照宏再加两行即可。
#include <cuda_runtime.h>

namespace owl_gdn {

// 发射形状(block 256 × grid (ns, HV);smem 25472B @ d=128)的单一事实源
// 在 Rust 侧 gdn_scalar::cubin(含 smem_bytes 推导),本文件不重复声明。
constexpr int kChunkSize = 16;
constexpr int kTileK = 8;
constexpr int kTileV = 8;
constexpr int kBlockThreads = 256;
constexpr int kSmemPad = 4;  // 行尾 padding 防 bank 冲突(lmdeploy 同款)

// 共享内存需求(字节):3 行块 + beta/g,handler/测试侧按此发射
constexpr int smem_bytes(int d) {
  return 3 * kChunkSize * (d + kSmemPad) * int(sizeof(float)) +
         2 * kChunkSize * int(sizeof(float));
}

// 载荷体:T = q/k/v I/O 元素类型(f32 金标;f16/bf16 引擎期再加)。
// d 仅支持 128(线程分工 16k×16v × 8×8 片按此成立)。
template <class T>
__device__ __forceinline__ void gdn_chunk_scalar_body(
    float* __restrict__ out,             // [T, HV, VD] f32
    const T* __restrict__ q,             // [T, NK, KD] 连续 token 主序
    const T* __restrict__ k,             // [T, NK, KD]
    const T* __restrict__ v,             // [T, HV, VD]
    const float* __restrict__ beta,      // [T, HV]
    const float* __restrict__ g,         // [T, HV] log-e 负域
    float* __restrict__ state,           // [num_seqs*HV, KD, VD] in/out
    const int* __restrict__ seq_off,     // [num_seqs+1]
    int nk, int hv, int d, float scale) {
  if (d != 128) {
    return;
  }
  const int k_threads = d / kTileK;             // 16
  const int v_threads = kBlockThreads / k_threads;  // 16
  const int offset_k = threadIdx.x % k_threads;
  const int offset_v = threadIdx.x / k_threads;
  (void)v_threads;

  const int seq = blockIdx.x;
  const int v_head = blockIdx.y;
  const int k_head = v_head / (hv / nk);
  const int t_begin = seq_off[seq];
  const int seq_len = seq_off[seq + 1] - t_begin;

  float* __restrict__ state_base = state + (int64_t)(seq * hv + v_head) * d * d;
  const T* q_base = q + (int64_t)t_begin * nk * d + (int64_t)k_head * d;
  const T* k_base = k + (int64_t)t_begin * nk * d + (int64_t)k_head * d;
  const T* v_base = v + (int64_t)t_begin * hv * d + (int64_t)v_head * d;
  float* __restrict__ out_base = out + (int64_t)t_begin * hv * d + (int64_t)v_head * d;
  const float* beta_base = beta + (int64_t)t_begin * hv + v_head;
  const float* g_base = g + (int64_t)t_begin * hv + v_head;

  // ---- 初态 → 寄存器(每线程 8×8 片;与终态写回同址)----
  float S[kTileK][kTileV];
#pragma unroll
  for (int ki = 0; ki < kTileK; ++ki) {
#pragma unroll
    for (int vi = 0; vi < kTileV; ++vi) {
      S[ki][vi] = state_base[(offset_k * kTileK + ki) * d + offset_v * kTileV + vi];
    }
  }

  // ---- chunk 分代 smem(q/k/v 行 + beta/g)----
  extern __shared__ __align__(16) float smem_buf[];
  const int srow = d + kSmemPad;
  float* k_smem = smem_buf;
  float* q_smem = k_smem + kChunkSize * srow;
  float* v_smem = q_smem + kChunkSize * srow;
  float* b_smem = v_smem + kChunkSize * srow;
  float* g_smem = b_smem + kChunkSize;

  // staging 分工:16 token × 16 线程 × 8 元素 = 128
  const int ld_tok = threadIdx.x / k_threads;   // 0..15
  const int ld_lane = threadIdx.x % k_threads;  // 0..15
  const int ld_elems = d / k_threads;           // 8

  const int chunk_count = (seq_len + kChunkSize - 1) / kChunkSize;
  for (int c = 0; c < chunk_count; ++c) {
    const int valid = min(kChunkSize, seq_len - c * kChunkSize);

    // ---- 装载本 chunk 的 q/k/v/beta/g ----
    if (ld_tok < valid) {
      const int gt = c * kChunkSize + ld_tok;  // 序列内 token
      const T* qp = q_base + (int64_t)gt * nk * d;
      const T* kp = k_base + (int64_t)gt * nk * d;
      const T* vp = v_base + (int64_t)gt * hv * d;
#pragma unroll
      for (int e = 0; e < ld_elems; ++e) {
        const int dd = ld_lane * ld_elems + e;
        k_smem[ld_tok * srow + dd] = float(kp[dd]);
        q_smem[ld_tok * srow + dd] = float(qp[dd]);
        v_smem[ld_tok * srow + dd] = float(vp[dd]);
      }
      if (ld_lane == 0) {
        b_smem[ld_tok] = beta_base[(int64_t)gt * hv];
        g_smem[ld_tok] = g_base[(int64_t)gt * hv];
      }
    }
    __syncthreads();

    // ---- 逐 token delta-rule(全员步进,状态片驻寄存器)----
    for (int tok = 0; tok < valid; ++tok) {
      const float decay = expf(g_smem[tok]);
      const float b = b_smem[tok];
      float vec_k[kTileK];
      float vec_q[kTileK];
#pragma unroll
      for (int ki = 0; ki < kTileK; ++ki) {
        vec_k[ki] = k_smem[tok * srow + offset_k * kTileK + ki];
        vec_q[ki] = q_smem[tok * srow + offset_k * kTileK + ki];
      }
      const int v_base = offset_v * kTileV;

      float vec_o[kTileV];
#pragma unroll
      for (int vi = 0; vi < kTileV; ++vi) {
        // S ← decay·S
#pragma unroll
        for (int ki = 0; ki < kTileK; ++ki) {
          S[ki][vi] *= decay;
        }
        // kv = k·S(warp 归约:16 lane 同 offset_v 组内,xor 8/4/2/1)
        float kv = 0.f;
#pragma unroll
        for (int ki = 0; ki < kTileK; ++ki) {
          kv += S[ki][vi] * vec_k[ki];
        }
#pragma unroll
        for (int mask = k_threads / 2; mask > 0; mask >>= 1) {
          kv += __shfl_xor_sync(0xffffffffu, kv, mask);
        }
        // δ = β·(v − kv);S += k ⊗ δ
        const float delta = (v_smem[tok * srow + v_base + vi] - kv) * b;
#pragma unroll
        for (int ki = 0; ki < kTileK; ++ki) {
          S[ki][vi] += vec_k[ki] * delta;
        }
        // o = q·S(同组归约)
        float o_val = 0.f;
#pragma unroll
        for (int ki = 0; ki < kTileK; ++ki) {
          o_val += S[ki][vi] * vec_q[ki];
        }
#pragma unroll
        for (int mask = k_threads / 2; mask > 0; mask >>= 1) {
          o_val += __shfl_xor_sync(0xffffffffu, o_val, mask);
        }
        vec_o[vi] = o_val * scale;
      }
      // offset_k==0 的 16 线程各写 8 连 f32 → 每行 128 全覆盖
      if (offset_k == 0) {
        float* o_row = out_base + (int64_t)(c * kChunkSize + tok) * hv * d;
#pragma unroll
        for (int vi = 0; vi < kTileV; ++vi) {
          o_row[v_base + vi] = vec_o[vi];
        }
      }
    }
    __syncthreads();
  }

  // ---- 终态写回(与初态同址)----
#pragma unroll
  for (int ki = 0; ki < kTileK; ++ki) {
#pragma unroll
    for (int vi = 0; vi < kTileV; ++vi) {
      state_base[(offset_k * kTileK + ki) * d + offset_v * kTileV + vi] = S[ki][vi];
    }
  }
}

}  // namespace owl_gdn

// ---- extern "C" 发射面(名字稳定,cudarc load_function 直取)----
// f32 金标通道;f16/bf16 引擎通道接线时按 body<T> 再加两条。
extern "C" __global__ void owl_gdn_chunk_scalar_f32(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ beta,
    const float* __restrict__ g,
    float* __restrict__ state,
    const int* __restrict__ seq_off,
    int nk, int hv, int d, float scale) {
  owl_gdn::gdn_chunk_scalar_body<float>(out, q, k, v, beta, g, state, seq_off,
                                        nk, hv, d, scale);
}
