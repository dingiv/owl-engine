// ============================================================================
// owl_int8_decode_paged(2026-11-03,INT8 KV 生产接线)
//
// 单 token decode:q int8(per token/head scale)+ K int8(per token/head scale)
// + V int8 packed(per page/head/64-dim scale)。无 V f16 shadow。
//
// 形态:一个 block = (batch, q_head);256 线程。
//   * QK:8 lane 一组算一个 key,32 个 key 并行;组内 shfl 归约。
//   * softmax:32 个 score 落 shared,thread0 串行做 block 级 max/sum(32 项)。
//   * PV:每线程一个 dim(256),沿 32 个 key 累加。
//   * 在线 softmax:维护 running max/sum,换页时 acc 乘 alpha。
//
// 这是**正确性优先**的首版 decode(未做 KV split / 多 block 并行);
// 性能 profile 见 docs/devlog/kernels/int8-kv-attention-project.md。
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace {
constexpr int kD = 256;
constexpr int kThreads = 256;
constexpr int kKeys = 32;  // page

__device__ __forceinline__ int8_t unpack_v(uint16_t word, int off) {
  const uint8_t b = (off & 1) ? static_cast<uint8_t>(word >> 8)
                              : static_cast<uint8_t>(word & 0xFFu);
  return static_cast<int8_t>(b);
}
}  // namespace

extern "C" __global__ void owl_int8_decode_paged(
    const int8_t* __restrict__ q,        // [B*Hq, D]
    const float* __restrict__ qscale,    // [B*Hq]
    const int8_t* __restrict__ k,        // [nb, page, Hkv, D]
    const float* __restrict__ kscale,    // [nb, page, Hkv]
    const uint16_t* __restrict__ vp,     // [nb, Hkv, D, page/2]
    const float* __restrict__ vscale,    // [nb, Hkv, 4]
    const float* __restrict__ block_table,  // [B, bt_stride] f32 物理页号
    const float* __restrict__ kv_lens,   // [B] 含当前 token
    float* __restrict__ out,             // [B, Hq, D]
    int B,
    int Hq,
    int Hkv,
    int page,
    int bt_stride,
    const __half* __restrict__ dep_k,
    const __half* __restrict__ dep_v) {
  (void)dep_k;
  (void)dep_v;
  if (page != kKeys) {
    return;  // 契约:page=32;驱动层已按此选核
  }
  __shared__ float s_score[kKeys];
  __shared__ float s_p[kKeys];
  __shared__ float s_m;
  __shared__ float s_l;
  __shared__ float s_alpha;

  const int h = blockIdx.x;
  const int b = blockIdx.y;
  if (b >= B || h >= Hq) {
    return;
  }
  const int kh = h / (Hq / Hkv);
  const int len = static_cast<int>(kv_lens[b]);
  if (len <= 0) {
    return;
  }
  const int tid = threadIdx.x;
  const int group = tid >> 3;   // 0..31
  const int lane8 = tid & 7;    // 0..7
  const bool is_v_thread = tid < kD;

  const int8_t* qrow = q + static_cast<long long>(b * Hq + h) * kD;
  const float qsc = qscale[b * Hq + h];
  const int pages = (len + page - 1) / page;
  const int PP = page / 2;

  float acc = 0.0f;
  if (tid == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int pg = 0; pg < pages; ++pg) {
    const int phys = static_cast<int>(block_table[b * bt_stride + pg]);
    const int base = pg * page;

    float dot = 0.0f;
    const int key = base + group;
    if (key < len) {
      const int8_t* krow =
          k + ((static_cast<long long>(phys) * page + group) * Hkv + kh) * kD;
      for (int d = lane8; d < kD; d += 8) {
        dot += static_cast<float>(qrow[d]) * static_cast<float>(krow[d]);
      }
    }
    dot += __shfl_down_sync(0xffffffffu, dot, 4, 8);
    dot += __shfl_down_sync(0xffffffffu, dot, 2, 8);
    dot += __shfl_down_sync(0xffffffffu, dot, 1, 8);
    if (lane8 == 0) {
      if (key < len) {
        const float ksc =
            kscale[((static_cast<long long>(phys) * page + group) * Hkv) + kh];
        s_score[group] = dot * qsc * ksc;
      } else {
        s_score[group] = -CUDART_INF_F;
      }
    }
    __syncthreads();

    if (tid == 0) {
      float bm = -CUDART_INF_F;
      for (int i = 0; i < kKeys; ++i) {
        bm = fmaxf(bm, s_score[i]);
      }
      const float nm = fmaxf(s_m, bm);
      const float alpha = (s_m == -CUDART_INF_F) ? 0.0f : expf(s_m - nm);
      float sum = 0.0f;
      for (int i = 0; i < kKeys; ++i) {
        const float p = (s_score[i] == -CUDART_INF_F) ? 0.0f
                                                      : expf(s_score[i] - nm);
        s_p[i] = p;
        sum += p;
      }
      s_alpha = alpha;
      s_m = nm;
      s_l = s_l * alpha + sum;
    }
    __syncthreads();

    if (is_v_thread) {
      const float alpha = s_alpha;
      const int d = tid;
      const int vg = d >> 6;
      const float vs = vscale[((static_cast<long long>(phys) * Hkv + kh) * 4) + vg];
      acc *= alpha;
      for (int i = 0; i < kKeys; ++i) {
        const float p = s_p[i];
        if (p == 0.0f) {
          continue;
        }
        const uint16_t word =
            vp[(((static_cast<long long>(phys) * Hkv + kh) * kD + d) * PP) + (i >> 1)];
        acc += p * static_cast<float>(unpack_v(word, i)) * vs;
      }
    }
    __syncthreads();
  }

  if (is_v_thread) {
    const float inv = (s_l > 0.0f) ? 1.0f / s_l : 0.0f;
    out[((static_cast<long long>(b) * Hq + h) * kD) + tid] = acc * inv;
  }
}

// ============================================================================
// Split decode(2026-11-03):把每头的 KV 页按 nparts 切给多个 block,
// 每块出 partial (acc, m, l),再由 reduce 合并。解决"32 block 打满
// 不了 80 SM"的 decode 并行度。
// ============================================================================
extern "C" __global__ void owl_int8_decode_paged_split(
    const int8_t* __restrict__ q,
    const float* __restrict__ qscale,
    const int8_t* __restrict__ k,
    const float* __restrict__ kscale,
    const uint16_t* __restrict__ vp,
    const float* __restrict__ vscale,
    const float* __restrict__ block_table,
    const float* __restrict__ kv_lens,
    float* __restrict__ partial_out,   // [B,Hq,nparts,D]
    float* __restrict__ partial_stat,  // [B,Hq,nparts,2] = (m,l)
    __half* __restrict__ out_dummy,
    int B,
    int Hq,
    int Hkv,
    int page,
    int bt_stride,
    int nparts,
    const __half* __restrict__ dep_k,
    const __half* __restrict__ dep_v) {
  (void)out_dummy;
  (void)dep_k;
  (void)dep_v;
  if (page != kKeys) {
    return;
  }
  __shared__ float s_score[kKeys];
  __shared__ float s_p[kKeys];
  __shared__ float s_m;
  __shared__ float s_l;
  __shared__ float s_alpha;

  const int h = blockIdx.x;
  const int b = blockIdx.y;
  const int part = blockIdx.z;
  if (b >= B || h >= Hq || part >= nparts) {
    return;
  }
  const int kh = h / (Hq / Hkv);
  const int len = static_cast<int>(kv_lens[b]);
  const int tid = threadIdx.x;
  const int group = tid >> 3;
  const int lane8 = tid & 7;
  const bool is_v_thread = tid < kD;
  const int PP = page / 2;

  const int pages = (len + page - 1) / page;
  const int per = (pages + nparts - 1) / nparts;
  const int pg0 = part * per;
  const int pg1 = min(pages, pg0 + per);
  const int out_base = ((b * Hq + h) * nparts + part) * kD;
  const int stat_base = ((b * Hq + h) * nparts + part) * 2;

  if (pg0 >= pages) {
    if (is_v_thread) {
      partial_out[out_base + tid] = 0.0f;
    }
    if (tid == 0) {
      partial_stat[stat_base] = -CUDART_INF_F;
      partial_stat[stat_base + 1] = 0.0f;
    }
    return;
  }

  const int8_t* qrow = q + static_cast<long long>(b * Hq + h) * kD;
  const float qsc = qscale[b * Hq + h];

  float acc = 0.0f;
  if (tid == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int pg = pg0; pg < pg1; ++pg) {
    const int phys = static_cast<int>(block_table[b * bt_stride + pg]);
    const int base = pg * page;
    float dot = 0.0f;
    const int key = base + group;
    if (key < len) {
      const int8_t* krow =
          k + ((static_cast<long long>(phys) * page + group) * Hkv + kh) * kD;
      for (int d = lane8; d < kD; d += 8) {
        dot += static_cast<float>(qrow[d]) * static_cast<float>(krow[d]);
      }
    }
    dot += __shfl_down_sync(0xffffffffu, dot, 4, 8);
    dot += __shfl_down_sync(0xffffffffu, dot, 2, 8);
    dot += __shfl_down_sync(0xffffffffu, dot, 1, 8);
    if (lane8 == 0) {
      if (key < len) {
        const float ksc =
            kscale[((static_cast<long long>(phys) * page + group) * Hkv) + kh];
        s_score[group] = dot * qsc * ksc;
      } else {
        s_score[group] = -CUDART_INF_F;
      }
    }
    __syncthreads();

    if (tid == 0) {
      float bm = -CUDART_INF_F;
      for (int i = 0; i < kKeys; ++i) {
        bm = fmaxf(bm, s_score[i]);
      }
      const float nm = fmaxf(s_m, bm);
      const float alpha = (s_m == -CUDART_INF_F) ? 0.0f : expf(s_m - nm);
      float sum = 0.0f;
      for (int i = 0; i < kKeys; ++i) {
        const float p = (s_score[i] == -CUDART_INF_F) ? 0.0f
                                                      : expf(s_score[i] - nm);
        s_p[i] = p;
        sum += p;
      }
      s_alpha = alpha;
      s_m = nm;
      s_l = s_l * alpha + sum;
    }
    __syncthreads();

    if (is_v_thread) {
      const float alpha = s_alpha;
      const int d = tid;
      const int vg = d >> 6;
      const float vs = vscale[((static_cast<long long>(phys) * Hkv + kh) * 4) + vg];
      acc *= alpha;
      for (int i = 0; i < kKeys; ++i) {
        const float p = s_p[i];
        if (p == 0.0f) {
          continue;
        }
        const uint16_t word =
            vp[(((static_cast<long long>(phys) * Hkv + kh) * kD + d) * PP) + (i >> 1)];
        acc += p * static_cast<float>(unpack_v(word, i)) * vs;
      }
    }
    __syncthreads();
  }

  if (is_v_thread) {
    partial_out[out_base + tid] = acc;
  }
  if (tid == 0) {
    partial_stat[stat_base] = s_m;
    partial_stat[stat_base + 1] = s_l;
  }
}

extern "C" __global__ void owl_int8_decode_reduce(
    const float* __restrict__ partial_out,
    const float* __restrict__ partial_stat,
    const __half* __restrict__ dep,
    float* __restrict__ out,
    int B,
    int Hq,
    int nparts,
    int D) {
  (void)dep;
  const int h = blockIdx.x;
  const int b = blockIdx.y;
  if (b >= B || h >= Hq || threadIdx.x >= D) {
    return;
  }
  const int d = threadIdx.x;
  float gm = -CUDART_INF_F;
  for (int p = 0; p < nparts; ++p) {
    gm = fmaxf(gm, partial_stat[((b * Hq + h) * nparts + p) * 2]);
  }
  float acc = 0.0f;
  float sum = 0.0f;
  for (int p = 0; p < nparts; ++p) {
    const int stat = ((b * Hq + h) * nparts + p) * 2;
    const float m = partial_stat[stat];
    const float l = partial_stat[stat + 1];
    const float w = (m == -CUDART_INF_F) ? 0.0f : expf(m - gm);
    acc += partial_out[((b * Hq + h) * nparts + p) * D + d] * w;
    sum += l * w;
  }
  out[((static_cast<long long>(b) * Hq + h) * D) + d] =
      (sum > 0.0f) ? acc / sum : 0.0f;
}
