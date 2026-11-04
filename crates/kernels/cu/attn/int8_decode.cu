// ============================================================================
// owl_int8_decode_paged(2026-11-03,INT8 KV 生产接线)
//
// 单 token decode:q int8(per token/head scale)+ K int8(per token/head scale)
// + V token-major int8(per-token scale)。无 V f16 shadow。
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
#include "int8_attn_helpers.cuh"

namespace {
constexpr int kD = 256;
constexpr float kAttnScale = 1.0f / 16.0f;  // 1/sqrt(256)
constexpr int kThreads = 256;
constexpr int kKeys = 32;  // page

}  // namespace

extern "C" __global__ void owl_int8_decode_paged(
    const int8_t* __restrict__ q,        // [B*Hq, D]
    const float* __restrict__ qscale,    // [B*Hq]
    const int8_t* __restrict__ k,        // [nb, page, Hkv, D]
    const float* __restrict__ kscale,    // [nb, page, Hkv]
    const int8_t* __restrict__ v,        // [nb, page, Hkv, D]
    const float* __restrict__ vscale,    // [nb, Hkv, 32]
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
  const float* qsc = &qscale[(b * Hq + h) * 8];
  const int pages = (len + page - 1) / page;

  float acc = 0.0f;
  if (tid == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int pg = 0; pg < pages; ++pg) {
    const int phys = static_cast<int>(block_table[b * bt_stride + pg]);
    const int base = pg * page;

    float gdot[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int key = base + group;
    if (key < len) {
      const int8_t* krow =
          k + ((static_cast<long long>(phys) * page + group) * Hkv + kh) * kD;
      for (int d = lane8; d < kD; d += 8) {
        gdot[d / 32] += static_cast<float>(qrow[d]) * static_cast<float>(krow[d]);
      }
    }
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      gdot[g] += __shfl_down_sync(0xffffffffu, gdot[g], 4, 8);
      gdot[g] += __shfl_down_sync(0xffffffffu, gdot[g], 2, 8);
      gdot[g] += __shfl_down_sync(0xffffffffu, gdot[g], 1, 8);
    }
    if (lane8 == 0) {
      if (key < len) {
        float dot = 0.0f;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          dot += gdot[g] * qsc[g] *
                 kscale[((static_cast<long long>(phys) * page + group) * Hkv + kh) * 8 + g];
        }
        s_score[group] = dot * kAttnScale;
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
      const int vg = d >> 3;
      acc *= alpha;
      for (int i = 0; i < kKeys; ++i) {
        const float p = s_p[i] *
            vscale[((static_cast<long long>(phys) * page + i) * Hkv) + kh];
        if (p == 0.0f) {
          continue;
        }
        const int8_t* vrow =
            v + ((static_cast<long long>(phys) * page + i) * Hkv + kh) * kD + d;
        acc += p * static_cast<float>(*vrow);
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
    const int8_t* __restrict__ v,
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
  const float* qsc = &qscale[(b * Hq + h) * 8];

  float acc = 0.0f;
  if (tid == 0) {
    s_m = -CUDART_INF_F;
    s_l = 0.0f;
  }
  __syncthreads();

  for (int pg = pg0; pg < pg1; ++pg) {
    const int phys = static_cast<int>(block_table[b * bt_stride + pg]);
    const int base = pg * page;
    float gdot[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const int key = base + group;
    if (key < len) {
      const int8_t* krow =
          k + ((static_cast<long long>(phys) * page + group) * Hkv + kh) * kD;
      for (int d = lane8; d < kD; d += 8) {
        gdot[d / 32] += static_cast<float>(qrow[d]) * static_cast<float>(krow[d]);
      }
    }
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      gdot[g] += __shfl_down_sync(0xffffffffu, gdot[g], 4, 8);
      gdot[g] += __shfl_down_sync(0xffffffffu, gdot[g], 2, 8);
      gdot[g] += __shfl_down_sync(0xffffffffu, gdot[g], 1, 8);
    }
    if (lane8 == 0) {
      if (key < len) {
        float dot = 0.0f;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          dot += gdot[g] * qsc[g] *
                 kscale[((static_cast<long long>(phys) * page + group) * Hkv + kh) * 8 + g];
        }
        s_score[group] = dot * kAttnScale;
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
      const int vg = d >> 3;
      acc *= alpha;
      for (int i = 0; i < kKeys; ++i) {
        const float p = s_p[i] *
            vscale[((static_cast<long long>(phys) * page + i) * Hkv) + kh];
        if (p == 0.0f) {
          continue;
        }
        const int8_t* vrow =
            v + ((static_cast<long long>(phys) * page + i) * Hkv + kh) * kD + d;
        acc += p * static_cast<float>(*vrow);
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
    __half* __restrict__ out,
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
      __float2half_rn((sum > 0.0f) ? acc / sum : 0.0f);
}

namespace tc {
constexpr int D = 256;
constexpr int PAGE = 32;
constexpr int PP = PAGE / 2;

__device__ __forceinline__ int swz(int row, int col) {
  return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}
__device__ __forceinline__ int swz_p(int row, int col) {
  return (((col >> 3) ^ (row & 1)) << 3) | (col & 7);
}
__device__ __forceinline__ int swz_v(int row, int col) {
  return (((col >> 3) ^ (row & 1)) << 3) | (col & 7);
}
}  // namespace tc

// ============================================================================
// Tensor-core GQA decode split (2026-11-03 perf): stages K/V once per
// (kv_head, part), QK/PV on m16n8k32 u8s8, 8 warps x 32 dims PV.
// ============================================================================
extern "C" __global__ void owl_int8_decode_gqa_tc_split(
    const int8_t* __restrict__ q, const float* __restrict__ qscale,
    const int8_t* __restrict__ k, const float* __restrict__ kscale,
    const int8_t* __restrict__ v, const float* __restrict__ vscale,
    const float* __restrict__ block_table, const float* __restrict__ kv_lens,
    float* __restrict__ partial_out, float* __restrict__ partial_stat,
    __half* __restrict__ out_dummy,
    int B, int Hq, int Hkv, int page, int bt_stride, int nparts,
    const __half* __restrict__ dep_k, const __half* __restrict__ dep_v) {
  using namespace tc;
  using namespace owl_int8_attn;
  (void)out_dummy;
  (void)dep_k;
  (void)dep_v;
  if (page != PAGE) return;

  __shared__ __align__(16) int8_t s_q[16 * D];
  __shared__ __align__(16) int8_t s_k[PAGE * D];
  __shared__ __align__(16) uint16_t s_v[D * PP];
  __shared__ __align__(16) int8_t s_vt[PAGE * D];
  __shared__ __align__(16) uint16_t s_p_hi[16 * PP];
  __shared__ __align__(16) uint16_t s_p_mid[16 * PP];
  __shared__ __align__(16) uint16_t s_p_lo[16 * PP];
  __shared__ __align__(16) uint16_t s_p_lo2[16 * PP];
  __shared__ float s_score[8 * PAGE];
  __shared__ uint32_t s_p24[8 * PAGE];
  __shared__ float s_ps[8], s_alpha[8], s_m[8], s_l[8];
  __shared__ float s_qs[8 * 8], s_ks[8 * PAGE], s_vs[PAGE];

  const int kh = blockIdx.x;
  const int b = blockIdx.y;
  const int part = blockIdx.z;
  if (b >= B || kh >= Hkv || part >= nparts) return;
  const int grp = Hq / Hkv;
  if (grp <= 0 || grp > 8) return;

  const int len = static_cast<int>(kv_lens[b]);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int pages = (len + page - 1) / page;
  const int per = (pages + nparts - 1) / nparts;
  const int pg0 = part * per;
  const int pg1 = min(pages, pg0 + per);

  if (pg0 >= pages) {
    for (int j = 0; j < grp; ++j) {
      const int h = kh * grp + j;
      const int out_base = ((b * Hq + h) * nparts + part) * D;
      const int stat_base = ((b * Hq + h) * nparts + part) * 2;
      for (int d = tid; d < D; d += blockDim.x) partial_out[out_base + d] = 0.0f;
      if (tid == 0) {
        partial_stat[stat_base] = -CUDART_INF_F;
        partial_stat[stat_base + 1] = 0.0f;
      }
    }
    return;
  }

  // zero Q rows 6..15 + P rows 6..15, then load Q 0..grp-1.
  for (int i = tid; i < 16 * D; i += blockDim.x) s_q[i] = 0;
  for (int i = tid; i < 16 * PP; i += blockDim.x) {
    s_p_hi[i] = 0;
    s_p_mid[i] = 0;
    s_p_lo[i] = 0;
    s_p_lo2[i] = 0;
  }
  // grp<8 时 6..7 行不会被 softmax 写;先全量初始化,避免 QK/PV 读未初始化槽。
  if (tid < 8) {
    s_alpha[tid] = 0.0f;
    s_ps[tid] = 0.0f;
    s_m[tid] = -CUDART_INF_F;
    s_l[tid] = 0.0f;
  }
  for (int i = tid; i < 8 * 8; i += blockDim.x) s_qs[i] = 0.0f;
  // 清零与 Q 行加载是不同线程的写/写;必须先等清零落地,否则 Q 会被覆盖。
  __syncthreads();
  for (int i = tid; i < grp * (D / 16); i += blockDim.x) {
    const int j = i / (D / 16);
    const int off = (i % (D / 16)) * 16;
    const int h = kh * grp + j;
    const int8_t* src = q + static_cast<long long>(b * Hq + h) * D + off;
    *reinterpret_cast<int4*>(&s_q[j * D + swz(j, off / 2) * 2]) =
        *reinterpret_cast<const int4*>(src);
  }
  for (int i = tid; i < grp * 8; i += blockDim.x) {
    const int row = i / 8;
    const int g = i % 8;
    s_qs[i] = qscale[(b * Hq + kh * grp + row) * 8 + g];
  }
  if (tid < grp) {
    s_m[tid] = -CUDART_INF_F;
    s_l[tid] = 0.0f;
  }
  __syncthreads();

  const int gid = lane >> 2;
  const int lid = lane & 3;
  const int a_mat = lane >> 3;
  const int a_rin = lane & 7;
  const int a_rowoff = a_rin + ((a_mat & 1) << 3);
  const int a_coloff = (a_mat >> 1) << 3;
  const int b_rin = lane & 7;
  const int b_koff = ((lane >> 3) & 1) << 3;

  float acc[4][2];
#pragma unroll
  for (int n = 0; n < 4; ++n) acc[n][0] = acc[n][1] = 0.0f;

  for (int pg = pg0; pg < pg1; ++pg) {
    const int phys = static_cast<int>(block_table[b * bt_stride + pg]);
    const int base = pg * page;

    for (int i = tid; i < PAGE * (D / 16); i += blockDim.x) {
      const int key = i / (D / 16);
      const int off = (i % (D / 16)) * 16;
      const int8_t* src = k + ((static_cast<long long>(phys) * page + key) *
                                   Hkv * D) +
                          static_cast<long long>(kh) * D + off;
      cp_async<16, Cache::cg>(&s_k[key * D + swz(key, off / 2) * 2], src);
    }
    // V 池已是 token-major [key, Hkv, D]:先连续搬进 s_vt,
    // cp.async 完成后再在 shared 内转成 PV 所需的 dim-major b16。
    for (int i = tid; i < PAGE * (D / 16); i += blockDim.x) {
      const int key = i / (D / 16);
      const int off = (i % (D / 16)) * 16;
      const int8_t* src = v + ((static_cast<long long>(phys) * page + key) *
                                   Hkv + kh) * D + off;
      cp_async<16, Cache::cg>(&s_vt[key * D + off], src);
    }
    for (int i = tid; i < 8 * PAGE; i += blockDim.x) {
      const int g = i / PAGE;
      const int key_in = i % PAGE;
      const int key = base + key_in;
      s_ks[i] = (key < len)
                    ? kscale[((static_cast<long long>(phys) * page + key_in) * Hkv + kh) * 8 + g]
                    : 0.0f;
    }
    for (int i = tid; i < PAGE; i += blockDim.x) {
      const int key = base + i;
      s_vs[i] = (key < len)
                    ? vscale[(static_cast<long long>(phys) * page + i) * Hkv + kh]
                    : 1.0f;
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();
    for (int i = tid; i < D * (PAGE / 2); i += blockDim.x) {
      const int d = i / (PAGE / 2);
      const int pair = i % (PAGE / 2);
      const int key0 = pair * 2;
      const uint32_t b0 = static_cast<uint8_t>(s_vt[key0 * D + d]);
      const uint32_t b1 = static_cast<uint8_t>(s_vt[(key0 + 1) * D + d]);
      s_v[d * PP + swz_v(d, pair)] = static_cast<uint16_t>(b0 | (b1 << 8));
    }
    __syncthreads();

    // ---- QK: 4 warps, each 8 keys; 8 k-steps of 32 dims ----
    if (warp < 4) {
      const int nkey = warp * 8;
      const int col0 = lid * 2;
      const int key0 = nkey + col0;
      const int key1 = key0 + 1;
      float sc0 = 0.0f, sc1 = 0.0f, sc2 = 0.0f, sc3 = 0.0f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        unsigned af[4], bf[2];
        ldmatrix_x4(af[0], af[1], af[2], af[3],
                    smem_addr(&s_q[a_rowoff * D +
                                  swz(a_rowoff, kk * 16 + a_coloff) * 2]));
        ldmatrix_x2(bf[0], bf[1],
                    smem_addr(&s_k[(nkey + b_rin) * D +
                                  swz(nkey + b_rin, kk * 16 + b_koff) * 2]));
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        mma_s8(c0, c1, c2, c3, af[0], af[1], af[2], af[3], bf[0], bf[1]);
        const int g = kk;
        const float ks0 = s_ks[g * PAGE + key0];
        const float ks1 = s_ks[g * PAGE + key1];
        const float qg0 = s_qs[gid * 8 + g] * kAttnScale;
        const float qg1 = (gid + 8 < grp) ? s_qs[(gid + 8) * 8 + g] * kAttnScale : 0.0f;
        sc0 += static_cast<float>(c0) * (ks0 * qg0);
        sc1 += static_cast<float>(c1) * (ks1 * qg0);
        sc2 += static_cast<float>(c2) * (ks0 * qg1);
        sc3 += static_cast<float>(c3) * (ks1 * qg1);
      }
      const int row0 = gid;
      const int row1 = gid + 8;
      if (row0 < grp) {
        s_score[row0 * PAGE + key0] = sc0;
        s_score[row0 * PAGE + key1] = sc1;
      }
      if (row1 < grp) {
        s_score[row1 * PAGE + key0] = sc2;
        s_score[row1 * PAGE + key1] = sc3;
      }
    }
    __syncthreads();

    // ---- softmax: one warp per q head, 32 keys ----
    if (warp < grp) {
      const int h = warp;
      const int key = base + lane;
      float sc = (key < len) ? s_score[h * PAGE + lane] : -CUDART_INF_F;
      float bm = sc;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, off));
      const float old_m = s_m[h];
      __syncwarp();
      const float nm = fmaxf(old_m, bm);
      const float alpha = (old_m == -CUDART_INF_F) ? 0.0f : expf(old_m - nm);
      const float p = (sc == -CUDART_INF_F) ? 0.0f : expf(sc - nm);
      const float pv = p * s_vs[lane];
      float pmax = pv;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        pmax = fmaxf(pmax, __shfl_xor_sync(0xffffffffu, pmax, off));
      const float ps = (pmax > 0.0f) ? pmax * (1.0f / 4294967295.0f) : (1.0f / 4294967295.0f);
      const float inv = (pmax > 0.0f) ? 4294967295.0f / pmax : 0.0f;
      float sum = p;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
      unsigned code = __float2uint_rn(pv * inv);
      s_p24[h * PAGE + lane] = code;
      if (lane == 0) {
        s_ps[h] = ps;
        s_alpha[h] = alpha;
        s_m[h] = nm;
        s_l[h] = s_l[h] * alpha + sum;
      }
    }
    __syncthreads();

    // ---- pack P rows 0..grp-1 into b16 swizzled A layout ----
    if (tid < grp * PP) {
      const int row = tid / PP;
      const int pair = tid % PP;
      const uint32_t p0 = s_p24[row * PAGE + 2 * pair];
      const uint32_t p1 = s_p24[row * PAGE + 2 * pair + 1];
      const int off = row * PP + swz_p(row, pair);
      s_p_hi[off] = static_cast<uint16_t>(((p0 >> 24) & 0xff) | (((p1 >> 24) & 0xff) << 8));
      s_p_mid[off] = static_cast<uint16_t>(((p0 >> 16) & 0xff) | (((p1 >> 16) & 0xff) << 8));
      s_p_lo[off] = static_cast<uint16_t>(((p0 >> 8) & 0xff) | (((p1 >> 8) & 0xff) << 8));
      s_p_lo2[off] = static_cast<uint16_t>((p0 & 0xff) | ((p1 & 0xff) << 8));
    }
    __syncthreads();

    // ---- PV: 8 warps x 64 dims; A=P(16x32), B=V(32x8) ----
    unsigned a_hi[4], a_mid[4], a_lo[4], a_lo2[4];
    ldmatrix_x4(a_hi[0], a_hi[1], a_hi[2], a_hi[3],
                smem_addr(&s_p_hi[a_rowoff * PP + swz_p(a_rowoff, a_coloff)]));
    ldmatrix_x4(a_mid[0], a_mid[1], a_mid[2], a_mid[3],
                smem_addr(&s_p_mid[a_rowoff * PP + swz_p(a_rowoff, a_coloff)]));
    ldmatrix_x4(a_lo[0], a_lo[1], a_lo[2], a_lo[3],
                smem_addr(&s_p_lo[a_rowoff * PP + swz_p(a_rowoff, a_coloff)]));
    ldmatrix_x4(a_lo2[0], a_lo2[1], a_lo2[2], a_lo2[3],
                smem_addr(&s_p_lo2[a_rowoff * PP + swz_p(a_rowoff, a_coloff)]));
    const float alpha_h = (gid < grp) ? s_alpha[gid] : 0.0f;
    if (gid < grp) {
#pragma unroll
      for (int n = 0; n < 4; ++n) {
        acc[n][0] *= alpha_h;
        acc[n][1] *= alpha_h;
      }
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      unsigned bf[2];
      const int brow = warp * 32 + n * 8 + b_rin;
      ldmatrix_x2(bf[0], bf[1],
                  smem_addr(&s_v[brow * PP + swz_v(brow, b_koff)]));
      int h0 = 0, h1 = 0, h2 = 0, h3 = 0;
      int m0 = 0, m1 = 0, m2 = 0, m3 = 0;
      int l0 = 0, l1 = 0, l2 = 0, l3 = 0;
      int o0 = 0, o1 = 0, o2 = 0, o3 = 0;
      mma_u8s8(h0, h1, h2, h3, a_hi[0], a_hi[1], a_hi[2], a_hi[3], bf[0], bf[1]);
      mma_u8s8(m0, m1, m2, m3, a_mid[0], a_mid[1], a_mid[2], a_mid[3], bf[0], bf[1]);
      mma_u8s8(l0, l1, l2, l3, a_lo[0], a_lo[1], a_lo[2], a_lo[3], bf[0], bf[1]);
      mma_u8s8(o0, o1, o2, o3, a_lo2[0], a_lo2[1], a_lo2[2], a_lo2[3], bf[0], bf[1]);
      if (gid < grp) {
        const float scale = s_ps[gid];
        const float scale_hi = scale * 65536.0f;
        const int hi0 = h0 * 256 + m0;
        const int hi1 = h1 * 256 + m1;
        const int lo0 = l0 * 256 + o0;
        const int lo1 = l1 * 256 + o1;
        acc[n][0] = __fmaf_rn(static_cast<float>(hi0), scale_hi, acc[n][0]);
        acc[n][0] = __fmaf_rn(static_cast<float>(lo0), scale, acc[n][0]);
        acc[n][1] = __fmaf_rn(static_cast<float>(hi1), scale_hi, acc[n][1]);
        acc[n][1] = __fmaf_rn(static_cast<float>(lo1), scale, acc[n][1]);
      }
    }
    __syncthreads();
  }

  if (gid < grp) {
    const int h = kh * grp + gid;
    const int out_base = ((b * Hq + h) * nparts + part) * D;
    const int stat_base = ((b * Hq + h) * nparts + part) * 2;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      const int d0 = warp * 32 + n * 8 + lid * 2;
      partial_out[out_base + d0] = acc[n][0];
      partial_out[out_base + d0 + 1] = acc[n][1];
    }
    if (lid == 0) {
      partial_stat[stat_base] = s_m[gid];
      partial_stat[stat_base + 1] = s_l[gid];
    }
  }
}

// ============================================================================
// PTH 结构实验臂(2026-11-04):
//   4 compute warps + 4 producer warps, 2-stage 64-token tile, f16 PV。
//   目的:验证 PTH 的“producer/consumer + 大 tile + 1 CTA/SM”结构，
//   而不是只换 PV 数据类型。K/V 仍用 owl 的 int8 生产布局与 per-32-group
//   K scale / per-token V scale；QK 仍 u8s8，PV 走 cvt4+f16 m16n8k16。
// ============================================================================
namespace pthpc {
constexpr int TILE = 64;      // 2 pages
constexpr int TILE_PAGES = 2;

struct F16x2Pair { uint32_t lo; uint32_t hi; };

__device__ __forceinline__ uint32_t hsub2_bits(uint32_t a, uint32_t b) {
  const __half2 ha = *reinterpret_cast<const __half2*>(&a);
  const __half2 hb = *reinterpret_cast<const __half2*>(&b);
  const __half2 hc = __hsub2(ha, hb);
  return *reinterpret_cast<const uint32_t*>(&hc);
}

__device__ __forceinline__ F16x2Pair cvt4_s8_f16x2(uint32_t w) {
  constexpr uint32_t kMagicSignFlip = 0x80808080u;
  constexpr uint32_t kMagicExpBytes = 0x64646464u;
  constexpr uint32_t kMagicBiasHalf2 = 0x64806480u;
  constexpr uint32_t kSelBytes01 = 0x4140u;
  constexpr uint32_t kSelBytes23 = 0x4342u;
  const uint32_t u = w ^ kMagicSignFlip;
  return {hsub2_bits(__byte_perm(u, kMagicExpBytes, kSelBytes01), kMagicBiasHalf2),
          hsub2_bits(__byte_perm(u, kMagicExpBytes, kSelBytes23), kMagicBiasHalf2)};
}

__device__ __forceinline__ int swz_p64(int row, int col) {
  return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int v_swz64(int tok, int byte) {
  return (tok * tc::D + byte) ^ (2 * ((tok >> 1) & 3) * 16);
}
__device__ __forceinline__ int v_token_offset(int i) {
  return (i & 1) + 8 * (i >> 1);
}
__device__ __forceinline__ F16x2Pair v_pair_tokens(uint32_t a, uint32_t b) {
  return {__byte_perm(a, b, 0x5140), __byte_perm(a, b, 0x7362)};
}
__device__ __forceinline__ void v_b_fragments(const uint32_t (&w)[4],
                                              uint32_t (&b)[4][2]) {
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const F16x2Pair p = v_pair_tokens(w[2 * r], w[2 * r + 1]);
    const F16x2Pair d01 = cvt4_s8_f16x2(p.lo);
    const F16x2Pair d23 = cvt4_s8_f16x2(p.hi);
    b[0][r] = d01.lo;
    b[1][r] = d01.hi;
    b[2][r] = d23.lo;
    b[3][r] = d23.hi;
  }
}

constexpr int kBarFull0 = 2;
constexpr int kBarEmpty0 = 4;
constexpr int kBarThreads = 256;

__device__ __forceinline__ void named_sync(int id, int n) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void named_arrive(int id, int n) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

// One producer warp loads 16 tokens (half a page) of K/V + K/V scales.
__device__ __forceinline__ void load_tile16(
    const int8_t* __restrict__ k, const int8_t* __restrict__ v,
    const float* __restrict__ kscale, const float* __restrict__ vscale,
    const float* __restrict__ block_table, int bt_stride, int b,
    int kh, int hkv, int page0, int page1, int pages, int len,
    int pw, int lane,
    int8_t* __restrict__ dst_k, int8_t* __restrict__ dst_v,
    float* __restrict__ dst_ks, float* __restrict__ dst_vs) {
  const int key_base = pw * 16;
  const int page = (key_base < 32) ? page0 : page1;
  const int page_key = key_base & 31;
  const bool page_ok = page >= 0 && page < pages;
  const int phys = page_ok ? static_cast<int>(block_table[b * bt_stride + page]) : -1;

  if (page_ok) {
    for (int j = lane; j < 16 * (tc::D / 16); j += 32) {
      const int key = j / (tc::D / 16);
      const int off = (j % (tc::D / 16)) * 16;
      const int8_t* src =
          k + ((static_cast<long long>(phys) * tc::PAGE + page_key + key) *
                   hkv * tc::D) +
              static_cast<long long>(kh) * tc::D + off;
      owl_int8_attn::cp_async<16, owl_int8_attn::Cache::cg>(
          &dst_k[(key_base + key) * tc::D +
                 tc::swz(key_base + key, off / 2) * 2], src);
    }
    for (int j = lane; j < 16 * (tc::D / 16); j += 32) {
      const int key = j / (tc::D / 16);
      const int off = (j % (tc::D / 16)) * 16;
      const int8_t* src =
          v + ((static_cast<long long>(phys) * tc::PAGE + page_key + key) *
                   hkv * tc::D) +
              static_cast<long long>(kh) * tc::D + off;
      const int dk = key_base + key;
      owl_int8_attn::cp_async<16, owl_int8_attn::Cache::cg>(
          &dst_v[v_swz64(dk, off)], src);
    }
  }

  for (int j = lane; j < 16 * 8; j += 32) {
    const int key = j / 8;
    const int g = j % 8;
    const int kk = key_base + key;
    const int global = page * tc::PAGE + page_key + key;
    dst_ks[kk * 8 + g] =
        (page_ok && global < len)
            ? kscale[((static_cast<long long>(phys) * tc::PAGE + page_key + key) *
                          hkv + kh) * 8 + g]
            : 0.0f;
  }
  if (lane < 16) {
    const int kk = key_base + lane;
    const int global = page * tc::PAGE + page_key + lane;
    dst_vs[kk] =
        (page_ok && global < len)
            ? vscale[((static_cast<long long>(phys) * tc::PAGE + page_key + lane) *
                          hkv + kh)]
            : 1.0f;
  }
  owl_int8_attn::cp_commit();
}
}  // namespace pthpc

extern "C" __global__ void owl_int8_decode_pth_pc_split(
    const int8_t* __restrict__ q, const float* __restrict__ qscale,
    const int8_t* __restrict__ k, const float* __restrict__ kscale,
    const int8_t* __restrict__ v, const float* __restrict__ vscale,
    const float* __restrict__ block_table, const float* __restrict__ kv_lens,
    float* __restrict__ partial_out, float* __restrict__ partial_stat,
    __half* __restrict__ out_dummy,
    int B, int Hq, int Hkv, int page, int bt_stride, int nparts,
    const __half* __restrict__ dep_k, const __half* __restrict__ dep_v) {
  using namespace tc;
  using namespace owl_int8_attn;
  using namespace pthpc;
  (void)out_dummy; (void)dep_k; (void)dep_v;
  if (page != PAGE) return;

  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int Q_OFF = 0;
  constexpr int K_OFF = Q_OFF + 16 * D;
  constexpr int V_OFF = K_OFF + 2 * TILE * D;
  constexpr int PPITCH = 72;  // pth P'' pitch: 64 f16 + 8 f16 B padding
  constexpr int P_OFF = V_OFF + 2 * D * 32 * 2;
  constexpr int S_OFF = P_OFF + 16 * PPITCH * 2;
  constexpr int KS_OFF = S_OFF + 8 * TILE * 4;
  constexpr int VS_OFF = KS_OFF + 2 * TILE * 8 * 4;
  constexpr int QS_OFF = VS_OFF + 2 * TILE * 4;
  constexpr int A_OFF = QS_OFF + 8 * 8 * 4;
  constexpr int M_OFF = A_OFF + 8 * 4;
  constexpr int L_OFF = M_OFF + 8 * 4;
  constexpr int SMEM_BYTES = L_OFF + 8 * 4;
  static_assert(SMEM_BYTES == 78944, "pthpc shared layout");

  int8_t* s_q = reinterpret_cast<int8_t*>(smem + Q_OFF);
  int8_t (*s_k)[TILE * D] = reinterpret_cast<int8_t (*)[TILE * D]>(smem + K_OFF);
  int8_t (*s_v)[TILE * D] = reinterpret_cast<int8_t (*)[TILE * D]>(smem + V_OFF);
  __half* s_p = reinterpret_cast<__half*>(smem + P_OFF);
  float* s_score = reinterpret_cast<float*>(smem + S_OFF);
  float (*s_ks)[TILE * 8] = reinterpret_cast<float (*)[TILE * 8]>(smem + KS_OFF);
  float (*s_vs)[TILE] = reinterpret_cast<float (*)[TILE]>(smem + VS_OFF);
  float* s_qs = reinterpret_cast<float*>(smem + QS_OFF);
  float* s_alpha = reinterpret_cast<float*>(smem + A_OFF);
  float* s_m = reinterpret_cast<float*>(smem + M_OFF);
  float* s_l = reinterpret_cast<float*>(smem + L_OFF);

  const int kh = blockIdx.x;
  const int b = blockIdx.y;
  const int part = blockIdx.z;
  if (b >= B || kh >= Hkv || part >= nparts) return;
  const int grp = Hq / Hkv;
  if (grp <= 0 || grp > 6) return;

  const int len = static_cast<int>(kv_lens[b]);
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int pages = (len + page - 1) / page;
  const int tiles = (pages + 1) / 2;
  const int per = (tiles + nparts - 1) / nparts;
  const int t0 = part * per;
  const int t1 = min(tiles, t0 + per);

  if (t0 >= tiles) {
    for (int j = 0; j < grp; ++j) {
      const int h = kh * grp + j;
      const int out_base = ((b * Hq + h) * nparts + part) * D;
      const int stat_base = ((b * Hq + h) * nparts + part) * 2;
      for (int d = tid; d < D; d += blockDim.x) partial_out[out_base + d] = 0.0f;
      if (tid == 0) {
        partial_stat[stat_base] = -CUDART_INF_F;
        partial_stat[stat_base + 1] = 0.0f;
      }
    }
    return;
  }

  const int gid = lane >> 2;
  const int lid = lane & 3;
  const int a_mat = lane >> 3;
  const int a_rin = lane & 7;
  const int a_rowoff = a_rin + ((a_mat & 1) << 3);
  const int a_coloff = (a_mat >> 1) << 3;
  const int b_rin = lane & 7;
  const int b_koff = ((lane >> 3) & 1) << 3;

  if (warp < 4) {
    for (int i = tid; i < 16 * D; i += 128) s_q[i] = 0;
    for (int i = tid; i < 16 * PPITCH; i += 128) s_p[i] = __float2half_rn(0.0f);
    if (tid < 8) {
      s_alpha[tid] = 0.0f;
      s_m[tid] = -CUDART_INF_F;
      s_l[tid] = 0.0f;
    }
    for (int i = tid; i < 8 * 8; i += 128) s_qs[i] = 0.0f;
    asm volatile("bar.sync 1, 128;" ::: "memory");
    for (int i = tid; i < grp * (D / 16); i += 128) {
      const int j = i / (D / 16);
      const int off = (i % (D / 16)) * 16;
      const int h = kh * grp + j;
      const int8_t* src = q + static_cast<long long>(b * Hq + h) * D + off;
      *reinterpret_cast<int4*>(&s_q[j * D + swz(j, off / 2) * 2]) =
          *reinterpret_cast<const int4*>(src);
    }
    for (int i = tid; i < grp * 8; i += 128) {
      const int row = i / 8;
      const int g = i % 8;
      s_qs[i] = qscale[(b * Hq + kh * grp + row) * 8 + g];
    }
    if (tid < grp) {
      s_m[tid] = -CUDART_INF_F;
      s_l[tid] = 0.0f;
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
  } else {
    const int pw = warp - 4;
    for (int t = t0; t < t1; ++t) {
      const int st = (t - t0) & 1;
      if ((t - t0) >= 2) named_sync(kBarEmpty0 + st, kBarThreads);
      const int page0 = 2 * t;
      const int page1 = page0 + 1;
      load_tile16(k, v, kscale, vscale, block_table, bt_stride, b, kh, Hkv,
                  page0, page1, pages, len, pw, lane,
                  s_k[st], s_v[st], s_ks[st], s_vs[st]);
      cp_wait<0>();
      named_arrive(kBarFull0 + st, kBarThreads);
    }
    return;
  }

  float acc[4][4];
#pragma unroll
  for (int mt = 0; mt < 4; ++mt)
#pragma unroll
    for (int j = 0; j < 4; ++j) acc[mt][j] = 0.0f;

  for (int t = t0; t < t1; ++t) {
    const int st = (t - t0) & 1;
    const int tile_key0 = t * TILE;
    named_sync(kBarFull0 + st, kBarThreads);
    {
      // QK: 4 warps x 16 keys, 2 n-tiles each.
      const int nkey = warp * 16;
      float sc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) sc[nt][j] = 0.0f;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        unsigned af[4];
        ldmatrix_x4(af[0], af[1], af[2], af[3],
                    smem_addr(&s_q[a_rowoff * D +
                                  swz(a_rowoff, kk * 16 + a_coloff) * 2]));
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          unsigned bf[2];
          const int nk = nkey + nt * 8;
          ldmatrix_x2(bf[0], bf[1],
                      smem_addr(&s_k[st][(nk + b_rin) * D +
                                    swz(nk + b_rin, kk * 16 + b_koff) * 2]));
          int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
          mma_s8(c0, c1, c2, c3, af[0], af[1], af[2], af[3], bf[0], bf[1]);
          const int key0 = nk + lid * 2;
          const int key1 = key0 + 1;
          const float ks0 = s_ks[st][key0 * 8 + kk];
          const float ks1 = s_ks[st][key1 * 8 + kk];
          const float qg0 = s_qs[gid * 8 + kk] * kAttnScale;
          const float qg1 = (gid + 8 < grp) ? s_qs[(gid + 8) * 8 + kk] * kAttnScale : 0.0f;
          sc[nt][0] += static_cast<float>(c0) * (ks0 * qg0);
          sc[nt][1] += static_cast<float>(c1) * (ks1 * qg0);
          sc[nt][2] += static_cast<float>(c2) * (ks0 * qg1);
          sc[nt][3] += static_cast<float>(c3) * (ks1 * qg1);
        }
      }
      if (gid < grp) {
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
          const int key0 = nkey + nt * 8 + lid * 2;
          s_score[gid * TILE + key0] = sc[nt][0];
          s_score[gid * TILE + key0 + 1] = sc[nt][1];
        }
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");

      // softmax: warp w handles head w and (w<2) head w+4.
      for (int h = warp; h < grp; h += 4) {
        const int k0 = lane;
        const int k1 = lane + 32;
        const int g0 = tile_key0 + k0;
        const int g1 = tile_key0 + k1;
        const float s0 = (g0 < len) ? s_score[h * TILE + k0] : -CUDART_INF_F;
        const float s1 = (g1 < len) ? s_score[h * TILE + k1] : -CUDART_INF_F;
        float bm = fmaxf(s0, s1);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, off));
        const float old_m = s_m[h];
        __syncwarp();
        const float nm = fmaxf(old_m, bm);
        const float alpha = (old_m == -CUDART_INF_F) ? 0.0f : expf(old_m - nm);
        const float p0 = (s0 == -CUDART_INF_F) ? 0.0f : expf(s0 - nm);
        const float p1 = (s1 == -CUDART_INF_F) ? 0.0f : expf(s1 - nm);
        s_p[h * PPITCH + k0] = __float2half_rn(p0 * s_vs[st][k0]);
        s_p[h * PPITCH + k1] = __float2half_rn(p1 * s_vs[st][k1]);
        float sum = p0 + p1;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == 0) {
          s_alpha[h] = alpha;
          s_m[h] = nm;
          s_l[h] = s_l[h] * alpha + sum;
        }
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");

      // PV (pth swap-AB): O^T[dim][row] = V^T[dim][token] * P[token][row].
      // 4 warps x 64 dims, 4 m-tiles; 64 tokens = 4 k-steps of 16.
      unsigned bp[4][2];
      const int l8 = lane & 7;
      const int mi = lane >> 3;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        ldmatrix_x2(bp[j][0], bp[j][1],
                    smem_addr(&s_p[l8 * PPITCH + j * 16 +
                                  ((mi & 1) << 3)]));
      }
      const float a0 = (2 * (lane & 3) < grp) ? s_alpha[2 * (lane & 3)] : 0.0f;
      const float a1 = (2 * (lane & 3) + 1 < grp) ? s_alpha[2 * (lane & 3) + 1] : 0.0f;
      if (2 * (lane & 3) < grp) {
        acc[0][0] *= a0; acc[1][0] *= a0; acc[2][0] *= a0; acc[3][0] *= a0;
        acc[0][2] *= a0; acc[1][2] *= a0; acc[2][2] *= a0; acc[3][2] *= a0;
      }
      if (2 * (lane & 3) + 1 < grp) {
        acc[0][1] *= a1; acc[1][1] *= a1; acc[2][1] *= a1; acc[3][1] *= a1;
        acc[0][3] *= a1; acc[1][3] *= a1; acc[2][3] *= a1; acc[3][3] *= a1;
      }
      const int g = lane >> 2;
      const int q = lane & 3;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
#pragma unroll
        for (int a = 0; a < 2; ++a) {
          uint32_t w[4];
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            const int tok = 16 * j + 2 * q + v_token_offset(i);
            const int dim = 32 * (2 * warp + a) + 4 * g;
            w[i] = *reinterpret_cast<const uint32_t*>(
                s_v[st] + v_swz64(tok, dim));
          }
          uint32_t b[4][2];
          v_b_fragments(w, b);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t am[4] = {b[2 * h][0], b[2 * h + 1][0],
                                    b[2 * h][1], b[2 * h + 1][1]};
            mma_f16(acc[2 * a + h][0], acc[2 * a + h][1],
                    acc[2 * a + h][2], acc[2 * a + h][3],
                    am[0], am[1], am[2], am[3], bp[j][0], bp[j][1]);
          }
        }
      }
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    if (t + 2 < t1) named_arrive(kBarEmpty0 + st, kBarThreads);
  }

  if (warp < 4) {

    const int g = lane >> 2;
    const int q = lane & 3;
#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int mt = 2 * a + h;
        const int d0 = warp * 64 + a * 32 + 4 * g + 2 * h;
        const int d1 = d0 + 1;
        const int row0 = 2 * q;
        const int row1 = row0 + 1;
        if (row0 < grp) {
          const int hh = kh * grp + row0;
          const int base = ((b * Hq + hh) * nparts + part) * D;
          partial_out[base + d0] = acc[mt][0];
          partial_out[base + d1] = acc[mt][2];
        }
        if (row1 < grp) {
          const int hh = kh * grp + row1;
          const int base = ((b * Hq + hh) * nparts + part) * D;
          partial_out[base + d0] = acc[mt][1];
          partial_out[base + d1] = acc[mt][3];
        }
      }
    }
    if (lane == 0) {
      for (int r = 0; r < grp; ++r) {
        const int h = kh * grp + r;
        const int stat_base = ((b * Hq + h) * nparts + part) * 2;
        partial_stat[stat_base] = s_m[r];
        partial_stat[stat_base + 1] = s_l[r];
      }
    }
  }
}
