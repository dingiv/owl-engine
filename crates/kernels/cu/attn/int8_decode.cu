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
#include "int8_attn_helpers.cuh"

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
    const uint16_t* __restrict__ vp, const float* __restrict__ vscale,
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
  __shared__ __align__(16) uint16_t s_p[16 * PP];
  __shared__ float s_score[8 * PAGE];
  __shared__ uint8_t s_p8[8 * PAGE];
  __shared__ float s_ps[8], s_alpha[8], s_m[8], s_l[8];
  __shared__ float s_qs[8], s_ks[PAGE];

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
  for (int i = tid; i < 16 * PP; i += blockDim.x) s_p[i] = 0;
  for (int i = tid; i < grp * (D / 16); i += blockDim.x) {
    const int j = i / (D / 16);
    const int off = (i % (D / 16)) * 16;
    const int h = kh * grp + j;
    const int8_t* src = q + static_cast<long long>(b * Hq + h) * D + off;
    *reinterpret_cast<int4*>(&s_q[j * D + swz(j, off / 2) * 2]) =
        *reinterpret_cast<const int4*>(src);
  }
  if (tid < grp) {
    s_qs[tid] = qscale[b * Hq + kh * grp + tid];
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
    for (int i = tid; i < D * (PP / 8); i += blockDim.x) {
      const int d = i / (PP / 8);
      const int pair0 = (i % (PP / 8)) * 8;
      const uint16_t* src = &vp[((static_cast<long long>(phys) * Hkv + kh) * D) * PP +
                                static_cast<long long>(d) * PP + pair0];
      cp_async<16, Cache::cg>(&s_v[d * PP + swz_v(d, pair0)], src);
    }
    if (tid < PAGE) {
      const int key = base + tid;
      s_ks[tid] = (key < len) ? kscale[static_cast<long long>(key) * Hkv + kh] : 0.0f;
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    // ---- QK: 4 warps, each 8 keys; 8 k-steps of 32 dims ----
    if (warp < 4) {
      int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
      const int nkey = warp * 8;
#pragma unroll
      for (int kk = 0; kk < 8; ++kk) {
        unsigned af[4], bf[2];
        ldmatrix_x4(af[0], af[1], af[2], af[3],
                    smem_addr(&s_q[a_rowoff * D +
                                  swz(a_rowoff, kk * 16 + a_coloff) * 2]));
        ldmatrix_x2(bf[0], bf[1],
                    smem_addr(&s_k[(nkey + b_rin) * D +
                                  swz(nkey + b_rin, kk * 16 + b_koff) * 2]));
        mma_s8(c0, c1, c2, c3, af[0], af[1], af[2], af[3], bf[0], bf[1]);
      }
      const int row0 = gid;
      const int row1 = gid + 8;
      const int col0 = lid * 2;
      const int key0 = nkey + col0;
      const int key1 = key0 + 1;
      if (row0 < grp) {
        const float qs = s_qs[row0];
        s_score[row0 * PAGE + key0] = static_cast<float>(c0) * qs * s_ks[key0];
        s_score[row0 * PAGE + key1] = static_cast<float>(c1) * qs * s_ks[key1];
      }
      if (row1 < grp) {
        const float qs = s_qs[row1];
        s_score[row1 * PAGE + key0] = static_cast<float>(c2) * qs * s_ks[key0];
        s_score[row1 * PAGE + key1] = static_cast<float>(c3) * qs * s_ks[key1];
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
      const float nm = fmaxf(old_m, bm);
      const float alpha = (old_m == -CUDART_INF_F) ? 0.0f : expf(old_m - nm);
      const float p = (sc == -CUDART_INF_F) ? 0.0f : expf(sc - nm);
      const float pmax = (bm == -CUDART_INF_F) ? 0.0f : expf(bm - nm);
      const float ps = (pmax > 0.0f) ? pmax * (1.0f / 255.0f) : (1.0f / 255.0f);
      const float inv = (pmax > 0.0f) ? 255.0f / pmax : 0.0f;
      float sum = p;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
      s_p8[h * PAGE + lane] =
          static_cast<uint8_t>(__float2uint_rn(p * inv));
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
      const uint8_t p0 = s_p8[row * PAGE + 2 * pair];
      const uint8_t p1 = s_p8[row * PAGE + 2 * pair + 1];
      s_p[row * PP + swz_p(row, pair)] =
          static_cast<uint16_t>(p0) | (static_cast<uint16_t>(p1) << 8);
    }
    __syncthreads();

    // ---- PV: 8 warps x 64 dims; A=P(16x32), B=V(32x8) ----
    unsigned a[4];
    ldmatrix_x4(a[0], a[1], a[2], a[3],
                smem_addr(&s_p[a_rowoff * PP + swz_p(a_rowoff, a_coloff)]));
    const float vs = vscale[(static_cast<long long>(phys) * Hkv + kh) * 4 + (warp >> 1)];
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
      int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
      mma_u8s8(c0, c1, c2, c3, a[0], a[1], a[2], a[3], bf[0], bf[1]);
      if (gid < grp) {
        const float scale = s_ps[gid] * vs;
        acc[n][0] += static_cast<float>(c0) * scale;
        acc[n][1] += static_cast<float>(c1) * scale;
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
