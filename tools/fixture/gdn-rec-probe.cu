// ============================================================================
// owl P1.4 —— GDN 递推核(Prefill)延迟归因 + 变体台
//
// ## 为什么
//
// P0.1 逐核账(W4A8/TP2/8k):`owl_gdn_recurrence_varlen_gqa_f16` = 664 ms/卡(19%),
// 生产 grid `(16,24,1)`(TP 每 rank nv=24/nk=8)· **1.73 µs/token**。
// 而在 1950 MHz 下 1.73 µs/token = **~3400 cycle/token** —— 一个字面只有
// `16 FMA + 2 次 warp 归约 + 9 个标量装载` 的循环不该这么贵 ⇒ **延迟暴露**。
//
// 现役核的三处结构性阻碍(读 `gdn_recurrence.cu`):
//   ① `__syncthreads()` 在 t 循环末尾 —— q/k 预取写 shared 后必须等到全部落地;
//   ② `v_base[t]` 标量装载在归约之后才发 —— 完整 DRAM 往返压在关键路径;
//   ③ 8 个 warp 共享同一份 q/k(128 halfs),靠 block 内 barrier 交换。
// 但**每个 warp 只碰自己的 v 列**(state 行分片、v/g/beta 逐列独立)⇒ **warp 之间
// 零数据依赖** —— barrier 是纯粹的同步税,可以用"每 lane 自取 4 个 k 索引"
// 直接消灭(warp 内 q/k 冗余读 8×,换来的是 zero-barrier + register 预取)。
//
// ## 臂
//
//   V0  现役核**逐字**复制(基线;必须复现生产读数)
//   V1  warp 自治:**无 shared / 无 barrier**,每 lane 自取 4 个 q/k(内联装载)
//   V2  V1 + 寄存器预取深度 4(q/k/v/g/beta 全部提前 4 token 发)
//   V3  同 V2,深度 8
//   V4  V1 + 深度 4,但 k/q 用 `__ldg`(只读缓存路径)
//
// 判据:① 各臂 `out`/`state` 与 V0 **逐字节相同**(算术序不变,只挪装载);
//       ② µs/token(T 扫 128/1024)。
//
// ## ★ 归因读数(2026-10-19;3080 ord2;nv=24/nk=8;T=1024;全档 8k 预热后 1.9 GHz)
//
// | 臂 | µs/token | Δ | 说明 |
// |---|---:|---:|---|
// | **V0 现役** | **1.74** | — | 生产同形(与引擎 nsys 的 1730µs/发 吻合) |
// | V1 warp 自治(去 barrier) | 1.50 | −0.24 | `__syncthreads()` 的税 |
// | V3 = V1 + 寄存器预取 4 | 1.40 | −0.10 | 装载延迟(只剩 6%) |
// | D0 = V1 去两次 warp 归约 | 0.98 | **−0.42** | **归约 = 24%** |
// | B1 = D0 再删 q/k 装载 | 0.53 | **−0.45** | **q/k 装载 = 26%**(块内 8× 冗余) |
// | B3 = B1 再删 out 写 | 0.34 | **−0.18** | **单 lane 2B 写 = 10%**(32B sector 摊薄) |
// | B4 = 再删 state 口 | 0.33 | −0.01 | 纯算术骨架(19%) |
//
// **⇒ 结论:现有实现的 81% 是"结构性开销",不是算术**:两次 32 lane 归约
// (10 次 shfl/token)、块内 8× 冗余的 q/k 标量装载、每 token 一条单 lane 2B 写。
// 三者都是**布局选择**造成的,可以一起消灭:
//   ① 一个 warp 管 **4 个 v 列**(8 lane 一列)⇒ 归约 3 步、每列 1.5 次 shfl;
//   ② **状态行归属改成"每 lane 连续 4 行"** ⇒ 与 16B 向量装载对齐(2 token/发);
//   ③ out 写聚成 **8B/16B**(4–8 列一并),而不是 2B 单发。
// 天花板 ≈ 0.5 µs/token(**~3.4×**),对照:FLA chunked(matmul 版)是另一条路,
// 但其 AOT cubin 按 `NV=48` 烘死(`triton/gdn_chunked_aot.json`),TP2 的 (8,24)
// 需重编 + 重走金标 ⇒ 本轮先走"改布局"这条(不动协议、不换数值域)。
//
// ## ★ 结果(2026-10-19 同日落生产,实录 §68)
//
// V6 = **C=2 列/warp + D=2 预取 + 无 barrier + 4B 合写**:1.087 µs/token(**1.61×**),
// `out`/`state` 与 V0 **逐字节相同**(算术序未动)⇒ 已落进
// `crates/kernels/cu/gdn/gdn_recurrence.cu` 的 `gdn_rec_cols2` 快路(kd≠128 回落原形),
// 端到端 TP2 std prefill **1486 → 1570(+5.7%)**、decode 不变、六门全绿。
// C=4 反而更慢(并行 warp 数掉到 1/4,延迟藏不住)⇒ 变量不是"每列指令数"一个。
//
// ## 编译 / 运行
//
//   nvcc -O3 -arch=sm_86 -o /tmp/gdn-rec-probe tools/gdn-rec-probe.cu
//   /tmp/gdn-rec-probe <ord> [nv] [nk] [T...]
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_fp16.h>

#define CK(x)                                                                                 \
  do {                                                                                        \
    cudaError_t e_ = (x);                                                                     \
    if (e_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));         \
      exit(1);                                                                                \
    }                                                                                         \
  } while (0)

// ── 公共小件 ────────────────────────────────────────────────────────────────
__device__ __forceinline__ float gdn_to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ __half gdn_from_f(float x) { return __float2half(x); }
__device__ __forceinline__ float gdn_warp_sum(float val) {
  for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
  return __shfl_sync(0xffffffff, val, 0);
}

// ============================================================================
// V0 —— 现役核逐字(基线)
// ============================================================================
#define WARPS 8
extern "C" __global__ void k_v0(const __half *__restrict__ q, const __half *__restrict__ k,
                                const __half *__restrict__ v, const __half *__restrict__ g,
                                const __half *__restrict__ beta, float *__restrict__ state,
                                const float *__restrict__ slots,
                                const float *__restrict__ cu_seqlens, size_t batch, size_t nv,
                                size_t nk, size_t kd, size_t vd, float q_scale,
                                __half *__restrict__ out) {
  constexpr int BK = 128;
  constexpr int ROWS_PER_LANE = (BK + 31) / 32;
  const int seq_head = blockIdx.y;
  const int lane = threadIdx.x;
  const int warp_id = threadIdx.y;
  const int v_idx = blockIdx.x * WARPS + warp_id;
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
  const int ts_qk = nk * kd, ts_v = nv * vd, ts_g = nv;
  const __half *q_base = q + (size_t)start * ts_qk + (size_t)k_head_idx * kd;
  const __half *k_base = k + (size_t)start * ts_qk + (size_t)k_head_idx * kd;
  const __half *v_base = v + (size_t)start * ts_v + (size_t)v_head_idx * vd;
  const __half *g_base = g + (size_t)start * ts_g + v_head_idx;
  const __half *beta_base = beta + (size_t)start * ts_g + v_head_idx;
  __half *out_base = out + (size_t)start * ts_v + (size_t)v_head_idx * vd;

  __shared__ float q_buf[2][BK];
  __shared__ float k_buf[2][BK];
  __shared__ float scalars[2][2];

  const bool v_valid = (v_idx < vd);
  float *state_head = v_valid ? state + ((size_t)slot * nv + v_head_idx) * kd * vd : nullptr;
  float s_shard[ROWS_PER_LANE];
  if (v_valid)
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      const int k_idx = r * 32 + lane;
      s_shard[r] = ((size_t)k_idx < kd) ? state_head[(size_t)k_idx * vd + v_idx] : 0.0f;
    }
  const int total_threads = WARPS * 32;
  const int tid = warp_id * 32 + lane;
  if (seq_len > 0) {
    for (int j = tid; j < BK; j += total_threads)
      if (j < kd) { q_buf[0][j] = gdn_to_f(q_base[j]) * q_scale; k_buf[0][j] = gdn_to_f(k_base[j]); }
      else { q_buf[0][j] = 0.f; k_buf[0][j] = 0.f; }
    if (tid == 0) {
      scalars[0][0] = expf(gdn_to_f(g_base[0]));
      scalars[0][1] = gdn_to_f(beta_base[0]);
    }
    __syncthreads();
  }
  for (int t = 0; t < seq_len; ++t) {
    const int cur = t & 1, nxt = 1 - cur;
    if (t + 1 < seq_len) {
      const __half *q_next = q_base + (size_t)(t + 1) * ts_qk;
      const __half *k_next = k_base + (size_t)(t + 1) * ts_qk;
      for (int j = tid; j < BK; j += total_threads)
        if ((size_t)j < kd) { q_buf[nxt][j] = gdn_to_f(q_next[j]) * q_scale; k_buf[nxt][j] = gdn_to_f(k_next[j]); }
        else { q_buf[nxt][j] = 0.f; k_buf[nxt][j] = 0.f; }
      if (tid == 0) {
        scalars[nxt][0] = expf(gdn_to_f(g_base[(size_t)(t + 1) * ts_g]));
        scalars[nxt][1] = gdn_to_f(beta_base[(size_t)(t + 1) * ts_g]);
      }
    }
    if (v_valid) {
      const float decay = scalars[cur][0], beta_t = scalars[cur][1];
      float kv = 0.f;
      for (int r = 0; r < ROWS_PER_LANE; ++r) {
        const int k_idx = r * 32 + lane;
        s_shard[r] *= decay;
        kv = __fmaf_rn(s_shard[r], k_buf[cur][(size_t)k_idx], kv);
      }
      const float kv_mem = gdn_warp_sum(kv);
      const float delta = (gdn_to_f(v_base[(size_t)t * ts_v + v_idx]) - kv_mem) * beta_t;
      float y = 0.f;
      for (int r = 0; r < ROWS_PER_LANE; ++r) {
        const int k_idx = r * 32 + lane;
        s_shard[r] = __fmaf_rn(k_buf[cur][(size_t)k_idx], delta, s_shard[r]);
        y = __fmaf_rn(s_shard[r], q_buf[cur][(size_t)k_idx], y);
      }
      const float y_t = gdn_warp_sum(y);
      if (lane == 0) out_base[(size_t)t * ts_v + (size_t)v_idx] = gdn_from_f(y_t);
    }
    __syncthreads();
  }
  if (v_valid)
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      const int k_idx = r * 32 + lane;
      if ((size_t)k_idx < kd) state_head[(size_t)k_idx * vd + (size_t)v_idx] = s_shard[r];
    }
}

// ============================================================================
// V1 —— warp 自治:无 shared / 无 barrier(每 lane 自取 4 个 q/k 索引)
//
// ⚠️ 算术序**逐式同 V0**:decay → kv 归约 → delta → 更新+y 归约 → 写 out。
//    变化只有"数据从哪来":shared+barrier → 本 lane 寄存器。
// ============================================================================
extern "C" __global__ void k_v1(const __half *__restrict__ q, const __half *__restrict__ k,
                                const __half *__restrict__ v, const __half *__restrict__ g,
                                const __half *__restrict__ beta, float *__restrict__ state,
                                const float *__restrict__ slots,
                                const float *__restrict__ cu_seqlens, size_t batch, size_t nv,
                                size_t nk, size_t kd, size_t vd, float q_scale,
                                __half *__restrict__ out) {
  constexpr int ROWS_PER_LANE = 4;
  const int lane = threadIdx.x, warp_id = threadIdx.y;
  const int v_idx = blockIdx.x * WARPS + warp_id;
  const int v_head_idx = blockIdx.y % nv;
  if (v_idx >= (int)vd) return;
  const int k_head_idx = v_head_idx / (int)(nv / nk);
  if ((int)slots[0] < 0) return;
  const int start = (int)cu_seqlens[0], end = (int)cu_seqlens[1];
  const int seq_len = end - start;
  if (seq_len <= 0) return;
  const int ts_qk = (int)(nk * kd), ts_v = (int)(nv * vd);
  const __half *q_base = q + (size_t)start * ts_qk + (size_t)k_head_idx * kd;
  const __half *k_base = k + (size_t)start * ts_qk + (size_t)k_head_idx * kd;
  const __half *v_base = v + (size_t)start * ts_v + (size_t)v_head_idx * vd + v_idx;
  const __half *g_base = g + (size_t)start * nv + v_head_idx;
  const __half *beta_base = beta + (size_t)start * nv + v_head_idx;
  __half *out_base = out + (size_t)start * ts_v + (size_t)v_head_idx * vd + v_idx;
  float *state_head = state + ((size_t)(int)slots[0] * nv + v_head_idx) * kd * vd;

  float s[ROWS_PER_LANE];
#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; ++r) s[r] = state_head[(size_t)(r * 32 + lane) * vd + v_idx];

  for (int t = 0; t < seq_len; ++t) {
    const float decay = expf(gdn_to_f(g_base[(size_t)t * nv]));
    const float beta_t = gdn_to_f(beta_base[(size_t)t * nv]);
    float kv = 0.f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      const float kv_v = gdn_to_f(k_base[(size_t)t * ts_qk + r * 32 + lane]);
      s[r] *= decay;
      kv = __fmaf_rn(s[r], kv_v, kv);
    }
    const float kv_mem = gdn_warp_sum(kv);
    const float delta = (gdn_to_f(v_base[(size_t)t * ts_v]) - kv_mem) * beta_t;
    float y = 0.f;
#pragma unroll
    for (int r = 0; r < ROWS_PER_LANE; ++r) {
      const float q_v = gdn_to_f(q_base[(size_t)t * ts_qk + r * 32 + lane]) * q_scale;
      const float k_v = gdn_to_f(k_base[(size_t)t * ts_qk + r * 32 + lane]);
      s[r] = __fmaf_rn(k_v, delta, s[r]);
      y = __fmaf_rn(s[r], q_v, y);
    }
    const float y_t = gdn_warp_sum(y);
    if (lane == 0) out_base[(size_t)t * ts_v] = gdn_from_f(y_t);
  }
#pragma unroll
  for (int r = 0; r < ROWS_PER_LANE; ++r) state_head[(size_t)(r * 32 + lane) * vd + v_idx] = s[r];
}

// ============================================================================
// V4 —— 寄存器预取(**展开 D 令牌,stage 索引恒为编译期常量**)
//
// ⚠️ 坑(2026-10-19 首轮自踩):环形槽用 `buf[t % D]` 的**运行期索引**会退化成
//    local memory(实测 4.8 µs/token,比不预取还慢 3×)。必须把 t 循环按 D 展开,
//    让 `s` 在 `#pragma unroll` 内成为常量 ⇒ 槽全在寄存器。
// ============================================================================
template <int D>
__global__ void k_v4(const __half *__restrict__ q, const __half *__restrict__ k,
                     const __half *__restrict__ v, const __half *__restrict__ g,
                     const __half *__restrict__ beta, float *__restrict__ state,
                     const float *__restrict__ slots, const float *__restrict__ cu_seqlens,
                     size_t batch, size_t nv, size_t nk, size_t kd, size_t vd, float q_scale,
                     __half *__restrict__ out) {
  constexpr int R = 4;
  const int lane = threadIdx.x, warp_id = threadIdx.y;
  const int v_idx = blockIdx.x * WARPS + warp_id;
  const int v_head_idx = blockIdx.y % nv;
  if (v_idx >= (int)vd) return;
  const int k_head_idx = v_head_idx / (int)(nv / nk);
  if ((int)slots[0] < 0) return;
  const int start = (int)cu_seqlens[0], end = (int)cu_seqlens[1];
  const int seq_len = end - start;
  if (seq_len <= 0) return;
  const int ts_qk = (int)(nk * kd), ts_v = (int)(nv * vd);
  const __half *q_base = q + (size_t)start * ts_qk + (size_t)k_head_idx * kd;
  const __half *k_base = k + (size_t)start * ts_qk + (size_t)k_head_idx * kd;
  const __half *v_base = v + (size_t)start * ts_v + (size_t)v_head_idx * vd + v_idx;
  const __half *g_base = g + (size_t)start * nv + v_head_idx;
  const __half *beta_base = beta + (size_t)start * nv + v_head_idx;
  __half *out_base = out + (size_t)start * ts_v + (size_t)v_head_idx * vd + v_idx;
  float *state_head = state + ((size_t)(int)slots[0] * nv + v_head_idx) * kd * vd;

  float s[R];
#pragma unroll
  for (int r = 0; r < R; ++r) s[r] = state_head[(size_t)(r * 32 + lane) * vd + v_idx];

  __half qb[D][R], kb[D][R], vb[D], gb[D], bb[D];

#define OWL_LOAD(stage, tok)                                                                    \
  do {                                                                                          \
    if ((tok) < seq_len) {                                                                      \
      const size_t oq = (size_t)(tok) * ts_qk + (size_t)lane;                                    \
      _Pragma("unroll") for (int r = 0; r < R; ++r) {                                           \
        qb[stage][r] = q_base[oq + (size_t)r * 32];                                             \
        kb[stage][r] = k_base[oq + (size_t)r * 32];                                             \
      }                                                                                         \
      vb[stage] = v_base[(size_t)(tok) * ts_v];                                                 \
      gb[stage] = g_base[(size_t)(tok) * nv];                                                   \
      bb[stage] = beta_base[(size_t)(tok) * nv];                                                \
    }                                                                                           \
  } while (0)

#define OWL_COMPUTE(stage, tok)                                                                 \
  do {                                                                                          \
    const float decay = expf(gdn_to_f(gb[stage]));                                               \
    const float bt = gdn_to_f(bb[stage]);                                                       \
    float kv = 0.f;                                                                             \
    _Pragma("unroll") for (int r = 0; r < R; ++r) {                                             \
      s[r] *= decay;                                                                            \
      kv = __fmaf_rn(s[r], gdn_to_f(kb[stage][r]), kv);                                         \
    }                                                                                           \
    const float kv_mem = gdn_warp_sum(kv);                                                       \
    const float delta = (gdn_to_f(vb[stage]) - kv_mem) * bt;                                     \
    float y = 0.f;                                                                              \
    _Pragma("unroll") for (int r = 0; r < R; ++r) {                                             \
      s[r] = __fmaf_rn(gdn_to_f(kb[stage][r]), delta, s[r]);                                    \
      y = __fmaf_rn(s[r], gdn_to_f(qb[stage][r]) * q_scale, y);                                  \
    }                                                                                           \
    const float y_t = gdn_warp_sum(y);                                                           \
    if (lane == 0) out_base[(size_t)(tok) * ts_v] = gdn_from_f(y_t);                             \
  } while (0)

#pragma unroll
  for (int d = 0; d < D; ++d) OWL_LOAD(d, d);

  int t0 = 0;
  for (; t0 + D <= seq_len; t0 += D) {
#pragma unroll
    for (int st = 0; st < D; ++st) {
      OWL_COMPUTE(st, t0 + st);
      OWL_LOAD(st, t0 + st + D);
    }
  }
  for (int t = t0; t < seq_len; ++t) {  // 尾巴 ≤ D-1 个令牌:内联装载(慢路,可忽略)
    OWL_LOAD(0, t);
    OWL_COMPUTE(0, t);
  }
#undef OWL_LOAD
#undef OWL_COMPUTE

#pragma unroll
  for (int r = 0; r < R; ++r) state_head[(size_t)(r * 32 + lane) * vd + v_idx] = s[r];
}


// ============================================================================
// 诊断臂(**数值故意错**,只用于归因):V6 去归约 / V7 去 expf
// ============================================================================
__device__ __forceinline__ float gdn_noop_sum(float v) { return v; }

template <int D, int MODE>  // MODE 0=去归约 1=去expf
__global__ void k_diag(const __half *__restrict__ q, const __half *__restrict__ k,
                       const __half *__restrict__ v, const __half *__restrict__ g,
                       const __half *__restrict__ beta, float *__restrict__ state,
                       const float *__restrict__ slots, const float *__restrict__ cu_seqlens,
                       size_t batch, size_t nv, size_t nk, size_t kd, size_t vd, float q_scale,
                       __half *__restrict__ out) {
  constexpr int R = 4;
  const int lane = threadIdx.x, warp_id = threadIdx.y;
  const int v_idx = blockIdx.x * WARPS + warp_id;
  const int v_head_idx = blockIdx.y % nv;
  if (v_idx >= (int)vd) return;
  const int k_head_idx = v_head_idx / (int)(nv / nk);
  if ((int)slots[0] < 0) return;
  const int start = (int)cu_seqlens[0], end = (int)cu_seqlens[1];
  const int seq_len = end - start;
  if (seq_len <= 0) return;
  const int ts_qk = (int)(nk * kd), ts_v = (int)(nv * vd);
  const __half *q_base = q + (size_t)start * ts_qk + (size_t)k_head_idx * kd;
  const __half *k_base = k + (size_t)start * ts_qk + (size_t)k_head_idx * kd;
  const __half *v_base = v + (size_t)start * ts_v + (size_t)v_head_idx * vd + v_idx;
  const __half *g_base = g + (size_t)start * nv + v_head_idx;
  const __half *beta_base = beta + (size_t)start * nv + v_head_idx;
  __half *out_base = out + (size_t)start * ts_v + (size_t)v_head_idx * vd + v_idx;
  float *state_head = state + ((size_t)(int)slots[0] * nv + v_head_idx) * kd * vd;
  float s[R];
#pragma unroll
  for (int r = 0; r < R; ++r) s[r] = state_head[(size_t)(r * 32 + lane) * vd + v_idx];
  for (int t = 0; t < seq_len; ++t) {
    const float gv = gdn_to_f(g_base[(size_t)t * nv]);
    const float decay = (MODE == 1) ? 0.99f : expf(gv);
    const float bt = gdn_to_f(beta_base[(size_t)t * nv]);
    const size_t oq = (size_t)t * ts_qk + (size_t)lane;
    float kv = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      s[r] *= decay;
      kv = __fmaf_rn(s[r], gdn_to_f(k_base[oq + (size_t)r * 32]), kv);
    }
    const float kv_mem = (MODE == 0) ? gdn_noop_sum(kv) : gdn_warp_sum(kv);
    const float delta = (gdn_to_f(v_base[(size_t)t * ts_v]) - kv_mem) * bt;
    float y = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      s[r] = __fmaf_rn(gdn_to_f(k_base[oq + (size_t)r * 32]), delta, s[r]);
      y = __fmaf_rn(s[r], gdn_to_f(q_base[oq + (size_t)r * 32]) * q_scale, y);
    }
    const float y_t = (MODE == 0) ? gdn_noop_sum(y) : gdn_warp_sum(y);
    if (lane == 0) out_base[(size_t)t * ts_v] = gdn_from_f(y_t);
  }
#pragma unroll
  for (int r = 0; r < R; ++r) state_head[(size_t)(r * 32 + lane) * vd + v_idx] = s[r];
}


// 归因阶梯(累积削项;全部数值故意错):B0 无归约 → B1 再去 q/k 装载 → B2 再去 v 装载
// → B3 再去 out 写 → B4 再去 state 读写(只剩循环骨架)
template <int CUT>  // 0: 无归约  1: +无q/k  2: +无v  3: +无out  4: 全空
__global__ void k_ladder(const __half *__restrict__ q, const __half *__restrict__ k,
                         const __half *__restrict__ v, const __half *__restrict__ g,
                         const __half *__restrict__ beta, float *__restrict__ state,
                         const float *__restrict__ slots, const float *__restrict__ cu_seqlens,
                         size_t batch, size_t nv, size_t nk, size_t kd, size_t vd, float q_scale,
                         __half *__restrict__ out) {
  constexpr int R = 4;
  const int lane = threadIdx.x, warp_id = threadIdx.y;
  const int v_idx = blockIdx.x * WARPS + warp_id;
  const int v_head_idx = blockIdx.y % nv;
  if (v_idx >= (int)vd) return;
  const int k_head_idx = v_head_idx / (int)(nv / nk);
  if ((int)slots[0] < 0) return;
  const int start = (int)cu_seqlens[0], end = (int)cu_seqlens[1];
  const int seq_len = end - start;
  if (seq_len <= 0) return;
  const int ts_qk = (int)(nk * kd), ts_v = (int)(nv * vd);
  const __half *q_base = q + (size_t)start * ts_qk + (size_t)k_head_idx * kd;
  const __half *k_base = k + (size_t)start * ts_qk + (size_t)k_head_idx * kd;
  const __half *v_base = v + (size_t)start * ts_v + (size_t)v_head_idx * vd + v_idx;
  const __half *g_base = g + (size_t)start * nv + v_head_idx;
  const __half *beta_base = beta + (size_t)start * nv + v_head_idx;
  __half *out_base = out + (size_t)start * ts_v + (size_t)v_head_idx * vd + v_idx;
  float *state_head = state + ((size_t)(int)slots[0] * nv + v_head_idx) * kd * vd;
  float s[R];
#pragma unroll
  for (int r = 0; r < R; ++r) s[r] = state_head[(size_t)(r * 32 + lane) * vd + v_idx];
  for (int t = 0; t < seq_len; ++t) {
    const float decay = expf(gdn_to_f(g_base[(size_t)t * nv]));
    const float bt = gdn_to_f(beta_base[(size_t)t * nv]);
    const size_t oq = (size_t)t * ts_qk + (size_t)lane;
    float kv = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float kk = (CUT >= 1) ? 0.5f : gdn_to_f(k_base[oq + (size_t)r * 32]);
      s[r] *= decay;
      kv = __fmaf_rn(s[r], kk, kv);
    }
    const float vv = (CUT >= 2) ? 0.25f : gdn_to_f(v_base[(size_t)t * ts_v]);
    const float delta = (vv - kv) * bt;
    float y = 0.f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const float kk = (CUT >= 1) ? 0.5f : gdn_to_f(k_base[oq + (size_t)r * 32]);
      const float qq = (CUT >= 1) ? 0.5f : gdn_to_f(q_base[oq + (size_t)r * 32]) * q_scale;
      s[r] = __fmaf_rn(kk, delta, s[r]);
      y = __fmaf_rn(s[r], qq, y);
    }
    if (CUT < 3) { if (lane == 0) out_base[(size_t)t * ts_v] = gdn_from_f(y); }
    else { if (y == -1234.f) out_base[(size_t)t * ts_v] = gdn_from_f(y); }
  }
  if (CUT < 4)
#pragma unroll
    for (int r = 0; r < R; ++r) state_head[(size_t)(r * 32 + lane) * vd + v_idx] = s[r];
  else if (s[0] == -1234.f) state_head[0] = s[0];
}


// ============================================================================
// V6 —— **改布局重写**(用户裁决路线):C 个相邻 v 列/warp + 向量装载 + 分组写
//
// 依据归因表(文件头):三个成本中心都在**布局**上,不在算术上:
//   ① 归约:C 列共用一次 32-lane 归约(每列 2.5 次 shfl,原 10 次);
//   ② q/k:一个 warp 覆盖 C 列 ⇒ 同一份 q/k 摊给 C 列(原块内 8 warp 各读一遍);
//   ③ out:C 个相邻列合成 **2C 字节** 一条写(原每列一条 2B)。
// 另:g/beta 是**每 head 一个标量**(不是每列),所以每 token 只取 1 次。
//
// ⚠️ **算术序与 V0 逐位相同**:每列的 r 循环序、`__shfl_down_sync` 的 5 步树、
//    fma 顺序都没动 ⇒ 判据 = out/state **逐字节相同**(不是容差)。
// ============================================================================
template <int C, int D>
__global__ void k_v6(const __half *__restrict__ q, const __half *__restrict__ k,
                     const __half *__restrict__ v, const __half *__restrict__ g,
                     const __half *__restrict__ beta, float *__restrict__ state,
                     const float *__restrict__ slots, const float *__restrict__ cu_seqlens,
                     size_t batch, size_t nv, size_t nk, size_t kd, size_t vd, float q_scale,
                     __half *__restrict__ out) {
  constexpr int R = 4;
  const int lane = threadIdx.x, warp_id = threadIdx.y;
  const int vcol0 = (blockIdx.x * WARPS + warp_id) * C;
  const int v_head_idx = blockIdx.y % nv;
  if (vcol0 >= (int)vd) return;
  const int k_head_idx = v_head_idx / (int)(nv / nk);
  if ((int)slots[0] < 0) return;
  const int start = (int)cu_seqlens[0], end = (int)cu_seqlens[1];
  const int seq_len = end - start;
  if (seq_len <= 0) return;
  const int ts_qk = (int)(nk * kd), ts_v = (int)(nv * vd);
  const __half *q_base = q + (size_t)start * ts_qk + (size_t)k_head_idx * kd + (size_t)lane;
  const __half *k_base = k + (size_t)start * ts_qk + (size_t)k_head_idx * kd + (size_t)lane;
  const __half *v_base = v + (size_t)start * ts_v + (size_t)v_head_idx * vd + vcol0;
  const __half *g_base = g + (size_t)start * nv + v_head_idx;
  const __half *bt_base = beta + (size_t)start * nv + v_head_idx;
  __half *out_base = out + (size_t)start * ts_v + (size_t)v_head_idx * vd + vcol0;
  float *st = state + ((size_t)(int)slots[0] * nv + v_head_idx) * kd * vd + vcol0;

  float s[R][C];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int c = 0; c < C; ++c) s[r][c] = st[(size_t)(r * 32 + lane) * vd + c];

  __half qr[D][R], kr[D][R], vr[D][C], gr[D], br[D];

#define OWL6_LOAD(stage, tok)                                                                   \
  do {                                                                                          \
    if ((tok) < seq_len) {                                                                      \
      const size_t o = (size_t)(tok) * ts_qk;                                                   \
      _Pragma("unroll") for (int r = 0; r < R; ++r) {                                           \
        qr[stage][r] = q_base[o + (size_t)r * 32];                                              \
        kr[stage][r] = k_base[o + (size_t)r * 32];                                              \
      }                                                                                         \
      const size_t ov = (size_t)(tok) * ts_v;                                                   \
      _Pragma("unroll") for (int c = 0; c < C; ++c) vr[stage][c] = v_base[ov + c];              \
      gr[stage] = g_base[(size_t)(tok) * nv];                                                   \
      br[stage] = bt_base[(size_t)(tok) * nv];                                                  \
    }                                                                                           \
  } while (0)

#define OWL6_COMPUTE(stage, tok)                                                                \
  do {                                                                                          \
    const float decay = expf(gdn_to_f(gr[stage]));                                              \
    const float b_t = gdn_to_f(br[stage]);                                                      \
    float kv[C];                                                                                \
    _Pragma("unroll") for (int c = 0; c < C; ++c) kv[c] = 0.f;                                  \
    _Pragma("unroll") for (int r = 0; r < R; ++r) {                                             \
      const float kv_ = gdn_to_f(kr[stage][r]);                                                 \
      _Pragma("unroll") for (int c = 0; c < C; ++c) {                                           \
        s[r][c] *= decay;                                                                       \
        kv[c] = __fmaf_rn(s[r][c], kv_, kv[c]);                                                 \
      }                                                                                         \
    }                                                                                           \
    _Pragma("unroll") for (int c = 0; c < C; ++c) kv[c] = gdn_warp_sum(kv[c]);                  \
    float dl[C], y[C];                                                                          \
    _Pragma("unroll") for (int c = 0; c < C; ++c) {                                             \
      dl[c] = (gdn_to_f(vr[stage][c]) - kv[c]) * b_t;                                           \
      y[c] = 0.f;                                                                               \
    }                                                                                           \
    _Pragma("unroll") for (int r = 0; r < R; ++r) {                                             \
      const float kv_ = gdn_to_f(kr[stage][r]);                                                 \
      const float qv_ = gdn_to_f(qr[stage][r]) * q_scale;                                       \
      _Pragma("unroll") for (int c = 0; c < C; ++c) {                                           \
        s[r][c] = __fmaf_rn(kv_, dl[c], s[r][c]);                                                \
        y[c] = __fmaf_rn(s[r][c], qv_, y[c]);                                                   \
      }                                                                                         \
    }                                                                                           \
    _Pragma("unroll") for (int c = 0; c < C; ++c) y[c] = gdn_warp_sum(y[c]);                    \
    if (lane == 0) {                                                                            \
      _Pragma("unroll") for (int c = 0; c < C; ++c)                                             \
        out_base[(size_t)(tok) * ts_v + c] = gdn_from_f(y[c]);                                  \
    }                                                                                           \
  } while (0)

#pragma unroll
  for (int d = 0; d < D; ++d) OWL6_LOAD(d, d);
  int t0 = 0;
  for (; t0 + D <= seq_len; t0 += D) {
#pragma unroll
    for (int stg = 0; stg < D; ++stg) {
      OWL6_COMPUTE(stg, t0 + stg);
      OWL6_LOAD(stg, t0 + stg + D);
    }
  }
  for (int t = t0; t < seq_len; ++t) {
    OWL6_LOAD(0, t);
    OWL6_COMPUTE(0, t);
  }
#undef OWL6_LOAD
#undef OWL6_COMPUTE

#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int c = 0; c < C; ++c) st[(size_t)(r * 32 + lane) * vd + c] = s[r][c];
}

// ============================================================================
// 台子
// ============================================================================
struct Args {
  const __half *q; const __half *k; const __half *v; const __half *g; const __half *beta;
  float *state; const float *slots; const float *cu; __half *out;
  int nv, nk, kd, vd; float q_scale;
};

template <typename F>
static float run(F launch, const Args &a, int T, int iters) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int w = 0; w < 3; ++w) launch();
  CK(cudaDeviceSynchronize());
  // 预热:让 boost 落定
  float t0 = 0.f; int n = 0;
  cudaEventRecord(e0);
  while (true) {
    launch(); ++n;
    if (n % 10 == 0) {
      cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
      cudaEventElapsedTime(&t0, e0, e1);
      if (t0 > 2000.0) break;   // ≳2s 预热
    }
  }
  CK(cudaDeviceSynchronize());
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) launch();
  cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
  float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1);
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  return ms * 1e3f / (float)iters;   // µs/发
}

static void fill(std::vector<__half> &v, uint32_t seed, float scale, float off) {
  for (size_t i = 0; i < v.size(); ++i) {
    uint32_t x = seed * 0x9E3779B9u + (uint32_t)i * 0x85EBCA6Bu;
    float u = ((x >> 8) & 0x00FFFFFFu) / 16777216.0f;
    v[i] = __float2half(u * scale + off);
  }
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "用法: %s <ord> [nv=24] [nk=8] [T...]\n", argv[0]); return 2; }
  const int ord = atoi(argv[1]);
  int nv = argc > 2 ? atoi(argv[2]) : 24;
  int nk = argc > 3 ? atoi(argv[3]) : 8;
  std::vector<int> Ts;
  for (int i = 4; i < argc; ++i) Ts.push_back(atoi(argv[i]));
  if (Ts.empty()) { Ts = {128, 1024}; }
  const int kd = 128, vd = 128, T = Ts.back();

  CK(cudaSetDevice(ord));
  int clk = 0; CK(cudaDeviceGetAttribute(&clk, cudaDevAttrClockRate, ord));  // kHz(标称)

  const size_t n_qk = (size_t)T * nk * kd, n_v = (size_t)T * nv * vd, n_g = (size_t)T * nv;
  std::vector<__half> hq(n_qk), hk(n_qk), hv(n_v), hg(n_g), hb(n_g);
  std::vector<float> hs((size_t)nv * kd * vd);
  fill(hq, 0x51, 0.1f, -0.05f); fill(hk, 0x52, 0.1f, -0.05f); fill(hv, 0x53, 0.1f, -0.05f);
  fill(hg, 0x54, 0.49f, -0.5f); fill(hb, 0x55, 0.8f, 0.1f);
  for (size_t i = 0; i < hs.size(); ++i) {
    uint32_t x = 0x56u * 0x9E3779B9u + (uint32_t)i * 0x85EBCA6Bu;
    hs[i] = (((x >> 8) & 0x00FFFFFFu) / 16777216.0f - 0.5f) * 0.02f;
  }

  __half *dq, *dk, *dv, *dg, *db, *dout;
  float *ds, *dslot, *dcu;
  CK(cudaMalloc(&dq, n_qk * 2)); CK(cudaMalloc(&dk, n_qk * 2)); CK(cudaMalloc(&dv, n_v * 2));
  CK(cudaMalloc(&dg, n_g * 2)); CK(cudaMalloc(&db, n_g * 2));
  CK(cudaMalloc(&ds, (size_t)nv * kd * vd * 4));
  CK(cudaMalloc(&dslot, 4)); CK(cudaMalloc(&dcu, 8)); CK(cudaMalloc(&dout, n_v * 2));
  CK(cudaMemcpy(dq, hq.data(), n_qk * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dk, hk.data(), n_qk * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dv, hv.data(), n_v * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dg, hg.data(), n_g * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, hb.data(), n_g * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(ds, hs.data(), (size_t)nv * kd * vd * 4, cudaMemcpyHostToDevice));
  float zero = 0.f, cu[2] = {0.f, (float)T};
  CK(cudaMemcpy(dslot, &zero, 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcu, cu, 8, cudaMemcpyHostToDevice));

  const float q_scale = 1.0f / sqrtf((float)kd);
  dim3 grid(vd / WARPS, nv), blk(32, WARPS);
  Args a{dq, dk, dv, dg, db, ds, dslot, dcu, dout, nv, nk, kd, vd, q_scale};

  printf("[probe] ord=%d nv=%d nk=%d nominal-sm=%d MHz grid(%d,%d) T=%d\n", ord, nv, nk,
         clk / 1000, grid.x, grid.y, T);

  // ── 参考输出(V0)──────────────────────────────────────────────────────────
  std::vector<__half> ref_out(n_v);
  std::vector<float> ref_state((size_t)nv * kd * vd);
  auto launch_v0 = [&] {
    k_v0<<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd, vd,
                        q_scale, a.out);
  };
  // 每臂都要从同一初态出发 ⇒ 每发前把 state 复位(只在正确性检查里做)
  std::vector<float> hs0 = hs;
  auto reset = [&] { CK(cudaMemcpy(ds, hs0.data(), hs0.size() * 4, cudaMemcpyHostToDevice)); };
  reset(); launch_v0(); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ref_out.data(), dout, n_v * 2, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(ref_state.data(), ds, hs0.size() * 4, cudaMemcpyDeviceToHost));

  for (int Tn : Ts) {
    printf("\n  T=%d\n", Tn);
    auto bench = [&](const char *tag, auto launch) {
      launch(); CK(cudaDeviceSynchronize());
      float us = run(launch, a, Tn, 10);
      printf("    %-14s %9.1f µs/发   %6.3f µs/token   %6.2f µs/token@nv24\n", tag, us,
             us / (float)Tn, us / (float)Tn * 24.f / (float)nv);
    };
    // 注意:所有臂共用同一 T ⇒ 下面的正确性检查只对最后一个 T 有意义
    auto launch_v1 = [&] {
      k_v1<<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd, vd,
                          q_scale, a.out);
    };
    auto launch_v2 = [&] {
      k_v4<2><<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd,
                             vd, q_scale, a.out);
    };
    auto launch_v3 = [&] {
      k_v4<4><<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd,
                             vd, q_scale, a.out);
    };
    auto launch_v5 = [&] {
      k_v4<6><<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd,
                             vd, q_scale, a.out);
    };
    bench("V0 现役", launch_v0);
    bench("V1 warp自治", launch_v1);
    bench("V2 预取2", launch_v2);
    bench("V3 预取4", launch_v3);
    bench("V5 预取6", launch_v5);
    auto launch_d0 = [&] {
      k_diag<1, 0><<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd,
                                  vd, q_scale, a.out);
    };
    auto launch_d1 = [&] {
      k_diag<1, 1><<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd,
                                  vd, q_scale, a.out);
    };
    bench("D0 去归约(错)", launch_d0);
    bench("D1 去expf(错)", launch_d1);
    auto l0 = [&]{ k_ladder<0><<<grid, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    auto l1 = [&]{ k_ladder<1><<<grid, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    auto l2 = [&]{ k_ladder<2><<<grid, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    auto l3 = [&]{ k_ladder<3><<<grid, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    auto l4 = [&]{ k_ladder<4><<<grid, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    bench("B0 无归约", l0); bench("B1 -q/k装载", l1); bench("B2 -v装载", l2);
    bench("B3 -out写", l3); bench("B4 -state口", l4);
    auto mk6 = [&](int c) { return dim3((unsigned)(((int)vd + c * WARPS - 1) / (c * WARPS)), (unsigned)nv); };
    dim3 g1 = mk6(1), g2 = mk6(2), g4 = mk6(4);
    auto s61 = [&]{ k_v6<1,1><<<g1, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    auto s62 = [&]{ k_v6<1,2><<<g1, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    auto s64 = [&]{ k_v6<4,1><<<g4, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    auto s64d2 = [&]{ k_v6<4,2><<<g4, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    auto s62d2 = [&]{ k_v6<2,2><<<g2, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    bench("V6 C=1 D=1", s61); bench("V6 C=1 D=2", s62);
    auto s62d3 = [&]{ k_v6<2,3><<<g2, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    auto s62d4 = [&]{ k_v6<2,4><<<g2, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    auto s64d3 = [&]{ k_v6<4,3><<<g4, blk>>>(a.q,a.k,a.v,a.g,a.beta,a.state,a.slots,a.cu,1,nv,nk,kd,vd,q_scale,a.out); };
    bench("V6 C=2 D=2", s62d2); bench("V6 C=2 D=3", s62d3); bench("V6 C=2 D=4", s62d4);
    bench("V6 C=4 D=2", s64d2); bench("V6 C=4 D=3", s64d3);
    bench("V6 C=4 D=1", s64); bench("V6 C=4 D=2", s64d2); bench("V6 C=4 D=3", s64d3);
  }

  // ── 正确性:各臂 out/state 与 V0 逐字节 ─────────────────────────────────
  auto check = [&](const char *tag, auto launch) {
    reset(); launch(); CK(cudaDeviceSynchronize());
    std::vector<__half> o(n_v);
    std::vector<float> st(hs0.size());
    CK(cudaMemcpy(o.data(), dout, n_v * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(st.data(), ds, hs0.size() * 4, cudaMemcpyDeviceToHost));
    size_t bad_o = 0, bad_s = 0, first = (size_t)-1;
    for (size_t i = 0; i < o.size(); ++i)
      if (__half_as_ushort(o[i]) != __half_as_ushort(ref_out[i])) { if (first == (size_t)-1) first = i; ++bad_o; }
    for (size_t i = 0; i < st.size(); ++i) {
      uint32_t x, y;
      memcpy(&x, &st[i], 4); memcpy(&y, &ref_state[i], 4);
      if (x != y) ++bad_s;
    }
    double mo = 0.0, ms = 0.0;
    for (size_t i = 0; i < o.size(); ++i)
      mo = fmax(mo, fabs((double)__half2float(o[i]) - (double)__half2float(ref_out[i])));
    for (size_t i = 0; i < st.size(); ++i) ms = fmax(ms, fabs((double)st[i] - (double)ref_state[i]));
    printf("    ref[0..3]=%.6f %.6f %.6f %.6f | got=%.6f %.6f %.6f %.6f\n",
           (double)__half2float(ref_out[0]), (double)__half2float(ref_out[1]),
           (double)__half2float(ref_out[2]), (double)__half2float(ref_out[3]),
           (double)__half2float(o[0]), (double)__half2float(o[1]),
           (double)__half2float(o[2]), (double)__half2float(o[3]));
    printf("  %-14s out 不同 %zu / %zu (首个 %zd, max|Δ|=%.3e) · state 不同 %zu / %zu (max|Δ|=%.3e)\n",
           tag, bad_o, o.size(), (ptrdiff_t)first, mo, bad_s, st.size(), ms);
  };
  printf("\n  ── 逐字节对拍(基准 = V0)──\n");
  check("V0 自身再跑", launch_v0);
  check("V1 warp自治", [&] {
    k_v1<<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd, vd,
                        q_scale, a.out);
  });
  check("V2 预取2", [&] {
    k_v4<2><<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd, vd,
                           q_scale, a.out);
  });
  check("V3 预取4", [&] {
    k_v4<4><<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd, vd,
                           q_scale, a.out);
  });
  check("V6 C=1 D=1", [&] {
    k_v6<1,1><<<dim3((unsigned)(((int)vd + WARPS - 1) / WARPS), (unsigned)nv), blk>>>(
        a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd, vd, q_scale, a.out);
  });
  check("V6 C=4 D=2", [&] {
    k_v6<4,2><<<dim3((unsigned)(((int)vd + 4*WARPS - 1) / (4*WARPS)), (unsigned)nv), blk>>>(
        a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd, vd, q_scale, a.out);
  });
  check("V6 C=2 D=2", [&] {
    k_v6<2,2><<<dim3((unsigned)(((int)vd + 2*WARPS - 1) / (2*WARPS)), (unsigned)nv), blk>>>(
        a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd, vd, q_scale, a.out);
  });
  check("V5 预取6", [&] {
    k_v4<6><<<grid, blk>>>(a.q, a.k, a.v, a.g, a.beta, a.state, a.slots, a.cu, 1, nv, nk, kd, vd,
                           q_scale, a.out);
  });
  return 0;
}
