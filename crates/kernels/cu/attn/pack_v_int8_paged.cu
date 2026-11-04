// ============================================================================
// INT8 V 的生产写者(2026-11-03,INT8 KV 生产接线;2026-11-04 token-major)
//
// 布局:
//   v:      [nb, page, Hkv, D] int8       (token-major;每 token 的 D 维连续)
//   vscale: [nb, page, Hkv] f32           (per token/head)
//
// 2026-11-03 质量修复:V scale 从"每页/每头/8-dim"改为"每 token/每头"。
// 页内 32 token 共用一个 absmax 会把 needle token 压扁;per-token scale
// 直接跟 token 走。PV 端把 s_t 折进 P(P' = P·s_t,V 存 v/s_t),因此
// tensor-core PV 仍只需一个沿 K 恒定的 scale(=1),不需要额外 mma 或 V shadow。
//
// 2026-11-04:PTH 对照实验证明 V 的 token-major 布局是 decode 的另一半
// 收益(65k split 181us -> 159us)。池字节数与 block_bytes 不变,只是把
// 同一页的 32KB 从 [Hkv,D,page/2] b16 重新解释为 [page,Hkv,D] int8;
// K/Kscale 完全不动。prefill 一个 block 负责 (逻辑页, kv_head),8 warp
// 各管一个 token;decode 单 token 直接写 D 字节,不再有 b16 RMW。
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
constexpr int kThreads = 256;

__device__ __forceinline__ int8_t quant_v_i8(float x, float inv) {
  float q = x * inv;
  q = (q >= 0.0f) ? (q + 0.5f) : (q - 0.5f);
  q = fminf(fmaxf(q, -127.0f), 127.0f);
  return static_cast<int8_t>(q);
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, off));
  }
  return x;
}

__device__ __forceinline__ uint2 pack_i8x8(const int8_t q[8]) {
  uint2 r;
  r.x = static_cast<uint32_t>(static_cast<uint8_t>(q[0])) |
        (static_cast<uint32_t>(static_cast<uint8_t>(q[1])) << 8) |
        (static_cast<uint32_t>(static_cast<uint8_t>(q[2])) << 16) |
        (static_cast<uint32_t>(static_cast<uint8_t>(q[3])) << 24);
  r.y = static_cast<uint32_t>(static_cast<uint8_t>(q[4])) |
        (static_cast<uint32_t>(static_cast<uint8_t>(q[5])) << 8) |
        (static_cast<uint32_t>(static_cast<uint8_t>(q[6])) << 16) |
        (static_cast<uint32_t>(static_cast<uint8_t>(q[7])) << 24);
  return r;
}
}  // namespace

/// 整页 V 量化(prefill)。grid = (npages_logical, Hkv, 1),block = 256。
extern "C" __global__ void owl_v_pack_int8_prefill(
    const __half* __restrict__ v,       // [T, Hkv, D]
    const float* __restrict__ slots,    // [T] 物理槽
    int T,
    int Hkv,
    int D,
    int page,
    float* __restrict__ vscale,         // [nb, page, Hkv]
    int8_t* __restrict__ v8) {          // [nb, page, Hkv, D]
  const int lp = blockIdx.x;
  const int h = blockIdx.y;
  const int t0 = lp * page;
  if (t0 >= T) {
    return;
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n = min(page, T - t0);
  // 一个 warp 负责一个 token:lane 读 8 个 half(16B),写 8 个 int8(8B)。
  // 同 warp 内 32 lane 覆盖 D=256,无跨 warp 读改写竞争,无需 block barrier。
  for (int i = warp; i < n; i += kThreads / 32) {
    const int slot = static_cast<int>(slots[t0 + i]);
    if (slot < 0) {
      continue;
    }
    const long long row = ((long long)(t0 + i) * Hkv + h) * D;
    float x[8];
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = lane * 8 + j;
      x[j] = __half2float(v[row + d]);
      amax = fmaxf(amax, fabsf(x[j]));
    }
    amax = warp_max(amax);
    const float scale = fmaxf(amax / 127.0f, 1e-6f);
    if (lane == 0) {
      vscale[(long long)slot * Hkv + h] = scale;
    }
    const float inv = 1.0f / scale;
    int8_t q[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      q[j] = quant_v_i8(x[j], inv);
    }
    // slot = phys*page + key ⇒ [key, Hkv, D] 的 token 行。
    int8_t* dst = v8 + ((long long)slot * Hkv + h) * D + lane * 8;
    *reinterpret_cast<uint2*>(dst) = pack_i8x8(q);
  }
}

/// 单 token 追加(decode)。grid = (T, Hkv, 1),block = 256。
extern "C" __global__ void owl_v_append_int8_paged(
    const __half* __restrict__ v,       // [T, Hkv, D]
    const float* __restrict__ slots,    // [T] 物理槽
    int T,
    int Hkv,
    int D,
    int page,
    float* __restrict__ vscale,         // [nb, page, Hkv]
    int8_t* __restrict__ v8,            // [nb, page, Hkv, D]
    const __half* __restrict__ dep,     // 树序依赖边(不解引用)
    __half* __restrict__ out) {         // 哑输出(槽序契约 4)
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int lane = threadIdx.x & 31;
  (void)dep;
  (void)out;
  if (t >= T || threadIdx.x >= 32) {
    return;
  }
  const int slot = static_cast<int>(slots[t]);
  if (slot < 0) {
    return;
  }
  const long long row = ((long long)t * Hkv + h) * D;
  float x[8];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int d = lane * 8 + j;
    x[j] = __half2float(v[row + d]);
    amax = fmaxf(amax, fabsf(x[j]));
  }
  amax = warp_max(amax);
  const float scale = fmaxf(amax / 127.0f, 1e-6f);
  if (lane == 0) {
    vscale[(long long)slot * Hkv + h] = scale;
  }
  const float inv = 1.0f / scale;
  int8_t q[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    q[j] = quant_v_i8(x[j], inv);
  }
  int8_t* dst = v8 + ((long long)slot * Hkv + h) * D + lane * 8;
  *reinterpret_cast<uint2*>(dst) = pack_i8x8(q);
}
