// ============================================================================
// INT8 V 的生产写者(2026-11-03,INT8 KV 生产接线)
//
// 布局(与 int8_prefill.cu 的读侧逐字同源):
//   vp:     [nb, Hkv, D, page/2] b16(low = 页内 key0, high = key1)
//   vscale: [nb, page, Hkv] f32(per token/head;沿 K 不再恒定)
//
// 2026-11-03 质量修复:V scale 从"每页/每头/8-dim"改为"每 token/每头"。
// 页内 32 token 共用一个 absmax 会把 needle token 压扁;per-token scale
// 直接跟 token 走。PV 端把 s_t 折进 P(P' = P·s_t,V 存 v/s_t),因此
// tensor-core PV 仍只需一个沿 K 恒定的 scale(=1),不需要额外 mma 或 V shadow。
//
// prefill:一个 block 负责 (逻辑页, kv_head);8 warp 各管两个 token 对,
// 每对一次写完整 uint16(b16 低/高字节),无需 block barrier。decode:单 token 追加。
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
}  // namespace

/// 整页统计 + 打包(prefill)。grid = (npages_logical, Hkv, 1),block = 256。
extern "C" __global__ void owl_v_pack_int8_prefill(
    const __half* __restrict__ v,       // [T, Hkv, D]
    const float* __restrict__ slots,    // [T] 物理槽
    int T,
    int Hkv,
    int D,
    int page,
    float* __restrict__ vscale,         // [nb, page, Hkv]
    uint16_t* __restrict__ vp) {        // [nb, Hkv, D, page/2]
  const int lp = blockIdx.x;
  const int h = blockIdx.y;
  const int t0 = lp * page;
  if (t0 >= T) {
    return;
  }
  const int slot0 = static_cast<int>(slots[t0]);
  if (slot0 < 0) {
    return;
  }
  const int phys = slot0 / page;
  if ((slot0 % page) != 0) {
    return;
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n = min(page, T - t0);
  const int PP = page / 2;
  const int npair = (n + 1) / 2;
  // 一个 warp 负责一个 token 对：两 token 的 256 维 absmax 用 warp shuffle，
  // 且同一 warp 写同一 uint16 的两个字节 ⇒ 无跨 warp 读改写竞争、无 block barrier。
  for (int pair = warp; pair < npair; pair += kThreads / 32) {
    const int i0 = pair * 2;
    const int i1 = i0 + 1;
    const long long row0 = ((long long)(t0 + i0) * Hkv + h) * D;
    const long long row1 = ((long long)(t0 + i1) * Hkv + h) * D;
    const bool have1 = i1 < n;
    float x0[8], x1[8];
    float amax0 = 0.0f, amax1 = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = lane * 8 + j;
      x0[j] = __half2float(v[row0 + d]);
      amax0 = fmaxf(amax0, fabsf(x0[j]));
      if (have1) {
        x1[j] = __half2float(v[row1 + d]);
        amax1 = fmaxf(amax1, fabsf(x1[j]));
      } else {
        x1[j] = 0.0f;
      }
    }
    amax0 = warp_max(amax0);
    amax1 = warp_max(amax1);
    const float scale0 = fmaxf(amax0 / 127.0f, 1e-6f);
    const float scale1 = fmaxf(amax1 / 127.0f, 1e-6f);
    if (lane == 0) {
      vscale[((long long)(phys * page + i0) * Hkv) + h] = scale0;
      if (have1) {
        vscale[((long long)(phys * page + i1) * Hkv) + h] = scale1;
      }
    }
    const float inv0 = 1.0f / scale0;
    const float inv1 = 1.0f / scale1;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int d = lane * 8 + j;
      const uint8_t b0 = static_cast<uint8_t>(quant_v_i8(x0[j], inv0));
      const uint8_t b1 = static_cast<uint8_t>(quant_v_i8(x1[j], inv1));
      vp[(((long long)phys * Hkv + h) * D + d) * PP + pair] =
          static_cast<uint16_t>(b0 | (static_cast<uint16_t>(b1) << 8));
    }
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
    uint16_t* __restrict__ vp) {        // [nb, Hkv, D, page/2]
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int lane = threadIdx.x & 31;
  if (t >= T || threadIdx.x >= 32) {
    return;
  }
  const int slot = static_cast<int>(slots[t]);
  if (slot < 0) {
    return;
  }
  const int phys = slot / page;
  const int off = slot % page;
  const int PP = page / 2;
  const int pair = off / 2;
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
    vscale[((long long)slot * Hkv) + h] = scale;
  }
  const float inv = 1.0f / scale;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int d = lane * 8 + j;
    const uint8_t byte = static_cast<uint8_t>(quant_v_i8(x[j], inv));
    uint16_t* word = &vp[(((long long)phys * Hkv + h) * D + d) * PP + pair];
    if ((off & 1) == 0) {
      *word = static_cast<uint16_t>((*word & 0xFF00u) | byte);
    } else {
      *word = static_cast<uint16_t>((*word & 0x00FFu) | (static_cast<uint16_t>(byte) << 8));
    }
  }
}
