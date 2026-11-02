// ============================================================================
// owl_quant_int8_qk_f16(2026-11-03,INT8 KV 生产接线)
//
// Q 的 per-(token, q_head) int8 量化。语义与实验核
// experiments/int8pv/full_4w_i8pv_correct.cu::quant_row 逐点一致:
//   scale = max(absmax / 127, 1e-6)
//   q     = clamp(round_half_away(x / scale), -127, 127)
//
// 几何:8 warp/block,一个 warp 负责一行 D=256。
// 为什么不是一 block 一行(首版):生产 prefill T=2048 × Hq=32 ⇒ 每层
// 65k 个 block,18 层 = 百万级小发射,实测把 prefill 从 ~3k 拉到 1.8k t/s。
// 本版把网格压到 rows/8,每 warp 8 个元素/通道 + shfl 归约。
// ============================================================================
#include <cuda_fp16.h>
#include <cstdint>

namespace {
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

__device__ __forceinline__ int8_t quant_i8_qk(float x, float inv) {
  float q = x * inv;
  q = (q >= 0.0f) ? (q + 0.5f) : (q - 0.5f);
  q = fminf(fmaxf(q, -127.0f), 127.0f);
  return static_cast<int8_t>(q);
}
}  // namespace

extern "C" __global__ void owl_quant_int8_qk_f16(
    const __half* __restrict__ x,  // [rows, cols]
    int rows,
    int cols,
    float* __restrict__ scales,   // [rows]
    int8_t* __restrict__ q) {     // [rows, cols]
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * kWarps + warp;
  if (row >= rows) {
    return;
  }
  const __half* src = x + static_cast<long long>(row) * cols;
  float amax = 0.0f;
  for (int d = lane; d < cols; d += 32) {
    amax = fmaxf(amax, fabsf(__half2float(src[d])));
  }
  amax = warp_max(amax);
  const float scale = fmaxf(amax / 127.0f, 1e-6f);
  const float inv = 1.0f / scale;
  int8_t* dst = q + static_cast<long long>(row) * cols;
  for (int d = lane; d < cols; d += 32) {
    dst[d] = quant_i8_qk(__half2float(src[d]), inv);
  }
  if (lane == 0) {
    scales[row] = scale;
  }
}
