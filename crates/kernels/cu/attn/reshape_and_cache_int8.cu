// ============================================================================
// owl_reshape_and_cache_int8_knhd(2026-11-03,INT8 KV 专项 P0)
//
// 批量 KV 写池:int8 主池 + per-(token, kv_head) fp32 scale。
//
// 布局(knHD,与 fp8 变体同源):
//   K/V  : [num_blocks, block_size, num_heads, head_size] int8
//   scale: [num_blocks, block_size, num_heads]            fp32
//
// 量化约定(pth / vLLM int8_per_token_head writer 的镜像,必须逐字节一致):
//   scale = max(max|x| / 127, 1e-6)
//   q     = x * (1/scale)
//   q     = q >= 0 ? q + 0.5 : q - 0.5
//   q     = clamp(q, -128, 127)
//   store int8(q)            // C cast:向零截断
//
// 线程映射:一个 warp 负责一个 (token, kv_head) 行。D=256 时每 lane 8 个
// 元素;warp 内 absmax 归约一次,再量化写回。一个 block 4 warps。
//
// FIXME(int8-kv):scale 目前作为独立 tensor 传入,尚未声明成池写者;
// P0 接池时必须把 scale plane 纳入 PoolAccess/E10 写集审计。
// ============================================================================
#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kThreads = kWarpsPerBlock * 32;

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

__device__ __forceinline__ int8_t quant_i8(float x, float inv_scale) {
  float q = x * inv_scale;
  q = (q >= 0.0f) ? (q + 0.5f) : (q - 0.5f);
  q = fminf(fmaxf(q, -128.0f), 127.0f);
  return static_cast<int8_t>(q);
}

}  // namespace

extern "C" __global__ void owl_reshape_and_cache_int8_knhd(
    const __half* __restrict__ key,       // [num_tokens, num_heads, head_size]
    const __half* __restrict__ value,     // [num_tokens, num_heads, head_size]
    int8_t* __restrict__ key_cache,       // [nb, block_size, num_heads, head_size]
    int8_t* __restrict__ value_cache,     // [nb, block_size, num_heads, head_size]
    float* __restrict__ k_scale,          // [nb, block_size, num_heads]
    float* __restrict__ v_scale,          // [nb, block_size, num_heads]
    const float* __restrict__ slot_mapping,
    const int key_stride,
    const int value_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int num_tokens,
    __half* __restrict__ out) {
  (void)out;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const long long unit =
      static_cast<long long>(blockIdx.x) * kWarpsPerBlock + warp;
  const long long units = static_cast<long long>(num_tokens) * num_heads;
  if (unit >= units) {
    return;
  }

  const int token = static_cast<int>(unit / num_heads);
  const int head = static_cast<int>(unit % num_heads);
  const long long slot = static_cast<long long>(slot_mapping[token]);
  if (slot < 0) {
    return;
  }

  const long long block = slot / block_size;
  const long long offset = slot % block_size;
  const long long row_base =
      (block * block_size + offset) * num_heads * head_size +
      static_cast<long long>(head) * head_size;
  const long long scale_idx =
      (block * block_size + offset) * num_heads + head;

  const long long src = static_cast<long long>(token) * key_stride +
                        static_cast<long long>(head) * head_size;

  float k_abs = 0.0f;
  float v_abs = 0.0f;
  for (int d = lane; d < head_size; d += 32) {
    k_abs = fmaxf(k_abs, fabsf(__half2float(key[src + d])));
    v_abs = fmaxf(v_abs, fabsf(__half2float(value[src + d])));
  }
  k_abs = warp_max(k_abs);
  v_abs = warp_max(v_abs);

  const float ks = fmaxf(k_abs / 127.0f, 1e-6f);
  const float vs = fmaxf(v_abs / 127.0f, 1e-6f);
  const float kinv = 1.0f / ks;
  const float vinv = 1.0f / vs;

  for (int d = lane; d < head_size; d += 32) {
    key_cache[row_base + d] = quant_i8(__half2float(key[src + d]), kinv);
    value_cache[row_base + d] = quant_i8(__half2float(value[src + d]), vinv);
  }
  if (lane == 0) {
    k_scale[scale_idx] = ks;
    v_scale[scale_idx] = vs;
  }
}

extern "C" __global__ void owl_reshape_and_cache_int8_k_only_knhd(
    const __half* __restrict__ key,       // [num_tokens, num_heads, head_size]
    int8_t* __restrict__ key_cache,       // [nb, block_size, num_heads, head_size]
    float* __restrict__ k_scale,          // [nb, block_size, num_heads]
    const float* __restrict__ slot_mapping,
    const int key_stride,
    const int num_heads,
    const int head_size,
    const int block_size,
    const int num_tokens,
    __half* __restrict__ out) {
  (void)out;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const long long unit =
      static_cast<long long>(blockIdx.x) * kWarpsPerBlock + warp;
  const long long units = static_cast<long long>(num_tokens) * num_heads;
  if (unit >= units) {
    return;
  }

  const int token = static_cast<int>(unit / num_heads);
  const int head = static_cast<int>(unit % num_heads);
  const long long slot = static_cast<long long>(slot_mapping[token]);
  if (slot < 0) {
    return;
  }

  const long long block = slot / block_size;
  const long long offset = slot % block_size;
  const long long row_base =
      (block * block_size + offset) * num_heads * head_size +
      static_cast<long long>(head) * head_size;
  const long long scale_idx =
      (block * block_size + offset) * num_heads + head;
  const long long src = static_cast<long long>(token) * key_stride +
                        static_cast<long long>(head) * head_size;

  float gmax[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  for (int d = lane; d < head_size; d += 32) {
    const int g = d / 32;
    gmax[g] = fmaxf(gmax[g], fabsf(__half2float(key[src + d])));
  }
#pragma unroll
  for (int g = 0; g < 8; ++g) gmax[g] = warp_max(gmax[g]);

  float ks[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) ks[g] = fmaxf(gmax[g] / 127.0f, 1e-6f);
  for (int d = lane; d < head_size; d += 32) {
    key_cache[row_base + d] = quant_i8(__half2float(key[src + d]), 1.0f / ks[d / 32]);
  }
  if (lane < 8) {
    k_scale[scale_idx * 8 + lane] = ks[lane];
  }
}
