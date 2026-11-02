// ============================================================================
// INT8 V 的生产写者(2026-11-03,INT8 KV 生产接线)
//
// 布局(与 int8_prefill.cu 的读侧逐字同源):
//   vp:     [nb, Hkv, D, page/2] b16(low = 页内 key0, high = key1)
//   vscale: [nb, Hkv, 4] f32(每页/每头/64-dim group;沿 K 不要求跨页恒定)
//
// 两条路径:
//   * prefill:一个 block 负责 (逻辑页, kv_head, group);整页 token 一起统计
//     absmax 并打包。前提 = 页对齐 chunk(引擎侧保证 ctx_base % page == 0)。
//   * decode:一个 token 追加进页;页已存在则复用该页 scale(不重算、不重打包),
//     新页才由当前 token 统计 scale。b16 的低/高字节按页内 offset 写。
//
// 为什么 V scale 改成**每页**而不是全局 [Hkv,4]:
//   chunked prefill / prefix cache / decode 追加都会让同一序列的 V 分多次写入;
//   全局单 scale 要么要求全序列先统计(不可行),要么让旧页用新 scale(错值)。
//   每页 scale 与页同生命周期,prefill/decode/前缀缓存四件套都能自洽。
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

__device__ __forceinline__ float block_max(float x, float* smem) {
  smem[threadIdx.x] = x;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  return smem[0];
}
}  // namespace

/// 整页统计 + 打包(prefill)。grid = (npages_logical, Hkv, 4),block = 256。
extern "C" __global__ void owl_v_pack_int8_prefill(
    const __half* __restrict__ v,       // [T, Hkv, D]
    const float* __restrict__ slots,    // [T] 物理槽
    int T,
    int Hkv,
    int D,
    int page,
    float* __restrict__ vscale,         // [nb, Hkv, 4]
    uint16_t* __restrict__ vp) {        // [nb, Hkv, D, page/2]
  __shared__ float smem[kThreads];
  const int lp = blockIdx.x;
  const int h = blockIdx.y;
  const int g = blockIdx.z;
  const int t0 = lp * page;
  if (t0 >= T) {
    return;
  }
  const int slot0 = static_cast<int>(slots[t0]);
  if (slot0 < 0) {
    return;
  }
  const int phys = slot0 / page;
  // 引擎契约:chunk 起点页对齐 ⇒ 本页第一个 token 的页内 offset = 0。
  // 违反 = 结构性错配,不静默写半页(层侧已有 page 对齐门)。
  if ((slot0 % page) != 0) {
    return;
  }

  float amax = 0.0f;
  const int n = min(page, T - t0);
  for (int i = threadIdx.x; i < n * 64; i += kThreads) {
    const int t = t0 + i / 64;
    const int d = g * 64 + (i % 64);
    amax = fmaxf(amax, fabsf(__half2float(v[((long long)t * Hkv + h) * D + d])));
  }
  amax = block_max(amax, smem);
  const float scale = fmaxf(amax / 127.0f, 1e-6f);
  const float inv = 1.0f / scale;
  if (threadIdx.x == 0) {
    vscale[((long long)phys * Hkv + h) * 4 + g] = scale;
  }

  const int PP = page / 2;
  for (int i = threadIdx.x; i < 64 * PP; i += kThreads) {
    const int d = g * 64 + i / PP;
    const int pair = i % PP;
    const int key0 = t0 + pair * 2;
    const int key1 = key0 + 1;
    const float x0 = (key0 < T)
        ? __half2float(v[((long long)key0 * Hkv + h) * D + d]) : 0.0f;
    const float x1 = (key1 < T)
        ? __half2float(v[((long long)key1 * Hkv + h) * D + d]) : 0.0f;
    const uint16_t a = static_cast<uint8_t>(quant_v_i8(x0, inv));
    const uint16_t b = static_cast<uint8_t>(quant_v_i8(x1, inv));
    vp[(((long long)phys * Hkv + h) * D + d) * PP + pair] = a | (b << 8);
  }
}

/// 单 token 追加(decode / 续写已有页)。grid = (T, Hkv, 4),block = 64。
/// 页已存在 ⇒ 复用该页 scale;新页(offset 0)⇒ 由当前 token 统计。
extern "C" __global__ void owl_v_append_int8_paged(
    const __half* __restrict__ v,       // [T, Hkv, D]
    const float* __restrict__ slots,    // [T] 物理槽
    int T,
    int Hkv,
    int D,
    int page,
    float* __restrict__ vscale,         // [nb, Hkv, 4]
    uint16_t* __restrict__ vp) {        // [nb, Hkv, D, page/2]
  __shared__ float smem[64];
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int g = blockIdx.z;
  if (t >= T || threadIdx.x >= 64) {
    return;
  }
  const int slot = static_cast<int>(slots[t]);
  if (slot < 0) {
    return;
  }
  const int phys = slot / page;
  const int off = slot % page;
  const int d = g * 64 + threadIdx.x;
  float* s = &vscale[((long long)phys * Hkv + h) * 4 + g];

  float scale;
  if (off == 0) {
    float amax = fabsf(__half2float(v[((long long)t * Hkv + h) * D + d]));
    smem[threadIdx.x] = amax;
    __syncthreads();
#pragma unroll
    for (int stride = 32; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x + stride]);
      }
      __syncthreads();
    }
    scale = fmaxf(smem[0] / 127.0f, 1e-6f);
    if (threadIdx.x == 0) {
      *s = scale;
    }
  } else {
    scale = *s;
    if (!(scale > 0.0f)) {
      // 旧页无 scale = 结构损坏;用当前 token 兜底会让旧字节错读,故退化为
      // 不写(层侧不会制造这种页;此处避免除零/NaN 扩散)。
      return;
    }
  }

  const float inv = 1.0f / scale;
  const uint8_t byte = static_cast<uint8_t>(quant_v_i8(
      __half2float(v[((long long)t * Hkv + h) * D + d]), inv));
  const int PP = page / 2;
  const int pair = off / 2;
  uint16_t* word = &vp[(((long long)phys * Hkv + h) * D + d) * PP + pair];
  if ((off & 1) == 0) {
    *word = static_cast<uint16_t>((*word & 0xFF00u) | byte);
  } else {
    *word = static_cast<uint16_t>((*word & 0x00FFu) | (static_cast<uint16_t>(byte) << 8));
  }
}
