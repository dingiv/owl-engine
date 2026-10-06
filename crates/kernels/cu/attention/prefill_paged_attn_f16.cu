// ============================================================================
// chunked prefill paged attention(K1 prefill 线;2026-09-27 port)
//
// 出处(vendor file-level 复用,Apache-2.0):
//   packages/xinfer/vendor/attention.rs/src/kernels/src/prefill_paged_attn_opt.cu
//   (rev c0f19f2,冻结只读;attention-rs 自研 chunked prefill 核:smem tile
//   协作装 K/V + 在线 softmax(逐块重缩放)+ 可选滑动窗 + ALiBi/sinks +
//   块边界跨序列处理;A4 纪律:头注 + 语义保持,f16 特化见逐条认领)
//
// port 适配(显式认领;与 pagedattention_f16.cu 同款者注「同前」):
//   ① stdint.h → 局部 typedef(同前);
//   ② extern "C" 特化入口(同前);
//   ③ f16/f16 特化,is_quantized 常量化 false,FP8 臂剥除,scales 形参随删;
//   ④ int 表 → f32 过线(契约 5,同前):block_tables/seq_lens/
//      query_start_len;空块哨兵 UINT32_MAX → -1(f32 负值);
//   ⑤ alibi/sinks nullptr 判定 → use_alibi_flag/use_sinks_flag 整参旗标
//      (owl 发射器无 null 块指针;旗标 0 不解引用,可挂任意块);
//   ⑥ 尾参 = 输出块(契约 4,同前):out 为真实输出,排末位;
//   ⑦ host launcher/dim3/cudaFuncSetAttribute 段剥除(owl 发射器接管;
//      smem ≤48KB 时无需 attr 扩展 —— HD256×BLOCK32×f16 = 32KB+64B ✓;
//      ⑧ 核体并入 namespace vllm(原名 vllm_rs;helper 全集在 vllm,
//         未限定名查找才可见;extern "C" 入口同步改调 vllm::);
//      更大组合需发射器 attr 通道,挂账);
//   ⑧ make_zero_opt(memset 版)未消费,删除。
//
// 布局契约(同 pagedattention_f16.cu):classic 布局 + f32 块表;
//   q/out [num_query_tokens, Hq, hd];query_start_len [num_seqs+1] cu-seqlens;
//   支持一次 chunk 跨两个序列(块边界路径:越界线程独立走 global 读)。
//
// grid / shared_mem 契约(发射方计算;禁哨兵):
//   grid  (num_queries_per_kv, num_kv_heads, ceil(num_query_tokens / 256))
//   block (256, 1, 1)(= TOKEN_CHUNK_SIZE)
//   smem  64 + 2 * HEAD_SIZE * BLOCK_SIZE * 2B
//   模板不变量:NUM_THREADS == TOKEN_CHUNK_SIZE(lane = tid % CHUNK)
// ============================================================================
#include <cuda_fp16.h>

// ---- nvrtc 序言(mini-stdint + 常量 + 宏;与 pagedattention_f16.cu 同款)----
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#ifndef INFINITY
#define INFINITY (__int_as_float(0x7f800000))
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
#define VLLM_LDG(arg) __ldg(arg)
#define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)

// ---- 展平支持层(逐字同源;includes 由本文件序言替代)----
// attention_generic.cuh
namespace vllm {


// A vector type to store Q, K, V elements.
template<typename T, int VEC_SIZE>
struct Vec {};

// A vector type to store FP32 accumulators.
template<typename T>
struct FloatVec {};

// Template vector operations.
template<typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b);

template<typename T>
inline __device__ float sum(T v);

template<typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<T, T, T>(a, b));
}

template<typename A, typename T>
inline __device__ float dot(T a, T b) {
  return sum(mul<A, T, T>(a, b));
}

template<typename T>
inline __device__ void zero(T& dst) {
  constexpr int WORDS = sizeof(T) / 4;
  union {
    T raw;
    uint32_t words[WORDS];
  } tmp;

#pragma unroll
  for (int ii = 0; ii < WORDS; ++ii) {
    tmp.words[ii] = 0u;
  }
  dst = tmp.raw;
}


} // namespace vllm

// dtype_float32.cuh
namespace vllm {


// Define custom FP32 vector data types.
struct Float4_ {
  float2 x;
  float2 y;
};

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

// FP32 vector types for Q, K, V.
template<>
struct Vec<float, 1> {
  using Type = float;
};
template<>
struct Vec<float, 2> {
  using Type = float2;
};
template<>
struct Vec<float, 4> {
  using Type = float4;
};

template<>
struct Vec<float, 8> {
  using Type = Float8_;
};

// FP32 accumulator vector types corresponding to Vec.
template<>
struct FloatVec<float> {
  using Type = float;
};
template<>
struct FloatVec<float2> {
  using Type = float2;
};
template<>
struct FloatVec<float4> {
  using Type = float4;
};

// Vector addition.
inline __device__ float add(float a, float b) {
  return a + b;
}

inline __device__ float2 add(float2 a, float2 b) {
  float2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ float4 add(float4 a, float4 b) {
  float4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

// Vector multiplication.
template<>
inline __device__ float mul<float, float>(float a, float b) {
  return a * b;
}

template<>
inline __device__ float2 mul(float2 a, float2 b) {
  float2 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  return c;
}

template<>
inline __device__ float2 mul(float a, float2 b) {
  float2 c;
  c.x = a * b.x;
  c.y = a * b.y;
  return c;
}

template<>
inline __device__ float4 mul(float4 a, float4 b) {
  float4 c;
  c.x = a.x * b.x;
  c.y = a.y * b.y;
  c.z = a.z * b.z;
  c.w = a.w * b.w;
  return c;
}

template<>
inline __device__ float4 mul(float a, float4 b) {
  float4 c;
  c.x = a * b.x;
  c.y = a * b.y;
  c.z = a * b.z;
  c.w = a * b.w;
  return c;
}

// Vector fused multiply-add.
inline __device__ float fma(float a, float b, float c) {
  return a * b + c;
}

inline __device__ float2 fma(float2 a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ float2 fma(float a, float2 b, float2 c) {
  float2 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline __device__ float4 fma(float4 a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ float4 fma(float a, float4 b, float4 c) {
  float4 d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c) {
  Float4_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  return d;
}

inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c) {
  Float8_ d;
  d.x = fma(a, b.x, c.x);
  d.y = fma(a, b.y, c.y);
  d.z = fma(a, b.z, c.z);
  d.w = fma(a, b.w, c.w);
  return d;
}

// Vector sum.
template<>
inline __device__ float sum(float v) {
  return v;
}

template<>
inline __device__ float sum(float2 v) {
  return v.x + v.y;
}

template<>
inline __device__ float sum(float4 v) {
  return v.x + v.y + v.z + v.w;
}

template<>
inline __device__ float sum(Float4_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y;
}

template<>
inline __device__ float sum(Float8_ v) {
  return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}

// Vector dot product.
inline __device__ float dot(float a, float b) {
  return a * b;
}

inline __device__ float dot(float2 a, float2 b) {
  float2 c = mul<float2, float2, float2>(a, b);
  return c.x + c.y;
}

inline __device__ float dot(Float4_ a, Float4_ b) {
  float2 acc = mul<float2, float2, float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  return acc.x + acc.y;
}

inline __device__ float dot(Float8_ a, Float8_ b) {
  float2 acc = mul<float2, float2, float2>(a.x, b.x);
  acc = fma(a.y, b.y, acc);
  acc = fma(a.z, b.z, acc);
  acc = fma(a.w, b.w, acc);
  return acc.x + acc.y;
}

// From float to float.
inline __device__ void from_float(float& dst, float src) {
  dst = src;
}

inline __device__ void from_float(float2& dst, float2 src) {
  dst = src;
}

inline __device__ void from_float(float4& dst, float4 src) {
  dst = src;
}

// From float to float.
inline __device__ float to_float(float u) {
  return u;
}

inline __device__ float2 to_float(float2 u) {
  return u;
}

inline __device__ float4 to_float(float4 u) {
  return u;
}

inline __device__ Float4_ to_float(Float4_ u) {
  return u;
}

inline __device__ Float8_ to_float(Float8_ u) {
  return u;
}

// Zero-out a variable.
inline __device__ void zero(float& dst) {
  dst = 0.f;
}


} // namespace vllm

// dtype_float16.cuh(f16 = uint16 位型 + PTX 算子;ROCM 臂由预处理器裁)
namespace vllm {


// FP16 vector types for Q, K, V.
template<>
struct Vec<uint16_t, 1> {
  using Type = uint16_t;
};
template<>
struct Vec<uint16_t, 2> {
  using Type = uint32_t;
};
template<>
struct Vec<uint16_t, 4> {
  using Type = uint2;
};
template<>
struct Vec<uint16_t, 8> {
  using Type = uint4;
};

// FP32 accumulator vector types corresponding to Vec.
template<>
struct FloatVec<uint16_t> {
  using Type = float;
};
template<>
struct FloatVec<uint32_t> {
  using Type = float2;
};
template<>
struct FloatVec<uint2> {
  using Type = Float4_;
};
template<>
struct FloatVec<uint4> {
  using Type = Float8_;
};

// Utility functions for type conversions.
inline __device__ uint32_t h0_h0(uint16_t a) {
#ifndef USE_ROCM
  uint32_t b;
  asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
  return b;
#else
  union {
   uint32_t u32;
   uint16_t u16[2];
  } tmp;
  tmp.u16[0] = a;
  tmp.u16[1] = a;
  return tmp.u32;
#endif
}

inline __device__ float half_to_float(uint16_t h) {
  float f;
#ifndef USE_ROCM
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
#else
  asm volatile("v_cvt_f32_f16 %0, %1;" : "=v"(f) : "v"(h));
#endif
  return f;
}

inline __device__ float2 half2_to_float2(uint32_t v) {
#ifndef USE_ROCM
  uint16_t lo, hi;
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
  return make_float2(half_to_float(lo), half_to_float(hi));
#else
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
  tmp.u32 = v;
  float2 ret;
  ret.x = half_to_float(tmp.u16[0]);
  ret.y = half_to_float(tmp.u16[1]);
  return ret;
#endif
}

inline __device__ uint16_t float_to_half(float f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
  asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#else
  asm volatile("v_cvt_f16_f32 %0, %1;\n" : "=v"(tmp.u32) : "v"(f));
#endif
  return tmp.u16[0];
}

inline __device__ uint32_t float2_to_half2(float2 f) {
  union {
    uint32_t u32;
    uint16_t u16[2];
  } tmp;
#ifndef USE_ROCM
  #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
  #else
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
  #endif
#else
  tmp.u16[0] = float_to_half(f.x);
  tmp.u16[1] = float_to_half(f.y);
#endif
  return tmp.u32;
}

// Vector addition.
inline __device__ uint16_t add(uint16_t a, uint16_t b) {
  uint16_t c;
#ifndef USE_ROCM
  asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
#else
  asm volatile("v_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

inline __device__ uint32_t add(uint32_t a, uint32_t b) {
  uint32_t c;
#ifndef USE_ROCM
  asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#else
  asm volatile("v_pk_add_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

inline __device__ uint2 add(uint2 a, uint2 b) {
  uint2 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}

inline __device__ uint4 add(uint4 a, uint4 b) {
  uint4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

inline __device__ float2 add(uint32_t a, float2 fb) {
  float2 fa = half2_to_float2(a);
  return add(fa, fb);
}

inline __device__ Float4_ add(uint2 a, Float4_ fb) {
  Float4_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  return fc;
}

inline __device__ Float8_ add(uint4 a, Float8_ fb) {
  Float8_ fc;
  fc.x = add(a.x, fb.x);
  fc.y = add(a.y, fb.y);
  fc.z = add(a.z, fb.z);
  fc.w = add(a.w, fb.w);
  return fc;
}

// Vector multiplication.
template<>
inline __device__ uint16_t mul(uint16_t a, uint16_t b) {
  uint16_t c;
#ifndef USE_ROCM
  asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
#else
  asm volatile("v_mul_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

template<>
inline __device__ uint32_t mul(uint32_t a, uint32_t b) {
  uint32_t c;
#ifndef USE_ROCM
  asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
#else
  asm volatile("v_pk_mul_f16 %0, %1, %2;\n" : "=v"(c) : "v"(a), "v"(b));
#endif
  return c;
}

template<>
inline __device__ uint32_t mul(uint16_t a, uint32_t b) {
  return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

template<>
inline __device__ uint2 mul(uint2 a, uint2 b) {
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  return c;
}

template<>
inline __device__ uint2 mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  uint2 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
  return c;
}

template<>
inline __device__ uint4 mul(uint4 a, uint4 b) {
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
  return c;
}

template<>
inline __device__ uint4 mul(uint16_t a, uint4 b) {
  uint32_t s = h0_h0(a);
  uint4 c;
  c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
  c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
  c.z = mul<uint32_t, uint32_t, uint32_t>(s, b.z);
  c.w = mul<uint32_t, uint32_t, uint32_t>(s, b.w);
  return c;
}

template<>
inline __device__ float mul(uint16_t a, uint16_t b) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb;
}

template<>
inline __device__ float2 mul(uint32_t a, uint32_t b) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return mul<float2, float2, float2>(fa, fb);
}

template<>
inline __device__ float2 mul(uint16_t a, uint32_t b) {
  return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);
}

template<>
inline __device__ Float4_ mul(uint2 a, uint2 b) {
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  return fc;
}

template<>
inline __device__ Float4_ mul(uint16_t a, uint2 b) {
  uint32_t s = h0_h0(a);
  Float4_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  return fc;
}

template<>
inline __device__ Float8_ mul(uint4 a, uint4 b) {
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);
  return fc;
}

template<>
inline __device__ Float8_ mul(uint16_t a, uint4 b) {
  uint32_t s = h0_h0(a);
  Float8_ fc;
  fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
  fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
  fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
  fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
  return fc;
}

// Vector fused multiply-add.
inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
#ifndef USE_ROCM
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
#else
  asm volatile("v_pk_fma_f16 %0, %1, %2, %3;\n" : "=v"(d) : "v"(a), "v"(b), "v"(c));
#endif
  return d;
}

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c) {
  return fma(h0_h0(a), b, c);
}

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c) {
  uint2 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  return d;
}

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c) {
  uint32_t s = h0_h0(a);
  uint2 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  return d;
}

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c) {
  uint4 d;
  d.x = fma(a.x, b.x, c.x);
  d.y = fma(a.y, b.y, c.y);
  d.z = fma(a.z, b.z, c.z);
  d.w = fma(a.w, b.w, c.w);
  return d;
}

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c) {
  uint32_t s = h0_h0(a);
  uint4 d;
  d.x = fma(s, b.x, c.x);
  d.y = fma(s, b.y, c.y);
  d.z = fma(s, b.z, c.z);
  d.w = fma(s, b.w, c.w);
  return d;
}

inline __device__ float fma(uint16_t a, uint16_t b, float fc) {
  float fa = half_to_float(a);
  float fb = half_to_float(b);
  return fa * fb + fc;
}

inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc) {
  float2 fa = half2_to_float2(a);
  float2 fb = half2_to_float2(b);
  return fma(fa, fb, fc);
}

inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc) {
  return fma(h0_h0(a), b, fc);
}

inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc) {
  Float4_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  return fd;
}

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc) {
  uint32_t s = h0_h0(a);
  Float4_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  return fd;
}

inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc) {
  Float8_ fd;
  fd.x = fma(a.x, b.x, fc.x);
  fd.y = fma(a.y, b.y, fc.y);
  fd.z = fma(a.z, b.z, fc.z);
  fd.w = fma(a.w, b.w, fc.w);
  return fd;
}

inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc) {
  uint32_t s = h0_h0(a);
  Float8_ fd;
  fd.x = fma(s, b.x, fc.x);
  fd.y = fma(s, b.y, fc.y);
  fd.z = fma(s, b.z, fc.z);
  fd.w = fma(s, b.w, fc.w);
  return fd;
}

// Vector sum.
template<>
inline __device__ float sum(uint16_t v) {
  return half_to_float(v);
}

template<>
inline __device__ float sum(uint32_t v) {
  float2 tmp = half2_to_float2(v);
  return tmp.x + tmp.y;
}

template<>
inline __device__ float sum(uint2 v) {
  uint32_t c = add(v.x, v.y);
  return sum(c);
}

template<>
inline __device__ float sum(uint4 v) {
  uint32_t c = add(v.x, v.y);
  c = add(c, v.z);
  c = add(c, v.w);
  return sum(c);
}

// From float32 to float16.
inline __device__ void from_float(uint16_t& dst, float src) {
  dst = float_to_half(src);
}

inline __device__ void from_float(uint32_t& dst, float2 src) {
  dst = float2_to_half2(src);
}

inline __device__ void from_float(uint2& dst, Float4_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
}

inline __device__ void from_float(uint4& dst, Float8_ src) {
  dst.x = float2_to_half2(src.x);
  dst.y = float2_to_half2(src.y);
  dst.z = float2_to_half2(src.z);
  dst.w = float2_to_half2(src.w);
}

// From float16 to float32.
inline __device__ float to_float(uint16_t u) {
  return half_to_float(u);
}

inline __device__ float2 to_float(uint32_t u) {
  return half2_to_float2(u);
}

inline __device__ Float4_ to_float(uint2 u) {
  Float4_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  return tmp;
}

inline __device__ Float8_ to_float(uint4 u) {
  Float8_ tmp;
  tmp.x = half2_to_float2(u.x);
  tmp.y = half2_to_float2(u.y);
  tmp.z = half2_to_float2(u.z);
  tmp.w = half2_to_float2(u.w);
  return tmp;
}

// Zero-out a variable.
inline __device__ void zero(uint16_t& dst) {
  dst = uint16_t(0);
}


} // namespace vllm

// attention_utils.cuh(qk_dot / Qk_dot)
namespace vllm {


// Q*K^T operation.
template<int THREAD_GROUP_SIZE, typename Vec, int N>
inline __device__ float qk_dot_(const Vec (&q)[N], const Vec (&k)[N]) {
  using A_vec = typename FloatVec<Vec>::Type;
  // Compute the parallel products for Q*K^T (treat vector lanes separately).
  A_vec qk_vec = mul<A_vec, Vec, Vec>(q[0], k[0]);
#pragma unroll
  for (int ii = 1; ii < N; ++ii) {
    qk_vec = fma(q[ii], k[ii], qk_vec);
  }

  // Finalize the reduction across lanes.
  float qk = sum(qk_vec);
#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += VLLM_SHFL_XOR_SYNC(qk, mask);
  }
  return qk;
}

template<typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template<typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};


} // namespace vllm


// vendor 逐字(prefill_paged_attn_opt.cu L63;sm86 ≥ sm75 走 tanh.approx)
inline __device__ float fast_tanh_opt(float x) {
#if defined(__CUDA_ARCH__)
#if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
  float y;
  asm volatile("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
#else
  return ::tanhf(x);
#endif
#else
  return ::tanhf(x);
#endif
}

// 适配⑧:helper 全集在 namespace vllm(本文件 §前奏);核体全局域经
// using 指令取用(extern "C" 入口必须留在全局域供 nvrtc 按名寻址)
#include <cuda_fp16.h>
#include <cuda_fp8.h>

using namespace vllm;

template<int HEAD_SIZE, int BLOCK_SIZE, int TOKEN_CHUNK_SIZE, bool KV_FP8 = false>
__global__ void chunked_prefill_paged_attention_opt_f16(
    uint16_t* __restrict__ out,              
    const uint16_t* __restrict__ q,          
    const uint16_t* __restrict__ k_cache,     
    const uint16_t* __restrict__ v_cache,     
    int32_t num_kv_heads,
    float sm_scale,
    const float* __restrict__ block_tables,
    const float* __restrict__ seq_lens,
    int32_t block_table_stride,
    int32_t num_seqs,
    int32_t num_query_heads,
    int32_t num_query_tokens,
    float softscapping,
    int32_t o_stride_tokens,
    const float* __restrict__ query_start_len,
    const float* __restrict__ alibi_slopes,
    const float* __restrict__ sinks,
    const int use_alibi_flag, const int use_sinks_flag,  // 适配⑤
    int32_t sliding_window,
    int32_t total_num_blocks,
    int32_t kv_block_stride,
    int32_t kv_head_stride
) {
    constexpr bool is_quantized = false;  // f16/f16 特化(适配③)
    
    // --- Shared Memory Layout ---
    // First: sequence info broadcast from lane 0 (16 bytes)
    // Then: K cache tile and V cache tile
    extern __shared__ char smem_buffer[];
    
    // Sequence info shared by all threads (computed by lane 0)
    struct SeqInfo {
        int seq_idx;
        int num_blocks;
        int start_block_idx;
        int start_token_idx;
        int seq_query_start;
        int seq_query_len;
        int q_pos_start;
        int boundary_seq_idx;    // second sequence if chunk spans boundary, -1 otherwise
        int boundary_token_pos;  // first token of second sequence within chunk
    };
    SeqInfo* shared_seq_info = reinterpret_cast<SeqInfo*>(smem_buffer);
    
    uint16_t* k_smem = reinterpret_cast<uint16_t*>(smem_buffer + 64);
    uint16_t* v_smem = k_smem + (HEAD_SIZE * BLOCK_SIZE);

    constexpr int THREAD_GROUP_SIZE = 1;
    constexpr int VEC_SIZE = 16 / sizeof(uint16_t);
    constexpr int NUM_VECS  = HEAD_SIZE / VEC_SIZE;
    constexpr int X = 16 / sizeof(uint16_t);

    const int tid = threadIdx.x;
    const int lane = tid % TOKEN_CHUNK_SIZE;
    const int block_dim = blockDim.x;

    const int NUM_BLOCK_VECS = BLOCK_SIZE / VEC_SIZE;
    const int qh_base_idx = blockIdx.x;
    const int kv_head_idx = blockIdx.y;
    const int chunk_start = blockIdx.z * TOKEN_CHUNK_SIZE;
    const int token_start = chunk_start + lane;

    const int num_queries_per_kv = num_query_heads / num_kv_heads;
    const bool use_alibi = (use_alibi_flag != 0);  // 适配⑤
    const bool use_sinks = (use_sinks_flag != 0);  // 适配⑤

    const long long q_stride_tokens = (long long)num_query_heads * (long long)HEAD_SIZE;
    const long long q_stride_heads = (long long)HEAD_SIZE;
    const long long o_stride_heads = (long long)HEAD_SIZE;

    // Lane 0 resolves the dominant sequence and detects chunk-boundary crossings
    if (lane == 0) {
        int seq_idx = 0;
        if (chunk_start < (int)query_start_len[num_seqs] && chunk_start >= (int)query_start_len[0]) {
            int left = 0, right = num_seqs - 1;
            while (left <= right) {
                int mid = (left + right) / 2;
                if ((int)query_start_len[mid + 1] <= chunk_start) {
                    left = mid + 1;
                } else if ((int)query_start_len[mid] > chunk_start) {
                    right = mid - 1;
                } else {
                    seq_idx = mid;
                    break;
                }
            }
        }
        
        int seq_len_full = (int)seq_lens[seq_idx];
        int seq_query_start = query_start_len[seq_idx];
        int seq_query_end = query_start_len[seq_idx + 1];
        int seq_query_len = seq_query_end - seq_query_start;
        int q_pos_start = (int)seq_len_full - seq_query_len;
        int first_local_q_pos = chunk_start - seq_query_start;
        if (first_local_q_pos < 0) first_local_q_pos = 0;
        int first_q_abs_pos = q_pos_start + first_local_q_pos;
        int num_blocks_seq = (int)((seq_len_full + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        int start_token_idx = 0;
        int start_block_idx = 0;
        if (sliding_window > 0 && sliding_window <= first_q_abs_pos) {
            start_token_idx = first_q_abs_pos + 1 - sliding_window;
            start_block_idx = start_token_idx / BLOCK_SIZE;
        }
        
        // Detect if this chunk spans a sequence boundary
        int chunk_end = chunk_start + TOKEN_CHUNK_SIZE;
        int boundary_seq = -1;
        int boundary_pos = TOKEN_CHUNK_SIZE;
        if (seq_idx + 1 < num_seqs && seq_query_end < chunk_end && seq_query_end > chunk_start) {
            boundary_seq = seq_idx + 1;
            boundary_pos = seq_query_end - chunk_start;
        }

        shared_seq_info->seq_idx = seq_idx;
        shared_seq_info->num_blocks = num_blocks_seq;
        shared_seq_info->start_block_idx = start_block_idx;
        shared_seq_info->start_token_idx = start_token_idx;
        shared_seq_info->seq_query_start = seq_query_start;
        shared_seq_info->seq_query_len = seq_query_len;
        shared_seq_info->q_pos_start = q_pos_start;
        shared_seq_info->boundary_seq_idx = boundary_seq;
        shared_seq_info->boundary_token_pos = boundary_pos;
    }
    __syncthreads();
    
    // All threads read shared info for the dominant (first) sequence
    const int dom_seq_idx = shared_seq_info->seq_idx;
    const int dom_num_blocks = shared_seq_info->num_blocks;
    const int dom_start_block_idx = shared_seq_info->start_block_idx;
    const int dom_start_token_idx = shared_seq_info->start_token_idx;
    const int dom_seq_query_start = shared_seq_info->seq_query_start;
    const int dom_q_pos_start = shared_seq_info->q_pos_start;
    const int boundary_seq_idx = shared_seq_info->boundary_seq_idx;
    const int boundary_token_pos = shared_seq_info->boundary_token_pos;

    // Per-thread sequence resolution: am I in the dominant or boundary sequence?
    const bool in_boundary_seq = (boundary_seq_idx >= 0 && lane >= boundary_token_pos);
    
    // Each thread resolves its own sequence context
    int my_seq_idx, my_num_blocks, my_start_block_idx, my_start_token_idx;
    int my_seq_query_start, my_q_pos_start;
    int my_seq_len_full;
    
    if (!in_boundary_seq) {
        my_seq_idx = dom_seq_idx;
        my_num_blocks = dom_num_blocks;
        my_start_block_idx = dom_start_block_idx;
        my_start_token_idx = dom_start_token_idx;
        my_seq_query_start = dom_seq_query_start;
        my_q_pos_start = dom_q_pos_start;
        my_seq_len_full = (int)seq_lens[dom_seq_idx];
    } else {
        my_seq_idx = boundary_seq_idx;
        my_seq_len_full = (int)seq_lens[boundary_seq_idx];
        my_seq_query_start = query_start_len[boundary_seq_idx];
        int my_seq_query_len = query_start_len[boundary_seq_idx + 1] - my_seq_query_start;
        my_q_pos_start = (int)my_seq_len_full - my_seq_query_len;
        my_num_blocks = (int)((my_seq_len_full + BLOCK_SIZE - 1) / BLOCK_SIZE);
        int local_q = token_start - my_seq_query_start;
        if (local_q < 0) local_q = 0;
        int abs_q = my_q_pos_start + local_q;
        my_start_token_idx = 0;
        my_start_block_idx = 0;
        if (sliding_window > 0 && sliding_window <= abs_q) {
            my_start_token_idx = abs_q + 1 - sliding_window;
            my_start_block_idx = my_start_token_idx / BLOCK_SIZE;
        }
    }

    const float* my_block_table = block_tables + (long long)my_seq_idx * (long long)block_table_stride;
    const int my_local_q_pos = token_start - my_seq_query_start;
    const int my_q_abs_pos = my_q_pos_start + my_local_q_pos;
    
    // Recompute sliding window for this specific thread
    if (sliding_window > 0 && sliding_window <= my_q_abs_pos && !in_boundary_seq) {
        my_start_token_idx = my_q_abs_pos + 1 - sliding_window;
    }

    using Q_vec = typename Vec<uint16_t, VEC_SIZE>::Type;
    using K_vec = typename Vec<uint16_t, VEC_SIZE>::Type;
    using Float_vec = typename Vec<float, VEC_SIZE>::Type;
    using Quant_vec = typename Vec<uint16_t, VEC_SIZE>::Type;

    Q_vec q_vec[NUM_VECS];
    float qk_block[BLOCK_SIZE];

    const int query_head_idx = kv_head_idx * num_queries_per_kv + qh_base_idx;
    const bool head_active = (qh_base_idx < num_queries_per_kv) && (query_head_idx < num_query_heads);
    const bool lane_active = (token_start < num_query_tokens);

    const long long q_off = (long long)token_start * q_stride_tokens + (long long)query_head_idx * q_stride_heads;
    const long long o_off = (long long)token_start * (long long)o_stride_tokens + (long long)query_head_idx * o_stride_heads;
    
    if (head_active && lane_active) {
        #pragma unroll
        for (int k = 0; k < NUM_VECS; k++) {
            int d_base = k * VEC_SIZE;
            q_vec[k] = *reinterpret_cast<const Q_vec*>(&q[q_off + d_base]);
        }
    }

    float acc_vec[HEAD_SIZE] = { 0.f };
    float M = (use_sinks && head_active && lane_active) ? sinks[query_head_idx] : -INFINITY;
    float alibi = (use_alibi && head_active && lane_active) ? alibi_slopes[query_head_idx] : 0.f;
    float L = 1.f;

    const int elems_per_block = HEAD_SIZE * BLOCK_SIZE;

    // --- Boundary-sequence threads: non-tiled path (read from global memory) ---
    // These threads cannot participate in cooperative KV loading for the dominant
    // sequence since they need different KV blocks. They run independently.
    if (in_boundary_seq && head_active && lane_active) {
        for (int blk = my_start_block_idx; blk < my_num_blocks; ++blk) {
            const int physical_block = (int)my_block_table[blk];
            const bool valid_block = (physical_block >= 0) &&
                                     (physical_block < total_num_blocks);  // 哨兵 -1(适配④)
            if (!valid_block) continue;

            const int block_in_full = blk * BLOCK_SIZE;
            const long long k_base = (long long)physical_block * kv_block_stride + (long long)kv_head_idx * kv_head_stride;
            const long long v_base = k_base;
            bool in_contexts[BLOCK_SIZE];

            for (int b = 0; b < BLOCK_SIZE; ++b) {
                const int token_idx_in_full = block_in_full + b;
                bool in_context = (token_idx_in_full <= my_q_abs_pos);
                bool in_window = (token_idx_in_full >= my_start_token_idx);
                in_contexts[b] = in_context && in_window;

                if (!in_context || !in_window) {
                    qk_block[b] = -INFINITY;
                } else {
                    K_vec k_vec_local[NUM_VECS];
                    #pragma unroll
                    for (int k = 0; k < NUM_VECS; k++) {
                        int d = k * VEC_SIZE;
                        int gy = d / X;
                        int gx = d % X;
                        long long k_idx = k_base + b * X + gy * (BLOCK_SIZE * X) + gx;
                        if constexpr (KV_FP8) {
                            // B6.3:fp8 e4m3 读入转 half(元素序同布局)
                            const unsigned char* k8 = reinterpret_cast<const unsigned char*>(&k_cache[k_idx]);
                            __half kt[VEC_SIZE];
                            #pragma unroll
                            for (int t = 0; t < VEC_SIZE; t++) {
                                __nv_fp8_e4m3 e; e.__x = k8[t];
                                kt[t] = __half(__nv_cvt_fp8_to_halfraw(e.__x, __NV_E4M3));
                            }
                            k_vec_local[k] = *reinterpret_cast<const K_vec*>(kt);
                        } else {
                            k_vec_local[k] = *reinterpret_cast<const K_vec*>(&k_cache[k_idx]);
                        }
                    }
                    float qk = Qk_dot<uint16_t, THREAD_GROUP_SIZE>::dot(q_vec, k_vec_local) * sm_scale;
                    if (softscapping != 1.0) qk = fast_tanh_opt(qk / softscapping) * softscapping;
                    if (use_alibi) qk += alibi * float(token_idx_in_full - my_q_abs_pos);
                    qk_block[b] = qk;
                }
            }

            float Smax = -INFINITY;
            #pragma unroll
            for (int b = 0; b < BLOCK_SIZE; ++b) Smax = fmaxf(Smax, qk_block[b]);
            const float m_j = fmaxf(M, Smax);
            const float alpha_v = __expf(M - m_j);
            M = m_j;
            L = L * alpha_v;
            #pragma unroll
            for (int i = 0; i < HEAD_SIZE; ++i) acc_vec[i] *= alpha_v;

            Float_vec p_vec[NUM_BLOCK_VECS];
            float acc_lane = 0.f;
            #pragma unroll
            for (int b = 0; b < BLOCK_SIZE; ++b) {
                if (in_contexts[b]) {
                    const float P = __expf(qk_block[b] - M);
                    reinterpret_cast<float*>(&p_vec[b/VEC_SIZE])[b % VEC_SIZE] = P;
                    acc_lane += P;
                } else {
                    reinterpret_cast<float*>(&p_vec[b/VEC_SIZE])[b % VEC_SIZE] = 0.f;
                }
            }
            L += acc_lane;

            for (int k = 0; k < HEAD_SIZE; ++k) {
                const unsigned char* v_row8 = reinterpret_cast<const unsigned char*>(&v_cache[v_base + (long long)k * BLOCK_SIZE]);
                const uint16_t* v_row = &v_cache[v_base + (long long)k * BLOCK_SIZE];
                for (int bv = 0; bv < NUM_BLOCK_VECS; bv++) {
                    Float_vec v_val;
                    if constexpr (KV_FP8) {
                        // B6.3:fp8 V 行读入转 half
                        __half vt[VEC_SIZE];
                        const unsigned char* vp = v_row8 + bv * VEC_SIZE;
                        #pragma unroll
                        for (int t = 0; t < VEC_SIZE; t++) {
                            __nv_fp8_e4m3 e; e.__x = vp[t];
                            vt[t] = __half(__nv_cvt_fp8_to_halfraw(e.__x, __NV_E4M3));
                        }
                        v_val = to_float(*reinterpret_cast<const K_vec*>(vt));
                    } else {
                        v_val = to_float(*reinterpret_cast<const K_vec*>(v_row + bv * VEC_SIZE));
                    }
                    acc_vec[k] += dot(p_vec[bv], v_val);
                }
            }
        }

        // Write boundary-thread output
        using O_vec = typename Vec<uint16_t, VEC_SIZE>::Type;
        O_vec o_vec[NUM_VECS];
        #pragma unroll
        for (int k = 0; k < HEAD_SIZE; k++) {
            float outv = acc_vec[k] / (L + 1e-6f);
            from_float(reinterpret_cast<uint16_t*>(&o_vec[k / VEC_SIZE])[k % VEC_SIZE], outv);
        }
        #pragma unroll
        for (int k = 0; k < NUM_VECS; k++) {
            *reinterpret_cast<O_vec*>(out + o_off + k * VEC_SIZE) = o_vec[k];
        }
    }

    // Boundary threads are done; they still participate in __syncthreads below
    // but skip computation for dominant-sequence KV blocks.
    
    // --- Dominant-sequence threads: cooperative shared-memory tiled path ---
    const float* dom_block_table = block_tables + (long long)dom_seq_idx * (long long)block_table_stride;

    for (int blk = dom_start_block_idx; blk < dom_num_blocks; ++blk) {
        const int physical_block = (int)dom_block_table[blk];
        const bool valid_block = (physical_block >= 0) &&
                                 (physical_block < total_num_blocks);  // 哨兵 -1(适配④)

        // ALL threads cooperatively load KV into shared memory
        if (valid_block) {
            const long long k_base = (long long)physical_block * kv_block_stride + (long long)kv_head_idx * kv_head_stride;
            if constexpr (KV_FP8) {
                // B6.3:fp8 → f16 转换拷贝(smem 宽度不变,读侧算术复用)
                const unsigned char* k_src8 = reinterpret_cast<const unsigned char*>(k_cache) + k_base;
                const unsigned char* v_src8 = reinterpret_cast<const unsigned char*>(v_cache) + k_base;
                for (int i = tid; i < elems_per_block; i += block_dim) {
                    __nv_fp8_e4m3 ke; ke.__x = k_src8[i];
                    __nv_fp8_e4m3 ve; ve.__x = v_src8[i];
                    k_smem[i] = __half(__nv_cvt_fp8_to_halfraw(ke.__x, __NV_E4M3));
                    v_smem[i] = __half(__nv_cvt_fp8_to_halfraw(ve.__x, __NV_E4M3));
                }
            } else {
            const uint16_t* k_src = k_cache + k_base;
            for (int i = tid; i < elems_per_block; i += block_dim) {
                k_smem[i] = k_src[i];
            }
            const uint16_t* v_src = v_cache + k_base;
            for (int i = tid; i < elems_per_block; i += block_dim) {
                v_smem[i] = v_src[i];
            }
            }
        }
        
        __syncthreads();

        // Only dominant-sequence threads compute attention from tiled data
        if (valid_block && !in_boundary_seq) {
            const int block_in_full = blk * BLOCK_SIZE;
            bool in_contexts[BLOCK_SIZE];

            for (int b = 0; b < BLOCK_SIZE; ++b) {
                const int token_idx_in_full = block_in_full + b;
                bool in_context = (token_idx_in_full <= my_q_abs_pos);
                bool in_window = (token_idx_in_full >= my_start_token_idx);
                in_contexts[b] = in_context && in_window;

                if (!in_context || !in_window || !lane_active) {
                    qk_block[b] = -INFINITY;
                } else {
                    K_vec k_vec_local[NUM_VECS];
                    #pragma unroll
                    for (int k = 0; k < NUM_VECS; k++) {
                        int d = k * VEC_SIZE;
                        int gy = d / X;
                        int gx = d % X;
                        int smem_idx = b * X + gy * (BLOCK_SIZE * X) + gx;
                        k_vec_local[k] = *reinterpret_cast<const K_vec*>(&k_smem[smem_idx]);
                    }
                    float qk = Qk_dot<uint16_t, THREAD_GROUP_SIZE>::dot(q_vec, k_vec_local) * sm_scale;
                    if (softscapping != 1.0) qk = fast_tanh_opt(qk / softscapping) * softscapping;
                    if (use_alibi) qk += alibi * float(token_idx_in_full - my_q_abs_pos);
                    qk_block[b] = qk;
                }
            }

            if (head_active && lane_active) {
                float Smax = -INFINITY;
                #pragma unroll
                for (int b = 0; b < BLOCK_SIZE; ++b) Smax = fmaxf(Smax, qk_block[b]);

                const float m_j = fmaxf(M, Smax);
                const float alpha_v = __expf(M - m_j);
                M = m_j;
                L = L * alpha_v;
                
                #pragma unroll
                for (int i = 0; i < HEAD_SIZE; ++i) acc_vec[i] *= alpha_v;

                Float_vec p_vec[NUM_BLOCK_VECS];
                float acc_lane = 0.f;
                #pragma unroll
                for (int b = 0; b < BLOCK_SIZE; ++b) {
                    if (in_contexts[b]) {
                        const float P = __expf(qk_block[b] - M);
                        reinterpret_cast<float*>(&p_vec[b/VEC_SIZE])[b % VEC_SIZE] = P;
                        acc_lane += P;
                    } else {
                        reinterpret_cast<float*>(&p_vec[b/VEC_SIZE])[b % VEC_SIZE] = 0.f;
                    }
                }
                L += acc_lane;

                for (int k = 0; k < HEAD_SIZE; ++k) {
                    const uint16_t* v_row_ptr = &v_smem[(long long)k * BLOCK_SIZE];
                    for (int b_vec = 0; b_vec < NUM_BLOCK_VECS; b_vec++) {
                        const uint16_t* src = v_row_ptr + b_vec * VEC_SIZE;
                        Float_vec v_val_vec;
                        v_val_vec = to_float(*reinterpret_cast<const K_vec*>(src));
                        acc_vec[k] += dot(p_vec[b_vec], v_val_vec);
                    }
                }
            }
        }
        
        __syncthreads();
    }

    // Write dominant-thread output
    if (!in_boundary_seq && head_active && lane_active) {
        using O_vec = typename Vec<uint16_t, VEC_SIZE>::Type;
        O_vec o_vec[NUM_VECS];
        #pragma unroll
        for (int k = 0; k < HEAD_SIZE; k++) {
            float outv = acc_vec[k] / (L + 1e-6f);
            from_float(reinterpret_cast<uint16_t*>(&o_vec[k / VEC_SIZE])[k % VEC_SIZE], outv);
        }
        #pragma unroll
        for (int k = 0; k < NUM_VECS; k++) {
            *reinterpret_cast<O_vec*>(out + o_off + k * VEC_SIZE) = o_vec[k];
        }
    }
}



extern "C" __global__ void vllm_chunked_prefill_paged_attn_opt_f16_hd128(
    const uint16_t* __restrict__ q,              // [num_query_tokens, Hq, hd]
    const uint16_t* __restrict__ k_cache,        // [num_blocks, Hkv, hd/8, 32, 8]
    const uint16_t* __restrict__ v_cache,        // [num_blocks, Hkv, hd, 32]
    const float* __restrict__ block_tables,      // [num_seqs, block_table_stride]
    const float* __restrict__ seq_lens,          // [num_seqs]
    const float* __restrict__ query_start_len,   // [num_seqs+1]
    const float* __restrict__ alibi_slopes,      // use_alibi_flag=0 不解引用
    const float* __restrict__ sinks,             // use_sinks_flag=0 不解引用
    const int num_kv_heads,
    const float sm_scale,
    const int block_table_stride,
    const int num_seqs,
    const int num_query_heads,
    const int num_query_tokens,
    const float softscapping,
    const int o_stride_tokens,
    const int sliding_window,
    const int total_num_blocks,
    const int kv_block_stride,
    const int kv_head_stride,
    const int use_alibi_flag,
    const int use_sinks_flag,
    uint16_t* __restrict__ out) {                // OUT(末参,契约 4)
  chunked_prefill_paged_attention_opt_f16<128, 32, 256>(
      out, q, k_cache, v_cache, num_kv_heads, sm_scale, block_tables, seq_lens,
      block_table_stride, num_seqs, num_query_heads, num_query_tokens,
      softscapping, o_stride_tokens, query_start_len, alibi_slopes, sinks,
      use_alibi_flag, use_sinks_flag, sliding_window, total_num_blocks,
      kv_block_stride, kv_head_stride);
}

extern "C" __global__ void vllm_chunked_prefill_paged_attn_opt_f16_hd256(
    const uint16_t* __restrict__ q,              // [num_query_tokens, Hq, hd]
    const uint16_t* __restrict__ k_cache,        // [num_blocks, Hkv, hd/8, 32, 8]
    const uint16_t* __restrict__ v_cache,        // [num_blocks, Hkv, hd, 32]
    const float* __restrict__ block_tables,      // [num_seqs, block_table_stride]
    const float* __restrict__ seq_lens,          // [num_seqs]
    const float* __restrict__ query_start_len,   // [num_seqs+1]
    const float* __restrict__ alibi_slopes,      // use_alibi_flag=0 不解引用
    const float* __restrict__ sinks,             // use_sinks_flag=0 不解引用
    const int num_kv_heads,
    const float sm_scale,
    const int block_table_stride,
    const int num_seqs,
    const int num_query_heads,
    const int num_query_tokens,
    const float softscapping,
    const int o_stride_tokens,
    const int sliding_window,
    const int total_num_blocks,
    const int kv_block_stride,
    const int kv_head_stride,
    const int use_alibi_flag,
    const int use_sinks_flag,
    uint16_t* __restrict__ out) {                // OUT(末参,契约 4)
  chunked_prefill_paged_attention_opt_f16<256, 32, 256>(
      out, q, k_cache, v_cache, num_kv_heads, sm_scale, block_tables, seq_lens,
      block_table_stride, num_seqs, num_query_heads, num_query_tokens,
      softscapping, o_stride_tokens, query_start_len, alibi_slopes, sinks,
      use_alibi_flag, use_sinks_flag, sliding_window, total_num_blocks,
      kv_block_stride, kv_head_stride);
}

// ---- B6.3:fp8 e4m3 KV 读变体(chunked prefill;hd128/256)----
extern "C" __global__ void vllm_chunked_prefill_paged_attn_opt_fp8_hd128(
    uint16_t* __restrict__ out,
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    int32_t num_kv_heads,
    float sm_scale,
    const float* __restrict__ block_tables,
    const float* __restrict__ seq_lens,
    int32_t block_table_stride,
    int32_t num_seqs,
    int32_t num_query_heads,
    int32_t num_query_tokens,
    float softscapping,
    int32_t o_stride_tokens,
    const float* __restrict__ query_start_len,
    const float* __restrict__ alibi_slopes,
    const float* __restrict__ sinks,
    const int use_alibi_flag, const int use_sinks_flag,
    int32_t sliding_window,
    int32_t total_num_blocks,
    int32_t kv_block_stride,
    int32_t kv_head_stride) {
  chunked_prefill_paged_attention_opt_f16<128, 32, 256, true>(
      out, q, k_cache, v_cache, num_kv_heads, sm_scale, block_tables, seq_lens,
      block_table_stride, num_seqs, num_query_heads, num_query_tokens,
      softscapping, o_stride_tokens, query_start_len, alibi_slopes, sinks,
      use_alibi_flag, use_sinks_flag, sliding_window, total_num_blocks,
      kv_block_stride, kv_head_stride);
}

extern "C" __global__ void vllm_chunked_prefill_paged_attn_opt_fp8_hd256(
    uint16_t* __restrict__ out,
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    int32_t num_kv_heads,
    float sm_scale,
    const float* __restrict__ block_tables,
    const float* __restrict__ seq_lens,
    int32_t block_table_stride,
    int32_t num_seqs,
    int32_t num_query_heads,
    int32_t num_query_tokens,
    float softscapping,
    int32_t o_stride_tokens,
    const float* __restrict__ query_start_len,
    const float* __restrict__ alibi_slopes,
    const float* __restrict__ sinks,
    const int use_alibi_flag, const int use_sinks_flag,
    int32_t sliding_window,
    int32_t total_num_blocks,
    int32_t kv_block_stride,
    int32_t kv_head_stride) {
  chunked_prefill_paged_attention_opt_f16<256, 32, 256, true>(
      out, q, k_cache, v_cache, num_kv_heads, sm_scale, block_tables, seq_lens,
      block_table_stride, num_seqs, num_query_heads, num_query_tokens,
      softscapping, o_stride_tokens, query_start_len, alibi_slopes, sinks,
      use_alibi_flag, use_sinks_flag, sliding_window, total_num_blocks,
      kv_block_stride, kv_head_stride);
}
