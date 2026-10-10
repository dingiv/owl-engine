// ============================================================================
// paged attention decode 家族(K1/K2;2026-09-27 port)
//
// 出处(vendor file-level 复用,Apache-2.0):
//   packages/xinfer/vendor/attention.rs/src/kernels/src/pagedattention.cuh
//   + paged_attention_v1.cu + paged_attention_v2.cu(rev c0f19f2,冻结只读;
//   上游 = vLLM csrc paged_attention_v1/v2 + FasterTransformer DT_MHA 模板;
//   A4 所有权纪律:头注 + 语义保持,dtype 收窄见逐条认领)
//
// port 适配(显式认领):
//   ① stdint.h 不可开(nvrtc 默认头集极简)→ 文件头局部 typedef 同宽替;
//   ② C++ 模板 __global__ 按 nvrtc 取名不可寻址 → extern "C" 入口按
//      (HEAD_SIZE × 内核族)显式实例化;__device__ 模板保留(nvrtc 支持解析);
//   ③ scalar_t=cache_t=uint16_t 单特化(f16 位型;bf16/fp8 臂与
//      is_quantized 分支剥除,scales 形参随删);is_quantized 常量化 false;
//   ④ int 表(block_tables/context_lens/seq_lens)→ const float* 过线
//      (owl 契约 5:索引/位置量 f32 数值,核内 cast;|值|<2^24 精确);
//   ⑤ alibi_slopes nullptr 判定 → use_alibi 旗标整参(owl 发射器无 null 块
//      指针;旗标 0 时指针不解引用,可挂任意块);
//   ⑥ 尾参 = 输出块(owl 槽序契约 4):v1/reduce 的 out、v2 的 tmp_out
//      均排在形参末位;
//   ⑦ device assert 删(NUM_THREADS % THREAD_GROUP_SIZE == 0 由实例化
//      组合保证,不进运行时)。
//
// 布局契约(vLLM classic,K0 reshape_and_cache 同款):
//   key_cache   [num_blocks, Hkv, D/x, block_size, x](f16 x=8)
//   value_cache [num_blocks, Hkv, D, block_size]
//   block_tables [num_seqs, max_num_blocks_per_seq](f32 过线,物理块号)
//   out/q        [num_seqs, num_heads, head_size]
//
// B6:fp8 e4m3 KV 臂需要 f16/fp8 设备类型(原 port ③ 为纯 uint16_t
// 位型零头文件;fp8 臂恢复标准头,nvrtc include 路径由发射方给)
#include <cuda_fp16.h>
#include <cuda_fp8.h>

// grid / shared_mem 契约(发射方计算;禁哨兵自动 grid,见 K0 头注):
//   v1     grid (num_heads, num_seqs, 1)
//   v2     grid (num_heads, num_seqs, max_num_partitions),
//          max_num_partitions = ceil(max_ctx / PARTITION_SIZE=512)
//   reduce grid (num_heads, num_seqs, 1)
//   block  (128,1,1)(NUM_THREADS=128)
//   shared v1/v2 = max(ceil(max_ctx,BLOCK)*BLOCK*4B, (NUM_WARPS/2)*HD*4B)
//          —— >48KB 需 cudaFuncSetAttribute(owl 发射器暂无此通道,
//          上限 ≈ ctx 12k @ HD256;超限场景列 launcher 扩展挂账)
//   reduce = 4B * max_num_partitions × 2
//
// 未验证风险(nvrtc 静态不可证,K1 真机对拍时收口):
//   - PTX 内联 asm(mov.b32/cvt.f32.f16/shfl)在 nvrtc compute_86 的展开;
//   - extern shared + 模板 __device__ 大函数的寄存器压力与编译时长;
//   - softscapping/fast_tanh 路径本仓库未消费(vLLM 默认 1.0 直通)。
// ============================================================================

// ---- nvrtc 序言:mini-stdint + 数学常量 + 宏(stdint.h/float.h 不可开)----
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned short uint16_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
#ifndef FLT_MAX
#define FLT_MAX 340282346638528859811704183484516925440.0f
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))
// cuda_compat.h(CUDA 臂)
#define VLLM_LDG(arg) __ldg(arg)
#define VLLM_SHFL_XOR_SYNC(var, lane_mask) __shfl_xor_sync(uint32_t(-1), var, lane_mask)
#define VLLM_SHFL_XOR_SYNC_WIDTH(var, lane_mask, width) __shfl_xor_sync(uint32_t(-1), var, lane_mask, width)
#define VLLM_SHFL_SYNC(var, src_lane) __shfl_sync(uint32_t(-1), var, src_lane)

// ---- 展平支持层(逐字同源;includes 由本文件序言替代)----

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

// ---- block_sum / fast_tanh(vendor 逐字)----
// Utility function for attention softmax.
template<int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < NUM_WARPS) {
    sum = red_smem[lane];
  }

  // Parallel reduction inside the warp.
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    sum += VLLM_SHFL_XOR_SYNC(sum, mask);
  }

  // Broadcast to other threads.
  return VLLM_SHFL_SYNC(sum, 0);
}

inline __device__ float fast_tanh(float x) {
  #if defined(__CUDA_ARCH__)
    #if (__CUDACC_VER_MAJOR__ >= 11) && (__CUDA_ARCH__ >= 750)
      float y;
      asm volatile ( "tanh.approx.f32 %0, %1; " : "=f"(y) : "f"(x));
      return y;
    #else
      return ::tanhf(x);
    #endif
  #else
  return std::tanh(x);
  #endif
}

// Grid: (num_heads, num_seqs, max_num_partitions).

// ---- kernel 体(f16 特化;vendor 逐字,见头注认领)----
namespace vllm {
template <int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
          int PARTITION_SIZE = 0,  // Zero means no partitioning.
          bool KV_FP8 = false,     // B6:KV 存储为 e4m3(1B/elem,同逻辑布局;
                                   // 读入即转 half,dot/累加全复用 f16 路径)
          bool KNHD = false>       // kv布局统一契约 P2:页内 [page,hkv,hd] 连续
                                   // (host 传 kv_block_stride=page·hkv·hd、
                                   // kv_head_stride=hd;逻辑 d→线程映射不变,
                                   // 输出与 classic 位等)
__device__ void paged_attention_kernel_f16(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    uint16_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    const uint16_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const uint16_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const uint16_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const float* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const float* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads](use_alibi=0 不解引用,适配⑤)
    const int use_alibi,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, 
    const int sliding_window) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int seq_len = (int)seq_lens[seq_idx];
  constexpr bool is_quantized = false;  // f16/f16 特化(适配③)

  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= seq_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);
  const int num_blocks_per_partition =
      USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_seq_blocks;

  // Sliding Window Logic: Calculate global start boundaries
  // If sliding_window is valid, we only attend to tokens in [seq_len - sliding_window, seq_len)
  const int global_start_token_idx = (sliding_window > 0 && sliding_window < seq_len) 
                                     ? (seq_len - sliding_window) : 0;
  const int global_start_block_idx = global_start_token_idx / BLOCK_SIZE;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  int start_block_idx =
      USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  
  // Adjust start_block_idx based on sliding window
  // We advance the start block to skip blocks strictly outside the window
  start_block_idx = MAX(start_block_idx, global_start_block_idx);

  const int end_block_idx =
      MIN(start_block_idx + num_blocks_per_partition, num_seq_blocks);
  
  // If the sliding window pushes start beyond end (e.g. this partition is entirely 
  // outside the window), terminate early.
  if (start_block_idx >= end_block_idx && USE_PARTITIONING) {
      return;
  }

  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx =
      MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE
                                        // divides NUM_THREADS
  constexpr int NUM_TOKENS_PER_THREAD_GROUP =
      DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = (use_alibi != 0) ? alibi_slopes[head_idx] : 0.f;  // 适配⑤

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(uint16_t)), 1);
  using K_vec = typename Vec<uint16_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<uint16_t, VEC_SIZE>::Type;
  using Quant_vec = typename Vec<uint16_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  // NOTE(woosuk): Because q is split from a qkv tensor, it may not be contiguous.
  const uint16_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads(); // TODO(naed90): possible speedup if this is replaced with a memory wall right before we use q_vecs

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(uint16_t);
  float qk_max = -FLT_MAX;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const float* block_table = block_tables + seq_idx * max_num_blocks_per_seq;


  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    const long long physical_block_number =
        static_cast<long long>((int)block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      K_vec k_vecs[NUM_VECS_PER_THREAD];

      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset1 = (vec_idx * VEC_SIZE) / x;
        const int offset2 = (vec_idx * VEC_SIZE) % x;
        if constexpr (KNHD) {
          // 统一契约 P2:页内 [page,hkv,hd] 连续 → vec 连续直读;
          // kv_block_stride/BLOCK_SIZE = hkv·hd(host 契约)
          const long long kn = physical_block_number * kv_block_stride
                             + kv_head_idx * kv_head_stride
                             + (long long)physical_block_offset * (kv_block_stride / BLOCK_SIZE);
          if constexpr (KV_FP8) {
            const unsigned char* k_ptr8 =
                reinterpret_cast<const unsigned char*>(k_cache) + kn + vec_idx * VEC_SIZE;
            __half kt[VEC_SIZE];
#pragma unroll
            for (int t = 0; t < VEC_SIZE; t++) {
              __nv_fp8_e4m3 e;
              e.__x = k_ptr8[t];
              kt[t] = __half(__nv_cvt_fp8_to_halfraw(e.__x, __NV_E4M3));
            }
            k_vecs[j] = *reinterpret_cast<const K_vec*>(kt);
          } else {
            k_vecs[j] = *reinterpret_cast<const K_vec*>(
                k_cache + kn + vec_idx * VEC_SIZE);
          }
        } else if constexpr (KV_FP8) {
          // B6:fp8 e4m3 读入(1B/elem,同逻辑布局;标量 u8 读,
          // warp 内按 offset 相邻合曲);stride 均为元素序(f16 侧传
          // 元素 stride,fp8 侧 host 传减半后的元素 stride)
          const unsigned char* k_ptr8 =
              reinterpret_cast<const unsigned char*>(k_cache) +
              physical_block_number * kv_block_stride +
              kv_head_idx * kv_head_stride + physical_block_offset * x;
          __half kt[VEC_SIZE];
#pragma unroll
          for (int t = 0; t < VEC_SIZE; t++) {
            __nv_fp8_e4m3 e;
            e.__x = k_ptr8[offset1 * BLOCK_SIZE * x + offset2 + t];
            kt[t] = __half(__nv_cvt_fp8_to_halfraw(e.__x, __NV_E4M3));
          }
          k_vecs[j] = *reinterpret_cast<const K_vec*>(kt);
        } else {
          const uint16_t* k_ptr =
              k_cache + physical_block_number * kv_block_stride +
              kv_head_idx * kv_head_stride + physical_block_offset * x;
          k_vecs[j] = *reinterpret_cast<const K_vec*>(
              k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<uint16_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);

      if (softscapping != 1.0) {
        qk = fast_tanh(qk / softscapping) * softscapping;
      }
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        
        // Sliding Window Masking
        // Mask if token is causal future (token_idx >= seq_len) OR 
        // token is outside sliding window history (token_idx < global_start_token_idx)
        const bool mask = token_idx >= seq_len || token_idx < global_start_token_idx;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdiv_rn(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions
                                       + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                   + head_idx * max_num_partitions
                                   + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(uint16_t), BLOCK_SIZE);
  using V_vec = typename Vec<uint16_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<uint16_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  uint16_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
       block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to
    // int64 because int32 can lead to overflow when this variable is multiplied
    // by large numbers (e.g., kv_block_stride).
    // For blocksparse attention: skip computation on blocks that are not
    // attended
    const long long physical_block_number =
        static_cast<long long>((int)block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;

    Float_L_vec  logits_vec = *reinterpret_cast<Float_L_vec*>(logits + token_idx -
                                                           start_token_idx);                                                 
    const uint16_t* v_ptr = v_cache + physical_block_number * kv_block_stride +
                           kv_head_idx * kv_head_stride;
    const unsigned char* v_ptr8 = reinterpret_cast<const unsigned char*>(v_cache) +
                            physical_block_number * kv_block_stride +
                            kv_head_idx * kv_head_stride;
    // kNHD:元素 (pbn, pbo+j, h, row_idx) 基址(页内 token 维 stride = hkv·hd)
    const long long v_kn_base = physical_block_number * kv_block_stride
                              + kv_head_idx * kv_head_stride
                              + (long long)physical_block_offset * (kv_block_stride / BLOCK_SIZE);
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec;
        if constexpr (KNHD) {
          // 统一契约 P2:token 维跨 hkv·hd 聚集(gather);值序不变 → 位等
          __half vt[V_VEC_SIZE];
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            if constexpr (KV_FP8) {
              __nv_fp8_e4m3 e;
              e.__x = reinterpret_cast<const unsigned char*>(v_cache)[v_kn_base
                  + j * (kv_block_stride / BLOCK_SIZE) + row_idx];
              vt[j] = __half(__nv_cvt_fp8_to_halfraw(e.__x, __NV_E4M3));
            } else {
              vt[j] = __ushort_as_half(
                  reinterpret_cast<const uint16_t*>(v_cache)[v_kn_base
                      + j * (kv_block_stride / BLOCK_SIZE) + row_idx]);
            }
          }
          v_vec = *reinterpret_cast<const V_vec*>(vt);
        } else if constexpr (KV_FP8) {
          // B6:fp8 e4m3 读入(标量 u8;V 布局 [hd, block] 行连续,
          // warp 内 lane 相邻 = token 相邻 = 字节相邻,合曲)
          __half vt[V_VEC_SIZE];
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            __nv_fp8_e4m3 e;
            e.__x = v_ptr8[offset + j];
            vt[j] = __half(__nv_cvt_fp8_to_halfraw(e.__x, __NV_E4M3));
          }
          v_vec = *reinterpret_cast<const V_vec*>(vt);
        } else {
          v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        }
        if (block_idx == num_seq_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the
          // context, we should explicitly zero out the values since they may
          // contain NaNs. See
          // https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          uint16_t* v_vec_ptr = reinterpret_cast<uint16_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
            if (token_idx + j >= seq_len)
              v_vec_ptr[j] = zero_value;
          }
        }
        Float_L_vec v = to_float(v_vec);
        accs[i] += dot(logits_vec, v);
      }
    }
  }

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    uint16_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                            + head_idx * max_num_partitions * HEAD_SIZE
                            + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}


// Grid: (num_heads, num_seqs, 1).
template<
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS>
__global__ void paged_attention_v1_kernel_f16(
  uint16_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const uint16_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const uint16_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  const uint16_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
  const int num_kv_heads,                 // [num_heads]
  const float scale,
  const float* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const float* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes, // [num_heads](适配⑤)
  const int use_alibi,
  const int q_stride,
  const int kv_block_stride,
  const int kv_head_stride,
  const float softscapping,
  const int sliding_window) {

  paged_attention_kernel_f16< HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>(
    /* exp_sums */ nullptr, /* max_logits */ nullptr,
    out, q, k_cache, v_cache, num_kv_heads, scale, block_tables, context_lens,
    max_num_blocks_per_seq, alibi_slopes, use_alibi, q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}


// Grid: (num_heads, num_seqs, max_num_partitions).
template<
  int HEAD_SIZE,
  int BLOCK_SIZE,
  int NUM_THREADS,
  int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel_f16(
  float* __restrict__ exp_sums,           // [num_seqs, num_heads, max_num_partitions]
  float* __restrict__ max_logits,         // [num_seqs, num_heads, max_num_partitions]
  uint16_t* __restrict__ tmp_out,         // [num_seqs, num_heads, max_num_partitions, head_size]
  const uint16_t* __restrict__ q,         // [num_seqs, num_heads, head_size]
  const uint16_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
  const uint16_t* __restrict__ v_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
  const int num_kv_heads,                 // [num_heads]
  const float scale,
  const float* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
  const float* __restrict__ context_lens,   // [num_seqs]
  const int max_num_blocks_per_seq,
  const float* __restrict__ alibi_slopes, // [num_heads](适配⑤)
  const int use_alibi,
  const int q_stride,
  const int kv_block_stride,
  const int kv_head_stride,
  const float softscapping,
  const int sliding_window) {
  paged_attention_kernel_f16< HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>(
    exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
    block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
    q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

// Grid: (num_heads, num_seqs).
template<
  int HEAD_SIZE,
  int NUM_THREADS,
  int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel_f16(
  uint16_t* __restrict__ out,             // [num_seqs, num_heads, head_size]
  const float* __restrict__ exp_sums,     // [num_seqs, num_heads, max_num_partitions]
  const float* __restrict__ max_logits,   // [num_seqs, num_heads, max_num_partitions]
  const uint16_t* __restrict__ tmp_out,   // [num_seqs, num_heads, max_num_partitions, head_size]
  const float* __restrict__ context_lens,   // [num_seqs]
  const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int context_len = (int)context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    uint16_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const uint16_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                          + head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions
                                           + head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = VLLM_SHFL_SYNC(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions
                                       + head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdiv_rn(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const uint16_t* tmp_out_ptr = tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE
                                        + head_idx * max_num_partitions * HEAD_SIZE;
  uint16_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}


} // namespace vllm
extern "C" __global__ void vllm_paged_attention_v1_f16_hd128(
    const uint16_t* __restrict__ q,              // [num_seqs, num_heads, hd]
    const uint16_t* __restrict__ k_cache,        // [num_blocks, Hkv, hd/8, 16, 8]
    const uint16_t* __restrict__ v_cache,        // [num_blocks, Hkv, hd, 16]
    const float* __restrict__ block_tables,      // [num_seqs, max_num_blocks_per_seq]
    const float* __restrict__ context_lens,      // [num_seqs]
    const float* __restrict__ alibi_slopes,      // [num_heads](use_alibi=0 不解引用)
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ out) {
  vllm::paged_attention_v1_kernel_f16<128, 16, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale, block_tables, context_lens,
      max_num_blocks_per_seq, alibi_slopes, use_alibi, q_stride, kv_block_stride,
      kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_f16_hd128(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    const float* __restrict__ context_lens,
    const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums,                // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,              // 同上
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ tmp_out) {           // OUT(末参,契约 4)
  vllm::paged_attention_v2_kernel_f16<128, 16, 128, 512>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_reduce_f16_hd128(
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const uint16_t* __restrict__ tmp_out,
    const float* __restrict__ context_lens,      // [num_seqs]
    const int max_num_partitions,
    uint16_t* __restrict__ out) {               // OUT(末参,契约 4)
  vllm::paged_attention_v2_reduce_kernel_f16<128, 128, 512>(
      out, exp_sums, max_logits, tmp_out, context_lens, max_num_partitions);
}

extern "C" __global__ void vllm_paged_attention_v1_f16_hd256(
    const uint16_t* __restrict__ q,              // [num_seqs, num_heads, hd]
    const uint16_t* __restrict__ k_cache,        // [num_blocks, Hkv, hd/8, 16, 8]
    const uint16_t* __restrict__ v_cache,        // [num_blocks, Hkv, hd, 16]
    const float* __restrict__ block_tables,      // [num_seqs, max_num_blocks_per_seq]
    const float* __restrict__ context_lens,      // [num_seqs]
    const float* __restrict__ alibi_slopes,      // [num_heads](use_alibi=0 不解引用)
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ out) {
  vllm::paged_attention_v1_kernel_f16<256, 16, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale, block_tables, context_lens,
      max_num_blocks_per_seq, alibi_slopes, use_alibi, q_stride, kv_block_stride,
      kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_f16_hd256(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    const float* __restrict__ context_lens,
    const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums,                // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,              // 同上
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ tmp_out) {           // OUT(末参,契约 4)
  vllm::paged_attention_v2_kernel_f16<256, 16, 128, 512>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_reduce_f16_hd256(
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const uint16_t* __restrict__ tmp_out,
    const float* __restrict__ context_lens,      // [num_seqs]
    const int max_num_partitions,
    uint16_t* __restrict__ out) {               // OUT(末参,契约 4)
  vllm::paged_attention_v2_reduce_kernel_f16<256, 128, 512>(
      out, exp_sums, max_logits, tmp_out, context_lens, max_num_partitions);
}

extern "C" __global__ void vllm_paged_attention_v1_f16_hd128bs32(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    const float* __restrict__ context_lens,
    const float* __restrict__ alibi_slopes,
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ out) {
  vllm::paged_attention_v1_kernel_f16<128, 32, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale, block_tables, context_lens,
      max_num_blocks_per_seq, alibi_slopes, use_alibi, q_stride, kv_block_stride,
      kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v1_f16_hd256bs32(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    const float* __restrict__ context_lens,
    const float* __restrict__ alibi_slopes,
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ out) {
  vllm::paged_attention_v1_kernel_f16<256, 32, 128>(
      out, q, k_cache, v_cache, num_kv_heads, scale, block_tables, context_lens,
      max_num_blocks_per_seq, alibi_slopes, use_alibi, q_stride, kv_block_stride,
      kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_f16_hd128bs32(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    const float* __restrict__ context_lens,
    const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  vllm::paged_attention_v2_kernel_f16<128, 32, 128, 512>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_f16_hd256bs32(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    const float* __restrict__ context_lens,
    const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  vllm::paged_attention_v2_kernel_f16<256, 32, 128, 512>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

// ---- B6:fp8 e4m3 KV 读变体(v2;4 形状)----
extern "C" __global__ void vllm_paged_attention_v2_fp8_hd128(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    const float* __restrict__ context_lens,
    const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  // B6:fp8 e4m3 KV 读变体(存储 1B/elem 同逻辑布局;读入即转 half;
  // stride 语义不变,host 传元素序 stride)
  vllm::paged_attention_kernel_f16<128, 16, 128, 512, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_fp8_hd128bs32(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    const float* __restrict__ context_lens,
    const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  // B6:fp8 e4m3 KV 读变体(存储 1B/elem 同逻辑布局;读入即转 half;
  // stride 语义不变,host 传元素序 stride)
  vllm::paged_attention_kernel_f16<128, 32, 128, 512, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_fp8_hd256(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    const float* __restrict__ context_lens,
    const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  // B6:fp8 e4m3 KV 读变体(存储 1B/elem 同逻辑布局;读入即转 half;
  // stride 语义不变,host 传元素序 stride)
  vllm::paged_attention_kernel_f16<256, 16, 128, 512, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_fp8_hd256bs32(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    const float* __restrict__ block_tables,
    const float* __restrict__ context_lens,
    const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    const int num_kv_heads,
    const float scale,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride,
    const float softscapping,
    const int sliding_window,
    const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  // B6:fp8 e4m3 KV 读变体(存储 1B/elem 同逻辑布局;读入即转 half;
  // stride 语义不变,host 传元素序 stride)
  vllm::paged_attention_kernel_f16<256, 32, 128, 512, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

// ---- kNHD 变体(kv布局统一契约 P2;模板尾参 KNHD=true)----
// host 传 kv_block_stride = page·hkv·hd、kv_head_stride = hd(元素序,
// fp8 = 字节序同式);d→线程映射与 classic 一致 → 输出位等。

extern "C" __global__ void vllm_paged_attention_v2_f16_knhd_hd256bs32(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const float* __restrict__ block_tables,
    const float* __restrict__ context_lens, const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    const int num_kv_heads, const float scale, const int max_num_blocks_per_seq,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window, const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  vllm::paged_attention_kernel_f16<256, 32, 128, 512, false, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_f16_knhd_hd256(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const float* __restrict__ block_tables,
    const float* __restrict__ context_lens, const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    const int num_kv_heads, const float scale, const int max_num_blocks_per_seq,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window, const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  vllm::paged_attention_kernel_f16<256, 16, 128, 512, false, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_fp8_knhd_hd256bs32(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const float* __restrict__ block_tables,
    const float* __restrict__ context_lens, const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    const int num_kv_heads, const float scale, const int max_num_blocks_per_seq,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window, const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  vllm::paged_attention_kernel_f16<256, 32, 128, 512, true, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_fp8_knhd_hd256(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const float* __restrict__ block_tables,
    const float* __restrict__ context_lens, const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    const int num_kv_heads, const float scale, const int max_num_blocks_per_seq,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window, const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  vllm::paged_attention_kernel_f16<256, 16, 128, 512, true, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_f16_knhd_hd128bs32(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const float* __restrict__ block_tables,
    const float* __restrict__ context_lens, const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    const int num_kv_heads, const float scale, const int max_num_blocks_per_seq,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window, const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  vllm::paged_attention_kernel_f16<128, 32, 128, 512, false, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_f16_knhd_hd128(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const float* __restrict__ block_tables,
    const float* __restrict__ context_lens, const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    const int num_kv_heads, const float scale, const int max_num_blocks_per_seq,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window, const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  vllm::paged_attention_kernel_f16<128, 16, 128, 512, false, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_fp8_knhd_hd128bs32(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const float* __restrict__ block_tables,
    const float* __restrict__ context_lens, const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    const int num_kv_heads, const float scale, const int max_num_blocks_per_seq,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window, const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  vllm::paged_attention_kernel_f16<128, 32, 128, 512, true, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}

extern "C" __global__ void vllm_paged_attention_v2_fp8_knhd_hd128(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const float* __restrict__ block_tables,
    const float* __restrict__ context_lens, const float* __restrict__ alibi_slopes,
    float* __restrict__ exp_sums, float* __restrict__ max_logits,
    const int num_kv_heads, const float scale, const int max_num_blocks_per_seq,
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float softscapping, const int sliding_window, const int use_alibi,
    uint16_t* __restrict__ tmp_out) {
  vllm::paged_attention_kernel_f16<128, 16, 128, 512, true, true>(
      exp_sums, max_logits, tmp_out, q, k_cache, v_cache, num_kv_heads, scale,
      block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, use_alibi,
      q_stride, kv_block_stride, kv_head_stride, softscapping, sliding_window);
}
