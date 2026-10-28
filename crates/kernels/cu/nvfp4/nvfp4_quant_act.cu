// ============================================================================
// owl_nvfp4 激活量化器(设备侧,sm_120a)—— 把 f16 `[M,K]` 打成
// **e2m1 + e4m3/16 块缩放**,与 host `owl_nvfp4_pack_a` 同语义。
//
// 为什么要有:host 打包必须把激活 D2H 再 H2D(prefill 一个 chunk 200+ 发 GEMM,
// 每次 10 MB 往返)⇒ 等于没有。这个核把量化留在设备上,只回读 4 字节全局缩放。
//
// ## 语义(与 host 侧逐条对齐)
// 1. **per-tensor 全局缩放** `g = amax/(6*448)`(amax 由两个核的第一发算)
// 2. **每 16 个 k 一个 e4m3 局部缩放** `sc = e4m3(max|a_g|/(6*g))`
// 3. 数据 `q = e2m1(a/(e4m3_dec(sc)*g))`,每字节两个 k(低 nibble = 偶数 k)
//
// ## 为什么"每线程一组 + 手算 SF 偏移"
// SF 是 CUTLASS 的**交错布局**(stride-0 维 + (32,4) atom + 128 行分块),核内
// 直接调 `tile_atom_to_shape_SFA` 拿不到运行期戳。做法:plan 期从 cute 布局**实测**
// 出 5 个仿射步长(`owl_nvfp4_sf_strides`),并对一批采样点断言仿射模型与 cute
// 逐点一致 —— 不一致就拒绝建 plan(宁可不跑,不写错位)。
// ============================================================================
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

namespace owl_nvfp4_q {
constexpr int kGroup = 16;   // 每 16 个 k 一个 e4m3 缩放
constexpr int kBlock = 256;

/// amax → 单元素(块内 warp 归约 + 全局 atomicMax;amax ≥ 0 ⇒ float 位序可比)
__global__ void k_amax(const __half* __restrict__ x, long n, float* __restrict__ out) {
  float m = 0.f;
  for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (long)gridDim.x * blockDim.x) {
    m = fmaxf(m, fabsf(__half2float(x[i])));
  }
  // warp 归约
  for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
  __shared__ float smem[kBlock / 32];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  if (lane == 0) smem[warp] = m;
  __syncthreads();
  if (warp == 0) {
    m = (lane < kBlock / 32) ? smem[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    if (lane == 0) atomicMax(reinterpret_cast<int*>(out), __float_as_int(m));
  }
}

/// 一组(16 个 k)→ 8 字节 fp4 + 1 字节 e4m3;SF 偏移用 plan 期实测的仿射步长
__global__ void k_quant(const __half* __restrict__ x, int rows, int K,
                        const float* __restrict__ d_g,
                        uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                        int s_m0, int s_m1, int s_m2, int s_g0, int s_g1) {
  const float g = d_g[0];
  const int groups = K / kGroup;
  const long total = (long)rows * groups;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (long)gridDim.x * blockDim.x) {
    const int row = (int)(idx / groups);
    const int gg = (int)(idx % groups);
    const __half* src = x + (long)row * K + (long)gg * kGroup;
    float v[kGroup];
    float mx = 0.f;
    #pragma unroll
    for (int i = 0; i < kGroup; ++i) {
      v[i] = __half2float(src[i]);
      mx = fmaxf(mx, fabsf(v[i]));
    }
    // 局部缩放:e4m3 编码(硬件转换,round-to-nearest)
    const float want = (mx > 0.f && g > 0.f) ? mx / (6.f * g) : 1.f;
    const __nv_fp8_storage_t sc = __nv_cvt_float_to_fp8(want, __NV_SATFINITE, __NV_E4M3);
    const __half_raw hr = __nv_cvt_fp8_to_halfraw(sc, __NV_E4M3);
    const float sd = __half2float(__half(hr)) * (g > 0.f ? g : 1.f);
    // 数据:e2m1,两两打包(低 nibble = 偶数 k)
    uint8_t out[kGroup / 2];
    #pragma unroll
    for (int i = 0; i < kGroup; i += 2) {
      float2 pair = make_float2(sd != 0.f ? v[i] / sd : 0.f, sd != 0.f ? v[i + 1] / sd : 0.f);
      out[i / 2] = __nv_cvt_float2_to_fp4x2(pair, __NV_E2M1, cudaRoundNearest);
    }
    uint8_t* dst = q + (long)row * (K / 2) + (long)gg * (kGroup / 2);
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(out);  // 8 字节
    // SF 交错布局偏移(仿射;plan 期已校验)
    const int m0 = row % 32, m1 = (row / 32) % 4, m2 = row / 128;
    const int g0 = gg % 4, g1 = gg / 4;
    sf[m0 * s_m0 + m1 * s_m1 + m2 * s_m2 + g0 * s_g0 + g1 * s_g1] = (uint8_t)sc;
  }
}
/// amax → 全局缩放 `g = amax/(6*448)`(单线程小核,省掉一次 D2H 往返)。
///
/// **同时产出 `alpha = g·gb`**(2026-10-28 进图刀):`gb` 是**装载期常量**
/// (权重打包时算出)⇒ 完全可在核内做乘法。这样 GEMM 的 epilogue 只需一个
/// **设备指针** `alpha_ptr` 就能拿到 alpha —— 不再需要回读 `g` 到 host,
/// 本族因此**可进 CUDA graph**(decode/propose/verify 三张图都能捕)。
///
/// **两块值挤一槽**(`d_g` = f32[2]):`d_g[0]` = `g`(`k_quant` 要读),
/// `d_g[1]` = `alpha`(GEMM 的 `alpha_ptr`,Rust 面传 `d_g + 1`)。
/// 为什么不分开两槽:op 参数表/客户端/层声明都要跟着多一个槽,而这两个值
/// 本来就同生同死(同一发小核写定)。
__global__ void k_scale_from_amax(float* d_g, float gb) {
  const float amax = d_g[0];
  const float g = amax > 0.f ? amax / (6.f * 448.f) : 1.f;
  d_g[0] = g;
  d_g[1] = g * gb;
}
}  // namespace owl_nvfp4_q

extern "C" {

/// 三发:`d_g` 清零 → amax → (g, quantize)。`d_g` **由本函数自己清** ——
/// 调用方不必再造一个 zeros 节点(层侧少一发 memcpy 节点,少一个顺序前提)。
/// 返回 0 = 已发射(未同步)。
int owl_nvfp4_quant_act_launch(const __half* x, int M, int K, uint8_t* q, uint8_t* sf,
                               float* d_g, float gb,
                               int s_m0, int s_m1, int s_m2, int s_g0, int s_g1,
                               void* stream) {
  if (!x || !q || !sf || !d_g || M <= 0 || K <= 0 || (K % 16) != 0) return -1;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  // amax 用 atomicMax 归约 ⇒ 先清零(与本流同序,零额外前提;`d_g` = f32[2])
  cudaMemsetAsync(d_g, 0, 2 * sizeof(float), s);
  const long n = (long)M * K;
  int blocks = (int)((n + owl_nvfp4_q::kBlock - 1) / owl_nvfp4_q::kBlock);
  if (blocks > 4096) blocks = 4096;
  owl_nvfp4_q::k_amax<<<blocks, owl_nvfp4_q::kBlock, 0, s>>>(x, n, d_g);
  // g = amax/(6*448) 与 alpha = g·gb(设备侧收尾;零 D2H)
  owl_nvfp4_q::k_scale_from_amax<<<1, 1, 0, s>>>(d_g, gb);
  const long total = (long)M * (K / 16);
  int qblocks = (int)((total + owl_nvfp4_q::kBlock - 1) / owl_nvfp4_q::kBlock);
  if (qblocks > 65535) qblocks = 65535;
  owl_nvfp4_q::k_quant<<<qblocks, owl_nvfp4_q::kBlock, 0, s>>>(
      x, M, K, d_g, q, sf, s_m0, s_m1, s_m2, s_g0, s_g1);
  return 0;
}

}  // extern "C"
