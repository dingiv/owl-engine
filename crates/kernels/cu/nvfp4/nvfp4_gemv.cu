// ============================================================================
// owl_nvfp4 **decode 臂**:NVFP4 权重 × fp4 激活的 GEMV/小 M 核(sm_120a)。
//
// ## 为什么需要它(2026-11-08 实测)
//
// `nvfp4_gemm.cu` 的 CUTLASS 主循环是 **prefill 形状**的机器(Tile 128×128×256、
// TMA 多级流水、warp-specialization)。按 2026-10-28 的"统一臂"裁决,decode(m=1)
// 也走它 —— 结果是 m=1 的 128 行 tile 空转:每 token 权重 2.049 GB,
// 实测只有 **242 GB/s**(峰值 896 的 27%),8.48 ms/token;
// 而同机 marlin(W4A16)在 m=1 上能跑 358 GB/s。
//
// 本核把 decode 从"张量核"里摘出来:不用 MMA、不用 TMA,**纯 SIMT 流式反量化**。
// 目标 = 贴带宽墙,与 `owl_gguf_gemm_q4k`(实测 860 GB/s)同一条路。
//
// ## 布局契约(全部是实测事实,不推理)
//
// 权重的 fp4 数据 `bq` = `[N][K/2]` 行主序(每字节两个 k,低 nibble 在前);
// 块缩放 `bsf` 走 CUTLASS 交错布局,偏移是**仿射**的(plan 建时实测步长):
// ```
//   off(n, gg) = s0·(n%32) + s1·((n/32)%4) + s2·(n/128) + s3·(gg%4) + s4·(gg/4)
// ```
// 本机实测(sm_120 / K=2560, 9216, 4096 全同形):`s0=16, s1=4, s3=1, s4=512`,
// `s2 = 128·K/16`(= 8K)。
//
// **关键观察**:固定一个 (group-quad, n 的 32 行块)时,`s0=16, s1=4, s3=1`
// ⇒ **32 行 × 4 组 = 512 字节连续**。于是"按 k 切 warp 内分工"(数据侧才连续)
// 与"按 n 读缩放"不再冲突:同一个 32 字节扇区里正好装着
// `{2 行} × {4 个 b 值} × {4 个组}`,一个 warp 只要负责这 8 行,缩放就**零浪费**。
//
// ## 分工(每 CTA = 一个 SFB 128 行块 × 一个 m)
//
// ```
//   block = 512 线程 = 16 warp;grid = (ceil(N/128), M)
//   warp w   → 扇区 p = w      → 行 n = n0 + 32b + 2p + r (b=0..3, r=0..1 → 8 行)
//   lane l   → 组 quad q = l, l+32, … → k ∈ [64q, 64q+64) → bq 字节 [32q, 32q+32)
// ```
// 每个 lane:每 q 读 8 行的各 32 字节(= 8 个满扇区)+ 8 个 u32 缩放(同一扇区,L1 命中)
// ⇒ 数据 100% 利用,缩放 1/1(按 CTA 摊),理论效率 ≈ 89%(缩放字节占 1/8)。
//
// ## 与 GEMM 臂的数值关系
//
// 输入是**同一条** `quant_act` 产出的 `aq/asf`(不是 f16 激活)⇒ 量化噪声与
// CUTLASS 臂一致,`ops/oracles/nvfp4.rs` 的 host f32 参考对两臂都成立
// (判据 1e-2;本核组内用 half2 FMA、组间 f32 累加 ⇒ 额外误差 ~1e-3)。
//
// ## 分流纪律
//
// `m ≤ 8` 且形状/布局过 [`owl_nvfp4_gemv_ok`] 才用本核;否则调用方回落 CUTLASS 臂。
// ============================================================================
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

namespace {

constexpr int kWarps = 2;           // 一个 CTA = 2 个扇区(4 行);切小是为了**波次**(实测 2 ≳ 4 > 8)
constexpr int kSectors = 16;        // 一个 SFB 行块 = 16 个扇区 = 128 行
constexpr int kRowsPerWarp = 8;     // = 4 个 b × 2 个 r
constexpr int kRowBlock = 128;      // SFB 的行块
constexpr int kRowsPerCta = kWarps * 2;  // CTA 覆盖的连续行数(2p 与 2p+1)
constexpr int kQuad = 4;            // 一个 quad = 4 组 = 64 个 k
constexpr int kKPerQuad = 64;

__device__ __forceinline__ float e4m3_to_f32(uint8_t c) {
  return __half2float(__half(__nv_cvt_fp8_to_halfraw(c, __NV_E4M3)));
}

/// 一个字节 = 两个 fp4 码点 → `__half2`(硬件 LUT,1 条指令)
__device__ __forceinline__ __half2 fp4_pair(uint8_t b) {
  const __half2_raw r = __nv_cvt_fp4x2_to_halfraw2(b, __NV_E2M1);
  return *reinterpret_cast<const __half2*>(&r);
}

/// 8 字节(16 个 fp4 = 2 个 u32)对 16 个激活值做点积(`av` 是激活的 8 个 half2)。
__device__ __forceinline__ float dot16(uint32_t x, uint32_t y, const __half2* av) {
  // 权重字节 → 8 个 half2(低 nibble 在前 ⇒ 字节序即 k 序)
  const __half2 b0 = fp4_pair(static_cast<uint8_t>(x));
  const __half2 b1 = fp4_pair(static_cast<uint8_t>(x >> 8));
  const __half2 b2 = fp4_pair(static_cast<uint8_t>(x >> 16));
  const __half2 b3 = fp4_pair(static_cast<uint8_t>(x >> 24));
  const __half2 b4 = fp4_pair(static_cast<uint8_t>(y));
  const __half2 b5 = fp4_pair(static_cast<uint8_t>(y >> 8));
  const __half2 b6 = fp4_pair(static_cast<uint8_t>(y >> 16));
  const __half2 b7 = fp4_pair(static_cast<uint8_t>(y >> 24));
  __half2 acc = __hmul2(av[0], b0);
  acc = __hfma2(av[1], b1, acc);
  acc = __hfma2(av[2], b2, acc);
  acc = __hfma2(av[3], b3, acc);
  acc = __hfma2(av[4], b4, acc);
  acc = __hfma2(av[5], b5, acc);
  acc = __hfma2(av[6], b6, acc);
  acc = __hfma2(av[7], b7, acc);
  return __low2float(acc) + __high2float(acc);
}

/// 16 字节 = 2 个组(32 个 fp4)→ 两个 `__half2[8]`
__device__ __forceinline__ void act4(const uint4 v, __half2* g0, __half2* g1) {
  g0[0] = fp4_pair(static_cast<uint8_t>(v.x));
  g0[1] = fp4_pair(static_cast<uint8_t>(v.x >> 8));
  g0[2] = fp4_pair(static_cast<uint8_t>(v.x >> 16));
  g0[3] = fp4_pair(static_cast<uint8_t>(v.x >> 24));
  g0[4] = fp4_pair(static_cast<uint8_t>(v.y));
  g0[5] = fp4_pair(static_cast<uint8_t>(v.y >> 8));
  g0[6] = fp4_pair(static_cast<uint8_t>(v.y >> 16));
  g0[7] = fp4_pair(static_cast<uint8_t>(v.y >> 24));
  g1[0] = fp4_pair(static_cast<uint8_t>(v.z));
  g1[1] = fp4_pair(static_cast<uint8_t>(v.z >> 8));
  g1[2] = fp4_pair(static_cast<uint8_t>(v.z >> 16));
  g1[3] = fp4_pair(static_cast<uint8_t>(v.z >> 24));
  g1[4] = fp4_pair(static_cast<uint8_t>(v.w));
  g1[5] = fp4_pair(static_cast<uint8_t>(v.w >> 8));
  g1[6] = fp4_pair(static_cast<uint8_t>(v.w >> 16));
  g1[7] = fp4_pair(static_cast<uint8_t>(v.w >> 24));
}

__global__ __launch_bounds__(kWarps * 32) void gemv_kernel(
    const uint8_t* __restrict__ aq, const uint8_t* __restrict__ asf,
    const float* __restrict__ alpha_dev, float alpha_host,
    const uint8_t* __restrict__ bq, const uint8_t* __restrict__ bsf,
    __half* __restrict__ out, int m, int n, int k, int sfa_base, int sfa0,
    int sfa1, int sfa2, int sfa4, int sfb_base, int sfb0, int sfb1, int sfb2,
    int sfb4) {
  const int lane = threadIdx.x & 31;
  // 扇区号 **全局**(CTA 只吃一小段,把波次拉起来:小 CTA = 更多 CTA = 尾巴更短)
  const int p_global = blockIdx.x * kWarps + (threadIdx.x >> 5);
  const int nblock = p_global / kSectors;     // 128 行块
  const int p = p_global % kSectors;          // 块内扇区 = 行对 (2p, 2p+1)
  const int n0 = nblock * kRowBlock;
  const int mi = blockIdx.y;
  if (mi >= m || n0 >= n) return;

  const int half_k = k >> 1;
  const int quads = k / kKPerQuad;

  // 本 m 行的 SFA 基址(交错布局;组内 4 字节连续 ⇒ 一次 u32 读)
  const int a_off_m = sfa_base + static_cast<int>(sfa0) * (mi % 32) +
                      static_cast<int>(sfa1) * ((mi / 32) % 4) +
                      static_cast<int>(sfa2) * (mi / 128);
  const uint8_t* arow = aq + static_cast<size_t>(mi) * half_k;
  const uint8_t* a_sf = asf + a_off_m;

  // 本 warp 的 8 行:n = n0 + 32b + 2p + r
  int rows[kRowsPerWarp];
  const uint8_t* sfb_row[kRowsPerWarp];
  const uint8_t* bq_row[kRowsPerWarp];
  bool live[kRowsPerWarp];
#pragma unroll
  for (int b = 0; b < 4; ++b) {
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int j = b * 2 + r;
      const int nn = n0 + 32 * b + 2 * p + r;
      rows[j] = nn;
      live[j] = nn < n;
      const size_t off = static_cast<size_t>(sfb_base) +
                         static_cast<size_t>(sfb0) * (2 * p + r) +
                         static_cast<size_t>(sfb1) * b +
                         static_cast<size_t>(sfb2) * nblock;
      sfb_row[j] = bsf + off;
      bq_row[j] = bq + static_cast<size_t>(nn) * half_k;
    }
  }

  float acc[kRowsPerWarp];
#pragma unroll
  for (int j = 0; j < kRowsPerWarp; ++j) acc[j] = 0.f;

  for (int q = lane; q < quads; q += 32) {
    // ---- 激活:4 个组 = 32 字节 = 两个 `uint4`(满扇区)----
    __half2 av[4][8];
    const uint8_t* ap = arow + static_cast<size_t>(q) * 32;
    const uint4 a16 = *reinterpret_cast<const uint4*>(ap);
    const uint4 a16b = *reinterpret_cast<const uint4*>(ap + 16);
    act4(a16, av[0], av[1]);
    act4(a16b, av[2], av[3]);

    // ---- 激活缩放:groups 4q..4q+3 ----
    const uint32_t sa_pack = *reinterpret_cast<const uint32_t*>(
        a_sf + static_cast<size_t>(q) * sfa4);
    float sa[kQuad];
    sa[0] = e4m3_to_f32(static_cast<uint8_t>(sa_pack));
    sa[1] = e4m3_to_f32(static_cast<uint8_t>(sa_pack >> 8));
    sa[2] = e4m3_to_f32(static_cast<uint8_t>(sa_pack >> 16));
    sa[3] = e4m3_to_f32(static_cast<uint8_t>(sa_pack >> 24));

    // 8 行的权重缩放先全部发出(4 字节/行,同扇区)⇒ 让 8 个读并行在飞
    uint32_t sb_pack[kRowsPerWarp];
#pragma unroll
    for (int j = 0; j < kRowsPerWarp; ++j)
      sb_pack[j] = *reinterpret_cast<const uint32_t*>(
          sfb_row[j] + static_cast<size_t>(q) * sfb4);

#pragma unroll
    for (int j = 0; j < kRowsPerWarp; ++j) {
      const uint8_t* wp = bq_row[j] + static_cast<size_t>(q) * 32;
      const uint4 w0 = *reinterpret_cast<const uint4*>(wp);
      const uint4 w1 = *reinterpret_cast<const uint4*>(wp + 16);
      float s = acc[j];
      s += (sa[0] * e4m3_to_f32(static_cast<uint8_t>(sb_pack[j]))) *
           dot16(w0.x, w0.y, av[0]);
      s += (sa[1] * e4m3_to_f32(static_cast<uint8_t>(sb_pack[j] >> 8))) *
           dot16(w0.z, w0.w, av[1]);
      s += (sa[2] * e4m3_to_f32(static_cast<uint8_t>(sb_pack[j] >> 16))) *
           dot16(w1.x, w1.y, av[2]);
      s += (sa[3] * e4m3_to_f32(static_cast<uint8_t>(sb_pack[j] >> 24))) *
           dot16(w1.z, w1.w, av[3]);
      acc[j] = s;
    }
  }

  // ---- warp 内归约(8 个行各一次)----
#pragma unroll
  for (int j = 0; j < kRowsPerWarp; ++j) {
    float v = acc[j];
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffffu, v, s);
    acc[j] = v;
  }

  if (lane == 0) {
    const float al = (alpha_dev != nullptr) ? *alpha_dev : alpha_host;
    __half* o = out + static_cast<size_t>(mi) * n;
#pragma unroll
    for (int j = 0; j < kRowsPerWarp; ++j) {
      if (live[j]) o[rows[j]] = __float2half(acc[j] * al);
    }
  }
}

}  // namespace

/// `1` = 本核能吃这个形状/布局;`0` = 调用方回落 CUTLASS 臂。
///
/// 硬要求(全部是布局前提,**不满足就不猜**):
/// - `m ∈ [1, 8]`(decode 档)、`n % 128 == 0`(CTA 整块)、`k % 64 == 0`(quad 整块);
/// - SFB 步长 = 实测的 `16/4/128·K/16/1/512`(32 字节扇区假设);
/// - SFA 组内连续(`s3 == 1`)且 `nnz`(`asf`)按 128 行块排(引擎既有口径)。
extern "C" int owl_nvfp4_gemv_ok(int m, int n, int k, const int* sfa, const int* sfb) {
  if (m < 1 || m > 8) return 0;
  if (n <= 0 || (n % 128) != 0) return 0;
  if (k <= 0 || (k % 64) != 0) return 0;
  if (!sfa || !sfb) return 0;
  if (sfa[3] != 1) return 0;                       // 组内 4 字节连续
  if (sfb[0] != 16 || sfb[1] != 4) return 0;       // 行内 32 行块 = 512B
  if (sfb[3] != 1 || sfb[4] != 512) return 0;      // 组交错
  if (sfb[2] != 128 * (k / 16)) return 0;          // 行块步长
  return 1;
}

/// 发射一次 decode GEMV。`alpha_dev` 非空则用它(进图路径),否则用 `alpha_host`。
/// 返回 0 = 已发射;负 = 主机侧失败(调用方回落 CUTLASS 臂)。
///
/// `sfa`/`sfb` = plan 建时实测的 5 个仿射步长;`sfa_base`/`sfb_base` = 布局原点。
extern "C" int owl_nvfp4_gemv_launch(
    const uint8_t* aq, const uint8_t* asf, const float* alpha_dev, float alpha_host,
    const uint8_t* bq, const uint8_t* bsf, __half* d, int m, int n, int k,
    int sfa_base, const int* sfa, int sfb_base, const int* sfb, void* stream) {
  if (!aq || !asf || !bq || !bsf || !d || !stream) return -1;
  if (!owl_nvfp4_gemv_ok(m, n, k, sfa, sfb)) return -1;
  const dim3 grid((unsigned)((n + kRowsPerCta - 1) / kRowsPerCta), (unsigned)m, 1);
  gemv_kernel<<<grid, kWarps * 32, 0, static_cast<cudaStream_t>(stream)>>>(
      aq, asf, alpha_dev, alpha_host, bq, bsf, d, m, n, k, sfa_base, sfa[0], sfa[1],
      sfa[2], sfa[4], sfb_base, sfb[0], sfb[1], sfb[2], sfb[4]);
  return (cudaGetLastError() == cudaSuccess) ? 0 : -2;
}
