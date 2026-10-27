// ============================================================================
// NVFP4(sm_120)算力探针 —— **纯寄存器 MMA 吞吐**(零访存),量天花板。
//
// 目的(2026-10-27,用户裁决「50 系上 Marlin 不如 CUTLASS」):
// 在写整族之前先用最小判据回答一个问题 —— sm_120 的 FP4 张量核到底给多少
// TOPS?若它 ≫ 现役 MMQ2 的 ~145 TFLOPS(int8 legacy mma),整族才值得做。
//
// 实测(RTX 5070 Ti / 70 SM / CUDA 13.2,寄存器内连发):
//
//   MMQ2 int8(现役 prefill)                          ≈ 145 TFLOPS
//   FP4 `kind::f8f6f4` m16n8k32(无块缩放)            ≈ 192 TFLOPS
//   NVFP4 `kind::mxf4nvf4.block_scale.4X` m16n8k64    ≈ 713 TFLOPS  ← 5×
//
// ⇒ 本族只做 **块缩放 NVFP4**(m16n8k64 + e4m3 每 16 缩放):
//   无缩放那条路(K=32/条)反而慢 3.7×,是**指令发射**而不是带宽受限。
//
// # 构建面铁律(踩过)
//
// 必须 `-gencode arch=compute_120a,code=sm_120a`:
// 只写 `-arch=sm_120a` 时 nvcc 把 PTX 目标退成 `compute_120`,ptxas 直接拒
// (`Feature '.kind::mxf4nvf4' not supported on .target 'sm_120'`)。
// 换言之:**FP4 是 arch-specific（'a'）指令集**,族侧按 `Arch::as_sm_a()` 走。
//
// # 口径
//
// FLOPs = grid × warps × iters × 2·M·N·K(取 atom 的 M16N8K64)。
// 累加器回打成 `out` 只在 `d0 == 1234.5f` 时发生(防 DCE,不影响计时)。
// ============================================================================
#include <cstdint>

#include <cute/arch/mma_sm120.hpp>

// 非 'a' 目标(或缺 CUTLASS 头)时不发射任何指令 —— 文件本身永远可编。
#if defined(CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED)

using owl_nvfp4_bs_atom =
    cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<cute::float_e2m1_t,
                                                  cute::float_e2m1_t,
                                                  float,
                                                  cute::float_ue4m3_t,
                                                  16>;

/// 块缩放 NVFP4(m16n8k64):每 warp 一条 = 2·16·8·64 FLOPs
extern "C" __global__ void owl_nvfp4_mma_peak_bs(float* __restrict__ out, int iters) {
  uint32_t a0 = threadIdx.x, a1 = a0 + 1u, a2 = a0 + 2u, a3 = a0 + 3u;
  uint32_t b0 = threadIdx.x * 7u, b1 = b0 + 1u;
  uint32_t sfa = 0x38383838u, sfb = 0x38383838u;  // 4 × e4m3(1.0)
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  for (int i = 0; i < iters; ++i) {
    owl_nvfp4_bs_atom::fma(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3, sfa, sfb);
    a0 += 1u;
  }
  if (d0 == 1234.5f) {
    out[blockIdx.x * blockDim.x + threadIdx.x] = d0 + d1 + d2 + d3;
  }
}

#else

extern "C" __global__ void owl_nvfp4_mma_peak_bs(float* __restrict__ out, int iters) {
  (void)out;
  (void)iters;
}

#endif
