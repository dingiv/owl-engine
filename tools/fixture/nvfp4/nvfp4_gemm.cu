// ============================================================================
// owl_nvfp4_gemm v1 适配器(sm_120a):CUTLASS SM120 块缩放 mainloop + 默认 epilogue。
// 本文件**先做成独立可执行**验证(下一步再拆成 extern "C" + Rust FFI)。
// ============================================================================
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp"
#include "cutlass/gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

using namespace cute;
using ElemA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElemB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using TileShape = Shape<_128, _128, _256>;

using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, Shape<_1, _1, _1>,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElemA, cutlass::layout::RowMajor, 32,
    ElemB, cutlass::layout::ColumnMajor, 32,
    float, TileShape, Shape<_1, _1, _1>,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

using Kernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, Mainloop, Epilogue, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using Sm1xxBlkScaledConfig = typename Mainloop::Sm1xxBlkScaledConfig;
using StrideA = typename Mainloop::StrideA;
using StrideB = typename Mainloop::StrideB;
using ElementA = typename Mainloop::ElementA;
using ElementB = typename Mainloop::ElementB;

// ---- e2m1 / e4m3 编码(host) ----
static const float E2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static inline uint8_t e2m1_enc(float x) {
  int s = x < 0 ? 8 : 0; float a = fabsf(x); int b = 0; float bd = 1e30f;
  for (int i = 0; i < 8; ++i) { float d = fabsf(a - E2M1[i]); if (d < bd) { bd = d; b = i; } }
  return (uint8_t)(s | b);
}
static inline float e4m3_dec(uint8_t c) {
  int ex = (c >> 3) & 0xF, mant = c & 7;
  return (ex == 0) ? (mant / 8.f) * powf(2.f, -6.f) : (1.f + mant / 8.f) * powf(2.f, ex - 7.f);
}
static inline uint8_t e4m3_enc(float x) {
  if (!(x > 0)) x = 0.001953125f;
  if (x > 448.f) x = 448.f;
  int b = 0; float bd = 1e30f;
  for (int code = 1; code < 127; ++code) { float d = fabsf(e4m3_dec((uint8_t)code) - x); if (d < bd) { bd = d; b = code; } }
  return (uint8_t)b;
}

// ---- 量化 + 按 CUTLASS SF 布局落盘 ----
// a: [rows,K] f16;输出 q:[rows][K/2] 与 sf(按 layout 交错),返回全局缩放
template <class TensorSF>
static float quantize(const __half* a, int rows, int K, uint8_t* q, TensorSF sf) {
  float amax = 0.f;
  for (size_t i = 0; i < (size_t)rows * K; ++i) amax = fmaxf(amax, fabsf(__half2float(a[i])));
  const float g = amax > 0 ? amax / (6.f * 448.f) : 1.f;
  for (int r = 0; r < rows; ++r) {
    for (int kg = 0; kg < K / 16; ++kg) {
      float mx = 0.f;
      for (int i = 0; i < 16; ++i) mx = fmaxf(mx, fabsf(__half2float(a[(size_t)r * K + kg * 16 + i])));
      uint8_t sc = e4m3_enc(mx > 0 ? mx / (6.f * g) : 1.f);
      sf(r, 16 * kg, 0) = sc;   // CUTLASS 交错布局:按 (row, k, l) 写,内核按同一布局读
      float sd = e4m3_dec(sc) * g;
      for (int i = 0; i < 16; ++i) {
        uint8_t qq = e2m1_enc(__half2float(a[(size_t)r * K + kg * 16 + i]) / (sd > 0 ? sd : 1.f));
        size_t idx = (size_t)r * (K / 2) + (kg * 16 + i) / 2;
        if (((kg * 16 + i) & 1) == 0) q[idx] |= qq; else q[idx] |= (uint8_t)(qq << 4);
      }
    }
  }
  return g;
}

int main(int argc, char** argv) {
  const char* wpath = argc > 1 ? argv[1] : "/tmp/w_gate.bin";
  int N = argc > 2 ? atoi(argv[2]) : 9216;
  int K = argc > 3 ? atoi(argv[3]) : 2560;
  int M = argc > 4 ? atoi(argv[4]) : 512;
  std::vector<__nv_bfloat16> wb((size_t)N * K);
  FILE* fp = fopen(wpath, "rb");
  fread(wb.data(), 2, (size_t)N * K, fp); fclose(fp);
  std::vector<__half> w((size_t)N * K);
  for (size_t i = 0; i < w.size(); ++i) w[i] = __float2half(__bfloat162float(wb[i]));
  // 激活:用同样的权重当 A(只为量吞吐;数值不与参考比)
  std::vector<__half> x((size_t)M * K);
  for (size_t i = 0; i < x.size(); ++i) x[i] = w[i % (size_t)N * K];

  auto l_sfa = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
  auto l_sfb = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
  size_t sfa_elems = size(l_sfa), sfb_elems = size(l_sfb);
  printf("SF 元素数: SFA=%zu SFB=%zu (行主序会是 %zu / %zu)\n", sfa_elems, sfb_elems,
         (size_t)M * (K / 16), (size_t)N * (K / 16));

  std::vector<uint8_t> aq((size_t)M * K / 2), bq((size_t)N * K / 2);
  std::vector<uint8_t> asf(sfa_elems), bsf(sfb_elems);
  auto tSFA = make_tensor(asf.data(), l_sfa);
  auto tSFB = make_tensor(bsf.data(), l_sfb);
  float ga = quantize(x.data(), M, K, aq.data(), tSFA);
  float gb = quantize(w.data(), N, K, bq.data(), tSFB);

  uint8_t *dA, *dB, *dSFA, *dSFB; __half *dD;
  cudaMalloc(&dA, aq.size()); cudaMalloc(&dB, bq.size());
  cudaMalloc(&dSFA, asf.size()); cudaMalloc(&dSFB, bsf.size());
  cudaMalloc(&dD, (size_t)M * N * sizeof(__half));
  cudaMemcpy(dA, aq.data(), aq.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, bq.data(), bq.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dSFA, asf.data(), asf.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dSFB, bsf.data(), bsf.size(), cudaMemcpyHostToDevice);

  // 打包 stride(单位 = 元素;fp4 由 subbyte 迭代器处理)
  StrideA stride_A{ (int64_t)K, cute::C<1>{}, (int64_t)M * K };
  StrideB stride_B{ (int64_t)K, cute::C<1>{}, (int64_t)N * K };
  using StrideC = typename Epilogue::StrideC;
  using StrideD = typename Epilogue::StrideD;
  StrideC stride_C{ (int64_t)N, cute::C<1>{}, (int64_t)M * N };
  StrideD stride_D{ (int64_t)N, cute::C<1>{}, (int64_t)M * N };
  typename Kernel::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, N, K, 1},
      {reinterpret_cast<ElementA const*>(dA), stride_A,
       reinterpret_cast<ElementB const*>(dB), stride_B,
       reinterpret_cast<typename Mainloop::ElementSF*>(dSFA), l_sfa,
       reinterpret_cast<typename Mainloop::ElementSF*>(dSFB), l_sfb},
      {{ga * gb, 0.f}, nullptr, stride_C, reinterpret_cast<typename Epilogue::ElementD*>(dD), stride_D}};

  Gemm gemm_op;
  auto st = gemm_op.can_implement(args);
  if (st != cutlass::Status::kSuccess) { printf("can_implement 失败\n"); return 1; }
  size_t ws = Gemm::get_workspace_size(args);
  void* dW = nullptr; if (ws) cudaMalloc(&dW, ws);
  cudaStream_t stream; cudaStreamCreate(&stream);
  st = gemm_op.initialize(args, dW, stream);
  if (st != cutlass::Status::kSuccess) { printf("initialize 失败\n"); return 1; }
  printf("workspace=%zu 字节\n", ws);
  for (int i = 0; i < 3; ++i) gemm_op.run(stream);
  cudaError_t e = cudaStreamSynchronize(stream);
  if (e != cudaSuccess) { printf("run 失败: %s\n", cudaGetErrorString(e)); return 1; }
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int iters = 50;
  cudaEventRecord(e0, stream);
  for (int i = 0; i < iters; ++i) gemm_op.run(stream);
  cudaEventRecord(e1, stream); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double tf = 2.0 * M * N * K * iters / (ms * 1e-3) / 1e12;
  printf("\nM=%d N=%d K=%d  ⇒  owl_nvfp4_gemm %8.0f GFLOPS (%.3f ms/iter)\n", M, N, K, tf * 1000, ms / iters);
  printf("参考:CUTLASS 官方样例同形状 429000 GFLOPS · 现役 MMQ2 145000 GFLOPS\n");

  // ---- 真权重数值对拍(4×4 切片;CPU 参考 = 同一批打包数据反量化)----
  if (M <= 512) {
    std::vector<__half> o((size_t)M * N);
    cudaMemcpy(o.data(), dD, o.size() * 2, cudaMemcpyDeviceToHost);
    auto deq = [&](const std::vector<uint8_t>& q, int r, int k, float g,
                   uint8_t sc) {
      uint8_t byte = q[(size_t)r * (K / 2) + k / 2];
      uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
      const float T[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
      float v = T[nib & 7] * ((nib & 8) ? -1.f : 1.f);
      return v * e4m3_dec(sc) * g;
    };
    double mx = 0, mxref = 0;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j) {
        int m = i * (M / 4), n = j * (N / 4);
        double ref = 0;
        for (int k = 0; k < K; ++k) {
          uint8_t sa = tSFA(m, 16 * (k / 16), 0), sb = tSFB(n, 16 * (k / 16), 0);
          ref += (double)deq(aq, m, k, ga, sa) * (double)deq(bq, n, k, gb, sb);
        }
        double got = __half2float(o[(size_t)m * N + n]);
        double d = fabs(got - ref);
        mx = fmax(mx, d); mxref = fmax(mxref, fabs(ref));
        if (i == 0 && j == 0) printf("  切片校验 [%d,%d] got=%.4f ref=%.4f\n", m, n, got, ref);
      }
    printf("  真权重对拍: max|ref|=%.2f  max|Δ|=%.4f (%.2e 相对) %s\n", mxref, mx, mx / (mxref + 1e-9),
           mx / (mxref + 1e-9) < 5e-3 ? "✅ 通过" : "❌ 超差");
  }
  return 0;
}
