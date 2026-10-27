// NVFP4 数值门(第一步):全 1.0 SF ⇒ 只验 fp4 数据布局 + MMA + epilogue + 输出布局。
// 参考 = 用同一批 fp4 nibble 做的 f32 GEMM(乘 per-tensor 全局缩放)。
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
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
#include "cute/tensor.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
using namespace cute;
using ElemA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElemB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using TileShape = Shape<_128,_128,_256>;
using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassTensorOp, TileShape, Shape<_1,_1,_1>,
    cutlass::epilogue::collective::EpilogueTileAuto, float, float,
    cutlass::half_t, cutlass::layout::RowMajor, 8, cutlass::half_t, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
    ElemA, cutlass::layout::RowMajor, 32, ElemB, cutlass::layout::ColumnMajor, 32,
    float, TileShape, Shape<_1,_1,_1>,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, Mainloop, Epilogue, cutlass::gemm::PersistentScheduler>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
using Cfg = typename Mainloop::Sm1xxBlkScaledConfig;
using ElementA = typename Mainloop::ElementA; using ElementB = typename Mainloop::ElementB;
using StrideA = typename Mainloop::StrideA; using StrideB = typename Mainloop::StrideB;

int main(int argc, char** argv) {
  int M = argc>1?atoi(argv[1]):128, N = argc>2?atoi(argv[2]):128, K = argc>3?atoi(argv[3]):256;
  std::mt19937 rng(7); std::normal_distribution<float> nd(0.f,1.f);
  // 直接生成 fp4 nibble(不经量化器),这样参考与核用的是同一批比特
  auto mk4 = [&](size_t n){ std::vector<uint8_t> q(n/2); for (auto& b: q) b = (uint8_t)(rng()&0xFF); return q; };
  auto aq = mk4((size_t)M*K), bq = mk4((size_t)N*K);
  float ga = 0.5f, gb = 0.25f;

  // 参考:同 nibble 的 f32 GEMM(decode e2m1,scale=1)
  auto dec = [](uint8_t nib){ const float T[8]={0,.5f,1,1.5f,2,3,4,6}; return T[nib&7]*((nib&8)?-1.f:1.f); };
  auto val = [&](const std::vector<uint8_t>& q, int r, int c){ uint8_t b=q[(size_t)r*(K/2)+c/2]; return dec((c&1)?(b>>4):(b&0xF)); };
  std::vector<float> ref((size_t)M*N,0.f);
  for (int m=0;m<M;++m) for (int n=0;n<N;++n) { float s=0; for (int k=0;k<K;++k) s += val(aq,m,k)*val(bq,n,k); ref[(size_t)m*N+n]=s*ga*gb; }

  auto lsfa = Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lsfb = Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  std::vector<uint8_t> asf(size(lsfa), 0x38), bsf(size(lsfb), 0x38);  // e4m3 1.0
  uint8_t *dA,*dB,*dSFA,*dSFB; __half* dD;
  cudaMalloc(&dA,aq.size()); cudaMalloc(&dB,bq.size()); cudaMalloc(&dSFA,asf.size()); cudaMalloc(&dSFB,bsf.size());
  cudaMalloc(&dD,(size_t)M*N*sizeof(__half));
  cudaMemcpy(dA,aq.data(),aq.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,bq.data(),bq.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSFA,asf.data(),asf.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSFB,bsf.data(),bsf.size(),cudaMemcpyHostToDevice);
  StrideA sA{(int64_t)K, C<1>{}, (int64_t)M*K}; StrideB sB{(int64_t)K, C<1>{}, (int64_t)N*K};
  using StrideD = typename Epilogue::StrideD; StrideD sD{(int64_t)N, C<1>{}, (int64_t)M*N};
  typename Kernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
    {reinterpret_cast<ElementA const*>(dA), sA, reinterpret_cast<ElementB const*>(dB), sB,
     reinterpret_cast<typename Mainloop::ElementSF*>(dSFA), lsfa,
     reinterpret_cast<typename Mainloop::ElementSF*>(dSFB), lsfb},
    {{ga*gb, 0.f}, nullptr, sD, reinterpret_cast<typename Epilogue::ElementD*>(dD), sD}};
  Gemm gemm; cudaStream_t st; cudaStreamCreate(&st);
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) { printf("can_implement 失败\n"); return 1; }
  size_t ws = Gemm::get_workspace_size(args); void* dW=nullptr; if (ws) cudaMalloc(&dW,ws);
  if (gemm.initialize(args, dW, st) != cutlass::Status::kSuccess) { printf("init 失败\n"); return 1; }
  if (gemm.run(st) != cutlass::Status::kSuccess) { printf("run 失败\n"); return 1; }
  cudaStreamSynchronize(st);
  std::vector<__half> out((size_t)M*N); cudaMemcpy(out.data(), dD, out.size()*2, cudaMemcpyDeviceToHost);
  double mx=0, mxref=0, sum=0; int bad=0;
  for (size_t i=0;i<out.size();++i){ double d=fabs((double)__half2float(out[i])-ref[i]); mx=fmax(mx,d); mxref=fmax(mxref,fabs(ref[i])); sum+=d; if (d>1e-2*fmax(1.0,fabs(ref[i]))) bad++; }
  printf("M=%d N=%d K=%d | max|ref|=%.1f | max|Δ|=%.4f | mean|Δ|=%.5f | 超差点 %d/%zu\n",
         M,N,K,mxref,mx,sum/out.size(),bad,out.size());
  printf("%s\n", bad==0 ? "✅ 数值门(数据通路)通过" : "❌ 有偏差");
  printf("样例 out[0][0..5]="); for (int i=0;i<6;i++) printf("%.3f ", (float)__half2float(out[i]));
  printf(" ref="); for (int i=0;i<6;i++) printf("%.3f ", ref[i]); printf("\n");
  return 0;
}
