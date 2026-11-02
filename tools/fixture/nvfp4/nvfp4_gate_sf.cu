// 数值门(第二步):SF 落位。数据全 1.0,SF 按 (m, 16g, l) 写进 CUTLASS 交错布局,
// 期望 D[m][n] = Σ_g 16 * sfa_g * sfb_g。
#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
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

static uint8_t e4m3(float x){ // 正数,覆盖 2^-9..448
  int best=1; float bd=1e30f;
  for (int c=1;c<127;++c){ int ex=(c>>3)&0xF,m=c&7;
    float v=(ex==0)?(m/8.f)*powf(2.f,-6.f):(1.f+m/8.f)*powf(2.f,ex-7.f);
    float d=fabsf(v-x); if(d<bd){bd=d;best=c;} }
  return (uint8_t)best; }
static float e4m3d(uint8_t c){ int ex=(c>>3)&0xF,m=c&7;
  return (ex==0)?(m/8.f)*powf(2.f,-6.f):(1.f+m/8.f)*powf(2.f,ex-7.f); }

int main() {
  const int M=128,N=128,K=256;
  std::vector<uint8_t> aq((size_t)M*K/2, 0x22), bq((size_t)N*K/2, 0x22);  // 全 1.0
  auto lsfa=Cfg::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
  auto lsfb=Cfg::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
  std::vector<uint8_t> asf(size(lsfa),0x38), bsf(size(lsfb),0x38);
  // A:组 g 的缩放 = 1,2,4,8 循环;B 全 1.0
  auto tA = make_tensor(asf.data(), lsfa);
  auto tB = make_tensor(bsf.data(), lsfb);
  const float cyc[4]={1.f,2.f,4.f,8.f};
  for (int m=0;m<M;++m) for (int g=0; g<K/16; ++g) tA(m, 16*g, 0) = e4m3(cyc[g%4]);
  for (int n=0;n<N;++n) for (int g=0; g<K/16; ++g) tB(n, 16*g, 0) = e4m3(1.f);
  // 期望
  double expect=0; for (int g=0; g<K/16; ++g) expect += 16.0 * cyc[g%4];
  printf("期望 D = %.3f (Σ_g 16·sfa_g)\n", expect);

  uint8_t *dA,*dB,*dSFA,*dSFB; __half* dD;
  cudaMalloc(&dA,aq.size()); cudaMalloc(&dB,bq.size()); cudaMalloc(&dSFA,asf.size()); cudaMalloc(&dSFB,bsf.size());
  cudaMalloc(&dD,(size_t)M*N*sizeof(__half));
  cudaMemcpy(dA,aq.data(),aq.size(),cudaMemcpyHostToDevice); cudaMemcpy(dB,bq.data(),bq.size(),cudaMemcpyHostToDevice);
  cudaMemcpy(dSFA,asf.data(),asf.size(),cudaMemcpyHostToDevice); cudaMemcpy(dSFB,bsf.data(),bsf.size(),cudaMemcpyHostToDevice);
  StrideA sA{(int64_t)K,C<1>{},(int64_t)M*K}; StrideB sB{(int64_t)K,C<1>{},(int64_t)N*K};
  using StrideD=typename Epilogue::StrideD; StrideD sD{(int64_t)N,C<1>{},(int64_t)M*N};
  typename Kernel::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm,{M,N,K,1},
    {reinterpret_cast<ElementA const*>(dA),sA,reinterpret_cast<ElementB const*>(dB),sB,
     reinterpret_cast<typename Mainloop::ElementSF*>(dSFA),lsfa,
     reinterpret_cast<typename Mainloop::ElementSF*>(dSFB),lsfb},
    {{1.f,0.f},nullptr,sD,reinterpret_cast<typename Epilogue::ElementD*>(dD),sD}};
  Gemm gemm; cudaStream_t st; cudaStreamCreate(&st);
  if (gemm.can_implement(args)!=cutlass::Status::kSuccess){printf("can_implement 失败\n");return 1;}
  size_t ws=Gemm::get_workspace_size(args); void* dW=nullptr; if(ws) cudaMalloc(&dW,ws);
  gemm.initialize(args,dW,st); gemm.run(st); cudaStreamSynchronize(st);
  std::vector<__half> out((size_t)M*N); cudaMemcpy(out.data(),dD,out.size()*2,cudaMemcpyDeviceToHost);
  // 检查所有元素,并按 m 分组看差异(不同 m 用了不同 SF 行?)
  double mn=1e30,mx=-1e30; int bad=0;
  for (size_t i=0;i<out.size();++i){ double v=__half2float(out[i]); mn=fmin(mn,v); mx=fmax(mx,v); if (fabs(v-expect)>0.02*expect) bad++; }
  printf("实测 D: min=%.3f max=%.3f  超差点 %d/%zu  %s\n", mn,mx,bad,out.size(),
         bad==0?"✅ SF 落位正确(交错布局按 (m,16g,l) 写即可)":"❌ SF 关联性不对");
  printf("样例:"); for (int i=0;i<5;i++) printf(" %.2f",(float)__half2float(out[i])); printf("\n");
  return 0;
}
