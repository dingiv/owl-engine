#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>
#include <cute/arch/mma_sm120.hpp>
using Atom = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cute::float_e2m1_t, cute::float_e2m1_t, float, cute::float_ue4m3_t, 16>;

// e4m3: 1.0=0x38, 2.0=0x40, 4.0=0x48, 8.0=0x50  → 用 lane%4 区分
__device__ __forceinline__ uint8_t enc(int i) { const uint8_t t[4]={0x38,0x40,0x48,0x50}; return t[i&3]; }

__global__ void calib_any(float* D, int which /*0=A,1=B*/) {
  const int lane = threadIdx.x & 31;
  const uint32_t ones = 0x22222222u;
  // 只变 LSB 字节(= k 组 0),其余 3 组固定 1.0
  uint32_t v = 0x38383800u | (uint32_t)enc(lane % 4);
  uint32_t sfa = which == 0 ? v : 0x38383838u;
  uint32_t sfb = which == 1 ? v : 0x38383838u;
  float c0=0,c1=0,c2=0,c3=0;
  Atom::fma(c0,c1,c2,c3, ones,ones,ones,ones, ones,ones, c0,c1,c2,c3, sfa, sfb);
  const int m = lane / 4, n = 2 * (lane % 4);
  D[(m+0)*8 + n + 0] = c0; D[(m+0)*8 + n + 1] = c1;
  D[(m+8)*8 + n + 0] = c2; D[(m+8)*8 + n + 1] = c3;
}

int main() {
  float* d; cudaMallocManaged(&d, 16*8*sizeof(float));
  const float val[4] = {1.f,2.f,4.f,8.f};
  for (int which=0; which<2; ++which) {
    for (int i=0;i<16*8;i++) d[i]=-1.f;
    calib_any<<<1,32>>>(d, which);
    cudaDeviceSynchronize();
    printf("\n== 按 lane%%4 变 %s 的 k 组 0 缩放 ==\n", which==0 ? "A(sfa)" : "B(sfb)");
    if (which==0) {
      printf("  m:  "); for (int m=0;m<16;m++) printf("%4d", m); printf("\n");
      printf("  sf: "); for (int m=0;m<16;m++) { float s = d[m*8+0]/16.f - 3.f; printf("%4.0f", s); } printf("\n");
      printf("  ⇒ 每个 m 用的是哪个 lane 的 sfa\n");
    } else {
      printf("  n:  "); for (int n=0;n<8;n++) printf("%4d", n); printf("\n");
      printf("  sf: "); for (int n=0;n<8;n++) { float s = d[0*8+n]/16.f - 3.f; printf("%4.0f", s); } printf("\n");
    }
  }
  return 0;
}
