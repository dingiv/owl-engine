// ============================================================================
// **owl 自有 sm120 FP4 MMA 探针**(2026-10-29,“正向”第一刀的可验证地基)
// ============================================================================
//
// 目标:用**裸 PTX**(不引 CUTLASS/cute)跑一条
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//     .row.col.f32.e2m1.e2m1.f32.ue4m3
// 并与 host 参考(反量化后内积)对比。
//
// 为什么要它:CUTLASS 之所以砍不掉,就是这个 atom 的**片段映射**:
//   - A: (T32,V32) → (M16,K64)   `Stride<Stride<_128,_1>,Stride<_16,_8,_512>>`
//   - B: (T32,V16) → (N8,K64)    `Stride<Stride<_64,_1>,Stride<_8,_256>>`
//   - SF: 每 lane 一个 u32(4 个 e4m3)+ `{bidA,tidA}`(CUTLASS 恒传 0)
// 这里把它**实测**出来:数据片段与 SF 两段分开验(SF 先取全 1 隔离)。
//
// 编法(不需要 CUTLASS 头):
//   nvcc -O3 -std=c++17 -arch=sm_120a probe_fp4_mma.cu -o /tmp/probe_fp4 -lcudart
// ============================================================================
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

// ---------------------------------------------------------------- host 参考
// e2m1 解码(LUT;与硬件语义一致:0,±0.5,±1,±1.5,±2,±3,±4,±6)
__host__ __device__ static inline float e2m1_dec(uint8_t c) {
  static const float t[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const int mag = c & 0x7;
  return (c & 0x8) ? -t[mag] : t[mag];
}

// ---------------------------------------------------------------- 设备片段映射
// A: lane L 的 32 个 nibble(j = v0 + 8*v1 + 16*v2)取自 (m, k):
//    t0 = L%4, t1 = L/4
//    m = t1 + 8*v1
//    k = 8*t0 + v0 + 32*v2
__device__ __forceinline__ void a_coord(int lane, int j, int& m, int& k) {
  const int t0 = lane & 3, t1 = lane >> 2;
  const int v0 = j & 7, v1 = (j >> 3) & 1, v2 = (j >> 4) & 1;
  m = t1 + 8 * v1;
  k = 8 * t0 + v0 + 32 * v2;
}
// B: lane L 的 16 个 nibble(j = v0 + 8*v1)取自 (n, k):
//    n = t1
//    k = 8*t0 + v0 + 32*v1
__device__ __forceinline__ void b_coord(int lane, int j, int& n, int& k) {
  const int t0 = lane & 3, t1 = lane >> 2;
  const int v0 = j & 7, v1 = (j >> 3) & 1;
  n = t1;
  k = 8 * t0 + v0 + 32 * v1;
}

// 取 A[m][k](A 按 (16,64) 行主序、每字节两个 nibble,低 nibble = 偶数 k)
__device__ __forceinline__ uint8_t nib(const uint8_t* p, int row, int col, int stride_u8) {
  const uint8_t b = p[row * stride_u8 + (col >> 1)];
  return (col & 1) ? (b >> 4) : (b & 0xF);
}

/// 一个 warp 算 D[16][8] = A[16][64] · B[8][64]ᵀ(SF 全为 1.0)
__global__ void k_probe(const uint8_t* __restrict__ A, int lda,
                        const uint8_t* __restrict__ B, int ldb,
                        float* __restrict__ D) {
  const int lane = threadIdx.x & 31;

  // --- A 片段:4 个 u32(每 u32 = 8 个 nibble,nibble j 在 byte j/2 的低/高半) ---
  uint32_t ra[4] = {0, 0, 0, 0};
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    uint32_t w = 0;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {           // 一个 u32 装 8 个 nibble
      const int j = r * 8 + b;              // 该 lane 的第 j 个 nibble
      int m, k;
      a_coord(lane, j, m, k);
      w |= uint32_t(nib(A, m, k, lda)) << (4 * b);
    }
    ra[r] = w;
  }
  // --- B 片段:2 个 u32(共 16 个 nibble) ---
  uint32_t rb[2] = {0, 0};
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    uint32_t w = 0;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int j = r * 8 + b;
      int n, k;
      b_coord(lane, j, n, k);
      w |= uint32_t(nib(B, n, k, ldb)) << (4 * b);
    }
    rb[r] = w;
  }

  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  // SF 全 1.0(e4m3: 0x38);CUTLASS 对 bid/tid 恒传 0
  const uint32_t sfa = 0x38383838u, sfb = 0x38383838u;
  asm volatile(
    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
    "{%0,  %1,  %2,  %3},"
    "{%4,  %5,  %6,  %7},"
    "{%8,  %9},"
    "{%10, %11, %12, %13},"
    "{%14},"
    "{%15, %16},"
    "{%17},"
    "{%18, %19};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    :  "r"(ra[0]), "r"(ra[1]), "r"(ra[2]), "r"(ra[3]),
       "r"(rb[0]), "r"(rb[1]),
       "f"(0.f),   "f"(0.f),   "f"(0.f),   "f"(0.f),
       "r"(sfa),  "h"(uint16_t(0)), "h"(uint16_t(0)),
       "r"(sfb),  "h"(uint16_t(0)), "h"(uint16_t(0)));

  // C 片段(T32,V4) → (M16,N8),SM80_16x8_Row:lane 的 4 个值盖 (m = lane/4 + {0,8}, n = 2*(lane%4) + {0,1})
  const int group = lane >> 2, tid = lane & 3;
  D[(group + 0) * 8 + 2 * tid + 0] = d0;
  D[(group + 0) * 8 + 2 * tid + 1] = d1;
  D[(group + 8) * 8 + 2 * tid + 0] = d2;
  D[(group + 8) * 8 + 2 * tid + 1] = d3;
}

// ---------------------------------------------------------------- host
int main() {
  const int M = 16, K = 64, N = 8;
  std::vector<uint8_t> A(size_t(M) * K / 2), B(size_t(N) * K / 2);
  std::vector<float> Af(size_t(M) * K), Bf(size_t(N) * K);

  unsigned seed = 12345u;
  auto rnd01 = [&] { seed = seed * 1664525u + 1013904223u; return (seed >> 9) & 0x7; };
  auto put = [](std::vector<uint8_t>& p, int stride_u8, int r, int c, uint8_t v) {
    uint8_t& b = p[r * stride_u8 + (c >> 1)];
    if (c & 1) b = uint8_t((b & 0x0F) | (v << 4)); else b = uint8_t((b & 0xF0) | v);
  };
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) {
      const uint8_t v = uint8_t(rnd01() | ((seed & 1) ? 8 : 0));  // 含负值
      put(A, K / 2, m, k, v);
      Af[m * K + k] = e2m1_dec(v & 0xF);
    }
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      const uint8_t v = uint8_t(rnd01() | ((seed & 1) ? 8 : 0));
      put(B, K / 2, n, k, v);
      Bf[n * K + k] = e2m1_dec(v & 0xF);
    }

  uint8_t *dA, *dB; float* dD;
  cudaMalloc(&dA, A.size()); cudaMalloc(&dB, B.size()); cudaMalloc(&dD, M * N * 4);
  cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
  k_probe<<<1, 32>>>(dA, K / 2, dB, K / 2, dD);
  auto rc = cudaDeviceSynchronize();
  if (rc != cudaSuccess) { printf("kernel 失败: %s\n", cudaGetErrorString(rc)); return 1; }
  std::vector<float> D(M * N);
  cudaMemcpy(D.data(), dD, M * N * 4, cudaMemcpyDeviceToHost);

  float worst = 0.f; int bad = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += double(Af[m * K + k]) * double(Bf[n * K + k]);
      const float e = std::fabs(float(acc) - D[m * N + n]);
      if (e > worst) worst = e;
      if (e > 1e-3f) ++bad;
    }
  printf("D[0][0..3] = %.4f %.4f %.4f %.4f (host %.4f %.4f %.4f %.4f)\n",
         D[0], D[1], D[2], D[3],
         [&]{double s=0;for(int k=0;k<K;++k)s+=Af[k]*Bf[k];return float(s);}(),
         [&]{double s=0;for(int k=0;k<K;++k)s+=Af[k]*Bf[K+k];return float(s);}(),
         [&]{double s=0;for(int k=0;k<K;++k)s+=Af[k]*Bf[2*K+k];return float(s);}(),
         [&]{double s=0;for(int k=0;k<K;++k)s+=Af[k]*Bf[3*K+k];return float(s);}());
  printf("片段映射对照: max|err| = %.6g, 超门 %d/%d\n", worst, bad, M * N);
  return bad == 0 ? 0 : 2;
}
