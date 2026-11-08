// ============================================================================
// **owl 自有 sm120 FP4 MMA 探针 · 第二刀:缩放因子(SF)寄存器约定**
// ============================================================================
//
// 一可知:`.block_scale.scale_vec::4X` 下每个 lane 的 SF 操作数 = **一个 u32**
// (4 个 e4m3)+ `{bid,tid}`(CUTLASS 恒传 0)。未知的是:**这 4 个字节分别
// 管哪个 (row, k-block)**。
//
// 量法(黑盒差分):
//   1. A 数据只在**一个** k-block(16 宽)里为 1.0,其余为 0 ⇒ 输出只反映该 block;
//   2. 所有 SF = 1.0,只把 (mark_lane, mark_byte) 那个字节改成 4.0;
//   3. 扫 lane×byte,看哪些行的 D 变了 ⇒ 该字节的 (row, block) 归属。
//   4. 再对 4 个 k-block 各跑一遍 ⇒ (lane, byte) → (row, k-block) 全表。
//
// 编: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a ...
// ============================================================================
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

__device__ __forceinline__ void a_coord(int lane, int j, int& m, int& k) {
  const int t0 = lane & 3, t1 = lane >> 2;
  const int v0 = j & 7, v1 = (j >> 3) & 1, v2 = (j >> 4) & 1;
  m = t1 + 8 * v1;
  k = 8 * t0 + v0 + 32 * v2;
}
__device__ __forceinline__ void b_coord(int lane, int j, int& n, int& k) {
  const int t0 = lane & 3, t1 = lane >> 2;
  const int v0 = j & 7, v1 = (j >> 3) & 1;
  n = t1;
  k = 8 * t0 + v0 + 32 * v1;
}
// e2m1 编码:给定量化档 s∈[0,8) ⇒ ±{0,.5,1,1.5,2,3,4,6}
__host__ __device__ static inline uint8_t e2m1_enc(int s, bool neg) {
  return uint8_t((s & 0x7) | (neg ? 0x8 : 0));
}

__global__ void k_mark(const uint8_t* A, int lda, int kb,
                       int mark_lane, int mark_byte, int mark_b,
                       float* D) {
  const int lane = threadIdx.x & 31;
  uint32_t ra[4] = {0, 0, 0, 0};
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    uint32_t w = 0;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int j = r * 8 + b;
      int m, k;
      a_coord(lane, j, m, k);
      const bool in = (k >= 16 * kb && k < 16 * kb + 16);
      const uint8_t v = mark_b ? 0x3u : (in ? 0x3u : 0x0u);
      (void)A; (void)lda; (void)m;
      w |= uint32_t(v) << (4 * b);
    }
    ra[r] = w;
  }
  uint32_t rb[2] = {0, 0};
  #pragma unroll
  for (int r = 0; r < 2; ++r) {
    uint32_t w = 0;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
      const int j = r * 8 + b;
      int n, k; b_coord(lane, j, n, k);
      const bool in = (k >= 16 * kb && k < 16 * kb + 16);
      const uint8_t v = mark_b ? (in ? 0x3u : 0x0u) : 0x3u;
      (void)n;
      w |= uint32_t(v) << (4 * b);
    }
    rb[r] = w;
  }

  // SF:全 1.0(0x38);标记字节 = 4.0(0x40 >> 1? e4m3: 4.0 = 0x44)
  uint32_t sfa = 0x38383838u, sfb = 0x38383838u;
  if (lane == mark_lane) {
    uint32_t& t = mark_b ? sfb : sfa;
    t = (t & ~(0xFFu << (8 * mark_byte))) | (0x44u << (8 * mark_byte));
  }
  float d0, d1, d2, d3;
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
  const int group = lane >> 2, tid = lane & 3;
  D[(group + 0) * 8 + 2 * tid + 0] = d0;
  D[(group + 0) * 8 + 2 * tid + 1] = d1;
  D[(group + 8) * 8 + 2 * tid + 0] = d2;
  D[(group + 8) * 8 + 2 * tid + 1] = d3;
}

int main(int argc, char** argv) {
  const int mark_b = (argc > 1 && argv[1][0] == 'b') ? 1 : 0;
  printf("### %s SF 测量\n", mark_b ? "B" : "A");
  float* dD;
  cudaMalloc(&dD, 16 * 8 * 4);
  std::vector<float> D(16 * 8);
  // 基线:无标记 ⇒ 每个 (m,n) 应等于"该 block 的 16 个 1.0 相加"=16
  for (int kb = 0; kb < 4; ++kb) {
    printf("== k-block %d(数据只在 k∈[%d,%d) 为 1.0,SF 全 1.0)\n", kb, 16 * kb, 16 * kb + 16);
    cudaMemset(dD, 0, 16 * 8 * 4);
    k_mark<<<1, 32>>>(nullptr, 0, kb, -1, 0, mark_b, dD);
    cudaDeviceSynchronize();
    cudaMemcpy(D.data(), dD, 16 * 8 * 4, cudaMemcpyDeviceToHost);
    const float base = D[0];
    printf("   未标记基线 D[0][0]=%.1f D[8][0]=%.1f\n", D[0], D[8 * 8]);
    // 扫 (lane, byte)
    for (int lane = 0; lane < 32; ++lane) {
      for (int byte = 0; byte < 4; ++byte) {
        cudaMemset(dD, 0, 16 * 8 * 4);
        k_mark<<<1, 32>>>(nullptr, 0, kb, lane, byte, mark_b, dD);
        cudaDeviceSynchronize();
        cudaMemcpy(D.data(), dD, 16 * 8 * 4, cudaMemcpyDeviceToHost);
        // 找被影响的 (行, 列)
        char rows[64]; int n = 0; int cols_seen[8] = {0};
        for (int m = 0; m < 16; ++m)
          for (int c = 0; c < 8; ++c)
            if (D[m * 8 + c] > base + 0.5f) { cols_seen[c] = 1; if (c == 0) rows[n++] = char(m); }
        int any = 0; for (int c = 0; c < 8; ++c) any |= cols_seen[c];
        if (any) {
          printf("   lane=%2d byte=%d -> 列", lane, byte);
          for (int c = 0; c < 8; ++c) if (cols_seen[c]) printf(" %d", c);
          printf("  | 行数 %d", n);
          printf("  (D=%.0f/%.0f = ×%.2f)\n", D[0], base, D[0] / base);
        }
      }
    }
  }
  return 0;
}
