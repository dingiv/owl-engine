// NVFP4 GEMM v1(手写;只用 cute 的块缩放原子):packer + kernel + CPU 参考对拍
// 布局: A/B 权重按 [rows][K/2] 字节(k 连续,2 个 e2m1/字节);SF 按 [rows][K/16] e4m3。
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <cuda_runtime.h>
#include <cute/arch/mma_sm120.hpp>

using Atom = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<
    cute::float_e2m1_t, cute::float_e2m1_t, float, cute::float_ue4m3_t, 16>;

// ---------- e2m1 / e4m3 编解码(host;K=16 块) ----------
static const float E2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
static inline uint8_t e2m1_encode(float x) {
  int sign = x < 0 ? 8 : 0; float a = fabsf(x);
  int best = 0; float bd = 1e30f;
  for (int i = 0; i < 8; ++i) { float d = fabsf(a - E2M1[i]); if (d < bd) { bd = d; best = i; } }
  return (uint8_t)(sign | best);
}
static inline float e2m1_decode(uint8_t v) { return E2M1[v & 7] * ((v & 8) ? -1.f : 1.f); }
static inline uint8_t e4m3_encode(float x) {  // 只覆盖我们的缩放范围(正值,2^-9..448)
  int e = 0; float a = x;
  if (a < 0.001953125f) a = 0.001953125f;          // 最小次正规 2^-9
  if (a > 448.f) a = 448.f;
  // 找最接近的 e4m3(3 位尾数)
  int best = 0; float bd = 1e30f;
  for (int code = 1; code < 127; ++code) {
    int ex = (code >> 3) & 0xF; int mant = code & 7;
    float v = (ex == 0) ? (mant / 8.f) * powf(2.f, -6.f) : (1.f + mant / 8.f) * powf(2.f, ex - 7.f);
    float d = fabsf(v - a); if (d < bd) { bd = d; best = code; }
  }
  (void)e; return (uint8_t)best;
}
static inline float e4m3_decode(uint8_t c) {
  int ex = (c >> 3) & 0xF; int mant = c & 7;
  return (ex == 0) ? (mant / 8.f) * powf(2.f, -6.f) : (1.f + mant / 8.f) * powf(2.f, ex - 7.f);
}

// ---------- 权重/激活打包 ----------
struct Packed { std::vector<uint8_t> q; std::vector<uint8_t> sf; float gscale; int rows, k; };
// RTN: 每 16 元素一个 e4m3 缩放 + 全局 fp32 缩放(NVFP4 语义)
static Packed pack(const std::vector<float>& w, int rows, int K) {
  Packed p; p.rows = rows; p.k = K; p.q.assign((size_t)rows * (K / 2), 0); p.sf.assign((size_t)rows * (K / 16), 0);
  float amax = 0.f;
  for (float v : w) amax = fmaxf(amax, fabsf(v));
  p.gscale = amax > 0 ? amax / (6.f * 448.f) : 1.f;
  for (int r = 0; r < rows; ++r) {
    for (int kb = 0; kb < K / 16; ++kb) {
      float mx = 0.f;
      for (int i = 0; i < 16; ++i) mx = fmaxf(mx, fabsf(w[(size_t)r * K + kb * 16 + i]));
      float s = (mx > 0) ? mx / (6.f * p.gscale) : 1.f;   // 归一化到 (0,1] ⇒ 全局缩放到 (0,6]
      uint8_t sc = e4m3_encode(s);
      p.sf[(size_t)r * (K / 16) + kb] = sc;
      float sdec = e4m3_decode(sc) * p.gscale;
      for (int i = 0; i < 16; ++i) {
        float v = w[(size_t)r * K + kb * 16 + i] / (sdec > 0 ? sdec : 1.f);
        uint8_t qq = e2m1_encode(v);
        size_t idx = (size_t)r * (K / 2) + (kb * 16 + i) / 2;
        if (((kb * 16 + i) & 1) == 0) p.q[idx] |= qq; else p.q[idx] |= (uint8_t)(qq << 4);
      }
    }
  }
  return p;
}

// ---------- kernel v1:BM=64(warp 4 个 m-tile)/ BN=64 / 128 线程 ----------
__global__ void nvfp4_gemm(const uint8_t* __restrict__ Aq, const uint8_t* __restrict__ Asf, float ga,
                           const uint8_t* __restrict__ Bq, const uint8_t* __restrict__ Bsf, float gb,
                           __half* __restrict__ D, int M, int N, int K) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int n0 = blockIdx.x * 64;          // BN
  const int m0 = blockIdx.y * 64 + warp * 16;
  if (m0 >= M) return;
  float acc[8][4];                          // 8 个 n-tile × 4 个 C 元素
  for (int j = 0; j < 8; ++j) for (int i = 0; i < 4; ++i) acc[j][i] = 0.f;

  const int tm = lane >> 2;                 // 线程在 16 行里的两半
  const int kq = (lane & 3) * 8;            // 该线程的 k 起点(0/8/16/24)
  const int nrow = lane >> 2;
  const uint32_t* Aq32 = reinterpret_cast<const uint32_t*>(Aq);
  const uint32_t* Bq32 = reinterpret_cast<const uint32_t*>(Bq);
  const int a_row0 = m0 + tm, a_row1 = m0 + tm + 8;
  const int sfa_m = m0 + 8 * (lane & 1) + tm;
  for (int kb = 0; kb < K / 64; ++kb) {
    const int k = kb * 64;
    uint32_t a0 = Aq32[(size_t)a_row0 * (K / 8) + (k + kq) / 8];
    uint32_t a1 = Aq32[(size_t)a_row1 * (K / 8) + (k + kq) / 8];
    uint32_t a2 = Aq32[(size_t)a_row0 * (K / 8) + (k + kq + 32) / 8];
    uint32_t a3 = Aq32[(size_t)a_row1 * (K / 8) + (k + kq + 32) / 8];
    const uint8_t* sfa_p = Asf + (size_t)sfa_m * (K / 16) + kb * 4;
    uint32_t sfa; memcpy(&sfa, sfa_p, 4);
    for (int j = 0; j < 8; ++j) {
      const int b_row = n0 + j * 8 + nrow;
      uint32_t b0 = Bq32[(size_t)b_row * (K / 8) + (k + kq) / 8];
      uint32_t b1 = Bq32[(size_t)b_row * (K / 8) + (k + kq + 32) / 8];
      const uint8_t* sfb_p = Bsf + (size_t)b_row * (K / 16) + kb * 4;
      uint32_t sfb; memcpy(&sfb, sfb_p, 4);
      float c0 = acc[j][0], c1 = acc[j][1], c2 = acc[j][2], c3 = acc[j][3];
      Atom::fma(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, sfb);
      acc[j][0] = c0; acc[j][1] = c1; acc[j][2] = c2; acc[j][3] = c3;
    }
  }
  const float sc = ga * gb;
  const int nm = 2 * (lane & 3);
  for (int j = 0; j < 8; ++j) {
    const int col = n0 + j * 8 + nm;
    if (col + 1 < N) {
      D[(size_t)(a_row0) * N + col + 0] = __float2half(acc[j][0] * sc);
      D[(size_t)(a_row0) * N + col + 1] = __float2half(acc[j][1] * sc);
      D[(size_t)(a_row1) * N + col + 0] = __float2half(acc[j][2] * sc);
      D[(size_t)(a_row1) * N + col + 1] = __float2half(acc[j][3] * sc);
    }
  }
}


// ---------- kernel v2:smem 分块(同步拷贝)+ 同款 warp 划分 ----------
// 块 = 64 行 × 64 列;smem 存 A[64][32B] / B[64][32B] / SF[64][4]×2 ⇒ 4KB + 512B
__global__ void nvfp4_gemm_v2(const uint8_t* __restrict__ Aq, const uint8_t* __restrict__ Asf, float ga,
                              const uint8_t* __restrict__ Bq, const uint8_t* __restrict__ Bsf, float gb,
                              __half* __restrict__ D, int M, int N, int K) {
  __shared__ uint8_t sA[64][32];    // 64 行 × 32B(= 64 个 k 的 fp4)
  __shared__ uint8_t sB[64][32];
  __shared__ uint8_t sSFA[64][4];
  __shared__ uint8_t sSFB[64][4];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n0 = blockIdx.x * 64, m0 = blockIdx.y * 64;
  const int tm = lane >> 2, kq = (lane & 3) * 8, nrow = lane >> 2;
  float acc[8][4];
  for (int j = 0; j < 8; ++j) for (int i = 0; i < 4; ++i) acc[j][i] = 0.f;
  const uint32_t* Aq32 = reinterpret_cast<const uint32_t*>(Aq);
  const uint32_t* Bq32 = reinterpret_cast<const uint32_t*>(Bq);

  for (int kb = 0; kb < K / 64; ++kb) {
    // ---- 载入:128 线程 × 1 行 × 32B(每线程一行的一半? 64 行 / 128 线程 ⇒ 每线程半行) ----
    // A: 64 行 × 32B;线程 tid<64 载 16B + tid>=64 载另 16B
    {
      int row = tid & 63, half = tid >> 6;         // 0/1
      int src = (m0 + row < M) ? (m0 + row) : (M - 1);
      uint4 v = *reinterpret_cast<const uint4*>(Aq + (size_t)src * (K / 2) + kb * 32 + half * 16);
      uint4* dst = reinterpret_cast<uint4*>(&sA[row][half * 16]);
      *dst = v;
    }
    {
      int row = tid & 63, half = tid >> 6;
      int src = (n0 + row < N) ? (n0 + row) : (N - 1);
      uint4 v = *reinterpret_cast<const uint4*>(Bq + (size_t)src * (K / 2) + kb * 32 + half * 16);
      uint4* dst = reinterpret_cast<uint4*>(&sB[row][half * 16]);
      *dst = v;
    }
    if (tid < 64) {
      memcpy(&sSFA[tid][0], Asf + (size_t)((m0 + tid < M) ? (m0 + tid) : (M - 1)) * (K / 16) + kb * 4, 4);
      memcpy(&sSFB[tid][0], Bsf + (size_t)((n0 + tid < N) ? (n0 + tid) : (N - 1)) * (K / 16) + kb * 4, 4);
    }
    __syncthreads();

    const int arow0 = warp * 16 + tm, arow1 = arow0 + 8;
    const int sfa_row = warp * 16 + 8 * (lane & 1) + tm;
    uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sA[arow0][kq / 2]);
    uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sA[arow1][kq / 2]);
    uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sA[arow0][(kq + 32) / 2]);
    uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sA[arow1][(kq + 32) / 2]);
    uint32_t sfa; memcpy(&sfa, &sSFA[sfa_row][0], 4);
    for (int j = 0; j < 8; ++j) {
      const int brow = j * 8 + nrow;
      uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sB[brow][kq / 2]);
      uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sB[brow][(kq + 32) / 2]);
      uint32_t sfb; memcpy(&sfb, &sSFB[brow][0], 4);
      float c0 = acc[j][0], c1 = acc[j][1], c2 = acc[j][2], c3 = acc[j][3];
      Atom::fma(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sfa, sfb);
      acc[j][0] = c0; acc[j][1] = c1; acc[j][2] = c2; acc[j][3] = c3;
    }
    __syncthreads();
  }
  const float sc = ga * gb;
  const int nm = 2 * (lane & 3);
  for (int j = 0; j < 8; ++j) {
    const int col = n0 + j * 8 + nm;
    const int r0 = m0 + warp * 16 + tm, r1 = r0 + 8;
    if (col + 1 < N && r1 < M) {
      D[(size_t)r0 * N + col + 0] = __float2half(acc[j][0] * sc);
      D[(size_t)r0 * N + col + 1] = __float2half(acc[j][1] * sc);
      D[(size_t)r1 * N + col + 0] = __float2half(acc[j][2] * sc);
      D[(size_t)r1 * N + col + 1] = __float2half(acc[j][3] * sc);
    }
  }
}

int main(int argc, char** argv) {
  int M = argc > 1 ? atoi(argv[1]) : 64;
  int N = argc > 2 ? atoi(argv[2]) : 64;
  int K = argc > 3 ? atoi(argv[3]) : 128;
  std::mt19937 rng(42); std::normal_distribution<float> nd(0.f, 1.f);
  std::vector<float> a((size_t)M * K), b((size_t)N * K);
  for (auto& v : a) v = nd(rng); for (auto& v : b) v = nd(rng);
  Packed pa = pack(a, M, K), pb = pack(b, N, K);

  // CPU 参考:用**量化后**的 fp4 值做 f32 GEMM(只验 kernel,不含量化误差)
  std::vector<float> ref((size_t)M * N, 0.f);
  auto deq = [&](const Packed& p, int r, int k) {
    uint8_t byte = p.q[(size_t)r * (p.k / 2) + k / 2];
    uint8_t nib = (k & 1) ? (byte >> 4) : (byte & 0xF);
    return e2m1_decode(nib) * e4m3_decode(p.sf[(size_t)r * (p.k / 16) + k / 16]) * p.gscale;
  };
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    float s = 0; for (int k = 0; k < K; ++k) s += deq(pa, m, k) * deq(pb, n, k);
    ref[(size_t)m * N + n] = s;
  }

  uint8_t *dAq, *dAsf, *dBq, *dBsf; __half* dD;
  cudaMalloc(&dAq, pa.q.size()); cudaMalloc(&dAsf, pa.sf.size());
  cudaMalloc(&dBq, pb.q.size()); cudaMalloc(&dBsf, pb.sf.size());
  cudaMalloc(&dD, (size_t)M * N * sizeof(__half));
  cudaMemcpy(dAq, pa.q.data(), pa.q.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dAsf, pa.sf.data(), pa.sf.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dBq, pb.q.data(), pb.q.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dBsf, pb.sf.data(), pb.sf.size(), cudaMemcpyHostToDevice);
  dim3 grid((N + 63) / 64, (M + 63) / 64);
  nvfp4_gemm<<<grid, 128>>>(dAq, dAsf, pa.gscale, dBq, dBsf, pb.gscale, dD, M, N, K);
  auto err = cudaGetLastError();
  if (err != cudaSuccess) { printf("kernel error: %s\n", cudaGetErrorString(err)); return 1; }
  cudaDeviceSynchronize();
  std::vector<__half> out((size_t)M * N);
  cudaMemcpy(out.data(), dD, out.size() * sizeof(__half), cudaMemcpyDeviceToHost);

  double maxre = 0, maxabs = 0; double scale_ref = 0;
  for (size_t i = 0; i < out.size(); ++i) scale_ref = fmax(scale_ref, fabs(ref[i]));
  for (size_t i = 0; i < out.size(); ++i) {
    double d = fabs((double)__half2float(out[i]) - ref[i]);
    maxabs = fmax(maxabs, d); maxre = fmax(maxre, d / (scale_ref + 1e-9));
  }
  printf("M=%d N=%d K=%d  参考量级 max|ref|=%.1f  max|Δ|=%.3f  归一化 max|Δ|/max|ref|=%.2e  %s\n",
         M, N, K, scale_ref, maxabs, maxre, maxre < 2e-2 ? "✅ 通过" : "❌ 超差");

  // ---- v2 对拍 ----
  {
    cudaMemset(dD, 0, (size_t)M * N * sizeof(__half));
    nvfp4_gemm_v2<<<grid, 128>>>(dAq, dAsf, pa.gscale, dBq, dBsf, pb.gscale, dD, M, N, K);
    auto e = cudaGetLastError();
    if (e != cudaSuccess) { printf("v2 error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaDeviceSynchronize();
    std::vector<__half> o2((size_t)M * N);
    cudaMemcpy(o2.data(), dD, o2.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    double mx = 0; for (size_t i = 0; i < o2.size(); ++i) mx = fmax(mx, fabs((double)__half2float(o2[i]) - ref[i]));
    printf("  v2 对拍 max|Δ|=%.3f %s\n", mx, mx / (scale_ref + 1e-9) < 2e-2 ? "✅" : "❌");
  }
  // 计时(f16 输出)
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int iters = 50;
  nvfp4_gemm<<<grid, 128>>>(dAq, dAsf, pa.gscale, dBq, dBsf, pb.gscale, dD, M, N, K);
  cudaDeviceSynchronize();
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) nvfp4_gemm<<<grid, 128>>>(dAq, dAsf, pa.gscale, dBq, dBsf, pb.gscale, dD, M, N, K);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  float ms; cudaEventElapsedTime(&ms, e0, e1);
  double tflops = 2.0 * M * N * K * iters / (ms * 1e-3) / 1e12;
  printf("  手写 v1:%8.1f GFLOPS (%.3f ms/iter)\n", tflops * 1000, ms / iters);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) nvfp4_gemm_v2<<<grid, 128>>>(dAq, dAsf, pa.gscale, dBq, dBsf, pb.gscale, dD, M, N, K);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  cudaEventElapsedTime(&ms, e0, e1);
  printf("  手写 v2:%8.1f GFLOPS (%.3f ms/iter)\n", 2.0 * M * N * K * iters / (ms * 1e-3) / 1e12 * 1000, ms / iters);
  return 0;
}
