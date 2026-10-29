// ============================================================================
// owl TP2 —— **写腿调优台**(2026-10-19;"小包优化"角度的分母)
//
// ## 立案
//
// 写腿 = AR 的 90%(§38.1:444µs / 494µs),它落在 **12.75 GB/s** =
// **Gen3 x16 的 81%**(15.75 GB/s 理论)。那 19% 缺口从哪来?两种解释先分开:
//
//   · **包开销(TLP)**:`MPS=128B` ⇒ 128/(128+24) = **84.2%**;`MPS=256B` ⇒ **91.4%**。
//     实测 D2H/H2D = 13.19/13.07 GB/s ≈ 128B 天花板的 99% ⇒ **host 那条路已顶满**。
//     **⇒ 写腿 12.75 距 MPS=128 的天花板只剩 4%。**
//   · **核/访问形态**:网格越大越慢(16 块 410µs → 680 块 543µs,§44.2②)。
//
// 本台测的是**在 MPS 不变的条件下,写腿还有没有软件余量**。
// (MPS 128→256 要 root 改 Device Control,属另一条线;见设计卷 §46。)
//
// ## 臂(全部同几何可比)
//
//   ① 存储策略 `st.global` / `__stcs` / `__stwt` / `__stcg`
//   ② 栅栏 `__threadfence_system()` 有 / 无
//   ③ 访问形态 线程跨步 / 每线程连续 64B
//   ④ 发射几何 grid 4 / 8 / 16 / 32 / 64 × block 512
//   ⑤ 对照 拷贝引擎(`cudaMemcpyPeerAsync`)
//
// 编译:`nvcc -O3 -arch=sm_86 -o /tmp/p2p-wrtune tools/p2p-wr-tune.cu`
// 运行:`/tmp/p2p-wrtune <ordA> <ordB> [bytes]`
// ⚠️ 卡序 = **CUDA ordinal**;跑前确认机器空闲 —— 实测别的卡跑满时本读数会腰斩(§44.3⑤)。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <unistd.h>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));  \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

#define BLOCK 512

enum Store { ST_DEF = 0, ST_CS = 1, ST_WT = 2, ST_CG = 3 };
enum Fence { FN_NONE = 0, FN_SC = 1, FN_ACQREL = 2, FN_RELST = 3 };

/// `fence.sc.sys` = 现役 `__threadfence_system()`(最重)
__device__ __forceinline__ void fence_sc_sys() { __threadfence_system(); }
/// `fence.acq_rel.sys` = 只保 release/acquire 语义(够我们的发布律用)
__device__ __forceinline__ void fence_acqrel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
/// LL 式:数据用 relaxed 写 + `__syncthreads()` + 单线程 `st.release.sys` 发布 flag
__device__ __forceinline__ void publish_release(unsigned int* __restrict__ flag, unsigned v) {
    asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(flag), "r"(v) : "memory");
}

template <int MODE, bool FENCE>
__global__ void k_wr(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n16) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
    const uint4 v = src[i];
    if (MODE == ST_DEF) dst[i] = v;
    else if (MODE == ST_CS) __stcs(&dst[i], v);
    else if (MODE == ST_WT) __stwt(&dst[i], v);
    else __stcg(&dst[i], v);
  }
  if (FENCE) __threadfence_system();
}

/// fence 档对比核:`FM` ∈ {0 无, 1 sc.sys, 2 acq_rel.sys, 3 release-store}
template <int FM>
__global__ void k_wr_fence(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n16,
                           unsigned int* __restrict__ flag) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride)
    dst[i] = src[i];
  if (FM == 1) { fence_sc_sys(); }
  else if (FM == 2) { fence_acqrel_sys(); }
  else if (FM == 3) {
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) publish_release(flag, 1u);
  }
}

template <bool FENCE>
__global__ void k_wr_contig(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n16) {
  const size_t nthreads = (size_t)gridDim.x * blockDim.x;
  const size_t per = 4;  // 每线程连续 4×16B = 64B
  const size_t chunk = n16 / per;
  for (size_t c = (size_t)blockIdx.x * blockDim.x + threadIdx.x; c < chunk; c += nthreads) {
    const size_t base = c * per;
    const uint4 v0 = src[base], v1 = src[base + 1], v2 = src[base + 2], v3 = src[base + 3];
    dst[base] = v0; dst[base + 1] = v1; dst[base + 2] = v2; dst[base + 3] = v3;
  }
  if (FENCE) __threadfence_system();
}

static cudaEvent_t g_e0, g_e1;

// 跑一条臂:3 次预热 + k 次计时,返回每次迭代的设备时(µs)
template <typename L>
static double runarm(const char* name, L launch, size_t bytes, int k) {
  for (int w = 0; w < 3; ++w) launch();
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(g_e0));
  for (int t = 0; t < k; ++t) launch();
  CK(cudaEventRecord(g_e1));
  CK(cudaEventSynchronize(g_e1));
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, g_e0, g_e1));
  const double us = (double)ms * 1e3 / k;
  printf("  %-34s %8.2f us %7.2f GB/s\n", name, us, (double)bytes / us / 1e3);
  return us;
}

template <int MODE, bool FENCE>
static double arm(const char* name, uint4* d, const uint4* s, size_t n16, int grid, int k,
                  size_t bytes) {
  return runarm(name, [&] { k_wr<MODE, FENCE><<<grid, BLOCK>>>(d, s, n16); }, bytes, k);
}

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "用法: %s <ordA> <ordB> [bytes]\n", argv[0]); return 2; }
  const int oa = atoi(argv[1]), ob = atoi(argv[2]);
  const size_t bytes = argc > 3 ? (size_t)atoll(argv[3]) : 5242880u;
  const size_t n16 = bytes / 16;
  const int k = getenv("WR_K") ? atoi(getenv("WR_K")) : 64;

  int can = 0;
  CK(cudaDeviceCanAccessPeer(&can, oa, ob));
  if (!can) { fprintf(stderr, "无 P2P\n"); return 3; }
  CK(cudaSetDevice(oa)); CK(cudaDeviceEnablePeerAccess(ob, 0));
  CK(cudaSetDevice(ob)); CK(cudaDeviceEnablePeerAccess(oa, 0));

  void *pb = nullptr, *sa = nullptr;
  CK(cudaSetDevice(ob)); CK(cudaMalloc(&pb, bytes)); CK(cudaMemset(pb, 0, bytes));
  CK(cudaSetDevice(oa)); CK(cudaMalloc(&sa, bytes)); CK(cudaMemset(sa, 0x5a, bytes));
  CK(cudaEventCreate(&g_e0)); CK(cudaEventCreate(&g_e1));

  printf("[wrtune] A=ord%d B=ord%d payload=%.2f MB K=%d\n", oa, ob, (double)bytes / 1048576.0, k);

  arm<ST_DEF, true>("st.global  + fence  (基准)", (uint4*)pb, (uint4*)sa, n16, 16, k, bytes);
  arm<ST_DEF, false>("st.global  - fence", (uint4*)pb, (uint4*)sa, n16, 16, k, bytes);
  arm<ST_CS, true>("__stcs     + fence", (uint4*)pb, (uint4*)sa, n16, 16, k, bytes);
  arm<ST_WT, true>("__stwt     + fence", (uint4*)pb, (uint4*)sa, n16, 16, k, bytes);
  arm<ST_CG, true>("__stcg     + fence", (uint4*)pb, (uint4*)sa, n16, 16, k, bytes);
  runarm("连续 64B/线程 + fence",
         [&] { k_wr_contig<true><<<16, BLOCK>>>((uint4*)pb, (uint4*)sa, n16); }, bytes, k);

  arm<ST_DEF, true>("st.global  + fence  grid=4", (uint4*)pb, (uint4*)sa, n16, 4, k, bytes);
  arm<ST_DEF, true>("st.global  + fence  grid=8", (uint4*)pb, (uint4*)sa, n16, 8, k, bytes);
  arm<ST_DEF, true>("st.global  + fence  grid=32", (uint4*)pb, (uint4*)sa, n16, 32, k, bytes);
  arm<ST_DEF, true>("st.global  + fence  grid=64", (uint4*)pb, (uint4*)sa, n16, 64, k, bytes);

  {  // 拷贝引擎对照(SM 零占用)
    cudaStream_t s;
    CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    runarm("cudaMemcpyPeerAsync(CE)",
           [&] { CK(cudaMemcpyPeerAsync(pb, ob, sa, oa, bytes, s)); }, bytes, k);
    CK(cudaStreamDestroy(s));
  }

  // ⑦ ★ **栅栏档扫描**(解码侧的 α 账;§50.2 已量出 10KB 档栅栏占写腿 38%)
  {
    unsigned int* flag = nullptr;
    CK(cudaMalloc(&flag, 4)); CK(cudaMemset(flag, 0, 4));
    printf("  ── 栅栏档(每档 K=%d)──\n", k);
    runarm("无栅栏", [&]{ k_wr_fence<0><<<4, BLOCK>>>((uint4*)pb,(uint4*)sa,n16,flag); }, bytes, k);
    runarm("fence.sc.sys(现役)", [&]{ k_wr_fence<1><<<4, BLOCK>>>((uint4*)pb,(uint4*)sa,n16,flag); }, bytes, k);
    runarm("fence.acq_rel.sys", [&]{ k_wr_fence<2><<<4, BLOCK>>>((uint4*)pb,(uint4*)sa,n16,flag); }, bytes, k);
    runarm("st.release.sys(LL 式)", [&]{ k_wr_fence<3><<<4, BLOCK>>>((uint4*)pb,(uint4*)sa,n16,flag); }, bytes, k);
    CK(cudaFree(flag));
  }

  // ⑥ ★ **诞生间隙假设**:生产里每个 AR 之间隔着 ~4ms 计算 ⇒ 链路可能降功耗;
  //    §38.1 的 444µs(生产)vs 本台 411µs(背靠背)那 33µs 会不会就是"每次唤醒税"?
  //    做法:单次写之间夹 host sleep,分别在 0 / 200µs / 2ms / 10ms 间隙下量**单次**写时。
  {
    cudaStream_t ss;
    CK(cudaStreamCreateWithFlags(&ss, cudaStreamNonBlocking));
    const int gap_us[4] = {0, 200, 2000, 10000};
    for (int gi = 0; gi < 4; ++gi) {
      for (int w = 0; w < 3; ++w) { k_wr<ST_DEF, true><<<4, BLOCK, 0, ss>>>((uint4*)pb, (uint4*)sa, n16); }
      CK(cudaStreamSynchronize(ss));
      double tot = 0.0;
      const int n = (gap_us[gi] == 0) ? 64 : 16;
      for (int t = 0; t < n; ++t) {
        if (gap_us[gi] > 0) { CK(cudaDeviceSynchronize()); usleep(gap_us[gi]); }
        CK(cudaEventRecord(g_e0, ss));
        k_wr<ST_DEF, true><<<4, BLOCK, 0, ss>>>((uint4*)pb, (uint4*)sa, n16);
        CK(cudaEventRecord(g_e1, ss));
        CK(cudaEventSynchronize(g_e1));
        float ms = 0.f; CK(cudaEventElapsedTime(&ms, g_e0, g_e1));
        tot += (double)ms * 1e3;
      }
      const double us = tot / n;
      char nm[64];
      if (gap_us[gi]) snprintf(nm, sizeof(nm), "单次写(间隙 %dus)", gap_us[gi]);
      else snprintf(nm, sizeof(nm), "单次写(背靠背)");
      printf("  %-34s %8.2f us %7.2f GB/s\n", nm, us, (double)bytes / us / 1e3);
    }
    CK(cudaStreamDestroy(ss));
  }

  {
    uint4 h = {};
    CK(cudaMemcpy(&h, pb, sizeof(h), cudaMemcpyDeviceToHost));
    printf("  [verify] 对端首字 = 0x%08x(期望 0x5a5a5a5a)\n", h.x);
  }
  return 0;
}
