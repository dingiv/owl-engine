// ★ **RMW 隐藏可行性探针**(2026-10-18;TP2 施工设计 §38)
//
// ## 立案
//
// push AR 的账已拆开(`ar_channel_bench` 的 `pS`/`pR` 探针;3080 对,f16,5.24MB):
//
// | 笔 | 设备时 |
// |---|---:|
// | 写腿(pS,双边)| **444 µs** |
// | 本地 RMW(pR)| **44.6 µs** |
// | 两核串联 + 握手 | **5.3 µs** |
// | = push AR | **494 µs** |
//
// ⇒ **协议税≈0**(融合两核只值 5µs);**唯一可挖的是那 44.6µs 的本地 RMW**。
// 两条候选路,本程序就是它们的分母:
//
// | 路 | 做法 | 判据 |
// |---|---|---|
// | **A. 双流重叠** | 写对端(流1)‖ 本地 RMW(流2) | `max(444, 44.6)=444` ⇒ 可藏;`444+44.6=489` ⇒ 不可藏 |
// | **B. 远端 reduce-add** | 用 `red.global.add` **直接累加进对端 payload** ⇒ **本地 RMW 整条消失** | 远端 atomic-add 带宽 ≈ 纯写 ⇒ 可用 |
//
// 路 A 的先验不利:§33 实测"**图内跨流零重叠**"(AR + add 链,aux ≡ same);
// 但 §33.3 自己指出那条判据**无效**(add 链是 L2 带宽饱和的)⇒ 本程序用
// **真实形状**(写 5.24MB ‖ 本地 RMW 5.24MB)重测。
//
// ## 编译 / 运行
//
//   nvcc -O3 -arch=sm_86 -o /tmp/p2p-rmw-probe tools/p2p-rmw-probe.cu
//   /tmp/p2p-rmw-probe <ordA> <ordB> [bytes]

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__,                     \
              cudaGetErrorString(e_));                                                 \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

#define GRID 16
#define BLOCK 512
#define VEC 8

// 写腿:把本端常量写进对端(等价于 push 的 stage 数据面)
__global__ void k_write_peer(uint4* __restrict__ dst, size_t n16, unsigned seed) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const uint4 v = make_uint4(seed, seed ^ 0x5a5a5au, seed + 0x1234u, seed ^ 0xdeadbeefu);
  for (size_t i = tid; i < n16; i += (size_t)VEC * stride) {
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      const size_t j = i + (size_t)k * stride;
      if (j < n16) dst[j] = v;
    }
  }
}

// ★ **读本地源 → 写对端**(= owl stage 核的数据面;与 `k_write_peer` 的唯一差别
// 是"源是真读,不是常量")—— 用来分辨"PCIe 写把本地读也挡住"这个假设。
__global__ void k_read_write_peer(uint4* __restrict__ dst, const uint4* __restrict__ src,
                                  size_t n16) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n16; i += (size_t)VEC * stride) {
    uint4 v[VEC];
    bool ok[VEC];
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      const size_t j = i + (size_t)k * stride;
      ok[k] = j < n16;
      if (ok[k]) v[k] = src[j];
    }
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      const size_t j = i + (size_t)k * stride;
      if (ok[k]) dst[j] = v[k];
    }
  }
}

// 本地 RMW(等价于 push 的 reduce 数据面:payload += inbox,纯本地)
__global__ void k_rmw_local(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n16) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n16; i += (size_t)VEC * stride) {
    uint4 v[VEC];
    bool ok[VEC];
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      const size_t j = i + (size_t)k * stride;
      ok[k] = j < n16;
      if (ok[k]) v[k] = src[j];
    }
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      if (ok[k]) {
        const size_t j = i + (size_t)k * stride;
        uint4 t = dst[j];
        t.x += v[k].x; t.y += v[k].y; t.z += v[k].z; t.w += v[k].w;
        dst[j] = t;
      }
    }
  }
}

// ★ **远端 reduce-add**:`red.global.add.f16x2` 直接累加进对端 payload
// (fire-and-forget ⇒ posted;返回值不用 ⇒ nvcc 出 `red` 而非 `atom`)。
__global__ void k_red_add_peer(__half2* __restrict__ dst, size_t n2, __half2 val) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = tid; i < n2; i += (size_t)VEC * stride) {
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      const size_t j = i + (size_t)k * stride;
      if (j < n2) atomicAdd(&dst[j], val);
    }
  }
}

/// 写 `[start,end]` 全局纳秒时间戳(自旋 `ns` 纳秒)—— 图内并行存在性证明用
__global__ void k_stamp(unsigned long long* __restrict__ out, int id, unsigned long long ns) {
  if (threadIdx.x != 0) return;
  unsigned long long t0, t1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
  unsigned long long t = t0;
  while (t - t0 < ns) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
  out[id * 2 + 0] = t0;
  out[id * 2 + 1] = t1;
}

static double ms_between(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, a, b));
  return (double)ms;
}

static void report(const char* tag, size_t bytes, int k, double ms) {
  const double us = ms * 1e3 / (double)k;
  printf("  %-22s %9.2f us %8.2f GB/s\n", tag, us, (double)bytes / (us * 1e-6) / 1e9);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "用法: %s <ordA> <ordB> [bytes]\n", argv[0]);
    return 2;
  }
  const int oa = atoi(argv[1]), ob = atoi(argv[2]);
  const size_t bytes = argc > 3 ? (size_t)atoll(argv[3]) : 5242880u;
  const int k = getenv("RMW_K") ? atoi(getenv("RMW_K")) : 200;
  const size_t n16 = bytes / 16, n2 = bytes / 4;

  int can = 0;
  CK(cudaDeviceCanAccessPeer(&can, oa, ob));
  if (!can) { fprintf(stderr, "[rmw] 无 P2P\n"); return 3; }
  CK(cudaSetDevice(oa)); CK(cudaDeviceEnablePeerAccess(ob, 0));
  CK(cudaSetDevice(ob)); CK(cudaDeviceEnablePeerAccess(oa, 0));

  void *peer_b = nullptr, *local_a = nullptr, *src_a = nullptr;
  CK(cudaSetDevice(ob)); CK(cudaMalloc(&peer_b, bytes)); CK(cudaMemset(peer_b, 0, bytes));
  CK(cudaSetDevice(oa)); CK(cudaMalloc(&local_a, bytes)); CK(cudaMemset(local_a, 0, bytes));
  CK(cudaMalloc(&src_a, bytes)); CK(cudaMemset(src_a, 1, bytes));

  cudaStream_t s1, s2;
  // ⚠️ **non-blocking**(复刻 owl `ctx.new_stream()` 的 cudarc 语义):默认(blocking)
  // 流与 legacy default stream 有隐式同步 ⇒ 捕获行为不同。
  CK(cudaSetDevice(oa));
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

  printf("[rmw] A=ord%d B=ord%d payload=%zu B (%.2f MB) K=%d grid=%d×%d\n", oa, ob, bytes,
         (double)bytes / 1048576.0, k, GRID, BLOCK);

  // ── ① 写腿(单边;= pS 的数据面)────────────────────────────────
  for (int w = 0; w < 3; ++w) k_write_peer<<<GRID, BLOCK, 0, s1>>>((uint4*)peer_b, n16, 0x1234u);
  CK(cudaStreamSynchronize(s1));
  CK(cudaEventRecord(e0, s1));
  for (int t = 0; t < k; ++t) k_write_peer<<<GRID, BLOCK, 0, s1>>>((uint4*)peer_b, n16, 0x1234u);
  CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
  report("W  (写对端,单边)", bytes, k, ms_between(e0, e1));

  // ── ② 本地 RMW(单边;= pR 的数据面)────────────────────────────
  CK(cudaEventRecord(e0, s1));
  for (int t = 0; t < k; ++t) k_rmw_local<<<GRID, BLOCK, 0, s1>>>((uint4*)local_a, (const uint4*)src_a, n16);
  CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
  const double rmw = ms_between(e0, e1);
  report("RMW(本地)", bytes, k, rmw);

  // ── ③ **串行**两笔(对照基线)──────────────────────────────────
  {
    CK(cudaEventRecord(e0, s1));
    for (int t = 0; t < k; ++t) {
      k_write_peer<<<GRID, BLOCK, 0, s1>>>((uint4*)peer_b, n16, 0x1234u);
      k_rmw_local<<<GRID, BLOCK, 0, s1>>>((uint4*)local_a, (const uint4*)src_a, n16);
    }
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    report("串行(W → RMW)", bytes, k, ms_between(e0, e1));
  }

  // ── ④ ★ **双流重叠**(W 在 s1 ‖ RMW 在 s2)──────────────────────
  {
    CK(cudaStreamSynchronize(s1)); CK(cudaStreamSynchronize(s2));
    CK(cudaEventRecord(e0, s1));
    for (int t = 0; t < k; ++t) {
      k_write_peer<<<GRID, BLOCK, 0, s1>>>((uint4*)peer_b, n16, 0x1234u);
      k_rmw_local<<<GRID, BLOCK, 0, s2>>>((uint4*)local_a, (const uint4*)src_a, n16);
    }
    CK(cudaEventRecord(e1, s1));
    CK(cudaStreamSynchronize(s2));
    CK(cudaEventSynchronize(e1));
    report("双流(W ‖ RMW)", bytes, k, ms_between(e0, e1));
  }

  // ── ⑤ ★★ **远端 reduce-add**(路 B:直接累加进对端 ⇒ 本地 RMW 消失)──
  {
    __half2 v = __float2half2_rn(1.0f);
    for (int w = 0; w < 3; ++w) k_red_add_peer<<<GRID, BLOCK, 0, s1>>>((__half2*)peer_b, n2, v);
    CK(cudaStreamSynchronize(s1));
    CK(cudaEventRecord(e0, s1));
    for (int t = 0; t < k; ++t) k_red_add_peer<<<GRID, BLOCK, 0, s1>>>((__half2*)peer_b, n2, v);
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    report("RED-ADD 对端(f16x2)", bytes, k, ms_between(e0, e1));
    // 落地校验:对端首元素应 = K+3 次累加(3 次预热 + k 次)
    __half2 h;
    CK(cudaSetDevice(ob));
    CK(cudaMemcpy(&h, peer_b, 4, cudaMemcpyDeviceToHost));
    const float got = __half2float(__low2half(h));
    const float want = (float)(k + 3);
    printf("  %-22s %s(got=%.0f want=%.0f)\n", "RED-ADD 落地校验",
           (got == want) ? "✓" : "✗", got, want);
    CK(cudaSetDevice(oa));
  }

  // ── ⑥ 远端 reduce-add 双向(两 rank 各累加进对端)────────────────
  {
    void *peer_a = nullptr;
    CK(cudaSetDevice(oa)); CK(cudaMalloc(&peer_a, bytes)); CK(cudaMemset(peer_a, 0, bytes));
    __half2 v = __float2half2_rn(1.0f);
    CK(cudaSetDevice(oa)); CK(cudaEventRecord(e0, s1));
    for (int t = 0; t < k; ++t) {
      CK(cudaSetDevice(oa));
      k_red_add_peer<<<GRID, BLOCK, 0, s1>>>((__half2*)peer_b, n2, v);
      CK(cudaSetDevice(ob));
      k_red_add_peer<<<GRID, BLOCK, 0, s2>>>((__half2*)peer_a, n2, v);
    }
    CK(cudaStreamSynchronize(s1)); CK(cudaStreamSynchronize(s2));
    CK(cudaSetDevice(oa)); CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    report("RED-ADD 双向", bytes, k, ms_between(e0, e1));
    CK(cudaSetDevice(oa)); CK(cudaFree(peer_a));
  }

  // ── ⑨ ★★ **读本地 → 写对端**(= owl stage 核的数据面)──────────────────
  //   与 ① 的 `k_write_peer`(写常量)**唯一差别 = 源是真读** ⇒ 若它掉到一半带宽,
  //   就定谳"**PCIe 写会把本地读挡住**"(与 §41 的"写独占资源"同一机制)。
  {
    for (int w = 0; w < 3; ++w)
      k_read_write_peer<<<GRID, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a, n16);
    CK(cudaStreamSynchronize(s1));
    CK(cudaEventRecord(e0, s1));
    for (int t = 0; t < k; ++t)
      k_read_write_peer<<<GRID, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a, n16);
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    report("读本地→写对端", bytes, k, ms_between(e0, e1));
    // 对照:同核但源在**本端**且只写本端(纯本地 读→写)
    CK(cudaEventRecord(e0, s1));
    for (int t = 0; t < k; ++t)
      k_read_write_peer<<<GRID, BLOCK, 0, s1>>>((uint4*)local_a, (const uint4*)src_a, n16);
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    report("读本地→写本端(对照)", bytes, k, ms_between(e0, e1));
  }

  // ── ⑦ ★★ **裸 CUDA 图**双流(分辨"CUDA 图不并行" vs "owl 捕获有问题")──
  //   形状与 owl 的 M6 完全同构:[写对端 ×K] 与 [本地 RMW ×K] 两条**独立分支**,
  //   事件 fork/join 成一张图 ⇒ 回放。判据:图回放 ≈ max(两链) ⇒ CUDA 图并行;
  //   ≈ 和 ⇒ CUDA 图本身就把无依赖分支串行化。
  {
    cudaStream_t s3;
    CK(cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking));
    cudaEvent_t fk, jn;
    CK(cudaEventCreateWithFlags(&fk, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&jn, cudaEventDisableTiming));

    // 图 A:[写链(COMPUTE)] ‖ [RMW 链(AUX)]
    // ⚠️ **首版写错了**(2026-10-18 自纠):当时在写链之前有 `cudaStreamWaitEvent(s1, jn)`
    // ⇒ 写链被挡在 RMW 链之后 ⇒ "并行图"其实等于串行图,读出的 1.000 是假的。
    // **真并行图 = 两条分支互不等待**(s1 只在链首给 s3 记 fork 边,之后各走各的)。
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
    CK(cudaEventRecord(fk, s1));
    CK(cudaStreamWaitEvent(s3, fk, 0));
    const int rmw_mult = getenv("RMW_MULT") ? atoi(getenv("RMW_MULT")) : 1;
    for (int t = 0; t < k * rmw_mult; ++t)
      k_rmw_local<<<GRID, BLOCK, 0, s3>>>((uint4*)local_a, (const uint4*)src_a, n16);
    for (int t = 0; t < k; ++t)
      k_write_peer<<<GRID, BLOCK, 0, s1>>>((uint4*)peer_b, n16, 0x1234u);
    // ★ **末尾 join**(必须在**两条链都发完之后**记):`cudaStreamWaitEvent` 只影响
    // **之后**发射的工作 ⇒ s1 上已发射的写链**不会**被挡在 s3 链之后。
    // ⚠️ 不 join 会被驱动以 UNJOINED 整窗拒掉(§39 四形状实测)。
    CK(cudaEventRecord(jn, s3));
    CK(cudaStreamWaitEvent(s1, jn, 0));
    cudaGraph_t g_par;
    CK(cudaStreamEndCapture(s1, &g_par));
    cudaGraphExec_t ge_par;
    CK(cudaGraphInstantiate(&ge_par, g_par, 0));

    // 图 B:同两链,但**全在一条流**(串行对照)
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
    for (int t = 0; t < k; ++t)
      k_write_peer<<<GRID, BLOCK, 0, s1>>>((uint4*)peer_b, n16, 0x1234u);
    for (int t = 0; t < k; ++t)
      k_rmw_local<<<GRID, BLOCK, 0, s1>>>((uint4*)local_a, (const uint4*)src_a, n16);
    cudaGraph_t g_ser;
    CK(cudaStreamEndCapture(s1, &g_ser));
    cudaGraphExec_t ge_ser;
    CK(cudaGraphInstantiate(&ge_ser, g_ser, 0));

    for (int w = 0; w < 3; ++w) {
      CK(cudaGraphLaunch(ge_par, s1));
      CK(cudaGraphLaunch(ge_ser, s1));
    }
    CK(cudaStreamSynchronize(s1));
    CK(cudaEventRecord(e0, s1));
    for (int t = 0; t < 10; ++t) CK(cudaGraphLaunch(ge_par, s1));
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    const double par = ms_between(e0, e1) / 10.0;
    CK(cudaEventRecord(e0, s1));
    for (int t = 0; t < 10; ++t) CK(cudaGraphLaunch(ge_ser, s1));
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    const double ser = ms_between(e0, e1) / 10.0;
    // ★ **串行度扫描**(与 owl 的 `push_graph_overlap_probe` 同判据):
    //   `写链 × k` ‖ `RMW 链 × k·mult` ⇒ 看 `D − S` 是否随 R 线性长。
    cudaGraph_t g_rmw;
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
    for (int t = 0; t < k * rmw_mult; ++t)
      k_rmw_local<<<GRID, BLOCK, 0, s1>>>((uint4*)local_a, (const uint4*)src_a, n16);
    CK(cudaStreamEndCapture(s1, &g_rmw));
    cudaGraphExec_t ge_rmw;
    CK(cudaGraphInstantiate(&ge_rmw, g_rmw, 0));
    for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge_rmw, s1));
    CK(cudaStreamSynchronize(s1));
    CK(cudaEventRecord(e0, s1));
    for (int t = 0; t < 10; ++t) CK(cudaGraphLaunch(ge_rmw, s1));
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    const double r_only = ms_between(e0, e1) / 10.0;
    CK(cudaGraphExecDestroy(ge_rmw)); CK(cudaGraphDestroy(g_rmw));

    printf("\n  **裸 CUDA 图**:双分支回放 %.3f ms | 单流(串行)回放 %.3f ms ⇒ 比 %.3f\n",
           par, ser, par / ser);
    printf("  **裸 CUDA 串行度扫描**(mult=%d):RMW 链单跑 %.3f ms | 双分支 %.3f ms ⇒ "
           "**D−S = %.3f ms**(应 ≈ RMW 链单跑 %.3f ms)⇒ 吸收率 %.1f%%\n",
           rmw_mult, r_only, par, par - (ser - r_only), r_only,
           100.0 * (1.0 - (par - (ser - r_only)) / (r_only > 0 ? r_only : 1)));
    printf("  (0%% = 完全串行;100%% = 完全吸收;与 owl 侧同判据)\n");

    // ★ **同质对照**:两条**本地 RMW 链**(同核、不饱和、不碰 PCIe)
    //   ⇒ 若图内能并行,双链 ≈ 单链(比值 1.0);若串行,双链 ≈ 2× 单链。
    {
      cudaGraph_t g2;
      CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
      CK(cudaEventRecord(fk, s1));
      CK(cudaStreamWaitEvent(s3, fk, 0));
      for (int t = 0; t < k; ++t)
        k_rmw_local<<<GRID, BLOCK, 0, s3>>>((uint4*)local_a, (const uint4*)src_a, n16);
      for (int t = 0; t < k; ++t)
        k_rmw_local<<<GRID, BLOCK, 0, s1>>>((uint4*)local_a, (const uint4*)src_a, n16);
      CK(cudaEventRecord(jn, s3));
      CK(cudaStreamWaitEvent(s1, jn, 0));
      CK(cudaStreamEndCapture(s1, &g2));
      cudaGraphExec_t ge2;
      CK(cudaGraphInstantiate(&ge2, g2, 0));
      cudaGraph_t g1;
      CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
      for (int t = 0; t < k; ++t)
        k_rmw_local<<<GRID, BLOCK, 0, s1>>>((uint4*)local_a, (const uint4*)src_a, n16);
      CK(cudaStreamEndCapture(s1, &g1));
      cudaGraphExec_t ge1;
      CK(cudaGraphInstantiate(&ge1, g1, 0));
      for (int w = 0; w < 3; ++w) { CK(cudaGraphLaunch(ge2, s1)); CK(cudaGraphLaunch(ge1, s1)); }
      CK(cudaStreamSynchronize(s1));
      CK(cudaEventRecord(e0, s1));
      for (int t = 0; t < 10; ++t) CK(cudaGraphLaunch(ge1, s1));
      CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
      const double one = ms_between(e0, e1) / 10.0;
      CK(cudaEventRecord(e0, s1));
      for (int t = 0; t < 10; ++t) CK(cudaGraphLaunch(ge2, s1));
      CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
      const double two = ms_between(e0, e1) / 10.0;
      printf("  **同质对照**(两条本地 RMW 链):单链 %.3f ms | 双链(跨流)%.3f ms ⇒ 比 %.3f\n",
             one, two, two / one);
      printf("  (≈1.0 = 图内**能**并行同质链;≈2.0 = 图内**串行化** —— 与资源争用无关)\n");
      CK(cudaGraphExecDestroy(ge1)); CK(cudaGraphExecDestroy(ge2));
      CK(cudaGraphDestroy(g1)); CK(cudaGraphDestroy(g2));
    }
    printf("  (≈0.5 = CUDA 图**能**并行无依赖分支 ⇒ owl 的 1.000 是 owl 侧问题;"
           " ≈1.0 = CUDA 图本身串行化 ⇒ 双流/微批死路)\n");

    CK(cudaGraphExecDestroy(ge_par)); CK(cudaGraphExecDestroy(ge_ser));
    CK(cudaGraphDestroy(g_par)); CK(cudaGraphDestroy(g_ser));
    CK(cudaEventDestroy(fk)); CK(cudaEventDestroy(jn));
    CK(cudaStreamDestroy(s3));
  }

  // ── ⑧ ★★★ **图内并行的存在性证明**(时间戳重叠;比吞吐比干净)──────────
  //   两个核,各把自己的 `[start,end]`(全局纳秒时钟)写进显存;捕获成一张
  //   **两条独立分支**的图,回放后读回时间戳 ⇒ **看两个区间是否重叠**。
  //   · 重叠 ⇒ 图内**真的并行**;· 不重叠 ⇒ 图内**串行**(吞吐比会受资源争用干扰,
  //     时间戳不会)。
  {
    unsigned long long* ts = nullptr;
    CK(cudaSetDevice(oa)); CK(cudaMalloc(&ts, 4 * sizeof(uint64_t)));
    CK(cudaMemset(ts, 0, 4 * sizeof(uint64_t)));
    cudaStream_t t1, t2;
    CK(cudaStreamCreateWithFlags(&t1, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&t2, cudaStreamNonBlocking));
    cudaEvent_t fk2;
    CK(cudaEventCreateWithFlags(&fk2, cudaEventDisableTiming));

    CK(cudaStreamBeginCapture(t1, cudaStreamCaptureModeThreadLocal));
    CK(cudaEventRecord(fk2, t1));
    CK(cudaStreamWaitEvent(t2, fk2, 0));
    k_stamp<<<1, 32, 0, t2>>>(ts, 1, 20000000ull);   // 分支 B(自旋 ~20ms)
    k_stamp<<<1, 32, 0, t1>>>(ts, 0, 20000000ull);   // 分支 A(不等 B)
    cudaGraph_t g8;
    CK(cudaStreamEndCapture(t1, &g8));
    cudaGraphExec_t ge8;
    CK(cudaGraphInstantiate(&ge8, g8, 0));
    CK(cudaGraphLaunch(ge8, t1));
    CK(cudaStreamSynchronize(t1));

    unsigned long long h[4] = {0, 0, 0, 0};
    CK(cudaMemcpy(h, ts, sizeof(h), cudaMemcpyDeviceToHost));
    const long long a0 = (long long)h[0], a1 = (long long)h[1];
    const long long b0 = (long long)h[2], b1 = (long long)h[3];
    const long long ov = (a0 < b1 && b0 < a1) ? (std::min(a1, b1) - std::max(a0, b0)) : 0;
    const long long span = std::max(a1 - a0, b1 - b0);
    printf("\n  **图内并行存在性证明**(时间戳):A=[%lld,%lld] B=[%lld,%lld] ⇒ 重叠 %lld ns / 核时长 %lld ns = %.1f%%\n",
           a0, a1, b0, b1, ov, span, span > 0 ? 100.0 * (double)ov / (double)span : 0.0);
    printf("  (重叠 >50%% = **图内真并行**;≈0 = 图内串行)\n");

    CK(cudaGraphExecDestroy(ge8)); CK(cudaGraphDestroy(g8));
    CK(cudaEventDestroy(fk2));
    CK(cudaStreamDestroy(t1)); CK(cudaStreamDestroy(t2));
    CK(cudaFree(ts));
  }

  CK(cudaSetDevice(oa));
  CK(cudaStreamDestroy(s1)); CK(cudaStreamDestroy(s2));
  printf("[rmw] done\n");
  return 0;
}
