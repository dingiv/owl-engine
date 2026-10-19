// ============================================================================
// owl TP2 —— "PCIe 写 ‖ 本地工作" 重叠归属探针(§41 的裂缝复检;2026-10-19)
//
// ## 立案(两条互斥的证据同时躺在 doc 里)
//
// | 出处 | 形状 | 结论 |
// |---|---|---|
// | §38.2(裸 CUDA、**两条 non-blocking 流**、非图)| W ‖ RMW = 400.5µs(W=399.6)| **重叠 97%** ⇒ "路 A 成立" |
// | §41.4(裸 CUDA、**图内两分支**、同形状)| D = 写链 + RMW 链(严格 1:1)| **完全串行** ⇒ "跨流藏=死" |
//
// 二者不可能同时为真。差别只可能来自三个地方,本探针把它们**逐个单拉出来**:
//
//   ① **图 / 非图**(§38.2 是非图、§41 是图)—— 但 §41.3 的"读‖本地"在图内能重叠 92%
//      ⇒ "图把分支串行化"这个解释本身就有反例;
//   ② **计时口径**(非图双流若只在 s1 上夹事件、不等 s2,量的就只是写链 —— 经典探针 bug);
//   ③ **真硬件资源冲突**。
//
// ## 臂(同形状 5.24MB、同几何、同计时口径;A 与 B 的块数之和 = E 的块数)
//
//   A  写链单跑(读本端 src → `st.global` 写对端 dst + `__threadfence_system`)—— 生产 stage 数据面
//   B  本地 RMW 链单跑(`dst += src`,读 2 写 1)—— 生产 reduce 数据面
//   C  **非图**双流:A ‖ B,末尾 **join 事件**(计时口径必须包含 s2!)
//   D  **图内**双分支:A ‖ B(链首 fork、链尾 join)
//   D2 **图内**同质对照:B ‖ B(§41 的控制臂:图到底跑不跑并行分支)
//   E  **单核融合**:一次发射,前一半 CTA 干 A、后一半干 B(不依赖任何跨流机制)
//   F  **单核交错**:每个 CTA 同时干 A 与 B 的各一段(核内流水)
//
// ## 网格扫描(`OVL_WGRID` / `OVL_RGRID`;缺省 16/16 = 与 §41 探针同几何)
//
// 本探针与 §41 的**唯一可疑差别**是发射几何:GRID=16 块只占 16 个 SM(68 个里)
// ⇒ 两条链天然落在不同 SM 上。把两条链的网格分别拉到**填满设备**再看串行度:
// 若那时才变 1.000 ⇒ 机制是 **SM 占用(并发核调度)**,**不是**“PCIe 写独占资源”
// ⇒ 处置完全不同(前者是调度问题,后者是硬件死结)。
//
// 判据:`串行度 = (arm − A) / B` —— 0 = 完全重叠,1 = 完全串行。
//
// ## 编译 / 运行
//
//   nvcc -O3 -arch=sm_86 -o /tmp/p2p-ovl tools/p2p-overlap-probe.cu
//   RMW_K=128 /tmp/p2p-ovl <ordA> <ordB> [bytes]
//
// ⚠️ 卡序 = **CUDA ordinal**(本机 3080 对 = 0/3;3090 对 = 1/2)。跑前确认目标卡空闲
// (`nvidia-smi --query-compute-apps`)—— P2P 与邻居的负载会互相污染。
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

// ── 写腿(生产 stage 数据面:读本端真源 ⇒ 写对端)────────────────────────────
__global__ void k_wp(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n16) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride)
    dst[i] = src[i];
  __threadfence_system();  // 发布律:数据系统可见之后才许发 flag(生产核有)
}

// ── ★ PCIe **读**腿(v1 AR 的 reduce 数据面:读对端 slot + 本地累加)─────────
// 立案:§41.3 量过 [读对端 ‖ 本地 RMW] = 92% 重叠,但那是在**图内**且对手是 RMW;
//       §44 量出 [**写**对端 ‖ 真 GEMM] = 只有 24%。**缺的那一格 = [读 ‖ 真 GEMM]。**
//       若这一格也 ~90% ⇒ "把 AR 藏进计算"(微批/TBO)在**读腿形态**下是活的,
//       而且**不吃任何精度**(对比 fp8 交换要砍到 e4m3)。
__global__ void k_rp(uint4* __restrict__ dst, const uint4* __restrict__ src_peer, size_t n16) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
    const uint4 v = src_peer[i];          // ← PCIe 读(non-posted,要等完成)
    uint4 d = dst[i];
    d.x += v.x; d.y += v.y; d.z += v.z; d.w += v.w;   // 本地累加:防读被优化掉 + 就是 v1 语义
    dst[i] = d;
  }
  __threadfence_system();
}

// ── 本地 RMW(生产 reduce 数据面:读 2 写 1)────────────────────────────────
__global__ void k_rmw(uint4* __restrict__ dst, const uint4* __restrict__ src, size_t n16) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
    uint4 a = dst[i], b = src[i];
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    dst[i] = a;
  }
}

// ── 单核融合(前一半 CTA 写对端,后一半 CTA 本地 RMW)──────────────────────
__global__ void k_fused(uint4* __restrict__ pdst, const uint4* __restrict__ psrc,
                        uint4* __restrict__ ldst, const uint4* __restrict__ lsrc,
                        size_t n16) {
  const unsigned half = gridDim.x / 2;
  const size_t stride = (size_t)half * blockDim.x;
  if (blockIdx.x < half) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride)
      pdst[i] = psrc[i];
    __threadfence_system();
  } else {
    const unsigned b = blockIdx.x - half;
    for (size_t i = (size_t)b * blockDim.x + threadIdx.x; i < n16; i += stride) {
      uint4 a = ldst[i], c = lsrc[i];
      a.x += c.x; a.y += c.y; a.z += c.z; a.w += c.w;
      ldst[i] = a;
    }
  }
}

// ── 单核交错(每个 CTA 各干一段写 + 一段 RMW)────────────────────────────────
__global__ void k_interleaved(uint4* __restrict__ pdst, const uint4* __restrict__ psrc,
                              uint4* __restrict__ ldst, const uint4* __restrict__ lsrc,
                              size_t n16) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride)
    pdst[i] = psrc[i];
  __threadfence_system();
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
    uint4 a = ldst[i], c = lsrc[i];
    a.x += c.x; a.y += c.y; a.z += c.z; a.w += c.w;
    ldst[i] = a;
  }
}

// ── 计算/访存竞争者(模拟 prefill 的 GEMM:读大块权重 ⇒ 吃满本地带宽)─────────
__global__ void k_stream(const uint4* __restrict__ w, float* __restrict__ sink, size_t n16) {
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
    uint4 v = w[i];
    acc.x += (float)(v.x & 0xffffu); acc.y += (float)(v.y & 0xffffu);
    acc.z += (float)(v.z & 0xffffu); acc.w += (float)(v.w & 0xffffu);
  }
  const float s = acc.x + acc.y + acc.z + acc.w;
  if (s == -1.0f) sink[0] = s;  // 防死代码消除;正常永不写
}

static double ms(cudaEvent_t a, cudaEvent_t b) {
  float out = 0.f;
  CK(cudaEventElapsedTime(&out, a, b));
  return (double)out;
}

static int g_iters = 128;
static size_t g_bytes = 5242880;
static double g_write_us = 0.0, g_rmw_us = 0.0;
static int g_wgrid = GRID, g_rgrid = GRID;

static void row(const char* tag, double total_ms) {
  const double us = total_ms * 1e3 / (double)g_iters;
  const double ser = g_rmw_us > 0.0 ? (us - g_write_us) / g_rmw_us : 0.0;
  printf("  %-30s %9.2f us   %6.2f GB/s   串行度 %+6.3f\n", tag, us,
         (double)g_bytes / (us * 1e-6) / 1e9, ser);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "用法: %s <ordA> <ordB> [bytes]\n", argv[0]);
    return 2;
  }
  const int oa = atoi(argv[1]), ob = atoi(argv[2]);
  if (argc > 3) g_bytes = (size_t)atoll(argv[3]);
  if (getenv("RMW_K")) g_iters = atoi(getenv("RMW_K"));
  if (getenv("OVL_WGRID")) g_wgrid = atoi(getenv("OVL_WGRID"));
  if (getenv("OVL_RGRID")) g_rgrid = atoi(getenv("OVL_RGRID"));
  const int k = g_iters;
  const size_t n16 = g_bytes / 16;

  int can = 0;
  CK(cudaDeviceCanAccessPeer(&can, oa, ob));
  if (!can) { fprintf(stderr, "[ovl] 无 P2P(ord %d -> %d)\n", oa, ob); return 3; }
  CK(cudaSetDevice(oa)); CK(cudaDeviceEnablePeerAccess(ob, 0));
  CK(cudaSetDevice(ob)); CK(cudaDeviceEnablePeerAccess(oa, 0));

  void *peer_b = nullptr, *local_a = nullptr, *src_a = nullptr, *src_b = nullptr;
  CK(cudaSetDevice(ob)); CK(cudaMalloc(&peer_b, g_bytes)); CK(cudaMemset(peer_b, 0, g_bytes));
  CK(cudaSetDevice(oa)); CK(cudaMalloc(&local_a, g_bytes)); CK(cudaMemset(local_a, 0, g_bytes));
  CK(cudaMalloc(&src_a, g_bytes)); CK(cudaMemset(src_a, 0x5a, g_bytes));
  CK(cudaMalloc(&src_b, g_bytes)); CK(cudaMemset(src_b, 0x3c, g_bytes));

  cudaStream_t s1, s2, s3;
  CK(cudaSetDevice(oa));
  CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking));
  cudaEvent_t e0, e1, fk, jn, eB;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&eB));
  CK(cudaEventCreateWithFlags(&fk, cudaEventDisableTiming));
  CK(cudaEventCreateWithFlags(&jn, cudaEventDisableTiming));

  printf("[ovl] A=ord%d B=ord%d payload=%.2f MB K=%d grid W=%d R=%d\n", oa, ob,
         (double)g_bytes / 1048576.0, k, g_wgrid, g_rgrid);

  // ── A:写链单跑 ────────────────────────────────────────────────────────────
  for (int w = 0; w < 3; ++w) k_wp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a, n16);
  CK(cudaStreamSynchronize(s1));
  CK(cudaEventRecord(e0, s1));
  for (int t = 0; t < k; ++t) k_wp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a, n16);
  CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
  g_write_us = ms(e0, e1) * 1e3 / k;
  row("A 写链(单)", ms(e0, e1));

  // ── B:本地 RMW 链单跑 ─────────────────────────────────────────────────────
  CK(cudaEventRecord(e0, s1));
  for (int t = 0; t < k; ++t) k_rmw<<<g_rgrid, BLOCK, 0, s1>>>((uint4*)local_a, (const uint4*)src_b, n16);
  CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
  g_rmw_us = ms(e0, e1) * 1e3 / k;
  row("B 本地 RMW(单)", ms(e0, e1));
  printf("  ── 串行预测 = A + B = %.2f us ──\n", g_write_us + g_rmw_us);

  // ── C:**非图**双流(计时口径含 join!)────────────────────────────────────
  CK(cudaEventRecord(e0, s1));
  CK(cudaEventRecord(fk, s1));
  CK(cudaStreamWaitEvent(s2, fk, 0));
  for (int t = 0; t < k; ++t) k_wp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a, n16);
  for (int t = 0; t < k; ++t) k_rmw<<<g_rgrid, BLOCK, 0, s2>>>((uint4*)local_a, (const uint4*)src_b, n16);
  CK(cudaEventRecord(jn, s2));
  CK(cudaStreamWaitEvent(s1, jn, 0));  // ★ 不 join 就量不到 s2 —— §38.2 的嫌疑点
  CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
  row("C 非图双流 A ‖ B(join 计时)", ms(e0, e1));

  // ── D:**图内**两分支 ──────────────────────────────────────────────────────
  cudaGraph_t g_par, g_hom;
  cudaGraphExec_t ge_par, ge_hom;
  CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
  CK(cudaEventRecord(fk, s1));
  CK(cudaStreamWaitEvent(s3, fk, 0));
  for (int t = 0; t < k; ++t) k_rmw<<<g_rgrid, BLOCK, 0, s3>>>((uint4*)local_a, (const uint4*)src_b, n16);
  for (int t = 0; t < k; ++t) k_wp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a, n16);
  CK(cudaEventRecord(jn, s3));
  CK(cudaStreamWaitEvent(s1, jn, 0));
  CK(cudaStreamEndCapture(s1, &g_par));
  // ★ §39 的教训:结构要**验**不要假设 —— 数节点数
  {
    size_t nn = 0;
    CK(cudaGraphGetNodes(g_par, nullptr, &nn));
    printf("  [graph] D 节点数 = %zu(期望 2k+1 = %d)\n", nn, 2 * k + 1);
  }
  CK(cudaGraphInstantiate(&ge_par, g_par, 0));

  // ── D2:图内同质对照(B ‖ B)—— 证明"图跑得来并行分支"──
  CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
  CK(cudaEventRecord(fk, s1));
  CK(cudaStreamWaitEvent(s3, fk, 0));
  for (int t = 0; t < k; ++t) k_rmw<<<g_rgrid, BLOCK, 0, s3>>>((uint4*)local_a, (const uint4*)src_b, n16);
  for (int t = 0; t < k; ++t) k_rmw<<<g_rgrid, BLOCK, 0, s1>>>((uint4*)local_a, (const uint4*)src_a, n16);
  CK(cudaEventRecord(jn, s3));
  CK(cudaStreamWaitEvent(s1, jn, 0));
  CK(cudaStreamEndCapture(s1, &g_hom));
  CK(cudaGraphInstantiate(&ge_hom, g_hom, 0));

  for (int w = 0; w < 3; ++w) { CK(cudaGraphLaunch(ge_par, s1)); CK(cudaGraphLaunch(ge_hom, s1)); }
  CK(cudaStreamSynchronize(s1));
  CK(cudaEventRecord(e0, s1));
  for (int t = 0; t < 10; ++t) CK(cudaGraphLaunch(ge_par, s1));
  CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
  row("D 图内两分支 A ‖ B", ms(e0, e1) / 10.0);
  CK(cudaEventRecord(e0, s1));
  for (int t = 0; t < 10; ++t) CK(cudaGraphLaunch(ge_hom, s1));
  CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
  {
    const double us = ms(e0, e1) / 10.0 * 1e3 / k;
    printf("  %-30s %9.2f us   (同质对照:两分支都是本地 RMW)\n", "D2 图内两分支 B ‖ B", us);
  }

  // ── E:单核融合(2×GRID 块:一半写、一半 RMW)──────────────────────────────
  for (int w = 0; w < 3; ++w)
    k_fused<<<2 * GRID, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a,
                                        (uint4*)local_a, (const uint4*)src_b, n16);
  CK(cudaStreamSynchronize(s1));
  CK(cudaEventRecord(e0, s1));
  for (int t = 0; t < k; ++t)
    k_fused<<<2 * GRID, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a,
                                        (uint4*)local_a, (const uint4*)src_b, n16);
  CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
  row("E 单核融合(半写半 RMW)", ms(e0, e1));

  // ── F:单核交错(每 CTA 先写后 RMW)────────────────────────────────────────
  for (int w = 0; w < 3; ++w)
    k_interleaved<<<GRID, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a,
                                          (uint4*)local_a, (const uint4*)src_b, n16);
  CK(cudaStreamSynchronize(s1));
  CK(cudaEventRecord(e0, s1));
  for (int t = 0; t < k; ++t)
    k_interleaved<<<GRID, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a,
                                          (uint4*)local_a, (const uint4*)src_b, n16);
  CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
  row("F 单核交错(每 CTA 写→RMW)", ms(e0, e1));

  // ── G/H/I:把"本地大负载(权重流)"当竞争者 —— TBO/微批是否有地基 ─────────────
  // 生产形状:stage 核是**小网格 + 链路受限**(16 块);竞争者是**填满设备**的 GEMM。
  {
    const size_t hier = 256u * 1024u * 1024u;  // 256MB 权重流(≈ prefill 一次权重读)
    const size_t n16w = hier / 16;
    const int sgrid = 4096;
    void* wbuf = nullptr; float* sink = nullptr;
    CK(cudaMalloc(&wbuf, hier)); CK(cudaMemset(wbuf, 0x11, hier));
    CK(cudaMalloc(&sink, 4)); CK(cudaMemset(sink, 0, 4));
    for (int t = 0; t < 3; ++t)
      k_stream<<<sgrid, 256, 0, s2>>>((const uint4*)wbuf, sink, n16w);
    CK(cudaStreamSynchronize(s2));
    CK(cudaEventRecord(e0, s2));
    for (int t = 0; t < k; ++t) k_stream<<<sgrid, 256, 0, s2>>>((const uint4*)wbuf, sink, n16w);
    CK(cudaEventRecord(e1, s2)); CK(cudaEventSynchronize(e1));
    const double g_us = ms(e0, e1) * 1e3 / k;
    printf("  %-30s %9.2f us   %6.2f GB/s   (竞争者分母)\n", "G 权重流链(单,4096 块)", g_us,
           (double)hier / (g_us * 1e-6) / 1e9);

    // H:非图双流(计时含 join)
    CK(cudaEventRecord(e0, s1));
    CK(cudaEventRecord(fk, s1)); CK(cudaStreamWaitEvent(s2, fk, 0));
    for (int t = 0; t < k; ++t) k_wp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a, n16);
    for (int t = 0; t < k; ++t) k_stream<<<sgrid, 256, 0, s2>>>((const uint4*)wbuf, sink, n16w);
    CK(cudaEventRecord(jn, s2)); CK(cudaStreamWaitEvent(s1, jn, 0));
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    {
      const double us = ms(e0, e1) * 1e3 / k;
      printf("  %-30s %9.2f us   串行度(对 G) %+6.3f  (串行预测 %.0f us)\n",
             "H 非图双流 写 ‖ 权重流", us, (us - g_us) / g_us, g_us + g_write_us);
    }

    // I:图内两分支(必须按 §39.2 的 join 语义:两链发完再 join)
    cudaGraph_t g_w; cudaGraphExec_t ge_w;
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
    CK(cudaEventRecord(fk, s1)); CK(cudaStreamWaitEvent(s3, fk, 0));
    for (int t = 0; t < k; ++t) k_stream<<<sgrid, 256, 0, s3>>>((const uint4*)wbuf, sink, n16w);
    for (int t = 0; t < k; ++t) k_wp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a, n16);
    CK(cudaEventRecord(jn, s3)); CK(cudaStreamWaitEvent(s1, jn, 0));
    CK(cudaStreamEndCapture(s1, &g_w));
    CK(cudaGraphInstantiate(&ge_w, g_w, 0));
    for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge_w, s1));
    CK(cudaStreamSynchronize(s1));
    CK(cudaEventRecord(e0, s1));
    for (int t = 0; t < 10; ++t) CK(cudaGraphLaunch(ge_w, s1));
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    {
      const double us = ms(e0, e1) / 10.0 * 1e3 / k;
      printf("  %-30s %9.2f us   串行度(对 G) %+6.3f\n", "I 图内两分支 写 ‖ 权重流", us,
             (us - g_us) / g_us);
    }
    CK(cudaGraphExecDestroy(ge_w)); CK(cudaGraphDestroy(g_w));
    CK(cudaFree(wbuf)); CK(cudaFree(sink));
  }

  // ── J/K/L:**真实 GEMM** 当竞争者(prefill 行并行 GEMM 形状 m=512,n=5120,k=5120,fp16)──
  // 这是"能否把 AR 藏进本层 GEMM"的直接分母:GEMM 与写腿互不依赖时能不能并行。
  {
    const int M = 512, N = 5120, K = 5120;
    void *Ab=nullptr,*Bb=nullptr,*Cb=nullptr;
    CK(cudaMalloc(&Ab,(size_t)M*K*2)); CK(cudaMemset(Ab,0x11,(size_t)M*K*2));
    CK(cudaMalloc(&Bb,(size_t)K*N*2)); CK(cudaMemset(Bb,0x11,(size_t)K*N*2));
    CK(cudaMalloc(&Cb,(size_t)M*N*2)); CK(cudaMemset(Cb,0,(size_t)M*N*2));
    cublasHandle_t hb; cublasCreate(&hb); cublasSetStream(hb, s2);
    const float one=1.f, zero=0.f;
    auto gemm = [&]{ cublasStatus_t st_ = cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
        Bb, CUDA_R_16F, N, Ab, CUDA_R_16F, K, &zero, Cb, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      if (st_ != CUBLAS_STATUS_SUCCESS) fprintf(stderr, "[ovl] cublas rc=%d\n", (int)st_); };
    for (int t=0;t<3;++t) gemm();
    CK(cudaStreamSynchronize(s2));
    CK(cudaEventRecord(e0, s2));
    for (int t=0;t<k;++t) gemm();
    CK(cudaEventRecord(e1, s2)); CK(cudaEventSynchronize(e1));
    const double j_us = ms(e0,e1)*1e3/k;
    printf("  %-30s %9.2f us   %6.1f TFLOPS  (竞争者分母)\n","J GEMM 链(m512 n5120 k5120)",
           j_us, 2.0*M*N*K/(j_us*1e-6)/1e12);

    CK(cudaEventRecord(e0, s1));
    CK(cudaEventRecord(fk, s1)); CK(cudaStreamWaitEvent(s2, fk, 0));
    for (int t=0;t<k;++t) k_wp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)peer_b,(const uint4*)src_a,n16);
    for (int t=0;t<k;++t) gemm();
    CK(cudaEventRecord(jn, s2)); CK(cudaStreamWaitEvent(s1, jn, 0));
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    {
      const double us = ms(e0,e1)*1e3/k;
      printf("  %-30s %9.2f us   相对 max(J,A)=%.0f 的超出 %+.1f%%   串行预测 %.0f us\n",
             "K 非图双流 写 ‖ GEMM", us, (j_us>g_write_us?j_us:g_write_us),
             100.0*(us-(j_us>g_write_us?j_us:g_write_us))/(j_us>g_write_us?j_us:g_write_us),
             j_us+g_write_us);
    }
    cudaGraph_t g_g; cudaGraphExec_t ge_g;
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
    CK(cudaEventRecord(fk, s1)); CK(cudaStreamWaitEvent(s3, fk, 0));
    if (cublasSetStream(hb, s3) != CUBLAS_STATUS_SUCCESS) fprintf(stderr,"[ovl] setStream s3 失败\n");
    for (int t=0;t<k;++t) gemm();
    if (cublasSetStream(hb, s2) != CUBLAS_STATUS_SUCCESS) fprintf(stderr,"[ovl] setStream s2 失败\n");
    for (int t=0;t<k;++t) k_wp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)peer_b,(const uint4*)src_a,n16);
    CK(cudaEventRecord(jn, s3)); CK(cudaStreamWaitEvent(s1, jn, 0));
    CK(cudaStreamEndCapture(s1, &g_g));
    // ★ 存在性判据(§39):节点数不足 ⇒ GEMM 根本没进图,L 的"完美重叠"就是假的
    {
      size_t nn = 0;
      CK(cudaGraphGetNodes(g_g, nullptr, &nn));
      printf("  [graph] L 节点数 = %zu(期望 2k = %d)⇒ %s\n", nn, 2 * k,
             nn >= (size_t)(2 * k) ? "GEMM 确在图内" : "⚠️ GEMM 缺失,读数作废");
    }
    CK(cudaGraphInstantiate(&ge_g, g_g, 0));
    for (int w=0;w<3;++w) CK(cudaGraphLaunch(ge_g, s1));
    CK(cudaStreamSynchronize(s1));
    CK(cudaEventRecord(e0, s1));
    for (int t=0;t<10;++t) CK(cudaGraphLaunch(ge_g, s1));
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    {
      const double us = ms(e0,e1)/10.0*1e3/k;
      printf("  %-30s %9.2f us   相对 max(J,A) 的超出 %+.1f%%\n","L 图内两分支 写 ‖ GEMM", us,
             100.0*(us-(j_us>g_write_us?j_us:g_write_us))/(j_us>g_write_us?j_us:g_write_us));
    }
    CK(cudaGraphExecDestroy(ge_g)); CK(cudaGraphDestroy(g_g));
    cublasDestroy(hb);
    CK(cudaFree(Ab)); CK(cudaFree(Bb)); CK(cudaFree(Cb));
  }

  // ── R 腿:**PCIe 读** 单独 + 与真 GEMM 并行(§44 缺的那一格)────────────────
  {
    const int M = 512, N = 5120, K = 5120;
    void *Ab=nullptr,*Bb=nullptr,*Cb=nullptr;
    CK(cudaMalloc(&Ab,(size_t)M*K*2)); CK(cudaMemset(Ab,0x11,(size_t)M*K*2));
    CK(cudaMalloc(&Bb,(size_t)K*N*2)); CK(cudaMemset(Bb,0x11,(size_t)K*N*2));
    CK(cudaMalloc(&Cb,(size_t)M*N*2)); CK(cudaMemset(Cb,0,(size_t)M*N*2));
    cublasHandle_t hb; cublasCreate(&hb); cublasSetStream(hb, s2);
    const float one=1.f, zero=0.f;
    auto gemm = [&]{ cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
        Bb, CUDA_R_16F, N, Ab, CUDA_R_16F, K, &zero, Cb, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT); };
    // ① 读链单跑
    for (int w=0;w<3;++w) k_rp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)local_a,(const uint4*)peer_b,n16);
    CK(cudaStreamSynchronize(s1));
    CK(cudaEventRecord(e0, s1));
    for (int t=0;t<k;++t) k_rp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)local_a,(const uint4*)peer_b,n16);
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    const double r_us = ms(e0,e1)*1e3/k;
    printf("  %-30s %9.2f us   %6.2f GB/s\n","R PCIe 读链(单)", r_us, (double)g_bytes/(r_us*1e-3)/1e9);
    // ② GEMM 单跑(分母)
    for (int t=0;t<3;++t) gemm();
    CK(cudaStreamSynchronize(s2));
    CK(cudaEventRecord(e0,s2));
    for (int t=0;t<k;++t) gemm();
    CK(cudaEventRecord(e1,s2)); CK(cudaEventSynchronize(e1));
    const double j_us = ms(e0,e1)*1e3/k;
    printf("  %-30s %9.2f us   %6.1f TFLOPS\n","J' GEMM 链(单;R 的分母)", j_us, 2.0*M*N*K/(j_us*1e-6)/1e12);
    // ③ 非图双流:读 ‖ GEMM(join 计时)
    CK(cudaEventRecord(e0, s1));
    CK(cudaEventRecord(fk, s1)); CK(cudaStreamWaitEvent(s2, fk, 0));
    for (int t=0;t<k;++t) k_rp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)local_a,(const uint4*)peer_b,n16);
    for (int t=0;t<k;++t) gemm();
    CK(cudaEventRecord(jn, s2)); CK(cudaStreamWaitEvent(s1, jn, 0));
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    {
      const double us = ms(e0,e1)*1e3/k;
      const double base = (j_us > r_us) ? j_us : r_us;
      printf("  %-30s %9.2f us   max(J',R)=%.0f 超出 %+.1f%%   串行预测 %.0f us\n",
             "M 非图双流 读 ‖ GEMM", us, base, 100.0*(us-base)/base, j_us+r_us);
    }
    // ④ 图内两分支:读 ‖ GEMM
    cudaGraph_t g_r; cudaGraphExec_t ge_r;
    CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
    CK(cudaEventRecord(fk, s1)); CK(cudaStreamWaitEvent(s3, fk, 0));
    if (cublasSetStream(hb, s3) != CUBLAS_STATUS_SUCCESS) fprintf(stderr,"[ovl] setStream 失败\n");
    for (int t=0;t<k;++t) gemm();
    if (cublasSetStream(hb, s2) != CUBLAS_STATUS_SUCCESS) fprintf(stderr,"[ovl] setStream 失败\n");
    for (int t=0;t<k;++t) k_rp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)local_a,(const uint4*)peer_b,n16);
    CK(cudaEventRecord(jn, s3)); CK(cudaStreamWaitEvent(s1, jn, 0));
    CK(cudaStreamEndCapture(s1, &g_r));
    { size_t nn2=0; CK(cudaGraphGetNodes(g_r,nullptr,&nn2));
      printf("  [graph] N 节点数 = %zu(期望 2k = %d)%s\n", nn2, 2*k, nn2>=(size_t)(2*k)?"":" ⚠️ GEMM 缺失"); }
    CK(cudaGraphInstantiate(&ge_r, g_r, 0));
    for (int w=0;w<3;++w) CK(cudaGraphLaunch(ge_r, s1));
    CK(cudaStreamSynchronize(s1));
    CK(cudaEventRecord(e0, s1));
    for (int t=0;t<10;++t) CK(cudaGraphLaunch(ge_r, s1));
    CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
    {
      const double us = ms(e0,e1)/10.0*1e3/k;
      const double base = (j_us > r_us) ? j_us : r_us;
      printf("  %-30s %9.2f us   max(J',R) 超出 %+.1f%%\n","N 图内两分支 读 ‖ GEMM", us, 100.0*(us-base)/base);
    }
    CK(cudaGraphExecDestroy(ge_r)); CK(cudaGraphDestroy(g_r));
    cublasDestroy(hb); CK(cudaFree(Ab)); CK(cudaFree(Bb)); CK(cudaFree(Cb));
  }

  // ── TBO 判据 v2:**微批数扫描**(N = 2/3/4)─────────────────────────────────
  //
  // v1(§49.2)的教训:两条流各跑一条链,只藏掉 36% 的 AR(吞吐比 1.22× / 理想 2×)。
  // 假设 = **两条链的满占用 GEMM 互相抢 SM** ⇒ 一条链的 AR 在飞时,另一条链没有 GEMM 可跑。
  // 判据:**N 变大时吞吐比是否往 N 走**。
  //   理想 = N(AR 全藏);若卡在 ~1.2–1.5 ⇒ **是资源死结,不是调度形状**。
  // ⚠️ 两条探针纪律(§49.2 踩过):①`cublasSetStream` 建 handle 后**立刻**设;
  //    ②读源必须在**对端**设备且工作集 > L2(这里 20MB 轮流 ⇒ 断掉 L2 供给)。
  {
    const int NG = 512;            // 全批 token 数
    const int N = 5120, K = 5120;
    const int NL = 6;              // 每链层数
    const int reps = 3;
    void *Ab=nullptr,*Bb=nullptr,*Cb=nullptr,*scr=nullptr;
    CK(cudaMalloc(&Ab,(size_t)NG*K*2)); CK(cudaMemset(Ab,0x11,(size_t)NG*K*2));
    CK(cudaMalloc(&Bb,(size_t)K*N*2)); CK(cudaMemset(Bb,0x11,(size_t)K*N*2));
    CK(cudaMalloc(&Cb,(size_t)NG*N*2)); CK(cudaMemset(Cb,0,(size_t)NG*N*2));
    CK(cudaSetDevice(ob)); CK(cudaMalloc(&scr,20u<<20)); CK(cudaMemset(scr,0x77,20u<<20));
    CK(cudaSetDevice(oa));
    const float one=1.f, zero=0.f;
    const int nlist[3]={2,3,4};
    for (int fix=0; fix<3; ++fix) {
      const int NB=nlist[fix];
      const int m = NG/NB;
      const size_t pl = g_bytes/(size_t)NB, pl16 = pl/16, nblk = (20u<<20)/pl;
      cudaStream_t st[4]; cublasHandle_t hh[4];
      for (int i=0;i<NB;++i) {
        CK(cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking));
        cublasCreate(&hh[i]);
        if (cublasSetStream(hh[i], st[i]) != CUBLAS_STATUS_SUCCESS) fprintf(stderr,"[ovl] setStream 失败\n");
      }
      auto gemm_i = [&](int i) {
        cublasGemmEx(hh[i],CUBLAS_OP_N,CUBLAS_OP_N,N,m,K,&one,Bb,CUDA_R_16F,N,Ab,CUDA_R_16F,K,
                     &zero,Cb,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
      };
      auto chain = [&](int i) {
        for (int L=0;L<NL;++L) {
          gemm_i(i);
          k_rp<<<g_wgrid, BLOCK, 0, st[i]>>>((uint4*)local_a,
              (const uint4*)scr + (size_t)((L*NB+i)%nblk)*pl16, pl16);
        }
      };
      for (int w=0;w<2;++w) chain(0);
      CK(cudaStreamSynchronize(st[0]));
      CK(cudaEventRecord(e0, st[0]));
      for (int t=0;t<reps;++t) chain(0);
      CK(cudaEventRecord(e1, st[0])); CK(cudaEventSynchronize(e1));
      const double solo = ms(e0,e1)/reps;
      CK(cudaEventRecord(e0, st[0]));
      CK(cudaEventRecord(fk, st[0]));
      for (int i=1;i<NB;++i) CK(cudaStreamWaitEvent(st[i], fk, 0));
      for (int i=0;i<NB;++i) for (int t=0;t<reps;++t) chain(i);
      for (int i=1;i<NB;++i) { CK(cudaEventRecord(jn, st[i])); CK(cudaStreamWaitEvent(st[0], jn, 0)); }
      CK(cudaEventRecord(e1, st[0])); CK(cudaEventSynchronize(e1));
      const double all = ms(e0,e1)/reps;
      printf("  %-30s N=%d(m=%3d,AR=%5.2fMB) 单链 %.3f ms | %d 链并行 %.3f ms ⇒ 吞吐比 **%.2f×** (理想 %d)\n",
             "V TBO 微批扫描", NB, m, (double)pl/1048576.0, solo, NB, all, NB*solo/all, NB);
      // ── Y:**错相启动**(只对 NB=2 做):链 1 先跑,链 0 等"链 1 的第一层 GEMM 完成"再开始
      //     ⇒ A 的 G 与 B 的 AR 天然错开半周期。判据:吞吐比是否从 1.2 跳到 ~2。
      if (NB == 2) {
        CK(cudaEventRecord(e0, st[0]));
        CK(cudaEventRecord(fk, st[0]));
        CK(cudaStreamWaitEvent(st[1], fk, 0));
        // 先只发链 1 的第一层 G,并在其后记事件
        for (int t=0;t<reps;++t) {
          gemm_i(1);
          CK(cudaEventRecord(eB, st[1]));
          CK(cudaStreamWaitEvent(st[0], eB, 0));   // ★ 链 0 从这儿才开始(错半周期)
          chain(0);
          for (int L=1;L<NL;++L) { gemm_i(1);
            k_rp<<<g_wgrid, BLOCK, 0, st[1]>>>((uint4*)src_b,(const uint4*)scr + (size_t)((L*2+1)%nblk)*pl16, pl16); }
        }
        CK(cudaEventRecord(jn, st[1])); CK(cudaStreamWaitEvent(st[0], jn, 0));
        CK(cudaEventRecord(e1, st[0])); CK(cudaEventSynchronize(e1));
        const double all2 = ms(e0,e1)/reps;
        printf("  %-30s N=2 **错相启动** = %.3f ms ⇒ 吞吐比 **%.2f×** (同相时 1.21×;理想 2)\n",
               "Y TBO 错相", all2, 2*solo/all2);
        CK(cudaStreamSynchronize(st[0])); CK(cudaStreamSynchronize(st[1]));
      }
      for (int i=0;i<NB;++i) { cublasDestroy(hh[i]); CK(cudaStreamDestroy(st[i])); }
    }
    CK(cudaFree(Ab)); CK(cudaFree(Bb)); CK(cudaFree(Cb));
    CK(cudaSetDevice(ob)); CK(cudaFree(scr)); CK(cudaSetDevice(oa));
  }

  // ── W:§49.4 的机制判据 —— **核级**单对 + 发序 + 双 GEMM 夹一个 AR ────────────
  // 立案:V 臂(N 链并行)吞吐比卡 1.21–1.28×、总时间 = GEMM + PCIe(零重叠);
  //       而 §48 的 M 臂(先发完读链、再发完 GEMM)重叠 97%。
  //       ⇒ 问:重叠到底取决于**核级资源**还是**发序/队列**?
  // 判据:pair ≈ max(单跑) ⇒ 能重叠;≈ 和 ⇒ 不能。
  {
    const int M = 512, N = 5120, K = 5120;
    void *Ab=nullptr,*Bb=nullptr,*Cb=nullptr,*scr=nullptr;
    CK(cudaMalloc(&Ab,(size_t)M*K*2)); CK(cudaMemset(Ab,0x11,(size_t)M*K*2));
    CK(cudaMalloc(&Bb,(size_t)K*N*2)); CK(cudaMemset(Bb,0x11,(size_t)K*N*2));
    CK(cudaMalloc(&Cb,(size_t)M*N*2)); CK(cudaMemset(Cb,0,(size_t)M*N*2));
    CK(cudaSetDevice(ob)); CK(cudaMalloc(&scr,20u<<20)); CK(cudaMemset(scr,0x77,20u<<20));
    CK(cudaSetDevice(oa));
    const float one=1.f, zero=0.f;
    cudaStream_t sg1,sg2,sar;
    CK(cudaStreamCreateWithFlags(&sg1,cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&sg2,cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&sar,cudaStreamNonBlocking));
    cublasHandle_t hA,hB; cublasCreate(&hA); cublasCreate(&hB);
    if (cublasSetStream(hA,sg1)!=CUBLAS_STATUS_SUCCESS||cublasSetStream(hB,sg2)!=CUBLAS_STATUS_SUCCESS) fprintf(stderr,"[ovl] setStream\n");
    auto gA=[&]{ cublasGemmEx(hA,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&one,Bb,CUDA_R_16F,N,Ab,CUDA_R_16F,K,&zero,Cb,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); };
    auto gB=[&]{ cublasGemmEx(hB,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&one,Bb,CUDA_R_16F,N,Ab,CUDA_R_16F,K,&zero,Cb,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); };
    auto ar=[&](int i){ k_rp<<<g_wgrid, BLOCK, 0, sar>>>((uint4*)local_a,(const uint4*)scr+(size_t)(i%8)*n16, n16); };
    const int R=5;
    // 分母:G 单跑 / AR 单跑
    auto timed=[&](cudaStream_t s, auto fn, const char* nm, size_t byt){
      for(int w=0;w<3;++w) fn();
      CK(cudaStreamSynchronize(s));
      CK(cudaEventRecord(e0,s)); for(int t=0;t<R;++t) fn(); CK(cudaEventRecord(e1,s)); CK(cudaEventSynchronize(e1));
      const double us=ms(e0,e1)*1e3/R;
      printf("  %-30s %9.2f us\n", nm, us);
      (void)byt; return us;
    };
    const double g_us = timed(sg1, gA, "X1 GEMM(单)", 0);
    const double r_us = timed(sar, [&]{ ar(0); }, "X2 AR 读(单,5.24MB)", 0);
    // ① AR 先 enqueue,G 后:同时发
    {
      CK(cudaEventRecord(e0, sg1));
      CK(cudaEventRecord(fk, sg1));
      CK(cudaStreamWaitEvent(sar, fk, 0)); CK(cudaStreamWaitEvent(sg2, fk, 0));
      for(int t=0;t<R;++t){ ar(t); gA(); }
      CK(cudaEventRecord(jn, sar)); CK(cudaStreamWaitEvent(sg1, jn, 0));
      CK(cudaEventRecord(e1, sg1)); CK(cudaEventSynchronize(e1));
      const double us=ms(e0,e1)*1e3/R;
      printf("  %-30s %9.2f us   max=%.0f 和=%.0f ⇒ %s\n","X3 [AR 先发 ‖ GEMM]", us,
             g_us>r_us?g_us:r_us, g_us+r_us, (us < (g_us>r_us?g_us:r_us)*1.15)?"重叠":"**串行**");
    }
    // ② G 先 enqueue,AR 后
    {
      CK(cudaEventRecord(e0, sg1));
      CK(cudaEventRecord(fk, sg1));
      CK(cudaStreamWaitEvent(sar, fk, 0));
      for(int t=0;t<R;++t){ gA(); ar(t); }
      CK(cudaEventRecord(jn, sar)); CK(cudaStreamWaitEvent(sg1, jn, 0));
      CK(cudaEventRecord(e1, sg1)); CK(cudaEventSynchronize(e1));
      const double us=ms(e0,e1)*1e3/R;
      printf("  %-30s %9.2f us   max=%.0f 和=%.0f ⇒ %s\n","X4 [GEMM 先发 ‖ AR]", us,
             g_us>r_us?g_us:r_us, g_us+r_us, (us < (g_us>r_us?g_us:r_us)*1.15)?"重叠":"**串行**");
    }
    // ③ 双 GEMM 夹一个 AR(模拟"总有一条链的 GEMM 可跑")
    {
      for(int w=0;w<2;++w){ gB(); }
      CK(cudaStreamSynchronize(sg2));
      CK(cudaEventRecord(e0, sg1));
      CK(cudaEventRecord(fk, sg1));
      CK(cudaStreamWaitEvent(sg2, fk, 0)); CK(cudaStreamWaitEvent(sar, fk, 0));
      for(int t=0;t<R;++t){ gA(); gB(); ar(t); }
      CK(cudaEventRecord(jn, sar)); CK(cudaStreamWaitEvent(sg1, jn, 0));
      CK(cudaEventRecord(e1, sg1)); CK(cudaEventSynchronize(e1));
      const double us=ms(e0,e1)*1e3/R;
      printf("  %-30s %9.2f us   对照:2 GEMM 单跑 = %.0f + AR %.0f\n","X5 [GEMM ‖ GEMM ‖ AR]", us, 2*g_us, r_us);
    }
    cublasDestroy(hA); cublasDestroy(hB);
    CK(cudaStreamDestroy(sg1)); CK(cudaStreamDestroy(sg2)); CK(cudaStreamDestroy(sar));
    CK(cudaFree(Ab)); CK(cudaFree(Bb)); CK(cudaFree(Cb));
    CK(cudaSetDevice(ob)); CK(cudaFree(scr)); CK(cudaSetDevice(oa));
  }

  // ── ★ 存在性判据(§39 教训:先证“真的跑了”,再信读数)──────────────────
  {
    CK(cudaMemset(peer_b, 0, g_bytes));
    CK(cudaMemset(local_a, 0, g_bytes));
    CK(cudaDeviceSynchronize());  // ⚠️ memset 走 legacy 默认流、核走 non-blocking 流:不夹同步会串序
    k_wp<<<g_wgrid, BLOCK, 0, s1>>>((uint4*)peer_b, (const uint4*)src_a, n16);
    k_rmw<<<g_rgrid, BLOCK, 0, s2>>>((uint4*)local_a, (const uint4*)src_b, n16);
    CK(cudaStreamSynchronize(s1)); CK(cudaStreamSynchronize(s2));
    uint4 hp = {}, hl = {};
    CK(cudaMemcpy(&hp, peer_b, sizeof(hp), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&hl, local_a, sizeof(hl), cudaMemcpyDeviceToHost));
    printf("  [verify] 对端首字 = 0x%08x(期望 0x5a5a5a5a,证明写腿落地)· "
           "本端首字 = 0x%08x(期望 0x3c3c3c3c,证明 RMW 落地)\n", hp.x, hl.x);
  }

  CK(cudaGraphExecDestroy(ge_par)); CK(cudaGraphDestroy(g_par));
  CK(cudaGraphExecDestroy(ge_hom)); CK(cudaGraphDestroy(g_hom));
  CK(cudaStreamSynchronize(s1)); CK(cudaStreamSynchronize(s2)); CK(cudaStreamSynchronize(s3));
  return 0;
}
