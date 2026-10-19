// ============================================================================
// owl TP2 —— **AR 结构对比台**(2026-10-19;§54/§55 的判据)
//
// ## 立案
//
// §54 读 vLLM 源码得到的结构差别:**vLLM = [CE 拷贝(流序、零同步语义)] + [1 个融合核:
// barrier_at_start → 读对端缓冲 → 加 → barrier_at_end]**;我们 v1 = **[stage 核(拷贝 +
// 系统级栅栏 + 发布 produced)] + [reduce 核(等 produced + 读对端槽 + 加)]**。
// §55.4 据此提出改造;本节就是它的**分母**。
//
// ## 三种结构(同载荷、同几何、同两卡、图内回放、host 只出一次 launch)
//
// | 臂 | 结构 |
// |---|---|
// | **A(现役 v1)** | `k_stage`(本端拷贝 + `__threadfence_system()` + `produced++`) → `k_reduce`(等 `peer.produced` + `out += peer_slot`)|
// | **B(vLLM 同构)** | `cudaMemcpyPeerAsync`(CE,**流序**)→ `k_fused`(到场 barrier → `out += peer_slot`)|
// | **C(B + 弱 flag)** | 同 B,但 flag 用 `st.release.sys` / `ld.acquire.sys`,末次 barrier 用 relaxed |
//
// ## 判据
//
// `B/C` 相对 `A` 省多少 µs/次;并**校验数值**(`out` 必须等于两边之和)——
// ⚠️ B/C 依赖"CE 写与 SM 写之间的可见性由流序保证",这是 vLLM 的做法;
// **本台带落地校验,若对不上会直接报错**(不静默)。
//
// 编译:`nvcc -O3 -arch=sm_86 -o /tmp/ar-struct-probe tools/ar-struct-probe.cu`
// 运行:`/tmp/ar-struct-probe <ordA> <ordB> [bytes..]`(卡序 = CUDA ordinal;跑前确认机器空闲)
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));  \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

#define BLK 512

// 头部(两侧各一块,互为 peer 映射):start/end 各两格(= 2 rank 的到达计数)
struct Hdr {
  // ★ vLLM 结构:**本端自己的 ticket(只在本地读写)** + **对端把它的 flag 写进来的格子**
  unsigned long long ticket;
  unsigned long long peer_flag;
  unsigned long long end_ticket;
  unsigned long long end_peer;
  unsigned long long prod;  // A 档:produced
  unsigned long long cons;  // A 档:consumed
};

__device__ __forceinline__ void fence_acqrel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ unsigned long long ld_acquire_sys(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(unsigned long long* p, unsigned long long v) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_vol(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ void st_vol(unsigned long long* p, unsigned long long v) {
  asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(p), "l"(v));
}
__device__ __forceinline__ unsigned long long ld_relaxed_sys(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}

// ── A①:stage —— 本端拷贝 → fence → 发布 produced ────────────────────────────
__global__ void k_stage(uint4* my_slot, const uint4* local, size_t n16, Hdr* my_hdr) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride)
    my_slot[i] = local[i];
  __threadfence_system();
  if (blockIdx.x == 0 && threadIdx.x == 0) st_vol(&my_hdr->prod, ld_vol(&my_hdr->prod) + 1ull);
}

// ── A②:reduce —— 等对端 produced → out += 对端槽 ────────────────────────────
__global__ void k_reduce(uint4* out, const uint4* peer_slot, const Hdr* peer_hdr, Hdr* my_hdr,
                         size_t n16) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const unsigned long long want = ld_vol(&my_hdr->cons) + 1ull;
    st_vol(&my_hdr->cons, want);
    while (ld_vol(&peer_hdr->prod) < want) {}
  }
  __syncthreads();
  __threadfence_system();
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
    uint4 a = out[i];
    const uint4 b = peer_slot[i];
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    out[i] = a;
  }
}

// ── B/C:融合核 —— 到场 barrier → out += 对端槽 ──────────────────────────────
// `weak=false`(B)= 用 fence.sc.sys(最重)+ volatile;`weak=true`(C)= release/acquire,
// **末次 barrier 用 relaxed**(vLLM 注释:最终同步不需要可见性保证)。
template <bool WEAK, bool IS_LAST>
__global__ void k_fused(uint4* out, const uint4* peer_slot, Hdr* my_hdr, Hdr* peer_hdr, int rank,
                        size_t n16) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const unsigned long long flag = ld_vol(&my_hdr->ticket) + 1ull;  // ★ 本端 ticket
    st_vol(&my_hdr->ticket, flag);
    if (WEAK) {
      st_release_sys(&peer_hdr->peer_flag, flag);          // 发布到对端
      while (ld_acquire_sys(&my_hdr->peer_flag) < flag) {} // 等对端发布到我这儿
    } else {
      __threadfence_system();
      st_vol(&peer_hdr->peer_flag, flag);
      __threadfence_system();
      while (ld_vol(&my_hdr->peer_flag) < flag) {}
    }
  }
  __syncthreads();
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
    uint4 a = out[i];
    const uint4 b = peer_slot[i];
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    out[i] = a;
  }
  __syncthreads();
  if (blockIdx.x == 0 && threadIdx.x == 0 && !IS_LAST) {
    const unsigned long long flag = ld_vol(&my_hdr->end_ticket) + 1ull;
    st_vol(&my_hdr->end_ticket, flag);
    // ★ vLLM 洞察:末次 barrier 用 relaxed(最终同步不需要可见性保证)
    st_vol(&peer_hdr->end_peer, flag);
    while (ld_vol(&my_hdr->end_peer) < flag) {}
  }
  (void)rank;
}

static double ms(cudaEvent_t a, cudaEvent_t b) {
  float v = 0.f; CK(cudaEventElapsedTime(&v, a, b)); return (double)v;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);  // 卡死时也要看得见进度
  if (argc < 4) { fprintf(stderr, "用法: %s <ordA> <ordB> <bytes> [bytes...]\n", argv[0]); return 2; }
  const int oa = atoi(argv[1]), ob = atoi(argv[2]);
  int can = 0; CK(cudaDeviceCanAccessPeer(&can, oa, ob));
  if (!can) { fprintf(stderr, "无 P2P\n"); return 3; }
  CK(cudaSetDevice(oa)); CK(cudaDeviceEnablePeerAccess(ob, 0));
  CK(cudaSetDevice(ob)); CK(cudaDeviceEnablePeerAccess(oa, 0));
  const int N = getenv("ARQ_N") ? atoi(getenv("ARQ_N")) : 32;   // 每张图的 AR 次数
  const int REP = getenv("ARQ_REPS") ? atoi(getenv("ARQ_REPS")) : 5;
  const int CAP = getenv("ARQ_BLOCKS") ? atoi(getenv("ARQ_BLOCKS")) : 16;

  for (int ai = 3; ai < argc; ++ai) {
    const size_t bytes = (size_t)atoll(argv[ai]);
    const size_t n16 = bytes / 16;
    printf("\n[arq] bytes=%zu(%.2f MB) N=%d/graph blocks=%d\n", bytes, (double)bytes / 1048576.0, N, CAP);

    // 两侧各一块:local(私有)、slot(对称)、out;hdr 在 slot 之外的独立块
    void *loc_a = nullptr, *slot_a = nullptr, *out_a = nullptr, *hdr_a = nullptr;
    void *loc_b = nullptr, *slot_b = nullptr, *out_b = nullptr, *hdr_b = nullptr;
    CK(cudaSetDevice(oa));
    CK(cudaMalloc(&loc_a, bytes)); CK(cudaMemset(loc_a, 0x11, bytes));
    CK(cudaMalloc(&slot_a, bytes)); CK(cudaMemset(slot_a, 0, bytes));
    CK(cudaMalloc(&out_a, bytes)); CK(cudaMemset(out_a, 0, bytes));
    CK(cudaMalloc(&hdr_a, sizeof(Hdr))); CK(cudaMemset(hdr_a, 0, sizeof(Hdr)));
    CK(cudaSetDevice(ob));
    CK(cudaMalloc(&loc_b, bytes)); CK(cudaMemset(loc_b, 0x22, bytes));
    CK(cudaMalloc(&slot_b, bytes)); CK(cudaMemset(slot_b, 0, bytes));
    CK(cudaMalloc(&out_b, bytes)); CK(cudaMemset(out_b, 0, bytes));
    CK(cudaMalloc(&hdr_b, sizeof(Hdr))); CK(cudaMemset(hdr_b, 0, sizeof(Hdr)));
    CK(cudaSetDevice(oa));

    const int blocks = (int)((n16 + BLK - 1) / BLK);      // 与 vLLM 同式:按尺寸定块数
    const int g = blocks < CAP ? blocks : CAP;

    auto reset = [&] {
      CK(cudaSetDevice(oa));
      CK(cudaMemsetAsync(out_a, 0, bytes)); CK(cudaMemcpyAsync(out_a, loc_a, bytes, cudaMemcpyDeviceToDevice));
      CK(cudaMemsetAsync(hdr_a, 0, sizeof(Hdr)));
      CK(cudaSetDevice(ob));
      CK(cudaMemsetAsync(out_b, 0, bytes)); CK(cudaMemcpyAsync(out_b, loc_b, bytes, cudaMemcpyDeviceToDevice));
      CK(cudaMemsetAsync(hdr_b, 0, sizeof(Hdr)));
      CK(cudaSetDevice(oa));
    };
    CK(cudaDeviceSynchronize());
    reset();
    CK(cudaDeviceSynchronize());

    cudaStream_t s1, s2;
    CK(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking));
    CK(cudaSetDevice(ob)); CK(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking));
    CK(cudaSetDevice(oa));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));

    // ── 统一形状:**每设备一张图 = N × [第一段; 第二段] 交替**(= 生产形状,不批)──────
    auto build_and_time = [&](const char* nm, bool variantA, int mode) {
      cudaGraph_t ga; cudaGraphExec_t ge;
      CK(cudaStreamBeginCapture(s1, cudaStreamCaptureModeThreadLocal));
      for (int i = 0; i < N; ++i) {
        if (variantA) {
          k_stage<<<g, BLK, 0, s1>>>((uint4*)slot_a, (const uint4*)loc_a, n16, (Hdr*)hdr_a);
          k_reduce<<<g, BLK, 0, s1>>>((uint4*)out_a, (const uint4*)slot_b, (const Hdr*)hdr_b,
                                      (Hdr*)hdr_a, n16);
        } else {
          CK(cudaMemcpyAsync(slot_a, loc_a, bytes, cudaMemcpyDeviceToDevice, s1));
          if (mode == 0)      k_fused<false, false><<<g, BLK, 0, s1>>>((uint4*)out_a, (const uint4*)slot_b, (Hdr*)hdr_a, (Hdr*)hdr_b, 0, n16);
          else if (mode == 1) k_fused<true,  false><<<g, BLK, 0, s1>>>((uint4*)out_a, (const uint4*)slot_b, (Hdr*)hdr_a, (Hdr*)hdr_b, 0, n16);
          else                k_fused<true,  true ><<<g, BLK, 0, s1>>>((uint4*)out_a, (const uint4*)slot_b, (Hdr*)hdr_a, (Hdr*)hdr_b, 0, n16);
        }
      }
      CK(cudaStreamEndCapture(s1, &ga)); CK(cudaGraphInstantiate(&ge, ga, 0));
      cudaGraph_t gb; cudaGraphExec_t geb;
      CK(cudaSetDevice(ob));
      CK(cudaStreamBeginCapture(s2, cudaStreamCaptureModeThreadLocal));
      for (int i = 0; i < N; ++i) {
        if (variantA) {
          k_stage<<<g, BLK, 0, s2>>>((uint4*)slot_b, (const uint4*)loc_b, n16, (Hdr*)hdr_b);
          k_reduce<<<g, BLK, 0, s2>>>((uint4*)out_b, (const uint4*)slot_a, (const Hdr*)hdr_a,
                                      (Hdr*)hdr_b, n16);
        } else {
          CK(cudaMemcpyAsync(slot_b, loc_b, bytes, cudaMemcpyDeviceToDevice, s2));
          if (mode == 0)      k_fused<false, false><<<g, BLK, 0, s2>>>((uint4*)out_b, (const uint4*)slot_a, (Hdr*)hdr_b, (Hdr*)hdr_a, 1, n16);
          else if (mode == 1) k_fused<true,  false><<<g, BLK, 0, s2>>>((uint4*)out_b, (const uint4*)slot_a, (Hdr*)hdr_b, (Hdr*)hdr_a, 1, n16);
          else                k_fused<true,  true ><<<g, BLK, 0, s2>>>((uint4*)out_b, (const uint4*)slot_a, (Hdr*)hdr_b, (Hdr*)hdr_a, 1, n16);
        }
      }
      CK(cudaStreamEndCapture(s2, &gb)); CK(cudaGraphInstantiate(&geb, gb, 0));
      CK(cudaSetDevice(oa));
      for (int r = 0; r < 3; ++r) { CK(cudaGraphLaunch(ge, s1)); CK(cudaGraphLaunch(geb, s2)); }
      CK(cudaDeviceSynchronize());
      reset(); CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(e0, s1));
      for (int r = 0; r < REP; ++r) { CK(cudaGraphLaunch(ge, s1)); CK(cudaGraphLaunch(geb, s2)); }
      CK(cudaEventRecord(e1, s1)); CK(cudaEventSynchronize(e1));
      const double us = ms(e0, e1) * 1e3 / (double)(N * REP);
      CK(cudaStreamSynchronize(s2));
      uint4 v = {}; CK(cudaMemcpy(&v, out_a, sizeof(v), cudaMemcpyDeviceToHost));
      printf("  %-42s %9.2f µs/AR   %6.0f MB/s   out[0]=0x%08x\n", nm, us,
             (double)bytes / us / 1e3, v.x);
      CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(ga));
      CK(cudaGraphExecDestroy(geb)); CK(cudaGraphDestroy(gb));
      return us;
    };
    const double ua = build_and_time("A 现役 v1(两核 + produced 协议)", true, 0);
    const double ub = build_and_time("B vLLM 同构(CE + 融合核, sc.sys)", false, 0);
    const double uc = build_and_time("C 同 B + release/acquire", false, 1);
    const double ud = build_and_time("D 同 C + 末次 barrier relaxed", false, 2);
    printf("  ⇒ **B−A=%+.2f  C−A=%+.2f  D−A=%+.2f µs/AR**(负 = 比现役快)\n", ub - ua, uc - ua, ud - ua);

    // ── 落地校验:A 与 D 的 out 都必须 = 0x11 + 0x22(逐 uint4)──────────────
    {
      CK(cudaSetDevice(oa));
      uint4 h = {};
      CK(cudaMemcpy(&h, (char*)out_a + (bytes / 2) - (bytes / 2 % 16), sizeof(h), cudaMemcpyDeviceToHost));
      uint4 hb = {};
      CK(cudaSetDevice(ob));
      CK(cudaMemcpy(&hb, (char*)out_b + (bytes / 2) - (bytes / 2 % 16), sizeof(hb), cudaMemcpyDeviceToHost));
      CK(cudaSetDevice(oa));
      printf("  [verify] A 侧 out 首字=0x%08x · B 侧 out 首字=0x%08x(期望均 = 0x33333333)\n", h.x, hb.x);
    }
    CK(cudaFree(loc_a)); CK(cudaFree(slot_a)); CK(cudaFree(out_a)); CK(cudaFree(hdr_a));
    CK(cudaSetDevice(ob)); CK(cudaFree(loc_b)); CK(cudaFree(slot_b)); CK(cudaFree(out_b)); CK(cudaFree(hdr_b));
    CK(cudaSetDevice(oa));
  }
  return 0;
}
