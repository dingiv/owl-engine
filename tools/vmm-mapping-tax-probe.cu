// ★ **映射路径的成本账**:VMM 显式授予 vs runtime(`cudaMalloc` + `cudaDeviceEnablePeerAccess`)
// (2026-10-19 立项;用户口径:"可不可以绕开 VMM 这个 API,你看一下这个 API 它的一个性能如何")
//
// ## 与 `vmm-peer-probe.cu` 的分工(两份都要留,别合并)
//   · `vmm-peer-probe.cu`(2026-10-18)= **单点定性**:同进程/同对卡/同核/同载荷,
//     只换"对端块是怎么来的" ⇒ 钉出"VMM 读腿 1-way 3.8–5.9%"这个**事实**;
//   · 本探针 = **记账面**:① VMM **API 自身**的一次性成本(与 runtime 对照);
//     ② **税是固定项还是比例项**(尺寸阶梯 10KB→32MiB);③ **写腿**(push 档的前置
//     —— 现役生产就是写腿!)的 RT/VMM 对照;④ **生产 AR 形状**两条协议各跑一遍。
//
// ## 为什么"税是固定还是比例"是决定性问题
//   现役生产 = **push**(写对端槽)⇒ 若税是比例项(≈4–6%),push 档每次 AR 都要吃;
//   若是固定项(每次访问一趟固定 µs),则在 5.24MB 档仍显著、在 decode 10KB 档
//   相对更贵 —— 两种结论指向**完全不同的刀**(换映射 vs 换协议)。
//
// ## 编译 / 运行
// ```bash
// nvcc -O3 -arch=sm_86 -o /tmp/vmm-tax tools/vmm-mapping-tax-probe.cu -lcuda
// /tmp/vmm-tax 2 3                      # 3080 对(默认尺寸阶梯)
// /tmp/vmm-tax 0 1                      # 3090 Ti 对
// TAX_SIZES=10240,5242880 TAX_K=200 /tmp/vmm-tax 2 3
// ```
// ⚠️ 纪律:跑前确认**目标卡空闲**(`nvidia-smi --query-compute-apps`);τ 本探针**只读写自己的
//   peer 块**,不碰别的会话的显存;VMM 块预算 ≤192MB/卡(BAR1−64MB,超了是响亮 OOM)。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__,                     \
              cudaGetErrorString(e_));                                                 \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

#define DK(x)                                                                          \
  do {                                                                                 \
    CUresult r_ = (x);                                                                 \
    if (r_ != CUDA_SUCCESS) {                                                          \
      const char* s_ = nullptr;                                                        \
      cuGetErrorName(r_, &s_);                                                         \
      fprintf(stderr, "DRIVER error %s:%d %s\n", __FILE__, __LINE__, s_ ? s_ : "?");   \
      exit(1);                                                                         \
    }                                                                                  \
  } while (0)

#define HDR 64ull   // 与 `comm/ar.rs::HEADER_BYTES` 同值
#define BLOCK 512   // 与生产核同(MAX_THREADS 8192 / BLOCK 512)
#define VEC 8

// ── 核体(与 vmm-peer-probe 同款:VEC 路展开 + 可选 system fence)──────────
template <bool FENCE>
__global__ void k_read_add(uint4* __restrict__ add_to, const uint4* __restrict__ read_from,
                           size_t n16) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  if (FENCE) __threadfence_system();
  for (size_t i = tid; i < n16; i += (size_t)VEC * stride) {
    uint4 v[VEC];
    bool ok[VEC];
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      const size_t j = i + (size_t)k * stride;
      ok[k] = j < n16;
      if (ok[k]) v[k] = read_from[j];
    }
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      if (ok[k]) {
        const size_t j = i + (size_t)k * stride;
        uint4 t = add_to[j];
        t.x += v[k].x; t.y += v[k].y; t.z += v[k].z; t.w += v[k].w;
        add_to[j] = t;
      }
    }
  }
  if (FENCE) __threadfence_system();
}

__global__ void k_write_only(uint4* __restrict__ write_to, size_t n16, unsigned seed) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const uint4 v = make_uint4(seed, seed ^ 0x5a5a5a5au, seed + 0x12345678u, 0xdeadbeefu ^ seed);
  for (size_t i = tid; i < n16; i += (size_t)VEC * stride) {
#pragma unroll
    for (int k = 0; k < VEC; ++k) {
      const size_t j = i + (size_t)k * stride;
      if (j < n16) write_to[j] = v;
    }
  }
}

// ── 计时:每臂跑 3 次取中位(µs/发)──────────────────────────────────────
static int g_grid1(uint64_t n16) {  // 小载荷 ⇒ 少块(与生产同律:10KB = grid 1)
  const uint64_t need = (n16 + (uint64_t)BLOCK * VEC - 1) / ((uint64_t)BLOCK * VEC);
  return (int)std::max<uint64_t>(1, std::min<uint64_t>(16, need));
}
struct TimeIt {
  cudaEvent_t a, b;
  /// ⚠️ **事件必须建在它自己设备的上下文里**(否则跨设备 record/elapsed 报无效句柄)
  void init(int ord) { CK(cudaSetDevice(ord)); CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
  template <typename F> double us(int k, F&& f) {  // k 次发射的中位 µs
    std::vector<double> r;
    for (int rep = 0; rep < 3; ++rep) {
      f();  // 暖机 1 发
      CK(cudaEventRecord(a));
      for (int t = 0; t < k; ++t) f();
      CK(cudaEventRecord(b));
      CK(cudaEventSynchronize(b));
      float ms = 0.f;
      CK(cudaEventElapsedTime(&ms, a, b));
      r.push_back((double)ms * 1e3 / (double)k);
    }
    std::sort(r.begin(), r.end());
    return r[1];
  }
};

// ── VMM 块(生产同款:一整个块 = 一次 create/map/setAccess/export)──────
struct VmmBlk { CUdeviceptr va = 0; CUmemGenericAllocationHandle h = 0; size_t size = 0; int fd = -1; };

static size_t gran_of(int ord) {
  CUmemAllocationProp p = {};
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  p.location.id = ord;
  size_t g = 0;
  DK(cuMemGetAllocationGranularity(&g, &p, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return g;
}

/// 逐步计时版:每一步耗时写进 `acc[](与生产 `vmm.rs::alloc_mapped` 同序列)
static VmmBlk vmm_create_timed(int ord, size_t bytes, double* acc) {
  CK(cudaSetDevice(ord));  // ★ 驱动调用绑定**当前上下文** ⇒ 必须自己钉设备(否则整块落在别的卡上)
  VmmBlk b;
  CUmemAllocationProp p = {};
  p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  p.location.id = ord;
  p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  const size_t g = gran_of(ord);
  b.size = ((bytes + g - 1) / g) * g;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto step = [&](int slot, auto&& fn) {
    CK(cudaEventRecord(e0));
    fn();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (acc) acc[slot] += ms * 1e3;
  };
  step(0, [&] { DK(cuMemAddressReserve(&b.va, b.size, g, 0, 0)); });
  step(1, [&] { DK(cuMemCreate(&b.h, b.size, &p, 0)); });
  step(2, [&] { DK(cuMemMap(b.va, b.size, 0, b.h, 0)); });
  step(3, [&] {
    CUmemAccessDesc d = {};
    d.location = p.location;
    d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DK(cuMemSetAccess(b.va, b.size, &d, 1));
  });
  step(4, [&] { DK(cuMemExportToShareableHandle(&b.fd, b.h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0)); });
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  return b;
}

static CUdeviceptr vmm_import_timed(int ord, int fd, size_t size, bool rw, double* acc) {
  CK(cudaSetDevice(ord));  // ★ 同上:导入的 VA 必须落在 `ord` 的地址空间
  CUmemGenericAllocationHandle h2 = 0;
  CUdeviceptr va = 0;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto step = [&](int slot, auto&& fn) {
    CK(cudaEventRecord(e0));
    fn();
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    if (acc) acc[slot] += ms * 1e3;
  };
  step(5, [&] {
    DK(cuMemImportFromShareableHandle(&h2, (void*)(intptr_t)fd,
                                      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  });
  step(6, [&] { DK(cuMemAddressReserve(&va, size, 0, 0, 0)); });
  step(7, [&] { DK(cuMemMap(va, size, 0, h2, 0)); });
  step(8, [&] {
    CUmemAccessDesc d = {};
    d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    d.location.id = ord;
    d.flags = rw ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE : CU_MEM_ACCESS_FLAGS_PROT_READ;
    DK(cuMemSetAccess(va, size, &d, 1));
  });
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  return va;
}

static void report(const char* tag, size_t bytes, double us) {
  printf("  %-18s %9.2f us %8.2f GB/s\n", tag, us, (double)bytes / (us * 1e-6) / 1e9);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "用法: %s <ordA> <ordB> [bytes...]\n", argv[0]);
    return 2;
  }
  const int oa = atoi(argv[1]), ob = atoi(argv[2]);
  const int k = getenv("TAX_K") ? atoi(getenv("TAX_K")) : 200;
  std::vector<size_t> sizes;
  if (argc > 3) {
    for (int i = 3; i < argc; ++i) sizes.push_back((size_t)atoll(argv[i]));
  } else if (getenv("TAX_SIZES")) {
    char* s = strdup(getenv("TAX_SIZES"));
    for (char* p = strtok(s, ","); p; p = strtok(nullptr, ",")) sizes.push_back((size_t)atoll(p));
  } else {
    sizes = {10240u, 131072u, 1310720u, 5242880u, 33554432u};  // decode / 128K / 1.31M / prefill / slot 上限
  }
  DK(cuInit(0));
  printf("═══ 映射路径成本账(VMM vs runtime)· A=ord%d B=ord%d · K=%d ═══\n", oa, ob, k);
  printf("[env] VMM 粒度 A=%zu B=%zu | BAR1 预算纪律 ≤192MB/卡\n\n", gran_of(oa), gran_of(ob));

  // ════════════════ 表 A:API 一次性成本(64MiB = 生产 staging 每 rank 的账)════
  {
    const size_t S = 64ull << 20;
    printf("── 表 A:一次性 API 成本(64MiB;生产 staging 同规模)──\n");
    double acc[10] = {0};
    int REP = 3;
    for (int r = 0; r < REP; ++r) {
      CK(cudaSetDevice(oa));
      VmmBlk a = vmm_create_timed(oa, S, acc);
      CK(cudaSetDevice(ob));
      VmmBlk b = vmm_create_timed(ob, S, acc);
      const CUdeviceptr x_ab = vmm_import_timed(oa, b.fd, b.size, true, acc);  // A 导入 B(RW)
      const CUdeviceptr x_ba = vmm_import_timed(ob, a.fd, a.size, true, acc);
      // 释放(**别名也必须 unmap** —— peer 导入占的是导入方的 BAR1 窗口,
      // 漏一个就漏 64MB;3080 的 BAR1 只有 256MB,3 次重复就能把窗口吃光 ⇒
      // 下一次 import 报 `out of memory`。实测踩到一次。)
      cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      CK(cudaEventRecord(e0));
      CK(cudaSetDevice(oa)); DK(cuMemUnmap(x_ab, b.size));
      CK(cudaSetDevice(ob)); DK(cuMemUnmap(x_ba, a.size));
      CK(cudaSetDevice(oa));
      DK(cuMemUnmap(a.va, a.size)); DK(cuMemRelease(a.h)); DK(cuMemAddressFree(a.va, a.size));
      CK(cudaSetDevice(ob));
      DK(cuMemUnmap(b.va, b.size)); DK(cuMemRelease(b.h)); DK(cuMemAddressFree(b.va, b.size));
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1)); acc[9] += ms * 1e3;
      CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
      CK(cudaSetDevice(oa));
    }
    const char* names[10] = {"cuMemAddressReserve", "cuMemCreate", "cuMemMap", "cuMemSetAccess",
                             "cuMemExportToShareableHandle", "cuMemImportFromShareableHandle",
                             "cuMemAddressReserve(import)", "cuMemMap(import)",
                             "cuMemSetAccess(peer,RW)", "释放(unmap+release+free)×2"};
    double tot = 0;
    for (int i = 0; i < 10; ++i) { printf("  %-30s %8.1f us\n", names[i], acc[i] / REP); tot += acc[i] / REP; }
    printf("  %-30s %8.1f us  (%.2f ms ⇒ 装配期一次性)\n", "VMM 合计(两卡两块)", tot, tot / 1e3);

    // runtime 对照(事件是**每设备一个上下文** ⇒ 两卡分别计时再相加;跨设备单对事件是非法的)
    double rt = 0;
    for (int r = 0; r < REP; ++r) {
      cudaEvent_t e0, e1;
      void *pa = nullptr, *pb = nullptr;
      CK(cudaSetDevice(oa)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      CK(cudaEventRecord(e0)); CK(cudaMalloc(&pa, S)); CK(cudaDeviceEnablePeerAccess(ob, 0));
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float m1 = 0.f; CK(cudaEventElapsedTime(&m1, e0, e1)); rt += m1 * 1e3;
      CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
      CK(cudaSetDevice(ob)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
      CK(cudaEventRecord(e0)); CK(cudaMalloc(&pb, S)); CK(cudaDeviceEnablePeerAccess(oa, 0));
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float m2 = 0.f; CK(cudaEventElapsedTime(&m2, e0, e1)); rt += m2 * 1e3;
      CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
      CK(cudaSetDevice(oa)); CK(cudaFree(pa)); CK(cudaDeviceDisablePeerAccess(ob));
      CK(cudaSetDevice(ob)); CK(cudaFree(pb)); CK(cudaDeviceDisablePeerAccess(oa));
    }
    printf("  %-30s %8.1f us  (cudaMalloc×2 + enablePeerAccess×2)\n\n", "runtime 合计", rt / REP);
  }

  // ════════════════ 表 B/C:访问税(尺寸阶梯 × 读/写 × 单边/双边 × RT/VMM)═══
  cudaStream_t sa, sb;
  CK(cudaSetDevice(oa)); CK(cudaStreamCreate(&sa));
  CK(cudaSetDevice(ob)); CK(cudaStreamCreate(&sb));
  TimeIt ta, tb;
  ta.init(oa); tb.init(ob);

  printf("── 表 B:访问税 × 尺寸(单边 1way / 双边 2way;RT = cudaMalloc+enable)──\n");
  printf("  %-10s %-22s %9s %9s %8s\n", "载荷", "臂", "RT µs", "VMM µs", "税%");
  for (size_t bytes : sizes) {
    const size_t n16 = bytes / 16;
    const int g = g_grid1(n16);
    // 分配:VMM 两块 + 交叉 RW 别名;RT 两块 + enable;本地 dst 两块
    CK(cudaSetDevice(ob)); VmmBlk vb = vmm_create_timed(ob, bytes + HDR, nullptr);
    CK(cudaSetDevice(oa)); VmmBlk va = vmm_create_timed(oa, bytes + HDR, nullptr);
    const CUdeviceptr vmm_b_on_a = vmm_import_timed(oa, vb.fd, vb.size, true, nullptr);
    const CUdeviceptr vmm_a_on_b = vmm_import_timed(ob, va.fd, va.size, true, nullptr);
    // ★ **只读导入**(与生产 v1 档同款:§2.2 导入侧授 PROT_READ;push 档才改 RW)
    //   专项对照 → **同一对卡、同一载荷、同一个核,只换导入的访问位**。
    CK(cudaSetDevice(ob)); VmmBlk vbr = vmm_create_timed(ob, bytes + HDR, nullptr);
    CK(cudaSetDevice(oa)); VmmBlk var = vmm_create_timed(oa, bytes + HDR, nullptr);
    const CUdeviceptr vmmr_b_on_a = vmm_import_timed(oa, vbr.fd, vbr.size, false, nullptr);
    const CUdeviceptr vmmr_a_on_b = vmm_import_timed(ob, var.fd, var.size, false, nullptr);
    CK(cudaSetDevice(ob)); CK(cudaMemset((void*)(vbr.va + HDR), 0x11, bytes));
    CK(cudaSetDevice(oa)); CK(cudaMemset((void*)(var.va + HDR), 0x33, bytes));
    void *rt_a = nullptr, *rt_b = nullptr;
    CK(cudaSetDevice(oa)); CK(cudaMalloc(&rt_a, bytes)); CK(cudaMemset(rt_a, 0x44, bytes));
    CK(cudaSetDevice(ob)); CK(cudaMalloc(&rt_b, bytes)); CK(cudaMemset(rt_b, 0x22, bytes));
    CK(cudaSetDevice(oa)); CK(cudaDeviceEnablePeerAccess(ob, 0));
    CK(cudaSetDevice(ob)); CK(cudaDeviceEnablePeerAccess(oa, 0));
    void *dst_a = nullptr, *dst_b = nullptr;
    CK(cudaSetDevice(oa)); CK(cudaMalloc(&dst_a, bytes + HDR)); CK(cudaMemset(dst_a, 0, bytes));
    CK(cudaSetDevice(ob)); CK(cudaMalloc(&dst_b, bytes + HDR)); CK(cudaMemset(dst_b, 0, bytes));
    CK(cudaSetDevice(ob)); CK(cudaMemset((void*)(vb.va + HDR), 0x11, bytes));
    CK(cudaSetDevice(oa)); CK(cudaMemset((void*)(va.va + HDR), 0x33, bytes));

    char sz[32];
    snprintf(sz, sizeof sz, "%.2f KB", (double)bytes / 1024.0);
    if (bytes >= (1u << 20)) snprintf(sz, sizeof sz, "%.2f MiB", (double)bytes / 1048576.0);

    // 1way 读
    CK(cudaSetDevice(oa));
    double rt_r1 = ta.us(k, [&]{ k_read_add<false><<<g, BLOCK, 0, sa>>>((uint4*)dst_a, (const uint4*)rt_b, n16); });
    double vm_r1 = ta.us(k, [&]{ k_read_add<false><<<g, BLOCK, 0, sa>>>((uint4*)dst_a, (const uint4*)(vmm_b_on_a + HDR), n16); });
    // 1way 写
    double rt_w1 = ta.us(k, [&]{ k_write_only<<<g, BLOCK, 0, sa>>>((uint4*)rt_b, n16, 0xA5A5u); });
    double vm_w1 = ta.us(k, [&]{ k_write_only<<<g, BLOCK, 0, sa>>>((uint4*)(vmm_b_on_a + HDR), n16, 0xA5A5u); });
    printf("  %-10s %-22s %9.2f %9.2f %7.1f%%\n", sz, "读 1way", rt_r1, vm_r1, 100.0 * (vm_r1 - rt_r1) / rt_r1);
    printf("  %-10s %-22s %9.2f %9.2f %7.1f%%\n", sz, "写 1way", rt_w1, vm_w1, 100.0 * (vm_w1 - rt_w1) / rt_w1);
    // fence 档(生产核体同款)
    double vm_rf = ta.us(k, [&]{ k_read_add<true><<<g, BLOCK, 0, sa>>>((uint4*)dst_a, (const uint4*)(vmm_b_on_a + HDR), n16); });
    double rt_rf = ta.us(k, [&]{ k_read_add<true><<<g, BLOCK, 0, sa>>>((uint4*)dst_a, (const uint4*)rt_b, n16); });
    printf("  %-10s %-22s %9.2f %9.2f %7.1f%%\n", sz, "读+fence 1way", rt_rf, vm_rf, 100.0 * (vm_rf - rt_rf) / rt_rf);

    // 2way(两卡同时;取两卡时延的较大者)
    auto two_way_read = [&](bool vmm) {
      CK(cudaEventRecord(ta.a, sa));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(tb.a, sb));
      for (int t = 0; t < k; ++t) {
        CK(cudaSetDevice(oa));
        const uint4* src = vmm ? (const uint4*)(vmm_b_on_a + HDR) : (const uint4*)rt_b;
        k_read_add<false><<<g, BLOCK, 0, sa>>>((uint4*)dst_a, src, n16);
        CK(cudaSetDevice(ob));
        const uint4* src2 = vmm ? (const uint4*)(vmm_a_on_b + HDR) : (const uint4*)rt_a;
        k_read_add<false><<<g, BLOCK, 0, sb>>>((uint4*)dst_b, src2, n16);
      }
      CK(cudaSetDevice(oa)); CK(cudaEventRecord(ta.b, sa)); CK(cudaEventSynchronize(ta.b));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(tb.b, sb)); CK(cudaEventSynchronize(tb.b));
      float m1 = 0.f, m2 = 0.f;
      CK(cudaSetDevice(oa)); CK(cudaEventElapsedTime(&m1, ta.a, ta.b));
      CK(cudaSetDevice(ob)); CK(cudaEventElapsedTime(&m2, tb.a, tb.b));
      return (double)std::max(m1, m2) * 1e3 / (double)k;
    };
    auto two_way_write = [&](bool vmm) {
      CK(cudaEventRecord(ta.a, sa));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(tb.a, sb));
      for (int t = 0; t < k; ++t) {
        CK(cudaSetDevice(oa));
        uint4* dst = vmm ? (uint4*)(vmm_b_on_a + HDR) : (uint4*)rt_b;
        k_write_only<<<g, BLOCK, 0, sa>>>(dst, n16, 0xA5A5u);
        CK(cudaSetDevice(ob));
        uint4* dst2 = vmm ? (uint4*)(vmm_a_on_b + HDR) : (uint4*)rt_a;
        k_write_only<<<g, BLOCK, 0, sb>>>(dst2, n16, 0x5A5Au);
      }
      CK(cudaSetDevice(oa)); CK(cudaEventRecord(ta.b, sa)); CK(cudaEventSynchronize(ta.b));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(tb.b, sb)); CK(cudaEventSynchronize(tb.b));
      float m1 = 0.f, m2 = 0.f;
      CK(cudaSetDevice(oa)); CK(cudaEventElapsedTime(&m1, ta.a, ta.b));
      CK(cudaSetDevice(ob)); CK(cudaEventElapsedTime(&m2, tb.a, tb.b));
      return (double)std::max(m1, m2) * 1e3 / (double)k;
    };
    double rt_r2 = two_way_read(false), vm_r2 = two_way_read(true);
    double rt_w2 = two_way_write(false), vm_w2 = two_way_write(true);
    printf("  %-10s %-22s %9.2f %9.2f %7.1f%%\n", sz, "读 2way", rt_r2, vm_r2, 100.0 * (vm_r2 - rt_r2) / rt_r2);
    printf("  %-10s %-22s %9.2f %9.2f %7.1f%%\n", sz, "写 2way", rt_w2, vm_w2, 100.0 * (vm_w2 - rt_w2) / rt_w2);

    // ★★ 导入访问位 A/B:同一对卡、同一载荷、同一个核,只换 `cuMemSetAccess` 的 flags
    CK(cudaSetDevice(oa));
    double vm_r1_ro = ta.us(k, [&]{ k_read_add<false><<<g, BLOCK, 0, sa>>>((uint4*)dst_a, (const uint4*)(vmmr_b_on_a + HDR), n16); });
    printf("  %-10s %-22s %9.2f %9.2f %7.1f%%   ← 只读导入(生产 v1 档)\n", sz, "VMM 读: READ 导入", rt_r1, vm_r1_ro, 100.0 * (vm_r1_ro - rt_r1) / rt_r1);
    printf("  %-10s %-22s %9.2f %9.2f %7.1f%%   ← RW 导入(生产 push 档)\n", sz, "VMM 读: RW 导入", rt_r1, vm_r1, 100.0 * (vm_r1 - rt_r1) / rt_r1);

    // ── 表 C:生产 AR 形状(两条协议)────────────────────────────────
    //   v1  = [读对端槽 + 本地累加]              (读腿主导)
    //   push= [写对端槽] → [读本端槽 + 本地累加] (写腿主导;现役生产)
    auto push_full = [&](bool vmm) {
      CK(cudaSetDevice(oa));
      return ta.us(k, [&]{
        uint4* peer = vmm ? (uint4*)(vmm_b_on_a + HDR) : (uint4*)rt_b;
        k_write_only<<<g, BLOCK, 0, sa>>>(peer, n16, 0xA5A5u);
        k_read_add<false><<<g, BLOCK, 0, sa>>>((uint4*)dst_a, (const uint4*)dst_a, n16);  // 本端槽腿
      });
    };
    double pf_v = push_full(true), pf_r = push_full(false);
    printf("  %-10s %-22s %9.2f %9.2f %7.1f%%   ← 现役 push\n", sz, "AR: push(写腿)", pf_r, pf_v, 100.0 * (pf_v - pf_r) / pf_r);
    printf("  %-10s %-22s %9.2f %9.2f %7.1f%%   ← v1(读腿)\n", sz, "AR: v1(读腿)", rt_r1, vm_r1, 100.0 * (vm_r1 - rt_r1) / rt_r1);

    // 清理(逐档;VMM 用 unmap+release+free,RT 用 cudaFree)
    // ⚠️ 别名先 unmap(见上;peer 导入占 BAR1 窗口)
    CUdeviceptr a1 = vmm_b_on_a, a2 = vmm_a_on_b, a3 = vmmr_b_on_a, a4 = vmmr_a_on_b;
    CK(cudaSetDevice(oa)); DK(cuMemUnmap(a1, vb.size)); DK(cuMemUnmap(a3, vbr.size));
    CK(cudaSetDevice(ob)); DK(cuMemUnmap(a2, va.size)); DK(cuMemUnmap(a4, var.size));
    CK(cudaSetDevice(oa)); CK(cudaFree(dst_a)); CK(cudaFree(rt_a)); CK(cudaDeviceDisablePeerAccess(ob));
    CK(cudaSetDevice(ob)); CK(cudaFree(dst_b)); CK(cudaFree(rt_b)); CK(cudaDeviceDisablePeerAccess(oa));
    CK(cudaSetDevice(oa)); DK(cuMemUnmap(va.va, va.size)); DK(cuMemRelease(va.h)); DK(cuMemAddressFree(va.va, va.size));
    CK(cudaSetDevice(oa)); DK(cuMemUnmap(var.va, var.size)); DK(cuMemRelease(var.h)); DK(cuMemAddressFree(var.va, var.size));
    CK(cudaSetDevice(ob)); DK(cuMemUnmap(vb.va, vb.size)); DK(cuMemRelease(vb.h)); DK(cuMemAddressFree(vb.va, vb.size));
    CK(cudaSetDevice(ob)); DK(cuMemUnmap(vbr.va, vbr.size)); DK(cuMemRelease(vbr.h)); DK(cuMemAddressFree(vbr.va, vbr.size));
    CK(cudaSetDevice(oa));
  }

  CK(cudaSetDevice(oa)); CK(cudaStreamDestroy(sa)); CK(cudaEventDestroy(ta.a)); CK(cudaEventDestroy(ta.b));
  CK(cudaSetDevice(ob)); CK(cudaStreamDestroy(sb)); CK(cudaEventDestroy(tb.a)); CK(cudaEventDestroy(tb.b));
  printf("\n[tax] done(税%% = (VMM-RT)/RT;负值 = VMM 更快 —— 以实测为准,别预设方向)\n");
  return 0;
}
