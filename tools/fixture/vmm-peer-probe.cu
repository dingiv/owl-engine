// ★ **映射路径 A/B 探针**(2026-10-18;TP2 施工设计 §36)
//
// ## 立案现场(同会话读数,3080 对 2/3,f16,5.24MB,K=200,设备时)
//
// | 臂 | 探针 | µs | GB/s |
// |---|---|---:|---:|
// | `R`(SM 读对端 + 本地加,单边)| `ar_channel_bench` | 527.5 | **9.93** |
// | `RR`(同,双边)| 同上 | 566.7 | 9.25 |
// | `SM-read-1way`(同核,但映射走 runtime)| `p2p-wr-probe` | **405.4** | **12.93** |
// | `CE-pull-1way` | `ar_channel_bench`(VMM)| 419.3 | 12.50 |
// | `CE-pull-1way` | `p2p-wr-probe`(runtime)| 405.1 | 12.94 |
//
// ⇒ **同一个核、同一对卡、同一载荷,换掉"对端内存是怎么映射进来的"这一步,
// SM 读从 9.93 变成 12.93 GB/s(+30%)。** 两个程序唯一的差别是映射路径:
//   · owl 引擎:`cuMemCreate → cuMemExportToShareableHandle(fd) →
//     cuMemImportFromShareableHandle → cuMemMap → cuMemSetAccess(PROT_READ)`
//     (`crates/backends/cuda/src/vmm.rs`;分配粒度 2MiB,PINNED + POSIX_FD);
//   · 本探针 runtime 臂:`cudaMalloc` + `cudaDeviceEnablePeerAccess`。
//
// **本程序把那条差别单独拎出来重做一遍**:同一进程、同一对卡、同一个核、
// 同一载荷与偏移,只换"对端块是怎么来的"。
//
// ## 臂
//
// | 臂 | 对端块 | 访问 |
// |---|---|---|
// | `RT-read` | `cudaMalloc`@对端 + `cudaDeviceEnablePeerAccess` | 读 |
// | `VMM-read` | `cuMemCreate`@对端 + fd 导入 + `cuMemSetAccess(PROT_READ)` | 读 |
// | `VMM-read-2way` | 双向各一份 VMM 块 | 读 |
// | `VMM-write` | 同上,但导入侧授 **RW**(`VMMPROBE_RW=1` 才跑) | 写 |
//
// ⚠️ **`VMMPROBE_RW=1` 是"push 协议前置"的那一问**:导入侧授 PROT_READWRITE =
// 3080 上 `cuMemSetAccess` 老雷区的邻居(见 `docs/llama.cpp/3080-p2p驱动攻坚-setAccess205...
// 定位实录.md`)。默认**不跑**;要跑先确认没有别的 agent 在占卡(硬重置会殃及)。
//
// ## 编译 / 运行
//
//   nvcc -O3 -arch=sm_86 -o /tmp/vmm-peer-probe tools/vmm-peer-probe.cu -lcuda
//   /tmp/vmm-peer-probe 2 3 [bytes]
//   VMMPROBE_K=200 VMMPROBE_RW=1 /tmp/vmm-peer-probe 2 3 5242880

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
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

#define HDR 64ull  // 与 `crates/backends/cuda/src/comm/ar.rs::HDR` 同值
#define GRID 16    // = owl 生产同档(MAX_THREADS 8192 / BLOCK 512)
#define BLOCK 512
#define VEC 8

template <bool FENCE>
__global__ void k_read_add(uint4* __restrict__ add_to, const uint4* __restrict__ read_from,
                           size_t n16) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  if (FENCE) __threadfence_system();  // owl `ar_reduce.cu` 数据循环**前**那道
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
  if (FENCE) __threadfence_system();  // owl `ar_reduce.cu` 数据循环**后**那道
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

// ── VMM:与 owl `vmm.rs` 同参数(PINNED + DEVICE + POSIX_FD,2MiB 粒度)──────
struct VmmBlk {
  CUdeviceptr va = 0;
  CUmemGenericAllocationHandle h = 0;
  size_t size = 0;
  int fd = -1;
  int ord = -1;
};

static size_t gran_of(int ord) {
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = ord;
  size_t g = 0;
  DK(cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return g;
}

static VmmBlk vmm_create(int ord, size_t bytes) {
  VmmBlk b;
  b.ord = ord;
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = ord;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  const size_t g = gran_of(ord);
  b.size = ((bytes + g - 1) / g) * g;
  DK(cuMemCreate(&b.h, b.size, &prop, 0));
  DK(cuMemAddressReserve(&b.va, b.size, 0, 0, 0));
  DK(cuMemMap(b.va, b.size, 0, b.h, 0));
  CUmemAccessDesc d = {};
  d.location = prop.location;
  d.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  DK(cuMemSetAccess(b.va, b.size, &d, 1));
  DK(cuMemExportToShareableHandle(&b.fd, b.h, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
  return b;
}

/// 导入对端块。`rw=false` = 与 owl `peer_import` 同款(PROT_READ)。
static CUdeviceptr vmm_import(int ord, int fd, size_t size, bool rw) {
  CUmemGenericAllocationHandle h2 = 0;
  DK(cuMemImportFromShareableHandle(&h2, (void*)(intptr_t)fd,
                                    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  CUdeviceptr va = 0;
  DK(cuMemAddressReserve(&va, size, 0, 0, 0));
  DK(cuMemMap(va, size, 0, h2, 0));
  CUmemAccessDesc d = {};
  d.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  d.location.id = ord;
  d.flags = rw ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE : CU_MEM_ACCESS_FLAGS_PROT_READ;
  DK(cuMemSetAccess(va, size, &d, 1));
  return va;
}

static double ms_between(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, a, b));
  return (double)ms;
}

static void report(const char* tag, size_t bytes, int k, double ms) {
  const double us = ms * 1e3 / (double)k;
  printf("  %-16s %9.2f us %8.2f GB/s\n", tag, us, (double)bytes / (us * 1e-6) / 1e9);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "用法: %s <ordA> <ordB> [bytes]\n", argv[0]);
    return 2;
  }
  const int oa = atoi(argv[1]), ob = atoi(argv[2]);
  const size_t bytes = argc > 3 ? (size_t)atoll(argv[3]) : 5242880u;
  const int k = getenv("VMMPROBE_K") ? atoi(getenv("VMMPROBE_K")) : 200;
  const bool do_rw = getenv("VMMPROBE_RW") != nullptr;
  const size_t n16 = bytes / 16;

  DK(cuInit(0));
  printf("[vmm] A=ord%d B=ord%d payload=%zu B K=%d 粒度(A)=%zu\n", oa, ob, bytes, k, gran_of(oa));

  // ── 两块 VMM(两侧各一)──────────────────────────────────────────────
  // ⚠️ **预算纪律(用户 2026-10-18)**:peer 窗口预算 = **BAR1 − 64MB = 192MB**,
  // 超了是**响亮 OOM**(不是硬重置)。⇒ 块按需最小化,`VMMPROBE_BLK_MB` 可覆盖。
  const size_t blk_mb =
      getenv("VMMPROBE_BLK_MB") ? (size_t)atoi(getenv("VMMPROBE_BLK_MB")) : 8;
  // 块必须**装得下 payload**(载荷大于块 ⇒ 越界读对端,必崩):取两者的大者。
  const size_t blk_bytes = ((blk_mb << 20) > bytes ? (blk_mb << 20) : bytes) + HDR;
  CK(cudaSetDevice(ob));
  VmmBlk vb = vmm_create(ob, blk_bytes);
  CK(cudaSetDevice(oa));
  VmmBlk va_own = vmm_create(oa, blk_bytes);
  printf("[vmm] VMM 块(各 %zuMiB): A va=0x%llx | B va=0x%llx(粒度 A=%zu B=%zu)\n",
         blk_bytes >> 20, (unsigned long long)va_own.va, (unsigned long long)vb.va, gran_of(oa),
         gran_of(ob));

  // 对端别名(与 owl `peer_import` 同款:只授 READ)
  const CUdeviceptr alias_b_on_a = vmm_import(oa, vb.fd, vb.size, false);
  const CUdeviceptr alias_a_on_b = vmm_import(ob, va_own.fd, va_own.size, false);
  printf("[vmm] 导入别名: B@A = 0x%llx | A@B = 0x%llx(PROT_READ,同 owl peer_import)\n",
         (unsigned long long)alias_b_on_a, (unsigned long long)alias_a_on_b);

  // ── runtime 对照臂的块(cudaMalloc + peer access)──────────────────────
  void *rt_a = nullptr, *rt_b = nullptr;
  CK(cudaSetDevice(oa)); CK(cudaMalloc(&rt_a, bytes));
  CK(cudaSetDevice(ob)); CK(cudaMalloc(&rt_b, bytes));
  CK(cudaSetDevice(oa)); CK(cudaDeviceEnablePeerAccess(ob, 0));
  CK(cudaSetDevice(ob)); CK(cudaDeviceEnablePeerAccess(oa, 0));

  // 本地 dst(两侧各一块 cudaMalloc;两臂共用 ⇒ 差异只能来自"对端块怎么来的")
  void *dst_a = nullptr, *dst_b = nullptr;
  CK(cudaSetDevice(oa)); CK(cudaMalloc(&dst_a, bytes)); CK(cudaMemset(dst_a, 0, bytes));
  CK(cudaSetDevice(ob)); CK(cudaMalloc(&dst_b, bytes)); CK(cudaMemset(dst_b, 0, bytes));

  // 播种对端块(两侧;保证读到的是真数据)
  CK(cudaSetDevice(ob));
  CK(cudaMemset((void*)(vb.va + HDR), 0x11, bytes));
  CK(cudaMemset(rt_b, 0x22, bytes));
  CK(cudaSetDevice(oa));
  CK(cudaMemset((void*)(va_own.va + HDR), 0x33, bytes));
  CK(cudaMemset(rt_a, 0x44, bytes));

  cudaStream_t sa, sb;
  CK(cudaSetDevice(oa)); CK(cudaStreamCreate(&sa));
  CK(cudaSetDevice(ob)); CK(cudaStreamCreate(&sb));
  cudaEvent_t e0, e1, f0, f1;
  CK(cudaSetDevice(oa)); CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaSetDevice(ob)); CK(cudaEventCreate(&f0)); CK(cudaEventCreate(&f1));

  const uint4* rt_src_on_a = (const uint4*)rt_b;                 // A 读 B(runtime)
  const uint4* vmm_src_on_a = (const uint4*)(alias_b_on_a + HDR); // A 读 B(VMM)

  // ── 臂 1:runtime 对照 ────────────────────────────────────────────────
  CK(cudaSetDevice(oa));
  for (int w = 0; w < 3; ++w) k_read_add<false><<<GRID, BLOCK, 0, sa>>>((uint4*)dst_a, rt_src_on_a, n16);
  CK(cudaStreamSynchronize(sa));
  CK(cudaEventRecord(e0, sa));
  for (int t = 0; t < k; ++t) k_read_add<false><<<GRID, BLOCK, 0, sa>>>((uint4*)dst_a, rt_src_on_a, n16);
  CK(cudaEventRecord(e1, sa)); CK(cudaEventSynchronize(e1));
  report("RT-read-1way", bytes, k, ms_between(e0, e1));

  // ── 臂 2:VMM(与 owl peer_import 同款)──────────────────────────────
  CK(cudaEventRecord(e0, sa));
  for (int t = 0; t < k; ++t)
    k_read_add<false><<<GRID, BLOCK, 0, sa>>>((uint4*)dst_a, vmm_src_on_a, n16);
  CK(cudaEventRecord(e1, sa)); CK(cudaEventSynchronize(e1));
  report("VMM-read-1way", bytes, k, ms_between(e0, e1));

  // ── 臂 2b:同上 + `__threadfence_system()` 前后各一道(owl 核体同款)──
  CK(cudaEventRecord(e0, sa));
  for (int t = 0; t < k; ++t)
    k_read_add<true><<<GRID, BLOCK, 0, sa>>>((uint4*)dst_a, vmm_src_on_a, n16);
  CK(cudaEventRecord(e1, sa)); CK(cudaEventSynchronize(e1));
  report("VMM-read+fence", bytes, k, ms_between(e0, e1));

  // ── 臂 3:VMM 双向(两 rank 同时互读)────────────────────────────────
  {
    const uint4* vmm_src_on_b = (const uint4*)(alias_a_on_b + HDR);
    CK(cudaEventRecord(e0, sa));
    CK(cudaSetDevice(ob)); CK(cudaEventRecord(f0, sb));
    for (int t = 0; t < k; ++t) {
      CK(cudaSetDevice(oa));
      k_read_add<false><<<GRID, BLOCK, 0, sa>>>((uint4*)dst_a, vmm_src_on_a, n16);
      CK(cudaSetDevice(ob));
      k_read_add<false><<<GRID, BLOCK, 0, sb>>>((uint4*)dst_b, vmm_src_on_b, n16);
    }
    CK(cudaSetDevice(oa)); CK(cudaEventRecord(e1, sa));
    CK(cudaSetDevice(ob)); CK(cudaEventRecord(f1, sb));
    CK(cudaEventSynchronize(e1)); CK(cudaEventSynchronize(f1));
    report("VMM-read-2way", bytes, k, ms_between(e0, e1) > ms_between(f0, f1) ? ms_between(e0, e1)
                                                                             : ms_between(f0, f1));
  }

  // ── 臂 3b:runtime 双向(**同程序、同核、同会话**的映射对照)────────────
  {
    CK(cudaSetDevice(oa)); CK(cudaEventRecord(e0, sa));
    CK(cudaSetDevice(ob)); CK(cudaEventRecord(f0, sb));
    for (int t = 0; t < k; ++t) {
      CK(cudaSetDevice(oa));
      k_read_add<false><<<GRID, BLOCK, 0, sa>>>((uint4*)dst_a, (const uint4*)rt_b, n16);
      CK(cudaSetDevice(ob));
      k_read_add<false><<<GRID, BLOCK, 0, sb>>>((uint4*)dst_b, (const uint4*)rt_a, n16);
    }
    CK(cudaSetDevice(oa)); CK(cudaEventRecord(e1, sa));
    CK(cudaSetDevice(ob)); CK(cudaEventRecord(f1, sb));
    CK(cudaEventSynchronize(e1)); CK(cudaEventSynchronize(f1));
    report("RT-read-2way", bytes, k, ms_between(e0, e1) > ms_between(f0, f1) ? ms_between(e0, e1)
                                                                            : ms_between(f0, f1));
  }

  // ── 臂 4:VMM **RW 导入** ⇒ 写对端(`VMMPROBE_RW=1`;push 协议的前置那一问)──
  // 用**独立的小块**(不碰上面读臂的映射),预算可控:payload+HDR 向上对齐。
  if (do_rw) {
    printf("[vmm] ⚠️ RW 导入臂开启(独立小块;预算纪律 192MB)\n");
    CK(cudaSetDevice(ob));
    VmmBlk rw_b = vmm_create(ob, bytes + HDR);
    CK(cudaMemset((void*)(rw_b.va + HDR), 0, bytes));
    CK(cudaSetDevice(oa));
    VmmBlk rw_a = vmm_create(oa, bytes + HDR);
    CK(cudaMemset((void*)(rw_a.va + HDR), 0, bytes));
    // 交叉 RW 导入(A 得写 B 的块,B 得写 A 的块)
    const CUdeviceptr rw_b_on_a = vmm_import(oa, rw_b.fd, rw_b.size, /*rw=*/true);
    CK(cudaSetDevice(ob));
    const CUdeviceptr rw_a_on_b = vmm_import(ob, rw_a.fd, rw_a.size, /*rw=*/true);
    printf("[vmm] RW 导入成功: B@A=0x%llx A@B=0x%llx(size 各 %zu;⇒ 可写对端映射是通的)\n",
           (unsigned long long)rw_b_on_a, (unsigned long long)rw_a_on_b, rw_b.size);

    CK(cudaSetDevice(oa));
    for (int w = 0; w < 3; ++w)
      k_write_only<<<GRID, BLOCK, 0, sa>>>((uint4*)(rw_b_on_a + HDR), n16, 0xA5A5u);
    CK(cudaStreamSynchronize(sa));
    CK(cudaEventRecord(e0, sa));
    for (int t = 0; t < k; ++t)
      k_write_only<<<GRID, BLOCK, 0, sa>>>((uint4*)(rw_b_on_a + HDR), n16, 0xA5A5u);
    CK(cudaEventRecord(e1, sa)); CK(cudaEventSynchronize(e1));
    report("VMM-write-1way", bytes, k, ms_between(e0, e1));

    // ★ 双向(生产形状:两 rank 同时把部分和写进对端)
    CK(cudaEventRecord(e0, sa));
    CK(cudaSetDevice(ob)); CK(cudaEventRecord(f0, sb));
    for (int t = 0; t < k; ++t) {
      CK(cudaSetDevice(oa));
      k_write_only<<<GRID, BLOCK, 0, sa>>>((uint4*)(rw_b_on_a + HDR), n16, 0xA5A5u);
      CK(cudaSetDevice(ob));
      k_write_only<<<GRID, BLOCK, 0, sb>>>((uint4*)(rw_a_on_b + HDR), n16, 0x5A5Au);
    }
    CK(cudaSetDevice(oa)); CK(cudaEventRecord(e1, sa));
    CK(cudaSetDevice(ob)); CK(cudaEventRecord(f1, sb));
    CK(cudaEventSynchronize(e1)); CK(cudaEventSynchronize(f1));
    report("VMM-write-2way", bytes, k,
           ms_between(e0, e1) > ms_between(f0, f1) ? ms_between(e0, e1) : ms_between(f0, f1));

    // 落地校验(posted 写必须证明真到了对端;两侧都查)
    size_t bad = 0;
    for (int side = 0; side < 2; ++side) {
      std::vector<uint4> h(256);
      const int dev = side == 0 ? ob : oa;
      const CUdeviceptr va = side == 0 ? rw_b.va : rw_a.va;
      CK(cudaSetDevice(dev));
      CK(cudaMemcpy(h.data(), (void*)(va + HDR), h.size() * 16, cudaMemcpyDeviceToHost));
      for (auto& v : h) {
        if (v.x != 0xA5A5u && v.x != 0x5A5Au) ++bad;
      }
    }
    printf("  %-16s %s(两侧各抽样 256 个 16B 向量)\n", "落地校验",
           bad == 0 ? "✓ 全等" : "✗ 有差");
    CK(cudaSetDevice(oa));
  } else {
    printf("[vmm] (RW 导入臂未跑;要跑加 VMMPROBE_RW=1)\n");
  }

  CK(cudaSetDevice(oa));
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1)); CK(cudaStreamDestroy(sa));
  CK(cudaSetDevice(ob));
  CK(cudaEventDestroy(f0)); CK(cudaEventDestroy(f1)); CK(cudaStreamDestroy(sb));
  printf("[vmm] done\n");
  return 0;
}
