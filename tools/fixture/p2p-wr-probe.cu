// ★ **写方向探针**(2026-10-18;TP2 施工设计 §36 立案)
//
// ## 为什么有它
//
// owl 的 AR 通道审计里,**所有**跨卡探针量的都是**读腿**:
//   · `R` / `RR`  = SM 的 `ld.global` 跨 PCIe 读对端;
//   · `C1` / `C2` = 拷贝引擎(CE)把对端 staging 读进本端;
//   · `AR` / `A3` = 完整协议(仍然全是读)。
// 同会话读数(3080 对 2/3;5.24MB f16;设备时):
//   NCCL 2-rank all-reduce = **463.2µs**,而我们**最快**的双向搬运臂 `C2`(CE 双向读)
//   = **496.1µs**,SM 双向读 `RR` = 566.7µs,**生产 AR = 601.7µs**。
//
// ⇒ **NCCL 比我们量过的任何一条"读腿"都快** —— 它走的大概率不是读腿。
// PCIe 语义上这是必然的:**读是 non-posted(必须等往返),写是 posted(发完就走)**
// ⇒ 写方向的吞吐通常高于读,且不吃往返延迟。NCCL world=2 是 write-based
// (1-stage 原地累加 / ring 回写),吃的是**写腿**。
//
// 本程序就是那条缺失的分母:**写腿**能跑多少,单向/双向各是多少,
// 与读腿**同会话、同载荷、同几何**对照。
//
// ## 为什么走 runtime 路径(`cudaDeviceEnablePeerAccess`)
//
// owl 引擎走 VMM 路径,而 `state.rs::peer_import` 只授 `PROT_READ`
// (`set_access_self(&map, ordinal, false)`)⇒ **引擎侧写对端当场 ILLEGAL_ADDRESS**
// (2026-10-17 的 `OWL_ARB_PUSH` 臂 0.9s 响亮失败就是这个)。
// 本探针**不为落地**,只回答"这对卡、这个驱动,写腿能不能到链路量级":
// runtime 的 peer access 是**读写全权**(没有只读选项)⇒ 是量写腿的最短路径。
// (落地形态另说:VMM + `CU_MEM_ACCESS_FLAGS_PROT_READWRITE`;**落到哪条腿由本读数决定**。)
//
// ## 口径
//
// · 载荷 = f16 `T × 5120` 同形:1.31MB(128 行)/ 5.24MB(chunk 512)/ 10.49MB(1024 行);
// · 几何 = **32 block × 512 线程**(与生产 `MAX_BLOCKS=32` 同档),核内 **8 路 16B 展开**
//   (与 `owl_ar_copy_bytes` 同律:跨 PCIe 的吞吐由在飞访存数决定);
// · 计时 = `cudaEvent` 包住 K 次核(纯设备时,不含 host 发射往返 —— 与
//   `nccl-ar-probe` / `ar_channel_bench` 的 `AR µs` 列同口径);
// · **单向臂只给 A 发**,双向臂两条流并发(A→B ‖ B→A),取两侧墙钟的**较大者**;
// · 有效带宽 = `bytes/t`,**每 rank 单口径**(与 `nccl-ar-probe` 的"每 rank 单向"同口径);
// · 写臂必须**校验落地**(写进对端 → 拷回 host 逐字节比):posted 写"发出去"不等于
//   "落地",不校验就可能量到一个假的漂亮数字。
//
// ## 编译 / 运行
//
//   nvcc -O3 -arch=sm_86 -o /tmp/p2p-wr-probe tools/p2p-wr-probe.cu
//   /tmp/p2p-wr-probe <ordA> <ordB> [bytes ...]
//
// 默认档:`1310720 5242880 10485760`;`P2PWR_K`(缺省 200)控重复次数。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#define CK(x)                                                                        \
  do {                                                                               \
    cudaError_t e_ = (x);                                                            \
    if (e_ != cudaSuccess) {                                                         \
      fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__,                  \
              cudaGetErrorString(e_));                                               \
      exit(1);                                                                       \
    }                                                                                \
  } while (0)

// 默认几何 = **owl 生产同档**:`MAX_THREADS = 32×256 = 8192`、`asset::BLOCK = 512`
// ⇒ grid = 8192/512 = **16 块**。核内 8 路 16B 展开(与 `owl_ar_copy_bytes` 同律)。
// ⚠️ 可被 `P2PWR_GRID` / `P2PWR_BLOCK` 覆盖 —— **线程数是在飞访存数那一维**,
// 扫它是为了分清"SM 读慢"与"在飞线程不够"。
#ifndef GRID
#define GRID 16
#endif
#ifndef BLOCK
#define BLOCK 512
#endif
#define VEC 8

// ── 读腿:读对端 → 加到本地(等价于生产 reduce 核的数据面)──────────────
__global__ void k_read_add(uint4* __restrict__ add_to, const uint4* __restrict__ read_from,
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
}

// ── 写腿:往对端写常量(等价于 push 协议的数据面)────────────────────
// `FENCE=true` 档在核尾加 `__threadfence_system()` —— push 协议发布 flag
// 之前**必须**有它;两档的差 = "posted 能省多少"。
template <bool FENCE>
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
  if (FENCE) __threadfence_system();
}

static double ms_between(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f;
  CK(cudaEventElapsedTime(&ms, a, b));
  return (double)ms;
}

static void report(const char* tag, size_t bytes, int k, double ms, double* out_us) {
  const double us = ms * 1e3 / (double)k;
  const double gbs = (double)bytes / (us * 1e-6) / 1e9;
  if (out_us) *out_us = us;
  printf("  %-14s %9.2f us %8.2f GB/s\n", tag, us, gbs);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "用法: %s <ordA> <ordB> [bytes ...]\n", argv[0]);
    return 2;
  }
  const int oa = atoi(argv[1]);
  const int ob = atoi(argv[2]);
  std::vector<size_t> sizes;
  for (int i = 3; i < argc; ++i) sizes.push_back((size_t)atoll(argv[i]));
  if (sizes.empty()) {
    sizes = {1310720u, 5242880u, 10485760u};
  }
  const int k = getenv("P2PWR_K") ? atoi(getenv("P2PWR_K")) : 200;
  const int grid = getenv("P2PWR_GRID") ? atoi(getenv("P2PWR_GRID")) : GRID;
  const int blk = getenv("P2PWR_BLOCK") ? atoi(getenv("P2PWR_BLOCK")) : BLOCK;

  int can_ab = 0, can_ba = 0;
  CK(cudaDeviceCanAccessPeer(&can_ab, oa, ob));
  CK(cudaDeviceCanAccessPeer(&can_ba, ob, oa));
  printf("[wr] devices A=%d B=%d canPeer(A->B)=%d canPeer(B->A)=%d K=%d\n", oa, ob, can_ab,
         can_ba, k);
  printf("[wr] 几何: grid=%d×block=%d = %d 线程在飞(8 路展开 ⇒ 每轮 %d 个 16B 在飞)\n",
         grid, blk, grid * blk, grid * blk * 8);
  if (!can_ab || !can_ba) {
    fprintf(stderr, "[wr] 这对卡不支持 P2P,退出\n");
    return 3;
  }
  CK(cudaSetDevice(oa));
  CK(cudaDeviceEnablePeerAccess(ob, 0));
  CK(cudaSetDevice(ob));
  CK(cudaDeviceEnablePeerAccess(oa, 0));

  cudaStream_t sa, sb;
  CK(cudaSetDevice(oa));
  CK(cudaStreamCreate(&sa));
  CK(cudaSetDevice(ob));
  CK(cudaStreamCreate(&sb));

  for (size_t bytes : sizes) {
    const size_t n16 = bytes / 16;
    void *buf_a = nullptr, *buf_b = nullptr;
    CK(cudaSetDevice(oa));
    CK(cudaMalloc(&buf_a, bytes));
    CK(cudaMemset(buf_a, 0, bytes));
    CK(cudaSetDevice(ob));
    CK(cudaMalloc(&buf_b, bytes));
    CK(cudaMemset(buf_b, 0, bytes));

    cudaEvent_t e0, e1;  // A 侧
    cudaEvent_t f0, f1;  // B 侧
    CK(cudaSetDevice(oa));
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaSetDevice(ob));
    CK(cudaEventCreate(&f0)); CK(cudaEventCreate(&f1));

    printf("\n[wr] payload=%zu B (%.2f MB)  n16=%zu  grid=%d×%d\n", bytes,
           (double)bytes / 1048576.0, n16, grid, blk);

    // ── 1. 读腿(对照):A 单边读 B ─────────────────────────────────
    CK(cudaSetDevice(oa));
    for (int w = 0; w < 3; ++w)
      k_read_add<<<grid, blk, 0, sa>>>((uint4*)buf_a, (const uint4*)buf_b, n16);
    CK(cudaStreamSynchronize(sa));
    CK(cudaEventRecord(e0, sa));
    for (int t = 0; t < k; ++t)
      k_read_add<<<grid, blk, 0, sa>>>((uint4*)buf_a, (const uint4*)buf_b, n16);
    CK(cudaEventRecord(e1, sa));
    CK(cudaEventSynchronize(e1));
    report("SM-read-1way", bytes, k, ms_between(e0, e1), nullptr);

    // ── 2. 写腿:A 单边写 B ───────────────────────────────────────
    for (int w = 0; w < 3; ++w)
      k_write_only<false><<<grid, blk, 0, sa>>>((uint4*)buf_b, n16, 0x1234u);
    CK(cudaStreamSynchronize(sa));
    CK(cudaEventRecord(e0, sa));
    for (int t = 0; t < k; ++t)
      k_write_only<false><<<grid, blk, 0, sa>>>((uint4*)buf_b, n16, 0x1234u);
    CK(cudaEventRecord(e1, sa));
    CK(cudaEventSynchronize(e1));
    report("SM-write-1way", bytes, k, ms_between(e0, e1), nullptr);

    // ── 3. 写腿 + fence(发布律的代价)────────────────────────────
    CK(cudaEventRecord(e0, sa));
    for (int t = 0; t < k; ++t)
      k_write_only<true><<<grid, blk, 0, sa>>>((uint4*)buf_b, n16, 0x1234u);
    CK(cudaEventRecord(e1, sa));
    CK(cudaEventSynchronize(e1));
    report("SM-write+fence", bytes, k, ms_between(e0, e1), nullptr);

    // ── 4. 写腿落地校验(posted 写必须证明真到了对端)────────────────
    {
      std::vector<uint4> h(n16 > 4096 ? 4096 : n16);
      CK(cudaSetDevice(ob));
      CK(cudaMemcpy(h.data(), buf_b, h.size() * 16, cudaMemcpyDeviceToHost));
      CK(cudaSetDevice(oa));
      size_t bad = 0;
      for (size_t i = 0; i < h.size(); ++i) {
        if (h[i].x != 0x1234u || h[i].w != (0xdeadbeefu ^ 0x1234u)) ++bad;
      }
      printf("  %-14s %s(抽样 %zu 个 16B 向量)\n", "落地校验", bad == 0 ? "✓ 全等" : "✗ 有差",
             h.size());
      if (bad) return 4;
    }

    // ── 5. 读腿双向:A 读 B ‖ B 读 A ───────────────────────────────
    {
      CK(cudaSetDevice(oa));
      k_read_add<<<grid, blk, 0, sa>>>((uint4*)buf_a, (const uint4*)buf_b, n16);
      CK(cudaSetDevice(ob));
      k_read_add<<<grid, blk, 0, sb>>>((uint4*)buf_b, (const uint4*)buf_a, n16);
      CK(cudaStreamSynchronize(sa));
      CK(cudaStreamSynchronize(sb));
      CK(cudaSetDevice(oa)); CK(cudaEventRecord(e0, sa));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(f0, sb));
      for (int t = 0; t < k; ++t) {
        CK(cudaSetDevice(oa));
        k_read_add<<<grid, blk, 0, sa>>>((uint4*)buf_a, (const uint4*)buf_b, n16);
        CK(cudaSetDevice(ob));
        k_read_add<<<grid, blk, 0, sb>>>((uint4*)buf_b, (const uint4*)buf_a, n16);
      }
      CK(cudaSetDevice(oa)); CK(cudaEventRecord(e1, sa));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(f1, sb));
      CK(cudaEventSynchronize(e1)); CK(cudaEventSynchronize(f1));
      const double m = ms_between(e0, e1) > ms_between(f0, f1) ? ms_between(e0, e1)
                                                               : ms_between(f0, f1);
      report("SM-read-2way", bytes, k, m, nullptr);
    }

    // ── 6. 写腿双向:A 写 B ‖ B 写 A ───────────────────────────────
    {
      CK(cudaSetDevice(oa)); CK(cudaEventRecord(e0, sa));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(f0, sb));
      for (int t = 0; t < k; ++t) {
        CK(cudaSetDevice(oa));
        k_write_only<false><<<grid, blk, 0, sa>>>((uint4*)buf_b, n16, 0x5678u);
        CK(cudaSetDevice(ob));
        k_write_only<false><<<grid, blk, 0, sb>>>((uint4*)buf_a, n16, 0x9abcu);
      }
      CK(cudaSetDevice(oa)); CK(cudaEventRecord(e1, sa));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(f1, sb));
      CK(cudaEventSynchronize(e1)); CK(cudaEventSynchronize(f1));
      const double m = ms_between(e0, e1) > ms_between(f0, f1) ? ms_between(e0, e1)
                                                               : ms_between(f0, f1);
      report("SM-write-2way", bytes, k, m, nullptr);
    }

    // ── 7. CE 拉(读):B → A,本端发起 ─────────────────────────────
    CK(cudaSetDevice(oa));
    for (int w = 0; w < 3; ++w)
      CK(cudaMemcpyPeerAsync(buf_a, oa, buf_b, ob, bytes, sa));
    CK(cudaStreamSynchronize(sa));
    CK(cudaEventRecord(e0, sa));
    for (int t = 0; t < k; ++t) CK(cudaMemcpyPeerAsync(buf_a, oa, buf_b, ob, bytes, sa));
    CK(cudaEventRecord(e1, sa));
    CK(cudaEventSynchronize(e1));
    report("CE-pull-1way", bytes, k, ms_between(e0, e1), nullptr);

    // ── 8. CE 推(写):A → B,本端发起 ─────────────────────────────
    CK(cudaSetDevice(oa));
    CK(cudaEventRecord(e0, sa));
    for (int t = 0; t < k; ++t) CK(cudaMemcpyPeerAsync(buf_b, ob, buf_a, oa, bytes, sa));
    CK(cudaEventRecord(e1, sa));
    CK(cudaEventSynchronize(e1));
    report("CE-push-1way", bytes, k, ms_between(e0, e1), nullptr);

    // ── 9. CE 双向:两 rank 各推 ───────────────────────────────────
    {
      CK(cudaSetDevice(oa)); CK(cudaEventRecord(e0, sa));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(f0, sb));
      for (int t = 0; t < k; ++t) {
        CK(cudaSetDevice(oa));
        CK(cudaMemcpyPeerAsync(buf_b, ob, buf_a, oa, bytes, sa));
        CK(cudaSetDevice(ob));
        CK(cudaMemcpyPeerAsync(buf_a, oa, buf_b, ob, bytes, sb));
      }
      CK(cudaSetDevice(oa)); CK(cudaEventRecord(e1, sa));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(f1, sb));
      CK(cudaEventSynchronize(e1)); CK(cudaEventSynchronize(f1));
      const double m = ms_between(e0, e1) > ms_between(f0, f1) ? ms_between(e0, e1)
                                                               : ms_between(f0, f1);
      report("CE-push-2way", bytes, k, m, nullptr);
    }

    // ── 10. CE 双向:两 rank 各拉 ──────────────────────────────────
    {
      CK(cudaSetDevice(oa)); CK(cudaEventRecord(e0, sa));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(f0, sb));
      for (int t = 0; t < k; ++t) {
        CK(cudaSetDevice(oa));
        CK(cudaMemcpyPeerAsync(buf_a, oa, buf_b, ob, bytes, sa));
        CK(cudaSetDevice(ob));
        CK(cudaMemcpyPeerAsync(buf_b, ob, buf_a, oa, bytes, sb));
      }
      CK(cudaSetDevice(oa)); CK(cudaEventRecord(e1, sa));
      CK(cudaSetDevice(ob)); CK(cudaEventRecord(f1, sb));
      CK(cudaEventSynchronize(e1)); CK(cudaEventSynchronize(f1));
      const double m = ms_between(e0, e1) > ms_between(f0, f1) ? ms_between(e0, e1)
                                                               : ms_between(f0, f1);
      report("CE-pull-2way", bytes, k, m, nullptr);
    }

    CK(cudaSetDevice(oa));
    CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
    CK(cudaSetDevice(ob));
    CK(cudaEventDestroy(f0)); CK(cudaEventDestroy(f1));
    CK(cudaFree(buf_a));
    CK(cudaFree(buf_b));
  }

  CK(cudaSetDevice(oa));
  CK(cudaStreamDestroy(sa));
  CK(cudaSetDevice(ob));
  CK(cudaStreamDestroy(sb));
  printf("\n[wr] done\n");
  return 0;
}
