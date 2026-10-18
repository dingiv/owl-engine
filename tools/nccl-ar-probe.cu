// **独立分母**:同一对卡上 NCCL all-reduce 的实测时间 —— 与 owl 自研 AR 对照。
//
// ## 为什么必须有它(2026-10-17)
//
// owl 的 AR 基准(`crates/backends/cuda/tests/comm_ar_bench.rs`)里有个 `RR` 探针
// = **两 rank 同时互读**,用来当"AR 数据面天花板"。但它**用的是 owl 自己的核**
// (单 block × 256 线程 + 8 路展开)⇒ 它只证明"我们的核能跑到多少",**不能**证明
// "链路能跑到多少"。下"通信已无算法空间"这种结论必须有**外部**分母。
//
// 本程序 = 那个分母:NCCL(2 rank,P2P over PCIe)在**同一对卡、同一载荷、同 dtype**
// 下的 all-reduce 时间。NCCL 是 write-based + 多 block + 全卡 ncclKernel 的业界
// 基线 ⇒ 若 owl AR ≈ NCCL,则数据面已到链路极限(该转结构/算法面);
// 若 owl AR ≫ NCCL,则**还有空间**,且 NCCL 的写法就是答案。
//
// ## 口径
//
// · 载荷 = f16 `T × 5120`(与 27B 的 hidden 同形):10KB(1 行)/ 1.31MB(128 行)/
//   5.24MB(512 行)/ 10.49MB(1024 行);
// · 计时 = `cudaEvent` 包住 K 次 `ncclAllReduce`,再除 K(设备时,不受 host 发射
//   往返污染 —— 与 owl 基准的 `dev` 列同口径);
// · 有效带宽报**每 rank 单向**口径(`bytes/t`);NCCL 惯用 bus bandwidth =
//   2(n-1)/n × bytes/t,n=2 时 = bytes/t ⇒ 两者同值,可直接与 owl 的 `AR MB/s` 比。
//
// ## 编译 / 运行
//
//   nvcc -O3 -arch=sm_86 -o /tmp/nccl-ar-probe tools/nccl-ar-probe.cu \
//        -I/usr/include -L<pkg>/lib -lnccl -lcudart -Wl,-rpath,<pkg>/lib
//   /tmp/nccl-ar-probe <ordA> <ordB> [bytes ...]
//
//   <pkg>/lib 例:`/home/div/Documents/codes/packages/mistral-test/lib`(libnccl.so.2)。
//
// ⚠️ **两个 rank 必须是两个进程**(NCCL 的进程模型);本程序用 `fork` + 管道传
// `ncclUniqueId`,单进程即可跑完,无需 MPI/launcher。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <nccl.h>
#include <sys/wait.h>
#include <unistd.h>

#define CK(call)                                                                        \
  do {                                                                                  \
    cudaError_t e_ = (call);                                                            \
    if (e_ != cudaSuccess) {                                                            \
      fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

#define NK(call)                                                                        \
  do {                                                                                  \
    ncclResult_t r_ = (call);                                                           \
    if (r_ != ncclSuccess) {                                                            \
      fprintf(stderr, "NCCL error %s:%d %s\n", __FILE__, __LINE__, ncclGetErrorString(r_)); \
      exit(1);                                                                          \
    }                                                                                   \
  } while (0)

/// 一个 rank 的全流程:`bytes` 列表逐个测(每档 warmup 后再计时)
static void run_rank(int rank, int ordinal, ncclUniqueId id, const long* sizes, int nsizes,
                     int reps, bool quiet) {
  CK(cudaSetDevice(ordinal));
  ncclComm_t comm;
  NK(ncclCommInitRank(&comm, 2, id, rank));

  cudaStream_t stream;
  CK(cudaStreamCreate(&stream));
  cudaEvent_t ev0, ev1;
  CK(cudaEventCreate(&ev0));
  CK(cudaEventCreate(&ev1));

  if (!quiet) {
    printf("[nccl] rank=%d ordinal=%d reps=%d\n", rank, ordinal, reps);
  }
  for (int s = 0; s < nsizes; ++s) {
    const long bytes = sizes[s];
    void* buf = nullptr;
    CK(cudaMalloc(&buf, bytes));
    CK(cudaMemset(buf, 0x11, bytes));

    // warmup(含建链/first-call 的算法选择)
    for (int i = 0; i < 5; ++i) NK(ncclAllReduce(buf, buf, bytes / 2, ncclHalf, ncclSum, comm, stream));
    CK(cudaStreamSynchronize(stream));

    CK(cudaEventRecord(ev0, stream));
    for (int i = 0; i < reps; ++i) {
      NK(ncclAllReduce(buf, buf, bytes / 2, ncclHalf, ncclSum, comm, stream));
    }
    CK(cudaEventRecord(ev1, stream));
    CK(cudaEventSynchronize(ev1));
    float ms = 0.f;
    CK(cudaEventElapsedTime(&ms, ev0, ev1));
    const double us = (double)ms * 1000.0 / reps;

    if (rank == 0) {
      printf("[nccl] %10ld  F16  %7.2f us   %8.0f MB/s(每 rank 单向)\n", bytes, us,
             (double)bytes / 1024.0 / 1024.0 / (us / 1e6));
    }
    CK(cudaFree(buf));
  }

  CK(cudaEventDestroy(ev0));
  CK(cudaEventDestroy(ev1));
  CK(cudaStreamDestroy(stream));
  ncclCommDestroy(comm);
  cudaDeviceReset();
}

int main(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "用法: %s <ordA> <ordB> [bytes ...]\n", argv[0]);
    return 2;
  }
  const int oa = atoi(argv[1]);
  const int ob = atoi(argv[2]);
  long sizes[16];
  int nsizes = 0;
  for (int i = 3; i < argc && nsizes < 16; ++i) {
    sizes[nsizes++] = atol(argv[i]);
  }
  if (nsizes == 0) {
    const long d[] = {10240, 1310720, 5242880, 10485760};
    for (int i = 0; i < 4; ++i) sizes[nsizes++] = d[i];
  }
  const int reps = 200;

  ncclUniqueId id;
  NK(ncclGetUniqueId(&id));

  int pipefd[2];
  if (pipe(pipefd) != 0) {
    perror("pipe");
    return 1;
  }
  const pid_t pid = fork();
  if (pid < 0) {
    perror("fork");
    return 1;
  }
  if (pid == 0) {
    close(pipefd[1]);
    ncclUniqueId child_id;
    if (read(pipefd[0], &child_id, sizeof(child_id)) != (ssize_t)sizeof(child_id)) {
      fprintf(stderr, "child: 读 uniqueId 失败\n");
      _exit(1);
    }
    close(pipefd[0]);
    run_rank(1, ob, child_id, sizes, nsizes, reps, true);
    _exit(0);
  }
  close(pipefd[0]);
  if (write(pipefd[1], &id, sizeof(id)) != (ssize_t)sizeof(id)) {
    fprintf(stderr, "parent: 写 uniqueId 失败\n");
    return 1;
  }
  close(pipefd[1]);
  run_rank(0, oa, id, sizes, nsizes, reps, false);
  int st = 0;
  waitpid(pid, &st, 0);
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
    fprintf(stderr, "子进程(rank1)异常退出: status=%d\n", st);
    return 1;
  }
  return 0;
}
