// ============================================================================
// owl TP2 —— AR **小包延迟分相探针**(§69.3 的下一刀:17.9 µs/对 到底是什么)
//
// ## 立案
//
// decode 的 `owl_comm_ar_fused_f16` 实测 **17.9 µs/对 × 128 对 = 2.28 ms/步(11%)**,
// 而 10 KB 载荷的**纯传输只有 ~0.8 µs** ⇒ 94% 是"非传输"。要动它,先把每一相拆开:
//
//   ① credit 等(本端环深;稳态应 ≈0)
//   ② 写本端 slot(10 KB 本地拷)**← 本地带宽**
//   ③ `__threadfence_system()`(system 域 fence)**← 疑似大头**
//   ④ 发布 `produced`(一条 volatile store + 对端可见性)
//   ⑤ **尾部 rendezvous:等对端 `produced`** ← **skew/等待项**
//   ⑥ 读对端 slot + 本地累加(PCIe 读 10 KB + 本地 RMW)
//   ⑦ 收尾 fence + 发布 `consumed`
//
// ## 判据 / 臂
//
// | 臂 | 做法 | 量的是 |
// |---|---|---|
// | `MODE=0` | 生产同形(单 block × 256 线程;10 KB ⇒ 一个 pass) | 全相 |
// | `MODE=1` | ③ 改 `__threadfence()`(device 域;**数值不安全**,只测成本)| fence 的 system 溢价 |
// | `MODE=2` | ⑤ 跳过 rendezvous(读可能在写之前)| **等对端**的净成本 |
// | `MODE=3` | ⑥ 只读不加(去掉 RMW 的写回)| 本地 RMW 的成本 |
// | `MODE=4` | ⑥ 完全不读对端 | **PCIe 读**的净成本 |
//
// 每相用 `%globaltimer`(ns)打点(block 0 的 thread 0),K 次取 p50/p90。
// **两 rank 同构**(与生产同律:同参 ⇒ 同序)。
//
// ## 编译 / 运行
//
//   nvcc -O3 -arch=sm_86 -o /tmp/ar-lat tools/ar-latency-probe.cu -lpthread
//   /tmp/ar-lat <ordA> <ordB> [payload_bytes=10240] [K=200]
//   MODE=0..4 /tmp/ar-lat 2 3
//
// ⚠️ 卡序 = **CUDA ordinal**(本机 3080 对 = 2/3)。跑前确认目标卡空闲。
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <algorithm>
#include <cuda_fp16.h>

#define CK(x)                                                                                  \
    do {                                                                                       \
        cudaError_t e_ = (x);                                                                  \
        if (e_ != cudaSuccess) {                                                               \
            fprintf(stderr, "CUDA %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));     \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

#define HDR_BYTES 64u
#define OFF_PRODUCED 0u
#define OFF_CONSUMED 8u
#define OFF_SLOT 16u
#define OFF_NSLOTS 20u
#define OFF_MAGIC 24u
#define OFF_ERR 28u
#define MAGIC 0x4F574C41u

/// ⚠️ `%globaltimer` 在 sm_86 上**量化 ~1 µs**(实测每格 1024 ns)⇒ 1 µs 级的相量不准;
/// 改 `clock64()`(SM 周期,单 block ⇒ 同一 SM,**跨相可比**),再按 SM 时钟折 µs。
static __device__ __forceinline__ unsigned long long ns_now() { return clock64(); }

// 相:0 entry→credit / 1 credit→写完slot / 2 →fence / 3 →发布 / 4 →rendezvous
//     5 →读加完 / 6 →收尾发布(6 = 名称占位;总时长另算)
__device__ __forceinline__ void stamp(unsigned long long* out, int k, int seg,
                                      unsigned long long t) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && out) out[k * 8 + seg] = t;
}

/// 生产同式的 16 B 向量拷贝(8 路展开;10 KB 一个 pass)
__device__ __forceinline__ void copy_bytes(unsigned char* __restrict__ dst,
                                           const unsigned char* __restrict__ src, size_t bytes) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t n16 = bytes >> 4;
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    for (size_t i = tid; i < n16; i += 8 * stride) {
        uint4 vs[8];
        bool ok[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const size_t idx = i + (size_t)j * stride;
            ok[j] = idx < n16;
            if (ok[j]) vs[j] = s4[idx];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j)
            if (ok[j]) d4[i + (size_t)j * stride] = vs[j];
    }
}

__device__ __forceinline__ void reduce_v16(__half* __restrict__ dst, const __half* __restrict__ src,
                                           size_t n, bool write_back) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t n_vec = n / 8;  // uint4 = 8 half
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    const uint4* s4 = reinterpret_cast<const uint4*>(src);
    for (size_t i = tid; i < n_vec; i += 8 * stride) {
        uint4 vs[8];
        bool ok[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const size_t idx = i + (size_t)j * stride;
            ok[j] = idx < n_vec;
            if (ok[j]) vs[j] = s4[idx];
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (!ok[j]) continue;
            const size_t idx = i + (size_t)j * stride;
            uint4 dv = d4[idx];
            const __half* sp = reinterpret_cast<const __half*>(&vs[j]);
            __half* dp = reinterpret_cast<__half*>(&dv);
#pragma unroll
            for (int m = 0; m < 8; ++m)
                dp[m] = __float2half(__half2float(dp[m]) + __half2float(sp[m]));
            if (write_back) d4[idx] = dv;
            else if (dv.x == 0xdeadbeefu) d4[idx] = dv;  // 防死代码消除
        }
    }
}

__global__ void gemm_like(const uint4* __restrict__ w, float* __restrict__ sink, size_t n16) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
        const uint4 v = w[i];
        acc.x += (float)(v.x & 0xffffu);
        acc.y += (float)(v.y & 0xffffu);
    }
    const float s = acc.x + acc.y + acc.z + acc.w;
    if (s == -1.0f) sink[0] = s;
}

extern "C" __global__ void ar_phase_probe(__half* __restrict__ payload,
                                          unsigned char* __restrict__ own_hdr,
                                          const unsigned char* __restrict__ peer_hdr,
                                          __half* __restrict__ own_slot,
                                          const __half* __restrict__ peer_slot, size_t n,
                                          unsigned long long* __restrict__ stamps, int mode) {
    volatile unsigned long long* own_prod =
        reinterpret_cast<volatile unsigned long long*>(own_hdr + OFF_PRODUCED);
    volatile unsigned long long* own_cons =
        reinterpret_cast<volatile unsigned long long*>(own_hdr + OFF_CONSUMED);
    const volatile unsigned long long* peer_prod =
        reinterpret_cast<const volatile unsigned long long*>(peer_hdr + OFF_PRODUCED);
    const volatile unsigned long long* peer_cons =
        reinterpret_cast<const volatile unsigned long long*>(peer_hdr + OFF_CONSUMED);
    volatile unsigned int* own_err = reinterpret_cast<volatile unsigned int*>(own_hdr + OFF_ERR);

    const unsigned long long prod0 = *own_prod;
    const unsigned long long cons0 = *own_cons;
    const int k = (int)prod0;
    const bool ctl = (blockIdx.x == 0 && threadIdx.x == 0);

    unsigned long long t = ns_now();
    stamp(stamps, k, 0, t);  // entry

    // ① credit:不覆盖对端未读走的槽(单槽 ⇒ 等价于等对端消费)
    if (ctl && mode != 5) {
        unsigned long long spin = 0;
        while (*peer_cons + 1ull <= prod0) {
            if (++spin > 20000000ull) { *own_err = 1u; break; }
        }
    }
    __syncthreads();
    t = ns_now();
    stamp(stamps, k, 1, t);  // after credit

    // ② 写本端 slot
    copy_bytes(reinterpret_cast<unsigned char*>(own_slot), reinterpret_cast<const unsigned char*>(payload),
               n * 2);
    t = ns_now();
    stamp(stamps, k, 2, t);  // after slot write

    // ③ fence(system 域 vs device 域)
    if (mode == 1) __threadfence();
    else __threadfence_system();
    t = ns_now();
    stamp(stamps, k, 3, t);  // after fence

    // ④ 发布 produced
    __syncthreads();
    if (ctl) *own_prod = prod0 + 1ull;
    t = ns_now();
    stamp(stamps, k, 4, t);  // after publish

    // ⑤ 尾部 rendezvous(等对端发布)
    if (mode != 2 && mode != 5) {
        if (ctl) {
            unsigned long long spin = 0;
            while (*peer_prod < prod0 + 1ull) {
                if (++spin > 20000000ull) { *own_err = 2u; break; }
            }
        }
        __syncthreads();
    }
    t = ns_now();
    stamp(stamps, k, 5, t);  // after rendezvous

    // ⑥ 读对端 + 累加
    if (mode != 4 && mode != 5) reduce_v16(payload, peer_slot, n, mode != 3);
    t = ns_now();
    stamp(stamps, k, 6, t);  // after reduce

    // ⑦ 收尾:再 fence + 发布 consumed(= 对端下一轮 credit)
    __syncthreads();
    __threadfence_system();
    if (ctl) *own_cons = cons0 + 1ull;
    t = ns_now();
    stamp(stamps, k, 7, t);  // after consumed
}

struct Rank {
    int dev = 0;
    int peer = 0;
    unsigned char* hdr = nullptr;
    unsigned char* peer_hdr = nullptr;
    __half* slot = nullptr;
    __half* peer_slot = nullptr;
    __half* payload = nullptr;
    unsigned long long* stamps = nullptr;
    size_t bytes = 0;
    int k = 0;
};

static Rank g_r[2];
static std::mutex g_mu;
static std::condition_variable g_cv;
static int g_ready = 0;
static int g_go = 0;

static void run_rank(int me, int mode) {
    Rank& r = g_r[me];
    CK(cudaSetDevice(r.dev));
    // 自报家门:header 清零 + 魔数/槽宽
    std::vector<unsigned char> h(HDR_BYTES, 0);
    const unsigned int slot = (unsigned int)r.bytes;
    memcpy(&h[OFF_SLOT], &slot, 4);
    const unsigned int nsl = 1u;
    memcpy(&h[OFF_NSLOTS], &nsl, 4);
    const unsigned int mg = MAGIC;
    memcpy(&h[OFF_MAGIC], &mg, 4);
    CK(cudaMemcpy(r.hdr, h.data(), HDR_BYTES, cudaMemcpyHostToDevice));
    CK(cudaMemset(r.slot, 0, r.bytes));
    std::vector<__half> p(r.bytes / 2);
    for (size_t i = 0; i < p.size(); ++i) p[i] = __float2half((float)((i * 37) % 101) * 0.001f);
    CK(cudaMemcpy(r.payload, p.data(), r.bytes, cudaMemcpyHostToDevice));
    CK(cudaMemset(r.stamps, 0, sizeof(unsigned long long) * 8 * (size_t)r.k));
    CK(cudaDeviceSynchronize());

    {
        std::unique_lock<std::mutex> lk(g_mu);
        ++g_ready;
        g_cv.notify_all();
        g_cv.wait(lk, [] { return g_go == 1; });
    }
    // 预热(不改 header 游标 ⇒ 游标从 0 开始计数)
    for (int i = 0; i < 20; ++i) {
        ar_phase_probe<<<1, 256>>>(r.payload, r.hdr, r.peer_hdr, r.slot, r.peer_slot,
                                   r.bytes / 2, nullptr, mode);
    }
    CK(cudaStreamSynchronize(0));
    // 复位游标(预热已把它们推过)
    std::vector<unsigned char> h0(HDR_BYTES, 0);
    memcpy(&h0[OFF_SLOT], &slot, 4);
    memcpy(&h0[OFF_NSLOTS], &nsl, 4);
    memcpy(&h0[OFF_MAGIC], &mg, 4);
    CK(cudaMemcpy(r.hdr, h0.data(), HDR_BYTES, cudaMemcpyHostToDevice));
    CK(cudaDeviceSynchronize());
    {
        std::unique_lock<std::mutex> lk(g_mu);
        ++g_ready;
        g_cv.notify_all();
        g_cv.wait(lk, [] { return g_go == 2; });
    }
    const bool graph = getenv("GRAPH") != nullptr;
    if (mode == 5 && graph) {
        // 单边图回放臂(mode 5 无协议,不需要对端)⇒ 量"节点税"
        cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
        cudaStream_t cap = nullptr;
        CK(cudaStreamCreateWithFlags(&cap, cudaStreamNonBlocking));
        CK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < r.k; ++i)
            ar_phase_probe<<<1, 256, 0, cap>>>(r.payload, r.hdr, r.peer_hdr, r.slot, r.peer_slot,
                                               r.bytes / 2, nullptr, mode);
        CK(cudaStreamEndCapture(cap, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        size_t nn = 0; CK(cudaGraphGetNodes(g, nullptr, &nn));
        printf("[ar-lat] 节点税臂:节点数 %zu(期望 %d)\n", nn, r.k);
        for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, cap));
        CK(cudaStreamSynchronize(cap));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, cap));
        for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ge, cap));
        CK(cudaEventRecord(e1, cap)); CK(cudaStreamSynchronize(cap));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("  图回放 %.3f ms/20 次 ⇒ **%.2f µs/节点**\n", ms, ms * 1e3 / (double)(20 * r.k));
    } else if (graph) {
        // ★ 双边图回放:两 rank 各捕获 K 节点(MODE=0 全协议),每节点烘自己的 stamps 槽
        cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
        cudaStream_t cap = nullptr;
        CK(cudaStreamCreateWithFlags(&cap, cudaStreamNonBlocking));
        CK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal));
        for (int i = 0; i < r.k; ++i)
            ar_phase_probe<<<1, 256, 0, cap>>>(r.payload, r.hdr, r.peer_hdr, r.slot, r.peer_slot,
                                               r.bytes / 2, r.stamps + (size_t)i * 8, mode);
        CK(cudaStreamEndCapture(cap, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        // 与对端对齐后再回放(否则第一个节点会等很久,污染首节点)
        {
            std::unique_lock<std::mutex> lk(g_mu);
            ++g_ready;
            g_cv.notify_all();
            g_cv.wait(lk, [] { return g_go == 3; });
        }
        CK(cudaGraphLaunch(ge, cap));
        CK(cudaStreamSynchronize(cap));
        cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
        CK(cudaEventRecord(e0, cap));
        for (int i = 0; i < 20; ++i) CK(cudaGraphLaunch(ge, cap));
        CK(cudaEventRecord(e1, cap)); CK(cudaStreamSynchronize(cap));
        float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
        printf("[ar-lat] 双边图回放(全协议,MODE=%d):%.3f ms/20 次 ⇒ **%.2f µs/节点**\n", mode,
               ms, ms * 1e3 / (double)(20 * r.k));
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    } else if (mode == 6) {
        // ★ 生产位形:AR **紧跟一个满占用核**(68 block 流式读 8 MB)⇒ 复现"GEMM 之后"的位形
        const size_t wb = 8u << 20;
        void* w = nullptr; float* sink = nullptr;
        CK(cudaMalloc(&w, wb)); CK(cudaMemset(w, 0x11, wb)); CK(cudaMalloc(&sink, 4));
        for (int i = 0; i < 5; ++i) gemm_like<<<68, 256>>>(reinterpret_cast<const uint4*>(w), sink, wb / 16);
        CK(cudaStreamSynchronize(0));
        for (int i = 0; i < r.k; ++i) {
            gemm_like<<<68, 256>>>(reinterpret_cast<const uint4*>(w), sink, wb / 16);
            ar_phase_probe<<<1, 256>>>(r.payload, r.hdr, r.peer_hdr, r.slot, r.peer_slot,
                                       r.bytes / 2, r.stamps, 0);   // ← 全协议
        }
        CK(cudaStreamSynchronize(0));
    } else {
        for (int i = 0; i < r.k; ++i) {
            ar_phase_probe<<<1, 256>>>(r.payload, r.hdr, r.peer_hdr, r.slot, r.peer_slot,
                                       r.bytes / 2, r.stamps, mode);
        }
        CK(cudaStreamSynchronize(0));
    }
    {
        // ⚠️ 收尾闸门**只在图档存在**(非图档主线程不再发 g_go=3);否则死等
        const bool need = getenv("GRAPH") != nullptr;
        std::unique_lock<std::mutex> lk(g_mu);
        ++g_ready;
        g_cv.notify_all();
        if (need) g_cv.wait(lk, [] { return g_go == 3; });
    }
}

/// 图内回放臂:`GRAPH=1` 时把 K 个 AR 节点捕获成一张图,再回放 R 次
static void graph_arm(int a, int b, size_t bytes, int K, int mode, double clk_ghz) {
    Rank& ra = g_r[0];
    CK(cudaSetDevice(a));
    cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
    cudaStream_t cap = nullptr;   // ⚠️ legacy 默认流不能捕获 ⇒ 专用非阻塞流
    CK(cudaStreamCreateWithFlags(&cap, cudaStreamNonBlocking));
    auto emit = [&](int i, cudaStream_t st) {
        ar_phase_probe<<<1, 256, 0, st>>>(ra.payload, ra.hdr, ra.peer_hdr, ra.slot, ra.peer_slot,
                                          ra.bytes / 2, (i == 1 ? ra.stamps : nullptr), mode);
    };
    CK(cudaStreamBeginCapture(cap, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < K; ++i) emit(0, cap);
    CK(cudaStreamEndCapture(cap, &g));
    size_t nn = 0; CK(cudaGraphGetNodes(g, nullptr, &nn));
    CK(cudaGraphInstantiate(&ge, g, 0));
    printf("[ar-lat] GRAPH 臂:节点数 %zu(期望 %d)⇒ %s\n", nn, K,
           nn >= (size_t)K ? "节点确在图内" : "⚠️ 捕获不全,读数作废");
    for (int w = 0; w < 3; ++w) CK(cudaGraphLaunch(ge, cap));
    CK(cudaStreamSynchronize(cap));
    const int R = 20;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    CK(cudaEventRecord(e0, cap));
    for (int i = 0; i < R; ++i) CK(cudaGraphLaunch(ge, cap));
    CK(cudaEventRecord(e1, cap)); CK(cudaStreamSynchronize(cap));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("  图回放:%.3f ms / %d 次回放 ⇒ **%.2f µs/节点**(含节点串联依赖)\n", ms, R,
           ms * 1e3 / (double)(R * K));
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
    (void)b; (void)clk_ghz;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "用法: %s <ordA> <ordB> [bytes=10240] [K=200]\n", argv[0]);
        return 2;
    }
    const int a = atoi(argv[1]), b = atoi(argv[2]);
    const size_t bytes = argc > 3 ? (size_t)atoll(argv[3]) : 10240;
    const int K = argc > 4 ? atoi(argv[4]) : 200;
    int mode = 0;
    if (const char* m = getenv("MODE")) mode = atoi(m);

    int can = 0;
    CK(cudaDeviceCanAccessPeer(&can, a, b));
    if (!can) { fprintf(stderr, "[ar-lat] 无 P2P(%d→%d)\n", a, b); return 3; }
    CK(cudaSetDevice(a)); CK(cudaDeviceEnablePeerAccess(b, 0));
    CK(cudaSetDevice(b)); CK(cudaDeviceEnablePeerAccess(a, 0));

    struct Alloc { unsigned char* hdr; __half* slot; __half* payload; unsigned long long* stamps; };
    Alloc al[2];
    for (int i = 0; i < 2; ++i) {
        const int d = i == 0 ? a : b;
        CK(cudaSetDevice(d));
        CK(cudaMalloc(&al[i].hdr, HDR_BYTES));
        CK(cudaMalloc(&al[i].slot, bytes));
        CK(cudaMalloc(&al[i].payload, bytes));
        CK(cudaMalloc(&al[i].stamps, sizeof(unsigned long long) * 8 * (size_t)K));
    }
    for (int i = 0; i < 2; ++i) {
        g_r[i].dev = i == 0 ? a : b;
        g_r[i].peer = i == 0 ? b : a;
        g_r[i].hdr = al[i].hdr;
        g_r[i].peer_hdr = al[1 - i].hdr;
        g_r[i].slot = al[i].slot;
        g_r[i].peer_slot = al[1 - i].slot;
        g_r[i].payload = al[i].payload;
        g_r[i].stamps = al[i].stamps;
        g_r[i].bytes = bytes;
        g_r[i].k = K;
    }

    std::thread t0(run_rank, 0, mode), t1(run_rank, 1, mode);
    {
        std::unique_lock<std::mutex> lk(g_mu);
        g_cv.wait(lk, [] { return g_ready == 2; });
        g_go = 1; g_cv.notify_all();
        g_cv.wait(lk, [] { return g_ready == 4; });
        g_go = 2; g_cv.notify_all();
        g_cv.wait(lk, [] { return g_ready == 6; });
        if (getenv("GRAPH")) {
            g_go = 3; g_cv.notify_all();
            g_cv.wait(lk, [] { return g_ready == 8; });
        }
    }
    t0.join(); t1.join();
    double clk0 = 1.9;
    if (const char* c = getenv("OWL_CLK_GHZ")) clk0 = atof(c);
    (void)graph_arm; (void)clk0;

    // ── 汇总(rank 0 的相时刻;两 rank 同构 ⇒ 取主卡)──
    std::vector<unsigned long long> st(8 * (size_t)K);
    CK(cudaSetDevice(a));
    CK(cudaMemcpy(st.data(), g_r[0].stamps, st.size() * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost));
    const char* names[8] = {"entry", "①credit", "②写slot", "③fence", "④发布",
                            "⑤等对端", "⑥读加", "⑦收尾"};
    double clk_ghz = 1.9;
    if (const char* c = getenv("OWL_CLK_GHZ")) clk_ghz = atof(c);
    std::vector<std::vector<double>> seg(7);
    double total = 0;
    int used = 0;
    for (int i = 0; i < K; ++i) {
        const unsigned long long* s = &st[(size_t)i * 8];
        if (s[0] == 0 || s[7] <= s[0]) continue;
        for (int j = 0; j < 7; ++j) seg[j].push_back((double)(s[j + 1] - s[j]) / clk_ghz / 1e3);
        total += (double)(s[7] - s[0]) / clk_ghz / 1e3;
        ++used;
    }
    auto pct = [](std::vector<double>& v, double q) {
        std::sort(v.begin(), v.end());
        return v[(size_t)((v.size() - 1) * q)];
    };
    printf("\n[ar-lat] ord %d/%d · payload %.2f KB · K=%d · MODE=%d · 有效样本 %d\n", a, b,
           bytes / 1024.0, K, mode, used);
    printf("  相        p50 µs   p90 µs   占比\n");
    double sum50 = 0;
    for (int j = 0; j < 7; ++j) sum50 += pct(seg[j], 0.5);
    for (int j = 0; j < 7; ++j) {
        printf("  %-10s %7.2f  %7.2f  %5.1f%%\n", names[j + 1] == nullptr ? "?" : names[j],
               pct(seg[j], 0.5), pct(seg[j], 0.9), 100.0 * pct(seg[j], 0.5) / sum50);
    }
    printf("  %-10s %7.2f  %7.2f  %5.1f%%  (相和 = 核内 p50)\n", "合计", sum50,
           pct(seg[6], 0.9) * 0, 100.0);
    printf("  核内均值 %.2f µs(entry→⑦;兼容多 block 的尾部)\n", total / used);
    return 0;
}
