// owl 通信算子族 —— allreduce **fused**(v1 两核 → 一核;2026-10-19 立案,路线图 §11.2)
//
// # 为什么有这一族
//
// §59.1 图内实测(10KB 档):全协议 AR **22.13 µs/次**、旁路协议 22.67(−2.4%)
// ⇒ **协议钱 ≈ 0**;**单核 11.13 µs、第二个核的边际成本 11.00 µs**,搬运仅 0.8 µs
// ⇒ 成本**全在"核的落地/排空固定开销"** ⇒ 靶 = **少发一个核**
// (`ar_stage` + `ar_reduce` 两核 ⇒ 本核一核做完同一协议)。
//
// # 语义(与两核**逐字同一协议**,只把两个体接在同一核里)
//
//   ① credit(写前;等 `peer.consumed + n_slots > own.produced`——不覆盖未读走的槽)
//   ② 写自己的槽(`data_w`;v1 = own、push = peer)⇒ per-thread fence ⇒ 最后一块
//      发布 `own.produced = prod0 + 1`
//   ③ **尾部 rendezvous**(等 `peer.produced >= prod0 + 1`——两核版这步在 stage 里
//      已存在,「这个等待原在 reduce 里,只是提前到同一条流上」)
//   ④ **全线程 acquire**(§56.5;与生产者 release 配对)⇒ 读对端槽(`data_r`;
//      v1 = peer、push = own)⇒ `payload[i] += peer[i]`
//   ⑤ 最后一块发布 `own.consumed = cons0 + 1`(= 对端下一轮 stage 的 credit)
//
// # 与两核版的**两处结构差**(都要记着)
//
// 1. **两段 block 间令牌**:stage 的 rendezvous 在"最后到的那一块"上顺带完成
//    (其余块已退出 ⇒ 核完成时刻 = rendezvous 结束);本核结束后还要**全体读**
//    ⇒ 必须放行全体 block ⇒ `wait_done` 走**两段值**:
//        第一段 `2·prod0 + 1`(credit 放行,block 0 写)
//        第二段 `2·prod0 + 2`(rendezvous 放行,最后到的一块写)
//    单调 ✓、相对量(图回放安全 ✓)、两段值域不重叠 ✓。
//    ⚠️ `wait_done` 是**本核私有语义**了 —— 两核版与融合版**不得混用同一套 header**
//    (eval 注入点二选一;换档 = 重新装配,header 由 boot 握手重新清零)。
// 2. **`produced` 每调用 +1**(两核版 stage +1、reduce 再 +1 ⇒ 实为 +2 —— 那是
//    1-stage 时代遗留,靠流序掩盖)。本核用**正确游标**(prod = 调用数)
//    ⇒ credit 判据真正生效(两核版因 prod 虚高而近乎恒真)。
//    ⚠️ 两 rank **同构**(都用本核)⇒ 游标语义一致 ✓
//
// # 与两核的关系(纪律)
//
// - **协议/几何/fence 律逐字沿用**:header 布局(权威注记 = `ar_stage.cu` 头)、
//   不变量 I1–I5、超时(D4:置 err 整轮失败,不静默)、失败路径不提前 return;
// - helper **逐字复制**(`owl_ar_copy_bytes` ← `ar_stage.cu`;`owl_ar_reduce_vec`/
//   `owl_ar_reduce_v16`/dtype 桥 ← `ar_reduce.cu`)—— 遵本仓纪律:**`.cu` 之间不
//   include**(构建期增量指纹只覆盖 `.cu` 自身);**改 helper = 三处同改 + 门重跑**;
// - **本版只覆盖非 fp8 载荷**(f32/f16/bf16 三实例):decode 的 AR(10KB)本就走
//   非量化档(`tp_ar_f8 && bytes >= 256KB` 才开 fp8)⇒ 收益面全覆盖;
//   prefill 的 fp8 腿**仍走两核**(将来要 fp8 融合,照 `ar_*_f8_*` 两族加实例)。
//
// # 前置清单(与两核同律;逐条对着查)
//
// ① 两 rank 同参(`slot_bytes` / `n_slots` / `slab_off`;header 校验同 stage);
// ② header 四格(produced/consumed/wait_done/blocks_done)**必须全 0** 起步
//    (分配侧保证;`wait_done` 非 0 残值 ⇒ 非 0 号 block 跳过等待 = 静默错值);
// ③ `payload` 是**私有块**(读+写同一块:v1 的私有 payload 语义,不是对称槽);
// ④ `data_w`/`data_r` 必须属于同一次 AR 的那一套槽(v1: w=own, r=peer);
// ⑤ 发射几何纯参数(与两核同;单 block 也能跑,多 block 才并行访存)。

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

// ---- header 布局(与 cu/comm/ar_stage.cu **逐字一致**;权威注记在那)----
#define OWL_AR_MAGIC 0x4F574C41u // "OWLA"(与 comm::ar::MAGIC 同值)
#define OWL_AR_HDR_BYTES 64u
#define OWL_AR_OFF_PRODUCED 0u
#define OWL_AR_OFF_CONSUMED 8u
#define OWL_AR_OFF_SLOT_BYTES 16u
#define OWL_AR_OFF_N_SLOTS 20u
#define OWL_AR_OFF_MAGIC 24u
#define OWL_AR_OFF_ERR 28u
#define OWL_AR_OFF_WAIT_DONE 32u
#define OWL_AR_OFF_BLOCKS_DONE 40u
#define OWL_AR_TIMEOUT_NS 5000000000ull
#define OWL_AR_SPIN_TIGHT 4096u

/// 设备全局时钟(%globaltimer,ns 级;sm_30+)
__device__ __forceinline__ unsigned long long owl_ar_timer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// ---- dtype 桥(与 cu/comm/ar_reduce.cu **逐字同式**:float 中间精度)----
template <typename T> __device__ __forceinline__ float owl_ar_to_float(T x);
template <> __device__ __forceinline__ float owl_ar_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float owl_ar_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float owl_ar_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T owl_ar_from_float(float x);
template <> __device__ __forceinline__ float owl_ar_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half owl_ar_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 owl_ar_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// ---- 字节拷贝(← ar_stage.cu **逐字**;16 B 向量 + 8 路展开 = MLP 律 §25)----
__device__ __forceinline__ void owl_ar_copy_bytes(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    const size_t bytes) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const unsigned long long da = reinterpret_cast<unsigned long long>(dst);
    const unsigned long long sa = reinterpret_cast<unsigned long long>(src);
    const bool aligned = ((da | sa) & 15ull) == 0ull;
    if (aligned) {
        const size_t n16 = bytes >> 4;
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        for (size_t i = tid; i < n16; i += 8 * stride) {
            uint4 vs[8];
            bool ok[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const size_t j = i + (size_t)k * stride;
                ok[k] = j < n16;
                if (ok[k]) {
                    vs[k] = s4[j];
                }
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                if (ok[k]) {
                    d4[i + (size_t)k * stride] = vs[k];
                }
            }
        }
        const size_t done = n16 << 4;
        for (size_t j = done + tid; j < bytes; j += stride) {
            dst[j] = src[j];
        }
    } else {
        for (size_t i = tid; i < bytes; i += stride) {
            dst[i] = src[i];
        }
    }
}

template <typename T>
__device__ __forceinline__ void owl_ar_copy_bytes_t(unsigned char* __restrict__ dst,
                                                     const T* __restrict__ src,
                                                     const size_t bytes) {
    owl_ar_copy_bytes(dst, reinterpret_cast<const unsigned char*>(src), bytes);
}

// ---- 归约读加(← ar_reduce.cu **逐字**;向量化 + 8 路展开 = 同一条尺)----
template <typename T, typename V>
__device__ __forceinline__ void owl_ar_reduce_vec(
    T* __restrict__ dst, const T* __restrict__ src, const size_t n) {
    constexpr int PER_VEC = (int)(sizeof(V) / sizeof(T));
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t n_vec = n / (size_t)PER_VEC;
    V* d4 = reinterpret_cast<V*>(dst);
    const V* s4 = reinterpret_cast<const V*>(src);
    for (size_t i = tid; i < n_vec; i += 8 * stride) {
        V vs[8];
        bool ok[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const size_t j = i + (size_t)k * stride;
            ok[k] = j < n_vec;
            if (ok[k]) {
                vs[k] = s4[j]; // 8 个在飞读(MLP 的全部意义)
            }
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (!ok[k]) {
                continue;
            }
            const size_t j = i + (size_t)k * stride;
            V dv = d4[j];
            const T* sp = reinterpret_cast<const T*>(&vs[k]);
            T* dp = reinterpret_cast<T*>(&dv);
#pragma unroll
            for (int m = 0; m < PER_VEC; ++m) {
                dp[m] = owl_ar_from_float<T>(owl_ar_to_float<T>(dp[m]) + owl_ar_to_float<T>(sp[m]));
            }
            d4[j] = dv;
        }
    }
    for (size_t j = n_vec * (size_t)PER_VEC + tid; j < n; j += stride) {
        dst[j] = owl_ar_from_float<T>(owl_ar_to_float<T>(dst[j]) + owl_ar_to_float<T>(src[j]));
    }
}

template <typename T>
__device__ __forceinline__ void owl_ar_reduce_v16(T* __restrict__ dst,
                                                  const T* __restrict__ src,
                                                  const size_t n) {
    owl_ar_reduce_vec<T, uint4>(dst, src, n);
}

// ============================================================================
// 融合核(单核两阶段;§11.2)
// ============================================================================

/// 两段 `wait_done` 令牌(见头注 1):
///   第一段 = `2·prod0 + 1`(credit 放行;block 0 写)
///   第二段 = `2·prod0 + 2`(rendezvous 放行;**最后到的一块**写,失败也写)
#define OWL_AR_WAIT_CREDIT(prod0) (2ull * (prod0) + 1ull)
#define OWL_AR_WAIT_RENDEZ(prod0) (2ull * (prod0) + 2ull)

#define OWL_AR_FUSED_KERNEL(NAME, T, PARAMS)                                              \
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "AR fused: 仅 f32/f16/bf16");         \
    extern "C" __global__ void NAME PARAMS {                                              \
        unsigned char* own_b = reinterpret_cast<unsigned char*>(own);                     \
        const unsigned char* peer_b = reinterpret_cast<const unsigned char*>(peer);       \
        unsigned char* w_b = reinterpret_cast<unsigned char*>(data_w);                    \
        const unsigned char* r_b = reinterpret_cast<const unsigned char*>(data_r);        \
        volatile unsigned long long* own_prod =                                           \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_PRODUCED);   \
        volatile unsigned long long* own_cons =                                           \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_CONSUMED);   \
        volatile unsigned int* own_err =                                                  \
            reinterpret_cast<volatile unsigned int*>(own_b + OWL_AR_OFF_ERR);              \
        const volatile unsigned int* own_magic =                                          \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_MAGIC);      \
        const volatile unsigned int* own_slot_bytes =                                     \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_SLOT_BYTES); \
        const volatile unsigned int* own_nslots =                                         \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_N_SLOTS);    \
        const volatile unsigned long long* peer_prod =                                    \
            reinterpret_cast<const volatile unsigned long long*>(peer_b + OWL_AR_OFF_PRODUCED); \
        const volatile unsigned long long* peer_cons =                                    \
            reinterpret_cast<const volatile unsigned long long*>(peer_b + OWL_AR_OFF_CONSUMED); \
        volatile unsigned long long* own_wait =                                           \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_WAIT_DONE);  \
        unsigned int* own_blocks_done =                                                   \
            reinterpret_cast<unsigned int*>(own_b + OWL_AR_OFF_BLOCKS_DONE);               \
        __shared__ unsigned long long s_prod;                                             \
        __shared__ unsigned long long s_cons;                                             \
        __shared__ int s_fail;                                                            \
        if (threadIdx.x == 0) {                                                           \
            s_prod = 0ull;                                                                \
            s_cons = 0ull;                                                                \
            s_fail = 0;                                                                   \
            if (*own_magic != OWL_AR_MAGIC) {                                             \
                *own_err = 3u; /* 串块/未初始化 */                                        \
                s_fail = 1;                                                               \
            } else if ((size_t)(*own_slot_bytes) != slot_bytes) {                         \
                *own_err = 3u; /* 声明与 header 漂移 */                                   \
                s_fail = 1;                                                               \
            } else if (*own_nslots == 0u) {                                               \
                *own_err = 3u; /* 0 格 = 无格可写(调用方错)⇒ 响亮拒 */                  \
                s_fail = 1;                                                               \
            } else {                                                                      \
                const unsigned long long prod = *own_prod;                                \
                const unsigned long long cons = *own_cons;                                \
                s_prod = prod;                                                            \
                s_cons = cons;                                                            \
                const unsigned long long n_slots = (unsigned long long)(*own_nslots);     \
                /* ① credit(写前;stage 既有律 I1):不覆盖对端未读走的槽。               \
                 * need = prod >= n_slots ? prod - n_slots + 1 : 0 */                     \
                const unsigned long long need =                                           \
                    prod >= n_slots ? (prod - n_slots + 1ull) : 0ull;                     \
                const unsigned long long t0 = owl_ar_timer_ns();                          \
                unsigned int spins = 0u;                                                  \
                if (blockIdx.x == 0) {                                                    \
                    *own_cons = prod; /* 代发 credit(值与 reduce 的发布收敛) */         \
                    while (*peer_cons < need) {                                           \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                                \
                            __nanosleep(200u);                                            \
                        }                                                                 \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {                 \
                            *own_err = 1u;                                                \
                            s_fail = 1;                                                   \
                            break;                                                        \
                        }                                                                 \
                    }                                                                     \
                    if (!s_fail) {                                                        \
                        __threadfence_system();                                           \
                    }                                                                     \
                    /* **无条件**放行第一段(即便失败;防其余 block 死等) */             \
                    *own_wait = OWL_AR_WAIT_CREDIT(prod);                                 \
                } else {                                                                  \
                    while (*own_wait < OWL_AR_WAIT_CREDIT(prod)) {                        \
                        if (*own_err != 0u) { /* 上游失败(设备内存可见)*/              \
                            s_fail = 1;                                                   \
                            break;                                                        \
                        }                                                                 \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                                \
                            __nanosleep(200u);                                            \
                        }                                                                 \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {                 \
                            *own_err = 1u;                                                \
                            s_fail = 1;                                                   \
                            break;                                                        \
                        }                                                                 \
                    }                                                                     \
                    if (!s_fail) {                                                        \
                        __threadfence_system();                                           \
                    }                                                                     \
                }                                                                         \
            }                                                                             \
        }                                                                                 \
        __syncthreads();                                                                  \
        if (s_fail) {                                                                     \
            return; /* 全体一致退出(thread0 不早退 —— 防 __syncthreads 死锁) */    \
        }                                                                                 \
        const unsigned int slot =                                                         \
            (unsigned int)(s_prod % (unsigned long long)(*own_nslots));                    \
        /* ② 写自己的槽(stage 搬运;v1 写 own、push 写 peer) */                      \
        owl_ar_copy_bytes_t<T>(                                                           \
            w_b + OWL_AR_HDR_BYTES + (size_t)slot * slot_bytes,                           \
            reinterpret_cast<const T*>(                                                   \
                reinterpret_cast<const unsigned char*>(payload) + slab_off),              \
            bytes);                                                                       \
        __threadfence_system(); /* 每线程:本线程写 → 对系统可见 */                    \
        __syncthreads();                                                                  \
        /* ③ 最后一块:发布 produced + **尾部 rendezvous** + 放行第二段 */         \
        if (threadIdx.x == 0) {                                                           \
            const unsigned int done = atomicAdd(own_blocks_done, 1u);                      \
            if (done == gridDim.x - 1u) {                                                 \
                *own_blocks_done = 0u; /* 回零:同流保序 ⇒ 下一次调用必见 0 */         \
                if (!s_fail) {                                                            \
                    *own_prod = s_prod + 1ull;                                            \
                    const unsigned long long want = s_prod + 1ull;                        \
                    const unsigned long long tr = owl_ar_timer_ns();                      \
                    unsigned int rs = 0u;                                                 \
                    while (*peer_prod < want) {                                           \
                        if (++rs > OWL_AR_SPIN_TIGHT) {                                   \
                            __nanosleep(200u);                                            \
                        }                                                                 \
                        if (owl_ar_timer_ns() - tr > OWL_AR_TIMEOUT_NS) {                 \
                            *own_err = 1u;                                                \
                            s_fail = 1;                                                   \
                            break;                                                        \
                        }                                                                 \
                    }                                                                     \
                    if (!s_fail) {                                                        \
                        __threadfence_system();                                           \
                    }                                                                     \
                }                                                                         \
                __threadfence_system(); /* err 先于 wait 可见(失败也要放行) */        \
                *own_wait = OWL_AR_WAIT_RENDEZ(s_prod); /* **无条件** */              \
            }                                                                             \
        }                                                                                 \
        /* ③' 全体等第二段(err 非 0 ⇒ 跳过读) */                                  \
        if (threadIdx.x == 0) {                                                           \
            const unsigned long long t2 = owl_ar_timer_ns();                              \
            unsigned int s2 = 0u;                                                         \
            while (*own_wait < OWL_AR_WAIT_RENDEZ(s_prod)) {                              \
                if (*own_err != 0u) {                                                     \
                    s_fail = 1;                                                           \
                    break;                                                                \
                }                                                                         \
                if (++s2 > OWL_AR_SPIN_TIGHT) {                                           \
                    __nanosleep(200u);                                                    \
                }                                                                         \
                if (owl_ar_timer_ns() - t2 > OWL_AR_TIMEOUT_NS) {                         \
                    *own_err = 1u;                                                        \
                    s_fail = 1;                                                           \
                    break;                                                                \
                }                                                                         \
            }                                                                             \
            if (!s_fail) {                                                                \
                __threadfence_system();                                                   \
            }                                                                             \
        }                                                                                 \
        __syncthreads();                                                                  \
        if (s_fail) {                                                                     \
            return;                                                                       \
        }                                                                                 \
        /* ④ 读对端槽 + 加(reduce 搬运;全线程 acquire §56.5 与生产者 release 配对) */ \
        __threadfence_system();                                                           \
        owl_ar_reduce_v16<T>(                                                             \
            reinterpret_cast<T*>(                                                         \
                reinterpret_cast<unsigned char*>(payload) + slab_off),                    \
            reinterpret_cast<const T*>(r_b + OWL_AR_HDR_BYTES +                           \
                                       (size_t)slot * slot_bytes),                        \
            bytes / sizeof(T));                                                           \
        __threadfence_system();                                                           \
        __syncthreads();                                                                  \
        /* ⑤ 最后一块发布 consumed(= 对端下一轮 stage 的 credit) */               \
        if (threadIdx.x == 0) {                                                           \
            const unsigned int done = atomicAdd(own_blocks_done, 1u);                      \
            if (done == gridDim.x - 1u) {                                                 \
                *own_blocks_done = 0u;                                                    \
                *own_cons = s_cons + 1ull;                                                \
            }                                                                             \
        }                                                                                 \
    }

OWL_AR_FUSED_KERNEL(owl_comm_ar_fused_f32, float, (float* __restrict__ payload, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data_w, const unsigned char* __restrict__ data_r, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_FUSED_KERNEL(owl_comm_ar_fused_f16, __half, (__half* __restrict__ payload, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data_w, const unsigned char* __restrict__ data_r, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_FUSED_KERNEL(owl_comm_ar_fused_bf16, __nv_bfloat16, (__nv_bfloat16* __restrict__ payload, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data_w, const unsigned char* __restrict__ data_r, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
