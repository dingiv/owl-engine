// owl 通信算子族 —— allreduce **stage**(TP2 S2;协议不变量 I1/I3/I4/I5)。
//
// 出处:`docs/arch/TP2-施工设计.md` §3(布局 / 两核 / 不变量)。
// 与 Rust 侧 `owl_comm::ar` 的常量**逐字节一致**(改 = 双侧同改 + 门重跑):
//
//   header(64 B,两条 cache line;slot 数据从偏移 64 开始)
//     +0   u64 produced    本端已写入调用数(stage 核递增;**本核是唯一写者**)
//     +8   u64 consumed    本端已消费对端调用数(reduce 核递增;对端 stage 当 credit 读)
//     +16  u32 slot_bytes  slot 字节数(自描述)
//     +20  u32 n_slots     slot 数(v1 = 2)
//     +24  u32 magic       0x4F574C41 "OWLA"(串块判据)
//     +28  u32 err         错误码:0 正常 / 1 stage 超时 / 2 reduce 超时 / 3 header 校验失败
//     +32..64             保留
//
// 语义:把本端 payload(`local`)第 `slab_off` 起 `bytes` 字节拷进本端 staging 的
// slot[produced % n_slots],系统 fence 后发布 produced++。
//
// 纪律(为什么这么写;每条都有设计出处):
// 1. **单 block**(grid = 1):produced 是**单写者游标** —— 多 block 要原子协调 +
//    "最后一块发布"协议;v1 单 block 循环拷贝(正确优先;S5 再谈流水)。
// 2. **每线程 __threadfence_system()**:发布律 = "数据系统可见 → 再写 flag"。
//    fence 只覆盖**本线程**的写 ⇒ 全线程各自 fence + `__syncthreads()` 之后,才由
//    thread0 写 produced(与 CUDA 编程指南 threadFenceReduction 同构;PCIe 语义上
//    `membar.sys` 是必要条件 —— I4)。
// 3. **credit 自旋(I1)**:`peer.consumed + n_slots > own.produced` 才可覆盖 slot
//    —— 否则会盖掉对端还没读走的数据;超时(5 s)置 err 并**放弃**(D4:宁可整轮
//    失败,不可静默出结果)。
// 4. **相对量(I3)**:两侧只做相对比较 + 设备内自增游标,不烘绝对值 ⇒ 图回放安全。
// 5. **失败路径不提前 return**(thread0 早退 = 其他线程卡死 `__syncthreads`):错误
//    写 shared 标志,全体一致退出。

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define OWL_AR_MAGIC 0x4F574C41u // "OWLA"(与 comm::ar::MAGIC 同值)
#define OWL_AR_HDR_BYTES 64u
#define OWL_AR_OFF_PRODUCED 0u
#define OWL_AR_OFF_CONSUMED 8u
#define OWL_AR_OFF_SLOT_BYTES 16u
#define OWL_AR_OFF_N_SLOTS 20u
#define OWL_AR_OFF_MAGIC 24u
#define OWL_AR_OFF_ERR 28u

/// 自旋超时(纳秒;设备全局时钟)。5 s —— 比任何正常等待都长,比"挂死"短得多。
#define OWL_AR_TIMEOUT_NS 5000000000ull

/// 紧密自旋次数上限(之后 `__nanosleep` 退避 —— 降 PCIe 读压力)
#define OWL_AR_SPIN_TIGHT 4096u

/// 设备全局时钟(%globaltimer,ns 级;sm_30+)
__device__ __forceinline__ unsigned long long owl_ar_timer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

/// 字节块拷贝(16 B 对齐走 uint4;否则退化为逐字节)
__device__ __forceinline__ void owl_ar_copy_bytes(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    const size_t bytes) {
    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;
    const unsigned long long da = reinterpret_cast<unsigned long long>(dst);
    const unsigned long long sa = reinterpret_cast<unsigned long long>(src);
    const bool aligned = ((da | sa) & 15ull) == 0ull;
    if (aligned) {
        const size_t n16 = bytes >> 4;
        uint4* d4 = reinterpret_cast<uint4*>(dst);
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        for (size_t i = tid; i < n16; i += stride) {
            d4[i] = s4[i];
        }
        const size_t done = n16 << 4;
        for (size_t i = done + tid; i < bytes; i += stride) {
            dst[i] = src[i];
        }
    } else {
        for (size_t i = tid; i < bytes; i += stride) {
            dst[i] = src[i];
        }
    }
}

// ---- stage 宏实例:纯字节搬运(dtype 只决定"拷多少字节",不改核体;
//      保留类型化指针让"表 ⇔ .cu 签名"机器互证可查)。----
#define OWL_AR_STAGE_KERNEL(NAME, T, PARAMS)                                          \
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "AR stage: 仅 f32/f16/bf16");      \
    extern "C" __global__ void NAME PARAMS {                                          \
        unsigned char* own_b = reinterpret_cast<unsigned char*>(own);                      \
        const unsigned char* peer_b = reinterpret_cast<const unsigned char*>(peer);   \
        volatile unsigned long long* own_prod =                                       \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_PRODUCED); \
        volatile unsigned int* own_err =                                              \
            reinterpret_cast<volatile unsigned int*>(own_b + OWL_AR_OFF_ERR);         \
        const volatile unsigned int* own_magic =                                      \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_MAGIC); \
        const volatile unsigned int* own_slot_bytes =                                 \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_SLOT_BYTES); \
        const volatile unsigned int* own_nslots =                                     \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_N_SLOTS); \
        const volatile unsigned long long* peer_cons =                                \
            reinterpret_cast<const volatile unsigned long long*>(peer_b + OWL_AR_OFF_CONSUMED); \
        __shared__ unsigned long long s_prod;                                         \
        __shared__ int s_fail;                                                        \
        if (threadIdx.x == 0) {                                                       \
            s_prod = 0ull;                                                            \
            s_fail = 0;                                                               \
            if (*own_magic != OWL_AR_MAGIC) {                                         \
                *own_err = 3u; /* 串块/未初始化 */                                    \
                s_fail = 1;                                                           \
            } else if ((size_t)(*own_slot_bytes) != slot_bytes) {                     \
                *own_err = 3u; /* 声明与 header 漂移(两处不同改的哨兵) */            \
                s_fail = 1;                                                           \
            } else {                                                                  \
                const unsigned long long prod = *own_prod;                            \
                const unsigned long long n_slots = (unsigned long long)(*own_nslots); \
                /* I1:不覆盖未读 —— peer.consumed + n_slots > own.produced */        \
                if (prod >= n_slots) {                                                \
                    const unsigned long long need = prod - n_slots + 1ull;            \
                    const unsigned long long t0 = owl_ar_timer_ns();                  \
                    unsigned int spins = 0u;                                          \
                    while (*peer_cons < need) {                                       \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                            \
                            __nanosleep(200u);                                        \
                        }                                                             \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {             \
                            *own_err = 1u;                                            \
                            s_fail = 1;                                               \
                            break;                                                    \
                        }                                                             \
                    }                                                                 \
                }                                                                     \
                s_prod = prod;                                                        \
            }                                                                         \
        }                                                                             \
        __syncthreads();                                                              \
        if (s_fail) {                                                                 \
            return; /* 全体一致退出(thread0 不早退 —— 防 __syncthreads 死锁) */    \
        }                                                                             \
        const unsigned int slot =                                                     \
            (unsigned int)(s_prod % (unsigned long long)(*own_nslots));               \
        owl_ar_copy_bytes(reinterpret_cast<unsigned char*>(own) + OWL_AR_HDR_BYTES +  \
                              (size_t)slot * slot_bytes,                              \
                          reinterpret_cast<const unsigned char*>(local) + slab_off,   \
                          bytes);                                                     \
        __threadfence_system(); /* 每线程:本线程写 → 对系统可见 */                 \
        __syncthreads();                                                              \
        if (threadIdx.x == 0) {                                                       \
            __threadfence_system(); /* 发布前再保一次(thread0 视角) */             \
            *own_prod = s_prod + 1ull;                                                \
        }                                                                             \
    }

OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_f32, float, (const float* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_f16, __half, (const __half* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_bf16, __nv_bfloat16, (const __nv_bfloat16* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
