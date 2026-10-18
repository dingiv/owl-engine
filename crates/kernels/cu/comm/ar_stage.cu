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
//     +32  u64 wait_done   多 block 两级等待令牌(**单调** = 本端 produced+1)
//     +40  u32 blocks_done 完成块计数(最后一块发布 produced)
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

// ── ★ **前置清单**(2026-10-17;照 NCCL `contrib/custom_algos` 的写法:核头逐条列,
//    误用要能"对着单子查",而不是散在各处)──────────────────────────────────
// ① **两 rank 同 plan**:`slot_bytes` / `n_slots` / `slab_off` 逐参一致(§3 不变量 I5,
//    序列同构的前提)。本核只校验**本端** header 与实参相符(不符 ⇒ err=3)。
// ② **header 必须先初始化**,且 `produced` / `consumed` / `wait_done` / `blocks_done`
//    **四格全 0**。⚠️ 其中 `wait_done` 最险:若留**非 0 残值**且 ≥ 本轮 want,
//    非 0 号 block 会**跳过两级等待** ⇒ 静默读到未就绪的对端数据(无报错)。
//    ⇒ 这就是本核加 `n_slots >= 2` 之外**唯一无法自检**的一条,必须在分配侧保证
//    (见 `ar.rs::hdr` 注释与 boot 期握手)。
// ③ **载荷 ≤ `slot_bytes`**(v1 单块):超了是调用方错,解释器层已响亮拒。
// ④ **`n_slots >= 2`**:credit 环靠"上一轮那格已被对端读走"兜底;`n_slots == 1`
//    会让本端立刻覆盖对端正要读的那一格 ⇒ 本核**响亮拒**(err=3;2026-10-17 加)。
// ⑤ **发射几何**:block 尺寸纯发射参数(核内全用 `blockDim.x` / `gridDim.x`);
//    grid 由 `server::comm::MAX_THREADS / BLOCK` 算 ⇒ 与核无耦合,故两 rank 之间
//    **不要求同 grid**(它只决定本端工作的摊法,不进协议)。
// ⑥ **超时**:credit 5s ⇒ err=1;数据 5s ⇒ err=2(D4:宁可整轮失败,不静默出结果)。

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
#define OWL_AR_OFF_WAIT_DONE 32u
#define OWL_AR_OFF_BLOCKS_DONE 40u

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
///
/// ★ **2026-10-17 MLP 律**:主循环 **8 路手工展开 + 逐通道谓词**(先把 8 个 16 B 读
/// 全发出再写回)——— 与 reduce 同病同药:逐条 `load→store` 的在飞访存 = 1/线程,
/// 本地往返 ~0.3µs。
/// **单循环形态(无"余数慢路")**:原"展开主循环 + 逐条余数循环"在小 `n16`
/// (余数段占主体)落回逐条速度 —— 基准里那正是"小载荷比大载荷慢"的怪相来源
/// (`comm_ar_bench` 尺寸阶梯:20KB 慢于 128KB)。谓词展开后**所有尺寸同一条路**。
/// 语义逐字节不变(纯搬移,无算术)。
__device__ __forceinline__ void owl_ar_copy_bytes(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    const size_t bytes) {
    // **多 block**:全局线程索引 + 全局 stride(块内仍 16 B 连续 ⇒ 合并访问不变)
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

// ---- stage 宏实例:纯字节搬运(dtype 只决定"拷多少字节",不改核体;
//      保留类型化指针让"表 ⇔ .cu 签名"机器互证可查)。----
#define OWL_AR_STAGE_KERNEL(NAME, T, PARAMS)                                          \
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "AR stage: 仅 f32/f16/bf16");      \
    extern "C" __global__ void NAME PARAMS {                                          \
        unsigned char* own_b = reinterpret_cast<unsigned char*>(own);                      \
        const unsigned char* peer_b = reinterpret_cast<const unsigned char*>(peer);   \
        volatile unsigned long long* own_prod =                                       \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_PRODUCED); \
        volatile unsigned long long* own_cons =                                       \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_CONSUMED); \
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
        const volatile unsigned long long* peer_prod =                                \
            reinterpret_cast<const volatile unsigned long long*>(peer_b + OWL_AR_OFF_PRODUCED); \
        volatile unsigned long long* own_wait =                                        \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_WAIT_DONE); \
        unsigned int* own_blocks_done =                                                \
            reinterpret_cast<unsigned int*>(own_b + OWL_AR_OFF_BLOCKS_DONE);           \
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
            } else if (*own_nslots < 2u) {                                            \
                /* 前置清单 ④:单格环会让本端立刻覆盖对端正要读的那格 ⇒ 响亮拒 */    \
                *own_err = 3u;                                                        \
                s_fail = 1;                                                           \
            } else {                                                                  \
                /* (a) **代发 credit**(2026-10-17 CE 前置;对 v1 亦安全)          \
                 * `own_cons = own_prod` = "我读过的对端调用数"。**值与 reduce 的发布   \
                 * 收敛**(reduce_i 发 i+1;stage_{i+1} 也发 i+1)⇒ 无条件加不会与      \
                 * reduce 打架 —— 这也是它不必进门的原因。CE 路径下 add 绑到 inbox,   \
                 * staging.consumed 就只剩这一个写者。 */                             \
                const unsigned long long prod0 = *own_prod;                           \
                *own_cons = prod0;                                                    \
                const unsigned long long prod = prod0;                                \
                const unsigned long long n_slots = (unsigned long long)(*own_nslots); \
                /* I1:不覆盖未读 —— peer.consumed + n_slots > own.produced */        \
                /* 两级等待(与 reduce 同构):**block 0** 去 PCIe 读对端 consumed;      \
                 * 其余 block 读**本端 header** 的 wait_done(本地 L2)。               \
                 * ⚠️ block 0 **无条件**发布令牌(即便本档无需 credit)——否则小档       \
                 * (`prod < n_slots`)里其余 block 会等到超时(写这刀时踩过的坑)。 */  \
                const unsigned long long need =                                    \
                    prod >= n_slots ? (prod - n_slots + 1ull) : 0ull;              \
                const unsigned long long t0 = owl_ar_timer_ns();                   \
                unsigned int spins = 0u;                                           \
                if (blockIdx.x == 0) {                                             \
                    while (*peer_cons < need) {                                    \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                         \
                            __nanosleep(200u);                                     \
                        }                                                          \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {          \
                            *own_err = 1u;                                         \
                            s_fail = 1;                                            \
                            break;                                                 \
                        }                                                          \
                    }                                                              \
                    if (!s_fail) {                                                 \
                        __threadfence_system();                                    \
                        *own_wait = prod + 1ull; /* 无条件放行其余 block */        \
                    }                                                              \
                } else {                                                           \
                    while (*own_wait < prod + 1ull) {                              \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                         \
                            __nanosleep(200u);                                     \
                        }                                                          \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {          \
                            *own_err = 1u;                                         \
                            s_fail = 1;                                            \
                            break;                                                 \
                        }                                                          \
                    }                                                              \
                    if (!s_fail) {                                                 \
                        __threadfence_system();                                    \
                    }                                                              \
                }                                                                  \
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
        /* ★ **最后一块发布 produced**(多 block;本块数据已系统可见)。 */          \
        if (threadIdx.x == 0) {                                                       \
            const unsigned int done = atomicAdd(own_blocks_done, 1u);                  \
            if (done == gridDim.x - 1u) {                                              \
                __threadfence_system();                                                \
                *own_blocks_done = 0u;  /* 回零:同流保序 ⇒ 下一次调用必见 0 */        \
                if (!s_fail) {                                                        \
                    *own_prod = s_prod + 1ull;                                        \
                    /* (b) **尾部 rendezvous**(CE 前置:CE 拷是主机侧的,不能自旋     \
                     * 等对端 ⇒ "对端已发布本轮"必须由核来等)。放在**最后到的那一块** \
                     * 上最省:其余块已退出 ⇒ 核完成时刻 = rendezvous 结束。        \
                     * 对 v1 亦安全:这个等待原在 reduce 里,只是提前到同一条流上。  \
                     * 无死锁:双方都先发布自己的 produced 再等。 */                 \
                    const unsigned long long want = s_prod + 1ull;                    \
                    const unsigned long long t0 = owl_ar_timer_ns();                  \
                    unsigned int spins = 0u;                                          \
                    while (*peer_prod < want) {                                       \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                            \
                            __nanosleep(200u);                                        \
                        }                                                             \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {             \
                            *own_err = 1u;                                            \
                            break;                                                    \
                        }                                                             \
                    }                                                                 \
                    __threadfence_system();                                           \
                }                                                                      \
            }                                                                          \
        }                                                                              \
    }

OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_f32, float, (const float* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_f16, __half, (const __half* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_STAGE_KERNEL(owl_comm_ar_stage_bf16, __nv_bfloat16, (const __nv_bfloat16* __restrict__ local, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
