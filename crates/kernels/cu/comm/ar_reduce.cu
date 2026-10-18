// owl 通信算子族 —— allreduce **reduce**(TP2 S2;协议不变量 I2/I3/I4/I5)。
//
// 出处:`docs/arch/TP2-施工设计.md` §3。header 布局与 `ar_stage.cu` **逐字节
// 同构**(单源说明住 ar_stage.cu 头注;err 码:2 = reduce 数据超时)。
//
// 语义:`out`(本端 payload)第 `slab_off` 起 `bytes` 字节 **原地累加**对端
// staging 的 slot[consumed % n_slots](即 `out += peer_partial`),随后发布
// consumed++(它是对端 stage 的 credit)。
//
// 纪律(与 stage 对应;I2 = 本核的主角):
// 1. **不读未写(I2)**:等 `peer.produced > own.consumed` 才读 slot;
// 2. **acquire 序(全线程)**:看到新 produced 之后 `__threadfence_system()` 再读数据
//    (防读被重排到 flag 读之前 —— 与生产者的 release 配对,I4)。
//    ⚠️ **2026-10-17 修**:原实现只有 thread0 acquire,其余线程靠 `__syncthreads()`
//    (block 域,不足 system 域)⇒ 向量化提速后暴露(见内核体内注);
// 3. **超时**(5 s)置 err=2 并放弃(D4);
// 4. **失败路径不提前 return**(shared 标志 + 全员一致退出,防 `__syncthreads` 死锁);
// 5. **累加用 float 中间**(与 elems/add 同律:存储换档,计算精度不换档)。

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

#define OWL_AR_MAGIC 0x4F574C41u
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

__device__ __forceinline__ unsigned long long owl_ar_timer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// ---- dtype 桥(与 cu/elems/add.cu 同款:float 中间精度)----
template <typename T> __device__ __forceinline__ float owl_ar_to_float(T x);
template <> __device__ __forceinline__ float owl_ar_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float owl_ar_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float owl_ar_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ T owl_ar_from_float(float x);
template <> __device__ __forceinline__ float owl_ar_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half owl_ar_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 owl_ar_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// ---- 归约循环:向量化 + 手工展开(★ 2026-10-17 MLP 律)------------------
//
// **为什么必须有这一段**(出处:AR 通道微基准 `comm_ar_bench.rs` + 施工设计 §25):
// 逐元素循环 = 每个元素**一趟串行访存往返**。对端读在 PCIe 上单趟 ~1.41µs
// ⇒ 实测 **363 MB/s**,且**成本与元素数成正比、与字节数无关**(同字节 f32 恰好
// 比 f16 快 2× ⇒ 不是带宽病,是**在飞访存数 = 1/线程**)。128 次 AR/步 ×
// 1.31MB/token ⇒ 3.9ms/token,正是 TP2 prefill 0.24× 的全部账。
//
// 修法 = 16 B 向量(10 倍少的访存次数)+ **8 路手工展开**(先把 8 个 16 B 读
// 全部发出,再做算术)⇒ 在飞 8×16 B/线程。**先把 vs[] 填满再消费**是刻意的:
// 那是给编译器的调度边界(逐条 load→use 会被逐条串行)。
//
// **数值语义逐位不变**:仍是"每元素恰好一次 add",无重排、无树形 ——
// 归约序仍与 shape 解耦(`ar.rs` 的确定性纪律;G8-a 门判据)。
// **尾部语义不变**:不足一个向量的**整元素**走标量尾巴;
// **亚元素字节保持原值**(门判据:非整元素尾部不得被动)。
template <typename T, typename V>
__device__ __forceinline__ void owl_ar_reduce_vec(
    T* __restrict__ dst, const T* __restrict__ src, const size_t n) {
    constexpr int PER_VEC = (int)(sizeof(V) / sizeof(T));
    // **多 block**:线程全局索引 + 全局 stride(块内仍 16B 连续 ⇒ 合并访问不变)
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
    // 尾巴:整元素但不足一个向量(n_vec*PER_VEC .. n);更后的字节不动
    for (size_t j = n_vec * (size_t)PER_VEC + tid; j < n; j += stride) {
        dst[j] = owl_ar_from_float<T>(owl_ar_to_float<T>(dst[j]) + owl_ar_to_float<T>(src[j]));
    }
}

/// 标量兜底(指针 16 B 未对齐时;与 v1 逐位同构 —— 对齐降级只降速不改语义)
template <typename T>
__device__ __forceinline__ void owl_ar_reduce_scalar(
    T* __restrict__ dst, const T* __restrict__ src, const size_t n) {
    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        dst[i] = owl_ar_from_float<T>(owl_ar_to_float<T>(dst[i]) + owl_ar_to_float<T>(src[i]));
    }
}

#define OWL_AR_REDUCE_KERNEL(NAME, T, PARAMS)                                          \
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "AR reduce: 仅 f32/f16/bf16");     \
    extern "C" __global__ void NAME PARAMS {                                           \
        unsigned char* own_b = reinterpret_cast<unsigned char*>(own);                      \
        const unsigned char* peer_b = reinterpret_cast<const unsigned char*>(peer);    \
        volatile unsigned long long* own_cons =                                        \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_CONSUMED); \
        volatile unsigned int* own_err =                                               \
            reinterpret_cast<volatile unsigned int*>(own_b + OWL_AR_OFF_ERR);          \
        const volatile unsigned int* own_magic =                                       \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_MAGIC);  \
        const volatile unsigned int* own_slot_bytes =                                  \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_SLOT_BYTES); \
        const volatile unsigned int* own_nslots =                                      \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_N_SLOTS); \
        const volatile unsigned long long* peer_prod =                                 \
            reinterpret_cast<const volatile unsigned long long*>(peer_b + OWL_AR_OFF_PRODUCED); \
        volatile unsigned long long* own_wait =                                        \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_WAIT_DONE); \
        unsigned int* own_blocks_done =                                                \
            reinterpret_cast<unsigned int*>(own_b + OWL_AR_OFF_BLOCKS_DONE);           \
        __shared__ unsigned long long s_cons;                                          \
        __shared__ int s_fail;                                                         \
        if (threadIdx.x == 0) {                                                        \
            s_cons = 0ull;                                                             \
            s_fail = 0;                                                                \
            if (*own_magic != OWL_AR_MAGIC) {                                          \
                *own_err = 3u;                                                         \
                s_fail = 1;                                                            \
            } else if ((size_t)(*own_slot_bytes) != slot_bytes) {                      \
                *own_err = 3u;                                                         \
                s_fail = 1;                                                            \
            } else {                                                                   \
                /* ★ 多 block 两级等待(2026-10-17;§27.7:瓶颈 = 单 block 打不满本地   \
                 * 访存)。**block 0** 去 PCIe 轮询对端 produced;其余 block 读**本端       \
                 * header** 的 `wait_done`(本地 L2)⇒ 不给对端同一行加 N 倍 PCIe 轮询。 \
                 * 令牌**单调**(= 本端 consumed+1)⇒ 与游标同源、图回放安全。 */        \
                const unsigned long long cons = *own_cons;                             \
                const unsigned long long want = cons + 1ull;                           \
                const unsigned long long t0 = owl_ar_timer_ns();                       \
                unsigned int spins = 0u;                                               \
                if (blockIdx.x == 0) {                                                 \
                    while (*peer_prod <= cons) { /* I2:不读未写 */                    \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                             \
                            __nanosleep(200u);                                         \
                        }                                                              \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {              \
                            *own_err = 2u;                                             \
                            s_fail = 1;                                                \
                            break;                                                     \
                        }                                                              \
                    }                                                                  \
                    if (!s_fail) {                                                     \
                        *own_wait = want;   /* 放行其余 block */                    \
                    }                                                                  \
                } else {                                                               \
                    while (*own_wait < want) { /* 读本端 header = 本地 */          \
                        if (++spins > OWL_AR_SPIN_TIGHT) {                             \
                            __nanosleep(200u);                                         \
                        }                                                              \
                        if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {              \
                            *own_err = 2u;                                             \
                            s_fail = 1;                                                \
                            break;                                                     \
                        }                                                              \
                    }                                                                  \
                }                                                                      \
                s_cons = cons;                                                         \
            }                                                                          \
        }                                                                              \
        __syncthreads();                                                               \
        if (s_fail) {                                                                  \
            return;                                                                    \
        }                                                                              \
        /* ★ acquire **全线程**(2026-10-17 修;原仅 thread0 ⇒ 分工不对称)           \
         * 出处:`tp_transport_bits_gate` 在 3090 Ti 对(0/1,同 PHB)上红、在 3080 对     \
         * (2/3,跨 root complex)上绿 —— **旧核在 0/1 上是绿的** ⇒ 慢把洞盖住了。      \
         * 机理:生产者是**全线程** release(每线程 fence → syncthreads →           \
         * thread0 STGSTRONG.SYS 发布);消费者若只有 thread0 acquire,其余线程只由   \
         * `__syncthreads()`(block 域)排序 ⇒ 加速后的**突发 8×16B 读**会跑到         \
         * 对端写可见之前。对称形式 = 全体 acquire 后再读(与生产者的 release 配对)。*/\
        __threadfence_system();                                                        \
        const unsigned int slot =                                                      \
            (unsigned int)(s_cons % (unsigned long long)(*own_nslots));                \
        const T* src = reinterpret_cast<const T*>(                                    \
            peer_b + OWL_AR_HDR_BYTES + (size_t)slot * slot_bytes);                    \
        T* dst = reinterpret_cast<T*>(                                                \
            reinterpret_cast<unsigned char*>(out) + slab_off);                         \
        const size_t n = bytes / sizeof(T);                                            \
        const unsigned long long da = reinterpret_cast<unsigned long long>(dst);       \
        const unsigned long long sa = reinterpret_cast<unsigned long long>(src);       \
        if (((da | sa) & 15ull) == 0ull) {                                             \
            owl_ar_reduce_vec<T, uint4>(dst, src, n); /* 快路:16 B 向量 + 8 路展开 */ \
        } else {                                                                       \
            owl_ar_reduce_scalar<T>(dst, src, n);     /* 未对齐:降级只降速 */        \
        }                                                                              \
        __threadfence_system();                                                        \
        __syncthreads();                                                               \
        /* ★ **最后一块发布 consumed**(多 block;本块数据已系统可见)。 */         \
        if (threadIdx.x == 0) {                                                        \
            const unsigned int done = atomicAdd(own_blocks_done, 1u);                  \
            if (done == gridDim.x - 1u) {                                              \
                __threadfence_system();                                                \
                *own_blocks_done = 0u;  /* 回零:同流保序 ⇒ 下一次调用必见 0 */        \
                *own_cons = s_cons + 1ull; /* 对端 stage 的 credit */                  \
            }                                                                          \
        }                                                                              \
    }

OWL_AR_REDUCE_KERNEL(owl_comm_ar_reduce_f32, float, (float* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_REDUCE_KERNEL(owl_comm_ar_reduce_f16, __half, (__half* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_REDUCE_KERNEL(owl_comm_ar_reduce_bf16, __nv_bfloat16, (__nv_bfloat16* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
