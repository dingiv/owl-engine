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
//    这是本核**唯一无法自检**的一条,必须在分配侧保证
//    (见 `ar.rs::hdr` 注释与 boot 期握手)。
// ③ **载荷 ≤ `slot_bytes`**(v1 单块):超了是调用方错,解释器层已响亮拒。
// ④ **`n_slots ≥ 1`**(2026-10-17 与 `ar_stage.cu` **同步放宽**;原 `≥ 2`):
//    本核无 n_slots 守卫(`slot = consumed % n_slots` 本身只要 n_slots ≥ 1 就成立),
//    与 stage 核的守卫**必须同律** —— 两核读同一个 header,一个要求 ≥2、一个只管 ≥1
//    就会出现"stage 拒、reduce 跑"的**半途错值**(CE 刀就是被那道不对称守卫打死的:
//    stage 核 err=3 退出 ⇒ 从不发 produced ⇒ 对端 reduce 5s 超时 err=2,
//    payload 一个字节都没加 —— 见 TP2-施工设计 §31 的四探针列表)。
// ⑤ **发射几何**:block 尺寸纯发射参数(核内全用 `blockDim.x` / `gridDim.x`);
//    grid 由 `server::comm::MAX_THREADS / BLOCK` 算 ⇒ 与核无耦合,故两 rank 之间
//    **不要求同 grid**(它只决定本端工作的摊法,不进协议)。
// ⑥ **超时**:credit 5s ⇒ err=1;数据 5s ⇒ err=2(D4:宁可整轮失败,不静默出结果)。

#include <cuda_fp16.h>
#include <cuda_fp8.h>
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
// ⏳ 2026-10-19 试过"到场 barrier"(把票写进对端 header;§57 记档)—— **L1 门失败,已回退**:
//   根因:vLLM 的 barrier 依赖"票写进对端",而本引擎的规则是 **header 留本端**(§36.8/§37);
//   且 1-stage 的载荷由**上游 GEMM 多 block 多线程**写,"单线程 fence"无法为它的系统可见性背书
//   (vLLM 的载荷是紧邻核的一条 CE memcpy,流序可背书)。⇒ **现有的 produced 轮询协议
//   其实已经是"到场 barrier 的 pull 形态"**;D 结构要落地必须连"flag 写向对端"一起改。

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
// ── ★ fp8(e4m3)交换的**数值等价件**(2026-10-19 立案;设计卷 §47)───────────────
//
// 语义:reduce 的**加法两侧各自过一遍 e4m3**:
//     out[i] = D_{s_g(a)}(q(a_i)) + D_{s_g(b)}(q(b_i))
// 尺度 `s_g(x) = max|x| over 组 g / 448`,**组 = 元素下标的集合**(两 rank 逐位同式)。
// ⇒ 两 rank 算出的和 = **同一对值的同一个加法**(IEEE 加法交换) ⇒ **L1 逐位一致不被破坏**。
// ⇒ 且这与**真实 fp8 交换数值等价**(送出前量化一次、收端反量化,共用同一组尺度)
//    —— 所以本件既是**判据先行**的测量仪,又是将来那份实现的一半。
//
// 粒度:每线程每次 8 路展开的 8×PER_VEC 个元素 = 一组(bf16 ⇒ **64 元素/尺度**)。
//   比 TRT-LLM 的 per-warp 496 更细(更保守);scale 开销 2B/64 元素 = 3.1% 载荷。
// ⚠️ **开关是编译期的**(`OWL_AR_QDQ`):本轮只用来**拿数值**,
//    拿完再走配置面 + 表参数(§47 的第二步)。
#ifndef OWL_AR_QDQ
#define OWL_AR_QDQ 0
#endif
#define OWL_AR_E4M3_MAX 448.0f

/// 调用点派发:`OWL_AR_QDQ=1` 时整条快路换成 fp8 变体(几何/签名/语义全同)
// ⚠️ 类型必须**当参数传**:核体是宏展开(2-stage/1-stage 两个宏都把类型名当宏参
//    文本替换),若把 `T` 写进本宏的**定义**里,外层宏替换不到 ⇒ "identifier T undefined"
#if OWL_AR_QDQ
#define OWL_AR_REDUCE_FAST(TY, D, S, N) owl_ar_reduce_vec_fp8<TY, uint4>(D, S, N)
#else
#define OWL_AR_REDUCE_FAST(TY, D, S, N) owl_ar_reduce_vec<TY, uint4>(D, S, N)
#endif

/// 组尺度(全零组取 1.0 ⇒ 量化结果仍为 0,两 rank 同式)
__device__ __forceinline__ float owl_ar_qdq_scale(float m) {
    return (m > 0.0f) ? (m / OWL_AR_E4M3_MAX) : 1.0f;
}

/// e4m3 往返(round-to-nearest;`inv = 1/scale`)
__device__ __forceinline__ float owl_ar_e4m3_round(float x, float inv) {
    __nv_fp8_e4m3 q(x * inv);
    return static_cast<float>(q) / inv;
}

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

#if OWL_AR_QDQ
// ── fp8 档:同一个函数名、同一套几何,但加法两侧各过一遍 e4m3 ──────────────
template <typename T, typename V>
__device__ __forceinline__ void owl_ar_reduce_vec_fp8(
    T* __restrict__ dst, const T* __restrict__ src, const size_t n) {
    constexpr int PER_VEC = (int)(sizeof(V) / sizeof(T));
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t n_vec = n / (size_t)PER_VEC;
    V* d4 = reinterpret_cast<V*>(dst);
    const V* s4 = reinterpret_cast<const V*>(src);
    for (size_t i = tid; i < n_vec; i += 8 * stride) {
        V vs[8], vd[8];
        bool ok[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const size_t j = i + (size_t)k * stride;
            ok[k] = j < n_vec;
            if (ok[k]) {
                vs[k] = s4[j];
                vd[k] = d4[j];
            }
        }
        // ① 组尺度:两侧各自的 max(组 = 本线程本轮这 8 个向量的元素下标集合)
        float ms = 0.0f, md = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (!ok[k]) { continue; }
            const T* sp = reinterpret_cast<const T*>(&vs[k]);
            const T* dp = reinterpret_cast<const T*>(&vd[k]);
#pragma unroll
            for (int m = 0; m < PER_VEC; ++m) {
                ms = fmaxf(ms, fabsf(owl_ar_to_float<T>(sp[m])));
                md = fmaxf(md, fabsf(owl_ar_to_float<T>(dp[m])));
            }
        }
        const float inv_s = 1.0f / owl_ar_qdq_scale(ms);
        const float inv_d = 1.0f / owl_ar_qdq_scale(md);
        // ② 各过一遍 e4m3 再相加(与真实 fp8 交换同一对值)
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (!ok[k]) { continue; }
            const T* sp = reinterpret_cast<const T*>(&vs[k]);
            T* dp = reinterpret_cast<T*>(&vd[k]);
#pragma unroll
            for (int m = 0; m < PER_VEC; ++m) {
                const float a = owl_ar_e4m3_round(owl_ar_to_float<T>(dp[m]), inv_d);
                const float b = owl_ar_e4m3_round(owl_ar_to_float<T>(sp[m]), inv_s);
                dp[m] = owl_ar_from_float<T>(a + b);
            }
            d4[i + (size_t)k * stride] = vd[k];
        }
    }
    // 尾巴:逐元素纯 e4m3 舍入(尺度 = 元素自身 ⇒ 等价于不分组;两 rank 同式)
    for (size_t j = n_vec * (size_t)PER_VEC + tid; j < n; j += stride) {
        const float x = owl_ar_to_float<T>(dst[j]);
        const float y = owl_ar_to_float<T>(src[j]);
        const float rx = owl_ar_e4m3_round(x, x != 0.0f ? OWL_AR_E4M3_MAX / fabsf(x) : 1.0f);
        const float ry = owl_ar_e4m3_round(y, y != 0.0f ? OWL_AR_E4M3_MAX / fabsf(y) : 1.0f);
        dst[j] = owl_ar_from_float<T>(rx + ry);
    }
}
#endif

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
        /* ★ **数据块**(2026-10-18;push 档 §36.7):slot 数据所在块。                \
         * v1 = `peer`(读对端);push = `own`(读本端 —— 对端已写进来)。               \
         * 与 header 指针**分开**的理由:`wait_done`/`blocks_done` 是本核内部      \
         * 的块间令牌,**必须留在本端**(否则 15 个 block 紧自旋远端读,实测把      \
         * AR 从 601µs 拉到 1225µs —— §36.8)。 */                                  \
        const unsigned char* data_b =                                                  \
            reinterpret_cast<const unsigned char*>(data);                              \
        volatile unsigned long long* own_prod =                                        \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_PRODUCED); \
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
                    /* (a) 发布 produced:上游 GEMM 的写已由流序 + L2 可见。         \
                     * `__threadfence_system()` 在此是**屏障点**(它只能排序本线程   \
                     * 的写 —— 上游写者的可见性靠"核结束"这条硬件事实;前置清单 ④)。 */ \
                    __threadfence_system();                                            \
                    *own_prod = *own_prod + 1ull;                                      \
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
            data_b + OWL_AR_HDR_BYTES + (size_t)slot * slot_bytes);                    \
        T* dst = reinterpret_cast<T*>(                                                \
            reinterpret_cast<unsigned char*>(out) + slab_off);                         \
        const size_t n = bytes / sizeof(T);                                            \
        const unsigned long long da = reinterpret_cast<unsigned long long>(dst);       \
        const unsigned long long sa = reinterpret_cast<unsigned long long>(src);       \
        if (((da | sa) & 15ull) == 0ull) {                                             \
            OWL_AR_REDUCE_FAST(T, dst, src, n); /* 快路:16 B 向量 + 8 路展开 */ \
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

OWL_AR_REDUCE_KERNEL(owl_comm_ar_reduce_f32, float, (float* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const unsigned char* __restrict__ data, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_REDUCE_KERNEL(owl_comm_ar_reduce_f16, __half, (__half* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const unsigned char* __restrict__ data, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_REDUCE_KERNEL(owl_comm_ar_reduce_bf16, __nv_bfloat16, (__nv_bfloat16* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const unsigned char* __restrict__ data, const size_t bytes, const size_t slot_bytes, const size_t slab_off))

// ============================================================================
// ★ **1-stage allreduce**(2026-10-18;对称缓冲 ⇒ 融合核)—— TP2-施工设计.md §35
// ============================================================================
//
// 与上面 2-stage 两核版(`ar_stage` + `ar_reduce`)的差别,一句话:
// **输入缓冲本身就是对称的** ⇒ 不需要 stage 拷贝,也不需要独立的 stage 核。
//
// 对齐参照:**vLLM `cross_device_reduce_1stage`**(`world_size_ == 2` 恒走这一支;
// 见 `packages/vllm/csrc/custom_all_reduce.cuh`)—— 它的 `result[idx] = Σ dp.ptrs[r][idx]`
// 就是"原地累加对端缓冲",因为 vLLM 的输入张量**从引擎启动起就是对称的**。
//
// # 本核做的五件事(顺序即语义)
//
// ① **发布 `own.produced++`** —— "本槽的数据就绪"。上游写者是**GEMM 核**(不是本核):
//    它在本核之前由**流序**完成,其写在核结束后已落到 L2 ⇒ 对端 P2P 读可见。
//    ⚠️ 这是本核与 2-stage 在**可见性**上的唯一结构差:2-stage 的写者是 stage 核自己
//    (所以它在发布前做 per-thread fence);这里写者是上游核,本核的 `__threadfence_system()`
//    只能作为屏障点,不能追认上游线程的写 —— 依赖的是"核结束 ⇒ 写对设备可见(到 L2)"
//    这条硬件事实 + 对端读走 L2。**门(`ar_transport_gate`)判数值,红即知。**
// ② **I2**:等 `peer.produced > own.consumed` 才读对端槽。
// ③ `out[i] += peer_data[i]`(**原地**;向量化 + 8 路展开,复用 `owl_ar_reduce_vec`)。
// ④ 发布 `own.consumed++` —— "我读完了对端这一轮"。
// ⑤ **credit**:等 `peer.consumed + N_SLOTS > own.produced` ⇒ 保护的是**下一个 GEMM
//    覆盖本槽**(不是本核的写 —— 本核原地加不会越过自己的数据)。放在"最后一块"里,
//    所以本核的**完成时刻**就把"下游 GEMM 可以安全覆盖"这件事钉住了。
//
// # 参数(比 2-stage 少两个)
//
// `out` = **本端槽本身**(就是 `own` 那一套里的 payload 槽;不是私有 payload);
// `peer_data` = 对端槽的跨卡别名;两个 header 分开传(槽与 header 不同块)。
// 没有 `slot_bytes` / `slab_off`:槽即块,偏移由调用方在**块**一级解决。
//
// # 前置清单(与 2-stage 同律,只有 ④ 变了)
//
// ① 两 rank 同参(header 的 slot_bytes/n_slots 只用于 tombstones;本核不读 stride)
// ② header 四格(produced/consumed/wait_done/blocks_done)必须全 0 —— 分配侧保证
// ③ **`out` 与 `peer_data` 必须属于同一次 AR 的那一套槽**(调用方的责任)
// ④ 对称性:**GEMM 必须直接写进本槽**(否则 ① 的"数据就绪"就是谎话)
// ⑤ 发射几何纯参数(与 2-stage 同)
//
// # 与 2-stage 的收益账(5.24MB / 3080 对)
//
// 省掉:(a) 一个图节点(每 token ~71 次 AR × ~1.5µs 派发税)
//       (b) stage 段引入的跨卡同步往返(2-stage 的 `AR - RR` ≈ 69µs 里的主要部分)
//       (c) 一次本地拷贝(0.36µs,可忽略 —— 收益不在它)
// 代价:GEMM 的输出目标必须是 peer-shared 块(engine 侧:分配点替换 + 对称 arena)

#define OWL_AR_1STAGE_KERNEL(NAME, T, PARAMS)                                          \
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "AR 1stage: 仅 f32/f16/bf16");     \
    extern "C" __global__ void NAME PARAMS {                                           \
        unsigned char* own_b = reinterpret_cast<unsigned char*>(own);                  \
        const unsigned char* peer_b = reinterpret_cast<const unsigned char*>(peer);    \
        volatile unsigned long long* own_prod =                                        \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_PRODUCED); \
        volatile unsigned long long* own_cons =                                        \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_CONSUMED); \
        volatile unsigned int* own_err =                                               \
            reinterpret_cast<volatile unsigned int*>(own_b + OWL_AR_OFF_ERR);          \
        const volatile unsigned int* own_magic =                                       \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_MAGIC);  \
        volatile unsigned long long* own_wait =                                        \
            reinterpret_cast<volatile unsigned long long*>(own_b + OWL_AR_OFF_WAIT_DONE); \
        unsigned int* own_blocks_done =                                                \
            reinterpret_cast<unsigned int*>(own_b + OWL_AR_OFF_BLOCKS_DONE);           \
        const volatile unsigned long long* peer_prod =                                 \
            reinterpret_cast<const volatile unsigned long long*>(peer_b + OWL_AR_OFF_PRODUCED); \
        const volatile unsigned long long* peer_cons =                                 \
            reinterpret_cast<const volatile unsigned long long*>(peer_b + OWL_AR_OFF_CONSUMED); \
        __shared__ unsigned long long s_cons;                                          \
        __shared__ int s_fail;                                                         \
        if (threadIdx.x == 0) {                                                        \
            s_cons = 0ull;                                                             \
            s_fail = 0;                                                                \
            if (*own_magic != OWL_AR_MAGIC) {                                          \
                *own_err = 3u;                                                         \
                s_fail = 1;                                                            \
            } else {                                                                   \
                const unsigned long long cons = *own_cons;                             \
                const unsigned long long want = cons + 1ull;                           \
                const unsigned long long t0 = owl_ar_timer_ns();                       \
                unsigned int spins = 0u;                                               \
                if (blockIdx.x == 0) {                                                 \
                    /* (a) 发布 produced:上游 GEMM 的写已由流序 + L2 可见。         \
                     * `__threadfence_system()` 在此是**屏障点**(它只能排序本线程   \
                     * 的写 —— 上游写者的可见性靠"核结束"这条硬件事实;前置清单 ④)。 */ \
                    __threadfence_system();                                            \
                    *own_prod = *own_prod + 1ull;                                      \
                    /* (b) I2:不读未写 */                                           \
                    while (*peer_prod <= cons) {                                       \
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
                        *own_wait = want; /* 无条件放行其余 block */                 \
                    }                                                                  \
                } else {                                                               \
                    while (*own_wait < want) {                                         \
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
        __threadfence_system(); /* acquire 全线程(与 2-stage 同律) */                 \
        T* dst = reinterpret_cast<T*>(out);                                            \
        const T* src = peer_data;                                                      \
        const size_t n = bytes / sizeof(T);                                            \
        const unsigned long long da = reinterpret_cast<unsigned long long>(dst);       \
        const unsigned long long sa = reinterpret_cast<unsigned long long>(src);       \
        if (((da | sa) & 15ull) == 0ull) {                                             \
            OWL_AR_REDUCE_FAST(T, dst, src, n);                                  \
        } else {                                                                       \
            owl_ar_reduce_scalar<T>(dst, src, n);                                      \
        }                                                                              \
        __threadfence_system();                                                        \
        __syncthreads();                                                               \
        if (threadIdx.x == 0) {                                                        \
            const unsigned int done = atomicAdd(own_blocks_done, 1u);                  \
            if (done == gridDim.x - 1u) {                                              \
                __threadfence_system();                                                \
                *own_blocks_done = 0u; /* 回零:同流保序 ⇒ 下一次必见 0 */             \
                *own_cons = s_cons + 1ull;                                             \
                /* (c) credit:保护**下一个 GEMM 覆盖本槽**。                          \
                 * 判据与 2-stage 的 I1 同式(peer.consumed + n_slots > produced),    \
                 * n_slots 恒 2(engine 侧双槽乒乓;见 TP2 §35)。               */    \
                /* need 与 2-stage 的 I1 **逐字同式**(否则 1-stage 会多等一轮、     \
                 * 白白吃掉流水余量 —— 2026-10-18 首测就是这么写严了一档):          \
                 * 2-stage 用 `prod_before - n_slots + 1`;本核的 `p` 是**发布后**     \
                 * 的值 ⇒ `prod_before = p - 1` ⇒ `need = p - n_slots` = `p - 2`。   */  \
                const unsigned long long p = *own_prod;                                \
                const unsigned long long need = (p >= 2ull) ? (p - 2ull) : 0ull;      \
                const unsigned long long t1 = owl_ar_timer_ns();                       \
                unsigned int spins2 = 0u;                                              \
                while (*peer_cons < need) {                                            \
                    if (++spins2 > OWL_AR_SPIN_TIGHT) {                                \
                        __nanosleep(200u);                                             \
                    }                                                                  \
                    if (owl_ar_timer_ns() - t1 > OWL_AR_TIMEOUT_NS) {                  \
                        *own_err = 1u;                                                 \
                        break;                                                         \
                    }                                                                  \
                }                                                                      \
            }                                                                          \
        }                                                                              \
    }

OWL_AR_1STAGE_KERNEL(owl_comm_ar_1stage_f32, float, (float* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const float* __restrict__ peer_data, const size_t bytes))
OWL_AR_1STAGE_KERNEL(owl_comm_ar_1stage_f16, __half, (__half* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const __half* __restrict__ peer_data, const size_t bytes))
OWL_AR_1STAGE_KERNEL(owl_comm_ar_1stage_bf16, __nv_bfloat16, (__nv_bfloat16* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const __nv_bfloat16* __restrict__ peer_data, const size_t bytes))
