// owl 通信算子族 —— **barrier-only AR-2**(2026-10-23;小包 graph 专用)
//
// # 与 fused 的区别(一句话)
//
// fused 把 stage/reduce 合到一核,但仍有 **尾部 rendezvous + wait_done +
// 全线程 `__threadfence_system()`**。本核只保留 v1 的
// **相对 produced/consumed 游标 + 双槽乒乓**,一核内完成:
//
//   等 credit → 写本端槽 → 发布 produced → 等 peer.produced → 读对端槽加
//   → 发布 consumed
//
// 没有尾部 rendezvous、没有 wait_done、没有全线程 system fence;
// 游标是**相对量**(不是绝对 epoch),graph capture/warmup 两侧差一拍也不会死锁;
// 两槽乒乓 + credit 保证不覆盖未读槽。
//
// # 对齐参照
//
// `packages/sglang/.../custom_all_reduce.cuh` 的 `cross_device_reduce_1stage`
// (world=2 固定一阶段;起/止 barrier + release/acquire flag,无信用/尾部
// rendezvous)。SGLang 10KB 实测 9.3µs;owl fused+push 26.3µs。
//
// # 协议(与 v1 逐字同式,只去掉尾部 rendezvous)
//
// - 本端 `produced` / `consumed` 是本端游标(只本端写;对端经只读别名读);
// - 写前 credit: `peer.consumed + n_slots > own.produced`(不覆盖未读槽);
// - 写 `slot = produced % n_slots`,然后 `st.release.sys` 发布 `produced+1`;
// - 读前: `peer.produced > own.consumed`(不读未写);
// - 读 `slot = consumed % n_slots`,加到 payload,`st.release.sys` 发布
//   `consumed+1`。
//
// 数据可见性: 写数据 → `__syncthreads` → 线程 0 release 发布游标;
// 对端 acquire 读游标 → `__syncthreads` → 读数据。只有游标线程做 system-scope
// release/acquire,没有全线程 system fence(与 SGLang 同构)。
//
// # 前置
//
// ① 单 block(host 强制 grid=1);② 两 rank 同参;③ header 四格初始 0;
// ④ `peer` 只读别名即可(本核不写对端 header/data);
// ⑤ `bytes` 只走小包档(host 门 ≤256KiB)。
//
// 失败: magic 错 / header err / 5s 超时 ⇒ 置 `own.err=1` 并让本 block 返回;
// 对端会由自己的超时收口(不静默读半状态)。

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define OWL_AR_MAGIC 0x4F574C41u
#define OWL_AR_HDR_BYTES 64u
#define OWL_AR_OFF_SLOT_BYTES 16u
#define OWL_AR_OFF_N_SLOTS 20u
#define OWL_AR_OFF_MAGIC 24u
#define OWL_AR_OFF_ERR 28u
// AR-2 专用游标(与 v1/fused 的 produced@0/consumed@8 分开;
// 否则 f8 prefill 的 v1 会把 produced 每 AR +2,AR2 的奇偶立即错位)
#define OWL_AR_OFF_PROD_SELF 48u
#define OWL_AR_OFF_CONS_SELF 56u
#define OWL_AR_TIMEOUT_NS 5000000000ull
#define OWL_AR_SPIN_TIGHT 4096u

__device__ __forceinline__ unsigned long long owl_ar_timer_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void owl_ar_st_release_u32(volatile unsigned int* p, unsigned int v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%1], %0;" ::"r"(v), "l"((const void*)p));
#else
    __threadfence_system();
    *p = v;
#endif
}

__device__ __forceinline__ unsigned int owl_ar_ld_acquire_u32(const volatile unsigned int* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    unsigned int v;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"((const void*)p));
    return v;
#else
    __threadfence_system();
    return *p;
#endif
}

// ---- 16B 向量拷贝(与 ar_fused 同尺:8 路在飞 + 标量尾) ----
__device__ __forceinline__ void owl_ar_copy_bytes(
    unsigned char* __restrict__ dst,
    const unsigned char* __restrict__ src,
    const size_t bytes) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const unsigned long long da = reinterpret_cast<unsigned long long>(dst);
    const unsigned long long sa = reinterpret_cast<unsigned long long>(src);
    if (((da | sa) & 15ull) == 0ull) {
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
        for (size_t j = (n16 << 4) + tid; j < bytes; j += stride) {
            dst[j] = src[j];
        }
    } else {
        for (size_t i = tid; i < bytes; i += stride) {
            dst[i] = src[i];
        }
    }
}

template <typename T>
__device__ __forceinline__ float owl_ar_to_float(T x);
template <> __device__ __forceinline__ float owl_ar_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float owl_ar_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float owl_ar_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T>
__device__ __forceinline__ T owl_ar_from_float(float x);
template <> __device__ __forceinline__ float owl_ar_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ __half owl_ar_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ __nv_bfloat16 owl_ar_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <typename T, typename V>
__device__ __forceinline__ void owl_ar_add_vec(
    T* __restrict__ dst,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const size_t n) {
    constexpr int PER_VEC = (int)(sizeof(V) / sizeof(T));
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const size_t n_vec = n / (size_t)PER_VEC;
    V* d4 = reinterpret_cast<V*>(dst);
    const V* a4 = reinterpret_cast<const V*>(a);
    const V* b4 = reinterpret_cast<const V*>(b);
    for (size_t i = tid; i < n_vec; i += 8 * stride) {
        V va[8], vb[8], vd[8];
        bool ok[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const size_t j = i + (size_t)k * stride;
            ok[k] = j < n_vec;
            if (ok[k]) {
                va[k] = a4[j];
                vb[k] = b4[j];
            }
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (!ok[k]) {
                continue;
            }
            const T* ap = reinterpret_cast<const T*>(&va[k]);
            const T* bp = reinterpret_cast<const T*>(&vb[k]);
            T* dp = reinterpret_cast<T*>(&vd[k]);
#pragma unroll
            for (int m = 0; m < PER_VEC; ++m) {
                dp[m] = owl_ar_from_float<T>(owl_ar_to_float<T>(ap[m]) + owl_ar_to_float<T>(bp[m]));
            }
            d4[i + (size_t)k * stride] = vd[k];
        }
    }
    for (size_t j = n_vec * (size_t)PER_VEC + tid; j < n; j += stride) {
        dst[j] = owl_ar_from_float<T>(owl_ar_to_float<T>(a[j]) + owl_ar_to_float<T>(b[j]));
    }
}

#define OWL_AR_BARRIER_KERNEL(NAME, T, PARAMS)                                              \
    static_assert(sizeof(T) == 4 || sizeof(T) == 2, "AR barrier: 仅 f32/f16/bf16");         \
    extern "C" __global__ void NAME PARAMS {                                                \
        unsigned char* own_b = reinterpret_cast<unsigned char*>(own);                       \
        const unsigned char* peer_b = reinterpret_cast<const unsigned char*>(peer);         \
        unsigned char* w_b = reinterpret_cast<unsigned char*>(data_w);                      \
        const unsigned char* r_b = reinterpret_cast<const unsigned char*>(data_r);          \
        volatile unsigned int* own_prod =                                                   \
            reinterpret_cast<volatile unsigned int*>(own_b + OWL_AR_OFF_PROD_SELF);         \
        volatile unsigned int* own_cons =                                                   \
            reinterpret_cast<volatile unsigned int*>(own_b + OWL_AR_OFF_CONS_SELF);         \
        const volatile unsigned int* peer_prod =                                            \
            reinterpret_cast<const volatile unsigned int*>(peer_b + OWL_AR_OFF_PROD_SELF);  \
        const volatile unsigned int* peer_cons =                                            \
            reinterpret_cast<const volatile unsigned int*>(peer_b + OWL_AR_OFF_CONS_SELF);  \
        volatile unsigned int* own_err =                                                    \
            reinterpret_cast<volatile unsigned int*>(own_b + OWL_AR_OFF_ERR);               \
        const volatile unsigned int* own_magic =                                            \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_MAGIC);       \
        const volatile unsigned int* own_slot_bytes =                                       \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_SLOT_BYTES);  \
        const volatile unsigned int* own_nslots =                                           \
            reinterpret_cast<const volatile unsigned int*>(own_b + OWL_AR_OFF_N_SLOTS);     \
        __shared__ int s_fail;                                                              \
        __shared__ unsigned int s_prod;                                                     \
        __shared__ unsigned int s_cons;                                                     \
        __shared__ size_t s_off_w;                                                          \
        __shared__ size_t s_off_r;                                                          \
        if (threadIdx.x == 0) {                                                             \
            s_fail = 0;                                                                     \
            if (*own_magic != OWL_AR_MAGIC) {                                               \
                *own_err = 3u;                                                              \
                s_fail = 1;                                                                 \
            } else if (*own_slot_bytes != (unsigned int)slot_bytes) {                       \
                *own_err = 3u;                                                              \
                s_fail = 1;                                                                 \
            } else if (*own_nslots == 0u) {                                                 \
                *own_err = 3u;                                                              \
                s_fail = 1;                                                                 \
            } else if (*own_err != 0u) {                                                    \
                s_fail = 1;                                                                 \
            }                                                                               \
        }                                                                                   \
        __syncthreads();                                                                    \
        if (s_fail) {                                                                       \
            return;                                                                         \
        }                                                                                   \
        /* ① 等 credit:对端已读走 slot = produced % n_slots */                            \
        if (threadIdx.x == 0) {                                                             \
            const unsigned int nslots = *own_nslots;                                        \
            const unsigned int p = *own_prod;                                               \
            const unsigned int c = *own_cons;                                               \
            s_prod = p;                                                                     \
            s_cons = c;                                                                     \
            s_off_w = (size_t)(p % nslots) * (size_t)slot_bytes + (size_t)slab_off;         \
            s_off_r = (size_t)(c % nslots) * (size_t)slot_bytes + (size_t)slab_off;         \
            const unsigned long long t0 = owl_ar_timer_ns();                                \
            unsigned int spins = 0u;                                                        \
            while (owl_ar_ld_acquire_u32(peer_cons) + nslots <= p) {                        \
                if (++spins > OWL_AR_SPIN_TIGHT) {                                          \
                    __nanosleep(200u);                                                      \
                }                                                                           \
                if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {                           \
                    *own_err = 1u;                                                          \
                    s_fail = 1;                                                             \
                    break;                                                                  \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
        __syncthreads();                                                                    \
        if (s_fail) {                                                                       \
            return;                                                                         \
        }                                                                                   \
        /* ② 写本端 staging 的 s_off_w(只本端写;对端经只读别名读) */                    \
        owl_ar_copy_bytes(                                                                  \
            w_b + OWL_AR_HDR_BYTES + s_off_w,                                               \
            reinterpret_cast<const unsigned char*>(payload) + slab_off,                     \
            bytes);                                                                         \
        __syncthreads();                                                                    \
        /* ③ release 发布 produced(本地游标;对端 acquire 读) */                         \
        if (threadIdx.x == 0) {                                                             \
            owl_ar_st_release_u32(own_prod, s_prod + 1u);                                 \
        }                                                                                   \
        __syncthreads();                                                                    \
        /* ④ 等对端 produced > own.consumed(不读未写) */                                 \
        if (threadIdx.x == 0) {                                                             \
            const unsigned long long t0 = owl_ar_timer_ns();                                \
            unsigned int spins = 0u;                                                        \
            while (owl_ar_ld_acquire_u32(peer_prod) <= s_cons) {                            \
                if (++spins > OWL_AR_SPIN_TIGHT) {                                          \
                    __nanosleep(200u);                                                      \
                }                                                                           \
                if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {                           \
                    *own_err = 2u;                                                          \
                    s_fail = 1;                                                             \
                    break;                                                                  \
                }                                                                           \
            }                                                                               \
        }                                                                                   \
        __syncthreads();                                                                    \
        if (s_fail) {                                                                       \
            return;                                                                         \
        }                                                                                   \
        /* ⑤ 读对端 staging 的 s_off_r,加到 payload */                                    \
        {                                                                                   \
            T* dst = reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(payload) + slab_off); \
            const T* src = reinterpret_cast<const T*>(r_b + OWL_AR_HDR_BYTES + s_off_r);    \
            const size_t n = bytes / sizeof(T);                                             \
            const unsigned long long da = reinterpret_cast<unsigned long long>(dst);        \
            const unsigned long long sa = reinterpret_cast<unsigned long long>(src);        \
            if (((da | sa) & 15ull) == 0ull) {                                              \
                owl_ar_add_vec<T, uint4>(dst, dst, src, n);                                 \
            } else {                                                                        \
                owl_ar_add_vec<T, T>(dst, dst, src, n);                                     \
            }                                                                               \
        }                                                                                   \
        __syncthreads();                                                                    \
        /* ⑥ release 发布 consumed(对端下一轮 credit 用) */                              \
        if (threadIdx.x == 0) {                                                             \
            owl_ar_st_release_u32(own_cons, s_cons + 1u);                                 \
        }                                                                                   \
        __syncthreads();                                                                    \
    }

OWL_AR_BARRIER_KERNEL(owl_comm_ar_barrier_f32, float, (float* __restrict__ payload, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data_w, const unsigned char* __restrict__ data_r, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_BARRIER_KERNEL(owl_comm_ar_barrier_f16, __half, (__half* __restrict__ payload, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data_w, const unsigned char* __restrict__ data_r, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_BARRIER_KERNEL(owl_comm_ar_barrier_bf16, __nv_bfloat16, (__nv_bfloat16* __restrict__ payload, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, unsigned char* __restrict__ data_w, const unsigned char* __restrict__ data_r, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
