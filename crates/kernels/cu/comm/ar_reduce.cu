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
// 2. **acquire 序**:看到新 produced 之后 `__threadfence_system()` 再读数据
//    (防读被重排到 flag 读之前 —— 与生产者的 release 配对,I4);
// 3. **超时**(5 s)置 err=2 并放弃(D4);
// 4. **失败路径不提前 return**(shared 标志 + 全员一致退出,防 `__syncthreads` 死锁);
// 5. **累加用 float 中间**(与 elems/add 同律:存储换档,计算精度不换档)。

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
                const unsigned long long cons = *own_cons;                             \
                const unsigned long long t0 = owl_ar_timer_ns();                       \
                unsigned int spins = 0u;                                               \
                while (*peer_prod <= cons) { /* I2:不读未写 */                        \
                    if (++spins > OWL_AR_SPIN_TIGHT) {                                 \
                        __nanosleep(200u);                                             \
                    }                                                                  \
                    if (owl_ar_timer_ns() - t0 > OWL_AR_TIMEOUT_NS) {                  \
                        *own_err = 2u;                                                 \
                        s_fail = 1;                                                    \
                        break;                                                         \
                    }                                                                  \
                }                                                                      \
                if (!s_fail) {                                                         \
                    __threadfence_system(); /* acquire:后续读不被提前(I4) */       \
                }                                                                      \
                s_cons = cons;                                                         \
            }                                                                          \
        }                                                                              \
        __syncthreads();                                                               \
        if (s_fail) {                                                                  \
            return;                                                                    \
        }                                                                              \
        const unsigned int slot =                                                      \
            (unsigned int)(s_cons % (unsigned long long)(*own_nslots));                \
        const T* src = reinterpret_cast<const T*>(                                    \
            peer_b + OWL_AR_HDR_BYTES + (size_t)slot * slot_bytes);                    \
        T* dst = reinterpret_cast<T*>(                                                \
            reinterpret_cast<unsigned char*>(out) + slab_off);                         \
        const size_t n = bytes / sizeof(T);                                            \
        for (size_t i = threadIdx.x; i < n; i += blockDim.x) {                         \
            dst[i] = owl_ar_from_float<T>(owl_ar_to_float<T>(dst[i]) +                 \
                                          owl_ar_to_float<T>(src[i]));                 \
        }                                                                              \
        __threadfence_system();                                                        \
        __syncthreads();                                                               \
        if (threadIdx.x == 0) {                                                        \
            __threadfence_system();                                                    \
            *own_cons = s_cons + 1ull;                                                 \
        }                                                                              \
    }

OWL_AR_REDUCE_KERNEL(owl_comm_ar_reduce_f32, float, (float* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_REDUCE_KERNEL(owl_comm_ar_reduce_f16, __half, (__half* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
OWL_AR_REDUCE_KERNEL(owl_comm_ar_reduce_bf16, __nv_bfloat16, (__nv_bfloat16* __restrict__ out, unsigned char* __restrict__ own, const unsigned char* __restrict__ peer, const size_t bytes, const size_t slot_bytes, const size_t slab_off))
