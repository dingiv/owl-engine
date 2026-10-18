// owl 语义算子族:RMSNorm(f32/f16/bf16)(2026-10-13 C 轮自 cu/owl/ops_pair.cu 按**算子族**拆出
// —— 一算子一 `.cu`,与 ops/ln.rs 的 `owl_rmsnorm_*` 声明对望)。
//
// 宏展开模式(nvrtc 铁律:禁 host launcher/禁模板具名实例 —— 模板同单元
// 双实例会撞 PTX 入口名,server 按 load_function(name) 找核)。
// 实例行**自带类型化形参**(宏第三参 PARAMS)—— C2 签名互证按实例行解析
// (宏定义不携带类型,机器对账以实例为准)。
//
// 规则:读 T → to_float → float 计算 → from_float 写 T(存储换档,计算
// 精度不换档 —— float 累计)。命名统一前缀 owl_;**输出块固定末参**
// (槽序契约 4);标量 size_t/i32 与 Arg::U64/I32 严格对位。
//
// ============================================================================
// 2026-10-19 §58 刀③(**decode 小核延迟治理**;读数出处 TP2-实录卷 §58.5/§58.7)
//
// 动机:decode 每步 64 个 rmsnorm + 64 个 fused_add_rmsnorm,旧版单核实测
// **9.35µs**(5120 宽、256 线程;生产 nsys 10.4–11.5µs)⇒ 每步 ~1.55ms(7.3%)。
// 病根是**每线程 20 次 2 字节标量依赖装载**(纯延迟受限:块 64/128/256 三档
// 实测 30.0/16.2/9.35µs ⇒ 与"每线程装载次数"成正比)。
//
// 三处结构改动(**归约序因此改变 ⇒ 数值会动几个 ULP**,对拍口径 = 相对误差
// ≤1e-2(f16 域),判据台 `crates/kernels/tests/ln_elems_bench.rs`):
//   ① **16B 向量装载**(uint4;VEC = 16/sizeof(T)= 8(16 位)/ 4(32 位));
//   ② **归约 = warp shuffle-xor(5 步)+ 每 warp 一个 smem 槽**(旧版 8 级
//      `__syncthreads` 二叉树);序明写于此:线程内按 c 升序、chunk 内 e 升序;
//   ③ **动态 smem**(`extern __shared__`)⇒ 块大小不再被静态数组写死
//      (旧版 `__shared__ float smem[256]` 把块上限锁死 256)。
//
// **调用方契约**(driver 按此算几何,两处一份):动态 smem ≥ ceil(blockDim/32)·4 字节;
//   blockDim 必须是 **32 的倍数且 ≥ 32**(shuffle 全掩码前提)。
// **对齐前提**:向量支只在 `n % VEC == 0` 时走;行基址 = row·n ⇒ n%VEC==0
//   即保证每行 16B 对齐(不满足则整体退回标量支 —— 两支逐元素同式)。
// ============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ---- dtype 桥(三 dtype;float 为恒等)----
template <typename T> __device__ __forceinline__ float owl_to_float(T x);
template <> __device__ __forceinline__ float owl_to_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float owl_to_float<__half>(__half x) { return __half2float(x); }
template <> __device__ __forceinline__ float owl_to_float<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <typename T> __device__ __forceinline__ float owl_from_float(float x);
template <> __device__ __forceinline__ float owl_from_float<float>(float x) { return x; }
template <> __device__ __forceinline__ float owl_from_float<__half>(float x) { return __float2half(x); }
template <> __device__ __forceinline__ float owl_from_float<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// ---- 块内和(线程内已累加完):warp shuffle-xor → 每 warp 一槽 → 首 warp 汇总 ----
// ⚠️ 与 `cu/ln/fused_add_rmsnorm.cu` 的同名函数**逐字同式**(两个翻译单元各一份
//   —— 不跨文件 include:构建期增量指纹只覆盖 `.cu` 自身,共享头会静默漏编)。
//   改动必须两处同步。
template <typename T>
__device__ __forceinline__ float owl_block_sum(float acc, float* smem, int tid, int stride) {
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if ((tid & 31) == 0) smem[tid >> 5] = acc;
    __syncthreads();
    if ((tid >> 5) == 0) {
        const int nw = (stride + 31) >> 5;
        float a = ((tid & 31) < nw) ? smem[tid & 31] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            a += __shfl_xor_sync(0xffffffffu, a, off);
        }
        if ((tid & 31) == 0) smem[0] = a;
    }
    __syncthreads();
    return smem[0];
}

// ---- rmsnorm:x / rms(x) × (alpha + w_off)(块内归约,float 累计)----
#define OWL_RMSNORM_KERNEL(NAME, T, PARAMS)                                            \
extern "C" __global__ void NAME PARAMS {                                               \
    extern __shared__ float smem[];                                                    \
    const int row = blockIdx.x;                                                        \
    const T *xin = x + (size_t)row * n;                                                \
    T *y = out + (size_t)row * n;                                                      \
    const int tid = threadIdx.x;                                                       \
    const int stride = blockDim.x;                                                     \
    const int VEC = (int)(16 / sizeof(T));                                             \
    const bool vec_ok = ((n % VEC) == 0);                                              \
    float local = 0.0f;                                                                \
    if (vec_ok) {                                                                      \
        for (int c = tid * VEC; c < n; c += stride * VEC) {                            \
            const uint4 raw = *reinterpret_cast<const uint4*>(xin + c);                \
            const T *h = reinterpret_cast<const T*>(&raw);                             \
            _Pragma("unroll")                                                          \
            for (int e = 0; e < VEC; ++e) {                                            \
                const float v = owl_to_float(h[e]);                                    \
                local += v * v;                                                        \
            }                                                                          \
        }                                                                              \
    } else {                                                                           \
        for (int c = tid; c < n; c += stride) {                                        \
            const float v = owl_to_float(xin[c]);                                      \
            local += v * v;                                                            \
        }                                                                              \
    }                                                                                  \
    const float ss = owl_block_sum<T>(local, smem, tid, stride);                       \
    const float inv = rsqrtf(ss / (float)n + eps);                                     \
    if (vec_ok) {                                                                      \
        for (int c = tid * VEC; c < n; c += stride * VEC) {                            \
            const uint4 xr = *reinterpret_cast<const uint4*>(xin + c);                 \
            const uint4 ar = *reinterpret_cast<const uint4*>(alpha + c);               \
            const T *xh = reinterpret_cast<const T*>(&xr);                             \
            const T *ah = reinterpret_cast<const T*>(&ar);                             \
            uint4 ov;                                                                  \
            T *oh = reinterpret_cast<T*>(&ov);                                         \
            _Pragma("unroll")                                                          \
            for (int e = 0; e < VEC; ++e) {                                            \
                const float a = w_off ? (owl_to_float(ah[e]) + 1.0f) : owl_to_float(ah[e]); \
                oh[e] = owl_from_float<T>(owl_to_float(xh[e]) * inv * a);              \
            }                                                                          \
            *reinterpret_cast<uint4*>(y + c) = ov;                                     \
        }                                                                              \
    } else {                                                                           \
        for (int c = tid; c < n; c += stride) {                                        \
            const float a = w_off ? (owl_to_float(alpha[c]) + 1.0f) : owl_to_float(alpha[c]); \
            y[c] = owl_from_float<T>(owl_to_float(xin[c]) * inv * a);                  \
        }                                                                              \
    }                                                                                  \
}


OWL_RMSNORM_KERNEL(owl_rmsnorm_f32, float, (const float* x, const float* alpha, float* out, const int n, const float eps, const int w_off))
OWL_RMSNORM_KERNEL(owl_rmsnorm_f16, __half, (const __half* x, const __half* alpha, __half* out, const int n, const float eps, const int w_off))
OWL_RMSNORM_KERNEL(owl_rmsnorm_bf16, __nv_bfloat16, (const __nv_bfloat16* x, const __nv_bfloat16* alpha, __nv_bfloat16* out, const int n, const float eps, const int w_off))
