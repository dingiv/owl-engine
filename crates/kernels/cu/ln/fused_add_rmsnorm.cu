// owl 融合核族:residual 原地加 + RMSNorm 融合(2026-10-01 C1 立项;2026-10-13 C 轮自 cu/owl/fused.cu
// 按**算子族**拆出 —— 一算子一 `.cu`,与 ops/fused.rs / ln.rs 的声明对望)。
//
// Ampere-first 准则(driver 立项同步确立):
//   ① 最小发射数优先于最小流量 —— 一次 DRAM 往返,中间值全寄存器;
//   ② 128-bit 向量化访问(half2/uint4);
//   ③ grid 铺满 SM(按元素/按 (t,head) 开,不整行串行);
//   ④ float 中间精度(与被替核同式,口径可对拍)。
//
// 本文件核族(全部单输出,SSA 契约友好):
//   owl_fused_add_rmsnorm_f16 / _bf16 —— residual 原地 += mixed + rmsnorm·w
//                          (vLLM port;residual 为 InOut 槽)
//
// 输出块固定末参(槽序契约 4)。
//
// ============================================================================
// 2026-10-19 §58 刀③(**decode 小核延迟治理**;读数出处 TP2-实录卷 §58.5/§58.7)
//
// 旧版单核实测 **9.35µs**(5120 宽、256 线程;生产 nsys 10.4–11.5µs)—— 病根是
// **每线程 20 次 2 字节标量依赖装载**。三处结构改动(与 `cu/ln/rmsnorm.cu` 同款):
//   ① 16B 向量装载(VEC = 16/sizeof(T));
//   ② 归约 = warp shuffle-xor + 每 warp 一个 smem 槽(旧版 8 级 `__syncthreads` 树);
//   ③ **动态 smem** ⇒ 块大小不再被静态数组写死(旧版 `__shared__ float smem[256]`)。
// **调用方契约**:动态 smem ≥ ceil(blockDim/32)·4;blockDim = 32 的倍数且 ≥ 32。
// **对齐前提**:向量支只在 `n % VEC == 0` 时走(行基址 = row·n ⇒ 每行 16B 对齐)。
// **代数不变**:归约用的是**未舍入**的 `v = m + r`(与 vLLM 被替核逐式一致),
//   原地点写的是舍入后的 f16 —— 归约序变 ⇒ 数值动几个 ULP(对拍 ≤1e-2,f16 域)。
// ⚠️ `owl_block_sum` 与 `cu/ln/rmsnorm.cu` 的同名函数**逐字同式**(两个翻译单元
//   各一份 —— 不跨文件 include:构建期增量指纹只覆盖 `.cu` 自身,共享头会静默漏编)。
//   改动必须两处同步。
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

/// 块内和(线程内已累加完):warp shuffle-xor → 每 warp 一槽 → 首 warp 汇总
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

/// f16/bf16 两实例逐一展开(逐式同律;不再经共用体函数 —— 两 dtype 的 `__half`/
/// `__nv_bfloat16` 转换函数不同名,共用体会引入一层分派而收益为零)

// ----------------------------------------------------------------------------
// owl_fused_add_rmsnorm_f16(vLLM layernorm_kernels.cu fused_add_rms_norm port;
// 双原地语义 —— residual 块原地 += mixed(已知副作用,GDN conv_upd 同款
// 单流保序律),out = rmsnorm(residual)·w^{w_off} 单输出)。
// 替 decoder 层 [残差 add + post_ln rmsnorm] 两发;grid (rows,1,1),
// block (512,1,1),动态 smem = ceil(block/32)·4B(行归约;n = hidden 5120)。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_fused_add_rmsnorm_f16(
    const __half* __restrict__ mixed,       // mixer 输出(只读)
    __half* __restrict__ residual,          // in/out:+= mixed(原地副作用)
    const __half* __restrict__ w,           // [n]
    float eps,
    size_t n,
    int w_off,
    __half* __restrict__ out)               // [rows, n] = rmsnorm(residual)·w
{
    extern __shared__ float smem[];
    const size_t base = (size_t)blockIdx.x * n;
    const __half* m = mixed + base;
    __half* r = residual + base;
    __half* o = out + base;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const bool vec_ok = ((n % 8) == 0);

    float local = 0.0f;
    if (vec_ok) {
        for (size_t c = (size_t)tid * 8; c < n; c += (size_t)stride * 8) {
            const uint4 mr = *reinterpret_cast<const uint4*>(m + c);
            const uint4 rr = *reinterpret_cast<const uint4*>(r + c);
            const __half* mh = reinterpret_cast<const __half*>(&mr);
            const __half* rh = reinterpret_cast<const __half*>(&rr);
            uint4 ov;
            __half* oh = reinterpret_cast<__half*>(&ov);
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float v = __half2float(mh[e]) + __half2float(rh[e]);
                oh[e] = __float2half(v);            // 原地写(回收前根已收割,单流序)
                local += v * v;                     // **未舍入**的 v(与被替核逐式一致)
            }
            *reinterpret_cast<uint4*>(r + c) = ov;
        }
    } else {
        for (size_t i = tid; i < n; i += stride) {
            const float v = __half2float(m[i]) + __half2float(r[i]);
            r[i] = __float2half(v);
            local += v * v;
        }
    }
    const float ss = owl_block_sum<__half>(local, smem, tid, stride);
    const float inv = rsqrtf(ss / (float)n + eps);
    if (vec_ok) {
        for (size_t c = (size_t)tid * 8; c < n; c += (size_t)stride * 8) {
            const uint4 rr = *reinterpret_cast<const uint4*>(r + c);
            const uint4 wr = *reinterpret_cast<const uint4*>(w + c);
            const __half* rh = reinterpret_cast<const __half*>(&rr);
            const __half* wh = reinterpret_cast<const __half*>(&wr);
            uint4 ov;
            __half* oh = reinterpret_cast<__half*>(&ov);
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float v = __half2float(rh[e]);
                const float wf = __half2float(wh[e]);
                oh[e] = __float2half(v * inv * (w_off ? (wf + 1.0f) : wf));
            }
            *reinterpret_cast<uint4*>(o + c) = ov;
        }
    } else {
        for (size_t i = tid; i < n; i += stride) {
            const float v = __half2float(r[i]);
            const float wf = __half2float(w[i]);
            o[i] = __float2half(v * inv * (w_off ? (wf + 1.0f) : wf));
        }
    }
}

// ----------------------------------------------------------------------------
// owl_fused_add_rmsnorm_bf16:全 bf16(含 gamma;检查点原生 BF16)。
// residual 原地 += mixed + rmsnorm·w,逐式同 f16 版。
// ----------------------------------------------------------------------------
extern "C" __global__ void owl_fused_add_rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ mixed,       // mixer 输出(只读)
    __nv_bfloat16* __restrict__ residual,          // in/out:+= mixed(原地副作用)
    const __nv_bfloat16* __restrict__ w,           // [n]
    float eps,
    size_t n,
    int w_off,
    __nv_bfloat16* __restrict__ out)               // [rows, n] = rmsnorm(residual)·w
{
    extern __shared__ float smem[];
    const size_t base = (size_t)blockIdx.x * n;
    const __nv_bfloat16* m = mixed + base;
    __nv_bfloat16* r = residual + base;
    __nv_bfloat16* o = out + base;
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    const bool vec_ok = ((n % 8) == 0);

    float local = 0.0f;
    if (vec_ok) {
        for (size_t c = (size_t)tid * 8; c < n; c += (size_t)stride * 8) {
            const uint4 mr = *reinterpret_cast<const uint4*>(m + c);
            const uint4 rr = *reinterpret_cast<const uint4*>(r + c);
            const __nv_bfloat16* mh = reinterpret_cast<const __nv_bfloat16*>(&mr);
            const __nv_bfloat16* rh = reinterpret_cast<const __nv_bfloat16*>(&rr);
            uint4 ov;
            __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ov);
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float v = __bfloat162float(mh[e]) + __bfloat162float(rh[e]);
                oh[e] = __float2bfloat16(v);
                local += v * v;
            }
            *reinterpret_cast<uint4*>(r + c) = ov;
        }
    } else {
        for (size_t i = tid; i < n; i += stride) {
            const float v = __bfloat162float(m[i]) + __bfloat162float(r[i]);
            r[i] = __float2bfloat16(v);
            local += v * v;
        }
    }
    const float ss = owl_block_sum<__nv_bfloat16>(local, smem, tid, stride);
    const float inv = rsqrtf(ss / (float)n + eps);
    if (vec_ok) {
        for (size_t c = (size_t)tid * 8; c < n; c += (size_t)stride * 8) {
            const uint4 rr = *reinterpret_cast<const uint4*>(r + c);
            const uint4 wr = *reinterpret_cast<const uint4*>(w + c);
            const __nv_bfloat16* rh = reinterpret_cast<const __nv_bfloat16*>(&rr);
            const __nv_bfloat16* wh = reinterpret_cast<const __nv_bfloat16*>(&wr);
            uint4 ov;
            __nv_bfloat16* oh = reinterpret_cast<__nv_bfloat16*>(&ov);
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float v = __bfloat162float(rh[e]);
                const float wf = __bfloat162float(wh[e]);
                oh[e] = __float2bfloat16(v * inv * (w_off ? (wf + 1.0f) : wf));
            }
            *reinterpret_cast<uint4*>(o + c) = ov;
        }
    } else {
        for (size_t i = tid; i < n; i += stride) {
            const float v = __bfloat162float(r[i]);
            const float wf = __bfloat162float(w[i]);
            o[i] = __float2bfloat16(v * inv * (w_off ? (wf + 1.0f) : wf));
        }
    }
}
