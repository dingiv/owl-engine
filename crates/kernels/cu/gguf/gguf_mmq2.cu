// ============================================================================
// owl GGUF 族:**head 版 int8-MMA MMQ** 适配 TU(W-2f;设计 §二十二)
//
// 内核本体 = **llama.cpp fork 逐字 vendoring**(`vendor2/`;两处已登记补丁 P1/P2,
// 见 `vendor2/README.md`)。本文件只做三件事:
//
//   ① **激活量化**(f16 → `block_q8_1_mmq`,144 B/128 值):上游在
//      `quantize.cu::quantize_mmq_q8_1`(f32 入);本仓激活是 f16、且上游那支是
//      `__global__`(fatbin 里只能当符号发射)⇒ 按其**布局定义**在核内逐式重写:
//
//          y[(k/128) * n_tokens + token] = { d4[4] | ds4[4] } , qs[128]
//          qs 内是 **k 序**(无块内转置);"转置"只在**外层** [k_block][token]。
//
//      判据 = 与 host 参考**逐字节相同**(门 `gpu_gguf_mmq2_quant_matches_host`)。
//      DS 布局按型(Q8_0/Q6_K = D4;Q4_K/Q5_K = DS4,需 `sum` 抵消 min 偏移)。
//
//   ② **内核入口**(8 个):`mul_mat_q<type,J,fallback>` 经 P1 降为 `__device__`,
//      由本文件的 `extern "C" __global__` wrapper 发射(launch_bounds 搬到这里,
//      与上游 `{nthreads=256, occupancy=1}` 同档)。J ∈ {64,128}:
//      `m ≥ 128` ⇒ J=128,`64 ≤ m < 128` ⇒ J=64。**fallback 恒 false**
//      (分派门要求 `n % 128 == 0`,见 `family::gguf::mmq2_ok`)。
//
//   ③ **StreamK 退化为 tiling**:上游 stream-k 支在"tiles 效率 ≥ 90%"时把 grid 设成
//      **一维 `ntiles_dst`** 且 `fixup_needed = false`(每 block 恰一个 tile)⇒
//      本臂直接取该形态:`grid = (nty·ntx, 1, 1)`、`tmp_fixup = nullptr`,
//      **零 fixup 核**(效率不足时只是 SM 利用差一点,不影响正确性)。
//
// # 槽序(与 `ops/gguf.rs` 的 MMQ2 声明逐参对位)
//
//   weights[U8 块] , xq[U8 q8_1_mmq] , out[F16] , n(权重行) , m(token 数) , k
//
// ⚠️ **`stride_row_x` 的单位是「块」不是字节**(2026-10-19 定谳):装载器里是
// `bxi = (block_q4_K *) x + kbx0 + i*stride`,而 kernel 把 `it*I*stride_row_x`
// 当**块偏移**传进去(`mmvq.cu` 侧同一口径:`off = (…+i*stride_row_x+…)*bs`)。
// ⇒ 本 wrapper 内部算 `k / qk`(qk 由型定:Q8_0=32,K-quant=256)。
//
// # 为什么值这一刀(§21.1/§22)
//
// llama.cpp 同机同形状实测 **103 TFLOPS 等效**(int8 mma.sync),本仓 cuBLAS 臂
// (f16×f32 累加)66 TFLOPS,且本臂**零权重搬运**(不需要 dequant→f16)
// ⇒ 单刀 −33% prefill 时间。
// ============================================================================
#include "vendor2/mmq.cu"

// ============================================================================
// ① 激活量化:f16 → block_q8_1_mmq
//   grid = (n_tokens, ceil(k / (4*BLK)), 1)、block = (BLK, 1, 1);每线程 4 值。
//   组 = 32 值(一个 d/ds 槽);`amax` 与 `sum` 在组内 8 线程间 `shfl` 归约
//   (与上游同法)。
// ============================================================================
namespace owl_mmq2 {

constexpr int BLK = 128;  // 每 CTA 线程数(每线程 4 值 ⇒ 512 值/CTA/行)

/// 组内(shfl 8 线程 = 32 值)归约最大绝对值
__device__ __forceinline__ float group_amax(float amax) {
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off, 32));
    }
    return amax;
}

/// 同组求和(DS4 布局要它;D4 不调)
__device__ __forceinline__ float group_sum(float s) {
#pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xFFFFFFFFu, s, off, 32);
    }
    return s;
}

/// `ds4 = true` ⇒ Q4_K/Q5_K(写 ds4[4] = half2(d, sum));
/// `ds4 = false` ⇒ Q8_0/Q6_K(写 d4[4] = d)
/// `bf16 = true` ⇒ 激活是 **BF16**(DFlash2 草稿执行域;2026-10-21)
template <bool ds4, bool bf16>
__device__ __forceinline__ void quant_body(const void * __restrict__ vx, void * __restrict__ vy,
                                           int n_tokens, int ne0) {
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) {
        return;
    }
    const int64_t row = blockIdx.x;              // token
    float2 f0, f1;
    if constexpr (bf16) {
        const __nv_bfloat16 * x =
            reinterpret_cast<const __nv_bfloat16 *>(vx) + row * (int64_t) ne0;
        f0 = make_float2(__bfloat162float(x[i0 + 0]), __bfloat162float(x[i0 + 1]));
        f1 = make_float2(__bfloat162float(x[i0 + 2]), __bfloat162float(x[i0 + 3]));
    } else {
        const __half * x = reinterpret_cast<const __half *>(vx) + row * (int64_t) ne0;
        // 4 个 f16 读成 2 个 half2(保持与上游 float4 相同的 lane→元素映射)
        f0 = __half22float2(*reinterpret_cast<const __half2 *>(x + i0 + 0));
        f1 = __half22float2(*reinterpret_cast<const __half2 *>(x + i0 + 2));
    }

    float amax = fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f1.x), fabsf(f1.y)));
    amax = group_amax(amax);
    float sum = 0.0f;
    if (ds4) {
        sum = group_sum(f0.x + f0.y + f1.x + f1.y);
    }

    // amax == 0 ⇒ 全零块:d = 0、q = 0(上游此处会出 inf/NaN;本仓响亮地给 0)
    const float d_inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    const char4 q = make_char4((signed char) roundf(f0.x * d_inv), (signed char) roundf(f0.y * d_inv),
                               (signed char) roundf(f1.x * d_inv), (signed char) roundf(f1.y * d_inv));

    // 落点:[k_block][token]
    const int64_t k_block = i0 / 128;
    const int64_t iqs = i0 % 128;
    block_q8_1_mmq * y = reinterpret_cast<block_q8_1_mmq *>(vy);
    block_q8_1_mmq * yb = y + k_block * (int64_t) n_tokens + row;
    reinterpret_cast<char4 *>(yb->qs)[iqs / 4] = q;
    if (iqs % 32 == 0) {
        if (ds4) {
            yb->ds4[iqs / 32] = make_half2(d, sum);
        } else {
            yb->d4[iqs / 32] = d;
        }
    }
}

} // namespace owl_mmq2

extern "C" __global__ void owl_mmq2_quant_q8_1_ds4(const void * __restrict__ vx,
                                                    void * __restrict__ vy, int n_tokens, int ne0) {
    owl_mmq2::quant_body<true, false>(vx, vy, n_tokens, ne0);
}

extern "C" __global__ void owl_mmq2_quant_q8_1_d4(const void * __restrict__ vx,
                                                  void * __restrict__ vy, int n_tokens, int ne0) {
    owl_mmq2::quant_body<false, false>(vx, vy, n_tokens, ne0);
}

// ---- BF16 激活变体(2026-10-21;DFlash2 草稿执行域)----
//
// 为什么要有:草稿的激活是 BF16(F16 装不下残差流),而 MMQ2 的量化手只吃 F16
// ⇒ 草稿的全部 GEMM 被迫退回复用臂(tile),propose 的 1.07 GB 权重按 290 GB/s
// 走。本变体让 MMQ2 直接吃 BF16(数值上只多一次 bf16→f32,精度不降)。
extern "C" __global__ void owl_mmq2_quant_q8_1_ds4_bf16(
    const void * __restrict__ vx, void * __restrict__ vy, int n_tokens, int ne0) {
    owl_mmq2::quant_body<true, true>(vx, vy, n_tokens, ne0);
}

extern "C" __global__ void owl_mmq2_quant_q8_1_d4_bf16(
    const void * __restrict__ vx, void * __restrict__ vy, int n_tokens, int ne0) {
    owl_mmq2::quant_body<false, true>(vx, vy, n_tokens, ne0);
}

// ============================================================================
// ② 内核 wrapper:8 个(type × J)。`fallback=false` 恒(分派门: n % 128 == 0)
//
// fastdiv 三元组:`vendor2/common.cuh::init_fastdiv_values` 是 **host-only**
// (上游在 host 算好再传进核)⇒ 本适配层给一份**逐式同构的设备版**(mp/L/d 三件,
// 整数运算 ⇒ 与 host 版逐位相同;除数恒 = 已知量)。
// ============================================================================
__device__ __forceinline__ uint3 owl_mmq2_fastdiv_init(uint32_t d) {
    uint32_t ll = 0;
    while (ll < 32 && (uint32_t{ 1 } << ll) < d) {
        ++ll;
    }
    const uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << ll) - d) / d + 1);
    return make_uint3(mp, ll, d);
}

// 256 线程 / occupancy 1 = 上游 ampere 档(nthreads=256, occupancy=1)
#define OWL_MMQ2_WRAP(NAME, TYPE, J)                                                             \
    extern "C" __global__ __launch_bounds__(256, 1) void NAME(                                   \
        const void * __restrict__ vx, const void * __restrict__ vy, __half * __restrict__ out,   \
        int n, int m, int k) {                                                                   \
        constexpr ggml_type ty = TYPE;                                                           \
        constexpr int jj = J;                                                                    \
        const uint3 bp = owl_mmq2_fastdiv_init((uint32_t) (k / ggml_cuda_type_traits<ty>::qk));  \
        const uint3 ntx = owl_mmq2_fastdiv_init((uint32_t) ((m + jj - 1) / jj));                 \
        const uint3 one = make_uint3(1u, 0u, 1u);                                                \
        /* ncols_dst = m(token 列数);stride_col_dst = n(输出行宽);                     */       \
        /* ncols_y   = m(y 的"列"= token;y 布局 [k_block][token]);                     */       \
        mul_mat_q<ty, jj, false>(                                                                \
            (const char *) vx, (const int *) vy, nullptr, nullptr, out, nullptr, nullptr,         \
            bp, n, m, k / ggml_cuda_type_traits<ty>::qk, m, n, one, one, 0, 0, 0, one, one, 0, 0, 0, ntx); \
    }

OWL_MMQ2_WRAP(owl_mmq2_q8_0_j64, GGML_TYPE_Q8_0, 64)
OWL_MMQ2_WRAP(owl_mmq2_q8_0_j128, GGML_TYPE_Q8_0, 128)
OWL_MMQ2_WRAP(owl_mmq2_q4k_j64, GGML_TYPE_Q4_K, 64)
OWL_MMQ2_WRAP(owl_mmq2_q4k_j128, GGML_TYPE_Q4_K, 128)
OWL_MMQ2_WRAP(owl_mmq2_q5k_j64, GGML_TYPE_Q5_K, 64)
OWL_MMQ2_WRAP(owl_mmq2_q5k_j128, GGML_TYPE_Q5_K, 128)
OWL_MMQ2_WRAP(owl_mmq2_q6k_j64, GGML_TYPE_Q6_K, 64)
OWL_MMQ2_WRAP(owl_mmq2_q6k_j128, GGML_TYPE_Q6_K, 128)
// IQ 族(2026-10-20;REQ-GGUF-04):vendor2 的 mmq-config-ampere 已带 IQ4_XS/IQ3_S/
// IQ3_XXS 的 J 档(布局 = Q8_0 ⇒ 激活量化走 D4)。IQ4_NL 是 32 元素块,
// vendor2 目录里没有它的 MMQ 配置 ⇒ 不进本臂(仍走 cuBLAS/MMVQ)。
OWL_MMQ2_WRAP(owl_mmq2_iq4xs_j64, GGML_TYPE_IQ4_XS, 64)
OWL_MMQ2_WRAP(owl_mmq2_iq4xs_j128, GGML_TYPE_IQ4_XS, 128)
OWL_MMQ2_WRAP(owl_mmq2_iq3s_j64, GGML_TYPE_IQ3_S, 64)
OWL_MMQ2_WRAP(owl_mmq2_iq3s_j128, GGML_TYPE_IQ3_S, 128)
OWL_MMQ2_WRAP(owl_mmq2_iq3xxs_j64, GGML_TYPE_IQ3_XXS, 64)
OWL_MMQ2_WRAP(owl_mmq2_iq3xxs_j128, GGML_TYPE_IQ3_XXS, 128)

// ---- J=16(2026-10-21):小 m(verify depth+1 = 2..8、propose 8 行)专用档 ----
// 复用同一 `mul_mat_q<ty, J, false>`(上游 J 表本就含 16)。J=64 在 m=8 时
// 有 8/64 的有用行 ⇒ 4× 的 MMA 浪费;J=16 同样一次读权重(ntx=ceil(8/16)=1),
// 但只算 16 行 ⇒ 实测 MMQ2 m=8 从 81 µs 降到 ~35 µs(29 MB 张量)。

OWL_MMQ2_WRAP(owl_mmq2_q8_0_j16, GGML_TYPE_Q8_0, 16)
OWL_MMQ2_WRAP(owl_mmq2_q4k_j16, GGML_TYPE_Q4_K, 16)
OWL_MMQ2_WRAP(owl_mmq2_q5k_j16, GGML_TYPE_Q5_K, 16)
OWL_MMQ2_WRAP(owl_mmq2_q6k_j16, GGML_TYPE_Q6_K, 16)
OWL_MMQ2_WRAP(owl_mmq2_iq4xs_j16, GGML_TYPE_IQ4_XS, 16)
OWL_MMQ2_WRAP(owl_mmq2_iq3s_j16, GGML_TYPE_IQ3_S, 16)
OWL_MMQ2_WRAP(owl_mmq2_iq3xxs_j16, GGML_TYPE_IQ3_XXS, 16)

// ---- J=16,SRAM 布局 = Q3_K(IQ2_S / IQ2_XS / Q3_K;2026-10-21 第二刀)----------------------
//
// 动机(verify 轮内 nsys 账):UD-IQ4_XS 混型档里 IQ2_S(7 张)/ IQ2_XS(2 张)/
// Q3_K(9 张)/ Q2_K(1 张)此前**没有 MMQ2 实例** ⇒ 全落复用臂(tile):
// IQ2_S 679 µs/张(42 GB/s)、IQ3_S 之外的 Q3_K 506 µs —— 四型合计只占权重
// 8.2%,却吃掉 verify 轮 17% 的时间。它们的 MMQ 配置在 vendor2 里**本来就有**
// (`mmq-config-ampere.cuh` 的 J=8..128 全档),只是没被适配层发射。
//
// ⚠️ 这族的 SRAM 布局是 `Q3_K`(X 面 stride 84,不是 76)
// ⇒ smem 尺寸必须按型取(`family::gguf::mmq2_sram_stride_for`),否则溢出。
// ⚠️ 只挂 J=16:小 m(verify/propose 的 2..16 行)。m > 16 仍回 tile ——
// MMQ 的 ntx = ceil(m/J) 会按 token 窗**重读权重**(prefill 语义上仍是重复读),
// 本刀不碰 prefill 侧。
OWL_MMQ2_WRAP(owl_mmq2_q3k_j16, GGML_TYPE_Q3_K, 16)
OWL_MMQ2_WRAP(owl_mmq2_iq2s_j16, GGML_TYPE_IQ2_S, 16)
OWL_MMQ2_WRAP(owl_mmq2_iq2xs_j16, GGML_TYPE_IQ2_XS, 16)
// ⚠️ Q2_K **不在本刀**:它的激活布局是 `MMQ_Q8_1_DS_LAYOUT_D2S6`
// (`mmq.cuh::mmq_get_q8_1_ds_layout`),而本适配层的量化手只有 D4 / DS4 两族 ⇒
// 硬接会出 inf(实测)。本档 Q2_K 只有 1 张(19 MB),留在 tile 臂。
