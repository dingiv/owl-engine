// ============================================================================
// owl FlashInfer prefill adapter —— **e4m3 Q × e4m3 KV 原生 FP8 MMA 臂**
//
// 2026-11-04 为 40 系（Ada / SM89）立案。旧 `owl_fi_attention_fp8.cu` 走
// `BatchPrefillPagedParams<half, e4m3, half>`：K/V 在核内 `fast_dequant_f8f16x4`
// 反量化成 half，再打 half MMA。本 TU 实例化
// `BatchPrefillPagedParams<e4m3, e4m3, half>`，由 vendored `prefill.cuh` 的
// `USE_FP8_MMA` 分支走 `mma.sync...m16n8k32...e4m3.e4m3.f32`（QK 与 PV 都不
// 反量化）。**只在 arch-sm89 构建里编**（见 `crates/kernels/build.rs` 与
// `server/flashinfer.rs` 的 `cfg(feature = "arch-sm89")` 门）。
//
// Q 侧量化：模型 NormRope 的输出仍是 f16（tap/覆盖率探针还要读），本 TU 提供
// `owl_fi_quant_f16_to_fp8`，由 Rust 侧先写 Q8 scratch，再调
// `owl_fi_prefill_run_fp8mma`。scale 目前 = 1.0（qk-norm 后 Q 的 RMS≈1，
// e4m3 上限 448 ⇒ 不饱和；若将来发现离群值，再把 q_scale 折进 sm_scale，
// 不要静默 clamp）。
//
// 与旧 fp8 臂的 ABI 差异只有入口名：参数表逐字相同（plan blob / workspace /
// scale / mask / stream）。Rust 侧按 arch feature 选入口，语义节点不变。
// ============================================================================
#include "owl_fi_attention_common.h"
#include <algorithm>

// ---------------------------------------------------------------------------
// Q 量化：f16 → e4m3（每线程 4 个元素，写一个 u32）
// ---------------------------------------------------------------------------
__global__ void owl_quant_f16_to_fp8_kernel(const __half* __restrict__ src,
                                            uint32_t* __restrict__ dst, int64_t n4) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    const uint2 h = reinterpret_cast<const uint2*>(src)[i];
    const __half2 h0 = *reinterpret_cast<const __half2*>(&h.x);
    const __half2 h1 = *reinterpret_cast<const __half2*>(&h.y);
    const float2 f0 = __half22float2(h0);
    const float2 f1 = __half22float2(h1);
    const uint32_t lo =
        __nv_cvt_float2_to_fp8x2(f0, __NV_SATFINITE, __NV_E4M3);
    const uint32_t hi =
        __nv_cvt_float2_to_fp8x2(f1, __NV_SATFINITE, __NV_E4M3);
    dst[i] = lo | (hi << 16);
  }
}

extern "C" int owl_fi_quant_f16_to_fp8(const void* src, void* dst, int64_t n,
                                       cudaStream_t stream) {
  if (src == nullptr || dst == nullptr || n <= 0 || (n % 4) != 0) return -302;
  const int64_t n4 = n / 4;
  constexpr int kThreads = 256;
  const int blocks = (int)std::min<int64_t>((n4 + kThreads - 1) / kThreads, 65535);
  owl_quant_f16_to_fp8_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(src), reinterpret_cast<uint32_t*>(dst), n4);
  return (int)cudaGetLastError();
}

// ---------------------------------------------------------------------------
// run：Q/KV 都是 e4m3，输出 half（与旧 fp8 臂同一 15×int64 plan blob）
// ---------------------------------------------------------------------------
extern "C" int owl_fi_prefill_run_fp8mma(
    const void* q_ptr, const void* k_data, const void* v_data, void* out_ptr,
    int32_t* q_cu_seqlens, int32_t* indices, int32_t* indptr, int32_t* last_len,
    const int64_t* plan15, void* int_ws, size_t int_ws_size, void* float_ws,
    size_t float_ws_size, int32_t batch_size, int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size, int32_t total_num_rows, float sm_scale,
    const void* block_valid_mask, cudaStream_t stream) {
  try {
    return owl_fi_run_impl<__nv_fp8_e4m3, __nv_fp8_e4m3>(
        q_ptr, k_data, v_data, out_ptr, q_cu_seqlens, indices, indptr, last_len, plan15, int_ws,
        int_ws_size, float_ws, float_ws_size, batch_size, num_qo_heads, num_kv_heads, head_dim,
        page_size, total_num_rows, sm_scale, block_valid_mask, stream, "run_fp8mma");
  } catch (const std::exception& e) {
    OWL_LOG_ERROR("owl-fi", "[run_fp8mma] 异常: %s", e.what());
    return -104;
  }
}
