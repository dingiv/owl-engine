// ============================================================================
// owl FlashInfer prefill adapter —— **上游 FlashInfer 0.6.18 版 · e4m3(KV)臂**
//
// 2026-10-15 从 `owl_fi_attention.cu` 拆出(拆 TU = 按 KV dtype 臂切):
// 该 TU 只实例化 `BatchPrefillPagedParams<half, __nv_fp8_e4m3, half, int32_t>` 一套
// FA2 模板;f16 臂(与 plan)住 `owl_fi_attention.cu`,共享模板住
// `owl_fi_attention_common.h`。拆前单 TU 双实例化会让 ptxas 病态卡住。
//
// 上游核内自带 fp8 反量化(`fast_dequant_f8f16x4`)⇒ **不需要**旧 fork 的
// `-DFLASHINFER_ENABLE_FP8_E4M3` 编译宏(见 `crates/kernels/build.rs`)。
// ABI(`owl_fi_prefill_run_fp8kv` 的名字/签名)与旧版逐字未变 ⇒ Rust 侧零改动。
// ============================================================================
#include "owl_fi_attention_common.h"
#include <algorithm>

// ---------------------------------------------------------------------------
// run:每 attention 层一次(**e4m3 KV 臂 = 生产**)
// ---------------------------------------------------------------------------
extern "C" int owl_fi_prefill_run_fp8kv(
    const void* q_ptr, const void* k_data, const void* v_data, void* out_ptr,
    int32_t* q_cu_seqlens, int32_t* indices, int32_t* indptr, int32_t* last_len,
    const int64_t* plan15, void* int_ws, size_t int_ws_size, void* float_ws,
    size_t float_ws_size, int32_t batch_size, int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size, int32_t total_num_rows, float sm_scale,
    const void* block_valid_mask, cudaStream_t stream) {
  try {
    return owl_fi_run_impl<__nv_fp8_e4m3>(q_ptr, k_data, v_data, out_ptr, q_cu_seqlens, indices,
                                          indptr, last_len, plan15, int_ws, int_ws_size, float_ws,
                                          float_ws_size, batch_size, num_qo_heads, num_kv_heads,
                                          head_dim, page_size, total_num_rows, sm_scale,
                                          block_valid_mask, stream, "run_fp8kv");
  } catch (const std::exception& e) {
    OWL_LOG_ERROR("owl-fi", "[run_fp8kv] 异常: %s", e.what());
    return -103;
  }
}

// ============================================================================
// FP8 → F16 shadow(2026-10-26;prefill-only)
//
// 生产 FP8 池不换;prefill 每层 attention 前把当前 ctx 的 K/V 物理页
// **一次**反量化进 dense f16 shadow,再走 FI f16 臂 —— 避免 FA2 fp8
// 核在每个 q tile 上重复 dequant/repack。
//
// 目标 layout = **dense 逻辑页** `[nb, page, hkv, hd]`;FI 侧配一张
// identity 页表(`[0,1,..,nb-1]`),因此 shadow 大小 = nb 页(不是 pool
// 全量),100k ≈ 205MB/rank,262k ≈ 537MB/rank。
// 源 layout = 生产 kNHD `[pool_nb, page, hkv, hd]`,按 `indices[p]` 取物理页。
// ============================================================================
__global__ void owl_kv_fp8_to_f16_kernel(
    const uint8_t* __restrict__ k_src, const uint8_t* __restrict__ v_src,
    __half* __restrict__ k_dst, __half* __restrict__ v_dst,
    const int32_t* __restrict__ indices, int64_t nb_pages, int32_t hkv, int32_t hd,
    int32_t page) {
  const int32_t vec_d = hd / 4;  // 一次 4 个 e4m3 → 4 个 half
  const int64_t vec_per_page = (int64_t)page * hkv * vec_d;
  const int64_t total = nb_pages * vec_per_page;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += stride) {
    const int64_t p = i / vec_per_page;
    const int64_t r = i - p * vec_per_page;
    const int64_t per_row = (int64_t)hkv * vec_d;
    const int32_t t = (int32_t)(r / per_row);
    const int32_t h = (int32_t)((r / vec_d) % hkv);
    const int32_t d4 = (int32_t)(r % vec_d);
    const int64_t src = (int64_t)indices[p] * page * hkv * hd + (int64_t)t * hkv * hd +
                        (int64_t)h * hd + (int64_t)d4 * 4;
    const int64_t dst = p * page * hkv * hd + (int64_t)t * hkv * hd + (int64_t)h * hd +
                        (int64_t)d4 * 4;
    uint32_t qk = *reinterpret_cast<const uint32_t*>(k_src + src);
    uint32_t qv = *reinterpret_cast<const uint32_t*>(v_src + src);
    uint2 ok, ov;
    fast_dequant_f8f16x4<__nv_fp8_e4m3, __half>(&qk, &ok);
    fast_dequant_f8f16x4<__nv_fp8_e4m3, __half>(&qv, &ov);
    *reinterpret_cast<uint2*>(k_dst + dst) = ok;
    *reinterpret_cast<uint2*>(v_dst + dst) = ov;
  }
}

extern "C" int owl_fi_dequant_fp8_to_f16(
    const void* k_src, const void* v_src, void* k_dst, void* v_dst,
    const int32_t* indices, int32_t nb_pages, int32_t hkv, int32_t hd, int32_t page,
    cudaStream_t stream) {
  if (nb_pages <= 0 || hkv <= 0 || hd <= 0 || page <= 0 || (hd % 4) != 0) return -301;
  const int64_t total = (int64_t)nb_pages * page * hkv * (hd / 4);
  constexpr int kThreads = 256;
  const int blocks = (int)std::min<int64_t>((total + kThreads - 1) / kThreads, 65535);
  owl_kv_fp8_to_f16_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(k_src), reinterpret_cast<const uint8_t*>(v_src),
      reinterpret_cast<__half*>(k_dst), reinterpret_cast<__half*>(v_dst), indices, nb_pages, hkv,
      hd, page);
  return (int)cudaGetLastError();
}
