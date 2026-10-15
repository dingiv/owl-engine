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

// ---------------------------------------------------------------------------
// run:每 attention 层一次(**e4m3 KV 臂 = 生产**)
// ---------------------------------------------------------------------------
extern "C" int owl_fi_prefill_run_fp8kv(
    const void* q_ptr, const void* k_data, const void* v_data, void* out_ptr,
    int32_t* q_cu_seqlens, int32_t* indices, int32_t* indptr, int32_t* last_len,
    const int64_t* plan15, void* int_ws, size_t int_ws_size, void* float_ws,
    size_t float_ws_size, int32_t batch_size, int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size, int32_t total_num_rows, float sm_scale,
    cudaStream_t stream) {
  try {
    return owl_fi_run_impl<__nv_fp8_e4m3>(q_ptr, k_data, v_data, out_ptr, q_cu_seqlens, indices,
                                          indptr, last_len, plan15, int_ws, int_ws_size, float_ws,
                                          float_ws_size, batch_size, num_qo_heads, num_kv_heads,
                                          head_dim, page_size, total_num_rows, sm_scale, stream,
                                          "run_fp8kv");
  } catch (const std::exception& e) {
    fprintf(stderr, "[owl-fi][run_fp8kv] %s\n", e.what());
    return -103;
  }
}
