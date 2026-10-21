// ============================================================================
// owl 视觉塔注意力适配器:**FlashInfer ragged prefill(非因果)**
//   —— 复用 owl 已 vendored 的 FlashInfer FA2 家族核,不新写核(用户裁决 2026-10-15:
//      "视觉塔别自己实现算子,拿 vLLM 那一路的算子用")。
//
// 为什么是它:vLLM 的 Qwen2/3-VL 视觉塔注意力走 `flash_attn_varlen_func(..., causal=False)`
//   (FA2 varlen 非因果);FlashInfer 的 `BatchPrefillWithRaggedKVCacheDispatched` 就是
//   同一路算法(非分页、dense K/V、varlen、MaskMode 模板参数)⇒ 一行模板参数即
//   `MaskMode::kNone`。**本文件只写 host 侧胶水**(计划 + 参数装配 + 发射),
//   核体零自研。
//
// 契约(与 Native `vis.attn_full` **同槽位**;driver 按 dtype/图数选臂):
//   q/k/v [rows, heads, head_dim] f16 连续;cu_seqlens [n_seq+1] f32(段界);
//   out [rows, heads, head_dim] f16;标量 rows/heads/head_dim/n_seq/scale。
//   **单图限制**(n_seq == 1):ragged plan 的 host 端点 = [0, rows](无需回读设备表);
//   多图走 native 臂(调用方/driver 已分流)。
// ============================================================================
// 算子侧唯一日志出口(编译门 OWL_LOG_ENABLED + 运行期 level/tag)
#include "owl_log.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/utils.cuh>

using namespace flashinfer;

using RaggedQ = half;
using RaggedOut = half;
using RaggedId = int32_t;
/// 变体与主路径同款(无 custom mask / 无滑窗 / 无 softcap / 无 alibi)
using RaggedAttn = DefaultAttention<false, false, false, false>;

// ---------------------------------------------------------------------------
// 发射(模板:CTA_TILE_Q × HEAD_DIM;**非因果** = MaskMode::kNone)
// ---------------------------------------------------------------------------
template <uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, typename Params>
static cudaError_t ragged_launch(Params& params, cudaStream_t stream) {
  return BatchPrefillWithRaggedKVCacheDispatched<CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
                                                PosEncodingMode::kNone,
                                                /*USE_FP16_QK_REDUCTION=*/false,
                                                MaskMode::kNone, RaggedAttn, Params>(
      params, /*tmp_v=*/nullptr, /*tmp_s=*/nullptr, /*enable_pdl=*/false, stream);
}

// ---------------------------------------------------------------------------
// plan(host;每形状一次 —— 与 paged 臂同型:窗口外算,捕获期命中缓存)
//   qo_indptr_h = [0, rows];kv_indptr_h = [0, rows](单图 vanilla 自注意力)
// ---------------------------------------------------------------------------
extern "C" int owl_fi_attn_ragged_plan(
    void* float_ws, size_t float_ws_size,
    void* int_ws, size_t int_ws_size,
    void* page_locked, size_t page_locked_size,
    int64_t* out_plan15,
    int32_t* out_cta_tile_q,
    int32_t* out_cu_dev,      // 设备端 int32 段界([0, rows];plan 期写一次,回放期只读)
    int32_t rows, int32_t num_heads, int32_t head_dim,
    cudaStream_t stream) {
  try {
    if (rows <= 0 || num_heads <= 0 || head_dim <= 0) return -203;
    if (out_cu_dev == nullptr) return -206;
    // ⚠️ 核读的是 **IdType(int32)** 段界;owl 的 `cu_seqlens` 张量是 f32(索引 f32 过线律)
    // ⇒ 这里在 plan 期写一份规范 int32(单图 ⇒ 恒 [0, rows]),**回放期零转换零回读**。
    {
      const int32_t cu_host[2] = {0, rows};
      cudaError_t cerr = cudaMemcpyAsync(out_cu_dev, cu_host, sizeof(cu_host),
                                         cudaMemcpyHostToDevice, stream);
      if (cerr != cudaSuccess) return (int)cerr;
    }
    const int32_t qo_h[2] = {0, rows};
    const int32_t kv_h[2] = {0, rows};
    // kv_dtype_bytes = 2(f16 ragged K/V;只影响选瓦启发式)
    size_t need_float = 0, need_int = 0;
    cudaError_t werr = PrefillPlanWorkspaceSize<RaggedId>(
        need_float, need_int, const_cast<int32_t*>(qo_h), const_cast<int32_t*>(kv_h),
        (uint32_t)rows, /*batch_size=*/1u, (uint32_t)num_heads, (uint32_t)num_heads,
        (uint32_t)head_dim, (uint32_t)head_dim, /*page_size=*/1u,
        /*enable_cuda_graph=*/false, /*sizeof_dtype_o=*/sizeof(half),
        /*window_left=*/-1, /*fixed_split_size=*/0, /*disable_split_kv=*/true,
        /*num_colocated_ctas=*/0, /*uniform_q_len=*/(int64_t)0, stream, /*kv_dtype_bytes=*/2u);
    if (werr != cudaSuccess) return (int)werr;
    if (need_float > float_ws_size || need_int > int_ws_size || need_int > page_locked_size) {
      OWL_LOG_ERROR("owl-fi",
                    "[ragged plan] workspace 不足:需 float=%zuB int=%zuB;现有 float=%zuB "
                    "int=%zuB page_locked=%zuB(rows=%d)",
                    need_float, need_int, float_ws_size, int_ws_size, page_locked_size, rows);
      return -202;
    }
    PrefillPlanInfo plan_info;
    cudaError_t err = PrefillPlan<RaggedId>(
        float_ws, float_ws_size, int_ws, page_locked, int_ws_size, plan_info,
        const_cast<int32_t*>(qo_h), const_cast<int32_t*>(kv_h), (uint32_t)rows,
        /*batch_size=*/1u, (uint32_t)num_heads, (uint32_t)num_heads, (uint32_t)head_dim,
        (uint32_t)head_dim, /*page_size=*/1u, /*enable_cuda_graph=*/false,
        /*sizeof_dtype_o=*/sizeof(half), /*window_left=*/-1, /*fixed_split_size=*/0,
        /*disable_split_kv=*/true, /*num_colocated_ctas=*/0, /*uniform_q_len=*/(int64_t)0,
        stream, /*kv_dtype_bytes=*/2u);
    if (err != cudaSuccess) return (int)err;
    auto v = plan_info.ToVector();
    if (v.size() != 15) return -100;
    for (int i = 0; i < 15; ++i) out_plan15[i] = v[i];
    *out_cta_tile_q = (int32_t)plan_info.cta_tile_q;
    return 0;
  } catch (const std::exception& e) {
    OWL_LOG_ERROR("owl-fi", "[ragged plan] 异常: %s", e.what());
    return -101;
  }
}

// ---------------------------------------------------------------------------
// run(每层一次;f16)
// ---------------------------------------------------------------------------
extern "C" int owl_fi_attn_ragged_run_f16(
    const void* q_ptr, const void* k_ptr, const void* v_ptr, const void* cu_ptr,
    void* out_ptr,
    const int64_t* plan15,
    void* int_ws, size_t int_ws_size,
    void* float_ws, size_t float_ws_size,
    int32_t rows, int32_t num_heads, int32_t head_dim, int32_t n_seq, float scale,
    cudaStream_t stream) {
  try {
    if (n_seq != 1) return -204;  // 单图契约(多图由 driver 分流到 native 臂)
    // ⚠️ 本适配器只实例化 HEAD_DIM=64(视觉塔 4B/9B 之外的 hd 走 native 臂)——
    // 不匹配时**响亮拒绝**,绝不静默按 64 维算(2026-10-15 实测踩过:假红/错值)
    if (head_dim != 64) return -207;
    if (rows <= 0) return cudaSuccess;

    PrefillPlanInfo plan_info;
    std::vector<int64_t> vec(plan15, plan15 + 15);
    plan_info.FromVector(vec);
    if (plan_info.total_num_rows_offset >= (int64_t)int_ws_size ||
        plan_info.v_offset >= (int64_t)float_ws_size ||
        plan_info.s_offset >= (int64_t)float_ws_size) {
      return -201;
    }

    using ParamsType = BatchPrefillRaggedParams<RaggedQ, RaggedQ, RaggedOut, RaggedId>;
    ParamsType params;
    params.q = (RaggedQ*)q_ptr;
    params.k = (RaggedQ*)k_ptr;
    params.v = (RaggedQ*)v_ptr;
    params.maybe_custom_mask = nullptr;
    params.q_indptr = (RaggedId*)cu_ptr;       // 设备端段界([0, rows];回放期读)
    params.kv_indptr = (RaggedId*)cu_ptr;
    params.maybe_mask_indptr = nullptr;
    params.maybe_q_rope_offset = nullptr;
    params.maybe_k_rope_offset = nullptr;
    params.o = (RaggedOut*)out_ptr;
    params.lse = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = uint_fastdiv(1u);
    params.num_qo_heads = (uint32_t)num_heads;
    params.num_kv_heads = (uint32_t)num_heads;
    params.q_stride_n = (RaggedId)(num_heads * head_dim);
    params.q_stride_h = (RaggedId)head_dim;
    params.k_stride_n = (RaggedId)(num_heads * head_dim);
    params.k_stride_h = (RaggedId)head_dim;
    params.v_stride_n = (RaggedId)(num_heads * head_dim);
    params.v_stride_h = (RaggedId)head_dim;
    params.k_sf_stride_page = 0;
    params.k_sf_stride_n = 0;
    params.k_sf_stride_h = 0;
    params.v_sf_stride_page = 0;
    params.v_sf_stride_n = 0;
    params.v_sf_stride_h = 0;
    params.window_left = -1;
    params.logits_soft_cap = 0.f;
    params.sm_scale = scale;
    params.rope_rcp_scale = 0.f;
    params.rope_rcp_theta = 0.f;
    params.request_indices =
        GetPtrFromBaseOffset<RaggedId>(int_ws, plan_info.request_indices_offset);
    params.qo_tile_indices =
        GetPtrFromBaseOffset<RaggedId>(int_ws, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices =
        GetPtrFromBaseOffset<RaggedId>(int_ws, plan_info.kv_tile_indices_offset);
    params.merge_indptr = nullptr;
    params.o_indptr = GetPtrFromBaseOffset<RaggedId>(int_ws, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr =
        GetPtrFromBaseOffset<RaggedId>(int_ws, plan_info.kv_chunk_size_ptr_offset);
    params.block_valid_mask = nullptr;
    params.max_total_num_rows = (uint32_t)plan_info.total_num_rows;
    params.total_num_rows = nullptr;
    params.padded_batch_size = (uint32_t)plan_info.padded_batch_size;
    params.partition_kv = false;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    switch ((uint32_t)plan_info.cta_tile_q) {
      case 64u:
        return (int)ragged_launch<64u, 64u>(params, stream);
      case 128u:
        return (int)ragged_launch<128u, 64u>(params, stream);
      case 256u:
        return (int)ragged_launch<256u, 64u>(params, stream);
      default:
        OWL_LOG_ERROR("owl-fi", "[ragged run] 未实例化的 cta_tile_q=%lld",
                      (long long)plan_info.cta_tile_q);
        return -205;
    }
  } catch (const std::exception& e) {
    OWL_LOG_ERROR("owl-fi", "[ragged run] 异常: %s", e.what());
    return -102;
  }
}
