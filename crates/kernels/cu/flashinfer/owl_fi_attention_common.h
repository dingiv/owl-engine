// ============================================================================
// owl FlashInfer prefill adapter —— **共享头**(2026-10-15 拆 TU 用)
//
// 为什么拆:换源到上游 0.6.18 后,`owl_fi_attention.cu` 单个 TU 里同时实例化
// **f16 臂**与 **fp8 臂**两套 `BatchPrefillWithPagedKVCacheDispatched<...>`
// (每套还要 × SAME_KV_STRIDES{true,false} × CTA_TILE_Q{64,128,256} ×
// HEAD_DIM{64,128,256}),ptxas 单个 TU 上病态(实测卡住分钟级)。
// 拆法 = **按 KV dtype 臂拆 TU**:模板体放本头(两 TU 各自实例化自己那份),
// f16 臂住 `owl_fi_attention.cu`(plan 也住那里),fp8 臂住
// `owl_fi_attention_fp8.cu`。**ABI 逐字未变**(三个 extern "C" 符号名/签名不动)
// ⇒ Rust 侧零改动;`build.rs` 编两个 TU 进同一个 `.a`。
//
// 本头只放**模板与常量**,不放 `extern "C"` 定义(符号唯一性)。
// ============================================================================
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

using namespace flashinfer;

using OwlQ = half;
using OwlOut = half;
using OwlId = int32_t;
/// 变体:无 custom mask / 无滑窗 / 无 softcap / 无 alibi(= owl 生产形状)
using OwlFiAttention = DefaultAttention<false, false, false, false>;

constexpr PosEncodingMode OWL_POS_MODE = PosEncodingMode::kNone;
constexpr bool OWL_FP16_QK_REDUCTION = false;
constexpr MaskMode OWL_MASK_MODE = MaskMode::kCausal;

// ---------------------------------------------------------------------------
// 探针旋钮(旧 fork 版同名保留)
// ---------------------------------------------------------------------------
/// split-kv 开关(缺省关 ⇒ 口径与历史一致;上游 plan 会再按 SM 数自行决定)
static bool owl_fi_split_kv_enabled() {
  const char* e = getenv("FI_SPLIT_KV");
  return e != nullptr && e[0] != '\0' && e[0] != '0';
}
/// KV 元素字节数(1 = e4m3 主池【生产】;2 = f16 池)。只喂 plan 的选瓦启发式。
static uint32_t owl_fi_kv_bytes() {
  const char* e = getenv("OWL_FI_KV_BYTES");
  if (e != nullptr && e[0] == '2') return 2;
  return 1;
}
static bool owl_fi_debug() {
  const char* e = getenv("FI_DEBUG");
  return e != nullptr && e[0] != '\0' && e[0] != '0';
}

// ---------------------------------------------------------------------------
// 发射单发(模板桥;SAME_KV_STRIDES + HEAD_DIM_QK/VO)
// `static` ⇒ 内部链接:两个 TU 各自实例化自己那份(拆 TU 的收益就在这里)
// ---------------------------------------------------------------------------
template <bool SAME_KV_STRIDES, uint32_t CTA_TILE_Q, uint32_t HEAD_DIM, typename Params>
static cudaError_t owl_fi_launch(Params& params, OwlOut* tmp_v, float* tmp_s,
                                 cudaStream_t stream) {
  return BatchPrefillWithPagedKVCacheDispatched<SAME_KV_STRIDES, CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
                                               OWL_POS_MODE, OWL_FP16_QK_REDUCTION, OWL_MASK_MODE,
                                               OwlFiAttention, Params>(
      params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
}

/// 按 plan 的 cta_tile_q + 运行期 hd 分派(与旧版同构)
template <typename Params>
static cudaError_t owl_fi_dispatch(Params& params, OwlOut* tmp_v, float* tmp_s,
                                   uint32_t cta_tile_q, int32_t head_dim, bool same_kv_strides,
                                   cudaStream_t stream) {
  // 注:两个 DISPATCH 宏的 default 分支都会 FLASHINFER_ERROR(抛异常)⇒
  // 走到这里必然赋值;初始值只为编译器。（不设"未赋值即成功"的兜底 —— 那会把
  // 真错吞成静默 no-op）
  cudaError_t st = cudaSuccess;
  if (same_kv_strides) {
    DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        st = owl_fi_launch<true, CTA_TILE_Q, HEAD_DIM>(params, tmp_v, tmp_s, stream);
      });
    });
  } else {
    DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        st = owl_fi_launch<false, CTA_TILE_Q, HEAD_DIM>(params, tmp_v, tmp_s, stream);
      });
    });
  }
  return st;
}

// ---------------------------------------------------------------------------
// run 公共体:plan blob → params → 发射(两条臂只差 DTypeKV)
// ---------------------------------------------------------------------------
template <typename DTypeKV>
static int owl_fi_run_impl(
    const void* q_ptr, const void* k_data, const void* v_data, void* out_ptr,
    int32_t* q_cu_seqlens, int32_t* indices, int32_t* indptr, int32_t* last_len,
    const int64_t* plan15, void* int_ws, size_t int_ws_size, void* float_ws,
    size_t float_ws_size, int32_t batch_size, int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size, int32_t total_num_rows, float sm_scale,
    cudaStream_t stream, const char* tag) {
  using ParamsType = BatchPrefillPagedParams<OwlQ, DTypeKV, OwlOut, OwlId>;

  // plan blob(15×int64;字段序 = PrefillPlanInfo,两端同源)
  PrefillPlanInfo plan_info;
  std::vector<int64_t> vec(plan15, plan15 + 15);
  plan_info.FromVector(vec);
  if (plan_info.total_num_rows_offset >= (int64_t)int_ws_size ||
      plan_info.v_offset >= (int64_t)float_ws_size ||
      plan_info.s_offset >= (int64_t)float_ws_size) {
    return -201;  // workspace 越界(尺寸不足)
  }

  ParamsType params;  // 上游结构体 = 默认构造 + 逐字段赋值
  params.q = (OwlQ*)q_ptr;
  params.paged_kv = paged_kv_t<DTypeKV, OwlId>(
      (uint32_t)num_kv_heads, (uint32_t)page_size, (uint32_t)head_dim, (uint32_t)batch_size,
      QKVLayout::kNHD, (DTypeKV*)k_data, (DTypeKV*)v_data, indices, indptr, last_len,
      /*rope_pos_offset=*/nullptr);
  params.maybe_custom_mask = nullptr;
  params.q_indptr = q_cu_seqlens;  // varlen:q 行端点 [0, T]
  params.maybe_mask_indptr = nullptr;
  params.maybe_q_rope_offset = nullptr;
  params.o = (OwlOut*)out_ptr;
  params.lse = nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.group_size = uint_fastdiv((uint32_t)(num_qo_heads / num_kv_heads));
  params.num_qo_heads = (uint32_t)num_qo_heads;
  params.q_stride_n = (OwlId)(num_qo_heads * head_dim);
  params.q_stride_h = (OwlId)head_dim;
  params.window_left = -1;
  params.logits_soft_cap = 0.f;
  params.sm_scale = sm_scale;
  params.rope_rcp_scale = 0.f;
  params.rope_rcp_theta = 0.f;
  // plan 产出的调度表(int ws)
  params.request_indices =
      GetPtrFromBaseOffset<OwlId>(int_ws, plan_info.request_indices_offset);
  params.qo_tile_indices = GetPtrFromBaseOffset<OwlId>(int_ws, plan_info.qo_tile_indices_offset);
  params.kv_tile_indices = GetPtrFromBaseOffset<OwlId>(int_ws, plan_info.kv_tile_indices_offset);
  params.o_indptr = GetPtrFromBaseOffset<OwlId>(int_ws, plan_info.o_indptr_offset);
  params.kv_chunk_size_ptr =
      GetPtrFromBaseOffset<OwlId>(int_ws, plan_info.kv_chunk_size_ptr_offset);
  params.merge_indptr = nullptr;
  params.block_valid_mask = nullptr;
  params.max_total_num_rows = (uint32_t)plan_info.total_num_rows;
  params.total_num_rows = nullptr;  // 非 cuda-graph 档
  params.padded_batch_size = (uint32_t)plan_info.padded_batch_size;
  params.partition_kv = false;  // 与上游参考实现同款(plan 已把 split 编进调度表)
  if (plan_info.split_kv) {
    params.merge_indptr = GetPtrFromBaseOffset<OwlId>(int_ws, plan_info.merge_indptr_offset);
  }

  OwlOut* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<OwlOut>(float_ws, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_ws, plan_info.s_offset);
  }

  if (owl_fi_debug()) {
    fprintf(stderr,
            "[owl-fi][%s] hd=%d page=%d T=%d hq=%d hkv=%d split=%d cta=%lld padded=%lld\n", tag,
            head_dim, page_size, total_num_rows, num_qo_heads, num_kv_heads,
            (int)plan_info.split_kv, (long long)plan_info.cta_tile_q,
            (long long)plan_info.padded_batch_size);
  }
  return (int)owl_fi_dispatch(params, tmp_v, tmp_s, (uint32_t)plan_info.cta_tile_q, head_dim,
                              /*same_kv_strides=*/true, stream);
}
