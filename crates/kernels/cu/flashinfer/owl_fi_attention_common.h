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
/// split-kv 开关(**缺省开**;`FI_SPLIT_KV=0` 关回历史口径)。
///
/// ⚠️ 2026-10-15 V1 定谳前的状态:本开关**形同虚设** —— 开与不开都错的,
/// 因为三处前提缺了/错了(全部实测,见 `crates/kernels/tests/fi_prefill_probe.rs`
/// 的 verify/decode 形状档):
///   ① plan 的 `kv_indptr` 我们传的是 **token 级**,而 plan 这套数组整体系
///      **page 级**(`kv_len_arr` → `num_chunks_kv` → 二分搜索的
///      `min_kv_chunk_size = 128/page`)⇒ split 档 plan 算 34 chunk、
///      核按 2912 token/chunk 算 2 ⇒ partial 行缩放与 `merge_indptr` 打架
///      ⇒ 除第 0 行全零(T=8/ctx3072:零值 43008/49152)。
///      分水岭 = `server/flashinfer.rs` 传页级 kv_indptr。
///   ② 尾部 CTA 掩码没接(plan 只在 split 档生成 `block_valid_mask`)。
///   ③ `partition_kv` 由 dispatcher 按 `tmp_v` 是否为空自行覆盖(我们那行
///      硬编码 `false` 其实被覆盖了,不影响;留着与 plan 同值更自明)。
/// 修好后实测(3090 Ti,fi_prefill_probe,host oracle 门 1e-2,全档 ≤5e-4):
///   | 形状 | 本开关=1 | =0 | XQA 同形状 |
///   |---|---|---|---|
///   | T=1/ctx=3072 | 0.030 ms | 0.126 | — |
///   | T=8/ctx=32768 | **0.249** | 3.604 | 1.541 |
///   | T=1/ctx=32768 | **0.118** | 1.264 | 0.910 |
///   | T=512/ctx=16384 | 5.393 | 5.394 | —(大 T 本就并行) |
/// ⇒ 小 T 形状收益 6–14×,大 T 中性;生产"尾块"面(T 很小、ctx 很长)是
/// 直接的净赚,**故缺省转正**。
static bool owl_fi_split_kv_enabled() {
  const char* e = getenv("FI_SPLIT_KV");
  if (e != nullptr && e[0] != '\0' && e[0] == '0') return false;
  return true;
}
/// KV 元素字节数(1 = e4m3 主池【生产】;2 = f16 池)。只喂 plan 的选瓦启发式。
static uint32_t owl_fi_kv_bytes() {
  const char* e = getenv("OWL_FI_KV_BYTES");
  if (e != nullptr && e[0] == '2') return 2;
  return 1;
}
/// **图档 plan**(P-5 前置;env `FI_CUDA_GRAPH`)。开启后 plan 按"图模式"定瓦:
/// `cta_tile_q` 取最坏情形、`padded_batch_size = max(2·SM/hkv, tiles)` ⇒
/// **尾部 CTA 恒存在**(小桶时远多于真 tile)⇒ 必须配 mask(见 `block_valid_mask`)。
static bool owl_fi_cuda_graph() {
  const char* e = getenv("FI_CUDA_GRAPH");
  return e != nullptr && e[0] != '\0' && e[0] != '0';
}
/// mask 开关(诊断/取证;`FI_NO_MASK=1` 关掉 mask —— 用来量"白跑的尾部 CTA")
static bool owl_fi_no_mask() {
  const char* e = getenv("FI_NO_MASK");
  return e != nullptr && e[0] != '\0' && e[0] != '0';
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
    const void* block_valid_mask, cudaStream_t stream, const char* tag) {
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
  // **P-5:尾部 CTA 掩码**(2026-10-15)。图档 plan 的 `padded_batch_size` 可为
  // `max(2·SM/hkv, tiles)` ⇒ 尾部 CTA 远超真 tile 数;不掩码时它们读 plan 表
  // 的**残留表项**(行为取决于上一次 plan 写了什么 —— 静默,不是不变量)。
  // 掩码内容(`i < tiles`)由 Rust 侧按 plan15 算好并在**窗外**上传一次
  // (桶级静态;`q_cu=[0,t_pad]` 常量 ⇒ mask 不随 chunk 变)。
  // `FI_NO_MASK=1` = 取证档(量白跑成本;真行仍不应被写坏 —— 见探针)。
  params.block_valid_mask = owl_fi_no_mask() ? nullptr : (bool*)block_valid_mask;
  params.max_total_num_rows = (uint32_t)plan_info.total_num_rows;
  params.total_num_rows = nullptr;  // 非 cuda-graph 档
  params.padded_batch_size = (uint32_t)plan_info.padded_batch_size;
  // ⚠️ **partition_kv 必须 = plan 的 split_kv**(2026-10-15 V1 定谳):
  // 上游 `include/` 从不设本字段(调用方责任),而核内三处都拿它分叉:
  //   max_chunk_size / chunk_start / chunk_end(是否按 kv_tile_idx 切 KV)
  //   与 `o_ptr_base`(partial 写 tmp_v 还是写 o)。
  // 本行曾是 `false` 硬编码 + "plan 已把 split 编进调度表" 的注释 —— **错的**:
  // 调度表只决定"发射多少个 kv_tile 项",核要不要真分块由本字段定。
  // 实测(FI_SPLIT_KV=1 探针):T=8/ctx=3072 数值 max|FI−host| = 1.39(门 1e-2)
  // ⇒ split 路**从未真正可用**;改回 = plan 值后转绿(见探针复跑)。
  params.partition_kv = plan_info.split_kv;
  if (plan_info.split_kv) {
    params.merge_indptr = GetPtrFromBaseOffset<OwlId>(int_ws, plan_info.merge_indptr_offset);
    // ⚠️ **尾部 CTA 掩码必须接上**(2026-10-15 V1 定谳,与 partition_kv 同一批):
    // split 档 `padded_batch_size = max(2·SM/hkv, new_batch_size)` 会**大于**真实
    // 工作项数 ⇒ 尾部 CTA 会读 `page_locked` 里的**陈旧** request/qo_tile/kv_tile
    // ⇒ 算出垃圾 partial 并被 merge 累加(实测 T=8/ctx=3072 max|FI−host| = 1.39)。
    // 上游只在 split 档分配这张表(`scheduler.cuh` 内 `block_valid_mask_offset` +
    // `mask[i] = i < new_batch_size`),核侧守卫 `if (block_valid_mask && !mask[bx]) return;`
    // 一直都在 —— 缺的只是**把指针传进去**。非 split 档该区不存在 ⇒ 保持 nullptr。
    params.block_valid_mask =
        GetPtrFromBaseOffset<bool>(int_ws, plan_info.block_valid_mask_offset);
  }

  OwlOut* tmp_v = nullptr;
  float* tmp_s = nullptr;
  if (plan_info.split_kv) {
    tmp_v = GetPtrFromBaseOffset<OwlOut>(float_ws, plan_info.v_offset);
    tmp_s = GetPtrFromBaseOffset<float>(float_ws, plan_info.s_offset);
  }

  if (owl_fi_debug()) {
    fprintf(stderr,
            "[owl-fi][%s] hd=%d page=%d T=%d hq=%d hkv=%d split=%d cta=%lld padded=%lld "
            "mask=%s graph=%d\n",
            tag, head_dim, page_size, total_num_rows, num_qo_heads, num_kv_heads,
            (int)plan_info.split_kv, (long long)plan_info.cta_tile_q,
            (long long)plan_info.padded_batch_size,
            params.block_valid_mask ? "on" : "off", (int)owl_fi_cuda_graph());
  }
  return (int)owl_fi_dispatch(params, tmp_v, tmp_s, (uint32_t)plan_info.cta_tile_q, head_dim,
                              /*same_kv_strides=*/true, stream);
}
