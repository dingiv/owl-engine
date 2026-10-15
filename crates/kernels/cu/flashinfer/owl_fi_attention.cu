// ============================================================================
// owl FlashInfer prefill adapter —— **上游 FlashInfer 0.6.18 版 · plan + f16 臂**
// (2026-10-14 换源:guoqingbao fork @0f06c230(2026-08-07)→ 上游
//  nightly-v0.6.18-20260819-317-gfdc56343(2026-09-17),`repos/flashinfer`;
//  2026-10-15 **拆 TU**:fp8 臂搬 `owl_fi_attention_fp8.cu`,共享模板住
//  `owl_fi_attention_common.h` —— 单 TU 双实例化会让 ptxas 病态卡住)
//
// 为什么换(单测实测,`crates/kernels/cu/flashinfer/bench/` +
// `docs/arch/prefill-attention-换核-候选单测.md`):同形状(T=512×ctx,hd256,
// GQA24/4,page32,kNHD,fp8 池)上游 **1.41× 快于** fork(0.568 vs 0.798 ms @ctx2048;
// 全 ctx 1.35–1.42×;f16 池 1.8×),数值 max|gpu−host| ≤3.4e-4 同 ulp 级。根因:
// fork 的 plan/split-kv 坏(**且 fork 是被 pin 的个人分支**),上游 plan 用
// **q 瓦片伪批**(本形状 `padded_batch_size=48` ⇒ 192 CTA)把单请求铺满 SM。
//
// **ABI 逐字未变**(`owl_fi_prefill_plan` / `owl_fi_prefill_run` /
// `owl_fi_prefill_run_fp8kv`;15×int64 的 plan blob 契约、槽序、环境旋钮全保留)
// ⇒ Rust 侧(family/server/探针)**零改动**。
//
// 布局契约(kNHD,与 owl 主池逐字对齐;上游 `paged_kv_t` 的 NHD 偏移公式
// `page·(hkv·page·hd) + h·hd + tok·(hkv·hd) + d` 与我们 `knhd_offset` 同式):
//   - k/v:`[nb, page, hkv, hd]` **f16 或 e4m3**
//   - q/out:`[T, hq, hd]` f16;页表/indptr/last_len:device int32
//   - 因果 = chunked prefill(行 i 看 `[0, ctx−T+i]`)= `MaskMode::kCausal`
//     + `window_left=-1`;q 行由 `q_indptr=[0,T]`(varlen)给出
//
// 与旧 fork 版的 API 差异(逐条对齐 `csrc/batch_prefill.cu` /
// `csrc/batch_prefill_paged.cuh` 的参考实现):
//   1. plan 多一个 `kv_dtype_bytes`(只影响 `FA2DetermineCtaTileQ` 的选瓦;
//      生产池 = fp8 ⇒ 缺省 1,`OWL_FI_KV_BYTES=2` 可覆写成 f16 口径);
//   2. plan 前先问 `PrefillPlanWorkspaceSize` 校验调用方给的 workspace 够不够
//      (旧 fork 靠 64MB/32MB 硬写蒙;现改为**显式校验 + 报出所需尺寸**);
//   3. params 由"大构造函数"改为**默认构造 + 逐字段赋值**(上游结构体无该 ctor);
//   4. 发射模板新增 `SAME_KV_STRIDES` 与独立的 `HEAD_DIM_VO`
//      (我们 K/V 同布局 ⇒ true);
//   5. fp8 臂**不再需要编译宏**(上游核内有 `fast_dequant_f8f16x4`);
//   6. `params.sm_scale` 直接是 f32(旧版线上要 f32 位型走私,本文件内已还原)。
// ============================================================================
#include "owl_fi_attention_common.h"

// ---------------------------------------------------------------------------
// plan:host 侧调度规划(每形状一次;Rust 侧一项缓存)
// 入参/出参 ABI 与旧版逐字一致;新增校验 = `PrefillPlanWorkspaceSize`。
// ---------------------------------------------------------------------------
extern "C" int owl_fi_prefill_plan(
    void* float_ws, size_t float_ws_size,
    void* int_ws, size_t int_ws_size,
    void* page_locked, size_t page_locked_size,
    int64_t* out_plan15,
    int32_t* out_cta_tile_q,
    int32_t* out_split_kv,
    const int32_t* qo_indptr_h,      // [B+1] host 端数组(plan 在 host 读)
    const int32_t* kv_indptr_h,      // [B+1] host(token 级)
    int32_t total_num_rows,
    int32_t batch_size,
    int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size,
    cudaStream_t stream) {
  try {
    auto* qo_h = const_cast<int32_t*>(qo_indptr_h);
    auto* kv_h = const_cast<int32_t*>(kv_indptr_h);
    const uint32_t kv_bytes = owl_fi_kv_bytes();
    const bool disable_split_kv = !owl_fi_split_kv_enabled();

    // ① 先问所需 workspace(上游要求尺寸精确;不够就**报出来**,不静默)
    size_t need_float = 0, need_int = 0;
    cudaError_t werr = PrefillPlanWorkspaceSize<OwlId>(
        need_float, need_int, qo_h, kv_h, (uint32_t)total_num_rows, (uint32_t)batch_size,
        (uint32_t)num_qo_heads, (uint32_t)num_kv_heads, (uint32_t)head_dim, (uint32_t)head_dim,
        (uint32_t)page_size, /*enable_cuda_graph=*/false, /*sizeof_dtype_o=*/sizeof(half),
        /*window_left=*/-1, /*fixed_split_size=*/0, disable_split_kv, /*num_colocated_ctas=*/0,
        /*uniform_q_len=*/(int64_t)0, stream, kv_bytes);
    if (werr != cudaSuccess) return (int)werr;
    if (need_float > float_ws_size || need_int > int_ws_size || need_int > page_locked_size) {
      fprintf(stderr,
              "[owl-fi][plan] workspace 不足:需 float=%zuB int=%zuB(page_locked 同尺寸);"
              "现有 float=%zuB int=%zuB page_locked=%zuB(T=%d ctx=%d)\n",
              need_float, need_int, float_ws_size, int_ws_size, page_locked_size, total_num_rows,
              kv_h[batch_size]);
      return -202;
    }

    // ② plan
    PrefillPlanInfo plan_info;
    cudaError_t err = PrefillPlan<OwlId>(
        float_ws, float_ws_size, int_ws, page_locked, int_ws_size, plan_info, qo_h, kv_h,
        (uint32_t)total_num_rows, (uint32_t)batch_size, (uint32_t)num_qo_heads,
        (uint32_t)num_kv_heads, (uint32_t)head_dim, (uint32_t)head_dim, (uint32_t)page_size,
        /*enable_cuda_graph=*/false, /*sizeof_dtype_o=*/sizeof(half),
        /*window_left=*/-1,
        /*fixed_split_size=*/0,
        /*disable_split_kv=*/disable_split_kv,
        /*num_colocated_ctas=*/0, /*uniform_q_len=*/(int64_t)0, stream, kv_bytes);
    if (err != cudaSuccess) return (int)err;
    if (owl_fi_debug()) {
      fprintf(stderr,
              "[owl-fi][plan] padded_batch=%lld total_rows=%lld split=%d cta_tile_q=%lld "
              "kv_bytes=%u disable_split=%d need(float=%zuB int=%zuB)\n",
              (long long)plan_info.padded_batch_size, (long long)plan_info.total_num_rows,
              (int)plan_info.split_kv, (long long)plan_info.cta_tile_q, kv_bytes,
              (int)disable_split_kv, need_float, need_int);
    }
    auto v = plan_info.ToVector();
    if (v.size() != 15) return -100;
    for (int i = 0; i < 15; ++i) out_plan15[i] = v[i];
    *out_cta_tile_q = (int32_t)plan_info.cta_tile_q;
    *out_split_kv = plan_info.split_kv ? 1 : 0;
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "[owl-fi][plan] %s\n", e.what());
    return -101;
  }
}

// ---------------------------------------------------------------------------
// run:每 attention 层一次(f16 KV 臂)
// ---------------------------------------------------------------------------
extern "C" int owl_fi_prefill_run(
    const void* q_ptr, const void* k_data, const void* v_data, void* out_ptr,
    int32_t* q_cu_seqlens, int32_t* indices, int32_t* indptr, int32_t* last_len,
    const int64_t* plan15, void* int_ws, size_t int_ws_size, void* float_ws,
    size_t float_ws_size, int32_t batch_size, int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size, int32_t total_num_rows, float sm_scale,
    cudaStream_t stream) {
  try {
    return owl_fi_run_impl<half>(q_ptr, k_data, v_data, out_ptr, q_cu_seqlens, indices, indptr,
                                 last_len, plan15, int_ws, int_ws_size, float_ws, float_ws_size,
                                 batch_size, num_qo_heads, num_kv_heads, head_dim, page_size,
                                 total_num_rows, sm_scale, stream, "run");
  } catch (const std::exception& e) {
    fprintf(stderr, "[owl-fi][run] %s\n", e.what());
    return -102;
  }
}
