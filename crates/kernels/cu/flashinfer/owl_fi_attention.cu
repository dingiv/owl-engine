// ============================================================================
// owl FlashInfer 预编 adapter(2026-10-03;E1.5 工作包落地,2026-09-27 定盘
// 搁置条件「E2 分页底座」已解除)。
//
// ported from: repos/attention.rs src/kernels/src/flashinfer_adapter_prefill.cu
//              (guoqingbao/flashinfer fork @ 0f06c230;非 SM90/FA2 分支)
// 裁剪:仅 half + paged + causal + hd256/hd128;砍 fp8/bf16/SM90/ragged/
//       alibi/sink/softcap(owl prefill 实际不用;要时再加臂)。
// 布局契约:FI paged_kv_t QKVLayout::kNHD(per page = [page_size, Hkv,
//   head_dim],page.cuh get_elem_offset:page·stride+h·hd+entry·(Hkv·hd)+d)
//   —— owl classic K(x-interleave)/ V(dim-major)均不匹配 → K/V 双影子
//   池由 owl_reshape_and_cache_dual_f16(K0-dual)同发射写出。
// 平面:plan(host,每 chunk 一次;server 侧 1 项缓存)+ run(每 attention
//   层一次)。批 = 1(owl 单会话),qo_indptr = [0,T],kv_indptr = [0,ctx]。
// 编译:nvcc -O3 -std=c++17 -arch=sm_86 -I<flashinfer>/include -c 本文件
//   → libowl_flashinfer.a(prebuilt 入库,marlin .a 先例;零构建期依赖)。
// ============================================================================
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <algorithm>
#include <sstream>
#include <stdexcept>

#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/page.cuh>
#include <flashinfer/utils.cuh>
#include <flashinfer/pos_enc.cuh>

using namespace flashinfer;

template <bool use_custom_mask, bool use_sliding_window, bool use_logits_soft_cap, bool use_alibi>
using OwlFiAttention = DefaultAttention<use_custom_mask, use_sliding_window,
                                        use_logits_soft_cap, use_alibi>;

// ---------------------------------------------------------------------------
// plan:host 侧调度规划。qo_indptr_h = [0, T](owl 单会话 chunk);
// kv_indptr_h = [0, ctx_total](token 级;页换算 plan 内部做)。
// 出参 plan15[15] = PrefillPlanInfo::ToVector(吃进 run);out_cta_tile_q /
// out_split_kv 供 server 缓存键与断言。
// ---------------------------------------------------------------------------
extern "C" int owl_fi_prefill_plan(
    void* float_ws, size_t float_ws_size,
    void* int_ws, size_t int_ws_size,
    void* page_locked, size_t page_locked_size,
    int64_t* out_plan15,
    int32_t* out_cta_tile_q,
    int32_t* out_split_kv,
    const int32_t* qo_indptr_h,      // [B+1] host
    const int32_t* kv_indptr_h,      // [B+1] host(token 级)
    int32_t total_num_rows,
    int32_t batch_size,
    int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size,
    cudaStream_t stream) {
  try {
    PrefillPlanInfo plan_info;
    cudaError_t err = PrefillPlan<int32_t>(
        float_ws, float_ws_size,
        int_ws, page_locked, int_ws_size,
        plan_info,
        const_cast<int32_t*>(qo_indptr_h),
        const_cast<int32_t*>(kv_indptr_h),
        (uint32_t)total_num_rows, (uint32_t)batch_size,
        (uint32_t)num_qo_heads, (uint32_t)num_kv_heads,
        (uint32_t)head_dim, (uint32_t)head_dim, (uint32_t)page_size,
        /*enable_cuda_graph=*/false, /*sizeof_dtype_o=*/sizeof(half),
        /*window_left=*/-1, /*fixed_split_size=*/-1,
        /*disable_split_kv=*/true,  // owl prefill qo=chunk(1024)→64 tiles 已
        // 吃满 SM;split-kv 是 decode(qo=1)形态 —— 且本 fork 非 graph 分块
        // merge 在 (row>0) 产出空(取证:q=0 仅 row 0 写入),禁之,挂账
        /*num_colocated_ctas=*/0, /*uniform_q_len=*/(int64_t)0, stream);
    if (err != cudaSuccess) return (int)err;
    if (getenv("FI_DEBUG")) {  // 取证开关(默认关)
        // 宿主调度向量取证(与 PrefillPlan 内部同参; Printf 路径)
        auto split = PrefillSplitQOKVIndptr(
            const_cast<int32_t*>(qo_indptr_h), const_cast<int32_t*>(kv_indptr_h),
            (uint32_t)total_num_rows, (uint32_t)batch_size,
            (uint32_t)num_qo_heads, (uint32_t)num_kv_heads,
            (uint32_t)head_dim, (uint32_t)page_size,
            /*max_batch_size_if_split=*/1024, /*enable_cuda_graph=*/false,
            /*window_left=*/-1, /*fixed_split_size=*/-1,
        /*disable_split_kv=*/true,  // owl prefill qo=chunk(1024)→64 tiles 已
        // 吃满 SM;split-kv 是 decode(qo=1)形态 —— 且本 fork 非 graph 分块
        // merge 在 (row>0) 产出空(取证:q=0 仅 row 0 写入),禁之,挂账
            /*uniform_q_len=*/(int64_t)0);
        fprintf(stderr, "[owl-fi][plan] split_kv=%d padded=%u cta_tile=%u kv_chunk=%u\n",
                (int)std::get<0>(split), std::get<2>(split), std::get<3>(split), std::get<4>(split));
        fprintf(stderr, "[owl-fi][plan] req=%p qo_tile=%p kv_tile=%p o_indptr=%p\n",
                (void*)&std::get<5>(split), (void*)&std::get<6>(split),
                (void*)&std::get<7>(split), (void*)&std::get<9>(split));
        fprintf(stderr, "[owl-fi][plan] plan_info: padded_batch_size=%u total_num_rows=%u split=%d cta_tile_q=%u\n",
                plan_info.padded_batch_size, plan_info.total_num_rows,
                (int)plan_info.split_kv, plan_info.cta_tile_q);
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
// run:每 attention 层一次。paged_kv_t(kHND;kc_fi + vc 共享池)+ FA2
// dispatched(causal;split_kv 时 tmp_v/tmp_s 入 float_ws)。
// ---------------------------------------------------------------------------
extern "C" int owl_fi_prefill_run(
    const void* q_ptr,        // [T, Hq, hd] half(本 chunk q)
    const void* k_data,       // k_fi [nb, page, hkv, hd] half(kNHD)
    const void* v_data,       // v_fi [nb, page, hkv, hd] half(kNHD)
    void* out_ptr,            // [T, Hq, hd] half
    int32_t* q_cu_seqlens,    // device [B+1] = [0, T]
    int32_t* indices,         // device 页表(物理页 id 拼接)
    int32_t* indptr,          // device [B+1] = [0, nb]
    int32_t* last_len,        // device [B] = ctx_total % page
    const int64_t* plan15,
    void* int_ws, size_t int_ws_size,
    void* float_ws, size_t float_ws_size,
    int32_t batch_size,
    int32_t num_qo_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t page_size,
    int32_t total_num_rows,
    float sm_scale,
    cudaStream_t stream) {
  try {
    using DTypeQ = half;
    using DTypeKV = half;
    using DTypeOut = half;
    using IdType = int32_t;
    using AttentionType = OwlFiAttention<false, false, false, false>;
    using ParamsType = BatchPrefillPagedParams<DTypeQ, DTypeKV, DTypeOut, IdType>;

    // 注:plan15 = PrefillPlanInfo::ToVector 的 15 字段(无 tag 前缀;
    // attention.rs 的 tag 约定在 Rust 侧)。合法性由 FromVector 断言。
    PrefillPlanInfo plan_info;
    std::vector<int64_t> vec(plan15, plan15 + 15);
    plan_info.FromVector(vec);
    if (plan_info.total_num_rows_offset >= (int64_t)int_ws_size ||
        plan_info.v_offset >= (int64_t)float_ws_size ||
        plan_info.s_offset >= (int64_t)float_ws_size) {
      return -201;  // workspace 越界(尺寸不足)
    }

    paged_kv_t<DTypeKV, IdType> paged_kv(
        (uint32_t)num_kv_heads, (uint32_t)page_size, (uint32_t)head_dim,
        (uint32_t)batch_size, QKVLayout::kNHD,
        (DTypeKV*)k_data, (DTypeKV*)v_data,
        indices, indptr, last_len);

    ParamsType params(
        (DTypeQ*)q_ptr, paged_kv, /*custom_mask=*/nullptr,
        q_cu_seqlens,
        /*qo_indptr=*/nullptr, /*kv_indptr=*/nullptr,
        (DTypeOut*)out_ptr, /*lse=*/nullptr, /*merge_lse=*/nullptr,
        (uint32_t)num_qo_heads, (uint32_t)(num_qo_heads * head_dim), (uint32_t)head_dim,
        /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale,
        /*rope_scale=*/1.0f, /*rope_theta=*/10000.0f);

    params.request_indices = GetPtrFromBaseOffset<IdType>(int_ws, plan_info.request_indices_offset);
    params.qo_tile_indices = GetPtrFromBaseOffset<IdType>(int_ws, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices = GetPtrFromBaseOffset<IdType>(int_ws, plan_info.kv_tile_indices_offset);
    params.o_indptr = GetPtrFromBaseOffset<IdType>(int_ws, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = GetPtrFromBaseOffset<IdType>(int_ws, plan_info.kv_chunk_size_ptr_offset);
    params.max_total_num_rows = plan_info.total_num_rows;
    params.padded_batch_size = plan_info.padded_batch_size;
    params.partition_kv = plan_info.split_kv;
    params.merge_indptr = nullptr;
    params.block_valid_mask = nullptr;
    params.total_num_rows = nullptr;
    if (plan_info.split_kv) {
      params.merge_indptr = GetPtrFromBaseOffset<IdType>(int_ws, plan_info.merge_indptr_offset);
    }

    DTypeOut* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (plan_info.split_kv) {
      tmp_v = GetPtrFromBaseOffset<DTypeOut>(float_ws, plan_info.v_offset);
      tmp_s = GetPtrFromBaseOffset<float>(float_ws, plan_info.s_offset);
    }

    if (getenv("FI_DEBUG")) {
        fprintf(stderr, "[owl-fi][run] hd=%d page=%d T=%d hq=%d hkv=%d split=%d cta=%u padded=%u\n",
                head_dim, page_size, total_num_rows, num_qo_heads, num_kv_heads,
                (int)plan_info.split_kv, plan_info.cta_tile_q, plan_info.padded_batch_size);
    }
    cudaError_t st = cudaSuccess;
    DISPATCH_CTA_TILE_Q(plan_info.cta_tile_q, CTA_TILE_Q, {
      DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
        st = BatchPrefillWithPagedKVCacheDispatched<
            CTA_TILE_Q, HEAD_DIM, HEAD_DIM,
            PosEncodingMode::kNone, false, MaskMode::kCausal,
            AttentionType, ParamsType>(params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
      });
    });
    return (int)st;
  } catch (const std::exception& e) {
    fprintf(stderr, "[owl-fi][run] %s\n", e.what());
    return -102;
  }
}
