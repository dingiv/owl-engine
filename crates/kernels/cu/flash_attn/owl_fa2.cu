// ============================================================================
// **FlashAttention-2 varlen forward** —— 直接搬 vLLM vendored FA2 的前向核。
//
// 来源：packages/vllm/.deps/vllm-flash-attn-src（BSD-3-Clause，Tri Dao /
// vLLM contributors；完整 LICENSE 在 ../vendor/LICENSE）。本文件**只写 host
// 胶水**：把 owl 的裸指针填进 Flash_fwd_params，调用上游
// `run_mha_fwd_hdim96<half,false>`。核体/分块/online softmax/裁剪全部是
// vLLM 原版，没有自研算子。
//
// 为什么 hd=72 填 hdim=96：
//   vLLM `set_params_fprop` 对 head_dim<=128 取 `round_multiple(d,32)`，
//   72 → **96**；`HEADDIM_SWITCH(params.d)` 选 `kHeadDim=96`，同时保留
//   `params.d=72`，`Is_even_K=false`，核内把 72 之后的 24 列按 0 处理。
//   这正是 vLLM Qwen3.5-ViT（hd72）实际走的路径，**不是 Python 侧 pad**。
// hd=64（4B 塔）直接用 `run_mha_fwd_hdim64`，_d_rounded=64，Is_even_K=true。
//
// 契约：q/k/v/out 均 [total_q, heads, head_dim] f16 连续；cu_seqlens int32
// [batch+1]（patch 级）；lse f32 [heads, total_q]（FA2 的 unpadded LSE 形状）。
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#define FLASHATTENTION_DISABLE_DROPOUT 1
#define FLASHATTENTION_DISABLE_SOFTCAP 1
#define FLASHATTENTION_DISABLE_LOCAL 1
#define FLASHATTENTION_DISABLE_ALIBI 1

#include "flash_fwd_launch_template.h"

extern "C" int owl_fa2_varlen_fwd_f16(
    const void* q_ptr, const void* k_ptr, const void* v_ptr,
    void* out_ptr, float* lse_ptr, int32_t* cu_dev,
    int batch, int total_q, int max_seqlen,
    int heads, int head_dim, float scale,
    cudaStream_t stream) {
  if (batch <= 0 || total_q <= 0 || max_seqlen <= 0 || heads <= 0 ||
      (head_dim != 64 && head_dim != 72) || q_ptr == nullptr || k_ptr == nullptr ||
      v_ptr == nullptr || out_ptr == nullptr || lse_ptr == nullptr ||
      cu_dev == nullptr) {
    return (int)cudaErrorInvalidValue;
  }
  using namespace FLASH_NAMESPACE;
  Flash_fwd_params p{};
  p.q_ptr = const_cast<void*>(q_ptr);
  p.k_ptr = const_cast<void*>(k_ptr);
  p.v_ptr = const_cast<void*>(v_ptr);
  p.o_ptr = out_ptr;
  p.softmax_lse_ptr = lse_ptr;
  const int64_t row_stride = (int64_t)heads * head_dim;
  p.q_row_stride = row_stride;
  p.k_row_stride = row_stride;
  p.v_row_stride = row_stride;
  p.o_row_stride = row_stride;
  p.q_head_stride = head_dim;
  p.k_head_stride = head_dim;
  p.v_head_stride = head_dim;
  p.o_head_stride = head_dim;
  p.h = heads;
  p.h_k = heads;
  p.h_h_k_ratio = 1;
  p.b = batch;
  p.seqlen_q = max_seqlen;
  p.seqlen_k = max_seqlen;
  p.seqlen_q_rounded = ((max_seqlen + 127) / 128) * 128;
  p.seqlen_k_rounded = p.seqlen_q_rounded;
  p.d = head_dim;
  p.d_rounded = ((head_dim + 31) / 32) * 32; // 72 -> 96（vLLM 同式）
  p.total_q = total_q;
  // 视觉契约的 cu 是 f32（native/FI 旧槽），FA2 要 int32；由 runtime 在
  // **每个 tower 请求的 cu 块**变化时转好并上传到 fa2_cu（batch 段界）。
  // 核内不再覆盖 cu：多图时否则会把真实分段写成 [0,total_q]。
  p.cu_seqlens_q = cu_dev;
  p.cu_seqlens_k = cu_dev;
  p.scale_softmax = scale;
  p.scale_softmax_log2 = scale * 1.4426950408889634f;
  p.softcap = 0.0f;
  p.p_dropout = 1.0f; // 保留概率 = 1 ⇒ Is_dropout=false
  p.rp_dropout = 1.0f;
  p.scale_softmax_rp_dropout = scale;
  p.p_dropout_in_uint8_t = 255;
  p.is_causal = false;
  p.window_size_left = -1;
  p.window_size_right = -1;
  p.is_seqlens_k_cumulative = true;
  p.unpadded_lse = true;
  p.is_bf16 = false;
  p.num_splits = 0;
  p.seqlenq_ngroups_swapped = false;

  if (head_dim == 64) {
    run_mha_fwd_hdim64<cutlass::half_t, false>(p, stream);
  } else {
    run_mha_fwd_hdim96<cutlass::half_t, false>(p, stream);
  }
  return (int)cudaGetLastError();
}
