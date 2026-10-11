// ============================================================================
// owl XQA 薄 binding(2026-10-13;长 ctx verify 立案 D3)
//
// ported from: repos/vllm-ampere-optimized/flashampere/xqa/csrc/xqa_wrapper.cu
//              (avesed/vllm-ampere-optimized @ f2e081d;XQA 本体 = FlashInfer
//               csrc/xqa,mha.cu 冻结;该 fork 做了两件事:Ampere un-gate +
//               裁掉 sm90/sm120 include 闭包)
// 上游基线:repos/flashinfer/csrc/xqa/(24 文件);本目录 = Ampere 闭包 16 文件。
//
// 为什么用它(而不是自研):XQA 是 **decode 形状**注意力 —— KV-split +
// warp-specialized + GQA 内共享 K/V 读,**q 维扩展近乎免费**(SPEC_DEC 模式
// q=1+K),正是 owl verify 缺的形态(现 v2 伪序列臂 = 8 条序列各读一遍全 ctx,
// 元素数 ×8)。参考 fork 实测:q=1+K 比 FA2 fwd_kvcache 快 1.8-4.3×(32-128k)。
//
// 本文件替换上游的 tvm_ffi binding(`flashinfer_xqa_binding.cu` 需要 TVM FFI
// 运行时,owl 不需要):只做 TensorView → 裸指针 + 调 launchMHAFlashInfer。
//
// 实例化参数(编译期,由 build.rs 传):
//   TOKENS_PER_PAGE=32 / HEAD_ELEMS=256 / HEAD_GRP_SIZE=6 / CACHE_ELEM_ENUM=2
//   (fp8 e4m3 KV)/ INPUT_FP16=1 + DTYPE=__half /
//   SLIDING_WINDOW=0 / LOW_PREC_OUTPUT=0 / USE_SM90_MHA=0 / MLA_WRAPPER=0
//
// **两份实例化(2026-10-13 ② 立案)**:
//   verify → `owl_xqa_launch`        (SPEC_DEC=1,M_TILESIZE=32:T=8 伪序列、48 行需 2 block)
//   decode → `owl_xqa_launch_decode` (SPEC_DEC=1,**M_TILESIZE=16**:q=1 只需 6 行)
// 为什么要两份:`rowsPerBlock = M_TILESIZE` 决定一个 CTA 覆盖多少 head-token 行,
// 而 `warpTile.y = roundUp(nbValidRows, 16)` ⇒ q=1 用 32 行 tile 时只 6 行有效
// (实测 ctx8192 仅 1.38×,vs q=8 的 4.73×)。收窄到 16 行后浪费减半。
// ⚠️ 不用 `SPEC_DEC=0`:`mha.cu:84` 在该模式下 `nbValidRows = headGrpSize*beamWidth`,
// 本 fork 编译出的实例化在 `cuLaunchKernelEx` 内 libcuda **SIGSEGV**(已实测,
// 参数/尺寸/符号均正常)—— 掩码路径(SPEC_DEC=1)已验证可用,只收窄 tile 最稳。
// 两份实例化用 `-D` 宏重命名内部全局符号(否则同进程两归档 `launchMHAFlashInfer`
// 撞名,链接器只取一个)。
//
// 签名两份一致(q_seq_len/mask 恒在,两者都是 SPEC_DEC=1)。
//
// 布局契约(与 owl kNHD 主池逐字对齐;XQA 文档名 = NHD):
//   KV = [num_pages, tokens_per_page, num_kv_heads, head_dim] —— owl 页内
//   [page,hkv,hd] 连续即此布局;strides 由调用方按元素给(host 契约)。
//   q/out = [batch, 1, q_seq_len, num_q_heads, head_dim](owl verify 的
//   [T, hq*hd] 是它的 free view)。
//   page_table = int32[batch, max_pages](⚠️ owl 现有 bt 是 f32 —— 需 i32 孪生);
//   seq_lens = uint32[batch](owl kv_lens 是 f32 —— 需 u32 孪生);
//   mask = spec-dec 位掩码,行 i 的 bit j = "可看本块内第 j 个 key",
//          因果 8 行 = 0x1,0x3,…,0xFF(见 roadmap §P2 尾 / flashampere
//          xqa_verify.py _buffers 逐式);
//   semaphores = uint32[round_up(Hkv*batch,2)+2+Hkv*batch+2](发射前清零);
//   scratch = 工作区(参考实现给 256MiB,持久复用;图内安全)。
// ============================================================================
#include <cuda_runtime.h>

#include <cstdio>
#include <exception>

#include "mha.h"

extern "C" int owl_xqa_launch(
    uint32_t sm_count,        // 3090 Ti = 84
    uint32_t nb_k_heads,      // hkv = 4
    uint32_t sliding_win_size,// 0 = 关(本实例化 SLIDING_WINDOW=0)
    float q_scale,            // = attn_scale * sqrt(head_dim)(核内再除 sqrt(D))
    void* output,             // [batch,1,q_seq_len,Hq,D] f16
    void* q,                  // [batch,1,q_seq_len,Hq,D] f16
    void* k_cache,            // [num_pages,tokens_per_page,Hkv,D] fp8 e4m3
    void* v_cache,            // 同上
    const int32_t* page_table,// [batch, max_pages]
    uint32_t max_seq_len,     // = max_pages * tokens_per_page(图内定形常量)
    const uint32_t* seq_lens, // [batch]
    uint32_t batch_size,
    float kv_scale,           // 1.0(fp8 KV 无 per-tensor scale)
    uint32_t q_seq_len,       // verify = depth+1 = 8;decode = 1(另一实例化)
    const uint32_t* mask,     // spec-dec 位掩码(q_seq_len>1 必填)
    uint32_t* semaphores,
    void* scratch,
    uint64_t kv_stride_page,  // 元素序(= tokens_per_page*Hkv*D)
    uint64_t kv_stride_token, // 元素序(= Hkv*D)
    uint64_t kv_stride_head,  // 元素序(= D)
    void* stream) {
  // ⚠️ XQA 宿主侧用 C++ 异常做参数校验(`hostUtils.h`/`utils.h` 的
  // throw std::runtime_error)。**绝不能让它穿过 `extern "C"` 边界**:
  // 实测 = 调用方进程直接 abort(`Rust cannot catch foreign exceptions`),
  // 且异常文本丢失(难以定位)。此处就地捕获 → 打印原因 + 返回非零码。
#if !SPEC_DEC
  // 若将来真要用 SPEC_DEC=0:q 维/掩码两个入参只为统一签名而保留
  (void)q_seq_len;
  (void)mask;
#endif
  try {
    launchMHAFlashInfer(sm_count, nb_k_heads, sliding_win_size, q_scale, /*qScalePtr=*/nullptr,
                      reinterpret_cast<OutputHead*>(output),
                      reinterpret_cast<InputHead const*>(q), /*attentionSinks=*/nullptr,
                      reinterpret_cast<GMemCacheHead*>(k_cache),
                      reinterpret_cast<GMemCacheHead*>(v_cache),
                      page_table, max_seq_len, seq_lens, batch_size, kv_scale,
                      /*kvScalePtr=*/nullptr,
#if SPEC_DEC
                      q_seq_len, /*qCuSeqLens=*/nullptr, mask,
#endif
                      semaphores, scratch, /*enable_pdl=*/false,
                        kv_stride_page, kv_stride_token, kv_stride_head,
                        reinterpret_cast<cudaStream_t>(stream));
  } catch (std::exception const& e) {
    std::fprintf(stderr, "[owl-xqa] 参数拒绝: %s\n", e.what());
    return 1;
  } catch (...) {
    std::fprintf(stderr, "[owl-xqa] 参数拒绝: 未知异常\n");
    return 2;
  }
  return 0;
}
