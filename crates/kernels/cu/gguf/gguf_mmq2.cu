// ============================================================================
// owl GGUF 族:**head 版 int8-MMA MMQ** 适配 TU(W-2f;设计 §二十二)
//
// 内核本体 = **llama.cpp fork 逐字 vendoring**(`vendor2/`,见其 README)。
// 本文件负责:①把 vendored 的 `mul_mat_q<type, J, fallback>` 暴露为
// `extern "C"` 入口(逐型一个字面入口,与既有核族同一条 H3 纪律);
// ②host 侧选 J / fallback 并按 J 分档发射(照 `vendor2/mmq.cuh::mul_mat_q_case`);
// ③**不**引入 ggml 的 split / stream-k 机制(首版强制 `stream_k=false`)。
//
// # 与非 tensor-core 版(§十八 `vendor/mmq.cu`,2024-05 同代)的差别
//
// 那一代 `vecdotq.cuh` 里 **0 条 `mma`**:点积走 dp4a,tile 配置再大也只有
// ~16 TFLOPS 等效(实测 27B prefill 557 t/s)。本版走 `mma.sync.m16n8k16.s8`
// (sm80+ 的 int8 张量核),llama.cpp 同机实测 **103 TFLOPS 等效**(§21.1)
// = 我们 cuBLAS 臂(f16×f32 累加,66 TFLOPS)的 **1.53×**,且**权重零搬运**。
//
// # 状态(2026-10-19):**只到"能编"**
//
// 本 TU 目前是**编译桩**:vendoring 的依赖闭包已证明可在本仓 nvcc 旗标下编过
// (`-fatbin -O3 -std=c++17 --expt-relaxed-constexpr`,sm_86,见 README);
// 真正的 `extern "C"` 入口 + `block_q8_1_mmq` 激活布局 + 分派接线按 §22.3 的
// 清单推进。**未接线 ⇒ 现役路径零影响**(既有 §十八 的 dp4a MMQ 与 cuBLAS 臂不动)。
// ============================================================================
#include "vendor2/mmq.cu"
