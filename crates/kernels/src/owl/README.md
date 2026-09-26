# src/owl —— NInfer 移植位(工单 N;2026-09-26)

> 源:repos/ninfer_duo_3080(C++/CUDA,bf16 基,SM86 调优,Qwen3.6/3.8
> GDN hybrid 原生引擎)。本目录 = 上游核的 **owl 契约改写**(extern "C"
> 单输出末参、f16 基线、无 namespace/launcher —— nvrtc 纪律)。

## 已移植(已验证)

| 核 | 上游 | owl 版 | 对拍 |
|---|---|---|---|
| sigmoid_gate_mul | `sigmoid_gate_mul.cuh`(bf16x8 包,fp32 sigmoid) | `owl_sigmoid_gate_mul_f16`(单输出契约;__half2 主路) | max_diff 0.00049(attention.rs 门乘融合:sigmoid+mul 双发射 → 单核) |

## 全库处置表(逐核盘点结论)

| NInfer 核 | owl 处置 | 理由 |
|---|---|---|
| l2norm / rmsnorm / rope / embed_gather | **已被覆盖**(F2/F3 桥) | owl 同语义核已在注册表(f32/f16 双档);上游价值 = 向量化手法(__nv_bfloat162 对装),融合优化期再摘 |
| causal_conv1d(decode 槽) | **已被覆盖**(F4 桥 conv_upd) | 同上;上游 k=4 收窄同款 |
| gqa_attention_decode | **候选**(F4 后置) | 分页 KV 形态 vs owl 连续槽窗契约差异大;owl naive 窗核已对拍可用,换装 = paged 立项时一并 |
| gqa_attention_prefill_bf16 | **候选**(PF1b 决策点) | chunked prefill attention,SM86 调优 —— 与 xinfer/FlashInfer 三选一,PF1b 时比选 |
| gated_delta_net/recurrent + chunked/ | **候选**(PF1b 决策点) | FLA chunk 算法(WY/WU + state passing,真并行 chunk)—— 与 xinfer t-loop varlen 双源比选;重构量大,立项时评估 |
| W4A4/int8/NVFP4 线性核 | **phase5 矿源** | SM86 tensor core MMA 量化 GEMM(Qwen3.8-27B NVFP4 实测 143-766 t/s decode);量化域立项主参照 |

## 移植手法备忘(本目录范式)

1. 上游核摘数学体 → extern "C" 具名核(nvrtc:无 namespace/<<<>>>/launcher);
2. bf16→f16:桥式读写(__nv_bfloat16 → __half,float 计算);
3. owl 槽序:输出块固定末参(lower_kernel 按 sig 自动装配,末位 T = out);
4. 对拍:照 layers/*.rs f16_tests 模式(host f32 参考 + f16 量化输入)。
