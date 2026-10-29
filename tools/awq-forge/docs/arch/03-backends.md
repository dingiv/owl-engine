# 后端选型与尖刺

## 结论（2026-10-29 晚更新）

**主后端：llm-compressor 0.15 的 AWQ。**

0.8B 实测：llm-compressor 输出 `391`；GPTQModel 7.7（含 GDN scale 分组修正）
输出 `1723`/全 0。GPTQModel 的 Qwen3.5 AWQ 映射仍不可用。

llm-compressor 调参后：

- 0.8B：`seq_offload=cuda` + `batch=8` + `n_grid=10` → **57s**，质量 `391`；
- 4B：`seq_offload=cuda` + `batch=4` + `n_grid=10` → **265s（4m25s）**，
  峰值 15.25 GiB，质量 `391`。

原因：

1. AWQ 映射对 Qwen3.5 的 GDN/full-attention/MLP 已验证；
2. 原生输出 compressed-tensors `pack-quantized`，与 cyankiwi 布局一致；
3. `sequential` + `sequential_offload_device="cuda"` 在 24G 内可跑 4B；
4. `n_grid` / `batch_size` / `pipeline` / `sequential_targets` 可调。

## GPTQModel 的定位

- 0.8B AWQ 质量失败，**不作为 Qwen3.5 AWQ 主后端**。
- 仅保留其 native FP8 module-local decoder（`AutoModuleDecoderConfig` +
  `LazyTurtle`）作为 27B FP8 流式反量化的候选，待 P1 单独验证。
- 它的 `true_sequential` / `chunked` 设计仍值得借鉴，但当前不能直接用于
  Qwen3.5 的 GDN AWQ。

## 候选对照

| 方案 | Qwen3.5 GDN | FP8 源 | 低显存 | AWQ | CT 导出 | 结论 |
|---|---|---|---|---|---|---|
| **llm-compressor 0.15** | 实测 0.8B/4B 质量通过 | CT 原生 | sequential + offload=cuda 后 4B 15.25G | 原生 | 原生 CT | **主** |
| GPTQModel 7.7 | 映射错误，0.8B 输出坏 | 原生 FP8/NVFP4 逐模块 | true_sequential + chunked | 原生 | GEMM → AutoAWQConverter | 仅 27B FP8 解码候选 |
| AutoAWQ fork `quivent/autoawq-qwen35` | 专门 patch | 不支持 | Catcher 只缓存 layer0 | 原生 | AutoAWQConverter | 参考 |
| 自研 | 要自己接 | 要自己写 | 可控 | 要自己写 | 要自己写 | 不采用 |

## 为什么 llm-compressor 能成为主后端

- 它的 AWQ 映射在 Qwen3.5 GDN 上实测正确（0.8B/4B 都输出 391）。
- 上次 4B 慢/ OOM 的根因是 `sequential_offload_device="cpu"` + `batch=1`；
  改成 `cuda` + `batch=4` 后 4B 为 4m25s，峰值 15.25 GiB。
- 原生 CT 导出，不需要 AutoAWQConverter。

> GPTQModel 曾因 FP8 逐模块解码和 `true_sequential` 被选为主后端；
> 但 0.8B AWQ 质量实测失败，已降级为 27B FP8 解码候选。

## 多卡语义（重要）

### 1. `sequential_offload_device` 卸的是激活，不是权重

llm-compressor 0.15 的 docstring 写得很明确：

> Device used to offload **intermediate activations** between sequential
> layers. It is recommended to use `cuda:1` if using more than one gpu.

- 0.8B：`cuda`（当前主卡）就够；
- 4B：`cuda:1` 把激活缓存放第二张 3090，主卡只放权重/当前子图，
  batch8 才能进 24G；实测 229s / 17.2G。
- 之前写 `cuda` 等于 `cuda:0`，没有换卡，所以 batch8 OOM。

### 2. 权重层切分没有现成调度器

`compressed_tensors.offload` 本身有 `DeviceCache`，`offload_module(
module, onload_device, offload_device)` 可以把**权重**卸到另一张卡；
`dispatch_with_map` 也接受 `(onload, offload)` 的 device map。

但 `SequentialPipeline` 只做：

```
onload_device = get_main_device()          # 当前 rank 的主卡
subgraph_onload_modules(...)               # 当前子图搬到主卡
...
subgraph_offload_modules(...)              # 搬回各自 offload_device
```

没有“按层轮流放到不同 GPU”的 scheduler。`device_map="auto"` 更被官方
明确不推荐（会把整个模型先塞进 GPU，反而 OOM）。

### 3. 官方多卡路径 = 分布式 oneshot（DDP 数据并行）

- 每个 rank 一份模型副本，吃 1/N 校准数据；
- AWQ 用 all-reduce 同步 activation sums / partial MSE；
- 官方 benchmark：8B 单卡 745s → 4 卡 264s。

我们实测 0.8B 4 卡 DDP：模型/数据分区正常，AWQ 跑到 25/25 层，但在
finalize 的 NCCL all-reduce 超时（600s），未跑通。这是当前 llm-compressor
在 Qwen3.5 AWQ 上的 DDP 兼容问题，P1 再排。

### 4. 当前正确姿势

| 模型 | 用法 |
|---|---|
| 0.8B | 单卡，`seq_offload=cuda`，batch8，57s |
| 4B | 双卡，`seq_offload=cuda:1`，batch8，229s |
| 四卡 | 并行跑四条配置/四个模型，不做单任务 layer split |
| 27B | 单卡放不下；要么 P1 解决 DDP/自定义层流，要么换更小源/更多显存 |

## 尖刺计划（P0a → P0b）

### A. 0.8B 接线（已完成）

- 源：`models/Qwen/Qwen3.5-0.8B`（24 层，1.7GB）。
- 校准：UltraChat 128×2048。
- 结果：llm-compressor `n_grid=10`、`batch=8`、`seq_offload=cuda` → 57s，
  vLLM 输出 `391`。
- 结论：Qwen3.5 GDN 量化面/忽略面正确；CT 四件套完整。

### B. 4B 性能/质量（已完成性能）

- 源：`models/empero-ai/Qwen3.8-4B-Distill`（32 层，8.68GB）。
- 配置：128×2048，`n_grid=10`、`batch=8`、`seq_offload=cuda:1`。
- 结果：**229s / 17.2 GiB / 391**；`seq_offload=cuda`（同卡）时 batch8 OOM。
- 说明：`sequential_offload_device` 只卸**中间激活**，不卸权重；权重层切分
  不在 llm-compressor 支持范围。
- 待做：GSM8K 200、needle 16k、std 15/15。

### C-27B. 27B FP8 → AWQ 质量门（2026-10-29 完成）

- 源 `bytkim/Qwen3.8-27B-pi-FP8`（blockwise 128×128，33G）
  → `bytkim/Qwen3.8-27B-pi-AWQ-INT4`（19G，g32-asym，400 Linear，无 MTP）。
- GSM8K 200（greedy/thinking-off，预算 1536）：AWQ **96.5%** vs FP8 参考
  **97.0%**，Δ=-0.5pp、McNemar p=1.00；owl 与 vLLM 跑同一产物逐题 0 翻转。
- needle 16k：head/mid/tail 全中（16233 tok）；std：15/15。
- 细节与复现：`docs/bench/2026-10-29-pi27b-fp8-awq.md`。

### C. 后端 A/B（已完成）

| 后端 | 0.8B 质量 | 0.8B 耗时 | 结论 |
|---|---|---:|---|
| llm-compressor | 391 | 57s | 主 |
| GPTQModel 7.7 | 1723 / 全 0 | 135s | 淘汰 |
| AutoAWQ fork | 未跑 | - | 仅参考 |

## 尖刺脚本边界

- `scripts/spike_llmcompressor.py`：主后端；
- `scripts/spike_gptqmodel.py`：仅保留 FP8 解码/历史对照；
- 尖刺通过后再抽成 `awq_forge.backends.llmcompressor`。

## 4B 测试中已确认

1. `linear_attn.in_proj_a/b`、`conv1d`、`vision`、`mtp` 保持 BF16；
2. 200 个 Linear 被量化（4B），CT 键面为 `weight_packed/weight_scale/weight_zero_point/weight_shape`；
3. vLLM compressed-tensors WNA16 走 Marlin kernel，g32 可装载；
4. `tie_word_embeddings=true` 不产生重复 `lm_head`。

## 版本锚点

- llm-compressor：0.15.0
- compressed-tensors：0.20.1
- transformers：5.21.0（llm-compressor 0.15 上限）
- torch：2.14.1（CUDA 13.x）
- GPTQModel：7.7.0（仅 FP8 解码候选）
