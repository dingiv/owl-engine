# awq-forge

FP8 / BF16 → **AWQ INT4**（W4A16，g32-asym，compressed-tensors `pack-quantized`）。

> 立项文档：`docs/REQUIREMENTS.md`
> 架构：`docs/arch/00-overview.md`
> 当前阶段：**0.8B / 4B 转换已通；27B FP8 → AWQ 已转换并通过质量门**
> （GSM8K 200 96.5% vs FP8 97.0%，Δ=-0.5pp/p=1.00；needle 16k 3/3；std 15/15）。
> 下一步是补 4B 的 GSM8K+needle+std 质量门。
> 调参实录：`docs/bench/2026-10-29-qwen35-08b-awq-tuning.md`。
> 27B 实录：`docs/bench/2026-10-29-pi27b-fp8-awq.md`。
> **速度与注意事项（为什么慢/怎么变快/评测口径）：
> `docs/bench/2026-10-29-speed-and-usage-notes.md`**。

## 为什么独立

owl 是推理引擎，只读现成的量化检查点。本工具负责离线转换：

- 输入：BF16 dense 或 FP8 blockwise；
- 输出：与 `models/cyankiwi/Qwen3.8-27B-AWQ-INT4` 同布局的 CT `pack-quantized`；
- 27B：逐层流式反量化，不落 54GB dense 中间态。

## 快速开始

```bash
cd packages/owl/tools/awq-forge
uv sync --extra quant --group dev
uv run awq-forge inspect /path/to/model
```

0.8B 冒烟：

```bash
uv run awq-forge inspect /home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B
```

当前已实现 `inspect`；`run/verify` 尚未接入。量化尖刺脚本：

```bash
# 主后端（llm-compressor）
/tmp/llmcomp-venv/bin/python scripts/spike_llmcompressor.py \
  --source models/Qwen/Qwen3.5-0.8B \
  --output /tmp/qwen35-08b-awq \
  --device cuda:0 --samples 128 --seq-len 2048 \
  --pipeline sequential --seq-offload cuda --batch-size 8 --n-grid 10

# 4B 生产档参数：batch 4
# 27B FP8 解码候选：scripts/spike_gptqmodel.py（只借解码，不借 AWQ）
```

## 后端

- 主：**llm-compressor 0.15** AWQ（Qwen3.5 实测通过，原生 CT 导出）
- 参考：AutoAWQ fork `quivent/autoawq-qwen35`
- GPTQModel 7.7：0.8B AWQ 质量失败，仅保留 FP8 解码候选
- 导出：llm-compressor 原生 CT；AutoAWQ 布局用官方 `AutoAWQConverter`

## 目录

```
docs/
  REQUIREMENTS.md
  arch/
  calltrees/
src/awq_forge/
  cli.py
  source.py
  inventory.py
tests/
```

## 非目标

不做推理、KV cache 量化、W4A8、NVFP4、GGUF、训练/QAT。
