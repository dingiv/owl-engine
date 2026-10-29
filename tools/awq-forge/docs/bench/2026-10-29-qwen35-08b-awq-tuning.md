# 2026-10-29 Qwen3.5-0.8B AWQ 调参 / 四卡实测

> 目的：验证 llm-compressor 的 AWQ 质量，并把 0.8B 的耗时压到可接受范围；
> 同时判定“多卡能否加速单模型 AWQ”。

## 结论

1. **主后端改为 llm-compressor**。GPTQModel 7.7 在 Qwen3.5-0.8B 上即使修正
   GDN scale 分组，输出仍是 `1723`/全 0；同模型 llm-compressor 输出 `391`。
2. 0.8B 最优档：`seq_offload=cuda`、`batch_size=8`、`n_grid=10`，
   **57s**，质量 `391`；`n_grid=5` 也通过，58s。
3. 4B 单卡 3090：`batch_size=8` + `seq_offload=cuda` OOM；
   `batch_size=4` + `seq_offload=cuda` **265s**，15.25 GiB，质量 `391`。
4. **4B 双卡（关键修正）**：`sequential_offload_device="cuda:1"` 把
   **中间激活** 卸到第二张 3090，主卡只放权重/当前子图：
   `batch=8` + `n_grid=10` → **229s（3m49s）**，17.2 GiB，质量 `391`。
   之前写 `cuda` 会解析成 `cuda:0`，等于没换卡，所以 batch8 OOM。
5. **单模型四卡不加速**：llm-compressor `device_map=auto` + sequential
   在 4 卡上仍 OOM；4 卡 DDP 数据并行 AWQ 在 Qwen3.5 上触发 NCCL
   all-reduce 超时（未解决）。AWQ 是逐层串行；四卡正确用法是并行跑
   四条配置（已实测）。

## 0.8B 矩阵

统一：128 条 × 2048 token，Qwen3.5-0.8B，g32-asym，`pipeline=sequential`。

| 配置 | n_grid | batch | seq_offload | 耗时 | 峰值 | 17*23 |
|---|---:|---:|---|---:|---:|---|
| 旧默认 | 20 | 1 | cpu | 394s | 4.9G | 391 |
| 单卡 | 20 | 1 | cuda | 131s | 7.8G | 391 |
| 单卡 | 20 | 8 | cuda | 124s | 11.2G | 391 |
| 单卡 | 10 | 4 | cuda | 72s | 6.2G | 391 |
| **单卡** | **10** | **8** | **cuda** | **57s** | **11.0G** | **391** |
| 单卡 | 5 | 4 | cuda | 58s | 7.4G | 391 |

> 0.8B 单卡已够快；`cuda:1` 是 4B 才需要的关键旋钮。

四卡并行：以上四条配置分别钉在 GPU 0/1/2/3，同时完成；说明四卡适合
“并行扫配置”，不适合“单任务加速”。

## 4B

统一：128×2048，g32-asym，`pipeline=sequential`。

| 配置 | 结果 |
|---|---|
| n_grid=10, batch=8, seq_offload=cuda | OOM（23.6G 满） |
| n_grid=10, batch=4, seq_offload=cuda | 265s / 15.25G / 391 |
| **n_grid=10, batch=8, seq_offload=cuda:1** | **229s / 17.2G / 391** |

4B 的 `cuda:1` 档在 `Propagating` 阶段有 allocator 重试 OOM 警告，但最终完成；
生产建议加 `PYTORCH_ALLOC_CONF=expandable_segments:True` 复测，或 batch=6。

## 可调参数（llm-compressor 0.15）

| 参数 | 作用 | 本次取值 |
|---|---|---|
| `pipeline` | `sequential` 逐层；`basic` 整模型（0.8B 都 OOM） | sequential |
| `sequential_offload_device` | **中间激活** offload 到 cpu/cuda；多卡推荐 `cuda:1` | 0.8B `cuda`；4B `cuda:1` |
| `sequential_targets` | 子图粒度，`Linear` 更省显存但更慢 | 默认 decoder layer |
| `AWQModifier.n_grid` | scale 搜索点数，20→10→5 | 10 |
| `AWQModifier.duo_scaling` | x/w 双缩放 | True |
| `AWQModifier.offload_device` | AWQ 激活缓存 offload | None |
| `oneshot.batch_size` | 校准 batch | 8（0.8B）/4（4B） |
| `num_calibration_samples` | 校准条数 | 128 |
| `max_seq_length` | 校准长度 | 2048 |

`apply_clip` 与 `max_chunk_memory` 在 llm-compressor 0.15 的 AWQModifier
没有公开字段；llm-compressor 的 AWQ 固定执行 clip。

## 复现

```bash
# 0.8B 最优档
CUDA_VISIBLE_DEVICES=1 /tmp/llmcomp-venv/bin/python scripts/spike_llmcompressor.py \
  --source models/Qwen/Qwen3.5-0.8B \
  --output models/empero-ai/Qwen3.5-0.8B-AWQ32-g10b8 \
  --device cuda:0 --samples 128 --seq-len 2048 \
  --pipeline sequential --seq-offload cuda --batch-size 8 --n-grid 10
```

## 判定

- llm-compressor：**质量通过，速度通过（4B 4m25s），主后端**。
- GPTQModel：0.8B 质量失败，**不用于 Qwen3.5 AWQ**；仅保留其 FP8
  module-local decoder 作为 27B 流式反量化候选（待 P1 验证）。
- 四卡：用于并行矩阵/多模型，不承诺单任务线性加速。
