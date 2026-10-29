# awq-forge 架构总览

## 一句话

`awq-forge` 是一台**批处理状态机**：源检查点 → 量化计划 → 逐层 AWQ → CT 重打包 →
验收 → 原子发布。它不是服务，没有常驻事件循环。

## 分层

```
CLI / job.toml
  └─ application
       ├─ source       源检查点探测、safetensors 索引、FP8 解码
       ├─ model        HF 模型壳 / Qwen3.5 模块树 / 设备与生命周期
       ├─ backend      GPTQModel AWQ（主）/ AutoAWQ（参考）
       ├─ export       compressed-tensors pack-quantized
       ├─ verify       结构、数值、引擎装载、质量门
       └─ publish      manifest + 原子重命名
```

依赖方向只能向下：

```
cli → application → {source, model, backend, export, verify} → domain
```

`domain` 只放纯数据与不变式，不 import torch / transformers。

## 领域状态

| 状态 | 所有者 | 生命周期 | 关键不变式 |
|---|---|---|---|
| `JobSpec` | application | 一次 run | 源/输出/方案/校准全部显式 |
| `SourcePlan` | source | 只读 | 张量清单、FP8 形态、忽略面、体积 |
| `CalibrationSet` | model | 一次 run | 固定 128×2048，tokenizer 与源一致 |
| `AwqScales` | backend | 单层临时 | 只对目标 Linear；每层用完即释放 |
| `PackedArtifact` | export | 发布前 | 四件套齐全，config group 精确 |
| `Manifest` | publish | 持久 | 源 hash、工具版本、方案、耗时、质量门 |

## 设备与内存纪律

- P0a/P0b：**单卡 24G**；模型可整卡驻留，但 AWQ 校准缓存必须按层释放。
- P1：双卡 24G；FP8 源**逐层解码**，同一时刻只保留当前层 dense 权重；
  禁止把 54GB BF16 落到磁盘。
- 任何阶段失败：保留 `.work/`，不发布半成品；输出目录用临时名 + rename。

## 为什么用 llm-compressor 作为主后端

- 0.8B / 4B 的 Qwen3.5 GDN AWQ 实测质量通过（输出 391）；
- 原生输出 compressed-tensors `pack-quantized`，与 cyankiwi 布局一致；
- `pipeline="sequential"` + `sequential_offload_device="cuda"` 在 24G 内
  可跑 4B（4m25s / 15.25 GiB）；
- `n_grid` / `batch_size` / `pipeline` / `sequential_targets` 可调。

GPTQModel 7.7 虽然 `true_sequential` + native FP8 设计很合适，但 0.8B
AWQ 质量失败（1723/全 0），已降级为 27B FP8 module-local decode 候选。

## 不做什么

- 不把 GPTQModel 的模型树复制进本项目；只做适配层和导出层。
- 不手写 AWQ 公式；先对拍，只有热点才下沉 C++/CUDA。
- 不把 `compressed-tensors` 的 pack 逻辑重写；调用官方 `pack_to_int32`。
