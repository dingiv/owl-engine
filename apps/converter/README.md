# owl-converter

离线 **FP8 / BF16 → 4bit** 转换器（owl 检查点）。

> 施工图：`docs/devlog/models/fp8-to-w4a16-converter.md`
> 当前阶段：**P0（inspect + normalize）已完成**；P1 RTN 格式链路已在 4B 模型上跑通；
> P2 AWQ g32 待做。默认 `uv sync` 不装 torch；重依赖在 `--extra quant`。

## 为什么独立

FP8 → INT4 是有损二次量化，必须离线 + 校准 + 质量门；运行时不改。
转换产物是 `compressed-tensors pack-quantized`，owl 的 `W4A16Source` /
`AwqSource` 直接装载。

## 安装（独立 venv，uv）

```bash
cd apps/converter
uv sync                 # 创建 .venv（py3.12）；P0 无运行时依赖
uv run owl-convert --help
```

重依赖（P1/P2 才需要）：

```bash
uv sync --extra quant     # torch / transformers / llmcompressor / compressed-tensors
```

开发：

```bash
uv sync --group dev
uv run pytest
```

## P0：inspect

```bash
uv run owl-convert inspect /path/to/models/Qwen3.8-4B-Distill
uv run owl-convert inspect /path/to/model --json --out plan.json
uv run owl-convert inspect /path/to/model --require-complete
```

输出：

- **源形态**：`dense-bf16` / `dense-f16` / `dense-f32` / `fp8-per-tensor` /
  `fp8-per-channel` / `fp8-blockwise` / `fp8-mixed` / `pre-quantized-4bit`；
- **完整性**：safetensors 头声明的数据区 vs 文件实际大小（下载中会报
  `INCOMPLETE` + 缺多少字节）；
- **清单**：可量化 Linear（owl `marlin_eligible` = out%256 / in%128）、
  被跳过的原因（embedding/head/vision/mtp/norm/ineligible…）；
- **体积**：dense 中间态、4bit 产物（packed + scales + zp）、保留件；
- **磁盘**：输出侧可用空间 vs 预估需求；
- **checks / next**：结构化 check + 下一步命令。

退出码：

| 码 | 含义 |
|---:|---|
| 0 | 检查完成（允许 warning / 下载中） |
| 2 | 输入/格式/方案不合法 |
| 3 | `--require-complete` 且检查点未下完 |
| 4 | `--require-disk` 且磁盘不足 |
| 64 | 子命令未实现（P1/P2） |

## P1：RTN 冒烟（dense BF16/F16）

```bash
uv run --extra quant python examples/rtn_w4a16.py \
  /path/to/model-bf16 /path/to/model-w4a16-rtn
uv run owl-convert normalize /path/to/model-w4a16-rtn
```

产物是 `(4,128,symmetric=true)`（W4A16 g128-sym），对应 owl
`[model] format = "pack-quantized"`、`quant = "w4a16"`。

## 目录

```
apps/converter/
  pyproject.toml / uv.lock
  src/owl_converter/
    cli.py            # inspect + P1/P2 占位
    st_index.py       # safetensors 头/分片/完整性
    source_profile.py # dense/FP8 + scale 形态识别
    inventory.py      # Linear 清单 + 4bit 体积
    plan.py           # plan JSON / 人读摘要 / 磁盘检查
  tests/
```

## 后续

| 阶段 | 内容 |
|---|---|
| P1 | **RTN g128 冒烟已完成**（`examples/rtn_w4a16.py`，4B BF16→4bit 端到端）；FP8→dense 的 `dequant` 待做 |
| P2 | `quantize`：AWQ g32-asym（默认，llm-compressor + 校准集）；verify/质量门 |
| P3 | GPTQ g128-sym A/B（可选） |

依赖（torch / compressed-tensors / llm-compressor）在 P2 才加入并锁进
`uv.lock`；P0/P1 不装。
