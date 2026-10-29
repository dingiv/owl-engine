# owl/tools

| 路径 | 用途 |
|---|---|
| `awq-forge/` | **独立 uv 项目**：FP8/BF16 → AWQ INT4（compressed-tensors pack-quantized，g32-asym）。不参与 owl 构建。 |
| `fixture/` | owl 的可复现测量/金标/探针脚本（Python / shell / CUDA）。 |

## awq-forge

```bash
cd tools/awq-forge
uv sync --extra quant --group dev
uv run awq-forge inspect /path/to/model
```

施工图与调参实录在 `tools/awq-forge/docs/`。

## fixture

- 金标/质量：`vision_golden.py`、`vlm_e2e_golden.py`、`quality_eval.py`、`needle_eval.py`
- 性能/分账：`ctxladder.py`、`prefill-account.sh`、`nsys-decode-account.sh`、`tp-ab-std.sh`
- KVMem：`kvmem_*gate.py`、`kvmem_repro_sparse_prefill.sh`
- GPU 探针：`*.cu`（AR / P2P / VMM / GDN / NCCL）
- 启动/跑批：`serve-w4a16-3090.sh`、`run-agent-workload.sh`、`run-*-dflash.sh`

脚本里的 `ROOT` 已按 `tools/fixture/` 上移两级修正；文档引用已同步为
`tools/fixture/<name>`。
