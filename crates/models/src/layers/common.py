"""HF baseline 复用框架:黄金参考脚本的进程协议 + 公共件。

就近约定(2026-09-26):每个 `layers/<name>.rs` 旁放一个 `<name>.py`
(同算法的金标准语义),Rust 层测试从旁 fork 本文件协议执行。

协议(进程边界;safetensors 交换):
    uv run python src/layers/<layer>.py <in.safetensors> <out.safetensors> [契约json]

脚本只写语义层 —— 定义 `test(tensors, contract) -> (outputs, provenance)`,
然后 `run(test)`。框架负责:argv/契约解析、f32 归一、HF 类 import
(provenance 记录)、输出落盘、manifest 打印。
"""

import json
import sys

import torch
from safetensors.torch import load_file, save_file


def hf_import(candidates: list[tuple[str, str]]) -> tuple[type | None, str]:
    """按候选序 import HF 类;返回 (类, provenance)。全失败 → (None, 标注)。"""
    for mod, name in candidates:
        try:
            m = __import__(mod, fromlist=[name])
            return getattr(m, name), f"{mod}.{name}"
        except Exception:
            continue
    return None, "manual(HF import 候选全失败:" + ",".join(n for _, n in candidates) + ")"


def run(test) -> None:
    """标准进程协议入口(脚本 main 只剩这一行)。

    test(tensors: dict[str, Tensor], contract: dict)
        -> (outputs: dict[str, Tensor], provenance: str)
    """
    inp, out = sys.argv[1], sys.argv[2]
    contract = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    tensors = {k: v.to(torch.float32) for k, v in load_file(inp).items()}
    outputs, provenance = test(tensors, contract)
    save_file({k: v.contiguous() for k, v in outputs.items()}, out)
    print(json.dumps({"provenance": provenance, **contract}))
