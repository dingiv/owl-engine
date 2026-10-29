"""测试夹具：手写 safetensors 头（P0 只需要头，不需要真实张量）。"""

from __future__ import annotations

import json
import struct
from pathlib import Path

from owl_converter.dtypes import DTYPE_BYTES


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...]]],
    *,
    truncate_data: int = 0,
    metadata: dict[str, str] | None = None,
) -> None:
    """tensors: name -> (dtype, shape)。数据区写零；truncate_data 模拟下载中。"""
    header: dict[str, object] = {"__metadata__": metadata or {"format": "pt"}}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        nbytes = numel * DTYPE_BYTES[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    data_len = max(0, offset - truncate_data)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * data_len)


def write_config(path: Path, config: dict[str, object]) -> None:
    path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
