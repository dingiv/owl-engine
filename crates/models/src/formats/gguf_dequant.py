#!/usr/bin/env python3
"""GGUF 张量 → f32(**对拍锚的另一端**;Rust 侧 = `formats/gguf/dequant.rs::tests`)。

用法:
    PYTHONPATH=<llama.cpp>/gguf-py uv run python src/formats/gguf_dequant.py <file.gguf> <tensor> <out.bin>

输出 = f32 LE 裸字节(行主序 = gguf-py 的 numpy 形状,**未**翻转),
即 `gguf.quants.dequantize(t.data, t.tensor_type)` 的 `.astype(np.float32).tobytes()`。

为什么用 gguf-py 当锚:它与我们的 Rust 端口是**同一规格的两个独立实现**
(numpy 位运算 vs C 循环 port)⇒ 一致 = 规格无歧义。
"""

import sys

import numpy as np

# PYTHONPATH 由调用方给
from gguf import GGUFReader
from gguf.quants import dequantize


def main() -> None:
    gguf_path, tensor_name, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    r = GGUFReader(gguf_path)
    hits = [t for t in r.tensors if t.name == tensor_name]
    if not hits:
        raise SystemExit(f"张量不存在: {tensor_name}")
    t = hits[0]
    y = dequantize(t.data, t.tensor_type).astype(np.float32)
    with open(out_path, "wb") as f:
        f.write(y.tobytes())
    print(f"{tensor_name}\t{t.tensor_type.name}\t{y.shape}\t{len(y.ravel())}", file=sys.stderr)


if __name__ == "__main__":
    main()
