#!/usr/bin/env python3
"""GGUF 容器事实导出(**对拍协议的另一端**;Rust 侧 = `formats/gguf/reader.rs::tests`)。

用法(PYTHONPATH 指向 llama.cpp 的 `gguf-py`):
    PYTHONPATH=<llama.cpp>/gguf-py uv run python src/formats/gguf_dump.py <file.gguf>

输出 = TSV 三行式(每行一种事实;字段用 `\\t` 分隔):

    version \\t <u32>
    kv      \\t <key> \\t <TYPENAME> \\t <len> \\t <fnv64hex>
    tensor  \\t <name> \\t <TYPENAME> \\t <ne csv> \\t <offset> \\t <nbytes>

## canonical 编码(两侧必须逐字节一致;见 Rust 侧 `canonical()`)

标量 → 定长 8 字节 LE:

    UINT8/INT8/UINT16/INT16/UINT32/INT32/UINT64/INT64 → 二补码 u64 LE
    BOOL                                             → 0/1 的 u64 LE
    FLOAT32                                          → f32 位型的 u64 LE(零扩展)
    FLOAT64                                          → f64 位型的 u64 LE
    STRING(标量)                                     → utf-8 裸字节(len = 字节数)

ARRAY → 元素编码**直接拼接**(无分隔),`len` = 元素数:

    字符串元素 = u64 LE 字节长度 + utf-8 裸字节
    其余元素   = 同上标量编码

`fnv64hex` = FNV-1a 64(offset basis 0xcbf29ce484222325,prime 0x100000001b3)。
"""

import struct
import sys

import numpy as np

from gguf import GGUFReader
from gguf.constants import GGUFValueType

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
MASK64 = 0xFFFFFFFFFFFFFFFF

TYNAME = {
    GGUFValueType.UINT8: "UINT8",
    GGUFValueType.INT8: "INT8",
    GGUFValueType.UINT16: "UINT16",
    GGUFValueType.INT16: "INT16",
    GGUFValueType.UINT32: "UINT32",
    GGUFValueType.INT32: "INT32",
    GGUFValueType.FLOAT32: "FLOAT32",
    GGUFValueType.BOOL: "BOOL",
    GGUFValueType.STRING: "STRING",
    GGUFValueType.ARRAY: "ARRAY",
    GGUFValueType.UINT64: "UINT64",
    GGUFValueType.INT64: "INT64",
    GGUFValueType.FLOAT64: "FLOAT64",
}

INT_TYPES = (
    GGUFValueType.UINT8,
    GGUFValueType.INT8,
    GGUFValueType.UINT16,
    GGUFValueType.INT16,
    GGUFValueType.UINT32,
    GGUFValueType.INT32,
    GGUFValueType.UINT64,
    GGUFValueType.INT64,
    GGUFValueType.BOOL,
)


def fnv1a64(buf: bytes) -> int:
    h = FNV_OFFSET
    for b in buf:
        h = ((h ^ b) * FNV_PRIME) & MASK64
    return h


def enc_scalar(t: int, v) -> bytes:
    """标量 → 8 字节 LE(整数取二补码位型;浮点取 IEEE 位型)"""
    if t == GGUFValueType.FLOAT32:
        return struct.pack("<Q", struct.unpack("<I", np.float32(v).tobytes())[0])
    if t == GGUFValueType.FLOAT64:
        return struct.pack("<Q", struct.unpack("<Q", np.float64(v).tobytes())[0])
    if t == GGUFValueType.BOOL:
        return struct.pack("<Q", 1 if v else 0)
    return struct.pack("<Q", int(v) & MASK64)


def enc_value(t: int, v) -> bytes:
    """KV 值 → canonical 字节"""
    if t == GGUFValueType.STRING:
        return v.encode("utf-8")
    return enc_scalar(t, v)


def enc_array(et: int, arr) -> bytes:
    """ARRAY → 元素编码拼接(字符串元素带 u64 长度前缀)"""
    if et == GGUFValueType.STRING:
        parts = []
        for s in arr:
            b = s.encode("utf-8")
            parts.append(struct.pack("<Q", len(b)))
            parts.append(b)
        return b"".join(parts)
    if et in INT_TYPES:
        # numpy 向量化(整数/布尔 → u64 LE,与 enc_scalar 同字节)
        a = np.asarray(arr)
        if not np.issubdtype(a.dtype, np.bool_):
            a = a.astype(np.int64 if np.issubdtype(a.dtype, np.signedinteger) else np.uint64)
        return a.astype(np.uint64).tobytes()
    if et == GGUFValueType.FLOAT32:
        return np.asarray(arr, dtype=np.float32).view(np.uint32).astype(np.uint64).tobytes()
    if et == GGUFValueType.FLOAT64:
        return np.asarray(arr, dtype=np.float64).view(np.uint64).tobytes()
    raise SystemExit(f"未处理的数组元素类型 {et}")


def main() -> None:
    path = sys.argv[1]
    r = GGUFReader(path)
    # 版本住字段表(`GGUF.version`),不是属性
    out = [f"version\t{int(r.fields['GGUF.version'].contents())}"]
    for name, field in r.fields.items():
        if name.startswith("GGUF."):
            continue
        types = field.types
        main_t = types[0]
        if main_t == GGUFValueType.ARRAY:
            et = types[-1]
            arr = field.contents()
            buf = enc_array(et, arr)
            length = len(arr)
        elif main_t == GGUFValueType.STRING:
            buf = enc_value(main_t, field.contents())
            length = len(buf)
        else:
            buf = enc_value(main_t, field.contents())
            length = 1
        out.append(f"kv\t{name}\t{TYNAME[main_t]}\t{length}\t{fnv1a64(buf):016x}")
    for t in r.tensors:
        ne = ",".join(str(int(d)) for d in t.shape)  # gguf-py 的 shape = ne 序(最内维在前)
        off = int(t.data_offset) - int(r.data_offset)  # 相对数据区(与 Rust 侧同口径)
        out.append(f"tensor\t{t.name}\t{t.tensor_type.name}\t{ne}\t{off}\t{int(t.n_bytes)}")
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
