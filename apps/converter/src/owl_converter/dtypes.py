"""safetensors dtype 词表（只收检查点里会出现的名字）。"""

from __future__ import annotations

DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

FP8_DTYPES = frozenset({"F8_E4M3", "F8_E5M2"})
# owl 明确不认的 fnuz 变体（见 formats/source.rs：见到就拒，不冒充 fn）
UNSUPPORTED_FP8_DTYPES = frozenset({"F8_E4M3FNUZ", "F8_E5M2FNUZ"})
DENSE_DTYPES = frozenset({"F16", "BF16", "F32"})
