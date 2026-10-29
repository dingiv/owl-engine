"""源精度/scale 形态识别（只读索引 + config.json）。

P0 支持：
- dense BF16/F16/F32（BF16 是 AWQ 的标准输入，也是本次测试模型的形态）；
- FP8 per-tensor / per-channel / blockwise(128×128)；
- 已经是 4bit 的包（weight_packed/qweight）—— 报告“无需转换”。

不猜：FNUZ、缺失 scale、未知 scale 形状一律结构化拒绝。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .dtypes import DENSE_DTYPES, FP8_DTYPES, UNSUPPORTED_FP8_DTYPES
from .errors import ConverterError
from .st_index import CheckpointIndex, TensorEntry

SCALE_SUFFIXES = ("weight_scale", "weight_scale_inv")


@dataclass(frozen=True)
class ScalePair:
    weight: str
    scale: str
    kind: str  # per-tensor | per-channel | blockwise
    inverse: bool


@dataclass
class SourceProfile:
    scheme: str
    dtypes: dict[str, int]
    scale_pairs: list[ScalePair] = field(default_factory=list)
    orphan_scales: list[str] = field(default_factory=list)
    block_size: tuple[int, int] | None = None
    quantization_config: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def is_fp8(self) -> bool:
        return self.scheme.startswith("fp8-")


def _quant_config(config: dict[str, Any]) -> dict[str, Any] | None:
    qc = config.get("quantization_config")
    if isinstance(qc, dict):
        return qc
    text = config.get("text_config")
    if isinstance(text, dict) and isinstance(text.get("quantization_config"), dict):
        return text["quantization_config"]
    return None


def _block_size(qc: dict[str, Any] | None) -> tuple[int, int]:
    if qc:
        raw = qc.get("weight_block_size")
        if (
            isinstance(raw, (list, tuple))
            and len(raw) == 2
            and all(isinstance(x, int) and x > 0 for x in raw)
        ):
            return (int(raw[0]), int(raw[1]))
    return (128, 128)


def _classify_scale(
    tensor: TensorEntry,
    out_dim: int,
    in_dim: int,
    block: tuple[int, int],
) -> str | None:
    shape = tensor.shape
    if tensor.numel == 1:
        return "per-tensor"
    if len(shape) == 1 and shape[0] == out_dim:
        return "per-channel"
    if len(shape) == 2 and shape == (out_dim, 1):
        return "per-channel"
    want = (math.ceil(out_dim / block[0]), math.ceil(in_dim / block[1]))
    if len(shape) == 2 and shape == want:
        return "blockwise"
    return None


def detect_source(index: CheckpointIndex, config: dict[str, Any]) -> SourceProfile:
    dtypes = index.dtype_counts()

    bad = [name for name, (t, _) in index.tensors.items() if t.dtype in UNSUPPORTED_FP8_DTYPES]
    if bad:
        raise ConverterError(
            "unsupported-fp8",
            f"检查点含 fnuz 版 FP8（{len(bad)} 张，首例 {bad[0]}）",
            hint="owl 只认 F8_E4M3/F8_E5M2 的 fn 变体；fnuz 必须先换源或用其它工具重存",
        )

    qc = _quant_config(config)
    block = _block_size(qc)
    has_packed = any(
        name.endswith((".weight_packed", ".qweight")) for name in index.tensors
    )

    fp8_weights = [
        (name, entry)
        for name, (entry, _) in index.tensors.items()
        if entry.dtype in FP8_DTYPES and name.endswith(".weight")
    ]

    if not fp8_weights:
        if has_packed:
            return SourceProfile(
                scheme="pre-quantized-4bit",
                dtypes=dtypes,
                quantization_config=qc,
                notes=["检查点已含 weight_packed/qweight；本转换器不需要再跑"],
            )
        dense = {d for d in dtypes if d in DENSE_DTYPES}
        if not dense:
            raise ConverterError(
                "unsupported-dtype",
                f"没有 FP8/dense 权重；dtype 分布 = {dtypes}",
                hint="P0 只接 BF16/F16/F32 与 F8_E4M3/F8_E5M2",
            )
        if len(dense) == 1:
            only = next(iter(dense))
            scheme = {"BF16": "dense-bf16", "F16": "dense-f16", "F32": "dense-f32"}[only]
        else:
            scheme = "dense-mixed"
        notes = []
        if qc:
            notes.append("config 有 quantization_config，但索引里没有 FP8 权重；按 dense 源处理")
        return SourceProfile(
            scheme=scheme, dtypes=dtypes, quantization_config=qc, notes=notes
        )

    pairs: list[ScalePair] = []
    missing: list[str] = []
    unknown: list[str] = []
    for name, weight in fp8_weights:
        if weight.rank != 2:
            raise ConverterError(
                "fp8-rank",
                f"{name} 是 rank-{weight.rank}，FP8 权重只支持 2D Linear",
            )
        out_dim, in_dim = weight.shape
        base = name.removesuffix(".weight")
        scale_name = None
        for suffix in SCALE_SUFFIXES:
            cand = f"{base}.{suffix}"
            if cand in index.tensors:
                scale_name = cand
                break
        if scale_name is None:
            missing.append(name)
            continue
        scale = index.tensors[scale_name][0]
        kind = _classify_scale(scale, out_dim, in_dim, block)
        if kind is None:
            unknown.append(f"{scale_name} shape={scale.shape} (weight {out_dim}x{in_dim})")
            continue
        pairs.append(
            ScalePair(
                weight=name,
                scale=scale_name,
                kind=kind,
                inverse=scale_name.endswith("_inv"),
            )
        )

    if missing:
        raise ConverterError(
            "missing-scale",
            f"{len(missing)} 个 FP8 权重缺 weight_scale/weight_scale_inv（首例 {missing[0]}）",
            hint="检查是否漏下 scale 张量；FP8 无 scale 时数值不可复原，禁止按 1.0 猜",
        )
    if unknown:
        raise ConverterError(
            "unknown-scale-shape",
            "无法识别的 FP8 scale 形状：\n  " + "\n  ".join(unknown[:5]),
            hint=f"当前只认 per-tensor / per-channel / blockwise{block}",
        )

    kinds = {p.kind for p in pairs}
    if len(kinds) == 1:
        kind = next(iter(kinds))
        scheme = {
            "per-tensor": "fp8-per-tensor",
            "per-channel": "fp8-per-channel",
            "blockwise": "fp8-blockwise",
        }[kind]
    else:
        scheme = "fp8-mixed"

    weights = set(index.tensors)
    orphan = [
        name
        for name in index.tensors
        if name.endswith((".weight_scale", ".weight_scale_inv"))
        and name.removesuffix(".weight_scale").removesuffix(".weight_scale_inv")
        + ".weight"
        not in weights
    ]

    notes = []
    if scheme == "fp8-mixed":
        notes.append("同时存在多种 scale 形态；P1 只处理单一形态（建议先统一或分组转换）")
    if any(p.inverse for p in pairs):
        notes.append("weight_scale_inv 按“除以 inv”还原（DeepSeek 系语义）")
    if any(p.kind == "blockwise" for p in pairs):
        notes.append(f"blockwise 使用 weight_block_size={block}")
    notes.append("input_scale/activation scale 不参与权重转换（输出激活为 F16）")

    return SourceProfile(
        scheme=scheme,
        dtypes=dtypes,
        scale_pairs=pairs,
        orphan_scales=orphan,
        block_size=block if any(p.kind == "blockwise" for p in pairs) else None,
        quantization_config=qc,
        notes=notes,
    )
