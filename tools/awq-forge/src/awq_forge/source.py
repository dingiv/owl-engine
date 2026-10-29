"""Read-only source checkpoint inspection.

Only the safetensors header and config.json are read. No torch, no GPU.
"""

from __future__ import annotations

import json
import os
import struct
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .inventory import Target, build_targets, classify_weight, ignored_reason, summarize_targets


@dataclass
class SourceInfo:
    source: str
    architecture: list[str] = field(default_factory=list)
    model_type: str | None = None
    quantization: dict[str, Any] | None = None
    files: list[str] = field(default_factory=list)
    tensor_count: int = 0
    dtypes: dict[str, int] = field(default_factory=dict)
    suffixes: dict[str, int] = field(default_factory=dict)
    target_count: int = 0
    target_roles: dict[str, int] = field(default_factory=dict)
    targets: list[dict[str, Any]] = field(default_factory=list)
    ignored: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_header(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                raise ValueError(f"{path}: truncated safetensors header length")
            (header_len,) = struct.unpack("<Q", raw_len)
            if header_len > path.stat().st_size:
                raise ValueError(f"{path}: header length exceeds file size")
            header = json.loads(fh.read(header_len))
    except OSError as exc:
        raise ValueError(f"{path}: cannot read safetensors header: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid safetensors header JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header is not an object")
    return header


def _index_files(source: Path) -> list[Path] | None:
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        return None
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{index_path}: invalid index: {exc}") from exc
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{index_path}: missing weight_map")
    shards = sorted({str(value) for value in weight_map.values()})
    return [source / shard for shard in shards]


def discover_files(source: Path) -> list[Path]:
    indexed = _index_files(source)
    if indexed is not None:
        return indexed
    files = sorted(source.glob("*.safetensors"))
    if not files:
        raise ValueError(f"{source}: no safetensors files found")
    return files


def _load_config(source: Path) -> dict[str, Any]:
    config_path = source / "config.json"
    if not config_path.exists():
        return {}
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{config_path}: invalid config: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path}: config is not an object")
    return payload


def inspect_source(raw_source: str | os.PathLike[str]) -> SourceInfo:
    source = Path(raw_source).expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"{source}: not a directory")

    config = _load_config(source)
    info = SourceInfo(source=str(source))
    info.architecture = list(config.get("architectures") or [])
    info.model_type = config.get("model_type")
    info.quantization = config.get("quantization_config")

    files = discover_files(source)
    info.files = [path.name for path in files]

    tensors: dict[str, tuple[int, ...]] = {}
    dtypes: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    for path in files:
        header = _read_header(path)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(meta, dict):
                continue
            shape = meta.get("shape")
            dtype = meta.get("dtype")
            if not isinstance(shape, list) or not isinstance(dtype, str):
                continue
            tensors[name] = tuple(int(v) for v in shape)
            dtypes[dtype] += 1
            suffixes[name.rsplit(".", 1)[-1]] += 1

    info.tensor_count = len(tensors)
    info.dtypes = dict(sorted(dtypes.items()))
    info.suffixes = dict(suffixes.most_common(16))

    targets = build_targets(tensors)
    # Packed AWQ/compressed-tensors checkpoints have no `.weight`; recover the
    # logical Linear targets from weight_packed + weight_scale shapes.
    if not targets:
        packed = []
        for name, shape in tensors.items():
            if not name.endswith(".weight_packed"):
                continue
            base = name.removesuffix(".weight_packed")
            scale = tensors.get(f"{base}.weight_scale")
            if not scale or len(scale) != 2:
                continue
            role = classify_weight(f"{base}.weight")
            if role is None:
                continue
            packed.append(Target(f"{base}.weight", role, int(scale[0]), int(scale[1]) * 32))
        targets = sorted(packed, key=lambda t: t.name)
    info.target_count = len(targets)
    info.target_roles = summarize_targets(targets)
    info.targets = [
        {
            "name": target.name,
            "role": target.role,
            "shape": [target.out_features, target.in_features],
            "packed_shape": list(target.packed_shape),
            "scale_shape": list(target.scale_shape),
            "zero_point_shape": list(target.zero_point_shape),
        }
        for target in targets
    ]

    ignored: Counter[str] = Counter()
    for name in tensors:
        reason = ignored_reason(name)
        if reason is not None:
            ignored[reason] += 1
        elif classify_weight(name) is None and name.endswith(".weight"):
            ignored["unclassified-weight"] += 1
    info.ignored = dict(sorted(ignored.items()))

    if info.quantization:
        quant_method = info.quantization.get("quant_method")
        if quant_method not in {"fp8", "compressed-tensors"}:
            info.warnings.append(f"unsupported quantization_config.quant_method={quant_method!r}")
    if not info.targets:
        info.warnings.append("no AWQ targets found; check model_type / tensor names")

    return info
