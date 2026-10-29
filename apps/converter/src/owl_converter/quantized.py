"""已量化检查点的方案校验与 owl 归一化。

`model_free_ptq` 产出的 config group 名可能是 `W4A16`，而 owl 的
`quant_family` 只读 `config_groups.group_0.weights` —— 这里做**唯一**的
改名/校验入口，不改张量。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConverterError
from .st_index import CheckpointIndex, load_checkpoint

OWL_SCHEMES = {
    (4, 32, False): "awq-g32",
    (4, 128, True): "w4a16-g128",
}


@dataclass(frozen=True)
class QuantGroup:
    name: str
    num_bits: int
    group_size: int
    symmetric: bool
    strategy: str
    fmt: str | None

    @property
    def owl_scheme(self) -> str | None:
        return OWL_SCHEMES.get((self.num_bits, self.group_size, self.symmetric))


@dataclass
class QuantizedInfo:
    quant_method: str | None
    fmt: str | None
    groups: dict[str, QuantGroup]

    @property
    def group_name(self) -> str | None:
        if "group_0" in self.groups:
            return "group_0"
        if len(self.groups) == 1:
            return next(iter(self.groups))
        return None

    @property
    def group(self) -> QuantGroup | None:
        name = self.group_name
        return self.groups.get(name) if name else None


def _group_from(name: str, raw: Any) -> QuantGroup:
    if not isinstance(raw, dict):
        raise ConverterError("bad-quant-config", f"config_groups.{name} 不是 object")
    weights = raw.get("weights")
    if not isinstance(weights, dict):
        raise ConverterError(
            "bad-quant-config",
            f"config_groups.{name} 缺 weights",
            hint="不是 compressed-tensors 的 pack-quantized config",
        )
    try:
        num_bits = int(weights["num_bits"])
        group_size = int(weights["group_size"])
        symmetric = bool(weights["symmetric"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ConverterError(
            "bad-quant-config",
            f"config_groups.{name}.weights 缺 num_bits/group_size/symmetric：{exc}",
        ) from exc
    return QuantGroup(
        name=name,
        num_bits=num_bits,
        group_size=group_size,
        symmetric=symmetric,
        strategy=str(weights.get("strategy", "")),
        fmt=raw.get("format"),
    )


def read_info(config: dict[str, Any]) -> QuantizedInfo | None:
    qc = config.get("quantization_config")
    if not isinstance(qc, dict):
        return None
    raw_groups = qc.get("config_groups")
    if not isinstance(raw_groups, dict) or not raw_groups:
        return None
    groups = {name: _group_from(name, raw) for name, raw in raw_groups.items()}
    return QuantizedInfo(
        quant_method=qc.get("quant_method"),
        fmt=qc.get("format"),
        groups=groups,
    )


def _expected_scale_shape(out_dim: int, in_dim: int, group_size: int) -> tuple[int, int]:
    if in_dim % group_size != 0:
        raise ConverterError(
            "bad-group",
            f"in_dim={in_dim} 不能被 group_size={group_size} 整除",
            hint="检查量化工具写出的 group_size；owl 只认整除的组",
        )
    return (out_dim, in_dim // group_size)


def check_layout(index: CheckpointIndex, group: QuantGroup) -> tuple[list[str], dict[str, int]]:
    """返回 (problems, counts)。只读索引，不读数据区。"""
    problems: list[str] = []
    packed = {n for n in index.tensors if n.endswith(".weight_packed")}
    scales = {n for n in index.tensors if n.endswith(".weight_scale")}
    zps = {n for n in index.tensors if n.endswith(".weight_zero_point")}
    shapes = {n for n in index.tensors if n.endswith(".weight_shape")}

    for name in sorted(packed):
        base = name.removesuffix(".weight_packed")
        packed_entry = index.tensors[name][0]
        if packed_entry.dtype != "I32":
            problems.append(f"{name}: weight_packed dtype={packed_entry.dtype}（owl 要 I32）")
        if packed_entry.rank != 2:
            problems.append(f"{name}: weight_packed rank={packed_entry.rank}（要 2D）")
            continue
        out_dim, words = packed_entry.shape
        in_dim = words * 8
        scale_name = f"{base}.weight_scale"
        if scale_name not in index.tensors:
            problems.append(f"{name}: 缺 {scale_name}")
            continue
        scale_entry = index.tensors[scale_name][0]
        if scale_entry.dtype != "BF16":
            problems.append(f"{scale_name}: dtype={scale_entry.dtype}（owl 要 BF16）")
        want = _expected_scale_shape(out_dim, in_dim, group.group_size)
        if scale_entry.shape != want:
            problems.append(
                f"{scale_name}: shape={scale_entry.shape}，期望 {want}（group={group.group_size}）"
            )
        shape_name = f"{base}.weight_shape"
        if shape_name in index.tensors:
            shape_entry = index.tensors[shape_name][0]
            if shape_entry.shape != (2,):
                problems.append(
                    f"{shape_name}: shape={shape_entry.shape}，期望 (2,)（[out,in] 两个 I64）"
                )
            if shape_entry.dtype != "I64":
                problems.append(
                    f"{shape_name}: dtype={shape_entry.dtype}（owl 不读它，但格式应为 I64）"
                )
        if out_dim % 256 != 0 or in_dim % 128 != 0:
            # 不是错误：owl 对这些层回退 F16，W4A16Source 会反量化；
            # 但要让读者知道“它们不会走 marlin”。
            problems.append(
                f"{name}: {out_dim}x{in_dim} 不满足 marlin_eligible（owl 会回退 F16）"
            )

    if not group.symmetric:
        if not zps:
            problems.append("symmetric=false 但没有任何 weight_zero_point")
        for name in sorted(packed):
            base = name.removesuffix(".weight_packed")
            zp_name = f"{base}.weight_zero_point"
            if zp_name not in index.tensors:
                problems.append(f"{name}: 缺 {zp_name}（asym 必须带 zp）")
    else:
        if zps:
            problems.append("symmetric=true 但存在 weight_zero_point（owl 会忽略，建议清掉）")

    for name in sorted(scales - {p.removesuffix(".weight_packed") + ".weight_scale" for p in packed}):
        problems.append(f"{name}: 孤儿 weight_scale（没有对应 weight_packed）")
    for name in sorted(shapes - {p.removesuffix(".weight_packed") + ".weight_shape" for p in packed}):
        problems.append(f"{name}: 孤儿 weight_shape")

    def total_bytes(names: set[str]) -> int:
        return sum(index.tensors[n][0].nbytes for n in names)

    counts = {
        "packed_layers": len(packed),
        "scale_layers": len(scales),
        "zero_point_layers": len(zps),
        "shape_layers": len(shapes),
        "packed_bytes": total_bytes(packed),
        "scale_bytes": total_bytes(scales),
        "zero_point_bytes": total_bytes(zps),
        "shape_bytes": total_bytes(shapes),
    }
    return problems, counts


def _config_path(root: Path) -> Path:
    return root / "config.json" if root.is_dir() else root.parent / "config.json"


def _write_config_atomic(path: Path, config: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def normalize(path: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """把 config_groups 收敛到 owl 可读的 `group_0`，并做结构校验。"""
    index = load_checkpoint(path)
    cfg_path = _config_path(index.root)
    if not cfg_path.is_file():
        raise ConverterError("no-config", f"{cfg_path} 不存在")
    try:
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConverterError("bad-config", f"读 {cfg_path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ConverterError("bad-config", f"{cfg_path} 不是 JSON object")

    info = read_info(config)
    if info is None:
        raise ConverterError(
            "not-quantized",
            "config.json 没有 quantization_config.config_groups",
            hint="这不是 compressed-tensors pack-quantized 检查点；先跑 quantize",
        )
    group = info.group
    if group is None:
        raise ConverterError(
            "ambiguous-groups",
            f"config_groups={sorted(info.groups)}；owl 只读 group_0，且无法自动选定",
            hint="只保留一个 group，或手工把它命名为 group_0",
        )
    scheme = group.owl_scheme
    if scheme is None:
        raise ConverterError(
            "unsupported-scheme",
            f"方案 (num_bits={group.num_bits}, group_size={group.group_size}, "
            f"symmetric={group.symmetric}) 不在 owl 两格内",
            hint="owl 只接受 (4,32,false) 与 (4,128,true)；改量化方案后重转",
        )

    problems, counts = check_layout(index, group)
    hard = [p for p in problems if "回退 F16" not in p]
    if hard:
        raise ConverterError(
            "bad-layout",
            "键面/形状校验失败：\n  " + "\n  ".join(hard[:8]),
            hint="不要手工修张量；回到 quantize 阶段修 ignore/方案",
        )

    changed = False
    if group.name != "group_0":
        qc = config["quantization_config"]
        raw_groups = qc["config_groups"]
        # 保留原 group 的全部字段（input_activations/format/...），仅换键名
        qc["config_groups"] = {"group_0": raw_groups[group.name]}
        changed = True
        if not dry_run:
            _write_config_atomic(cfg_path, config)

    return {
        "path": str(index.root),
        "changed": changed,
        "dry_run": dry_run,
        "group_from": group.name,
        "group_to": "group_0",
        "scheme": scheme,
        "quant_method": info.quant_method,
        "format": info.fmt,
        "counts": counts,
        "warnings": [p for p in problems if "回退 F16" in p][:20],
    }
