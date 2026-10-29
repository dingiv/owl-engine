"""inspect 计划：把索引/源形态/清单/空间检查收成一份可审计的 plan。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from . import __version__
from .errors import ConverterError
from .inventory import SCHEMES, build_inventory
from .quantized import check_layout, read_info
from .source_profile import detect_source
from .st_index import CheckpointIndex, load_checkpoint


def format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{n} B"


def read_config(root: Path) -> tuple[dict[str, Any], str | None]:
    """返回 (config, warning)。缺 config.json 不致命（单文件/GGUF 场景），
    但 inspect 的输出会显式记一条 warning。"""
    path = root / "config.json" if root.is_dir() else root.parent / "config.json"
    if not path.is_file():
        return {}, f"未找到 {path.name}；按默认结构继续"
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"{path.name} 读取/解析失败：{exc}；按默认结构继续"
    if not isinstance(raw, dict):
        return {}, f"{path.name} 不是 JSON object；按默认结构继续"
    return raw, None


def _config_summary(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    text = text if isinstance(text, dict) else {}
    return {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "dtype": text.get("dtype") or config.get("torch_dtype"),
        "tie_word_embeddings": config.get("tie_word_embeddings"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "hidden_size": text.get("hidden_size"),
        "quantization_config": config.get("quantization_config"),
    }


def _check(level: str, code: str, message: str, hint: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"level": level, "code": code, "message": message}
    if hint:
        item["hint"] = hint
    return item


def _disk_check(
    source: Path, needed: int, checks: list[dict[str, Any]], require_disk: bool
) -> None:
    probe = source if source.is_dir() else source.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        checks.append(_check("warn", "disk-unknown", f"无法读取磁盘余量：{exc}"))
        return
    required = int(needed * 1.1) + (1 << 30)  # 10% 余量 + 1GiB 系统余量
    if usage.free < required:
        level = "error" if require_disk else "warn"
        checks.append(
            _check(
                level,
                "insufficient-disk",
                f"预计需要 {format_bytes(required)}，可用 {format_bytes(usage.free)}",
                "清理 .work/ 或换输出目录；磁盘不足时不要开始长跑",
            )
        )
    else:
        checks.append(
            _check(
                "info",
                "disk-ok",
                f"预计需要 {format_bytes(required)}，可用 {format_bytes(usage.free)}",
            )
        )


def _next_steps(
    profile_scheme: str, incomplete: bool, scheme: str, *, normalize_needed: bool = False
) -> list[str]:
    if incomplete:
        return [
            "等待下载完成",
            "重跑：owl-convert inspect <src> --require-complete",
            f"然后：owl-convert quantize <src> <out> --scheme {scheme} --calib <calib>",
        ]
    if profile_scheme == "pre-quantized-4bit":
        if normalize_needed:
            return [
                "owl-convert normalize <src>",
                "然后按 owl 的 W4A16/AWQ 档起服务（--set model.dir=<src>）",
            ]
        return ["检查点已是 4bit；直接按 owl 的 W4A16/AWQ 档起服务"]
    if profile_scheme.startswith("fp8-"):
        return [
            f"P1：owl-convert dequant <src> .work/dense --scheme {scheme}",
            f"P2：owl-convert quantize .work/dense <out> --scheme {scheme} --calib <calib>",
        ]
    return [
        f"P2：owl-convert quantize <src> <out> --scheme {scheme} --calib <calib>",
    ]


def build_plan(
    source: Path,
    scheme: str,
    *,
    require_complete: bool = False,
    require_disk: bool = False,
) -> dict[str, Any]:
    if scheme not in SCHEMES:
        raise ConverterError(
            "bad-scheme",
            f"未知输出方案 {scheme}",
            hint="P0 支持：" + " / ".join(SCHEMES),
        )
    index: CheckpointIndex = load_checkpoint(source)
    config, config_warning = read_config(index.root)
    profile = detect_source(index, config)
    inventory = build_inventory(index, scheme)

    checks: list[dict[str, Any]] = []
    quantized: dict[str, Any] | None = None
    normalize_needed = False
    if profile.scheme == "pre-quantized-4bit":
        info = read_info(config)
        if info is None or info.group is None:
            checks.append(
                _check(
                    "error",
                    "no-quant-config",
                    "检查点含 weight_packed，但 config 没有可读的 config_groups",
                    "用 compressed-tensors 的量化产物；或先跑 normalize 前检查 config",
                )
            )
        else:
            group = info.group
            try:
                problems, counts = check_layout(index, group)
            except ConverterError as exc:
                problems, counts = [str(exc)], {}
            hard = [p for p in problems if "回退 F16" not in p]
            quantized = {
                "group_name": group.name,
                "scheme": group.owl_scheme,
                "num_bits": group.num_bits,
                "group_size": group.group_size,
                "symmetric": group.symmetric,
                "counts": counts,
                "problems": problems[:20],
            }
            if group.name != "group_0":
                normalize_needed = True
                checks.append(
                    _check(
                        "warn",
                        "quant-group-name",
                        f"config group = {group.name}；owl 的 quant_family 只读 group_0",
                        "owl-convert normalize <src>",
                    )
                )
            if group.owl_scheme is None:
                checks.append(
                    _check(
                        "error",
                        "unsupported-scheme",
                        f"方案 ({group.num_bits},{group.group_size},symmetric={group.symmetric}) 不在 owl 两格内",
                        "owl 只认 (4,32,false) / (4,128,true)",
                    )
                )
            for problem in hard[:8]:
                checks.append(_check("error", "quant-layout", problem))
    if config_warning:
        checks.append(_check("warn", "config-missing", config_warning))

    if index.problems:
        for problem in index.problems:
            checks.append(_check("warn", "index-problem", problem))

    incomplete = not index.complete
    if incomplete:
        missing = index.missing_bytes
        detail = (
            f"缺 {format_bytes(missing)}"
            if missing
            else "索引引用的分片未齐"
        )
        checks.append(
            _check(
                "error" if require_complete else "warn",
                "incomplete-download",
                f"检查点未下完（{detail}）",
                "等下载完成；inspect 只读了头部，未加载权重",
            )
        )

    for shard in index.shards:
        for warning in shard.warnings[:3]:
            checks.append(_check("info", "shape-byte-mismatch", f"{shard.path.name}: {warning}"))

    needed = inventory.dense_intermediate_bytes + inventory.final_bytes
    if profile.scheme.startswith("dense-") and inventory.dense_intermediate_bytes == (
        inventory.total_numel * 2
    ):
        # dense BF16/F16 源不需要额外 dense 中间态（F32 仍需落 BF16，按同式保守估）
        needed = inventory.final_bytes + inventory.source_bytes
    _disk_check(index.root, needed, checks, require_disk)

    has_error = any(c["level"] == "error" for c in checks)
    if has_error:
        status = "blocked"
    elif incomplete:
        status = "incomplete"
    else:
        status = "ok"

    skipped = [
        {
            "reason": reason,
            "count": count,
            "examples": inventory.skipped_examples.get(reason, []),
        }
        for reason, count in sorted(inventory.skipped_counts.items())
    ]

    plan = {
        "schema": "owl-convert-plan/1",
        "tool": {"name": "owl-convert", "version": __version__},
        "status": status,
        "source": {
            "path": str(index.root),
            "shards": len(index.shards),
            "files": [s.path.name for s in index.shards],
            "complete": index.complete,
            "missing_bytes": index.missing_bytes,
            "problems": index.problems,
            "tensors": len(index.tensors),
            "total_bytes": index.total_bytes,
            "dtypes": index.dtype_counts(),
            "config": _config_summary(config),
        },
        "scale": {
            "scheme": profile.scheme,
            "pairs": len(profile.scale_pairs),
            "kinds": sorted({p.kind for p in profile.scale_pairs}),
            "block_size": profile.block_size,
            "orphan_scales": profile.orphan_scales[:10],
            "notes": profile.notes,
        },
        "inventory": {
            "linear_total": len(inventory.candidates) + inventory.skipped_counts.get(
                "marlin-ineligible", 0
            ),
            "quantizable": len(inventory.candidates),
            "skipped": skipped,
            "eligible_numel": inventory.eligible_numel,
            "total_numel": inventory.total_numel,
            "source_bytes": inventory.source_bytes,
        },
        "output": {
            "scheme": scheme,
            "group_size": inventory.group,
            "symmetric": inventory.symmetric,
            "q4_bytes": inventory.q4_bytes,
            "keep_bytes": inventory.keep_bytes,
            "final_bytes": inventory.final_bytes,
            "dense_intermediate_bytes": inventory.dense_intermediate_bytes,
        },
        "checks": checks,
        "next": _next_steps(
            profile.scheme, incomplete, scheme, normalize_needed=normalize_needed
        ),
    }
    if quantized is not None:
        counts = quantized["counts"]
        q4 = (
            counts.get("packed_bytes", 0)
            + counts.get("scale_bytes", 0)
            + counts.get("zero_point_bytes", 0)
        )
        plan["inventory"].update(
            {
                "quantizable": counts.get("packed_layers", 0),
                "linear_total": counts.get("packed_layers", 0),
                "skipped": [],
            }
        )
        plan["output"].update(
            {
                "scheme": quantized["scheme"] or "unsupported",
                "group_size": quantized["group_size"],
                "symmetric": quantized["symmetric"],
                "q4_bytes": q4,
                "keep_bytes": max(0, index.total_bytes - q4),
                "final_bytes": index.total_bytes,
                "dense_intermediate_bytes": index.total_bytes,
            }
        )
        plan["quantized"] = quantized
    return plan


def write_plan(plan: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")


def human_summary(plan: dict[str, Any]) -> str:
    src = plan["source"]
    scale = plan["scale"]
    inv = plan["inventory"]
    out = plan["output"]
    lines = [
        f"source   : {src['path']}",
        f"kind     : {scale['scheme']}",
        (
            f"state    : {'complete' if src['complete'] else 'INCOMPLETE'}"
            + (
                f"（缺 {format_bytes(src['missing_bytes'])}）"
                if not src["complete"]
                else ""
            )
        ),
        f"tensors  : {src['tensors']} · {format_bytes(src['total_bytes'])}",
        f"dtypes   : {', '.join(f'{k}={v}' for k, v in src['dtypes'].items())}",
        (
            f"scale    : pairs={scale['pairs']} kinds={','.join(scale['kinds']) or '-'}"
            + (f" block={scale['block_size']}" if scale["block_size"] else "")
        ),
        (
            f"inventory: quantizable={inv['quantizable']} / linear_total={inv['linear_total']}"
            f" · skipped={sum(s['count'] for s in inv['skipped'])}"
        ),
        (
            f"estimate : {format_bytes(out['dense_intermediate_bytes'])} dense"
            f" → {format_bytes(out['final_bytes'])} 4bit"
            f"（q4 {format_bytes(out['q4_bytes'])} + keep {format_bytes(out['keep_bytes'])}）"
        ),
        f"output   : {out['scheme']}（group={out['group_size']}, symmetric={out['symmetric']}）",
    ]
    if "quantized" in plan:
        q = plan["quantized"]
        lines.insert(
            6,
            f"quantized: packed={q['counts'].get('packed_layers', 0)}"
            f" / scales={q['counts'].get('scale_layers', 0)}"
            f" / zp={q['counts'].get('zero_point_layers', 0)}"
            f" · group={q['group_name']} scheme={q['scheme']}",
        )
    if inv["skipped"]:
        lines.append("skipped  :")
        for item in inv["skipped"]:
            example = f"  e.g. {item['examples'][0]}" if item["examples"] else ""
            lines.append(f"  - {item['reason']}: {item['count']}{example}")
    if plan["checks"]:
        lines.append("checks   :")
        for c in plan["checks"]:
            lines.append(f"  - [{c['level']}] {c['code']}: {c['message']}")
            if c.get("hint"):
                lines.append(f"      hint: {c['hint']}")
    lines.append("next     :")
    for step in plan["next"]:
        lines.append(f"  - {step}")
    return "\n".join(lines)
