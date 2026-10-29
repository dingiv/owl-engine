"""owl-convert CLI（P0：inspect + normalize；其余子命令是 P1/P2 的占位）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .errors import ConverterError
from .inventory import SCHEMES
from .plan import build_plan, human_summary, write_plan
from .quantized import normalize as normalize_checkpoint

STUBS = {
    "dequant": "P1：FP8/BF16 → dense BF16（compressed-tensors / 本地插件）",
    "quantize": "P2：dense → AWQ g32 / GPTQ g128 / RTN",
    "verify": "P2：结构门 + 数值门 + owl 装载冒烟 + 质量门",
    "run": "P1/P2：inspect → dequant → quantize → normalize → verify → package",
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="owl-convert",
        description="Offline FP8/BF16 → 4bit converter for owl checkpoints",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    ins = sub.add_parser(
        "inspect",
        help="只读检查点：dtype/scale 形态、完整性、Linear 清单、体积估算",
    )
    ins.add_argument("path", type=Path, help="模型目录或单个 .safetensors")
    ins.add_argument("--json", action="store_true", help="输出完整 plan JSON")
    ins.add_argument("--out", type=Path, help="把 plan JSON 写到文件")
    ins.add_argument(
        "--scheme",
        choices=sorted(SCHEMES),
        default="awq-g32",
        help="预估的目标方案（默认 awq-g32）",
    )
    ins.add_argument(
        "--require-complete",
        action="store_true",
        help="检查点未下完时以退出码 3 失败（默认只警告）",
    )
    ins.add_argument(
        "--require-disk",
        action="store_true",
        help="磁盘不足时以退出码 4 失败（默认只警告）",
    )

    norm = sub.add_parser(
        "normalize",
        help="把 config_groups 收敛到 owl 的 group_0，并校验 packed/scale/zp 键面",
    )
    norm.add_argument("path", type=Path, help="量化后的模型目录")
    norm.add_argument("--dry-run", action="store_true", help="只校验/报告，不改 config")
    norm.add_argument("--json", action="store_true")

    for name, desc in STUBS.items():
        stub = sub.add_parser(name, help=desc)
        stub.add_argument("args", nargs=argparse.REMAINDER)
    return p


def _print_error(exc: ConverterError) -> int:
    print(f"error: {exc}", file=sys.stderr)
    return exc.exit_code


def _normalize_summary(report: dict) -> str:
    c = report["counts"]
    lines = [
        f"path     : {report['path']}",
        f"group    : {report['group_from']} -> {report['group_to']}"
        f"{' (changed)' if report['changed'] else ' (already)'}"
        f"{' [dry-run]' if report['dry_run'] else ''}",
        f"scheme   : {report['scheme']}",
        f"counts   : packed={c.get('packed_layers', 0)}"
        f" scales={c.get('scale_layers', 0)}"
        f" zp={c.get('zero_point_layers', 0)}"
        f" shape={c.get('shape_layers', 0)}",
    ]
    if report["warnings"]:
        lines.append("warnings :")
        for w in report["warnings"]:
            lines.append(f"  - {w}")
    return "\n".join(lines)


def _plan_exit(plan: dict) -> int:
    """按失败原因给退出码：3=不完整，4=磁盘，2=其它结构错误。"""
    if plan.get("status") != "blocked":
        return 0
    codes = {c["code"] for c in plan.get("checks", []) if c.get("level") == "error"}
    if "incomplete-download" in codes:
        return 3
    if "insufficient-disk" in codes:
        return 4
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.cmd in STUBS:
        print(
            f"error: [{args.cmd}] 未实现（P0 只做 inspect）\n  {STUBS[args.cmd]}",
            file=sys.stderr,
        )
        return 64

    if args.cmd == "normalize":
        try:
            report = normalize_checkpoint(args.path, dry_run=args.dry_run)
        except ConverterError as exc:
            return _print_error(exc)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(_normalize_summary(report))
        return 0

    try:
        plan = build_plan(
            args.path,
            args.scheme,
            require_complete=args.require_complete,
            require_disk=args.require_disk,
        )
    except ConverterError as exc:
        return _print_error(exc)

    if args.out:
        write_plan(plan, args.out)
    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
    else:
        print(human_summary(plan))

    return _plan_exit(plan)


if __name__ == "__main__":
    raise SystemExit(main())
