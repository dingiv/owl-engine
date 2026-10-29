"""awq-forge command line."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from . import __version__
from .source import inspect_source


def _print_inspect(info, as_json: bool) -> None:
    if as_json:
        print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))
        return
    print(f"source       : {info.source}")
    print(f"architecture : {', '.join(info.architecture) or '-'}")
    print(f"model_type   : {info.model_type or '-'}")
    print(f"quantization : {info.quantization.get('quant_method', '-') if info.quantization else '-'}")
    print(f"files        : {len(info.files)}")
    print(f"tensors      : {info.tensor_count}")
    print(f"dtypes       : {info.dtypes}")
    print(f"targets      : {info.target_count} {info.target_roles}")
    print(f"ignored      : {info.ignored}")
    for warning in info.warnings:
        print(f"warning      : {warning}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="awq-forge")
    parser.add_argument("--version", action="version", version=f"awq-forge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="inspect a source checkpoint (no torch)")
    inspect.add_argument("source", help="model directory with config.json + safetensors")
    inspect.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "inspect":
        try:
            info = inspect_source(args.source)
        except ValueError as exc:
            print(f"awq-forge: {exc}", file=sys.stderr)
            return 2
        _print_inspect(info, args.json)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
