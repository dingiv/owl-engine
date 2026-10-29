#!/usr/bin/env python3
"""RTN W4A16 g128-sym 冒烟：dense BF16/F16 -> compressed-tensors pack-quantized。

这是 P1 的“格式链路证明”，不是质量交付；生产质量请走 AWQ g32。
用法（需先 `uv sync --extra quant`）：
  uv run --extra quant python examples/rtn_w4a16.py <src> <dst>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

IGNORE = [
    "lm_head",
    r"re:.*visual.*",
    r"re:.*mtp.*",
    r"re:.*embed_tokens.*",
    r"re:.*conv1d.*",
    r"re:.*A_log.*",
    r"re:.*dt_bias.*",
    r"re:.*in_proj_a.*",
    r"re:.*in_proj_b.*",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if args.dst.exists():
        print(f"error: {args.dst} 已存在（拒绝覆盖）", file=sys.stderr)
        return 2

    from llmcompressor import model_free_ptq

    print(f"src={args.src}\ndst={args.dst}\nscheme=W4A16 (g128 sym, RTN)")
    model_free_ptq(
        model_stub=str(args.src),
        save_directory=str(args.dst),
        scheme="W4A16",
        ignore=IGNORE,
        device=args.device,
        max_workers=1,
    )
    print("done; 下一步：uv run owl-convert normalize", args.dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
