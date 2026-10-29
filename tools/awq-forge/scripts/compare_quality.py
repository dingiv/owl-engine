#!/usr/bin/env python3
"""配对比较两份 quality_gate 结果:准确率差 + 精确 McNemar。

用法:
  python scripts/compare_quality.py --a out/awq.json --b out/fp8.json \
      --label-a AWQ --label-b FP8

两份结果必须同一套题(逐题记录按索引对齐);只比较 GSM8K。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def load_items(result_path: Path) -> tuple[dict, list[dict]]:
    meta = json.loads(result_path.read_text())
    items_path = meta.get("gsm8k", {}).get("items_path")
    if not items_path:
        raise SystemExit(f"✗ {result_path} 没有 gsm8k.items_path")
    path = Path(items_path)
    if not path.is_absolute():
        path = result_path.parent / path.name
    if not path.is_file():
        raise SystemExit(f"✗ 逐题记录不存在: {path}")
    items = [json.loads(line) for line in path.open() if line.strip()]
    return meta, items


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="配对比较 quality_gate 结果")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args(argv)

    meta_a, items_a = load_items(Path(args.a))
    meta_b, items_b = load_items(Path(args.b))
    if len(items_a) != len(items_b):
        raise SystemExit(f"✗ 题数不同:{len(items_a)} vs {len(items_b)}")

    both = sum(a["ok"] and b["ok"] for a, b in zip(items_a, items_b))
    only_a = sum(a["ok"] and not b["ok"] for a, b in zip(items_a, items_b))
    only_b = sum(b["ok"] and not a["ok"] for a, b in zip(items_a, items_b))
    neither = sum((not a["ok"]) and (not b["ok"]) for a, b in zip(items_a, items_b))
    n = len(items_a)
    acc_a = (both + only_a) / n
    acc_b = (both + only_b) / n
    p = mcnemar_exact(only_a, only_b)

    print(f"{args.label_a}: {meta_a.get('gsm8k', {}).get('correct')}/{n} = {acc_a*100:.1f}%"
          f"  (max_tokens={meta_a.get('max_tokens')}, concurrency={meta_a.get('concurrency')})")
    print(f"{args.label_b}: {meta_b.get('gsm8k', {}).get('correct')}/{n} = {acc_b*100:.1f}%"
          f"  (max_tokens={meta_b.get('max_tokens')}, concurrency={meta_b.get('concurrency')})")
    print(f"delta = {(acc_a - acc_b)*100:+.1f}pp")
    print(f"paired: both={both} {args.label_a}-only={only_a} "
          f"{args.label_b}-only={only_b} neither={neither}")
    print(f"McNemar exact two-sided p = {p:.4f}")
    flips = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(items_a, items_b))
        if a["ok"] != b["ok"]
    ]
    for i, a, b in flips:
        print(f"  flip #{i}: {args.label_a}={a['ok']}({a['pred']}) "
              f"{args.label_b}={b['ok']}({b['pred']}) gold={a['gold']} "
              f"tok={a['completion_tokens']}/{b['completion_tokens']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
