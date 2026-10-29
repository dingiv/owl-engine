#!/usr/bin/env python3
"""Build a **true fixed-tail** long-context prompt for AL A/B.

Why this tool exists: the 2026-10-22 first fixed-tail harness sliced
`prose-zh.txt[60000:68000]`, but that file is only 50,301 chars, so the
tail was empty and the "fixed-tail" experiment actually used different
prefixes. The bug is silent in a plain slice; this tool fails loudly.

Usage:
  python3 tools/fixedtail_prompt.py --length 32768 --out /tmp/fixedtail32k.txt

The prompt = variable prefix + a fixed tail + a fixed instruction. The tail
is taken from a repeated copy of the source, so it exists even for a short
source. The last ~2K tokens (tail end + instruction) are identical across
lengths, which is what the DFlash2 draft sees.
"""
import argparse
import pathlib
import sys

DEFAULT_SRC = pathlib.Path(
    "/home/div/Documents/codes/packages/llm_speedtest/prompts/prose-zh.txt"
)
INSTR = "请接着上面这段文字继续写下去，保持同样的风格，写300字左右，不要重复原文。"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--length", type=int, required=True, help="target prompt tokens (approx)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default=str(DEFAULT_SRC))
    ap.add_argument("--source-repeat", type=int, default=4)
    ap.add_argument("--tail-start", type=int, default=60000)
    ap.add_argument("--tail-chars", type=int, default=8000)
    ap.add_argument("--chars-per-token", type=float, default=1.85)
    args = ap.parse_args()

    src = pathlib.Path(args.source).read_text()
    src = (src + "\n\n") * max(1, args.source_repeat)
    if args.tail_start + args.tail_chars > len(src):
        raise SystemExit(
            f"tail slice out of range: source={len(src)} chars, "
            f"need [{args.tail_start}, {args.tail_start + args.tail_chars})"
        )
    tail = src[args.tail_start : args.tail_start + args.tail_chars]
    if len(tail) < args.tail_chars:
        raise SystemExit(
            f"tail too short: got {len(tail)} < {args.tail_chars}; "
            "increase --source-repeat or lower --tail-start"
        )
    tail_tokens = max(1, int(len(tail) / args.chars_per_token))
    prefix_chars = int(max(0, args.length - tail_tokens) * args.chars_per_token)
    prefix = src[:prefix_chars]
    prompt = (prefix + "\n\n" if prefix else "") + tail + "\n\n" + INSTR
    pathlib.Path(args.out).write_text(prompt)
    print(
        f"L={args.length} prefix_chars={prefix_chars} tail_chars={len(tail)} "
        f"total_chars={len(prompt)} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
