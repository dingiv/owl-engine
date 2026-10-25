#!/usr/bin/env python3
"""Needle-in-a-haystack probe for owl's OpenAI chat API.

Adapted from `repos/HyperQwen/bench/needle_test.py`:
- build ~TARGET_TOKENS filler (11 tokens per unit),
- hide a passcode at a fractional depth,
- ask for it, report RETRIEVED / MISSED and elapsed time.

Note: max_tokens must not be tiny while DFlash2 spec is on (tail-cap P0).
Default 512; the model normally stops much earlier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

NEEDLE = "ZXCVBNM12345"
UNIT = "All work and no play makes Jack a dull boy. "


def post(base: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8134")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--tokens", type=int, default=100_000)
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # ~11 tokens per unit (HyperQwen's calibration)
    filler = UNIT * max(1, int(args.tokens / 11))
    cut = int(len(filler) * args.depth)
    needle_line = f"The secret passcode is {NEEDLE}. Remember it exactly."
    context = filler[:cut] + "\n\n" + needle_line + "\n\n" + filler[cut:]
    prompt = context + "\n\nQuestion: what is the secret passcode? Reply with the passcode only."

    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    try:
        d = post(args.base, body, args.timeout)
        msg = d["choices"][0]["message"]
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        finish = d["choices"][0].get("finish_reason")
        err = None
    except Exception as e:
        text, finish, err = "", None, f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0
    ok = NEEDLE in text

    print(
        f"target~{args.tokens} tok depth={args.depth:.0%} "
        f"elapsed={elapsed:.1f}s finish={finish} "
        f"answer={text[:160]!r} -> {'RETRIEVED' if ok else 'MISSED'}"
    )
    rec = {
        "tag": args.tag,
        "base": args.base,
        "target_tokens": args.tokens,
        "depth": args.depth,
        "needle": NEEDLE,
        "retrieved": ok,
        "finish": finish,
        "error": err,
        "elapsed_s": elapsed,
        "answer": text[:1000],
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(rec, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
