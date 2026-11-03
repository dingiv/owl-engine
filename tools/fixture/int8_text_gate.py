#!/usr/bin/env python3
"""INT8 KV 文本质量门（最小可执行版）。

对一个**刚启动、未跑过请求**的 INT8 端点跑 needle；默认只跑 1k
（prompt≈1040），因为这是当前 INT8 在 `prefill_chunk=2048` +
`prefix_cache=true`（末段页对齐 ⇒ 1024+16）下会失败的最小用例。

用法：
  ./target/release/server -p profiles/measurement/qwen38_27b-3090-int8.toml ... &
  python3 tools/fixture/int8_text_gate.py --base http://127.0.0.1:8141

退出码：0 = 全 RETRIEVED；1 = 有 MISSED（质量门失败）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

NEEDLE = "ZXCVBNM12345"
UNIT = "All work and no play makes Jack a dull boy. "


def run_one(base: str, model: str, tokens: int, depth: float, max_tokens: int, timeout: int) -> dict:
    filler = UNIT * max(1, int(tokens / 11))
    cut = int(len(filler) * depth)
    needle = f"The secret passcode is {NEEDLE}. Remember it exactly."
    prompt = filler[:cut] + "\n\n" + needle + "\n\n" + filler[cut:]
    prompt += "\n\nQuestion: what is the secret passcode? Reply with the passcode only."
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
        msg = d["choices"][0]["message"]
        text = (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        finish = d["choices"][0].get("finish_reason")
        err = None
    except Exception as e:  # 网络/服务错误也计失败，不静默跳过
        text, finish, err = "", None, f"{type(e).__name__}: {e}"
    return {
        "tokens": tokens,
        "retrieved": NEEDLE in text,
        "elapsed_s": time.time() - t0,
        "finish": finish,
        "answer": text[:200],
        "error": err,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8141")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--sizes", default="1000", help="逗号分隔；120000 很慢，默认不跑")
    ap.add_argument("--depth", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    sizes = [int(x) for x in a.sizes.split(",") if x.strip()]
    bad = 0
    for n in sizes:
        r = run_one(a.base, a.model, n, a.depth, a.max_tokens, a.timeout)
        print(
            f"target~{n} tok: {'RETRIEVED' if r['retrieved'] else 'MISSED'} "
            f"{r['elapsed_s']:.1f}s finish={r['finish']} answer={r['answer']!r}"
        )
        if r["error"]:
            print(f"  error: {r['error']}")
        if not r["retrieved"]:
            bad += 1
    print(f"\nINT8 text gate: {len(sizes) - bad}/{len(sizes)} passed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
