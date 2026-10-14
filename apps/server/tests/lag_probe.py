#!/usr/bin/env python3
"""多请求「一轮滞后」污染探针(agent 服务面专项 §7.6 最小复现器)。

对**已启动**的 owl server 顺序发 4 个短问题,逐个比对「本题答案」与
「上一题答案」:若后者命中,即 §7.6 的一轮滞后污染(KV/状态跨请求残留)。

用法:
    python3 lag_probe.py --base-url http://127.0.0.1:8137 --model qwen3.8-27b
        [--mode stateless|session]   # 默认 stateless = pi 的真实姿势(不带 session)
        [--think on|off]
退出码:0 = 无滞后,2 = 检出滞后,1 = 请求失败。
"""

import argparse
import json
import sys
import urllib.error
import urllib.request

# (提问, 本题期望答案关键词, 唯一标记)
CASES = [
    ("法国的首都是哪座城市?只回答城市名", "巴黎"),
    ("1+1=? 只回答数字", "2"),
    ("3+4=? 只回答数字", "7"),
    ("5+6=? 只回答数字", "11"),
]


def post(base: str, body: dict) -> dict:
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8137")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--mode", choices=["stateless", "session"], default="stateless")
    ap.add_argument("--think", choices=["on", "off"], default="on")
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    base = a.base_url.rstrip("/")

    answers, verdict = [], []
    hist = []
    for i, (q, want) in enumerate(CASES):
        msgs = list(hist) + [{"role": "user", "content": q}]
        body = {
            "model": a.model,
            "messages": msgs,
            "max_tokens": a.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": a.think == "on",
                                     "preserve_thinking": True},
        }
        if a.mode == "session":
            body["session"] = "lag-probe"
        try:
            d = post(base, body)
        except urllib.error.HTTPError as e:
            print(f"[ERROR] 第 {i + 1} 题 HTTP {e.code}: {e.read()[:200]!r}")
            return 1
        m = d["choices"][0]["message"]
        text = (m.get("content") or "") + "||" + (m.get("reasoning_content") or "")
        answers.append(text)
        prev = CASES[i - 1][1] if i else None
        hit_own = want in text
        hit_prev = bool(prev) and prev in text and not hit_own
        tag = "own ✓" if hit_own else ("PREV ✗" if hit_prev else "other ?")
        verdict.append(hit_own and not hit_prev)
        print(f"[{i + 1}] 期望 {want:>3s}  prev={prev or '-':>3s}  → {tag}   {(m.get('content') or '').strip()[:48]!r}")
        hist.append({"role": "user", "content": q})
        hist.append({"role": "assistant", "content": m.get("content") or "",
                     **({"reasoning_content": m["reasoning_content"]}
                        if m.get("reasoning_content") else {})})

    lag = sum(1 for v in verdict if not v)
    print()
    if lag == 0:
        print("✅ 无滞后(4/4 各答各题)")
        return 0
    print(f"❌ 检出 {lag}/4 题异常 —— 一轮滞后污染(§7.6)")
    return 2


if __name__ == "__main__":
    sys.exit(main())
