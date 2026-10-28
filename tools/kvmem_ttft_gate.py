#!/usr/bin/env python3
"""KVMem TTFT 门（W4;验收表直接可贴台账）。

口径（与参照实现对照时的纪律）：
  · 一次服务只跑**同一前缀**的一串 turn（换文档会污染宿主层）；
  · 每轮形状 = agent 真实会话：`+assistant(上一轮回答) +user(新问题)`；
  · TTFT = 流式首个**内容/思考/工具** delta 的到达时刻（不是总和）；
  · 中段针 = 文档 50% 处插入的唯一事实，追加轮里提问（检验懒回灌后的召回）。

用法：
  uv run python tools/kvmem_ttft_gate.py --ctx 111k --turns 4 --port 8140
  uv run python tools/kvmem_ttft_gate.py --doc /tmp/agent_doc_100k.txt
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request

NEEDLE = "KPX-7731"
NEEDLE_LINE = (
    "\n\n> 【运维附录·并发席位台账】编号 " + NEEDLE + " 的席位专供 KVMem 长上下文单卡档使用。\n\n"
)
ASK_MIDDLE = f"《运维附录·并发席位台账》里那个席位的编号是多少？只回编号本身（形如 KPX-0000）。"


def first_token_ttft(port: int, model: str, msgs, max_tokens: int) -> tuple[float | None, float, str]:
    body = {
        "model": model,
        "messages": msgs,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "reasoning_effort": "none",
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft: float | None = None
    text: list[str] = []
    with urllib.request.urlopen(req, timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            j = json.loads(payload)
            ch = j["choices"][0].get("delta", {})
            piece = ch.get("content") or ch.get("reasoning_content")
            if ch.get("tool_calls"):
                piece = (piece or "") + "<tool_call>"
            if piece:
                if ttft is None:
                    ttft = time.time() - t0
                text.append(piece)
    return ttft, time.time() - t0, "".join(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--doc", default="/tmp/agent_doc_100k.txt")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--ctx", default="(档自定)", help="仅打印用标签")
    ap.add_argument("--no-needle", action="store_true")
    args = ap.parse_args()

    doc = open(args.doc, encoding="utf-8").read()
    if not args.no_needle:
        cut = len(doc) // 2
        doc = doc[:cut] + NEEDLE_LINE + doc[cut:]
    sys_msg = "你是 pi 编码代理，严格遵守工作区规则。回答尽量短。"
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": "以下是项目台账，请通读：\n" + doc},
    ]
    budget_chars = sum(len(m["content"]) for m in msgs)
    print(f"# ctx 档={args.ctx} doc_chars={budget_chars} needle={'no' if args.no_needle else NEEDLE}")
    print("| 轮 | TTFT(s) | 总时长(s) | 答案 | 针 |")
    print("|---:|---:|---:|---|---|")
    hit = False
    for i in range(1, args.turns + 1):
        ask = ASK_MIDDLE if (i == 2 and not args.no_needle) else f"第{i}问：用一句话说一条要点。"
        msgs.append({"role": "user", "content": ask})
        ttft, total, answer = first_token_ttft(
            args.port, args.model, msgs, args.max_tokens
        )
        got = NEEDLE in answer
        hit = hit or got
        flag = "✅" if (not (i == 2) or args.no_needle or got) else "❌"
        print(
            f"| {i} | {'-' if ttft is None else f'{ttft:.2f}'} | {total:.1f} | "
            f"{answer.strip()[:40]!r} | {flag} |",
            flush=True,
        )
        msgs.append({"role": "assistant", "content": answer or "（已读）"})
    print(f"\n中段针命中: {'是' if hit else '否'}")
    return 0 if (args.no_needle or hit) else 1


if __name__ == "__main__":
    raise SystemExit(main())
