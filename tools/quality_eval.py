#!/usr/bin/env python3
"""FP8/量化基线质量评测（OpenAI chat API）。

借自：
- `repos/vllm-ampere-optimized/eval/gsm8k_api_eval.py`
- `repos/vllm-ampere-optimized/eval/gsm8k_eval.py`
- `repos/HyperQwen/bench/quality_battery.py`

只使用标准库；数据用 vllm-ampere 的 JSONL（gsm8k / mmlu_pro）。

示例：
  # 5 题 coherence smoke（先跑这个）
  python3 tools/quality_eval.py --task gsm8k --n 5 \
    --data /home/div/Documents/codes/repos/vllm-ampere-optimized/benchmarks/gsm8k.jsonl \
    --base http://127.0.0.1:8134 --tag 3090-fp8-smoke

  # 200 题（thinking off, greedy）
  python3 tools/quality_eval.py --task gsm8k --n 200 \
    --data .../gsm8k.jsonl --base http://127.0.0.1:8134 --tag 3090-fp8

  # thinking on（Qwen 推理协议）
  python3 tools/quality_eval.py --task gsm8k --n 200 --thinking on \
    --temperature 0.6 --top-p 0.95 --max-tokens 24576 ...

  # MMLU-Pro
  python3 tools/quality_eval.py --task mmlu_pro --n 200 \
    --data .../mmlu_pro.jsonl --base http://127.0.0.1:8135 --tag 3080-fp8-mmlu
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
LETTERS = "ABCDEFGHIJ"


# ── GSM8K 抽取（与 vllm-ampere eval/gsm8k_eval.py 同口径） ──────────────
def split_after_think(text: str) -> str:
    marker = "</think>"
    i = text.rfind(marker)
    return text[i + len(marker):] if i != -1 else text


def delatex(text: str) -> str:
    text = text.replace(",", "").replace("$", "")
    text = re.sub(
        r"\\[dt]?frac\s*\{\s*(-?[\d.]+)\s*\}\s*\{\s*(-?[\d.]+)\s*\}",
        lambda m: (
            f"{float(m.group(1)) / float(m.group(2)):.6f}".rstrip("0").rstrip(".")
            if float(m.group(2)) else " "
        ),
        text,
    )
    text = re.sub(r"\\[a-zA-Z]+", " ", text)
    text = re.sub(r"\{\s*\}", "", text)
    text = text.replace("{", " ").replace("}", " ").replace("\\", " ")
    return re.sub(r"(?<=\d)[\s~]+(?=\d{3}(?!\d))", "", text)


def boxed_number(text: str) -> str | None:
    last = None
    for m in re.finditer(r"\\boxed\s*\{", text):
        i, depth, buf = m.end(), 1, []
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            buf.append(c)
            i += 1
        nums = NUM.findall(delatex("".join(buf)))
        if nums:
            last = nums[-1].rstrip(".")
    return last


def pred_number(text: str) -> str | None:
    ans = split_after_think(text)
    b = boxed_number(ans)
    if b is not None:
        return b
    nums = NUM.findall(delatex(ans))
    return nums[-1].rstrip(".") if nums else None


def gold_number(answer: str) -> str | None:
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", answer)
    return m.group(1).replace(",", "") if m else None


def numbers_match(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False


# ── MMLU-Pro 抽取 ───────────────────────────────────────────────────────
def mmlu_prompt(row: dict) -> str:
    lines = "\n".join(f"{LETTERS[i]}. {x}" for i, x in enumerate(row["options"]))
    return (
        f"Question: {row['question']}\nOptions:\n{lines}\n\n"
        "Think step by step, then end with 'The answer is (X)' where X is the correct option letter."
    )


def mmlu_pred(text: str, n_options: int) -> str | None:
    ls = LETTERS[:n_options]
    for pat in (
        rf"answer is[:\s\*\(]*([{ls}])\b",
        rf"\\boxed\{{\s*([{ls}])",
        rf"answer[:\s\*\(]*([{ls}])\b",
        rf"\(([{ls}])\)",
    ):
        m = re.findall(pat, text, re.I)
        if m:
            return m[-1].upper()
    m = re.findall(rf"\b([{ls}])\b", text)
    return m[-1].upper() if m else None


# ── HTTP ────────────────────────────────────────────────────────────────
def post(base: str, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def load_rows(task: str, path: str, n: int) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return rows[:n] if n else rows


def answer_text(msg: dict) -> str:
    return (msg.get("reasoning_content") or "") + (msg.get("content") or "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k", "mmlu_pro"], default="gsm8k")
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8134")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="从 --out 旁的 .partial.jsonl 续跑（已完成的 idx 跳过）")
    args = ap.parse_args()

    if args.temperature is None:
        args.temperature = 0.0 if args.thinking == "off" else 0.6

    rows = load_rows(args.task, args.data, args.n)
    if not rows:
        print("no rows", file=sys.stderr)
        return 2

    partial_path = (args.out + ".partial.jsonl") if args.out else None
    existing = {}
    if args.resume and partial_path and os.path.exists(partial_path):
        for line in open(partial_path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                existing[int(rec["idx"])] = rec
            except Exception:
                pass
        existing = {i: r for i, r in existing.items() if i < len(rows)}
    if partial_path:
        os.makedirs(os.path.dirname(os.path.abspath(partial_path)), exist_ok=True)
        if not args.resume:
            open(partial_path, "w").close()

    print(
        f"[eval {args.tag}] task={args.task} n={len(rows)} base={args.base} "
        f"thinking={args.thinking} temp={args.temperature} top_p={args.top_p} "
        f"max_tokens={args.max_tokens} conc={args.concurrency} "
        f"resume={len(existing)}",
        flush=True,
    )

    done = [len(existing)]
    lock = threading.Lock()
    t0 = time.time()

    def work(ir):
        i, row = ir
        if args.task == "gsm8k":
            prompt = (
                row["question"].strip()
                + "\n\nSolve step by step, then give the final answer as 'Final answer: <number>'."
            )
        else:
            prompt = mmlu_prompt(row)

        body = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "chat_template_kwargs": {"enable_thinking": args.thinking == "on"},
        }
        err = None
        text, ctoks, finish = "", 0, None
        for attempt in range(3):
            try:
                d = post(args.base, body, args.timeout)
                msg = d["choices"][0]["message"]
                text = answer_text(msg)
                ctoks = d.get("usage", {}).get("completion_tokens", 0)
                finish = d["choices"][0].get("finish_reason")
                err = None
                break
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))
        if err:
            text, ctoks, finish = "", 0, f"ERR:{err}"

        if args.task == "gsm8k":
            gold = gold_number(row["answer"])
            pred = pred_number(text)
            ok = numbers_match(pred, gold)
        else:
            gold = str(row["answer"]).strip().upper()
            pred = mmlu_pred(text, len(row["options"]))
            ok = pred == gold

        rec = {
            "idx": i,
            "gold": gold,
            "pred": pred,
            "ok": bool(ok),
            "completion_tokens": ctoks,
            "finish": finish,
            "error": err,
            "text": None if ok else text[:8000],
        }
        with lock:
            done[0] += 1
            if partial_path:
                with open(partial_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if done[0] % 10 == 0 or done[0] == len(rows):
                print(f"  [{done[0]}/{len(rows)}] {time.time()-t0:.0f}s", flush=True)
        return rec

    todo = [(i, row) for i, row in enumerate(rows) if i not in existing]
    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
        new_results = list(ex.map(work, todo))

    results = list(existing.values()) + new_results
    results.sort(key=lambda x: x["idx"])
    n = len(results)
    correct = sum(1 for r in results if r["ok"])
    errors = sum(1 for r in results if r["error"])
    truncated = sum(1 for r in results if r["finish"] == "length")
    toks = sum(r["completion_tokens"] for r in results)
    wall = time.time() - t0
    acc = 100.0 * correct / n
    summary = {
        "tag": args.tag,
        "task": args.task,
        "base": args.base,
        "model": args.model,
        "n": n,
        "correct": correct,
        "acc": acc,
        "errors": errors,
        "truncated": truncated,
        "mean_completion_tokens": toks / max(1, n),
        "wall_s": wall,
        "thinking": args.thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "regime": {
            "thinking": args.thinking,
            "greedy": args.temperature == 0.0,
        },
        "results": results,
    }
    print(
        f"\nRESULT {args.tag}: {args.task} {correct}/{n} = {acc:.2f}% "
        f"errors={errors} truncated={truncated} "
        f"mean_tokens={toks/max(1,n):.0f} wall={wall/60:.1f}min",
        flush=True,
    )
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(summary, open(args.out, "w"), ensure_ascii=False, indent=2)
        print(f"  wrote {args.out}", flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
