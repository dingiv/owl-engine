#!/usr/bin/env python3
"""质量门:GSM8K 200 + needle 16k(OpenAI 兼容端点,黑盒)。

口径(awq-forge REQUIREMENTS S2/S3):
  * GSM8K 200 题,greedy(temperature=0),关闭 thinking;
  * needle 16k:暗号埋在长文 head / mid / tail,要求原样检索;
  * 同一端点、同一 prompt、同一采样参数即可跨权重/引擎对照。

只依赖标准库。用法:
  python scripts/quality_gate.py --api-url http://127.0.0.1:8199/v1/chat/completions \
      --model qwen3.8-27b --tag awq-int4 --out out/quality-awq.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_GSM8K_CANDIDATES = (
    Path("repos/vllm-ampere-optimized/benchmarks/gsm8k.jsonl"),
    Path(__file__).resolve().parents[5]
    / "repos/vllm-ampere-optimized/benchmarks/gsm8k.jsonl",
)
NEEDLE = "K7-ZEBRA-4713"
WORDS = (
    "the of and to in is it that for with as on by from this be are or not we "
    "you they can will if all one time world life work data model token text "
    "idea mind story light space future human system simple clear reason"
).split()


# ── HTTP ────────────────────────────────────────────────────────────────
def post_json(url: str, payload: dict, timeout: float, api_key: str = "") -> dict:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data, headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:800]
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"连接失败 {url}: {exc.reason}") from exc


def chat(
    url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    timeout: float,
    api_key: str = "",
    system: str | None = None,
) -> tuple[str, dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "chat_template_kwargs": {
            "enable_thinking": False,
            "reasoning_effort": "low",
        },
    }
    data = post_json(url, body, timeout, api_key)
    choice = data["choices"][0]
    content = (choice.get("message") or {}).get("content") or ""
    usage = data.get("usage") or {}
    return content, usage


# ── GSM8K ───────────────────────────────────────────────────────────────
FINAL_RE = re.compile(r"Final answer:\s*\**\s*\$?(-?[\d,]*\.?\d+)", re.I)
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")
GSM_PROMPT = (
    "{question}\n\n"
    "Solve step by step, then give the final answer as 'Final answer: <number>'."
)


def gold_of(answer: str) -> str:
    if "####" not in answer:
        raise ValueError(f"GSM8K 答案缺少 ####:{answer[-80:]!r}")
    return answer.split("####")[-1].strip().replace(",", "")


def extract_pred(text: str) -> str | None:
    m = FINAL_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = NUM_RE.findall(text.replace("$", ""))
    return nums[-1].replace(",", "") if nums else None


def gsm_one(item: dict, cfg: argparse.Namespace) -> dict:
    content, usage = chat(
        cfg.api_url,
        cfg.model,
        GSM_PROMPT.format(question=item["question"]),
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        api_key=cfg.api_key,
    )
    pred = extract_pred(content)
    gold = gold_of(item["answer"])
    try:
        ok = pred is not None and math.isclose(
            float(pred), float(gold), rel_tol=0, abs_tol=1e-6
        )
    except (TypeError, ValueError):
        ok = False
    return {
        "ok": bool(ok),
        "gold": gold,
        "pred": pred,
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "tail": content.strip()[-160:],
    }


def resolve_gsm_data(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for cand in DEFAULT_GSM8K_CANDIDATES:
        if cand.is_file():
            return cand
    raise SystemExit(
        "✗ 找不到 GSM8K 数据;用 --gsm-data 指定,候选:"
        + ", ".join(str(c) for c in DEFAULT_GSM8K_CANDIDATES)
    )


def load_gsm(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    if len(rows) < n:
        raise SystemExit(f"✗ GSM8K 数据不足:{path} 只有 {len(rows)} 行,需要 {n}")
    return rows


def run_gsm(cfg: argparse.Namespace) -> dict:
    rows = load_gsm(resolve_gsm_data(cfg.gsm_data), cfg.gsm_n)
    t0 = time.time()
    if cfg.concurrency <= 1:
        results = []
        for i, row in enumerate(rows, 1):
            r = gsm_one(row, cfg)
            results.append(r)
            if i % 20 == 0 or i == len(rows):
                acc = sum(x["ok"] for x in results) / len(results)
                print(
                    f"  gsm {i}/{len(rows)} acc={acc:.3f} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
    else:
        with ThreadPoolExecutor(cfg.concurrency) as pool:
            results = list(pool.map(lambda r: gsm_one(r, cfg), rows))
    acc = sum(x["ok"] for x in results) / len(results)
    toks = [
        x["completion_tokens"] for x in results if x["completion_tokens"] is not None
    ]
    return {
        "n": len(results),
        "acc": round(acc, 4),
        "correct": sum(x["ok"] for x in results),
        "mean_completion_tokens": round(sum(toks) / len(toks), 1) if toks else None,
        "wall_s": round(time.time() - t0, 1),
        "items": results,
    }


# ── needle ──────────────────────────────────────────────────────────────
def pile(words: int, seed: int = 0) -> str:
    rng = random.Random(seed)
    return " ".join(rng.choice(WORDS) for _ in range(words))


def calibrate_words(cfg: argparse.Namespace, target_tokens: int) -> tuple[int, float]:
    """用 1000 词探针测 tokens/word,再换算目标词数。"""
    probe_words = 1000
    text = pile(probe_words, seed=1)
    _, usage = chat(
        cfg.api_url,
        cfg.model,
        text,
        max_tokens=1,
        timeout=cfg.timeout,
        api_key=cfg.api_key,
    )
    pt = usage.get("prompt_tokens")
    if not pt:
        raise RuntimeError("探针没有返回 usage.prompt_tokens,无法校准")
    per_word = pt / probe_words
    words = int(target_tokens / per_word)
    return words, per_word


def run_needle(cfg: argparse.Namespace) -> dict:
    words, per_word = calibrate_words(cfg, cfg.needle_target)
    print(
        f"  needle 校准:{per_word:.3f} tok/word → {words} 词 "
        f"(目标 {cfg.needle_target} tok)",
        flush=True,
    )
    question = (
        "What is the passphrase mentioned in the document above? "
        "Answer with just the passphrase."
    )
    cases = []
    for pos in ("head", "mid", "tail"):
        body = pile(words, seed=7)
        needle_line = f"The passphrase is {NEEDLE}."
        if pos == "head":
            prompt = f"{needle_line}\n\n{body}\n\n{question}"
        elif pos == "mid":
            cut = len(body) // 2
            prompt = f"{body[:cut]}\n\n{needle_line}\n\n{body[cut:]}\n\n{question}"
        else:
            prompt = f"{body}\n\n{needle_line}\n\n{question}"
        t0 = time.time()
        content, usage = chat(
            cfg.api_url,
            cfg.model,
            prompt,
            max_tokens=96,
            timeout=cfg.timeout * 4,
            api_key=cfg.api_key,
        )
        hit = NEEDLE.lower() in content.lower()
        cases.append(
            {
                "position": pos,
                "hit": hit,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "wall_s": round(time.time() - t0, 2),
                "answer": content.strip()[:200],
            }
        )
        print(
            f"  needle-{pos}: prompt_tok={usage.get('prompt_tokens')} "
            f"hit={hit} {time.time() - t0:.1f}s | {content.strip()[:60]!r}",
            flush=True,
        )
    return {
        "target_tokens": cfg.needle_target,
        "measured_tokens_per_word": round(per_word, 4),
        "words": words,
        "passed": all(c["hit"] for c in cases),
        "cases": cases,
    }


# ── main ────────────────────────────────────────────────────────────────
def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="质量门:GSM8K + needle")
    ap.add_argument("--api-url", required=True, help="OpenAI 兼容 chat/completions URL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--tag", required=True, help="结果标签(写进 JSON)")
    ap.add_argument("--out", required=True, help="结果 JSON 路径")
    ap.add_argument("--suite", default="gsm8k,needle", help="逗号分隔:gsm8k,needle")
    ap.add_argument("--gsm-data", default=None, help="GSM8K jsonl;缺省自动找工作区路径")
    ap.add_argument("--gsm-n", type=int, default=200)
    ap.add_argument("--needle-target", type=int, default=16384)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--timeout", type=float, default=1800)
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    cfg = parse_args(argv)
    suites = [s.strip() for s in cfg.suite.split(",") if s.strip()]
    unknown = set(suites) - {"gsm8k", "needle"}
    if unknown:
        raise SystemExit(f"✗ 未知 suite: {sorted(unknown)}")
    if "gsm8k" in suites:
        resolve_gsm_data(cfg.gsm_data)  # 早失败:数据路径必须在跑之前存在

    result: dict = {
        "tag": cfg.tag,
        "api_url": cfg.api_url,
        "model": cfg.model,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "max_tokens": cfg.max_tokens,
        "concurrency": cfg.concurrency,
        "suite": suites,
    }
    out = Path(cfg.out)
    t0 = time.time()
    if "gsm8k" in suites:
        print(f"[gsm8k] n={cfg.gsm_n} concurrency={cfg.concurrency}", flush=True)
        result["gsm8k"] = run_gsm(cfg)
        items = result["gsm8k"].pop("items")  # 逐题另存,主结果保持精简
        items_path = out.with_suffix(out.suffix + ".items.jsonl")
        items_path.parent.mkdir(parents=True, exist_ok=True)
        with items_path.open("w") as fh:
            for idx, item in enumerate(items):
                fh.write(json.dumps({"i": idx, **item}, ensure_ascii=False) + "\n")
        result["gsm8k"]["items_path"] = str(items_path)
    if "needle" in suites:
        print("[needle]", flush=True)
        result["needle"] = run_needle(cfg)
    result["wall_s"] = round(time.time() - t0, 1)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    ok = True
    if "gsm8k" in suites:
        ok = ok and result["gsm8k"]["acc"] >= 0.0  # 相对门在对照后判
    if "needle" in suites:
        ok = ok and result["needle"]["passed"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
