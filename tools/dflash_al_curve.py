#!/usr/bin/env python3
"""DFlash2 长上下文 AL / 分相扫描(metrics-optimize worktree;2026-10-22)。

测法(口径纪律,勿改):
  · AL = Δ`spec.accepted` / Δ`spec.rounds`(引擎内计数器;不用 epoch 视图 ——
    counters 无时间戳,epoch 对计数类结构性失明);
  · 分相 = Δtotal_ms / Δcount(两次 `/debug/metrics` 快照差分;p50 不可差);
  · prompt = 随机词池(prose)/ 随机数字流(math),固定 seed ⇒ 点间可复现;
    每点独立 session + prefix_cache 关 ⇒ 不吃别的前缀;
  · 非流式请求,记录 `usage.prompt_tokens` 作真实 x 轴;
  · 服务端必须已用 profiles/*-dflash-longctx.toml 起在 --url。

用法:
  python3 tools/dflash_al_curve.py --url http://127.0.0.1:8140 \
      --lengths 512,2048,8192,16384,32768,49152,81920 \
      --output-length 256 --domain prose --out perf-runs/metrics-optimize/al-prose

产出:
  <out>/points.jsonl    每点一行(含 m 直方图与全部分相)
  <out>/summary.tsv     人读汇总
  <out>/meta.json       运行参数 + 模型名
"""
import argparse
import hashlib
import json
import pathlib
import random
import sys
import time
import urllib.error
import urllib.request

# 与 packages/llm_speedtest 的 PROMPT_TOKEN_WORDS / DEFAULT_BENCHMARK_SUFFIX 同源
PROSE_WORDS = """a the and of to in is it that for with as on by from this be are or not
we you they can will if all one time world life work data model token text idea mind
story light space future human system simple clear reason change value truth""".split()
PROSE_SUFFIX = (
    "Based on the words above, write a short philosophical essay discussing "
    "the meaning of existence, the nature of consciousness, and humanity's "
    "place in the universe. Use clear, coherent sentences."
)
MATH_SUFFIX = (
    "Using these numbers, continuously generate and solve simple addition problems, "
    "one per line, strictly in the format 'a + b = c'. Keep computing new problems "
    "without stopping, commenting, or explaining."
)

SPEC_TAGS = (
    "spec.round",
    "spec.verify",
    "spec.verify.read",
    "spec.propose",
    "spec.snap",
    "spec.fold",
    "spec.replay",
    "spec.emit",
    "spec.round.spec",
    "spec.round.deg",
    "srv.wait.dtoh",
    "prefill.chunk",
    "prefill.extend",
    "dflash.prefill.encode",
    "decode.step",
    "engine.phase.prefill",
    "engine.phase.spec",
)
COUNTER_TAGS = (
    "spec.rounds",
    "spec.tokens",
    "spec.accepted",
    "spec.deg.rounds",
) + tuple(f"spec.m{i}" for i in range(8))


def post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def snapshot(base: str) -> dict:
    rows: dict = {}
    for prefix in ("spec", "prefill", "decode", "srv", "engine"):
        try:
            js = get_json(f"{base}/debug/metrics/{prefix}")
        except (urllib.error.URLError, TimeoutError) as e:  # 诊断面:抓不到就记下
            print(f"  ! metrics/{prefix} 抓取失败: {e}", file=sys.stderr)
            continue
        for r in js.get("rows", []):
            rows[r["tag"]] = r
    return rows


def row(rows: dict, tag: str) -> dict:
    return rows.get(tag) or {}


def d_counter(a: dict, b: dict, tag: str) -> float:
    return float(row(b, tag).get("counter") or 0) - float(row(a, tag).get("counter") or 0)


def d_count(a: dict, b: dict, tag: str) -> float:
    return float(row(b, tag).get("count") or 0) - float(row(a, tag).get("count") or 0)


def d_total_ms(a: dict, b: dict, tag: str) -> float:
    return float(row(b, tag).get("total_ms") or 0) - float(row(a, tag).get("total_ms") or 0)


def make_body(target: int, domain: str, rng: random.Random) -> str:
    """生成 ~target token 的正文(后缀另加;实际 token 数由 /tokenize 校准)。"""
    if domain == "math":
        words = [str(rng.randint(1, 9999)) for _ in range(max(target, 1))]
        return " ".join(words)
    # prose:短词≈1 token/词;按 1.0 起步,靠校准逼近
    words = [rng.choice(PROSE_WORDS) for _ in range(max(int(target), 1))]
    return " ".join(words)


def calibrate_prompt(base: str, target: int, domain: str, seed: int,
                     corpus: str | None = None) -> tuple[str, int]:
    """用 /tokenize(raw) 校准到 target ±3%;返回 (text, raw_tokens)。"""
    suffix = MATH_SUFFIX if domain == "math" else PROSE_SUFFIX
    rng = random.Random(f"owl-dflash-al-{seed}-{domain}-{target}")
    if corpus:
        # 固定语料:按 (seed,target) 旋转起点,二分字符切片逼近 target token
        off = abs(hash((seed, target, domain))) % max(len(corpus), 1)
        rot = corpus[off:] + corpus[:off]
        lo, hi = 1, len(rot)
        text = rot
        for _ in range(24):
            mid = (lo + hi) // 2
            cand = rot[:mid] + "\n" + suffix
            c = int(post_json(f"{base}/tokenize", {"prompt": cand, "mode": "raw"}, timeout=180)["count"])
            if c >= target:
                text, hi = cand, mid
            else:
                lo = mid + 1
            if abs(c - target) <= max(4, int(target * 0.03)):
                text = cand
                break
        count = int(post_json(f"{base}/tokenize", {"prompt": text, "mode": "raw"}, timeout=180)["count"])
        return text, count
    n = target
    text = make_body(n, domain, rng) + "\n" + suffix
    count = 0
    for _ in range(4):
        js = post_json(f"{base}/tokenize", {"prompt": text, "mode": "raw"}, timeout=120)
        count = int(js["count"])
        if abs(count - target) <= max(4, int(target * 0.03)):
            break
        n = max(1, int(round(n * target / max(count, 1))))
        text = make_body(n, domain, rng) + "\n" + suffix
    return text, count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8140")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--lengths", default="512,2048,8192,16384,32768,49152,81920")
    ap.add_argument("--output-length", type=int, default=256)
    ap.add_argument("--domain", choices=("prose", "math"), default="prose")
    ap.add_argument("--seed", type=int, default=20261022)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--sleep-between", type=float, default=0.0,
                    help="两点之间空闲秒数(nsys 按期段切分时建议 2)")
    ap.add_argument("--prompt-files", default="",
                    help="逗号分隔的真实语料文件(给了就不用随机词池;按 token 切片)")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    corpus = None
    if args.prompt_files:
        corpus = "\n\n".join(pathlib.Path(x).read_text(errors="replace") for x in args.prompt_files.split(","))
        print(f"[corpus] chars={len(corpus)} files={args.prompt_files}")
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    # 健康检查 + 模型名
    health = get_json(f"{base}/health")
    model = args.model or health.get("model")
    meta = {
        "url": base,
        "model": model,
        "health": health,
        "lengths": lengths,
        "output_length": args.output_length,
        "domain": args.domain,
        "seed": args.seed,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[meta] {base} model={model} domain={args.domain}")

    summary = out / "summary.tsv"
    with summary.open("w") as f:
        f.write(
            "target\tprompt_tok\tcompl_tok\twall_s\tal\ttpr"
            "\trounds\taccepted\ttokens\tm_hist"
            "\tround_ms\tverify_ms\tpropose_ms\tsnap_ms\tfold_ms\treplay_ms\temit_ms"
            "\tprefill_phase_ms\tprefill_chunks\tprefill_chunk_ms\tspec_phase_ms"
            "\tverify_read_ms\tsrv_dtoh_ms\tprefill_extend_ms\tdflash_encode_ms"
            "\tround_spec_ms\tround_deg_ms\tdeg_rounds\n"
        )

    with (out / "points.jsonl").open("w") as pj:
        for idx, target in enumerate(lengths):
            if idx and args.sleep_between > 0:
                time.sleep(args.sleep_between)
            print(f"\n===== target={target} ({args.domain}) =====", flush=True)
            text, raw_tok = calibrate_prompt(base, target, args.domain, args.seed, corpus)
            print(f"  prompt: raw_tok={raw_tok} chars={len(text)}", flush=True)
            before = snapshot(base)
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": text}],
                "max_tokens": args.output_length,
                "temperature": 0,
                "ignore_eos": True,
                "stream": False,
            }
            t0 = time.time()
            try:
                resp = post_json(f"{base}/v1/chat/completions", payload, timeout=args.timeout)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")
                print(f"  ✗ HTTP {e.code}: {body[:400]}", file=sys.stderr, flush=True)
                pj.write(json.dumps({"target": target, "error": body}, ensure_ascii=False) + "\n")
                continue
            wall = time.time() - t0
            after = snapshot(base)

            usage = resp.get("usage") or {}
            choice = (resp.get("choices") or [{}])[0]
            out_text = (choice.get("message") or {}).get("content") or ""
            out_sha1 = hashlib.sha1(out_text.encode()).hexdigest()[:16]
            finish_reason = choice.get("finish_reason")
            prompt_tok = int(usage.get("prompt_tokens") or raw_tok)
            compl_tok = int(usage.get("completion_tokens") or 0)

            rounds = d_counter(before, after, "spec.rounds")
            accepted = d_counter(before, after, "spec.accepted")
            tokens = d_counter(before, after, "spec.tokens")
            al = accepted / rounds if rounds else float("nan")
            tpr = tokens / rounds if rounds else float("nan")

            def avg(tag: str) -> float:
                n = d_count(before, after, tag)
                return d_total_ms(before, after, tag) / n if n else float("nan")

            def total(tag: str) -> float:
                return d_total_ms(before, after, tag)

            m_hist = [int(d_counter(before, after, f"spec.m{i}")) for i in range(8)]
            epoch_rows = None
            try:
                ej = get_json(f"{base}/debug/perf/epoch/json")
                epoch_rows = [
                    r for r in ej.get("rows", [])
                    if r["tag"].startswith(("spec.", "engine.phase", "gpu."))
                ]
            except (urllib.error.URLError, TimeoutError):
                pass
            rec = {
                "target": target,
                "prompt_sha1": hashlib.sha1(text.encode()).hexdigest()[:16],
                "raw_tokens": raw_tok,
                "prompt_tokens": prompt_tok,
                "completion_tokens": compl_tok,
                "wall_s": round(wall, 3),
                "al": round(al, 4) if rounds else None,
                "tokens_per_round": round(tpr, 4) if rounds else None,
                "rounds": int(rounds),
                "accepted": int(accepted),
                "spec_tokens": int(tokens),
                "m_hist": m_hist,
                "phase": {
                    t: {"count": d_count(before, after, t), "total_ms": round(total(t), 3), "avg_ms": round(avg(t), 4)}
                    for t in SPEC_TAGS
                },
                "usage": usage,
                "finish_reason": finish_reason,
                "output_sha1": out_sha1,
                "output_chars": len(out_text),
                "output_text": out_text,
                "epoch_rows": epoch_rows,
            }
            pj.write(json.dumps(rec, ensure_ascii=False) + "\n")
            pj.flush()

            with summary.open("a") as f:
                f.write(
                    f"{target}\t{prompt_tok}\t{compl_tok}\t{wall:.2f}\t{al:.4f}\t{tpr:.4f}"
                    f"\t{int(rounds)}\t{int(accepted)}\t{int(tokens)}\t{','.join(map(str, m_hist))}"
                    f"\t{avg('spec.round'):.3f}\t{avg('spec.verify'):.3f}\t{avg('spec.propose'):.3f}"
                    f"\t{avg('spec.snap'):.3f}\t{avg('spec.fold'):.3f}\t{avg('spec.replay'):.3f}"
                    f"\t{avg('spec.emit'):.3f}"
                    f"\t{total('engine.phase.prefill'):.1f}\t{int(d_count(before, after, 'prefill.chunk'))}"
                    f"\t{avg('prefill.chunk'):.2f}\t{total('engine.phase.spec'):.1f}"
                    f"\t{avg('spec.verify.read'):.3f}\t{avg('srv.wait.dtoh'):.3f}"
                    f"\t{total('prefill.extend'):.2f}\t{total('dflash.prefill.encode'):.2f}"
                    f"\t{avg('spec.round.spec'):.3f}\t{avg('spec.round.deg'):.3f}"
                    f"\t{int(d_counter(before, after, 'spec.deg.rounds'))}\n"
                )
            print(
                f"  → prompt_tok={prompt_tok} compl={compl_tok} wall={wall:.1f}s "
                f"AL={al:.3f} tpr={tpr:.2f} rounds={int(rounds)} "
                f"round={avg('spec.round'):.1f}ms verify={avg('spec.verify'):.1f}ms "
                f"propose={avg('spec.propose'):.1f}ms",
                flush=True,
            )

    print(f"\n完成 ⇒ {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
