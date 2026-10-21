#!/usr/bin/env python3
"""One-point MTP ctx curve: llm_speedtest + server metrics delta.

Moved out of /tmp on 2026-10-22 (T1b calibration): keep the measurement
recipe in-tree so a fresh boot can reproduce the numbers.
"""
import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

SPEEDTEST = pathlib.Path("/home/div/Documents/codes/packages/llm_speedtest")


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.loads(r.read().decode())


def rows(data):
    return {r["tag"]: r for r in data.get("rows", [])}


def diff(a, b, tag, field="counter"):
    return (b.get(tag, {}).get(field, 0) or 0) - (a.get(tag, {}).get(field, 0) or 0)


def avg_ms(a, b, tag):
    n = diff(a, b, tag, "count")
    ms = diff(a, b, tag, "total_ms")
    return ms / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8140")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--output-length", type=int, default=256)
    ap.add_argument("--domain", default="math")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--temperature", default="0")
    ap.add_argument("--top-p", default="1.0")
    ap.add_argument("--timeout", type=int, default=900000)
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = args.url.rstrip("/")
    before = get_json(f"{base}/debug/metrics")
    t0 = time.time()
    cmd = [
        "uv", "run", "cli",
        "--api-url", f"{base}/v1/chat/completions",
        "--model", args.model,
        "--lengths", str(args.length),
        "--rounds", str(args.rounds),
        "--output-length", str(args.output_length),
        "-c", "1",
        "--ignore-eos",
        "--prompt-domain", args.domain,
        "--temperature", args.temperature,
        "--top-p", args.top_p,
        "--timeout", str(args.timeout),
        "--json", str(out / f"L{args.length}.json"),
        "--csv", str(out / f"L{args.length}.csv"),
        "--notes", f"mtp ctx curve L={args.length}",
    ]
    if args.prompt_file:
        cmd += ["--prompt-file", str(pathlib.Path(args.prompt_file).resolve())]
    run = subprocess.run(cmd, cwd=SPEEDTEST, capture_output=True, text=True)
    (out / f"L{args.length}.cli.log").write_text(run.stdout + "\n--- stderr ---\n" + run.stderr)
    print(run.stdout[-1800:])
    if run.returncode != 0:
        print(f"CLI failed rc={run.returncode}; see L{args.length}.cli.log", file=sys.stderr)
        return 1

    after = get_json(f"{base}/debug/metrics")
    (out / f"L{args.length}.metrics.before.json").write_text(json.dumps(before, indent=1))
    (out / f"L{args.length}.metrics.after.json").write_text(json.dumps(after, indent=1))

    js = json.loads((out / f"L{args.length}.json").read_text())
    pts = js.get("results") or []
    def avg(key):
        vals = [float(p[key]) for p in pts if p.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")
    ttft = avg("avg_ttft_ms")
    pref = avg("avg_prefill_speed")
    dec = avg("avg_output_speed")
    prompt_tok = avg("avg_prompt_tokens")
    out_tok = avg("avg_output_tokens")

    a, b = rows(before), rows(after)
    def c(tag): return diff(a, b, tag, "counter")
    rounds = c("spec.rounds")
    accepted = c("spec.accepted")
    tokens = c("spec.tokens")
    hist = {f"m{i}": c(f"spec.m{i}") for i in range(8)}
    depth = max([1] + [i for i in range(1, 8) if hist[f"m{i}"] > 0])
    pos_acc = []
    for i in range(depth):
        pos_acc.append(sum(hist[f"m{j}"] for j in range(i + 1, depth + 1)) / rounds if rounds else float("nan"))

    row = {
        "L": args.length,
        "prompt_tok": round(prompt_tok, 1),
        "out_tok": round(out_tok, 1),
        "ttft_ms": round(ttft, 1),
        "prefill_tps": round(pref, 1),
        "decode_tps": round(dec, 2),
        "rounds": rounds,
        "accepted": accepted,
        "tokens": tokens,
        "AL": round(accepted / rounds, 3) if rounds else float("nan"),
        "tokens_per_round": round(tokens / rounds, 3) if rounds else float("nan"),
        **hist,
        "pos_acc": ",".join(f"{x:.3f}" for x in pos_acc),
        "round_ms": round(avg_ms(a, b, "spec.round"), 2),
        "verify_ms": round(avg_ms(a, b, "spec.verify"), 2),
        "propose_ms": round(avg_ms(a, b, "spec.propose"), 2),
        "propose_read_ms": round(avg_ms(a, b, "spec.propose.read"), 2),
        "propose_step_ms": round(avg_ms(a, b, "spec.propose.step"), 2),
        "snap_ms": round(avg_ms(a, b, "spec.snap"), 2),
        "fold_ms": round(avg_ms(a, b, "spec.fold"), 2),
        "extend_ms": round(avg_ms(a, b, "spec.extend"), 2),
        "extend_n": diff(a, b, "spec.extend", "count"),
        "wall_s": round(time.time() - t0, 1),
    }
    csv_path = out / "curve.csv"
    fields = list(row)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
