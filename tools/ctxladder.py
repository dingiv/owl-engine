#!/usr/bin/env python3
"""长 ctx 归因驱动(perf 会话;2026-10-15)。

逐 ctx 点跑 `llm_speedtest`(黑盒曲线),再抓引擎侧账(perf epoch / 发射账),
全部落到一个 run 目录 —— 「同一 prompt、同一时刻」的两套仪器口径。

用法(在 packages/worktrees/owl/perf 下):
    python3 tools/ctxladder.py --out perf-runs/ctxladder-1 \
        --lengths 512,2048,8192,16384,32768 [--rounds 1] [--url http://127.0.0.1:8138]

产出(每点):
    L{L}.csv / L{L}.json          ← llm_speedtest 原始(含逐请求时间戳)
    L{L}.perf.json / .perf.txt    ← /debug/perf/epoch{,/json}(本轮 tag 树)
    L{L}.launches.json            ← /debug/launches/4000(发射序;含 kind/核名)
    summary.tsv                   ← 汇总一行一点(ctx/prefill/decode/TTFT/相位)
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

SPEEDTEST_DIR = pathlib.Path("./packages/llm_speedtest")


def get(url: str, timeout: float = 60.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def try_get(url: str, timeout: float = 60.0) -> str:
    try:
        return get(url, timeout)
    except Exception as e:  # noqa: BLE001 - 诊断脚本:抓不到就记下,不中断扫描
        return json.dumps({"error": str(e), "url": url})


def row_of(perf_json: dict, tag: str):
    for r in perf_json.get("rows", []):
        if r["tag"] == tag:
            return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8138")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lengths", default="512,2048,8192,16384,32768")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--output-length", type=int, default=256)
    ap.add_argument("--timeout-ms", type=int, default=900_000)
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    out = pathlib.Path(args.out).resolve()  # 绝对化:cli 子进程 cwd = SPEEDTEST_DIR
    out.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    base = args.url.rstrip("/")
    summary = out / "summary.tsv"

    with summary.open("w") as f:
        f.write(
            "ctx\tttft_ms\tprefill_tps\tdecode_tps\titl_ms"
            "\tphase_prefill_ms\tphase_prefill_n\tphase_spec_ms\tphase_spec_n"
            "\tprefill_per_chunk_ms\tspec_per_round_ms"
            "\tspec_round_ms\tspec_verify_read_ms\tspec_propose_ms\tspec_snap_ms\tspec_fold_ms"
            "\tsrv_dtoh_wait_ms\tsrv_dtoh_n\trounds_n\tspec_tokens\tspec_accepted\tal"
            "\twall_s\n"
        )

    for L in lengths:
        print(f"\n===== ctx={L} =====", flush=True)
        t0 = time.time()
        cmd = [
            "uv", "run", "cli",
            "--api-url", f"{base}/v1/chat/completions",
            "--model", args.model,
            "--lengths", str(L),
            "--rounds", str(args.rounds),
            "--output-length", str(args.output_length),
            "-c", "1",
            "--ignore-eos",
            "--timeout", str(args.timeout_ms),
            "--csv", str(out / f"L{L}.csv"),
            "--json", str(out / f"L{L}.json"),
            "--notes", args.notes or f"ctxladder L={L}",
        ]
        r = subprocess.run(cmd, cwd=SPEEDTEST_DIR, capture_output=True, text=True)
        (out / f"L{L}.cli.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        print(r.stdout[-1500:], flush=True)
        if r.returncode != 0:
            print(f"!! cli 退出码 {r.returncode}(详见 L{L}.cli.log)", flush=True)

        # 引擎侧账(同刻快照)
        pj = try_get(f"{base}/debug/perf/epoch/json")
        (out / f"L{L}.perf.json").write_text(pj)
        (out / f"L{L}.perf.txt").write_text(try_get(f"{base}/debug/perf/epoch"))
        (out / f"L{L}.launches.json").write_text(try_get(f"{base}/debug/launches/4000"))

        # 汇总一行(黑盒 JSON + 相位账)
        # llm_speedtest JSON 真实键(2026-10-15 对档:"results" 每 rounds 一行):
        #   avg_prefill_speed / avg_output_speed / avg_ttft_ms /
        #   concurrent_details[0].itl_stats.mean
        ttft = pref = dec = itl = float("nan")
        try:
            js = json.loads((out / f"L{L}.json").read_text())
            pts = js.get("results") or []
            if pts:
                def avg(key):
                    vals = [float(p[key]) for p in pts if p.get(key) is not None]
                    return sum(vals) / len(vals) if vals else float("nan")

                ttft, pref, dec = avg("avg_ttft_ms"), avg("avg_prefill_speed"), avg("avg_output_speed")
                itls = [
                    float(p["concurrent_details"][0]["itl_stats"]["mean"])
                    for p in pts
                    if p.get("concurrent_details")
                    and p["concurrent_details"][0].get("itl_stats")
                    and p["concurrent_details"][0]["itl_stats"].get("mean") is not None
                ]
                if itls:
                    itl = sum(itls) / len(itls)
                print(
                    f"  → prefill {pref:.1f} t/s · decode {dec:.1f} t/s · "
                    f"TTFT {ttft:.0f} ms · ITL {itl:.2f} ms(n={len(pts)})",
                    flush=True,
                )
        except Exception as e:  # noqa: BLE001
            print(f"(解析 cli json 失败:{e})", flush=True)

        ph_pf = ph_dc = float("nan")
        n_pf = n_dc = 0
        pjd: dict = {}
        tags: dict = {}

        def grab(name: str):
            return row_of(pjd, name) or {}

        try:
            pjd = json.loads(pj)
            r1, r2 = row_of(pjd, "engine.phase.prefill"), row_of(pjd, "engine.phase.spec")
            if r1:
                ph_pf, n_pf = r1["total_ms"], r1["count"]
            if r2:
                ph_dc, n_dc = r2["total_ms"], r2["count"]
            for t in (
                "spec.round", "spec.verify.read", "spec.propose", "spec.snap",
                "spec.fold", "srv.wait.dtoh",
            ):
                tags[t] = grab(t)
        except Exception:  # noqa: BLE001
            pass

        def avg_ms(name: str) -> float:
            return float(tags.get(name, {}).get("avg_ms", "nan"))

        def cnt(name: str) -> int:
            return int(tags.get(name, {}).get("count", 0) or 0)

        tok_c = float(grab("spec.tokens").get("counter", "nan") or "nan")
        acc_c = float(grab("spec.accepted").get("counter", "nan") or "nan")
        rnd_c = float(grab("spec.rounds").get("counter", "nan") or "nan")
        # AL(接受长度)= 接受草稿数 / 轮数(均值;含 bonus 的 emit 长度 = AL+1)
        al = (acc_c / rnd_c) if (rnd_c and rnd_c == rnd_c and rnd_c > 0) else float("nan")

        with summary.open("a") as f:
            f.write(
                f"{L}\t{ttft:.2f}\t{pref:.2f}\t{dec:.2f}\t{itl:.2f}"
                f"\t{ph_pf:.1f}\t{n_pf}\t{ph_dc:.1f}\t{n_dc}"
                f"\t{ph_pf / n_pf if n_pf else float('nan'):.2f}"
                f"\t{ph_dc / n_dc if n_dc else float('nan'):.2f}"
                f"\t{avg_ms('spec.round'):.2f}\t{avg_ms('spec.verify.read'):.2f}"
                f"\t{avg_ms('spec.propose'):.2f}\t{avg_ms('spec.snap'):.2f}"
                f"\t{avg_ms('spec.fold'):.2f}\t{avg_ms('srv.wait.dtoh'):.2f}\t{cnt('srv.wait.dtoh')}"
                f"\t{cnt('spec.round')}\t{tok_c:.0f}\t{acc_c:.0f}\t{al:.3f}"                f"\t{time.time() - t0:.1f}\n"
            )

    (out / "boot.txt").write_text(
        f"url={base}\nlengths={lengths}\nrounds={args.rounds}\nout_len={args.output_length}\n"
        f"notes={args.notes}\nts={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    print(f"\n完成 ⇒ {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
