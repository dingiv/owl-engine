#!/usr/bin/env python3
"""显存时间线采样(2026-10-22 · metrics-optimize)。

目的:把"显存暴增"拆成 **boot 稳态 / prefill 峰值 / decode 稳态 / 请求后残留**。
`/debug/vram` 是 actor 旁路(生成期会等 turn 结束)⇒ 时间线用 `nvidia-smi` 采样,
`/debug/vram` 只在每个请求前后(空闲点)取快照。

用法(先按目标档起服):
  python3 tools/vram_timeline.py --url http://127.0.0.1:8140 \
      --gpu-index 1 --lengths 2048,16384,65536 --out perf-runs/.../vram-timeline
"""
import argparse
import json
import pathlib
import subprocess
import threading
import time
import urllib.request

PROSE_WORDS = """a the and of to in is it that for with as on by from this be are or not
we you they can will if all one time world life work data model token text idea mind
story light space future human system simple clear reason change value truth""".split()
PROSE_SUFFIX = (
    "Based on the words above, write a short philosophical essay discussing "
    "the meaning of existence, the nature of consciousness, and humanity's "
    "place in the universe. Use clear, coherent sentences."
)


def post_json(url, payload, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get_json(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def smi_used(idx):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(idx)],
        text=True,
    )
    return int(out.strip().splitlines()[0])


class Sampler(threading.Thread):
    def __init__(self, idx, interval=0.25):
        super().__init__(daemon=True)
        self.idx, self.interval = idx, interval
        self.stop = threading.Event()
        self.series = []

    def run(self):
        while not self.stop.is_set():
            try:
                self.series.append((time.time(), smi_used(self.idx)))
            except Exception:  # 诊断采样:瞬时失败不中断
                pass
            self.stop.wait(self.interval)


def vram(base):
    d = get_json(f"{base}/debug/vram")
    pools = d.get("pools") or [{}]
    p = pools[0] if pools else {}
    cells = {c["kind"]: c["bytes"] for c in d.get("tiers", {}).get("cells", [])}
    return {
        "used_mib": round(d["tiers"]["gpu_used"] / 2**20, 1),
        "pool_used_mib": round(p.get("used_mib", 0.0), 1),
        "pool_reserved_mib": round(p.get("reserved_mib", 0.0), 1),
        "pool_cached_mib": round(p.get("cached_mib", 0.0), 1),
        "cells_mib": {k: round(v / 2**20, 1) for k, v in cells.items()},
        "kv_slots_free": d.get("kv", {}).get("slots_free"),
        "kv_slots_total": d.get("kv", {}).get("slots_total"),
    }


def make_prompt(base, target, seed, rng):
    words = [rng.choice(PROSE_WORDS) for _ in range(max(1, target))]
    text = " ".join(words) + "\n" + PROSE_SUFFIX
    for _ in range(3):
        n = int(post_json(f"{base}/tokenize", {"prompt": text, "mode": "raw"}, timeout=120)["count"])
        if abs(n - target) <= max(4, target // 30):
            break
        words = [rng.choice(PROSE_WORDS) for _ in range(max(1, int(len(words) * target / max(n, 1))))]
        text = " ".join(words) + "\n" + PROSE_SUFFIX
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8140")
    ap.add_argument("--gpu-index", type=int, required=True)
    ap.add_argument("--lengths", default="2048,16384,65536")
    ap.add_argument("--output-length", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261022)
    args = ap.parse_args()
    import random
    base = args.url.rstrip("/")
    out = pathlib.Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]

    points = []
    boot = {"t": time.time(), "smi": smi_used(args.gpu_index), "vram": vram(base)}
    print("boot", boot, flush=True)
    for L in lengths:
        text = make_prompt(base, L, args.seed, rng)
        before = {"t": time.time(), "smi": smi_used(args.gpu_index), "vram": vram(base)}
        s = Sampler(args.gpu_index)
        s.start()
        t0 = time.time()
        resp = post_json(f"{base}/v1/chat/completions", {
            "model": "qwen3.8-27b",
            "messages": [{"role": "user", "content": text}],
            "max_tokens": args.output_length,
            "temperature": 0,
            "ignore_eos": True,
            "stream": False,
        })
        t1 = time.time()
        time.sleep(0.8)
        s.stop.set(); s.join()
        after = {"t": time.time(), "smi": smi_used(args.gpu_index), "vram": vram(base)}
        series = [(round(t - t0, 3), v) for t, v in s.series]
        rec = {
            "target": L,
            "prompt_tokens": (resp.get("usage") or {}).get("prompt_tokens"),
            "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
            "wall_s": round(t1 - t0, 3),
            "before": before, "after": after,
            "series": series,
            "peak_smi": max((v for _, v in series), default=None),
            "min_smi": min((v for _, v in series), default=None),
        }
        points.append(rec)
        print(f"L={L} prompt={rec['prompt_tokens']} wall={rec['wall_s']}s "
              f"before={before['smi']} peak={rec['peak_smi']} after={after['smi']} "
              f"pool_used={after['vram']['pool_used_mib']} cached={after['vram']['pool_cached_mib']}",
              flush=True)

    (out / "timeline.json").write_text(json.dumps({"boot": boot, "points": points}, indent=1))
    print("saved", out / "timeline.json", flush=True)


if __name__ == "__main__":
    main()
