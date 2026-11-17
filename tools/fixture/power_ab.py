#!/usr/bin/env python3
"""同模型单卡/TP2 decode 功耗对照（稳态平均功率）。

用法：
  python3 /tmp/owl_power_ab.py --port 8136 --log /tmp/owl-tp2.log --gpus 1,2
"""
import argparse, json, re, subprocess, threading, time, urllib.request

def sample(gpus, out, stop):
    while not stop.is_set():
        try:
            r = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,power.draw,utilization.gpu",
                 "--format=csv,noheader", "-i", gpus],
                text=True, timeout=2)
        except Exception:
            continue
        t = time.time()
        d = {}
        for line in r.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    d[int(parts[0])] = (float(parts[1].split()[0]), float(parts[2].split()[0]))
                except ValueError:
                    pass
        if d:
            out.append((t, d))
        time.sleep(0.1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--gpus", default="1")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--engine", choices=["owl","vllm"], default="owl")
    ap.add_argument("--max-tokens", type=int, default=1000)
    ap.add_argument("--prompt", default="请只输出 1 到 1000 的数字，每行一个。")
    a = ap.parse_args()

    body = {
        "model": a.model,
        "messages": [{"role": "user", "content": a.prompt}],
        "max_tokens": a.max_tokens, "temperature": 0, "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(body).encode()

    def req(timeout=120):
        r = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions",
                                   data=data, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            d = json.loads(resp.read().decode())
        return d, t0, time.time()

    # 预热（填前缀/拉时钟），不采样
    req()
    time.sleep(2.0)

    samples, stop = [], threading.Event()
    th = threading.Thread(target=sample, args=(a.gpus, samples, stop), daemon=True)
    th.start()
    d, t0, t1 = req()
    time.sleep(0.3)
    stop.set(); th.join(timeout=2)

    # 稳态窗口：请求中段、util>=50% 的样本（排除 prefill 爬坡/尾部掉电）
    lo, hi = t0 + 0.25, t1 - 0.05
    steady = [(t, x) for (t, x) in samples if lo <= t <= hi]
    per = {}
    for g in [int(x) for x in a.gpus.split(",")]:
        vals = [x[g][0] for (_, x) in steady if g in x and x[g][1] >= 50]
        per[g] = (len(vals), sum(vals) / len(vals) if vals else 0.0)
    total = sum(v[1] for v in per.values())
    all_vals = [(t, x) for (t, x) in samples if lo <= t <= hi]
    # 整请求窗（含 prefill）平均
    all_per = {}
    for g in [int(x) for x in a.gpus.split(",")]:
        vals = [x[g][0] for (_, x) in all_vals if g in x]
        all_per[g] = sum(vals) / len(vals) if vals else 0.0

    # 从服务日志抓最后一条 decode 指标
    metric = ""
    try:
        with open(a.log, errors="ignore") as f:
            for line in f:
                if "指标:" in line and "decode=" in line:
                    metric = line.strip()
    except FileNotFoundError:
        pass
    m = re.search(r"decode=(\d+)ms/(\d+)tok=([\d.]+)t/s", metric)
    dec_ms, dec_tok, dec_tps = (int(m.group(1)), int(m.group(2)), float(m.group(3))) if m else (0, 0, 0.0)
    usage = d.get("usage") or {}
    if a.engine == "vllm" and dec_tps == 0.0:
        comp = int(usage.get("completion_tokens") or 0)
        dec_ms = int(max((t1 - t0 - 0.2) * 1000, 0))
        dec_tok = comp
        dec_tps = comp / max((t1 - t0 - 0.2), 0.01)
    print(json.dumps({
        "port": a.port, "gpus": a.gpus,
        "wall_s": round(t1 - t0, 3),
        "completion_tokens": usage.get("completion_tokens"),
        "decode_ms": dec_ms, "decode_tokens": dec_tok, "decode_tps": dec_tps,
        "steady_samples": {str(g): per[g][0] for g in per},
        "avg_power_w": {str(g): round(per[g][1], 1) for g in per},
        "total_avg_power_w": round(total, 1),
        "whole_req_avg_power_w": {str(g): round(v, 1) for g, v in all_per.items()},
        "joules_per_token": round(total / dec_tps, 3) if dec_tps else None,
        "metric": metric,
    }, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
