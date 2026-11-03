#!/usr/bin/env python3
"""spec 的 AL / 吞吐 A/B 探针（2026-11-03；DFlash2 AL 升级专卷）。

口径（照 `roadmap.local/26-11-03-ddtree-AL升级-交接.md` §5，勿改）：

  · AL = Δ`spec.tokens` / Δ`spec.rounds`（引擎内计数器；`spec.tokens` 每轮
    记 **m+1**，即"每轮推进的 token 数"，含 bonus）；
  · 吞吐 = `usage.completion_tokens` / 墙钟（非流式，一次请求一测）；
  · 每次请求前后各拍一次 `/debug/metrics/spec` 做差分 ⇒ **每轮一个点**，
    多轮取均值（单轮噪声大，AL 差 0.1 需 ≥5 轮才看得出）；
  · 三条负载写死在本文（创作 / 复制 / 数学-代码），同 prompt 才能跨臂比；
  · `enable_thinking=false`；temp 缺省 0.7 / top_p 0.95。

用法：

    python3 tools/fixture/spec_al_ab.py --url http://127.0.0.1:8135 \\
        --label base --rounds 5 --out /tmp/spec-al.jsonl

    python3 tools/fixture/spec_al_ab.py --loads creative --rounds 2   # 冒烟
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

# 三条负载（§5 的原文；改动即作废跨臂可比性）
LOADS = {
    "creative": "写一段关于秋天落叶与旧书店的散文，四百字左右，语言要具体、有画面感。",
    "copy": "把下面这句话原样重复 30 遍，用换行分隔：秋天的风把旧书店的招牌吹得吱呀作响。",
    "math": (
        "小明有 3 个苹果，又买了 5 个，然后吃掉 2 个；小红有 7 个苹果，"
        "给了小明 3 个，又买了 4 个。两人现在一共多少个苹果？请逐步推理并给出答案。"
    ),
}


def get_json(url, timeout=30.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def post_json(url, payload, timeout=600.0):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def snap(base):
    """本轮快照：tag -> row（抓不到就响亮失败——A/B 口径不许静默丢指标）。"""
    try:
        js = get_json(f"{base}/debug/metrics/spec")
    except (urllib.error.URLError, TimeoutError) as e:
        raise SystemExit(f"metrics 抓取失败: {e}")
    return {r["tag"]: r for r in js.get("rows", [])}


def counter(rows, tag):
    return float((rows.get(tag) or {}).get("counter") or 0)


def total_ms(rows, tag):
    return float((rows.get(tag) or {}).get("total_ms") or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8135")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--label", default="arm")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--loads", default="creative,copy,math")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--topp", type=float, default=0.95)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    loads = [s.strip() for s in args.loads.split(",") if s.strip()]
    bad = [s for s in loads if s not in LOADS]
    if bad:
        raise SystemExit(f"未知负载 {bad}（可选 {list(LOADS)}）")

    print(f"# arm={args.label} url={base} rounds={args.rounds} "
          f"temp={args.temp} top_p={args.topp} max_tokens={args.max_tokens}")
    print(f"{'load':<9}{'rnd':>4}{'tokens':>7}{'rounds':>7}{'AL':>7}"
          f"{'tok/s':>8}{'verify':>8}{'propose':>8}{'round_ms':>9}")

    summary = {}
    for load in loads:
        als, tps = [], []
        for i in range(args.rounds):
            a = snap(base)
            t0 = time.monotonic()
            js = post_json(f"{base}/v1/chat/completions", {
                "model": args.model,
                "messages": [{"role": "user", "content": LOADS[load]}],
                "temperature": args.temp,
                "top_p": args.topp,
                "max_tokens": args.max_tokens,
                "stream": False,
                "chat_template_kwargs": {"enable_thinking": False},
            })
            wall = time.monotonic() - t0
            b = snap(base)
            out_tok = float((js.get("usage") or {}).get("completion_tokens") or 0)
            dtok = counter(b, "spec.tokens") - counter(a, "spec.tokens")
            drnd = counter(b, "spec.rounds") - counter(a, "spec.rounds")
            al = dtok / drnd if drnd > 0 else float("nan")
            tps_r = out_tok / wall if wall > 0 else float("nan")
            dv = total_ms(b, "spec.verify") - total_ms(a, "spec.verify")
            dp = total_ms(b, "spec.propose") - total_ms(a, "spec.propose")
            dr = total_ms(b, "spec.round") - total_ms(a, "spec.round")
            print(f"{load:<9}{i:>4}{int(dtok):>7}{int(drnd):>7}{al:>7.3f}"
                  f"{tps_r:>8.1f}{dv / max(drnd, 1):>8.1f}{dp / max(drnd, 1):>8.1f}"
                  f"{dr / max(drnd, 1):>9.1f}")
            als.append(al)
            tps.append(tps_r)
            if args.out:
                with open(args.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "label": args.label, "load": load, "round": i,
                        "tokens": dtok, "spec_rounds": drnd, "al": al,
                        "tok_s": tps_r, "wall_s": wall,
                        "verify_ms_per_round": dv / max(drnd, 1),
                        "propose_ms_per_round": dp / max(drnd, 1),
                        "round_ms": dr / max(drnd, 1),
                        "completion_tokens": out_tok,
                    }, ensure_ascii=False) + "\n")
        m = sum(als) / len(als)
        t = sum(tps) / len(tps)
        summary[load] = (m, t)
        print(f"{'  MEAN':<9}{'':>4}{'':>7}{'':>7}{m:>7.3f}{t:>8.1f}")

    print("\n# summary (AL / tok/s)")
    for load, (m, t) in summary.items():
        print(f"{load:<9}AL={m:.3f}  tok/s={t:.1f}")


if __name__ == "__main__":
    sys.exit(main())
