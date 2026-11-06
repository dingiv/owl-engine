#!/usr/bin/env python3
"""**双会话并发门**（最常见形态：两个 agent 会话交替/交错跑）。

覆盖四件事（用户 2026-11-06 指定）：
  ① 两个并发请求下的 **KV 前缀命中**（同会话重发 / 跨会话复用）；
  ② **KV 调度**：池压满时是否把**另一会话的缓存链**下沉到宿主；
  ③ **召回**：被下沉的链在下一轮还能不能拿来用（不是"有缓存但用不上"）；
  ④ 并发下**不失败**（500 / 准入拒绝 / 命中塌到 0）。

场景（A/B 两个独立 session，内容随机且每次运行唯一）：
  P0 cold        A、B 同时首发（真并发）
  P1 repeat      A、B 同时重发同一前缀        ⇒ 命中应 ≥ 0.90
  P2 cross       A 发 B 的内容（B 同时重发）  ⇒ 跨会话命中 ≥ 0.85
  P3 grow        A/B 交替增长（各 +grow tok）  ⇒ 每轮命中 ≈ 旧前缀/新前缀
                 —— A 增长时 B 空闲 ⇒ 池紧时 B 的链应被下沉宿主
  P4 recall      被下沉的会话重发当前前缀       ⇒ 命中仍 ≥ 0.85（召回成功）
  P5 final       A、B 同时重发                  ⇒ 命中仍 ≥ 0.90

前置（2 并发档，2026-11-06 实测）：
  · `pool.snap_max ≥ 6`（3 会丢空闲会话的尾锚点 ⇒ 下一轮复用被裁到上一个快照；
    单并发 2 格够用）；
  · 要测**磁盘**层：`--set pool.host_cache_mb` 小到装不下整条链是**触发不了**磁盘的
    —— store 的轮一只把**整链装得进宿主**的链下沉（`hooks.rs` 的 `span > host_kv_room
    ⇒ skip`），轮 1.5 只搬**已在宿主**的链；链比宿主预算大时磁盘不可达，会在起 turn
    处 500。生产档（host 10G）链普遍装得下，故 disk 通常只在宿主满后才被使用。

口径：
  · prompt 用随机词池生成（不重复文本；AGENTS 测量纪律）；
  · 命中 = usage.prompt_tokens_details.cached_tokens / prompt_tokens；
  · 每请求抓一次 /debug/vram（cached/host/free）作为现场；
  · "有缓存但用不上" = 复用查询显示缓存块 > 0，而该请求命中 < 0.5。

用法：
  uv run python tools/fixture/two_session_concurrency.py --port 8136
  uv run python tools/fixture/two_session_concurrency.py --port 8136 \
      --base-tokens 80000 --grow-tokens 10000 --max-rounds 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict

WORDS = (
    "amber basin cobalt delta ember fjord garnet harbor iris jasper kelp lumen "
    "meadow nimbus onyx prism quartz raven sable tundra umber vapor willow xenon "
    "yarrow zephyr anchor beacon cinder drift echo fable glacier hollow indigo "
    "juniper kestrel lantern marble nectar orbit pebble quiver ripple summit "
    "thistle upland vertex whisper yonder zinnia arbor bramble cedar dune ember "
    "fern grove heather inlet jetty knoll lagoon moss nook orchard pine quill "
    "reed stone trail vale wren yew zenith alder birch cliff dew elm frost "
    "glen haze isle juniper kite leaf mist nest oak path quay root sage tide "
    "vine wave yard zinc acorn bloom cloud creek dawn dusk field grain hill"
).split()


def make_words(rng: random.Random, n: int) -> list[str]:
    return [rng.choice(WORDS) for _ in range(n)]


def make_text(rng: random.Random, tokens: int, nonce: str) -> str:
    # 实测随机短词 ≈ 1.4 tok/词（AGENTS 测量纪律）；首句带唯一 nonce 防跨运行命中。
    words = [f"run{nonce}"] + make_words(rng, max(1, int(tokens / 1.4)))
    return " ".join(words)


def extend_text(text: str, rng: random.Random, tokens: int) -> str:
    return text + " " + " ".join(make_words(rng, max(1, int(tokens / 1.4))))


@dataclass
class Rec:
    phase: str
    session: str
    label: str
    status: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    wall_s: float = 0.0
    error: str = ""
    vram: dict = field(default_factory=dict)

    @property
    def hit(self) -> float:
        return (self.cached_tokens / self.prompt_tokens) if self.prompt_tokens else 0.0


def http_json(url: str, body: dict | None = None, timeout: float = 900.0) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", "ignore"))
        except Exception:
            return e.code, {}
    except Exception as e:  # 连接失败/超时
        return 0, {"error": {"message": f"{type(e).__name__}: {e}"}}


def vram_snapshot(port: int) -> dict:
    code, d = http_json(f"http://127.0.0.1:{port}/debug/vram", timeout=10)
    if code != 200:
        return {}
    kv = d.get("kv") or {}
    faces = d.get("faces") or {}
    tiers = d.get("tiers") or {}
    disk = d.get("disk") or {}
    dkv = disk.get("kv") or {}
    dsnap = disk.get("snap") or {}
    return {
        "free": kv.get("slots_free"),
        "cached": kv.get("cached_blocks"),
        "host": kv.get("host_blocks"),
        "hits": kv.get("hits"),
        "misses": kv.get("misses"),
        "evicted": kv.get("evicted"),
        "host_used": tiers.get("host_used"),
        "active": faces.get("active_turns"),
        # 磁盘两层（`None` / 未接 = 0）
        "disk": int(dkv.get("entries") or 0) + int(dsnap.get("entries") or 0),
        "disk_kv": int(dkv.get("entries") or 0),
        "disk_snap": int(dsnap.get("entries") or 0),
        "disk_used": int(dkv.get("used_bytes") or 0) + int(dsnap.get("used_bytes") or 0),
    }


def chat(port: int, model: str, session: str, text: str, max_new: int, phase: str, label: str) -> Rec:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_new,
        "temperature": 0,
        "stream": False,
        "session": session,
    }
    t0 = time.time()
    code, d = http_json(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout=1800
    )
    wall = time.time() - t0
    rec = Rec(phase=phase, session=session, label=label, status=code, wall_s=wall)
    usage = (d.get("usage") or {}) if isinstance(d, dict) else {}
    rec.prompt_tokens = int(usage.get("prompt_tokens") or 0)
    rec.cached_tokens = int(
        ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    )
    rec.completion_tokens = int(usage.get("completion_tokens") or 0)
    if code != 200:
        rec.error = str((d.get("error") or {}).get("message") or d)[:300]
    rec.vram = vram_snapshot(port)
    return rec


def run_pair(fn_a, fn_b) -> tuple[Rec, Rec]:
    out: dict[str, Rec] = {}

    def wrap(k, fn):
        out[k] = fn()

    ta = threading.Thread(target=wrap, args=("a", fn_a))
    tb = threading.Thread(target=wrap, args=("b", fn_b))
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    return out["a"], out["b"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8136)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--base-tokens", type=int, default=80000)
    ap.add_argument("--grow-tokens", type=int, default=10000)
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="perf-runs")
    ap.add_argument(
        "--expect-disk",
        action="store_true",
        help="继续增长直到磁盘层出现条目（验证 disk 溢写 + 召回），再做召回检查",
    )
    args = ap.parse_args()

    seed = args.seed or int(time.time())
    nonce = f"{seed:x}"[-6:]
    rng_a = random.Random(seed * 2 + 1)
    rng_b = random.Random(seed * 2 + 2)
    rng_g = random.Random(seed * 2 + 3)

    text_a = make_text(rng_a, args.base_tokens, f"a{nonce}")
    text_b = make_text(rng_b, args.base_tokens, f"b{nonce}")

    recs: list[Rec] = []
    failures: list[str] = []

    def note(recs_: list[Rec], where: str):
        for r in recs_:
            recs.append(r)
            if r.status != 200:
                failures.append(f"{where}: HTTP {r.status} {r.error[:160]}")
            print(
                f"[{r.phase:7s}] {r.session:6s} {r.label:6s} "
                f"prompt={r.prompt_tokens:6d} cached={r.cached_tokens:6d} "
                f"hit={r.hit:5.1%} wall={r.wall_s:7.2f}s "
                f"free={r.vram.get('free')} host={r.vram.get('host')} "
                f"disk_kv={r.vram.get('disk_kv')}/snap={r.vram.get('disk_snap')} "
                f"active={r.vram.get('active')}"
                + (f" err={r.error[:80]}" if r.error else "")
            )

    print(f"=== two-session concurrency (port {args.port}, seed {seed}) ===")

    # P0: 冷启动并发
    a, b = run_pair(
        lambda: chat(args.port, args.model, "conc-A", text_a, args.max_new, "P0cold", "A"),
        lambda: chat(args.port, args.model, "conc-B", text_b, args.max_new, "P0cold", "B"),
    )
    note([a, b], "P0 cold")
    cold_wall = max(a.wall_s, b.wall_s)

    # P1: 同前缀并发重发
    a, b = run_pair(
        lambda: chat(args.port, args.model, "conc-A", text_a, args.max_new, "P1rep", "A"),
        lambda: chat(args.port, args.model, "conc-B", text_b, args.max_new, "P1rep", "B"),
    )
    note([a, b], "P1 repeat")
    for r in (a, b):
        if r.hit < 0.90:
            failures.append(f"P1 repeat {r.session} 命中 {r.hit:.1%} < 90%")

    # P2: 跨会话命中（A 发 B 的内容；B 同时重发）
    a, b = run_pair(
        lambda: chat(args.port, args.model, "conc-A", text_b, args.max_new, "P2cross", "A→B"),
        lambda: chat(args.port, args.model, "conc-B", text_b, args.max_new, "P2cross", "B"),
    )
    note([a, b], "P2 cross")
    if a.hit < 0.85:
        failures.append(f"P2 cross 跨会话命中 {a.hit:.1%} < 85%")

    # P3: 交替增长（A 增长时 B 空闲；池紧时 B 的链应下沉宿主）
    # 只认 **KV 面**下沉（快照面的磁盘条目是常态，不能当"下沉发生"）
    offload_seen = bool((a.vram.get("host") or 0) > 0 or (a.vram.get("disk_kv") or 0) > 0)
    text_a_cur, text_b_cur = text_a, text_b
    recall_done = False
    for rnd in range(1, args.max_rounds + 1):
        text_a_cur = extend_text(text_a_cur, rng_g, args.grow_tokens)
        ra = chat(args.port, args.model, "conc-A", text_a_cur, args.max_new, "P3grow", f"A{rnd}")
        note([ra], f"P3 A{rnd}")
        text_b_cur = extend_text(text_b_cur, rng_g, args.grow_tokens)
        rb = chat(args.port, args.model, "conc-B", text_b_cur, args.max_new, "P3grow", f"B{rnd}")
        note([rb], f"P3 B{rnd}")
        host = rb.vram.get("host") or 0
        dsk = rb.vram.get("disk_kv") or 0
        if (host > 0 or dsk > 0) and not offload_seen:
            offload_seen = True
            print(f"[P3] 首次观察到 KV 下沉：host_blocks={host} disk_kv={dsk}")
        # 期望：每轮命中 ≈ 旧长/新长；给 10% 余量
        want_a = 1.0 - (args.grow_tokens / max(1, ra.prompt_tokens))
        want_b = 1.0 - (args.grow_tokens / max(1, rb.prompt_tokens))
        if ra.hit < want_a - 0.10:
            failures.append(
                f"P3 A{rnd} 命中 {ra.hit:.1%} < 期望 {want_a:.0%}（增长段之外应命中）"
            )
        if rb.hit < want_b - 0.10:
            failures.append(
                f"P3 B{rnd} 命中 {rb.hit:.1%} < 期望 {want_b:.0%}（增长段之外应命中）"
            )
        # 下沉到位后做一次召回验证（B 重发当前前缀）
        disk_ready = (not args.expect_disk) or dsk > 0
        if offload_seen and disk_ready and not recall_done:
            rr = chat(args.port, args.model, "conc-B", text_b_cur, args.max_new, "P4recall", "B")
            note([rr], "P4 recall")
            recall_done = True
            if rr.hit < 0.85:
                failures.append(
                    f"P4 recall 命中 {rr.hit:.1%} < 85%（宿主链未召回）"
                )
            if rr.wall_s > cold_wall * 0.8:
                failures.append(
                    f"P4 recall 墙钟 {rr.wall_s:.1f}s 接近冷启动 {cold_wall:.1f}s（疑似全量重算）"
                )
        if offload_seen and disk_ready and recall_done:
            break

    if not offload_seen:
        print("[P3] ⚠️ 未观察到宿主/磁盘下沉（增长量不足；可加 --base-tokens/--grow-tokens）")
    elif args.expect_disk and not any((r.vram.get("disk_kv") or 0) > 0 for r in recs):
        failures.append("--expect-disk 但全程未观察到磁盘层条目（host 未满/溢写未触发）")

    # P5: 最终并发重发
    a, b = run_pair(
        lambda: chat(args.port, args.model, "conc-A", text_a_cur, args.max_new, "P5final", "A"),
        lambda: chat(args.port, args.model, "conc-B", text_b_cur, args.max_new, "P5final", "B"),
    )
    note([a, b], "P5 final")
    for r in (a, b):
        if r.hit < 0.90:
            failures.append(f"P5 final {r.session} 命中 {r.hit:.1%} < 90%")

    # —— 异常扫描：有缓存但用不上 ——
    for r in recs:
        if r.phase == "P0cold":
            continue
        if r.hit < 0.5 and (r.vram.get("cached") or 0) > 64:
            failures.append(
                f"{r.phase} {r.session} 命中 {r.hit:.1%} 而池内仍有 "
                f"{r.vram.get('cached')} 缓存块（疑似『有 KV 用不上』）"
            )

    # —— 产出 ——
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out_root, f"two-session-concurrency-{ts}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(
            {
                "seed": seed,
                "port": args.port,
                "base_tokens": args.base_tokens,
                "grow_tokens": args.grow_tokens,
                "offload_seen": offload_seen,
                "cold_wall_s": cold_wall,
                "records": [asdict(r) for r in recs],
                "failures": failures,
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    with open(os.path.join(out_dir, "report.md"), "w") as f:
        f.write(f"# 双会话并发门 {ts}\n\n")
        f.write(f"seed={seed} port={args.port} base={args.base_tokens} grow={args.grow_tokens}\n\n")
        f.write("| phase | session | label | prompt | cached | hit | wall(s) | free | host | disk |\n")
        f.write("|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in recs:
            f.write(
                f"| {r.phase} | {r.session} | {r.label} | {r.prompt_tokens} | "
                f"{r.cached_tokens} | {r.hit:.1%} | {r.wall_s:.2f} | "
                f"{r.vram.get('free')} | {r.vram.get('host')} | {r.vram.get('disk')} |\n"
            )
        f.write("\n**failures**:\n")
        for x in failures or ["（无）"]:
            f.write(f"- {x}\n")
    print(f"\nreport: {out_dir}/report.md")
    if failures:
        print(f"❌ {len(failures)} 项失败")
        for x in failures:
            print(" -", x)
        return 1
    print("✅ 双会话并发门全绿")
    return 0


if __name__ == "__main__":
    sys.exit(main())
