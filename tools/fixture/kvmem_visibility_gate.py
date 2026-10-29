#!/usr/bin/env python3
"""**KVMem 覆盖率门**（P1-3；2026-10-28）。

为什么有这道门：KVMem 稀疏档的"看得见多少"= `workspace_per_req / ctx`。
这个比值直接决定**全文类**能力，而且**没有任何选择器能替代容量**
（实测：72k 文档 / 4096 工作集下，纯 top-k 与"top-k + 48 块时间序分层"
召回一模一样 —— 因为 48/2240 = 2%，采样间隔 1472 tok 远大于一个标题块）。

## 两道子门

1. **可见性门（主）** —— `--task sections`
   文档由 N 个 `=== 资料 <名> ===` 小节拼成，问"按顺序列出你确实读到的小节名"。
   低方差、敏感、与"全文通读"高度相关；72k 实测 3/7 → 5/7 → 6/7（ws 4k/8k/12k）。
2. **检索门（辅）** —— `--task needle`
   单条事实针放在 15/50/85% 深度，MC 问法。**方差大**（每次采样 ≈ 命中率），
   必须多 seeds 才能当门用；适合"确认没坏"而非"精确定量"。

## 用法

    python3 tools/kvmem_visibility_gate.py --task sections --ws 4096 8192 16384
    python3 tools/kvmem_visibility_gate.py --task needle --ws 4096 --seeds 3
    python3 tools/kvmem_visibility_gate.py --task sections --ws 4096 --strat 48

前置：仓库根已构建 `target/release/server`；`GPU` 必须是**空闲卡**的 UUID；
端口 8140 空闲。**一次服务只测一发**（第 2 发会命中前缀缓存，走另一条路径）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import subprocess
import time
import urllib.request

OWL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8140
GPU = os.environ.get("KVMEM_GPU", "GPU-89dffb90-d810-7ce7-f649-60d781508f21")

DOCS = [
    "roadmap.local/26-10-27-KVMem-a8-重叠与视图口径.md",
    "roadmap.local/26-10-27-KVMem-缺陷清单与后续路线.md",
    "roadmap.local/26-10-19-store回收完全体-立案.md",
    "roadmap.local/26-10-19-decode稳定性与GDN小刀.md",
    "roadmap.local/26-10-17-模块审查-次轮.md",
    "roadmap.local/26-10-17-模块审查-四轮.md",
    "roadmap.local/代差分析与演进路线.md",
    "roadmap.local/优化候选总表-2026-10-14.md",
    "roadmap.local/spec长ctx衰减-立项设计与施工.md",
    "roadmap.local/26-10-15-prefill-graph-分支分析与合并预案.md",
]


# ── 服务生命周期（**只能靠端口取 PID**：pkill -f 会自杀/误杀） ──────────────
def stop() -> None:
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{PORT}" in line:
            m = re.search(r"pid=(\d+)", line)
            if m:
                os.kill(int(m.group(1)), signal.SIGINT)
                time.sleep(6)
                return


def start(sets: list[str], log: str) -> bool:
    stop()
    cmd = [
        "./target/release/server",
        "-p",
        "profiles/measurement/qwen38_27b-3090-kvmem.toml",
        "--set",
        f'parallel.device=["{GPU}"]',
        "--set",
        "kvmem.prefill_view=always",  # 池内也走稀疏 ⇒ 覆盖率可由 ws 单独控制
        "--log-level",
        "warn",
    ]
    for s in sets:
        cmd += ["--set", s]
    with open(log, "w") as f:
        subprocess.Popen(cmd, cwd=OWL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(180):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.loads(r.read()).get("ok"):
                    return True
        except Exception:
            time.sleep(1)
    return False


def ask(msgs: list[dict], max_tokens: int) -> dict:
    body = {
        "model": "qwen3.8-27b",
        "messages": msgs,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "reasoning_effort": "none",  # 本模型关思考后偶尔写草稿 ⇒ 门里统一关掉
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


# ── 语料 ─────────────────────────────────────────────────────────────────
def corpus(chars: int) -> str:
    buf = ""
    for p in DOCS:
        with open(os.path.join(OWL, p), encoding="utf-8") as f:
            buf += f"\n\n=== 资料 {p.rsplit('/', 1)[-1]} ===\n" + f.read()
        if len(buf) >= chars:
            break
    return buf[:chars]


SECT_Q = (
    "以上资料由若干小节组成，每节以「=== 资料 <文件名> ===」开头。\n"
    "请按出现顺序列出你**确实读到**的每一个「=== 资料 X ===」里的文件名。\n"
    "不要猜、不要补全；只输出文件名，一行一个。"
)


def task_sections(doc: str) -> tuple[int, int, str]:
    truth = re.findall(r"=== 资料 (\S+?) ===", doc)
    j = ask([{"role": "user", "content": doc + "\n\n" + SECT_Q}], 400)
    txt = j["choices"][0]["message"]["content"] or ""
    hit = sum(1 for t in truth if t in txt)
    return hit, len(truth), txt


NEEDLE = "\n\n【事实 X-9】本项目的密钥轮换周期是 四十五天。\n\n"
NEEDLE_Q = (
    "资料中有一条【事实 X-9】。它的密钥轮换周期是多久？\n"
    "(A) 四十五天  (B) 九十天  (C) 十五天\n只输出一个字母。"
)


def task_needle(doc: str, seeds: int) -> tuple[int, int, str]:
    """针放在 seed 决定的深度；**每次都要新文档**（否则命中前缀缓存）。"""
    hit = 0
    detail = []
    for s in range(seeds):
        depth = (s + 1) / (seeds + 1)
        i = doc.find("\n\n", int(len(doc) * depth))
        i = len(doc) if i < 0 else i
        d = doc[:i] + NEEDLE + doc[i:]
        j = ask([{"role": "user", "content": d + "\n\n" + NEEDLE_Q}], 24)
        txt = (j["choices"][0]["message"]["content"] or "").strip()
        ok = txt.startswith("A")
        hit += 1 if ok else 0
        detail.append(f"{depth:.0%}:{'✅' if ok else '❌'}")
    return hit, seeds, " ".join(detail)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["sections", "needle"], default="sections")
    ap.add_argument("--ws", type=int, nargs="+", default=[4096, 8192, 12288, 16384])
    ap.add_argument("--strat", type=int, nargs="*", default=[0], help="select_stratified 取值(0=关)")
    ap.add_argument("--chars", type=int, default=140000, help="文档字符数(≈0.52 tok/字符)")
    ap.add_argument("--seeds", type=int, default=3, help="needle 门的采样数(方差大,建议 ≥3)")
    a = ap.parse_args()

    doc = corpus(a.chars)
    print(f"语料 {len(doc)} 字符(≈{int(len(doc) * 0.52)} tok)；服务 {GPU}；task={a.task}\n", flush=True)
    rows = []
    for ws in a.ws:
        for st in a.strat:
            sets = [f"kvmem.workspace_per_req={ws}"]
            if st:
                sets.append(f"kvmem.select_stratified={st}")
            label = f"ws={ws:6d} strat={st:3d}"
            if not start(sets, f"/tmp/kv_gate_{ws}_{st}.log"):
                print(f"{label}: BOOT FAIL(见 /tmp/kv_gate_{ws}_{st}.log)")
                continue
            try:
                if a.task == "sections":
                    hit, tot, txt = task_sections(doc)
                else:
                    hit, tot, txt = task_needle(doc, a.seeds)
            finally:
                stop()
            ratio = ws / max(1, int(len(doc) * 0.52))
            rows.append((ws, st, hit, tot, ratio))
            print(f"{label} | 命中 {hit}/{tot} = {100 * hit / max(1, tot):.0f}% | 物化 {ratio * 100:.1f}% | {txt[:60]}", flush=True)

    print("\n| ws | strat | 命中 | 物化比 |")
    print("|---|---|---|---|")
    for ws, st, hit, tot, ratio in rows:
        print(f"| {ws} | {st} | {hit}/{tot} ({100 * hit / max(1, tot):.0f}%) | {ratio * 100:.1f}% |")


if __name__ == "__main__":
    main()
