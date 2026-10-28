#!/usr/bin/env python3
"""KVMem 材料可见性门 v2：**随机码标记**（防编造）。

v1 的两个坑（都踩过）：
- 只问"列出你看到的" ⇒ 模型偷懒只报 5/20（dense 也如此）⇒ 低估;
- 告知"一共 20 条，按序全列" ⇒ 模型**编造**缺失的（71.5% 物化也报 20/20）⇒ 高估。

v2:插 20 条 `【ZC-XXX】`（字母随机、顺序打乱），**不给总数**，明确"不确定就不要写"，
问哪些码在资料里出现过；评分 = recall/precision（编造 = precision 掉）。

用法: kv_gate.py <ws1> [ws2 ...]     MODE=always|on-hole  REPS=3
"""
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import urllib.request

OWL = "/home/div/Documents/codes/packages/owl"
GPU = "GPU-89dffb90-d810-7ce7-f649-60d781508f21"
PORT = 8140
LOG = "/tmp/kv_gate.log"
DOC_CHARS = int(os.environ.get("DOC_CHARS", "42000"))
N_MARK = int(os.environ.get("N_MARK", "20"))
MODE = os.environ.get("MODE", "always")
REPS = int(os.environ.get("REPS", "1"))
SEED = int(os.environ.get("SEED", "20261028"))

DOCS = [
    "roadmap.local/26-10-27-KVMem-a8-重叠与视图口径.md",
    "roadmap.local/26-10-27-KVMem-缺陷清单与后续路线.md",
    "roadmap.local/26-10-19-store回收完全体-立案.md",
    "roadmap.local/26-10-19-decode稳定性与GDN小刀.md",
    "roadmap.local/26-10-17-模块审查-次轮.md",
]


def stop():
    out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{PORT}" in line:
            m = re.search(r"pid=(\d+)", line)
            if m:
                os.kill(int(m.group(1)), signal.SIGINT)
                time.sleep(6)
                return


def start(ws: int):
    stop()
    cmd = [
        "./target/release/server", "-p", "profiles/measurement/qwen38_27b-3090-kvmem.toml",
        "--set", f"parallel.device=[\"{GPU}\"]",
        "--set", f"kvmem.prefill_view={MODE}",
        "--set", f"kvmem.workspace_per_req={ws}",
        "--log-level", "warn",
    ]
    with open(LOG, "w") as f:
        subprocess.Popen(cmd, cwd=OWL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(180):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.loads(r.read()).get("ok"):
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError("服务未就绪")


def build(seed: int):
    rnd = random.Random(seed)
    codes = set()
    while len(codes) < N_MARK:
        codes.add("".join(rnd.choice("BCDFGHJKLMNPQRSTVWXZ") for _ in range(3)))
    codes = sorted(codes)
    rnd.shuffle(codes)
    buf, n = [], 0
    for p in DOCS:
        with open(os.path.join(OWL, p), encoding="utf-8") as f:
            t = f.read()
        buf.append(f"\n\n=== 资料 {p.rsplit('/', 1)[-1]} ===\n" + t)
        n += len(t)
        if n >= DOC_CHARS:
            break
    doc = "".join(buf)[:DOC_CHARS]
    pieces, last = [], 0
    for i, code in enumerate(codes):
        target = len(doc) * (i + 1) // (len(codes) + 1)
        cut = doc.find("\n\n", target)
        cut = len(doc) if cut < 0 else cut
        pieces.append(doc[last:cut])
        pieces.append(f"\n\n【ZC-{code}】本条为资料原文之外的独立标记句。\n")
        last = cut
    pieces.append(doc[last:])
    doc = "".join(pieces)
    q = ("以上资料里**插入了若干**形如「【ZC-XXX】」的独立标记句（XXX 为三个大写字母）。\n"
         "请只列出你**确实在资料中读到**的 ZC 编号，逗号分隔。\n"
         "**不要猜测、不要补全、不要凑数**：不确定的一律不写。只输出编号，不要解释。")
    return doc, q, codes


def ask(doc: str, q: str):
    body = {
        "model": "qwen3.8-27b",
        "messages": [
            {"role": "system", "content": "你是严谨的技术助手，只依据给定资料回答。"},
            {"role": "user", "content": doc + "\n\n" + q},
        ],
        "temperature": 0,
        "max_tokens": 256,
        "stream": False,
        "reasoning_effort": "none",
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def main():
    wss = [int(x) for x in sys.argv[1:]] or [4096]
    doc, q, codes = build(SEED)
    truth = set(codes)
    print(f"真值 {len(truth)} 条:{','.join(codes)}\n", flush=True)
    rows = []
    for ws in wss:
        start(ws)
        try:
            for rep in range(REPS):
                r = ask(doc, q)
                ch = r["choices"][0]
                txt = ch["message"]["content"] or ""
                seen = set(re.findall(r"ZC-([A-Z]{3})", txt))
                tp = len(seen & truth)
                rec = tp / len(truth)
                prec = tp / len(seen) if seen else 0.0
                rows.append((ws, rep + 1, r["usage"]["prompt_tokens"],
                             r["usage"]["completion_tokens"], ch.get("finish_reason"),
                             f"{tp}/{len(truth)}", f"{rec*100:.0f}%", f"{prec*100:.0f}%"))
                print(rows[-1], flush=True)
        finally:
            stop()
    print("\n| ws | rep | prompt | out | finish | recall | 命中率 | 精确率 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")


if __name__ == "__main__":
    main()
