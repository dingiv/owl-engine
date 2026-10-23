#!/usr/bin/env python3
"""nsys 报告 → DFlash2 decode 逐核分账(metrics-optimize worktree;2026-10-22)。

判据(与 tools/nsys_decode_account.py 同源精神,但适配"prefill eager / decode 图"):
  · prefill 自 2026-10-21 起恒 eager(graphId = null),decode 的 verify/propose/fold
    全部走捕获图(graphId != null)⇒ **每段请求里第一个 graph kernel = decode 起点**;
  · 请求段 = 内核时间线按空闲间隔 > --gap-ms 切段(驱动里每请求之间 sleep);
  · 每段 decode = [第一个 graph kernel, 段尾] 的所有核;
  · 每核 per-round = 段内该核总设备时间 / 该段 spec 轮数(轮数由 --rounds 传入或自动估)。

用法:
  python3 tools/nsys_dflash_breakdown.py /tmp/diag/al-nsys/nsys.nsys-rep \
      [--rounds 20,20,20] [--gap-ms 1000] [--top 40]
  报告不存在 sqlite 时自动 `nsys export --type sqlite`。
"""
import argparse
import collections
import os
import sqlite3
import subprocess
import sys


def ensure_sqlite(rep: str) -> str:
    db = os.path.splitext(rep)[0] + ".sqlite"
    if not os.path.exists(db):
        subprocess.run(
            ["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", db, rep],
            check=True,
        )
    return db


def load(db: str):
    c = sqlite3.connect(db)
    names = {i: s for i, s in c.execute("select id,value from StringIds")}
    rows = []
    for start, end, dev, gid, nm in c.execute(
        "select start,end,deviceId,graphId,demangledName from CUPTI_ACTIVITY_KIND_KERNEL"
        " where end > start order by start"
    ):
        rows.append((start, end, dev, gid, names.get(nm, f"<id:{nm}>")))
    return rows


def bursts(rows, gap_ms: float):
    out = []
    cur = []
    for r in rows:
        if cur and (r[0] - cur[-1][1]) > gap_ms * 1e6:
            out.append(cur)
            cur = []
        cur.append(r)
    if cur:
        out.append(cur)
    return out


def group_of(name: str) -> str:
    n = name.lower()
    if "marlin" in n:
        return "marlin GEMM"
    if "kernel_mha" in n or "xqa" in n:
        return "XQA kernel_mha"
    if "gdn" in n:
        return "GDN 族"
    if "flashinfer" in n or "prefill" in n or "batchprefill" in n:
        return "FI prefill"
    if "cublas" in n or "gemv" in n:
        return "cuBLAS"
    if "dflash" in n or "naive_attn" in n or "topk16" in n:
        return "DFlash 草稿"
    if "rmsnorm" in n or "norm" in n or "rope" in n:
        return "norm/rope"
    if "reshape_and_cache" in n or "cache" in n:
        return "KV cache"
    if "memcpy" in n or "copy" in n:
        return "copy"
    return "其他"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rep")
    ap.add_argument("--rounds", default="", help="逗号分隔:每个含 graph 的请求段的 spec 轮数(空 = 只出总量)")
    ap.add_argument("--gap-ms", type=float, default=1000.0)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    db = ensure_sqlite(args.rep)
    rows = load(db)
    if not rows:
        print("无 kernel 记录", file=sys.stderr)
        return 1
    print(f"[db] {db} kernels={len(rows)}  span={(rows[-1][1]-rows[0][0])/1e9:.1f}s")

    bs = bursts(rows, args.gap_ms)
    gbs = [b for b in bs if any(r[3] for r in b)]
    rounds = [int(x) for x in args.rounds.split(",") if x.strip()]
    print(f"[seg] 总段={len(bs)} 其中含 graph 的请求段={len(gbs)}"
          + (f" rounds={rounds}" if rounds else ""))

    for i, b in enumerate(gbs):
        gstarts = [r[0] for r in b if r[3]]
        dec = [r for r in b if r[0] >= gstarts[0]]
        nr = rounds[i] if i < len(rounds) and rounds[i] > 0 else 1
        per = collections.Counter()
        cnt = collections.Counter()
        for start, end, dev, gid, nm in dec:
            per[nm] += (end - start) / 1e6
            cnt[nm] += 1
        total = sum(per.values())
        wall = (b[-1][1] - gstarts[0]) / 1e6
        print(
            f"\n═══ 段 {i}: 请求 span={((b[-1][1]-b[0][0])/1e6):.1f}ms "
            f"decode(首个图核→尾)={wall:.1f}ms 核数={len(dec)} "
            f"设备Σ={total:.1f}ms 轮={nr} ⇒ {total/nr:.2f}ms/轮"
        )
        print(f"  graphIds: {sorted({r[3] for r in dec if r[3]})}")
        if not rounds:
            print("  (未给 --rounds:以下为整段总量)")
        print(f"  {'核名':<46}{'ms合计':>10}{'次':>6}{'ms/轮':>10}{'占比':>8}")
        for nm, v in per.most_common(args.top):
            print(f"  {nm[:44]:<46}{v:>10.2f}{cnt[nm]:>6}{v/nr:>10.3f}{100*v/total:>7.1f}%")
        # 逐图(gid)分账
        per_gid = collections.Counter()
        cnt_gid = collections.Counter()
        top_gid = collections.defaultdict(collections.Counter)
        for start, end, dev, gid, nm in dec:
            if gid:
                per_gid[gid] += (end - start) / 1e6
                cnt_gid[gid] += 1
                top_gid[gid][nm] += (end - start) / 1e6
        # prefill 窗口(段首 → 第一个 graph kernel)
        pre = [r for r in b if r[0] < gstarts[0]]
        if pre:
            pper = collections.Counter(); pcnt = collections.Counter()
            for start, end, dev, gid, nm in pre:
                pper[nm] += (end - start) / 1e6; pcnt[nm] += 1
            ptot = sum(pper.values())
            print(f"  —— prefill 窗口(首个图核之前):{ptot:.1f}ms / {len(pre)} 核 ——")
            for nm, v in pper.most_common(10):
                print(f"    {nm[:44]:<46}{v:>10.2f}ms{pcnt[nm]:>7}次{100*v/ptot:>7.1f}%")
            pgrp = collections.Counter()
            for nm, v in pper.items():
                pgrp[group_of(nm)] += v
            print("    分组:" + " | ".join(f"{g} {100*v/ptot:.1f}%" for g, v in pgrp.most_common()))
        print("  —— 逐图(gid)——")
        for gid, v in per_gid.most_common():
            tops = ", ".join(f"{n[:28]} {x:.1f}" for n, x in top_gid[gid].most_common(3))
            print(f"  gid {gid:<4} {v:>9.2f}ms {cnt_gid[gid]:>6}次 {v/nr:>9.3f}ms/轮 {100*v/total:>6.1f}%  | {tops}")
        # 分组小计
        grp = collections.Counter()
        for nm, v in per.items():
            grp[group_of(nm)] += v
        print("  —— 分组小计 ——")
        for g, v in grp.most_common():
            print(f"  {g:<46}{v:>10.2f}{'':>6}{v/nr:>10.3f}{100*v/total:>7.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
