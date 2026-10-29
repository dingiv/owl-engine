#!/usr/bin/env python3
"""nsys 报告 → **decode 单步逐核分账**(两卡并排;§58 的判读面)。

用法: python3 tools/nsys_decode_account.py <tag1> [<tag2> ...]
      报告在 /tmp/diag/<tag>/nsys.nsys-rep(由 tools/nsys-decode-account.sh 生成);
      首次调用会自动 `nsys export --type sqlite`。

量法(判据口径,勿改):
  · 段切分 = 按内核开始时间排序,**间隔 >200µs 即段边界** ⇒ 一段 = 一次图回放 = 一 token 步;
    跳过头两段(预热),其余段取**每核中位数**(逐段同构,中位抗噪);
  · "设备时间" = 该步所有核的 (end−start) 之和 —— 与 ITL 的差额 = **host 面/步**;
  · 两卡(deviceId)分别列出 —— 两卡的 span 差 <0.05ms 即"无拖后腿 rank"的实证。
"""
import sqlite3, collections, statistics, sys, os, subprocess


def load(tag):
    d = f"/tmp/diag/{tag}"
    db = f"{d}/nsys.sqlite"
    if not os.path.exists(db):
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", db, f"{d}/nsys.nsys-rep"], capture_output=True, check=True)
    c = sqlite3.connect(db)
    name = {i: s for i, s in c.execute("select id,value from StringIds")}
    gid = {}
    for dev, gn, n in c.execute("select deviceId,graphId,count(*) n from CUPTI_ACTIVITY_KIND_KERNEL"
                                " where graphId is not null group by 1,2 order by n desc"):
        gid.setdefault(dev, gn)
    out = {}
    for dev, gn in gid.items():
        rows = sorted([(s, e, name[nm]) for s, e, nm in c.execute(
            "select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL"
            " where deviceId=? and graphId=?", (dev, gn))])
        segs = []; cur = [rows[0]]
        for k in rows[1:]:
            if k[0] - cur[-1][1] > 200_000: segs.append(cur); cur = [k]
            else: cur.append(k)
        segs.append(cur)
        segs = [s for s in segs if len(s) > 100][2:]
        per = collections.defaultdict(list); nk = []
        for s in segs:
            agg = collections.Counter()
            for st, e, nm in s: agg[nm] += (e - st) / 1e3
            for nm, v in agg.items(): per[nm].append(v)
            nk.append(len(s))
        gaps = [(segs[i + 1][0][0] - segs[i][-1][1]) / 1e6 for i in range(len(segs) - 1)]
        gaps = [g for g in gaps if g < 500]
        out[dev] = dict(
            per={nm: statistics.median(v) for nm, v in per.items()},
            span=statistics.median([(s[-1][1] - s[0][0]) / 1e6 for s in segs]),
            nk=int(statistics.median(nk)), nseg=len(segs),
            gap=statistics.median(gaps) if gaps else float('nan'))
    return out


def main(tags):
    reps = {t: load(t) for t in tags}
    # 主卡(每 tag 取 deviceId 最小者)与其余卡
    for t, r in reps.items():
        cards = sorted(r)
        s = r[cards[0]]
        print(f"\n═══ {t}:主卡 dev{cards[0]} · 单步 {s['nk']} 核 · 设备 Σ={sum(s['per'].values()):.0f}µs "
              f"· span {s['span']:.2f}ms · 段间空隙(host){s['gap']:.3f}ms · {s['nseg']} 段")
        for dev in cards[1:]:
            q = r[dev]
            print(f"    (对端 dev{dev}:Σ={sum(q['per'].values()):.0f}µs span {q['span']:.2f}ms "
                  f"空隙 {q['gap']:.3f}ms — 与主卡差 <0.05ms 即无拖后腿 rank)")
    base = reps[tags[0]][sorted(reps[tags[0]])[0]]['per']
    names = sorted(set().union(*[set(reps[t][sorted(reps[t])[0]]['per']) for t in tags]),
                   key=lambda n: -base.get(n, 0))
    hdr = f"{'核名':<44}" + "".join(f"{t+' µs':>12}" for t in tags)
    print("\n" + hdr)
    for nm in names:
        row = f"{nm[:42]:<44}"
        for t in tags:
            v = reps[t][sorted(reps[t])[0]]['per'].get(nm, 0.0)
            row += f"{v:>12.1f}" if v else f"{'—':>12}"
        print(row)
    for grp, key in (("marlin 小计", "Marlin<"), ("AR 小计", "ar_"), ("norm 小计", "rmsnorm")):
        row = f"{grp:<44}"
        for t in tags:
            per = reps[t][sorted(reps[t])[0]]['per']
            row += f"{sum(v for k,v in per.items() if key in k):>12.1f}"
        print(row)
    row = f"{'设备合计':<44}"
    for t in tags:
        row += f"{sum(reps[t][sorted(reps[t])[0]]['per'].values()):>12.1f}"
    print(row)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    main(sys.argv[1:])
