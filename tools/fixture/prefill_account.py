#!/usr/bin/env python3
"""nsys 报告 → **prefill(eager)逐核分账**(P0.1;`tools/prefill-account.sh` 的判读面)。

量法(判据口径,勿改):
  · prefill 走 eager ⇒ 核的 `graphId is null`(图回放核排除);
  · **窗口 = 最后一次成簇的 prefill**:按核 start 排序,间隔 > 200ms 即"事件边界",
    取**最后一段**(curl 触发的 8k prompt;boot/warmup 的段被切掉);
  · "设备时间" = 段内所有核 (end−start) 之和;每核列 次数/合计/每次;
  · 每 token 归一分账 = 合计 / prompt_tokens(`resp.json` 的 usage)。
"""
import sqlite3, collections, os, subprocess, sys, json


def seg_after_last_gap(rows, gap_ns=200_000_000):
    """按 start 切段,返回最后一段(间隔 > gap_ns 即边界)。"""
    segs, cur = [], [rows[0]]
    for k in rows[1:]:
        if k[0] - cur[-1][1] > gap_ns:
            segs.append(cur); cur = [k]
        else:
            cur.append(k)
    segs.append(cur)
    return segs[-1], len(segs)


def load(tag):
    d = f"/tmp/diag/{tag}"
    db = f"{d}/nsys.sqlite"
    if not os.path.exists(db):
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true",
                        "-o", db, f"{d}/nsys.nsys-rep"], capture_output=True, check=True)
    c = sqlite3.connect(db)
    rows = list(c.execute(
        "select k.start,k.end,s.value,k.deviceId from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id=k.demangledName where k.graphId is null "
        "order by k.start"))
    if not rows:
        return None
    seg, nseg = seg_after_last_gap(rows)
    tk = json.load(open(f"{d}/resp.json")).get("usage", {}).get("prompt_tokens", 0) if os.path.exists(f"{d}/resp.json") else 0
    per = collections.defaultdict(list)
    perdev = collections.defaultdict(lambda: collections.defaultdict(float))
    for st, e, nm, dev in seg:
        per[nm].append((e - st) / 1e3)
        perdev[dev][nm] += (e - st) / 1e3
    return dict(per=per, perdev=perdev, nseg=nseg, seg_n=len(seg), tok=tk,
                span=(seg[-1][1] - seg[0][0]) / 1e6, devs=sorted(perdev))


def main(tags):
    reps = {}
    for t in tags:
        r = load(t)
        if r is None:
            print(f"{t}: 无 eager 核(报告缺失?)"); continue
        reps[t] = r
        s = sum(v for nm in r["per"] for v in r["per"][nm])
        print(f"\n═══ {t}: eager 段 = 最后一个事件窗(总 {r['nseg']} 段)· {r['seg_n']} 核 · "
              f"设备 Σ={s:.0f}µs · 窗 span {r['span']:.1f}ms · prompt {r['tok']} tok "
              f"⇒ {s / max(r['tok'], 1):.1f} µs/token")
        for dev in r["devs"]:
            q = sum(r["perdev"][dev].values())
            print(f"    dev{dev}: Σ={q:.0f}µs")
    if not reps:
        return
    base = list(reps.values())[0]
    names = sorted(set().union(*[set(r["per"]) for r in reps.values()]),
                   key=lambda n: -sum(base["per"].get(n, [0])))
    hdr = f"{'核名':<46}" + "".join(f"{t + ' µs':>13}" for t in reps)
    print("\n" + hdr)
    for nm in names[:28]:
        row = f"{nm[:44]:<46}"
        for t in reps:
            v = sum(reps[t]["per"].get(nm, [0]))
            row += f"{v:>13.0f}" if v else f"{'—':>13}"
        print(row)
    # 分层小计
    def bucket(nm):
        if "marlin" in nm: return "marlin(权重 GEMM)"
        if "owl_comm_ar_" in nm: return "AR(通信)"
        if "rmsnorm" in nm or "norm" in nm: return "norm"
        if "attention" in nm or "gdn" in nm or "rope" in nm: return "attn/GDN"
        if "quant_int8" in nm: return "quant_int8(W4A8)"
        if "gemv" in nm or "gemm" in nm or "cutlass" in nm or "internal::" in nm: return "gemm/lm_head"
        return "其它"
    print()
    tot = {}
    for t, r in reps.items():
        agg = collections.Counter()
        for nm in r["per"]:
            agg[bucket(nm)] += sum(r["per"][nm])
        tot[t] = agg
        s = sum(agg.values())
        print(f"  {t}: " + " | ".join(f"{k} {v:.0f}µs({100 * v / max(s, 1):.0f}%)"
                                       for k, v in agg.most_common()))


if __name__ == "__main__":
    main(sys.argv[1:])
