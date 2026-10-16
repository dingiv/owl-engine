#!/bin/bash
# 快速诊断:boot(带 env)→ 拉 perf epoch 的 host-node 直方图 → 停
# 用法: tools/pw-probe.sh <args.toml> <日志> <ENV 串> <前缀>
set -u
source "$(dirname "${BASH_SOURCE[0]}")/pw-env.sh"
ARGS="$1"; LOG="$2"; ENVV="$3"; PFX="${4:-interp.host.node}"
pw_kill_port
sleep 4
CUDA_VISIBLE_DEVICES="$PW_CVD" \
  env $ENVV nohup "$PW_ROOT/target/release/server" --flavor spec-dflash "$ARGS" --perf-level host --log-level info >"$LOG" 2>&1 &
SPID=$!
for _ in $(seq 1 90); do curl -s --max-time 2 "http://127.0.0.1:$PW_PORT/health" | grep -q '"ok"' && break; sleep 2; done
curl -s --max-time 20 "http://127.0.0.1:$PW_PORT/debug/perf/$PFX" >"$LOG.perf.txt"
python3 - "$LOG.perf.txt" <<'PY'
import re, sys, collections
txt = open(sys.argv[1], encoding='utf-8', errors='replace').read()
rows = re.findall(r'^\s*(\S+)\s+n=(\d+)\s+self=([0-9.]+)(µs|ms|s)\s', txt, re.M)
# 扁平化:取末段类名
agg = collections.Counter()
for tag, n, v, u in rows:
    cls = tag.split('.')[-1]
    agg[cls] += int(n)
tot = sum(agg.values())
print("total launches(host-node rows):", tot, "| classes:", len(agg))
for cls, n in agg.most_common(18):
    print(f"  {n:>6}  {cls}")
PY
kill "$SPID" 2>/dev/null || true
sleep 2
