#!/bin/bash
# 白名单"实测台账"重测(2026-10-16 perf-review;修 L2 的依据)
#
# 做什么:起 `--flavor awq27b` + 该档 `[cuda] launch_time = true` 记账 →
#   发 3 个形状的请求(短/长 prefill/长输出)→ 取**本次实际发射**的核名集合,
#   与 (a) flavor 的 allowed_ops、(b) kernels 台账 `PROD_AWQ27B_2026_10_13`
#   两侧对照,打印差集。
#
# 用法:
#   PORT 由 args 档的 bind 推;卡走 CVD/pw-env 自动挑。
#   tools/wl-audit.sh testdata/args/wl-audit-awq27b.toml      /tmp/wl-audit.log
#   tools/wl-audit.sh testdata/args/wl-audit-awq27b-a8.toml   /tmp/wl-audit-a8.log
#
# ⚠️ 判读纪律:
#   - `实测 − flavor` 非空 = 门开着却没拦住(或引擎在跑白名单外的核)⇒ P0;
#   - `flavor − 实测` 非空 = 声明多出来的条目(候选删;但要先确认"该档能覆盖
#     到所有臂" —— 本档只跑裸 27B 的 prefill+decode,不跑 spec/视觉路径);
#   - 捕获期发射不打(录制中);warmup 契约保证 eager 已跑过同一批核 ⇒ 名集合完整。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/pw-env.sh"

ARGS=${1:?用法: wl-audit.sh <args.toml> [日志]}
LOG=${2:-/tmp/wl-audit.log}

# 端口:从 args 档的 bind 推(覆盖 pw-env 的缺省)
PORT_FROM_ARGS=$(grep -m1 -o 'bind *= *"127\.0\.0\.1:[0-9]*"' "$ARGS" | sed 's/.*:\([0-9]*\)".*/\1/')
if [ -n "$PORT_FROM_ARGS" ]; then PW_PORT="$PORT_FROM_ARGS"; fi
echo "[wl-audit] 档=$ARGS 端口=$PW_PORT 卡=$PW_CVD 日志=$LOG"

pw_kill_port
sleep 3

CUDA_VISIBLE_DEVICES="$PW_CVD" nohup "$PW_ROOT/target/release/server" \
  --flavor awq27b "$ARGS" --log-level debug >"$LOG" 2>&1 &
SPID=$!

ok=0
for _ in $(seq 1 120); do
  if curl -s --max-time 2 "http://127.0.0.1:$PW_PORT/health" | grep -q '"ok"'; then ok=1; break; fi
  sleep 2
done
if [ "$ok" != 1 ]; then
  echo "BOOT-FAIL"; tail -20 "$LOG"; kill "$SPID" 2>/dev/null; exit 1
fi
echo "[wl-audit] 就绪,发请求……"

req() { # $1 = prompt 文本; $2 = max_tokens
  curl -s --max-time 300 "http://127.0.0.1:$PW_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"max_tokens\":$2,\"temperature\":0}" \
    -o /tmp/wl-audit-resp.json
  python3 - <<'PY'
import json, sys
FAIL = 0
try:
    d = json.load(open('/tmp/wl-audit-resp.json'))
except Exception as e:
    print("   !! resp 解析失败:", e)
    FAIL = 1
else:
    u = d.get('usage')
    ch = len(d.get('choices', []) or [])
    print("   usage:", u)
    if (not ch) or (not u):
        # 请求失败(准入拒 / turn 弃置)⇒ 名集合不完整,不得当台账依据
        print(f"   !! 请求未成功(choices={ch})⇒ 本次实测不可作台账依据")
        FAIL = 1
if FAIL:
    raise SystemExit(3)
PY
  rc=$?
  if [ "$rc" != 0 ]; then echo "[wl-audit] 请求失败(rc=$rc)——中止,不产出不可靠的台账"; kill "$SPID" 2>/dev/null; exit 1; fi
}

req "用一句话说明 Redis 和 Memcached 的区别。" 32
# 长 prefill(跨 chunk):重复文本凑长度
LONG=$(python3 -c "print('继续。'*600)")
req "$LONG" 8
# 长输出(多步 decode)
req "从 1 数到 120,每个数字之间用逗号分隔,不要换行。" 256

kill "$SPID" 2>/dev/null || true
sleep 2

python3 - "$LOG" "$PW_ROOT" <<'PY'
import re, sys, pathlib, tomllib, collections
log = pathlib.Path(sys.argv[1]).read_text(errors="replace")
root = pathlib.Path(sys.argv[2])

hits = re.findall(r'launch-time\]\s+\S+\s+([A-Za-z_][A-Za-z0-9_]*)\s*$', log, re.M)
cnt = collections.Counter(hits)
measured = set(hits)
print(f"\n== 本次实测发射核名:{len(measured)} 种 / {sum(cnt.values())} 次 ==")
for n, c in cnt.most_common():
    print(f"  {c:>7}  {n}")

# flavor 的 allowed_ops
fl = tomllib.loads((root / "flavors/awq27b.toml").read_text())
flavor = set(fl.get("dispatch", {}).get("allowed_ops", []))

# kernels 台账(PROD_AWQ27B_2026_10_13 的 Rust 列表字面量)
src = (root / "crates/kernels/src/ops/mod.rs").read_text()
m = re.search(r'PROD_AWQ27B_2026_10_13\s*:\s*&\[&str\]\s*=\s*&\[(.*?)\];', src, re.S)
table = set(re.findall(r'"([^"]+)"', m.group(1))) if m else set()
if not m:
    print("!! 没能从 ops/mod.rs 解析 PROD_AWQ27B_2026_10_13")

def show(tag, a, b):
    d = sorted(a - b)
    print(f"\n== {tag}({len(d)}) ==")
    for x in d:
        print("  ", x)

show("实测 − flavor(应为空:门开着却发了白名单外的核 = P0)", measured, flavor)
show("flavor − 实测(声明多出来的条目)", flavor, measured)
show("实测 − 台账(台账缺的实发核 ⇒ 台账要补)", measured, table)
show("台账 − 实测(台账多出来的条目)", table, measured)
print(f"\n[计数] 实测 {len(measured)} / flavor {len(flavor)} / 台账 {len(table)}")
PY
