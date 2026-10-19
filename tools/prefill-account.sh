#!/usr/bin/env bash
# **P0.1 prefill 逐核账仪器**(2026-10-19) —— nsys 抓"长 prompt + 出 1 token"。
#
# 为什么与 `nsys-decode-account.sh` 分开:decode 是**图回放**(分析见 graphId 非空),
# 而 prefill 走 **eager**(逐核串行)⇒ 分析面是 `graphId is null` 的核,段切分法也不同。
#
# 用法:`prefill-account.sh <tag> <flavor> <运行档>`
#   prompt = 6000 随机词(≈8192 tok,与 llm_speedtest 同词池纪律),max_tokens = 1
#   ⇒ 只有 prefill 段。分析:`prefill_account.py <tag>...`
set -u
TAG="$1"; FLAVOR="$2"; CFG="$3"
PORT="${PORT:-8141}"
ROOT="${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
OUT="/tmp/diag/$TAG"
mkdir -p "$OUT"; rm -f "$OUT"/nsys.nsys-rep "$OUT"/nsys.sqlite "$OUT"/srv.log 2>/dev/null
cd "$ROOT" || exit 1

# 军规:按监听 PID 收旧实例(宽匹配任意地址 —— 见 tp-ab-std.sh 的同款注释)
for p in $(ss -ltnp 2>/dev/null | grep -E ":$PORT([^0-9]|$)" | grep -o 'pid=[0-9]*' | cut -d= -f2); do
  kill "$p" 2>/dev/null || true
done
sleep 2

python3 - "$OUT/req.json" <<'PY'
import json, random, sys
random.seed(11)
w = ["alpha","bravo","charlie","delta","echo","foxtrot","golf","hotel","india","juliet",
     "kilo","lima","mike","november","oscar","papa","quebec","romeo","sierra","tango",
     "uniform","victor","whiskey","xray","yankee","zulu","anchor","basket","candle","dagger",
     "engine","fabric","gadget","harbor","island","jacket","kettle","lantern","magnet","needle",
     "oyster","puzzle","quartz","ribbon","saddle","tunnel","umbrella","vessel","walnut","yellow"]
text = " ".join(random.choice(w) for _ in range(6000))
json.dump({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": text}],
           "max_tokens": 1, "temperature": 0, "ignore_eos": True}, open(sys.argv[1], "w"))
print("[prefill-account] prompt = 6000 词(≈8k tok),max_tokens = 1")
PY

nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  -o "$OUT/nsys" ./target/release/server --flavor "$FLAVOR" "$CFG" --log-level warn \
  > "$OUT/srv.log" 2>&1 &
NSYS=$!
ok=0
for _ in $(seq 1 150); do
  grep -q "就绪 http" "$OUT/srv.log" 2>/dev/null && { ok=1; break; }
  grep -qE "❌|装配失败|拒绝启动" "$OUT/srv.log" 2>/dev/null && { echo "[$TAG] ❌ 启动被拒"; tail -6 "$OUT/srv.log"; break; }
  sleep 2
done
[ "$ok" = 1 ] || { echo "[$TAG] ❌ 超时"; kill "$NSYS" 2>/dev/null; exit 3; }
grep -m1 "选卡" "$OUT/srv.log" || true
sleep 3

echo "[$TAG] prefill 主测(8k prompt)"
/usr/bin/time -f "[$TAG] curl 墙钟 %e s" curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @"$OUT/req.json" > "$OUT/resp.json" 2>"$OUT/curl.time"
cat "$OUT/curl.time"
python3 - "$OUT/resp.json" <<'PY'
import json, sys
u = json.load(open(sys.argv[1])).get('usage', {})
print(f"  usage: prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}")
PY
sleep 2

for p in $(ss -ltnp 2>/dev/null | grep -E ":$PORT([^0-9]|$)" | grep -o 'pid=[0-9]*' | cut -d= -f2); do
  kill -TERM "$p" 2>/dev/null || true
done
sleep 8
kill -TERM "$NSYS" 2>/dev/null || true
wait "$NSYS" 2>/dev/null
ls -la "$OUT/nsys.nsys-rep" 2>/dev/null && echo "[$TAG] ✓ 分析:python3 tools/prefill_account.py $TAG"
