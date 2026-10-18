#!/bin/bash
# piecewise A/B · greedy + 哑草稿档:同一请求、同一 prompt,两臂文本逐字对比
# 用法: tools/pw-ab-greedy.sh <args.toml> <日志> <临时文本>
# 判据:verify 行 0 = 真模型下一 token(哑草稿必被拒)⇒ 两臂文本必须逐字一致。
set -u
source "$(dirname "${BASH_SOURCE[0]}")/pw-env.sh"
ARGS="$1"; LOG="$2"; OUT="$3"

pw_kill_port
sleep 4

CUDA_VISIBLE_DEVICES="$PW_CVD" \
  env ${PW_ENV:-OWL_PW_NOP=1} nohup "$PW_ROOT/target/release/server" --flavor spec-dflash "$ARGS" --log-level info >"$LOG" 2>&1 &
SPID=$!

ok=0
for _ in $(seq 1 90); do
  if curl -s --max-time 2 "http://127.0.0.1:$PW_PORT/health" | grep -q '"ok"'; then ok=1; break; fi
  sleep 2
done
if [ "$ok" != 1 ]; then echo "BOOT-FAIL"; tail -5 "$LOG"; kill "$SPID" 2>/dev/null; exit 1; fi

curl -s --max-time 180 "http://127.0.0.1:$PW_PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"用五十字介绍长城。"}],"max_tokens":32,"temperature":0}' \
  > "$OUT"
python3 - "$OUT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding='utf-8'))
c = d.get('choices', [{}])[0]
print("TEXT:", repr(c.get('message', {}).get('content')))
print("USAGE:", d.get('usage'))
PY
kill "$SPID" 2>/dev/null || true
sleep 2
