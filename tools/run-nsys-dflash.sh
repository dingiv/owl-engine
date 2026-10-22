#!/usr/bin/env bash
# DFlash2 decode 逐核分账(nsys)跑批(metrics-optimize worktree;2026-10-22)
#
# 判据:prefill 恒 eager(graphId=null),decode 的 verify/propose/fold 走捕获图
# (graphId != null)⇒ 每段请求里第一个 graph kernel = decode 起点。
# 请求间 sleep 2s ⇒ 分析器按空闲间隔切段不会粘连。
#
# 用法: tools/run-nsys-dflash.sh <out_dir> [lengths] [outlen]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?out_dir}"
LENGTHS="${2:-512,2048,16384,65536}"
OUTLEN="${3:-64}"
PORT=8140
cd "$ROOT"
mkdir -p "$OUT"

idx=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v b="00000000:41:00.0" '$2==b{print $1}')
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx")
[ "$used" -lt 500 ] || { echo "✗ GPU$idx 不空:${used}MiB"; exit 2; }
ss -ltnp 2>/dev/null | grep -q ":$PORT " && { echo "✗ 端口 $PORT 被占"; exit 2; }

nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --force-overwrite=true -o "$OUT/nsys" \
  ./target/release/server -p "$ROOT/profiles/qwen38_27b-3090-w4a16-dflash-longctx.toml" --log-level warn \
  > "$OUT/server.log" 2>&1 &
NSYS=$!
ok=0
for _ in $(seq 1 40); do
  grep -q "就绪 http" "$OUT/server.log" 2>/dev/null && { ok=1; break; }
  kill -0 "$NSYS" 2>/dev/null || break
  sleep 1
done
[ "$ok" = 1 ] || { echo "✗ 启动失败"; tail -20 "$OUT/server.log"; kill "$NSYS" 2>/dev/null; exit 3; }
echo "[boot] ok"

python3 tools/dflash_al_curve.py --url "http://127.0.0.1:$PORT" --model qwen3.8-27b \
  --lengths "$LENGTHS" --output-length "$OUTLEN" --domain prose --seed 20261022 \
  --sleep-between 2 --out "$OUT/al" 2>&1 | tee "$OUT/al.log"

pid=$(ss -ltnp 2>/dev/null | sed -n "s/.*127\.0\.0\.1:$PORT.*pid=\([0-9]*\).*/\1/p" | head -1)
[ -n "${pid:-}" ] && kill -TERM "$pid" 2>/dev/null || true
# nsys 在 app 退出后要 "Collecting data..." 一段时间才落盘:等它自然退出,别 kill
for _ in $(seq 1 120); do
  [ -s "$OUT/nsys.nsys-rep" ] && break
  kill -0 "$NSYS" 2>/dev/null || break
  sleep 1
done
wait "$NSYS" 2>/dev/null || true
ls -la "$OUT/nsys.nsys-rep" 2>/dev/null && echo "[ok] 报告:$OUT/nsys.nsys-rep" || echo "[✗] 无报告"
