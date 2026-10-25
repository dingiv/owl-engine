#!/usr/bin/env bash
# DFlash2 长上下文 AL 扫描跑批(metrics-optimize worktree;2026-10-22)
#
# 前置:已 `cargo build --release -p server --features perf,cupti`
# 行为:
#   1. 查卡(BDF 41:00.0 必须 <500MiB)+ 查端口 8140;
#   2. --check 预检 → 起服(日志落 run 目录)→ 等就绪(>30s 警告);
#   3. prose / math 两域各扫一遍 ctx 阶梯(AL + 分相 metrics);
#   4. 按 PID 停服,复查显存归零。
#
# 用法: tools/run-dflash-longctx-probe.sh [out_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/perf-runs/metrics-optimize/max-$(date +%m%d-%H%M)}"
PROFILE="$ROOT/profiles/qwen38_27b-3090.toml"
MAX_SEQ="${MAX_SEQ:-67584}"
PORT=8140
BDF="41:00.0"
LENGTHS="${LENGTHS:-512,2048,8192,16384,32768,49152,65536}"
OUTLEN="${OUTLEN:-256}"
SEED="${SEED:-20261022}"
CORPUS="${CORPUS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DOMAINS="${DOMAINS:-prose,math}"
cd "$ROOT"

mkdir -p "$OUT"
echo "[out] $OUT"

# ── 1. 卡与端口 ─────────────────────────────────────────────────────────────
idx=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v b="00000000:$BDF" '$2==b{print $1}')
[ -n "$idx" ] || { echo "✗ 找不到 BDF $BDF 的卡"; exit 2; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx")
[ "$used" -lt 500 ] || { echo "✗ GPU$idx ($BDF) 不空:${used}MiB"; exit 2; }
if ss -ltnp 2>/dev/null | grep -q ":$PORT "; then echo "✗ 端口 $PORT 被占"; exit 2; fi
echo "[gpu] GPU$idx $BDF used=${used}MiB"

# ── 2. 预检 + 起服 ──────────────────────────────────────────────────────────
./target/release/server -p "$PROFILE" --check > "$OUT/check.txt" 2>&1 || {
  echo "✗ --check 失败"; tail -20 "$OUT/check.txt"; exit 3; }
echo "[check] OK"

./target/release/server -p "$PROFILE" --log-level info $EXTRA_ARGS > "$OUT/server.log" 2>&1 &
SRV=$!
t0=$(date +%s)
ok=0
for _ in $(seq 1 40); do
  grep -q "就绪 http" "$OUT/server.log" 2>/dev/null && { ok=1; break; }
  kill -0 "$SRV" 2>/dev/null || break
  sleep 1
done
boot=$(( $(date +%s) - t0 ))
if [ "$ok" != 1 ]; then
  echo "✗ 启动失败(${boot}s)"; tail -30 "$OUT/server.log"; kill "$SRV" 2>/dev/null || true; exit 4
fi
echo "[boot] ${boot}s$( [ "$boot" -gt 30 ] && echo '  ⚠️ >30s')"
curl -s "http://127.0.0.1:$PORT/health" > "$OUT/health.json"
cat "$OUT/health.json"; echo
if [ -n "${PRE_HOOK:-}" ]; then bash -c "$PRE_HOOK"; fi

# ── 3. 扫描 ────────────────────────────────────────────────────────────────
IFS=',' read -ra DOMS <<< "$DOMAINS"
for dom in "${DOMS[@]}"; do
  echo "[sweep] domain=$dom lengths=$LENGTHS"
  python3 tools/dflash_al_curve.py \
    --url "http://127.0.0.1:$PORT" --model qwen3.8-27b \
    --lengths "$LENGTHS" --output-length "$OUTLEN" --domain "$dom" --seed "$SEED" \
    --out "$OUT/al-$dom" ${CORPUS:+--prompt-files "$CORPUS"} 2>&1 | tee "$OUT/al-$dom.log"
done

# ── 3.5 可选:抓一次引擎内 perf 快照(如 CUPTI gpu 档)─────────────────────
if [ -n "${POST_HOOK:-}" ]; then bash -c "$POST_HOOK"; fi

# ── 4. 停服 ────────────────────────────────────────────────────────────────
pid=$(ss -ltnp 2>/dev/null | sed -n "s/.*127\.0\.0\.1:$PORT.*pid=\([0-9]*\).*/\1/p" | head -1)
[ -n "${pid:-}" ] && kill -TERM "$pid" 2>/dev/null || true
wait "$SRV" 2>/dev/null || true
sleep 2
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx")
echo "[bye] GPU$idx used=${used}MiB"
[ "$used" -lt 500 ] && echo "[ok] 显存已归零" || echo "[warn] 显存未归零"
echo "[done] $OUT"
