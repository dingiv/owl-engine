#!/usr/bin/env bash
# 3090 单卡 W4A16 满血档启动(常驻档 = `profiles/qwen38_27b-3090.toml`)
#
#   · 行为面(上下文/池/FI 图/GDN chunked/无 KVMem)**全在意图档**;
#   · 本脚本只做**站点面**:卡位(BDF)+ 端口 + 模型路径(--site 站点档);
#   · **不再生成临时 TOML**(配置面 P0:--set 取代 extends 拼档)。
#
# 用法(在任意目录;脚本自解析仓库根):
#   tools/serve-w4a16-3090.sh                       # BDF=42:00.0, PORT=8139
#   BDF=41:00.0 PORT=8140 tools/serve-w4a16-3090.sh
#   tools/serve-w4a16-3090.sh /path/site.toml       # 自备站点档(须含 [model].dir)
#   CHECK=1 tools/serve-w4a16-3090.sh              # 只预检,不起服(<2s,不装权重)
#
# 站点档缺省 = `testdata/args/base.toml`(本机路径,不入库)。
#
# 前置: cargo build --release --bin server
#
# 卡位纪律(AGENTS):**用 BDF 钉卡,不用 CUDA_VISIBLE_DEVICES 数字序**
# (vLLM 内部会按 PCI_BUS_ID 重解释数字序)。本机 3090 Ti:
#   41:00.0 / 42:00.0(`nvidia-smi --query-gpu=index,name,pci.bus_id --format=csv`)
# 起服前先确认目标卡空闲:
#   nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/target/release/server}"
PROFILE="${PROFILE:-qwen38_27b-3090}"
BDF="${BDF:-42:00.0}"
PORT="${PORT:-8139}"
LOG_LEVEL="${LOG_LEVEL:-info}"
SITE="${1:-$ROOT/testdata/args/base.toml}"

if [[ ! -x "$BIN" ]]; then
    echo "缺少 $BIN —— 先在 $ROOT 下 cargo build --release --bin server" >&2
    exit 2
fi
if [[ ! -f "$SITE" ]]; then
    echo "站点档不存在:$SITE(需含 [model].dir;可从 testdata/args/base.toml 复制)" >&2
    exit 2
fi

ARGS=(
    -p "$PROFILE"
    --site "$SITE"
    --set "runtime.bind=127.0.0.1:$PORT"
    --set "runtime.device.bdf=$BDF"
)

echo "[serve] profile=$PROFILE bdf=$BDF bind=127.0.0.1:$PORT site=$SITE"
if [[ "${CHECK:-0}" == "1" ]]; then
    exec "$BIN" "${ARGS[@]}" --check
fi
"$BIN" "${ARGS[@]}" --log-level "$LOG_LEVEL" &
OWL_PID=$!
trap 'kill "$OWL_PID" 2>/dev/null || true' EXIT INT TERM
wait "$OWL_PID"
