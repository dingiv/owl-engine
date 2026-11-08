#!/bin/bash
# **CLI 双会话 decode×prefill 黑盒门**（2026-11-06）。
#
# 场景：A 会话流式输出（decode），B 会话带长 prompt 进来（prefill）——
# 验两边都完成、两个 turn 同时在册（真并发档）、解码侧不被整段 prefill 冻结。
#
# 用法：
#   tools/fixture/run-cli-decode-prefill.sh                     # 默认 8137 + 单 3090 并发测试档
#   tools/fixture/run-cli-decode-prefill.sh --only decode_vs_prefill_24k
#   tools/fixture/run-cli-decode-prefill.sh --url http://127.0.0.1:8135   # 只跑已起服务
#   tools/fixture/run-cli-decode-prefill.sh --profile profiles/xxx.local.toml
#
# 前置档要求（见脚本头与 profiles/README.md）：
#   · 真并发：单卡 + max_batch=2 + gdn_slots>=2 + 无选择层 + spec off
#     + runtime.slice_every>0（否则解码被整段 prefill 冻结）；
#   · 队列形态（TP2 生产档 max_batch=1）也能跑：测试报 mode=queued。
set -u
cd "$(dirname "$0")/../.."

URL="http://127.0.0.1:8137"
PROFILE="profiles/qwen38_27b-3090-conc.local.toml"
ONLY=""
JSON=""
STARTED=""

while [ $# -gt 0 ]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --json) JSON="--json"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done

PORT=$(echo "$URL" | sed -n 's#.*:\([0-9][0-9]*\)/*$#\1#p')
[ -n "$PORT" ] || { echo "无法从 $URL 解析端口" >&2; exit 2; }

echo "[cli-dp] build cli..." >&2
cargo build --release -p cli >/dev/null 2>&1 || { echo "[cli-dp] cli 构建失败" >&2; exit 2; }

healthy() { curl -s -o /dev/null --max-time 2 "$URL/health"; }

if ! healthy; then
  [ -f "$PROFILE" ] || { echo "[cli-dp] $URL 未就绪且 profile 不存在: $PROFILE" >&2; exit 2; }
  echo "[cli-dp] $URL 未就绪 ⇒ 启动 $PROFILE" >&2
  setsid nohup ./target/release/server -p "$PROFILE" > /tmp/owl-cli-dp.log 2>&1 < /dev/null &
  for i in $(seq 1 35); do
    if healthy; then echo "[cli-dp] 就绪 ${i}s" >&2; break; fi
    sleep 1
  done
  if ! healthy; then
    echo "[cli-dp] 启动失败（tail /tmp/owl-cli-dp.log）" >&2
    tail -5 /tmp/owl-cli-dp.log >&2
    exit 2
  fi
  STARTED=$(ss -ltnp 2>/dev/null | grep ":$PORT" | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1)
fi

cleanup() {
  if [ -n "$STARTED" ] && kill -0 "$STARTED" 2>/dev/null; then
    echo "[cli-dp] 停测试服务 pid=$STARTED" >&2
    kill -KILL "$STARTED" 2>/dev/null
  fi
}
trap cleanup EXIT

# 兜底时限：引擎在"池满 + 起 turn 失败"时曾出现单个 HTTP 请求不返回
# （/debug/agent 记"未结算"）—— 挂住就按失败收口，别把 CI 拖到 30 分钟。
TIMEOUT_BIN=$(command -v timeout || true)
ARGS=(--workload decode-prefill --url "$URL")
[ -n "$ONLY" ] && ARGS+=(--only "$ONLY")
[ -n "$JSON" ] && ARGS+=("$JSON")
if [ -n "$TIMEOUT_BIN" ]; then
  "$TIMEOUT_BIN" 600 ./target/release/cli "${ARGS[@]}"
else
  ./target/release/cli "${ARGS[@]}"
fi
