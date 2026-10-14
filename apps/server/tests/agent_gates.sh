#!/usr/bin/env bash
# agent 服务面回归门(2026-10-14;B4 = 两脚本纳入常规纪律)
#
# 用法:
#   apps/server/tests/agent_gates.sh --base-url http://127.0.0.1:8137 --model qwen3.8-27b
#
# 前提:server 已在跑(生产档 = `--flavor spec-dflash testdata/server/agent-27b.toml`;
# 端口 8137 / CUDA ordinal 0,见该档头注)。脚本**不启 server、不杀进程**
# (杀进程纪律:只杀自己的 PID —— 见 AGENTS.md / 设计 §八施工纪律)。
#
# 三把门(缺一不可;谁都不能替代谁):
#   1. lag_probe.py   —— **多请求内容正确性**(吞吐 bench 测不出来的那一类:
#                        XQA 页表恒等链 bug 就是被它抓住的)
#   2. agent_wire_smoke.py —— wire 契约(三通道帧序/finish_reason/usage/400 面)
#   3. e2e.rs(27B 黑盒,cargo test -p server --test e2e)
#                        —— L1 协议面 + agent 面(工具历史/stop/400/工具闭环)
#
# 判据:三者全绿 = 本专项回归通过。任何一把先红 = 停止后续门(不掩盖)。
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8137}"
MODEL="${MODEL:-qwen3.8-27b}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url) BASE_URL="$2"; shift 2 ;;
    --model)    MODEL="$2";    shift 2 ;;
    -h|--help)  sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "未知参数 $1(--help)"; exit 2 ;;
  esac
done

echo "=== agent 回归门 @ ${BASE_URL}(model=${MODEL})==="

echo "--- [1/3] lag_probe(多请求内容正确性)---"
python3 "${HERE}/lag_probe.py" --base-url "${BASE_URL}" --model "${MODEL}"

echo "--- [2/3] agent_wire_smoke(wire 契约 17 项)---"
python3 "${HERE}/agent_wire_smoke.py" --base-url "${BASE_URL}" --model "${MODEL}"

echo "--- [3/3] e2e(27B 黑盒;自带 boot,端口 8135)---"
echo "注:e2e 自起进程(8135/CUDA 1),启用前请确认该卡空闲(上面那个 server 在 8137)"
(cd "${HERE}/../../.." && cargo test -p server --test e2e -- --nocapture)

echo "=== 三门全绿 ==="
