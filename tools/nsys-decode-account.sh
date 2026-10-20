#!/usr/bin/env bash
# **图模式逐核分账仪器**(2026-10-19 立;§58 的取证工具) —— nsys + `--cuda-graph-trace=node`
#
# 用途(为什么需要它):生产 decode 是**整图单发射**,`OWL_GPU_PROF`(逐核同步)
# 在图模式归因不到核;而 eager 档会注入 host 提交噪声。nsys 的 graph-node 展开
# **不改代码、不改档位**就能拿到:每核真设备时间、每核次数、**两 rank 分卡时间线**。
#
# 用法(worktree 根;需先 `cargo build --release --bin server`):
#   tools/nsys-decode-account.sh <tag> <flavor> <运行档>
#   # 例:tools/nsys-decode-account.sh tp2 tp27b testdata/args/bench-27b-tp2.toml
#   python3 tools/nsys_decode_account.py <tag>        # 出单步逐核表(两卡并排)
#
# 纪律:
#   · 运行档里 `bind` 必须与别处错开(本脚本用 8139),避免踩现役服务;
#   · 起服前确认目标卡空闲(`nvidia-smi --query-compute-apps`)—— 别的会话占卡会让读数失真(律㉘);
#   · 报告/tag 目录落在 `/tmp/diag/<tag>/`(nsys 报告体积大,不入树)。
set -u
TAG="$1"; FLAVOR="$2"; CFG="$3"
PORT="${PORT:-8139}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="/tmp/diag/$TAG"
API="http://127.0.0.1:$PORT/v1/chat/completions"
mkdir -p "$OUT"; rm -f "$OUT"/nsys.nsys-rep "$OUT"/nsys.sqlite "$OUT"/srv.log 2>/dev/null
cd "$ROOT" || exit 1

# 军规:按监听 PID 收旧实例(不用 `pkill -f` —— 模式会匹配自身命令行)
for p in $(ss -ltnp 2>/dev/null | grep -E ":$PORT([^0-9]|$)" | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill "$p" 2>/dev/null || true; done
sleep 2

nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --force-overwrite=true -o "$OUT/nsys" \
  ./target/release/server -p "profiles/legacy/$FLAVOR.toml" "$CFG" --log-level warn \
  > "$OUT/srv.log" 2>&1 &
NSYS=$!

ok=0
for _ in $(seq 1 150); do
  grep -q "就绪 http" "$OUT/srv.log" 2>/dev/null && { ok=1; break; }
  grep -qE "❌|装配失败|拒绝启动" "$OUT/srv.log" 2>/dev/null && { echo "[$TAG] ❌ 启动被拒"; tail -5 "$OUT/srv.log"; break; }
  sleep 2
done
[ "$ok" = 1 ] || { echo "[$TAG] ❌ 超时"; kill "$NSYS" 2>/dev/null; exit 3; }
grep -m1 "选卡" "$OUT/srv.log" || true
sleep 3

req() { # req <max_tokens>
  curl -s "$API" -H 'Content-Type: application/json' \
    -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"在深海下面有一座城市它的名字没有人记得。请续写一段话。\"}],\"max_tokens\":$1,\"temperature\":0,\"ignore_eos\":true}" \
    > "$OUT/req$1.json"
  python3 - "$OUT/req$1.json" <<'PY'
import json,sys
u=json.load(open(sys.argv[1])).get('usage',{})
print(f"  usage: prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}")
PY
}
echo "[$TAG] 预热(4 tok)"; req 4
sleep 2
echo "[$TAG] 主测(64 tok)"; req 64
sleep 2

for p in $(ss -ltnp 2>/dev/null | grep -E ":$PORT([^0-9]|$)" | grep -o 'pid=[0-9]*' | cut -d= -f2); do kill -TERM "$p" 2>/dev/null || true; done
sleep 8
kill -TERM "$NSYS" 2>/dev/null || true
wait "$NSYS" 2>/dev/null
ls -la "$OUT/nsys.nsys-rep" 2>/dev/null && echo "[$TAG] ✓ 报告:$OUT/nsys.nsys-rep(分析:python3 tools/nsys_decode_account.py $TAG)"
