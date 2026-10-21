#!/usr/bin/env bash
# TP std A/B 跑手(2026-10-17;TP 线性能面的**唯一量法**)
#
# 一条命令跑完一"臂" = 起服 → `llm_speedtest --flavor std`(五点三轮)→ 停服;
# `ab` 模式把 TP1 对照臂与 TP2 目标臂**依次跑完并打差值表**。
#
# 为什么要有它:本文件的配方原先只活在 `/tmp`(tmpfs,**重启即失**)+
# `testdata/args/*.toml` 的注释里 ⇒ 编码进树(与 `tools/pw-ab-*.sh` 同款理由)。
#
# 用法(在 worktree 根;需先 `cargo build --release --bin server`):
#   tools/tp-ab-std.sh tp2            # 只跑 TP2 臂
#   tools/tp-ab-std.sh tp1            # 只跑 TP1 对照臂
#   tools/tp-ab-std.sh ab             # TP1 → TP2 依次跑 + 差值表
#   tools/tp-ab-std.sh tp2 --extra --ignore-eos   # `--extra` 之后原样转给 cli
#
# 口径纪律(**改任一条都会让两臂不可比**):
#   · 档位几何/白名单在 flavor(`profiles/qwen38_27b-3090-w4a16.toml` vs `qwen38_27b-3090-w4a16.toml`);
#     运行档 `testdata/args/bench-27b-tp{1,2}.toml` 只管模型路径与卡位;
#   · **TP1 与 TP2 必须逐项同参** —— 唯一允许的差别 = rank 数(单卡 vs 双卡);
#   · `--ignore-eos` = greedy 档跑满 256 tok(否则 OOD 词池下提前停 ⇒ decode 成伪影);
#   · 换卡位后先看启动日志 `[boot] 选卡(TP 档):rank0 = ordinal …` 自证;
#   · 结果落 `docs/bench/data/owl-tp-<日期>/` 或本工具输出的 `OUT_DIR`。
#
# ⚠️ 本机默认对:3080 20G = CUDA ordinal **2/3**(与 §24.5/§25.6 基线同对);
#    3090 Ti 对 = 0/1。**起服前先确认目标卡空闲**
#    (`nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv`)——
#    本机常有别的 agent 会话在 0/1 或 2/3 上跑 e2e(实测过一次,让基准读数失真)。
#
# 🔴 **端口纪律(2026-10-19 现场立案)**:本机可能同时有**多个 agent 会话**在别的
#    worktree(`packages/owl-worktrees/*`)里跑 owl server,而大家默认都绑 **8137**。
#    两边脚本若都“按端口收旧实例”,就会**互相击杀**(症状:客户端 `ConnectError:
#    All connection attempts failed` + **服务端日志无任何错** + 进程已消失 ⇒ 看上去
#    像“引擎静默崩”,实为被别的会话清掉。当天两次 A/B 就这样废了)。
#    ⇒ 规矩两条:
#      ① 端口占用时**只收自己 worktree 的实例**,别人的 ⇒ **响亮拒**(不禁自)
#         —— 见下面 preflight;
#      ② 与别的会话并行跑时用**专属端口**,且**运行档里必须声明同一个 `bind`**
#         (server 的 bind 来自配置文件,不是 `$PORT`:flavor 里是 8137 ⇒ 要用
#          `PORT=8141` + 一份把 `[runtime] bind` 改成 8141 的运行档):
#            PORT=8141 TP1_CFG=/tmp/xxx/tp1-8141.toml TP2_CFG=… tools/tp-ab-std.sh ab
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT" || exit 1

OUT_DIR="${OUT_DIR:-/tmp/owl-bench}"
mkdir -p "$OUT_DIR"
PORT="${PORT:-8137}"
API="http://127.0.0.1:${PORT}/v1/chat/completions"
MODEL="${MODEL:-qwen3.8-27b}"
SPEEDTEST="${SPEEDTEST:-/home/div/Documents/codes/packages/llm_speedtest}"

ARM="${1:-ab}"
shift || true
EXTRA=()
if [ "${1:-}" = "--extra" ]; then shift; EXTRA=("$@"); else [ $# -gt 0 ] && EXTRA=("$@"); fi

[ -x ./target/release/server ] || { echo "先 cargo build --release --bin server"; exit 2; }

# 目标卡空闲自检(非阻断,只警告 —— 别的会话可能正要起)
busy() {
  nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader 2>/dev/null \
    | grep -c . || true
}
if [ "$(busy)" != "0" ]; then
  echo "⚠️ 有别的进程占卡(下面可能不止本轮);读数可能失真:"
  nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_memory --format=csv,noheader
fi

# 一臂 = 起服 → std → 停服。profile 路径与运行档由调用方给(两臂唯一差别在 rank 数)。
run_arm() {
  local tag="$1" flavor="$2" cfg="$3"
  local log="$OUT_DIR/$tag.log" csv="$OUT_DIR/$tag-std.csv" json="$OUT_DIR/$tag-std.json"
  : > "$log"
  # 军规:按监听 PID 收旧实例,不用 `pkill -f`(模式会匹配自身命令行而自杀);
  # 且**只收自己 worktree 的**:别的会话的实例 ⇒ 响亮拒(不互相击杀,见文件头端口纪律)
  # 匹配**任意地址**上的该端口(0.0.0.0/[::]/127.0.0.1)—— 原实现只匹配
  # `127.0.0.1:$PORT`,别的会话绑 `0.0.0.0` 时漏检 ⇒ 装载完才在 bind 处 panic
  # (2026-10-19 现场:白装一次 27B)。`([^0-9]|$)` 防 `8137` 误配 `81370`。
  for p in $(ss -ltnp 2>/dev/null | grep -E ":$PORT([^0-9]|$)" | grep -o 'pid=[0-9]*' | cut -d= -f2); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
    # 归属判定:cmdline 含 ROOT **或 cwd == ROOT**(2026-10-19 现场:自己起的实例
    # 用 `./target/release/server` 相对路径 ⇒ cmdline 里没有 ROOT 字样 ⇒ 原判据
    # 把自家实例误判成"别的会话"而拒跑。cwd 判定补上这条缝)
    pcwd=$(readlink "/proc/$p/cwd" 2>/dev/null || true)
    case "$cmd$pcwd" in
      *"$ROOT"*) echo "[$tag] 收旧实例 pid=$p(本 worktree)"; kill "$p" 2>/dev/null || true ;;
      *) echo "[$tag] ❌ 端口 $PORT 被**别的会话**占用:pid=$p ⇒ $cmd";
         echo "[$tag]    规矩:换 PORT(并在运行档里声明同一 bind),或先等它跑完 —— 本脚本不代为清理"
         return 3 ;;
    esac
  done
  sleep 3
  setsid nohup ./target/release/server -p "profiles/$flavor.toml" "$cfg" --log-level warn \
    >> "$log" 2>&1 < /dev/null &
  local srv=$!
  local ok=0
  for _ in $(seq 1 150); do
    grep -q "就绪 http" "$log" 2>/dev/null && { ok=1; break; }
    grep -qE "❌|装配失败|拒绝启动" "$log" 2>/dev/null && { echo "[$tag] ❌ 启动被拒"; tail -6 "$log"; break; }
    sleep 2
  done
  if [ "$ok" != 1 ]; then echo "[$tag] ❌ 超时/被拒"; kill "$srv" 2>/dev/null; return 3; fi
  grep -m1 "选卡" "$log" || echo "[$tag] (单卡档:无'选卡'行)"
  local -a cli_extra=()
  [ "${#EXTRA[@]}" -gt 0 ] && cli_extra=("${EXTRA[@]}")
  ( cd "$SPEEDTEST" && uv run cli --api-url "$API" --model "$MODEL" \
      --flavor std --notes "$tag" --csv "$csv" --json "$json" --ignore-eos "${cli_extra[@]}" ) \
    2>&1 | tail -12
  kill -TERM "$srv" 2>/dev/null; sleep 5; kill -9 "$srv" 2>/dev/null
  echo "[$tag] stopped(数据:$csv)"
}

TP1_CFG="${TP1_CFG:-testdata/args/bench-27b-tp1.toml}"
TP2_CFG="${TP2_CFG:-testdata/args/bench-27b-tp2.toml}"

case "$ARM" in
  tp1) run_arm tp1 qwen38_27b-3090-w4a16 "$TP1_CFG" ;;
  tp2) run_arm tp2 qwen38_27b-3080_tp2-w4a16  "$TP2_CFG" ;;
  ab)
    run_arm tp1 qwen38_27b-3090-w4a16 "$TP1_CFG"
    run_arm tp2 qwen38_27b-3080_tp2-w4a16  "$TP2_CFG"
    python3 - "$OUT_DIR/tp1-std.csv" "$OUT_DIR/tp2-std.csv" <<'PY'
import csv, sys
def load(p):
    rows = []
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.reader(f):
            if len(r) >= 9 and r[0].strip().isdigit():
                rows.append((int(r[0]), float(r[5]), float(r[8])))  # ctx, prefill t/s, decode t/s
    return rows
a, b = load(sys.argv[1]), load(sys.argv[2])
pa = sum(x[1] for x in a) / len(a); pb = sum(x[1] for x in b) / len(b)
da = sum(x[2] for x in a) / len(a); db = sum(x[2] for x in b) / len(b)
print("\n══════ TP A/B(std;同二进制;唯一差别 = rank 数)══════")
print(f"  prefill  TP1 {pa:8.2f} t/s   TP2 {pb:8.2f} t/s   比值 {pb/pa:.3f}×")
print(f"  decode   TP1 {da:8.2f} t/s   TP2 {db:8.2f} t/s   比值 {db/da:.3f}×")
PY
    ;;
  *) echo "用法:$0 {tp1|tp2|ab} [--extra <给 cli 的额外参数>]"; exit 1 ;;
esac
