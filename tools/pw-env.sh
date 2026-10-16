#!/bin/bash
# piecewise 性能工具链的公共前置(2026-10-16 抽出,修 G7):
#   ① 工作树 root 从**脚本自身位置**推导(原硬编码到 packages/worktrees/owl/perf
#      —— 那条工作树 2026-10-16 已删 ⇒ 两条脚本当场不可用);
#   ② 卡 = `CVD` 覆盖,缺省自动挑**最空闲**卡(按已用显存升序;打印所选卡;
#      原硬编码 GPU-89dffb90… = nvidia-smi #2,常被别的会话占);
#   ③ 端口 = `PORT` 覆盖(缺省 8138)。
# 用法:脚本里 `source "$(dirname "${BASH_SOURCE[0]}")/pw-env.sh"`。
set -u

PW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PW_PORT="${PORT:-8138}"

# 挑卡:按**剩余可用**显存降序(不是"已用最少"—— 20G 3080 在 27B spec 档会请求级 OOM,
# 有 24G 3090 Ti 空着时应优先选它)
PW_PICK="$(nvidia-smi --query-gpu=uuid,memory.total,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, '{gsub(/ /,""); print $1, $2-$3}' | sort -k2 -n -r | head -1)"
PW_CVD="${CVD:-$(echo "$PW_PICK" | cut -d' ' -f1)}"
PW_FREE="$(echo "$PW_PICK" | cut -d' ' -f2)"
if [ -z "$PW_CVD" ]; then
    echo "[pw] 挑不到可用卡(nvidia-smi 无输出?)—— 显式给 CVD=<卡 UUID>" >&2
    exit 1
fi

# 发版二进制必须在场(缺 = 响亮拒,不许静默跑别处的旧二进制)
if [ ! -x "$PW_ROOT/target/release/server" ]; then
    echo "[pw] 缺 $PW_ROOT/target/release/server —— 先跑:cargo build --release -p server" >&2
    exit 1
fi

# 停掉占用本端口的旧实例(按**监听 PID**;不用 pkill -f —— 会匹配自身命令行)
pw_kill_port() {
    local p
    for p in $(ss -ltnp 2>/dev/null | awk "/127\.0\.0\.1:${PW_PORT}/{print \$NF}" | grep -o 'pid=[0-9]*' | cut -d= -f2); do
        kill "$p" 2>/dev/null || true
    done
}

echo "[pw] root=$PW_ROOT 端口=$PW_PORT 卡=$PW_CVD(free ${PW_FREE}MiB)(可用 CVD= / PORT= 覆盖;卡缺省取剩余最多者)"
