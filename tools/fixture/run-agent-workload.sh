#!/usr/bin/env bash
# 拉起一套隔离的 image-evict 测试 server → 跑 CLI agent workload → 关停。
#
# 用法:
#   tools/run-agent-workload.sh [suite] [cli 额外参数...]
#   suite = core | gui | gui-capability | image-evict | all
#
# 前置:
#   - 端口 $OWL_WORKLOAD_PORT(缺省 8135)空闲;脚本绝不抢占现有 server。
#   - 2×3080 空闲(TP2 测试档;3090 生产档可继续跑在 8134)。
#
# 环境:
#   OWL_WORKLOAD_PORT / OWL_WORKLOAD_LOG / OWL_WORKLOAD_MAX_SEQ
#   OWL_WORKLOAD_PROFILE(缺省 profiles/qwen38_27b-3080_tp2.toml)
#   OWL_WORKLOAD_SERVER_BIN / OWL_WORKLOAD_CLI_BIN
#
# 纪律:起服前查端口,退出时只 kill 自己记录的 PID;不用 pkill -f。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PORT="${OWL_WORKLOAD_PORT:-8135}"
SUITE="${1:-image-evict}"
if (($# > 0)); then shift; fi
LOG="${OWL_WORKLOAD_LOG:-/tmp/owl-agent-workload-${PORT}.log}"
PROFILE="${OWL_WORKLOAD_PROFILE:-profiles/qwen38_27b-3080_tp2.toml}"
MAX_SEQ="${OWL_WORKLOAD_MAX_SEQ:-131072}"
SERVER_BIN="${OWL_WORKLOAD_SERVER_BIN:-$ROOT/target/release/server}"
CLI_BIN="${OWL_WORKLOAD_CLI_BIN:-$ROOT/target/release/cli}"

if [[ ! -x "$SERVER_BIN" || ! -x "$CLI_BIN" ]]; then
    echo "[workload] release 二进制缺失,先构建 server + cli" >&2
    cargo build --release -p server -p cli
fi

# 端口占用 = 拒绝启动(避免打到别人的 server 造成假绿)。
if ss -tln | awk '{print $4}' | grep -Eq "[:.]${PORT}$"; then
    echo "[workload] 端口 ${PORT} 已被占用;先停掉现有 server(本脚本不会抢占)" >&2
    ss -tlnp | grep -E "[:.]${PORT} " >&2 || true
    exit 2
fi

echo "[workload] 启动测试 server: $PROFILE -> 127.0.0.1:${PORT} (leave-gap)" >&2
echo "[workload] 日志: $LOG" >&2
# 子 shell 内 exec ⇒ `$!` 就是 server 本体(不是 nohup/wrapper)，
# 清理时只 kill 这个 PID。
# 生产口径 = FP8 KV + 投机(profile 自带 spec.depth=7);不再强制 f16。
(
    exec "$SERVER_BIN" -p "$PROFILE" \
        --set "server.bind=127.0.0.1:${PORT}" \
        --set "pool.kv_fp8=true" \
        --set "pool.image_evict=true" \
        --set "pool.prefix_cache=true" \
        --set "pool.snap_max=2" \
        --set "pool.snap_host_mb=1024" \
        --set "runtime.max_seq=${MAX_SEQ}" \
        --set "runtime.implicit_session=true" \
        --log-level info >"${LOG}" 2>&1
) &
SERVER_PID=$!

cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        for _ in $(seq 1 20); do
            if kill -0 "$SERVER_PID" 2>/dev/null; then
                sleep 0.5
            else
                break
            fi
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[workload] server 未在 10s 内退出,发送 KILL" >&2
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
    fi
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# owl 启动应 <30s;这里给 60s 余量,同时校验子进程仍存活。
READY=0
for _ in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[workload] server 子进程已退出,日志尾部:" >&2
        tail -40 "$LOG" >&2 || true
        exit 1
    fi
    if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        READY=1
        break
    fi
    sleep 1
done
if [[ "$READY" != 1 ]]; then
    echo "[workload] server health 超时(60s),日志尾部:" >&2
    tail -40 "$LOG" >&2 || true
    exit 1
fi

"$CLI_BIN" --workload "$SUITE" --url "http://127.0.0.1:${PORT}" \
    --server-log "$LOG" "$@"
