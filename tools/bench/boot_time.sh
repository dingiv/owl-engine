#!/bin/bash
# 装载性能基线(boot→health)+ 分相摘取 —— loader 改造的**回归门**用探针。
#
# 用法:
#   tools/bench/boot_time.sh awq-spec    [设备BDF]
#   tools/bench/boot_time.sh awq-nospec  [设备BDF]
#   tools/bench/boot_time.sh gguf        [设备BDF]
#
# 输出一行(可直接粘进设计文档的基线表):
#   [tag] wall_to_health=9770ms load=7.81s assemble=9.41s draft=0.42s
#
# 纪律(与 docs/devlog/dev/loader-refactor-ModelSource-设计-2026-11-04.md §5 同源):
#   · 每档连跑 3 次:**第 1 次 = 冷(页缓存未热),2~3 次 = 热**;回归判定用热态中位数;
#   · 同机同卡(BDF 固定,默认 41:00 = 单卡 3090 Ti 测试卡),同 profile/参数;
#   · 起机期间别跑别的 GPU 负载(共享 PCIe/HBM);
#   · 只跑装载、不发请求(装载速度与推理无关)。
set -u
OL=$(cd "$(dirname "$0")/../.." && pwd)
kind=${1:?用法: boot_time.sh <awq-spec|awq-nospec|gguf|awq-spec-8k> [BDF]}
bdf=${2:-41:00.0}
port=${PORT:-8155}
tag="$kind-$(date +%H%M%S)"
log="/tmp/owl-boot-$tag.log"

case "$kind" in
  awq-spec)
    prof="$OL/profiles/measurement/qwen38_27b-3090-int8.toml"
    extra=( --set 'runtime.max_seq=40960'
            --set 'spec.depth=7' --set 'spec.lookup=true'
            --set "spec.draft_path=\"$OL/models/syvai/Qwen3.8-27B-DFlash2-W4A16\"" ) ;;
  awq-nospec)
    prof="$OL/profiles/measurement/qwen38_27b-3090-int8.toml"
    extra=( --set 'runtime.max_seq=40960' --set 'spec.depth=0' --set 'spec.lookup=false' ) ;;
  gguf)
    prof="$OL/profiles/qwen38_27b-gguf_3090.toml"
    extra=( --set 'runtime.max_seq=16384' --set 'spec.depth=0' --set 'spec.lookup=false' ) ;;
  *) echo "未知档 $kind(awq-spec|awq-nospec|gguf)"; exit 2 ;;
esac

t0=$(date +%s%N)
"$OL/target/release/server" -p "$prof" \
  --set "parallel.device=[\"0000:$bdf\"]" \
  --set "server.bind=\"127.0.0.1:$port\"" \
  "${extra[@]}" --log-level info > "$log" 2>&1 &
pid=$!
ok=0
for _ in $(seq 1 240); do
  if curl -s -m 2 "http://127.0.0.1:$port/health" 2>/dev/null | grep -q '"ok":true'; then ok=1; break; fi
  sleep 0.5
done
t1=$(date +%s%N)
if [ $ok -ne 1 ]; then echo "[$tag] BOOT-FAIL(log=$log)"; tail -8 "$log"; kill $pid 2>/dev/null; exit 1; fi
wall=$(( (t1 - t0) / 1000000 ))
pick() { grep -m1 "$1" "$log" | grep -oE '[0-9.]+s$' | head -1 | tr -d 's'; }
echo "[$tag] wall_to_health=${wall}ms load=$(pick '权重/tokenizer/rope 装载')s \
assemble=$(pick '引擎装配(run)总计')s draft=$(pick 'DFlash2 草稿装载')s log=$log"
kill $pid 2>/dev/null
wait $pid 2>/dev/null
for _ in $(seq 1 60); do
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  [ "$n" = "0" ] && break
  sleep 1
done
