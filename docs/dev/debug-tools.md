# 调试工具速查(owl server / engine)

前提:服务已起(例:`cd packages/owl && ./target/release/server -p profiles/qwen38_27b-3090-w4a16-dflash.toml testdata/server/agent-27b.toml --log-level info`)。
下面例子默认端口 8137,按实际改。

## 1. 诊断端点(`GET`,全部只读)

| 端点 | 拿什么 | 例 |
|---|---|---|
| `/health` | 存活 + 模型名 | `curl -s 127.0.0.1:8137/health` |
| `/v1/models` | OpenAI 兼容模型表 | `curl -s 127.0.0.1:8137/v1/models` |
| `/debug/metrics[/<prefix>]` | **分相账**(每 tag:`count/total_ms/avg_ms/p50_ms/p99_ms`) | `curl -s 127.0.0.1:8137/debug/metrics/prefill` |
| `/debug/launches[/<n>[/<kind>]]` | 最近 n 条**发射记录**(按发射序) | `curl -s 127.0.0.1:8137/debug/launches/50` |
| `/debug/log` | 日志面状态;`/debug/log/level/info` 在线提级;`/debug/log/tags/<allow>/<deny>` 切过滤 | `curl -s 127.0.0.1:8137/debug/log/level/info` |
| `/debug/step` | 单步门状态;`POST /debug/step` 放行 1 步;`/debug/step/mode/<off\|sticky\|single>`;`/debug/step/ms/<n>`;`/debug/step/every/<n>` | `curl -s 127.0.0.1:8137/debug/step` |
| `/debug/perf[/<prefix>][?epoch=1]` | 性能框架账(带 self/树形;`epoch=1` = 只看最近一次 epoch 之后) | `curl -s '127.0.0.1:8137/debug/perf?epoch=1'` |

### metrics 常用 tag 族(tag 名字即契约)

- **prefill**:`prefill.chunk`(每 chunk host 墙时)、`prefill.bnd`(边界事件：快照 + 召回/打分/调和;
  与 `prefill.chunk` 分「前向 / 边界」两半)、`prefill.graph`(桶图**发射**成本,异步 ⇒ 天然小)、
  `prefill.bucket.hit[.t<n>]` / `.bypass.pad` / `.pad_rows[_skipped]` / `prefill.eager.{chunks,tokens}`、
  `prefill.extend`(草稿 KV 物化)、`dflash.prefill.encode`(逐块 DFlash2 encode)。
- **spec**:`spec.round[.deg|.spec]` / `spec.verify` / `spec.propose[.step|.read|.btwt]` / `spec.extend` / `spec.snap` / `spec.emit` / `spec.mtp.window.{rounds,full,visible_tokens}`。
- **decode**:`decode.step[.deg]` / `decode.sync`。
- 计数器(counter)与计时器(timer)共用一个 tag 空间:采样面是 `count/total/p50/p99`,事实面是 `counter`。

### ⚠️ metrics 是**累计快照**,不是窗口

`/debug/metrics` 给的是**自 boot 累计**。要读"某一相位的均值",**取两次快照差分**:
`Δtotal_ms / Δcount`(均值可差;p50/p99 **不可差** —— 需要单相位数据就重启服务或单跑一个相位)。

## 2. 日志

- 五级 + tag 前缀白/黑名单;`--log-level <off|error|warn|info|debug|trace>`、`--log-file <path>`。
- **缺省级别 = warn**(2026-10-14 裁决):装配/装载报告默认静默,开机 ~10 行;要全量报告用 `--log-level info`(≈335 行)。
- 运行期提级:`curl 127.0.0.1:8137/debug/log/level/info`(不必重启)。
- **拒启类错误与就绪横幅恒输出**(裸 eprintln,不随级别静默)——"服务没起来"一定能在日志里看到原因。
- ⚠️ `kernels` crate 的 `[kernels]/[selfcheck]/[xqa]` 13 处**尚未收编**(它不许依赖 `owl-shared`,需 `LogSink` 注入设计)⇒ 这部分始终直出。

## 3. 探针与诊断档(`[probes]` 配置,不是 env)

诊断旋钮走配置文件(env 已退出正式配置链):

```toml
[probes]
pf_fence = true      # prefill from_host 叶子建完插流栅栏(判 H2D 与 COMPUTE 跨流序竞态)
pf_bisect = true     # prefill 逐层 checksum 二分
pf_stages = true     # prefill 分阶段
dflash_probe = true  # 草稿侧取证(concat 父块落盘等)
gdn_dbg = true       # reset_gdn 后 GDN rec 格画像
debug = true         # 图捕获序列化走查
```

**律:诊断在场 ⇒ 快路径让位,且**不静默**。** 例:prefill 桶图已于 2026-10-21 退役;
历史行为是诊断档在场时让位 eager。现在 prefill 恒 eager,诊断面不再与桶冲突。

## 4. 常用取证组合

```bash
# 桶图到底有没有在跑(产线稳态)
curl -s 127.0.0.1:8137/debug/metrics/prefill.bucket   # hit / bypass.pad / pad_rows[_skipped]
curl -s 127.0.0.1:8137/debug/metrics/prefill.graph    # 发射次数与耗时(0.2ms 量级 = 正常)

# 一次请求的相位账(先清空心思:看差分)
curl -s 127.0.0.1:8137/debug/metrics/spec > /tmp/a.json
# ……(打一次请求)
curl -s 127.0.0.1:8137/debug/metrics/spec > /tmp/b.json
python3 -c "import json;a={r['tag']:r for r in json.load(open('/tmp/a.json'))['rows']};b={r['tag']:r for r in json.load(open('/tmp/b.json'))['rows']};
[print(f\"{t:28s} Δn={b[t]['count']-a.get(t,{'count':0})['count']:4d} 均={(b[t]['total_ms']-a.get(t,{'total_ms':0})['total_ms'])/max(1,b[t]['count']-a.get(t,{'count':0})['count']):8.2f}ms\") for t in b]"
```

## 5. 排障顺序(踩过才写的顺序)

1. **进程还在吗**:`ss -ltnp | grep <port>`;`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv`。
   进程静默消失且日志无 panic ⇒ 先怀疑**外部 kill / 抢卡**(本工作区多会话常态)。
2. **是不是 OOM**:`grep '^oom_kill' /proc/vmstat`(自 boot 累计,**无需 root**)。`0` = 内核没杀过。
3. **可复现吗**:同二进制同序列复跑一遍(单次 ≠ 结论)。
4. **卡被占了吗**:启动需要**全空卡**(27B 档 boot 瞬时峰值 ~23.9G/24.5G,占几百 MB 就会权重装载 OOM)。
5. **要细节** ⇒ `/debug/log/level/info`;要分相 ⇒ metrics 差分;要逐核 ⇒ `profiling.md` 的 nsys。
6. 记档:结论 + 原始产物路径(`/tmp/...` 会被清,关键产物拷到 `roadmap.local/artifacts/`)。

## 6. 杀进程纪律(**反面教材**)

- **禁 `pkill -f <名字>`**:模式会匹配**自己这条命令行** —— 已实测把自己 kill 掉(AGENTS.md 明令)。
- 正确:先取 PID 列表再 kill。

```bash
pid=$(ss -ltnp | sed -n 's/.*127\.0\.0\.1:8137.*pid=\([0-9]*\).*/\1/p' | head -1); kill -TERM "$pid"
# worker 会改名(VLLM::Worker_TP* / sglang::scheduler_TP* / owl 的 server 子进程),用 exe 双验:
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do echo "$p $(readlink -f /proc/$p/exe)"; done
```

## 7. 卡序 ≠ nvidia-smi 序号(必读)

owl 的 `runtime.device` 是 **CUDA ordinal**(`cudaSetDevice` 的序),与 `nvidia-smi` 的 `index` **不是一回事**。
拿不准就用 PCI 地址对:

```bash
nvidia-smi --query-gpu=index,name,pci.bus_id,memory.used --format=csv
python3 - <<'PY'   # 打印 CUDA ordinal 序(与 owl 同序)
import ctypes,glob
rt=ctypes.CDLL(glob.glob('/home/div/Documents/codes/packages/comfyui-test/.venv/lib/python3*/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12')[0])
n=ctypes.c_int(); rt.cudaGetDeviceCount(ctypes.byref(n)); buf=ctypes.create_string_buffer(256)
for i in range(n.value):
    rt.cudaDeviceGetPCIBusId(buf,256,i); print(f'CUDA ordinal {i}: {buf.value.decode()}')
PY
```
