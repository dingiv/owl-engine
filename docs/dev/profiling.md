# Profiler 与性能测量口径

## 0. 先选武器

| 问题 | 用什么 |
|---|---|
| "这一步/这一相花了多少" | **引擎内 metrics**(`/debug/metrics/*`,host 墙时口径)—— 真值,零客户端污染 |
| "哪个核慢、核间空隙多大、图里在跑什么" | **nsys**(逐核时间线) |
| "端到端快不快、用户等多久" | `packages/llm_speedtest`(统一口径,跨引擎可比) |
| "显存账" | boot 日志 `slab 峰值归因:` 行 + `nvidia-smi`/`[vram-trace]` 行 |

⚠️ **客户端逐帧计时会污染 decode**:Python 逐 SSE 帧消费曾把 28.5ms/步量成 51.3ms。
要 decode 单步就信 `decode.step`(带 `.deg` 是降级档)。

## 1. nsys

```bash
cd packages/owl && nsys profile --cuda-graph-trace=node --trace=cuda,nvtx,osrt \
  -o /tmp/owl-$(date +%H%M) --force-overwrite=true \
  ./target/release/server -p profiles/qwen38_27b-3090-w4a16-dflash.toml testdata/server/agent-27b.toml --log-level warn
# 另开:打 1-2 个请求(短输出),然后 Ctrl-C/停服,再用 nsys stats / GUI 看
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum /tmp/owl-*.nsys-rep
```

本仓已踩的坑:
- **图回放必须 `--cuda-graph-trace=node`**,否则时间线上只有一条 `Graph replay` —— 看不到图内逐核。
- **步界**:`lm_head` 每步一次,可当"一步"的分界标(图内融合臂会把多核并进一次 launch,别按 launch 数数步)。
- **CUPTI 语义**:`utilization` 是"任一 kernel 驻留"的占比,**不等于 busy 占比**;别拿它当"GPU 打满"的证据。
- nsys 会给所有流加开销,**别在 nsys 下测吞吐**;测吞吐用 `llm_speedtest`。

## 2. 带宽墙判定法(算力/带宽谁是瓶颈)

```
实测量 = 该核搬运字节数 ÷ 核耗时        # 与卡峰值带宽比
```
- **字节数要按 dtype 算对**:f16 = **2B/elem**(历史上有人按 0.5B 算,得出"带宽只有 47%"的假结论,
  真相 = 1021GB/s ≈ 100% 峰值)。
- 27B AWQ-INT4 marlin m=1:44.6MB/46µs ⇒ ~1021GB/s ⇒ **decode 的 GEMM 侧零空间**;剩余空间在
  非 GEMM 尾巴(state/胶水/gap)。

## 3. 统一测速口径(`packages/llm_speedtest`)

```bash
cd packages/llm_speedtest
uv run cli --api-url http://127.0.0.1:8137/v1/chat/completions --model qwen3.8-27b \
  --flavor std --notes "<谁在什么档上跑>" --json /tmp/x.json --csv /tmp/x.csv
# 小 prompt / 单 chunk 面:--lengths 32,64,96,128,192,256,480 --rounds 3 --output-length 256
# 分域(draft 接受率强相关):--prompt-domain math|prose
```
- **std 矩阵** = 512/1024/2048/4096/8192 × 3 轮、输出 256 tok、并发 1、**轮优先执行**(同一长度三轮连跑)。
- 读 **`usage.completion_tokens`**,不要数 SSE chunk(合并 chunk 会低估 ~2.1 倍)。
- prompt 由**随机词池**生成:重复文本会让 draft 接受率虚低;本地估 token 量用 ~1.4 tok/词。
- 服务端 metrics 是**累计**的:逐相位均值要用快照差分 `Δtotal/Δcount`(p50 不可差)。

## 4. A/B 方法论(三臂 + 噪声界)

一次可信的性能验收至少要有:

1. **同卡同时段**、同 config、同序列(先测后改;改完复测);
2. **同二进制对拍**:特性开关(如 `decode_fi = on|off`)在同一二进制里切 ⇒ 唯一变量;
3. **漂移对照臂**:跨二进制比较时,带上"新二进制 × 旧档"一臂,把 build 漂移和特性收益分开;
4. **噪声界**:同一臂跑两遍,给出同点差(实测:热漂移可达 +2…+44ms@8k,越长越明显);
5. **起前查卡**(全空)+ 记录 boot 峰值;跑完确认显存归零。

参考数据(2026-10-16 B3 三臂,27B 生产档 std 五点):特性收益 **prefill +1.2…+3.1%**、
TTFT@8192 **−203ms**;同二进制服务端 281 chunk 均值 **−7.4ms**、p50 **−10.6ms**;
`prefill.graph` 发射 **0.20ms/次**。两遍噪声界:同点差 ≤ +1.2ms(Boff 臂)。

## 5. 常见误读清单

- **拿旧二进制当基线**:归档二进制可能早于最近的合并(本次踩过:`/tmp/owl-server-baseline`
  早于当天的 refactor 合并,导致收益虚高)—— 基线必须**现场重编**并记录 md5。
- **拿"正在服务自己推理的进程"当测试臂**:它同时在处理 pi 的请求。
- **spec 档方差 ±20%**:单次采样别据以调参;解码指标看 `B TBT p50` 或多次中位。
- **greedy 提前停**:decode tok/s 会被截断口径影响,看好输出长度是否真的一直在跑。
- **padding/白算不计**:桶图类优化要把"白算行数 × 单行成本"算进净收益(本仓单行 ≈0.28–0.67ms,随 ctx 增长)。
- **"利用率"当性能**:见 §1 CUPTI 说明。
