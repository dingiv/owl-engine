# KV 格式三向对照:INT8 vs FP8(+shadow) vs FP8(无 shadow)—— 32k/64k/100k

> 2026-11-04 · 单 3090 Ti(`0000:41:00.0`)· Qwen3.8-27B AWQ **W4A8** ·
> `prefill_chunk=2048` · `prefix_cache=false` · spec off · max_seq 131072。
> 目的:给「30 系默认 INT8 主池」一个同卡同模型的对照读数,并量化
> `kernel.prefill_shadow`(FP8→F16 prefill staging)在长上下文的价值。

## 1. 协议(为什么这样跑)

- **每个 (变体 × 长度) 都 fresh server**:同一 server 连跑三个长度会让 KV 池
  累积 → 触发驱逐/重算(实测 64k 那次 TTFT 从 34 s 涨到 147 s),读数作废。
- 三种变体用**同 seed + 同 nonce** ⇒ 同一长度下 prompt 逐字节相同
  (`prompt_sha256` 见数据文件)。
- `prefix_cache=false`:避免前缀命中把 prefill 时间抹掉(纯预填口径)。
- 输出 32 tok:让 TTFT/预填吞吐不被 decode 干扰;decode 速率另列。
- 变体只差 KV 池与 shadow 开关,权重/量化/W4A8 GEMM 全同:

| 变体 | profile | KV 主池 | prefill 注意力核 |
|---|---|---|---|
| `int8` | `qwen38_27b-3090-int8.toml` | int8 K + packed V(无 shadow) | 本仓 P8 核(今日 +23%) |
| `fp8shadow` | `qwen38_27b-3090-fp8.toml` + `kernel.prefill_shadow=true` | fp8 e4m3 + **f16 staging shadow** | FI(f16) |
| `fp8plain` | 同上 + `kernel.prefill_shadow=false` | fp8 e4m3(现场反量化) | FI(fp8→half) |

## 2. 结果(单轮)

### 预填吞吐 t/s

| 长度 | **INT8** | FP8+shadow | FP8 无 shadow | INT8 vs shadow | INT8 vs plain |
|---:|---:|---:|---:|---:|---:|
| 32k | **2161.9** | 2063.9 | 2068.7 | **+4.8%** | +4.5% |
| 64k | **1886.2** | 1816.8 | 1737.9 | **+3.8%** | +8.5% |
| 100k | **1646.4** | 1599.0 | 1475.3 | **+3.0%** | **+11.6%** |

### TTFT(ms)

| 长度 | INT8 | FP8+shadow | FP8 无 shadow |
|---:|---:|---:|---:|
| 32k | **14 804** | 15 507 | 15 470 |
| 64k | **33 933** | 35 228 | 36 829 |
| 100k | **60 740** | 62 541 | 67 786 |

### Decode t/s(单流,32 tok)

| 长度 | INT8 | FP8+shadow | FP8 无 shadow |
|---:|---:|---:|---:|
| 32k | 41.21 | **43.16** | 42.91 |
| 64k | 39.16 | **41.57** | 40.89 |
| 100k | 37.33 | **39.47** | 39.14 |

## 3. 读法

1. **预填:INT8 全场第一**(32k/64k/100k 分别 +4.8/+3.8/+3.0% vs fp8+shadow)。
   这与内核侧一致:int8 QK 用 s8 张量核(sm86 上 2× f16 吞吐),而 fp8 池
   在 sm86 **没有 fp8 MMA**,prefill 必须 dequant→half 再算。
2. **decode:FP8 略快 4~5%**(每步都要读整个 KV;fp8 的 1 B/elem 与 int8 相同,
   但 int8 还要读 per-token scale + 反量化,FP8 的指数位省掉这层)。
   量级上 decode 的差距(≈1.5 t/s)远小于预填差距(≈50~170 t/s)。
3. **shadow 的价值随上下文增长**:fp8 加 shadow 在 32k 只有 +0.2%,64k +4.5%,
   100k **+8.4%**(TTFT 67.8→62.5 s)—— 与「shadow = prefill 先反量化进 f16
   staging」的机制吻合:上下文越长,反量化占比越大。
   **但 INT8(无 shadow)仍比 fp8+shadow 快**,且不多占那份 staging 显存。
4. 口径提醒:单轮、无锁频(同二进制重复跑有 ±2~4% 漂移),所以 3~5% 的差距
   看**方向一致性**(三点同向);11.6% 那档(100k vs fp8 无 shadow)已超出噪声。

## 4. 复现

```bash
# 例:int8 @100k(fresh server)
./target/release/server -p profiles/measurement/qwen38_27b-3090-int8.toml \
  --set 'parallel.device=["0000:41:00.0"]' --set 'model.quant="w4a8"' \
  --set 'pool.prefix_cache=false' --log-level warn &
cd ../llm_speedtest && uv run cli \
  --api-url http://127.0.0.1:8141/v1/chat/completions \
  --lengths 100000 --rounds 1 -c 1 --output-length 32 --timeout 300000 \
  --prompt-nonce kvbench-100000 --json out.json
# fp8 变体:换 qwen38_27b-3090-fp8.toml(端口 8144)+ kernel.prefill_shadow=true/false
```

原始 JSON(每点的 TTFT/decode 逐请求时间戳)落在
`docs/bench/data/kv-format-3way/` —— 该目录被 `.gitignore` 挡着(数据不入库,
与其它 bench 数据同规);要复现请按上面的命令重跑。
