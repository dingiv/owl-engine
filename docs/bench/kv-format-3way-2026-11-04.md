# KV 格式对照:INT8 vs FP8(+shadow)—— 32k/64k/100k(两轮)

> 2026-11-04 · 单 3090 Ti(`0000:41:00.0`)· Qwen3.8-27B AWQ **W4A8** ·
> `prefill_chunk=2048` · `prefix_cache=false` · spec off · max_seq 131072。
> 目的:给「30 系默认 INT8 主池」一个同卡同模型的对照读数,并量化
> `kernel.prefill_shadow`(FP8→F16 prefill staging)在长上下文的价值。
>
> **口径**:两轮独立复跑,每轮每个 (变体 × 长度) 都 **fresh server**;
> 三种变体用同 seed + 同 nonce ⇒ 同一长度下 prompt 逐字节相同。
> 第一轮另测过「FP8 无 shadow」作参照(本轮按需停测,数据保留在 §2.4)。

## 1. 变体(只差 KV 池与 shadow 开关)

| 变体 | profile | KV 主池 | prefill 注意力核 |
|---|---|---|---|
| `int8` | `qwen38_27b-3090-int8.toml` | int8 K + packed V(无 shadow) | 本仓 P8 核(2026-11-04 +23%) |
| `fp8shadow` | `qwen38_27b-3090-fp8.toml` + `kernel.prefill_shadow=true` | fp8 e4m3 + **f16 staging shadow** | FI(f16) |

权重/量化/W4A8 GEMM 全同;`prefix_cache=false`(纯预填口径);输出 32 tok。

## 2. 结果(两轮)

### 2.1 预填吞吐 t/s

| 长度 | INT8 r1 | INT8 r2 | **INT8 均值** | FP8+shadow r1 | r2 | **shadow 均值** | INT8 领先 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32k | 2161.9 | 2150.1 | **2156.0** | 2063.9 | 2063.6 | **2063.8** | **+4.5%** |
| 64k | 1886.2 | 1876.6 | **1881.4** | 1816.8 | 1818.6 | **1817.7** | **+3.5%** |
| 100k | 1646.4 | 1646.3 | **1646.4** | 1599.0 | 1595.4 | **1597.2** | **+3.1%** |

### 2.2 TTFT(ms)

| 长度 | INT8 r1 / r2 → 均值 | FP8+shadow r1 / r2 → 均值 |
|---:|---|---|
| 32k | 14 804 / 14 885 → **14 845** | 15 507 / 15 509 → **15 508** |
| 64k | 33 933 / 34 105 → **34 019** | 35 228 / 35 193 → **35 211** |
| 100k | 60 740 / 60 744 → **60 742** | 62 541 / 62 682 → **62 611** |

### 2.3 Decode t/s(单流,32 tok)

| 长度 | INT8 r1 / r2 → 均值 | FP8+shadow r1 / r2 → 均值 |
|---:|---|---|
| 32k | 41.21 / 40.83 → 41.02 | 43.16 / 42.68 → **42.92** |
| 64k | 39.16 / 39.04 → 39.10 | 41.57 / 40.89 → **41.23** |
| 100k | 37.33 / 37.47 → 37.40 | 39.47 / 39.12 → **39.30** |

### 2.4 参照:FP8 **无 shadow**(仅第一轮,已停测)

| 长度 | 预填 t/s | TTFT ms | decode t/s |
|---:|---:|---:|---:|
| 32k | 2068.7 | 15 470 | 42.91 |
| 64k | 1737.9 | 36 829 | 40.89 |
| 100k | 1475.3 | 67 786 | 39.14 |

⇒ 对照 §2.1:shadow 的收益随上下文增长(plain→shadow:32k +0.2%、
64k +4.5%、100k **+8.4%**),但 **INT8 无 shadow 仍比 fp8+shadow 快**,
且不占那份 staging 显存(100k≈200 MiB/rank)。

## 3. 读法

1. **预填:INT8 三点全场第一**(+4.5 / +3.5 / +3.1% vs fp8+shadow)。
   机制:sm86 **没有 fp8 MMA**,fp8 池的 prefill 必须 dequant→half 再算;
   int8 QK 直接吃 s8 张量核(f16 的 2× 吞吐)。
2. **decode:FP8 略快 4.6~5.5%**(每步读全 KV;int8 还要读 per-token scale
   并反量化,fp8 用指数位免掉这层)。绝对差只有 ~1.6 t/s。
3. **可复现性**:同协议两轮差异 ≤0.5%(100k 的 int8 两轮 1646.4 / 1646.3)。
   早前看到的 ±4% 漂移**不是时钟**,而是"同一 server 连跑多个长度"导致的
   KV 池累积/驱逐(实测 64k 那次 TTFT 34 s→147 s) ⇒ **长上下文对照必须
   每点 fresh server**。这一点现在是本卷最重要的方法论结论。

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
# fp8 变体:换 qwen38_27b-3090-fp8.toml(端口 8144)+ kernel.prefill_shadow=true
```

原始 JSON(每点的 TTFT/decode 逐请求时间戳)落在
`docs/bench/data/kv-format-3way/` —— 该目录被 `.gitignore` 挡着(数据不入库,
与其它 bench 数据同规);要复现请按上面的命令重跑。
