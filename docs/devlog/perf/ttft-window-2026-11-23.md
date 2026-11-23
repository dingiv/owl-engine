# TTFT 实测数据（2026-11-23，worktree `ttft-1s`）

> 目的：把 1 张图 / ~1k 新增 token 的 TTFT 拆开，区分
> **模型 prefill**、**vision**、**切块固定税**、**快照换入换出**。
> 生产 8135 未改；下列 A/B 在 3090 TP2 本地档上跑，window 默认关。

## 1. 生产 8135 实际 agent（只读采集）

3080 TP2 / INT8 / DFlash2 d7 / 3 图窗口。

| 项 | 数值 |
|---|---:|
| 累计 turn | 13 |
| 3 图 turn | 2（prompt 185,540；image_tokens 2,640；cached 182,848） |
| `prefill.chunk` | 538 次 / 285.5s / avg **530.6ms** / p99 **1003ms** |
| `prefill.bnd` | 269 次 / 5.51s / avg **20.5ms** / p99 159ms |
| `pf.vlm` | 3 次 / 1.37s / avg **456ms** / p99 **1369ms** |
| `srv.seg.swap` | 174 次 / 9.32s / avg 53.6ms / p99 **2889ms** |
| `spec.round` | 2361 次 / 92.1s / avg **39.0ms** |

当前 epoch（短文本 24k，非长图 turn）：`prefill.chunk` 6 次 776ms，
`prefill.bnd` 3 次 153ms，`srv.seg.swap` 8 次 21ms。

## 2. 200k 前缀 + ~1k 追加（3090 TP2，window 8k/32k）

文本追加（base 200k，cached 199,968）：

| 配置 | chunks | prefill.chunk | bnd | TTFT 备注 |
|---|---:|---:|---:|---|
| 全量 attention（window=0） | 992+32+9 | **2.50s** | 69ms | swap 3.46s 另计 |
| window=32k + sink 8k | 992+32+9 | **1.35s** | 26ms | |
| window=8k + sink 8k | 992+32+9 | **1.19s** | 21ms | |

图像追加（同 base，Qwen VL，1 图）：

| 图 | image tokens | new tokens | chunks | prefill.chunk | vision |
|---|---:|---:|---|---:|---:|
| 1280×720 | 880 | 915 | 864+32+19 | **1.05s** | tower 59ms / combine 97ms |
| 960×540 | 510 | 545 | 3 块（未取逐块尺寸） | **0.86s** | cached 时 0；新图约 50+38ms |

结论：window 8k 已把 attention 压到很小；**200k 下 0.5MP 图的模型 prefill
本身 0.86s**，加新图 vision 约 0.1s，模型侧约 0.96s。

## 3. 30k 前缀（快速 A/B，同一 3090）

| 图 | new tokens | server TTFT | prefill | tower | combine |
|---|---:|---:|---:|---:|---:|
| 960×540 | 526 | **745ms** | 680ms | 49ms | 38ms |
| 1280×720 | 896 | **877ms** | 810ms | 59ms | 93ms |

## 4. 关键判读

1. **1.2s 不是 1k token 的 GEMM/GDN 物理地板**：200k 文本 window=8k
   prefill 1.19s，但 chunk = 992 + 32 + 9；两个 <32/9 token 的块
   各约 0.19s，固定税约 0.4s。
2. 全量 attention 约 **1.3s**（2.50 → 1.19 的差值），与 16 个 full
   attention 层 × 200k × 1k 的量级相符。
3. 生产 8135 的 `srv.seg.swap` 当前 epoch 仅 21ms，但累计 p99 2.9s；
   合成 200k 文本/图请求仍会触发 **3.4s** 快照换入换出（snapshot churn）。
4. 0.5MP 图 + window 8k 在 200k 上模型 prefill 已 <1s；若快照 churn 不在
   关键路径，TTFT 可到 ~1.0s。0.92MP 图则约 1.1–1.2s。

## 5. 复现

- worktree：`packages/owl-worktrees/ttft-1s`（`feat/ttft-1s`）
- 配置：`profiles/qwen38_27b-3090_tp2.local.toml`（ignored，UUID 钉 3090 对）
- 启动：`--set kvmem.prefill_window_tokens=8192 --set runtime.prefill_chunk=2048`
- 观测：`/debug/perf/epoch/json`、`/debug/metrics`、`/debug/agent`；
  `--log-level info` 看 `[vlm-cost]`、`[chat] ... ttft=`
- 原始数据：`/tmp/ttft_*`（临时；本文件是收口表）

## 6. H2D 合并（2026-11-23 修复）

`swap_in` 原先逐个 `SwapPiece` 发 `memcpy_htod_async`；200k 恢复产生
**399,936 片**，命令开销把有效带宽压到 **1.97GB/s**。

修法：按 `(block, dev_off)` 排序，把同一层连续设备页合并成一次 H2D
（host 侧仍按各片自己的 connector slot/off 取）。

| 指标 | 修复前 | 修复后 |
|---|---:|---:|
| 单 rank `swap-in` | 1720ms / 1.97GB/s | **795ms / 4.26GB/s** |
| DMA 段 | 1417ms | **260ms** |
| 两 rank 合计 | 3.45s | **1.67s** |
| 稳态第二次 image turn | 3.45s | **1.20s（5.7GB/s）** |

## 7. 最终 30k 复验（window 8k + sink 8k）

| 图 | server TTFT | prefill | tower |
|---|---:|---:|---:|
| 960×540（526 tok） | **741ms** | 676ms | 54ms |
| 1280×720（896 tok） | **869ms** | 801ms | 155ms |

## 8. 200k 仍剩什么

- 模型 prefill（window 8k）：**0.85–1.07s**
- 新图 vision：~0.1s
- KV 前缀回灌：3.45s → **1.2–1.67s**（已合并）
- 余下 2–3s 是 **admission/上一轮收口等待**（背靠背请求才显形；agent 工具延迟下通常会消化），不是 prefill。

## 9. 已否证（不要重走）

- 直接 `Protected.chains` 保护整条前缀：admission 只看全量 floor，
  保护后无可腾页 ⇒ **35k 次准入排队**，已撤回。
- 保留最近会话设备表：会改变 base 链发布/复用面，实测出现
  `cached=0` + 200k 全量重算，已撤回。
- 下一步若继续追 1s：需要 **admission 前的前缀 floor** 与设备表保留
  一起设计（不能只做一半），或继续优化 KV 回灌/避免回灌。
