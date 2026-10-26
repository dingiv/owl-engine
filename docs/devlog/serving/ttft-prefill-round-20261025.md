# TTFT / prefill 优化轮（2026-10-25）

> **2026-10-26 结案**：最终只保留 **leave-gap**。旧的 `rebase` / K 旋转路径
> （配置 `image_evict_mode` / `image_evict_delta_limit`、`kv_rope_shift` 核、
> rerope 计划与相关测试）已从主干删除；本文中相关章节仅作历史记录。


> 状态：**基线已冻结，第一轮归因完成；优化实验未开始。**
> 原始现场：`perf-runs/ttft-prefill-20261025/`（gitignored，含日志/CSV/回放脚本/配置指纹）。
> 服务：`qwen38_27b-3080_tp2.toml`，`0.0.0.0:8135`，3080×2 TP2，
> fp8 KV / spec d7 / snap2 / `snap_msg_min_gap=2048`。

---

## §0 一句话

这轮 59 条前端请求的 TTFT 平均 **4.55s**，其中 **4.51s 是 prefill**；
prefill 吞吐约 **973 t/s**，而每轮仍有约 **4.5k token 未命中缓存**。
decode 与投机健康（111.6 t/s、AL 4.61）。

所以 TTFT 优化的主战场不是 decode，而是：

1. **把 4.5k 未命中尾巴压下去**（缓存/历史形状）；
2. **在未命中量无法再压时，提高 prefill t/s**。

---

## §1 基线（59 条前端请求）

| 指标 | 平均值 | 中位数 | p90 |
|---|---:|---:|---:|
| TTFT | **4.55 s** | 4.55 s | 5.41 s |
| prefill_ms | 4.51 s | 4.50 s | 5.37 s |
| prefill_tps | **973 t/s** | 986 t/s | 1045 t/s |
| decode_tps | **111.6 t/s** | 109.9 t/s | 140.8 t/s |
| spec AL | **4.61** | 4.50 | 6.00 |

token 加权：prefill **1012 t/s**，decode **107.4 t/s**，AL **4.33**。

缓存：

```text
prompt 总量      1,934,317 tok
cached           1,664,032 tok
加权命中率       86.07%
单轮中位         86.39%
冷启动           1 / 59（turn 3：20,352 prompt，TTFT 13.9s）
```

排除冷启动后 TTFT 平均 **4.39s**，最高 6.12s。

---

## §2 关键发现：缓存每 ~2K 才前进一次

前端请求的 `cached_tokens` 不是逐轮增长，而是台阶式：

| turn | prompt | cached | miss |
|---:|---:|---:|---:|
| 9 | 25,286 | 22,080 | 4,934 |
| 10 | 25,567 | 22,048 | 3,519 |
| 11 | 25,712 | 22,080 | 3,632 |
| 12 | 26,016 | 22,080 | 3,936 |
| 13 | 26,263 | 22,080 | 4,183 |
| 14 | 26,585 | 22,080 | 4,505 |
| 15 | 26,821 | 22,080 | 4,741 |
| 16 | 26,974 | 22,080 | 4,894 |
| 17 | 27,317 | 22,080 | 5,237 |
| 18 | 27,474 | 22,080 | 5,394 |
| 19 | 27,722 | 24,160 | 3,562 |

`cached` 在 22,080 停住 10 轮，直到 24,160 才跳一次；
后续同样以约 2,208 token 为台阶（24,160 → 26,368 → 28,544 → 31,136 …）。
这个步长与配置 `snap_msg_min_gap = 2048` 高度吻合。

**假设 H1**：每轮新增的 assistant/tool 尾巴没有及时变成可恢复快照，
缓存只到上一条 user/message 边界；台阶之间的 2–5K 尾巴被重复 prefill。

### H1 的反证：纯文本/工具回放并不塌

用 `replay.py --mode tool` 做同一 session 的 18 轮长历史（15K 起、每轮 +300 tok）：

```text
round 1:  prompt=15,688  cached=15,360  miss=328
round 10: prompt=18,497  cached=18,176  miss=321
round 17: prompt=20,657  cached=20,320  miss=337
```

`snap_msg_min_gap=2048` 下仍然 **98% 命中**，每轮只 miss 约 300 token。
⇒ **H1 不是通用文本/工具回放的规律**；前端台阶更像是
**图像/工具结果形状 + 消息边界快照** 的组合问题，而不是单纯 min_gap。

### 图像删除对照（强证据）

`image_evict_sliding_probe.py` 三图滑窗、10 轮、512×512：

| 臂 | 删图后 cached | 每轮 prefill |
|---|---:|---:|
| 带 `owl_evict` | 1,056 / 1,424 | 368 tok（≈26%） |
| 不带 `owl_evict` | 32 / 1,262 | 1,230 tok（≈97%） |

不标记的删除直接把前缀缓存打穿；标记后只重算新图尾巴。
前端 59 轮日志里 `[img-evict] 删除识别` 只有 **1 次**（还是联调 demo），
说明**前端尚未稳定使用图片无感删除**，或者图片并未按窗口删除。

### 当前图像事实

旧批 `/debug/agent` 的 prompt 头/尾可见每轮约 **1,760 个 `<|image_pad|>`**
（3 张左右截图量级），但日志没有对应的 `owl_evict` 删除。
这是接下来必须先查清的输入事实。

---

## §3 已排除/待验证的杠杆

| # | 假设 | 状态 | 判据 |
|---|---|---|---|
| H1 | `snap_msg_min_gap=2048` 本身造成台阶 | **纯文本回放反证** | tool replay 98% 命中 |
| H2 | 图像删图未标记/未吸收造成 4K 级 miss | **强怀疑** | 前端 img-evict=1；对照臂 cached 塌到 32 |
| H3 | prefill 吞吐低（~1K）是 chunk/尾块配置 | 待测 | 冷 15K replay 1.49K t/s；需 256/512/1024 A/B |
| H4 | 图像 token 本身占 tail，无法避免 | 待测 | 需加 `images/image_tokens/evict_images` 请求日志 |
| H5 | snapshot 槽位不足（snap_max=2） | 待测 | 改 4 槽看 cached 台阶是否变密 |
| H6 | decode/AL 不是瓶颈 | 已确认 | decode 111.6 t/s，AL 4.61 |

---

## §4 优化实验单

### E0：把输入事实补齐（先做）

- `ReqFacts` 增加 `images / image_tokens / evict_images`，`受理` 日志打印；
- 前端下一批请求确认：三图窗口是否真的删图、是否带 `owl_evict`；
- 目标：先区分“图多”与“删图未吸收”。

### E1：图片无感删除闭环

如果 E0 显示图在删但没标记：
- 客户端所有可能 prune 的 image part 从第一次发送起带 `owl_evict: true`；
- 只删 image part，保留文字与 tool_call_id；
- 保持同一 session、全量历史；
- 预期：miss 从 4–5K 降到“新图 + 新工具文本”量级（约 0.5–1K）。

### E2：min_gap A/B（仅在 E0 排除图片后）

```text
snap_msg_min_gap = 2048 (baseline) vs 512 vs 256
```

固定同一前端 transcript / 回放，记录 cached、miss、TTFT、prefill、snapshot 开销。
不要在没有实际 transcript 的情况下只凭合成文本改生产值。

### E3：prefill 吞吐 A/B（已完成第一组）

用 `replay.py --mode tool --rounds 1`，三个**不同 seed / 不同前缀**的冷请求
（15K / 25K / 37K prompt），避免跨 session 内容缓存污染：

| chunk | 15K prompt | 25K prompt | 37K prompt |
|---:|---:|---:|---:|
| 512 (baseline) | 1,486 t/s | 1,446 t/s | 1,359 t/s |
| 1024 | 1,576 t/s (+6.1%) | 1,521 t/s (+5.2%) | 1,454 t/s (+7.0%) |
| 2048 | **1,605 t/s (+8.0%)** | **1,543 t/s (+6.7%)** | **1,477 t/s (+8.7%)** |

同批 TTFT：

```text
512:  10.28 / 17.35 / 27.66 s
1024:  9.70 / 16.58 / 25.87 s
2048:  9.57 / 16.33 / 25.45 s
```

⇒ 3080 TP2 采用 `prefill_chunk = 2048`（只改 3080 TP2 profile；3090 不动）。
预期对前端 4.5K 尾巴的 TTFT 改善约 5–8%，但这不是 86% 缓存问题的替代品。

`prefill_tail_merge=512` 保持；2048 chunk 已能覆盖尾块，暂不混调两个变量。

### E4：图像/工具 payload 瘦身

- 截图分辨率/编码是否可降（视觉 token 与准确性 A/B）；
- 工具结果文本是否可缩短；
- 保持 3 图窗口，不要让历史无限长。

---

## §5 验收线

| 指标 | 基线 | 本轮目标 |
|---|---:|---:|
| 前端 TTFT 中位 | 4.55 s | **≤3.0 s** |
| 缓存加权命中率 | 86.07% | **≥95%** |
| 每轮 miss | ~4.5K | **≤1.5K** |
| prefill t/s | 973 | **≥1.2K**（若 E2/E3 有效） |
| decode t/s | 111.6 | 不退化 |
| spec AL | 4.61 | 不退化 |

---

## §6 现场纪律

- 当前服务：`0.0.0.0:8135`，PID/启动命令见 `perf-runs/.../server-process.txt`；
- 原始日志和 CSV 在 `perf-runs/ttft-prefill-20261025/`，**不入 git**；
- 本次诊断代码（TTFT/prefill/decode/AL 日志、`TurnStats`、`/debug/agent` 64 轮）
  仍是未提交现场，patch 见 `perf-runs/.../git-diff.patch`；
- 任何 profile 实验只改一个变量，先保留 baseline 日志再重启；
- 3090 生产档不在本轮改动范围。

---

## §7 E0/E1 首轮实测（2026-10-25 晚，41 轮）

前端已经开始使用 `images=3 / evict=3`（三图窗口全部带 `owl_evict`），
缓存与 TTFT 都明显改善。但 `image_evict_delta_limit=4` 的**整段重建**
现在成了新的长尾。

### 7.1 缓存

| 口径 | n | token 加权命中率 | 单轮中位 | cold |
|---|---:|---:|---:|---:|
| 全部 | 41 | **81.82%** | 93.66% | 5 |
| warm（去掉 cold） | 36 | **94.78%** | 94.47% | 0 |
| 主 GUI 会话全部 | 26 | 81.35% | **97.37%** | 4 |
| 主 GUI 会话 warm | 22 | **96.29%** | **97.42%** | 0 |
| 旧的无标记会话 | 3 | 77.86% | 77.95% | 0 |

结论：

- 正常热轮已经到 **94–97%**，不再是上一批的 86%；
- 全部口径被 **5 次 cold** 拉低：2 次新会话冷启 + 3 次图片叠转到顶；
- `evict=3` 的会话稳定吸收；`evict=0` 的 3 轮明显更低。

### 7.2 TTFT

| 口径 | 平均 | 中位 | p90 |
|---|---:|---:|---:|
| 全部 41 | 4.78 s | **1.98 s** | 8.00 s |
| warm 36 | **2.01 s** | **1.94 s** | 3.48 s |
| 主 GUI warm | 2.02 s | 1.98 s | 2.95 s |

正常轮已经从 4.55 s 降到 **约 2 s**；平均值被 cold 的 31–38 s 拉高。

### 7.3 新的头号瓶颈：delta_limit 整段重建

主 GUI 会话在 turn 30 / 35 / 40 各触发一次：

```text
叠转到顶: rot_rounds=4, delta_limit=4
⇒ 本轮整段重建（全量 prefill）
```

三次 cold：

```text
prompt 43,180 → TTFT 31.2 s
prompt 46,092 → TTFT 33.5 s
prompt 51,564 → TTFT 38.2 s
```

图片删除本身只要约 1.5 ms K 重转；真正贵的是每 4 次叠转后的
**整段 43–52K prefill**。

`delta_limit` 是 FP8 精度硬上限（1..=4），不能靠调大绕过。下一轮优先级：

1. **避免/隐藏 rebase**：能否在 idle 预热、异步重建、或把重建放到前端不可见时点；
2. **降低单次重建成本**：chunk/prefill 已到 2048；继续压要换重建策略；
3. **减少触发频率**：图片窗口更大/更少 prune，或只在必要时删图；
4. 若业务不能接受 30s 尖峰，考虑 f16 KV 的 image_evict 档（delta_limit 32）
   作为专用 GUI 档，不与 262k FP8 生产档混用。

### 7.4 当前配置

```text
prefill_chunk = 2048（3080 TP2）
snap_msg_min_gap = 2048
image_evict_delta_limit = 4
```

现场文件：

```text
perf-runs/ttft-prefill-20261025/owl-8135-ttft-2048-41.log
perf-runs/ttft-prefill-20261025/owl-agent-2048-41.json
perf-runs/ttft-prefill-20261025/turns-2048-41.csv
perf-runs/ttft-prefill-20261025/cache-summary-2048-41.txt
```
