# 交接：KVMem 262K / CPU 调度层性能优化（2026-10-23）

> 工作树：`packages/owl-worktrees/metrics-optimize`
> 分支：`feat/metrics-optimize`
> 状态：工作树干净；GPU 已归零
> 下一刀：**Mean-K 索引采集 GPU 化**（262K prefill 的 87.6s 主项）

---

## 1. 一句话现状

GGUF KVMem 已在 24GB 3090 Ti 上做到：

```text
80K execution view（FP8 KV）
设备池 85.5K < 262K 上下文
90K coherent needle 命中（含 host 下沉）
262K prefill 跑通：355.5s / 零 IMA
```

但 **262K prefill 的 CPU 调度层仍是主瓶颈**：

```text
index=87.6s + qface=31.5s + reconcile=30.8s
score=2.8s + select=4.5s + view=1.4s
```

其中 `qface` 已经从 124.3s 降到 31.5s（廉价边界跳 Q）；
**下一刀必须是 Mean-K 索引采集 GPU 化**，否则 262K prefill 下不来。

---

## 2. 本轮提交（从旧到新）

```text
c81f688 feat(kvmem): DFlash2 视图化 verify 与 80k 选择轨迹修复
024a068 feat(kvmem): verify 视图带宽上限实验 + 8192 定档
5e89dab feat(kvmem): 移植 GQA max 聚合（上游产品默认）
7f92020 test(gguf): baseline + KVMem smoke（60k 稀疏视图退化待查）
ac81502 feat(kvmem): 最终 query replay 最小闭环（负结果）
c124090 test(gguf): FP8 KV 论文量级 execution view（64K/80K）
ad93e47 test(gguf): 80K view + 85K 池的完整虚拟化组合
3e7fd99 perf(kvmem): 廉价边界跳过 Q 面 + 262K CPU 相位实测
```

核心文档：

```text
docs/devlog/perf/kvmem-262k-cpu-20261023.md       # 262K 相位账（本交接的主要依据）
docs/devlog/memory/kvmem-query-replay-20261023.md # query replay 负结果
docs/devlog/models/gguf-kvmem-80k-20261023.md     # 80K view + host 下沉
docs/devlog/memory/kvmem-dflash2-80k-handoff-20261023.md # 80k 选择轨迹历史
docs/devlog/memory/kvmem-selection.md             # B 轴设计/施工全卷
```

---

## 3. 关键概念：80K view 是什么

**不是 80K 模型上下文，不是 80K prompt，不是 80K 显存。**

它是每个 decode step 模型实际可见的 **KV token 预算**（KVMem 论文的 `Ba` / execution view）：

```text
逻辑上下文 262K
物理池     85.5K（FP8）
execution view 80K = 2560 页 × 32 token
  = 选中的历史块 + 尾部
```

- `workspace_per_req` 是 view 预算；
- `pool` 是物理 KV 容量；
- `select_view_step` 把选中页 + 尾部写成 `sel_pt`，decode 只读这些页；
- 未选中的块即使仍在池/宿主，也不在页表里。

262K 实测：view 2560 页 / 81920 token，块 902（code@~28.9K）**在选中集**，
但输出 `**`，未答出 `BLUEBIRD-42`。⇒ 80K view 对 90K 够，对 262K 不够；
下一步质量要更大 view 或 compact re-RoPE，不能只靠 query replay。

---

## 4. 已落地的机制与实测结论

### 4.1 视图化 verify（c81f688 / 024a068）

- `exec/spec.rs::verify_view_tables`：同一页表 + 逐行 `kv_lens`；
- 90K/131K/260K 零 IMA；视图 verify 比 dense 快 ~9×；
- `verify_view_tokens` 是**实验档**：只裁 verify，decode 不动；
  16K decode + 8K verify 实测 AL 1.90 但输出退化 ⇒ 生产保持 0。

### 4.2 GQA 聚合（5e89dab）

```toml
[kvmem]
gqa_aggregation = "max"   # mean | max
```

- `ScoreMode::CosineMax`：组内逐 query head 余弦取最大，再对 (层, KV 头) 取均；
- 90K 同 prompt：mean AL=1.30 且输出退化；max 连贯且未触发 spec-degrade；
- ⚠️ 这是 **ActKV/上游产品**口径；KVMem 论文 Eq.10 是 mean/softmax。
  论文保真档应设 `gqa_aggregation="mean"`，并考虑 `SoftmaxMean`。

### 4.3 Query replay（ac81502，负结果）

已实现：

```text
末 chunk 前拍 GDN replay 快照
→ 最终 recall/视图装配
→ 恢复快照，在新视图上重跑末 chunk
→ decode
```

限制：`spec.depth=0` + 单卡；`spec>0`/TP 响亮拒。
实测：

- 80K view + 85K 池、90K 上下文：query_replay on/off 输出**逐字相同**
  `: BLUEBIRD-42`；
- 60K/90K 的球都在 **execution view 预算**，不是 replay；
- 保留代码作为 step-level 基础件，但**不要当 262K 的修复**。

### 4.4 262K CPU 相位（3e7fd99）

```text
profile: qwen38_27b-3090-gguf-kvmem-262k.toml
Q4_K_M / 3090 Ti / FP8 KV / 80K view / pool 85568 / recall_every=4
prompt 260018 tok / WALL 355.5s
boundaries=507 full=127 cheap=380
index=87573.5 qface=31527.1 score=2831.9 select=4460.9
view=1386.3 reconcile=30837.4 | total=41468.6ms
```

- 廉价边界（380/507）现在**不采 Q**：qface 124.3s → 31.5s；
- 但墙钟没有相应下降：`index` 与 `reconcile` 成为新主项；
- `recall_every=4` 是当前折中；`recall_every=16` 的尾部会让池下限
  变成 `2560 + 16×16 = 2816 页 > 2674`，**boot 已改成响亮拒**，不要再用
  旧 r16 结果。

### 4.5 质量矩阵（GGUF coherent needle）

| ctx | view | pool | 结果 |
|---:|---:|---:|---|
| 60K | 64K | 100K | `BLUEBIRD-42` |
| 90K | 64K | 100K | `:`（丢针） |
| 90K | 80K | 100K | `: BLUEBIRD-42` |
| 90K | 80K | 85.5K（host 下沉） | `: BLUEBIRD-42` |
| 262K | 80K | 85.5K | `**`（块 902 选中但答不出） |

---

## 5. 现有 profiles

```text
profiles/qwen38_27b-3090-gguf.toml
  GGUF baseline（32K，无 KVMem）

profiles/qwen38_27b-3090-gguf-kvmem.toml
  GGUF+KVMem（8K view，f16，query_replay=true，snap_max=0）

profiles/qwen38_27b-3090-gguf-kvmem-64k.toml
  FP8 KV，pool ~100K，view 65536

profiles/qwen38_27b-3090-gguf-kvmem-80k.toml
  FP8 KV，view 81920，reserve 1500 ⇒ pool 85568（host 下沉）

profiles/qwen38_27b-3090-gguf-kvmem-262k.toml
  max_seq=262144，view 81920，host 11264，recall_every=4

profiles/qwen38_27b-3090-w4a16-dflash-kvmem.toml
  AWQ 27B + DFlash2 d7 + KVMem（8K view）
```

启动（注意 GPU 锁与其他 agent）：

```bash
GDN_TRITON_PY=/home/div/Documents/codes/packages/vllm-test/.venv/bin/python \
  cargo build --release -p server --features perf,cupti

./target/release/server -p profiles/qwen38_27b-3090-gguf-kvmem-80k.toml \
  --set 'server.bind="127.0.0.1:8145"' \
  --set probes.dflash_probe=true \
  --log-level info
```

- `--set` 数组/字符串要整体加引号：`--set 'parallel.device=["42:00.0"]'`；
- **不要 `pkill -f`**；按 PID kill；
- GPU1 = `41:00.0`，GPU2 = `42:00.0`；其他 agent 的 GPU lock 不要杀；
- 探针：`probes.dflash_probe=true` 打 `[kvmem-prof]` / `view-diag`；
- `probes.view_hash_block=<id>` 验证 stage-in 页哈希。

---

## 6. 下一刀：Mean-K 索引采集 GPU 化（P0）

### 6.1 现状为什么慢

`crates/engine/src/exec/spec.rs::collect_index_for_blocks`：

```text
每个新块 × 16 层 K 页 D2H（约 1MB/块）
→ host 16 线程 decode
→ 逐 token 逆 RoPE
→ MeanKCollector 累加
→ f16 索引登记
```

262K ≈ 8GB K 回读；实测 `index=87.6s`。这是当前 prefill 的第一主项。

### 6.2 目标

只回读 **Mean-K 索引**：

```text
每块 16 层 × 4 KV 头 × 256 dim × 2B = 32KB
262K 8192 块 ≈ 256MB
```

把 87.6s 压到几秒。

### 6.3 推荐实现

**新增 GPU 块级 Mean-K 归约**（不改 host 打分）：

```text
输入：
  - 主池该块 16 层 K 页（post-RoPE）
  - 该块 32 个位置的 cos/sin（设备表或 host 预取）
输出：
  - [16 layer][4 hkv][256] f16 Mean-K
```

两种接法：

| 接法 | 做法 | 评价 |
|---|---|---|
| **A. 池 K + 逆 RoPE 核** | 新核读主池 K，按 cos/sin 逆旋转，32 token 归约 | 不改 hot insert；推荐先做 |
| B. 融合 `qknorm_insert` | 在写 K 时累加 pre-RoPE normalized K | 零回读，但要改融合核 + 边界归约 |

P0 先走 A：

1. 新增 `owl_mean_k_block`（或同族）CUDA 核 + Rust binding；
2. `collect_index_for_blocks` 改成：
   - 对 `todo` 块启动核；
   - D2H 仅 `[blocks][layers][hkv][hd] f16`（32KB/块）；
   - 按 `MeanKEntry` 登记（host 逆 RoPE 删除）；
3. host 旧路径保留作 golden/回退；
4. 单测/对拍：同块 host collector vs GPU 索引（逐元素/排序门）；
5. 262K 验收：`index < 5s`，prefill 墙钟目标 **< 250s**，零 IMA；
6. 质量：90K/262K coherent needle 与当前一致（view 不变）。

### 6.4 P1：索引按召回节奏批量

- 廉价边界不 score ⇒ 索引不必每 512 tok 同步；
- 只在完整召回前补齐，或每 `recall_every` 个 chunk 批量 D2H；
- 先做 P0（GPU 化）再决定是否还需要 P1；P0 后索引可能已不是瓶颈。

### 6.5 P2：reconcile 异步化

当前 `reconcile=30.8s`，同步等待。按论文：

```text
chunk n 完成 → 异步 stage-out，与 chunk n+1 重叠
下一召回直接回收
packed/pipelined raw-K 搬运
```

先做 P0/P1，再评估 P2 的实际收益；如果 262K 仍高，再做。

### 6.6 P3：step-level + compact re-RoPE

- 真正 step-level：prefill 中间冻结工作集，末尾一次完整召回；
- 尾部预算 = `workspace + recall_every × chunk`，已进 boot 下限；
- compact position + raw K + bounded delta re-RoPE 是 KVMem §4.4 的缺口；
- 262K 质量（80K view 选中块但答不出）最终要靠更大 view 或 compact
  re-RoPE，不是 query replay。

---

## 7. 验收门（下一刀）

```text
[ ] 262K prompt 260K：
    - index < 5s
    - prefill wall < 250s（当前 355.5s）
    - qface 不再每廉价边界采（保持）
    - zero IMA / Xid
[ ] 90K coherent needle：仍答 BLUEBIRD-42
[ ] 262K coherent needle：block 902 选中且（若 view 不变）输出不退化；
    若仍答不出，记录为 view/compact 缺口，不归因 index
[ ] host/index 对拍：GPU Mean-K vs host MeanKCollector
[ ] cargo test -p owl-engine --lib / owl-shared --lib 全绿
```

---

## 8. 未决/风险

1. **262K 质量未解**：80K view 选中 block 902 但输出 `**`；可能是 view
   覆盖不足或模型长距离问题，下一步在 P0 后单独定位。
2. **query replay 负结果**：目前 on/off 无差异；保留为基础件，不宣传收益。
3. **GQA max vs 论文 mean/softmax**：当前默认 max 是上游/ActKV 口径；
   论文保真档应 A/B `mean` + `SoftmaxMean`。
4. **verify_view_tokens**：实验档，实测会改变草稿接受/输出，生产 0。
5. **TP/spec query replay**：当前响亮拒；草稿环/双 rank replay 另立刀。
6. **ActKV LRFU**：owl 的 `hit` = recall，不是 action-region attention；
   只影响 host 下沉，不影响 top-k；默认 LRU。
7. **`recall_every` 尾部**：已修 boot 下限；不要再跑 k16 + 80K view 的
   小池旧档。

---

## 9. 复现资料

```text
perf-runs/metrics-optimize/gguf-kvmem-262k-k4-qskip2/
  server.log
  client262k.txt
  out262k.txt

perf-runs/metrics-optimize/gguf-kvmem-90k-80k-p85/
  server.log
  client90k.txt
  out90k.txt

perf-runs/metrics-optimize/gguf-kvmem-64k/
  server.log
  client60k.txt
```

固定 coherent needle 生成器：`/tmp/needle_coherent10_64.py`
（临时文件；长期应移到 `perf-runs/` 或 `packages/llm_speedtest`）。
