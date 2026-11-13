# 2026-11-13 turn24 准入拒绝与块表回收结案

## 现象

最终 8135 日志（`/tmp/owl-8135-final13.log`）：

```text
[INFO] turn 23: decode=134100ms/9870tok=73.6t/s; cache=58592/61873
[WARN] 生成预算夹紧：max_new 32768 → 0
[WARN] 生成预算被夹紧:请求 32768 → 实际 1
[INFO] 受理 turn 24(...prompt=71384;cached=0;max_new=1)
[WARN] 准入拒绝：Strict turn(零耐受)显存不足:需 2231 页(...欠 222216192),
       可驱逐面仅 2244608
```

服务没有卡死；turn23 正常收口，turn24 被准入拒绝。turn23 的 134s
是 9870 token @ 73.6 t/s，不是 GPU hang。

## 根因一：跨 turn 保表

`f9852c26` 曾把 `prefix_cache && cached_len > 0` 的会话 `block_table`
跨 turn 保留。两个会话合计持有 3.56 GiB KV，store 不能下沉/逐出这些块；
turn23 的生成预算被压到 9871，turn24 的 strict 准入差 212 MiB 被拒。

**该机制已移除**：`release_turn_table` 每个 turn 末释放全部表，
`cached_len = 0`，清空 `block_hashes`/`chain`。所有块回到 store 缓存面，
由引用计数 + LRU/过期策略回收；下一轮按内容重新命中，命中不了就重建。

## 根因二：已驻留前缀双算

旧准入把 `session_held` 与全量 `floor(ctx)` 相加；缓存命中时同一段
prompt 被算两遍。turn 5 现场：

```text
prompt 133597, cached 131904
session_held = 131904 tok ≈ 4122 页
floor(ctx)   = 133597 tok ≈ 4175 页
4122 + 4175 = 8297 > 8192  ⇒ 欠 58,920,960 B ⇒ 拒
```

新增只有 1,693 tok（53 页）。修复：

- `TurnFacts.cached_tokens`；`need_pages`/`cap_for` 只算
  `prompt − cached + max_new`；
- `admission_gate` 的 `rel.check` 传 `new_ctx = ctx − cached`；
- `cached_hit` 再按实际 `block_table` 长度封顶（表已释放时归零）。

## 块回收规则

- turn 末无例外释放 `block_table`；空闲会话**没有块表**可被钉住；
- 不通过关会话来腾 KV：请求进来由 store 按当前余量出淘汰计划；
- KV 块按会话/深度/LRU 固定顺序淘汰，回 store 缓存面；
- GDN 快照按自己的档位顺序淘汰：`Transient → Extra → 老 Binary → Dense`；
  Binary 节点 + Dense 尾锚点保持稀疏、覆盖整段对话，回退到较前位置也能
  复用已拍快照，避免整段重算；
- 在役 turn 的快照也在同一条优先级队列里：最近 K 个 turn 尾为 `Dense`，
  只是最后淘汰，不是硬保护；没有 `snap_guard` 之类的点名豁免；
- 宿主梯满时 `plan_snap_offload` 也必须走同一份快照出序
  `order_snap_victims`（`SnapClass` 档位 + 同档 `EvictPolicy`），不能直接
  `EvictPolicy::order`；否则会先丢老 `Dense`、留下新 `Extra`（2026-11-13
  已修，门 `host_eviction_respects_snap_class_before_lru`）；
- 只有在役 turn 正在使用的 KV 块由当前 turn 持有，这是使用中的所有权，
  不是永久保护。

## 验证

- `cargo check -p owl-engine --lib`：通过；
- `cargo test -p owl-engine --lib`：**238/238**；
- 真机 final24（PID 603460）：
  - 一次带 session 的请求完成后 `/debug/vram`：
    `cached_len=0`、`session_held_bytes=0`、KV 块在 cache 面；
    会话投影不再有 `positioned` 字段；
  - 长会话准入修复（final19 历史验证）：135661 prompt / 134400 cached
    时准入 `40 页`；138061 prompt / 135648 cached、`max_new=512` 时
    无生成预算夹紧。
