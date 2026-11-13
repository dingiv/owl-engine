# 相邻截图只删一张 ⇒ 前缀缓存塌到图片前（2026-11-13）

> 真机：8135 / 3080 TP2 / Qwen3.8-27B AWQ-INT4 / INT8 KV / image_evict。
> 关联：`2026-11-12-session-aware-kv-snap-设计.md`、
> `2026-11-13-image-append-ttft.md`、`2026-11-13-ttft-prefill-handoff.md`。

## 1. 现象

agent 会话从 30k 增长到 ~100k，一直 99% 命中；截图替换后突然掉到 26k 前缀：

| turn | prompt | cached | 说明 |
|---:|---:|---:|---|
| 52 | 98 617 | 98 560 (99.9%) | 正常 |
| 53 | 101 182 | 98 560 (97.4%) | 仍可 |
| 54 | 102 502 | 98 592 (**96.2%**) | 开始丢 `session-local` suffix |
| 55 | 102 956 | **26 336 (25.6%)** | 新截图替换旧图；客户端随后断流 |

对应日志（修复前）：

```text
[img-evict] session-local 未用: ... turn=55 chain=3251 bh=3251 table=0 common=26336 blocks=823 tree_m=759
[kvmem] 复用面:matched=759 m_raw=0 ...
```

客户端请求形态（同一 tool result 内**两张相邻截图**）：

```text
turn 54: [text] [image A 389718B] [image B 108138B] [text] ... [image C]
turn 55: [text] [image B 108138B] [text] ... [image C] [image D 新截图]
```

即：**只删掉相邻图像组的第一张 A，保留第二张 B，再在末尾追加新图 D**。

## 2. 根因 A：相邻图像被合并，半删识别不了

`detect_text_delete` 把文本坐标里首尾相接的已标记图像合并成一个游程：

```rust
Some(last) if last.1 == s.text_start => { last.1 = s.text_end; last.3 = s.end; }
```

于是 A/B 被当成一个候选 `[A.start, B.end)`。客户端只删 A 后：

- `old_text[B.end..]`（B 之后的文本）与 `raw[A.start..]`（从 B 开始）逐 token 对不上；
- `k == 0` ⇒ 不识别删除；
- 回退 `common` 前缀停在 A 之前 = 26 336 tok，整个 B 后缀白重算。

旧实现没有测试“相邻图只删第一张”的形态；已有测试都是图之间有文本。

## 3. 根因 B：positioned 链的 `block_table` 每轮被 release

`session-local hit` 依赖 `AgentSession.block_table`，但 turn 末
`release_turn_table` 会把整张表 decref 并 clear。`publish_turn_chain`
又对 **positioned** 链直接跳过（位置基准不在身份里），共享树只能命中
图片前的前缀：

```text
turn 54: session-local 未用 ... table=0 ... tree_m=3093
         ⇒ cached 102080 被校正成 98592
```

### 3.1 最新一次复发：全量重建后 `local_chain` 被清，但 `positioned` 还在

turn 24→25 客户端把旧图组整体换成新图组；旧图不连续，删除识别不命中：
turn 25 全量重算 127k。`plan_fresh` 调 `rebase()` 清了 `local_chain`，但
submit 时算出的 `turn.positions_gap` 仍是旧值，turn 收口又把会话写回
`positioned=true`。于是：

- `publish_turn_chain` 因 `positioned` 跳过 ⇒ 新链不进共享树；
- `release_turn_table` 只认 `local_chain` ⇒ 表被释放；
- 下一轮（turn 26）`table=0`、`tree_m=759` ⇒ **0 命中**。

修法：

1. `release_turn_table` 对 `positioned`（不只是 `local_chain`）保留
   `block_table`；`local_chain` 只保留作 gapped 诊断标记。
2. **全量重建（`cached_len == 0`）后强制位置压实**：旧 K 一点没复用，
   旧位置洞已无对象；重算紧凑 positions，`positions_gap=false`，turn 收口
   `positioned=false` ⇒ `publish_turn_chain` 正常发布，下一轮走共享树。

## 4. 修法

### 4.1 带内容指纹的删除识别

`crates/engine/src/vision/img_evict.rs`：

- 新增 `detect_text_delete_tagged(old, raw, spans, new_tags)`；
- 对每个相邻已标记游程，枚举**所有连续子段** `[i, j)`，而不是只试整段；
- 用图像内容指纹消歧：`new_tags` 必须以
  `old_tags[..i] ++ old_tags[j..]` 为前缀（尾部允许追加新图）；
  token 占位符相同但内容不同的相邻图不会被错绑；
- 旧账 `img_tag == 0` 时退回旧整游程识别（宁可少吸收，不做 token 盲猜）。

`image_pad_law` 只算一次 `new_tags`，删除识别与 span 盖章共用，避免每图重复哈希。

### 4.2 positioned / gapped 链跨轮保留 block_table

`crates/engine/src/serve/session.rs` + `exec/turn_end.rs`：

- 新增 `AgentSession.local_chain`（leave-gap 后置 true，`rebase` 清 false）；
- `release_turn_table`：**positioned（含 gapped）**会话保留 `block_table` 引用，
  只释放 `tokens.len()/page` 之外的预分配尾块；普通会话仍按旧行为整表释放。
  `publish_turn_chain` 对 positioned 链跳过，表是唯一槽账。
- `reset()`（前缀命中续写路径）保留 L1 资产/`local_chain`；`rebase()`（真换链）
  才清 tail/msg keys/local_chain。
- `apply_image_leave_gap` 在**新表 incref 完成后**摘旧表并 decref；本地链
  保留后旧表不再天然为空，不释放会每轮漏引用（第二次 leave-gap 才暴露）。
- 代价：空闲 gapped 会话在 turn 间仍持 KV；由会话 LRU/显式关闭回收。
  这是共享树无法承载 gapped 链时的必要取舍。

### 4.3 非连续多图全删：多段 leave-gap

`crates/engine/src/vision/img_evict.rs` + `sched/scheduler.rs` + `running.rs`：

- `DeletionPlan` 携带多段 `TextDelete`；
- `detect_all_images_deleted_tagged`：旧图全部被删、新图只在尾部时，识别所有
  旧图段；客户端 raw 的幸存文本必须覆盖到最后一段删除之前；
- `image_pad_law` 保留旧 **padded** 幸存序列（含 orphan pad），只把客户端
  raw 的新尾巴接在共同前缀后；新图照常走 pad 律；
- `positions_after_deletes`：所有幸存段 position 原样保留，新尾巴从旧
  `max_pos+1` 起；
- `plan_image_leave_gap_multi`：从旧块表去掉所有被删图页，保留全部幸存页
  （不是只留首尾两截）；链身份从第一段删除处重算，快照逐边界重挂；
  恢复点用旧链 `tail_snap`（turn 末 Dense）并重挂到新链最后一个幸存块。

### 4.4 热前缀表：不再整条 host→device swap-in

`exec/turn_end.rs::release_turn_table`：

- 旧 S2 语义：turn 末把块表还给缓存，让 idle 时下沉宿主。
- 真机 128k 会话：turn 1 冷灌后，turn 2 命中 99.9%，但 4021 块
  / **2.1GiB** 全部在宿主层；turn 2/3 每次启动都
  `swap-in 2.1GiB / 2s`，TTFT 9.4s，客户端在 turn 3 prefill 阶段直接断流。
- 现语义：`prefix_cache && cached_len > 0` 的会话保留热表（引用不还），
  只释放预分配尾块。缓存块 refs>1 ⇒ store 不再整条下沉；下一轮只分配新增页。
- 代价：热会话在 turn 间持 KV；由会话 LRU/close/`rebase` 回收。多会话
  同驻时需要按池压力再做主动 release（当前单/双并发档可接受）。

复测（同一 125971 tok 请求连发两次，冷/热）：

```text
turn 1: cache=0/125971, ttft=68914ms
turn 2: cache=125280/125971 (99.5%), ttft=1003ms
        **无 swap-in**
```

## 5. 验证

单测：

```bash
cargo test -p owl-engine --lib partial_adjacent_image_delete_uses_content_tags
cargo test -p owl-engine --lib vision::img_evict
# 12/12 全绿（含 all_old_images_deleted_multi_splice）
```

真机回放（同一 session，turn 54 原始 102 502 tok → turn 55 替换首图）：

```text
[img-evict] 删除识别: old=102490 raw=102910 images=3 ⇒
  删 text=[24279,26321), padded=[24288,26336), 2048 tok/64 页
[img-evict] leave-gap: 保留前缀 759 + 后缀 2367 页 / 100032 tok
[kvmem] 复用面: matched=3187 m=3187 尾快照层=Some(Gpu)
turn 5: cache=100032/102956 (97.2%)
```

下一轮同一请求（无删除）：

```text
matched=3187 m_raw=3187
turn 6: cache=101984/102956 (99.1%)  ttft=1221ms
```

### 全量换图回放（多段）

在已有位置洞的会话上回放 turn 24 → 25 → 26 原始请求（输出夹到 16 tok），
turn 25 把三张旧图整体换成三张新图：

```text
turn 24: cache=0/125971（冷启动）
turn 25: 多段删除识别: 删 3 段 / 4966 tok，保留旧 padded 前缀 120981 tok
         多段 leave-gap: 保留 3780 页 / 120960 tok，新链重算 3021 页
         cache=120960/127372 (95.0%)  ttft=8399ms
turn 26: 复用面 matched=3979 m_raw=3979
         cache=127328/128443 (99.1%)  ttft=2056ms
```

注意 turn 25 的 5% 缺口主要是**新图本身的 4960 tok**，不是旧图重算。

对照修复前：turn 55 只命中 `26336`，且客户端断流。

## 6. 未覆盖

- **非连续删除但保留部分旧图**（例如删 A、保留 B、删 C）：当前多段路径
  只在“所有旧图都被删”时启用；保留旧图会涉及重排/重补 pad，仍安全回退
  到图片前的前缀。
- `local_chain` 表的空闲驻留：单会话/双并发档可接受；更多会话需要
  后续按 LRU 主动 release，或给 gapped 链增加 session-salt 的私有树。
- 旧的 `detect_text_delete` 保留给 tag=0 旧账与测试；新路径必须带 tags。
