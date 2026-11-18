# 面向 Agent 会话的 KV Cache 与 GDN 快照管理（设计 v4）

> 2026-11-18 · 本版刷新 2026-11-12 v3。
>
> **v4 裁决**
> 1. 持久 GDN 快照只有两类：**Tail**（最近尾部，连续保留）与
>    **History**（前缀历史，每次重新均匀选点）。
> 2. 删除 `Dense / Binary / Extra` 三种结构档；`Transient` 只保留为
>    生成段临时生命周期，不是持久快照类型。
> 3. 每会话配额：**总快照 30 张，尾部 12 张**，其余 18 张为均匀历史。
> 4. 历史快照不靠“档位 + LRU”保持覆盖；每次根据当前完整消息边界
>    重新计算均匀子集，删掉不在新子集里的历史点。
> 5. `snap_session_cap` = Tail + History 的逻辑总上限；
>    `snap_max` 仍只是设备热槽数，历史点可落宿主/盘。
>
> v3 的 `Dense/Binary/Extra` 与 `snap_structural_reserve` 设计已被本版取代；
> 旧实现只作为迁移对照，不再作为目标语义。

---

## 0. 一句话

- **KV**：块级内容寻址 + 前缀缓存，独立命中、独立驱逐。
- **GDN 快照**：`Tail`（最后 12 张，连续）+ `History`（前缀 18 张，均匀）。
- **召回**：KV 与快照分别计算，组合时取 `start = 最近 ≤ kv_hit 的快照`；
  一方缺失不清零另一方资产。
- **删除历史快照**：重新算 18 点均匀子集，删除不在子集内的旧点；Tail 最后才动。

---

## 1. 前端事实（输入）

1. 组织单位 = session + messages[]；客户端每轮重发完整历史，新内容通常追加
   在消息末尾。
2. 最近若干条消息可能回撤/重生成；回撤点是消息边界。
3. TTFT 敏感：append 时理想成本 = 只 prefill 新消息。
4. 客户端可能重新序列化 assistant 消息，因此消息边界是可靠锚点；
   生成 token cursor 只作附加锚点。
5. 客户端会主动取消请求；取消时已经 prefill/生成的页边界也需要快照。

---

## 2. 目标与旧结构问题

### 2.1 目标

- 每次“用户请求 / 工具结果 / 正常结束 / 主动断开”都留下可恢复边界。
- 最近 12 张快照连续保留，保证 append、浅回撤、工具结果续写的 TTFT。
- 更早历史在整条消息链上**按 token 距离均匀覆盖**，避免深回撤时只剩很老的
  稀疏锚点。
- 历史密度由总配额决定，不靠人为配置 `min_gap`/结构档。

### 2.2 旧结构的问题

- `Dense/Binary/Extra` 把“位置覆盖”与“重要性档位”混在一起；
- 二分点随 L 移动，需要反复重分类/降级；
- `snap_structural_reserve` 只是给结构档留名额的补丁；
- 删除历史点时按档位/LRU 容易留下局部空洞，覆盖不均匀；
- `snap_msg_min_gap` 让“最近短消息”和“中段长消息”得到不同密度。

v4 不再使用这套分类。

---

## 3. 快照模型

### 3.1 两类持久快照

```text
Tail    = 最近 snap_tail 张快照，连续保留
History = Tail 之前的快照中，均匀选出的
          snap_session_cap - snap_tail 张
```

默认：

```toml
[store.gdn]
snap_session_cap = 30
snap_tail = 12
```

于是：

```text
Tail    = 12
History = 18
```

`Transient`（生成段 rollback）继续存在，但不计入持久快照配额，也不参与
Tail/History 均匀布局。

### 3.2 快照点的来源

快照点仍然要求**页对齐**（key 粒度 = 页，32 token）。候选来源：

- 消息开启标记前一条消息的结束点（现有 `message_snap_points`）；
- 每条完整请求的 prompt 最后一个页边界；
- 生成结束/中断时的最后一个已生成页边界；
- 可删图锚点等硬边界（仍可拍，但进入统一 Tail/History 布局）。

### 3.3 配额与物理层

| 资源 | 语义 |
|---|---|
| `snap_session_cap=30` | 单会话逻辑快照总数，Tail+History |
| `snap_tail=12` | 最近 12 张逻辑快照，连续保留 |
| `snap_max` | 设备热槽数（当前 TP2 档 4）；不决定逻辑总数 |
| `snap_host_mb` / `snap_disk_mb` | 历史点可下沉的宿主/盘容量 |

如果 `snap_session_cap=0`：保持旧行为（不启用每会话上限），不压缩历史。
如果 `snap_tail > snap_session_cap`：Tail 只保留最后 `snap_session_cap` 张，
History 为空。

### 3.4 历史选点算法

输入：

```text
C = Tail 之前的所有页对齐候选边界，升序
L = C 的最后一个位置（或 Tail 起点之前）
M = snap_session_cap - tail_count
```

选择 `M` 个点：

```text
if M <= 0: []
if len(C) <= M: return C

for j in 0..M:
    target = L * j / (M - 1)
    在 C 中选择离 target 最近、且尚未选过的点
```

约束：

- 首、尾都要有锚点；
- 按 token 距离均匀，而不是按消息条数均匀；
- 消息长短差异大时不会全部挤在短消息区；
- 同页候选先由上游去重。

复杂度 `O(M·len(C))` 可接受；消息数大时可改为二分查找最近点。

### 3.5 删除/缩容

不做“从历史里随便删一个”。每次重新计算目标集合：

```text
当前 History = H
新目标 = M - k
new_H = uniform_select(C, M - k)
删除 H \ new_H
```

效果：

- 删除后剩下的历史点仍近似等距；
- 不产生局部空洞；
- 需要删 1 个时也走同一逻辑，不引入第二套贪心规则。

内存压力逐出顺序：

1. 先删 History（按“删除后均匀性损失最小”的顺序；等价于重算更小的 M）；
2. History 空了才允许动 Tail；
3. Tail 从最老的开始，最后才丢最新尾锚点。

Tail 不是永久 pin；它是最后淘汰层。

### 3.6 拍快照的时机

| 事件 | 拍点 | 归类 |
|---|---|---|
| 用户主动发完整请求 | prompt prefill 完成、prompt 末页边界 | Tail |
| 工具调用结果返回 | 新请求 prompt prefill 完成 | Tail |
| 正常 turn 结束（EOS / max_tokens / tool_calls） | 生成结束的最后页边界 | Tail |
| 用户 abort（prefill 中） | 最后一个已算页边界 | Tail |
| 用户 abort（generation 中） | 最后一个已生成页边界 | Tail |
| 历史消息边界 | 每次布局时从当前消息链重算 | History |

说明：

- 工具结果没有独立引擎分支；它就是一次新请求，走 prompt 末快照；
- 请求全命中、状态已恢复时，同内容身份做 `Touch`，不重复拷贝；
- 任何 turn 正常停止都必须拍最后一张，不能只在长生成时拍；
- 当前 `capture_abort_snap` 只覆盖 prefill 中段，v4 必须扩到 generation。

### 3.7 布局重算时机

在这些事件后重算 Tail/History：

- submit / prompt prefill 完成；
- 正常 turn 收口；
- abort 收口；
- 历史消息 append / 回撤导致消息边界变化；
- 会话恢复或显式清理时。

重算只改变“哪些历史点属于目标集合”，不复制内容；已有同键快照保留，
缺失的历史点按需补拍，不在目标集合的历史点删除/下沉。

### 3.8 与 KV 前缀缓存的关系

- KV 块链仍然内容寻址、独立 LRU/频率；
- 快照布局不裁剪 KV 命中面；KV 命中面也不因快照缺失被清零；
- 组合规则：

```text
start = 最近一个 ≤ kv_hit 的快照边界
若不存在 ⇒ start = 0（全量重算）
```

- 快照晚于 `kv_hit` 时不能反向使用；但快照资产保留，等 KV 回填/重算到该边界；
- 没有快照时才全量重算，这是失败态，不是默认路径。

---

## 4. 代码结构

### 4.1 存储层

- `SnapClass` → `SnapRole`（或保留 `SnapClass` 名称但只留）：

```rust
enum SnapRole { Tail, History }
```

- `Transient` 不再作为持久 role；由 `SnapDurability::Transient` 表达；
- `SnapAdmitPolicy` 删除 `structural_reserve`，只保留 `session_cap`；
- `admit_snap`：
  - 同键 Touch；
  - `Transient` / 无会话 / cap=0 ⇒ Add；
  - 有空间 ⇒ Add；
  - 满了 ⇒ History 先淘汰；History 空才淘汰 Tail；
- `order_snap_victims`：History 先、Tail 后，不再按 Dense/Binary/Extra 排序。

### 4.2 引擎层

- `serve/msg_boundary.rs`：
  - 保留候选扫描；
  - 新增 `uniform_select(points, m)`；
  - 新增 `plan_snap_layout(candidates, prompt_len, tail_count, cap)`；
- `running.rs`：
  - `msg_snaps` + `msg_snap_classes` 改为一次 layout 计算；
  - 新请求 prompt 结束时调用；
- `exec/phases.rs` / `exec/spec.rs`：
  - 末边界 = Tail；
  - 历史候选 = History；
  - 生成段 rollback = Transient；
- `exec/turn_end.rs`：
  - 正常收口拍 Tail；
  - abort（prefill/generation）拍 Tail；
  - 重算布局，不整段删；
- `serve/turn.rs` / `serve/session.rs`：
  - 字段从 class 改为 role；
  - 会话关闭时释放归属，内容仍可被其他会话复用。

### 4.3 配置

v4 目标：

```toml
[store.gdn]
snap_session_cap = 30
snap_tail = 12
```

退休：

- `snap_structural_reserve`
- `snap_msg_recent`（改由 `snap_tail` 表达快照张数）
- `snap_msg_min_gap`（历史密度由均匀选点决定）

保留：

- `snap_max`：设备热槽
- `snap_host_mb` / `snap_disk_mb`：历史点下沉容量

---

## 5. 验收

### 5.1 单元/纯函数

- `uniform_select`：
  - 输出数量 = min(m, len(C))；
  - 升序、无重复、首尾覆盖；
  - 删除后最大间隔不超过理论均值的常数倍；
  - m=0/1/len 边界正确。
- `plan_snap_layout`：
  - Tail = 最后 12；
  - History = 18 均匀；
  - Tail 与 History 不重叠；
  - cap=0 旧行为；
  - cap<tail 的夹紧。
- 历史删除：删除点后重算，剩余点仍均匀；不产生局部空洞。

### 5.2 集成门

1. 用户请求 → prompt 末 Tail 快照；同一请求全命中时 Touch 不重复。
2. 工具结果新请求 → 新 prompt 末 Tail 快照。
3. 正常 turn 结束 → 生成末尾 Tail 快照。
4. abort（prefill 中）→ 最后已算页边界。
5. abort（generation 中）→ 最后已生成页边界。
6. 超过 30 张 → History 被均匀压缩到 18，Tail 12 保留。
7. 深回撤 → 最近历史点覆盖均匀，不出现“只能退到很老锚点”。
8. 前缀缓存/Prefix reuse 的 cached_tokens 不回退。
9. 3090/3080 TP2 真机：append、工具结果、abort、长会话各一轮。

### 5.3 指标

- `snap.tail.count` / `snap.history.count`
- `snap.history.target` / `snap.history.evicted`
- `snap.history.recomputed`
- `snap.capture.request` / `tool_result` / `complete` / `abort_prefill` / `abort_generation`
- `snap.layout.max_gap` / `snap.layout.gap_cv`（均匀性）
- `snap.touch` / `snap.skip` / `snap.evict`

---

## 6. 迁移阶段

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | 纯函数 `uniform_select` + `plan_snap_layout` + 单测 | 算法门 |
| P1 | `SnapRole`/store 准入/淘汰改为 Tail/History | store 单测 |
| P2 | engine 布局重算 + 请求/工具结果/完成/abort 拍点 | 集成门 1–6 |
| P3 | 配置迁移：30/12，删除 reserve/min_gap/recent | `--check` 全档 |
| P4 | 真机 3090/3080 TP2 append/abort/长会话 | 性能/命中率 |
| P5 | 文档/指标/清理 | 交接 |

### 6.1 风险

- Tail 太大而设备槽 `snap_max` 太小 ⇒ 12 张 Tail 多数在宿主/盘，回灌有延迟；
- 30 张逻辑快照的宿主预算需核算（当前 3GB 级）；
- `Transient` 与持久布局混用时要确保 rollback 不被历史删除吃掉；
- 旧 `msg_snap_keys` 跨轮沿用逻辑要随 role 重写，不能保留旧 class 假设。

---

## 7. 与 v3 的差异

| 版本 | 快照组织 | 删除 | 配额 |
|---|---|---|---|
| v1 | 所有消息边界同密度 | 整段 drop | 无 |
| v2 | stride + 两套管理器 | 降级/不删 | 自定义 |
| v3 | Dense/Binary/Extra 二分 + 尾部 | 档位 + LRU | cap + reserve |
| **v4** | **Tail + History 均匀** | **重算均匀子集** | **30 / 12** |

---

## 8. 状态

- 本文件 = v4 设计基线；v3 施工状态仅作历史参考。
- 代码尚未按 v4 改造；下一会话按 `roadmap.local/26-11-18-snapshot-tail-history-交接.md`
  执行。
