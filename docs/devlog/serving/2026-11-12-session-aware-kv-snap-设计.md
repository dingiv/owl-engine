# 面向 Agent 会话的 KV Cache 与 GDN 快照管理（设计 v3）

> 2026-11-12 · worktree `packages/owl-worktrees/session-kv-snap` · 分支 `feat/session-kv-snap`
>
> **v3 修正**
> 1. 快照稀疏采用**二分**：以当前上下文长度 L 为基准，保留 `L/2, L/4, L/8, …`
>    附近的**消息边界**（不是每 4/16/64 条消息的 stride）。目标点 X 找不到时，
>    退到最近的二分点，**重算量 ≤ X/2**——"先复用一半"。
> 2. KV 召回与快照召回**分开**：各自算各自的命中面、各自驱逐、各自预算；
>    一方缺失**不把另一方清零**，只在最后组合时取 `min(kv_hit, snap_hit)`，
>    已有资产继续保留（可被后续/其他会话复用）。

---

## 0. 一句话

- **两套管理器**：KV 前缀缓存（块级、内容寻址、LRU+频率）与 GDN 快照
  （消息边界级、二分稀疏 + 尾部稠密 + LRU/频率）独立运作。
- **两次独立召回**：`kv_recall()` 与 `snap_recall()` 分别计算；组合时
  `reuse = min(kv_hit, snap_hit)`，但**不因为一方缺失就丢弃另一方**。
- **二分保证**：上下文长 L 时，快照保留在 `L/2, L/4, …` 附近；任意目标 X
  的最近快照 ≥ X/2，**最坏重算 ≤ X/2**。
- 回撤只 **降级**，不整段删除；由档位 + LRU/频率自然老化。

---

## 1. 前端事实（输入）

1. 组织单位 = **session + messages[]**；前端每轮重发完整历史，新内容只追加在
   **消息末尾**（回撤/改写除外）。
2. 最近 K 条消息可能被回撤/重生成；回撤点是**消息边界**。
3. TTFT 敏感：append 时理想成本 = 只 prefill 新消息。
4. 客户端可能重新序列化 assistant 消息，因此 **消息边界** 是可靠锚点，
   生成 token 的 cursor 只作附加锚点。

---

## 2. 现状缺口

### 2.1 追加点不保证拍

`serve/msg_boundary.rs::message_snap_points` 只有第一条边界无条件保留，
后续受 `min_gap=2048` 约束；最近几条短消息（工具结果）恰好是 TTFT 最敏感的，
被跳过。

### 2.2 尾部锚点跨轮无保护

活跃 turn 的 `last_snap` 不享有硬保护；它和所有快照一起按 `SnapClass` 优先级排队（两轮之间会话空闲，尾部快照
裸奔；回撤窗口（倒数第 2/3 条）无保护。

### 2.3 快照没有"二分"层级

现在所有消息边界同密度 + 每 4 chunk 锚点 + decode 每 32 token transient，
一起抢 6 个设备槽 / 3 GB host；没有 `L/2, L/4, …` 的二分保留，深回撤时
没有"退一半"的锚点。

### 2.4 两次召回耦合

`reuse_face` 把 KV 命中面裁到"有快照的边界"（`m`）；`m=0` 直接
`plan_fresh`。效果是：**快照缺失 → KV 命中面也被清零/丢弃**，已有的 KV 资产
白费。反过来，KV 缺失时快照也没有独立保留/复用路径。

### 2.5 现场证据（3080 TP2 / 117k 会话）

```text
Gpu  Snap: 6 entries (snap_max=6，满)
Host Snap: 16 entries, 2.55 GB / 3 GB
[img-evict] leave-gap 后缀边界无快照: keep=0 ⇒ 不吸收
[store] 钩子(Grow) ... 快照 下沉 0 溢盘 0 丢 1
→ 每轮 cached=0，prefill 45~60 s
```

---

## 3. 设计

### 3.1 KV 前缀缓存管理器（沿用 + tail 优先保留）

- 命中：`match_prefix_tagged`（token + 图像指纹），块级、内容寻址；
- 驱逐：叶级 / 整枝 / LRU + 频率（现有 `EvictPolicy`）；
- **新增 tail 优先保留**：每个会话的最近 K 条消息边界以内的 KV 块链进入优先保留窗口，保证
  append/浅回撤的 TTFT；
- 中段/老段：正常 LRU；**不因回撤强制删除**（旧分支可 undo；内容寻址可复用）；
- **不感知快照是否存在**：KV 的命中面、保留、驱逐都不以快照为条件。

### 3.2 GDN 快照管理器（二分稀疏淘汰 + 尾部稠密）

**粒度**：客户端消息边界（`<|im_start|>` 前一 token，页对齐下取整）。

**保留集合**（每会话）：

```text
dense    : 最近 K 条消息边界（K=4），每条都拍（Dense 档，最后淘汰）
binary   : 以当前上下文长 L 为基准，取消息边界中“≤ 且最接近” L/2、L/4、L/8、…
           每个二分点保留 1 张（level anchor，Binary 档）
extra    : 同一 level 内可额外保留若干张（LRU + 频率，Extra 档）
```

- **二分保证**：目标边界 X 的最近 level anchor ≤ X 且 ≥ X/2（只要各级
  level anchor 在场）⇒ 从该 anchor 起算，**重算 ≤ X/2**。
- **二分点会随 L 增长移动**：新点加入，旧点降级为 `extra`（仍是合法前缀，
  可被 undo/其他会话复用），由 LRU 老化；**不删段**。
- **level anchor 不是 pin**：它只是 `Binary(level)` 档，淘汰时排在 `Extra` 之后、
  `Dense` 之前；某一级被淘汰只意味着该级以下的回退保证退化，不会阻止淘汰。
  数量 = `log2(L / dense_window)`，200K 上下文约 5 张。
- **频率**：每个档位内按 `last_use` 主、`hits` 次淘汰。
- **transient**（decode/spec rollback）：per-turn ring 覆盖写，`Transient` 档，
  最先淘汰。

### 3.2.1 淘汰顺序（二分稀疏删除；**不是 pin**）

快照条目的档位（`SnapClass`）与淘汰顺序：

| 档位 | 含义 | 淘汰顺序 |
|---|---|---|
| `Transient` | 生成段 rollback | 1（最先，零成本） |
| `Extra` | 不在二分节点、不在 dense 尾部 | 2 |
| `Binary(level)` | 二分节点；level 越大越老 | 3（老节点先走） |
| `Dense` | 最近 K 条消息边界 / 会话尾锚点 | 4（最后，但**可淘汰**） |

- 档内按 `last_use`（LRU）+ `hits`（频率）排序；
- **没有硬保护节点**：所有快照（包括活跃 turn 的最新边界）都进同一条
  `SnapClass` 优先级队列；`Dense` 只是最后被选中，空间不足时照样被逐出；
- 当前正在载入/拷贝的单个 key 只在本次操作内不参与 victim，操作结束即回队列；
- 回撤/改写只把旧 tail 从 `Dense` 降级为普通档（或直接交给 LRU），不整段删。

### 3.3 两套召回（分开算，最后组合）

```rust
struct Recall {
    kv_hit: usize,     // KV 能匹配的最长前缀（独立）
    snap_hit: usize,   // 快照能提供的最近边界（独立）
}
```

**KV 召回**：只看块链（内容寻址 + 图像指纹），返回 `kv_hit` 与 tier；
**不**因为 `snap_hit < kv_hit` 就把 `kv_hit` 裁掉。

**快照召回**：按 `dense → binary levels → extra` 找 ≤ 目标边界的最近一张；
返回 `snap_hit` 与 tier；**不**因为 `kv_hit < snap_hit` 就把快照丢掉。

**组合**（前向起点，不是资产清零）：

```text
start = last_snapshot_at_or_below(min(kv_hit, prompt_end))
若不存在这样的快照 ⇒ start = 0
从 start 恢复 GDN；KV 用 [0, start) 的 past；重算 (start, prompt_end]
```

注意 `min(kv_hit, snap_hit)` 只是起点上界；真正的起点必须是**≤ kv_hit 的快照**
（快照在 kv_hit 之后时不能反向用）。例：

| kv_hit | 最近快照 | 前向起点 | 重算 |
|---:|---:|---:|---:|
| 200K | 200K | 200K | 0（只算新增） |
| 100K | 100K | 100K | P-100K |
| 100K | 80K | **80K** | P-80K（不是从 0） |
| 60K | 80K | 最近 ≤60K 的快照（如 40K） | P-40K；80K 快照保留待用 |
| 100K | 无 ≤100K | 0 | P（全量；KV 只保 attention past） |

- `snap_hit` 是"能恢复 GDN 状态"的边界；`kv_hit` 是"块可用"的边界。
  起点 = **≤ kv_hit 的最近快照**（快照晚于 kv_hit 不能反向用）；但**资产不丢**：
  - `kv_hit > snap_hit`：snap 之后到 kv_hit 的 KV 块保留（重算会覆写，
    但块身份仍在缓存里，其他会话/undo 可复用）；
  - `snap_hit > kv_hit`：快照保留；等 KV 回灌/重算到该边界时可直接用。
- **禁止**：`snap_hit == 0 ⇒ plan_fresh` 把 KV 也清零。没有快照时才全量重算，
  这是失败态，不是默认路径。

### 3.4 预算与分层（各自独立）

| 资源 | 设备 | 宿主 | 驱逐 |
|---|---|---|---|
| KV 块 | KV 池 | host KV | 块级 LRU/频率 + tail 优先保留（非 hard pin） |
| 快照 | `snap_max` 槽 | `snap_host_mb` | 档位（Transient→Extra→Binary→Dense）+ LRU/频率 |

- **L1 每会话上限**：`snap_session_cap`（0 = 不限；只数非 Transient，设备+宿主+盘都算）；
  `snap_structural_reserve` 给 Dense/Binary 预留 ⇒ `Extra ≤ cap − reserve`。
  超限时只替换“档位不高于新档”的最冷者；不划算就 **Skip（不拷贝）**。不是 pin。

- 设备快照槽优先级：
  1. 当前/最近会话的 dense（K 条）
  2. 二分 level anchor（当前会话优先，其他会话次之）
  3. transient ring（1/turn）
  4. extra 条目（host 优先）
- 空间不足：先把 `Extra`/老 `Binary` 下沉 host/disk；host 也满时按档位淘汰，
  `Dense` 最后（但照样可走）。
- host 预算建议 `snap 6~8G + KV 4~7G`（KV host 实测只用 ~0.6G），作为调优项，
  不是设计前提。

### 3.5 生命周期

| 事件 | KV | 快照 |
|---|---|---|
| append | 新消息 prefill；tail 段进优先保留窗口 | 新 tail 拍 1 张（Dense）；旧 tail 降级 |
| 上下文增长 | tail 窗口前移 | 重新计算 `L/2, L/4…` 二分点；新 level anchor 加入，旧点降级 extra |
| rollback | 旧分支块保留（内容寻址）；旧 tail 降级 | 旧 dense/tail 降级为普通档；level/extra 不删，按档位/LRU 老化 |
| leave-gap | 旧链尾释放引用 | 用 suffix 的 dense/level 边界（二分点天然覆盖） |
| prune/drop chain | 块级回收 | orphan 上报给快照管理器，由它按层级/LRU 回收 |
| session close | 回收该会话保留优先级/引用 | 回收该会话全部锚点（无复用可能） |
| transient | — | round 结束覆盖/释放 |

> 回撤/删除都**不整段删**；只调整档位归属，让档位/LRU/频率决定去留。

### 3.6 代码映射

| 位置 | 改动 |
|---|---|
| `serve/msg_boundary.rs` | 返回 `(boundary, msg_index)`；最近 K 条无条件；输出二分点候选 |
| `exec/phases.rs::snap_keep` | dense + 二分点才拍；非锚点不拍 durable |
| `exec/phases.rs` decode / `exec/spec.rs` spec | transient 改 per-turn ring 覆盖 |
| `exec/turn_end.rs` | tail 快照（Dense）+ tail KV 优先保留；二分点重算/level 晋升 |
| `running.rs::snap_victim_order` | 全部快照按 `SnapClass` 优先级出序；无活跃会话保护面 |
| `running.rs` KV 驱逐 | tail 优先保留；**不因快照缺失裁 KV** |
| `sched::reuse_face` | 拆成 `kv_recall` 与 `snap_recall`；组合取 min，不丢资产 |
| `sched::plan_image_leave_gap` | 用 suffix 的 dense/level 边界 |
| `state.rs` / `rel.rs` | 快照增加 `hits`、`level`；层级 + LRU 独立排序 |
| `SessionBank` | per-session anchors 元数据；会话关闭回收 |
| 配置 | `anchor_recent=4`、`anchor_binary=true`、`anchor_level_caps`、`anchor_device_sessions` |

### 3.7 指标与验收

**指标（两套分开）**

- KV：`kv.hit_tokens`、`kv.miss_tokens`、`kv.tail_evicted`（压力下可非 0）、
  `kv.orphan_reclaimed`；
- 快照：`snap.dense_hit`、`snap.binary_hit{level}`、`snap.extra_hit`、
  `snap.level_missing`（目标 0）、`snap.orphan_reclaimed`、`snap.dense_evicted`（压力下可非 0，不是错误）；
- 组合：`reuse.kv_hit`、`reuse.snap_hit`、`reuse.gap_tokens`（= 重算量）、
  `reuse.fallback_level`（二分退了几级）；
- append：`cached_tokens == prev_tail_tokens` 比例（100%）；
- TTFT：append / rollback / leave-gap 三档 p50/p99。

**验收门**

1. append：追加消息 → 命中上一条消息末尾；只 prefill 新增。
2. rollback：回撤 1/2/3 条 → 命中 dense；重算 ≤ dense 窗口。
3. 深回撤：目标 X 处无快照 → 命中 `L/2` 或 `L/4` 二分点，**重算 ≤ X/2**。
4. leave-gap：98→101→104 重放，不再 `keep=0`。
5. 解耦：人为丢掉快照 → KV 命中面仍上报、KV 块不被驱逐；恢复快照后
   无需重算 KV。
6. 内存：3 会话 × (K + log2(L)) 快照可算；orphan 不增长；Dense/tail 在压力下可被逐出且不泄漏。
7. 回归：3090 档、纯文本、单会话逐字节旧行为。

### 3.8 分阶段

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | 两套观测面：`kv_recall` / `snap_recall` / level/orphan/hits | 看到现状分布 |
| P1 | 快照二分层级 + dense + extra LRU/频率；transient ring | 深回撤 ≤ X/2 |
| P2 | 两套召回解耦；KV 不因快照缺失被裁；tail 优先保留跨轮 | append 100%；解耦门 |
| P3 | rollback 降级不删；leave-gap 用二分边界；orphan 回收 | rollback/leave-gap 门 |
| P4 | 预算/设备槽分配/层级 caps 调优 | 内存门 + TTFT |
| P5 | 3080 双卡现场 + 3090 回归 | 全门 |

### 3.9 开放问题

1. **K**（dense 尾部）：默认 4？
2. **二分点是否每级显式补拍一张**：当前用 dense+sparse 选点再做 `Binary` 分类；
   若要严格保证"每级都有锚点"，可在 prefill 时补拍（每级 1 张，淘汰仍按档位）。
3. **二分点的基准**：按 token 的 `L/2^k`（用户例子）还是按消息数的 `N/2^k`？
   建议 token 基准、snap 到最近的 ≤ 消息边界。
4. **客户端 cursor**：是否逐字回传 assistant 输出，决定 cursor 是否单独保留。
5. **host 预算**：snap/KV 份额（建议先 6/4 试）。
6. **session close**：立即回收还是交给 LRU。

---

## 4. 三层资源管理架构（代码映射）

> 注意：这里是**资源管理**的三层，与 `serve/session.rs` 头注里的
> **应用层级**（Session ⊃ Turn ⊃ Step）是正交的两套东西。
> 应用层级回答"谁在说话"，资源层级回答"资产归谁、谁管策略、谁管物理"。

### 4.1 L1 Session 管理层

**职责**：每个会话拥有哪些资产。

- 会话 token/位置/图像账（模型语义）；
- **KV 资产句柄**（链身份，不是物理块号）；
- **GDN 快照句柄集合**（当前完全没有落在 L1）；
- **消息边界锚点**（msg_index / boundary / level）；
- **保留优先级窗口**（最近 K 条 + 二分 level；不是 pin）；
- session LRU / close。

**现状代码**：

- `serve/session.rs::AgentSession`：`tokens / cached_len / block_table /
  mtp_block_table / chain(ChainCursor) / img_spans / pads / positions /
  positioned / position_epoch`；
- `serve/turn.rs::ActiveTurn`：`gdn_slot / grant / last_snap / msg_snaps /
  prompt_ids / positions / fed / out / img_spans / img_pads`；
- `SessionTable`：会话 LRU / `max_sessions`。

**层泄漏**：

- `block_table` / `mtp_block_table` 是**物理槽号**（L3 细节）住在 L1；
- `chain: ChainCursor` 是**块身份**（L2 细节）住在 L1；
- L1 **没有快照资产表**：快照归属只在 L2 `StateStore.snaps[].session` 和
  `ActiveTurn.last_snap`，会话空闲时 L1 根本不知道"我有哪些快照"；
- `msg_snaps` 挂在 **ActiveTurn** 上，turn 结束即消失，下一轮要重新算；
  它不是"会话资产"，而是"本次 turn 的临时视图"。

**目标形态**：

```rust
struct SessionAssets {
    session: u64,
    kv_chain: ChainHandle,          // 逻辑句柄（L2 解析）
    snaps: Vec<SnapHandle>,         // 逻辑句柄（L2 解析）
    anchors: Vec<AnchorMeta>,       // msg_index/boundary/level/class
    tokens/positions/img_spans,     // 模型语义
}
```

L1 不出现 `Tier`、slot id、DMA、`PoolLayout`。

### 4.2 L2 Block 管理层

**职责**：两套独立管理器 + 召回/驱逐策略。

- **KV block manager**：内容寻址树（`KvStore`）、命中/登记、块级 LRU+频率、
  tail 优先保留（非 hard pin）；
- **GDN snapshot manager**：`StateStore`（行 + 快照）、durability、
  hits/recency、dense/binary/extra 档位（不是 pin）；
- 两者**独立预算/独立驱逐**，只在召回时组合（§3.3）。

**现状代码**：

- `store/rel.rs::Rel`（门面）：`leg` + `state` + `snap_host` + `disk` +
  `snap_disk` + `index` + `ledger` + `policy` + `admission` + `reservations`
  + `budget` + `evict`；
- `store/leg.rs::BlockLeg`：`kv` + `host` + `layout` + `ops` + `planes`；
- `store/state.rs::StateStore`：`rows` + `snaps` + `tick` + `hits/misses`；
- `store/kv/`（tree/store）、`hooks.rs`（驱逐计划）、`policy.rs`、
  `reservations.rs`、`select.rs`。

**层泄漏**：

- `Rel`/`BlockLeg` 里直接持有 **host/disk 物理**（`HostTier`、`DiskTier`），
  L2 和 L3 混在一起；
- 设备物理在 store 外（engine `dev.pool`），宿主/盘在 store 内 ⇒ L3 被劈成两半；
- 前缀复用/leave-gap 的**策略**在 `sched/scheduler.rs`（engine），不在 L2；
- `Tier` / `SlotTable` / `snap_bytes` 等物理概念从 L2 的 API 泄漏到 engine。

**目标形态**：

```rust
trait KvBlockManager {
    fn recall(&mut self, tokens: &[u32], tags: &[BlockTag]) -> KvRecall;
    fn register(&mut self, tokens: &[u32], tags: &[BlockTag]) -> ChainHandle;
    fn mark_priority(&mut self, chain: ChainHandle, class: SnapClass);
    fn evict_plan(&mut self, need: Need) -> Vec<Move>;
}
trait SnapshotManager {
    fn recall(&mut self, boundary: usize) -> SnapRecall; // dense/binary/extra
    fn capture(&mut self, boundary: usize, level: Level) -> SnapHandle;
    fn mark_priority(&mut self, snap: SnapHandle, class: SnapClass);
    fn evict_plan(&mut self, need: Need) -> Vec<Move>;
}
```

### 4.3 L3 Physical 管理层

**职责**：设备/宿主/磁盘的块分配、搬运、预算。只认**不透明句柄/槽**，
不认 session/message/chain 语义。

**现状代码**：

- 设备：`crates/backends/cuda`（pool / slots / `snap_dev` / `swap`）、
  engine `dev.pool`（`PoolLayout` / `SlotTable`）；
- 宿主：`store/host.rs::HostTier`（KV 键 `ChainHash`，快照键 `SnapKey`）；
- 磁盘：`store/disk.rs::DiskTier` + `disk_index.rs` + engine `mem/disk_io.rs`；
- 层账：`store/tier.rs::TierLedger` / `SharedLedger`、`budget.rs`。

**层泄漏**：

- 设备在 engine、宿主/盘在 store，两个物理实现两套 API；
- 搬运的**计划**在 `store/hooks.rs`，**执行**在 `engine/running.rs` 的
  `swap` / `mem/disk_io.rs`；
- `Tier` 枚举、`SlotTable`、`snap_file_bytes` 同时出现在 L2/L3/engine。

**目标形态**：

```rust
trait TierStore {
    fn acquire(&mut self, tier: Tier, bytes: usize) -> Slot;
    fn move_(&mut self, h: Handle, from: Tier, to: Tier) -> MoveReceipt;
    fn release(&mut self, h: Handle);
    fn tier_of(&self, h: Handle) -> Tier;
}
```

设备/宿主/盘各实现一份；L2 只发 `Move`，不碰 DMA/槽位。

### 4.4 编排层（薄）

**职责**：把 L1 的会话视图 → L2 的召回/驱逐计划 → L3 的搬运串起来。

**现状代码**：

- `sched/scheduler.rs`：`plan_prefix_reuse` / `plan_image_leave_gap` /
  `ensure_start_chain`；
- `running.rs::store_hook` + `mem/store_hook.rs`：`Acquire/Release/Maintain`；
- `exec/phases.rs`：`capture_gdn_snap`；`exec/turn_end.rs`：commit/release。

**问题**：scheduler 同时持有 L1（session/turn）、L2（命中/驱逐）、L3（池容量/
DMA）三层的知识，是当前最大的**跨层耦合点**。

**目标**：scheduler 只做排序/切分/事件泵；策略在 L1（保留优先级窗口）和 L2
（recall/evict）；物理在 L3。它不应该再直接算 `max_hit`、选 victim、改
`block_table`。

### 4.5 层间契约（设计硬约束）

| 方向 | 接口 | 约束 |
|---|---|---|
| L1 → L2 | `resolve(prompt)` / `recall()` / `mark_priority()` / `close()` | L1 只传 prompt、会话 id、逻辑锚点/档位；不传槽号 |
| L2 → L3 | `evict_plan(need) -> [Move]` / `acquire/move/release` | L2 只发"把 X 从 A 层搬到 B 层"；不碰 DMA/地址 |
| L3 → L2 | `tier_of` / `MoveReceipt` | 只回执层与成败；不解释内容 |
| 编排 | `start = last_snap_leq(min(kv_hit, end))` | 只组合，不拥有策略 |

不变量：

- L1 不出现 `Tier` / slot id / `PoolLayout`；
- L2 不出现 `Session`/message 语义（只带 `session_id` 做 pin/归属）；
- L3 不出现 `ChainHash` 的内容解释（只当不透明键）；
- KV 与快照两套管理器不共享驱逐（只共享档位提示，不共享硬保护）。

### 4.6 迁移顺序（低风险）

| 步 | 动作 | 风险 |
|---|---|---|
| 0 | 加 `SessionAssets` **只读视图**（从 `AgentSession` + `StateStore.snaps` 投影） | 零行为变化 |
| 1 | `last_snap`/`msg_snaps` 收进 L1 的 session 资产表；`recall()` 做成门面 | 小 |
| 2 | `Rel` 拆成 `KvBlockManager` / `SnapshotManager` + `PhysicalTiers` trait；旧 `Rel` 留 shim | 中 |
| 3 | `plan_prefix_reuse` / `plan_image_leave_gap` 的策略下沉到 L2；scheduler 只排序 | 中 |
| 4 | 二分快照 + 档位淘汰 + 解耦召回（§3） | 中 |

---

## 5. 与 v1/v2 的差异

| 版本 | 快照稀疏 | 两套关系 | 回撤 |
|---|---|---|---|
| v1 | 无（全消息边界同密度） | Anchor 同生共死 | 整段 drop |
| v2 | stride 4/16/64 | 两套管理器 + 查找对齐 | 降级，不删 |
| v3 | **二分 L/2, L/4…**（重算 ≤ X/2） | 两套管理器 + **独立召回**，缺失不互害 | 档位淘汰/降级，不删 |

---

## 6. 施工状态（2026-11-12 第一/二刀）

已完成（worktree `feat/session-kv-snap`，未提交）：

| 项 | 落点 | 状态 |
|---|---|---|
| L1 会话资产：`tail_snap` + `msg_snaps` 跨 turn 保留 | `serve/session.rs` | ✅ |
| turn 收口/中断写回 L1 资产 | `exec/turn_end.rs` | ✅ |
| 最近 K 条消息边界无条件保留（`snap_msg_recent=4`） | `serve/msg_boundary.rs` + `[pool]` | ✅ |
| `SnapClass`（Transient/Extra/Binary/Dense）+ 档位淘汰顺序 | `store/state.rs` + `store/hooks.rs` | ✅ |
| prefill 边界分类（dense/binary/extra） | `msg_boundary::classify_snap_points` + `running.rs` | ✅ |
| turn 末 Dense 尾锚点（cursor 升级；**不是 pin**） | `exec/turn_end.rs::complete` + `exec/spec.rs` | ✅ |
| L1 资产只读投影 | `running.rs::session_assets` + `/debug/vram.sessions` | ✅ |
| **快照准入（纯账）**：`snap_session_cap` / `snap_structural_reserve`；同键 Touch、满则 Replace、不划算 Skip | `store/state.rs::admit_snap` + `store/rel.rs` | ✅ |
| **capture 早退/替换接线**：Skip 不选槽、不拷贝；Replace 先整体释放 victim（设备槽 + 宿主/盘 + 账） | `exec/spec.rs::capture_snap_at_boundary` | ✅ |
| **L1 快照键沿用 + 重分类**：`msg_snap_keys` 跨轮保留；submit 按新 `msg_snap_classes` 重分类（旧 Dense 尾巴降级，不搬内容）；截短/前缀失配时旧锚点统降 Extra | `serve/{session,turn}.rs` + `running.rs::submit` | ✅ |
| **最近 K 个 turn 尾**：新 tail 入 `recent_tails`；超过 K 的旧尾降为 Extra（Dense 只是最后淘汰，不是永久保护） | `exec/turn_end.rs` + `serve/session.rs` | ✅ |
| **会话关闭释放 L1 归属**：内容保留，非 Transient 改为无主 Extra（其他会话仍可复用） | `running.rs::close_session` + `state.rs::detach_session` | ✅ |
| **host-pressure 混合出序统一**：同会话内 State 腿按 `SnapClass` 档位序（Extra 先于更冷的 Dense） | `store/hooks.rs::order_state_in_session` | ✅ |
| 配置：`[pool] snap_session_cap` / `snap_structural_reserve` + 越界拒启；3080 档 20/4 | `shared/config` + `profiles/qwen38_27b-3080_tp2.toml` | ✅ |
| **gapped 链同 turn 内持有**：`block_hashes` 只在当前 turn 有块表时用于 session-local；turn 末释放表，下一轮按内容/快照重新取链或重建 | `serve/session.rs` + `sched/scheduler.rs` + `exec/spec.rs` | ✅ |
| **相邻图半删识别**：`detect_text_delete_tagged` 枚举连续子段 + 新旧图像内容指纹消歧；截图替换首图不再把删除误判成整组 | `vision/img_evict.rs` + `running.rs::image_pad_law` | ✅ |
| **位置洞只属于当前 turn**：leave-gap 的 `block_table` 只在本 turn 内有效；`release_turn_table` 在 turn 末释放全部表并清空位置洞元数据，下一轮按内容/快照重新取链或全量重建 | `serve/session.rs` + `exec/turn_end.rs` + `sched/scheduler.rs` | ✅ |
| **全量重建位置压实**：`cached_len==0` 且 turn 位置有洞 ⇒ 重算紧凑 positions，链可发布共享树，下一轮不再 0 命中 | `sched/scheduler.rs::start_one` | ✅ |
| **非连续多图全删（多段 leave-gap）**：`DeletionPlan` 多区间；保留全部幸存 padded 页+orphan pad；旧尾 `tail_snap` 重挂，只重跑新尾巴 | `vision/img_evict.rs` + `sched/scheduler.rs` + `running.rs` | ✅ |
| ~~热前缀表跨 turn~~ | **已移除（2026-11-13）**：turn 末释放全部 `block_table`；所有块回 store 缓存面，按引用/LRU 回收；不存在跨 turn 不可动表 | `exec/turn_end.rs::release_turn_table` | 移除 |
| 单测：msg_boundary 7 / snap_leg 4 / store victim 2 / **admission 5 + reclassify 1 + detach 1 + drop_asset 1 + mixed-order 1 + carry_keys 1 + config 1** | — | ✅ |

指标（`/debug/metrics/snap`）：`snap.admit.add/touch/skip/replace`、
`snap.reclassify`(+`.miss`)、`snap.demote.tail`、`snap.session.detached`。
`/debug/metrics/kvmem` 另有 `kvmem.session_local_hit/blocks`（gapped 链同会话复用）。

**TTFT 实测（2026-11-13，自测，非用户）**：36k prompt + ~100 字符 append：

| 场景 | client TTFT | server prefill | 说明 |
|---|---:|---:|---|
| 3 图，修复前 | 2076ms | 555ms/92tok | vision 预处理每轮重做 |
| 3 图，vision LRU 后 | 794ms | 542ms/92tok | VLM tower cache 已命中 0ms |
| 纯文本 | 235ms | 195ms/92tok | 固定 host 税 |
| `prefill_chunk=128` | — | 195ms/92tok | cold 30k 反而 20.5s；不是 chunk 大小 |

剩余：prefill 固定 host 税（~150ms，6395 条 alloc/launch/free）+ 图像 KV
attention（~350ms）。<100ms 需要 prefill graph / persistent kernel，不是配置能解。

**没有硬保护节点**：

- 所有快照共用同一条优先级队列；活跃 turn 的最新边界也是 `Dense` 优先级；
- 会话 tail 只是 `Dense` 档（最后淘汰，但空间不足时照样走）；
- 任何档位都有淘汰路径，不会因"保留"而泄漏；
- 大档位（`Dense`/`Binary`）在 `snap_victim_order` 里排后，`Transient`/`Extra` 先走。

未做（下一步）：

1. **二分节点的显式补拍**：当前用 dense+sparse 选点 + `Binary` 分类；若要严格保证
   "每级都有锚点"，prefill 时按 `P/2^k` 补拍（淘汰仍按档位，不 pin）；
2. **KV 与快照解耦召回**（`kv_recall` / `snap_recall` 分开，start 组合规则）；
   当前 `sched::reuse_face` 在无快照边界时会把 `m` 降到 0，`plan_fresh` 随即
   `reset_session_for_rebuild` 释放块表 —— 目标：`start=0` 但 **KV 资产不丢**
   （块仍在缓存树里可被后续复用）；
   ✅ 2026-11-13：`plan_image_leave_gap` 已优先用 `session_local_hit` 取旧链
   （`block_hashes` + `block_table`，仅当前 turn 有表时），表已释放则回落 `match_prefix_tagged`/重建；
   gapped 会话的第二次删除不再因共享树缺 suffix 落 `旧前缀链不足`。
3. **L2 拆分**（`KvBlockManager` / `SnapshotManager` / `PhysicalTiers`）；
4. **orphan/transient 的独立回收**；
5. 端到端重放（98→101→104）与 `cached_tokens == 上一条消息末尾` 验收；
6. **多会话公平性**：per-session cap 已落地，但全局 host/device 压力下
   跨会话仍按「冷会话先 + 档位」；尚未做按活跃度分配/配额；
7. **Replace 原子性**：当前是「先删 victim → 再拍新快照」；拍失败（GPU 错）
   则 turn 响亮失败，victim 不恢复（快照内容无法廉价回滚）。这是已知取舍，
   不是静默失败。

验证命令：

```bash
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-engine --lib msg_boundary
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-engine --lib snap
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-store --test snap_leg
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-store --lib snap_victim_order
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-store --lib state::tests::admit
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-store --lib drop_snap_asset
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-store --lib reclassify_snap
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-store --lib detach_session
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-engine --lib carry_msg_snap_keys
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-store --lib extra_snapshot_goes_before_colder_dense
GDN_TRITON_BUILD=0 CARGO_TARGET_DIR=/home/div/Documents/codes/packages/owl/target \
  cargo test -p owl-shared --lib validate_rejects_structural_reserve
```
