# 显存调度架构 review（store × engine）—— 2026-10-27

> 工作树：`packages/owl-worktrees/store-vram-review`
> 分支：`review/store-vram-arch`（based on `ec01388e`）
> 性质：**只读 review + 下一阶段设计提案**；本文件不改生产代码。
>
> **落地注记（2026-10-28）**：V0–V5 已由 `feat/vram-plan` 落地
> （`db86f987` / `b0573474` / `4012b704`）。本文保留为立项时的架构 review
> 与迁移刀序依据；当前实现细节以 `docs/devlog/memory/store.md` §33–§36 为准。
>
> 结论先行：store 的 A 轴（容量/三层）与 B 轴（选择层）已经各自落地，
> 3-hook 也已收口；但**显存调度还没有一个单一事实源**。当前是
> “两套预算算式 + 两套准入账 + 状态腿一条旁路”的拼装形态。
> 下一步不应继续加特性，而应先把这三处收成一个闭环：
> **MemoryPlan（boot）→ Reservation（runtime）→ Reconciliation（report）**。

---

## 0. 当前形态（calltree）

```text
boot
  Engine::on
    └─ ledger = tier_budget_of(host_cache_mb, snap_host_mb, disk…)
    └─ BootVram{card_free, total, live_at_new}
  Engine::run
    └─ boot_rank
       ├─ StatePool::alloc
       │  ├─ PerTokBytes + FixedAfter
       │  └─ pool_cap(driver free, target, reserve, fixed_after)
       │     ⇒ pool_tokens / nb / snap / gdn / draft
       └─ into_running
          ├─ assemble_budget(weights=live_loaded−live_at_new,
          │                  runtime_floor=live_now−live_loaded−pools)
          ├─ Rel::open(BudgetInputs, KV leg, ledger)
          └─ GraphFleet::assemble
runtime
  pump
    ├─ step_need (Admit/Grow 槽位)
    ├─ admission_gate
    │  └─ sync_pool_account → Rel::check(floor, ctx)
    ├─ StoreHook::Acquire(Capacity)
    │  └─ pool_housekeeping
    │     └─ Rel::plan_need → plan_capacity(两腿候选)
    │        → apply_verdict(swap/drop) → Applied
    ├─ schedule
    │  └─ ensure_for_len（纯分配；不足 = Exhausted）
    ├─ execute_begin
    │  └─ StoreHook::Acquire(Load) → store_load
    │     ├─ ensure_host_resident（disk→host）
    │     ├─ 必要时 pool_housekeeping
    │     └─ plan_onload → swap → commit_onload
    ├─ prefill/decode/spec
    │  ├─ snap_ensure_slot → StoreHook::State(Slot)
    │  │  └─ store_state_slot（快照受害者的**独立**路径）
    │  └─ snap_ensure_resident → StoreHook::State(Resident)
    │     └─ Host/Disk → snap_onload
    └─ Idle
       ├─ trim/drain
       └─ Maintain::Idle（target = pool/8，capped 16）
turn_end
  └─ StoreHook::Release
     ├─ register_chain_owned（H3）
     ├─ release_turn_table（H6）
     ├─ return_row（S2b）
     └─ N4 sweep/report/force-release/trim
```

---

## 1. Review findings

### P0-1 `runtime_floor` 在生产路径恒为 0，`VramBudget` 不是真正的 A5.1 分解

证据：

```text
crates/engine/src/engine.rs:900-924
  pools_total = (kv + gdn + snap + draft) × world
  runtime_floor = live_now − live_loaded − pools_total
```

`into_running` 里 `live_loaded` 与 `live_now` 是相邻两次 `live_bytes()`：

```text
crates/engine/src/engine.rs:1208-1210
  live_loaded = owl_shared::vram::live_bytes();
  assemble_budget(..., live_loaded, owl_shared::vram::live_bytes())
```

所以：

```text
runtime_floor ≡ 0
```

真实运行日志也如此：

```text
[boot] 准入预算:卡可用 19851.8MiB · 权重 10757.8MiB · 运行底价 0.0MiB …
        · 卡级实测=false
```

后果：

- 生产档走固定池（`kv_face_bytes > 0`）时，`decompose` 允许卡级三项缺项；
  所以现在**没炸**，但 `VramBudget` 已经不是“实测分解”，而是“池墙面”；
- 动态池档（`kv_face_bytes = 0`）会因 `runtime_floor=0` 被当成“未实测”而拒启；
- `card_measured` 永远 false；报告面无法回答“固定面到底在哪、余量多少”；
- 与 `StatePool::pool_cap` 的 `FixedAfter`（图 slab 320MiB / warmup 138MiB /
  GDN / snap / draft ring）是**另一套算式**。两套算式当前靠固定池“不对账”，
  一旦引入动态池、权重分层或 B 轴变更就会漂。

这是本次 review 的最高优先级：**不是数值 bug，而是预算来源分裂**。

### P0-2 快照槽选择绕过 `plan_capacity` 的三层策略，磁盘层被浪费

`store_state_slot`：

```text
crates/engine/src/running.rs:1367-1427
  order = Rel::snap_victim_order(&[])
  pick = pool.snap_pick_slot(key, order)
  Victim ⇒ snap_offload(victim, slot)   // 只搬 host
       失败 ⇒ drop_snap_host + state.drop_snap + snap.drop
```

而 `plan_capacity` 对快照已经支持 `Sink::Host | Disk | Drop`：

```text
crates/store/src/hooks.rs
  EvictOut::Snap { key, sink }
  Sink::Disk ⇒ snap_spill
```

后果：

- 快照池满时，即使 `snap_disk_mb > 0`、盘层有空位，`store_state_slot`
  也不会把最冷快照溢写到盘，而是直接丢内容；
- 这条路径不经过 `Verdict / Applied`，`StoreOutcome.applied.snap_*` 不完整，
  `store.hook` 指标也看不到这次快照决策；
- 与 A 轴的“先下沉、再驱逐、仍不足才拒”次序不一致。

建议：把 `StateAcquire::Slot` 纳入 `plan_capacity` 的同一张票（见 §2.3），
至少要让 `Sink::Disk` 在这条路径可达。

### P0-3 活跃 turn 的预留按“每会话整条块表长度”求和，共享前缀会双计

`sync_pool_account`：

```text
crates/engine/src/running.rs:2188-2210
  active_holds[i] = session.block_table.len() × block_bytes
  held = min(Σ active_holds, occupied − movable)
  ad.set_floor(turn, active_holds[i])
```

`Admission.reserved_bytes()` 是各 turn floor 之和。若两个活跃 turn 共享前缀：

```text
共享块 refs ≥ 2 ⇒ 不在 movable
每个 turn 的 block_table.len() 都包含共享块
⇒ Σ floors > occupied − movable
⇒ reserved + cache > 池实占
```

后果：

- `max_batch > 1` 时，共享前缀会抬高 `reserved`，让第三个 turn 被误拒；
- `reserved + cache = occupied` 的不变式只在“无共享前缀”时成立；
- 报告面的 `reserved` 不是“唯一活跃工作集”，而是“会话表长度之和”。

当前 3080/3090 生产档 `max_batch = 1`，所以没有暴露；B5.1 多会话/合批一旦
打开就会踩到。建议引入 `ActiveSet`（按块身份去重）或“共享块另账”的口径。

### P0-4 `Acquire` 没有 Grant/Reservation，plan 与 allocation 之间没有结构约束

`StoreOutcome` 只有：

```text
crates/engine/src/store_hook.rs
  applied / loaded_blocks / state_slot
```

没有“已授予槽数/字节”的 token。`Acquire::Capacity` 只执行一次
`pool_housekeeping`，随后 `schedule` 里的 `ensure_for_len` 重新按池自由槽
纯分配：

```text
crates/engine/src/running.rs:2874-2892
  plan_need → apply_verdict
```

当前引擎是单线程 actor，所以“判完到分配”之间没有别的分配者；但这是
**纪律靠时序、不是靠类型**：

- `Acquire::Load` 也是先 `pool_housekeeping` 再 `plan_onload`，没有配额移交；
- 未来 batch / 异步预取 / 多 actor 时，plan 与 allocate 之间可被插队；
- 无法回答“这个 turn 的 grant 是否已经兑现”，只能看 `Exhausted`。

建议把 `Acquire` 的返回面升级为：

```rust
Grant { id, slots, bytes, owner, expires_at? }
ensure_for_len(grant, table, target)
Release(grant)
```

这与设计卷 §3.2 的 `Grant` 草案一致，但目前代码只实现了 `Applied`。

### P1-1 两套准入账没有对账

- `crates/engine/src/sched_policy.rs::Admission`：按 turn 预留页、会话守卫、退避；
- `crates/store/src/budget.rs::Admission`：按 floor/cache/session_held/elastic。

`scheduler::store_facts()` 读 `turns.admission.reserved_pages()`；
`admission_gate()` 调 `sync_pool_account()` 更新 `rel.admission` 后调 `Rel::check`。
两者是不同账本，没有交叉校验：

- `Rel::Admission` 的 floor 来自 `sync_pool_account`；
- `sched_policy` 的 reserved 来自 `note_started`；
- 两者可以在共享前缀、拒绝、队列、turn_end 顺序上漂。

设计卷 §10.3 说“两层判决有意保留”是合理的，但**保留两层判决不等于保留两本账**。
建议：`sched_policy` 只消费 `StoreFacts` 做策略；预留账唯一真源在 store
（或反过来），另一层只做只读投影，并在每次段边界断言相等。

### P1-2 `Protected` 在生产路径恒为空

`pool_housekeeping`：

```text
crates/engine/src/running.rs:2890
  plan_need(need, &Protected::default())
```

KV 活跃面由 `SlotPool.refs` 兜底（`Busy`），所以目前不误逐；
但：

- 快照面没有引用计数，`StateStore::evict_candidates` 只按 `protect_snaps`
  排除；
- engine 从不填 `Protected.snaps`；
- 当前 spec 的 `spec_snap` 是独立缓冲，暂时不依赖前缀快照；
- 将来“活跃 turn 的快照也是回滚/删除恢复钥匙”时，没有显式保护面。

建议：把 `Protected` 的构造权收进 engine 的 `store_hook` 层，
段边界自动注入：
`active chains`（虽然 refs 已保护，仍用于报告/断言）+ `active snapshot keys`
（至少本 turn 最新边界、正在回滚的 key）。

### P1-3 固定水位仍在：`pressure_need` 与 `idle_tick` 都是 `pool/8`

```text
crates/engine/src/running.rs:1695-1723
  pressure target = clamp(total/8, 1, 64)
  idle target     = clamp(total/8, 1, 16)
```

设计卷 §12.4 明确反对“固定水位”，改为需求驱动。
`Pressure` 只剩 `PoolExhausted` 后触发，这可以理解；但 `Idle` 仍然每 N 拍
按池比例搬，属于“后台预搬”的固定策略。文档 §20-B9 也把它列为“应显式立项，
而不是默认行为”。

建议二选一：

1. 明确把 `Maintain::Idle` 定义为“后台整理策略”，给它独立配置和预算；
2. 默认关闭 idle 预搬，只在 host/disk 压力或显式 prefetch 时触发。

### P1-4 固定池挡掉了驱动 free 的运行时变化

生产档 `kv_face_bytes > 0`，`VramBudget.face_bytes()` 直接返回池字节；
`v_card_bytes / weights / runtime_floor` 不参与准入。池在 boot 预分配，
所以“池内”是对的；但：

- 外部进程、CUDA context、图捕获残差、allocator 惰性保留变化时，
  驱动 free 可能低于 reserve；`vram_budget_guard` 只能等到分配时炸；
- `Maintain::Pressure` 按池自由槽，不看驱动 free；
- 报告里 `卡级实测=false`，运维无法从统一入口判断“是池满还是驱动紧”。

建议：固定池仍以池墙为准，但增加一个 **Headroom gate**：
每次 `Acquire::Capacity`/`Admit` 读一次 `mem_get_info`（或缓存的低频采样），
`free − reserve` 低于阈值时只触发 `trim/drain` 或拒新，不假装是池槽不足。

### P1-5 状态腿 row 的 owner 仍是 `System`

`Rel::leg_accounting()`：

```text
GDN rows = Owner::System
snapshots = Owner::Cache
```

`StateStore` 本身知道 `row` 的借还，但报告不把 row 归到 `Turn(t)`。
后果：`/debug/vram` 无法回答“哪个活跃 turn 占了几行/多少字节”；N4 也只看
`DeviceHandle` 的 Turn owner，不看 GDN row。

建议：`StateStore::borrow_row/return_row` 接收 `Owner`（或 turn id），
报告用 `Turn(t)`；`Rel::open` 的 row 预算与 active reservations 同源。

### P2-1 `store.report()` 的 host/disk 总账是“行求和”，不是账本总数

```text
crates/store/src/rel.rs:1133-1171
  tiers = ledger.tiers().report()
  tiers.rows += leg_accounting().rows
  host_used = Σ rows[Host]
```

这修了“腿自报没进总账”的旧问题，但也意味着：

- 如果 `TierLedger` 和腿报告对同一字节各有一行，`host_used` 会双计；
- 当前 KV/快照宿主不在 `TierLedger`（只在腿里），Index 在 ledger 里，
  所以暂时没有重复；但这是**约定**，不是不变量；
- `EventBook` 与腿报告的交叉校验没有强制断言。

建议：报告面加 `reconciliation` 段，逐层比较
`ledger` / `leg` / `state` / `event`，不一致 = 响亮 WARN，不给“看起来平”的数字。

### P2-2 `pool_cap` 的固定面仍是硬编码常数

```text
crates/engine/src/state.rs:1939-1972
  GRAPH_SLAB_BUDGET = 320MiB
  WARMUP_MISC_BUDGET = 138MiB
  fixed_after = ...
```

图族实际捕获后的 slab 总账是动态的（日志里常见 0MiB / 284MiB 等），
当前用 320MiB 保守覆盖。这符合“安全方向”，但不满足 A5.1 的“每项有实测来源”。
B 轴 working set / graph 族变化后，320/138 还会继续变成历史常数。

建议：`MemoryPlan` 里把 graph slab 作为 `MeasuredAfterCapture` 的项，
boot 先用 `GraphPlan` 的预算上界，捕获后用实测回填；报告同时显示
“预算上界 / 实测”。

### P2-3 `store_load` 的容量计划与回灌不是一笔账

```text
crates/engine/src/running.rs:1440-1489
  if free < need → pool_housekeeping(Need::Grow)
  plan_onload
  swap
  commit_onload
```

如果 `pool_housekeeping` 只搬了部分（`deny` 但 `out` 非空），随后
`plan_onload` 仍会尝试；失败点落在 `plan_onload`/`Exhausted`，而不是
“Grant 不足”。这与 P0-4 同源。

### P2-4 `Maintain::Pressure` 只处理池自由槽，不处理 host/disk 压力

```text
crates/engine/src/running.rs:1691-1710
  target = total/8; snap_bytes only if snap_host_free_slots > 0
```

- host 快照满、KV 池空闲时，不会触发；
- disk 满时，只有 `spill_*` 执行面自己临时腾位；
- `NeedWhy::Pressure` 的 snap 目标最多一条快照，不能表达“host 快照面整体超压”。

建议：`Need` 增加 `host_bytes / disk_bytes` 目标，或 `Maintain` 增加
`TierPressure { host, disk }` 维度；目前 `snap_bytes` 只能算半个。

---

## 2. 下一阶段设计：单一显存调度闭环

### 2.1 MemoryPlan：boot 唯一事实源

目标：把 `pool_cap` + `FixedAfter` + `assemble_budget` 三处算式收成一个
可打印、可断言、可复用的结构。

```rust
pub struct MemoryPlan {
    pub epoch: u64,
    pub card: CardPlan,          // total / free_at_boot / reserve
    pub fixed: Vec<MemComponent>,// weights / graph / warmup / rows / snap / draft
    pub pool: PoolPlan,          // tokens / pages / per_token_bytes
    pub tiers: TierPlan,         // host / disk split
    pub policy: PlanPolicy,      // fixed/dynamic, admission defaults
}

pub struct MemComponent {
    pub id: ComponentId,         // Weights / GraphSlab / Warmup / GdnGrid / ...
    pub bytes: usize,
    pub source: PlanSource,      // Measured / Derived / Configured / Conservative
    pub tier: Tier,
    pub owner: Owner,
}
```

编译步骤：

1. 先测 `weights`、`graph_slab`、`warmup`、`gdn_row`、`snap`、`draft`；
2. 计算 `pool_budget = card_free − reserve − Σfixed`；
3. `pool_tokens = min(max_seq, pool_budget / per_token)`；
4. `TierPlan` 由 `host_cache_mb + snap_host_mb`、`disk_cache_mb + snap_disk_mb`
   显式分片；
5. 断言：`Σfixed + pool_bytes + reserve ≤ card_free`；
6. `BudgetInputs` 变成 `MemoryPlan::to_budget_inputs()` 的**投影**，
   不再由 `assemble_budget` 用残差反推；
7. `StatePool::alloc` 与 `Rel::open` 都消费同一个 `MemoryPlan`；
   `runtime_floor` 由 `fixed` 中显式项求和，而不是 `live_now − live_loaded − pools`。

验收：

- boot 日志一行 `MemoryPlan{components, source, sum, reserve, pool}`；
- `--check` 能验证分解封闭；
- 固定池档 `card_measured=true`（因为 fixed 项都有来源），
  `runtime_floor` 不再为 0；
- 动态池档用同一 plan，不再有第二套 `pool_cap`。

### 2.2 Reservation：把 `Acquire` 变成“授予”

```rust
pub struct ReservationId(u64);

pub struct Grant {
    pub id: ReservationId,
    pub owner: Owner,
    pub slots: usize,
    pub bytes: usize,
}

pub enum ReservationKind {
    ActiveFloor,   // turn 的不可驱逐地板
    Elastic,       // 地板之上的可动面
    Cache,         // idle/prefix cache
    Row,           // GDN row
    Snap,          // 快照
    Temp,          // 临时工作区
}
```

- `Acquire::Capacity` 返回 `Grant`（不只是 `Applied`）；
- `Acquire::Load` 返回 `Grant` + `loaded_blocks`；
- `ensure_for_len(grant, table, target)` 只消费 grant；
- `Release` 归还 grant；
- `Admission` 是 `Grant` 的集合，不是另一个按页求和的影子账；
- `sched_policy::Admission` 只输出策略判决，不再保存 `reserved_pages`；
  `StoreFacts` 从 `ReservationLedger` 读。
- 单线程下仍可保留 plan→apply 两相，但“分配必须带 grant”由类型保证。

共享前缀：

- `ActiveSet { blocks: BTreeSet<ChainHash> }` 按内容身份去重；
- `Reserved` = 唯一块字节；`Cache` = 可动块字节；
- 每 turn 的 `Floor` 只用于 G2b 保护与报告，**不直接相加成池占用**；
- 增加 `shared_blocks` 账，报告显示“共享块字节 / 被几个 turn 引用”。

验收：

- 两个共享 50% 前缀的活跃 turn：`reserved + cache == occupied`；
- 第三个同尺寸 turn 的准入仍按唯一工作集计算，不因共享前缀误拒；
- `ensure_for_len` 无 grant ⇒ 编译期/结构化错误。

### 2.3 State leg：走同一张票

把 `StateAcquire::Slot` 改成：

```text
plan_state_slot(key, protect) -> Verdict {
    out: [Snap { key, sink: Host|Disk|Drop }],
    deny: ...
}
apply_verdict(verdict) -> Applied {
    snap_offloaded / snap_spilled / snap_dropped
}
```

`store_state_slot` 只保留“key→槽”的物理映射；受害者选择、宿主/盘层选择、
丢弃都由 `plan_capacity` 的同一份 `Room` / `EvictPolicy` 决定。
`Protected.snaps` 由 engine 在 `store_hook` 调用前填入本 turn 的必要 key。

验收：

- host 满 + disk 有位：快照必须 `spill` 而不是 `drop`；
- disk 也满：才 `drop`，并且 `Applied.snap_dropped` / metrics 完整；
- state 路径与 `Maintain::Pressure` 的 snap 票同序。

### 2.4 Pressure / idle：显式信号，不用固定水位

```rust
struct PressureSignals {
    pool_free_pages: usize,
    host_free_slots: usize,
    disk_free_slots: usize,
    driver_free_bytes: usize,
    reserve_bytes: usize,
}
```

触发分层：

| 信号 | 动作 |
|---|---|
| `pool_free < low` | `Maintain::Pressure`（KV + snap，先下沉后驱逐） |
| `host_free < low` | `Maintain::Pressure` 带 `Sink::Disk`；host-only 档则 drop |
| `driver_free < reserve` | 先 `drain_temp_pool` + `trim_pools`；仍不足才拒新 |
| idle | **默认不预搬**；显式 `IdlePolicy::Compact` 才按 host/disk 目标搬 |

水位用高/低双阈值 + 最小间隔，避免 `PoolExhausted` 前每拍 churn。
这一条同时回应 P1-3 / P2-4。

### 2.5 Reconciliation：三个来源必须能对上

`/debug/vram` 增一层：

```text
MemoryPlan:
  fixed: weights / graph / warmup / gdn / snap / draft
  pool:  tokens × bytes
  reserve
Ledger (DeviceHandle):
  live / free / pool cached / event open
Store:
  reserved / cache / session_held / host / disk / index
Driver:
  mem_get_info free / total
```

断言（不一致 = WARN + metrics，不静默）：

1. `Σ plan.fixed + plan.pool + reserve ≤ card_free_at_boot`
2. `Σ reservations + pool.free == pool.total`
3. `reserved + cache + session_held == occupied`
4. `leg.host + state.host + index.host == store.host_used`
5. `event.open_total` 与 leg/state 生命周期面一致（允许 known class 映射）

---

## 3. 迁移刀序（每刀独立可验）

| 刀 | 内容 | 门 |
|---|---|---|
| **V0** | `MemoryPlan` 纯结构 + 现有数值回填（不改行为） | 日志 components 和 == 现有 budget；`--check` 通过 |
| **V1** | `pool_cap` / `assemble_budget` / `BudgetInputs` 改吃同一 plan | `runtime_floor > 0`；固定池 `card_measured=true`；262k/3080 boot 不变 |
| **V2** | `Reservation/Grant`；`ensure_for_len` 收 grant；`StoreFacts` 只读 store ledger | 无 grant 分配 = 结构化拒；单测 + CPU mock |
| **V3** | 活跃共享前缀去重；`reserved+cache==occupied` 门 | 2 turn 共享前缀 + 第三 turn 准入 |
| **V4** | `StateAcquire::Slot` 走 `plan_capacity`，打通 disk sink | host 满/disk 有位 = spill；disk 满 = drop + metrics |
| **V5** | `PressureSignals` + 高/低水位；idle 默认关 | 长跑零 churn；PoolExhausted 前不抖 |
| **V6（可选）** | 权重/专家驻留（`BlockClass::Expert`） | 等 MoE 模型落地再开 |

**不建议**在 V0–V2 之前继续扩大 B 轴或加新的 tier 特性：否则预算与准入的
事实源会继续分裂，最后只能靠日志猜。

---

## 4. 待拍问题

1. **固定池 vs 动态池**：生产固定池是默认，`MemoryPlan` 是否仍保留
   `dynamic` 分支给 CPU mock / 未来弹性池？
2. **Reservation 归属**：Grant 由 store 发还是 engine 发？
   建议 store 发（账在 store），engine 只持 token + 设备句柄。
3. **共享块记账**：共享块归 `Shared` 桶，还是归“最后一个引用者”？
   建议 `Shared` 桶，报告显示引用数；不做按 turn 分摊。
4. **活跃快照保护**：只保护本 turn 最新边界，还是保护所有活跃 turn 的
   restore key？建议先保护“最新 + 正在回滚”的两类。
5. **Idle 预搬**：默认关，还是保留后台 compact 但显式配置？
6. **MoE / 权重分层**：现在只留 `BlockClass::Expert` 词汇，还是等模型？

---

## 5. 一句话结论

> A/B 轴与 3-hook 已经把“**搬什么、何时搬**”做出来了；
> 现在缺的是“**谁批准搬、搬完账是否闭合**”的单一回路。
> 先做 MemoryPlan + Grant + Reconciliation，再谈权重分层 / 专家驻留。
