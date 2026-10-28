# MemoryPlan V0/V1 + 显存调度第一批重构（2026-10-27）

> 分支：`feat/vram-plan`  
> 前序 review：`<worktree>/docs/devlog/memory/2026-10-27-vram-scheduling-review.md`  
> 本地交接：`roadmap.local/26-10-27-显存调度架构-设计交接.md`

## 0. 结论

本轮把 review 的 **P0-1（runtime_floor 恒 0 / 两套预算）**、
**P0-3（共享前缀双计）**、**P0-2（状态腿绕过三层策略）**、
**P0-4（Grant/Reservation）** 落成代码：

```text
MemoryPlan（boot 唯一预算事实源）
  → BudgetInputs 只做 plan 投影
  → Reservation/共享块去重（P0-3）
  → StateAcquire::Slot 走 plan_capacity（P0-2）
  → Grant/Reservation（P0-4）
  → /debug/vram 暴露 plan + Grant 面
```

P0-4 的 Grant 是 store 的单页账：准入批准、`ensure_for_len` 消费、
`Release` 归还；`sched_policy::StoreFacts.reserved_pages` 只读它。

## 1. V0：MemoryPlan 结构

新增 `crates/store/src/plan.rs`：

- `MemoryPlan { epoch, card, fixed[], pool, tiers, policy }`
- `MemComponent { id, bytes, source, tier, owner }`
- `ComponentId`：`Weights / GdnGrid / GdnSnapshots / SpecSnapshots /
  ReplaySnapshots / GraphSlab / WarmupMisc / DraftPool`
- `PlanSource`：`Measured / Derived / Configured / Conservative`
- `PlanCheck`：`fixed + pool + reserve` 对 `free_at_boot / total` 的闭合检查
- `MemoryPlan::to_budget_inputs()`：**唯一预算投影**

`StatePool::memory_components()` 在池分配后读真实账：

| 项 | 来源 |
|---|---|
| Weights | `live_before_pool − boot.live_at_new`（per-card） |
| GdnGrid | `gdn_row_bytes × gdn_slots`（实测） |
| GdnSnapshots / SpecSnapshots / ReplaySnapshots | 真实缓冲账（实测） |
| DraftPool | `draft_pool_bytes()`（MTP / DFlash2 ring，实测） |
| GraphSlab | 捕获后 `GraphFleet::slab_bytes()` 实测；boot 预算上界仍打印 `GRAPH_SLAB_BUDGET` |
| WarmupMisc | `WARMUP_MISC_BUDGET`（瞬态保守上界；不拿运行期残差冒充实测） |

主 KV 池单列在 `PoolPlan`，不混进 `fixed`。

## 2. V1：预算单一来源

`engine.rs::build_memory_plan()` 在池装配后编译 plan；`assemble_budget()`
改为 `plan.to_budget_inputs()`：

```text
runtime_floor = Σfixed − weights − gdn_grid − draft
              = 快照池 + graph_slab + warmup_misc
```

不再使用：

```text
live_now − live_loaded − pools_total
```

这条残差式在生产路径恒为 0，是 P0-1 的根因。

`Rel::attach_memory_plan()` 会把 plan 投影与 `Rel::open` 的预算入参逐字比较；
不一致 = `StoreError::Invariant`，不静默采信任何一方。

真机 3080 TP2 boot 证据（本地日志）：

```text
MemoryPlan: fixed(weights=10757.8MiB + gdn_grid=75.9 + gdn_snap=151.9
  + spec_snap=75.9 + replay_snap=0 + graph_slab=320
  + warmup_misc=138 + draft_pool=20.3)
  + pool=4096.0MiB(8192页×32tok×16384B) + reserve=512.0MiB
  = 16147.9MiB / free 19851.8MiB

准入预算: ... 运行底价 685.8MiB ... 卡级实测=true
```

`check_card().fits() == true`；`/debug/vram.plan` 已能逐项读出。

## 3. P0-3：共享前缀只计一次

`Admission` 的活跃 turn 现在可携带物理块槽集合：

```rust
ad.set_active_slots(turn, slots, block_bytes)
```

`reserved_bytes()` 取所有活跃 turn 槽号的**并集**：

```text
reserved = |∪ slots| × slot_bytes
```

旧口径 `Σ(session.block_table.len())` 会把共享前缀算 N 次。现在：

- 两个活跃 turn 共享前缀：`reserved + cache = occupied` 仍成立；
- 第三个同尺寸 turn 不因共享前缀被误拒；
- 释放共享方之一，并集仍覆盖另一 turn 的全部块。

`sync_pool_account()` 同时过滤 `HOST_PLACEHOLDER`，不把宿主占位符当设备槽。

门：`budget::tests::shared_prefix_is_counted_once`。

## 4. P0-2：状态腿磁盘可达

`store_state_slot()` 删除“只搬 host、失败即 drop”的独立旁路，改为：

```text
snap_pick_slot → Victim
  → pool_housekeeping(Need { slots: 0, snap_bytes: 一条快照, why: Grow })
  → plan_capacity 出 Host / Disk / Drop 票
  → apply_verdict
  → 重试取槽（最多 3 拍：Host→Disk 腾位 → GPU→Host → 取槽）
```

宿主满、盘有位时先 `spill` 宿主冷快照；设备快照仍走“GPU → Host”层级律。
只有 store 无票可出（host/disk 都关）才走最后兜底 drop，并打点。

门：`engine::tests::state_slot_spills_host_snapshot_to_disk_before_drop`
（CPU mock，host 1 条 / disk 1 条；断言三条旧快照都还在账、盘上 1 条）。

## 4.1 P0-4：Grant / Reservation

新增 `crates/store/src/reservations.rs`：

```text
ReservationId / ReservationKind / Grant / ReservationLedger
```

语义：

- `Grant.slots/bytes` = **准入面**批准总量（超卖档 = 工作集）；
- `Grant.alloc_slots/alloc_bytes` = **分配面**可消费总量（超卖档 = 整条 ctx 的块表上限）；
- `reserved_slots()` = Σ准入面总量（不是 remaining）—— 已兑现的槽变成活跃块，
  不能从承诺面减掉；
- `consume` 只减 remaining，`refund` 只用于分配失败回滚；
- `release_turn` 幂等。

接线：

```text
schedule().await
  decide Admit
    → StoreHook::capacity_reserve(...)
    → store Reserves grant
  start_one(turn, grant)
    → Rel::ensure_for_len(grant, table, target)
  ActiveTurn.grant
  complete/abandon/Release
    → release_turn_grants(turn)
```

`sched_policy` 删除影子 `admitted: turn → pages`；`StoreFacts` 新增
`in_flight_turns`，`reserved_pages` 来自 `ReservationLedger`。策略层只读账。

门：

- `store::reservations::tests::*`（额度、回滚、超卖双面、幂等）；
- `store::rel::tests::ensure_for_len_consumes_grant_and_release_returns_it`
- `store::rel::tests::ensure_for_len_refunds_grant_on_pool_exhaustion`
- `engine::tests::store_admission_gate::admission_issues_and_releases_grant`

## 4.2 P1-2：活跃快照保护面

快照面没有引用计数（不像 KV 由 `SlotPool.refs` 兜底），**只靠 `Protected.snaps` 排除**。
此前 `pool_housekeeping` 恒传 `Protected::default()`：Idle/Pressure 可能把活跃 turn 的
最新边界当成最冷条目搬走，`snap_pick_slot` 的池内 LRU 回退也能越过 store 序。

本轮把保护面的构造收进 engine 的 store_hook 层：

```text
ActiveTurn.last_snap = restore_key（turn 起）
                     = 新 capture key（capture 成功后）

snap_guard(extra, skip_turn)
  = 所有活跃 turn 的 last_snap
  + 本次操作点名的 key（如正在就位的 restore key）
  - skip_turn 的本 turn 旧边界（只给 capture：新边界会替代它）
```

接线：

- `pool_housekeeping(need, extra_snaps, skip_turn)` 统一构造 `Protected.snaps`；
- `Acquire::State::Slot`（capture）传 `skip_turn = 当前活跃 turn`，
  允许旧边界被新边界替代；其它活跃 turn 仍受保护；
- `Acquire::State::Resident`（回滚/就位）把 `key` 放入 `extra_snaps`，
  Host/Disk 回灌期间绝不把它当 victim；
- `store_state_slot` 的 `snap_victim_order` 与 `plan_capacity` 吃**同一份**
  protect；池内 LRU 兜底若选出受保护 key，直接结构化报错，不再静默 drop；
- `Acquire::Capacity` / `Acquire::Load` / `Maintain` 自动带上所有活跃 turn 的
  `last_snap`。

门：

- `running::snap_guard_tests::keeps_active_latest_and_restore_key_without_duplicates`
- `running::snap_guard_tests::capture_skips_only_its_own_previous_boundary`

## 4.3 P1-3：PressureSignals + Idle 默认关

原实现的两处固定水位：

```text
pressure_need: target = clamp(total/8, 1, 64)
idle_tick:     每 16 拍 fixed total/8（clamp 1..16）
```

现在改成显式信号 + 低/高双水位：

```rust
PressureSignals {
    pool_free_pages, pool_total_pages,
    host_kv_enabled, host_snap_enabled,
    host_kv_total_slots, host_snap_total_slots,
    host_kv_free_slots, host_snap_free_slots,
    disk_kv_free_slots, disk_snap_free_slots,
    driver_free_bytes,        // TP2 取两 rank min
    reserve_bytes,
}
```

判据：

```text
free >= low  ⇒ 不搬
free <  low  ⇒ 目标 = high − free（cap 见调用点）
```

- `pressure_low_pct = 12` / `pressure_high_pct = 25`（占主池页；`PressurePolicy`
  内部钳位，low ≥ 1、high ≥ low）。
- `PoolExhausted` 后的 `relieve_pressure` 先 drain/trim，再按信号算目标；
  `driver_free < reserve` 会单独 WARN（不把它误报成“池槽不足”）。
- **`runtime.idle_compact` 缺省 false**：空转只做池缓存回收，不再每 16 拍
  固定预搬；显式 true 才按同一套低/高水位 compact（上限 16 槽）。
- **HostPressure**：宿主 KV/快照腿低于**独立低水位**（
  `host_pressure_low_pct=12`，按该腿总槽）且对应盘腿有位时，先发
  `Maintain::HostPressure { kv_slots, snap_slots }`；目标补到**独立高水位**
  `host_pressure_high_pct=25`（不是只补回 low），形成滞回。
  单次目标 = `high − free`；`plan_capacity` 只出 `Sink::Disk`，跳过 GPU→Host。
- **每请求分腿配额**（2026-10-27 裁决）：
  `host_pressure_request_slots=16`（KV 宿主块/请求）+
  `host_pressure_request_snaps=1`（快照/请求）。
  8 并发最坏 = `8×16` KV 槽 + `8×1` 快照；不做 turn 累计账/全局窗口。
- **长链不超调**：HostPressure 的 Disk 票 `span` = 从 root 起的前缀块数；
  长链超过目标时只搬前缀（盘上仍是连续可重建前缀），不再整链越过 high。
- 空转 compact 的 HostPressure 有 `host_pressure_interval_ms=1000` 最小间隔；
  `PoolExhausted` 的应急收缩不节流（并会把节流时间戳刷新）。
- `Need` 增加 `host_kv_slots` / `host_snap_slots`（HostPressure 分腿），
  `Need::host(kv, snap)` 是唯一构造；普通 `Need::kv` 两字段为 0。
- `/debug/vram` 的压力日志带 `pool/host/disk/driver` 一行诊断。

门：

- `running::pressure_tests::low_watermark_triggers_and_high_is_the_target`
- `running::pressure_tests::small_pool_and_bad_percentages_are_clamped`
- `running::pressure_tests::driver_headroom_is_part_of_the_signal`
- `running::pressure_tests::host_pressure_uses_independent_low_watermark_and_disk_room`
  （low 触发 / high 目标 / 每腿独立）
- `running::pressure_tests::request_quota_clamps_kv_and_snapshot_independently`
- `housekeeping::tests::host_pressure_interval_gates_idle_compaction`
- `store::rel::tests::host_pressure_skips_gpu_to_host_and_emits_disk_only`
- `store::rel::tests::host_pressure_spills_only_the_prefix_needed_to_meet_quota`
- `store::rel::tests::host_pressure_limits_snapshot_tickets_independently`
- `store::rel::tests::host_pressure_keeps_kv_and_snapshot_goals_independent`
- `config::auto_pool_tests::idle_and_pressure_defaults_are_conservative`

## 4.4 P2-2：GraphSlab 捕获后实测回填

`GraphFleet` 新增：

```rust
pub(crate) fn slab_bytes(&self) -> usize
```

口径与 boot 的“slab 总账”一致：

```text
session + verify + fold + propose + dflash + dflash_select
```

`build_memory_plan(..., graph_slab_bytes)` 在 `into_running` 里读：

```text
graph_slab_bytes = max(rank0.graphs.slab_bytes(), rank1.graphs.slab_bytes())
```

然后：

- `MemoryPlan.fixed[GraphSlab].bytes = 实测 Σ`；
- `source = Measured`；
- boot 日志同时打印 **实测 / `GRAPH_SLAB_BUDGET` 保守上界**；
- `WarmupMisc` **仍保持 `Conservative`**：它是 warmup 瞬态上界，不是捕获后
  的常驻实测，不能拿 `live` 残差冒充。

门：

- `engine::tests::memory_plan_uses_measured_graph_slab_after_capture`
  （CPU mock 捕获后：GraphSlab = `Measured`，WarmupMisc = `Conservative`）

## 4.5 P1-5：GDN 行 owner

`StateStore` 增加 `row_owners: BTreeMap<row, Owner>`：

- `borrow_row(Owner::Turn(turn.id))`：借行时登记归属；
- `return_row(...)`：还行时清归属（重复归还仍拒）；
- `row_owner` / `row_owners()`：报告面读取。

`Rel::leg_accounting()` 不再把行统一记 `System`，而是按 `row_owners` 分组：

```text
Turn(t) + AssetKind::Row + n × row_bytes
```

只有异常的无主在借行才兜底 `System`。engine 的 `start_one` 已传
`Owner::Turn(turn.id)`。

门：

- `engine::tests::gdn_row_is_owned_by_the_active_turn`
- `store::rel::tests::report_surfaces_both_legs_and_faces`（行 owner 断言）

## 4.6 Load / Row / Snap / Temp Grant

五类票现在都在同一本 `ReservationLedger`：

| kind | 来源 | 计量 |
|---|---|---|
| `ActiveFloor` | 准入 | KV 槽（唯一进准入面） |
| `Load` | `store_load` 回灌块 | 块数 × `slot_bytes` |
| `Row` | `start_one` 借 GDN 行 | 1 × `row_bytes` |
| `Snap` | `store_state_slot` 取快照槽 | 1 × `snap_bytes` |
| `Temp` | `store_load` 周转/临时工作区 | 调用方给 slots/bytes |

关键纪律：

```text
Rel::reserved_slots() = reserved_slots_of(ActiveFloor)
```

Row/Snap/Load/Temp **不污染** `sched_policy::StoreFacts.reserved_pages`。
`StoreReport.grants` 新增按 kind 汇总的 `GrantReport`，`/debug/vram →
faces.grants` 输出。

门：

- `store::reservations::tests::kinds_are_accounted_separately`
- `store::rel::tests::grant_ledger_keeps_kinds_separate`
- `engine::tests::gdn_row_is_owned_by_the_active_turn`（Row Grant 同报）

## 5. 观测面

`/debug/vram` 新增 `plan`：

```json
{
  "plan": {
    "summary": "...",
    "card": { "total": ..., "free_at_boot": ..., "reserve": ... },
    "fixed": [{ "id": "weights", "source": "measured", ... }],
    "pool": { "pages": ..., "tokens": ..., "per_token_bytes": ... },
    "check": { "fits": true, "over_free_bytes": 0, "over_total_bytes": 0 }
  },
  "faces": {
    "reserved_slots": 0,
    "in_flight_turns": 0,
    "grants": [{ "kind": "active_floor", "grants": 0, "slots": 0, "bytes": 0, "turns": 0 }]
  }
}
```

## 6. 验证

已跑：

```text
cargo test -p owl-store --lib                 157 passed
cargo test -p owl-engine --lib                182 passed
cargo test -p owl-engine --lib snap_guard_tests       2 passed
cargo test -p owl-engine --lib pressure_tests         5 passed
cargo test -p owl-engine --lib host_pressure_interval_gates_idle_compaction 1 passed
OWL_CPU_MOCK=1 cargo test -p owl-engine --test store_admission_gate 7 passed
cargo test -p owl-shared --lib                125 passed
cargo check -p server                          PASS
```

`server --test config_check` 仍有 3 项既有 `runtime.max_seq` 期望值失败
（57344 vs 79872），与本次改动无关。

真机：

- P2-2 前：3080 TP2 `262k` boot 成功、`MemoryPlan` 闭合、`卡级实测=true`；
- P2-2 后尚未重启 8135；GraphSlab 实测值需在下一次 boot 日志复核。

## 7. 未完成 / 下一刀

1. `MemoryPlan.epoch` 目前恒 0；运行期重编译（若做）再递增。
2. **WarmupMisc**：保留保守上界；若要实测，需要单独的热身瞬态账，
   不能用捕获后的 live 残差替代。
3. **Grant 粒度**：Load/Snap/Temp 当前是 turn 生命周期票；若要按单次操作
   即时归还，需要再区分“操作票”和“turn 票”的生命周期。

## 8. 文件落点

```text
crates/store/src/plan.rs            新增 MemoryPlan
crates/store/src/lib.rs             导出
crates/store/src/budget.rs          BudgetInputs 投影来源;共享块并集
crates/store/src/rel.rs             attach_memory_plan / StoreReport.plan
crates/engine/src/engine.rs         build_memory_plan;assemble_budget 投影
crates/engine/src/state.rs          memory_components / 分项快照访问器
crates/engine/src/running.rs        snap_guard / PressureSignals / PressurePolicy / state slot
crates/engine/src/exec/spec.rs      capture 后更新 ActiveTurn.last_snap
crates/engine/src/turn.rs           ActiveTurn.last_snap
crates/engine/src/scheduler.rs      turn 起初始化 last_snap
crates/shared/src/config/mod.rs     runtime.idle_compact / pressure_low_pct / pressure_high_pct
profiles/full.toml                  三个新键的缺省与注释
crates/engine/src/tests.rs          状态腿 disk spill 门
apps/server/src/debug.rs            /debug/vram.plan
```
