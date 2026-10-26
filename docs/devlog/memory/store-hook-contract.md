# Store 钩子契约（现状审计 + 三钩子收敛设计）

> 2026-10-25。状态：**Phase 1 + Phase 2 + Phase 3 + Phase 4 已落地**
> （Phase 1 行为保持；Phase 2 纯分配；Phase 3 H3/H6 + 状态腿；
> Phase 4 账目/压力触发）。
> 目标：把 store 的“何时问、问什么、谁搬、搬完怎么记账”收成**一个契约**；
> 把当前散在 engine 的 19 个 store 调用点收编为 **3 个语义钩子**。
>
> 代码标记：**`FIXME(store-hook)` 已清零**。`coverage.rs` 的
> `plan_offload/plan_onload` 是显式 `diag-only(store-hook)`，不进生产链路。
>
> **类型/实现以 §7–§10 为准**；§3 的 `Acquire` 结构是草案。

---

## 0. 一句话

当前不是“钩子太多”，而是**同一件事有三套入口**：

1. `pool_housekeeping → Rel::plan_need → apply_verdict`（策略 + 三层搬运）；
2. `ensure_for_len`（同步槽兜底，绕过策略，可逐设备叶/丢宿主副本）；
3. `plan_onload` / `snap_onload` / `snap_offload`（直接搬运旁路）。

这三套入口共同覆盖 19 个生产点，导致状态机、metrics、账目、压力语义都分裂。
新版只保留 **3 个钩子：Acquire / Release / Maintain**。

---

## 1. 现状审计：19 个生产调用点（Phase 1 前）

> 本节是 Phase 1 前的审计底稿，保留用于解释“为什么要收”；
> 现役调用点见 §7.2。

口径：生产路径；不含测试、诊断探针（`coverage.rs` 的 plan 单独注明）。
“三层搬运”= KV/快照在 device / host / disk 之间移动；“槽兜底”= 只逐设备缓存叶，
不产生 host/disk 搬运。

### 1.1 正式 `Node`：4 类 / 6 个调用点

`Node` 定义在 `crates/engine/src/running.rs`，唯一入口 `RunningEngine::node()`。

| # | Node | 位置 | store 调用 | 搬运 |
|---|---|---|---|---|
| 1 | `Need`（Admit/Grow） | `pump()` | `step_need` + `admission_gate` → `pool_housekeeping` → `plan_need` → `apply_verdict` | 三层 |
| 2 | `Pressure` | `pump()` | 同上，`Need::kv(k, Pressure)` | 三层；`snap_bytes=0` |
| 3 | `Pressure` | `relieve_pressure()` | 同上，`PoolExhausted` 后的补救 | 三层；`snap_bytes=0` |
| 4 | `Idle` | `pump()` Idle 分支 | `pool_housekeeping`，带 `snap_bytes` | 三层 |
| 5 | `TurnEnd` | `exec/turn_end.rs` 弃置路径 | `n4_pool_sweep` / `n4_report_unreturned` / `n4_force_release` / `trim_pools` | 否；N4 归属/归还/trim |
| 6 | `TurnEnd` | `exec/turn_end.rs` 正常完成 | 同上 | 否 |

要点：

- `Need` 一个调用点承担 **H1 Admit + H2 Grow**；H1 还分裂为
  `scheduler` 的 `turns.admission.decide` 与 `admission_gate` 的 `Rel::check` 两层判决。
- `Pressure` 有两个调用点，且只带 KV 槽目标（`Need::kv`），**不搬快照**。
- `TurnEnd` 不是容量钩子；H3 的链登记在它之前由 `register_chain_owned` 完成。

### 1.2 旁路：13 个生产点

| # | 位置 | store 调用 | 类型 |
|---|---|---|---|
| 7 | `exec/phases.rs::execute_begin`（H4） | `pool_housekeeping` + `plan_onload` | host→device 回灌；绕过 `apply_verdict` |
| 8 | `scheduler.rs::decode_row_of` | `ensure_for_len` | 槽兜底 |
| 9 | `scheduler.rs::start_one` | `ensure_for_len` | 槽兜底 |
| 10 | `scheduler.rs::decode` 相位 | `ensure_for_len` | 槽兜底 |
| 11 | `exec/phases.rs::execute_prefill` | `ensure_for_len` | 槽兜底 |
| 12 | `exec/draft.rs` target 预增长 | `ensure_for_len` | 槽兜底 |
| 13 | `exec/draft.rs` target 跨页 | `ensure_for_len` | 槽兜底 |
| 14 | `exec/draft.rs` MTP propose 链 | `ensure_for_len` | 槽兜底 |
| 15 | `exec/draft.rs` MTP 覆盖 | `ensure_for_len` | 槽兜底 |
| 16 | `exec/draft.rs` MTP 批量行 | `ensure_for_len` | 槽兜底 |
| 17 | `exec/spec.rs` 快照受害者下沉 | `snap_offload` | 状态腿 host |
| 18 | `exec/spec.rs` 快照回灌 | `snap_onload` | 状态腿 device |
| 19 | `exec/spec.rs` 快照回灌（第二处） | `snap_onload` | 状态腿 device |

另有**生命周期旁路**（不计入 19，但也绕过了统一 hook）：

- `exec/turn_end.rs::register_chain_owned`（H3）：更新 prefix tree，可能逐出缓存条目；
- `exec/turn_end.rs::release_turn_table` / `burn_ephemeral`（H6）：直接 decref，无 store event；
- `exec/coverage.rs` 的 `plan_offload_blocks` / `plan_onload`：诊断/选择层，非生产主路径。

### 1.3 调用图

```text
pump()
├─ step_need()                         # H1/H2 需求（Admit/Grow）
├─ admission_gate()                    # H1: Rel::check
├─ node(Node::Need)                    # 正式 ①
│  └─ pool_housekeeping
│     ├─ Rel::plan_need
│     └─ apply_verdict                 # host/disk/drop + onload
├─ pressure_probe()
│  └─ node(Node::Pressure)             # 正式 ②
├─ schedule()                          # 同步、零 await
│  ├─ start_one → ensure_for_len       # 旁路 ①
│  ├─ decode_row_of → ensure_for_len   # 旁路 ②
│  └─ decode → ensure_for_len          # 旁路 ③
├─ Idle → node(Node::Idle)             # 正式 ③
└─ execute_begin
   ├─ pool_housekeeping                # H4 腾槽
   └─ plan_onload                      # H4 回灌（旁路 ④）

turn_end
├─ register_chain_owned                # H3（不是容量钩子）
├─ node(Node::TurnEnd)                 # 正式 ④（N4）
└─ release_turn_table                  # H6（无钩子）

spec
├─ snap_offload                        # 旁路 ⑤
└─ snap_onload ×2                      # 旁路 ⑥⑦
```

---

## 2. 为什么现在的状态机难维护

### 2.1 文档 H1–H7 与代码不是一一对应

| 文档钩子 | 代码现状 | 问题 |
|---|---|---|
| H1 Admit | `step_need` + `admission_gate` + `Node::Need` | 与 H2 共用 Node；admission 有两套判决 |
| H2 Grow | `Node::Need` + 9 个 `ensure_for_len` | 一半走策略，一半走同步兜底 |
| H3 TurnEnd | `register_chain_owned` + `Node::TurnEnd` | 链登记不在 Node；Node 只做 N4 |
| H4 PrefixHit | `execute_begin` 直接 `pool_housekeeping` + `plan_onload` | 不是 Node；回灌绕过 `apply_verdict` |
| H5 Idle | `Node::Idle` | 唯一完整钩子 |
| H6 Release | `release_table` / `burn_ephemeral` | 没有 store event |
| H7 StateLeg | 仅 `Node::Idle` 有 `snap_bytes`；另有 `snap_*` 直通 | H1/H2/N6 的 `snap_bytes=0` |

### 2.2 三套入口

- **策略入口**：`plan_need` / `apply_verdict`（唯一执行臂 `pool_housekeeping`）。
- **槽兜底入口**：`ensure_for_len`，只逐设备叶，必要时 `drop_host_evicted`；
  不走 `plan_capacity` 的候选表/策略序，`ops.ensure_evicted` 是错位计数器。
- **直通入口**：H4 `plan_onload`、spec `snap_*`、coverage `plan_*`。

### 2.3 触发与需求错位

- N6 由驱动 `free < max(2*reserve, 1GiB)` 触发，但需求换算成“KV 槽”；
  固定池下沉不增加驱动 free，且 `Need::kv` 不搬快照 ⇒ N6 既救不了驱动压力，
  也救不了状态腿压力。
- `Node::Need` 同时是“需求”和“触发”；H1/H2/H3 的时点被压进一个 Node。
- H3/H6 是生命周期事件，不是容量需求；把它们塞进容量状态机只会让状态更多。

---

## 3. 新版：3 个钩子

### 3.1 语义

```rust
pub enum StoreHook {
    /// 前场：任何需要设备资源/回灌的动作，先问 store。
    Acquire(Acquire),
    /// 生命周期：turn 结束 / session 关闭，资产从 Active 转 Cache/Free。
    Release(Release),
    /// 后场：空转/压力下的分层维护，目标量而不是缺口。
    Maintain(Maintain),
}
```

只有一个入口：

```rust
impl RunningEngine {
    async fn store_hook(&mut self, hook: StoreHook) -> Result<StoreOutcome>;
}
```

内部固定两相：

```text
plan(hook)  → Verdict   // 纯 host，零 I/O
apply(verdict) → Applied // 段边界，async，唯一执行臂
```

### 3.2 `Acquire`：前场需求 + 回灌

```rust
pub struct Acquire {
    /// 为什么问：准入 / 增长 / 前缀命中 / 状态腿
    pub kind: AcquireKind, // Admit | Grow | Load | State
    /// 设备槽缺口（不是总量）
    pub slots: usize,
    /// 状态腿设备字节缺口
    pub snap_bytes: usize,
    /// 需要 host/disk → device 的链（H4 回灌）
    pub load: Vec<ChainHash>,
    /// 不可动面（活跃 turn / 点名快照）
    pub protect: Protected,
}
```

`Acquire` 的返回不是裸 `()`，而是：

```rust
pub struct StoreOutcome {
    pub verdict: Verdict,   // out / in_ / deny
    pub applied: Applied,   // 实际搬了多少
    pub granted: Grant,     // 已确保可用的槽/字节（或 AdmissionVerdict）
}
```

纪律：

- **先 Acquire，后分配**。`ensure_for_len` 只做“纯分配”，不再驱逐；
  不够就是 `Acquire` 没给够，属于引擎错，不是 store 的兜底路径。
- H4 的回灌进入 `Acquire.load`，由 `apply_verdict.in_` 统一执行；
  不再直接 `plan_onload`。
- 状态腿的 `snap_bytes` 是所有 Acquire 的一等字段，不再只在 Idle 里给。
- `AcquireKind::Admit` 可以返回 `Admit/EvictThen/Queue/Reject` 的 store 判决；
  与 `scheduler` 的策略判决合并成**一个** admission 出口（见 §5 待决）。

### 3.3 `Release`：生命周期

```rust
pub struct Release {
    pub kind: ReleaseKind, // TurnEnd | SessionClose | Abort
    pub turn: u64,
    pub session: u64,
    /// TurnEnd 时带上链资产：tokens / block_table / owner
    pub chain: Option<ChainAssets>,
}
```

职责：

- H3：`register_chain_owned` 的链登记、归属、缓存树更新；
- H6：`release_table` / `burn_ephemeral` 的引用归零；
- TurnEnd 的 N4 清点/未归还报告/force release/trim 仍归 store 的 release 执行臂；
- **不产生容量 evict**；只改变“谁可动/谁还在”的账。

`Release` 是同步记账 + 可选异步 trim；如果实现上需要 face，仍走 `store_hook`。

### 3.4 `Maintain`：后场

```rust
pub struct Maintain {
    pub reason: MaintainReason, // Idle | Pressure
    /// 目标腾出量（不是缺口）
    pub target_slots: usize,
    pub target_snap_bytes: usize,
    /// Pressure 用：硬余量判据（free − reserve），不再用 1 GiB 固定目标
    pub hard_headroom: Option<u64>,
}
```

职责：

- H5 Idle：冷资产下沉/清池；
- N6 Pressure：只在**硬余量真的不足**时触发；目标包含 KV + 快照；
- 固定池档如果“腾槽不增加驱动 free”，Pressure 只用于池内槽周转，
  不再伪装成驱动压力救援。

### 3.5 三钩子覆盖旧 19 点

| 旧点 | 新归属 |
|---|---|
| Node::Need（Admit/Grow） | `Acquire`（Admit/Grow） |
| Node::Pressure ×2 | `Maintain(Pressure)` |
| Node::Idle | `Maintain(Idle)` |
| Node::TurnEnd ×2 | `Release(TurnEnd/Abort)` |
| H4 pool_housekeeping + plan_onload | `Acquire(Load)` |
| scheduler 3 × ensure_for_len | `Acquire(Grow)` 预留后，`ensure_for_len` 纯分配 |
| prefill 1 × ensure_for_len | 同上 |
| draft 5 × ensure_for_len | `Acquire(Grow)`（target/MTP 目标并入 step_need 或 draft 前 Acquire） |
| spec snap_offload/onload ×3 | `Acquire(State)` / `Maintain(State)` |
| register_chain / release_table | `Release` |
| coverage plan_*（诊断） | 保持诊断，不进生产 `store_hook` |

收敛后：

- **语义钩子：3 个**；
- **生产调用点**：目标 ≤ 8 个（Acquire 3–4 + Release 2–3 + Maintain 2）；
- `plan_need` / `apply_verdict` / `plan_onload` / `snap_*` 只允许出现在
  `store_hook` 内部（grep 门）。

---

## 4. 迁移顺序（先契约，后行为）

### Phase 0：只标记，不改行为（本文件配套）

- 在所有 19 个生产点加 `FIXME(store-hook)`，写明当前归属与新钩子；
- 在 `Node` / `pool_housekeeping` / `ensure_for_len` / `pressure_probe` 加结构性 FIXME；
- 不改变任何调用顺序。

### Phase 1：引入 `StoreHook` 壳 —— ✅ 已完成（2026-10-25）

- 新增 `StoreHook` / `Acquire` / `Release` / `Maintain` / `StoreOutcome` 类型
  （`crates/engine/src/store_hook.rs`）；
- `store_hook` 转发到现有 `pool_housekeeping` / `plan_onload` / N4 收尾，
  行为不变；
- 正式 Node 与 H4 已改为 `store_hook`；旧 `Node` **直接删除**（没有保留
  兼容壳，避免双入口）。

实施细节与偏差见 §7。

### Phase 2：消灭同步驱逐 —— ✅ 已完成（2026-10-25）

- `KvStore::ensure_for_len` / `BlockLeg::ensure_for_len` 改为**纯分配**；
  不够 = 响亮 `Exhausted`（带「Acquire(Grow) 预留不足」文案）；
- `step_need` 的生成相位改用 `generation_target`，覆盖 verify + 轮末
  propose：DFlash2 `pos+2d+2` / MTP `pos+2d` / 裸 decode `pos+d+1`；
- MTP 腿 `cache=Off`/无宿主，本来就不产生同步驱逐；其 3 个
  `ensure_for_len` 已是纯分配，长度由 target 预留配对覆盖；
- 门：`ops.ensure_evicted == 0`（现在是回归 canary；`register_chain_offloaded`
  的树容量驱逐改记 `tree_evicted`）。

实施细节与实测见 §8。

### Phase 3：收编生命周期与状态腿 —— ✅ 已完成（2026-10-25）

- H3 链登记/盘索引 + H6 表释放 + GDN 行归还 + 归属下行 + N4 全部进入
  `StoreHook::Release(TurnEnd)`；弃置走 `Release(Abort)`（H6/N4 同款，
  链不发布）；
- 状态腿进入 `Acquire(State)`：
  - `StateAcquire::Slot`：拍摄选槽（受害者沉宿主/丢弃）；
  - `StateAcquire::Resident`：Gpu/Host/Disk 就位；
- `spec.rs` 的 `snap_offload/onload` 只保留执行定义，生产直呼清零；
- 收尾 grep 门（coverage 诊断例外）。

实施细节与实测见 §9。

### Phase 4：修触发与账目 —— ✅ 已完成（2026-10-25）

- **删除 pump 的 N6**（驱动 free < 1GiB 触发）；`Maintain::Pressure` 只在
  `PoolExhausted` 后的 `relieve_pressure` 使用，判据改成**池自由槽**，
  并先 `trim_pools` 真还驱动；
- `FixedAfter` 补 `spec_snap` / `replay_snap`（各 = 一行 GDN 状态）；
  `snapshot_pool_bytes` 也补 replay；
- Admission **保留两层**（批量策略 vs VRAM 地板），理由见 §5 Q1；
- metrics label 用纯单测钉住；e2e 计数门未强制（全局 metrics 易串扰）。

实施细节见 §10。

---

## 5. 待决问题

1. **Admission 两套判决**：`turns.admission.decide`（策略）与 `Rel::check`
   （store 地板）是否合并为 `Acquire(Admit)` 一个出口？
   **状态：已评审，保留两层**（Phase 4）：前者是批量状态机（会话守卫/槽位/
   生成预算/退避），后者是单 turn VRAM 地板（dense floor/EvictThen）；
   两者共享 store facts，但合并会把预算内部暴露给策略状态机。
2. **Draft/MTP 增长**：并入 `step_need` 的上界，还是在 draft 前单独
   `Acquire(Grow)`？前者简单，后者精确。
   **状态：已完成（Phase 2）**；采用前者：`step_need::generation_target`
   用一个 pump 步的最远位预留，draft/MTP 的 `ensure_for_len` 只纯分配。
   这样不在 draft 内新增钩子，避免同一轮两次问 store。
3. **StateLeg 是否独立**：作为 `Acquire.snap_bytes` 的一维，还是单独
   `Acquire(State)` / `Maintain(State)`？建议先作为维度，不增加第 4 钩子。
   **状态：已完成（Phase 3）**；采用 `Acquire::State(StateAcquire)`，
   不增加第 4 个 top-level 钩子。`Maintain::Pressure` 的 `snap_bytes`
   在 Phase 4 已接到 `relieve_pressure`（宿主有位时带一条快照目标）。
4. **Pressure 是否保留**：固定池档如果“腾槽不增驱动 free”，是否只保留
   `Maintain(Idle)` + `Acquire` 的 Deny，删除 N6？
   **状态：已关闭（Phase 4）**；删除 pump 的 N6 驱动-free 触发；
   `Maintain::Pressure` 只在 `PoolExhausted` 后按池自由槽收缩。
5. **TurnEnd 的 trim**：`trim_pools` 是 release 的一部分，还是单独的
   `Maintain(Trim)`？建议归 Release，避免第 4 钩子。
   **状态：已归 `Release`**（Phase 1 的 N4 实现内）。

---

## 6. 验收门

- **grep 门**：`plan_need` / `apply_verdict` / `plan_onload` / `snap_onload` /
  `snap_offload` 只出现在 `store_hook` 实现内；engine 其他位置零直呼。
  **Phase 4 通过（生产面）**：`plan_need` / `apply_verdict` 只在
  `store_hook` 家族；`plan_onload` 仅剩 `coverage.rs` 的 `diag-only`
  诊断；`snap_onload/offload` 的调用只在 `running.rs` 的 store_hook 内
  （`spec.rs` 仅保留执行定义）。`FIXME(store-hook)` = 0。
- **无同步驱逐门**：生产 e2e 的 `ops.ensure_evicted == 0`；
  非零即有人把逐缓存重新塞回分配路径。**Phase 2 通过**：
  纯分配实现 + store/engine 单测 + CPU mock + DFlash2/MTP 恒等门。
- **钩子覆盖门**：每个 e2e 场景断言实际发生的
  `store.hook.{acquire,release,maintain}` 计数与预期一致。
  **Phase 1 已换 metrics 名**：`store.node.*` → `store.hook.*`；
  Phase 4 用 `labels_are_stable` 钉住全部 label。e2e 计数断言未强制
  （metrics 全局，多场景串扰；需要时单测里串行读）。
- **状态机门**：`StoreHook` 只有 3 个变体；新增场景只能加 `kind`，
  不得新增第 4 个 top-level 钩子（除非另行评审）。**Phase 1 通过**。

---

## 7. Phase 1 实施记录（2026-10-25）

### 7.1 实际类型

```rust
// crates/engine/src/store_hook.rs
pub(crate) enum StoreHook {
    Acquire(Acquire),
    Release(Release),
    Maintain(Maintain),
}

pub(crate) enum Acquire {
    Capacity(Need),          // kind 由 Need.why 表达：Admit / Grow
    Load(LoadRequest),       // session + Vec<(块表下标, ChainHash)>
}

pub(crate) enum Release { TurnEnd(u64), Abort(u64) }

pub(crate) enum Maintain {
    Idle(usize),
    Pressure { slots: usize, snap_bytes: usize },
}

pub(crate) struct StoreOutcome {
    pub applied: Applied,       // 容量/维护的实际搬运计数
    pub loaded_blocks: usize,   // Load 实际回灌块数
}
```

与草案的偏差：

- 不单独设 `AcquireKind`：`Need.why` 已经是 kind 的唯一真源，避免两个字段漂移；
  `Acquire::Capacity` 收到 `Idle/Pressure` 时**响亮拒**（`EngineInvariant`）。
- `Acquire::Load` 用 `LoadRequest` 而不是把 `slots/load/protect` 混在一个结构里：
  当前只有一种回灌载荷，字段更少、非法状态更少。
- `Maintain::Pressure` 保留 `snap_bytes` 字段（测试与未来 N6 两腿目标用）；
  当前生产压力路径仍传 0。

### 7.2 已迁移调用点

| 旧点 | 新调用 | 文件 |
|---|---|---|
| `Node::Need`（pump H1/H2） | `StoreHook::capacity(need)` | `running.rs` |
| `Node::Pressure`（pump N6 / relieve） | `StoreHook::pressure(k)` | `running.rs` |
| `Node::Idle`（pump H5） | `StoreHook::idle(k)` | `running.rs` |
| `Node::TurnEnd`（complete/abandon） | `StoreHook::turn_end / abort` | `turn_end.rs` |
| H4 `pool_housekeeping + plan_onload` | `StoreHook::load(session, blocks)` | `phases.rs` → `running.rs` |

`Node` enum 已删除，`store.node.*` metrics 已改名为
`store.hook.{acquire,release,maintain}.*`。

### 7.3 仍未收编（Phase 1 时点；后由 Phase 4 关闭）

- Phase 4 的 N6 / `FixedAfter` / admission 已在 §10 处理。

> Phase 2 的 9 个 `ensure_for_len` 已收口；其中 3 个 MTP 点因腿
> `cache=Off` 本来就不产生同步驱逐，未新增钩子。

### 7.4 验证

- `cargo check -p owl-engine --tests`：通过（仅存量 `SnapDiskOp::Drop` dead-code 警告）。
- `cargo check -p server`：通过。
- `cargo test -p owl-engine --lib -- --test-threads=1`：**157 passed / 0 failed**。
- `OWL_CPU_MOCK=1 cargo test -p owl-engine --test store_admission_gate -- --test-threads=1`：
  **6 passed / 0 failed**（H1 先腾后进、池墙拒、无兜底驱逐）。
- `OWL_CPU_MOCK=1 OWL_OFFLOAD_PROBE=1 OWL_E2E_ONLY=offload_probe \
   cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1`：
  **1 passed**；轮 4 真实走 H4：`回灌 2 块/1 命令 · 兜底驱逐 0`。
- 测试夹具修正：`store_admission_gate::non_strict_dense_floor_is_allowed...`
  显式给 `pool_tokens = 16 × 32`（与它自己的“本档 16 页”注释一致；
  自动池推导是另一条线，不在本门范围）。
- 行为保持：H4 的回灌顺序、`note_stage_in` 计时起点、N4 的 trim/expire 时序、
  Idle 的 snap 目标计算均逐段保留。唯一可见变化是 metrics 名与
  `StoreHook` 单入口。

---

## 8. Phase 2 实施记录（2026-10-25）

### 8.1 纯分配

- `KvStore::ensure_for_len`：只做 `need = ceil(target/page)` 与 free 槽检查；
  不再调用 `tree.evict_device`、不再 `drop_host_evicted`，返回 `StoreResult<()>`。
- `BlockLeg::ensure_for_len`：只转发纯分配 + 链名错误映射；
  `ops.ensure_evicted` 不再有任何自增点（回归 canary，恒 0）。
- 错误文案：`KV 块池纯分配不足…Acquire(Grow) 预留不足（同步驱逐已删除…）`。
- `register_chain_offloaded` 的 `cache_cap` 树驱逐改记 `LegOps::tree_evicted`，
  不再污染 `ensure_evicted`。

### 8.2 生成目标上界

`RunningEngine::generation_target(pos)`（`running.rs`）：

| 档位 | 公式 | 来源 |
|---|---|---|
| 裸 decode / Dumb | `pos + d + 1` | verify 行 |
| MTP | `pos + 2d` | verify + 轮末 propose `pos+m+d`（m≤d） |
| DFlash2 | `pos + 2d + 2` | propose `fp+block`，`fp=pos+m+1`、`block=d+1`、m≤d |

`step_need` 的生成相位直接用该上界；`schedule`/`prefill`/draft/MTP 的
`ensure_for_len` 都只消费已预留的 free 槽。

MTP 腿是独立页池、`cache=Off`、`host_bytes=0`：其 `ensure_for_len` 从
Phase 2 前就不会产生同步驱逐（树无可逐候选）。Phase 2 只验证其长度
不超过 target 的配对上界，**不新增 draft 内钩子**（避免同一轮二次问 store）。

### 8.3 9 个生产点归属

| 点 | 归属 | 处理 |
|---|---|---|
| scheduler `decode_row_of` | target | 纯分配；`generation_target` 覆盖 |
| scheduler `start_one` | target | 纯分配；H1/H2 预留覆盖 |
| scheduler `decode` 相位 | target | 纯分配；`pos+spec+1` ≤ `generation_target` |
| prefill `execute_prefill` | target | 纯分配；`fed+chunk` 预留覆盖 |
| draft DFlash2 graph round | target | 纯分配；`pos+2d+2` 覆盖 |
| draft DFlash2 eager propose | target | 纯分配；`pos+2d+2` 覆盖 |
| draft MTP `mtp_chain_bt` | MTP | 本来就是纯分配（cache=Off）；长度配对覆盖 |
| draft MTP `propose_graph_step` | MTP | 同上 |
| draft MTP `prefill_extend` | MTP | 同上；`base+t_rows` 与 prefill 目标同源 |

### 8.4 验证（本工作树）

- `cargo test -p owl-store --lib -- --test-threads=1`：**136 passed / 0 failed**。
- `cargo test -p owl-engine --lib -- --test-threads=1`：**158 passed / 0 failed**
  （新增 `generation_target` 公式钉子）。
- `cargo check -p server`：通过。
- `OWL_CPU_MOCK=1 cargo test -p owl-engine --test store_admission_gate -- --test-threads=1`：
  **6 passed / 0 failed**。
- `OWL_CPU_MOCK=1 OWL_OFFLOAD_PROBE=1 OWL_E2E_ONLY=offload_probe \
   cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1`：
  **1 passed**；`回灌 2 块/1 命令 · 同步驱逐 canary 0`。
- `OWL_TEST_DEVICE=42:00.0 OWL_E2E_ONLY=dflash2 \
   cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1`：
  **1 passed**；DFlash2 depth=7，15 轮，AL=1.40，greedy spec ≡ no-spec 逐位一致；
  并断言 `ops.ensure_evicted == 0`。
- `OWL_TEST_DEVICE=42:00.0 OWL_E2E_ONLY=spec_identity_gate_27b \
   cargo test -p owl-engine --features e2e --test engine_e2e -- --test-threads=1`：
  **1 passed**；MTP depth=3，13 轮 / sum_m=23，greedy spec ≡ no-spec 逐位一致；
  并断言 `ops.ensure_evicted == 0`。
- 测试改造：`kv/mod.rs` 的 4 个“靠 `ensure_for_len` 逐缓存”用例改为显式
  `evict_cached` / `evict_device_cached`（测试专用），保留原判据；
  `leg.rs` 的关宿主压力用例改为显式 `evict_cached(1)`。

### 8.5 剩余（Phase 2 时点）

- Phase 4 已在 §10 完成。

---

## 9. Phase 3 实施记录（2026-10-25）

### 9.1 类型

```rust
// crates/engine/src/store_hook.rs
pub(crate) enum Acquire {
    Capacity(Need),
    Load(LoadRequest),
    State(StateAcquire),
}

pub(crate) enum StateAcquire {
    Slot { key: SnapKey },      // 拍摄选槽（受害者沉宿主/丢弃）
    Resident { key: SnapKey },  // Gpu/Host/Disk 就位
}

pub(crate) struct Release {
    pub kind: ReleaseKind,      // TurnEnd | Abort
    pub turn: u64,
    pub session: u64,
    pub gdn_slot: usize,
    pub ephemeral: bool,
}

pub(crate) struct StoreOutcome {
    pub applied: Applied,
    pub loaded_blocks: usize,
    pub state_slot: Option<u32>, // State 就位/选槽结果
}
```

### 9.2 H3/H6 收编

`StoreHook::Release` 现在按原 `complete`/`abandon_active` 的顺序执行：

```text
TurnEnd:
  H3  register_chain_owned → write_chain_index → index_probe 排空
  H6  burn_ephemeral / release_turn_table
  GDN return_row
  set_ctx_after_turn
  N4  pool_sweep → report_unreturned → force_release → trim_pools
Abort:
  跳过 H3
  H6  burn_ephemeral / release_turn_table（失败只 WARN，继续收尾）
  GDN return_row（失败只 WARN）
  set_ctx_after_turn
  N4（失败只 WARN，跳过 trim）
```

`complete()` 只保留 `commit_turn` / `set_img_spans`，然后一次
`Release(TurnEnd)`；`abandon_active()` 只保留墙钟 EWMA，然后一次
`Release(Abort)`。H3/H6 的 FIXME 已清零。

### 9.3 状态腿收编

- `spec::snap_ensure_slot` / `snap_ensure_resident` 现在都是
  `store_hook` 的**薄包装**：只负责取 `StoreOutcome.state_slot`；
- `snap_offload` / `snap_onload` 仍是执行定义（在 `spec.rs`），但生产直呼
  只在 `running.rs` 的 `store_state_slot` / `store_state_resident` 内；
- `capture_gdn_snap` 的「选槽」也走 `StateAcquire::Slot`，因此实际收编
  的是 **4 个**状态点（原审计的 3 个 `snap_*` + capture 的选槽）。

### 9.4 验证

- `cargo test -p owl-engine --lib -- --test-threads=1`：**159 passed / 0 failed**
  （新增 `state_hook_slot_offload_and_resident_roundtrip`）。
- `OWL_CPU_MOCK=1 cargo test -p owl-engine --test store_admission_gate`：
  **6 passed / 0 failed**。
- CPU offload probe：**1 passed**，H4 回灌 2 块/1 命令，canary 0。
- DFlash2 / MTP 27B 恒等门：各 **1 passed**，canary 0。
- `cargo check -p server`：通过。

**已知预存失败（与本刀无关）**：`cpu_mock_state_leg_probe` 在
`snap_max=1 + churn` 断言处失败（宿主 0 条）。在**主工作树基线**
（Phase 3 前，`dc5194b3`）复跑同样失败，故判定为存量测试/配置漂移，
不是状态钩子回归。本刀新增的 `state_hook_slot_offload_and_resident_roundtrip`
直接覆盖了 `StateAcquire::Slot` + `Resident` 两条新路径。

### 9.5 剩余（Phase 3 时点）

- Phase 4 已在 §10 完成。

---

## 10. Phase 4 实施记录（2026-10-25）

### 10.1 `FixedAfter` 补账

`state.rs::FixedAfter` 新增两项：

```text
spec_snap   = spec_mode != Off && paged ? n_gdn × gdn_slot_elems × 4 : 0
replay_snap = query_replay && paged     ? n_gdn × gdn_slot_elems × 4 : 0
```

两项各自等于**一行 GDN 状态**（27B ≈ 151.9 MiB）。此前：

- `FixedAfter` 只算 `gdn_snap`（前缀快照池），漏掉 spec/replay；
- `snapshot_pool_bytes` 只算 `spec_snap`，漏掉 replay。

后果：`pool_cap` / `[ctx-max]` 虚高约 4,860 tok/份（27B），3090 档实际
ctx 上限被高估。现在 `pool_cap` 与 `snapshot_pool_bytes` 都按真分配扣账。

`fixed_after_matches_production_and_keeps_historical_total` 增加断言：
`spec_snap == gdn_snap`、`replay_snap == gdn_snap`、total 增量 = 2×一行。

### 10.2 N6 删除与 `Maintain::Pressure` 重定义

- **删除 pump 的 N6**：不再按 `free < max(2×reserve, 1GiB)` 每拍触发；
  固定池下沉不增加驱动 free，原逻辑只制造 churn。
- `Maintain::Pressure` 只在 `PoolExhausted` 后的 `relieve_pressure` 使用：
  1. 复用池排空（`drain_temp_pool`，真还驱动）；
  2. `trim_pools` 把已 free 缓存页还驱动；
  3. 按**池自由槽**算目标：`target = clamp(total/8, 1, 64)`，
     `slots = target − free`；宿主有位且设备上有快照时带一条 `snap_bytes`；
  4. 一次 `Maintain::Pressure { slots, snap_bytes }`（两腿）。
- `StoreHook::pressure_with_snap` 是唯一构造；旧 `pressure(slots)` 删除
  （pump 不再用，避免死代码）。
- server actor 的注释已同步为“池排空 + trim + Maintain(Pressure)”。

### 10.3 Admission 两层判决（评审结论：保留）

- `scheduler::admission.decide` = **批量准入策略**：同会话守卫、槽位上限、
  生成预算、newcomer reserve、退避；
- `admission_gate` 的 `Rel::check` = **单 turn VRAM 地板**：dense floor /
  `EvictThen` / `Queue` / `Reject`；
- 两者共享 `StoreFacts`，但判据不同；合并会把 `Budget/Admission` 内部
  暴露给策略状态机，收益低、回归面大 ⇒ **不合并**，只在注释中写明分工。
- 原 H1 的两个 `FIXME(store-hook)` 已删除；`FIXME(store-hook)` 全局归零。

### 10.4 metrics

- `store_hook::label_tests::labels_are_stable` 钉住全部 9 个 label：
  `acquire.admit/grow/load/state.slot/state.resident`、
  `release.turn_end/abort`、`maintain.idle/pressure`。
- e2e 的“每个场景断言计数”未强制：metrics 是进程全局，并行/多场景会串扰；
  需要时在单测里串行读。当前以 label 稳定 + 单 hook 入口为门。

### 10.5 验证

- `cargo test -p owl-engine --lib -- --test-threads=1`：**160 passed / 0 failed**
  （新增 `labels_are_stable`）。
- `OWL_CPU_MOCK=1 cargo test -p owl-engine --test store_admission_gate`：
  **6 passed / 0 failed**。
- CPU offload probe：**1 passed**，H4 回灌 2 块/1 命令，canary 0。
- DFlash2 / MTP 27B 恒等门：各 **1 passed**，canary 0。
- `cargo check -p server`：通过。
- 预存 `cpu_mock_state_leg_probe` 失败仍与本刀无关（主树基线同样失败，
  见 §9.4）。

### 10.6 剩余

- 无阻塞项；`FIXME(store-hook)` = 0。
- 可选后续：e2e `store.hook` 计数断言（串行 metrics 场景）、
  `cpu_mock_state_leg_probe` 存量修复独立案。

---

## 11. 2026-10-26 接管收尾：rebase 到 main

Phase 1–4 的 WIP 原先停在 `dc5194b3`（落后 main 12 个提交）。本轮已 rebase 到
`a23eae04`，只解两处冲突：

- `running.rs`：保留 main 的 `NeedWhy::Idle` 守卫，同时按 Phase 4 删除 pump N6；
- `docs/README.md`：同时保留 leave-gap 交接条目与本文“Phase 1–4 complete”。

rebase 后复验（2026-10-26）：

```text
cargo test -p owl-engine --lib -- --test-threads=1     168 passed / 0 failed
cargo test -p owl-store  --lib -- --test-threads=1     136 passed / 0 failed
OWL_CPU_MOCK=1 cargo test -p owl-engine --test store_admission_gate
                                                        6 passed / 0 failed
OWL_CPU_MOCK=1 OWL_OFFLOAD_PROBE=1 OWL_E2E_ONLY=cpu_mock_offload_probe
                                                        1 passed（H4 回灌 2 块/1 命令）
OWL_E2E_ONLY=gpu_image_evict_gate                       1 passed
OWL_E2E_ONLY=gpu_image_evict_spec_identity_gate         1 passed
OWL_E2E_ONLY=gpu_spec_identity_gate                     1 passed（DFlash2/0.8B）
cargo check -p server                                   passed
```

**未复验**：27B DFlash2 / MTP 恒等门。当前 3090 卡被其他进程占用（约
13.7 GiB/卡），无法给 27B 双卡/单卡测试留出足够余量；Phase 4 时点曾各
1 passed。待卡空闲后补跑。

`FIXME(store-hook)` 仍为 0；生产面直呼 `plan_need` / `plan_onload` / `snap_*`
已收口到 `store_hook`，`coverage.rs` 的 `plan_*` 保持显式 diag-only。
