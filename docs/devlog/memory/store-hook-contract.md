# Store 钩子契约（现状审计 + 三钩子收敛设计）

> 2026-10-25。状态：**Phase 1 已落地**（行为保持；旧 `Node` 四变体已删除）。
> 目标：把 store 的“何时问、问什么、谁搬、搬完怎么记账”收成**一个契约**；
> 把当前散在 engine 的 19 个 store 调用点收编为 **3 个语义钩子**。
>
> 代码标记：本文件配套的 `FIXME(store-hook)` 注释就是当前调用点清单；
> 搜索该标记可得到全部待收编位置。Phase 1 后剩 **23 个标记** =
> 12 个生产旁路（`ensure_for_len` 9 + spec `snap_*` 3）
> + H3/H6 生命周期 + 结构性/诊断标记（N6、H1、coverage 等）。
>
> **类型以 §7 的实施记录为准**；§3 的 `Acquire` 结构是草案，落地时已
> 收敛为更小的 enum（`Acquire::Capacity(Need)` / `Acquire::Load(LoadRequest)`）。

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

### Phase 2：消灭同步驱逐

- `ensure_for_len` 改为**纯分配**（不够 = `EngineInvariant`）；
- `pump` 的 `Acquire(Grow)` 覆盖 prefill/decode/draft/MTP 的全部目标；
- 门：`ops.ensure_evicted == 0`；任何非零 = Acquire 算少了。

### Phase 3：收编旁路

- H4 `plan_onload` 进入 `Acquire(Load).load`；
- spec `snap_onload/offload` 进入 `Acquire(State)` / `Maintain`；
- H3/H6 进入 `Release`；
- 删除 `Node` 的 4 个变体，只留 `StoreHook`。

### Phase 4：修触发与账目

- N6 去掉 `max(2*reserve, 1GiB)`，改 `reserve + headroom` 或只走 `Acquire`；
- `FixedAfter` 补 `spec_snap` / `replay_snap`（账目先对，再谈容量）；
- 统一 metrics：`store.hook.acquire.*` / `.release.*` / `.maintain.*`。

---

## 5. 待决问题

1. **Admission 两套判决**：`turns.admission.decide`（策略）与 `Rel::check`
   （store 地板）是否合并为 `Acquire(Admit)` 一个出口？
   **状态：未做（Phase 2）**；Phase 1 保持 `Need.why = Admit` + 原有两层判决。
2. **Draft/MTP 增长**：并入 `step_need` 的上界，还是在 draft 前单独
   `Acquire(Grow)`？前者简单，后者精确。
   **状态：未做（Phase 2）**；当前 5 个 `ensure_for_len` 仍是槽兜底。
3. **StateLeg 是否独立**：作为 `Acquire.snap_bytes` 的一维，还是单独
   `Acquire(State)` / `Maintain(State)`？建议先作为维度，不增加第 4 钩子。
   **状态：未做（Phase 3）**；Phase 1 的 `Maintain::Pressure` 已带
   `snap_bytes` 字段，但生产压力路径仍传 0。
4. **Pressure 是否保留**：固定池档如果“腾槽不增驱动 free”，是否只保留
   `Maintain(Idle)` + `Acquire` 的 Deny，删除 N6？
   **状态：保留但未修（Phase 4）**；Phase 1 只改入口，1 GiB/`snap=0` 语义未动。
5. **TurnEnd 的 trim**：`trim_pools` 是 release 的一部分，还是单独的
   `Maintain(Trim)`？建议归 Release，避免第 4 钩子。
   **状态：已归 `Release`**（Phase 1 的 N4 实现内）。

---

## 6. 验收门

- **grep 门**：`plan_need` / `apply_verdict` / `plan_onload` / `snap_onload` /
  `snap_offload` 只出现在 `store_hook` 实现内；engine 其他位置零直呼。
  **Phase 1 部分通过**：生产 `plan_need` / `apply_verdict` 已收进
  `store_hook` 家族；`plan_onload` 仍有 `coverage.rs` 诊断直呼，
  `snap_*` 仍在 spec 旁路（Phase 3 清零）。
- **无同步驱逐门**：生产 e2e 的 `ops.ensure_evicted == 0`；
  非零即 Acquire 目标算少。**Phase 1 未动**（Phase 2）。
- **钩子覆盖门**：每个 e2e 场景断言实际发生的
  `store.hook.{acquire,release,maintain}` 计数与预期一致。
  **Phase 1 已换 metrics 名**：`store.node.*` → `store.hook.*`；
  计数门待 Phase 2 的 e2e 场景补。
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

### 7.3 仍未收编（按 Phase）

- **Phase 2**：`ensure_for_len` 9 个生产点（scheduler 3 / prefill 1 / draft 5）；
  `pump` 的 `Acquire(Grow)` 要覆盖 draft/MTP 目标，之后 `ensure_for_len`
  才能改成纯分配。
- **Phase 3**：H3 `register_chain_owned`、H6 `release_turn_table`；
  spec 的 3 个 `snap_*` 直呼。
- **Phase 4**：N6 的 1 GiB 目标、`FixedAfter` 漏 `spec_snap/replay_snap`、
  `store_hook` 覆盖门。

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
