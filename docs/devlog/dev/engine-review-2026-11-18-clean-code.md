# engine crate review —— 模块拆分与 clean code（2026-11-18）

> 范围：`crates/engine` 的**物理模块边界、状态所有权、长函数、配置面、死代码**。
> 不重复 2026-11-05 的 correctness findings（F-01..F-26 多数已修，见
> `docs/devlog/dev/engine-crate-导读.md` §12 与
> `docs/devlog/dev/engine-review-清单-2026-11-05.md`）。
> 本轮只做静态检视 + `cargo check` + CPU 单测；**未改任何代码**。
>
> 基线：`cargo test -p owl-engine --lib -- --test-threads=1` = **245 passed**。

---

## 0. 一句话结论

**目录已经按域分好了，但“行为”仍集中在几个 god object 上。**

`graph/ mem/ sched/ exec/ serve/ spec/ vision/` 的物理分层是对的；问题在：

1. `RunningEngine`（4663 行生产代码）被 **8 个文件**用 `impl` 扩展，字段
   `pub(crate)` 全仓可写；
2. `EngineKnobs` 是 **71 个 pub 字段**的配置大袋，其中 `gqa_max` 已经是
   **零消费者**的死旋钮；
3. `exec` 子模块用 `use super::*`，`sched ↔ exec` 有逻辑环；
4. `execute_spec_round` **938 行**、`submit_session_checked` **471 行**、
   `image_pad_law` **430 行**，都是“一个函数一个子系统”；
5. 生产路径还有 `expect("账在")` 与 `OWL_EXECUTOR_PROF` env 直读。

⇒ 下一步不是“再按行数切文件”，而是**按状态所有权切**：
把 `RunningEngine`/`TurnLedger`/`EngineKnobs` 里的子状态收成独立 struct，
让 `impl` 跟着 struct 走；否则拆完文件，逻辑仍然是跨模块 poke 字段。

---

## 1. 现状地图（生产代码行数；不含 `#[cfg(test)]` 之后的测试）

| 域 / 文件 | 生产行数 | 现状 | 备注 |
|---|---:|---|---|
| `running.rs` | **4663** | `RunningEngine` + turn 账 + pump + 准入 + store + 图像 + 诊断 | 8 个 `impl` 文件之一 |
| `mem/state.rs` | **3149** | `MemoryPlan` + `StatePool` + `KvBank/GdnBank/DraftBank` | 已有四户，但同文件 |
| `engine.rs` | **3025** | `Engine` 门面 + `EngineKnobs` + boot/plan/budget | 单文件 |
| `exec/spec.rs` | **2989** | spec 轮机器 + verify + fold + index + 快照 | `execute_spec_round` 938 行 |
| `sched/scheduler.rs` | **2612** | `schedule` + prefix/image 计划 + start_one | 5 个长函数 |
| `exec/coverage.rs` | **2378** | 覆盖率/方案对照诊断 + `impl RunningEngine` | 始终编译 |
| `exec/phases.rs` | **2322** | begin/prefill/decode/batch/verify + VLM | 单文件 |
| `exec/draft.rs` | **2271** | DFlash2 propose/select/encode + MTP | 单文件 |
| `graph/graph_fleet.rs` | **2014** | 图族装配 + 各 family planner | 单文件 |
| `vision/img_evict.rs` | **1919** | pad/detect/positions/layout | 单文件 |
| `parallel/tp.rs` | **1340** | TP2 装配/握手/自证 | 3 处 `unsafe` |
| `sched/sched_policy.rs` | 约 680 + 测试 | 准入/相位状态机（纯） | 单文件 |
| `serve/turn.rs` | 794 | Turn 协议类型 + `TurnLedger` | 混协议与可变账 |
| `service/{mod,boot,pump}.rs` | 367/272/456 | 引擎线程/装配/泵 | P1 下沉后边界较清 |

**已经做得好的地方（不要回退）：**

- `StatePool` 已按 `PoolGeom / KvBank / GdnBank / DraftBank` 分户，字段私有，
  只给语义方法；这是全仓最好的“状态所有权”样板。
- `SelectRuntime` 字段私有，只暴露 `note_* / set_* / view()` 意图方法。
- `BootCtx` 明确只读（配置/tokenizer/probes/boot seq），运行期零写入。
- `EngineService` 把线程/泵/卡锁封在引擎内，server 只见 `EngineClient`。
- `exec/prefill_chunk.rs`、`exec/mtp_window.rs` 等小文件边界清楚。

---

## 2. Findings（按“拆分收益 / 风险”排序）

> 行号 = 当前工作树。S1 = 结构性债，下一阶段先做；S2 = 明确的长函数/状态混居；
> S3 = 卫生/死代码/文档漂移。

### C-01（S1）`RunningEngine` 是跨 8 个文件扩展的 god object

- 证据：`running.rs:168`（字段全 `pub(crate)`）、`running.rs:523`
  （主 `impl`）；另有 `sched/scheduler.rs:247`、`exec/coverage.rs:616`、
  `exec/draft.rs:52`、`exec/spec.rs:146`、`exec/turn_end.rs:18`、
  `exec/phases.rs:237`、`exec/img_evict.rs:10` 共 8 个 `impl RunningEngine`。
- 影响：模块目录只是“文件分组”；任何模块都能改 `turns/bank/sel/hk/dev/peer`，
  不变量没有单一守门人。`turn_end`/`coverage`/`img_evict` 这些“域外”文件
  也在给核心对象加方法。
- 建议（按序）：
  1. 先把 `running.rs` 按**函数职责**拆成 `running/{mod,submit,pump,image,store,admission,diag}.rs`
     （纯移动，`impl` 仍可跨文件，行为不变）；
  2. 再把 `turns/bank/sel/hk` 改为**私有字段**，把对应方法移到各自的 struct；
     `exec`/`sched` 只接收 `&mut TurnLedger` / `&mut SessionBank` 等子状态；
  3. 最后把 `exec/*` 里的 `impl RunningEngine` 改成 `impl Executor<'a>` 或
     自由函数（`fn verify_round(exec: &mut ExecCtx, ...)`）。
- 验收：`cargo test -p owl-engine --lib` 245 全绿；一次 3060 生成 + TP2 冒烟
  token 级不变；`grep -c "impl.*RunningEngine"` 从 8 降到 ≤2。

### C-02（S1）`execute_spec_round` 938 行，单函数承载整轮状态机

- 证据：`exec/spec.rs:152`（938 行）；同文件 `verify_round` 411 行
  （`:1380`）、`capture_gdn_snap_tail` 204 行（`:2161`）。
- 影响：spec 的 prepare/verify/fold/accept/emit/metrics 全在一个函数里；
  任何一段改动都要重读 900 行，单测只能从外部打整轮。
- 建议：抽 `SpecRoundCtx`（token/pos/depth/drafts/accept 账/snap 句柄），
  拆成 `prepare_round` / `verify_round` / `fold_round` / `accept_round` /
  `emit_round` / `record_round`，每个返回显式 `Result`；纯判决
  （`depth_ladder_step`、`spec_degrade_verdict`）已在 `exec/mod.rs`，继续保留。
- 验收：同一 prompt（greedy，spec d7）逐 token 一致；245 + 长输出真机门。

### C-03（S1）`EngineKnobs` 71 个 pub 字段 + 死旋钮 `gqa_max`

- 证据：`engine.rs:98`（71 个 `pub` 字段，约 276 行）、`engine.rs:650`
  `from_config` 161 行、`engine.rs:217` `gqa_max`；全仓 `knobs.gqa_max`
  引用 **0 次**。`[kvmem].gqa_aggregation` 也**没有消费者**（只在
  `shared/config` 的 struct/default 出现）；真正的 GQA 聚合由
  `owl_store::select` 按 `ScoreMode::CosineMax` 决定
  （`crates/store/src/select.rs:714`）。
- 影响：配置模型与 `OwlConfig` 平行维护，新增字段要同步 3 处
  （struct/default/from_config）；`gqa_max` 证明已经漂移。运维写
  `[kvmem] gqa_aggregation` 不会改变引擎行为。
- 建议：
  1. 删除 `gqa_max` 或把它接到 `SelectRuntime`（二选一，不能留 no-op）；
  2. 把剩余字段按域收成 `KvmemKnobs / StoreKnobs / RuntimeKnobs /
     VisionKnobs / LoadKnobs`，`EngineKnobs` 只做组合；
  3. 每个子结构自带 `Default` + `from_config`，`from_config` 按域拆函数；
  4. 加一个**消费者门**：遍历 `EngineKnobs` 字段（或维护显式表），
     每个字段必须至少有一个 `knobs.<field>` 读点，否则测试红。
- 验收：`--print-config` 输出逐键不变；删除死字段后 grep 归零；245 绿。

### C-04（S1）`TurnLedger` 混了 turn 账、准入状态机、相位状态机、统计

- 证据：`serve/turn.rs:535` `TurnLedger`；字段含 `active/executing/next_id/
  pending_events`（turn 账）+ `admit_wait/reject_stats/wait_hist/turn_ms_ewma`
  （准入统计）+ `phase: PhaseSched`、`admission: Admission`（调度状态机）。
  `sched/scheduler.rs` 直接 `self.turns.admission.*` / `self.turns.phase.*`。
- 影响：`TurnLedger` 同时是“turn 容器”和“调度器状态袋”；`serve` 域反向
  持有 `sched_policy` 的状态机，`sched` 又通过 `RunningEngine` 改它。
- 建议：拆 `TurnLedger`（只留 turns/events/id/终态账）+
  `AdmissionState`（admit_wait/reject_stats/wait_hist/ewma）+
  `ScheduleState`（phase/rr 游标）；`Scheduler` 持有后两者，`TurnLedger`
  只被 turn 生命周期消费。
- 验收：`scheduler.rs` 不再出现 `turns.admission`/`turns.phase`；245 绿。

### C-05（S1）`exec` 子模块靠 `use super::*`；`sched ↔ exec` 有逻辑环

- 证据：`exec/phases.rs:3`、`exec/spec.rs:3`、`exec/draft.rs:6`、
  `exec/coverage.rs:39`、`exec/prefill_chunk.rs:3`、`exec/turn_end.rs:2`
  全部 `use super::*;`。`sched/scheduler.rs:2505` 调
  `crate::exec::bare_cost_rearm`；`sched/select.rs:16` 从
  `exec::coverage` 导入 `CoveragePending/Report`；`exec` 又 `impl RunningEngine`
  并调 `crate::running`。
- 影响：`exec/mod.rs` 的 `use` 变成隐式内部 API；`sched`/`exec` 谁依赖谁
  说不清；新增一个 helper 可能被 6 个子模块意外看到。
- 建议：
  1. `exec/prelude.rs` 显式列出共享类型，子模块 `use crate::exec::prelude::*`；
  2. 把 spec 降级/窗口纯函数从 `exec/mod.rs` 移到 `sched/spec_policy.rs`
     （或 `spec::policy`），`sched` 不再依赖 `exec`；
  3. 把 `CoveragePending/Report` 移到 `sched/select.rs` 或中性
     `diag/types.rs`，`select` 不再依赖 `exec::coverage`。
- 验收：`rg "use super::\*" crates/engine/src/exec` 生产文件为 0；
  模块依赖图无 `sched -> exec` 边。

### C-06（S2）`running.rs` 的两个超大入口

- 证据：`submit_session_checked` 471 行（`running.rs:1374`）；
  `image_pad_law` 430 行（`running.rs:860`）。
- 影响：submit 同时做校验/tokenize/内容身份/pad 律/预算/准入/入队；
  image_pad_law 同时做删除识别/pad/位置/快照计划，二者都难单测。
- 建议：`running/submit.rs`（校验 → tokenize → session 解析 → budget）、
  `running/image.rs`（pad 律 + 删除识别 + 调度票），主函数只留编排。
- 验收：纯 host 单测覆盖 submit 的每条拒绝路径；image 门沿用现有 20+ 测试。

### C-07（S2）`exec/phases.rs` 按相位混居

- 证据：`execute_prefill` 424 行（`:418`）、`execute_decode` 463 行
  （`:963`）、`execute_batch_decode` 278 行（`:1435`）、
  `verify_forward_dev` 253 行（`:1938`）、`prepare_vlm_rank` 196 行（`:23`）。
- 建议：拆 `exec/phases/{prefill,decode,batch,verify,vlm}.rs`；共享
  `PhaseCtx`/`DecodeStepInputs` 放 `phases/mod.rs`。`prefill_chunk.rs` 保持
  “块执行器”，`phases/prefill.rs` 只做 chunk 编排。
- 验收：三相位 token 级一致；245 + 8k/32k 真机门。

### C-08（S2）`sched/scheduler.rs` 的计划函数各自成山

- 证据：`schedule` 485 行（`:2125`）、`plan_prefix_reuse` 411 行（`:1706`）、
  `start_one` 327 行（`:578`）、`plan_image_leave_gap` 289 行（`:1039`）、
  `plan_image_leave_gap_multi` 294 行（`:1336`）。
- 建议：`sched/plan/{prefix,image,start,step}.rs`；`schedule` 只做
  “取事实 → 调 plan_* → 组 SchedulerOutput”，每个 plan 返回纯 `Result`。
- 验收：现有 CPU 调度矩阵（`sched_policy` 测试）全绿；`max_batch=1/2` 行为不变。

### C-09（S2）`mem/state.rs` 预算/池/三户仍在同一文件

- 证据：`AllocCtx::build` 270 行（`:2104`）、`KvBank::alloc`/`DraftBank::alloc`
  等（`:2867` 起）；四户 struct 已经存在，但 impl/工厂仍同文件。
- 建议：`mem/plan.rs`（MemoryPlan/BudgetInputs/auto_pool_tokens）、
  `mem/pool.rs`（StatePool/PoolGeom/AllocCtx）、`mem/kv_bank.rs`、
  `mem/gdn_bank.rs`、`mem/draft_bank.rs`。`StatePool` 已经是好边界，照它切。
- 验收：`[vram-gov]`/`[ctx-max]` 输出逐字一致；245 + 起机门。

### C-10（S2）`graph_fleet.rs` 按 family 拆

- 证据：`plan_verify` 364 行（`:1031`）、`plan_session` 319 行（`:592`）、
  `assemble` 265 行（`:266`）、`plan_dflash` 183 行（`:1542`）。
- 建议：`graph/fleet/{assemble,session,verify,dflash,batch}.rs`；
  `GraphFleet` 只留共享账与 `face_clone`。
- 验收：图捕获/回放/降级 eager 的现有门全绿；TP2 双 rank 图族字节一致。

### C-11（S2）生产路径仍有“结构性 expect”

- 证据（账在/刚判过）：
  - `exec/draft.rs:39`、`:1850`、`:2062`、`:2075`
  - `exec/phases.rs:994`（`.expect("账在")`）
  - `exec/turn_end.rs:295`、`:448`（`.expect("刚判过")`）
  - `mem/disk_runtime.rs:208`、`:220`、`:456`、`:468`（“调用方判过”）
- 影响：违反 AGENTS「长驻服务内部不变量失守禁止 panic」；会话账/池一旦漂移，
  引擎线程被打死，而不是把错误上抛到泵循环优雅收摊。
- 建议：全部改 `ok_or_else(ModelError::EngineInvariant { what })` 或显式
  `let Some(x) = ... else { return Err(...) }`；`world==2`/64B 固定头这类
  结构上不可失败的 expect 保留并写判据。
- 验收：生产段（`#[cfg(test)]` 之前）`grep "expect(\"账在\")\|expect(\"刚判过\")"`
  为 0；注入账漂移的单测返回 Err 而非 panic。

### C-12（S2）生产 env / 全局可变状态

- 证据：`service/pump.rs:92` `std::env::var_os("OWL_EXECUTOR_PROF")`；
  `service/pump.rs:36` `static WARNED: AtomicBool`；
  `engine.rs:1120` `static BOOT_SEQ`；`running.rs:4334` `static PUMP_METRIC_TICK`。
- 影响：P1 下沉后 env 规则在 `service` 域回潮；`WARNED`/`PUMP_METRIC_TICK`
  是跨实例全局状态，测试无法 reset。
- 建议：`OWL_EXECUTOR_PROF` → `StepProbes`/`perf_level` 配置项；
  `WARNED` → pump 局部 `bool` 或 `FactsCell` 字段；`PUMP_METRIC_TICK` →
  `Housekeeping` 字段；`BOOT_SEQ` 若必须进程级唯一，保留并写明理由，
  否则移到 `EngineService` 的 boot 计数。
- 验收：生产段 `std::env::var` 为 0（测试例外）；`static` 只剩不可变常量。

### C-13（S2）测试面是 9k 行单文件

- 证据：`crates/engine/src/tests.rs` **9154 行**，`tests_gguf.rs` 955 行；
  37 个测试散在单文件。
- 建议：`tests/{boot,sched,mem,exec,spec,vision,store}.rs` + `tests/mod.rs`，
  共享 fixture 进 `tests/common.rs`；`tests_gguf` 保持独立。
- 验收：`cargo test -p owl-engine --lib` 245 全绿；测试名/模块路径不变
  （或一次性迁移并更新 e2e 引用）。

### C-14（S3）诊断代码与死声明面常驻生产 crate

- 证据：`exec/coverage.rs` 2378 行生产代码，`impl RunningEngine`（`:616`）
  与 `impl RankDev`（`:2232`）；`exec/e4_api.rs` 68 行全部 `Err(未实现)`，
  `exec/mod.rs:322` 用 `#[allow(dead_code)]`；`sched/mod.rs:9` 整模块
  `#[allow(dead_code)]`。
- 影响：`coverage` 是 O(ctx²) 诊断面，却始终参与编译；E4 是“声明面先就位”
  的第二套接口，容易被误调用。
- 建议：`coverage` 移到 `exec/diag/coverage.rs`，用 **feature + 配置双门**
  （feature 关时只留类型/空实现，配置开才可调用）；`e4_api` 未接线前删除，
  或 `#[cfg(feature = "e4")]`。`allow(dead_code)` 从模块级收窄到具体项。
- 验收：默认二进制 `size`/编译时间下降；`--features diag` 仍能跑覆盖率门。

### C-15（S3）配置迁移后的文档/错误串仍在说旧 env

- 证据：`mem/state.rs:1869` “OWL_SPEC_DEPTH 未开”、`:2059`
  “OWL_COVERAGE_PROBE 未开?”、`:2141` “OWL_VRAM_TARGET/OWL_VRAM_RESERVE_MB
  可调”；`exec/coverage.rs:617` “OWL_SELECT_BLOCKS/SINK/RECENT”；
  `exec/phases.rs:1005` “OWL_KV_DUMP=<pos>”、`:1083` “OWL_HOST_ARGMAX=1”。
- 影响：运维按错误串找不到旋钮；与 2026-11-05 F-03 的“配置单源”结论冲突。
- 建议：错误串一律写当前配置键（`[spec].depth`、`[kvmem].coverage_probe`、
  `[store].vram_budget`、`[probes].kv_dump_at`…）；注释里的历史 env 名保留
  但标注“已退役”。

### C-16（S3）`sampler.rs` 的 `partial_cmp().unwrap()`

- 证据：`sampler.rs:157`、`:158`（top-k min 追踪）；候选已过滤
  `is_finite`，结构上不会 NaN，但这是“靠前置过滤”的隐式不变量。
- 建议：`f32::total_cmp`（或 `unwrap_or(Ordering::Equal)` 并写明判据），
  去掉生产 panic 面。

### C-17（S3）同名模块 `exec/img_evict.rs` vs `vision/img_evict.rs`

- 证据：`exec/img_evict.rs` 是盘层收尾（`execute_snap_disk_ops`，60 行）；
  `vision/img_evict.rs` 是 host 纯函数（1919 行）。
- 建议：`exec/img_evict.rs` 改名 `exec/snap_disk.rs`（或 `exec/image_disk.rs`），
  避免“同名两套实现”的搜索歧义。

### C-18（S3）`lib.rs` 的公开面没有随 P1/P3 收窄

- 证据：`lib.rs:130-138` 仍 `pub use RunningEngine/TpEngine/GraphPlan/
  ModelLoader/...`；`apps/server` 现在只用 `EngineService/EngineClient/
  DiagSnapshot/BlockRow/TurnOpts/StreamEvent`，`RunningEngine/TpEngine/
  GraphPlan` 只剩 tests/examples 消费。
- 影响：公开 API 大于实际契约，外部误用内部装配面；后续重构受 SemVer 约束。
- 建议：先 consumer audit；把仅测试/示例用的类型改为
  `#[cfg(feature = "e2e")]` 或 `pub(crate)`，保留 `EngineService` 窄面。

---

## 3. 目标模块树（按“状态所有权”而不是行数）

```text
engine/
├── mod.rs                  # 只 re-export 窄门面（Engine/EngineConfig/EngineKnobs）
├── knobs.rs                # EngineKnobs + 域子结构（Kvmem/Store/Runtime/...）
├── boot/
│   ├── mod.rs              # run/into_running 编排
│   ├── rank.rs             # boot_rank / RankDev / RankBoot
│   ├── plan.rs             # build_memory_plan
│   └── budget.rs           # assemble_budget
├── running/
│   ├── mod.rs              # RunningEngine 组合 + 生命周期
│   ├── submit.rs           # submit_session_checked
│   ├── pump.rs             # pump 步循环
│   ├── admission.rs        # admission_gate/apply_verdict
│   ├── store.rs            # store_load/release/checkpoint
│   ├── image.rs            # image_pad_law
│   └── diag.rs             # 诊断 accessor（只读）
├── sched/
│   ├── mod.rs              # Scheduler + ScheduleState
│   ├── policy.rs           # 纯准入/相位状态机
│   ├── plan/
│   │   ├── prefix.rs
│   │   ├── image.rs
│   │   ├── start.rs
│   │   └── step.rs
│   ├── select.rs
│   └── housekeeping.rs
├── exec/
│   ├── mod.rs              # ExecCtx + prelude + 薄 helper
│   ├── prelude.rs          # 显式共享类型
│   ├── prefill.rs
│   ├── decode.rs
│   ├── batch.rs
│   ├── verify.rs
│   ├── spec/
│   │   ├── mod.rs          # SpecRoundCtx + 编排
│   │   ├── round.rs        # prepare/accept/emit
│   │   ├── verify.rs
│   │   ├── fold.rs
│   │   └── index.rs
│   ├── draft/
│   │   ├── mod.rs
│   │   ├── dflash.rs
│   │   └── mtp.rs
│   └── diag/               # coverage / e4（feature-gated）
├── mem/
│   ├── mod.rs
│   ├── plan.rs             # MemoryPlan/BudgetInputs/auto_pool_tokens
│   ├── pool.rs             # StatePool/PoolGeom/AllocCtx
│   ├── kv_bank.rs
│   ├── gdn_bank.rs
│   ├── draft_bank.rs
│   └── disk*.rs
├── graph/
│   ├── mod.rs
│   ├── plan.rs
│   └── fleet/{mod,assemble,session,verify,dflash,batch}.rs
├── vision/
│   ├── mod.rs
│   ├── vlm.rs
│   └── img_evict/{mod,pad,detect,positions,layout}.rs
└── serve/
    ├── mod.rs
    ├── turn/{mod,types,ledger}.rs
    └── msg_boundary.rs
```

---

## 4. 施工顺序（每刀一个 commit；先纯移动，后改所有权）

### S0 —— 卫生刀（低风险，先做）

- 删除/接线 `EngineKnobs.gqa_max`（C-03 第一条）。
- 生产 `expect("账在")/expect("刚判过")` 改 `Result`（C-11）。
- `OWL_EXECUTOR_PROF` → `StepProbes`；`WARNED`/`PUMP_METRIC_TICK` 去 static（C-12）。
- 旧 env 文案改当前配置键（C-15）。
- `sampler.rs` 改 `total_cmp`（C-16）。
- **验收**：245 绿；`grep` 归零；无行为变化。

### S1 —— `running/` 纯移动（行为零变化）

- `running.rs` → `running/{mod,submit,pump,image,store,admission,diag}.rs`。
- `TurnLedger` 暂不拆；先把 `impl` 与函数搬过去，保持 `pub(crate)`。
- **验收**：245 + 一次 3060 生成 + `/debug/*` 冒烟；token 级不变。

### S2 —— `TurnLedger`/`Scheduler` 状态收口

- 拆 `AdmissionState`/`ScheduleState`，`TurnLedger` 只留 turn 账（C-04）。
- `Scheduler` 持有调度状态；`scheduler.rs` 不再 `turns.admission.*`。
- **验收**：max_batch=1/2 调度矩阵全绿；取消/排队/超时 e2e。

### S3 —— `exec` 拆分（收益最大、风险最高）

- `exec/mod.rs` prelude + spec policy 外移（C-05）。
- `spec.rs` 拆 `SpecRoundCtx` + phase 函数（C-02）。
- `phases.rs` 拆 prefill/decode/batch/verify/vlm（C-07）。
- `draft.rs` 拆 dflash/mtp（C-10 同族）。
- **验收**：greedy spec d7 逐 token 一致；AL/吞吐噪声内；长输出真机门。

### S4 —— `mem`/`graph`/`sched`/`vision` 物理拆分

- `mem/state.rs` → plan/pool/banks（C-09）。
- `graph_fleet.rs` → fleet family（C-10）。
- `scheduler.rs` → plan/*（C-08）。
- `vision/img_evict.rs` → 子模块（C-17）。
- **验收**：`[vram-gov]`/`[ctx-max]` 逐字一致；图捕获/TP2/图像删除门全绿。

### S5 —— 测试面与公开面

- `tests.rs` → `tests/*`（C-13）。
- `lib.rs` 公开面 consumer audit（C-18）。
- **验收**：245 + e2e 路径不变；外部 crate 编译不破。

---

## 5. 不要做的事

- **不要按行数机械切文件**：`StatePool` 的四户是正面样板；切完文件但字段仍
  `pub(crate)`，等于没切。
- **不要先动 `exec/spec.rs` 的数值路径**：先抽状态/函数边界，臂体逐字不动。
- **不要为了“看起来分层”给 `RunningEngine` 加更多 trait**：当前问题是状态所有权，
  不是缺少抽象；先私有字段，再谈 trait。
- **不要重复 2026-11-05 的 F-01..F-26**：那些 correctness 项多数已修；本轮只处理
  结构/卫生。F-20（两套准入）已于 2026-11-18 收口：生产 = `decide_slots` +
  store `plan_admission → commit_admission`；`capacity` 只留诊断。

---

## 6. 验证记录

- `cargo check -p owl-engine`：通过。
- `cargo test -p owl-engine --lib -- --test-threads=1`：**245 passed**。
- 未跑 GPU 套件 / TP2 真机（本轮只做静态检视）；S1 之后每刀再跑对应门。
- 未改代码、未提交。
