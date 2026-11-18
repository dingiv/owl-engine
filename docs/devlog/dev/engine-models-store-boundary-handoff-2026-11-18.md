# engine × models × store 边界重构 —— 设计 / 交接（2026-11-18）

> 给下一绘画的交接卷。配套阅读：
> `docs/devlog/dev/engine-review-2026-11-18-clean-code.md`（本次 review 的 findings）
> · `docs/devlog/dev/engine-crate-导读.md`（boot/run/TP2 现状）
> · `docs/devlog/memory/store.md`（三层状态）
> · `docs/devlog/canon/charter.md`（A1/A2/A5/A6）。
>
> 用户裁决（2026-11-18）：
>
> 1. **engine 现在做了很多 models / store 该做的事**；
> 2. **模型执行下沉 `owl-models`**（models 负责“怎么算”）；
> 3. **`owl-store` 负责状态与状态机转移**（“东西在哪、这一步该看到什么、
>    放不下丢谁”），**store 不做 I/O**；
> 4. **engine 在做 I/O 之前先问 store 要计划/指引**，拿到计划后自己执行 I/O，
>    回来向 store `commit`；
> 5. 本卷是设计 + 交接，不要求本轮改代码。

---

## 0. 一句话目标

**把 engine 从“模型执行 + 状态决策 + I/O 执行 + 请求编排”四合一，
收敛成“请求编排 + I/O 执行 + 图/TP 设备程序”的薄层。**

目标依赖方向（不可反向）：

```text
apps/server
    │  EngineClient / StreamEvent
    ▼
owl-engine        编排：turn 生命周期 / 相位调度 / 图捕获回放 / TP AR / 物理设备句柄 / I/O 执行
    ├── owl-models   执行：模型 forward / verify / propose / fold / sampling / KV 编解码 / MTP 窗口
    └── owl-store    状态：session / block / snapshot / tier / reservation 的状态机与计划（零 I/O）
```

- `owl-models` **不得**依赖 `owl-engine` / `owl-store`；
- `owl-store` **不得**依赖 `owl-engine` / `owl-models`；
- 二者之间也不互相依赖（需要共享的纯数据类型可放 `owl-iface` 或各自端口）。

---

## 1. 现状审计：engine 正在做谁的活

> 只列“职责错位”的实证；行为正确性 findings 见配套 review（C-01..C-18）。

### 1.1 模型执行（应在 models）

| engine 现状 | 位置 | 现在做什么 | 应该 |
|---|---|---|---|
| `exec/phases.rs` | `execute_prefill` 424 行（`:418`）、`execute_decode` 463 行（`:963`）、`execute_batch_decode` 278 行（`:1435`）、`verify_forward_dev` 253 行（`:1938`） | 组 `ForwardCtx`、发模型核、读 logits、采样前置 | models 提供 `runtime::run_phase(...)`；engine 只给 `ModelExecCtx` + `GraphRunner` |
| `exec/spec.rs` | `execute_spec_round` 938 行（`:152`）、`verify_round` 411 行（`:1380`） | verify/fold/q/接受账/草稿 | models `runtime::spec`（verify/fold/q）；engine 保留图/AR/turn 事件 |
| `exec/draft.rs` | `dflash_propose` 472 行（`:412`）、`dflash_select_merged` 310 行（`:1480`）、`propose_round` 181 行（`:1169`） | DFlash2/MTP 的 propose/select/encode | models `runtime::draft`（族内数学/层/选择器）；engine 保留图/TP |
| `exec/kvpage.rs` | 421 行 | KV 页 host 编解码（f16/fp8/int8） | models `kv_codec`（格式/布局属于模型契约）；store 只给地址/计划 |
| `exec/mtp_window.rs` / `mtp_ring.rs` | 308 / 277 行 | MTP 窗口读集、物理环映射 | models `mtp`（窗口/环是模型注意力语义） |
| `exec/proposal_q.rs` | 1072 行（生产约 530） | 拒绝采样 q / 走格 | models `spec::q`（纯数学） |
| `exec/prefill_chunk.rs` | 1075 行 | chunk host 编排 + 模型调用 | 一半是 engine 的 chunk/状态；模型调用下沉 models |

**边界判据**：

- **models** 拥有：模型几何、层语义、forward/verify/propose/fold 的**数学与核发射**、
  采样（host/device）、KV 编解码、MTP 窗口/环、DFlash2 选择器；
- **engine** 拥有：CUDA graph 捕获/回放、TP2 AR/stream 序、物理 `Bytes` 分配、
  turn 生命周期、事件序、取消/超时、服务泵；
- models 可以拿 `&mut D: DeviceClient` 发模型核；但**不得**拿
  `RunningEngine` / `TurnLedger` / `SessionBank` / `StatePool` 具体类型。

### 1.2 状态与计划（应在 store）

| engine 现状 | 位置 | 现在做什么 | 应该 |
|---|---|---|---|
| `mem/state.rs` `MemoryPlan` / `BudgetInputs` / `auto_pool_tokens` / `pool_cap` | `build` 270 行（`:2104`）、`alloc`（`:2867`） | 设备池预算、每 token 字节、fixed_after、池容量 | store `plan`（已有 `MemoryPlan/BudgetInputs/VramBudget`）；engine 只接 plan 做物理分配 |
| `running.rs` `admission_gate` / `apply_verdict` | `:3416` / `:4015` | 地板/腾位/准入判决 | store `check_admission` + `ReservationLedger`；engine 执行 verdict |
| `sched/sched_policy.rs` `Admission` | `:353` | 第二套容量/退避状态机 | 容量部分并入 store；engine 只留 `PhaseSched`（turn 相位/公平） |
| `running.rs` `store_load/store_release/checkpoint_session_kv` | `:2265/:2442/:3604` | 会话/快照/落盘编排 | store 出 `LoadPlan/ReleasePlan/CheckpointPlan`；engine 执行 + commit |
| `mem/disk_runtime.rs` | 515 行 | 磁盘溢写/回填策略 + I/O | store `disk` 只出 `SpillPlan/FillPlan/EvictPlan`；engine `io::disk` 执行 |
| `exec/turn_end.rs` | 500 行 | 终局快照/发布/清理决策 | store `state`/`rel` 出计划；engine 只执行 |
| `sched/select.rs` | 478 行 | 选择层运行态（pending/report/stage） | store 出 `ViewPlan/SelectBudget`；engine 保留 in-flight 运行态与设备表写入 |

**store 的边界（用户裁决）**：

- store **只做账、身份、状态机、策略、计划**；
- store **不发** `DeviceClient` 命令、**不读写文件**、不持有设备块所有权；
- engine 在 I/O 前问 store：`plan_alloc / plan_release / plan_evict /
  plan_snapshot / plan_restore / plan_offload / plan_onload / check_admission`；
- engine 执行后 `commit_*`，store 只更新账；失败走 `abort_*`。

### 1.3 store 自己也曾有 I/O（S1 已修）与 god object

| 现状 | 位置 | 问题 | 目标 |
|---|---|---|---|
| `store/src/moves.rs` | 192 行 | `download_span/upload_span` 直接调 `DeviceClient` | **已迁出**：`engine/src/io/moves.rs`；store 生产代码已无 `DeviceClient`（S1，见 §6） |
| `store/src/rel.rs` | **4810 行** | KV 腿 + 状态腿 + 准入/loan/report 全在一个 `Rel` | 先拆 `rel/{kv,state,admin,disk}.rs`，再接 engine 的状态计划；否则 store 会变成更大的 god object |
| `store/src/leg.rs` | 2189 行 | 腿状态 + 搬运计划 + 测试编排 | 保留 plan 类型；把“测试侧 face 编排”移到 engine 测试 helper |
| `store/src/select.rs` | 1997 行 | 打分/视图计划（纯 host） | 保留；engine `SelectRuntime` 只留运行态 |

> 顺序很重要：**先拆 store 的 `Rel`，再把 engine 的状态职责搬进来**；
> 否则只是把 engine 的 god object 换到 store。

---

## 2. 目标契约

### 2.1 models：`runtime` 端口

建议新增 `crates/models/src/runtime/`（名字可改，职责固定）：

```rust
// models/src/runtime/mod.rs
pub enum Phase {
    Prefill { chunk: Range<usize> },
    Decode,
    Verify { rows: usize },
    Propose { depth: usize },
    Fold { accepted: usize },
}

pub struct ModelExecCtx<'a> {
    pub model: &'a Model,
    pub env: &'a EnvProvider,
    /// 只读/可写 KV 视图；store 的 logical block 由 engine 映射成 KvBuffers
    pub kv: KvView<'a>,
    pub gdn: GdnView<'a>,
    pub phase: EvalPhase,
    pub positions: PositionView<'a>,
}

pub trait GraphRunner {
    /// 回放已捕获的图；返回 tap/输出块。engine 实现（GraphFleet/GraphPlan）。
    fn replay(&mut self, key: GraphKey, inputs: &[(&str, &[f32])]) -> Result<GraphOut>;
}

pub async fn run_phase<D, G>(
    ctx: &mut ModelExecCtx<'_>,
    face: &mut D,
    graph: &mut G,
    phase: Phase,
) -> Result<PhaseOutput>
where
    D: DeviceClient,
    G: GraphRunner;
```

约束：

- `ModelExecCtx` 只含**模型/张量/KV 视图**，不含 turn/session/queue；
- `GraphRunner` 由 engine 实现；models 不捕获 CUDA graph；
- `PhaseOutput` = logits/tap/采样候选，不含 `TurnEvent`；
- 采样可作为 `models::sampling`（host/device），engine 把 `TurnSampler`
  适配成 `SamplerCfg`；
- 迁移顺序：先 `mtp_window/mtp_ring/kvpage/proposal_q`（纯/低耦合），
  再 `draft`/`spec` 数学，最后 `phases` 的 forward 组。

> 不建议一次性把整个 `exec/` 搬进 models：`exec` 里还有 chunk 编排、图输入槽、
> TP2 AR、turn 事件，这些是 engine 的设备/生命周期职责。先搬“纯模型数学”，
> 再用 `ModelExecCtx` 把剩余 forward 收进去。

### 2.2 store：`plan → engine I/O → commit`

现有 `store/src/disk.rs` 已经是正确样板（`plan_spill → 调用方写 → commit_spill`）。
把它推广到所有层：

```rust
// store/src/plan.rs 的方向（已有 MemoryPlan/BudgetInputs/VramBudget）
pub struct AllocPlan { pub blocks: Vec<BlockRef>, pub evict: Vec<EvictRef>, ... }
pub struct ReleasePlan { ... }
pub struct LoadPlan { pub pieces: Vec<PieceSpec>, ... }

impl Rel {
    pub fn plan_alloc(&mut self, req: &TurnReq, vram: VramSnapshot) -> Result<AllocPlan>;
    pub fn commit_alloc(&mut self, plan: &AllocPlan, slots: &[SlotId]) -> Result<()>;
    pub fn abort_alloc(&mut self, plan: &AllocPlan) -> Result<()>;

    pub fn plan_release(&mut self, turn: TurnId) -> ReleasePlan;
    pub fn commit_release(&mut self, plan: &ReleasePlan) -> Result<()>;

    pub fn plan_offload(&mut self, need: Need, protect: &[BlockRef]) -> Result<OutPlan>;
    pub fn commit_offload(&mut self, plan: &OutPlan) -> Result<()>;
    pub fn abort_offload(&mut self, plan: &OutPlan) -> Result<()>;
}
```

- `AllocPlan/ReleasePlan` 只含**逻辑块/槽/键/路径**，不含 `Bytes`；
- engine 把 `BlockRef` 映射成物理 `Bytes`，执行 D2H/H2D/读写；
- store 的 `commit_*` 只更新身份/账/引用，不碰设备；
- 失败必须 `abort_*`，不允许“I/O 失败但账已提交”的半态；
- 准入 = `check_admission` 返回 `AdmissionVerdict`；engine 只执行
  `Admit/EvictThen/Queue/Reject`，不再自己算容量；
- **store 不依赖 `owl-models`**：池几何/每 token 字节用 store 自己的
  `PoolLayout`/`KvConfig` DTO；engine 把 `ModelDims/QuantPlan` 映射成这些
  DTO 后再调 `plan_*`。不要把 `ModelDims` 塞进 store。

### 2.3 engine：编排 + I/O 执行

建议新增/收敛：

```text
engine/src/io/
  mod.rs        # 顶层 I/O 函数（AGENTS：I/O 统一顶层函数）
  moves.rs      # 从 store::moves 迁入：dtoh/upload/offload/onload
  disk.rs       # 从 mem/disk_io + disk_runtime 的 I/O 半迁入
  snapshot.rs   # 快照拍摄/恢复的设备半
```

- `running.rs` 只留 turn 生命周期与 `store.plan_*` 调用；
- `mem/state.rs` 收敛为 `DevicePool`（物理分配/释放/清零），
  逻辑池/预算/准入全部走 store；
- `exec/` 收敛为 `runtime` 适配器：组 `ModelExecCtx`、调 `models::runtime`、
  处理 `TurnEvent`/取消/超时；不再写模型层数学；
- `graph/`、`parallel/` 保持 engine（CUDA graph、TP2 AR、stream 序）；
- `sched/PhaseSched` 保持 engine（turn 相位/公平）；容量准入走 store。

### 2.4 一次 turn 的目标调用树

```calltree
engine/service/pump::run
  → engine/running::pump
      → engine/sched/PhaseSched::next            # 谁跑、跑什么相位（engine 状态机）
      → store/plan::check_admission              # 容量/地板/腾位（纯 host，零 I/O）
          → store/rel::plan_alloc                # 逻辑块/槽/驱逐计划
      → engine/io::apply(plan)                   # 执行 D2H/H2D/零化/快照搬运
          → store/rel::commit_alloc              # 只落账（I/O 成功之后）
      → models/runtime::run_phase(ctx, face, graph, phase)
          → models/layers/interpreters           # 模型数学/核发射
          → engine/graph::GraphRunner::replay     # 已捕获图的回放（engine 实现）
      → engine/running::emit(TurnEvent)          # 事件序/取消/背压
```

纪律：**store 的 `plan_*` 在任何 I/O 之前；`commit_*` 只在 I/O 成功后。**
`plan` 失败 ⇒ 不碰设备；I/O 失败 ⇒ `abort_*`，不允许半账。

---

## 3. 迁移切片（每刀一个 commit）

### S0：冻结边界 + 盘点（本卷）

- 新增本交接卷 + review 卷；
- 不改代码；
- 下一绘画先读本卷 §1/§2，确认 open questions（§5）。

### S1：store 去 I/O（最小、可独立验收）

- `store/src/moves.rs` → `engine/src/io/moves.rs`；
- store 内 `download_span/upload_span/offload_to_host/onload_from_host`
  调用点改为 engine `io`；
- store 删 `DeviceClient` 依赖（保留 `owl-iface` 的 `Bytes/ModelError/ConnSlot`）；
- 测试侧 `owl-cpu` face 编排移到 engine 测试 helper；
- **验收**：`rg "DeviceClient|std::fs|tokio::fs" crates/store/src` = 0；
  `cargo test -p owl-store`；engine 245 全绿；一次 D2H/H2D 搬运冒烟。

### S2：预算/准入单一真源

- engine `mem/state.rs` 的 `MemoryPlan/BudgetInputs/auto_pool_tokens/pool_cap`
  删除，改用 store `plan/budget`；
- engine `sched_policy::Admission` 的容量部分删除，改用 `store::check_admission`
  + `ReservationLedger`；`PhaseSched` 保留；
- `running::admission_gate/apply_verdict` 只执行 store verdict；
- **验收**：同一组 `StoreFacts/TurnFacts` 下，准入决策表与旧实现逐项一致
  （录旧 verdict 回放）；`[vram-gov]`/`[ctx-max]` 输出逐字一致；245 绿。

### S3：逻辑状态 / 物理池分离

- store 拥有 `SessionBank/KvStore/StateStore`（多数已拥有）；
- engine `StatePool` 改名/收敛为 `DevicePool`：只做 `alloc/free/zero/write`；
- `mem/state.rs` 的池几何/每 token 字节/fixed 项由 store plan 提供；
- **验收**：池账/块表/快照键逐字一致；起机 + 2k/32k 生成门；TP2 池账一致。

### S4：models runtime（先纯后深）

1. `mtp_window/mtp_ring/kvpage/proposal_q` → `models::runtime`（纯/低耦合）；
2. `draft` 的 DFlash2/MTP 数学 → `models::runtime::draft`；
3. `spec` 的 verify/fold/q → `models::runtime::spec`；
4. `phases` 的 forward 组 → `models::runtime::run_phase`；
- engine 保留 `GraphRunner` 实现、TP2 AR、turn 事件、I/O；
- **验收**：greedy spec-off / spec-d7 逐 token 一致；AL/吞吐噪声内；
  `cargo test -p owl-models` + engine 245 + 长输出真机门。

### 3.1 源 → 目标速查

| 源（当前） | 目标 | 迁移类型 |
|---|---|---|
| `store/src/moves.rs` | `engine/src/io/moves.rs` | 文件搬移 + 调用点改 |
| `engine/src/mem/state.rs` 的 `MemoryPlan/BudgetInputs/auto_pool_tokens/pool_cap` | `store/src/plan.rs` / `store/src/budget.rs` | 纯函数搬移；engine 传 `PoolLayout/VramSnapshot` |
| `engine/src/sched/sched_policy.rs` 的 `Admission` 容量部分 | `store/src/budget.rs` + `reservations.rs` | 状态机合并；`PhaseSched` 留 engine |
| `engine/src/exec/kvpage.rs` | `models/src/runtime/kv_codec.rs` | 纯格式/编解码 |
| `engine/src/exec/mtp_window.rs` / `mtp_ring.rs` | `models/src/runtime/mtp.rs` | 纯窗口/环语义 |
| `engine/src/exec/proposal_q.rs` | `models/src/runtime/spec/q.rs` | 纯 q 数学 |
| `engine/src/exec/draft.rs` 的 DFlash2/MTP 数学 | `models/src/runtime/draft/{dflash,mtp}.rs` | 先搬纯函数，图/TP 留 engine |
| `engine/src/exec/spec.rs` 的 verify/fold/q | `models/src/runtime/spec/{verify,fold}.rs` | `fold_tree` 已在 models，先接线 |
| `engine/src/exec/phases.rs` 的 forward 组 | `models/src/runtime/phase/{prefill,decode,verify}.rs` | 最后搬；`GraphRunner` 端口 |

> 搬移顺序 = 表从上到下；越靠下越依赖 `ModelExecCtx`/`GraphRunner` 端口，
> 所以先做 S1/S2，再逐步 S4，不要一次性翻 `exec/`。

### S5：清理

- `exec/coverage`/`e4_api` 收进 `engine/diag`（feature-gated）；
- `EngineKnobs` 按域拆（见 review C-03）；
- `tests.rs` 拆 `tests/*`；
- `lib.rs` 公开面 consumer audit；
- **验收**：245 + models/store 测试；无行为变化。

---

## 4. 关键风险与设计决策

### 4.1 不要让 models 拿 engine 类型

如果 `models::runtime` 参数里出现 `RunningEngine/RankDev/StatePool`，迁移就失败
（models 反向依赖 engine，Cargo 也编不过）。正确做法：

- models 定义 `ModelExecCtx` / `GraphRunner` / `KvView` / `GdnView` **端口**；
- engine 实现这些端口（`GraphFleet`/`TpCtx`/`StatePool` 适配）；
- 热路径用泛型（`G: GraphRunner`），不要 `Box<dyn>` 每步动态派发；
- 如果端口方法需要 async，用 `async fn` in trait（Rust 1.75+）或
  `impl Future` 的显式泛型；避免 `dyn`。

### 4.2 图捕获归属

**建议 engine 拥有 graph capture/replay**：

- CUDA graph 的形状/流/事件/TP2 同步是设备程序，不是模型数学；
- models 只给 `GraphKey`（phase/depth/shape）和输入槽；
- engine 的 `GraphRunner::replay` 返回输出块；捕获失败回落 eager 仍由 engine 决定。

如果下一绘画认为 graph 属于模型执行，需要先证明 models 能在不依赖
`owl-cuda` 的情况下表达 capture（目前 models 依赖 `owl-iface` 的 `DeviceClient`，
不应再加 `owl-cuda`）。

### 4.3 采样归属

**建议 models 拥有采样数学**（top-k/top-p/rep penalty/候选合并），engine
只传 `SamplerCfg` 和 `TurnSampler` 适配。理由：采样是模型输出分布的一部分；
但 `TurnSampler` 是服务面协议，放在 engine，models 不认它。

### 4.4 store 的 `Rel` 必须先拆

`store/src/rel.rs` 4810 行，承载 KV 腿/状态腿/准入/loan/report。直接把 engine
的 `MemoryPlan`/`admission`/`snapshot` 搬进去，会制造第二个 `RunningEngine`。

建议先拆：

```text
store/src/rel/
  mod.rs      # Rel 组合 + 公共不变式
  kv.rs       # KV 链/块/前缀匹配
  state.rs    # GDN 快照/状态腿
  admin.rs    # 准入/loan/report/诊断
  disk.rs     # 磁盘 attach/adopt
```

`Rel` 只保留 `parts()` 组合与跨腿不变量；每腿自己的状态机在子模块。
这一步是 S2/S3 的前置，不是可选项。

### 4.5 性能红线

- 迁移只做**职责搬移 + 端口适配**，不改数值/图/TP 路径；
- 每步热路径禁止新增 `dyn`/`Arc<Mutex>`/跨线程通道；
- 变更前后跑：engine 245、models 全量、spec-off 逐 token、spec-d7 长输出、
  TP2 双 rank 图字节一致；decode/ prefill 吞吐在 ±3% 内。

### 4.6 不要做的事

- 不要为了“models 负责执行”把 `turn`/`session`/`queue`/`pump` 搬进 models；
- 不要把 store 变成 I/O 执行器（`moves` 必须出 store）；
- 不要在 engine 里保留第二套 `MemoryPlan/Admission` 兼容层；
- 不要先改 `exec/spec` 数值路径；先搬纯函数/端口，最后搬 forward；
- 不要把 `GraphPlan`/`TpEngine` 的捕获面暴露给 models。

---

## 5. 交接清单 / 开放问题

### 5.1 当前基线

- `cargo check -p owl-engine`：通过；
- `cargo test -p owl-engine --lib -- --test-threads=1`：**245 passed**；
- server 分层重构已提交 `258bde7e`（`view/service/repo/domain/infra`）；
- engine 的工作树**未提交**改动：`crates/shared/src/config/mod.rs` 有 ar2push/
  准入相关改动；engine 本体当前无未提交 diff（以 `git status` 为准）。

### 5.2 下一绘画先读

1. 本卷 §1/§2；
2. `docs/devlog/dev/engine-review-2026-11-18-clean-code.md`（C-01..C-18）；
3. `docs/devlog/dev/engine-crate-导读.md` §2-§8（boot/run/TP2）；
4. `docs/devlog/memory/store.md`（store 的 plan/commit 纪律）；
5. `crates/store/src/disk.rs` 头注（**正确样板**：plan → 调用方 I/O → commit）。

### 5.3 先做哪一刀

**S1 已完成**（见 §6），store 生产代码已无 I/O。下一步做 **S2（预算/准入单一真源）**；
按 §4.4，先拆 `store/src/rel.rs`，再搬 engine 的 `MemoryPlan/Admission`。不要先动
`exec/phases`（S4 最后做）。

### 5.4 开放问题（需用户/下一绘画拍板）

1. `models::runtime` 的模块名/端口名（`runtime` vs `exec` vs `engine_api`）？
2. graph capture 是否确定留在 engine？（本卷建议：是）
3. 采样是否确定下沉 models？（本卷建议：数学下沉，`TurnSampler` 留 engine）
4. `StatePool` 是否改名为 `DevicePool`，逻辑块全部归 store？
5. `sched_policy::PhaseSched` 是否确认留 engine（只把 `Admission` 容量搬 store）？
6. store `Rel` 拆分的提交边界：是否先独立一刀（推荐）再做 S1/S2？
7. 迁移期间是否允许 engine 保留**只读兼容 re-export**（`crate` 内），
   还是要求一次性改完调用点？

### 5.5 验收命令（每刀）

```bash
cargo check --workspace --all-targets
cargo test -p owl-engine --lib -- --test-threads=1     # 245
cargo test -p owl-models --lib
cargo test -p owl-store --lib
# GPU（串行，空卡）：
OWL_TEST_DEVICE=41:00.0 cargo test -p owl-engine --lib tests_gguf::gpu_gguf_27b_smoke -- --test-threads=1
# TP2 门按需：tp_assemble_gate / tp_determinism_gate / tp_transport_bits_gate
```

> 迁移期任何“看起来等价”的重排都要用**同 prompt 逐 token** + 池账/日志逐字对拍；
> 不要用吞吐单次采样判断正确性（±20% 方差，AGENTS 测量纪律）。

---

## 6. S1 实施记录（2026-11-18）

- `crates/store/src/moves.rs` 删除；搬运原语迁至
  `crates/engine/src/io/moves.rs`（`engine::io` 模块头写清
  `store plan_* → engine io 执行 → store commit_*`）。
- store 的 `OutPlan/InPlan/ReadPlan` 与 `SpillPlan/FillPlan` 保持为唯一
  I/O 描述；`store/src/lib.rs` 头注明确“store 不做 I/O”。
- store **生产代码**不再引用 `DeviceClient`；`leg.rs` 的
  `#[cfg(test)]` 仍用 `CpuFace` 驱动 plan/commit 闭环（测试编排，不是 store
  运行期行为）。
- 验证：`cargo test -p owl-store --lib` = 205 passed；
  `cargo test -p owl-store --tests` = 23 passed + 1 ignored；
  `cargo test -p owl-engine --lib` = 248 passed（含新增 io::moves 3 项）；
  `cargo check -p server` 通过。
- 遗留：`io::moves` 当前是逐片/整块原语，生产主路径仍走 `face.swap`
  （批量、带槽台）；待后续逐片路径接线时移除模块级 `allow(dead_code)`。
- 下一刀：S2 预算/准入单一真源；按 §4.4，先拆 `store/src/rel.rs` 再搬
  engine 的 `MemoryPlan/Admission`。

---

## 7. S2a 准入 Plan/DryRun（2026-11-18）

用户裁决：store 是**纯状态机**，不要事务 / 半开半闭的 commit 概念。
`plan` 与 `commit` 是**两套独立 API**：

```text
Rel::plan_admission(turn, floor, ctx, reserved_slots, alloc_slots) -> AdmissionPlan
    // 纯 host、零状态变更；verdict = Admit / EvictThen / Queue / Reject

Rel::commit_admission(turn, floor_bytes, reserved_slots, alloc_slots)
    -> AdmissionApplied
    // engine 做完后把**实际值**报给 store；一次状态迁移：
    //   ActiveFloor Grant + admission 在册 + GDN 行 + Row Grant
    // 不传 plan，没有 open/commit/abort 三态，也没有回滚日志
```

- 完整请求链：`plan_admission` 判不过（`EvictThen`）⇒ `plan_need` 告诉
  engine 搬哪些 KV/丢谁 ⇒ engine 执行 swap/drop ⇒ `commit_offload/onload`
  按实际搬成的块改账 ⇒ 重新 `plan_admission` ⇒ `commit_admission` ⇒ 开始执行。
- `EvictThen` / `Queue` / `Reject` 都不是可提交态；`EvictThen` 必须由 engine
  先执行搬迁，再重新 plan（账已变，旧 plan 不再有效）。
- `AdmissionApplied` 只是落账结果（`grant` / `gdn_row` / `row_grant` /
  `floor_bytes`），不是事务句柄。
- 可失败前置（turn 重复、GDN 行耗尽/重复）都在第一次改账前检查。
- 测试：`admission_plan_is_dry_run_then_commit_changes_state`、
  `plan_admission_reports_non_admit_without_touching_store`；
  `cargo test -p owl-store --lib` = 205 passed。
- **已接线（生产全路径）**：scheduler 先 `decide_slots`（只判槽位/会话/退避），
  再 `plan_admission`：
  - `Admit` → `commit_admission(实际值)` 一次落 Grant + admission + GDN 行；
  - `EvictThen` → `store_hook(capacity)` 执行 `plan_need/swap/drop` → 重新 plan；
  - `Queue` → 留队；`Reject` / commit 前置失败 → 本 turn `Failed`。
  `start_one_planned` 只做启动，不再重复三步落账；`start_one` 旧路径仅测试（生产已无回退路径）。
- `sched_policy::Admission::capacity` 已退出决策路径，只保留为 `/debug` 容量诊断
  （`capacity_probe`）；`decide` 为 `#[cfg(test)]` 历史对照，生产用 `decide_slots`。
- `StoreHook::Capacity` 不再发 Grant（`capacity_reserve`/`ReserveRequest` 已删）；
  Grant 只经 `commit_admission` 发放，增长复用 `ActiveTurn.grant`。
- 下一刀：`store/src/rel.rs` 拆分（S2/S3 前置）；再统一 engine 的
  `MemoryPlan`/`auto_pool_tokens` 与 store plan。
