# engine crate review 清单（交接给下一绘画）

> **用户裁决（2026-11-05）**：下一阶段最需要**解释**的是 `crates/engine`；
> 本清单给下一绘画做检视入口。**先写解释 / 画调用树，再动代码**。
>
> 当前工作树**未提交**（`owl/AGENTS.md` §D-21：提交须等用户明确指示）；
> 本会话已完成的 5 个批次见 §1，**不要重复 review**。

---

## 0. 一句话现状

engine crate ≈ **40.6k 行**（不含 `tests.rs`/`tests_gguf.rs`），是「槽管理 + 步循环 +
图三态 + 内存预算」的治理层；其中 11 个文件超过 AGENTS 的 ≤2000 行纪律线。
本会话刚把**装载面 / 配置预检 / 附属件 opts / spec q 守卫**收口，engine 的
**运行态核心**（boot → running → turn → step → graph/sched/mem/TP）仍是黑箱。

---

## 1. 已 review 完毕（本会话；别重复）

| 批次 | 内容 | 文档 |
|---|---|---|
| 1 | 装载器 `ModelSource` 统一流（HF/GGUF 一条流） | `loader-refactor-ModelSource-设计-2026-11-04.md` §9 |
| 2 | `--check` 收编 `HfFacts`/`GgufFacts`（删第三套解析） | 同上 §9.5 |
| 3 | GPU 测试口径（`BUILD_GATED_FAMILIES` 孤儿门豁免）+ TP2 测试布置 | 同上 §9.4 |
| 4 | `--check` 与 boot 同臂、GGUF head/embed 能力门、`[model.rope]` 非 GGUF 拒启、HF rope 越表拒 | 同上 §9.6 |
| 5 | `LoadOpts::with_cfg` 单一映射、`open_source` 拆分、spec `q_host/q_dump` 族门 | 同上 §9.7/§9.8 |
| 6 | **P0-1 `engine.rs` + P0-3 `parallel/tp.rs` + P0-2 `running/scheduler` 首过**（boot/run 生命周期、TP2 装配、失败路径、一次请求状态机；findings F-01..F-26） | `engine-crate-导读.md` §2-§8 |
| 7 | **修复落地**：F-16/F-17/F-18/F-19（S0/S1）、F-01..F-04、F-06..F-08、F-11..F-14、F-23、F-25；F-05 部分；F-20/F-22/F-24/F-26 挂账 | `engine-crate-导读.md` §12 |

**工作树里的未提交改动**（20 个 M + 1 个新文件）就是这 5 批，按批次拆 5 笔提交；
不要在一个大 commit 里混入别的会话改动。

---

## 2. 为什么是 engine（解释目标）

### 2.1 体量与热点（`wc -l`，不含测试模块）

| 模块 | 行数 | 角色 | 备注 |
|---|---:|---|---|
| `running.rs` | 4949 | 执行态：会话/轮/队列/pump/事件 | 超过纪律线 2.5× |
| `mem/state.rs` | 3131 | 池/预算/准入/快照/驱逐 | 超线 |
| `engine.rs` | 2942 | `Engine` 门面：boot/run/装配 | 超线 |
| `exec/spec.rs` | 2918 | spec 轮内执行（verify/fold/lookup/q） | 超线 |
| `exec/coverage.rs` | 2437 | 覆盖诊断/选择层 | 超线 |
| `exec/phases.rs` | 2299 | 分相执行 | 超线 |
| `exec/draft.rs` | 2174 | 草稿/propose/q host | 超线 |
| `sched/sched_policy.rs` | 2057 | 调度策略数学 | 超线 |
| `graph/graph_fleet.rs` | 2014 | 图族账/捕获回放 | 超线 |
| `sched/scheduler.rs` | 1994 | 调度器 | 贴线 |
| `graph/graph_plan.rs` | 1205 | 图三态编排 | |
| `parallel/tp.rs` | 1148 | TP2 双 rank 装配 | `unsafe` ×3 |
| `exec/proposal_q.rs` | 1072 | 拒绝采样 q | |
| `spec/spec_runtime.rs` | 1043 | spec boot/mode/草稿 | 刚 review q 守卫 |
| `exec/prefill_chunk.rs` | 1032 | 分块 prefill | |
| `vision/img_evict.rs` | 1001 | 图像段删除 | |
| `loader.rs` | 697 | 装载（已 review） | |
| 其余 | <700 | serve/perf/kernel_cfg/… | |

### 2.2 先产出「解释」，交付物建议

新建 `docs/devlog/dev/engine-crate-导读.md`（或等价），必须让没读过源码的人能答出：

1. `Engine::new` → `Engine::run` 之间，**设备句柄 / 账本 / 池 / 图 / 会话**各由谁持有、何时释放？
2. 一次请求从 `submit_session` 到 `Token`/`Completed`，经过哪些**队列 / 线程 / 锁**？
3. `Session` / `Turn` / `Step` 三层各存什么？谁负责回收 KV 与 GDN 快照？
4. `GraphPlan` 三态（eager/capture/replay）何时切？捕获失败/回放失败各走哪条退化路径？
5. `MemoryPlan` / `BudgetInputs` 每个字段从哪来？`v_card`/`boot free` 在何时快照、会不会过期？
6. TP2 两个 rank 的**账本 / 池 / 图 / 事件**如何合流？哪一步失败允许回滚、哪一步不允许？
7. spec（DFlash2 / MTP）在哪个阶段装载？propose / verify / fold / accept 的边界与所有权？
8. 客户端断开 / 超时 / 取消如何传播？哪些资源在 `Drop` 里释放、哪些必须显式收尾？
9. 哪些路径允许 `panic!`/`expect`？哪些必须 `Result` 上抛到最外层主循环并优雅收摊？
10. 新增一个配置旋钮要改哪些地方（`OwlConfig` → `EngineKnobs` → boot → run → metrics）？

每域至少给一张 **calltree**（用 `calltree` skill 的格式：入口置顶、缩进、分支
冒号结尾、路径前缀），并列出**状态机**与**不变量**。

---

## 3. 分域 review 清单（按优先级）

> 方法：每个域先画 calltree + 列不变量，再找缺陷；**一次只改一个切片**。
> 不要先做"文件拆分"——先证明行为边界，再按边界拆。

### P0-1 `engine.rs`（2942 行）：boot / run 生命周期

- 入口：`Engine::new` / `new_with_ledger` / `on` / `on_with_ledger` /
  `run` / `into_running` / `boot_rank`（2738 行）。
- 核查：
  - 失败路径：`?` 在 `into_running` 里任何一步失败（`MemoryPlan`/预算/`SessionBank`/
    spec/`boot_rank`）时，**设备块 / 账本 / 图 / 池**由谁释放？是否有半装配态？
  - `BootVram` 的 `live_at_new`/`v_card_bytes` 语义；TP 取 `boot0` 的隐含前提。
  - `EngineConfig`/`EngineKnobs`/`OwlConfig` 三者的**单一来源**；有没有"配置改了但
    引擎没读"的字段（逐个字段找消费者）。
  - `Engine::run` 消费 `self`；`LoadedModel` 里的 tokenizer/rope/spec 是否都落到
    `RunningEngine`；端口/并发上限在哪一层。
- 交付：boot calltree + 失败路径表 + 不变量（谁拥有、谁收尾）。

### P0-2 `running.rs`（4949 行）：会话/轮/队列/pump

- 公开面：`submit_session*` / `pump` / `close_session` / `wait_hist` /
  `reject_snapshot` / `read_output_f32` / `face_clone`…
- 核查：
  - Session→Turn→Step 状态机：合法迁移、终态、取消/超时/客户端断开；事件顺序
    （`Token`/`Completed`/`Failed`/`Idle`）与背压。
  - 27 处生产侧 `unwrap/expect`（启发式统计）逐个判定：是否结构上不可失败？
    不是 → 改 `Result` 上抛（AGENTS 硬律）。
  - `TurnEvent` 的队列容量与丢失路径；`pump` 是否可能在 idle/关闭后永久阻塞。
  - 与 `sched`/`exec`/`serve` 的边界：谁决策、谁执行、谁收尾。
- 交付：一次 turn 的 calltree + 状态迁移表 + 取消/断开路径。

### P0-3 `parallel/tp.rs`（1148 行）+ TP2

- 核查：
  - `TpEngine::assemble`：staging 握手、`adopt_ledger`（**必须在装载前**）、
    `boot_rank` 双 rank 对称、AR 计数、draft TP（`tp_draft`）。
  - 3 处 `unsafe`（`copy_nonoverlapping` 的 host staging 视图）：长度/对齐/
    host-mapped 指针生命周期；缺 `SAFETY:` 注释的补。
  - 14 处生产 `unwrap/expect`；错误清理：一 rank 失败时另一 rank 的 face/池。
  - 分片预算：`weights_per_card`、`boot0` 快照、`into_running(peer)`。
- 交付：TP handshake calltree + 失败回滚表 + 双卡实测（见 §6）。

### P0-4 `mem/state.rs`（3131 行）+ `store` 预算

- 核查：
  - `MemoryPlan`/`BudgetInputs` 每一项的来源（实测 vs 保守）；`fixed` 面、
    `pool` 面、`reserve`、`target` 的关系；`v_card` 过期/并发引擎的已知坑。
  - `pool_tokens`/`cap`/`Strict` 准入；超卖（`transient_reserve`）与驱逐
    （LRU/选择层）的交互；`bulk_drop`/`image_evict` 的边界。
  - 快照布局（`snap_bytes × world`）、宿主/磁盘层的交叉校验。
- 交付：预算公式表（每项：来源/单位/何时测）+ OOM 现场判定树。

### P1-1 `exec/{phases,draft,spec}.rs`（≈7.4k 行）

- 核查：一次 decode/prefill/verify 的分相边界；spec 的 propose→verify→fold→accept
  数据流；`q_host`/`q_dump`（刚加守卫）在生产路径的开关；错误如何回到 turn。
- 重点：`exec/spec.rs` 的 1 处生产 `expect`、`exec/phases.rs` 8 处、
  `exec/draft.rs` 4 处（启发式）逐个判定。
- 交付：step calltree + spec 数据流图 + 失败降级路径。

### P1-2 `graph/{graph_plan,graph_fleet}.rs`（≈3.2k 行）

- 核查：三态迁移表；捕获失败/回放失败/退休（retire）路径；图 slab 预算；
  `PlanOutcome` 的消费者；图与 KV/GDN 快照的时序。
- 交付：状态机图 + 「哪些形状可捕获」矩阵。

### P1-3 `sched/*`（≈5k 行）

- 核查：`sched_policy` 的数学（优先级/aging/预算）；`select` 选择层；
  `housekeeping` 的回收节奏；饥饿/公平性；14 处生产 `unwrap/expect`。
- 交付：调度决策表 + 反例（小 max_batch / 长队列 / 取消风暴）。

### P2 其余

| 面 | 文件 | 看点 |
|---|---|---|
| 请求准入 | `serve/session.rs`（676）/`turn.rs`（620） | 队列/取消/超时；17 处生产 `unwrap` |
| 视觉 | `vision/img_evict.rs`（1001）/`vlm.rs` | 24 处生产 `unwrap`；pad/位置洞 |
| 诊断 | `exec/coverage.rs`（2437） | 是否可能误入生产；成本 O(ctx²) |
| q 数学 | `exec/proposal_q.rs`（1072） | 20 处生产 `unwrap`；拒绝采样正确性 |
| 分页/窗口 | `exec/{kvpage,mtp_window,mtp_ring}.rs` | 边界/页对齐/窗口读集 |
| 采样/配置 | `sampler.rs`、`kernels/kernel_cfg.rs`、`perf/` | 配置→行为的单源 |

---

## 4. 横切审计（grep 可复核）

```bash
cd packages/owl

# ① 生产路径零 env(⚠️ 2026-11-05 复核修正:此前的"只有 tests 读 env"是错的)
#    engine.rs::EngineKnobs::from_config / running.rs::StepProbes::from_config 仍读
#    OWL_SCHEME_PROBE / OWL_COV_ANALYZE_EVERY / OWL_ADMIT_ON_FULL /
#    OWL_MAX_ACTIVE_TURNS / OWL_SNAP_MAX / OWL_DEBUG(见导读 F-03)
grep -rn "env_reader::\|std::env::var" crates/engine/src --include=*.rs | grep -v tests

# ② 生产侧 unwrap/expect/panic(启发式:取最后一个 #[cfg(test)] 之前)
#    本次统计(行):running 27* / img_evict 24* / proposal_q 20* / mtp_window 19* /
#    serve/session 17* / tp 14* / sched_policy 14* / phases 8* / … (带 * = 需人工确认
#    是否真生产路径;脚本会把"测试 helper 在文件中部"也算进来)
for f in $(find crates/engine/src -name '*.rs'); do
  cut=$(grep -n '#\[cfg(test)\]' "$f" | tail -1 | cut -d: -f1)
  ...
done

# ③ unsafe(3 处,全在 parallel/tp.rs 的 staging 视图)
grep -rn "unsafe" crates/engine/src --include=*.rs

# ④ 线程/阻塞(3 处 std::thread::scope + 9 处 spawn_blocking)
grep -rn "std::thread::\|spawn_blocking" crates/engine/src --include=*.rs

# ⑤ 文件尺寸债(AGENTS:普通 ≤1000 / 核心 ≤2000)
find crates/engine/src -name '*.rs' -not -path '*/tests*' | xargs wc -l | sort -rn | head -15
```

判定纪律（来自 AGENTS）：

- `unwrap/expect` 只许用在**结构上不可失败**处（切片长度由构造保证、已过有限性
  过滤…），并写判据；否则 `Result` 上抛。
- 长驻服务的内部不变量失守**禁止 panic**：显式 `Result` → 最外层主循环打印 →
  丢弃请求/turn → 优雅收摊；`catch_unwind` 只兜设计外恐慌。
- 依赖前向包含：engine → models/backends；不许反向。
- 新增配置旋钮 = 结构体字段 + `from_config` 登记 + 调用点显式传参；生产零 env。

---

## 5. 验收门（下一绘画照跑）

### 5.1 CPU / 编译

```bash
cargo check --workspace --all-targets
cargo test -p owl-engine --lib -- --test-threads=1     # 当前 222
cargo test -p owl-models --lib                          # 325
cargo test -p owl-shared --lib                          # 129
cargo test -p owl-kernels --lib                         # 120
```

### 5.2 GPU（串行；`--test-threads=1`）

```bash
# 当前已通过（单 3090 Ti 41:00）
OWL_TEST_DEVICE=41:00.0 cargo test -p owl-engine --lib tests_gguf::gpu_gguf_27b_smoke -- --nocapture --test-threads=1
OWL_TEST_DEVICE=41:00.0 cargo test -p owl-engine --lib tests_gguf::gpu_gguf_27b_vlm_image_smoke -- --nocapture --test-threads=1
OWL_TEST_DEVICE=41:00.0 cargo test -p owl-engine --lib tests_gguf::gpu_gguf_iq4xs_0_8b_vs_q4km -- --nocapture --test-threads=1
OWL_GGUF_TP_DEV_A=41:00.0 OWL_GGUF_TP_DEV_B=42:00.0 cargo test -p owl-engine --lib tests_gguf::gpu_gguf_27b_tp2 -- --nocapture --test-threads=1
```

> TP2 测试前提（2026-11-05 修）：单卡参考放在 **rank1 的卡**上（否则 rank0 的
> boot free 快照会把上一份 18GB 误算进预算）；rank1 装载前用
> `Engine::new_with_ledger(cfg1, rank0.ledger_cell(), 1)`。

### 5.3 起机 / 性能

```bash
tools/bench/boot_time.sh awq-spec     # 3 次热态；load/assemble/key loop ≤ 基线 ×1.05
tools/bench/boot_time.sh awq-nospec
tools/bench/boot_time.sh gguf
```

基线见 `loader-refactor-ModelSource-设计-2026-11-04.md` §5；`wall_to_health` 有
0.5s 轮询量化，判据优先 `assemble` / `load`。

### 5.4 资源/环境限制（别浪费时间）

- **4B GGUF 缺失**（`bartowski/Qwen_Qwen3.5-4B-GGUF`）⇒ `gpu_gguf_4b_*` 与
  HF↔GGUF 交叉对拍 skip；0.8B 替换无效（两个 GGUF 互一致、HF 首 token 不同）。
- **无 sm120** ⇒ `nvfp4` 执行面不可测；只验证了"未编入时孤儿门豁免"。
- 卡位用 BDF（`OWL_TEST_DEVICE`/`OWL_GGUF_TP_DEV_*`），禁 CUDA 数字序。

---

## 6. 下一绘画的交付格式

1. 先交 `engine-crate-导读.md`（§2 的 10 问 + 每个 P0 域的 calltree/状态机/不变量）；
2. 再交 **findings 表**：`severity | file:line | 复现 | 影响 | 建议修法`（按 P0→P2）；
3. 每个修复切片 = 独立测试（最小用例）+ 行为等价门（起机/生成/TP2/性能）；
4. **不要主动 commit/push**；不混入别的会话的改动；`rustfmt` 对模块根会递归
   （用 `--config skip_children=true`，或用 `cargo fmt -p` 后检查 `git status`）。

---

## 7. 已知坑（本会话踩过）

- **共享工作树**：本工作区多会话共用；动手前后看 `git status`，不 `stash/checkout`
  他人的文件；不 `pkill -f`（模式会匹配自身），用 PID 列表。
- **rustfmt 递归**：对 `src/lib.rs` / `src/config/mod.rs` 直接 `rustfmt` 会连带
  重排所有子模块（本会话误伤过 engine/shared，已回退）；用
  `rustfmt --config skip_children=true` 或 `cargo fmt -p` 后核对 diff。
- **GPU 测试**必须串行；空卡 `<500MiB`；起机 <30s，超时即故障。
- **`--check` 与 boot 同源**：源族由 `LoadArm::decl_source()` 决定，不再
  `path.is_file()` 猜；新增源/臂时两处都要走 `LoadArm`。
- **Kernel 装配**：通配激活下 `nvfp4` 声明由 `BUILD_GATED_FAMILIES` 豁免；
  新增"声明恒在、执行面随 feature"的家族必须改
  `registry::tests::build_gated_families_are_exempt_from_orphan_gate`。
- **spec q 旋钮**：`q_dump` 需要 DFlash2 + `q_host=true`；`q_host` 缺省 true，
  非 DFlash2 档只拦"显式 false"。
