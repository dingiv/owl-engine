# engine crate 导读（第一版：boot / run / TP2 装配）

> 2026-11-05。配套清单：`engine-review-清单-2026-11-05.md`。
> 本版覆盖 **P0-1 `engine.rs`** 与 **P0-3 `parallel/tp.rs`**；P0-2
> `running.rs`、P0-4 `mem/state.rs` 与 P1 各域在后续版本续写。
> 纪律：先解释、再找缺陷；findings 表在 §6（只列本轮已核实的）。

---

## 1. 一句话结构

engine crate 的三层：

- **装载面**（`loader.rs` + `models::formats::ModelSource`）—— 已 review，见
  `loader-refactor-ModelSource-设计-2026-11-04.md` §9；
- **构造/装配面**（`engine.rs` + `parallel/tp.rs`）—— 本版；
- **运行面**（`running.rs` + `exec/` + `graph/` + `sched/` + `mem/`）—— 下一版。

`Engine<D>` 是“资源句柄 + 只读配置 + boot 快照”，**不执行**；
`Engine::run` 把模型交给设备侧装配，产出 `RunningEngine<D>`；
`RunningEngine` 才持有池/图/会话/账，由 server 的 `engine-executor` 单线程泵驱动。

---

## 2. Q1：谁持有、何时释放

| 资源 | 持有者 | 创建点 | 正常释放 | 失败路径 |
|---|---|---|---|---|
| GPU 命令线程 / CUDA ctx | `GpuClient`（`DeviceClient` actor） | `spawn_gpu_face` | 最后一个 face 句柄 drop ⇒ 命令通道关闭 ⇒ 设备栅栏退出 | 同左（随进程） |
| 事件账 | `SharedLedger` / `LedgerCell` | `Engine::on`（`shared_ledger`） | 最后一个 `Arc` drop | 同左 |
| 权重/临时块 | `RankModel`（loader 产出） | `ModelLoader` | `LoadedModel` drop / `RankModel` drop | 装载失败时随 `LoadedModel` drop，事件账记 Free |
| KV/GDN/快照池 | `StatePool`（`RankDev.pool`） | `StatePool::alloc` | `retire_rank`（显式）或进程退出 | **无 Drop 收尾**（见 F-05） |
| 图族 + slab | `GraphFleet`（`RankDev.graphs`） | `GraphFleet::assemble` | `graphs.close()`（`retire_rank`） | 崩溃/早退只 drop，不 `close()` |
| staging + VMM 别名 | `TpStaging` / `TpCtx` | `handshake_staging` | `retire_rank` → `PeerRelease`/`HostRelease` | 握手失败仅 host 路径有回滚（F-06） |
| 会话/块账房 | `SessionBank`（host 侧单份） | `into_running` | `RunningEngine` drop（host 账）；设备侧靠 `Release` hook | 失败时随 drop |
| 磁盘层 | `bank.rel.disk` | `attach_disk` | 进程退出；不主动删 | 启动只报不删（用户裁决） |

**结论**：设备侧资源没有 RAII 收尾，正常路径靠 `TpEngine::shutdown` /
`RunningEngine` 的收尾函数（`shutdown_tp`）显式归还；**boot 中途失败一律
靠进程退出回收**。这一点要在上层明确写成契约（F-05）。

---

## 3. boot calltree（单卡）

```calltree
apps/server/src/executor.rs/main
  → engine.rs/Engine::new(cfg)
      engine.rs/spawn_gpu_face()
          GpuClient::spawn_with(DeviceSelector::Ordinal, DiagOpts, debug, log::sink)
      engine.rs/Engine::on(cfg, face)
          owl_store::shared_ledger(tier_budget_of(host/snap/disk))
          engine.rs/Engine::on_with_ledger()
              face.vram_snapshot() → BootVram{free,total,live_at_new}
              owl_store::ledger_entry() → LedgerCell
              DeviceHandle::new(face, Some(cell)).with_sink(...)
  → engine.rs/Engine::loader() → ModelLoader
      .mtp_override(.rope(.vision(.mmproj)))
  → ModelLoader::load_* / slice → LoadedModel
  → engine.rs/Engine::run(loaded)
      loaded.into_parts() → (RankModel, Tokenizer)
      live_before_pool = face.vram_snapshot().live
      weights_bytes = live_before_pool - boot.live_at_new
      engine.rs/boot_rank(face, cfg, rm, tp=None, live_before_pool, t_run)
          boot_knobs_check(cfg, tp)                    // 纯配置；任何设备动作之前
              max_batch×TP/选择层、depth < CONCAT_ROWS_SLOTS
          boot_env_and_spec(face, cfg, rm, tp)
              face.device_arch() / face.sm_count()     // 硬件事实
              model 几何 → env；admit::check_env
              spec_runtime::load_spec_runtime()        // 草稿器装载
              kernel_env 快照（target + draft）
              kernel::kv_manifest_gate
          kernel_cfg::apply(knobs, max_seq, arch)      // 全局族配置（F-09）
          boot_kvmem_check(cfg)                        // 纯配置，但已晚于草稿装载（F-01）
          drain_temp_pool → mem_pool_trim → sync → mem_info
          boot_pool_plan → StatePool::alloc
          boot_pool_floor → derive_turn_ws（select_blocks/dense_floor 唯一产地）
          GraphFleet::assemble（warmup + capture；失败可降级 eager）
          boot_post_check（DFlash2 必须有 verify 图；词表切 × 草稿档）
      engine.rs/into_running(cfg, tok, boot, dev, peer=None, ledger, weights_bytes)
          build_memory_plan → MemoryPlan（fixed：pool 实测 + GraphSlab 实测）
          assemble_budget → BudgetInputs（非 Strict 地板门在这里）
          SessionBank::new（含宿主腿、准入策略）
          attach_pool_peer / set_pool_cap / evict_policy / bulk_drop / index
          attach_snap_layout(_peer) / attach_disk + adopt_disk_boot + adopt_disk_chains
          attach_memory_plan（plan 与 Rel 入参交叉校验）
          image_evict 门 / 消息边界 delimiter tokenize
          RunningEngine{ boot, turns, bank, sel, hk, dev, peer, vlm_cache, pending_out }
      RunningEngine::prewarm_pinned_staging()          // 仅 select_partial
```

关键顺序（不能重排）：

1. **账本先于任何设备动作**（`Engine::on` 造账本；装载即发事件）。
2. **`BootVram` 先于权重分配**（`live_at_new` 是 weights 的基线）。
3. **`kernel_cfg::apply` 在 env 之后、plan/图之前**（族配置一次落）。
4. **staging 握手在 TP 图捕获之前**（捕获期拒 `Peer*` 命令）。
5. **`MemoryPlan` 在池/图之后**（GraphSlab 用实测回填）。
6. **`attach_memory_plan` 在 `SessionBank::new` 之后**（交叉校验入参）。

---

## 4. TP2 calltree（`TpEngine`）

```calltree
parallel/tp.rs/TpEngine::assemble(engines, loadeds)
  probe_caps_pair(a,b) → PeerPlan::plan            // UUID/BAR1/P2P 分档
  engines.into_parts() ×2 → (face, cfg, cell, boot)
  face1.set_dev_rank(1)
  face1.adopt_ledger(cell0)                        // 装载前已合流 ⇒ 幂等复核
  tp.rs/handshake_staging(face0, face1, uuid0, uuid1, layout, opts)
      P2P:   peer_alloc×2 → peer_import×2          // 失败无回滚（F-06）
      Host:  host_alloc×2 → host_attach×2          // 每步失败显式回滚
  Engine::from_handle(cfg.clone(), face_i, cell0.clone(), boot0) ×2
  tp.rs/assemble_with_staging(engines, loadeds, staging0, staging1)
      world==2 / len 校验；rank1 孤儿账判据（cell 同一性）
      v1 护栏：draft×tp_draft、读数探针、两 rank 量化同档
      init_staging_header ×2（host 直写 / H2D + sync）
      baseline_i = face_i.vram_snapshot().live
      weights_i = baseline_i - boot_i.live_at_new
      tokio::join!(
          engine.rs/boot_rank(face0, cfg0, rm0, TpCtx(plan0), baseline0),
          engine.rs/boot_rank(face1, cfg1, rm1, TpCtx(plan1), baseline1),
      )
      装配后自证：读 header，produced==consumed==0 且 err==0
      TpEngine{ ranks, tokenizer(rank0), cfg(rank0), boot(rank0), weights_per_card }
  → tp.rs/TpEngine::run()
      engine.rs/into_running(cfg0, tok0, boot0, rank0.dev, Some(rank1.dev), ledger)
```

TP2 的不变量：

- **两 rank 同一本账**；rank1 必须在**装载前**用
  `Engine::new_with_ledger(cfg1, rank0.ledger_cell(), 1)` 构造；
- **staging 布局对称**（`slot_bytes` 两侧相等），header 在捕获前初始化；
- **装配期零 AR**：warmup 的 allreduce 被摘掉，自证 `produced==consumed==0`；
- **每卡预算各算各的**（`baseline_i` per-device），`weights_per_card` 再摊回单卡；
- **图/池/face 每 rank 自成一族**，执行码同一份、换束重跑。

---

## 5. 失败路径表（装配期）

| 失败点 | 已释放 | 未释放 / 风险 |
|---|---|---|
| `boot_knobs_check` / `boot_env_and_spec` 早期 | face / 账本随 `Engine` drop | 无设备大分配，可接受 |
| 草稿装载后、池分配前（`boot_kvmem_check`） | 草稿权重已装载 | `exit(2)` 直接杀进程（F-01） |
| `StatePool::alloc` | 池未建成 | 临时块随账本/进程 |
| 图捕获中 | 图族内部按租约回收 | 捕获失败的 slab 由 `GraphFleet` 账处理 |
| `into_running`（plan/budget/bank/disk） | `RankDev` drop（池/图 drop） | **未 `graphs.close()` / 未 `PeerRelease`**（F-05） |
| `prewarm_pinned_staging` | 同左 | 已完整装配的引擎被 drop（F-05） |
| TP `handshake_staging` P2P 中途 | host 路径有回滚 | P2P 路径无回滚（F-06） |
| TP rank0 成功 / rank1 `boot_rank` 失败 | rank0 `RankDev` drop | staging 双块未释放（F-05） |
| TP 装配后自证失败 | 两 `RankDev` drop | staging 未释放；图已捕获（F-05） |

---

## 6. findings（本轮已核实）

> 行号 = 当前工作树。S0 = 立刻修（正确性/安全）；S1 = 下一刀；
> S2 = 记债；S3 = 卫生。**未提交**。

| # | 级 | 位置 | 事实 | 影响 | 建议 |
|---|---|---|---|---|---|
| F-01 | S1 | `engine.rs:2684` | `boot_kvmem_check` 对 `select_partial && !select_decode` 调 `std::process::exit(2)`；其余门全是 `Result` | ① 检查发生在 `boot_env_and_spec` 之后（草稿权重已装载）；② `exit` 绕开 Drop/设备收尾；③ 测试/嵌入无法捕获 | 改 `Err(ModelError::Msg)`；纯配置门前移到 `boot_knobs_check` |
| F-02 | S1 | `engine.rs:788-790` | `from_config` 对坏 `OWL_ADMIT_ON_FULL` 直接 `exit(2)`（`from_config -> Self` 无 Result） | 同上；且 `from_config` 是唯一入口构造器，库调用者被强杀 | `from_config -> Result<Self>`（或解析上移到 config loader）；`admission_on_full` 进 `OwlConfig` |
| F-03 | S1 | `engine.rs:754,755,757,809,833,362`；`running.rs:96` | 生产 `from_config` / `StepProbes::from_config` / `Default` 仍读 `OWL_SCHEME_PROBE` / `OWL_SCHEME_MIN_CTX` / `OWL_COV_ANALYZE_EVERY` / `OWL_ADMIT_ON_FULL` / `OWL_MAX_ACTIVE_TURNS` / `OWL_SNAP_MAX` / `OWL_DEBUG`；`snap_max`/`gdn_slots` 有配置键却被 env 覆盖 | 违反「配置单一来源 / 生产零 env」；两份来源静默打架（已知债见 `docs/devlog/serving/config-ux.md` 第 13/349 行） | 全部登记进 config loader；引擎侧只读 `OwlConfig`；诊断 env 至少收进 `probes` 配置 |
| F-04 | S1 | `tp.rs:565-566,598-601`；`tp.rs:650-651,668-671` | `assemble_with` / `assemble_same_device` 重装 rank1 引擎壳时传入 `cfg.clone()` 与 **`boot0`**，丢弃 `_cfg1/_boot1`；随后 `assemble_with_staging` 用 `boot1 = engines[1].boot_vram()` 算 `weights1` | rank1 的 per-card 权重基线用了 rank0 的 `live_at_new`；两卡 baseline 不同时预算错位（且 `device_ordinal` 也被覆盖） | rank1 传 `cfg1` / `boot1`（先 clone）；账本单元仍用 `cell0` |
| F-05 | S1 | `engine.rs:1356`、`1413-1420`、`tp.rs:380-420` | boot/装配失败路径只 drop `RankDev`：无 `graphs.close()`、无 `PeerRelease`；`Engine::run` 的 prewarm 失败也整机 drop | 设备分配/staging 只能等进程退出回收；测试或嵌入调用会累积泄漏；优雅收摊契约缺失 | 加 `retire_rank` 的错误路径调用（或 `TpEngine::assemble*` 的 guard）；把「boot 失败 ⇒ 进程退出」写成显式契约 |
| F-06 | S2 | `tp.rs:~930-960` vs `~840-900` | `handshake_staging` 的 host 分支逐步回滚；P2P 分支 `peer_alloc×2 + peer_import×2` 任一步失败直接 `?` | 半握手泄漏 VMM 块/导入别名；测试反复装配会累积 | P2P 分支补与 host 对称的回滚（`peer_release` 已分配块） |
| F-07 | S1 | `engine.rs:1413-1420` | `assemble_budget` 的「非 Strict 地板需要显式虚拟化」门在 `into_running`，即 **图捕获/池分配之后** | 纯配置错误要付整段 boot 代价才拒（仅直构 knobs 可触发；配置路径已由 `dense_floor_of` 归一） | 前移到 `boot_knobs_check`（与 F-01 同刀） |
| F-08 | S2 | `engine.rs:1112-1119` | 真实 GPU 上 `weights_bytes == 0` 只 `log_warn`，仍继续装配预算 | 违反 A5.1「没有数字的项不许进预算式」；预算过宽 → 运行期池耗尽 | GPU 面（`total_bytes>0` 且非固定池档）改为拒启；仅 CPU mock 放行 |
| F-09 | S2 | `kernel_cfg.rs:38-60`；`tp.rs:376-379` | `kernel_cfg::apply` 写**进程全局** atomics + `FI_*` env；TP2 用 `tokio::join!` 让两 rank 并发 `boot_rank`，各自 apply | 同架构两 rank 值相同（无害）；**混插 3090/3080** 时 `fp8_native`/族配置 last-writer-wins，两 rank 可能按同一份配置跑；并发写全局也无同一性保护 | 至少串行 apply（装配前一次）；跨代际档在 TP 装配门显式拒 |
| F-10 | S2 | `engine.rs:1356`；`tp.rs:426` | 单卡 `Engine::run` 调 `prewarm_pinned_staging`，TP `TpEngine::run` 不调 | `select_partial` + TP2 档首次下沉仍付 ~600ms pinned alloc（TTFT） | TP run 也调；或在 boot 门拒 `select_partial × TP`（若本就不支持） |
| F-11 | S3 | `running.rs:4337,4354` | `(prof || true)` 恒真：`t_loophead`/`t_sched` 永远 `Some`，`note_sched` 每步计时、每 N 步打印，即使 `step_profile=false` | 探针关闭时仍有每步 `Instant::now` 与周期日志；像遗留调试 | 去掉 `|| true`；`note_sched` 受 `prof` 门控 |
| F-12 | S3 | `engine.rs:631` 附近 `digest()` | 摘要硬编码 `@32tok/page`，且并不包含 `depth/max_batch/sampler/image_evict/TP` 等影响行为的旋钮，却宣称「不在这行里的就不影响行为」 | 运维据摘要判断条件会误读；页宽非 32 的档位字符串是错的 | 用 `pool.geom.page`；把 digest 的声明改成「关键子集」或补全字段 |
| F-13 | S3 | `engine.rs:2386` | `env.kv.grp = hq.checked_div(hkv).unwrap_or(0)`：`hkv==0` 静默给 0 | 畸形 spec 下 `OpEnv.grp=0` 会静默走保守/异常路径，而不是拒启 | 返回 `Err`（`hkv==0` 是声明错误） |
| F-14 | S3 | `engine.rs:2746-2772` | `boot_rank` 函数体整体多缩进一层；`#[allow(clippy::arc_with_non_send_sync)]` 落在 `boot_knobs_check` 上（该函数无 Arc）；`#[allow(clippy::too_many_arguments)]` 落在文档注释前 | 属性错位会掩盖真实 lint；缩进误导读者以为还有外层块 | 清属性、平缩进（纯卫生，单独一刀） |
| F-15 | S3 | `engine.rs` / `tp.rs` | `read_staging_header` / `staging_state_of` 的 `try_into().unwrap()` 由 64B 固定数组保证（结构上不可失败） | 无 | 保留；补一句判据注释即可 |

**已核实不是缺陷**（避免下一轮重复）：

- `memory_components()` 不含 `Weights`，`build_memory_plan` 的 `insert(0)` 不会重复；
- `fi_plan_out_on` 以 face 指针缓存，重捕时 `running.rs:1531` 会 `clear()`；
- `staging_header_bytes` 的 `debug_assert` 只覆盖 release 之外，但构造路径有单测；
- `TpEngine::assemble_with_staging` 的 `expect("world = 2 已守门")` 由前置 `world != 2` 分支保证；
- `into_running` 的 `attach_memory_plan` 与 `MemoryPlan` 同源，交叉校验成立。

---

## 7. P0-2 首过：一次请求的完整链路（running/scheduler）

```calltree
apps/server/src/openai.rs/chat()
  → EngineReq::Chat{session,prompt,max_new,opts,sink} ──mpsc(64)──▶ executor.rs/run_executor
      accept() → running.submit_session_opts()
          max_queue 闸（active+queue ≥ max_batch+max_queue ⇒ no_slot）
          同会话闸（max_batch>1）/ VLM 门 / chat_wrap / tokenize
          image_pad_law（pad 律 + 删除识别）/ prompt_positions / msg_snaps
          budget_clamp（装不下先收缩，再拒）
          SessionBank.get_or_create（ephemeral 或键控）
          admission.cap_for → 夹紧 max_new
          TurnLedger.queue.push_back
  → loop { running.pump() }
      pending_events.pop_front()                    // spec 多 token 事件优先
      step_need() → admission_gate()                 // rel.check: Admit/EvictThen/Queue/Reject
      store_hook(Capacity{need})                     // 需求驱动保洁（段边界）
      scheduler.rs/schedule()
          admit_blocked?（admit_wait = 队首）⇒ 不起新 turn
          Admission::decide()                        // 第二套准入判决
              Admit  → capacity_reserve → Grant
              Hold   → 留队首（本拍只推活跃）
              Reject → pop + Failed(pending_events)   // ⚠ 不 burn ephemeral（F-18）
          start_one(turn, grant)                     // host 账：admission/borrow_row/reserve_row
              plan_image_leave_gap | plan_prefix_reuse
              ensure_for_len（虚拟化档只到首 chunk）
              ActiveTurn.push
          Idle / BatchDecode / Prefill / Decode / SpecRound
      execute_begin / execute_prefill / execute_decode / execute_spec_round / execute_batch_decode
      TurnEvent → executor.dispatch → sink（首个事件前补发 cached_tokens）
  → 终局
      complete()  → commit_turn + set_img_spans + Release(TurnEnd) + close_admission
      abandon()   → Release(Abort, publish=Skip) + close_admission
      cancel_turn → 活跃：capture_abort_snap + Release(Abort, publish=WrittenPrefix)
                    排队：出队 + close_admission（⚠ 不 burn ephemeral，F-18）
      Release → Grant 归还 / 链登记（publish）/ 盘索引 / 表释放 / burn ephemeral / 还 GDN 行 / N4
```

状态迁移（显式）：

| 起点 | 事件 | 终点 | 清理 |
|---|---|---|---|
| `queue` | `schedule` Admit → `start_one` | `active` | 借 GDN 行 + admission 登记 + Grant |
| `queue` | admission Reject | `Failed` | 出队；**ephemeral 未焚**（F-18） |
| `queue` | `cancel_turn` | `Failed` | 出队；**ephemeral 未焚**（F-18） |
| `active` | 正常生成完 | `Completed` | `commit_turn` + `Release(TurnEnd)` |
| `active` | `cancel_turn` / 断流 | `Failed` | `Release(Abort, WrittenPrefix)` |
| `active` | `pump` 错误 → `abandon` | `Failed` | `Release(Abort, Skip)` |

不变量：

- `executing` 游标每拍由 `schedule` 设定；单 turn 代码必须经
  `turns.active()/active_mut()`，批动作才允许 `executing=None`；
- **Grant 与 turn 同生共死**：`Release`/`close_admission` 幂等归还；
- **GDN 行与 turn 同生共死**：`start_one` 借、`Release` 还；
- `pending_events` 在 `pump` 最前消费，保证终态帧不会被下一步调度事件穿插。

---

## 8. P0-2 findings（首过）

| # | 级 | 位置 | 事实 | 影响 | 建议 |
|---|---|---|---|---|---|
| F-16 | **S0** | `sched/scheduler.rs:1799-1805` | decode 相位的 `gdn_row` 取 `turns.active.first()`，而本步执行者是 `turns.active()`（`executing` 选中的 turn） | `max_batch>1` + spec：`active.first` 与轮转选中的 turn 不同 ⇒ 用**别的 turn 的 GDN 行**做 verify/decode，状态污染（可达：配置层不拒此组合） | 改 `let gdn_row = self.turns.active()?.gdn_slot;`；补「两活跃 + depth>0，轮转到 turn#2」单测断言 `action.gdn_slot` |
| F-17 | S1 | `sched/scheduler.rs:568-640`；调用方 `1573-1577` | `start_one` 先 `admission.admit` + `borrow_row` + `reserve_row`，随后 `plan_*` / `ensure_for_len` 都是 `?`；调用方只 `release_turn_grants` | 启动失败后：GDN 行不还（`row_owners` 残留）、admission 在册记录不释放；反复失败会耗尽行池/准入账 | `start_one` 内做错误回滚（`return_row` + `admission.release`），或把借行/登记挪到所有可失败 host 工作之后；补注入失败用例 |
| F-18 | S2 | `sched/scheduler.rs:1520-1545`；`exec/turn_end.rs:103-116` | 排队 turn 的两条终局（admission Reject / cancel）只出队 + `close_admission`，不 `bank.burn(session)`；而 `running.rs:2953-2960` 的 `reject_queued_turn` 对 ephemeral **会焚** | ephemeral 会话（`submit_session(None)`）在拒/取消后永久留在 `SessionTable`，host token 账无主 | 两条路径统一走一个 `drop_queued_turn()`（含 ephemeral burn），消除第三套语义 |
| F-19 | S2 | `running.rs:2920-2936`；`sched/scheduler.rs:1444-1450` | `admission_on_full=Queue` 时：`admission_gate` 只设 `admit_wait`，`schedule` 见 `admit_blocked` 直接跳过，active 空 ⇒ `pump` 返回 `Idle`，executor 断开泵循环去等新请求 | 单请求 + 池满：没有下一次 pump，`admission_max_wait_ms` 永不触发 ⇒ 客户端只能靠自己的队列超时；服务端超时语义失效 | 空转时若 `admit_wait` 在场，泵不要返回 `Idle`（或 executor 加定时 tick）；把超时重判放进 `pump` 自身 |
| F-20 | S2 | `running.rs:2867`（`admission_gate`）vs `sched/scheduler.rs:1465-1505`（`Admission::decide`） | 两套准入判决并存：前者看 `rel.check`（地板/腾位/Queue），后者看 `StoreFacts` + `Admission` 状态机（Admit/Hold/Reject）；两套 verdict 的优先级无文档 | 「谁决策」不唯一：同一队首可能被一套 Hold、另一套 Reject；`Queue` 只在其中一套存在；调参时无法从单点推断行为 | 明确分层：`Admission::decide` 只做容量/公平，`rel.check` 只做地板；把两者的 verdict 合成一张表（或合并到 `Admission`），并在导读里写死优先级 |
| F-21 | S2 | `apps/server/src/executor.rs:409-421` | `abandon_turn` 在找不到本 turn 的 sink 时**向所有 sink 广播** `Failed`；`drop_turn` 明确只投递本 turn | 合批/多会话时把 A 的失败终态投给 B 的连接（B 的流被提前终止或被错误标记） | 未知 sink 时只记日志，不广播（与 `drop_turn` 同律）；或按 turn 过滤后丢弃 |
| F-22 | S2 | `serve/session.rs:241-327`；`running.rs:1400` | `SessionTable::lru_victim` 在生产无调用者；`close_session` 也无服务面调用者；会话表只增不减 | 键控会话（任意字符串 → 稳定 id）可无限增长；每会话 `tokens` 历史留在 host，无预算/淘汰 ⇒ 长驻服务 host RAM 随会话数无界 | 接上 LRU 淘汰（或加 `[server] max_sessions`）并把 `tokens` 体量纳入预算；至少把 `lru_victim` 的“未接线”写成显式 TODO |
| F-23 | S3 | `serve/session.rs:113-116` | `guard()` 直接 `self.tokens[..self.cached_len]`；`cached_len ≤ tokens.len()` 是跨方法不变量，未在函数内校验 | 不变量失守时 panic（长驻服务硬律禁止），而不是返回“失配/不可信” | 用 `self.tokens.get(..self.cached_len)`，None ⇒ 返回 false + 一次性 WARN；或返回 `Result<bool>` |
| F-24 | S3 | `running.rs:2585-2670` 等 | 大量 `expect("判过")` / `expect("上面判过")` 依赖「判据与使用之间无 await」；目前单线程成立，但没有注释/类型强制 | 未来在判据与使用之间插入 await/回调就会变成 TOCTOU panic | 对跨 await 的 `expect` 逐个复核；能改成 `let Some(x)=... else { return Err }` 的就改（尤其 disk/snap 分支） |

---

## 9. P0-4 首过：预算公式与来源

`StatePool::alloc` 的池上限（`pool_cap`）：

```text
per_tok = n_full·2·hkv·hd·kv_elem        // 主池（+INT8 的 36·hkv/层/ token 的 scale）
        + draft_pool                     // DFlash2 已改固定 ring ⇒ 0
        + mtp_chain                      // MTP 线性档（ring 档走固定项）
fixed_after = gdn_grid(gdn_slots) + gdn_snap(snap_max) + spec_snap + replay_snap
            + GRAPH_SLAB_BUDGET(320MiB) + WARMUP_MISC_BUDGET(138MiB) + draft_ring
base  = min(free, free − total·(1−target))        // 驱动 free；target 只作上限
cap   = (base − reserve − fixed_after) / per_tok
```

`MemoryPlan → BudgetInputs`（准入面）：

```text
fixed            = Σ MemoryPlan.fixed（weights + GdnGrid + 快照三腿 + GraphSlab(实测) + WarmupMisc + DraftPool）
runtime_floor    = fixed − weights − gdn_grid − draft_pool
V_avail          = v_card − weights − runtime_floor − gdn_row×gdn_slots − draft_pool   // 动态池档
固定池档（kv_face_bytes>0）⇒ 准入走池墙，V_avail 只作报告/诊断
```

要点：

- `v_card` 两次来源不同：`BootVram.v_card_bytes`（`Engine::new`，权重前）与
  `boot_rank` 里 `face.vram_snapshot()`（`drain/trim + mem_info` 后）。前者进
  `MemoryPlan.check_card`（快照语义），后者进 `pool_cap`（新快照）；两行日志不可混读。
- `weights`：单卡 = `live_before_pool − boot.live_at_new`；TP2 = 两 rank 各自
  `baseline_i − boot_i.live_at_new` 后取均值（F-04 影响 rank1 的那一项）。
- `fixed_pool` 恒真（主池预分配）⇒ 准入的实际判据是池页墙 + `rel.check` 地板；
  卡级 `V_avail` 只在 `check_card`/报告里出现。

| # | 级 | 位置 | 事实 | 影响 | 建议 |
|---|---|---|---|---|---|
| F-25 | S3 | `mem/state.rs:2181,2197,2257` | `page_guess = 32` 写死，用于 `ctx_max = cap − page_guess` 与“还剩多少”告警；真页宽由 `kv_paged_policy` 得，可能不是 32（legacy 档 `pol=None` 时 page=seq_tokens） | `ctx_max` 报告与自动池的 `seq_tokens > ctx_max` 判据按错页宽算：小幅多拒/少拒；与后面按真页宽的硬门各说各话 | 先由 `pol` 得到 page，再做 `ctx_max`；去掉 `page_guess` |
| F-26 | S3 | `mem/state.rs:700-742`；`KvBank::alloc_scratch/alloc_xqa/alloc_tables` | `FixedAfter` 不含 attention v2 scratch / verify / batch / XQA / `fi_tables` / `bt`（都在池后分配）；`[vram-gov] fixed_after` 只报已列项 | `vram_target≈1.0` 时 cap 可能高估几十 MiB（通常被 reserve 1GiB 盖住）；budget 报告不是全量 | 加一个 `scratch` 固定项（按 geometry 算）或把 reserve 的覆盖范围写进日志 |

---

## 10. 下一版要回答的

1. **P0-2 `running.rs` 余下**：`submit_session_opts → queue → schedule →
   execute_* → turn_end` 的首过已完成（§7/§8）；剩下 27 处 `unwrap/expect`
   逐条判定、`store_*` 磁盘/快照分支的跨 `await` 不变量、`pending_events` 背压。
2. **P0-4 `mem/state.rs` 余下**：预算公式已过（§9）；剩下 `Strict`/`Pages`
   准入与 `transient_reserve` 的交互、OOM 现场判定树、`pool_cap`/`FixedAfter`
   的 scratch 补齐（F-25/F-26）。
3. **P1**：`exec/spec.rs` 的 propose→verify→fold；`graph_plan` 三态；
   `sched_policy` 的公平性反例。
4. 修复顺序建议：**F-16（S0，一行）** → F-17（启动失败回滚）→
   F-01/F-02/F-07（boot 门前移 + `from_config -> Result`）→ F-04（TP 预算）→
   F-05/F-06（失败收尾）→ F-18/F-19/F-20（排队/准入语义）→ F-03（配置收编）→
   其余卫生。

---

## 11. 验收门（本版结论不改变行为）

本版只写文档，无代码改动。复跑清单见
`engine-review-清单-2026-11-05.md` §5；修 F-01..F-07 时任选最小用例：

- 纯配置拒启：`select_partial=true, select_decode=false` 应 `Err`（不 `exit`）；
- TP2 预算：两卡 baseline 不同（其中一卡先占 1GiB）时，`weights1` 仍按 rank1
  自身基线；
- 失败收尾：注入 `into_running` 失败，断言 `graphs.close()`/`PeerRelease` 被调用
  （或按契约断言进程退出路径）。

---

## 12. 修复记录（2026-11-05，一一落地）

> 纪律：每刀最小行为变更 + 最小用例；本轮修复已单独提交（见 `git log`）。

| 发现 | 状态 | 改动 |
|---|---|---|
| F-16 | ✅ | `sched/scheduler.rs` decode 的 `gdn_row` 改取 `turns.active()?.gdn_slot`；新增 CPU 回归 `tests::multi_active_decode_uses_selected_turn_gdn_row`（修复前 `got 0, want 1`，修复后绿） |
| F-17 | ✅ | `start_one` 的可失败 host 段包成内层计划；`reserve_row`/计划/`ensure_for_len` 失败统一走 `rollback_failed_start`（还 GDN 行 + `close_admission`）；新增 `tests::failed_start_returns_gdn_row`（修复前 `rows_free 1→0`，修复后不变） |
| F-18 | ✅ | 新增 `RunningEngine::drop_queued_turn`（ephemeral burn + `admit_wait` 清理 + `close_admission`），三条排队终局（admission Reject / 准入超时 / `cancel_turn`）共用；新增 `queued_ephemeral_session_is_burned_on_cancel` |
| F-19 | ✅ | `RunningEngine::admission_pending()` + executor 在 `Idle` 且排队中时 5ms 后重泵（不再断流等新请求） |
| F-01 | ✅ | `boot_kvmem_check` 的 `select_partial && !select_decode` 改 `Err`；纯配置门前移到 `boot_rank` 最前（草稿装载之前）；新增 `boot_kvmem_check_rejects_partial_without_decode` |
| F-07 | ✅ | `check_dense_floor_gate` 单源；`boot_knobs_check` 早期调用，`assemble_budget` 复用同一函数 |
| F-02 | ✅ | `EngineKnobs::from_config -> Result<Self>`；`[runtime] admission_on_full`（`reject|queue|queue:<ms>`）收编，删除 `OWL_ADMIT_ON_FULL` 与 `process::exit(2)`；`ServerConfig::from_config/from_resolved` 同步 Result；新增 `knobs_from_config_rejects_bad_admission_on_full` |
| F-03 | ✅ | 删除 engine 生产侧全部 env 读取：`OWL_MAX_ACTIVE_TURNS/OWL_GDN_SLOTS/OWL_SNAP_MAX/OWL_DEBUG/OWL_KV_DUMP/OWL_SCHEME_PROBE/OWL_SCHEME_MIN_CTX/OWL_COV_ANALYZE_EVERY`；新增 `[kvmem] coverage_scheme_*` / `coverage_analyze_every`；8 个 profile 显式补键 |
| F-04 | ✅ | TP2 重装引擎壳时 rank1 用自己的 `cfg1`/`boot1`（不再传 `boot0`）；同卡档同步 |
| F-05 | 🟡 部分 | TP `boot_rank` 失败/自证失败时先 `graphs.close()`（新 `close_graphs_best_effort`）；单卡 `prewarm` 失败走 `close_graphs_on_boot_failure`。**staging 双块的 PeerRelease 仍需 GpuClient 特化面**，泛型路径仍以“进程退出回收”为契约（显式 WARN） |
| F-06 | ✅ | P2P 握手分支补齐与 host 分支对称的回滚（`peer_alloc`/`peer_import` 任一步失败 `peer_release` 已分配块/别名） |
| F-08 | ✅ | GPU 面 `weights_bytes == 0` 从 WARN 改为拒启（A5.1） |
| F-11 | ✅（澄清） | `(prof || true)` → `Some(Instant::now())`；保留“恒开打点”的原意，去掉误导表达式 |
| F-12 | ✅ | digest 用 `env.kv.page` 而不是硬编码 32；文档改成“关键子集”，不再宣称完整清单 |
| F-13 | ✅ | `hkv == 0` 从静默 `grp=0` 改为 boot `Err` |
| F-14 | ✅ | 移除错位的 `arc_with_non_send_sync`；把 `boot_rank` 的文档注移回函数；`boot_knobs_check` 接受真实属性 |
| F-23 | ✅ | `AgentSession::guard` 先校验 `cached_len <= tokens.len()`，失守按失配（回退全量）而不是切片 panic |
| F-25 | ✅ | `pool_cap` 的 `ctx_max` 用真页宽（`pol.page`；legacy = 0），删除 `page_guess=32` |
| F-20 | 🟡 注释 | 已在 `admission_gate` 与 `Admission::decide` 补"两段分工/顺序固定"注释；**状态机合并仍挂账**（改行为风险高） |
| F-22 | ✅ | 新增 `[runtime] max_sessions`（0 = 不限）；`enforce_session_cap` 在新请求前按 LRU 淘汰**空闲**会话（活跃/排队不动）；`SessionTable::{len,lru_victim_excluding}`；新增 `session_cap_evicts_lru_idle`（修复前 len=3，修复后 ≤2） |
| F-24 | ✅ 主要 | 生产路径 5 处 `expect` 改为显式 `Result`/`if let`（3×`submit 已建会话账`、探测 `刚查过`、队列 `刚定位过`）；剩余 `expect` 均为测试 helper、`world==2` 前置门或 64B 固定头（结构上不可失败），已注明 |
| F-26 | ⏳ | `FixedAfter` 未计 attention/XQA/FI 表等池后 scratch；S3，暂以 reserve 覆盖 |

### 验证

```bash
cargo test -p owl-engine --lib -- --test-threads=1        # 228 passed
cargo test -p owl-shared --lib -- --test-threads=1        # 130 passed
cargo test -p owl-engine --test store_admission_gate -- --test-threads=1  # 7 passed
cargo test -p server --bins -- --test-threads=1           # 104 passed / 1 known-skip
# 唯一 server 失败 = chat_tpl::gguf_template_equals_hf_snapshot：4B GGUF 资产缺失（既有）
cargo check --workspace --all-targets                     # 0 error

# GPU 起机门（单卡 3090 Ti 41:00；F-08 权重实测非零）
OWL_TEST_DEVICE=41:00.0 cargo test -p owl-engine --lib \
  tests_gguf::gpu_gguf_27b_smoke -- --nocapture --test-threads=1   # ok
OWL_GGUF_TP_DEV_A=41:00.0 OWL_GGUF_TP_DEV_B=42:00.0 cargo test -p owl-engine --lib \
  tests_gguf::gpu_gguf_27b_tp2 -- --nocapture --test-threads=1     # ok（cos 0.99981 / 24 token 逐字）
```

新增/修改的 profile 键（8 个文件）：

- `[runtime] admission_on_full = "reject"`
- `full.toml` 的 `[kvmem] coverage_scheme_probe / coverage_scheme_min_ctx / coverage_analyze_every = false/0/0`
