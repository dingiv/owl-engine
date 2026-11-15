# server ↔ engine 边界评审 —— 业务下沉立案（2026-11-15）

> 工作树：`packages/owl-worktrees/server-sink`（分支 `review/server-sink`，base `main@5c774b03`）。
> 对象：`apps/server`（13,000 行 / 14 个模块）。
> 触发：用户裁决 —— **引擎内部特性不应该由 server 层实现**，应下沉到
> engine 的 init / 运行时面；本轮先评审、后分批落刀。
> 结论口径：**能等价、可门控、不改变对外行为的**才下沉；协议/传输/进程生命周期留在 server。

---

## 0. 一句话

server 现在不只是"HTTP + OpenAI 协议壳"，它还**亲手做了三件引擎的事**：

1. **装配与生命周期**（选卡/卡锁/双 rank 并行装载/TP 装配/ready/收尾）——`main.rs`；
2. **泵循环策略**（准入重试、显存压力收缩、取消/弃置、缓存命中校正、遥测快照）——`executor.rs`；
3. **引擎事实与预检**（队列/容量快照、`/debug/vram` 八源拼装、`--check` 的检查点几何校验、图像预处理）——`telemetry.rs` / `debug.rs` / `config.rs` / `vision_processor.rs`。

合计约 **1,900 行（~15%）** 属于引擎域（还不含间接重复：`max_concurrency` 双闸、
引擎错误串解析、server 直连 `owl_cuda`/`owl_store`）。

---

## 1. 判据：谁拥有什么

| 面 | 归属 | 理由 |
|---|---|---|
| HTTP/1.1 帧、SSE、keep-alive、连接生命周期 | **server** | 传输层，与引擎无关 |
| OpenAI / agent 协议（请求解析、`tool_calls` 流、thinking 通道、模板渲染） | **server** | 客户端契约；换协议不动引擎 |
| 进程生命周期（信号、退出码、线程栈、logger/metrics 初始化） | **server** | 操作系统面 |
| **设备所有权**（选卡、卡锁、上下文/线程亲和、释放） | **engine** | 谁持有 CUDA context / 块账，谁负责 |
| **装配**（单卡/TP、账本合流、装载、warmup、图捕获、ready） | **engine** | 全是 A5/A6（预算/池契约）内部事实 |
| **装载前的检查点事实**（几何/量化族/草稿头/MTP 键） | **engine**（preflight） | 与 loader 同一份解析，两处必漂 |
| **调度策略**（准入、排队、取消、压力收缩、会话串行） | **engine** | 状态机在引擎里；server 再做一套 = 双真源 |
| **引擎事实快照**（队列/容量/账/池/会话资产） | **engine**（生产快照） | server 只做 JSON 形状 |
| 图像预处理（smart_resize/patchify/归一化） | **models**（视觉塔契约的产地） | 与 `preprocessor_config.json` 同源，引擎零图像知识是伪命题 |

---

## 2. 现状地图（模块 × 判定）

| 模块 | 行数 | 判定 | 说明 |
|---|---|---|---|
| `http.rs` | 417 | ✅ server | 帧解析/回写、chunked/SSE；协议无关 |
| `openai.rs` | 2103 | ✅ server（含 2 处异味） | 协议面；但**解析引擎错误串**（§6.2）应换结构化 |
| `toolcall.rs` | 3051 | ✅ server | 生成文本 → OpenAI `tool_calls`；协议语义 |
| `chat_tpl.rs` | 484 | ✅ server（加载可下沉） | 渲染 = 协议；模板资产来自模型（GGUF KV / 快照） |
| `agent_trace.rs` | 533 | ✅ server | agent 诊断环（prompt/usage/解析轨迹） |
| `repeat_guard.rs` | 255 | ✅ server（可另立 crate） | 跨轮重复启发式，只可见不改生成 |
| `load.rs` | 626 | ✅ server（dev 工具） | 内源灌流测量，本质是客户端 |
| `shutdown.rs` | 89 | 🟡 混合 | 信号/退出码 = server；**设备收尾（`shutdown_tp`）应在 engine** |
| `main.rs` | 1372 | 🔴 **~560 行应下沉** | 选卡后置：卡锁、engine 线程、装配、ready、TP 收尾 |
| `executor.rs` | 628 | 🔴 **~500 行应下沉** | 泵循环策略 + 引擎事实快照 |
| `session_gate.rs` | 127 | 🔴 **应下沉** | 引擎对同会话并发**硬拒**，串行契约却由 server 补 |
| `telemetry.rs` | 108 | 🔴 **应下沉** | 纯引擎队列/容量事实的缓存 |
| `config.rs` | 1174 | 🔴 **~550 行应下沉** | 选卡（~150）+ `--check` 检查点校验（~400） |
| `vision_processor.rs` | 394 | 🔴 **应下沉（到 models）** | 图像预处理 = 视觉塔语义 |
| `debug.rs` | 648 | 🟡 数据面下沉 | JSON 投影留 server；八源拼装归 engine 快照 |

---

## 3. 下沉候选（按收益 / 风险排序）

### A. Engine init：设备 + 装配 + 生命周期（**最高优先**）

**现状锚点**
- `apps/server/src/main.rs:241-300` —— 卡锁（`LockGuard::acquire`）；
- `main.rs:620-681` —— `engine_of` / `engine_of_tp` / `engine_cfg` / `load_shard`；
- `main.rs:674-855` —— `executor_main`：单卡 / TP 双 rank **并行装载**、`TpEngine::assemble_with`、
  `vram_phase` 分相、`ready` 回报、`shutdown_tp` 收尾；
- `apps/server/src/config.rs:569-690` —— `resolve_devices` / `pick_distinct`
  （BDF/UUID 匹配、卡锁避让、`nvidia-smi` 空闲率过滤、`owl_cuda::ffi::device_rows`）。

**问题**
- server 直连 `owl_cuda::ffi`（选卡）与 `owl_cuda::GpuClient`（构造）—— app 层知道后端类型；
- 卡锁的持有/释放跨 `main.rs` 与 `shutdown.rs` 两处，TP 档每 rank 一把，
  释放顺序与引擎线程的析构顺序有隐式耦合（注释里全是纪律）；
- 「装载白名单 / 臂 → 装载器 / 单卡与 TP 两条装配路径 / 失败回滚」是**引擎知识**，
  进来的第三个入口（测试、CLI、将来的 headless）必复制。

**建议接口（engine 侧新模块 `owl_engine::boot`）**

```rust
pub struct Devices { /* ordinal + uuid，构造即校验 */ }

/// 选卡：配置 + 卡锁 + 实占内存过滤（现在 server 的三段合一）
pub fn resolve_devices(par: &ParallelCfg, min_free_ratio: f64) -> Result<Devices, String>;

pub struct BootSpec {
    pub devices: Devices,
    pub engine: EngineConfig,          // ordinal 由 boot 逐 rank 回填
    pub model: PathBuf,
    pub arm: LoadArm,
    pub intent: LoadIntent,
    pub tp: TpOpts,
    pub lock: LockInfo,                // 端口/角色
}

/// **在调用线程内**完成：锁 → 装配（单卡/TP）→ 装载 → run；
/// 返回的 `Booted` 持有锁与收尾责任（Drop 保序：先 shutdown_tp 再放锁）。
pub fn boot_in_thread<D: DeviceClient>(spec: BootSpec, face: impl Fn(usize) -> Result<D>)
    -> Result<Booted<D>>;

impl<D> Booted<D> {
    pub fn running(&mut self) -> &mut RunningEngine<D>;
    pub async fn shutdown(self) -> Result<()>;
}
```

`RunningEngine` 非 Send（A3 / 设备线程亲和），所以 server 仍保留 **"开线程 + ready 信号"**
（进程线程模型 = server 职责），但线程体缩成一次 `boot_in_thread(...)`。

**收益**：main.rs 少 ~500 行；选卡/锁/装配对测试与 CLI 可复用；server 不再 import `owl_cuda::ffi`。

---

### B. 泵循环策略（`executor.rs`）

**现状锚点**：`run_executor`（`apps/server/src/executor.rs:166-520`）逐条包含：

| 段 | 行 | 内容 | 归属判定 |
|---|---|---|---|
| 取消/弃置 | 239-267 | `active_turn_id` + `sink.is_closed()` ⇒ `cancel_turn` | 🟡 断流信号来自 server，**弃置语义在 engine** |
| 准入重试 | 291-300 | `admission_pending` ⇒ sleep 5ms 再 pump | 🔴 引擎排队策略 |
| 压力收缩 | 361-392 | `PoolExhausted` ⇒ `relieve_pressure` ⇒ 继续 | 🔴 引擎显存策略 |
| 缓存校正 | 268-287 | `active_cached_len` ⇒ `SinkEvent::Session` | 🔴 引擎事实，不该由 server 轮询 |
| 遥测 | 288-315 | `queue_len/active/batch_steps/queued/reject/wait_hist` | 🔴 引擎事实 |
| perf dump | 341-350 | metrics dump | 🟡 诊断，可留 |
| 致命判据 | 393-404 | `ModelError::is_fatal` | 🔴 引擎错误分类 |

**建议**：引擎暴露**一步泵 + 事件**（已是 `pump()`），把"策略"收成
`owl_engine::pump::Driver`：

```rust
pub struct StepOutput { pub event: TurnEvent, pub diag: QueueFacts, pub cancelled: Vec<u64> }

impl<D> RunningEngine<D> {
    /// 收件 → 准入 → 一步 → 事件；取消由 `submit_epoch`/`cancel` 显式驱动。
    pub async fn step(&mut self) -> Result<StepOutput>;
}
```

server 保留：`EngineReq` 通道、`SinkEvent` 扇出、HTTP 写失败 ⇒ 调 `cancel`。
取消的**判据**（断流）在 server，**语义**（abandon vs cancel、块链/快照处置）在 engine。

---

### C. 引擎事实快照（`telemetry.rs` + `executor.rs` 的 `VramReport` 组包）

**现状**：`executor.rs:521-600` 一次拼 8 个引擎调用（`handle_probe`/`block_ledger_probe`/
`pool_probe`/`store_report`/`snap_evict_preview`/`session_assets`/`capacity_probe`/`mem_info`），
`VramSnapshot` 类型也长在 server；`telemetry.rs` 是第二份（每拍队列事实）。

**建议**：

```rust
pub struct DiagSnapshot { /* store 报告 + 块账 + 池 + 会话资产 + 容量 + 队列 */ }
impl<D> RunningEngine<D> { pub async fn diag_snapshot(&mut self) -> DiagSnapshot; }
pub fn queue_facts<D>(&RunningEngine<D>) -> QueueFacts;   // 每拍零等待
```

server 只留 `debug.rs` 的 JSON 形状（现在已经在做）。

---

### D. 预检（`--check` 的检查点几何校验）

**现状锚点**：`config.rs:721-1127`
- `check_report`（意图/内存 + 模型/草稿校验）；
- `check_model_decl`（`open_decl` + `check_geom_against` + 视觉声明）；
- `check_draft_dir`（DFlash2 architectures + `pack-quantized` 判定）；
- `check_mtp_head`（检查点 `mtp.*` 键）。

**问题**：这些是 **loader 的同一份事实**（`LoadArm`/`LoadIntent`/`decl`），
规则一改（新增格式、草稿族、MTP 变体）就要在 server 同步 —— 历史已发生
`format=awq + .gguf 路径` 的"检查通过、启动失败"。

**建议**：`owl_engine::preflight::check(&ResolvedConfig) -> Report`；
intent/内存段可留 server（那是配置语义），模型/草稿段整段下沉。

---

### E. 视觉预处理（`vision_processor.rs` → `owl_models`）

**现状**：smart_resize / PIL 双三次 / patchify / 归一化，394 行，产出 `VlmImage`。
引擎"零图像知识"的边界是对的，但**边界应画在 models**：`preprocessor_config.json`
的语义属于视觉塔契约；现在任何人想用 owl 的 VLM 能力当库，都得把 `apps/server` 拖进来。

**建议**：`owl_models::vlm::preprocess::{preprocess_image, ImagePreprocCfg}`；
server 只做 base64/URL 解码与 `TurnImage` 组装。**独立一轮**（有像素级对拍门）。

---

### F. 准入/错误语义（`openai.rs` 的字符串适配）

**现状**：`admission_retry_ms` / `retry_after_secs` / `admission_code` /
`is_budget_reject`（`openai.rs:573-600, 1493-1527`）**解析引擎错误串**：

```text
"admission:no_slot retry_ms=137 —— 在飞 8 已达上限 8"
```

**问题**：错误串 = 人读文本，被当协议用了；改一个字就静默改变 HTTP 语义。

**建议**：

```rust
pub struct AdmissionReject { pub code: AdmissionCode, pub retry_in: Option<Duration>, pub detail: String }
impl ModelError { pub fn admission(&self) -> Option<&AdmissionReject>; }
```

`TurnEvent::Failed` 携带结构化拒绝；server 只做 `429`/SSE 映射。**与 B 同批**。

---

## 4. 明确**不**下沉的（及理由）

| 模块 | 理由 |
|---|---|
| `openai.rs` / `toolcall.rs` / `http.rs` | 客户端契约与传输；换协议/换前端不应动引擎 |
| `chat_tpl.rs`（渲染） | OpenAI messages → prompt 的**协议语义**（tools/thinking/历史工具轮）；模板**加载**可走 models API |
| `agent_trace.rs` | agent 面诊断（prompt/usage/解析轨迹），引擎不感知 agent 词汇 |
| `repeat_guard.rs` | 跨轮固定点启发式；将来若要动作（注入/截断）再单独立 crate |
| `load.rs` | 本质是内源客户端/压测驱动，不是引擎特性 |
| `shutdown.rs`（信号/退出码/线程栈） | 进程生命周期；只有设备收尾一段归 engine（见 A） |

---

## 5. 建议的分批（每批都有"行为等价"门）

| 批 | 内容 | 风险 | 门 |
|---|---|---|---|
| **P1** | `boot`：选卡/卡锁/单卡+TP 装配/装载/收尾 下沉（A） | 低（搬家 + 顺序不变） | 现有 `apps/server/tests/e2e.rs` 全绿；`--check` 输出逐字不变；TP2 起机日志同构 |
| **P2** | `preflight`：`--check` 的模型/草稿段下沉（D） | 低 | `--check` 对 5 个常驻档 + 坏例（AWQ+gguf / BF16 草稿）输出逐字不变 |
| **P3** | `pump` 策略 + 结构化准入错误 + `diag_snapshot`（B/C/F） | 中（调度语义） | 单测：取消/压力/准入重试逐条；真机：断流、池压、429 重试三条 e2e |
| **P4** | 会话串行下沉（引擎排队替代 server 闸，`session_gate` 删） | 中高（行为变化） | 同会话并发提交不再 409/硬拒，而是排队；合批档 A/B |
| **P5** | 视觉预处理 → `owl_models`（E） | 低 | 像素对拍（现有 `vision_processor` 单测迁移）+ VLM e2e 答案不变 |

**量化目标**：P1–P3 后 `apps/server` 14 模块 → 11（`session_gate`/`telemetry` 删，
`executor` 缩为通道扇出），代码量 −1,200 行左右；`main.rs` 从 1,372 → ~800。

---

## 6. 顺带的边界异味（不阻塞，但建议同批清理）

### 6.1 server 直连后端/存储类型

`apps/server/Cargo.toml` 依赖 `owl-cuda` / `owl-store` / `owl-iface`；
`main.rs` 的 `DbgFace` 手工拼 `Arc<OnceLock<Arc<tokio::Mutex<GpuFace>>>>`，
`config.rs` 用 `owl_cuda::ffi::device_rows()` 选卡，`debug.rs` 读 `owl_cuda::trace`。

A/C 落地后：`DbgFace` 应变成 `engine::diag::FaceHandle`（类型与克隆口由引擎给），
`owl_cuda` 依赖可从 server 的 `[dependencies]` 移除（只剩 e2e/dev 需要）。

### 6.2 双闸

server `Semaphore(max_concurrency)` + 引擎 `admission`（`n_slots`/排队）。
需要明确：**服务面并发上限**是 server 策略还是引擎策略？现状是两层都限，
合批档的 `/debug/load` 读数要同时看这两层（`load.rs` 头注已承认）。
P3/P4 时一并裁决（建议：server 只做连接/请求面限流，turn 准入全归引擎）。

---

## 7. P1 落地（2026-11-16）：engine init + 线程 + 泵

> 用户裁决（2026-11-15）：**"初始化函数初始化之后直接给一个 running engine；
> 线程由 GPU engine 内部管理，与外面的 server 无关"** —— 采纳，按
> `owl-cuda` 的 `GpuServer`/`GpuClient` 同形实现。

### 7.1 形态

```text
owl_engine::EngineService::boot(BootSpec)          ← 一次调用
   └─ 内部 spawn "engine-service" OS 线程（32MiB 栈）
        选卡（resolve_devices）→ 卡锁 → 单卡/TP 装配 → 装载（TP 双线程并行）
        → run（warmup + 图捕获）→ 武装调试门 → 就绪
        → 泵循环（收件/取消/压力收缩/缓存校正/每拍事实）
        → TP 收尾 → 释放卡锁 → join
   ↑ 返回 EngineClient（Send + Clone；chat/cancel/tokenize/diag/queue_facts）
```

- **server 不再有**：GPU 卡锁、executor 线程、ready 信号、单卡/TP 装配、
  选卡（`owl_cuda::ffi`）、泵策略、遥测单元；
- 新增 `crates/engine/src/service/{mod,boot,devices,pump,diag}.rs`；
- 删除 `apps/server/src/{executor,telemetry,shutdown}.rs`；
- server `[dependencies]` 仍在用 `owl_cuda` 的地方只剩 `/debug/launches`
  的**投影**（`owl_cuda::trace`）与 `BlockLedgerRow` 类型 —— 属 C 组（诊断快照），
  下一批收编。

### 7.2 行为等价证据（真机，3080 TP2 生产实例未动）

| 场景 | 结果 |
|---|---|
| 单 3090（UUID 钉卡）起机 | 选卡/装载/图捕获全在 engine；`/health` 6–16s 就绪 |
| 非流式 chat | `2+3 等于 5。`（reasoning 28 tok；usage 真账 57/37/94） |
| 流式 SSE | role 帧 + reasoning delta 正常 |
| `/tokenize`（raw） | `count=2 / max_model_len=8448` |
| `/debug/queue` | `queue_len/active/batch_steps/capacity`（泵每拍写 FactsCell） |
| `/debug/vram` | store 三层账 + 块账房 + 池 + 会话资产 + 设备余量（同一刻取） |
| `/debug/blocks` | 4108 块 / `clean:true`（逐 rank 投影） |
| SIGTERM | **2s 退出**，卡锁释放，GPU 归 18MiB |
| TP2（双 3090，`mode=tensor`） | 并行装载 + `TpEngine::assemble_with` + chat + diag 全通；SIGTERM 2s，`shutdown_tp` 无 WARN |

### 7.3 修掉的一个真问题

初版 `pump` 只在**收到请求后**检查停服信号 ⇒ 空闲时 `rx.recv()` 永不返回，
SIGTERM 后主线程 join 挂死（实测 30s 未退，只能 KILL）。修法：入口 `recv`
改成 `tokio::select! { stop.changed() => 收摊, r = rx.recv() => ... }`
（请求方会长期持 `EngineClient` 克隆 ⇒ 只靠通道关闭判停服不可行）。

### 7.4 未做（P2 起）

- `--check` 的检查点几何校验（D 组）→ `engine::preflight`；
- 结构化准入拒绝（F 组，去掉 server 解析引擎错误串）；
- `/debug/launches` 与 `BlockLedgerRow` 的投影归 engine（C 组收尾）；
- 会话串行下沉（P4，删 `session_gate`）；
- 视觉预处理 → `owl_models`（P5）。
