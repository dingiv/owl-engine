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

---

## 8. B 组缺陷修复（2026-11-16，复审第二轮）

> 同一工作树 `review/server-sink`；P1 已提交（`104a1547`），本节改动**未提交**。
> 触发：P1 之后复审 server 代码，发现的问题全部属"server 自己的实现"而非边界归属。

### 8.1 判据

P1 把设备/装配/泵切干净后，server 只剩 HTTP/SSE、OpenAI/agent 协议、进程生命周期。
这轮复审确认**没有新的结构性违规**，但实现层有 6 处该响不响 / 该快不快的地方。

### 8.2 修了什么

| 编号 | 问题 | 位置 | 修法 |
|---|---|---|---|
| **H1** | 状态行说谎：`status_text` 缺 502/505，`_ => "OK"` ⇒ 线上是 `HTTP/1.1 505 OK` / `502 OK` | `http.rs:status_text` | 补 502 `Bad Gateway` / 505 `HTTP Version Not Supported`；未登记码回落 `Unknown`（新增响应码必须登记） |
| **H2** | permit 白名单只收 `/debug/queue|/health|/v1/models`，其余一律先占 turn 位 ⇒ 满并发时 `/debug/metrics`、`/debug/launches`、`/debug/perf`、`/debug/log`、`/debug/step` 被并发位自己锁死；`/debug/load` 自己绕 permit 却占一位跑完全程 | `main.rs::serve` | 显式 `PermitScope{Free,Engine,Turn}` 策略表：`Turn` 仅 chat（在会话闸后自取）、`Engine` 仅 `/tokenize`、其余（含全部 `/debug/*`、`/debug/load`、未知路径）`Free` |
| **H3** | 会话闸拿在模板渲染 + 图像预处理**之后** ⇒ 同会话第二个请求白做 ~430ms/张的预处理 | `openai.rs::chat` | 会话 id 计算 + `gates.acquire` 提到 `tpl.render` 之前；permit 仍在闸之后、提交之前 |
| **M1** | 请求定界静默降级：`content_length = parse().unwrap_or(0)`；短体直接 `truncate`；管线化残留字节丢弃；chunked trailer/残留丢弃 | `http.rs::read_request` / `read_chunked` / `serve` | 非法/重复不一致 Content-Length 拒；CL+chunked 同给拒；短体报"截断"；帧后残留报"不支持管线化"；chunked trailer 按语法跳过、终止块后残留拒；`serve` 对帧错误**先回 400 再关** |
| **M2** | 断流收尾 4 处手写（关 sink → 取消 → 等回执），超时 5s 内的两把锁语义没写清 | `openai.rs` | 收成 `close_and_abort()` + `drop_turn!` 宏单一路径；`ABORT_ACK_TIMEOUT` 具名并写清"为何必须持会话闸等回执"（真解在 P4） |
| **M3** | `/debug/queue`、`/debug/step` 回执、`/debug/log` 状态的 JSON 形状内联在 `main::handle` | `main.rs` → `debug.rs` | 搬成 `debug::{queue_json, step_action_json, log_state_json}`，现都可单测 |

### 8.3 证据

```text
cargo check -p server --tests       零错零 warning
cargo test -p server --bin server   113 passed（新增:http 定界/状态行 8 项、permit 策略表 1 项）
cargo test -p server --test config_check  3 红 = 存量（期望 max_context_length=79872 / profile 是 57344），无新增
--check（qwen38_27b-3090.toml）     输出形状不变，exit 0
单 3090（UUID 钉卡，8399，max_concurrency=1，spec off）:
  chat 非流式 / SSE / tokenize / queue / vram / blocks   全通
  H2 证明:流式 turn 占满唯一 permit 时 /debug/metrics 0.10s、/debug/vram 0.16s 返回
  M1 证明(裸 socket): 非法 CL → 400 Bad Request；管线化 → 400；
                      短体 → 400「请求体截断:声明 20B,实收 15B」；HTTP/1.0 → 505 正确文案
  SIGTERM → 0.85s 退出、GPU 归 18MiB
```

### 8.4 这轮**没有**做（仍是 P2–P5）

- `config.rs` 的检查点几何预检（P2，`engine::preflight`）；
- 结构化准入拒绝 + `/debug/launches`/`BlockLedgerRow` 投影收编（P3）；
- 会话串行下沉 / 删 `session_gate`（P4）—— M2 的 5s 等待真解在这里；
- `vision_processor` → `owl_models`（P5）。

### 8.5 复审新发现、留给后续的边界残留

| 位置 | 内容 | 建议 |
|---|---|---|
| `chat_tpl.rs::load_gguf` | server 直接 `Gguf::open` 读 `tokenizer.chat_template` KV —— 模型格式知识漏到协议层 | `owl_models` 出「模板资产」loader（与 GGUF/快照同源） |
| `openai.rs::SINK_CAPACITY` / `main.rs::CHANNEL_CAPACITY` | 事件通道深度与请求通道深度都由 server 定；`deliver().await` 背压直接决定引擎每拍节奏 | 默认值给 engine（server 可覆盖），或明确记为"服务面契约" |
| `debug.rs::r_events` / `recon_json` | 投影签名钉死 `owl_store::StoreReport` / `owl_iface::contract::BlockLedgerRow` | 让 `DiagSnapshot` 给投影中立形状，server 才能真删后端依赖 |

---

## 9. P2 预检下沉（2026-11-16）

> 工作树 `review/server-sink`；改动**未提交**（与 §8 同批）。

### 9.1 形态

新增 `crates/engine/src/preflight.rs`（387 行，含 4 项单测）：

```rust
pub enum CheckItem { Pass(String), Fail(String) }      // 报告文本，不带 [check] 前缀
pub struct CheckLines { pub items: Vec<CheckItem> }    // 顺序即报告顺序

pub struct ModelCheckSpec<'a> { path, format, quant, head_quant, embed_quant,
                                max_context_length, vision, vision_path, profile }
pub fn check_model_dir(spec) -> Result<CheckLines, String>;  // Err 仅配置级（臂分派）
pub fn check_draft_dir(dir: &Path) -> CheckLines;            // DFlash2 / pack-quantized / GGUF
pub fn check_mtp_head(dir: &Path) -> CheckLines;             // mtp.* 键 / nextn.eh_proj
```

engine 内部完成 `load_arm_of → LoadIntent → open_decl → check_geom_against →
视觉声明 → 摘要`（与 boot 同一份解析），server 只：
1. 决定「该查哪个目录」（配置语义：`model.path` 缺省 0.8B 资产、`spec.draft`
   的 dflash2/mtp/dumb 分支、`model.mtp_dir` 覆盖）；
2. 用 `push_check()` 按既有前缀打印。

**server 不再认识** `LoadArm` / `LoadIntent` / `open_decl` / `pack-quantized` /
`mtp.*`——`config.rs` 净减 ~130 行（`check_model_decl`/`check_draft_dir`/
`check_mtp_head` 三个函数 162 行整体删除，换 33 行排版 + 26 行调用）。

### 9.2 硬门：`--check` 输出逐字不变

基线在改动**前**用同一台机器抓取（`/tmp/chk-before/`），改后重跑 diff：

```text
IDENTICAL full.out
IDENTICAL qwen38_27b-3080_tp2.out
IDENTICAL qwen38_27b-3090.out
IDENTICAL qwen38_27b-4080.out
IDENTICAL qwen38_27b-gguf_3090.out
IDENTICAL qwen38_27b-gguf-iq3_3090.out
IDENTICAL bad-awq-gguf.out      （format=awq + .gguf 路径，exit 1）
IDENTICAL bad-bf16-draft.out    （z-lab BF16 草稿，exit 1）
```

其余门：`cargo test -p owl-engine --lib` **244 passed**（含 preflight 4 项）；
`cargo check -p server --tests` 零错零 warning；`config_check` 3 红 = 存量。

### 9.3 真机：双 3090 Ti TP2（8398，UUID 钉卡，spec off）

```text
起机 27s（TP2 装配 25.6s；rank0/rank1 staging 自证 err=0；捕获态 Captured）
chat 非流式  → 「中国」（prompt 57 / completion 22 / reasoning 20，usage 真账）
chat SSE     → include_usage 末帧正确（cached_tokens=32 命中校正）
tokenize     → count=2 / max_model_len=8448
/debug/blocks → 逐 rank（3050 块，dup_ids/overlaps 空，clean）
/debug/vram  → recon only_block=13，名字全可解释：tp.peer.r0 / tp.staging.r0 / selfcheck.marlin.*
断流弃置      → FIN 分支命中：「弃置 turn Some(3)」+ executor「投递失败 ⇒ 弃置」
H2           → 在飞唯一 turn 时 /debug/metrics 0.14s、/debug/vram 0.33s
SIGTERM      → 1.28s 退出、两把卡锁释放、两张 3090 归 18MiB
```

> 生产 8135（双 3080 TP2）全程未动；期间观察到它被外部重启两次
> （`499133 → 557365 → 578445`，cgroup 在 `session-2.scope`，非 systemd），
> 与本轮无关。

### 9.4 剩余（P3–P5）

P3 结构化准入拒绝 + `/debug/launches`/`BlockLedgerRow` 收编；P4 会话串行下沉；
P5 视觉预处理 → `owl_models`。另见 §8.5 的三条新边界残留。

---

## 10. P3 准入结构化 + 诊断投影收编（2026-11-16）

> 工作树 `review/server-sink`；改动**未提交**（与 §8/§9 同批）。

### 10.1 结构化准入拒绝（去掉错误串解析）

引擎新增（`crates/engine/src/serve/turn.rs`）：

```rust
pub struct AdmissionReject { code: &'static str, retry_in_ms: Option<u64>, detail: String }
impl Display for AdmissionReject   // 仍是 "admission:<code> retry_ms=<n> —— <detail>"（日志不变）

pub struct TurnError { message: String, reject: Option<AdmissionReject> }
impl Display for TurnError          // 仍是 message（日志/既有断言不变）

pub enum SubmitError { Rejected(AdmissionReject), Engine(ModelError) }   // running.rs
impl RunningEngine {
    pub fn submit_session_opts(..)    -> Result<..>        // 旧面，43 个调用点零改动
    pub fn submit_session_checked(..) -> Result<.., SubmitError>   // 新面（service 用）
}
```

事件形状：`TurnEvent::Failed { turn, err: TurnError }`、`StreamEvent::Rejected { err: TurnError }`
—— 字段名不变、`Display` 不变，所以**引擎 50+ 处测试模式零改动**（`{ turn, err }` 仍匹配）。

三个产生点全部结构化：`submit` 的 `no_slot`（带 `retry_ms` EWMA）、调度准入的
`no_memory`/`queue_timeout`/`session_busy`、`reject_queued_turn` 的池墙拒。

服务面（`openai.rs`）删除 `admission_code()` / `admission_retry_ms()`（**不再解析任何错误串**），
改为 `err.reject()` + `retry_after_secs(&AdmissionReject)`（`ceil(ms/1000)` 夹 1..=30，
无估计 = 5s，与旧口径逐值一致）。`error_json_for_reject` 的"超预算 → 400
context_length_exceeded"路径不变（那是另一类拒绝，仍看文案标记）。

### 10.2 诊断投影收编（server 摘掉 `owl-cuda` / `owl-iface`）

`service/diag.rs` 新增引擎面稳定形状：

```rust
pub struct BlockRow { id, bytes, addr, ledger, name }        // From<owl_iface::..BlockLedgerRow>
pub struct LaunchRow { seq, t_us, cmd, stream, name, grid, block, smem, bytes, ids, out_id, ptrs }
pub struct LaunchTrace { enabled, capacity, in_ring, total, rows }
pub fn launch_trace(n, kind) -> LaunchTrace                  // 进程级 trace 环，零引擎往返
```

`DiagSnapshot.blocks` 改为 `Vec<(u8, Result<Vec<BlockRow>, String>)>`；
`debug.rs` 只做 JSON。`apps/server/Cargo.toml` 删 `owl-cuda` / `owl-iface` 两个依赖，
build-time 代际 feature 改由 `owl-engine` 代转（engine 新增 `arch-*` / `fi-*` / `cupti`
转发，单源仍是 `owl-kernels`）。`owl-store` 仍留（`StoreReport` 尚未投影中立化，见 §8.5）。

### 10.3 证据

```text
cargo test -p owl-engine --lib                    244 passed
cargo test -p owl-engine --test store_admission_gate  7 passed（新增 no_memory 结构化断言）
cargo test -p server --bin server                 114 passed（admission 测试改为结构面）
cargo check -p server --tests                     零错零 warning
--check 6 档 + 2 坏例                             仍逐字 IDENTICAL
```

双 3090 Ti TP2 真机（8398，`trace_ring=256`，`max_queue=1`，`max_concurrency=4`）：

| 场景 | 结果 |
|---|---|
| chat | 200，usage 真账 |
| **并发 4 打 `no_slot`** | #1/#2 200；**#3/#4 → `429` + `Retry-After: 6` + `code:"no_slot"`**（结构穿透，零解析） |
| `/debug/launches/3` | `enabled=true cap=256 in_ring=256 total=30177`，行含 `cmd/stream/ptrs`；`/launches/2/launch` 过滤生效 |
| `/debug/blocks` | 逐 rank 3022 块 `clean:true`，键 = `name/id/bytes/addr/ledger`（BlockRow） |
| `/debug/vram` | 双 rank recon 正常（13 unaccounted = tp.peer/staging/selfcheck，可解释） |
| SIGTERM | 1.26s，双卡归 18MiB |

> 老行为保持：`429` 的 `type`/`code`/`Retry-After` 与旧版逐值一致（旧版是从
> `retry_ms=5xxx` 解出 6）。

### 10.4 剩余

P4 会话串行下沉（删 `session_gate`；M2 的 5s 回执等待随之消失）、P5 视觉预处理 →
`owl_models`；§8.5 三条边界残留（模板 loader / 通道容量 / `StoreReport` 中立化）。

---

## 11. P4 会话串行下沉（2026-11-16）

> 工作树 `review/server-sink`；改动**未提交**（与 §8–§10 同批）。

### 11.1 形态

同会话串行从"服务面闸"搬进引擎队列：

| 位置 | 旧 | 新 |
|---|---|---|
| 引擎 `submit` | `max_batch>1` 且同会话已有活跃/排队 ⇒ **硬拒**（`submit: 会话 … 已有活跃/排队 turn`） | **接受并入队**；顺序在队里 |
| 引擎 `scheduler::schedule` | `queue.front()` 判决 + `pop_front()` 起跑 | `startable_queue_idx()` 选**第一个会话空闲**的候选；判决/出队/起跑都用它 |
| 引擎 `admission_gate` | 判 `queue.front()`（会把还不能起的 turn 误判/误拒） | 判同一个候选 |
| 引擎 `admission_pending` | 与队首比 | 与候选比 |
| 服务面 | `session_gate.rs`（127 行 + 3 单测）在提交前串行 | **模块删除**；`ConnCtx` / `chat()` 去掉闸参数 |
| 提交预算 | 命中估计 = 会话账现值 | **同会话有活跃/排队 turn 时记 0**（会话仍在写 ⇒ 账是未来态；真值由 `CachedTokens` 帧校正） |

`Reason::SessionBusy` 与 `sched_policy` 的会话守卫**保留**：现在它是"候选选择失守"
的不变量兜底（正常路径不可达，单测仍覆盖）。

**断流取消的按会话兜底也补了一刀**：P4 之后同会话可以"活跃 1 条 + 排队 N 条"，
而客户端在 `Accepted` 之前断开时只带 session。旧 `cancel_session` 先取活跃 ⇒
会误杀**上一条**请求。新增 `RunningEngine::session_turns(session)`（活跃在前，
其后按提交序），pump 的按会话取消改为**优先挑接收方已关的那一条**，都不关才回
老口径。

### 11.2 行为变化（有意）

- 同会话并发**不再 429 / 不再被服务面阻塞**，而是进引擎队列（`/debug/queue` 可见）；
- 断流取消的回执等待（`ABORT_ACK_TIMEOUT`）不再是正确性依赖，只用于尽快让取消落地；
- 合批档（`max_batch>1`）下同会话仍严格串行；跨会话照旧合批。

### 11.3 证据

```text
cargo test -p owl-engine --lib      245 passed（新增 same_session_queues_instead_of_rejecting）
cargo test -p server --bin server   111 passed（−3 = 删掉的 session_gate 单测）
cargo check -p server --tests       零错零 warning
config_check                        3 红 = 存量
```

真机（**双 3090 TP2** 8398，`max_batch=1` 与 TP 不兼容）：同会话两个并发请求
→ 中途 `/debug/queue` 拍到 `queue 1 active 1`、`queued[0].waited_ms≈978`，
两个都 **200**，`rejects` 为空（旧行为：第二个会被服务面闸挡住或撞 429）。

真机（**单 3090** 8397，`max_batch=2`，`gdn_slots=4`）：

| 场景 | 结果 |
|---|---|
| 跨会话并发两请求 | `active=2 batch_steps 0→95`（真合批），两个 200，各 3.07s |
| 同会话并发两请求 | 全程 `active` 最大 = **1**（串行），两个 200（2.37s / 1.96s） |
| 同会话排队中客户端断开 | t=2s 拍到 `queue 1 active 1`；先到的那条**不受影响**（200 / `finish=length` / 160 tok），只有断开的 turn 被弃置 |
| SIGTERM | 0.85s，卡锁释放，GPU 归 18MiB |

### 11.4 剩余

P5 视觉预处理 → `owl_models`；§8.5 三条边界残留（模板 loader / 通道容量 /
`StoreReport` 中立化）；§6.2 双闸中"`max_concurrency` 是服务面策略"已在
`PermitScope` 里定调（turn 位），引擎准入仍是硬约束。

---

## 12. P5 视觉预处理 → `owl_models`（2026-11-16）

> 工作树 `review/server-sink`；改动**未提交**（与 §8–§11 同批）。

### 12.1 形态

```text
owl_models::vlm::preprocess                        ← 新增（crates/models/src/vlm/preprocess.rs）
  pub const PATCH/TEMPORAL/MERGE/MIN_PIXELS/MAX_PIXELS
  pub fn smart_resize(h, w, max_pixels) -> Result<(usize, usize), String>
  pub fn from_bytes(bytes: &[u8], max_pixels) -> Result<VlmImage, String>
  pub fn from_image(img: DynamicImage, max_pixels) -> Result<VlmImage, String>
  （内部：PIL BICUBIC 两趟 u8 重采样 / weights / bicubic / clamp / patchify 原样搬）

apps/server/src/vision_processor.rs                ← 394 → 152 行
  pub fn from_data_url(url, max_pixels) -> Result<VlmImage, String>
    = data URL 解析（base64）+ LRU 内容缓存（8 张）+ 调 models::preprocess::from_bytes
```

`crates/models/src/vlm.rs` → `crates/models/src/vlm/mod.rs`（加 `pub mod preprocess;`）。
`owl-models` 新增 `image` 依赖（png/jpeg）；`apps/server` **删掉 `image` 依赖**
（server 只剩 base64）。

### 12.2 门

**像素指纹（搬家门）**：搬家**前**在旧实现（server）上采真图全链指纹，
搬家后固化为常量：

```text
assets/images/bird.png → grid=(1,34,22)  len=1_148_928
FNV-1a(f32 bits) = 0xd44c4b2a274d0a91
```

`vlm::preprocess::parity_exact::bird_pixel_fingerprint` 断言这三项 ⇒ **一个 bit
都没动**。（顺带修了两个陈年死测试：`parity_probe` / `parity_exact` 的资产路径
`../../../../assets/bird.png` 从来读不到，一直在静默 return；现改为
`../../assets/images/bird.png`，真跑。）

```text
cargo test -p owl-models --lib                 330 passed（+5 = preprocess）
cargo test -p server --bin server              108 passed（−4 搬家 +1 bad_base64）
cargo check -p server --tests                  零错零 warning
```

**VLM e2e（答案不变）**：单 3090、`vision.enabled=true`、bird.png 走 data URL、
`temperature=0 seed=7 max_tokens=220`，搬前/搬后（各自冷启实例）逐字节相同：

```text
content = "Penguin"
reasoning = （66 tok，逐字相同）  finish = stop
usage = prompt 249 / completion 69   （搬前搬后一致）
```

### 12.3 剩余

§8.5 三条边界残留（模板 loader / 通道容量 / `StoreReport` 中立化）。
评审卷 §5 的 P1–P5 到此**全部落地**。
