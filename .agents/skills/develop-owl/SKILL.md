---
name: develop-owl
description: 开发 owl 推理引擎(Rust + CUDA sm_86)——声明式模型层、GPU-server 契约分离、kernel 清单管理、执行解释器、store KV 虚拟化。要改 owl 的代码/算子/模型/配置/测试,或要量性能、排障、加门时用。Keywords: owl, owl-cuda, owl-kernels, TensorOps, DeviceClient, GpuServer, interpreter, store, Governor ledger, metrics, /debug, owl_test, kernel admission.
---

# develop-owl —— owl 引擎开发

> 本 skill 是**操作手册**。架构事实的权威在 `docs/devlog/`(索引 `docs/README.md`),
> 本文给的是"改代码时要知道的形状与纪律"。与包根 `AGENTS.md` 冲突时以 `AGENTS.md` 为准。

---

## 0. 这个项目是什么

**owl = 在消费级硬件上跑本地模型的推理引擎**(Rust + CUDA sm_86,单卡或双卡 TP)。

**对外定位(卖点)= 显存与上下文友好**:显存是本地 AI 部署的**第一道硬限制**,
所以整套设计围着它转 —— 启动声明预算且终生不超、KV 溢出下沉而不是 OOM、
量化当第一杠杆(权重省下的每 1 GB 都还给上下文)。
**内部架构裁决(不是宣传语)= graph 与 TP 一等公民**(charter A1/A2)。

| | |
|---|---|
| 对外定位 | 消费级硬件(1–2 GPU)上的本地 AI 部署;**显存硬合同 + 上下文虚拟化** |
| 模型面(硬限制) | **只有 Qwen3.8 / Qwen3.5 家族**(27B hybrid GDN+attn 生产 / 0.8B dev);新架构 = spec + 层实现,不是 fork |
| 硬件面 | sm_86 上实测调优;HAL 是**厂商无关闭**(`DeviceClient` + CPU 参考实现 + ROCm 占位)⇒ 第二厂商是"实现"而非"重构" |
| 内部公理 | A1 图治理 · A2 通信一等 · A3 同步隔离 · A4 vendor 纪律 · A5 严格显存预算 · A6 池契约 |
| 不做 | 多节点服务框架、高并发(>8)深水区、非 CUDA 平台、模型广度 |
| 前身 | `xinfer`(MIT,上游 `guoqingbao/xinfer`)—— 冻结为判例库;owl 是架构重写 |
| 里程碑 | M0 骨架 ✅ → M1 图治理 ✅ → M2 TP2 ✅ → M3 模型+DFlash2 ✅ → **M4 冲刺线** |
| 验收线 | 继承 xinfer requirements v1.0:prefill 2178/7656/9k · decode 133/157/165 · ctx 131k/262k/1M |

### 仓库骨架

```
crates/
  sink       Log/metric 下沉口(依赖图叶子,零依赖)
  store      KV/状态三层虚拟化 + 块账(纯状态机,不做 I/O)
  backends/iface   HAL 契约:contract::DeviceClient + DeviceHandle(分配账/命令面)
  kernels    算子分发 + 设备码资产(AOT cubin;Marlin/FlashInfer/XQA/GDN/GGUF)
  backends/cuda    GpuServer(actor)/ GpuClient(DeviceClient 实现)
  backends/cpu     CpuFace(进程内同步实现:测试/对拍/无卡开发)
  models     声明式模型层(装载器 / 解释器 / 层)
  engine     调度 / 图族 / 投机解码 / 会话
  shared     配置面 / 日志 / metrics / testkit
  macros     #[owl_test] 等过程宏
apps/
  server     OpenAI 兼容 SSE 服务(engine actor 独占设备)
  cli        终端 REPL
```

---

## 1. 开发规则(硬约束)

### A. 代码与架构

1. **上层只 import `owl-iface`**;禁止 include 后端内部类型(engine/models/server 都如此)。
2. **一切设备分配过 `owl-cuda` 的 Governor 账本**(A5.2),且 `alloc`/`htod`/`alloc_pinned`
   的 **`name` 是必填参数**(零未命名是硬不变式,`debug_assert` 拒空名)。
   池外原语(cuBLAS workspace 等)必须预分配并登记。
3. **FROZEN 快照只读**:`cu/flashinfer/include/**` · `cu/xqa/**` · `cu/gguf/vendor/**` ·
   `triton/fla/**`(`cu/marlin/vllm_marlin/**` 是 OWNED-derived,可改但要登记)。
   要动 FROZEN 语义 = 走 charter A4 显式认领(登记 + 命名门 + 差异清单),不许顺手 patch;
   清单与落点见 `THIRD_PARTY.md`。
4. 需求基线 = `docs/devlog/canon/requirements.md`;加格式走 `models/new-format-checklist.md`,
   加/换算子走 `kernels/kernel-admission.md` + `kernels/operator-contract.md`。
5. **文件及时拆分**:原则上 ≤1000 行,核心复杂模块 ≤2000 行。
6. **禁全局变量与线程变量**:变量的构造写成构造函数导出,由调用者创建与维护;
   依赖不足就写进函数参数 —— 变量从哪来由调用者负责。
7. **新增配置旋钮** = 结构体加字段 + `from_env` 登记 + 调用点显式传参;
   **生产路径零 env 直读**(唯一合法读取面 = 入口侧构造器族)。
8. env / fs 调用一律走 `owl_shared::{env_reader, file_loader}`;
   `owl-kernels` 的 lib **不依赖 `owl-shared`**(依赖图 `owl-shared → owl-iface → owl-kernels`)
   ⇒ 它要出声走 `klog_*!` + `sink::LogSink` **注入**。
9. **拒启必须响亮**:配置错误 / 资源不足一律结构化 `Err` + 非零退出;
   禁止静默 exit、禁止静默降级(静默拒启 = 运维事故)。
10. **可失败的东西不许吞**:`Result` 一路 `?` 上抛到最外层;`unwrap/expect` 只允许用在
    结构上不可失败处(并写明判据);长驻服务里内部不变量失守 → 优雅收摊,不 `panic`。

### B. 测试与门

11. 用例用 `#[owl_test]` 宏(`crates/macros`);资源清单 `testdata/resources.toml`。
12. **GPU 套件必须 `--test-threads=1`**(并行 = 多 CUDA 上下文捕获竞争 ⇒ 假挂)。
13. 跑门前**轮询等空卡**(`memory.used < 500MiB`)。未守护的 e2e 会**伪造"正确性回归"**:
    ①失败断言每次不同 ②伴随 `[vram-ledger] alloc 失败…驱动 free≈0` ③守护空卡复跑即全绿。
14. **恒等/真值类门必须 spec-off 跑**;spec 改动的硬门 = greedy spec ≡ no-spec **逐位一致**。
15. 只在长输出显形的缺陷,必须用**真机长回合**做门 —— 短探针会给假绿。
16. **args 档覆盖 profile**(深合并只覆盖列出的键):参考臂要显式写全相关键,否则静默继承。

### C. 设备与卡

17. **UUID / BDF 钉卡,禁 CUDA 序号**(数字序会被重解释成混插对);卡位只走 profile 的 `[parallel].device`。
18. 杀进程用 `nvidia-smi --query-compute-apps=pid` + `/proc/pid/exe` 双验;**禁用 `pkill -f`**。
19. 多会话共用卡:起机前查卡、`/proc/vmstat` 看 `oom_kill`、被外部 kill 是常态。
20. **owl 启动 <30s**,超过即有故障;不要 sleep 超过 30s 去等启动。
21. 测量:`packages/llm_speedtest` 统一尺(`--flavor std` 五点三轮)+ 引擎内 metrics 优先;
    禁裸用 wall÷tokens;±20% 逐次方差警觉。

### D. 协作与流程

22. **git 提交须等用户明确指示**,不自行 commit / push。
23. 新 worktree 建在包外同盘的 worktree 根下(`<worktree-root>/<名字>/`);清理只经 git 原生命令。
24. 近期任务与计划写本地台账(不入库);改接口/旋钮/端点/门控 ⇒ 顺手改 `docs/devlog/dev/` 与 `docs/README.md`。

---

## 2. 本仓特有工具

### 2.1 metrics —— 数值账(`owl_shared::metrics`)

**定位**:计时/计数的**数值账**,采样 → 聚合 → 报告。与日志(事件流)严格分工。

```rust
use owl_shared::{timer_start, timer_end, timer_scope, perf_span, counter_inc, counter_add, event};

init_metrics();                       // 程序早期一次(OnceLock;重复 = AlreadyInit 错)
timer_start!("spec.verify.read");     // 严格两两配对(同线程 LIFO 栈)
timer_end!("spec.verify.read");       // 样本带 file!/line!/column! 与父 tag
timer_scope!("engine.phase.prefill", { /* … */ });   // 作用域糖
counter_inc!("spec.degrade.m0");      // 计数
event!("kv.evict", "leg={} n={}", leg, n);
```

**分级**(`probes.perf_level` / `--perf-level` / 在线切):
`off`(零成本基线臂) < `counters` < `host` < **`gpu`**(CUPTI 图内逐核)。
宏在 `debug_assertions` **或** cargo feature `perf` 下展开 ⇒ 生产 release 零成本。

**查询面**:
```bash
curl 127.0.0.1:8135/debug/metrics/spec          # tag 前缀过滤
curl 127.0.0.1:8135/debug/perf/epoch/json       # 树状 + 排他 self
curl 127.0.0.1:8135/debug/perf/level/gpu        # 在线切级
```

**三条判读纪律**(血换来的):
1. **一律用 `epoch` 视图**(`mark_epoch` 每 turn 打标)—— since-boot 被首核/捕获暖机 outlier 独占;
2. **每 tag 样本 cap = 4096**,count 会饱和;判读看 `drop` 字段;
3. **图回放沉默律** —— decode 走捕获图 ⇒ `interp.host.*` 结构性为零;
   decode 耗时分布必须走**设备侧**(`gpu.kern.*` / CUPTI / nsys)。

### 2.2 日志 —— 事件流(`owl_shared::log`)

```rust
use owl_shared::{log_error, log_warn, log_info, log_debug, log_trace};

log_info!("boot", "引擎就绪(捕获态 Captured)");
log_warn!("vram-ledger", "alloc 失败:{} free≈{}", what, free);
```

- 渲染 `[LEVEL] [tag] msg`;**tag 是稳定词汇**(`boot` / `spec` / `vram-ledger` / `fi-plan`…),前缀可过滤。
- 编译门:feature `log`(默认开;不开 = 五级宏空展开,参数不求值)。
- **缺省级别 = WARN**(开箱安静);`--log-level info` 或 `GET /debug/log/level/info` 打开。
- 运行期:`/debug/log/tags/<allow>/<deny>` 前缀白/黑名单;tee 落盘 + 按尺寸 rotation。
- **拒启类错误与就绪横幅用裸 `eprintln!` 恒输出**(静默拒启 = 运维事故)。
- 内核侧(CUDA / `owl-kernels`)不能依赖 `owl-shared`:走 `crates/kernels` 的
  `klog_error!/klog_warn!/klog_info!` + C 侧 `cu/owl_log.h`,统一经 `sink::LogSink` 注入。

### 2.3 `/debug` 面 —— 现场

| 端点 | 看什么 |
|---|---|
| `/debug/metrics[/<prefix>]` | metrics 聚合(前缀过滤;行数上限 512) |
| `/debug/perf[/epoch][/json\|<prefix>]` | 性能树(span 栈 / 排他 self / 导出) |
| `/debug/perf/level/<off\|counters\|host\|gpu>` | 在线切分级 |
| `/debug/log[/level/<n>][/tags/<a>/<d>]` | 在线调级别 / 换过滤 |
| `/debug/vram` | 显存账(账本 + 驱动对账) |
| `/debug/blocks` | 块账房(按名清单 / top-N) |
| `/debug/queue` · `/debug/step` · `/debug/load` · `/debug/launches/<…>` | 调度 / 单步 / 装载 / 发射 |
| `/debug/agent/…` | agent 服务面(会话/轮次) |

排障顺序、杀进程纪律、nsys 用法见 `docs/devlog/dev/debug-tools.md` 与 `profiling.md`。

### 2.4 `#[owl_test]` —— 测试宏

```rust
use owl_macros::owl_test;

#[owl_test]                                // 纯 async 包装(≡ #[tokio::test])
async fn cpu_case(cfg: &OwlConfig) {}      // cfg 自动注入

#[owl_test(gpu)]                           // OWL_TEST_DEVICE(BDF)未设 ⇒ skip
async fn gpu_case() {}

#[owl_test(gpu, hf_parity)]                // OWL_HF_PARITY=1 才跑
async fn parity_case() {}

#[owl_test(gpu, needs = "ckpt:27b-awq")]   // 资源清单缺 ⇒ skip(点名缺哪个)
async fn needs_ckpt() {}

#[owl_test(gpu)]                           // 同步体也收:只做门控/配置
fn sync_probe() {}
```

配置入口 = `OWL_TEST_CONF`(环境档)+ 约定 overlay(`testdata/cases/<crate>/<模块>/<用例>.toml`);
**"没跑" 从不伪装成 "通过"**(skip 文案可见)。全貌 = `docs/devlog/dev/testing.md`。

---

## 3. 设计哲学(五条)

1. **声明 / 执行分离(全栈同一形状)** —— 每层都写成"声明 + 解释器":
   模型层出 `TensorOps` 声明树,解释器翻译成 `LaunchMsg`;kernel 层出 `KernelSpec` 声明,
   registry/device 执行;store 出账与策略,engine 执行搬运。**声明面零执行、零 I/O。**
2. **单一真源 + 清单化** —— 同一件事只允许一个住址:算子元数据只在 `ops/` 声明面
   (`FAMILY`/`TAGS`/参数表全从那里派生,族侧不再手抄);池 dtype 改动必须先**穷举写者**;
   cargo 依赖的真源是 `Cargo.lock`(不在文档里抄第二份)。
3. **显式优于隐式** —— 禁全局/线程变量;依赖从参数进;生产路径零 env 直读;
   配置全字段显式(缺键 = 拒启,不猜默认)。
4. **响亮失败** —— 拒启带结构化错误一次列全;分配零未命名;毒值(`LazyError`)沿声明链流动、
   在解释器边界收割;`Drop` 只做兜底且把"忘记收尾"变成可见事件。
5. **账本先行(A5 严格显存预算)—— 这是产品的第一性原则,不只是纪律** ——
   显存是本地部署的硬限制:预算分解封闭(权重/KV/图/激活/workspace/通信/底价/余量),
   没实测数字拒启;一切分配**带名**过账本 + 周期性驱动对账;超支 = fail-fast(带账本快照),
   不是优雅降级。**进程永不主动 OOM**;上下文不够时走虚拟化(下沉/选择层),不靠调参硬撑。

---

## 4. 五块核心架构

### 4.1 模型层:声明式定义(`crates/models`)

**模型层是纯函数树**,三句话:

```rust
forward(&self, xs, ctx) -> TensorOps      // 无 Result、无 async、无相位分支
```

- **`TensorOps` = 声明链**:反向多叉树上的节点视图;运算 = **往树上追加节点,不执行**;
  数据张量 = `eval` 之后的产物。维度只读声明 shape(`Bytes.len` 不参与语义 —— C1 律)。
- 错误只在**两个边界**存在:构造期装载(`eval_load`)、执行收割(`eval`);
  中间用**毒值** `LazyError` 流动,`is_poisoned()` 可查。
- **层协议双面**:`Module`/`ForwardCtx`(计算)+ `Loadable`/`Weight`/`Want`/`LoaderOps`(装载)。
- 模型规格住 `specs/`(每档一文件,纯参数事实),装载域住 `formats/`。

```rust
// 典型:声明一条链,交给解释器
let y = self.o_proj.forward(&attn_out, ctx);   // 只追加节点
let logits = eval(&y, &mut face)?;             // 边界:翻译 + 发射 + 收割
```

### 4.2 后端层:GPU server(接口 / 实现分离)

```text
models/engine  ──只认识──►  owl-iface::contract::DeviceClient   (契约)
                                   ▲                    ▲
                          GpuClient(异步)        CpuFace(同步)
                                   │
                            GpuServer(actor,独占 CUDA 上下文)
```

- **server 是哑执行器**:不认识任何算子,只认 `LaunchMsg` / alloc / htod / dtoh / sync。
- **组装三分离**:管道(channel)外部创建 → `GpuServer::new(rx, …)`(纯构造,零 CUDA)
  → `GpuClient::new(tx)`(纯句柄)→ `thread::spawn(server.run())`。
- **异步模型**:API 全非阻塞;完成通知走 `cuLaunchHostFunc`(驱动随流序回调,零轮询),
  回调只投递 channel,真正的 finish 由 server 内部派发线程做,经 Ack 回执客户端。
- **同步点隔离(A3)**:全引擎唯一真阻塞点 = logits D2H;tokio 侧零 CUDA 调用。
- **三方关系**:`Device` = 一卡一账本;`Pool` = 分配者(scratch / persistent / peer_shared 三原语,
  kind 错配 = LawViolation);`Buf` = 出生带 `BufToken`。
- 改这块的准入:`crates/backends/README.md`(第三方 CUDA 准入铁律)+ `crates/backends/cuda/README.md`(显存记账 / 四阶段硬预算)。

### 4.3 kernel 层:清单管理(`crates/kernels`)

**"声明面"是唯一真源,执行面只是解释它。**

```text
ops/<域>.rs          impl KernelSpec  ← 单算子声明:参数表 / 形状族 / 池访问 /
                     (零 feature 门)     数值律 / FAMILY / TAGS
cu/<域>/x.cu         .cu 源 + 资产 id(SRC = "cu/…/x.cu")
registry.rs          OpRegistry(名字无关入口)+ KernelFamily(族运行时:assemble/close/run)
device.rs            DeviceRes(资源面)+ Exec(执行引擎:LaunchVal owned 值表)
driver.rs            唯一发射分派口(KernelPick::of::<T>(),零核名字符串)
tag_index.rs         tag → 算子 倒排(运行时显式构造,无全局)
bootcheck.rs/cu_sig.rs  自检基建 / .cu 取证面(源文本、符号、签名互证)
```

- **加一个算子的完整动作**:写 `.cu` → `ops/<域>.rs` 加 `impl KernelSpec`(参数表与
  `.cu` 形参**逐参对位**:`arg_usize`↔`size_t`、`arg_f32`↔`float`;**输出块固定末参**)→
  打 tag → 接线 `driver` → 过 `kernel-admission.md` 的准入清单与门。
- **未建表 = "仅引入"**:物理可发射,但名字路由/白名单/池审计看不到它(boot 报告会列
  "编出但未声明")。
- **资产纪律**:设备码一律 cubin 预编 + 主机侧一律 Rust + **零运行期编译**
  (结构性门 `tests/no_runtime_compile.rs`);改 `.cu` 走 build.rs 增量闸。
- 家族 runtime 的 `assemble` = 构造函数(校验符号/资产 + 资源预钉),`Ok` ⟺ 已准入。

### 4.4 模型层执行:解释器(`crates/models/src/interpreters`)

**声明 → 原语**的翻译层,后端经 face 泛型注入(**静态分发,零虚表**),解释器本身零后端代码。

文件均在 `crates/models/src/interpreters/`(装载域在 `crates/models/src/formats/load.rs`):

| 变体 | 文件 | 职责 |
|---|---|---|
| 计算 | `eval.rs` | `TensorOps` DAG 归约(CSE + 毒值落地 + 断言);入口 `eval` / `eval_ops` / `eval_ops_tap` |
| 装载 | `formats/load.rs` | `Loadable::layout` → 取数/变换/物化(流式分块 + 并发流水);`eval_load` |
| 观测 | `observe.rs` | 节点级单步事件(`Tap` / `StatsTap`)挂在归约点 |
| 生成 | `generate.rs` | 步循环域:`GenSpec` + 逐 step 驱动整模单树(采样/停机) |
| 内存规划 | `planner.rs` | 块池:块粒度 slab 复用(best-fit + liveness;port 自 llama.cpp `ggml_dyn_tallocr`) |
| 参考 | `reference.rs` | 同步参考解释器(`CpuInterpreter`,对拍锚,**永不优化**) |

**改这里的纪律**:新变体按域另起文件,公共词汇上提 `mod.rs`,变体间不互相依赖;
维度推导只读声明 shape;毒值必须在边界收割(结构化报错 + depth 归因)。

### 4.5 store:声明式调度(`crates/store`)

**只做账、身份与策略** —— 东西在哪、这一步该看到什么、放不下时丢谁。
不认模型层、不做前向、不持设备块句柄所有权(所有权归 engine)。

- **四个词**:会话(客户端身份 + 历史)· turn(一次提交)· **活跃 turn**(唯一占不可驱逐资源者)
  · idle 会话(只占可驱逐资产)。
- **两腿独立**:attention 块链(可哈希/可共享)与 GDN 状态快照(不可哈希/一格一会话)
  **两本账各自独立**,由 `rel` 按前缀身份关联并判定召回语义(`Exact` / `ContextOnly`)。
- **四条容量线**:①单请求 dense ②selection 常驻 ③多会话池 ④性能交叉 —— 越线后果见 `lib.rs` 头注。
- **两个旋钮**:boot `gdn_slots`(硬天花板);per-turn `TurnFloor`(`Strict` = 零耐受 ⇒ 禁选择层,不足即响亮拒)。
- **策略是对象,不是分支**:`EvictPolicy`(`Lru` / attention-aware `Lrfu(λ)`)—— 换对象即换取舍;
  准入姿态 `QueuePolicy::{Reject, Queue}`。
- **选择层(B 轴)**:`select.rs` 纯 host 逻辑(索引/打分/选择/装配计划),设备面三件
  (索引采集 / tile 打分 / re-RoPE+scatter)分别落 engine / kernels,接口由类型定死。

---

## 5. 任务 → 落点速查

| 我要… | 去哪 |
|---|---|
| 加 / 改一个算子 | `crates/kernels`:`.cu` + `ops/<域>.rs` 声明 + tag + `driver` 接线;准入清单 `kernels/kernel-admission.md` |
| 加一个模型 | `crates/models`:`specs/`(参数事实)+ `layers/` + `formats/`(装载)+ `interpreters/` |
| 加 / 改配置键 | `crates/shared/src/config/` + `profiles/full.toml`(唯一全量档)+ `docs/devlog/dev/config.md` |
| 加一个门 / 测试 | `#[owl_test]` + `testdata/cases/` overlay;资源清单 `testdata/resources.toml` |
| 量性能 | metrics 打点 → `/debug/perf/epoch` → `packages/llm_speedtest --flavor std` |
| 排障 | `/debug/*` + `--log-level debug` + `docs/devlog/dev/debug-tools.md` 的排障顺序 |
| 改显存预算 / 分配 | `owl-cuda` Governor 账本 + `crates/backends/cuda/README.md`(四阶段硬预算) |
| 动图捕获 | `docs/devlog/graph/*` + `docs/devlog/canon/charter.md` A1(预算前置 / 生命周期律 / 单共享捕获池) |
| 动 TP / 通信 | `docs/devlog/comm/tp2-design.md`(契约与不变量)+ `tp2-log.md`(全账) |
| 改服务面 / agent | `docs/devlog/serving/agent-surface.md` + `apps/server` |

---

## 6. 参考(按需读,别一次全读)

| 文档 | 什么时候 |
|---|---|
| `AGENTS.md`(包根) | 每次会话开头 —— 纪律与现状快照 |
| `docs/README.md` | 找文档(10 主题 / 81 篇索引) |
| `docs/devlog/canon/charter.md` | 公理 A1–A6、所有权边界、借力清单 |
| `docs/devlog/dev/{config,testing,debug-tools,profiling,fixtures}.md` | 上手干活四件套 |
| `docs/devlog/perf/optimization-candidates.md` | 调性能前**先看这张表**(含"禁止重挖"清单) |
| `docs/devlog/perf/pitfalls-general.md` · `pitfalls-cuda.md` | 遇到怪现象先对号 |
| `docs/devlog/graph/graph-model-seam.md` · `memory/store.md` | 改执行路径 / 状态存储前 |
| `THIRD_PARTY.md` | 引入第三方代码 / 加 vendor / 移植上游 kernel 前 |
