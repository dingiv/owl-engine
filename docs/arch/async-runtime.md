# GPU 异步编程模型设计(async-runtime)

> 版本:v0.1(2026-09-23 立项)。上游:A3 同步隔离(charter)、每卡一线程裁决;
> 实现位:`crates/backends/asyncrt`(owl-asyncrt);消费方:engine server(Tokio)、
> runner decode 循环。
> 状态:**设计稿,待用户拍板后实施**。A0 加固补丁(event-tracking)可先行。

---

## 一、问题定义

CUDA 驱动面是**全阻塞 API**:memcpy 等待、事件等待、`synchronize`、捕获事务、
池 H2D 同步——没有一条能安全地跑在 Tokio worker 上。直接混用的后果(全部实测过):

| 症状 | 根因 |
|---|---|
| Tokio worker 被钉死数毫秒~秒 | 阻塞调用占住调度线程 |
| `CAPTURE_ISOLATION` | cudarc event-tracking 给发射挂跨流事件等待(捕获窗内 = 依赖未捕获工作) |
| `begin_capture` 被拒 | pageable host 指针的 memcpy 被驱动经 legacy NULL 流中转,流上留 legacy 依赖 |
| decode 步进延迟毛刺 | 提交时机受调度噪声支配,host 端准备与 GPU 执行无法重叠 |

**解法公理(A3 的结构化形态)**:每卡一条专用 GPU 线程,唯一触碰设备执行面;
async 侧通过命令队列委托执行,以事件回执收结果。

## 二、线程模型(唯一的正确形态)

```text
Tokio 任务(任意数量,任意迁移,零阻塞)
  │  命令(mpsc)+ 回执(oneshot)
  ▼
GPU actor 线程(每卡一条;ctx.bind_to_thread 一次到位)
  │  串行执行;一切阻塞只发生在这里
  ▼
owl-cuda 治理面(池/租约/哨兵/相位——语义不变,宿主换了)
```

三条公理:

1. **设备执行权归 actor 线程独占**。`Arc<CudaDevice>` 不外泄(`spawn` 只返回
   `AsyncDevice` 门面);一切工作必须走命令面——绕过状态机直接用设备 = 架构违约。
2. **多线程共享 context 的三重税**(bind 重绑/驱动 context 锁竞争/多 context
   交替 = 架构死路)由"单线程独占"结构性消灭;async 面共享的只剩 Rust 队列锁。
3. **actor 串行化的是"提交",不是"执行"**。kernel/graph 发射本身异步于 GPU,
   提交即回执;性能预期见 §八(诚实条款:actor 不创造 GPU 并行度)。

## 三、actor 状态机

```text
Booting ──(bind_to_thread 成功)──▶ Ready ──(Close 排空)──▶ Closing
                                      │ ▲
                                      └─┘  Run / Capture 循环
```

- 非法命令(非 Ready 态收到工作命令)= **结构化 LawViolation 回执**,不 panic;
- **捕获窗的 MemPhase 进出由 CaptureSession 内部自持**(owl-cuda 既有治理),
  actor 不重复管理相位——actor 只管"事务是否被允许开始"。

## 四、命令面与闭包契约(核心)

| 命令 | 闭包面 | ready 语义 |
|---|---|---|
| `run(job)` | `FnOnce(&CudaDevice) -> Result<R>`——覆盖池分配/memcpy/装载/查询 | **完成 ready**(阻塞消化在 actor) |
| `capture(job)` | `FnOnce(&mut CaptureSession) -> Result<DeviceGraph>`——原子捕获事务 | 完成 ready |
| `replay(graph)` | 图所有权回传 actor,租约校验 + launch | **提交 ready**(发射即回) |
| `sync()` / `close()` | 全设备排空 | 完成 ready |

### 契约 1:异步停在 actor 门口,窗内只有同步发射面

图捕获窗(begin→end)是**单线程原子段**:闭包必须一口气跑完,
**窗内禁止任何 `.await`**。理由(结构性,非纪律性):

- Rust async = 控制流挂起,恢复线程不保证是原线程 → 跨线程捕获窗 = 驱动级失效;
- 窗内挂起期间窗口仍开着,其他任务的发射会被录进图(RELAXED 模式静默污染);
- 窗内 await 的完成回执恰恰需要 actor 空出来处理 → 自锁。

注意区分两种"异步":**GPU 异步**(发射后不等执行,窗内合法且正是捕获的
工作方式)与 **Rust 异步**(控制流挂起,窗内禁止)。捕获是启动期一次性事务,
同步原子执行零性能损失;async 的价值在回放热路径,不在窗内。

### 契约 2:capture 闭包面收窄,禁止旁路发射

窗内发射**只允许走 KernelCtx/frame 发射面**(owl 治理的登记通道)。
raw cudarc 直发 = 契约外对象进审计 → 池块生命周期与租约登记全部失配
(2026-09-23 实锤:绕路直发 → end_capture/libcuda 路径 SIGSEGV)。

推论:capture 闭包签名不再暴露 `&mut CaptureSession`(太宽),改为暴露
frame 发射面 + 预绑定缓冲;"窗内至少一次发射"校验前移到命令边界。

### 契约 3:host 侧输入仓一律 pinned

pageable 指针的拷贝必经 legacy NULL 流中转(坑 B,§七)。actor 域内一切
host 侧 DMA 源/宿使用 `alloc_pinned`;数据生成**直接写 pinned 仓**,禁止
"Vec 中转再 copy"(双重拷贝;2026-09-23 社区调研确认:官方 best practice
即 pinned 直填/直接读文件,mapped zero-copy 仅限小数据一次性流式)。

### 契约 4:回执用 oneshot,不用闭包 future

oneshot 回执不依赖 future 被 poll 到底——future 遗漏 = UB 的雷
(async-cuda 的教训)在我们的形态下结构性不存在。

### 契约 5:描述层纯函数化(2026-09-23 定稿,详见 declarative-tensor.md)

模型层(layer/model/算子/采样)是**纯函数树**:无 Result、无 async、无相位
分支。forward 签名定稿 = `forward(&self, tx: &mut Tx, xs: &Tensor, 动态依赖) ->
Tensor`(惰性声明);Result 只存于 `new`(P 阶段装载)与执行边界(`interpret`/
`to_host`)。外部依赖二分:静态(权重/Comm/KvManager 本体)构造捕获,动态
(KvCtx)forward 参数;KV 副作用经 `tx.slot_write` 显式声明为节点。

## 五、完成语义三档

| 档 | 机制 | 适用 |
|---|---|---|
| 档一:提交即回 | launch/graph.launch 异步于 GPU,提交成功即回执 | decode 回放热路径;fire-and-forget |
| 档二:事件完成 | 池流 record event;专用 waiter 线程 `query()` 轮询或 blocking wait → oneshot | 单次拷贝/单图完成通知;runner 接入 |
| 档三:全量同步 | `ctx.synchronize`(actor 线程) | 排空/关机/测试 |

一期实现档一 + 档三(简单正确);档二是 runner decode 循环接入时的增量
(单 waiter 线程服务全设备事件)。

## 六、参照物与不采纳理由(2026-09-23 调研)

| 参照 | 机制 | 判决 |
|---|---|---|
| `async-cuda`(oddity-ai v0.6) | 单 runtime 线程 + mpsc 闭包队列 + 两档 ready 语义——**与我们设计同构**,社区验证 | ❌ 不采纳:独占设备所有权,与治理互斥;intentionally unsafe(future 遗漏 = UB);自持 FFI 非 cudarc |
| `cuTile`/`cuda_async`(NVIDIA) | DeviceOp 描述/执行分离;sync/async 双 API 决策表;spawn 强制 Arc | ❌ 不采纳:完整张量框架,抽象层级与 S1-S6 语义表冲突。**偷思想**:描述/执行分离列为二期批处理调度候选;双 API 决策表收进文档 |
| `cuda-oxide` | wrapper + async 章节 | 仅概念参考 |

共识(三方收敛,即行业标准形态):专用 GPU 线程 + channel 命令 + 两档 ready
+ spawn 时借用必须 Arc('static)。我们自研的理由:actor 必须长在治理层上
(池/相位/租约是 owl 独有维度),这部分没有任何现成库能替。

## 七、坑目录(全部实测实锤,设计逐一封死)

| 坑 | 症状 | 封法 |
|---|---|---|
| A. cudarc event-tracking 默认开 | 捕获窗内发射 = CAPTURE_ISOLATION | CudaDevice::new 显式 `disable_event_tracking()`(unsafe)+ 钉测。**A0 补丁,先行独立落地,与 async 无关也该落** |
| B. pageable memcpy 经 legacy NULL 流中转 | begin_capture 被拒 | 契约 3:pinned 一律 |
| C. 空捕获窗 | 哨兵③ 取节点 INVALID_VALUE | capture 命令边界校验窗内发射数 |
| D. 窗内旁路发射 | 治理契约外对象进审计 → libcuda SIGSEGV | 契约 2:闭包面收窄 |
| E. nvrtc arch 未钉 | INVALID_IMAGE | 显式 arch(既有) |
| F. launch 参数与 kernel 签名不严格一致 | INVALID_VALUE | 裁决 3①(既有) |

## 八、性能预期(诚实条款)

actor + async **不创造 GPU 并行度**;GPU 并行度 = 流拓扑 × 数据依赖图。
收益来源逐一列明,防止将来误判:

| 收益 | 机制 | decode 适用性 |
|---|---|---|
| CPU/GPU 流水线 | host 准备与 GPU 执行重叠(队列解耦) | ✅ 主要收益 |
| 提交时机精确 | 阻塞调度噪声归零 | ✅(20µs/步级场景,host 延迟即吞吐) |
| 图回放省发射开销 | 与 actor 正交,叠加收益 | ✅ |
| 多流真并行 | **需要流拓扑立项**(copy 流重叠/多请求多流),actor 不白送 | 二期:copy 流重叠是唯一"白送"型(拷贝引擎独立于 SM);多请求多流 = KV 分帐后立项 |
| 双调度线程并行 | 仅当依赖图宽(多请求独立链) | 一期伪需求:decode 链窄,单调度面即满配 |

## 九、里程碑

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **A0**(先行) | event-tracking 显式关闭 + infra_fixes 钉测 | 捕获窗内带事件等待的发射 = 结构化报错 |
| **A1** | actor 骨架:单线程泵 + Run/Sync/Close + 三态机;**Arc 不外泄**;按契约 2 收窄命令面;现有测试撤下重写 | Tokio 多线程并发 run 全绿;回执无错乱 |
| **A2** | Capture/Replay 命令;测试走 dry_run 同款 KernelCtx 闭包(契约 2 验收) | 捕获事务原子;回放判别三连过 |
| **A3** | 事件完成语义(档二) | decode 循环经 async 面跑通 |
| **A4** | engine server(Tokio)接入 | 多请求并发提交,回执各归各 |

## 十、开放问题

1. capture 闭包签名收窄后,dflash2 verify 第二图族是否需要独立命令变体?
   (倾向:否,同一事务面,族参数进闭包)
2. 档二 waiter 线程归 actor 管辖还是独立?"单 waiter 服务全设备"与 A3 的
   线程纪律兼容性待论证。
3. `run` 闭包内禁止 `.await` 是类型系统不可表达的(闭包非 async),仅能靠
   review + 惯例。是否需要 debug 断言(如闭包执行时长上限告警)?待议。
