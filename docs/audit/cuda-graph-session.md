# 审查报告:CUDA Graph / Session 机制

> 审查域:crates/backends/cuda/src/graph.rs(285 行)+ crates/engine/src/session.rs(493)
> + graphplan.rs(499)+ runner/mod.rs(674)+ backends/iface/src/signal.rs(158)
> 审查人:主会话(2026-09-23)。只读审查,未改动任何源码。

## 【危险清单】

### P0-1 捕获闭包 panic = 捕获流永久悬死

- **位置**:graph.rs `CaptureSession::capture`(`f(&frame)` 调用点,~行 145)
- **违反契约**:捕获窗口必须被 `stream_end_capture` 闭合(配对纪律)
- **现状**:`f` 返回 `Err` 有兜底(Err 分支 end_capture + destroy);但 `f` **panic**(闭包内
  多处 `expect`/索引越界)会栈展开跳过 end_capture——**stream 永久处于 capture 态**,
  该设备后续所有发射失效,且无诊断。
- **触发条件**:捕获闭包内任何 panic。forward 闭包由用户编排代码构成,panic 面真实存在
  (dry_run 时代就发生过 unwrap 崩)。
- **修法**:`std::panic::catch_unwind(AssertUnwindSafe(|| f(&frame)))` 包裹;或写一个
  `CaptureGuard`(Drop 里检查"未 end_capture 则补一次 end+destroy")。

### P0-2 哨兵③ 只告警不拦截:嫌疑指针照常 instantiate

- **位置**:graph.rs(`if !audit.suspicious.is_empty()` 分支,~行 183)
- **违反契约**:哨兵③"违约即毁图报错"的口径(模块头自述)
- **现状**:`audit.suspicious` 非空只 `eprintln! WARN`,然后**继续 instantiate + 返回图**。
  哨兵的全部价值(捕获未知指针入图 = 图回放读写悬空/未知地址)被降级成日志。
- **触发条件**:任何未走 ops(无 emit)、未手工 lease 的缓冲被捕获进图——恰恰是最危险的
  那类(与 slab 悬空案同族)。
- **修法**:suspicious 非空 = 拒绝 instantiate(`graph_destroy` + LawViolation),需要豁免
  时走显式 allowlist(带注释)。

### P1-1 捕获失败路径不清空租约:CaptureSession 复用时租约污染

- **位置**:graph.rs `capture()` 的 Err 分支与 audit 失败分支
- **现状**:成功路径 `mem::take(&mut self.leases / self.keepalive)`;**失败路径两个 Vec 原样
  留在 self 里**。若复用同一 CaptureSession 再捕获(下一档位),上一轮(失败的)租约混入
  新图租约集——哨兵③对账基线被污染,DeviceGraph 持有无效租约。
- **触发条件**:CaptureSession 复用 + 首捕失败。当前 `Session::plan` 每档 `dev.capture_session()?`
  新建未触发,但类型契约已坏(公开 API,无"一次性"标注)。
- **修法**:失败分支 `self.leases.clear(); self.keepalive.clear();`,或类型改名
  `OneShotCaptureSession` 表达一次性。

### P1-2 捕获窗口结束 = scratch 悬空窗口(emit 覆盖面缺口)

- **位置**:graph.rs 租约升级逻辑 + session.rs 捕获闭包
- **违反契约**:图内引用的缓冲必须存活至 DeviceGraph drop
- **现状**:自动租约靠 **emit 时刻**升级 Weak→Arc。emit 只覆盖"被 ops 读过的"缓冲;
  **纯写入不被 ops 读的缓冲**(末端输出、raw FFI 写)在 Capturing 相结束后照常回收,
  图内地址悬空,回放写他人内存 = 静默数据腐蚀。这缺口就是 varlen slab 悬空案的同族
  (slab 案发生在 eager 侧,若在捕获侧发生则是回放腐蚀,更隐蔽)。
- **缓解现状**:session bank 有手工 `lease_all`(注释自认"ops emit 盖不到");scrub 池
  中间量多数会被后续 kernel 读(emit)而幸存——**但"多数"不是契约**。
- **修法**:①哨兵③收紧(见 P0-2)兜底;②长期:scratch 池在 Capturing 相的分配默认
  租约化(分配即登记),归还与租约联动。

### P1-3 租约校验仅 debug 构建

- **位置**:graph.rs `DeviceGraph::launch`(`#[cfg(debug_assertions)]` 包住校验循环)
- **违反契约**:A1.2"replay 前逐租约世代校验,失效 = 结构化报错"
- **现状**:release 构建下 A1.2 校验完全消失——租约失效从"结构化报错"退化为静默数据错误。
- **触发条件**:任何 release 部署 + 缓冲生命周期违约。
- **修法**:校验循环 N≤几百个 token、纯内存比较,成本可忽略——直接去掉 cfg,release 也查。

### P1-4 Session 槽位的隐式持久语义:陈旧数据静默参与计算

- **位置**:session.rs `step()`(`bs = max(inputs.len())`;未提供的槽保留上一步数据)
- **现状**:输入槽是持久的;step 只 H2D 提供的槽。若编排闭包对"本步未写的槽"敏感
  (如 positions 槽忘写),replay 静默用陈旧值。`bs` 又只按**提供的数据长度**选档,
  槽位实际有效长度与档位不一致时(槽 len 8、档位 2、数据 1),narrow 视图 [bs.min(len)]
  的边界假设全靠调用方自律。
- **触发条件**:调用方漏写一个槽 / 长度语义理解偏差。测试代码里已出现两种写法
  (write+step 空参 vs step 全参),说明契约模糊。
- **修法**:Session 记录"已初始化槽集合",首步强制全量;或 step 后记录槽状态并
  在 debug 下对未写槽告警。

### P1-5 双图体系并存:graphplan deprecated 但 runner 仍全量依赖

- **位置**:graphplan.rs(deprecated 头注释)+ runner/mod.rs 引用点清单:
  - `AdapterFwd`(runner:576)impl `graphplan::GraphForward`
  - `GraphDecodeRunner.plan: Option<graphplan::GraphPlan>`(runner:594)
  - `GraphPlan::planned_batches / capture / CaptureOutcome`(runner:617-619)
  - qwen3_5.rs `DecodeGraphAdapter`(为 graphplan 而生)
- **风险**:两套 capture/租约/审计/回放逻辑并存,bug 修复(Session 侧已修的 panic 豁口、
  租约污染、哨兵收紧)不会自动传导到 graphplan 侧——runner 走的是旧路。
- **修法**:runner 迁 Session 适配(编排闭包复用,GraphBindings → InputSlot 映射机械),
  然后 graphplan.rs + DecodeGraphAdapter 整体删除。估计一个 worker 日。

### P2-1 TrackingHandle 可跨线程 drop:TLS 栈错线程弹出静默失败

- **位置**:iface/src/signal.rs `TrackingHandle::drop`(try_with + 栈顶 id 匹配)
- **现状**:handle 是 `Clone` 且隐式 `Send`(Sink 是 Send+Sync)。跨线程 drop 时
  `stack.last().id` 不匹配 → 静默不弹 → **原线程的栈永久残留一层**(后续 emit 全部路由
  到死 sink)。A3 单线程契约下不会发生;多线程化(A2.6)即是雷。
- **修法**:`impl !Send for TrackingHandle`(PhantomData<*const ()>),把违约变编译错。

### P2-2 signal TLS 线程退出期静默放弃(已知过渡态)

- **位置**:signal.rs `emit`/`TrackingHandle::drop` 的 `try_with` 容忍
- **现状**:dry-run 线程退出 abort 事故的止血补丁。根治 = signal 局部 scope 重构
  (用户已拍板,挂账 §五-1)。无新增风险,但 **tracking 语义在线程析构窗口内不可依赖**,
  若未来有 atexit 式的落盘/flush 依赖 emit,会静默丢依赖。

### P2-3 AUTO_FREE_ON_LAUNCH 与 keepalive 的交互未审计

- **位置**:session.rs 捕获旗标 `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH`
- **现状**:该旗标语义 = "launch 时可释放图不再引用的分配"。依赖"keepalive 全量持有"
  这一前提;而 keepalive 覆盖面有缺口(P1-2)。旗标会放大任何租约缺口的后果。
- **修法**:P1-2 收紧后保留旗标;或短期换 `DEVICE_SYNCHRONIZE` 旗标规避。

### P2-4 H2D 同步 memcpy 与 capture 流的序依赖(当前安全,记录在案)

- **位置**:session.rs `write_u32`(`cuMemcpyHtoD_v2` 同步)→ `graph_launch`(capture 流)
- **现状**:同步 memcpy 保证 launch 前数据可见——正确。但正确性建立在"同步 memcpy"
  这一隐式点上;若未来换 `cuMemcpyHtoDAsync` 提性能,流序断裂,图读到旧数据。
- **修法**:加注释钉死依赖;或直接改 async + 事件序,把正确性写进代码而非文档。

## 【设计问题】

1. **捕获失败不降级 eager**:A1.4 预检只对显存不足降级;捕获中途失败(某档位 instantiate
   失败)整体 Err——已捕获的前几档图被静默丢弃,无"EagerFallback 带部分图"的路径。
   语义分叉:调用方无法区分"完全没图"和"图建了一半"。
2. **sealed_bytes 恒 0**:`PlanOutcome::Captured { sealed_bytes: 0 }`——A1.4 的
   "seal 实测超支弃档"(注释自述)未实现,档位内存核算目前是空头支票。
3. **暖场 replay 在 capture() 内部不做、由 plan 做**(session.rs `graph.launch()` 暖场):
   CaptureSession API 的调用方若忘暖场,首次正式 replay 承担 JIT/懒初始化抖动。
   建议暖场下沉进 capture。
4. **output_ptr 逃生口无防护**:`output_ptr() -> *const f32` 注释"调用方保证生命周期"——
   裸指针逃逸面。短期可接受(采样同址读),长期应提供带租约的读取句柄。
5. **`step` 的 `&self` + `forward: Box<dyn Fn>`**:eager 态下 step 通过 &self 调用
   forward——若 forward 闭包捕获了可变状态(如 rc RefCell),并发 step 即数据竞争;
   A2.6 单线程契约下安全,多线程化时需要 &mut 或内部锁设计。

## 【安全项(无需改动,注明理由)】

1. **emit-时刻升级 keepalive 的时序**(graph.rs 注释"2026-09-22 关键时序修正"):在 emit
   瞬间缓冲必然存活,Weak→Arc 当场升级——设计正确,且注释把教训写清楚了。
2. **PhaseGuard RAII 相位恢复**:闭包 Err 时 Capturing 相正确弹出(panic 路径除外,
   见 P0-1);恢复原相而非压 Idle——尊重调用方嵌套。
3. **DeviceGraph Drop 顺序**:先 exec 后 cu_graph(exec 引用 graph,反序会悬空)——正确。
4. **捕获流 non-blocking + RELAXED 捕获模式**:A3 单线程契约下,RELAXED 的并发捕获
   风险不适用;若多卡多流并行捕获需复核。
5. **session bank 独立 Weights 池**:槽地址稳定,回放写同址的前提成立;槽池与 scratch
   池分离避免地址复用干扰。
6. **暖场 replay + sync**:seal 后立即 launch + synchronize 一次,实测图完整性——好习惯。
7. **signal 单测覆盖**(嵌套/不穿透/id 单调/线程退出容忍):语义钉测齐全。
8. **graph_launch/graph_instantiate 的 FFI 封装**(lib.rs):错误码转结构化报错,无裸
   CUDA_SUCCESS 比较散落——封装面干净。

## 统计

- 危险清单:P0 × 2,P1 × 5,P2 × 4(共 11 条)
- 设计问题 × 5,安全项 × 8
- graph.rs 9 处 unsafe 逐一核对:2 处错误路径 destroy、1 处 instantiate、1 处 upload、
  1 处 launch、1 处 end_capture、2 处 Drop、1 处闭包失败 destroy——**每处都有正确性
  论证,无裸指针算术**;真正的高危不在 unsafe 本身,而在 unsafe 的**调用时序**
  (P0-1/P0-2)。
