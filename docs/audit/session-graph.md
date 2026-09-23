# Session/Graph 机制审查清单(CUDA Graph 捕获会话 + 引擎 Session 编排)

> 审查域:backends/cuda/graph.rs(CaptureSession/DeviceGraph/Governor 相位)+
> engine session.rs + graphplan.rs(deprecated)+ shared/signal(依赖追踪)。
> 风险级:P0 = 错误数值/UB;P1 = 特定条件;P2 = 债。

## 危险清单

- **S1** `graph.rs:259 launch()` — replay 租约校验仅在 `debug_assertions` 下执行;
  release 构建中,A1.2(图依赖令牌失效,如权重池页被复用)静默执行错图。
  风险 **P0(release 路径)**。修法:校验移出 cfg(release 保留廉价表查询,
  debug 才做全量 audit)。
- **S2** `graph.rs:62 capture()` — 捕获期 `PhaseGuard` Drop 恢复相位,
  但捕获中 panic(unwind)时 stream_end_capture 未调用 → stream 停留在
  capture 状态,后续发射全部丢失。风险 **P1**。修法:guard Drop 内补
  `stream_end_capture` 兜底(吞错)或捕获改 no_unwind 闭包边界。
- **S3** `graph.rs:274 Drop for DeviceGraph` — exec/graph 销毁在 Drop,
  未与 stream 同步(replay 尚在飞行时 drop → use-after-free of exec)。
  当前单线程 E 阶段顺序安全;多 rank/多流后为 **P1**。修法:Drop 前
  stream synchronize(吞错)。
- **S4** `graphplan.rs`(deprecated)仍被 runner/mod.rs 引用(allowance 类型),
  其 Raw 视图租约语义与 Session 不一致 —— 迁移未完成即有双轨。
  风险 **P1(迁移期)**。修法:按挂账 §五·5 择一收敛。
- **S5** `session.rs` 输入槽 U32 定死(`InputSlot::u32`)——M-Ⅱ 起 KV lens
  需要 i64 边界时为 **P2**;输出槽 f32 定死,量化 logits 需扩。
- **S6** signal 依赖追踪:emit 全覆盖借/写/出(审查通过);但 `TrackingHandle`
  TLS 栈在线程退出期的 abort 已由 try_with 容忍(过渡态),
  signal-scope 重构(挂账 §五·1)后应删除该容忍路径 —— **P2(挂账已有)**。

## 设计问题

1. warmup 6 门禁 / A1.4 预检 / 哨兵③ audit / seal 的流程全部封在
   `Session::plan` 内(正确);但**门禁判定所需的 `eager_launches` 计数**
   依赖 dry-kernels trace 的计数器,而 erased 算子直接发射(cublas 面)
   不经 trace —— 预检的覆盖面 cublas 段为盲区,依赖 per_profile_bytes
   人工预算。建议:OpsCtx::note_launch 也计入(cublas.rs 已调用,复核覆盖)。
2. EagerFallback 同闭包重放设计正确;但 fallback 后无"这次没走图"的
   持久标记,上层 runner 无法区分 —— 观测面缺口(P2)。

## 安全项

- capture 期间禁止 Persistent 分配(E 阶段纪律)+ scratch 池独立:
  池内分配不动 VMM 映射,图地址稳定(审查通过);
- 哨兵③ `enumerate_nodes/audit_kernel_node`(audit.rs,unsafe fn ×3):
  只读枚举 + 结构化报告,FFI 参数构造正确(审查通过);
- A1.2 租约表 debug 校验 + 哨兵③世代校验的双层结构合理(S1 修 release
  覆盖后即闭环);
- phase 栈单写者(Governor),相位回退路径经 PhaseGuard RAII(S2 兜底后闭环)。
