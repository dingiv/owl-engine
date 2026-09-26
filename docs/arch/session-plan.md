# Session 编排面设计 —— 消灭上层手工流程纪律(P0)

> 2026-09-22。动机(用户):dry_run 测试暴露上层被迫手工执行 warmup/装填/
> 捕获/回放全流程;eager 与 graph 两种模式要写两遍。目标:**算子编排闭包
> 只写一份,三态(eager/捕获/回放)由 Session 承载**。

## 一、API

```rust
// ── 声明(全项目唯一一份编排逻辑)──
let plan = Session::plan(&dev, SessionDesc {
    profiles: vec![1, 2, 4],
    inputs: vec![
        InputSlot::u32("frontier", 4).init(vec![0, 0]),
        InputSlot::u32("positions", 4).init(vec![0, 1]),
        ...
    ],
    outputs: vec![OutSlot::f32("logits", 4, 96)],
    scratch_bytes: 64 << 20,
}, move |sc: &StepCtx| -> Result<()> {
    let ids   = sc.input("frontier")?;      // 本档 narrow 视图
    let pos   = sc.input("positions")?;
    let kv    = &kv_caches;                 // 外部 state(闭包捕获)
    let h     = model.forward(ids, pos, ...)?;
    sc.output("logits", &h)?;               // 写入输出槽(D2D,容量守卫)
    Ok(())
})?;

// ── 每步执行(eager/replay 由 plan 内部决定,上层零分支)──
plan.write("frontier", &[11, 22])?;
plan.step()?;                                // H2D 装填 → 门禁 → replay/直发
dev.ctx().synchronize()?;
let logits = plan.read_output_f32("logits")?; // D2H 读槽
```

## 二、语义

1. **槽(slot)统一词汇**:
   - 输入槽 = u32 `[len]`(frontier/positions/slots…);P 阶段池分配,
     每步 H2D 装填;捕获期按档 narrow 视图(narrow_dim0 元数据偏移);
   - 输出槽 = f32(形状任意);闭包产出经 D2D 写入槽(容量守卫);
     采样/回读同址;
   - state 槽(KV/GDN)暂由闭包外部持有(P0 不入 session;GDN state
     纳管列为阶段三)。
2. **warmup = 首次 dry 执行**:plan() 内部以 eager ctx 跑一遍闭包——
   同时完成姿势 6 门禁、shape 学习、EagerOnly 审计。上层无感。
3. **三态统一**:step() 按档位 replay;A1.4 预检失败 → EagerFallback
   (同闭包直发)。上层零分支。
4. **作用域生命周期**:闭包通过 `ctx_scope::push(sc.ctx())` 把执行凭证
   交给层代码(candle 形态签名桥,现状保留);捕获态 ctx = frame ctx
   (自动租约),eager 态 = Live ctx + scratch。
5. **signal 全局栈的死刑缓期**:本次不删 TLS(空跑兵 ctx_scope 依赖),
   但 Session 化之后追踪作用域生命周期 = IoCtx 生命周期,全局栈的删除
   是纯机械收尾(单独一刀)。

## 三、实现落点

- `crates/engine/src/session.rs`(新):SlotSpec/StepCtx/SessionPlan;
  捕获循环逻辑自 graphplan.rs 泛化(姿势 6/A1.4 预检/Governor 单档周期/
  哨兵③/seal 全保留);
- graphplan.rs 保留(decode 专用,标 deprecated 语义;runner 真接线时
  迁移到 Session 后删除);
- DecodeGraphAdapter:被 Session 闭包替代,测试迁移后删除。

---

## 实施状态(2026-09-26)

- **新 engine crate 立项落地**(旧 engine 25k 行移 engine-bak 存档,复用审计见
  `crates/engine/src/lib.rs`):M0 骨架 + P0 eager 闭环。
- **生命周期 facade**(用户裁决):`Engine::new/on`(构造,零执行)→
  `ModelLoader`(spec 声明驱动装载)→ `Engine::run(model)`(装配 Session +
  warmup,进入执行态)→ `RunningEngine::submit/pump`(turn 流)。
- **turn/step 词汇**(用户裁决):前端一个 turn ⇋ 引擎一个调度单元,内含
  多步;step 治理(槽装填/状态推进/停机)归引擎;事件流
  `Idle/Prefill/Token/Completed/Failed` 回吐。并发:M0.5 单槽串行
  (GDN per-turn 零化;KV 由 kv_len 窗口 + 先写后打分天然隔离),
  M2 continuous batching 换真并发,submit/pump 形状不变。
- **附带战果**:session 测试逮到 owl-cpu `matmul` k 自推缺陷(alloc 块无
  形状 → k=1;修复 = launch 标量权威)。
- 待办:M1 捕获三态接线(姿势 6 门禁/A1.4 预检/GraphLease)、设备采样、
  M2 batching。
