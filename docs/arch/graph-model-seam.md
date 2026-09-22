# Signal-Graph × 模型层对接设计(graph-model-seam)

> 版本:v0.1(2026-09-22)。上游:xinfer-transplant.md;执行面:owl-nn OpsCtx /
> owl-cuda CaptureSession;被对接层:owl-engine runner/models。

---

## 一、xinfer 现状(被替换件解剖)

| xinfer 机制 | 实现 | 问题(owl 视角) |
|---|---|---|
| `CudaGraph` 裸 FFI 包裹(utils/graph.rs) | begin/end_capture + instantiate + replay | 无依赖登记、无相位、无审计——死亡姿势图鉴 10 条全开放 |
| `GraphCaptureVars` | positions/slot_mapping/kv_len/indptr/last_len 等**静态钉住张量**,每步 host 填充 | 语义正确(A1.5 雏形)但无哨兵①:这些缓冲若被释放,replay 盲飞 |
| 每图私有池 + AUTO_FREE_ON_LAUNCH | 捕获池私有化 + 每次 launch 后自动释放图池分配 | 与单共享池(A1.3)冲突;AUTO_FREE 是地址不稳定症状的补丁 |
| logits D2D 到非池缓冲 | capture 期把 logits 拷到"池外"稳定地址 | = 租约系统缺失的手工替代 |
| decode_capturer + mtp_capturer | 双图族,1..=32 每 bs 一图 | 档位规划可继承;捕获事务面需重造 |

## 二、对接面:三层各管一段

```text
runner(T3 搬运)          models(qwen3_5 等)           owl-nn/owl-cuda
────────────────         ───────────────────           ────────────────
GraphPlan                 forward(&KernelCtx, &Bindings) OpsCtx(eager/capturing)
 档位表 + 图实例表 + 绑定   单一代码路径,相位无感          CaptureSession(自动租约
 bindings 填充(EagerOnly)  全部动态量走 device 张量         + 相位 + 哨兵③ + 定影)
```

**核心原则:模型 forward 只有一份代码,相位无感。** 同一个
`forward(ctx, bindings)` 在 eager 下直发、在 Capturing 相下被烙进图;
ctx(owl-nn `KernelCtx`)决定 emit/record,模型层不出现任何 `if capturing`。

## 三、GraphPlan(engine 侧新模块,替换 GraphCapturer)

```rust
pub struct GraphPlan {
    /// bs → 定影图(单共享捕获池,A1.3:峰值 ≈ 最大一张图)
    graphs: BTreeMap<usize, DeviceGraph>,
    /// 档位表(继承 xinfer planned_graph_capture_batches:1..=32 精确档;
    /// GDN/mamba 的 slot 映射不可 pad,精确档是语义要求不是优化)
    profiles: Vec<usize>,
    /// 全档共享的钉住缓冲(租约常驻;host 每步填充,图内读取)
    bindings: GraphBindings,
}

pub struct GraphBindings {
    /// A1.5:全部动态量 = device 张量(U32;S4 索引律)
    frontier: DynTensor<CudaDevice>,   // 每步 token ids [max_bs]
    positions: DynTensor<CudaDevice>,  // [max_bs]
    slot_mapping: DynTensor<CudaDevice>,
    kv_lens: DynTensor<CudaDevice>,
    /// 输出绑定:logits 常驻租约缓冲(替代 xinfer 的 D2D 手工拷贝——
    /// 图直接写这里,采样 radix kernel(P4)同地址读,全程零 D2H)
    logits_out: DynTensor<CudaDevice>,
}
```

### 捕获(每档位一次,启动期净空窗口)

```rust
let bs = *profile;
// 1. warmup(姿势 6 门禁:CaptureSession 会拒未发射设备)
model.forward(&eager_ctx, &bindings.view(bs))?;      // eager 真发一次
// 2. 捕获(相位自动入 Capturing;信号 sink 自动租约)
let mut session = dev.capture_session()?;
let (_, graph) = session.capture(inst_flags, |frame| {
    // bindings 各缓冲先整档租入(frame 自动记 touched);
    // narrow 到 bs 行是元数据视图,不出捕获段
    model.forward(&frame.ctx(), &bindings.view(bs))
})?;
// 3. 定影(A1.2:measured allowance;超收即销图报错)
graph.seal(measured)?;
plan.insert(bs, graph);
```

捕获期激活:scratch 池流有序分配(**捕获安全**),物理页由图租约
keepalive 钉住——**不需要 AUTO_FREE_ON_LAUNCH**,也不需要 logits 池外拷贝:
稳定性由租约给,不给池属性。跨图共享池(裁决 A1.3)天然挤平峰值。

### 回放(每 decode step)

```rust
// 1. 调度器产出 → H2D 写 bindings(EagerOnly;必须图同流发射,
//    流序天然先于 graph.launch;禁止 capture 相)
bindings.write_frontier(&tokens)?;   // cuMemcpyHtoD on graph.stream
// 2. replay(debug 构建逐租约世代校验,失效 = 结构化报错非 Xid)
plan.replay(bs)?;                     // 找 ≥bs 的最小档 + graph.launch()
// 3. logits 就在 logits_out(租约钉住),radix 采样直读
```

## 四、运行里程碑回填清单(OwlTensor 层)

GraphPlan 依赖的最小真实 kernel 面(其余 unimplemented 不阻塞):

1. `DynTensor::write_from_host`(H2D,图主流,EagerOnly);
2. `copy_d2d`(bindings → narrow 视图同址;推理激活前的 slot gather);
3. `matmul`(cublas,已通)+ add/silu/rmsnorm(已通)——decode FULL 图
   的链路只缺 `rope`/`paged attention`/`embedding lookup` 三个 kernel
   (rope/embedding nn 已有,剩 paged attention 从 attention-rs port,
   与 vendor 归宿裁决联动);
4. `argmax/sort`(采样链;radix 已定谳直吃 logits)。

## 五、裁决请求(用户拍板)

1. **档位表**:1..=32 精确档(xinfer 语义,GDN 硬要求)——建议原样继承,
   图预算 A1.1 = Σ 各档 bindings + 最大档激活 × 共享池系数;
2. **dflash2 verify 图**:第二图族,同池共预算(A1.3);锚复用 bonus
   逻辑在 engine 采样层,图面只是第二个 forward 路径;
3. **vendor 归宿联动**:paged attention 是 decode FULL 图的最大缺口,
   三选一(自研/保留 attention-rs/换 flashinfer)直接决定 A1.5 的
   可捕获性审计工作量——attention-rs 的 kernel 是裸 CUDA 天生可捕获,
   保留依赖的最小改动路径 = 只 port kernel 不 port 其运行时;
4. **mamba/GDN 前缀回滚**(xinfer capture_mamba_prefix_state):状态
   快照属于 slot 语义,建议随 qwen3_5 搬运时以 owl 池缓冲重表达,
   快照点 = 捕获前钉住的 slot 池整段 memcpy。
