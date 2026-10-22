# owl 异步编程模型(四层架构 + 声明式 Tensor)

> 版本:v0.3(2026-09-24)。v0.2 合并 declarative-tensor.md;v0.3 同步
> 标量类型化/Kernel 节点落地与坑 I-L。上游:charter.md、graph-model-seam.md。
> 状态:**设计定稿,实施进行中**。

---

## 一、四层架构

```text
┌─────────────────────────────────────────────────────┐
│ 模型层 models                                        │
│  Module::forward → TensorOps(声明链,零执行)         │
│  .cu + 包装函数 = kernel 绑定本体                     │
│  禁止:执行、设备、错误面                              │
├─────────────────────────────────────────────────────┤
│ 调度层                                               │
│  连续推理;只碰解释层 API                              │
│  禁止:碰设备                                         │
├─────────────────────────────────────────────────────┤
│ 解释层 client                                        │
│  DAG 展开 · 预定义动作表 · KernelCtx/ForwardCtx 注入  │
│  LaunchMsg 打包 · 后端硬件屏蔽                        │
│  禁止:算法语义                                       │
├─────────────────────────────────────────────────────┤
│ 硬件层 server(owl-cuda gpu_server)                   │
│  发射(LaunchMsg)· 搬运 · 显存池 · Buf 编号           │
│  Capture session 收拢管理面                           │
│  禁止:算法语义(纯哑执行器)                           │
└─────────────────────────────────────────────────────┘
```

### 职责矩阵

| 层 | 持有 | 禁止 | 对应 crate |
|---|---|---|---|
| 模型层 | 算法结构声明 + 加载布局 | 执行、设备、错误面 | `owl-models` |
| 调度层 | 业务节奏(连续推理) | 碰设备 | `owl-engine` |
| 解释层 | DAG 展开 + 动作表 + ctx 注入 | 算法语义 | `owl-models::client` |
| 硬件层 | 账房 + actor 线程 + kernel 缓存 | 算法语义 | `owl-cuda` |

---

## 二、核心类型

### 2.1 TensorOps(声明链)

```rust
pub struct TensorOps {
    id: u64,                    // 全局唯一自增(跨线程)
    parents: Vec<TensorOps>,    // 反向边(值语义深拷贝;无 Arc)
    depth: u32,                 // 拓扑深度
    op: Op,                     // 语义运算
    dtype: Dtype,
    shape: Shape,
    args: Vec<KernelArg>,       // Kernel 节点的有序参数槽
    err: Option<LazyError>,     // 毒值
}
```

- **值语义**:clone = 深拷贝子树(配置面一次性成本,可接受)
- **无 Arc、无 Tx、无 Step/TensorMeta 中间类型**——一个 struct 就是全部
- 声明期 total:违约变毒值,永不 panic、永不提前 return Err

### 2.2 Tensor(运行时数据)

```rust
pub struct Tensor<D: Device> {
    dev: D,
    block: D::Bytes,     // 池块(数据所在)
    offset: usize,       // 块内偏移(视图)
    dtype: Dtype,
    shape: Shape,
    id: u64,
}
```

- 跨设备统一:`D: Device` 对 CPU(`Cpu`)与 GPU server(将来)同一形状
- clone = 共享底仓 + 同偏移(零拷贝视图)
- `as_declaration()` → TensorOps(桥:数据进声明图)

### 2.3 Kernel(自描述发射包)

```rust
pub struct Kernel {
    pub name: &'static str,       // kernel 入口名(编译缓存键成分)
    pub source: &'static str,     // .cu 源码(内嵌;后端 nvrtc 懒编译)
    pub launch: LaunchShape,      // grid/block/shared_mem;grid (0,0,0) 哨兵 = 自动 1D
}
```

- 参数槽 `KernelArg::{T, Bits, I32, F32}` **类型化**:与 kernel 形参宽度严格对位
  (CUDA 参数空间按形参自然对齐,宽槽顶窄形参会错位读参 —— 坑 I);

### 2.4 DeviceClient(异步能力契约)

```rust
pub trait DeviceClient: Send {
    async fn alloc(&mut self, n_bytes: usize) -> Result<Bytes, ModelError>;
    async fn htod(&mut self, dtype: Dtype, shape: &Shape, src: &[u8])
        -> Result<Bytes, ModelError>;
    async fn dtoh(&mut self, b: &Bytes, out: &mut [u8]) -> Result<(), ModelError>;
    async fn launch(&mut self, msg: LaunchMsg) -> Result<Bytes, ModelError>;
    async fn sync(&mut self) -> Result<(), ModelError>;
}
```

五原语。**不暴露具体算子接口**——算子语义在解释层动作表里 lower 为 LaunchMsg。

---

## 三、声明式编程模型

### 3.1 工厂(TensorOps 关联函数)

| 工厂 | 说明 |
|---|---|
| `TensorOps::zeros(dtype, shape)` | 清零分配声明 |
| `TensorOps::from_host(dtype, shape, data)` | host 数据声明(值语义:数据随树走) |
| `TensorOps::of_block(id, dtype, shape)` | 引用已物化数据块 |
| `TensorOps::of(kernel)` | Kernel 节点声明 |

### 3.2 链式运算(total,永不失败)

| 方法 | 语义 | 输出形状 |
|---|---|---|
| `.matmul(&b)` | [m,k]×[k,n]→[m,n] | self.shape 替换末维 |
| `.add(&b)` | 同形逐元素加 | self.shape |
| `.silu()` | 逐元素激活 | self.shape |
| `.rmsnorm(&alpha, eps, w_off)` | ×(1+w) 或 ×w | self.shape |
| `.arg(&t)` / `.arg_f32(v)` / `.arg_i32(v)` / `.arg_usize(v)` | Kernel 节点参数入槽(类型化) | — |
| `.with_shape(dtype, shape)` | Kernel 节点输出形状标注 | 标注后 shape |
| `.narrow(dim, start, len)` | 视图(零节点追加) | 收窄后 shape |

**毒值传播**:构造期违约(shape/dtype 不符)→ `Poisoned(LazyError)` 随链流动,
算子对毒恒等(不追加节点、不重复报),eval 边界收割(LazyError 携带 depth 归因)。

### 3.3 Module trait(layer 形态)

```rust
trait Module {
    /// 构造期(new):唯一保留 Result 的位置(装载可败)
    /// forward:同步 · total · 纯描述(零 ? 零 await 零 Tx)
    fn forward(&self, prev: &Tensor, ctx: &ForwardCtx) -> TensorOps;
}
```

- **静态依赖**(权重/Comm/KvManager 本体)→ 构造函数传引用,layer 持有
- **动态依赖**(KvCtx/positions)→ forward 参数
- **KV 副作用** → `tx.slot_write(kv, slots, &v)` 显式声明为节点

---

## 四、执行模型

### 4.1 解释器

同一棵 TensorOps 树,三种游走:

| 解释器 | Op 执行体 | 场景 |
|---|---|---|
| eager(档一) | 逐节点提交 server(提交即回) | decode 热路径 |
| capture(烘焙) | 逐节点"录"→ 整单 instantiate → 哨兵③对账 | 图捕获 |
| CPU(测试) | 纯 Rust 闭包归约 | nn 测试(保持同步写法) |

**warmup = 用 eager 解释器先跑一遍同一份描述**(姿势 6 制度化)。

### 4.2 KernelCtx(算子开口子 grab-bag)

Fn 节点的发射上下文。算子缺什么放什么:stream / kernel 表 / scratch / kv manager。
**由解释层构造注入**——Fn 只管发射,不碰设备管理。

### 4.3 ForwardCtx(每步动态依赖 grab-bag)

同性质:positions / kv 上下文 / 采样参数。每步由 runner 构造传入。

---

## 五、GPU server(gpu_server.rs)

### 5.1 actor 线程模型

```text
async 面(零阻塞 Tokio 任务)
  │ mpsc<Job> + oneshot 回执
  ▼
GPU actor 线程(唯一触碰 CudaContext/stream;bind_to_thread 一次到位)
  ▼
池块账房 + 懒编译缓存 + kernel 发射
```

### 5.2 懒编译

`ensure_kernel(name, source)`:按名查缓存 → 未装载 → nvrtc 编译 CUDA C →
PTX → load_module → load_function。arch 显式钉(不钉 = INVALID_IMAGE)。

### 5.3 LaunchMsg 发射

```text
1. 解析参数槽:Block(id) → 账房取设备指针;标量按类型化宽度入槽
2. ensure_kernel(name, source) 懒编译(nvrtc → PTX → module)
3. builder.arg(逐槽) + launch(cfg)
```

**输出块由 eval 预先 alloc**(描述层职责;非 server 代分配),
槽序契约:**输出块固定是最后一个 Block 参数**,server 按“最后一个 Block”
回传句柄。`Bytes.len` 语义 = **元素数**(非字节;alloc/htod 均按元素登记)。

---

## 六、坑目录(全部实测,设计逐一封死)

| 坑 | 症状 | 封法 |
|---|---|---|
| A. cudarc event-tracking 默认开 | CAPTURE_ISOLATION | `disable_event_tracking()`(A0 补丁) |
| B. pageable memcpy 经 legacy 流 | begin_capture 被拒 | pinned 仓(alloc_pinned) |
| C. 空捕获窗 | 哨兵③ INVALID_VALUE | 发射数 > 0 校验 |
| D. 窗内旁路发射 | 审计 SIGSEGV | 契约 2:只走 frame 发射面 |
| E. nvrtc arch 未钉 | INVALID_IMAGE | build.rs 显式 arch |
| F. launch 参数不严格一致 | INVALID_VALUE | 裁决 3①:标量+裸指针 |
| G. 同语句双锁(parking_lot) | 死锁 | 显式 drop 后再锁 |
| H. htod 丢声明 shape | 后续 matmul 维度 OOB | htod 契约带 dtype+shape |
| I. 标量参数全按 u64 推槽 | CUDA 参数空间错位读参(rmsnorm 出 garbage;Kernel 节点非法访问) | Arg/KernelArg 类型化(U64/I32/F32),与形参宽度严格对位 |
| J. Bytes.len 当字节数再 ÷4 | n 缩水 4 倍(add/rmsnorm 只算首元素) | 定谐:len = 元素数,lower 层禁二次换算 |
| K. Kernel 节点槽序与签名不一致 | n=4 被当指针解 → ILLEGAL_ADDRESS | 槽序契约:输出块固定最后;示例/文档双重标注 |
| L. launch 结果写进新块(CPU 面) | eval 回传句柄读到预 alloc 零块 | CpuFace 写回 out 块原 id(与 GPU 语义一致) |

---

## 七、实施状态(2026-09-23)

| 里程碑 | 状态 | 说明 |
|---|---|---|
| A0:加固补丁 | ✅ | event-tracking 关闭 + 双锁死锁修复 |
| A1:server 骨架 | ✅ | actor + 五原语 + 状态机(_gpu_server.rs 全绿) |
| A2:声明式 TensorOps | ✅ | 值语义 + 毒值 + 链式 API(playground 四 demo 全绿) |
| A3:GPU 端到端 | ✅ | mlp_server 经 GPU server 真发射,对拍一致(2026-09-24 复验) |
| A3.5:Op::Kernel(动作表二期) | ✅ | lower_kernel + eval Kernel 分支 + kernel_node 示例双路径全绿 |
| A4:nn/engine 迁移 | ⏳ | owl-shared/nn/engine 旧消费面待迁(重写期已知破损) |
| A5:memory-planner | 未立项 | liveness 着色 + 状态区(周级) |

---

## 八、关键裁决记录

1. **不用 Arc**(值语义):配置面一次性成本可接受,零共享别名;
2. **不用 Tx**:深度构造时算、副作用即树节点、相位由解释器承担;
3. **不用宏**:TensorOps 层定义空间足够,包装函数即绑定本体;
4. **Server 哑执行器**:五原语,零算子知识,新算子 = load_kernel 注册一次;
5. **KernelFn 自描述**:name + ptx + slots + grid + block 全随节点走;
6. **Bytes = server 块身份证**:id 跨进程唯一,账房按它反查池块。
