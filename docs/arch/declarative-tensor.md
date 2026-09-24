# 声明式 Tensor 编程模型(纯函数 forward + 惰性执行)

> 版本:v0.1(2026-09-23 定稿,用户拍板)。上游:async-runtime.md(契约五)、
> graph-model-seam.md(相位无感原则)、funio Pack 代数(用户自有先验,范式来源);
> 消费方:owl-nn(Tensor/ops)、engine(models/runner)。
> 配套可执行规格:`crates/backends/cuda/examples/playground.rs`(CPU 主干四 demo
> + GPU 执行器对拍,全绿)、`tensor_chain_api.rs` / `tensor_decl_api.rs`(愿景稿)。

---

## 一、一句话

**模型层是纯函数树:描述运算而不执行;GPU 是一个解释器;错误只在两个边界上
存在(构造期装载、执行期资源操作),中间整条 forward 链——无 Result、无 async、
无相位分支。**

动机:显存/悬空 bug(candle/mistral.rs/xinfer/ owl 自身判例)的共同根源是
"内存生命周期靠人手管理"。声明式把生命周期变成**可计算问题**(依赖图上的
liveness),这是本模型的终局价值;纯函数化只是它的第一块地基。

## 二、核心对象

### 2.1 计划树(反向多叉树;值语义)

```rust
struct Step {
    parents: Vec<Step>,   // 反向边:本节点归约所需全部输入(值语义:完整持有子树)
    depth: u32,           // 拓扑深度(归约排序;兼作错误归因坐标)
    op: Op,               // 语义枚举(Matmul/Rmsnorm/SlotWrite/...)
    shape: Shape,
    err: Option<String>,  // 毒值:构造期违约随链流动,边界收割
}
struct Tensor { head: Step, shape: Shape }   // Tensor = 自己那棵反向树的根
```

- **存储裁决(2026-09-23 用户拍板):不用 Arc,值语义**。每个节点完整持有输入
  子树(深拷贝)。理由:声明树是配置面产物(启动期一次,非热路径),几百节点
  的深拷贝在 Rust 里纳秒级;换来零引用计数、零共享别名、Send/Sync 自动成立。
- 既定代价:重复消费 = 重复计算、树体积 O(n²)——配置面一次性成本,可接受;
  热点出现时在解释器层加结构哈希 memo,不动类型。
- **归约 = 自叶向根**:叶子(Host/Zeros)是数据源;每节点等全部输入就绪;
  root 的值 = 输出。任何中间值可沿 parents 反演(错误归因/liveness 的输入)。

### 2.2 Tx(发射上下文)

forward 的第一个参数。对模型层它只是"把声明记进节目单"的手柄:
`tx.matmul(a, b)` / `tx.slot_write(kv, slots, &v)`——同步,不可败,纯追加。
Tx 的相位(eager 直发解释器 / 捕获录制解释器)由 runner 决定,模型层无感
(seam 核心原则的落地形态)。

## 三、layer / model 形态(全链签名定稿)

```rust
impl MLP {
    /// 构造期(P 阶段):**唯一保留 Result 的模型层位置**。
    /// 静态依赖全部在此捕获:权重装载、comm、quant、外部管理器引用(kv 等)。
    pub fn new(vb: &VarBuilderX, comm: Rc<Comm>, kv: Rc<KvManager>, ...) -> Result<Self>;

    /// forward:同步 · total · 纯描述。无 Result、无 async、无相位分支。
    /// 静态依赖在 self;动态依赖(kv 上下文)走参数;返回惰性声明。
    pub fn forward(&self, tx: &mut Tx, xs: &Tensor, kv: &KvCtx) -> Tensor;
}
```

迁移与现状(owl/xinfer)的同构性:`forward(&self, xs) -> Result<Tensor>` →
`forward(&self, tx, xs, kv) -> Tensor`。改动 = 加 tx 参数、去 Result、
算子调用 `op(..)?` → `tx.op(..)`。MLP/attention/qwen3_5 骨架不动。

### 3.1 外部依赖二分(2026-09-23 用户裁决)

| 类 | 传法 | 例子 |
|---|---|---|
| 静态依赖(影响"这个 layer 是什么") | 构造函数传引用,layer 持有 | 权重、Comm、QuantCfg、**KvManager 本体** |
| 动态依赖(随步变的数据/上下文) | forward 参数 | KvCtx(slots/step)、positions |

### 3.2 KV 副作用:显式声明节点

`tx.slot_write(kv, slots, &v)` 不是 opaque 调用——它声明"本节目写 kv manager
的这些格"。由此跨节目依赖机械可导:同一状态格的写节目与读节目之间插事件边
(单流当下靠流序;多流/多卡时是正确性的门)。**传引用进来可以,写动作必须
走 tx 声明**——治理不失守的交汇点。

## 四、执行边界(全仓库剩余 Result/async 的完整清单)

| 位置 | 可败/异步原因 |
|---|---|
| layer/model `new` | 权重装载(键/形状/dtype)——P 阶段 |
| `Client::spawn` | 设备/池初始化 |
| `interpret(program)` / `to_host` / `sync` | 资源操作与等待——执行边界 |
| **其余全部(layer/forward/采样/视图)** | ——— **纯** ——— |

## 五、解释器(同一棵反向树,三种游走)

| 解释器 | 行为 | 场景 |
|---|---|---|
| eager(档一) | 逐节点提交 server,提交即回 | decode 热路径 |
| capture(烘焙) | 逐节点"录"而非"发",整单 → instantiate → 哨兵③对账 | 图捕获 |
| CPU(测试) | 纯 Rust 闭包归约,零 GPU | nn 全部测试保持同步写法 |

- **warmup = 用 eager 解释器先跑一遍同一份描述**(姿势 6 的制度化,自动正确);
- **哨兵③升级**:对账对象 = 节目单(ground truth)。diff(声明引用集,
  驱动观测集),不一致 = 结构化报错——带外状态(cublas workspace/RNG)
  因此必须声明化为节点(准入由 diff 强制);
- **多档位捕获**:值语义下 base 深拷贝进各档,档位 = 底稿 + 各自尾巴;
  共享前缀的复用由解释器层结构哈希 memo(可选优化,配置面成本本可接受)。

## 六、毒值与错误边界(两族错误,分而治之)

| 族 | 例子 | 传播 | 收割 |
|---|---|---|---|
| 描述层逻辑违约 | 形状不符/dtype 不配 | Poisoned(LazyError{depth, detail})随链透传,算子对毒恒等 | eval 边界结构化报错,depth → 案发节点回溯 |
| 执行层资源错误 | 池耗尽/令牌死亡/拷贝失败 | **不经毒值**——发生在解释器,直接 Err 过线 | 调用方(await 点) |

构造期验证(工厂锁 dtype/shape)使描述层 95% 违约在出生即拦;剩余随毒流动。
playground demo2 为可执行验收。

## 七、CUDA graph 依赖问题的终局回答(设计论证)

graph 依赖问题的本质 = "图引用一组地址,但无人记录这组地址是谁、活多久"。
声明式模型的回答,逐层:

1. **依赖闭包可计算**:图的缓冲集 = 捕获子图 parents 传递闭包。租约机械导出,
   漏登类 bug(P1-2)类型上不存在;
2. **所有权层级化**:Program 持 Arena,块无独立生死 → "块死图活"不可表达
   → AUTO_FREE_ON_LAUNCH、池私有化、手工 lease 全部退役;
3. **liveness 着色**:中间块存活区间 = [出生, 最后消费者],区间不重叠共享
   显存;池容量 = 峰值存活字节(算出,非拍定);
4. **输入格 + 状态格**:bindings(每步写,同流有序)与 KV(跨步,SlotWrite
   声明)显式化,跨图事件边机械可导;
5. **哨兵③永久保留**:静态规划(liveness)+ 运行时对账(diff)双保险,
   有意识的冗余——规划器算错复用 = 合法化悬空,验证臂兜底。

**苦功夫所在(不粉饰)**:带外状态全部声明化(迁移期最大活)、liveness 分析
正确性(含 KV/conv_state 状态区三类生命周期:权重永生/激活单步/KV 跨步)。
周级工程,单独立项:`memory-planner.md`(待立项)。

## 八、迁移账(按序)

> 追记(2026-09-23,用户提出):**loader 同样需要函数化**——它也摸显存
> (H2D 装载就是资源操作),应与 Tensor 同型:描述(装载声明)与执行
> (server 消化)分离,pinned 码头/双拷贝消除一并纳入。**后面再议**
> (A4 loader 项的前置设计)。

1. **A0**:cudarc event-tracking 显式关闭 + 钉测(async-runtime.md §七坑 A);
2. **A1**:server/client/protocol 伪代码定稿落肉(骨架已写:cuda/src/{server,client,protocol}.rs);
3. **A2**:Tx + ops 声明面接入 owl-nn:`op(..)?` → `tx.op(..)`,逐算子;
   CPU 解释器(playground eval)先行,保证 nn 测试全绿不改 async;
4. **A3**:捕获烘焙解释器替换 CaptureSession 调用面;dry_run 迁移验收;
5. **A4**:engine server(Tokio)接入;loader pinned 直填(权重的双拷贝消除);
6. **M-P**:memory-planner 立项(§七)。

## 九、验收口径(可执行规格)

- `playground.rs` 四 demo + GPU 对拍:**当前全绿**(CPU 归约 / GPU 真发射,
  结果逐位一致);
- A2 出口:owl-nn 全测试绿(同步写法不变)+ demo5 等价物跑在 Tx 面上;
- A3 出口:dry_run 判别三连经声明式捕获路径通过;哨兵③对账零 allowlist。
