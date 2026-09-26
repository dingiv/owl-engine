# 解释器观测面(tap)—— 节点级单步调试能力

> 2026-09-26 立项。动机:Qwen3.5-0.8B 真权重 step1 塌零排查暴露的观测缺陷
> —— 现状 `testkit::harvest` 是"逐层整链重放"语义,对带状态层(GDN
> conv/delta 写状态)每观测一次就多执行一遍,**观测行为改变被观测系统**,
> diag 的逐层结论不可信。
> 本设计不修任何具体 bug:把"解释器可在任意算子节点插入并查看中间计算"
> 从测试内联技巧升格为架构能力,塌零排查是这个能力的第一个用户。
> 上游:`charter.md`(公理)、`qwen3-mini-demo.md`(§四 律/C 契约)、
> `async-runtime.md`(四层架构)。

---

## 〇、为什么这是 owl 独有的能力

owl 三层分工:层与模型 = **纯声明**(TensorOps DAG,值语义);解释器 =
唯一执行者(`_eval_node` 逐节点归约,CSE memo);后端 = 字节世界原语。

全部数据流**收口在解释器的节点归约点**。每个中间值都是一块真实的池块,
在 memo 中可寻址(id → Bytes)。因此:

> **声明一次,任一中间量即可寻址、可回读、可统计 —— 零重放、零污染。**

对照业界引擎,这件事结构性做不到:

| 引擎形态 | 中间量状态 | 节点级观测 |
|---|---|---|
| vLLM / sglang(CUDA graph 回放) | 回放态图内,不可寻址 | ✗ 只能整段输入/输出 |
| xinfer(冻结判例) | eager 但无节点坐标 | ✗ 人工插桩逐点编译 |
| HF / candle(命令式) | 中间量在宿主张量里 | △ 手工断点,逐处改代码 |
| **owl(声明式 + 解释器)** | **每节点 memo 池块** | **✓ tap 声明一次,全图生效** |

命令式框架要观测必须改模型代码(断点/打印散落各处);owl 的模型代码
零改动 —— 观测是解释器侧的**声明**,与计算声明同范式。这就是
"层只声明、执行归解释器"哲学的红利:执行收口 = 观测收口。

---

## 一、病根解剖:harvest 重放污染(本案直接动因)

`gpu_model_step_diag`(specs/qwen35.rs)的观测方式:

```rust,ignore
for layer in layers {                      // 逐层
    xs = layer.forward(&xs, &sub);         // 声明链条增长一层
    let got = harvest(&xs).await;          // ← 整链重放!embed..L_i 全部重算
}
```

`harvest` = `eval_ops(t)` 独立归约一次。链条第 i 层的 harvest 会把
L0..Li 的**全部 kernel 再执行一遍**:

- 纯函数 kernel:浪费算力,数值不变 —— 旧测试(Layer 五批)全是纯函数,
  重放无害,所以这颗雷一直没炸;
- **带状态 kernel(GDN conv_upd / delta_dec 写状态槽)**:第 k 次重放
  读到的是第 k-1 次滑过的状态 → 观测值 ≠ 单遍真值。实测现象与之吻合:
  "L0 起 hidden 全零"与"embed 单独收割非零"互相矛盾 —— 前者是重放污染
  的读数,后者才是块数据的真相(320/320 checksum 已证块数据正确)。

**教训入律(候选 §四 律 25)**:带状态声明的观测禁止重放;观测必须
寄生于单遍归约。本设计把这条律做成结构保证,而不是纪律约定。

> **⚠️ 结案补记(2026-09-26当晚)**:旧 diag 的“L0 起全零”事后定谳为
> **双重缺陷叠加**,本能力首战即双重破案:
> ① **主犯 = D2H/COMPUTE 跨流竞速**(server 侧):dtoh 在 STREAM_D2H 异步
>    拷贝,不等 STREAM_COMPUTE 上的产出 kernel —— 早读的块是刚 alloc
>    的零块,晚读的正确(旧 smoke step1 top@248319 全零 = 竞速输掉的
>    那次)。修复:handle_dtoh 先排空 COMPUTE( owl-cuda server.rs);
> ② **从犯 = harvest 重放污染**:带状态层每观测一次多执行一遍,污染
>    跨步状态(真缺陷,但不是全零读数的来源)。
> 修复后单遍曲线:24 层全非零、窗口读数与 sync 后重读逐位一致,
> step0 top@84 logit=14.21 / **step1 top@279 logit=14.91(塔零消失)**;
> 59/59 测试全绿(含原 OOM 误报的 smoke)。

---

## 二、事件词汇(观察什么)

### §2.1 节点事件

```rust,ignore
/// 节点归约完成事件(元数据借用声明节点;数据块引用)
pub struct NodeEvent<'a> {
    pub id: u64,              // 节点身份证(同 id = 同节点 = 同值,C2)
    pub op: &'a Op,           // 语义算子
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub depth: u32,           // 拓扑深度(错误归因坐标,现成)
    pub tag: Option<&'a str>, // 语义坐标(§三;层根标签)
    pub out: BlockRef,        // 输出块(已入 memo)
}

pub struct BlockRef { pub block_id: u64, pub len: usize }  // Bytes 的只读投影
```

### §2.2 观测统计(数值指纹的最小集)

```rust,ignore
pub struct BlockStats {
    pub rms: f32,
    pub min: f32,
    pub max: f32,
    pub zeros: usize,         // 塌零检测的直接判据
    pub first8: [f32; 8],     // 样本(对拍粗定位)
}
```

全量 dtoh 永远可得(§3.2 `Want::Bytes`);Stats 是"每节点都想看"时的
经济形态(80B 回传 vs 全块)。

---

## 三、语义坐标:tag(声明式 API 的结合点)

**观测的第一难题不是读数,是定位** —— "第 37 号节点塌了"无人能读;
"layers.5.linear_attn 输出塌了"才是调试语言。而节点 id 每步重新声明
都变(全局自增),不能当坐标用。

### §3.1 标注字段(纯标注,不动值世界)

TensorOps 增加标注族字段(与 dtype/shape 同族——**描述而非身份**):

```rust,ignore
pub(crate) label: Option<Arc<str>>,   // 语义坐标;Arc 保 clone O(1)

impl TensorOps {
    /// 打标:clone 本节点设 label(同 id 不变 → memo/对齐不受影响)
    pub fn tag(self, s: impl Into<Arc<str>>) -> TensorOps { ... }
}
```

- clone 保留标注(同 id 不变量追加:同 id 必同 label);
- 不开 tag:一个 `Option` 字段,执行路径零参与;
- tag 与 CSE:memo 键 = id,标注只是乘客,**语义零影响**。

### §3.2 层根自动打标(Model 是唯一知道层坐标的声明者)

`Model::last_hidden` 的层循环里,每层产出的根节点自动打标:

```rust,ignore
xs = layer.forward(&xs, &sub).tag(format!("L{li}.{}", if full {"attn"} else {"gdn"}));
// embed → "embed";终局 norm → "final_norm";logits → "logits"
```

- 注入点在**声明侧**,免费获得 —— 不需要在解释器里反推层边界
  (解释器看裸 DAG,永远不知道"层"是什么;层坐标是 Model 的声明知识);
- tag 成本 = 每层根节点一次浅 clone(Arc parents,O(1));
- 层内热点由层自行打标(如 gdn 的 `…/delta_dec`),**可选不默认** ——
  MVP 靠"层根 tag + op/depth 过滤"已够用(塌零二分:先层根,再入层)。

---

## 四、Tap 协议(怎么插)

### §4.1 核心裁决:tap 只声明想要什么,执行归解释器

face(`DeviceClient`)是 async 且独占借用;tap 若自己摸 face,要么
async trait(dyn 不可用)要么借用冲突。**裁决:观测动作由解释器代执行,
tap 是纯同步声明者** —— 与"把副作用移出层、执行归解释器"同构:
观测也归解释器,tap 连执行的权力都没有。

```rust,ignore
pub trait Tap: Send {
    /// 每节点归约完成(输出已入 memo)调用;返回观测意愿。
    fn on_node(&mut self, ev: &NodeEvent<'_>) -> Want { Want::Quiet }
    /// 解释器代执行的统计回传(on_node 返 Stats 时)
    fn on_stats(&mut self, ev_id: u64, s: BlockStats) {}
    /// 全量回传(host f32;on_node 返 Bytes 时)
    fn on_bytes(&mut self, ev: &NodeEvent<'_>, host: &[f32]) {}
    /// 毒值落地(Err 路径;观测流里也能看到"死在哪")
    fn on_poison(&mut self, id: u64, detail: &str) {}
}

pub enum Want {
    Quiet,   // 不观测(默认;零代价)
    Stats,   // 解释器 dtoh → 算 BlockStats → on_stats
    Bytes,   // 解释器 dtoh 全量 → on_bytes
    Keep,    // (二期)钉住本块:eval 结束后仍可读(§六)
}
```

- 全默认方法 → 一个闭包/结构体实现一个方法就是完整 tap;
- 同步 trait → `&mut dyn Tap` 可用,eval 热路径无泛型膨胀;
- on_node 里做谓词过滤(tag/op/depth 匹配才返 Stats)→
  "层根曲线"探针 5 行,"全节点指纹"也是同一个 trait。

### §4.2 注入点与归约语义

```rust,ignore
// interpreters/eval.rs
EvalCtx { face, memo, tap: Option<&mut dyn Tap> }        // ctx 增一字段

pub fn eval_ops_tap<'a, D>(t, face, tap: &mut dyn Tap) -> ...   // 新入口
pub fn eval_ops<'a, D>(t, face)                    // = eval_ops_tap(t, face, &mut QuietTap)
```

`_eval_node` 归约序列不变,只加三个事件点(≈10 行):

1. **After**:输出入 memo 后 → `tap.on_node(ev)`;按 Want 代执行
   dtoh/stats → 回传;
2. **Poison**:毒值落地(Err 路径)前 → `on_poison`(毒值节点无输出块,
   无 After;归约仍照旧 Err);
3. ~~Before~~:不设 —— 输入观测 = 父节点的 After(父块同在 memo),
   二期 Halt 断点再引入。

**顺序裁决**:毒值检查先于 After(毒值节点直接 Err,见 2);C1 断言
(debug_assert panic)位置不动 —— tap 是旁路,不改变任何归约语义。

### §4.3 窗口律(候选 §四 律 25 / C14,实施时回写)

> **观测窗口律**:tap 的数据读取(Stats/Bytes)由解释器在 After 窗口内
> 代执行完毕。块存活期 = memo 存活期(池 scratch 块,eval 结束句柄失效,
> 池可复用);**窗口外读块未定义**。tap 无 launch/alloc/htod 权 ——
> 观测在结构上不可能改变计算(对照 §一 重放污染:病根是观测自带执行)。
> 读序保证 = `DeviceClient::dtoh` **读语义**(iface contract「顺序语义」,
> 2026-09-26 塔零案后成文:issue 排空 COMPUTE)—— 窗口读不再依赖竞速。

MVP 纪律:统计都在窗口内完成,不依赖窗口外存活。事后回看 = `Want::Keep`
(二期,§六)。

---

## 五、双锚逐节点对拍(自动 bisect;塌零终局手段)

`reference.rs` 的同步参考解释器(永不优化的对拍锚)走**装饰器**,零侵入:

```rust,ignore
pub struct TapInterpreter<I: Interpreter> { inner: I, tap: Box<dyn Tap> }
impl Interpreter for TapInterpreter<I> { /* 每方法前后发同款 NodeEvent;
    值已在手上(Values 即数据),stats 本地即算,窗口律天然满足 */ }
```

**对拍协议**(同进程声明一次,双锚各求值一遍):

1. 同一棵声明树,GPU 走 `eval_ops_tap`,CPU 走 `reduce(TapInterpreter)`
   —— id 在声明期已定,**两锚天然同 id 对齐**(无需 coord 匹配);
2. 两侧 StatsTap 收 `Vec<(id, BlockStats)>`,按 id 拉链,首个超阈节点 =
   **自动 bisect,一次跑完**(对照旧方式:bisect 13 段人工切片两小时);
3. 判读表:

| 现象 | 定性 | 下一步 |
|---|---|---|
| 双锚在某节点分叉 | 后端 kernel / 解释器翻译 bug | 该 kernel 单测对拍 |
| 双锚一致,与 HF 不一致 | 层语义 bug | HF parity 套件(src/layers/*.py)对拍 |
| 双锚一致,HF 也一致,但 step1 仍塌 | 状态推进/跨步 bug | 层根曲线对比 step0/step1 |

跨进程/跨次声明场景(id 不相通)用 coord 对齐:
`(tag, op 名, depth, 同层同 op 序号)` —— 人类阅读主键,机器对齐退路。

---

## 六、二期形态(先裁形态,不实施)

- **Keep(事后回看)**:on_node 返 Keep → 解释器把该块转 persistent
  分配(或挪入 keep 集免归池);eval 结束后句柄仍可 dtoh。供 REPL 式
  交互调试/多轮假设检验;内存代价 = keep 名单显式承担。
- **Halt(暂停式断点)**:on_node 内阻塞等调试器指令(通道),检查请求
  仍由解释器代执行(face 独占借用 → 串行化天然成立)。单步 = 每节点
  Halt;条件断点 = coord 谓词。需要调试器宿主(REPL/协议),另立项。
- **poke(写值假设检验)**:窗口内允许向指定块写值(清零某中间量看下游
  变化 = 因果验证的实验法)。违反"tap 只读"窗口律 → 独立能力(Experimental
  Tap),不进本 trait。

---

## 七、与 M-f(图捕获)的边界

- **eager 永远全量观测**(本设计的主场);
- 回放态:中间块在图内不可寻址(A1.5:热路径禁 D2H/同步)→ tap 在
  `graph_begin` 后强制失效(debug_assert),降级形态 = 段边界快照
  (Session 输出槽,见 session-plan.md)——eager 调试 → 图化生产,
  中间态观测留在 eager 调试期完成;
- 两者不冲突:捕获前 warmup 本来就要求同一声明树 eager 跑通(iface
  graph_begin 护栏)——**warmup 即全量观测期**,tap 顺水推舟。

---

## 八、零开销与纪律

| 场景 | 成本 |
|---|---|
| 不挂 tap(eval_ops 原入口) | ctx 一个 `Option` 检查/节点(≈免费) |
| 挂 tap + Quiet | 一次虚调用/节点(调试模式,性能不设防) |
| Stats | dtoh 回传 80B + host 计算(MVP);优化挂账:stats kernel 服务端一次归约回传 20B |
| tag 打标 | 声明期每层根一次浅 clone(O(1));执行期零参与 |

- 观测只读 + 解释器代执行(§4.3)→ **结构性免疫重放污染**;
- `testkit::harvest` 保留但注释降级:"重放语义,禁用于带状态层;
  带状态观测走 tap"—— 纯函数小对拍(五批层测试)继续用,不迁移;
- 值世界无损:label 是标注不是身份;同 id 必同 label(不变量追加)。

---

## 九、实施落点

| 文件 | 变更 |
|---|---|
| `interpreters/observe.rs`(新) | NodeEvent/BlockRef/BlockStats/Want/Tap + StatsTap/QuietTap + 事件词汇(新变体按域另起文件纪律) |
| `interpreters/eval.rs` | EvalCtx 增 tap;_eval_node 三事件点(≈10 行);eval_ops_tap 入口 |
| `interpreters/mod.rs` | 导出 + 变体表加一行 |
| `tensor.rs` | label 字段 + tag 组合子(标注族;同 id 必同 label) |
| `model.rs` | last_hidden 层根/embed/final_norm/logits 自动打标 |
| `reference.rs` | TapInterpreter 装饰器(§五) |
| `testkit.rs` | harvest 注释降级;公共探针件(rms 曲线打印/golden 指纹) |
| `specs/qwen35.rs` | gpu_model_step_diag 重写为 tap 探针版(单遍,零重放) |
| 律回写 | qwen3-mini-demo §四 律 25(观测窗口律);候选 C14 |

## 十、里程碑

- **T-a(观测面 MVP)**:词汇 + tag + tap 注入 + StatsTap;真模型
  step0/step1 层根 rms 曲线(单遍)→ **塌零层定位**;
- **T-b(双锚对拍)**:TapInterpreter 装饰器 + id 拉链比对 → 首个发散
  节点;顺带 smoke 加固(golden 指纹锁,曲线哈希入断言,全零必红);
- **T-c(交互)**:Keep 钉块;Halt 暂停式断点(通道形态已裁,§六)。

## 变更记录

| 日期 | 变更 |
|---|---|
| 2026-09-26 | 立项设计:harvest 重放污染定谳为方法论病根;tap 协议(声明/执行分离:tap 只声明 Want,解释器代执行);tag 语义坐标;双锚 id 对齐协议;窗口律草案 |
| 2026-09-26 | **T-a 落地即结案**:observe.rs/eval.rs 三事件点/tensor.rs label+tag/model.rs 层根打标/reference.rs reduce_tap/TapChain/StatsTap;首战定谳塔零案 = D2H/COMPUTE 跨流竞速(主犯,server handle_dtoh 排空修复)+ harvest 重放污染(从犯);step1 top@279 恢复正常,59/59 全绿 |
