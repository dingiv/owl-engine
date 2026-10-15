# Piecewise(可断)捕获图 —— 设计与实现方案(2026-10-15;perf 会话立项)

> **触发**:V1 核级验证已达标(FI+split 在 verify 形状 0.249 ms/发 vs XQA 1.541),
> 但**换不进引擎** —— 现役 verify/decode 是**捕获图**,而 FI 的 plan 在 split 档**依赖 ctx**
> (chunk 数 / `o_indptr` / `merge_indptr` / `padded_batch`),一张图只对一个 ctx 有效。
> 仓库里没有任何 piecewise 实现 ⇒ 本文从零设计。
>
> 价值:verify 24.66 ms/轮 → ≈4.0 ⇒ **decode +42%**(32k);裸 decode 步 14.52 → ≈1.9 ms ⇒ **+50%**。
> 复用地:prefill-graph 设计的 M2c、以及"想换任意核但被 ctx 卡住"的所有场景。

---

## 1. 与仓库现有机制的关系(先划清边界)

| 现有件 | 位置 | 本设计怎么用 |
|---|---|---|
| **声明树 + 解释器**(`TensorOps` DAG、`_eval_rec`、**`EvalCtx.memo` 按节点 id 缓存**) | `crates/models/src/interpreters/eval.rs` | **断点的天然落点**:树执行天然有"节点序";memo 就是跨段的接力棒 |
| `GraphPlan::{plan,step}` + `InputSlot/OutputSlot` | `engine/src/graph_plan.rs` | 段 = 一次 `graph_begin..graph_end`;槽面复用 |
| `graph_begin/graph_end` + `CaptureState{stream,slab,used}` | `backends/cuda/src/state.rs:456/487` | **要改**:现在是"一 plan 一窗";piecewise 要"一 plan 一条 slab + N 段" |
| `Block::{Owned,Carved}`、`carve_block`、capture 窗内 `free_blocks` 走 arena 归还 | 同上 + M1/S1.5 | **要动**:跨段存活的值必须**钉住**,否则被归还后地址复用 ⇒ 前一段的烘焙指针指向别人 |
| 捕获窗白名单(G2) | `backends/cuda/src/server/mod.rs` | 断点在**窗外**执行 ⇒ 不涉白名单;但需新增"续段"语义 |
| fold 桶族(m=0..6) | `engine/src/graph_fleet.rs` | **先例**:一族多图、逐轮按 m 选图 —— piecewise 是它的"段内版" |

**关键判断**:owl 的执行面不是"逐算子 imperative 循环",而是**一张声明树交给解释器**。
⇒ piecewise 不该做成"引擎里手写 N 个闭包",而应做成**解释器里的断点原语**(在遍历到被标记的节点时切段),
这样它与模型结构无关(27B/其他模型/其他臂都能用),也不会退化成"每层手抄一遍前向"。

---

## 2. 为什么必须 piecewise(而不是别的三条路)

| 路 | 为什么不行 |
|---|---|
| **A. FI 无 split 进图** | plan 与 ctx 无关 ✓(F-5),但 T=8/ctx32k = **3.604 ms/发**(纯延迟,4 CTA)⇒ 16 层 = 57.7 ms/轮 ⇒ **比 XQA 还慢** ✗ |
| **B. 按 ctx 桶预捕 verify 图** | 桶内 kv_len 仍会漂 ⇒ plan 的 chunk 数与核在运行期算的 `ceil_div(kv_len, chunk)` 不一致 ⇒ partial 行 stride 与 `merge_indptr` 打架(**正是 V1 抓到的那个 bug 的形态**)⇒ 除非把"行 stride"改成**传入常量**(改上游核)|
| **C. 继续修 XQA** | 几何扫描已证明只有 3% 空间;`rowsPerBlock ≤ 32` + 2 token block 是**内核结构**(实测 2.09 TFLOPS = 5% 峰值)|
| **D. 引擎级 split(自己切窗口多次调 FI)** | 截断 KV 窗口会**改因果掩码边界**(行 i 的上界 = `kv_len - qo_len + i + 1`)⇒ 塌成错的 mask ✗ |

⇒ 只有"**注意力在窗外 eager 跑、其余在段里回放**"这一条既保图收益、又让 plan 按需重算。

---

## 3. 机制

### 3.1 三个概念(一句话各一条)

1. **段(segment)**:一次 `graph_begin → … → graph_end` 捕获出的一张子图。
2. **断点(break)**:被标记的节点(`Op::Call` 一族,本期 = attention)。**它的求值永远在窗外**(eager),
   且在**捕获期真跑一次**(否则下一段拿不到输入)。
3. **钉块(pin)**:断点的**输出块**,由 plan 在捕获前一次性分配并持有 ⇒ 断点在捕获/回放两期写**同一地址**。

### 3.2 捕获期时序(一次 plan)

```
pin_blocks = 为每个断点分配输出块(池块,plan 持有)
graph_begin ─► 求值 seg0(解释器遇到断点即停) ─► graph_end ⇒ g0
              eager 求值断点0(**写 pin0**,不在任何窗内;FI 的 plan 在这里算,可 H2D)
graph_begin ─► 求值 seg1 ─► graph_end ⇒ g1
              eager 求值断点1 …
…
graph_begin ─► 尾段 ─► graph_end ⇒ gN
⇒ PiecewisePlan { segments: [g0..gN], breaks: [b0..bN-1], pins: [pin0..], slots }
```

### 3.3 回放期时序(每轮)

```
step(slots): 装填输入槽(write_block:ids/pos/slots/lens/seq_cu/**FI 四件套**)
for i in 0..N:
    graph_launch(seg[i])
    breaks[i].run()      ← eager:同一个节点、同一批输入块、写同一 pin 块
                            (FI 的 plan 在这里按 (T,ctx) 命中/重算;~0.1 ms/轮)
graph_launch(seg[N])
```
⇒ 17 段、16 个断点(verify:16 个全注意力层)。图内节点**一个没少**(1484−16),
所以**派发税不变**(16.5 ms/轮);换掉的只有注意力的 24.66 → 4.0 ms。

### 3.4 树怎么切(切点定义)

- 切点 = "父树里引用 attention 节点的那条边":`A` 的**操作数子图**归上一段,`A` 本身是断点,
  `A` 的**消费者**归下一段。
- 实现上不需要真的改树:**解释器在遍历时按 `memo` 判断"该节点是否已算"**,遇到断点标记就:
  - 捕获期:`graph_end` → 记段 → eager 跑该节点(写 pin)→ `graph_begin` → 继续(结果塞进 `memo`);
  - 回放期:段序列 + 断点闭包顺序驱动(上面的循环)。
- 这与 vLLM `breakable_cudagraph` / SGLang `eager_on_graph` 同构,但**贴合声明树**:断点是**节点 id 集合**,
  不是"函数装饰器"。

---

## 4. 要落的 API(四处;都带落点)

| # | 面 | 改动 | 量 |
|---|---|---|---|
| **A1** | 解释器:`EvalCtx` 增 **`overrides: HashMap<node_id, Bytes>`**(写进给定块)+ **`pins: HashSet<node_id>`**(禁止归还)+ memo 可跨调用带走 | `crates/models/src/interpreters/eval.rs`:`alloc_out` 调用点(635/658/747)改成"有 override 就用它";末尾回收处排除 `pins` | ~40 行 |
| **A2** | 解释器:断点钩子 **`trait BreakCtx { fn at_break(&mut self, node: &TensorOps) -> Option<Bytes> }`**;`eval_ops_multi_*` 增可选 `break_ctx` 与 `memo_in/out` | 同上 + `eval.rs` 的 `_eval_rec` 入口 | ~60 行 |
| **A3** | 捕获面:**一条 plan = 一条 slab + N 段**。`graph_end` 不再 `take()` 掉 slab/arena(改成"结算当前段");新增 `graph_end_segment()`/`graph_begin_segment()`(或 `graph_begin{continue_session:bool}`);`free_blocks` 对 Carved 的归还**在段间保持**(但 `pins` 不许还) | `backends/cuda/src/state.rs`(CaptureState 所有权)+ `server/mod.rs`(命令白名单/语义) | ~120 行 |
| **A4** | plan 侧:**`PiecewisePlan`**(`segments: Vec<GraphId>` + `breaks: Vec<BreakOp>` + `pins` + `slots` + `output_blocks`)+ `step()` 的段循环;`PlanOutcome::CapturedSegmented{..}` | `engine/src/graph_plan.rs` | ~150 行 |

**A2 的 `BreakOp` 内容**(回放期可重放的最小闭包):
```rust
struct BreakOp {
    node: TensorOps,        // 断点节点(操作数全是稳定块叶 ⇒ 重跑零递归)
    out: Bytes,             // pin 块(捕获期分配,plan 持有)
    kind: BreakKind,        // 本期只有 Attention(FI/XQA);留 GDN? —— 见 I-4,禁止
}
impl BreakOp { async fn run<D: DeviceClient>(&self, face: &mut D) -> Result<()> {
    // 等价于 interpreters::eval_ops_into(self.node.step(), self.out.clone(), face)
}}
```

---

## 5. 不变量(每条都要有守门;缺一条就是静默错值)

| # | 不变量 | 为什么 | 守门 |
|---|---|---|---|
| **I-1** | 断点的**输入**必须是稳定块(上段产出的 Carved/池块,或持久槽) | 回放期要按同地址读 | 捕获期断言:断点操作数全为 `Op::Block` 叶;否则拒捕(回退 eager 路) |
| **I-2** | 断点的**输出** = plan 持有的 pin 块,捕获/回放同址 | 下一段烘焙的是这个地址 | `pins` 参与回收排除(A1);plan 持有句柄 |
| **I-3** | **段间 slab/arena 状态不能把 pin 块归还** | `graph_end` 现会 `take()` 掉 capture;归还后被复用 ⇒ 前段指针指向别人 | A3 的"结算当前段"语义 + A1 的 `pins` 排除 + 一条**捕获期断言**:归还集合 ∩ pins = ∅ |
| **I-4** | **断点必须是幂等/无状态的** | 捕获期跑一次 + 每轮回放跑一次 | 白名单:本期只允许注意力(FI/XQA)。**GDN 递推/KV 写禁入断点**(KV 写虽幂等,但它的副作用面与 fold/snapshot 耦合,先不碰) |
| **I-5** | 断点的**宿主动作在窗外**:plan 计算、H2D、任何 sync 都不许出现在两段之间? | 窗外 = 不在捕获态 ✓ 但**捕获期的断点是 eager 且窗已关** ⇒ H2D 合法 | 断点执行前后由 `GraphPlan` 断言 `capturing()==false`(服务端 `res.capturing()` 现成) |
| **I-6** | 段数/断点数在捕获后**冻结** | 段列表即语义 | `PiecewisePlan` 不可变;新增断点 = 重捕 |

---

## 6. 成本账(verify @32k;16 断点/17 段)

| 项 | 现在(单图+XQA) | piecewise(FI) |
|---|---:|---:|
| 图内派发税(1484 节点 × ~11 µs) | 16.5 ms | 16.4 ms(少 16 个节点) |
| 注意力设备时间 | 24.66 ms | **≈4.0 ms**(FI+split 0.249×16) |
| 段数量附加(17 × 图回放 ~20 µs) | ~0 | 0.34 ms |
| 断点宿主(16 × ~30 µs) + plan(1/轮 ~0.1 ms) | ~0 | 0.6 ms |
| **每轮合计** | **~70 ms** | **~50 ms**(−29%) |

判据:
1. **图内一致门(硬)**:piecewise-FI 回放 ≡ 同一 plan 的 eager-FI(逐位);
2. **跨臂门(软)**:piecewise-FI vs 现役 graph-XQA ⇒ 数值 ≤1e-3(探针已证 5e-4)+ 贪心逐 token 一致;
3. 段/断点数与预期一致(17/16),`/debug/launch` 账可见;
4. decode t/s:32k 档 **+35% 以上**(先过 42% 的理论值再谈优化)。

---

## 7. 分期(每期独立可回滚)

| 期 | 内容 | 判据 |
|---|---|---|
| **P0(半天)** | 解释器侧 A1+A2 的独立单测:一个两段一断点的小树,断点写 pin 块,验证 memo/override/pins 三件语义 | 单测全绿(CPU face 即可) |
| **P1(1 天)** | 捕获面 A3:一条 slab 多段;**段间归还守卫**;`server` 侧的续段命令 | `gpu_server.rs` 级测试:两段空图 + 中间 eager 写,回放两次结果正确 |
| **P2(1-2 天)** | `PiecewisePlan`(A4)+ **verify 图改 piecewise**:断点 = 16 个注意力节点;注意力仍走 XQA(先验机制,不换臂) | **图内一致门** + 端到端轮时不变(±1%)——机制零收益时先证不回归 |
| **P3(0.5 天)** | 断点换 **FI+split**(给 verify 路造 FI 四件套:`q_cu=[0,depth+1]`/`indices`=页链/`indptr`/`last_len`,全部走输入槽) | 跨臂门 + **轮时 ≤52 ms @32k** |
| **P4(可选)** | 同一机制复用到 prefill 图(M2c)、或裸 decode 图(XQA→FI) | 各自判据 |

---

## 8. 风险与逃生阀

| # | 风险 | 逃生阀 |
|---|---|---|
| R1 | **段间地址复用**(最危险,P0/P1 专治) | 捕获期断言(归还 ∩ pins = ∅)+ 回放期"两次回放结果一致"门 |
| R2 | FI plan 在断点里算 ⇒ 首次进新 ctx 有抖动 | plan 预热(按桶预扫 ctx 列表)或接受(实测 ~0.1 ms) |
| R3 | 段数多 ⇒ 回放启动开销 | 段合并(两层一段);实测 17 × 20 µs 可忽略 |
| R4 | 注意力换臂带来的数值差(5e-4) | 已由 FI 探针 host oracle 背书;门按"跨臂 ≤1e-3 + 逐 token 一致" |
| R5 | 与 main 会话的 prefill-graph 工作冲突(同一批文件) | 本设计**先只落在 perf worktree**;合并时以 `GraphPlan` 的段列表为唯一新增面 |

---

## 9. ⚠️ 对 main 会话 prefill-graph 设计的回写(必须同步)

`docs/arch/prefill-graph-设计与实现.md` 的 **M2b「整段单图」前提 P-1「plan 与 ctx 无关」**
只在 **`disable_split_kv = true`(split 关)** 时成立 —— 依据是 F-5:
"`kv_len_arr`/`effective_kv_len_arr`/`kv_chunk_size` 三条链只在 split 分支被消费"。
**split 一开,plan 立刻变成 ctx 依赖**(`num_chunks_kv` ⇒ `o_indptr`/`merge_indptr`/`padded_batch`),
那张图就只能对一个 ctx 有效 ⇒ M2b 的整段单图**不再成立**,必须改走 M2c(piecewise)。

⇒ 两条设计现在**汇到同一条路**上:**piecewise 是主线,不是后备**。
两边应共用同一份断点原语(本次 A1/A2 的接口就是按"能被两条线同时用"设计的:
断点只认"节点 + pin 块 + 幂等"三件事,与是 prefill 还是 verify 无关)。

---

## 10. 施工记录:P0 / P1(2026-10-15 晚)

### 10.1 P0 ✅ 已落(`adf5b47`)—— 求值面原语

| 件 | 落点 | 说明 |
|---|---|---|
| `EvalCarry { memo, keep }` + `resolve(node, block)` | `interpreters/eval.rs` | 段间最小状态;`resolve` = 调用方登记"外部已决"节点(断点) |
| `eval_segment(roots, face, env, carry)` | 同上 | 在接力下求值可达子图;段末只把 `keep` 命中的值留给下一段 |
| `EvalCtx.{keep_nodes, keep_blocks}` + `memo_put` + `is_kept/_node` | 同上 | 两个回收出口(`release_block`、中途 `block_users` 归零)与段末候选表全部绕开 keep 块 |
| `CpuFace::freed_ids()`(归还记账面) | `backends/cpu/src/face.rs` | additive;不焚数据 ⇒ 存量"free 后仍可读"语义不变。**没有它,"kept 块没被归还"这条判据在 CPU 面无法观测** |

**设计简化(比原设计 §4 A2 少一层)**:原设计要"断点钩子 `BreakCtx::at_break` + 段内暂停"。
实际**不需要** —— 每段的根 = **下一个断点的操作数**,`memo` 天然接力;断点在段间由调用方自己
求值并 `resolve` 进 carry。⇒ 解释器里**零图概念、零算子知识**,上层也零算子知识。通用性由
"节点 id + 块 + 幂等"三件事保证。

**P0 门抓到的真问题**:活性回收在"最后一个消费者用完"时 `ctx.memo.remove(&p.id)`,
而 `keep` 的语义恰是"**下一段还要用**" ⇒ 摘 handle = 丢跨段接线(块被保护但拿不到)。
修:keep 节点的 memo 条目不摘。**这条只有真跑才会暴露** —— 三条门(≡整树、存活+阳性对照、
未 keep 不接力)当场逮住。

### 10.2 P1 ✅ 已落(CUDA 面骨架门)—— 并**收缩**了 A3 的范围

原设计 A3 = "一 plan 一条 slab + N 段 + 续段命令"(~120 行)。实测**本树不需要**:

- 本树 `free_blocks` 对 `Block::Carved` 是 **no-op**(捕获窗内不回收)⇒ 段间不可能发生
  "归还→复用"⇒ **每段各用一条 slab 就已经安全**,跨段钉块靠 plan 持有的句柄 + 服务端账房
  (不 free ⇒ 不回收 id)保活 ✓;
- "一条 slab N 段"的真实动机 = 让 slab 容量回到**峰值**(S1.5 的 arena 回收,主会话那条线);
  它落地时才需要 `pin` 的**后端级**守卫。

⇒ A3 收缩为两件:**(a) pin 契约(防御纵深)、(b) 跨会话接口注记**(见 §11),其余等 S1.5。

**已落的门**(`crates/backends/cuda/tests/gpu_server.rs` 步骤 5b,真机绿):
```
seg0 捕获(x+one→out0) → 窗外 eager 断点(mid=out0+out0) → seg1 捕获(mid+one→tail)
回放序列 ×2:graph_launch(g0) → 断点重放 → graph_launch(g1)
判据:tail=5.0、mid=4.0,**两次回放一致** ⇒ 跨段地址接力 + 可重复性成立
```
这条门是**引擎侧 `PiecewisePlan` 的机制前提**:它证明"分段捕获 + 段间窗外 eager 写 +
顺序回放"在真 CUDA 面上按预期工作。

### 10.3 下一步(A4 驱动件;给"别的模块也要用"定的形态)

`engine/src/graph_plan.rs` 增 **`PiecewisePlan`**(与算子无关):

```rust
pub struct PiecewisePlan<D> {
    slots:   Vec<InSlotDev>,        // 复用现有输入槽面(ids/pos/kv 表/FI 四件套…)
    segs:    Vec<GraphId>,          // 段(顺序回放)
    breaks:  Vec<BreakOp>,          // 断点:节点 + pin 块 + 幂等求值闭包
    tail:    GraphId,               // 尾段(= segs.last(),显式留位以便段合并)
    outputs: Vec<(String, Bytes, Dtype, usize)>,
}
impl PiecewisePlan<D> {
    /// 捕获:按 `breaks` 切段,每段 graph_begin → eval_segment(roots=下一断点操作数) → graph_end
    pub async fn plan(face, env, desc, tree, breaks) -> Result<(Self, PlanOutcome)>;
    /// 回放:槽装填 → for (seg, brk) { graph_launch(seg); brk.run(face) } → graph_launch(tail)
    pub async fn step(&mut self, inputs: &[(&str, &[f32])]) -> Result<()>;
}
```
**通用性约束(三条,写死在这里当验收)**:
1. `PiecewisePlan` **不认识** attention/FI/GDN —— 断点只由"节点 id 集合"给定;
2. 段的根由调用方给(`roots_of_next_break`),驱动件不解析模型结构;
3. 幂等性由**调用方声明**(`BreakOp::idempotent: bool` 之类)→ 驱动件只做"重放"。

**收益判据(P3)**:verify 轮时 70 → **≤52 ms @32k**(断点换 FI+split 后 ≤50)。

### 10.4 A4 ✅ 已落 —— 段驱动件(并修了两处接口形态)

| 件 | 落点 | 说明 |
|---|---|---|
| `PlanCtx::break_node(&t)` | `engine/src/graph_plan.rs` | **断点在闭包建树处声明**(见下"形态修正 1") |
| `GraphPlan::plan_piecewise(face, env, desc, idempotent, forward)` | 同上 | 逐段:干跑计量 → 捕获 → 窗外断点;尾段同律 |
| `replay_piecewise()` / `piecewise_shape()` | 同上 | 段→断点→…→尾段顺序回放;段数口径**含尾段**(N 断点 ⇒ N+1 段) |
| `TensorOps::operands()` | `models/src/tensor.rs` | 操作数只读视图(计划器推导段根用) |
| `EvalCarry::overrides` + `override_out` + `EvalCtx::alloc_out_node` | `models/src/interpreters/eval.rs` | 输出覆盖:被覆盖节点写进给定块;覆盖块一律钉住(3 个 alloc 调用点统一走新入口) |

**形态修正 1(结构性,原设计不可行)**:原稿设计"caller 传 `Vec<BreakPoint{node, out}>`" —— **做不到**:
断点节点是在闭包**内部**构造的(它要拿 plan 的槽块),外部根本拿不到那个 `TensorOps`;而 str
`out` 又必须在捕获前分配。⇒ 改成:① 断点在**建树处**由闭包声明(`PlanCtx::break_node`);
② 钉块由**驱动件按声明形状自己分配**(dtype/元素数取自节点)⇒ caller 侧只剩一句幂等表态。

**形态修正 2(显存)**:若每段 `graph_begin` 都用同一份 hint(= 整张 plan 的分配总和),
N 段就是 N 倍显存。⇒ 每段捕获**前**先做一次**逐段干跑**(窗外、独立 scratch carry)
量出本段的分配总和,再 `set_hint` ⇒ 每段 slab 按自己的需要定价,总量回到与"整图"同量级。

**两条门**:
- **CPU 档**(`cargo test -p owl-engine --lib`):① 无图能力 ⇒ `EagerFallback` + eager 臂照常出值
  (不留半成品/段表);② 未声明幂等 ⇒ 拒捕(结构化,不静默);
- **GPU 档**(`engine.e2e` 场景 `gpu_piecewise_segments`,`OWL_E2E_ONLY=piecewise`):
  `y=x+c`(段0)→ `brk=y+y`(**断点,窗外**)→ `out=brk+c`(尾段),
  **真机绿**:捕获 `(段,断点)=(2,1)`、回放 ×2 值恒为 5.0 ✓
  (该场景**不装模型** —— 纯 GraphPlan 面,秒级,与 CUDA 面步骤 5b 一起钉住机制)。

---

## 11. 施工记录:P2 接线 + **真机追凶实录**(2026-10-16 凌晨)

### 11.1 已落(全部在 perf worktree;`dispatch.verify_piecewise` **缺省关** = 生产零变化)

| 件 | 落点 | 说明 |
|---|---|---|
| 断点采集面 `break_sink()` | `engine/src/graph_plan.rs` | `PlanCtx.breaks` 改 `Rc<RefCell<…>>`;模型层 push = 本 plan 登记断点(**单一拼写点**)。⚠️ 踩坑:自建一张表 ⇒ 驱动件永远看到 0 断点(实测"piecewise 无断点"降级)|
| `ForwardCtx.attn_break` | `models/src/module.rs`(+`model.rs::layer_ctx`) | 形态同 `gdn_tap`;7 个构造器全 None(零开销) |
| 层侧登记 | `models/src/layers/attention.rs::note_attn_kernel` | 三处:**XQA verify/decode/FI** 的**核节点**(不含 GateMul/o_proj) |
| `plan_verify` 接线 | `engine/src/graph_fleet.rs` | 开关开 ⇒ `plan_piecewise`(零断点 = **退化单图**,与 `plan` 同形 ⇒ 不静默降级为 eager);开关关 ⇒ 单图原路 |
| **段边界 keep** | `graph_plan.rs` | 见 11.3(不变量 I-7) |
| `SegMode::{Capturing,Eager}` | `models/src/interpreters/eval.rs` | 捕获窗内**禁一切回收**(捕获期 free 被服务端白名单拒:实测"图捕获进行中:该操作被拒");口径 = 单图 path 的 `eval_ops_multi`。dry 计量仍走 Eager(回收 + 段末清扫,不泄漏) |
| 空段/空尾守卫 | `graph_plan.rs` | 根全在 memo ⇒ 不建图(空捕获窗是硬错);`segs: Vec<Option<GraphId>>` + `tail: Option<GraphId>` |
| **图内一致门**(diag) | `graph_plan.rs`(`OWL_PW_SELFCHECK=1`) | 回放一次 vs eager 整树,逐输出比 max\|Δ\|;**NaN 记 ∞** |
| 断点二分(⚙) | `graph_plan.rs`(`OWL_PW_BREAKS=k`) | 只保留前 k 个断点(其余退化为普通节点)⇒ 可对"第几个断点开始错"二分 |
| 取证面(⚙) | `engine/src/exec/spec.rs`(`OWL_PW_DUMP_TOK=1`) | verify `tok[8]`/`hid` 画像/钉块画像/断点操作数画像 |

**两条门**:CPU(`cargo test -p owl-engine --lib` 33/33、`cargo test -p owl-models --lib` 128/128)+ GPU
(`gpu_piecewise_segments` 真机绿:捕获 (段,断点)=(2,1)、双回放值恒 5.0)。**⚠️ 测试要在空闲卡上跑**
(GPU2 上有实例时该场景挂起;GPU0/3 空闲即秒过)。

### 11.2 🔴 **真机 verify 断裂**(未解;P3 的前置)

**症状**:27B + fp8 池 + XQA + `degrade_after=0`(关自适应降级)+ greedy + 哑草稿:
- `k=0`(退化单图,piecewise path、零断点)⇒ **文本与基线逐字一致 ✓**(⇒ 段捕获/harvest/fold/propose 全链无问题)
- `k=1`(只第一个注意力核做断点)⇒ **文本 "用户!!!!…"**(首轮即崩)
- 取证(`OWL_PW_DUMP_TOK`,pos=53):
  - `tok = [-1,-1,…]`(argmax 内核的"无最大值"哨兵 `bi=-1` ⇒ **logits 全非数**)
  - `hid`:40960 元素 **nan=40960**(全非数)
  - **`pw-pin`:钉块 49152 元素 **nan=49152** ⇒ **断点(急切 XQA)自身产出全非数**
  - **`pw-op`(断点操作数)全部有限**:`q`(F16, max=1.02e1)✓ / 池(U32 fp8 字节)✓ / 页表/seq_lens/mask(位型)✓ /
    semaphores=0 ✓ / scratch(F16 16M, 66048 个 NaN —— 池内常驻 scratch,旧内容)✓ / `wr`(1 元素)=0 ✓

⇒ **定位:eager(图外)XQA 发射产出 NaN,而它的输入全有限;同一个核在图内跑是对的。**
差异只剩:①发射路径(捕获 vs 直接)、②`scratch`(常驻池块)里的陈旧 NaN 内容、③图形语义。
**已排除**:跨流无序(两路都在 `STREAM_COMPUTE`)、`set_shared_limit`(assemble 期已设)、
几何/常量(同一 `consts`,同网格/块/smem)、K0 依赖边(同流序)、页表/槽(逐轮写 ✓)、
断点机制本身(`k=0` 绿 + 段边界 keep 已修)。

**⇒ 下一刀(按性价比)**:
1. **断点改挂注意力输出**(`GateMul`+`o_proj` 之后,纯 GEMM 节点)⇒ 二分"eager XQA 发射"vs"分裂机制";
   若这一步绿 ⇒ 收敛到 XQA 发射面(与机制无关)。
2. **scratch 归因**:在同一 boot 里把 `scratch` 清零后重放一次(或在 `plan_piecewise` 捕获前 `memset_zero` scratch)
   ⇒ 判"陈旧 NaN 被核读取"。
3. **`XqaGeom` 对照**:同形状下 eager 走 `Single`(KV-split 关)vs `Capped` ⇒ 判"split 归约(scratch + semaphores)
   在**非图**发射下的语义"(图内 = 一次流内串行;图外 = 单发,理论同序,但归约缓冲/计数器协议需实证)。
4. 与探针交叉:同一 boot 内用 `family::xqa::launch_on_stream` 直发一次(探针路)对拍 ——
   探针路已知绿(`xqa_decode_probe`/`xqa_prefill_probe`),两者只差 `Exec::launch` 的封装与参数来源。

**⚠️ 教训(方法论)**:`OWL_PW_SELFCHECK` 首版用 `max|Δ| > 1e-3` 判差 —— **NaN 全盲**
(`NaN > 1e-3 = false`)⇒ 曾把"两臂都全 NaN"读成"一致(仅 16 输出差)"。已修(NaN 记 ∞);
修后同一条件报 **383/392 输出差** ✓ 与 `tok=-1` 自洽。

### 11.3 不变量补条(施工中新增,缺一条就是静默错值)

| # | 不变量 | 为什么 | 守门 |
|---|---|---|---|
| **I-7** | **段边界值必须 `keep`**:凡"本段算、后续段还要用"的节点(静态判:`A_i ∩ {last_need > i}`)一律跨段存活 | 段末 memo 只留 keep ⇒ 未 keep 的值在**后段会被重算**。纯算子重算只是白烧;但**带副作用的节点**(池写 `wr` / GDN conv·rec 递推)**重算 = 双写** ⇒ 状态被推进两遍(实测 `r*` fold 记录首偏) | `plan_piecewise` 的段边界分析(零算子知识)+ `OWL_PW_SELFCHECK` |
| **I-8** | **捕获窗内零回收** | 捕获期 `free` 被服务端白名单结构化拒(实测);且 Carved 块本就不可还 | `SegMode::Capturing`(禁池、禁回收、免段末清扫) |
| **I-9** | 空段/空尾**不建图** | `graph_end` 对空捕获窗硬错(哨兵③) | `segs: Vec<Option<GraphId>>` / `tail: Option<GraphId>` |
| **I-10** | 断点采集面必须与 `PlanCtx` 的断点表**同一张**(`break_sink()`) | 自建表 ⇒ 驱动件永远看到 0 断点(静默降级) | 单一拼写点(`sc.break_sink()`);`piecewise 无断点` 分支已删(零断点 = 退化单图) |

### 11.4 复现(repro;本地档在 `testdata/`,`gitignore` 内)

```bash
# 档:perf-ab-base.toml / perf-ab-pw.toml(greedy + dumb 哑草稿 + degrade_after=0;
#     后者比前者只多 [dispatch] verify_piecewise = true)
# ① 文本 A/B(判"verify 行 0 是否是真模型下一 token")
PW_ENV="OWL_PW_BREAKS=1" bash tools/pw-ab-greedy.sh testdata/args/perf-ab-pw.toml /tmp/pw.log /tmp/pw.json
#   k=0 ⇒ 与 base 逐字同;k=1 ⇒ "用户!!!!…"
# ② 取证(钉块/操作数/tok/hid 画像)
PW_ENV="OWL_PW_BREAKS=1 OWL_PW_DUMP_TOK=1" bash tools/pw-ab-greedy.sh ...
#   grep -E "pw-tok|pw-hid|pw-pin|pw-op" /tmp/pw.log
# ③ 图内一致门(必须 NaN 记 ∞;boot 期就能红)
PW_ENV="OWL_PW_BREAKS=1 OWL_PW_SELFCHECK=1" bash tools/pw-ab-greedy.sh ...
# ④ 恒等门(0.8B/27B;注意这两条门在本 worktree 有**既有环境性失败**:
#    submit 期就报 "prompt+生成 超预算",与本改动无关 ⇒ 用 ①②③)
OWL_GATE_PIECEWISE=1 OWL_GATE_XQA=1 OWL_E2E_ONLY=dflash2_identity cargo test ...
```

### 11.5 ✅ **定谳(2026-10-16 凌晨)**:根因 = 断点回放被 **memo 短路**(驱动件 bug),非发射面

**症状回顾**:`k=1` 首轮即崩(文本 `用户!!!!…`),`tok` 全 `-1`、`hid` 全 NaN、钉块全 NaN,
而断点操作数全有限;`k=0`(退化单图)逐字正确。

**追凶链(每步都是仪器给的事实,不是推理)**:
1. `pw-op`(断点操作数画像)⇒ 输入全有限(q max=1.0e1 / 池 / 表 / semaphores=0 / scratch / wr)
   ⇒ 排除"上游没算/地址错";
2. `sync_mask=16`(图回放后同步)⇒ 症状不变 ⇒ 排除"跨流无序";
3. **`OWL_PW_PIN_ZERO=1`(回放**前**把钉块清零)⇒ 令牌/`hid` 与基线臂**逐位一致**,
   而钉块回放后**仍是零** ⇒ 关键异常:**断点的写没落**;
4. **`pw-pin-now`(断点求值后立刻读钉块)⇒ 仍是零** ⇒ **eager 断点根本没发射**;
5. 读代码定谳:capture 期求值已把 `节点 id → 钉块` 写进 memo,回放期
   `eval_segment(&[&node], carry, …)` **一见 memo 命中就短路**(`memo` 取的是 carry 的),
   ⇒ 断点退化成"静态数据",尾段一路吃 capture 期的陈旧钉块(NaN)。

**修法**(1 行,`replay_piecewise`):求值前 `carry.memo.remove(&node.id())` —— 断点的语义
就是"每轮真跑",memo 里的那条正是 capture 期的残留。⚠️ 注意这条只打**静态断点**(无宿主标量);
带宿主标量的断点因每轮 id 换新而天然免疫(这也是为什么 FI 臂本来不会踩到)。

**修后判据(27B + fp8 + XQA + greedy + 哑草稿)**:
- `k=1` 与 `k=16`(全 16 断点)——`tok`/`hid`(absmax 26.17 / mean 0.0213,与 `k=0` 逐位同)/
  **生成文本与基线逐字一致** ✓;16 个钉块全区有限(max 3.2/6.36/3.89/4.66…)✓
- **正式 A/B**(`perf-ctxladder-27b{,-pw}.toml`,同二进制、sampling、degradation 开):

| ctx | 臂 | 轮时 ms/轮 | verify read ms | prefill t/s |
|---:|---|---:|---:|---:|
| 8192 | base | 52.81 | 36.70 | 1300.9 |
| 8192 | **pw** | **52.86 (+0.1%)** | **26.58** | 1294.1 |
| 32768 | base | 69.27 | 53.12 | 1141.9 |
| 32768 | **pw** | **69.87 (+0.9%)** | **43.19** | 1136.6 |

⇒ **判据通过**:端到端轮时 ±1%(设计文档 §7 P2 门),分段本身还把 verify 图**设备侧 read 时间**
降了 ~10ms(17 张小图 vs 一张 1484 节点大图的派发/交接税)。**⇒ 机制可用,可进 P3(换 FI 断点)。**

**方法论留档(两条会重复踩的)**:
- **"输入有限 ⇒ 输出 NaN"不一定指向核** —— 先证"产出到底是不是本轮写的"
  (`PIN_ZERO` 哨兵 + 求值后立刻读 = 一步分清"没发射"与"算错了");
- **自检门要能看 NaN**(首版 `max|Δ|>1e-3` 对 NaN 全盲),且**在会改状态的模型上**
  它会把"重跑漂移"混进差异(GDN conv/rec 原地递推:两臂各跑一遍 ⇒ 状态推进两遍 ⇒ tap 必偏)
  ⇒ 它是**分诊器**不是等价门;等价门要用 ③/④ 那种"单变量、单轮、逐位"的口径。

### 11.6 ✅ **P3 换装 FI 断点臂**(2026-10-16 凌晨;V1b 达成)

三件(全部落在 perf worktree;开关 `dispatch.verify_fi` + `verify_piecewise`,**缺省关**):
1. **FI 四件套走图输入槽**(`fi_q_cu/fi_indices/fi_indptr/fi_last_len`;init 与占位 ctx_total
   = depth1 自洽,nb_pages=ceil(T/page)=1)+ 槽→**U32 视图**(算子准入按声明 dtype 校验输入,
   糊 f32 叶会被拒:实测"q_cu 期望 U32 实得 F32");
2. **层侧 FI 断点登记**:`fi_core_node()` 抽成单一拼写点(prefill 主臂与断点共用),
   `paged_prefill_fi_output` 里按 **宿主标量槽** 登记(`FI_CTX_TOTAL_ARG = 1`,契约槽序 =
   `client::flashinfer::PARSE_SLOTS`)。⚠️ 必须**只有**这一条登记:同时留旧的静态登记
   ⇒ 33 段/32 断点(每个核跑两遍;实测 32k 轮时 97ms ↔ 16 断点 48ms);
3. **引擎逐轮写值**(`exec/spec.rs`):`cell.store(pos + rows)` + 四件套(页链/indptr/last_len
   全部 u32 位型过线);`GraphFleet::verify_fi_ctx()` 是唯一取用处(与 `xqa_pt_v` 同纪律:
   在 `self.graphs` 可变借用前取好)。

**真机判据**(27B + fp8 + greedy/dumb;`perf-ctxladder-27b{-fi}.toml`;同一 prompt/同一二进制):

| ctx | 臂 | **轮时 ms/轮** | verify read | prefill t/s |
|---:|---|---:|---:|---:|
| 8192 | base(XQA 图内) | 52.81 | 36.70 | 1300.9 |
| 8192 | **FI 断点** | **48.64 (−7.9%)** | **22.38** | 1304.7 |
| 32768 | base | 69.27 | 53.12 | 1142.0 |
| 32768 | **FI 断点** | **48.41 (−30.1%)** | **21.65** | 1145.0 |

⇒ ① **设计文档 §6 的终局判据(32k 轮时 ≤52ms,换臂后 ≤50)达标**(48.4);
② **轮时对 ctx 变平**(8k 48.6 / 32k 48.4)—— 长 ctx 衰减的根因(verify 图里的 ctx 依赖 XQA)消失;
③ prefill 不动(±0.3%);④ 功能与基线逐位一致(tok/hid/文本)。

**转正待办(需用户拍板)**:`spec-dflash` 档加 `verify_piecewise = true` + `verify_fi = true`
(+ `verify_xqa/v2` 关)即进生产;decode(T=1)仍是 XQA(V1c 可同法换,断点在 session 图)。

### 11.7 转正(2026-10-16):生产档换装 FI 断点臂 + 红线门接线

**① 生产档**
- `spec-dflash` flavor(`shared/config.rs`,**committed**):`verify_xqa = false`、
  `verify_v2 = false`、`verify_piecewise = true`、`verify_fi = true`(注释里写了换代理由与换代数据);
  `decode_xqa` 保持 true —— **decode(T=1)仍是 XQA**(V1c 可同法换:session 图 + 断点挂 XQA 核)。
- ⚠️ **args 档会覆盖 flavor**:`testdata/args/spec-27b-prod.toml` 自己也带
  `[dispatch] verify_xqa = true`(深度合并 ⇒ 只覆盖列出的键)⇒ 必须**同时**把 args 档那几行改掉,
  否则 flavor 的换代被 args 覆盖回 XQA(实测第一轮就踩到:日志/文本看不出,要看轮时签名)。
- 生产档实测(flavor+args 双改,8192):**轮时 52.8 → 46.7ms(−11.5%)、verify read 36.7 → 20.6ms**、
  prefill 1305(不动)、decode 71.1 t/s。(生产几何 max_seq=16384 ⇒ 32k 点测不了;
  32k 的 −30% 见 §11.6 的长几何档。)

**② 红线门(接线完成,当前因 MTP 路径的环境性问题为红)**
- `gpu_spec_identity_gate_27b` 的 `OWL_GATE_PIECEWISE=1` 现在**整臂**打开
  (piecewise + verify_fi + verify_xqa/v2 关)⇒ 这条门 = "断点臂不破文本"红线
  (baseline `spec_depth=0` vs spec 的贪心 token 流逐位比)。
- 该门在本 worktree 有两处**既有**环境性红,已修一处:`_MIN_GEN_TOKENS = 1024`
  要求 `max_seq − prompt ≥ 1024`(门用 256 ⇒ 恒拒发"超预算")⇒ 抬到 2048 后 baseline 轮通过;
  另一处在 **MTP propose 路径**(`exec/draft.rs:852` panic,spec 轮),与断点臂无关
  (piecewise 关闭时同样复现)⇒ **在这条 MTP 路径修好前,实际红线用**
  `tools/pw-ab-greedy.sh` + `testdata/args/perf-ab-fi.toml`(greedy + 哑草稿,文本逐字对比,
  实测 k=16 与 base 全等)。
