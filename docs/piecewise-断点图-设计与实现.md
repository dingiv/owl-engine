# Piecewise(可断)捕获图 —— 设计与实现方案(2026-10-15;perf 会话立项)

> 🔴 **已退役(2026-10-18,W5)** —— 本卷保留为**历史设计与实录**,机构已整体删除:
> 本卷的代码面(`plan_piecewise` / `replay_piecewise` / `BreakPoint`·`PiecewiseRt` /
> `pw_carry` 接力棒 / `BreakDecl`+`EvalCarry`+`SegMode`+`eval_segment` / 断点登记面
> `attn_break`·`AttnFiBreak` / 旋钮 `verify_piecewise`·`break_at_out`·`fi_bucket`·
> `pw_dump_tok`)已从工作树删除;**配了旧键的配置会响亮拒启**(迁移门,见
> `config.rs::reject_retired_keys`)。
>
> **为什么退役**:W3/W4 之后 FI 核留捕获图内、调度表由窗外 `fi_plan_sync` 刷
> ⇒ 断点面零消费者;而它正是"接力棒每步只增 ⇒ 段末 `memo.retain` 每步 16×O(N)
> ⇒ 端到端 O(步数²)"那条慢化病的宿主(设计卷 §3.3)。现行形态见
> `docs/arch/图档-FI-单图-设计与施工.md`(§6 W4 转正 / §7 终版组合)。
> 需要分段求值时从 git 历史取回(本卷即那时的设计依据)。

## §0 现状与用法(先读这一页;实录从 §1 起)

| 项 | 值 |
|---|---|
| **状态** | 🔴 **已退役**(2026-10-18 W5;见卷头)—— 历史转正记录:prefill / verify / decode 三条路 FI+split 的**分段承载**已被单图 + 窗外刷表取代 |
| **开关**(flavor `spec-dflash`,已置位) | `verify_piecewise = true`、`verify_fi = true`、`verify_xqa = false`、`verify_v2 = false`、`decode_fi = true`、`decode_xqa = false`(外加既有 `kv_fp8 = true`) |
| **回退** | 把 `verify_piecewise/verify_fi/decode_fi` 关掉 ⇒ 回 XQA/v2(**不改代码**);⚠️ **args 档覆盖 flavor**(深合并),`testdata/args/spec-27b-prod.toml` 等本地档必须同步改,否则静默跑回旧臂 |
| **拓扑** | 27B:verify = 17 段/16 断点(T=8);session = 17 段/16 断点(T=1 每步) |
| **收益**(27B+fp8,greedy) | verify 32k 轮时 69.3 → **48.4ms(−30%)**;decode 裸步 32k 47.8 → **29.0ms(−39%)**;两路**对 ctx 平坦**;文本与旧臂**逐位一致**;prefill ±0.3% |
| **代价/未决** | 17+17 条 slab:实测 **485MiB 图族总账**(session 47 / verify 313,见 **§12 显存账**);**段 slab 合并只值 −71MiB**(单 slab 口径 26/263MiB);⚠️ 合入 main 后**桶族(geobuckets)在 eval 档 5/5 全灭** = 起跑线太高(本设计边际 +580MiB),生产档不受影响(flavor 缺省 `prefill_bucket=off`)|
| **不变量** | **I-1..I-6**(§5)+ **I-7 段边界值必须 keep**、**I-8 捕获窗内零回收**、**I-9 空段/空尾不建图**、**I-10 断点采集面与 plan 的断点表同源**(§11.3) |
| **机制一句话** | 树被**断点**(核节点)切成段,各段单独捕获;断点由驱动件在**段间窗外 eager 重放**并写进**钉块**(plan 持有、两期同址);核里"每轮变的宿主标量"走**宿主标量槽**(`BreakDecl::host_scalar`,下标 = kernel 契约槽序),每轮变的设备值走**图输入槽** |
| **取证仪器**(env diag,§11.4/§11.12) | **boot 期 `现役臂: prefill=… \| verify=… \| decode=… \| spec=… \| kv_fp8=…` 一行**(臂身份自证);`session/verify 分段捕获(段=N 断点=M)`;`OWL_PW_SELFCHECK`(回放≡eager,NaN 记 ∞);`OWL_PW_BREAKS=k`;`OWL_PW_PIN_ZERO` + `[pw-pin-now]`;`OWL_KV_DUMP` + `tools/kv_footprint.py`;`[fi-plan]`(OWL_FI_DEBUG);`tools/pw-ab-greedy.sh`、`tools/pw-probe.sh` |
| **门** | engine lib 33/33、models lib 129/129、`gpu_piecewise_segments`;`OWL_GATE_PIECEWISE=1` ⇒ 27B 恒等门整臂(断点臂不破文本红线) |

**三条最贵的教训**(全文最有复用价值的部分):
1. **追"连贯但错"的输出,先证"这条路径到底有没有跑"** —— 段/断点计数、plan 计数、
   KV 字节足迹,三个几行的仪器各排掉一个方向(核错/布局错/表错);本次真因是
   `plan_session` **少交断点采集面** ⇒ FI 核被烘进图 + plan 用占位 `ctx_total=1` 烘死。
2. **"看起来像 A 臂"的数字可能是 B 臂**(档被拼坏,把 v2 当 FI 报过一栏)⇒ 报数前先验臂身份。
3. **对 NaN 全盲的门等于没有门**(首版 `max|Δ|>1e-3` 把"两臂都 NaN"读成"一致")。

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

### 11.8 ⚠️ V1c(decode T=1 换 FI)接线完成,**数值不一致 ⇒ 未转正**(2026-10-16)

**落的件**(开关 `dispatch.decode_fi`,**缺省关**):
- 模型:decode 分派新增 FI 断点分支(`paged_decode_v2_k0` 的 XQA 检查之前),
  与 verify 臂同构(宿主标量槽 + `fi_core_node`);⚠️ decode 路的 `wr` 是 **2 维**哑输出,
  FI 声明 rank=1 ⇒ 必须取 `narrow_strided(wr,1,1,0,1,[1])` 一元素视图(准入会拒:
  "wr 期望 rank = 1 实得 2")。
- 引擎:`plan_session` 在 `decode_fi` 开时走 `plan_piecewise`(session 常驻图 → 段驱动件);
  四件套槽(`fi_q_cu/fi_indices/fi_indptr/fi_last_len`,位型过线)+ `decode_fi_ctx()`
  宿主标量槽;`exec/phases.rs` 每步写 `ctx_total = pos+1` 与四件套(判据 = 槽在场,
  与 `xqa_seq` 同款纪律)。踩坑:`expect("cell 必存在")` 会在关闭态 panic(已改为 Option 分流)。

**为什么未转正**(硬判据):
- 裸 decode 专档(`depth=0` + greedy,同一 prompt):XQA 臂 vs FI 臂 **文本不一致**——
  两者都连贯(base "…介绍。让我来数一下。…" vs FI "…我用中文回答,因为…"),
  且**从第一个 decode 步就分叉**(前缀 "用户要求" 相同 = prefill 产物相同)
  ⇒ 不是 NaN/崩坏,而是 **FI 读到的 KV/几何与 XQA 不等价**。
- ⇒ **按"先证不回归"纪律停手**:decode 臂是纯聊天的主路径,数值不一致 = 不可发布;生产档
  保持 `decode_xqa = true`(V1b 的 verify 换代不受影响)。

**下一步嫌疑表(按可能性)**:
1. **最新一行的 fp8 写者不同**:decode 的当前行由**融合插池** `qknorm_rope_kv_insert_f16_fp8kv_knhd`
   写,而 FI 的同形状核级门(`fi_prefill_probe` 的 T=1 档)与 verify 臂验证时,池里的行都是
   **K0 家族**写的 ⇒ 先查"同一行的两类写者是否逐字节同布局"(fp8 packing/`x` 向量宽/scale 约定);
2. **解码步的页链/`last_len`**:跨页时 `nb_used/last_len` 的口径(现按 `ceil((pos+1)/page)` 推);
3. **`q`/scale 口径**:`fi_scale_bits` vs `xqa_q_scale_bits`(verify 形状已互等,但 decode 的
   `sm_scale` 走的是另一条实例化,需逐位对拍);
4. **最快的判据**:同 boot 内对同一 (q, 池, 表) 跑 XQA-decode 与 FI-decode 两个节点,
   d2h 比 max|Δ|(把"臂"从"文本"里剥出来;`pw-pin` 那套仪器可直接复用)。

### 11.9 🔎 V1c 病灶二分:**0.8B(f16)两臂逐字一致 ⇒ 病灶在 fp8 专用段**(2026-10-16)

**为什么要换小模型**:0.8B 是 **f16 池**、且 XQA 不可用(hq/hkv≠6)⇒ 参考臂天然是 **v2 decode**,
正好把"fp8 行写者/读法"这一大类嫌疑**裁掉一层**;且 boot ~10s(27B 是 ~90s)⇒ 追凶迭代快一个量级。
落点:`testdata/args/dec08-{base,fi}.toml`(kind="0.8b", `dir=models/Qwen/Qwen3.5-0.8B`,
greedy, `depth=0`,`max_seq=2048`(⚠️ `_MIN_GEN_TOKENS=1024` ⇒ max_seq 太小会恒拒发),
`pool_tokens=4096`);run:`FLAVOR=awq27b CVD=<空闲 3080 uuid> bash tools/pw-ab-greedy.sh …`
(harness 的 flavor/CVD 已参数化;⚠️ GPU1/2 是主会话实例,别动,用 GPU0/3 的 3080)。

**结果**:
```
0.8B 参考臂(v2): 长城是古代中国防御工事，历经千年风雨，见证历史沧桑。
0.8B FI 臂     : 长城是古代中国防御工事，历经千年风雨，见证历史沧桑。   ⇒ 逐字一致 ✓
```
⇒ **piecewise(session 图)+ FI 四件套槽 + `ctx_total` 宿主标量槽 + 段间 eager 重放 全链正确**;
27B 的不一致只能出在**fp8 专用段**:

**下一步判据(最锋利的一刀)**:27B + fp8 下把参考臂换成 **v2 decode**(`decode_xqa = false`,
不进 XQA 臂)⇒
- 若 **FI ≡ v2**(27B)→ 病灶在 **XQA 与 FI 对同一 fp8 行的读法差异**
  (嫌疑:融合插池 `qknorm_rope_kv_insert_f16_fp8kv_knhd` 写的行 vs K0 家族写的行,
  在 FI 的 fp8 读侧假设不同 —— `x` 向量宽/scale/packing);
- 若 **FI ≢ v2** → 病灶在 FI 的 fp8 读核本身在 **decode 形状(T=1)** 下的前提
  (核级门只覆盖过 T=1 的 **f16/探针表**,fp8×T=1 的核级对拍要做一档)。

### 11.10 🔎 V1c 定谳刀:27B+fp8 **FI ≢ v2(同池同写者)** ⇒ 病灶在 FI 自己的 fp8 读法(2026-10-16)

**实验**(`testdata/args/dec27-{base,fi}.toml`:27B + fp8 + `depth=0` + **`decode_xqa=false`** ⇒
参考臂 = **v2 decode**,两臂同池同"融合插池"写者;greedy;3080 上跑):

| 臂 | 文本 |
|---|---|
| v2(参考) | "用户要求用五十字介绍长城。我需要写一段恰好50个中文字符的介绍。让我来数一下。…" |
| FI       | "用户要求我用中文回答,因为用户用中文提问了。但等等,…" |

⇒ **FI ≢ v2** 而 **v2 ≡ XQA**(v2 这条文本与先前 XQA 臂逐字同)⇒ 病灶**不是**"XQA 与插池写者的约定差异",
而是 **FI 自己对 fp8 池的读法/前提**。

**三条事实的交集把范围夹到"fp8 × T=1"**:
| 场景 | 池/形状 | 结果 |
|---|---|---|
| verify(T=8)FI vs XQA | **fp8** × T=8 | ≡ ✓(§11.6) |
| 0.8B 裸 decode FI vs v2 | **f16** × T=1 | ≡ ✓(§11.9,harness 跑) |
| 27B 裸 decode FI vs v2 | **fp8** × T=1 | **≢ ✗**(本表) |
⇒ 唯一的未覆盖组合 = **fp8 × T=1**。而 V1 的核级探针 **只覆盖 fp8**(f16 臂未验)且 T=1 档是探针**自建的表** ⇒
嫌疑落在"**引擎为 decode 步造的四件套口径**"与"探针那套"之间的差异(多页/末页 `last_len`/`q_cu`/`indptr`),
或 FI 的 fp8 读核在 **T=1 多页**下的前提。

**下一刀(核级,不碰引擎,最快)**:把 `crates/kernels/tests/fi_prefill_probe.rs` 补一档
**"引擎形状的 T=1 多页 fp8"**(fp8 池、kv_len > page、末页半满、`q_cu=[0,1]`/`indptr=[0,nb_used]`/
`last_len=kv_len-(nb_used-1)·page`),对 host oracle 判 —— 若探针绿 ⇒ 病灶在引擎表(逐字段对拍引擎写法);
若探针红 ⇒ 核在 fp8×T=1 多页下的前提有洞(核侧修)。

⚠️ 方法论留档:本轮后段两次"临时 curl + 短/长 prompt 对照"的结果**不可信**——不同模型/不同 prompt
返回了**逐字相同**的文本(客户端/排队伪影),故**未采信**、也未据此改动结论。追凶只认:①harness 的两臂同 prompt 对照
②核级 host oracle。后者(探针)是唯一能把"引擎表"与"核读法"干净分开的面。

### 11.11 🔎 探针判"核无罪" + 两个 fp8 写者的块内元素序对不上(2026-10-16)

**① 探针补"末页半满"档 → 全绿**(新增用例已入 `crates/kernels/tests/fi_prefill_probe.rs`):
```
(1,53) 2页 last_len=21   max|FI−host| 2.46e-4   零值 0/6144
(1,3073) 97页 last_len=1 4.41e-4   0/6144
(1,3080) 97页 last_len=8 5.20e-4   0/6144
(1,1057) 34页 last_len=1 5.23e-4   0/6144
(8,3080) verify+半满     5.32e-4   0/49152
```
⇒ **核在"fp8 × T=1 × 多页 + 末页半满"下是对的**(此前后所有 host 校验档的 ctx 都是 PAGE 整数倍
⇒ 半满从没测过)。⇒ 病灶在**引擎侧**(表/写者),不在核。

**② 读两个 fp8 写者的写址算式(块内元素序)**:
| 写者 | 算式(块内) | 元素序 |
|---|---|---|
| `owl_reshape_and_cache_fp8kv`(**无后缀**) | `head*(hd/x)*page*x + xidx*page*x + off*x + xoff` | **x-packed / page-minor**:`[blk][hkv][hd/x][page][x]` |
| `owl_reshape_and_cache_fp8kv_**knhd**` | `off*hkv*hd + head*hd + d` | **page-major**:`[blk][page][hkv][hd]` |
| `owl_qknorm_rope_kv_insert_f16_fp8kv_**knhd**` | `((blk*hkv+kh)*(hd/8)+d/8)*page*8 + off*8 + d%8` | **x-packed(与无后缀同式,x=8)** |
而 `EnvProvider::k0_write_op()` 对 fp8 返回 **`K0WriteFp8Knhd`**(= page-major 那个)⇒
**prefill/verify 的行 = page-major,decode 的当前行 = x-packed** ⇒ 若两者真不同序,则
FI 读 decode 行错(本次症状 ✓:FI 只与 K0 写的行对过账,verify 臂全 K0 ✓ 0.8B f16 又是同一族 ✓)。

⚠️ **但我不能就此定谳**:若两者真不同序,则 decode 臂(XQA/v2)读 *prefill* 写的 prompt 行也该错
(实测它们对 ⇒ 说明要么两者同序,要么读侧另有分派)⇒ **必须做字节级 A/B 才能定案**。

**③ 下一刀(唯一能干净定案的)**:核级 **写者 A/B** —— 同一行 k/v 分别过
`K0WriteFp8Knhd` 与 `owl_qknorm_rope_kv_insert_f16_fp8kv_knhd`,d2h 比**同一 slot 的字节**:
- 字节同 ⇒ 布局一致,病灶回到"引擎为 decode 造的**表/值**"(逐字段对拍 `phases.rs` 的写法:
  `indices` 只写前 `nb_used` 项、其余补 0;`indptr=[0,nb_used]`;`last_len=ctx−(nb_used−1)·page`;
  `q_cu=[0,1]`;`ctx_total=pos+1` —— 与探针表逐字段比);
- 字节不同 ⇒ **同池两套序**(潜在老账):FI 必须按"它读的那一族"的序取,或统一写者;
  判据 = "FI 读 insert 写的行 vs 读 K0 写的行"两侧对 host oracle。

### 11.12 ✅ 真机字节差分定案:两个 fp8 写者**同序(page-major)**——"写者不一致"出局(2026-10-16)

**仪器**(用户点醒的做法:**不看日志看设备字节**,且不需要参考值):
`OWL_KV_DUMP=<decode_pos>` ⇒ 在 `exec/phases.rs` 的 decode 步**写前/写后各 d2h 一页**
(`copy_block_at` 抠出该页 → `dtoh` → `/tmp/kv-{before,after}.bin` + `kv-meta.txt`);
分析器 `tools/kv_footprint.py`:**差分足迹**即结构性事实 ——
- page-major ⇒ 该行 `hkv·hd` 字节是**一整段连续**(页内 `off·hkv·hd` 起);
- x-packed  ⇒ 是 128 个 **8B 块 strided 256B**。

**实测(27B + fp8,slot=54 ⇒ blk=1/off=22)**:
```
改动字节 1024(=hkv·hd)  改动区间 [22528,23551] = 22*1024 一整段
假设 A(page-major) 命中;假设 B(x-packed) 覆盖 0.03 ⇒ 不符
```
⇒ **decode 的写者(融合插池)= page-major,与 `K0WriteFp8Knhd` 同序**;FI 的 `QKVLayout::kNHD`
正是 page-major ⇒ **读法与池的写法一致** ⇒ §11.11 的"两套序"**出局**(当时从公式推的"无后缀=x-packed"
是误读:调用方按**打包单位**传 `num_heads/head_size`,公式实际坍缩成 page-major)。

**于是 V1c 剩下的差异面收窄为**(核 ✓、池写 ✓、表与 verify 同式 ✓ 都已排):
1. **`q` 操作数的形态**:decode 路的 `q` 来自 `qknorm_rope_kv_insert` 家族(该家族与 XQA 的 fp8
   q_scale 配套),而 FI 的 `Param::t_in("q", F16, 2)` 期望 f16 规整 `[T, hq·hd]`
   —— 需核对该 op 在 **fp8 池**下返回的 q 是否仍为 f16/该形状(准入只校 dtype/rank,不校"语义");
2. **`ctx_total` 宿主槽与 plan 的时序**:plan 每步 miss ⇒ 重算 + H2D(在窗外 ✓),需证实
   "回放期真正读到的是本轮值"(仪器:`OWL_FI_DEBUG=1` 打印 plan 的 split/num_chunks,逐步对照);
3. **`gate`/`wr` 依赖边**在 FI 节点里的槽序是否与 decode 路一致(两路 arm 的实参顺序不同)。

### 11.13 ✅ V1c 结案(2026-10-16):真因 = session 图**没交断点采集面**(一行),FI 核被烘进图用占位 plan

**取证**(两件仪器:FI plan 的 ctx_total 打印 + session 分段日志):
```
session 分段捕获(段=1 断点=0)         ← decode_fi 开着却是 0 断点
[fi-plan] ctx_total=1 nb_pages=1 T=1   ← 占位值(建树期的 ctx_total=tokens=1)
[fi-plan] ctx_total=53 nb_pages=2 T=53 ← 全程只有 prefill 的两次
```
⇒ 模型侧 FI 分支照常返回 FI 节点,但 `ctx.attn_break` 为 None ⇒ **登记被跳过** ⇒ 核留在
session 图里,plan 用**占位 ctx_total=1** 烘死 ⇒ 每步拿"ctx=1"的调度表读 53+ 的池
⇒ 输出连贯但错(先前"FI ≢ v2"的症状全由此解释,**不是**核/布局/表的问题)。

**修复**:`plan_session` 闭包里与 `plan_verify` 同律 `ctx.attn_break = Some(sc.break_sink())`
⇒ `session 分段捕获(段=17 断点=16)`,FI 臂文本 ≡ v2/XQA **逐字一致**。

**收益(裸 decode 阶梯,27B+fp8,greedy)**:
| ctx | XQA(生产) | **FI 断点** | FI+桶 2048 |
|---:|---:|---:|---:|
| 8192 | 33.13 ms/tok | **27.83** | 28.37 |
| 32768 | 47.78 ms/tok | **29.00(−39%)** | 29.66 |
⇒ 对 ctx **平坦**(27.8/29.0),fi-plan 每步重算(609 次/两点)其实便宜 ⇒ **分桶无益**(+1%),
不做桶(设计上少一层)。**转正**:flavor `spec-dflash` + 生产 args 档 `decode_xqa=false /
decode_fi=true`(⚠️ args 覆盖 flavor,两处都要改;同一坑第二次踩)。

**方法论**(本轮的账):追踪"连贯但错"的输出时,**先证"这条路径到底有没有跑"**(段/断点计数、
plan 计数、字节差分),再怀疑数值 —— 本轮三个仪器(分段计数/plan 打印/KV 足迹)各只花几行,
却一次排掉"核错/布局错/表错"三个方向;**而"看起来像 A 臂"的数字可能是 B 臂**(档被我拼坏过一次,
把 v2 当成了 FI)⇒ 报数前先验"臂真的在跑"。

### 11.14 转正落码后的验证状态(2026-10-16 收尾)

- **落码**:flavor `spec-dflash`(`verify_xqa/v2=false`,`verify_piecewise/fi=true`,`decode_xqa=false`,
  `decode_fi=true`)+ 生产 args 档同步(**args 覆盖 flavor,两处都要改** —— 本轮踩了三次:两次
  `verify_*`、一次 `decode_*`;凡"改 flavor 就生效"的直觉都是错的)。
- **已验证**:生产几何下 boot 成功,且**两条 plan 都是分段态**:
  `session 分段捕获(段=17 断点=16)` + `verify 分段捕获(T=8;段=17 断点=16)` ✓;
  裸 decode 臂文本 ≡ v2/XQA 逐位、timing −16%/−39%(§11.13,同 flavor 同池)。
- ⚠️ **未验证**:生产档**跑一个带 spec 的请求**(`prod-fi-check.toml`)在 20G 的 3080 上
  **OOM**(`alloc 1.7MiB` 失败)——不是接线问题:两条 piecewise plan 各 17 段 ⇒ 17+17 条
  slab(每段按自己的 dry 计量开)+ fold/propose 图族,总占用在 20G 卡上贴顶。
  ⇒ 待 3090 Ti 空出(GPU1/2 现被主会话两实例占用,不动)时补一次请求级验证;
  并记一笔**优化项**:段 slab 合并("一条 slab N 段",设计 §10.2 的 A3 完整版)由主会话
  S1.5 arena 落地后启用 ⇒ 显存占用可回到单图量级。

### 11.15 ✅ A 组验证收官(2026-10-16;**24G 卡到位后全部跑完**)

| 项 | 结果 |
|---|---|
| **A① 生产档请求级(带 spec,depth=7,fp8,24G 3090 Ti)** | ✅ boot 两条 plan 分段态;**greedy 请求文本与基线逐字一致**;轮时 **16.3ms/轮**(短 ctx) |
| **A② 长 ctx decode 文本级 A/B** | ✅ 参考臂 `decode=XQA` vs 实验臂 `decode=FI断点(17/16)`,**prompt=9298 tok(291 页)× 32 tok 逐字一致** |
| **A③ soak(生产档,sampling)** | ✅ 225 tok / 4.77s、**16.0ms/轮**、AL≈4.2、**零 error/panic/Xid 迹象** |
| **A④ 门** | ✅ engine lib 33/33、models lib 129/129、`gpu_piecewise_segments` 真机绿 |
| ⚠️ 仍未做 | 27B 恒等门整臂(`OWL_GATE_PIECEWISE=1`)因**既有 MTP 路径红**(`draft.rs:852` index OOB,**piecewise 关亦复现**)而红 ⇒ 实际红线仍用 `tools/pw-ab-greedy.sh`(已两次证明:1 条/16 条断点 + 长 ctx) |

**⚠️ 本轮又踩一次"args 覆盖 flavor",但这次是反过来**:**没写的键继承 flavor** ⇒ 我写
`decode_xqa = true` 想造 XQA 参考臂,却漏了 `decode_fi = false` ⇒ 参考臂实际跑 FI(A② 第一轮
"一致"是假绿灯)。**是 boot 期那行『现役臂』把它当场抓出来的**(仪器一上线就还本)⇒ 参考臂一律
**显式写全六键**(`decode_fi=false` 之类)。

**A 组结论**:V1 三段换装(来源核/收益/正确性)**全部自证完毕**;生产档在 24G 卡上可跑、文本正确、
轮时正确、无 Xid。剩下只有 C 组债务(段 slab 合并等 S1.5、pin 守卫、diag 登记)与 D 组通报。

---

## 12. 显存账(2026-10-16;合并入 main 后"桶族全灭"的三段 A/B 归因)

**触发**:`90a00a7`(本设计合入 main)之后,主会话的 **prefill 桶族(M2b)在 27B 档
5/5 全灭**(t_pad=512 起就 `CUDA_ERROR_OUT_OF_MEMORY`,桶族摘要阶梯为空),而合并前
`6f128ad` 同卡同档 **5/5 Captured**。用户令"**先归因**再来动显存"。

### 12.1 口径(可复现的 A/B 三重奏)

同一张卡(3090 Ti 24G,cuda:0)、同一份档(`m2b-bucket-27b.toml` 派生:pool 33280 /
max_seq 32768 / geobuckets / prefix_cache=true / gdn_slots=8)、同一 env
(`OWL_LOAD_STAGING_MB=1024`;缺省 128MB 会让 AWQ repack 键超槽 ⇒ 装配失败),
**只换二进制**:

| 臂 | 二进制 | 来处 | 桶族结果 |
|---|---|---|---|
| **B** | `/tmp/owl-server-m2b-b6` | 对方分支(§4.7e 那批) | ✅ 5/5 捕获(t512 = 91MiB) |
| **C** | `6f128ad` 现场重编 | main 合并前(含 M2b + M1) | ✅ 5/5 捕获(t512 = 91MiB) |
| **A/E** | `perf-squashed` 现场重编 | main 合并后(本设计) | ❌ 5/5 失败(t512 warmup 死) |

⇒ **回归确实由本设计的合并引入**(A ≠ C);后来的两笔排除实验:
**① `FI_SPLIT_KV=0`(关 split)⇒ 一样死**(三处数字逐字同)⇒ **不是 FI split**;
**② `verify_piecewise=false`+`decode_xqa=true`(关本设计两臂)⇒ 一样死** ⇒
**不是段驱动件本身,而是它留下的显存**;
**③ `90a00a7`(E1 三流骨架 revert 前的合并)⇒ 一样死** ⇒ 不是 revert。
**④ 档几何缩小(pool 33280→16896,+0.88GB 余量)⇒ 一样死**,连失败块的序号都一样
⇒ **不是"余量不够"这么简单**。

### 12.2 账(新增三面仪器;见 §12.4)

**(a) 段 slab 账(本设计自己的边账)** —— 27B prod 几何,每段一条 slab(现状):

| plan | 段数 | 段定尺合计(**现状**) | 单 slab 口径(跨段累计探针) | 差 |
|---|---:|---:|---:|---:|
| session(decode T=1) | 17 | **47 MiB** | **26 MiB** | −21 |
| verify(T=8) | 17 | **313 MiB** | **263 MiB** | −50 |
| fold 族(m=0..6) | 7 | 102 MiB | — | — |
| dflash propose | 1 | 21 MiB | — | — |
| **合计(总账行)** | 42 图 | **485 MiB** | — | **段合并只值 −71 MiB** |

⇒ **"段 slab 合并"的天花板是 71MiB**(此前无账,只能猜);而合并前的同档 slab
总量 ≈ 248MiB ⇒ **本设计在 slab 上的边际开销 ≈ +237MiB**(verify 313 vs 单图 133、
session 47 vs 单图 13)。

**(b) 失败现场三账**(`vram-ledger` WARN,分配失败时打;这是本轮的关键面):

```
alloc 失败(17.0MiB) ‖ 池 used=23683MiB(高水同)/reserved=23808MiB(高水同)
                     ‖ 驱动 free=12MiB/total=24164MiB ‖ 引擎 live=23524MiB
```
⇒ **设备真的占满了**:池没有抱死页(used≈reserved),引擎自己的账(23524)与池账
对得上 ⇒ **不是账目错、不是池缓存**,而是**引擎实打实持有 23GB 块**。

**(c) 逐 plan 引擎账(台阶)** —— `slab 定尺`/`段 slab 账`/`总账` 行尾的 `引擎 live`:

| 相位 | live | Δ |
|---|---:|---:|
| 治理行(boot 记账) | 17.5G | — |
| session 段账后 | 21217 MiB | **+3297** |
| verify 段账后 | 21664 MiB | +447 |
| fold 族(m=0..6) | 21822 MiB | +158 |
| dflash | 21860 MiB | +38 |

⇒ **那 +3.3GB 与 piecewise 无关**(session 图的探针账只有 **26.9MiB/1608 块**,
分配总量 0.0GB);它是**主侧既有的懒分配**(prefix_cache 块 / 快照 / draft 池 ——
`fixed_after=1.70G` 就是它的预算影子,`prefix_cache=false`+`gdn_slots=1` 的**生产档
只有 0.68G**)。**交叉证据**:桶关档的 boot 稳态 —— 合并后 22457 vs 合并前 21877
⇒ **本设计的总边际只有 +580MiB**,若 session 相位真是 +3.3GB,这个差会是 GB 级。

**(d) 桶 t512 warmup 的探针画像**(合并后):峰值 **1619MiB / 270 块**(分配总量
1.5GB,构成与总量逐名同比例 = 一字节没回收的形态),而合并前同图 **91MiB / 9 块**;
`活性回收账`:新块 265 / 真回收 261 —— 但**回收集中在段末清台**(中途活性回收在
"快满"的卡上触发不了),故存活峰值≈总量。**⇒ 结论:桶族的死因是"起跑线太高"
(设备仅剩 ~0.3GB,而它要 ~0.3GB+ 的 warmup 与 174MiB 的全族 slab),不是它自己漏。**

### 12.3 结论与杠杆(按实测收益排序)

1. **生产档不受影响**(实测):`spec-dflash` flavor 缺省 `prefill_bucket = off`
   ⇒ 桶族是 **opt-in**;生产档 boot 稳态 22.4G 可跑(本设计三段 FI 全开)。
   ⇒ 这是"**feature 在 eval 档被挤出**",不是产线事故。
2. **段 slab 合并 = −71MiB**(已量化;`docs` §10.2 A3 完整版;一条 slab 走 N 段,
   容量 = 12.2(a) 的"单 slab 口径"列)。
3. **不段化 = −313MiB**(verify 回 XQA 单图),代价 = 放弃 V1 的 verify 收益
   (32k 轮时 +20ms/轮)+ decode 的 FI 断点臂 ⇒ **纯性能/显存取舍**,归用户裁决。
4. **调 m2b 档的懒分配**(最大头):`prefix_cache=true`(块驻留=池半)+ `snap_max=1`
   + `gdn_slots=8` + `pool_tokens=33280` ⇒ `fixed_after` 1.70G(生产档 0.68G)。
   桶族测试档建议对齐生产几何(16896/1 格/无前缀缓存)再谈"桶族吃得下"。
5. **桶族自己收窄**(主侧):实测 t512 桶 91MiB、全族 174MiB;桶族摘要已是"按桶
   优雅回退" ⇒ 只要余量 ≥0.3GB 就能自愈(第 4 条执行后即成立)。

### 12.4 留下的仪器(下一次"显存不够"直接读)

| 仪器 | 位置 | 读法 |
|---|---|---|
| **段 slab 账** | `[graph-plan] piecewise 段 slab 账:…` | 段定尺合计 vs **单 slab 口径** |
| **图显存总账** | `[boot] slab 总账:session … = N MiB ‖ 引擎账 vram::live=…` | 全族 slab + 引擎账 |
| **逐 plan 台阶** | 每行 `定尺/段账` 尾 `引擎 live=` | 相位增长台阶(本轮抓出 +3.3GB) |
| **活性回收账** | `[graph-plan] 活性回收账:新块/真回收(拒:pending/users/alias/非竞技场/keep/未登记)` | "一字节没回收"时指认出口 |
| **失败现场池账** | `[vram-ledger] alloc 失败… ‖ 池 used/reserved(高水) ‖ 驱动 free ‖ 引擎 live` | 分辨"真占满"vs"池抱页" |

**方法论三则**(本轮最贵的教训):
1. **"看着有 free"不等于分得出来** —— 三账(pool used/reserved、驱动 free、引擎
   live)一起打才能定案;单看 `mem_get_info` 会误判。
2. **"回归"要先证它真的来自回归方** —— 本例一开始怀疑 FI split/E1 revert,四处
   排除(§12.1 ①②③④)才收到"+580MiB 边际 ⇒ 起跑线太高"这一个结论。
3. **本设计的提交无法单独 bisect**:那 9 笔的祖先不含 M2b(prefill 桶是 main 侧
   后合的)⇒ 拿它们 checkout 出来的二进制**根本没有桶可测**(配置直接 `unknown
   field: prefill_bucket`)。**回归只能在"合并后 vs 合并前"这一对上调**。

### 12.5 后续(2026-10-16):泄漏真因落到"双分配"+"错误路径不抽干",并留下一个未闭合项

**(1) 已修:合并引入的每节点双分配(真因之一)**。`alloc_out_node` 重构时把旧 `alloc_out(...)`
那行**留在了新行旁边**(三个臂全中)⇒ 每节点分两块:第一块白分配(不入 memo ⇒ 无消费者 ⇒
**永不释放**,`probe_name` 还把名字挂在它上面)⇒ 探针里的"未命名"那一砣就是它。
**实测**:verify 段定尺 313→**165**(单 slab 口径 263→**131**,单图 133)、session 47→**32**(26→**13**)、
图族总账 485→**271MiB**、boot 稳态引擎账 22457→**21428MiB**(**已优于合并前 21877**)。

**(2) 已修:错误路径不抽干竞技场**。`eval_ops_multi_scoped_env`/`eval_segment` 求值中途失败
(`?` 早退)⇒ 末尾清扫跑不到 ⇒ **本次求值的活块整套泄漏**(prefill 一次 GB 级)。桶族失败链
正是被它毒死:第一个桶 warmup 失败 → 泄漏 ~GB → 其后每个桶(含只需 5MiB 的)全跟着失败。
修后实测:失败后 `引擎 live` **回落到基线** ✓。

**(3) ⚠️ 未闭合:prefill 桶 warmup 的**中途活性回收**在合并后的树里不触发。**
用"计数器对拍"定谳(**把 `live_diag` 移植到 `6f128ad` 同档同图跑**,这是唯一可信的判据):

| t512 桶 warmup | 新块 | 真回收 | 峰值(探针) | 总量 |
|---|---:|---:|---:|---:|
| **6f128ad(合并前 + 计数器)** | 1566 | **1545(99%)** | **91MiB** | 8321MiB |
| **main tip(fix1/2 后)** | 1018 | **36(3.5%)** | **3031MiB** | 3.0GB(死在 OOM) |

- **总量逐桶一致**(t32:0.5 vs 0.52GB、t64:0.7 vs 1.04GB、t128:2.0 vs 2.1GB、t512:3.0 vs 8.3GB)
  ⇒ **同一张图、同一批分配**(树的形状没变);
- 回收**代码逐字相同**(`count_pending`/`track_new`/`track_alias`/`release_block`/回收循环
  四处 diff 全等;keep 白名单在非段化路径恒空)⇒ **同一段逻辑,结果 3.5% vs 99%**;
- 后果:桶的 **slab 定尺被放大 10-50×**(t32:519MiB vs 5MiB、t128:2079 vs 22、t512:3031 vs 91)
  ⇒ 桶族(五桶)在 24G 档要 **~1756-2925MiB** slab(合并前 174MiB);
- 已排除:FI split(关掉同败)、双分配(已修)、piecewise 两臂(关掉同败)、E1 revert、
  池几何(缩池同败)。
- **下一步(未做)**:在探针**峰值时刻**把每块的 `block_users`/`pending` 一并 dump
  (`live_diag` 加一张 id→(users,pending) 表,报告里打 top-N)⇒ 直接指认"它凭什么是活的"
  (users 抬着?pending 没减?);`/debug/metrics` 不暴露 `sink.count` 面,故计数器对拍只能
  靠移植(本轮已做,tooling 见 §12.1)。

**⚠️ 给主会话的安全建议**:在这条闭合前,**不要把 `prefill_bucket` 打开**(geobuckets)。
当前状态下:①t512/t256 仍拿不到;②即使拿到的桶(t32/t64/t128)也会**静默**占住
**519-2079MiB** 级的 slab ⇒ 净效果是负的。降到 `off` 或 `t512` 单桶前,先跑一遍
`slab 总账` 那一行确认桶的 MiB 数是否已回到个位/十位数。

### 12.6 ✅ 结案(2026-10-16):真因 = `Op::Call` 臂重复的 `track_new`

§12.5 的"未闭合项"已定谳并修复。三处**同源**合并缺陷(全部出自把 `alloc_out(...)`
机械替换成 `alloc_out_node(...)` 那一步):

| # | 缺陷 | 后果 | 修 |
|---|---|---|---|
| 1 | 三臂各多一行 `alloc_out_node`(**双分配**) | 每节点泄漏一块(不入 memo ⇒ 永不释放);探针里那砣"未命名"就是它 | `20a2609` |
| 2 | `Op::Call` 臂多一行 `track_new`(**users +1**) | **中途活性回收全死**(prefill live≈总量)⇒ 桶 warmup 峰值/定尺放大 10-50× ⇒ 桶族 0/5 | 本笔 |
| 3 | 错误路径不抽干竞技场(既有缺口,非本次引入) | 一次失败泄漏整套活块(GB 级)⇒ 失败链被毒死 | `20a2609` |

**定位链(可复用的方法学)**:①回收决定点**按名分账**(`users …MB×N(残余users≤2)`⇒
指向"记账多 +1"而非树/顺序);②`double_new` 检测 + 前 12 例 id/名字上屏 ⇒ 抓出重复行;
③**把计数器移植到对照版本对拍**(同档同图:真回收 99% vs 3.5%,而**总量逐桶一致** ⇒
同图同分配 ⇒ 只能差在记账)。

**结案实测**(27B,m2b 档,24G):桶族 **5/5 Captured**、prefill 桶族 slab **203MiB**(合并前
174)、prod 档引擎账 **19520MiB**(修复前 20111)、红线门逐字一致 ✓。
⇒ §12.5 的"安全建议(先别开 prefill_bucket)"**作废**;`slab 总账` 一行仍建议保留为
日常体检项(桶族应为 ~200MiB 量级,出现 GB 级即回归)。

### 12.7 ⚠️ 方法论:GPU/e2e 门**必须在守护的空卡上跑**(否则会伪造"正确性回归")

本轮一次**自伤型误判**实录:同一份代码,全场景 e2e

| 跑法 | 结果 |
|---|---|
| 未守护(卡上同时有主会话进程 16-17GB) | **0/3 失败**(三个不同断言) |
| 守护空卡(`memory.used < 500MiB` 才开跑) | **3/3 全绿**(54.2/53.1/53.8s) |
| 合并前 6f128ad(未守护,一次) | 失败:baseline 跨 boot 文本**乱码** |

**机理**(为什么抢占会伪造正确性回归):卡被共驻 ⇒ 显存不足 ⇒ 部分**图捕获失败**
(引擎按设计优雅降级回 eager)⇒ 同一份权重下**臂不同**(图 vs eager、capture vs
fallback)⇒ 数值路径不同 ⇒ 恒等门(spec ≡ 无 spec)、跨 boot 确定性门、token 账逐位
全都会"合理地"破 ⇒ **看起来像正确性回归,实际是环境**。典型签名:
①失败断言五花八门(每次不同);②伴随 `[vram-ledger] alloc 失败 … 驱动 free≈0` 或
`CUDA_ERROR_OUT_OF_MEMORY`(引擎构造/装载期);③**空卡复跑即全绿**。

**纪律**:跑任何 GPU 门/全场景前,先查卡(`nvidia-smi --query-gpu=memory.used`,
本工作区多会话共用,别信一次快照,要**轮询等空**);失败先按上表三签名判环境,再怀疑代码。
`[vram-ledger]` 那行(warn)就是为这个场景加的:它把"池/驱动/引擎"三账一次打全。
