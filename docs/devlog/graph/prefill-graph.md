# prefill CUDA Graph —— 设计与实现方案(2026-10-15 立项设计;首轮)

> 定位:本文件是 **prefill 图**的**设计**文档(不含施工记录)。设计输入有三份,均已入档:
> ① 分支 `feat/prefill-graph` 的实测台账(`工作树 prefill-graph(旧位置)/docs/arch/prefill-graph-施工.md`,
>    850 行,S0–S1.5-b + §12 设备侧定谳 + §13/§14/§15 真收益/判死);
> ② 本仓**现役**图机制(`crates/engine/src/graph_plan.rs` 5 处在用 + `crates/backends/cuda/src/state.rs` 捕获面);
> ③ 本轮新取的两把核尺子(`crates/kernels/tests/{fi,xqa}_prefill_probe.rs`)与已落的 `ChunkTables` / FI plan 硬门。
> ④ **参考实现**(2026-10-15 二轮指定)= `工作树 prefill-graph(旧位置)`(分支
>    `feat/prefill-graph` @ `dc7027e`,54 个未提交文件;S1/S1.5 已把**整模图跑成 `Captured`**,
>    `prefill_plan.rs` 214 行 = 槽面 + 根 concat 的现成骨架,`slab_hint.rs` 的 `SlabArena`
>    + 影子探针 = M1 的现成实现)。本设计**参考**它,但二轮把它的前提逐条按源码复核过。
> 与分支的关系:**分支的图代码一行未合**(用户裁决);本设计**重新设计**而不是照搬 ——
> 分支的结论(尤其"图对长 chunk = 0 收益")是本设计的地基,不是被推翻的对象。
>
> **修订记录**:2026-10-15 首轮(§0–§8 原样);2026-10-15 **二轮**(新增 §4.1c 源码级复核:
> F-5…F-10 六条事实 ⇒ **P-4 删除、P-5 新增、padding 升为主路、数值口径分两层**;§0/§3.2/§5/§6/§8 随之改)。

> **退役（2026-10-21）**：prefill 桶（`prefill_bucket` / `PrefillBuckets`）已在
> `perf-review` 工作树删除（待合并/待提交）；备份分支 =
> `backup/prefill-bucket-2026-10-21`（`main@23116c7`）。
> 退役依据：W4A8 主场景实测 prefill 仅 **+0.9…+1.0%**、TTFT@8192 −27…−45 ms；
> W4A16 + DFlash2 现场两对 **+1.1% / −2.2%**（不可复现，六点合计 −0.6%）；
> 代价 555–900 MiB + 2.5 s boot + 一套图维护；历史 AWQ +3.6% 在当前
> `chunk 1024 + tail_merge 1024` 形状下未复现。
> 本文保留为历史设计与施工记录，不再代表现役代码。
>
---

## 0. 先说结论(TL;DR)

1. **图的性能价值面很窄,且不在主战场**:分支设备侧定谳 —— prefill 是 **GEMM 绑死**
   (385 ms 窗口里 GPU 忙 94.6%,marlin 主发 290 ms = 76.5 TFLOPS ≈ 96–98% 的
   fp32-累加张量墙)⇒ 图省的是 host 命令税(~113 ms/385 ms,且被设备时间掩盖)
   ⇒ **长 chunk(t ≥ 256)图 = 0 收益**(分支实测 792/769/766 vs 803/772/770 ms)。
2. **图真正买得到的东西**:
   - **小 chunk(host 瓶颈区)**:分支扫长度实测"与 t 无关的固定段 ≈ 110 ms(1465 条命令的管道 + 每 chunk 首调 stall)",
     t=33/127 的墙时 136/133 ms ⇒ 这一段几乎全是命令税,图能整段抹掉;
   - **H2D/首调税归零**:`from_host` 叶子(ids/pos/slots/lens/seq_cu)+ FI plan/FI 表上传,
     在图上变成"输入槽原位写"与在窗外预先算好;
   - **时序鲁棒性**:A13 一类"host 节奏敏感"面的面积收窄(图内零 host 参与);
   - **为 PDL / 断点 / 未来 piecewise 铺路**(架构价值,不当提速卖)。
3. **本设计的核心判断(2026-10-15 二轮,源码级复核后)**:prefill 图的必需前置件是 **两件** ——
   **M-B slab 峰值定量 + 捕获窗内存复用**(分支 S1.5 实测:总和口径 9362 MiB ⇒ OOM;
   峰值口径 **193 MiB** ⇒ Captured;43× 收敛),**加 P-5 尾部 CTA 掩码**(二轮新发现,§4.1c)。
   **整段单图成立,且不再依赖任何推断**:二轮逐行核到,split 关时 FI 的 **plan 与 ctx 完全无关**
   (数组只由 qo 长度定),回放期 ctx 全走**设备表**(页链/`indptr`/`last_len`)⇒ 一个桶一张图,
   图内含注意力。piecewise 仅作 M2c 后备。
4. **建议执行序**:先量 **P1‴(命令批 / 零回执)** —— 它与图**争同一笔收益**(host↔server 往返),
   但**零捕获风险、覆盖全部 chunk**;量完仍不够(小 chunk 仍被卡)再投 M-A/M-B。
   本文档按"若要做,就这么做"给出完整设计与分阶段判据(§4–§6)。
5. **数值口径分两层(二轮修正;验证门照此写)**:
   - **图内一致门**(硬):同一桶下"捕获图回放" ≡ "同桶 eager(同 `q_cu`、同表件)" ⇒ **逐位相等**;
   - **跨档门**(软):图档 ≡ 现行**真 t** eager ⇒ tiling 随 t 变(tile 数不同)⇒ **数值近似 + 贪心逐 token 一致**,
     **不承诺逐位**。P3 原写"图 ≡ eager 逐 token 一致"对跨档不成立(除非 eager 也改走桶计划),故拆开。

---

## 1. 立项依据:两条已被钉死的实测(不要重新验证)

### 1.1 设备侧定谳(分支 §12;nsys 逐核 + CUPTI in-engine 互证)

| 项 | 量 | 判据 |
|---|---:|---|
| nsys 窗口(N=461 tok 单 chunk) | 385.1 ms / 2552 发内核 | `--cuda-graph-trace=node` 按内核时间轴切窗 |
| **GPU 忙** | **364.2 ms(94.6%)** | 逐核设备时间求和 |
| marlin 主 GEMM | **290.3 ms(75.4%)** | 326 发;单体 75-77 TFLOPS |
| 主机侧 CUDA API busy | 126.1 ms(31%) | launch 类 113 ms / 2552 发(**~44 µs/发**) |
| GPU 空转 | 20.9 ms(5.4%) | FI/cublas **每次 chunk 首调** 9.65 + 4.43 ms |

**三条旧账更正**(分支 §12.3,同一份证据):① `srv.op.*` 是服务端下发耗时,**不是设备时间**;
② "逐命令同步往返 = 385 ms 全部"是**假**的(设备 94.6% 忙);③ `[dflash] prefill encode 156 ms`
是 host 等 GPU,真设备时间 5.2 ms。

### 1.2 命令税的真实形状(本设计的收益池)

- 每 chunk **~1465 条命令**(launch/alloc/free 各约 1465,见分支 §11 残余账);
- 固定段 **≈110 ms 与 t 无关**(分支"扫长度"表:ptok 33/127/239/461/895/1778 →
  136/133/211/366/726/1432 ms ⇒ 边际 0.7–0.83 ms/tok);
- ⇒ **短 chunk(t ≤ ~128)是 host 绑死区,长 chunk 是设备绑死区**。
  本设计的一切收益都必须落在这个判据上说话。

### 1.3 两把核尺子(本轮实测,用于"注意力能不能进图"的取舍)

| 形状(单层注意力) | FI(生产,fp8) | XQA | 回落核 `chunked knhd` |
|---|---:|---:|---:|
| T=1 / ctx=3072 | **0.287 ms** | — | — |
| T=54 / ctx=3072 | **0.437 ms** | 0.91 ms | 6.7 ms |
| T=128 / ctx=512 | ~0.44 ms | 0.361 ms | 1.408 ms |
| T=512 / ctx=512 | **0.112 ms** | 1.114 ms | 2.305 ms |
| T=512 / ctx=16384 | **4.84 ms** | 26.9 ms | 79.8 ms |

⇒ ① 生产注意力**必须留 FI**(t=512 时 FI 比 XQA 快 5.6×,比回落核快 16×);
② **XQA 只在 `t<fi_min_tokens` 的尾块有位置**(3.9× 于回落核,已是生产事实);
③ ⇒ 图里要放注意力就只能放 **FI**;而它**能不能进图**取决于发射几何是否与 ctx 无关 ——
见 §4.1(已核:**能**,在 split 关 + `enable_cuda_graph` + 表件进槽三条下)。

---

## 2. 现状盘点:我们树里已经有什么(设计的前置件)

| 面 | 现状 | 位置 |
|---|---|---|
| 三态(捕获/回放/eager 兜底) | ✅ `GraphPlan::{plan,step}` + `PlanOutcome::{Captured, EagerFallback}` | `crates/engine/src/graph_plan.rs` |
| 静态输入缓冲 | ✅ `InputSlot` + `write_block` 原位写(**块 id 面,不是裸指针**) | 同上 |
| 输出收割 | ✅ `OutputSlot` / `output_block(name)` / `read_output_f32` | 同上 |
| **桶族先例** | ✅ MTP propose 图族:`for m in 0..=spec_depth { GraphPlan::plan(…) }` —— 一族多图、逐轮按 m 选 | `crates/engine/src/engine.rs:1010+` |
| 捕获机制 | ✅ `GpuCtx::graph_begin/graph_end`、`CaptureState{stream, slab, used, launches, cap}`、`carve_block`(零 cudaMalloc)、`Block::{Owned, Carved}` | `crates/backends/cuda/src/state.rs` |
| slab 定量 | 🟡 只有**总和口径**:`slab_hint::{set_hint,take_hint,meter_add,meter_take}`(warmup 计量=分配总和) | `crates/shared/src/slab_hint.rs` |
| G2 捕获窗护栏 | ✅ 放行 `Launch / Alloc / MemsetZero / GraphEnd / Close`;**Free/Htod/Dtoh/Sync/GraphLaunch 结构化拒** | `crates/backends/cuda/src/server/mod.rs:569` |
| **本设计的 S0 前置件** | ✅ **已落**(本轮从分支择取):`ChunkTables::build()` = 本 chunk host 侧表的唯一取值点(ids/pos/slots/lens/gdn_slot/seq_cu/FI 四件/XQA 三件) | `crates/engine/src/exec/prefill_chunk.rs` |
| FI plan 雷的守门 | ✅ **已落**(本轮择取):捕获窗内 plan 未命中 ⇒ **结构化拒**(plan 内 H2D 会被烘进图) | `crates/kernels/src/server/flashinfer.rs` |
| 快照/多会话面 | ✅ 与图无冲突(图只吃"本 chunk 表";会话/页链由输入槽注入) | — |

**缺口(本设计要补的)**:

| # | 缺口 | 影响 |
|---|---|---|
| G-1 | 捕获 slab **总和口径** ⇒ 一个 512-token chunk 需 9362 MiB(实测算)→ OOM ⇒ EagerFallback;根因:捕获窗内 `free_blocks` 对 `Carved` 块 **no-op**(`state.rs`),块不可复用 | 图根本装不下 |
| G-2 | 无**断点原语**:`GraphPlan` 只跑**一个**闭包,捕获窗内不能"图段 → eager → 图段" | 注意力进不了图 ⇒ 无法组出整 chunk 图 |
| G-3 | `EagerFallback` 态下 `step()` 会**再跑一遍 eager**;外层 eager 路又跑一遍 ⇒ **状态双推进**(分支 §9 已实测;我们现有 decode 图靠"外层不再跑"避免) | 正确性雷 |
| G-4 | 无"**布局 t / 语义 t**"分离:padding 后图内步长 = `t_pad`,下游切片/dtoh 用真 t ⇒ **静默错位读** | 正确性雷 |
| G-5 | 无**捕获静止点**约定:prefill 有状态副作用(KV 池写 + GDN 递推),warmup 会灌假数据 | 正确性雷 |
| **G-6** | **无尾部 CTA 掩码**:`params.block_valid_mask = nullptr`,而上游只在 split 档分配该区 ⇒ `padded_batch_size > 真 tile 数` 时尾部 CTA 读 staging 残留(**静默**) | 桶几何下必然触发(§4.1c F-6/F-7),P-5 补 |

---

## 3. 设计目标与不做清单

### 3.1 目标(按优先级)

| # | 目标 | 判据 |
|---|---|---|
| P1 | **小 chunk 墙时**(host 绑死区) | `t ≤ 128` 桶:chunk 墙时 **−60…−100 ms**(抹掉 ~110 ms 固定段的大部) |
| P2 | **零 H2D / 零首调税** | 每 chunk `from_host` 次数 = 0(全走输入槽);FI plan 全在窗外预置 |
| P3 | **数值恒等** | greedy 逐 token 一致(图 ≡ eager);spec-off 下**token 级逐字**;spec-on 下 content 级 |
| P4 | **零功能风险** | knob 缺省关;桶未命中/捕获失败/任何 `Err` ⇒ **eager 原路**(不给部分结果) |
| P5 | 架构收益 | 图内 host 参与 = 0(A13 类面收窄);为 PDL(`enable_pdl`)留接口 |

### 3.2 **不做**(分支已证伪或已关闭;白纸黑字免得重走)

| 不做 | 依据 |
|---|---|
| 为"长 chunk 提速"做图 | 设备 94.6% 忙、GEMM 96–98% 墙 ⇒ 图 0 收益(§1.1) |
| 拿 XQA 换 FI 进图 | FI 在 t=512 快 5.6×(§1.3) |
| `OWL_NO_RECLAIM` 式"永不 free" | 那个 5× 是**伪影**(请求 OOM 直接失败,`curl -w` 量的是失败返回) |
| 精修"No padding 单桶" | 调度器给的 chunk 长 = 510/498(生产实测)⇒ 单桶 `t==512` **几乎不命中** |
| 捕获窗内做 FI plan / `from_host` | FI plan 内 H2D 会被烘进图(可变 staging ⇒ 回放取旧内容=静默错调度);本题已立硬门 |
| turn 中途惰性捕获 | 静止点律(§4.5) |
| 靠"陈旧表项恰好无害"兜尾部 CTA | §4.1c F-6/F-8:尾部 CTA 的行为由 `page_locked staging` **残留内容**决定 = 静默依赖 ⇒ 必须显式 mask(P-5) |

---

## 4. 机制设计

### 4.1 整段单图(首选)与它的五条前提(P-4 已删、P-5 新增;见 §4.1c)

**目标形态**:一个桶一张图,图内**含注意力**:
```
一次捕获(桶 t_pad,窗口外把 plan 算好):
  embed → [64 层:norm/qkv/o/MLP/GDN(varlen) + KV 写(t_pad 格) + FI 注意力] → 全行 hidden + taps
回放(每 chunk):写输入槽(ids/pos/slots/lens/seq_cu + FI 三件)**→ 1 条 graph_launch → 出图尾(eager)
```

**为什么成立(2026-10-15 代码级核实)**:

| # | 事实 | 出处 |
|---|---|---|
| F-1 | FI 的发射几何 = `dim3 nblks(padded_batch_size, 1, num_kv_heads)` —— **只由 q 瓦片数(即 q 行数)+ KV 头数决定** | `cu/flashinfer/include/flashinfer/attention/prefill.cuh:4598` |
| F-2 | 源码注释明写:`padded_batch_size == 0` 分支 "**won't happen in CUDAGraph mode because we fixed the padded_batch_size**" ⇒ 上游**就是**为图模式设计的 | 同上 `:4592-4594` |
| F-3 | KV 长度/页链在**回放期从设备表读**(`paged_kv.indptr` / `last_page_len`,即我们的 `indptr`/`last_len` 输入槽);图模式另有 `params.total_num_rows` **设备指针**槽 —— ⚠️ **二轮更正(F-9)**:该槽只在 split 合并路径被读,生产档 split 关 ⇒ **不需要**(P-4 已删) | `prefill.cuh:4579-4600/4558` + 我方 `server/flashinfer.rs` |
| F-4 | 我们的 plan **默认就是 split 关**:`disable_split_kv = !FI_SPLIT_KV`(env 缺省空 ⇒ true)⇒ 工作项 = q 瓦片 × 内部 KV 循环,plan 表与 ctx 无关 | `cu/flashinfer/owl_fi_attention.cu` + `server/flashinfer.rs` |

**五条前提(设计要改的地方;二轮:P-4 删除、P-5 新增)**:

| # | 前提 | 现状 → 改法 |
|---|---|---|
| **P-1** | ✅ **已落(2026-10-15)** —— `plan_key()` 纯函数:split 关时 `ctx_total` **归一化为 0**(F-5:那三条 ctx 链只在 split 分支被消费)⇒ 一个桶一份 plan 跨 ctx 复用;`split`/`graph` 两个开关位显式入键(切档不得复用错 plan);单测 `ctx_enters_key_only_when_split_on`。原设计文本: | 现状:plan 键含 `ctx_total`(`PlanKey{…, ctx_total, …}`)⇒ 每 chunk 重算,且捕获窗内必然 miss(→ 硬门拒捕)。改:split 关时**键去掉 ctx_total**(§4.1c F-5 已证其不影响结果,但**split 开时必须留着** ⇒ 键加 `split_on`);`total_rows`(=t_pad)仍在键;图档统一 `enable_cuda_graph=true` |
| P-2 | **plan 的 workspace 块地址恒定** | 现状:`int_ws`/`float_ws`/`page_locked` 是 `FiRuntime` 装配期持久 scratch ✓ **已满足** |
| **P-3** | ✅ **已落(2026-10-15)** —— `StatePool.fi_tables`(boot 一次分配 `q_cu/indices/indptr/last_len` 四块,容量 = 池页数)+ 每 chunk `write_blocks_f32` **原位写**(单条批量 `HtodChunks` 命令;回执 = 设备侧搬运完成 ⇒ 回执即落地,无跨流竞态);旧实现每 chunk **4 次 `eval_ops(from_host)` 往返同步** ⇒ 现少 3 次往返,且地址恒定(图可直接烘)。原设计文本: | 现状:每 call 由 `eval_ops` **新建块** ⇒ 地址每次变 ⇒ 图读不到新值。改:桶级固定块(容量 = 池页上限)+ `write_block` 原位写。~~二轮收敛到三件:`q_cu` 走 `[0, t_pad]` 常量(槽 init 一次,捕获烘死,不必每 chunk 写)~~ ← **三轮更正(M2b 实测)**:`q_cu` 也**必须每 chunk 写**(写「真」`[0, t_real]`)—— 核的 `qo_len` 从**设备** `q_indptr` 读(`BatchPrefillPagedParams::get_qo_len`),因果窗口 = `kv_len(设备) − qo_len(设备) + row`;桶几何下 `t_pad` 只该进**形状**(与 plan tiles),真语义全由设备表定 ⇒ `qo_len`/`kv_len` 必须**同口径**,否则窗口错位 `t_pad − t_real`(静默错值) |
| P-5 | **尾部 CTA 必须显式 mask**(二轮新发现) | 现状:`params.block_valid_mask = nullptr`,而**上游只在 `split_kv` 档分配该区**(§4.1c F-7)⇒ `padded_batch_size > 真 tile 数` 时尾部 CTA 读 staging 残留(**静默**,F-6/F-8)。改:适配器在 `int_ws` **自建 mask 区** + plan 后按 `i < tiles_plan` 填;**行 padding 下该 mask 是静态的**(桶级常量,零回放开销) |
| ~~P-4~~ | ~~`total_num_rows` 走设备指针(真行数)~~ | **二轮删除**:该字段只在 split-kv 合并核(`VariableLengthMergeStates` / `VariableLengthAttentionSum`)被读(F-9),生产档 `disable_split_kv = true` ⇒ 结构上不读。**将来若开 split 再补**(一个 1×u32 输入槽)** |

**唯一必须的实验** = §4.1c 末尾的 M2a 四步(0.5d;FI 探针已就位)。三条推断(F-5/F-8/F-9)
任一被实测否证 ⇒ 退 §4.1b 的断点后备路。

### 4.1b M-A(后备):断点原语(capture 分段 + 段列表回放)

**形态**(对齐 SGLang `BreakableCUDAGraph` / vLLM `breakable_cudagraph.py`,但用我们的块 id 面):

```
捕获期(一次 plan,闭包顺序不变):
  graph_begin ─► [非注意力段:embed/norm/qkv-proj/MLP/GDN…] ─► graph_end ⇒ seg0
               ─► eager 断点:KV 写 + FI 注意力(用 t_pad 几何 + 真表件)
               ─► graph_begin ─► [下一层段] … (每层两个图段 + 一个 eager 断点)
回放期(每 chunk):
  for (i, seg) in segments: graph_launch(seg[i]); break_fn[i]()   ← 断点 eager 在回放时**重放**
```

**四个必须闭合的不变量**(缺一即静默错值):

| # | 不变量 | 落地 |
|---|---|---|
| I-1 | **断点 eager 的输出必须落在"下一段读的那个地址"** | 断点输出**绑定到稳定槽**(下一段的 `InputSlot` 块);回放时 eager 写槽原位 ⇒ 地址恒定(我们 `InputSlot`/`write_block` 现成) |
| I-2 | 断点 eager **输入**也必须稳定(上一段的输出块) | 上一段由 slab `carve_block` 切出 ⇒ `Block::Carved` **活到 slab 释放**、`free_blocks` 对它 no-op ⇒ 地址天然恒定(分支 §5.2 的"白给"结论) |
| I-3 | 段间 mempool 同源(slab 同一条) | 全部段共用**同一次 `graph_begin` 创建的 slab**(段只"暂停"捕获,不重开 slab) ⇒ 见 I-4 |
| I-4 | `graph_end → graph_begin` 期间 **slab 不释放** | 需要把 `CaptureState` 从"一次捕获"升级为"**一次 plan = 一条 slab + N 段**";`graph_end` 只结束**当前段**,slab 由 plan 级持有 |

**服务端改动**(最小面):

- `Command` 侧新增/复用:`GraphEnd`(已存在)与 **`GraphBegin` 的"续段"语义**——
  加一个 `Command::GraphSegmentBegin`(或 `GraphBegin { continue_session: bool }`),
  语义 = "在同一 slab 上开新段";G1(不可嵌套)约束改为"不可**并行**开两段",允许串行续段。
- G2 护栏放行清单 **+ `Free`**(捕获窗内 `free_blocks` 三条路零 CUDA 调用:
  跨流栅栏本就在 `capture.is_none()` 下跳过;Carved→arena 纯 host 簿记;Owned 防御性不释放)
  —— 这是 M-B(4.3)的前提,也是"锚定活性回收"的前提。
- `graph_launch` 回放序由引擎侧驱动(段列表住 `GraphPlan`),服务端不加新概念。

**引擎侧改动**:

- `GraphPlanDesc` 增 `breaks: &[BreakSpec]`(**声明式**:断点位于第几个闭包节点/第几层),
  闭包侧不改语义 —— 解释器在 `_eval_node` 命中断点标记时调用 `PlanCtx::break_here()`;
- `PlanCtx::break_here()` 在**捕获期**做 `graph_end → 记段 → 执行断点 eager → 绑定输出槽 → 续段`;
  在**回放期**注册为 `break_fn[i]`;
- `PlanOutcome` 增第三态 `CapturedSegmented { segments, breaks }`;`step()` 走段列表;
- **I-1 的 API 缺口**:现有解释器只会 `alloc_out`(新块)⇒ 需要 `eval_ops_into(树, 稳定块)`
  ("把归约结果写进给定块"),断点 eager 与图输出回填都用它。

### 4.1c 二轮源码级复核(2026-10-15)—— 决定 P-1…P-3/P-5 与 padding 主路

逐行核了三处:FI 0.6.18 的 vendored 源码(`scheduler.cuh` / `prefill.cuh`,行号见下)、
我方适配器(`owl_fi_attention_common.h` + `server/flashinfer.rs`)、GDN 两核(`cu/gdn/*.cu`)。

| # | 事实(出处) | 推论 |
|---|---|---|
| **F-5** | **graph 档的 plan 与 ctx 无关**:`disable_split_kv=true` ⇒ `kv_chunk_size=INT64_MAX`、`split_kv=false`、`num_chunks_kv=1`;`kv_len_arr` / `effective_kv_len_arr` / `kv_chunk_size` 三条链**只在 split 分支被消费**(`scheduler.cuh:620-700`)。数组内容只由 `qo_indptr_h`(即 t)与 hq/hkv/hd/page 定 | **plan 按桶一次、跨 ctx 复用** ⇒ P-1 成立,且比首轮写的更强(原设计只说"图模式下按桶算",这里证明**结果本身**与 ctx 无关) |
| **F-6** | graph 档 `padded_batch_size = max(max_batch_size_if_split, total_num_tiles_q)`(`scheduler.cuh:673`),而 `max_batch_size_if_split = 2·SM / hkv`(**与 t 无关**) | 小桶必然 `padded ≫ 真 tile 数`:27B(hkv=4)+3090(82 SM)⇒ 41;t_pad=512 时 tiles = ceil(512·6/64) = 48 ⇒ **无尾**;t_pad=256/128/64/32 ⇒ tiles 24/12/6/3 ⇒ 尾 **17/29/35/38** 个 CTA |
| **F-7** | `block_valid_mask` 只在 `if (split_kv)` 块内分配(`scheduler.cuh:891`);核侧守卫 = `if (block_valid_mask && !block_valid_mask[bx]) return;`(`prefill.cuh:3019`) | split 关 ⇒ **上游不生成 mask**;P-5 须由适配器在 `int_ws` 自建(语义上游已有,只是不给分配) |
| **F-8** | 尾部 CTA 若真跑:`request_idx/qo_tile_idx` 读 staging **残留**;batch=1 下 `request_idx` 恒 0 ⇒ 不越界;越界的写入被 `if (o_idx < qo_upper_bound)` 守卫(`prefill.cuh:2175/2227`)⇒ **不写错行,但白跑一遍 KV 循环** | 残留"无害"是**巧合不是不变量**(残留内容取决于上一次 plan 写了什么)⇒ 显式 mask(P-5),顺带省下白跑的 tile |
| **F-9** | `params.total_num_rows` 只在 split 合并路径被读(`prefill.cuh:4558/4750`) | **P-4 删除**(生产档 split 关;`params.total_num_rows = nullptr` 维持合法) |
| **F-10** | graph 档 `cta_tile_q` 由最坏情形 `max_seq_len = total_rows - batch + 1` 定(`scheduler.cuh:602-608`);**batch=1 时与 eager 同式**(本仓 FI 调用恒 `batch=1`) | 开 `enable_cuda_graph` **不改瓦型** ⇒ 无性能副作用;可统一用 graph 档 plan 服务 eager 与图(eager 只需同时带 mask) |

**推论 A:padding 是主路(不是可选)**

- 桶几何下 FI **张量**行数 = `t_pad`(布局),但**语义表全真**:`q_cu`(`qo_len`)、`indices`/`indptr`/`last_len`(`kv_len`)
  都按 `t_real` 每 chunk 写进**持久设备槽**(M2b 实测;见 P-3 的三轮更正)
  ⇒ ① mask 静态(P-5 **零回放开销**)② 因果窗口与 eager **逐字同式**(`kv_len − qo_len + row`)③ 代价 = 假行白算。
- 假行的因果掩码**天然封死**:核内 `kv_len` 来自 `indptr`(真 ctx)、写位由 `qo_upper_bound = min(qo_len, …)` 封顶
  ⇒ 假行"看得到真 KV、写不到真行",只白算,**不向真行泄 NaN/Inf**(每行 softmax 独立)。
- 白算的**绝对量**在噪声里:FI 只占 chunk 的 0.1–0.5% 量级(16 层 × ~0.1 ms @t=512/ctx=512);
  最坏情形(33 行装进 512 桶)放大 ~10× ⇒ 仍在 5% 内。**GEMM 侧的桶 padding 白算才是主流**(vLLM 同款,与图无关)。
- GDN 面**已是 varlen 驱动**(二轮逐核核过,律① 成立):
  `gdn_conv_fwd` 行区 = `[cu_seqlens[s], cu_seqlens[s+1])`,`t0>=t1` 的 tile 直接 return,state 写回用 `end`;
  `gdn_chunk_scalar` 同构(`seq_off` 驱动、state 按 seq 边界写回)⇒ 传 `seq_cu=[0,t_real]` + t_pad 张量时,
  **假行既不推进 conv/rec、也不碰 state**。
- 池写面:`k0_write` 的网格/形状取自 `tokens`(= t_pad),槽表给假行**必须**指向 dump 页(§4.2 律②)——
  否则假行写进会话页链(**静默污染**)。

**推论 B:数值口径分两层**(写进验证门,取代 P3 的粗写)

| 门 | 比较对象 | 判据 |
|---|---|---|
| **图内一致(硬)** | 捕获图回放 vs **同桶 eager**(同 `q_cu`、同表件) | **逐位相等**(同 tiling、同调度数组) |
| **跨档门(软)** | 图档(桶 t_pad) vs 现行**真 t** eager | 数值近似(容差)+ **贪心逐 token 一致**为目标;tile 数/边界随 t 变 ⇒ **不承诺逐位** |

**M2a 具体实验** —— ✅ **已做(2026-10-15);见 §4.3f 判决**

原设计稿:

**M2a 具体实验(0.5d;`crates/kernels/tests/fi_graph_replay_probe.rs`)**

1. 单层 FI,桶 `t_pad = 512`,**graph 档 plan 在窗外算一次**(带 P-5 mask);
2. **表件换 ctx 回放**:表件 A(ctx=512)捕获 ⇒ 回放表件 B(ctx=1024/2048)⇒ 与 eager(同 B)对拍 ⇒ 验 F-5;
3. **小 t 落大桶**:桶 512 + `q_cu=[0,512]` + 表件 ctx=512,**真行 t=64、行 64..511 填 NaN** ⇒ 与 eager(t=64)对拍**前 64 行** ⇒ 验"假行不泄 NaN"(推论 A);
4. **尾掩码**:构造 `padded_batch_size > tiles`(t_pad=128 ⇒ 尾 29)⇒ 未掩码跑一遍**取证**(白跑 / 不越界),
   带掩码跑一遍 ⇒ 逐位等于 masked-eager ⇒ 验 F-6/F-7/F-8。

判据:全部成立 ⇒ **M2b 直接上**;任一否证 ⇒ 退 §4.1b(piecewise)。

### 4.2 padding 与"布局 t / 语义 t"分离(桶策略)

**桶族**:`t_pad ∈ {32, 64, 128, 256, 512}`(几何阶梯;t < 32 用 32 桶 = 浪费 ≤31 行)。
每桶**一条 slab + 一条段列表**;桶选择在 `prefill_chunk` 入口做(调度器给 `t` ⇒ 取 `t_pad = ceil(t)` 桶)。

**三律**(分支 §3 的硬约束;**下面三条的落地方式都已在现役代码里核对过**,无需改核):

| # | 律 | 本设计的落地(已核对) |
|---|---|---|
| ① | 假行**不得推进状态** | **`seq_cu` 槽传 `[0, t_real]`**(varlen)。已核对:`owl_gdn_conv_fwd_f16` 的行区 = `[cu_seqlens[s], cu_seqlens[s+1])`(`gdn_conv_fwd.cu:79-82`,N1 后 `ts = ceil(seq_len/gridDim.y)` 由**真长度**定 ⇒ gridDim.y 是 t_pad 定型也不多算);`gdn_chunked`/`gdn_scalar` 两臂本就是 varlen(`slot`+`cu_seqlens`)⇒ 假行不推进 conv/rec 状态 |
| ② | 假行的**池写必须无害** | **dump 页**:池多留 **1 页**(`PoolPlan.nb + 1`),`slot_mapping` 槽的 padding 区全指向 dump 页的同一排槽位(session 页链**不含** dump 页 ⇒ 注意力永读不到)⇒ KV 写可以**进图**(t_pad 几何)、不改核。已核对:`k0_write*` 的 pick 是 `grid = (tokens,1,1)`,`tokens` 取自输入形状(`driver.rs:661/1267`)⇒ 图内 t_pad ⇒ 只多写 padding 行,落点就是 slot 槽给的 dump 页;`qknorm_rope_kv_insert_*` 同构(`grid=(tokens, hq+hkv, 1)`) |
| ③ | 采样读**真行** | 图输出**全行 hidden [t_pad, hidden]**(+taps):末行窄切 + lm_head + argmax 一律**出图**走 eager —— 偏移 `(t_real-1)·dim` 是**烘焙不得的 host 标量**(分支 §3.E 同判)⇒ 天然图边界,顺带把 dtoh/收割面积收到一条 |

**副产品**:律③ 让"最后一层 attn 的输出"成为图的**唯一**输出面(少一块收割/dtoh);
DFlash2 的 taps 同理(`tap_i` 出全行,切片在 eager 侧做)。

### 4.3 M-B:slab 峰值定量 + 捕获窗内存复用

**病根**(分支 §9 代码级定谳):`carve_block` 是**纯 bump**、`free_blocks` 跳过 `Carved`
⇒ 捕获窗内 free 全 no-op ⇒ 容量下限 = **分配总和**(512-token chunk 实测 **9362 MiB**)。

**设计**(分支 S1.5 的方案,经其实测背书):

```
SlabArena     bump + 尺寸类(size-class)free list,256B 对齐;
              free_blocks 对 Carved 块**推回 arena**(捕获后保持 no-op)
捕获窗归约   GraphPlanDesc.reclaim_capture = true ⇒ 捕获/回放期的输出归约走
              scoped 回收(捕获期死块即时归还)⇒ 需求 = **峰值活跃集**
定量口径     warmup 用**影子 arena 探针**(probe_on/probe_alloc/probe_end)取高水位;
              plain 族(无归约回收)仍用 meter 总和 —— **口径必须与模式一致**
             (分支实测:混用会把 fold/propose 族定小 ⇒ 捕获中途耗尽当场炸)
```

**实测背书**(分支,真机):总和 8321 MiB ⇒ OOM;峰值 **193 MiB** ⇒ `Captured`;
其它图族 13/133/1/3/4/6/7 MiB **零回归**;持久 arena(opt-in)墙时 **0 变化**
⇒ **arena 只作定尺价值,不作提速项**。

#### 4.3b 显存账与省显存方案(S-1…S-4;2026-10-15 二轮追问补)

**先纠一笔账**:峰值活跃集与 t 成比例(所有中间张量都是 `t × 常量`:MLP gate/up 35.6 MB + 
qkv 13.6 + GDN 16.8 + 输出 hidden+taps 31 @t=512/27B)⇒ 桶族之和是**几何级数**而非线性:

```
Σ peaks ≈ 193 × (1 + 1/2 + 1/4 + 1/8 + 1/16) ≈ 374 MiB   (全 5 桶;非头注那句粗写的 ~1 GB)
三小桶 {32,64,128} ≈ 193 × (1/16 + 1/8 + 1/4) ≈ 45 MiB
```

**换算成"钱"**:fp8 KV 主池 = 16896 tok × 32 KiB(2·hkv·hd·1B·16 层)/ tok = **540 MiB**
⇒ 全 5 桶 ≈ **0.69 × 池**,等价损失 ~11.5k token 的上下文/并发槽;三小桶 ≈ **0.08 × 池**。

| # | 方案 | 省多少 | 代价 | 新代码 |
|---|---|---|---|---|
| **S-1** | **裁桶**(只建价值面 = 小桶):`t_pad ∈ {32,64,128}`,大 chunk 走 eager | 374 → **45 MiB** | 长 prompt 首块(512)拿不到图 —— 但它在**设备绑死区**(分支 §12),图本就 0 收益 | **零**(只改桶表) |
| **S-2** | **惰性捕获 + LRU 驱逐**(稳态 = 活跃桶数 × 档) | 到 **max 单桶峰值 × N**(N=1 ⇒ 193 MiB 封顶) | 需 `GraphPlanDesc.warmup:false` + 固定 slab 档(`capture_slab_mb`;否则捕获前不知道容量 ⇒ 窗内 OOM 不可恢复);换出一次重捕 = 图对象释放 + 重归约(~tens of ms;**不含** warmup —— warmup 单桶 ≈ 0.8 s,eager 整模一遍) | 中(§4.5 已挂的 v2 项) |
| **S-3** | **单图覆盖区间**(上界捕获 + 设备侧早退):捕获 `t_top`,回放 `t_real ≤ t_top` | 等效于"桶数↓" | **GEMM 不能早退**(M 是网格)⇒ 白算 `t_top/t_real` 倍 GEMM —— 而 GEMM 是 prefill 主头(§1.1 76%)⇒ 只适合比值小的区间;{512 覆盖 384–512} + {128 覆盖 96–128} + {32} ⇒ 全区间覆盖、显存 ≈ **255 MiB**、白算 ≤ 4× | 中(需逐算子核"能否早退":GDN ✓(F-10 旁证)、FI ✓(P-5)、KV 写 ✓(slot 驱动)、GEMM ✗) |
| **S-4** | **压每桶峰值**:① 影子探针加"按分配名汇总"先归因 ② marlin `ws`(`v2_workspace_len(M)×4`,每层 carve)改**持久 scratch**(同 FI `int_ws/float_ws` 体例;峰值从"重叠 N 层 ws"→"1 份")③ taps 输出面(6×t×dd×2B = 31 MB @512)降精度/按需 | 待归因实测(预估 1/3–1/2 峰值) | ②③ 是行为/精度改动,需单独恒等门 | 中 |

**建议组合**(待用户裁):**S-1 常驻 {32,64,128} + S-2 惰性补大桶** ⇒ 稳态 45–85 MiB;
若不愿动 `warmup:false`,则退回 **S-1 + S-3 的 {512,128,32} 三档**(255 MiB、覆盖全区间、零新机制)。
⚠️ 唯一**不要**做的是 S-1 的"精确桶补齐"(每 t 一图)—— 图数/显存都爆,且静止点律禁 turn 中途捕获。

### 4.3d **已落:内存管理的四把原语 + M1 峰值口径**(2026-10-15;M2b/M2c 的地基)

这一节记录**已经落地并过门**的机制(不再是设计意图)。四把原语解决同一族问题:
**图烘的是裸设备指针 ⇒ 谁也不敢释放、谁也不敢复用**;拆开后各自可判定:

| # | 原语 | 机制 | 代码 | 它挡的错 |
|---|---|---|---|---|
| **①** | **图引用租约** | 捕获窗内每次 `block_ptr`(只对 `Owned`)⇒ 记窗口集;`graph_end` 收窗 ⇒ 入租约(id→存活图计数);`free_blocks` 命中即**拒收 + 计数 `srv.count.free_pinned_reject` + WARN(带持有图 id)** | `state.rs::GraphRefs` | 池件/输入槽被 free ⇒ 设备池原址重发 ⇒ 图变**陈旧指针**(静默错值)。窗内 Owned 死块另有一道防御性 no-op(租约的窗内盲区) |
| **②** | **abort(作废窗)** | `Command::GraphAbort`(窗内**允许**):退出捕获态 + 销毁半成品图 + **窗内自产块整批移出账房** ⇒ slab 释放 | `state.rs::graph_abort` / `release_window_blocks` | 失败捕获**留尸**:旧实现窗内块无人认领 ⇒ 整条 slab(最坏 GB 级)钉到进程结束(旧注 "挂账:graph abort" 关闭) |
| **③** | **destroy(单图退休)** | `Command::GraphDestroy`:排空回放 → 解除本图租约(计数递减,归零解钉)→ 本图 slab 块整批移出 | `state.rs::graph_destroy` | 图永驻(显存 = 活跃桶数 → 全桶数);也是 S-2/LRU 的前置件 |
| **④** | **close / abandon(plan 侧)** | `GraphPlan::close()`:destroy 图 + **归还输入槽**(次序不可颠倒:图在时槽被租约钉住,先 free 会被响亮拒收);`plan()` 失败路径 `abandon()`:abort + 槽归还 | `engine/graph_plan.rs` | 每次失败捕获漏一批槽;单图退休的半途状态 |

**M1(slab 峰值口径)同轮落地** —— 这是让**大图能捕上来**的那一刀:

| 面 | 旧 | 新 |
|---|---|---|
| 捕获窗分配器 | 纯 bump(无回收)⇒ 容量 = **分配总和**(27B/512-tok 实测 **9362 MiB** ⇒ OOM ⇒ 图根本捕不上) | `SlabArena`(bump + 尺寸类回收,256B 对齐)⇒ 容量 = **峰值活跃集**(分支实测 **193 MiB**) |
| 窗内 Free | **结构化拒**(G2 白名单) | **放行**(零 CUDA 调用:Carved → 回 arena;Owned → 防御性不释放;跨流栅栏本就在 `capture.is_none()` 守卫下跳过) |
| 归约 | `eval_ops_multi`(plain,窗内块常驻) | `reclaim_capture: true` ⇒ `eval_ops_multi_scoped_env`(**只归还不可达中间块** ⇒ 复用点在所有消费者之后) |
| slab 定尺 | 总和口径 meter ×1.15 | **口径与模式同源**:锚回收 ⇒ 影子 arena 高水位(峰值);plain ⇒ meter(总和)。混用会把族定小 ⇒ 捕获中途炸 |
| 惰性捕获前置 | `warmup` 恒跑(计量 + 声明门禁,但有状态副作用) | `GraphPlanDesc.warmup` 开关:false ⇒ 不跑闭包(⇒ 无副作用,可脱离静止点律)**但必须 `slab_fixed`**(否则结构化拒),代价 = 声明门禁移到窗内(失败走 abort) |

**验证(全绿,可复现)**

| 门 | 内容 | 结果 |
|---|---|---|
| `cargo test -p owl-shared --lib` | `SlabArena` 对齐/复用拆余量/耗尽拒/零字节退化/探针峰值 | **43/43**(含新增 4) |
| `cargo test -p owl-cuda --lib` | `GraphRefs` 窗范围/收窗入租/新窗不清旧租/多图计数解钉/幂等 | **7/7** |
| `OWL_TEST_DEVICE=0 cargo test -p owl-cuda --test gpu_server` | 步骤 9 租约(捕获块 free 后仍在册/仍可发射;自由块回收 ⇒ `DeadBlock`)、**步骤 10 窗内 Free 复用**(6×16MiB 总和 96MiB > 默认 64MiB slab,峰值 2 块 ⇒ 有复用才 Captured;**负例对照实测:去掉归还即 `捕获 slab 耗尽`**)、**步骤 11 abort/destroy**(窗内块整批作废;destroy 后图不可回放 + 租约解除 ⇒ 可 free) | **2/2** |
| `OWL_TEST_DEVICE=0 cargo test -p owl-engine --lib graph_plan` | `gpu_capture_matches_eager`(plain 口径回放 ≡ eager)+ **`gpu_capture_reclaim_matches_eager`**(锚回收口径回放 ≡ eager —— 静默错值的守门) | **7/7** |
| `cargo test -p owl-engine --lib` | 全库回归 | **34/34** |
| `cargo check --workspace --all-targets` | 线协议/面扩展的涟漪 | 干净(仅存量告警) |

**⚠️ 复用安全性的唯一依据**(写进代码注释,免得后人误改):{scoped 归约只归还**不可达**中间块}
+ {捕获按**求值序**记录} ⇒ 复用点在所有消费者之后;回放按同一序跑 ⇒ 复用等价于"更大的一次性
arena"。**输入/输出槽不参与归还**;`Carved` 的捕获后 free 仍是 no-op(要回收只能整图 destroy)。

**未做(有意)**:分支的 **S1.5-b 持久 arena**(`vram_arena_mb`,运行期块不回驱动池)——分支自己
实测**墙时零变化**(施工文档 §11),属"定尺价值不是提速项" ⇒ 不引入(免得给 eager 路添
"arena 耗尽"这一类新失败模式)。

**仍未落(⑤ 的策略面)**:惰性捕获的**策略**(桶 LRU 驱逐)要等 prefill 桶族(M2b)存在才有
调用方 —— 现在写 `BucketCache` 只能是不可测的死码。原语(②③④ + `warmup:false` + `slab_fixed`)
已就位,M2b 落地时直接组合即可。

### 4.3e **家族面捕获入境纪律**(2026-10-15,M2b 的第二道硬门)

> **2026-10-15 补:P-6b 已落地**(worktree `feat/prefill-graph-v2`)—— `gdn_chunked`
> 的三张 FLA 表(cu/coff/idx)从 runtime 私有缓存改为**调用入参 + 引擎持久槽**,
> 与 FLA 上游 API(`cu_seqlens`/`chunk_indices` 入参)一致。判据:
> **红门探针** `crates/kernels/tests/gdn_bucket_pad_probe.rs`(桶 128 行装 64 真行、
> 假行填有限垃圾)—— 修前**终态 Δ=3.578e-2**(真行输出 Δ=0)、修后**两者都是 0.000e0**。
> 实现要点:①表布局常量单源 `family::gdn_chunked::{CU_OFF, COFF_OFF, IDX_OFF, BT,
> table_bytes, table_host_bytes}`(`BT` 由单测与 AOT 配置表锁死)②引擎 `StatePool::gdn_tables`
> 每 chunk 写(`nt = ceil(t_real/BT)`)③`ForwardCtx::gdn_tables` 注入,缺了**拒发**
> (与 `gdn_slot_host` 同纪律)④声明面 +1 块 +1 标量,锁表/F3/家族测试同步。

**为什么单列**:G2 捕获窗白名单只管 client 来的 `Command`(Htod/Dtoh/Sync 拒),
而家族 `run()` 在同一 actor 线程里**直接调 `DeviceRes`** ⇒ **白名单对它无效**。
旧口径是"家族 grave 律"(代码原话:"捕获窗内由家族 grave 律保证不再分配")= **靠自律**。
实测定谳(本地台账区,2026-10-03 深夜):cuBLAS gemv
splitK 在捕获流上内部 `malloc` ⇒ 图里烙下 **192 对 `MEM_ALLOC/MEM_FREE`(3088B)** 节点
(回放期重分配 + 派发税);修法 = `cublasSetWorkspace` **预绑进程级驻留工作区**。

**⇒ stream-ordered 分配(`cudaMallocAsync`)是"可捕获"的** —— 家族随手分配不会当场炸,
而是**静默地把分配搬进图**;而 `memcpyAsync` 的源若是 run 内现建的栈上/可变缓冲,
回放期**读已释放内存** ⇒ 静默错值(与 FI plan 的内嵌 H2D 同族)。

**三条正路**(窗内材料的合法来源):

| # | 正路 | 现例 |
|---|---|---|
| ① | **装配期分配** | FI 的 `int_ws`/`float_ws`、XQA consts、cuBLAS 工作区 |
| ② | **不变盒**(`InvariantBox`:构造即冻结、生命周期 ≥ 图) | `gdn_scalar` 的 `soff`(注释原文:"捕获窗 async 节点,回放恒读盒") |
| ③ | **引擎真块**(走 `Command::Alloc` ⇒ 捕获期落 slab、进账房) | marlin `ws` / XQA `scratch`/`semaphores` |

**已落(本轮)**:

| 面 | 内容 |
|---|---|
| 判定 | `state.rs::FamAct` + `fam_window_verdict`(**纯函数 + 单测**):不变盒恒放行;走第四条路的动作 **观察档放行(计数 + WARN)/ `OWL_CAP_STRICT=1` 结构化拒** |
| 入口 | `DeviceRes::alloc` 与 `upload` 的 cuda 实现窗内计数(`srv.count.fam_alloc_in_window` / `fam_upload_in_window`)+ WARN(`fam-alloc-in-window` / `fam-upload-in-window`) |
| 正路② 显式化 | 新 `DeviceRes::upload_invariant(dst, &InvariantBox)`(缺省转发 `upload`);`gdn_scalar` 的 soff 上传改走它 |
| 哨兵③ 修正 | `record_capture`(A1.6)**从空实现落地**:家族/外来核不走 `note_launch` ⇒ 原哨兵把**纯家族窗误判为空窗**(2026-10-03 `graph_dispatch_probe2` 踩过);现共用 `CaptureNodes` 登记表(总数 + 内核直方图),`graph_end`/`graph_destroy` 打印 top-5/种类数 ⇒ **"图里烘了什么"首次有账** |

**真机证据**(GPU0/3080;`gpu_server` 步 12,用 **GDN chunked 家族真身**):

- **观察档(缺省)**:窗内家族调用**放行**、`graph_end` 成功 —— 证实该家族**确实**在 run 内
  `res.alloc`(scratch + `(t,nt)` 键 meta 表)且 `res.upload` 的源是**栈上 `Vec`**
  (测试**故意不回放**该图:回放必错,这正是本条守门的东西);
- **`OWL_CAP_STRICT=1`**:同一调用**结构化拒**(拒语含"捕获窗内不得现建 scratch")⇒ 窗已脏 ⇒
  `graph_abort` 收窗成功,且**窗外**同一家族调用照常(eager 路零回归)。

**⇒ 硬前置件 P-6(M2b 必需;比 P-5 更要命)** —— **P-6a 已落(2026-10-15)**:

| 子件 | 状态 |
|---|---|
| **P-6a** `gdn_chunked` 的 meta 表 H2D **源**改冻结盒 | ✅ 已落(被 P-6b 取代 —— 表已不在 runtime 内,冻结盒随 meta 缓存一并删除) |
| **P-6b** `gdn_chunked` 三张 FLA 表(cu/coff/idx)**槽化 + 进 call 面** | ✅ **已落(2026-10-15,worktree `feat/prefill-graph-v2`)**:表由**引擎按 `t_real`** 写持久槽(`StatePool::gdn_tables`,每 chunk `write_block_f32` 原位写),runtime 只切视图读;**假行不在序列里**(律①成立)、图可烘(地址恒定)、runtime 少一条窗内 alloc 路径。声明面 +1 块(`tables` U32x1)+1 标量(`nt`);A17 锁表/F3 覆盖门/家族测试同步。红门探针 `gdn_bucket_pad_probe`:终态 Δ 3.578e-2 → **0.000e0** |
| **P-6b-②** `gdn_scalar` 的 `soff`(= `[0,T]` 冻结盒)**同族问题** | ⬜ 未落(同一配方:表进 call 面 / 引擎槽)。该臂是 `gdn_chunked=false` 的回落档与 `nv≠48` 形状,生产 27B 走 chunked ⇒ 优先级后置 |

原设计文本(节选):

1. **`gdn_chunked` 的 meta 表**(cu/coff/idx,键 `(t,nt)`,内容随 t 变)⇒ 改**设备槽**
   (`ctx.seq_cu` 注入真长度 + idx/coff 预放进槽);
2. **`gdn_scalar` 的 soff** 已是冻盒 ✓,但**值 `[0, T]` 被烘死** ⇒ 桶 padding 下会按
   `t_pad` 推进递推 ⇒ 同样要槽化;
3. 两族的**容量扩容**(`ensure_one` / `o_f32` 的 `t > t_cap`)⇒ 改**装配期按最大 t 定尺**
   (图内形状固定 ⇒ 运行时不需要扩容)。

**⚠️ 为什么这条比 P-5 更硬**:P-5(尾部 CTA)错了只是**白跑**;P-6 错了是
**假行推进 GDN 递推态**(律① 在 `conv_fwd` 上成立、在 **delta-rule 两臂上不成立** ——
二轮稿里"GDN 面已是 varlen 驱动"只对 conv 成立),属于**静默错值**,且污染会
跨 chunk 累积。

### 4.3f **P-5 已落 + M2a 判决**(2026-10-15;整段单图的三条假设全部实测成立)

**(a) P-5(尾部 CTA 掩码)落地**

| 面 | 内容 |
|---|---|
| 图档 plan | `FI_CUDA_GRAPH=1`(env,与 C++ `owl_fi_cuda_graph()` 同源同读法)⇒ plan 走 `enable_cuda_graph=true`(最坏情形定瓦 + `padded_batch_size = max(2·SM/hkv, tiles)`) |
| 掩码 | `server/flashinfer.rs` 按 **plan15** 算 `tiles = ceil_div(total_rows·gqa, cta_tile_q)`(`batch=1`;非 1 结构化拒),在**窗外**分配 `fi.mask` 并上传 `[i < tiles]`;**与 plan 同车缓存**(单条 plan ⇒ 单条 mask;共用一条缓冲会被一次换形 eager 调用改写 ⇒ 旧图回放掩码失配 = 真行漏算) |
| 核侧 | 两个 run 入口 ABI +1 参 `block_valid_mask`;`params.block_valid_mask = (bool*)mask`(dump 需 `FI_NO_MASK=1` 取证档) |
| 调试 | `[owl-fi][mask] 图档 plan:T=.. padded=.. tiles=.. ⇒ 掩 N 个尾部 CTA`(FI_DEBUG=1) |

**(b) M2a 判决(真机 3080/CUDA-2;`fi_graph_replay_probe.rs`)**

| # | 实验 | 实测 |
|---|---|---|
| ① | 表件 A(ctx=512)捕获 ⇒ **回放前换表件到 ctx=2048** | 回放(A)≡eager(A) **+0.00e0(逐位)**;回放(B)≡eager(B) **+0.00e0(逐位)**;B≠A 8.90e2(证 ctx 真生效)⇒ **F-5 成立** |
| ② | 桶 512 行、真行 64、假行填 **NaN**、`q_cu=[0,512]` 常量 | 真行 **finite** 且 ≡ eager(t=64) **+0.00e0** ⇒ **假行不泄 NaN/不污染真行**(设计 §4.2 律①) |
| ③ | t=128 图档 ⇒ `padded=34 > tiles=12` ⇒ 掩 22 个尾部 CTA;再 `FI_NO_MASK=1` 回放 | 掩码回放 ≡ eager **+0.00e0**;关掩码**真行仍 ≡ eager +0.00e0** ⇒ **F-8 成立**(尾部 CTA 不写坏真行);耗时两档都在 ~30µs = **回放开销地板**(单发同步口径,见探针头注) |

⇒ **M2b 的前提成立:一个桶一张图、图内含注意力、回放按设备表走** —— 不再需要 piecewise(M2c 仅作后备)。

**(c) 三条"不这么写就必失败"的姿态(实测踩过;生产侧早有护栏)**

| # | 坑 | 现象 | 正解 |
|---|---|---|---|
| 1 | **legacy 默认流不可捕获** | `cudarc::default_stream()` = stream 0 ⇒ `begin_capture` 报 `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`(措辞误导) | 自建流(生产 = 三固定流) |
| 2 | **cudarc event-tracking** | 窗内那次 FI run 返回 **901 `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`**(窗外同一调用正常)—— 因为 `device_ptr()` 在块操作上插 event | `ctx.disable_event_tracking()`(= 生产 `GpuCtx::new` 的 G0 护栏) |
| 3 | **窗内 sync** | 同上 901 | 窗内只发不等(`execute`),收尾 `sync` 出窗 |

**M2b 剩余前置件**:P-1(plan 键去 ctx + `graph` 位已就位)、P-2 ✓、P-3(FI 三件进槽 —— 探针已按"换表件"验证语义)、P-5 ✓,**以及 P-6(§4.3e:GDN 两族 cu/coff/idx 槽化)**。

### 4.4 命令账(设计预算;M2b 整段图口径 + M2c 断点口径)

模型规模(27B):**64 层 = 48 GDN 层 + 16 全注意力层**(分支 §12.2 逐核账同源)。

**M2b(整段单图)= 首选口径**:

| 段 | 内容 | 每 chunk 命令数 |
|---|---|---:|
| 图 | **1 条** `graph_launch`(整 chunk:embed → 64 层(KV 写 + FI 注意力在图内)→ final norm → 全行 hidden + taps) | **1** |
| 输入槽写 | ids / pos / kv_slots / kv_lens / gdn_slot / seq_cu + FI 三件(indices/indptr/last_len) | ~9 |
| 出图尾 | 末行窄切 + lm_head + argmax(仅末 chunk;§4.2 律③) | ~3 |
| 草稿面(DFlash2 encode) | M2b 仍走 eager 原路(进图是 M4) | ~40 |

⇒ 逐 chunk **~1465 → ~13 条(≈110×)** ⇒ §1.2 那 110 ms 固定段基本抹平(**P1 判据的强版**)。
注意:FI 的 plan/表上传**只在捕获期发生一次**,回放期零 host 参与(这是整段图相对断点图的关键优势)。

**M2c(断点 / piecewise)= 后备口径**:

| 段 | 内容 | 每 chunk 命令数 |
|---|---|---:|
| 图段 ×64 | 每层"非注意力部分"(norm / qkv-proj / o-proj / MLP / GDN 六核 / rope / conv 更新…) | **64**(= 回放驱动:每段 1 条 `graph_launch`) |
| 断点 ×16 | KV 写(`reshape_and_cache` / `qknorm_rope_kv_insert`)+ FI 注意力(每层 2 条,16 个注意力层) | **~32**(KV 写 16 + FI 16;FI 家族内部另有 plan/表上传,见 R5) |
| 输入槽写 | ids/pos/slots/lens/seq_cu/gdn_slot/(XQA 或 bt 页链) 每 chunk 各 1 次 `write_block` | ~7 |
| 出图尾 | 末行窄切 + lm_head + argmax(仅末 chunk) | ~3 |
| 其它 | alloc/free(槽/断点缓冲,走 arena 后基本归零) | ~5 |

⇒ 逐 chunk **~1465 → ~110 条(≈13×)**;其中**回放驱动(64 条)是新引入的部分**,
其余是"本来就该有的 eager 断点 + 槽写"。
⇒ **命令税的消除量 ≈ (1465−110)/1465 ≈ 92%**,对应 §1.2 的固定段 ~110 ms ⇒ **P1 判据成立**。
⚠️ 段粒度取舍:一层一段(64 段)时,KV 写与注意力必须在同一断点内连续执行(它们共享
`k/v` 张量);若把一层拆两段(前后各一),段数 129、回放驱动翻倍而断点更细 —— M2 用一层一段起步。

### 4.5 静止点律(捕获时机)

prefill **有状态副作用**(KV 池写 + GDN 递推)⇒ `GraphPlan::plan(capture=true)` 的 warmup
会真跑一遍 eager ⇒ **严禁 turn 中途捕获**。

**落地**:捕获只在 **boot**(`pool` 清零 + 页分配就绪后)对每个桶做一次;若某桶捕获失败 ⇒
该桶永久 `EagerFallback`(warn 一次,不重试);turn 内只回放。
(分支把"惰性按需捕获"标为 v2 事,需给 `GraphPlanDesc` 加 `warmup: false` + 固定 slab 档 —— 本设计不做。)

### 4.6 两条雷的守门(M2b/M2c 都要落)

| 雷 | 守门 |
|---|---|
| **双执行**(`EagerFallback` 态 `step()` 再跑 eager) | `captured` 门守死:非 `Captured` 态 `step()` 返回 `Ok(None)`;外层 eager 路**不再**跑(调用点按返回值分支,单一路径执行)。分支 S1 实测已踩过这一条(GDN 态双推进) |
| **FI plan 进图** | 已在 `server/flashinfer.rs` 立硬门(捕获窗内 plan 未命中 ⇒ 结构化拒 ⇒ 该桶放弃捕获)。**M2b 下 FI 的核在图内、但 plan 必须在窗外** ⇒ 该硬门就是 M2b 的安全带(配合 P-1 的桶级 plan + 键去 ctx) |
| **int_ws 串台**(多键共存新雷;§4.7a) | **plan 自携**:每键一份不变盒快照 + 回放前 `upload_invariant` 刷 int_ws(窗内那次烘进图)⇒ 图不依赖"最后一个 plan"。取证档 `OWL_FI_NO_PLAN_RESTORE=1` 反证机制承重 |

---

## 4.7 **M2b 落地记(2026-10-15;施工实录与新增前置件)**

### 4.7a 新增硬前置件:**FI plan 自携**(多桶共存的地基)

**病**(落地时才发现,设计稿没写):`int_ws` 是 FI 家族**唯一共享区** —— 每次
`owl_fi_prefill_plan` 都把调度表 `cudaMemcpyAsync` 进同一个 int_ws,而图里烘的是
`int_ws + offset` 指针 ⇒ **"最后一个算过的 plan"决定所有图读到什么**。多桶(geobuckets)
在 boot 逐个捕获,每个桶捕获前都要自己的 plan 在 int_ws 里;运行期 eager 换形
(尾块长度不同 = 新键)、草稿面(Draft Encode 也走 FI prefill 家族)都会再刷一次。

**修**(`crates/kernels/src/server/flashinfer.rs`):

| 面 | 内容 |
|---|---|
| 条目表 | `plans: HashMap<PlanKey, PlanEntry{plan15, staged: InvariantBox, mask, tiles, born, graph_owned}>`;单条缓存退役 |
| 快照 | plan 调用的 H2D 源是**共享可变** staging(换形即重写)⇒ 窗外把**有效前缀**(`need_int` 字节)考进不变盒。`need_int` 由 C++ wrapper 新增出参 `out_int_bytes` 给出(= `PrefillPlanImpl<true>` 里那次 memcpy 的尺寸,同源) |
| 激活 | `active: Option<PlanKey>`;`active != key` ⇒ `res.upload_invariant(int_ws, &staged)`。**窗内那次 = 图里一个 memcpy 节点** ⇒ **图自携 plan**(回放先把表刷回自己的) |
| 窗上升沿 | `capturing() && !last_capturing` ⇒ `active = None`(强制本窗首个 FI 调用重激活)—— 否则 warmup 之后 active 已等于本键,捕获期就不会烘那个节点 |
| 纪律 | plan **未命中且在窗内**仍结构化拒(C++ plan 自身那个 H2D 的源是共享 staging,烘了必错);`graph_owned` 标记挡住驱逐(图引用掩码/快照) |

**证据**(`fi_graph_replay_probe` ④;红绿两档):

```text
绿(机制开):B 形状把 int_ws 刷掉(首个差异字节 436)⇒ 回放 A 后 int_ws **复位=true**;输出 |Δ| = 0.000e0
红(OWL_FI_NO_PLAN_RESTORE=1):同一实验 ⇒ int_ws **复位=false**(断言当场红)⇒ 机制是承重件
```

> 诚实注记:本档(split 关 + batch=1)下 plan 数组的**内容**几乎形状无关
> (`request_indices` 恒 0、`qo_tile_indices` 恒等),所以"中毒"时输出**可能**恰好相等
> —— 那正是设计 §3.2 明列不做的"靠陈旧表项恰好无害"。结构判据(直接读 int_ws)
> 才是权威;内容巧合不能当不变量。

### 4.7b 桶几何下的**设备表口径**(P-3 三轮更正 + GDN P-6b 收口)

| 面 | 口径 |
|---|---|
| 张量形状 | 布局 `t_pad`(embed/GEMM/KV 写/FI q 行数都按布局) |
| FI `q_cu`/`indices`/`indptr`/`last_len` | **每 chunk 写真值**(`qo_len`/`kv_len` 设备同口径 ⇒ 因果窗口 = eager 逐字同式) |
| FI mask | 桶级静态(P-5;内容 `i < tiles_plan`) |
| `seq_cu` | `[0, t_real]`(律①:varlen 核按它划行区间) |
| GDN 序列表 | `cu=[0,t_real]` + **布局块数** `nt_layout=ceil(t_pad/BT)`(coff/idx 铺满布局)—— 图内 `nt` 是标量,必须烘死;尾部越界 chunk 的 CTA **读写都被 T 夹回**(探针 `gdn_bucket_pad_probe::gdn_bucket_pad_graph_posture_nt_layout_is_inert` 实测:真行输出/终态逐位 Δ=0) |
| 假行 KV 槽 | dump 页(`StatePool.nb_kv = nb + 1`;不在页链 ⇒ 注意力结构上读不到;律②) |
| 出图尾 | 末行窄切 + lm_head + argmax **图外**(律③;偏移是烘不得的 host 标量) |

### 4.7c 落地件清单

| 面 | 文件 |
|---|---|
| 桶族/捕获/回放 | `crates/engine/src/prefill_plan.rs`(ladder/pick_bucket/`gdn_host_slot_arm` 纯函数 + `BucketGraph`/`PrefillBuckets`) |
| 分派与出图尾 | `crates/engine/src/exec/prefill_chunk.rs`(`bucket_prefill_chunk` + `bucket_tail`;未命中/不可用 ⇒ eager 原路) |
| 桶表构造 | `ChunkTables::build_bucket`(真行真值 + 假行 dump;`fi_tables()` 三件套单源) |
| 槽面 | `GraphPlan` 的 `InputSlot` 增 `dtype`/`shape`(`rank1()`/`u32()` 构造;FI 表件槽 = U32 位型) |
| 池 | `PoolPlan.prefill_dump_page` + `StatePool.{nb_kv,dump_page,dump_slot_base,k_shape_kv}` |
| 装配 | `GraphFleet::assemble` ④ 段(降序捕获;off/no_graph/xqa_prefill/非 FI 四道响亮降级门)+ `Engine::on` 撤回"配了没实现就拒" |
| 开关 | `[graph] prefill_bucket = off|t512|geobuckets`(off = 今天行为) |

### 4.7f **收尾小改:B1 诊断让位 / B2 padding 护栏 / B4 命中率统计**(2026-10-15 深夜)

| # | 内容 | 判据/实测 |
|---|---|---|
| **B1** | **诊断档让位**:`pf_bisect` / `pf_stages` / `dflash_probe` 在场 ⇒ 桶**全部让位 eager** + 一次 warn(此前桶命中会**静默绕过**这些诊断——它们的实现只在 eager 路) | 优先级"诊断 > 图";不静默 |
| **B2** | **padding 护栏**:`pad_rows = t_pad − t_real > 32` ⇒ 该 chunk 走 eager。阈值来自实测:pad **29 行仍净赢 −9.4ms**、pad **42 行净输 +5.4ms**、pad **64 行净输 +34ms**(@27B,0.78ms/行;图的固定节余 ≈20–30ms)⇒ 取"一个最小桶宽" | **实测复验**(86/192 两点,同卡同时段):86: 110.4→**99.9/102.6ms**(回到基线甚至更优);192: 201.4→**161.6/161.8ms**(−40ms,白算被拨回)。其余点不变 |
| **B4** | **命中率/白算统计**(`/debug/metrics`):`prefill.bucket.hit[.t<t_pad>]` / `.bypass.pad` / `.pad_rows[_skipped]` / `prefill.eager.{chunks,tokens}` | 实测逐项对上:t128×6 / t256×2 / t512×3 / pad 跳过 4(跳过 212 行 = 42×2+64×2)/ eager 556 tok = (86+192)×2 |
| **B7** | 观测面对齐:桶路与 eager 末块都记 `prefill.chunk`(此前桶路只有 `prefill.graph` = 发射成本、eager 末块缺行) | 图表按 chunk 数口径不再丢行 |

⇒ `t512` 单桶档的自伤(小 chunk 进 512 桶 = 15× GEMM)被 B2 结构性封死;

### 4.7e **显存归因 + slab arena 修复**(2026-10-15 深夜;用户令"先归因"→"开始")

**工具**(入树,常驻诊断):影子探针加**双口径**(容量峰值 = bump 高水位 / 活跃峰值 =
同时在用字节)与**按名字汇总**;求值器三个分配臂回填核名(`probe_name`);每张图
boot 打一行 `slab 峰值归因:…`(两个探针单测,带串行锁防全局态互噬)。

**归因结论(27B 生产档,t512 桶)**:

| 量 | 修前 | 说明 |
|---|---:|---|
| 容量定尺 | **193 MiB** | bump 高水位 |
| 真·同时活跃 | **81 MiB** | `marlin_w4a16 34MB`(=MLP gate/up 输出)+ `add 30MB` + `silu_and_mul 17MB` |
| 空洞 | **112 MiB(58%)** | 大/小块交替的分配序列把 bump 顶高,碎片洞填不回大请求 |
| 分配总量 | 8.1 GB | marlin 46% / silu_and_mul 13% / narrow 10% / gdn_conv_fwd 6% |

⇒ **第一杠杆是分配器,不是搬算子(piecewise)**。

**修法①**:`SlabArena` 从"尺寸类 best-fit + 从不合并 + 从不退顶"改为
**空闲块偏移有序表(BTreeMap)+ 相邻合并(coalescing)+ 贴顶回收(trim_top)**
⇒ 层间同形状序列(64 层 × 同批张量)复用同一段地址。

**修法②**:decode/verify/fold/propose 四族 `reclaim_capture: false → true`
(捕获归约口径与探针口径对齐;此前 warmup 用 scoped 而捕获用 plain —— 口径不一致
只是靠"这些图峰值 == 总和"侥幸不炸)。

**实测(同配置 boot;每桶都 Captured)**:

| 桶 | 修前容量 | 修后容量 | 修后活跃 | 修后空洞 |
|---|---:|---:|---:|---:|
| t512 | 193 MiB | **91 MiB** | 81 MiB | 10 MiB(11%) |
| t256 | 96 | **45** | 40.5 | 5.0 |
| t128 | 48 | **22** | 20.2 | 2.5 |
| t64 | 24 | **11** | 10.1 | 1.2 |
| t32 | 12 | **5** | 5.1 | 0.6 |
| **Σ** | **373** | **174 MiB**(定尺 ≈200) | — | — |

功能/性能零回归:烟测 128→110ms / 512→386ms、prefill 1334 t/s(与修前逐点同);
spec 档回答连贯 ⇒ verify/fold/propose 三族 scoped 捕获健康。单测 8/8、engine lib 37/37。

**图显存总账**:640 MiB → **364 MiB**(−276 MiB ≈ 5.4k token 的 fp8 KV)。
⚠️ verify 图的 133 MiB **不降** —— 它的 384 个逐层记录都是**已登记输出**,
活跃集 == 总量(要降得改"少登记输出",另案)。

**下一步候选**(按归因排序):③ 输出面瘦身(taps 仅 spec-on 产:活跃 81→~50MB)
④ 元素级 in-place(add/silu/cast/rmsnorm)⑤ S-2 弹性常驻。

### 4.7d 验收现状(①—⑥ 对表)

| # | 判据 | 现状 |
|---|---|---|
| ① | 图内一致门(逐位) | ✅ 原子层:`fi_graph_replay_probe`(M2a ①②③;抽屉件换 ctx/假行 NaN/尾掩码全逐位) |
| ② | 跨档门(贪心逐 token) | ✅ 引擎级:`gpu_prefill_bucket_matches_eager`(0.8B;geobuckets 全捕获,桶真回放(t128×1 + t64×1),输出与 eager **逐字相同**) |
| ③ | 回放命令数 ≤ 60/chunk | ✅ 结构上 = 1 条 `graph_launch` + 3 条批量写(槽 / FI 表 / GDN 表)+ 出图尾(设计预算 ~13);真机发射成本 `prefill.graph` = **0.201ms avg / 0.17 p50 / 1.16 p99**(281 chunk) |
| ④ | `t=512` 墙时不劣化 ±1% | ✅ **已量(§4.7g)**:TTFT −4.1…−203.3ms(−1.1…−3.2%)、prefill **+1.2…+2.9%**、per-chunk p50 −10.6ms ⇒ 不劣化成立且**净正** |
| ⑤ | 桶不命中 ⇒ eager | ✅(`pick_bucket` 向上取整 + `usable`(未捕获/槽不匹配)双守门;单一路径执行;B2 护栏再叠一层) |
| ⑥ | `t ≤ 128` 墙时 −60…−100 ms | ⬜→**作废(§4.7g 重定档)**:实测 pad=0 点 −2.3…−4.8ms、pad=29 ≈0、**pad=32 反输 +12.9…+15.0ms**;新线 = pad ≤ 8 走图时 TTFT 中位 ≥ −2ms |
| — | 回归 | ✅ engine 36/36(CPU);kernels 侧 M2a/图姿态/探针全绿;0.8B e2e 共享组 4/5 通过(第 5 个 `gpu_longctx_4k_prefix_qa` 是**基线既有红**:prompt 4057 + 生成 16 超预算 4096,stash 到基线复现同样失败) |

> ⬜ 两项是**性能验收**,需要真机墙时口径(设计 §1.2 的固定段)与 27B 档;
> **2026-10-16 凌晨已补量(见 §4.7g)** —— 六项全绿,其中 ⑥ 原线作废、重定为 pad 分档线。

### 4.7g **B3 墙时验收:三臂实测**(2026-10-16 凌晨;空闲 3090 Ti = CUDA ordinal 1;spec-dflash 生产档)

**三臂**(同配置 = `testdata/server/agent-27b.toml` 派生:device=1 / bind 8140;同卡同时段;
每臂同序列 = 桶扫 9 点 ×3 轮 + std×2 + 小桶×2):

| 臂 | 二进制 | `prefill_bucket` |
|---|---|---|
| **A** | 主树 HEAD `60515fd` **现场重编**(`/tmp/b3/owl-server-A`,md5 `dde9cafa`) | 无(prefill 图不存在) |
| **Boff** | worktree 二进制(含 M2b 代码;B6 版 `/tmp/owl-server-m2b-b6`) | `off`(漂移对照) |
| **Bgeo** | 同一二进制 | `geobuckets` |

⚠️ **口径修正(重要)**:上一轮 A/B 的“基线” = 归档 `/tmp/owl-server-baseline`(10-15 **08:43** 构建,早于当晚 20:04 的 refactor 合并),且 A 臂 = 当时**正在服务 pi 自身推理流量**的生产进程 ⇒ 旧结论 `+4.5…+9.6%` 混入了 build 漂移与自身流量。本轮 A 臂 = 主树 HEAD 独立进程,Boff 臂 兼作漂移对照(实测漂移很小:短点 +1.3ms、@8192 −36.8ms)。

**表 1:std 五点三轮(TTFT ms / prefill t·s⁻¹;std1+std2 六点中位)**

| 长度 | A TTFT | Boff | Bgeo | Δ(Boff→Bgeo) | A prefill | Boff | Bgeo | Δ(Boff→Bgeo) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 370.6 | 370.6 | 366.5 | −4.1ms(−1.1%) | 1389 | 1388 | 1404 | **+1.2%** |
| 1024 | 748.2 | 746.4 | 729.8 | −18.4(−2.5%) | 1372 | 1376 | 1407 | **+2.3%** |
| 2048 | 1523.9 | 1513.3 | 1478.1 | −45.8(−3.0%) | 1346 | 1355 | 1387 | **+2.4%** |
| 4096 | 3093.1 | 3074.0 | 2989.2 | −103.9(−3.4%) | 1325 | 1333 | 1371 | **+2.9%** |
| 8192 | 6335.8 | 6299.0 | 6132.6 | **−203.3**(−3.2%) | 1293 | 1301 | 1336 | **+2.7%** |

**表 2:小 chunk(单 chunk 提示;三臂 TTFT 中位;桶/垫按**实到** token 算)**

| 请求 | 实到 | 桶(垫行) | 路径 | A | Boff | Bgeo | Δ(Boff→Bgeo) |
|---|---:|---|---|---:|---:|---:|---:|
| 32 | 86 | t128(42) | eager(B2 让位) | 99.1 | 101.7 | 100.6 | −1.0 |
| 64/96 | 99 | t128(29) | 图 | 104.3–104.5 | 103.8–104.3 | 102.6–104.1 | −0.2…−1.2 |
| 128 | 128 | t128(0) | 图 | 109.9 | 112.5 | 107.7 | **−4.8** |
| 160 | 160 | t256(96) | eager | 137.4 | 139.8 | 141.1 | +1.2 |
| 192 | 192 | t256(64) | eager | 153.7 | 154.9 | 156.5 | +1.5 |
| 224 | 224 | t256(32) | 图 | 181.2 | 182.4 | 195.3 | **+12.9** |
| 256 | 256 | t256(0) | 图 | 196.6 | 198.0 | 195.5 | −2.5 |
| 480 | 480 | t512(32) | 图 | 351.8 | 351.6 | 366.6 | **+15.0** |
| 512 | 512 | t512(0) | 图 | 369.3 | 369.9 | 367.6 | −2.3 |

**表 3:服务端 chunk 墙时(同二进制、同序列 **281 chunk** 累计对拍 —— 口径同,唯一干净的对拍面)**

| 量 | Boff | Bgeo | Δ |
|---|---:|---:|---:|
| 全臂均值(281 chunk) | 313.6ms | 306.2ms | **−7.4ms(−2.4%)** |
| 全臂 p50 | 370.5 | 359.9 | **−10.6ms** |
| std1 / std2 相位均值(长 chunk 为主) | 320.8 / 327.5 | 311.9 / 319.5 | −8.9 / −8.0 |
| `prefill.graph`(图发射:槽写 + graph_launch) | — | 0.201ms avg(0.17 p50 / 1.16 p99) | 达标(预算 ≤60 命令/chunk) |
| 桶命中 | — | **260/281 = 92.5%**(std 相内 100%) | t512×211 / t128×24 / t64×13 / t256×12 |
| `bypass.pad` / `eager.tokens` | — | 21 chunk / 2982 tok(均来自刻意坏垫点) | pad 行:垫 975 / 跳过 1242 |

**结论与新接受线(设计 §3.1 目标表按实测重定)**

| # | 设计线 | 实测 | **新线** |
|---|---|---|---|
| ④ | t=512 不劣化 ±1% | TTFT −1.1…−3.2%;prefill **+1.2…+2.9%**;per-chunk p50 −10.6ms | **及格 = 不劣化;达标 = prefill ≥ +1% 且 TTFT@8192 ≤ −100ms;冲刺 = prefill ≥ +3% 或 TTFT@8192 ≤ −200ms** ⇒ 本轮**达标**(@8192 逼近冲刺) |
| ⑥ | t ≤ 128 墙时 −60…−100ms | pad=0 点 −2.3…−4.8ms;pad=29 ≈ 0;**pad=32 反输 +12.9…+15.0ms** | **原线作废**(它基于旧 build 的 ~110ms 固定段);新线 = **pad ≤ 8 走图时「TTFT 中位 ≥ −2ms 且不劣化」** |

**B2 阈值定谳(应下调)**:图的每 chunk 固定节余 H ≈ 2.3–4.8ms(短 ctx),白算成本 = pad × s,
s ≈ **0.28ms/行**(t128)/ **0.67**(t256)/ **0.57**(t512)(由 eager 臂同桶内长度差直接量得)⇒
盈亏平衡 pad ≈ **4…17 行**,现行 32 行**太大**(pad 32 必输)。建议
`BUCKET_PAD_MAX_ROWS`(单档 32 行)→ **按桶定阈 `bucket_pad_max_rows()` {t128:12, t256:4, t512:4, t32/t64:16}**
(转正时已改,见 §4.7g);H 随 ctx 增长(长提示 t512 chunk −8…−9ms),
但长提示 chunk 恒 pad=0 ⇒ **收紧护栏不影响 std 收益**。

**异常与风险(未结,另案)**

1. **A 臂(主树 HEAD 新编二进制)在 small2 相位中途静默死亡**(03:05:24;日志末行 `[sched] n=4624`,
   无 panic、无 OOM 记录、显存立即释放);同序列 Boff/Bgeo 两臂全绿。⇒ 主树 HEAD 有一次**未解释崩溃**,
   已挂账;**当前生产 8137 仍跑旧二进制**(见 2)。
2. 本轮进行中发现**生产进程(旧二进制)已静默死亡**(无日志尾),已用原二进制原样重启;
   ⚠️ 即**生产当前 ≠ 主树 HEAD**,推动合并/转正时需一并处理。
3. 噪声界:同臂 std1→std2 的 TTFT 差在 A/Bgeo 达 +2.3…+44.5ms(@8192,疑似热漂移),Boff 平坦(+1.2ms)
   ⇒ 表 1 取六点中位,结论按 **Boff↔Bgeo 同二进制对拍**读。
4. 启动显存:本卡 boot 瞬时峰值 ~23.7G/24.5G(稳态 21.9G = 与生产档同);Bgeo 终态 free 1602MB vs
   Boff 1231MB(桶常驻 ≈ +370MB)⇒ 换 ctx 容量时需计此账(§4.7e 后 Σ174MiB + arena 余量)。

**转正版复测**(2026-10-16 03:47;分支提交 `eb9462d` + merge `60515fd`(=`30ae324`,冲突仅
engine.rs / state.rs 两处“两边各加一字段” ⇒ 取并集)+ 护栏改**按桶定阈**)

| 面 | Moff | Mgeo | Δ |
|---|---|---|---|
| std prefill(512…8192) | 1389 / 1372 / 1348 / 1325 / 1294 | **1406 / 1411 / 1388 / 1357 / 1334** | **+1.2 / +2.8 / +3.0 / +2.4 / +3.1 %** |
| std TTFT | 370.4 / 748.3 / 1521.4 / 3093.4 / 6332.3 | 365.8 / 727.9 / 1476.9 / 3019.5 / **6141.7** | −4.6 / −20.4 / −44.5 / −73.9 / −190.6 ms |
| 小桶 pad=0(128/256/512) | 111.5 / 198.5 / 370.8 | 105.5 / 194.2 / 365.9 | **−6.0 / −4.3 / −4.9 ms** |
| 小桶 pad=32(224/480) | 182.1 / 352.3 | 184.0 / 353.3 | **+1.9 / +1.0 ms**(护栏改后**反输消失**;前 +12.9 / +15.0) |
| 小桶 pad 29/42/64/96 | — | 全部让位 eager | ≈ 0(与 Moff 同路) |
| 计数(全臂 281 chunk) | — | hit **233(82.9%)** / bypass.pad 48 / eager.tokens 9459 | t128 24→9、t256 12→9、t512 211→202(pad ≤32 的净输点被主动让位) |

⇒ **复测判定**:std 增益完整保持(Moff ≈ A ⇒ 合并无回归);小桶反输清零;
**② 命中率线须分相位读**(真实长提示 std 相 100%;对抗性桶扫面 82.9% 是收紧护栏后的**设计后果**,非退化)。
门:engine lib **40/40**(含 `gpu_prefill_bucket_matches_eager` 桶≡eager 逐位门,GPU1)。

> **产物**:`/tmp/b3/`(三臂 × 13 相位 × json/csv/服务端 metrics 快照 + 启动峰值采样 + 日志)。
> 复现命令 = `bash /tmp/b3/run_arm.sh <tag> <bin> <cfg> sweep` + `python3 /tmp/b3/analyze.py`。

---

## 5. 分阶段落地(每阶段独立可回滚)

| 阶段 | 内容 | 判据 | 量级 |
|---|---|---|---|
| **M0** | **命令税对照实验(先量后建)**:实现"命令批/零回执"(P1‴ 的一条子集:同一条 `Command` 携带 N 条 launch,或 free/alloc 合批)并量 `t ≤ 128` 与 `t = 512` 两档 | 若 `t ≤ 128` 已拿到 ≥60% 的固定段收益 ⇒ **图不必做**;否则进 M1 | 2-3d |
| **M1** | ✅ **已落(2026-10-15)** —— slab 峰值定量 + 捕获窗放行 Free + abort/destroy/close 三把生命周期原语 + `warmup:false`/`slab_fixed` 组合。详情与验证矩阵见 **§4.3d** | 单桶捕获不再受"分配总和 9362 MiB"限制(峰值口径);窗内 Free/复用/abort/destroy 全部有真机门 | ✅ 完成 |
| **M2a** | ✅ **已做(2026-10-15)** —— P-5 掩码落地 + 三实验对拍(表件换 ctx / 假行 NaN / 尾掩码),全绿见 **§4.3f**;顺带钉死三条姿态坑(默认流/event-tracking/窗内 sync) | 三条假设 **F-5(plan 与 ctx 无关)/F-8(尾 CTA 不写坏真行)/律①(假行不污染)** 全部实测成立 | ✅ 完成 |
| **M2b** | ✅ **已落(2026-10-15)** —— 整段单图 + **桶族**(M2b 与 M3 的桶几何一次做完:`t_pad` 向上取整、dump 页、布局/语义 t 分离、末行出图)。前置件 P-1/P-2/P-3(三轮更正)/P-5/P-6b 全落,**新增硬前置件 = FI plan 自携**(§4.7a,多桶共存的地基);落地件清单 §4.7c | ① 图内逐位 ✅(M2a + 探针)② 跨档贪心逐 token ✅(`gpu_prefill_bucket_matches_eager`,0.8B 逐字相同)③ 命令数 ✅(发射 0.201ms/chunk)④/⑥ **已量并重定档**(§4.7g:④ 达标、⑥ 原线作废)⑤ 桶不命中 ⇒ eager ✅ | ✅ 完成(B3 墙时表见 §4.7g;待用户裁决:护栏 32→8) |
| **M2c** | **后备:断点原语**(§4.1b;M-A)+ `eval_ops_into` + 段列表回放 | 同 M2b(但 FI 留在图外) | 4-6d |
| **M3** | ✅ **桶族已并入 M2b**(2026-10-15):ladder {32,64,128,256,512} + dump 页 + 布局/语义 t 分离 + 末行出图 + `prefill_bucket` 三档开关;**B3 墙时表已量(§4.7g,2026-10-16)**:std prefill +1.2…+2.9% / TTFT −203ms@8192,桶命中 92.5%(std 相 100%) | ① 每桶恒等门:桶内逐位由 M2a 覆盖;每桶「图≡eager」由引擎门覆盖(本机 0.8B 五桶)② 命中率 ✅(生产档 281 chunk 实测)③ ④ 墙时 ✅(重定档) | ✅ 完成(剩:护栏 32→8 的裁决与落地) |
| **M4** | 收尾:PDL(`enable_pdl`)、**草稿面桶**(encode/extend;DFlash2)、XQA 尾块桶(小于 `fi_min_tokens` 的桶)、FI 槽容量/多会话面(multi-session batch>1 时 `q_cu` 不能再常量) | ① 恒等门 ② 首调税 = 0(§1.1 的 20.9 ms 空转)③ 缺省档仍关/按 flavor 开 | 3-5d |

**回滚开关(✅ 配置面已落,2026-10-15;worktree `feat/prefill-graph-v2`)**

```toml
[graph]
no_graph = false            # 总闸:一刀切关掉**全部**图捕获(decode/verify/fold/propose + prefill)
prefill_bucket = "off"      # prefill **专属**档位(缺省 off = 今天行为)
                            #   off        : 不建 prefill 图
                            #   t512       : 只建 512 桶(最小面:显存最小、命中面 = 满块)
                            #   geobuckets : 全桶族 {32,64,128,256,512}(尾块也吃图;显存 ≈ 374 MiB)
```

- **为什么分层而不是 bool**:回滚要能**逐级降**(`geobuckets → t512 → off`),不必在"全开/全关"之间跳;
- **与总闸的关系**:`no_graph = true` 优先级最高(连 prefill 图也不建);两者独立,可只关 prefill 而不动现役 decode 族;
- **配了就得生效(装配期护栏)**:`prefill_bucket ≠ off` 而实现尚未落地时,`Engine::on` **boot 最前**结构化拒启
  (拒语点名档位 + "施工中" + 可行动指引)—— 静默忽略配置是本仓明令禁止的隐瞒行为;M2b 落地时把该护栏
  换成真接线,**配置面零改动**(判据:单测 `prefill_bucket_non_off_is_rejected`);
- **运行时熔断(叠加在档位之上,M2b 落地时实现)**:某桶捕获失败 / 回放数值门不过 ⇒ **该桶永久降级 eager**
  (warn 一次、不重试),其余桶不受影响 —— 即"部分回滚"的动力学版本;
- **可观测**:boot 日志逐桶打建图结果;`CaptureNodes`(2026-10-15,A1.6 落地)在图捕获时记录
  "发射 N 发 / 内核 K 种 / 钉住哪些块",`graph_destroy` 时打印 ⇒ 运维一眼看到"哪张图在跑、烘了什么"。

---

## 6. 风险与逃生阀

| # | 风险 | 量级/判据 | 逃生阀 |
|---|---|---|---|
| R1 | **段过碎**回放驱动本身成为新税(97 条 `graph_launch`) | 每条 ~10-20 µs ⇒ 1-2 ms/chunk(可接受) | 段合并(一层一段 / 两层一段) |
| R2 | **slab 常驻显存**(几何级数:全 5 桶 ≈ **374 MiB** ≈ 0.69 × KV 池 540 MiB;非头注粗写的 ~1 GB) | 与 KV 池争 VRAM/上下文 | 见 **§4.3b**:S-1 裁桶(45 MiB)/ S-2 惰性+LRU / S-3 单图覆盖区间(255 MiB)/ S-4 压峰值 |
| R3 | 断点 eager 与图段的**时序**:回放期断点 eager 的 launch 与图段在同一流,序正确 | 恒等门会抓 | 断点前后插 `stream.sync`(调试档) |
| R4 | padding 假行污染(三律任一条破) | 数值门 + 逐 token 恒等 | 桶关 / 该桶降级 eager |
| R5 | FI 表件/plan 仍在窗外算 ⇒ 每 chunk 表上传税 | 表件已槽化(P-3)后仅剩 `write_block` 原位写(3 件,~百字节) | plan 桶级一次(P-1) |
| R6 | 收益被"设备绑死"吃掉 | 必须有 M0 的对照数据 | 只开小桶 |
| ~~R7a~~ | **FI 表件跨 ctx 回放**是否真按设备表走 | ✅ **已验(§4.3f ①)**:回放(B) ≡ eager(B) **逐位** | 关闭 |
| ~~R7b~~ | **尾部 CTA**(F-6)未掩码时会否写坏真行 | ✅ **已验(§4.3f ③)**:关掩码真行仍逐位一致(F-8);掩码已落(P-5) | 关闭(掩码仍留着——不是为了正确性,是为了卫生+白跑) |
| **R8** | **跨档数值门误判**(拿"逐位"当跨档判据 ⇒ 假红) | 口径分两层(§4.1c 推论 B) | 门按层写死;图内门仍逐位 |
| **R9** | 假行 NaN 经 **taps** 泄进草稿面(DFlash2 encode) | 因果方向是"未来行"⇒ 结构上不回流;M2a 步骤 3 顺带验 | 草稿面待 M4 进图时再复验 |
| **R10** | **窗内复用判据错**(把仍被图节点引用的块归还 arena 并复用)⇒ 回放读被覆盖内存 = 静默错值。依据 = "scoped 只归还不可达 + 捕获按求值序" | `gpu_capture_reclaim_matches_eager`(锚回收口径回放 ≡ eager)守门;窗内 Owned 死块另有防御性 no-op | 若判据被推翻 ⇒ `reclaim_capture:false`(回 plain 总和口径,只是捕不上大图) |

---

## 7. 与分支的分工(避免重复劳动)

| 分支产出 | 本设计如何用 |
|---|---|
| S0 `ChunkTables` | **已合入本仓**(本轮择取)⇒ M2 的输入槽就是它的值 |
| S1 `prefill_plan.rs`(214 行) | **骨架参考**(槽面 10 件 + 根 concat + `captured` 门 + `run(&tb)` 装填);二轮只改其中两处:FI 三件槽化(P-3)、`q_cu` 常量 + 参数行 padding |
| S1 单桶 / S1.5 slab 峰值 / S1.5-b arena | ✅ **已合入 M1**(2026-10-15):`SlabArena` + 影子探针 + 窗内 Free + `reclaim_capture` 按模式选口径;**S1.5-b 持久 arena 有意不合**(分支自测墙时零变化,见 §4.3d) |
| §12 设备侧定谳 | **本设计的地基**(收益定档,不做长 chunk 提速梦) |
| S2 断点 / S3 padding | **本设计的 M2c/M3**(分支没做;本设计给出了 I-1…I-4 与三律落地)。⚠️ 分支的"abort 缺口"已由本轮的 `GraphAbort/GraphDestroy/close` 补齐 |
| S4 桶矩阵 / S5 草稿面图 | M3/M4 的子集 |
| §13 N1 / §15 FI 门槛 / XQA 尾块臂 | **已合入**(与图无关的独立收益) |
| v3 核 / fa2v / bench | 不涉及 |

---

## 8. 待用户裁决(设计冻结前)

0. **"整段单图 vs piecewise"已定调(二轮确认)**:整段单图优先 —— 依据从 F-1…F-4(几何)
   升级为 **F-5…F-10**(plan 与 ctx 无关 / 发射几何 / mask / `total_num_rows` 不读);
   piecewise 降为 M2c 后备;M2a(0.5d、四步)是唯一必须的实验。
1. **先做 M0(命令批)还是直接进 M1/M2?** —— 本文档建议 M0 先量(2-3d),它与图**争同一笔收益**
   且不需要任何捕获机制;量完再决定是否投 8-13d 做 M1-M3。
   ⚠️ 二轮补:**图还买"小 chunk 的墙时"之外另两样** —— 零 H2D/零首调税、时序鲁棒性(A13 类面);
   若这两样对 agent 服务面有独立价值,则 M0 的量可以**并行**而非前置。
2. **桶集合**:{32,64,128,256,512} 是否全建(全建 ≈ 374 MiB 常驻,§4.3b),还是只建 {32,64,128}(小 chunk 价值面,≈ 45 MiB)?
   ⚠️ 二轮补:烦请给一次真实 chunk 长度分布(`[engine] prefill` 日志即可)—— 分支实测过 510/498
   (非 512),说明"满块"不一定正好等于 `prefill_chunk`;桶阶梯要不要加 {448,480} 取决于此。
3. **值不值得**:若 M0 拿到 ≥60% 固定段收益 ⇒ 图只留"短 prompt TTFT"与"鲁棒性"两个理由 —— 是否仍要投?
4. **新(二轮)**:接受"跨档门只到数值近似 + 贪心一致"吗?(若要逐位跨档,就得让 **eager 也走桶计划**
   —— 代价 = 小 chunk 也按桶白算 GEMM/FI;好处 = 图档/eager 档完全同 tiling、省掉一整套"双口径"验收)
5. **新(二轮·显存,§4.3b)**:走哪条?① **S-1 小桶裁桶**(45 MiB,零代码)② **S-1+S-2 惰性+LRU**
   (45–85 MiB,需 `warmup:false` + 固定 slab 档)③ **S-1+S-3 三档覆盖全区间**(255 MiB,零新机制)。
   注:显存换的是 KV 上下文容量(0.69 × 池 = ~11.5k token),而 agent 面实测 prompt 达 12.7k ⇒ 这笔账偏向"小桶"。

---

## 9. 解禁:`prefill_bucket` × `fi_plan_bake`(2026-10-19)—— **W5 之后重新可用**

### 9.1 为什么曾被锁死

2026-10-18 的 W5(FI 单图,见 `fi-single-graph.md`)把 `fi_plan_bake` 升成 FI 臂的
**强制前置**(防"占位 ctx 烘表"的静默错值),而它与桶路在 `validate` 里写成**互斥**:
- 单图档:host 是 plan 表唯一权威(`fi_plan_restore = false` + 每步 `fi_plan_sync`);
- 桶图档(M2b):plan **自携**进图(`fi_plan_restore = true`,捕获期烘 memcpy)。
两条承载没接上 ⇒ `prefill_bucket ≠ off` 直接拒启。生产档因此一直跑 eager prefill。

### 9.2 接线(本轮)

1. **解禁**:`validate` 删互斥检查,保留 `fi_plan_bake ⇒ !fi_plan_restore`;
2. **桶路接 sync**:`EngineKnobs += fi_plan_restore`;`BucketBoot`/`BucketGraph += needs_plan_sync`
   (= `!restore`,捕获期定);`exec::prefill_chunk` 在**每次桶回放前**调
   `fi_plan_sync(t = t_pad, ctx_total = base + t_real, …)`(与 decode 的 `pos+1`、
   verify 的 `ctx_total_fi` 同律);
3. `restore = true`(缺省)时整段跳过 ⇒ 旧行为逐字节不变。

### 9.3 ★ 关键坑(必须写进任何后续实现)

**`ctx_total` 必须是"本块真实 ctx"(= `base + t_real`),不能用桶的 `t_pad`。**
FI plan 的工作项按 **ctx 页数**枚举(`nb_pages = ctx_total / page`);用 `t_pad`(=512)
铸出的表只看前 16 页 ⇒ **长 ctx 读错 KV**。实测:4460 tok needle 对、**7860 tok 答错**
(输出 `2`,eager 为 `55021`),且与前缀缓存无关(隔离过)。修后
needle 1.6k/4.9k/7.9k/9.3k 全对。

> 附带的"假快":错 plan 少枚举工作项 = 少干活 ⇒ @8192 读 1386~1399;修后 1358。

### 9.4 实测(单 3090 Ti;std 五点三轮;同二进制;唯一差别 `prefill_bucket`)

| ctx | eager | geobuckets | Δ | TTFT(geo) |
|---:|---:|---:|---:|---:|
| 512 | 1387.7 | 1423.7 | +2.6% | 362 ms |
| 1024 | 1417.0 | 1432.7 | +1.1% | 717 ms |
| 2048 | 1396.1 | 1417.3 | +1.5% | 1447 ms |
| 4096 | 1356.1 | 1394.0 | +2.8% | 2940 ms |
| **8192** | **1310.6** | **1358.0** | **+3.6%** | **6034 ms(−3.5%)** |

- 逐轮:@8192 geo = 1201/1359/1357,**无冷启爬坡**(eager = 1148/1311/1311 且单点可爬到 1356);
- 正确性:needle 4 档全对 + 17/30 tok 与 eager **逐字相同**;
- 代价:桶族 slab **+203 MiB**(总 296.5)⇒ `GRAPH_SLAB_BUDGET` 262 → **320 MiB**;
  ⚠️ 同轮稍后阶梯尾档 **+t1024**(chunk 1024 生产档)把桶族 Σ 抬到 **502.3 MiB**
  ⇒ 预算最终定在 **640 MiB**(总治理 400 → 778 MiB);见 `state.rs` 与 §9.6 追注;
  桶命中(std 相)~100%;`pad > 4` 行的尾块自动让位 eager。

### 9.5 已知缺口

1. **`gpu_prefill_bucket_matches_eager`(0.8B)在干净 main 上就红**
   (`CUDA_ERROR_ILLEGAL_ADDRESS` @ `record_tail(stream 1)`;stash 对照确认预存)。
   ⇒ 桶路当前**没有绿的硬等价门**,本轮的证据是 27B 真机探针 + std;
2. plan sync 首轮成本 ~1.5 ms/块(同 ctx 的第二轮起缓存命中)⇒ 可做请求级 plan 预热;
3. 1400 t/s 未达(1358):差头寸 = 非 GEMM ~52 ms/块(attention 18.2 + GDN/FLA ~20 + norms/silu ~12)。

### 9.6 KVMem × prefill 桶:Q 槽阻断(2026-10-21)

**KVMem 档的 prefill 桶当前 100% 被旁路**,不是配置漏配:
`exec/prefill_chunk.rs::bucket_prefill_chunk` 在 `select_decode=true` 时
`return Ok(None)`(选择层是 KVMem 的硬前置),于是桶即使捕获也一次不回放。
根因 = 选择层要每层全注意力 **post-RoPE Q 的末 8 行**,eager 路经
`ctx.q_sink → eval_root_and_q → coverage_take_q`,桶图只输出 hidden/taps,
没有 Q 窗出口。

实测(单 3090 Ti,27B Q4_K_M,8k):
- KVMem + geobuckets:6 桶 `Captured`(571 MiB),请求后
  `bypass.diag=10 / hit=0 / eager.chunks=10`;
- 非 KVMem eager vs geo:**1824.4 → 1841.8 t/s(+0.95%)**、
  TTFT **4491.8 → 4449.4 ms(−42.5 ms)**,8/9 命中(尾块 569 tok 因
  `t1024 pad=455 > 4` 正确回 eager)。

修法 = 捕获期把每层 Q 的末 W 行 concat 成桶输出(`≈1.5 MiB/桶`),回放后走
窗口版 `coverage_take_q_window` 填 `CoveragePending`;选择层仍在图外跑。
详细代码路径、次要缺口(t1024 padding 护栏/阶梯顶 1024/GDN host 槽/
`fi_plan_sync`/无绿硬门)与验收见
`prefill-bucket-qslot.md`。
