# 图档 FI 单图 —— 设计与施工（2026-10-18 立项；性能专项 · 图治理）

> **一句话**：把 FI 的"宿主按真实 ctx 算的调度表"改成**图档 plan（形状全域恒定）**
> + **窗外按需刷表（内容精确）** ⇒ **decode / verify 都回单张捕获图，零断点、零核改动**，
> piecewise 整体退役。依据 = 本卷 §2 的三条实测判据（含一条 host-oracle 硬门）。
>
> 方法论来源：`repos/ninfer-3090`（同为 Qwen3.8-27B / sm86 / 单卡 24G 的生产引擎）——
> 它的三阶段图化只有三件事：**信封上界定形 / 动态量走固定地址 ingress / 空工作者安全**。
> 本设计把这三件事逐条映射到我们已有的机制上（不改 FI 数学）。

## §1 目标形态（终局，不做过渡）

```
boot:   图档 plan(FI_CUDA_GRAPH=1) → 捕获 decode 图 1 张(+可选桶图) / verify 图 K 张(T=K+1)
        + 关"图自携 plan restore"(改由窗外刷表)
每步:   写 ingress(既有输入槽:frontier/pos/kv_len/kv_slot/gdn_slot + FI 表四件套 + 宿主标量)
        → graph_launch(整模单图) → 出口 D2H(既有)
按需:   host 在**窗外**把当前桶/当前轮的 plan 表刷进 int_ws(一条 KB 级 H2D;`activate` 已有)
```

**收益**：① 去掉 decode 每步 17 段 launch + 16 次 eager 断点；② **断点退役 ⇒ `pw_carry`
的线性增长病从结构上消失**；③ verify 每轮同样受益；④ 长 ctx 保留 split 的平坦性。

## §2 判据（全部实测；探针 = `crates/kernels/tests/{fi_plan_scan,fi_prefill_probe}`）

### 2.1 图档 plan 的形状**全域恒定**（`fi_plan_scan`，新增）

| 档 | ctx 32 → 262144 |
|---|---|
| `FI_CUDA_GRAPH=1`, T=1 | `padded=42, cta=16, split=1, int_bytes=618` —— **一条直线** |
| `FI_CUDA_GRAPH=1`, T=8 | `padded=42, cta=64, split=1, int_bytes=650` —— **一条直线** |
| `FI_CUDA_GRAPH=0`, T=1 | 16 段跳变 + 非单调（1→2→…→39→**37**→39→…→42） |

`padded=42 = 2·SM/hkv`（producer grid cap）；`int_bytes` 恒定 ⇒ **表布局/偏移恒定 ⇒
一张图可服务任意 ctx，只需换表内容**。这就是 ninfer 的"envelope 上界定形"。

### 2.2 过供给（表按大上界 / 运行期真 ctx）：**T=1 精确，T=8 会错**（`fi_prefill_probe` 新腿
`FI_PF_PLAN_CTX`）

| 组 | T=1 | T=8 |
|---|---|---|
| 控制组（plan = 真值 ctx=3072） | ✓ max\|FI−host\|=0.000274 | ✓ 0.000500 |
| 过供给（plan 按 262144） | ✓ **0.000274（逐位同值）** | ✗ **1.391**，8/8 行超门，零值 43008/49152 |

`1.391 / 43008 / 49152` **与设计卷 §2-B 记录的 V1 bug 逐字相同** ⇒ 机制 =
`num_kv_chunks = ceil_div(kv_len, kv_chunk_size)`（运行期真值）vs `o_indptr/merge_indptr`
（plan 按上界生成）不一致 ⇒ partial 行 stride 错 ⇒ 除第 0 行全零。
**T=1 不参与该 stride ⇒ 过供给精确**（这也解释了为何 decode 桶化实测文本逐字一致）。

### 2.3 空 tile 输出**本来就是安全的**（读码定谳，故零核改动）

- `init_states`：`m = -inf, d = 1`；
- `OutputTransform`（上游默认）：`d_rcp = (m != -inf) ? rcp(d) : 0` ⇒ **空 tile 输出精确 0**，
  `lse = -inf` ⇒ merge 权重 `exp(-inf − m_max) = 0` ⇒ 零贡献。
⇒ **上游无需补丁**；断裂点只在"表与真实 ctx 不精确对应"。

## §3 施工项（按序；每项带门）

| # | 项 | 内容 | 门 |
|---|---|---|---|
| W1 | **图档 plan 转正** | `[dispatch] fi_cuda_graph = true`（现为 env `FI_CUDA_GRAPH`）；plan 形状全域恒定 | §2.1 扫描复跑；`fi_prefill_probe` 全档绿（图档瓦） |
| W2 | **关"图自携 plan"** | 图内不烘 restore memcpy（现 diag `OWL_FI_NO_PLAN_RESTORE` 转正式配置）；int_ws 唯一写者 = 窗外刷表 | 同格复用腿仍绿；`[fi-plan]` 计数 = 刷表次数 |
| W3 | **decode 单图** | `decode_fi` 走单图（不再 `plan_piecewise`）；桶只用于压刷表次数（缺省 1-2 桶） | 文本与现役断点臂**逐位一致**（恒等门）；ms/tok 不劣；soak 40 请求零漂移 |
| W4 | **verify 单图** | 每个 K 一张（T=K+1 固定）；**每轮**窗外刷表（ctx 每轮变） | 同上 + `OWL_GATE_PIECEWISE=1` 恒等门整臂 |
| W5 | **piecewise 退役** | 删 `plan_piecewise` 调用面/`pw_carry`/断点登记；`fi_prefill_probe` 的过供给腿留作回归门 | 全库测试 + 门套；文档/坑册同步 |

**显存账**（实测 `slab 总账`）：decode 单图 20-32 MiB；verify 单图 165 MiB；
K=3/7 各一张 ⇒ **总计 < 0.4 G**（对比"每桶独立图"：7 桶 verify 会吃 1.15 G ⇒ 不可行）。
**桶数**：因形状恒定，桶只为省 H2D/白跑 CTA ⇒ **1-2 个即足**（无需 ninfer 那样的 7 段阶梯）。

## §4 红线与记账

1. **表必须与真实 ctx 精确对应**（§2.2）⇒ 桶化只允许发生在 **T=1**；verify 必须每轮刷精确表。
   （2026-10-18 反例留档：verify 桶 2048 ⇒ 空文本/`!!!`/42.9 t/s；回退即恢复 71 t/s。）
2. 改 vendored `cu/flashinfer/*` 的语义须按 charter A4 显式认领并登记 `VENDOR.list.md`
   —— **本设计不需要改核**（§2.3），若将来要"表用上界"再走该流程。
3. 图档 plan 的 `padded=42` 对小 ctx 是白跑 CTA ⇒ W3 的门里必须量税（探针 per-call ms 同档可比）。
4. `cuGraphExecUpdate`（cudarc `driver/sys`）**非必需**（形状恒定 ⇒ 桶少），留作后续优化项。

## §5 施工状态（滚动更新）

| # | 项 | 状态 | 判据/证据 |
|---|---|---|---|
| — | 前置：FI plan 键同源 + decode 桶化（Phase 1） | ✅ 2026-10-18 | 32k 裸 decode 38.17 → **23.53 ms/tok（1.62×）**，文本与逐 ctx 档逐字同；f16 std 15/15 零回归 |
| — | 判据装置：`fi_plan_scan.rs` + `FI_PF_PLAN_CTX` 过供给腿 | ✅ | §2.1 / §2.2 两表 |
| **W1** | **图档 plan 转正为配置项** | ✅ 2026-10-18 | `[dispatch] fi_cuda_graph`；boot 直写 `FI_CUDA_GRAPH`（唯一写者=入口，读侧唯一=kernels）。门：knob off ⇒ `graph=0` + `padded∈{1,16,25,…}`（跳变）；knob on ⇒ **`graph=1` + `padded=42` 恒定** |
| **W2** | 关"图自携 plan restore"，改 host 每步刷表 | ✅ 2026-10-18 | `OWL_FI_NO_PLAN_RESTORE=1`(bake 档 boot 期置) + **`activate` 语义收窄**:该档只在**窗内**不烘 restore,窗外**照旧重刷**(旧写法窗外也 `return` ⇒ 刷新被掐,是本批第一只拦路虎)。门:`int_ws 激活 cap=true` 计数 = 0 |
| **W3** | **decode 单图 + 每步 `fi_plan_sync`** | ✅ 2026-10-18 | 见 §5.2:文本与 piecewise 臂**逐位一致**,ms/tok **−4~6%**,图显存 **32→16 MiB**;臂事实位 `decode 面事实:FI 节点登记=true` |
| **W4** | **verify 单图 + 每轮 `fi_plan_sync`** | ✅ 2026-10-18 | 见 §6:整轮 GPU **−5.0%（对 ctx 平坦）**、发射税 10.9-12.0 → **0.2-0.3 ms**、三档 ctx `reasoning_content` **逐字节恒等**、971 轮 soak σ=0.11 ms。⚠️ 首版负收益是**假绿**（FI 节点根本没进图）⇒ 坑册 §58 |
| **W5** | **piecewise 整体退役** | ✅ 2026-10-18 | 删除面:`plan_piecewise` / `replay_piecewise` / `Mode::CapturedPiecewise` / `BreakPoint`·`PiecewiseRt` / `pw_carry` / `BreakDecl`+`EvalCarry`+`SegMode`+`eval_segment` / 断点登记(`attn_break`·`AttnFiBreak`·`note_attn_kernel`) / 诊断面(`pw_*`) / 旋钮(`verify_piecewise`·`break_at_out`·`fi_bucket`·`pw_dump_tok`·`fi_plan_bake_verify`)。**新增门**:①FI 臂 ⇒ `fi_plan_bake` 必在场(缺上界 = 占位 ctx 烘表 ⇒ 文本连贯但错);②`fi_plan_bake` ⇒ `fi_cuda_graph`;③退役键**响亮拒启**(迁移门)。规模:engine −约 1.1k 行 / models −约 460 行。判据:四包测试全绿(shared 68 / engine 40+3 / models 178+2 / kernels lib 77)+ 生产档冒烟(`verify 面事实:FI 节点登记=true | bake=Some(16384) | 段=0 断点=0`,verify-prof `pure≈24.7 ms` 与 W4 档同尺)+ 退役键拒启实测 |

### 5.1 图档瓦的**税**（引擎级实测，scripts: /tmp/tax.py；spec off，greedy，128 tok）

| ctx | 桶 2048（2 档阶梯的近档） | 桶 262144（一图吃满上界） | 税 |
|---:|---:|---:|---:|
| 512 | 22.42 ms/tok | 22.84 | +1.9% |
| 2048 | 22.47 | 23.45 | +4.4% |
| 8192 | 22.81 | **25.58** | **+12.1%** |
| 32768 | 23.55 | 25.56 | +8.5% |

⇒ **结论**：图档瓦本身零代价（桶 2048 档与 eager 瓦逐点同值）；税只来自"桶取多大"
⇒ **桶阶梯 2-4 档（2048 / 8192 / 32768 / max）即把税压到 ≤2%**。
⚠️ 附带定谳：过供给在 T=1 是**合法但非逐位**（32k 档文本 hash 从 `f1b25f8126`
变为 `eb6c6fdbd5` = 另一套 split 归约序）⇒ **恒等/逐位类门必须同 plan 档内比对**；
跨档只能比"与 host oracle 的误差 ≤1e-2"。

---
## §6 W4（verify 单图）实测：**转正**（2026-10-18；首版负收益为**假绿**）

### 6.1 首版实测是假绿 —— 那一档里 FI 节点**从未进图**

首版结论（本卷旧 §6）："机制成立、GPU 前向 15.1 → 47.8 ms(3.2×) ⇒ 默认关"。
**不成立**：`plan_verify` 把 `fi_cell` 乘了 `pw_on`，单图档（`pw_on=false`）⇒
`fi_cell=None` ⇒ 整个 FI 面（四件套 + `ctx.fi` + `fi_plan_ctx`）被跳过 ⇒
层侧静默掉回 prefill 族（`chunked knhd`），而臂自证按**旋钮**还报 "FI(单图)"。

硬证（三条，账见 `perf-runs/2026-10-18-baseline/诊断-W4-verify单图-负收益-归因.md`）：

| 判据 | 事实 |
|---|---|
| boot 期 `FI_DEBUG=1` 的 `[owl-fi][run]` 计数 | 该档 **T=8 计数 = 0**（decode 同档 = 32）；boot 期 plan 只有 T=1/ctx=16384 一条 |
| 同 prompt A/B | 该档 `read` 30.9 ms / `pure` 31.0（FI断点 15.2 / 26.0） |
| "两边 plan 逐条相同" | **自产自销**：`fi_plan_sync` 的门 = `has_input("fi_q_cu")`（图级槽声明，恒真）⇒ 表照刷、无人读 |

⇒ 教训入坑册 **§58**（"槽在场 ≠ 节点在场"、臂自证必须事实位、read 不可跨臂比、
文本恒等是弱门）。

### 6.2 修后实测（生产 spec 档同参 A/B；自然英文 prompt，128 tok，greedy）

| ctx | 臂 | step | read | pure（整轮） | t/s | `reasoning_content` |
|---:|---|---:|---:|---:|---:|---|
| ~93 | ref（FI断点） | 10.92 | 15.22 | 25.94 | 57.56 | `1ac3387f37d5` |
| ~93 | **verify 单图** | **0.31** | 24.62 | **24.69（−4.8%）** | **59.02** | 同 ✓ |
| ~500 | ref | 11.30 | 14.94 | 26.01 | 67.01 | `910efa9ed236` |
| ~500 | **verify 单图** | **0.28** | 24.61 | **24.72（−5.0%）** | **68.67** | 同 ✓ |
| ~1832 | ref | 11.98 | 14.27 | 26.00 | 41.63 | `9756aea669b8` |
| ~1832 | **verify 单图** | **0.22** | 24.58 | **24.70（−5.0%）** | **42.17** | 同 ✓ |

- **发射税 10.9-12.0 → 0.2-0.3 ms**（verify 断点面整体退役，段=0/断点=0）；
- **整轮 GPU −5.0%，且对 ctx 平坦**（ref 也平坦 ⇒ 差值恒定，无长 ctx 劣化）；
- **恒等门**：三档 `reasoning_content` **逐字节一致**（510/706/703 B）；
- **soak**（40 请求 / 971 轮 / ctx 93→3608）：pure 首十轮 24.70 → 末十轮 24.87 ms
  （σ=0.11），零文本异常、零 ERROR/WARN 行。

### 6.3 附带修（同批）

1. **`fi_cell` 不许乘 `pw_on`**：断点**承载**只有 piecewise 需要，`ctx.fi`/图内位是**意图**；
2. **臂事实位**：`verify/decode 面事实:FI 节点登记=…` 随图落日志（臂自证不再按旋钮猜）；
3. **`fi_plan_sync` 门改意图三件套**（`bake_verify && bake.is_some() && slots`）；
4. **图归属登记**：`activate` 的 no-restore 早退分支也置 `graph_owned`
   —— 图里烘的掩码指针不许被驱逐面释放（split 关时的潜在悬垂）；
5. **配置门**：`fi_plan_bake_verify` 缺 `fi_plan_bake` = **响亮拒启**（旧行为 = 该档静默无效）。

### 6.4 生产档（2026-10-18 落）

`profiles/spec-dflash.toml`：`fi_cuda_graph = true` / `fi_plan_bake = 0`
（**哨兵 = 取 `[runtime] max_seq`** —— 档位改 max_seq 时上界自动跟随，不写死）/
`fi_plan_restore = false` / `fi_plan_bake_verify = true`。
冒烟（裸起生产档）：`verify 面事实:FI 节点登记=true | bake=Some(16384) | 段=0 断点=0`、
t/s 72.65、文本连贯；prefill 不变（eager FI）。

## §7 转正组合（2026-10-18 终版）

`decode=FI(单图)` + `verify=FI(单图)` + `fi_plan_restore=false` + 每步/每轮
`fi_plan_sync` 刷**精确表**（= 生产 flavor 现档）：

| 指标 | ref（decode/verify 都断点） | 本档 | 差 |
|---|---:|---:|---:|
| verify 发射税 | 10.9-12.0 ms | **0.2-0.3 ms** | −97% |
| 整轮 GPU（pure） | 25.94-26.01 ms | **24.69-24.72 ms** | **−5.0%** |
| t/s（ctx 93 / 500） | 57.6 / 67.0 | **59.0 / 68.7** | +2.5% |
| 文本 | — | **逐字节恒等** | ✓ |
| 图断点 | 16(decode) + 16(verify) | **0 + 0** | 退役 |

⇒ §1 目标形态达成：**decode/verify 都回单张捕获图、零断点、零核改动**；
`pw_carry` 的线性增长病在两路上**从结构上消失**。
**W5 已完成(2026-10-18)**:piecewise 面**整体删除**(见 §5 表 W5 行)—— O(步数²) 从"没代码走到"变成"没代码";同时把"缺上界 ⇒ 占位 ctx 烘表"这条静默错值路**结构性焊死**(FI 臂 ⇒ `fi_plan_bake` 拒启门)。
