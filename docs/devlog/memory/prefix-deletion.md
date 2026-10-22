# 前缀无感删除（Prefix-Transparent Deletion）—— 设计与施工

> 2026-10-21 改判立项（原《图像窗口旋转-设计与施工》**已否决**，可复用结论吸收进 §9 附录 A）。
> 一句话：**历史里删掉一段（比如一张旧截图），前缀缓存照旧命中，删点之后不重算
> —— 只把后缀的 K 乘一个旋转 `R(Δ)`。**
>
> 用户口径（2026-10-21）：
> 「我们的目的就是 GUI agent 会从历史里抽掉一张图片，而且还要保持前缀缓存不失效、
> 不需要重算。就按这个需求做，不需要图片旋转/滑窗那一套。」

| 修订 | 内容 |
|---|---|
| 2026-10-20 | 首轮：图像窗口旋转（**已否决**，见 §9 附录 A） |
| 2026-10-21 | **改判**：窗口/滑窗整段作废；改为「页对齐图像块 + 删除即位置帧平移 + 后缀 Δ-rope」，与 KVMem B-2 合流 |

---

## 0. 结论速览（TL;DR）

1. **触发形态**：客户端（harness）照旧每轮发全量历史，只是把某张旧截图删掉。
   引擎**不需要任何新字段**——用 `老 tokens` vs `新 prompt` 求
   「最长公共前缀 `L` + 最长公共后缀 `R`」即可识别删除区间。
2. **为什么能不重算**：RoPE 只吃**相对位置**。删掉一段后，其后所有 token 的位置
   整体前移**同一个量 `S`**（一段的位置推进）⇒ 只要把**后缀的 K** 乘 `R(−S)`：
   - 后缀内部：两端同时旋转 ⇒ 相对位置不变 ⇒ 注意力**精确不变**；
   - 后缀 ↔ 前缀：相对位置恰好等于"真的删了"之后的值 ⇒ **精确**；
   - **V 不用动**（V 无位置），**前缀一个字节都不用碰**（连 K 都不转）。
3. **唯一的机械约束**：块表是**页粒度**（32 tok/页）。要让"删"变成**零数据搬运**，
   被删区间必须是**整页数、且起点落页边界** ⇒ 图像块强制**页对齐**（前后各补 pad）。
   由于删除长度是页的倍数，**后续图像块的页对齐自动保持**（每轮 pad 数不变）。
4. **每轮成本**：页表拼接（µs）+ **后缀驻留块**的 K 旋转（20k ctx ≈ 300 MiB/rank，
   ≈0.5–1 ms）+ 只 prefill 真正新增的 token。对比今天是"删点之后全量重灌"
   （20k ctx ≈ 13–17 s）。
5. **与 KVMem 合流（关键）**：`crates/store/src/select.rs` 已经有位置帧账
   （`PosFrame`/`PosAction{Keep,DeltaReRope,Rebuild,StageIn}`/`assemble_view`/`delta_limit`）
   和"raw K 是重建权威"的口子（`push_pre_rope`），**只缺一枚执行核**；
   而 `coverage.rs` 的方案 B 一直挂在 `POS_SCHEME = "A(原位置稀疏)"` 上，注记是
   "B 需 B-3 的核"。那枚核 = 本轮已交付并真机验过的 `owl_kv_rope_k_shift_f16`。
   ⇒ **删除 = 位置帧平移的一个 producer**，不另立系统。
6. **语义边界（必须先认账）**：被删块**之后**的 KV 是"当年看得见被删内容"时算出的
   ⇒ 这是**搬运记忆**，不是重算。删点**之前**是**真等价**（可做逐位门）。

---

## 1. 需求与判据

GUI agent 的一步：

```text
[system][shot1][动作1][shot2][动作2]…[shotN][本轮问题]
                ↑ harness 把最早的截图抽掉（控上下文），其余全部前移
```

今天的代价：`match_prefix` 命中到删点就断，**删点之后 19k token 全部重灌**。

| 编号 | 判据 |
|---|---|
| **D1** | 删点之前：`cached_tokens` = `L`（页对齐），**逐字节**复用 |
| **D2** | 删点之后：**零 prefill**（除真正新增的尾部），KV 由"拼接 + 旋转"得到 |
| **D3** | 机械门：删除后后缀每层 K == `R(−S)·` 删除前同一批页的 K（配对量级 ≤2 ulp） |
| **D4** | 语义门：与"全量重算"对比 —— 删点之前**逐位**；删点之后 top-1 一致率/`cos`（监控） |
| **D5** | 延迟：每轮旋转段 ≤ 2 ms（20k ctx / 27B TP2），整步 TTFT 回到"只喂新 token"量级 |

---

## 2. 为什么不做窗口旋转（否决记录）

原方案（`图像窗口旋转`）要求：会话冻结"槽数 `N` / 槽长 / 网格"，客户端把**保留的
图片全部放在 prompt 尾部**，引擎维护 `N` 个等长槽、每轮**左旋页序并覆写最老槽**，
只 prefill 新槽 + 问题；窗口页**不进内容树**（Transient）。

**否决理由（不是"实现错了"，是"形态错"）**：

| # | 代价 | 说明 |
|---|---|---|
| 1 | **强加 prompt 布局** | 图必须集中在尾部、槽数冻结、网格冻结；harness 得为引擎改格式 |
| 2 | **会话级几何状态** | `slots/l_slot/npb/base/phys/老图指纹` 一整套账，生命周期与失效面都大 |
| 3 | **与 KVMem 记账逆向** | 窗口页 Transient ⇒ 不可下沉/不可召回/不可盘，等于在分层记账旁边挖了个洞 |
| 4 | **省的是小钱** | 它省的是"保留的 `N−1` 张图的 K 旋转"；而删除方案的旋转是**毫秒级**，本来就不是瓶颈 |
| 5 | 收益依赖"图只在尾部"这一前提 | 真实 harness 的历史是**图文交错**的，前提不成立就得先做 prompt 重排 |

⇒ 方案作废，保留其中**仍然成立**的两件：Δ-rope 核 与 **页对齐 pad 律**。

---

## 3. 方案

### 3.1 不变量（全部可验证）

设老序列 `O`、新序列 `N`，删除区间在老序列上是 `[L, L+D)`，删除后其后整体前移。

1. **前缀不变**：`N[..L] == O[..L]`（按定义）。
2. **后缀平移**：`N[L..] == O[L+D..]`（长度相同则为纯删除）。
3. **位置同移**：令 `S = pos_O[L+D] − pos_O[L]`（被删块的位置推进），则
   `pos_N[i] == pos_O[i+D] − S` 对所有 `i ≥ L` 成立。
   *（m-RoPE 下图像段三分量都以 `cur` 为锚 ⇒ 整体加同一 `S` ⇒ 一个标量就够。）*
4. **页对齐**：`L % 32 == 0` 且 `D % 32 == 0`。
   ⇒ 块表拼接为 `T' = T[..L/32] ++ T[(L+D)/32..]`，**零数据搬运**；
   ⇒ 且 `S` **不必**是 32 的倍数（旋转量是位置量，与页无关）。

**为什么 (4) 是自洽的（不会越删越歪）**：删除长度是页的倍数 ⇒ 删点之后的**每个
token 的页内偏移不变** ⇒ 后续图像块的"起点是否页对齐"不变 ⇒ **每轮补的 pad 数稳定**
⇒ 前后两轮在删点之外逐字节可比，前缀缓存自然连续命中。

### 3.2 页对齐 pad 律（常开，不再是窗口档专用）

对**每条 prompt 里的每一段图像占位游程**：

```
补 pad 到「起点页对齐 + 整段整页」：
  pad_before = (32 − 图像段起点 % 32) % 32        # 让 L % 32 == 0
  pad_after  = (32 − (L_img + pad_before) % 32) % 32
```

* pad token = `\n`（要求 tokenizer 恰 1 token；否则响亮拒）。
* pad 属于邻近**文本**（不是图像 token），不改变 `L_img`/位置推进；
* 由**引擎**做（服务面不需要知道），且**幂等**：同一段图像在同一文本偏移下补的
  数量恒定。

### 3.3 三步算法

```text
① 识别   diff(老 tokens, 新 prompt)
         L = 最长公共前缀；R = 最长公共后缀；D = old_len − L − R
         要求：L%32==0 ∧ D%32==0 ∧ D>0 ∧ 位置同移自证（§3.1-3）
         不满足 ⇒ **不吸收**，走今天的路径（命中到 L，之后重填；行为逐字节不变）

② 拼接   老链 = match_prefix(老 tokens)          # 内容树里取回，含引用
         T' = T[..L/32] ++ T[(L+D)/32..]         # 去掉被删块，其余原样
         会话块表 := T'（后续 prefill 追加尾部新页）

③ 旋转   对 T' 里**驻留**的后缀页发 R(−S)：
         每层一发（`owl_kv_rope_k_shift_f16`，物理页列表 + 单行 Δ 表）
         非驻留页只更新位置帧（`StageIn` 语义：搬入时按新位置重打）
```

**顺序纪律**：页表拼接 **早于** `write_bt`（否则注意力读旧页序）；旋转 **早于**任何读
后缀的 chunk；全程**无** `GraphBegin/GraphEnd`（只改块表内容 + 原地重打 K）。

**这枚核目前住在哪**（P2 的移植清单，3 个文件）：

| 文件 | 内容 |
|---|---|
| `crates/kernels/cu/elems/kv_rope_shift.cu` | 设备核：按**物理页列表**原地 rotate-half，单行 `cos1/sin1` 广播 |
| `crates/kernels/src/ops/elems.rs` + `driver.rs` | 算子声明（`KvRopeKShiftF16`，含 `POOL = kv.main/Knhd/F16`）与拾取（grid = 块×kv头×半维组，block = `page·x`） |
| `crates/kernels/tests/kv_rope_shift_parity.rs` | 真机门：Δ=0 逐字节 / Δ≠0 配对量级 ≤2 ulp / 未列页不动 / 往返 |

（三者现在在 `feat/image-rotation` 工作树里；本卷方案落地时原样搬进 main，
`kvshift.rs` 的 `delta_table`/`KvShiftGeom`/host 参考一并搬。）

### 3.4 与 KVMem B-2 合流（唯一的位置真源）

| KVMem 既有件 | 本方案怎么用 |
|---|---|
| `PosFrame { block, pos, action }` / `assemble_view()` | 删除 = 摘掉被删块后**整体重算 `pos`**（此后每块 `pos -= S`） |
| `PosAction::Keep` | 删点**之前**的块（`Δ=0`，零动作） |
| `PosAction::DeltaReRope` | **驻留**的后缀块（在页内原地重打 `R(−S)`） |
| `PosAction::Rebuild` | `delta_limit` 超限 ⇒ 从 **raw K（pre-RoPE）** 重建一次（防 f16 累积漂移） |
| `PosAction::StageIn` | **非驻留**的后缀块：**删除那一刻什么都不做**，等召回时"搬入 + 按新位置重打" |
| `push_pre_rope`（raw K 权威） | 漂移的唯一解药：device 只持"当前 `pos` 编码"，累计 Δ 只记账 |
| `POS_SCHEME`（现为 `"A(原位置稀疏)"`） | 本轮交付的核让 **B（紧凑窗口 + 有界 delta）** 可行 ⇒ 可作为后续档位打开 |

> **单一真源律**：位置账只允许一份（`frames`）。任何"另记一套 base/Δ"的实现都算旁路，
> 评审直接拒。

### 3.5 每轮命令序（顺序即语义）

```text
⓪ diff + 位置同移自证（纯 host）      ← 失败 ⇒ 退回普通路径（零副作用）
① host: 块表拼接 T'（`match_prefix(老 tokens)` 取链）
② write_bt(T')                        ← 内容重写，图不重捕
③ re-rope: 后缀驻留块 × 每层一发（Δ = −S）
④ prefill: 只有**本轮新增**的尾部（新图 + 问题）
⑤ decode（位置照常，`ins.delta` 已由新 prompt 的 m-RoPE 给出）
```

---

## 4. 记账

| 对象 | 处置 |
|---|---|
| 删点之前的块 | 内容身份不变 ⇒ **树里不动**，引用照旧 |
| 后缀块 | **同物理槽、新内容身份**（父哈希变了）⇒ `register_chain_inc(新父, &[(tokens, slot)])` 重挂；引用 +1，旧身份自然变垃圾等 LRU |
| 盘链索引 | **跳过后缀**（每轮几百条 I/O 不值）；只索引新增尾部 |
| GDN 快照 | **后缀内部边界在新链上没有合法快照**（快照键是 old-chain）⇒ 复用面必须**二分**：*页可复用（不含快照）/ 边界可复用（含快照）*。GDN 只从"上一轮尾巴快照"续 |
| **共享块（COW）** | 删点之后的存活块可能被**别的会话**按内容树共享（引用 >1）⇒ **禁止原地重转**：要么 COW（新页 + copy + 转），要么该块不吸收（退回重算）。vLLM 同款预检见 §11-1 |
| Transient | **不需要**（与窗口方案相反：后缀留在树里，可下沉/可召回/可盘） |
| 超卖/下沉档 | 天然兼容：非驻留部分走 `StageIn` 惰性重打 |

### 4.1 `reuse_face` 的二分（必须显式，否则静默丢复用）

今天 `scheduler.rs::reuse_face` 的判据是「末块对齐 + **快照在场**」⇒ 它会把复用面裁到
"最后一个有快照的边界"。删除路径要的复用面 = **页在、快照可以不在**（因为不 prefill 后缀）。
不做这个二分 ⇒ 后缀页被丢掉，收益归零且**没有任何报错**。

---

## 5. 门

| 门 | 判据 | 备注 |
|---|---|---|
| **G0 核门** | `Δ=0` 逐字节不变；`Δ≠0` 与 host 参考**配对量级 ≤2 ulp**；未列页一字未动 | ✅ 已交付（`kernels/tests/kv_rope_shift_parity.rs`，真机绿） |
| **G1 机械等价** | 删除后后缀每层 K == `R(−S)·` 删除前同一批页的 K | 错页/错 Δ/错层都会打出 ≫2 ulp |
| **G2 命中** | `cached_tokens` = `L`（删点页对齐）；删点之后**零 prefill** | `/debug/vram` + `usage` |
| **G3 延迟** | 旋转段 ≤2 ms @20k/27B；整步 TTFT ≈ "只喂新 token" | 与"不删"对照 |
| **G4 语义** | 删点之前**逐位**一致；删点之后 top-1 一致率 / `cos`（**只监控**） | 语义非等价是已知前提 |
| **G5 退化** | 任何不满足 §3.1 的情形 ⇒ 自动退回今天路径，**行为逐字节不变** | 兜底不许静默错值 |
| **G6 共享安全** | 存活块 `ref_cnt > 1` ⇒ 必须走 COW 或整轮放弃（**禁止**原地重转） | 跨会话共享是内容树的常态，不是边角 |
| **G7 Δ 一次性** | 同一次删除的 `Δ` 只能施加一次：`frames` 的 `pos` 是唯一真源，`Δ` 用完即清 | 重复施加 = 静默损坏（vLLM 的 preempt/resume 就踩这个） |

---

## 6. 分阶段与工作量

| 阶段 | 内容 | 估时 |
|---|---|---|
| P0 | **识别**：`prompt_diff`（L/R/D/S）+ 位置同移自证 + 页对齐 pad 律常开 | 0.5 天 |
| P1 | **拼接**：`match_prefix(老 tokens)` 取链 → 块表拼接 → `write_bt`；`reuse_face` 二分 | 0.5–1 天 |
| P2 | **旋转**：接线 `owl_kv_rope_k_shift_f16`（每层一发）+ 接 KVMem `frames`（`pos -= S`） | 0.5 天 |
| P3 | **记账**：链身份重挂（`register_chain_inc`）+ 盘索引跳过 + GDN 尾巴快照 | 0.5 天 |
| P4 | 门 G1/G2/G3 + 0.8B/27B 真机；G4 监控面 | 1 天 |
| P5 | 漂移：`delta_limit` + raw K 重建路径（KVMem 已设计，接上） | 0.5–1 天 |

合计 ≈ **3.5–4.5 天**（不含 letterbox/混合分辨率）。

---

## 7. 风险与不做的事

**风险**

1. **语义近似**：后缀 KV 带着"被删内容"的上下文（§0-6）。**删点之前才是真等价**。
2. **GDN 不可倒带**：被删图仍在 GDN 里；要"严格忘掉"只能重算。
3. **f16 累积漂移**：逐轮叠加旋转会漂 ⇒ 走 KVMem 的 `delta_limit` + raw K 重建（P5）；
   在 P5 之前，**同一块被旋转次数计入 frame**，超限标记 "需要重建"并可降级重算。
4. **识别误判**：客户端做的是"编辑"而不是"删除"（尾部也变了）⇒ `R=0` ⇒ 仍可吸收删除部分
   （`D` 页对齐时），否则退回。**永远不许**在自证失败时动手。
5. **`D` 必须是页倍数**：文本压缩（删任意长度文字）不在 v1；要支持就得搬 ≤31 token/页
   （一页 32 tok × 层数，可做，另案）。
6. **长上下文线性成本**：旋转 O(后缀)；1M 档要裁剪策略（只转最近 N 页 + 更早的重建）。

**不做（明确挂账）**

- 不做窗口/滑窗（已否决，§9 附录 A）；
- 不做**插入**（插一张图 = Δ 为正 + 页表插入，需要页分裂/搬页，另案）；
- 不做跨句"任意文本压缩"；
- 不与投机解码的 verify 行混用（verify 行不进位置帧，先各自干净）。

---

## 7bis. 相关工作（2026-10-21 调研；社区已有人做，且**共识缺口 = 那枚重转核**）

### 一、直接同题：「删中段 + 位置重转」

| # | 工作 | 做法 | 与我们的关系 |
|---|---|---|---|
| 1 | **vLLM PR #43374**《Experimental session KV eviction with attention sinks》(open) | 直播会话里删中段 `[s,e)`，头部 sink 钉住（StreamingLLM）；**只发引擎管道 + `_post_add_requests` 钩子**，重转留给 runner | **同一段推理**（原话：*"surviving tokens shift from p to p−(e−s). Their cached K still carries the old RoPE rotation. Without re-rotation, attention … is numerically wrong on every RoPE model"*）；并明确 **"a future in-tree RoPE re-rotation kernel"** 是缺件 |
| 2 | **vLLM RFC #57672** | *"survivor keys stored raw, rotated once at eviction time rather than continuously, to avoid the compounding drift a naive per-step re-rotation kernel would hit"* | 与我们 §3.4（raw K 权威 + 一次性旋转）**逐字同构**，独立提出 |
| 3 | **vLLM #53871 / #53876 / RFC #51948**（*二手*：见 #43374 评论区 nvbfalk） | 块对齐的"释放一段 token"原语 + 保留/逐出核心 | 对应我们 P1 的页表拼接；**并暴露一个设计分叉**：那条线**不压缩位置**（存活 token 保持原位置），本卷要压缩 ⇒ 即 KVMem `POS_SCHEME` 的 A/B 之争 |
| 4 | **NVIDIA kvpress `KeyRerotationPress`**（已合入库） | `delta_pos = kept_idx − selected_pos`；`k*cos + rotate_half(k)*sin`；docstring 点名 **HF `SinkCache`** 与 **FINCH** 同法 | 与我们的核**逐式相同**（含符号方向）；说明"prune 后重转"在推理库层面已是既成做法 |
| 5 | **llama.cpp discussion #7887** | *"store non-RoPEd KV cache, 'shift' it when the context is full, compute RoPE over the entire cache"* | 同一条 raw-K 路线（上下文满时移窗） |

**⇒ 我们不是首创者，但我们是"把它接到分页 KV + 内容树 + 位置帧"上的那一版。**
vLLM 那边的重转核**至今没落地**（PR 仍 open、env 门仍在、RFC 还在讨论形态）；
而 `kvpress` 的实现是 HuggingFace 张量 + 全量 gather，**不适用于分页池**。
我们这枚核（池内按物理页列表原地、单行 Δ 表、无 scratch）填的正是这个洞。

### 二、vLLM 踩出来、我们必须吸收的三条硬约束

1. **共享块不许原地重转**：`evict_and_compact` 预检「删点之后存活块 `ref_cnt > 1` ⇒ `RuntimeError`」，
   要求这类会话**关掉前缀缓存**；COW 是 future work。
   ⇒ 我们更危险：内容树天然跨会话共享 ⇒ 落成 **G6**（COW 或整轮放弃）。
2. **多模态原子性**：每个 mm item 必须"完全在删点前/完全在内/完全在后"，跨越就 `ValueError`
   （保 m-RoPE 位置数学）。⇒ 对应我们的 §3.2 页对齐 pad 律 + 原子删除判据。
3. **Δ 一次性**：`num_tokens_evicted` 传给 runner 后必须清零，否则 preempt/resume 会**重复施加**旋转 ⇒ KV 损坏。
   ⇒ 落成 **G7**（`frames.pos` 是唯一真源）。
4. **先证后动（预检三段式）**：`Scheduler.evict_token_range()` 在**改任何状态之前**依次校验
   「请求状态 → 多模态原子性 → 共享块资格」，任一步抛错则**请求原样不动**。
   ⇒ 我们照抄这条顺序纪律（与 KVMem/Admission 的"先判决、后动手"同源），写进 P0：
   `diff/自证` → `共享块资格` → `页表拼接`，**前两步都是纯读**。

### 三、邻近但不同的技术族（可对照/可组合）

| 工作 | 路线 | 取舍 |
|---|---|---|
| **PromptCache**（MLSys'24，arXiv [2311.04934](https://arxiv.org/abs/2311.04934)） | 模块化 prompt 的 attention 复用；核心难点"位置相关 ⇒ 需 **positional adjustment**" | 我们是它的"删除版"：只补偿一个标量 Δ |
| **CacheBlend**（EuroSys'25，arXiv [2405.16444](https://arxiv.org/abs/2405.16444)） | 非前缀块复用 + **selective recomputation**（重算一小部分以融合知识） | **替代路线**：不转、重算 x%；比我们准、比我们贵 ⇒ 可组合（关键块重算 + 其余旋转） |
| **EPIC**（arXiv [2410.15332](https://arxiv.org/abs/2410.15332)） | 把 KV 算成**与位置无关**（固定位置窗口），任意位置复用 | 思路相反：他们消掉位置依赖，我们按需补偿 |
| **FINCH / HF SinkCache / H2O / SnapKV** | 都是 KV 压缩；**kvpress 把重转抽成可插拔 wrapper** | 工程范式可借（我们的核也可做成"press 式"） |
| **GUI-KV**（arXiv [2510.00536](https://arxiv.org/abs/2510.00536)；abs 页未抽出标题，方法名取自正文） | **GUI agent 专用** KV 压缩：空间显著性 + 时间冗余打分剪枝；5 屏 AgentNetBench 上 decode FLOPs **−38.9%**、步准确率 **+4.1%** | 同域但**不同轴**：它剪 token、不做整块删除、也不做位置补偿 ⇒ **"删一整张图 + 前缀不失效"这条路还没人完整做完** |

### 四、来源与核验状态（区分**一手**与**二手**，别把传闻当事实）

| 来源 | 链接 | 核验 |
|---|---|---|
| vLLM PR #43374 | <https://github.com/vllm-project/vllm/pull/43374> | **一手**：正文 + 评论 + review thread 全文读过（关键引文逐字摘录） |
| vLLM RFC #57672 / #53871 / #53876 / RFC #51948 | 未单独打开 | **二手**：引文来自 #43374 评论区（`null-Exception1` / `nvbfalk`），编号存在但**内容未核** |
| kvpress `KeyRerotationPress` | <https://github.com/NVIDIA/kvpress/blob/main/kvpress/presses/key_rerotation_press.py> | **一手**：源码读过（Δ 定义与符号方向已比对） |
| HF `SinkCache` / FINCH 也用重转 | — | **二手**：kvpress docstring 声称；本卷**未**独立核 |
| llama.cpp discussion #7887 | <https://github.com/ggml-org/llama.cpp/discussions/7887> | **二手**：搜索摘要引文；原帖未读 |
| PromptCache / CacheBlend / EPIC / GUI-KV | 见上表链接 | **一手**：abs/正文摘要读过（GUI-KV 的数值取自正文） |

> 纪律：**二手引用一律标注**。落地前若要引用其结论（例如"vLLM 那条线不压缩位置"），
> 先回去读一手，别拿评论区一句话当设计依据。

### 五、结论

1. 方向被社区验证（vLLM 三条线 + kvpress + llama.cpp 都在同一处），**不担心走错路**。
2. 我们的差异化 = **分页池内的原地重转核**（别人缺的那块）+ **统一位置账**（KVMem `frames`）
   + **删除后重挂内容身份**（不用整块 COW 就能保住跨轮复用）。
3. 必须补的两条工程硬点来自 vLLM 的踩坑记录：**共享块 COW**、**Δ 一次性**（§4 / G6 / G7）。

---

## 8. 与其它能力的关系

| 能力 | 关系 |
|---|---|
| **KVMem 选择层（B 轴）** | **同一套位置帧**；删除是它的 producer，它是删除在 overcommit 档的兜底（`StageIn` 惰性重打） |
| 投机解码 | 正交（verify 行是 turn 内临时行） |
| 双槽/notebook | 正交（那些是"插入/可写区"，本卷只做"删"） |
| 视觉塔 | 新图照常跑塔（**不需要** `keep_from` 那套） |

---

## 9. 附录 A：已否决的窗口旋转（原卷摘要，仅供考古）

原方案：会话冻结 `N` 个等长"视觉尾窗"槽，每轮左旋页序 + 覆写最老槽 + 对保留槽的 K
乘 `R(Δ)`，窗口页标 Transient。**核心结论仍然成立并已被本卷吸收**：

| 原结论 | 本卷归宿 |
|---|---|
| Δ 表可由 `MropeCfg::cos_sin(&[[Δ,Δ,Δ]])` **单行**给出（与 prefill 表同源） | **原样保留**（§3.3 ③） |
| 池 K 是 `[nb,hkv,hd/x,page,x]`，拿不到 token-major 视图 ⇒ **必须新核** | **已交付**：`owl_kv_rope_k_shift_f16` |
| `Δ = (base₂−base₁) − S`，`S = adv + (l_slot − l_img)`（**位置**推进，不是 token 位移） | **原样保留**（§3.1-3） |
| "核 ≡ host 逐位"只对 `Δ=0` 成立；Δ≠0 要按**配对量级 ≤2 ulp**（nvcc FMA 收缩） | **原样保留**（§5 G0） |
| `P` 必须纯文本、`pos[i]==i` | **原样保留**（§3.1-1 的前缀不变式） |
| 窗口页不进树（Transient） | **否决**（与 KVMem 记账逆向） |
| 槽几何/冻结/槽数/老图指纹/`skip_to` 跳越 | **否决**（形态错，§2） |

## 10. 附录 B：施工中已确认的既有红（与本卷无关，但挡过路）

| # | 症状 | 归属 |
|---|---|---|
| V1 | `TagFilter::default()` = **deny-all** ⇒ 全部引擎 e2e 第一发池写核被拒 | ✅ 已修（main） |
| V2 | 0.8B 硬拒 `[model].vision = true` | ✅ 已修（main，`load_vision_into` 提公共） |
| V3 | 塔输出块被段边界 `drain_turn_stale` 当残块 free ⇒ 第 2 chunk `image_scatter` `DeadBlock` | ✅ 已修（main，`drain_owner_except` 例外集） |
| V4 | 小网格 → FI ragged `cta_tile_q` 缺口 ⇒ `err -205` | ✅ 已修（main，`rows >= 64` 自动退 native） |
| F1 | boot 后**第一轮就带图**吐退化串（0.8B 特有；27B 不复现） | ⏳ `fix/vlm` 另案（P2） |
