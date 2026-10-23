# KVMem × DFlash2 交叉分析（ninfer-kvmem-actkv，2026-10-23）

> 对象：`repos/ninfer-kvmem-actkv`（HEAD `079779b`，2026-10-21）。
> 目的：回答“KVMem 与 DFlash2 如何同时使用”，并给 owl 的落地路径。
> 结论先行：**两者不是共享同一份 KV，而是两层 KV 解耦**——主模型 KV 可换出，
> DFlash2 conditioning ring 独立；ninfer 已把这条链路跑通（256k 召回 33/33，
> decode 最高 120 tok/s @5060Ti）。owl 目前缺的不是“算法”，而是
> **verify 的视图化页表/长度契约**。

---

## 1. 项目是什么

`repos/ninfer-kvmem-actkv` 是 **ninfer 后端 + KVMem/ActKV** 的 C++/CUDA 工作树：

```text
kvmem/                主机侧 KV 存储/检索/分层（与推理后端解耦）
src/adapter/          llama.cpp 适配（llama_memory_i）
backends/ninfer/      ninfer 子模块（Qwen3.5/3.8、DFlash2、CUDA graph）
docs/patches/         0001-...patch = 完整的 DFlash2 × KVMem 工作补丁（183 文件）
patches/              ninfer-kvmem-working-20261018.patch（2.2MB，含更早基线）
```

注意：当前子模块 `backends/ninfer` 是**上游 master**，不一定含 KVMem 改动；
可复现的代码在 `docs/patches/0001-feat-qwen3_5-DFlash2-KVMem-window-recall-with-merged.patch`
与 `patches/ninfer-kvmem-working-20261018.patch`。不要只读子模块就判断“没有实现”。

关键文档：

- `README.md`：DFlash2 × KVMem 修复摘要、256k 召回复测、传输优化。
- `docs/dflash2-kvmem-fixes-20261018.md`：五个修复的根因与验证矩阵。
- `docs/dflash2-kvmem-audit-20261019.md`：F1–F5 审查 + floor 语义修正。
- `docs/architecture.md`：KVMem 的 block window / host / 检索模型。
- `docs/active-kv-tiering.md`：host/SSD 分层与所有权。
- `docs/kvmem-incremental-restore-20261020.md`：差异 restore。
- `docs/kvmem-actkv-*.md`：ActKV LRFU / action-query 的尝试与失败边界。

---

## 2. ninfer 的核心模型：主 KV 窗口 + 草稿环

### 2.1 主模型：B + R 固定设备窗口

```text
逻辑上下文 0 ................................ L
                 ┌──────────────────────────┐
设备窗口         │ B: 选中历史块 │ R: 生成长度 │
                 └──────────────────────────┘
宿主/磁盘        │ 未选中的历史块（原生 K/V + scale）│
```

- `B = --kvmem-budget`（默认 36864 tok），`R = --kvmem-gen-reserve`（默认 16384）。
- 每个块 = 64 token（ninfer 固定页），选中块被**搬到设备窗口的连续页**；
  不是 owl 那种“页表别名、物理页散落”。
- `publish()` 设置 `removed_tokens = text_kv_valid - visible_tokens`；
  `compact_position(logical) = logical - removed_tokens`。
  **cache 位置用 compact，RoPE 位置仍用 logical**。
- 因此主模型的页表永远是 `0..visible_tokens/page` 连续；verify 不需要懂“洞”。

### 2.2 DFlash2：独立的绝对位置环

- 草稿 conditioning ring 容量 `S = draft.sliding_window`（DFlash2 常见 2048）。
- 槽号 = `logical_pos % S`；每轮 `append_context_impl` 无条件追加，
  与主模型 KV 的换出/重选无关。
- 草稿注意力只读最近 `S` 个绝对位置；`draft_kv_len = min(frontier, S) + 8`。
- 主模型换块**不会**让 ring 行失效；ring 是另一份设备分配。

> 一句话：**主模型 KV 可以稀疏/换出，DFlash2 ring 仍然是最近 2048 的连续环。**
> 这是两者能同时工作的结构性前提。

---

## 3. ninfer 踩过的五个坑（必须带进 owl 的设计）

| # | 坑 | 根因 | 修法 |
|---|---|---|---|
| 1 | ctx > 2048 后草稿接受率 0% | `dflash_envelopes.local.max_context` 被设成环容量 2048，并烘进 CUDA graph；核守卫 `length > max_context` 直接输出 0 | local envelope 的 `max_context` 传**逻辑上限 capacity**（262144），不是环容量 |
| 2 | 重选后草稿读到“缝合文本” | KVMem 换块后 ring 里残留被换出 token 的特征 | `dflash_context_floor = end - max(recent, ring)`；核新增 `min_positions`，低于 floor 的 ring slot 不可见；`floor = min(floor, frontier)` 夹紧 |
| 3 | 窗口满时 verify 溢出 | dflash 轮没有像普通/MTP 轮那样 `prepare_window`；`tokens=53250 vs 53248` | dflash 验证前 `prepare_window(frontier, frontier + verify_width)` |
| 4 | state fork 后 ring 写错槽 | ring lane 绑定 StateImage 物理槽；prefill 多 chunk 可能 fork | `dflash_state_slot` 跟随 destination，变化时 `copy_slot_from` 复制整槽 |
| 5 | 止损轮崩溃/诊断缺失 | bypass 路由到 ordinary batch 无资源 | 只保留 `KVMEM_DFLASH_BYPASS` 诊断；draftless 轮在 dflash 批内做 k=0 |

F1/F2 补充（audit）：

- floor 不能用 `recent_blocks * page`（默认只有 64 token），要用**环容量 2048**；
  否则 ring 只剩 3% 可见，接受率被钉在 30–39%。
- floor 必须有 `floor <= frontier` 不变式；否则截断/复用后全掩码，静默 0 接受。
- 页大小引用 `kPagedKVPageSize`，不要三处硬编码 64。

---

## 4. owl 的现状与真正的卡点

### 4.1 owl 的选择层是“零拷贝页表别名”

`select_view_step`：

```text
sel_pt  = 选中块（时间序）++ 尾部页(recall_pos/page ..= pos/page)
sel_len = selected_tokens + (pos + 1 - recall_pos)
```

- K 不做 re-RoPE；池里就是原位置算出的 post-RoPE K。
- 所有被选中的 key 都在 query 之前，因果掩码全可见；页表压缩不改变 dot product。
- 方案 A 在 owl 已实测：50% 物化率输出逐字一致；25% 仍切题。

### 4.2 卡点在 verify，不在 decode

`select_partial` 把下沉块写成 `PENDING_LOAD_PLACEHOLDER`。decode/prefill 有视图，
verify 没有：

```text
exec/spec.rs:
  bt_chain = session.block_table.clone()
  slots_f  = chain[p/page] * page + p%page
  bt8      = bt_chain × (depth+1)
  lens_f   = (pos+1)..=(pos+depth+1)
  fi_indices = bt_chain
```

占位符被核当页号 ⇒ CUDA 700，且污染整个 CUDA context（之后每个 turn 都炸）。
所以 `[kvmem] select_partial && [spec] depth > 0` 目前**拒启**。

### 4.3 owl 的 DFlash2 ring 本身不是问题

- owl 草稿环 = `pos % ring`，`draft_kv_len = min(fp, ring) + 8`；
- 65k 长上下文 DFlash2 实测 AL 2.955，说明**没有 ninfer 的 2048 envelope 短路**；
- 主模型选择层换块不影响 ring；ring 由 prefill/decode 的 DFlash encode 独立维护。

⇒ owl 要补的不是“草稿环”，而是**主模型 verify 的视图消费方**。

### 4.4 另一个隐性约束：FI 也不懂洞

`select_partial` 启动检查只关 XQA，没有关 FI。但 decode 的 `fi_indices` 仍取
`bt_chain`（含占位符）。所以 `select_partial` 档必须同时：

- `decode_arm = v2`（或补 FI 视图）；
- `verify_arm = v2`（或补 FI 视图）；
- `xqa` 已自动关。

在补视图之前，不要把 `select_partial` 与 `decode_arm=fi` 混用。

---

## 5. 方案 A（推荐先做）：视图化 verify

### 5.1 机制

对 verify 的每一行 `r`（逻辑位置 `p = pos + r`）：

```text
pages_r = select_view.pages ++ chain[recall_pos/page ..= p/page]
len_r   = select_view.tokens + (p + 1 - recall_pos)
slot_r  = chain[p/page] * page + p % page
```

然后：

```text
bt8[r]     = pages_r（零填充到 nb）
kv_lens[r] = len_r
kv_slots[r]= slot_r
pos[r]     = logical p      # RoPE 仍用逻辑位置
```

- 不需要 re-RoPE：K 已经是原位置 post-RoPE；视图只是压缩 key 列表。
- 不需要搬 KV：物理页仍是池里的选中页 + 尾部页。
- `bt8` 从 `[depth+1, nb]` 共享一张表变成**逐行不同表**；v2 核本来就吃
  `[T, nb]` + `kv_lens[T]`，形状不变。
- 新增显存：`8 × nb × 4B`；4955 页时约 **158 KiB**，可忽略。

### 5.2 必须同时处理的边界

1. **尾部页必须在设备且非占位符**
   - `p/page` 必须 `< chain.len()`；
   - `chain[p/page] != PENDING_LOAD_PLACEHOLDER`；
   - 否则先 stage-in 或结构化拒发本轮（不要发核）。
2. **验证宽度**
   - `depth+1` 行可能跨页；每行 `pages_r` 的尾部页数不同，必须逐行构造；
   - 验证前确保 `pos + depth + 1` 的页已在 session chain（`ensure_for_len`）。
3. **select_recent 覆盖**
   - `select_recent * page >= depth + 1`；当前 4 页 = 128 tok，depth 7 足够；
   - 若以后 depth/窗口变化，boot 检查 `workspace` 时要把 verify 宽度算进去。
4. **FI**
   - 第一版直接 `verify_arm=v2`，`decode_arm=v2`；
   - FI 的 `indices/indptr/last_len` 目前是单表，不是逐行表；要做 FI 视图需要
     额外的逐行契约，收益不值当，先不做。
5. **XQA**
   - 已自动关；不要回开。
6. **视图生命周期**
   - 视图态是 turn 域；verify 前后不能清；
   - 无视图（召回点之前）⇒ 走全量链，保持现有 dense 路径逐字不变。

### 5.3 为什么不需要 ninfer 的 floor（第一版）

ninfer 的 floor 是为“主 KV 物理换块 + ring 绝对位置”准备的；
owl 的 ring 是独立池，主 KV 换块不影响 ring 的最近 2048 行。
所以第一版可以**不加** `min_positions`，先验证：

- 重选后 AL 是否掉；
- prefix 复用/回滚后 ring 是否读到旧 token；
- 若掉，再加 `draft_floor`（用环容量，`floor <= frontier`，reset on reuse）。

这能避免一上来就改 sliding-window 核，缩小第一刀。

### 5.4 验收门

- **零 IMA/CUDA 700**：8k/32k/65k/131k/260k，各 1000+ decode 步；
- **AL**：同 prompt 下 `select_partial + dflash2` 的 AL 不低于
  `select_decode + dflash2` 的 80%，且显著高于 spec off；
- **decode**：高于 `select_partial + spec off`（当前 8k 40.2 t/s）；
- **输出**：固定 prompt 与 dense/spec-off 对照，前缀一致率 ≥ 60%（首刀口径）；
- **召回**：NIAH/agent 任务不退化（参考 ninfer 33/33）；
- **内存**：设备 KV 与 `select_partial + spec off` 同量级；
  额外只有 158 KiB 页表；
- **复用**：prefix cache / 回滚 / 多 turn 不毒化上下文。

---

## 6. 方案 B（备选/北向）：物理紧凑窗口

如果 A 的逐行页表/lens 契约在 FI/partial page/跨页上反复出问题，再上 B：

- 把选中块 **stage-in 到连续 B+R 窗口**（ninfer 的 `WindowMemoryBackend`）；
- `removed_tokens = logical - visible`；页表 `0..visible/page` 连续；
- verify 不需要视图，FI 也自然可用；
- 代价：store/session/prefix cache/重写路径都要改；每次重选要搬
  B+R 页（ninfer 64k 约 0.5–1.4s，256k 稳态 ~0.26s，优化后）。
- 这是 ninfer 的成熟路径，但比 A 大一个量级；owl 已有零拷贝视图，
  没必要先付这笔搬家费。

方案 C（verify 时临时 gather）不建议：每轮搬 B 页，成本不可接受；
若做持久 staging，本质就是 B。

---

## 7. 建议的施工顺序

1. **配置闸**：`select_partial` 档强制 `decode_arm=v2`、`verify_arm=v2`，
   并在 `--check` 里写明（FI/XQA 视图未实现）。
2. **抽 helper**：把 `select_view_step` 的“选中 + 尾部”逻辑抽成
   `view_for(pos) -> (pages, len)`；decode 与 verify 共用，避免两套语义。
3. **verify 视图化**：`bt8` 逐行、`kv_lens` 逐行、`slots_f` 保持物理槽；
   无视图时逐字走旧路径。
4. **尾页守卫**：占位符/越界/缺页 ⇒ 结构化拒发本轮；
   必要时先 stage-in。
5. **测试阶梯**：8k → 32k → 65k → 131k → 260k；每档先 spec off 对照，
   再 spec on；记录 AL/decode/IMA。
6. **可选 floor**：只有重选后 AL 明显掉才加；加则按 ninfer 的
   `max(recent, ring)` + `floor <= frontier` + reset。
7. **FI 视图**：如果 v2 verify 性能不够，再评估逐行 FI 表；否则维持 v2。

---

## 8. 风险清单

| 风险 | 说明 | 对策 |
|---|---|---|
| 逐行页表契约 | `bt8` 行长度不同、`kv_lens` 与页表容量不一致 | 单元门：构造 3 行跨页 view，逐行比对 attention 输出 |
| partial page | `v.tokens = pages*page` 可能含无效尾行 | 验证 `len_r` 与页内有效 token；必要时用 `visible_tokens` 夹 |
| tail 被下沉 | `select_recent` 太小 / 池满 | boot 校验 `select_recent*page >= depth+1`；verify 前 stage-in |
| FI 单表 | `fi_indices` 不支持逐行 | 第一版 v2；FI 另立 |
| ring 旧特征 | 重选/复用后接受率掉 | 先实测；必要时加 floor |
| GDN 状态 | verify 拒绝时 GDN 回滚 | 复用现有 GDN 快照/回滚；不要只回 KV |
| 上下文毒化 | 一次非法页号污染整个 CUDA context | 所有页号在发射前 host 校验；结构化拒发，不发核 |

---

## 9. 参考坐标

- ninfer 修复记录：`repos/ninfer-kvmem-actkv/docs/dflash2-kvmem-fixes-20261018.md`
- 审查/F1–F5：`.../docs/dflash2-kvmem-audit-20261019.md`
- 主 KV 窗口：`.../docs/architecture.md`、`.../docs/active-kv-tiering.md`
- 完整补丁：`.../docs/patches/0001-feat-qwen3_5-DFlash2-KVMem-window-recall-with-merged.patch`
- owl 现状：`docs/devlog/memory/kvmem-selection.md` §8.11/§8.14/§9；
  `crates/engine/src/exec/spec.rs` 的 verify 守卫；
  `crates/engine/src/exec/coverage.rs::select_view_step`。

> 最终判断：**同时使用是可行的，而且 ninfer 已经证明**。owl 的最短路径不是
> 照搬“物理紧凑窗口”，而是把 owl 已有的零拷贝视图补到 verify 上；先做 A，
> 用实测决定要不要升级到 B。
