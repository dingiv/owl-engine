# MTP 长上下文抗衰减 —— Windowed-MTP + 自适应深度 设计与施工

> 2026-10-21 · 设计稿 v1
> 工作树：`工作树 mtp-spec` @ `9bb7414`
> **本轮施工范围 = Phase 0（Windowed-MTP 读视图）+ T1a/T1b（自适应 depth）。**
> Phase 2（置信度截断）只锁接口/约束，等 T1 数据后再开工。
> 基线原始数据（本地未入库）：本地实测档(未入库)。

---

## 0. TL;DR

长上下文下 MTP 的衰减由两件事叠加：

1. **草稿侧 propose 成本随 ctx 增长**：MTP 草稿层对整条 MTP KV 做全注意力；
2. **深草稿尾部接受率衰减**：d7 的 pos7 从 8k 的 `.389` 掉到 64k 的 `.218`。

基线（单 3090 Ti，greedy，256 tok，单请求）：

| 域 | ctx | d3 净加速 | d7 净加速 | 备注 |
|---|---:|---:|---:|---|
| MATH | 8k | 2.29× | **2.68×** | d7 胜 |
| MATH | 32k | 2.15× | **2.26×** | d7 小胜 |
| MATH | 64k | **1.92×** | 1.86× | d3 反超 |
| DOCS | 6.7k | **1.62×** | — | 真实文本，AL 1.64 |
| DOCS | 29.7k | **1.38×** | — | AL 1.40 |
| DOCS | 56.8k | **1.28×** | — | AL 1.32；余量已薄 |

因此本轮先做 **Windowed-MTP**：只改草稿注意力的**读视图**（sink + 最近窗口），
target verify 仍全上下文，训练免费、目标分布不变；预计直接砍 `spec.propose.read`。
Phase 1 再做 d3/d7 自适应，解决“8k 吃 d7、64k 回落 d3”的交叉点。

---

## 1. 基线结论与边界

### 1.1 数字口径

- `AL = E[m]`：每轮**被接受的草稿数**；`tokens/round = E[m+1]`（另加 bonus）。
- `pos_i`：第 i 个草稿位置的接受率；由 `spec.m*` 直方图反推：
  `pos_i = count(m >= i+1) / rounds`。
- 计时：`spec.round / spec.verify / spec.propose / spec.propose.read`，
  `/debug/metrics` 前后差分。
- 采样：greedy（`temperature=0`，`top_p=1.0`）；`ignore_eos`；输出 256 tok；并发 1。
- MATH：随机数字词池；DOCS：工作区文档拼接的固定长文本（真实文本，非重复）。

完整表见本地实测档(未入库)。

### 1.2 关键读数

MATH d3：

- 64k `AL 2.545`、`tokens/round 3.545`、`round 40.65ms`、`verify 27.38`、`propose 12.03`、`propose.read 10.67`。
- 64k 时 verify 占 round **67%**，propose 占 **30%**。
- `propose.read` 从 8k 的 6.39ms 涨到 64k 的 10.67ms（+67%）。

MATH d7：

- 64k `AL 3.836`、`tokens/round 4.836`、`round 56.54ms`、`verify 28.53`、`propose 25.99`、`propose.read 22.68`。
- `pos7`：8k `.389` → 32k `.340` → 64k `.218`；`pos1` 仅 `.922→.909`。
- 8k/32k/64k 每 token：`7.46 / 9.21 / 11.69ms`；d3 对应 `8.65 / 9.65 / 11.47ms`。
- 交叉点约 **50k**。

DOCS d3：

- 6.7k `AL 1.643`、净加速 1.62×；29.7k `AL 1.398`、1.38×；56.8k `AL 1.324`、1.28×。
- `pos1`：`.776→.667→.649`；`pos3`：`.367→.278→.306`（尾部有噪声，但总体下行）。
- 这条曲线说明：**MATH 域会显著高估接受率**，生产长文本的衰减更接近 DOCS。

### 1.3 结论

1. 我们的 d3 在 64k 仍净正，和 vLLM issue #47602 的 n=6 在 16k 转负**不矛盾**：
   模型/深度/域不同；d7 的尾部衰减已经复现了同一形状。
2. 不能再只盯“接受率”：64k 的 verify 已占 2/3，MTP 侧可动的最大一块是
   `propose.read`（草稿全历史注意力），这正是 Windowed-MTP 的目标。
3. d7 不是“越深越好”：短 ctx 收益大，长 ctx 被 propose 成本与尾部拒绝吃掉；
   Phase 1 必须做 depth 选择，而不是把生产档固定成 d7。

---

## 2. 目标与非目标

### 2.1 本轮目标（Phase 0）

- 给 MTP 草稿增加 **sink + 最近窗口** 的 KV 读视图；target verify 完全不动。
- 只改读路径，不改写路径、不缩池、不改 MTP 权重、不加新 CUDA kernel。
- 默认关闭；通过配置显式开启，先做 A/B，不直接改生产档。
- 在 8k/32k/64k × MATH/DOCS 上量出 AL、propose.read、decode 的变化。

### 2.2 非目标

- 不做 MTP KV 的物理环形缓冲/显存回收（论文的 ring pool 是下一层优化；
  本轮只做页表视图，先把收益和接受率验证清楚）。
- 不做 target verify 的稀疏化/换核（verify 已经是 FI；那是另一条战线）。
- 不做采样域 rejection（本轮门禁仍以 greedy 恒等门为准）。
- 不做 tree draft、L-MTP、AdaMTP、post-training、FOLD。
- 不改 DFlash2 的 draft window（SGLang 的 `--speculative-dflash-draft-window-size`
  是另一族；本轮 `spec.mtp_*` 只服务 MTP，配到 DFlash2 上必须响亮拒）。

### 2.3 验收门（Phase 0）

正确性：

- `gpu_spec_identity_gate_27b`（f16 主池 + fp8 主池各一次）绿：
  spec 输出与 no-spec 逐位一致。
- `window=0` 与改动前的 propose 结果逐位一致（同 prompt/同 seed/同权重）。
- 窗口开启后无 CUDA 错误、无图捕获失败、无 OOB/NaN。

性能（以 64k 为主，8k 为回归门）：

- `propose.read` 至少下降 **30%**（d7）/ **20%**（d3）。
- 64k decode 至少提升 **10%**；8k decode 回退不超过 **2%**。
- AL 不低于 full-context 的 **95%**（同域、同 ctx、同 depth）。
- 若 64k 收益 <5% 或 AL 掉 >5%，先扫 W=8192/16384；仍不达标则 Phase 0 不转正，
  直接进入 Phase 1（自适应 depth）评估。

---

## 3. 方案总览

### 3.1 两条轴

```text
                  ┌─ Phase 0：Windowed-MTP（降草稿读成本）
MTP 长上下文抗衰减 ┤
                  └─ Phase 1：自适应 depth（短 ctx 吃 d7，长 ctx 回 d3）
                       └─ Phase 2：置信度截断（可选，裁低置信尾部）
```

### 3.2 为什么先做 Windowed-MTP

- 论文（arXiv 2607.21535）在 1M/d7 上给出 per-step cost −28%~−44%、AL ±10%；
  机制就是 draft block table 的 `[sink]‖[recent]` 视图，**不需要新 kernel**。
- 我们的 v2 分页注意力已经有“页表别名 + `kv_lens` 视图”的硬门：
  `crates/models/src/layers/attention.rs::gpu_paged_view_aliases_selected_pages`
  已证明“选中的页按时间序拼表 + `kv_lens=W`”与物理搬移逐位等价。
- 我们的 MTP 三个 propose 路径（首轮/轮末 eager/图步）都显式构造
  `bt` / `bt8` / `slots` / `lens`，改读视图的落点清楚，不需要碰模型层。
- 基线已经显示：64k 时 propose 占 30%（d3）/46%（d7），且 propose.read 是其中
  唯一随 ctx 线性增长的项；window 是直接命中。

### 3.3 前车之鉴

2026-10-21 曾试过 **window-only（4096/6144，无 sink）**：

```text
8k 固定 prompt：m0 6/76 → 97/144（W=4096）/ 98/146（W=6144）
decode 75.4 → 45.8 / 41.3 t/s
```

该实验**没有 attention sink**，不能作为 Windowed-MTP 的否证。本轮必须：

- `sink=64`（StreamingLLM 标准量级；按页对齐）；
- 保留当前块；
- 只对“足够的 ctx”启用（短 ctx 直接 full，避免无意义改写）；
- 先 A/B，不直接改生产档。

---

## 4. Phase 0：Windowed-MTP 设计

### 4.1 配置面

在 `[spec]` 增加两个字段（`SpecCfg`，`serde(default)`，旧档无需补写）：

```toml
[spec]
# MTP 草稿注意力的滑动窗口（token；0 = 关闭，走全上下文，逐字节旧行为）。
# 只作用于 draft 读；target verify 始终全上下文。
mtp_window = 0
# 草稿注意力 sink：最老的 N 个 token 始终可见（token；按页向上取整）。
# 仅 mtp_window > 0 时生效；推荐 64。
mtp_sink = 64
# 窗口生效的最小绝对 ctx（token；0 = 不设额外门槛）。
# 8k prompt + 256 输出会跨过 8192，实测取 12288 才真正保 full。
mtp_window_min_ctx = 0
```

语义：

- `mtp_window = 0`：完全关闭，所有路径走原 full 表；这是默认值。
- `mtp_window > 0`：仅当 `spec.draft` 解析为 **MTP** 且 `depth > 0` 时生效。
- `mtp_sink` 是**最老的前缀**，不是“最近窗口的一部分”；最终可见集 =
  `[0, sink_eff) ∪ [recent_start, row]`，其中 `recent_start` 按页向下对齐。
- `sink_eff = ceil(mtp_sink / page) * page`，`page` 来自 `StatePool::geom.page`
  （当前 32）。`mtp_window` 也按页粒度取整；启动日志打印 effective 值，
  不静默。
- 若 `mtp_window > 0` 且 `mtp_sink >= mtp_window`：启动拒（避免“sink 吃掉窗口”）。
- 若 `mtp_window > 0` 但模式不是 MTP（Off/DFlash2/Dumb）：启动拒，不静默忽略。
- 若 `mtp_window > 0` 但 pool 缺 `attn_v2_scratch` / `verify_attn_v2_scratch`：
  启动拒（窗口读依赖 v2 伪序列臂；缺了会落 chunked/full lens 语义，不能混）。

> 不在 `[spec]` 里复用 DFlash2 的 `draft_window`；两族语义不同，配置面必须分开。

### 4.2 可见集定义

记号：

- `row` = 当前 query 的绝对位置（0-based，已提交 token 的位置）。
- `tokens = row + 1`（含当前 query 自身）。
- `page` = 池页 token 数（32）。
- `full_bt[p]` = MTP 链第 p 页的物理块号（会话 `mtp_block_table`，完整链）。

```text
sink_eff   = ceil(sink / page) * page
if window == 0 || tokens <= min_ctx || tokens <= sink_eff + window:
    visible = [0, tokens)                     # full
else:
    recent_start = max(sink_eff, tokens - window)
    recent_start = floor(recent_start / page) * page
    if recent_start <= sink_eff:
        visible = [0, tokens)                 # full（窗口覆盖到 sink）
    else:
        visible = [0, sink_eff) ∪ [recent_start, tokens)
```

页表（逻辑顺序 = sink 页，再 recent 页）：

```text
blocks = full_bt[0 .. sink_eff/page)
       ‖ full_bt[recent_start/page ..= row/page]
```

去重：

- 若 `recent_start <= sink_eff`：直接 full；
- 否则 sink 页与 recent 页不重叠（recent_start 已按页对齐）。

`kv_lens`（v2 的每行可见长度）：

```text
visible_len = sink_eff + (tokens - recent_start)
```

注意：

- `visible_len` 是**压缩后逻辑长度**，不是绝对位置；
- 页表里的 K/V 已经带绝对 RoPE，重排页表只改变“读哪些 key”，不重新旋转；
- 当前页必在表内（`row/page`），`visible_len` 截断当前页中 row 之后的槽；
- `slots` 仍用**完整 `full_bt`** 计算物理写入槽；window 只影响读，不影响写。

示例（`sink=64`, `window=4032`, `page=32`）：

```text
row=8000, tokens=8001
sink_eff=64, recent_start=max(64, 8001-4032)=3969 → 3968
blocks = pages[0..2] + pages[124..=250]
visible_len = 64 + (8001-3968) = 4097
```

### 4.3 数据流 / 调用面

```text
mtp_chain_bt(cover)
  └─ full_bt：完整 MTP 链（写路径/槽换算的唯一真源）
       ├─ MtpWindowPlan::row(row) → { blocks, visible_len }（只读视图）
       │    ├─ propose_first：每步 decode 的 KvBuffers.block_tables / kv_lens
       │    ├─ propose_round：extend 的 v2_bt8 + lens；chain 的 block_tables + lens
       │    └─ propose_graph_step：v2_bt8 [m+1,nb] + lens [m+1]
       └─ slots：始终用 full_bt[p/page]*page + p%page（写槽）
```

**不动**：

- `mtp_chain_bt` / `ensure_for_len` / `write_bt_mtp` 的完整链表；
- `prefill_extend`：它是**写**路径，必须写全链，不能 window；
- `MtpPredictor::propose` / `propose_ext` 的模型层逻辑（只吃调用方给的
  `block_tables` / `kv_lens`）；
- target verify 的 `block_table` / `kv_lens` / FI 表。

### 4.4 `MtpWindowPlan`（新模块）

位置：`crates/engine/src/exec/mtp_window.rs`（`exec/mod.rs` 挂 `mod mtp_window;`）。
职责：纯函数 + 小结构，不碰设备、不读全局、不做 I/O。

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MtpWindowCfg {
    pub window: usize,
    pub sink: usize,
}

impl MtpWindowCfg {
    pub fn active(self) -> bool { self.window > 0 }
}

pub(crate) struct MtpRowView {
    /// 压缩页表（物理块号，sink 页在前，recent 页在后）
    pub blocks: Vec<u32>,
    /// v2 的可见长度（不是绝对 pos）
    pub visible: u32,
}

pub(crate) struct MtpWindowPlan<'a> {
    full_bt: &'a [u32],
    page: usize,
    cfg: MtpWindowCfg,
}

impl<'a> MtpWindowPlan<'a> {
    pub fn new(full_bt: &'a [u32], page: usize, cfg: MtpWindowCfg) -> Result<Self>;
    pub fn row(&self, row: usize) -> Result<MtpRowView>;
    pub fn rows(&self, base: usize, n: usize) -> Result<Vec<MtpRowView>>;
}
```

错误面：

- `page == 0` → `ModelError::EngineInvariant`（不 panic）；
- `row / page >= full_bt.len()` → `EngineInvariant`，带 `row/page/full_bt.len()`；
- `cfg.window > 0 && cfg.sink >= cfg.window` → 在 boot 面拒，不在热路径；
- `visible > u32::MAX` → 不可能（ctx 上限），仍用 `u32::try_from` 显式转换。

热路径每轮最多 `m+1 + depth-1` 个 view（d7 最多 14 个），每个 `Vec` 长度
≤ `sink_pages + window_pages + 1`（64k 时约 129），分配量可忽略。

### 4.5 三个 propose 路径的接线

#### A. `propose_graph_step`（生产热路径，优先改）

现状：

- `bt = ensure_for_len(...)` 完整链；
- `write_bt_mtp(face, &bt)` 写完整表；
- `bt8_f` 每行复制同一张 `bt`；
- `lens_f = pos+1..=pos+m+1`（绝对）；
- chain 每步 `lens = p+1`（绝对）。

改法：

1. `let plan = MtpWindowPlan::new(&bt, page, cfg)?;`
2. `let ext_views = plan.rows(pos, m+1)?;`
3. `bt8_f`：每行填 `ext_views[r].blocks`，其余补 0；
4. `lens_f`：每行填 `ext_views[r].visible`；
5. `slots_f` 不变（仍用 full `bt` 算物理槽）；
6. chain 第 j 步：`let v = plan.row(fp+j-1)?;`
   - `c{j}_lens` = `v.visible`；
   - `c{j}_slots` 仍用 full `bt`；
   - **图内 chain 是 decode T=1 的 v2，读的是 `KvBuffers.block_tables`，
     不是 `v2_bt8`**（后者只服务 extend/verify 的多行伪序列）。当前
     `plan_propose_family` 里 chain 的 `block_tables: bt.clone()` 是
     `bt_mtp` 持久叶子（完整表）⇒ 必须给 chain 每步单独的视图表，否则
     chain 仍读完整链，window 只对 extend 生效。
7. `write_bt_mtp` 仍写完整表（写路径/持久槽）。

> 关键点：`propose_ext` 的 `chain` 参数是 `KvBuffers`，其中 `block_tables`
> 在**图态**是持久 `bt_mtp` 叶子；窗口 chain 需要把每步的视图表作为
> **图输入槽**，不能直接把 host 视图塞进持久叶子（图会烘死旧表）。
> 实现时二选一：
> - 方案 G1（推荐）：给 chain 每步增加 `c{j}_bt` 输入槽（`[1,nb]`），
>   `KvBuffers.block_tables` 指向该槽；
> - 方案 G2：把 chain 的视图表也压进同一个 `v2_bt8` 的额外行，按 chain
>   序号索引；需要改 `propose_ext` 读取面。
>
> 先做 G1：图输入多 6 个 `[1,nb]` 槽（d7），每轮上传 ~6×2304×4B ≈ 55KB，
> 可忽略；但 capture 槽位/装填要一次改干净。

#### B. `propose_round`（eager fallback）

现状：

- `bt` 完整链；`extend_kv.block_tables = bt_t`；`lens_x` 绝对；
- `bt8_f` 同一张 `bt` 填所有行；
- chain `KvBuffers.block_tables = bt_t`；`kv_lens = p+1`。

改法：

- `plan.rows(pos,m+1)` → `bt8_f` / `lens_x`；`extend_kv.block_tables` 保持
  full `bt_t`（只在 `extend_v2=None` 的 fallback 使用，此时强制 full）；
- chain 每步 `KvBuffers.block_tables` 用该步 view 的 `[1, blocks]` 表；
  `kv_lens=visible`；
- `slots` 不变；
- 若 `extend_v2` / `attn_v2` 任一为 None：整轮走 full（绝不半 window）。

#### C. `propose_first`（prefill 后首轮）

现状：

- 每步 `KvBuffers` 用 `bt_t` 完整链；`kv_lens=p+1`。

改法：

- 每步 `plan.row(p)` → `KvBuffers.block_tables = [1, blocks]`，`kv_lens=visible`；
- `slots` 仍用 full `bt`；
- 若 `attn_v2=None`：full。

### 4.6 图/缓冲影响

- `propose_graph_step` 的 `v2_bt8` 形状不变 `[m+1, nb]`；只是内容从“同一张
  全链”变成“每行一张窗口表 + 补零”。
- `c{j}_lens` 从绝对长度变为可见长度；必须在 `propose_ext` 的 chain
  `KvBuffers` 中同步。
- chain 的 `block_tables` 若走 G1，需要新增 `c{j}_bt` 输入槽（`[1,nb]`），
  并在 `graph_fleet::plan_propose_family` 的 `pins` / `PlanCtx` 里登记；
  形状固定，不需要重捕图族数量。
- `write_bt_mtp` 仍写完整 `bt_mtp` 叶子；`bt_mtp` 叶子的消费者若是
  `propose_ext` 的 chain，需改为 `c{j}_bt` 输入槽，否则 window 不生效。
- `fold` 图不动；`verify` 图不动。

### 4.7 fallback 规则

| 场景 | 行为 |
|---|---|
| `mtp_window=0` | full，逐字节旧行为 |
| ctx 太短（`tokens <= sink_eff + window`） | full，不做无意义压缩 |
| `extend_v2` / `attn_v2` 缺 | full（但 boot 已对 `mtp_window>0` 拒缺 v2） |
| `draft` 非 MTP | boot 拒（配置错误，不静默） |
| `depth=0` | boot 拒（window 无消费者） |
| TP2 | 每 rank 用自己的 `bt`/page，表视图各自构造；先单卡验收，TP2 后测 |
| 采样域 | 本轮不验收；window 只影响 draft 提议，target 采样/verify 不动 |

### 4.8 日志与指标

boot：

```text
[spec] MTP window: window=4032 sink=64 page=32 => sink_pages=2 window_pages=126 (effective sink=64, window=4032)
```

每轮（debug 门控，前 3 轮）：

```text
[mtp-window] pos=... m=... rows=[(visible, blocks), ...] full_fallback=...
```

metrics（低基数）：

- `spec.mtp.window.rounds`：窗口实际生效的轮数；
- `spec.mtp.window.full`：因短 ctx/fallback 走 full 的轮数；
- `spec.mtp.window.visible_tokens`：每轮可见 token 总数（counter）；
- `spec.mtp.window.build`：view 构造耗时（timer，诊断用）；
- 既有 `spec.propose.read` / `spec.round` / `spec.verify` 保持原义。

### 4.9 错误与边界

- **绝不**在写路径使用 window：`prefill_extend`、`ensure_for_len`、
  `write_bt_mtp` 全链。
- **绝不**让 `slots` 用 window 表：写槽必须全链物理槽。
- **绝不**在 `extend_v2=None` 时传 window lens（会落 chunked prefill 的
  绝对长度语义）。
- `sink_eff` 可能大于 `mtp_sink`（页对齐）；日志必须打印 effective 值。
- `recent_start` 页对齐向下，`visible_len` 必须包含实际读取的页内 token；
  不能把页对齐的额外 token 算漏（否则 attention 少读）。
- 当前页在 `row/page`；`full_bt` 必须已 `ensure_for_len(row+1)`；
- `row=0`、`sink=0`、`window` 大于 ctx 都要有单测。

### 4.10 文件改动清单

| 面 | 文件 | 改动 |
|---|---|---|
| 配置 | `crates/shared/src/config/mod.rs` | `SpecCfg.mtp_window/mtp_sink` + 默认值 + 组合校验 |
| 配置模板 | `profiles/full.toml` | 两键 + 注释（默认 0/64） |
| 引擎旋钮 | `crates/engine/src/engine.rs` | `DraftCfg` 两字段；`from_config` 搬运；`SpecBootInput` 透传；boot 期 v2 检查 |
| spec 运行态 | `crates/engine/src/spec_runtime.rs` | `SpecRuntime.mtp_window`；模式/深度/资产校验；boot 日志 |
| 窗口纯函数 | `crates/engine/src/exec/mtp_window.rs`（新） | `MtpWindowCfg` / `MtpRowView` / `MtpWindowPlan` + 单测 |
| 执行面 | `crates/engine/src/exec/mod.rs` | `mod mtp_window;` |
| 草稿路径 | `crates/engine/src/exec/draft.rs` | `propose_first` / `propose_round` / `propose_graph_step` 接线 |
| 图面 | `crates/engine/src/graph_fleet.rs` | chain 的 `c{j}_bt` 输入槽（G1） |
| 文档 | `config.md` / `profiles/README.md` | 配置说明 + 验证档 |
| 本卷 | `mtp-longctx-windowed.md` | 本文 |

> `profiles/qwen38_27b-3090-w4a16-mtp.toml` 已写入 `mtp_window=4032` / `mtp_sink=64`
> （Phase 0 首轮实测后转正；`full.toml` 只保留 schema 缺省 0/64）。A/B 回退用
> `--set spec.mtp_window=0`。

### 4.11 测试计划

**单元（CPU，快）**

- `window=0`：任意 `row` 返回 full 表 + `visible=row+1`。
- `tokens <= sink+window`：full，不压缩。
- 长 ctx：sink 页 + recent 页顺序、无重复、当前页在表内。
- `sink=0`：recent-only。
- `sink`/`window` 非页倍数：effective 页对齐、`visible` 与实际页内 token 对账。
- `row` 跨页 / `row=0` / `window>tokens`。
- `full_bt` 不足：返回 `EngineInvariant`，不 panic。
- 若 `visible_len` 与“物化页表”不一致：测试直接失败（用 host 对照）。

**引擎/图**

- `window=0` 的 draft 输出与改前 golden 逐位一致（同 prompt/seed）。
- `window>0` 的 propose 图捕获成功（d3/d7 全 m 桶）。
- 图态 `v2_bt8` 每行表与 host `plan.row` 对账（可通过 debug 探针导出）。
- `write_bt_mtp` 仍是完整表；`mtp_block_table` 长度不变。
- `prefill_extend` 在 window 开启时仍写全链（首轮 propose 的 seed 依赖它）。

**E2E（27B，单 3090 Ti）**

- `OWL_E2E_ONLY=gpu_spec_identity_gate_27b`：
  - 主池 f16 / fp8 各一次；
  - spec on（window）≡ spec off（逐位）。
- 性能 A/B（同二进制、同卡、同 prompt）：
  - d3/d7 × 8k/32k/64k × MATH/DOCS；
  - window=0 vs 4032（若失败再 8192/16384）；
  - 每点 2 轮（64k 至少 1 轮 + 复跑一次）；
  - 记录 `AL`、`pos_i`、`round/verify/propose/read`、decode、TTFT。

### 4.12 Phase 0 施工顺序

1. 配置字段 + `full.toml` 注释 + `--check/--print-config` 可见。
2. `MtpWindowPlan` + 单测（先不接线）。
3. `propose_graph_step` 接线（生产热路径）+ d7 64k 小 A/B，先看
   `propose.read` 是否下降、AL 是否保持。
4. `propose_round` / `propose_first` 接线（eager/首轮），保证 window=0 逐位旧行为。
5. chain 的 `c{j}_bt` 图输入槽（G1）+ 图捕获验证。
6. 指标/日志 + 身份门 + 文档。
7. 若 4032 不达标，扫 W=8192/16384；仍不达标则 Phase 0 不转正，进入 Phase 1。

### 4.13 Phase 0 首轮实测（2026-10-21，实现后）

口径同基线：单 3090 Ti，greedy，256 tok，单并发，MATH 词池；
`window=4032 sink=64`，`window=0` 为同二进制 full 对照。

| 点 | decode full → win | AL full → win | propose.read full → win | round full → win |
|---|---:|---:|---:|---:|
| MATH 8k（win 1 轮） | 113.60 → **114.99** | 2.688 → **2.718** | 6.39 → **6.08** | 31.91 → 31.87 |
| MATH 32k（2 轮） | 102.46 → **106.69** | 2.693 → **2.601**（−3.4%） | 8.22 → **6.09** | 35.64 → 33.50 |
| MATH 64k（2 轮） | 86.62 → **93.60** | 2.545 → **2.424**（−4.8%） | 10.67 → **6.08** | 40.65 → 36.12 |
| DOCS 56.8k（1 轮） | 58.58 → **69.39** | 1.324 → **1.574**（+18.9%） | 11.18 → **6.08** | 39.69 → 36.84 |

读数结论：

- `propose.read` 在 32k/64k 被压到 **~6.1ms 平台**，不再随 ctx 线性增长；
  64k 降幅 **43%**。
- MATH 64k decode **+8.1%**，AL 只掉 **4.8%**（刚好在 95% 门内）；
  DOCS 56.8k 反而 **AL +18.9% / decode +18.5%**（window 去掉了远距离稀释）。
- 32k MATH AL −3.4%、decode +4.1%；8k 基本持平（窗口只覆盖一半，收益小）。
- 短 ctx（512）仍走 full；decode 116.89 vs 117.85（噪声内），无回归。

恒等门（`OWL_E2E_ONLY=gpu_spec_identity_gate_27b`，长 prompt 重复到 >4k ctx）：

- `OWL_GATE_MTP_WINDOW=0`（f16 主池）：PASS；
- `OWL_GATE_MTP_WINDOW=4032`（f16 主池）：PASS；
- `OWL_GATE_MTP_WINDOW=4032 + OWL_GATE_FP8=1`（fp8 主池）：PASS。

**T0 补测（W=6144/8192，2026-10-21/22）**：

| 点 | 4032 | 6144 | 8192 |
|---|---:|---:|---:|
| MATH 64k decode | **93.60** | 92.79 | 93.77 |
| MATH 64k AL | 2.424 | 2.401 | 2.424 |
| DOCS 56.8k decode | **69.39** | 62.53 | 63.82 |
| DOCS 56.8k AL | **1.574** | 1.304 | 1.324 |

MATH 三点基本持平；DOCS 4032 明显最优 ⇒ **保持 4032/64**，窗口不是越大越好。
测量脚本已收编 `tools/mtp_curve.py`。当前数据 = Phase 0 + T0 收尾；生产 MTP 档
已写 `mtp_window=4032` / `mtp_window_min_ctx=12288`，A/B 回退用
`--set spec.mtp_window=0`。

---

## 5. Phase 1：自适应 depth（T1a 已落地）

### 5.1 事实

- d7 在 8k/32k 优于 d3，64k 被 d3 反超；交叉点约 50k。
- d7 的 `pos7` 在 64k 只剩 `.218`，但 `pos1` 仍有 `.909`；不是“草稿全坏”，
  而是**尾部不值得验**。
- 固定 depth 的两种档都浪费：d3 在 8k 少赚 17%，d7 在 64k 少赚 3%。

### 5.2 方案轮廓

- 调度按 **ctx + 滚动 AL** 选 depth：
  - 短 ctx / 高 AL：d7；
  - 长 ctx / 低 AL：d3；
  - 用 EMA + hysteresis，避免抖动（SGLang `adaptive_spec_params.py` 同构）。
- 每个候选 depth 需要自己的：
  - `verify` 图（T=depth+1）；
  - `propose` 图族（链步数/输出宽度由 depth 定形，**不能共用**）；
  - `fold` 桶（烘焙了本 depth verify 的 `r{i}` 输出块，**不能共用**）。
- **禁止在线重捕图**；所有候选深度 boot 期预捕获，切档只换引用。
- 关键施工事实：只补 verify 一张图而共用 propose/fold 是**假自适应**——
  d3 轮仍付 d7 的链步/输出宽度，fold 还会读到另一张 verify 图的陈旧记录。

### 5.3 T1a 施工结果（2026-10-21）

配置面（缺省关，旧档逐字节不变）：

```toml
[spec]
depth = 7                 # 上限
depth_min = 3             # 长 ctx 下限
adaptive_depth = true
depth_switch_ctx = 49152
adaptive_min_tpr = 3.0    # T1b 两连低窗门（tokens/round）
```

结构：

- `SpecRuntime::depth_candidates()` = `[depth_min, depth]`；
  `depth_for_pos(pos)` 在 `pos >= switch_ctx` 时回下限。
- `GraphFleet.depths: Vec<DepthGraphs>`：每档独立的
  `verify` + `fold[0..depth]` + `propose[0..=depth]`；
  face 预算从单档 16 → 两档 24（d3+d7；含 DFlash2 备用面）。
- `StepAction::SpecRound { depth }` 显式携带本轮 active depth；
  verify/fold/propose 全部按它选图，不在热路径读全局状态。
- 两档 verify 捕获失败 = **boot 硬失败**（adaptive 档不允许半档回退）。

首轮实测（单 3090 Ti，greedy，256 tok，window 4032/64）：

| 点 | active | decode | AL | propose.read | 对照 |
|---|---:|---:|---:|---:|---|
| MATH 8.1k | d7 | **139.18** | 5.023 | 14.47 | 静态 d7 full 132.79 / 4.789 |
| MATH 31.9k | d7 | **110.41** | 3.981 | 13.94 | 静态 d7 full 107.80 / 4.300 |
| MATH 64.4k | d3 | **92.93** | 2.442 | 6.25 | 静态 d3 window 93.60 / 2.424 |
| DOCS 29.7k | d7 | 62.42 | 1.796 | 13.82 | 静态 d3 full 66.09 / 1.398 |
| DOCS 56.8k | d3 | **69.41** | 1.559 | 6.20 | 静态 d3 window 69.39 / 1.574 |

**8k 窗口门槛**：`mtp_window_min_ctx=12288`（8k prompt + 256 输出会跨过 8192，
阈值取 8192 仍会压到尾部）；修复后 8k d7 从 126.60/AL 4.521 回到
139.18/AL 5.023。

**域漂移**：DOCS 29.7k 的 d7 AL 1.796 > d3 1.398，但 d7 的 propose/verify 成本
更大 ⇒ decode 62.42 < 静态 d3 66.09。单靠 ctx 阈值无法同时赢 MATH 32k 与
DOCS 29.7k；静态档不能直接转正，下一步必须做 T1b 的 AL/成本门。

恒等门：`OWL_GATE_SPEC_DEPTH=3` 与 `=7` 各一次，均 PASS（window=4032）。
未测：采样域、TP2、GGUF MTP；16k 自动档未测。

### 5.4 T1b v2：两连低窗 + 可配门（2026-10-22）

静态 ctx 阈值在 DOCS 29.7k 误选 d7（62.42 < 静态 d3 66.09），因此加了
**只降不升**的 turn 级观察窗：

- 每轮累计 `tokens/round = m+1`；每 8 轮评估一次。
- 窗口均值 `< adaptive_min_tpr`（缺省 3.0）记一个低窗；
  **连续 2 个低窗**才 `adaptive_downshift = true`，后续轮固定 `depth_min`。
- 低窗后出现高窗 ⇒ 低窗计数清零；避免 16k 冷启动瞬态误切。
- 指标 `spec.adaptive.down`；日志 `spec-adaptive pos/depth/m/window/tpr/low_windows`。

实测（单 3090 Ti，greedy，256 tok）：

| 点 | T1b v2 | 对照 |
|---|---:|---|
| DOCS 29.7k | **68.91** / AL 1.68 | 无门 62.42；静态 d3 66.09 |
| MATH 16k | **105.23** / AL 3.596 | 静态 d7 105.46（不误切） |

DOCS 29.7k 的低窗是 2.88/2.00，约 16 轮内降档；16k 首窗低、次窗高，
低窗计数清零，保持 d7。**4.0 不需要**：两连低窗规则下它不会更早，反而放宽
低窗判据。

后续：采样/TP2 补测；再上 EMA + hysteresis（SGLang 同构）。多请求/BS 分档后置。

---

## 6. Phase 2：置信度截断（T2 立项）

> 设计与施工卷：`mtp-confidence-cutoff.md`（2026-10-22）。

- 参考 DSpark（arXiv 2607.05147）的 confidence-scheduled verification：
  按 prefix survival 裁 verify 长度；SGLang 已有 block accept estimator。
- MTP 当前图只输出 `drafts`；需要新增每步 confidence（top1-top2 margin 或
  entropy）输出，并把 verify 图做成至少两档（如 4/8 行）。
- 单请求下收益主要是“少验低置信尾部”；大 batch 下才是吞吐杠杆。
- 本轮不做；Phase 0/1 后再评估。

---

## 7. 风险与开放问题

> T3 前置（MTP 草稿 KV 追赶）见 `mtp-draft-kv-catchup.md`。

| 风险 | 影响 | 缓解 |
|---|---|---|
| 4032 窗口仍伤 AL（尤其 MATH 算术） | 净收益被吃掉 | 先 4032，再扫 8192/16384；sink 必留；AL 门 95% |
| 页表/`lens` 错位 | 静默错值/文本漂移 | 单测 + `gpu_paged_view_aliases_selected_pages` 同构门 + 恒等门 |
| chain 的 `block_tables` 仍指向完整叶子 | window 只对 extend 生效、chain 仍全读 | G1 新增 `c{j}_bt` 输入槽；图态/ eager 双路对账 |
| `extend_v2=None` 时传 window lens | chunked prefill 读错 | boot 拒缺 v2；运行时显式 full fallback |
| 64k 以上窗口收益变小/接受率掉 | Phase 0 不转正 | 先 64k 定谳；128k 另案（当前 128k 受 VRAM hard budget 限制） |
| 采样域行为不同 | 生产温度下收益/质量未知 | 本轮只报 greedy；采样另立门 |
| TP2 双 rank 表不同步 | 错值/挂死 | 单卡先验收；TP2 后测，两 rank 各自构造同一 `bt` 视图 |

开放问题：

1. `sink` 是否要按 **页** 对齐（当前设计是）；论文用 64 token，正好 2 页。
2. `window` 是否要包含当前块（当前块通过 causal 必在，无需计入 W）。
3. 64k 以上是否要改成“sink + 多个 recent 段”（当前单段）；先单段。
4. 是否需要把 window 写进 profile 默认值；Phase 0 数据前保持 0。

---

## 8. 复现命令（Phase 0）

```bash
# 编译（本工作树缺 kernels venv 时借用主树）
GDN_TRITON_PY=/path/to/owl/crates/kernels/.venv/bin/python \
  cargo build --release -p server --features perf

# 预检
./target/release/server -p profiles/qwen38_27b-3090-w4a16-mtp.toml \
  --set spec.mtp_window=4032 --set spec.mtp_sink=64 --check

# 启动（A/B 臂）
./target/release/server -p profiles/qwen38_27b-3090-w4a16-mtp.toml \
  --set runtime.max_seq=73728 --set pool.pool_tokens=73728 \
  --set spec.mtp_window=4032 --set spec.mtp_sink=64

# 恒等门
OWL_E2E_ONLY=gpu_spec_identity_gate_27b \
  cargo test -p owl-engine --features e2e --test engine_e2e

# 曲线（临时脚本；Phase 0 可收编为 tools/mtp_ctx_curve.py）
python3 /tmp/mtp_curve.py \
  --out <out-dir>/spec-window \
  --length 65536 --rounds 2
```

---

## 9. 附：社区依据

- **Windowed-MTP**：arXiv 2607.21535，NVIDIA；sink=64 + W=4032；
  draft block table 视图；1M/d7 per-step −28%~−44%，AL ±10%。
- **SGLang adaptive spec**：`sglang`(外部存档)
  （EMA + hysteresis + 预捕获图）。
- **SGLang DFlash draft window**：`dflash_worker_v2.py` 的
  `--speculative-dflash-draft-window-size`（recent-only；本轮 MTP 不复用其配置）。
- **SGLang DSpark block accept estimator**：
  `dspark_components/dspark_block_accept_estimator.py`（Phase 2 参考）。
- **FOLD**：arXiv 2610.20392（外部 block drafter，不适用于 MTP）。
- **L-MTP / AdaMTP**：arXiv 2505.17505 / 2608.00434（训练线，不作本轮）。
- **vLLM issue #47602**：Qwen3.6-27B MTP n=6 长上下文净负实证（问题成立，
  无实现方案）。

---

## 10. 下一阶段交接

Phase 1/2、W 扫点、MTP 专用 C7、采样 rejection、ring buffer、tree、头量化、
GGUF/VLM/TP2/128k 验证矩阵与操作纪律，见：

`mtp-longctx-handoff.md`

下一会话先读该交接文档，按 T0 → T1 → T3 → T2 的顺序施工。
