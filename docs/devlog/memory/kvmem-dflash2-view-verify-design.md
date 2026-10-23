# KVMem 部分驻留 × DFlash2：视图化 verify 设计（2026-10-23）

> 状态：设计定稿，准备施工。
> 前置研究：`docs/devlog/memory/kvmem-dflash2-crosscheck-20261023.md`。
> 目标：让 owl 的 `[kvmem] select_partial = true` 与 `[spec] depth = 7`（DFlash2）
> 同时工作；不搬 KV、不做 re-RoPE，保留 owl 现有零拷贝页表别名架构。

---

## 1. 需求与边界

### 1.1 要解决什么

`select_partial` 把下沉块写成 `PENDING_LOAD_PLACEHOLDER`（`u32::MAX`）。
decode/prefill 已有视图消费方，verify 直接读会话链：

```text
bt8      = 会话链 × (depth+1)
kv_slots = 会话链[page] * page + offset
kv_lens  = (pos+1) ..= (pos+depth+1)
```

占位符被核当页号 ⇒ CUDA 700 + 整个 CUDA context 毒化。
当前配置层直接拒启 `select_partial && spec.depth > 0`。

### 1.2 本设计的目标

- 给 verify 增加**逐行视图页表 + 视图长度**；
- 只在“会话链真的出现洞”时启用视图；无洞时逐字走旧 dense 路径；
- 不改变 decode/prefill 的现有视图语义；
- 不引入物理 KV 搬迁、不 re-RoPE；
- 第一版 verify 走 v2；FI/XQA 的视图化不在本刀。

### 1.3 非目标

- 不做 ninfer 式物理紧凑窗口（备选方案 B，另立）；
- 不做 FI 逐行页表（`fi_indices` 单表契约不兼容）；
- 不做 draft ring 的 `min_positions` 掩码（owl ring 独立，先实测是否需要）；
- 不改选择器打分/召回策略；
- 不改 prefix cache / store 的下沉策略。

---

## 2. 总体方案

### 2.1 核心观察

owl 的视图是**零拷贝页表别名**：

```text
sel_pt = 选中块（时间序）++ 尾部页(recall_pos/page ..= pos/page)
sel_len = selected_tokens + (pos + 1 - recall_pos)
```

K 已经是原位置 post-RoPE；所有选中 key 都在 query 之前，因果掩码全可见。
因此“压缩页表”不改变 dot product，也不需要 re-RoPE。

verify 只是把同一条视图规则**逐行**应用。第一版按行生成不同页表；实测
DFlash2 接受行出错后改成**同一张页表（最后一行，尾部最长）+ 逐行 lens**，
与 dense verify 的“8 行同表 + 逐行 kv_lens”同构：

```text
last = pos + rows - 1
pages = selected_pages ++ chain[recall/page ..= last/page]   # 所有行共用
for r in 0..rows:
  p = pos + r
  len_r  = selected_tokens + (p + 1 - recall_pos)
  slot_r = chain[p/page] * page + p % page
```

```text
bt8[r]      = pages（零填充到 nb；行 0 的额外尾页由 len_0 限界，不读）
kv_lens[r]  = len_r
kv_slots[r] = slot_r
pos[r]      = logical p
```

`slot_r` 仍是会话链的物理页，K/V 写入位置不变；`bt8[r]` 只是把“读哪些页”
变成视图。新 token 所在页必是 `pages_r` 的最后一页（tail_hi = p/page）。

### 2.2 数据流

```text
select_view_at(sid, pos)
  ├─ 无视图 / pos < recall_pos → None
  └─ 有视图 → (pages, len, recall_pos)

verify_view_tables(sid, pos, rows)
  ├─ 会话链 [0..ceil((pos+rows)/page)] 无占位符 → None（dense 路径）
  ├─ 有占位符 + 无视图 → Err（响亮拒发本轮）
  └─ 有占位符 + 有视图 → Some(bt8, lens, slots)

verify_round
  ├─ Some(view) → 用逐行 bt8/lens/slots
  └─ None       → 用旧 bt8/lens/slots（逐字不变）
```

---

## 3. 接口

### 3.1 新增 `SelectViewAt`

`crates/engine/src/exec/coverage.rs`：

```rust
pub(crate) struct SelectViewAt {
    /// 选中块物理页（时间序）。
    pub pages: Vec<u32>,
    /// 视图内 token 数（选中块 + 尾部）。
    pub len: usize,
    /// 召回点（尾部起点）。
    pub recall_pos: usize,
}
```

纯函数：

```rust
impl<D: DeviceClient> RunningEngine<D> {
    /// 当前 pos 的视图；无视图 / pos 在召回点之前 => None。
    pub(crate) fn select_view_at(&self, sid: u64, pos: usize) -> Option<SelectViewAt>;
}
```

`select_view_step` 改为调用它，计数器/告警留在原处；现有行为不变。

### 3.2 新增 `verify_view_tables`

`crates/engine/src/exec/spec.rs`：

```rust
struct VerifyViewTables {
    bt8: Vec<f32>,   // [rows, nb]
    lens: Vec<f32>,  // [rows]
    slots: Vec<f32>, // [rows]
}

impl<D: DeviceClient> RunningEngine<D> {
    fn verify_view_tables(
        &self,
        sid: u64,
        pos: usize,
        rows: usize,
        page: usize,
        nb: usize,
    ) -> Result<Option<VerifyViewTables>>;
}
```

判据：

- `need_pages = ceil((pos + rows) / page)`；
- 会话链前 `need_pages` 项**无占位符** ⇒ `None`（dense）；
- 有占位符但无视图 / `pos < recall_pos` ⇒ 结构化 `Err`；
- 逐行构造 `bt8/lens/slots`；
- 任一行 `pages.len() > nb`、页号越界、`len > pages.len()*page` ⇒ `Err`。

### 3.3 verify_round 接线

`verify_round` 里：

1. 先取 `page`、`bt_chain`；
2. 调 `verify_view_tables`；
3. `Some(view)` ⇒ 用 `view.bt8/lens/slots`；
4. `None` ⇒ 用现有 `bt8_f/lens_f/slots_f`；
5. `fi_slots_on && view.is_some()` ⇒ 结构化 Err（配置层也应拒）；
6. 旧的“占位符守卫”由新函数统一接管：有洞 + 无视图 = Err，有洞 + 有视图 = 放行。

---

## 4. 配置与启动门

### 4.1 允许组合

`crates/shared/src/config/mod.rs::validate`：

- `select_partial + spec.depth > 0` 不再一律拒；
- 但要求：
  - `kernel.verify_arm == V2`
  - `kernel.decode_arm == V2`
- 否则拒启，文案：

```text
select_partial + spec 目前只支持 v2 臂：
FI/XQA 的逐行视图页表未实现。请设 [kernel] verify_arm = "v2"、
decode_arm = "v2"（或关 select_partial / 关投机）。
```

### 4.2 启动自检

`crates/engine/src/engine.rs::boot_kvmem_check` 增加纵深守卫：
`select_partial && spec.depth > 0` 时若臂不是 v2 ⇒ `Err`。
避免测试/手工构造 `EngineKnobs` 绕过配置层。

### 4.3 合法组合

| select_partial | spec | 臂 | 结果 |
|---|---|---|---|
| false | on | fi/v2 | 现有 |
| true | off | v2/fi? | 现有（但 FI 仍有洞隐患，见 §7） |
| true | on | v2/v2 | **本设计放行** |
| true | on | fi/xqa | 拒启 |

---

## 5. 施工步骤（按序）

### S0 设计文档

- 本文件。
- 更新 `vram-map` / `kvmem-selection` 的挂账条目。

### S1 纯函数抽取

- `coverage.rs` 新增 `SelectViewAt` + `select_view_at`；
- `select_view_step` 改为调用它；
- 现有 decode 行为逐字不变；
- 单元测试：伪造 `SelectView` + chain，验证
  `pos` 在召回点前/后、页数、len。

### S2 verify 视图表

- `spec.rs` 新增 `VerifyViewTables` + `verify_view_tables`；
- 逐行构造、边界守卫；
- 单元测试（host-only）：3 行、跨页、占位符、无视图。

### S3 verify_round 接线

- `Some/None` 分支；
- 移除旧占位符守卫（由新函数接管）；
- FI 防御门；
- 无视图路径逐字不变（用现有测试/e2e 对照）。

### S4 配置门

- 允许 `select_partial + spec`（v2）；
- 拒绝 FI/XQA；
- 更新 `select_partial_rejects_spec` 测试：
  - v2/v2 通过；
  - fi/xqa 拒；
  - 文案可行动。

### S5 Profile

新增：

```text
profiles/qwen38_27b-3090-w4a16-dflash-kvmem.toml
```

关键值：

```toml
[runtime]
max_seq = 262144

[spec]
depth = 7
draft_path = "../models/syvai/Qwen3.8-27B-DFlash2-W4A16"

[pool]
prefix_cache = true
kv_fp8 = true
host_cache_mb = 10240
snap_max = 1
vram_reserve_mb = 512

[kvmem]
select_decode = true
select_partial = true
index = true
pool_overcommit = true
workspace_per_req = 2240
select_sink = 2
select_recent = 4

[kernel]
decode_arm = "v2"
verify_arm = "v2"
fi_cuda_graph = false
```

### S6 测试

先独立小测：

1. `cargo test -p owl-shared --lib`（配置门）；
2. `cargo test -p owl-engine --lib`（现有 130 项）；
3. GPU 短档：2k/8k/32k，`select_partial + dflash2`，预热 + 贪心 64 tok；
   - 零 IMA；
   - AL > 0；
   - 与 `select_decode + dflash2` 对照。
4. GPU 长档：65k/131k/260k，逐档：
   - 先 spec off 建池；
   - 再 spec on；
   - 记录 AL、decode、`/debug/vram`、`/debug/metrics`；
   - 每档至少一次真下沉（`host_blocks > 0`）。
5. 复用/回滚：同 session 多 turn、prefix cache 命中、spec reject 回滚。

### S7 收尾

- 若 AL 明显低于 dense（<80%），评估是否加 `draft_floor`：
  - `floor = frontier - ring`；
  - `floor <= frontier`；
  - reset on reuse；
  - DFlash2 核加 `min_positions`。
- 若 view verify 在 FI 上成为瓶颈，另立 FI 视图刀。
- 更新文档与挂账。

---

## 6. 验收门

| 门 | 判据 |
|---|---|
| 不崩 | 8k/32k/65k/131k/260k，各 1000+ decode 步，零 IMA/CUDA 700 |
| 不漏页 | 每次 verify 前 host 校验：无占位符进 `bt8`、页号 < nb、`len ≤ pages*page` |
| AL | `select_partial + dflash2` ≥ `select_decode + dflash2` 的 80% |
| decode | 高于 `select_partial + spec off`（同档同 prompt） |
| 质量 | 固定 prompt 与 dense/spec-off 前缀一致率 ≥ 60% |
| 召回 | NIAH/agent 不退化 |
| 内存 | 设备 KV 与 `select_partial + spec off` 同量级；额外 ≤ 1 MiB 页表 |
| 复用 | prefix cache / 回滚 / 多 turn 不毒化上下文 |

---

## 7. 风险与对策

| 风险 | 说明 | 对策 |
|---|---|---|
| 逐行页表契约 | `bt8` 行长度不同；`kv_lens` 与页表容量不一致 | 单元门 + host 校验 |
| partial page | `view.tokens = pages*page` 可能含无效尾行 | `len ≤ pages*page`；必要时用 `visible_tokens` 夹 |
| tail 被下沉 | `select_recent` 太小 / 池满 | boot 校验 `select_recent*page ≥ depth+1`；verify 前 stage-in |
| FI 单表 | `fi_indices` 不支持逐行 | 本刀 v2；配置拒 FI |
| XQA | `xqa_pt` 取会话链 | 配置拒 XQA；已有 select_partial 自动关 |
| ring 旧特征 | 重选/复用后 AL 掉 | 先实测；必要时加 `draft_floor` |
| GDN 回滚 | verify 拒绝时状态 | 复用现有 GDN 快照/回滚 |
| 上下文毒化 | 一次非法页号污染 context | 发射前 host 校验；结构化拒发，不发核 |

---

## 8. 回滚

- 本设计只动 `coverage.rs` / `spec.rs` / `config/mod.rs` / `engine.rs`（自检）
  / 新 profile / 文档；
- 回滚：恢复这些文件，删除新 profile；
- 旧路径（无视图 / select_partial 关）逐字不变；
- 配置门回滚后 `select_partial + spec` 重新拒启，不会静默跑错。

---

## 9. 施工记录（2026-10-23）

### 9.1 已落地

| 件 | 位置 | 状态 |
|---|---|---|
| `view_pages_at` 纯函数 + 单测 | `exec/coverage.rs` | ✅ |
| `verify_view_tables` | `exec/spec.rs` | ✅ |
| verify_round 接线 | `exec/spec.rs` | ✅ |
| `write_bt` 视图态跳过 | `exec/spec.rs` | ✅ |
| `xqa_pt_of` 按池容量截断 | `exec/spec.rs` | ✅ |
| FI 表只在 FI 臂构建 | `exec/spec.rs` | ✅ |
| 配置门（v2-only） | `shared/config/mod.rs` | ✅ |
| boot 自检 | `engine.rs` | ✅ |
| profile | `profiles/qwen38_27b-3090-w4a16-dflash-kvmem.toml` | ✅ |
| 证据计数 | `spec.verify.view` | ✅ |

### 9.2 实测

| ctx | 路径 | AL | spec 轮 | verify | 备注 |
|---:|---|---:|---:|---:|---|
| 65k | dense（无洞） | 0.75 | 346ms | 327ms | v2 verify 扫全 65k；退化 |
| 90k | 视图 | 2.70 | 51ms | 36ms | `spec.verify.view=10` |
| 131k | 视图 | 2.00 | 48ms | 36ms | 零 IMA |
| 260k | 视图 | 3.00 | 48ms | 36ms | 零 IMA |

视图化 verify 把 90k 的 verify 从 326ms 降到 36ms（**~9×**）。

### 9.3 发现的阻塞（未结案）

DFlash2 + 视图在 **80,480 tok 池**（profile 缺省）下，90k no-thinking 输出
退化为 `Yes / user / assistant`；同一 prompt 下：

| 配置 | 池 | 输出 |
|---|---:|---|
| spec off + 视图 | 76,640 | 连贯 |
| spec on + dumb 草稿 + 视图 | 23,264 | 连贯 |
| spec on + DFlash2 + 视图 | 74,464 | 连贯 |
| **spec off + 视图** | **80,480** | **退化** |
| **spec on + dumb 草稿 + 视图** | **80,480** | **退化** |
| **spec on + DFlash2 + 视图** | **80,480** | **退化** |

**spec off 在同池 80,480 也退化** ⇒ 不是 DFlash2、不是视图化 verify，
而是 **KVMem 选择轨迹在 80k 池下分叉**。

诊断（逐层收窄）：

1. **视图页没有失守**：给 `SelectView` 加 `blocks` 身份后，
   `carrier_of(block) == Device(slot)` 且 `slot == view_page` 全部成立；
   不是“页被复用/没 pin”。
2. **同一选中块的内容一致**：80k 档 idx2 页 2448 与 76k 档 idx2 页 2336
   的 **16 层 K/V 全量字节哈希完全相同**（`fe6653c2c199f040`）；
   不是单页内容腐败。
3. **选中集合不同**：80k 与 76k 的 `set.blocks` 全列表哈希：
   `bhash=4cc648da4e019847`（80k）vs `3c57d56185382fa0`（76k）；
   first/last 相同，但中间块不同。
4. 候选块数/索引在场数相同（2812/2812）。
5. **同 block 分数不同**（末边界 top8）：
   - 80k：`116843…:0.188614994`、`124373…:0.187117472`
   - 76k：`124373…:0.194004610`、`116843…:0.193152875`
   ⇒ 不是 tie-break，是 **query/index 真的变了**。

⇒ **80k 的问题不是 view/verify，也不是页 pin/单页内容，而是选择轨迹
（query → score → view → 下一边界 query）随池布局分叉**。
同一 block 的分数不同 = 目标 hidden/query 或 Mean-K 索引不同；
随机词 prompt 分数接近，轨迹一旦分叉就会换掉中间块。
76k 恰好选中了一组能产出连贯续写的块；80k 是另一条合法但质量差的轨迹。

下一刀（选择层，不是 verify）：

1. **每个边界**打印 top-8 `(block, score)`，找 80k/76k 第一次分叉的 ctx；
2. 在该边界对比同一 block 的 Mean-K 索引字节 + 当轮 query 向量；
3. 若 query 不同 ⇒ 追上一边界 view；若索引不同 ⇒ 查 `collect_index_for_block`；
4. 目标：选择轨迹与池布局解耦（固定调度/冻结 query/索引单源），
   或明确接受“不同池 = 不同轨迹”，只保证同池重跑可复现；
5. 确定性门：同 prompt/同池两次跑 `bhash` 相同；跨池若要同 bhash，
   必须先把 query/index 做成池无关。

**不要在生产档默认开 spec**，直到选择确定性门结案。

### 9.4 验收门现状

- 不崩：90k/131k/260k 通过；
- AL：通过（>0，且视图档 2.0–3.0）；
- 输出一致性：**失败**（80k 池）；
- 因此本设计 **未达生产验收**；代码保留为研究分支。

---

## 10. 参考

- ninfer 修复：`repos/ninfer-kvmem-actkv/docs/dflash2-kvmem-fixes-20261018.md`
- ninfer 审查：`.../docs/dflash2-kvmem-audit-20261019.md`
- owl 当前视图：`crates/engine/src/exec/coverage.rs::select_view_step`
- owl verify：`crates/engine/src/exec/spec.rs::verify_round`
- owl 配置门：`crates/shared/src/config/mod.rs::validate`
- owl 启动门：`crates/engine/src/engine.rs::boot_kvmem_check`

---

## 10. 80k 续查与工作集修复（2026-10-23）

逐边界取证（`top8` + `qfnv` + `bhash`）显示：80k 与 76k 在 **74240 之前完全一致**，
74240 时 76k 池先满并开始 partial 下沉，80k 仍 dense；从 **74752** 起
prefill 的输入视图不同，hidden/query 才分叉。**不是同一输入上的选择器错排。**

按上游 `mean_key_retrieval.cpp` 补了 `ScoreMode::CosineMean`（余弦，
key 模长归一），但 2240 工作集下输出仍退化。受控 A/B 定谳：

```text
2240 tok (70 页, top-k 64)   → 80,480 池输出 "Yes/user/assistant"
8192 tok (256 页, top-k 250)  → 连贯
16384 tok (512 页, top-k 506) → 连贯，但 verify 轮 194ms / AL 1.60
```

根因 = **工作集预算**：90k 上下文只留 2.2k 可见历史，top-k 被近期块占满。
`[kvmem].workspace_per_req` 已从 2240 提到 **8192**（profile 默认）。
余弦口径保留（与 KVMem 产品实现一致）；`LogitMean` 继续作对照。

已知代价：8192 档 DFlash2 `AL=3.00 < 4.07`，触发 `spec-degrade` 降级裸 decode；
16384 档更低。spec 速度与长上下文可见面需要另立一刀，不同时解决。

### 10.1 选择层反证（90k needle）

同一 needle（token 10,000 = 块 312）：

```text
40k: 选中块 312，hash 48c9a8f016f32bbf，答 BLUEBIRD-42
90k: 选中块 312，hash 48c9a8f016f32bbf，答 No
```

90k 的 `bids` 含 312–319，stage-in 后页哈希与 40k 一致 ⇒ **不是选错块、
不是搬运丢内容**，是 80k 相对距离下的模型检索失败。80k 工作集退化的修复
仍是 2240→8192；不要把长距离 `No` 归因给 top-k。

### 10.2 query 窗

`W_QUERY_TOKENS` 8→64：上游 last-user span 的缺省回退就是 64
（`--kvmem-query-last 64`，最多 512）。8 行时结尾
`"Reply with only the code, no explanation. Code:"` 会把 `"secret"` 挤出窗；
64 行把完整问句纳回。该改动不改变 90k 的 `No`（见 10.1），但属于上游口径修正。
