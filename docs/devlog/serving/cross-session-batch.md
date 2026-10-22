# 跨会话合批 decode（B5.1）—— 调研与设计

> 2026-10-20 立项调研。上游立项：`multi-session.md` §六“真批处理（B5 批桶）＝另案”；
> 核级证可：`crates/kernels/tests/gdn_decode_batch_probe.rs`（B5.0，2026-10-10）。
> **边界**：本方案只动 **engine / models / kernels**；HTTP、队列、SSE、会话映射等
> **上层服务面由另一路负责**（`multi-session.md` 已就绪，见 §1.2）。

---

## 0. TL;DR

- **收益**：以生产形（27B W4A16 + DFlash2，单 3090 Ti，4k ctx，AL≈2.86）为锚，
  一阶模型给出

  | 形态 | 每趟权重产出的 token | 聚合吞吐（B 路） |
  |---|---|---|
  | 现状：spec 单会话 | 2.86 | 70 t/s（B=1） |
  | **M1 裸 decode 合批**（spec off） | B | B=4：~94 t/s（+34%，且是 M2 的降险阶梯） |
  | **M2 spec 回合合批**（verify 合批） | 2.86·B | B=4：**~126 t/s（+80%）**；B=8：~145（+107%） |
  | M3 + propose 也合批 | 2.86·B | B=4：**~164 t/s（+134%）**；B=8：~197（+181%） |

- **合批不改变 AL**：吞吐 = AL ÷ 轮时，合批只摊薄**每轮固定成本**（一趟全权重
  ≈14.8ms 的 target verify 权重遍历 + 宿主/发射税）。
- **批大小是显存问题，不是算力问题**：27B boot live **17.52G**、每 token 池
  53248B（主池 32768 + 草稿池 20480）⇒ 16k ctx 下 **B≈4 封顶**（22.8G 可用）；
  8k ctx 可到 B≈6–8；**M1（无草稿池、无快照）更宽松**（每 token 32768B）。
- **推荐路线**：M0 门 → **M1（裸 decode 合批，零草稿池/零快照）** → M2（spec 回合合批，
  需草稿池 per-session 分区 + 快照 ≥ B）→ M3（propose 合批 + 调度策略）。
- **不需要的**：不改 vLLM/marlin 核；v2 **seq** 变体（`decode_v2_seq_op`）已带 seqs 维
  （verify 8 伪序列在用），M1 直接复用；GDN 全族核已支持 per-row slots（B5.0 已证）。

---

## 1. 边界与现状

### 1.1 本方案负责（引擎侧）

| 面 | 现状 | 目标 |
|---|---|---|
| 调度 | `scheduler::schedule()` **单活跃 turn**（`turns.active: Option<ActiveTurn>`，`turn.rs:223`） | 多活跃（≤B）+ 相位同质拼批 |
| 执行 | `StepAction::Decode { token,pos,kv_slot,gdn_slot,grew }` 单行（`scheduler.rs:585`） | `BatchDecode { rows: Vec<RowSpec> }` |
| 图 | session 图 rows=1（`graph_fleet.rs:531-535`、输出 `logits[1,vocab]` `:577`） | `batch_decode` 图族（rows=T，页表 [T,nb]） |
| 状态 | 单 `ActiveTurn`（含 spec 七件套）、单快照 | `Vec<ActiveTurn>`；快照/草稿池按会话分区 |
| 采样 | host 单行（`sampler.rs:73`） | 逐行（一次 dtoh `[T,vocab]`，行切片） |

### 1.2 不负责（上层，已就绪）

- `server.max_concurrency` 语义门（`apps/server/src/main.rs:271-278`）+ `Semaphore`；
- `gdn_slots ≥ max_concurrency` 启动 fail-fast（`crates/shared/src/config.rs` validate）；
- session 键 → `AgentSession`（GDN 格/KV 块表/前缀守卫/LRU）——`session.rs` 已有；
- 逐请求事件流（SSE）：引擎的 `TurnEvent` 已是 per-turn，合批后仍 per-turn 发。

### 1.3 已就绪的“批语义”资产（复用，不重写）

| 资产 | 位置 | 说明 |
|---|---|---|
| **GDN 批语义** | `crates/kernels/tests/gdn_decode_batch_probe.rs` | B=2 单发 ≡ 两次单格**逐位**；`slots[i]<0` = padding 掩码（B5.0 门，已绿） |
| GDN per-row 槽 | `gdn_conv_upd.cu:48`、`gdn_decode_step.cu:72`、`gdn_recurrence.cu:81/204` | 核内 `slot = slots[b]` / `slots[seq_idx]`；decode 路径原生 `[B]` |
| GDN 序列包 | `gdn.rs:622`（`cu_seqlens`；注释“M2 packed 直用”） | `[0,T]` → 批化即 `[0,t1,t1+t2,…]`（varlen 核 `seq_idx` 定 slot） |
| **多序列注意力** | `attention.rs:731` `verify_v2_arm`（`decode_v2_seq_op`，`seqs=depth1=8`） | 核已有 **seq 维**（grid.y=seqs，页表 `bt8 [seqs,nb]`、`kv_lens [seqs]`）—— B5.0 头注“v2 无 seq 维”已过时 |
| 多行页表 | `spec.rs:131`（`bt8` 平铺）+ `graph_fleet.rs:760+`（verify 图 `bt8_in`） | 每行一份页链；批化 = 每会话一行链 |
| 多行 K0 写 | `verify_v2_arm` 的 `k0_write`（`slots [T]`） | 逐行物理槽写池，已验证 T=8 |
| 多行 lm_head/argmax | session 图 `token [1]` 设备 argmax；verify 图 `logits [8,vocab]` | 行批天然支持 |
| host 采样 | `sampler.rs:73 sample(logits,cfg,rng,history)` | 逐行调用即可 |
| 宿主并发闸门 | `apps/server/src/main.rs:271-278` | 已有；引擎只需真的并发推进 |

### 1.4 单行假设清单（批化要逐条改）

| # | 假设 | 锚点 | M1 | M2 |
|---|---|---|---|---|
| 1 | session 图槽全 `[1]` | `graph_fleet.rs:531-535` | [T] | [T] |
| 2 | 页表单表（`bt_leaf`/`sel_pt [nb]`） | `graph_fleet.rs:429-431`、`phases.rs:394-397` | [T,nb] | [T,nb] |
| 3 | attention 臂要求 `tokens==1` | `attention.rs:473`（FI）、`:503`（XQA） | 走 v2 seq（`seqs=T`） | FI batch 或 v2 seq |
| 4 | `gdn_slot` 标量 `[1]` | `graph_fleet.rs:535`、`phases.rs:345` | [T] | [B] + `gdn_cu [B+1]` |
| 5 | `seq_cu=[0,T]` | `gdn.rs:622` | decode 路不用（fused/`slots[b]`） | `[0,8,16,…,8B]` |
| 6 | 采样单行 | `phases.rs` decode 尾 | 逐行 | 逐行（`[T,vocab]`） |
| 7 | spec 状态单 turn | `turn.rs:168` `TurnSpecState` | 不涉及 | 逐会话 |
| 8 | 草稿池线性寻址（跨会话互踩） | `state.rs:1018` 注释 | 不涉及 | **per-session 分区** |
| 9 | 快照池 `snap_max` 槽 | `state.rs` 快照池；flavor `snap_max=4` | 不涉及 | **`snap_max ≥ B`** |
| 10 | `max_concurrency`/`gdn_slots`=1（生产 flavor） | `flavors/spec-dflash.toml` | 配置放开 | 同左 |

---

## 2. 收益模型（一阶）

### 2.1 实测锚点（27B W4A16 + DFlash2 d7，单 3090 Ti，生产形）

| 项 | 值 | 出处 |
|---|---:|---|
| spec 轮 p50（ctx 512→8192 恒定） | **41.0–42.5 ms** | 本地台账《26-10-19-decode稳定性与GDN小刀》 §一 |
| ├ verify | 29.3 | 同上（marlin 17.9 + attention 4.3 + GDN 2.5 + 宿主/其它 4.6） |
| ├ propose | 7.8 | 同上 |
| └ fold | 2.2 | 同上 |
| 裸 decode 步 | 20.7 ms @~200 tok；24.1 @564；**41.7 @32k** | 同上 §一/§2 注 |
| 权重遍历（一趟） | **≈14.8 ms**（13GB @900GB/s） | 推导；marlin decode 主核 57µs×260 发/轮 |
| AL（生产草稿） | 2.86 / 5.42 | 本日复测（固定 seed 采样） |
| AL（全精度草稿 z-lab） | 3.10 / 4.96 | 同上（**精度不抬 AL**） |

### 2.2 模型

```
T_round(B) ≈ W_t + B·(A(C) + G + P + F) + H
W_t ≈ 17.9ms   target verify 权重遍历（m=8B 行；带宽界 ⇒ 对 B 近似常数）
A(C) ≈ 4.3ms @4k，≈ C/4k 线性增长（KV 读带宽界）
G ≈ 2.5ms GDN；P ≈ 7.8ms propose；F ≈ 2.2ms fold；H ≈ 6ms 宿主/发射
```
B=1 回代 = 41.0 ✓。

| B | T_round（4k） | 聚合 t/s（4k） | T_round（16k，A=17.2） | 聚合 t/s（16k） |
|---:|---:|---:|---:|---:|
| 1 | 41.0 | 70 | 53.6 | 53 |
| 2 | 57.5 | 99 (+41%) | 83.3 | 69 (+29%) |
| 4 | 91.1 | **126 (+80%)** | 142.7 | 80 (+50%) |
| 8 | 158.3 | 145 (+107%) | 261.5 | 88 (+64%) |
| 4（propose 也合批，P→2.5） | 69.9 | **164 (+134%)** | 121.5 | 94 (+76%) |

- **关键判读**：合批收益 ∝ 固定成本（权重遍历 + 宿主税）占比。ctx 越大、attention
  越重，摊薄越少 ⇒ **合批最值钱的是 8k 以内 ctx**；16k 生产档仍 +29~64%。
- **每会话延迟会上升**（B=4/4k：41→91ms/token），交互单会话体验最好仍是 B=1；
  多子代理/后台任务（吞吐优先）用大 B。
- **敏感度**：以上 A=4.3ms 是 **FI verify** 的成本；若 M2 只能走 v2 seq 臂且其
  单行成本 ≈ 2×FI（R2），B=4 的聚合从 126 降到 ~105 t/s（+51%）。⇒ **FI 批化
  （R1）是 M2 收益的关键前置**，先做小门验证 plan 形状恒定。
- **spec 合批的价值**：每趟权重的 token 产量 2.86B vs 裸批 B —— 同 B 下 spec 批
  恒优（B=4：126 vs 94 t/s）；**M1 不是终态**，价值 = 零草稿池/零快照 → 先把
  调度/图/采样三件套打通，且它就是生产“降级态”的正确路径。

### 2.3 显存账（决定 B 上限）

- 27B boot live **17.52G**（权重 + 固定件），可用 ≈ 22.8G（99% − reserve 512M）；
- 每会话每 token：**主池 32768B + 草稿池 20480B = 53248B**（fp8 主池/BF16 草稿池）；
  16k ctx ⇒ **0.87G/会话**；8k ⇒ 0.44G；
- 图 slab 实测 896MiB（族换代后预算 640MiB 偏小，另一笔账）；
- 快照 152MiB × `snap_max`（spec 合批需 ≥ B）。
- ⇒ **B=4 @16k 可行但贴顶**（17.52+3.5+0.9+0.6 ≈ 22.5G）；**B=8 需 ctx ≤ 8k**；
  M1（无草稿池/无快照）每 token 32768B ⇒ B=4 @16k = 2.1G，宽松。

---

## 3. 设计

### 3.1 词汇（引擎侧新增/改动）

```rust
// 本步一行 = 一个会话的一次 decode（或一次 verify 块）
pub(crate) struct RowSpec {
    pub turn: u64, pub session_id: u64,
    pub token: u32, pub pos: usize,          // frontier / 位置
    pub kv_slot: u32, pub kv_len: usize,     // 物理槽与可见长
    pub gdn_slot: usize,                     // GDN 格
    pub grew: bool,                          // 本步块链增长(需写该会话 bt 行)
    pub sample: Option<RowSample>,           // 采样档;None=贪心(设备 argmax)
}
pub(crate) enum StepAction {
    // 既有:Prefill / Decode{单行} / SpecRound{单会话}  (B=1 原路,零回归)
    BatchDecode { rows: Vec<RowSpec> },      // M1
    BatchSpec { rows: Vec<RowSpec>, depth: usize }, // M2(rows = Σ(depth+1))
}
```
`TurnTable`：`active: Option<ActiveTurn>` → `active: Vec<ActiveTurn>`（长度 ≤ `knobs.max_batch`），
`turns.active()` 改 `turns.rows()`；所有“单 turn”调用点逐条迁（`rg 'turns\.active' crates/engine`）。

### 3.2 设备面（M1）

- **图族 `batch_decode`**：按 **t_pad 阶梯**（{2,4,8}，与 prefill 桶同款
  “向上取整 + 真行数 = t_real”）在 boot 捕获；失败该桶永久 eager（同 `prefill_plan` 纪律）。
- **输入槽**（全 [T] 或 [T,nb]）：
  `frontier`、`pos`、`kv_len`、`kv_slot`、`gdn_slot`、`bt [T,nb]`
  （可选 `sel_pt/sel_len [T,nb]`，若 `select_decode` 档；M1 先要求 `select_decode=off`）。
- **输出槽**：`token [T]`（设备 argmax）+ `logits [T,vocab]`（仅采样档；
  1024·B·0.5MB 级 D2H）。
- **注意力臂**：`decode_v2_seq_op`（`seqs=T`，页表 `bt [T,nb]`、`kv_lens [T]`），
  scratch 用 `AttnV2Scratch` 新开 `seqs=T` 一组（`state.rs::attn_v2_scratch` 旁加
  `batch_attn_v2_scratch(T)`；nparts 同源）。**FI/XQA 臂 M1 不用**（单批/单序列）。
- **GDN**：`forward_decode_decl`（tokens=T）原生 `slots[b]`；无 `cu`。
- **K0 写**：`kv_slots [T]` + `kv_lens [T]`（verify 同款）。
- **lm_head/argmax**：行批天然。
- 图捕获固定 `T` 行（t_pad），回放写**全 t_pad** 行（真行 + 假行 padding：
  假行 `kv_lens=0` / `slots=-1`（GDN 掩码）/ 页表补 0 —— 与 B5.0 的 padding 语义一致）。

### 3.3 宿主面（M1）

- **调度**（`scheduler.rs`）：`schedule_batch()`：
  1. 从 `active` 里取**相位相同**（都 bare decode）的 turn，最多 B 个；
  2. 每个 turn 复用既有派生（`ensure_for_len` / `kv_slot` / `grow` / spec 降级判定）；
  3. `begin` 可先跑（新 turn 准入与首步同发，历史语义保持）。
- **执行**：`execute_batch_decode(rows)`：
  - 写 `bt [T,nb]`（仅 `grew` 行；其余保持）与行槽；
  - 一次 `graph.step`；读 `token [T]`（贪心）或 `logits [T,vocab]`（采样）；
  - 逐行 `sample()`（`SamplerCfg` per-turn，RNG per-turn）→ 逐行 `out.push`；
  - 逐行 EOS/长度/停止串判定 → `TurnEvent`（per-turn 事件入 `pending_events`）。
- **会话状态**：每行 `gdn_slot` 来自 `ActiveTurn`（已存在，turn 域借还）；
  KV 块链 per-session（`AgentSession.block_table`），`bt` 行 = 该会话链。
- **spec 降级**：M1 要求 spec off（`depth=0` 或降级态）；生产档跑 M2 前，
  合批只对“降级/无草稿”会话生效（正确但收益有限）——**M2 才是生产收益点**。

### 3.4 M2：spec 回合合批（verify）

- rows = Σ(depth+1)（同 depth ⇒ T = B·8）；`gdn_cu` [B+1] = `[0,8,16,…]`；
  `gdn_slot` [B]（varlen 核 `slots[seq_idx]`）；`kv_slots`/`kv_lens` [T]；
  `bt` [T,nb]（每会话链平铺 8 行）；`pos` [T]。
- 图 = 现有 verify 图的**推广**（`plan_verify`，`graph_fleet.rs:760+`），
  T 从 `depth1` → `B·depth1`（阶梯 {8,16,32}）。
- **FI 批化**（生产 verify 臂）：`owl_fi_prefill_plan/run` 已收 `batch_size`；
  需 Rust 侧：`qo_indptr=[0,8,16,…]`、`kv_indptr=[0,nb1,nb1+nb2,…]`、`indices`
  = 各会话链拼接、`last_len` [B]；plan 键加 batch；`fi_plan_sync` 每步按行刷。
  ⚠️ 现状槽面是**单批定形**：session/verify 图把 `fi_q_cu`/`fi_indptr` 写成 `[2]`、
  `fi_last_len` `[1]`（`graph_fleet.rs:489-495`）⇒ 批化要新增 `[B+1]`/`[B]` 声明，
  且 `FiPlanSpec`/`plan_sync` 增加“逐行 ctx/页表”字段（现只有单值）。
  **前置验证**：图档瓦（`fi_cuda_graph=true`）下 plan 形状对“行数×ctx”是否恒定
  （不恒定 ⇒ 同一图不可复用不同 ctx 组合，需退回 v2 seq 臂）。
- **草稿池 per-session 分区**（`state.rs:1018` 注释的未决项）：`draft_base(session)
  = session_row · max_seq`，池容量 ×B；encode/propose 的 `slot_at`/`prefix` 全走分区。
- **快照 ≥ B**：`snap_max ≥ B`（每会话轮首一张），VRAM +152MiB×B（进预算公式）。
- 采样接受：逐会话（现有 `ids/logits` 行切片，`spec.rs` accept 逻辑 per turn）。

### 3.5 M3：propose 合批 + 调度策略

- propose 行批：draft 模型 T=B 行（selector/NC 核已有 batch 形状），
  权重遍历摊薄（P 7.8→~2.5ms）；
- 调度策略：**批窗口**（凑批等 1–2ms）、**公平性**（大 ctx 会话不与小 ctx 混批？
  或按行代价分摊）、**max_concurrency 与 batch 上限分离**；
- 度量：per-session 延迟 vs 聚合吞吐曲线（上层提供多路客户端）。

---

## 4. 门与验收（每里程碑独立）

| 门 | 判据 | 跑法 |
|---|---|---|
| **M0** `batch_decode_matches_serial` | N∈{2,4} 会话、同/异 prompt，**逐 token** 与串行单会话相同（eager 与图两档；含 sampling seed 固定档） | `OWL_TEST_DEVICE=<bdf> cargo test -p owl-engine --release batch_decode` |
| **M1** 裸 decode 合批 | 上述门 + 聚合吞吐 ≥ 1.8×（B=4，4k ctx，AL 归一后按 t/s 实测）；单会话 B=1 路径逐字节不变（回归门） | 同上 + 并发客户端（上层） |
| **M2** spec 回合合批 | 各行 accept/bonus 与串行**逐位**相同（同 seed）；AL 与轮数不退化；sanitizer 0 errors | `OWL_TEST_DEVICE=… … batch_spec` |
| **M3** propose 合批 | 草稿 id 与串行逐位相同（同 seed）；轮时下降 | 同上 |
| 回归 | `owl-engine --lib` 67/67；`graph_plan::tests` 7/7；生产 AL 2.86/5.42 不变；`gpu_prefill_bucket_matches_eager` 绿 | 每笔必跑 |

---

## 5. 风险与未决

| # | 风险 | 判断/缓解 |
|---|---|---|
| R1 | **FI 批化不确定**（plan 形状随行数×ctx 变） | M2 前先写小门：同图两批不同 ctx 组合回放，plan 形状/offsets 必须一致；否则退回 v2 seq 臂（代价见 R2） |
| R2 | v2 **seq** 臂比 FI 慢 | flavor 注：v2 臂档实测 **38.8 t/s**（引擎判“轮时过高/不划算”后降级裸 decode），FI 臂档 **87.2 t/s**（@8192）。⇒ M1 前先测 v2 seq 单行 decode vs FI 的差距（预期 1.5–2×）；M2 的收益表按 2× 打折到 +51%，必要时 FI 批化提前 |
| R3 | **草稿池 per-session 分区**（线性寻址互踩） | 明确改：`draft_base(session)`；容量 ×B；encode/propose 槽基址全走分区（`state.rs:1018`） |
| R4 | **快照 VRAM**（152MiB×B） | 进预算；或首版只对“可能部分接受”的轮拍快照（不可预知）⇒ 接受成本，或 M2 先 B=2 |
| R5 | 长 ctx 摊薄（16k 仍 +29~64%，32k 更少） | 明确合批定位 = 中小 ctx 的并发代理；长 ctx 单会话仍 B=1 |
| R6 | 数值/确定性（marlin m=8B 形状、v2 seq） | 门按 token 级判；共享前缀档与 select_decode 档先不混批 |
| R7 | 队头阻塞（新 turn prefill 插队） | 相位同质批（decode 批不与 prefill 混）；prefill 独占步（M1 策略） |
| R8 | TP/双卡 | 本方案先单卡；TP 的 AR 需按行批语义重审（另案） |
| R9 | 采样 D2H（`[T,vocab]`） | 贪心档零 D2H（设备 argmax）；采样档 ≈ B×0.5MB/步（~0.1–0.3ms） |

---

## 6. 交接清单（下一个会话从这里开始）

1. **先读**：本文；`multi-session.md`（上层边界）；
   `crates/kernels/tests/gdn_decode_batch_probe.rs`（B5.0 门）；
   `crates/engine/src/graph_fleet.rs::plan_session`（session 图槽面）；
   `crates/engine/src/exec/phases.rs::execute_decode`。
2. **第一件事 = M0 门**（不改生产路径）：新建 `crates/engine/src/tests.rs::batch_decode_matches_serial`
   - 两会话（不同 prompt，已知 KV 状态）各跑 8 步：串行 vs 合批；
   - 判据：**逐 token 相同**（贪心档；采样档固定 seed）；
   - 先 eager 后图（`no_graph` 档可先验语义）；
   - GPU 门：`OWL_TEST_DEVICE=<bdf>`（本机 3090 Ti = `41:00.0`；3080 = `01:00.0`）。
3. **第二件事 = 图**：`plan_batch_decode`（阶梯 {2,4,8}，槽面见 §3.2），
   复用 `prefill_plan::ladder`/`FaceClone` 纪律；臂 = v2 seq；先不接 FI/XQA。
4. **第三件事 = 调度**：`Turns.active: Vec<ActiveTurn>` + `schedule_batch()`，
   **B=1 必须走既有单行路径**（零回归门）。
5. **不要动**：`apps/server/*`（上层另一路负责）、`flavors/*` 生产档
   （合批档另立 flavor，例如 `spec-dflash-batch`：`gdn_slots=B`、`max_concurrency=B`、
   `snap_max≥B`、`pool_tokens=B×16896`）。
6. **配置面新增旋钮**（引擎）：`knobs.max_batch`（缺省 1 = 现状；生产另档设 4）。
7. **验收命令**：`OWL_TEST_DEVICE=41:00.0 cargo test -p owl-engine --release batch_`
   + 并发客户端（上层）测聚合 t/s（AL 归一：`AL = 1 + Δaccepted/Δround`）。

---

## 7. M1 落地实录（2026-10-20）

### 7.1 已落地（M0 + M1 + M1b 图臂）

| 面 | 落点 | 说明 |
|---|---|---|
| 配置 | `[runtime] max_batch`（缺省 **1**） | 校验：`pool.gdn_slots ≥ max_batch`、`max_batch ≤ 8`（Concat 8 槽）、合批与 TP / `kvmem.select_decode` 互斥；boot 同门（测试直连档也拦） |
| 活跃账 | `TurnLedger.active: Vec<ActiveTurn>` + `executing: Option<u64>` | **单一阵列**；单 turn 执行码经选择器读 `active()`（多活跃未选定 = 显式错误）；`rr` 公平游标；`take_active/take_exec` |
| 调度 | `StepAction::BatchDecode { rows: Vec<RowSpec> }` | 门 = 活跃 ≥2 ∧ 全 gen 相位 ∧ `spec depth == 0` ∧ 单卡 ∧ 无选择层视图；不成立走原单 turn 路径（B=1 逐字节不变） |
| 行派生 | `Scheduler::decode_row_of` | 与单行 `Decode` 同式：块链 `ensure_for_len` → `kv_slot/kv_len=pos+1/gdn_slot/grew` |
| 设备面 | `StatePool::bt_batch [B,nb]` + `batch_attn_v2`（seqs=B） | `write_bt_batch` 每步整表重写（T×nb×4B 极小）；假行 `slot=-1/kv_len=0` |
| 模型臂 | `attention.rs` decode **T>1 批臂** → `verify_v2_arm` | 分页池 + `attn_v2.seqs==T` + `v2_bt8` 三件套；缺任一 = **毒化**（不再落回单 seq 核读错链）；融合插池核收窄到 `tokens==1` |
| 图 | `GraphFleet::plan_batch_decode`（M1b） | T=`max_batch` 定形；输入 `frontier/pos/kv_len/kv_slot/gdn_slot [T]`；输出 `logits [T,vocab]` + `token [8,1]`（Concat 固定 8 槽，取前 T 行）；捕获失败回 eager 臂 |
| 执行 | `execute_batch_decode`（`exec/phases.rs`） | 一次 T=B 前向（图优先 / eager 兜底）→ **逐行** `fed++` / 页边界快照 / 采样（per-turn cfg/seed/历史）/ `sample_and_emit`；首个事件返回，其余入 `pending_events` |

### 7.2 门与实测

- **M0 模型级**：`gpu_model_batch_decode_matches_serial`（fixture 全模，naive 槽直排）——
  T=2 与两次 T=1 **logits 逐位同级（1e-5）**，含第二步（KV+GDN 状态续跑）。
- **M1 引擎级**：`gpu_batch_decode_matches_serial`（0.8B，max_batch=2）——
  两会话并发跑出 **15 个批步**，输出与串行两跑**逐字相同**；`has_batch_graph()` 断言图臂在场。
- **M1b 吞吐（27B AWQ，spec off，greedy+ignore_eos，64 tok，ctx 2048）**：

  | 臂 | 墙钟（首 Token→双完成） | 聚合吞吐 |
  |---|---:|---:|
  | 串行 B=1 ×2 | 2.97 s | **43.1 t/s** |
  | 合批 B=2 | 1.64 s | **78.2 t/s（×1.82）** |

  ⇒ 实测优于一阶模型（+41% @ B=2）：裸 decode 的固定成本（权重遍历 + 宿主税）占比更大。
- **回归**：`owl-engine --lib` 67/67；`owl-shared` 80/80；`graph_plan::tests` 7/7；
  spec 恒等门（0.8B Dumb + 27B MTP）全绿；two_turns / session_continuity / multi_session_isolation 绿。
  ⚠️ `gpu_prefix_cache_hit` 为**存量红**（在 stash 掉的基线同样 B=6 chunks vs A=9，
  与本案无关；另案）。

### 7.2b 落地时的三条边界事实（审查补记）

1. **同会话并发 = 结构性互踩** ⇒ `submit_session_opts` 在 `max_batch > 1` 时对
   “会话已有活跃/排队 turn” **亮拒**（上层应按会话串行）。
2. **保洁需求按批求和**：`step_need` 对全部活跃 turn 累加 `目标页 − 已有页`
   （只算首 turn 会少腾槽 ⇒ 批步中途 `Exhausted` 把整批打掉）。
3. **终局只弃本步执行 turn**：`abandon` 不整批连坐（批动作失败时执行选择器为空，
   取活跃首）；其余行由服务面继续泵。
   ⚠️ 已知边界：批步**中段**（前向已推进、逐行采样阶段）出错时，各行 `fed` 与设备
   KV/GDN 可能不同步——M2 前应把“采样/事件段”做成不可失败段，或整批弃置（另案）。

### 7.2c M2 语义底（同会话多序列）已证 + 两处 batch=1 硬编码修复

- **模型门** `gpu_model_batch_prefill_matches_serial`（F16 fixture，2 会话）：
  - 多序列 prefill（T=3，`seq_cu=[0,2,3]`）与逐会话串行**logits 同级（1e-5）**；
  - verify 形（T=5，`seq_cu=[0,2,5]`，A 2 行 + B 3 行）同样逐位同级。
- **修的两处硬编码**（M2 的硬前置，单序列逐字节不变）：
  1. `conv_fwd` 的 `batch` 标量恒 1 → 改由 `cu_seqlens.len − 1` 派生；
  2. `recurrence_varlen` 的 `batch` 恒 1 + kernel pick `grid.y = nv` →
     改 `grid.y = nv·batch`（核内 `seq_head/nv` 的 seq_idx 寻址才成立）。
  - 回归：`owl-kernels --lib` 93/93；单序列 verify（27B spec 恒等门）与 prefill 桶门全绿。
- 注意力 T>1 分派补订正：**页配对不支持的小几何**（fixture hd=4）回落
  `NaiveDecode` 逐行槽窗（与 T=1 同律）；只有“分页几何成立但批面缺席”才毒化。

### 7.3 R1 判据（FI 批化形状探针，`crates/kernels/tests/fi_batch_plan_probe.rs`）

同 `total_rows/batch`、不同 nb 切分（含极端 500/10/1/1）：

| 档 | 形状签名 | 判读 |
|---|---|---|
| eager（`FI_CUDA_GRAPH=0`） | `padded 37/38`、`int_bytes 613/614` 随切分变 | 不可烘图 |
| **图档（`FI_CUDA_GRAPH=1`）** | **p0=42, p3=64, cta=64, split=1 恒定**；`int_bytes` 只随 **B**（682/762/906），跨 total 1024 vs 917 亦同 | ✅ **R1 通过**：plan 形状只由 (B, T) 定 ⇒ FI 批化可行（表内容每步窗外刷） |

### 7.4 M2（spec 回合合批）的修订路线

先把 **verify 批化** 打通，FI 后置：

1. **v2 seq verify 批**（先不接 FI）：T = B·(depth+1)（B=2 ⇒ 16）；`gdn_cu=[0,8,16]` +
   `gdn_slot[B]`（varlen 核 `slots[seq_idx]`）；`bt8 [T,nb]`（每会话链平铺 8 行）、
   `kv_slots/kv_lens [T]`；scratch seqs=T。代价 ≈ attention 臂 2×（+4ms 级），先换正确性。
   ✅ 模型语义已证（§7.2c）+ GDN varlen batch 已修。
2. **逐 turn accept/fold**：verify 输出按会话切 8 行，复用既有 `spec_fold_to`/接受逻辑
   （每 turn 自己的 GDN 槽/快照）。
3. **草稿池 per-session 分区**（R3，硬前置）：propose/encode 的逻辑槽加 `session_row × max_seq`
   基址；池容量 ×B。
4. **快照 ×B**（R4）：`spec_snap` 单缓冲 → 按 turn 行的 B 份（或快照池按边界键分区）。
5. **调度门**：全活跃都是 spec 相位 ⇒ `BatchSpecRound`；混合相位（prefill/裸步）回单 turn。
6. **FI 批化（M2b）**：R1 已证图档形状恒定；需要 `[B+1]` qo/kv indptr、`[B]` last_len、
   `fi_plan_sync` 的逐行 ctx 字段。预计把 verify attention 从 2× 拉回 FI 档。

### 7.5 服务面契约（2026-10-20,`feat/b5-batch`）

合批落地后 `apps/server` 侧的接口收口（精确取消 / 会话闸 / 可观测）见
本地台账《26-10-20-B5.1-服务面契约-立案与施工》。三条要点：

1. **同会话并发**：引擎在 `max_batch>1` 时硬拒 ⇒ 服务层必须有**会话闸**
   （同会话串行 / 跨会话并行；无闸时并发同会话必 500，且 `implicit_session`
   把所有无键请求并到会话 0）；
2. **取消必须指名**：`EngineReq::Abort` 带 `turn` + 回执；断流路径要**先关 sink**
   再等回执（否则 actor 察觉不到断流，会把 turn 跑完 —— 实测 4.88s → 0.089s）；
3. **`/debug/vram → batch{max_batch,steps,graph}`**：判"是否真在合批"的唯一现场
   （`max_batch>1` 而 `steps` 不涨 = 批门从未成立）。
