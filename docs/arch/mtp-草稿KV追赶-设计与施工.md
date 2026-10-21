# MTP 草稿 KV 追赶 —— T3 前置设计与施工

> 2026-10-22 · 工作树 `packages/owl-worktrees/mtp-spec`
> 分支 `feat/mtp-spec` · 关联：
> - `docs/arch/mtp-长上下文抗衰减-下一阶段施工与交接.md` §7（T3）
> - `docs/arch/mtp-长上下文抗衰减-设计与施工.md` §5（T1 自适应 depth）

---

## 0. 一句话

B4 自动降级对 DFlash2 可用、对 MTP **结构上不可用**：降级期裸 decode 不写
MTP KV，且 `spec_seed_hidden` 是一次性种子。T3 的第一步不是打开降级门，
而是补一条 **MTP 草稿 KV 追赶**：每裸步把 target hidden + 下一 token 写进
MTP 链，并把当前 hidden 留给下一次重入 propose 当 seed。

---

## 1. 现状与证据

### 1.1 B4 降级期在做什么

`crates/engine/src/exec/phases.rs::execute_decode`：

```text
execute_decode
  ├─ session 图 replay
  │    └─ outputs: token / logits
  │         （MTP 模式目前没有 hidden 输出）
  ├─ 采样 → nt
  ├─ if (spec_degraded || bare_seed_step) && dflash_tap_count > 0
  │    ├─ taps = session 图 tap0..tap4
  │    ├─ prefill_dflash_encode(taps)      # DFlash2 草稿池追赶
  │    └─ bare_cost_ewma / steps_degraded 记账
  └─ sample_and_emit
```

MTP 的 `dflash_tap_count = 0`，因此：

- 裸步**没有**草稿 KV 追赶；
- B4 门 `if dflash_tap_count > 0 && degrade_after > 0` 永远不进 MTP 分支。

### 1.2 重入时缺什么

`crates/engine/src/exec/spec.rs::execute_spec_round` → `acquire_drafts`：

```text
acquire_drafts
  ├─ spec_drafts_host = Some(d)  → 截断到 depth 直接消费
  │    （降级时 B4 已把它清 None；若残留 = 陈旧草稿）
  ├─ DFlash2                      → dflash_propose
  └─ MTP 且无 drafts
       └─ propose_first(anchor, pos, depth)
            └─ tspec.spec_seed_hidden.take()
                 └─ None → Err("spec seed 缺失")
```

`spec_seed_hidden` 只在 prefill 末块 harvest 一次
（`exec/phases.rs` 的 prefill 收尾）。首次 `propose_first` 消费后为 `None`，
所以 MTP 降级后**无法重入**：

- 走 `propose_first`：缺 seed ⇒ 报错/丢 turn；
- 不走 `propose_first`：只能用降级前的旧 drafts ⇒ 草稿分布错、MTP 链有洞。

### 1.3 结论

T3 的正确顺序：

```text
MTP 草稿 KV 追赶（本卷）
  → B4 门放行 MTP
  → 真机 A/B（prose/随机域自动回退，MATH 不误降）
```

---

## 2. 方案

### 2.1 追赶语义

MTP 链的一行 `p` 语义 = `(target hidden @ p-1, embedding(token @ p))`。
裸步在 `pos` 前向后得到：

- `hidden @ pos`（target 末层 hidden）；
- `nt`（`pos+1` 的 token）。

因此一次追赶写一行：

```text
prefill_extend(
  hid   = hidden @ pos,
  t_rows = 1,
  ids   = [token @ pos, nt],
  base  = pos,
)
```

`prefill_extend` 已存在（`crates/engine/src/exec/draft.rs::prefill_extend`），
prefill 每 chunk 就是走它；这里只是把它复用到裸步，每步 1 行。

### 2.2 重入 seed

同一步顺手把 `hidden @ pos` 存进 turn 状态：

```text
tspec.spec_seed_hidden = Some(hid)
```

下一次 `propose_first(anchor=nt, pos+1)` 就有一份**新鲜 seed**；
`propose_first` 写 `pos` 的 extend 行时与追赶行同值（幂等覆写）。

### 2.3 图输出面

`crates/engine/src/graph_fleet.rs::plan_session` 的 session 图目前输出：

```text
outputs: logits / token (+ tap0..tap4 仅 DFlash2)
```

需要新增（仅 MTP 且 `degrade_after > 0` 时声明，保持旧档拓扑不变）：

```text
outputs: hid  [1, hidden]  dtype = dims.dtype
```

`forward` 里 `model.forward` 或 `tapped_hidden` 已拿到 hidden，直接
`sc.output("hid", &hidden)`；零额外计算，只多一个 `[1,hidden]` 输出槽写。

### 2.4 调用面重构

`prefill_extend` 现在挂在 `RunningEngine`，签名含 `bank`/`sid`；在
`execute_decode` 里直接调会遇到 `&mut self` 与 `&mut self.bank` 的借用冲突。
推荐做法（与 `fold_step_dev` 同律）：

```text
RankDev::prefill_extend_dev(&mut self, bank: &mut SessionBank, sid, hid, t_rows, ids, base)
  └─ 现 prefill_extend 主体（self.pool/self.graphs/self.boot/spec_rt 都在 RankDev）
RunningEngine::prefill_extend(...)
  └─ 转发到 dev.prefill_extend_dev(...)
```

`execute_decode` 在裸步尾部：

```text
if degraded || seeding:
    if DFlash2: prefill_dflash_encode(taps)   # 原路
    if MTP:     prefill_extend_dev(hid, 1, [token, nt], pos)
                tspec.spec_seed_hidden = Some(hid)
```

### 2.5 B4 门放行

```rust
// 现在
if self.dev.spec_rt.dflash_tap_count > 0 && self.boot.probes.degrade_after > 0
// 改为
if self.dev.spec_rt.drafter.is_some() && self.boot.probes.degrade_after > 0
```

MTP 的 `m=0` 不必然等于亏本：沿用窗口 AL vs 盈亏线，不要单看连击。

---

## 3. 调用树（施工后目标态）

```text
execute_decode
  ├─ session 图 replay
  │    └─ outputs: token / logits / hid (MTP 档)
  ├─ sample_and_emit → nt
  ├─ if (degraded || seeding):
  │    ├─ DFlash2: prefill_dflash_encode(taps)
  │    └─ MTP:     RankDev::prefill_extend_dev(bank, sid, hid, 1, [token, nt], pos)
  │                 └─ tspec.spec_seed_hidden = Some(hid)
  └─ 滚动裸步成本/降级计数（原路）

scheduler::SpecRound（探测/重入）
  └─ execute_spec_round
       └─ acquire_drafts
            ├─ spec_drafts_host = None（B4 降级时已清）
            └─ propose_first(anchor=nt, pos+1, seed_hidden)
                 ├─ mtp_chain_bt(pos+depth-1)   # 链已在追赶期补齐
                 └─ MtpPredictor::propose(...)  # 产出新 drafts
```

---

## 4. 验收门

### 4.1 正确性

- `gpu_spec_identity_gate_27b`：MTP + `degrade_after > 0`，spec on ≡ spec off
  逐位一致（含一次人工降级 + 探测重入）。
- 追赶后重入 `m >= 1`（日志/metrics）；不得出现 `spec seed 缺失`。
- MTP KV 链无洞：追赶计数 = 裸步数；重入后 `propose_first` 覆盖写幂等。

### 4.2 性能

- 追赶成本 ≤ 0.5ms/裸步（目标；若 >1ms 则 B4 自动降级本身可能净亏）。
- prose/随机域：AL 低于盈亏线时自动降级，探测后能恢复。
- MATH 域：不误降级（窗口 AL 高于盈亏线时保持 spec）。
- 对比无追赶臂：降级期间 decode 不劣于纯裸步 + 追赶成本。

### 4.3 指标

- `spec.decode.sync`（追赶耗时，已有 DFlash2 同名）
- `spec.degrade` / `spec.undegrad`（已有）
- `spec.seed.reharvest`（新增，重入 seed 次数）
- `spec.depth.active`（已有）

---

## 5. 风险与开放问题

| 风险 | 影响 | 缓解 |
|---|---|---|
| 追赶成本过高 | 降级期反而更慢 | 先测单步成本；必要时批量追赶（2–4 步一写） |
| seed 句柄被下一 replay 覆写 | 重入用错 hidden | seed 只在 probe 前一步有效；scheduler 保证 probe 紧随；日志校验 |
| TP2 双 rank | 只追 rank0 ⇒ 两 rank 链不同 | 与 `fold_step_dev` 同律，双 rank 镜像 |
| 跨 turn 前缀复用 | 上 turn 追赶行是否覆盖 | 审计 `prefill_extend` 的 base 语义；必要时加链完整性断言 |
| 与 T1b 自适应叠加 | 降级 + 降 depth 双状态 | B4 只降 spec on/off；T1b 只降 depth；两状态独立记录 |
| 图拓扑变化 | 旧档捕获/恒等门回归 | `hid` 输出只在 MTP + `degrade_after > 0` 声明；默认关 |

开放问题：

1. 追赶行与 `propose_first` extend 行数值是否逐位一致（target hidden 同源）？
2. 降级期是否需要继续维护 `mtp_block_table` 的页表（`write_bt_mtp`）？
3. probe 前是否必须保证“至少 depth 个追赶行”才能重入？
4. 采样域下 seed hidden 是否同样适用（`row_logits` 采样路径）？

---

## 6. 施工顺序

1. `RankDev::prefill_extend_dev` 重构（不改语义，单测/恒等门）。
2. session 图加 `hid` 输出（MTP + `degrade_after > 0`），捕获验证。
3. `execute_decode` MTP 追赶分支 + `spec_seed_hidden` 回填。
4. B4 门放行 MTP；补 metrics/log。
5. 恒等门（d3/d7） + 降级/重入 A/B。
6. 文档回填交接卷 §7。

---

## 7. 首轮试做记录（2026-10-22）

按本卷方案实际改过一版，结论：**追赶链路能跑通不报错，但重入草稿仍全错，
已回退，保持默认安全。**

### 7.1 试做内容

- session 图加 `hid [1, hidden]` 输出（MTP + `degrade_after > 0`）；
- `execute_decode` 裸步尾部调 `prefill_extend(bank, sid, &hid, 1, [token, nt], pos)`；
- `spec_seed_hidden` 回填 graph `hid`；
- B4 门放行 MTP；
- 修掉 `propose_first` 对图输出 seed 的 **ghost free**（加 `spec_seed_graph_owned` 标志，
  图输出 seed 不 free）。

### 7.2 实测

`degrade_after=1` + DOCS 6.7k（单 3090 Ti）：

- 降级/追赶链路不报错，`spec.seed.reharvest` 按裸步数增长；
- 但 4 次 probe 全部 `m=0`，且 `drafts` 与 target `ids` 完全无关；
- 例：`drafts=[11176, 11, 18100]` vs `ids=[21265, 29997, 55968]`；
- `spec.undegrad=0`，最终 256 tok 里大部分走裸 decode。

### 7.3 未定谳的根因候选

1. `prefill_extend` 的 `t_rows=1` 路径可能退化：prefill 生产路径恒
   `t_rows = chunk-1 ≥ 2`，单行写（`cu=[0,1]`）未在 MTP 上验证过；
   若 K0/GDN 写核在 t=1 下不写或写错，追赶行就是空的。
2. seed hidden 与 MTP 期望的 `h_{pos-1}` 是否严格同源（decode graph
   `last_hidden` vs prefill harvest hidden）未对拍。
3. MTP 链页表在追赶期只 `ensure_for_len` 未 `write_bt_mtp`，是否影响
   `propose_first` 的读表未验证。
4. `propose_first` 覆盖写 `pos-1` extend 行与追赶行的数值一致性未对拍。

### 7.4 下一步（施工前）

- 先做 **MTP KV 行对拍**：追赶写 1 行 vs 同位置 prefill_extend 写多行，读回
  KV 逐位比较；
- 若 t=1 写核有问题，改 **批量追赶**（保留最近 2–4 个 hidden/token，一次
  `prefill_extend(t_rows=2..4)`）；
- seed 对拍：decode graph hidden vs prefill harvest hidden，同 ctx 下逐位比较；
- 全部对拍绿后再重开 B4 门。

---

## 8. 对拍结果（2026-10-22）

新增 e2e 门 `gpu_mtp_kv_parity`（`crates/engine/src/tests.rs`，
`OWL_E2E_ONLY=gpu_mtp_kv_parity`）：

- 多行 `prefill_extend(t_rows=4)` 写 row 0..3；
- 逐行 `t_rows=1` 写零 hidden（哨兵）⇒ 该行 `propose(k=1)` draft 必须变化；
- 逐行 `t_rows=1` 写回原 hidden ⇒ draft 必须等于多行基线。

结果：**PASS**。即 `prefill_extend` 的 `t_rows=1` 与多行写在同输入下
MTP draft 逐位一致 ⇒ **候选 1（t=1 写核退化）排除**。

剩余候选（按优先级）：

1. seed hidden 同源：decode graph `last_hidden` vs prefill harvest hidden
   是否逐位一致（同 ctx 对拍）；
2. MTP 链页表：追赶期只 `ensure_for_len` 未 `write_bt_mtp`，是否让
   `propose_first` 读到旧表；
3. `propose_first` 覆盖写 `pos-1` extend 行与追赶行的数值一致性；
4. 降级前最后一次 propose 写入的未来链行（self-generated hidden）是否
   污染了 `propose_first` 的链首读。

下一刀：写 **seed hidden 对拍** e2e（同 ctx：prefill 末 hidden vs
decode graph hidden），再查 `write_bt_mtp` 时序。
