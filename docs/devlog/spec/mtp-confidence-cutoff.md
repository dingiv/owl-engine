# MTP 置信度截断 —— T2 设计与施工

> 2026-10-22 · 工作树 `packages/owl-worktrees/mtp-spec`
> 关联：
> - `mtp-longctx-handoff.md` §6（T2）
> - `mtp-longctx-windowed.md` §5（T1a/T1b 两档图族）
> - `perf-runs/mtp-longctx-2026-10-21/REPORT.md` §8/§9（greedy + 采样基线）

---

## 0. 一句话

MTP 的 `propose_ext` 只输出 drafts，没有置信度；verify 图按 active depth 定形，
即使 host 知道尾部不值得验也省不掉 verify 行。T2 的目标 = 给 propose 图加
**每步 margin 输出**，host 按置信度选**已预捕获的浅档 verify/fold/propose**，
先把“少验低价值尾部”做成真收益，而不是只改 drafts。

---

## 1. 现状

### 1.1 propose 图输出

`crates/models/src/layers/mtp.rs::propose_ext`：

```text
extend 批行 → 末行 argmax = d1
chain 每步 → logits → argmax = d_{j+1}
返回: concat_rows(drafts)  [depth, 1]
```

`crates/engine/src/graph_fleet.rs::plan_propose_family` 输出槽只有：

```text
OutputSlot { name: "drafts", shape: [depth, 1], dtype: F32 }
```

### 1.2 verify 行预算

`crates/engine/src/graph_fleet.rs::plan_verify`：

```text
T = depth + 1 定形（depth1 = boot depth + 1）
输出 tok/hid/logits + fold 记录 r{i}
```

T1a 后 `GraphFleet.depths` 已有 **每档独立** 的 verify/fold/propose；
所以“选浅档”基础设施已经存在，缺的是**何时选浅档**的输入。

### 1.3 采样域证据（REPORT §9）

| 点 | 配置 | decode | AL | verify |
|---|---|---:|---:|---:|
| MATH 8.1k | 静态 d3 采样 | 102.14 | 2.771 | 26.03 |
| MATH 32.0k | 静态 d3 采样 | 77.67 | 2.541 | 29.75 |
| MATH 8.1k | adaptive d7 采样 | 115.50 | 5.023 | 29.70 |
| MATH 31.9k | adaptive d7 采样 | 94.40 | 3.981 | 30.64 |

采样路径每轮固定税 ~3–5ms（per-row logits 回读 + host 采样）；
verify 行数下降对采样域收益可能大于 greedy。

---

## 2. 方案

### 2.1 置信度输出（第一步）

在 `propose_ext` 每步 logits 上取 top1/top2：

```text
margin_j = logit(top1) - logit(top2)      # j = 0..depth-1
```

实现路径（按成本递增）：

1. **复用 `owl_topk16_f16`**（DFlash2 已用）对每步 logits 取 top-16，
   graph 输出 `topk` `[depth, 16]`（值区；或值+索引双区）；
   host 读 topk 后算 `margin = v[0]-v[1]`（D2H 仅 depth×16 floats）。
2. 新小核 `owl_margin_top2_f16`：每行输出 1 个 f32，graph 输出 `conf [depth,1]`，
   D2H 更小、图内更省。

首版建议走 (1)：**复用现成 op，不改 kernel**；离线校准后再决定是否做 (2)。

### 2.2 host 侧策略（第二步）

`propose_graph_step` 同时读 `drafts` 与 `conf`，在轮首前给 scheduler 一个
`conf_depth`：

```text
conf_depth = 1 + max { j : margin_j >= tau }     # 至少 1（anchor）
verify_depth = clamp(conf_depth, depth_min, active_depth)
```

- `tau` 离线扫（目标：AL ≥ full 的 95–98%，verify 行数明显下降）。
- `verify_depth` 必须是 **已预捕获档**；T1a 的 d3/d7 对就是首版候选。
- 若 `verify_depth < active_depth`：本轮的 verify/fold/propose 全部用浅档；
  下轮 propose 仍按 `active_depth` 产草稿（或按浅档产，二选一，先选前者）。
- 只降不升/防抖：连续 N 轮低置信才降档，与 T1b 的低窗计数同构。

### 2.3 图族扩展（第三步，可选）

若需要比 d3 更浅（d1/d2）或任意行数：

- `GraphFleet.depths` 加 `depth_min = 1` 档；
- face 预算按 `fleet_faces(depths)` 增加；
- 仍禁止在线重捕图。

### 2.4 采样域的特殊性

采样路径已有 per-row logits（`verify` 的 `logits` 槽），target 分布天然可得；
但 **draft 置信度**仍来自 propose 图。T2 的 `conf` 输出对采样域同样适用，
且采样域更值得做（每轮固定税更大）。

---

## 3. 调用树（目标态）

```text
execute_spec_round
  ├─ acquire_drafts
  ├─ verify_round(verify_depth)        # 浅档图
  ├─ accept / fold(verify_depth)
  └─ propose_graph_step(active_depth)
       ├─ drafts = read_output("drafts")
       ├─ conf   = read_output("conf" / "topk")
       └─ tspec.conf_depth = policy(conf, tau, depth_min, active_depth)

scheduler (下一轮)
  └─ SpecRound { depth = verify_depth } # 由 conf_depth 决定
```

---

## 4. 验收门

### 4.1 离线校准

- 收集 d7 greedy 的每步 `margin_j` 与真实 `accept_j`；
- 画 `margin_j` 分桶 → 接受率曲线；
- 选 `tau` 使 AL 保持 ≥ full 的 95–98%，同时 verify 行数下降 ≥20%。

### 4.2 在线门

- `gpu_spec_identity_gate_27b`：T2 开/关都逐位一致（target 输出不变）。
- 单请求 decode：同 ctx 同域有正收益。
- 采样域：decode 正收益，AL 不低于 full 的 95%。
- 无 `margin` 与接受率弱相关导致的“降了档但 AL 崩”组合。

### 4.3 指标

- `spec.conf.depth`（host 选择结果）
- `spec.conf.margin`（窗口均值）
- `spec.verify.rows`（实际 verify 行数/轮）
- `spec.fold.rows`

---

## 5. 风险与开放问题

| 风险 | 影响 | 缓解 |
|---|---|---|
| margin 与 target 接受率弱相关 | 降档后 AL 崩 | 离线先校准；tau 宁高勿低 |
| topk16 每步成本 | propose 图变慢 | 先测；必要时新小核 |
| 图输出槽增多 | 捕获/slab 回归 | 首版只加 `topk` 输出；默认关 |
| 浅档图缺失 | 运行期错误 | 只允许已捕获档；boot 期检查 |
| 与 T1b 自适应叠加 | 两个降档状态抖动 | conf_depth 只降不升 + 低窗计数 |
| 采样/贪心两路 | 行为分叉 | 采样域单独门；`conf` 输出两路共用 |

开放问题：

1. `margin` 还是 `entropy`/`top1_prob` 与接受率更相关？
2. `verify_depth` 由 propose 阶段决定，还是 verify 前由 target logits 决定？
3. 浅档 verify 的 fold 记录行数不同，是否需要新 fold 桶？
4. 大 batch 下收益是否反转（首版单请求）。

---

## 6. 施工顺序

1. 离线校准脚本：d7 greedy 收集 `margin_j`/`accept_j`（propose 图先加 `topk` 输出）。
2. `plan_propose_family` 加 `topk` 输出槽（DFlash2 同款，零新 kernel）。
3. `propose_graph_step` 读 `topk` → `tspec.conf_depth`；日志/metrics。
4. host 策略 + scheduler 选浅档；先只允许 d3/d7。
5. 在线 A/B：greedy + 采样；恒等门。
6. 文档回填交接卷 §6。
