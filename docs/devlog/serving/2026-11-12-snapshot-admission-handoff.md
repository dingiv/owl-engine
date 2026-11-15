# 交接：单会话 100 消息节点 / 快照上限 20 的“新增一张快照”决策

> 2026-11-12 · 下一会话入口 · 关联：
> `docs/devlog/serving/2026-11-12-session-aware-kv-snap-设计.md`
> （§3.2.1 淘汰顺序、§6 施工状态）
> worktree：`packages/owl-worktrees/session-kv-snap`，分支 `feat/session-kv-snap`

---

## 0. 场景（单会话）

- 当前会话有 **100 个消息节点**（客户端消息边界）；
- 快照容量 **20**（`snap_max` 的简化值；真实 store 全局共享，本场景先按单会话算）；
- 当前已有 **15 张**快照；
- 现在要新增 **1 张**。问：

1. 直接新增（15→16）；
2. 先删一些，再新增。

**结论（先给）**：

- **容量还有空位（15 < 20）时，默认直接新增**；为了“新增一张”去删已有快照是浪费。
- **但新增必须过“准入”**：不是每个边界都值得占一格。只有满足以下之一才新增：
  - 结构档（`Dense` / `Binary`）；
  - `Extra` 且与最近快照的距离 ≥ `min_gap`，且 Extra 数量未到“保留位”上限；
  - 热 Extra（最近被复用/命中次数高）。
- **容量满（20）或 Extra 超保留位时，才走“先删后加”**：
  - 按淘汰顺序选 victim：`Transient → Extra（最冷） → Binary（老节点先） → Dense（最后）`；
  - **只有新快照比 victim 更值得留**（档位更高，或同档更热/更近）才替换；否则**跳过新增**，不驱逐。
- **任何档位都不是永久保留**：`Dense` 只是最后被淘汰；没有 pin，不会泄漏。

---

## 1. 为什么“15/20 直接加”不一定对

如果只按“有空位就加”：

- 100 个消息节点会产生大量 `Extra`（每个消息边界都值得拍吗？）；
- 15 张里可能挤满了最近 15 个消息边界，**二分节点（50/25/12/6/3）一个都没有**；
- 下一次要深回撤时，最近可用快照可能在 10 号节点——重算 90 条消息，二分保证失效。

所以“直接加”的前提是：**新增的是结构档，或 Extra 但保留了结构档的储备位**。
15/20 的 5 个空位不是“随便填”，其中一部分应预留给未来的 `Dense`/`Binary`。

建议的保留位（单会话）：

```text
cap = 20
structural_reserve = 4~6   # 给 Dense(K=4) + 未来二分节点
extra_limit = cap - structural_reserve = 14~16
```

- `Extra` 最多占 `extra_limit`；超过就“先删最冷 Extra 再加”或直接跳过；
- `Dense`/`Binary` 不受 `extra_limit` 约束，但**总量 ≤ cap**；
- 容量满时，`Extra` 先走，`Binary` 老节点次之，`Dense` 最后。

---

## 2. 准入决策（伪代码）

```text
fn admit(new: SnapCandidate, session: SessionSnaps, cap: usize) -> Admit {
    // 0) 去重：同一 key 已在账 ⇒ 只更新 last_use/class（不占新格）
    if session.contains(new.key) { return Admit::Touch; }

    // 1) 名额未满
    if session.len() < cap {
        if new.class == Dense || new.class == Binary { return Admit::Add; }
        // Extra：看保留位 + 距离 + 热度
        if session.extra_count() < cap - structural_reserve
            && new.distance_to_nearest >= min_gap_or_hot(new) {
            return Admit::Add;
        }
        return Admit::Skip;   // 不占格，不拷贝 159MB
    }

    // 2) 名额已满：按淘汰顺序找 victim
    let victim = session.victim_in_order(); // Transient -> Extra(cold) -> Binary(old) -> Dense(LRU)
    // 3) 只有“新比 victim 更值得留”才替换
    if rank(new) > rank(victim) { return Admit::Replace(victim); }
    if rank(new) == rank(victim) && hotter(new, victim) { return Admit::Replace(victim); }
    Admit::Skip
}
```

`rank` 与现有 `SnapClass::evict_rank()` 同源：

```text
Transient (0) < Extra (1) < Binary(level) (2, -level) < Dense (3)
```

- `rank` 大 = 更值得留（更晚淘汰）；
- `Binary` 内 `level` 越大（越老）rank 越小 ⇒ 先被替换/淘汰；
- `Dense` 最大，但**不是不可淘汰**：容量满且没有更低档位时，Dense 之间按 LRU 走。

### 对 15/20 场景的三种输入

| 新快照 | 已有 15 张的构成 | 决策 |
|---|---|---|
| `Dense`（最近 K 条） | 任意 | **直接 Add**（16）；它是最该留的档 |
| `Binary(k)` | 任意 | **直接 Add**（16）；补结构 |
| `Extra`，距最近快照 ≥ min_gap | Extra 数量 < extra_limit | **Add**（16） |
| `Extra`，距最近快照 < min_gap（冗余） | 任意 | **Skip**（不占格） |
| `Extra` | Extra 数量 ≥ extra_limit | **先删最冷 Extra 再加**（15→15，结构不变）；若新 Extra 不比最冷 Extra 热，Skip |
| 任意 | 已有 20 张 | 按 §2 比较 rank；只有新 ≥ victim 才 Replace，否则 Skip |

---

## 3. 与现有实现的距离

已落地（见设计文档 §6）：

- `SnapClass`（Transient/Extra/Binary/Dense）与 `evict_rank()`；
- `order_snap_victims` 已按档位排序（槽位淘汰路径）；
- `msg_boundary::classify_snap_points` 已把 dense/binary/extra 分类；
- turn 末 cursor 补拍升级为 `Dense`；
- `snap_guard` 只硬保护 active turn/capture，**没有 pin**。

**未落地（本交接要做的）**：

1. **准入控制**：`capture_gdn_snap_tail` / `capture_snap_at_boundary` 在
   `snap_ensure_slot` **之前**调用 `admit()`；`Skip` 直接返回，不拷贝、不占槽。
2. **保留位**：`extra_limit = cap - structural_reserve`；`structural_reserve`
   建议 4~6（单会话）。需要成为配置（`[pool] snap_structural_reserve`），
   不要写死。
3. **“先删后加”**：`Replace(victim)` 时先 `drop_snap(victim)`（含宿主/盘副本），
   再 capture 新键；**同一笔操作**内完成，避免中间态超容。
4. **去重/触达**：同 key 重拍只更新 `last_use`/class（现有
   `register_snap_captured` 已有 ghost 清理），不占新格。
5. **host-pressure 路径统一**：`order_candidates`（`plan_need` 的混合 plan）
   目前还是“Transient 优先 + 会话/LRU”，需要与 `order_snap_victims` 用同一份
   档位判据，否则 host 下沉时可能先搬 Dense。
6. **显式二分补拍（可选，严格保证）**：当前 `Binary` 是“对 dense+sparse 选点
   分类”；若要求“每级都有锚点”，prefill 时按 `P/2^k` 补拍（仍按档位淘汰）。
7. **多会话**：本交接先按单会话；store 全局共享时，`admit` 需要
   **per-session 配额**（每会话 cap 或按活跃度分配），否则一个会话的 Extra
   会挤掉另一个会话的 Dense。建议先做单会话，再加 per-session 配额。

---

## 4. 代码落点（建议）

| 位置 | 动作 |
|---|---|
| `crates/store/src/state.rs` | `StateStore::admit_snap(session, key, class, cap, reserve) -> Admit`（纯账，不搬内容） |
| `crates/store/src/rel.rs` | `Rel::admit_snap(..)` 包一层；`Replace` 时先 `drop_snap_host` + `state.drop_snap` |
| `crates/engine/src/exec/spec.rs::capture_gdn_snap_tail` | 在 `snap_on` 前调用 admit；`Skip` 早退（不 `snap_ensure_slot`、不 `snap_dev`） |
| `crates/engine/src/exec/spec.rs::capture_snap_at_boundary` | 同上；`Dense` 优先级最高，通常 Add/Replace Extra |
| `crates/store/src/hooks.rs::order_candidates` | State 腿复用 `order_snap_victims` 的档位序 |
| `crates/shared/src/config/mod.rs` | `[pool] snap_structural_reserve: usize`（默认 4） |
| 指标 | `snap.admit.add/skip/replace/evict_extra/evict_binary/evict_dense` |

注意：`cap` 是**设备槽**还是**单会话快照数**？当前 `snap_max` 是设备槽总数；
单会话 20 的语义要先定：建议先实现“**每会话快照数上限**”
（`[pool] snap_session_cap`），再映射到全局槽位。否则 20 是全局槽，
单会话 15 时别的会话可能已占满。

---

## 5. 验收（单会话）

构造一个纯 store 测试（不跑 GPU）：

```text
cap = 20
已有 15 张：4 Dense + 5 Binary + 6 Extra
新增 1 张 Extra（距最近 >= min_gap）
  ⇒ Add，总数 16；无驱逐
新增 1 张 Dense
  ⇒ Add，总数 16；
把 Extra 填到 extra_limit（如 16）后
  ⇒ 新 Extra 来时 Replace 最冷 Extra，总数不变；
把总数填到 20 后
  ⇒ 新 Binary 来时 Replace 最冷 Extra；
  ⇒ 新 Extra 来时（victim = Binary）Skip，不驱逐；
  ⇒ 新 Dense 来时 Replace 最冷 Extra / 老 Binary；
压力到只剩 Dense
  ⇒ 新 Dense 来时按 LRU 替换最老 Dense（仍可淘汰，无泄漏）。
```

再加一个引擎级 e2e（可选，需 GPU）：

- 100 条消息追加，快照数始终 ≤ 20；
- 任意回撤目标 X 的 `start = last_snap_leq(min(kv_hit, X))`，重算 ≤ X/2；
- `snap_used ≤ snap_max`，无 orphan 增长。

---

## 6. 下一步（按顺序）

> **2026-11-12 更新**：1~6 已落地（worktree 未提交）——
> `StateStore::admit_snap` + `Rel::admit_snap` / `drop_snap_asset`，
> `capture_snap_at_boundary` 早退/替换，`order_state_in_session`，
> `[pool] snap_session_cap` / `snap_structural_reserve`（3080 档 20/4）。
> **二刀**另补：L1 `msg_snap_keys` 跨轮沿用 + submit 重分类（旧 Dense 尾巴
> 降级，不搬内容）；turn 末旧 tail 降为 Extra；会话关闭把快照归属释放为
> 无主 Extra（内容保留）。**三刀（2026-11-13）**：gapped 链同会话复用
> （`AgentSession.block_hashes` + `plan_prefix_reuse` session-local hit）——
> leave-gap 后 suffix 不再每轮重算；`plan_image_leave_gap` 的第二次删除
> 也已优先走同一份 session-local 旧链（2026-11-13 补）；相邻图半删用
> 内容指纹消歧，gapped 链在 turn 间保留 block_table。剩余 = 7
> （多会话配额/公平性）与上文 §3 的显式二分补拍等。
> 详见 `2026-11-13-adjacent-image-delete-prefix-cache.md`。

1. ~~定 `cap` 语义~~：先做**每会话快照数**（`snap_session_cap`；
   0 = 旧行为），全局设备槽仍是 `snap_max`；
2. ~~加 `admit_snap` 纯账 + 单测~~（100 节点/20 容量/15 已有/新增 1 张）；
3. ~~接入 `capture_gdn_snap_tail` 的早退路径~~（Skip 不拷贝）；
4. ~~Replace 的原子性~~：先删 victim 再 capture；**拍失败不恢复**（已知取舍，
   见设计文档 §6 未做 7）；
5. ~~统一 `order_candidates` 的 State 腿~~（同会话内按档位）；
6. ~~保留位配置 + 指标~~（`snap.admit.add/touch/skip/replace`）；
7. **多会话 per-session 配额/公平性**（下一刀）：当前已有每会话上限，但
   全局 host/device 压力下跨会话仍按「冷会话先 + 档位」；未按活跃度分配。

---

## 7. 提醒

- **不要 pin**；`Dense` 只是最后淘汰，空间不足时照样走。
- 15/20 时**默认直接加**，但只加“值得留”的（结构档 / 非冗余 Extra / 热 Extra）；
- “先删后加”只用于：容量满，或 Extra 超保留位；且**新必须比 victim 更值得**，
  否则跳过——不要为了加一张差快照而删一张好快照。
- 单会话先做；多会话的配额是另一层，别把全局 20 当成单会话 20。
