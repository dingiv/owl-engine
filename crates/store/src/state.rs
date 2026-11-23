//! GDN(线性注意力)状态账(`StateStore`)—— 与 `KvStore` **平级独立**(裁决②)。
//!
//! 结构性差异(设计卷 §3.1 表):几何 = 每层 4 块(conv×3 + rec)、**一格一会话**、
//! **不可内容哈希**(依赖 hidden/递推)⇒ 只能按块边界(`SnapKey`)标识。
//!
//! 本刀(S1)落**账与生命周期语义**:
//! - **行(row)= 活跃 turn 借还**(S2b 的目标语义,先在账上成立):
//!   `borrow_row(owner)` 拿行 → 用毕 `return_row(row, Some(snap))` **落快照后还格**;
//! - 快照 = 可驱逐资产(所在层由 `Tier` 记账;下沉 host/盘是 S2 的搬运活);
//! - 行数上限 = `gdn_slots`(**并发 turn 天花板**,不是"能挂多少会话")。

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{StoreError, StoreResult};
use crate::id::SnapKey;
use crate::policy::{EvictCandidate, Leg};
use crate::tier::{Owner, Tier};

/// 快照账。
#[derive(Clone, Copy, Debug)]
struct SnapEntry {
    tier: Tier,
    bytes: usize,
    last_use: u64,
    /// 重要性档(引擎给的事实;政策在 `plan_snap_offload`)
    durability: SnapDurability,
    /// **会话归属**(0 = 无归属;会话感知淘汰用)。
    session: u64,
    /// **边界块号**(同会话内深度大者先逐 —— 末端优先,保留前缀)。
    depth: usize,
    /// **持久快照角色**(Tail / History；2026-11-18 v4；不是 pin)。
    /// `Transient` 不进本字段 —— 由 `durability` 表达。
    role: SnapRole,
}

/// 快照的重要性档(2026-10-19;§9.1(6)a 轮内页边界快照的配套)。
///
/// 事实面:块边界快照有两类来源 —— **前缀段**(prompt 的页边界,内容跨会话
/// 可共享)与**生成段**(解码相位逐页边界,链一往前就只剩"更短前缀"这点价值)。
/// store 据此定政策:
///
/// | 档 | 逐出时 | 理由 |
/// |---|---|---|
/// | [`Durable`](Self::Durable) | 可下沉宿主(占宿主带宽) | 跨会话共享前缀靠它 |
/// | [`Transient`](Self::Transient) | **丢** | 下沉一条 159 MiB ≈ **50 ms**(实测),
///   而生成段每 32 tok 就出一条 ⇒ 全下沉 = 每 token 5 MiB 级 PCIe 流量,不可承受 |
///
/// ⚠️ 不变量:**最新的一条必须留在设备**(调用方每拍必取槽;LRU 保证最新最热 ⇒
/// 逐出的总是最旧的 ⇒ 链头那条在看)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapDurability {
    /// 前缀段边界(长期价值;可下沉)
    Durable,
    /// 生成段边界(随链推进即被取代;逐出即丢)
    Transient,
}

/// 快照的**持久角色**（2026-11-18 v4；不是 pin，是淘汰顺序）。
///
/// 淘汰顺序（小 = 先走）：
/// 1. `History` —— Tail 之前的均匀历史点；
/// 2. `Tail` —— 最近 `snap_tail_keep` 张连续尾快照，最后才动。
///
/// `Transient` 不再作为 role：生成段 rollback 由
/// [`SnapDurability::Transient`] 表达，逐出即丢且不计入会话配额。
///
/// 任何角色都**不是永久保留**：空间不足时 Tail 也会被逐出 —— 否则就是
/// 资源泄漏。保护面只应包含“当前正在使用”的 key（active turn / capture 目标）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SnapRole {
    /// Tail 之前的均匀历史点（优先淘汰）。
    History,
    /// 最近尾部连续快照（最后淘汰，但可淘汰）。
    Tail,
}

impl SnapRole {
    /// 淘汰排序键（小 = 先淘汰）。`Transient` 由 durability 先处理。
    pub fn evict_rank(self) -> u8 {
        match self {
            SnapRole::History => 0,
            SnapRole::Tail => 1,
        }
    }
}

/// **快照准入策略**（2026-11-18 v4；纯账，不搬内容）。
///
/// 目的：不是每个边界都值得占一格 159 MiB。真正决定“要不要拍”的是
/// **角色 + 会话资产上限**，而不是设备的 LRU 偶然性。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapAdmitPolicy {
    /// 每会话持久快照上限（Tail + History；只数**非 Transient** 条目；
    /// 0 = 不限 = 旧行为）。
    pub session_cap: usize,
}

impl Default for SnapAdmitPolicy {
    fn default() -> Self {
        Self { session_cap: 0 }
    }
}

/// **准入结果**(引擎据此决定是否 `snap_ensure_slot` / 拷贝 159 MiB)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapAdmit {
    /// 直接新增(占一格)。
    Add,
    /// 同键已在账:只触达/升级角色,不占新格。
    Touch,
    /// 先淘汰 `victim` 再新增(总数不变;引擎负责释放设备槽/宿主/盘副本)。
    Replace(SnapKey),
    /// 不值得占格:**不拷贝、不选槽**。
    Skip(SnapSkip),
}

/// 准入跳过的原因(可观测;不是静默失败)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapSkip {
    /// 会话已满，且没有未保护的 victim 可替换。
    SessionFull,
}

/// GDN 状态账。
#[derive(Debug)]
pub struct StateStore {
    rows_total: usize,
    rows_free: BTreeSet<usize>,
    /// 行 → 归属(turn/session/system;报告面 P1-5)
    row_owners: BTreeMap<usize, Owner>,
    row_bytes: usize,
    snaps: BTreeMap<SnapKey, SnapEntry>,
    tick: u64,
    hits: u64,
    misses: u64,
}

impl StateStore {
    /// 构造。`rows_total` = `gdn_slots`;`row_bytes` = 每会话全 48 层的行合计
    /// (27B 实测 ≈162MB)。
    pub fn new(rows_total: usize, row_bytes: usize) -> Self {
        Self {
            rows_total,
            rows_free: (0..rows_total).collect(),
            row_owners: BTreeMap::new(),
            row_bytes,
            snaps: BTreeMap::new(),
            tick: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// 每行字节(预算/报告用)。
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// 行总数(= 并发 turn 天花板)。
    pub fn rows_total(&self) -> usize {
        self.rows_total
    }

    /// 借行(**活跃 turn 起**)——耗尽 = 响亮拒;调用方应先查 `rows_free()`。
    ///
    /// `owner` = 行归属（正常 = `Owner::Turn(turn)`）；报告面据此按 turn 归因。
    pub fn borrow_row(&mut self, owner: Owner) -> StoreResult<usize> {
        // 一个 owner 至多一行：`release_row_of` 只释放第一行，重复借行
        // 会让第二行永久占格（直到进程重启）。
        if self.row_owners.values().any(|o| *o == owner) {
            return Err(StoreError::Invariant(format!(
                "owner {} 已占一行，不能重复借行",
                owner.label()
            )));
        }
        let row =
            *self.rows_free.iter().next().ok_or_else(|| {
                StoreError::Exhausted(format!("GDN 行耗尽({} 行)", self.rows_total))
            })?;
        self.rows_free.remove(&row);
        self.row_owners.insert(row, owner);
        Ok(row)
    }

    /// 还行(**活跃 turn 止**;`snap = Some` 时先把状态登记为快照)。
    ///
    /// 这正是设计卷 §2.3 的改造点:idle 会话只留快照(可驱逐/可下沉),
    /// **不再占行** —— 于是"能挂多少会话"与"能并发几个 turn"解耦。
    pub fn return_row(&mut self, row: usize, snap: Option<SnapKey>) -> StoreResult<()> {
        if row >= self.rows_total {
            return Err(StoreError::Invariant(format!(
                "行号 {row} 越界(共 {})",
                self.rows_total
            )));
        }
        if !self.rows_free.insert(row) {
            return Err(StoreError::Invariant(format!("行 {row} 重复归还")));
        }
        self.row_owners.remove(&row);
        if let Some(k) = snap {
            // **同键重拍不得覆盖既有归属/角色**：行归还快照是 Tail 尾锚点，
            // 但若该键此前已被某会话登记，`session`/`depth` 必须保留。
            self.register_snap_owned_role(
                k,
                self.row_bytes,
                Tier::Gpu,
                SnapDurability::Durable,
                crate::policy::NO_SESSION,
                0,
                SnapRole::Tail,
            );
        }
        Ok(())
    }

    /// **按归属还一行**（turn 终局单一围栏用；幂等）。
    ///
    /// 找不到该 owner 的行 = `Ok(None)`（已还/从未借）；同一 owner 至多一行。
    pub fn release_row_of(&mut self, owner: Owner) -> StoreResult<Option<usize>> {
        let row = self
            .row_owners
            .iter()
            .find(|(_, o)| **o == owner)
            .map(|(r, _)| *r);
        match row {
            Some(row) => {
                self.return_row(row, None)?;
                Ok(Some(row))
            }
            None => Ok(None),
        }
    }

    /// 空闲行数。
    pub fn rows_free(&self) -> usize {
        self.rows_free.len()
    }

    /// 行归属(报告面;P1-5)。
    pub fn row_owner(&self, row: usize) -> Option<Owner> {
        self.row_owners.get(&row).copied()
    }

    /// 当前所有在借行的 `(row, owner)`（报告/对账口）。
    pub fn row_owners(&self) -> impl Iterator<Item = (usize, Owner)> + '_ {
        self.row_owners.iter().map(|(row, owner)| (*row, *owner))
    }

    /// **登记快照**(S2:引擎每次 `capture_snap` 后调用;幂等 —— 已存在则只触达)。
    ///
    /// `bytes` = 一条快照的全量字节(Σ 层×quad 行宽);`tier` 由调用方给
    /// (新拍的必在设备层);`durability` = 重要性档(见 [`SnapDurability`])。
    pub fn register_snap(
        &mut self,
        key: SnapKey,
        bytes: usize,
        tier: Tier,
        durability: SnapDurability,
    ) {
        self.register_snap_owned(key, bytes, tier, durability, crate::policy::NO_SESSION, 0);
    }

    /// **带会话归属的登记**(2026-10-19 会话感知淘汰):`session` = 该快照属于谁,
    /// `depth` = 边界块号(同会话内**深者先逐**,保留链头前缀)。
    pub fn register_snap_owned(
        &mut self,
        key: SnapKey,
        bytes: usize,
        tier: Tier,
        durability: SnapDurability,
        session: u64,
        depth: usize,
    ) {
        // 旧入口（测试/旁路）：持久快照默认 History；Tail 由 capture/return_row
        // 显式给。Transient 只由 durability 表达。
        self.register_snap_owned_role(
            key,
            bytes,
            tier,
            durability,
            session,
            depth,
            SnapRole::History,
        );
    }

    /// **带角色的登记**（2026-11-18 v4）：capture 路径用它标 Tail/History。
    ///
    /// **角色以最新布局为准**：同键重拍时允许 `Tail → History` 降级 ——
    /// 会话增长后旧的尾快照会变成中段历史点，布局重算会重新分类。
    ///
    /// 唯一例外：**`Transient` 不覆盖持久条目**。生成段 rollback 边界若与
    /// 之前 Durable 的锚点同键，重拍不能把锚点降成“逐出即丢”。
    pub fn register_snap_owned_role(
        &mut self,
        key: SnapKey,
        bytes: usize,
        tier: Tier,
        durability: SnapDurability,
        session: u64,
        depth: usize,
        role: SnapRole,
    ) {
        self.tick += 1;
        let tick = self.tick;
        match self.snaps.get_mut(&key) {
            Some(e) => {
                e.last_use = tick;
                e.tier = tier;
                e.bytes = bytes;
                // 角色以最新布局为准；`Transient` 不覆盖持久条目（见方法文档）。
                if durability != SnapDurability::Transient {
                    e.role = role;
                    e.durability = durability;
                } else if e.durability == SnapDurability::Transient {
                    e.durability = durability;
                    e.role = role;
                }
                if session != crate::policy::NO_SESSION {
                    e.session = session;
                }
                if depth != 0 {
                    e.depth = depth;
                }
            }
            None => {
                self.snaps.insert(
                    key,
                    SnapEntry {
                        tier,
                        bytes,
                        last_use: tick,
                        durability,
                        session,
                        depth,
                        role,
                    },
                );
            }
        }
    }

    /// 一条快照的持久角色（不在账 = `None`；Transient 也有占位 role=History，
    /// 但调用方应先看 [`Self::snap_durability`]）。
    pub fn snap_role(&self, key: SnapKey) -> Option<SnapRole> {
        self.snaps.get(&key).map(|s| s.role)
    }

    /// **本会话的非 Transient 资产计数**。
    ///
    /// Transient 是生成段 rollback 快照（设备槽内自限、逐出零成本），
    /// 不计入会话资产上限 —— 否则一次长 decode 的页边界会把持久配额吃光。
    fn session_live(&self, session: u64) -> usize {
        self.snaps
            .values()
            .filter(|e| e.session == session && e.durability != SnapDurability::Transient)
            .count()
    }

    /// **快照准入（纯账；2026-11-18 v4）**。
    ///
    /// 规则（与设计 v4 对齐）：
    /// 1. 同键 ⇒ `Touch`（不占新格；角色由 `register_snap_owned_role` 更新）；
    /// 2. `Transient` / 无会话归属 ⇒ `Add`（rollback 正确性优先，旧行为不变）；
    /// 3. `session_cap = 0` ⇒ `Add`（旧行为，不启用 per-session 上限）；
    /// 4. 还有空位 ⇒ `Add`；
    /// 5. 满了 ⇒ 先选 `History` victim，History 空后才动 `Tail`；
    ///    同角色内 History 按 LRU，Tail 按边界深度（最老先走）。
    ///
    /// ⚠️ **不是 pin**：Tail 只是最晚被选；受保护 key 不选。
    pub fn admit_snap(
        &self,
        session: u64,
        key: SnapKey,
        durability: SnapDurability,
        policy: SnapAdmitPolicy,
        protect: &[SnapKey],
    ) -> SnapAdmit {
        if self.snaps.contains_key(&key) {
            return SnapAdmit::Touch;
        }
        if session == crate::policy::NO_SESSION
            || durability == SnapDurability::Transient
            || policy.session_cap == 0
        {
            return SnapAdmit::Add;
        }
        if self.session_live(session) < policy.session_cap {
            return SnapAdmit::Add;
        }
        // 需要替换：本会话、非 Transient、未被保护。History 先走；Tail 只在
        // History 空后动，且从最老的边界开始（最新尾锚点最后）。
        let victim = self
            .snaps
            .iter()
            .filter(|(k, e)| {
                e.session == session
                    && e.durability != SnapDurability::Transient
                    && !protect.contains(k)
            })
            .min_by_key(|(_, e)| match e.role {
                SnapRole::History => (0u8, e.last_use, 0usize),
                SnapRole::Tail => (1u8, 0, e.depth),
            })
            .map(|(k, _)| *k);
        match victim {
            Some(k) => SnapAdmit::Replace(k),
            None => SnapAdmit::Skip(SnapSkip::SessionFull),
        }
    }

    /// **只重分类，不搬内容**：布局重算后把同键条目从 Tail 降为 History
    /// 或反向升级。重分类不改 `last_use`（不是一次“使用”）。
    ///
    /// - 不存在 = `false`（已被逐出是正常事件，不是错误）；
    /// - 重分类只写持久条目；Transient 的升级走 capture。
    pub fn reclassify_snap(&mut self, key: SnapKey, role: SnapRole) -> bool {
        match self.snaps.get_mut(&key) {
            Some(e) if e.durability != SnapDurability::Transient => {
                e.role = role;
                true
            }
            _ => false,
        }
    }

    /// **会话关闭：释放 L1 归属**。
    ///
    /// 不丢内容（内容身份可被其他会话复用）：非 Transient 条目改为
    /// **无归属 + History**；Transient 保持 Transient。返回改写的条数。
    pub fn detach_session(&mut self, session: u64) -> usize {
        let mut n = 0usize;
        for e in self.snaps.values_mut() {
            if e.session != session {
                continue;
            }
            e.session = crate::policy::NO_SESSION;
            if e.durability != SnapDurability::Transient {
                e.role = SnapRole::History;
            }
            n += 1;
        }
        n
    }

    /// **会话归属与边界深度**(不在账 = `None`;诊断/门用)。
    pub fn snap_session(&self, key: SnapKey) -> Option<(u64, usize)> {
        self.snaps.get(&key).map(|s| (s.session, s.depth))
    }

    /// 一条快照的字节(不在账 = `None`)。
    pub fn snap_bytes(&self, key: SnapKey) -> Option<usize> {
        self.snaps.get(&key).map(|s| s.bytes)
    }

    /// 一条快照的重要性档(不在账 = `None`)。
    pub fn snap_durability(&self, key: SnapKey) -> Option<SnapDurability> {
        self.snaps.get(&key).map(|s| s.durability)
    }

    /// 逐页生成段快照的条数(诊断口径:应≈设备槽数,不应无限增长)
    pub fn snaps_transient(&self) -> usize {
        self.snaps
            .values()
            .filter(|s| s.durability == SnapDurability::Transient)
            .count()
    }

    /// 按层统计(`(条数, 字节)`;报告/对账口)。
    pub fn snaps_in_tier(&self, tier: Tier) -> (usize, usize) {
        self.snaps
            .values()
            .filter(|s| s.tier == tier)
            .fold((0, 0), |(n, b), s| (n + 1, b + s.bytes))
    }

    /// 定位快照(probe:纯查询)。
    pub fn locate(&self, key: SnapKey) -> Option<Tier> {
        self.snaps.get(&key).map(|s| s.tier)
    }

    /// **快照键重挂**(前缀无感删除;2026-10-23):同一条快照换一个边界身份。
    ///
    /// 用途:删掉中段之后,后缀的链身份整体变了(父链哈希变)—— 但**状态内容
    /// 仍在原槽/原缓冲/原文件**(删除是"搬运记忆",不是重算)。重挂只动账,
    /// 内容零搬运;层随条目**原样跟着走**(设备/宿主/盘三层一视同仁)。
    ///
    /// 注:本方法只管**账**。内容侧的重挂由调用方分头做
    /// (`Rel::rekey_snap` 管宿主、engine 的盘层重挂走 plan/commit + rename、
    /// 设备槽由池按 key 重标记)—— 三者都做完才算"新身份完整在场"。
    ///
    /// 语义边界(响亮):
    /// - `old` 不在账 ⇒ `false`(调用方当"没重挂",不许猜);
    /// - `new` 已在账 ⇒ `false`(拒绝覆盖 —— 否则账与槽位/文件映射会分叉)。
    pub fn rekey_snap(&mut self, old: SnapKey, new: SnapKey) -> bool {
        if self.snaps.contains_key(&new) {
            return false;
        }
        let Some(e) = self.snaps.remove(&old) else {
            return false;
        };
        self.snaps.insert(new, e);
        true
    }

    /// 触达快照(命中新鲜度)。
    pub fn touch(&mut self, key: SnapKey) {
        self.tick += 1;
        let tick = self.tick;
        if let Some(s) = self.snaps.get_mut(&key) {
            s.last_use = tick;
        }
    }

    /// 记命中/未命中(报告面)。
    pub fn note_probe(&mut self, hit: bool) {
        if hit {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
    }

    /// 快照换层(下沉/回灌)。
    pub fn set_tier(&mut self, key: SnapKey, tier: Tier) -> StoreResult<()> {
        let s = self
            .snaps
            .get_mut(&key)
            .ok_or_else(|| StoreError::Unknown(format!("快照 {key:?}")))?;
        s.tier = tier;
        Ok(())
    }

    /// 驱逐候选(快照面全部可驱逐;`protect` = 点名保住)。
    pub fn evict_candidates(&self, protect: &[SnapKey]) -> Vec<EvictCandidate> {
        self.snaps
            .iter()
            .filter(|(k, _)| !protect.contains(k))
            .map(|(k, s)| EvictCandidate {
                leg: Leg::State,
                tier: s.tier,
                bytes: s.bytes,
                last_use: s.last_use,
                chain: None,
                snap: Some(*k),
                session: s.session,
                depth: s.depth,
            })
            .collect()
    }

    /// 回收快照(调用方已从 `TierLedger` 出账)。
    pub fn drop_snap(&mut self, key: SnapKey) -> StoreResult<usize> {
        let s = self
            .snaps
            .remove(&key)
            .ok_or_else(|| StoreError::Unknown(format!("快照 {key:?}")))?;
        Ok(s.bytes)
    }

    /// 统计(报告面)。
    pub fn stats(&self) -> StateStats {
        StateStats {
            rows_total: self.rows_total,
            rows_free: self.rows_free.len(),
            snaps: self.snaps.len(),
            snap_bytes: self.snaps.values().map(|s| s.bytes).sum(),
            snap_gpu: self.snaps_in_tier(Tier::Gpu),
            snap_host: self.snaps_in_tier(Tier::Host),
            hits: self.hits,
            misses: self.misses,
        }
    }
}

/// `StateStore` 统计。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StateStats {
    /// 行总数。
    pub rows_total: usize,
    /// 空闲行数。
    pub rows_free: usize,
    /// 快照条数。
    pub snaps: usize,
    /// 快照字节合计。
    pub snap_bytes: usize,
    /// 设备层快照(条数, 字节)。
    pub snap_gpu: (usize, usize),
    /// 宿主层快照(条数, 字节)。
    pub snap_host: (usize, usize),
    /// 命中次数。
    pub hits: u64,
    /// 未命中次数。
    pub misses: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::ChainHash;

    #[test]
    fn duplicate_owner_borrow_is_loud() {
        let mut s = StateStore::new(2, 1);
        s.borrow_row(Owner::Turn(1)).expect("首行");
        let err = s
            .borrow_row(Owner::Turn(1))
            .expect_err("同一 owner 不得借两行");
        assert!(matches!(err, StoreError::Invariant(_)), "得 {err:?}");
        assert_eq!(s.rows_free(), 1, "失败不得消耗行");
    }

    #[test]
    fn return_row_snapshot_preserves_existing_owner() {
        let mut s = StateStore::new(1, 4096);
        let key = SnapKey::at_boundary(ChainHash::new(0x71));
        s.register_snap_owned(key, 4096, Tier::Gpu, SnapDurability::Durable, 7, 3);
        let row = s.borrow_row(Owner::Turn(1)).expect("借行");
        s.return_row(row, Some(key)).expect("还行并落快照");
        assert_eq!(
            s.snap_session(key),
            Some((7, 3)),
            "同键重拍必须保留既有会话归属/深度"
        );
    }

    #[test]
    fn rows_are_borrowed_per_turn_and_returned_with_snapshot() {
        let mut s = StateStore::new(2, 162 << 20);
        let r0 = s.borrow_row(Owner::Turn(1)).expect("借行");
        let r1 = s.borrow_row(Owner::Turn(2)).expect("借行");
        assert_eq!(s.row_owner(r0), Some(Owner::Turn(1)), "行归属 = turn");
        assert!(
            s.borrow_row(Owner::Turn(3)).is_err(),
            "第三行必须响亮拒(天花板)"
        );
        // turn 止:落快照 + 还行 ⇒ idle 会话不再占行
        s.return_row(r0, Some(SnapKey::at_boundary(ChainHash::new(7))))
            .expect("还行");
        assert_eq!(s.rows_free(), 1);
        assert_eq!(
            s.locate(SnapKey::at_boundary(ChainHash::new(7))),
            Some(Tier::Gpu)
        );
        assert_eq!(s.row_owner(r0), None, "还行后归属清空");
        assert_eq!(
            s.borrow_row(Owner::Turn(4)).expect("再借"),
            0,
            "还回的行号应可复用"
        );
        s.return_row(r1, None).expect("无快照也行");
        assert_eq!(s.rows_free(), 1);
    }

    /// **快照键重挂**(前缀无感删除):只动账;三个响亮拒绝各一例。
    #[test]
    fn rekey_snap_moves_account_without_copy() {
        let mut s = StateStore::new(1, 1);
        let old = SnapKey::at_boundary(ChainHash::new(1));
        let new = SnapKey::at_boundary(ChainHash::new(2));
        assert!(!s.rekey_snap(old, new), "不在账 ⇒ false(不许猜)");
        s.register_snap(old, 4096, Tier::Gpu, SnapDurability::Durable);
        assert!(s.rekey_snap(old, new), "设备层在账 ⇒ 重挂");
        assert_eq!(s.locate(old), None, "旧键必须消失");
        assert_eq!(s.locate(new), Some(Tier::Gpu), "新键拿同一份账");
        assert_eq!(s.snap_bytes(new), Some(4096), "字节随账走");
        // 目标已存在 ⇒ 不许覆盖(账与槽位映射会分叉)
        let third = SnapKey::at_boundary(ChainHash::new(3));
        s.register_snap(third, 1, Tier::Gpu, SnapDurability::Durable);
        assert!(!s.rekey_snap(third, new), "目标已在账 ⇒ false");
        // 宿主/盘层也重挂(内容侧由 `Rel::rekey_snap` / 盘层 plan+commit 负责)
        let host = SnapKey::at_boundary(ChainHash::new(4));
        s.register_snap(host, 1, Tier::Host, SnapDurability::Durable);
        let host_new = SnapKey::at_boundary(ChainHash::new(5));
        assert!(s.rekey_snap(host, host_new), "宿主层在账 ⇒ 重挂");
        assert_eq!(s.locate(host_new), Some(Tier::Host), "层随账走");
        // 盘层同理
        let disk = SnapKey::at_boundary(ChainHash::new(6));
        s.register_snap(disk, 1, Tier::Disk, SnapDurability::Durable);
        assert!(s.rekey_snap(disk, SnapKey::at_boundary(ChainHash::new(7))));
        assert_eq!(
            s.locate(SnapKey::at_boundary(ChainHash::new(7))),
            Some(Tier::Disk)
        );
    }

    #[test]
    fn double_return_is_loud() {
        let mut s = StateStore::new(1, 1);
        let r = s.borrow_row(Owner::System).expect("借行");
        s.return_row(r, None).expect("还行");
        assert!(matches!(
            s.return_row(r, None),
            Err(StoreError::Invariant(_))
        ));
    }

    #[test]
    fn snaps_are_evictable_but_protect_wins() {
        let mut s = StateStore::new(1, 1024);
        let k = SnapKey::at_boundary(ChainHash::new(3));
        let r = s.borrow_row(Owner::System).expect("借行");
        s.return_row(r, Some(k)).expect("落快照");
        assert!(s.evict_candidates(&[k]).is_empty(), "点名保住");
        assert_eq!(s.evict_candidates(&[]).len(), 1);
    }

    fn put(s: &mut StateStore, session: u64, n: u64, role: SnapRole) -> SnapKey {
        let key = SnapKey::at_boundary(ChainHash::new(n));
        s.register_snap_owned_role(
            key,
            1024,
            Tier::Gpu,
            SnapDurability::Durable,
            session,
            n as usize,
            role,
        );
        key
    }

    fn put_transient(s: &mut StateStore, session: u64, n: u64) -> SnapKey {
        let key = SnapKey::at_boundary(ChainHash::new(n));
        s.register_snap_owned_role(
            key,
            1024,
            Tier::Gpu,
            SnapDurability::Transient,
            session,
            n as usize,
            SnapRole::History,
        );
        key
    }

    fn policy(cap: usize) -> SnapAdmitPolicy {
        SnapAdmitPolicy { session_cap: cap }
    }

    /// 同键重拍：角色以最新布局为准（Tail → History 可降级）；
    /// `Transient` 不能覆盖持久条目。
    #[test]
    fn recapture_reclassifies_but_transient_never_downgrades() {
        let mut s = StateStore::new(1, 1024);
        let k = SnapKey::at_boundary(ChainHash::new(0xF1));
        s.register_snap_owned_role(
            k,
            1,
            Tier::Gpu,
            SnapDurability::Durable,
            1,
            10,
            SnapRole::Tail,
        );
        assert_eq!(s.snap_role(k), Some(SnapRole::Tail));
        s.register_snap_owned_role(
            k,
            1,
            Tier::Gpu,
            SnapDurability::Durable,
            1,
            10,
            SnapRole::History,
        );
        assert_eq!(
            s.snap_role(k),
            Some(SnapRole::History),
            "会话增长后旧 tail 必须降为 History"
        );
        s.register_snap_owned_role(
            k,
            1,
            Tier::Gpu,
            SnapDurability::Durable,
            1,
            10,
            SnapRole::Tail,
        );
        assert_eq!(s.snap_role(k), Some(SnapRole::Tail), "布局可重新升级");
        s.register_snap_owned_role(
            k,
            1,
            Tier::Gpu,
            SnapDurability::Transient,
            1,
            10,
            SnapRole::History,
        );
        assert_eq!(
            s.snap_role(k),
            Some(SnapRole::Tail),
            "Transient 重拍不能把锚点降级"
        );
        assert_eq!(
            s.snap_durability(k),
            Some(SnapDurability::Durable),
            "耐久档也不能被 Transient 覆盖"
        );
    }

    /// 重分类口：存在且持久 ⇒ 改角色；不存在/Transient = false。
    #[test]
    fn reclassify_snap_updates_role_only() {
        let mut s = StateStore::new(1, 1024);
        let k = put(&mut s, 21, 100, SnapRole::Tail);
        assert!(s.reclassify_snap(k, SnapRole::History), "存在 ⇒ 改角色");
        assert_eq!(s.snap_role(k), Some(SnapRole::History));
        assert_eq!(s.snap_durability(k), Some(SnapDurability::Durable));
        let t = put_transient(&mut s, 21, 101);
        assert!(!s.reclassify_snap(t, SnapRole::Tail), "Transient 不走本口");
        let missing = SnapKey::at_boundary(ChainHash::new(102));
        assert!(
            !s.reclassify_snap(missing, SnapRole::History),
            "不在账 ⇒ false"
        );
    }

    /// 会话关闭：内容保留，归属释放为无主 History；Transient 保持 Transient。
    #[test]
    fn detach_session_releases_assets_to_unowned_history() {
        let mut s = StateStore::new(1, 1024);
        let tail = put(&mut s, 7, 100, SnapRole::Tail);
        let hist = put(&mut s, 7, 101, SnapRole::History);
        let tr = put_transient(&mut s, 7, 102);
        let keep = put(&mut s, 8, 103, SnapRole::Tail);
        assert_eq!(s.detach_session(7), 3, "三条都改归属");
        assert_eq!(
            s.snap_session(tail),
            Some((crate::policy::NO_SESSION, 100)),
            "Tail → 无主"
        );
        assert_eq!(
            s.snap_role(tail),
            Some(SnapRole::History),
            "无主锚点降为 History"
        );
        assert_eq!(s.snap_role(hist), Some(SnapRole::History));
        assert_eq!(
            s.snap_durability(tr),
            Some(SnapDurability::Transient),
            "Transient 仍由 durability 表达"
        );
        assert_eq!(s.snap_session(keep), Some((8, 103)), "别的会话不动");
        assert_eq!(s.detach_session(7), 0, "幂等:已无归属");
    }

    /// 还有空位 ⇒ 直接新增，不为“腾一格”去驱逐。
    #[test]
    fn admit_adds_while_room() {
        let mut s = StateStore::new(1, 1024);
        for n in 0..3 {
            put(&mut s, 7, 100 + n, SnapRole::Tail);
        }
        for n in 0..6 {
            put(&mut s, 7, 200 + n, SnapRole::History);
        }
        let new = SnapKey::at_boundary(ChainHash::new(400));
        assert_eq!(
            s.admit_snap(7, new, SnapDurability::Durable, policy(20), &[],),
            SnapAdmit::Add,
            "9/20 ⇒ 直接新增"
        );
    }

    /// 满了：History 先走，History 空后才动 Tail；Tail 从最老 depth 开始。
    #[test]
    fn admit_replaces_history_before_tail() {
        let mut s = StateStore::new(1, 1024);
        for n in 0..2 {
            put(&mut s, 9, 100 + n, SnapRole::Tail);
        }
        let mut hist = Vec::new();
        for n in 0..3 {
            hist.push(put(&mut s, 9, 300 + n, SnapRole::History));
        }
        let p = policy(5);
        let new_tail = SnapKey::at_boundary(ChainHash::new(500));
        assert_eq!(
            s.admit_snap(9, new_tail, SnapDurability::Durable, p, &[]),
            SnapAdmit::Replace(hist[0]),
            "先替换最冷 History，不碰 Tail"
        );
        let new_hist = SnapKey::at_boundary(ChainHash::new(501));
        assert_eq!(
            s.admit_snap(9, new_hist, SnapDurability::Durable, p, &[]),
            SnapAdmit::Replace(hist[0]),
            "新 History 也只替换最冷 History"
        );

        // History 空后：Tail 从最老（depth 小）开始，最新尾最后。
        let mut t = StateStore::new(1, 1024);
        let a = put(&mut t, 11, 100, SnapRole::Tail);
        let _b = put(&mut t, 11, 101, SnapRole::Tail);
        let new = SnapKey::at_boundary(ChainHash::new(600));
        assert_eq!(
            t.admit_snap(11, new, SnapDurability::Durable, policy(2), &[]),
            SnapAdmit::Replace(a),
            "最老 Tail 先走"
        );
    }

    /// 同键 / Transient / 无会话归属：不参与会话上限（旧行为）。
    #[test]
    fn admit_touch_transient_and_unowned_bypass_cap() {
        let mut s = StateStore::new(1, 1024);
        let k = put(&mut s, 13, 100, SnapRole::Tail);
        assert_eq!(
            s.admit_snap(13, k, SnapDurability::Durable, policy(1), &[],),
            SnapAdmit::Touch
        );
        let t = SnapKey::at_boundary(ChainHash::new(101));
        assert_eq!(
            s.admit_snap(13, t, SnapDurability::Transient, policy(1), &[],),
            SnapAdmit::Add,
            "Transient 是 rollback 正确性路径，不因会话满而跳过"
        );
        let unowned = SnapKey::at_boundary(ChainHash::new(102));
        assert_eq!(
            s.admit_snap(
                crate::policy::NO_SESSION,
                unowned,
                SnapDurability::Durable,
                policy(1),
                &[],
            ),
            SnapAdmit::Add,
            "无会话归属 = 旧行为"
        );
        let legacy = SnapKey::at_boundary(ChainHash::new(103));
        assert_eq!(
            s.admit_snap(13, legacy, SnapDurability::Durable, policy(0), &[],),
            SnapAdmit::Add,
            "cap=0 = 不启用 per-session 上限"
        );
    }

    /// 保护面（active turn / capture 目标）不能被选成 admission victim。
    #[test]
    fn admit_never_picks_protected_victim() {
        let mut s = StateStore::new(1, 1024);
        let a = put(&mut s, 17, 100, SnapRole::History);
        let b = put(&mut s, 17, 101, SnapRole::History);
        let p = policy(2);
        let new = SnapKey::at_boundary(ChainHash::new(700));
        assert_eq!(
            s.admit_snap(17, new, SnapDurability::Durable, p, &[a],),
            SnapAdmit::Replace(b),
            "保护 a ⇒ 只能换 b"
        );
        assert_eq!(
            s.admit_snap(17, new, SnapDurability::Durable, p, &[a, b],),
            SnapAdmit::Skip(SnapSkip::SessionFull),
            "全被保护 ⇒ 不替换"
        );
    }
}
