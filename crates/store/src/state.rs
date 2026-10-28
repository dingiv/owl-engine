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
        let row = *self
            .rows_free
            .iter()
            .next()
            .ok_or_else(|| StoreError::Exhausted(format!("GDN 行耗尽({} 行)", self.rows_total)))?;
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
            return Err(StoreError::Invariant(format!("行号 {row} 越界(共 {})", self.rows_total)));
        }
        if !self.rows_free.insert(row) {
            return Err(StoreError::Invariant(format!("行 {row} 重复归还")));
        }
        self.row_owners.remove(&row);
        if let Some(k) = snap {
            self.tick += 1;
            let tick = self.tick;
            // 行归还时的快照 = **链头**(turn 收口那一刻的态;idle 会话靠它复活)
            // ⇒ 长期价值 ⇒ Durable(可下沉)
            self.snaps.insert(
                k,
                SnapEntry {
                    tier: Tier::Gpu,
                    bytes: self.row_bytes,
                    last_use: tick,
                    session: crate::policy::NO_SESSION,
                    depth: 0,
                    durability: SnapDurability::Durable,
                },
            );
        }
        Ok(())
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
        self.tick += 1;
        let tick = self.tick;
        match self.snaps.get_mut(&key) {
            Some(e) => {
                e.last_use = tick;
                e.tier = tier;
                e.bytes = bytes;
                e.durability = durability;
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
                    SnapEntry { tier, bytes, last_use: tick, durability, session, depth },
                );
            }
        }
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
    fn rows_are_borrowed_per_turn_and_returned_with_snapshot() {
        let mut s = StateStore::new(2, 162 << 20);
        let r0 = s.borrow_row(Owner::Turn(1)).expect("借行");
        let r1 = s.borrow_row(Owner::Turn(2)).expect("借行");
        assert_eq!(s.row_owner(r0), Some(Owner::Turn(1)), "行归属 = turn");
        assert!(s.borrow_row(Owner::Turn(3)).is_err(), "第三行必须响亮拒(天花板)");
        // turn 止:落快照 + 还行 ⇒ idle 会话不再占行
        s.return_row(r0, Some(SnapKey::at_boundary(ChainHash::new(7)))).expect("还行");
        assert_eq!(s.rows_free(), 1);
        assert_eq!(s.locate(SnapKey::at_boundary(ChainHash::new(7))), Some(Tier::Gpu));
        assert_eq!(s.row_owner(r0), None, "还行后归属清空");
        assert_eq!(s.borrow_row(Owner::Turn(4)).expect("再借"), 0, "还回的行号应可复用");
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
        assert_eq!(s.locate(SnapKey::at_boundary(ChainHash::new(7))), Some(Tier::Disk));
    }

    #[test]
    fn double_return_is_loud() {
        let mut s = StateStore::new(1, 1);
        let r = s.borrow_row(Owner::System).expect("借行");
        s.return_row(r, None).expect("还行");
        assert!(matches!(s.return_row(r, None), Err(StoreError::Invariant(_))));
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
}
