//! **Grant / Reservation 账**(P0-4):store 是“谁被批准占多少设备槽”的唯一真源。
//!
//! 旧形态:
//!
//! ```text
//! sched_policy::Admission.admitted: turn → pages   // 影子账
//! engine ensure_for_len(table, target)              // 不看任何票据
//! ```
//!
//! 现在:
//!
//! ```text
//! Acquire::Capacity { reserve: Some(..) } → Grant
//! ensure_for_len(grant, table, target)    → 只消费 grant
//! Release                                 → 归还 grant
//! sched_policy::StoreFacts.reserved_pages → 本账的 total slots
//! ```
//!
//! 语义:
//! - `Grant.slots` = **准入面**批准总量(正常档 = 整条上下文;超卖档 = 工作集);
//! - `Grant.alloc_slots` = **分配面**可消费总量(超卖档 = 整条 ctx 块表上限);
//! - `remaining_slots` = 分配面里还没被 `ensure_for_len` 兑现的额度;
//! - `reserved_slots()` = Σ`Grant.slots`(总量),不是 remaining ——
//!   已兑现的槽已经变成活跃块,不能从“承诺面”里减掉;否则新 turn
//!   会把同一批容量再批一次。
//! - `consume` 只减 remaining;`refund` 只用于分配失败回滚;
//! - `release_turn` 幂等(无 grant = 返回空表)。

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{StoreError, StoreResult};
use crate::tier::Owner;

/// 预约身份(单调递增;turn 内可有多张票)。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ReservationId(pub u64);

impl ReservationId {
    /// 原始 id(报告/日志用)。
    pub fn get(self) -> u64 {
        self.0
    }
}

/// 票的用途。第一刀只发 `ActiveFloor`;其余变体留给后续的 load/snap/temp 接线。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReservationKind {
    /// 活跃 turn 的设备槽地板(准入时一次批准,`ensure_for_len` 逐页消费)。
    ActiveFloor,
    /// 前缀回灌(host → device)的临时槽。
    Load,
    /// GDN 行。
    Row,
    /// 状态快照。
    Snap,
    /// 临时工作区。
    Temp,
}

impl ReservationKind {
    /// 稳定标签(日志/报告)。
    pub fn label(self) -> &'static str {
        match self {
            Self::ActiveFloor => "active_floor",
            Self::Load => "load",
            Self::Row => "row",
            Self::Snap => "snap",
            Self::Temp => "temp",
        }
    }
}

/// 一张已批准的票。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grant {
    /// 票号。
    pub id: ReservationId,
    /// 归属 turn。
    pub turn: u64,
    /// 归属主体。
    pub owner: Owner,
    /// 用途。
    pub kind: ReservationKind,
    /// **准入面**批准总量(槽;超卖档 = 工作集，不是整条 ctx)。
    pub slots: usize,
    /// **准入面**批准总字节(`slots × slot_bytes`)。
    pub bytes: usize,
    /// **分配面**可消费总量(槽;超卖档 = 整条 ctx 的块表上限)。
    pub alloc_slots: usize,
    /// **分配面**可消费总字节。
    pub alloc_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    grant: Grant,
    remaining_slots: usize,
    remaining_bytes: usize,
}

/// 预约账(store 自持;`Rel` 的门面方法负责加锁/接线)。
#[derive(Debug, Default)]
pub struct ReservationLedger {
    next: u64,
    entries: BTreeMap<ReservationId, Entry>,
}

impl ReservationLedger {
    /// 空账。
    pub fn new() -> Self {
        Self::default()
    }

    /// 批准一张票。`slots = 0` = 调用方错配(响亮拒)。
    pub fn grant(
        &mut self,
        turn: u64,
        owner: Owner,
        kind: ReservationKind,
        slots: usize,
        bytes: usize,
        alloc_slots: usize,
        alloc_bytes: usize,
    ) -> StoreResult<Grant> {
        if slots == 0 || alloc_slots == 0 {
            return Err(StoreError::Budget(
                "ReservationLedger::grant: slots/alloc_slots = 0(空票无意义)".into(),
            ));
        }
        if alloc_slots < slots || alloc_bytes < bytes {
            return Err(StoreError::Budget(format!(
                "ReservationLedger::grant: 分配面 {alloc_slots} 槽/{alloc_bytes}B                  小于准入面 {slots} 槽/{bytes}B"
            )));
        }
        let id = ReservationId(self.next);
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| StoreError::Invariant("ReservationId 溢出".into()))?;
        let grant = Grant {
            id,
            turn,
            owner,
            kind,
            slots,
            bytes,
            alloc_slots,
            alloc_bytes,
        };
        self.entries.insert(
            id,
            Entry {
                grant,
                remaining_slots: alloc_slots,
                remaining_bytes: alloc_bytes,
            },
        );
        Ok(grant)
    }

    /// 消费 `slots` 个槽(只减 remaining;总量保留给准入面)。
    pub fn consume(&mut self, id: ReservationId, slots: usize, bytes: usize) -> StoreResult<()> {
        let e = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| StoreError::Unknown(format!("grant {id:?} 不在账")))?;
        if e.remaining_slots < slots || e.remaining_bytes < bytes {
            return Err(StoreError::Budget(format!(
                "grant {id:?} 额度不足:要 {slots} 槽/{bytes}B,剩 {} 槽/{}B",
                e.remaining_slots, e.remaining_bytes
            )));
        }
        e.remaining_slots -= slots;
        e.remaining_bytes -= bytes;
        Ok(())
    }

    /// 回滚一次 `consume`(分配失败时;不得回滚超过 remaining + 已消费)。
    pub fn refund(&mut self, id: ReservationId, slots: usize, bytes: usize) -> StoreResult<()> {
        let e = self
            .entries
            .get_mut(&id)
            .ok_or_else(|| StoreError::Unknown(format!("grant {id:?} 不在账")))?;
        let new_slots = e
            .remaining_slots
            .checked_add(slots)
            .ok_or_else(|| StoreError::Invariant("grant refund 槽溢出".into()))?;
        let new_bytes = e
            .remaining_bytes
            .checked_add(bytes)
            .ok_or_else(|| StoreError::Invariant("grant refund 字节溢出".into()))?;
        if new_slots > e.grant.alloc_slots || new_bytes > e.grant.alloc_bytes {
            return Err(StoreError::Invariant(format!(
                "grant {id:?} 回滚越界:remaining {new_slots}/{new_bytes} > total {}/{}",
                e.grant.alloc_slots, e.grant.alloc_bytes
            )));
        }
        e.remaining_slots = new_slots;
        e.remaining_bytes = new_bytes;
        Ok(())
    }

    /// 归还一张票。
    pub fn release(&mut self, id: ReservationId) -> StoreResult<Grant> {
        self.entries
            .remove(&id)
            .map(|e| e.grant)
            .ok_or_else(|| StoreError::Unknown(format!("grant {id:?} 不在账")))
    }

    /// 归还一个 turn 的全部票(幂等;释放顺序按 id)。
    pub fn release_turn(&mut self, turn: u64) -> Vec<Grant> {
        let ids: Vec<ReservationId> = self
            .entries
            .iter()
            .filter(|(_, e)| e.grant.turn == turn)
            .map(|(id, _)| *id)
            .collect();
        ids.into_iter()
            .filter_map(|id| self.entries.remove(&id).map(|e| e.grant))
            .collect()
    }

    /// 查票(诊断)。
    pub fn get(&self, id: ReservationId) -> Option<Grant> {
        self.entries.get(&id).map(|e| e.grant)
    }

    /// 剩余可消费槽。
    pub fn remaining_slots(&self, id: ReservationId) -> Option<usize> {
        self.entries.get(&id).map(|e| e.remaining_slots)
    }

    /// 剩余可消费字节。
    pub fn remaining_bytes(&self, id: ReservationId) -> Option<usize> {
        self.entries.get(&id).map(|e| e.remaining_bytes)
    }

    /// 准入面:所有票的**批准总量**之和(已消费的槽变成活跃块,不能减)。
    pub fn reserved_slots(&self) -> usize {
        self.entries.values().map(|e| e.grant.slots).sum()
    }

    /// 准入面:所有票的批准总字节。
    pub fn reserved_bytes(&self) -> usize {
        self.entries.values().map(|e| e.grant.bytes).sum()
    }

    /// 在册 turn 数(去重;槽位策略用)。
    pub fn in_flight_turns(&self) -> usize {
        self.entries
            .values()
            .map(|e| e.grant.turn)
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// 在册票数。
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 空账?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grant_consume_and_release_are_balanced() {
        let mut l = ReservationLedger::new();
        let g = l
            .grant(
                7,
                Owner::Turn(7),
                ReservationKind::ActiveFloor,
                10,
                10 << 20,
                10,
                10 << 20,
            )
            .expect("grant");
        assert_eq!(l.reserved_slots(), 10);
        assert_eq!(l.in_flight_turns(), 1);
        l.consume(g.id, 4, 4 << 20).expect("consume");
        assert_eq!(l.remaining_slots(g.id), Some(6));
        assert_eq!(l.reserved_slots(), 10, "总量不因消费而减少");
        l.release_turn(7);
        assert_eq!(l.reserved_slots(), 0);
        assert!(l.release_turn(7).is_empty(), "重复归还幂等");
    }

    #[test]
    fn over_consume_is_loud_and_does_not_mutate() {
        let mut l = ReservationLedger::new();
        let g = l
            .grant(1, Owner::Turn(1), ReservationKind::ActiveFloor, 2, 2, 2, 2)
            .expect("grant");
        let err = l.consume(g.id, 3, 3).expect_err("超额必须拒");
        assert!(matches!(err, StoreError::Budget(_)));
        assert_eq!(l.remaining_slots(g.id), Some(2), "失败不得部分扣账");
        l.consume(g.id, 2, 2).expect("刚好");
        assert_eq!(l.remaining_slots(g.id), Some(0));
    }

    #[test]
    fn refund_restores_only_up_to_total() {
        let mut l = ReservationLedger::new();
        let g = l
            .grant(1, Owner::Turn(1), ReservationKind::ActiveFloor, 2, 2, 2, 2)
            .expect("grant");
        l.consume(g.id, 1, 1).expect("consume");
        l.refund(g.id, 1, 1).expect("refund");
        let err = l.refund(g.id, 1, 1).expect_err("回滚越界必须拒");
        assert!(matches!(err, StoreError::Invariant(_)));
        assert_eq!(l.remaining_slots(g.id), Some(2));
    }

    #[test]
    fn multi_turn_count_is_deduplicated() {
        let mut l = ReservationLedger::new();
        l.grant(1, Owner::Turn(1), ReservationKind::ActiveFloor, 1, 1, 1, 1)
            .expect("a");
        l.grant(1, Owner::Turn(1), ReservationKind::ActiveFloor, 1, 1, 1, 1)
            .expect("b");
        l.grant(2, Owner::Turn(2), ReservationKind::ActiveFloor, 1, 1, 1, 1)
            .expect("c");
        assert_eq!(l.len(), 3);
        assert_eq!(l.in_flight_turns(), 2);
        assert_eq!(l.reserved_slots(), 3);
    }

    #[test]
    fn oversub_admission_and_allocation_faces_are_separate() {
        let mut l = ReservationLedger::new();
        let g = l
            .grant(
                1,
                Owner::Turn(1),
                ReservationKind::ActiveFloor,
                2,
                2 << 20,
                10,
                10 << 20,
            )
            .expect("grant");
        assert_eq!(l.reserved_slots(), 2, "准入面只看工作集");
        assert_eq!(l.remaining_slots(g.id), Some(10), "分配面可长到整条 ctx");
        l.consume(g.id, 8, 8 << 20).expect("消费 8 页");
        assert_eq!(l.reserved_slots(), 2, "消费不改变准入面");
        assert_eq!(l.remaining_slots(g.id), Some(2));
    }

    #[test]
    fn alloc_face_cannot_be_smaller_than_admission_face() {
        let mut l = ReservationLedger::new();
        assert!(matches!(
            l.grant(1, Owner::Turn(1), ReservationKind::ActiveFloor, 4, 4, 2, 2),
            Err(StoreError::Budget(_))
        ));
    }

    #[test]
    fn zero_slot_grant_is_rejected() {
        let mut l = ReservationLedger::new();
        assert!(matches!(
            l.grant(1, Owner::Turn(1), ReservationKind::ActiveFloor, 0, 0, 0, 0),
            Err(StoreError::Budget(_))
        ));
    }
}
