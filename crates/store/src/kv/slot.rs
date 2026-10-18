//! 物理槽账(`SlotPool`)—— 设备块池的**分配 / 引用 / 回收**。
//!
//! 语义与 engine 现行 `BlockManager` 同族(S3 第一刀的硬门 = **现役行为等价**):
//!
//! - **分配 = 队首弹出;归还 = 压队尾** ⇒ FIFO 复用(队列序 = 未来驱逐序);
//! - `refs` = 持有者数:每条引用该块的会话链各持一份,**前缀树绑定再持一份**
//!   (缓存持有的那一份 —— 会话释放后块仍驻留,直到缓存把它逐掉);
//! - **归零 = 回自由队列**;`refs > 0` 的槽**不可能**被回收(池自肃:
//!   分配/释放只改自己的两个字段,不摸别人的变量)。
//!
//! 槽号是**可回收资源,不是身份**(`id.rs` 身份律)—— 本模块只发号与记账,
//! 谁占着这个号由调用方(前缀树绑定 / 会话块表)负责。

use std::collections::VecDeque;

use crate::error::{StoreError, StoreResult};

/// 物理槽池(设备块号的分配与引用账)。
#[derive(Debug)]
pub(crate) struct SlotPool {
    /// 每槽的引用数(0 = 自由;1 = 独占;≥2 = 共享/缓存持有)。
    refs: Vec<u32>,
    /// 自由槽队列(分配 = 队首;归还 = 队尾)。
    free: VecDeque<u32>,
}

impl SlotPool {
    /// 全池空闲起步。
    pub(crate) fn new(slots: usize) -> Self {
        Self { refs: vec![0; slots], free: (0..slots as u32).collect() }
    }

    /// 池容量(槽数)。
    pub(crate) fn capacity(&self) -> usize {
        self.refs.len()
    }

    /// 自由槽数。
    pub(crate) fn free(&self) -> usize {
        self.free.len()
    }

    /// 取一个槽(**引用置 1**;调用方即成为持有者)。池空 = 响亮拒。
    pub(crate) fn acquire(&mut self) -> StoreResult<u32> {
        let slot = self
            .free
            .pop_front()
            .ok_or_else(|| StoreError::Exhausted(format!("槽池耗尽(共 {} 槽,无自由槽)", self.refs.len())))?;
        self.refs[slot as usize] = 1;
        Ok(slot)
    }

    /// 引用加一(共享/缓存持有)。
    pub(crate) fn incref(&mut self, slot: u32) -> StoreResult<()> {
        let n = self.refs.len();
        let r = self
            .refs
            .get_mut(slot as usize)
            .ok_or_else(|| StoreError::Invariant(format!("槽 {slot} 越界(共 {n})")))?;
        if *r == 0 {
            // 自由槽被引用 = 账目已坏(持有者应经 acquire);不许静默吞掉
            return Err(StoreError::Invariant(format!("槽 {slot} 是自由槽,不能加引用")));
        }
        *r = r.saturating_add(1);
        Ok(())
    }

    /// 引用减一;归零 = 回自由队列(`true` = 本槽已回收)。
    pub(crate) fn decref(&mut self, slot: u32) -> StoreResult<bool> {
        let n = self.refs.len();
        let r = self
            .refs
            .get_mut(slot as usize)
            .ok_or_else(|| StoreError::Invariant(format!("槽 {slot} 越界(共 {n})")))?;
        if *r == 0 {
            return Err(StoreError::Invariant(format!("槽 {slot} 引用已为 0,不能再减")));
        }
        *r -= 1;
        let freed = *r == 0;
        if freed {
            self.free.push_back(slot);
        }
        Ok(freed)
    }

    /// 当前引用数。
    pub(crate) fn refs(&self, slot: u32) -> u32 {
        self.refs.get(slot as usize).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acquire_release_is_fifo_and_refs_are_counted() {
        let mut p = SlotPool::new(2);
        assert_eq!(p.free(), 2);
        let a = p.acquire().expect("槽 A");
        let b = p.acquire().expect("槽 B");
        assert_eq!((a, b), (0, 1), "分配从队首");
        assert!(matches!(p.acquire(), Err(StoreError::Exhausted(_))), "池空响亮拒");

        // 共享:缓存 + 会话各持一份
        p.incref(a).expect("共享加引用");
        assert_eq!(p.refs(a), 2);
        assert!(!p.decref(a).expect("会话释放"), "refs 2 → 1,不回池");
        assert!(p.decref(a).expect("缓存释放"), "refs 1 → 0,回池");
        assert_eq!(p.free(), 1);
        assert_eq!(p.acquire().expect("复用"), a, "归还压队尾 ⇒ 复用序 FIFO");
        let _ = p.decref(a).expect("收拾");
        let _ = p.decref(b).expect("收拾");
    }

    #[test]
    fn free_slot_cannot_be_increfed_and_zero_cannot_be_decrefed() {
        let mut p = SlotPool::new(1);
        assert!(matches!(p.incref(0), Err(StoreError::Invariant(_))), "自由槽不许加引用");
        assert!(matches!(p.decref(0), Err(StoreError::Invariant(_))), "0 引用不许再减");
        assert!(matches!(p.incref(9), Err(StoreError::Invariant(_))), "越界响亮拒");
    }
}
