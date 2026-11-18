//! **跨腿共享的层预算**（host / disk）。
//!
//! 2026-11-18 用户裁决：`ram_pool_size` / `disk_pool_size` 是**一层一个总池**，
//! KV 与 GDN 快照共享，不再用 `snap_host_mb` / `snap_disk_mb` 预切分片。
//!
//! 形态律：
//! - **单一总量**：`total_bytes` 由配置给一次；两条腿的物理占用都记在同一本账；
//! - **显式拒绝**：`try_reserve` 越界返回 `false`，调用方转成 `Exhausted`；
//! - **可观测**：`used_bytes` / `available_bytes` 是报告与诊断的唯一真源；
//! - **不静默**：`release` 发现账目下溢时返回 `Invariant`（调用方必须上抛或
//!   在 Drop 路径记录 WARN，不许悄悄归零）。
//!
//! 本模块只做字节账，不认识 KV / 快照，也不做驱逐策略 —— 驱逐仍由 store 的
//! `plan_*` / `Rel` 统一出序，engine 执行 I/O。

use std::sync::{Arc, Mutex};

use crate::error::{StoreError, StoreResult};

/// 共享层预算句柄（host 或 disk 各一个）。
pub type SharedTierBudgetRef = Arc<Mutex<SharedTierBudget>>;

/// 一层（host / disk）的总预算与已用字节。
#[derive(Debug)]
pub struct SharedTierBudget {
    total_bytes: usize,
    used_bytes: usize,
}

impl SharedTierBudget {
    /// 新建共享预算；`total_bytes = 0` = 关层。
    pub fn new(total_bytes: usize) -> SharedTierBudgetRef {
        Arc::new(Mutex::new(Self {
            total_bytes,
            used_bytes: 0,
        }))
    }

    /// 本层总预算（字节；0 = 关层）。
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// 已用字节（两腿合计；含驻留、空转缓冲、在飞借台与已取出未归还的块）。
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// 剩余字节（饱和到 0；总账被 boot 回填暂时撑破时也不会下溢）。
    pub fn available_bytes(&self) -> usize {
        self.total_bytes.saturating_sub(self.used_bytes)
    }

    /// 尝试预留 `bytes`；越界 = `false`（**零副作用**）。
    pub fn try_reserve(&mut self, bytes: usize) -> bool {
        let Some(after) = self.used_bytes.checked_add(bytes) else {
            return false;
        };
        if after > self.total_bytes {
            return false;
        }
        self.used_bytes = after;
        true
    }

    /// **boot 回填专用**：允许暂时超过总预算（随后由带保护集的配额对账逐出）。
    ///
    /// 只给 `adopt_with(enforce_budget = false)` 用；运行期任何分配都不得走这里。
    pub fn reserve_unchecked(&mut self, bytes: usize) -> StoreResult<()> {
        self.used_bytes = self
            .used_bytes
            .checked_add(bytes)
            .ok_or_else(|| StoreError::Invariant("共享层预算 used 字节溢出".into()))?;
        Ok(())
    }

    /// 释放 `bytes`；账目下溢 = 不变量违反（响亮拒，不静默归零）。
    pub fn release(&mut self, bytes: usize) -> StoreResult<()> {
        self.used_bytes = self.used_bytes.checked_sub(bytes).ok_or_else(|| {
            StoreError::Invariant(format!(
                "共享层预算释放下溢:used={} < {bytes}",
                self.used_bytes
            ))
        })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_and_release_are_exact() {
        let b = SharedTierBudget::new(10);
        let mut g = b.lock().expect("锁");
        assert!(g.try_reserve(4));
        assert!(g.try_reserve(6));
        assert!(!g.try_reserve(1), "满额必须拒");
        assert_eq!(g.used_bytes(), 10);
        assert_eq!(g.available_bytes(), 0);
        g.release(6).expect("释放");
        assert_eq!(g.used_bytes(), 4);
    }

    #[test]
    fn release_underflow_is_loud() {
        let b = SharedTierBudget::new(10);
        let mut g = b.lock().expect("锁");
        assert!(g.release(1).is_err(), "无账可出必须拒");
    }
}
