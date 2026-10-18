//! 三层账:层枚举 / 层预算 / 层占用账(设计卷 §3.5)。
//!
//! **账目律**(A5.2 的状态面):任何"进了多少字节"都必须能对上"出去多少字节";
//! host/盘两层有**显式上限**(`TierBudget`)⇒ 越界 = 响亮拒(不许悄悄超发)。
//! GPU 层的总量约束不在这里 —— 那是 `budget.rs`(卡级预算分解)+ 保留额不变量的事。

use std::collections::BTreeMap;

use crate::error::{StoreError, StoreResult};

/// 存储层(设计卷 §3.1:GPU 池 → host pinned → 磁盘)。
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Tier {
    /// 设备显存(常驻池;保留面 + 弹性面都在这层)。
    Gpu,
    /// 宿主 pinned 内存(可配上限;0 = 关层)。
    Host,
    /// 磁盘(二期;0 = 关层)。
    Disk,
}

/// 资产类别 —— **保留面与可驱逐面必须分开记账**(设计卷 §2.4 保留额不变量)。
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum AssetKind {
    /// attention 历史块(**可驱逐**缓存面)。
    KvBlock,
    /// 活跃 turn 的直接显存地板(`dense_floor`;保留面,**不可驱逐**)。
    Floor,
    /// 活跃 turn 的弹性余量(地板之上;可驱逐)。
    Elastic,
    /// GDN 状态快照(可驱逐;可下沉 host/盘)。
    Snap,
    /// 活跃 turn 占用的 GDN 行(不可驱逐;行数受 `gdn_slots` 约束)。
    Row,
    /// 会话元数据(索引/账目;µs 级,不参与驱逐)。
    Meta,
    /// **选择层检索索引**(B 轴:Mean-K;host 驻留,可驱逐/可下沉 —— 论文同款)。
    Index,
}

impl AssetKind {
    /// 本类别是否属于**保留面**(驱逐面之外)。
    pub fn is_reserved(self) -> bool {
        matches!(self, AssetKind::Floor | AssetKind::Row | AssetKind::Meta)
    }
}

/// 各层上限(字节;GPU 层不用此值 —— 卡级约束在 `budget.rs`)。
#[derive(Clone, Copy, Debug, Default)]
pub struct TierBudget {
    /// host pinned 上限(0 = 关层;要求 C4「预算显式配置」)。
    pub host_bytes: usize,
    /// 磁盘配额(0 = 关层)。
    pub disk_bytes: usize,
}

impl TierBudget {
    /// 本层上限(字节;`None` = 本层不受此账约束,即 GPU)。
    pub fn cap(&self, tier: Tier) -> Option<usize> {
        match tier {
            Tier::Gpu => None,
            Tier::Host => Some(self.host_bytes),
            Tier::Disk => Some(self.disk_bytes),
        }
    }
}

/// 单层单类别的账目。
#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    bytes: usize,
    entries: usize,
}

/// 三层占用账(可 dump、可断言)。
#[derive(Debug, Default)]
pub struct TierLedger {
    budget: TierBudget,
    cells: BTreeMap<(Tier, AssetKind), Cell>,
}

impl TierLedger {
    /// 构造(唯一入口:预算随账走 —— 不设全局态)。
    pub fn new(budget: TierBudget) -> Self {
        Self { budget, cells: BTreeMap::new() }
    }

    /// 预算(供报告/断言)。
    pub fn budget(&self) -> TierBudget {
        self.budget
    }

    /// 入账。**越界 = 响亮拒**(host/盘层);GPU 层不受此约束。
    pub fn add(&mut self, tier: Tier, kind: AssetKind, bytes: usize) -> StoreResult<()> {
        if let Some(cap) = self.budget.cap(tier) {
            let after = self.used(tier).saturating_add(bytes);
            if after > cap {
                return Err(StoreError::Exhausted(format!(
                    "{tier:?} 层越界:{after} > {cap} 字节(增加 {bytes})"
                )));
            }
        }
        let cell = self.cells.entry((tier, kind)).or_default();
        cell.bytes += bytes;
        cell.entries += 1;
        Ok(())
    }

    /// 出账(必须与入账成对;不平衡 = 不变量违反,响亮拒)。
    pub fn remove(&mut self, tier: Tier, kind: AssetKind, bytes: usize) -> StoreResult<()> {
        let cell = self
            .cells
            .get_mut(&(tier, kind))
            .ok_or_else(|| StoreError::Invariant(format!("{tier:?}/{kind:?} 无账可出")))?;
        if cell.bytes < bytes || cell.entries == 0 {
            return Err(StoreError::Invariant(format!(
                "{tier:?}/{kind:?} 出账超账:{} < {bytes}",
                cell.bytes
            )));
        }
        cell.bytes -= bytes;
        cell.entries -= 1;
        if cell.bytes == 0 && cell.entries == 0 {
            self.cells.remove(&(tier, kind));
        }
        Ok(())
    }

    /// 某层总占用。
    pub fn used(&self, tier: Tier) -> usize {
        self.cells
            .iter()
            .filter(|((t, _), _)| *t == tier)
            .map(|(_, c)| c.bytes)
            .sum()
    }

    /// 某层某类占用。
    pub fn kind_bytes(&self, tier: Tier, kind: AssetKind) -> usize {
        self.cells.get(&(tier, kind)).map_or(0, |c| c.bytes)
    }

    /// 某层条目数。
    pub fn entries(&self, tier: Tier) -> usize {
        self.cells
            .iter()
            .filter(|((t, _), _)| *t == tier)
            .map(|(_, c)| c.entries)
            .sum()
    }

    /// 报告(设计卷 §2.4 报告面)。
    pub fn report(&self) -> TierReport {
        let mut rows = Vec::new();
        for ((tier, kind), cell) in &self.cells {
            rows.push(TierRow { tier: *tier, kind: *kind, bytes: cell.bytes, entries: cell.entries });
        }
        TierReport {
            rows,
            gpu_used: self.used(Tier::Gpu),
            host_used: self.used(Tier::Host),
            disk_used: self.used(Tier::Disk),
            host_cap: self.budget.host_bytes,
            disk_cap: self.budget.disk_bytes,
        }
    }
}

/// 报告的一行。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierRow {
    /// 层。
    pub tier: Tier,
    /// 类别。
    pub kind: AssetKind,
    /// 占用字节。
    pub bytes: usize,
    /// 条目数。
    pub entries: usize,
}

/// 三层报告。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TierReport {
    /// 明细行。
    pub rows: Vec<TierRow>,
    /// GPU 总占用。
    pub gpu_used: usize,
    /// host 总占用。
    pub host_used: usize,
    /// 盘总占用。
    pub disk_used: usize,
    /// host 上限(0 = 关层)。
    pub host_cap: usize,
    /// 盘上限(0 = 关层)。
    pub disk_cap: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger(host: usize) -> TierLedger {
        TierLedger::new(TierBudget { host_bytes: host, disk_bytes: 0 })
    }

    #[test]
    fn add_remove_keeps_books_balanced() {
        let mut l = ledger(1 << 20);
        l.add(Tier::Gpu, AssetKind::Floor, 4096).expect("入账");
        l.add(Tier::Gpu, AssetKind::KvBlock, 8192).expect("入账");
        assert_eq!(l.used(Tier::Gpu), 4096 + 8192);
        assert_eq!(l.kind_bytes(Tier::Gpu, AssetKind::Floor), 4096);
        l.remove(Tier::Gpu, AssetKind::Floor, 4096).expect("出账");
        assert_eq!(l.kind_bytes(Tier::Gpu, AssetKind::Floor), 0);
        // 出到负 = 不变量违反(响亮)
        assert!(l.remove(Tier::Gpu, AssetKind::Floor, 1).is_err());
    }

    #[test]
    fn host_cap_is_enforced_loudly() {
        let mut l = ledger(1024);
        l.add(Tier::Host, AssetKind::KvBlock, 1024).expect("占满");
        let err = l.add(Tier::Host, AssetKind::Snap, 1).expect_err("越界必须拒");
        assert!(matches!(err, StoreError::Exhausted(_)), "得 {err:?}");
    }

    #[test]
    fn gpu_tier_is_not_capped_here() {
        // GPU 总量约束归 budget.rs(卡级预算分解),账本层不设上限
        let mut l = TierLedger::new(TierBudget::default());
        l.add(Tier::Gpu, AssetKind::KvBlock, usize::MAX / 2).expect("GPU 无账本上限");
    }
}
