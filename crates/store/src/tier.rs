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
    /// **临时工作区**(chunk 激活面 / plan sync 缓冲 / eager 回退 / logits 中转;
    /// 2026-10-19 DeviceHandle 事件账引入 —— 生命周期挂 turn,N4 归还)。
    Temp,
    /// **权重与装载产物**(常驻;装载相位 —— 2026-10-19 用户裁决扩词表)。
    Weight,
    /// **图 slab 与图内自产块**(随图存活;捕获窗)。
    Graph,
    /// **池骨架**(KV/GDN/快照/草稿池的容器块;boot 期)。
    Pool,
    /// **未分类兜底**(尚未接线相位的分配;**可见** —— 字节数是迁移进度指标)。
    Unclassified,
}

impl From<owl_iface::handle::AssetClass> for AssetKind {
    /// iface 的中性类别 → store 的资产类别(单点映射;新增类别在此补一行)。
    fn from(c: owl_iface::handle::AssetClass) -> Self {
        use owl_iface::handle::AssetClass as A;
        match c {
            A::KvBlock => AssetKind::KvBlock,
            A::Floor => AssetKind::Floor,
            A::Elastic => AssetKind::Elastic,
            A::Snap => AssetKind::Snap,
            A::Row => AssetKind::Row,
            A::Meta => AssetKind::Meta,
            A::Index => AssetKind::Index,
            A::Temp => AssetKind::Temp,
            A::Weight => AssetKind::Weight,
            A::Graph => AssetKind::Graph,
            A::Pool => AssetKind::Pool,
            A::Unclassified => AssetKind::Unclassified,
        }
    }
}

impl AssetKind {
    /// 本类别是否属于**保留面**(驱逐面之外)。
    ///
    /// 常驻系统面(Weight/Graph/Pool)也是保留面 —— 它们不参与 store 的
    /// 驱逐/下沉决策(不是池的可回收块)。`Unclassified` **不声称**保留
    /// (分类未知就不许替决策面下结论;当前决策面根本不读本函数)。
    /// **可池化?**(缺口 4「临时池」的门;2026-10-19)——**决策产地 = store**:
    /// 只有临时面(chunk 激活/工作区)进复用池;常驻面(权重/图/池骨架/地板/行)
    /// 与缓存面(KV/快照/索引)**不许**被当作可复用临时块发放(它们的生命周期
    /// 各自有主)。
    pub fn poolable(self) -> bool {
        matches!(self, AssetKind::Temp)
    }

    pub fn is_reserved(self) -> bool {
        matches!(
            self,
            AssetKind::Floor
                | AssetKind::Row
                | AssetKind::Meta
                | AssetKind::Weight
                | AssetKind::Graph
                | AssetKind::Pool
        )
    }
}

/// 各层上限(字节;GPU 层不用此值 —— 卡级约束在 `budget.rs`)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

/// **中层预算的唯一算式**(2026-10-19):boot 造账(`Engine::on`)与准入预算入参
/// (`BudgetInputs.host_bytes`)必须**同源** —— 两处各算一遍必然漂,而漂了之后
/// `Rel::open` 的交叉校验会把启动拒掉(好过静默两把尺)。
///
/// 口径:`host_cache_mb` = KV 块腿的宿主分片,`snap_host_mb` = 状态腿(快照)分片;
/// 两者之和 = store 的中层总账。磁盘层二期(恒 0)。
pub fn tier_budget_of(
    host_cache_mb: usize,
    snap_host_mb: usize,
    disk_cache_mb: usize,
    snap_disk_mb: usize,
) -> TierBudget {
    TierBudget {
        host_bytes: (host_cache_mb + snap_host_mb) << 20,
        disk_bytes: (disk_cache_mb + snap_disk_mb) << 20,
    }
}

/// 单层单类别的账目。
#[derive(Clone, Copy, Debug, Default)]
struct Cell {
    bytes: usize,
    entries: usize,
}

/// **归属**(owner 维)—— **单一真源 = `owl-iface`**(契约面;store 只复用不另立,
/// 避免"两套接口"):`Turn(u64)` / `Session(u64)` / `Cache` / `System`。
pub use owl_iface::handle::Owner;

/// owner 维的一行(报告/端点原料)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OwnerRow {
    /// 归属主体。
    pub owner: Owner,
    /// 所在层。
    pub tier: Tier,
    /// 资产类别。
    pub kind: AssetKind,
    /// 占用字节。
    pub bytes: usize,
    /// 条目数。
    pub entries: usize,
}

/// **事件账的一行**(rank × owner × tier × kind 聚合;`/debug/vram` 的事件面)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EventRow {
    /// 设备序(TP 分卡;单卡恒 0)。
    pub rank: u8,
    /// 归属主体。
    pub owner: Owner,
    /// 所在层(换层时跟着变)。
    pub tier: Tier,
    /// 资产类别。
    pub kind: AssetKind,
    /// 占用字节。
    pub bytes: usize,
    /// 条目数。
    pub entries: usize,
}

/// 三层占用账(可 dump、可断言)。
#[derive(Debug, Default)]
pub struct TierLedger {
    budget: TierBudget,
    cells: BTreeMap<(Tier, AssetKind), Cell>,
    /// owner 维明细(第 0 刀;与 `cells` **同源同步** —— 入账/出账两处一起改)。
    owners: BTreeMap<(Owner, Tier, AssetKind), Cell>,
}

impl TierLedger {
    /// 构造(唯一入口:预算随账走 —— 不设全局态)。
    pub fn new(budget: TierBudget) -> Self {
        Self { budget, cells: BTreeMap::new(), owners: BTreeMap::new() }
    }

    /// 预算(供报告/断言)。
    pub fn budget(&self) -> TierBudget {
        self.budget
    }

    /// 入账(存量入口:归到 [`Owner::System`],随接线逐步排空)。
    /// **越界 = 响亮拒**(host/盘层);GPU 层不受此约束。
    pub fn add(&mut self, tier: Tier, kind: AssetKind, bytes: usize) -> StoreResult<()> {
        self.add_owned(Owner::System, tier, kind, bytes)
    }

    /// 出账(存量入口;与 [`Self::add`] 成对)。
    pub fn remove(&mut self, tier: Tier, kind: AssetKind, bytes: usize) -> StoreResult<()> {
        self.remove_owned(Owner::System, tier, kind, bytes)
    }

    /// **入账(带归属)**;越界 = 响亮拒(host/盘层);GPU 层不受此约束。
    pub fn add_owned(
        &mut self,
        owner: Owner,
        tier: Tier,
        kind: AssetKind,
        bytes: usize,
    ) -> StoreResult<()> {
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
        let ocell = self.owners.entry((owner, tier, kind)).or_default();
        ocell.bytes += bytes;
        ocell.entries += 1;
        Ok(())
    }

    /// **出账(带归属)**;必须与入账成对(不平衡 = 不变量违反,响亮拒)。
    /// 超发**逐 owner 判**:归到别的 owner 的字节不会替它平账。
    pub fn remove_owned(
        &mut self,
        owner: Owner,
        tier: Tier,
        kind: AssetKind,
        bytes: usize,
    ) -> StoreResult<()> {
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
        let ocell = self.owners.get_mut(&(owner, tier, kind)).ok_or_else(|| {
            StoreError::Invariant(format!(
                "{} 在 {tier:?}/{kind:?} 无账可出(归属不匹配 = 不变量违反)",
                owner.label()
            ))
        })?;
        if ocell.bytes < bytes || ocell.entries == 0 {
            return Err(StoreError::Invariant(format!(
                "{} 在 {tier:?}/{kind:?} 出账超账:{} < {bytes}",
                owner.label(),
                ocell.bytes
            )));
        }
        cell.bytes -= bytes;
        cell.entries -= 1;
        ocell.bytes -= bytes;
        ocell.entries -= 1;
        if cell.bytes == 0 && cell.entries == 0 {
            self.cells.remove(&(tier, kind));
        }
        if ocell.bytes == 0 && ocell.entries == 0 {
            self.owners.remove(&(owner, tier, kind));
        }
        Ok(())
    }

    /// **换层**(下沉/回加载;2026-10-19):只改 tier,**owner/kind/entries 不变** ——
    /// 账本律"下沉 ≠ 归还"的落点。源层无账 ⇒ 响亮拒(不静默造账)。
    pub fn move_owned(
        &mut self,
        owner: Owner,
        kind: AssetKind,
        from: Tier,
        to: Tier,
        bytes: usize,
    ) -> StoreResult<()> {
        if from == to {
            return Err(StoreError::Invariant(format!(
                "换层源=目的({from:?});调用方算错相位"
            )));
        }
        let src = self.owners.get_mut(&(owner, from, kind)).ok_or_else(|| {
            StoreError::Invariant(format!(
                "{} 在 {from:?}/{kind:?} 无账可换(幽灵换层)",
                owner.label()
            ))
        })?;
        if src.bytes < bytes || src.entries == 0 {
            return Err(StoreError::Invariant(format!(
                "{} 换层超账:{from:?}/{kind:?} {} < {bytes}",
                owner.label(),
                src.bytes
            )));
        }
        src.bytes -= bytes;
        let src_entries = src.entries;
        src.entries -= 1;
        if src.bytes == 0 && src.entries == 0 {
            self.owners.remove(&(owner, from, kind));
        }
        // 分类账同步(entries 也搬 —— 一条资产换层,不该多出一条)
        if let Some(c) = self.cells.get_mut(&(from, kind)) {
            c.bytes = c.bytes.saturating_sub(bytes);
            c.entries = c.entries.saturating_sub(1);
            if c.bytes == 0 && c.entries == 0 {
                self.cells.remove(&(from, kind));
            }
        }
        let dst = self.owners.entry((owner, to, kind)).or_default();
        dst.bytes += bytes;
        dst.entries += src_entries;
        let c = self.cells.entry((to, kind)).or_default();
        c.bytes += bytes;
        c.entries += src_entries;
        Ok(())
    }

    /// **改归属**(tier/kind 不变,只换 owner;缺口 4 池清点用)。
    ///
    /// 判据:源 owner 必须在场且账够(不足 = 响亮拒);`cells`(tier,kind)不变
    /// —— 同一层的同一类资产只是换了主。
    pub fn reown(
        &mut self,
        from: Owner,
        to: Owner,
        tier: Tier,
        kind: AssetKind,
        bytes: usize,
    ) -> StoreResult<()> {
        if from == to {
            return Ok(()); // 幂等
        }
        let src = self.owners.get_mut(&(from, tier, kind)).ok_or_else(|| {
            StoreError::Invariant(format!(
                "{} 在 {tier:?}/{kind:?} 无账可改归属(幽灵)",
                from.label()
            ))
        })?;
        if src.bytes < bytes || src.entries == 0 {
            return Err(StoreError::Invariant(format!(
                "{} 改归属超账:{tier:?}/{kind:?} {} < {bytes}",
                from.label(),
                src.bytes
            )));
        }
        src.bytes -= bytes;
        src.entries -= 1;
        if src.bytes == 0 && src.entries == 0 {
            self.owners.remove(&(from, tier, kind));
        }
        let dst = self.owners.entry((to, tier, kind)).or_default();
        dst.bytes += bytes;
        dst.entries += 1;
        Ok(())
    }

    /// **owner 维明细**(字节降序;`/debug/vram` 的原料)。
    /// 判据:同一 owner 的字节逐轮**单调爬坡** = 泄漏;持平 = 正常。
    pub fn detail(&self) -> Vec<OwnerRow> {
        let mut rows: Vec<OwnerRow> = self
            .owners
            .iter()
            .map(|((owner, tier, kind), c)| OwnerRow {
                owner: *owner,
                tier: *tier,
                kind: *kind,
                bytes: c.bytes,
                entries: c.entries,
            })
            .collect();
        rows.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.owner.cmp(&b.owner)));
        rows
    }

    /// 某 owner 的总占用(逐 owner 判据的便捷入口)。
    pub fn owner_bytes(&self, owner: Owner) -> usize {
        self.owners
            .iter()
            .filter(|((o, _, _), _)| *o == owner)
            .map(|(_, c)| c.bytes)
            .sum()
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

    /// 报告(设计卷 §2.4 报告面;第 0 刀起带 **owner 维**)。
    pub fn report(&self) -> TierReport {
        let mut rows = Vec::new();
        for ((tier, kind), cell) in &self.cells {
            rows.push(TierRow { tier: *tier, kind: *kind, bytes: cell.bytes, entries: cell.entries });
        }
        TierReport {
            rows,
            owners: self.detail(),
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
    /// **owner 维明细**(字节降序;第 0 刀 —— 泄漏判据面)。
    pub owners: Vec<OwnerRow>,
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

    /// **步 2 门 ①**:事件账与 store 账**同一本**(alloc→free 账平 + 登记清空)
    #[test]
    fn ledger_book_alloc_free_balances_same_book() {
        use owl_iface::handle::{AllocEvent, AllocLedger, AssetClass, FreeEvent, Owner as IfOwner};
        let cell = shared_ledger(TierBudget { host_bytes: 1 << 20, disk_bytes: 0 });
        let mut h = LedgerEntry { book: cell.clone() };
        h.on_alloc(&AllocEvent {
            id: 1,
            bytes: 4096,
            name: "kv.workspace".into(),
            class: AssetClass::Temp,
            owner: IfOwner::Turn(7),
            rank: 0,
        });
        {
            let b = book_lock(&cell);
            assert_eq!(b.open_count(), 1);
            assert_eq!(b.open_bytes(IfOwner::Turn(7)), 4096);
            assert_eq!(b.open_total_bytes(), 4096);
            // store 自己的账立刻看到(同一本)
            assert_eq!(b.tiers().kind_bytes(Tier::Gpu, AssetKind::Temp), 4096);
            assert_eq!(b.tiers().owner_bytes(IfOwner::Turn(7)), 4096);
            assert_eq!(b.open_rows().len(), 1, "出借明细(owner×kind)");
        }
        h.on_free(&FreeEvent { id: 1, bytes: 0 }); // 字节按登记扣
        let b = book_lock(&cell);
        assert_eq!(b.open_count(), 0);
        assert_eq!(b.tiers().kind_bytes(Tier::Gpu, AssetKind::Temp), 0);
        assert_eq!(b.open_total_bytes(), 0);
        assert_eq!(b.hook_fail(), 0);
    }

    /// **步 2 门 ②**:双持 / 幽灵归还**响亮拒**(计数,不 panic、不静默)
    #[test]
    fn ledger_book_rejects_double_alloc_and_ghost_free() {
        use owl_iface::handle::{AllocEvent, AllocLedger, AssetClass, FreeEvent, Owner as IfOwner};
        let cell = shared_ledger(TierBudget { host_bytes: 1 << 20, disk_bytes: 0 });
        let mut h = LedgerEntry { book: cell.clone() };
        let ev = AllocEvent {
            id: 9,
            bytes: 128,
            name: "dup".into(),
            class: AssetClass::Elastic,
            owner: IfOwner::Session(1),
            rank: 0,
        };
        h.on_alloc(&ev);
        h.on_alloc(&ev); // 双持
        {
            let b = book_lock(&cell);
            assert_eq!(b.hook_fail(), 1, "双持必须计数");
            assert_eq!(b.tiers().kind_bytes(Tier::Gpu, AssetKind::Elastic), 128, "只入账一次");
        }
        h.on_free(&FreeEvent { id: 999, bytes: 0 }); // 幽灵
        let b = book_lock(&cell);
        assert_eq!(b.hook_fail(), 2, "幽灵归还必须计数");
        assert_eq!(b.open_count(), 1, "合法那条仍在场");
    }

    /// **步 2 门 ③**(2026-10-19 修订):换层 = **不注销**(owner/entries 不变;
    /// tier 迁移);片级换层(片和 ≠ 登记)= **跳过 + 单独计数**(口径不支持,
    /// 不是 `hook_fail` —— 真机 4352 条噪声定谳)。
    #[test]
    fn ledger_book_move_keeps_owner_and_skips_piecewise() {
        use owl_iface::handle::{
            AllocEvent, AllocLedger, AssetClass, MoveDir, MoveEvent, Owner as IfOwner,
        };
        let cell = shared_ledger(TierBudget { host_bytes: 1 << 20, disk_bytes: 0 });
        let mut h = LedgerEntry { book: cell.clone() };
        h.on_alloc(&AllocEvent {
            id: 5,
            bytes: 8192,
            name: "snap.1".into(),
            class: AssetClass::Snap,
            owner: IfOwner::Cache,
            rank: 0,
        });
        h.on_move(&MoveEvent { id: 5, bytes: 8192, dir: MoveDir::Out });
        {
            let b = book_lock(&cell);
            assert_eq!(b.tiers().kind_bytes(Tier::Gpu, AssetKind::Snap), 0, "设备层已清");
            assert_eq!(b.tiers().kind_bytes(Tier::Host, AssetKind::Snap), 8192, "宿主层在位");
            assert_eq!(b.tiers().owner_bytes(IfOwner::Cache), 8192, "owner 不变(下沉≠归还)");
            let rows = b.tiers().detail();
            assert_eq!(rows.len(), 1, "换层不多出一条:{rows:?}");
            assert_eq!(rows[0].entries, 1, "entries 跟着走");
            assert_eq!(b.open_rows()[0].tier, Tier::Host, "登记跟着换层");
            assert_eq!(b.hook_fail(), 0);
        }
        // 方向不符(Host 上再 Out)= 响亮拒
        h.on_move(&MoveEvent { id: 5, bytes: 8192, dir: MoveDir::Out });
        assert_eq!(book_lock(&cell).hook_fail(), 1, "方向不符必须计数");
        // **片级换层**(片和 ≠ 登记整块)= 口径不支持 ⇒ **跳过 + 单独计数**
        // (不是记账失败:`hook_fail` 必须恒 0 —— 2026-10-19 真机 4352 条噪声的定谳)
        h.on_move(&MoveEvent { id: 5, bytes: 4096, dir: MoveDir::In });
        let b = book_lock(&cell);
        assert_eq!(b.hook_fail(), 1, "部分换层不得计为记账失败");
        assert_eq!(b.partial_moves(), 1, "片级换层单独计数");
        assert_eq!(b.tiers().kind_bytes(Tier::Host, AssetKind::Snap), 8192, "账未被部分搬运搅动");
    }

    /// **步 3 门 ⑥**:词表映射**全单射**(12 类 iface ↔ 12 类 store;无遗漏)
    #[test]
    fn asset_class_maps_onto_asset_kind_injectively() {
        use owl_iface::handle::AssetClass as A;
        let all = [
            A::KvBlock,
            A::Floor,
            A::Elastic,
            A::Snap,
            A::Row,
            A::Meta,
            A::Index,
            A::Temp,
            A::Weight,
            A::Graph,
            A::Pool,
            A::Unclassified,
        ];
        let kinds: Vec<AssetKind> = all.iter().map(|c| AssetKind::from(*c)).collect();
        let mut uniq = kinds.clone();
        uniq.sort_by_key(|k| format!("{k:?}"));
        uniq.dedup();
        assert_eq!(uniq.len(), all.len(), "映射必须单射(两类别落同一 AssetKind = 记账混淆):{kinds:?}");
        assert!(AssetKind::from(A::Weight).is_reserved(), "权重是常驻面");
        assert!(AssetKind::from(A::Pool).is_reserved(), "池骨架是常驻面");
        assert!(!AssetKind::from(A::Unclassified).is_reserved(), "未分类不声称保留");
    }

    /// **缺口 4 门:临时池三条律**
    #[test]
    fn temp_pool_reuses_released_blocks_and_sweeps_owner() {
        use owl_iface::handle::{AllocEvent, FreeEvent, Owner as IfOwner};
        let cell = shared_ledger(TierBudget { host_bytes: 1 << 20, disk_bytes: 0 });
        let mut b = LedgerBook::new(TierBudget { host_bytes: 1 << 20, disk_bytes: 0 });
        // ① 入账(temp 类)+ 释放 ⇒ 入池(不回驱动),仍在借
        b.on_alloc(&AllocEvent {
            id: 11,
            bytes: 4096,
            name: "chunk.ws".into(),
            class: owl_iface::handle::AssetClass::Temp,
            owner: IfOwner::Turn(1),
            rank: 0,
        });
        assert!(b.pool_release(11).is_empty(), "空 = 已入池");
        assert_eq!(b.open_count(), 1, "**入池 ≠ 归还**:块仍在借");
        assert_eq!(b.pool_stats(), (1, 4096));
        // ② 同类同尺寸命中;异类/异尺寸不命中(尺寸错 = 双错)
        assert_eq!(b.pool_acquire(0, AssetKind::Temp, 4096), Some(11));
        assert_eq!(b.pool_acquire(0, AssetKind::Temp, 8192), None);
        assert_eq!(b.pool_acquire(0, AssetKind::KvBlock, 4096), None);
        let (hits, misses, _) = b.pool_counters();
        assert_eq!((hits, misses), (1, 2), "命中/未中计数");
        // **P0-1**:rank 不匹配不命中(TP 两 rank 共写一本账 ⇒ 键必须带 rank)
        assert!(b.pool_release(11).is_empty());
        assert_eq!(b.pool_acquire(1, AssetKind::Temp, 4096), None, "rank1 不许拿 rank0 的块");
        assert_eq!(b.pool_acquire(0, AssetKind::Temp, 4096), Some(11), "本 rank 可拿");
        // **P0-2**:同一块二次入池 ⇒ 响亮拒(不 push 第二份),且不交回调用方去 free
        assert!(b.pool_release(11).is_empty(), "首次入池");
        let fails_before = b.hook_fail();
        assert!(b.pool_release(11).is_empty(), "二次入池 ⇒ 拒(不产生第二份)");
        assert_eq!(b.hook_fail(), fails_before + 1, "双次入池必须计数");
        assert_eq!(b.pool_stats(), (1, 4096), "池里仍只有 1 份");
        let _ = b.pool_acquire(0, AssetKind::Temp, 4096);
        // ③ 不可池化类:拒绝(bool = false ⇒ 调用方必须真归还)
        b.on_alloc(&AllocEvent {
            id: 12,
            bytes: 1024,
            name: "weight".into(),
            class: owl_iface::handle::AssetClass::Weight,
            owner: IfOwner::Turn(1),
            rank: 0,
        });
        assert_eq!(b.pool_release(12), vec![12], "常驻面不许进池 ⇒ 同 id 真归还");
        // ④ 清点:池内 Turn(1) → Cache(改归属,不变 tier/kind);非池块**不动**
        assert!(b.pool_release(11).is_empty(), "重新入池(供清点)");
        let swept = b.pool_sweep(IfOwner::Turn(1), IfOwner::Cache);
        assert_eq!(swept, (1, 4096), "只清点池内那块");
        assert_eq!(b.open_bytes(IfOwner::Turn(1)), 1024, "非池块仍归 turn(泄漏可见)");
        assert_eq!(b.open_bytes(IfOwner::Cache), 4096);
        assert_eq!(b.tiers().kind_bytes(Tier::Gpu, AssetKind::Temp), 4096, "temp 面仍平");
        assert_eq!(b.tiers().kind_bytes(Tier::Gpu, AssetKind::Weight), 1024, "常驻面另账");
        let _ = cell;
    }

    /// **池上限门**(2026-10-19 实测:27B 一轮后池囤到 669 MiB ⇒ 必须有界):
    /// 超上限时 `pool_release` 顶出**最旧**块并把它的 id 交回(调用方真还驱动)。
    #[test]
    fn pool_cap_evicts_oldest_and_hands_back_id() {
        use owl_iface::handle::{AllocEvent, Owner as IfOwner};
        let mut b = LedgerBook::new(TierBudget { host_bytes: 1 << 20, disk_bytes: 0 });
        b.set_pool_cap(4096); // 只装 1 块
        for (id, name) in [(1u64, "a"), (2, "b")] {
            b.on_alloc(&AllocEvent {
                id,
                bytes: 4096,
                name: name.into(),
                class: owl_iface::handle::AssetClass::Temp,
                owner: IfOwner::Turn(1),
                rank: 0,
            });
        }
        assert!(b.pool_release(1).is_empty(), "第一块入池");
        assert_eq!(b.pool_release(2), vec![1], "超上限 ⇒ 顶出最旧(1),由调用方真还");
        assert_eq!(b.pool_stats(), (1, 4096), "池内只剩最新那块");
        assert_eq!(b.pool_evicted(), 1, "顶出计数");
        assert_eq!(b.pool_cap(), 4096, "上限可查(报告面)");
    }

    /// **第 0 刀:owner 维**(每字节唯一归属;出账逐 owner 判,不替他人平账)
    #[test]
    fn owner_dimension_balances_and_rejects_cross_owner_release() {
        let mut l = ledger(1 << 20);
        l.add_owned(Owner::Turn(7), Tier::Gpu, AssetKind::Elastic, 4096).expect("turn 入账");
        l.add_owned(Owner::Cache, Tier::Gpu, AssetKind::KvBlock, 8192).expect("cache 入账");
        // 存量入口落到 system(未接线桶)
        l.add(Tier::Gpu, AssetKind::Floor, 1024).expect("system 入账");
        assert_eq!(l.owner_bytes(Owner::Turn(7)), 4096);
        assert_eq!(l.owner_bytes(Owner::Cache), 8192);
        assert_eq!(l.owner_bytes(Owner::System), 1024);
        // 逐 owner 明细(字节降序)
        let d = l.detail();
        assert_eq!(d[0].owner, Owner::Cache, "最大的排头:{d:?}");
        assert_eq!(d.iter().map(|r| r.bytes).sum::<usize>(), 4096 + 8192 + 1024);
        // 跨 owner 出账 = 不变量违反(响亮拒;不替别人平账)
        let e = l.remove_owned(Owner::Turn(7), Tier::Gpu, AssetKind::KvBlock, 1).expect_err("跨 owner 必须拒");
        assert!(matches!(e, StoreError::Invariant(_)), "得 {e:?}");
        // 本 owner 超发也拒
        assert!(l.remove_owned(Owner::Turn(7), Tier::Gpu, AssetKind::Elastic, 8192).is_err());
        // 正常出账:两个账(分类 + owner)同步归零
        l.remove_owned(Owner::Turn(7), Tier::Gpu, AssetKind::Elastic, 4096).expect("出账");
        assert_eq!(l.owner_bytes(Owner::Turn(7)), 0);
        assert_eq!(l.kind_bytes(Tier::Gpu, AssetKind::Elastic), 0, "分类账同步");
        assert_eq!(l.used(Tier::Gpu), 8192 + 1024, "总量仍平");
        // report 带上 owner 维
        assert_eq!(l.report().owners.len(), 2);
    }
}

// ============================================================================
// §6 共享账本:`AllocLedger` 的 store 实现(DeviceHandle 钩子的落点)
// ============================================================================

/// **出借登记的一行**(事件账:`id` → 归属/类别/层/字节/点名)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Loan {
    /// 设备序(TP 两 rank 共用一本账时的分卡键;单卡恒 0)。
    pub rank: u8,
    /// 归属主体。
    pub owner: Owner,
    /// 资产类别(iface 中性类别的 store 映射)。
    pub kind: AssetKind,
    /// 当前所在层(换层时跟着变)。
    pub tier: Tier,
    /// 全量字节(登记值 —— 出账/换层都按它扣)。
    pub bytes: usize,
    /// 分配点点名(排障:哪一处要的这块;池复用**不改名** —— 名字 = 首生名,见收尾单 P2-2)。
    pub name: String,
    /// **是否在池内**(P0-2:同一块二次入池 = 别名事故 ⇒ 响亮拒)。
    /// 入池置 true;池取命中置 false;真归还随登记一起删。
    pub pooled: bool,
}

/// **共享账本单元** = 三层账(决策面)+ 出借登记(事件面)。
///
/// 为什么两件事同屋:它们必须**同时改**(一笔出借既进三层账,也进登记 ——
/// 出账时要按登记扣字节)。分两处就会分叉。
///
/// **store 依然零 I/O、零句柄**:它只被调用(`on_*` 改自己的账)。
pub struct LedgerBook {
    tiers: TierLedger,
    open: BTreeMap<u64, Loan>,
    hook_fail: u64,
    /// **片级换层跳过**计数(2026-10-19 真机:KV 池一个物理块 = 多层片,
    /// `swap` 逐片发 ⇒ "片和 ≠ 登记整块"是**常态口径**,不是记账失败)。
    /// 与 `hook_fail` 分开:后者必须恒为 0(真失败信号),本计数只说明
    /// "该资产的事件层账不适用(以腿自报为准)"。
    partial_moves: u64,
    /// **临时块池**(缺口 4;2026-10-19):`(kind, bytes) → 可复用 id 栈`(LIFO)。
    ///
    /// 三条律:
    /// 1. **同 rank + 同 kind + 同尺寸**才复用(P0-1:TP 两 rank 共写一本账 ⇒
    ///    键必须带 rank,否则跨卡发句柄;尺寸不等 = 形状/账目双错 ⇒ 不许);
    /// 2. **只有"已被持有者释放"的块入池**(`on_free` 路径),池成员 = 无活持有者
    ///    ⇒ 再发放不会造成别名/UAF;
    /// 3. 入池**不等于归还**:块仍在借(账上照旧占着),只是**可复用** ——
    ///    真还驱动 = 压力/关会话时的池排空(N6)。
    pool: BTreeMap<(u8, AssetKind, usize), Vec<u64>>,
    /// **池容量上限**(字节;2026-10-19 实测 27B 一轮后池囤到 669 MiB ⇒ 必须有界):
    /// 超限时 `pool_release` **顶出最旧块**并把 id 交回调用方(真还驱动)。
    /// 缺省 = `pool_cap_default()`;boot 可用 `set_pool_cap` 覆写。
    pool_cap: usize,
    /// 池命中/未命中(打点:复用率 = hits/(hits+misses))
    pool_hits: u64,
    pool_misses: u64,
    /// 池排空(真还驱动)计数 —— N6 前恒 0
    pool_drained: u64,
    /// **超上限顶出**计数(顶出的块已真还驱动)
    pool_evicted: u64,
}

impl LedgerBook {
    /// 构造(唯一入口:预算随账走)。
    pub fn new(budget: TierBudget) -> Self {
        Self {
            tiers: TierLedger::new(budget),
            open: BTreeMap::new(),
            hook_fail: 0,
            partial_moves: 0,
            pool: BTreeMap::new(),
            pool_cap: pool_cap_default(),
            pool_hits: 0,
            pool_misses: 0,
            pool_drained: 0,
            pool_evicted: 0,
        }
    }

    /// 三层账(决策/报告面)。
    pub fn tiers(&self) -> &TierLedger {
        &self.tiers
    }

    /// 三层账(可变;store 自查入口,如索引腿 `add_owned`)。
    pub fn tiers_mut(&mut self) -> &mut TierLedger {
        &mut self.tiers
    }

    /// 钩子失败计数(非零 = 有事件没入账;**不许静默**)。
    pub fn hook_fail(&self) -> u64 {
        self.hook_fail
    }

    /// 片级换层跳过计数(口径不支持;不是失败 —— 见字段注释)。
    pub fn partial_moves(&self) -> u64 {
        self.partial_moves
    }

    /// 当前出借条数(诊断面:与块账房/腿自报**互相校验**)。
    pub fn open_count(&self) -> usize {
        self.open.len()
    }

    /// 某 owner 的出借字节(诊断面)。
    pub fn open_bytes(&self, owner: Owner) -> usize {
        self.open.values().filter(|l| l.owner == owner).map(|l| l.bytes).sum()
    }

    /// 出借总字节(与块账房全等门的左值)。
    pub fn open_total_bytes(&self) -> usize {
        self.open.values().map(|l| l.bytes).sum()
    }

    /// **出借明细**(按 rank × owner × tier × kind 聚合;字节降序 —— 事件账报告面)。
    pub fn open_rows(&self) -> Vec<EventRow> {
        let mut agg: BTreeMap<(u8, Owner, Tier, AssetKind), (usize, usize)> = BTreeMap::new();
        for l in self.open.values() {
            let e = agg.entry((l.rank, l.owner, l.tier, l.kind)).or_insert((0, 0));
            e.0 += l.bytes;
            e.1 += 1;
        }
        let mut rows: Vec<EventRow> = agg
            .into_iter()
            .map(|((rank, owner, tier, kind), (bytes, entries))| EventRow {
                rank,
                owner,
                tier,
                kind,
                bytes,
                entries,
            })
            .collect();
        rows.sort_by(|a, b| {
            b.bytes.cmp(&a.bytes).then(a.rank.cmp(&b.rank)).then(a.owner.cmp(&b.owner))
        });
        rows
    }

    /// 出借块 id 集(与块账房做**全等门**用;`rank = None` = 全部)。
    pub fn open_ids(&self, rank: Option<u8>) -> Vec<u64> {
        self.open
            .iter()
            .filter(|(_, l)| rank.is_none_or(|r| l.rank == r))
            .map(|(id, _)| *id)
            .collect()
    }

    /// **池取块**(同 kind + 同尺寸才给;`None` = 未命中 ⇒ 调用方走真分配)。
    pub fn pool_acquire(&mut self, rank: u8, kind: AssetKind, bytes: usize) -> Option<u64> {
        let id = self.pool.get_mut(&(rank, kind, bytes)).and_then(|v| v.pop());
        match id {
            Some(id) => {
                self.pool_hits = self.pool_hits.saturating_add(1);
                // 出池即回到"借出"态(再次入池前不许被二次释放)
                if let Some(l) = self.open.get_mut(&id) {
                    l.pooled = false;
                }
                Some(id)
            }
            None => {
                self.pool_misses = self.pool_misses.saturating_add(1);
                None
            }
        }
    }

    /// **池还块**(只在 `on_free` 路径调:持有者已声明不再引用)。
    ///
    /// 返回 = **必须真归还驱动**的 id 列表(调用方逐条 `free` + `FreeEvent`):
    /// - 空 = 本次**已入池**(调用方不要碰驱动);
    /// - `[id]` = 该块不可池化/不在借(同一个 id),或**单块就超池上限**;
    /// - `[old…]` = 为腾容量**顶出的最旧块**(可能**多块**)。
    ///
    /// ⚠️ **2026-10-20 定谳**:旧实现只返回**最后一块**被顶出的 id —— 一次
    /// 还块要腾多块时,前面被顶出的 id 从池里摘掉却**无人释放**:设备内存
    /// 永远不还、loan 永远 `pooled=true` 且不在池里 ⇒ 每 chunk 泄漏数 MiB
    /// (GGUF 27B 262k 联调实测:5k prompt 泄漏 314 MiB,turn 账与池账对不上)。
    /// 本函数现在**逐块收集**顶出集,一个都不丢。
    ///
    /// 顶出策略 = **FIFO 最旧**(池是"复用缓存",旧块留着只是白占;上限内保留
    /// 最近释放的,与 CPU cache 的 LRU 直觉一致)。计数入 `pool_evicted`。
    pub fn pool_release(&mut self, id: u64) -> Vec<u64> {
        let Some(l) = self.open.get(&id).cloned() else { return vec![id] };
        if !l.kind.poolable() {
            return vec![id];
        }
        // **P0-2**:同一块二次入池 ⇒ 池里会有两份 ⇒ 两个未来 alloc 拿到同一 id(别名)。
        // 响亮拒(计数可见),且**不**把它交回调用方去 free(它已在池里)。
        if l.pooled {
            self.fail("双次入池(同一块重复释放)", format!("id={id} name={}", l.name));
            return Vec::new();
        }
        let key = (l.rank, l.kind, l.bytes);
        // 上限:池内字节 + 本块 > cap ⇒ 顶出最旧块(跨尺寸类也顶,先保总量)。
        // **所有**被顶出的块都要交回调用方(见函数头注的 2026-10-20 定谳)。
        let mut evicted = Vec::new();
        while self.pool_bytes() + l.bytes > self.pool_cap {
            let Some(old_key) = self
                .pool
                .iter()
                .find(|(_, v)| !v.is_empty())
                .map(|(k, _)| *k)
            else {
                break;
            };
            if let Some(v) = self.pool.get_mut(&old_key) {
                if let Some(old) = v.first().copied() {
                    v.remove(0);
                    self.pool_evicted = self.pool_evicted.saturating_add(1);
                    evicted.push(old);
                    continue;
                }
            }
            self.pool.remove(&old_key);
        }
        // 单块就超上限(极端档):腾空也装不下 ⇒ 本块也真归还(池不可用)。
        if self.pool_bytes() + l.bytes > self.pool_cap {
            evicted.push(id);
            return evicted;
        }
        self.pool.entry(key).or_default().push(id);
        if let Some(m) = self.open.get_mut(&id) {
            m.pooled = true;
        }
        evicted
    }

    /// 池内字节(键 × 块数的快速求和)。
    fn pool_bytes(&self) -> usize {
        self.pool.iter().map(|((_, _, b), v)| b * v.len()).sum()
    }

    /// boot/测试:设池容量上限(0 = 不限;>0 = 硬上限)。
    pub fn set_pool_cap(&mut self, bytes: usize) {
        self.pool_cap = bytes;
    }

    /// 池上限(报告面)。
    pub fn pool_cap(&self) -> usize {
        self.pool_cap
    }

    /// **N4 池清点**:把**池内**`from` 名下的块改归属 `to`(池 = 共享复用面,
    /// 不再是某个 turn 的临时面),返回 `(块数, 字节)`。
    ///
    /// 为什么安全:池成员**按构造**无活持有者(只在 `free` 路径入池)⇒ 换归属
    /// 不改变任何人的持有事实;而"真没归还"的块(不在池里)**不换** ——
    /// 它们保持 `Turn(t)` 归属,继续在 N4 清单里点名(泄漏 vs 复用从此可分)。
    pub fn pool_sweep(&mut self, from: Owner, to: Owner) -> (usize, usize) {
        let pooled: Vec<u64> = self.pool.values().flatten().copied().collect();
        let mut n = 0usize;
        let mut bytes = 0usize;
        for id in pooled {
            let Some(l) = self.open.get(&id).cloned() else { continue };
            if l.owner != from {
                continue;
            }
            if self.tiers.reown(l.owner, to, l.tier, l.kind, l.bytes).is_err() {
                self.fail("池清点改归属失败", format!("id={id}"));
                continue;
            }
            if let Some(m) = self.open.get_mut(&id) {
                m.owner = to;
            }
            n += 1;
            bytes += l.bytes;
        }
        (n, bytes)
    }

    /// **池排空**(N6 压力路径):清空池并**交出全部 id** —— 调用方负责真归还驱动
    /// (逐条 `FreeEvent` 销账)。返回的 id 顺序 = 池内 LIFO 逆序(无语义)。
    pub fn pool_drain(&mut self) -> Vec<u64> {
        let ids: Vec<u64> = self.pool.values_mut().flat_map(|v| v.drain(..)).collect();
        self.pool.clear();
        if !ids.is_empty() {
            self.pool_drained = self.pool_drained.saturating_add(1);
        }
        ids
    }

    /// **按类明细**(P2-3;报告面):`(kind, entries, bytes)`,字节降序。
    pub fn pool_by_class(&self) -> Vec<(AssetKind, usize, usize)> {
        let mut agg: BTreeMap<AssetKind, (usize, usize)> = BTreeMap::new();
        for ((_, kind, bytes), v) in &self.pool {
            let e = agg.entry(*kind).or_insert((0, 0));
            e.0 += v.len();
            e.1 += bytes * v.len();
        }
        let mut out: Vec<(AssetKind, usize, usize)> =
            agg.into_iter().map(|(k, (n, b))| (k, n, b)).collect();
        out.sort_by(|a, b| b.2.cmp(&a.2));
        out
    }

    /// 池内块数 / 字节(一等指标)。
    pub fn pool_stats(&self) -> (usize, usize) {
        let n = self.pool.values().map(|v| v.len()).sum();
        let b: usize = self.pool.iter().map(|((_, _, bytes), v)| bytes * v.len()).sum();
        (n, b)
    }

    /// 池命中/未命中/排空计数。
    pub fn pool_counters(&self) -> (u64, u64, u64) {
        (self.pool_hits, self.pool_misses, self.pool_drained)
    }

    /// 超上限顶出计数(顶出块已真还驱动)。
    pub fn pool_evicted(&self) -> u64 {
        self.pool_evicted
    }

    /// **在借清单**(逐条带 **id 与点名**;N4"未归还清单"的原料)。
    ///
    /// `owner_filter`: `None` = 全部;`Some(f)` = 只留 `f(owner)` 为真的那些
    /// (报告面按需取 lifecycle 面 —— turn/session,即"本该在 N4/N5 归还"的那些)。
    /// 排序 = 点名 → owner → id(同名成组,便于"哪一处要的、要了几块")。
    pub fn open_loans(
        &self,
        owner_filter: impl Fn(Owner) -> bool,
    ) -> Vec<(u64, Loan)> {
        let mut out: Vec<(u64, Loan)> = self
            .open
            .iter()
            .filter(|(_, l)| owner_filter(l.owner))
            .map(|(id, l)| (*id, l.clone()))
            .collect();
        out.sort_by(|a, b| {
            a.1.name
                .cmp(&b.1.name)
                .then(a.1.owner.cmp(&b.1.owner))
                .then(a.0.cmp(&b.0))
        });
        out
    }

    /// **逐 rank 的出借 id 集**(与块账房做**全等门**的原料;按 rank 升序)。
    pub fn open_ids_by_rank(&self) -> Vec<(u8, Vec<u64>)> {
        let mut agg: BTreeMap<u8, Vec<u64>> = BTreeMap::new();
        for (id, l) in &self.open {
            agg.entry(l.rank).or_default().push(*id);
        }
        agg.into_iter().collect()
    }

    /// 某 rank 的出借字节(`None` = 全部)。
    pub fn open_total_bytes_rank(&self, rank: Option<u8>) -> usize {
        self.open
            .values()
            .filter(|l| rank.is_none_or(|r| l.rank == r))
            .map(|l| l.bytes)
            .sum()
    }

    /// 出借条目数(`None` = 全部)。
    pub fn open_count_rank(&self, rank: Option<u8>) -> usize {
        self.open.values().filter(|l| rank.is_none_or(|r| l.rank == r)).count()
    }

    /// 记账失败:计数 + 一条裸日志(store 无日志依赖 ⇒ eprintln;机读面 = `hook_fail`)。
    fn fail(&mut self, what: &str, detail: String) {
        self.hook_fail = self.hook_fail.saturating_add(1);
        eprintln!("[store.ledger] {what} 未入账:{detail}");
    }

    /// 出借(登记 + 入账)。失败**响亮计数**,不 panic、不静默。
    pub fn on_alloc(&mut self, ev: &owl_iface::handle::AllocEvent) {
        if self.open.contains_key(&ev.id) {
            self.fail("双持(同 id 重复 alloc)", format!("id={} name={}", ev.id, ev.name));
            return;
        }
        let kind = AssetKind::from(ev.class);
        // 分配发生在设备侧 ⇒ tier = Gpu(下沉/回加载另发 on_move)
        let res = self.tiers.add_owned(ev.owner, Tier::Gpu, kind, ev.bytes);
        match res {
            Ok(()) => {
                self.open.insert(
                    ev.id,
                    Loan {
                        rank: ev.rank,
                        owner: ev.owner,
                        kind,
                        tier: Tier::Gpu,
                        bytes: ev.bytes,
                        name: ev.name.clone(),
                        pooled: false,
                    },
                );
            }
            Err(e) => self.fail("入账失败", format!("id={} {e}", ev.id)),
        }
    }

    /// 归还(注销 + 出账;字节按**登记**扣 —— 事件只带 id)。
    pub fn on_free(&mut self, ev: &owl_iface::handle::FreeEvent) {
        let Some(l) = self.open.remove(&ev.id) else {
            self.fail("幽灵归还(未登记)", format!("id={}", ev.id));
            return;
        };
        let res = self.tiers.remove_owned(l.owner, l.tier, l.kind, l.bytes);
        if let Err(e) = res {
            self.fail("出账失败", format!("id={} {e}", ev.id));
        }
    }

    /// 换层(**不注销**:owner/entries 不变;tier 迁移)。
    /// 部分搬运(片和 ≠ 登记整块字节)= **响亮拒**(不许把"沉了一半"记成"整块下沉")。
    pub fn on_move(&mut self, ev: &owl_iface::handle::MoveEvent) {
        use owl_iface::handle::MoveDir;
        let Some(l) = self.open.get(&ev.id).cloned() else {
            self.fail("幽灵换层(未登记)", format!("id={}", ev.id));
            return;
        };
        if ev.bytes != 0 && ev.bytes != l.bytes {
            // **片级换层 = 口径不支持,不是失败**(2026-10-19 真机定谳):
            // KV 池一个物理块 = 多层片(`kv.k.<层>` × 每页字节),`swap` 按片发
            // ⇒ 片和 < 登记整块是**常态**。事件账按整块记 ⇒ 该资产的层归属
            // **以腿自报为准**(`kv.host_blocks`);这里只计数(首条+每 1024 条一条日志)。
            // ⚠️ 真"半块搬运"(调用方切半)同样落这里 —— 判据是"片和 ≠ 登记",
            // 无法与常态区分;靠腿自报与事件账的对账面兜底(§13.4)。
            self.partial_moves = self.partial_moves.saturating_add(1);
            let n = self.partial_moves;
            if n == 1 || n.is_multiple_of(1024) {
                eprintln!(
                    "[store.ledger] 片级换层跳过(账按整块;腿自报层账为准,累计 {n}):                     id={} 片和={} 登记={} name={}",
                    ev.id, ev.bytes, l.bytes, l.name
                );
            }
            return;
        }
        let to = match (ev.dir, l.tier) {
            (MoveDir::Out, Tier::Gpu) => Tier::Host,
            (MoveDir::In, Tier::Host) => Tier::Gpu,
            (d, t) => {
                self.fail("换层方向与所在层不符", format!("id={} {d:?} from={t:?}", ev.id));
                return;
            }
        };
        let res = self.tiers.move_owned(l.owner, l.kind, l.tier, to, l.bytes);
        if let Err(e) = res {
            self.fail("换层失败", format!("id={} {e}", ev.id));
            return;
        }
        if let Some(m) = self.open.get_mut(&ev.id) {
            m.tier = to;
        }
    }
}

/// **共享账本句柄**(单一真源:client 钩子与 store 自查**共看同一本**)。
///
/// 为什么 `Arc<Mutex>` 而不是 `Rc<RefCell>`(2026-10-19 用户裁决):`DeviceClient`
/// 的 `Send` 是硬约束,且 `/debug/blocks` 的 `face_clone()` 会把句柄真的送进
/// HTTP 线程 ⇒ `Rc` 在那条路径上是真的 UB。单线程热路径无争用锁 ≈ ns 级。
pub type SharedLedger = std::sync::Arc<std::sync::Mutex<LedgerBook>>;

/// **池容量缺省上限** = 256 MiB。
///
/// 依据(2026-10-19 真机):27B 单轮释放后池天然囤到 **669 MiB**;而下一轮几乎
/// 不会用到同一尺寸集合的全部(复用率 ~37%)⇒ 无界池 = 新形态囤积。256 MiB
/// 覆盖实测单轮热集的大头,h按超出量逐块顶出回驱动。
pub fn pool_cap_default() -> usize {
    256 << 20
}

/// **建账**(boot 唯一入口;预算只能给一次 —— 池上限随账走)。
pub fn shared_ledger(budget: TierBudget) -> SharedLedger {
    std::sync::Arc::new(std::sync::Mutex::new(LedgerBook::new(budget)))
}

/// **加锁取账**:锁中毒 = 有 panic 发生在持账期间(设计上不应发生)⇒
/// 取回内部数据继续(**不许 panic 级联**),并计数可见(`hook_fail`)。
pub fn book_lock(cell: &SharedLedger) -> std::sync::MutexGuard<'_, LedgerBook> {
    let poisoned = cell.is_poisoned();
    let mut g = match cell.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    if poisoned {
        g.fail("账本锁中毒", "已续用内部数据(计数 +1)".into());
    }
    g
}

/// **事件入口**(`DeviceHandle` 的账本面):薄转发到 [`LedgerBook`]。
///
/// 两层锁的分工:**外层**锁的是"事件入口对象"(iface 侧 `LedgerCell` 是
/// `Arc<Mutex<dyn AllocLedger>>`),**内层**锁的是账本本体(store 自查也要它)。
/// 两层都只在同步段里短暂持有,绝不跨 await;内层锁持有时不回调任何外部代码。
pub struct LedgerEntry {
    book: SharedLedger,
}

impl owl_iface::handle::AllocLedger for LedgerEntry {
    fn on_alloc(&mut self, ev: &owl_iface::handle::AllocEvent) {
        book_lock(&self.book).on_alloc(ev);
    }

    fn pool_acquire(
        &mut self,
        rank: u8,
        class: owl_iface::handle::AssetClass,
        bytes: usize,
    ) -> Option<u64> {
        let kind = AssetKind::from(class);
        book_lock(&self.book).pool_acquire(rank, kind, bytes)
    }

    fn pool_release(&mut self, id: u64) -> Vec<u64> {
        book_lock(&self.book).pool_release(id)
    }
    fn on_free(&mut self, ev: &owl_iface::handle::FreeEvent) {
        book_lock(&self.book).on_free(ev);
    }
    fn on_move(&mut self, ev: &owl_iface::handle::MoveEvent) {
        book_lock(&self.book).on_move(ev);
    }
}

/// **boot 接线**:账本单元 → `DeviceHandle` 的事件入口(`LedgerCell`)。
///
/// 用途:`DeviceHandle::new(client, Some(ledger_entry(&cell)))`,
/// 同时把 `cell` 交给 `Rel::open`(两边看**同一本账**)。
pub fn ledger_entry(cell: &SharedLedger) -> owl_iface::handle::LedgerCell {
    std::sync::Arc::new(std::sync::Mutex::new(LedgerEntry { book: cell.clone() }))
}
