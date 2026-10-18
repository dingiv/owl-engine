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
        }
    }
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
    fn ledger_handle_alloc_free_balances_same_book() {
        use owl_iface::handle::{AllocEvent, AllocLedger, AssetClass, FreeEvent, Owner as IfOwner};
        let l = std::rc::Rc::new(std::cell::RefCell::new(ledger(1 << 20)));
        let mut h = LedgerHandle::new(l.clone());
        h.on_alloc(&AllocEvent {
            id: 1,
            bytes: 4096,
            name: "kv.workspace".into(),
            class: AssetClass::Temp,
            owner: IfOwner::Turn(7),
        });
        assert_eq!(h.open_count(), 1);
        assert_eq!(h.open_bytes(IfOwner::Turn(7)), 4096);
        // store 自己的账立刻看到(同一本)
        assert_eq!(l.borrow().kind_bytes(Tier::Gpu, AssetKind::Temp), 4096);
        assert_eq!(l.borrow().owner_bytes(IfOwner::Turn(7)), 4096);
        h.on_free(&FreeEvent { id: 1, bytes: 0 }); // 字节按登记扣
        assert_eq!(h.open_count(), 0);
        assert_eq!(l.borrow().kind_bytes(Tier::Gpu, AssetKind::Temp), 0);
        assert_eq!(h.hook_fail(), 0);
    }

    /// **步 2 门 ②**:双持 / 幽灵归还**响亮拒**(计数,不 panic、不静默)
    #[test]
    fn ledger_handle_rejects_double_alloc_and_ghost_free() {
        use owl_iface::handle::{AllocEvent, AllocLedger, AssetClass, FreeEvent, Owner as IfOwner};
        let l = std::rc::Rc::new(std::cell::RefCell::new(ledger(1 << 20)));
        let mut h = LedgerHandle::new(l.clone());
        let ev = AllocEvent {
            id: 9,
            bytes: 128,
            name: "dup".into(),
            class: AssetClass::Elastic,
            owner: IfOwner::Session(1),
        };
        h.on_alloc(&ev);
        h.on_alloc(&ev); // 双持
        assert_eq!(h.hook_fail(), 1, "双持必须计数");
        assert_eq!(l.borrow().kind_bytes(Tier::Gpu, AssetKind::Elastic), 128, "只入账一次");
        h.on_free(&FreeEvent { id: 999, bytes: 0 }); // 幽灵
        assert_eq!(h.hook_fail(), 2, "幽灵归还必须计数");
        assert_eq!(h.open_count(), 1, "合法那条仍在场");
    }

    /// **步 2 门 ③**:换层 = **不注销**(owner/entries 不变;tier 迁移)
    #[test]
    fn ledger_handle_move_keeps_owner_and_entries() {
        use owl_iface::handle::{AllocEvent, AllocLedger, AssetClass, MoveDir, MoveEvent, Owner as IfOwner};
        let l = std::rc::Rc::new(std::cell::RefCell::new(ledger(1 << 20)));
        let mut h = LedgerHandle::new(l.clone());
        h.on_alloc(&AllocEvent {
            id: 5,
            bytes: 8192,
            name: "snap.1".into(),
            class: AssetClass::Snap,
            owner: IfOwner::Cache,
        });
        h.on_move(&MoveEvent { id: 5, bytes: 8192, dir: MoveDir::Out });
        {
            let b = l.borrow();
            assert_eq!(b.kind_bytes(Tier::Gpu, AssetKind::Snap), 0, "设备层已清");
            assert_eq!(b.kind_bytes(Tier::Host, AssetKind::Snap), 8192, "宿主层在位");
            assert_eq!(b.owner_bytes(IfOwner::Cache), 8192, "owner 不变(下沉≠归还)");
            let rows = b.detail();
            assert_eq!(rows.len(), 1, "换层不多出一条:{rows:?}");
            assert_eq!(rows[0].entries, 1, "entries 跟着走");
        }
        assert_eq!(h.hook_fail(), 0);
        // 方向不符(Host 上再 Out)= 响亮拒
        h.on_move(&MoveEvent { id: 5, bytes: 8192, dir: MoveDir::Out });
        assert_eq!(h.hook_fail(), 1, "方向不符必须计数");
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
// §6 账本句柄:`AllocLedger` 的 store 实现(DeviceHandle 钩子的落点)
// ============================================================================

/// **账本句柄**(实现 `owl_iface::handle::AllocLedger`;供 `DeviceHandle` 在客户端侧调用)。
///
/// 与 [`TierLedger`] 是**同一本账**(共享 `Rc<RefCell<TierLedger>>`):事件进来即入账,
/// store 自己的 `report()/detail()` 立刻看到同样的数 —— 这正是"store 主管显存账"的落点,
/// 而 store 依然**零 I/O、零句柄**(它只被调用)。
///
/// **出借登记**(`open`):`id → (owner, kind, tier, bytes)`。
/// `FreeEvent` 只带 id ⇒ 字节/归属按登记扣;双持(同 id 重复 alloc)与幽灵归还
/// (未登记就 free/move)**响亮拒**并计入 `hook_fail`(不 panic、不静默)。
pub struct LedgerHandle {
    ledger: std::rc::Rc<std::cell::RefCell<TierLedger>>,
    open: BTreeMap<u64, (Owner, AssetKind, Tier, usize)>,
    hook_fail: u64,
}

impl LedgerHandle {
    /// 构造(唯一入口:账本随句柄走)。
    pub fn new(ledger: std::rc::Rc<std::cell::RefCell<TierLedger>>) -> Self {
        Self { ledger, open: BTreeMap::new(), hook_fail: 0 }
    }

    /// 钩子失败计数(非零 = 有事件没入账;**不许静默**)。
    pub fn hook_fail(&self) -> u64 {
        self.hook_fail
    }

    /// 当前出借条数(诊断面:与腿自报的条目数**互相校验**)。
    pub fn open_count(&self) -> usize {
        self.open.len()
    }

    /// 某 owner 的出借字节(诊断面)。
    pub fn open_bytes(&self, owner: Owner) -> usize {
        self.open.values().filter(|(o, _, _, _)| *o == owner).map(|(_, _, _, b)| *b).sum()
    }

    /// 记账失败:计数 + 一条裸日志(store 无日志依赖 ⇒ eprintln;机读面 = `hook_fail`)。
    fn fail(&mut self, what: &str, detail: String) {
        self.hook_fail += 1;
        eprintln!("[store.ledger] {what} 未入账:{detail}");
    }
}

impl owl_iface::handle::AllocLedger for LedgerHandle {
    fn on_alloc(&mut self, ev: &owl_iface::handle::AllocEvent) {
        if self.open.contains_key(&ev.id) {
            self.fail("双持(同 id 重复 alloc)", format!("id={} name={}", ev.id, ev.name));
            return;
        }
        let kind = AssetKind::from(ev.class);
        // 分配发生在设备侧 ⇒ tier = Gpu(下沉/回加载另发 on_move)
        // 借用先物化再判(否则 `self.fail(&mut self)` 与 RefCell 临时借用冲突)
        let res = self.ledger.borrow_mut().add_owned(ev.owner, Tier::Gpu, kind, ev.bytes);
        match res {
            Ok(()) => {
                self.open.insert(ev.id, (ev.owner, kind, Tier::Gpu, ev.bytes));
            }
            Err(e) => self.fail("入账失败", format!("id={} {e}", ev.id)),
        }
    }

    fn on_free(&mut self, ev: &owl_iface::handle::FreeEvent) {
        let Some((owner, kind, tier, bytes)) = self.open.remove(&ev.id) else {
            self.fail("幽灵归还(未登记)", format!("id={}", ev.id));
            return;
        };
        let res = self.ledger.borrow_mut().remove_owned(owner, tier, kind, bytes);
        if let Err(e) = res {
            self.fail("出账失败", format!("id={} {e}", ev.id));
        }
    }

    fn on_move(&mut self, ev: &owl_iface::handle::MoveEvent) {
        use owl_iface::handle::MoveDir;
        let Some((owner, kind, from, bytes)) = self.open.get(&ev.id).copied() else {
            self.fail("幽灵换层(未登记)", format!("id={}", ev.id));
            return;
        };
        let to = match (ev.dir, from) {
            (MoveDir::Out, Tier::Gpu) => Tier::Host,
            (MoveDir::In, Tier::Host) => Tier::Gpu,
            (d, t) => {
                self.fail("换层方向与所在层不符", format!("id={} {d:?} from={t:?}", ev.id));
                return;
            }
        };
        let res = self.ledger.borrow_mut().move_owned(owner, kind, from, to, bytes);
        if let Err(e) = res {
            self.fail("换层失败", format!("id={} {e}", ev.id));
            return;
        }
        self.open.insert(ev.id, (owner, kind, to, bytes));
    }
}
