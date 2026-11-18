//! `Rel` 的报告面（两腿 + 保留/弹性 + 事件账）。
//!
//! 只读：report / leg_accounting / ledger_cell / leg_stats；不改变任何账。

use super::*;

impl Rel {
    /// 报告(两腿 + 保留/弹性面)。
    ///
    /// `tiers` 已拆成 **cells**(事件/语义互斥账)与 **views**(腿自报逻辑面)；
    /// 同 key 的腿行已扣除 cells，调用方不得再把两者相加。
    pub fn report(&self) -> StoreReport {
        // **第 0.5 刀:腿自报 → owner 行(对账式记账)**
        //
        // 为什么先"对账"而不是"事件记账":引擎有几十处直接 poke 腿
        // (`bank.rel.leg.*` / `bank.rel.state.*`),事件记账要逐个接线(后续);
        // 而对账在**报告时刻**取腿的权威状态 ⇒ 零漏记风险(腿是唯一真源),
        // 已能回答"谁占着多少"与"逐轮是否爬坡"(泄漏判据)。
        // 事件式记账的收益 = 精确区分"仍被持有"vs"未归还",留作接线后的第二步。
        let mut tiers = crate::tier::book_lock(&self.ledger).tiers().report();
        let legs = self.leg_accounting();
        // **cells / views 分家**：ledger 的 cells 是互斥账；腿自报是逻辑视图。
        // 同 key 的腿行先扣掉 cells（否则 `/debug/vram` 把同一字节算两遍）。
        let mut cell_map: std::collections::BTreeMap<(Tier, AssetKind), (usize, usize)> =
            std::collections::BTreeMap::new();
        for row in &tiers.cells {
            let e = cell_map.entry((row.tier, row.kind)).or_default();
            e.0 = e.0.saturating_add(row.bytes);
            e.1 += row.entries;
        }
        let mut views = Vec::new();
        for row in legs.views {
            let (cell_bytes, cell_entries) = cell_map
                .get(&(row.tier, row.kind))
                .copied()
                .unwrap_or((0, 0));
            let bytes = row.bytes.saturating_sub(cell_bytes);
            let entries = row.entries.saturating_sub(cell_entries);
            if bytes > 0 {
                views.push(crate::tier::TierRow {
                    tier: row.tier,
                    kind: row.kind,
                    bytes,
                    entries,
                });
            }
        }
        tiers.views = views;
        tiers.gpu_views_bytes = tiers
            .views
            .iter()
            .filter(|r| r.tier == crate::tier::Tier::Gpu)
            .map(|r| r.bytes)
            .sum();
        // owner 维也只保留一份：ledger owner + 腿 owner 扣掉同 key 的 ledger 部分。
        let mut cell_owner_map: std::collections::BTreeMap<
            (crate::tier::Owner, Tier, AssetKind),
            (usize, usize),
        > = std::collections::BTreeMap::new();
        for o in &tiers.owners {
            let e = cell_owner_map.entry((o.owner, o.tier, o.kind)).or_default();
            e.0 = e.0.saturating_add(o.bytes);
            e.1 += o.entries;
        }
        for o in legs.owners {
            let (cell_bytes, cell_entries) = cell_owner_map
                .get(&(o.owner, o.tier, o.kind))
                .copied()
                .unwrap_or((0, 0));
            let bytes = o.bytes.saturating_sub(cell_bytes);
            let entries = o.entries.saturating_sub(cell_entries);
            if bytes > 0 {
                tiers.owners.push(crate::tier::OwnerRow {
                    owner: o.owner,
                    tier: o.tier,
                    kind: o.kind,
                    bytes,
                    entries,
                });
            }
        }
        let mut owner_map: std::collections::BTreeMap<
            (crate::tier::Owner, Tier, AssetKind),
            (usize, usize),
        > = std::collections::BTreeMap::new();
        for o in tiers.owners.drain(..) {
            let e = owner_map.entry((o.owner, o.tier, o.kind)).or_default();
            e.0 = e.0.saturating_add(o.bytes);
            e.1 += o.entries;
        }
        tiers.owners = owner_map
            .into_iter()
            .map(|((owner, tier, kind), (bytes, entries))| crate::tier::OwnerRow {
                owner,
                tier,
                kind,
                bytes,
                entries,
            })
            .collect();
        tiers
            .owners
            .sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.owner.cmp(&b.owner)));
        // host/盘总账 = cells + views 的层和；GPU 只报 cells（views 在 Pool 内）。
        tiers.host_used = tiers
            .cells
            .iter()
            .chain(tiers.views.iter())
            .filter(|r| r.tier == crate::tier::Tier::Host)
            .map(|r| r.bytes)
            .sum();
        tiers.disk_used = tiers
            .cells
            .iter()
            .chain(tiers.views.iter())
            .filter(|r| r.tier == crate::tier::Tier::Disk)
            .map(|r| r.bytes)
            .sum();
        StoreReport {
            tiers,
            kv: {
                // 宿主账漂指纹合并(真源 = 宿主层;`KvStore` 看不到宿主 ⇒ 报告处补)。
                // **tier 真相**：`KvStore` 的 `host_blocks` 是 `slot=None` 二值计数，
                // 盘上-only 也在内；报告面必须从 DiskTier 扣掉，单独报 `disk_blocks`。
                let mut k = self.leg.kv_stats();
                let disk_entries = self.disk.as_ref().map(|d| d.resident_count()).unwrap_or(0);
                k.disk_blocks = disk_entries;
                k.host_blocks = k.host_blocks.saturating_sub(disk_entries);
                k.host_duplicate_landings = self.leg.host_duplicate_landings();
                k.host_drop_missing = self.leg.host_drop_missing();
                k
            },
            kv_diag: self.leg.kv_diag(),
            state: self.state.stats(),
            active_turns: self.admission.active_turns(),
            reserved_bytes: self.admission.reserved_bytes(),
            reserved_slots: self
                .reservations
                .reserved_slots_of(ReservationKind::ActiveFloor),
            in_flight_turns: self.reservations.in_flight_turns(),
            grants: self.reservations.report(),
            session_held_bytes: self.admission.session_held_bytes(),
            elastic_bytes: self.admission.elastic_bytes(),
            cache_bytes: self.admission.cache_bytes(),
            policy: self.evict.name().to_string(),
            host_resident_bytes: self.leg.host_used_bytes(),
            host_free_bytes: self.leg.host_free_bytes(),
            disk_kv: self.disk.as_ref().map(|d| d.stats()),
            disk_snap: self.snap_disk.as_ref().map(|d| d.stats()),
            disk_sessions: self.disk_session_usage(),
            plan: self.plan.clone(),
            events: {
                // 事件账(出借登记;与腿自报互为校验 —— 两个独立来源)
                let b = crate::tier::book_lock(&self.ledger);
                EventBook {
                    rows: b.open_rows(),
                    entries: b.open_count_rank(None),
                    bytes: b.open_total_bytes_rank(None),
                    hook_fail: b.hook_fail(),
                    partial_moves: b.partial_moves(),
                    ids: b.open_ids_by_rank(),
                    open_lifecycle: b
                        .open_loans(|o| o.is_lifecycle())
                        .into_iter()
                        .map(|(id, l)| OpenLoan {
                            id,
                            name: l.name,
                            owner: l.owner,
                            kind: l.kind,
                            bytes: l.bytes,
                            rank: l.rank,
                        })
                        .collect(),
                }
            },
        }
    }

    /// **腿自报 → owner 行**(第 0.5 刀;对账式记账的原料)。
    ///
    /// 归属规则(有意保守,不猜):
    /// - KV 块(设备/宿主)= `Cache`(跨会话共享,无单会话归属);
    /// - 快照(设备/宿主)= `Cache`(Durable 可下沉 / Transient 逐出即丢,都是缓存面);
    /// - GDN 行 = 借行时登记的 `Turn(t)`（P1-5；无归属兜底 `System`）。
    fn leg_accounting(&self) -> crate::tier::TierReport {
        let mut owners: Vec<crate::tier::OwnerRow> = Vec::new();
        let mut rows: Vec<crate::tier::TierRow> = Vec::new();
        let push = |owners: &mut Vec<crate::tier::OwnerRow>,
                    rows: &mut Vec<crate::tier::TierRow>,
                    owner: crate::tier::Owner,
                    tier: Tier,
                    kind: AssetKind,
                    bytes: usize,
                    entries: usize| {
            if bytes == 0 {
                return;
            }
            owners.push(crate::tier::OwnerRow {
                owner,
                tier,
                kind,
                bytes,
                entries,
            });
            rows.push(crate::tier::TierRow {
                tier,
                kind,
                bytes,
                entries,
            });
        };
        // ① KV 腿:设备驻留(gpu_bytes)+ 宿主驻留(host_used_bytes)
        let kv = self.leg.kv_stats();
        let kv_dev_entries = kv.cached_blocks.saturating_sub(kv.host_blocks);
        push(
            &mut owners,
            &mut rows,
            crate::tier::Owner::Cache,
            Tier::Gpu,
            AssetKind::KvBlock,
            kv.gpu_bytes,
            kv_dev_entries,
        );
        {
            // 非设备面的 KV 块 = RAM 驻留(真源 = 宿主层账) + 磁盘驻留(DiskTier 账);
            // `kv.host_blocks`(tree 的 `slot = None` 计数)是两者之和 ⇒ 分开报,不重复计。
            let disk_entries = self.disk.as_ref().map(|d| d.resident_count()).unwrap_or(0);
            let disk_bytes = self.disk.as_ref().map(|d| d.used_bytes()).unwrap_or(0);
            let host_entries = kv.host_blocks.saturating_sub(disk_entries);
            push(
                &mut owners,
                &mut rows,
                crate::tier::Owner::Cache,
                Tier::Host,
                AssetKind::KvBlock,
                self.leg.host_used_bytes(),
                host_entries,
            );
            push(
                &mut owners,
                &mut rows,
                crate::tier::Owner::Cache,
                Tier::Disk,
                AssetKind::KvBlock,
                disk_bytes,
                disk_entries,
            );
        }
        // ② 状态腿:快照(设备/宿主)+ 行
        let st = self.state.stats();
        push(
            &mut owners,
            &mut rows,
            crate::tier::Owner::Cache,
            Tier::Gpu,
            AssetKind::Snap,
            st.snap_gpu.1,
            st.snap_gpu.0,
        );
        push(
            &mut owners,
            &mut rows,
            crate::tier::Owner::Cache,
            Tier::Host,
            AssetKind::Snap,
            st.snap_host.1,
            st.snap_host.0,
        );
        // P1-5：GDN 行按借行 owner 分组；正常 = `Turn(t)`。
        let rows_in_use = st.rows_total.saturating_sub(st.rows_free);
        let mut row_groups: std::collections::BTreeMap<crate::tier::Owner, usize> =
            std::collections::BTreeMap::new();
        for (_row, owner) in self.state.row_owners() {
            *row_groups.entry(owner).or_insert(0) += 1;
        }
        let mut owned_rows = 0usize;
        for (owner, n) in row_groups {
            owned_rows += n;
            push(
                &mut owners,
                &mut rows,
                owner,
                Tier::Gpu,
                AssetKind::Row,
                n * self.state.row_bytes(),
                n,
            );
        }
        let unowned = rows_in_use.saturating_sub(owned_rows);
        if unowned > 0 {
            push(
                &mut owners,
                &mut rows,
                crate::tier::Owner::System,
                Tier::Gpu,
                AssetKind::Row,
                unowned * self.state.row_bytes(),
                unowned,
            );
        }
        crate::tier::TierReport {
            views: rows,
            owners,
            ..Default::default()
        }
    }

    /// **账本单元**(boot 接线:DeviceHandle 的事件入口与 store 的 `Rel` 要**同一本**)
    pub fn ledger_cell(&self) -> crate::tier::SharedLedger {
        self.ledger.clone()
    }

    /// 腿报告(KV 腿的完整面:槽/缓存/宿主/搬运计数 —— 报告与对账口用)。
    ///
    /// **tier 真相**：`BlockLeg` 只看到 `slot=None`，盘上-only 也会被报成
    /// Host；这里用 DiskTier 的驻留数校正，并单列 `disk_blocks`。
    pub fn leg_stats(&self) -> LegStats {
        let mut st = self.leg.report();
        let disk_entries = self.disk.as_ref().map(|d| d.resident_count()).unwrap_or(0);
        st.disk_blocks = disk_entries;
        st.host_blocks = st.host_blocks.saturating_sub(disk_entries);
        st
    }
}
