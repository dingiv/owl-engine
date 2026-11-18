//! `Rel` 的准入 / Grant 账（2026-11-18 S2a 从 `rel.rs` 拆出）。
//!
//! 只碰 store 自己的账：plan 是纯 dry-run，commit 按 engine 报回的实际值落账。
//! 真正的 I/O 在 engine；这里没有 `DeviceClient`，也没有事务对象。

use super::*;

impl Rel {
    /// **准入 dry-run**：只读账，返回“如果现在放行”的计划。
    ///
    /// 不建 Grant、不登记 admission、不借 GDN 行。完整请求链：
    /// ```text
    /// plan_admission -> Admit
    ///     -> commit_admission(实际 floor/槽数) -> 开始执行
    /// plan_admission -> EvictThen
    ///     -> plan_need(need)（store 描述搬哪些块/丢谁）
    ///     -> engine 执行 swap/drop
    ///     -> commit_offload/commit_onload(实际搬成的块)
    ///     -> 重新 plan_admission -> commit_admission
    /// ```
    /// 搬迁后账变了，旧 plan 不再有效；commit 只收实际完成值。
    pub fn plan_admission(
        &self,
        turn: u64,
        floor: TurnFloor,
        ctx_tokens: usize,
        reserved_slots: usize,
        alloc_slots: usize,
    ) -> AdmissionPlan {
        let reserved_slots = reserved_slots.max(1);
        let alloc_slots = alloc_slots.max(reserved_slots).max(1);
        let floor_bytes = floor.floor_bytes(
            ctx_tokens,
            self.budget.inputs().page_tokens,
            self.budget.inputs().kv_bytes_per_token,
        );
        let base = self.check(floor, ctx_tokens);
        // 活跃天花板/地板已经拒或排队：保持原判决，不再用完整预算改写原因。
        let verdict = if matches!(
            &base,
            AdmissionVerdict::Reject { .. } | AdmissionVerdict::Queue { .. }
        ) {
            base
        } else {
            self.plan_admission_full(floor, ctx_tokens, reserved_slots, floor_bytes, base)
        };
        AdmissionPlan {
            turn,
            floor,
            ctx_tokens,
            verdict,
            reserved_slots,
            alloc_slots,
        }
    }

    /// **完整预算 / 超卖落地**校验（`plan_admission` 的第二段）。
    ///
    /// `floor` 只是 per-turn 的显存下限；真正要批准的是 `reserved_slots`
    /// （非超卖 = ctx + max_new；超卖 = 设备工作集）。旧 `Admission::decide`
    /// 在生产路径被 `decide_slots` 替掉后，这一段不能一起丢：
    /// - 非超卖：`reserved` 字节放不进判据面 ⇒ EvictThen/Reject；
    /// - 超卖：`ctx_pages − reserved_pages` 必须有宿主/磁盘位可落。
    fn plan_admission_full(
        &self,
        floor: TurnFloor,
        ctx_tokens: usize,
        reserved_slots: usize,
        floor_bytes: usize,
        base: AdmissionVerdict,
    ) -> AdmissionVerdict {
        let inp = self.budget.inputs();
        let slot_bytes = self.leg.slot_bytes().max(1);
        let need_bytes = reserved_slots.saturating_mul(slot_bytes);
        let committed = self.admission.reserved_bytes()
            + self.admission.session_held_bytes()
            + self.admission.elastic_bytes()
            + self.admission.cache_bytes();
        let face = self.budget.face_bytes();
        let total = committed.saturating_add(need_bytes);
        if total > face {
            let short = total - face;
            if self.evictable_bytes() >= short {
                return AdmissionVerdict::EvictThen {
                    floor_bytes,
                    need_bytes: short,
                };
            }
            return crate::budget::verdict_for_full(
                self.policy.on_full,
                format!(
                    "完整预算 {reserved_slots} 页(={need_bytes} 字节)放不下：\
                     已用 {committed}，判据面 {face}，欠 {short}；\
                     可驱逐面仅 {}",
                    self.evictable_bytes()
                ),
            );
        }
        // 非 Strict = 允许虚拟化：设备只驻工作集，差额必须真的能落到宿主/磁盘。
        // Strict 不做这条（它的全量预算已由上面的 face 检查覆盖）。
        if !floor.is_strict() {
            let page = inp.page_tokens.max(1);
            let ctx_pages = ctx_tokens.div_ceil(page);
            let offload_pages = ctx_pages.saturating_sub(reserved_slots);
            if offload_pages > 0 {
                let host_pages = self
                    .leg
                    .host_total_slots()
                    .saturating_mul(self.leg.planes());
                let disk_pages = self
                    .kv_disk_free_slots()
                    .saturating_mul(self.kv_disk_block_bytes().unwrap_or(0).div_ceil(slot_bytes));
                let room = host_pages.saturating_add(disk_pages);
                if offload_pages > room {
                    return crate::budget::verdict_for_full(
                        self.policy.on_full,
                        format!(
                            "超卖档溢出 {offload_pages} 页 > 宿主/磁盘可落 {host_pages}+{disk_pages} 页\
                             （ctx {ctx_tokens} tok，驻留 {reserved_slots} 页）"
                        ),
                    );
                }
            }
        }
        base
    }

    /// **提交准入状态**：engine 按 plan 做完后，把**实际值**报给 store。
    ///
    /// 这是独立的状态修改 API，不是事务：没有 open/commit/abort 三态，
    /// 也没有回滚日志。`plan` 不传进来；engine 只报实际批准的
    /// `floor_bytes / reserved_slots / alloc_slots`。
    ///
    /// 可失败前置（turn 重复、GDN 行耗尽/重复）都在第一次改账前检查；
    /// 调用方应只在 plan 为 `Admit` 后调用。
    pub fn commit_admission(
        &mut self,
        turn: u64,
        floor_bytes: usize,
        reserved_slots: usize,
        alloc_slots: usize,
    ) -> StoreResult<AdmissionApplied> {
        if turn == 0 {
            return Err(StoreError::Invariant(
                "commit_admission: turn = 0（缺少归属）".into(),
            ));
        }
        if self.admission.has_active(turn) {
            return Err(StoreError::Invariant(format!(
                "commit_admission: turn {turn} 已在准入账"
            )));
        }
        // 行是唯一可能被“借空”的资源；先看，再改账。
        if self.state.rows_free() == 0 {
            return Err(StoreError::Exhausted("GDN 行耗尽".into()));
        }
        let owner = crate::tier::Owner::Turn(turn);
        if self.state.row_owners().any(|(_, o)| o == owner) {
            return Err(StoreError::Invariant(format!(
                "commit_admission: turn {turn} 已占一行"
            )));
        }
        // 先做会失败且会改账的 Grant；失败时尚未动 admission/行。
        let grant = self.reserve_capacity(
            turn,
            reserved_slots,
            alloc_slots,
            ReservationKind::ActiveFloor,
        )?;
        // 到这里前置已成立：准入登记、借行、行票都不会因资源不足失败。
        // `floor_bytes` 不是报告字段：它必须立刻进入准入账；
        // 否则同一拍内下一次 plan 会把刚提交的 turn 当成零地板。
        self.admission.admit(turn, floor_bytes, 0)?;
        let gdn_row = self.borrow_row(owner)?;
        let row_grant = self.reserve_row(turn)?;
        Ok(AdmissionApplied {
            grant,
            gdn_row,
            row_grant,
            floor_bytes,
        })
    }
    /// **批准一张容量票**(准入后;store 账)。`slots` = 整条上下文/生成预算页面。
    pub fn reserve_capacity(
        &mut self,
        turn: u64,
        slots: usize,
        alloc_slots: usize,
        kind: ReservationKind,
    ) -> StoreResult<Grant> {
        let slot_bytes = self.leg.slot_bytes();
        let bytes = slots
            .checked_mul(slot_bytes)
            .ok_or_else(|| StoreError::Budget("reserve_capacity: 字节溢出".into()))?;
        let alloc_bytes = alloc_slots
            .checked_mul(slot_bytes)
            .ok_or_else(|| StoreError::Budget("reserve_capacity: 分配面字节溢出".into()))?;
        self.reserve_direct(turn, kind, slots, bytes, alloc_slots, alloc_bytes)
    }

    /// **通用票**（Row/Snap/Load/Temp 等非 KV 槽单位资源）。
    ///
    /// `slots`/`bytes` 由调用方按资源语义给；KV 准入只读
    /// [`ReservationKind::ActiveFloor`]，不会被这些票污染。
    pub fn reserve_direct(
        &mut self,
        turn: u64,
        kind: ReservationKind,
        slots: usize,
        bytes: usize,
        alloc_slots: usize,
        alloc_bytes: usize,
    ) -> StoreResult<Grant> {
        if turn == 0 {
            return Err(StoreError::Invariant(
                "reserve_direct: turn = 0(票必须有 turn 归属)".into(),
            ));
        }
        self.reservations.grant(
            turn,
            crate::tier::Owner::Turn(turn),
            kind,
            slots,
            bytes,
            alloc_slots,
            alloc_bytes,
        )
    }

    /// GDN 行票（1 行 = `row_bytes`；turn 生命周期）。
    pub fn reserve_row(&mut self, turn: u64) -> StoreResult<Grant> {
        let bytes = self.state.row_bytes();
        self.reserve_direct(turn, ReservationKind::Row, 1, bytes, 1, bytes)
    }

    /// 前缀回灌票（`blocks` 个 KV 槽；turn 生命周期）。
    pub fn reserve_load(&mut self, turn: u64, blocks: usize) -> StoreResult<Grant> {
        let bytes = blocks
            .checked_mul(self.leg.slot_bytes())
            .ok_or_else(|| StoreError::Budget("reserve_load: 字节溢出".into()))?;
        self.reserve_direct(turn, ReservationKind::Load, blocks, bytes, blocks, bytes)
    }

    /// 快照设备槽票(1 条 = **一条快照的总字节** `snap_bytes × world`;
    /// 与下沉检查/宿主块/盘文件同一口径 —— TP 下按单 rank 记会少一半,
    /// 消费时立刻撞「额度不足 159MB vs 79MB」)。
    pub fn reserve_snap(&mut self, turn: u64) -> StoreResult<Grant> {
        let bytes = self.snap_file_bytes();
        self.reserve_direct(turn, ReservationKind::Snap, 1, bytes, 1, bytes)
    }

    /// 临时工作区票（调用方给槽/字节；turn 生命周期或调用方显式归还）。
    pub fn reserve_temp(
        &mut self,
        turn: u64,
        slots: usize,
        bytes: usize,
    ) -> StoreResult<Grant> {
        self.reserve_direct(turn, ReservationKind::Temp, slots, bytes, slots, bytes)
    }

    /// 消费一张票的额度(只减 remaining;总额保留给准入面)。
    pub fn consume_capacity(
        &mut self,
        id: ReservationId,
        slots: usize,
        bytes: usize,
    ) -> StoreResult<()> {
        self.reservations.consume(id, slots, bytes)
    }

    /// 归还一张票。
    pub fn release_grant(&mut self, id: ReservationId) -> StoreResult<Grant> {
        self.reservations.release(id)
    }

    /// 归还一个 turn 的全部票(幂等;Release / 准入收口共用)。
    pub fn release_turn_grants(&mut self, turn: u64) -> Vec<Grant> {
        self.reservations.release_turn(turn)
    }

    /// **turn 终局单一围栏**：把该 turn 在 store 里的**一切**一次归还 ——
    /// Grant（ReservationLedger）+ 准入账（Admission）+ GDN 行（StateStore）。
    ///
    /// 为什么必须原子成一处：三本账同源（turn 生命周期），拆成三个调用就必然
    /// 出现"还了票没还准入 / 还了准入没还行"的半态 —— 2026-11-06 真机 3080：
    /// `start_one` 在 `admit + borrow_row` 之后失败，三本账只还了票，
    /// 准入与行永久挂着 ⇒ 之后每个请求都撞"活跃 turn 天花板(1/1)"。
    ///
    /// 幂等：重复调用 = 全零报告。**调用方只该调这一个**，不要再分三步。
    pub fn finish_turn(&mut self, turn: u64) -> StoreResult<TurnRelease> {
        let grants = self.release_turn_grants(turn).len();
        // release 返回 Unknown = 本来就不在账上（幂等语义，不是错）。
        let admission = self.admission.release(turn).is_ok();
        let row = self.state.release_row_of(crate::tier::Owner::Turn(turn))?;
        Ok(TurnRelease {
            admission,
            grants,
            row,
        })
    }

    /// **孤儿收口**：准入账里有、但不在 `live`（引擎当前活跃集）里的 turn。
    ///
    /// 这种条目只可能来自"start 半态"（`finish_turn` 没跑到）。引擎在
    /// `abandon` 拿不到归属 turn 时调它 —— 保证泄漏能自愈，而不是永久占天花板。
    pub fn finish_orphan_turns(
        &mut self,
        live: &[u64],
    ) -> StoreResult<Vec<(u64, TurnRelease)>> {
        let orphans: Vec<u64> = self
            .admission
            .active_ids()
            .filter(|id| !live.contains(id))
            .collect();
        let mut out = Vec::with_capacity(orphans.len());
        for id in orphans {
            out.push((id, self.finish_turn(id)?));
        }
        Ok(out)
    }

    /// 准入面:已批准总量(槽;含已兑现的活跃块,不能减)。
    pub fn reserved_slots(&self) -> usize {
        // 准入/槽位策略只读 KV 地板；Row/Snap/Load/Temp 是分维度票，
        // 不能污染 `reserved_pages`。
        self.reservations
            .reserved_slots_of(ReservationKind::ActiveFloor)
    }

    /// 准入面:已批准总字节。
    pub fn reserved_bytes(&self) -> usize {
        self.reservations
            .reserved_bytes_of(ReservationKind::ActiveFloor)
    }

    /// 在册 turn 数(槽位策略用)。
    pub fn in_flight_turns(&self) -> usize {
        self.reservations.in_flight_turns()
    }

    /// 准入判决(§2.4 准入式;`evictable` = 缓存面 + 弹性面可动字节)。
    pub fn check(&self, floor: TurnFloor, ctx_tokens: usize) -> AdmissionVerdict {
        let evictable = self.evictable_bytes();
        crate::budget::check_admission(
            &self.budget,
            &self.admission,
            floor,
            ctx_tokens,
            evictable,
            self.policy.on_full,
        )
    }

    /// 可驱逐面字节(缓存 + 弹性)。
    pub fn evictable_bytes(&self) -> usize {
        self.admission.cache_bytes() + self.admission.elastic_bytes()
    }

    /// **唯一调度入口**(§30):给需求 ⇒ 两腿统一的“谁走谁进/拒”。纯 host、零 I/O。
    ///
    /// 判据全在 [`crate::hooks::plan_capacity`](唯一产地);本方法只做两件事:
    /// 算各层可容纳量与“盘层是否启用”、把 `Rel` 的两腿账与策略对象递进去。
    /// engine 在**段边界**调本方法取单,自己发 `swap`/`drop`(§21 裁决)。
    pub fn plan_need(
        &self,
        need: crate::hooks::Need,
        protect: &Protected,
    ) -> crate::hooks::Verdict {
        let room = crate::hooks::Room {
            host_enabled: self.leg.host_enabled() || self.snap_host_budget() > 0,
            host_kv_slots: self.leg.host_free_slots(),
            host_snap_slots: self.snap_host_free_slots(),
            disk_kv_slots: self.disk.as_ref().map(|d| d.free_slots()).unwrap_or(0),
            disk_snap_slots: self.snap_disk.as_ref().map(|d| d.free_slots()).unwrap_or(0),
            disk_enabled: self.disk.is_some() || self.snap_disk.is_some(),
            bulk_drop_when_tier_full: self.bulk_drop_when_tier_full,
        };
        crate::hooks::plan_capacity(
            &self.leg,
            &self.state,
            self.evict.as_ref(),
            need,
            &protect.chains,
            &protect.snaps,
            room,
        )
    }
}
