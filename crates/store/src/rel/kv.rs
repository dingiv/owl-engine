//! `Rel` 的 KV 腿（块身份 / 前缀命中 / 宿主与设备搬运计划）。
//!
//! 只做 store 侧账与描述；真正的 swap/drop 在 engine。

use super::*;

impl Rel {
    // ---------------------------------------------------------------------
    // KV 腿语义面（store 内部字段私有；engine 只调这些方法）
    // ---------------------------------------------------------------------

    /// 注册 target 腿的设备布局（boot 一次）。
    pub fn attach_pool(&mut self, layout: PoolLayout) -> StoreResult<u64> {
        self.leg.attach_pool(layout)
    }

    /// 单槽字节（目标腿几何）。
    pub fn slot_bytes(&self) -> usize {
        self.leg.slot_bytes()
    }

    /// 设备侧实占字节。
    pub fn occupied_bytes(&self) -> usize {
        self.leg.occupied_bytes()
    }

    /// 可移动（缓存）字节。
    pub fn movable_bytes(&self) -> usize {
        self.leg.movable_bytes()
    }

    /// 设备 KV 空闲槽数。
    pub fn slots_free(&self) -> usize {
        self.leg.slots_free()
    }

    /// 设备 KV 总槽数。
    pub fn slots_total(&self) -> usize {
        self.leg.slots_total()
    }

    /// 槽引用数（0 = 槽空；>0 = 被链/表持有）。
    pub fn slot_refs(&self, slot: u32) -> u32 {
        self.leg.slot_refs(slot)
    }

    /// 内容树条目数。
    pub fn tree_len(&self) -> usize {
        self.leg.tree_len()
    }

    /// 内容树容量上限。
    pub fn tree_cap(&self) -> usize {
        self.leg.tree_cap()
    }

    /// KV 腿统计（host 块/命中/搬运）。
    pub fn kv_stats(&self) -> KvStats {
        self.leg.kv_stats()
    }

    /// 宿主层是否有该链内容。
    pub fn host_contains(&self, chain: ChainHash) -> bool {
        self.leg.host_contains(chain)
    }

    /// 只读宿主副本（checkpoint 的复制语义；不搬内容）。
    pub fn host_block(&self, chain: ChainHash) -> Option<&HostBlock> {
        self.leg.host_block(chain)
    }

    /// 宿主让位计数（执行面兜底真删的块数）。
    pub fn host_relief_blocks(&self) -> u64 {
        self.leg.host_relief_blocks()
    }

    /// 宿主层是否启用。
    pub fn host_enabled(&self) -> bool {
        self.leg.host_enabled()
    }

    /// 宿主层空闲块数。
    pub fn host_free_slots(&self) -> usize {
        self.leg.host_free_slots()
    }

    /// 宿主层总块数。
    pub fn host_total_slots(&self) -> usize {
        self.leg.host_total_slots()
    }

    /// 宿主层预算字节（0 = 关层）。
    pub fn host_budget_bytes(&self) -> usize {
        self.leg.host_budget_bytes()
    }

    /// 整链块身份（根 → tail）。
    pub fn chain_blocks(&self, tail: ChainHash) -> Vec<ChainHash> {
        self.leg.chain_blocks(tail)
    }

    /// 链当前载体（设备槽/宿主/磁盘）。
    ///
    /// **盘层也是 store 的事实**：树的 `slot = None` 只说明“不在设备”，
    /// 还要问 `DiskTier` 才能区分 Host / Disk（checkpoint 的复制语义下
    /// 可能两份都在，此时优先 Host —— 运行期可直读，盘只是备份）。
    pub fn carrier_of(&self, chain: ChainHash) -> StoreResult<Carrier> {
        let c = self.leg.carrier_of(chain)?;
        if matches!(c, Carrier::Host)
            && !self.leg.host_contains(chain)
            && self.kv_disk_contains(chain)
        {
            return Ok(Carrier::Disk);
        }
        Ok(c)
    }

    /// 链所属会话（0 = 无主）。
    pub fn session_of(&self, chain: ChainHash) -> u64 {
        self.leg.session_of(chain)
    }

    /// 链深度（块数）。
    pub fn depth_of(&self, chain: ChainHash) -> usize {
        self.leg.depth_of(chain)
    }

    /// 链尾内容身份（纯文本链）。
    pub fn tail_hash(&self, tokens: &[u32], full_blocks: usize) -> Option<ChainHash> {
        self.leg.tail_hash(tokens, full_blocks)
    }

    /// 链尾内容身份（带逐块多模态指纹）。
    pub fn tail_hash_tagged(
        &self,
        tokens: &[u32],
        full_blocks: usize,
        tags: &[crate::kv::BlockTag],
    ) -> Option<ChainHash> {
        self.leg.tail_hash_tagged(tokens, full_blocks, tags)
    }

    /// 槽引用 +1（前缀复用/回灌绑定）。
    pub fn incref(&mut self, slot: u32) -> StoreResult<()> {
        self.leg.incref(slot)
    }

    /// 槽引用 -1（表归还/逐出）。
    pub fn decref(&mut self, slot: u32) -> StoreResult<()> {
        self.leg.decref(slot)
    }

    /// 设置链的会话归属（0 = 无主）。
    pub fn set_chain_owner(&mut self, session: u64, chains: &[ChainHash]) {
        self.leg.set_owner(session, chains);
    }

    /// 登记一条由指定会话持有的链（带逐块多模态指纹）。
    pub fn register_chain_owned_tagged(
        &mut self,
        tokens: &[u32],
        table: &[u32],
        session: u64,
        tags: &[crate::kv::BlockTag],
    ) -> StoreResult<CacheUpdate> {
        self.leg.register_chain_owned_tagged(tokens, table, session, tags)
    }

    /// 从宿主层取走内容（溢写/下沉）。
    pub fn host_take(&mut self, chain: ChainHash) -> StoreResult<HostBlock> {
        self.leg.host_take(chain)
    }

    /// 把内容放回宿主层（溢写失败回滚/回填）。
    pub fn host_put(&mut self, chain: ChainHash, buf: HostBlock) -> StoreResult<()> {
        self.leg.host_put(chain, buf)
    }

    /// **TP peer 平面布局**（boot 一次；两 rank 各一套 PieceSpec）。
    pub fn attach_pool_peer(&mut self, layout: PoolLayout) -> StoreResult<u64> {
        self.leg.attach_pool_peer(layout)
    }

    /// 增量登记一条链（父链之后追加块；带逐块多模态指纹）。
    pub fn register_chain_inc_tagged(
        &mut self,
        parent0: ChainHash,
        blocks: &[(&[u32], u32)],
        tags: &[crate::kv::BlockTag],
    ) -> StoreResult<CacheUpdate> {
        self.leg.register_chain_inc_tagged(parent0, blocks, tags)
    }

    /// 制定回灌计划（宿主 → 设备；纯 host，零 I/O）。
    pub fn plan_onload(&mut self, chains: &[ChainHash]) -> StoreResult<InPlan> {
        crate::leg::plan_onload(&mut self.leg, chains)
    }

    /// 回灌记账（搬运成功后）。
    pub fn commit_onload(
        &mut self,
        chains: &[ChainHash],
        slots: &[u32],
        table: SlotTable,
    ) -> StoreResult<Vec<u32>> {
        crate::leg::commit_onload(&mut self.leg, chains, slots, table)
    }

    /// 回灌回滚（搬运失败后）。
    pub fn abort_onload(&mut self, chains: &[ChainHash], slots: &[u32]) {
        crate::leg::abort_onload(&mut self.leg, chains, slots);
    }

    /// 制定只读拷贝计划（设备 → 宿主；纯 host，零 I/O）。
    pub fn plan_read_blocks(&mut self, chains: &[ChainHash]) -> StoreResult<ReadPlan> {
        crate::leg::plan_read_blocks(&mut self.leg, chains)
    }

    /// 归还只读计划的宿主缓冲（swap 回执的槽台）。
    pub fn recycle_read_plan(&mut self, table: SlotTable) {
        crate::leg::recycle_read_plan(&mut self.leg, table);
    }

    /// 换出记账（搬运成功后）。
    pub fn commit_offload(
        &mut self,
        blocks: &[(ChainHash, u32)],
        table: SlotTable,
    ) -> StoreResult<usize> {
        crate::leg::commit_offload(&mut self.leg, blocks, table)
    }

    /// 换出回滚（搬运失败后）。
    pub fn abort_offload(&mut self, blocks: &[(ChainHash, u32)]) {
        crate::leg::abort_offload(&mut self.leg, blocks);
    }

    /// **命中查询**(按身份;纯 host、零副作用 —— 律⑥)。
    ///
    /// 与 [`Self::match_prefix`] 二选一使用(两条查询路径各自记一次命中账;
    /// 同一次查询不要两条都调,否则计数双算)。
    pub fn probe(&mut self, id: PrefixId) -> RecallHit {
        let kv_tier = self.carrier_of(id.chain).ok().map(|c| match c {
            Carrier::Device(_) => Tier::Gpu,
            Carrier::Host => Tier::Host,
            Carrier::Disk => Tier::Disk,
        });
        let snap_tier = self.state.locate(id.snap_key());
        let hit = RecallHit {
            kv_tier,
            snap_tier,
            blocks: kv_tier.map_or(0, |_| self.leg.chain_len(id.chain)),
        };
        self.leg.note_probe(hit.kv_tier.is_some());
        self.state.note_probe(hit.snap_tier.is_some());
        hit
    }

    /// **前缀复用面**（单源；engine 的 submit / step_need / scheduler 共用）。
    ///
    /// 返回 `(m, dev)`：
    /// - `m` = **可复用块数**（完全对齐保留末块重算；边界快照不在场则降一）；
    /// - `dev` = 其中**设备已驻留**的块数（换入时不占新槽）。
    ///
    /// host 命中的块虽“命中”，回灌时照样要设备槽 ⇒ 需求侧只能减 `dev`，
    /// 不能减 `m`。这条判据住在 store，engine 只消费结果。
    pub fn reuse_face(
        &self,
        hit: &ChainHit,
        recall: &RecallHit,
        prompt_len: usize,
        page: usize,
    ) -> (usize, usize) {
        let mut m = hit.matched_blocks;
        let tail = m;
        if m > 0 && m.saturating_mul(page) == prompt_len {
            m -= 1; // 完全对齐保留末块重算（非空 prefill）
        }
        if m > 0 && m == tail && recall.snap_tier.is_some() {
            let dev = hit.slots[..m].iter().filter(|s| s.is_some()).count();
            return (m, dev);
        }
        while m > 0 {
            if self
                .state
                .locate(SnapKey::at_boundary(hit.hashes[m - 1]))
                .is_some()
            {
                break;
            }
            m -= 1; // 无快照的边界不可复用（GDN 态对不上）
        }
        let dev = hit.slots[..m].iter().filter(|s| s.is_some()).count();
        (m, dev)
    }

    /// **按 token 命中查询**(engine L1 的第一跳):逐块内容匹配,回块链 + 绑定。
    ///
    /// 未绑定的格(`slot = None`)= 内容在宿主层,调用方须在**段边界**回灌
    /// (律③)—— 本方法自身不做设备动作。
    pub fn match_prefix(&mut self, tokens: &[u32]) -> StoreResult<(ChainHit, RecallHit)> {
        self.match_prefix_tagged(tokens, &[])
    }

    /// 同 [`Self::match_prefix`],但带**逐块多模态指纹**(图像内容;图像链的唯一
    /// 正确入口 —— 见 `KvStore::match_prefix_tagged`)。
    pub fn match_prefix_tagged(
        &mut self,
        tokens: &[u32],
        tags: &[crate::kv::BlockTag],
    ) -> StoreResult<(ChainHit, RecallHit)> {
        let m = self.leg.match_prefix_tagged(tokens, tags)?;
        let tail = m.hashes.last().copied();
        let kv_tier = tail.and_then(|t| self.carrier_of(t).ok()).map(|c| match c {
            Carrier::Device(_) => Tier::Gpu,
            Carrier::Host => Tier::Host,
            Carrier::Disk => Tier::Disk,
        });
        let snap_tier = tail.and_then(|t| self.state.locate(SnapKey::at_boundary(t)));
        let hit = RecallHit {
            kv_tier,
            snap_tier,
            blocks: m.matched_blocks,
        };
        Ok((m, hit))
    }

    /// **整枝塌缩开关**(boot 一次;`[kvmem].bulk_drop_when_tier_full`)。
    pub fn set_bulk_drop_when_tier_full(&mut self, on: bool) {
        self.bulk_drop_when_tier_full = on;
    }

    /// **整枝塌缩**(批量过期;见 [`KvStore::prune_branch`])。
    pub fn prune_branch(&mut self, tail: ChainHash, max_blocks: usize) -> StoreResult<usize> {
        self.leg.prune_branch(tail, max_blocks)
    }

    /// **废弃链身份**(前缀无感删除的"改写即失效";见 `KvStore::drop_chain`)。
    pub fn drop_chain(&mut self, chain: ChainHash) -> StoreResult<bool> {
        self.leg.drop_chain(chain)
    }

    /// **登记一条前缀链**(turn 收口 L3):缓存取一份绑定引用,容量驱逐同步丢宿主副本。
    ///
    /// 只做 KV 腿;快照腿由 [`StateStore::return_row`] 登记(两腿独立 —— 裁决②)。
    pub fn register_chain(&mut self, tokens: &[u32], table: &[u32]) -> StoreResult<CacheUpdate> {
        self.leg.register_chain(tokens, table)
    }

    /// 同 [`Self::register_chain`],但带逐块多模态指纹(图像链的唯一正确入口)。
    pub fn register_chain_tagged(
        &mut self,
        tokens: &[u32],
        table: &[u32],
        tags: &[crate::kv::BlockTag],
    ) -> StoreResult<CacheUpdate> {
        self.leg.register_chain_tagged(tokens, table, tags)
    }

    /// **块表增长 = 消费 grant**(P0-4)。无 grant / 额度不足 = 结构化预算错,
    /// 不再靠“池里刚好有空槽”静默放行。
    pub fn ensure_for_len(
        &mut self,
        grant: ReservationId,
        table: &mut Vec<u32>,
        target_len: usize,
    ) -> StoreResult<()> {
        let page = self.leg.block_tokens().max(1);
        let need = target_len.div_ceil(page);
        let delta = need.saturating_sub(table.len());
        if delta == 0 {
            return Ok(());
        }
        let bytes = delta
            .checked_mul(self.leg.slot_bytes())
            .ok_or_else(|| StoreError::Budget("ensure_for_len: 字节溢出".into()))?;
        self.reservations.consume(grant, delta, bytes)?;
        match self.leg.ensure_for_len(table, target_len) {
            Ok(()) => Ok(()),
            Err(e) => {
                // 分配失败 ⇒ 把刚消费的额度原样退回(账不能半截)
                if let Err(rb) = self.reservations.refund(grant, delta, bytes) {
                    return Err(StoreError::Invariant(format!(
                        "ensure_for_len 分配失败后回滚 grant 失败:{rb}(原错误:{e})"
                    )));
                }
                Err(e)
            }
        }
    }

    /// **测试/旧装配专用**:不消费 grant 的纯分配口。
    #[cfg(test)]
    pub fn ensure_for_len_unreserved(
        &mut self,
        table: &mut Vec<u32>,
        target_len: usize,
    ) -> StoreResult<()> {
        self.leg.ensure_for_len(table, target_len)
    }

    /// 归还会话块表。
    pub fn release_table(&mut self, table: &mut Vec<u32>) -> StoreResult<()> {
        self.leg.release_table(table)
    }

    /// **磁盘回填落宿主**(执行面兜底):宿主满 ⇒ 先逐最冷(保护本链)再放。
    ///
    /// 为什么不是裸 `host_put`:回填发生在 `plan_onload` **之前** —— 宿主满时
    /// 它同样会撞 `Exhausted`(与 offload 同一类洞)。保护面 = 本链(它正在被
    /// 回填,删掉就是自拆)。
    pub fn host_fill(&mut self, chain: ChainHash, buf: crate::host::HostBlock) -> StoreResult<()> {
        if self.leg.host_enabled() && self.leg.host_free_slots() == 0 {
            let protect = self.leg.chain_blocks(chain);
            let _ = self.leg.host_relieve(1, &protect)?;
        }
        self.leg.host_put(chain, buf)
    }

    /// **KV 链下沉计划**(engine 执行臂用;宿主满时**先逐最冷再重试**)。
    ///
    /// 与 `plan_capacity` 轮零同一条政策：最冷链尾向内削、保护面（本链）整条不动。
    /// 这口是**执行面兜底** —— 计划面已按宿主容量出票时通常一次成功；
    /// 盘写失败/账漂时不让 turn 硬崩（`ops.host_relief_blocks` 就是
    /// “计划曾经乐观/滞后”的指纹，engine 据此打点）。
    ///
    /// 宿主关层（0）= 旧行为：直接 `Exhausted`，不静默降级。
    pub fn plan_kv_offload(&mut self, tail: ChainHash) -> StoreResult<crate::leg::OutPlan> {
        match crate::leg::plan_offload(&mut self.leg, tail) {
            Ok(p) => Ok(p),
            Err(StoreError::Exhausted(why)) => {
                let Some(need) = self.leg.chain_dev_span(tail) else {
                    return Err(StoreError::Exhausted(why));
                };
                // 保护面 = 本链全部块（已在宿主的那些不能被“让位”删掉 ——
                // 否则搬完设备的块，链上缺一段，前缀永远拼不回来）
                let protect = self.leg.chain_blocks(tail);
                if self.leg.host_relieve(need, &protect)? == 0 {
                    return Err(StoreError::Exhausted(why));
                }
                crate::leg::plan_offload(&mut self.leg, tail)
            }
            Err(e) => Err(e),
        }
    }

    /// **逐块下沉计划**（选择层生产路径;宿主满时**先逐最冷再重试**）。
    ///
    /// 与 [`Self::plan_kv_offload`]（链级）**同一条政策**：计划面负责经济性
    /// （`plan_capacity` 按宿主容量出票），执行面负责**活性**（计划乐观/滞后时
    /// 不许让 turn 硬崩）。
    ///
    /// 为什么必须有这一口（2026-10-28 W2）：KVMem 选择层的下沉是**逐块**的
    /// （`exec/coverage.rs` 每个边界踢出若干块），旧实现直呼
    /// `plan_offload_blocks` ⇒ 宿主被**别的会话的旧链**填满时 `lend_slots` 直接
    /// `Exhausted` ⇒ 整发 500。链级票当时已有 `plan_kv_offload` 兜底，逐块票
    /// 漏了 —— 两条路必须对称。
    ///
    /// `protect` = 本批要沉的链自己（它们的内容即将落宿主，让位时不能反手删掉
    /// 其中已在宿主的那几块，否则链上缺一段、前缀永远拼不回来）。
    pub fn plan_kv_offload_blocks(
        &mut self,
        chains: &[ChainHash],
    ) -> StoreResult<crate::leg::BlockOutPlan> {
        match crate::leg::plan_offload_blocks(&mut self.leg, chains) {
            Ok(p) => Ok(p),
            Err(StoreError::Exhausted(why)) => {
                // 需求 = 本批块数（上界；让位是多删无害的冷块，少删才会再撞）
                if self.leg.host_relieve(chains.len(), chains)? == 0 {
                    return Err(StoreError::Exhausted(why));
                }
                crate::leg::plan_offload_blocks(&mut self.leg, chains)
            }
            Err(e) => Err(e),
        }
    }
}
