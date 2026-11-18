//! `Rel` 的状态腿（GDN 快照 / 行 / 宿主与设备搬运计划）。
//!
//! 纯账与描述：`StateStore` 的语义在这里投影成 engine 可执行的 plan/commit；
//! 没有设备 I/O。

use super::*;

impl Rel {
    // ---------------------------------------------------------------------
    // 状态腿语义面（快照/行；engine 不直接拿 StateStore）
    // ---------------------------------------------------------------------

    /// 快照当前所在层（None = 不在账）。
    pub fn state_locate(&self, key: SnapKey) -> Option<Tier> {
        self.state.locate(key)
    }

    /// 状态腿统计（行/快照/命中）。
    pub fn state_stats(&self) -> StateStats {
        self.state.stats()
    }

    /// 快照的会话归属与边界深度。
    pub fn snap_session_info(&self, key: SnapKey) -> Option<(u64, usize)> {
        self.state.snap_session(key)
    }

    /// 状态腿淘汰候选（按策略序；保护集不入序）。
    pub fn snap_evict_candidates(&self, protect: &[SnapKey]) -> Vec<EvictCandidate> {
        self.state.evict_candidates(protect)
    }

    /// 快照持久性档（Transient 不可下沉）。
    pub fn snap_durability(&self, key: SnapKey) -> Option<SnapDurability> {
        self.state.snap_durability(key)
    }

    /// 指定层快照条数与字节。
    pub fn snaps_in_tier(&self, tier: Tier) -> (usize, usize) {
        self.state.snaps_in_tier(tier)
    }

    /// 生成段（Transient）快照条数。
    pub fn snaps_transient(&self) -> usize {
        self.state.snaps_transient()
    }

    /// GDN 行当前 owner。
    pub fn row_owner(&self, row: usize) -> Option<crate::tier::Owner> {
        self.state.row_owner(row)
    }

    /// GDN 单行字节。
    pub fn row_bytes(&self) -> usize {
        self.state.row_bytes()
    }

    /// 登记一条快照的层账（测试/旧装配口；生产走 `register_snap_captured`）。
    pub fn register_snap_account(
        &mut self,
        key: SnapKey,
        bytes: usize,
        tier: Tier,
        durability: SnapDurability,
    ) {
        self.state.register_snap(key, bytes, tier, durability);
    }

    /// 登记一条带会话归属/边界深度的快照（L1 资产）。
    pub fn register_snap_owned(
        &mut self,
        key: SnapKey,
        bytes: usize,
        tier: Tier,
        durability: SnapDurability,
        session: u64,
        depth: usize,
    ) {
        self.state
            .register_snap_owned(key, bytes, tier, durability, session, depth);
    }

    /// 使用即触达（store 淘汰序真源）。
    pub fn touch_snap(&mut self, key: SnapKey) {
        self.state.touch(key);
    }

    /// 改快照层账（下沉/回填成功后调用）。
    pub fn set_snap_tier(&mut self, key: SnapKey, tier: Tier) -> StoreResult<()> {
        self.state.set_tier(key, tier)
    }

    /// 快照出账（不搬内容）。
    pub fn drop_snap_account(&mut self, key: SnapKey) -> StoreResult<usize> {
        self.state.drop_snap(key)
    }

    /// 归还 GDN 行（可同时挂一条快照）。
    pub fn return_state_row(&mut self, row: usize, snap: Option<SnapKey>) -> StoreResult<()> {
        self.state.return_row(row, snap)
    }

    /// 状态腿宿主层是否有该快照内容。
    pub fn snap_host_contains(&self, key: SnapKey) -> bool {
        self.snap_host.get(key).is_some()
    }

    /// 从宿主层取走一条快照（下沉/回填）。
    pub fn snap_host_take(&mut self, key: SnapKey) -> StoreResult<HostBlock> {
        self.snap_host.take(key)
    }

    /// 把快照内容放回宿主层（下沉失败回滚/回填）。
    ///
    /// **共享池**：宿主满时先逐本腿最冷快照，再让 KV 腿宿主副本让位；
    /// 仍装不下 = 响亮拒（调用方按“快照不可落宿主”处理，不静默丢）。
    pub fn snap_host_put(&mut self, key: SnapKey, buf: HostBlock) -> StoreResult<()> {
        if self.snap_host_free_slots() == 0 {
            let need = self.snap_file_bytes().max(1);
            let _ = self.evict_snap_host_bytes(need, &[key])?;
            if self.snap_host_free_slots() == 0 {
                let slots = need.div_ceil(self.leg.host_slot_bytes().max(1));
                let _ = self.leg.host_relieve(slots, &[])?;
                self.leg.host_drop_free();
            }
        }
        self.snap_host.put(key, buf)
    }

    /// **按字节逐出状态腿宿主副本**（共享池让位口；纯账，零 I/O）。
    ///
    /// 出序与 `plan_snap_offload` 同一份（`Transient → History → Tail` + 策略）。
    /// 逐出后 `drop_free()` 把空转缓冲真正还给共享总账；返回释放字节。
    pub(crate) fn evict_snap_host_bytes(
        &mut self,
        need_bytes: usize,
        protect: &[SnapKey],
    ) -> StoreResult<usize> {
        let mut freed = 0usize;
        while freed < need_bytes {
            let cands: Vec<EvictCandidate> = self
                .state
                .evict_candidates(protect)
                .into_iter()
                .filter(|c| c.tier == Tier::Host && c.snap.is_some())
                .collect();
            if cands.is_empty() {
                break;
            }
            let victim = crate::hooks::order_snap_victims(self.evict.as_ref(), cands, &self.state)
                .into_iter()
                .filter_map(|c| c.snap)
                .next();
            let Some(v) = victim else { break };
            let bytes = self
                .snap_host
                .get(v)
                .map(|b| b.len())
                .or_else(|| self.state.snap_bytes(v))
                .unwrap_or(0);
            self.snap_host.evict(v)?;
            self.state.drop_snap(v)?;
            freed = freed.saturating_add(bytes);
        }
        self.snap_host.drop_free();
        Ok(freed)
    }

    // ───────────────── 状态腿搬运口(§25.2;与 KV 腿同形:store 出计划与账) ─────────────────

    /// 状态腿的片(逐 (层,quad);`dev_off = 槽 × 行宽`)。
    fn snap_pieces(&self, slot: u32, plane: usize) -> StoreResult<Vec<SwapPiece>> {
        let l = if plane == 0 {
            self.snap_layout.as_ref()
        } else {
            self.snap_layout_peer.as_ref()
        };
        let layout = l.ok_or_else(|| {
            StoreError::Invariant(format!(
                "状态腿布局(平面 {plane})未注册(engine 应在 boot 调 attach_snap_layout/…_peer)"
            ))
        })?;
        Ok(layout
            .pieces
            .iter()
            .map(|p| SwapPiece {
                block: p.buf.id,
                dev_off: slot as usize * p.stride_bytes,
                conn_off: 0, // 单条快照的宿主槽 = 连续一条(片内偏移在下面补)
                len: p.stride_bytes,
                slot: ConnSlot(0),
            })
            .collect())
    }

    /// **逐出一条状态腿宿主快照**（共享池让位/复用宿主块用；纯账）。
    ///
    /// 出序 = `order_snap_victims`（Transient → History → Tail + 策略）；
    /// 返回被逐键（None = 没有可逐候选）。
    fn evict_one_snap_host(&mut self, protect: &[SnapKey]) -> StoreResult<Option<SnapKey>> {
        let cands: Vec<EvictCandidate> = self
            .state
            .evict_candidates(protect)
            .into_iter()
            .filter(|c| c.tier == Tier::Host && c.snap.is_some())
            .collect();
        if cands.is_empty() {
            return Ok(None);
        }
        let victim = crate::hooks::order_snap_victims(self.evict.as_ref(), cands, &self.state)
            .into_iter()
            .filter_map(|c| c.snap)
            .next();
        let Some(v) = victim else { return Ok(None) };
        self.snap_host.evict(v)?;
        self.state.drop_snap(v)?;
        Ok(Some(v))
    }

    /// **快照下沉计划**(设备槽 `slot` 上的快照 → 宿主)。
    /// `Busy` = 快照不在账/宿主层已满(零副作用,调用方跳过)。
    ///
    /// **不重要就不搬**(2026-10-19):生成段边界快照逐出即丢 —— 下沉一条
    /// 159 MiB ≈ **50 ms**(实测,页锁定外的 PCIe 速),而生成段每 32 tok 就出
    /// 一条 ⇒ 全下沉 = 每 token 数 MiB 级 PCIe 流量,不可承受。拒语即事实
    /// (调用方回落已有"沉不下⇒丢"路;见 [`SnapDurability`])。
    pub fn plan_snap_offload(
        &mut self,
        key: SnapKey,
        slot: u32,
    ) -> StoreResult<(Vec<Vec<SwapPiece>>, SlotTable)> {
        let bytes = self
            .state
            .snap_bytes(key)
            .ok_or_else(|| StoreError::Unknown(format!("快照 {key:?} 不在账")))?;
        if self.state.snap_durability(key) != Some(SnapDurability::Durable) {
            return Err(StoreError::Invariant(format!(
                "快照 {key:?} 不可下沉(生成段边界:链一推进就被取代 ⇒ 逐出即丢)"
            )));
        }
        if bytes != self.snap_file_bytes() {
            return Err(StoreError::Invariant(format!(
                "快照字节不符:账 {bytes} ≠ 布局 {}(单 rank {} × world {})",
                self.snap_file_bytes(),
                self.snap_bytes,
                self.world
            )));
        }
        // **宿主满 ⇒ 先逐最冷**(2026-10-19 政策补口;G3 定档轮实测立案)。
        //
        // 旧行为:`acquire()` 预算越界直接 `Exhausted` ⇒ 调用方回落"丢**新**
        // 快照" ⇒ **宿主梯从不逐出**。实测后果:开机做过几轮会话后,3 GB 宿主
        // 梯被旧链边界占满 ⇒ 14k prompt 的 28 个边界里 **18 个被丢** ⇒ 后来
        // 的"部分前缀"复用面从 ~97% 塔到 29%(7k 共享只复用 2k = 白重算 5k
        // token ≈ 5 s/请求)。
        //
        // 政策 = **同一份快照出序**(`order_snap_victims`):先按 `SnapRole`
        // 角色(`Transient → Extra → 老 Binary → Dense`),同档再交给
        // `EvictPolicy`。只比 `Tier::Host` 的候选,排序后取第一个。
        //
        // 不能直接 `self.evict.order`:那会绕过角色,宿主满时可能先丢
        // 一条 Tail 尾锚点,却留下更冷的 History。代价说明:先逐后搬 ⇒ 若接下来
        // 的 DMA 失败(`abort_snap_offload`),被逐条目已丢(比"丢新条目"代价小
        // —— 它本来就在淘汰序最前)。
        // **TTFT 保护（2026-11-18 现场）**：宿主块是 pageable 大块，池接近
        // 容量时新分配/首次触碰会偶发数秒级停顿；`prefill.bnd` 曾实测
        // `srv.swap.out.put` 5.8s ⇒ 首 token 门超时。若本腿已接近 3/4 容量
        // 且空转表为空，先逐一条最冷宿主快照，**复用它的宿主块**，避免在
        // prefill 关键路径上分配新大块。旧快照按 LRU/角色出序，代价可见。
        let reuse_floor = self.snap_host.budget_bytes().saturating_mul(3) / 4;
        if self.snap_host.free_bytes() == 0 && self.snap_host.used_bytes() >= reuse_floor {
            if let Some(v) = self.evict_one_snap_host(&[key])? {
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.snap.host_reuse_evict", 1, file!(), line!());
                });
                owl_shared::log_debug!(
                    "store",
                    "宿主快照近满 ⇒ 逐 {v:?} 复用宿主块(避免 prefill 路径新分配大块)"
                );
            }
        }
        let last = self.snap_host.used_bytes();
        let mut guard = 0usize;
        let buf = loop {
            match self.snap_host.acquire() {
                Ok(b) => break b,
                Err(StoreError::Exhausted(why)) => {
                    let cands: Vec<EvictCandidate> = self
                        .state
                        .evict_candidates(&[])
                        .into_iter()
                        .filter(|c| c.tier == Tier::Host)
                        .collect();
                    if cands.is_empty() {
                        // **共享池**：快照自己无可逐时，让 KV 腿的宿主副本让位。
                        // 先逐 KV 最冷（内容丢失；若盘层可用，计划面通常已先搬盘），
                        // 再把 KV 空转缓冲真正 drop 掉 —— 空转缓冲仍占共享预算，
                        // 不 drop 就只是把账从驻留挪到 free，腾不出给快照。
                        let need_slots = self
                            .snap_file_bytes()
                            .div_ceil(self.leg.host_slot_bytes().max(1));
                        let dropped = self.leg.host_relieve(need_slots, &[])?;
                        self.leg.host_drop_free();
                        // KV 空转缓冲也可能占着共享预算；drop 后仍有位就继续。
                        if dropped == 0 && self.snap_host_free_slots() == 0 {
                            // 宿主账与层账不一致(账说满而层无可逐)—— 不许静默
                            return Err(StoreError::Invariant(format!(
                                "宿主梯满且无 Host 层候选可逐({why};层账={last}B)"
                            )));
                        }
                        continue;
                    }
                    let v =
                        crate::hooks::order_snap_victims(self.evict.as_ref(), cands, &self.state)
                            .into_iter()
                            .filter_map(|c| c.snap)
                            .next()
                            .ok_or_else(|| StoreError::Invariant("驱逐候选缺 snap 键".into()))?;
                    let freed = self.snap_host.evict(v)?;
                    self.state.drop_snap(v)?;
                    let _ = freed;
                    guard += 1;
                    if guard > 16 {
                        return Err(StoreError::Invariant(
                            "宿主梯逐出循环超限(16):预算与释放字节不一致".into(),
                        ));
                    }
                }
                Err(e) => return Err(e),
            }
        };
        let all = (0..self.world.max(1))
            .map(|plane| {
                let mut pieces = self.snap_pieces(slot, plane)?;
                // 片在宿主槽内**首尾相接**(每条快照 = 一个槽 × world 平面)
                // 平面基址 = 该 rank 的快照字节(漏了 = peer 覆盖 rank0 的内容)
                let mut off = plane * self.snap_bytes;
                for p in &mut pieces {
                    p.conn_off = off;
                    off += p.len;
                }
                Ok(pieces)
            })
            .collect::<StoreResult<Vec<_>>>()?;
        Ok((all, SlotTable::one(buf)))
    }

    /// **快照下沉记账**(搬运成功后;纯 host):内容落宿主 + 换层。
    pub fn commit_snap_offload(&mut self, key: SnapKey, table: SlotTable) -> StoreResult<()> {
        let buf = table
            .into_slots()
            .into_iter()
            .next()
            .ok_or_else(|| StoreError::Invariant("快照槽台为空".into()))?;
        // **诊断面**(2026-10-28):同键重复落宿主 = 宿主副本留尸(幽灵),
        // 会把 put 变成不变量违反 ⇒ 直接 500。这里先响亮报出世面,再照常拒。
        if self.snap_host.get(key).is_some() {
            owl_shared::log_warn!(
                "snap-trace",
                "下沉落宿主撞幽灵 {key:?}:宿主已有该键(状态账={:?},重复落地累计={}) ⇒ 拒落",
                self.state.locate(key),
                self.snap_host.duplicate_landings()
            );
        }
        self.snap_host.put(key, buf)?;
        self.state.set_tier(key, Tier::Host)?;
        owl_shared::log_debug!("snap-trace", "offload→Host {key:?}");
        Ok(())
    }

    /// **快照键重挂**(前缀无感删除;见 [`StateStore::rekey_snap`])。
    ///
    /// 按层分派:**设备**只动账(槽由引擎按 key 重标);**宿主**连内容一起
    /// 重挂(`HostTier::rekey`,零搬运);**盘**不由本方法做 —— 盘内容要真
    /// `rename`,归 engine 的异步 I/O(`plan_rekey_snap_disk` →
    /// `disk_io::rename_block` → `commit_rekey_snap_disk`),本方法对盘层返 `false`。
    ///
    /// 拒绝(全列,均**不动任何东西**):
    /// - `old` 不在账 ⇒ `false`;
    /// - `new` 已在账 ⇒ `false`(不许覆盖);
    /// - 宿主层账与驻留表不平(账说在宿主、表里没有)⇒ `false`(不许猜)。
    pub fn rekey_snap(&mut self, old: SnapKey, new: SnapKey) -> bool {
        if self.state.locate(new).is_some() {
            return false;
        }
        let r = match self.state.locate(old) {
            Some(Tier::Gpu) => self.state.rekey_snap(old, new),
            Some(Tier::Host) => {
                if self.snap_host.rekey(old, new).is_err() {
                    return false;
                }
                // 前置已查 new 不在账 ⇒ 这里失败只可能是并发错账(单线程不会)
                self.state.rekey_snap(old, new)
            }
            Some(Tier::Disk) | None => false,
        };
        owl_shared::log_debug!(
            "snap-trace",
            "rekey {old:?} → {new:?} tier={:?} ok={r}",
            self.state.locate(new).or_else(|| self.state.locate(old))
        );
        r
    }

    /// **盘层快照重挂:计划**(`(旧路径, 新路径)`;真 I/O 归 engine 阻塞池)。
    pub fn plan_rekey_snap_disk(
        &self,
        old: SnapKey,
        new: SnapKey,
    ) -> StoreResult<(std::path::PathBuf, std::path::PathBuf)> {
        if self.state.locate(new).is_some() {
            return Err(StoreError::Invariant(format!("盘层重挂:{new:?} 已在账")));
        }
        let Some(d) = self.snap_disk.as_ref() else {
            return Err(StoreError::Unknown("盘层重挂:未接盘层".into()));
        };
        d.plan_rekey(old, new)
    }

    /// **盘层快照重挂:落账**(rename 成功后):盘账 + 状态账一起换键。
    pub fn commit_rekey_snap_disk(&mut self, old: SnapKey, new: SnapKey) -> StoreResult<()> {
        let Some(d) = self.snap_disk.as_mut() else {
            return Err(StoreError::Unknown("盘层重挂落账:未接盘层".into()));
        };
        d.commit_rekey(old, new)?;
        if !self.state.rekey_snap(old, new) {
            return Err(StoreError::Invariant(format!(
                "盘层重挂落账:状态账拒了 {old:?} → {new:?}(账不平)"
            )));
        }
        Ok(())
    }

    /// **盘层快照定向丢弃:计划**(路径;不在盘层 = `None`)。
    ///
    /// 删除后整条旧链都不可达 —— 盘上那份"旧边界的 GDN 态"同样成了孤儿,
    /// 不丢就会把盘预算永远压在死身份上。
    pub fn plan_drop_snap_disk(&self, key: SnapKey) -> Option<std::path::PathBuf> {
        self.snap_disk
            .as_ref()?
            .path_of(key)
            .map(|p| p.to_path_buf())
    }

    /// **盘层快照定向丢弃:落账**(文件删掉或本来就不在)。
    pub fn commit_drop_snap_disk(&mut self, key: SnapKey) -> StoreResult<()> {
        if let Some(d) = self.snap_disk.as_mut() {
            d.commit_evict(&[key]);
        }
        self.state.drop_snap(key).map(|_| ())
    }

    /// **快照下沉回滚**(搬运失败):宿主缓冲随失败的 connector 释放,层不动。
    pub fn abort_snap_offload(&mut self, _key: SnapKey) {}

    /// **快照回灌计划**(宿主 → 设备槽 `slot`)。
    pub fn plan_snap_onload(
        &mut self,
        key: SnapKey,
        slot: u32,
    ) -> StoreResult<(Vec<Vec<SwapPiece>>, SlotTable)> {
        owl_shared::log_debug!(
            "snap-trace",
            "onload←Host {key:?} slot={slot} 宿主在场={}",
            self.snap_host.get(key).is_some()
        );
        let buf = self.snap_host.take(key)?;
        let all = (0..self.world.max(1))
            .map(|plane| {
                let mut pieces = self.snap_pieces(slot, plane)?;
                let mut off = plane * self.snap_bytes;
                for p in &mut pieces {
                    p.conn_off = off;
                    off += p.len;
                }
                Ok(pieces)
            })
            .collect::<StoreResult<Vec<_>>>()?;
        Ok((all, SlotTable::one(buf)))
    }

    /// **快照回灌记账**(搬运成功后):缓冲回池 + 层回设备。
    pub fn commit_snap_onload(&mut self, key: SnapKey, table: SlotTable) -> StoreResult<()> {
        self.snap_host.recycle_many(table.into_slots());
        self.state.set_tier(key, Tier::Gpu)?;
        Ok(())
    }

    /// **快照回灌回滚**(搬运失败):台子已随失败的 connector 移出(内容丢)
    /// ⇒ 条目一并删(不留假账;与 KV 腿 `abort_offload` 同律)。
    pub fn abort_snap_onload(&mut self, key: SnapKey) {
        let _ = self.state.drop_snap(key);
    }

    /// **整体丢弃一条快照资产**(准入替换路径;2026-11-12)。
    ///
    /// 与 `drop_snap_host` 的区别:本函数同时清**宿主 + 盘 + 状态账**,
    /// 并返回丢弃前所在层 —— 设备层的槽必须由引擎的池清(store 不持句柄)。
    /// 不在账 = 响亮拒(账/槽不平不允许静默)。
    pub fn drop_snap_asset(&mut self, key: SnapKey) -> StoreResult<(crate::tier::Tier, usize)> {
        let tier = self
            .state
            .locate(key)
            .ok_or_else(|| StoreError::Unknown(format!("快照 {key:?}")))?;
        if self.snap_host.get(key).is_some() {
            self.snap_host.evict(key)?;
        }
        if let Some(d) = self.snap_disk.as_mut() {
            if d.contains(key) {
                d.commit_evict(&[key]);
            }
        }
        let bytes = self.state.drop_snap(key)?;
        owl_shared::log_debug!(
            "snap-trace",
            "drop-asset {key:?} tier={tier:?} bytes={bytes}"
        );
        Ok((tier, bytes))
    }

    /// 丢弃一条快照的宿主副本(内容整体淘汰时;不在宿主 = 幂等)。
    pub fn drop_snap_host(&mut self, key: SnapKey) -> StoreResult<usize> {
        let had = self.snap_host.get(key).is_some();
        let r = match self.snap_host.evict(key) {
            Ok(n) => Ok(n),
            Err(StoreError::Unknown(_)) => Ok(0),
            Err(e) => Err(e),
        };
        owl_shared::log_debug!("snap-trace", "drop-host {key:?} 在场={had}");
        r
    }
    /// 借一条 GDN 行（turn 生命周期）。
    pub fn borrow_row(&mut self, owner: crate::tier::Owner) -> StoreResult<usize> {
        self.state.borrow_row(owner)
    }
    /// **状态腿的出序(判据唯一的一面;§30.6-②)** —— "设备层快照谁该先走"。
    ///
    /// 与 [`Self::plan_need`] 的 `plan_capacity` **共用同一份排序**
    /// (事实档:Transient 先;档内 `EvictPolicy`)。engine 的池只负责
    /// key→槽的物理映射 —— 不再在池里另搞一套 `rank + LRU`。
    ///
    /// `protect` = 点名保住的快照键(绝不入序)。
    pub fn snap_victim_order(&self, protect: &[SnapKey]) -> Vec<SnapKey> {
        let cands: Vec<EvictCandidate> = self
            .state
            .evict_candidates(protect)
            .into_iter()
            .filter(|c| c.tier == Tier::Gpu)
            .collect();
        crate::hooks::order_snap_victims(self.evict.as_ref(), cands, &self.state)
            .into_iter()
            .filter_map(|c| c.snap)
            .collect()
    }

    /// **注册状态腿的设备布局**(boot 一次;§25.2):快照 = 一种"块"
    /// (槽 × 片),片 = 每个 (层,quad) 一行宽缓冲。
    pub fn attach_snap_layout(&mut self, layout: crate::leg::PoolLayout) -> StoreResult<u64> {
        let bytes = layout.slot_bytes();
        if bytes == 0 {
            return Err(StoreError::Invariant("状态腿布局单块字节为 0".into()));
        }
        // 宿主层的块粒度 = 一条快照 × world 平面(建 host 时未知,故此处重建:预算不变)
        let budget = self.snap_host.budget_bytes();
        self.snap_host = match self.snap_host.shared_budget().cloned() {
            Some(shared) => HostTier::new_shared(bytes.saturating_mul(self.world), shared)?,
            None => HostTier::new(bytes.saturating_mul(self.world), budget)?,
        };
        let epoch = layout.epoch;
        self.snap_layout = Some(layout);
        self.snap_bytes = bytes;
        Ok(epoch)
    }

    /// **注册 peer 快照布局**(TP;`world > 1` 必调;两 rank 的快照槽同形)。
    pub fn attach_snap_layout_peer(&mut self, layout: crate::leg::PoolLayout) -> StoreResult<u64> {
        let got = layout.slot_bytes();
        if got != self.snap_bytes {
            return Err(StoreError::Invariant(format!(
                "peer 快照布局单块字节不符:布局 {got} ≠ 本 rank {}",
                self.snap_bytes
            )));
        }
        let epoch = layout.epoch;
        self.snap_layout_peer = Some(layout);
        Ok(epoch)
    }

    /// 一条快照的字节(布局未注册 = 0)。
    pub fn snap_bytes(&self) -> usize {
        self.snap_bytes
    }

    /// **快照新拍进入设备层的落账口**(单源;引擎在 `capture` 后只调这一个)。
    ///
    /// 为何不能直接 `state.register_snap_owned(.., Tier::Gpu)`:
    /// `SnapKey` 是**块边界内容身份** ⇒ 同一条 token 链重算会得到**同一个键**。
    /// 若该键此前已被下沉到宿主/盘(宿主还留着副本),直接改层账 Gpu 就会
    /// **宿主副本留尸** —— 之后任何 Host 下沉走到 `snap_host.put` 都撞
    /// 「块 … 已在宿主层」⇒ 整个 turn 500(2026-10-28 实测:同 prompt 第 3 次
    /// 必现,truncate 尾部重算时重拍同名边界快照)。
    ///
    /// 判据与 KV 腿的 `drop_host_chains` 同律:**同身份重绑到设备新槽 ⇒
    /// 旧的宿主/盘副本是幽灵,必须当场丢**。
    ///
    /// 返回被丢掉的幽灵数(宿主 + 盘各至多 1;非零 = 上游重算了同一身份,
    /// 属正常事件但要看得见)。
    pub fn register_snap_captured(
        &mut self,
        key: SnapKey,
        bytes: usize,
        durability: crate::state::SnapDurability,
        session: u64,
        depth: usize,
        role: crate::state::SnapRole,
    ) -> StoreResult<usize> {
        let mut ghosts = 0usize;
        if self.snap_host.get(key).is_some() {
            self.snap_host.evict(key)?;
            ghosts += 1;
        }
        if let Some(d) = self.snap_disk.as_mut() {
            if d.contains(key) {
                d.commit_evict(&[key]);
                ghosts += 1;
            }
        }
        self.state.register_snap_owned_role(
            key,
            bytes,
            Tier::Gpu,
            durability,
            session,
            depth,
            role,
        );
        if ghosts > 0 {
            owl_shared::log_debug!(
                "snap-trace",
                "重拍同名边界快照 {key:?} ⇒ 丢掉旧副本 {ghosts} 份(宿主/盘)"
            );
        }
        Ok(ghosts)
    }

    /// **快照准入(纯账;不搬内容)** —— 引擎在 `snap_ensure_slot` 之前调用。
    ///
    /// `Skip` ⇒ 调用方必须**不拷贝**;`Replace(victim)` ⇒ 调用方先释放该
    /// victim 的设备槽/宿主/盘副本,再照常 capture。
    pub fn admit_snap(
        &self,
        session: u64,
        key: SnapKey,
        durability: crate::state::SnapDurability,
        policy: crate::state::SnapAdmitPolicy,
        protect: &[SnapKey],
    ) -> crate::state::SnapAdmit {
        self.state
            .admit_snap(session, key, durability, policy, protect)
    }

    /// **只重分类快照条目的角色**(不搬内容)。
    /// 返回 `false` = 键已不在账(正常逐出)。
    pub fn reclassify_snap(&mut self, key: SnapKey, role: crate::state::SnapRole) -> bool {
        self.state.reclassify_snap(key, role)
    }

    /// **会话关闭:释放该会话的快照归属**(不丢内容;2026-11-12 二刀)。
    pub fn detach_session_snaps(&mut self, session: u64) -> usize {
        self.state.detach_session(session)
    }

    /// 状态腿宿主层预算(字节;0 = 关层)。
    pub fn snap_host_budget(&self) -> usize {
        self.snap_host.budget_bytes()
    }

    /// 状态腿宿主层驻留字节。
    pub fn snap_host_used(&self) -> usize {
        self.snap_host.used_bytes()
    }

    /// 状态腿宿主块**大页提示未生效的块数**（0 = 全生效）。
    /// >0 不是错误：只是冷写回到 4 KiB 缺页速度（159 MiB 约 80ms vs 12ms）——
    /// > 引擎在下沉点打点上报（不许静默）。
    pub fn snap_host_hint_failures(&self) -> usize {
        self.snap_host.hint_failures()
    }

    /// 状态腿宿主层**累计逐出的字节**(政策面:宿主满时逐最冷条目)。
    /// 引擎在下沉点读差分上报(store 不引日志依赖)。
    pub fn snap_host_evicted_bytes(&self) -> u64 {
        self.snap_host.evicted_bytes()
    }

    /// 状态腿宿主层**还能装几条快照**。
    pub fn snap_host_free_slots(&self) -> usize {
        self.snap_host.free_slots()
    }

    /// 状态腿宿主层总槽数(= 预算 / 单条快照字节;0 = 关层)。
    pub fn snap_host_total_slots(&self) -> usize {
        self.snap_host
            .budget_bytes()
            .checked_div(self.snap_file_bytes().max(1))
            .unwrap_or(0)
    }

    ///
    /// **一条快照的总字节 = `snap_bytes × world`**(2026-11-01 P1 收尾)。
    ///
    /// 三处口径必须一致,TP 下差一个 `world` 就会"账 159 MB ≠ 布局 79 MB":
    /// - 盘上一个文件(内容腿)= 全平面;
    /// - 宿主层一块(`attach_snap_layout` 按 `bytes × world` 建);
    /// - `StateStore` 记的 `snap_bytes(key)` = 全平面。 单 rank 的 `snap_bytes` 只用于**平面内**偏移(swap piece 的 stride)。
    pub fn snap_file_bytes(&self) -> usize {
        self.snap_bytes.saturating_mul(self.world)
    }
}
