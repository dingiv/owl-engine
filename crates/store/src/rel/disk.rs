//! `Rel` 的磁盘腿（KV / 快照的 attach、adopt、spill/fill/evict 计划与账）。
//!
//! 数据搬运仍由 engine 按这里的 `plan_*` 执行；本模块只认盘层账与描述类型。

use super::*;

impl Rel {
    /// KV 盘层是否有该链内容。
    pub fn kv_disk_contains(&self, key: ChainHash) -> bool {
        self.disk.as_ref().is_some_and(|d| d.contains(key))
    }

    /// KV 盘层单块字节（未接线 = `None`）。
    pub fn kv_disk_block_bytes(&self) -> Option<usize> {
        self.disk.as_ref().map(|d| d.block_bytes())
    }

    /// **规划一次 KV checkpoint**：由 `tokens + tags` 重算内容链，跳过盘上
    /// 已有前缀，按 `quota_bytes` 小步取连续段，按真实 carrier 分成
    /// `dev`（D2H 复制）/ `host`（clone 字节）。
    ///
    /// - 盘上-only 块 `slot=None` 但**不在 host**：显式走 `carrier_of` 判成
    ///   `Disk`，绝不再落进 host 分支（`KvStore::locate` 的二值口径在这里
    ///   被门面校正）；
    /// - 遇到洞（既不在设备也不在宿主）/盘上已有块/配额到顶即停，**不跳块**；
    /// - 返回 `None` = 没有可写内容（全在盘、配额 0、或前缀断在根）。
    pub fn plan_checkpoint_kv(
        &self,
        tokens: &[u32],
        tags: &[crate::kv::BlockTag],
        quota_bytes: usize,
    ) -> StoreResult<Option<KvCheckpointPlan>> {
        if quota_bytes == 0 || !self.disk_enabled() {
            return Ok(None);
        }
        let Some(block_bytes) = self.kv_disk_block_bytes() else {
            return Ok(None);
        };
        if block_bytes == 0 {
            return Ok(None);
        }
        let page = self.leg.block_tokens().max(1);
        let full = tokens.len() / page;
        if full == 0 {
            return Ok(None);
        }
        // 身份来源 = tokens（必须与 ChainIndex 同源；不能走树祖先链）。
        let mut parent = ChainHash::new(0);
        let mut chain = Vec::with_capacity(full);
        for (i, block) in tokens.chunks(page).take(full).enumerate() {
            parent = crate::kv::hash_block(parent, block, tags.get(i).copied().unwrap_or(0));
            chain.push(parent);
        }
        let mut start = 0usize;
        while start < chain.len() && self.kv_disk_contains(chain[start]) {
            start += 1;
        }
        if start >= chain.len() {
            return Ok(None); // 整链已在盘上
        }
        let max_blocks = (quota_bytes / block_bytes).max(1);
        let mut dev = Vec::new();
        let mut host = Vec::new();
        for h in chain.iter().skip(start) {
            if dev.len() + host.len() >= max_blocks {
                break;
            }
            if self.kv_disk_contains(*h) {
                break; // 连续前缀断在盘上已有块（理论上 start 已越过）
            }
            match self.carrier_of(*h) {
                Ok(Carrier::Device(_)) => dev.push(*h),
                Ok(Carrier::Host) => {
                    if self.host_block(*h).is_none() {
                        break; // 账说宿主、内容不在 ⇒ 不准跳洞
                    }
                    host.push(*h);
                }
                Ok(Carrier::Disk) | Err(_) => break,
            }
        }
        let wrote = dev.len() + host.len();
        if wrote == 0 {
            return Ok(None);
        }
        Ok(Some(KvCheckpointPlan {
            chain: chain[..start + wrote].to_vec(),
            dev,
            host,
            block_bytes,
            already_on_disk: start,
        }))
    }

    /// KV 盘层空闲槽数（未接线 = 0）。
    pub fn kv_disk_free_slots(&self) -> usize {
        self.disk.as_ref().map_or(0, |d| d.free_slots())
    }

    /// KV 盘层配额字节（0 = 未接线）。
    pub fn kv_disk_budget_bytes(&self) -> usize {
        self.disk.as_ref().map_or(0, |d| d.budget_bytes())
    }

    /// KV 盘层根目录。
    pub fn kv_disk_root(&self) -> Option<&std::path::Path> {
        self.disk.as_ref().map(|d| d.root())
    }

    /// KV 盘层内容路径。
    pub fn kv_disk_path_of(&self, key: ChainHash) -> Option<&std::path::Path> {
        self.disk.as_ref().and_then(|d| d.path_of(key))
    }

    /// KV 盘层驻留条数。
    pub fn kv_disk_resident_count(&self) -> usize {
        self.disk.as_ref().map_or(0, |d| d.resident_count())
    }

    /// KV 盘层驻留键。
    pub fn kv_disk_resident_keys(&self) -> Vec<ChainHash> {
        self.disk.as_ref().map(|d| d.resident_keys()).unwrap_or_default()
    }

    /// 制定 KV 溢写计划（未接线 = 结构化错）。
    pub fn kv_disk_plan_spill(&self, key: ChainHash) -> StoreResult<SpillPlanOf<ChainHash>> {
        self.disk
            .as_ref()
            .ok_or_else(|| StoreError::Invariant("KV 盘层未接线".into()))?
            .plan_spill(key)
    }

    /// KV 溢写记账。
    pub fn kv_disk_commit_spill(&mut self, plan: &SpillPlanOf<ChainHash>) -> StoreResult<()> {
        self.disk
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("KV 盘层未接线".into()))?
            .commit_spill(plan)
    }

    /// 制定 KV 回填计划。
    pub fn kv_disk_plan_fill(&self, key: ChainHash) -> StoreResult<FillPlanOf<ChainHash>> {
        self.disk
            .as_ref()
            .ok_or_else(|| StoreError::Invariant("KV 盘层未接线".into()))?
            .plan_fill(key)
    }

    /// KV 回填记账。
    pub fn kv_disk_commit_fill(&mut self, key: ChainHash) -> StoreResult<()> {
        self.disk
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("KV 盘层未接线".into()))?
            .commit_fill(key)
    }

    /// KV 盘层逐出计划（未接线 = 空计划）。
    pub fn kv_disk_plan_evict(
        &self,
        need_bytes: usize,
        protect: &[ChainHash],
    ) -> DiskEvictPlanOf<ChainHash> {
        self.disk
            .as_ref()
            .map(|d| d.plan_evict(need_bytes, protect))
            .unwrap_or_default()
    }

    /// **session 感知的 KV 盘层逐出计划**：先避开活跃 session，不够才动活跃面。
    ///
    /// 盘内容是全局共享缓存，但淘汰应先动“冷会话”的历史；活跃 session 的
    /// 盘块只有在整个盘都腾不出时才允许作为最后来源。
    pub fn kv_disk_plan_evict_sessions(
        &self,
        need_bytes: usize,
        protect: &[ChainHash],
        active_sessions: &[u64],
    ) -> DiskEvictPlanOf<ChainHash> {
        let Some(d) = self.disk.as_ref() else {
            return DiskEvictPlanOf::default();
        };
        let is_active = |k: &ChainHash| active_sessions.contains(&self.leg.session_of(*k));
        let mut plan = d.plan_evict_where(need_bytes, protect, |k| !is_active(k));
        if plan.freed_bytes < need_bytes {
            let more = d.plan_evict_where(need_bytes - plan.freed_bytes, protect, is_active);
            plan.freed_bytes += more.freed_bytes;
            plan.victims.extend(more.victims);
        }
        plan
    }

    /// KV 盘层逐出记账/删账（返回释放字节；未接线 = 0）。
    pub fn kv_disk_commit_evict(&mut self, keys: &[ChainHash]) -> usize {
        self.disk.as_mut().map_or(0, |d| d.commit_evict(keys))
    }

    /// KV 盘层触达。
    pub fn kv_disk_touch(&mut self, key: ChainHash) -> StoreResult<()> {
        self.disk
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("KV 盘层未接线".into()))?
            .touch(key)
    }

    /// 状态腿盘层是否接线。
    pub fn snap_disk_enabled(&self) -> bool {
        self.snap_disk.as_ref().is_some_and(|d| !d.is_disabled())
    }

    /// 状态腿盘层是否有该快照内容。
    pub fn snap_disk_contains(&self, key: SnapKey) -> bool {
        self.snap_disk.as_ref().is_some_and(|d| d.contains(key))
    }

    /// 状态腿盘层单块字节（未接线 = `None`）。
    pub fn snap_disk_block_bytes(&self) -> Option<usize> {
        self.snap_disk.as_ref().map(|d| d.block_bytes())
    }

    /// 状态腿盘层空闲槽数（未接线 = 0）。
    pub fn snap_disk_free_slots(&self) -> usize {
        self.snap_disk.as_ref().map_or(0, |d| d.free_slots())
    }

    /// 状态腿盘层配额字节（0 = 未接线）。
    pub fn snap_disk_budget_bytes(&self) -> usize {
        self.snap_disk.as_ref().map_or(0, |d| d.budget_bytes())
    }

    /// 状态腿盘层根目录。
    pub fn snap_disk_root(&self) -> Option<&std::path::Path> {
        self.snap_disk.as_ref().map(|d| d.root())
    }

    /// 状态腿盘层内容路径。
    pub fn snap_disk_path_of(&self, key: SnapKey) -> Option<&std::path::Path> {
        self.snap_disk.as_ref().and_then(|d| d.path_of(key))
    }

    /// 状态腿盘层驻留条数。
    pub fn snap_disk_resident_count(&self) -> usize {
        self.snap_disk.as_ref().map_or(0, |d| d.resident_count())
    }

    /// 状态腿盘层驻留键。
    pub fn snap_disk_resident_keys(&self) -> Vec<SnapKey> {
        self.snap_disk.as_ref().map(|d| d.resident_keys()).unwrap_or_default()
    }

    /// 制定快照溢写计划（未接线 = 结构化错）。
    pub fn snap_disk_plan_spill(&self, key: SnapKey) -> StoreResult<SpillPlanOf<SnapKey>> {
        self.snap_disk
            .as_ref()
            .ok_or_else(|| StoreError::Invariant("状态腿盘层未接线".into()))?
            .plan_spill(key)
    }

    /// 快照溢写记账。
    pub fn snap_disk_commit_spill(&mut self, plan: &SpillPlanOf<SnapKey>) -> StoreResult<()> {
        self.snap_disk
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("状态腿盘层未接线".into()))?
            .commit_spill(plan)
    }

    /// 制定快照回填计划。
    pub fn snap_disk_plan_fill(&self, key: SnapKey) -> StoreResult<FillPlanOf<SnapKey>> {
        self.snap_disk
            .as_ref()
            .ok_or_else(|| StoreError::Invariant("状态腿盘层未接线".into()))?
            .plan_fill(key)
    }

    /// 快照回填记账。
    pub fn snap_disk_commit_fill(&mut self, key: SnapKey) -> StoreResult<()> {
        self.snap_disk
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("状态腿盘层未接线".into()))?
            .commit_fill(key)
    }

    /// 快照盘层逐出计划（未接线 = 空计划）。
    pub fn snap_disk_plan_evict(
        &self,
        need_bytes: usize,
        protect: &[SnapKey],
    ) -> DiskEvictPlanOf<SnapKey> {
        self.snap_disk
            .as_ref()
            .map(|d| d.plan_evict(need_bytes, protect))
            .unwrap_or_default()
    }

    /// **session 感知的快照盘层逐出计划**：先避开活跃 session，不够才动活跃面。
    pub fn snap_disk_plan_evict_sessions(
        &self,
        need_bytes: usize,
        protect: &[SnapKey],
        active_sessions: &[u64],
    ) -> DiskEvictPlanOf<SnapKey> {
        let Some(d) = self.snap_disk.as_ref() else {
            return DiskEvictPlanOf::default();
        };
        let is_active = |k: &SnapKey| {
            self.state
                .snap_session(*k)
                .map_or(false, |(session, _)| active_sessions.contains(&session))
        };
        let mut plan = d.plan_evict_where(need_bytes, protect, |k| !is_active(k));
        if plan.freed_bytes < need_bytes {
            let more = d.plan_evict_where(need_bytes - plan.freed_bytes, protect, is_active);
            plan.freed_bytes += more.freed_bytes;
            plan.victims.extend(more.victims);
        }
        plan
    }

    /// 快照盘层逐出记账/删账（返回释放字节；未接线 = 0）。
    pub fn snap_disk_commit_evict(&mut self, keys: &[SnapKey]) -> usize {
        self.snap_disk.as_mut().map_or(0, |d| d.commit_evict(keys))
    }

    /// 快照盘层触达。
    pub fn snap_disk_touch(&mut self, key: SnapKey) -> StoreResult<()> {
        self.snap_disk
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("状态腿盘层未接线".into()))?
            .touch(key)
    }

    /// **接线磁盘层**(boot 一次;S6)。
    ///
    /// - 目录布局:`root/kv/` + `root/snap/`(两腿分目录 —— 混放会让启动扫描
    ///   与清理串腿);
    /// - 配额：**分片不许超总账**（`kv_bytes + snap_bytes ≤ inputs.disk_bytes`，
    ///   与 host 同一条律；总账为 0 = 配置没给盘配额 ⇒ 响亮拒）；
    /// - **盘块 = 一条链的宿主槽**（`world ×` 单页）：溢写取料口是 `host_take`
    ///   （整槽交出去），回填也落整槽 —— 用单平面字节会让 `plan_spill.bytes`
    ///   与内容长度不符（TP 下写 2 倍、读 1 倍 = 静默截断）；
    /// - `snap_bytes` 的分片粒度 = 一条快照的字节（需先 `attach_snap_layout`）。
    pub fn attach_disk(
        &mut self,
        root: impl Into<std::path::PathBuf>,
        kv_bytes: usize,
        snap_bytes: usize,
    ) -> StoreResult<DiskAttach> {
        let total = self.budget.tier_budget().disk_bytes;
        if total == 0 {
            return Err(StoreError::Budget(
                "磁盘层配置为 0(先给 [store] disk_pool_size)".into(),
            ));
        }
        let want = kv_bytes
            .checked_add(snap_bytes)
            .ok_or_else(|| StoreError::Budget("磁盘分片溢出".to_string()))?;
        if want > total {
            return Err(StoreError::Budget(format!(
                "磁盘分片超总账:{kv_bytes} + {snap_bytes} > {total}(中层分片不许超总账)"
            )));
        }
        let snap_blk = self.snap_bytes;
        if snap_bytes > 0 && snap_blk == 0 {
            return Err(StoreError::Invariant(
                "状态腿盘层要先注册快照布局(attach_snap_layout)".into(),
            ));
        }
        let root = root.into();
        let kv_root = root.join("kv");
        let snap_root = root.join("snap");
        let kv = DiskTier::<ChainHash>::new(kv_root.clone(), self.leg.host_slot_bytes(), kv_bytes)?;
        // **盘上一个快照文件 = 全部 rank 平面**(2026-10-31 修):宿主层早就按
        // `slot_bytes × world` 建块(`attach_snap_layout`),盘层却只按**单 rank**
        // (`snap_bytes`)建 ⇒ TP2 写出的文件是单块的 2 倍 ⇒ 启动回填逐条
        // 「大小不符 ⇒ 隔离」⇒ 锚点快照全废、跨重启召回被 `reuse_face` 裁到 0。
        // 实测:2400 块 KV 全部回填,但 6 张快照全部隔离(159252480 ≠ 79626240)。
        let snap = DiskTier::<SnapKey>::new(
            snap_root.clone(),
            snap_blk.max(1).saturating_mul(self.world),
            snap_bytes,
        )?;
        let attach = DiskAttach {
            kv_root,
            snap_root,
            kv_slots: kv.free_slots(),
            snap_slots: snap.free_slots(),
        };
        self.disk = Some(kv);
        self.snap_disk = Some(snap);
        Ok(attach)
    }

    /// **启动扫描回填**(engine 读完两个目录后调用;S6 —— 账先回来,内容惰性读)。
    pub fn adopt_disk(
        &mut self,
        kv: Vec<(ChainHash, std::path::PathBuf, usize)>,
        snap: Vec<(SnapKey, std::path::PathBuf, usize)>,
    ) -> (AdoptReport, AdoptReport) {
        let a = match self.disk.as_mut() {
            Some(d) => d.adopt(kv),
            None => AdoptReport::default(),
        };
        let b = match self.snap_disk.as_mut() {
            Some(d) => d.adopt(snap),
            None => AdoptReport::default(),
        };
        self.adopt_disk_restore_snaps();
        (a, b)
    }

    /// 状态腿:盘上的快照要重建 `StateStore` 账(否则重启后 `probe` 看不见它们
    /// ⇒ 文件在却复用不了)。`SnapKey` 是块边界内容身份 ⇒ 键可独立恢复;
    /// `durability` 盘上没存 ⇒ 按 `Durable` 保守记(Transient 本来就不写盘)。
    fn adopt_disk_restore_snaps(&mut self) {
        let snap_keys: Vec<(SnapKey, usize)> = match self.snap_disk.as_ref() {
            Some(d) => {
                let bytes = d.block_bytes();
                d.resident_keys().into_iter().map(|k| (k, bytes)).collect()
            }
            None => Vec::new(),
        };
        for (k, bytes) in snap_keys {
            self.state
                .register_snap(k, bytes, Tier::Disk, SnapDurability::Durable);
        }
    }

    /// **状态腿淘汰序预览**(诊断面;`/debug/vram` 用)。
    ///
    /// 直接投影 [`Self::snap_victim_order`](与容量调度同一份判据)的前 `limit` 条,
    /// 带**会话归属**与**边界深度** —— 真机上可直接看"同会话深者先走"是否生效。
    pub fn snap_evict_preview(&self, limit: usize) -> Vec<SnapEvictPreview> {
        self.snap_victim_order(&[])
            .into_iter()
            .take(limit)
            .map(|k| {
                let (session, depth) = self.state.snap_session(k).unwrap_or((0, 0));
                SnapEvictPreview {
                    key: k,
                    session,
                    depth,
                    transient: self.state.snap_durability(k)
                        == Some(crate::state::SnapDurability::Transient),
                    role: self
                        .state
                        .snap_role(k)
                        .unwrap_or(crate::state::SnapRole::History),
                }
            })
            .collect()
    }

    /// **按盘上索引重建树**(S6.1;boot 扫完 `.idx` 后调用)。
    ///
    /// 逐链:**逐块重算身份**(`hash_block(parent, tokens)`)并**逐个查"内容文件在盘上"**
    /// —— 遇到第一块缺内容就**截断**(命中语义本就是"连续前缀",断链即停)。
    /// 重建的条目 `slot = None`(`locate` 归 `Tier::Host`),命中后由
    /// `fill_kv_from_disk` 读回宿主 —— 该链路在 S6 已打通。
    ///
    /// 页宽不符 / 一块内容都没有的链 = **跳过并列出**(不静默)。
    /// **boot 回填(不按预算裁)** —— 见 [`crate::disk::DiskTier::adopt_with`]:
    /// 配额交给索引重建之后的 `reconcile`(带保护集,删尾部保前缀)。
    pub fn adopt_disk_boot(
        &mut self,
        kv: Vec<(ChainHash, std::path::PathBuf, usize)>,
        snap: Vec<(SnapKey, std::path::PathBuf, usize)>,
    ) -> (AdoptReport, AdoptReport) {
        let a = match self.disk.as_mut() {
            Some(d) => d.adopt_with(kv, false),
            None => AdoptReport::default(),
        };
        let b = match self.snap_disk.as_mut() {
            Some(d) => d.adopt_with(snap, false),
            None => AdoptReport::default(),
        };
        self.adopt_disk_restore_snaps();
        (a, b)
    }

    /// **对账删除后回账 + 摘树**(KV 腿;2026-11-01)。
    ///
    /// 配额对账删掉的是**尾块**;树里那些条目(第一次重建时按"内容在场"建好)
    /// 现在内容没了 —— 必须一起摘掉,否则 `match_prefix` 会匹配到不存在的块,
    /// 回灌读盘时炸在 turn 里。返回 (出账字节, 摘掉的树条目数)。
    pub fn disk_forget_paths(&mut self, paths: &[std::path::PathBuf]) -> (usize, usize) {
        let mut keys = Vec::new();
        for p in paths {
            let stem = p
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            if let Some(k) = <ChainHash as crate::disk::DiskKey>::parse_stem(&stem) {
                keys.push(k);
            }
        }
        let bytes = match self.disk.as_mut() {
            Some(d) => d.commit_evict(&keys),
            None => 0,
        };
        let mut dropped = 0usize;
        for k in &keys {
            if self.leg.drop_chain(*k).unwrap_or(false) {
                dropped += 1;
            }
        }
        (bytes, dropped)
    }

    /// 认领磁盘上的块链(跨重启召回入口):按盘上内容的哈希重建链记录。
    ///
    /// 只做记账与链挂接;**不**落盘、不动 KV 池(回灌由调用方按需触发)。
    pub fn adopt_disk_chains(
        &mut self,
        chains: &[crate::disk_index::ChainIndex],
    ) -> StoreResult<AdoptChainsReport> {
        let mut rep = AdoptChainsReport::default();
        let Some(disk) = self.disk.as_ref() else {
            return Ok(rep);
        };
        let page = self.leg.block_tokens().max(1);
        for idx in chains {
            if idx.block_tokens != page {
                rep.skipped.push((
                    format!("{} 块", idx.blocks()),
                    format!("索引页宽 {} ≠ 腿 {page}", idx.block_tokens),
                ));
                continue;
            }
            let mut parent = ChainHash::new(0);
            let mut ok = 0usize;
            for (i, block) in idx.tokens.chunks(page).enumerate() {
                let tag = idx.tags.get(i).copied().unwrap_or(0);
                let h = crate::kv::hash_block(parent, block, tag);
                if !disk.contains(h) {
                    break; // 内容不在盘上 ⇒ 连续前缀到此为止
                }
                parent = h;
                ok += 1;
            }
            if ok == 0 {
                // **身份对拍诊断**(2026-11-01 P1):索引算出的前几个块身份 vs 盘上
                // 真实键 —— "盘上有内容却一块都不认"时,这一行直接给谁对不上。
                if let (Some(first), Some(disk_keys)) = (
                    idx.tokens.chunks(page).next(),
                    Some(disk.resident_keys()),
                ) {
                    let h0 = crate::kv::hash_block(ChainHash::new(0), first, 0);
                    let sample: Vec<String> = disk_keys
                        .iter()
                        .take(3)
                        .map(|k| format!("{k:?}"))
                        .collect();
                    owl_shared::log_warn!(
                        "store",
                        "身份对拍:索引首块 h={h0:?}(tokens[0..{page}]) 盘上含={} · \
                         盘前 3 键={sample:?} · 盘上块数={} · 索引块数={}",
                        disk.contains(h0),
                        disk_keys.len(),
                        idx.blocks(),
                    );
                }
                rep.skipped.push((
                    format!("{} 块", idx.blocks()),
                    "盘上一块内容都没有(索引孤儿)".to_string(),
                ));
                continue;
            }
            let prefix = idx.truncated(ok);
            let inserted = self
                .leg
                .register_chain_offloaded_tagged(&prefix.tokens, &prefix.tags)?;
            rep.chains += 1;
            rep.blocks += inserted.len();
            rep.kept.extend(inserted);
            if ok < idx.blocks() {
                rep.truncated.push((format!("{} 块", idx.blocks()), ok));
            }
        }
        Ok(rep)
    }

    /// 磁盘腿是否接线。
    pub fn disk_enabled(&self) -> bool {
        self.disk.as_ref().is_some_and(|d| !d.is_disabled())
    }

    /// **账上活内容的盘上路径**(磁盘清理白名单;绝不禁删)。
    /// = 盘层 resident 的全部路径(账内内容),外加快照腿。
    pub fn disk_live_paths(&self) -> std::collections::HashSet<std::path::PathBuf> {
        let mut out = std::collections::HashSet::new();
        if let Some(d) = self.disk.as_ref() {
            for k in d.resident_keys() {
                if let Some(p) = d.path_of(k) {
                    out.insert(p.to_path_buf());
                }
            }
        }
        out
    }

    /// **快照盘层账上路径**(清理白名单)。
    pub fn snap_disk_live_paths(&self) -> std::collections::HashSet<std::path::PathBuf> {
        let mut out = std::collections::HashSet::new();
        if let Some(d) = self.snap_disk.as_ref() {
            for k in d.resident_keys() {
                if let Some(p) = d.path_of(k) {
                    out.insert(p.to_path_buf());
                }
            }
        }
        out
    }

    /// **盘层账上实占字节**(诊断;与物理实占对账用)。
    pub fn disk_used_bytes(&self) -> usize {
        self.disk.as_ref().map(|d| d.used_bytes()).unwrap_or(0)
    }

    /// **快照盘层账上实占字节**(诊断)。
    pub fn snap_disk_used_bytes(&self) -> usize {
        self.snap_disk.as_ref().map(|d| d.used_bytes()).unwrap_or(0)
    }

    /// **快照盘层的保护集**(boot 配额对账用):账上仍在(Disk/Host/Gpu 任一)
    /// 的快照文件路径 —— 它们还能当复用锚点,对账时排最后删。
    pub fn snap_disk_protected_paths(&self) -> std::collections::HashSet<std::path::PathBuf> {
        let mut out = std::collections::HashSet::new();
        let Some(d) = self.snap_disk.as_ref() else {
            return out;
        };
        for k in d.resident_keys() {
            // **必须在**"KV 前缀里还有这块"的前提下才算活锚点(2026-11-01):
            // 配额对账把 KV 截到前 N 块后,落在 N 之后的锚点**没有 KV 内容可配**
            // ⇒ 留着也没用(实测:留下的是尾端锚点 ⇒ `reuse_face` 一路回退到 0)。
            let boundary_alive = self.carrier_of(k.boundary()).is_ok();
            if boundary_alive && self.state.locate(k).is_some() {
                if let Some(p) = d.path_of(k) {
                    out.insert(p.to_path_buf());
                }
            }
        }
        out
    }

    /// **磁盘层的会话归属汇总**（KV 链 + 快照；纯 host，零 I/O）。
    ///
    /// 盘上的内容是全局共享缓存，但调度仍需回答"这块盘是谁的历史" ——
    /// KV 归属来自内容树（`session_of`），快照归属来自 `StateStore`。
    pub fn disk_session_usage(&self) -> Vec<DiskSessionUsage> {
        use std::collections::BTreeMap;
        let mut by_session: BTreeMap<u64, DiskSessionUsage> = BTreeMap::new();
        if let Some(d) = self.disk.as_ref() {
            let block_bytes = d.block_bytes();
            for key in d.resident_keys() {
                let session = self.leg.session_of(key);
                let e = by_session.entry(session).or_insert(DiskSessionUsage {
                    session,
                    ..DiskSessionUsage::default()
                });
                e.kv_blocks += 1;
                e.kv_bytes = e.kv_bytes.saturating_add(block_bytes);
            }
        }
        if let Some(d) = self.snap_disk.as_ref() {
            let block_bytes = d.block_bytes();
            for key in d.resident_keys() {
                let session = self
                    .state
                    .snap_session(key)
                    .map_or(crate::policy::NO_SESSION, |(session, _)| session);
                let e = by_session.entry(session).or_insert(DiskSessionUsage {
                    session,
                    ..DiskSessionUsage::default()
                });
                e.snap_entries += 1;
                e.snap_bytes = e.snap_bytes.saturating_add(block_bytes);
            }
        }
        by_session.into_values().collect()
    }
}
