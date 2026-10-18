//! 关系层门面(`Rel`)—— 两个独立管理器 + 一层关联(裁决②;设计卷 §3.1/§3.2)。
//!
//! **职责只有四条**:①身份映射 ②召回语义判定 ③换出单合并 ④报告口径。
//! 不含前向计算;搬运只经 [`crate::moves`] 的自由函数(律⑦)。
//!
//! ```text
//! probe(PrefixId) → RecallHit { kv_tier, snap_tier, blocks }
//!                   ├ 两腿都在 ⇒ RecallMode::Exact      (精确续跑;本引擎独有)
//!                   ├ 只 KV    ⇒ RecallMode::ContextOnly(论文式上下文召回;近似)
//!                   └ 都无     ⇒ RecallMode::Miss
//! ```
//!
//! # S3 身份迁移(本刀)
//!
//! KV 腿从"物理槽号账"升级为 **槽池 + 内容前缀树**(`kv`);`Rel` 仍是**唯一门面**:
//! engine 侧的块表操作(`ensure_for_len`/`register_chain`/`release_table`)与
//! 卸载/回灌都从这里进 —— 跨腿台账([`TierLedger`])只在门面上维护,腿内自洽。

use crate::budget::{Admission, AdmissionVerdict, BudgetInputs, TurnFloor, VramBudget};
use crate::error::{StoreError, StoreResult};
use crate::host::HostTier;
use crate::id::{ChainHash, PrefixId, SnapKey};
use crate::kv::{CacheUpdate, KvStats};
use crate::leg::{BlockLeg, ChainHit, LegConfig, LegStats};
use crate::policy::{AdmissionPolicy, EvictCandidate, EvictPolicy, Leg, LruPolicy};
use crate::state::{SnapDurability, StateStats, StateStore};
use crate::host::SlotTable;
use owl_iface::contract::{ConnSlot, SwapPiece};
use crate::tier::{AssetKind, Tier, TierLedger, TierReport};

/// KV 腿几何(boot 装配一次;`block_tokens` 本引擎恒 32 —— 设计卷 §3.6.2)。
///
/// 收编后本类型只是 [`LegConfig`] 的**别名位**(门面签名可读性用):几何的唯一产地
/// 是腿自己的 `LegConfig`(§23.1)。
pub type KvConfig = LegConfig;

/// 召回语义(关系层的核心产物;两腿独立 ⇒ 命中可以只有一半)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecallMode {
    /// 块链 + 状态快照都在 ⇒ **精确续跑**(位置不重排、状态直接还原)。
    Exact,
    /// 只有块链 ⇒ **上下文召回**(近似:复发状态沿原路,不重算历史)。
    ContextOnly,
    /// 都没有 ⇒ 冷启动(逐 token 重算)。
    Miss,
}

/// 召回命中面(层信息;组装与搬运在 engine 侧段边界执行)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecallHit {
    /// attention 块链所在层(`None` = 未命中)。
    pub kv_tier: Option<Tier>,
    /// GDN 快照所在层(`None` = 无快照腿)。
    pub snap_tier: Option<Tier>,
    /// 命中块数(0 = 无 KV 腿)。
    pub blocks: usize,
}

impl RecallHit {
    /// 未命中。
    pub fn miss() -> Self {
        Self { kv_tier: None, snap_tier: None, blocks: 0 }
    }

    /// 语义判定(唯一产地)。
    pub fn mode(&self) -> RecallMode {
        match (self.kv_tier, self.snap_tier) {
            (Some(_), Some(_)) => RecallMode::Exact,
            (Some(_), None) => RecallMode::ContextOnly,
            _ => RecallMode::Miss,
        }
    }

    /// 是否命中任意一腿。
    pub fn hit(&self) -> bool {
        self.kv_tier.is_some()
    }
}

/// 换出单的一行(动作 = 下沉一层 或 释放;执行归 engine)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvictItem {
    /// 腿。
    pub leg: Leg,
    /// 从哪层出。
    pub from: Tier,
    /// 字节。
    pub bytes: usize,
    /// 块链身份(KV 腿;**内容身份** —— 执行按身份找宿主副本/树条目)。
    pub chain: Option<ChainHash>,
    /// 快照键(若为 State 腿)。
    pub snap: Option<SnapKey>,
}

/// 换出单(两腿候选**合并后**的统一序 —— 关系层职责③)。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvictPlan {
    /// 统一序(前面的先动)。
    pub items: Vec<EvictItem>,
    /// 合计字节。
    pub freed_bytes: usize,
}

/// 保护面(活跃 turn 的地板/行 + 显式点名)—— 驱逐绝不碰。
#[derive(Clone, Debug, Default)]
pub struct Protected {
    /// 活跃 turn 的 attention 链(其块不可逐)。
    pub chains: Vec<ChainHash>,
    /// 活跃 turn 的快照键(点名保住)。
    pub snaps: Vec<SnapKey>,
}

/// 全量报告(设计卷 §2.4 报告面)。
#[derive(Clone, Debug, Default)]
pub struct StoreReport {
    /// 三层账明细。
    pub tiers: TierReport,
    /// attention 腿统计。
    pub kv: KvStats,
    /// 状态腿统计。
    pub state: StateStats,
    /// 活跃 turn 数。
    pub active_turns: usize,
    /// 保留面字节(地板 + 行)。
    pub reserved_bytes: usize,
    /// 非 turn 归属的不可动面字节(idle 会话持着的历史;§23.2)。
    pub session_held_bytes: usize,
    /// 弹性面字节(可驱逐)。
    pub elastic_bytes: usize,
    /// 缓存面字节(可驱逐/可下沉)。
    pub cache_bytes: usize,
    /// 策略名(驱逐序)。
    pub policy: String,
    /// 宿主层驻留字节。
    pub host_resident_bytes: usize,
    /// 宿主层空转缓冲字节(真实 RAM;报告必须算上)。
    pub host_free_bytes: usize,
}

/// 关系层门面:两个独立管理器 + 一层关联。
///
/// **收编后**(§23.1):KV 腿的全部状态(槽池/内容身份树/宿主物理)只住 [`BlockLeg`];
/// 本层不另记一份 —— `ledger` 只记 store 自管面(状态腿/保留面/弹性面),
/// KV 块面的真源 = 腿的报告(`LegStats`)。
pub struct Rel {
    /// attention 块腿(账 + 宿主物理 + 搬运口 —— 唯一状态持有者)。
    pub leg: BlockLeg,
    /// GDN 状态账(行 + 快照;快照的**层**是真源 —— §25.2)。
    pub state: StateStore,
    /// 状态腿的宿主层(键 = [`SnapKey`];与 KV 腿各自一本账)。
    pub snap_host: HostTier<SnapKey>,
    /// **选择层检索索引**(B 轴:Mean-K 块级索引;host 驻留 —— 论文同款)。
    /// 几何由 engine 在 boot 注册([`Rel::attach_index`])。
    pub index: Option<crate::select::MeanKIndex>,
    /// 语义账(状态腿快照/行 + 保留面/弹性面 + **DeviceHandle 事件账**)。
    ///
    /// `Rc<RefCell<..>>`(2026-10-19 步 2):DeviceHandle 侧的钩子要**与 store 看同一本账**
    /// (方案甲)⇒ 共享所有权;引擎单线程 ⇒ 无锁。store 自己的读写走 `borrow()/borrow_mut()`。
    pub ledger: std::rc::Rc<std::cell::RefCell<TierLedger>>,
    /// 准入策略(boot 装配)。
    pub policy: AdmissionPolicy,
    /// 活跃 turn 在飞账。
    pub admission: Admission,
    budget: VramBudget,
    evict: Box<dyn EvictPolicy>,
    /// 状态腿的设备布局(快照 = 「槽 × 片」;engine 在 boot 注册,§25.2)。
    snap_layout: Option<crate::leg::PoolLayout>,
    /// 一条快照的全量字节(布局注册后由 `slot_bytes()` 定谳)。
    snap_bytes: usize,
}

impl Rel {
    /// 装配(唯一入口:预算分解失败 = **拒绝启动**,A5.1)。
    ///
    /// `kv_cfg` = KV 腿几何(含宿主层预算) —— 每 token KV 字节由它推
    /// (`block_bytes / block_tokens`),与 `inputs.kv_bytes_per_token`
    /// **交叉校验**(同一件事只有一个入口;矛盾 = 装配错,响亮拒)。
    pub fn open(
        inputs: BudgetInputs,
        policy: AdmissionPolicy,
        kv_cfg: LegConfig,
    ) -> StoreResult<Self> {
        if kv_cfg.block_bytes == 0 || kv_cfg.block_tokens == 0 || kv_cfg.slots == 0 {
            return Err(StoreError::Budget(format!(
                "KV 腿几何非法:slots={} block_tokens={} block_bytes={}",
                kv_cfg.slots, kv_cfg.block_tokens, kv_cfg.block_bytes
            )));
        }
        let per_token = kv_cfg.block_bytes / kv_cfg.block_tokens;
        if per_token != inputs.kv_bytes_per_token {
            return Err(StoreError::Budget(format!(
                "每 token KV 字节不一致:腿几何算得 {per_token} ≠ 预算入参 {}(装配参数矛盾)",
                inputs.kv_bytes_per_token
            )));
        }
        if kv_cfg.block_tokens != inputs.page_tokens {
            return Err(StoreError::Budget(format!(
                "页宽不一致:腿几何 block_tokens={} ≠ 预算入参 page_tokens={}\
                 (地板按页口径算、链增长也按页 —— 两把尺必须同一把)",
                kv_cfg.block_tokens, inputs.page_tokens
            )));
        }
        // host 预算只有一个总量(`inputs.host_bytes` = 整个 store 的中层预算),腿拿其中
        // 一段 ⇒ 腿那段不许超过总量(超了就是两套规范,不许静默对不上)。
        if kv_cfg.host_bytes > inputs.host_bytes {
            return Err(StoreError::Budget(format!(
                "KV 腿 host 预算 {} > store host 总量 {}(中层分片不许超总账)",
                kv_cfg.host_bytes, inputs.host_bytes
            )));
        }
        let budget = VramBudget::decompose(inputs, policy.gdn_slots)?;
        let ledger = std::rc::Rc::new(std::cell::RefCell::new(TierLedger::new(budget.tier_budget())));
        let state = StateStore::new(policy.gdn_slots, inputs.gdn_row_bytes);
        // 状态腿宿主层:预算 = store host 总量 − KV 腿分片(负数 = 装配错,响亮拒)
        let snap_host_bytes = inputs.host_bytes.checked_sub(kv_cfg.host_bytes).ok_or_else(|| {
            StoreError::Budget(format!(
                "KV 腿 host 分片 {} > store host 总量 {}",
                kv_cfg.host_bytes, inputs.host_bytes
            ))
        })?;
        Ok(Self {
            leg: BlockLeg::open(kv_cfg)?,
            index: None,
            state,
            snap_host: HostTier::new(1, snap_host_bytes)?,
            ledger,
            policy,
            admission: Admission::new(),
            budget,
            evict: Box::new(LruPolicy),
            snap_layout: None,
            snap_bytes: 0,
        })
    }

    /// **注册状态腿的设备布局**(boot 一次;§25.2):快照 = 一种"块"
    /// (槽 × 片),片 = 每个 (层,quad) 一行宽缓冲。
    pub fn attach_snap_layout(&mut self, layout: crate::leg::PoolLayout) -> StoreResult<u64> {
        let bytes = layout.slot_bytes();
        if bytes == 0 {
            return Err(StoreError::Invariant("状态腿布局单块字节为 0".into()));
        }
        // 宿主层的块粒度 = 一条快照(建 host 时未知,故此处重建:预算不变)
        let budget = self.snap_host.budget_bytes();
        self.snap_host = HostTier::new(bytes, budget)?;
        let epoch = layout.epoch;
        self.snap_layout = Some(layout);
        self.snap_bytes = bytes;
        Ok(epoch)
    }

    /// **注册选择层索引几何**(B 轴 boot 一次;§8):`dim` = head_dim。
    pub fn attach_index(&mut self, dim: usize, layers: usize, kv_heads: usize) -> StoreResult<()> {
        self.index = Some(crate::select::MeanKIndex::new(dim, layers, kv_heads)?);
        Ok(())
    }

    /// **登记一块的索引面**(B 轴):形状校验在 `MeanKIndex` 内;入账到 host 层。
    pub fn index_register(
        &mut self,
        block: ChainHash,
        entries: Vec<crate::select::MeanKEntry>,
    ) -> StoreResult<()> {
        let ix = self
            .index
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("选择层索引未注册(engine 应在 boot 调 attach_index)".into()))?;
        let bytes = ix.block_bytes();
        ix.register(block, entries)?;
        // 第 0 刀:索引面 = **缓存归属**(无单会话归属;跨会话共享)
        self.ledger.borrow_mut().add_owned(crate::tier::Owner::Cache, Tier::Host, AssetKind::Index, bytes)
    }

    /// 丢一块的索引(出账)。
    pub fn index_drop(&mut self, block: ChainHash) -> StoreResult<()> {
        let ix = self
            .index
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("选择层索引未注册".into()))?;
        let bytes = ix.block_bytes();
        ix.drop_block(block)?;
        self.ledger.borrow_mut().remove_owned(crate::tier::Owner::Cache, Tier::Host, AssetKind::Index, bytes)
    }

    /// 索引总字节(0 = 未注册)。
    pub fn index_bytes(&self) -> usize {
        self.index.as_ref().map(|i| i.bytes()).unwrap_or(0)
    }

    /// 一条快照的字节(布局未注册 = 0)。
    pub fn snap_bytes(&self) -> usize {
        self.snap_bytes
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
    /// 引擎在下沉点打点上报（不许静默）。
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
        let used = self.snap_host.used_bytes() + self.snap_host.free_bytes();
        self.snap_host
            .budget_bytes()
            .saturating_sub(used)
            / self.snap_bytes.max(1)
    }

    /// 卡级预算(只读)。
    pub fn budget(&self) -> &VramBudget {
        &self.budget
    }

    /// 换驱逐策略(策略对象可换,判据不散 —— §3.2)。
    /// **召回信号转发**(B-4):召回点选中哪些块 ⇒ 策略面(attention-aware LRFU)。
    pub fn note_recall(&self, selected: &[ChainHash]) {
        self.evict.note_recall(selected);
    }

    /// 驱逐策略名(报告/测试面)。
    pub fn evict_policy_name(&self) -> &'static str {
        self.evict.name()
    }

    /// 换驱逐策略(`[kvmem] evict_policy`;装配期一次)。
    pub fn set_evict_policy(&mut self, p: Box<dyn EvictPolicy>) {
        self.evict = p;
    }

    /// **命中查询**(按身份;纯 host、零副作用 —— 律⑥)。
    ///
    /// 与 [`Self::match_prefix`] 二选一使用(两条查询路径各自记一次命中账;
    /// 同一次查询不要两条都调,否则计数双算)。
    pub fn probe(&mut self, id: PrefixId) -> RecallHit {
        let kv_tier = self.leg.carrier_of(id.chain).ok().map(|c| match c {
            crate::leg::Carrier::Device(_) => Tier::Gpu,
            crate::leg::Carrier::Host => Tier::Host,
            crate::leg::Carrier::Disk => Tier::Disk,
        });
        let snap_tier = self.state.locate(id.snap_key());
        let hit = RecallHit { kv_tier, snap_tier, blocks: kv_tier.map_or(0, |_| self.leg.chain_len(id.chain)) };
        self.leg.note_probe(hit.kv_tier.is_some());
        self.state.note_probe(hit.snap_tier.is_some());
        hit
    }

    /// **按 token 命中查询**(engine L1 的第一跳):逐块内容匹配,回块链 + 绑定。
    ///
    /// 未绑定的格(`slot = None`)= 内容在宿主层,调用方须在**段边界**回灌
    /// (律③)—— 本方法自身不做设备动作。
    pub fn match_prefix(&mut self, tokens: &[u32]) -> StoreResult<(ChainHit, RecallHit)> {
        let m = self.leg.match_prefix(tokens)?;
        let tail = m.hashes.last().copied();
        let kv_tier = tail.and_then(|t| self.leg.carrier_of(t).ok()).map(|c| match c {
            crate::leg::Carrier::Device(_) => Tier::Gpu,
            crate::leg::Carrier::Host => Tier::Host,
            crate::leg::Carrier::Disk => Tier::Disk,
        });
        let snap_tier = tail.and_then(|t| self.state.locate(SnapKey::at_boundary(t)));
        let hit = RecallHit { kv_tier, snap_tier, blocks: m.matched_blocks };
        Ok((m, hit))
    }

    /// **登记一条前缀链**(turn 收口 L3):缓存取一份绑定引用,容量驱逐同步丢宿主副本。
    ///
    /// 只做 KV 腿;快照腿由 [`StateStore::return_row`] 登记(两腿独立 —— 裁决②)。
    pub fn register_chain(&mut self, tokens: &[u32], table: &[u32]) -> StoreResult<CacheUpdate> {
        self.leg.register_chain(tokens, table)
    }

    /// 块表增长(engine 每步/每 turn 调;压力驱逐与宿主副本回收均在腿内)。
    pub fn ensure_for_len(&mut self, table: &mut Vec<u32>, target_len: usize) -> StoreResult<()> {
        self.leg.ensure_for_len(table, target_len)
    }

    /// 归还会话块表。
    pub fn release_table(&mut self, table: &mut Vec<u32>) -> StoreResult<()> {
        self.leg.release_table(table)
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

    /// 换出单(两腿候选 → 统一序;`need_bytes` 满足即停)。
    pub fn plan_evict(&self, need_bytes: usize, protect: &Protected) -> EvictPlan {
        let mut cands: Vec<EvictCandidate> = Vec::new();
        cands.extend(self.leg.evict_candidates(&protect.chains));
        cands.extend(self.state.evict_candidates(&protect.snaps));
        cands.sort_by_key(|c| (c.leg, c.chain, c.snap, c.tier));
        cands.dedup_by_key(|c| (c.leg, c.chain, c.snap, c.tier));
        self.evict.order(&mut cands);
        let mut items = Vec::new();
        let mut freed = 0usize;
        for c in cands {
            if freed >= need_bytes {
                break;
            }
            items.push(EvictItem {
                leg: c.leg,
                from: c.tier,
                bytes: c.bytes,
                chain: c.chain,
                snap: c.snap,
            });
            freed += c.bytes;
        }
        EvictPlan { items, freed_bytes: freed }
    }

    // ───────────────── 状态腿搬运口(§25.2;与 KV 腿同形:store 出计划与账) ─────────────────

    /// 状态腿的片(逐 (层,quad);`dev_off = 槽 × 行宽`)。
    fn snap_pieces(&self, slot: u32) -> StoreResult<Vec<SwapPiece>> {
        let layout = self
            .snap_layout
            .as_ref()
            .ok_or_else(|| StoreError::Invariant("状态腿布局未注册(engine 应在 boot 调 attach_snap_layout)".into()))?;
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

    /// **快照下沉计划**(设备槽 `slot` 上的快照 → 宿主)。
    /// `Busy` = 快照不在账/宿主层已满(零副作用,调用方跳过)。
    ///
    /// **不重要就不搬**(2026-10-19):生成段边界快照逐出即丢 —— 下沉一条
    /// 159 MiB ≈ **50 ms**(实测,页锁定外的 PCIe 速),而生成段每 32 tok 就出
    /// 一条 ⇒ 全下沉 = 每 token 数 MiB 级 PCIe 流量,不可承受。拒语即事实
    /// (调用方回落已有"沉不下⇒丢"路;见 [`SnapDurability`])。
    pub fn plan_snap_offload(&mut self, key: SnapKey, slot: u32) -> StoreResult<(Vec<SwapPiece>, SlotTable)> {
        let bytes = self
            .state
            .snap_bytes(key)
            .ok_or_else(|| StoreError::Unknown(format!("快照 {key:?} 不在账")))?;
        if self.state.snap_durability(key) != Some(SnapDurability::Durable) {
            return Err(StoreError::Invariant(format!(
                "快照 {key:?} 不可下沉(生成段边界:链一推进就被取代 ⇒ 逐出即丢)"
            )));
        }
        if bytes != self.snap_bytes {
            return Err(StoreError::Invariant(format!(
                "快照字节不符:账 {bytes} ≠ 布局 {}",
                self.snap_bytes
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
        // 政策 = **同一份策略对象**(`EvictPolicy`,与 KV 腿共用):只比 `Tier::Host`
        // 的候选,排序后取第一个。代价说明:先逐后搬 ⇒ 若接下来的 DMA 失败
        // (`abort_snap_offload`),冷条目已丢(比"丢新条目"代价小 —— 冷条目
        // 本来就要被逐)。
        let last = self.snap_host.used_bytes();
        let mut guard = 0usize;
        let buf = loop {
            match self.snap_host.acquire() {
                Ok(b) => break b,
                Err(StoreError::Exhausted(why)) => {
                    let mut cands: Vec<EvictCandidate> = self
                        .state
                        .evict_candidates(&[])
                        .into_iter()
                        .filter(|c| c.tier == Tier::Host)
                        .collect();
                    if cands.is_empty() {
                        // 宿主账与层账不一致(账说满而层无可逐)—— 不许静默
                        return Err(StoreError::Invariant(format!(
                            "宿主梯满且无 Host 层候选可逐({why};层账={last}B)"
                        )));
                    }
                    self.evict.order(&mut cands);
                    let v = cands[0]
                        .snap
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
        let mut pieces = self.snap_pieces(slot)?;
        // 片在宿主槽内**首尾相接**(每条快照 = 一个槽)
        let mut off = 0usize;
        for p in &mut pieces {
            p.conn_off = off;
            off += p.len;
        }
        Ok((pieces, SlotTable::one(buf)))
    }

    /// **快照下沉记账**(搬运成功后;纯 host):内容落宿主 + 换层。
    pub fn commit_snap_offload(
        &mut self,
        key: SnapKey,
        table: SlotTable,
    ) -> StoreResult<()> {
        let buf = table
            .into_slots()
            .into_iter()
            .next()
            .ok_or_else(|| StoreError::Invariant("快照槽台为空".into()))?;
        self.snap_host.put(key, buf)?;
        self.state.set_tier(key, Tier::Host)?;
        Ok(())
    }

    /// **快照下沉回滚**(搬运失败):宿主缓冲随失败的 connector 释放,层不动。
    pub fn abort_snap_offload(&mut self, _key: SnapKey) {}

    /// **快照回灌计划**(宿主 → 设备槽 `slot`)。
    pub fn plan_snap_onload(&mut self, key: SnapKey, slot: u32) -> StoreResult<(Vec<SwapPiece>, SlotTable)> {
        let buf = self.snap_host.take(key)?;
        let mut pieces = self.snap_pieces(slot)?;
        let mut off = 0usize;
        for p in &mut pieces {
            p.conn_off = off;
            off += p.len;
        }
        Ok((pieces, SlotTable::one(buf)))
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

    /// 丢弃一条快照的宿主副本(内容整体淘汰时;不在宿主 = 幂等)。
    pub fn drop_snap_host(&mut self, key: SnapKey) -> StoreResult<usize> {
        match self.snap_host.evict(key) {
            Ok(n) => Ok(n),
            Err(StoreError::Unknown(_)) => Ok(0),
            Err(e) => Err(e),
        }
    }

    /// 报告(两腿 + 保留/弹性面)。
    ///
    /// KV 块面从**腿**报出(单一真源),`tiers` 只列 store 自管面
    /// (状态腿/保留面/弹性面)—— 同一字节不两处记(§13.1/§23.1)。
    pub fn report(&self) -> StoreReport {
        // **第 0.5 刀:腿自报 → owner 行(对账式记账)**
        //
        // 为什么先"对账"而不是"事件记账":引擎有几十处直接 poke 腿
        // (`bank.rel.leg.*` / `bank.rel.state.*`),事件记账要逐个接线(后续);
        // 而对账在**报告时刻**取腿的权威状态 ⇒ 零漏记风险(腿是唯一真源),
        // 已能回答"谁占着多少"与"逐轮是否爬坡"(泄漏判据)。
        // 事件式记账的收益 = 精确区分"仍被持有"vs"未归还",留作接线后的第二步。
        let mut tiers = self.ledger.borrow().report();
        let legs = self.leg_accounting();
        for row in &legs.owners {
            tiers.rows.push(crate::tier::TierRow {
                tier: row.tier,
                kind: row.kind,
                bytes: row.bytes,
                entries: row.entries,
            });
        }
        tiers.owners.extend(legs.owners);
        tiers.owners.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.owner.cmp(&b.owner)));
        StoreReport {
            tiers,
            kv: self.leg.kv_stats(),
            state: self.state.stats(),
            active_turns: self.admission.active_turns(),
            reserved_bytes: self.admission.reserved_bytes(),
            session_held_bytes: self.admission.session_held_bytes(),
            elastic_bytes: self.admission.elastic_bytes(),
            cache_bytes: self.admission.cache_bytes(),
            policy: self.evict.name().to_string(),
            host_resident_bytes: self.leg.host_used_bytes(),
            host_free_bytes: self.leg.host_free_bytes(),
        }
    }

    /// **腿自报 → owner 行**(第 0.5 刀;对账式记账的原料)。
    ///
    /// 归属规则(有意保守,不猜):
    /// - KV 块(设备/宿主)= `Cache`(跨会话共享,无单会话归属);
    /// - 快照(设备/宿主)= `Cache`(Durable 可下沉 / Transient 逐出即丢,都是缓存面);
    /// - GDN 行 = `System`(逐会话归属需引擎的会话表;接线后再细化,不猜)。
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
            owners.push(crate::tier::OwnerRow { owner, tier, kind, bytes, entries });
            rows.push(crate::tier::TierRow { tier, kind, bytes, entries });
        };
        // ① KV 腿:设备驻留(gpu_bytes)+ 宿主驻留(host_used_bytes)
        let kv = self.leg.kv_stats();
        let kv_dev_entries = kv.cached_blocks.saturating_sub(kv.host_blocks);
        push(&mut owners, &mut rows, crate::tier::Owner::Cache, Tier::Gpu,
             AssetKind::KvBlock, kv.gpu_bytes, kv_dev_entries);
        push(&mut owners, &mut rows, crate::tier::Owner::Cache, Tier::Host,
             AssetKind::KvBlock, self.leg.host_used_bytes(), kv.host_blocks);
        // ② 状态腿:快照(设备/宿主)+ 行
        let st = self.state.stats();
        push(&mut owners, &mut rows, crate::tier::Owner::Cache, Tier::Gpu,
             AssetKind::Snap, st.snap_gpu.1, st.snap_gpu.0);
        push(&mut owners, &mut rows, crate::tier::Owner::Cache, Tier::Host,
             AssetKind::Snap, st.snap_host.1, st.snap_host.0);
        let rows_in_use = st.rows_total.saturating_sub(st.rows_free);
        push(&mut owners, &mut rows, crate::tier::Owner::System, Tier::Gpu,
             AssetKind::Row, rows_in_use * self.state.row_bytes(), rows_in_use);
        crate::tier::TierReport { rows, owners, ..Default::default() }
    }

    /// **账本句柄**(给 `DeviceHandle` 的钩子用;与 `self.ledger` **同一本账**)。
    ///
    /// 用途(boot 接线):`DeviceHandle::new(client, Some(Rc::new(RefCell::new(rel.ledger_handle()))))`。
    pub fn ledger_handle(&self) -> crate::tier::LedgerHandle {
        crate::tier::LedgerHandle::new(self.ledger.clone())
    }

    /// 腿报告(KV 腿的完整面:槽/缓存/宿主/搬运计数 —— 报告与对账口用)。
    pub fn leg_stats(&self) -> LegStats {
        self.leg.report()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::BudgetInputs;
    use crate::id::ChainHash;
    use crate::policy::QueuePolicy;

    const BLOCK_BYTES: usize = 1 << 20;

    fn inputs() -> BudgetInputs {
        BudgetInputs {
            v_card_bytes: 25_000_000_000,
            weights_bytes: 17_500_000_000,
            runtime_floor_bytes: 2_000_000_000,
            draft_pool_bytes: 0,
            gdn_row_bytes: 162_000_000,
            kv_bytes_per_token: 32 * 1024,
            page_tokens: 32,
            host_bytes: 8 << 20,
            disk_bytes: 0,
            kv_face_bytes: 0,
        }
    }

    fn kv_cfg(slots: usize, cache_cap: usize) -> KvConfig {
        KvConfig {
            slots,
            block_tokens: 32,
            block_bytes: BLOCK_BYTES,
            cache: crate::leg::CachePolicy::Cap(cache_cap),
            host_bytes: 8 << 20,
            class: crate::leg::BlockClass::KvPage,
            label: "test",
        }
    }

    fn rel(turns: usize) -> Rel {
        let mut r = Rel::open(inputs(), AdmissionPolicy::new(turns, QueuePolicy::Reject), kv_cfg(64, 32))
            .expect("装配");
        // 注册池布局(单片段;测试只跑账/计划,不跑真搬运 —— 搬迁在 store_e2e 覆盖)
        r.leg
            .attach_pool(crate::leg::PoolLayout {
                class: crate::leg::BlockClass::KvPage,
                epoch: 0,
                pieces: vec![crate::leg::PieceSpec {
                    buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                    stride_bytes: BLOCK_BYTES,
                }],
            })
            .expect("布局");
        r
    }

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn open_rejects_unmeasured_budget_and_geometry() {
        // 动态池档(无池面兜底):卡级缺项 = 拒
        let mut i = inputs();
        i.weights_bytes = 0;
        assert!(Rel::open(i, AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(8, 8)).is_err());
        // 几何非法 = 拒
        assert!(
            Rel::open(inputs(), AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(0, 8)).is_err()
        );
        // 每 token 字节与腿几何矛盾 = 拒(同一件事只能有一个入口)
        let mut i2 = inputs();
        i2.kv_bytes_per_token = 4096;
        assert!(Rel::open(i2, AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(8, 8)).is_err());
        // 页宽与腿几何矛盾 = 拒(A7:地板按页算、链增长也按页 ⇒ 同一把尺)
        let mut i3 = inputs();
        i3.page_tokens = 16;
        assert!(Rel::open(i3, AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(8, 8)).is_err());
    }

    /// @cu: 固定池档:池是准入面,卡级可缺项 —— 但"未实测"必须可查(不许静默)。
    #[test]
    fn fixed_pool_rel_opens_without_card_measurements() {
        let mut i = inputs();
        i.v_card_bytes = 0;
        i.weights_bytes = 0;
        i.runtime_floor_bytes = 0;
        i.kv_face_bytes = 64 * BLOCK_BYTES;
        let r = Rel::open(i, AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(64, 32))
            .expect("固定池档应能装配");
        assert!(!r.budget().card_measured());
        assert_eq!(r.budget().face_bytes(), 64 * BLOCK_BYTES, "判据面 = 池");
    }

    #[test]
    fn facade_delegates_to_the_leg_single_source() {
        let mut r = rel(2);
        let tokens = toks(64);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 64).expect("块表");
        assert_eq!(r.leg_stats().slots_total, 64);
        r.register_chain(&tokens, &table).expect("登记");
        assert_eq!(r.report().kv.cached_blocks, 2, "缓存条目走腿报告,不另记一份");
        assert_eq!(r.leg_stats().cached_blocks, 2);
        // tiers 只记 store 自管面(KV 块面归腿)⇒ 这里恒 0
        assert_eq!(r.report().tiers.gpu_used, 0, "KV 块不在语义账里重复记");

        // 命中面:门面回 (块腿三元组, 关系层语义) 且两者一致
        let (hit, recall) = r.match_prefix(&tokens).expect("命中查询");
        assert_eq!(hit.matched_blocks, 2);
        assert_eq!(recall.blocks, 2);
        assert_eq!(recall.kv_tier, Some(Tier::Gpu));
        assert_eq!(recall.mode(), RecallMode::ContextOnly, "状态腿不在 ⇒ 上下文召回");
        // 会话表还在 ⇒ 活跃面不可下沉(Busy);归还后 ⇒ 可沉(搬迁本体在 store_e2e 覆盖)
        let tail = r.leg.tail_hash(&tokens, 2).expect("尾身份");
        // 会话表还在 ⇒ 活跃面:整条链不出票(§14.6 律②)
        assert!(matches!(crate::leg::plan_offload(&mut r.leg, tail), Err(StoreError::Busy(_))));
        r.release_table(&mut table).expect("归还");
        assert!(crate::leg::plan_offload(&mut r.leg, tail).is_ok(), "缓存独占可搬");
    }

    #[test]
    fn recall_semantics_are_the_relation_layer_business() {
        let mut r = rel(2);
        let tokens = toks(64);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 64).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        let tail = r.leg.tail_hash(&tokens, 2).expect("尾身份");
        let id = PrefixId::new(tail);

        let h = r.probe(id);
        assert_eq!(h.mode(), RecallMode::ContextOnly, "只 KV ⇒ 上下文召回");
        assert_eq!(h.blocks, 2);
        assert_eq!(r.probe(PrefixId::new(ChainHash::new(9999))).mode(), RecallMode::Miss);

        // 状态腿到位 ⇒ Exact
        let row = r.state.borrow_row().expect("借行");
        r.state.return_row(row, Some(id.snap_key())).expect("落快照");
        assert_eq!(r.probe(id).mode(), RecallMode::Exact);
    }

    #[test]
    fn evict_plan_merges_both_legs_and_respects_protect() {
        let mut r = rel(2);
        let ta = toks(32);
        let tb: Vec<u32> = (100..132).collect();
        for t in [&ta, &tb] {
            let mut table = Vec::new();
            r.ensure_for_len(&mut table, 32).expect("块表");
            r.register_chain(t, &table).expect("登记");
        }
        let a = r.leg.tail_hash(&ta, 1).expect("A 身份");
        let b = r.leg.tail_hash(&tb, 1).expect("B 身份");
        let row = r.state.borrow_row().expect("借行");
        r.state.return_row(row, Some(SnapKey::at_boundary(ChainHash::new(9)))).expect("快照");

        let plan = r.plan_evict(usize::MAX, &Protected { chains: vec![a], snaps: vec![] });
        assert_eq!(plan.items.len(), 2, "B 的块 + 快照(两腿合并):{plan:?}");
        assert!(plan.items.iter().any(|i| i.leg == Leg::Kv && i.chain == Some(b)));
        assert!(plan.items.iter().any(|i| i.leg == Leg::State));
        assert_eq!(plan.freed_bytes, BLOCK_BYTES + 162_000_000);
    }

    #[test]
    fn protected_active_turn_floor_is_never_evicted() {
        let mut r = rel(1);
        let tokens = toks(32);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 32).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        let chain = r.leg.tail_hash(&tokens, 1).expect("身份");
        r.admission.admit(1, 0, 0).expect("登记活跃 turn");
        r.admission.set_floor(1, BLOCK_BYTES).expect("实占更新");
        let plan = r.plan_evict(usize::MAX, &Protected { chains: vec![chain], snaps: vec![] });
        assert!(
            plan.items.iter().all(|i| i.chain != Some(chain)),
            "保留额被逐 = 不变量违反:{plan:?}"
        );
    }

    #[test]
    fn report_surfaces_both_legs_and_faces() {
        let mut r = rel(2);
        let tokens = toks(32);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 32).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        r.admission.admit(1, 0, 0).expect("活跃 turn");
        r.admission.set_floor(1, 4 << 20).expect("实占");
        r.admission.set_elastic(1, 1 << 20).expect("弹性");
        let rep = r.report();
        assert_eq!(rep.active_turns, 1);
        assert_eq!(rep.reserved_bytes, 4 << 20);
        assert_eq!(rep.elastic_bytes, 1 << 20);
        assert_eq!(rep.policy, "lru");
        assert_eq!(rep.state.rows_total, 2);
        assert_eq!(rep.kv.cached_blocks, 1, "缓存面 1 块(腿报)");
        assert_eq!(rep.kv.gpu_bytes, BLOCK_BYTES, "设备绑定面 1 块");
        assert_eq!(rep.tiers.gpu_used, 0, "语义账不重复记 KV 块");
    }

    /// **B-0 门(store 侧)**:索引几何注册 + 登记入账(host/Index 行)+ 出账平衡。
    #[test]
    fn index_registers_into_the_ledger_and_balances() {
        let mut r = rel(2);
        r.attach_index(2, 1, 2).expect("注册几何");
        assert_eq!(r.index_bytes(), 0);
        let b = ChainHash::new(42);
        let entries = vec![
            crate::select::MeanKEntry { layer: 0, kv_head: 0, k_bar: vec![0, 0] },
            crate::select::MeanKEntry { layer: 0, kv_head: 1, k_bar: vec![0, 0] },
        ];
        r.index_register(b, entries.clone()).expect("登记");
        let per_block = 2 * 2 * 2; // 条目 × dim × f16
        assert_eq!(r.index_bytes(), per_block);
        let rep = r.report();
        assert_eq!(rep.tiers.host_used, per_block, "索引进 host 层账");
        assert!(
            rep.tiers.rows.iter().any(|x| x.kind == AssetKind::Index && x.bytes == per_block),
            "报告应有 Index 行:{:?}",
            rep.tiers.rows
        );
        // 形状错 = 响亮拒(不进账)
        assert!(r.index_register(ChainHash::new(43), vec![entries[0].clone()]).is_err());
        assert_eq!(r.index_bytes(), per_block, "拒后账不动");
        // 出账
        r.index_drop(b).expect("出账");
        assert_eq!(r.index_bytes(), 0);
        assert_eq!(r.report().tiers.host_used, 0, "出账后 host 层归零");
    }

    #[test]
    fn strict_turn_is_rejected_when_vram_short() {
        let r = rel(1);
        let v = r.check(TurnFloor::Strict, 262_144);
        assert!(matches!(v, AdmissionVerdict::Reject { .. }), "得 {v:?}");
        let v = r.check(TurnFloor::Pages(2048), 262_144);
        assert!(matches!(v, AdmissionVerdict::Admit { .. }), "有耐受 ⇒ 准入,得 {v:?}");
    }
}
