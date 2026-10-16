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
use crate::kv::{CacheUpdate, KvStats, KvStore, PrefixMatch};
use crate::policy::{AdmissionPolicy, EvictCandidate, EvictPolicy, Leg, LruPolicy};
use crate::state::{StateStats, StateStore};
use crate::tier::{AssetKind, Tier, TierLedger, TierReport};
use owl_iface::contract::{Bytes, DeviceClient};

/// KV 腿几何(boot 装配一次;`block_tokens` 本引擎恒 32 —— 设计卷 §3.6.2)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KvConfig {
    /// 池块容量(**管理面**页数;不含 padding dump 页)。
    pub slots: usize,
    /// 单块 token 数。
    pub block_tokens: usize,
    /// 单块字节(**真源**;见坑册 §42)。
    pub block_bytes: usize,
    /// 前缀缓存块数上限(0 = 关缓存)。
    pub cache_cap: usize,
}

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
    /// 弹性面字节(可驱逐)。
    pub elastic_bytes: usize,
    /// 策略名(驱逐序)。
    pub policy: String,
    /// 宿主层驻留字节。
    pub host_resident_bytes: usize,
    /// 宿主层空转缓冲字节(真实 RAM;报告必须算上)。
    pub host_free_bytes: usize,
}

/// 关系层门面:两个独立管理器 + 一层关联。
pub struct Rel {
    /// attention 块腿(槽池 + 内容前缀树)。
    pub kv: KvStore,
    /// GDN 状态账(行 + 快照)。
    pub state: StateStore,
    /// 宿主层(有界 + 可驱逐 + 缓冲复用;vLLM 式 CPU 卸载层)。
    pub host: HostTier,
    /// 三层账。
    pub ledger: TierLedger,
    /// 准入策略(boot 装配)。
    pub policy: AdmissionPolicy,
    /// 活跃 turn 在飞账。
    pub admission: Admission,
    budget: VramBudget,
    evict: Box<dyn EvictPolicy>,
}

impl Rel {
    /// 装配(唯一入口:预算分解失败 = **拒绝启动**,A5.1)。
    pub fn open(
        inputs: BudgetInputs,
        policy: AdmissionPolicy,
        kv_cfg: KvConfig,
        kv_bytes_per_token: usize,
    ) -> StoreResult<Self> {
        if policy.max_active_turns != 0 && inputs.kv_bytes_per_token != kv_bytes_per_token {
            return Err(StoreError::Budget("每 token KV 字节不一致(装配参数矛盾)".into()));
        }
        if kv_cfg.block_bytes == 0 || kv_cfg.block_tokens == 0 || kv_cfg.slots == 0 {
            return Err(StoreError::Budget(format!(
                "KV 腿几何非法:slots={} block_tokens={} block_bytes={}",
                kv_cfg.slots, kv_cfg.block_tokens, kv_cfg.block_bytes
            )));
        }
        let budget = VramBudget::decompose(inputs, policy.max_active_turns)?;
        Ok(Self {
            kv: KvStore::new(kv_cfg.slots, kv_cfg.block_tokens, kv_cfg.block_bytes, kv_cfg.cache_cap),
            host: HostTier::new(kv_cfg.block_bytes, inputs.host_bytes)?,
            state: StateStore::new(policy.max_active_turns, inputs.gdn_row_bytes),
            ledger: TierLedger::new(budget.tier_budget()),
            policy,
            admission: Admission::new(),
            budget,
            evict: Box::new(LruPolicy),
        })
    }

    /// 卡级预算(只读)。
    pub fn budget(&self) -> &VramBudget {
        &self.budget
    }

    /// 换驱逐策略(策略对象可换,判据不散 —— §3.2)。
    pub fn set_evict_policy(&mut self, p: Box<dyn EvictPolicy>) {
        self.evict = p;
    }

    /// **命中查询**(按身份;纯 host、零副作用 —— 律⑥)。
    ///
    /// 与 [`Self::match_prefix`] 二选一使用(两条查询路径各自记一次命中账;
    /// 同一次查询不要两条都调,否则计数双算)。
    pub fn probe(&mut self, id: PrefixId) -> RecallHit {
        let kv_tier = self.kv.locate(id.chain);
        let snap_tier = self.state.locate(id.snap_key());
        let blocks = if kv_tier.is_some() { self.kv.chain_len(id.chain) } else { 0 };
        let hit = RecallHit { kv_tier, snap_tier, blocks };
        self.kv.note_probe(hit.kv_tier.is_some());
        self.state.note_probe(hit.snap_tier.is_some());
        hit
    }

    /// **按 token 命中查询**(engine L1 的第一跳):逐块内容匹配,回块链 + 绑定。
    ///
    /// 未绑定的格(`slot = None`)= 内容在宿主层,调用方须在**段边界**回灌
    /// (律③)—— 本方法自身不做设备动作。
    pub fn match_prefix(&mut self, tokens: &[u32]) -> (PrefixMatch, RecallHit) {
        let m = self.kv.match_prefix(tokens);
        let kv_tier = m.tail.and_then(|t| self.kv.locate(t));
        let snap_tier = m.tail.and_then(|t| self.state.locate(SnapKey::at_boundary(t)));
        let hit = RecallHit { kv_tier, snap_tier, blocks: m.matched_blocks };
        (m, hit)
    }

    /// **登记一条前缀链**(turn 收口 L3):缓存取一份绑定引用,容量驱逐同步出账。
    ///
    /// 只做 KV 腿;快照腿由 [`StateStore::return_row`] 登记(两腿独立 —— 裁决②)。
    pub fn register_chain(
        &mut self,
        tokens: &[u32],
        table: &[u32],
    ) -> StoreResult<CacheUpdate> {
        let up = self.kv.register_chain(tokens, table)?;
        let b = self.kv.block_bytes();
        for _ in &up.inserted {
            self.ledger.add(Tier::Gpu, AssetKind::KvBlock, b)?;
        }
        self.account_evicted(&up.evicted)?;
        Ok(up)
    }

    /// 块表增长(engine 每步/每 turn 调;内部含压力驱逐,逐出的宿主条目
    /// 由门面**同步丢弃宿主副本**)。
    pub fn ensure_for_len(
        &mut self,
        table: &mut Vec<u32>,
        target_len: usize,
    ) -> StoreResult<()> {
        let evicted = self.kv.ensure_for_len(table, target_len)?;
        self.account_evicted(&evicted)
    }

    /// 归还会话块表。
    pub fn release_table(&mut self, table: &mut Vec<u32>) -> StoreResult<()> {
        self.kv.release_table(table)
    }

    /// 按策略点名的驱逐(engine 在段边界执行):逐出缓存条目并出账。
    pub fn evict_cached(&mut self, n: usize) -> StoreResult<Vec<ChainHash>> {
        let batch = self.kv.evict_cached(n)?;
        let mut chains = Vec::with_capacity(batch.len());
        for (chain, slot) in batch {
            match slot {
                Some(_) => self.ledger.remove(Tier::Gpu, AssetKind::KvBlock, self.kv.block_bytes())?,
                None => {
                    let _ = self.host.evict(chain)?;
                    self.ledger.remove(Tier::Host, AssetKind::KvBlock, self.kv.block_bytes())?;
                }
            }
            chains.push(chain);
        }
        Ok(chains)
    }

    /// **卸载一个 attention 块到宿主层**(L2 换出的一半;段边界调用)。
    ///
    /// 顺序 = 取宿主缓冲 → `dtoh` → **摘设备绑定 + 还槽**(引用必须为 1;
    /// 仍有会话在用 = 响亮拒,`arch.md` 律④)→ 落宿主 → 账目 `Gpu → Host`。
    /// 失败不留半态(绑定未摘则宿主缓冲直接回收)。
    pub async fn offload_block<D: DeviceClient>(
        &mut self,
        face: &mut D,
        chain: ChainHash,
        dev: &Bytes,
    ) -> StoreResult<()> {
        let bytes = self.kv.block_bytes();
        let mut buf = self.host.acquire()?;
        crate::moves::offload_to_host(face, dev, bytes, &mut buf).await?;
        self.kv.unbind(chain)?;
        self.host.put(chain, buf)?;
        self.ledger.remove(Tier::Gpu, AssetKind::KvBlock, bytes)?;
        self.ledger.add(Tier::Host, AssetKind::KvBlock, bytes)?;
        Ok(())
    }

    /// **从宿主层回灌一个块到设备**(段边界调用;`slot` 由调用方在设备上备好)。
    pub async fn onload_block<D: DeviceClient>(
        &mut self,
        face: &mut D,
        chain: ChainHash,
        slot: u32,
        dev: &Bytes,
    ) -> StoreResult<()> {
        let bytes = self.kv.block_bytes();
        let buf = self.host.take(chain)?;
        crate::moves::onload_from_host(face, &buf, dev, bytes).await?;
        self.kv.rebind(chain, slot)?;
        self.ledger.remove(Tier::Host, AssetKind::KvBlock, bytes)?;
        self.ledger.add(Tier::Gpu, AssetKind::KvBlock, bytes)?;
        Ok(())
    }

    /// 逐出条目的跨腿出账(KV 腿已自行摘绑定;此处只对账 + 丢宿主副本)。
    fn account_evicted(&mut self, evicted: &[(ChainHash, Option<u32>)]) -> StoreResult<()> {
        let b = self.kv.block_bytes();
        for (chain, slot) in evicted {
            match slot {
                Some(_) => self.ledger.remove(Tier::Gpu, AssetKind::KvBlock, b)?,
                None => {
                    // host 条目被逐 = 内容整体消失:宿主副本必须同步丢掉
                    let _ = self.host.evict(*chain)?;
                    self.ledger.remove(Tier::Host, AssetKind::KvBlock, b)?;
                }
            }
        }
        Ok(())
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
        cands.extend(self.kv.evict_candidates(&protect.chains));
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

    /// 报告(三层 + 两腿 + 保留/弹性面)。
    pub fn report(&self) -> StoreReport {
        StoreReport {
            tiers: self.ledger.report(),
            kv: self.kv.stats(),
            state: self.state.stats(),
            active_turns: self.admission.active_turns(),
            reserved_bytes: self.admission.reserved_bytes(),
            elastic_bytes: self.admission.elastic_bytes(),
            policy: self.evict.name().to_string(),
            host_resident_bytes: self.host.used_bytes(),
            host_free_bytes: self.host.free_bytes(),
        }
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
            host_bytes: 16 << 30,
            disk_bytes: 0,
        }
    }

    fn kv_cfg(slots: usize, cache_cap: usize) -> KvConfig {
        KvConfig { slots, block_tokens: 32, block_bytes: BLOCK_BYTES, cache_cap }
    }

    fn rel(turns: usize) -> Rel {
        Rel::open(inputs(), AdmissionPolicy::new(turns, QueuePolicy::Reject), kv_cfg(64, 32), 32 * 1024)
            .expect("装配")
    }

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn open_rejects_unmeasured_budget() {
        let mut i = inputs();
        i.weights_bytes = 0;
        assert!(Rel::open(i, AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(8, 8), 1024).is_err());
        assert!(
            Rel::open(inputs(), AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(0, 8), 1024).is_err(),
            "几何非法也要拒"
        );
    }

    #[test]
    fn recall_semantics_are_the_relation_layer_business() {
        let mut r = rel(2);
        let tokens = toks(64);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 64).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        let tail = r.kv.tail_hash(&tokens, 2).expect("尾身份");
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
        let a = r.kv.tail_hash(&ta, 1).expect("A 身份");
        let b = r.kv.tail_hash(&tb, 1).expect("B 身份");
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
        let chain = r.kv.tail_hash(&tokens, 1).expect("身份");
        r.admission.admit(1, 1 << 20, 0).expect("登记活跃 turn");
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
        r.admission.admit(1, 4 << 20, 1 << 20).expect("活跃 turn");
        let rep = r.report();
        assert_eq!(rep.active_turns, 1);
        assert_eq!(rep.reserved_bytes, 4 << 20);
        assert_eq!(rep.elastic_bytes, 1 << 20);
        assert_eq!(rep.policy, "lru");
        assert_eq!(rep.state.rows_total, 2);
        assert_eq!(rep.kv.cached_blocks, 1, "缓存面 1 块");
        assert_eq!(rep.tiers.gpu_used, BLOCK_BYTES, "台账只记 store 管的缓存块面");
    }

    #[tokio::test]
    async fn kv_block_migrates_gpu_to_host_and_back_with_books_balanced() {
        let mut r = rel(2);
        let mut face = owl_cpu::CpuFace::default();
        let dev = face.alloc(owl_iface::contract::Dtype::F32, BLOCK_BYTES / 4).await.expect("设备块");
        let tokens = toks(32);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 32).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        let chain = r.kv.tail_hash(&tokens, 1).expect("身份");

        // 会话归还 ⇒ 缓存独占 ⇒ 可下沉
        r.release_table(&mut table).expect("归还");
        r.offload_block(&mut face, chain, &dev).await.expect("卸载");
        let hit = r.probe(PrefixId::new(chain));
        assert_eq!(hit.kv_tier, Some(Tier::Host), "层应下沉");
        assert_eq!(hit.mode(), RecallMode::ContextOnly, "状态腿不在 ⇒ 上下文召回");
        assert_eq!(r.report().host_resident_bytes, BLOCK_BYTES);
        assert_eq!(r.report().tiers.gpu_used, 0, "Gpu/KvBlock 已出账");

        // 回灌:拿一个新槽(旧槽已还池)重绑
        let mut fresh = Vec::new();
        r.ensure_for_len(&mut fresh, 32).expect("新槽");
        r.onload_block(&mut face, chain, fresh[0], &dev).await.expect("回灌");
        assert_eq!(r.probe(PrefixId::new(chain)).kv_tier, Some(Tier::Gpu));
        assert_eq!(r.report().host_resident_bytes, 0);
        assert_eq!(r.report().tiers.gpu_used, BLOCK_BYTES);
        assert_eq!(r.report().tiers.host_used, 0);
    }

    #[tokio::test]
    async fn host_tier_full_is_rejected_loudly() {
        // host 预算 = 单块 ⇒ 第二块卸载必拒(先驱逐再搬是调用方的顺序责任)
        let mut i = inputs();
        i.host_bytes = BLOCK_BYTES;
        let mut r = Rel::open(i, AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(8, 8), 32 * 1024)
            .expect("装配");
        let mut face = owl_cpu::CpuFace::default();
        let dev = face.alloc(owl_iface::contract::Dtype::F32, BLOCK_BYTES / 4).await.expect("设备块");
        let ta = toks(32);
        let tb: Vec<u32> = (100..132).collect();
        for t in [&ta, &tb] {
            let mut table = Vec::new();
            r.ensure_for_len(&mut table, 32).expect("块表");
            r.register_chain(t, &table).expect("登记");
            r.release_table(&mut table).expect("归还");
        }
        let a = r.kv.tail_hash(&ta, 1).expect("A 身份");
        let b = r.kv.tail_hash(&tb, 1).expect("B 身份");
        r.offload_block(&mut face, a, &dev).await.expect("第一块");
        let err = r.offload_block(&mut face, b, &dev).await.expect_err("第二块必须拒");
        assert!(matches!(err, StoreError::Exhausted(_)), "得 {err:?}");
        // 拒后不留半态:A 仍在 host,B 仍在设备
        assert_eq!(r.probe(PrefixId::new(a)).kv_tier, Some(Tier::Host));
        assert_eq!(r.probe(PrefixId::new(b)).kv_tier, Some(Tier::Gpu));
    }

    #[tokio::test]
    async fn offload_refuses_while_session_holds_the_block() {
        let mut r = rel(2);
        let mut face = owl_cpu::CpuFace::default();
        let dev = face.alloc(owl_iface::contract::Dtype::F32, BLOCK_BYTES / 4).await.expect("设备块");
        let tokens = toks(32);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 32).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        let chain = r.kv.tail_hash(&tokens, 1).expect("身份");
        // 会话仍持链 ⇒ 活跃面不可下沉(律④)
        let err = r.offload_block(&mut face, chain, &dev).await.expect_err("必须拒");
        assert!(matches!(err, StoreError::Invariant(_)), "得 {err:?}");
        assert_eq!(r.report().tiers.gpu_used, BLOCK_BYTES, "账目未动");
    }

    #[test]
    fn strict_turn_is_rejected_when_vram_short() {
        let r = rel(1);
        let v = r.check(TurnFloor::Strict, 262_144);
        assert!(matches!(v, AdmissionVerdict::Reject { .. }), "得 {v:?}");
        let v = r.check(TurnFloor::Bytes(2 << 30), 262_144);
        assert!(matches!(v, AdmissionVerdict::Admit { .. }), "有耐受 ⇒ 准入,得 {v:?}");
    }
}
