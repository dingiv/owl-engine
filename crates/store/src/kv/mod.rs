//! attention 块腿(`KvStore`)—— **槽池 + 内容身份前缀树**的组合(设计卷 §3.1)。
//!
//! 本模块是 S3 第一刀的落点:把 engine 现行**两本账**
//! (`blocks.rs::BlockManager` 的槽引用/自由队列 + `prefix_cache.rs` 的内容前缀树)
//! 合并成一个自洽对象,并把身份从**物理块号**切到**内容哈希**(`id.rs` 身份律)。
//!
//! # 为什么必须合并 + 换身份
//!
//! 1. **合并**:两本账原本各持一半真相 —— 前缀树存"内容 → 物理块号",
//!    槽池存"物理块 → 引用数";驱逐时要两边对账(engine 现版靠调用方
//!    `for id in evicted { decref(id) }` 手工配对)。合并后绑定引用由本对象自持,
//!    驱逐/登记自成平衡(S3 的 host 卸载要在同一个动作里"摘绑定 + 还槽",
//!    两本账更是必须同屋)。
//! 2. **换身份**:物理槽是可回收资源 —— 卸载(槽还给池)+ 复用后,任何
//!    以槽号为键的东西都会指向新内容。host 层(以及 GDN 快照键)必须以
//!    **内容身份**记账,否则假命中。(引擎侧现存隐患见 `id.rs` 头注。)
//!
//! # 与 engine 现版的等价承诺(HARD)
//!
//! 分配序(FIFO 复用)、驱逐序(叶 LRU + 惰性校验)、前缀匹配语义(逐块父链、
//! 指纹防碰撞、断链即停)、压力驱逐轮次(≤4 轮)全部**逐条对齐** `BlockManager`
//! + `PrefixCache` 的现役行为 —— 接线刀的门就是"行为不变"(见卷内 §8 S3-2)。
//!
//! # 宿主承载(本刀只立形态,搬运在下一刀)
//!
//! 树条目持 `slot: Option<u32>`:`Some` = 设备上有副本;`None` = 内容在 host 层
//! (`Rel::offload_block` 之后)。匹配面对未绑定格返回 `slot = None`,
//! 上层据此在**段边界**回灌(`arch.md` 律③)。

pub mod chain;
pub mod slot;
pub mod tree;

use slot::SlotPool;

use crate::error::{StoreError, StoreResult};
use crate::id::ChainHash;
use crate::policy::{EvictCandidate, Leg};
use crate::tier::Tier;
use tree::PrefixTree;

pub use chain::{hash_block, tail_hash, ChainCursor};
pub use tree::{MatchedBlock, PrefixMatch};

/// 登记/驱逐的结果(上层据此维护跨腿台账 [`crate::tier::TierLedger`]）。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheUpdate {
    /// 新进缓存(设备绑定 +1)的块身份。
    pub inserted: Vec<ChainHash>,
    /// 因容量被逐出的条目(内容身份, 设备槽 = `None` 表示它本来就在 host 层;
    /// 上层须为 host 条目丢掉宿主副本)。
    pub evicted: Vec<(ChainHash, Option<u32>)>,
}

impl CacheUpdate {
    /// 是否无变化。
    pub fn is_empty(&self) -> bool {
        self.inserted.is_empty() && self.evicted.is_empty()
    }
}

/// attention 块腿统计(报告面)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KvStats {
    /// 槽总容量。
    pub slots_total: usize,
    /// 自由槽数。
    pub slots_free: usize,
    /// 缓存驻留块数(树条目)。
    pub cached_blocks: usize,
    /// 其中 host 承载的块数(`slot = None`)。
    pub host_blocks: usize,
    /// 缓存驻留字节(GPU 绑定块 × 单块字节)。
    pub gpu_bytes: usize,
    /// 命中次数(块级)。
    pub hits: u64,
    /// 未命中次数(查询级)。
    pub misses: u64,
    /// 累计被逐块数(容量驱逐)。
    pub evicted: u64,
}

/// attention 块腿(槽池 + 前缀树)。
pub struct KvStore {
    pool: SlotPool,
    tree: PrefixTree,
    block_tokens: usize,
    block_bytes: usize,
    hits: u64,
    misses: u64,
    evicted: u64,
}

impl KvStore {
    /// 构造。`slots` = 池块容量;`block_tokens` = 逻辑块 token 数(本引擎 = 32);
    /// `block_bytes` = 单块字节(**真源在调用方** —— 见坑册 §42);
    /// `cache_cap` = 前缀缓存块数上限(0 = 关缓存,树不匹配不登记)。
    pub fn new(slots: usize, block_tokens: usize, block_bytes: usize, cache_cap: usize) -> Self {
        Self {
            pool: SlotPool::new(slots),
            tree: PrefixTree::new(block_tokens, cache_cap),
            block_tokens,
            block_bytes,
            hits: 0,
            misses: 0,
            evicted: 0,
        }
    }

    /// 单块字节(报告/搬运的长度真源)。
    pub fn block_bytes(&self) -> usize {
        self.block_bytes
    }

    /// 单块 token 数。
    pub fn block_tokens(&self) -> usize {
        self.block_tokens
    }

    /// 池容量(槽数)。
    pub fn slots_total(&self) -> usize {
        self.pool.capacity()
    }

    /// 自由槽数。
    pub fn slots_free(&self) -> usize {
        self.pool.free()
    }

    /// 缓存启用?
    pub fn cache_enabled(&self) -> bool {
        self.tree.enabled()
    }

    /// 覆盖 `target_len` token 的块表长度。
    fn blocks_for(&self, target_len: usize) -> usize {
        target_len.div_ceil(self.block_tokens)
    }

    /// 块表增长至覆盖 `target_len` token(**只增不减、幂等、事务性**)。
    ///
    /// 池不足时先逐前缀缓存叶(≤4 轮,与 engine 现版同),仍不足 = 结构性
    /// 响亮拒,且**不留半截块表**(调用方可安全重试)。
    ///
    /// 返回:被逐出的缓存条目(内容身份, 设备槽 `None` = host 承载)。
    /// 上层须为 host 条目丢掉 host 副本(`Rel` 职责 —— store 不做跨层副作用)。
    pub fn ensure_for_len(
        &mut self,
        table: &mut Vec<u32>,
        target_len: usize,
    ) -> StoreResult<Vec<(ChainHash, Option<u32>)>> {
        let need = self.blocks_for(target_len);
        if need <= table.len() {
            return Ok(Vec::new());
        }
        let delta = need - table.len();
        let mut evicted = Vec::new();
        if self.pool.free() < delta {
            let mut passes = 0;
            while self.pool.free() < delta && passes < 4 {
                let shortage = delta - self.pool.free();
                // 腾**设备槽**只逐设备叶:host 叶被逐不腾槽却丢内容(§16 定谳)
                let batch = self.tree.evict_device(shortage);
                if batch.is_empty() {
                    break;
                }
                for (chain, slot) in &batch {
                    if let Some(slot) = slot {
                        // 缓存那一份引用随绑定一起还(会话持有的引用不动)
                        self.pool.decref(*slot)?;
                    }
                    let _ = chain;
                }
                self.evicted += batch.len() as u64;
                evicted.extend(batch);
                passes += 1;
            }
        }
        if self.pool.free() < delta {
            return Err(StoreError::Exhausted(format!(
                "KV 块池耗尽:需 {need} 块,空闲 {},table={} —— 缓存压力驱逐后仍不足 \
                 = 真实容量不足(降并发/上下文,或调大池容量)",
                self.pool.free(),
                table.len()
            )));
        }
        for _ in 0..delta {
            table.push(self.pool.acquire()?);
        }
        Ok(evicted)
    }

    /// 归还会话整条块表(逆序减引用;共享块自然只剩减引用)。
    pub fn release_table(&mut self, table: &mut Vec<u32>) -> StoreResult<()> {
        for &slot in table.iter().rev() {
            self.pool.decref(slot)?;
        }
        table.clear();
        Ok(())
    }

    /// 取一个**新槽**(引用置 1;调用方即持有者)—— 回灌落位用。
    pub fn alloc_slot(&mut self) -> StoreResult<u32> {
        self.pool.acquire()
    }

    /// 查一块的**设备绑定**(搬出用):必须存在且 `slot = Some`。
    pub fn bound_slot(&self, chain: ChainHash) -> StoreResult<u32> {
        let e = self
            .tree
            .get(chain)
            .ok_or_else(|| StoreError::Unknown(format!("块 {chain:?} 不在缓存树里")))?;
        e.slot
            .ok_or_else(|| StoreError::Invariant(format!("块 {chain:?} 无设备绑定(已在 host 层)")))
    }

    /// **换出前置检查**(失败快于搬运):条目在 + 设备绑定 + 引用数 = 1(缓存独占)。
    /// 还有会话在用 = 活跃面,响亮拒(设计卷 §10.4 律⑤)。
    pub fn ensure_offloadable(&self, chain: ChainHash) -> StoreResult<u32> {
        let slot = self.bound_slot(chain)?;
        let refs = self.pool.refs(slot);
        if refs != 1 {
            return Err(StoreError::Busy(format!(
                "块 {chain:?}(槽 {slot})仍被 {refs} 个持有者引用,不可下沉(活跃面)"
            )));
        }
        Ok(slot)
    }

    /// 某槽当前引用数(驱逐前置检查用)。
    pub fn slot_refs(&self, slot: u32) -> u32 {
        self.pool.refs(slot)
    }

    /// 引用加一(前缀复用/共享)。
    pub fn incref(&mut self, slot: u32) -> StoreResult<()> {
        self.pool.incref(slot)
    }

    /// 引用减一(归零 = 槽回自由队列)。
    pub fn decref(&mut self, slot: u32) -> StoreResult<()> {
        let _ = self.pool.decref(slot)?;
        Ok(())
    }

    /// 命中查询(纯查询 + 触达;块级命中/未命中计数入账)。
    pub fn match_prefix(&mut self, tokens: &[u32]) -> PrefixMatch {
        let m = self.tree.matched_blocks(tokens);
        if m.matched_blocks > 0 {
            self.hits += m.matched_blocks as u64;
        } else {
            self.misses += 1;
        }
        m
    }

    /// 登记一条链的前缀(**整块才登记**;`table` = 该会话的逻辑块 → 槽)。
    ///
    /// 新进树的条目 ⇒ 缓存持一份绑定引用(`incref`);被容量逐出的条目 ⇒
    /// 释放绑定引用(`decref`)。引用配对全在本方法内自洽 —— 调用方不再手工配。
    pub fn register_chain(
        &mut self,
        tokens: &[u32],
        table: &[u32],
    ) -> StoreResult<CacheUpdate> {
        let up = self.tree.insert(tokens, table);
        let mut out = CacheUpdate::default();
        for (chain, slot) in &up.inserted {
            self.pool.incref(*slot)?;
            out.inserted.push(*chain);
        }
        for (chain, slot) in &up.evicted {
            if let Some(slot) = slot {
                self.pool.decref(*slot)?;
            }
            self.evicted += 1;
            out.evicted.push((*chain, *slot));
        }
        Ok(out)
    }

    /// 记一次命中/未命中(身份查询路径;`match_prefix` 自带计数 ——
    /// 同一次查询不要两条路径都调)。
    pub fn note_probe(&mut self, hit: bool) {
        if hit {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
    }

    /// 按容量驱逐 `n` 个叶(压力面/策略面共用);返回被逐条目。
    pub fn evict_cached(&mut self, n: usize) -> StoreResult<Vec<(ChainHash, Option<u32>)>> {
        let batch = self.tree.evict(n);
        for (_chain, slot) in &batch {
            if let Some(slot) = slot {
                self.pool.decref(*slot)?;
            }
        }
        self.evicted += batch.len() as u64;
        Ok(batch)
    }

    /// 驱逐候选(**叶 + 未被保护**;两腿候选合并由关系层做)。
    pub fn evict_candidates(&self, protect: &[ChainHash]) -> Vec<EvictCandidate> {
        self.tree
            .leaves()
            .into_iter()
            .filter(|(chain, _, _)| !protect.contains(chain))
            .map(|(chain, last_use, slot)| EvictCandidate {
                leg: Leg::Kv,
                tier: if slot.is_some() { Tier::Gpu } else { Tier::Host },
                bytes: self.block_bytes,
                last_use,
                chain: Some(chain),
                snap: None,
            })
            .collect()
    }

    /// 定位一块(内容身份 → 所在层)。
    pub fn locate(&self, chain: ChainHash) -> Option<Tier> {
        self.tree.get(chain).map(|e| if e.slot.is_some() { Tier::Gpu } else { Tier::Host })
    }

    /// **链级可下沉评估**(§14.6 律):整条链(从根到 `tail`)能不能一起下沉,
    /// 返回**设备上可搬的块数**(`None` = 本链不该出票)。
    ///
    /// 判据(逐块;任一不满足即 `None` —— **半截链 = 不可复用碎片,禁**):
    /// 1. 链上每块都在树里(身份链完整);
    /// 2. **设备块**引用 = 1(缓存独占;会话在持 = 活跃面 ⇒ 整条不出票);
    /// 3. 已在宿主的块幂等跳过(同链/他链已搬走它);全链都在宿主 ⇒ `None`
    ///    (无槽可腾,别出空票)。
    pub fn chain_down_span(&self, tail: ChainHash) -> Option<usize> {
        let blocks = self.tree.chain_of(tail);
        if blocks.is_empty() {
            return None; // 链尾不在树里
        }
        let mut span = 0usize;
        for b in &blocks {
            if let Some(slot) = b.slot {
                if self.pool.refs(slot) != 1 {
                    return None; // 活跃面:整条链不出票
                }
                span += 1;
            }
        }
        (span > 0).then_some(span)
    }

    /// 链块序(逻辑序,根→尾;`slot = None` = 已在宿主层)。
    pub fn chain_of(&self, tail: ChainHash) -> Vec<MatchedBlock> {
        self.tree.chain_of(tail)
    }

    /// 链长(该身份回到根的块数;含 host 格)。
    pub fn chain_len(&self, tail: ChainHash) -> usize {
        self.tree.chain_of(tail).len()
    }

    /// 从零算一条 token 链的链尾身份(一次性;热路径用 [`ChainCursor`])。
    pub fn tail_hash(&self, tokens: &[u32], full_blocks: usize) -> Option<ChainHash> {
        chain::tail_hash(tokens, self.block_tokens, full_blocks)
    }

    /// **摘绑定下沉**(host 卸载的一半:摘绑定 + 还槽)。
    ///
    /// 只允许**缓存独占**的块被下沉(引用数 = 1):还有会话在用 = 响亮拒
    /// (`arch.md` 律④:活跃面的块不许被搬走)。
    pub fn unbind(&mut self, chain: ChainHash) -> StoreResult<()> {
        let Some(entry) = self.tree.get(chain) else {
            return Err(StoreError::Unknown(format!("块 {chain:?} 不在缓存树里")));
        };
        let Some(slot) = entry.slot else {
            return Err(StoreError::Invariant(format!("块 {chain:?} 已无设备绑定")));
        };
        if self.pool.refs(slot) != 1 {
            return Err(StoreError::Invariant(format!(
                "块 {chain:?}(槽 {slot})仍被 {} 个持有者引用,不可下沉(先清理活跃面)",
                self.pool.refs(slot)
            )));
        }
        let _ = self.tree.set_slot(chain, None);
        self.pool.decref(slot)?;
        Ok(())
    }

    /// **重绑回灌**(host → 设备的一半:绑槽 + 缓存持一份引用)。
    pub fn rebind(&mut self, chain: ChainHash, slot: u32) -> StoreResult<()> {
        match self.tree.set_slot(chain, Some(slot)) {
            None => Err(StoreError::Unknown(format!("块 {chain:?} 不在缓存树里"))),
            Some(Some(_)) => Err(StoreError::Invariant(format!("块 {chain:?} 已有设备绑定"))),
            Some(None) => {
                self.pool.incref(slot)?;
                Ok(())
            }
        }
    }

    /// 丢弃一条缓存条目(host 副本已由上层处理;返回是否真的移除)。
    pub fn drop_entry(&mut self, chain: ChainHash) -> StoreResult<Option<Option<u32>>> {
        let removed = self.tree.remove(chain);
        if let Some(Some(slot)) = removed {
            self.pool.decref(slot)?;
        }
        Ok(removed)
    }

    /// 统计(报告面)。
    pub fn stats(&self) -> KvStats {
        let cached_blocks = self.tree.len();
        let bound = self.tree.bound_count();
        KvStats {
            slots_total: self.pool.capacity(),
            slots_free: self.pool.free(),
            cached_blocks,
            host_blocks: cached_blocks.saturating_sub(bound),
            gpu_bytes: bound * self.block_bytes,
            hits: self.hits,
            misses: self.misses,
            evicted: self.evicted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    fn kv(slots: usize, cap: usize) -> KvStore {
        KvStore::new(slots, crate::id::TOKENS_PER_BLOCK, 1 << 20, cap)
    }

    #[test]
    fn alloc_grow_release_roundtrip_matches_engine_semantics() {
        let mut m = kv(4, 0);
        let mut table = Vec::new();
        m.ensure_for_len(&mut table, 1).expect("1 tok");
        assert_eq!(table.len(), 1, "1 token 也要 1 块");
        assert_eq!(m.slots_free(), 3);
        m.ensure_for_len(&mut table, 64).expect("64 tok");
        assert_eq!(table.len(), 2, "64 tok @ page32 = 2 块");
        m.ensure_for_len(&mut table, 64).expect("幂等");
        assert_eq!(m.slots_free(), 2);
        m.release_table(&mut table).expect("归还");
        assert!(table.is_empty());
        assert_eq!(m.slots_free(), 4);
    }

    #[test]
    fn exhaustion_is_loud_and_transactional() {
        let mut m = kv(2, 0);
        let mut table = Vec::new();
        let err = m.ensure_for_len(&mut table, 100).expect_err("池不足必须拒");
        assert!(matches!(err, StoreError::Exhausted(_)), "得 {err:?}");
        assert!(format!("{err}").contains("真实容量不足"), "{err}");
        assert!(table.is_empty(), "事务性:失败不留半截块表");
        assert_eq!(m.slots_free(), 2);
    }

    #[test]
    fn refcount_shared_blocks_free_on_last() {
        let mut m = kv(4, 0);
        let mut a = Vec::new();
        m.ensure_for_len(&mut a, 32).expect("A");
        let shared = a[0];
        m.incref(shared).expect("共享"); // ref = 2: A + B
        m.decref(shared).expect("A 释放"); // ref = 1
        assert_eq!(m.slots_free(), 3, "还有 B 持有 ⇒ 不回池");
        m.decref(shared).expect("B 释放");
        assert_eq!(m.slots_free(), 4, "归零才回池");
    }

    #[test]
    fn pressure_eviction_frees_cache_held_blocks() {
        let mut m = kv(8, 8);
        let mut ta = Vec::new();
        m.ensure_for_len(&mut ta, 8 * 32).expect("A 建链");
        let toks_a = toks(8 * 32);
        let up = m.register_chain(&toks_a, &ta).expect("登记缓存");
        assert_eq!(up.inserted.len(), 8);
        m.release_table(&mut ta).expect("归还 A(缓存持引 ⇒ 块不回池)");
        assert_eq!(m.slots_free(), 0, "缓存应持有全部 8 块");

        let mut tb = Vec::new();
        let evicted = m.ensure_for_len(&mut tb, 8 * 32).expect("压力驱逐后放行");
        assert_eq!(evicted.len(), 8, "逐出的正是 A 的 8 块");
        assert_eq!(tb.len(), 8);
        let err = m.ensure_for_len(&mut tb, 9 * 32).expect_err("真耗尽");
        assert!(format!("{err}").contains("真实容量不足"), "{err}");
    }

    #[test]
    fn cache_hit_is_content_addressed_and_survives_slot_recycling() {
        let mut m = kv(4, 8);
        let toks_a = toks(64);
        let mut ta = Vec::new();
        m.ensure_for_len(&mut ta, 64).expect("A 建链");
        m.register_chain(&toks_a, &ta).expect("登记");
        let tail = m.tail_hash(&toks_a, 2).expect("尾身份");
        assert_eq!(m.locate(tail), Some(Tier::Gpu));
        m.release_table(&mut ta).expect("A 归还");

        // 复现"槽被回收 + 复用"：登记后把 A 的块逐出(等价于槽被回收)
        let evicted = m.evict_cached(2).expect("逐缓存");
        assert_eq!(evicted.len(), 2);
        assert_eq!(m.slots_free(), 4, "槽真的回池了");
        assert_eq!(m.locate(tail), None, "条目一起消失 ⇒ 不会假命中");

        // 同一段内容再来一次：重新登记,身份一致(内容地址,与槽号无关)
        let mut tb = Vec::new();
        m.ensure_for_len(&mut tb, 64).expect("B 建链");
        let m2 = m.match_prefix(&toks_a);
        assert_eq!(m2.matched_blocks, 0, "已被逐 ⇒ 未命中(不是假命中)");
        m.register_chain(&toks_a, &tb).expect("重新登记");
        let m3 = m.match_prefix(&toks_a);
        assert_eq!(m3.matched_blocks, 2, "同内容 ⇒ 命中,且槽号可能已换");
        assert_eq!(m3.tail, Some(tail), "内容身份不随槽号变");
    }

    #[test]
    fn host_binding_is_invisible_to_slots_but_keeps_identity() {
        let mut m = kv(4, 8);
        let toks_a = toks(32);
        let mut ta = Vec::new();
        m.ensure_for_len(&mut ta, 32).expect("建链");
        m.register_chain(&toks_a, &ta).expect("登记");
        let tail = m.tail_hash(&toks_a, 1).expect("尾身份");
        m.release_table(&mut ta).expect("会话归还(缓存独占)");

        m.unbind(tail).expect("下沉");
        assert_eq!(m.locate(tail), Some(Tier::Host), "层变 host");
        assert_eq!(m.slots_free(), 4, "槽已还池");
        assert_eq!(m.match_prefix(&toks_a).matched_blocks, 1, "host 条目仍命中");
        assert_eq!(m.match_prefix(&toks_a).blocks[0].slot, None, "设备上没有副本");

        // 回灌:会话拿一个**新的**槽(旧槽早已被复用),再重绑
        let mut table = Vec::new();
        m.ensure_for_len(&mut table, 32).expect("分配新槽");
        m.rebind(tail, table[0]).expect("重绑");
        let hit = m.match_prefix(&toks_a);
        assert_eq!(hit.blocks[0].slot, Some(table[0]), "绑定换成新槽,身份不变");
        m.release_table(&mut table).expect("收拾");
    }

    #[test]
    fn unbind_refuses_while_a_session_holds_the_block() {
        let mut m = kv(4, 8);
        let toks_a = toks(32);
        let mut ta = Vec::new();
        m.ensure_for_len(&mut ta, 32).expect("建链");
        m.register_chain(&toks_a, &ta).expect("登记"); // 会话 + 缓存 = refs 2
        let tail = m.tail_hash(&toks_a, 1).expect("尾身份");
        let err = m.unbind(tail).expect_err("活跃面不可下沉");
        assert!(matches!(err, StoreError::Invariant(_)), "得 {err:?}");
        assert!(format!("{err}").contains("持有者"), "{err}");
    }

    /// **槽复用不撞快照键(回归,迁自 engine `blocks.rs`)** —— 该用例随双份账房
    /// 删除而迁入接手侧:物理槽会被回收复用;快照键(游标)与缓存身份(树)都走
    /// **内容身份** ⇒ 复用槽的不同内容不产生同一键(旧实现键 = 物理槽号 ⇒ 撞 = 假命中)。
    #[test]
    fn recycled_slots_do_not_forge_snapshot_keys() {
        let bs = 4usize; // 本测用小页(4 token),池 4 槽
        let mut m = KvStore::new(4, bs, 1 << 20, 4);
        let toks_a: Vec<u32> = (0..8).collect(); // 2 块
        let mut ta = Vec::new();
        m.ensure_for_len(&mut ta, 8).expect("A 块表");
        m.register_chain(&toks_a, &ta).expect("A 登记缓存");
        let a_tail_slot = *ta.last().expect("A 尾块");
        m.release_table(&mut ta).expect("A 会话归还(缓存持引)");

        // B:更大的链压进来 ⇒ 缓存被逐、槽回池并被复用
        let toks_b: Vec<u32> = (500..516).collect(); // 4 块
        let mut tb = Vec::new();
        m.ensure_for_len(&mut tb, 16).expect("B 块表(压力驱逐放行)");
        assert!(tb.contains(&a_tail_slot), "前提:槽 {a_tail_slot} 确实被复用(tb={tb:?})");
        m.register_chain(&toks_b, &tb).expect("B 登记缓存");

        // 身份:游标 ≡ 缓存树,且不随槽号走
        let pm = m.match_prefix(&toks_b);
        assert_eq!(pm.matched_blocks, tb.len(), "B 全链命中");
        let mut cur_b = ChainCursor::new();
        let key_b = cur_b.advance(bs, &toks_b, tb.len()).expect("B 链");
        assert_eq!(key_b, pm.blocks[tb.len() - 1].chain, "游标 ≡ 缓存身份");

        let mut cur_a = ChainCursor::new();
        let key_a = cur_a.advance(bs, &toks_a, 2).expect("A 链");
        assert_ne!(key_a, key_b, "槽复用不得制造同一快照键(旧实现会撞)");
    }

    /// **身份一致性(HARD)**:游标(增量)算出的链尾身份必须与缓存树条目的
    /// 身份**逐位相等** —— 快照键(游标)与跨会话匹配(树)认同一条身份链。
    /// 这是 S3-2 接线后 engine 两处身份不漂移的机器门。
    #[test]
    fn cursor_identity_equals_cache_entry_identity() {
        let mut m = kv(4, 8);
        let tokens = toks(64);
        let mut table = Vec::new();
        m.ensure_for_len(&mut table, 64).expect("块表");
        m.register_chain(&tokens, &table).expect("登记");

        let pm = m.match_prefix(&tokens);
        assert_eq!(pm.matched_blocks, 2);
        let mut cur = ChainCursor::new();
        let got = cur.advance(crate::id::TOKENS_PER_BLOCK, &tokens, 2).expect("游标");
        assert_eq!(got, pm.blocks[1].chain, "游标身份 ≡ 缓存条目身份");
        assert_eq!(Some(got), m.tail_hash(&tokens, 2), "游标 ≡ 从零算");
    }

    /// **链级可下沉评估(§14.6)**:整条链要么全可搬、要么不出票。
    #[test]
    fn chain_down_span_walks_the_whole_path() {
        let mut m = kv(8, 8);
        let toks_a = toks(96); // 3 块
        let mut ta = Vec::new();
        m.ensure_for_len(&mut ta, 96).expect("建链");
        m.register_chain(&toks_a, &ta).expect("登记");
        let tail = m.tail_hash(&toks_a, 3).expect("尾身份");
        // 会话仍持有 ⇒ 活跃面 ⇒ 整条不出票
        assert_eq!(m.chain_down_span(tail), None, "活跃面不得出票");
        m.release_table(&mut ta).expect("归还(缓存独占)");
        assert_eq!(m.chain_down_span(tail), Some(3), "整条设备驻留");

        // 沉尾块 ⇒ 设备块数 2,仍可整条沉(幂等跳过已在宿主的块)
        m.unbind(tail).expect("沉尾块");
        assert_eq!(m.chain_down_span(tail), Some(2));

        // 全链沉完 ⇒ 无槽可腾 ⇒ 不出空票
        let blocks = m.chain_of(tail);
        assert_eq!(blocks.len(), 3, "链 = 根→尾三块");
        for b in blocks.iter().rev() {
            if b.slot.is_some() {
                m.unbind(b.chain).expect("沉");
            }
        }
        assert_eq!(m.chain_down_span(tail), None, "全链在宿主 ⇒ 不出票");
        assert_eq!(m.chain_down_span(crate::id::ChainHash::new(0xDEAD)), None, "不在账");
    }

    /// **腾设备槽只逐设备叶**:host 叶被逐腾不出槽、白丢内容(§16 定谳)。
    #[test]
    fn capacity_eviction_skips_host_leaves_so_slots_actually_free() {
        let mut m = kv(3, 8); // 池 3 槽
        // 链 A(2 块):尾块沉 host ⇒ A 只占 1 设备槽
        let toks_a = toks(64);
        let mut ta = Vec::new();
        m.ensure_for_len(&mut ta, 64).expect("A 建链");
        m.register_chain(&toks_a, &ta).expect("A 登记");
        let tail_a = m.tail_hash(&toks_a, 2).expect("A 尾");
        m.release_table(&mut ta).expect("A 归还");
        m.unbind(tail_a).expect("A 尾沉 host");
        assert_eq!(m.locate(tail_a), Some(Tier::Host));
        assert_eq!(m.slots_free(), 2);

        // 链 B(2 块)⇒ 设备槽满
        let toks_b: Vec<u32> = (1000..1064).collect();
        let mut tb = Vec::new();
        m.ensure_for_len(&mut tb, 64).expect("B 建链");
        m.register_chain(&toks_b, &tb).expect("B 登记");
        let tail_b = m.tail_hash(&toks_b, 2).expect("B 尾");
        m.release_table(&mut tb).expect("B 归还");
        assert_eq!(m.slots_free(), 0, "A头 + B两块 = 3 槽满");

        // 增长 1 块 ⇒ 驱逐:A 尾(host 叶,LRU 更老)必须被跳过,逐 B 尾
        let mut tc = Vec::new();
        let evicted = m.ensure_for_len(&mut tc, 32).expect("腾槽后放行");
        assert_eq!(evicted.len(), 1, "只需要一个槽:{evicted:?}");
        assert_eq!(evicted[0].0, tail_b, "逐的是设备叶(B 尾)");
        assert_eq!(m.locate(tail_a), Some(Tier::Host), "host 叶不得被腾槽驱逐逐掉");
        assert_eq!(tc.len(), 1);
    }

    /// 叶级候选面(驱逐/退化下沉)**仍只出叶**且 protect 有效。
    #[test]
    fn evict_candidates_are_leaves_only_and_protect_wins() {
        let mut m = kv(8, 8);
        let toks_a = toks(64);
        let mut ta = Vec::new();
        m.ensure_for_len(&mut ta, 64).expect("建链");
        m.register_chain(&toks_a, &ta).expect("登记");
        let first = m.tail_hash(&toks_a, 1).expect("首块身份");
        let tail = m.tail_hash(&toks_a, 2).expect("尾身份");
        let cands = m.evict_candidates(&[]);
        assert_eq!(cands.len(), 1, "只有叶可逐:{cands:?}");
        assert_eq!(cands[0].chain, Some(tail));
        assert_eq!(cands[0].bytes, 1 << 20);
        assert!(m.evict_candidates(&[tail]).is_empty(), "保护面绝不入候选");
        assert_ne!(first, tail, "两块身份不同");
    }
}
