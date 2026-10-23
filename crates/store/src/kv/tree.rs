//! 前缀树(内容身份)—— 迁自 engine `prefix_cache.rs`(E2c),**语义等价是硬门**。
//!
//! 结构 = vLLM v1 前缀缓存同族:**逐块父链哈希**建树,只有**叶**可被驱逐
//! (逐父节点会孤儿化子链)。叶子 LRU 用"惰性校验 + 批量压实"策略
//! (push 时带当时的 access 号,弹出时号对不上 = 陈旧条目,跳过)。
//!
//! 与 engine 版的唯一结构差异:**槽号是绑定,不是身份** ——
//! 条目持 `slot: Option<u32>`(`None` = 内容在 host 层,不在设备上);
//! 槽被回收/复用不影响条目身份(`id.rs` 身份律)。
//!
//! 哈希函数 = [`super::chain`](唯一产地;本模块不再自持副本)。

use std::collections::{HashMap, HashSet, VecDeque};

use super::chain::{content_fingerprint, hash_block};
use crate::id::ChainHash;

/// 树里的一格 = 一个逻辑块(内容身份 + 设备绑定)。
#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    pub(crate) parent: Option<ChainHash>,
    /// 子节点数(> 0 ⇒ 非叶,不可驱逐)。
    pub(crate) children: usize,
    /// 最近触达序号(缓存命中新鲜度)。
    pub(crate) access: u64,
    /// 内容指纹(仅防哈希碰撞;与父链无关的"字面是不是同一段 token")。
    pub(crate) fingerprint: u64,
    /// 设备槽绑定(`None` = 条目在 host 层,设备上没有副本)。
    pub(crate) slot: Option<u32>,
}

/// 命中面的一格(逻辑序)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatchedBlock {
    /// 块的内容身份。
    pub chain: ChainHash,
    /// 设备槽绑定(`None` = 内容只在 host 层)。
    pub slot: Option<u32>,
}

/// 一次命中查询的结果。
#[derive(Clone, Debug, Default)]
pub struct PrefixMatch {
    /// 命中的**整块**数(= `blocks.len()`)。
    pub matched_blocks: usize,
    /// 命中块链(逻辑序,含未绑定的 host 格 —— 上层据此决定是否回灌)。
    pub blocks: Vec<MatchedBlock>,
    /// 链尾块身份(前缀身份的来源)。
    pub tail: Option<ChainHash>,
}

/// 树变更(调用方据此调槽池引用 / 台账)。
#[derive(Clone, Debug, Default)]
pub(crate) struct TreeUpdate {
    /// 新进树的条目(内容身份, 设备槽;`None` = 盘/宿主层重建,无设备绑定)。
    pub(crate) inserted: Vec<(ChainHash, Option<u32>)>,
    /// **重绑定**(2026-10-20):同内容身份已在树里,但本次写到**新槽**
    /// (`旧绑定, 新槽`)。典型来源 = 前缀复用的"复用面裁短"
    /// (`m < matched`):被裁掉的段会**重算并写新槽**,而树里旧条目仍指着老槽
    /// ⇒ 不重绑的话新槽无树引用(`refs==1`)⇒ **结构性不可沉**
    /// (GGUF 262k 联调实测:第二 turn 池被不可沉块填满 ⇒ Exhausted)。
    /// `旧 = None` 表示旧副本在宿主(上层须弃置宿主副本)。
    pub(crate) rebound: Vec<(ChainHash, Option<u32>, u32)>,
    /// 因容量被逐出的条目(内容身份, 设备槽 = `None` 表示它本来就在 host 层)。
    /// **有槽 ≠ 槽被回收**:槽此刻可能仍被会话持有(回收判据在槽池引用账)。
    pub(crate) evicted: Vec<(ChainHash, Option<u32>)>,
}

/// 内容身份前缀树。
pub(crate) struct PrefixTree {
    block_tokens: usize,
    /// 缓存容量(块数;0 = 关缓存)。
    cap: usize,
    entries: HashMap<ChainHash, Entry>,
    leaf_set: HashSet<ChainHash>,
    leaf_lru: VecDeque<(ChainHash, u64)>,
    access: u64,
}

impl PrefixTree {
    /// 构造(容量 0 = 关缓存:不匹配、不登记)。
    pub(crate) fn new(block_tokens: usize, cap: usize) -> Self {
        Self {
            block_tokens,
            cap,
            entries: HashMap::new(),
            leaf_set: HashSet::new(),
            leaf_lru: VecDeque::new(),
            access: 0,
        }
    }

    /// 是否启用(容量 > 0)。
    pub(crate) fn enabled(&self) -> bool {
        self.cap > 0
    }

    /// 条目数(驻留缓存块数)。
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// 设备绑定条目数(`slot = Some`;报告面用)。
    pub(crate) fn bound_count(&self) -> usize {
        self.entries.values().filter(|e| e.slot.is_some()).count()
    }

    /// 逐个设备绑定条目回吐槽号(只读;可动面统计用 —— 只有槽引用账能判"能否搬")。
    pub(crate) fn for_each_bound<F: FnMut(u32)>(&self, mut f: F) {
        for e in self.entries.values() {
            if let Some(slot) = e.slot {
                f(slot);
            }
        }
    }

    /// 取条目(只读)。
    pub(crate) fn get(&self, chain: ChainHash) -> Option<&Entry> {
        self.entries.get(&chain)
    }

    /// 设定绑定(`None` = 下沉到 host;`Some(slot)` = 回灌到设备)。
    pub(crate) fn set_slot(&mut self, chain: ChainHash, slot: Option<u32>) -> Option<Option<u32>> {
        self.entries.get_mut(&chain).map(|e| std::mem::replace(&mut e.slot, slot))
    }

    /// 命中查询(**纯查询面:匹配本身要 touch ⇒ 需要 `&mut self`**)。
    ///
    /// 从第一个整块起逐块走父链;任一环缺哈希或指纹不符即停(前缀语义)。
    pub(crate) fn matched_blocks(&mut self, tokens: &[u32]) -> PrefixMatch {
        if !self.enabled() {
            return PrefixMatch::default();
        }
        let full = tokens.len() / self.block_tokens;
        if full == 0 {
            return PrefixMatch::default();
        }
        let mut matched = 0usize;
        let mut parent = ChainHash::new(0);
        let mut tail = None;
        for block in tokens.chunks(self.block_tokens).take(full) {
            let hash = hash_block(parent, block);
            match self.entries.get(&hash) {
                Some(e) if e.fingerprint == content_fingerprint(block) => {
                    matched += 1;
                    parent = hash;
                    tail = Some(hash);
                    self.touch(hash);
                }
                _ => break,
            }
        }
        let blocks = tail.map(|t| self.chain_of(t)).unwrap_or_default();
        PrefixMatch { matched_blocks: matched, blocks, tail }
    }

    /// 链块序(从根走到 `tail`;逻辑序,含 `slot = None` 的 host 格)。
    pub(crate) fn chain_of(&self, tail: ChainHash) -> Vec<MatchedBlock> {
        let mut out = Vec::new();
        let mut cur = Some(tail);
        while let Some(h) = cur {
            let Some(e) = self.entries.get(&h) else { break };
            out.push(MatchedBlock { chain: h, slot: e.slot });
            cur = e.parent;
        }
        out.reverse();
        out
    }

    /// 登记一条链(`slots[i]` = 第 i 个逻辑块的设备槽;只登记整块)。
    pub(crate) fn insert(&mut self, tokens: &[u32], slots: &[u32]) -> TreeUpdate {
        if !self.enabled() {
            return TreeUpdate::default();
        }
        let full = tokens.len() / self.block_tokens;
        let n = full.min(slots.len());
        if n == 0 {
            return TreeUpdate::default();
        }
        let mut update = TreeUpdate::default();
        let mut base = ChainHash::new(0); // 从根起(全量登记)
        for (block, &slot) in tokens.chunks(self.block_tokens).take(n).zip(slots) {
            let Some(next) = self.insert_one(base, block, Some(slot), &mut update) else {
                break; // 哈希碰撞(指纹不符):当断链处理,响亮地不登记
            };
            base = next;
        }
        self.trim(&mut update);
        update
    }

    /// **无设备绑定的链登记**(S6.1:启动时按盘上索引**重建树**)。
    ///
    /// 语义与 [`Self::insert`] 逐块一致(同一份哈希/指纹/父链),唯一区别是
    /// 新条目 `slot = None` —— 内容在盘上(或宿主),`locate` 归 `Tier::Host`
    /// ⇒ 命中后由 `fill_kv_from_disk` 读回宿主。**不复查内容是否存在**
    /// (调用方已按"内容文件在场"截断过 tokens)。
    pub(crate) fn insert_offloaded(&mut self, tokens: &[u32]) -> TreeUpdate {
        if !self.enabled() {
            return TreeUpdate::default();
        }
        let n = tokens.len() / self.block_tokens;
        let mut update = TreeUpdate::default();
        let mut base = ChainHash::new(0);
        for block in tokens.chunks(self.block_tokens).take(n) {
            let Some(next) = self.insert_one(base, block, None, &mut update) else {
                break;
            };
            base = next;
        }
        self.trim(&mut update);
        update
    }

    /// **增量登记**(B-3e-2 第 2b 项):从**已注册的链尾** `parent0` 续挂新块。
    ///
    /// 与 [`Self::insert`] 的分工:`insert` 从根重走全链(**O(全长)** —— turn 收口
    /// 一次性登记用);本函数只处理"本次新增的整块"(**每块一次哈希 ⇒ 全 turn O(n)**),
    /// 用在**轮内**登记(链一写完就进树 ⇒ 该块才可被逐块下沉票选中:宿主层按树条目键控)。
    ///
    /// `blocks = &[(tokens_of_block, slot)]`(每项长度必须 = `block_tokens`)。
    pub(crate) fn insert_inc(&mut self, parent0: ChainHash, blocks: &[(&[u32], u32)]) -> TreeUpdate {
        if !self.enabled() || blocks.is_empty() {
            return TreeUpdate::default();
        }
        let mut update = TreeUpdate::default();
        let mut base = parent0;
        for (block, slot) in blocks {
            if block.len() != self.block_tokens {
                break; // 非整块 = 结构错(调用方保证;不静默凑数)
            }
            let Some(next) = self.insert_one(base, block, Some(*slot), &mut update) else {
                break;
            };
            base = next;
        }
        self.trim(&mut update);
        update
    }

    /// **单块登记**(两条入口的唯一实现):返回新块身份;`None` = 哈希碰撞(断链)。
    ///
    /// **占位符归一**(2026-10-20):`slot = Some(u32::MAX)` = 该块内容在宿主
    /// ([`crate::leg::HOST_PLACEHOLDER`])⇒ 归一成 `None`("无设备绑定")。
    /// 旧实现把它当**真槽**存进 `Entry.slot` ⇒ 后读 `bound_slot`/`decref`
    /// 拿 u32::MAX 去碰槽池 = 越界响亮拒(虚拟化档 turn 收口必踩:
    /// `register_chain_owned` 的整张表本来就会带占位符)。
    /// 同理，重绑定分支也不得把占位符当真槽绑。
    fn insert_one(
        &mut self,
        base: ChainHash,
        block: &[u32],
        slot: Option<u32>,
        update: &mut TreeUpdate,
    ) -> Option<ChainHash> {
        let slot = slot.filter(|s| *s != crate::leg::HOST_PLACEHOLDER);
        let hash = hash_block(base, block);
        let fingerprint = content_fingerprint(block);
        match self.entries.get(&hash) {
            Some(e) if e.fingerprint == fingerprint => {
                // 已是同一块:触达 + **必要时重绑新槽**(见 `TreeUpdate::rebound`)。
                // 不重绑会让重算出来的新槽没有树引用 ⇒ 永远不可沉。
                if let Some(new_slot) = slot {
                    let old = e.slot;
                    if old != Some(new_slot) {
                        if let Some(em) = self.entries.get_mut(&hash) {
                            em.slot = Some(new_slot);
                        }
                        update.rebound.push((hash, old, new_slot));
                    }
                }
                let access = self.next_access();
                if let Some(e) = self.entries.get_mut(&hash) {
                    e.access = access;
                }
                self.touch_leaf(hash);
            }
            Some(_) => return None,
            None => {
                let parent = if base == ChainHash::new(0) { None } else { Some(base) };
                if let Some(p) = parent {
                    if let Some(pe) = self.entries.get_mut(&p) {
                        if pe.children == 0 {
                            self.leaf_set.remove(&p);
                        }
                        pe.children += 1;
                    }
                }
                let access = self.next_access();
                self.entries.insert(
                    hash,
                    Entry { parent, children: 0, access, fingerprint, slot },
                );
                self.leaf_set.insert(hash);
                self.leaf_lru.push_back((hash, access));
                update.inserted.push((hash, slot));
            }
        }
        Some(hash)
    }

    /// 容量收口(两条入口共用):超 `cap` 的部分按最老叶逐出。
    fn trim(&mut self, update: &mut TreeUpdate) {
        let excess = self.entries.len().saturating_sub(self.cap);
        if excess > 0 {
            update.evicted = self.evict(excess);
        }
    }

    /// 驱逐 `n` 个叶(从最老开始);返回被逐条目(内容身份, 槽 `Option`)。
    ///
    /// 逐出只摘树里的绑定 —— **不判定槽能不能回收**(那是槽池的引用账;
    /// 会话仍持有时槽不回自由队列)。
    ///
    /// ⚠️ 容量语义:先设备叶,不足再 host 叶(`host` 叶本就在宿主层,
    /// 逐它只丢内容)。要**腾设备槽**请用 [`Self::evict_device`]。
    pub(crate) fn evict(&mut self, n: usize) -> Vec<(ChainHash, Option<u32>)> {
        self.evict_impl(n, false)
    }

    /// **只逐设备叶**(腾设备槽语义;§16 定谳:host 叶被逐腾不出槽,白丢内容)。
    pub(crate) fn evict_device(&mut self, n: usize) -> Vec<(ChainHash, Option<u32>)> {
        self.evict_impl(n, true)
    }

    fn evict_impl(&mut self, mut n: usize, device_only: bool) -> Vec<(ChainHash, Option<u32>)> {
        let mut evicted = Vec::new();
        // 本趟被跳过的非设备叶(不删条目 —— 容量驱逐那趟还要它们)
        let mut deferred: Vec<(ChainHash, u64)> = Vec::new();
        // 守卫 = 请求量 + 基线:逐掉一片后父块会**追加**到 leaf_lru 尾部,
        // 深链要能在**同一次调用**里一路走上去(旧 guard 只有 ~10 步,深链一次
        // 只能塌 10 块 ⇒ 容量裁剪要拖很多拍)。
        let mut guard = n + self.leaf_lru.len() * 2 + 8;
        while n > 0 && guard > 0 {
            guard -= 1;
            let Some((hash, access)) = self.leaf_lru.pop_front() else { break };
            if !self.leaf_set.contains(&hash) {
                continue; // 已非叶(长出了子链)
            }
            let Some(e) = self.entries.get(&hash) else { continue };
            if e.access != access || e.children > 0 {
                continue; // 陈旧条目(期间被重新触达)
            }
            if device_only && e.slot.is_none() {
                // host 叶:逐它**腾不出设备槽**(反而白丢内容)⇒ 留账、跳过本趟
                deferred.push((hash, access));
                continue;
            }
            let Entry { parent, slot, .. } = self.entries.remove(&hash).expect("刚查过");
            self.leaf_set.remove(&hash);
            // `slot = None` = host 承载条目被逐:内容整体消失,无设备槽可还;
            // 上层须同步把 host 副本丢掉(见 `KvStore::evict_cached`)
            evicted.push((hash, slot));
            n = n.saturating_sub(1);
            if let Some(p) = parent {
                if let Some(pe) = self.entries.get_mut(&p) {
                    pe.children = pe.children.saturating_sub(1);
                    if pe.children == 0 {
                        self.leaf_set.insert(p);
                        self.leaf_lru.push_back((p, pe.access));
                    }
                }
            }
        }
        self.leaf_lru.extend(deferred);
        evicted
    }

    /// 叶条目(驱逐候选面;调用方按策略排序)。
    pub(crate) fn leaves(&self) -> Vec<(ChainHash, u64, Option<u32>)> {
        self.entries
            .iter()
            .filter(|(h, e)| e.children == 0 && self.leaf_set.contains(h))
            .map(|(h, e)| (*h, e.access, e.slot))
            .collect()
    }

    /// 触达(命中新鲜度)。
    pub(crate) fn touch(&mut self, chain: ChainHash) {
        let access = self.next_access();
        if let Some(e) = self.entries.get_mut(&chain) {
            e.access = access;
        }
        self.touch_leaf(chain);
    }

    /// **可整枝塌缩的块数**(前缀无感删除/整枝驱逐用):从 `tail` 往上,只要
    /// 每一层都**没有兄弟**(父的 `children == 1`)就继续;遇到分叉点(父有别的
    /// 孩子)立即停 —— **分叉点以上是共享前缀,永远不动**。
    ///
    /// 语义 = "从叶往根连续可逐的块数"(含 `tail` 自己;`tail` 不在树/不是叶 ⇒ 0)。
    /// 执行侧的 [`KvStore::prune_branch`] 会再逐块校验引用(活跃会话持有则停),
    /// 所以本函数只做**结构**判定(纯读,零副作用)。
    pub(crate) fn collapsible_span(&self, tail: ChainHash) -> usize {
        let Some(e) = self.entries.get(&tail) else { return 0 };
        if e.children > 0 {
            return 0; // 非叶不可逐
        }
        let mut span = 1usize;
        let mut cur = tail;
        while let Some(e) = self.entries.get(&cur) {
            let Some(parent) = e.parent else { break };
            // 父只能有这一个孩子 ⇒ 逐掉 cur 后父才成为叶
            match self.entries.get(&parent) {
                Some(pe) if pe.children == 1 => {
                    span += 1;
                    cur = parent;
                }
                _ => break,
            }
        }
        span
    }

    /// 单条移除(host 条目被逐时用;返回其绑定槽,`None` = 本来就在 host)。
    pub(crate) fn remove(&mut self, chain: ChainHash) -> Option<Option<u32>> {
        let e = self.entries.remove(&chain)?;
        self.leaf_set.remove(&chain);
        if let Some(p) = e.parent {
            if let Some(pe) = self.entries.get_mut(&p) {
                pe.children = pe.children.saturating_sub(1);
                if pe.children == 0 {
                    self.leaf_set.insert(p);
                    self.leaf_lru.push_back((p, pe.access));
                }
            }
        }
        Some(e.slot)
    }

    fn touch_leaf(&mut self, hash: ChainHash) {
        if self.leaf_set.contains(&hash) {
            if let Some(e) = self.entries.get(&hash) {
                self.leaf_lru.push_back((hash, e.access));
            }
        }
        self.compact_lru_if_needed();
    }

    fn compact_lru_if_needed(&mut self) {
        let threshold = self.entries.len().max(64) * 4;
        if self.leaf_lru.len() <= threshold {
            return;
        }
        self.leaf_lru.retain(|(hash, access)| {
            if !self.leaf_set.contains(hash) {
                return false;
            }
            match self.entries.get(hash) {
                Some(e) => e.access == *access,
                None => false,
            }
        });
    }

    fn next_access(&mut self) -> u64 {
        self.access = self.access.wrapping_add(1);
        self.access
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn match_walks_parent_chain_and_stops_at_miss() {
        let mut t = PrefixTree::new(4, 16);
        let tokens = toks(8);
        assert_eq!(t.matched_blocks(&tokens).matched_blocks, 0, "空树必 miss");

        let up = t.insert(&tokens, &[10, 11]);
        assert_eq!(up.inserted.len(), 2);
        let m = t.matched_blocks(&toks(12));
        assert_eq!(m.matched_blocks, 2, "只匹配已登记的整块");
        assert_eq!(m.blocks.iter().map(|b| b.slot).collect::<Vec<_>>(), vec![Some(10), Some(11)]);
        assert_eq!(m.tail, Some(super::super::chain::tail_hash(&tokens, 4, 2).expect("尾哈希")));

        // 字面相同但历史不同 ⇒ 不同身份(父链在身份里)
        let mut other = PrefixTree::new(4, 16);
        let shifted: Vec<u32> = (1..13).collect();
        other.insert(&shifted, &[20, 21, 22]);
        let mut mixed = vec![99u32];
        mixed.extend(toks(12));
        assert_eq!(other.matched_blocks(&mixed).matched_blocks, 0, "父链不同 ⇒ 不命中");
    }

    #[test]
    fn insert_reports_excess_eviction_and_leaf_lru_order() {
        let mut t = PrefixTree::new(4, 1);
        let up = t.insert(&toks(8), &[21, 22]);
        assert_eq!(up.evicted.len(), 1, "容量 1 ⇒ 逐一");
        assert_eq!(up.evicted[0].1, Some(22), "逐最老叶(块 22 是后插入的叶)");
        assert_eq!(t.matched_blocks(&toks(8)).matched_blocks, 1);
    }

    #[test]
    fn host_binding_keeps_identity_without_slot() {
        let mut t = PrefixTree::new(4, 8);
        t.insert(&toks(8), &[7, 8]);
        let tail = super::super::chain::tail_hash(&toks(8), 4, 2).expect("尾哈希");
        assert_eq!(t.set_slot(tail, None), Some(Some(8)), "摘绑定返回旧槽");
        let m = t.matched_blocks(&toks(8));
        assert_eq!(m.matched_blocks, 2, "host 条目仍按内容命中");
        assert_eq!(m.blocks.last().and_then(|b| b.slot), None, "设备上没有副本");
    }

    /// **树形分叉**（前缀缓存是树）：两条链共享前缀、末尾分叉。
    ///
    /// 判据：
    /// 1. 共享块是**同一内容身份**（不是复制），两条链各自都能整链命中；
    /// 2. 只有叶可逐 ⇒ **最冷的叶先走**（不按深度）；
    /// 3. 某分支的叶逐光后，它让出的共享父块**只有在另一枝也走了之后**才变成叶
    ///    ⇒ 共享前缀在任一分支活着时**不会被逐**。
    #[test]
    fn branches_share_prefix_and_evict_leaf_first() {
        let bs = 4;
        let mut t = PrefixTree::new(bs, 16);
        let base = toks(8); // 前两块
        let mut a = base.clone();
        a.extend([100u32, 101, 102, 103]); // A 的第三块
        let mut b = base.clone();
        b.extend([200u32, 201, 202, 203]); // B 的第三块
        t.insert(&a, &[10, 11, 12]);
        t.insert(&b, &[20, 21, 22]);
        let ha = t.matched_blocks(&a);
        let hb = t.matched_blocks(&b);
        assert_eq!((ha.matched_blocks, hb.matched_blocks), (3, 3), "两枝各自整链命中");
        assert_eq!(ha.blocks[0].chain, hb.blocks[0].chain, "共享块 = 同一身份");
        assert_eq!(ha.blocks[1].chain, hb.blocks[1].chain);
        assert_ne!(ha.blocks[2].chain, hb.blocks[2].chain, "分叉点身份不同");
        // 刚用过 A ⇒ B 的尾叶最冷 ⇒ 先逐它
        let _ = t.matched_blocks(&a);
        let ev = t.evict_device(1);
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].1, Some(22), "先逐最冷的叶(B 的尾巴)");
        assert_eq!(t.matched_blocks(&a).matched_blocks, 3, "A 不受影响");
        assert_eq!(t.matched_blocks(&b).matched_blocks, 2, "B 只剩共享前缀");
        let shared_tail = hb.blocks[1].chain;
        assert_eq!(
            t.get(shared_tail).expect("共享块在").children,
            1,
            "A 还挂着 ⇒ 共享块仍非叶(不可逐)"
        );
        // 再逐 A 的尾叶 ⇒ 两枝都没了 ⇒ 共享前缀才变叶
        let ev = t.evict_device(1);
        assert_eq!(ev[0].1, Some(12), "接着逐 A 的尾叶");
        assert!(
            t.leaves().iter().any(|(h, _, _)| *h == shared_tail),
            "两枝尽 ⇒ 共享前缀成为可逐的叶"
        );
    }

    #[test]
    fn only_leaves_are_evictable() {
        let mut t = PrefixTree::new(4, 8);
        t.insert(&toks(8), &[3, 4]);
        let parent = super::super::chain::tail_hash(&toks(4), 4, 1).expect("首块哈希");
        let e = t.get(parent).expect("条目在");
        assert_eq!(e.children, 1, "有子链 ⇒ 非叶");
        assert!(t.leaves().iter().all(|(h, _, _)| *h != parent), "非叶不入候选");
    }
}
