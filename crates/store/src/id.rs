//! 身份:链内容哈希 / 前缀身份 / 快照键(设计卷 §3.2;**S3 身份迁移版**)。
//!
//! # 身份律(S3 定案,2026-10-16)
//!
//! store 里一切"**是谁**"都用**内容身份**([`ChainHash`])表达;**物理槽号不入身份**。
//!
//! 为什么:物理槽是**可回收资源** —— 驱逐/释放后回自由队列,下一次分配即被复用
//! (`blocks.rs::decref` → `push_back` → `pop_front`)。拿槽号当键 ⇒ 槽被复用时
//! 旧键指向新内容 = **假命中**。
//! 现存的落地隐患:engine 的 GDN 快照键 = `block_table[边界]`(物理块号,
//! `exec/spec.rs::capture_gdn_snap`),而快照池**不随前缀缓存驱逐失效** ⇒
//! 槽复用后 `has_snap(旧槽号)` 会命中**别人的状态**(静默错值)。修法 = 本文件的
//! 内容身份:`SnapKey` 改由**边界块的内容哈希**导出(engine 侧切键见卷内挂账)。
//!
//! code-master:类型即约束 —— 身份用 newtype 表达,禁止裸 `u32`/`u64` 在接口间流窜。

/// 逻辑块覆盖的 token 数。
///
/// 设计卷 §3.6.2 定案:**32 token**(KVMem 缺省 32-token 块;且天然等于本引擎
/// XQA 的 `tokensPerPage = 32` ⇒ 逻辑块与物理页一一对齐,**不改任何 attention 契约**)。
pub const TOKENS_PER_BLOCK: usize = 32;

/// 链内容哈希(**块的身份**) —— 逐块父链式哈希(parent → block tokens),
/// 语义与引擎 E2c 前缀缓存同源:相同 = 同一段 token 且共享同一历史路径。
///
/// 注意:**块的内容身份包含它的父链** ⇒ 两段"字面相同但历史不同"的 token
/// 不会被当成同一块(前缀共享的正确语义)。
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ChainHash(u64);

impl ChainHash {
    /// 构造。
    pub fn new(v: u64) -> Self {
        Self(v)
    }
    /// 裸值。
    pub fn raw(self) -> u64 {
        self.0
    }
}

/// 快照键 = **块边界的内容身份**(边界块 = 覆盖该边界 token 的最后一块)。
///
/// 与 attention 块链**同源但不共享**:快照内容(递推状态)依赖 hidden 与历史路径
/// 而不可内容寻址 —— 这里用的是"它对应哪一段内容"的身份,不是快照本身的内容哈希
/// (`arch.md` 律②的结构性差异:KV 可内容寻址,状态只能按边界关联)。
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SnapKey(ChainHash);

impl SnapKey {
    /// 由边界块的内容身份构造。
    pub fn at_boundary(block: ChainHash) -> Self {
        Self(block)
    }
    /// 边界块的内容身份。
    pub fn boundary(self) -> ChainHash {
        self.0
    }
}

/// 前缀身份(关系层的键)= 前缀链尾块的内容身份。
///
/// 两条腿各认它的一个投影:`KvStore` 认 [`Self::chain`](块链能不能命中);
/// `StateStore` 认 [`Self::snap_key`](这块边界上有没有状态快照)。
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct PrefixId {
    /// 链尾块的内容身份(= 整条前缀的身份)。
    pub chain: ChainHash,
}

impl PrefixId {
    /// 构造(参数 = 链尾块的内容身份)。
    pub fn new(chain: ChainHash) -> Self {
        Self { chain }
    }

    /// 快照腿的键(同一身份的两个投影 —— 关系层的核心映射)。
    pub fn snap_key(&self) -> SnapKey {
        SnapKey::at_boundary(self.chain)
    }

    /// 覆盖 `tokens` 个 token 所需的逻辑块数(向上取整)。
    pub fn blocks_for(tokens: usize) -> usize {
        tokens.div_ceil(TOKENS_PER_BLOCK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_cover_tokens_round_up() {
        assert_eq!(PrefixId::blocks_for(0), 0);
        assert_eq!(PrefixId::blocks_for(1), 1);
        assert_eq!(PrefixId::blocks_for(32), 1);
        assert_eq!(PrefixId::blocks_for(33), 2);
        assert_eq!(PrefixId::blocks_for(262_144), 8192);
    }

    #[test]
    fn snap_key_is_boundary_identity() {
        let id = PrefixId::new(ChainHash::new(7));
        assert_eq!(id.snap_key().boundary(), ChainHash::new(7));
    }
}
