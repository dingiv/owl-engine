//! 链身份原语:**逐块父链哈希**与**链哈希游标**(设计卷 §9.1 身份律的唯一产地)。
//!
//! ```text
//! hash_block(parent, tokens)   →  ChainHash        单块身份(含父链)
//! content_fingerprint(tokens)  →  u64             仅防哈希碰撞(与父链无关)
//! tail_hash(tokens, bs, n)     →  Option<ChainHash> 从零算(一次性,别放热路径)
//! ChainCursor                  →  懒惰推进(只喂新增整块 ⇒ 全 turn O(n))
//! ```
//!
//! # 为什么游标必须存在(不是过早优化)
//!
//! 快照/复用的身份是**边界块的内容哈希**。而"某边界块的链尾哈希"按定义是从
//! token 0 起的逐块父链哈希 —— 每个块边界都从头算 = 长 prefill 的 O(n²)
//! (128k 提示词 × 每块一算 ≈ 5×10⁸ 次哈希)。游标把这件事摊平:只在边界处
//! 把**新增的整块**喂进来,全 turn O(n);前缀命中时由调度层 `seed()` 接种,
//! 于是身份链跨会话接得上。
//!
//! # 唯一产地律
//!
//! 建树(`tree.rs`)、快照键(engine 侧 `capture_gdn_snap`)、跨会话匹配全走
//! **这一份**实现 —— 两处实现 = 两套身份,迟早对不上(HARD 门:
//! 游标 ≡ 缓存哈希链,见 `tree.rs` 测试与 engine 侧回归)。

use crate::id::{ChainHash, TOKENS_PER_BLOCK};

/// 逐块父链哈希(`parent → block tokens`)。
///
/// ⚠️ 换实现 = 换身份(同进程内缓存/快照互相认不出)。跨进程持久化(S6 磁盘层)
/// 时必须换成稳定哈希 —— 那是另一个面,别悄悄改这里。
pub fn hash_block(parent: ChainHash, tokens: &[u32]) -> ChainHash {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    parent.raw().hash(&mut h);
    tokens.hash(&mut h);
    ChainHash::new(h.finish())
}

/// 内容指纹(仅防哈希碰撞:同父链同位、字面不同的两段 token)。
pub fn content_fingerprint(tokens: &[u32]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    0xDEAD_BEEFu64.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

/// 从零算一条 token 链的链尾身份(`full_blocks` 个整块)。
///
/// **一次性用途**(拍单张快照/对拍);热路径一律用 [`ChainCursor`]。
pub fn tail_hash(tokens: &[u32], block_tokens: usize, full_blocks: usize) -> Option<ChainHash> {
    if full_blocks == 0 || block_tokens == 0 {
        return None;
    }
    let mut parent = ChainHash::new(0);
    for block in tokens.chunks(block_tokens).take(full_blocks) {
        parent = hash_block(parent, block);
    }
    Some(parent)
}

/// 链哈希游标(**懒惰推进**;会话级状态,生命周期跟随链)。
///
/// 不变量:**喂进来的序列必须是从 0 开始的完整 token 链**,且边界单调不减
/// (`advance` 对回退/段长不足**响亮拒**,不静默记错)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainCursor {
    blocks: usize,
    tail: ChainHash,
}

impl Default for ChainCursor {
    fn default() -> Self {
        Self::new()
    }
}

impl ChainCursor {
    /// 零起点(新链)。
    pub fn new() -> Self {
        Self { blocks: 0, tail: ChainHash::new(0) }
    }

    /// 已哈希的整块数。
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// 链尾身份(已哈希部分的最后一块)。
    pub fn tail(&self) -> ChainHash {
        self.tail
    }

    /// 重置(前缀失配/临时会话焚毁:新链从零起)。
    pub fn reset(&mut self) {
        self.blocks = 0;
        self.tail = ChainHash::new(0);
    }

    /// 前缀命中后播种(`blocks` 个块 + 该块的内容身份,来自缓存哈希链)。
    pub fn seed(&mut self, blocks: usize, tail: ChainHash) {
        self.blocks = blocks;
        self.tail = tail;
    }

    /// 推进到 `full_blocks` 个整块。`seg` = **从 `self.blocks × block_size`
    /// 开始的 token 段**(长度须 ≥ 新增块数 × block_size)。
    ///
    /// 返回推进后的链尾身份;`None` = 回退/段长不足(调用方响亮拒)。
    pub fn advance(
        &mut self,
        block_size: usize,
        seg: &[u32],
        full_blocks: usize,
    ) -> Option<ChainHash> {
        if block_size == 0 || full_blocks < self.blocks {
            return None;
        }
        let n = full_blocks - self.blocks;
        if seg.len() < n * block_size {
            return None;
        }
        for k in 0..n {
            self.tail = hash_block(self.tail, &seg[k * block_size..(k + 1) * block_size]);
        }
        self.blocks = full_blocks;
        Some(self.tail)
    }

    /// 按 token 数推进(整块口径;不足一块 = 不动)。
    pub fn advance_tokens(&mut self, tokens: &[u32]) -> Option<ChainHash> {
        let full = tokens.len() / TOKENS_PER_BLOCK;
        self.advance(TOKENS_PER_BLOCK, tokens, full)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| (i * 7919) % 100_003).collect()
    }

    #[test]
    fn cursor_advances_incrementally_and_matches_from_scratch() {
        let bs = TOKENS_PER_BLOCK;
        let t = toks(bs * 8);
        let mut cur = ChainCursor::new();
        cur.advance(bs, &t[..3 * bs], 3).expect("3 块");
        cur.advance(bs, &t[3 * bs..5 * bs], 5).expect("5 块");
        cur.advance(bs, &t[5 * bs..8 * bs], 8).expect("8 块");
        assert_eq!(cur.blocks(), 8);

        for blocks in [1usize, 3, 5, 8] {
            let expect = tail_hash(&t, bs, blocks).expect("从零算");
            let mut c = ChainCursor::new();
            let got = c.advance(bs, &t[..blocks * bs], blocks).expect("推进");
            assert_eq!(got, expect, "{blocks} 块边界:游标 ≠ 从零算(身份链必须唯一)");
        }
    }

    #[test]
    fn cursor_rejects_rollback_and_short_segment() {
        let bs = TOKENS_PER_BLOCK;
        let t = toks(bs * 8);
        let mut cur = ChainCursor::new();
        cur.advance(bs, &t, 8).expect("8 块");
        assert!(cur.advance(bs, &t[..bs], 2).is_none(), "回退必拒");
        assert!(cur.advance(bs, &t[..bs], 10).is_none(), "要 2 块只给 1 块必拒");
        assert_eq!(cur.blocks(), 8, "被拒后游标不动");
    }

    #[test]
    fn identity_chain_carries_parents() {
        let bs = TOKENS_PER_BLOCK;
        let a = toks(bs * 2);
        let b: Vec<u32> = (7u32..7 + bs as u32).collect(); // 字面相同段落在不同位置
        let mut tail_a = ChainHash::new(0);
        let mut tail_b = ChainHash::new(0);
        for blk in a.chunks(bs) {
            tail_a = hash_block(tail_a, blk);
        }
        for blk in b.chunks(bs) {
            tail_b = hash_block(tail_b, blk);
        }
        assert_ne!(tail_a, tail_b);
    }
}
