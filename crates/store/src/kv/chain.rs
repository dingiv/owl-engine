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

/// **块的多模态内容指纹**(该块覆盖的图像内容的 64 位摘要;0 = 无图像)。
///
/// 为什么必须有:token 流里图像只是 `grid` 个**字面相同**的占位符 ⇒ 换一张图
/// 不换一个 token。只看 token 的身份会把两张不同的截图判成同一条链 ⇒ 直接
/// 复用上一条请求的 KV,模型对着旧截图答(2026-11-03 定谳:同文本 + 换图 ⇒
/// `cached_tokens` 1088/1100 且答案逐点等于上一张图的方块位置)。
/// vLLM 的 `BlockHash` 里放 `mm_hashes` 就是这件事。
///
/// 语义:同一块位置上的**全部多模态内容**折成一个数(通常一张图落在若干块里,
/// 每块用同一枚图指纹;纯文本块 = 0)。
pub type BlockTag = u64;

/// 逐块父链哈希(`parent → block tokens`,含块的多模态指纹)。
///
/// `tag = 0` = 该块不含多模态内容 ⇒ 与旧实现逐位相同(纯文本缓存面零变化)。
///
/// ⚠️ 换实现 = 换身份(同进程内缓存/快照互相认不出)。跨进程持久化(S6 磁盘层)
/// 时必须换成稳定哈希 —— 那是另一个面,别悄悄改这里。
pub fn hash_block(parent: ChainHash, tokens: &[u32], tag: BlockTag) -> ChainHash {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    parent.raw().hash(&mut h);
    tokens.hash(&mut h);
    // 域分隔再混入指纹:tag=0 时**不改动**旧身份(纯文本链身份逐字节不变)。
    if tag != 0 {
        IMAGE_TAG_SALT.hash(&mut h);
        tag.hash(&mut h);
    }
    ChainHash::new(h.finish())
}

/// 图像指纹的域分隔盐(与 token 流混入同一哈希流时防止结构化碰撞)。
pub(crate) const IMAGE_TAG_SALT: u64 = 0x006D_6D2D_6861_7368;

/// 内容指纹(仅防哈希碰撞:同父链同位、字面不同的两段 token)。
pub fn content_fingerprint(tokens: &[u32], tag: BlockTag) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    0xDEAD_BEEFu64.hash(&mut h);
    tokens.hash(&mut h);
    if tag != 0 {
        IMAGE_TAG_SALT.hash(&mut h);
        tag.hash(&mut h);
    }
    h.finish()
}

/// 从零算一条 token 链的链尾身份(`full_blocks` 个整块)。
///
/// `tags` = **逐块**多模态指纹(长度须 ≥ `full_blocks`;**空 = 纯文本**)。
///
/// **一次性用途**(拍单张快照/对拍);热路径一律用 [`ChainCursor`]。
pub fn tail_hash(
    tokens: &[u32],
    block_tokens: usize,
    full_blocks: usize,
    tags: &[BlockTag],
) -> Option<ChainHash> {
    if full_blocks == 0 || block_tokens == 0 {
        return None;
    }
    if !tags.is_empty() && tags.len() < full_blocks {
        return None;
    }
    let mut parent = ChainHash::new(0);
    for (i, block) in tokens.chunks(block_tokens).take(full_blocks).enumerate() {
        parent = hash_block(parent, block, tags.get(i).copied().unwrap_or(0));
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
    /// `seg_tags` = 与 `seg` **按块对齐**的多模态指纹(第 i 项 = `seg` 第 i 块;
    /// 空 = 该段纯文本)。指到一半 ⇒ `None`(账不平,不静默按 0 处理)。
    ///
    /// 返回推进后的链尾身份;`None` = 回退/段长不足/指纹不足(调用方响亮拒)。
    pub fn advance(
        &mut self,
        block_size: usize,
        seg: &[u32],
        full_blocks: usize,
        seg_tags: &[BlockTag],
    ) -> Option<ChainHash> {
        if block_size == 0 || full_blocks < self.blocks {
            return None;
        }
        let n = full_blocks - self.blocks;
        if seg.len() < n * block_size {
            return None;
        }
        if !seg_tags.is_empty() && seg_tags.len() < n {
            return None;
        }
        for k in 0..n {
            self.tail = hash_block(
                self.tail,
                &seg[k * block_size..(k + 1) * block_size],
                seg_tags.get(k).copied().unwrap_or(0),
            );
        }
        self.blocks = full_blocks;
        Some(self.tail)
    }

    /// 按 token 数推进(整块口径;不足一块 = 不动;`tags` 逐块,空 = 纯文本)。
    pub fn advance_tokens(&mut self, tokens: &[u32], tags: &[BlockTag]) -> Option<ChainHash> {
        let full = tokens.len() / TOKENS_PER_BLOCK;
        self.advance(TOKENS_PER_BLOCK, tokens, full, tags)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).map(|i| (i * 7919) % 100_003).collect()
    }

    /// 🔴 **红门(2026-11-03)**:同 token、不同多模态指纹 ⇒ **必须**不同身份。
    ///
    /// 现状(修前):图像在 token 流里只是字面相同的占位符 ⇒ 两张不同截图共享
    /// 一条链 ⇒ `cached_tokens` 满命中 + 答案对着旧图。真机复现见
    /// `docs/devlog/serving/multimodal-block-identity-立案.md`。
    #[test]
    fn identical_tokens_different_media_tag_do_not_alias() {
        let bs = TOKENS_PER_BLOCK;
        let t = toks(bs * 4);
        let tags_a = [0u64, 0xAAAA_BBBB_CCCC_DDDD, 0, 0];
        let tags_b = [0u64, 0x1111_2222_3333_4444, 0, 0];
        let a = tail_hash(&t, bs, 4, &tags_a).expect("tags_a");
        let b = tail_hash(&t, bs, 4, &tags_b).expect("tags_b");
        assert_ne!(
            a, b,
            "同 token 流 + 不同图像指纹被判成同一条链 ⇒ 会复用陈旧图像 KV"
        );
        // 同一张图(同指纹)必须仍然同身份 —— 否则跨会话共享图像链的收益归零。
        assert_eq!(a, tail_hash(&t, bs, 4, &tags_a).expect("tags_a 再来"));
        // 纯文本(tag 全 0)= 旧身份,一个字节都不许变。
        assert_eq!(
            tail_hash(&t, bs, 4, &[0u64; 4]).expect("zero tags"),
            tail_hash(&t, bs, 4, &[]).expect("no tags"),
            "纯文本链身份必须与旧实现逐位相同(缓存面零回归)"
        );
        // 指纹不足 = 响亮拒(不许静默按 0 继续)。
        assert!(
            tail_hash(&t, bs, 4, &tags_a[..2]).is_none(),
            "逐块指纹不足必须拒,不许把图像块当纯文本"
        );
    }

    #[test]
    fn cursor_advances_incrementally_and_matches_from_scratch() {
        let bs = TOKENS_PER_BLOCK;
        let t = toks(bs * 8);
        let mut cur = ChainCursor::new();
        cur.advance(bs, &t[..3 * bs], 3, &[]).expect("3 块");
        cur.advance(bs, &t[3 * bs..5 * bs], 5, &[]).expect("5 块");
        cur.advance(bs, &t[5 * bs..8 * bs], 8, &[]).expect("8 块");
        assert_eq!(cur.blocks(), 8);

        for blocks in [1usize, 3, 5, 8] {
            let expect = tail_hash(&t, bs, blocks, &[]).expect("从零算");
            let mut c = ChainCursor::new();
            let got = c.advance(bs, &t[..blocks * bs], blocks, &[]).expect("推进");
            assert_eq!(got, expect, "{blocks} 块边界:游标 ≠ 从零算(身份链必须唯一)");
        }

        // 带指纹的推进 ≡ 带指纹的从零算(唯一产地律:两条路必须同身份)
        let tags = [0u64, 7, 0, 0, 9, 0, 0, 0];
        let expect = tail_hash(&t, bs, 8, &tags).expect("从零算(带指纹)");
        let mut c = ChainCursor::new();
        let got = c.advance(bs, &t[..8 * bs], 8, &tags).expect("推进(带指纹)");
        assert_eq!(got, expect, "带指纹时:游标 ≠ 从零算");
    }

    #[test]
    fn cursor_rejects_rollback_and_short_segment() {
        let bs = TOKENS_PER_BLOCK;
        let t = toks(bs * 8);
        let mut cur = ChainCursor::new();
        cur.advance(bs, &t, 8, &[]).expect("8 块");
        assert!(cur.advance(bs, &t[..bs], 2, &[]).is_none(), "回退必拒");
        assert!(cur.advance(bs, &t[..bs], 10, &[]).is_none(), "要 2 块只给 1 块必拒");
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
            tail_a = hash_block(tail_a, blk, 0);
        }
        for blk in b.chunks(bs) {
            tail_b = hash_block(tail_b, blk, 0);
        }
        assert_ne!(tail_a, tail_b);
    }
}
