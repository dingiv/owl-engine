//! **链索引**(S6.1):盘上"这条链对应的 token 序列"—— 跨重启重建树的唯一依据。
//!
//! # 为什么需要它
//!
//! 块身份 = `hash_block(父身份, 本块 tokens)`(父链哈希)。盘上的内容文件名只有
//! **块身份**,而树的重建需要**父链 + 逐块 tokens**(才能重算身份、重建父指针、
//! 恢复命中面的逐块语义)。⇒ 重启时缺的从来不是内容,而是"**这些内容对应哪段 token**"。
//!
//! # 形态(自包含、廉价)
//!
//! ```text
//! magic  "OWLIDX01"        8 B
//! block_tokens  u32 LE     4 B   (本引擎恒 32)
//! n_tokens      u32 LE     4 B
//! tokens        n × u32 LE      (4 B/token)
//! ```
//!
//! 8k token 的链 ≈ **32 KB**(相对内容 2 MiB/块的千分之几)。**只增不删**:
//! 链内容被驱逐/丢弃后索引留着无害(boot 以"内容文件在场"为准截断),
//! 省掉"驱逐时同步删索引"的一条跨层一致性问题(卷内 §31.7)。
//!
//! # 谁写、谁读
//!
//! - **写**:engine 在 turn 收口(`register_chain` 处)按整链写一份(`<tail>.idx`);
//! - **读**:boot 扫目录 → [`ChainIndex::decode`] → 逐块校验"内容在盘上" →
//!   截断到**连续前缀** → `Rel::adopt_disk_chains` 重建树(`slot = None`)。
//!
//! 编码/解码是纯数据变换(store);文件读写归 engine(§21)。

use crate::error::{StoreError, StoreResult};

/// 索引文件魔数(带版本;不认 = 响亮拒,不猜)。
pub const INDEX_MAGIC: [u8; 8] = *b"OWLIDX01";

/// 一条链的索引(重建树所需的最小信息)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainIndex {
    /// 每块 token 数(页宽;与腿几何一致才可用)。
    pub block_tokens: usize,
    /// 整链 token 序列(从根开始;尾部不满一块的部分**不重建**)。
    pub tokens: Vec<u32>,
}

impl ChainIndex {
    /// 构造(校验几何:`block_tokens > 0` 且 `tokens` 至少一块)。
    pub fn new(block_tokens: usize, tokens: Vec<u32>) -> StoreResult<Self> {
        if block_tokens == 0 {
            return Err(StoreError::Invariant("索引 block_tokens = 0".into()));
        }
        if tokens.len() < block_tokens {
            return Err(StoreError::Invariant(format!(
                "索引 token 数 {} < 一块 {}(空链无意义)",
                tokens.len(),
                block_tokens
            )));
        }
        Ok(Self { block_tokens, tokens })
    }

    /// 整块数(向下取整;尾部残块不参与身份)。
    pub fn blocks(&self) -> usize {
        self.tokens.len() / self.block_tokens
    }

    /// 编码后的字节数。
    pub fn encoded_len(&self) -> usize {
        16 + self.tokens.len() * 4
    }

    /// 编码(固定小端;见模块头注的布局)。
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.encoded_len());
        out.extend_from_slice(&INDEX_MAGIC);
        out.extend_from_slice(&(self.block_tokens as u32).to_le_bytes());
        out.extend_from_slice(&(self.tokens.len() as u32).to_le_bytes());
        for t in &self.tokens {
            out.extend_from_slice(&t.to_le_bytes());
        }
        out
    }

    /// 解码(魔数/长度/几何任一不符 = **响亮拒**,不静默截断)。
    pub fn decode(bytes: &[u8]) -> StoreResult<Self> {
        if bytes.len() < 16 {
            return Err(StoreError::Invariant(format!(
                "索引太短:{} < 16(文件损坏?)",
                bytes.len()
            )));
        }
        if bytes[..8] != INDEX_MAGIC {
            return Err(StoreError::Invariant(
                "索引魔数不符(不是本版索引文件;空的/别的格式一律拒)".into(),
            ));
        }
        let block_tokens = u32::from_le_bytes(bytes[8..12].try_into().expect("切片长度由上面保证")) as usize;
        let n = u32::from_le_bytes(bytes[12..16].try_into().expect("同上")) as usize;
        let want = 16 + n * 4;
        if bytes.len() != want {
            return Err(StoreError::Invariant(format!(
                "索引长度不符:{} ≠ 声明 {}(截断/多余字节)",
                bytes.len(),
                want
            )));
        }
        let mut tokens = Vec::with_capacity(n);
        for i in 0..n {
            let at = 16 + i * 4;
            tokens.push(u32::from_le_bytes(bytes[at..at + 4].try_into().expect("上面校验过")));
        }
        ChainIndex::new(block_tokens, tokens)
    }

    /// 截断到 `n` 个整块(按"内容在场"的连续前缀重建用)。
    pub fn truncated(&self, n: usize) -> ChainIndex {
        let n = n.min(self.blocks());
        ChainIndex {
            block_tokens: self.block_tokens,
            tokens: self.tokens[..n * self.block_tokens].to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_roundtrip() {
        let idx = ChainIndex::new(32, (0..96u32).collect()).expect("构造");
        assert_eq!(idx.blocks(), 3);
        let bytes = idx.encode();
        assert_eq!(bytes.len(), idx.encoded_len());
        assert_eq!(ChainIndex::decode(&bytes).expect("解码"), idx);
    }

    #[test]
    fn decode_rejects_bad_magic_and_length() {
        let idx = ChainIndex::new(32, (0..64u32).collect()).expect("构造");
        let mut bytes = idx.encode();
        bytes[0] = b'X';
        assert!(ChainIndex::decode(&bytes).is_err(), "魔数不符必须拒");
        let mut bytes = idx.encode();
        bytes.push(0);
        assert!(ChainIndex::decode(&bytes).is_err(), "多余字节必须拒");
        let short = &idx.encode()[..20];
        assert!(ChainIndex::decode(short).is_err(), "截断必须拒");
    }

    #[test]
    fn truncation_keeps_whole_blocks_only() {
        let idx = ChainIndex::new(32, (0..100u32).collect()).expect("构造");
        assert_eq!(idx.blocks(), 3, "100/32 = 3 整块");
        let t = idx.truncated(2);
        assert_eq!(t.tokens.len(), 64);
        assert_eq!(t.blocks(), 2);
        assert_eq!(idx.truncated(99).tokens.len(), 96, "截断不超过整块数");
    }
}
