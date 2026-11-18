//! `Rel` 的选择层索引账（Mean-K index 的几何与增减）。
//!
//! 纯 host 账；索引内容由 engine 计算后登记。

use super::*;

impl Rel {
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
        let ix = self.index.as_mut().ok_or_else(|| {
            StoreError::Invariant("选择层索引未注册(engine 应在 boot 调 attach_index)".into())
        })?;
        let bytes = ix.block_bytes();
        ix.register(block, entries)?;
        // 第 0 刀:索引面 = **缓存归属**(无单会话归属;跨会话共享)
        crate::tier::book_lock(&self.ledger).tiers_mut().add_owned(
            crate::tier::Owner::Cache,
            Tier::Host,
            AssetKind::Index,
            bytes,
        )
    }

    /// **登记一块的扁平索引段**(GPU Mean-K 出口):与 [`Self::index_register`]
    /// 同账同语义，只是段已是 `[layer][kv_head][dim]` 规范序。
    pub fn index_register_flat(&mut self, block: ChainHash, flat: Vec<u16>) -> StoreResult<()> {
        let ix = self.index.as_mut().ok_or_else(|| {
            StoreError::Invariant("选择层索引未注册(engine 应在 boot 调 attach_index)".into())
        })?;
        let bytes = ix.block_bytes();
        ix.register_flat(block, flat)?;
        crate::tier::book_lock(&self.ledger).tiers_mut().add_owned(
            crate::tier::Owner::Cache,
            Tier::Host,
            AssetKind::Index,
            bytes,
        )
    }

    /// 丢一块的索引(出账)。
    pub fn index_drop(&mut self, block: ChainHash) -> StoreResult<()> {
        let ix = self
            .index
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("选择层索引未注册".into()))?;
        let bytes = ix.block_bytes();
        ix.drop_block(block)?;
        crate::tier::book_lock(&self.ledger)
            .tiers_mut()
            .remove_owned(
                crate::tier::Owner::Cache,
                Tier::Host,
                AssetKind::Index,
                bytes,
            )
    }

    /// 索引总字节(0 = 未注册)。
    pub fn index_bytes(&self) -> usize {
        self.index.as_ref().map(|i| i.bytes()).unwrap_or(0)
    }
}
