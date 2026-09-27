//! KV 物理块账房(E2b,2026-09-27)。
//!
//! 母本 = engine-bak / xinfer `block_manager.rs`(字节级同源,见
//! roadmap 代差分析 §四),按 owl 单序列引擎形态剥离裁剪:
//! - 剥:Sequence/Runner 耦合、CPU swap(owl 无抢占,≤8 会话 M0.5
//!   调度)、Mamba 快照机(E2c 按 turn 边界快照另立)、多模态种子;
//! - 留:free 队列 + 引用计数 + 块表增长契约(语义与 vLLM v1
//!   BlockPool 同族:分配从队首 = 未来驱逐序;ref 1 = 会话独占,
//!   >1 = 前缀共享,E2c 接 PrefixCache 时直接继承)。
//!
//! 契约(与 kernels 层对齐):物理槽 = block_table[逻辑块]·block_size
//! + 块内偏移;K0 写核吃 slot_mapping(物理槽),v1/prefill 吃
//! block_tables(块表)—— 两表均由引擎按本账房状态装配(f32 过线,
//! 契约 5,物理槽 < 2²⁴ 恒成立)。

use crate::Result;
use owl_iface::contract::ModelError;
use std::collections::VecDeque;

/// KV 物理块账房(单实例;RunningEngine 持有)
pub struct BlockManager {
    block_size: usize,
    /// ref_count[i] = 物理块 i 的引用数;0 = 空闲
    ref_counts: Vec<u32>,
    /// 空闲块队列(分配 = pop_front;归还 = push_back —— FIFO 复用,
    /// E2c 前缀缓存接入后换 LRU 驱逐序)
    free_block_ids: VecDeque<u32>,
}

impl BlockManager {
    /// 全池空闲起步(num_blocks = 池块容量;block_size = 页大小)
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        Self {
            block_size,
            ref_counts: vec![0; num_blocks],
            free_block_ids: (0..num_blocks as u32).collect(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.ref_counts.len()
    }

    pub fn free_blocks(&self) -> usize {
        self.free_block_ids.len()
    }

    /// 覆盖 target_len tokens 所需块表长度
    fn blocks_for(&self, target_len: usize) -> usize {
        target_len.div_ceil(self.block_size)
    }

    /// 增长块表至覆盖 target_len tokens(只增不减;幂等;事务性 ——
    /// 池不足时整单拒绝,不留半截块表,调用方可安全重试/失败)。
    pub fn ensure_for_len(&mut self, table: &mut Vec<u32>, target_len: usize) -> Result<()> {
        let need = self.blocks_for(target_len);
        let have = table.len();
        if need <= have {
            return Ok(());
        }
        let delta = need - have;
        if self.free_block_ids.len() < delta {
            return Err(ModelError::Msg(format!(
                "KV 块池耗尽:需 {need} 块,空闲 {} —— 降低并发/上下文(E2c 驱逐另接)",
                self.free_block_ids.len()
            )));
        }
        for _ in 0..delta {
            let id = self.free_block_ids.pop_front().expect("已查空闲");
            self.ref_counts[id as usize] = 1;
            table.push(id);
        }
        Ok(())
    }

    /// 引用减一;归零归还空闲队列(块表尾序 = 驱逐候选序)
    pub fn decref(&mut self, id: u32) {
        let rc = &mut self.ref_counts[id as usize];
        *rc = rc.saturating_sub(1);
        if *rc == 0 {
            self.free_block_ids.push_back(id);
        }
    }

    /// 归还会话整条块表(逆序 decref;E2c 前缀共享块自然只剩减引用)
    pub fn release_table(&mut self, table: &mut Vec<u32>) {
        for &id in table.iter().rev() {
            self.decref(id);
        }
        table.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_grow_release_roundtrip() {
        let mut m = BlockManager::new(4, 32);
        assert_eq!(m.free_blocks(), 4);

        let mut table = Vec::new();
        m.ensure_for_len(&mut table, 1).expect("1 tok");
        assert_eq!(table.len(), 1, "1 token 也要 1 块");
        assert_eq!(m.free_blocks(), 3);

        m.ensure_for_len(&mut table, 64).expect("64 tok");
        assert_eq!(table.len(), 2, "64 tok @ page32 = 2 块");
        assert_eq!(m.free_blocks(), 2);

        // 幂等:够长不再分配
        m.ensure_for_len(&mut table, 64).expect("幂等");
        assert_eq!(m.free_blocks(), 2);

        // 归零归还,队列 FIFO
        m.release_table(&mut table);
        assert!(table.is_empty());
        assert_eq!(m.free_blocks(), 4);
    }

    #[test]
    fn exhaustion_is_structured_error() {
        let mut m = BlockManager::new(2, 32);
        let mut table = Vec::new();
        let err = m.ensure_for_len(&mut table, 100).unwrap_err();
        assert!(format!("{err}").contains("块池耗尽"));
        // 事务性:失败不留半截块表
        assert!(table.is_empty());
        assert_eq!(m.free_blocks(), 2);
    }

    #[test]
    fn refcount_shared_blocks_free_on_last() {
        let mut m = BlockManager::new(4, 32);
        let mut a = Vec::new();
        m.ensure_for_len(&mut a, 32).expect("A");
        let shared = a[0];
        // 前缀共享(E2c 形态):B 复用 A 的块 0,ref = 2
        m.ref_counts[shared as usize] = 2;
        let b = vec![shared];
        a.clear();
        m.decref(shared); // A 释放 → ref 1,块不回池
        assert_eq!(m.free_blocks(), 3);
        m.decref(shared); // B 释放 → ref 0,回池
        assert_eq!(m.free_blocks(), 4);
    }
}
