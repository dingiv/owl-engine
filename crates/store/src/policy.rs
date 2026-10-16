//! 策略对象:驱逐序 / 准入姿态(设计卷 §3.2)。
//!
//! **判据全在策略对象里,不散在调用点** —— 这是"线是动的"(§2.2)能落地的前提:
//! 同一个引擎,换个策略对象就换一套取舍(生产档 / 诊断档 / 保真档)。

use crate::id::{ChainHash, SnapKey};
use crate::tier::Tier;

/// 准入满员时的姿态。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueuePolicy {
    /// **响亮拒**(不给近似、不给排队;客户端可自行降级或改地板)。
    Reject,
    /// 排队等待释放(上限 = `max_wait_ms`,超时按拒处理)。
    Queue {
        /// 最长等待(毫秒)。
        max_wait_ms: u64,
    },
}

/// 驱逐候选(两腿共用;`protect` 之外才可能出现)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvictCandidate {
    /// 腿:attention 块链 或 GDN 快照。
    pub leg: Leg,
    /// 所在层(驱逐是"下沉"或"释放";GPU → host → 盘)。
    pub tier: Tier,
    /// 字节数。
    pub bytes: usize,
    /// 最近使用序号(策略输入;越小越老)。
    pub last_use: u64,
    /// 块链身份(KV 腿;**内容身份,不是物理槽号** —— 见 `id.rs` 身份律)。
    pub chain: Option<ChainHash>,
    /// 快照键(状态腿)。
    pub snap: Option<SnapKey>,
}

/// 两条腿(独立管理 —— 裁决②;候选合并由关系层做)。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Leg {
    /// attention 块链。
    Kv,
    /// GDN 状态快照。
    State,
}

/// 驱逐策略(只**排序**,不搬 —— 搬运是 engine 在段边界做的事)。
pub trait EvictPolicy: Send + Sync {
    /// 给候选排序:返回的序 = 驱逐先后(前面的先动)。
    fn order(&self, candidates: &mut [EvictCandidate]);

    /// 策略名(报告/mertics 用)。
    fn name(&self) -> &'static str;
}

/// 纯 LRU(最老先逐)。
///
/// 现役基线;`ARC`/`chain-LRU` 属 A 轴第三刀(`S3`),届时并列实现而非改这个
/// (设计要求:策略是对象,不是分支)。
#[derive(Debug, Default)]
pub struct LruPolicy;

impl EvictPolicy for LruPolicy {
    fn order(&self, candidates: &mut [EvictCandidate]) {
        candidates.sort_by_key(|c| (c.last_use, c.leg, c.tier));
    }

    fn name(&self) -> &'static str {
        "lru"
    }
}

/// 准入策略(boot 装配一次;`max_active_turns` 是硬天花板)。
#[derive(Debug)]
pub struct AdmissionPolicy {
    /// 最多几个 turn 同时在飞(boot 旋钮;正名自 `gdn_slots`)。
    pub max_active_turns: usize,
    /// 满员姿态。
    pub on_full: QueuePolicy,
}

impl AdmissionPolicy {
    /// 构造(上限为 0 = 装配错,响亮拒 —— 由 `VramBudget::decompose` 兜底)。
    pub fn new(max_active_turns: usize, on_full: QueuePolicy) -> Self {
        Self { max_active_turns, on_full }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(last_use: u64, bytes: usize, leg: Leg) -> EvictCandidate {
        EvictCandidate { leg, tier: Tier::Gpu, bytes, last_use, chain: None, snap: None }
    }

    #[test]
    fn lru_orders_oldest_first() {
        let p = LruPolicy;
        let mut v = vec![c(30, 1, Leg::Kv), c(10, 1, Leg::State), c(20, 1, Leg::Kv)];
        p.order(&mut v);
        assert_eq!(v.iter().map(|x| x.last_use).collect::<Vec<_>>(), vec![10, 20, 30]);
        assert_eq!(p.name(), "lru");
    }
}
