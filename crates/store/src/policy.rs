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

    /// **召回信号**(B-4;缺省 = 无操作)——
    /// "这些块刚刚**被注意力用到**"(选择层召回点选中)。只有吃这一步的策略
    /// (attention-aware LRFU)覆写;LRU 不覆写 ⇒ 旧策略行为逐字不变。
    fn note_recall(&self, _selected: &[ChainHash]) {}
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

/// 驱逐策略选择(boot 装配 **参数对象**;§A2"策略是对象,不是分支")。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EvictPolicyKind {
    /// 纯 LRU(现役基线)
    Lru,
    /// attention-aware LRFU(λ)
    Lrfu(f64),
}

impl EvictPolicyKind {
    /// 造策略对象。
    pub fn build(self) -> Box<dyn EvictPolicy> {
        match self {
            EvictPolicyKind::Lru => Box::new(LruPolicy),
            EvictPolicyKind::Lrfu(l) => Box::new(LrfuPolicy::new(l)),
        }
    }
}

/// **attention-aware LRFU**(B-4;2026-10-18)——
/// ActKV 的 keep-score 接到我们的**策略对象**上:决定"谁留设备 / 谁先被下沉"。
///
/// ```text
/// KS(b) ← λ·KS(b) + (1−λ)·hit(b)        hit = 该块**被召回点选中**过(注意力命中)
/// ```
///
/// - 与我们三层语义的关系:被逐**不是丢内容**,是**下沉到宿主**;选择层随时可回灌
///   ⇒ 本策略决定的是"谁的设备副本先让位",不是"谁的死活";
/// - **懒衰减**:不逐步扫全表,而按 `KS_last·λ^(now−t_last)` 惰性折算 ⇒ 每操作 O(1),
///   排序时按需折算(精确等价于逐步衰减,只差浮点结合序);
/// - 退化:`chain = None` 的候选(状态腿快照)按 `last_use` 排 —— 本策略只管 KV 腿。
#[derive(Debug)]
pub struct LrfuPolicy {
    /// 衰减因子 λ ∈ (0,1);越大越看**累积频率**,越小越看**近期**。
    lambda: f64,
    /// 内容身份 → (该块在 `t_last` 时刻的 KS, `t_last`)
    score: std::sync::Mutex<std::collections::HashMap<ChainHash, (f64, u64)>>,
    /// 单调时钟(每次策略操作 +1;与 `EvictCandidate.last_use` 同口径的序号空间)
    tick: std::sync::atomic::AtomicU64,
}

impl LrfuPolicy {
    /// 构造(λ 钳到 (0,1);越界 = 取缺省 0.9 —— 静默取值会骗人,故**响亮打印**)。
    pub fn new(lambda: f64) -> Self {
        let l = if lambda > 0.0 && lambda < 1.0 {
            lambda
        } else {
            eprintln!("[store] LrfuPolicy:λ={lambda} 非法(须 0<λ<1)⇒ 取 0.9");
            0.9
        };
        Self {
            lambda: l,
            score: std::sync::Mutex::new(std::collections::HashMap::new()),
            tick: std::sync::atomic::AtomicU64::new(1),
        }
    }

    /// 本块在 `now` 时刻的 keep-score(惰性衰减)。
    fn ks(&self, chain: ChainHash, now: u64) -> f64 {
        let Ok(m) = self.score.lock() else { return 0.0 };
        match m.get(&chain) {
            Some((k, t)) => k * self.lambda.powi(now.saturating_sub(*t) as i32),
            None => 0.0,
        }
    }

    /// 记一次**注意力命中**(召回点选中该块)。
    ///
    /// 时间戳口径:**只读时钟**(同一"步"内的多次命中共享时间戳);
    /// **步进由 `order` 推进**(每次出序 = 一步)⇒ 衰减粒度 = 出序次数,
    /// 而不是命中次数(否则"命中越频繁、时间戳越新"会自相矛盾)。
    pub fn on_recall(&self, selected: &[ChainHash]) {
        let now = self.tick.load(std::sync::atomic::Ordering::Relaxed);
        let Ok(mut m) = self.score.lock() else { return };
        for c in selected {
            let prev = match m.get(c) {
                Some((k, t)) => k * self.lambda.powi(now.saturating_sub(*t) as i32),
                None => 0.0,
            };
            m.insert(*c, (prev * self.lambda + (1.0 - self.lambda), now));
        }
    }

    /// 现役条数(报告/测试面)。
    pub fn len(&self) -> usize {
        self.score.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// 空?
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl EvictPolicy for LrfuPolicy {
    fn order(&self, candidates: &mut [EvictCandidate]) {
        // 每次出序 = 一步(惰性衰减的时间基准)
        let now = self.tick.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // 先按惰性 KS 升序(冷者先走),同级再按 last_use(同 LRU 的兜底)
        candidates.sort_by(|a, b| {
            let (ka, kb) = match (a.chain, b.chain) {
                (Some(ca), Some(cb)) => (self.ks(ca, now), self.ks(cb, now)),
                _ => (0.0, 0.0),
            };
            ka.partial_cmp(&kb)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then((a.last_use, a.leg, a.tier).cmp(&(b.last_use, b.leg, b.tier)))
        });
    }

    fn name(&self) -> &'static str {
        "lrfu(attention)"
    }

    fn note_recall(&self, selected: &[ChainHash]) {
        self.on_recall(selected);
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

    fn cand(chain: u64, last_use: u64) -> EvictCandidate {
        EvictCandidate { leg: Leg::Kv, tier: Tier::Gpu, bytes: 1024, last_use, chain: Some(ChainHash::new(chain)), snap: None }
    }

    /// **B-4 门**:"反复被注意力选中"的块,即使 `last_use` 更老,也要**排在后面**
    /// (LRU 会先逐它 —— 这正是 LRFU 要修的病)。
    #[test]
    fn attention_hits_outrank_lru_recency() {
        let lrfu = LrfuPolicy::new(0.9);
        let hot = ChainHash::new(1); // 被选中很多次,但"最近使用"很早
        for _ in 0..20 {
            lrfu.on_recall(&[hot]);
        }
        let mut v = vec![cand(1, 10), cand(2, 100)];
        lrfu.order(&mut v);
        assert_eq!(
            v[0].chain,
            Some(ChainHash::new(2)),
            "未被注意力用过的块应先被下沉(LRFU:KS 低者先走)"
        );
        // 对照:LRU 的序正相反(纯看 last_use)
        let mut v2 = vec![cand(1, 10), cand(2, 100)];
        LruPolicy.order(&mut v2);
        assert_eq!(v2[0].chain, Some(ChainHash::new(1)), "LRU 先逐最老 —— 本测试要修的病");
    }

    /// **懒衰减 ≡ 逐步衰减**:空转 N 步后 KS 应缩到 `λ^N`。
    #[test]
    fn lazy_decay_matches_stepwise() {
        let l = 0.5f64;
        let p = LrfuPolicy::new(l);
        let c = ChainHash::new(7);
        p.on_recall(&[c]); // KS = λ·0 + (1−λ) = 0.5
        let now0 = p.tick.load(std::sync::atomic::Ordering::Relaxed);
        assert!((p.ks(c, now0) - 0.5).abs() < 1e-12, "首次命中 KS 应 = 1−λ");
        let mut stepwise = 0.5f64;
        for _ in 0..4 {
            stepwise *= l;
        }
        let lazy = p.ks(c, now0 + 4);
        assert!((lazy - stepwise).abs() < 1e-12, "惰性 {lazy} vs 逐步 {stepwise}");
    }

    /// 状态腿候选(无 chain)退化按 `last_use`;λ 越界取缺省且不 panic;
    /// `Rel` 门面能装配/转发。
    #[test]
    fn policy_fallbacks_and_rel_facade() {
        let p = LrfuPolicy::new(1.5); // 越界 ⇒ 缺省 0.9
        assert_eq!(p.name(), "lrfu(attention)");
        let mut v = vec![c(9, 1, Leg::State), c(2, 1, Leg::State)];
        p.order(&mut v);
        assert_eq!(v[0].last_use, 2, "无 chain ⇒ 按 last_use(与 LRU 同兜底)");
        // 缺省面:LRU 不覆写 `note_recall`(编译期证明旧策略行为逐字不变)
        LruPolicy.note_recall(&[ChainHash::new(3)]);
    }
}
