//! **池容量治理的纯算式**(唯一产地)。
//!
//! 口径(2026-10-22 起):
//!
//! ```text
//! base   = min(free, free − total×(1−target))       // 驱动 free;target 只作上限
//! budget = base − reserve − fixed_after
//! cap    = max(0, budget) / per_token_bytes
//! ```
//!
//! `free = 0` 时回退旧口径:`total×target − live`(测试/旧路没有驱动读数)。
//! `vram_budget == 0` 或 `total == 0` = 治理禁用,返回 `None`。
//!
//! 本模块**不读设备、不读配置**:engine 把 boot 快照与几何事实填进来,
//! 只拿回可打印、可断言的容量结果。日志仍由 engine 输出(它掌握上下文标签)。

use crate::error::{StoreError, StoreResult};

/// 主池 / 草稿池 / MTP 链的**逐 token 线性项**。
///
/// engine 只负责把模型几何映射成这三个字节数;求和与解释归本模块,
/// 避免 `pool_cap`、`[ctx-max]` 与测试各算一遍。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolTokenTerms {
    /// 主池 KV:全注意力层 × (K+V) × hkv × hd × elem。
    pub main_kv: usize,
    /// 草稿池按 token 线性的部分(DFlash2 ring 档为 0)。
    pub draft_pool: usize,
    /// MTP 草稿链按 token 线性的部分(ring 档为 0)。
    pub mtp_chain: usize,
}

impl PoolTokenTerms {
    /// 显式构造(三个字节数都由调用方给出)。
    pub fn new(main_kv: usize, draft_pool: usize, mtp_chain: usize) -> Self {
        Self {
            main_kv,
            draft_pool,
            mtp_chain,
        }
    }

    /// 每 token 合计字节。
    pub fn total(self) -> usize {
        self.main_kv + self.draft_pool + self.mtp_chain
    }

    /// 一行可读分解(给 `[ctx-max]` / `[vram-gov]` 用)。
    pub fn explain(self) -> String {
        format!(
            "主池 {}B + 草稿池 {}B + MTP 链 {}B",
            self.main_kv, self.draft_pool, self.mtp_chain
        )
    }
}

/// 治理器之后的固定分配项输入。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolFixedInputs {
    /// GDN 层数。
    pub gdn_layers: usize,
    /// 活跃 turn 格数。
    pub gdn_slots: usize,
    /// 单层单格元素数(四件套之和)。
    pub gdn_slot_elems: usize,
    /// GDN 状态元素字节(f32 = 4)。
    pub gdn_elem_bytes: usize,
    /// 前缀快照槽数。
    pub snap_device_slots: usize,
    /// 草稿 ring 字节(DFlash2 / MTP ring)。
    pub draft_ring_bytes: usize,
    /// 是否分配 spec 回滚快照。
    pub spec_snapshot: bool,
    /// 是否分配 query replay 回滚快照。
    pub replay_snapshot: bool,
}

/// 治理器之后的固定分配项(全部是真实分配,不含图 slab / warmup)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolFixedTerms {
    /// GDN 状态格(`gdn_slots × 每行字节`)。
    pub gdn_grid: usize,
    /// GDN 前缀快照池。
    pub gdn_snap: usize,
    /// spec 回滚专用快照。
    pub spec_snap: usize,
    /// query replay 回滚缓冲。
    pub replay_snap: usize,
    /// 草稿池 ring(固定项)。
    pub draft_ring: usize,
}

impl PoolFixedTerms {
    /// 从几何事实编译固定项(唯一公式)。
    pub fn from_inputs(i: PoolFixedInputs) -> Self {
        let grid = i
            .gdn_layers
            .saturating_mul(i.gdn_slots)
            .saturating_mul(i.gdn_slot_elems)
            .saturating_mul(i.gdn_elem_bytes);
        let row = i
            .gdn_layers
            .saturating_mul(i.gdn_slot_elems)
            .saturating_mul(i.gdn_elem_bytes);
        Self {
            gdn_grid: grid,
            gdn_snap: i.snap_device_slots.saturating_mul(row),
            spec_snap: if i.spec_snapshot { row } else { 0 },
            replay_snap: if i.replay_snapshot { row } else { 0 },
            draft_ring: i.draft_ring_bytes,
        }
    }

    /// 固定项合计。
    pub fn total(self) -> usize {
        self.gdn_grid + self.gdn_snap + self.spec_snap + self.replay_snap + self.draft_ring
    }

    /// 一行可读分解(给 `[vram-gov]` 用)。
    pub fn explain(self) -> String {
        let mib = |v: usize| v as f64 / 1048576.0;
        format!(
            "gdn格 {:.0}MiB + gdn快照 {:.0}MiB + spec快照 {:.0}MiB + replay快照 {:.0}MiB + 草稿ring {:.0}MiB",
            mib(self.gdn_grid),
            mib(self.gdn_snap),
            mib(self.spec_snap),
            mib(self.replay_snap),
            mib(self.draft_ring)
        )
    }
}

/// 池容量算式的输入(全部为显式事实)。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoolCapInputs {
    /// 卡标称总字节(驱动 `total`)。
    pub card_total_bytes: u64,
    /// 池前驱动 `free`(字节;0 = 无设备读数,走旧口径)。
    pub card_free_bytes: u64,
    /// 池前引擎 live 基线(仅在 `card_free_bytes == 0` 的旧口径里使用)。
    pub live_baseline_bytes: u64,
    /// 显存治理占比(`0` = 禁用;`1.0` 合法 = 吃满 total,reserve 仍生效)。
    pub vram_budget: f64,
    /// 显式预留(驱动/allocator/瞬态安全垫)。
    pub reserve_bytes: u64,
    /// 治理器之后的固定分配(GDN 格/快照/草稿 ring 等;不含主 KV 池)。
    pub fixed_after_bytes: usize,
    /// 主池每 token 字节(必须 > 0)。
    pub per_token_bytes: usize,
}

/// 池容量结果(保留中间量供 `[vram-gov]` 取证)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolCap {
    /// 预算允许的池 token 上限。
    pub cap_tokens: usize,
    /// `base − reserve − fixed_after`(可为负;负值按 0 计容量)。
    pub budget_bytes: i64,
    /// `free` 与 target 上限取小后的预算基。
    pub base_bytes: i64,
    /// `free − total×(1−target)`(target 上限,可为负)。
    pub target_cap_bytes: i64,
    /// 实际使用的 live 基线(仅旧口径参考)。
    pub live_baseline_bytes: u64,
}

/// 计算池 token 上限;返回 `None` = 治理禁用。
///
/// 这是 `StatePool::alloc` 与报告面共用的**唯一**容量公式。调用方只负责
/// 收集 `PoolCapInputs` 并打印结果,不再在 engine 里复制算式。
pub fn pool_cap_tokens(inputs: PoolCapInputs) -> StoreResult<Option<PoolCap>> {
    if inputs.per_token_bytes == 0 {
        return Err(StoreError::Budget(
            "pool_cap: per_token_bytes = 0(池容量算式无分母)".into(),
        ));
    }
    // `vram_budget = 0` = 治理禁用;`1.0` 是合法档(吃满总显存,reserve 仍是
    // 唯一缓冲)。旧条件写 `target < 1.0` 会把 1.0 当禁用 ⇒ 硬预算静默失效。
    if !(inputs.vram_budget > 0.0) || inputs.card_total_bytes == 0 {
        return Ok(None);
    }

    let free = inputs.card_free_bytes;
    let total = inputs.card_total_bytes;
    let target_cap = free as i64 - ((total as f64 * (1.0 - inputs.vram_budget)) as i64);
    let live = inputs.live_baseline_bytes;
    // 无 free 落账(测试/旧路)⇒ 回退旧口径。
    let base = if free > 0 {
        (free as i64).min(target_cap.max(0))
    } else {
        (total as f64 * inputs.vram_budget) as i64 - live as i64
    };
    let budget = base - inputs.reserve_bytes as i64 - inputs.fixed_after_bytes as i64;
    let cap_tokens = (budget.max(0) as u64 / inputs.per_token_bytes as u64) as usize;
    Ok(Some(PoolCap {
        cap_tokens,
        budget_bytes: budget,
        base_bytes: base,
        target_cap_bytes: target_cap,
        live_baseline_bytes: live,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const MIB: u64 = 1 << 20;

    fn inputs() -> PoolCapInputs {
        PoolCapInputs {
            card_total_bytes: 24 * GIB,
            card_free_bytes: 20 * GIB,
            live_baseline_bytes: 0,
            vram_budget: 0.5,
            reserve_bytes: GIB,
            fixed_after_bytes: 1 * MIB as usize,
            per_token_bytes: 1 * MIB as usize,
        }
    }

    #[test]
    fn target_cap_clamps_free() {
        // target = 0.5 ⇒ target_cap = 20G − 24G×0.5 = 8G;
        // budget = 8G − 1G − 1M ⇒ cap = 7167 tok。
        let cap = pool_cap_tokens(inputs())
            .expect("formula")
            .expect("enabled");
        assert_eq!(cap.target_cap_bytes, (8 * GIB) as i64);
        assert_eq!(cap.base_bytes, (8 * GIB) as i64);
        assert_eq!(cap.budget_bytes, (7 * GIB - MIB) as i64);
        assert_eq!(cap.cap_tokens, 7167);
    }

    #[test]
    fn free_zero_falls_back_to_live_baseline() {
        let mut i = inputs();
        i.card_free_bytes = 0;
        i.live_baseline_bytes = GIB;
        // 旧口径:24G×0.5 − 1G = 11G;再减 reserve/fixed ⇒ 10G−1M。
        let cap = pool_cap_tokens(i).expect("formula").expect("enabled");
        assert_eq!(cap.base_bytes, (11 * GIB) as i64);
        assert_eq!(cap.budget_bytes, (10 * GIB - MIB) as i64);
        assert_eq!(cap.cap_tokens, 10239);
    }

    #[test]
    fn negative_budget_has_zero_capacity() {
        let mut i = inputs();
        i.card_free_bytes = GIB;
        i.reserve_bytes = 4 * GIB;
        let cap = pool_cap_tokens(i).expect("formula").expect("enabled");
        assert!(cap.budget_bytes < 0);
        assert_eq!(cap.cap_tokens, 0);
    }

    #[test]
    fn disabled_when_budget_or_total_is_zero() {
        let mut i = inputs();
        i.vram_budget = 0.0;
        assert!(pool_cap_tokens(i).expect("formula").is_none());

        let mut i = inputs();
        i.card_total_bytes = 0;
        assert!(pool_cap_tokens(i).expect("formula").is_none());
    }

    #[test]
    fn full_budget_is_legal_and_uses_free() {
        let mut i = inputs();
        i.vram_budget = 1.0;
        let cap = pool_cap_tokens(i).expect("formula").expect("enabled");
        assert_eq!(cap.target_cap_bytes, (20 * GIB) as i64);
        assert_eq!(cap.base_bytes, (20 * GIB) as i64);
        assert_eq!(cap.cap_tokens, 19455);
    }

    #[test]
    fn token_terms_sum_and_explain() {
        let t = PoolTokenTerms::new(10, 20, 30);
        assert_eq!(t.total(), 60);
        assert_eq!(t.explain(), "主池 10B + 草稿池 20B + MTP 链 30B");
    }

    #[test]
    fn fixed_terms_sum_only_real_allocations() {
        let t = PoolFixedTerms::from_inputs(PoolFixedInputs {
            gdn_layers: 2,
            gdn_slots: 3,
            gdn_slot_elems: 5,
            gdn_elem_bytes: 4,
            snap_device_slots: 2,
            draft_ring_bytes: 100,
            spec_snapshot: true,
            replay_snapshot: true,
        });
        assert_eq!(t.gdn_grid, 120);
        assert_eq!(t.gdn_snap, 80);
        assert_eq!(t.spec_snap, 40);
        assert_eq!(t.replay_snap, 40);
        assert_eq!(t.draft_ring, 100);
        assert_eq!(t.total(), 380);
        assert!(t.explain().contains("草稿ring"));
    }

    /// 固定项从 `PoolFixedTerms` 投影到 `MemoryPlan/BudgetInputs` 时不得重算/漏算。
    #[test]
    fn fixed_terms_match_memory_plan_projection() {
        use crate::plan::{
            CardPlan, ComponentId, MemComponent, MemoryPlan, PlanPolicy, PlanSource, PoolPlan,
            TierPlan,
        };
        use crate::tier::{Owner, Tier};

        let terms = PoolFixedTerms::from_inputs(PoolFixedInputs {
            gdn_layers: 2,
            gdn_slots: 3,
            gdn_slot_elems: 5,
            gdn_elem_bytes: 4,
            snap_device_slots: 2,
            draft_ring_bytes: 100,
            spec_snapshot: true,
            replay_snapshot: true,
        });
        let graph = 2 << 20;
        let fixed = vec![
            MemComponent::new(
                ComponentId::Weights,
                1 << 30,
                PlanSource::Measured,
                Tier::Gpu,
                Owner::System,
            ),
            MemComponent::new(
                ComponentId::GdnGrid,
                terms.gdn_grid,
                PlanSource::Measured,
                Tier::Gpu,
                Owner::System,
            ),
            MemComponent::new(
                ComponentId::GdnSnapshots,
                terms.gdn_snap,
                PlanSource::Measured,
                Tier::Gpu,
                Owner::Cache,
            ),
            MemComponent::new(
                ComponentId::SpecSnapshots,
                terms.spec_snap,
                PlanSource::Measured,
                Tier::Gpu,
                Owner::Cache,
            ),
            MemComponent::new(
                ComponentId::ReplaySnapshots,
                terms.replay_snap,
                PlanSource::Measured,
                Tier::Gpu,
                Owner::Cache,
            ),
            MemComponent::new(
                ComponentId::GraphSlab,
                graph,
                PlanSource::Conservative,
                Tier::Gpu,
                Owner::System,
            ),
            MemComponent::new(
                ComponentId::DraftPool,
                terms.draft_ring,
                PlanSource::Measured,
                Tier::Gpu,
                Owner::System,
            ),
        ];
        let plan = MemoryPlan::new(
            0,
            CardPlan {
                total_bytes: 24 << 30,
                free_at_boot_bytes: 16 << 30,
                reserve_bytes: 1 << 30,
            },
            fixed,
            PoolPlan {
                pages: 4,
                page_tokens: 32,
                tokens: 128,
                per_token_bytes: 32 << 10,
                block_bytes: 1 << 20,
                bytes: 4 << 20,
                gdn_slots: 3,
                gdn_row_bytes: terms.gdn_grid / 3,
            },
            TierPlan::default(),
            PlanPolicy { fixed_pool: true },
        )
        .expect("plan");
        let inp = plan.to_budget_inputs().expect("projection");
        assert_eq!(inp.weights_bytes, 1 << 30);
        assert_eq!(inp.gdn_row_bytes * 3, terms.gdn_grid);
        assert_eq!(inp.draft_pool_bytes, terms.draft_ring);
        assert_eq!(
            inp.runtime_floor_bytes,
            terms.gdn_snap + terms.spec_snap + terms.replay_snap + graph
        );
        assert_eq!(
            inp.weights_bytes
                + inp.runtime_floor_bytes
                + inp.draft_pool_bytes
                + inp.gdn_row_bytes * 3,
            plan.fixed_bytes()
        );
    }

    #[test]
    fn zero_per_token_is_rejected() {
        let mut i = inputs();
        i.per_token_bytes = 0;
        let err = pool_cap_tokens(i).expect_err("must reject");
        assert!(matches!(err, StoreError::Budget(_)));
    }
}
