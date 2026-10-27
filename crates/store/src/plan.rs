//! **显存计划(MemoryPlan)**:boot 期唯一可打印、可对账的卡级预算事实源。
//!
//! 本模块只做**结构与投影**,不做分配、不做准入判决:
//!
//! ```text
//! CardPlan    = total / free_at_boot / reserve
//! Fixed[]     = 权重 / GDN 格 / 快照 / 图 slab / warmup / 草稿池 ...
//! PoolPlan    = 主 KV 池的页/token/字节几何
//! TierPlan    = host / disk 分片
//! MemoryPlan  = 上述四件的组合 + 卡面闭合校验
//! ```
//!
//! 为什么单独一层:旧路径里 `StatePool::pool_cap`、`assemble_budget` 与
//! `BudgetInputs` 各算一遍;`runtime_floor` 甚至由两次相邻 `live_bytes()` 的差
//! 反推(生产恒为 0)。现在固定项、池面、预留额先编译成一份 plan,预算入参
//! 只是 plan 的**投影** —— 数字漂了在 boot 就能看见。
//!
//! 纪律:
//! - 本模块**不读设备、不读配置**;调用方把实测/配置事实填进来;
//! - 每项必须带来源([`PlanSource`]),禁止把常数伪装成实测;
//! - `Σfixed + pool + reserve` 超出卡面 ⇒ [`MemoryPlan::check_card`] 响亮报出
//!   (是否拒启由 boot 策略决定,本模块不静默吞掉)。

use crate::error::{StoreError, StoreResult};
use crate::tier::{Owner, Tier};

/// 固定项身份(新增项必须显式命名;禁止 `Other("...")` 兜底)。
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComponentId {
    /// 模型权重与装载产物(常驻)。
    Weights,
    /// GDN 状态格(`gdn_slots × gdn_row_bytes`)。
    GdnGrid,
    /// GDN 前缀快照池。
    GdnSnapshots,
    /// 投机回滚专用快照缓冲。
    SpecSnapshots,
    /// query replay 回滚缓冲。
    ReplaySnapshots,
    /// 图 slab 预算(捕获期产物)。
    GraphSlab,
    /// warmup 瞬态与治理器之后的杂项。
    WarmupMisc,
    /// 草稿池(MTP 链 / DFlash2 ring)。
    DraftPool,
}

impl ComponentId {
    /// 日志/报告标签(稳定字符串,不是 Debug 格式)。
    pub fn label(self) -> &'static str {
        match self {
            Self::Weights => "weights",
            Self::GdnGrid => "gdn_grid",
            Self::GdnSnapshots => "gdn_snap",
            Self::SpecSnapshots => "spec_snap",
            Self::ReplaySnapshots => "replay_snap",
            Self::GraphSlab => "graph_slab",
            Self::WarmupMisc => "warmup_misc",
            Self::DraftPool => "draft_pool",
        }
    }
}

/// 数字来源(报告面必须能区分"实测"与"保守上界")。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanSource {
    /// 分配后实测(或由实测账直接读出)。
    Measured,
    /// 由已实测的几何/配置推导。
    Derived,
    /// 显式配置值。
    Configured,
    /// 尚未逐项实测的保守上界(方向安全,但报告必须标出来)。
    Conservative,
}

impl PlanSource {
    /// 日志/报告标签。
    pub fn label(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Derived => "derived",
            Self::Configured => "configured",
            Self::Conservative => "conservative",
        }
    }
}

/// 一个固定显存项。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemComponent {
    /// 身份。
    pub id: ComponentId,
    /// 字节数。
    pub bytes: usize,
    /// 数字来源。
    pub source: PlanSource,
    /// 所在层(当前固定项全在设备层;留字段给未来的权重分层)。
    pub tier: Tier,
    /// 归属主体(报告/对账口径)。
    pub owner: Owner,
}

impl MemComponent {
    /// 构造一项(字段全显式;没有默认来源)。
    pub fn new(
        id: ComponentId,
        bytes: usize,
        source: PlanSource,
        tier: Tier,
        owner: Owner,
    ) -> Self {
        Self {
            id,
            bytes,
            source,
            tier,
            owner,
        }
    }
}

/// 卡面事实(实测;CPU mock 的 free/total 可为 0)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CardPlan {
    /// 卡标称 total(字节)。
    pub total_bytes: usize,
    /// boot 时驱动 free(字节;CUDA ctx 已扣)。
    pub free_at_boot_bytes: usize,
    /// 显式预留(驱动/allocator/瞬态的安全垫)。
    pub reserve_bytes: usize,
}

/// 主 KV 池几何。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolPlan {
    /// 管理面页数。
    pub pages: usize,
    /// 页宽(token/页)。
    pub page_tokens: usize,
    /// 池 token 容量(页对齐后的实际值)。
    pub tokens: usize,
    /// 每 token KV 字节。
    pub per_token_bytes: usize,
    /// 单块(页)字节。
    pub block_bytes: usize,
    /// 主池总字节。
    pub bytes: usize,
    /// 活跃 turn 硬天花板。
    pub gdn_slots: usize,
    /// 每活跃 turn 的 GDN 行字节。
    pub gdn_row_bytes: usize,
}

impl PoolPlan {
    /// 池是否已预分配(固定池档)。
    pub fn fixed(&self) -> bool {
        self.bytes > 0
    }
}

/// 中层分片预算(与 [`crate::tier::tier_budget_of`] 同源填入)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TierPlan {
    /// host 层总预算。
    pub host_bytes: usize,
    /// 磁盘层总预算。
    pub disk_bytes: usize,
}

/// 计划策略(只放影响解释/投影的显式选择,不放运行期状态)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanPolicy {
    /// 固定池档:准入面 = 预分配池;动态池档 = 卡级余量。
    pub fixed_pool: bool,
}

/// 卡面闭合检查(不返回 Result:超了要能打印/上报,不是把 plan 构造炸掉)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanCheck {
    /// 固定项合计。
    pub fixed_bytes: usize,
    /// 主 KV 池字节。
    pub pool_bytes: usize,
    /// 预留字节。
    pub reserve_bytes: usize,
    /// fixed + pool + reserve。
    pub total_bytes: usize,
    /// boot free。
    pub card_free_bytes: usize,
    /// 卡 total。
    pub card_total_bytes: usize,
    /// 超出 boot free 的字节(0 = 闭合)。
    pub over_free_bytes: usize,
    /// 超出卡 total 的字节(0 = 闭合)。
    pub over_total_bytes: usize,
}

impl PlanCheck {
    /// 是否卡面闭合(free/total 未知时不声称失败)。
    pub fn fits(&self) -> bool {
        self.over_free_bytes == 0 && self.over_total_bytes == 0
    }
}

/// boot 唯一预算事实源。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryPlan {
    /// 计划代次(boot 序号;运行期重编译时递增)。
    pub epoch: u64,
    /// 卡面事实。
    pub card: CardPlan,
    /// 固定项(权重/GDN/快照/图/warmup/草稿池)。
    pub fixed: Vec<MemComponent>,
    /// 主 KV 池。
    pub pool: PoolPlan,
    /// 中层分片。
    pub tiers: TierPlan,
    /// 策略选择。
    pub policy: PlanPolicy,
}

impl MemoryPlan {
    /// 编译(唯一入口)。
    ///
    /// 只做结构校验:池几何不能为 0;固定项/池/预留的加法不能溢出。
    /// 卡面是否装得下由 [`Self::check_card`] 单独回答(便于 boot 先打印再决策)。
    pub fn new(
        epoch: u64,
        card: CardPlan,
        fixed: Vec<MemComponent>,
        pool: PoolPlan,
        tiers: TierPlan,
        policy: PlanPolicy,
    ) -> StoreResult<Self> {
        if pool.page_tokens == 0 {
            return Err(StoreError::Budget("MemoryPlan: page_tokens = 0".into()));
        }
        if pool.per_token_bytes == 0 {
            return Err(StoreError::Budget("MemoryPlan: per_token_bytes = 0".into()));
        }
        if pool.bytes == 0 && policy.fixed_pool {
            return Err(StoreError::Budget(
                "MemoryPlan: fixed_pool = true 但池字节为 0".into(),
            ));
        }
        if pool.bytes != pool.pages.saturating_mul(pool.block_bytes) {
            return Err(StoreError::Budget(format!(
                "MemoryPlan: 池字节不平 pages={} × block={} ≠ {}",
                pool.pages, pool.block_bytes, pool.bytes
            )));
        }
        if pool.tokens != pool.pages.saturating_mul(pool.page_tokens) {
            return Err(StoreError::Budget(format!(
                "MemoryPlan: 池 token 不平 pages={} × page={} ≠ {}",
                pool.pages, pool.page_tokens, pool.tokens
            )));
        }
        if pool.per_token_bytes != pool.block_bytes / pool.page_tokens {
            return Err(StoreError::Budget(format!(
                "MemoryPlan: 每 token 字节不平 block={} / page={} ≠ {}",
                pool.block_bytes, pool.page_tokens, pool.per_token_bytes
            )));
        }
        let plan = Self {
            epoch,
            card,
            fixed,
            pool,
            tiers,
            policy,
        };
        // 加法溢出在构造时就响亮拒(不让报告面出现回绕数字)。
        plan.fixed_bytes_checked()?;
        plan.total_bytes_checked()?;
        Ok(plan)
    }

    fn fixed_bytes_checked(&self) -> StoreResult<usize> {
        self.fixed.iter().try_fold(0usize, |acc, c| {
            acc.checked_add(c.bytes)
                .ok_or_else(|| StoreError::Budget("MemoryPlan: 固定项合计溢出".into()))
        })
    }

    fn total_bytes_checked(&self) -> StoreResult<usize> {
        self.fixed_bytes_checked()?
            .checked_add(self.pool.bytes)
            .and_then(|v| v.checked_add(self.card.reserve_bytes))
            .ok_or_else(|| StoreError::Budget("MemoryPlan: fixed+pool+reserve 溢出".into()))
    }

    /// 固定项合计。
    pub fn fixed_bytes(&self) -> usize {
        self.fixed
            .iter()
            .fold(0usize, |acc, c| acc.saturating_add(c.bytes))
    }

    /// fixed + pool + reserve。
    pub fn total_bytes(&self) -> usize {
        self.fixed_bytes()
            .saturating_add(self.pool.bytes)
            .saturating_add(self.card.reserve_bytes)
    }

    /// 取一项固定项(重复项 = 调用方构造错;返回第一项,报告面另有
    /// [`Self::duplicate_components`] 可查)。
    pub fn component(&self, id: ComponentId) -> Option<MemComponent> {
        self.fixed.iter().copied().find(|c| c.id == id)
    }

    /// 重复登记的固定项(空 = 合法)。
    pub fn duplicate_components(&self) -> Vec<ComponentId> {
        let mut seen = std::collections::BTreeSet::new();
        let mut dup = std::collections::BTreeSet::new();
        for c in &self.fixed {
            if !seen.insert(c.id) {
                dup.insert(c.id);
            }
        }
        dup.into_iter().collect()
    }

    /// 卡面闭合检查(free/total 为 0 时视为未知,不报超)。
    pub fn check_card(&self) -> PlanCheck {
        let fixed_bytes = self.fixed_bytes();
        let pool_bytes = self.pool.bytes;
        let reserve_bytes = self.card.reserve_bytes;
        let total_bytes = fixed_bytes
            .saturating_add(pool_bytes)
            .saturating_add(reserve_bytes);
        let over_free_bytes = if self.card.free_at_boot_bytes == 0 {
            0
        } else {
            total_bytes.saturating_sub(self.card.free_at_boot_bytes)
        };
        let over_total_bytes = if self.card.total_bytes == 0 {
            0
        } else {
            total_bytes.saturating_sub(self.card.total_bytes)
        };
        PlanCheck {
            fixed_bytes,
            pool_bytes,
            reserve_bytes,
            total_bytes,
            card_free_bytes: self.card.free_at_boot_bytes,
            card_total_bytes: self.card.total_bytes,
            over_free_bytes,
            over_total_bytes,
        }
    }

    /// 转成准入预算入参(**唯一投影**;不再由调用方用残差反推 `runtime_floor`)。
    ///
    /// `runtime_floor` = 固定项中除权重、GDN 格、草稿池之外的部分
    /// (快照池 + 图 slab + warmup/杂项)。这样与 `VramBudget::v_avail`
    /// 的减法项一一对应,不会把同一批字节算两遍。
    pub fn to_budget_inputs(&self) -> StoreResult<crate::budget::BudgetInputs> {
        let weights = self.component(ComponentId::Weights).map_or(0, |c| c.bytes);
        let gdn_grid = self.component(ComponentId::GdnGrid).map_or(0, |c| c.bytes);
        let draft = self
            .component(ComponentId::DraftPool)
            .map_or(0, |c| c.bytes);
        let fixed = self.fixed_bytes();
        let runtime_floor = fixed
            .checked_sub(weights)
            .and_then(|v| v.checked_sub(gdn_grid))
            .and_then(|v| v.checked_sub(draft))
            .ok_or_else(|| {
                StoreError::Budget(format!(
                    "MemoryPlan: 固定项分类超过合计(fixed={fixed}, weights={weights}, \
                     gdn_grid={gdn_grid}, draft={draft})"
                ))
            })?;
        Ok(crate::budget::BudgetInputs {
            v_card_bytes: self.card.free_at_boot_bytes,
            weights_bytes: weights,
            runtime_floor_bytes: runtime_floor,
            draft_pool_bytes: draft,
            gdn_row_bytes: self.pool.gdn_row_bytes,
            kv_bytes_per_token: self.pool.per_token_bytes,
            page_tokens: self.pool.page_tokens,
            host_bytes: self.tiers.host_bytes,
            disk_bytes: self.tiers.disk_bytes,
            kv_face_bytes: if self.policy.fixed_pool {
                self.pool.bytes
            } else {
                0
            },
        })
    }

    /// 一行可读摘要(固定项按身份排序;给 boot 日志与 `/debug/vram` 用)。
    pub fn summary(&self) -> String {
        let mut items: Vec<String> = self
            .fixed
            .iter()
            .map(|c| {
                format!(
                    "{}={:.1}MiB[{}]",
                    c.id.label(),
                    c.bytes as f64 / 1048576.0,
                    c.source.label()
                )
            })
            .collect();
        items.sort();
        format!(
            "fixed({}) + pool={:.1}MiB({}页×{}tok×{}B) + reserve={:.1}MiB = {:.1}MiB / free {:.1}MiB",
            items.join(" + "),
            self.pool.bytes as f64 / 1048576.0,
            self.pool.pages,
            self.pool.page_tokens,
            self.pool.per_token_bytes,
            self.card.reserve_bytes as f64 / 1048576.0,
            self.total_bytes() as f64 / 1048576.0,
            self.card.free_at_boot_bytes as f64 / 1048576.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(free: usize) -> CardPlan {
        CardPlan {
            total_bytes: 24 << 30,
            free_at_boot_bytes: free,
            reserve_bytes: 1 << 30,
        }
    }

    fn pool() -> PoolPlan {
        PoolPlan {
            pages: 4,
            page_tokens: 32,
            tokens: 128,
            per_token_bytes: 32 << 10,
            block_bytes: 1 << 20,
            bytes: 4 << 20,
            gdn_slots: 2,
            gdn_row_bytes: 1 << 20,
        }
    }

    fn fixed() -> Vec<MemComponent> {
        vec![
            MemComponent::new(
                ComponentId::Weights,
                8 << 30,
                PlanSource::Measured,
                Tier::Gpu,
                Owner::System,
            ),
            MemComponent::new(
                ComponentId::GdnGrid,
                2 << 20,
                PlanSource::Measured,
                Tier::Gpu,
                Owner::System,
            ),
            MemComponent::new(
                ComponentId::GdnSnapshots,
                3 << 20,
                PlanSource::Measured,
                Tier::Gpu,
                Owner::Cache,
            ),
            MemComponent::new(
                ComponentId::GraphSlab,
                320 << 20,
                PlanSource::Conservative,
                Tier::Gpu,
                Owner::System,
            ),
            MemComponent::new(
                ComponentId::WarmupMisc,
                138 << 20,
                PlanSource::Conservative,
                Tier::Gpu,
                Owner::System,
            ),
            MemComponent::new(
                ComponentId::DraftPool,
                5 << 20,
                PlanSource::Measured,
                Tier::Gpu,
                Owner::System,
            ),
        ]
    }

    #[test]
    fn projection_keeps_each_fixed_byte_once() {
        let p = MemoryPlan::new(
            0,
            card(16 << 30),
            fixed(),
            pool(),
            TierPlan::default(),
            PlanPolicy { fixed_pool: true },
        )
        .expect("plan");
        let inp = p.to_budget_inputs().expect("projection");
        // runtime_floor = 快照 3MiB + graph 320MiB + warmup 138MiB
        assert_eq!(
            inp.runtime_floor_bytes,
            (3 << 20) + (320 << 20) + (138 << 20)
        );
        assert_eq!(inp.weights_bytes, 8 << 30);
        assert_eq!(inp.draft_pool_bytes, 5 << 20);
        assert_eq!(inp.gdn_row_bytes, 1 << 20);
        assert_eq!(inp.kv_face_bytes, 4 << 20);
        // 分类不重不漏
        assert_eq!(
            inp.weights_bytes
                + inp.runtime_floor_bytes
                + inp.draft_pool_bytes
                + inp.gdn_row_bytes * p.pool.gdn_slots,
            p.fixed_bytes()
        );
    }

    #[test]
    fn card_check_reports_over_free() {
        let p = MemoryPlan::new(
            0,
            card(9 << 30),
            fixed(),
            pool(),
            TierPlan::default(),
            PlanPolicy { fixed_pool: true },
        )
        .expect("plan");
        let c = p.check_card();
        assert!(c.total_bytes > c.card_free_bytes);
        assert!(!c.fits());
        assert_eq!(c.over_free_bytes, c.total_bytes - c.card_free_bytes);
    }

    #[test]
    fn duplicate_fixed_component_is_visible() {
        let mut f = fixed();
        f.push(MemComponent::new(
            ComponentId::GraphSlab,
            1,
            PlanSource::Conservative,
            Tier::Gpu,
            Owner::System,
        ));
        let p = MemoryPlan::new(
            0,
            card(16 << 30),
            f,
            pool(),
            TierPlan::default(),
            PlanPolicy { fixed_pool: true },
        )
        .expect("plan");
        assert_eq!(p.duplicate_components(), vec![ComponentId::GraphSlab]);
    }

    #[test]
    fn fixed_pool_requires_a_pool() {
        let mut pool = pool();
        pool.bytes = 0;
        let err = MemoryPlan::new(
            0,
            card(16 << 30),
            fixed(),
            pool,
            TierPlan::default(),
            PlanPolicy { fixed_pool: true },
        )
        .unwrap_err();
        assert!(matches!(err, StoreError::Budget(_)));
    }

    #[test]
    fn dynamic_pool_projects_zero_face() {
        let p = MemoryPlan::new(
            0,
            card(16 << 30),
            fixed(),
            pool(),
            TierPlan::default(),
            PlanPolicy { fixed_pool: false },
        )
        .expect("plan");
        assert_eq!(p.to_budget_inputs().expect("projection").kv_face_bytes, 0);
    }
}
