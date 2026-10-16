//! 卡级预算分解 + 准入判据(设计卷 §2.2/§2.3/§2.4 的式子,唯一落点)。
//!
//! ```text
//! V_avail(卡)   = V_card − 权重 − 运行底价 − Σ_active GDN 行 − 草稿池
//! N*(dense 上限) = V_avail / k                      ← 容量线①
//! 准入(turn_i) ⇔ Σ_active floor_j + floor_i + 缓存驻留 + 弹性余量 + 运行底价 ≤ V_card
//!                ∧ #active_turn + 1 ≤ max_active_turns
//! ```
//!
//! **A5.1 律**:`BudgetInputs` 的每一项都必须来自实测;缺项(0)在 `decompose` 里
//! 只允许"确实为 0 的项"(草稿池关/host 关),权重与运行底价为 0 = 分解失败。

use crate::error::{StoreError, StoreResult};
use crate::tier::TierBudget;

/// 预算输入(**全部为实测字节数**;设计卷 §2.2)。
#[derive(Clone, Copy, Debug)]
pub struct BudgetInputs {
    /// 卡可用字节(实测 `cudaMemGetInfo` free,不是标称容量)。
    pub v_card_bytes: usize,
    /// 权重常驻(27B AWQ 实测 ≈17.5GB —— 账面 13.7GB,**2G 待审计**)。
    pub weights_bytes: usize,
    /// 运行底价:图 slab + 激活峰值 + 上下文 + warmup 瞬态(实测)。
    pub runtime_floor_bytes: usize,
    /// 草稿池(MTP/DFlash2;关 = 0。按 `pool_tokens` 全额预留,是第二主池)。
    pub draft_pool_bytes: usize,
    /// **每活跃 turn** 的 GDN 行(27B:48 层 × 3.38MB ≈ 162MB)。
    pub gdn_row_bytes: usize,
    /// 每 token KV 字节(27B:`k` = 32KiB fp8 / 64KiB f16)。
    pub kv_bytes_per_token: usize,
    /// host 层上限(0 = 关层;要求 C4 显式配置)。
    pub host_bytes: usize,
    /// 磁盘层上限(0 = 关层)。
    pub disk_bytes: usize,
}

/// per-turn 预算诉求 = **KVMem 式损失的耐受度**(设计卷 §2.4)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnFloor {
    /// 零耐受:`dense_floor = N·k` ⇒ **禁用选择层**,不足即响亮拒/排队。
    Strict,
    /// 直接显存地板(字节);越小 = 越耐受近似。`0` = 允许全量下沉(仅受窗口墙约束)。
    Bytes(usize),
}

impl TurnFloor {
    /// 本 turn 的直接显存地板字节数。
    pub fn floor_bytes(self, ctx_tokens: usize, kv_bytes_per_token: usize) -> usize {
        match self {
            TurnFloor::Strict => ctx_tokens.saturating_mul(kv_bytes_per_token),
            TurnFloor::Bytes(b) => b.min(ctx_tokens.saturating_mul(kv_bytes_per_token)),
        }
    }

    /// 是否零耐受(禁选择层)。
    pub fn is_strict(self) -> bool {
        matches!(self, TurnFloor::Strict)
    }
}

/// 卡级预算分解结果。
#[derive(Clone, Copy, Debug)]
pub struct VramBudget {
    inputs: BudgetInputs,
    /// 本档允许的活跃 turn 上限(boot 旋钮;硬天花板)。
    max_active_turns: usize,
}

impl VramBudget {
    /// 分解(唯一入口)。
    ///
    /// 失败条件(A5.1 响亮拒):权重或运行底价为 0(说明没有实测数字)、
    /// `max_active_turns == 0`、或按上限满编的活跃 turn 行都放不下。
    pub fn decompose(inputs: BudgetInputs, max_active_turns: usize) -> StoreResult<Self> {
        if inputs.weights_bytes == 0 {
            return Err(StoreError::Budget("权重字节为 0(缺实测数字,禁进预算式)".into()));
        }
        if inputs.runtime_floor_bytes == 0 {
            return Err(StoreError::Budget("运行底价为 0(缺实测数字,禁进预算式)".into()));
        }
        if inputs.kv_bytes_per_token == 0 {
            return Err(StoreError::Budget("每 token KV 字节为 0".into()));
        }
        if max_active_turns == 0 {
            return Err(StoreError::Budget("max_active_turns = 0(没有任何 turn 能跑)".into()));
        }
        let rows = inputs
            .gdn_row_bytes
            .checked_mul(max_active_turns)
            .ok_or_else(|| StoreError::Budget("GDN 行预算溢出".into()))?;
        let fixed = inputs
            .weights_bytes
            .checked_add(inputs.runtime_floor_bytes)
            .and_then(|v| v.checked_add(rows))
            .and_then(|v| v.checked_add(inputs.draft_pool_bytes))
            .ok_or_else(|| StoreError::Budget("固定面预算溢出".into()))?;
        if fixed >= inputs.v_card_bytes {
            return Err(StoreError::Budget(format!(
                "固定面 {fixed} ≥ 卡可用 {}:装不下权重+底价+{}×{}MB 行+草稿",
                inputs.v_card_bytes,
                max_active_turns,
                inputs.gdn_row_bytes / (1 << 20)
            )));
        }
        Ok(Self { inputs, max_active_turns })
    }

    /// 当前档位下,满编活跃 turn 时留给 KV 的字节(= 容量线①的分子)。
    pub fn v_avail_bytes(&self) -> usize {
        self.inputs
            .v_card_bytes
            .saturating_sub(self.inputs.weights_bytes)
            .saturating_sub(self.inputs.runtime_floor_bytes)
            .saturating_sub(self.inputs.gdn_row_bytes.saturating_mul(self.max_active_turns))
            .saturating_sub(self.inputs.draft_pool_bytes)
    }

    /// 容量线①:`N*` = 当前档位下**dense 全量**能放下的最大 token 数。
    pub fn kv_tokens_capacity(&self) -> usize {
        self.v_avail_bytes() / self.inputs.kv_bytes_per_token.max(1)
    }

    /// 输入(报告/断言用)。
    pub fn inputs(&self) -> BudgetInputs {
        self.inputs
    }

    /// 活跃 turn 上限(硬天花板)。
    pub fn max_active_turns(&self) -> usize {
        self.max_active_turns
    }

    /// 三层预算(交给 `TierLedger`)。
    pub fn tier_budget(&self) -> TierBudget {
        TierBudget { host_bytes: self.inputs.host_bytes, disk_bytes: self.inputs.disk_bytes }
    }
}

/// 活跃 turn 的在飞占用(准入面的状态;方法自肃,只改自身)。
#[derive(Debug, Default)]
pub struct Admission {
    active: std::collections::BTreeMap<u64, ActiveTurn>,
    /// 缓存面驻留(可驱逐)。
    cache_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct ActiveTurn {
    floor_bytes: usize,
    elastic_bytes: usize,
}

impl Admission {
    /// 空账。
    pub fn new() -> Self {
        Self::default()
    }

    /// 登记一个活跃 turn(准入前置:调用方必须先用 [`check_admission`] 判过)。
    pub fn admit(&mut self, turn: u64, floor_bytes: usize, elastic_bytes: usize) -> StoreResult<()> {
        if self.active.contains_key(&turn) {
            return Err(StoreError::Invariant(format!("turn {turn} 已在活跃集")));
        }
        self.active.insert(turn, ActiveTurn { floor_bytes, elastic_bytes });
        Ok(())
    }

    /// 释放(终结)一个 turn。
    pub fn release(&mut self, turn: u64) -> StoreResult<(usize, usize)> {
        let t = self
            .active
            .remove(&turn)
            .ok_or_else(|| StoreError::Unknown(format!("turn {turn}")))?;
        Ok((t.floor_bytes, t.elastic_bytes))
    }

    /// 弹性余量增删(池压力下回收/恢复;地板不动 —— 保留额不变量)。
    pub fn set_elastic(&mut self, turn: u64, bytes: usize) -> StoreResult<()> {
        let t = self
            .active
            .get_mut(&turn)
            .ok_or_else(|| StoreError::Unknown(format!("turn {turn}")))?;
        t.elastic_bytes = bytes;
        Ok(())
    }

    /// 缓存面驻留登记(入账/出账由调用方配对)。
    pub fn set_cache_bytes(&mut self, bytes: usize) {
        self.cache_bytes = bytes;
    }

    /// 活跃 turn 数。
    pub fn active_turns(&self) -> usize {
        self.active.len()
    }

    /// 保留面合计(活跃 turn 的地板之和 —— **不可驱逐**面)。
    pub fn reserved_bytes(&self) -> usize {
        self.active.values().map(|t| t.floor_bytes).sum()
    }

    /// 弹性面合计(可驱逐)。
    pub fn elastic_bytes(&self) -> usize {
        self.active.values().map(|t| t.elastic_bytes).sum()
    }

    /// 缓存面(可驱逐)。
    pub fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }
}

/// 准入判决(结构化:调用方据此决策,不解析字符串)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionVerdict {
    /// 直接准入(余量够)。
    Admit {
        /// 授予的直接显存地板(字节)。
        floor_bytes: usize,
    },
    /// 先驱逐再准入(差额 = `need_bytes`,可动面 = 缓存 + 弹性余量)。
    EvictThen {
        /// 授予的地板。
        floor_bytes: usize,
        /// 需腾出的字节。
        need_bytes: usize,
    },
    /// 排队(策略允许等待;资源可能稍后释放)。
    Queue {
        /// 原因(给人看)。
        reason: String,
    },
    /// **响亮拒**(`Strict` 且装不下 —— 绝不降级成近似)。
    Reject {
        /// 原因(含具体数字)。
        reason: String,
    },
}

/// 准入判据(自由函数:I/O 与判定分离;`Admission` 只提供状态)。
///
/// 判据顺序(设计卷 §2.4 准入式):
/// ① 并发天花板(`max_active_turns`)→ ② 直接地板能否放下(含驱逐面)→
/// ③ `Strict` 时还需"全量在卡"(不满足即拒/排队)。
pub fn check_admission(
    budget: &VramBudget,
    admission: &Admission,
    floor: TurnFloor,
    ctx_tokens: usize,
    evictable_bytes: usize,
    on_full: crate::policy::QueuePolicy,
) -> AdmissionVerdict {
    let inp = budget.inputs();
    if admission.active_turns() >= budget.max_active_turns() {
        let reason = format!(
            "活跃 turn 已达天花板({}/{})",
            admission.active_turns(),
            budget.max_active_turns()
        );
        return verdict_for_full(on_full, reason);
    }
    let floor_bytes = floor.floor_bytes(ctx_tokens, inp.kv_bytes_per_token);
    let ctx_bytes = ctx_tokens.saturating_mul(inp.kv_bytes_per_token);
    // 已用 = 保留面 + 弹性面 + 缓存面;固定面(权重/底价/行/草稿)已从 V_avail 扣掉
    let committed = admission.reserved_bytes() + admission.elastic_bytes() + admission.cache_bytes();
    let v_avail = budget.v_avail_bytes();
    if committed.saturating_add(floor_bytes) <= v_avail {
        // 地板放得下;Strict 还需全量
        if floor.is_strict() && ctx_bytes > floor_bytes {
            return AdmissionVerdict::Reject {
                reason: format!(
                    "Strict turn 需全量 {ctx_bytes} 字节,地板仅 {floor_bytes}(语义:零耐受 ⇏ 禁选择层)"
                ),
            };
        }
        return AdmissionVerdict::Admit { floor_bytes };
    }
    let short = (committed + floor_bytes).saturating_sub(v_avail);
    if evictable_bytes >= short {
        return AdmissionVerdict::EvictThen { floor_bytes, need_bytes: short };
    }
    let reason = if floor.is_strict() {
        format!(
            "Strict turn(零耐受)显存不足:需 {floor_bytes} 字节(欠 {short}),可驱逐面仅 {evictable_bytes};ctx={ctx_tokens} tok"
        )
    } else {
        format!(
            "显存不足:需 {floor_bytes} 字节(欠 {short}),可驱逐面仅 {evictable_bytes};ctx={ctx_tokens} tok"
        )
    };
    if floor.is_strict() {
        // 零耐受:不许降级 ⇒ 拒(或按策略排队)
        verdict_for_full(on_full, reason)
    } else {
        // 有耐受:把地板压到"能给的量",并把差额推给三层(下限 = 0)
        let grantable = v_avail.saturating_sub(committed);
        if grantable == 0 {
            return verdict_for_full(on_full, reason);
        }
        AdmissionVerdict::Admit { floor_bytes: grantable }
    }
}

fn verdict_for_full(on_full: crate::policy::QueuePolicy, reason: String) -> AdmissionVerdict {
    match on_full {
        crate::policy::QueuePolicy::Reject => AdmissionVerdict::Reject { reason },
        crate::policy::QueuePolicy::Queue { max_wait_ms } => AdmissionVerdict::Queue {
            reason: format!("{reason}(排队 ≤{max_wait_ms}ms 等待释放)"),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::QueuePolicy;

    fn budget(max_turns: usize) -> VramBudget {
        // 24G 卡档:卡 25.0GB 可用;权重 17.5GB;底价 2.0GB;行 162MB;fp8 k=32KiB;草稿 0
        let inp = BudgetInputs {
            v_card_bytes: 25_000_000_000,
            weights_bytes: 17_500_000_000,
            runtime_floor_bytes: 2_000_000_000,
            draft_pool_bytes: 0,
            gdn_row_bytes: 162_000_000,
            kv_bytes_per_token: 32 * 1024,
            host_bytes: 16 << 30,
            disk_bytes: 0,
        };
        VramBudget::decompose(inp, max_turns).expect("分解")
    }

    #[test]
    fn decompose_rejects_missing_measurements() {
        let inp = BudgetInputs {
            v_card_bytes: 25_000_000_000,
            weights_bytes: 0, // 没实测
            runtime_floor_bytes: 1,
            draft_pool_bytes: 0,
            gdn_row_bytes: 1,
            kv_bytes_per_token: 1,
            host_bytes: 0,
            disk_bytes: 0,
        };
        assert!(matches!(VramBudget::decompose(inp, 1), Err(StoreError::Budget(_))));
    }

    #[test]
    fn v_avail_and_capacity_match_hand_math() {
        let b = budget(2);
        // 25.0 − 17.5 − 2.0 − 2×0.162 = 5.176GB
        assert_eq!(b.v_avail_bytes(), 25_000_000_000 - 17_500_000_000 - 2_000_000_000 - 324_000_000);
        // N* = 5.176GB / 32KiB ≈ 157k tok
        let n_star = b.kv_tokens_capacity();
        assert!((150_000..165_000).contains(&n_star), "N* = {n_star}");
    }

    #[test]
    fn strict_turn_beyond_capacity_is_rejected_loudly() {
        let b = budget(1);
        let a = Admission::new();
        // 262k × 32KiB = 8.59GB > v_avail(≈5.3GB)⇒ 零耐受只能拒
        let v = check_admission(&b, &a, TurnFloor::Strict, 262_144, 0, QueuePolicy::Reject);
        match v {
            AdmissionVerdict::Reject { reason } => assert!(reason.contains("Strict"), "{reason}"),
            other => panic!("应拒,得 {other:?}"),
        }
    }

    #[test]
    fn tolerant_turn_gets_grantable_floor() {
        let b = budget(1);
        let a = Admission::new();
        // 地板只要 2.1GB(64K 执行视图)⇒ 放得下 ⇒ 准入
        let floor = TurnFloor::Bytes(64 * 1024 * 32 * 1024 / 1024);
        let v = check_admission(&b, &a, floor, 262_144, 0, QueuePolicy::Reject);
        assert!(matches!(v, AdmissionVerdict::Admit { .. }), "得 {v:?}");
    }

    #[test]
    fn tolerant_turn_is_squeezed_when_tight_but_never_below_available() {
        let b = budget(2);
        let mut a = Admission::new();
        // 两个满编 turn 把 v_avail 吃掉
        let v_avail = b.v_avail_bytes();
        a.admit(1, v_avail, 0).expect("占满");
        let v = check_admission(&b, &a, TurnFloor::Bytes(1 << 20), 4096, 0, QueuePolicy::Reject);
        // 剩余 0 ⇒ 拒(不会给出 0 字节地板假装成功)
        assert!(matches!(v, AdmissionVerdict::Reject { .. }), "得 {v:?}");
    }

    #[test]
    fn evictable_face_can_cover_the_gap() {
        let b = budget(1);
        let mut a = Admission::new();
        let v_avail = b.v_avail_bytes();
        a.set_cache_bytes(v_avail - (1 << 20)); // 缓存几乎占满
        let v = check_admission(&b, &a, TurnFloor::Bytes(4 << 20), 8192, 8 << 20, QueuePolicy::Reject);
        assert!(matches!(v, AdmissionVerdict::EvictThen { .. }), "得 {v:?}");
    }

    #[test]
    fn concurrency_ceiling_is_hard() {
        let b = budget(1);
        let mut a = Admission::new();
        a.admit(1, 1 << 20, 0).expect("第一个");
        let v = check_admission(&b, &a, TurnFloor::Bytes(1), 32, 1 << 30, QueuePolicy::Reject);
        match v {
            AdmissionVerdict::Reject { reason } => assert!(reason.contains("天花板"), "{reason}"),
            other => panic!("应拒,得 {other:?}"),
        }
        let q = check_admission(&b, &a, TurnFloor::Bytes(1), 32, 1 << 30, QueuePolicy::Queue { max_wait_ms: 50 });
        assert!(matches!(q, AdmissionVerdict::Queue { .. }), "得 {q:?}");
    }
}
