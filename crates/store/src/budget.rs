//! 卡级预算分解 + 准入判据(设计卷 §2.2/§2.3/§2.4 的式子,唯一落点)。
//!
//! ```text
//! V_avail(卡)   = V_card − 权重 − 运行底价 − Σ_active GDN 行 − 草稿池
//! N*(dense 上限) = V_avail / k                      ← 容量线①
//! 准入面(face)  = kv_face_bytes > 0 ? 池 : V_avail  ← 固定池档 / 动态池档(§23.2)
//! 准入(turn_i) ⇔ Σ_active floor_j + floor_i + 缓存驻留 + 弹性余量 ≤ face
//!                ∧ #active_turn + 1 ≤ gdn_slots
//! ```
//!
//! **A5.1 律**:`BudgetInputs` 的每一项都必须来自实测;缺项(0)在 `decompose` 里
//! 只允许"确实为 0 的项"(草稿池关/host 关),权重与运行底价为 0 = 分解失败。
//! **例外 = 固定池档**(`kv_face_bytes > 0`):准入判据走池墙,卡级三项可以不实测
//! (记 0,报告面由 [`VramBudget::card_measured`] 标出"未实测";绝不静默)。

use crate::error::{StoreError, StoreResult};
use crate::tier::TierBudget;

/// 预算输入(**全部为实测字节数**;设计卷 §2.2)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BudgetInputs {
    /// 卡可用字节(实测 `cudaMemGetInfo` free,不是标称容量)。
    pub v_card_bytes: usize,
    /// 权重常驻(27B AWQ 实测 ≈17.5GB —— 账面 13.7GB,**2G 待审计**)。
    pub weights_bytes: usize,
    /// 运行底价:图 slab + warmup 瞬态 + **快照池** + 其它非池固定项。
    /// boot 由 [`crate::plan::MemoryPlan::to_budget_inputs`] 从固定项投影,
    /// 不再用两次 `live_bytes()` 的残差反推。
    pub runtime_floor_bytes: usize,
    /// 草稿池(MTP/DFlash2;关 = 0。按 `pool_tokens` 全额预留,是第二主池)。
    pub draft_pool_bytes: usize,
    /// **每活跃 turn** 的 GDN 行(27B:48 层 × 3.38MB ≈ 162MB)。
    pub gdn_row_bytes: usize,
    /// 每 token KV 字节(27B:`k` = 32KiB fp8 / 64KiB f16)。
    pub kv_bytes_per_token: usize,
    /// **页宽(token/页)**。与 `kv_bytes_per_token` 合起来定义页字节
    /// = `page_tokens × kv_bytes_per_token`(`Rel::open` 与 KV 腿几何交叉校验)。
    ///
    /// 为什么进预算入参(A7 收口,2026-10-19):准入地板与池下限算式必须是
    /// **同一把尺** —— 块表/池容量都按页粒度走(`div_ceil(seq/page)`),
    /// 地板若按字节算,页宽或布局一变就会“各说各话”。
    pub page_tokens: usize,
    /// host 层上限(0 = 关层;要求 C4 显式配置)。
    pub host_bytes: usize,
    /// 磁盘层上限(0 = 关层)。
    pub disk_bytes: usize,
    /// **预分配的设备 KV 面**(池字节;设计卷 §23.2)。
    ///
    /// `> 0` = **固定池档**:设备 KV 在 boot 预分配完,准入按池墙判(§2.3 的
    /// `Σ N_j·k + N_i·k + 缓存 ≤ pool·k`);`0` = **动态池档**:按卡级余量
    /// `V_avail` 判(§2.4 式,池随请求增长)。
    pub kv_face_bytes: usize,
}

/// per-turn 预算诉求 = **KVMem 式损失的耐受度**(设计卷 §2.4)。
///
/// **单位 = 页**(页宽由 `BudgetInputs::page_tokens` 定义) —— A7 收口
/// (2026-10-19):旧版是 `Bytes`,与块表/池容量的页口径各算一套;现在地板
/// 与池下限/(链增长)共用同一把尺。字节只在**报告面**由 `floor_bytes` 派生。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TurnFloor {
    /// 零耐受:`dense_floor = ceil(N/page)` 页 ⇒ **禁用选择层**,不足即响亮拒/排队。
    Strict,
    /// 直接显存地板(**页数**);越小 = 越耐受近似。`0` = 允许全量下沉
    /// (仅受窗口墙约束)。高于本 turn 的 `ctx` 页数时按 `ctx` 页数封顶。
    Pages(usize),
}

impl TurnFloor {
    /// 本 turn 的**页数地板**(未封顶的入参 → 实际生效页数)。
    pub fn floor_pages(self, ctx_tokens: usize, page_tokens: usize) -> usize {
        let page = page_tokens.max(1);
        let ctx_pages = ctx_tokens.div_ceil(page);
        match self {
            TurnFloor::Strict => ctx_pages,
            TurnFloor::Pages(n) => n.min(ctx_pages),
        }
    }

    /// 本 turn 的直接显存地板字节数。
    ///
    /// **页粒度**:`floor_pages × page_tokens × kv_bytes_per_token` —— 与
    /// 链增长(`div_ceil`)同源;`ctx` 非页对齐时地板**向上取整到页数**
    /// (与设备上的整页分配一致)。
    pub fn floor_bytes(
        self,
        ctx_tokens: usize,
        page_tokens: usize,
        kv_bytes_per_token: usize,
    ) -> usize {
        let page = page_tokens.max(1);
        self.floor_pages(ctx_tokens, page)
            .saturating_mul(page)
            .saturating_mul(kv_bytes_per_token)
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
    gdn_slots: usize,
    /// 卡级三项是否实测(`v_card`/权重/底价)。固定池档允许缺项 —— 报告面必须能看见。
    card_measured: bool,
}

impl VramBudget {
    /// 分解(唯一入口)。
    ///
    /// 失败条件(A5.1 响亮拒):权重或运行底价为 0(说明没有实测数字)、
    /// `gdn_slots == 0`、或按上限满编的活跃 turn 行都放不下。
    pub fn decompose(inputs: BudgetInputs, gdn_slots: usize) -> StoreResult<Self> {
        if inputs.kv_bytes_per_token == 0 {
            return Err(StoreError::Budget("每 token KV 字节为 0".into()));
        }
        if inputs.page_tokens == 0 {
            return Err(StoreError::Budget(
                "页宽(page_tokens)为 0(地板/池下限无从按页对齐)".into(),
            ));
        }
        if gdn_slots == 0 {
            return Err(StoreError::Budget(
                "gdn_slots = 0(没有任何 turn 能跑)".into(),
            ));
        }
        // 固定池档(池已预分配)⇒ 准入判据不看卡级三项 ⇒ 允许未实测(记 0 + 标记);
        // 动态池档 ⇒ 卡级三项是判据本身,缺项 = 分解失败(A5.1)。
        let fixed_pool = inputs.kv_face_bytes > 0;
        let card_measured = inputs.weights_bytes > 0 && inputs.runtime_floor_bytes > 0;
        if !card_measured && !fixed_pool {
            let miss = if inputs.weights_bytes == 0 {
                "权重字节"
            } else {
                "运行底价"
            };
            return Err(StoreError::Budget(format!(
                "{miss}为 0(缺实测数字,禁进预算式;动态池档无池面兜底)"
            )));
        }
        let rows = inputs
            .gdn_row_bytes
            .checked_mul(gdn_slots)
            .ok_or_else(|| StoreError::Budget("GDN 行预算溢出".into()))?;
        let fixed = inputs
            .weights_bytes
            .checked_add(inputs.runtime_floor_bytes)
            .and_then(|v| v.checked_add(rows))
            .and_then(|v| v.checked_add(inputs.draft_pool_bytes))
            .ok_or_else(|| StoreError::Budget("固定面预算溢出".into()))?;
        if card_measured && fixed >= inputs.v_card_bytes {
            return Err(StoreError::Budget(format!(
                "固定面 {fixed} ≥ 卡可用 {}:装不下权重+底价+{}×{}MB 行+草稿",
                inputs.v_card_bytes,
                gdn_slots,
                inputs.gdn_row_bytes / (1 << 20)
            )));
        }
        Ok(Self {
            inputs,
            gdn_slots,
            card_measured,
        })
    }

    /// 卡级三项(`v_card`/权重/底价)是否实测。固定池档允许缺项 —— 缺了不是
    /// "没这回事",而是**这个档的准入不依赖它**;报告面与日志必须区分。
    pub fn card_measured(&self) -> bool {
        self.card_measured
    }

    /// **准入判据面**(唯一产地;设计卷 §23.2):固定池档 = 预分配池;动态池档 = 卡级余量。
    pub fn face_bytes(&self) -> usize {
        if self.inputs.kv_face_bytes > 0 {
            self.inputs.kv_face_bytes
        } else {
            self.v_avail_bytes()
        }
    }

    /// 当前档位下,满编活跃 turn 时留给 KV 的字节(= 容量线①的分子)。
    pub fn v_avail_bytes(&self) -> usize {
        self.inputs
            .v_card_bytes
            .saturating_sub(self.inputs.weights_bytes)
            .saturating_sub(self.inputs.runtime_floor_bytes)
            .saturating_sub(self.inputs.gdn_row_bytes.saturating_mul(self.gdn_slots))
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
    pub fn gdn_slots(&self) -> usize {
        self.gdn_slots
    }

    /// 三层预算(交给 `TierLedger`)。
    pub fn tier_budget(&self) -> TierBudget {
        TierBudget {
            host_bytes: self.inputs.host_bytes,
            disk_bytes: self.inputs.disk_bytes,
        }
    }
}

/// 活跃 turn 的在飞占用(准入面的状态;方法自肃,只改自身)。
///
/// 四个面(§23.2 拆分规则):
/// ```text
/// reserved     = Σ_active turn 的实占地板(不可动;G2b 保留额)
/// session_held = idle 会话还持着的 KV 历史(不可动;目标模型 §2.3 里它应可动)
/// cache        = 纯缓存面(可动/可驱逐/可下沉)
/// elastic      = 地板之上的灵活面(可动;B 轴预留)
/// ⇒ committed = reserved + session_held + cache
/// ⇒ evictable = cache + elastic
/// ```
/// 固定池档下引擎同步后 `committed = 池实占`(**精确**,不双算)。
#[derive(Debug, Default)]
pub struct Admission {
    active: std::collections::BTreeMap<u64, ActiveTurn>,
    /// 缓存面驻留(可驱逐)。
    cache_bytes: usize,
    /// 非 turn 归属的**不可动面**(idle 会话持着的历史;引擎段边界同步)。
    session_held_bytes: usize,
    /// 活跃块槽宽(字节;`0` = 未接块身份,退回按 turn floor 求和)。
    slot_bytes: usize,
}

#[derive(Clone, Debug, Default)]
struct ActiveTurn {
    floor_bytes: usize,
    elastic_bytes: usize,
    /// 本 turn 持有的设备块槽(共享前缀会出现在多个 turn 的集合里,
    /// `Admission::reserved_bytes` 取并集,不按会话表长度重复计数)。
    slots: std::collections::BTreeSet<u32>,
}

impl Admission {
    /// 空账。
    pub fn new() -> Self {
        Self::default()
    }

    /// 登记一个活跃 turn(准入前置:调用方必须先用 [`check_admission`] 判过)。
    pub fn admit(
        &mut self,
        turn: u64,
        floor_bytes: usize,
        elastic_bytes: usize,
    ) -> StoreResult<()> {
        if self.active.contains_key(&turn) {
            return Err(StoreError::Invariant(format!("turn {turn} 已在活跃集")));
        }
        self.active.insert(
            turn,
            ActiveTurn {
                floor_bytes,
                elastic_bytes,
                ..ActiveTurn::default()
            },
        );
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

    /// **保留额更新**(活跃 turn 的设备实占;设计卷 §23.2):turn 增长过程中
    /// 由引擎在段边界同步(与 `set_elastic` 同形 —— 登记一次、更新多次)。
    ///
    /// 为什么不用 `admit` 重登记:重登记会丢掉 `elastic`(两个字段各有语义,
    /// 更新旧一个不许顺手清另一个)。
    pub fn set_floor(&mut self, turn: u64, bytes: usize) -> StoreResult<()> {
        let t = self
            .active
            .get_mut(&turn)
            .ok_or_else(|| StoreError::Unknown(format!("turn {turn}")))?;
        t.floor_bytes = bytes;
        Ok(())
    }

    /// **活跃块集合更新**(段边界;P0-3)。
    ///
    /// `slots` = 本 turn 当前持有的设备块槽号;多 turn 共享前缀时同一槽号
    /// 出现在多个集合里,保留面按**并集**计,不再把共享块算两次。
    /// `slot_bytes` 在首次调用时定档;后续不一致 = 装配错(响亮拒)。
    pub fn set_active_slots(
        &mut self,
        turn: u64,
        slots: impl IntoIterator<Item = u32>,
        slot_bytes: usize,
    ) -> StoreResult<()> {
        if slot_bytes == 0 {
            return Err(StoreError::Invariant(
                "set_active_slots: slot_bytes = 0".into(),
            ));
        }
        if self.slot_bytes == 0 {
            self.slot_bytes = slot_bytes;
        } else if self.slot_bytes != slot_bytes {
            return Err(StoreError::Invariant(format!(
                "set_active_slots: 槽宽漂移 {} ≠ {slot_bytes}",
                self.slot_bytes
            )));
        }
        let blocks: std::collections::BTreeSet<u32> = slots.into_iter().collect();
        let t = self
            .active
            .get_mut(&turn)
            .ok_or_else(|| StoreError::Unknown(format!("turn {turn}")))?;
        t.floor_bytes = blocks.len().saturating_mul(slot_bytes);
        t.slots = blocks;
        Ok(())
    }

    /// 缓存面驻留登记(入账/出账由调用方配对)。
    pub fn set_cache_bytes(&mut self, bytes: usize) {
        self.cache_bytes = bytes;
    }

    /// **非 turn 归属的不可动面**(例:idle 会话还持着的 KV 历史)。
    ///
    /// 它必须单独占一格:算进 `cache` ⇒ 准入会先把话说满(`EvictThen`)
    /// 而搬家时 `Busy`;算进 `reserved` ⇒ 保留额报告失真(G2b)。
    pub fn set_session_held(&mut self, bytes: usize) {
        self.session_held_bytes = bytes;
    }

    /// 活跃 turn 数。
    pub fn active_turns(&self) -> usize {
        self.active.len()
    }

    /// 保留面合计(**唯一**活跃块字节 —— 共享前缀只计一次)。
    ///
    /// 未接块身份(`slot_bytes = 0`,旧测试/非块资产)= 退回各 turn floor 之和。
    pub fn reserved_bytes(&self) -> usize {
        if self.slot_bytes == 0 {
            return self.active.values().map(|t| t.floor_bytes).sum();
        }
        let union: std::collections::BTreeSet<u32> = self
            .active
            .values()
            .flat_map(|t| t.slots.iter().copied())
            .collect();
        union.len().saturating_mul(self.slot_bytes)
    }

    /// 弹性面合计(可驱逐)。
    pub fn elastic_bytes(&self) -> usize {
        self.active.values().map(|t| t.elastic_bytes).sum()
    }

    /// 缓存面(可驱逐)。
    pub fn cache_bytes(&self) -> usize {
        self.cache_bytes
    }

    /// 非 turn 归属的不可动面(idle 会话持着的历史)。
    pub fn session_held_bytes(&self) -> usize {
        self.session_held_bytes
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
/// ① 并发天花板(`gdn_slots`)→ ② 直接地板能否放下(含驱逐面)→
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
    if admission.active_turns() >= budget.gdn_slots() {
        let reason = format!(
            "活跃 turn 已达天花板({}/{})",
            admission.active_turns(),
            budget.gdn_slots()
        );
        return verdict_for_full(on_full, reason);
    }
    let floor_bytes = floor.floor_bytes(ctx_tokens, inp.page_tokens, inp.kv_bytes_per_token);
    let ctx_pages = ctx_tokens.div_ceil(inp.page_tokens.max(1));
    // 已用 = 保留面 + 会话持有面 + 弹性面 + 缓存面;固定面(权重/底价/行/草稿)已从 V_avail 扣掉
    let committed = admission.reserved_bytes()
        + admission.session_held_bytes()
        + admission.elastic_bytes()
        + admission.cache_bytes();
    // 判据面 = 固定池档的池墙 / 动态池档的卡级余量(§23.2,一产地)
    let face = budget.face_bytes();
    if committed.saturating_add(floor_bytes) <= face {
        // 地板放得下。`Strict` 的语义已由 `floor_pages = ctx 页数` 直接兑现
        // (A7:不再需要“精确 ctx 字节 > 地板”的二次判定 —— 两者同源)。
        return AdmissionVerdict::Admit { floor_bytes };
    }
    let short = (committed + floor_bytes).saturating_sub(face);
    if evictable_bytes >= short {
        return AdmissionVerdict::EvictThen {
            floor_bytes,
            need_bytes: short,
        };
    }
    let pages = floor.floor_pages(ctx_tokens, inp.page_tokens);
    let reason = if floor.is_strict() {
        format!(
            "Strict turn(零耐受)显存不足:需 {pages} 页(={floor_bytes} 字节,ctx {ctx_tokens} tok = {ctx_pages} 页,欠 {short}),可驱逐面仅 {evictable_bytes}"
        )
    } else {
        format!(
            "显存不足:需 {pages} 页(={floor_bytes} 字节,ctx {ctx_tokens} tok,欠 {short}),可驱逐面仅 {evictable_bytes}"
        )
    };
    if floor.is_strict() {
        // 零耐受:不许降级 ⇒ 拒(或按策略排队)
        verdict_for_full(on_full, reason)
    } else {
        // 有耐受:把地板压到"能给的量",并把差额推给三层(下限 = 0)
        let grantable = face.saturating_sub(committed);
        if grantable == 0 {
            return verdict_for_full(on_full, reason);
        }
        AdmissionVerdict::Admit {
            floor_bytes: grantable,
        }
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
            page_tokens: 32,
            host_bytes: 16 << 30,
            disk_bytes: 0,
            kv_face_bytes: 0,
        };
        VramBudget::decompose(inp, max_turns).expect("分解")
    }

    /// **固定池档**:池已预分配 ⇒ 准入按池墙判;卡级三项未实测也允许装配
    /// (但报告面必须能看见 —— `card_measured() == false`)。
    fn pool_budget(max_turns: usize, face: usize) -> VramBudget {
        let inp = BudgetInputs {
            v_card_bytes: 0,
            weights_bytes: 0,
            runtime_floor_bytes: 0,
            draft_pool_bytes: 0,
            gdn_row_bytes: 1,
            kv_bytes_per_token: 32 * 1024,
            page_tokens: 32,
            host_bytes: 0,
            disk_bytes: 0,
            kv_face_bytes: face,
        };
        VramBudget::decompose(inp, max_turns).expect("固定池档装配")
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
            page_tokens: 32,
            host_bytes: 0,
            disk_bytes: 0,
            kv_face_bytes: 0, // 也没有池面兑底 ⇒ 必须拒
        };
        assert!(matches!(
            VramBudget::decompose(inp, 1),
            Err(StoreError::Budget(_))
        ));
    }

    #[test]
    fn fixed_pool_absorbs_unmeasured_card_but_says_so() {
        let b = pool_budget(1, 1 << 20);
        assert!(!b.card_measured(), "未实测必须可查（不许静默）");
        assert_eq!(b.face_bytes(), 1 << 20, "判据面 = 池");
        assert_eq!(b.v_avail_bytes(), 0, "卡级余量确实为 0（没实测）");
    }

    #[test]
    fn pool_face_is_the_admission_wall() {
        let b = pool_budget(2, 4 << 20); // 池 4MiB；k=32KiB ⇒ 128 tok 容量
        let a = Admission::new();
        // 128 tok × 32KiB = 4MiB ⇒ 刚好装下
        assert!(matches!(
            check_admission(&b, &a, TurnFloor::Strict, 128, 0, QueuePolicy::Reject),
            AdmissionVerdict::Admit { .. }
        ));
        // 129 tok ⇒ 超池 ⇒ 响亮拒（卡级 v_avail=0 时不误报 Admit）
        match check_admission(&b, &a, TurnFloor::Strict, 129, 0, QueuePolicy::Reject) {
            AdmissionVerdict::Reject { reason } => assert!(reason.contains("Strict"), "{reason}"),
            other => panic!("应拒，得 {other:?}"),
        }
        // 可动面够 ⇒ 先腾后进
        let mut a2 = Admission::new();
        a2.set_cache_bytes(4 << 20); // 池满（全在缓存面）
        let v = check_admission(
            &b,
            &a2,
            TurnFloor::Strict,
            128,
            4 << 20,
            QueuePolicy::Reject,
        );
        match v {
            AdmissionVerdict::EvictThen { need_bytes, .. } => assert_eq!(need_bytes, 4 << 20),
            other => panic!("应先腾后进，得 {other:?}"),
        }
    }

    #[test]
    fn committed_is_exactly_pool_occupancy() {
        // §23.2 拆分规则:reserved(活跃实占) + cache(其余池实占) = 池实占
        let b = pool_budget(2, 8 << 20);
        let mut a = Admission::new();
        a.admit(1, 0, 0).expect("登记");
        a.set_floor(1, 2 << 20).expect("实占更新");
        a.set_cache_bytes(3 << 20);
        assert_eq!(a.reserved_bytes(), 2 << 20);
        assert_eq!(a.cache_bytes(), 3 << 20);
        // 池 8MiB，已 commit 5MiB ⇒ 再要 3MiB 刚好；要 3MiB+1 就超
        assert!(matches!(
            check_admission(&b, &a, TurnFloor::Strict, 96, 0, QueuePolicy::Reject),
            AdmissionVerdict::Admit { .. }
        ));
        assert!(matches!(
            check_admission(&b, &a, TurnFloor::Strict, 97, 0, QueuePolicy::Reject),
            AdmissionVerdict::Reject { .. }
        ));
    }

    #[test]
    fn v_avail_and_capacity_match_hand_math() {
        let b = budget(2);
        // 25.0 − 17.5 − 2.0 − 2×0.162 = 5.176GB
        assert_eq!(
            b.v_avail_bytes(),
            25_000_000_000 - 17_500_000_000 - 2_000_000_000 - 324_000_000
        );
        // N* = 5.176GB / 32KiB ≈ 157k tok
        let n_star = b.kv_tokens_capacity();
        assert!((150_000..165_000).contains(&n_star), "N* = {n_star}");
    }

    /// **A7 收口(2026-10-19)**:地板是**页口径** —— 与链增长/池下限同一把尺。
    #[test]
    fn floor_is_page_granular_and_capped() {
        // 页 = 32 tok × 32KiB = 1 MiB
        assert_eq!(
            TurnFloor::Strict.floor_pages(100, 32),
            4,
            "100 tok 占 4 页(ceil)"
        );
        assert_eq!(
            TurnFloor::Strict.floor_bytes(100, 32, 32 * 1024),
            4 << 20,
            "Strict = 整页粒度(不是精确 100 tok)"
        );
        assert_eq!(TurnFloor::Strict.floor_pages(96, 32), 3, "对齐不浪费页");
        assert_eq!(TurnFloor::Pages(2).floor_pages(100, 32), 2);
        assert_eq!(
            TurnFloor::Pages(9).floor_pages(100, 32),
            4,
            "高于 ctx 页数 ⇒ 封顶到 ctx"
        );
        assert_eq!(
            TurnFloor::Pages(0).floor_bytes(100, 32, 32 * 1024),
            0,
            "0 页 = 允许全量下沉(不是“未设”)"
        );
        assert_eq!(
            TurnFloor::Pages(1).floor_bytes(0, 32, 32 * 1024),
            0,
            "零长 ctx = 零地板"
        );
    }

    #[test]
    fn decompose_rejects_zero_page_width() {
        let mut inp = pool_budget(1, 1 << 20).inputs();
        inp.page_tokens = 0;
        assert!(matches!(
            VramBudget::decompose(inp, 1),
            Err(StoreError::Budget(_))
        ));
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
        let floor = TurnFloor::Pages(64);
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
        let v = check_admission(&b, &a, TurnFloor::Pages(1), 4096, 0, QueuePolicy::Reject);
        // 剩余 0 ⇒ 拒(不会给出 0 字节地板假装成功)
        assert!(matches!(v, AdmissionVerdict::Reject { .. }), "得 {v:?}");
    }

    #[test]
    fn evictable_face_can_cover_the_gap() {
        let b = budget(1);
        let mut a = Admission::new();
        let v_avail = b.v_avail_bytes();
        a.set_cache_bytes(v_avail - (1 << 20)); // 缓存几乎占满
        let v = check_admission(
            &b,
            &a,
            TurnFloor::Pages(4),
            8192,
            8 << 20,
            QueuePolicy::Reject,
        );
        assert!(matches!(v, AdmissionVerdict::EvictThen { .. }), "得 {v:?}");
    }

    /// **P0-3**:两个活跃 turn 共享前缀时,保留面按槽号并集计,
    /// 第三个同尺寸 turn 不应因重复计数被误拒。
    #[test]
    fn shared_prefix_is_counted_once() {
        let b = pool_budget(3, 4 << 20); // 4MiB 池;1MiB/块
        let mut a = Admission::new();
        a.admit(1, 0, 0).expect("t1");
        a.admit(2, 0, 0).expect("t2");
        a.set_active_slots(1, [1u32, 2], 1 << 20).expect("t1 slots");
        a.set_active_slots(2, [1u32, 2, 3], 1 << 20)
            .expect("t2 slots");
        assert_eq!(a.reserved_bytes(), 3 << 20, "共享块只计一次");
        // 第三 turn 1 块:3MiB + 1MiB = 4MiB,刚好准入(双计则会 5MiB 被拒)
        assert!(matches!(
            check_admission(&b, &a, TurnFloor::Strict, 32, 0, QueuePolicy::Reject),
            AdmissionVerdict::Admit { .. }
        ));
        // 释放共享方之一,并集仍覆盖另一 turn 的全部块
        a.release(1).expect("release");
        assert_eq!(a.reserved_bytes(), 3 << 20);
    }

    #[test]
    fn concurrency_ceiling_is_hard() {
        let b = budget(1);
        let mut a = Admission::new();
        a.admit(1, 1 << 20, 0).expect("第一个");
        let v = check_admission(
            &b,
            &a,
            TurnFloor::Pages(1),
            32,
            1 << 30,
            QueuePolicy::Reject,
        );
        match v {
            AdmissionVerdict::Reject { reason } => assert!(reason.contains("天花板"), "{reason}"),
            other => panic!("应拒,得 {other:?}"),
        }
        let q = check_admission(
            &b,
            &a,
            TurnFloor::Pages(1),
            32,
            1 << 30,
            QueuePolicy::Queue { max_wait_ms: 50 },
        );
        assert!(matches!(q, AdmissionVerdict::Queue { .. }), "得 {q:?}");
    }
}
