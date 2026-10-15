//! 图族域(2026-10-14 拆分自 running.rs:RunningEngine 万能结构体按
//! 状态域拆解 —— 本模块收编**全部 GraphPlan 实例**:主会话图 + spec
//! 三族图桶形阵 + 捕获三态结果)。
//!
//! 图生命周期纪律(charter A1):实例化后禁 trim/池属性变更;相位
//! 定相(perf tag 维度)须同步全部图 —— [`GraphFleet::set_phase`] 是
//! 唯一扇出点,禁止散点逐图设置(漏一张 = perf 归因错相)。

use owl_iface::contract::DeviceClient;

use crate::graph_plan::{GraphPlan, PlanOutcome};

/// 图族(组合叶;RunningEngine 持有)。字段 pub(crate) 供 exec 直读
/// (`graphs.session.face_mut()` 等),相位扇出走方法。
pub(crate) struct GraphFleet<D: DeviceClient> {
    /// 主会话图(prefill/decode 主路径;槽 = frontier/pos/kv_len/…)
    pub session: GraphPlan<D>,
    /// 捕获三态结果(Captured = 回放态;EagerFallback = 同闭包直发)
    pub capture_outcome: PlanOutcome,
    /// verify 桶形图(E5-M4;T = depth+1 定形,输出 tok/hid + fold 记录;
    /// None = eager fallback 走旧 verify+重放路)
    pub verify: Option<GraphPlan<D>>,
    /// fold 桶形图族(E5-M5;index = m,状态 = spec 快照 buf 就地,
    /// slot 恒 0;重放后 restore 回写会话格)
    pub fold: Vec<GraphPlan<D>>,
    /// propose 桶形图族(E5-M5;index = m,extend m+1 行 + k-1 链步,
    /// 输出 drafts;首轮 propose_first 保持 eager)
    pub propose: Vec<GraphPlan<D>>,
    /// DFlash2 propose 单图(E5-DF4;encode 全 8 行 + 噪声块 + selector,
    /// 固定几何;None = eager 回退)
    pub dflash: Option<GraphPlan<D>>,
}

impl<D: DeviceClient> GraphFleet<D> {
    /// 相位定相扇出(唯一入口;引擎 env 快照与各图 env 同步 ——
    /// GraphPlan 构造期只拿到 env 快照 ⇒ 逐相位显式下发,
    /// perf-framework.md §十.3)。
    pub(crate) fn set_phase(&mut self, p: owl_models::env::EvalPhase) {
        self.session.set_phase(p);
        if let Some(g) = self.verify.as_mut() {
            g.set_phase(p);
        }
        for g in self.fold.iter_mut() {
            g.set_phase(p);
        }
        for g in self.propose.iter_mut() {
            g.set_phase(p);
        }
        if let Some(g) = self.dflash.as_mut() {
            g.set_phase(p);
        }
    }
}
