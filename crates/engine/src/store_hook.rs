//! Store 钩子契约：**Acquire / Release / Maintain** 三个语义钩子。
//!
//! 本模块只放**请求与回执类型**；执行臂是
//! [`RunningEngine::store_hook`](crate::running::RunningEngine::store_hook)
//! （在 `running.rs`，因为要访问设备/账房的私有方法）。
//!
//! # 当前迁移阶段（2026-10-25，Phase 1）
//!
//! 已收口：
//! - `Acquire::Capacity`：准入/增长（原 `Node::Need`）；
//! - `Acquire::Load`：前缀命中的 host→device 回灌（原 H4 双旁路）；
//! - `Release::{TurnEnd,Abort}`：N4 清点/未归还/force release/trim；
//! - `Maintain::{Idle,Pressure}`：空转与压力整理（原 `Node::Idle/Pressure`）。
//!
//! 仍待收编（Phase 2/3，见 `docs/devlog/memory/store-hook-contract.md`）：
//! - `scheduler` / `prefill` / `draft` / MTP 的 `ensure_for_len` 槽兜底；
//! - H3 的链登记与 H6 的引用归零；
//! - spec 的 `snap_onload` / `snap_offload` 状态腿直通。
//!
//! # 纪律
//!
//! engine 其它位置不得直接调 `plan_need` / `plan_onload` / `snap_*`；
//! 只能经 `store_hook`。`Acquire::Capacity` 的 kind 由 `Need.why` 表达
//! （`Admit` / `Grow`），不另设一个可能和 `Need` 漂移的冗余枚举。

use owl_store::{Applied, ChainHash, Need};

/// 顶层三钩子。
#[derive(Clone, Debug)]
pub(crate) enum StoreHook {
    /// 前场：需要设备资源或回灌。
    Acquire(Acquire),
    /// 生命周期：turn 结束 / 弃置，资产从 Active 转 Cache/Free。
    Release(Release),
    /// 后场：空转/压力下的分层维护，给的是目标量而不是缺口。
    Maintain(Maintain),
}

/// 前场需求 + 回灌。
#[derive(Clone, Debug)]
pub(crate) enum Acquire {
    /// 准入 / 增长：`need.why` 必须是 `Admit` / `Grow`。
    Capacity(Need),
    /// 前缀命中回灌：先确保设备槽，再统一执行 host→device onload。
    Load(LoadRequest),
}

/// 一次前缀回灌请求（原 `BeginPlan.pending_loads` 的执行面）。
#[derive(Clone, Debug)]
pub(crate) struct LoadRequest {
    /// 会话 id：回灌后把新槽写回它的 `block_table`。
    pub(crate) session: u64,
    /// `(块表下标, 内容身份)`；顺序与 `BeginPlan.pending_loads` 一致。
    pub(crate) blocks: Vec<(usize, ChainHash)>,
}

/// 生命周期钩子。
///
/// Phase 1 只承接原 `Node::TurnEnd` 的 N4 收尾；H3/H6 的链登记与引用归零
/// 仍在 `turn_end.rs`，Phase 3 再并入这里（届时增加链资产载荷）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Release {
    /// 正常 turn 收尾。
    TurnEnd(u64),
    /// 弃置 / 取消（不发布链，但同样要做 N4 清点与兜底释放）。
    Abort(u64),
}

/// 后场维护钩子。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Maintain {
    /// 空转整理（冷面下沉/清池）；状态腿目标由钩子按宿主余位补。
    Idle(usize),
    /// 压力整理（N6）；`snap_bytes` 显式给出（0 = 不动状态腿）。
    Pressure {
        /// 目标腾出的 KV 槽数（不是缺口）。
        slots: usize,
        /// 目标腾出的状态腿设备字节。
        snap_bytes: usize,
    },
}

/// 钩子回执：调用方只消费“实际发生了什么”，不反推计划。
#[derive(Clone, Debug, Default)]
pub(crate) struct StoreOutcome {
    /// 容量执行臂的实际搬运/驱逐计数。
    pub(crate) applied: Applied,
    /// 本次 `Load` 实际回灌的块数（其它钩子为 0）。
    pub(crate) loaded_blocks: usize,
}

impl StoreHook {
    pub(crate) fn capacity(need: Need) -> Self {
        Self::Acquire(Acquire::Capacity(need))
    }

    pub(crate) fn load(session: u64, blocks: Vec<(usize, ChainHash)>) -> Self {
        Self::Acquire(Acquire::Load(LoadRequest { session, blocks }))
    }

    pub(crate) fn turn_end(turn: u64) -> Self {
        Self::Release(Release::TurnEnd(turn))
    }

    pub(crate) fn abort(turn: u64) -> Self {
        Self::Release(Release::Abort(turn))
    }

    pub(crate) fn idle(slots: usize) -> Self {
        Self::Maintain(Maintain::Idle(slots))
    }

    pub(crate) fn pressure(slots: usize) -> Self {
        Self::Maintain(Maintain::Pressure { slots, snap_bytes: 0 })
    }

    /// 指标标签（`store.hook.<label>`）。
    ///
    /// `Capacity` 的非法 `why` 由执行臂响亮拒；这里给 `invalid` 只是让
    /// 计数先落账，不把错误吞成正常标签。
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Self::Acquire(Acquire::Capacity(need)) => match need.why {
                owl_store::NeedWhy::Admit => "acquire.admit",
                owl_store::NeedWhy::Grow => "acquire.grow",
                _ => "acquire.invalid",
            },
            Self::Acquire(Acquire::Load(_)) => "acquire.load",
            Self::Release(Release::TurnEnd(_)) => "release.turn_end",
            Self::Release(Release::Abort(_)) => "release.abort",
            Self::Maintain(Maintain::Idle(_)) => "maintain.idle",
            Self::Maintain(Maintain::Pressure { .. }) => "maintain.pressure",
        }
    }
}
