//! Store 钩子契约：**Acquire / Release / Maintain** 三个语义钩子。
//!
//! 本模块只放**请求与回执类型**；执行臂是
//! [`RunningEngine::store_hook`](crate::running::RunningEngine::store_hook)
//! （在 `running.rs`，因为要访问设备/账房的私有方法）。
//!
//! # 当前迁移阶段（2026-10-25，Phase 1–4 全部完成）
//!
//! 已收口：
//! - `Acquire::Capacity`：准入/增长（原 `Node::Need`）；
//! - `Acquire::Load`：前缀命中的 host→device 回灌（原 H4 双旁路）；
//! - `Acquire::State`：快照选槽/就位（原 H7 的 `snap_ensure_slot` /
//!   `snap_ensure_resident` 直呼）；
//! - `Release::{TurnEnd,Abort}`：H3 链登记/盘索引 + H6 表释放 +
//!   GDN 行归还 + 归属下行 + N4 清点/force release/trim；
//! - `Maintain::{Idle,Pressure}`：空转与压力整理（原 `Node::Idle/Pressure`）；
//! - **Phase 2**：`ensure_for_len` 改为纯分配；生成相位的 `Acquire(Grow)`
//!   按 `step_need::generation_target` 覆盖 verify + 轮末 propose
//!   （DFlash2 `pos+2d+2` / MTP `pos+2d`）。MTP 腿 cache=Off/无宿主，
//!   本来就不产生同步驱逐。
//! - **Phase 4**：`FixedAfter` 补 spec/replay 快照；删除 pump N6，
//!   `Maintain::Pressure` 只在 `PoolExhausted` 后按池自由槽收缩；
//!   metrics label 已钉住。
//!
//! `FIXME(store-hook)` 已清零；coverage 的 `plan_*` 是显式 diag-only。
//!
//! # 纪律
//!
//! engine 生产位置不得直接调 `plan_need` / `plan_onload` / `snap_*`；
//! 只能经 `store_hook`。`Acquire::Capacity` 的 kind 由 `Need.why` 表达
//! （`Admit` / `Grow`），不另设一个可能和 `Need` 漂移的冗余枚举。

use owl_store::{Applied, ChainHash, Need, SnapKey};

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
    /// 状态腿：选槽或就位。
    State(StateAcquire),
}

/// 状态腿前场请求。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StateAcquire {
    /// 为一次拍摄/回灌选设备槽（必要时下沉/丢弃最冷快照）。
    Slot {
        /// 快照内容身份。
        key: SnapKey,
    },
    /// 把快照就位到设备（Gpu 命中 / Host/Disk 回灌）。
    Resident {
        /// 快照内容身份。
        key: SnapKey,
    },
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
/// Phase 3 起承接 H3（链登记/盘索引）、H6（表/引用归零）、GDN 行归还、
/// 归属下行与 N4（清点/未归还/force release/trim）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Release {
    pub(crate) kind: ReleaseKind,
    pub(crate) turn: u64,
    pub(crate) session: u64,
    pub(crate) gdn_slot: usize,
    pub(crate) ephemeral: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReleaseKind {
    /// 正常 turn 收尾：发布链 + 写盘索引。
    TurnEnd,
    /// 弃置 / 取消：不发布链，但同样要做 H6/N4。
    Abort,
}

/// 后场维护钩子。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Maintain {
    /// 空转整理（冷面下沉/清池）；状态腿目标由钩子按宿主余位补。
    Idle(usize),
    /// 紧急池压力（`PoolExhausted` 后的收缩）；`slots` 是目标腾出量，
    /// `snap_bytes` 显式给出（0 = 不动状态腿）。
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
    /// 本次 `State` 就位后的设备槽（其它钩子为 None）。
    pub(crate) state_slot: Option<u32>,
}

impl StoreHook {
    pub(crate) fn capacity(need: Need) -> Self {
        Self::Acquire(Acquire::Capacity(need))
    }

    pub(crate) fn load(session: u64, blocks: Vec<(usize, ChainHash)>) -> Self {
        Self::Acquire(Acquire::Load(LoadRequest { session, blocks }))
    }

    pub(crate) fn state(key: SnapKey) -> Self {
        Self::Acquire(Acquire::State(StateAcquire::Resident { key }))
    }

    pub(crate) fn state_slot(key: SnapKey) -> Self {
        Self::Acquire(Acquire::State(StateAcquire::Slot { key }))
    }

    pub(crate) fn turn_end(
        turn: u64,
        session: u64,
        gdn_slot: usize,
        ephemeral: bool,
    ) -> Self {
        Self::Release(Release {
            kind: ReleaseKind::TurnEnd,
            turn,
            session,
            gdn_slot,
            ephemeral,
        })
    }

    pub(crate) fn abort(
        turn: u64,
        session: u64,
        gdn_slot: usize,
        ephemeral: bool,
    ) -> Self {
        Self::Release(Release {
            kind: ReleaseKind::Abort,
            turn,
            session,
            gdn_slot,
            ephemeral,
        })
    }

    pub(crate) fn idle(slots: usize) -> Self {
        Self::Maintain(Maintain::Idle(slots))
    }

    pub(crate) fn pressure_with_snap(slots: usize, snap_bytes: usize) -> Self {
        Self::Maintain(Maintain::Pressure { slots, snap_bytes })
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
            Self::Acquire(Acquire::State(StateAcquire::Slot { .. })) => "acquire.state.slot",
            Self::Acquire(Acquire::State(StateAcquire::Resident { .. })) => "acquire.state.resident",
            Self::Release(Release { kind: ReleaseKind::TurnEnd, .. }) => "release.turn_end",
            Self::Release(Release { kind: ReleaseKind::Abort, .. }) => "release.abort",
            Self::Maintain(Maintain::Idle(_)) => "maintain.idle",
            Self::Maintain(Maintain::Pressure { .. }) => "maintain.pressure",
        }
    }
}

#[cfg(test)]
mod label_tests {
    use super::*;
    use owl_store::NeedWhy;

    /// 指标名是外部观测契约；改名 = 看板断链，故用纯单测钉住。
    #[test]
    fn labels_are_stable() {
        assert_eq!(
            StoreHook::capacity(Need::kv(1, NeedWhy::Admit)).label(),
            "acquire.admit"
        );
        assert_eq!(
            StoreHook::capacity(Need::kv(1, NeedWhy::Grow)).label(),
            "acquire.grow"
        );
        assert_eq!(StoreHook::load(1, vec![]).label(), "acquire.load");
        let key = SnapKey::at_boundary(ChainHash::new(0xA1));
        assert_eq!(StoreHook::state(key).label(), "acquire.state.resident");
        assert_eq!(
            StoreHook::state_slot(key).label(),
            "acquire.state.slot"
        );
        assert_eq!(StoreHook::turn_end(1, 2, 0, false).label(), "release.turn_end");
        assert_eq!(StoreHook::abort(1, 2, 0, true).label(), "release.abort");
        assert_eq!(StoreHook::idle(1).label(), "maintain.idle");
        assert_eq!(
            StoreHook::pressure_with_snap(1, 2).label(),
            "maintain.pressure"
        );
    }
}
