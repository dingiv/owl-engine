//! **调度域**(S0 决策半):决策词汇 + 政策 + 选择层 + 保洁账。
//!
//! 分工:`scheduler` 产出 `SchedulerOutput`(词汇 + 决策逻辑);`sched_policy`
//! 是两个互不耦合的状态机(准入 / 相位,纯数据 + 零 I/O,事实由引擎每拍注入);
//! `select` 是 B 轴(视图 / 覆盖率探针 / 驻留调和账);`housekeeping` 收节拍与计数。
//! **执行**在 `exec/` —— 本域无 I/O、不发命令。

pub(super) mod housekeeping;
#[allow(dead_code)] // 准入/相位状态机:接线中(见模块头)
pub(super) mod sched_policy;
pub(super) mod scheduler;
pub(super) mod select;
