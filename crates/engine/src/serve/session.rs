//! 会话账/块账房已下沉 `owl_store::session`（2026-11-15 用户裁定）。
//!
//! **为什么**：KV/快照/准入的容量决策需要直接看到会话账（`tokens`、块链、
//! 快照资产），会话模块必须与 `Rel`/`BlockLeg`/`StateStore` 同屋。
//!
//! 本模块只保留 engine 侧路径兼容（`crate::session::*` /
//! `crate::serve::session::*`）；新代码请直接用 `owl_store::session`。
pub use owl_store::session::*;
