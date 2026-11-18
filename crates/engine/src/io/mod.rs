//! **engine I/O 执行面**（2026-11-18 S1）。
//!
//! 边界（用户裁决 / `engine-models-store-boundary-handoff-2026-11-18.md`）：
//!
//! ```text
//! owl-store   plan_*  -> 纯 host 的 I/O **描述**（片/槽台/路径，不含 Bytes）
//! owl-engine  io::*   -> 真正执行 DMA / 文件打开读写
//! owl-store   commit_* -> I/O 成功后只落账
//! ```
//!
//! 本模块只放**执行原语**；一次 turn / 一次溢写的编排仍在
//! `running.rs`、`mem/disk_runtime.rs`，它们负责在正确的段边界调用这里。
//! 目录纪律（`AGENTS.md`）：I/O 统一走顶层自由函数，不在状态结构的方法里藏 I/O。

pub(crate) mod moves;
