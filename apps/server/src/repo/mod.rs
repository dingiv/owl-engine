//! **持久层（repo）**：进程内状态与后端端口的唯一入口。
//!
//! - `engine`：`EngineClient` 的门面（turn 提交/取消/tokenize/诊断/队列）；
//! - `trace`：`/debug/agent` 的环形诊断账（进程内持久）；
//! - `vram`：互斥 VRAM 账（`/debug/vram` 的账本投影）。
pub(crate) mod engine;
pub(crate) mod trace;
pub(crate) mod vram;
