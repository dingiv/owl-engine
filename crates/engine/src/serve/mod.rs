//! **请求面 / 会话域**:客户端 Session(应用层词汇)→ Turn → Step 三层,
//! 以及"消息边界 → 页对齐快照点"(前缀复用的分叉点)。
//!
//! **`session` 已于 2026-11-15 下沉 `owl_store::session`**（KV/快照/准入
//! 容量决策需要直接看到会话账）；本域保留路径兼容。`turn` 词义权威仍在本域；
//! turn 生命周期账与 pump 主循环在 `crate::running`。面向上层的协议
//! (HTTP/OpenAI)在 `apps/server`。

pub(super) mod msg_boundary;
pub mod session;
pub mod turn;
