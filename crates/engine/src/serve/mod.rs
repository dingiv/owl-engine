//! **请求面 / 会话域**:客户端 Session(应用层词汇)→ Turn → Step 三层,
//! 以及"消息边界 → 页对齐快照点"(前缀复用的分叉点)。
//!
//! 词义权威在本域(`session` / `turn`);turn 生命周期账与 pump 主循环在
//! `crate::running`。面向上层的协议(HTTP/OpenAI)在 `apps/server`。

pub(super) mod msg_boundary;
pub mod session;
pub mod turn;
