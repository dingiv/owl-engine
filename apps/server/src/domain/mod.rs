//! **领域层（domain）**：OpenAI 协议 DTO、chat 模板、流式三通道解析、视觉预处理。
//!
//! 纯数据/纯函数为主；不依赖 view（套接字）与 repo（引擎句柄）。
pub(crate) mod chat_tpl;
pub(crate) mod openai;
pub(crate) mod repeat_guard;
pub(crate) mod toolcall;
pub(crate) mod vision;
