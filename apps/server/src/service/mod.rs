//! **服务层（service）**：请求级业务编排（chat / tokenize / load / 诊断投影）。
//!
//! 输入是协议 DTO 或已渲染的 prompt，输出是事件流/应答；不解析 HTTP 帧，
//! 也不直接持有套接字（流式写回由 view 层消费）。
pub(crate) mod chat;
pub(crate) mod load;
