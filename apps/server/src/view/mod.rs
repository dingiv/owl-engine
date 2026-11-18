//! **视图层（view）**：HTTP/1.1 帧、连接生命周期、路由与响应写回。
//!
//! 只做协议适配：把 `http::Request` 变成 service 调用，再把结果写成 JSON/SSE。
//! 不直接碰引擎/存储实现（经 `repo` 门面）。
pub(crate) mod debug;
pub(crate) mod http;
pub(crate) mod router;
