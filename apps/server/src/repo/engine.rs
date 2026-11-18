//! **引擎端口（repo/engine）**：server 对 `EngineClient` 的唯一门面。
//!
//! 视图/服务层不直接依赖 `owl_engine` 的协议细节；换后端（如未来进程外
//! 引擎）只改本文件。当前是薄封装：`EngineClient` 本身已是 Send+Clone 的
//! 稳定句柄，门面只负责命名与边界。

use owl_engine::EngineClient;

/// 引擎仓库句柄（Clone 廉价；底层 `EngineClient` 自带引用计数）。
#[derive(Clone)]
pub(crate) struct EngineRepo {
    client: EngineClient,
}

impl EngineRepo {
    pub(crate) fn new(client: EngineClient) -> Self {
        Self { client }
    }

    /// 只读底层句柄（给仍需 `EngineClient` 具体类型的 service 调用点）。
    pub(crate) fn client(&self) -> &EngineClient {
        &self.client
    }
}
