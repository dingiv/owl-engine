//! owl-engine —— xinfer 换头移植的顶层 crate(2026-09-22 立项)。
//!
//! 模块树镜像 `packages/xinfer/crates/core/src`(只读参照源,不依赖):
//! 上层业务逻辑(scheduler/runner/models/speculative/server)收编于此,
//! 算子层 API 统一翻译为 owl-nn/owl-cuda 契约(分配入口在 Pool,
//! 捕获统一走 owl CaptureSession;禁止 candle/cudarc 依赖)。
//!
//! 总设计:`docs/arch/xinfer-transplant.md`。
//! 验收口径(用户裁决):**先编译,后运行** —— 各搬运阶段以
//! `cargo build --workspace` 全绿为准,运行指标另立里程碑。

pub mod block_manager;
pub mod config;
pub mod dim;
pub mod hybrid;
pub mod downloader;
pub mod distributed;
pub mod env;
pub mod error;
pub mod image;
pub mod graphplan;
pub mod kvcache;
pub mod loader;
pub mod models;
pub mod multi_node;
pub mod prefix_cache;
pub mod runner;
pub mod sampler;
pub mod scheduler;
pub mod session;
pub mod sequence;
pub mod server;
pub mod speculative;
pub mod transfer;
pub mod utils;

pub use error::{Error, Result};

/// 日志宏(自 xinfer `core/mod.rs` 搬运;engine 无 python 特性,
/// 走 tracing 分支;python 分支随 T4 服务层再议)。
#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        tracing::info!($($arg)*)
    };
}

#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => {
        tracing::warn!($($arg)*)
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        tracing::error!($($arg)*)
    };
}
