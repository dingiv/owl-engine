//! owl-cuda —— GPU server 后端:实现 owl-models 的 `DeviceClient` 契约。
//!
//! 分层:owl-kernels(kernel 描述)→ owl-cuda(actor 执行)→ owl-models(契约)。
//! server 是**哑执行器**:不认识具体算子,只认 LaunchMsg / alloc / htod / dtoh / sync。
//!
//! 模块地图:
//! - [`ffi`]:cudarc 受控再导出(A4;上层唯一可见的 driver 表面)
//! - [`command`]:命令信封(封闭枚举)+ Ack/Waiter 回执桥(公开;手工组装用)
//! - [`state`]:GpuCtx(三固定流/池块账房/图捕获状态机)+ KernelCache + Staging
//! - [`launch`]:LaunchMsg 装配发射(哑执行器;零算子语义)
//! - [`server`]:GpuServer 事件循环(rx 阻塞监听 + issue 相)+ 完成派发线程
//! - [`gpu_client`]:GpuClient 异步门面(DeviceClient 实现)
//!
//! 组装形态(构造/管道/启动三者分离):
//! ```text
//! let (tx, rx) = mpsc::channel::<Command>();       // 管道:外部创建
//! let server = GpuServer::new(rx, selector, None); // server 构造(纯构造,零 CUDA)
//! let client = GpuClient::new(tx);                 // client 构造(纯句柄)
//! thread::spawn(move || server.run());             // 启动(专用线程)
//! ```
//!
//! 异步模型:GPU API 全走非阻塞形态(launch/memset/异步 memcpy 提交进流即返回);
//! 完成通知用现代回调 API `cuLaunchHostFunc`(host 回调随流序由**驱动主动触发**,
//! 零轮询零阻塞收割)—— 回调跑在驱动线程,只做 channel 投递;真正的 finish
//! (含 free_host 等 CUDA 操作)由 server 内部的派发线程执行,结果经 Ack
//! 回执客户端,以此构成对外的异步 API(命令流水线化;server 不因 GPU 工作而停摆)。

mod command;
mod gpu_client;
mod launch;
mod server;
mod state;

pub mod ffi;

pub use command::Command;
pub use gpu_client::GpuClient;
pub use server::GpuServer;
pub use state::DeviceSelector;

// 线格式与词汇再导出(客户只依赖 owl-cuda 即可组装命令,不必直连 owl-models;
// FIXME 升级:将来把契约类型迁出自 owl-models,彻底解除反向依赖)
pub use owl_models::client::{Arg, Bytes, GraphId, KernelSpec, LaunchMsg};
pub use owl_models::shape::Shape;
pub use owl_models::{Dtype, ModelError};

/// DeviceClient 能力契约(实现于 GpuClient;从 owl-models 契约层 re-export)
pub use owl_models::client::DeviceClient;

/// 测试/示例的设备序号(OWL_TEST_DEVICE,默认 0)
pub fn test_device_ordinal() -> usize {
    std::env::var("OWL_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}
