//! GPU server:事件循环式 actor(设备执行权唯一宿主)。
//!
//! 实现 owl-models 的 `DeviceClient` 能力契约(五原语)。
//!
//! 模块地图:
//! - [`command`]:命令信封(封闭枚举)+ Ack/Waiter 回执桥(公开;手工组装用)
//! - [`state`]:GpuCtx(三固定流/池块账房/图捕获状态机)+ KernelCache + Staging
//! - [`launch`]:LaunchMsg 装配发射(哑执行器;零算子语义)
//! - [`server`]:GpuServer 事件循环(rx 阻塞监听 + issue 相)+ 完成派发线程
//! - [`client`]:GpuClient 异步门面(DeviceClient 实现)
//!
//! 组装形态(构造/管道/启动三者分离):
//! ```text
//! let (tx, rx) = mpsc::channel::<Command>();   // 管道:外部创建
//! let server = GpuServer::new(rx, ordinal, None);  // server 构造(纯构造,零 CUDA)
//! let client = GpuClient::new(tx);                 // client 构造(纯句柄)
//! thread::spawn(move || server.run());             // 启动(专用线程)
//! ```
//! 便捷路径:`GpuClient::spawn(ordinal)` 一键完成上述组装 + boot 握手。
//!
//! 异步模型:GPU API 全走非阻塞形态(launch/memset/异步 memcpy 提交进流即返回);
//! 完成通知用现代回调 API `cuLaunchHostFunc`(host 回调随流序由**驱动主动触发**,
//! 零轮询零阻塞收割)—— 回调跑在驱动线程,只做 channel 投递;真正的 finish
//! (含 free_host 等 CUDA 操作)由 server 内部的派发线程执行,结果经 Ack
//! 回执客户端,以此构成对外的异步 API(命令流水线化;server 不因 GPU 工作而停摆)。

mod server;
mod client;
pub mod command;
mod launch;
mod state;

pub use client::GpuClient;
pub use command::Command;
pub use server::GpuServer;
