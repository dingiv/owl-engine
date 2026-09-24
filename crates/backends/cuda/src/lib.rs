//! owl-cuda —— GPU server 后端:实现 owl-models 的 `DeviceClient` 契约。
//!
//! 分层:owl-kernels(kernel 描述)→ owl-cuda(actor 执行)→ owl-models(契约)。
//! server 是**哑执行器**:不认识具体算子,只认 LaunchMsg / alloc / htod / dtoh / sync。
//!
//! 模块地图:
//! - [`ffi`]:cudarc 受控再导出(A4;上层唯一可见的 driver 表面)
//! - [`gpu_server`]:actor 线程 + 池块账房 + 懒编译 + LaunchMsg 发射
//!
//! 历史档案(`.rsx` 后缀 = Tx 时代伪代码骨架,不编译,仅供追溯):
//! `client.rsx` / `protocol.rsx` / `server.rsx` / `server-state-machine.rsx`。
//! 旧世界(device/pool/governor/graph/buffers)归档于 `../cuda_bak/`。

pub mod ffi;
pub mod gpu_server;

pub use gpu_server::GpuClient;
pub use owl_models::client::DeviceClient;
