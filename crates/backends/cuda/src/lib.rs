//! owl-cuda —— GPU server 后端:实现 owl-models 的 `DeviceClient` 契约。
//!
//! 分层:owl-kernels(kernel 描述)→ owl-cuda(actor 执行)→ owl-models(契约)。
//! server 是**哑执行器**:不认识具体算子,只认 LaunchMsg / Alloc / Htod / Dtoh / Sync。
//!
//! 旧世界(device/pool/governor/graph/buffers)已归档至 `cuda_bak/`,
//! 待 nn/engine 迁移至声明式 API 后再按需重建。

pub mod ffi;
pub mod gpu_server;

/// 测试/示例的设备序号(OWL_TEST_DEVICE,默认 0)
pub fn test_device_ordinal() -> usize {
    std::env::var("OWL_TEST_DEVICE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

/// 测试/示例默认池容量
pub const TEST_POOL_BYTES: u64 = 64 << 20;

pub use gpu_server::GpuClient;
pub use owl_models::client::DeviceClient;
