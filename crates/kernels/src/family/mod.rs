//! 家族底座 —— 资产/FFI/ABI(两平面共用的单源)。
//!
//! - [`gdn_chunked`]/[`gdn_scalar`]:AOT cubin 声明 + 发射常量 + ABI 注记
//! - [`cublas`]:OwlCublas 封装(handle/工作区/gemm)
//! - [`marlin`]:FFI + repack(装载域 weight packing)
//! - [`flashinfer`]:FFI + workspace 常量
//! - [`xqa`]:XQA 契约 + FFI(Ampere spec-verify;2026-10-13)
//!
//! cublas/marlin/flashinfer 随各自 FFI feature 门;gdn 两家族零门。

/// cuBLAS 封装(feature=cublas;cudarc cublas 绑定)
#[cfg(feature = "cublas")]
pub mod cublas;
/// XQA 契约 + FFI(feature=xqa 时链接;契约面零门 —— 层侧 sig 查询要它)
pub mod xqa;
/// FlashInfer FFI + workspace 常量(feature=flashinfer)
#[cfg(feature = "flashinfer")]
pub mod flashinfer;
/// GDN chunked AOT cubin + 发射常量(零门)
pub mod gdn_chunked;
/// GDN scalar AOT cubin(零门)
pub mod gdn_scalar;
/// Marlin FFI + repack(feature=marlin)
#[cfg(feature = "marlin")]
pub mod marlin;
/// **构建期 cubin 资产登记表**(build.rs 生成;源文本哈希 → fatbin 字节)。
/// 门 = `device` —— 只有执行面(backends/cuda)按源查资产;models 侧零成本。
#[cfg(feature = "device")]
pub mod cubins;
