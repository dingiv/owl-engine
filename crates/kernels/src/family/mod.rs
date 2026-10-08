//! 家族底座 —— 资产/FFI/ABI(两平面共用的单源)。
//!
//! - [`gdn_chunked`]/[`gdn_scalar`]:AOT cubin 声明 + 发射常量 + ABI 注记
//! - [`cublas`]:OwlCublas 封装(handle/工作区/gemm)
//! - [`marlin`]:FFI + repack(装载域 weight packing)
//! - [`flashinfer`]:FFI + workspace 常量
//!
//! cublas/marlin/flashinfer 随各自 FFI feature 门;gdn 两家族零门。

/// cuBLAS 封装(feature=cublas;cudarc cublas 绑定)
#[cfg(feature = "cublas")]
pub mod cublas;
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
