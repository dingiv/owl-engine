//! 客户端平面 —— 类型化调用框架(models 视角;两平面之一)。
//!
//! 每家族一文件:face = 名字常量/槽序单源/builder 校验/线格式。
//! models 只 `use` 本平面,不拼名字字符串、不摆参数槽。
//! 对偶面 = [`crate::server`](家族 runtime;DeviceRes 由后端实现)。

/// native 语义算子胖算子(样例:NarrowStrided;值域校验内嵌)
pub mod native;
/// GDN chunked(face;零门)
pub mod gdn_chunked;
/// GDN scalar(face;零门)
pub mod gdn_scalar;
/// cuBLAS GEMM face(feature=cublas,随底座)
#[cfg(feature = "cublas")]
pub mod cublas;
/// Marlin face(feature=marlin,随底座)
#[cfg(feature = "marlin")]
pub mod marlin;
/// FlashInfer face(feature=flashinfer,随底座)
#[cfg(feature = "flashinfer")]
pub mod flashinfer;
