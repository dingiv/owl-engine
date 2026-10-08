//! 服务端平面 —— 家族 runtime(两平面之一;backends 视角)。
//!
//! 每家族一文件:FamilyRuntime 实现(init 装配门 + run Result 全链)。
//! 依赖 = 本 crate 全量(contract/device/registry/family 底座)+
//! DeviceRes(由后端或测试实现)。对偶面 = [`crate::client`]。

/// cuBLAS GEMM runtime(feature=cublas,随底座)
#[cfg(feature = "cublas")]
pub mod cublas;
/// FlashInfer prefill runtime(feature=flashinfer,随底座)
#[cfg(feature = "flashinfer")]
pub mod flashinfer;
/// GDN chunked runtime(零门)
pub mod gdn_chunked;
/// GDN scalar runtime(零门)
pub mod gdn_scalar;
/// Marlin runtime(feature=marlin,随底座)
#[cfg(feature = "marlin")]
pub mod marlin;

use crate::registry::FamilyRuntime;

/// 全家族装配表(名字→runtime 的唯一住址;M5 manifest 门挂各家族 init)
pub fn all_families() -> Vec<Box<dyn FamilyRuntime>> {
    vec![
        Box::new(gdn_chunked::GdnChunkedRuntime::default()),
        Box::new(gdn_scalar::GdnScalarRuntime::default()),
        #[cfg(feature = "cublas")]
        Box::new(cublas::CublasRuntime::default()),
        #[cfg(feature = "marlin")]
        Box::new(marlin::MarlinRuntime),
        #[cfg(feature = "flashinfer")]
        Box::new(flashinfer::FiRuntime::default()),
    ]
}
