//! 服务端平面 —— 家族 runtime(两平面之一;backends 视角)。
//!
//! 每家族一文件:KernelFamily 实现(assemble 构造即装配门 + run Result 全链)。
//! 依赖 = 本 crate 全量(contract/device/registry/family 底座)+
//! DeviceRes(由后端或测试实现)。对偶面 = [`crate::client`]。

/// cuBLAS GEMM runtime(feature=cublas,随底座)
#[cfg(feature = "cublas")]
pub mod cublas;
#[cfg(feature = "xqa")]
pub mod xqa;
/// FlashInfer prefill runtime(feature=flashinfer,随底座)
#[cfg(feature = "flashinfer")]
pub mod flashinfer;
/// GDN chunked runtime(零门)
pub mod gdn_chunked;
/// GDN scalar runtime(零门)
pub mod gdn_scalar;
/// 通信族 runtime(TP2 S2;零门)
pub mod comm;
/// Marlin runtime(feature=marlin,随底座)
#[cfg(feature = "marlin")]
pub mod marlin;

pub mod selfcheck;

use crate::registry::{FamilyFactory, KernelFamily};

/// 全家族装配表(名字→runtime 的唯一住址;M5 manifest 门挂各家族 assemble)。
/// 每项 = 族的构造函数指针(`KernelFamily::assemble`);boot 逐个调用。
pub fn family_factories() -> Vec<FamilyFactory> {
    vec![
        gdn_chunked::GdnChunkedRuntime::assemble,
        gdn_scalar::GdnScalarRuntime::assemble,
        comm::CommArStageRuntime::assemble,
        comm::CommArReduceRuntime::assemble,
        comm::CommAr1StageRuntime::assemble,
        comm::CommArFusedRuntime::assemble,
        #[cfg(feature = "cublas")]
        cublas::CublasRuntime::assemble,
        #[cfg(feature = "marlin")]
        marlin::MarlinRuntime::assemble,
        #[cfg(feature = "flashinfer")]
        flashinfer::FiRuntime::assemble,
        #[cfg(feature = "xqa")]
        xqa::XqaRuntime::assemble,
        #[cfg(feature = "xqa")]
        xqa::XqaDecodeRuntime::assemble,
        #[cfg(feature = "xqa")]
        xqa::XqaPrefillRuntime::assemble,
    ]
}
