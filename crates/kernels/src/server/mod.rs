//! 服务端平面 —— 家族 runtime(两平面之一;backends 视角)。
//!
//! 每家族一文件:KernelFamily 实现(assemble 构造即装配门 + run Result 全链)。
//! 依赖 = 本 crate 全量(contract/device/registry/family 底座)+
//! DeviceRes(由后端或测试实现)。对偶面 = [`crate::client`]。

/// cuBLAS GEMM runtime(feature=cublas,随底座)
#[cfg(feature = "cublas")]
pub mod cublas;#[cfg(feature = "xqa")]
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

/// **服务端面诊断旋钮**(进程级快照;后端 boot 期一次下发 —— 本 crate 零 env)。
///
/// 2026-10-21 收编:原 `OWL_GDN_DIAG/TOUCH/ZERO_SCRATCH/FRESH_SCRATCH`、
/// `FI_DEBUG`、`OWL_L3_AUDIT` 六类 env 全部退场 —— 唯一写点 = 后端
/// (`owl_cuda::GpuServer::with_diag_sink`)经本结构显式下发。
#[derive(Clone, Debug, Default)]
pub struct DiagTune {
    /// FlashInfer plan 期打印(原 `FI_DEBUG` / `OWL_FI_DEBUG`)
    pub fi_debug: bool,
    /// L3 事实面审计(原 `OWL_L3_AUDIT`)
    pub l3_audit: bool,
    /// GDN chunked 仪表(原 `OWL_GDN_*` 四键)
    pub gdn: gdn_chunked::GdnDiag,
}

/// 唯一诊断写入点(boot 期一次;重入不覆盖 —— 诊断档本就该在启动时定谳)
pub fn configure_diag(t: &DiagTune) {
    crate::spec::set_l3_audit(t.l3_audit);
    #[cfg(feature = "flashinfer")]
    flashinfer::set_debug(t.fi_debug);
    gdn_chunked::configure_diag(t.gdn.clone());
}
