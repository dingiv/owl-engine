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
///
/// `sink` = **输出口**(`klog_*!` 与 C++ `OWL_LOG_*` 共用):它承载运行期
/// 可见性(level + tag 白黑名单),故 `fi_debug` 这类"纯打印开关"不再需要
/// 配置面旗标(2026-10-21 裁决:**bool 管采集,日志管可见性**)。
#[derive(Clone, Default)]
pub struct DiagTune {
    /// 下层输出口(未接线 = 全部静默;接线后受 logger 分级/tag 过滤)
    pub sink: crate::sink::Sink,
    /// L3 事实面审计(**采集**:声明 rank=0 的槽实际阶数的登记)
    pub l3_audit: bool,
    /// GDN chunked 仪表(**采集**:逐 call 额外的 dtoh 回读/往返/清零)
    pub gdn: gdn_chunked::GdnDiag,
}

impl std::fmt::Debug for DiagTune {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // LogSink = fn 指针包,Debug 无信息量;只标接线与否
        f.debug_struct("DiagTune")
            .field("sink", &if self.sink.is_none() { "none" } else { "wired" })
            .field("l3_audit", &self.l3_audit)
            .field("gdn", &self.gdn)
            .finish()
    }
}

/// 诊断下发失败(boot 后冻结:二次不同值 = 拒,不静默覆盖)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DiagError {
    /// 输出口已配置(sink 值不同)
    Sink(crate::sink::SinkError),
    /// GDN 仪表档已冻结(值不同)
    Gdn(gdn_chunked::DiagError),
}

impl std::fmt::Display for DiagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sink(e) => write!(f, "{e}"),
            Self::Gdn(e) => write!(f, "{e}"),
        }
    }
}
impl std::error::Error for DiagError {}

/// 唯一诊断写入点(boot 期一次;重入不同值 = [`DiagError`],不再静默覆盖)
pub fn configure_diag(t: &DiagTune) -> Result<(), DiagError> {
    // 输出口先接:后面两句自己的告警才能落地
    crate::sink::configure(t.sink).map_err(DiagError::Sink)?;
    crate::spec::set_l3_audit(t.l3_audit);
    gdn_chunked::configure_diag(t.gdn.clone()).map_err(DiagError::Gdn)
}
