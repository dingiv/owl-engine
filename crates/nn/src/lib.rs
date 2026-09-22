//! owl-nn —— 最小张量层 + NN 算子(一期)。
//!
//! 职责边界(roadmap.local.md 裁决 2/3/5):
//! - **分配与使用两阶段分离**:`Tensor` 只能经 Device 池分配获得;
//!   算子层(ops)只接收 [`KernelCtx`]——它没有分配能力,类型上保证
//!   E 阶段(使用)不可能发生分配;
//! - **out-style**:所有算子显式写出到既有缓冲,owl 层不做隐式分配;
//! - **port 纪律**:kernel 移植自 candle-kernels 的 .cu 源文件,
//!   每处标注 `// ported from candle-kernels/src/xxx.cu`。

pub mod cublas;
pub mod kernels;
pub mod ops;
pub mod tensor;
pub use tensor::TensorPoolOps;

use owl_iface::{BufToken, MemPhase};
use std::collections::BTreeSet;
use std::sync::{Arc, Mutex};

/// 哨兵①产物:一次捕获的依赖痕迹(launch 序列 + 触碰的全部缓冲令牌)。
/// replay 前可据此做世代校验(死亡令牌 = 结构化报错,而非 Xid 盲死)。
#[derive(Debug, Default, Clone)]
pub struct CaptureRecord {
    /// 捕获期 kernel 发射序列(顺序即图内节点序)
    pub launches: Vec<&'static str>,
    /// 触碰的全部缓冲令牌(图依赖集合;封闭性由裁决 5 保证)
    pub touched: BTreeSet<BufToken>,
}

/// 捕获记录器:挂进 [`KernelCtx`] 后,launch/缓冲触碰被自动记录。
/// Clone 共享同一记录(进入闭包/跨函数)。
#[derive(Debug, Clone)]
pub struct CaptureRecorder {
    record: Arc<Mutex<CaptureRecord>>,
}

impl CaptureRecorder {
    pub fn new() -> Self {
        Self {
            record: Arc::new(Mutex::new(CaptureRecord::default())),
        }
    }

    fn trace_launch(&self, kernel: &'static str) {
        self.record.lock().expect("CaptureRecord 中毒").launches.push(kernel);
    }

    fn trace_buf(&self, t: BufToken) {
        self.record.lock().expect("CaptureRecord 中毒").touched.insert(t);
    }

    /// 取当前痕迹快照
    pub fn snapshot(&self) -> CaptureRecord {
        self.record.lock().expect("CaptureRecord 中毒").clone()
    }
}

impl Default for CaptureRecorder {
    fn default() -> Self {
        Self::new()
    }
}

/// E 阶段的唯一上下文:只有 launch 能力(**流 + 发射**,**没有分配能力**)。
/// 算子函数签名只允许接收它(裁决 5 的类型强制)。
/// 捕获场景:带 [`CaptureRecorder`] —— launch/触碰自动留痕(哨兵①)。
///
/// M1②:**流由 ctx 携带,不再由 OpsCtx 持有**——eager 持设备主流,
/// 捕获持会话捕获流;算子发射到哪条流在构造 ctx 时即定,无运行时状态。
#[derive(Clone)]
pub struct KernelCtx {
    pub(crate) phase: MemPhase,
    pub(crate) stream: Arc<owl_cuda::ffi::CudaStream>,
    pub(crate) recorder: Option<CaptureRecorder>,
    /// Signal 追踪作用域(capturing 态持有;drop 时弹栈——持有即语义)
    #[allow(dead_code)]
    pub(crate) track: Option<owl_signal::TrackingHandle>,
}

impl KernelCtx {
    /// eager 上下文(无痕迹记录;stream = 发射目标)
    pub fn eager(phase: MemPhase, stream: Arc<owl_cuda::ffi::CudaStream>) -> Self {
        Self {
            phase,
            stream,
            recorder: None,
            track: None,
        }
    }

    /// 捕获上下文(带记录器;记录器即追踪 sink)
    pub fn capturing(
        recorder: CaptureRecorder,
        stream: Arc<owl_cuda::ffi::CudaStream>,
    ) -> Self {
        // Signal 接线:recorder 即追踪 sink——ops 的 owl_signal::emit
        // 自动落到这里(依赖追踪自动化,roadmap §四·五)
        let rec = recorder.clone();
        let sink: owl_signal::Sink =
            Arc::new(move |t: owl_signal::Token| rec.trace_buf(t));
        Self {
            phase: MemPhase::Capturing,
            stream,
            recorder: Some(recorder),
            track: Some(owl_signal::enter(sink)),
        }
    }

    /// 绑会话帧的捕获上下文:launch 留痕(recorder)+ 自动租约
    /// (frame.lease_sink)共用一条 signal 通道(signal 设计 §2.1)。
    pub fn capturing_leased(
        recorder: CaptureRecorder,
        frame: &owl_cuda::CaptureFrame<'_>,
    ) -> Self {
        let rec = recorder.clone();
        let lease = frame.lease_sink.clone();
        let sink: owl_signal::Sink = Arc::new(move |t: owl_signal::Token| {
            rec.trace_buf(t);
            (lease)(t);
        });
        Self {
            phase: MemPhase::Capturing,
            stream: Arc::clone(frame.stream),
            recorder: Some(recorder),
            track: Some(owl_signal::enter(sink)),
        }
    }

    pub fn phase(&self) -> MemPhase {
        self.phase
    }

    /// 发射目标流(eager = 设备主流;捕获 = 会话捕获流)
    pub fn stream(&self) -> &Arc<owl_cuda::ffi::CudaStream> {
        &self.stream
    }

    /// 是否处于记录态(捕获中)
    pub fn recording(&self) -> bool {
        self.recorder.is_some()
    }

    /// 记录一次 kernel 发射(非记录态为 no-op)
    pub fn trace_launch(&self, kernel: &'static str) {
        if let Some(r) = &self.recorder {
            r.trace_launch(kernel);
        }
    }

    /// 记录一次缓冲触碰(非记录态为 no-op)
    pub fn trace_buf(&self, t: BufToken) {
        if let Some(r) = &self.recorder {
            r.trace_buf(t);
        }
    }

    /// 痕迹快照(非记录态 = None)
    pub fn snapshot(&self) -> Option<CaptureRecord> {
        self.recorder.as_ref().map(|r| r.snapshot())
    }
}

/// 捕获安全标记(A1.5/裁决 3③):实现者声明该算子可进入捕获段
/// (无同步、无 D2H、无 host 分支、无隐式分配)。
pub trait CaptureSafe {
    const CAPTURE_SAFE: bool;
}

/// EagerOnly 算子(如 to_vec 回读)的标记;禁入捕获段。
pub trait EagerOnly {
    const EAGER_ONLY: bool;
}
