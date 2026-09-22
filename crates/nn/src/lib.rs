//! owl-nn —— 最小张量层 + NN 算子(一期)。
//!
//! 职责边界(roadmap.local.md 裁决 2/3/5):
//! - **分配与使用两阶段分离**:`OwlTensor` 只能经 Device 池分配获得;
//!   算子层(ops)只接收 [`KernelCtx`]——它没有分配能力,类型上保证
//!   E 阶段(使用)不可能发生分配;
//! - **out-style**:所有算子显式写出到既有缓冲,owl 层不做隐式分配;
//! - **port 纪律**:kernel 移植自 candle-kernels 的 .cu 源文件,
//!   每处标注 `// ported from candle-kernels/src/xxx.cu`。

pub mod cublas;
pub mod kernels;
pub mod ops;
pub mod tensor;

use owl_iface::MemPhase;

/// E 阶段的唯一上下文:只有 launch 能力,**没有分配能力**。
/// 算子函数签名只允许接收它(裁决 5 的类型强制)。
#[derive(Clone)]
pub struct KernelCtx {
    pub(crate) phase: MemPhase,
}

impl KernelCtx {
    pub fn phase(&self) -> MemPhase {
        self.phase
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
