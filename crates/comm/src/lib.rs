//! TP 通信 —— 公理 A2(charter.md §二)的代码落点。
//!
//! 通信在这里是架构对象:后端按消息尺寸路由(A2.1)、可捕获性是类型信息
//! (A2.2)、融合口子内建(A2.3)。数据面的 CUDA 句柄在 M2 接线,本 crate
//! 先立策略骨架。

use owl_graph::GraphPhase;

/// 后端的可捕获性声明(A2.2)—— 编排层据此排段,禁止运行时撞上。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommCap {
    /// 裸 kernel 实现(one-shot AR):可在图内烙
    InGraph,
    /// NCCL 类:只能在段边界 eager 执行;编排层保证不落进捕获段
    SegmentBoundary,
}

/// 通信后端 trait(静态分发为主,注册表期允许 dyn,见 REQ-CODE-01)。
///
/// 实现者契约:
/// - `InGraph` 后端的 `allreduce` 不得含同步/D2H/host 分支(A1.5);
/// - `SegmentBoundary` 后端在 `GraphPhase::Capturing` 下调用即契约违约,
///   实现应 panic(架构错误,不是错误恢复场景)。
pub trait CommBackend: Send + Sync {
    fn cap(&self) -> CommCap;

    /// 字节尺寸路由阈值:小于此值本后端有优势。
    /// (one-shot AR ≈ 0..=1MB;NCCL ≈ 大消息无上界。)
    fn sweet_spot(&self) -> core::ops::RangeInclusive<u64>;

    /// allreduce。buf 的设备指针契约与 kernel 层一致(见 kernels crate)。
    /// M2 接 CUDA 实现;此处仅策略签名。
    fn allreduce(&self, buf: *mut u8, bytes: u64, phase: GraphPhase);
}

/// 尺寸路由器(A2.1):策略显式化 —— 同一次 forward 内,decode 步与
/// prefill 步可以(且应该)落在不同后端。
pub struct CommRouter {
    backends: Vec<Box<dyn CommBackend>>,
}

impl CommRouter {
    pub fn new(backends: Vec<Box<dyn CommBackend>>) -> Self {
        assert!(!backends.is_empty(), "至少一个通信后端");
        Self { backends }
    }

    /// 按消息尺寸选后端。多个后端覆盖同尺寸时取 sweet_spot 更窄者
    /// (更专一者胜)。
    pub fn route(&self, bytes: u64) -> &dyn CommBackend {
        self.backends
            .iter()
            .map(|b| b.as_ref())
            .filter(|b| b.sweet_spot().contains(&bytes))
            .min_by_key(|b| {
                let r = b.sweet_spot();
                (*r.end() - *r.start()).saturating_sub(bytes)
            })
            .unwrap_or_else(|| {
                self.backends
                    .iter()
                    .map(|b| b.as_ref())
                    .find(|b| b.cap() == CommCap::SegmentBoundary)
                    .expect("必须有 SegmentBoundary 兜底后端")
            })
    }
}

/// 卡拓扑(A2.4):UUID 钉卡唯一合法;数字序在此类型上不可表达。
#[derive(Debug, Clone)]
pub struct Topology {
    /// UUID 字符串,顺序 = rank 序
    pub uuids: Vec<String>,
    /// P2P 能力矩阵;探测在初始化完成,热路径零探测
    pub p2p_matrix: Vec<Vec<bool>>,
}

impl Topology {
    pub fn tp(&self) -> usize {
        self.uuids.len()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CommError {
    #[error("UUID 钉卡失败: {0}")]
    PinFailed(String),
    #[error("P2P 探测发现雷卡组合(黑名单命中),拒绝初始化")]
    P2PBlacklisted,
}
