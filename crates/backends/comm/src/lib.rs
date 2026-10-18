//! TP 通信 —— 公理 A2(charter.md §二)的代码落点。
//!
//! 通信在这里是架构对象:后端按消息尺寸路由(A2.1)、可捕获性是类型信息
//! (A2.2)、融合口子内建(A2.3)。数据面的 CUDA 句柄在 M2 接线,本 crate
//! 先立策略骨架。
//!
//! # 模块地图(TP2 施工中)
//!
//! - [`peer`]:对等体能力分档与授予编排(S1;唯一判定点 —— 见
//!   `docs/arch/TP2-施工设计.md` §2)
//! - [`ar`]:allreduce 的 staging 布局与 chunk 计划(S2;协议不变量 I1-I5)
//! - 本文件:策略骨架(`CommCap`/`CommBackend`/`CommRouter`/`Topology`)
//!
//! **壳子状态**:纯逻辑已落地可测;碰设备的两处入口
//! ([`peer::PeerPlan::open`] / [`ar::AllReduce::enqueue`])是结构化
//! `NotYetImplemented` 桩 —— 按 AGENTS 纪律,可失败逻辑不得写成静默空函数。

pub mod ar;
pub mod peer;

use owl_graph::GraphPhase;

/// 后端的可捕获性声明(A2.2)—— 编排层据此排段,禁止运行时撞上。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommCap {
    /// 裸 kernel 实现(one-shot/two-shot AR):可在图内烙
    InGraph,
    /// 段边界专用(SHM host 中转保险丝等):编排层保证不落进捕获段。
    /// 注:NCCL 已除名(A2.5)——账外显存不可登记,违反 A5.2。
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
///
/// `mechanisms` = 分档矩阵(S1):**协议选型依据** —— 热路径零重判,
/// 初始化期一次性算定(判据见 [`peer::classify`])。
#[derive(Debug, Clone)]
pub struct Topology {
    /// UUID 字符串,顺序 = rank 序
    pub uuids: Vec<String>,
    /// P2P 能力矩阵;探测在初始化完成,热路径零探测
    pub p2p_matrix: Vec<Vec<bool>>,
    /// 机制分档矩阵(`mechanisms[i][j]` = i 看 j 的机制;对角线无意义)
    pub mechanisms: Vec<Vec<peer::PeerMechanism>>,
}

impl Topology {
    /// 从能力表构建(纯逻辑;真正的探测在 S1 的 boot 路径)。
    ///
    /// 对角线取 `Static`(= 本卡自身,不参与 peer 协议),避免用 `Option`
    /// 让每个取值点都要解包。
    pub fn from_caps(caps: &[peer::PeerCaps]) -> Self {
        let n = caps.len();
        let p2p_matrix = caps
            .iter()
            .map(|a| caps.iter().map(|b| a.p2p_supported && b.p2p_supported).collect())
            .collect();
        let mechanisms = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if i == j {
                            peer::PeerMechanism::Static
                        } else {
                            peer::classify(&caps[i], &caps[j])
                        }
                    })
                    .collect()
            })
            .collect();
        Self {
            uuids: caps.iter().map(|c| c.uuid.clone()).collect(),
            p2p_matrix,
            mechanisms,
        }
    }

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
    /// 分档/额度层拒绝(S1):不支持 P2P / 混插(v1 不支) / 额度为 0 等。
    /// 带 detail 是为了让拒启日志能直接指向成因(不猜)。
    #[error("peer 组合被拒: {detail}")]
    PeerRejected { detail: String },
    /// 单笔映射额度超限(192MB 墙一族;响亮拒,不静默缩容)
    #[error("peer 单笔额度超限: 请求 {requested}B > 上限 {limit}B(机制 {mechanism:?})")]
    PeerAllocOverLimit {
        requested: u64,
        limit: u64,
        mechanism: peer::PeerMechanism,
    },
    /// 施工中未实现的阶段(**不许改成静默 `Ok(())`**;S1/S2 落地后逐条删)
    #[error("未实现(TP2 施工): {stage}")]
    NotYetImplemented { stage: &'static str },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_from_caps_fills_mechanisms() {
        let a = peer::PeerCaps::new("GPU-a", 32768 << 20, true);
        let b = peer::PeerCaps::new("GPU-b", 256 << 20, true);
        let tp = Topology::from_caps(&[a, b]);
        assert_eq!(tp.tp(), 2);
        assert_eq!(tp.uuids, vec!["GPU-a".to_string(), "GPU-b".to_string()]);
        // 对角线 = Static(自身,不参与协议)
        assert_eq!(tp.mechanisms[0][0], peer::PeerMechanism::Static);
        // 混插 = 拒(D5)
        assert_eq!(tp.mechanisms[0][1], peer::PeerMechanism::Rejected);
        assert_eq!(tp.mechanisms[1][0], peer::PeerMechanism::Rejected);
        // 3080 对 = Window
        let c = || peer::PeerCaps::new("GPU-c", 256 << 20, true);
        let t2 = Topology::from_caps(&[c(), c()]);
        assert_eq!(t2.mechanisms[0][1], peer::PeerMechanism::Window);
    }

    /// 路由的兜底律:必须存在 `SegmentBoundary` 后端(否则 `route` panic)。
    struct DummySeg;
    impl CommBackend for DummySeg {
        fn cap(&self) -> CommCap {
            CommCap::SegmentBoundary
        }
        fn sweet_spot(&self) -> core::ops::RangeInclusive<u64> {
            0..=u64::MAX
        }
        fn allreduce(&self, _buf: *mut u8, _bytes: u64, _phase: GraphPhase) {}
    }

    #[test]
    fn router_falls_back_to_segment_backend() {
        let r = CommRouter::new(vec![Box::new(DummySeg)]);
        assert!(matches!(r.route(0).cap(), CommCap::SegmentBoundary));
        assert!(matches!(r.route(u64::MAX).cap(), CommCap::SegmentBoundary));
    }
}
