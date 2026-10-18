//! 对等体能力分档与 PeerShared 授予编排 —— **TP2 S1 的唯一判定点**(A2.4 落地)。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §2(判据 = 单笔上限;对称握手序列)。
//!
//! # 纪律(D1 裁决配套)
//!
//! **唯一规范 = 本模块 + `owl-cuda::command` 的 `Peer*` 三命令**。
//! `owl-iface` 根模块里那套 `Pool::malloc_peer_shared` / `Device::enable_peer_access`
//! / `Device::map_remote` 声明是**历史设计记录(非规范)**,不得作为实现依据。
//!
//! # 职责边界(S4 ⑤-K2 定稿)
//!
//! 本模块 = **策略**:能力分档(`classify`)/ 单笔上限(`per_alloc_limit`)/
//! 额度核算与拒绝(`PeerPlan::plan`)/ staging 布局(`ar::StagingLayout`)
//! —— 全部可在 CPU 上单测。
//!
//! **握手编排(分配→导出→导入→授予)不在本 crate**:它要同时持有两侧
//! actor 句柄(`peer_alloc`/`peer_import`),而 `owl-comm` 不依赖 `owl-cuda`
//! —— 实现落在 `owl-engine::tp`(S4 ⑤-K2;`TpEngine::assemble`),那里
//! 消费本模块的 `PeerPlan` 作为唯一判定。

use crate::CommError;

// 三个额度常量**单源在 `owl_shared::device`**(2026-10-16 C1:执行层
// `owl-cuda` 也要读同一份 —— 不许两边各写一套)。此处 re-export 保持
// 本模块 API 路径(`crate::peer::PEER_ALLOC_HARD_CAP`)稳定。
pub use owl_shared::device::{PEER_ALLOC_HARD_CAP, STATIC_BAR1_MIN, WINDOW_BAR1_RESERVE};

/// 单卡的对等体能力描述(boot 期探测一次;热路径零探测 —— A2.4)。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerCaps {
    /// PCI 总线 UUID(钉卡唯一合法凭据,数字序禁)
    pub uuid: String,
    /// BAR1 总量(bytes;`nvidia-smi --query-gpu=bar1.total` 或 NVML)
    pub bar1_bytes: u64,
    /// 该卡能否参与 P2P(`cuDeviceGetP2PAttribute` 的 ACCESS_SUPPORTED)
    pub p2p_supported: bool,
}

impl PeerCaps {
    pub fn new(uuid: impl Into<String>, bar1_bytes: u64, p2p_supported: bool) -> Self {
        Self { uuid: uuid.into(), bar1_bytes, p2p_supported }
    }

    /// 本卡是否够宽(BAR1 ≥ [`STATIC_BAR1_MIN`])。
    pub fn is_wide_bar1(&self) -> bool {
        self.bar1_bytes >= STATIC_BAR1_MIN
    }
}

/// 一对卡之间的可映射机制(分档结果,**不是**运行时探测)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerMechanism {
    /// 双方 BAR1 均 ≥ [`STATIC_BAR1_MIN`]:静态映射,无动态窗口墙
    /// (单笔仍受 [`PEER_ALLOC_HARD_CAP`] 管辖,便于统一纪律)。
    Static,
    /// 双方均为窄 BAR1(动态 BAR1 窗口):单笔 ≤ [`per_alloc_limit`]。
    Window,
    /// 不允许:任一方不支持 P2P,或 v1 不支持的组合(混插 Static×Window;D5 裁决)。
    Rejected,
}

/// 分档 —— **唯一判据 = BAR1 尺寸与 P2P 支持**,不掺任何经验规则。
///
/// 混插(一宽一窄)v1 直接拒(D5):两侧机制不同会让协议必须同时兼容两条路,
/// 与"最小库"目标冲突;将来要支持时在此处开分支,而不是在授予路径里打补丁。
pub fn classify(a: &PeerCaps, b: &PeerCaps) -> PeerMechanism {
    if !a.p2p_supported || !b.p2p_supported {
        return PeerMechanism::Rejected;
    }
    match (a.is_wide_bar1(), b.is_wide_bar1()) {
        (true, true) => PeerMechanism::Static,
        (false, false) => PeerMechanism::Window,
        _ => PeerMechanism::Rejected,
    }
}

/// 单笔上限(字节)。
///
/// - `Static`  → 硬上限(窄口统一纪律;实践上静态路的真实上限远高于此);
/// - `Window`  → `min(两者 BAR1 较小者 − 预留, 硬上限)`;
/// - `Rejected`→ `0`(任何非零请求都必须被拒 —— 拒启比"降级跑"优先)。
pub fn per_alloc_limit(mech: PeerMechanism, a: &PeerCaps, b: &PeerCaps) -> u64 {
    match mech {
        PeerMechanism::Static => PEER_ALLOC_HARD_CAP,
        PeerMechanism::Window => {
            let narrow = a.bar1_bytes.min(b.bar1_bytes);
            narrow
                .saturating_sub(WINDOW_BAR1_RESERVE)
                .min(PEER_ALLOC_HARD_CAP)
        }
        PeerMechanism::Rejected => 0,
    }
}

/// 一对卡的 peer 计划(分档 + 上限 + staging 额度核算;S1 的握手入口)。
///
/// 纯逻辑部分(本文件)先落地并可测;真正碰设备的部分在 [`PeerPlan::open`]。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerPlan {
    pub mechanism: PeerMechanism,
    /// 单笔可映射上限(bytes)
    pub per_alloc_limit: u64,
    /// 每 rank 需要的 staging 额度(bytes;含 header + N 个 slot)
    pub staging_bytes: u64,
}

impl PeerPlan {
    /// 计划:分档 → 上限 → 额度核算。**额度超限 = 结构化拒绝**(不静默缩容)。
    ///
    /// `staging_bytes` 由调用方按 [`crate::ar::StagingLayout::total_bytes`] 算出
    /// (两 rank 必须同值 —— 协议对称性前提)。
    pub fn plan(
        a: &PeerCaps,
        b: &PeerCaps,
        staging_bytes: u64,
    ) -> Result<Self, CommError> {
        let mechanism = classify(a, b);
        let limit = per_alloc_limit(mechanism, a, b);
        if mechanism == PeerMechanism::Rejected {
            return Err(CommError::PeerRejected {
                detail: format!(
                    "P2P 组合被拒: a={}(bar1={}MiB, p2p={}) b={}(bar1={}MiB, p2p={})",
                    a.uuid,
                    a.bar1_bytes >> 20,
                    a.p2p_supported,
                    b.uuid,
                    b.bar1_bytes >> 20,
                    b.p2p_supported
                ),
            });
        }
        if staging_bytes == 0 {
            return Err(CommError::PeerRejected {
                detail: "staging_bytes = 0(协议不可能成立)".into(),
            });
        }
        if staging_bytes > limit {
            return Err(CommError::PeerAllocOverLimit {
                requested: staging_bytes,
                limit,
                mechanism,
            });
        }
        Ok(Self { mechanism, per_alloc_limit: limit, staging_bytes })
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    fn wide() -> PeerCaps {
        PeerCaps::new("GPU-3090ti-a", 32768 << 20, true)
    }
    fn narrow() -> PeerCaps {
        PeerCaps::new("GPU-3080-a", 256 << 20, true)
    }

    #[test]
    fn classify_three_ways() {
        // 3090 对 → Static
        assert_eq!(classify(&wide(), &wide()), PeerMechanism::Static);
        // 3080 对 → Window
        assert_eq!(classify(&narrow(), &narrow()), PeerMechanism::Window);
        // 混插 → 拒(D5)
        assert_eq!(classify(&wide(), &narrow()), PeerMechanism::Rejected);
        assert_eq!(classify(&narrow(), &wide()), PeerMechanism::Rejected);
        // P2P 不支持 → 拒(哪怕 BAR1 宽)
        let mut no_p2p = wide();
        no_p2p.p2p_supported = false;
        assert_eq!(classify(&no_p2p, &wide()), PeerMechanism::Rejected);
    }

    #[test]
    fn window_limit_is_min_bar1_minus_reserve() {
        let m = classify(&narrow(), &narrow());
        // 256MiB − 64MiB = 192MiB,恰好等于硬上限
        assert_eq!(per_alloc_limit(m, &narrow(), &narrow()), 192 << 20);
        // 若 BAR1 更窄(如 128MiB):128 − 64 = 64MiB
        let tiny = PeerCaps::new("GPU-tiny", 128 << 20, true);
        assert_eq!(per_alloc_limit(classify(&tiny, &tiny), &tiny, &tiny), 64 << 20);
    }

    #[test]
    fn rejected_limit_is_zero() {
        let m = classify(&wide(), &narrow());
        assert_eq!(per_alloc_limit(m, &wide(), &narrow()), 0);
    }

    #[test]
    fn plan_rejects_over_limit_and_zero() {
        let m = 2 * (32 << 20) + crate::ar::HEADER_BYTES; // 64MiB + 64B
        // 正常:3080 对 staging 64MiB 在 192MiB 内
        let p = PeerPlan::plan(&narrow(), &narrow(), m).expect("3080 对计划应成立");
        assert_eq!(p.mechanism, PeerMechanism::Window);
        assert_eq!(p.staging_bytes, m);
        // 超限:200MiB
        let e = PeerPlan::plan(&narrow(), &narrow(), 200 << 20).unwrap_err();
        assert!(matches!(e, CommError::PeerAllocOverLimit { .. }));
        // 零额度
        assert!(PeerPlan::plan(&narrow(), &narrow(), 0).is_err());
        // 混插拒
        assert!(matches!(
            PeerPlan::plan(&wide(), &narrow(), m).unwrap_err(),
            CommError::PeerRejected { .. }
        ));
    }
}
