//! **TP 通信面(`ar` + `peer`)—— GPU server 的通信职责**。
//!
//! 2026-10-18:**由独立 crate `owl-comm` 合并进 `owl-cuda`**。理由(用户裁决):
//! "你本来就是在写 GPU server —— 通信集合就是 GPU server 做的事;需要什么能力
//! 就在 server 里写出来、暴露接口给上层调。"
//!
//! 合并的实际收益 = **没有跨包约束**:协议(本模块)· VMM 授予(server 的
//! `Peer*` 三命令)· 两核(`owl-kernels`)· 编排(`engine::tp`)现在可以在
//! 同一个 crate 内一起改 —— 例如"对称缓冲 ⇒ 1-stage allreduce"(把 GEMM 输出
//! 直接落进 peer-shared 区、省掉整段 stage 协议)不再需要跨包对齐接口。
//!
//! # 模块地图
//!
//! - [`peer`]:对等体能力分档与授予额度(S1;唯一判定点 —— `TP2-施工设计.md` §2)
//! - [`ar`]:allreduce 的 staging 布局与 chunk 计划(S2;不变量 I1–I5)
//!
//! # 已退役:M2 之前的"策略骨架"(合并时删除)
//!
//! 原 `owl-comm` 还带过一层规划占位(`CommCap` / `CommBackend` / `CommRouter` /
//! `Topology` + `allreduce(buf, bytes, GraphPhase)` 签名)。**它们从未被实现、
//! 也从未被调用**(全仓 0 处使用),是 M2 之前的骨架;真正的通信实现早已落在
//! `ar.rs` + kernels 两核 + server 的 `Peer*` 三命令 + `engine::tp` 的编排上。
//! 保留它们会把 `owl-cuda → owl-graph` 这条方向可疑的依赖带进来 ⇒ 删除;
//! 历史见 git(`crates/backends/comm/src/lib.rs` 的 HEAD 版本)。

pub mod ar;
pub mod peer;

/// 通信面错误(peer 分档 / 额度层;S1)。
#[derive(Debug, thiserror::Error)]
pub enum CommError {
    #[error("UUID 钉卡失败: {0}")]
    PinFailed(String),
    #[error("P2P 探测发现雷卡组合(黑名单命中),拒绝初始化")]
    P2PBlacklisted,
    /// 分档/额度层拒绝(S1):不支持 P2P / 混插(v1 不支持) / 额度为 0 等。
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
}
