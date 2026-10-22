//! **生命周期态的命令负门表**(2026-10-17 审查 §C3 拆分自 `server/mod.rs`)。
//!
//! 两张表都是"**逐命令**表态",与命令面(`Command` 变体集)必须同步 ——
//! 与 `trace_map`(发射账三表)、`dispatch_inner`(正门)是同族:命令面新增变体时
//! 编译器会把这几处**逐一点名**(这正是它们该挨在一起的理由)。
//! - [`reject_closed`]:Closing 态 —— **每条命令的 ack 都必须回** `ServerClosed`
//!   (否则客户端 Future 永远 pending;这是"本轮挂死"的历史根因);
//! - [`reject_with`]:捕获窗负门 —— 捕获期只允许 Launch/Alloc(切 slab)/
//!   MemsetZero/GraphEnd/Close,其余按名响亮拒(带原因)。
//!
//! 拆出来只为 `mod.rs` 回到 2000 生产行硬顶内(AGENTS #7);语义零变化。

use crate::command::Command;
use owl_iface::contract::ModelError;

    /// Closing 态拒绝:每条命令的 ack 都必须回 ServerClosed
    /// (否则客户端 Future 永远 pending —— 本轮挂死根因)
pub(super) fn reject_closed(cmd: Command) {
        macro_rules! closed { ($ack:expr) => { $ack.send(Err(ModelError::ServerClosed)) } }
        match cmd {
            Command::Close { ack } => ack.send(Ok(())), // 幂等
            Command::Alloc { ack, .. } => closed!(ack),
            Command::MemsetZero { ack, .. } => closed!(ack),
            Command::Htod { ack, .. } => closed!(ack),
            Command::HtodChunk { ack, .. } => closed!(ack),
            Command::HtodChunks { ack, .. } => closed!(ack),
            Command::AllocPinned { ack, .. } => closed!(ack),
            Command::LedgerReport { ack, .. } => closed!(ack),
            Command::LedgerJson { ack, .. } => closed!(ack),
            Command::UploadPinned { ack, .. } => closed!(ack),
            Command::DeviceSmCount { ack } => closed!(ack),
            Command::BlockLedgerDump { ack } => closed!(ack),
            Command::MemPoolTrim { ack } => closed!(ack),
            Command::MemPoolStats { ack } => closed!(ack),
            Command::MemInfo { ack } => closed!(ack),
            Command::FiPlanSync { ack, .. } => closed!(ack),
            Command::UploadAndLaunch { ack, .. } => closed!(ack),
            Command::LaunchAndHarvest { ack, .. } => closed!(ack),
            Command::Dtoh { ack, .. } => closed!(ack),
            Command::DtohAt { ack, .. } => closed!(ack),
            Command::DtohBatch { ack, .. } => closed!(ack),
            Command::Swap { ack, .. } => closed!(ack),
            Command::Free { ack, .. } => closed!(ack),
            Command::Batch { ack, .. } => closed!(ack),
            Command::CopyBlock { ack, .. } => closed!(ack),
            Command::CopyBatch { ack, .. } => closed!(ack),
            Command::Launch { ack, .. } => closed!(ack),
            Command::Sync { ack, .. } => closed!(ack),
            Command::GraphBegin { ack, .. } => closed!(ack),
            Command::GraphEnd { ack, .. } => closed!(ack),
            Command::GraphLaunch { ack, .. } => closed!(ack),
            Command::GraphAbort { ack, .. } => closed!(ack),
            Command::GraphDestroy { ack, .. } => closed!(ack),
            Command::PeerAlloc { ack, .. } => closed!(ack),
            Command::PeerImport { ack, .. } => closed!(ack),
            Command::PeerRelease { ack, .. } => closed!(ack),
            Command::CaptureRoute { ack, .. } => closed!(ack),
            Command::CaptureJoin { ack, .. } => closed!(ack),
        }
    }

/// 捕获窗负门:按 `why` 逐命令响亮拒(负门清单见模块头)。
pub(super) fn reject_with(cmd: Command, why: String, sink: owl_shared::log::LogSink) {
        macro_rules! reject { ($ack:expr) => { $ack.send(Err(ModelError::Msg(why.clone()))) } }
        let name = |c: &Command| match c {
            // E2 复合(分相计时标签;与 Command::kind() 同名单一真源)
            Command::DeviceSmCount { .. } => "device_sm_count",
            Command::BlockLedgerDump { .. } => "block_ledger_dump",
            Command::MemPoolTrim { .. } => "mem_pool_trim",
            Command::MemPoolStats { .. } => "mem_pool_stats",
            Command::MemInfo { .. } => "mem_info",
            Command::FiPlanSync { .. } => "FiPlanSync",
            Command::UploadAndLaunch { .. } => "upload_and_launch",
            Command::LaunchAndHarvest { .. } => "launch_and_harvest",
            Command::GraphBegin { .. } => "GraphBegin",
            Command::GraphEnd { .. } => "GraphEnd",
            Command::GraphLaunch { .. } => "GraphLaunch",
            Command::GraphAbort { .. } => "GraphAbort",
            Command::GraphDestroy { .. } => "GraphDestroy",
            Command::Dtoh { .. } => "Dtoh",
            Command::DtohAt { .. } => "DtohAt",
            Command::DtohBatch { .. } => "DtohBatch",
            Command::Swap { .. } => "Swap",
            Command::Alloc { .. } => "Alloc",
            Command::MemsetZero { .. } => "MemsetZero",
            Command::Htod { .. } => "Htod",
            Command::HtodChunk { .. } => "HtodChunk",
            Command::HtodChunks { .. } => "HtodChunks",
            Command::AllocPinned { .. } => "AllocPinned",
            Command::LedgerReport { .. } => "LedgerReport",
            Command::LedgerJson { .. } => "LedgerJson",
            Command::UploadPinned { .. } => "UploadPinned",
            Command::Free { .. } => "Free",
            Command::CopyBlock { .. } => "CopyBlock",
            Command::CopyBatch { .. } => "CopyBatch",
            Command::Sync { .. } => "Sync",
            Command::Launch { .. } => "Launch",
            Command::PeerAlloc { .. } => "PeerAlloc",
            Command::PeerImport { .. } => "PeerImport",
            Command::PeerRelease { .. } => "PeerRelease",
            Command::CaptureRoute { .. } => "CaptureRoute",
            Command::CaptureJoin { .. } => "CaptureJoin",
            Command::Batch { .. } => "Batch",
            Command::Close { .. } => "Close",
        };
        sink.logf(owl_shared::log::LVL_WARN, "owl-gpu", format_args!("捕获期拒绝: {} —— {why}", name(&cmd)));
        match cmd {
            Command::GraphBegin { ack } => reject!(ack),
            Command::GraphEnd { ack } => reject!(ack),
            Command::GraphLaunch { ack, .. } => reject!(ack),
            Command::GraphAbort { ack, .. } => reject!(ack),
            Command::GraphDestroy { ack, .. } => reject!(ack),
            // **FI plan 表同步(W3)**:含 H2D(表刷进 int_ws)⇒ 与 Htod 同类,
            // 捕获期**结构化拒**(图回放期不会经过宿主处理器,刷表必须由图外做)
            Command::FiPlanSync { ack, .. } => reject!(ack),
            Command::Alloc { ack, .. } => reject!(ack),
            Command::MemsetZero { ack, .. } => reject!(ack),
            Command::Htod { ack, .. } => reject!(ack),
            Command::HtodChunk { ack, .. } => reject!(ack),
            Command::HtodChunks { ack, .. } => reject!(ack),
            Command::AllocPinned { ack, .. } => reject!(ack),
            Command::LedgerReport { ack, .. } => reject!(ack),
            Command::LedgerJson { ack, .. } => reject!(ack),
            Command::UploadPinned { ack, .. } => reject!(ack),
            // **E2 负门**:复合命令内含搬运/回读(不可捕获)⇒ 捕获期结构化拒
            // 设备 SM 数 = 恒定属性查询(**不碰设备状态、不发射**)⇒ 捕获期**放行**
            // (与 LedgerDump/Trim 不同:那两个要排空/改池;本命令纯读属性)
            Command::DeviceSmCount { .. } => {}
            Command::BlockLedgerDump { ack } => reject!(ack),
            // 池回收 = host 侧池操作(与图捕获无关的全局动作)⇒ 捕获期拒
            Command::MemPoolTrim { ack } => reject!(ack),
            Command::MemPoolStats { ack } => reject!(ack),
            Command::MemInfo { ack } => reject!(ack),
            Command::UploadAndLaunch { ack, .. } => reject!(ack),
            Command::LaunchAndHarvest { ack, .. } => reject!(ack),
            Command::Dtoh { ack, .. } => reject!(ack),
            // 带偏移读同 Dtoh:捕获窗内**结构化拒**(状态存储的搬运只在段边界)
            Command::DtohAt { ack, .. } => reject!(ack),
            Command::DtohBatch { ack, .. } => reject!(ack),
            // **swap 同 Dtoh/Batch**:搬运只在段边界,捕获窗内结构化拒
            Command::Swap { ack, .. } => reject!(ack),
            Command::Sync { ack, .. } => reject!(ack),
            Command::Free { ack, .. } => reject!(ack),
            Command::CopyBlock { ack, .. } => reject!(ack),
            Command::CopyBatch { ack, .. } => reject!(ack),
            Command::Launch { ack, .. } => reject!(ack),
            // **捕获期专属口**(图内 fork/join;B1 门):捕获窗内**放行** ——
            // 它们本就是为捕获期而生的(非捕获期由 `dispatch` 正门拒)
            Command::CaptureRoute { .. } => {}
            Command::CaptureJoin { .. } => {}
            // peer 面(VMM 分配/映射):要动 BAR1/映射表(host 侧动作)⇒ 捕获期结构化拒
            Command::PeerAlloc { ack, .. } => reject!(ack),
            Command::PeerImport { ack, .. } => reject!(ack),
            Command::PeerRelease { ack, .. } => reject!(ack),
            // 批外壳:本条路径不该到(捕获期由 `handle_batch` 逐条过白名单)
            Command::Batch { ack, .. } => reject!(ack),
            Command::Close { ack } => ack.send(Ok(())), // 幂等
        }
    }
