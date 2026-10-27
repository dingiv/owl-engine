//! **发射账三表**(2026-10-17 审查 §C3 拆分自 `server/mod.rs`)。
//!
//! 这三件必须**同口径**(旧实现散在同一文件的不同段,靠注释互指):
//! - [`stream_label`]:流位序 → 上屏字符串(发射账用);
//! - [`cmd_stream`]:命令 → 目标流位图(事件记账用;与上表同源);
//! - [`entry_of`]:命令 → 一条发射账目(核名/几何/块 id → 设备地址/输出块)。
//!
//! 拆出来的动机:①`mod.rs` 超 2000 生产行(AGENTS #7 硬顶);②三者是**同一张表的
//! 三面**,"扩充命令面时必须同步"只有放一起才守得住(与 `Command::kind()` 单一真源
//! 同律)。记账语义与纪律见 `crate::trace` 模块头。
//!
//! 注:`entry_of` 需要 `GpuCtx` 把块 id 解析成设备地址,故入参带 `ctx`
//! (原为 `GpuServer::trace_entry_of`;改成自由函数只为可搬,语义逐字不变)。

use crate::command::Command;
use crate::state::{GpuCtx, StreamId, STREAM_COMPUTE, STREAM_D2H, STREAM_H2D};
use crate::trace::TraceEntry;
use owl_iface::contract::Arg;

    /// 命令 → 发射账目(派发前调用;块 id 顺带解析成设备地址 —— 地址复用判据)
/// 命令 → 发射账目(派发前调用;块 id 顺带解析成设备地址 —— 地址复用判据。
/// 语义与纪律见 `crate::trace` 模块头;`out_id` = 最后一个 Block 实参 = 输出块)
pub(super) fn entry_of(ctx: &GpuCtx, cmd: &Command) -> TraceEntry {
        use crate::trace::TraceEntry as E;
        let resolve = |ids: &[u64]| -> Vec<u64> {
            let Ok(st) = ctx.stream(crate::state::STREAM_COMPUTE) else {
                return Vec::new();
            };
            ids.iter()
                .map(|id| ctx.block_ptr(*id, st).map(|(p, _)| p).unwrap_or(0))
                .collect()
        };
        match cmd {
            Command::DeviceSmCount { .. } => E::new("device_sm_count", ""),
            Command::DeviceArch { .. } => E::new("device_arch", ""),
            Command::BlockLedgerDump { .. } => E::new("block_ledger_dump", ""),
            // FI plan 表同步(W3):控制面命令(无 payload/无块)⇒ kind=ctl
            Command::FiPlanSync { .. } => E::new("fi_plan_sync", "ctl"),
            Command::MemPoolTrim { .. } => E::new("mem_pool_trim", ""),
            Command::MemPoolStats { .. } => E::new("mem_pool_stats", ""),
            Command::MemInfo { .. } => E::new("mem_info", ""),
            Command::UploadAndLaunch { dst, chunks, .. } => {
                let n: usize = chunks.iter().map(|c| c.len_bytes).sum();
                E::new("upload_and_launch", "h2d").bytes(n).id(dst.id)
            }
            Command::LaunchAndHarvest { id, want_bytes, .. } => {
                E::new("launch_and_harvest", "d2h").bytes(*want_bytes).id(*id)
            }
            Command::Launch { msg, .. } => {
                let ids: Vec<u64> = msg
                    .args
                    .iter()
                    .filter_map(|a| match a {
                        Arg::Block { id, .. } => Some(*id),
                        Arg::BlockSlice { id, .. } => Some(*id),
                        _ => None,
                    })
                    .collect();
                let ptrs = resolve(&ids);
                // 契约:最后一个 Block/BlockSlice = 输出(launch.rs 回传口径)
                let out_id = msg
                    .args
                    .iter()
                    .rev()
                    .find_map(|a| match a {
                        Arg::Block { id, .. } => Some(*id),
                        Arg::BlockSlice { id, .. } => Some(*id),
                        _ => None,
                    })
                    .unwrap_or(0);
                let mut e = E::new("launch", "compute")
                    .name(msg.kernel.name.clone())
                    .grid(msg.grid, msg.block, msg.shared_mem)
                    .bytes(msg.out_elems)
                    .ids(ids)
                    .ptrs(ptrs);
                e.out_id = out_id;
                e
            }
            Command::Htod { data, name, stream, .. } => E::new("htod", stream_label(*stream))
                .bytes(data.len())
                .name(name.clone()),
            Command::HtodChunk { block, data, stream, .. } => {
                E::new("htod_chunk", stream_label(*stream)).bytes(data.len()).id(*block)
            }
            Command::HtodChunks { writes, stream, .. } => E::new("htod_chunks", stream_label(*stream))
                .bytes(writes.iter().map(|w| w.2.len()).sum())
                .ids(writes.iter().map(|w| w.0)),
            Command::UploadPinned { dst, len_bytes, stream, .. } => {
                E::new("upload_pinned", stream_label(*stream)).bytes(*len_bytes).id(dst.id)
            }
            Command::Dtoh { id, want_bytes, .. } => {
                E::new("dtoh", "d2h").bytes(*want_bytes).id(*id)
            }
            Command::DtohAt { id, want_bytes, .. } => {
                E::new("dtoh_at", "d2h").bytes(*want_bytes).id(*id)
            }
            Command::DtohBatch { pieces, .. } => E::new("dtoh_batch", "d2h")
                .bytes(pieces.iter().map(|p| p.2).sum())
                .ids(pieces.iter().map(|p| p.0)),
            Command::Swap { dir, pieces, .. } => match dir {
                owl_iface::contract::SwapDir::Out => E::new("swap_out", "d2h")
                    .bytes(pieces.iter().map(|p| p.len).sum())
                    .ids(pieces.iter().map(|p| p.block)),
                owl_iface::contract::SwapDir::In => E::new("swap_in", stream_label(STREAM_COMPUTE as u8))
                    .bytes(pieces.iter().map(|p| p.len).sum())
                    .ids(pieces.iter().map(|p| p.block)),
            },
            Command::Alloc { n_bytes, zero, name, .. } => E::new("alloc", "compute")
                .bytes(*n_bytes)
                .name(format!("{}{name}", if *zero { "zero:" } else { "uninit:" })),
            Command::MemsetZero { block, len_bytes, .. } => {
                let ptrs = resolve(&[*block]);
                E::new("memset_zero", "compute").bytes(*len_bytes).id(*block).ptrs(ptrs)
            }
            Command::Free { ids, .. } => {
                // 回收前解析(此后账房已移除)⇒ free 条目自带地址
                let ptrs = resolve(ids);
                E::new("free", "").ids(ids.clone()).ptrs(ptrs)
            }
            // 原生批:外壳只计一条(逐条的真实发射账由图内**内层信封**各自记)
            Command::Batch { cmds, .. } => E::new("batch", "ctl").bytes(cmds.len()),
            Command::CopyBlock { src, dst, len_bytes, .. } => {
                E::new("copy_block", "compute").bytes(*len_bytes).ids([*src, *dst])
            }
            Command::CopyBatch { copies, .. } => {
                let ids: Vec<u64> = copies.iter().flat_map(|c| [c.0, c.2]).collect();
                let ptrs = resolve(&ids);
                E::new("copy_batch", "compute")
                    .bytes(copies.iter().map(|c| c.4).sum())
                    .ids(ids)
                    .ptrs(ptrs)
            }
            Command::Sync { .. } => E::new("sync", ""),
            Command::GraphBegin { .. } => E::new("graph_begin", ""),
            Command::GraphEnd { .. } => E::new("graph_end", ""),
            Command::GraphLaunch { .. } => E::new("graph_launch", "compute"),
            Command::GraphAbort { .. } => E::new("graph_abort", ""),
            Command::GraphDestroy { .. } => E::new("graph_destroy", ""),
            Command::AllocPinned { bytes, name, .. } => E::new("alloc_pinned", "").bytes(*bytes).name(name.clone()),
            // 查询面:零字节、零流、零地址(它不产设备动作);名字位记 top-N
            Command::LedgerReport { top, .. } => E::new("ledger_report", "").name(format!("top{top}")),
            Command::LedgerJson { .. } => E::new("ledger_json", ""),
            // peer 面(TP2 S1):VMM 分配/映射不触计算流 ⇒ 流列留空
            Command::PeerAlloc { bytes, .. } => E::new("peer_alloc", "").bytes(*bytes),
            Command::PeerImport { bytes, .. } => E::new("peer_import", "").bytes(*bytes),
            Command::PeerRelease { .. } => E::new("peer_release", ""),
            // 捕获期发射路由切换(图内 fork/join;B1 门):不触流(只建图的依赖边)
            Command::CaptureRoute { .. } => E::new("capture_route", ""),
            Command::CaptureJoin { .. } => E::new("capture_join", ""),
            Command::Close { .. } => E::new("close", ""),
        }
    }

    /// 命令的**目标流**(E1 记账表:哪些命令在 GPU 上落活、落在哪条流)。
    ///
    /// 与 `trace_entry_of` 的流标签同表(那里是给 trace 看的字符串;本表是
    /// 给事件记账看的 StreamId)。**扩充命令面时必须同步**(E2 复合命令按
    /// 自身语义定流)。
pub(super) fn cmd_stream(c: &Command) -> u8 {
        match c {
            // H2D 流:一切 host→device 搬运
            Command::Htod { stream, .. } => 1u8 << (*stream as usize),
            // **上传落点流由命令自述**(2026-10-17):参数面走 COMPUTE(同流有序),
            // 批量面走 H2D(重叠优先);记账/审计据此上屏
            Command::HtodChunk { stream, .. } | Command::HtodChunks { stream, .. } => {
                1u8 << (*stream as usize)
            }
            Command::UploadPinned { stream, .. } => 1u8 << (*stream as usize),
            // COMPUTE 流:发射 / 清零分配 / 设备内拷贝 / 图回放
            Command::Launch { .. }
            | Command::Alloc { .. }
            | Command::MemsetZero { .. }
            | Command::CopyBlock { .. }
            | Command::CopyBatch { .. }
            | Command::GraphLaunch { .. } => 1u8 << STREAM_COMPUTE,
            // D2H 流:一切 device→host 回读
            Command::Dtoh { .. } | Command::DtohAt { .. } | Command::DtohBatch { .. } => {
                1u8 << STREAM_D2H
            }
            // swap:按方向落流(Out = D2H 回读;**In = COMPUTE 写入** —— 内容
            // 就是下一步计算要读的状态,同流 FIFO 才结构性有序;见 harvest.rs 注)
            Command::Swap { dir, .. } => match dir {
                owl_iface::contract::SwapDir::Out => 1u8 << STREAM_D2H,
                owl_iface::contract::SwapDir::In => 1u8 << STREAM_COMPUTE,
            },
            // 复合:两条腿各触一条流(E2 位图的第一个用例)
            Command::UploadAndLaunch { .. } => (1u8 << STREAM_H2D) | (1u8 << STREAM_COMPUTE),
            Command::LaunchAndHarvest { .. } => (1u8 << STREAM_COMPUTE) | (1u8 << STREAM_D2H),
            // 无 GPU 活/图控制/同步/宿主面:不记账(调试投影亦不触流)
            _ => 0,
        }
    }

/// 墙钟毫秒(诊断落盘用;时钟回拨 = 0,非失败路径 —— 与 `prompt_dump` 同律)
/// **流位序 → 上屏标签**(单一事实源;发射账 `trace_entry_of` 与事件记账
/// `GpuServer::cmd_stream` 共用 —— 两处必须同口径。旧版在 trace 侧硬写字符串,
/// 曾把参数面的写一律标成 "h2d",2026-10-17 桶图复用案排查时被它误导过一轮)。
pub(super) fn stream_label(bit: u8) -> &'static str {
    match bit as StreamId {
        STREAM_H2D => "h2d",
        STREAM_COMPUTE => "compute",
        STREAM_D2H => "d2h",
        _ => "?",
    }
}
