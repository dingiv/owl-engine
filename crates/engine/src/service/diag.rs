//! **引擎诊断快照（engine 侧组装，server 只做 JSON 形状）**。
//!
//! 2026-11-15 P1 自 `apps/server`（`executor.rs::VramSnapshot` + `telemetry.rs`）下沉：
//! 这些事实（store 三层账 / 块账房 / 池账 / 会话资产 / 淘汰预览 / 合批状态）
//! 全部是引擎内部状态，组装必须发生在**引擎线程内**（同一时刻取，避免交错），
//! 且类型不该长在 HTTP 应用里。
//!
//! 两个快照：
//! - [`DiagSnapshot`]：按需（`/debug/vram`）—— 含设备命令（块账/池账），**空闲期**取；
//! - [`QueueFacts`]：每拍（泵循环顺手抄一份）—— 纯 host 事实，HTTP 零等待读。

use std::sync::{Arc, Mutex};

use crate::RunningEngine;
use owl_iface::contract::DeviceClient;

/// **块账行**（engine 面投影）。
///
/// 2026-11-16 P3：`/debug/blocks` 的投影以前直接吃
/// `owl_iface::contract::BlockLedgerRow` ⇒ server 永远摘不掉后端依赖。
/// 这里是引擎给出的稳定形状；JSON 排版仍是 server 的事。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockRow {
    pub id: u64,
    /// 账长（字节口径；块账本形态为字节，不带 dtype）
    pub bytes: usize,
    /// 设备地址（解析当刻）
    pub addr: u64,
    /// 所属账本（actor 序号）
    pub ledger: u32,
    /// 分配点点名（`selfcheck.*` / `kv.*` / 层名 / 核名）
    pub name: String,
}

impl From<owl_iface::contract::BlockLedgerRow> for BlockRow {
    fn from(r: owl_iface::contract::BlockLedgerRow) -> Self {
        Self {
            id: r.id,
            bytes: r.bytes,
            addr: r.addr,
            ledger: r.ledger,
            name: r.name,
        }
    }
}

/// **发射账行**（`/debug/launches` 的投影面）。
///
/// `ptrs` = 块 id → 设备地址（地址复用是跨流 UAF 类缺陷的唯一钥匙）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchRow {
    pub seq: u64,
    /// host 单调时钟（微秒）
    pub t_us: u64,
    pub cmd: &'static str,
    pub stream: &'static str,
    pub name: String,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
    pub bytes: usize,
    pub ids: Vec<u64>,
    pub out_id: u64,
    pub ptrs: Vec<u64>,
}

/// 发射账快照（`enabled=false` = 该档没开 trace 环）。
#[derive(Clone, Debug, Default)]
pub struct LaunchTrace {
    pub enabled: bool,
    pub capacity: usize,
    pub in_ring: usize,
    pub total: u64,
    pub rows: Vec<LaunchRow>,
}

/// 读**发射账**（进程级 trace 环；`[cuda] trace_ring = N` 开启）。
///
/// 为什么是自由函数而不是 `EngineClient` 请求：这账是**后端写入的进程内环形**，
/// 读它不需要引擎线程（也不该排在 turn 后面）。HTTP 线程直接调，零往返。
pub fn launch_trace(n: usize, kind: Option<&str>) -> LaunchTrace {
    let (in_ring, total, capacity) = owl_cuda::trace::stats();
    let rows = owl_cuda::trace::recent(n, kind)
        .into_iter()
        .map(|e| LaunchRow {
            seq: e.seq,
            t_us: e.host_ns / 1000,
            cmd: e.cmd,
            stream: e.stream,
            name: e.name,
            grid: e.grid,
            block: e.block,
            smem: e.smem,
            bytes: e.bytes,
            ids: e.ids,
            out_id: e.out_id,
            ptrs: e.ptrs,
        })
        .collect();
    LaunchTrace {
        enabled: capacity > 0,
        capacity,
        in_ring,
        total,
        rows,
    }
}

/// **`/debug/vram` 的一次快照**（三面事实：store 账 / 块账房 / 句柄计数）。
///
/// 为什么三面同屋：它们必须**同一时刻**取（引擎不在 pump 时）—— 分开取会让
/// 「事件账 vs 块账房」的差集混入并发窗口的噪声，判据就不再是判据。
// 结构性豁免（2026-11-02 clippy 清理）：类型面 = 派发/表项签名，别名会挡住可读性
#[allow(clippy::type_complexity)]
pub struct DiagSnapshot {
    /// store 报告（三层账 + owner 维 + 事件账）
    pub report: owl_store::StoreReport,
    /// 块账房快照（逐 rank；`Err` = 后端不支持/读失败，如实带出，不静默）
    pub blocks: Vec<(u8, Result<Vec<BlockRow>, String>)>,
    /// 记账失败计数（非零 = 有事件没入账）
    pub hook_fail: u64,
    /// 窗未收次数（图窗失败且去向未定）
    pub window_unclosed: u64,
    /// 窗内未归还块数
    pub open_window_blocks: usize,
    /// 状态腿淘汰序预览（前 8 条；会话感知淘汰的可观测面）
    pub snap_evict: Vec<owl_store::SnapEvictPreview>,
    /// L1 会话资产投影（每会话的 tail 快照 / 消息边界；2026-11-12）
    pub sessions: Vec<crate::SessionAssets>,
    /// 默认池账（逐 rank）：(used, used_high, reserved, reserved_high)。
    /// 与块账之差 = 池缓存（已 free 仍占物理显存）。
    pub pools: Vec<(u8, Result<(u64, u64, u64, u64), String>)>,
    /// 合批事实（上限 / 已走批步数 / 批图是否在场）。
    pub batch: BatchFacts,
    /// 设备显存实况（`cuMemGetInfo`）：`(free, total)`。`Err` = 后端不支持。
    pub device: Result<(u64, u64), String>,
    /// **逐 rank 设备显存实况**（TP2 闭合账；`device` 是 rank0 兼容面）。
    pub devices: Vec<(u8, Result<(u64, u64), String>)>,
}

/// 合批事实（与 [`DiagSnapshot`] 同刻取）
pub struct BatchFacts {
    pub max_batch: usize,
    pub steps: u64,
    pub graph: bool,
    /// 准入容量快照（store 事实 + 判决输出；判"还能进几个"的现场）
    pub free_pages: usize,
    pub cache_pages: usize,
    pub reserved_pages: usize,
    pub usable_pages: usize,
    pub n_slots: usize,
    pub n_mem: usize,
    pub n_eff: usize,
    /// 排队快照（最多 16，按等待降序）：`(turn, session, waited_ms)`
    pub queued: Vec<(u64, u64, u64)>,
}

/// 灌流测量用的轻量事实（见 [`crate::service::Request::BatchStats`]）
pub struct BatchStatsSnapshot {
    pub max_batch: usize,
    pub batch_steps: u64,
    pub has_batch_graph: bool,
    pub queue_len: usize,
    /// 当前活跃 turn 数（**多活跃**：合批档 >1）
    pub active: usize,
}

/// **每拍队列事实**（原 `telemetry::QueueSnapshot`；泵循环写、HTTP 读，零等待）。
#[derive(Clone, Debug, Default)]
pub struct QueueFacts {
    pub queue_len: usize,
    pub active: usize,
    pub batch_steps: u64,
    pub max_batch: usize,
    pub has_graph: bool,
    /// `(turn, session, waited_ms)`，按等待降序，最多 16 条
    pub queued: Vec<(u64, u64, u64)>,
    /// 容量快照（降频更新；None = 尚未采到）
    pub capacity: Option<CapacityFacts>,
    /// 拒绝分桶（会话, 码, 次数；按次数降序，最多 16）
    pub rejects: Vec<(u64, String, u64)>,
    /// 排队时长直方图（`<50 / <200 / <1s / <5s / ≥5s` ms）
    pub wait_hist: [u64; 5],
}

impl QueueFacts {
    /// 从引擎取一份（容量面**只在需要时**采 —— 它要遍历前缀树）。
    pub fn sample<D: DeviceClient>(running: &RunningEngine<D>, capacity: Option<CapacityFacts>) -> Self {
        Self {
            queue_len: running.queue_len(),
            active: running.active_len(),
            batch_steps: running.batch_steps(),
            max_batch: running.max_batch(),
            has_graph: running.has_batch_graph(),
            queued: running.queued_snapshot(16),
            capacity,
            rejects: running.reject_snapshot(16),
            wait_hist: running.wait_hist(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CapacityFacts {
    pub free_pages: usize,
    pub cache_pages: usize,
    pub reserved_pages: usize,
    pub usable_pages: usize,
    pub n_slots: usize,
    pub n_mem: usize,
    pub n_eff: usize,
}

impl CapacityFacts {
    /// 采一次（要遍历前缀树 ⇒ 泵循环里**降频**调用）
    pub fn probe<D: DeviceClient>(running: &RunningEngine<D>) -> Self {
        let (f, c) = running.capacity_probe();
        Self {
            free_pages: f.free_pages,
            cache_pages: f.cache_pages,
            reserved_pages: f.reserved_pages,
            usable_pages: c.usable_pages,
            n_slots: c.n_slots,
            n_mem: c.n_mem,
            n_eff: c.n_eff,
        }
    }
}

/// **共享事实单元**（`Arc`；引擎线程写、HTTP 线程读）。
///
/// 锁内只做 clone/move —— 无 I/O；毒化只可能来自外部 panic，取回内层继续用
/// （长驻服务不硬退）。
#[derive(Default)]
pub struct FactsCell {
    cell: Mutex<QueueFacts>,
}

impl FactsCell {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// 写一份快照（泵循环里调用）
    pub fn publish(&self, snap: QueueFacts) {
        match self.cell.lock() {
            Ok(mut c) => *c = snap,
            Err(p) => *p.into_inner() = snap,
        }
    }

    /// 读一份快照（HTTP 侧；零等待）
    pub fn read(&self) -> QueueFacts {
        match self.cell.lock() {
            Ok(c) => c.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }
}

/// **一次诊断快照**（引擎线程内调用；零副作用）。
pub async fn snapshot<D: DeviceClient>(running: &mut RunningEngine<D>) -> DiagSnapshot {
    let (hook_fail, window_unclosed, open_window_blocks) = running.handle_probe();
    // 后端行 → 引擎投影行（服务面不认 `owl_iface`/`owl_cuda` 类型；P3）
    let blocks = running
        .block_ledger_probe()
        .await
        .into_iter()
        .map(|(rank, rows)| (rank, rows.map(|rs| rs.into_iter().map(BlockRow::from).collect())))
        .collect();
    let pools = running.pool_probe().await;
    let report = running.store_report();
    // **一次取全 rank**：`device` 只是 rank0 兼容面，不能再单独发一次
    // `mem_info`（TP2 下会重复读设备且可能拿到不同刻的数字）。
    let devices = running.mem_info_all().await;
    let device = devices
        .iter()
        .find(|(rank, _)| *rank == 0)
        .map(|(_, r)| r.clone())
        .unwrap_or_else(|| Err("mem_info: 无 rank0 读数".into()));
    let snap_evict = running.snap_evict_preview(8);
    let sessions = running.session_assets();
    let batch = {
        let (facts, cap) = running.capacity_probe();
        BatchFacts {
            max_batch: running.max_batch(),
            steps: running.batch_steps(),
            graph: running.has_batch_graph(),
            free_pages: facts.free_pages,
            cache_pages: facts.cache_pages,
            reserved_pages: facts.reserved_pages,
            usable_pages: cap.usable_pages,
            n_slots: cap.n_slots,
            n_mem: cap.n_mem,
            n_eff: cap.n_eff,
            queued: running.queued_snapshot(16),
        }
    };
    DiagSnapshot {
        report,
        blocks,
        hook_fail,
        window_unclosed,
        open_window_blocks,
        snap_evict,
        sessions,
        pools,
        batch,
        device,
        devices,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 读写往返：投什么读什么；空表 = 全零（不是 None/panic）。
    #[test]
    fn facts_cell_roundtrip() {
        let t = FactsCell::new();
        assert_eq!(t.read().queue_len, 0);
        t.publish(QueueFacts {
            queue_len: 3,
            active: 2,
            batch_steps: 17,
            max_batch: 4,
            has_graph: true,
            queued: vec![(9, 0xabc, 250)],
            rejects: vec![(0xabc, "no_memory".into(), 2)],
            wait_hist: [1, 2, 3, 4, 5],
            capacity: Some(CapacityFacts {
                free_pages: 100,
                cache_pages: 5,
                reserved_pages: 20,
                usable_pages: 77,
                n_slots: 4,
                n_mem: 3,
                n_eff: 3,
            }),
        });
        let s = t.read();
        assert_eq!((s.queue_len, s.active, s.batch_steps), (3, 2, 17));
        assert_eq!(s.queued, vec![(9, 0xabc, 250)]);
        assert_eq!(s.capacity.expect("cap").n_eff, 3);
        assert_eq!(s.rejects[0].2, 2);
        assert_eq!(s.wait_hist[4], 5);
    }
}
