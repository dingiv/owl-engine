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
    pub blocks: Vec<(u8, Result<Vec<owl_iface::contract::BlockLedgerRow>, String>)>,
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
    let blocks = running.block_ledger_probe().await;
    let pools = running.pool_probe().await;
    let report = running.store_report();
    let device = running.mem_info().await;
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
