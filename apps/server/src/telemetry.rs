//! **遥测单元**（每拍更新；HTTP 侧零等待读取）。
//!
//! 为什么需要它：actor 是单线程且"泵到 Idle 才收件" ⇒ `/debug/vram` 这类
//! 走 actor 的查询在**忙时会被拖到整批跑完**才回（实测：3 个 turn 在跑时，
//! 200ms 后发的 vram 查询要等它们全结束）。队列/活跃/批步这些**每拍都在变**
//! 的事实必须在泵循环里顺手抄一份到共享单元，HTTP 读它即可（O(1)，不阻塞）。
//!
//! 纪律：
//! - 单元里只放**小尺寸快照**（无 Vec 增长；queued 截断 16 条）；
//! - 重活（块账房/三层报告）仍走 actor 的 `/debug/vram`（那是空闲期诊断）；
//! - 容量快照（要遍历前缀树算 movable）**降频**（每 32 拍一次）。

use std::sync::{Arc, Mutex};

/// 一拍抄一份的事实（小、可 Clone）
#[derive(Clone, Debug, Default)]
pub struct QueueSnapshot {
    pub queue_len: usize,
    pub active: usize,
    pub batch_steps: u64,
    pub max_batch: usize,
    pub has_graph: bool,
    /// (turn, session, waited_ms)，按等待降序，最多 16 条
    pub queued: Vec<(u64, u64, u64)>,
    /// 容量快照（降频更新；None = 尚未采到）
    pub capacity: Option<CapacitySnapshot>,
    /// 拒绝分桶（会话, 码, 次数；按次数降序，最多 16）
    pub rejects: Vec<(u64, String, u64)>,
    /// 排队时长直方图（`<50 / <200 / <1s / <5s / ≥5s` ms）
    pub wait_hist: [u64; 5],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CapacitySnapshot {
    pub free_pages: usize,
    pub cache_pages: usize,
    pub reserved_pages: usize,
    pub usable_pages: usize,
    pub n_slots: usize,
    pub n_mem: usize,
    pub n_eff: usize,
}

/// 共享单元（`Arc` 由 main 建、actor 写、HTTP 读）
#[derive(Default)]
pub struct Telemetry {
    cell: Mutex<QueueSnapshot>,
}

impl Telemetry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// 写一份快照（泵循环里调用；锁内只做 clone/move —— 无 I/O）
    pub fn publish(&self, snap: QueueSnapshot) {
        match self.cell.lock() {
            Ok(mut c) => *c = snap,
            // 锁毒化只可能来自外部 panic；取回内层继续用（长驻服务不硬退）
            Err(p) => *p.into_inner() = snap,
        }
    }

    /// 读一份快照（HTTP 侧；零等待）
    pub fn read(&self) -> QueueSnapshot {
        match self.cell.lock() {
            Ok(c) => c.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 读写往返：投什么读什么；空表 = 全零（不是 None/panic）。
    #[test]
    fn publish_read_roundtrip() {
        let t = Telemetry::new();
        assert_eq!(t.read().queue_len, 0);
        t.publish(QueueSnapshot {
            queue_len: 3,
            active: 2,
            batch_steps: 17,
            max_batch: 4,
            has_graph: true,
            queued: vec![(9, 0xabc, 250)],
            rejects: vec![(0xabc, "no_memory".into(), 2)],
            wait_hist: [1, 2, 3, 4, 5],
            capacity: Some(CapacitySnapshot {
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
