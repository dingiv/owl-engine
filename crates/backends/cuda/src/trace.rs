//! **GPU server 发射账**(最近命令环形账;2026-10-14 立项)。
//!
//! # 定位
//!
//! server 是**全部设备动作的唯一咽喉**(launch / htod / dtoh / alloc / memset /
//! free / copy / graph —— 见 `command.rs` 的 `Command` 词表)。本模块在
//! `GpuServer::dispatch_inner` **派发之前**记一条,于是得到:
//!
//! * **单调序号**(`seq`):发射顺序 = host 视角的事实顺序(与 `[call]`
//!   解释器账不同 —— 那张账只到解释器,看不到 server 的搬运/回收/图);
//! * **流归属**(`stream`):`compute` / `h2d` / `d2h` —— 三流并发是引擎的
//!   设计前提,而"谁和谁之间没有序"正是出事时的第一问;
//! * **块 id 与设备地址**(`ids` / `ptrs`):地址复用是"跨流 UAF"类缺陷的
//!   唯一钥匙(仓内 2026-10-08 E5 跨流回收律 / 2026-10-03 FI 表渐进腐败案
//!   都栽在这里);
//! * **核名与几何**(`name` / `grid` / `block` / `smem`)。
//!
//! # 成本与开关
//!
//! `[cuda] trace_ring = <N>`(0 = 关,缺省):关时**逐命令零开销**(一个
//! `Option` 判定);开时一条 `Mutex` 入账 + 环形淘汰(容量 N)。
//! 容量建议 4096~65536(一次 27B prefill ≈ 数万条命令)。
//!
//! # 查询面
//!
//! * 进程内:[`recent`](取最近 N 条,可按 `cmd` 类过滤);
//! * HTTP:`GET /debug/launches[/<n>[/<kind>]]`(见 apps/server 的路由)。
//!
//! # 纪律
//!
//! 1. **记账失败不许影响执行**:本模块任何路径都不得 panic/阻塞主流程
//!    (锁中毒就地恢复;容量 0 = 关);
//! 2. **账是"发射序",不是"执行序"**:同流内发射序 = 执行序(CUDA 保证),
//!    跨流之间**无序** —— 账里两条相邻条目完全可能并发执行,这正是要看的;
//! 3. `host_ns` 是**host 单调时钟**(相对 server 建账),不是 GPU 时间;
//!    设备侧时间轴请用 CUPTI(见 perf-framework.md §十一)。

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// 一条发射账目(字段全部公开:查询面直接序列化)
#[derive(Clone, Debug)]
pub struct TraceEntry {
    /// 单调发射序号(1 起)
    pub seq: u64,
    /// host 单调时钟(相对建账,纳秒)
    pub host_ns: u64,
    /// 命令类(`Command::kind()` 的有界词表)
    pub cmd: &'static str,
    /// 流归属(compute / h2d / d2h / 空 = 与流无关)
    pub stream: &'static str,
    /// 核名(launch)或补充说明
    pub name: String,
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub smem: u32,
    /// 字节数(搬运/清零类)
    pub bytes: usize,
    /// 涉及块 id(顺序 = 实参顺序)
    pub ids: Vec<u64>,
    /// 块 id → 设备地址(账房解析;0 = 未解析)。**地址复用判据**。
    pub ptrs: Vec<u64>,
    /// 输出块 id(launch;契约:**最后一个 Block/BlockSlice 实参 = 输出**。
    /// 0 = 无/非 launch)。用它区分"读到被覆写的**输入**"与"写到一半的**输出**"。
    pub out_id: u64,
}

impl TraceEntry {
    /// 空模板(`seq`/`host_ns` 由账本填)
    pub fn new(cmd: &'static str, stream: &'static str) -> Self {
        Self {
            seq: 0,
            host_ns: 0,
            cmd,
            stream,
            name: String::new(),
            grid: (0, 0, 0),
            block: (0, 0, 0),
            smem: 0,
            bytes: 0,
            ids: Vec::new(),
            ptrs: Vec::new(),
            out_id: 0,
        }
    }

    pub fn name(mut self, n: impl Into<String>) -> Self {
        self.name = n.into();
        self
    }

    pub fn bytes(mut self, b: usize) -> Self {
        self.bytes = b;
        self
    }

    pub fn id(mut self, id: u64) -> Self {
        self.ids.push(id);
        self
    }

    pub fn ids(mut self, ids: impl IntoIterator<Item = u64>) -> Self {
        self.ids.extend(ids);
        self
    }

    pub fn ptrs(mut self, p: impl IntoIterator<Item = u64>) -> Self {
        self.ptrs.extend(p);
        self
    }

    pub fn grid(mut self, g: (u32, u32, u32), b: (u32, u32, u32), smem: u32) -> Self {
        self.grid = g;
        self.block = b;
        self.smem = smem;
        self
    }
}

/// 发射账(定容环形;**容量 0 = 关**)
pub struct TraceRing {
    cap: usize,
    start: Instant,
    seq: AtomicU64,
    /// 总发射数(不被环形淘汰影响;= 最后一次的 seq)
    total: AtomicU64,
    buf: Mutex<VecDeque<TraceEntry>>,
}

impl TraceRing {
    /// 建账(容量 = 保留最近多少条;0 = 关)
    pub fn new(cap: usize) -> Arc<Self> {
        Arc::new(Self {
            cap,
            start: Instant::now(),
            seq: AtomicU64::new(0),
            total: AtomicU64::new(0),
            buf: Mutex::new(VecDeque::with_capacity(cap.min(1 << 16))),
        })
    }

    pub fn enabled(&self) -> bool {
        self.cap > 0
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// 入账(容量 0 直接返回;锁中毒就地恢复 —— 记账不许拖垮执行)
    pub fn push(&self, mut e: TraceEntry) {
        if self.cap == 0 {
            return;
        }
        let seq = self.seq.fetch_add(1, Ordering::Relaxed) + 1;
        self.total.store(seq, Ordering::Relaxed);
        e.seq = seq;
        e.host_ns = self.start.elapsed().as_nanos() as u64;
        let mut b = self.buf.lock().unwrap_or_else(|p| p.into_inner());
        if b.len() == self.cap {
            b.pop_front();
        }
        b.push_back(e);
    }

    /// 最近 `n` 条(新→旧;`kind` = 只取该类,None = 全部)
    pub fn recent(&self, n: usize, kind: Option<&str>) -> Vec<TraceEntry> {
        let b = self.buf.lock().unwrap_or_else(|p| p.into_inner());
        let mut out: Vec<TraceEntry> = b
            .iter()
            .rev()
            .filter(|e| kind.is_none_or(|k| e.cmd == k || e.stream == k || e.name.contains(k)))
            .take(n)
            .cloned()
            .collect();
        out.reverse(); // 返回按发射序(旧→新),便于直接阅读
        out
    }

    /// (在账条数, 总发射数, 容量)
    pub fn stats(&self) -> (usize, u64, usize) {
        let b = self.buf.lock().unwrap_or_else(|p| p.into_inner());
        (b.len(), self.total.load(Ordering::Relaxed), self.cap)
    }

    pub fn clear(&self) {
        let mut b = self.buf.lock().unwrap_or_else(|p| p.into_inner());
        b.clear();
    }
}

// ============================================================================
// 全局发布面(server 建账 → app 直读;不占 DeviceClient 契约)
// ============================================================================

static GLOBAL: OnceLock<Arc<TraceRing>> = OnceLock::new();

/// server 侧建账后发布(重复发布 = 首个生效;进程内单设备档)
pub fn publish(ring: Arc<TraceRing>) {
    let _ = GLOBAL.set(ring);
}

/// 取账(None = 未建/未开)
pub fn get() -> Option<Arc<TraceRing>> {
    GLOBAL.get().cloned()
}

/// 最近 N 条(未开账 = 空表)
pub fn recent(n: usize, kind: Option<&str>) -> Vec<TraceEntry> {
    get().map(|r| r.recent(n, kind)).unwrap_or_default()
}

/// (在账, 总数, 容量);未开账 = (0,0,0)
pub fn stats() -> (usize, u64, usize) {
    get().map(|r| r.stats()).unwrap_or((0, 0, 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_keeps_last_n_in_order() {
        let r = TraceRing::new(3);
        for i in 1..=5u64 {
            r.push(TraceEntry::new("launch", "compute").name(format!("k{i}")).id(i));
        }
        let v = r.recent(10, None);
        assert_eq!(v.len(), 3, "定容 3 ⇒ 只留最后 3 条");
        assert_eq!(v.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![3, 4, 5], "按发射序返回");
        assert_eq!(v[0].name, "k3");
        let (n, total, cap) = r.stats();
        assert_eq!((n, total, cap), (3, 5, 3), "总发射数不被淘汰影响");
    }

    #[test]
    fn disabled_ring_is_free_and_queryable() {
        let r = TraceRing::new(0);
        assert!(!r.enabled());
        r.push(TraceEntry::new("launch", "compute"));
        assert!(r.recent(10, None).is_empty());
        assert_eq!(r.stats(), (0, 0, 0));
    }

    #[test]
    fn kind_filter_matches_cmd_stream_or_name() {
        let r = TraceRing::new(16);
        r.push(TraceEntry::new("launch", "compute").name("marlin_gemm_w4a16_awq"));
        r.push(TraceEntry::new("htod", "h2d").bytes(4096));
        r.push(TraceEntry::new("dtoh", "d2h").bytes(4));
        assert_eq!(r.recent(10, Some("htod")).len(), 1);
        assert_eq!(r.recent(10, Some("h2d")).len(), 1, "按流名过滤");
        assert_eq!(r.recent(10, Some("marlin")).len(), 1, "按核名子串过滤");
        assert_eq!(r.recent(10, None).len(), 3);
    }
}
