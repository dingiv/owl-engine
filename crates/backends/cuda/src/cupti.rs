//! CUPTI 进程内归因(**引擎自己分析自己**;docs/arch/perf-framework.md §十一)。
//!
//! # 为什么需要它
//!
//! decode / verify 走 **捕获图回放** ⇒ 解释器钩子结构性为零
//! (`interp.host.*` 0 节点)、`gpu_prof` 又与图互斥(它逐发射同步)—— 也就是说
//! **图里发生了什么,我们的两套仪器都看不见**。CUPTI(nsys 的内核)是唯一能
//! 在生产图态拿到**逐核 GPU 时间**、且零核改动的路:
//! 每条 `CUpti_ActivityKernel11` 记录自带 `graphId` / `graphNodeId`
//! ⇒ 归因维度 = `gpu.kern.<核名>`(全量)+ `gpu.graph.<gid>.<核名>`(图内逐节点)。
//!
//! # 形态
//!
//! ```text
//! CUPTI 驱动回调线程 ──(shim,只 memcpy)──► 无锁旁路缓冲 ──► 排空线程
//!                                                       └► sink.timer(...)  ← metrics store
//! ```
//!
//! 纪律(与项目 host 回调节律同款):**回调线程只投递**,不做日志/分配/
//! CUPTI 调用;排空在自有线程做,`try_lock` 拿不到就丢弃并计数(宁可丢样本,
//! 绝不阻塞驱动线程)。
//!
//! # 开销与开关
//!
//! 只在 **`perf level = gpu`** 时武装(`LogSink::gpu_armed`;
//! 默认 `host` ⇒ **根本不调 `owl_cupti_start`**,二进制路径不变)。
//! CUPTI 只开 `CUPTI_ACTIVITY_KIND_KERNEL` 一类,预期开销个位数百分比
//! (`level=off/host` vs `level=gpu` 同负载差分可测 —— 复用既有 `off` 基线臂)。
//!
//! # 已知边界(PoC)
//!
//! - 全局状态:FFI 回调签名无捕获 ⇒ 状态只能挂全局(C 侧约束,已文档化);
//! - 样本容量:metrics store 每 tag 4096 样本上限 ⇒ **看 `drop` 字段**,
//!   均值仍有效、计数会饱和(判读纪律见 perf-framework §10.3);
//! - 只做 KERNEL 类;memcpy/memset 与 GRAPH_TRACE 未接(需要时再开)。

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

/// 与 `native/cupti_shim.c` 的 `owl_cupti_kern` **逐字段同构**
/// (改一处必须改两处;shim 侧有 `memset(&e,0,…)` + `strncpy` 兜底)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KernRec {
    /// 核名内联(NUL 结尾)
    pub name: [u8; 64],
    pub start_ns: u64,
    pub end_ns: u64,
    /// 0 = 图外(直接发射);非 0 = 捕获图内的节点
    pub graph_node_id: u64,
    pub graph_id: u32,
    pub stream_id: u32,
}

extern "C" {
    fn owl_cupti_start(cb: extern "C" fn(*const KernRec)) -> i32;
    fn owl_cupti_stop();
    fn owl_cupti_flush();
    fn owl_cupti_last_error() -> *const std::os::raw::c_char;
}

/// 旁路缓冲容量(条);满了丢并计数
const CAP: usize = 1 << 16;
/// 排空周期
const DRAIN_MS: u64 = 50;

struct State {
    buf: Mutex<Vec<KernRec>>,
    sink: owl_shared::log::LogSink,
    stop: AtomicBool,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
    // ── 诊断计数(2026-10-21:**字段私有,随 State 走** —— 原来是 4 个
    //    模块级 `static`,与状态分家 ⇒ 无 State 时也能被写、`stats()` 与
    //    生命周期脱节)──
    /// 回调收到的记录总数
    total: AtomicU64,
    /// 旁路缓冲满/拿不到锁而丢弃的记录
    dropped: AtomicU64,
    /// 排空批次数
    batches: AtomicU64,
    /// 图内记录数(`graph_id != 0`)
    graph_recs: AtomicU64,
}

static STATE: OnceLock<State> = OnceLock::new();

/// CUPTI 驱动回调(只投递:memcpy 已由 shim 做完,这里只入缓冲)
extern "C" fn on_kernel(rec: *const KernRec) {
    if rec.is_null() {
        return;
    }
    // SAFETY:`KernRec` 是 `#[repr(C)]` + Copy 的 plain struct,shim 侧已填齐
    let rec = unsafe { *rec };
    // 未武装(回调不该发生,但 ABI 上可能)⇒ 连计数都不落,避免账目脱离生命周期
    let Some(st) = STATE.get() else { return };
    st.total.fetch_add(1, Ordering::Relaxed);
    if rec.graph_id != 0 {
        st.graph_recs.fetch_add(1, Ordering::Relaxed);
    }
    // 排空线程持锁期间 = 丢(绝不阻塞驱动线程);溢出同样丢
    match st.buf.try_lock() {
        Ok(mut v) if v.len() < CAP => v.push(rec),
        _ => {
            st.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn cstr(bytes: &[u8; 64]) -> &str {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(64);
    std::str::from_utf8(&bytes[..end]).unwrap_or("?")
}

fn drain_once(st: &State) {
    let batch = {
        let Ok(mut v) = st.buf.try_lock() else { return };
        if v.is_empty() {
            return;
        }
        std::mem::take(&mut *v)
    };
    st.batches.fetch_add(1, Ordering::Relaxed);
    for r in &batch {
        let name = cstr(&r.name);
        let dur = Duration::from_nanos(r.end_ns.saturating_sub(r.start_ns));
        st.sink.timer(&format!("gpu.kern.{name}"), dur);
        if r.graph_id != 0 {
            // 图内逐节点:唯一能在生产图态分辨"图里哪个核"的标签面
            st.sink
                .timer(&format!("gpu.graph.{}.{name}", r.graph_id), dur);
            st.sink
                .count(&format!("gpu.nodes.{}", r.graph_id), 1);
        }
    }
}

/// 武装(幂等;由 `GpuServer::run` 在 `perf level = gpu` 时调用)
pub fn start(sink: owl_shared::log::LogSink) -> Result<(), String> {
    if STATE.get().is_some() {
        return Ok(());
    }
    let st = State {
        buf: Mutex::new(Vec::with_capacity(4096)),
        sink,
        stop: AtomicBool::new(false),
        worker: Mutex::new(None),
        total: AtomicU64::new(0),
        dropped: AtomicU64::new(0),
        batches: AtomicU64::new(0),
        graph_recs: AtomicU64::new(0),
    };
    STATE
        .set(st)
        .map_err(|_| "cupti: 状态重复初始化".to_string())?;

    // 先起排空线程(避免首次回调时无人排空),再让 CUPTI 开始产记录
    let worker = std::thread::Builder::new()
        .name("owl-cupti-drain".into())
        .spawn(|| {
            while let Some(st) = STATE.get() {
                if st.stop.load(Ordering::Relaxed) {
                    break;
                }
                // **先 flush 再排空**:CUPTI 的 activity 记录只在内部缓冲满或
                // `cuptiActivityFlushAll` 时才经 `on_buffer_completed` 交付。
                // 早期 PoC 只在 stop 时 flush ⇒ 低体量负载(几十轮 decode ≈
                // 几 MB,不满 2 MiB 分片)永远拿不到样本(实测 0 行)。
                // flush 跑在专用线程,与驱动线程回调解耦(nsys 同款姿势)。
                unsafe { owl_cupti_flush() };
                drain_once(st);
                std::thread::sleep(Duration::from_millis(DRAIN_MS));
            }
            // 收尾:再 flush 一次拿残留,然后排空
            if STATE.get().is_some() {
                unsafe { owl_cupti_flush() };
            }
            if let Some(st) = STATE.get() {
                drain_once(st); // 收尾排空
            }
        })
        .map_err(|e| format!("cupti: 排空线程启动失败 {e}"))?;
    if let Some(st) = STATE.get() {
        *st.worker.lock().unwrap() = Some(worker);
    }

    let rc = unsafe { owl_cupti_start(on_kernel) };
    if rc != 0 {
        let msg = unsafe {
            let p = owl_cupti_last_error();
            if p.is_null() {
                format!("owl_cupti_start 返回 {rc}")
            } else {
                std::ffi::CStr::from_ptr(p).to_string_lossy().to_string()
            }
        };
        // 回滚:停线程(不留半武装态)
        if let Some(st) = STATE.get() {
            st.stop.store(true, Ordering::Relaxed);
        }
        return Err(msg);
    }
    Ok(())
}

/// 卸载(幂等):停 CUPTI → flush → 收尾排空 → join
pub fn stop() {
    let Some(st) = STATE.get() else { return };
    unsafe {
        owl_cupti_flush();
        owl_cupti_stop();
    }
    st.stop.store(true, Ordering::Relaxed);
    if let Some(h) = st.worker.lock().unwrap().take() {
        let _ = h.join();
    }
}

/// (总记录, 丢弃, 排空批数, 图内记录)。未武装 = 全零(账随 `State` 生灭)
pub fn stats() -> (u64, u64, u64, u64) {
    match STATE.get() {
        Some(st) => (
            st.total.load(Ordering::Relaxed),
            st.dropped.load(Ordering::Relaxed),
            st.batches.load(Ordering::Relaxed),
            st.graph_recs.load(Ordering::Relaxed),
        ),
        None => (0, 0, 0, 0),
    }
}

/// 是否已武装
pub fn running() -> bool {
    STATE.get().is_some()
}
