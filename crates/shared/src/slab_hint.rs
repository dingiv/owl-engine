//! 捕获 slab 定量(E8 工程债 → A1 解锁,2026-10-10)。
//!
//! 病:每图 `graph_begin` 预留**满额 slab**(`OWL_CAPTURE_SLAB_MB`,缺省
//! 64MiB),slab 经 carve 输出块被图持久引用 → 常驻。图族(verify +
//! fold×7 + propose ≈ 10 图)× 满额 = 0.6-1.3GiB 纯预留浪费,24G 贴顶
//! 场景(27B + draft)直接把 propose 图挤成 eager 降级(2026-10-10 A1
//! 复测实录:64/96/128MB 三档 carve 全崩,160MB slab 本体 OOM)。
//!
//! 修:**warmup 计量定量** —— GraphPlan 的 warmup(姿势 6 lite)eager
//! dry 一遍闭包,`meter_take()` 取闭包真分配字节 → hint = ×1.15 + 1MiB
//! → `graph_begin` 消费。mid-capture 分配非法(malloc 在捕获窗内非法,
//! 这正是 slab 预分配存在的原因),故只能预定量、不能边捕边长。
//!
//! - env 显式设 `OWL_CAPTURE_SLAB_MB` = 旧固定档(测试/诊断兼容,忽略 hint);
//! - env 缺省 = hint 定量(hint 缺席/为 0 → 缺省 64MiB 兜底);
//! - `CaptureState.cap` 记本图实际 cap,carve 校验对齐同一口(旧实现
//!   carve 校验读 env,与 hint 定量的 slab 脱口)。
//!
//! 全进程单引擎假设(actor 串行 boot;多引擎共驻非现行形态)。

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static METER: AtomicU64 = AtomicU64::new(0);
static HINT: AtomicUsize = AtomicUsize::new(0);

/// 设备侧真分配计量(server Alloc 处理器、非捕获窗分支调用;字节)。
pub fn meter_add(n_bytes: u64) {
    METER.fetch_add(n_bytes, Ordering::Relaxed);
}

/// 取走并清零累计计量字节(GraphPlan warmup 前弃一次旧账,后取真账)。
pub fn meter_take() -> u64 {
    METER.swap(0, Ordering::Relaxed)
}

/// 置本图 slab hint(字节;GraphPlan warmup 后、捕获前)。
pub fn set_hint(n_bytes: usize) {
    HINT.store(n_bytes, Ordering::Relaxed);
}

/// 消费 hint(graph_begin;None = 缺席)。
pub fn take_hint() -> Option<usize> {
    let v = HINT.swap(0, Ordering::Relaxed);
    if v == 0 { None } else { Some(v) }
}

/// 丢弃 hint(捕获失败/未请求捕获的收尾;防陈旧 hint 污染下一图)。
pub fn clear_hint() {
    HINT.store(0, Ordering::Relaxed);
}
