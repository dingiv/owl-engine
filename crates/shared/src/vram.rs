//! B6.5 显存硬预算治理(2026-10-10):设备侧总量落账 + live 分配账。
//!
//! 病:池容量此前纯手工拍(OWL_POOL_TOKENS),24G 贴顶场景反复 OOM/
//! 过保守二选一;实测 32768 fp8 池 = 23874MiB(97.2%)稳跑 —— 97% 是
//! 可用预算而非危险区。
//!
//! 机制:
//! - `total_set`:设备线程 boot 时 cuDeviceTotalMem 落账(一次);
//! - `live_add/live_sub`:server Alloc 成功/块释放全程记账(字节);
//! - 引擎 StatePool::alloc 读 `total/live` → 池容量 =
//!   min(手工上限, (target×total − live − reserve)/每 token 字节)。
//!
//! 全进程单引擎假设(与 slab_hint 同);非追踪面(CUDA ctx ~430MB、
//! async 池惰性保留、捕获瞬态)= reserve 兜底,默认 300MB,梯子校准。

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};

static TOTAL: AtomicU64 = AtomicU64::new(0);
static LIVE: AtomicI64 = AtomicI64::new(0);

/// 设备线程 boot:落账设备总显存(字节;cuDeviceTotalMem)。
pub fn total_set(bytes: u64) {
    TOTAL.store(bytes, Ordering::Relaxed);
}

/// 总显存(未落账 = 0,引擎侧视为禁用治理)。
pub fn total_bytes() -> u64 {
    TOTAL.load(Ordering::Relaxed)
}

/// live 分配账(alloc 成功 +,块释放 −;字节;可为负 = 释放未追踪初值)。
pub fn live_add(delta: i64) {
    LIVE.fetch_add(delta, Ordering::Relaxed);
}

/// 当前 live 字节。
pub fn live_bytes() -> u64 {
    LIVE.load(Ordering::Relaxed).max(0) as u64
}

static FREE: AtomicU64 = AtomicU64::new(0);

/// 设备线程:free 显存落账(cuMemGetInfo;boot 后每次调用覆盖)。
pub fn free_set(bytes: u64) {
    FREE.store(bytes, Ordering::Relaxed);
}

pub fn free_bytes() -> u64 {
    FREE.load(Ordering::Relaxed)
}
