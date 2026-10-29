//! 下层的**唯一输出口**：kernels 面（Rust + C/C++）共用同一条 [`LogSink`]。
//!
//! # 为什么需要这一层（而不是直接 `eprintln!` / `fprintf`）
//!
//! kernels 是依赖图最底层（`owl-iface → owl-kernels`，而 `owl-shared →
//! owl-iface`）⇒ **不能依赖 `owl-shared`**（cargo: cyclic package dependency）。
//! 于是本 crate 只持一个**注入来的函数指针包**，由后端 boot 期经
//! [`crate::server::configure_diag`] 下发 —— 与既有的 `GdnDiag` 同一写点、
//! 同一纪律（零 env、boot 一次、之后冻结）。
//!
//! ```text
//! owl-cuda::GpuServer ──(DiagTune{sink})──► kernels::sink::configure
//!                                            │
//!   Rust 调用点 ──klog_debug!(...)───────────┤
//!   C/C++ 调用点 ──OWL_LOG_DEBUG(...)───────┘──► LogSink ──► owl_shared::log
//! ```
//!
//! # 三道门（与 `owl_shared::log` / `metrics` 逐条对齐）
//!
//! 1. **编译期**：cargo feature `log`（缺省开）。不开 ⇒ 本模块五个 `klog_*!`
//!    宏**空展开**（参数不求值），且 `build.rs` 不给 C++ 定义 `OWL_LOG_ENABLED`
//!    ⇒ 两侧同时逐字节零成本；
//! 2. **运行期分级 + tag**：`LogSink::enabled(level, tag)` = logger 阈值
//!    （`--log-level` / `/debug/log/level/<l>`）+ tag 白黑名单
//!    （`/debug/log/tags/a,b/-x`）。**C++ 与 Rust 走同一个判据**；
//! 3. **调用点**：先准入再 `format`（不过门时零分配）。
//!
//! # 全局槽的合法性（唯一一处）
//!
//! C ABI(`owl_log_on`/`owl_log_emit`)与 `FamilyGuard::drop` 都**拿不到上下文**
//! ⇒ 只能有一个进程级取值点。纪律：
//! - **唯一写点** = [`configure`]（boot 一次）；重入不同值 = 响亮 `Err`；
//! - 字段私有，读写只经 [`get`]；
//! - 不在热路径加状态：`get()` 是一次 `OnceLock` 读 + `Copy`。
//!
//! # 设备侧 printf **不在此列**
//!
//! `__device__` 里的 `printf` 走 CUDA printf 缓冲，无法回调宿主 ⇒ 只受
//! **编译期** `OWL_LOG_DEVICE` 约束（见 `cu/owl_log.h`），且默认关闭。

use std::sync::OnceLock;

// **词汇面住在接口层**(2026-10-29 反置):这里只 re-export + 实现全局槽。
pub use owl_iface::sink::{
    level_name_of, LogSink, Sink, LVL_DEBUG, LVL_ERROR, LVL_INFO, LVL_OFF, LVL_TRACE, LVL_WARN,
};

/// 注入失败原因（唯一 = 重入且值不同）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SinkError {
    /// boot 后冻结：二次下发不同值被拒（保留首个，调用方必须响亮处理）
    AlreadyConfigured,
}

impl std::fmt::Display for SinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyConfigured => {
                write!(f, "kernels::sink: 输出口已配置(boot 后冻结),二次下发不同值被拒")
            }
        }
    }
}
impl std::error::Error for SinkError {}

/// 唯一写点（后端 boot 期一次；重入同值 = Ok，异值 = `Err`）
pub fn configure(s: LogSink) -> Result<(), SinkError> {
    match SINK.set(s) {
        Ok(()) => Ok(()),
        Err(_) => {
            // 同值重入 = 幂等成功（多后端/多测试并行构造的常见形态）
            if SINK.get() == Some(&s) {
                Ok(())
            } else {
                Err(SinkError::AlreadyConfigured)
            }
        }
    }
}

/// 当前输出口（未注入 = [`LogSink::none`] ⇒ 全部静默）
#[inline]
pub fn get() -> LogSink {
    match SINK.get() {
        Some(s) => *s,
        None => LogSink::none(),
    }
}

/// 是否已注入（诊断面/测试判据；不代表"会打出来"，那是 `get().enabled(..)`）
#[inline]
pub fn configured() -> bool {
    SINK.get().is_some()
}

static SINK: OnceLock<LogSink> = OnceLock::new();

// ============================================================================
// §1 Rust 侧宏（编译期门：feature `log`）
// ============================================================================
//
// 双定义法（与 `owl_shared::log` 同款）：不开 feature 时宏体为空语句 ⇒
// 参数**不被求值**，下游 crate 也不需要写 `cfg(feature = "log")`。

#[doc(hidden)]
#[cfg(feature = "log")]
pub fn emit(level: u8, tag: &str, args: std::fmt::Arguments<'_>) {
    get().logf(level, tag, args);
}

#[doc(hidden)]
#[cfg(not(feature = "log"))]
pub fn emit(_level: u8, _tag: &str, _args: std::fmt::Arguments<'_>) {}

/// 记一行 ERROR（tag = 字面量）
#[cfg(feature = "log")]
#[macro_export]
macro_rules! klog_error {
    ($tag:expr, $($arg:tt)*) => {
        $crate::sink::emit($crate::sink::LVL_ERROR, $tag, ::std::format_args!($($arg)*))
    };
}
#[cfg(not(feature = "log"))]
#[macro_export]
macro_rules! klog_error {
    ($tag:expr, $($arg:tt)*) => {};
}

/// 记一行 WARN
#[cfg(feature = "log")]
#[macro_export]
macro_rules! klog_warn {
    ($tag:expr, $($arg:tt)*) => {
        $crate::sink::emit($crate::sink::LVL_WARN, $tag, ::std::format_args!($($arg)*))
    };
}
#[cfg(not(feature = "log"))]
#[macro_export]
macro_rules! klog_warn {
    ($tag:expr, $($arg:tt)*) => {};
}

/// 记一行 INFO（装配报告面）
#[cfg(feature = "log")]
#[macro_export]
macro_rules! klog_info {
    ($tag:expr, $($arg:tt)*) => {
        $crate::sink::emit($crate::sink::LVL_INFO, $tag, ::std::format_args!($($arg)*))
    };
}
#[cfg(not(feature = "log"))]
#[macro_export]
macro_rules! klog_info {
    ($tag:expr, $($arg:tt)*) => {};
}

/// 记一行 DEBUG（诊断/探针面 —— 产线靠 level=debug 或 tag 白名单放行）
#[cfg(feature = "log")]
#[macro_export]
macro_rules! klog_debug {
    ($tag:expr, $($arg:tt)*) => {
        $crate::sink::emit($crate::sink::LVL_DEBUG, $tag, ::std::format_args!($($arg)*))
    };
}
#[cfg(not(feature = "log"))]
#[macro_export]
macro_rules! klog_debug {
    ($tag:expr, $($arg:tt)*) => {};
}

/// 记一行 TRACE（逐次/逐发的细粒度；只应出现在可重复的小循环里）
#[cfg(feature = "log")]
#[macro_export]
macro_rules! klog_trace {
    ($tag:expr, $($arg:tt)*) => {
        $crate::sink::emit($crate::sink::LVL_TRACE, $tag, ::std::format_args!($($arg)*))
    };
}
#[cfg(not(feature = "log"))]
#[macro_export]
macro_rules! klog_trace {
    ($tag:expr, $($arg:tt)*) => {};
}

// ============================================================================
// §2 C ABI（C/C++ 侧唯一入口；由 `cu/owl_log.h` 的宏调用）
// ============================================================================
//
// 符号名带 `owl_log_` 前缀（全仓唯一，避免与其他 FFI 撞名）。
// 只读访问：不写任何状态（写点仍是 `configure`）。

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

/// C 侧传进来的级别归一(越界 = 按 TRACE 截断;不 panic —— 日志面不许崩)
fn clamp_level(level: u32) -> u8 {
    if level >= u32::from(LVL_TRACE) {
        LVL_TRACE
    } else {
        level as u8
    }
}

/// C 侧准入判定：`1` = 放行（调用方随后才格式化）
///
/// # Safety
/// `tag` 必须是 NUL 结尾的合法 C 字符串（C 侧传字面量）。
#[no_mangle]
pub unsafe extern "C" fn owl_log_on(level: u32, tag: *const c_char) -> c_int {
    if tag.is_null() {
        return 0;
    }
    // SAFETY: 调用契约要求 NUL 结尾；非法 UTF-8 走 lossy（日志面不许 panic）
    let tag = match CStr::from_ptr(tag).to_str() {
        Ok(t) => t,
        Err(_) => return 0,
    };
    i32::from(get().enabled(clamp_level(level), tag))
}

/// C 侧发射：`msg` 是 `len` 字节的 UTF-8 文本（C 侧已截断到缓冲区内）
///
/// # Safety
/// `tag` 同 [`owl_log_on`]；`msg` 必须指向 `len` 字节可读区间（可为空串）。
#[no_mangle]
pub unsafe extern "C" fn owl_log_emit(
    level: u32,
    tag: *const c_char,
    msg: *const c_char,
    len: usize,
) {
    if tag.is_null() || (msg.is_null() && len != 0) {
        return;
    }
    // SAFETY: 调用契约要求 tag 为 NUL 结尾 C 字符串
    let tag = match CStr::from_ptr(tag).to_str() {
        Ok(t) => t,
        Err(_) => return,
    };
    let body: &str = if len == 0 {
        ""
    } else {
        // SAFETY: 调用方保证 msg 指向 len 字节可读区间
        let bytes = std::slice::from_raw_parts(msg as *const u8, len);
        // 非法 UTF-8 不许丢行（丢行 = 静默失败）：占位替代
        std::str::from_utf8(bytes).unwrap_or("<非 UTF-8 日志体,已丢弃内容>")
    };
    get().logf(clamp_level(level), tag, format_args!("{body}"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// 进程级单槽 ⇒ 触碰它的测试一律经这把锁(不用测试执行顺序做假设)
    static SLOT: Mutex<()> = Mutex::new(());
    /// C ABI 发射面回执(最近一条消息体)
    static LAST: Mutex<String> = Mutex::new(String::new());

    fn rec_emit(_l: u8, _t: &str, msg: &str) {
        *LAST.lock().unwrap_or_else(|p| p.into_inner()) = msg.to_string();
    }
    fn yes(_l: u8, _t: &str) -> bool {
        true
    }
    fn timing_of() -> bool {
        true
    }

    /// **输出口全生命周期**:接线 → C ABI 准入/发射 → boot 后冻结(异值响亮拒)。
    ///
    /// 三件事必须在**同一个测试**里按序验证(进程级 `OnceLock`;拆开会互相
    /// 污染,得到与执行顺序相关的假绿/假红)。"未接线 = 恒静默"由本文件的 `LogSink`
    /// 的单测覆盖(`none_is_silent_and_unarmed`),不依赖本全局槽。
    #[test]
    fn sink_lifecycle_and_c_abi() {
        let _g = SLOT.lock().unwrap_or_else(|p| p.into_inner());
        // 用会记账的实现接线(若别处已接线则退化为只验幂等)
        let mine = LogSink::new(Some(rec_emit), Some(yes), None, None, None, None);
        let is_mine = match configure(mine) {
            Ok(()) => get() == mine,
            Err(_) => false,
        };
        assert!(configured(), "接线后 configured() 恒真");
        assert_eq!(configure(get()), Ok(()), "同值重入必须幂等");

        if is_mine {
            unsafe {
                assert_eq!(
                    owl_log_on(u32::from(LVL_INFO), c"t".as_ptr()),
                    1,
                    "接线 + 门放行 ⇒ C 侧准入为真"
                );
                let tag = c"t".as_ptr();
                owl_log_emit(u32::from(LVL_INFO), tag, c"hello".as_ptr(), 5);
                assert_eq!(LAST.lock().unwrap_or_else(|p| p.into_inner()).as_str(), "hello");
                // 空消息体(合法:C 侧格式化后为空)不许 panic
                owl_log_emit(u32::from(LVL_INFO), tag, std::ptr::null(), 0);
                // 空 tag = 拒(不 panic)
                assert_eq!(owl_log_on(u32::from(LVL_INFO), std::ptr::null()), 0);
            }
        }

        // 冻结:构造一个与该口不同的实现(timing 指针不同)
        let other = LogSink::new(Some(rec_emit), Some(yes), None, None, None, Some(timing_of));
        if get() != other {
            assert_eq!(
                configure(other),
                Err(SinkError::AlreadyConfigured),
                "异值重入必须响亮拒(不许静默覆盖)"
            );
        }
    }

    /// 常量与 C 头文件逐值对齐（`OWL_LVL_*` 与本文件 `LVL_*` 是同一事实；
    /// 头文件是手写的 ⇒ 机器判据防漂移）
    #[test]
    fn c_header_levels_match() {
        let hdr = include_str!("../cu/owl_log.h");
        for (name, val) in [
            ("OWL_LVL_ERROR", LVL_ERROR),
            ("OWL_LVL_WARN", LVL_WARN),
            ("OWL_LVL_INFO", LVL_INFO),
            ("OWL_LVL_DEBUG", LVL_DEBUG),
            ("OWL_LVL_TRACE", LVL_TRACE),
        ] {
            let pat = format!("#define {name} {val}u");
            assert!(hdr.contains(&pat), "cu/owl_log.h 缺 `{pat}`（Rust 与 C 级别漂移）");
        }
    }
}

