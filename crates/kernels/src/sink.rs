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
use std::time::Duration;

// ============================================================================
// §1 级别常量（与 `owl_shared::log::Level` 一一对应；这里只是 u8 面）
// ============================================================================
//
// **为什么常量住 kernels**（2026-10-29 起，独立 crate `owl-sink` 已合并入本文件）：下层（本 crate）
// 调 [`LogSink::logf`] 必须给级别，而它拿不到 `owl_shared::log::Level`。
// 常量 = 唯一事实，`Level::to_u8` 引用它们（shared 侧 re-export，不另立数值）。

/// ERROR（最低；阈值语义：记录 `level <= threshold`）
pub const LVL_ERROR: u8 = 0;
/// WARN
pub const LVL_WARN: u8 = 1;
/// INFO
pub const LVL_INFO: u8 = 2;
/// DEBUG
pub const LVL_DEBUG: u8 = 3;
/// TRACE（最高）
pub const LVL_TRACE: u8 = 4;
/// 静音（阈值取本值 = 一条都不记）
pub const LVL_OFF: u8 = 9;

/// 级 → 名称（报告面与日志前缀共用）。
///
/// ⚠️ 名字带 `_of`，与 `owl_shared::log::level_name()`（读**当前 logger 阈值**）
/// 区分 —— 同名不同义会逼出一个别名 API（两套规范）。
pub fn level_name_of(l: u8) -> &'static str {
    match l {
        LVL_ERROR => "error",
        LVL_WARN => "warn",
        LVL_INFO => "info",
        LVL_DEBUG => "debug",
        LVL_TRACE => "trace",
        _ => "off",
    }
}

/// 本 crate 内部的历史别名（`server::DiagTune::sink` 等字段用它）。
pub type Sink = LogSink;

// ============================================================================
// §2 LogSink：**依赖倒置口**
// ============================================================================
//
// 2026-10-29 合并决定（用户）:这个 ABI 属于「算子层要求的输出口」,所以定义住
// kernels;实现由**上层**给 —— `owl-shared`(能打日志的那层)把它的 logger 包成
// fn 指针值,经 `server::DiagTune{sink}` 在 boot 期注入。kernels 自己**不打日志、
// 不持全局 logger**,只持这个可 Copy 的包。
//
// 纪律(自原 `owl-sink` 卷,该 crate 已合并):
// - **状态在实现方**:本模块没有 logger 状态,只有注入来的函数指针;
// - fn 指针不可捕获 ⇒ 实现方若需要上下文,只能走自己的进程级单例;
// - 准入(`enabled`)必须**先于** `format!` —— 否则静音档白分配。
//
// `PartialEq` = **接线身份比较**(逐 fn 指针地址):用于"boot 后冻结"的重入判定
// (同值 = 幂等,异值 = 响亮拒)。
#[derive(Clone, Copy, Default)]
pub struct LogSink {
    log: Option<fn(u8, &str, &str)>,
    enabled: Option<fn(u8, &str) -> bool>,
    timer: Option<fn(&str, Duration)>,
    count: Option<fn(&str, u64)>,
    /// 设备侧采集是否武装(`perf level = gpu`)—— 由下游(如 CUPTI)在构造期问一次
    gpu: Option<fn() -> bool>,
    /// 宿主计时是否值得起表(`perf level >= host`)—— `off` 档不该白付
    /// `Instant::now()`。⚠️ 与 `gpu` 分开住:两者是不同分级门。
    timing: Option<fn() -> bool>,
}

impl PartialEq for LogSink {
    fn eq(&self, o: &Self) -> bool {
        // 逐 fn 指针**地址**比较(None = 未接线,只在双方都 None 时相等)。
        macro_rules! same {
            ($a:expr, $b:expr) => {
                match ($a, $b) {
                    (Some(x), Some(y)) => x as usize == y as usize,
                    (None, None) => true,
                    _ => false,
                }
            };
        }
        same!(self.log, o.log)
            && same!(self.enabled, o.enabled)
            && same!(self.timer, o.timer)
            && same!(self.count, o.count)
            && same!(self.gpu, o.gpu)
            && same!(self.timing, o.timing)
    }
}
impl Eq for LogSink {}

impl LogSink {
    /// 空口(全部丢弃;测试/无宿主场景)
    pub const fn none() -> Self {
        Self { log: None, enabled: None, timer: None, count: None, gpu: None, timing: None }
    }

    /// 显式组装(各面可单独给)
    pub const fn new(
        log: Option<fn(u8, &str, &str)>,
        enabled: Option<fn(u8, &str) -> bool>,
        timer: Option<fn(&str, Duration)>,
        count: Option<fn(&str, u64)>,
        gpu: Option<fn() -> bool>,
        timing: Option<fn() -> bool>,
    ) -> Self {
        Self { log, enabled, timer, count, gpu, timing }
    }

    /// 是否未接线(免无谓 format)
    #[inline]
    pub fn is_none(&self) -> bool {
        self.log.is_none() && self.timer.is_none() && self.count.is_none()
    }

    /// 级/tag 准入(未接 enabled = 有 log 就放)
    #[inline]
    pub fn enabled(&self, level: u8, tag: &str) -> bool {
        match self.enabled {
            Some(f) => f(level, tag),
            None => self.log.is_some(),
        }
    }

    /// 记一行(先准入再 format —— 不过门时零分配)
    #[inline]
    pub fn logf(&self, level: u8, tag: &str, args: std::fmt::Arguments<'_>) {
        if !self.enabled(level, tag) {
            return;
        }
        if let Some(f) = self.log {
            f(level, tag, &std::fmt::format(args));
        }
    }

    /// **宿主计时是否值得起表**(接线存在 + metrics 已到 HOST 级)
    #[inline]
    pub fn timing(&self) -> bool {
        match self.timing {
            Some(f) => self.timer.is_some() && f(),
            None => self.timer.is_some(),
        }
    }

    /// **设备侧采集是否武装**(`perf level = gpu`)
    #[inline]
    pub fn gpu_armed(&self) -> bool {
        match self.gpu {
            Some(f) => f(),
            None => false,
        }
    }

    /// 记一段耗时(落 metrics;tag 律 `<plane>.<class>[.<detail>]`)
    #[inline]
    pub fn timer(&self, tag: &str, d: Duration) {
        if let Some(f) = self.timer {
            f(tag, d);
        }
    }

    /// 计数
    #[inline]
    pub fn count(&self, tag: &str, n: u64) {
        if let Some(f) = self.count {
            f(tag, n);
        }
    }
}

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

/// `LogSink` 值语义回归（自原 `owl-sink` crate 搬入；该 crate 已于 2026-10-29 合并）。
#[cfg(test)]
mod value_tests {
    use super::*;

    static SEEN: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    static LAST_LVL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(255);

    fn rec_log(level: u8, _tag: &str, _msg: &str) {
        SEEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        LAST_LVL.store(level, std::sync::atomic::Ordering::Relaxed);
    }
    fn never(_level: u8, _tag: &str) -> bool {
        false
    }
    fn yes(_level: u8, _tag: &str) -> bool {
        true
    }
    fn timing_on() -> bool {
        true
    }
    fn gpu_on() -> bool {
        true
    }

    /// 空口:`none()` 恒静默、两个分级门恒 false(下层零开销的前提)
    #[test]
    fn none_is_silent_and_unarmed() {
        let s = LogSink::none();
        assert!(s.is_none());
        assert!(!s.enabled(LVL_ERROR, "any"));
        s.logf(LVL_ERROR, "any", format_args!("不应出现"));
        assert!(!s.timing(), "无 timer 接线 ⇒ 不起表");
        assert!(!s.gpu_armed());
    }

    /// 准入先于 format:判据拒 ⇒ 连实现都不进
    #[test]
    fn gate_precedes_emit() {
        let before = SEEN.load(std::sync::atomic::Ordering::Relaxed);
        let closed = LogSink::new(Some(rec_log), Some(never), None, None, None, None);
        closed.logf(LVL_WARN, "t", format_args!("拒"));
        assert_eq!(
            SEEN.load(std::sync::atomic::Ordering::Relaxed),
            before,
            "被拒的行不许落地"
        );

        let open = LogSink::new(Some(rec_log), Some(yes), None, None, None, None);
        open.logf(LVL_WARN, "t", format_args!("放行"));
        assert_eq!(SEEN.load(std::sync::atomic::Ordering::Relaxed), before + 1);
        assert_eq!(LAST_LVL.load(std::sync::atomic::Ordering::Relaxed), LVL_WARN);
    }

    /// 两个分级门彼此独立
    #[test]
    fn grade_gates_are_independent() {
        fn noop_timer(_: &str, _: Duration) {}
        let s = LogSink::new(None, None, Some(noop_timer), None, Some(gpu_on), None);
        assert!(s.gpu_armed(), "gpu 门接线即真");
        assert!(s.timing(), "未给 timing 判据 ⇒ 有 timer 即起表");
        let s = LogSink::new(None, None, None, None, None, Some(timing_on));
        assert!(!s.timing(), "无 timer 接线 ⇒ timing 恒 false(不许白起表)");
        assert!(!s.gpu_armed(), "未接 gpu 门 ⇒ false");
    }

    /// 级别常量 ↔ 名称(报告面 grep 锚点)
    #[test]
    fn level_names() {
        assert_eq!(level_name_of(LVL_WARN), "warn");
        assert_eq!(level_name_of(LVL_OFF), "off");
    }
}
