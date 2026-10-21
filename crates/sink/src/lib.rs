//! 日志 / 计量**下沉口**(`LogSink`)—— 依赖图的**叶子 crate**,零依赖。
//!
//! # 为什么单独成 crate(而不是待在 `owl-shared::log`)
//!
//! 依赖图底部是 `owl-kernels`(`owl-iface → owl-kernels`,`owl-shared →
//! owl-iface`)⇒ **kernels 不能依赖 owl-shared**(cargo 实测:cyclic package
//! dependency)。但 kernels 也必须能出声(装配报告 / 自检 / 诊断)。
//!
//! 三条路:
//! 1. kernels 自建镜像结构 —— 同一契约两份定义,**禁止**(AGENTS.md:
//!    接口不唯一);
//! 2. 把 `LogSink` 塞进 kernels、让 shared 反过来引 —— 方向错(日志 ABI
//!    住进算子 crate);
//! 3. **本 crate**:纯 fn 指针值类型 + 级别常量,零依赖零状态,shared 与
//!    kernels 都依赖它 ⇒ 全仓只有一份 `LogSink`。
//!
//! # 定位
//!
//! 下层(kernels / 设备面)只持一个**可 Copy 的函数指针包**,由上层
//! (`owl-shared::log::sink()` 的现成接线 + GPU server 构造函数)显式注入。
//! 一个对象管三件事(都是「把事实抽出去」):
//!
//! - `log(level, tag, msg)` —— 事件流(`owl_shared::log`);
//! - `timer(tag, dur)` / `count(tag, n)` —— 数值账(`owl_shared::metrics`);
//! - `timing()` / `gpu_armed()` —— 采集是否值得起表(分级门)。
//!
//! 未注入(= [`LogSink::none`])时全部静默,且除一次判空外零开销。
//!
//! # 纪律
//!
//! - **状态在实现方,不在本 crate**:本 crate 没有 static、没有 OnceLock
//!   (「谁持有全局」这件事留给各层的单一写点显式声明);
//! - fn 指针不可捕获 ⇒ 实现方若需要上下文,只能走自己的进程级单例
//!   (这是 C 侧回调约束,不是本 crate 的选择);
//! - 准入(`enabled`)必须**先于** `format!` —— 否则静音档白分配。

#![forbid(unsafe_code)]

use std::time::Duration;

// ============================================================================
// §1 级别常量(与 `owl_shared::log::Level` 一一对应;这里只是 u8 面)
// ============================================================================
//
// 为什么常量住这里:下层(kernels)调 [`LogSink::logf`] 必须给级别,而它
// 拿不到 `owl_shared::log::Level`。常量 = 唯一事实,`Level::to_u8` 引用它们
// (shared 侧 re-export,不另立第二套数值)。

/// ERROR(最低,阈值语义:记录 `level <= threshold`)
pub const LVL_ERROR: u8 = 0;
/// WARN
pub const LVL_WARN: u8 = 1;
/// INFO
pub const LVL_INFO: u8 = 2;
/// DEBUG
pub const LVL_DEBUG: u8 = 3;
/// TRACE(最高)
pub const LVL_TRACE: u8 = 4;
/// 静音(阈值取本值 = 一条都不记)
pub const LVL_OFF: u8 = 9;

/// 级 → 名称(`level`/`off` 语义;报告面与日志前缀共用)
///
/// ⚠️ 名字带 `_of`,与 `owl_shared::log::level_name()`(读**当前 logger 阈值**)
/// 区分 —— 同名不同义会逼出一个别名 API(两套规范)。
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

// ============================================================================
// §2 LogSink
// ============================================================================

/// 日志 + 计时下沉口(值语义;fn 指针不可捕获 ⇒ 实现方只能走进程级单例)
///
/// `PartialEq` = **接线身份比较**(逐 fn 指针地址):用于"boot 后冻结"的
/// 重入判定(同值 = 幂等,异值 = 响亮拒)。手写实现是为了显式用地址比较
/// (`fn as usize`),绕开对 fn 指针直接用 `==` 的可预测性告警。
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
        // 不用 `==`(fn 指针比较有可预测性告警;地址身份是本处的确切语义)。
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

    /// 级/tag 准入(未接 enabled = 有 log 就放 —— 「没接就不打」的反面:
    /// 接线方未给判据时,由实现方自行决定静默)
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

    /// **设备侧采集是否武装**(`perf level = gpu`)。下游(如 CUPTI)在
    /// 构造期问一次即决定是否装载采集器 —— 默认 `host` 档 ⇒ false ⇒
    /// 采集器**根本不启动**(零开销)。
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

// ============================================================================
// §3 宏:下层的最短调用形态
// ============================================================================
//
// `sink.logf(LVL_INFO, "kernels", format_args!("…"))` 是对的但太长;下层
// 每个调用点都写一遍 `format_args!` 会让"要不要打"这件事反复出现。
//
// 契约与 `log_*!` 同款:**tag 是字面量**(准入先行,不过门时零分配)。

/// 记一行(显式 sink;准入先行)
#[macro_export]
macro_rules! sink_log {
    ($sink:expr, $lvl:expr, $tag:expr, $($arg:tt)*) => {
        $sink.logf($lvl, $tag, ::std::format_args!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
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

    /// 准入先于 format:判据拒 ⇒ 连实现都不进(下方 `SEEN` 计数不变)
    #[test]
    fn gate_precedes_emit() {
        let before = SEEN.load(std::sync::atomic::Ordering::Relaxed);
        let closed = LogSink::new(Some(rec_log), Some(never), None, None, None, None);
        closed.logf(LVL_WARN, "t", format_args!("拒"));
        assert_eq!(SEEN.load(std::sync::atomic::Ordering::Relaxed), before, "被拒的行不许落地");

        let open = LogSink::new(Some(rec_log), Some(yes), None, None, None, None);
        open.logf(LVL_WARN, "t", format_args!("放行"));
        assert_eq!(SEEN.load(std::sync::atomic::Ordering::Relaxed), before + 1);
        assert_eq!(LAST_LVL.load(std::sync::atomic::Ordering::Relaxed), LVL_WARN);
    }

    /// 两个分级门彼此独立:gpu 门不管宿主计时,反之亦然
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
