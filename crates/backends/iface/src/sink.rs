//! **日志/计量下沉口的词汇面**(`LogSink` 值类型 + 级别常量)。
//!
//! 2026-10-29 反置(依赖方向:`kernels → iface`):`LogSink` 是"下层要求上层
//! 提供的输出口"——**接口**住接口层(本文件),**实现**住 `owl-shared`
//! (`log::sink()` 把 logger/metrics 包成这个值),**全局槽 + C ABI + klog 宏**
//! 住 `owl-kernels::sink`(实现侧)。
//!
//! `owl_kernels::sink::{LogSink, LVL_*, level_name_of, Sink}` 保留为 re-export
//! 兼容层,旧路径不变。

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
