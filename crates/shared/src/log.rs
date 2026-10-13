//! owl logger 框架(2026-10-14 立项;与 [`crate::metrics`] **同风格、不同定位**)。
//!
//! # 两件事,别混
//!
//! | | [`crate::metrics`] | 本模块 |
//! |---|---|---|
//! | 语义 | **数值账**(计时/计数魔集) | **事件流**(离散、带级别) |
//! | 用途 | 性能判读(可聚合/可导出) | 排障、装配报告、异常回卷 |
//! | 形态 | 采样 → 聚合报告 | 一行一条,顺序不可丢 |
//!
//! # 三层机制(与 metrics 对齐)
//!
//! 1. **宏风格**:[`log_error!`](crate::log_error) / `log_warn!` / `log_info!` /
//!    `log_debug!` / `log_trace!(tag, fmt, args..)` —— 签名与 metrics 同款;
//! 2. **编译期门**:cargo feature `log`。**不开时五级宏全是空语句**
//!    (编译期剔除,零运行时,tag/参数表达式不求值)。双定义法 ⇒ 下游 crate
//!    零 `cfg` 样板,且不会触发 `unexpected_cfgs`;
//! 3. **运行期门**(store 内):级别阈值 + **tag 前缀白/黑名单** + tee 落盘
//!    + **按尺寸 rotation**。
//!
//! # 缺省级别 = WARN(用户裁决 2026-10-14)
//!
//! 开箱**只出 warn/error** —— 装配报告/装载进度等 info 面默认不输出
//! (需要时 `--log-level info` 或 `/debug/log/level/info`)。feature `log`
//! 本身入 `owl-shared` 的 `default`(默认编译进来),故「安静」由**级别**
//! 承担;要连编译都去掉则依赖侧 `default-features = false`。
//!
//! # 渲染格式
//!
//! ```text
//! [INFO] [boot] 引擎就绪(捕获态 Captured)
//! ```
//!
//! `[tag] msg` 部分与旧 `eprintln!("[boot] …")` **逐字相同** ⇒ 收编 = 机械
//! 替换,既有脚本/测试的 grep 不受影响;`[LEVEL]` 前缀是新增信息。
//! `timestamp = true` 时再前置 `2026-10-14T00:12:33.123Z `。
//!
//! # 落盘与 rotation
//!
//! `LogConfig.file` 给出路径时 tee 到文件;`max_bytes > 0` 时按尺寸切:
//! `<file>` → `<file>.1` → `<file>.2` … 最多 `keep` 份(超出丢最旧)。
//! 全部 I/O 走 [`crate::file_loader`](统一入口律)。

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ============================================================================
// §1 级别
// ============================================================================

/// 级别常量(宏桥参;ERROR 最低、TRACE 最高 —— 阈值语义:记录 level <= 阈值)
pub const LVL_ERROR: u8 = 0;
pub const LVL_WARN: u8 = 1;
pub const LVL_INFO: u8 = 2;
pub const LVL_DEBUG: u8 = 3;
pub const LVL_TRACE: u8 = 4;
/// 静音(阈值取本值 = 一条都不记)
pub const LVL_OFF: u8 = 9;

/// 日志级别
///
/// **缺省 = [`Level::Warn`]**(2026-10-14 用户裁决:开箱零刷屏,只保留
/// warn/error;装配报告/排障需显式 `--log-level info|debug|trace`)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Error,
    #[default]
    Warn,
    Info,
    Debug,
    Trace,
}

impl Level {
    pub fn to_u8(self) -> u8 {
        match self {
            Self::Error => LVL_ERROR,
            Self::Warn => LVL_WARN,
            Self::Info => LVL_INFO,
            Self::Debug => LVL_DEBUG,
            Self::Trace => LVL_TRACE,
        }
    }

    /// 渲染用的短名(不可变 —— 报告/日志面 grep 锚点)
    pub fn tag(self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warn => "WARN",
            Self::Info => "INFO",
            Self::Debug => "DEBUG",
            Self::Trace => "TRACE",
        }
    }

    /// 解析(`off` 亦是合法输入 = 静音)
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "error" => Self::Error,
            "warn" | "warning" => Self::Warn,
            "info" => Self::Info,
            "debug" => Self::Debug,
            "trace" => Self::Trace,
            _ => return None,
        })
    }

    /// 名称 ↔ u8(含 off)
    pub fn name_of(l: u8) -> &'static str {
        match l {
            LVL_ERROR => "error",
            LVL_WARN => "warn",
            LVL_INFO => "info",
            LVL_DEBUG => "debug",
            LVL_TRACE => "trace",
            _ => "off",
        }
    }

    /// 阈值解析(含 `off`)
    pub fn parse_threshold(s: &str) -> Option<u8> {
        if s.trim().eq_ignore_ascii_case("off") {
            return Some(LVL_OFF);
        }
        Self::parse(s).map(|l| l.to_u8())
    }
}

// ============================================================================
// §2 目标与配置
// ============================================================================

/// 输出去向
#[derive(Clone)]
pub enum Target {
    /// 仅 stderr(缺省)
    Stderr,
    /// 仅内存(测试断言/宿主方自取;不做 rotation)
    Memory(Arc<Mutex<Vec<String>>>),
    /// stderr + 文件(tee)
    Tee(PathBuf),
    /// 仅文件
    File(PathBuf),
}

impl std::fmt::Debug for Target {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stderr => write!(f, "Stderr"),
            Self::Memory(_) => write!(f, "Memory"),
            Self::Tee(p) => write!(f, "Tee({})", p.display()),
            Self::File(p) => write!(f, "File({})", p.display()),
        }
    }
}

/// logger 配置(入口侧组装;`[log]` 配置 module → 本结构)
#[derive(Clone, Debug)]
pub struct LogConfig {
    /// 阈值:记录 `level <= threshold`(缺省 INFO;`LVL_OFF` = 静音)
    pub level: u8,
    /// tag 前缀白名单(空 = 全放行)
    pub tags_allow: Vec<String>,
    /// tag 前缀黑名单(命中即拒;优先于白名单)
    pub tags_deny: Vec<String>,
    /// 输出去向
    pub target: Target,
    /// 单文件上限字节(0 = 不切)
    pub max_bytes: u64,
    /// rotation 保留份数
    pub keep: usize,
    /// 是否前置时间戳
    pub timestamp: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: LVL_WARN,
            tags_allow: Vec::new(),
            tags_deny: Vec::new(),
            target: Target::Stderr,
            max_bytes: 0,
            keep: 3,
            timestamp: false,
        }
    }
}

/// logger 错误
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LogError {
    /// 重复 init(保留首个)
    AlreadyInit,
    /// 落盘目标打不开
    Open(String),
}

impl std::fmt::Display for LogError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyInit => write!(f, "log: 全局 logger 已初始化(重复 init 被拒)"),
            Self::Open(p) => write!(f, "log: 落盘目标打不开: {p}"),
        }
    }
}
impl std::error::Error for LogError {}

// ============================================================================
// §3 logger
// ============================================================================

/// 写文件状态(max_bytes = 0 时不切断)
struct FileSink {
    path: PathBuf,
    file: std::fs::File,
    written: u64,
    max_bytes: u64,
    keep: usize,
}

impl FileSink {
    fn open(path: PathBuf, max_bytes: u64, keep: usize) -> Result<Self, LogError> {
        let file = crate::file_loader::append(&path)
            .map_err(|e| LogError::Open(format!("{}: {e}", path.display())))?;
        let written = crate::file_loader::metadata(&path).map(|m| m.len()).unwrap_or(0);
        Ok(Self { path, file, written, max_bytes, keep })
    }

    /// 按尺寸 rotation:`f` → `f.1` → `f.2`…(最多 keep 份,超出丢最旧)
    fn rotate(&mut self) -> Result<(), LogError> {
        let _ = self.file.flush();
        if self.keep == 0 {
            // 不留备份:直接截断重开
            self.file = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&self.path)
                .map_err(|e| LogError::Open(format!("{}: {e}", self.path.display())))?;
            self.written = 0;
            return Ok(());
        }
        for i in (1..self.keep).rev() {
            let from = self.backup(i);
            if crate::file_loader::exists(&from) {
                let _ = crate::file_loader::rename(&from, self.backup(i + 1));
            }
        }
        let _ = crate::file_loader::rename(&self.path, self.backup(1));
        self.file = crate::file_loader::append(&self.path)
            .map_err(|e| LogError::Open(format!("{}: {e}", self.path.display())))?;
        self.written = 0;
        Ok(())
    }

    fn backup(&self, i: usize) -> PathBuf {
        let mut s = self.path.clone().into_os_string();
        s.push(format!(".{i}"));
        PathBuf::from(s)
    }

    fn write_line(&mut self, line: &str) -> Result<(), LogError> {
        let n = line.len() as u64;
        if self.max_bytes > 0 && self.written + n > self.max_bytes {
            self.rotate()?;
        }
        self.file
            .write_all(line.as_bytes())
            .map_err(|e| LogError::Open(format!("{}: {e}", self.path.display())))?;
        self.written += n;
        Ok(())
    }
}

struct Inner {
    tags_allow: Vec<String>,
    tags_deny: Vec<String>,
    timestamp: bool,
    memory: Option<Arc<Mutex<Vec<String>>>>,
    to_stderr: bool,
    file: Option<FileSink>,
}

/// logger 本体(可与全局指针并存直接持有 —— 测试/多实例)
pub struct Logger {
    level: AtomicU8,
    inner: Mutex<Inner>,
}

impl Logger {
    /// 构造(文件目标打开失败 = Err;stderr/memory 恒成功)
    pub fn new(cfg: LogConfig) -> Result<Self, LogError> {
        let (to_stderr, memory, file_path) = match cfg.target {
            Target::Stderr => (true, None, None),
            Target::Memory(buf) => (false, Some(buf), None),
            Target::Tee(p) => (true, None, Some(p)),
            Target::File(p) => (false, None, Some(p)),
        };
        let file = match file_path {
            Some(p) => Some(FileSink::open(p, cfg.max_bytes, cfg.keep)?),
            None => None,
        };
        Ok(Self {
            level: AtomicU8::new(cfg.level),
            inner: Mutex::new(Inner {
                tags_allow: cfg.tags_allow,
                tags_deny: cfg.tags_deny,
                timestamp: cfg.timestamp,
                memory,
                to_stderr,
                file,
            }),
        })
    }

    pub fn level(&self) -> u8 {
        self.level.load(Ordering::Relaxed)
    }

    pub fn set_level(&self, l: u8) {
        self.level.store(l, Ordering::Relaxed);
    }

    /// 级别名(报告/HTTP 回执)
    pub fn level_name(&self) -> &'static str {
        Level::name_of(self.level())
    }

    /// 替换 tag 过滤(空 allow = 全放行;deny 优先)
    pub fn set_tags(&self, allow: Vec<String>, deny: Vec<String>) {
        let mut inner = self.inner.lock().unwrap();
        inner.tags_allow = allow;
        inner.tags_deny = deny;
    }

    /// 当前 tag 过滤(诊断)
    pub fn tags(&self) -> (Vec<String>, Vec<String>) {
        let inner = self.inner.lock().unwrap();
        (inner.tags_allow.clone(), inner.tags_deny.clone())
    }

    /// 准入判定:阈值 + tag 前缀白/黑名单
    #[inline]
    pub fn enabled(&self, level: u8, tag: &str) -> bool {
        let thr = self.level.load(Ordering::Relaxed);
        if thr == LVL_OFF || level > thr {
            return false;
        }
        let inner = self.inner.lock().unwrap();
        if inner.tags_deny.iter().any(|p| tag.starts_with(p.as_str())) {
            return false;
        }
        inner.tags_allow.is_empty()
            || inner.tags_allow.iter().any(|p| tag.starts_with(p.as_str()))
    }

    /// 产出一行(准入判定 + 渲染 + 分发;rotation 内聚)
    ///
    /// 准入在**本方法**内再判一次(不只靠 `__log` 预判):直接 API 与测试
    /// 同样受级别/tag 过滤约束 —— 两条入口同一道门。
    pub fn emit(&self, level: u8, tag: &str, msg: &str) {
        if !self.enabled(level, tag) {
            return;
        }
        let mut line = String::with_capacity(msg.len() + tag.len() + 16);
        {
            let inner = self.inner.lock().unwrap();
            if inner.timestamp {
                line.push_str(&iso_ms(SystemTime::now()));
                line.push(' ');
            }
            line.push_str(&format!("[{}] [{}] {}\n", level_tag(level), tag, msg));
            if let Some(buf) = &inner.memory {
                buf.lock().unwrap().push(line.clone());
            }
            if inner.to_stderr {
                let _ = std::io::stderr().write_all(line.as_bytes());
            }
        }
        // 文件单独持锁(不与内存/stderr 的长临界区混合)
        let mut inner = self.inner.lock().unwrap();
        if let Some(f) = inner.file.as_mut() {
            if let Err(e) = f.write_line(&line) {
                // 落盘失败不吞:降级到 stderr 一次(避免日志系统本身静默失联)
                static WARNED: std::sync::atomic::AtomicBool =
                    std::sync::atomic::AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    let _ = writeln!(std::io::stderr(), "[log] 落盘失败(后续同类不再报): {e}");
                }
            }
        }
    }
}

fn level_tag(l: u8) -> &'static str {
    match l {
        LVL_ERROR => "ERROR",
        LVL_WARN => "WARN",
        LVL_INFO => "INFO",
        LVL_DEBUG => "DEBUG",
        _ => "TRACE",
    }
}

/// `2026-10-14T00:12:33.123Z`(UTC;零依赖 —— 自算 civil date)
fn iso_ms(t: SystemTime) -> String {
    let d = t.duration_since(UNIX_EPOCH).unwrap_or_default();
    let (secs, ms) = (d.as_secs() as i64, d.subsec_millis());
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400);
    let (y, m, day) = civil_from_days(days);
    format!(
        "{y:04}-{m:02}-{day:02}T{:02}:{:02}:{:02}.{ms:03}Z",
        sod / 3600,
        (sod % 3600) / 60,
        sod % 60
    )
}

/// 天数(1970-01-01 为 0)→ (年, 月, 日);Howard Hinnant 民用历算法
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 }.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

// ============================================================================
// §4 全局指针 + 公共 API
// ============================================================================

static LOGGER: OnceLock<Logger> = OnceLock::new();

/// 本构建是否带日志面(宏展开与否)
#[cfg(feature = "log")]
pub const LOG_BUILD: bool = true;
#[cfg(not(feature = "log"))]
pub const LOG_BUILD: bool = false;

/// 程序早期初始化(重复 = [`LogError::AlreadyInit`],保留首个)
pub fn init_logger(cfg: LogConfig) -> Result<(), LogError> {
    LOGGER.set(Logger::new(cfg)?).map_err(|_| LogError::AlreadyInit)
}

/// 全局是否已初始化
pub fn logger_ready() -> bool {
    LOGGER.get().is_some()
}

/// 编程访问全局 logger
pub fn with_logger<R>(f: impl FnOnce(&Logger) -> R) -> Option<R> {
    LOGGER.get().map(f)
}

/// 运行期调级(HTTP `/debug/log/level/<name>` 与 CLI 共用)
pub fn set_level(l: u8) -> Option<&'static str> {
    LOGGER.get().map(|g| {
        g.set_level(l);
        g.level_name()
    })
}

/// 当前级别名(未 init = "uninit")
pub fn level_name() -> &'static str {
    LOGGER.get().map(|g| g.level_name()).unwrap_or("uninit")
}

/// 运行期改 tag 过滤
pub fn set_tags(allow: Vec<String>, deny: Vec<String>) -> bool {
    match LOGGER.get() {
        Some(g) => {
            g.set_tags(allow, deny);
            true
        }
        None => false,
    }
}

// ============================================================================
// §4b 下沉口(LogSink):**注入面**
// ============================================================================
//
// 为什么需要:设备面(`crates/backends/cuda` 的 GPU server)与算子面
// (`crates/kernels`,**lib 零依赖律** —— 不能依赖 owl-shared)都要能说话,
// 但不能反过来依赖日志/指标 crate。正解 = **注入**:这类下层只持一个
// **零状态可 Copy 的函数指针包**,由上层(GPU server 构造函数 / 引擎)
// 显式传入。
//
// 一个对象管三件事(都是「把事实抽出去」,不是新概念):
// - `log(level, tag, msg)` —— 事件流(本模块);
// - `timer(tag, dur)` / `count(tag, n)` —— 数值账(`crate::metrics`)。
//
// 未注入(= [`LogSink::none`])时全部静默且**零分支以外的开销**。

/// 日志 + 计时下沉口(值语义;fn 指针不可捕获 ⇒ 实现方只能走进程级单例)
#[derive(Clone, Copy, Default)]
pub struct LogSink {
    log: Option<fn(u8, &str, &str)>,
    enabled: Option<fn(u8, &str) -> bool>,
    timer: Option<fn(&str, Duration)>,
    count: Option<fn(&str, u64)>,
    /// 设备侧采集是否武装(`perf level = gpu`)—— 由下游(如 CUPTI)在构造期问一次
    gpu: Option<fn() -> bool>,
}

impl LogSink {
    /// 空口(全部丢弃;测试/无宿主场景)
    pub const fn none() -> Self {
        Self { log: None, enabled: None, timer: None, count: None, gpu: None }
    }

    /// 显式组装(各面可单独给)
    pub const fn new(
        log: Option<fn(u8, &str, &str)>,
        enabled: Option<fn(u8, &str) -> bool>,
        timer: Option<fn(&str, Duration)>,
        count: Option<fn(&str, u64)>,
        gpu: Option<fn() -> bool>,
    ) -> Self {
        Self { log, enabled, timer, count, gpu }
    }

    /// 是否未接线(免无谓 format)
    #[inline]
    pub fn is_none(&self) -> bool {
        self.log.is_none() && self.timer.is_none() && self.count.is_none()
    }

    /// 级/tag 准入(未接 enabled = 一律不过 —— 「没接就不打」)
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

    /// 计时**是否值得起表**(接线存在 + metrics 已到 HOST 级)
    /// —— `off` 档不该白付 `Instant::now()`
    #[inline]
    pub fn timing(&self) -> bool {
        self.timer.is_some()
            && crate::metrics::PERF_BUILD
            && crate::metrics::level() >= crate::metrics::level::HOST
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

fn sink_log(level: u8, tag: &str, msg: &str) {
    if let Some(g) = LOGGER.get() {
        g.emit(level, tag, msg);
    }
}

fn sink_enabled(level: u8, tag: &str) -> bool {
    LOGGER.get().is_some_and(|g| g.enabled(level, tag))
}

fn sink_timer(tag: &str, d: Duration) {
    crate::metrics::with_metrics_store(|s| s.timer_record_tag(tag, d, file!(), line!()));
}

fn sink_count(tag: &str, n: u64) {
    crate::metrics::with_metrics_store(|s| s.counter_add(tag, n, file!(), line!()));
}

/// 全局 logger + metrics 的现成接线(应用入口/引擎构造时取一次,显式下传)
pub fn sink() -> LogSink {
    LogSink::new(
        Some(sink_log),
        Some(sink_enabled),
        Some(sink_timer),
        Some(sink_count),
        Some(sink_gpu_armed),
    )
}

fn sink_gpu_armed() -> bool {
    crate::metrics::PERF_BUILD && crate::metrics::level() >= crate::metrics::level::GPU
}

// ============================================================================
// §5 宏(编译期分派:**双定义**,cfg 只在本 crate 求值)
// ============================================================================
// 不开 feature `log` ⇒ 宏展开为**空语句**:编译期剔除、零运行时、tag 与
// 参数表达式**不被求值**(与 metrics 的 `perf` 门同款纪律)。
// 双定义法 ⇒ 下游 crate 的 cfg 里不出现 `feature = "log"`(否则触发
// `unexpected_cfgs`)。

#[cfg(feature = "log")]
#[doc(hidden)]
pub fn __log(level: u8, tag: &str, args: std::fmt::Arguments<'_>) {
    if let Some(g) = LOGGER.get() {
        if g.enabled(level, tag) {
            g.emit(level, tag, &std::fmt::format(args));
        }
    }
}

#[cfg(not(feature = "log"))]
#[doc(hidden)]
pub fn __log(_level: u8, _tag: &str, _args: std::fmt::Arguments<'_>) {}

// 十份显式定义(不用内层生成宏:内层宏体里的 `$($arg:tt)*` 在生成层会被
// 当成外层重复符 —— Rust 无 `$$` 转义,故直接展开写)。

#[cfg(feature = "log")]
#[macro_export]
macro_rules! log_error {
    ($tag:expr, $($arg:tt)*) => {
        $crate::log::__log($crate::log::LVL_ERROR, $tag, format_args!($($arg)*))
    };
}
#[cfg(not(feature = "log"))]
#[macro_export]
macro_rules! log_error {
    ($tag:expr, $($arg:tt)*) => {
        // 无 feature `log`:编译期剔除 + 零运行时。`if false` 块保持
        // **词法使用**(变量不被判 unused、不产生告警),但被优化器整块消去
        // —— 参数表达式**不求值**。
        if false {
            let _ = ($tag, format_args!($($arg)*));
        }
    };
}

#[cfg(feature = "log")]
#[macro_export]
macro_rules! log_warn {
    ($tag:expr, $($arg:tt)*) => {
        $crate::log::__log($crate::log::LVL_WARN, $tag, format_args!($($arg)*))
    };
}
#[cfg(not(feature = "log"))]
#[macro_export]
macro_rules! log_warn {
    ($tag:expr, $($arg:tt)*) => {
        // 无 feature `log`:编译期剔除 + 零运行时。`if false` 块保持
        // **词法使用**(变量不被判 unused、不产生告警),但被优化器整块消去
        // —— 参数表达式**不求值**。
        if false {
            let _ = ($tag, format_args!($($arg)*));
        }
    };
}

#[cfg(feature = "log")]
#[macro_export]
macro_rules! log_info {
    ($tag:expr, $($arg:tt)*) => {
        $crate::log::__log($crate::log::LVL_INFO, $tag, format_args!($($arg)*))
    };
}
#[cfg(not(feature = "log"))]
#[macro_export]
macro_rules! log_info {
    ($tag:expr, $($arg:tt)*) => {
        // 无 feature `log`:编译期剔除 + 零运行时。`if false` 块保持
        // **词法使用**(变量不被判 unused、不产生告警),但被优化器整块消去
        // —— 参数表达式**不求值**。
        if false {
            let _ = ($tag, format_args!($($arg)*));
        }
    };
}

#[cfg(feature = "log")]
#[macro_export]
macro_rules! log_debug {
    ($tag:expr, $($arg:tt)*) => {
        $crate::log::__log($crate::log::LVL_DEBUG, $tag, format_args!($($arg)*))
    };
}
#[cfg(not(feature = "log"))]
#[macro_export]
macro_rules! log_debug {
    ($tag:expr, $($arg:tt)*) => {
        // 无 feature `log`:编译期剔除 + 零运行时。`if false` 块保持
        // **词法使用**(变量不被判 unused、不产生告警),但被优化器整块消去
        // —— 参数表达式**不求值**。
        if false {
            let _ = ($tag, format_args!($($arg)*));
        }
    };
}

#[cfg(feature = "log")]
#[macro_export]
macro_rules! log_trace {
    ($tag:expr, $($arg:tt)*) => {
        $crate::log::__log($crate::log::LVL_TRACE, $tag, format_args!($($arg)*))
    };
}
#[cfg(not(feature = "log"))]
#[macro_export]
macro_rules! log_trace {
    ($tag:expr, $($arg:tt)*) => {
        // 无 feature `log`:编译期剔除 + 零运行时。`if false` 块保持
        // **词法使用**(变量不被判 unused、不产生告警),但被优化器整块消去
        // —— 参数表达式**不求值**。
        if false {
            let _ = ($tag, format_args!($($arg)*));
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_cfg(level: u8) -> (LogConfig, Arc<Mutex<Vec<String>>>) {
        let buf = Arc::new(Mutex::new(Vec::new()));
        (
            LogConfig { level, target: Target::Memory(buf.clone()), ..Default::default() },
            buf,
        )
    }

    fn tmp(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "owl_log_{}_{}_{tag}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
        ))
    }

    #[test]
    fn level_threshold_filters() {
        let (cfg, buf) = mem_cfg(LVL_INFO);
        let l = Logger::new(cfg).expect("mem logger");
        l.emit(LVL_ERROR, "boot", "e");
        l.emit(LVL_INFO, "boot", "i");
        l.emit(LVL_DEBUG, "boot", "d");
        let got = buf.lock().unwrap().clone();
        assert_eq!(got.len(), 2, "INFO 阈值只放 ERROR/INFO:{got:?}");
        assert!(got[0].starts_with("[ERROR] [boot] e"));
        assert!(got[1].starts_with("[INFO] [boot] i"));
    }

    #[test]
    fn off_silences_everything() {
        let (cfg, buf) = mem_cfg(LVL_OFF);
        let l = Logger::new(cfg).expect("mem logger");
        for lv in [LVL_ERROR, LVL_WARN, LVL_INFO, LVL_DEBUG, LVL_TRACE] {
            l.emit(lv, "x", "should not appear");
        }
        assert!(buf.lock().unwrap().is_empty(), "off = 静音");
    }

    #[test]
    fn tag_prefix_allow_and_deny() {
        let (mut cfg, buf) = mem_cfg(LVL_INFO);
        cfg.tags_allow = vec!["boot".into(), "gl-prof".into()];
        cfg.tags_deny = vec!["gl-prof".into()];
        let l = Logger::new(cfg).expect("mem logger");
        l.emit(LVL_INFO, "boot", "yes");
        l.emit(LVL_INFO, "gl-prof", "denied");
        l.emit(LVL_INFO, "sched", "not allowed");
        let got = buf.lock().unwrap().clone();
        assert_eq!(got.len(), 1, "deny 优先于 allow:{got:?}");
        assert!(got[0].contains("[boot]"));
        // enabled 直接面
        assert!(l.enabled(LVL_INFO, "boot.x"));
        assert!(!l.enabled(LVL_INFO, "sched"));
        assert!(!l.enabled(LVL_DEBUG, "boot"));
    }

    #[test]
    fn tee_writes_file_and_rotates() {
        let p = tmp("rot.log");
        let cfg = LogConfig {
            level: LVL_INFO,
            target: Target::File(p.clone()),
            max_bytes: 80,
            keep: 2,
            ..Default::default()
        };
        let l = Logger::new(cfg).expect("file logger");
        for i in 0..20 {
            l.emit(LVL_INFO, "boot", &format!("line-{i:02}"));
        }
        let cur = crate::file_loader::read(&p).expect("read cur");
        assert!(!cur.is_empty(), "当前文件非空");
        // rotation 至少产生过 .1
        let b1 = PathBuf::from(format!("{}.1", p.display()));
        assert!(crate::file_loader::exists(&b1), "应产生备份 .1");
        // 单文件不超过上限(行小、80B 上限 ⇒ 允许超出最后一行)
        assert!(cur.len() as u64 <= 80 + 64, "当前文件尺寸 {} 应受 max_bytes 约束", cur.len());
        for f in [p.clone(), b1, PathBuf::from(format!("{}.2", p.display()))] {
            let _ = crate::file_loader::remove_file(&f);
        }
    }

    #[test]
    fn timestamp_prefix_is_iso() {
        let (mut cfg, buf) = mem_cfg(LVL_INFO);
        cfg.timestamp = true;
        let l = Logger::new(cfg).expect("mem logger");
        l.emit(LVL_INFO, "boot", "t");
        let line = buf.lock().unwrap()[0].clone();
        // 形如 2026-10-14T00:12:33.123Z [INFO] [boot] t
        assert!(line.len() > 24 && line.as_bytes()[4] == b'-' && line.contains("T"), "{line}");
        assert!(line.ends_with("[INFO] [boot] t\n"), "{line}");
    }

    #[test]
    fn civil_date_conversion() {
        // 1970-01-01 = day 0;2026-10-14 = 20740
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(20_740), (2026, 10, 14));
        // 闰日
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
    }

    #[test]
    #[cfg(feature = "log")]
    fn macros_route_to_global() {
        if !logger_ready() {
            let (cfg, _buf) = mem_cfg(LVL_TRACE);
            let _ = init_logger(cfg);
        }
        crate::log_info!("unit-test", "value={}", 42);
        crate::log_debug!("unit-test", "dbg");
        // 全局 logger 已 init ⇒ 至少不 panic;路径正确性由 Logger 测试覆盖
        assert!(logger_ready());
    }
}
