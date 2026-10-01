//! 统一性能 metrics 框架(2026-10-01 立项;合并散装计时刻度的落点 ——
//! 原 `OWL_STEP_PROFILE` 分相 eprintln、engine-bak layer_metrics A5 挂账
//! 的接线面)。
//!
//! # 四件核心 API
//!
//! 1. [`init_metrics`] —— 程序早期调用一次,把 store 挂上全局指针
//!    ([`OnceLock`];重复 init = [`MetricsError::AlreadyInit`]);
//! 2. [`timer_start!`](crate::timer_start) / [`timer_end!`](crate::timer_end)
//!    —— 任意两段代码间计时:**严格两两配对**(同 tag 同线程 LIFO 栈),
//!    不打印,样本落内存 store;样本带 **调用点位置**(`file!/line!/column!`
//!    在 macro_rules 展开点取值 = C 宏 `__FILE__/__LINE__` 同款语义);
//! 3. [`query_metrics`] —— 按 [`MetricsFilter`] 过滤、聚合、**打印**并返回
//!    [`TagReport`](`collect_metrics` = 纯收集不打印);
//! 4. 周边面:counter(`counter_inc!`/`counter_add!`)、事件(`event!`)、
//!    手动记录(`timer_record!`)、作用域糖(`timer_scope!`)、
//!    [`reset_metrics`] / [`metrics_ready`] / [`with_metrics_store`]。
//!
//! # 零开销纪律(编译期分派)
//!
//! 全部计时/计数宏在 **debug(`debug_assertions`)展开为实现调用,
//! release 展开为空语句** —— 零运行时开销,tag 表达式不被求值
//! (带副作用的 tag 表达式在 release 下不执行,契约上 tag 应为字面量)。
//! 位置参数由宏体内 `file!/line!/column!` 提供,调用方零样板。
//!
//! # 线程与配对契约
//!
//! - start/end **同线程配对**(按 (tag, thread) 分栈;跨线程 start→end
//!   是病态用法,不做支持);
//! - 同名 tag 可嵌套(LIFO 弹栈);
//! - 无配对 start 的 end 不 panic:计 orphan(`TagReport.orphan_ends`);
//! - 程序退出时的未闭合 start 计入 `TagReport.open_spans`。
//!
//! # 直用形态
//!
//! [`MetricsStore`] 的方法全部 pub:不经全局(如单测、多 store 并存)
//! 可直接持有 store 调用;宏 = 全局指针的快捷方式。

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant, SystemTime};

// ============================================================================
// §1 数据面:样本 / 聚合报告 / 过滤器
// ============================================================================

/// 单条计时样本(end 时刻定格)
#[derive(Clone, Debug)]
pub struct Sample {
    /// 跨度时长
    pub dur: Duration,
    /// end 时刻墙钟
    pub at: SystemTime,
    /// start 调用点文件(宏展开点)
    pub file: &'static str,
    /// start 调用点行号
    pub line: u32,
    /// 线程序号(进程内按首次出现分配,稳定可读)
    pub thread: u64,
}

/// 单 tag 聚合报告(query 的行)
#[derive(Clone, Debug, Default)]
pub struct TagReport {
    pub tag: String,
    /// 已配对样本数(timer)
    pub count: u64,
    pub total: Duration,
    pub min: Duration,
    pub max: Duration,
    pub avg: Duration,
    pub p50: Duration,
    pub p99: Duration,
    /// 未闭合 start(LIFO 栈深)
    pub open_spans: u64,
    /// 无配对 start 的 end 计数
    pub orphan_ends: u64,
    /// 因 store 容量上限被驱逐的样本数
    pub dropped: u64,
    /// 最近样本 end 时刻
    pub last_at: Option<SystemTime>,
    /// 最近样本 start 位置
    pub last_file: &'static str,
    pub last_line: u32,
    /// 计数器当前值(counter 面;timer tag 恒 0)
    pub counter: u64,
}

/// 排序键
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MetricsSort {
    /// 总耗时降序(热点视角;默认)
    #[default]
    TotalDesc,
    /// 次数降序
    CountDesc,
    /// 最近样本时间降序
    RecentDesc,
    /// tag 字典序
    NameAsc,
}

/// query 过滤器(全字段可选;零值 = 全量)
#[derive(Clone, Debug, Default)]
pub struct MetricsFilter {
    /// tag 精确匹配
    pub tag: Option<String>,
    /// tag 前缀
    pub tag_prefix: Option<String>,
    /// tag 子串
    pub tag_contains: Option<String>,
    /// 最小样本数(阈值以下不报)
    pub min_count: Option<u64>,
    /// 只统计该时刻之后的样本
    pub since: Option<SystemTime>,
    pub sort: MetricsSort,
    /// 返回条数上限
    pub limit: Option<usize>,
}

impl MetricsFilter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = Some(tag.into());
        self
    }
    pub fn tag_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.tag_prefix = Some(prefix.into());
        self
    }
    pub fn tag_contains(mut self, s: impl Into<String>) -> Self {
        self.tag_contains = Some(s.into());
        self
    }
    pub fn min_count(mut self, n: u64) -> Self {
        self.min_count = Some(n);
        self
    }
    pub fn since(mut self, t: SystemTime) -> Self {
        self.since = Some(t);
        self
    }
    pub fn sort(mut self, s: MetricsSort) -> Self {
        self.sort = s;
        self
    }
    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }
}

// ============================================================================
// §2 store
// ============================================================================

/// 同 key 开栈上限(病态使用的失控闸;超过 = start 被拒 + debug 告警)
const MAX_OPEN_PER_KEY: usize = 4096;

/// metrics 错误
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetricsError {
    /// 重复 init(全局指针已被占用,保留首个)
    AlreadyInit,
    /// 未 init 就查/记(宏路径静默忽略此错;直用 API 才会见到)
    NotReady,
    /// timer_end 无配对 start(已计 orphan;此错误仅直用 API 返回)
    OrphanEnd(String),
    /// 同 key 开栈超 [`MAX_OPEN_PER_KEY`](start 被拒)
    OpenLimit(String),
}

impl std::fmt::Display for MetricsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyInit => write!(f, "metrics: 全局 store 已初始化(重复 init 被拒)"),
            Self::NotReady => write!(f, "metrics: 未初始化(先 init_metrics)"),
            Self::OrphanEnd(tag) => write!(f, "metrics: timer_end 无配对 start(tag={tag})"),
            Self::OpenLimit(tag) => write!(f, "metrics: 同 key 开栈超限(tag={tag})"),
        }
    }
}
impl std::error::Error for MetricsError {}

#[derive(Default)]
struct Inner {
    /// 未配对 start 栈:key = (tag, thread 序号) → (Instant, file, line) 栈
    open: HashMap<(String, u64), VecDeque<(Instant, &'static str, u32)>>,
    /// 已配对样本:key = tag(容量 cap 环形驱逐)
    samples: HashMap<String, VecDeque<Sample>>,
    /// 计数器:key = (count, last 位置)
    counters: HashMap<String, (u64, &'static str, u32)>,
    /// 容量驱逐计数
    dropped: HashMap<String, u64>,
    /// orphan end 计数(按 tag)
    orphans: HashMap<String, u64>,
    /// 线程序号分配(ThreadId 无序号 → 首现序)
    threads: HashMap<ThreadId, u64>,
    next_thread: u64,
}

/// metrics 存储(内存态;线程安全 = 单把 Mutex —— 打点粒度微秒级,
/// 锁开销远小于被测段,不优化)
pub struct MetricsStore {
    inner: Mutex<Inner>,
    cap_per_tag: usize,
}

impl Default for MetricsStore {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsStore {
    /// 默认容量:每 tag 4096 样本(超出环形驱逐最旧)
    pub fn new() -> Self {
        Self::with_cap(4096)
    }

    pub fn with_cap(cap_per_tag: usize) -> Self {
        Self { inner: Mutex::new(Inner::default()), cap_per_tag }
    }

    fn thread_no(&self, inner: &mut Inner) -> u64 {
        let id = std::thread::current().id();
        let next = &mut inner.next_thread;
        *inner.threads.entry(id).or_insert_with(|| {
            let n = *next;
            *next += 1;
            n
        })
    }

    /// timer_start 实现start 入 LIFO 栈
    pub fn timer_begin(&self, tag: &str, file: &'static str, line: u32) {
        let mut inner = self.inner.lock().unwrap();
        let t = self.thread_no(&mut inner);
        let stack = inner.open.entry((tag.to_string(), t)).or_default();
        if stack.len() >= MAX_OPEN_PER_KEY {
            #[cfg(debug_assertions)]
            eprintln!("[metrics] 同 key 开栈超限(>{MAX_OPEN_PER_KEY}),start 被拒: {tag}");
            return;
        }
        stack.push_back((Instant::now(), file, line));
    }

    /// timer_end 实现:弹栈配对 → 样本落库
    pub fn timer_end(&self, tag: &str, file: &'static str, line: u32) -> Result<(), MetricsError> {
        let now = Instant::now();
        let at = SystemTime::now();
        let mut inner = self.inner.lock().unwrap();
        let t = self.thread_no(&mut inner);
        let Some((start, sfile, sline)) = inner.open.get_mut(&(tag.to_string(), t)).and_then(
            |stack| stack.pop_back(),
        ) else {
            #[cfg(debug_assertions)]
            eprintln!("[metrics] timer_end 无配对 start: {tag} @ {file}:{line}");
            *inner.orphans.entry(tag.to_string()).or_default() += 1;
            return Err(MetricsError::OrphanEnd(tag.to_string()));
        };
        let _ = (file, line); // end 位置并入诊断可在需要时记录;样本带 start 位置
        let sample = Sample {
            dur: now.saturating_duration_since(start),
            at,
            file: sfile,
            line: sline,
            thread: t,
        };
        // 驱逐判定与落库分相(避免 inner 双可变借)
        let evict = {
            let dq = inner.samples.entry(tag.to_string()).or_default();
            if dq.len() >= self.cap_per_tag {
                dq.pop_front();
                true
            } else {
                false
            }
        };
        if evict {
            *inner.dropped.entry(tag.to_string()).or_default() += 1;
        }
        inner.samples.entry(tag.to_string()).or_default().push_back(sample);
        Ok(())
    }

    /// 手动记录一次耗时(等价一对 start/end 的样本)
    pub fn timer_record(&self, tag: &str, dur: Duration, file: &'static str, line: u32) {
        let mut inner = self.inner.lock().unwrap();
        let t = self.thread_no(&mut inner);
        let evict = {
            let dq = inner.samples.entry(tag.to_string()).or_default();
            if dq.len() >= self.cap_per_tag {
                dq.pop_front();
                true
            } else {
                false
            }
        };
        if evict {
            *inner.dropped.entry(tag.to_string()).or_default() += 1;
        }
        inner
            .samples
            .entry(tag.to_string())
            .or_default()
            .push_back(Sample { dur, at: SystemTime::now(), file, line, thread: t });
    }

    /// timer_record 的动态 tag 形态(直用 API;宏路径 tag 恒字面量)
    pub fn timer_record_tag(&self, tag: &str, dur: Duration, file: &'static str, line: u32) {
        self.timer_record(tag, dur, file, line)
    }

    /// 计数器 += n
    pub fn counter_add(&self, tag: &str, n: u64, file: &'static str, line: u32) {
        let mut inner = self.inner.lock().unwrap();
        let e = inner.counters.entry(tag.to_string()).or_insert((0, file, line));
        e.0 += n;
        e.1 = file;
        e.2 = line;
    }

    /// 聚合 + 过滤(query 数据面)
    pub fn collect(&self, filter: &MetricsFilter) -> Vec<TagReport> {
        let inner = self.inner.lock().unwrap();
        let mut tags: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
        for t in inner.samples.keys() {
            tags.insert(t.as_str());
        }
        for t in inner.counters.keys() {
            tags.insert(t.as_str());
        }
        for t in inner.orphans.keys() {
            tags.insert(t.as_str());
        }
        let mut reports: Vec<TagReport> = tags
            .into_iter()
            .map(|tag| self.report_one(&inner, tag, filter))
            // 全零 tag 不报(零样本纯 open 的除外 —— open_spans 是信号)
            .filter(|r| r.count > 0 || r.counter > 0 || r.open_spans > 0 || r.orphan_ends > 0)
            .filter(|r| match filter.min_count {
                Some(m) => r.count >= m,
                None => true,
            })
            .filter(|r| match (
                filter.tag.as_deref(),
                filter.tag_prefix.as_deref(),
                filter.tag_contains.as_deref(),
            ) {
                (Some(t), _, _) => r.tag == t,
                (_, Some(p), _) => r.tag.starts_with(p),
                (_, _, Some(c)) => r.tag.contains(c),
                _ => true,
            })
            .collect();
        match filter.sort {
            MetricsSort::TotalDesc => reports.sort_by(|a, b| b.total.cmp(&a.total)),
            MetricsSort::CountDesc => reports.sort_by(|a, b| b.count.cmp(&a.count)),
            MetricsSort::RecentDesc => reports.sort_by(|a, b| b.last_at.cmp(&a.last_at)),
            MetricsSort::NameAsc => reports.sort_by(|a, b| a.tag.cmp(&b.tag)),
        }
        if let Some(l) = filter.limit {
            reports.truncate(l);
        }
        reports
    }

    /// 单 tag 聚合(锁内调用)
    fn report_one(&self, inner: &Inner, tag: &str, filter: &MetricsFilter) -> TagReport {
        let mut r = TagReport { tag: tag.to_string(), ..Default::default() };
        if let Some(dq) = inner.samples.get(tag) {
            let mut durs: Vec<Duration> = dq
                .iter()
                .filter(|s| filter.since.map_or(true, |t| s.at >= t))
                .map(|s| s.dur)
                .collect();
            r.count = durs.len() as u64;
            if !durs.is_empty() {
                durs.sort();
                r.min = durs[0];
                r.max = durs[durs.len() - 1];
                r.total = durs.iter().fold(Duration::ZERO, |a, d| a + *d);
                r.avg = r.total / (durs.len() as u32);
                r.p50 = durs[(durs.len() as f64 * 0.50) as usize % durs.len()];
                r.p99 = durs[(durs.len() as f64 * 0.99) as usize % durs.len()];
            }
            if let Some(s) = dq.back() {
                r.last_at = Some(s.at);
                r.last_file = s.file;
                r.last_line = s.line;
            }
        }
        if let Some((c, f, l)) = inner.counters.get(tag) {
            r.counter = *c;
            if r.last_at.is_none() {
                r.last_file = f;
                r.last_line = *l;
            }
        }
        // open spans 跨线程累加(同 tag 多线程各自栈)
        for ((t, _), st) in inner.open.iter() {
            if t == tag {
                r.open_spans += st.len() as u64;
            }
        }
        r.orphan_ends = inner.orphans.get(tag).copied().unwrap_or(0);
        r.dropped = inner.dropped.get(tag).copied().unwrap_or(0);
        r
    }

    /// 清空全部记录(store 本体保留;全局指针不动)
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.open.clear();
        inner.samples.clear();
        inner.counters.clear();
        inner.dropped.clear();
        inner.orphans.clear();
    }
}

// ============================================================================
// §3 全局指针 + 公共 API
// ============================================================================

static STORE: OnceLock<MetricsStore> = OnceLock::new();

/// 程序早期初始化:把 store 挂上全局指针(幂等失败 = [`MetricsError::AlreadyInit`],
/// 保留首个 —— 引擎/server 单 store 契约)。
pub fn init_metrics(store: MetricsStore) -> Result<(), MetricsError> {
    STORE.set(store).map_err(|_| MetricsError::AlreadyInit)
}

/// 全局是否已初始化(宏路径的前置事实;未 init 时宏全部静默)
pub fn metrics_ready() -> bool {
    STORE.get().is_some()
}

/// 编程访问全局 store(直用 API 面)
pub fn with_metrics_store<R>(f: impl FnOnce(&MetricsStore) -> R) -> Option<R> {
    STORE.get().map(f)
}

/// 收集符合过滤条件的聚合报告(不打印)
pub fn collect_metrics(filter: &MetricsFilter) -> Vec<TagReport> {
    STORE.get().map(|s| s.collect(filter)).unwrap_or_default()
}

/// query_metrics:收集 + **打印**符合条件的信息,返回报告(程序化消费)
pub fn query_metrics(filter: &MetricsFilter) -> Vec<TagReport> {
    let reports = collect_metrics(filter);
    if reports.is_empty() {
        eprintln!("[metrics] 无匹配记录(filter: tag={:?} prefix={:?} contains={:?})",
            filter.tag, filter.tag_prefix, filter.tag_contains);
        return reports;
    }
    eprintln!(
        "[metrics] {:<32} {:>7} {:>11} {:>10} {:>10} {:>10} {:>10} {:>5} {:>6} {:>6}  {}",
        "tag", "count", "total", "avg", "p50", "p99", "max", "open", "orph", "drop", "last@"
    );
    for r in &reports {
        let (f, l) = (r.last_file.rsplit('/').next().unwrap_or(r.last_file), r.last_line);
        eprintln!(
            "[metrics] {:<32} {:>7} {:>11} {:>10} {:>10} {:>10} {:>10} {:>5} {:>6} {:>6}  {}:{}",
            truncate32(&r.tag),
            r.count,
            fmt_dur(r.total),
            fmt_dur(r.avg),
            fmt_dur(r.p50),
            fmt_dur(r.p99),
            fmt_dur(r.max),
            r.open_spans,
            r.orphan_ends,
            r.dropped,
            f,
            l
        );
        if r.counter > 0 {
            eprintln!("[metrics] {:<32} counter = {}", "", r.counter);
        }
    }
    reports
}

/// 全量清空(全局)
pub fn reset_metrics() {
    if let Some(s) = STORE.get() {
        s.reset();
    }
}

fn truncate32(s: &str) -> String {
    if s.chars().count() <= 32 { s.to_string() } else { format!("{}…", s.chars().take(31).collect::<String>()) }
}

fn fmt_dur(d: Duration) -> String {
    let us = d.as_nanos() as f64 / 1e3;
    if us < 1_000.0 { format!("{us:.1}µs") }
    else if us < 1e6 { format!("{:.2}ms", us / 1e3) }
    else { format!("{:.2}s", us / 1e6) }
}

// ============================================================================
// §4 宏(release 空展开)
// ============================================================================
// 展开纪律:debug(debug_assertions)→ __timer_start/end 等实现调用,
// file!/line!/column! 在宏调用点取值(C __FILE__/__LINE__ 同语义);
// release → cfg 剔除整个语句,零运行时,tag 表达式不被求值。

/// 计时段开始(与 [`timer_end!`] 同 tag 同线程两两配对;LIFO 支持嵌套)
#[macro_export]
macro_rules! timer_start {
    ($tag:expr) => {
        #[cfg(debug_assertions)]
        $crate::__timer_start($tag, file!(), line!(), column!());
    };
}

/// 计时段结束(与同 tag 的最近 [`timer_start!`] 配对,样本落 store)
#[macro_export]
macro_rules! timer_end {
    ($tag:expr) => {
        #[cfg(debug_assertions)]
        $crate::__timer_end($tag, file!(), line!(), column!());
    };
}

/// 作用域糖:`timer_scope!("x", { ... })` = start + 块 + end(保留块值语义)
#[macro_export]
macro_rules! timer_scope {
    ($tag:expr, $body:expr) => {{
        #[cfg(debug_assertions)]
        $crate::__timer_start($tag, file!(), line!(), column!());
        let __owl_m_val = $body;
        #[cfg(debug_assertions)]
        $crate::__timer_end($tag, file!(), line!(), column!());
        __owl_m_val
    }};
}

/// 计数器 +1
#[macro_export]
macro_rules! counter_inc {
    ($tag:expr) => {
        #[cfg(debug_assertions)]
        $crate::__counter_add($tag, 1, file!(), line!());
    };
}

/// 计数器 += n
#[macro_export]
macro_rules! counter_add {
    ($tag:expr, $n:expr) => {
        #[cfg(debug_assertions)]
        $crate::__counter_add($tag, $n, file!(), line!());
    };
}

/// 一次性事件计数(counter 的语义糖)
#[macro_export]
macro_rules! event {
    ($tag:expr) => {
        #[cfg(debug_assertions)]
        $crate::__counter_add($tag, 1, file!(), line!());
    };
}

/// 手动记录一次已知耗时(如外部 Instant 差值灌入)
#[macro_export]
macro_rules! timer_record {
    ($tag:expr, $dur:expr) => {
        #[cfg(debug_assertions)]
        $crate::__timer_record($tag, $dur, file!(), line!());
    };
}

// ============================================================================
// §5 宏实现桥(全局未 init 时全部静默;non-debug 编译产物不引用)
// ============================================================================

#[doc(hidden)]
#[cfg(debug_assertions)]
pub fn __timer_start(tag: &'static str, file: &'static str, line: u32, _col: u32) {
    if let Some(s) = STORE.get() {
        s.timer_begin(tag, file, line);
    }
}

#[doc(hidden)]
#[cfg(debug_assertions)]
pub fn __timer_end(tag: &'static str, file: &'static str, line: u32, _col: u32) {
    if let Some(s) = STORE.get() {
        let _ = s.timer_end(tag, file, line);
    }
}

#[doc(hidden)]
#[cfg(debug_assertions)]
pub fn __counter_add(tag: &'static str, n: u64, file: &'static str, line: u32) {
    if let Some(s) = STORE.get() {
        s.counter_add(tag, n, file, line);
    }
}

#[doc(hidden)]
#[cfg(debug_assertions)]
pub fn __timer_record(tag: &'static str, dur: Duration, file: &'static str, line: u32) {
    if let Some(s) = STORE.get() {
        s.timer_record(tag, dur, file, line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// 进程级全局 store 的幂等装配(测试间共享 OnceLock;竞态下
    /// 首个成功者定义 store,其余 AlreadyInit 被忽略)
    fn ensure_init() {
        let _ = init_metrics(MetricsStore::new());
        assert!(metrics_ready());
        // 幂等断言:再 init 必 AlreadyInit(任何测试顺序下都成立)
        assert_eq!(init_metrics(MetricsStore::new()), Err(MetricsError::AlreadyInit));
    }

    fn dur_ms(d: Duration) -> f64 {
        d.as_nanos() as f64 / 1e6
    }

    #[test]
    #[cfg(debug_assertions)] // 宏行为测试:release 空展开(零开销),无记录可查
    fn paired_timer_records_sample_with_location() {
        ensure_init();
        timer_start!("t.paired");
        thread::sleep(Duration::from_millis(2));
        timer_end!("t.paired");
        let reps = collect_metrics(&MetricsFilter::new().tag("t.paired"));
        assert_eq!(reps.len(), 1, "精确 tag 过滤应命中");
        let r = &reps[0];
        assert!(r.count >= 1, "至少一条样本");
        assert!(dur_ms(r.min) >= 1.5, "时长应覆盖 sleep: {:?}", r.min);
        // 位置 = 本测试文件(宏展开点);metrics 拆独立 crate 后测试
        // 驻 lib.rs(断言随搬家订正,原 "metrics.rs" 系 shared 时代文件名)
        assert!(r.last_file.ends_with("lib.rs"), "file = {}", r.last_file);
        assert!(r.last_line > 0);
        assert_eq!(r.orphan_ends, 0);
    }

    #[test]
    fn orphan_end_counted_not_panicking() {
        ensure_init();
        let direct = MetricsStore::new();
        assert!(matches!(
            direct.timer_end("t.orphan", file!(), line!()),
            Err(MetricsError::OrphanEnd(_))
        ));
        let r = direct.collect(&MetricsFilter::new().tag("t.orphan"));
        assert_eq!(r[0].orphan_ends, 1, "orphan 应计数");
    }

    #[test]
    fn nested_same_tag_is_lifo() {
        let s = MetricsStore::new();
        s.timer_begin("t.nest", file!(), line!());
        thread::sleep(Duration::from_millis(1));
        s.timer_begin("t.nest", file!(), line!());
        thread::sleep(Duration::from_millis(2));
        s.timer_end("t.nest", file!(), line!()).expect("内层配对");
        s.timer_end("t.nest", file!(), line!()).expect("外层配对");
        let r = s.collect(&MetricsFilter::new().tag("t.nest"));
        assert_eq!(r[0].count, 2);
        assert_eq!(r[0].open_spans, 0);
        // LIFO:后入先出 → 先 end 的是内层(2ms),后 end 的外层 ≥ 内层 + 1ms
        let dq_samples = &s.inner.lock().unwrap().samples["t.nest"];
        let (outer, inner_d) = (dur_ms(dq_samples[1].dur), dur_ms(dq_samples[0].dur));
        assert!(outer >= inner_d + 0.8, "外层应包住内层: {outer} vs {inner_d}");
    }

    #[test]
    fn multithread_same_tag_pairs_independently() {
        let s = std::sync::Arc::new(MetricsStore::new());
        let mut hs = Vec::new();
        for _ in 0..4 {
            let s2 = s.clone();
            hs.push(thread::spawn(move || {
                // 多线程同名 tag 并发配对(按 (tag, thread) 分栈互不干扰)
                for _ in 0..50 {
                    s2.timer_begin("t.mt", file!(), line!());
                    thread::sleep(Duration::from_micros(50));
                    s2.timer_end("t.mt", file!(), line!()).expect("配对");
                }
            }));
        }
        for h in hs {
            h.join().expect("线程");
        }
        let r = s.collect(&MetricsFilter::new().tag("t.mt"));
        assert_eq!(r[0].count, 200, "4 线程 × 50 次");
        assert_eq!(r[0].open_spans, 0);
    }

    #[test]
    fn filter_axes_and_sort() {
        let s = MetricsStore::new();
        s.timer_record("f.a.total", Duration::from_millis(10), file!(), line!());
        s.timer_record("f.a.total", Duration::from_millis(20), file!(), line!());
        s.timer_record("f.b.total", Duration::from_millis(5), file!(), line!());
        s.counter_add("f.a.count", 7, file!(), line!());
        // 精确
        assert_eq!(s.collect(&MetricsFilter::new().tag("f.a.total")).len(), 1);
        // 前缀
        let all_f = s.collect(&MetricsFilter::new().tag_prefix("f."));
        assert_eq!(all_f.len(), 3, "f.a.total/f.b.total/f.a.count");
        // 包含
        assert_eq!(s.collect(&MetricsFilter::new().tag_contains(".b.")).len(), 1);
        // min_count(timer 样本数)
        assert_eq!(s.collect(&MetricsFilter::new().min_count(2)).len(), 1, "仅 f.a.total 有 2 样本");
        // 排序:TotalDesc → f.a.total(30ms)第一
        let sorted = s.collect(&MetricsFilter::new().tag_prefix("f.").sort(MetricsSort::TotalDesc));
        assert_eq!(sorted[0].tag, "f.a.total");
        assert_eq!(dur_ms(sorted[0].total), 30.0);
        assert_eq!(dur_ms(sorted[0].p50), 20.0, "两样本 p50 = 偏大者");
        // limit
        assert_eq!(s.collect(&MetricsFilter::new().tag_prefix("f.").limit(2)).len(), 2);
    }

    #[test]
    fn counter_and_reset() {
        let s = MetricsStore::new();
        s.counter_add("c.x", 3, file!(), line!());
        s.counter_add("c.x", 4, file!(), line!());
        let r = s.collect(&MetricsFilter::new().tag("c.x"));
        assert_eq!(r[0].counter, 7);
        s.reset();
        // reset 后 tag 记录全清 → 无报告(store 本体保留,可继续用)
        assert!(s.collect(&MetricsFilter::new().tag("c.x")).is_empty());
        s.counter_add("c.x", 1, file!(), line!());
        assert_eq!(s.collect(&MetricsFilter::new().tag("c.x"))[0].counter, 1);
    }

    #[test]
    fn cap_evicts_oldest_and_counts_dropped() {
        let s = MetricsStore::with_cap(8);
        for i in 0..20 {
            s.timer_record("cap.x", Duration::from_micros(i), file!(), line!());
        }
        let r = s.collect(&MetricsFilter::new().tag("cap.x"));
        assert_eq!(r[0].count, 8, "cap 环形");
        assert_eq!(r[0].dropped, 12);
        // 最旧被驱逐 → min = 12µs 起
        assert_eq!(r[0].min, Duration::from_micros(12));
    }

    #[test]
    #[cfg(debug_assertions)] // 宏行为测试:release 空展开(零开销),无记录可查
    fn macros_compile_and_route_to_global() {
        ensure_init();
        timer_start!("t.macro.route");
        counter_inc!("t.macro.cnt");
        counter_add!("t.macro.cnt", 2);
        event!("t.macro.evt");
        timer_record!("t.macro.rec", Duration::from_millis(3));
        timer_end!("t.macro.route");
        let scoped = timer_scope!("t.macro.scope", 40 + 2);
        assert_eq!(scoped, 42, "scope 保留块值");
        let reps = collect_metrics(&MetricsFilter::new().tag_prefix("t.macro."));
        eprintln!("[dbg] reps = {:?}", reps.iter().map(|r| (r.tag.clone(), r.count, r.counter)).collect::<Vec<_>>());
        assert!(reps.len() >= 4, "route/cnt/evt/rec/scope 至少四类");
        let route = collect_metrics(&MetricsFilter::new().tag("t.macro.route"));
        assert_eq!(route[0].count, 1);
        let cnt = collect_metrics(&MetricsFilter::new().tag("t.macro.cnt"));
        assert_eq!(cnt[0].counter, 3);
    }

    #[test]
    #[cfg(debug_assertions)] // 宏行为测试:release 空展开(零开销),无记录可查
    fn query_prints_and_returns() {
        ensure_init();
        timer_record!("t.query.x", Duration::from_millis(1));
        let reps = query_metrics(&MetricsFilter::new().tag_prefix("t.query."));
        assert_eq!(reps.len(), 1);
    }
}
