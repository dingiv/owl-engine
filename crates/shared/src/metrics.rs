//! 统一性能 metrics 框架(2026-10-01 立项;合并散装计时刻度的落点 ——
//! 原 `OWL_STEP_PROFILE` 分相 eprintln、engine-bak layer_metrics A5 挂账
//! 的接线面)。
//!
//! # 四件核心 API
//!
//! 1. [`init_metrics`] —— 程序早期调用一次,把 store 挂上全局指针
//!    ([`OnceLock`];重复 init = [`MetricsError::AlreadyInit`]);
//! 2. [`timer_start!`](crate::timer_start) / [`timer_end!`](crate::timer_end)
//!    —— 任意两段代码间计时:**严格两两配对**(同线程 LIFO 栈),
//!    不打印,样本落内存 store;样本带 **调用点位置**(`file!/line!/column!`
//!    在 macro_rules 展开点取值 = C 宏 `__FILE__/__LINE__` 同款语义)
//!    与 **父 tag**(调用树边 —— 见下「性能框架」);
//! 3. [`query_metrics`] —— 按 [`MetricsFilter`] 过滤、聚合、**打印**并返回
//!    [`TagReport`](`collect_metrics` = 纯收集不打印);
//! 4. 周边面:counter(`counter_inc!`/`counter_add!`)、事件(`event!`)、
//!    手动记录(`timer_record!`)、作用域糖(`timer_scope!`)、RAII 段
//!    (`perf_span!`)、[`reset_metrics`] / [`metrics_ready`] /
//!    [`with_metrics_store`]。
//!
//! # 性能框架(`docs/arch/perf-framework.md`,2026-10-14)
//!
//! 在原通用打点器之上补齐三件:
//!
//! - **release 可见性**:宏在 `debug_assertions` **或** cargo feature
//!   `perf` 下展开(生产二进制不带 feature = 逐字节零成本;基准二进制
//!   `cargo build --release -p server --features perf`);
//! - **运行期分级**([`level`],`off`/`counters`/`host`/`gpu`):门在 store
//!   入口,`off` = 零成本基线对照臂(量仪表自身税);缺省 `host` =
//!   历史行为零变化;
//! - **调用树与排他时间**:样本自带 `parent`(同线程单栈推出),故
//!   [`TagReport::self_total`] = 真自耗;配套 [`render_tree`] / [`rollup`]
//!   / [`mark_epoch`] / [`export_csv`] / [`export_json`] / [`dump`]。
//!
//! # 零开销纪律(编译期 + 运行期两道门)
//!
//! 1. **编译期**:不带 `perf` feature 的 release 里,全部计时/计数宏
//!    展开为空语句 —— 零运行时开销,tag 表达式不被求值(带副作用的
//!    tag 表达式在 release 下不执行,契约上 tag 应为字面量);
//! 2. **运行期**:即使带 feature,每条钩子仍过[分级门](level)(一次
//!    relaxed 原子读);`off` 级下 [`Instant::now`] 都不调。
//! 3. 位置参数由宏体内 `file!/line!/column!` 提供,调用方零样板。
//!
//! # 线程与配对契约
//!
//! - start/end **同线程配对**(每线程一条 span 栈,父 = 其下一条);
//! - 同名 tag 可嵌套(LIFO 弹栈);栈顶不符时向下找最近匹配(计
//!   `TagReport.out_of_order`,不报 orphan);
//! - 无配对 start 的 end 不 panic:计 orphan(`TagReport.orphan_ends`);
//! - 程序退出时的未闭合 start 计入 `TagReport.open_spans`。
//!
//! ⚠️ **asyn 跨度**:span 跨越 `.await` 且 future 可能被搬线程的场合,栈
//! 配对会错位(引擎 actor = current_thread 运行时,单线程闭合 —— 现役
//! 用法安全)。注解快路径(interpreter)不走栈,直接 `timer_record_tag`。
//!
//! # 直用形态
//!
//! [`MetricsStore`] 的方法全部 pub:不经全局(如单测、多 store 并存)
//! 可直接持有 store 调用;宏 = 全局指针的快捷方式。

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, OnceLock};
use std::thread::ThreadId;
use std::time::{Duration, Instant, SystemTime};

// ============================================================================
// §0 性能分级(perf-framework.md §一)
// ============================================================================
//
// 两级可见性:
// - **编译期**:宏在 `debug_assertions` **或** cargo feature `perf` 下展开
//   ⇒ 不带 feature 的生产二进制逐字节零成本;
// - **运行期**:本节分级门(Level 0-3)控制采集深度,门在 **store 入口**
//   (timer/counter 的公开方法),宏与直用 API 走同一道门,不可能绕。
//
// 缺省 = [`LEVEL_HOST`](2):历史打点行为零变化;`off` 是新增的**零成本
// 基线对照臂**(量仪表自身税),`gpu` 是设备侧采集的准入级。

/// 性能分级(运行期)
pub mod level {
    /// 全关(零成本基线对照臂:量仪表自身税)
    pub const OFF: u8 = 0;
    /// 计数面(counter / gauge;无时延噪声)
    pub const COUNTERS: u8 = 1;
    /// + 宿主计时(**缺省**,与历史行为逐位一致)
    pub const HOST: u8 = 2;
    /// + 设备侧采集(eager 逐发射 event 对 / 图态节点账)
    pub const GPU: u8 = 3;

    /// 名称 → 级(`/debug/perf/level/<name>` 与配置面共用)
    pub fn parse(s: &str) -> Option<u8> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "off" => OFF,
            "counters" => COUNTERS,
            "host" => HOST,
            "gpu" => GPU,
            _ => return None,
        })
    }

    /// 级 → 名称(报告/日志面)
    pub fn name(l: u8) -> &'static str {
        match l {
            OFF => "off",
            COUNTERS => "counters",
            HOST => "host",
            _ => "gpu",
        }
    }
}

/// 未 init 时的缺省级(init 时种入 store;`set_level` 同时更新两者)
static DEFAULT_LEVEL: AtomicU8 = AtomicU8::new(level::HOST);

/// 本构建是否带性能面(宏展开 / 钩子代码是否存活)。
///
/// `false` ⇒ 调用方写 `if PERF_BUILD && ...` 的整段被**常量折叠**掉
/// (非 perf 的 release 里逐字节零成本),且不引入 `cfg(feature)` 到下游
/// crate —— 下游看不见本 crate 的 feature,也就不会触发
/// `unexpected_cfgs` 告警。
#[cfg(any(debug_assertions, feature = "perf"))]
pub const PERF_BUILD: bool = true;
#[cfg(not(any(debug_assertions, feature = "perf")))]
pub const PERF_BUILD: bool = false;

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
    /// 外层 span 的 tag(同线程开栈的直接父;None = 最外层)。
    /// 这就是**调用树**的来源:报告按此算 self(排他)时间。
    pub parent: Option<Box<str>>,
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
    /// **排他时间** = total − Σ(以本 tag 为直接父的样本时长)。
    /// 平铺 total 会把父节点重复计一遍;self 才是热点判据。
    pub self_total: Duration,
    /// self_total / 参考总时(填写它调 [`render_tree`];否则 0)
    pub self_pct: f64,
    /// 未闭合 start(LIFO 栈深)
    pub open_spans: u64,
    /// 无配对 start 的 end 计数
    pub orphan_ends: u64,
    /// end 与栈顶 tag 不符但向下找到匹配的计数(乱序配对;LIFO 违例信号)
    pub out_of_order: u64,
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
    /// **排他**耗时降序(真正的自耗时热点;父节点不重复计)
    SelfDesc,
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

/// 同线程开栈上限(病态使用的失控闸;超过 = start 被拒 + debug 告警)
const MAX_OPEN_PER_THREAD: usize = 4096;

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

/// 未闭合 span(单栈 LIFO;其下一条 = 本条的父)
#[derive(Clone, Debug)]
struct OpenSpan {
    tag: String,
    start: Instant,
    file: &'static str,
    line: u32,
}

#[derive(Default)]
struct Inner {
    /// 未闭合 span 栈:**每线程一条**(2026-10-14 性能框架 —— 原按 (tag,thread)
    /// 分栈改为单栈,父指针/调用树由此而来;同 tag 嵌套仍 LIFO)
    open: HashMap<u64, Vec<OpenSpan>>,
    /// 已配对样本:key = tag(容量 cap 环形驱逐)
    samples: HashMap<String, VecDeque<Sample>>,
    /// 计数器:key = (count, last 位置)
    counters: HashMap<String, (u64, &'static str, u32)>,
    /// 容量驱逐计数
    dropped: HashMap<String, u64>,
    /// orphan end 计数(按 tag)
    orphans: HashMap<String, u64>,
    /// 乱序配对计数(end 与栈顶不符但向下找到匹配;LIFO 违例信号)
    out_of_order: HashMap<String, u64>,
    /// epoch 标记([`MetricsStore::mark_epoch`];「只看这一轮」的起点)
    epoch: Option<SystemTime>,
    /// 线程序号分配(ThreadId 无序号 → 首现序)
    threads: HashMap<ThreadId, u64>,
    next_thread: u64,
}

/// metrics 存储(内存态;线程安全 = 单把 Mutex —— 打点粒度微秒级,
/// 锁开销远小于被测段,不优化)
pub struct MetricsStore {
    inner: Mutex<Inner>,
    cap_per_tag: usize,
    /// 运行期分级(见 §0;缺省 = [`level::HOST`] = 历史行为)
    level: AtomicU8,
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
        Self { inner: Mutex::new(Inner::default()), cap_per_tag, level: AtomicU8::new(level::HOST) }
    }

    /// 带分级的显式构造(测试/基准域;隔离级别的唯一安全形态)
    pub fn with_cap_level(cap_per_tag: usize, l: u8) -> Self {
        Self { inner: Mutex::new(Inner::default()), cap_per_tag, level: AtomicU8::new(l.min(level::GPU)) }
    }

    /// 本 store 的运行期分级
    #[inline]
    pub fn level(&self) -> u8 {
        self.level.load(Ordering::Relaxed)
    }

    /// 置分级(在线切换;`off` = 零成本基线对照臂)
    pub fn set_level(&self, l: u8) {
        self.level.store(l.min(level::GPU), Ordering::Relaxed);
    }

    /// 钩子准入判定(热路径唯一分支)
    #[inline]
    pub fn armed(&self, need: u8) -> bool {
        self.level.load(Ordering::Relaxed) >= need
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

    /// timer_start 实现:start 入本线程 LIFO 栈
    pub fn timer_begin(&self, tag: &str, file: &'static str, line: u32) {
        if !self.armed(level::HOST) {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        let t = self.thread_no(&mut inner);
        let stack = inner.open.entry(t).or_default();
        if stack.len() >= MAX_OPEN_PER_THREAD {
            #[cfg(any(debug_assertions, feature = "perf"))]
            crate::log_debug!("metrics", "同线程开栈超限(>{MAX_OPEN_PER_THREAD}),start 被拒: {tag}");
            return;
        }
        stack.push(OpenSpan { tag: tag.to_string(), start: Instant::now(), file, line });
    }

    /// timer_end 实现:弹栈配对 → 样本落库(**附父 tag = 调用树边**)
    pub fn timer_end(&self, tag: &str, file: &'static str, line: u32) -> Result<(), MetricsError> {
        if !self.armed(level::HOST) {
            return Ok(());
        }
        let now = Instant::now();
        let at = SystemTime::now();
        let mut inner = self.inner.lock().unwrap();
        let t = self.thread_no(&mut inner);
        let stack = inner.open.entry(t).or_default();
        // 严格 LIFO 优先;栈顶不符时向下找最近匹配(乱序容错 + 计数)
        let Some(pos) = stack.iter().rposition(|s| s.tag == tag) else {
            #[cfg(any(debug_assertions, feature = "perf"))]
            crate::log_debug!("metrics", "timer_end 无配对 start: {tag} @ {file}:{line}");
            *inner.orphans.entry(tag.to_string()).or_default() += 1;
            return Err(MetricsError::OrphanEnd(tag.to_string()));
        };
        let disorder = pos + 1 != stack.len();
        let open = stack.remove(pos);
        let parent =
            if pos > 0 { Some(stack[pos - 1].tag.clone().into_boxed_str()) } else { None };
        if disorder {
            *inner.out_of_order.entry(tag.to_string()).or_default() += 1;
        }
        let sample = Sample {
            dur: now.saturating_duration_since(open.start),
            at,
            file: open.file,
            line: open.line,
            thread: t,
            parent,
        };
        self.push_sample(&mut inner, tag, sample);
        Ok(())
    }

    /// 样本落库(容量环形驱逐 + 计数;调用方持锁)
    fn push_sample(&self, inner: &mut Inner, tag: &str, sample: Sample) {
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
    }

    /// 手动记录一次耗时(等价一对 start/end 的样本;无父:平铺计入)
    pub fn timer_record(&self, tag: &str, dur: Duration, file: &'static str, line: u32) {
        if !self.armed(level::HOST) {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        let t = self.thread_no(&mut inner);
        let sample =
            Sample { dur, at: SystemTime::now(), file, line, thread: t, parent: None };
        self.push_sample(&mut inner, tag, sample);
    }

    /// timer_record 的动态 tag 形态(直用 API;宏路径 tag 恒字面量)
    pub fn timer_record_tag(&self, tag: &str, dur: Duration, file: &'static str, line: u32) {
        self.timer_record(tag, dur, file, line)
    }

    /// 计数器 += n
    pub fn counter_add(&self, tag: &str, n: u64, file: &'static str, line: u32) {
        if !self.armed(level::COUNTERS) {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        let e = inner.counters.entry(tag.to_string()).or_insert((0, file, line));
        e.0 += n;
        e.1 = file;
        e.2 = line;
    }

    /// gauge:置当前值(覆盖语义;AL 窗口/pos/配置开关等"现在多少"类,
    /// 与 counter 的累计语义区分;/debug/metrics 的 counter 字段直读)
    pub fn counter_set(&self, tag: &str, v: u64, file: &'static str, line: u32) {
        if !self.armed(level::COUNTERS) {
            return;
        }
        let mut inner = self.inner.lock().unwrap();
        let e = inner.counters.entry(tag.to_string()).or_insert((0, file, line));
        e.0 = v;
        e.1 = file;
        e.2 = line;
    }

    /// 聚合 + 过滤(query 数据面)。含**排他时间**计算:
    /// self = total − Σ(以本 tag 为直接父的样本时长)。
    pub fn collect(&self, filter: &MetricsFilter) -> Vec<TagReport> {
        let inner = self.inner.lock().unwrap();
        // 父 → 子时长和(一次扫描;调用树边已在样本里)
        let mut child_sum: HashMap<&str, Duration> = HashMap::new();
        for dq in inner.samples.values() {
            for s in dq.iter() {
                if let Some(p) = s.parent.as_deref() {
                    *child_sum.entry(p).or_insert(Duration::ZERO) += s.dur;
                }
            }
        }
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
            .map(|tag| self.report_one(&inner, tag, filter, &child_sum))
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
            MetricsSort::TotalDesc => reports.sort_by_key(|b| std::cmp::Reverse(b.total)),
            MetricsSort::SelfDesc => reports.sort_by_key(|b| std::cmp::Reverse(b.self_total)),
            MetricsSort::CountDesc => reports.sort_by_key(|b| std::cmp::Reverse(b.count)),
            MetricsSort::RecentDesc => reports.sort_by_key(|b| std::cmp::Reverse(b.last_at)),
            MetricsSort::NameAsc => reports.sort_by(|a, b| a.tag.cmp(&b.tag)),
        }
        if let Some(l) = filter.limit {
            reports.truncate(l);
        }
        reports
    }

    /// 单 tag 聚合(锁内调用)
    fn report_one(
        &self,
        inner: &Inner,
        tag: &str,
        filter: &MetricsFilter,
        child_sum: &HashMap<&str, Duration>,
    ) -> TagReport {
        let mut r = TagReport { tag: tag.to_string(), ..Default::default() };
        if let Some(dq) = inner.samples.get(tag) {
            let mut durs: Vec<Duration> = dq
                .iter()
                .filter(|s| filter.since.is_none_or(|t| s.at >= t))
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
            // 排他时间:扣掉直接子样本(since 窗口下的近似 —— 子样本未必同窗)
            r.self_total = r.total.saturating_sub(child_sum.get(tag).copied().unwrap_or(Duration::ZERO));
        }
        if let Some((c, f, l)) = inner.counters.get(tag) {
            r.counter = *c;
            if r.last_at.is_none() {
                r.last_file = f;
                r.last_line = *l;
            }
        }
        // open spans 跨线程累加(同 tag 多线程各自栈)
        for st in inner.open.values() {
            r.open_spans += st.iter().filter(|s| s.tag == tag).count() as u64;
        }
        r.orphan_ends = inner.orphans.get(tag).copied().unwrap_or(0);
        r.out_of_order = inner.out_of_order.get(tag).copied().unwrap_or(0);
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
        inner.out_of_order.clear();
        inner.epoch = None;
    }

    // ── epoch(「只看这一轮/本请求」;不 reset,保住历史样本)────────────

    /// 打 epoch 标记:此后 [`collect_since_epoch`](Self::collect_since_epoch)
    /// 只看标记之后的样本(历史「同 session 二发走前缀缓存污染差分」即缺
    /// 此机制)
    pub fn mark_epoch(&self) -> SystemTime {
        let now = SystemTime::now();
        self.inner.lock().unwrap().epoch = Some(now);
        now
    }

    /// 当前 epoch 时刻(未打标 = None)
    pub fn epoch(&self) -> Option<SystemTime> {
        self.inner.lock().unwrap().epoch
    }

    /// 按 epoch 过滤(未打标 = 全量)
    pub fn collect_since_epoch(&self, mut filter: MetricsFilter) -> Vec<TagReport> {
        if filter.since.is_none() {
            filter.since = self.epoch();
        }
        self.collect(&filter)
    }

    // ── 导出(离线比对多配置;CSV 一行一 tag)─────────────────────

    /// CSV 导出(tag,count,counter,total_ms,self_ms,avg_ms,p50_ms,p99_ms,max_ms)
    pub fn export_csv(&self, filter: &MetricsFilter) -> String {
        let mut out =
            String::from("tag,count,counter,total_ms,self_ms,avg_ms,p50_ms,p99_ms,max_ms\n");
        for r in self.collect(filter) {
            out.push_str(&format!(
                "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                r.tag,
                r.count,
                r.counter,
                ms(r.total),
                ms(r.self_total),
                ms(r.avg),
                ms(r.p50),
                ms(r.p99),
                ms(r.max),
            ));
        }
        out
    }

    /// JSON 导出(手写零依赖;与 `/debug/metrics` 同形 + self/orphan/ooo)
    pub fn export_json(&self, filter: &MetricsFilter) -> String {
        let rows = self.collect(filter);
        let mut s = format!("{{\"level\":\"{}\",\"rows\":[", level::name(self.level()));
        for (i, r) in rows.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                "{{\"tag\":\"{}\",\"count\":{},\"counter\":{},\"total_ms\":{:.4},\"self_ms\":{:.4},\"avg_ms\":{:.4},\"p50_ms\":{:.4},\"p99_ms\":{:.4},\"max_ms\":{:.4},\"open\":{},\"orphan\":{},\"ooo\":{},\"drop\":{}}}",
                json_escape(&r.tag),
                r.count,
                r.counter,
                ms(r.total),
                ms(r.self_total),
                ms(r.avg),
                ms(r.p50),
                ms(r.p99),
                ms(r.max),
                r.open_spans,
                r.orphan_ends,
                r.out_of_order,
                r.dropped,
            ));
        }
        s.push_str("]}");
        s
    }

    /// 落盘(CSV + 同名 .json;经 [`crate::file_loader`] 统一 I/O 口)。
    /// 每轮覆写 —— 进程被 kill 也不丢账。
    pub fn dump(&self, base: impl AsRef<std::path::Path>, filter: &MetricsFilter) -> std::io::Result<()> {
        let base = base.as_ref();
        crate::file_loader::write(base, self.export_csv(filter))?;
        let json_path = base.with_extension("json");
        crate::file_loader::write(&json_path, self.export_json(filter))?;
        Ok(())
    }

    /// 树状报告(含 self/占比/次数;父/子由 tag 前缀推导)
    pub fn render_tree(&self, filter: &MetricsFilter, ref_total: Duration) -> String {
        render_tree(&self.collect(filter), ref_total)
    }
}

// ============================================================================
// §3 全局指针 + 公共 API
// ============================================================================

static STORE: OnceLock<MetricsStore> = OnceLock::new();

/// 程序早期初始化:把 store 挂上全局指针(幂等失败 = [`MetricsError::AlreadyInit`],
/// 保留首个 —— 引擎/server 单 store 契约)。
pub fn init_metrics(store: MetricsStore) -> Result<(), MetricsError> {
    // 缺省级种入(入口可能在 init 前用 `set_level` 配过)
    store.set_level(DEFAULT_LEVEL.load(Ordering::Relaxed));
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
        crate::log_debug!("metrics", "无匹配记录(filter: tag={:?} prefix={:?} contains={:?})",
            filter.tag, filter.tag_prefix, filter.tag_contains);
        return reports;
    }
    crate::log_debug!("metrics", "{:<32} {:>7} {:>11} {:>10} {:>10} {:>10} {:>10} {:>5} {:>6} {:>6}  last@",
        "tag", "count", "total", "avg", "p50", "p99", "max", "open", "orph", "drop"
    );
    for r in &reports {
        let (f, l) = (r.last_file.rsplit('/').next().unwrap_or(r.last_file), r.last_line);
        crate::log_debug!("metrics", "{:<32} {:>7} {:>11} {:>10} {:>10} {:>10} {:>10} {:>5} {:>6} {:>6}  {}:{}",
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
            crate::log_debug!("metrics", "{:<32} counter = {}", "", r.counter);
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

// ── 分级(全局;见 §0)──────────────────────────────────────────────

/// 置分级(`off` = 零成本基线臂;`gpu` = 设备侧采集准入)
pub fn set_level(l: u8) {
    DEFAULT_LEVEL.store(l.min(level::GPU), Ordering::Relaxed);
    if let Some(s) = STORE.get() {
        s.set_level(l);
    }
}

/// 当前全局分级(store 未 init 时读缺省级)
pub fn level() -> u8 {
    STORE.get().map(|s| s.level()).unwrap_or_else(|| DEFAULT_LEVEL.load(Ordering::Relaxed))
}

/// 当前分级名(`off`/`counters`/`host`/`gpu`)
pub fn level_name() -> &'static str {
    level::name(level())
}

// ── epoch / 导出(全局)────────────────────────────────────────────

/// 打 epoch 标记(「只看这一轮」起点);沿用 store 既有 since 过滤面
pub fn mark_epoch() -> SystemTime {
    STORE.get().map(|s| s.mark_epoch()).unwrap_or_else(SystemTime::now)
}

/// 按 epoch 过滤收集
pub fn collect_since_epoch(filter: MetricsFilter) -> Vec<TagReport> {
    STORE.get().map(|s| s.collect_since_epoch(filter)).unwrap_or_default()
}

/// CSV 导出(全局)
pub fn export_csv(filter: &MetricsFilter) -> String {
    STORE.get().map(|s| s.export_csv(filter)).unwrap_or_default()
}

/// JSON 导出(全局)
pub fn export_json(filter: &MetricsFilter) -> String {
    STORE.get().map(|s| s.export_json(filter)).unwrap_or_default()
}

/// 落盘(全局;CSV + JSON 双份)
pub fn dump(filter: &MetricsFilter, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    match STORE.get() {
        Some(s) => s.dump(path, filter),
        None => Ok(()),
    }
}

// ============================================================================
// §3.5 报告渲染(树 / rollup)
// ============================================================================

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn json_escape(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

/// 树节点:tag 前缀分段
#[derive(Default)]
struct Node {
    /// 本节点自身的样本(只有完整 tag 落在这里)
    own: Option<(u64, Duration, Duration)>, // (count, total, self)
    children: std::collections::BTreeMap<String, Node>,
}

/// 树状报告:按 tag 的 `.` 分段建树,每行给 `self`(排他)、`total`、
/// `%`(相对 `ref_total`,通常 = 顶层包络 span 的 total)与样本数。
///
/// 判读纪律:**先看 self 排序** —— 平铺 `total` 会把父节点重复计一遍。
pub fn render_tree(reports: &[TagReport], ref_total: Duration) -> String {
    let mut root = Node::default();
    for r in reports {
        let mut cur = &mut root;
        for seg in r.tag.split('.') {
            cur = cur.children.entry(seg.to_string()).or_default();
        }
        cur.own = Some((r.count, r.total, r.self_total));
    }
    let mut out = String::new();
    let base = ref_total.as_nanos().max(1) as f64;
    for (name, n) in root.children.iter() {
        render_node(&mut out, name, n, 0, base);
    }
    if out.is_empty() {
        out.push_str("(无样本)\n");
    }
    out
}

fn render_node(out: &mut String, name: &str, n: &Node, depth: usize, base: f64) {
    let pad = "  ".repeat(depth);
    // 子树聚合(自耗 = 子树 self 和;用于父行展示)
    let (cnt, _tot, sub_self) = subtree(n);
    let self_d = n.own.map(|(_, _, s)| s).unwrap_or(Duration::ZERO);
    let self_pct = self_d.as_nanos() as f64 / base * 100.0;
    if n.own.is_some() {
        out.push_str(&format!(
            "{pad}{name:<34} n={cnt:<7} self={:>9} {:>6.2}%\n",
            fmt_dur(self_d),
            self_pct,
        ));
    } else {
        out.push_str(&format!(
            "{pad}{name:<34} n={cnt:<7} subself={:>9} {:>6.2}%\n",
            fmt_dur(sub_self),
            sub_self.as_nanos() as f64 / base * 100.0,
        ));
    }
    // 子按 self 降序(热点优先)
    let mut kids: Vec<(&String, u64, Duration)> = n
        .children
        .iter()
        .map(|(k, c)| {
            let (_, _, s) = subtree(c);
            let own_s = c.own.map(|(_, _, s)| s).unwrap_or(Duration::ZERO);
            (k, 0u64, own_s + s)
        })
        .collect();
    kids.sort_by_key(|b| std::cmp::Reverse(b.2));
    for (k, _, _) in kids {
        render_node(out, k, &n.children[k], depth + 1, base);
    }
}

/// (样本数, total 和, self 和)全子树聚合
fn subtree(n: &Node) -> (u64, Duration, Duration) {
    let mut cnt = 0u64;
    let mut tot = Duration::ZERO;
    let mut self_sum = Duration::ZERO;
    if let Some((c, t, s)) = n.own {
        cnt += c;
        tot += t;
        self_sum += s;
    }
    for c in n.children.values() {
        let (c2, t2, s2) = subtree(c);
        cnt += c2;
        tot += t2;
        self_sum += s2;
    }
    (cnt, tot, self_sum)
}

/// 前缀 rollup:把 tag 按前 `depth` 段折并(不看层次,只看名字)
pub fn rollup(reports: &[TagReport], depth: usize) -> Vec<TagReport> {
    let mut m: std::collections::BTreeMap<String, TagReport> = Default::default();
    for r in reports {
        let key: String = r.tag.split('.').take(depth.max(1)).collect::<Vec<_>>().join(".");
        let e = m.entry(key.clone()).or_insert_with(|| TagReport { tag: key, ..Default::default() });
        e.count += r.count;
        e.total += r.total;
        e.self_total += r.self_total;
        e.counter += r.counter;
    }
    for e in m.values_mut() {
        if e.count > 0 {
            e.avg = e.total / (e.count as u32);
        }
    }
    let mut v: Vec<TagReport> = m.into_values().collect();
    v.sort_by_key(|b| std::cmp::Reverse(b.self_total));
    v
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
// §3.6 RAII span([`perf_span!`] 的实现载体)
// ============================================================================

/// 一个正在跑的计时段。`Drop` 时落样本(等价 timer_end)。
/// 捕获宏调用点的 file/line(start 位置)。
pub struct Span {
    tag: &'static str,
    start: Instant,
    file: &'static str,
    line: u32,
}

impl Span {
    /// 准入判定 + 起表(未到 [`level::HOST`] = None,零成本)
    pub fn begin(tag: &'static str, file: &'static str, line: u32) -> Option<Self> {
        if !STORE.get().map(|s| s.armed(level::HOST)).unwrap_or(false) {
            return None;
        }
        Some(Self { tag, start: Instant::now(), file, line })
    }

    /// 当前 span 的 tag(诊断面)
    pub fn tag(&self) -> &'static str {
        self.tag
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(s) = STORE.get() {
            s.timer_record(self.tag, self.start.elapsed(), self.file, self.line);
        }
    }
}

// ============================================================================
// §4 宏(编译期分派:**双定义**,cfg 只在定义域求值)
// ============================================================================
// 纪律:每个宏在本 crate 内按 `any(debug_assertions, feature = "perf")`
// 分两种定义 —— **有性能面 = 实现调用;无 = 空展开**。这样:
//   ① 下游 crate 的 `cfg` 里不会出现 `feature = "perf"`(它没有这个
//      feature ⇒ 否则触发 `unexpected_cfgs` 告警);
//   ② 非 perf 的 release 里宏展开为**空语句**:零运行时,tag 表达式不被
//      求值(带副作用的 tag 表达式在该档不执行 —— 契约上 tag 应为字面量);
//   ③ `file!/line!/column!` 在宏调用点取值(C `__FILE__/__LINE__` 同语义)。

#[cfg(any(debug_assertions, feature = "perf"))]
#[macro_export]
macro_rules! timer_start {
    ($tag:expr) => {
        $crate::metrics::__timer_start($tag, file!(), line!(), column!());
    };
}
#[cfg(not(any(debug_assertions, feature = "perf")))]
#[macro_export]
macro_rules! timer_start {
    ($tag:expr) => {};
}

#[cfg(any(debug_assertions, feature = "perf"))]
#[macro_export]
macro_rules! timer_end {
    ($tag:expr) => {
        $crate::metrics::__timer_end($tag, file!(), line!(), column!());
    };
}
#[cfg(not(any(debug_assertions, feature = "perf")))]
#[macro_export]
macro_rules! timer_end {
    ($tag:expr) => {};
}

#[cfg(any(debug_assertions, feature = "perf"))]
#[macro_export]
macro_rules! timer_scope {
    ($tag:expr, $body:expr) => {{
        $crate::metrics::__timer_start($tag, file!(), line!(), column!());
        let __owl_m_val = $body;
        $crate::metrics::__timer_end($tag, file!(), line!(), column!());
        __owl_m_val
    }};
}
#[cfg(not(any(debug_assertions, feature = "perf")))]
#[macro_export]
macro_rules! timer_scope {
    ($tag:expr, $body:expr) => {{
        $body
    }};
}

/// **RAII 计时段(推荐形态)**:`let _s = perf_span!("tag");` —— 离开
/// 作用域即落样本,免疫忘记配对 / 早退漏 end。返回 `Option<Span>`
/// (未到 HOST 级 = None,零成本)。
#[cfg(any(debug_assertions, feature = "perf"))]
#[macro_export]
macro_rules! perf_span {
    ($tag:expr) => {
        $crate::metrics::Span::begin($tag, file!(), line!())
    };
}
#[cfg(not(any(debug_assertions, feature = "perf")))]
#[macro_export]
macro_rules! perf_span {
    ($tag:expr) => {
        None
    };
}

#[cfg(any(debug_assertions, feature = "perf"))]
#[macro_export]
macro_rules! counter_inc {
    ($tag:expr) => {
        $crate::metrics::__counter_add($tag, 1, file!(), line!());
    };
}
#[cfg(not(any(debug_assertions, feature = "perf")))]
#[macro_export]
macro_rules! counter_inc {
    ($tag:expr) => {};
}

#[cfg(any(debug_assertions, feature = "perf"))]
#[macro_export]
macro_rules! counter_add {
    ($tag:expr, $n:expr) => {
        $crate::metrics::__counter_add($tag, $n, file!(), line!());
    };
}
#[cfg(not(any(debug_assertions, feature = "perf")))]
#[macro_export]
macro_rules! counter_add {
    ($tag:expr, $n:expr) => {};
}

#[cfg(any(debug_assertions, feature = "perf"))]
#[macro_export]
macro_rules! event {
    ($tag:expr) => {
        $crate::metrics::__counter_add($tag, 1, file!(), line!());
    };
}
#[cfg(not(any(debug_assertions, feature = "perf")))]
#[macro_export]
macro_rules! event {
    ($tag:expr) => {};
}

#[cfg(any(debug_assertions, feature = "perf"))]
#[macro_export]
macro_rules! timer_record {
    ($tag:expr, $dur:expr) => {
        $crate::metrics::__timer_record($tag, $dur, file!(), line!());
    };
}
#[cfg(not(any(debug_assertions, feature = "perf")))]
#[macro_export]
macro_rules! timer_record {
    ($tag:expr, $dur:expr) => {};
}

// ============================================================================
// §5 宏实现桥(全局未 init 时全部静默;non-debug 编译产物不引用)
// ============================================================================

#[doc(hidden)]
#[cfg(any(debug_assertions, feature = "perf"))]
pub fn __timer_start(tag: &'static str, file: &'static str, line: u32, _col: u32) {
    if let Some(s) = STORE.get() {
        s.timer_begin(tag, file, line);
    }
}

#[doc(hidden)]
#[cfg(any(debug_assertions, feature = "perf"))]
pub fn __timer_end(tag: &'static str, file: &'static str, line: u32, _col: u32) {
    if let Some(s) = STORE.get() {
        let _ = s.timer_end(tag, file, line);
    }
}

#[doc(hidden)]
#[cfg(any(debug_assertions, feature = "perf"))]
pub fn __counter_add(tag: &'static str, n: u64, file: &'static str, line: u32) {
    if let Some(s) = STORE.get() {
        s.counter_add(tag, n, file, line);
    }
}

#[doc(hidden)]
#[cfg(any(debug_assertions, feature = "perf"))]
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
        // 位置 = 本测试文件(宏展开点);断言对文件名鲁棒 —— 本模块
        // 历经 shared/metrics.rs → owl-metrics/lib.rs → shared/metrics.rs
        // 三次搬家(2026-10-01 定档独立模块),只锁尾部文件名不锁全路径
        assert!(
            r.last_file.ends_with("metrics.rs") || r.last_file.ends_with("lib.rs"),
            "file = {}",
            r.last_file
        );
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
        crate::log_debug!("dbg", "reps = {:?}", reps.iter().map(|r| (r.tag.clone(), r.count, r.counter)).collect::<Vec<_>>());
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

    // ── 性能框架新增面(docs/arch/perf-framework.md)──────────────

    /// 分级门:OFF 全关、COUNTERS 只收计数、HOST 收计时(store 本地级,
    /// 与全局静态无关 —— 测试并行安全)
    #[test]
    fn level_gates_timer_and_counter() {
        let s = MetricsStore::with_cap_level(64, level::OFF);
        s.timer_record("lvl.t", Duration::from_millis(1), file!(), line!());
        s.counter_add("lvl.c", 1, file!(), line!());
        s.timer_begin("lvl.s", file!(), line!());
        s.timer_end("lvl.s", file!(), line!()).expect("off 下 end 不报 orphan");
        assert!(s.collect(&MetricsFilter::new().tag_prefix("lvl.")).is_empty(), "OFF = 全关");

        s.set_level(level::COUNTERS);
        s.timer_record("lvl.t", Duration::from_millis(1), file!(), line!());
        s.counter_add("lvl.c", 1, file!(), line!());
        let r = s.collect(&MetricsFilter::new().tag_prefix("lvl."));
        assert_eq!(r.len(), 1, "COUNTERS 级只收计数");
        assert_eq!(r[0].tag, "lvl.c");
        assert_eq!(r[0].counter, 1);

        s.set_level(level::HOST);
        s.timer_record("lvl.t", Duration::from_millis(1), file!(), line!());
        assert_eq!(s.collect(&MetricsFilter::new().tag("lvl.t"))[0].count, 1, "HOST 收计时");
    }

    /// 父指针 + 排他时间:嵌套 A⊃B 时 B 的父 = A;A.self = A.total − B.total
    #[test]
    fn parent_tree_and_exclusive_self_time() {
        let s = MetricsStore::new();
        s.timer_begin("tr.A", file!(), line!());
        std::thread::sleep(Duration::from_millis(2));
        s.timer_begin("tr.B", file!(), line!());
        std::thread::sleep(Duration::from_millis(3));
        s.timer_end("tr.B", file!(), line!()).expect("B 配对");
        std::thread::sleep(Duration::from_millis(2));
        s.timer_end("tr.A", file!(), line!()).expect("A 配对");
        let reps = s.collect(&MetricsFilter::new().tag_prefix("tr."));
        let a = reps.iter().find(|r| r.tag == "tr.A").expect("A 有账");
        let b = reps.iter().find(|r| r.tag == "tr.B").expect("B 有账");
        // B 的父 = A(样本级调用树边)
        let bs = &s.inner.lock().unwrap().samples["tr.B"];
        assert_eq!(bs[0].parent.as_deref(), Some("tr.A"));
        // B.total ≈ B.self(无子);A.self < A.total(扣掉 B)
        assert_eq!(b.self_total, b.total, "叶子 self == total");
        assert!(a.self_total < a.total, "A 应扣掉子 B:{:?} vs {:?}", a.self_total, a.total);
        let diff = a.total - a.self_total;
        let diff_ms = diff.as_nanos() as f64 / 1e6;
        assert!((2.5..4.0).contains(&diff_ms), "扣除量应≈B 时长 3ms,实得 {diff_ms}");
    }

    /// 乱序配对:end 与栈顶不符但向下找到匹配 → 计数 out_of_order,不报 orphan
    #[test]
    fn out_of_order_end_is_tolerated_and_counted() {
        let s = MetricsStore::new();
        s.timer_begin("oo.A", file!(), line!());
        s.timer_begin("oo.B", file!(), line!());
        s.timer_end("oo.A", file!(), line!()).expect("向下找到 A");
        let r = s.collect(&MetricsFilter::new().tag("oo.A"));
        assert_eq!(r[0].count, 1);
        assert_eq!(r[0].out_of_order, 1);
        assert_eq!(r[0].orphan_ends, 0);
    }

    /// 树渲染 / rollup / 导出 smoke + epoch 过滤
    #[test]
    fn tree_rollup_export_and_epoch() {
        let s = MetricsStore::new();
        s.timer_begin("perf.phase.prefill", file!(), line!());
        s.timer_record("interp.host.node.marlin_gemm", Duration::from_millis(9), file!(), line!());
        s.timer_record("interp.host.node.owl_gdn_conv_fwd_f16", Duration::from_millis(3), file!(), line!());
        s.counter_add("interp.count.node", 12, file!(), line!());
        s.timer_end("perf.phase.prefill", file!(), line!()).expect("prefill 配对");
        let reps = s.collect(&MetricsFilter::new());
        let tree = render_tree(&reps, Duration::from_millis(12));
        assert!(tree.contains("marlin_gemm"), "树应含叶子:\n{tree}");
        let roll = rollup(&reps, 3);
        assert!(roll.iter().any(|r| r.tag == "interp.host.node"), "rollup 到第 3 段");
        let csv = s.export_csv(&MetricsFilter::new().tag_prefix("interp."));
        assert!(csv.starts_with("tag,count,counter"));
        assert!(csv.lines().count() >= 3, "表头 + ≥2 行");
        let js = s.export_json(&MetricsFilter::new().tag_prefix("interp."));
        assert!(js.contains("\"self_ms\"") && js.contains("\"level\""));
        // epoch:打标后旧样本不进新账
        std::thread::sleep(Duration::from_millis(2));
        s.mark_epoch();
        std::thread::sleep(Duration::from_millis(2));
        s.timer_record("interp.host.node.marlin_gemm", Duration::from_millis(1), file!(), line!());
        let ep = s.collect_since_epoch(MetricsFilter::new().tag_prefix("interp."));
        let m = ep.iter().find(|r| r.tag.ends_with("marlin_gemm")).expect("epoch 内有账");
        assert_eq!(m.count, 1, "epoch 只看标记之后的样本");
    }
}
