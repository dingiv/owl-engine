//! **单步 / 粘滞调试门(step gate)** —— 把解释器按"检查点"放慢或逐点暂停,
//! 供人工/外部程序观察引擎行为(2026-10-14 立项)。
//!
//! # 定位
//!
//! 与 [`crate::metrics`](性能账)与 [`crate::log`](事件流)同风格、不同用途:
//! 前两者是**观察**(只读),本模块是**干预**(让执行停下来)。三者共用同一套
//! 纪律:**宏 + 编译期 feature 门 + 运行期档位 + 全局单例(在线可切)**。
//!
//! # 两种模式(用户裁决 2026-10-14)
//!
//! | 模式 | 唤醒条件 | 用途 |
//! |---|---|---|
//! | `Sticky`(**粘滞**) | **定时器**(每检查点睡 `interval_ms`) | 把引擎整体放慢 N 倍,便于逐帧看日志/指标/显存,不改变执行序 |
//! | `Single`(**单步**) | **外部显式 step 信号**(tx/rx 通道) | 逐检查点手动推进:发一步走一步 |
//! | `Off` | — | 零干预(缺省) |
//!
//! # 零开销律(宏)
//!
//! 调用点一律写 [`step_gate!`]:
//!
//! ```ignore
//! owl_shared::step_gate!("interp.node").await;
//! ```
//!
//! * **不带 `stepgate` feature 的构建**:宏展开为 `core::future::ready(())`
//!   —— **参数不求值、无分支、无全局查询**,与 `perf`/`log` 的"生产零成本"
//!   同款(见 `Cargo.toml` 的 `stepgate` feature);
//! * 带 feature:展开为一次检查点调用(全局档位判定 + 命中计数 + 可选等待)。
//!
//! # 控制面(外部信号从哪来)
//!
//! * 启动参数:`--step-gate <off|sticky|single>` / `--step-gate-ms <n>` /
//!   `--step-gate-every <n>`,或配置 `[probes] step_gate*`(入口侧
//!   [`configure`] 落到本模块全局 —— 显式依赖律:热路径零 env 直读);
//! * HTTP(在线):`GET /debug/step`(状态)、`POST /debug/step/mode/<name>`、
//!   `POST /debug/step/release/<n>`(**发 step 信号**;`n = 1` 走一步,
//!   `n = 0`/`inf` = 放行不限步)、`POST /debug/step/ms/<n>`、
//!   `POST /debug/step/every/<n>`;
//! * 进程内:直接拿 [`control_tx`] 或 [`release`] / [`set_mode`]。
//!
//! # 纪律(踩过的坑,写进契约)
//!
//! 1. **boot 完成后才生效**:boot 期的 warmup / 图捕获也会跑解释器,而那时
//!    HTTP 面还没起来 ⇒ 若一开始就 Single 会**自锁**。故 `configure` 只在
//!    入口的"就绪"之后调用(见 `apps/server`),状态面用 `armed` 标明;
//! 2. **通道关闭 = 放行**(`recv()` 得 `None` 即返回):宁可少等一步,
//!    也不把引擎焊死;
//! 3. **等待是 async 的**(`tokio::sync::Mutex` + `Notify`/`mpsc`),绝不阻塞
//!    运行时线程 —— 否则 HTTP 面无法回送 step 信号 = 死锁;
//! 4. 指标键:`stepgate.hits`(命中)、`stepgate.waits`(真等了几次)、
//!    `stepgate.wait`(每次等待时延,按 `file!/line!` 分站点)、
//!    `stepgate.released`(外部放行的步数)。

// ============================================================================
// 模式 / 控制消息(两种构建共用:类型面无 feature 门,消费方零分叉)
// ============================================================================

/// 调试门档位(`off` / `sticky` / `single`;枚举严格 —— 非法值拒启)
///
/// 配置面反序列化用 serde 严格名(`off|sticky|single`);[`StepMode::parse`]
/// 另接受少量别名(CLI/HTTP 用,不放松配置面)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepMode {
    /// 零干预(缺省)
    #[default]
    Off,
    /// **粘滞**:每个检查点按定时器睡 `interval_ms`
    Sticky,
    /// **单步**:每个检查点等外部 `step` 信号
    Single,
}

impl StepMode {
    /// 名称 → 档(`--step-gate` 与 `/debug/step/mode/<name>` 共用)
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "off" | "none" => Self::Off,
            "sticky" | "sticky-timer" => Self::Sticky,
            "single" | "step" => Self::Single,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Sticky => "sticky",
            Self::Single => "single",
        }
    }

    #[cfg_attr(not(feature = "stepgate"), allow(dead_code))]
    fn code(self) -> u8 {
        match self {
            Self::Off => 0,
            Self::Sticky => 1,
            Self::Single => 2,
        }
    }

    #[cfg_attr(not(feature = "stepgate"), allow(dead_code))]
    fn from_code(c: u8) -> Self {
        match c {
            1 => Self::Sticky,
            2 => Self::Single,
            _ => Self::Off,
        }
    }
}

/// tx/rx 通道载荷 = **纯唤醒信号**(不携带状态)。
///
/// 为什么不带载荷(2026-10-14 实测踩中):载荷式消息会在通道里**排队**,
/// 一个后来的检查点可能吃到**早先的**旧档位/旧放行 ⇒ "翻旧账"(实测:切回
/// single 后,请求跑到第 3 个检查点吃到更早的 `Mode(off)` ⇒ 整轮跑飞)。
/// 因此**状态唯一住址 = 原子量**(`mode`/`credits`/`interval`/`every`),
/// 通道只负责"叫醒在等的那一个";被叫醒者一律**重读**当前状态。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepMsg {
    /// 叫醒在等的检查点(状态由原子量重读)
    Wake,
}

/// 状态快照(`GET /debug/step`;两种构建同型,`compiled=false` = 未编入 feature)
#[derive(Clone, Debug, Default)]
pub struct StepState {
    pub mode: &'static str,
    pub every: u64,
    pub interval_ms: u64,
    /// 亚毫秒档(µs;>0 时覆盖 interval_ms)
    pub interval_us: u64,
    pub hits: u64,
    /// 真正**通过**门的检查点数(等待结束/吃到信用/被 every 过滤掉都算通过)
    pub advanced: u64,
    pub waits: u64,
    pub waited_ms: u64,
    pub credits: u64,
    pub released: u64,
    pub waiting: bool,
    /// 已由入口 `configure` 武装(boot 完成前为 false —— 见模块头纪律 1)
    pub armed: bool,
    /// 本二进制是否编入 `stepgate` feature
    pub compiled: bool,
}

// ============================================================================
// 宏(编译期门;两种构建都导出,调用点零分叉)
// ============================================================================

/// 打一个检查点(见模块头)。**必须 `.await`**:
/// `owl_shared::step_gate!("interp.node").await;`
#[cfg(feature = "stepgate")]
#[macro_export]
macro_rules! step_gate {
    ($tag:expr) => {
        $crate::stepgate::checkpoint($tag, file!(), line!())
    };
}

/// 生产档(`stepgate` feature 未开):**空展开**,参数不求值、零运行时开销。
#[cfg(not(feature = "stepgate"))]
#[macro_export]
macro_rules! step_gate {
    ($($tt:tt)*) => {
        core::future::ready(())
    };
}

// ============================================================================
// 实现档(feature on)
// ============================================================================

#[cfg(feature = "stepgate")]
mod imp {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
    use std::sync::OnceLock;
    use std::time::Duration;
    use tokio::sync::{mpsc, Mutex};

    struct Gate {
        mode: AtomicU8,
        every: AtomicU64,
        /// 间隔内部单位 = **微秒**(`step_gate_ms` 写 1000×;亚毫秒档走自旋)
        interval_us: AtomicU64,
        hits: AtomicU64,
        advanced: AtomicU64,
        waits: AtomicU64,
        waited_us: AtomicU64,
        released: AtomicU64,
        credits: AtomicU64,
        waiting: AtomicBool,
        armed: AtomicBool,
        tx: mpsc::UnboundedSender<StepMsg>,
        rx: Mutex<Option<mpsc::UnboundedReceiver<StepMsg>>>,
    }

    fn gate() -> &'static Gate {
        static G: OnceLock<Gate> = OnceLock::new();
        G.get_or_init(|| {
            let (tx, rx) = mpsc::unbounded_channel();
            Gate {
                mode: AtomicU8::new(StepMode::Off.code()),
                every: AtomicU64::new(1),
                interval_us: AtomicU64::new(50_000),
                hits: AtomicU64::new(0),
                advanced: AtomicU64::new(0),
                waits: AtomicU64::new(0),
                waited_us: AtomicU64::new(0),
                released: AtomicU64::new(0),
                credits: AtomicU64::new(0),
                waiting: AtomicBool::new(false),
                armed: AtomicBool::new(false),
                tx,
                rx: Mutex::new(Some(rx)),
            }
        })
    }

    impl Gate {
        fn mode(&self) -> StepMode {
            StepMode::from_code(self.mode.load(Ordering::Relaxed))
        }
        fn set_mode(&self, m: StepMode) {
            self.mode.store(m.code(), Ordering::Relaxed);
        }
        fn every(&self) -> u64 {
            self.every.load(Ordering::Relaxed).max(1)
        }
        fn interval(&self) -> Duration {
            Duration::from_micros(self.interval_us.load(Ordering::Relaxed))
        }
        /// 粘滞等待:**≥1ms 走 tokio 定时器;亚毫秒走自旋**
        /// (tokio 定时器粒度 ≈1ms ⇒ 300µs 会被抬到 1ms;自旋才准)
        async fn sticky_wait(&self, rx: &mut mpsc::UnboundedReceiver<StepMsg>) {
            let d = self.interval();
            if d >= Duration::from_millis(1) {
                let _ = tokio::time::timeout(d, rx.recv()).await;
            } else {
                let deadline = std::time::Instant::now() + d;
                while std::time::Instant::now() < deadline {
                    std::hint::spin_loop();
                }
            }
        }
        /// 放行 `n` 步进信用池(`0` = 不限步;`consumed_now` = 本检查点自吃 1 步)
        fn grant(&self, n: u64, consumed_now: bool) {
            self.released.fetch_add(n, Ordering::Relaxed);
            if n == 0 {
                self.credits.store(u64::MAX / 2, Ordering::Relaxed);
            } else {
                let add = if consumed_now { n.saturating_sub(1) } else { n };
                self.credits.fetch_add(add, Ordering::Relaxed);
            }
        }

        /// 吃一个信用(返回是否吃到)
        fn take_credit(&self) -> bool {
            let mut c = self.credits.load(Ordering::Relaxed);
            while c > 0 {
                match self.credits.compare_exchange_weak(
                    c,
                    c - 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return true,
                    Err(now) => c = now,
                }
            }
            false
        }
    }

    fn tim(tag: &'static str, d: Duration) {
        if crate::metrics::level() == crate::metrics::level::OFF {
            return;
        }
        crate::metrics::with_metrics_store(|s| s.timer_record_tag(tag, d, file!(), line!()));
    }

    /// **检查点**(宏展开目标;见模块头)。参数 `file/line` 由宏注入 ⇒ 站点可辨。
    pub async fn checkpoint(tag: &'static str, file: &'static str, line: u32) {
        let g = gate();
        if g.mode() == StepMode::Off {
            return;
        }
        let n = g.hits.fetch_add(1, Ordering::Relaxed) + 1;
        if crate::metrics::level() != crate::metrics::level::OFF {
            crate::metrics::with_metrics_store(|s| s.counter_add("stepgate.hits", 1, file, line));
        }
        let every = g.every();
        if every > 1 && !n.is_multiple_of(every) {
            g.advanced.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let _ = tag; // 站点已由 file/line 携带;tag 供日志/扩展
        // 单步:先吃"预放行信用"(release/<n> 立即入池),信用耗尽才真等
        if g.take_credit() {
            g.advanced.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let t0 = std::time::Instant::now();
        g.waits.fetch_add(1, Ordering::Relaxed);
        g.waiting.store(true, Ordering::Relaxed);
        match g.mode() {
            StepMode::Off => {}
            StepMode::Sticky => {
                // 定时器唤醒;同时监听唤醒通道(改档/放行能打断长睡眠)。
                // 用 timeout 而非 select!(不引入 tokio `macros` feature)。
                // 唤醒消息**不携带状态** ⇒ 内容无关,仅用于提前结束睡眠。
                let mut guard = g.rx.lock().await;
                let rx = guard.as_mut().expect("rx 唯一消费者(解释器侧)");
                g.sticky_wait(rx).await;
                // 超时 = 粘滞唤醒;通道关闭 = 放行(纪律 2)
            }
            StepMode::Single => {
                // 唤醒循环:**每次醒来都重读当前状态**(档位/信用)—— 通道不带
                // 载荷,所以不存在"翻旧账"(见 StepMsg 注)。
                let mut guard = g.rx.lock().await;
                let rx = guard.as_mut().expect("rx 唯一消费者(解释器侧)");
                loop {
                    if g.mode() != StepMode::Single {
                        break; // 被外部切档 ⇒ 放行
                    }
                    if g.take_credit() {
                        break; // 有信用(放行 N 步中的一步)⇒ 走
                    }
                    match rx.recv().await {
                        None => break, // 纪律 2:通道关闭 = 放行
                        Some(StepMsg::Wake) => continue, // 重读状态再判
                    }
                }
            }
        }
        g.waiting.store(false, Ordering::Relaxed);
        g.advanced.fetch_add(1, Ordering::Relaxed);
        let d = t0.elapsed();
        g.waited_us
            .fetch_add(d.as_micros() as u64, Ordering::Relaxed);
        tim("stepgate.wait", d);
    }

    /// 外部控制发送端(HTTP / 测试用;可克隆)
    pub fn control_tx() -> mpsc::UnboundedSender<StepMsg> {
        gate().tx.clone()
    }

    /// 发一个控制消息(失败 = 引擎侧消费者已退出,静默 —— 控制面不许炸)
    pub fn send(msg: StepMsg) -> bool {
        gate().tx.send(msg).is_ok()
    }

    /// 便捷:放行 `n` 步(`0` = 不限步)。**立即生效**(直接进信用池)+ 唤醒等待者。
    pub fn release(n: u64) -> bool {
        gate().grant(n, false);
        send(StepMsg::Wake)
    }

    /// 便捷:改档。**立即生效**(状态面同步更新)+ 唤醒等待者。
    pub fn set_mode(m: StepMode) -> bool {
        gate().set_mode(m);
        send(StepMsg::Wake)
    }

    /// 便捷:改粘滞间隔(ms;立即生效 + 唤醒)
    pub fn set_interval_ms(ms: u64) -> bool {
        gate().interval_us.store(ms.max(1) * 1000, Ordering::Relaxed);
        send(StepMsg::Wake)
    }

    /// 便捷:改粘滞间隔(**µs**;亚毫秒档走自旋)
    pub fn set_interval_us(us: u64) -> bool {
        gate().interval_us.store(us.max(1), Ordering::Relaxed);
        send(StepMsg::Wake)
    }

    /// 便捷:改"每 N 个检查点停一次"(立即生效 + 唤醒)
    pub fn set_every(n: u64) -> bool {
        gate().every.store(n.max(1), Ordering::Relaxed);
        send(StepMsg::Wake)
    }

    /// 入口侧武装(见模块头纪律 1:boot 完成后调用)。
    /// `interval_us = 0` ⇒ 用 `interval_ms × 1000`;>0 ⇒ 以 µs 为准(亚毫秒档)。
    pub fn configure(mode: StepMode, interval_ms: u64, every: u64) {
        configure_ex(mode, interval_ms, 0, every)
    }

    /// 带 µs 覆盖的武装
    pub fn configure_ex(mode: StepMode, interval_ms: u64, interval_us: u64, every: u64) {
        let g = gate();
        g.set_mode(mode);
        let us = if interval_us > 0 {
            interval_us.max(1)
        } else {
            interval_ms.max(1) * 1000
        };
        g.interval_us.store(us, Ordering::Relaxed);
        g.every.store(every.max(1), Ordering::Relaxed);
        g.armed.store(true, Ordering::Relaxed);
    }

    /// 状态快照
    pub fn state() -> StepState {
        let g = gate();
        StepState {
            mode: g.mode().name(),
            every: g.every(),
            interval_ms: g.interval_us.load(Ordering::Relaxed) / 1000,
            interval_us: g.interval_us.load(Ordering::Relaxed),
            hits: g.hits.load(Ordering::Relaxed),
            advanced: g.advanced.load(Ordering::Relaxed),
            waits: g.waits.load(Ordering::Relaxed),
            waited_ms: g.waited_us.load(Ordering::Relaxed) / 1000,
            credits: {
                let c = g.credits.load(Ordering::Relaxed);
                if c > u64::MAX / 4 {
                    0
                } else {
                    c
                }
            },
            released: g.released.load(Ordering::Relaxed),
            waiting: g.waiting.load(Ordering::Relaxed),
            armed: g.armed.load(Ordering::Relaxed),
            compiled: true,
        }
    }

    /// 复位计数(测试/长跑用)
    pub fn reset() {
        let g = gate();
        g.hits.store(0, Ordering::Relaxed);
        g.advanced.store(0, Ordering::Relaxed);
        g.waits.store(0, Ordering::Relaxed);
        g.waited_us.store(0, Ordering::Relaxed);
        g.released.store(0, Ordering::Relaxed);
        g.credits.store(0, Ordering::Relaxed);
    }
}

// ============================================================================
// 生产档(feature off):同型 API,全空实现 —— 消费方(server)零分叉
// ============================================================================

#[cfg(not(feature = "stepgate"))]
mod imp {
    use super::*;

    /// 检查点空实现(宏在无 feature 时根本不调用它;留作直接调用方的兜底)
    pub async fn checkpoint(_tag: &'static str, _file: &'static str, _line: u32) {}

    pub fn control_tx() -> Option<()> {
        None
    }

    pub fn send(_msg: StepMsg) -> bool {
        false
    }

    pub fn release(_n: u64) -> bool {
        false
    }

    pub fn set_mode(_m: StepMode) -> bool {
        false
    }

    pub fn set_interval_ms(_ms: u64) -> bool {
        false
    }

    pub fn set_every(_n: u64) -> bool {
        false
    }

    pub fn set_interval_us(_us: u64) -> bool {
        false
    }

    /// 带 µs 覆盖的武装(生产档 = 同 configure)
    pub fn configure_ex(mode: StepMode, interval_ms: u64, interval_us: u64, every: u64) {
        configure(mode, if interval_us > 0 { interval_us.div_ceil(1000) } else { interval_ms }, every)
    }

    /// 未编入 feature 时的 configure:**非 off 档要可见地报出来**(不静默失效)
    pub fn configure(mode: StepMode, interval_ms: u64, every: u64) {
        if mode != StepMode::Off {
            crate::log_warn!(
                "stepgate",
                "配置要求调试门 mode={} ms={} every={},但本二进制未编入 `stepgate` feature \
                 —— 需以 `--features stepgate` 构建(见 docs/arch/perf-framework.md)",
                mode.name(),
                interval_ms,
                every
            );
        }
    }

    pub fn state() -> StepState {
        StepState {
            mode: "off",
            every: 1,
            interval_ms: 0,
            interval_us: 0,
            compiled: false,
            ..Default::default()
        }
    }

    pub fn reset() {}
}

pub use imp::*;

// ============================================================================
// 测试(最小代码;`--features stepgate`)
// ============================================================================

#[cfg(all(test, feature = "stepgate"))]
mod tests {
    use super::*;
    use std::time::Duration;

    /// 全局门是**单例** ⇒ 测试必须串行(否则互相偷 step 信用/偷档位);
    /// 再加一层硬超时:任何等待都不得把测试挂死。
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    async fn with_gate<F, Fut>(f: F)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        tokio::time::timeout(Duration::from_secs(10), f())
            .await
            .expect("stepgate 测试超时(10s)—— 门被卡住");
        configure(StepMode::Off, 1, 1);
    }

    /// 单步:发一步走一步(命中计数严格对齐),信用机制不丢步
    #[tokio::test]
    async fn single_step_advances_exactly_on_release() {
        with_gate(|| async {

            reset();
            configure(StepMode::Single, 1, 1);
            let h = tokio::spawn(async {
                for _ in 0..3 {
                    crate::step_gate!("test").await;
                }
            });
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert_eq!(state().advanced, 0, "未放行前不应有任何检查点通过(已进入并阻塞:hits=1)");
            assert_eq!(state().hits, 1, "第一个检查点应已进入并阻塞");
            release(1);
            tokio::time::sleep(Duration::from_millis(50)).await;
            assert_eq!(state().advanced, 1, "放行 1 步只应通过 1 个检查点");
            release(2);
            tokio::time::timeout(Duration::from_secs(2), h)
                .await
                .expect("放行 2 步后应跑完")
                .expect("task");
            let s = state();
            assert_eq!(s.advanced, 3, "3 个检查点全部通过");
            assert_eq!(s.released, 3, "外部放行 3 步");
            configure(StepMode::Off, 1, 1);
        })
        .await;
    }

    /// 粘滞:定时器唤醒,总耗时 ≈ 检查点数 × 间隔
    #[tokio::test]
    async fn sticky_sleeps_per_checkpoint() {
        with_gate(|| async {

            reset();
            configure(StepMode::Sticky, 20, 1);
            let t0 = std::time::Instant::now();
            for _ in 0..3 {
                crate::step_gate!("test").await;
            }
            let el = t0.elapsed();
            assert!(el >= Duration::from_millis(55), "3×20ms 应 ≥55ms,实得 {el:?}");
            assert_eq!(state().hits, 3);
            configure(StepMode::Off, 1, 1);
        })
        .await;
    }

    /// every=N:只有每 N 个检查点才等
    #[tokio::test]
    async fn every_n_coarsens() {
        with_gate(|| async {

            reset();
            configure(StepMode::Sticky, 15, 3);
            let t0 = std::time::Instant::now();
            for _ in 0..6 {
                crate::step_gate!("test").await;
            }
            let el = t0.elapsed();
            assert!(el >= Duration::from_millis(25), "6 个检查点每 3 停 1 ⇒ ≥30ms 量级,实得 {el:?}");
            assert_eq!(state().hits, 6);
            assert_eq!(state().waits, 2, "6/3 = 2 次真等");
            configure(StepMode::Off, 1, 1);
        })
        .await;
    }

    /// **回归**:控制消息不得"翻旧账"(2026-10-14 真机踩中)—— 先 off→single
    /// 再 release,通道里会留下多条唤醒;**后来的检查点不得被早先的档位覆盖**。
    #[tokio::test]
    async fn stale_control_does_not_clobber_mode() {
        with_gate(|| async {
            reset();
            configure(StepMode::Single, 1, 1);
            set_mode(StepMode::Off);
            set_mode(StepMode::Single);
            release(2);
            // 3 个检查点:前 2 个吃信用,第 3 个应**阻塞**(且档位仍是 single)
            let h = tokio::spawn(async {
                for _ in 0..3 {
                    crate::step_gate!("test").await;
                }
            });
            tokio::time::sleep(Duration::from_millis(80)).await;
            let s = state();
            assert_eq!(s.mode, "single", "档位不得被通道里的旧消息改回(翻旧账)");
            assert_eq!(s.advanced, 2, "只有 2 步信用 ⇒ 应通过 2 个");
            assert!(s.waiting, "第 3 个应阻塞等待");
            release(0);
            tokio::time::timeout(Duration::from_secs(2), h)
                .await
                .expect("release/0 后应跑完")
                .expect("task");
            assert_eq!(state().advanced, 3);
        })
        .await;
    }

    /// off 档:零等待(且不再计数)
    #[tokio::test]
    async fn off_is_free() {
        with_gate(|| async {

            reset();
            configure(StepMode::Off, 500, 1);
            let t0 = std::time::Instant::now();
            for _ in 0..100 {
                crate::step_gate!("test").await;
            }
            assert!(t0.elapsed() < Duration::from_millis(50), "off 档不许有任何等待");
            assert_eq!(state().hits, 0);
        })
        .await;
    }

    /// 指标面:hits/waits 进 metrics store(性能框架同口径)
    #[tokio::test]
    async fn metrics_records_gate_activity() {
        with_gate(|| async {

            reset();
            crate::metrics::init_metrics(crate::metrics::MetricsStore::default()).ok();
            configure(StepMode::Sticky, 1, 1);
            for _ in 0..4 {
                crate::step_gate!("test").await;
            }
            configure(StepMode::Off, 1, 1);
            let reports = crate::metrics::query_metrics(&crate::metrics::MetricsFilter::new());
            let hit = reports
                .iter()
                .find(|r| r.tag.contains("stepgate.hits"))
                .expect("应有 stepgate.hits 指标");
            assert_eq!(hit.counter, 4, "4 个检查点 ⇒ counter=4(与 state().hits 同口径)");
            let wait = reports.iter().find(|r| r.tag == "stepgate.wait");
            assert!(wait.is_some_and(|r| r.count == 4), "4 次等待各有计时样本");
        })
        .await;
    }
}
