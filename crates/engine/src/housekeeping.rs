//! **引擎保洁账**(2026-10-21 S2 收户)。
//!
//! 原本摊在 `RunningEngine` 上的六个字段(调度计时 / 空转节拍 / 池缓存回收
//! 节拍 / 降级计数 / 合批步数)收成一户:**节拍判据**与**计数语义**住在这里,
//! `pump` 只剩"到点了吗 / 记一笔"。
//!
//! 纪律:字段私有;读 = 具名访问器;写 = `note_*` / `mark_*`。
//! 本户零 I/O、零设备知识(只记账与判节拍),因此可单测。

use std::time::{Duration, Instant};

/// 空转整理节拍(pump 拍数;H5)。
const IDLE_HK_EVERY: u64 = 16;
/// 池缓存回收节拍(墙钟):server 泵循环 `Idle` 即断,`StoreHook::idle` 到不了,
/// 所以在空转出口按时间节流清一次(§23.9)。
const IDLE_TRIM_EVERY: Duration = Duration::from_millis(500);

/// 保洁 / 遥测账(跨 turn 累计;纯数据,Default = 冷启动)。
#[derive(Default)]
pub(crate) struct Housekeeping {
    /// E5 性能:schedule 累计账(轮间 host 成本剖析)
    sched_acc_ns: u64,
    sched_cnt: usize,
    /// H5 空转整理节拍(空转 pump 计数;每 `IDLE_HK_EVERY` 拍整理一次)
    idle_pumps: u64,
    /// **默认池缓存回收的节拍**(§23.9:turn 收尾之后才 free 的页要在空转期
    /// 再清一次;失败只 WARN 不打断空转)。
    last_trim: Option<Instant>,
    /// N6/保洁 deny 的**累计**次数(限流日志用:首条 + 每 64 条打一次;
    /// 计数本身**不丢** —— 机读面 = `store.degrade` metrics)
    degrade_seen: u64,
    /// **合批 decode 已执行步数**(B5.1 事实位;门用它证明批路径真被走到)
    batch_steps: u64,
}

impl Housekeeping {
    /// schedule 计时入账。
    ///
    /// 返回 `Some((样本数, 平均 ms, 本次 ms))` = 到打点节拍(`SCHED_LOG_EVERY`),
    /// 调用方照原样打一行;`None` = 本拍不打(判据不散在调用点)。
    pub(crate) fn note_sched(&mut self, elapsed: Duration) -> Option<(usize, f64, f64)> {
        self.sched_acc_ns += elapsed.as_nanos() as u64;
        self.sched_cnt += 1;
        if !self.sched_cnt.is_multiple_of(SCHED_LOG_EVERY) {
            return None;
        }
        let avg_ms = self.sched_acc_ns as f64 / self.sched_cnt as f64 / 1e6;
        Some((self.sched_cnt, avg_ms, elapsed.as_secs_f64() * 1e3))
    }

    /// 空转一拍;返回 `true` = 本拍到点(调用方调 `StoreHook::idle` 整理)。
    pub(crate) fn note_idle_pump(&mut self) -> bool {
        self.idle_pumps = self.idle_pumps.wrapping_add(1);
        self.idle_pumps.is_multiple_of(IDLE_HK_EVERY)
    }

    /// 池缓存回收到点?(首拍 / 距上次 ≥ `IDLE_TRIM_EVERY`)
    pub(crate) fn trim_due(&self) -> bool {
        self.last_trim
            .is_none_or(|t| t.elapsed() >= IDLE_TRIM_EVERY)
    }

    /// 记一次回收节拍(`trim_due` 判真后立刻调)。
    pub(crate) fn mark_trim(&mut self) {
        self.last_trim = Some(Instant::now());
    }

    /// **作废节拍**:turn 收尾路径里还有一批大额临时块要在 TurnEnd 之后才
    /// free ⇒ 让紧随其后的 Idle 出口**再清一次**(不等 500ms)。
    pub(crate) fn expire_trim(&mut self) {
        self.last_trim = None;
    }

    /// 真容量不足(degrade)记一笔;返回累计数。
    ///
    /// 日志限流判据(`n == 1 || n % 64 == 0`)留在调用点 —— 它同时决定
    /// **打什么级别/什么内容**,属于日志策略,不属于账。
    pub(crate) fn note_degrade(&mut self) -> u64 {
        self.degrade_seen = self.degrade_seen.saturating_add(1);
        self.degrade_seen
    }

    /// 合批 decode 走了一步(eager 或图回放;事实位)。
    pub(crate) fn note_batch_step(&mut self) {
        self.batch_steps += 1;
    }

    /// 合批 decode 已执行步数。
    pub(crate) fn batch_steps(&self) -> u64 {
        self.batch_steps
    }
}

/// schedule 打点节拍(每 N 次 schedule 打一行;原实现硬编码 16)。
const SCHED_LOG_EVERY: usize = 16;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_due_is_true_at_first_tick_then_throttled() {
        let mut hk = Housekeeping::default();
        assert!(hk.trim_due(), "首拍必须到点(冷启动没有 last_trim)");
        hk.mark_trim();
        assert!(!hk.trim_due(), "刚记过节拍,500ms 内不该再清");
        hk.expire_trim();
        assert!(hk.trim_due(), "expire 之后必须立刻到点(turn 收尾路径)");
    }

    #[test]
    fn idle_tick_fires_every_nth_pump() {
        let mut hk = Housekeeping::default();
        for i in 1..IDLE_HK_EVERY {
            assert!(!hk.note_idle_pump(), "第 {i} 拍不该整理");
        }
        assert!(hk.note_idle_pump(), "第 {IDLE_HK_EVERY} 拍该整理");
    }

    #[test]
    fn sched_accounting_reports_only_on_tick() {
        let mut hk = Housekeeping::default();
        let d = Duration::from_millis(2);
        for i in 1..SCHED_LOG_EVERY {
            assert!(hk.note_sched(d).is_none(), "第 {i} 次不该打点");
        }
        let (n, avg, last) = hk.note_sched(d).expect("第 SCHED_LOG_EVERY 次打点");
        assert_eq!(n, SCHED_LOG_EVERY);
        assert!((avg - 2.0).abs() < 1e-6, "平均 = 2ms,实得 {avg}");
        assert!((last - 2.0).abs() < 1e-6, "本次 = 2ms,实得 {last}");
    }

    #[test]
    fn degrade_counter_saturates_and_reports() {
        let mut hk = Housekeeping::default();
        assert_eq!(hk.note_degrade(), 1);
        assert_eq!(hk.note_degrade(), 2);
        hk.note_batch_step();
        hk.note_batch_step();
        assert_eq!(hk.batch_steps(), 2);
    }
}
