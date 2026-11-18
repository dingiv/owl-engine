//! **压力 / 腾挪策略**（store 自持；2026-11-16 用户裁决）。
//!
//! # 为什么在这里（而不是 `[runtime]` 的配置键）
//!
//! 用户裁决原话："这些关于压力的东西不应该放在配置里面，它应该放在 store
//! 里面去，让 store 来去做统一的调度。"
//!
//! 三个理由，逐条对应旧键：
//!
//! 1. **水位是池几何的函数**：`12% / 25%` 说的是"主池页数的占比" —— 池多大
//!    是 store 分配的，操作员既算不出也不该算；
//! 2. **每次腾多少是 store 的延迟预算**：一次腾挪 = N 条设备命令，
//!    单次上限（64 / 16 / 64）保护的是**在跑 turn 的工作集**与尾延迟，
//!    不是容量策略；
//! 3. **空转要不要整理由真实余量回答**：低水位才触发（滞回补到高水位），
//!    余量健康时恒为 0 ⇒ 不需要一个"开/关"旋钮。
//!
//! 引擎只做两件事：**读现场信号**（[`PressureSignals`]，纯 host 账、零 I/O）
//! 和**执行**本模块给出的目标量。

use std::time::Duration;

// ===========================================================================
// §1 现场信号（引擎注入；纯数据）
// ===========================================================================

/// **压力信号快照**：把"池/宿主/磁盘/驱动"的余量一次读齐。
///
/// 读的时候全部是纯 host 账；不触发 I/O，也不改任何状态。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PressureSignals {
    /// 主池（KV 页池）自由页。
    pub pool_free_pages: usize,
    /// 主池总页数（水位的分母）。
    pub pool_total_pages: usize,
    /// KV 宿主腿 + 状态腿的剩余槽（合计；用于判断是否还能下沉）。
    pub host_free_slots: usize,
    /// KV 宿主腿是否启用（`host_bytes > 0`）。
    pub host_kv_enabled: bool,
    /// 状态（快照）宿主腿是否启用。
    pub host_snap_enabled: bool,
    /// KV 宿主腿总槽数（水位分母）。
    pub host_kv_total_slots: usize,
    /// 状态宿主腿总槽数（水位分母）。
    pub host_snap_total_slots: usize,
    /// KV 宿主腿剩余槽。
    pub host_kv_free_slots: usize,
    /// 状态宿主腿剩余槽。
    pub host_snap_free_slots: usize,
    /// KV 盘腿 + 状态盘腿的剩余槽（合计）。
    pub disk_free_slots: usize,
    /// KV 盘腿剩余槽（宿主腿下沉的去处）。
    pub disk_kv_free_slots: usize,
    /// 状态盘腿剩余槽。
    pub disk_snap_free_slots: usize,
    /// 最紧一张卡的驱动 free（TP2 取两 rank 的 min）。
    pub driver_free_bytes: u64,
    /// boot 时为驱动的硬墙留的预留（`[store] vram_activation_reserve`）。
    pub reserve_bytes: u64,
    // 以上字段全是**纯 host 账**，引擎每拍读一次；本结构零 I/O。
}

impl PressureSignals {
    /// 驱动余量已低于 boot 预留。
    pub fn driver_below_reserve(&self) -> bool {
        self.driver_free_bytes < self.reserve_bytes
    }

    /// 一行诊断（日志/报告；不参与判据）。
    pub fn summary(&self) -> String {
        format!(
            "pool {}/{} 页 · host {} 槽 · disk {} 槽 · driver {:.0}MiB/reserve {:.0}MiB{}",
            self.pool_free_pages,
            self.pool_total_pages,
            self.host_free_slots,
            self.disk_free_slots,
            self.driver_free_bytes as f64 / 1048576.0,
            self.reserve_bytes as f64 / 1048576.0,
            if self.driver_below_reserve() {
                " · 驱动低于预留"
            } else {
                ""
            }
        )
    }
}

// ===========================================================================
// §2 策略（**无旋钮**：全部由本模块定值）
// ===========================================================================

/// **主池低水位**（占池总页数 %）：自由页低于它才算"池在压力下"。
///
/// 12% 的依据：单 turn 的 prefill 片（2k tok = 64 页）与一次生成扩张
/// （8192 tok = 256 页）都要从自由页里立刻拿 —— 池低到这个量级时下一次
/// `ensure_for_len` 必然撞墙，先腾比撞墙再腾便宜。
const POOL_LOW_PCT: usize = 12;

/// **主池目标水位**（%）：触发后把自由页补到这里，形成滞回，
/// 避免在阈值附近每拍抖动。
const POOL_HIGH_PCT: usize = 25;

/// 宿主腿低水位 / 目标水位（% of 该腿总槽）。与主池同口径：两条腿
/// 各自的"再沉就写不下"点。
const HOST_LOW_PCT: usize = 12;
const HOST_HIGH_PCT: usize = 25;

/// **应急收缩**（`PoolExhausted` 后）单次腾槽上限。
///
/// 上限存在的理由不是容量，是**在跑 turn 的工作集**：一口气搬空会把它
/// 冲掉（每块搬走都要回灌）。64 页 ≈ 2k token 的 KV。
const RELIEF_MAX_SLOTS: usize = 64;

/// **空转整理**单次腾槽上限（更小：没有人在等，不需要一次到位）。
const IDLE_MAX_SLOTS: usize = 16;

/// 单次 HostPressure 的**每请求**配额（KV 前缀块 / 快照条数）。
///
/// 一次搬迁是一串设备命令 + 一次宿主写；配额 = 尾延迟预算。快照一条
/// 就是 ~159MiB 量级，所以按"条"而不是按字节。
const HOST_REQUEST_SLOTS: usize = 16;
const HOST_REQUEST_SNAPS: usize = 1;

/// 单次 HostPressure 的绝对上限（防止低水位目标过大时一次搬空）。
const HOST_MAX_SLOTS: usize = 64;

/// 空转 HostPressure 的最小间隔（节流；应急收缩不节流）。
const HOST_INTERVAL: Duration = Duration::from_millis(1000);

/// 压力/腾挪策略（**零字段**：没有任何东西需要操作员告诉它）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PressurePolicy;

impl PressurePolicy {
    /// 构造（零参数：策略没有任何可调项）。
    pub fn new() -> Self {
        Self
    }

    fn slots_at(total: usize, pct: usize) -> usize {
        if total == 0 {
            return 0;
        }
        (total.saturating_mul(pct) / 100).clamp(1, total)
    }

    /// 低水位（页）。
    pub fn pool_low_slots(&self, total: usize) -> usize {
        Self::slots_at(total, POOL_LOW_PCT)
    }

    /// 目标水位（页；≥ 低水位）。
    pub fn pool_high_slots(&self, total: usize) -> usize {
        Self::slots_at(total, POOL_HIGH_PCT).max(self.pool_low_slots(total))
    }

    /// 宿主腿低水位（按该腿总槽数）。
    pub fn host_low_slots(&self, total: usize) -> usize {
        Self::slots_at(total, HOST_LOW_PCT)
    }

    /// 宿主腿目标水位（触发后补到该值；≥ 低水位）。
    pub fn host_high_slots(&self, total: usize) -> usize {
        Self::slots_at(total, HOST_HIGH_PCT).max(self.host_low_slots(total))
    }

    /// **应急收缩目标**（`PoolExhausted` 后）：`free ≥ low` ⇒ 0（不搬）；
    /// 否则补到 high，单次不超过 [`RELIEF_MAX_SLOTS`]。
    pub fn relief_pool_slots(&self, total: usize, free: usize) -> usize {
        self.pool_target(total, free, RELIEF_MAX_SLOTS)
    }

    /// **空转整理目标**：判据同上（低触发 → 高目标），单次上限更小。
    pub fn idle_pool_slots(&self, total: usize, free: usize) -> usize {
        self.pool_target(total, free, IDLE_MAX_SLOTS)
    }

    fn pool_target(&self, total: usize, free: usize, cap: usize) -> usize {
        if free >= self.pool_low_slots(total) {
            return 0;
        }
        self.pool_high_slots(total)
            .saturating_sub(free)
            .min(cap)
    }

    /// **宿主腿目标**（KV 前缀块数 / 快照条数；已含每请求配额与绝对上限）。
    ///
    /// 触发条件 = 该腿**启用 + 低于低水位 + 盘上有位置**；三条缺一即 0
    /// （没启用的腿不搬、够用的腿不搬、无盘可落的腿搬了也是白搬）。
    pub fn host_targets(&self, s: &PressureSignals) -> (usize, usize) {
        let kv = if s.host_kv_enabled
            && s.host_kv_free_slots < self.host_low_slots(s.host_kv_total_slots)
            && s.disk_kv_free_slots > 0
        {
            self.host_high_slots(s.host_kv_total_slots)
                .saturating_sub(s.host_kv_free_slots)
        } else {
            0
        };
        let snap = if s.host_snap_enabled
            && s.host_snap_free_slots < self.host_low_slots(s.host_snap_total_slots)
            && s.disk_snap_free_slots > 0
        {
            self.host_high_slots(s.host_snap_total_slots)
                .saturating_sub(s.host_snap_free_slots)
        } else {
            0
        };
        (
            kv.min(HOST_REQUEST_SLOTS).min(HOST_MAX_SLOTS),
            snap.min(HOST_REQUEST_SNAPS),
        )
    }

    /// 空转 HostPressure 的最小间隔（节流；应急收缩不节流）。
    pub fn host_interval(&self) -> Duration {
        HOST_INTERVAL
    }
}

// ===========================================================================
// §3 门（纯函数；每个都对应真机上会遇到的一种情形）
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// 低水位触发、高水位目标、单次上限生效。
    #[test]
    fn low_watermark_triggers_and_high_is_the_target() {
        let p = PressurePolicy::new();
        assert_eq!(p.relief_pool_slots(100, 11), 14, "低水位下补到高水位");
        assert_eq!(p.relief_pool_slots(100, 12), 0, "等于低水位不触发");
        assert_eq!(p.relief_pool_slots(100, 0), 25, "空池补到高水位");
        assert_eq!(p.relief_pool_slots(1000, 0), 64, "应急单次上限生效");
        assert_eq!(p.idle_pool_slots(1000, 0), 16, "空转单次上限更小");
        assert_eq!(p.relief_pool_slots(0, 0), 0, "无池不触发");
    }

    /// 小池至少取 1 页（百分比下取整不能变成 0 = 永不触发）。
    #[test]
    fn small_pool_still_triggers() {
        let p = PressurePolicy::new();
        assert_eq!(p.pool_low_slots(4), 1);
        assert_eq!(p.pool_high_slots(4), 1);
        assert_eq!(p.relief_pool_slots(4, 0), 1);
    }

    /// 宿主两腿独立：各自低水位触发、各自补到高水位、各自受盘位约束。
    #[test]
    fn host_pressure_uses_independent_low_watermark_and_disk_room() {
        let p = PressurePolicy::new();
        let mut s = PressureSignals::default();
        s.host_kv_enabled = true;
        s.host_snap_enabled = true;
        s.host_kv_total_slots = 100; // low = 12, high = 25
        s.host_kv_free_slots = 11;
        s.disk_kv_free_slots = 100;
        assert_eq!(p.host_targets(&s), (14, 0), "低于低水位补到高水位");
        s.host_kv_free_slots = 0;
        assert_eq!(p.host_targets(&s), (16, 0), "每请求配额 16 生效");
        s.host_kv_free_slots = 12;
        assert_eq!(p.host_targets(&s), (0, 0), "等于低水位不触发");

        s.host_snap_total_slots = 10; // low = 1, high = 2
        s.host_snap_free_slots = 0;
        s.disk_snap_free_slots = 10;
        assert_eq!(p.host_targets(&s), (0, 1), "状态腿独立 + 每请求 1 条");
        s.disk_kv_free_slots = 0;
        assert_eq!(p.host_targets(&s), (0, 1), "KV 无盘位不影响快照");
        s.host_snap_free_slots = 2;
        assert_eq!(p.host_targets(&s), (0, 0));
    }

    /// 关掉的腿不搬（host 层没开 = 目标恒 0）。
    #[test]
    fn disabled_host_leg_never_moves() {
        let p = PressurePolicy::new();
        let mut s = PressureSignals::default();
        s.host_kv_total_slots = 100;
        s.host_kv_free_slots = 0;
        s.disk_kv_free_slots = 100;
        assert_eq!(p.host_targets(&s), (0, 0), "host_kv_enabled = false");
    }

    /// 驱动余量是信号的一部分（诊断面）。
    #[test]
    fn driver_headroom_is_part_of_the_signal() {
        let mut s = PressureSignals::default();
        s.driver_free_bytes = 100;
        s.reserve_bytes = 200;
        assert!(s.driver_below_reserve());
        assert!(s.summary().contains("驱动低于预留"));
        s.driver_free_bytes = 300;
        assert!(!s.driver_below_reserve());
    }
}
