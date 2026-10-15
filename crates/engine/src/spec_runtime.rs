//! spec(投机解码)运行域(2026-10-14 拆分自 running.rs:RunningEngine
//! 万能结构体按状态域拆解 —— 本模块收编**引擎级 spec 状态**:模式/深度/
//! 草稿器/统计/诊断门/B4 分母保鲜)。
//!
//! 生命周期注记:turn 域 spec 状态(TurnSpecState,七件套)随
//! ActiveTurn 生死,住 [`crate::turn`];本模块的字段是**跨 turn 引擎级**
//! 的(草稿器装载一次、统计全程累计、B4 分母跨 turn 保鲜)—— 两层
//! 不许混居,历史教训见 ActiveTurn::spec 头注(跨 turn 残留 = GDN 污染)。

use std::sync::Arc;

use owl_models::layers::rope::Rope;

/// E5-M2b:spec 模式三态(C7 回退挂钩)。裁决在 boot:
/// depth>0 且检查点有 mtp.* → Mtp;dumb 诊断位 → Dumb(哑草稿 =
/// anchor 重复);否则 Off(无草稿器不白发 verify 税)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum SpecMode {
    Off,
    Dumb,
    Mtp,
    /// DFlash2 码本选择器草稿(E5-DF;dflash2_dir)
    DFlash2,
}

/// 草稿器枚举(E5-DF;MTP 链式 / DFlash2 噪声块式双族)
pub(crate) enum Drafter {
    Mtp(Arc<owl_models::layers::mtp::MtpPredictor>),
    DFlash2(Arc<owl_models::layers::dflash2::DFlash2Draft>),
}

/// spec 接受账(轮级;恒等门/接受率观测面)
#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct SpecStats {
    pub rounds: usize,
    pub sum_m: usize,
    pub full: usize,
    pub partial: usize,
    pub zero: usize,
}

/// 引擎级 spec 运行态(组合叶;RunningEngine 持有)。字段全
/// pub(crate) 供 exec 热路径直读,复合逻辑做成方法。
pub(crate) struct SpecRuntime {
    /// E5-M2:投机轮深(0 = 关;>0 = 每轮草稿数,调度发 SpecRound)
    pub depth: usize,
    /// E5-M2b:spec 模式三态(诊断面)
    #[allow(dead_code)] // 诊断面字段(写入后当前无读者;boot 裁决记录)
    pub mode: SpecMode,
    /// 草稿头(Mtp/DFlash2;embed/lm_head 共享 target 不持有;Arc =
    /// propose 图闭包共享)
    pub drafter: Option<Drafter>,
    /// DFlash2 草稿 rope(θ 1e7;DFlash2 模式 Some)
    pub draft_rope: Option<Arc<Rope>>,
    /// DFlash2 target taps 数(dflash 模式 = 5;其余 0)
    pub dflash_tap_count: usize,
    /// spec 接受账(M2b 恒等门覆盖断言:三路 m 分布 + 账目闭合)
    pub stats: SpecStats,
    /// taps/memory 探针落盘一次性门(OWL_DFLASH_PROBE;AL=0 排查)
    pub mem_dumped: bool,
    /// concat 父块落盘一次性门(同上)
    pub mem_dumped2: bool,
    /// propose hidden 落盘一次性门(同上)
    pub hid_dumped: bool,
    /// B4 判据跨 turn 保鲜(2026-10-14):裸步成本 EWMA(ms)+ 其采样
    /// ctx。**为什么必须跨 turn**:TurnSpecState 随 turn 销毁 ⇒ 每 turn
    /// 起步分母 = 0 ⇒ 首个窗口只能用保守兜底线(2.5),在"真线 1.4 而
    /// AL≈2"的域会误降级并抖动。带 ctx 保鲜后,同档续聊直接沿用真线。
    pub carry_ms: f32,
    /// B4 保鲜 ctx(pos;涨过 25% 即作废,换档后旧采样失真)
    pub carry_ctx: usize,
}
