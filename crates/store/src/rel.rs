//! 关系层门面(`Rel`)—— 两个独立管理器 + 一层关联(裁决②;设计卷 §3.1/§3.2)。
//!
//! **职责只有四条**:①身份映射 ②召回语义判定 ③换出单合并 ④报告口径。
//! 不含前向计算;搬运只出计划（`plan_*`），执行在 engine 的 `io`（律⑦ / S1）。
//!
//! ```text
//! probe(PrefixId) → RecallHit { kv_tier, snap_tier, blocks }
//!                   ├ 两腿都在 ⇒ RecallMode::Exact      (精确续跑;本引擎独有)
//!                   ├ 只 KV    ⇒ RecallMode::ContextOnly(论文式上下文召回;近似)
//!                   └ 都无     ⇒ RecallMode::Miss
//! ```
//!
//! # S3 身份迁移(本刀)
//!
//! KV 腿从"物理槽号账"升级为 **槽池 + 内容前缀树**(`kv`);`Rel` 仍是**唯一门面**:
//! engine 侧的块表操作(`ensure_for_len`/`register_chain`/`release_table`)与
//! 卸载/回灌都从这里进 —— 跨腿台账([`TierLedger`])只在门面上维护,腿内自洽。

use crate::budget::{Admission, AdmissionVerdict, BudgetInputs, TurnFloor, VramBudget};
use crate::disk::{AdoptReport, DiskEvictPlanOf, DiskTier, FillPlanOf, SpillPlanOf};
use crate::error::{StoreError, StoreResult};
use crate::host::{HostBlock, HostTier};
use crate::host::SlotTable;
use crate::id::{ChainHash, PrefixId, SnapKey};
use crate::kv::{CacheUpdate, KvDiag, KvStats};
use crate::leg::{BlockLeg, Carrier, ChainHit, InPlan, LegConfig, LegStats, PoolLayout, ReadPlan};
use crate::plan::MemoryPlan;
use crate::policy::{AdmissionPolicy, EvictCandidate, EvictPolicy, LruPolicy};
use crate::reservations::{Grant, ReservationId, ReservationKind, ReservationLedger};
use crate::state::{SnapDurability, StateStats, StateStore};
use crate::tier::{AssetKind, Tier, TierReport};
use owl_iface::contract::{ConnSlot, SwapPiece};

mod admin;
mod disk;
mod state;
mod kv;
mod report;
mod index;
mod boot;

/// KV 腿几何(boot 装配一次;`block_tokens` 本引擎恒 32 —— 设计卷 §3.6.2)。
///
/// 收编后本类型只是 [`LegConfig`] 的**别名位**(门面签名可读性用):几何的唯一产地
/// 是腿自己的 `LegConfig`(§23.1)。
pub type KvConfig = LegConfig;

/// 召回语义(关系层的核心产物;两腿独立 ⇒ 命中可以只有一半)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecallMode {
    /// 块链 + 状态快照都在 ⇒ **精确续跑**(位置不重排、状态直接还原)。
    Exact,
    /// 只有块链 ⇒ **上下文召回**(近似:复发状态沿原路,不重算历史)。
    ContextOnly,
    /// 都没有 ⇒ 冷启动(逐 token 重算)。
    Miss,
}

/// **KV checkpoint 的 store 侧计划**（store 出层/连续性，engine 只做 I/O）。
///
/// `chain` = 从根到本次可落盘前缀的**连续**链（含之前已在盘上的块）；
/// `dev` / `host` = 本次还要复制的块（源层）。盘上-only 不会进 `host`，
/// 也不会被误判成可从宿主复制 —— tier 真相只在 store 一处。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvCheckpointPlan {
    /// 连续前缀链（到本次写完的块为止；供快照逐边界落盘）。
    pub chain: Vec<ChainHash>,
    /// 本次需从设备 D2H 复制的块。
    pub dev: Vec<ChainHash>,
    /// 本次需从宿主 clone 字节的块（必有宿主内容）。
    pub host: Vec<ChainHash>,
    /// 盘层单块字节。
    pub block_bytes: usize,
    /// `chain` 开头已经在盘上的块数（本次从这之后开始写）。
    pub already_on_disk: usize,
}

/// **准入计划（dry-run）**：只读 Store 状态算出的“如果现在放行，会占多少 / 会得到什么判决”。
///
/// 判据分两段：
/// 1. `Rel::check(floor, ctx)`：per-turn 地板 / 腾位；
/// 2. `reserved_slots` 的**完整预算**（非超卖 = ctx + max_new；超卖 = 工作集）
///    以及超卖溢出是否真有宿主/磁盘可落。
///
/// 纪律（2026-11-18 用户裁决）：
/// - [`Rel::plan_admission`] **零状态变更**，可任意次调用；
/// - 计划结果为 `Admit` 时，engine 才可调 [`Rel::commit_admission`]；
/// - `EvictThen` / `Queue` / `Reject` 都不是可提交态：先执行搬迁/排队/拒绝；
/// - `commit` 只收**实际完成值**（floor/槽数），是独立的状态修改 API；
///   没有 open/commit/abort 三态，也没有回滚日志。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdmissionPlan {
    /// 待准入 turn。
    pub turn: u64,
    /// 本 turn 的显存地板口径。
    pub floor: TurnFloor,
    /// 本 turn 的上下文 token 数（含生成长度；调用方已按 cached 前缀扣除）。
    pub ctx_tokens: usize,
    /// dry-run 判决（`Admit` 才可 commit）。
    pub verdict: AdmissionVerdict,
    /// 准入面批准槽数（超卖档 = 工作集）。
    pub reserved_slots: usize,
    /// 分配面可消费槽数（超卖档 = 整条 ctx 的块表上限）。
    pub alloc_slots: usize,
}

impl AdmissionPlan {
    /// 是否可直接提交。
    pub fn is_admit(&self) -> bool {
        matches!(self.verdict, AdmissionVerdict::Admit { .. })
    }

    /// 准入面授予的地板字节（非 `Admit` 为 0）。
    pub fn floor_bytes(&self) -> usize {
        match self.verdict {
            AdmissionVerdict::Admit { floor_bytes } => floor_bytes,
            AdmissionVerdict::EvictThen { floor_bytes, .. } => floor_bytes,
            AdmissionVerdict::Queue { .. } | AdmissionVerdict::Reject { .. } => 0,
        }
    }
}

/// **准入提交的结果**：只报告落账后的账，不是事务句柄。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionApplied {
    /// 准入面 Grant（ActiveFloor；`ensure_for_len` 消费它）。
    pub grant: Grant,
    /// 已借出的 GDN 行号（`Owner::Turn(turn)`）。
    pub gdn_row: usize,
    /// GDN 行票（Row；turn 生命周期）。
    pub row_grant: Grant,
    /// 本 turn 的显存地板字节（报告面）。
    pub floor_bytes: usize,
}

/// 召回命中面(层信息;组装与搬运在 engine 侧段边界执行)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecallHit {
    /// attention 块链所在层(`None` = 未命中)。
    pub kv_tier: Option<Tier>,
    /// GDN 快照所在层(`None` = 无快照腿)。
    pub snap_tier: Option<Tier>,
    /// 命中块数(0 = 无 KV 腿)。
    pub blocks: usize,
}

impl RecallHit {
    /// 未命中。
    pub fn miss() -> Self {
        Self {
            kv_tier: None,
            snap_tier: None,
            blocks: 0,
        }
    }

    /// 语义判定(唯一产地)。
    pub fn mode(&self) -> RecallMode {
        match (self.kv_tier, self.snap_tier) {
            (Some(_), Some(_)) => RecallMode::Exact,
            (Some(_), None) => RecallMode::ContextOnly,
            _ => RecallMode::Miss,
        }
    }

    /// 是否命中任意一腿。
    pub fn hit(&self) -> bool {
        self.kv_tier.is_some()
    }
}

/// 保护面(活跃 turn 的地板/行 + 显式点名)—— 驱逐绝不碰。
#[derive(Clone, Debug, Default)]
pub struct Protected {
    /// 活跃 turn 的 attention 链(其块不可逐)。
    pub chains: Vec<ChainHash>,
    /// 活跃 turn 的快照键(点名保住)。
    pub snaps: Vec<SnapKey>,
}

/// **turn 终局归还报告**（[`Rel::finish_turn`] 的返回值；诊断/日志用）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TurnRelease {
    /// 准入账里确实有该 turn 并被移除。
    pub admission: bool,
    /// 归还的 Grant 张数。
    pub grants: usize,
    /// 归还的 GDN 行（`None` = 该 turn 无在借行）。
    pub row: Option<usize>,
}

/// 全量报告(设计卷 §2.4 报告面)。
#[derive(Clone, Debug, Default)]
pub struct StoreReport {
    /// 三层账明细。
    pub tiers: TierReport,
    /// attention 腿统计。
    pub kv: KvStats,
    /// **槽引用 / 可下沉性诊断**(为何腾不出槽的判据面;`/debug/vram`)。
    pub kv_diag: KvDiag,
    /// 状态腿统计。
    pub state: StateStats,
    /// 活跃 turn 数。
    pub active_turns: usize,
    /// 保留面字节(地板 + 行)。
    pub reserved_bytes: usize,
    /// **Grant 批准总量(槽)**(P0-4;准入面真源)。
    pub reserved_slots: usize,
    /// **Grant 在册 turn 数**(P0-4;槽位策略真源)。
    pub in_flight_turns: usize,
    /// 按用途汇总的票账（Load/Row/Snap/Temp 等；不参与准入）。
    pub grants: Vec<crate::reservations::GrantReport>,
    /// 非 turn 归属的不可动面字节(idle 会话持着的历史;§23.2)。
    pub session_held_bytes: usize,
    /// 弹性面字节(可驱逐)。
    pub elastic_bytes: usize,
    /// 缓存面字节(可驱逐/可下沉)。
    pub cache_bytes: usize,
    /// 策略名(驱逐序)。
    pub policy: String,
    /// 宿主层驻留字节。
    pub host_resident_bytes: usize,
    /// 宿主层空转缓冲字节(真实 RAM;报告必须算上)。
    pub host_free_bytes: usize,
    /// 磁盘层(KV 腿;`None` = 关层)。
    pub disk_kv: Option<crate::disk::DiskStats>,
    /// 磁盘层(状态腿;`None` = 关层)。
    pub disk_snap: Option<crate::disk::DiskStats>,
    /// **磁盘层按会话归属**(KV 链 + 快照；`NO_SESSION` = 无主/孤儿)。
    pub disk_sessions: Vec<DiskSessionUsage>,
    /// **事件账**(DeviceHandle 钩子的出借登记;与腿自报互为校验 —— 两独立来源)。
    pub events: EventBook,
    /// **boot 显存计划**(V0;`None` = 旧装配/测试未注入)。
    pub plan: Option<MemoryPlan>,
}

/// **磁盘层的会话归属**(调度/淘汰的会话感知面；`NO_SESSION` = 无主/孤儿)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DiskSessionUsage {
    /// 会话 id（`NO_SESSION` = 无主/孤儿；会话 0 是合法 id）。
    pub session: u64,
    /// 该会话在盘上的 KV 块数。
    pub kv_blocks: usize,
    /// 该会话在盘上的 KV 字节。
    pub kv_bytes: usize,
    /// 该会话在盘上的快照条数。
    pub snap_entries: usize,
    /// 该会话在盘上的快照字节。
    pub snap_bytes: usize,
}

/// **事件账报告**(`LedgerBook` 的出借登记面投影)。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EventBook {
    /// rank × owner × tier × kind 明细(字节降序)。
    pub rows: Vec<crate::tier::EventRow>,
    /// 在借条目数。
    pub entries: usize,
    /// 在借总字节。
    pub bytes: usize,
    /// 记账失败计数(非零 = 有事件没入账;**不许静默**)。
    pub hook_fail: u64,
    /// **片级换层跳过**计数(口径不支持:KV 池物理块 = 多层片,swap 逐片发 ⇒
    /// 事件账的层归属对该资产不适用,以腿自报为准)。**不是**失败。
    pub partial_moves: u64,
    /// **逐 rank 在借 id 集**(与块账房做全等门的原料)。
    pub ids: Vec<(u8, Vec<u64>)>,
    /// **lifecycle 在借清单**(turn/session 归属的逐条:id/点名/字节/类 ——
    /// N4/N5"本该归还却还没归还"的**逐块指纹**;系统常驻/缓存面不在列)。
    pub open_lifecycle: Vec<OpenLoan>,
}

/// 在借清单的一行(报告面投影;`Loan` + id)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenLoan {
    /// 设备块 id。
    pub id: u64,
    /// 分配点点名(哪一处要的)。
    pub name: String,
    /// 归属主体。
    pub owner: crate::tier::Owner,
    /// 资产类别。
    pub kind: AssetKind,
    /// 字节。
    pub bytes: usize,
    /// 设备序(TP 分卡)。
    pub rank: u8,
}

/// **状态腿淘汰序的一条**(诊断;会话感知淘汰的可观测面)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SnapEvictPreview {
    /// 快照键。
    pub key: SnapKey,
    /// 会话归属(0 = 无)。
    pub session: u64,
    /// 边界块号(同会话内深者先走)。
    pub depth: usize,
    /// 是否为生成段(逐出即丢)。
    pub transient: bool,
    /// 持久角色（Tail / History；Transient 看 `transient`）。
    pub role: crate::state::SnapRole,
}

/// **盘上链索引重建报告**(S6.1;boot)。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AdoptChainsReport {
    /// 成功重建的链数。
    pub chains: usize,
    /// 成功重建的块数。
    pub blocks: usize,
    /// 被跳过的链(原因;**不静默**)。
    pub skipped: Vec<(String, String)>,
    /// 被截断的链(原块数, 重建块数)—— 盘上内容只覆盖前缀。
    pub truncated: Vec<(String, usize)>,
    /// **重建真正用到的块身份**(engine 据此清理盘上的孤儿内容 —— 没被任何
    /// 重建链引用的内容文件占配额且永远用不上)。
    pub kept: Vec<ChainHash>,
}

/// **磁盘接线回执**(boot;engine 建目录 / 扫盘 / 报告用)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskAttach {
    /// KV 腿内容目录。
    pub kv_root: std::path::PathBuf,
    /// 状态腿内容目录。
    pub snap_root: std::path::PathBuf,
    /// KV 腿可容块数。
    pub kv_slots: usize,
    /// 状态腿可容快照数。
    pub snap_slots: usize,
}

/// 关系层门面:两个独立管理器 + 一层关联。
///
/// **收编后**(§23.1):KV 腿的全部状态(槽池/内容身份树/宿主物理)只住 [`BlockLeg`];
/// 本层不另记一份 —— `ledger` 只记 store 自管面(状态腿/保留面/弹性面),
/// KV 块面的真源 = 腿的报告(`LegStats`)。
pub struct Rel {
    /// attention 块腿(账 + 宿主物理 + 搬运口 —— 唯一状态持有者)。
    ///
    /// 字段私有：engine 只能经 [`Rel::leg`] / [`Rel::leg_mut`] 取用。
    leg: BlockLeg,
    /// GDN 状态账(行 + 快照;快照的**层**是真源 —— §25.2)。
    state: StateStore,
    /// 状态腿的宿主层(键 = [`SnapKey`];与 KV 腿各自一本账)。
    snap_host: HostTier<SnapKey>,
    /// **KV 腿磁盘层**(S6;`None` = 未接线/关层)。
    ///
    /// 语义:磁盘是宿主的下游 —— 条目在 tree 里仍是"非设备槽"(`slot = None`,
    /// `locate` 归 `Tier::Host` 的**可回灌**面),物理承载在磁盘上
    /// (`DiskTier` 自己的账/报告才是盘层的真源)。
    disk: Option<DiskTier<ChainHash>>,
    /// **状态腿磁盘层**(S6);与 `state.set_tier(Disk)` 配套(快照层账可精确到盘)。
    snap_disk: Option<DiskTier<SnapKey>>,
    /// **选择层检索索引**(B 轴:Mean-K 块级索引;host 驻留 —— 论文同款)。
    /// 几何由 engine 在 boot 注册([`Rel::attach_index`])。
    index: Option<crate::select::MeanKIndex>,
    /// 语义账(状态腿快照/行 + 保留面/弹性面 + **DeviceHandle 事件账**)。
    ///
    /// `Arc<Mutex<..>>`(2026-10-19 用户裁决):DeviceHandle 侧的钩子要**与 store 看同一本账**
    /// ⇒ 共享所有权。为何不是方案甲字面的 `Rc<RefCell>`:`DeviceClient` 的 `Send`
    /// 是硬约束,且 `/debug/blocks` 的 `face_clone()` 会把句柄真的送进 HTTP 线程
    /// ⇒ `Rc` 在那条路径上是真的 UB。单线程热路径无争用锁 ≈ ns 级。
    /// store 自己的读写统一走 [`crate::tier::book_lock`]。
    ledger: crate::tier::SharedLedger,
    /// 准入策略(boot 装配)。
    policy: AdmissionPolicy,
    /// 活跃 turn 在飞账。
    admission: Admission,
    /// **Grant / Reservation 账**(P0-4;store 是“谁被批准占多少槽”的唯一真源)。
    reservations: ReservationLedger,
    budget: VramBudget,
    evict: Box<dyn EvictPolicy>,
    /// 整枝塌缩开关(见 [`Self::set_bulk_drop_when_tier_full`])。
    bulk_drop_when_tier_full: bool,
    /// 状态腿的设备布局(快照 = 「槽 × 片」;engine 在 boot 注册,§25.2)。
    snap_layout: Option<crate::leg::PoolLayout>,
    /// **peer 快照布局**(TP;快照槽既写两 rank ⇒ 各自块 id)。
    snap_layout_peer: Option<crate::leg::PoolLayout>,
    /// 一条快照的全量字节(布局注册后由 `slot_bytes()` 定谳)。
    snap_bytes: usize,
    /// **TP world**(2026-10-20):宿主腿每条链/每条快照占 `world` 个平面
    /// (= 每 rank 半份)。单卡恒 1。
    world: usize,
    /// **boot 显存计划**(V0;`attach_memory_plan` 注入,与预算入参交叉校验)。
    plan: Option<MemoryPlan>,
}

impl Rel {
    /// 选择层索引（未注册 = `None`）。
    pub fn index(&self) -> Option<&crate::select::MeanKIndex> {
        self.index.as_ref()
    }

    /// 活跃 turn 在飞账（只读）。
    pub fn admission(&self) -> &Admission {
        &self.admission
    }

    /// 活跃 turn 在飞账（可变；仅限 `SessionBank` 的复合账口）。
    pub fn admission_mut(&mut self) -> &mut Admission {
        &mut self.admission
    }

    /// 卡级预算(只读)。
    pub fn budget(&self) -> &VramBudget {
        &self.budget
    }

    /// 换驱逐策略(策略对象可换,判据不散 —— §3.2)。
    /// **召回信号转发**(B-4):召回点选中哪些块 ⇒ 策略面(attention-aware LRFU)。
    pub fn note_recall(&self, selected: &[ChainHash]) {
        self.evict.note_recall(selected);
    }

    /// 驱逐策略名(报告/测试面)。
    pub fn evict_policy_name(&self) -> &'static str {
        self.evict.name()
    }

    /// 换驱逐策略(`[kvmem] evict_policy`;装配期一次)。
    pub fn set_evict_policy(&mut self, p: Box<dyn EvictPolicy>) {
        self.evict = p;
    }

}

#[cfg(test)]
mod tests;
