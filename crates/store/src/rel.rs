//! 关系层门面(`Rel`)—— 两个独立管理器 + 一层关联(裁决②;设计卷 §3.1/§3.2)。
//!
//! **职责只有四条**:①身份映射 ②召回语义判定 ③换出单合并 ④报告口径。
//! 不含前向计算;搬运只经 [`crate::moves`] 的自由函数(律⑦)。
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
use crate::error::{StoreError, StoreResult};
use crate::disk::{AdoptReport, DiskTier};
use crate::host::HostTier;
use crate::id::{ChainHash, PrefixId, SnapKey};
use crate::kv::{CacheUpdate, KvStats};
use crate::leg::{BlockLeg, ChainHit, LegConfig, LegStats};
use crate::policy::{AdmissionPolicy, EvictCandidate, EvictPolicy, LruPolicy};
use crate::state::{SnapDurability, StateStats, StateStore};
use crate::host::SlotTable;
use owl_iface::contract::{ConnSlot, SwapPiece};
use crate::tier::{AssetKind, Tier, TierReport};

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
        Self { kv_tier: None, snap_tier: None, blocks: 0 }
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

/// 全量报告(设计卷 §2.4 报告面)。
#[derive(Clone, Debug, Default)]
pub struct StoreReport {
    /// 三层账明细。
    pub tiers: TierReport,
    /// attention 腿统计。
    pub kv: KvStats,
    /// 状态腿统计。
    pub state: StateStats,
    /// 活跃 turn 数。
    pub active_turns: usize,
    /// 保留面字节(地板 + 行)。
    pub reserved_bytes: usize,
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
    /// **事件账**(DeviceHandle 钩子的出借登记;与腿自报互为校验 —— 两独立来源)。
    pub events: EventBook,
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
    /// **复用池**(缺口 4;一等指标:池内可复用字节)
    pub pool: PoolReport,
    /// **lifecycle 在借清单**(turn/session 归属的逐条:id/点名/字节/类 ——
    /// N4/N5"本该归还却还没归还"的**逐块指纹**;系统常驻/缓存面不在列)。
    pub open_lifecycle: Vec<OpenLoan>,
}

/// **复用池报告**(缺口 4:池内可复用面 —— 不再是告警里的一行,而是一等指标)。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PoolReport {
    /// 池内块数。
    pub entries: usize,
    /// 池内可复用字节。
    pub bytes: usize,
    /// 取块命中次数(零命令复用)。
    pub hits: u64,
    /// 取块未命中次数(走了真分配)。
    pub misses: u64,
    /// 池排空次数(N6)。
    pub drained: u64,
    /// 超上限**顶出**块数(已真还驱动)。
    pub evicted: u64,
    /// 池上限字节(0 = 不限)。
    pub cap: usize,
    /// **按类明细**(P2-3):`(kind, entries, bytes)`,字节降序。
    pub by_class: Vec<(AssetKind, usize, usize)>,
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
    pub leg: BlockLeg,
    /// GDN 状态账(行 + 快照;快照的**层**是真源 —— §25.2)。
    pub state: StateStore,
    /// 状态腿的宿主层(键 = [`SnapKey`];与 KV 腿各自一本账)。
    pub snap_host: HostTier<SnapKey>,
    /// **KV 腿磁盘层**(S6;`None` = 未接线/关层)。
    ///
    /// 语义:磁盘是宿主的下游 —— 条目在 tree 里仍是"非设备槽"(`slot = None`,
    /// `locate` 归 `Tier::Host` 的**可回灌**面),物理承载在磁盘上
    /// (`DiskTier` 自己的账/报告才是盘层的真源)。
    pub disk: Option<DiskTier<ChainHash>>,
    /// **状态腿磁盘层**(S6);与 `state.set_tier(Disk)` 配套(快照层账可精确到盘)。
    pub snap_disk: Option<DiskTier<SnapKey>>,
    /// **选择层检索索引**(B 轴:Mean-K 块级索引;host 驻留 —— 论文同款)。
    /// 几何由 engine 在 boot 注册([`Rel::attach_index`])。
    pub index: Option<crate::select::MeanKIndex>,
    /// 语义账(状态腿快照/行 + 保留面/弹性面 + **DeviceHandle 事件账**)。
    ///
    /// `Arc<Mutex<..>>`(2026-10-19 用户裁决):DeviceHandle 侧的钩子要**与 store 看同一本账**
    /// ⇒ 共享所有权。为何不是方案甲字面的 `Rc<RefCell>`:`DeviceClient` 的 `Send`
    /// 是硬约束,且 `/debug/blocks` 的 `face_clone()` 会把句柄真的送进 HTTP 线程
    /// ⇒ `Rc` 在那条路径上是真的 UB。单线程热路径无争用锁 ≈ ns 级。
    /// store 自己的读写统一走 [`crate::tier::book_lock`]。
    pub ledger: crate::tier::SharedLedger,
    /// 准入策略(boot 装配)。
    pub policy: AdmissionPolicy,
    /// 活跃 turn 在飞账。
    pub admission: Admission,
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
}

impl Rel {
    /// 装配(唯一入口:预算分解失败 = **拒绝启动**,A5.1)。
    ///
    /// `kv_cfg` = KV 腿几何(含宿主层预算) —— 每 token KV 字节由它推
    /// (`block_bytes / block_tokens`),与 `inputs.kv_bytes_per_token`
    /// **交叉校验**(同一件事只有一个入口;矛盾 = 装配错,响亮拒)。
    pub fn open(
        inputs: BudgetInputs,
        policy: AdmissionPolicy,
        kv_cfg: LegConfig,
        ledger: crate::tier::SharedLedger,
    ) -> StoreResult<Self> {
        Self::open_world(inputs, policy, kv_cfg, ledger, 1)
    }

    /// **TP 档装配**(`world = 2`):KV 腿与状态腿的宿主槽都按 `world` 个平面开
    /// (`host_cache_mb` 语义不变 = **两 rank 合计**;单链字节 = `world ×` 单页)。
    ///
    /// 搬运面契约:engine 对每个 rank 各发一条 `swap`(片偏移用
    /// [`crate::pieces_on_plane`]),槽台在两条命令之间**原箱传递** ⇒ 一次
    /// 借台/一次落账(预算与账目仍是一条链一份)。
    pub fn open_world(
        inputs: BudgetInputs,
        policy: AdmissionPolicy,
        kv_cfg: LegConfig,
        ledger: crate::tier::SharedLedger,
        world: usize,
    ) -> StoreResult<Self> {
        if world == 0 {
            return Err(StoreError::Invariant("Rel::open_world: world = 0".into()));
        }
        if kv_cfg.block_bytes == 0 || kv_cfg.block_tokens == 0 || kv_cfg.slots == 0 {
            return Err(StoreError::Budget(format!(
                "KV 腿几何非法:slots={} block_tokens={} block_bytes={}",
                kv_cfg.slots, kv_cfg.block_tokens, kv_cfg.block_bytes
            )));
        }
        let per_token = kv_cfg.block_bytes / kv_cfg.block_tokens;
        if per_token != inputs.kv_bytes_per_token {
            return Err(StoreError::Budget(format!(
                "每 token KV 字节不一致:腿几何算得 {per_token} ≠ 预算入参 {}(装配参数矛盾)",
                inputs.kv_bytes_per_token
            )));
        }
        if kv_cfg.block_tokens != inputs.page_tokens {
            return Err(StoreError::Budget(format!(
                "页宽不一致:腿几何 block_tokens={} ≠ 预算入参 page_tokens={}\
                 (地板按页口径算、链增长也按页 —— 两把尺必须同一把)",
                kv_cfg.block_tokens, inputs.page_tokens
            )));
        }
        // host 预算只有一个总量(`inputs.host_bytes` = 整个 store 的中层预算),腿拿其中
        // 一段 ⇒ 腿那段不许超过总量(超了就是两套规范,不许静默对不上)。
        if kv_cfg.host_bytes > inputs.host_bytes {
            return Err(StoreError::Budget(format!(
                "KV 腿 host 预算 {} > store host 总量 {}(中层分片不许超总账)",
                kv_cfg.host_bytes, inputs.host_bytes
            )));
        }
        let budget = VramBudget::decompose(inputs, policy.gdn_slots)?;
        // **账本预算交叉校验**(2026-10-19):账本由 boot 先造(DeviceHandle 要拿着它
        // 包住装载/捕获期的分配),这里**必须**与预算分解算出的三层预算**逐字相符** ——
        // 不符 = 两处算式漂了(响亮拒,不是静默采信任何一方)。
        {
            let want = budget.tier_budget();
            let got = crate::tier::book_lock(&ledger).tiers().budget();
            if got != want {
                return Err(StoreError::Budget(format!(
                    "账本预算与分解不符:账本 {got:?} ≠ 分解 {want:?}(boot 的算式与 `BudgetInputs` 漂了)"
                )));
            }
        }
        let state = StateStore::new(policy.gdn_slots, inputs.gdn_row_bytes);
        // 状态腿宿主层:预算 = store host 总量 − KV 腿分片(负数 = 装配错,响亮拒)
        let snap_host_bytes = inputs.host_bytes.checked_sub(kv_cfg.host_bytes).ok_or_else(|| {
            StoreError::Budget(format!(
                "KV 腿 host 分片 {} > store host 总量 {}",
                kv_cfg.host_bytes, inputs.host_bytes
            ))
        })?;
        Ok(Self {
            leg: BlockLeg::open_planes(kv_cfg, world)?,
            index: None,
            state,
            snap_host: HostTier::new(1, snap_host_bytes)?,
            disk: None,
            snap_disk: None,
            ledger,
            policy,
            admission: Admission::new(),
            budget,
            evict: Box::new(LruPolicy),
            bulk_drop_when_tier_full: false,
            snap_layout: None,
            snap_layout_peer: None,
            snap_bytes: 0,
            world,
        })
    }

    /// 宿主腿平面数(= TP world;1 = 单卡)。
    pub fn world(&self) -> usize {
        self.world
    }


    /// **状态腿的出序(判据唯一的一面;§30.6-②)** —— "设备层快照谁该先走"。
    ///
    /// 与 [`Self::plan_need`] 的 `plan_capacity` **共用同一份排序**
    /// (事实档:Transient 先;档内 `EvictPolicy`)。engine 的池只负责
    /// key→槽的物理映射 —— 不再在池里另搞一套 `rank + LRU`。
    ///
    /// `protect` = 点名保住的快照键(绝不入序)。
    pub fn snap_victim_order(&self, protect: &[SnapKey]) -> Vec<SnapKey> {
        let cands: Vec<EvictCandidate> = self
            .state
            .evict_candidates(protect)
            .into_iter()
            .filter(|c| c.tier == Tier::Gpu)
            .collect();
        crate::hooks::order_snap_victims(self.evict.as_ref(), cands, &self.state)
            .into_iter()
            .filter_map(|c| c.snap)
            .collect()
    }

    /// **接线磁盘层**(boot 一次;S6)。
    ///
    /// - 目录布局:`root/kv/` + `root/snap/`(两腿分目录 —— 混放会让启动扫描
    ///   与清理串腿);
    /// - 配额：**分片不许超总账**（`kv_bytes + snap_bytes ≤ inputs.disk_bytes`，
    ///   与 host 同一条律；总账为 0 = 配置没给盘配额 ⇒ 响亮拒）；
    /// - **盘块 = 一条链的宿主槽**（`world ×` 单页）：溢写取料口是 `host_take`
    ///   （整槽交出去），回填也落整槽 —— 用单平面字节会让 `plan_spill.bytes`
    ///   与内容长度不符（TP 下写 2 倍、读 1 倍 = 静默截断）；
    /// - `snap_bytes` 的分片粒度 = 一条快照的字节（需先 `attach_snap_layout`）。
    pub fn attach_disk(
        &mut self,
        root: impl Into<std::path::PathBuf>,
        kv_bytes: usize,
        snap_bytes: usize,
    ) -> StoreResult<DiskAttach> {
        let total = self.budget.tier_budget().disk_bytes;
        if total == 0 {
            return Err(StoreError::Budget(
                "磁盘层配置为 0(先给 [pool] disk_cache_mb)".into(),
            ));
        }
        let want = kv_bytes.checked_add(snap_bytes).ok_or_else(|| {
            StoreError::Budget("磁盘分片溢出".to_string())
        })?;
        if want > total {
            return Err(StoreError::Budget(format!(
                "磁盘分片超总账:{kv_bytes} + {snap_bytes} > {total}(中层分片不许超总账)"
            )));
        }
        let snap_blk = self.snap_bytes;
        if snap_bytes > 0 && snap_blk == 0 {
            return Err(StoreError::Invariant(
                "状态腿盘层要先注册快照布局(attach_snap_layout)".into(),
            ));
        }
        let root = root.into();
        let kv_root = root.join("kv");
        let snap_root = root.join("snap");
        let kv = DiskTier::<ChainHash>::new(kv_root.clone(), self.leg.host_slot_bytes(), kv_bytes)?;
        let snap = DiskTier::<SnapKey>::new(snap_root.clone(), snap_blk.max(1), snap_bytes)?;
        let attach = DiskAttach {
            kv_root,
            snap_root,
            kv_slots: kv.free_slots(),
            snap_slots: snap.free_slots(),
        };
        self.disk = Some(kv);
        self.snap_disk = Some(snap);
        Ok(attach)
    }

    /// **启动扫描回填**(engine 读完两个目录后调用;S6 —— 账先回来,内容惰性读)。
    pub fn adopt_disk(
        &mut self,
        kv: Vec<(ChainHash, std::path::PathBuf, usize)>,
        snap: Vec<(SnapKey, std::path::PathBuf, usize)>,
    ) -> (AdoptReport, AdoptReport) {
        let a = match self.disk.as_mut() {
            Some(d) => d.adopt(kv),
            None => AdoptReport::default(),
        };
        let b = match self.snap_disk.as_mut() {
            Some(d) => d.adopt(snap),
            None => AdoptReport::default(),
        };
        // **状态腿:盘上的快照要重建 `StateStore` 账**(否则重启后 `probe` 看不见它们
        // ⇒ 文件在却复用不了)。`SnapKey` 是块边界内容身份 ⇒ 键可独立恢复;
        // `durability` 盘上没存 ⇒ 按 `Durable` 保守记(Transient 本来就不写盘)。
        let snap_keys: Vec<(SnapKey, usize)> = match self.snap_disk.as_ref() {
            Some(d) => {
                let bytes = d.block_bytes();
                d.resident_keys().into_iter().map(|k| (k, bytes)).collect()
            }
            None => Vec::new(),
        };
        for (k, bytes) in snap_keys {
            self.state.register_snap(k, bytes, Tier::Disk, SnapDurability::Durable);
        }
        (a, b)
    }

    /// **状态腿淘汰序预览**(诊断面;`/debug/vram` 用)。
    ///
    /// 直接投影 [`Self::snap_victim_order`](与容量调度同一份判据)的前 `limit` 条,
    /// 带**会话归属**与**边界深度** —— 真机上可直接看"同会话深者先走"是否生效。
    pub fn snap_evict_preview(&self, limit: usize) -> Vec<SnapEvictPreview> {
        self.snap_victim_order(&[])
            .into_iter()
            .take(limit)
            .map(|k| {
                let (session, depth) = self.state.snap_session(k).unwrap_or((0, 0));
                SnapEvictPreview {
                    key: k,
                    session,
                    depth,
                    transient: self.state.snap_durability(k)
                        == Some(crate::state::SnapDurability::Transient),
                }
            })
            .collect()
    }

    /// **按盘上索引重建树**(S6.1;boot 扫完 `.idx` 后调用)。
    ///
    /// 逐链:**逐块重算身份**(`hash_block(parent, tokens)`)并**逐个查"内容文件在盘上"**
    /// —— 遇到第一块缺内容就**截断**(命中语义本就是"连续前缀",断链即停)。
    /// 重建的条目 `slot = None`(`locate` 归 `Tier::Host`),命中后由
    /// `fill_kv_from_disk` 读回宿主 —— 该链路在 S6 已打通。
    ///
    /// 页宽不符 / 一块内容都没有的链 = **跳过并列出**(不静默)。
    pub fn adopt_disk_chains(
        &mut self,
        chains: &[crate::disk_index::ChainIndex],
    ) -> StoreResult<AdoptChainsReport> {
        let mut rep = AdoptChainsReport::default();
        let Some(disk) = self.disk.as_ref() else {
            return Ok(rep);
        };
        let page = self.leg.block_tokens().max(1);
        for idx in chains {
            if idx.block_tokens != page {
                rep.skipped.push((
                    format!("{} 块", idx.blocks()),
                    format!("索引页宽 {} ≠ 腿 {page}", idx.block_tokens),
                ));
                continue;
            }
            let mut parent = ChainHash::new(0);
            let mut ok = 0usize;
            for block in idx.tokens.chunks(page) {
                let h = crate::kv::hash_block(parent, block);
                if !disk.contains(h) {
                    break; // 内容不在盘上 ⇒ 连续前缀到此为止
                }
                parent = h;
                ok += 1;
            }
            if ok == 0 {
                rep.skipped.push((
                    format!("{} 块", idx.blocks()),
                    "盘上一块内容都没有(索引孤儿)".to_string(),
                ));
                continue;
            }
            let prefix = idx.truncated(ok);
            let inserted = self.leg.register_chain_offloaded(&prefix.tokens)?;
            rep.chains += 1;
            rep.blocks += inserted.len();
            rep.kept.extend(inserted);
            if ok < idx.blocks() {
                rep.truncated.push((format!("{} 块", idx.blocks()), ok));
            }
        }
        Ok(rep)
    }

    /// 磁盘腿是否接线。
    pub fn disk_enabled(&self) -> bool {
        self.disk.as_ref().is_some_and(|d| !d.is_disabled())
    }

    /// **注册状态腿的设备布局**(boot 一次;§25.2):快照 = 一种"块"
    /// (槽 × 片),片 = 每个 (层,quad) 一行宽缓冲。
    pub fn attach_snap_layout(&mut self, layout: crate::leg::PoolLayout) -> StoreResult<u64> {
        let bytes = layout.slot_bytes();
        if bytes == 0 {
            return Err(StoreError::Invariant("状态腿布局单块字节为 0".into()));
        }
        // 宿主层的块粒度 = 一条快照 × world 平面(建 host 时未知,故此处重建:预算不变)
        let budget = self.snap_host.budget_bytes();
        self.snap_host = HostTier::new(bytes.saturating_mul(self.world), budget)?;
        let epoch = layout.epoch;
        self.snap_layout = Some(layout);
        self.snap_bytes = bytes;
        Ok(epoch)
    }

    /// **注册 peer 快照布局**(TP;`world > 1` 必调;两 rank 的快照槽同形)。
    pub fn attach_snap_layout_peer(
        &mut self,
        layout: crate::leg::PoolLayout,
    ) -> StoreResult<u64> {
        let got = layout.slot_bytes();
        if got != self.snap_bytes {
            return Err(StoreError::Invariant(format!(
                "peer 快照布局单块字节不符:布局 {got} ≠ 本 rank {}",
                self.snap_bytes
            )));
        }
        let epoch = layout.epoch;
        self.snap_layout_peer = Some(layout);
        Ok(epoch)
    }

    /// **注册选择层索引几何**(B 轴 boot 一次;§8):`dim` = head_dim。
    pub fn attach_index(&mut self, dim: usize, layers: usize, kv_heads: usize) -> StoreResult<()> {
        self.index = Some(crate::select::MeanKIndex::new(dim, layers, kv_heads)?);
        Ok(())
    }

    /// **登记一块的索引面**(B 轴):形状校验在 `MeanKIndex` 内;入账到 host 层。
    pub fn index_register(
        &mut self,
        block: ChainHash,
        entries: Vec<crate::select::MeanKEntry>,
    ) -> StoreResult<()> {
        let ix = self
            .index
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("选择层索引未注册(engine 应在 boot 调 attach_index)".into()))?;
        let bytes = ix.block_bytes();
        ix.register(block, entries)?;
        // 第 0 刀:索引面 = **缓存归属**(无单会话归属;跨会话共享)
        crate::tier::book_lock(&self.ledger).tiers_mut().add_owned(
            crate::tier::Owner::Cache,
            Tier::Host,
            AssetKind::Index,
            bytes,
        )
    }

    /// **登记一块的扁平索引段**(GPU Mean-K 出口):与 [`Self::index_register`]
    /// 同账同语义，只是段已是 `[layer][kv_head][dim]` 规范序。
    pub fn index_register_flat(
        &mut self,
        block: ChainHash,
        flat: Vec<u16>,
    ) -> StoreResult<()> {
        let ix = self
            .index
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("选择层索引未注册(engine 应在 boot 调 attach_index)".into()))?;
        let bytes = ix.block_bytes();
        ix.register_flat(block, flat)?;
        crate::tier::book_lock(&self.ledger).tiers_mut().add_owned(
            crate::tier::Owner::Cache,
            Tier::Host,
            AssetKind::Index,
            bytes,
        )
    }

    /// 丢一块的索引(出账)。
    pub fn index_drop(&mut self, block: ChainHash) -> StoreResult<()> {
        let ix = self
            .index
            .as_mut()
            .ok_or_else(|| StoreError::Invariant("选择层索引未注册".into()))?;
        let bytes = ix.block_bytes();
        ix.drop_block(block)?;
        crate::tier::book_lock(&self.ledger).tiers_mut().remove_owned(
            crate::tier::Owner::Cache,
            Tier::Host,
            AssetKind::Index,
            bytes,
        )
    }

    /// 索引总字节(0 = 未注册)。
    pub fn index_bytes(&self) -> usize {
        self.index.as_ref().map(|i| i.bytes()).unwrap_or(0)
    }

    /// 一条快照的字节(布局未注册 = 0)。
    pub fn snap_bytes(&self) -> usize {
        self.snap_bytes
    }

    /// 状态腿宿主层预算(字节;0 = 关层)。
    pub fn snap_host_budget(&self) -> usize {
        self.snap_host.budget_bytes()
    }

    /// 状态腿宿主层驻留字节。
    pub fn snap_host_used(&self) -> usize {
        self.snap_host.used_bytes()
    }

    /// 状态腿宿主块**大页提示未生效的块数**（0 = 全生效）。
    /// >0 不是错误：只是冷写回到 4 KiB 缺页速度（159 MiB 约 80ms vs 12ms）——
    /// 引擎在下沉点打点上报（不许静默）。
    pub fn snap_host_hint_failures(&self) -> usize {
        self.snap_host.hint_failures()
    }

    /// 状态腿宿主层**累计逐出的字节**(政策面:宿主满时逐最冷条目)。
    /// 引擎在下沉点读差分上报(store 不引日志依赖)。
    pub fn snap_host_evicted_bytes(&self) -> u64 {
        self.snap_host.evicted_bytes()
    }

    /// 状态腿宿主层**还能装几条快照**。
    pub fn snap_host_free_slots(&self) -> usize {
        let used = self.snap_host.used_bytes() + self.snap_host.free_bytes();
        self.snap_host
            .budget_bytes()
            .saturating_sub(used)
            / self.snap_bytes.max(1)
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

    /// **命中查询**(按身份;纯 host、零副作用 —— 律⑥)。
    ///
    /// 与 [`Self::match_prefix`] 二选一使用(两条查询路径各自记一次命中账;
    /// 同一次查询不要两条都调,否则计数双算)。
    pub fn probe(&mut self, id: PrefixId) -> RecallHit {
        let kv_tier = self.leg.carrier_of(id.chain).ok().map(|c| match c {
            crate::leg::Carrier::Device(_) => Tier::Gpu,
            crate::leg::Carrier::Host => Tier::Host,
            crate::leg::Carrier::Disk => Tier::Disk,
        });
        let snap_tier = self.state.locate(id.snap_key());
        let hit = RecallHit { kv_tier, snap_tier, blocks: kv_tier.map_or(0, |_| self.leg.chain_len(id.chain)) };
        self.leg.note_probe(hit.kv_tier.is_some());
        self.state.note_probe(hit.snap_tier.is_some());
        hit
    }

    /// **按 token 命中查询**(engine L1 的第一跳):逐块内容匹配,回块链 + 绑定。
    ///
    /// 未绑定的格(`slot = None`)= 内容在宿主层,调用方须在**段边界**回灌
    /// (律③)—— 本方法自身不做设备动作。
    pub fn match_prefix(&mut self, tokens: &[u32]) -> StoreResult<(ChainHit, RecallHit)> {
        let m = self.leg.match_prefix(tokens)?;
        let tail = m.hashes.last().copied();
        let kv_tier = tail.and_then(|t| self.leg.carrier_of(t).ok()).map(|c| match c {
            crate::leg::Carrier::Device(_) => Tier::Gpu,
            crate::leg::Carrier::Host => Tier::Host,
            crate::leg::Carrier::Disk => Tier::Disk,
        });
        let snap_tier = tail.and_then(|t| self.state.locate(SnapKey::at_boundary(t)));
        let hit = RecallHit { kv_tier, snap_tier, blocks: m.matched_blocks };
        Ok((m, hit))
    }

    /// **整枝塌缩开关**(boot 一次;`[kvmem].bulk_drop_when_tier_full`)。
    pub fn set_bulk_drop_when_tier_full(&mut self, on: bool) {
        self.bulk_drop_when_tier_full = on;
    }

    /// **整枝塌缩**(批量过期;见 [`KvStore::prune_branch`])。
    pub fn prune_branch(&mut self, tail: ChainHash, max_blocks: usize) -> StoreResult<usize> {
        self.leg.prune_branch(tail, max_blocks)
    }

    /// **废弃链身份**(前缀无感删除的"改写即失效";见 `KvStore::drop_chain`)。
    pub fn drop_chain(&mut self, chain: ChainHash) -> StoreResult<bool> {
        self.leg.drop_chain(chain)
    }

    /// **登记一条前缀链**(turn 收口 L3):缓存取一份绑定引用,容量驱逐同步丢宿主副本。
    ///
    /// 只做 KV 腿;快照腿由 [`StateStore::return_row`] 登记(两腿独立 —— 裁决②)。
    pub fn register_chain(&mut self, tokens: &[u32], table: &[u32]) -> StoreResult<CacheUpdate> {
        self.leg.register_chain(tokens, table)
    }

    /// 块表增长(engine 每步/每 turn 调;压力驱逐与宿主副本回收均在腿内)。
    pub fn ensure_for_len(&mut self, table: &mut Vec<u32>, target_len: usize) -> StoreResult<()> {
        self.leg.ensure_for_len(table, target_len)
    }

    /// 归还会话块表。
    pub fn release_table(&mut self, table: &mut Vec<u32>) -> StoreResult<()> {
        self.leg.release_table(table)
    }

    /// 准入判决(§2.4 准入式;`evictable` = 缓存面 + 弹性面可动字节)。
    pub fn check(&self, floor: TurnFloor, ctx_tokens: usize) -> AdmissionVerdict {
        let evictable = self.evictable_bytes();
        crate::budget::check_admission(
            &self.budget,
            &self.admission,
            floor,
            ctx_tokens,
            evictable,
            self.policy.on_full,
        )
    }

    /// 可驱逐面字节(缓存 + 弹性)。
    pub fn evictable_bytes(&self) -> usize {
        self.admission.cache_bytes() + self.admission.elastic_bytes()
    }

    /// **唯一调度入口**(§30):给需求 ⇒ 两腿统一的“谁走谁进/拒”。纯 host、零 I/O。
    ///
    /// 判据全在 [`crate::hooks::plan_capacity`](唯一产地);本方法只做两件事:
    /// 算两腿的宿主可容纳量、把 `Rel` 的两腿账与策略对象递进去。
    /// engine 在**段边界**调本方法取单,自己发 `swap`/`drop`(§21 裁决)。
    pub fn plan_need(
        &self,
        need: crate::hooks::Need,
        protect: &Protected,
    ) -> crate::hooks::Verdict {
        let room = crate::hooks::Room {
            host_kv_slots: self.leg.host_free_slots(),
            host_snap_slots: self.snap_host_free_slots(),
            disk_kv_slots: self.disk.as_ref().map(|d| d.free_slots()).unwrap_or(0),
            disk_snap_slots: self.snap_disk.as_ref().map(|d| d.free_slots()).unwrap_or(0),
            bulk_drop_when_tier_full: self.bulk_drop_when_tier_full,
        };
        crate::hooks::plan_capacity(
            &self.leg,
            &self.state,
            self.evict.as_ref(),
            need,
            &protect.chains,
            &protect.snaps,
            room,
        )
    }

    /// **磁盘回填落宿主**(执行面兜底):宿主满 ⇒ 先逐最冷(保护本链)再放。
    ///
    /// 为什么不是裸 `host_put`:回填发生在 `plan_onload` **之前** —— 宿主满时
    /// 它同样会撞 `Exhausted`(与 offload 同一类洞)。保护面 = 本链(它正在被
    /// 回填,删掉就是自拆)。
    pub fn host_fill(
        &mut self,
        chain: ChainHash,
        buf: crate::host::HostBlock,
    ) -> StoreResult<()> {
        if self.leg.host_enabled() && self.leg.host_free_slots() == 0 {
            let protect = self.leg.chain_blocks(chain);
            let _ = self.leg.host_relieve(1, &protect)?;
        }
        self.leg.host_put(chain, buf)
    }

    /// **KV 链下沉计划**(engine 执行臂用;宿主满时**先逐最冷再重试**)。
    ///
    /// 与 `plan_capacity` 轮零同一条政策：最冷链尾向内削、保护面（本链）整条不动。
    /// 这口是**执行面兜底** —— 计划面已按宿主容量出票时通常一次成功；
    /// 盘写失败/账漂时不让 turn 硬崩（`ops.host_relief_blocks` 就是
    /// “计划曾经乐观/滞后”的指纹，engine 据此打点）。
    ///
    /// 宿主关层（0）= 旧行为：直接 `Exhausted`，不静默降级。
    pub fn plan_kv_offload(&mut self, tail: ChainHash) -> StoreResult<crate::leg::OutPlan> {
        match crate::leg::plan_offload(&mut self.leg, tail) {
            Ok(p) => Ok(p),
            Err(StoreError::Exhausted(why)) => {
                let Some(need) = self.leg.chain_dev_span(tail) else {
                    return Err(StoreError::Exhausted(why));
                };
                // 保护面 = 本链全部块（已在宿主的那些不能被“让位”删掉 ——
                // 否则搬完设备的块，链上缺一段，前缀永远拼不回来）
                let protect = self.leg.chain_blocks(tail);
                if self.leg.host_relieve(need, &protect)? == 0 {
                    return Err(StoreError::Exhausted(why));
                }
                crate::leg::plan_offload(&mut self.leg, tail)
            }
            Err(e) => Err(e),
        }
    }

    // ───────────────── 状态腿搬运口(§25.2;与 KV 腿同形:store 出计划与账) ─────────────────

    /// 状态腿的片(逐 (层,quad);`dev_off = 槽 × 行宽`)。
    fn snap_pieces(&self, slot: u32, plane: usize) -> StoreResult<Vec<SwapPiece>> {
        let l = if plane == 0 { self.snap_layout.as_ref() } else { self.snap_layout_peer.as_ref() };
        let layout = l
            .ok_or_else(|| StoreError::Invariant(format!(
                "状态腿布局(平面 {plane})未注册(engine 应在 boot 调 attach_snap_layout/…_peer)"
            )))?;
        Ok(layout
            .pieces
            .iter()
            .map(|p| SwapPiece {
                block: p.buf.id,
                dev_off: slot as usize * p.stride_bytes,
                conn_off: 0, // 单条快照的宿主槽 = 连续一条(片内偏移在下面补)
                len: p.stride_bytes,
                slot: ConnSlot(0),
            })
            .collect())
    }

    /// **快照下沉计划**(设备槽 `slot` 上的快照 → 宿主)。
    /// `Busy` = 快照不在账/宿主层已满(零副作用,调用方跳过)。
    ///
    /// **不重要就不搬**(2026-10-19):生成段边界快照逐出即丢 —— 下沉一条
    /// 159 MiB ≈ **50 ms**(实测,页锁定外的 PCIe 速),而生成段每 32 tok 就出
    /// 一条 ⇒ 全下沉 = 每 token 数 MiB 级 PCIe 流量,不可承受。拒语即事实
    /// (调用方回落已有"沉不下⇒丢"路;见 [`SnapDurability`])。
    pub fn plan_snap_offload(
        &mut self,
        key: SnapKey,
        slot: u32,
    ) -> StoreResult<(Vec<Vec<SwapPiece>>, SlotTable)> {
        let bytes = self
            .state
            .snap_bytes(key)
            .ok_or_else(|| StoreError::Unknown(format!("快照 {key:?} 不在账")))?;
        if self.state.snap_durability(key) != Some(SnapDurability::Durable) {
            return Err(StoreError::Invariant(format!(
                "快照 {key:?} 不可下沉(生成段边界:链一推进就被取代 ⇒ 逐出即丢)"
            )));
        }
        if bytes != self.snap_bytes {
            return Err(StoreError::Invariant(format!(
                "快照字节不符:账 {bytes} ≠ 布局 {}",
                self.snap_bytes
            )));
        }
        // **宿主满 ⇒ 先逐最冷**(2026-10-19 政策补口;G3 定档轮实测立案)。
        //
        // 旧行为:`acquire()` 预算越界直接 `Exhausted` ⇒ 调用方回落"丢**新**
        // 快照" ⇒ **宿主梯从不逐出**。实测后果:开机做过几轮会话后,3 GB 宿主
        // 梯被旧链边界占满 ⇒ 14k prompt 的 28 个边界里 **18 个被丢** ⇒ 后来
        // 的"部分前缀"复用面从 ~97% 塔到 29%(7k 共享只复用 2k = 白重算 5k
        // token ≈ 5 s/请求)。
        //
        // 政策 = **同一份策略对象**(`EvictPolicy`,与 KV 腿共用):只比 `Tier::Host`
        // 的候选,排序后取第一个。代价说明:先逐后搬 ⇒ 若接下来的 DMA 失败
        // (`abort_snap_offload`),冷条目已丢(比"丢新条目"代价小 —— 冷条目
        // 本来就要被逐)。
        let last = self.snap_host.used_bytes();
        let mut guard = 0usize;
        let buf = loop {
            match self.snap_host.acquire() {
                Ok(b) => break b,
                Err(StoreError::Exhausted(why)) => {
                    let mut cands: Vec<EvictCandidate> = self
                        .state
                        .evict_candidates(&[])
                        .into_iter()
                        .filter(|c| c.tier == Tier::Host)
                        .collect();
                    if cands.is_empty() {
                        // 宿主账与层账不一致(账说满而层无可逐)—— 不许静默
                        return Err(StoreError::Invariant(format!(
                            "宿主梯满且无 Host 层候选可逐({why};层账={last}B)"
                        )));
                    }
                    self.evict.order(&mut cands);
                    let v = cands[0]
                        .snap
                        .ok_or_else(|| StoreError::Invariant("驱逐候选缺 snap 键".into()))?;
                    let freed = self.snap_host.evict(v)?;
                    self.state.drop_snap(v)?;
                    let _ = freed;
                    guard += 1;
                    if guard > 16 {
                        return Err(StoreError::Invariant(
                            "宿主梯逐出循环超限(16):预算与释放字节不一致".into(),
                        ));
                    }
                }
                Err(e) => return Err(e),
            }
        };
        let all = (0..self.world.max(1))
            .map(|plane| {
                let mut pieces = self.snap_pieces(slot, plane)?;
                // 片在宿主槽内**首尾相接**(每条快照 = 一个槽 × world 平面)
                // 平面基址 = 该 rank 的快照字节(漏了 = peer 覆盖 rank0 的内容)
                let mut off = plane * self.snap_bytes;
                for p in &mut pieces {
                    p.conn_off = off;
                    off += p.len;
                }
                Ok(pieces)
            })
            .collect::<StoreResult<Vec<_>>>()?;
        Ok((all, SlotTable::one(buf)))
    }

    /// **快照下沉记账**(搬运成功后;纯 host):内容落宿主 + 换层。
    pub fn commit_snap_offload(
        &mut self,
        key: SnapKey,
        table: SlotTable,
    ) -> StoreResult<()> {
        let buf = table
            .into_slots()
            .into_iter()
            .next()
            .ok_or_else(|| StoreError::Invariant("快照槽台为空".into()))?;
        self.snap_host.put(key, buf)?;
        self.state.set_tier(key, Tier::Host)?;
        Ok(())
    }

    /// **快照键重挂**(前缀无感删除;见 [`StateStore::rekey_snap`])。
    ///
    /// 按层分派:**设备**只动账(槽由引擎按 key 重标);**宿主**连内容一起
    /// 重挂(`HostTier::rekey`,零搬运);**盘**不由本方法做 —— 盘内容要真
    /// `rename`,归 engine 的异步 I/O(`plan_rekey_snap_disk` →
    /// `disk_io::rename_block` → `commit_rekey_snap_disk`),本方法对盘层返 `false`。
    ///
    /// 拒绝(全列,均**不动任何东西**):
    /// - `old` 不在账 ⇒ `false`;
    /// - `new` 已在账 ⇒ `false`(不许覆盖);
    /// - 宿主层账与驻留表不平(账说在宿主、表里没有)⇒ `false`(不许猜)。
    pub fn rekey_snap(&mut self, old: SnapKey, new: SnapKey) -> bool {
        if self.state.locate(new).is_some() {
            return false;
        }
        match self.state.locate(old) {
            Some(Tier::Gpu) => self.state.rekey_snap(old, new),
            Some(Tier::Host) => {
                if self.snap_host.rekey(old, new).is_err() {
                    return false;
                }
                // 前置已查 new 不在账 ⇒ 这里失败只可能是并发错账(单线程不会)
                self.state.rekey_snap(old, new)
            }
            Some(Tier::Disk) | None => false,
        }
    }

    /// **盘层快照重挂:计划**(`(旧路径, 新路径)`;真 I/O 归 engine 阻塞池)。
    pub fn plan_rekey_snap_disk(
        &self,
        old: SnapKey,
        new: SnapKey,
    ) -> StoreResult<(std::path::PathBuf, std::path::PathBuf)> {
        if self.state.locate(new).is_some() {
            return Err(StoreError::Invariant(format!("盘层重挂:{new:?} 已在账")));
        }
        let Some(d) = self.snap_disk.as_ref() else {
            return Err(StoreError::Unknown("盘层重挂:未接盘层".into()));
        };
        d.plan_rekey(old, new)
    }

    /// **盘层快照重挂:落账**(rename 成功后):盘账 + 状态账一起换键。
    pub fn commit_rekey_snap_disk(&mut self, old: SnapKey, new: SnapKey) -> StoreResult<()> {
        let Some(d) = self.snap_disk.as_mut() else {
            return Err(StoreError::Unknown("盘层重挂落账:未接盘层".into()));
        };
        d.commit_rekey(old, new)?;
        if !self.state.rekey_snap(old, new) {
            return Err(StoreError::Invariant(format!(
                "盘层重挂落账:状态账拒了 {old:?} → {new:?}(账不平)"
            )));
        }
        Ok(())
    }

    /// **盘层快照定向丢弃:计划**(路径;不在盘层 = `None`)。
    ///
    /// 删除后整条旧链都不可达 —— 盘上那份"旧边界的 GDN 态"同样成了孤儿,
    /// 不丢就会把盘预算永远压在死身份上。
    pub fn plan_drop_snap_disk(&self, key: SnapKey) -> Option<std::path::PathBuf> {
        self.snap_disk.as_ref()?.path_of(key).map(|p| p.to_path_buf())
    }

    /// **盘层快照定向丢弃:落账**(文件删掉或本来就不在)。
    pub fn commit_drop_snap_disk(&mut self, key: SnapKey) -> StoreResult<()> {
        if let Some(d) = self.snap_disk.as_mut() {
            d.commit_evict(&[key]);
        }
        self.state.drop_snap(key).map(|_| ())
    }

    /// **快照下沉回滚**(搬运失败):宿主缓冲随失败的 connector 释放,层不动。
    pub fn abort_snap_offload(&mut self, _key: SnapKey) {}

    /// **快照回灌计划**(宿主 → 设备槽 `slot`)。
    pub fn plan_snap_onload(
        &mut self,
        key: SnapKey,
        slot: u32,
    ) -> StoreResult<(Vec<Vec<SwapPiece>>, SlotTable)> {
        let buf = self.snap_host.take(key)?;
        let all = (0..self.world.max(1))
            .map(|plane| {
                let mut pieces = self.snap_pieces(slot, plane)?;
                let mut off = plane * self.snap_bytes;
                for p in &mut pieces {
                    p.conn_off = off;
                    off += p.len;
                }
                Ok(pieces)
            })
            .collect::<StoreResult<Vec<_>>>()?;
        Ok((all, SlotTable::one(buf)))
    }

    /// **快照回灌记账**(搬运成功后):缓冲回池 + 层回设备。
    pub fn commit_snap_onload(&mut self, key: SnapKey, table: SlotTable) -> StoreResult<()> {
        self.snap_host.recycle_many(table.into_slots());
        self.state.set_tier(key, Tier::Gpu)?;
        Ok(())
    }

    /// **快照回灌回滚**(搬运失败):台子已随失败的 connector 移出(内容丢)
    /// ⇒ 条目一并删(不留假账;与 KV 腿 `abort_offload` 同律)。
    pub fn abort_snap_onload(&mut self, key: SnapKey) {
        let _ = self.state.drop_snap(key);
    }

    /// 丢弃一条快照的宿主副本(内容整体淘汰时;不在宿主 = 幂等)。
    pub fn drop_snap_host(&mut self, key: SnapKey) -> StoreResult<usize> {
        match self.snap_host.evict(key) {
            Ok(n) => Ok(n),
            Err(StoreError::Unknown(_)) => Ok(0),
            Err(e) => Err(e),
        }
    }

    /// 报告(两腿 + 保留/弹性面)。
    ///
    /// KV 块面从**腿**报出(单一真源),`tiers` 只列 store 自管面
    /// (状态腿/保留面/弹性面)—— 同一字节不两处记(§13.1/§23.1)。
    pub fn report(&self) -> StoreReport {
        // **第 0.5 刀:腿自报 → owner 行(对账式记账)**
        //
        // 为什么先"对账"而不是"事件记账":引擎有几十处直接 poke 腿
        // (`bank.rel.leg.*` / `bank.rel.state.*`),事件记账要逐个接线(后续);
        // 而对账在**报告时刻**取腿的权威状态 ⇒ 零漏记风险(腿是唯一真源),
        // 已能回答"谁占着多少"与"逐轮是否爬坡"(泄漏判据)。
        // 事件式记账的收益 = 精确区分"仍被持有"vs"未归还",留作接线后的第二步。
        let mut tiers = crate::tier::book_lock(&self.ledger).tiers().report();
        let legs = self.leg_accounting();
        for row in &legs.owners {
            tiers.rows.push(crate::tier::TierRow {
                tier: row.tier,
                kind: row.kind,
                bytes: row.bytes,
                entries: row.entries,
            });
        }
        tiers.owners.extend(legs.owners);
        tiers.owners.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.owner.cmp(&b.owner)));
        // **host/盘总账 = 合并后 rows 的层和**(2026-10-23 修)。
        //
        // 原实现只取 ledger 的 `host_used`:腿自报的 Host KvBlock / Host Snap
        // 只进了 `rows`/`owners`,没进总数 ⇒ `/debug/vram` 在 260k kvmem 档
        // 报 host_used≈204MiB,而实际 host 驻留≈5GiB(误导内存排查)。
        // GPU 不重算:腿的 KvBlock/Snap/Row 在 `Pool` 预留块**内部**,
        // `gpu_used` 的语义是「账本预留面」,重算会把同一批字节算两遍。
        tiers.host_used = tiers
            .rows
            .iter()
            .filter(|r| r.tier == crate::tier::Tier::Host)
            .map(|r| r.bytes)
            .sum();
        tiers.disk_used = tiers
            .rows
            .iter()
            .filter(|r| r.tier == crate::tier::Tier::Disk)
            .map(|r| r.bytes)
            .sum();
        StoreReport {
            tiers,
            kv: self.leg.kv_stats(),
            state: self.state.stats(),
            active_turns: self.admission.active_turns(),
            reserved_bytes: self.admission.reserved_bytes(),
            session_held_bytes: self.admission.session_held_bytes(),
            elastic_bytes: self.admission.elastic_bytes(),
            cache_bytes: self.admission.cache_bytes(),
            policy: self.evict.name().to_string(),
            host_resident_bytes: self.leg.host_used_bytes(),
            host_free_bytes: self.leg.host_free_bytes(),
            disk_kv: self.disk.as_ref().map(|d| d.stats()),
            disk_snap: self.snap_disk.as_ref().map(|d| d.stats()),
            events: {
                // 事件账(出借登记;与腿自报互为校验 —— 两个独立来源)
                let b = crate::tier::book_lock(&self.ledger);
                EventBook {
                    rows: b.open_rows(),
                    entries: b.open_count_rank(None),
                    bytes: b.open_total_bytes_rank(None),
                    hook_fail: b.hook_fail(),
                    partial_moves: b.partial_moves(),
                    ids: b.open_ids_by_rank(),
                    pool: {
                        let (n, by) = b.pool_stats();
                        let (hits, misses, drained) = b.pool_counters();
                        PoolReport {
                            entries: n,
                            bytes: by,
                            hits,
                            misses,
                            drained,
                            evicted: b.pool_evicted(),
                            cap: b.pool_cap(),
                            by_class: b.pool_by_class(),
                        }
                    },
                    open_lifecycle: b
                        .open_loans(|o| o.is_lifecycle())
                        .into_iter()
                        .map(|(id, l)| OpenLoan {
                            id,
                            name: l.name,
                            owner: l.owner,
                            kind: l.kind,
                            bytes: l.bytes,
                            rank: l.rank,
                        })
                        .collect(),
                }
            },
        }
    }

    /// **腿自报 → owner 行**(第 0.5 刀;对账式记账的原料)。
    ///
    /// 归属规则(有意保守,不猜):
    /// - KV 块(设备/宿主)= `Cache`(跨会话共享,无单会话归属);
    /// - 快照(设备/宿主)= `Cache`(Durable 可下沉 / Transient 逐出即丢,都是缓存面);
    /// - GDN 行 = `System`(逐会话归属需引擎的会话表;接线后再细化,不猜)。
    fn leg_accounting(&self) -> crate::tier::TierReport {
        let mut owners: Vec<crate::tier::OwnerRow> = Vec::new();
        let mut rows: Vec<crate::tier::TierRow> = Vec::new();
        let push = |owners: &mut Vec<crate::tier::OwnerRow>,
                    rows: &mut Vec<crate::tier::TierRow>,
                    owner: crate::tier::Owner,
                    tier: Tier,
                    kind: AssetKind,
                    bytes: usize,
                    entries: usize| {
            if bytes == 0 {
                return;
            }
            owners.push(crate::tier::OwnerRow { owner, tier, kind, bytes, entries });
            rows.push(crate::tier::TierRow { tier, kind, bytes, entries });
        };
        // ① KV 腿:设备驻留(gpu_bytes)+ 宿主驻留(host_used_bytes)
        let kv = self.leg.kv_stats();
        let kv_dev_entries = kv.cached_blocks.saturating_sub(kv.host_blocks);
        push(&mut owners, &mut rows, crate::tier::Owner::Cache, Tier::Gpu,
             AssetKind::KvBlock, kv.gpu_bytes, kv_dev_entries);
        {
            // 非设备面的 KV 块 = RAM 驻留(真源 = 宿主层账) + 磁盘驻留(DiskTier 账);
            // `kv.host_blocks`(tree 的 `slot = None` 计数)是两者之和 ⇒ 分开报,不重复计。
            let disk_entries = self.disk.as_ref().map(|d| d.resident_count()).unwrap_or(0);
            let disk_bytes = self.disk.as_ref().map(|d| d.used_bytes()).unwrap_or(0);
            let host_entries = kv.host_blocks.saturating_sub(disk_entries);
            push(&mut owners, &mut rows, crate::tier::Owner::Cache, Tier::Host,
                 AssetKind::KvBlock, self.leg.host_used_bytes(), host_entries);
            push(&mut owners, &mut rows, crate::tier::Owner::Cache, Tier::Disk,
                 AssetKind::KvBlock, disk_bytes, disk_entries);
        }
        // ② 状态腿:快照(设备/宿主)+ 行
        let st = self.state.stats();
        push(&mut owners, &mut rows, crate::tier::Owner::Cache, Tier::Gpu,
             AssetKind::Snap, st.snap_gpu.1, st.snap_gpu.0);
        push(&mut owners, &mut rows, crate::tier::Owner::Cache, Tier::Host,
             AssetKind::Snap, st.snap_host.1, st.snap_host.0);
        let rows_in_use = st.rows_total.saturating_sub(st.rows_free);
        push(&mut owners, &mut rows, crate::tier::Owner::System, Tier::Gpu,
             AssetKind::Row, rows_in_use * self.state.row_bytes(), rows_in_use);
        crate::tier::TierReport { rows, owners, ..Default::default() }
    }

    /// **复用池上限**(缺口 4 收尾 P1-1;boot 一次,显式配置)。`0` = 不限。
    pub fn set_pool_cap(&mut self, bytes: usize) {
        crate::tier::book_lock(&self.ledger).set_pool_cap(bytes);
    }

    /// **账本单元**(boot 接线:DeviceHandle 的事件入口与 store 的 `Rel` 要**同一本**)
    pub fn ledger_cell(&self) -> crate::tier::SharedLedger {
        self.ledger.clone()
    }

    /// 腿报告(KV 腿的完整面:槽/缓存/宿主/搬运计数 —— 报告与对账口用)。
    pub fn leg_stats(&self) -> LegStats {
        self.leg.report()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::budget::BudgetInputs;
    use crate::hooks::{EvictOut, Need, NeedWhy, Sink};
    use crate::id::ChainHash;
    use crate::policy::QueuePolicy;

    const BLOCK_BYTES: usize = 1 << 20;

    fn inputs() -> BudgetInputs {
        BudgetInputs {
            v_card_bytes: 25_000_000_000,
            weights_bytes: 17_500_000_000,
            runtime_floor_bytes: 2_000_000_000,
            draft_pool_bytes: 0,
            gdn_row_bytes: 162_000_000,
            kv_bytes_per_token: 32 * 1024,
            page_tokens: 32,
            host_bytes: 8 << 20,
            disk_bytes: 0,
            kv_face_bytes: 0,
        }
    }

    /// 测试用账本单元(预算 = 入参的 host/disk 总量 —— 与 boot 同式)
    fn cell_for(i: &BudgetInputs) -> crate::tier::SharedLedger {
        crate::tier::shared_ledger(crate::tier::TierBudget {
            host_bytes: i.host_bytes,
            disk_bytes: i.disk_bytes,
        })
    }

    fn kv_cfg(slots: usize, cache_cap: usize) -> KvConfig {
        KvConfig {
            slots,
            block_tokens: 32,
            block_bytes: BLOCK_BYTES,
            cache: crate::leg::CachePolicy::Cap(cache_cap),
            host_bytes: 8 << 20,
            class: crate::leg::BlockClass::KvPage,
            label: "test",
        }
    }

    fn rel(turns: usize) -> Rel {
        let i = inputs();
        let mut r =
            Rel::open(i.clone(), AdmissionPolicy::new(turns, QueuePolicy::Reject), kv_cfg(64, 32), cell_for(&i))
                .expect("装配");
        // 注册池布局(单片段;测试只跑账/计划,不跑真搬运 —— 搬迁在 store_e2e 覆盖)
        r.leg
            .attach_pool(crate::leg::PoolLayout {
                class: crate::leg::BlockClass::KvPage,
                epoch: 0,
                pieces: vec![crate::leg::PieceSpec {
                    buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                    stride_bytes: BLOCK_BYTES,
                }],
            })
            .expect("布局");
        r
    }

    fn toks(n: usize) -> Vec<u32> {
        (0..n as u32).collect()
    }

    #[test]
    fn open_rejects_unmeasured_budget_and_geometry() {
        // 动态池档(无池面兜底):卡级缺项 = 拒
        let mut i = inputs();
        i.weights_bytes = 0;
        assert!(Rel::open(i.clone(), AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(8, 8), cell_for(&i)).is_err());
        // 几何非法 = 拒
        assert!(
            Rel::open(inputs(), AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(0, 8), cell_for(&inputs())).is_err()
        );
        // 每 token 字节与腿几何矛盾 = 拒(同一件事只能有一个入口)
        let mut i2 = inputs();
        i2.kv_bytes_per_token = 4096;
        assert!(Rel::open(i2.clone(), AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(8, 8), cell_for(&i2)).is_err());
        // 页宽与腿几何矛盾 = 拒(A7:地板按页算、链增长也按页 ⇒ 同一把尺)
        let mut i3 = inputs();
        i3.page_tokens = 16;
        assert!(Rel::open(i3.clone(), AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(8, 8), cell_for(&i3)).is_err());
        // **账本预算不一致 = 响亮拒**(boot 的算式与 BudgetInputs 漂了)
        let wrong = crate::tier::shared_ledger(crate::tier::TierBudget { host_bytes: 1, disk_bytes: 0 });
        assert!(
            Rel::open(inputs(), AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(8, 8), wrong).is_err(),
            "账本预算与分解不符必须拒"
        );
    }

    /// @cu: 固定池档:池是准入面,卡级可缺项 —— 但"未实测"必须可查(不许静默)。
    #[test]
    fn fixed_pool_rel_opens_without_card_measurements() {
        let mut i = inputs();
        i.v_card_bytes = 0;
        i.weights_bytes = 0;
        i.runtime_floor_bytes = 0;
        i.kv_face_bytes = 64 * BLOCK_BYTES;
        let r = Rel::open(i.clone(), AdmissionPolicy::new(1, QueuePolicy::Reject), kv_cfg(64, 32), cell_for(&i))
            .expect("固定池档应能装配");
        assert!(!r.budget().card_measured());
        assert_eq!(r.budget().face_bytes(), 64 * BLOCK_BYTES, "判据面 = 池");
    }

    #[test]
    fn facade_delegates_to_the_leg_single_source() {
        let mut r = rel(2);
        let tokens = toks(64);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 64).expect("块表");
        assert_eq!(r.leg_stats().slots_total, 64);
        r.register_chain(&tokens, &table).expect("登记");
        assert_eq!(r.report().kv.cached_blocks, 2, "缓存条目走腿报告,不另记一份");
        assert_eq!(r.leg_stats().cached_blocks, 2);
        // tiers 只记 store 自管面(KV 块面归腿)⇒ 这里恒 0
        assert_eq!(r.report().tiers.gpu_used, 0, "KV 块不在语义账里重复记");

        // 命中面:门面回 (块腿三元组, 关系层语义) 且两者一致
        let (hit, recall) = r.match_prefix(&tokens).expect("命中查询");
        assert_eq!(hit.matched_blocks, 2);
        assert_eq!(recall.blocks, 2);
        assert_eq!(recall.kv_tier, Some(Tier::Gpu));
        assert_eq!(recall.mode(), RecallMode::ContextOnly, "状态腿不在 ⇒ 上下文召回");
        // 会话表还在 ⇒ 活跃面不可下沉(Busy);归还后 ⇒ 可沉(搬迁本体在 store_e2e 覆盖)
        let tail = r.leg.tail_hash(&tokens, 2).expect("尾身份");
        // 会话表还在 ⇒ 活跃面:整条链不出票(§14.6 律②)
        assert!(matches!(crate::leg::plan_offload(&mut r.leg, tail), Err(StoreError::Busy(_))));
        r.release_table(&mut table).expect("归还");
        assert!(crate::leg::plan_offload(&mut r.leg, tail).is_ok(), "缓存独占可搬");
    }

    #[test]
    fn recall_semantics_are_the_relation_layer_business() {
        let mut r = rel(2);
        let tokens = toks(64);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 64).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        let tail = r.leg.tail_hash(&tokens, 2).expect("尾身份");
        let id = PrefixId::new(tail);

        let h = r.probe(id);
        assert_eq!(h.mode(), RecallMode::ContextOnly, "只 KV ⇒ 上下文召回");
        assert_eq!(h.blocks, 2);
        assert_eq!(r.probe(PrefixId::new(ChainHash::new(9999))).mode(), RecallMode::Miss);

        // 状态腿到位 ⇒ Exact
        let row = r.state.borrow_row().expect("借行");
        r.state.return_row(row, Some(id.snap_key())).expect("落快照");
        assert_eq!(r.probe(id).mode(), RecallMode::Exact);
    }

    #[test]
    fn plan_need_merges_both_legs_and_respects_protect() {
        let mut r = rel(2);
        let ta = toks(32);
        let tb: Vec<u32> = (100..132).collect();
        for t in [&ta, &tb] {
            let mut table = Vec::new();
            r.ensure_for_len(&mut table, 32).expect("块表");
            r.register_chain(t, &table).expect("登记");
        }
        let a = r.leg.tail_hash(&ta, 1).expect("A 身份");
        let b = r.leg.tail_hash(&tb, 1).expect("B 身份");
        let row = r.state.borrow_row().expect("借行");
        r.state.return_row(row, Some(SnapKey::at_boundary(ChainHash::new(9)))).expect("快照");

        // 压力节点(= 零缺口也搬):KV 1 槽 + 状态腿 162MB(一张快照)
        let v = r.plan_need(
            Need { slots: 1, snap_bytes: 162_000_000, why: NeedWhy::Pressure },
            &Protected { chains: vec![a], snaps: vec![] },
        );
        assert!(v.deny.is_none(), "可搬面充足:{:?}", v.deny);
        assert!(
            v.out.iter().any(|o| matches!(o, EvictOut::Kv { chain, .. } if *chain == b)),
            "B 是可沉链(A 被 protect):{v:?}"
        );
        assert!(v.out.iter().any(|o| o.is_snap()), "状态腿必须同表出票:{v:?}");
        assert!(
            v.out.iter().all(|o| !matches!(o, EvictOut::Kv { chain, .. } if *chain == a)),
            "protect 的链绝不出票:{v:?}"
        );
    }

    #[test]
    fn protected_active_turn_floor_is_never_evicted() {
        let mut r = rel(1);
        let tokens = toks(32);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 32).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        let chain = r.leg.tail_hash(&tokens, 1).expect("身份");
        r.admission.admit(1, 0, 0).expect("登记活跃 turn");
        r.admission.set_floor(1, BLOCK_BYTES).expect("实占更新");
        let v = r.plan_need(
            Need { slots: 1, snap_bytes: 0, why: NeedWhy::Pressure },
            &Protected { chains: vec![chain], snaps: vec![] },
        );
        assert!(
            v.out.iter().all(|o| !matches!(o, EvictOut::Kv { chain: c, .. } if *c == chain)),
            "保留额被逐 = 不变量违反:{v:?}"
        );
    }

    /// **状态腿出序唯一(§30.6-②)**:`snap_victim_order` 与 `plan_need` 的
    /// `Snap` 票**同序**(事实档 + 策略),且 Transient 永远先走。
    #[test]
    fn snap_victim_order_is_the_plan_order() {
        let mut r = rel(4);
        let k_old = SnapKey::at_boundary(ChainHash::new(1));
        let k_tr = SnapKey::at_boundary(ChainHash::new(2));
        let k_new = SnapKey::at_boundary(ChainHash::new(3));
        r.state.register_snap(k_old, BLOCK_BYTES, Tier::Gpu, SnapDurability::Durable);
        r.state.register_snap(k_tr, BLOCK_BYTES, Tier::Gpu, SnapDurability::Transient);
        r.state.register_snap(k_new, BLOCK_BYTES, Tier::Gpu, SnapDurability::Durable);

        let order = r.snap_victim_order(&[]);
        assert_eq!(order.first(), Some(&k_tr), "Transient 是事实档第一:{order:?}");
        assert_eq!(order, vec![k_tr, k_old, k_new], "同档内按 last_use(先登记先走)");

        // 与容量调度同序:plan_need 的 Snap 票(宿主关 ⇒ 全 Drop)必须逐条相同
        let v = r.plan_need(
            Need { slots: 0, snap_bytes: 3 * BLOCK_BYTES, why: NeedWhy::Pressure },
            &Protected::default(),
        );
        let planned: Vec<SnapKey> = v
            .out
            .iter()
            .filter_map(|o| match o {
                EvictOut::Snap { key, .. } => Some(*key),
                EvictOut::Kv { .. } => None,
            })
            .collect();
        assert_eq!(planned, order, "两处必须同一份判据:{v:?}");
    }

    /// **第三层门(S6,逐层律)**:宿主满 ⇒ 把**宿主里最冷的**挪到磁盘
    /// (腾宿主位,内容不丢);本轮**不出 Drop、不 deny**(留一拍)。
    #[test]
    fn plan_need_spills_host_kv_to_disk_when_host_full() {
        let mut i = inputs();
        i.disk_bytes = 8 << 20;
        let mut cfg = kv_cfg(64, 32);
        cfg.host_bytes = 1 << 20; // 宿主体量 = 单块(1MB)⇒ 满
        let mut r = Rel::open(
            i.clone(),
            AdmissionPolicy::new(2, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
        )
        .expect("装配");
        let attach = r.attach_disk("/tmp/owl-disk-test-kv", 4 << 20, 0).expect("接盘");
        assert_eq!(attach.kv_slots, 4, "盘配额 4MB / 单块 1MB");
        // 两条链(各 1 块)都放进宿主 ⇒ 宿主满、设备无货
        let ta = toks(32);
        let tb: Vec<u32> = (100..132).collect();
        for t in [&ta, &tb] {
            let mut table = Vec::new();
            r.ensure_for_len(&mut table, 32).expect("块表");
            r.register_chain(t, &table).expect("登记");
            r.release_table(&mut table).expect("归还");
        }
        let a = r.leg.tail_hash(&ta, 1).expect("A 身份");
        let _b = r.leg.tail_hash(&tb, 1).expect("B 身份");
        // A 入宿主(宿主满);B 留设备(轮一沉不动它 —— 宿主没位)
        r.leg
            .put_host(a, crate::host::HostBlock::from_bytes(vec![7u8; BLOCK_BYTES]))
            .expect("A 入宿主");
        assert_eq!(r.leg.host_free_slots(), 0, "宿主体量 1 块 ⇒ 满");
        assert_eq!(r.leg.kv_stats().host_blocks, 1, "B 仍在设备(宿主只有 A)");

        let v = r.plan_need(
            Need { slots: 1, snap_bytes: 0, why: NeedWhy::Pressure },
            &Protected::default(),
        );
        assert!(v.deny.is_none(), "腾过宿主 ⇒ 本轮不 deny:{:?}", v.deny);
        let disk_tickets = v
            .out
            .iter()
            .filter(|o| matches!(o, EvictOut::Kv { sink: Sink::Disk, span: 1, .. }))
            .count();
        assert_eq!(disk_tickets, 1, "应出一条「宿主 → 磁盘」票:{v:?}");
        assert!(
            v.out.iter().all(|o| o.sink() != Sink::Drop),
            "内容不丢优先:不得出 Drop 票:{v:?}"
        );
    }

    /// **宿主预算单位 = 链块（world × 单页）**：TP 下旧口径（除单平面页）高估
    /// world 倍 ⇒ 计划按“有位”出票、`lend_slots` 半途撞 `Exhausted`
    /// （双卡 1M 后第二个长 turn 的硬错，§23.11）。
    #[test]
    fn host_free_slots_is_counted_in_chain_slots_under_tp() {
        let i = inputs();
        let mut cfg = kv_cfg(64, 32);
        cfg.host_bytes = 4 * 2 * BLOCK_BYTES; // 4 个链块（每块 = 两平面）
        let mut r = Rel::open_world(
            i.clone(),
            AdmissionPolicy::new(1, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
            2,
        )
        .expect("装配 world=2");
        assert_eq!(r.leg.host_slot_bytes(), 2 * BLOCK_BYTES);
        assert_eq!(r.leg.host_free_slots(), 4, "预算 8MB / 链块 2MB");
        let t = toks(32);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 32).expect("块表");
        r.register_chain(&t, &table).expect("登记");
        r.release_table(&mut table).expect("归还");
        let a = r.leg.tail_hash(&t, 1).expect("身份");
        r.leg
            .put_host(a, crate::host::HostBlock::zeroed(2 * BLOCK_BYTES))
            .expect("入宿主");
        assert_eq!(r.leg.host_free_slots(), 3, "一块 = 两个平面，不是两格");
    }

    /// **盘块 = 宿主槽（world × 单页）**：TP 档也能接盘；溢写计划字节必须等于
    /// 一条链的宿主槽（否则写到盘上的只有半个链块 ⇒ 回填必截断）。
    #[test]
    fn disk_block_is_the_host_slot_under_tp() {
        let mut i = inputs();
        i.disk_bytes = 8 << 20;
        let mut cfg = kv_cfg(64, 32);
        cfg.host_bytes = 2 * 2 * BLOCK_BYTES;
        let mut r = Rel::open_world(
            i.clone(),
            AdmissionPolicy::new(1, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
            2,
        )
        .expect("装配 world=2");
        let at = r
            .attach_disk("/tmp/owl-disk-test-tp", 4 * 2 * BLOCK_BYTES, 0)
            .expect("TP 接盘");
        assert_eq!(at.kv_slots, 4, "4 个链块");
        let a = ChainHash::new(42);
        let p = r.disk.as_ref().expect("盘").plan_spill(a).expect("计划");
        assert_eq!(p.bytes, 2 * BLOCK_BYTES, "一个盘块 = 两个平面");
        r.disk.as_mut().expect("盘").commit_spill(&p).expect("落账");
        assert_eq!(r.disk.as_ref().expect("盘").free_slots(), 3);
    }

    /// **磁盘回填落宿主的兜底**：宿主满 ⇒ 先逐最冷（否则 onload 整单 `Exhausted`）。
    #[test]
    fn host_fill_relieves_cold_copy_when_full() {
        let i = inputs();
        let mut cfg = kv_cfg(64, 32);
        cfg.host_bytes = BLOCK_BYTES;
        let mut r = Rel::open(
            i.clone(),
            AdmissionPolicy::new(2, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
        )
        .expect("装配");
        let ta = toks(32);
        let tb: Vec<u32> = (100..132).collect();
        for t in [&ta, &tb] {
            let mut table = Vec::new();
            r.ensure_for_len(&mut table, 32).expect("块表");
            r.register_chain(t, &table).expect("登记");
            r.release_table(&mut table).expect("归还");
        }
        let a = r.leg.tail_hash(&ta, 1).expect("A 身份");
        let b = r.leg.tail_hash(&tb, 1).expect("B 身份");
        r.leg
            .put_host(a, crate::host::HostBlock::from_bytes(vec![7u8; BLOCK_BYTES]))
            .expect("A 入宿主");
        assert_eq!(r.leg.host_free_slots(), 0, "宿主已满");
        r.host_fill(b, crate::host::HostBlock::from_bytes(vec![9u8; BLOCK_BYTES]))
            .expect("先逐最冷再落");
        assert!(r.leg.host_contains(b), "回填内容已落宿主");
        assert!(!r.leg.host_contains(a), "最冷副本已让位");
        assert_eq!(r.leg.host_relief_blocks(), 1, "让位入账");
    }

    /// **轮零：宿主满 ⇒ 先削最冷宿主链，再沉设备链**（不许直接丢新块）。
    #[test]
    fn plan_need_prunes_cold_host_chain_to_make_room_for_device_chain() {
        let i = inputs();
        let mut cfg = kv_cfg(64, 32);
        cfg.host_bytes = BLOCK_BYTES; // 宿主只装一个链块
        let mut r = Rel::open(
            i.clone(),
            AdmissionPolicy::new(2, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
        )
        .expect("装配");
        let ta = toks(32);
        let tb: Vec<u32> = (100..132).collect();
        for t in [&ta, &tb] {
            let mut table = Vec::new();
            r.ensure_for_len(&mut table, 32).expect("块表");
            r.register_chain(t, &table).expect("登记");
            r.release_table(&mut table).expect("归还");
        }
        let a = r.leg.tail_hash(&ta, 1).expect("A 身份");
        let b = r.leg.tail_hash(&tb, 1).expect("B 身份");
        r.leg
            .put_host(a, crate::host::HostBlock::from_bytes(vec![7u8; BLOCK_BYTES]))
            .expect("A 入宿主");
        assert_eq!(r.leg.host_free_slots(), 0, "宿主已满");

        let v = r.plan_need(
            Need { slots: 1, snap_bytes: 0, why: NeedWhy::Pressure },
            &Protected::default(),
        );
        assert!(v.deny.is_none(), "腾过宿主 ⇒ 本轮不 deny:{:?}", v.deny);
        let drop_at = v.out.iter().position(|o| {
            matches!(o, EvictOut::Kv { sink: Sink::Drop, chain, .. } if *chain == a)
        });
        let host_at = v.out.iter().position(|o| {
            matches!(o, EvictOut::Kv { sink: Sink::Host, chain, .. } if *chain == b)
        });
        assert!(drop_at.is_some(), "应丢最冷宿主链:{v:?}");
        assert!(host_at.is_some(), "腾出宿主位后应下沉设备链:{v:?}");
        assert!(drop_at < host_at, "宿主让位票必须先执行:{v:?}");
    }

    /// **执行面兜底**：计划没预见（直接调 offload）时，宿主满要先逐最冷，
    /// 不许 `Exhausted` 硬崩（旧行为：第二个长 turn 直接失败）。
    #[test]
    fn plan_kv_offload_relieves_host_before_lending_slots() {
        let i = inputs();
        let mut cfg = kv_cfg(64, 32);
        cfg.host_bytes = BLOCK_BYTES;
        let mut r = Rel::open(
            i.clone(),
            AdmissionPolicy::new(2, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
        )
        .expect("装配");
        let ta = toks(32);
        let tb: Vec<u32> = (100..132).collect();
        for t in [&ta, &tb] {
            let mut table = Vec::new();
            r.ensure_for_len(&mut table, 32).expect("块表");
            r.register_chain(t, &table).expect("登记");
            r.release_table(&mut table).expect("归还");
        }
        let a = r.leg.tail_hash(&ta, 1).expect("A 身份");
        let b = r.leg.tail_hash(&tb, 1).expect("B 身份");
        r.leg
            .attach_pool(crate::leg::PoolLayout {
                class: crate::leg::BlockClass::KvPage,
                epoch: 0,
                pieces: vec![crate::leg::PieceSpec {
                    buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                    stride_bytes: BLOCK_BYTES,
                }],
            })
            .expect("布局");
        r.leg
            .put_host(a, crate::host::HostBlock::from_bytes(vec![7u8; BLOCK_BYTES]))
            .expect("A 入宿主");
        assert_eq!(r.leg.host_free_slots(), 0, "宿主已满");

        let p = r.plan_kv_offload(b).expect("先逐最冷再借台");
        assert_eq!(p.blocks.len(), 1, "B 的一整块拿到票");
        assert_eq!(r.leg.host_relief_blocks(), 1, "让位块数入账（可观测）");
        assert!(!r.leg.host_contains(a), "最冷宿主链已被丢");
        assert_eq!(p.table.len(), 1, "借到 1 个宿主槽台（未提交前 B 仍在设备）");
    }

    /// **第三层门(S6)**:状态腿宿主的 Durable 挪去盘;`Transient` **不写盘**。
    #[test]
    fn plan_need_spills_host_snapshot_to_disk_but_not_transient() {
        let mut i = inputs();
        i.disk_bytes = 8 << 20;
        let mut cfg = kv_cfg(8, 8);
        cfg.host_bytes = i.host_bytes; // 状态腿宿主 = 0(本测试直接造"已在宿主"的账)
        let mut r = Rel::open(
            i.clone(),
            AdmissionPolicy::new(2, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
        )
        .expect("装配");
        r.attach_snap_layout(crate::leg::PoolLayout {
            class: crate::leg::BlockClass::GdnRow,
            epoch: 1,
            pieces: vec![crate::leg::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: BLOCK_BYTES,
            }],
        })
        .expect("状态腿布局");
        r.attach_disk("/tmp/owl-disk-test-snap", 0, 4 << 20).expect("接盘");

        let k_dur = SnapKey::at_boundary(ChainHash::new(0xD1));
        let k_tr = SnapKey::at_boundary(ChainHash::new(0x77));
        r.state.register_snap(k_dur, BLOCK_BYTES, Tier::Host, SnapDurability::Durable);
        r.state.register_snap(k_tr, BLOCK_BYTES, Tier::Host, SnapDurability::Transient);

        let v = r.plan_need(
            Need { slots: 0, snap_bytes: 2 * BLOCK_BYTES, why: NeedWhy::Pressure },
            &Protected::default(),
        );
        assert!(v.deny.is_none(), "{:?}", v.deny);
        assert!(
            v.out.iter().any(|o| matches!(o, EvictOut::Snap { key, sink: Sink::Disk } if *key == k_dur)),
            "Durable 应出盘票:{v:?}"
        );
        assert!(
            v.out.iter().all(|o| !matches!(o, EvictOut::Snap { sink: Sink::Disk, key } if *key == k_tr)),
            "Transient 不写盘:{v:?}"
        );
    }

    /// **快照淘汰门(2026-10-23 改判)**:快照**不走** KV 块的"末端先削"结构档 ——
    /// 每条快照按自己的边界身份独立键控,逐浅的不会让深的失效;而"深度大者先逐"
    /// 会先逐掉**最有用的一条**(会话上一轮尾巴 = 增量续写/前缀无感删除的 GDN
    /// 恢复点)。现行 = **Transient 最前 + 其余按新鲜度(LRU)**。
    #[test]
    fn snap_victim_order_is_recency_based() {
        let mut r = rel(4);
        let k = |n: u64| SnapKey::at_boundary(ChainHash::new(n));
        // 会话 1:depth 1/2/3(先登记 ⇒ last_use 小)
        for (i, d) in [1usize, 2, 3].into_iter().enumerate() {
            r.state.register_snap_owned(
                k(100 + i as u64),
                BLOCK_BYTES,
                Tier::Gpu,
                SnapDurability::Durable,
                1,
                d,
            );
        }
        // 会话 2:depth 1/2(后登记 ⇒ 更新)
        for (i, d) in [1usize, 2].into_iter().enumerate() {
            r.state.register_snap_owned(
                k(200 + i as u64),
                BLOCK_BYTES,
                Tier::Gpu,
                SnapDurability::Durable,
                2,
                d,
            );
        }
        // 把会话 1 的**末端**触达成最新:它既是"上一轮尾巴"(增量续写的复用面),
        // 也是前缀无感删除的 GDN 恢复点 ⇒ 必须**最后**被逐(旧策略会最先逐它)
        r.state.touch(k(102));
        let order = r.snap_victim_order(&[]);
        // 纯新鲜度:登记序 = 100,101,200,201;102 被触达 ⇒ 最后
        assert_eq!(
            order,
            vec![k(100), k(101), k(200), k(201), k(102)],
            "快照出序 = 新鲜度(LRU):刚用过的尾巴最后逐,深度不参与"
        );
    }

    /// **快照键重挂**(前缀无感删除):宿主层连内容一起换身份(零搬运),
    /// 设备层只动账(槽是池的活);目标已在账 / 账表不平 ⇒ 拒且原样不动。
    #[test]
    fn rekey_snap_moves_host_content_without_copy() {
        // 宿主腿要有预算(snap_host = 总量 − KV 腿分片);默认测试账本全给 KV 腿
        let mut i = inputs();
        i.host_bytes = 16 << 20;
        let mut cfg = kv_cfg(64, 32);
        cfg.host_bytes = 8 << 20;
        let mut r = Rel::open(
            i.clone(),
            AdmissionPolicy::new(4, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
        )
        .expect("装配");
        // 宿主层块粒度 = 一条快照字节(attach 前是占位 1 字节)
        r.attach_snap_layout(crate::leg::PoolLayout {
            class: crate::leg::BlockClass::GdnRow,
            epoch: 1,
            pieces: vec![crate::leg::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: BLOCK_BYTES,
            }],
        })
        .expect("状态腿布局");
        let old = SnapKey::at_boundary(ChainHash::new(0xA1));
        let new = SnapKey::at_boundary(ChainHash::new(0xA2));
        r.state.register_snap(old, BLOCK_BYTES, Tier::Host, SnapDurability::Durable);
        let buf = r.snap_host.acquire().expect("借缓冲");
        r.snap_host.put(old, buf).expect("驻留");
        assert!(r.rekey_snap(old, new), "宿主层重挂");
        assert_eq!(r.state.locate(new), Some(Tier::Host), "账随键走");
        assert!(r.snap_host.get(old).is_none(), "旧键内容必须已换身份");
        assert!(r.snap_host.get(new).is_some(), "新键拿到同一份内容");
        assert_eq!(r.snap_host.used_bytes(), BLOCK_BYTES, "零搬运");
        // 设备层只动账
        let g_old = SnapKey::at_boundary(ChainHash::new(0xB1));
        let g_new = SnapKey::at_boundary(ChainHash::new(0xB2));
        r.state.register_snap(g_old, BLOCK_BYTES, Tier::Gpu, SnapDurability::Durable);
        assert!(r.rekey_snap(g_old, g_new));
        assert_eq!(r.state.locate(g_new), Some(Tier::Gpu));
        // 目标已在账 ⇒ 拒(不许覆盖)
        assert!(!r.rekey_snap(g_new, new));
        // 账说宿主、驻留表里没有 ⇒ 拒(不许猜),且账原样不动
        let h = SnapKey::at_boundary(ChainHash::new(0xC1));
        r.state.register_snap(h, BLOCK_BYTES, Tier::Host, SnapDurability::Durable);
        assert!(!r.rekey_snap(h, SnapKey::at_boundary(ChainHash::new(0xC2))));
        assert_eq!(r.state.locate(h), Some(Tier::Host), "拒时必须原样不动");
    }

    /// **整枝塌缩**(2026-10-23):无宿主档下,一条 D 块冷链**一轮一张票、`span = D`**
    /// —— 不再是"一轮一块"(旧行为见台账 2026-10-23)。
    #[test]
    fn cold_deep_branch_collapses_in_one_planning_round() {
        // **无宿主/无磁盘档**:只剩 Drop 轮(有宿主时同义:整链一张 span 票)
        let mut i = inputs();
        i.host_bytes = 0;
        let mut cfg = kv_cfg(64, 32);
        cfg.host_bytes = 0; // 两条腿的宿主预算必须一致(装配期交叉校验)
        let mut r = Rel::open(
            i.clone(),
            AdmissionPolicy::new(4, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
        )
        .expect("装配");
        r.leg
            .attach_pool(crate::leg::PoolLayout {
                class: crate::leg::BlockClass::KvPage,
                epoch: 0,
                pieces: vec![crate::leg::PieceSpec {
                    buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                    stride_bytes: BLOCK_BYTES,
                }],
            })
            .expect("布局");

        let tokens = toks(8 * 32); // 8 块
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, tokens.len()).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        r.release_table(&mut table).expect("归还");

        let need = |slots: usize| crate::hooks::Need {
            slots,
            snap_bytes: 0,
            why: crate::hooks::NeedWhy::Pressure,
        };
        let v = r.plan_need(need(8), &Protected::default());
        let tickets: Vec<_> = v
            .out
            .iter()
            .filter_map(|o| match o {
                crate::hooks::EvictOut::Kv { chain, sink: crate::hooks::Sink::Drop, span } => {
                    Some((*chain, *span))
                }
                _ => None,
            })
            .collect();
        assert_eq!(tickets.len(), 1, "一条深链 ⇒ 一张票:{v:?}");
        assert_eq!(tickets[0].1, 8, "span = 整枝块数(一轮塌干净):{v:?}");
        // 执行
        let n = r.leg.prune_branch(tickets[0].0, tickets[0].1).expect("整枝塌缩");
        assert_eq!(n, 8, "实际丢 8 块");
        // 缓存已空 ⇒ 再要也不会有 KV 票
        let v2 = r.plan_need(need(8), &Protected::default());
        assert!(
            !v2.out.iter().any(|o| matches!(o, crate::hooks::EvictOut::Kv { .. })),
            "缓存空 ⇒ 没有 KV 票:{v2:?}"
        );
    }

    /// **有分层时不许整枝丢**(2026-10-23):宿主/磁盘可为内容留一条命时,Drop 票
    /// 保持逐块(让链先缩小到装得进宿主,再由下沉/搬盘接走);整枝塌缩只用于
    /// **没有分层可搬**的档位。
    #[test]
    fn drop_span_stays_one_when_a_content_preserving_tier_exists() {
        let mut r = rel(4); // host 8MiB ⇒ ~21 块;建一条 30 块的链(装不下)
        let tokens = toks(30 * 32);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, tokens.len()).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        r.release_table(&mut table).expect("归还");

        let v = r.plan_need(
            crate::hooks::Need { slots: 30, snap_bytes: 0, why: crate::hooks::NeedWhy::Pressure },
            &Protected::default(),
        );
        let drops: Vec<usize> = v
            .out
            .iter()
            .filter_map(|o| match o {
                crate::hooks::EvictOut::Kv { sink: crate::hooks::Sink::Drop, span, .. } => Some(*span),
                _ => None,
            })
            .collect();
        assert!(!drops.is_empty(), "宿主装不下整链 ⇒ 应走 Drop:{v:?}");
        assert!(drops.iter().all(|s| *s == 1), "有分层时 Drop 必须逐块(内容优先保留):{v:?}");
    }

    /// **整枝开关打开**(`bulk_drop_when_tier_full = true`):宿主装不下整链时,
    /// 允许一张票丢整段(否则要 O(深度) 个调度轮)。
    #[test]
    fn bulk_flag_lets_a_tier_short_chain_collapse_in_one_round() {
        let mut r = rel(4); // host ✓ ~21 槽;30 块链装不下
        r.set_bulk_drop_when_tier_full(true);
        let tokens = toks(30 * 32);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, tokens.len()).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        r.release_table(&mut table).expect("归还");
        let v = r.plan_need(
            crate::hooks::Need { slots: 30, snap_bytes: 0, why: crate::hooks::NeedWhy::Pressure },
            &Protected::default(),
        );
        let drops: Vec<usize> = v
            .out
            .iter()
            .filter_map(|o| match o {
                crate::hooks::EvictOut::Kv { sink: crate::hooks::Sink::Drop, span, .. } => Some(*span),
                _ => None,
            })
            .collect();
        assert_eq!(drops, vec![30], "开关打开 + 装不下 ⇒ 一轮整枝:{v:?}");
        assert!(r.leg.prune_branch(r.leg.tail_hash(&tokens, 30).expect("尾"), 30).is_ok());
    }

    /// **分叉点停住**:共享前缀 + 两条尾巴 ⇒ 塌缩一条尾巴**只丢它自己**,
    /// 共享前缀不动(它还有另一个孩子)。
    #[test]
    fn prune_branch_stops_at_the_branch_point() {
        let mut r = rel(4);
        let a = toks(2 * 32); // 共享前缀(2 块)
        let mut b = a.clone();
        b.extend((900u32..932).collect::<Vec<_>>()); // 第 3 个**整块**(分支 B)
        let mut c = a.clone();
        c.extend((940u32..972).collect::<Vec<_>>()); // 第 3 个整块(分支 C)
        for t in [&a[..], &b[..], &c[..]] {
            let mut table = Vec::new();
            r.ensure_for_len(&mut table, t.len()).expect("块表");
            r.register_chain(t, &table).expect("登记");
            r.release_table(&mut table).expect("归还");
        }
        let tail_b = r.leg.tail_hash(&b, 3).expect("B 尾");
        let tail_c = r.leg.tail_hash(&c, 3).expect("C 尾");
        let tail_a = r.leg.tail_hash(&a, 2).expect("A 尾");
        assert_eq!(r.leg.collapsible_span(tail_b), 1, "B 尾的父有 2 个孩子 ⇒ 只塌 1 块");
        assert_eq!(r.leg.collapsible_span(tail_a), 0, "共享前缀非叶 ⇒ 不可塌");
        let n = r.leg.prune_branch(tail_b, 4).expect("塌缩 B");
        assert_eq!(n, 1, "分叉点停住:只丢 B 的尾巴");
        // B 只剩共享前缀;C 整链不受影响
        let hb = r.match_prefix(&b).expect("B 命中");
        assert_eq!(hb.0.matched_blocks, 2, "B 剩共享前缀");
        let hc = r.match_prefix(&c).expect("C 命中");
        assert_eq!(hc.0.matched_blocks, 3, "C 完整");
        // B 已走 ⇒ C 的尾巴成了"唯一子链":塌它时共享前缀跟着一起走(合法:
        // 逐掉唯一孩子后父成为叶)。3 块 = C 尾 + 共享 2 块。
        let n = r.leg.prune_branch(tail_c, 4).expect("塌缩 C");
        assert_eq!(n, 3, "唯一枝尽 ⇒ 共享前缀可整段塌(3 块)");
        assert_eq!(r.leg.collapsible_span(tail_a), 0, "共享前缀已不在树");
        assert_eq!(r.match_prefix(&a).expect("A 查询").0.matched_blocks, 0, "缓存已空");
    }

    /// **KV 归属门**:`register_chain_owned` 记会话;深度 = 从根到该块的块数。
    #[test]
    fn kv_register_chain_owned_records_session_and_depth() {
        let mut r = rel(2);
        let tokens = toks(96); // 3 块
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 96).expect("块表");
        r.leg
            .register_chain_owned(&tokens, &table, 7)
            .expect("带归属登记");
        let tail = r.leg.tail_hash(&tokens, 3).expect("尾");

        assert_eq!(r.leg.session_of(tail), 7, "归属应记上");
        assert_eq!(r.leg.depth_of(tail), 3, "深度 = 从根到该块");
        // 无归属的块 = 0(不会被会话档提前);必须是**另一条链**(同一链首块已归 7)
        let other_tokens: Vec<u32> = (1000..1032).collect();
        let other = r.leg.tail_hash(&other_tokens, 1).expect("另一条");
        assert_eq!(r.leg.session_of(other), crate::policy::NO_SESSION);
    }

    /// **S6.1 候选门**:整链在宿主 ⇒ 以**叶**为锚出候选(span = 整链块数);
    /// 链里有块在设备 = 不出票(否则盘上只有半链)。
    #[test]
    fn host_chain_candidates_anchor_at_leaf_and_require_whole_chain() {
        let mut i = inputs();
        i.disk_bytes = 8 << 20;
        let mut r = rel_with_disk(&mut i, 4 << 20, 0);
        // A = 前 1 块;B = A 的延伸(A→B)
        let ta = toks(32);
        let tb = toks(64);
        for t in [&ta, &tb] {
            let mut table = Vec::new();
            r.ensure_for_len(&mut table, t.len()).expect("块表");
            r.register_chain(t, &table).expect("登记");
            r.release_table(&mut table).expect("归还");
        }
        let a = r.leg.tail_hash(&ta, 1).expect("A");
        let b = r.leg.tail_hash(&tb, 2).expect("B");
        // 只把 A 放宿主 ⇒ 整链不在宿主(B 在设备)⇒ 无候选
        r.leg.put_host(a, crate::host::HostBlock::from_bytes(vec![1u8; BLOCK_BYTES])).expect("A 入宿主");
        assert!(r.leg.host_chain_candidates(&[]).is_empty(), "半链不出票");
        // 再把 B 放宿主 ⇒ 候选 = (B, 2)
        r.leg.put_host(b, crate::host::HostBlock::from_bytes(vec![2u8; BLOCK_BYTES])).expect("B 入宿主");
        let c = r.leg.host_chain_candidates(&[]);
        assert_eq!(c.len(), 1, "{c:?}");
        assert_eq!(c[0].0, b, "锚必须是叶(B)");
        assert_eq!(c[0].1, 2, "span = 整链块数");
    }

    /// **S6.1 门:按盘上索引重建树**。内容在盘、索引完整 ⇒ 重建为**非设备层**
    /// (`slot = None`),命中面按重建的树工作。
    #[test]
    fn adopt_disk_chains_rebuilds_tree_from_index() {
        let mut i = inputs();
        i.disk_bytes = 8 << 20;
        let mut r = rel_with_disk(&mut i, 4 << 20, 0);
        let tokens: Vec<u32> = (0..96).collect(); // 3 块 × 32
        let mut parent = ChainHash::new(0);
        for blk in tokens.chunks(32) {
            let h = crate::kv::hash_block(parent, blk);
            parent = h;
            let plan = r.disk.as_ref().expect("盘").plan_spill(h).expect("计划");
            r.disk.as_mut().expect("盘").commit_spill(&plan).expect("落账");
        }
        let idx = crate::disk_index::ChainIndex::new(32, tokens.clone()).expect("索引");
        let rep = r.adopt_disk_chains(&[idx]).expect("重建");
        assert_eq!((rep.chains, rep.blocks), (1, 3), "{rep:?}");
        assert_eq!(rep.kept.len(), 3);
        assert_eq!(r.leg.kv_stats().host_blocks, 3, "重建条目应为非设备层");
        let (hit, _recall) = r.match_prefix(&tokens).expect("命中");
        assert_eq!(hit.matched_blocks, 3, "命中面应按重建的树工作");
    }

    /// **S6.1 截断**:索引 3 块、盘上只有前 2 块 ⇒ 重建 2 块 + 记截断。
    #[test]
    fn adopt_disk_chains_truncates_at_missing_content() {
        let mut i = inputs();
        i.disk_bytes = 8 << 20;
        let mut r = rel_with_disk(&mut i, 4 << 20, 0);
        let tokens: Vec<u32> = (0..96).collect();
        let mut parent = ChainHash::new(0);
        for (k, blk) in tokens.chunks(32).enumerate() {
            let h = crate::kv::hash_block(parent, blk);
            parent = h;
            if k < 2 {
                let plan = r.disk.as_ref().expect("盘").plan_spill(h).expect("计划");
                r.disk.as_mut().expect("盘").commit_spill(&plan).expect("落账");
            }
        }
        let idx = crate::disk_index::ChainIndex::new(32, tokens.clone()).expect("索引");
        let rep = r.adopt_disk_chains(&[idx]).expect("重建");
        assert_eq!((rep.chains, rep.blocks), (1, 2), "{rep:?}");
        assert_eq!(rep.truncated.len(), 1, "应记一条截断");
        // 只有前缀命中(第 3 块不在树里 ⇒ 断链)
        let (hit, _) = r.match_prefix(&tokens).expect("命中");
        assert_eq!(hit.matched_blocks, 2);
    }

    /// 装配带盘层的 Rel(测试辅助)。
    fn rel_with_disk(i: &mut BudgetInputs, kv_bytes: usize, snap_bytes: usize) -> Rel {
        let mut r = Rel::open(
            i.clone(),
            AdmissionPolicy::new(2, QueuePolicy::Reject),
            kv_cfg(64, 32),
            cell_for(i),
        )
        .expect("装配");
        r.leg
            .attach_pool(crate::leg::PoolLayout {
                class: crate::leg::BlockClass::KvPage,
                epoch: 0,
                pieces: vec![crate::leg::PieceSpec {
                    buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                    stride_bytes: BLOCK_BYTES,
                }],
            })
            .expect("布局");
        r.attach_disk("/tmp/owl-disk-test-adopt", kv_bytes, snap_bytes)
            .expect("接盘");
        r
    }

    /// **两腿同表(状态腿 Host 方向)** §30:状态腿宿主有位 ⇒ 快照出
    /// `Snap{Host}` 票(不再只能等 capture 撞满时被 engine 池被动逐)。
    #[test]
    fn plan_need_moves_durable_snapshot_to_host_when_room() {
        let i = inputs();
        // KV 腿只拿一半 host ⇒ 状态腿有 4 MiB 宿主编
        let mut cfg = kv_cfg(8, 8);
        cfg.host_bytes = 4 << 20;
        let mut r = Rel::open(
            i.clone(),
            AdmissionPolicy::new(2, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
        )
        .expect("装配");
        // 注册状态腿布局(一条快照 = 1 MiB)
        r.attach_snap_layout(crate::leg::PoolLayout {
            class: crate::leg::BlockClass::GdnRow,
            epoch: 1,
            pieces: vec![crate::leg::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: BLOCK_BYTES,
            }],
        })
        .expect("状态腿布局");
        assert!(r.snap_host_free_slots() >= 1, "状态腿宿主应可容至少一张");
        let key = SnapKey::at_boundary(ChainHash::new(9));
        r.state
            .register_snap(key, BLOCK_BYTES, Tier::Gpu, SnapDurability::Durable);

        let v = r.plan_need(
            Need { slots: 0, snap_bytes: BLOCK_BYTES, why: NeedWhy::Pressure },
            &Protected::default(),
        );
        assert!(v.deny.is_none(), "宿主有位 ⇒ 不拒:{:?}", v.deny);
        assert!(
            matches!(v.out.as_slice(), [EvictOut::Snap { key: k, sink: Sink::Host }] if *k == key),
            "状态腿应出 Host 票:{v:?}"
        );
    }

    /// **Transient 只丢不沉**(事实档,§30.1):生成段快照即使宿主有位也不下沉。
    #[test]
    fn plan_need_never_offloads_transient_snapshot() {
        let i = inputs();
        let mut cfg = kv_cfg(8, 8);
        cfg.host_bytes = 4 << 20;
        let mut r = Rel::open(
            i.clone(),
            AdmissionPolicy::new(2, QueuePolicy::Reject),
            cfg,
            cell_for(&i),
        )
        .expect("装配");
        r.attach_snap_layout(crate::leg::PoolLayout {
            class: crate::leg::BlockClass::GdnRow,
            epoch: 1,
            pieces: vec![crate::leg::PieceSpec {
                buf: owl_iface::contract::Bytes::new(0, 1, owl_iface::contract::Dtype::F32, 1),
                stride_bytes: BLOCK_BYTES,
            }],
        })
        .expect("状态腿布局");
        let key = SnapKey::at_boundary(ChainHash::new(11));
        r.state
            .register_snap(key, BLOCK_BYTES, Tier::Gpu, SnapDurability::Transient);

        let v = r.plan_need(
            Need { slots: 0, snap_bytes: BLOCK_BYTES, why: NeedWhy::Pressure },
            &Protected::default(),
        );
        assert!(
            v.out.iter().all(|o| !matches!(o, EvictOut::Snap { sink: Sink::Host, .. })),
            "Transient 不得下沉:{v:?}"
        );
        assert!(
            v.out.iter().any(|o| matches!(o, EvictOut::Snap { sink: Sink::Drop, .. })),
            "买不起宿主 ⇒ 仍有 Drop 票:{v:?}"
        );
    }

    #[test]
    fn report_surfaces_both_legs_and_faces() {
        let mut r = rel(2);
        let tokens = toks(32);
        let mut table = Vec::new();
        r.ensure_for_len(&mut table, 32).expect("块表");
        r.register_chain(&tokens, &table).expect("登记");
        r.admission.admit(1, 0, 0).expect("活跃 turn");
        r.admission.set_floor(1, 4 << 20).expect("实占");
        r.admission.set_elastic(1, 1 << 20).expect("弹性");
        let rep = r.report();
        assert_eq!(rep.active_turns, 1);
        assert_eq!(rep.reserved_bytes, 4 << 20);
        assert_eq!(rep.elastic_bytes, 1 << 20);
        assert_eq!(rep.policy, "lru");
        assert_eq!(rep.state.rows_total, 2);
        assert_eq!(rep.kv.cached_blocks, 1, "缓存面 1 块(腿报)");
        assert_eq!(rep.kv.gpu_bytes, BLOCK_BYTES, "设备绑定面 1 块");
        assert_eq!(rep.tiers.gpu_used, 0, "语义账不重复记 KV 块");
    }

    /// **host 总账必须含腿自报的 Host KvBlock**(2026-10-23)。
    ///
    /// 原 `report()` 只取 ledger 的 `host_used`(Index 行),腿的 Host KvBlock
    /// 只进 rows/owners ⇒ 260k kvmem 档报 204MiB,实际约 5GiB。
    #[test]
    fn report_host_used_includes_leg_host_blocks() {
        let mut r = rel(2);
        let chain = ChainHash::new(7);
        r.leg
            .host_put(chain, crate::host::HostBlock::zeroed(BLOCK_BYTES))
            .expect("放一块进 host");
        let rep = r.report();
        assert_eq!(rep.tiers.host_used, BLOCK_BYTES, "host 总账应含腿自报块");
        assert_eq!(rep.host_resident_bytes, BLOCK_BYTES, "腿自报同源");
        assert!(
            rep.tiers.rows.iter().any(|r| r.tier == Tier::Host
                && r.kind == AssetKind::KvBlock
                && r.bytes == BLOCK_BYTES),
            "报告应有 Host/KvBlock 行:{:?}",
            rep.tiers.rows
        );
        assert_eq!(rep.tiers.gpu_used, 0, "GPU 预留面语义不因 host 行走样");
    }

    /// **B-0 门(store 侧)**:索引几何注册 + 登记入账(host/Index 行)+ 出账平衡。
    #[test]
    fn index_registers_into_the_ledger_and_balances() {
        let mut r = rel(2);
        r.attach_index(2, 1, 2).expect("注册几何");
        assert_eq!(r.index_bytes(), 0);
        let b = ChainHash::new(42);
        let entries = vec![
            crate::select::MeanKEntry { layer: 0, kv_head: 0, k_bar: vec![0, 0] },
            crate::select::MeanKEntry { layer: 0, kv_head: 1, k_bar: vec![0, 0] },
        ];
        r.index_register(b, entries.clone()).expect("登记");
        let per_block = 2 * 2 * 2; // 条目 × dim × f16
        assert_eq!(r.index_bytes(), per_block);
        let rep = r.report();
        assert_eq!(rep.tiers.host_used, per_block, "索引进 host 层账");
        assert!(
            rep.tiers.rows.iter().any(|x| x.kind == AssetKind::Index && x.bytes == per_block),
            "报告应有 Index 行:{:?}",
            rep.tiers.rows
        );
        // 形状错 = 响亮拒(不进账)
        assert!(r.index_register(ChainHash::new(43), vec![entries[0].clone()]).is_err());
        assert_eq!(r.index_bytes(), per_block, "拒后账不动");
        // 出账
        r.index_drop(b).expect("出账");
        assert_eq!(r.index_bytes(), 0);
        assert_eq!(r.report().tiers.host_used, 0, "出账后 host 层归零");
    }

    #[test]
    fn strict_turn_is_rejected_when_vram_short() {
        let r = rel(1);
        let v = r.check(TurnFloor::Strict, 262_144);
        assert!(matches!(v, AdmissionVerdict::Reject { .. }), "得 {v:?}");
        let v = r.check(TurnFloor::Pages(2048), 262_144);
        assert!(matches!(v, AdmissionVerdict::Admit { .. }), "有耐受 ⇒ 准入,得 {v:?}");
    }
}
