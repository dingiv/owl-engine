//! 状态存储(三层:GPU 池 → host pinned → 磁盘)—— 立项卷
//! `docs/arch/store-状态存储-设计与迁移.md` 的代码落点。
//!
//! 本 crate **只做账、身份与策略**:东西在哪、这一步该看到什么、放不下时丢谁。
//! 不认模型层、不做前向、不持设备块句柄的所有权(所有权归 engine;见 `arch.md` 律②)。
//!
//! # 四个词(设计卷 §2.3 定义;本 crate 全程按此用词)
//!
//! - **会话(session)**:客户端身份 + 历史(块链/快照/索引);跨轮存在。
//! - **turn(轮次)**:一次提交(prefill → decode 到 stop)= 一个请求。
//! - **活跃 turn**:已提交且未终结的 turn —— **唯一占不可驱逐资源**的东西。
//! - **idle 会话**:turn 已终结、历史还在 —— **只占可驱逐资产**(快照 + 缓存块)。
//!
//! # 两腿独立 + 一层关系(裁决②)
//!
//! attention 块链(内容可哈希、可共享)与 GDN 状态快照(不可哈希、一格一会话)
//! 的几何/共享性/复算代价结构性不同 ⇒ **两本账各自独立**(`kv` / `state`),
//! 由 `rel` 按前缀身份关联,并判定召回语义(`Exact` / `ContextOnly`)。
//!
//! # 分界点(设计卷 §2.2):三条容量线 + 一条性能交叉线
//!
//! | 线 | 判据 | 越线后果 |
//! |---|---|---|
//! | ① 单请求 dense | `N·k ≤ V_avail` | 强制选择层(或拒/排队) |
//! | ② selection 常驻 | `W·k ≤ V_avail` 且 `N ≤ N_pos` | 触位置墙 ⇒ YaRN / 紧凑窗口 |
//! | ③ 多会话池 | `ΣN_i·k ≤ pool` 且 `活跃 turn ≤ gdn_slots` | A 轴驱逐序 |
//! | ④ 性能交叉 | `W < N ≤ N*` 两可 ⇒ 择优 | dense 保真 vs selection 省带宽 |
//!
//! # 两个用户旋钮(设计卷 §2.4)
//!
//! - boot:**`gdn_slots`**(最多几个 turn 同时在飞;硬天花板);
//! - per-turn:**`TurnFloor`**(直接显存地板 = KVMem 式损失的耐受度;
//!   `Strict` = 零耐受 ⇒ 禁选择层,不足即响亮拒)。

#![deny(clippy::unwrap_used)]
#![warn(missing_docs)]

pub mod budget;
pub mod error;
pub mod host;
pub mod moves;
pub mod id;
pub mod hooks;
pub mod kv;
pub mod leg;
pub mod policy;
pub mod rel;
pub mod select;
pub mod state;
pub mod tier;

pub use error::{StoreError, StoreResult};
pub use host::{HostBlock, HostTier};
pub use moves::{bytes_of, download_span, offload_to_host, onload_from_host, upload_span};
pub use budget::{check_admission, Admission, AdmissionVerdict, BudgetInputs, TurnFloor, VramBudget};
pub use id::{ChainHash, PrefixId, SnapKey, TOKENS_PER_BLOCK};
pub use kv::{hash_block, tail_hash, CacheUpdate, ChainCursor, KvStats, KvStore, MatchedBlock, PrefixMatch};
pub use leg::{
    abort_offload, abort_onload, commit_offload, commit_onload, is_host_placeholder, plan_offload,
    plan_offload_blocks, plan_onload, BlockOutPlan, SkipWhy, HOST_PLACEHOLDER,
    reclaim_slot_table, BlockClass, BlockLeg, Carrier, ChainCand, ChainHit, InPlan, LegConfig,
    LegOps, LegStats, OutPlan, PieceSpec, PoolLayout,
};
pub use hooks::{plan, Applied, EvictOut, Need, NeedWhy, Sink, Verdict};
pub use policy::{
    AdmissionPolicy, EvictCandidate, EvictPolicy, EvictPolicyKind, Leg, LrfuPolicy, LruPolicy,
    QueuePolicy,
};
pub use leg::CachePolicy;
pub use rel::{EvictItem, EvictPlan, KvConfig, Protected, RecallHit, RecallMode, Rel, StoreReport};
pub use select::{
    assemble_view, f32_to_f16, score_blocks, select, select_recency, BlockSet, IndexStats,
    MeanKEntry, MeanKIndex, block_mass, coverage_curve, derope_halves_inplace, derope_pair,
    f16_to_f32_pub, MeanKCollector, PosAction, PosFrame, QueryFace, ScoreMode, ScoreTuning,
    SelectBudget, ViewPlan,
};
pub use state::{SnapDurability, StateStore};
pub use tier::{
    AssetKind, LedgerHandle, Owner, OwnerRow, Tier, TierBudget, TierLedger, TierReport,
};
