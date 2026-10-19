//! 引擎钩子框架(§14)—— store 的插入语义:**"给你一个需求,谁该离开、谁该进来?"**
//!
//! ```text
//! 相一 plan_capacity(need, protect, room) -> Verdict   纯 host、零 I/O(任意时刻可调,含捕获窗内)
//! 相二 apply(face, &Verdict)                            段边界、async、显式收 face(搬运归 engine 执行)
//! ```
//!
//! # 判据唯一(2026-10-19 收口;§30)
//!
//! 在本次收口之前,"谁该走"有三套规范:`hooks::plan`(只吃 KV 腿)、
//! `Rel::plan_evict`(两腿合并,但**零调用点**)、`KvStore::ensure_for_len`(同步兜底)。
//! 现在**唯一产地 = [`plan_capacity`]**,并且:
//!
//! - **两腿同堂**:KV 链/叶候选与 GDN 快照候选进**同一张候选表**、同一份
//!   [`EvictPolicy`] 出序(策略对象从此真的决定生产路径,不再是死代码);
//! - **双目标**:`Need.slots`(KV 设备槽)与 `Need.snap_bytes`(状态腿设备字节)
//!   各自独立满足 —— KV 缺槽不能靠丢快照补,反之亦然;
//! - **批次档位**:`Transient` 快照(生成段,逐出即丢、零代价)先于其它候选;
//!   同档内才由策略对象排序(事实档 ≠ 策略)。
//!
//! 次序律(§12.3,唯一正确顺序):
//! **① 先下沉**(宿主有位就搬,内容不丢)→ **② 再驱逐**(宿主满,真丢)→ **③ 仍不足 ⇒ 拒**
//! (绝不动活跃面;抢占 = 另一个功能,S2b 的状态腿配合后再说)。
//!
//! 两相之间状态可变 ⇒ `apply` **重验**:引用变了(`Busy`)只**跳过该条**,不是错误。

use crate::id::{ChainHash, SnapKey};
use crate::leg::BlockLeg;
use crate::policy::{EvictCandidate, EvictPolicy, Leg};
use crate::state::{SnapDurability, StateStore};
use crate::tier::Tier;

/// 钩子需求(§12 需求侧的落点):**要几个设备槽/多少状态字节,为什么**。
///
/// 两个字段是**两腿的独立目标**(§30.2):
/// - `slots` = KV 设备槽(32 token 页)要腾出的量;
/// - `snap_bytes` = 状态腿(GDN 快照)要腾出的设备字节(0 = 不动快照)。
///
/// 压力/空转节点里它们是"**目标腾出量**"(零缺口也搬);准入/增长节点里
/// `slots` 读作"缺口"(已扣自由槽),`snap_bytes` 恒 0。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Need {
    /// 需要的**设备槽数**(KV 腿;周转量,不是总量)。
    pub slots: usize,
    /// 需要的**状态腿设备字节**(快照;0 = 不动状态腿)。
    pub snap_bytes: usize,
    /// 需求来源(打点/排障口径;H1 准入 / H2 增长 / H5 空转整理 / N6 压力)。
    pub why: NeedWhy,
}

impl Need {
    /// 构造(状态腿目标缺省 0 —— 绝大多数节点只动 KV)。
    pub fn kv(slots: usize, why: NeedWhy) -> Self {
        Self { slots, snap_bytes: 0, why }
    }

    /// 是否零需求(KV 与状态腿都不动)。
    pub fn is_zero(&self) -> bool {
        self.slots == 0 && self.snap_bytes == 0
    }
}

/// 需求来源(与 §14.3 的钩子表一一对应)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NeedWhy {
    /// H1:turn 准入(`Δ_j = ceil(N_j/32) − 命中块数`)。
    Admit,
    /// H2:块表增长(prefill chunk 间 / decode 跨页 / spec verify 行)。
    Grow,
    /// H5:空转整理(无活跃 turn;冷资产下沉)。
    Idle,
    /// **N6 压力节点**(缺口 3;2026-10-19):**零缺口也搬** —— 驱动/池告急时
    /// 主动把冷面腾到宿主(pressure 次序 = 先下沉、再驱逐、仍不足才拒),
    /// `slots`/`snap_bytes` 读作"**目标腾出量**"(不是缺口)。与其它节点的区别
    /// 只有一条:不吃 `free >= 目标` 早退(压力下"手上有货"也要整理)。
    Pressure,
}

/// 一块该去哪儿(次序:Host → Disk → Drop;§12.3 的三步唯一正确顺序)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sink {
    /// 下沉到宿主层(内容留,纳秒级可达)。
    Host,
    /// **再下沉到磁盘层**(宿主也满;内容留,毫秒级可达 —— S6)。
    Disk,
    /// 驱逐(宿主与磁盘都满;内容丢,下次重算)。
    Drop,
}

impl Sink {
    /// 是否"内容不丢"(用户 2026-10-19 裁决:不急着丢)。
    pub fn keeps_content(self) -> bool {
        matches!(self, Sink::Host | Sink::Disk)
    }
}

/// "谁该离开"的一条(**两腿同一种票**,§30.1)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictOut {
    /// KV 腿:链级下沉(`span = 整条链的设备块数`)或叶级驱逐(`span = 1`)。
    Kv {
        /// 块身份(链级条目 = **链尾**)。
        chain: ChainHash,
        /// 去哪。
        sink: Sink,
        /// 覆盖面:`span = 1` = 单块;`span > 1` = **整条链**(§14.6 链级卸载:
        /// 从链尾向链头搬完,`chain` 是链尾)。
        ///
        /// - `Sink::Host` / `Sink::Drop`:值 = **设备槽数**;
        /// - `Sink::Disk`(S6.1):值 = 链上**在宿主**的块数(整链搬盘 ⇒ 盘上可重建前缀)。
        span: usize,
    },
    /// 状态腿:快照(粒度 = 单张快照;`SnapKey` 是内容身份)。
    Snap {
        /// 快照键。
        key: SnapKey,
        /// 去哪(`Host` = 下沉宿主;`Drop` = 丢内容)。
        sink: Sink,
    },
}

impl EvictOut {
    /// 去哪(报告/打点口径)。
    pub fn sink(&self) -> Sink {
        match self {
            EvictOut::Kv { sink, .. } | EvictOut::Snap { sink, .. } => *sink,
        }
    }

    /// **KV 槽当量**(状态腿 = 0;两腿目标分开记账,不能互相折算)。
    pub fn kv_span(&self) -> usize {
        match self {
            EvictOut::Kv { span, .. } => *span,
            EvictOut::Snap { .. } => 0,
        }
    }

    /// 状态腿票?(报告/打点口径)
    pub fn is_snap(&self) -> bool {
        matches!(self, EvictOut::Snap { .. })
    }
}

/// **决策单**(§14.2):谁走 / 谁进 / 拒。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Verdict {
    /// 谁该离开(按序执行;`Busy` 的在 `apply` 里跳过)。
    pub out: Vec<EvictOut>,
    /// 谁该进来(需物化的 **KV** 回灌清单;H4 用。状态腿回灌走 `snap_onload` 快路)。
    pub in_: Vec<ChainHash>,
    /// 谁都不该动(**响亮拒**;含具体数字,§12.3 第③步)。
    pub deny: Option<String>,
}

impl Verdict {
    /// 是否无事可做。
    pub fn is_noop(&self) -> bool {
        self.out.is_empty() && self.in_.is_empty() && self.deny.is_none()
    }
}

/// `apply` 的实际结果(打点口径:**判**了多少、**搬**成了多少)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Applied {
    /// 实际下沉成功数(KV 块)。
    pub offloaded: usize,
    /// 实际驱逐数(KV 块)。
    pub dropped: usize,
    /// 实际回灌数(KV 块)。
    pub onloaded: usize,
    /// 实际**溢写到磁盘**的 KV 块数(内容移动:宿主 → 磁盘)。
    pub spilled: usize,
    /// 实际溢写到磁盘的快照张数。
    pub snap_spilled: usize,
    /// 实际下沉成功的快照张数。
    pub snap_offloaded: usize,
    /// 实际丢弃的快照张数。
    pub snap_dropped: usize,
    /// 因活跃面(`Busy`/不在设备)被跳过的条数。
    pub skipped_busy: usize,
}

/// **各层可容纳量**(两腿 × 两层;§30.2)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Room {
    /// KV 腿宿主**还能装几个块**。
    pub host_kv_slots: usize,
    /// 状态腿宿主**还能装几张快照**。
    pub host_snap_slots: usize,
    /// KV 腿磁盘**还能装几个块**(0 = 盘层关)。
    pub disk_kv_slots: usize,
    /// 状态腿磁盘**还能装几张快照**(0 = 盘层关)。
    pub disk_snap_slots: usize,
}

/// **相一:判**(纯 host、零 I/O)。给需求 ⇒ 谁走、谁进、还是拒。
///
/// # 判据(唯一产地)
///
/// 1. **候选同表**:KV 链级下沉候选(§14.6)+ 状态腿快照候选 + (宿主满后)
///    KV 叶级驱逐候选,全部转成 [`EvictCandidate`];
/// 2. **策略出序**:`Transient` 快照先走(事实档),同档内 [`EvictPolicy::order`]
///    (含 LRFU/召回信号)—— 策略对象第一次真正作用在生产路径;
/// 3. **先沉后逐**:轮一只出 `Sink::Host`(宿主有位且装得下);轮二对未满足的
///    目标出 `Sink::Drop`(KV 按**叶**,不动链头);
/// 4. **双目标独立**:`slots`(KV 槽)与 `snap_bytes`(状态腿字节)分别判"够没够";
/// 5. 仍不足 ⇒ `deny`(**响亮**,带两腿数字;不许动活跃面)。
///
/// `protect_chains` = 活跃 turn 的 KV 链 + 显式点名;`protect_snaps` = 点名保住的快照。
/// 两者之内的块**绝不**入 `out`。
///
/// # 为什么状态腿要进这张表
///
/// N6 压力/H5 空转的目标是"把设备腾出来",而设备上除了 KV 还有 GDN 快照
/// (27B 一条 ≈162 MB)。只搬 KV ⇒ 压力下快照面不在调度视野里,只能等
/// capture 撞满时被动逐(判据在 engine 的池里 = 第二套规范)。
pub fn plan_capacity(
    kv: &BlockLeg,
    state: &StateStore,
    policy: &dyn EvictPolicy,
    need: Need,
    protect_chains: &[ChainHash],
    protect_snaps: &[SnapKey],
    room: Room,
) -> Verdict {
    let mut v = Verdict::default();
    if need.is_zero() {
        return v;
    }
    let force = matches!(need.why, NeedWhy::Pressure | NeedWhy::Idle);
    // 非压力/空转:零缺口 ⇒ 不动任何人。状态腿目标只有压力/空转节点才给
    // (`snap_bytes > 0` 且非 force 属调用方错配 —— 仍按"要腾"处理,不静默忽略)。
    if !force && need.slots <= kv.slots_free() && need.snap_bytes == 0 {
        return v;
    }

    let slot_bytes = kv.slot_bytes().max(1);
    // 目标(force = 指定量,夹在"已用量"内;非 force = 缺口)
    let kv_goal = if force {
        need.slots
            .min(kv.slots_total().saturating_sub(kv.slots_free()))
    } else {
        need.slots.saturating_sub(kv.slots_free())
    };
    let snap_goal = need.snap_bytes;

    let mut got_slots = 0usize;
    let mut got_snap = 0usize;
    let mut picked: Vec<ChainHash> = protect_chains.to_vec();
    let mut kv_host_room = room.host_kv_slots;
    let mut snap_host_room = room.host_snap_slots;

    // ── 轮一:先下沉(宿主有位就搬,内容不丢) ─────────────────────────────
    let mut cands: Vec<EvictCandidate> = Vec::new();
    if got_slots < kv_goal {
        let want = (kv_goal - got_slots).max(1);
        for c in kv.chain_offload_candidates(&picked, want) {
            cands.push(EvictCandidate {
                leg: Leg::Kv,
                tier: Tier::Gpu,
                // 整链下沉腾出的设备字节(span 由 bytes/slot_bytes 还原)
                bytes: c.dev_blocks * slot_bytes,
                last_use: c.last_use,
                session: kv.session_of(c.tail),
                depth: kv.depth_of(c.tail),
                chain: Some(c.tail),
                snap: None,
            });
        }
    }
    if got_snap < snap_goal {
        cands.extend(
            state
                .evict_candidates(protect_snaps)
                .into_iter()
                .filter(|c| c.tier == Tier::Gpu),
        );
    }
    for c in order_candidates(policy, cands, state) {
        if got_slots >= kv_goal && got_snap >= snap_goal {
            break;
        }
        match c.leg {
            Leg::Kv => {
                let Some(tail) = c.chain else { continue };
                if got_slots >= kv_goal {
                    continue;
                }
                let span = c.bytes / slot_bytes;
                // 宿主装不下整链 ⇒ 本轮不出票(留给轮二;半截下沉无复用价值)
                if span == 0 || span > kv_host_room {
                    continue;
                }
                v.out.push(EvictOut::Kv { chain: tail, sink: Sink::Host, span });
                picked.push(tail);
                kv_host_room -= span;
                got_slots += span;
            }
            Leg::State => {
                let Some(key) = c.snap else { continue };
                if got_snap >= snap_goal || snap_host_room == 0 {
                    continue;
                }
                // 生成段快照 = 逐出即丢(下沉一条 159 MiB 换不回共享价值)
                if state.snap_durability(key) == Some(SnapDurability::Transient) {
                    continue;
                }
                v.out.push(EvictOut::Snap { key, sink: Sink::Host });
                snap_host_room -= 1;
                got_snap += c.bytes;
            }
        }
    }

    // ── 轮 1.5:**宿主满 ⇒ 把宿主里最冷的挪到磁盘**(腾宿主位;内容不丢) ────────
    //
    // 层级律:**逐层**(设备 → 宿主 → 磁盘)。设备上的块不直连磁盘 —— 先落宿主
    // (轮一),宿主满了再把宿主里最冷的挪到盘(本轮的活),于是**下一拍**宿主有位,
    // 轮一就能继续接设备块。代价 = 一拍延迟(需求/压力节点每拍都跑 ⇒ 实际可忽略),
    // 换来的是**内容永不先丢**(用户裁决 2026-10-19:不急着丢)。
    //
    // `Transient` 快照照旧不写盘(写 159 MiB 换不回共享价值 ⇒ 只配 Drop)。
    let mut host_relief = false;
    if room.disk_kv_slots > 0 || room.disk_snap_slots > 0 {
        let mut cands15: Vec<EvictCandidate> = Vec::new();
        if room.disk_kv_slots > 0 {
            // **整链搬盘**(S6.1):只挪宿主里的叶会拆散链 ⇒ 盘上只剩尾部,
            // 重启重建不出"从根起的前缀"(命中语义要求连续)⇒ 必须**整链都在宿主**
            // 才出票;链装不下 = 不出票(半链无复用价值,与 §14.6 同一律)。
            //
            // ⚠️ **不排除本拍轮一的票**:`apply_verdict` 按序执行(Host 在前、Disk 在后)
            // ⇒ 同拍可以"设备 → 宿主 → 磁盘"两跳完成;排除掉就永远只能搬"上一拍的宿主"
            // (真机定谳:轮一刚沉的链被 `picked` 挡掉 ⇒ 磁盘票恒零)。
            for (tail, span, last_use) in kv.host_chain_candidates(protect_chains) {
                if span == 0 || span > room.disk_kv_slots {
                    continue;
                }
                cands15.push(EvictCandidate {
                    leg: Leg::Kv,
                    tier: Tier::Host,
                    // 用 span 编码"整链块数"(engine 按 `chain_blocks` 逐块搬)
                    bytes: span * slot_bytes,
                    last_use,
                    session: kv.session_of(tail),
                    depth: kv.depth_of(tail),
                    chain: Some(tail),
                    snap: None,
                });
                if cands15.len() >= room.disk_kv_slots {
                    break;
                }
            }
        }
        if room.disk_snap_slots > 0 {
            // 同 KV 的次序理由:本拍轮一刚沉到宿主的快照也可以在 Disk 票里继续走
            // (执行序 Host → Disk),不因 `already_snap` 挡掉。
            cands15.extend(
                state
                    .evict_candidates(protect_snaps)
                    .into_iter()
                    .filter(|c| c.tier == Tier::Host)
                    .filter(|c| {
                        c.snap
                            .map(|k| state.snap_durability(k) != Some(SnapDurability::Transient))
                            .unwrap_or(true)
                    }),
            );
        }
        let mut disk_kv = room.disk_kv_slots;
        let mut disk_snap = room.disk_snap_slots;
        for c in order_candidates(policy, cands15, state) {
            if disk_kv == 0 && disk_snap == 0 {
                break;
            }
            match c.leg {
                Leg::Kv => {
                    let Some(tail) = c.chain else { continue };
                    let span = (c.bytes / slot_bytes).max(1);
                    if disk_kv == 0 || span > disk_kv {
                        continue;
                    }
                    v.out.push(EvictOut::Kv { chain: tail, sink: Sink::Disk, span });
                    picked.push(tail);
                    disk_kv -= span;
                    host_relief = true;
                }
                Leg::State => {
                    let Some(key) = c.snap else { continue };
                    if disk_snap == 0 {
                        continue;
                    }
                    v.out.push(EvictOut::Snap { key, sink: Sink::Disk });
                    disk_snap -= 1;
                    host_relief = true;
                }
            }
        }
    }

    // ── 轮二:宿主与磁盘都放不下 ⇒ 驱逐(内容真丢;次序律最后一步) ───────────
    //
    // **`host_relief` 优先**(内容不丢):本轮已把宿主里最冷的挪去磁盘 ⇒
    // 宿主位下一拍就回来。此时**不做 Drop、不 deny** —— 设备缺口留一拍
    // (节点每拍都跑);代价是那一拍 `ensure_for_len` 可能响亮拒,而丢掉的是内容。
    if !host_relief && (got_slots < kv_goal || got_snap < snap_goal) {
        let mut cands2: Vec<EvictCandidate> = Vec::new();
        if got_slots < kv_goal {
            // 叶级:丢尾部不动链头(前缀仍可复用)。已出票的链尾已在 `picked` 里。
            cands2.extend(
                kv.evict_candidates(&picked)
                    .into_iter()
                    .filter(|c| c.tier == Tier::Gpu && c.chain.is_some()),
            );
        }
        if got_snap < snap_goal {
            cands2.extend(
                state
                    .evict_candidates(protect_snaps)
                    .into_iter()
                    .filter(|c| {
                        c.tier == Tier::Gpu
                            && !v.out.iter().any(|o| matches!(o, EvictOut::Snap { key, .. } if Some(*key) == c.snap))
                    }),
            );
        }
        for c in order_candidates(policy, cands2, state) {
            if got_slots >= kv_goal && got_snap >= snap_goal {
                break;
            }
            match c.leg {
                Leg::Kv => {
                    let Some(tail) = c.chain else { continue };
                    if got_slots >= kv_goal {
                        continue;
                    }
                    v.out.push(EvictOut::Kv { chain: tail, sink: Sink::Drop, span: 1 });
                    got_slots += 1;
                }
                Leg::State => {
                    let Some(key) = c.snap else { continue };
                    if got_snap >= snap_goal {
                        continue;
                    }
                    v.out.push(EvictOut::Snap { key, sink: Sink::Drop });
                    got_snap += c.bytes;
                }
            }
        }
    }

    // ── ③ 仍不足 ⇒ 拒(不许动活跃面;腾过宿主的一拍不算"凑不够") ──────────
    if !host_relief && (got_slots < kv_goal || got_snap < snap_goal) {
        let mut why = format!(
            "{} 判据:KV 需求 {} 槽(自由 {}),状态腿需求 {} B;可搬面凑出 {} 槽/{} B(短 {} 槽/{} B);\
             宿主预算 KV {} B/{} 槽、状态腿 {} 张",
            kv.label(),
            kv_goal,
            kv.slots_free(),
            snap_goal,
            got_slots,
            got_snap,
            kv_goal.saturating_sub(got_slots),
            snap_goal.saturating_sub(got_snap),
            kv.host_budget_bytes(),
            room.host_kv_slots,
            room.host_snap_slots,
        );
        if force {
            why.push_str("(压力/空转:目标腾出量)");
        }
        v.deny = Some(why);
    }
    v
}

/// **按策略出序**(§30.1):事实档先分组,档内才交给策略对象。
///
/// 档 0 = 生成段快照(`Transient`:逐出即丢、零代价,先清理);
/// 档 1 = 其余(KV 链/叶 + `Durable` 快照)—— 同档内由 [`EvictPolicy::order`]
/// 决定(LRU / LRFU / 召回信号),这正是"策略是对象,不是分支"的落点。
pub(crate) fn order_candidates(
    policy: &dyn EvictPolicy,
    cands: Vec<EvictCandidate>,
    state: &StateStore,
) -> Vec<EvictCandidate> {
    // ① **事实档**:`Transient` 快照最前(生成段,逐出即丢、零代价)
    let mut first: Vec<EvictCandidate> = Vec::new();
    let mut rest: Vec<EvictCandidate> = Vec::new();
    for c in cands {
        let transient = c.leg == Leg::State
            && c.snap
                .map(|k| state.snap_durability(k) == Some(SnapDurability::Transient))
                .unwrap_or(false);
        if transient {
            first.push(c);
        } else {
            rest.push(c);
        }
    }
    policy.order(&mut first);

    // ② **会话档**(2026-10-19 会话感知淘汰):
    //    同一会话内**深度大者先逐**(末端先走 —— 保留前缀,不让"头部被逐 ⇒ 后面全垃圾");
    //    会话之间按"该会话最冷的候选"(越冷越先);无归属(session = 0)排最后。
    // ③ **策略档**:同一 (会话, 深度) 组内才交给 `EvictPolicy`(LRU / LRFU / 召回信号)。
    //
    // 为什么 depth 优先于策略:策略是"跨会话的新鲜度",而"从末端削"是**单会话内**的
    // 结构约束 —— 结构约束先满足,新鲜度只在同层同深度的候选间决胜负。
    let mut sess_cold: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    for c in &rest {
        let e = sess_cold.entry(c.session).or_insert(u64::MAX);
        if c.last_use < *e {
            *e = c.last_use;
        }
    }
    let mut groups: std::collections::BTreeMap<(u64, usize), Vec<EvictCandidate>> =
        std::collections::BTreeMap::new();
    for c in rest {
        groups.entry((c.session, c.depth)).or_default().push(c);
    }
    let mut keys: Vec<(u64, usize)> = groups.keys().copied().collect();
    // **深度在前、会话在后**(2026-10-19 用户语义精修):先按**全局深度降序**
    // —— 每个会话都先削自己的**最深边界**(末端),浅的(前缀)留到最后 ⇒
    // "优先留存已计算的前缀内容" 对**每个会话**成立,而不是"最冷会话被一路削光"。
    // 同一深度层内才按会话冷热(冷者先),再交给策略对象。
    keys.sort_by_key(|(s, d)| {
        (std::cmp::Reverse(*d), *s == crate::policy::NO_SESSION, *sess_cold.get(s).unwrap_or(&u64::MAX))
    });
    let mut ordered: Vec<EvictCandidate> = Vec::with_capacity(groups.values().map(|g| g.len()).sum());
    for k in keys {
        if let Some(mut g) = groups.remove(&k) {
            policy.order(&mut g);
            ordered.extend(g);
        }
    }
    first.into_iter().chain(ordered).collect()
}

// **自证(§30.4)**:判据唯一由本文件的 `plan_capacity` 承担 ——
// `Rel::plan_need` 是唯一生产入口,`Rel::plan_evict`/旧 `hooks::plan` 已删除;
// `KvStore::ensure_for_len` 保留为**兜底**(报告面 `ops.ensure_evicted > 0`
// = 调度器失职的告警信号,不是另一套判据)。
