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
/// - `snap_bytes` = 状态腿(GDN 快照)要腾出的设备字节(0 = 不动快照);
/// - `host_kv_slots` = **KV 宿主前缀块数**(HostPressure;只出 `Sink::Disk` 票);
/// - `host_snap_slots` = **快照宿主条数**(HostPressure;每张票 1 条)。
///
/// 压力/空转节点里它们是"**目标腾出量**"(零缺口也搬);准入/增长节点里
/// `slots` 读作"缺口"(已扣自由槽),其余三个目标恒 0。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Need {
    /// 需要的**设备槽数**(KV 腿;周转量,不是总量)。
    pub slots: usize,
    /// 需要的**状态腿设备字节**(快照;0 = 不动状态腿)。
    pub snap_bytes: usize,
    /// HostPressure:KV 宿主**从 root 起的前缀块数**。
    pub host_kv_slots: usize,
    /// HostPressure:快照宿主条数(每张 Disk 票一条)。
    pub host_snap_slots: usize,
    /// 需求来源(打点/排障口径;H1 准入 / H2 增长 / H5 空转整理 / N6 压力)。
    pub why: NeedWhy,
}

impl Need {
    /// 构造(状态腿/宿主目标缺省 0 —— 绝大多数节点只动 KV 设备槽)。
    pub fn kv(slots: usize, why: NeedWhy) -> Self {
        Self {
            slots,
            snap_bytes: 0,
            host_kv_slots: 0,
            host_snap_slots: 0,
            why,
        }
    }

    /// **宿主压力票**：KV 前缀 / 快照条数分腿给目标；只把宿主冷面挪去磁盘。
    pub fn host(kv_slots: usize, snap_slots: usize) -> Self {
        Self {
            slots: 0,
            snap_bytes: 0,
            host_kv_slots: kv_slots,
            host_snap_slots: snap_slots,
            why: NeedWhy::HostPressure,
        }
    }

    /// 是否零需求(KV / 状态腿 / 两条宿主腿都不动)。
    pub fn is_zero(&self) -> bool {
        self.slots == 0
            && self.snap_bytes == 0
            && self.host_kv_slots == 0
            && self.host_snap_slots == 0
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
    /// **宿主压力**(P1-3 后续):宿主满/低 ⇒ 只把宿主冷面挪到磁盘,
    /// **跳过 GPU→Host**,不把设备面再压进已经紧的宿主。
    HostPressure,
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
        /// - `Sink::Disk`(S6.1):值 = 从**根**起要搬盘的块数;整链压力票 = 链长,
        ///   HostPressure 可只取前缀(盘上仍是 root 连续的可用前缀)。
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
    /// **整枝塌缩开关**(`[kvmem].bulk_drop_when_tier_full`):链这一轮搬不走时
    /// 是否允许一张票丢整段。`false` = 只逐块(内容优先保留)。
    pub bulk_drop_when_tier_full: bool,
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
    let force = matches!(
        need.why,
        NeedWhy::Pressure | NeedWhy::Idle | NeedWhy::HostPressure
    );
    // 非压力/空转:零缺口 ⇒ 不动任何人。状态腿/宿主目标只有压力/空转节点才给
    // (`snap_bytes/host_*_slots > 0` 且非 force 属调用方错配 —— 仍按"要腾"处理,
    // 不静默忽略)。
    if !force
        && need.slots <= kv.slots_free()
        && need.snap_bytes == 0
        && need.host_kv_slots == 0
        && need.host_snap_slots == 0
    {
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
    // HostPressure 只腾宿主:设备 leg 的两个目标归零,轮一不产生 GPU→Host 票。
    // KV/快照分腿:两者的槽单位不同,不能合成一个 host_goal。
    let host_disk_only = need.host_kv_slots > 0 || need.host_snap_slots > 0;
    let host_kv_goal = need.host_kv_slots;
    let host_snap_goal = need.host_snap_slots;

    let mut got_slots = 0usize;
    let mut got_snap = 0usize;
    let mut got_host_kv = 0usize;
    let mut got_host_snap = 0usize;
    let mut picked: Vec<ChainHash> = protect_chains.to_vec();
    let mut kv_host_room = if host_disk_only { 0 } else { room.host_kv_slots };
    let mut snap_host_room = if host_disk_only {
        0
    } else {
        room.host_snap_slots
    };

    // ── 轮零：宿主满 ⇒ 先腾宿主位（否则轮一无位可出票） ─────────────────────
    //
    // 次序与三层级律一致：**能搬盘就搬盘（内容留）** → **再丢最冷（内容真丢）**。
    // 为什么必须在轮一之前：轮一只在 `host_kv_room` 内出票；宿主满时它一张不出，
    // 轮二只能丢**新算好的设备块**（更热），而旧的宿主冷链永远躺着 ——
    // 双卡 1M 实测：上一 turn 的宿主块一直不逐 ⇒ 第二个长 turn 硬错。
    let mut host_relief = false;
    let mut disk_kv = room.disk_kv_slots;
    if kv.host_enabled() && got_slots < kv_goal {
        let want = (kv_goal - got_slots).max(1);
        let mut short = want.saturating_sub(kv_host_room);
        if short > 0 && disk_kv > 0 {
            // 整链搬盘（与轮 1.5 同判据：半链无复用价值）；执行序在下文 Host 票之前
            for (tail, span, _) in kv.host_chain_candidates(&picked) {
                if short == 0 {
                    break;
                }
                if span == 0 || span > disk_kv {
                    continue;
                }
                v.out.push(EvictOut::Kv { chain: tail, sink: Sink::Disk, span });
                picked.push(tail);
                disk_kv -= span;
                kv_host_room += span;
                short -= span;
                host_relief = true;
            }
        }
        if short > 0 {
            // 盘也装不下（或无盘层）⇒ 丢宿主最冷：从链尾向内削，前缀留着
            for chain in kv.host_prune_candidates(&picked, short) {
                v.out.push(EvictOut::Kv { chain, sink: Sink::Drop, span: 1 });
                picked.push(chain);
                kv_host_room += 1;
            }
        }
    }

    // ── 轮一：先下沉（宿主有位就搬，内容不丢） ─────────────────────────────
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
    let mut host_relief_host = false;
    if disk_kv > 0 || room.disk_snap_slots > 0 {
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
                if span == 0 || span > disk_kv || picked.contains(&tail) {
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
                    let take = if host_disk_only {
                        if got_host_kv >= host_kv_goal {
                            continue;
                        }
                        span.min(host_kv_goal - got_host_kv)
                    } else {
                        span
                    };
                    if take == 0 {
                        continue;
                    }
                    v.out.push(EvictOut::Kv { chain: tail, sink: Sink::Disk, span: take });
                    picked.push(tail);
                    disk_kv -= take;
                    got_host_kv += take;
                    host_relief_host = true;
                }
                Leg::State => {
                    let Some(key) = c.snap else { continue };
                    if disk_snap == 0 {
                        continue;
                    }
                    if host_disk_only && got_host_snap >= host_snap_goal {
                        continue;
                    }
                    v.out.push(EvictOut::Snap { key, sink: Sink::Disk });
                    disk_snap -= 1;
                    got_host_snap += 1;
                    host_relief_host = true;
                }
            }
        }
    }

    // ── 轮二:宿主与磁盘都放不下 ⇒ 驱逐(内容真丢;次序律最后一步) ───────────
    //
    // **`host_relief` 优先**(内容不丢):本轮已把宿主里最冷的挪去磁盘 ⇒
    // 宿主位下一拍就回来。此时**不做 Drop、不 deny** —— 设备缺口留一拍
    // (节点每拍都跑);代价是那一拍 `ensure_for_len` 可能响亮拒,而丢掉的是内容。
    if !host_disk_only
        && !host_relief
        && !host_relief_host
        && (got_slots < kv_goal || got_snap < snap_goal)
    {
        let mut cands2: Vec<EvictCandidate> = Vec::new();
        if got_slots < kv_goal {
            // 叶级:丢尾部不动链头(前缀仍可复用)。已出票的链尾已在 `picked` 里。
            // **P0 修正**：活跃 turn/会话持着的块 `refs > 1`，plan 不得把它
            // 列进 Drop 票（执行侧只会 Busy 跳过，制造 plan 级过度承诺）。
            cands2.extend(
                kv.evict_candidates(&picked)
                    .into_iter()
                    .filter(|c| {
                        c.tier == Tier::Gpu
                            && c.chain
                                .is_some_and(|chain| kv.device_leaf_is_cache(chain))
                    }),
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
                    // **整枝塌缩**（2026-10-23）：一张票 = 从叶往上到**分叉点**的整段
                    // （只走"父无兄弟"的唯一子链；共享前缀永远不动）。执行侧
                    // `prune_branch` 再逐块校验引用（活跃会话持有则停）。
                    // **整枝塌缩只在"没有分层可搬"时启用**(2026-10-23):
                    // 有宿主/磁盘层时保持逐块(既有语义:先让链缩小到装得进宿主,
                    // 再由轮一/轮 1.5 搬走 —— 内容不丢);没有分层时逐块要 O(深度)
                    // 个调度轮,整枝一次丢才合理。
                    // 何时允许整枝：
                    // · **没有分层可搬**（宿主/磁盘都没有）⇒ 只能丢，一次丢干净；
                    // · 或开了 `bulk_drop_when_tier_full` **且这条链这一轮搬不走**
                    //   （`collapsible_span > 宿主剩余槽`：逐层律要求整链才搬，装不下
                    //   就等于搬不走 ⇒ 再逐块只会拖 O(深度) 个调度轮）。
                    // 有分层且装得下 ⇒ 保持逐块（轮一/轮 1.5 会把内容搬走）。
                    let span_full = kv.collapsible_span(tail);
                    let no_tier = room.host_kv_slots == 0 && room.disk_kv_slots == 0;
                    let tier_cannot_take = span_full > kv_host_room;
                    let bulk = no_tier || (room.bulk_drop_when_tier_full && tier_cannot_take);
                    let span = if bulk {
                        span_full.clamp(1, kv_goal - got_slots)
                    } else {
                        1
                    };
                    v.out.push(EvictOut::Kv { chain: tail, sink: Sink::Drop, span });
                    got_slots += span;
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
    let host_kv_short = got_host_kv < host_kv_goal;
    let host_snap_short = got_host_snap < host_snap_goal;
    if host_kv_short
        || host_snap_short
        || (!host_relief && !host_relief_host && (got_slots < kv_goal || got_snap < snap_goal))
    {
        let mut why = format!(
            "{} 判据:KV 需求 {} 槽(自由 {}),状态腿需求 {} B,宿主需求 {} KV 块/{} 快照;\
             可搬面凑出 {} 槽/{} B/{} KV 宿主块/{} 宿主快照\
             (短 {} 槽/{} B/{} KV/{} 快照);\
             宿主预算 KV {} B/{} 槽、状态腿 {} 张",
            kv.label(),
            kv_goal,
            kv.slots_free(),
            snap_goal,
            host_kv_goal,
            host_snap_goal,
            got_slots,
            got_snap,
            got_host_kv,
            got_host_snap,
            kv_goal.saturating_sub(got_slots),
            snap_goal.saturating_sub(got_snap),
            host_kv_goal.saturating_sub(got_host_kv),
            host_snap_goal.saturating_sub(got_host_snap),
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

    // ② **会话档 + 策略档（2026-10-23 改判：去掉"深度为主键"）**
    //
    // 前缀缓存是**树**（内容身份 = 父链哈希）：一个节点可以有多个孩子 = 分叉。
    // 在这种结构里做淘汰，只有两条规则站得住：
    //   1. **只有叶可逐**（结构律，由 `PrefixTree` 保证 —— 逐父节点会孤儿化整条子树）；
    //   2. **叶之间按"这一枝有多冷"排**（策略律；LRU 缺省 = 历史更久的那枝先走）。
    //
    // 旧实现把 `depth` 当**全局主键**（深者先逐），它的本意是"每个会话先削自己的
    // 末端"，但在树形分叉下 depth 与"冷热"无关：会把**刚用过的活尾巴**先逐掉，
    // 却留下一条更冷、更短的枝（与快照面 2026-10-23 修掉的是同一个病）。
    // 现在：会话之间按"该会话最冷的候选"（冷会话先，无主最后）；**会话内交给
    // `EvictPolicy`（缺省 LRU）**。`depth` 仍记账（诊断/口径），但不再参与排序。
    let mut sess_cold: std::collections::BTreeMap<u64, u64> = std::collections::BTreeMap::new();
    for c in &rest {
        let e = sess_cold.entry(c.session).or_insert(u64::MAX);
        if c.last_use < *e {
            *e = c.last_use;
        }
    }
    let mut groups: std::collections::BTreeMap<u64, Vec<EvictCandidate>> =
        std::collections::BTreeMap::new();
    for c in rest {
        groups.entry(c.session).or_default().push(c);
    }
    let mut keys: Vec<u64> = groups.keys().copied().collect();
    keys.sort_by_key(|s| (*s == crate::policy::NO_SESSION, *sess_cold.get(s).unwrap_or(&u64::MAX)));
    let mut ordered: Vec<EvictCandidate> = Vec::with_capacity(groups.values().map(|g| g.len()).sum());
    for k in keys {
        if let Some(mut g) = groups.remove(&k) {
            policy.order(&mut g);
            ordered.extend(g);
        }
    }
    first.into_iter().chain(ordered).collect()
}

/// **快照槽位出序**(专用;2026-10-23 —— 与 KV 块的"末端先削"脱钩)。
///
/// 为什么不能沿用 [`order_candidates`] 的会话深度档:那条规则是为 **KV 块**
/// 立的(父块被逐 ⇒ 子链不可达),对**快照**并不成立 —— 每条快照按自己的边界
/// 身份独立键控,逐掉浅的不会让深的失效。而"深度大者先逐"恰好会**先逐掉
/// 最有用的一条**:会话**上一轮的尾巴**既是指令续写的前缀复用面,也是
/// 前缀无感删除的 GDN 恢复点(实测:滑窗 6 轮后 `snap_max=4/16` 都把尾巴
/// 逐掉 ⇒ 删除退化为全量重算)。
///
/// 现行语义 = **事实档 + 新鲜度**:
/// 1. `Transient`(生成段快照,逐出零代价)最前;
/// 2. 其余按 `EvictPolicy`(缺省 LRU = `last_use`)—— 刚拍/刚用的尾巴自然
///    最后被逐,冷会话先走。
pub(crate) fn order_snap_victims(
    policy: &dyn EvictPolicy,
    cands: Vec<EvictCandidate>,
    state: &StateStore,
) -> Vec<EvictCandidate> {
    let mut transient: Vec<EvictCandidate> = Vec::new();
    let mut durable: Vec<EvictCandidate> = Vec::new();
    for c in cands {
        let t = c
            .snap
            .map(|k| state.snap_durability(k) == Some(SnapDurability::Transient))
            .unwrap_or(false);
        if t {
            transient.push(c);
        } else {
            durable.push(c);
        }
    }
    policy.order(&mut transient);
    policy.order(&mut durable);
    transient.into_iter().chain(durable).collect()
}

// **自证(§30.4)**:判据唯一由本文件的 `plan_capacity` 承担 ——
// `Rel::plan_need` 是唯一生产入口,`Rel::plan_evict`/旧 `hooks::plan` 已删除;
// `KvStore::ensure_for_len` 已改为**纯分配**(Phase 2,2026-10-25),
// 不再逐缓存；`ops.ensure_evicted` 是回归门（应恒 0）。

#[cfg(test)]
mod tests {
    use super::*;
    use crate::id::ChainHash;
    use crate::policy::EvictPolicyKind;

    fn cand(session: u64, depth: usize, last_use: u64, chain: u64) -> EvictCandidate {
        EvictCandidate {
            leg: Leg::Kv,
            tier: Tier::Gpu,
            bytes: 1,
            last_use,
            chain: Some(ChainHash::new(chain)),
            snap: None,
            session,
            depth,
        }
    }

    /// **树形分叉的出序**（2026-10-23 改判）：叶之间按**冷热**（历史更久的那枝先），
    /// 不按深度 —— 深度与"这一枝有多冷"无关，深度优先会把刚用过的活尾巴先逐掉。
    #[test]
    fn branch_victim_is_the_coldest_not_the_deepest() {
        let state = StateStore::new(1, 1);
        let policy = EvictPolicyKind::Lru.build();
        let cands = vec![
            cand(1, 90, 100, 1), // 很深,但刚用过(活尾巴)
            cand(1, 3, 5, 2),    // 很浅,但最冷(旧分叉)
            cand(1, 50, 200, 3), // 更深更新
        ];
        let order = order_candidates(policy.as_ref(), cands, &state);
        assert_eq!(
            order[0].chain,
            Some(ChainHash::new(2)),
            "先逐最冷的那枝(不按深度)"
        );
        assert_eq!(order[2].chain, Some(ChainHash::new(3)), "最新的最后走");
    }

    /// 会话之间：**冷会话先走**（无主最后），会话内仍是新鲜度。
    #[test]
    fn cold_session_is_evicted_before_hot_one() {
        let state = StateStore::new(1, 1);
        let policy = EvictPolicyKind::Lru.build();
        let cands = vec![
            cand(7, 40, 900, 1),  // 热会话
            cand(9, 2, 10, 2),    // 冷会话
            cand(7, 80, 950, 3),
        ];
        let order = order_candidates(policy.as_ref(), cands, &state);
        assert_eq!(order[0].chain, Some(ChainHash::new(2)), "冷会话先走");
    }
}
