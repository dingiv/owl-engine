//! 调度决策词汇 + 决策逻辑(S0,2026-09-30 调度层铺开立项施工;
//! 2026-10-01 拆分:`schedule()` 自 engine.rs 收编至此,决策词汇与
//! 决策逻辑同屋)。
//!
//! **决策/执行分离**(对标 vLLM v1 `Scheduler::schedule() → SchedulerOutput`):
//! - **决策**([`RunningEngine::schedule`],本模块):只动主机侧账
//!   (队列/会话账/块账房/活跃 turn),**零设备 IO**,同步可单测 ——
//!   输出 = 本 pump 步的完整计划;
//! - **执行**(exec.rs async 面):设备侧照办(reset/快照恢复/write_bt/
//!   prefill chunk/decode step/采样),无决策权。
//!
//! ≤8 并发收缩口径(需求 §二):无抢占、无优先级、无换出 ——
//! continuous batching 值得搬的只有档位拼批(S3)。

use owl_iface::contract::{DeviceClient, ModelError};

use crate::running::RunningEngine;

/// 待回灌块的**占位槽号**(只活在 schedule → execute_begin 之间;执行器在
/// `write_bt` 之前必须把全部占位替换成真槽 —— 否则写进设备的就是坏块表)。
pub(crate) const PENDING_LOAD_PLACEHOLDER: u32 = owl_store::HOST_PLACEHOLDER;

type Result<T> = std::result::Result<T, ModelError>;

/// 一次 pump 的调度决策(执行器照办的完整计划)
pub(crate) enum SchedulerOutput {
    /// 队列排空且无活跃(调用方可安全挂起等新提交)
    Idle,
    /// 本步计划(turn 启动可与本步动作同发 —— 保持历史语义:
    /// 开 turn 与首个 chunk/decode 在同一次 pump 完成)
    Step {
        begin: Option<BeginPlan>,
        action: StepAction,
    },
}

/// turn 启动执行指令(缓存复用判定已在决策面定案,事实记入
/// ActiveTurn;执行器只收设备侧动作所需的最小四元组)
pub(crate) struct BeginPlan {
    pub session_id: u64,
    pub gdn_slot: usize,
    /// 前缀命中时 = 快照恢复键(**边界块的内容身份**;物理块号会被回收复用,
    /// 不可作键 —— 见 `owl_store::ChainCursor` 头注);GuardHit/Fresh = None
    pub restore_key: Option<owl_store::SnapKey>,
    /// Fresh(零复用)→ 执行器重置会话 GDN 格;复用路径保持状态原样
    pub reset_gdn: bool,
    /// **待回灌清单**(S3-3b):命中面里内容在宿主层的块 = `(块表下标, 内容身份)`。
    /// 执行器在**段边界**逐块 `store::onload_block` 并把返回的新槽写回块表
    /// (在 `write_bt` 之前)⇒ 决策在 schedule(纯 host),搬运在 exec(段边界)。
    pub pending_loads: Vec<(usize, owl_store::ChainHash)>,
    /// **前缀无感删除的收尾动作**(leave-gap;`None` = 本 turn 没删)。
    /// 只携带盘层快照重挂票与旧链尾释放清单;不转 K、不移动 position。
    pub image_exec: Option<ImageExec>,
    /// **懒回灌的 turn 起点视图种子**(W3;`None` = 本 turn 不需要合成视图)。
    ///
    /// 语义:`plan_prefix_reuse` 只把 **sink + recent 窗** 回灌进设备,其余命中块
    /// 留在会话链上作占位符 ⇒ 执行器必须在**第一个 chunk 之前**立一个视图
    /// (否则 `past_view_of`/`select_view_step` 会退回 dense 链读占位符 = 静默错页)。
    pub lazy_view: Option<LazySeed>,
}

/// **懒回灌起点视图种子**(调度面算好,m 与 `pending_loads` **同一份算式**)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LazySeed {
    /// 保底 sink 块数(链头,永不进视图洞)。
    pub sink_blocks: usize,
    /// 尾部连续留驻的 token 数(= 视图召回点到 `cached_len` 的距离;页对齐)。
    pub recent_tokens: usize,
    /// 复用面 token 数(= 视图召回点之后的"当前位置"起点)。
    pub cached_len: usize,
    /// 视图中**驻留段**的块身份(`sink 块 ++ recent 窗` 同序;驻留校验/探针用)。
    pub blocks: Vec<owl_store::ChainHash>,
}

/// **leave-gap 删除计划的落账包**(不转 K;只摘页 + 位置账保留旧值)。
pub(crate) struct ImageLeaveGap {
    /// 拼接后的会话块表(前缀 ++ 后缀;会话引用已 `incref`)
    table: Vec<u32>,
    /// 删除区间(旧 token 坐标;观测账用)
    a: usize,
    b: usize,
    /// 设备层快照重挂条数(观测账用)
    rekeyed: usize,
    /// 复用面 token 数(= 表长 × 页)
    cached_len: usize,
    /// 拼接边界的链尾身份(新链)
    tail_hash: owl_store::ChainHash,
    /// GDN 恢复键(删除后的边界;设备层已重挂则用新键,否则旧键)
    restore_key: Option<owl_store::SnapKey>,
    /// 盘层快照收尾(重挂;**不做 K 重转**)
    snap_disk: Vec<SnapDiskOp>,
    /// 宿主占位页的段边界回灌清单(逻辑块下标, 内容身份)
    pending_loads: Vec<(usize, owl_store::ChainHash)>,
    /// 旧链尾身份:等宿主回灌完成后再释放引用(不能先丢宿主内容)
    release_chains: Vec<owl_store::ChainHash>,
}

/// leave-gap 删除的设备/盘层收尾(不转 K)。
#[derive(Clone, Debug, Default)]
pub(crate) struct ImageExec {
    /// **盘层快照的一次性收尾**(重挂 / 丢弃旧身份)。
    /// 真 I/O 在 `execute_begin` 最前(快照恢复之前)。
    pub snap_disk: Vec<SnapDiskOp>,
    /// 待释放旧链尾身份;`execute_begin` 在宿主回灌之后 drop。
    pub release_chains: Vec<owl_store::ChainHash>,
}

/// 盘层快照的删除收尾动作(决策面出票,执行面照办)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SnapDiskOp {
    /// 同一条快照换边界身份(`rename` 旧文件 → 新路径 + 两本账换键)
    Rekey {
        old: owl_store::SnapKey,
        new: owl_store::SnapKey,
    },
}

/// 快照重挂的**层分派**(纯;门与文档的唯一真源)。
///
/// 分派律(每层只给一条路,不允许两边都算一份):
/// - 设备:账 + 两 rank 槽标记(内容本就在槽上);
/// - 宿主:账 + 驻留内容一起换键(`HostTier::rekey`,零搬运);
/// - 盘:只出票(真 `rename` 在异步段,见 `exec::img_evict`);
/// - 其余(不在账 / 账说设备但槽上没有)= 不重挂(账不平不许猜)。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum RekeyTarget {
    Device,
    Host,
    Disk,
    Skip,
}

pub(crate) fn snap_rekey_target(tier: Option<owl_store::Tier>, dev_slot: bool) -> RekeyTarget {
    match tier {
        Some(owl_store::Tier::Gpu) if dev_slot => RekeyTarget::Device,
        Some(owl_store::Tier::Host) => RekeyTarget::Host,
        Some(owl_store::Tier::Disk) => RekeyTarget::Disk,
        _ => RekeyTarget::Skip,
    }
}

/// 本步动作(相位决策;执行器无脑照办)
pub(crate) enum StepAction {
    /// 灌一个 prefill chunk(块式;is_last = 末块,末行采样出首 token)
    Prefill {
        chunk_ids: Vec<u32>,
        /// KV 行基址(增量 turn 从 cached_len 起步)
        base: usize,
        is_last: bool,
    },
    /// decode 单步(host 派生量已备齐 —— 决策面可单测的关键)
    Decode {
        /// frontier token(上一步采样输出;本步是首步时 = 末块采样结果)
        token: u32,
        /// KV 行位(= 活跃 fed;物理槽换算决策面完成)
        pos: usize,
        /// 物理槽(块链换算;legacy = pos)
        kv_slot: u32,
        gdn_slot: usize,
        /// 块链跨页增长 → 执行器重写持久块表(write_bt)
        grew: bool,
    },
    /// 投机轮(E5-M2;spec_depth > 0 时替代 Decode):块 = [anchor,
    /// d1..d_depth],槽位 depth+1 个(pos..pos+depth)
    SpecRound {
        token: u32,
        pos: usize,
        kv_slots: [u32; owl_shared::config::SPEC_VERIFY_ROWS_MAX + 1],
        gdn_slot: usize,
        grew: bool,
        /// 本轮 active depth（自适应档 = d3/d7；图族按它选）
        depth: usize,
    },
    /// **跨会话合批 decode**(B5.1 M1;spec off):每行 = 一个会话的一步。
    /// 执行器一次 T=B 前向 + 逐行采样/事件(每行各自 `TurnEvent`)。
    BatchDecode { rows: Vec<RowSpec> },
}

/// 合批 decode 的一行(host 派生量已备齐;与单行 `Decode` 同语义)。
#[derive(Debug, Clone, Copy)]
pub(crate) struct RowSpec {
    pub turn: u64,
    pub session_id: u64,
    pub token: u32,
    pub pos: usize,
    pub kv_slot: u32,
    /// 本步可见 KV 长(= pos + 1;批内各行独立)
    pub kv_len: usize,
    pub gdn_slot: usize,
    /// 本步是否跨页(保洁需求面;生产路径不读,**测试**用于"页内不跨页"断言)
    #[allow(dead_code)]
    pub grew: bool,
}

impl<D: DeviceClient> RunningEngine<D> {
    /// **前缀命中的复用面**(单源:scheduler 与需求驱动保洁 §12.4 同用)。
    ///
    /// 返回 `(m, dev)`:
    /// - `m` = **可复用块数**(受两条裁剪:**完全对齐保留末块重算** /
    ///   **边界快照不在场则降一** —— 与旧 scheduler 逐条一致);
    /// - `dev` = 其中**设备已驻留**的块数(换入时**不占新槽**)。
    ///
    /// 为什么 `dev` 单独给:host 命中的块虽"命中",回灌时**照样要设备槽**
    /// ⇒ 需求侧只能减 `dev`,不能减 `m`(否则少腾槽 → 回灌时 Exhausted)。
    pub(crate) fn reuse_face(
        rel: &owl_store::Rel,
        hit: &owl_store::ChainHit,
        recall: &owl_store::RecallHit,
        prompt_len: usize,
        page: usize,
    ) -> (usize, usize) {
        let mut m = hit.matched_blocks;
        let tail = m;
        if m > 0 && m * page == prompt_len {
            m -= 1; // 完全对齐保留末块重算(非空 prefill;xinfer 同语义)
        }
        // **门面给的两腿语义(§25.2)**:尾部边界的快照**已在场**(设备或宿主
        // ——`RecallHit.snap_tier` 就是它)且没被上面的"末块重算"裁掉 ⇒
        // 复用面就是 m,不必再逐边界回查 store 的账(省 O(边界) 次查询)。
        // 这就是 `RecallMode::Exact` 的判据(块链 + 快照都在)在调度面的落点。
        if m > 0 && m == tail && recall.snap_tier.is_some() {
            let dev = hit.slots[..m].iter().filter(|s| s.is_some()).count();
            return (m, dev);
        }
        while m > 0 {
            // 快照键 = **边界块的内容身份**(物理块号会被回收复用 ⇒ 旧槽号会命中
            // 别人的状态:静默错值。store 卷 §9.1)。
            //
            // **S2(§25.2)**:在场性查 **store 的层账**(`StateStore`)——
            // 设备层与宿主层都算"在场"(宿主层在段边界回灌;`RecallMode::Exact`
            // 的判据就是这一条)。旧实现只看引擎的设备池 ⇒ 下沉后的快照被当成
            // "不在场"、白裁掉复用面。
            if rel
                .state
                .locate(owl_store::SnapKey::at_boundary(hit.hashes[m - 1]))
                .is_some()
            {
                break;
            }
            m -= 1; // 无快照的边界不可复用(GDN 态对不上)
        }
        let dev = hit.slots[..m].iter().filter(|s| s.is_some()).count();
        (m, dev)
    }

    /// **prefill 本步块长**（单源：调度决策与需求驱动保洁 §12.4 同用这一份）。
    ///
    /// `None` = 本步不是 prefill（已说完 prompt，或没有活跃 turn）。
    /// 块长 = min(配置块长, 剩余);`max(1)` 保证结尾剩 1 token 也推得动。
    ///
    /// **尾块合并（L4，2026-10-18）**：`tail_merge > 0` 且
    /// `剩余 ≤ chunk_cfg + tail_merge` ⇒ 本步**吃完剩余**（块长上限
    /// `chunk_cfg + tail_merge`）。动机：每 chunk 有一份与尺寸无关的固定
    /// 宿主税（实测 ~250ms：6395 条 alloc/launch/free 往返；**4-token 的
    /// 尾块同样是 249ms**，见 `perf-runs/prefill-prof-2026-10-18/REPORT.md` §4）
    /// ⇒ `< chunk` 的尾块纯亏。`tail_merge = 0` ⇒ 逐字节旧行为。
    ///
    /// **末段页对齐**(`page = 0` 可关;2026-10-19 前缀复用闭环):块边界快照
    /// 只在**页对齐**处拍。与 L4 **只在一处冲突** —— "整个 prompt 落进一个块"
    /// 时 L4 会一口吃完(尾端不对齐 ⇒ **一个快照都没有** ⇒ 下一轮全量重算:
    /// 实测 83 tok prompt 复用 0 / 12s)。⇒ 判据 = **"至今没有任何对齐边界"**
    /// (`!(fed > 0 && fed % page == 0)`):多 chunk 档前面的 chunk 尾本来就对齐
    /// (快照已在)⇒ L4 照旧、不多付那 250ms 宿主税;只有单块档切一刀。
    ///
    /// 切点取**最后一个整页边界**(而不是第一个):切出一小步尾巴,但换来
    /// "prompt 里最靠后的那个整页"也有快照 ⇒ 下一轮复用面最大
    /// (83 tok 档:64+19 ⇒ 复用 64;187 tok 档:160+27 ⇒ 复用 160)。
    ///
    /// ⚠️ **不要加"前段已对齐就不切"的守卫**(2026-10-19 踩过):生产 flavor 的
    /// L4 档(`prefill_tail_merge = 1024`)会把"从头对齐的长尾段"合并成**尾端
    /// 不对齐**的一大块 —— 前段尾对齐 ≠ 尾段尾对齐。实测该守卫把 1.7k 档的
    /// 复用面从 1728 压到 512(最后一个快照退到第一块)。判据只看**尾端**:
    /// **末段页对齐的唯一判据**(单源;2026-10-19 审查收口)。
    ///
    /// 对齐的**唯一目的** = 让 prompt 尾部的那个页边界**有 GDN 快照可拍**
    /// (前缀复用面);`prefix_cache = false` ⇒ 复用面结构上恒空 ⇒ 对齐只是
    /// "白切一刀"(多一个 chunk + 多一份宿主税)⇒ 返回 0 = 不对齐。
    ///
    /// ⚠️ 本函数是**四处调用点的公共输入**(`prefill_step_chunk` 的 `page` 实参):
    /// 分块决策(scheduler) / kvmem 准入估算(scheduler) / 池需求估算(running) /
    /// 边界预测(phases) —— 曾经各传各的(一处门了、三处没门 ⇒ 估算与真调度
    /// 打架),现收成这一处。
    pub(crate) fn prefill_align_page(
        align_tail: bool,
        prefix_cache: bool,
        pool_page: usize,
    ) -> usize {
        if align_tail && prefix_cache {
            pool_page
        } else {
            0
        }
    }

    /// 旧行为入口（测试用；生产统一走 [`Self::prefill_step_chunk_msg`]）。
    #[cfg(test)]
    pub(crate) fn prefill_step_chunk(
        fed: usize,
        prompt_len: usize,
        chunk_cfg: usize,
        tail_merge: usize,
        page: usize,
    ) -> Option<usize> {
        Self::prefill_step_chunk_msg(fed, prompt_len, chunk_cfg, tail_merge, page, &[], 0)
    }

    /// 同 [`Self::prefill_step_chunk`],但允许在**消息边界页点**提前切块。
    ///
    /// `msg_snaps` = 页对齐的消息边界快照点（升序；由
    /// `msg_boundary::message_snap_points` 产出）。本函数只做“取第一个远离
    /// `fed` 足够远的点”这一段纯算术；分割符/角色/快照预算都不在这里。
    ///
    /// `min_span` = 距 `fed` 小于本值的边界不单独切（防止 20-token 消息
    /// 把 prefill 打碎）。`msg_snaps` 为空 = 逐字节旧行为。
    pub(crate) fn prefill_step_chunk_msg(
        fed: usize,
        prompt_len: usize,
        chunk_cfg: usize,
        tail_merge: usize,
        page: usize,
        msg_snaps: &[usize],
        min_span: usize,
    ) -> Option<usize> {
        let remaining = prompt_len.checked_sub(fed)?;
        if remaining == 0 {
            return None;
        }
        // L4:尾块合并(与尺寸无关的宿主税)
        let mut step = if remaining <= chunk_cfg.saturating_add(tail_merge) {
            remaining.max(1)
        } else {
            remaining.min(chunk_cfg).max(1)
        };
        // 页对齐(**只看尾端**:最后一段 + 尾端不对齐 + 有可落的整页边界)
        if page > 0 && step == remaining && !(fed + step).is_multiple_of(page) {
            let last_boundary = (fed + step) / page * page;
            if last_boundary > fed {
                step = last_boundary - fed;
            }
        }
        // 消息边界：只在**正常步长内部**提前切，不扩大 step。
        // 取第一个 `>= fed + min_span` 且 `< fed + step` 的点；`== end`
        // 不需要切（该块本来就在边界结束，snap_keep 会认它）。
        if !msg_snaps.is_empty() {
            let lower = fed.saturating_add(min_span);
            let end = fed + step;
            // `b > fed` 是硬条件：上一块恰好在消息边界结束时，本块 fed 就等于
            // 该点；再切一次会得到 step=0（零元素 prefill ⇒ 算子准入失败）。
            if let Some(&b) = msg_snaps
                .iter()
                .find(|&&b| b > fed && b >= lower && b < end)
            {
                step = b - fed;
            }
        }
        Some(step)
    }

    /// **本轮该拍的页边界**(纯函数;`[pool] snap_decode_stride`;§9.1(6)a)。
    ///
    /// 语义 = 本轮提交 `(pos, fed_new]` 跨过的**最大**页边界 `b`,且该边界
    /// 页号是 `stride` 的整数倍(spec 轮提交 ≤ depth+1 ≤ 8 行 < 页 32 ⇒
    /// 一轮最多跨一个边界;`stride` 只是节流,不参与位置)。
    /// `None` = 本轮不拍(未跨边界 / 跨的边界不在步长上 / 关或页宽为 0)。
    ///
    /// **为何取最大**:链身份与快照只增不减 ⇒ 下一个 turn 的复用面只认
    /// "离链头最近的那个在场边界";小的边界留给旧调用点(prefill 段已逐块
    /// 拍,见 `capture_gdn_snap` 的链推进)。
    pub(crate) fn decode_boundary_in(
        pos: usize,
        fed_new: usize,
        page: usize,
        stride: usize,
    ) -> Option<usize> {
        if page == 0 || stride == 0 {
            return None;
        }
        let b = fed_new / page * page; // 最大页边界 ≤ fed_new
        (b > pos && (b / page).is_multiple_of(stride)).then_some(b)
    }

    /// **合批行派生**(B5.1;host 侧,零 await):刷新块链(可能跨页增长)并
    /// 算出本步物理槽。与单行 `Decode` 分支的派生逐式同源(pos+1 行)。
    fn decode_row_of(&mut self, idx: usize) -> Result<RowSpec> {
        let (turn, sid, gdn_slot, pos, token, grant) = {
            let a = &self.turns.active[idx];
            let token = *a.out.last().ok_or_else(|| ModelError::EngineInvariant {
                what: "合批 decode:行 out 为空(相位判定漂移)".into(),
            })?;
            (a.id, a.session_id, a.gdn_slot, a.fed, token, a.grant)
        };
        let grew = {
            let s = self.bank.sessions.session_mut(sid)?;
            let before = s.block_table.len();
            // H2 已由 pump 的 Acquire(Grow) 预留；这里只纯分配
            // (`generation_target` 覆盖 verify+propose)。
            self.bank
                .rel
                .ensure_for_len(grant, &mut s.block_table, pos + 1)?;
            s.block_table.len() != before
        };
        let page = self.dev.pool.geom.page;
        let kv_slot = {
            let s = self.bank.sessions.session_mut(sid)?;
            let b = s.block_table[pos / page];
            b * page as u32 + (pos % page) as u32
        };
        Ok(RowSpec {
            turn,
            session_id: sid,
            token,
            pos,
            kv_slot,
            kv_len: pos + 1,
            gdn_slot,
            grew,
        })
    }

    /// **store 事实**（唯一注入面；准入/夹紧/诊断共用）。
    ///
    /// 页账口径（单源，勿在别处另算）：
    /// - `free_pages` = 物理空闲 ∧ 未被预留承诺占用 = `min(slots_free, total − reserved)`；
    /// - `cache_pages` = **可回收缓存页**（设备面引用=1 的块；
    ///   `LegStats.movable_bytes` 是 store 的 §23.2「准入可动面真源」）−
    ///   已被上面 `free` 吃掉的部分（不许和 `free` 重叠，否则双算）；
    /// - `reserved_pages` = store 的 `ReservationLedger` 准入面（Grant 总量）；
    /// - `in_flight_turns` = store 的 Grant 在册 turn 数（槽位策略）。
    ///
    /// ⇒ `usable = free + cache − reserve ≤ total − reserved − reserve`：
    /// 物理可用、缓存可回收、未来承诺三者同时成立才放人。
    pub(crate) fn store_facts(&self) -> crate::sched_policy::StoreFacts {
        let leg = self.bank.rel.leg.report();
        // **P0-4**：页账唯一真源 = store 的 ReservationLedger（不再读 sched_policy
        // 的影子 admitted 账）。
        let reserved = self.bank.rel.reserved_slots();
        let in_flight_turns = self.bank.rel.in_flight_turns();
        let claimed_free = leg.slots_total.saturating_sub(reserved);
        let free_pages = leg.slots_free.min(claimed_free);
        let movable_pages = if leg.block_bytes > 0 {
            leg.movable_bytes / leg.block_bytes
        } else {
            0
        };
        let cache_pages = movable_pages.min(claimed_free.saturating_sub(free_pages));
        // **磁盘腿**（超卖档的第三层）：`DiskTier` 的块 = 一条链（单卡 = 一页，
        // TP 合并后 = world 页 —— 与宿主槽同粒度；`attach_disk` 按 `host_slot_bytes`
        // 开层）。配额接了就把它算进“能落哪儿”，否则宿主满 = 拒新（配了盘却
        // 不用 = 假容量，超卖档等于白配）。
        let disk_free_pages = self.bank.rel.disk.as_ref().map_or(0, |d| {
            let per = d.block_bytes() / leg.block_bytes.max(1);
            d.free_slots().saturating_mul(per)
        });
        crate::sched_policy::StoreFacts {
            free_pages,
            cache_pages,
            reserved_pages: reserved,
            in_flight_turns,
            // **宿主腿**（超卖档溢出面）：预算 − 已驻留，按块字节折页。
            host_free_pages: if leg.block_bytes > 0 {
                leg.host_budget_bytes.saturating_sub(leg.host_bytes) / leg.block_bytes
            } else {
                0
            },
            host_used_pages: if leg.block_bytes > 0 {
                leg.host_bytes / leg.block_bytes
            } else {
                0
            },
            disk_free_pages,
        }
    }

    /// 准入容量快照（诊断：`/debug/vram → batch.capacity`）
    pub fn capacity_probe(
        &self,
    ) -> (
        crate::sched_policy::StoreFacts,
        crate::sched_policy::Capacity,
    ) {
        let l = self.admission_limits();
        let s = self.store_facts();
        // 用一个"零成本"的代表 turn 报容量：取当前账上最大在跑尺寸，
        // 没有在跑时用 1 页（n_mem 于是等于"还能进几个 1 页的人"，仅供观察）。
        let facts = self.turns.actives().map(Self::facts_of_active).next();
        let rep = facts.unwrap_or(crate::sched_policy::TurnFacts {
            turn: 0,
            session: 0,
            prompt_tokens: 0,
            max_new: 0,
            fed: 0,
            out_len: 0,
            phase: crate::sched_policy::PhaseSz::Gen,
            spec: false,
        });
        let cap = self.turns.admission.capacity(&l, &s, &[], &rep);
        (s, cap)
    }

    /// **调度几何**（boot 静态；`sched_policy::Limits` 的注入面）
    pub(crate) fn admission_limits(&self) -> crate::sched_policy::Limits {
        let page = self.dev.pool.geom.page.max(1);
        crate::sched_policy::Limits {
            page,
            max_batch: self.boot.cfg.knobs.max_batch.max(1),
            gdn_slots: self.boot.cfg.knobs.gdn_slots.max(1),
            // 引擎不知道服务面的请求上限 ⇒ 用格数当 n_target 的上界
            // （`n_target = min(max_batch, gdn_slots, max_concurrency)`）
            max_concurrency: self.boot.cfg.knobs.gdn_slots.max(1),
            oversub: self.boot.cfg.knobs.kvmem_virtualization,
            workspace_pages: self.boot.cfg.knobs.workspace_per_req.div_ceil(page),
            chunk_pages: self.boot.cfg.prefill_chunk.div_ceil(page),
            // 池保留：0 = 自动（按池比例、上限 8 页；小池不被"保留"吃光）
            reserve_pages: if self.boot.cfg.knobs.reserve_pages == 0 {
                (self.dev.pool.geom.nb / 64).min(8)
            } else {
                self.boot.cfg.knobs.reserve_pages
            },
            slice_every: self.boot.cfg.knobs.slice_every,
            slice_tokens: self.boot.cfg.knobs.slice_tokens,
            newcomer_min_tokens: self.boot.cfg.knobs.newcomer_min_tokens,
            min_gen_tokens: self.boot.cfg.knobs.min_gen_tokens,
            max_queue: self.boot.cfg.knobs.max_queue,
        }
    }

    /// 排队 turn → 事实（准入面）
    fn facts_of_queued(t: &crate::turn::Turn) -> crate::sched_policy::TurnFacts {
        crate::sched_policy::TurnFacts {
            turn: t.id,
            session: t.session_id,
            prompt_tokens: t.prompt_ids.len(),
            max_new: t.spec.max_new,
            fed: 0,
            out_len: 0,
            phase: crate::sched_policy::PhaseSz::Prefill,
            spec: false,
        }
    }

    /// 活跃 turn → 事实（准入/相位共用）
    fn facts_of_active(a: &crate::turn::ActiveTurn) -> crate::sched_policy::TurnFacts {
        let fed = a.fed.min(a.prompt_ids.len());
        crate::sched_policy::TurnFacts {
            turn: a.id,
            session: a.session_id,
            prompt_tokens: a.prompt_ids.len(),
            max_new: a.max_new,
            fed,
            out_len: a.out.len(),
            phase: if fed < a.prompt_ids.len() {
                crate::sched_policy::PhaseSz::Prefill
            } else {
                crate::sched_policy::PhaseSz::Gen
            },
            spec: false,
        }
    }

    /// **启动一个排队 turn**（纯 host 账：准入登记/前缀匹配/块链/GDN 行/装配）。
    ///
    /// 从 `schedule()` 抽出（2026-10-20）：准入判决必须**在启动之外**做
    /// `Hold/Reject` —— 判决嵌在启动体里时，Hold 早退会停摆已有活跃、
    /// 不早退则队首照常被弹起（准入形同虚设）。**行为逐字节不变**。
    fn start_one(&mut self, turn: crate::turn::Turn, grant: owl_store::Grant) -> Result<BeginPlan> {
        let grant_id = grant.id;
        // **准入登记**(S3b §23.2):turn 从此占用一块保留额(实占随
        // `sync_pool_account` 在段边界更新;天花板门 count 靠它)。
        self.bank.rel.admission.admit(turn.id, 0, 0)?;
        self.turns.admit_wait = None;
        // **S2b(§28):turn 起借 GDN 行**(账在 store:行 = 活跃 turn 的资源;
        // 物理行号就是账里的行号 —— 一本账,不两份)。行耗尽 = 响亮拒
        // (`Exhausted`;正常路径由准入面 `gdn_slots` 天花板先挡)。
        let gdn_row = self
            .bank
            .rel
            .state
            .borrow_row(owl_store::tier::Owner::Turn(turn.id))?;
        // **P0-4 x P1-5**：行也进同一本 ReservationLedger（turn 生命周期）。
        let row_grant = self.bank.rel.reserve_row(turn.id)?;
        debug_assert_eq!(row_grant.kind, owl_store::ReservationKind::Row);
        // E2c 复用形态:**PrefixHit(内容寻址,块链复用 + 快照恢复)/ Fresh(全量)**
        //
        // **S2(§25.1①)删掉 GuardHit 快路**:idle 会话不再持设备表 ⇒
        // "同会话原样复用块链"这条快路不再成立(表已释放),"同会话增量"
        // 由**内容寻址**接管:tokens 前缀一致 ⇒ 命中;客户端改写历史 ⇒
        // 命中自然变短(自回退全量)—— 守卫(isolation)的活由内容身份干,
        // 少一条与内容寻址并行的"账本信任"路径。
        // **前缀无感删除**(2026-10-23):先试"历史里正好少了一段**已标记**
        // 图像"。命中 ⇒ 后缀页保留 + 位置洞 + 零重算;不命中 ⇒ 逐字节旧路径
        // (G5 兜底,`plan_image_delete` 全程纯读,失败零副作用)。
        // **前缀无感删除 = leave-gap 唯一路径**(2026-10-26):删图后保留
        // 幸存 position、不转 K;不命中则逐字节旧前缀复用路径。
        let (cached_len, restore_key, pending_loads, image_exec, lazy_view) = if turn.deletion.is_some() {
            match self.plan_image_leave_gap(&turn)? {
                Some(p) => {
                    let (c, r, pl, im) = self.apply_image_leave_gap(p, &turn)?;
                    (c, r, pl, im, None)
                }
                None => {
                    let (c, r, pl, lz) = self.plan_prefix_reuse(&turn)?;
                    (c, r, pl, None, lz)
                }
            }
        } else {
            let (c, r, pl, lz) = self.plan_prefix_reuse(&turn)?;
            (c, r, pl, None, lz)
        };
        // E2b:块表确保覆盖 prompt(增量 turn 块链已存,只长新增段)
        //
        // **锁①后半(2026-10-19)**:原实现一次长到**整个 prompt** ——
        // 池 ≥ ctx 时从不暴露;池 < ctx(虚拟化档)时那一步就要几千页槽
        // ⇒ 必 `Exhausted`。虚拟化档只长到**首个 chunk** 的那几页,
        // 后面由 `execute_prefill` 逐 chunk 增量长(用到哪长到哪);
        // 腾位归召回点的驻留调和。
        {
            let s = self
                .bank
                .sessions
                .get_mut(turn.session_id)
                .expect("submit 已建会话账");
            let target = if self.boot.cfg.knobs.kvmem_virtualization {
                Self::prefill_step_chunk_msg(
                    0,
                    turn.prompt_ids.len(),
                    self.boot.cfg.prefill_chunk,
                    self.boot.cfg.knobs.prefill_tail_merge,
                    Self::prefill_align_page(
                        self.boot.cfg.knobs.prefill_align_tail,
                        self.boot.cfg.knobs.prefix_cache,
                        self.dev.pool.geom.page,
                    ),
                    &turn.msg_snaps,
                    self.boot.cfg.knobs.snap_msg_min_span,
                )
                .unwrap_or(1)
            } else {
                turn.prompt_ids.len()
            };
            // **P0-4**：只消费准入 Grant（分配失败会把额度回滚）。
            self.bank
                .rel
                .ensure_for_len(grant_id, &mut s.block_table, target)?;
        }
        let begin = BeginPlan {
            session_id: turn.session_id,
            gdn_slot: gdn_row,
            restore_key,
            reset_gdn: cached_len == 0 && image_exec.is_none(),
            pending_loads,
            image_exec,
            lazy_view,
        };
        let ignore_eos = turn.spec.ignore_eos;
        let opts = turn.opts.clone();
        // 采样面合并(AGT-06):请求覆写按字段并入引擎 knobs;
        // 请求未表达采样意图 ⇒ 完全走 knobs(flavor 语义不变)
        let (eff_sampler, sampler_on) = match opts.sampler {
            Some(o) if !o.is_empty() => {
                let mut s = self.boot.cfg.knobs.sampler;
                if let Some(t) = o.temp {
                    s.temp = t;
                }
                if let Some(k) = o.topk {
                    s.topk = k;
                }
                if let Some(p) = o.topp {
                    s.topp = p;
                }
                let on = s.temp > 0.0;
                (s, on)
            }
            _ => (
                self.boot.cfg.knobs.sampler,
                self.boot.cfg.knobs.sampler_enabled,
            ),
        };
        // **[N1 TurnBegin]**:turn 上设备 —— 归属 `Turn(id)`;类别 `Temp`
        //    (turn 域的一切分配都是临时面:激活/工作区/回退;常驻面在 boot 已归 System)
        let admit_id = turn.id;
        self.set_dev_ctx(
            owl_iface::handle::Owner::Turn(admit_id),
            owl_iface::handle::AssetClass::Temp,
        );
        // 排队时长（提交 → 起跑）：直方图 + 起跑时刻（墙钟 EWMA 用）
        let started_ms = crate::running::now_ms();
        self.turns
            .note_wait(started_ms.saturating_sub(turn.submitted_ms));
        let next_rope_pos = turn
            .positions
            .iter()
            .flat_map(|p| p.iter())
            .copied()
            .max()
            .unwrap_or(-1)
            .saturating_add(1);
        self.turns.active.push(crate::turn::ActiveTurn {
            id: turn.id,
            session_id: turn.session_id,
            gdn_slot: gdn_row,
            grant: grant_id,
            // 回滚键 = 当前 turn 的第一个已知边界;capture 后会被更新。
            last_snap: restore_key,
            started_ms,
            ephemeral: turn.ephemeral,
            prompt_ids: turn.prompt_ids,
            msg_snaps: turn.msg_snaps,
            max_new: turn.spec.max_new,
            positions: turn.positions,
            next_rope_pos,
            rope_gap: turn.positions_gap,
            ignore_eos,
            sampler: eff_sampler,
            sampler_on,
            stop: opts.stop,
            emitted: 0,
            trim_text_to: None,
            seed: opts.seed,
            first_token_ms: None,
            out: Vec::new(),
            fed: cached_len,
            cached_start: cached_len,
            decoded: String::new(),
            // 视觉输入（主分支 V1–V5）：图像随 turn 携带，prefill 首块注入
            images: opts.images.clone(),
            img_spans: turn.img_spans,
            img_pads: turn.img_pads,
            vlm: None,
            vlm_peer: None,
            spec: crate::turn::TurnSpecState {
                // B4 分母**跨 turn 保鲜**(见 RunningEngine::bare_cost_carry_ms);
                // 其余 spec 态仍全零 = 生命周期收口纪律不动
                bare_cost_ewma_ms: self.dev.spec_rt.carry_ms,
                bare_cost_ctx: self.dev.spec_rt.carry_ctx,
                // 分母未测 ⇒ 本 turn 发播种裸步(见 TurnSpecState::bare_seed_left)
                bare_seed_left: if self.dev.spec_rt.carry_ms > 0.0 {
                    0
                } else {
                    crate::turn::SPEC_SEED_SAMPLES
                },
                ..Default::default()
            },
        });
        Ok(begin)
    }

    /// **Fresh 起跑**(零复用):释放旧块表/旧 MTP 链 + 会话账归零。
    fn plan_fresh(
        &mut self,
        turn: &crate::turn::Turn,
    ) -> Result<(
        usize,
        Option<owl_store::SnapKey>,
        Vec<(usize, owl_store::ChainHash)>,
        Option<LazySeed>,
    )> {
        let s = self
            .bank
            .sessions
            .get_mut(turn.session_id)
            .expect("submit 已建会话账");
        let mut table = std::mem::take(&mut s.block_table);
        self.bank.rel.leg.release_table(&mut table)?;
        let mut mold = std::mem::take(&mut s.mtp_block_table);
        self.bank.leg_mtp.release_table(&mut mold)?;
        s.rebase();
        Ok((0, None, Vec::new(), None))
    }

    /// **leave-gap 删除计划**(纯 host):摘掉被删图像页,保留前缀/后缀页,
    /// **不转 K、不移动幸存 token 的 position**。快照只换边界身份(元数据),
    /// 内容仍是同一份 GDN 状态/K。
    fn plan_image_leave_gap(&mut self, turn: &crate::turn::Turn) -> Result<Option<ImageLeaveGap>> {
        if !self.boot.cfg.knobs.image_evict || !self.dev.pool.geom.paged {
            return Ok(None);
        }
        let page = self.dev.pool.geom.page;
        if page == 0 {
            return Ok(None);
        }
        let Some(del) = turn.deletion else {
            return Ok(None);
        };
        if del.a >= del.b || del.a % page != 0 || (del.b - del.a) % page != 0 {
            owl_shared::log_warn!(
                "img-evict",
                "leave-gap 删除计划非法:session={:#x} turn={} a={} b={} page={} ⇒ 不吸收",
                turn.session_id,
                turn.id,
                del.a,
                del.b,
                page
            );
            return Ok(None);
        }
        let (old, old_pos, old_spans) = {
            let s = self.bank.sessions.session(turn.session_id)?;
            (s.tokens.clone(), s.positions.clone(), s.img_spans.clone())
        };
        if old_pos.len() != old.len() {
            owl_shared::log_info!(
                "img-evict",
                "leave-gap 跳过:session={:#x} 旧位置账 {} ≠ token {}（上一版无位置账）",
                turn.session_id,
                old_pos.len(),
                old.len()
            );
            return Ok(None);
        }
        if del.b > old.len()
            || del.a > turn.prompt_ids.len()
            || turn.positions.len() != turn.prompt_ids.len()
        {
            owl_shared::log_warn!(
                "img-evict",
                "leave-gap 删除越界:session={:#x} turn={} del=[{},{}), old={}, raw={}, pos={} ⇒ 不吸收",
                turn.session_id,
                turn.id,
                del.a,
                del.b,
                old.len(),
                turn.prompt_ids.len(),
                turn.positions.len()
            );
            return Ok(None);
        }
        if old[..del.a] != turn.prompt_ids[..del.a] {
            owl_shared::log_warn!(
                "img-evict",
                "leave-gap 删点之前不一致:session={:#x} turn={} a={} ⇒ 不吸收",
                turn.session_id,
                turn.id,
                del.a
            );
            return Ok(None);
        }
        let (hit, _recall) = self.bank.rel.match_prefix(&old)?;
        // leave-gap 会话的链只允许**本会话**复用:token-only 链哈希无法区分
        // 同 token 不同 position 的 K(跨会话命中 = 静默错值)。
        if hit
            .hashes
            .last()
            .is_some_and(|h| self.bank.rel.leg.session_of(*h) != turn.session_id)
        {
            owl_shared::log_info!(
                "img-evict",
                "leave-gap 链归属不符:session={:#x} turn={} tail_owner={:?} ⇒ 不吸收",
                turn.session_id,
                turn.id,
                hit.hashes.last().map(|h| self.bank.rel.leg.session_of(*h))
            );
            return Ok(None);
        }
        let a_blocks = del.a / page;
        let b_blocks = del.b / page;
        if hit.matched_blocks < a_blocks {
            owl_shared::log_info!(
                "img-evict",
                "leave-gap 旧前缀链不足:session={:#x} turn={} matched={} need={} 页 ⇒ 不吸收",
                turn.session_id,
                turn.id,
                hit.matched_blocks,
                a_blocks
            );
            return Ok(None);
        }
        // 后缀 = 删点之后逐 token 相同、且页对齐的段。
        let k_pad = old[del.b..]
            .iter()
            .zip(&turn.prompt_ids[del.a..])
            .take_while(|(x, y)| x == y)
            .count();
        let mut keep = (k_pad / page).min(hit.matched_blocks.saturating_sub(b_blocks));
        if keep == 0 {
            owl_shared::log_info!(
                "img-evict",
                "leave-gap 后缀无可复用整页:session={:#x} turn={} k_pad={k_pad} page={page} matched={} a_blk={a_blocks} b_blk={b_blocks} first_slot={:?} ⇒ 不吸收",
                turn.session_id,
                turn.id,
                hit.matched_blocks,
                hit.slots.get(b_blocks).copied().flatten()
            );
            return Ok(None);
        }
        // 至少一个后缀边界有快照(GDN 态只能从边界续)。
        let mut boundary = None;
        while keep > 0 {
            let bi = b_blocks + keep - 1;
            let k = owl_store::SnapKey::at_boundary(hit.hashes[bi]);
            if self.bank.rel.state.locate(k).is_some() {
                boundary = Some((bi, k));
                break;
            }
            keep -= 1;
        }
        let Some((bi, old_key)) = boundary else {
            owl_shared::log_info!(
                "img-evict",
                "leave-gap 后缀边界无快照:session={:#x} turn={} keep={} ⇒ 不吸收",
                turn.session_id,
                turn.id,
                keep
            );
            return Ok(None);
        };
        let mut cur = owl_store::ChainCursor::new();
        if a_blocks > 0 {
            cur.seed(a_blocks, hit.hashes[a_blocks - 1]);
        }
        let mut boundary_new: Option<owl_store::SnapKey> = None;
        let mut rekeyed = 0usize;
        let mut disk_ops: Vec<SnapDiskOp> = Vec::new();
        for i in 0..keep {
            let lo = (a_blocks + i) * page;
            let h = cur
                .advance(page, &turn.prompt_ids[lo..], a_blocks + i + 1)
                .ok_or_else(|| ModelError::EngineInvariant {
                    what: format!("leave-gap:新链哈希推进失败(块 {})", a_blocks + i),
                })?;
            let k_old = owl_store::SnapKey::at_boundary(hit.hashes[b_blocks + i]);
            let k_new = owl_store::SnapKey::at_boundary(h);
            let tier = self.bank.rel.state.locate(k_old);
            let target = snap_rekey_target(tier, self.dev.pool.snap_slot_of(k_old).is_some());
            let ok = match target {
                RekeyTarget::Device => {
                    if self.bank.rel.rekey_snap(k_old, k_new) {
                        let _ = self.dev.pool.snap_rekey(k_old, k_new);
                        if let Some(p) = self.peer.as_mut() {
                            let _ = p.pool.snap_rekey(k_old, k_new);
                        }
                        true
                    } else {
                        false
                    }
                }
                RekeyTarget::Host => self.bank.rel.rekey_snap(k_old, k_new),
                RekeyTarget::Disk => {
                    disk_ops.push(SnapDiskOp::Rekey {
                        old: k_old,
                        new: k_new,
                    });
                    true
                }
                RekeyTarget::Skip => false,
            };
            if ok {
                rekeyed += 1;
            }
            if b_blocks + i == bi {
                let pending_disk = disk_ops
                    .iter()
                    .any(|o| matches!(o, SnapDiskOp::Rekey { old, new } if *old == k_old && *new == k_new));
                boundary_new = if self.bank.rel.state.locate(k_new).is_some() || pending_disk {
                    Some(k_new)
                } else {
                    None
                };
            }
        }
        // 前缀/后缀槽位:设备页直接持引用;宿主占位页记回灌清单(段边界
        // 由 `execute_begin` 搬到设备后再写块表)。**不能**在这里先 drop 旧链,
        // 否则宿主内容先被 evict,回灌必失败。
        let mut table = Vec::with_capacity(a_blocks + keep);
        let mut pending_loads: Vec<(usize, owl_store::ChainHash)> = Vec::new();
        for i in 0..a_blocks {
            let idx = table.len();
            match hit.slots[i] {
                Some(slot) => {
                    self.bank.rel.leg.incref(slot)?;
                    table.push(slot);
                }
                None => {
                    table.push(PENDING_LOAD_PLACEHOLDER);
                    pending_loads.push((idx, hit.hashes[i]));
                }
            }
        }
        for i in 0..keep {
            let bi = b_blocks + i;
            let idx = table.len();
            match hit.slots[bi] {
                Some(slot) => {
                    self.bank.rel.leg.incref(slot)?;
                    table.push(slot);
                }
                None => {
                    table.push(PENDING_LOAD_PLACEHOLDER);
                    pending_loads.push((idx, hit.hashes[bi]));
                }
            }
        }
        let cached_len = (a_blocks + keep) * page;
        let old_marked = old_spans.iter().filter(|s| s.evict).count();
        // **待释放旧链尾**（不是 free）：新表已持前缀/后缀引用；旧链从
        // 删点起的所有身份（被删图 + 未复用旧尾）在执行面回灌完成后摘掉
        // ⇒ 缓存不再站住这些槽。槽何时真正回收由 store/调度器决定。
        let release_chains: Vec<owl_store::ChainHash> =
            hit.hashes[a_blocks..hit.matched_blocks].to_vec();
        owl_shared::log_info!(
            "img-evict",
            "leave-gap:session={:#x} turn={} 删 [{}, {})={} tok/{} 页 · 保留前缀 {} + 后缀 {} 页 / {} tok · 不改 K · 待释放旧链 {} 块 · 宿主回灌 {} 页 · 快照重挂 {} 条(盘待搬 {}) · marked={}",
            turn.session_id,
            turn.id,
            del.a,
            del.b,
            del.b - del.a,
            (del.b - del.a) / page,
            a_blocks,
            keep,
            cached_len,
            release_chains.len(),
            pending_loads.len(),
            rekeyed,
            disk_ops.len(),
            old_marked
        );
        Ok(Some(ImageLeaveGap {
            table,
            a: del.a,
            b: del.b,
            rekeyed,
            cached_len,
            tail_hash: cur.tail(),
            restore_key: boundary_new.or(Some(old_key)),
            snap_disk: disk_ops,
            pending_loads,
            release_chains,
        }))
    }

    /// leave-gap 删除计划的落账(表/游标/token/位置账);设备半只做盘快照收尾。
    fn apply_image_leave_gap(
        &mut self,
        d: ImageLeaveGap,
        turn: &crate::turn::Turn,
    ) -> Result<(
        usize,
        Option<owl_store::SnapKey>,
        Vec<(usize, owl_store::ChainHash)>,
        Option<ImageExec>,
    )> {
        let page = self.dev.pool.geom.page;
        let s = self.bank.sessions.session_mut(turn.session_id)?;
        let mut mold = std::mem::take(&mut s.mtp_block_table);
        self.bank.leg_mtp.release_table(&mut mold)?;
        s.block_table = d.table;
        s.cached_len = d.cached_len;
        s.tokens.clear();
        s.tokens.extend_from_slice(&turn.prompt_ids[..d.cached_len]);
        // **中途落账也必须自洽**：Abort 不会再 commit_turn；pad/spans 若仍
        // 指向被截掉的后缀，下一轮 unpad 会 400。
        let (spans, pads) =
            crate::img_evict::ledger_prefix(&turn.img_spans, &turn.img_pads, d.cached_len);
        s.set_img_spans(spans, pads);
        s.chain.seed(d.cached_len / page, d.tail_hash);
        s.set_positions(&turn.positions[..d.cached_len], true)?;
        s.position_epoch = s.position_epoch.wrapping_add(1);
        self.img_evict.absorbed += 1;
        self.img_evict.last = Some(crate::running::ImgEvictTrace {
            a: d.a,
            b: d.b,
            keep_blocks: d.cached_len / page - d.a / page,
            rekeyed: d.rekeyed,
        });
        owl_shared::counter_add!("image_evict.blocks_released", d.release_chains.len() as u64);
        Ok((
            d.cached_len,
            d.restore_key,
            d.pending_loads,
            Some(ImageExec {
                snap_disk: d.snap_disk,
                release_chains: d.release_chains,
            }),
        ))
    }

    /// **前缀复用面决策**(自 `start_one` 抽出的纯 host 段;2026-10-23)。
    ///
    /// 返回 `(cached_len, restore_key, pending_loads)`。行为逐字节不变 ——
    /// 抽出来只是为了让「前缀无感删除」能在它**之前**短路(删除路径自己取链)。
    fn plan_prefix_reuse(
        &mut self,
        turn: &crate::turn::Turn,
    ) -> Result<(
        usize,
        Option<owl_store::SnapKey>,
        Vec<(usize, owl_store::ChainHash)>,
        Option<LazySeed>,
    )> {
        let s = self
            .bank
            .sessions
            .get_mut(turn.session_id)
            .expect("submit 已建会话账");
        let positioned = s.positioned;
        if self.dev.pool.geom.paged {
            // 前缀缓存匹配(先匹配后释放旧链;内容寻址 = 跨会话)。
            // **S3b 收尾**:经 `Rel` 门面**一次问两腿**(块链 + 快照层)
            // —— 少一条"引擎自己分别查两处"的旁路(§10.4 律⑤)。
            let (hit, recall) = self.bank.rel.match_prefix(&turn.prompt_ids)?;
            if positioned
                && hit
                    .hashes
                    .last()
                    .is_some_and(|h| self.bank.rel.leg.session_of(*h) != turn.session_id)
            {
                owl_shared::log_info!(
                    "img-evict",
                    "leave-gap 会话拒绝跨会话复用:session={:#x} turn={} tail_owner={:?} ⇒ Fresh",
                    turn.session_id,
                    turn.id,
                    hit.hashes.last().map(|h| self.bank.rel.leg.session_of(*h))
                );
                return self.plan_fresh(turn);
            }
            // S3-3b:宿主承载格(内容在 host)不再整条拒 —— 占位 + 记进
            // `pending_loads`,由 `execute_begin` 在段边界回灌(真槽写回,
            // 必须在 `write_bt` 之前换掉)。
            let (matched_raw, hashes) = (hit.matched_blocks, hit.hashes.clone());
            let mut chain: Vec<u32> = Vec::with_capacity(hit.slots.len());
            let mut pending_loads: Vec<(usize, owl_store::ChainHash)> = Vec::new();
            for (i, (slot, h)) in hit.slots.iter().zip(&hashes).enumerate() {
                match slot {
                    Some(s) => chain.push(*s),
                    None => {
                        chain.push(PENDING_LOAD_PLACEHOLDER);
                        pending_loads.push((i, *h));
                    }
                }
            }
            if self.boot.cfg.knobs.probes.debug {
                // 在场性 = **store 的层账**(设备/宿主都算);层号一并打出。
                // 头尾都要看(2026-10-19):复用面只认**最靠近链头**的在场面 ⇒
                // "前 4 全 '-'"曾把"链头那张缺位"误读成"全丢";故头 4 + 尾 6。
                let tier_of = |h: &owl_store::ChainHash| match self
                    .bank
                    .rel
                    .state
                    .locate(owl_store::SnapKey::at_boundary(*h))
                {
                    Some(owl_store::Tier::Gpu) => "dev",
                    Some(owl_store::Tier::Host) => "host",
                    Some(owl_store::Tier::Disk) => "disk",
                    None => "-",
                };
                let head: Vec<&str> = hashes.iter().take(4).map(tier_of).collect();
                let tail: Vec<&str> = hashes.iter().rev().take(6).rev().map(tier_of).collect();
                owl_shared::log_debug!(
                    "dbg prefix",
                    "match = {matched_raw} blocks / chain = {chain:?} · snap_used={}/{} 边界在场 头4={head:?} 尾6={tail:?}",
                    self.dev.pool.snap_used(),
                    self.dev.pool.snap_capacity()
                );
            }
            // 裁剪(末块对齐 + 快照在场)= **单源** `reuse_face`
            // (需求驱动保洁 §12.4 用同一份算式,不许各算一份)
            let (m_raw, _dev) = Self::reuse_face(
                &self.bank.rel,
                &hit,
                &recall,
                turn.prompt_ids.len(),
                self.dev.pool.geom.page,
            );
            // ── **超卖档的命中面上限**(2026-10-27 立案并修) ──
            //
            // 命中的**宿主块**要在 `execute_begin` 里**全量回灌**进设备(占位符
            // 换成真槽,`write_bt` 之前必须换掉)⇒ 命中面一旦超过设备池容量,
            // 回灌阶段必 `Exhausted`。
            //
            // 实测:池 1492 槽 / 48k prompt(=1500 块) 的**第二次**请求
            // (前缀缓存全命中)必 `500 / 槽池耗尽(共 1492 槽,无自由槽)`;
            // 第一次能跑是因为新算的块可以边算边沉,而回灌的块**全在视图
            // 工作集里**、不可再沉。⇒ 命中面必须预留出"本 turn 还要新算的
            // chunk + 工作集"的槽,尾部重算。
            //
            // 非超卖档(`pool ≥ ctx`)cap 恒大于链长 ⇒ 本段自动不生效。
            let page = self.dev.pool.geom.page.max(1);
            let cap = self.dev.pool.geom.nb.max(1);
            // **余量单源**（`sched_policy::reclaim_margin`）—— 与懒沉同一份算式。
            // 旧式额外把 `workspace_per_req` 也扣进来，是 P1-2 的一半：命中面
            // 被白砍（实测 `ws=32768` 档 22.9k prompt：716 → 256 块，6.5k token
            // 白重算）。工作集是命中面的子集，不该再扣第二遍。
            let reserve = crate::sched_policy::reclaim_margin(
                self.boot.cfg.prefill_chunk.div_ceil(page),
            );
            let max_hit = cap.saturating_sub(reserve).max(1);
            // **懒回灌（W3, 2026-10-28）**：虚拟化档（`select_decode && select_partial`）
            // 下命中面**不再按池容量裁剪**。
            //
            // 旧口径（保留给非虚拟化档）：命中块必须在 `execute_begin` **全量**回灌，
            // 回灌块全在视图工作集里、不可再沉 ⇒ 命中面一旦超池必 `Exhausted`
            // ⇒ 只能裁到 `池 − 余量`，尾部整段重算（111k 会话每轮重算 38.6k tok，
            // TTFT 44 s；参照实现在同形状下 1.2 s —— 差距全在这里）。
            //
            // 新口径：回灌只给 **sink + recent 窗**（保底续写连续性），其余命中块
            // 在会话链上留 `PENDING_LOAD_PLACEHOLDER`，由**视图**选择 + `reconcile`
            // 按需 stage-in（稳态 `select_partial` 本来就有洞，沿同一条路）。
            let lazy_stage = self.boot.cfg.knobs.kvmem_virtualization && self.boot.cfg.knobs.select_partial;
            let m = if !lazy_stage && m_raw > max_hit {
                // 裁完还要**向下对齐到有快照的边界** —— `cached_len` 与 GDN 态
                // 必须成对:`restore_snap` 拿不到快照会响亮失败,而不是静默错值。
                // (锚点节奏 = 每 `SNAP_EVERY` 个 chunk + 末边界 + msg 边界。)
                let mut m = max_hit;
                while m > 0
                    && self
                        .bank
                        .rel
                        .state
                        .locate(owl_store::SnapKey::at_boundary(hashes[m - 1]))
                        .is_none()
                {
                    m -= 1;
                }
                owl_shared::log_warn!(
                    "kvmem",
                    "前缀命中面 {m_raw} 块 > 池可容 {max_hit}(池 {cap} 槽 − 新写/工作集 {reserve}) \
                     ⇒ 裁到 {m} 块(对齐全场快照),尾部重算(否则回灌阶段必槽池耗尽)"
                );
                owl_shared::counter_inc!("kvmem.prefix.hit_capped");
                m
            } else {
                m_raw
            };
            if m == 0 {
                // 裁到零 = 没有可用复用面 ⇒ 走全新链(语义与零命中一致)
                return self.plan_fresh(turn);
            }
            if self.boot.cfg.knobs.probes.debug {
                // 裁剪结果 = 复用面真值(块);页对齐后 = 可复用 token 数
                let found = if m > 0 {
                    match self
                        .bank
                        .rel
                        .state
                        .locate(owl_store::SnapKey::at_boundary(hashes[m - 1]))
                    {
                        Some(owl_store::Tier::Gpu) => "dev",
                        Some(owl_store::Tier::Host) => "host",
                        _ => "?",
                    }
                } else {
                    "(无)"
                };
                owl_shared::log_debug!(
                    "dbg prefix",
                    "复用面 m={m}/{matched_raw} 块(= {} tok)/ 落在边界层={found}",
                    m * self.dev.pool.geom.page
                );
            }
            if positioned {
                owl_shared::log_info!(
                    "img-evict",
                    "leave-gap 复用:session={:#x} turn={} prompt={} matched={} m={} tail_snap={:?}",
                    turn.session_id,
                    turn.id,
                    turn.prompt_ids.len(),
                    matched_raw,
                    m,
                    recall.snap_tier
                );
            }
            if m > 0 {
                // 只对**已驻留**块加会话引用(宿主格的引用由回灌侧
                // `alloc_slot` + `rebind` 建立;占位值不参与账)
                for &b in &chain[..m] {
                    if b != PENDING_LOAD_PLACEHOLDER {
                        self.bank.rel.leg.incref(b)?;
                    }
                }
                // ── **待回灌清单 = 起点视图要读的那一小批**（W3）──
                //
                // 视图口径（与 `past_view_of`/`view_pages_at` 逐字同源）：
                //   页链 = sink 块 ++ 链[recall_pos..=pos]（tail）
                //   len  = sink_tokens + (pos+1 − recall_pos)
                // ⇒ 设备必须驻留的只有：**sink 块** + **recent 窗**（tail 的前半）。
                // 命中面里其余块全留占位符，由首次边界召回后的 `reconcile` 按选择面
                // stage-in（既有路径，不需要新机制）。
                let lazy = if lazy_stage && m * page > 3 * page {
                    let sink_blocks = self.boot.cfg.knobs.select_sink.min(m);
                    // recent 窗：sink/recent 旋钮的块口径，再抬到**参照实现同量级**
                    // （1024 tok）—— 太小会让"续写连续性"的尾部过低。
                    let recent_tokens = (self.boot.cfg.knobs.select_recent * page)
                        .max(1024)
                        .min(m * page);
                    let recent_lo = (m * page).saturating_sub(recent_tokens) / page;
                    let mut view_blocks: Vec<owl_store::ChainHash> =
                        Vec::with_capacity(sink_blocks + m - recent_lo);
                    view_blocks.extend_from_slice(&hashes[..sink_blocks]);
                    view_blocks.extend_from_slice(&hashes[recent_lo..m]);
                    Some(LazySeed {
                        sink_blocks,
                        recent_tokens,
                        cached_len: m * page,
                        blocks: view_blocks,
                    })
                } else {
                    None
                };
                match lazy.clone() {
                    Some(seed) => {
                        // 只保留 sink + recent 窗的待回灌项；其余本轮不进设备。
                        let recent_lo = seed
                            .cached_len
                            .saturating_sub(seed.recent_tokens)
                            / page;
                        let keep_idx = |i: usize| i < seed.sink_blocks || i >= recent_lo;
                        let before = pending_loads.len();
                        pending_loads.retain(|(i, _)| keep_idx(*i));
                        let holes = m - pending_loads.len()
                            - chain[..m]
                                .iter()
                                .filter(|s| **s != PENDING_LOAD_PLACEHOLDER)
                                .count();
                        owl_shared::log_info!(
                            "kvmem",
                            "懒回灌:复用面 {m} 块({} tok)= sink {} + recent {} tok \
                             ⇒ 回灌 {} 块(旧口径 {before})· 留洞 {holes} 块 \
                             (由视图选择 + reconcile 按需 stage-in)",
                            seed.cached_len,
                            seed.sink_blocks,
                            seed.recent_tokens,
                            pending_loads.len()
                        );
                        owl_shared::counter_inc!("kvmem.lazy_prefix.used");
                        owl_shared::metrics::with_metrics_store(|m| {
                            m.counter_add(
                                "kvmem.lazy_prefix.holes",
                                holes as u64,
                                file!(),
                                line!(),
                            )
                        });
                    }
                    None => {
                        pending_loads.truncate(
                            pending_loads.iter().take_while(|(i, _)| *i < m).count(),
                        );
                    }
                }
                let mut old = std::mem::take(&mut s.block_table);
                self.bank.rel.leg.release_table(&mut old)?;
                s.block_table = chain[..m].to_vec();
                // MTP 链同步重置(E5-M2b):保留前缀的 mtp 条目虽
                // 语义可复用(同 token 同 hidden 确定性),但 M2b
                // 无 prefill extend,重建面 = 0 起步(草稿质量降,
                // 恒等不受影响 —— 草稿只影响速度)
                let mut mold = std::mem::take(&mut s.mtp_block_table);
                self.bank.leg_mtp.release_table(&mut mold)?;
                s.reset();
                // 链哈希游标播种:身份从缓存哈希链接管(此后 prefill/
                // decode 的边界快照键从游标出,不再碰物理块号)
                s.chain.seed(m, hashes[m - 1]);
                s.cached_len = m * self.dev.pool.geom.page;
                Ok((
                    m * self.dev.pool.geom.page,
                    Some(owl_store::SnapKey::at_boundary(hashes[m - 1])),
                    pending_loads,
                    lazy,
                ))
            } else {
                self.plan_fresh(turn)
            }
        } else {
            self.plan_fresh(turn)
        }
    }

    /// 调度决策(主机侧)：turn 启动判定(会话守卫/前缀匹配/快照
    /// 边界)+ 相位决策(prefill chunk / decode 步,host 派生量备齐)。
    /// 只动账(队列/会话账/块账房/活跃 turn)。
    ///
    /// **P0-4**：准入 Admit 后、启动之前经 `Acquire::Capacity { reserve }`
    /// 向 store 领一张 Grant；`start_one`/后续 `ensure_for_len` 只消费这张票。
    /// 因此本函数是 `async`（只多一次“零 I/O”的 grant 发放；搬位仍在 pump）。
    pub(crate) async fn schedule(&mut self) -> Result<SchedulerOutput> {
        let mut begin = None;
        // B5.1:多活跃 —— 有空位才允许起新 turn(其余逻辑不变)。
        // **准入排队中**(S3b §23.4 `Queue` 姿态):留队首不弹 —— 起 turn
        // 的权利在准入判决手里;弹了就会绕过判决直接撞 `ensure_for_len`。
        // ⚠ 段首既有“排队中 = Idle”在**多活跃**下必须收窄:已有活跃 turn
        // 时不许因队首被准入卡住而停泵(否则活跃会话被饿死)。
        let admit_blocked = self
            .turns
            .admit_wait
            .map(|(id, _)| self.turns.queue.front().map(|t| t.id) == Some(id))
            .unwrap_or(false);
        if self.turns.active.len() < self.boot.cfg.knobs.max_batch.max(1) && !admit_blocked {
            // **准入（唯一放行口）**：纯状态机 + store 事实注入。
            // · Admit  → 领 grant → 启动（`start_one`）并入账；
            // · Hold   → 队首**留原位**，本拍只推已有活跃（不许停摆）；
            // · Reject → 出队 + `Failed{admission:<code>}`（服务面翻 429）。
            let mut start_ok = true;
            let mut admission_grant: Option<owl_store::Grant> = None;
            if let Some(head_facts) = self.turns.queue.front().map(Self::facts_of_queued) {
                let l = self.admission_limits();
                let s = self.store_facts();
                let running: Vec<crate::sched_policy::TurnFacts> =
                    self.turns.actives().map(Self::facts_of_active).collect();
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);
                owl_shared::log_debug!(
                    "admit-dbg",
                    "head turn {} prompt={} max_new={} | free={} reserved={} usable={} cap={} run={} in_flight={}",
                    head_facts.turn,
                    head_facts.prompt_tokens,
                    head_facts.max_new,
                    s.free_pages,
                    s.reserved_pages,
                    s.usable_pages(l.reserve_pages),
                    self.turns.admission.cap_for(&l, &s, &head_facts),
                    running.len(),
                    s.in_flight_turns
                );
                match self
                    .turns
                    .admission
                    .decide(&l, &s, &running, &head_facts, now_ms)
                {
                    crate::sched_policy::Verdict::Admit => {
                        owl_shared::metrics::with_metrics_store(|m| {
                            m.counter_add("admit.decision.admit", 1, file!(), line!())
                        });
                        // **Grant 必须在判决之后发**：否则它会进入本次判决的
                        // `StoreFacts.reserved_pages`，变成“自己占自己的位”。
                        let cap = self.turns.admission.cap_for(&l, &s, &head_facts);
                        let reserved_slots = head_facts.need_pages(&l, cap).max(1);
                        // **分配面**必须覆盖整条块表（超卖档的准入面只是工作集，
                        // 后续 prefill 仍会按 chunk 长表）；非超卖档两者相等。
                        // 上界单源 = `RunningEngine::chain_alloc_slots`（
                        // 含 spec 轮 verify/propose 的滑越，见其判据）。
                        let alloc_slots = self
                            .chain_alloc_slots(head_facts.prompt_tokens, head_facts.max_new)
                            .max(reserved_slots);
                        let out = self
                            .store_hook(crate::store_hook::StoreHook::capacity_reserve(
                                owl_store::Need::kv(0, owl_store::NeedWhy::Admit),
                                head_facts.turn,
                                reserved_slots,
                                alloc_slots,
                            ))
                            .await?;
                        admission_grant = out.grant;
                        if admission_grant.is_none() {
                            return Err(ModelError::EngineInvariant {
                                what: "准入 Admit 后 Capacity 未返回 Grant".into(),
                            });
                        }
                    }
                    crate::sched_policy::Verdict::Hold => {
                        start_ok = false;
                        owl_shared::metrics::with_metrics_store(|m| {
                            m.counter_add("admit.decision.hold", 1, file!(), line!())
                        });
                    }
                    crate::sched_policy::Verdict::Reject { reason, .. } => {
                        owl_shared::metrics::with_metrics_store(|m| {
                            m.counter_add(
                                &format!("admit.decision.reject.{}", reason.code()),
                                1,
                                file!(),
                                line!(),
                            )
                        });
                        start_ok = false;
                        let t = self.turns.queue.pop_front().ok_or_else(|| {
                            ModelError::EngineInvariant {
                                what: "admission: 刚判过的队首消失".into(),
                            }
                        })?;
                        self.turns
                            .admission
                            .note_rejected(head_facts.session, now_ms, reason);
                        self.turns.note_reject(head_facts.session, reason.code());
                        // 按**真实队列 + EWMA** 估退避（P3-c）
                        let eta = self.turns.eta_ms(l.max_batch, self.turns.queue.len());
                        owl_shared::log_warn!(
                            "admit",
                            "准入拒绝 turn {}（会话 {:#x}）：{}",
                            t.id,
                            head_facts.session,
                            reason.code()
                        );
                        self.turns
                            .pending_events
                            .push_back(crate::turn::TurnEvent::Failed {
                                turn: t.id,
                                err: format!("admission:{} retry_ms={eta}", reason.code()),
                            });
                    }
                }
            }
            if start_ok {
                if let Some(turn) = self.turns.queue.pop_front() {
                    let facts = Self::facts_of_queued(&turn);
                    let grant =
                        admission_grant
                            .take()
                            .ok_or_else(|| ModelError::EngineInvariant {
                                what: "准入放行但无 Grant（store 账未发）".into(),
                            })?;
                    let turn_id = turn.id;
                    begin = Some(match self.start_one(turn, grant) {
                        Ok(b) => b,
                        Err(e) => {
                            // 启动失败 ⇒ 票必须归还，不能让一次失败吞掉容量。
                            self.bank.rel.release_turn_grants(turn_id);
                            return Err(e);
                        }
                    });
                    self.turns.admission.note_started(&facts);
                } else if let Some(g) = admission_grant.take() {
                    // 队首在判决后消失（理论竞态）⇒ 票必须归还。
                    self.bank.rel.release_turn_grants(g.turn);
                }
            }
        }
        // 队列空 + 无活跃 ⇒ Idle(有活跃则继续本步相位决策)
        if self.turns.active.is_empty() {
            return Ok(SchedulerOutput::Idle);
        }

        // ── B5.1 批 decode 门 ────────────────────────────────────────────
        // 判据(全部成立才拼批;任一不成立走单 turn 旧路,零行为变化):
        // ① 至少 2 个活跃且都在生成相位(无 prefill 欠账);
        // ② **无投机**(depth==0):spec 轮的 GDN 是顺序块 + 快照,批化是 M2;
        // ③ 单 rank(TP 的 AR 批语义未审,R8);④ 选择层视图关闭(页表需 per-row)。
        if self.boot.cfg.knobs.max_batch > 1
            && self.turns.active.len() >= 2
            && self
                .turns
                .active
                .iter()
                .all(|a| a.fed >= a.prompt_ids.len())
            && self.dev.spec_rt.depth == 0
            && self.peer.is_none()
            && !self.sel.has_view()
        {
            let n = self.turns.active.len().min(self.boot.cfg.knobs.max_batch);
            let mut rows = Vec::with_capacity(n);
            for i in 0..n {
                rows.push(self.decode_row_of(i)?);
            }
            self.turns.select(None);
            return Ok(SchedulerOutput::Step {
                begin,
                action: StepAction::BatchDecode { rows },
            });
        }

        // 相位决策:prompt 相位(块式 prefill;W1/PF1)优先。
        // 多活跃时选**欠账最多的 prefill turn**(先把 prompt 灌完),否则
        // round-robin 推进生成相(单 turn 档 = 唯一活跃,行为不变)。
        // **相位决策**（`[runtime] slice_every > 0` 时走纯状态机；=0 走旧逻辑逐字节不变）
        let pick_idx = if self.boot.cfg.knobs.slice_every > 0 {
            // 事实注入（引擎只提供事实，策略全在 `sched_policy::PhaseSched`）
            let facts: Vec<crate::sched_policy::TurnFacts> = self
                .turns
                .active
                .iter()
                .map(|a| {
                    let fed = a.fed.min(a.prompt_ids.len());
                    let phase = if fed < a.prompt_ids.len() {
                        crate::sched_policy::PhaseSz::Prefill
                    } else {
                        crate::sched_policy::PhaseSz::Gen
                    };
                    crate::sched_policy::TurnFacts {
                        turn: a.id,
                        session: a.session_id,
                        prompt_tokens: a.prompt_ids.len(),
                        max_new: a.max_new,
                        fed,
                        out_len: a.out.len(),
                        phase,
                        spec: self.dev.spec_rt.depth > 0,
                    }
                })
                .collect();
            let l = crate::sched_policy::Limits {
                slice_every: self.boot.cfg.knobs.slice_every,
                slice_tokens: self.boot.cfg.knobs.slice_tokens,
                ..Default::default()
            };
            // `batch_ok = false`：批 decode 已在上面的批门处理过，这里只决定单拍相位。
            match self.turns.phase.next(&l, &facts, false) {
                Some(plan) => self
                    .turns
                    .active
                    .iter()
                    .position(|a| a.id == plan.turn)
                    .ok_or_else(|| ModelError::EngineInvariant {
                        what: format!("phase 计划指向不存在的 turn {}", plan.turn),
                    })?,
                None => 0,
            }
        } else if let Some(i) = self
            .turns
            .active
            .iter()
            .enumerate()
            .filter(|(_, a)| a.fed < a.prompt_ids.len())
            .min_by_key(|(_, a)| a.fed)
            .map(|(i, _)| i)
        {
            i
        } else {
            let n = self.turns.active.len();
            let i = self.turns.rr % n;
            self.turns.rr = self.turns.rr.wrapping_add(1);
            i
        };
        let pick_id = self.turns.active[pick_idx].id;
        self.turns.select(Some(pick_id));

        let act = self.turns.active()?;
        if act.fed < act.prompt_ids.len() {
            let base = act.fed;
            let remaining = act.prompt_ids.len() - base;
            // 块长:配置块长 ∩ 剩余(E1 paged 核后无 256 窗钳制;
            // 全局注意力语义,长 ctx 安全)
            let chunk = Self::prefill_step_chunk_msg(
                base,
                act.prompt_ids.len(),
                self.boot.cfg.prefill_chunk,
                self.boot.cfg.knobs.prefill_tail_merge,
                Self::prefill_align_page(
                    self.boot.cfg.knobs.prefill_align_tail,
                    self.boot.cfg.knobs.prefix_cache,
                    self.dev.pool.geom.page,
                ),
                &act.msg_snaps,
                self.boot.cfg.knobs.snap_msg_min_span,
            )
            .unwrap_or_else(|| remaining.max(1));
            // 时间片档：**只在"有人正在 decode"时**才把 prefill 切小 ——
            // 独自一人时切片只会白拉长自己的 TTFT（实测 106→638ms），
            // 且对谁都没好处（没有在位者需要保护）。
            let interleaving = self
                .turns
                .active
                .iter()
                .any(|a| a.fed >= a.prompt_ids.len());
            let chunk = if self.boot.cfg.knobs.slice_every > 0
                && self.boot.cfg.knobs.slice_tokens > 0
                && interleaving
            {
                chunk.min(self.boot.cfg.knobs.slice_tokens)
            } else {
                chunk
            };
            let action = StepAction::Prefill {
                chunk_ids: act.prompt_ids[base..base + chunk].to_vec(),
                base,
                is_last: base + chunk >= act.prompt_ids.len(),
            };
            return Ok(SchedulerOutput::Step { begin, action });
        }

        // 生成相位:decode 单步 / 投机轮(host 派生量备齐 —— 物理槽/跨页增长)
        let (sid, token, pos) = {
            let act = self.turns.active()?;
            (
                act.session_id,
                *act.out.last().ok_or_else(|| ModelError::EngineInvariant {
                    what: "生成相位但 out 为空(相位判定漂移)".into(),
                })?,
                act.fed,
            )
        };
        // B4 自适应降级(§6.24):降级态发裸 Decode(草稿池逐 token 同步);
        // 每 probe_every 步发一轮真 SpecRound 探测(m≥1 → 解除降级,
        // 见 execute_spec_round 尾态机)。非降级 = 原调度,零变化。
        // B4 三态在 ActiveTurn.spec(turn 生命周期;2026-10-10 收口)
        let active_depth =
            if self.dev.spec_rt.adaptive_depth && self.turns.active()?.spec.adaptive_downshift {
                self.dev.spec_rt.depth_min
            } else {
                self.dev.spec_rt.depth_for_pos(pos)
            };
        // **target live 宽度 = P+1**（NInfer 同律）：proposal 仍算满 W=K+1，
        // target 图也仍是 T=K+1（owl 图定形），但 P< K 时只提交前 P+1 行。
        // rem=1 时直接裸 decode；P 的截断在 `execute_spec_round` 内做
        // （那里有 `max_new - out.len()` 与 fold/发射账）。
        let remaining_out = {
            let act = self.turns.active()?;
            act.max_new.saturating_sub(act.out.len())
        };
        // leave-gap 的 target verify/replay 已用 `next_rope_pos` 的 rope_delta；
        // 草稿仍走既有 token 位置（只影响 AL，不影响 greedy 恒等）。
        let spec = if self.dev.spec_rt.depth == 0 || remaining_out <= 1 {
            0
        } else if !self.turns.active()?.spec.spec_degraded {
            // **分母与 ctx 配对**(2026-10-17):裸步成本**并非** ctx 不变 ——
            // 实测 24.1ms@pos564 → 41.7ms@32k(**+73%**);那条"±5%"的实测只
            // 覆盖到 9k,2026-10-14 据此取消“ctx 换档作废分母”= 长 ctx 上分母
            // 偏小 ⇒ 盈亏线偏高 ⇒ **把盈利档误降级**(32k:线 3.22 vs 真线 2.06,
            // AL 2.25 ⇒ 实测 decode 24.5 t/s,而修后 38.0 t/s)。
            // ⇒ **上行超 25% 即作废并重新播种**(播种 = 唯一实测重采路径);
            // **下行不作废**:真实比在各档实测都是 1.86–2.06,而 [`BREAK_EVEN_MIN`]
            // = 2.0 的下限把它卡住 ⇒ 陈旧的大分母对判据无影响(少发无谓播种步)。
            {
                let a = &mut self.turns.active_mut()?.spec;
                if a.bare_cost_ctx > 0 && pos as f32 > a.bare_cost_ctx as f32 * 1.25 {
                    a.bare_cost_ewma_ms = 0.0;
                    a.bare_seed_left = crate::turn::SPEC_SEED_SAMPLES;
                }
            }
            // **分母播种**:盈亏线的分母(裸步成本)只在降级期采样 ⇒ 新进程
            // 首个窗口没分母(旧:兑底常量 2.5 ⇒ 长 ctx 误降级)。分母未测
            // 时先跑 `SPEC_SEED_SAMPLES` 步裸 decode(合法产出)把它量出来。
            let seed_due = {
                let a = &self.turns.active()?.spec;
                a.bare_cost_ewma_ms <= 0.0 && a.bare_seed_left > 0
            };
            if seed_due {
                let a = &mut self.turns.active_mut()?.spec;
                a.bare_seed_left -= 1;
                a.bare_seed_step = true;
                owl_shared::log_info!(
                    "spec-seed",
                    "分母未测 ⇒ 本步发播种裸 decode(还欠 {} 步)",
                    a.bare_seed_left
                );
                0
            } else {
                active_depth
            }
        } else if self.turns.active()?.spec.spec_steps_degraded
            >= self.turns.active()?.spec.spec_probe_every
        {
            self.turns.active_mut()?.spec.spec_steps_degraded = 0;
            active_depth
        } else {
            0
        };
        let gdn_row = self
            .turns
            .active
            .first()
            .map(|a| a.gdn_slot)
            .ok_or_else(|| {
                owl_iface::contract::ModelError::Msg("decode 相位:无活跃 turn(行号真源缺失)".into())
            })?;
        let grant = self.turns.active()?.grant;
        let (kv_slot, kv_slots, gdn_slot, grew) = {
            let s = self.bank.sessions.session_mut(sid)?;
            let need = pos + spec + 1;
            let before = s.block_table.len();
            // **P0-4**：消费本 turn 的准入 Grant。
            self.bank
                .rel
                .ensure_for_len(grant, &mut s.block_table, need)?;
            let grew = s.block_table.len() != before;
            let page = self.dev.pool.geom.page;
            let slot_at = |p: usize| {
                let b = s.block_table[p / page];
                b * page as u32 + (p % page) as u32
            };
            if spec > 0 {
                // 槽表 = verify 块位(pos..pos+spec);余槽补 0(消费面
                // 自算物理槽,此表仅 decode 回退臂消费 depth+1 个)
                let mut kv_slots = [0u32; owl_shared::config::SPEC_VERIFY_ROWS_MAX + 1];
                for (i, sl) in kv_slots.iter_mut().enumerate() {
                    if i <= spec {
                        *sl = slot_at(pos + i);
                    }
                }
                (slot_at(pos), Some(kv_slots), gdn_row, grew)
            } else {
                (slot_at(pos), None, gdn_row, grew)
            }
        };
        match kv_slots {
            Some(kv_slots) => Ok(SchedulerOutput::Step {
                begin,
                action: StepAction::SpecRound {
                    token,
                    pos,
                    kv_slots,
                    gdn_slot,
                    grew,
                    depth: spec,
                },
            }),
            None => Ok(SchedulerOutput::Step {
                begin,
                action: StepAction::Decode {
                    token,
                    pos,
                    kv_slot,
                    gdn_slot,
                    grew,
                },
            }),
        }
    }
}

#[cfg(test)]
mod prefill_tail_merge_tests {
    use super::RunningEngine;

    /// 纯函数面:本模块的测试不需要设备/引擎实例,只需一个具体的 `D` 占位
    /// (`prefill_step_chunk` 不使用 `D`;用 `owl_cpu::CpuFace` 作哑元)。
    type R = RunningEngine<owl_cpu::CpuFace>;

    /// tail_merge = 0 ⇒ 旧行为(min(cfg, 剩余),永不合并)。
    #[test]
    fn tail_merge_off_is_old_behavior() {
        // (page = 0 = 关页对齐 ⇒ 纯 L4 语义;与旧断言逐字相同)
        assert_eq!(R::prefill_step_chunk(0, 2048, 512, 0, 0), Some(512));
        assert_eq!(R::prefill_step_chunk(2048 - 4, 2048, 512, 0, 0), Some(4));
        assert_eq!(R::prefill_step_chunk(0, 100, 512, 0, 0), Some(100));
        assert_eq!(R::prefill_step_chunk(100, 100, 512, 0, 0), None);
    }

    /// tail_merge = 512 ⇒ 剩余 ≤ 1024 一口气吃完(尾块合并;最大块 1024)。
    #[test]
    fn tail_merge_merges_small_tail() {
        // 516 ⇒ 单块(消灭 512+4 的两块结构)
        assert_eq!(R::prefill_step_chunk(0, 516, 512, 512, 0), Some(516));
        // 8244:前段仍 512,最后一段 564(合并 512+52)
        assert_eq!(R::prefill_step_chunk(0, 8244, 512, 512, 0), Some(512));
        assert_eq!(R::prefill_step_chunk(7680, 8244, 512, 512, 0), Some(564));
        // 1024 恰在界上 ⇒ 吃完;1025 ⇒ 回 512
        assert_eq!(R::prefill_step_chunk(0, 1024, 512, 512, 0), Some(1024));
        assert_eq!(R::prefill_step_chunk(0, 1025, 512, 512, 0), Some(512));
        // 1 token 也推得动
        assert_eq!(R::prefill_step_chunk(0, 1, 512, 512, 0), Some(1));
    }

    /// **页对齐**(page = 32;前缀复用闭环):只在"至今没有对齐边界"的单块档
    /// 切一刀,切点 = **最后一个整页边界**;多 chunk 档(L4 合并)不受影响。
    #[test]
    fn page_align_only_for_single_chunk_prompts() {
        // 单块 + 尾端不对齐 ⇒ 切在最后一个整页边界(64 + 19)
        assert_eq!(R::prefill_step_chunk(0, 83, 512, 512, 32), Some(64));
        assert_eq!(R::prefill_step_chunk(64, 83, 512, 512, 32), Some(19));
        // 187 tok(探针 T2 形状)⇒ 160 + 27
        assert_eq!(R::prefill_step_chunk(0, 187, 512, 512, 32), Some(160));
        // 尾端**恰好对齐** ⇒ 不动(已有边界)
        assert_eq!(R::prefill_step_chunk(0, 128, 512, 512, 32), Some(128));
        // 短于一页 ⇒ 切了也没边界可落 ⇒ 不动
        assert_eq!(R::prefill_step_chunk(0, 20, 512, 512, 32), Some(20));
        // **中途段**仍按配置(不是最后一段 ⇒ 不动)
        assert_eq!(R::prefill_step_chunk(512, 8244, 512, 512, 32), Some(512));
        // **L4 长尾段**(生产档 tail_merge=1024):尾端不对齐 ⇒ 切在最后一个整页
        // (8244 → 8224),尾巴 20 另一步 —— 这一刀换来"1728 有快照"级别的复用面
        assert_eq!(R::prefill_step_chunk(7680, 8244, 512, 1024, 32), Some(544));
        assert_eq!(R::prefill_step_chunk(8224, 8244, 512, 1024, 32), Some(20));
        // tail_merge=0 ⇒ 564 拆成 512 + 52(尾段再按对齐切 32 + 20)
        assert_eq!(R::prefill_step_chunk(7680, 8244, 512, 0, 32), Some(512));
        assert_eq!(R::prefill_step_chunk(8192, 8244, 512, 0, 32), Some(32));
        // 关对齐(page = 0)= 纯 L4
        assert_eq!(R::prefill_step_chunk(0, 83, 512, 512, 0), Some(83));
    }

    /// 消息边界切块:只在正常 step 内部提前切;点等于 end/太近/已过都不切。
    #[test]
    fn msg_boundary_cuts_inside_normal_step() {
        let page = 32;
        // 9728 + 512 = 10240;边界 9920 在中间 ⇒ 切到 9920(192 tok)。
        assert_eq!(
            R::prefill_step_chunk_msg(9728, 12000, 512, 0, page, &[9920], 0),
            Some(192)
        );
        // 距 fed 192 < min_span 256 ⇒ 不切,保留正常 512。
        assert_eq!(
            R::prefill_step_chunk_msg(9728, 12000, 512, 0, page, &[9920], 256),
            Some(512)
        );
        // 点在 normal step 之外(== end)⇒ 不额外切。
        assert_eq!(
            R::prefill_step_chunk_msg(9728, 12000, 512, 0, page, &[10240], 0),
            Some(512)
        );
        // 点已经 <= fed ⇒ 忽略；点恰等于 fed 也必须忽略（step=0 会炸算子）。
        assert_eq!(
            R::prefill_step_chunk_msg(9728, 12000, 512, 0, page, &[9696], 0),
            Some(512)
        );
        assert_eq!(
            R::prefill_step_chunk_msg(9728, 12000, 512, 0, page, &[9728], 0),
            Some(512)
        );
        // 多个点:取第一个 >= fed+min_span 的。
        assert_eq!(
            R::prefill_step_chunk_msg(9728, 12000, 512, 0, page, &[9792, 9920], 128),
            Some(192)
        );
        // 与 L4 尾块合并共存:step 原本 564,边界 8192 ⇒ 切到 512。
        assert_eq!(
            R::prefill_step_chunk_msg(7680, 8244, 512, 1024, page, &[8192], 0),
            Some(512)
        );
        // 空点集 = 旧行为。
        assert_eq!(
            R::prefill_step_chunk_msg(9728, 12000, 512, 0, page, &[], 0),
            Some(512)
        );
    }

    /// **生成相位页边界快照的一轮判定**(`decode_boundary_in`)——
    /// 本轮提交 `(pos, fed_new]` 跨过哪张边界、步长何时把它节掉。
    #[test]
    fn decode_boundary_is_largest_crossed_page() {
        let page = 32;
        // 跨过 160(pos=158,m+1=5 ⇒ fed_new=163)⇒ 拍 160
        assert_eq!(R::decode_boundary_in(158, 163, page, 1), Some(160));
        // 恰好落在边界上(fed_new = 160)也算跨过(状态就在那)—— 但 pos=160 时不算
        assert_eq!(R::decode_boundary_in(155, 160, page, 1), Some(160));
        assert_eq!(R::decode_boundary_in(160, 160, page, 1), None);
        // 边界在 pos 之下(本轮未跨)⇒ 不拍
        assert_eq!(R::decode_boundary_in(161, 168, page, 1), None);
        // 一步跨多个边界(页宽 < 提交行数;现役 page=32 不会发生)⇒ 取最大
        assert_eq!(R::decode_boundary_in(30, 70, 8, 1), Some(64));
        // 节流:只在页号是 stride 倍数的边界拍(stride=4 ⇒ 页 4/8/12…)
        assert_eq!(R::decode_boundary_in(158, 163, page, 4), None); // 页 5
        assert_eq!(R::decode_boundary_in(222, 228, page, 4), None); // 页 7
        assert_eq!(R::decode_boundary_in(250, 257, page, 4), Some(256)); // 页 8 ✓
        assert_eq!(R::decode_boundary_in(250, 257, page, 8), Some(256)); // 页 8 ✓(stride 8)
        assert_eq!(R::decode_boundary_in(186, 193, page, 8), None); // 页 6
        assert_eq!(R::decode_boundary_in(186, 193, page, 2), Some(192)); // 页 6 ✓(stride 2)
                                                                         // 关(stride=0)/无页(page=0)
        assert_eq!(R::decode_boundary_in(158, 163, page, 0), None);
        assert_eq!(R::decode_boundary_in(158, 163, 0, 1), None);
        // 尾首:pos=0 且 fed_new 已过第一页 ⇒ 拍
        assert_eq!(R::decode_boundary_in(0, 33, page, 1), Some(32));
        assert_eq!(R::decode_boundary_in(0, 31, page, 1), None);
    }
}
