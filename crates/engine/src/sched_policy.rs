//! **调度政策**：准入与相位 —— **两个互不耦合的状态机**（纯数据/纯函数）。
//!
//! # 为什么单独成文件
//!
//! 准入（"谁能进"）与相位（"这一拍谁跑"）是**两件事**，混在 `schedule()` 里会互相
//! 污染判据（历史教训：三套"谁该走"的判据，见 store 设计卷 §30）。本模块：
//!
//! - **零 I/O、零设备、零引擎类型**（只 `use std`）—— 单测不需要卡；
//! - **事实全部由外部注入**：每次调用把 `StoreFacts`（store 的池/层账）与
//!   `&[TurnFacts]`（引擎的活跃/队首事实）传进来；本模块**不查**任何全局/单例；
//! - **状态机只记自己的账**（预留页、退避时刻、时间片游标），不碰会话/KV/GDN。
//!
//! # 两个状态机
//!
//! ```text
//! ┌─────────────── Admission（准入）───────────────┐   ┌────────── PhaseSched（相位）──────────┐
//! │ 输入: StoreFacts + 队首/活跃 TurnFacts           │   │ 输入: &[TurnFacts]（相位/欠账）        │
//! │ 状态: 预留账 reserved / 本批 admitted / 退避表   │   │ 状态: rr 游标 / 距上一片 prefill 的拍数 │
//! │ 输出: Admit | Hold | Reject{reason,retry_after}  │   │ 输出: Plan{kind, turn, slice_tokens}   │
//! │ 律: A1..A4（见下）                               │   │ 律: P1..P4（见下）                     │
//! └──────────────────────────────────────────────────┘   └────────────────────────────────────────┘
//! ```
//!
//! 两者的唯一共同输入是 `TurnFacts` 的**只读快照**；**没有共享可变状态**。
//!
//! # 不变式
//!
//! ## 准入（Admission）
//! - **A1 单一放行口**：引擎只在本模块返回 `Admit` 时才借 GDN 行 / 长块链；
//!   其它路径不得自行判定"容量够"。
//! - **A2 预留口径**：
//!   - 不超卖（`oversub = false`）：`pages = ceil((prompt + max_new) / page)`
//!     —— prefill 要读前缀链、decode 每步读整条链，**整条链必须驻留**；
//!   - 超卖（`oversub = true`）：`pages = workspace_pages + chunk_pages`
//!     —— 只有工作集驻留，其余沉宿主/磁盘。
//! - **A3 在跑者不受新人影响**：预留账满 ⇒ 新人 `Hold`/`Reject`，**绝不**驱逐/收缩
//!   已在跑的 turn（"实质性进展"底线）。
//! - **A4 拒绝可行动**：必带 `Reason`（内存/槽位/会话/排队超时）+ `retry_after_ms`。
//!
//! ## 相位（PhaseSched）
//! - **P1 在位者 ITL 上界**：每 `slice_every` 拍 decode 之后**至多**插一片 prefill；
//!   由此 `ITL_A ≤ solo_ITL + slice_ms / (slice_every + 1)`（可写成门）。
//! - **P2 一片一个 turn**：prefill 片 = `min(剩余 prompt, slice_tokens)`，一次只喂一个；
//! - **P3 批优先**：全活跃都在 gen 且批位够 ⇒ `BatchDecode`（权重摊薄）；
//!   相位混合 ⇒ 单动作（批门要求同相位，不许硬拼）；
//! - **P4 公平**：prefill 挑**欠账最多**（`fed` 最小）；gen 挑 round-robin。
//!
//! # 与引擎的接线（后续，不在本文件）
//!
//! ```text
//! submit/schedule:
//!   let verdict = admission.decide(&store_facts, &running, &head, now_ms);
//!   match verdict { Admit => 借 GDN 行 + ensure_for_len + 入 active,
//!                   Hold  => 留队首不动（下一拍再判）,
//!                   Reject{reason, retry_after_ms} => 出队 + 事件（服务面翻 429） }
//! 每拍执行:
//!   match phase.next(&actives, batch_ok) { Some(Plan{..}) => 发对应 StepAction, None => Idle }
//! ```

use std::collections::{BTreeMap, BTreeSet};

// ===========================================================================
// §1 静态几何（boot 派生一次；由引擎注入）
// ===========================================================================

/// 调度几何（**静态**；改这些必须重启引擎 —— 与配置链同律）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// 池页宽（token/页；现役 32）
    pub page: usize,
    /// 批位上限（`[runtime] max_batch`；1 = 串行）
    pub max_batch: usize,
    /// GDN 状态格数（每条活跃 turn 占一格）
    pub gdn_slots: usize,
    /// 服务面请求上限（**不是容量**；只做请求侧闸）
    pub max_concurrency: usize,
    /// 超卖两扇门都开（`select_decode ∧ select_partial`）
    pub oversub: bool,
    /// 超卖档：每条 turn 的设备工作集页数（`ceil(workspace_per_req / page)`）
    pub workspace_pages: usize,
    /// 超卖档：本拍 prefill chunk 页数（`ceil(prefill_chunk / page)`）
    pub chunk_pages: usize,
    /// 池保留页（增长/腾挪余量；几页级，**不是**按比例）
    pub reserve_pages: usize,
    /// 时间片：每 `slice_every` 拍 decode 后插一片 prefill（0 = 不插，prefill 独占）
    pub slice_every: usize,
    /// 时间片：单片 prefill token 数（0 = 用整 chunk）
    pub slice_tokens: usize,
    /// **给未来新人的预留**（token/人；不超卖档用来反推在位者的生成封顶）
    pub newcomer_min_tokens: usize,
    /// 在位者的最小生成量（低于它就不留新人预留 —— 在位者优先）
    pub min_gen_tokens: usize,
    /// **等待队列上限**（0 = 无界）。`Hold` 的 turn 数超过它 ⇒ 改判
    /// `Reject{NoSlot}`（429），否则狂发请求会把 host 队列撑爆。
    pub max_queue: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            page: 32,
            max_batch: 1,
            gdn_slots: 1,
            max_concurrency: 1,
            oversub: false,
            workspace_pages: 0,
            chunk_pages: 0,
            reserve_pages: 8,
            slice_every: 4,
            slice_tokens: 64,
            newcomer_min_tokens: 4096,
            min_gen_tokens: 256,
            max_queue: 0,
        }
    }
}

// ===========================================================================
// §2 外部注入的事实（只读快照）
// ===========================================================================

/// **store 的账**（唯一内存事实源；引擎每拍注入，不许自己反推）
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoreFacts {
    /// 设备池空闲页（`nb − 已占`）
    pub free_pages: usize,
    /// 设备池里"可回收的缓存页"（非活跃、非预留；可被驱逐/下沉）
    pub cache_pages: usize,
    /// 准入账已预留页（store 侧 `admission` 账）
    pub reserved_pages: usize,
    /// 宿主层空闲页（超卖档的下沉腿；不超卖档 = 0 也无妨）
    pub host_free_pages: usize,
    /// 磁盘层空闲页（0 = 无磁盘腿）
    pub disk_free_pages: usize,
}

impl StoreFacts {
    /// 设备侧"可支配页" = 空闲 + 可回收缓存 − 保留（饱和减）
    pub fn usable_pages(&self, reserve: usize) -> usize {
        (self.free_pages + self.cache_pages).saturating_sub(reserve)
    }
}

/// 一条 turn 的只读事实（队首或活跃；引擎注入）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TurnFacts {
    pub turn: u64,
    pub session: u64,
    /// prompt token 数（已 tokenize）
    pub prompt_tokens: usize,
    /// 本 turn 生成预算（含预算收缩前的请求值）
    pub max_new: usize,
    /// 已喂入的 prompt token 数（prefill 进度；= prompt_tokens 即进 gen）
    pub fed: usize,
    /// 已产出 token 数（gen 进度）
    pub out_len: usize,
    pub phase: PhaseSz,
    /// 投机档（depth > 0）—— 批门要求无投机
    pub spec: bool,
}

/// 相位（**事实**，不是决策）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhaseSz {
    /// `fed < prompt_tokens`
    Prefill,
    /// `fed == prompt_tokens`，正常 decode
    Gen,
    /// 投机轮（verify/spec）
    Spec,
}

impl TurnFacts {
    /// **超卖档：需要沉到宿主/磁盘的页数**（= 全量 ctx − 设备工作集）。
    /// 非超卖 = 0（全量驻留，不沉）。
    pub fn offload_pages(&self, l: &Limits) -> usize {
        if !l.oversub {
            return 0;
        }
        let total = self.prompt_tokens.saturating_add(self.max_new);
        let resident = l.workspace_pages.saturating_add(l.chunk_pages).saturating_mul(l.page.max(1));
        total.saturating_sub(resident).div_ceil(l.page.max(1))
    }

    /// 距 prompt 灌完还差多少 token（0 = 已进 gen）
    pub fn prefill_left(&self) -> usize {
        self.prompt_tokens.saturating_sub(self.fed)
    }

    /// **A2 预留口径**：本 turn 需要多少设备页（`cap` = 生成封顶；
    /// `usize::MAX` = 不封顶）。封顶由 [`Admission::cap_for`] 反推 ——
    /// 见"在位者优先 + 给新人留头寸"的口径（工作单 §七）。
    pub fn need_pages(&self, l: &Limits, cap: usize) -> usize {
        if l.oversub {
            // 超卖：只驻工作集 + 本拍 chunk（生成预算不额外占设备）
            l.workspace_pages + l.chunk_pages
        } else {
            // 不超卖：**整条链**（prompt + 生成预算）必须驻留
            let new = self.max_new.min(cap);
            let tokens = self.prompt_tokens.saturating_add(new);
            tokens.div_ceil(l.page.max(1)).max(1)
        }
    }
}

// ===========================================================================
// §3 准入状态机（Admission）
// ===========================================================================

/// 容量快照（诊断/门用；一次判定的中间结果，纯值）
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Capacity {
    /// 槽位上限 = `min(max_batch, gdn_slots, max_concurrency)`
    pub n_slots: usize,
    /// 内存上限（当前事实下还能再进几个同尺寸 turn）
    pub n_mem: usize,
    /// 实际放行上限 = `min(n_slots, n_mem, 活跃+已预留的余额)`
    pub n_eff: usize,
    /// 设备池已用页（含预留）
    pub used_pages: usize,
    /// 设备池可支配页
    pub usable_pages: usize,
    /// 在跑 turn 里最大的一份预留页（诊断；页账仍以 store 单源为准）
    pub largest_running_pages: usize,
}

/// 拒绝原因（**必须可行动**；服务面据此翻 429/409 + `Retry-After`）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// 槽位满（批位/GDN 格/请求上限）
    NoSlot,
    /// 设备内存不足（不超卖：Σctx；超卖：工作集 × N）
    NoMemory { need_pages: usize, free_pages: usize },
    /// 同会话已有活跃/排队 turn（引擎不变量；服务面应已串行）
    SessionBusy,
    /// 排队超时（服务面按 `retry_after_ms` 退避）
    QueueTimeout { waited_ms: u64 },
}

impl Reason {
    /// 稳定的机器可读标签（进日志/JSON；**不许**用 `Debug` 当协议）
    pub fn code(self) -> &'static str {
        match self {
            Reason::NoSlot => "no_slot",
            Reason::NoMemory { .. } => "no_memory",
            Reason::SessionBusy => "session_busy",
            Reason::QueueTimeout { .. } => "queue_timeout",
        }
    }
}

/// 准入判决
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// 放行（引擎：借 GDN 行 + 长块链 + 入活跃）
    Admit,
    /// 留队首（下一拍再判；**不算失败**）
    Hold,
    /// 拒绝（出队 + 事件；服务面翻 429 + `Retry-After`）
    Reject { reason: Reason, retry_after_ms: u32 },
}

/// 准入状态机
#[derive(Debug, Default)]
pub struct Admission {
    /// 已放行、尚未终了的 turn（预留账；终了归还）
    admitted: BTreeMap<u64, usize>, // turn → 预留页
    /// 会话 → 在场 turn（同会话并发守卫；值 = turn id）
    sess_turn: BTreeMap<u64, u64>,
    /// 退避表：会话 → 最近一次拒绝时刻（毫秒；防打摆子）
    backoff: BTreeMap<u64, u64>,
    /// 已放行的 turn 集合（幂等；重复 `Admit` 不允许）
    once: BTreeSet<u64>,
}

impl Admission {
    pub fn new() -> Self {
        Self::default()
    }

    /// 目标并发（份额分母）：`min(max_batch, gdn_slots, max_concurrency)`
    pub fn n_target(&self, l: &Limits) -> usize {
        l.max_batch.min(l.gdn_slots).min(l.max_concurrency.max(1)).max(1)
    }

    /// **在位者优先 + 给新人留头寸** 的生成封顶（token）。
    ///
    /// 口径（用户裁决 2026-10-20）：
    /// ```text
    /// 可用 token = store.usable_pages × page
    /// 封顶 = 可用 − 其它在跑 turn 的预留 − (n_target−1) × newcomer_min − 本 turn prompt
    /// 若该值 < min_gen（把在位者压得不能生成）⇒ **放弃新人预留**，封顶 = 可用 − 其它 − prompt
    /// ```
    /// 超卖档：生成不额外占设备 ⇒ 返回 `max_new`（不封顶）。
    pub fn cap_for(
        &self,
        l: &Limits,
        s: &StoreFacts,
        head: &TurnFacts,
    ) -> usize {
        if l.oversub {
            return head.max_new;
        }
        // `usable` 已是**剩余**（不含任何已预留）⇒ 这里只减本 turn 的 prompt。
        let usable_tokens = s.usable_pages(l.reserve_pages).saturating_mul(l.page);
        let free_after_prompt = usable_tokens.saturating_sub(head.prompt_tokens);
        let reserve_new = l
            .newcomer_min_tokens
            .saturating_mul(self.n_target(l).saturating_sub(1));
        let with_reserve = free_after_prompt.saturating_sub(reserve_new);
        if with_reserve >= l.min_gen_tokens {
            with_reserve
        } else {
            free_after_prompt
        }
    }

    /// **容量快照**（纯函数；门/诊断/单测直接用）。
    ///
    /// `running` = 活跃 turn（它们已占预留）；`incoming` = 待判 turn。
    pub fn capacity(&self, l: &Limits, s: &StoreFacts, running: &[TurnFacts], incoming: &TurnFacts) -> Capacity {
        // **页账单源 = store**（`reserved_pages` 就是在跑 turn 的预留;不许再
        // 叠加 `running` 的 need —— 那会双算（本骨架第一版就踩了,测试抓到））。
        let used_pages = s.reserved_pages;
        let usable = s.usable_pages(l.reserve_pages);
        let n_slots = l.max_batch.min(l.gdn_slots).min(l.max_concurrency.max(1));
        let cap_tokens = self.cap_for(l, s, incoming);
        // 与 `decide` 同判据：非超卖档下 cap < min_gen = 进得来也吐不出 token
        // ⇒ 不计入容量（否则 n_mem 会给出"能进 1 个"的假象）。
        let per_turn = incoming.need_pages(l, cap_tokens).max(1);
        let n_mem = if !l.oversub {
            if cap_tokens < l.min_gen_tokens {
                0
            } else {
                usable / per_turn
            }
        } else {
            // 超卖档：设备工作集之外的空间要落宿主/磁盘 ⇒ 两腿取小
            let offload = incoming.offload_pages(l).max(1);
            let n_host = (s.host_free_pages.saturating_add(s.disk_free_pages)) / offload;
            (usable / per_turn).min(n_host)
        };
        // `running` 的用途 = 诊断（最大在跑 turn 的尺寸;门/日志读）
        let largest_running = running.iter().map(|t| t.need_pages(l, usize::MAX)).max().unwrap_or(0);
        glue(capacity_result(n_slots, n_mem, used_pages, usable, largest_running))
    }

    /// **放行判定**（唯一放行口）。`now_ms` = 单调时钟毫秒（由引擎注入）。
    pub fn decide(
        &mut self,
        l: &Limits,
        s: &StoreFacts,
        running: &[TurnFacts],
        head: &TurnFacts,
        now_ms: u64,
    ) -> Verdict {
        // ① 同会话守卫（引擎不变量：同会话两 turn 会共享 KV/GDN = 互踩）
        if let Some(t) = self.sess_turn.get(&head.session) {
            if *t != head.turn {
                return Verdict::Reject { reason: Reason::SessionBusy, retry_after_ms: 1000 };
            }
        }
        // ② 槽位
        let cap = self.capacity(l, s, running, head);
        let gen_cap = self.cap_for(l, s, head);
        // 非超卖档：
        // · 连 1 token 都放不下 ⇒ 拒（进来也吐不出东西）；
        // · 有**同住者**且本 turn 的生成余量 < `min_gen` ⇒ 拒（别让新人当陪跑）；
        // · **独自一人**时即便余量 < min_gen 也放行（大 prompt 独占是合法形态，
        //   生成到 `cap` 就 `length` 收尾）—— 判据是"有没有别人"，不是绝对值。
        if !l.oversub && (gen_cap == 0 || (gen_cap < l.min_gen_tokens && !self.admitted.is_empty()))
        {
            return Verdict::Reject {
                reason: Reason::NoMemory {
                    need_pages: head.need_pages(l, 0),
                    free_pages: 0,
                },
                retry_after_ms: 1000,
            };
        }
        let need = head.need_pages(l, gen_cap);
        if need > cap.usable_pages {
            return Verdict::Reject {
                reason: Reason::NoMemory {
                    need_pages: need,
                    free_pages: cap.usable_pages,
                },
                retry_after_ms: 1000,
            };
        }
        // **超卖档：宿主/磁盘腿**（被挤出去的上下文必须有地方落）
        if l.oversub {
            let off = head.offload_pages(l);
            let host = s.host_free_pages.saturating_add(s.disk_free_pages);
            if off > host {
                return Verdict::Reject {
                    reason: Reason::NoMemory {
                        need_pages: off,
                        free_pages: host,
                    },
                    retry_after_ms: 1000,
                };
            }
        }
        if self.admitted.len() >= cap.n_slots {
            return Verdict::Hold; // 槽满但内存够 ⇒ 排队（不是失败）
        }
        // ③ 退避（同会话刚被拒过 ⇒ 本拍不重判，避免同一拍反复出队/入队）
        if let Some(ts) = self.backoff.get(&head.session) {
            if now_ms.saturating_sub(*ts) < 200 {
                return Verdict::Hold;
            }
        }
        Verdict::Admit
    }

    /// 引擎在**真正起 turn 后**回报（预留入账；幂等）。
    pub fn note_started(&mut self, l: &Limits, s: &StoreFacts, t: &TurnFacts) {
        if self.once.insert(t.turn) {
            self.admitted.insert(t.turn, t.need_pages(l, self.cap_for(l, s, t)));
            self.sess_turn.insert(t.session, t.turn);
        }
    }

    /// 引擎在 turn 终了（complete/abandon/拒）后回报（预留归还）。
    pub fn note_finished(&mut self, turn: u64) {
        if let Some(pages) = self.admitted.remove(&turn) {
            let _ = pages;
            self.once.remove(&turn);
        }
        self.sess_turn.retain(|_, t| *t != turn);
    }

    /// 记账拒绝（供退避用；服务面/引擎在拒时调用）。
    pub fn note_rejected(&mut self, session: u64, now_ms: u64, reason: Reason) {
        let _ = reason;
        self.backoff.insert(session, now_ms);
    }

    /// 诊断：在册预留页 / 在登 turn（`/debug/vram` 用）。
    pub fn reserved_pages(&self) -> usize {
        self.admitted.values().sum()
    }

    pub fn in_flight(&self) -> usize {
        self.admitted.len()
    }
}

/// 小工具：把中间量装进 [`Capacity`]（保持结构体字段全显式）
fn capacity_result(
    n_slots: usize,
    n_mem: usize,
    used_pages: usize,
    usable_pages: usize,
    largest_running_pages: usize,
) -> Capacity {
    Capacity {
        n_slots,
        n_mem,
        n_eff: n_slots.min(n_mem),
        used_pages,
        usable_pages,
        largest_running_pages,
    }
}

/// 恒等（占位：保留"结果必须过一处构造"的形状，便于后续加审计）
fn glue(c: Capacity) -> Capacity {
    c
}

// ===========================================================================
// §4 相位状态机（PhaseSched）
// ===========================================================================

/// 本拍动作（**只表达"谁/什么相位/多少 token"**；设备派生量留给引擎）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// 灌一片 prefill（`slice_tokens` ≤ 剩余）
    Prefill,
    /// 单行 decode
    Decode,
    /// 多行 decode（批）
    BatchDecode,
}

/// 本拍计划
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub kind: Kind,
    /// 目标 turn（BatchDecode = 0，表示"全部 gen 行"）
    pub turn: u64,
    /// Prefill 片长（其它相位 = 0）
    pub slice_tokens: usize,
}

/// 相位状态机
#[derive(Debug, Default)]
pub struct PhaseSched {
    /// 距上一片 prefill 的 decode 拍数（P1）
    decodes_since_slice: usize,
    /// gen 相位 round-robin 游标（P4）
    rr: usize,
    /// 上一拍挑中的 turn（同优先级时轮转用；0 = 无）
    last: u64,
}

impl PhaseSched {
    pub fn new() -> Self {
        Self::default()
    }

    /// **本拍计划**（纯函数 + 两个游标；`batch_ok` = 引擎的批门外因，
    /// 如 TP/选择层/投机 —— 本模块不判断这些，只接受一个布尔）。
    pub fn next(&mut self, l: &Limits, actives: &[TurnFacts], batch_ok: bool) -> Option<Plan> {
        if actives.is_empty() {
            return None;
        }
        // 相位分类
        let gen: Vec<&TurnFacts> = actives.iter().filter(|t| t.phase != PhaseSz::Prefill).collect();
        let pending: Vec<&TurnFacts> = actives.iter().filter(|t| t.phase == PhaseSz::Prefill).collect();

        // P3:全 gen + 批位 ⇒ 批（不管时间片；批本身就是"人人前进"）
        if pending.is_empty() {
            if gen.len() >= 2 && batch_ok && l.max_batch >= gen.len() {
                self.decodes_since_slice = self.decodes_since_slice.saturating_add(1);
                self.last = 0;
                return Some(Plan { kind: Kind::BatchDecode, turn: 0, slice_tokens: 0 });
            }
            let tid = self.pick_gen(&gen);
            self.decodes_since_slice = self.decodes_since_slice.saturating_add(1);
            self.last = tid;
            return Some(Plan { kind: Kind::Decode, turn: tid, slice_tokens: 0 });
        }

        // 有 prefill 欠账：没有 gen 可跑 ⇒ prefill **独占**（不切片）。
        // 判据：切片只为"保护在位者"；没有在位者时切片只会白拉长自己的 TTFT
        // （真机实测 106→638ms，2026-10-20）。
        if gen.is_empty() {
            let t = pick_prefill(&pending);
            self.decodes_since_slice = 0;
            self.last = t.turn;
            return Some(Plan {
                kind: Kind::Prefill,
                turn: t.turn,
                slice_tokens: t.prefill_left().max(1),
            });
        }

        // P1:每 `slice_every` 拍 decode 之后插一片（=0 ⇒ 不插 = 旧行为）
        if l.slice_every == 0 || self.decodes_since_slice >= l.slice_every {
            let t = pick_prefill(&pending);
            self.decodes_since_slice = 0;
            self.last = t.turn;
            return Some(Plan {
                kind: Kind::Prefill,
                turn: t.turn,
                slice_tokens: slice_of(l, t),
            });
        }

        // 否则 decode（相位混合时批门不成立 ⇒ 单行）
        let tid = self.pick_gen(&gen);
        self.decodes_since_slice = self.decodes_since_slice.saturating_add(1);
        self.last = tid;
        Some(Plan { kind: Kind::Decode, turn: tid, slice_tokens: 0 })
    }

    /// P4:round-robin（跳过上一拍挑中的；单元素时直接返回其 turn id）。
    ///
    /// 返回 **turn id** 而不是引用:调用方紧接着要改自己的游标(借用不可交叠)。
    fn pick_gen(&mut self, gen: &[&TurnFacts]) -> u64 {
        debug_assert!(!gen.is_empty(), "pick_gen: 空集不许进来（调用方已判）");
        if gen.len() == 1 {
            return gen[0].turn;
        }
        let n = gen.len();
        let mut i = self.rr % n;
        if gen[i].turn == self.last {
            i = (i + 1) % n;
        }
        self.rr = (i + 1) % n;
        gen[i].turn
    }
}

/// 片长 = `min(剩余, slice_tokens)`；`slice_tokens = 0` ⇒ 整剩余（旧行为）
fn slice_of(l: &Limits, t: &TurnFacts) -> usize {
    let left = t.prefill_left().max(1);
    if l.slice_tokens == 0 {
        left
    } else {
        left.min(l.slice_tokens)
    }
}

/// 欠账最多者（`fed` 最小；同值取 turn 小的 —— 确定性）
fn pick_prefill<'a>(pending: &[&'a TurnFacts]) -> &'a TurnFacts {
    debug_assert!(!pending.is_empty(), "pick_prefill: 空集不许进来");
    let mut best = pending[0];
    for t in &pending[1..] {
        if (t.fed, t.turn) < (best.fed, best.turn) {
            best = t;
        }
    }
    best
}

// ===========================================================================
// §5 单测（纯；零 GPU）
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn limits_non_oversub() -> Limits {
        Limits { page: 32, max_batch: 2, gdn_slots: 2, max_concurrency: 8, reserve_pages: 8, ..Default::default() }
    }

    fn limits_oversub() -> Limits {
        Limits {
            page: 32,
            max_batch: 8,
            gdn_slots: 8,
            max_concurrency: 8,
            oversub: true,
            workspace_pages: 32, // 1024 tok
            chunk_pages: 32,     // 1024 tok
            reserve_pages: 8,
            slice_every: 4,
            slice_tokens: 64,
            newcomer_min_tokens: 4096,
            min_gen_tokens: 256,
            max_queue: 0,
        }
    }

    fn turn(id: u64, sess: u64, prompt: usize, max_new: usize) -> TurnFacts {
        TurnFacts {
            turn: id,
            session: sess,
            prompt_tokens: prompt,
            max_new,
            fed: prompt, // 默认已进 gen
            out_len: 0,
            phase: PhaseSz::Gen,
            spec: false,
        }
    }

    fn store(free: usize, cache: usize) -> StoreFacts {
        StoreFacts { free_pages: free, cache_pages: cache, host_free_pages: 1 << 20, ..Default::default() }
    }

    /// 不超卖：两路各 4k prompt + 4k gen（256 页）在 528 页池里 ⇒ 2 个；
    /// 满 ctx（512 页）⇒ 1 个。
    #[test]
    fn non_oversub_capacity_follows_ctx() {
        let l = limits_non_oversub();
        let a = Admission::new();
        let s = store(520, 0);
        let t = turn(1, 10, 4096, 4096);
        assert_eq!(t.need_pages(&l, usize::MAX), 256);
        let cap = a.capacity(&l, &s, &[], &t);
        assert_eq!(cap.n_mem, 2);
        let big = turn(2, 11, 8192, 8192);
        assert_eq!(big.need_pages(&l, usize::MAX), 512);
        let cap = a.capacity(&l, &s, &[], &big);
        assert_eq!(cap.n_mem, 1);
        // 两路 half ctx 已在跑（store 已记 512 页预留）⇒ 新人放不下
        let running = [turn(1, 10, 4096, 4096), turn(2, 11, 4096, 4096)];
        let busy = StoreFacts { free_pages: 8, reserved_pages: 512, ..Default::default() };
        let cap = a.capacity(&l, &busy, &running, &turn(3, 12, 4096, 4096));
        assert_eq!(cap.n_mem, 0, "两份已占满 512 页，第三个不许进");
        assert_eq!(cap.largest_running_pages, 256, "诊断面:最大在跑预留");
    }

    /// 超卖：N 只跟工作集有关（1024 页池 / 64 页 = 15），与 ctx 无关。
    #[test]
    fn oversub_capacity_ignores_ctx() {
        let l = limits_oversub();
        let a = Admission::new();
        let s = store(1016, 0);
        let small = turn(1, 10, 2048, 2048);
        let huge = turn(2, 11, 200_000, 8192);
        assert_eq!(small.need_pages(&l, usize::MAX), 64);
        assert_eq!(huge.need_pages(&l, usize::MAX), 64, "超卖:只驻工作集");
        assert_eq!(a.capacity(&l, &s, &[], &small).n_mem, 15);
        assert_eq!(a.capacity(&l, &s, &[], &huge).n_mem, 15);
    }

    /// 准入：内存够 ⇒ 放行；槽满 ⇒ Hold（不是拒绝）；同会话 ⇒ 拒；在跑者不被挤。
    #[test]
    fn admission_verdicts_are_explicit() {
        let l = limits_non_oversub();
        let s = store(520, 0);
        let mut a = Admission::new();
        let t1 = turn(1, 10, 4096, 4096); // 256 页
        assert_eq!(a.decide(&l, &s, &[], &t1, 0), Verdict::Admit);
        a.note_started(&l, &s, &t1);
        // 同会话第二 turn ⇒ SessionBusy
        let dup = turn(2, 10, 1024, 1024);
        assert!(matches!(
            a.decide(&l, &s, &[t1], &dup, 0),
            Verdict::Reject { reason: Reason::SessionBusy, .. }
        ));
        // 新会话 + 槽位还有 1 个 ⇒ Admit（页账由 store 注入:第一人已占 256）
        let s1 = StoreFacts { free_pages: 264, reserved_pages: 256, ..Default::default() };
        let t2 = turn(3, 11, 1024, 1024); // 64 页
        assert_eq!(a.decide(&l, &s1, &[t1], &t2, 0), Verdict::Admit);
        a.note_started(&l, &s1, &t2);
        // 槽位满（2）⇒ 第三人 Hold（内存仍够:320/512）
        let s2 = StoreFacts { free_pages: 200, reserved_pages: 320, ..Default::default() };
        let t3 = turn(4, 12, 1024, 1024);
        assert_eq!(a.decide(&l, &s2, &[t1, t2], &t3, 0), Verdict::Hold);
        // 在跑者终了 ⇒ 名额与预留一起回来（剩第二人的 64 页）
        a.note_finished(1);
        assert_eq!(a.in_flight(), 1);
        assert_eq!(a.reserved_pages(), 64);
    }


    /// **用户场景门（2026-10-20 裁决）**：200K 池、A 已 150K、B 全新进来。
    ///
    /// 口径 = "在位者优先 + 给新人留头寸"：
    /// `cap = usable − 其它预留 − (n_target−1)×newcomer_min − prompt`；
    /// 若把在位者压到 < min_gen ⇒ 放弃新人预留（在位者独占）。
    #[test]
    fn incumbent_cap_leaves_room_for_newcomer() {
        let l = Limits {
            page: 32,
            max_batch: 2,
            gdn_slots: 2,
            max_concurrency: 2,
            reserve_pages: 8,
            newcomer_min_tokens: 4096,
            min_gen_tokens: 256,
            ..Default::default()
        };
        let mut a = Admission::new();
        // 200K 池 = 6250 页;可用 = 6242 页 = 199,744 tok
        let s0 = StoreFacts { free_pages: 6250, ..Default::default() };
        // A:prompt 142K,请求 50K 生成（约占满 192K）
        let ta = turn(1, 10, 142_000, 50_000);
        assert_eq!(a.cap_for(&l, &s0, &ta), 53_648, "A 的封顶（留 4K 给新人）");
        assert_eq!(ta.need_pages(&l, a.cap_for(&l, &s0, &ta)), 6000);
        a.note_started(&l, &s0, &ta);
        // B:全新,prompt 300,请求 8K ⇒ 被夹到 3,348（**必须如实告知客户端**）
        let tb = turn(2, 11, 300, 8192);
        let s1 = StoreFacts { free_pages: 250, reserved_pages: 6000, ..Default::default() };
        let cap_b = a.cap_for(&l, &s1, &tb);
        assert_eq!(cap_b, 3_348, "B 的生成封顶（A 已占 192K）");
        assert_eq!(tb.need_pages(&l, cap_b), 114);
        assert_eq!(a.decide(&l, &s1, &[ta], &tb, 0), Verdict::Admit, "B 应能进来");
    }

    /// 在位者把池吃满（prompt 196K + 生成）⇒ 新人**响亮拒**（不是进来撞墙）。
    #[test]
    fn newcomer_rejected_when_incumbent_claims_pool() {
        let l = Limits {
            page: 32,
            max_batch: 2,
            gdn_slots: 2,
            max_concurrency: 2,
            reserve_pages: 8,
            newcomer_min_tokens: 4096,
            min_gen_tokens: 256,
            ..Default::default()
        };
        let mut a = Admission::new();
        let s0 = StoreFacts { free_pages: 6250, ..Default::default() };
        let ta = turn(1, 10, 196_000, 8192);
        // 新人预留会把 A 压到不能生成 ⇒ 放弃预留,A 吃掉剩余（3,744 tok）
        assert_eq!(a.cap_for(&l, &s0, &ta), 3_744);
        assert_eq!(ta.need_pages(&l, 3_744), 6242);
        a.note_started(&l, &s0, &ta);
        let tb = turn(2, 11, 300, 8192);
        let s1 = StoreFacts { free_pages: 8, reserved_pages: 6242, ..Default::default() };
        match a.decide(&l, &s1, &[ta], &tb, 0) {
            Verdict::Reject { reason: Reason::NoMemory { need_pages, free_pages }, .. } => {
                assert_eq!(need_pages, 10);
                assert_eq!(free_pages, 0);
                assert_eq!(Reason::NoMemory { need_pages, free_pages }.code(), "no_memory");
            }
            other => panic!("应拒（newcomer_rejected）,得到 {other:?}"),
        }
    }

    /// 相位：全 gen 两行 ⇒ BatchDecode；混合 ⇒ 每 4 拍 decode 后插 1 片 prefill。
    #[test]
    fn phase_slices_prefill_after_k_decodes() {
        let l = limits_oversub(); // slice_every = 4, slice = 64
        let mut p = PhaseSched::new();
        let mut gen = turn(1, 10, 128, 64);
        let mut pre = turn(2, 11, 4096, 64);
        pre.fed = 0;
        pre.phase = PhaseSz::Prefill;
        let two = [gen, pre];
        // 前 4 拍：decode（每拍一次）
        for k in 0..4 {
            match p.next(&l, &two, true).expect("plan") {
                Plan { kind: Kind::Decode, turn: 1, .. } => assert_eq!(k, k),
                other => panic!("第 {k} 拍应为 Decode,得到 {other:?}"),
            }
        }
        // 第 5 拍：prefill 片
        match p.next(&l, &two, true).expect("plan") {
            Plan { kind: Kind::Prefill, turn: 2, slice_tokens: 64 } => {}
            other => panic!("应为 Prefill 64,得到 {other:?}"),
        }
        // 全 gen ⇒ 批
        gen.phase = PhaseSz::Gen;
        pre.phase = PhaseSz::Gen;
        let two = [gen, pre];
        assert_eq!(
            p.next(&l, &two, true),
            Some(Plan { kind: Kind::BatchDecode, turn: 0, slice_tokens: 0 })
        );
        // 只剩 gen 一人 ⇒ 单行
        assert_eq!(p.next(&l, &[gen], true), Some(Plan { kind: Kind::Decode, turn: 1, slice_tokens: 0 }));
        // 空集 ⇒ None
        assert_eq!(p.next(&l, &[], true), None);
    }
}

// ===========================================================================
// §6 场景模拟（时间线；两个状态机按引擎的调用序拼起来，纯逻辑零 GPU）
// ===========================================================================
#[cfg(test)]
mod scenario_tests {
    use super::*;
    use std::collections::BTreeMap;

    /// 时间线模拟器：**每拍** = ①准入（队列→活跃）②相位（谁跑）③推进/终局。
    /// 与引擎的调用序一致（`decide` → `note_started` → `phase.next` → 推进 → `note_finished`）。
    pub(super) struct Sim {
        l: Limits,
        adm: Admission,
        ph: PhaseSched,
        queued: Vec<TurnFacts>,
        actives: Vec<TurnFacts>,
        total_pages: usize,
        /// 宿主腿（超卖档用；非超卖档不读）
        host_pages: usize,
        now: u64,
        pub(super) finished: Vec<u64>,
        pub(super) rejected: Vec<(u64, Reason)>,
        pub(super) clamped: Vec<(u64, usize)>,
        pub(super) kinds: Vec<Kind>,
        /// 每 turn：上一次推进之后的拍数 / 历史最大停顿（gen 相位才计）
        gap: BTreeMap<u64, usize>,
        max_gap: BTreeMap<u64, usize>,
    }

    impl Sim {
        pub(super) fn new(l: Limits, total_pages: usize) -> Self {
            Self {
                l,
                adm: Admission::new(),
                ph: PhaseSched::new(),
                queued: Vec::new(),
                actives: Vec::new(),
                total_pages,
                host_pages: 1 << 20,
                now: 0,
                finished: Vec::new(),
                rejected: Vec::new(),
                clamped: Vec::new(),
                kinds: Vec::new(),
                gap: BTreeMap::new(),
                max_gap: BTreeMap::new(),
            }
        }

        /// store 事实（页账单源：已预留 = 准入账）
        fn store(&self) -> StoreFacts {
            let used = self.adm.reserved_pages();
            StoreFacts {
                free_pages: self.total_pages.saturating_sub(used),
                reserved_pages: used,
                host_free_pages: self.host_pages,
                ..Default::default()
            }
        }

        pub(super) fn submit(&mut self, t: TurnFacts) {
            self.queued.push(t);
        }

        /// 一拍；返回 false = 彻底空了（无人可跑）
        pub(super) fn step(&mut self) -> bool {
            self.now += 1;
            // ① 准入
            loop {
                let Some(head) = self.queued.first().copied() else { break };
                let s = self.store();
                match self.adm.decide(&self.l, &s, &self.actives, &head, self.now) {
                    Verdict::Admit => {
                        let cap = self.adm.cap_for(&self.l, &s, &head);
                        let mut t = self.queued.remove(0);
                        if t.max_new > cap {
                            self.clamped.push((t.turn, cap));
                            t.max_new = cap.max(1);
                        }
                        self.adm.note_started(&self.l, &s, &t);
                        self.actives.push(t);
                    }
                    Verdict::Hold => break,
                    Verdict::Reject { reason, .. } => {
                        let t = self.queued.remove(0);
                        self.adm.note_rejected(t.session, self.now, reason);
                        self.rejected.push((t.turn, reason));
                    }
                }
            }
            if self.actives.is_empty() {
                return false;
            }
            // ② 相位
            let Some(plan) = self.ph.next(&self.l, &self.actives, true) else { return false };
            self.kinds.push(plan.kind);
            // ③ 推进
            let mut advanced: Vec<u64> = Vec::new();
            match plan.kind {
                Kind::Prefill => {
                    if let Some(t) = self.actives.iter_mut().find(|t| t.turn == plan.turn) {
                        t.fed = (t.fed + plan.slice_tokens).min(t.prompt_tokens);
                        if t.fed >= t.prompt_tokens {
                            t.phase = PhaseSz::Gen;
                        }
                        advanced.push(t.turn);
                    }
                }
                Kind::Decode => {
                    if let Some(t) = self.actives.iter_mut().find(|t| t.turn == plan.turn) {
                        t.out_len += 1;
                        advanced.push(t.turn);
                    }
                }
                Kind::BatchDecode => {
                    for t in self.actives.iter_mut() {
                        if t.phase != PhaseSz::Prefill {
                            t.out_len += 1;
                            advanced.push(t.turn);
                        }
                    }
                }
            }
            // ④ 停顿统计（只算 gen 相位：prefill 中的等待是应然）
            let ids: Vec<u64> = self.actives.iter().map(|t| t.turn).collect();
            for id in ids {
                let g = self.gap.entry(id).or_insert(0);
                if advanced.contains(&id) {
                    let v = *g;
                    *g = 0;
                    let m = self.max_gap.entry(id).or_insert(0);
                    *m = (*m).max(v);
                } else if self.actives.iter().any(|t| t.turn == id && t.phase != PhaseSz::Prefill) {
                    *g += 1;
                }
            }
            // ⑤ 终局
            let done: Vec<u64> = self
                .actives
                .iter()
                .filter(|t| t.out_len >= t.max_new)
                .map(|t| t.turn)
                .collect();
            for id in done {
                self.actives.retain(|t| t.turn != id);
                self.adm.note_finished(id);
                self.finished.push(id);
            }
            true
        }

        pub(super) fn run_until_idle(&mut self, max_steps: usize) -> usize {
            let mut n = 0;
            while n < max_steps && self.step() {
                n += 1;
            }
            n
        }
    }

    fn two_slot() -> Limits {
        Limits {
            page: 32,
            max_batch: 2,
            gdn_slots: 2,
            max_concurrency: 2,
            reserve_pages: 8,
            newcomer_min_tokens: 4096,
            min_gen_tokens: 256,
            slice_every: 4,
            slice_tokens: 64,
            ..Default::default()
        }
    }

    fn gen_turn(id: u64, sess: u64, prompt: usize, max_new: usize, out: usize) -> TurnFacts {
        TurnFacts {
            turn: id,
            session: sess,
            prompt_tokens: prompt,
            max_new,
            fed: prompt,
            out_len: out,
            phase: PhaseSz::Gen,
            spec: false,
        }
    }

    pub(super) fn fresh(id: u64, sess: u64, prompt: usize, max_new: usize) -> TurnFacts {
        TurnFacts {
            turn: id,
            session: sess,
            prompt_tokens: prompt,
            max_new,
            fed: 0,
            out_len: 0,
            phase: PhaseSz::Prefill,
            spec: false,
        }
    }

    /// **用户场景**：200K 池 / A 已 150K（142K prompt + 8K 已生成）/ B 全新。
    /// 预期：B 被准入（不拒）、两人都跑完、A 的停顿 ≤ K+1 拍（时间片生效）。
    #[test]
    fn user_case_200k_incumbent_then_fresh_arrival() {
        let mut sim = Sim::new(two_slot(), 6250);
        sim.submit(gen_turn(1, 10, 142_000, 8_064, 8_000));
        assert!(sim.step(), "A 应进活跃");
        sim.submit(fresh(2, 11, 300, 64));
        let steps = sim.run_until_idle(4000);
        assert_eq!(sim.finished.len(), 2, "两人都应跑完（{steps} 拍）");
        assert!(sim.rejected.is_empty(), "B 不该被拒：{:?}", sim.rejected);
        assert!(sim.clamped.is_empty(), "本例 cap 充足，不该夹：{:?}", sim.clamped);
        let gap = *sim.max_gap.get(&1).unwrap_or(&0);
        assert!(gap <= 5, "A 的最大停顿 {gap} 拍应 ≤ slice_every+1");
    }

    /// 在位者吃满池（prompt 199,680 ≈ 6240 页）⇒ 新人**拒**（no_memory），
    /// 等 A 终了后重提 ⇒ 放行。
    #[test]
    fn full_pool_rejects_then_admits_after_incumbent_finishes() {
        let mut sim = Sim::new(two_slot(), 6250);
        sim.submit(gen_turn(1, 10, 199_680, 64, 0));
        assert!(sim.step());
        sim.submit(fresh(2, 11, 300, 64));
        let _ = sim.run_until_idle(4000);
        assert_eq!(sim.rejected.len(), 1, "应拒一次");
        assert_eq!(sim.rejected[0].0, 2);
        assert!(matches!(sim.rejected[0].1, Reason::NoMemory { .. }));
        assert_eq!(sim.rejected[0].1.code(), "no_memory");
        assert_eq!(sim.finished, vec![1]);
        // 客户端按 Retry-After 退避后重提（新 turn id；时间前进越过退避窗）
        sim.now += 500;
        sim.submit(fresh(3, 11, 300, 64));
        let _ = sim.run_until_idle(4000);
        assert!(sim.finished.contains(&3), "A 终了后 B 应能进");
    }

    /// 同时到达：一拍内两人都准入；都进 gen 后走 BatchDecode；两人都完成。
    #[test]
    fn simultaneous_arrivals_share_a_batch() {
        let mut sim = Sim::new(two_slot(), 6250);
        sim.submit(fresh(1, 10, 300, 32));
        sim.submit(fresh(2, 11, 300, 32));
        let steps = sim.run_until_idle(4000);
        assert_eq!(sim.finished.len(), 2, "两人都应完成（{steps} 拍）");
        assert!(
            sim.kinds.iter().any(|k| *k == Kind::BatchDecode),
            "同时到达应出现批 decode（kinds={:?}）",
            sim.kinds.iter().filter(|k| **k == Kind::BatchDecode).count()
        );
    }

    /// GDN 格 = 1（槽位满）⇒ 新人 **Hold**（排队，不是拒）；前者终了后自动放行。
    #[test]
    fn slot_exhaustion_holds_instead_of_rejecting() {
        let mut l = two_slot();
        l.gdn_slots = 1;
        l.max_batch = 1;
        let mut sim = Sim::new(l, 6250);
        sim.submit(gen_turn(1, 10, 1_000, 32, 0));
        assert!(sim.step());
        sim.submit(fresh(2, 11, 300, 32));
        let _ = sim.run_until_idle(4000);
        assert!(sim.rejected.is_empty(), "槽满应 Hold 而不是拒：{:?}", sim.rejected);
        assert_eq!(sim.finished.len(), 2, "前者终了后第二个应被放行");
    }

    /// 超卖档：200K 上下文的两个 turn 在 32K 池上同住（只驻工作集）。
    #[test]
    fn oversub_two_huge_contexts_coexist() {
        let l = Limits {
            page: 32,
            max_batch: 2,
            gdn_slots: 2,
            max_concurrency: 2,
            oversub: true,
            workspace_pages: 32,
            chunk_pages: 32,
            slice_every: 4,
            slice_tokens: 64,
            reserve_pages: 8,
            newcomer_min_tokens: 4096,
            min_gen_tokens: 256,
            max_queue: 0,
        };
        let mut sim = Sim::new(l, 1024); // 32K 池
        sim.submit(fresh(1, 10, 200_000, 32));
        sim.submit(fresh(2, 11, 200_000, 32));
        let _ = sim.run_until_idle(20000);
        assert_eq!(sim.finished.len(), 2, "超卖档两人应都能跑完");
        assert!(sim.rejected.is_empty());
    }
}

// ===========================================================================
// §7 边界用例（纯；每个都是"真机上会遇到的一种情形"）
// ===========================================================================
#[cfg(test)]
mod edge_tests {
    use super::*;

    fn l() -> Limits {
        Limits {
            page: 32,
            max_batch: 2,
            gdn_slots: 2,
            max_concurrency: 2,
            reserve_pages: 8,
            newcomer_min_tokens: 4096,
            min_gen_tokens: 256,
            slice_every: 4,
            slice_tokens: 64,
            ..Default::default()
        }
    }

    fn gen(id: u64, sess: u64, prompt: usize, max_new: usize) -> TurnFacts {
        TurnFacts { turn: id, session: sess, prompt_tokens: prompt, max_new, fed: prompt, out_len: 0, phase: PhaseSz::Gen, spec: false }
    }

    fn pre(id: u64, sess: u64, prompt: usize, fed: usize, max_new: usize) -> TurnFacts {
        TurnFacts { turn: id, session: sess, prompt_tokens: prompt, max_new, fed, out_len: 0, phase: PhaseSz::Prefill, spec: false }
    }

    fn store(free: usize, cache: usize, reserved: usize) -> StoreFacts {
        StoreFacts { free_pages: free, cache_pages: cache, reserved_pages: reserved, host_free_pages: 1 << 20, ..Default::default() }
    }

    // ── 准入：退避 / 会话互斥 / 幂等 ──────────────────────────────

    /// 拒绝后立刻重试 = Hold（退避窗 200ms）；越过窗口 = Admit。
    ///
    /// 注：同会话在跑时的重试会先被 `SessionBusy` 拦住（优先级更高）；
    /// 退避只对"**新会话**曾被临时拒（内存/槽位）"生效。
    #[test]
    fn reject_backoff_is_honored_then_expires() {
        let l = l();
        let s = store(6250, 0, 0);
        let mut a = Admission::new();
        let t = gen(1, 20, 100, 32);
        a.note_rejected(20, 5000, Reason::NoMemory { need_pages: 1, free_pages: 0 });
        assert_eq!(a.decide(&l, &s, &[], &t, 5100), Verdict::Hold, "退避窗内应 Hold");
        assert_eq!(a.decide(&l, &s, &[], &t, 5300), Verdict::Admit, "越过窗口应放行");
    }

    /// 同会话 exclusivity：A 在跑 ⇒ 同会话新 turn = SessionBusy；A 终了后放行。
    #[test]
    fn same_session_is_exclusive_until_finish() {
        let l = l();
        let s = store(6250, 0, 0);
        let mut a = Admission::new();
        let a1 = gen(1, 77, 1000, 64);
        assert_eq!(a.decide(&l, &s, &[], &a1, 0), Verdict::Admit);
        a.note_started(&l, &s, &a1);
        let a2 = gen(2, 77, 500, 64);
        match a.decide(&l, &s, &[a1], &a2, 0) {
            Verdict::Reject { reason: Reason::SessionBusy, retry_after_ms } => assert!(retry_after_ms > 0),
            other => panic!("同会话应 SessionBusy，得到 {other:?}"),
        }
        a.note_finished(1);
        assert_eq!(a.decide(&l, &s, &[], &a2, 10_000), Verdict::Admit, "前者终了后应放行");
    }

    /// note_started / note_finished 幂等（重复调用不许重复记账/下溢）。
    #[test]
    fn ledger_notes_are_idempotent() {
        let l = l();
        let s = store(6250, 0, 0);
        let mut a = Admission::new();
        let t = gen(1, 10, 1000, 64);
        a.note_started(&l, &s, &t);
        a.note_started(&l, &s, &t);
        assert_eq!(a.in_flight(), 1);
        assert_eq!(a.reserved_pages(), t.need_pages(&l, usize::MAX));
        a.note_finished(1);
        a.note_finished(1);
        assert_eq!(a.in_flight(), 0);
        assert_eq!(a.reserved_pages(), 0);
    }

    // ── 准入：容量算式边界 ───────────────────────────────────────

    /// prompt 恰好占满可用池 ⇒ cap 0 ⇒ 拒（连一个 token 都生不出来）。
    #[test]
    fn prompt_that_fills_pool_is_rejected() {
        let l = l();
        let s = store(6250, 0, 0); // usable = 6242 页 = 199,744 tok
        let mut a = Admission::new();
        // prompt = 199,744（= 恰好用满）⇒ cap 0
        let t = TurnFacts { turn: 1, session: 1, prompt_tokens: 199_744, max_new: 64, fed: 0, out_len: 0, phase: PhaseSz::Prefill, spec: false };
        assert_eq!(a.cap_for(&l, &s, &t), 0);
        assert!(matches!(a.decide(&l, &s, &[], &t, 0), Verdict::Reject { reason: Reason::NoMemory { .. }, .. }));
    }

    /// 超卖档：生成预算**不占设备** ⇒ 永不夹（哪怕 200 万 token 的 max_new）。
    #[test]
    fn oversub_never_caps_generation() {
        let mut l = l();
        l.oversub = true;
        l.workspace_pages = 32;
        l.chunk_pages = 32;
        let s = store(4, 0, 0); // 只剩 4 页，但超卖只看工作集
        let t = gen(1, 10, 200_000, 2_000_000);
        let a = Admission::new();
        assert_eq!(a.cap_for(&l, &s, &t), 2_000_000, "超卖不夹生成");
        assert_eq!(a.capacity(&l, &s, &[], &t).n_mem, 0, "但设备侧仍按工作集算不进来");
    }

    /// `n_target` 取三者最小：`max_batch=8, gdn_slots=2` ⇒ 新人预留只按 2 人算。
    #[test]
    fn n_target_is_min_of_batch_slots_and_concurrency() {
        let mut l = l();
        l.max_batch = 8;
        l.gdn_slots = 2;
        l.max_concurrency = 8;
        let s = store(6250, 0, 0);
        let a = Admission::new();
        let t = gen(1, 10, 1000, 1000);
        // usable 199,744；预留 = (n_target−1)×4096 = 4096；cap = 199,744−4,096−1,000
        assert_eq!(a.n_target(&l), 2);
        assert_eq!(a.cap_for(&l, &s, &t), 194_648);
        // gdn_slots = 8 ⇒ 预留 7×4096
        l.gdn_slots = 8;
        assert_eq!(a.n_target(&l), 8);
        assert_eq!(a.cap_for(&l, &s, &t), 199_744 - 7 * 4096 - 1000);
    }

    /// 新人预留会把在位者压到 < min_gen ⇒ **放弃预留**（在位者独占）。
    #[test]
    fn min_gen_floor_drops_the_newcomer_reserve() {
        let l = l();
        let s = store(6250, 0, 0);
        let a = Admission::new();
        // prompt 199,600 ⇒ 剩余 144 < min_gen 256 ⇒ 放弃预留（否则 cap=0）
        let t = gen(1, 10, 199_600, 64);
        assert_eq!(a.cap_for(&l, &s, &t), 144);
        // 剩余 = 4,596 ⇒ 留 4,096 后还剩 500 ≥ 256 ⇒ 保预留
        let t2 = gen(2, 11, 195_148, 64);
        assert_eq!(a.cap_for(&l, &s, &t2), 500);
    }

    /// 缓存页可回收 ⇒ 计入容量；保留页要扣。
    #[test]
    fn cache_counts_but_reserve_is_subtracted() {
        let l = l();
        let a = Admission::new();
        let t = gen(1, 10, 1024, 1024); // 64 页/人
        assert_eq!(a.capacity(&l, &store(100, 0, 0), &[], &t).n_mem, 1, "(100−8)/64 = 1");
        assert_eq!(a.capacity(&l, &store(100, 100, 0), &[], &t).n_mem, 3, "(200−8)/64 = 3");
        assert_eq!(store(100, 0, 0).usable_pages(8), 92);
        assert_eq!(store(100, 0, 0).usable_pages(50), 50);
    }

    /// `n_eff = min(n_slots, n_mem)`。
    #[test]
    fn n_eff_is_min_of_slots_and_memory() {
        let l = l();
        let a = Admission::new();
        let huge = gen(1, 10, 32, 352); // 384 tok = 12 页
        let tiny = gen(2, 11, 32, 352);
        let c = a.capacity(&l, &store(20, 0, 0), &[], &tiny); // usable 12 页 / 12 页
        assert_eq!((c.n_slots, c.n_mem, c.n_eff), (2, 1, 1), "内存成为瓶颈");
        let c = a.capacity(&l, &store(6250, 0, 0), &[], &huge);
        assert_eq!((c.n_slots, c.n_eff), (2, 2), "槽位成为瓶颈");
        assert!(c.n_mem >= 2, "内存侧远超槽位（n_mem={}）", c.n_mem);
    }

    /// 单调性：空闲页只增不减坏容量；ctx 只增不减好容量。
    #[test]
    fn capacity_is_monotone() {
        let l = l();
        let a = Admission::new();
        let t = gen(1, 10, 2048, 2048);
        let mut prev = 0;
        for free in (100..=1000).step_by(50) {
            let n = a.capacity(&l, &store(free, 0, 0), &[], &t).n_mem;
            assert!(n >= prev, "空闲页增加不许让 n_mem 下降（{prev} → {n}）");
            prev = n;
        }
        let mut prev = usize::MAX;
        for prompt in (256..=4000).step_by(256) {
            let t = gen(1, 10, prompt, 2048);
            let n = a.capacity(&l, &store(6250, 0, 0), &[], &t).n_mem;
            assert!(n <= prev, "ctx 增加不许让 n_mem 上升（{prev} → {n}）");
            prev = n;
        }
    }

    // ── 相位：K=0 / K=1 / 独跑 / 公平 / 投机 ─────────────────────

    /// `slice_every = 0` = 旧行为：只要有人欠 prefill 就**一直**给它。
    #[test]
    fn phase_k_zero_is_prefill_first() {
        let mut l = l();
        l.slice_every = 0;
        let mut p = PhaseSched::new();
        let a = gen(1, 10, 100, 64);
        let b = pre(2, 11, 1000, 0, 64);
        for _ in 0..3 {
            match p.next(&l, &[a, b], true).expect("plan") {
                Plan { kind: Kind::Prefill, turn: 2, .. } => {}
                other => panic!("K=0 应一直 prefill，得到 {other:?}"),
            }
        }
    }

    /// `slice_every = 1` ⇒ decode/prefill 严格交替。
    #[test]
    fn phase_k_one_alternates() {
        let mut l = l();
        l.slice_every = 1;
        let mut p = PhaseSched::new();
        let a = gen(1, 10, 100, 64);
        let b = pre(2, 11, 1000, 0, 64);
        let mut kinds = Vec::new();
        for _ in 0..6 {
            kinds.push(p.next(&l, &[a, b], true).expect("plan").kind);
        }
        assert_eq!(
            kinds,
            vec![Kind::Decode, Kind::Prefill, Kind::Decode, Kind::Prefill, Kind::Decode, Kind::Prefill]
        );
    }

    /// 没有 gen 可跑 ⇒ prefill 用**整剩余**（不切片；否则白拉长自己的 TTFT）。
    #[test]
    fn phase_lone_prefill_uses_full_remaining() {
        let mut l = l();
        l.slice_tokens = 64;
        let mut p = PhaseSched::new();
        let b = pre(2, 11, 1000, 100, 64); // 剩 900
        let plan = p.next(&l, &[b], true).expect("plan");
        assert_eq!(plan.kind, Kind::Prefill);
        assert_eq!(plan.slice_tokens, 900, "独跑不许切片");
    }

    /// gen 轮转公平：3 人依次轮转，不连续点同一个。
    #[test]
    fn phase_gen_round_robin_is_fair() {
        let mut l = l();
        l.max_batch = 3;
        l.gdn_slots = 3;
        l.max_concurrency = 3;
        let mut p = PhaseSched::new();
        let acts = [gen(1, 10, 100, 64), gen(2, 11, 100, 64), gen(3, 12, 100, 64)];
        let mut seq = Vec::new();
        for _ in 0..6 {
            seq.push(p.next(&l, &acts, false).expect("plan").turn);
        }
        assert_eq!(seq, vec![1, 2, 3, 1, 2, 3], "round-robin 应轮满一圈再回头");
    }

    /// 投机相位（`Spec`）算 gen：它可以被 decode，不算 prefill 欠账。
    #[test]
    fn phase_spec_is_gen_not_prefill() {
        let mut l = l();
        l.slice_every = 1;
        let mut p = PhaseSched::new();
        let mut spec = gen(1, 10, 100, 64);
        spec.phase = PhaseSz::Spec;
        let b = pre(2, 11, 1000, 0, 64);
        let first = p.next(&l, &[spec, b], true).expect("plan");
        assert_eq!((first.kind, first.turn), (Kind::Decode, 1), "Spec 应能先跑 decode");
    }

    /// prefill 挑选：欠账最多优先；同欠账按 turn id 定序（确定性）。
    #[test]
    fn phase_prefill_picks_least_fed_then_smallest_turn() {
        let mut p = PhaseSched::new();
        let mut l = l();
        l.slice_every = 1;
        let a = pre(5, 10, 1000, 200, 64);
        let b = pre(3, 11, 1000, 50, 64);
        let plan = p.next(&l, &[a, b], true).expect("plan");
        assert_eq!(plan.turn, 3, "fed 最小者先");
        let c = pre(7, 12, 1000, 50, 64);
        let mut p2 = PhaseSched::new();
        let plan = p2.next(&l, &[b, c], true).expect("plan");
        assert_eq!(plan.turn, 3, "同 fed ⇒ turn id 小者先");
    }

    // ── 时间线：三人合批 / 夹紧终局 ───────────────────────────────

    /// 三人同时到达（三个槽）⇒ 都进、出现批 decode、都完成。
    #[test]
    fn three_simultaneous_arrivals_all_finish() {
        use super::scenario_tests::Sim; // 复用时间线模拟器
        let mut l = l();
        l.max_batch = 3;
        l.gdn_slots = 3;
        l.max_concurrency = 3;
        let mut sim = Sim::new(l, 6250);
        for i in 0..3 {
            sim.submit(super::scenario_tests::fresh(i + 1, 10 + i, 300, 24));
        }
        let _ = sim.run_until_idle(5000);
        assert_eq!(sim.finished.len(), 3, "三人都应完成");
        assert!(sim.kinds.iter().any(|k| *k == Kind::BatchDecode), "应出现批 decode");
        assert!(sim.rejected.is_empty());
    }
}

// ===========================================================================
// §8 更多边界用例（预算夹紧 / 缓存回收 / 不扰动 / 超卖容量 / 事件序）
// ===========================================================================
#[cfg(test)]
mod more_tests {
    use super::*;

    fn l() -> Limits {
        Limits {
            page: 32,
            max_batch: 2,
            gdn_slots: 2,
            max_concurrency: 2,
            reserve_pages: 8,
            newcomer_min_tokens: 4096,
            min_gen_tokens: 256,
            slice_every: 4,
            slice_tokens: 64,
            ..Default::default()
        }
    }

    fn gen(id: u64, sess: u64, prompt: usize, max_new: usize) -> TurnFacts {
        TurnFacts { turn: id, session: sess, prompt_tokens: prompt, max_new, fed: prompt, out_len: 0, phase: PhaseSz::Gen, spec: false }
    }

    fn store(free: usize, cache: usize, reserved: usize) -> StoreFacts {
        StoreFacts { free_pages: free, cache_pages: cache, reserved_pages: reserved, host_free_pages: 1 << 20, ..Default::default() }
    }

    /// **A3**：判决（拒绝新人）不许改在跑者的账 —— `admitted`/预留页一字不动。
    #[test]
    fn admission_does_not_disturb_the_running() {
        let l = l();
        let s = store(6250, 0, 0);
        let mut a = Admission::new();
        let running = gen(1, 10, 199_000, 64);
        a.note_started(&l, &s, &running);
        let before = (a.in_flight(), a.reserved_pages());
        let used = a.reserved_pages();
        let s2 = store(6250 - used, 0, used);
        let newcomer = gen(2, 11, 1000, 64);
        let v = a.decide(&l, &s2, &[running], &newcomer, 0);
        assert!(matches!(v, Verdict::Reject { reason: Reason::NoMemory { .. }, .. }), "{v:?}");
        assert_eq!((a.in_flight(), a.reserved_pages()), before, "在跑者的账不许被拒新扰动");
    }

    /// **预算夹紧真的进预留**：夹紧后的 `need_pages` 必须小于未夹紧（否则会撞墙）。
    #[test]
    fn clamped_budget_is_what_gets_reserved() {
        let l = l();
        let s = store(6250, 0, 0);
        let a = Admission::new();
        let t = gen(1, 10, 1000, 300_000);
        let cap = a.cap_for(&l, &s, &t);
        assert!(cap < 300_000, "应被夹（cap={cap}）");
        let clamped = t.need_pages(&l, cap);
        let uncapped = t.need_pages(&l, usize::MAX);
        assert!(clamped < uncapped, "夹紧后预留必须变小（{clamped} vs {uncapped}）");
        assert_eq!(clamped, (1000 + cap).div_ceil(32), "预留 = prompt + min(max_new, cap)");
    }

    /// **缓存页可回收**：`free=0` 但 `cache>0` 时仍能进人（先逐缓存，不逐活跃）。
    #[test]
    fn cache_only_pool_can_still_admit() {
        let l = l();
        let a = Admission::new();
        let t = gen(1, 10, 1024, 1024); // 64 页
        let c = a.capacity(&l, &store(0, 100, 0), &[], &t);
        assert_eq!(c.usable_pages, 92, "usable = cache − reserve");
        assert_eq!(c.n_mem, 1);
        // 缓存太少 ⇒ 进不来（不像"free=0 就必拒"，缓存是真容量）
        let c = a.capacity(&l, &store(0, 40, 0), &[], &t);
        assert_eq!(c.n_mem, 0);
    }

    /// `usable_pages` 饱和减：reserve 大于 free 不许下溢。
    #[test]
    fn usable_saturates_instead_of_underflow() {
        assert_eq!(store(4, 0, 0).usable_pages(8), 0);
        assert_eq!(store(0, 0, 0).usable_pages(0), 0);
        assert_eq!(store(10, 10, 0).usable_pages(8), 12);
    }

    /// **超卖容量**：同一请求形状下，容量只由工作集决定，与 ctx 无关（已有单测），
    /// 这里补"多人已占 ⇒ 余额按人扣"的算式：15 人额度里已占 14 人 ⇒ 还能进 1。
    #[test]
    fn oversub_remaining_capacity_is_per_workspace() {
        let mut l = l();
        l.oversub = true;
        l.workspace_pages = 32;
        l.chunk_pages = 32;
        l.max_batch = 16;
        l.gdn_slots = 16;
        l.max_concurrency = 16;
        let a = Admission::new();
        let t = gen(1, 10, 10_000, 64);
        // 1024 页池：usable 1016；每人对 64 页 ⇒ 15
        assert_eq!(a.capacity(&l, &store(1024, 0, 0), &[], &t).n_mem, 15);
        // 已预留 14 人 = 896 页 ⇒ 余额 120 页 ⇒ 还能进 1
        let c = a.capacity(&l, &store(1024 - 896, 0, 896), &[], &t);
        assert_eq!(c.n_mem, ((1024 - 896) - 8) / 64);
        assert_eq!(c.n_mem, 1);
    }

    /// **事件序**：准入拒绝走 `pending_events`，且同一拍里 `Failed` 只出现一次。
    /// （引擎侧的接入在 `scheduler.rs`；这里用模拟器验语义：拒 → 记录 → 不再重复。）
    #[test]
    fn rejected_turn_is_recorded_once_and_not_retried() {
        use super::scenario_tests::{fresh, Sim};
        let mut sim = Sim::new(l(), 6250);
        sim.submit(gen(1, 10, 199_000, 64));
        assert!(sim.step(), "在位者应进活跃");
        sim.submit(fresh(2, 11, 1500, 64));
        // 同一拍内反复判：只应拒一次（出队后不再重判）
        let _ = sim.step();
        let _ = sim.step();
        assert_eq!(sim.rejected.len(), 1, "同一 turn 只许被拒一次：{:?}", sim.rejected);
        assert_eq!(sim.rejected[0].0, 2);
    }
}

// ===========================================================================
// §9 P0 验收（队列上限 / 超卖宿主腿）
// ===========================================================================
#[cfg(test)]
mod p0_tests {
    use super::*;

    fn l() -> Limits {
        Limits {
            page: 32,
            max_batch: 1,
            gdn_slots: 1,
            max_concurrency: 1,
            reserve_pages: 8,
            newcomer_min_tokens: 4096,
            min_gen_tokens: 256,
            max_queue: 2,
            slice_every: 0,
            slice_tokens: 0,
            ..Default::default()
        }
    }

    fn gen(id: u64, sess: u64, prompt: usize, max_new: usize) -> TurnFacts {
        TurnFacts { turn: id, session: sess, prompt_tokens: prompt, max_new, fed: prompt, out_len: 0, phase: PhaseSz::Gen, spec: false }
    }

    fn store(free: usize) -> StoreFacts {
        StoreFacts { free_pages: free, ..Default::default() }
    }

    /// **P0-3 超卖宿主腿**：设备装得下、宿主装不下 ⇒ 拒（不许"进来再丢块"）。
    #[test]
    fn oversub_host_leg_gates_admission() {
        let mut l = l();
        l.oversub = true;
        l.workspace_pages = 32;
        l.chunk_pages = 32;
        l.max_batch = 2;
        l.gdn_slots = 2;
        l.max_concurrency = 2;
        let mut a = Admission::new();
        let t = gen(1, 10, 200_000, 64);
        let off = t.offload_pages(&l);
        assert!(off > 6000, "20 万 ctx 应要沉 {off} 页");
        // 宿主够 ⇒ 放行
        let ok = StoreFacts { free_pages: 1024, host_free_pages: off + 10, ..Default::default() };
        assert_eq!(a.decide(&l, &ok, &[], &t, 0), Verdict::Admit);
        // 宿主不够 ⇒ 拒（NoMemory 报的是**宿主页**收支）
        let tight = StoreFacts { free_pages: 1024, host_free_pages: off / 2, ..Default::default() };
        match a.decide(&l, &tight, &[], &t, 0) {
            Verdict::Reject { reason: Reason::NoMemory { need_pages, free_pages }, .. } => {
                assert_eq!(need_pages, off);
                assert_eq!(free_pages, off / 2);
            }
            other => panic!("宿主不够应拒，得到 {other:?}"),
        }
        // 容量侧也按宿主截断
        let c = a.capacity(&l, &tight, &[], &t);
        assert!(c.n_mem <= (off / 2) / off.max(1) + 1);
    }

    /// **P2 磁盘腿**：宿主满时磁盘空位顶上（`host + disk` 是一条“能落哪儿”的账）。
    /// 旧行为 `disk_free_pages: 0` ⇒ 配了盘也拒新。
    #[test]
    fn oversub_disk_leg_counts_when_host_is_full() {
        let mut l = l();
        l.oversub = true;
        l.workspace_pages = 32;
        l.chunk_pages = 32;
        l.max_batch = 2;
        l.gdn_slots = 2;
        l.max_concurrency = 2;
        let mut a = Admission::new();
        let t = gen(1, 10, 200_000, 64);
        let off = t.offload_pages(&l);
        assert!(off > 6000, "20 万 ctx 应要沉 {off} 页");
        let s = StoreFacts {
            free_pages: 1024,
            host_free_pages: 0,
            disk_free_pages: off + 10,
            ..Default::default()
        };
        assert_eq!(a.decide(&l, &s, &[], &t, 0), Verdict::Admit, "宿主满但盘够 ⇒ 放行");
        assert!(a.capacity(&l, &s, &[], &t).n_mem >= 1, "容量侧同样把盘算进去");
    }

    /// 非超卖档不看宿主（全量驻留设备）—— 回归：别把 host 门误开。    #[test]
    fn non_oversub_ignores_host_tier() {
        let l = l();
        let s = StoreFacts { free_pages: 6250, host_free_pages: 0, ..Default::default() };
        let mut a = Admission::new();
        assert_eq!(a.decide(&l, &s, &[], &gen(1, 10, 1000, 64), 0), Verdict::Admit);
    }
}
