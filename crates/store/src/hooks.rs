//! 引擎钩子框架(§14)—— store 的插入语义:**"给你一个需求,谁该离开、谁该进来?"**
//!
//! ```text
//! 相一 plan(need, protect) -> Verdict   纯 host、零 I/O(任意时刻可调,含捕获窗内)
//! 相二 apply(face, &Verdict)            段边界、async、显式收 face(搬运归 server 执行)
//! ```
//!
//! 次序律(§12.3,唯一正确顺序):
//! **① 先下沉**(宿主有位就搬,内容不丢)→ **② 再驱逐**(宿主满,真丢)→ **③ 仍不足 ⇒ 拒**
//! (绝不动活跃面;抢占 = 另一个功能,S2b 的状态腿配合后再说)。
//!
//! 两相之间状态可变 ⇒ `apply` **重验**:引用变了(`Busy`)只**跳过该条**,不是错误。


use crate::id::ChainHash;
use crate::leg::BlockLeg;

/// 钩子需求(§12 需求侧的落点):**要几个设备槽,为什么**。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Need {
    /// 需要的**设备槽数**(周转量;不是总量)。
    pub slots: usize,
    /// 需求来源(打点/排障口径;H1 准入 / H2 增长 / H5 空转整理)。
    pub why: NeedWhy,
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
    /// `slots` 读作"**目标腾出的槽数**"(不是缺口)。与其它节点的区别只有一条:
    /// 不吃 `free >= slots` 早退(压力下"手上有货"也要整理)。
    Pressure,
}

/// 一块该去哪儿。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sink {
    /// 下沉到宿主层(内容留)。
    Host,
    /// 驱逐(宿主满;内容丢,下次重算)。
    Drop,
}

/// "谁该离开"的一条。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvictOut {
    /// 块身份(链级条目 = **链尾**)。
    pub chain: ChainHash,
    /// 去哪。
    pub sink: Sink,
    /// 覆盖面:`span = 1` = 单块;`span > 1` = **整条链**(§14.6 链级卸载:
    /// 从链尾向链头搬完,`chain` 是链尾;值 = 链上设备块数)。
    pub span: usize,
}

/// **决策单**(§14.2):谁走 / 谁进 / 拒。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Verdict {
    /// 谁该离开(按序执行;`Busy` 的在 `apply` 里跳过)。
    pub out: Vec<EvictOut>,
    /// 谁该进来(需物化的回灌清单;H4 用)。
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
    /// 实际下沉成功数。
    pub offloaded: usize,
    /// 实际驱逐数。
    pub dropped: usize,
    /// 实际回灌数。
    pub onloaded: usize,
    /// 因活跃面(`Busy`)被跳过的条数。
    pub skipped_busy: usize,
}

/// **相一:判**(纯 host、零 I/O)。给需求 ⇒ 谁走、谁进、还是拒。
///
/// 次序律(§12.3 + §14.6):
/// 1. **先下沉,粒度 = 链**(§14.6):按链尾 LRU 出**整条可下沉的链**——
///    因为前缀匹配从链头走,只沉链尾的块对复用零贡献;
/// 2. 链不够(没有整条可沉的链)⇒ **退化按叶**(同律附则:打点区分);
/// 3. 宿主位用尽 ⇒ **驱逐**(丢内容,仍按叶;丢尾部不动链头,前缀仍可复用);
/// 4. 仍不足 ⇒ **拒**(响亮,不许动活跃面)。
///
/// `protect` = 活跃 turn 的链 + 显式点名(§2.4 保留额不变量;这些块**绝不**入 `out`)。
pub fn plan(leg: &BlockLeg, need: Need, protect: &[ChainHash]) -> Verdict {
    let mut v = Verdict::default();
    let free = leg.slots_free();
    // **N6 压力 / H5 空转整理**:目标 = "腾出 slots 个槽"(与缺口无关)⇒ 不吃早退。
    // (两者只差 `why` 的打点口径:Pressure = 救火,H5 = 空转期主动整理)
    if matches!(need.why, NeedWhy::Pressure | NeedWhy::Idle) {
        let want = need.slots.min(leg.slots_total().saturating_sub(free));
        let mut got = 0usize;
        let mut picked: Vec<ChainHash> = protect.to_vec();
        // ① 先下沉(内容不丢):整链 → 退化按叶
        let host_room = leg.host_free_slots().min(want);
        if host_room > 0 {
            for c in leg.chain_offload_candidates(&picked, host_room) {
                v.out.push(EvictOut { chain: c.tail, sink: Sink::Host, span: c.dev_blocks });
                picked.push(c.tail);
                got += c.dev_blocks;
                if got >= host_room {
                    break;
                }
            }
            if got < host_room {
                for chain in leg.offload_candidates(&picked, host_room - got) {
                    v.out.push(EvictOut { chain, sink: Sink::Host, span: 1 });
                    picked.push(chain);
                    got += 1;
                }
            }
        }
        // ② 仍要腾(宿主满)⇒ 驱逐(丢内容,按叶)
        if got < want {
            for chain in leg.offload_candidates(&picked, want - got) {
                v.out.push(EvictOut { chain, sink: Sink::Drop, span: 1 });
                picked.push(chain);
                got += 1;
            }
        }
        if got < want {
            v.deny = Some(format!(
                "{} 压力:要腾 {} 槽,可搬面只凑出 {} 槽(短 {});宿主预算 {} 字节",
                leg.label(),
                want,
                got,
                want - got,
                leg.host_budget_bytes()
            ));
        }
        return v;
    }
    if free >= need.slots {
        return v; // 手上有货,不动任何人
    }
    let short = need.slots - free;
    // 本趟已点名/已入单的块(去重:链级与叶级候选可能重叠,后续调用传 `&picked`)
    let mut picked: Vec<ChainHash> = protect.to_vec();
    let host_room = leg.host_free_slots();
    let mut got = 0usize;

    // ① 链级下沉(§14.6):宿主有位就搬,内容不丢
    if host_room > 0 {
        let want = short.min(host_room);
        for c in leg.chain_offload_candidates(&picked, want) {
            v.out.push(EvictOut { chain: c.tail, sink: Sink::Host, span: c.dev_blocks });
            picked.push(c.tail);
            got += c.dev_blocks;
            if got >= want {
                break;
            }
        }
        // ② 退化按叶(没有可整条沉的链;宿主还有位 ⇒ 仍是下沉)
        if got < want {
            for chain in leg.offload_candidates(&picked, want - got) {
                v.out.push(EvictOut { chain, sink: Sink::Host, span: 1 });
                picked.push(chain);
                got += 1;
            }
        }
    }
    // ③ 仍不足 ⇒ 驱逐(宿主满,真丢内容;按叶:丢尾部不动链头)
    if got < short {
        for chain in leg.offload_candidates(&picked, short - got) {
            v.out.push(EvictOut { chain, sink: Sink::Drop, span: 1 });
            picked.push(chain);
            got += 1;
        }
    }
    // ④ 仍不足 ⇒ 拒(不许动活跃面)
    if got < short {
        v.deny = Some(format!(
            "{} 链:需求 {} 槽,自由 {};可下沉/驱逐面只凑出 {} 槽(短 {});\
             宿主预算 {} 字节 ⇒ 拒(降并发/上下文,或加宿主预算)",
            leg.label(),
            need.slots,
            free,
            got,
            short - got,
            leg.host_budget_bytes()
        ));
    }
    v
}


// **N6-① 次序门已收口**(2026-10-19):`tests/store_e2e.rs::pressure_offloads_without_deficit_and_prefers_host`
// —— 真装 2 条链(池满)后断言:①`Grow` + 零缺口 = noop ②`Pressure` 目标 3 槽必出单
// ③出单全部 `Sink::Host`(内容不丢)④span 合计 = 目标。
