//! 执行器(设备侧照办;2026-10-01 拆分自 engine.rs §3;2026-10-10 再拆
//! 六模块:phases 调度三相位/spec 轮机器/draft 草稿器面/prefill_chunk
//! 块式预填/turn_end 终局)。S0 决策/执行分离的**执行半**:调度面
//! (scheduler.rs)产出 SchedulerOutput,本模块无决策权。
//! 状态块触碰一律经 StatePool(state.rs),不直摸块句柄。

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_models::interpreters::eval_ops_scoped_env;
use owl_models::layers::gdn::GdnBuffers;
use owl_models::module::{FiPrefillCtx, ForwardCtx, KvBuffers};
use owl_models::tokenizer::Tokenizer;
use owl_models::TensorOps;

use crate::graph_plan::f32b;
use crate::scheduler::BeginPlan;

/// **B4 降级探测周期上限**(tok):退避 ×2 后封顶,避免探针稀疏到失效。
/// **降级态探测间隔上限**(2026-10-27 由 512 下修到 64)。
///
/// 原值 512 是"指数退避,别在真亏本时白试"的思路;但实测(48k 真实文档):
/// 一次 `m=0×7` 连击降级后,`probe_every` 从 8 逐轮翻倍到 512 —— **而请求
/// 总共才 400 token**,于是**一次探测都不会发生** ⇒ 整段永久裸 decode
/// (346/400 token,28 ms/token vs 投机 12 ms/token)。
/// 封顶 64 后最坏情况 = 每 64 token 试一轮投机:多付 ≈ (34−28×AL)/64
/// ≈ 0.1 ms/token,却保证"内容一旦回到可投机域就自动恢复"。
/// **2026-11-03 再下修到 32**:探测本身是**保险不是惩罚** —— 一次探测轮 ≈ 一轮
/// 投机(33 ms),摊到 32 token 上 ≈ 1 ms/token(5%);而"该恢复却还在裸跑"的
/// 代价是 **2.1×**(健康档 9.8 ms vs 裸 21 ms)。真机实测降级期 `deg` 占到
/// 解码步的 50%,恢复延迟直接吃吞吐 ⇒ 宁可多探。
const DEGRADE_PROBE_MAX: usize = 32;

/// **降级确认窗口数**(2026-11-03 用户裁决:不许"一个窗口就降级")。
///
/// 降级需要 **2 个连续**"低收益窗口"(16 轮或 600 ms 一窗)。理由是不对称性:
///
/// - 误降(健康档被降):9.8 ms/token → 21 ms/token = **2.1×**;
/// - 误留(死档没降):26 ms/轮 vs 21 ms/token = **1.24×**。
///
/// ⇒ 宁可多等一窗(≈1.2 s),也不要误降。连击(m=0×N)只能**减半**确认
/// (1 窗 + 连击佐证),不再单独构成降级依据。
pub(crate) const DEGRADE_CONFIRM_WINDOWS: u8 = 2;

/// **深饿阈值**：本窗 `AL < 盈亏线 × 本值` ⇒ **一窗确认**（浅饿仍要两窗）。
/// 取 0.95 = "低到噪声之外"（16 轮窗口的 AL 噪声 ≈ ±0.1，线 ≈1.5 ⇒ 3%，
/// 0.95 留一倍余量）：低过线本身就已经在亏，不必再等一窗。
///
/// 依据（2026-11-03 真机，双 3080 / 27B）：创作类负载 AL ≈ 1.36、盈亏线
/// ≈ 1.60（轮 28.3 / 裸 17.7）⇒ 比 0.85 —— 这不是"偶发一段饿"，而是
/// **整条负载都亏**。等第二个窗口 = 白亏 1.2 s；一窗（600 ms）足够定罪。
/// 健康档（数学 AL 3.9 / 复制 7.9）要到 `AL < 0.9×线 ≈ 1.44` 才会走快路，
/// 实测不触发。
pub(crate) const DEGRADE_DEEP_RATIO: f32 = 0.95;

/// **恢复需要的连续探测命中数**（2026-11-03 第二版）。
///
/// 旧实现 = 一次探测 `m ≥ 1` 就恢复。真机 3 条创作请求（363 轮）实测
/// `spec.degrade = 4 / spec.undegrad = 3` —— 12 s 的请求里降级↔恢复抖了 7 次，
/// 而该负载 `P(m ≥ 1) ≈ 0.29` ⇒ 单次命中不能证明"内容回到可投机域"。
/// ⇒ 恢复改为**连续两次命中**（`≈0.08`）**或**一次决定性命中。
///
/// 健康档不受损：数学/复制 `P(m ≥ 1) ≈ 1`，第一探就命中 ⇒ 周期复位到 8，
/// 第二次探测 8 token 后到达，恢复延迟 ≈ 8 token（≈0.14 s）。
pub(crate) const RESTORE_HITS: u8 = 2;

/// **决定性命中**：单次探测 `m ≥ 本值` ⇒ 立即恢复（一轮吃满 = 确定可投机）。
pub(crate) const RESTORE_DECISIVE_M: usize = 3;

/// 判据窗口：满 `ROUNDS` 轮，或 满 `MS` 毫秒**且**至少 `MIN_ROUNDS` 轮。
///
/// `MIN_ROUNDS` 的来历（2026-11-03 真机）：turn 的**第一轮**常是冷轮
/// （eager seed propose / 首次图发射），可以花几百毫秒；时间门 `ms ≥ 600`
/// 于是让它**一轮成窗**，AL=1.0（恰好 m=0）< 深饿线 ⇒ 误降级（实测复制档
/// 首个请求 204 tok/s vs 278）。窗口至少要有 8 轮（≈16 个样本 token）才配定罪。
pub(crate) const WINDOW_ROUNDS: u32 = 16;
pub(crate) const WINDOW_MIN_ROUNDS: u32 = 8;
pub(crate) const WINDOW_MS: f32 = 600.0;

/// 窗口是否评估（纯函数；见 [`WINDOW_ROUNDS`] 族常量）。
pub(crate) fn spec_window_done(rounds: u32, ms: f32) -> bool {
    rounds >= WINDOW_ROUNDS || (ms >= WINDOW_MS && rounds >= WINDOW_MIN_ROUNDS)
}

/// **spec 态降级判决**（纯函数；按**窗口经济性**判，不按单轮命中）。
///
/// `window_done` = 本窗刚评估完；`window_al` = 本窗 tokens/轮；
/// `line` = 盈亏线（`None` = 分母未测 ⇒ 不判，由播种步补分母）；
/// `low0` = 进入本窗时已累计的连续低收益窗口数。
///
/// 返回 `(是否降级, 更新后的连续低窗数)`：
/// 盈利窗 ⇒ 清零；浅饿 ⇒ `+1`（两窗确认）；深饿（[`DEGRADE_DEEP_RATIO`]）
/// ⇒ `+2`（一窗即降）。
pub(crate) fn spec_degrade_verdict(
    window_done: bool,
    window_al: f32,
    line: Option<f32>,
    low0: u8,
) -> (bool, u8) {
    let (Some(line), true) = (line, window_done) else {
        return (false, low0);
    };
    if !(window_al.is_finite() && window_al >= line) {
        let step = if window_al < line * DEGRADE_DEEP_RATIO {
            2
        } else {
            1
        };
        let low = low0.saturating_add(step);
        return (low >= DEGRADE_CONFIRM_WINDOWS, low);
    }
    (false, 0)
}

/// **降级态恢复判决**（纯函数；连续命中制，见 [`RESTORE_HITS`]）。
/// 返回 `(是否恢复, 更新后的连续命中数)`。
pub(crate) fn spec_restore_verdict(m: usize, hits0: u8) -> (bool, u8) {
    if m >= RESTORE_DECISIVE_M {
        return (true, 0);
    }
    if m >= 1 {
        let hits = hits0.saturating_add(1);
        return (
            hits >= RESTORE_HITS,
            if hits >= RESTORE_HITS { 0 } else { hits },
        );
    }
    (false, 0)
}

/// **连击账 + 探测节奏**（纯函数；原 `spec_degrade_transition` 的漂移量部分）。
///
/// 返回 `(streak, 下一个探测周期)`：`m ≥ 1` ⇒ 连击清零、周期复位；
/// 降级态探测失败 ⇒ 周期 `×2`（封顶 [`DEGRADE_PROBE_MAX`]）；
/// 非降级态 ⇒ 周期不动。
pub(crate) fn spec_streak_step(
    streak: usize,
    degraded: bool,
    probe_every: usize,
    m: usize,
    floor: usize,
) -> (usize, usize) {
    if m >= 1 {
        return (0, floor);
    }
    let streak = streak + 1;
    if degraded {
        (streak, (probe_every * 2).clamp(1, DEGRADE_PROBE_MAX))
    } else {
        (streak, probe_every)
    }
}

/// **双 rank 镜像执行**(S6f):`dev` 必跑;`peer` 在场则**并发**跑同一动作。
///
/// 为什么是宏而不是对象:两边的 future 都**借用各自 rank 的 `&mut`** ——
/// 用闭包传会撞上 `for<'a> FnOnce(&'a mut RankDev) -> Future<'a>` 的高阶
/// 生命周期(只能靠 Box 表达,热路径不划算)。宏展开保持零开销,同时把
/// **纪律**收进一处:两半**都**校验(`join2` 等另一半到底,吞错 = 挂住)、
/// 错误先报 peer、单卡直跑零分支。
///
/// 用法(先拆借再调用,避免 `&mut self` 双借):
/// ```ignore
/// let (dev, peer) = (&mut self.dev, self.peer.as_mut());
/// mirror2!(dev, peer,
///     d => d.pool.write_bt(d.graphs.session.face_mut(), &table),
///     p => p.pool.write_bt(p.graphs.session.face_mut(), &table))
/// ```
macro_rules! mirror2 {
    ($dev:expr, $peer:expr, $d:ident => $fd:expr, $p:ident => $fp:expr) => {{
        match $peer {
            None => {
                let $d = $dev;
                $fd.await
            }
            Some($p) => {
                let $d = $dev;
                let (r0, r1) = $crate::exec::join2($fd, $fp).await;
                r1?;
                r0
            }
        }
    }};
}

pub(crate) use mirror2;

/// **U32 位型走私**(引擎内唯一入口):把 u32 账以 f32 承载过线。
///
/// 池/表的数值面恒 f32(`bt`/`xqa_pt`/`fi_*` 同约定),页号/长度这些
/// u32 只能**位型**过线;`from_bits` 与 `to_le_bytes` 两种编码都在这里,
/// 不再各文件各写一个闭包。
pub(crate) fn u32s_as_f32(v: &[u32]) -> Vec<f32> {
    v.iter().map(|x| f32::from_bits(*x)).collect()
}

/// **手工 join 两 future**(S4 ⑤-K3):引擎 lib **不依赖 async 运行时/组合器**
/// (只有 owl-* + half)—— 两 rank 的设备半必须**并发推进**(见
/// `phases.rs` 的 TP 驱动注:单边先行会让对端的 AR 撞 5s 自旋超时)。
/// 语义 = `futures::join!`:两 future 都推进到完成,返回各自结果。
pub(crate) async fn join2<A, B>(a: A, b: B) -> (A::Output, B::Output)
where
    A: std::future::Future,
    B: std::future::Future,
{
    use std::task::Poll;
    let (mut a, mut b) = (Box::pin(a), Box::pin(b));
    let (mut ra, mut rb) = (None, None);
    std::future::poll_fn(|cx| {
        if ra.is_none() {
            if let Poll::Ready(v) = a.as_mut().poll(cx) {
                ra = Some(v);
            }
        }
        if rb.is_none() {
            if let Poll::Ready(v) = b.as_mut().poll(cx) {
                rb = Some(v);
            }
        }
        if ra.is_some() && rb.is_some() {
            Poll::Ready(())
        } else {
            Poll::Pending
        }
    })
    .await;
    (ra.expect("join2 a"), rb.expect("join2 b"))
}

/// **候选合并**(分布式采样,2026-10-26):两 rank 各 32 个 `(val, idx)`
/// (半表域)→ 每行全局索引升序的并集(≤64)。
///
/// 为什么按**索引升序**:[`crate::sampler::sample_from_candidates`] 复用
/// `distribution` 的扫描序(index 升序 + 严格 `>`)才与全表路径逐位等价。
/// **单卡候选表解析**(2026-10-31):`[2·rows·K]` 扁平 → 每行 `(值, 全局索引)`
/// 按索引升序(与 [`merge_sample_cands`] 的产物**同一形态**,下游无需分支)。
///
/// 布局由 `topk32_of_logits` 定义:前 `rows·K` 是值、后 `rows·K` 是索引(as f32);
/// 单卡时 `vocab_offset = 0` ⇒ 索引已全局,不需再加偏移。
pub(crate) fn parse_local_cands(a: &[f32]) -> Vec<Vec<(f32, u32)>> {
    const K: usize = 32;
    let rows = a.len() / (2 * K);
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut u: Vec<(f32, u32)> = Vec::with_capacity(K);
        for k in 0..K {
            u.push((a[r * K + k], a[rows * K + r * K + k] as u32));
        }
        u.sort_unstable_by_key(|&(_, i)| i);
        out.push(u);
    }
    out
}

pub(crate) fn merge_sample_cands(
    a: &[f32],
    b: &[f32],
    rows: usize,
    half_vocab: usize,
) -> Result<Vec<Vec<(f32, u32)>>> {
    const K: usize = 32;
    let need = 2 * rows * K;
    if a.len() < need || b.len() < need {
        return Err(ModelError::EngineInvariant {
            what: format!(
                "分布式采样候选尺寸不符:a={} b={} 需 {}",
                a.len(),
                b.len(),
                need
            ),
        });
    }
    let mut out = Vec::with_capacity(rows);
    for r in 0..rows {
        let mut u: Vec<(f32, u32)> = Vec::with_capacity(2 * K);
        for k in 0..K {
            u.push((a[r * K + k], a[rows * K + r * K + k] as u32));
            u.push((
                b[r * K + k],
                b[rows * K + r * K + k] as u32 + half_vocab as u32,
            ));
        }
        u.sort_unstable_by_key(|&(_, i)| i);
        out.push(u);
    }
    Ok(out)
}

pub mod phases;

pub mod coverage;
pub mod draft;
/// **E4 接入面(声明;未实现)**——见 `docs/arch/三流事件序-设计与迁移.md` §6
#[allow(dead_code)] // E4 声明面(未接线;见模块头:全部 Err(未实现),调用即响亮拒 ⇒ 不静默)
pub mod e4_api;
/// 前缀无感删除:设备半(后缀 K 原地重转)
pub mod img_evict;
/// **KV 池页的宿主解码面**(单源;f16 / fp8 e4m3 两侧)
pub mod kvpage;
/// **DFlash2 lookup-augmented drafting**(LABD;宿主后缀查找 + 草稿头融合)
pub mod lookup;
/// **MTP 草稿注意力读视图**(Windowed-MTP;纯函数)
pub mod mtp_ring;
pub mod mtp_window;
pub mod prefill_chunk;
pub mod proposal_q;
pub mod spec;
pub mod turn_end;

#[cfg(test)]
mod break_even_tests {
    use super::*;

    /// **盈亏线口径门**(2026-10-17):真值用 32k 实测样本钉住 ——
    /// 轮 77 ms / 裸 38.4 ms ⇒ 线 **2.0**(下限),而 AL 2.25 应判“盈利”。
    /// 旧行为(未测时用兑底常量 2.5)会把它误判成亏本 ⇒ 本门同时钉住
    /// “分母未测 ⇒ None(窗口判据不生效)”。
    #[test]
    fn break_even_line_pins_32k_sample() {
        let line = break_even_line(77.0, 38.4).expect("两分母都有 ⇒ 必有线");
        assert!(
            (line - 77.0 / 38.4).abs() < 1e-5,
            "区间内原样取比,实得 {line}"
        );
        assert!(
            (line - 2.005).abs() < 1e-3,
            "32k 实测样本 ≈ 2.0(不是旧兑底 2.5)"
        );
        assert!(
            2.25 >= line,
            "AL 2.25 ≥ 真线 {line:.3} ⇒ 应判**盈利**;旧兑底 2.5 会把它误降级"
        );
        // 2.9k 生产档样本(§Q):轮 41.79 / 裸 22.48 ⇒ 1.86(**不再被下限钳**;旧下限 2.0)
        assert!((break_even_line(41.79, 22.48).unwrap() - 41.79 / 22.48).abs() < 1e-6);
        // 8.5k 生产档样本(2026-10-18 propose 立案同批):轮 36.7 / 裸 27.5 ⇒ 1.33 → 下限 1.35
        assert!((break_even_line(36.7, 27.5).unwrap() - 1.35).abs() < 1e-6);
        // 上限夹取(超长轮/极端比)
        assert_eq!(break_even_line(600.0, 20.0), Some(6.0));
        // 中间值原样
        assert_eq!(break_even_line(120.0, 40.0), Some(3.0));
    }

    /// 分母未测 ⇒ None(而不是一个会误导判据的常量);
    /// 零/负输入同样归为“未测”(宁可不算,不可乱算)
    #[test]
    fn break_even_line_is_none_when_denominator_unmeasured() {
        assert_eq!(break_even_line(77.0, 0.0), None);
        assert_eq!(break_even_line(0.0, 38.4), None);
        assert_eq!(break_even_line(0.0, 0.0), None);
        assert_eq!(break_even_line(-1.0, 38.4), None);
        assert_eq!(break_even_text(None), "未测");
        assert_eq!(break_even_text(Some(2.0)), "2.00");
    }
}

use crate::turn::TurnEvent;

/// **盈亏线夹取区间**(2026-10-09 立;下限 = 32k 实测比,上限 = 防超长轮)
/// **盈亏线下限**(2026-10-18 由 2.0 下修到 1.35)。
///
/// 下限的来历是"32k 实测比 2.0"——但那是 **32k** 的比;实测比随 ctx **下探**:
/// **8.5k 生产档实测 轮 36.7ms / 裸 27.5ms = 1.33**(本次 propose 立案同批实测),
/// 2.9k 档 1.86。旧下限 2.0 在 ≤16k 档位**钳住了真线** ⇒ AL ∈ [1.33, 2.0] 这段
/// "投机其实更快"的区间被误判亏本 ⇒ 降级裸步(**每 token 亏 12%~50%**)。
/// 取 1.35 = 8.5k 实测比 + 小余量(EWMA 噪声)。
pub(crate) const BREAK_EVEN_MIN: f32 = 1.35;
pub(crate) const BREAK_EVEN_MAX: f32 = 6.0;

/// **投机盈亏线**(纯函数;2026-10-17 从 `exec/spec.rs` 的内联式抽出以便单测)。
///
/// 口径:`滚动轮成本 / 裸步成本`,夹 [`BREAK_EVEN_MIN`, `BREAK_EVEN_MAX`]。
/// 返回 `None` = **分母未测**(本进程还没跑过裸步/播种步)⇒ 窗口判据**不生效**
/// (只记日志),由连击判据守底。
///
/// 为什么不能拿一个兜底常数顶:32k 实测 `轮 77ms / 裸 38.4ms ⇒ 真线 2.0`,
/// 而旧兜底常量 **2.5 把“AL 2.25 的盈利档”误判成亏本** ⇒ 整会话降级裸 decode
/// (实测 decode 24.56 t/s、214 步里只有 18 个投机轮)。
/// 分母现在由**播种步**(`TurnSpecState::bare_seed_left`)在首个 turn 内置上:
/// 见 `roadmap.local/26-10-17-verify核级账-投机盈亏线.md` §3。
pub(crate) fn break_even_line(round_cost_ms: f32, bare_cost_ms: f32) -> Option<f32> {
    if round_cost_ms > 0.0 && bare_cost_ms > 0.0 {
        Some((round_cost_ms / bare_cost_ms).clamp(BREAK_EVEN_MIN, BREAK_EVEN_MAX))
    } else {
        None
    }
}

/// 盈亏线的**日志文本**(单一措词源;`None` = 未测)
pub(crate) fn break_even_text(line: Option<f32>) -> String {
    match line {
        Some(v) => format!("{v:.2}"),
        None => "未测".to_string(),
    }
}

/// E5-M5 spec 分相探针(owl-metrics 直用 API:release 恒开,热路径每轮
/// ~8 条环形推入 ≈ µs 级;/debug/metrics 查询。宏是 debug-only,装载域
/// 「直用 API」双轨纪律的 server 侧复刻)
fn mrec(tag: &str, dur: std::time::Duration) {
    owl_shared::metrics::with_metrics_store(|s| {
        s.timer_record_tag(tag, dur, file!(), line!());
    });
}
fn mcnt(tag: &str, n: u64) {
    owl_shared::metrics::with_metrics_store(|s| s.counter_add(tag, n, file!(), line!()));
}
/// 事实计数器(E5-DF4 崩坏排查):per-boot 命名空间记事实值
/// (token id / m / pos……counter 累计语义下每 tag 只记一次 = 事实值)
fn mfact(boot: u64, fact: &str, v: u64) {
    // **置值语义**(`counter_set`)而非累加:S4 ⑤-K3 起双 rank 会各写一次同名事实
    // (AR 之后两侧同值 ⇒ 幂等);累加会让 63 位 checksum 当场溢出(实测定谳)。
    owl_shared::metrics::with_metrics_store(|s| {
        s.counter_set(&format!("b{boot}.{fact}"), v, file!(), line!());
    });
}
type Result<T> = std::result::Result<T, ModelError>;

/// 增量解码(全量重解后取后缀差分;多字节字符跨 token 边界时以
/// 全文重解兜底,流式末字符可能延后一笔)
pub(crate) fn decode_delta(tok: &Tokenizer, out: &[u32], decoded: &mut String) -> String {
    const REPL: &str = "\u{FFFD}";
    let full = tok.decode(out);
    let common = decoded
        .as_bytes()
        .iter()
        .zip(full.as_bytes())
        .take_while(|(a, b)| a == b)
        .count();
    let mut cut = common;
    while cut > 0 && (!full.is_char_boundary(cut) || !decoded.is_char_boundary(cut)) {
        cut -= 1;
    }
    let mut emit = full[cut..].to_string();
    if emit.ends_with(REPL) {
        emit.truncate(emit.len() - REPL.len());
    }
    *decoded = full[..cut + emit.len()].to_string();
    emit
}

/// **采样接受的前缀长度**(A11,2026-10-14):首个 `drafts[i] != sampled[i]`
/// 的 `i`。语义 = 发射 `sampled[0..=m]`(前 m 个恰等于 drafts,末位是
/// 在拒绝点的重抽)。
///
/// 分布正确性(逐位归纳):发 x 的概率 =
///   P(s=draft)·1[x=draft] + P(s≠draft)·P(s=x | s≠draft) = p_target(x)
/// —— 即"抽-比"式投机采样,**不依赖草稿分布**(DFlash2 选择器只有 lattice
/// 码本、无全词表分布 ⇒ 无法用 p/q 比值式)。
pub(crate) fn spec_sampled_accept_prefix(drafts: &[u32], sampled: &[u32]) -> usize {
    drafts
        .iter()
        .zip(sampled.iter())
        .take_while(|(d, s)| d == s)
        .count()
}

/// **非 DFlash2（one-hot 草稿）采样接受的纯逻辑**:`sampled` = 逐行目标抽样
/// (长 `live+1`),返回 `(m, 提交序列)`。
///
/// 提交序列 = `sampled[0..=m]`:前 m 个恰等于 drafts(被接受),末位是拒绝点的
/// 重抽(可能全接受 ⇒ 末位 = 第 live 行抽样 = bonus)。
pub(crate) fn sample_accept(drafts: &[u32], sampled: &[u32]) -> (usize, Vec<u32>) {
    let m = spec_sampled_accept_prefix(drafts, sampled);
    (m, sampled[..=m.min(sampled.len().saturating_sub(1))].to_vec())
}

/// **惩罚历史入账**(`rep_penalty` 的成员集):只吃**已提交**的 token。
///
/// 2026-11-03 定谳:旧实现边抽样边 `hist.push`,把 `live+1` 个候选全部入账,
/// 而真正提交的只有 `m+1` 个 ⇒ 被拒/未提交的 token 也进了惩罚集
/// (`rep_penalty = 1.0` 时无感;开启后是语义错,与裸 decode 不一致)。
/// 统一走这里,新路径想漏也漏不了。
pub(crate) fn hist_commit(hist: &mut Vec<u32>, emitted: &[u32]) {
    hist.extend_from_slice(emitted);
}

/// 字节偏移向下取 char 边界(超界 = 全长;切 `&str` 前必须过一道)
pub(crate) fn floor_boundary(s: &str, i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    let mut i = i;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// **stop 尾缓冲上界**:返回最大的 `end` 使 `s[..end]` 的尾部不包含任一停止串的
/// **真前缀**(否则半截停止串会被当成正文流给客户端,命中时收不回来)。
/// 空 `stops` = 全长(零开销)。
pub(crate) fn safe_end(s: &str, stops: &[String]) -> usize {
    let mut end = s.len();
    for st in stops {
        // 尾部 k 字节(1 ≤ k < |st|)若恰为 st 的前缀 ⇒ 回退到该起点
        let maxk = (st.len() - 1).min(s.len());
        for k in (1..=maxk).rev() {
            let cut = s.len() - k;
            if s.is_char_boundary(cut) && st.starts_with(&s[cut..]) && cut < end {
                end = cut;
                break;
            }
        }
    }
    end
}

#[cfg(test)]
mod spec_accept_tests {
    use super::spec_sampled_accept_prefix;

    /// 全同 ⇒ 全接受(深度 = drafts 长度)
    #[test]
    fn all_match_accepts_everything() {
        assert_eq!(spec_sampled_accept_prefix(&[1, 2, 3], &[1, 2, 3, 9]), 3);
    }

    /// 首个不等处停(末位是重抽,不参与接受)
    #[test]
    fn first_mismatch_stops() {
        assert_eq!(spec_sampled_accept_prefix(&[1, 2, 3], &[1, 7, 3, 4]), 1);
        assert_eq!(spec_sampled_accept_prefix(&[1, 2, 3], &[9, 2, 3, 4]), 0);
    }

    /// 边界:空草稿 / 采样表短于草稿(取较短者,不越界)
    #[test]
    fn bounds_are_safe() {
        assert_eq!(spec_sampled_accept_prefix(&[], &[1]), 0);
        assert_eq!(spec_sampled_accept_prefix(&[1, 2], &[1]), 1);
    }

    /// 🔴 **红包(2026-11-03)**:`hist` 只许吃**已提交** token。
    ///
    /// 修前:边抽样边 push(live+1 个全进惩罚集),而实际只提交 m+1 个
    /// ⇒ 被拒/未提交的 token 也进 `rep_penalty` 历史。修后:先判接受,
    /// 再按 emitted 入账。
    #[test]
    fn hist_only_gets_committed_tokens() {
        use super::{hist_commit, sample_accept};
        // drafts = [5,6,7];逐行抽样 = [5,6,9,10] ⇒ 第 2 位拒,bonus=9
        let drafts = [5u32, 6, 7];
        let sampled = [5u32, 6, 9, 10];
        let (m, emitted) = sample_accept(&drafts, &sampled);
        assert_eq!((m, emitted.as_slice()), (2, &[5u32, 6, 9][..]));
        let mut hist = vec![1u32];
        hist_commit(&mut hist, &emitted);
        assert_eq!(hist, vec![1, 5, 6, 9], "只有已提交 token 能进惩罚集");
        assert!(!hist.contains(&7), "未提交的 draft 不得入账");
        assert!(!hist.contains(&10), "被拒行之后的抽样不得入账");

        // 全接受:emitted = drafts + bonus(第 live 行抽样)
        let (m, emitted) = sample_accept(&drafts, &[5, 6, 7, 11]);
        assert_eq!((m, emitted.as_slice()), (3, &[5u32, 6, 7, 11][..]));
        let mut hist = Vec::new();
        hist_commit(&mut hist, &emitted);
        assert_eq!(hist, vec![5, 6, 7, 11]);
    }
}

#[cfg(test)]
mod stop_tests {
    use super::{floor_boundary, safe_end};

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// 空 stop 列表 = 零回吐(全长)
    #[test]
    fn no_stops_returns_full_len() {
        assert_eq!(safe_end("abc", &[]), 3);
    }

    /// 尾部是停止串的**真前缀** ⇒ 回退(半截不许流给客户端)
    #[test]
    fn holdback_on_partial_prefix() {
        let stops = s(&["</tool_call>"]);
        assert_eq!(safe_end("abc</", &stops), 3, "`</` 是前缀 ⇒ 回吐");
        assert_eq!(safe_end("abc</to", &stops), 3);
        assert_eq!(safe_end("abc", &stops), 3, "无前缀 ⇒ 原样");
        assert_eq!(safe_end("abc<x", &stops), 5, "`<x` 不是前缀 ⇒ 放行");
        // 完整串不算"部分前缀":命中由调用方的 `find` 单独判定(本函数只
        // 负责把**半截**留住)。故此处 = 全长
        assert_eq!(safe_end("abc</tool_call>", &stops), 15);
    }

    /// 多串 / 长串前缀关系(短串是长串前缀时按最长回吐)
    #[test]
    fn holdback_multiple_stops() {
        let stops = s(&["\n\n", "STOP", "<|im_end|>"]);
        assert_eq!(safe_end("hi\n", &stops), 2);
        assert_eq!(safe_end("hi<|im_e", &stops), 2);
        assert_eq!(safe_end("hiSTO", &stops), 2);
        assert_eq!(safe_end("hi XX", &stops), 5);
    }

    /// floor_boundary:绝不切进多字节字符中间(中文/emoji)
    #[test]
    fn boundary_is_char_safe() {
        assert_eq!(floor_boundary("中文", 1), 0);
        assert_eq!(floor_boundary("中文", 3), 3);
        assert_eq!(floor_boundary("中文", 99), 6);
        let e = "a😀b";
        assert_eq!(floor_boundary(e, 2), 1, "emoji 中间回退到 1");
        assert_eq!(floor_boundary(e, 5), 5);
    }
}
