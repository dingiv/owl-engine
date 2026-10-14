//! engine actor(§A3 同步隔离):`RunningEngine` 单所有权独占设备,
//! 外界经 mpsc 提交、经每请求 sink 收事件流。设备侧的一切异步都在
//! actor 内闭合,HTTP 层零设备句柄。
//!
//! 调度纪律(取消安全):**收件与泵不相交** —— pump 的 await 点半途
//! 取消会把 GDN 递推状态留在半推进态(设备侧已推进、host 账未记),
//! 因此先排空收件箱、再一口气泵到 Idle;泵期间的提交在 channel 缓冲,
//! 提交延迟 ≈ 一个采样步(架子可接受;真并发随 M2 batching 立项)。
//!
//! **取消面(AGT-10,2026-10-14)**:`pump` 的每个 await 点后回收到货 ——
//! 客户端断流时 `sink.send` 失败(接收端已 drop)即标记该 turn 为弃置,
//! 下一步之前 `abandon` 掉(≤1 步延迟)。旧行为 = 引擎继续烧 GPU 到
//! 自然终了(长输出/长思考场景可白白烧掉几十秒)。

use std::collections::HashMap;

use owl_engine::RunningEngine;
use owl_engine::TurnEvent;
use owl_iface::contract::DeviceClient;
use tokio::sync::mpsc;

/// 引擎请求(HTTP 层 → actor)
pub enum EngineReq {
    Chat {
        /// None = 临时会话(每请求独立);Some = 连续会话(增量 prefill)
        session: Option<u64>,
        prompt: String,
        max_new: usize,
        /// per-turn 可选项(AGT-06:采样覆写/停止串/种子;测速口径 ignore_eos)
        opts: owl_engine::TurnOpts,
        sink: mpsc::Sender<SinkEvent>,
    },
    /// 取消活跃 turn(AGT-10:客户端断流 ⇒ 不再烧 GPU)。
    /// **窄语义**:只终结**当前活跃** turn(GDN 格/块链正常回收,
    /// 引擎可继续接新请求);无活跃 turn 时为 no-op。
    Abort { session: u64 },
}

/// 事件投递(律 #1:不许静默丢弃可失败返回值)。
/// `send` 失败 = **请求方已断开** —— 返回 false 让调用方标记弃置
/// (AGT-10:宁停引擎,不烧 GPU)。首次失败打一条 warn(不刷屏)。
async fn deliver(sink: &mpsc::Sender<SinkEvent>, ev: SinkEvent) -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    match sink.send(ev).await {
        Ok(()) => true,
        Err(e) => {
            if !WARNED.swap(true, Ordering::Relaxed) {
                owl_shared::log_warn!("actor", "事件投递失败(请求方已断开 ⇒ 本 turn 弃置): {e}");
            }
            false
        }
    }
}

/// actor → 请求方事件(首笔必为 Accepted/Rejected,其后 Turn 流)
#[derive(Clone, Debug)]
pub enum SinkEvent {
    Accepted {
        session: u64,
        turn: u64,
        prompt_tokens: usize,
        /// 前缀命中量(usage `prompt_tokens_details.cached_tokens` 真源;AGT-12)
        cached_tokens: usize,
    },
    Rejected { err: String },
    /// 定相后的**真实前缀命中量**校正帧(AGT-12):块级前缀缓存命中在
    /// submit 时尚未决定 ⇒ `Accepted.cached_tokens` 可能低估(会话守卫
    /// 命中在 submit 时已知,块级命中只有 begin 后才知道)。服务面以本帧
    /// 覆盖计数 —— 不许让客户端看到偏小的数(AGT-12 就是命中率窗户)
    Session { turn: u64, cached_tokens: usize },
    Turn(TurnEvent),
}

/// actor 主循环(独占线程内持 `&mut RunningEngine`;随 channel 关闭退出)
pub async fn run_actor<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    mut rx: mpsc::Receiver<EngineReq>,
    perf_dump: Option<std::path::PathBuf>,
) {
    // turn_id → 事件回流(sink;Completed/Failed 后摘除)
    let mut sinks: HashMap<u64, mpsc::Sender<SinkEvent>> = HashMap::new();
    // 已发过缓存校正帧的 turn(AGT-12;终态摘除)
    let mut cache_reported: std::collections::HashSet<u64> = std::collections::HashSet::new();
    // 待发缓存校正(turn → 真值 cached_tokens):在**该 turn 首个事件投递前**
    // 发出 —— 否则短 turn(单 pump 内跑完)会因 sink 已摘除而丢帧(实测踩过:
    // 196 行 prefill 后紧跟 turn end,校正帧无处可投)
    let mut pending_cache: HashMap<u64, usize> = HashMap::new();
    // AGT-10:客户端断开且尚未弃置(下一个 pump 边界前处理)
    let mut dropped = false;
    loop {
        // ① 阻塞等首个请求(通道关 = 引擎退役)
        let Some(req) = rx.recv().await else { return };
        // 取消请求(AoT 客户端主动 abort):立刻弃置活跃 turn
        if let EngineReq::Abort { session } = &req {
            drop_turn(running, &mut sinks, *session).await;
        } else {
            accept(running, &mut sinks, req).await;
        }
        // ② 排空积压(pump 前收完,保证收发不相交)
        while let Ok(req) = rx.try_recv() {
            if let EngineReq::Abort { session } = &req {
                drop_turn(running, &mut sinks, *session).await;
            } else {
                accept(running, &mut sinks, req).await;
            }
        }
        // ③ 泵到排空,事件按 turn 路由
        loop {
            // AGT-10:断流 turn 在下一步之前弃置(≤1 采样步延迟)
            if dropped {
                dropped = false;
                abandon_turn(running, &mut sinks, "客户端断流").await;
            }
            // AGT-12:turn 已定相 ⇒ 补发真值 `cached_tokens`(块级前缀缓存
            // 命中只有此刻知道;无法掩盖——偏小的数字会误导 pi 的 cacheRead)
            if let Some(t) = running.active_turn_id() {
                if !cache_reported.contains(&t) {
                    if let Some(cached) = running.active_cached_len(t) {
                        pending_cache.insert(t, cached);
                    }
                }
            }
            match running.pump().await {
                Ok(TurnEvent::Idle) => break,
                Ok(ev) => {
                    // turn 终了 = 性能账落盘点(每 turn 覆写,进程被 kill 也不丢)
                    let terminal = matches!(
                        ev,
                        TurnEvent::Completed { .. } | TurnEvent::Failed { .. }
                    );
                    if terminal {
                        if let Some(turn) = event_turn(&ev) {
                            cache_reported.remove(&turn);
                            pending_cache.remove(&turn);
                        }
                    }
                    // AGT-12:本 turn 首个事件之前先把真值 `cached_tokens` 发出去
                    // (块级前缀缓存命中只有 begin 后才知道;短 turn 无第二次机会)
                    if let Some(t) = event_turn(&ev) {
                        if let Some(cached) = pending_cache.remove(&t) {
                            if cache_reported.insert(t) {
                                if let Some(sink) = sinks.get(&t) {
                                    deliver(
                                        sink,
                                        SinkEvent::Session { turn: t, cached_tokens: cached },
                                    )
                                    .await;
                                }
                            }
                        }
                    }
                    if !dispatch(&mut sinks, ev).await {
                        // 投递失败 = 客户端已走 ⇒ 本 turn 弃置
                        dropped = true;
                    }
                    if terminal {
                        if let Some(p) = &perf_dump {
                            let f = owl_shared::metrics::MetricsFilter::new();
                            if let Err(e) = owl_shared::metrics::dump(&f, p) {
                                owl_shared::log_warn!("perf", "dump 失败 {}: {e}", p.display());
                            }
                        }
                    }
                }
                Err(e) => {
                    // 引擎错误(架子):烧毁 active turn(E5 性能会战:
                    // 弃置滞留 = GDN 格泄漏 → 2 格两轮耗尽)+ 广播
                    owl_shared::log_warn!("actor", "pump 错误: {e}");
                    abandon_turn(running, &mut sinks, &format!("engine: {e}")).await;
                    break;
                }
            }
        }
    }
}

/// 事件所属 turn(Idle 无归属 ⇒ None)
fn event_turn(ev: &TurnEvent) -> Option<u64> {
    match ev {
        TurnEvent::Prefill { turn, .. }
        | TurnEvent::Token { turn, .. }
        | TurnEvent::Completed { turn, .. }
        | TurnEvent::Failed { turn, .. } => Some(*turn),
        TurnEvent::Idle => None,
    }
}

/// 弃置活跃 turn(`session` 仅作日志标识;0 = 未知/错误路径)+ 广播终态。
/// 引擎侧 `abandon` 走正常回收路径(块链/快照留留,GDN 格释放)⇒
/// 后继请求不受影响。
async fn abandon_turn<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<SinkEvent>>,
    why: &str,
) {
    let failed = running.abandon(why.to_string()).await;
    let failed_turn = match &failed {
        TurnEvent::Failed { turn, .. } => *turn,
        _ => 0,
    };
    owl_shared::log_info!("actor", "弃置 turn {failed_turn}({why})");
    match sinks.remove(&failed_turn) {
        Some(sink) => {
            deliver(&sink, SinkEvent::Turn(failed)).await;
        }
        None => {
            for (_, sink) in sinks.drain() {
                deliver(&sink, SinkEvent::Turn(failed.clone())).await;
            }
        }
    }
}

/// 客户端主动 abort:[`EngineReq::Abort`] ⇒ 弃置**当前活跃** turn。
/// 无活跃 turn = no-op 语义(`abandon` 自带,不报错 —— 竞态下 turn
/// 可能刚自然终了)。`session` 仅作日志标识。
async fn drop_turn<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<SinkEvent>>,
    session: u64,
) {
    abandon_turn(running, sinks, &format!("客户端 abort(session {session:#x})")).await;
}

/// 登记并提交(预算越界/空 prompt 在 submit 边界落地为 Rejected)
async fn accept<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<SinkEvent>>,
    req: EngineReq,
) {
    let EngineReq::Chat { session, prompt, max_new, opts, sink } = req else {
        // Abort 由调用方(accept）处理;到达此处 = 编程错误
        owl_shared::log_warn!("actor", "accept 收到非 Chat 请求(内部错误)");
        return;
    };
    match running.submit_session_opts(session, prompt, max_new, opts) {
        Ok((sid, tid)) => {
            // usage 真账:prompt token 数 = 队列/活跃 turn 的 prompt_ids
            // 长度(末块 prefill 不发 Prefill 事件,短 prompt 场景事件面
            // 拿不到 —— submit 后即刻取用,单源);cached 同源(AGT-12)
            let pt = running.turn_prompt_len(tid).unwrap_or(0);
            let cached = running.turn_cached_len(tid).unwrap_or(0);
            sinks.insert(tid, sink.clone());
            deliver(
                &sink,
                SinkEvent::Accepted { session: sid, turn: tid, prompt_tokens: pt, cached_tokens: cached },
            )
            .await;
        }
        Err(e) => {
            deliver(&sink, SinkEvent::Rejected { err: e.to_string() }).await;
        }
    }
}

/// 事件路由:按 turn id 转发;终态事件后摘 sink。返回 false = 投递失败
/// (请求方已断开 ⇒ 调用方标记弃置,AGT-10)
async fn dispatch(
    sinks: &mut HashMap<u64, mpsc::Sender<SinkEvent>>,
    ev: TurnEvent,
) -> bool {
    let (tid, done) = match &ev {
        TurnEvent::Prefill { turn, .. }
        | TurnEvent::Token { turn, .. }
        | TurnEvent::Completed { turn, .. }
        | TurnEvent::Failed { turn, .. } => (*turn, matches!(ev, TurnEvent::Completed { .. } | TurnEvent::Failed { .. })),
        TurnEvent::Idle => return true,
    };
    let mut ok = true;
    if let Some(sink) = sinks.get(&tid) {
        ok = deliver(sink, SinkEvent::Turn(ev)).await;
    }
    if done {
        // 摘除即 drop(mpsc Sender 无显式 close;对端 recv 尽后自然断流)
        sinks.remove(&tid);
    }
    ok
}