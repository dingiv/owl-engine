//! **引擎泵循环（engine 侧）** —— 2026-11-15 P1 自 `apps/server/src/executor.rs` 下沉。
//!
//! 为什么归引擎：准入重试、显存压力收缩、取消/弃置语义、缓存命中校正、
//! 每拍队列快照 —— 全部是引擎状态机的一部分。server 原先自己实现了一遍
//! （`executor.rs` + `telemetry.rs`），等于在引擎外持有第二份调度策略。
//!
//! 线程模型（与 `owl-cuda` 的 device server 同形）：
//! - 本模块的 `run` **独占一条 OS 线程**（`service::EngineService::boot` 里 spawn）；
//! - `RunningEngine` 非 Send（设备/图/池全部线程亲和）⇒ 只能在这一条线程里
//!   构造、泵、销毁；
//! - 外界只见 [`super::EngineClient`]（mpsc + 事实单元，全 Send）。
//!
//! 调度纪律（原样保留）：**收件与泵不相交** —— pump 的 await 点半途取消会把
//! GDN 递推状态留在半推进态（设备侧已推进、host 账未记）⇒ 先排空收件箱、
//! 再一口气泵到 Idle；泵期间的提交在 channel 缓冲。
//!
//! 取消面（AGT-10）：`pump` 的每个 await 点后回收到货 —— 客户端断流时
//! `sink.send` 失败（接收端已 drop）即标记该 turn 为弃置，下一拍之前
//! `cancel_turn`（≤1 步延迟）。

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use owl_iface::contract::DeviceClient;
use tokio::sync::mpsc;

use super::diag::{self, BatchStatsSnapshot, CapacityFacts, DiagSnapshot, FactsCell, QueueFacts};
use super::{Request, ServiceExit, StreamEvent};
use crate::RunningEngine;
use crate::TurnEvent;

/// 事件投递（律 #1：不许静默丢弃可失败返回值）。
/// `send` 失败 = **请求方已断开** —— 返回 false 让调用方标记弃置。
async fn deliver(sink: &mpsc::Sender<StreamEvent>, ev: StreamEvent) -> bool {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    match sink.send(ev).await {
        Ok(()) => true,
        Err(e) => {
            if !WARNED.swap(true, Ordering::Relaxed) {
                owl_shared::log_warn!("executor", "事件投递失败(请求方已断开 ⇒ 本 turn 弃置): {e}");
            }
            false
        }
    }
}

/// 引擎线程主循环（独占线程内持 `&mut RunningEngine`；随停服信号/通道关闭退出）。
pub async fn run<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    mut rx: mpsc::Receiver<Request>,
    perf_dump: Option<std::path::PathBuf>,
    facts: Arc<FactsCell>,
    mut stop: tokio::sync::watch::Receiver<bool>,
) -> ServiceExit {
    // turn_id → 事件回流（sink；Completed/Failed 后摘除）
    let mut sinks: HashMap<u64, mpsc::Sender<StreamEvent>> = HashMap::new();
    // 已发过缓存校正帧的 turn（AGT-12；终态摘除）
    let mut cache_reported: HashSet<u64> = HashSet::new();
    // 待发缓存校正（turn → 真值 cached_tokens）：在**该 turn 首个事件投递前**发出
    let mut pending_cache: HashMap<u64, usize> = HashMap::new();
    // AGT-10：哪一 turn 的接收方（HTTP 任务）走了 —— 在下一个 pump 边界前弃置。
    // 语义 = **客户端取消**（不是引擎错误）⇒ 走 `cancel_turn`（发布已写入前缀 +
    // 中断补拍快照），不走 `abandon_turn`（那档声明"账面不可信、不发布"）。
    let mut dropped: Option<u64> = None;
    // 容量快照降频（`movable_bytes` 要遍历前缀树，不能每拍）
    let mut last_capacity: Option<CapacityFacts> = None;
    let mut tele_tick: u64 = 0;
    loop {
        // ① 阻塞等首个请求 —— **同时**等停服信号：
        //    请求方（server 的连接任务）会长期持有 `EngineClient` 克隆 ⇒ 只靠
        //    `rx.recv()` 返回 None 判停服是**不可行的**（空闲时永不到来）。
        //    旧实现靠 main drop 发送端，现在线程归引擎所有，必须在 recv 上 select。
        let req = tokio::select! {
            biased;
            _ = stop.changed() => {
                owl_shared::log_info!("executor", "停服信号(watch)⇒ 引擎泵循环收摊");
                return ServiceExit::Normal;
            }
            r = rx.recv() => r,
        };
        let Some(req) = req else {
            owl_shared::log_info!("executor", "请求通道关闭 ⇒ 引擎泵循环收摊");
            return ServiceExit::Normal;
        };
        handle_req(running, &mut sinks, req).await;
        // ② 排空积压（pump 前收完，保证收发不相交）
        while let Ok(req) = rx.try_recv() {
            handle_req(running, &mut sinks, req).await;
        }
        // ③ 泵到排空，事件按 turn 路由
        let exec_prof = std::env::var_os("OWL_EXECUTOR_PROF").is_some();
        loop {
            // **泵间收件**（2026-10-20 真机定谳）：错峰到达的请求必须能在两拍之间
            // 进引擎，否则合批/时间片全成摆设。安全性：此处是两拍之间，提交/取消
            // 都是 host 账操作，不打断设备侧半推进态。
            while let Ok(req) = rx.try_recv() {
                handle_req(running, &mut sinks, req).await;
            }
            // ① 停服检查（步边界）：watch 信号 / 通道关闭
            if *stop.borrow() || rx.is_closed() {
                owl_shared::log_info!(
                    "executor",
                    "停服信号({})⇒ 本拍不推进，收摊",
                    if *stop.borrow() { "watch/SIGTERM" } else { "请求通道关闭" }
                );
                return ServiceExit::Normal;
            }
            // ② "还有人要吗"状态检查（步边界）：接收方关闭 ⇒ 下一拍之前弃置
            if let Some(t) = running.active_turn_id() {
                if sinks.get(&t).is_some_and(|s| s.is_closed()) {
                    owl_shared::log_info!(
                        "executor",
                        "turn {t} 的接收方已关闭 ⇒ 下一拍之前弃置(客户端断流/取消)"
                    );
                    dropped = Some(t);
                }
            }
            // ③ 断流 turn 在下一步之前弃置（≤0 拍延迟）
            if let Some(t) = dropped.take() {
                cancel_turn(running, &mut sinks, 0, Some(t)).await;
            }
            // ③b **首 token 门**（2026-11-16）：起跑时按新增 token 定的预算到期
            // 仍无 `first_token_ms` ⇒ 响亮强停。防一个卡死的 prefill 拖住整泵
            // （客户端只会看到 HTTP 超时，而引擎日志能归因到 turn/token 数）。
            ttft_gate(running, &mut sinks).await;
            // AGT-12：turn 已定相 ⇒ 补发真值 `cached_tokens`
            if let Some(t) = running.active_turn_id() {
                if !cache_reported.contains(&t) {
                    if let Some(cached) = running.active_cached_len(t) {
                        pending_cache.insert(t, cached);
                    }
                }
            }
            // 每拍队列事实（容量每 32 拍）
            {
                tele_tick = tele_tick.wrapping_add(1);
                if tele_tick.is_multiple_of(32) {
                    last_capacity = Some(CapacityFacts::probe(running));
                }
                facts.publish(QueueFacts::sample(running, last_capacity));
            }
            let t_l0 = std::time::Instant::now();
            match running.pump().await {
                Ok(TurnEvent::Idle) => {
                    // `admission_on_full = Queue` 时，队首在等池腾位；重判与超时
                    // 都发生在 `pump` 里 ⇒ 不能在这里断流去等新请求。小幅 sleep 防空转。
                    if running.admission_pending().is_some() {
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                        continue;
                    }
                    break;
                }
                Ok(ev) => {
                    let d_pump = t_l0.elapsed();
                    let terminal =
                        matches!(ev, TurnEvent::Completed { .. } | TurnEvent::Failed { .. });
                    // AGT-12：本 turn 首个事件之前先把真值 `cached_tokens` 发出去
                    // （次序纪律：必须在终态清理之前 —— 丢帧 = 已记账的 bug）
                    if let Some(t) = event_turn(&ev) {
                        if let Some(cached) = pending_cache.remove(&t) {
                            if cache_reported.insert(t) {
                                if let Some(sink) = sinks.get(&t) {
                                    deliver(
                                        sink,
                                        StreamEvent::CachedTokens { turn: t, cached_tokens: cached },
                                    )
                                    .await;
                                }
                            }
                        }
                    }
                    if terminal {
                        if let Some(turn) = event_turn(&ev) {
                            cache_reported.remove(&turn);
                            if pending_cache.remove(&turn).is_some() {
                                owl_shared::log_debug!(
                                    "executor",
                                    "turn {turn} 终态时仍有待发校正帧(已丢弃)"
                                );
                            }
                        }
                    }
                    let t_d0 = std::time::Instant::now();
                    if let Some(t) = dispatch(&mut sinks, ev).await {
                        dropped = Some(t);
                    }
                    if exec_prof {
                        let d_disp = t_d0.elapsed();
                        owl_shared::log_info!(
                            "executor-prof",
                            "pump={d_pump:?} disp={d_disp:?} rest={:?}",
                            t_l0.elapsed()
                        );
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
                    owl_shared::log_warn!("executor", "pump 错误: {e}");
                    abandon_turn(running, &mut sinks, &format!("engine: {e}")).await;
                    // 缺口 3（2026-10-19）：硬预算/池耗尽 ⇒ **不直通致命**：
                    // 丢本 turn + 立即收缩（池排空 + trim + Maintain(Pressure)）+ 继续跑。
                    if matches!(e, owl_iface::contract::ModelError::PoolExhausted { .. }) {
                        match running.relieve_pressure().await {
                            Ok((freed, k, hkv, hsnap)) => owl_shared::log_warn!(
                                "executor",
                                "显存压力:已收缩(还缓存 {freed} 块 + 目标腾 {k} KV 槽 + {hkv} KV 宿主块 + {hsnap} 快照)⇒ 引擎继续(丢本 turn)"
                            ),
                            Err(pe) => owl_shared::log_warn!(
                                "executor",
                                "收缩失败(继续跑,下次再试): {pe}"
                            ),
                        }
                        continue;
                    }
                    if e.is_fatal() {
                        owl_shared::log_warn!("executor", "致命引擎错误({e})⇒ 交回 main 收摊");
                        return ServiceExit::Fatal(e.to_string());
                    }
                    // 非致命：丢本 turn，回外层等新请求
                    break;
                }
            }
        }
    }
}

/// 事件所属 turn（Idle 无归属 ⇒ None）
fn event_turn(ev: &TurnEvent) -> Option<u64> {
    match ev {
        TurnEvent::Prefill { turn, .. }
        | TurnEvent::Token { turn, .. }
        | TurnEvent::Completed { turn, .. }
        | TurnEvent::Failed { turn, .. } => Some(*turn),
        TurnEvent::Idle => None,
    }
}

/// 弃置活跃 turn（`session` 仅作日志标识；0 = 未知/错误路径）+ 广播终态。
async fn abandon_turn<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<StreamEvent>>,
    why: &str,
) {
    let failed = running.abandon(why.to_string()).await;
    let failed_turn = match &failed {
        TurnEvent::Failed { turn, .. } => *turn,
        _ => 0,
    };
    owl_shared::log_info!("executor", "弃置 turn {failed_turn}({why})");
    match sinks.remove(&failed_turn) {
        Some(sink) => {
            deliver(&sink, StreamEvent::Turn(failed)).await;
        }
        // 0 = 引擎没有给出归属 turn（无活跃时的早退）。只有这种"无归属"才广播；
        // **已知 turn 的 sink 不在 = 请求方已走**，广播会把 A 的失败终态投给 B（F-21）。
        None if failed_turn == 0 => {
            for (_, sink) in sinks.drain() {
                deliver(&sink, StreamEvent::Turn(failed.clone())).await;
            }
        }
        None => {
            owl_shared::log_debug!(
                "executor",
                "弃置 turn {failed_turn} 的 sink 已不在（请求方已断开）"
            );
        }
    }
}

/// **首 token 门**：逐拍检查预算到期的活跃 turn，强停并投递终态。
///
/// 与客户端 cancel 同一条清理路径（发布已写入前缀 + 中断补拍快照），
/// 终态文案归因为 `TTFT 超时`（服务面把它翻成 5xx/断流，前端能立刻重试）。
async fn ttft_gate<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<StreamEvent>>,
) {
    for e in running.ttft_expired(crate::running::now_ms()) {
        owl_shared::log_warn!(
            "ttft",
            "turn {} 首 token 超时: 起跑后 {}ms > 预算 {}ms（prompt={} cached={} new={} tok）⇒ 强制停止",
            e.turn,
            e.waited_ms,
            e.budget_ms,
            e.prompt_tokens,
            e.cached_tokens,
            e.new_tokens
        );
        let Some(ev) = running.cancel_turn_reason(e.turn, "TTFT 超时").await else {
            continue;
        };
        let tid = event_turn(&ev).unwrap_or(e.turn);
        if let Some(sink) = sinks.remove(&tid) {
            let _ = deliver(&sink, StreamEvent::Turn(ev)).await;
        }
    }
}

/// 客户端主动 abort：[`Request::Cancel`] ⇒ **只弃该会话/该 turn**。
async fn cancel_turn<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<StreamEvent>>,
    session: u64,
    turn: Option<u64>,
) {
    let ev = match turn {
        Some(t) => running.cancel_turn(t).await,
        // 不知道 turn（客户端在 `Accepted` 之前就断了）：同会话可能"活跃 1 条 +
        // 排队 N 条"（P4 起），所以**优先挑接收方已关的那一条**；都不关（正常
        // 的会话级取消）才按老口径取活跃/最早排队。
        None => {
            let candidates = running.session_turns(session);
            let pick = candidates
                .iter()
                .copied()
                .find(|tid| sinks.get(tid).is_some_and(|s| s.is_closed()))
                .or_else(|| candidates.first().copied());
            match pick {
                Some(t) => running.cancel_turn(t).await,
                None => None,
            }
        }
    };
    let Some(ev) = ev else {
        owl_shared::log_debug!(
            "executor",
            "abort 未命中(session {session:#x} turn {turn:?}) —— 可能刚自然终了"
        );
        return;
    };
    let tid = event_turn(&ev).unwrap_or(0);
    match sinks.remove(&tid) {
        Some(sink) => {
            if !deliver(&sink, StreamEvent::Turn(ev)).await {
                owl_shared::log_debug!(
                    "executor",
                    "abort 终态投递失败(turn {tid};请求方已断开 —— 终态本来就没人收)"
                );
            }
        }
        None => owl_shared::log_debug!("executor", "abort 终态无人接收(turn {tid})"),
    }
}

/// 单条请求处置：Cancel = 弃置；Tokenize/Diag/BatchStats = 旁路；Chat = accept。
async fn handle_req<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<StreamEvent>>,
    req: Request,
) {
    match req {
        Request::Cancel { session, turn, ack } => {
            cancel_turn(running, sinks, session, turn).await;
            if ack.send(()).is_err() {
                owl_shared::log_debug!("executor", "abort 回执投递失败(请求方已断开)");
            }
        }
        Request::Tokenize { text, chat_wrap, reply } => {
            let out = running.tokenize(&text, chat_wrap);
            if reply.send(out).is_err() {
                owl_shared::log_debug!("executor", "tokenize 应答投递失败(请求方已断开)");
            }
        }
        Request::BatchStats { reply } => {
            let snap = BatchStatsSnapshot {
                max_batch: running.max_batch(),
                batch_steps: running.batch_steps(),
                has_batch_graph: running.has_batch_graph(),
                queue_len: running.queue_len(),
                active: running.active_len(),
            };
            if reply.send(snap).is_err() {
                owl_shared::log_debug!("executor", "batch stats 应答投递失败(请求方已断开)");
            }
        }
        Request::Diag { reply } => {
            let snap: DiagSnapshot = diag::snapshot(running).await;
            if reply.send(snap).is_err() {
                owl_shared::log_debug!("executor", "vram 账应答投递失败(请求方已断开)");
            }
        }
        req @ Request::Chat { .. } => accept(running, sinks, req).await,
    }
}

/// 登记并提交（预算越界/空 prompt 在 submit 边界落地为 Rejected）
async fn accept<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<StreamEvent>>,
    req: Request,
) {
    let Request::Chat { session, prompt, max_new, opts, sink } = req else {
        owl_shared::log_warn!("executor", "accept 收到非 Chat 请求(内部错误)");
        return;
    };
    // 用**结构化拒绝版**：429 语义由 `TurnError::reject()` 决定（P3）
    match running.submit_session_checked(session, prompt, max_new, opts) {
        Ok((sid, tid)) => {
            // usage 真账：prompt token 数 = 队列/活跃 turn 的 prompt_ids 长度
            // （末块 prefill 不发 Prefill 事件，submit 后即刻取，单源）
            let pt = running.turn_prompt_len(tid).unwrap_or(0);
            let cached = running.turn_cached_len(tid).unwrap_or(0);
            sinks.insert(tid, sink.clone());
            deliver(
                &sink,
                StreamEvent::Accepted {
                    session: sid,
                    turn: tid,
                    prompt_tokens: pt,
                    cached_tokens: cached,
                    max_new: running.turn_max_new(tid).unwrap_or(0),
                },
            )
            .await;
        }
        Err(e) => {
            deliver(&sink, StreamEvent::Rejected { err: e.into_turn_error() }).await;
        }
    }
}

/// 事件路由：按 turn id 转发；终态事件后摘 sink。返回 `Some(turn)` = 该 turn
/// 的接收方已不在（投递失败 ⇒ 客户端取消）。
async fn dispatch(
    sinks: &mut HashMap<u64, mpsc::Sender<StreamEvent>>,
    ev: TurnEvent,
) -> Option<u64> {
    let (tid, done) = match &ev {
        TurnEvent::Prefill { turn, .. }
        | TurnEvent::Token { turn, .. }
        | TurnEvent::Completed { turn, .. }
        | TurnEvent::Failed { turn, .. } => (
            *turn,
            matches!(ev, TurnEvent::Completed { .. } | TurnEvent::Failed { .. }),
        ),
        TurnEvent::Idle => return None,
    };
    let mut gone = None;
    if let Some(sink) = sinks.get(&tid) {
        if !deliver(sink, StreamEvent::Turn(ev)).await {
            gone = Some(tid);
        }
    }
    if done {
        sinks.remove(&tid);
    }
    gone
}
