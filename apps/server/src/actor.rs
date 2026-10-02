//! engine actor(§A3 同步隔离):`RunningEngine` 单所有权独占设备,
//! 外界经 mpsc 提交、经每请求 sink 收事件流。设备侧的一切异步都在
//! actor 内闭合,HTTP 层零设备句柄。
//!
//! 调度纪律(取消安全):**收件与泵不相交** —— pump 的 await 点半途
//! 取消会把 GDN 递推状态留在半推进态(设备侧已推进、host 账未记),
//! 因此先排空收件箱、再一口气泵到 Idle;泵期间的提交在 channel 缓冲,
//! 提交延迟 ≈ 一个采样步(架子可接受;真并发随 M2 batching 立项)。
//!
//! 挂账:pump Err 后引擎态可能滞留(active turn 无 abort 原语,iface
//! 挂账)—— 现行处置 = 广播错误 + 清 sink,拒绝后续提交前先排干泵。

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
        sink: mpsc::Sender<SinkEvent>,
    },
}

/// actor → 请求方事件(首笔必为 Accepted/Rejected,其后 Turn 流)
#[derive(Clone, Debug)]
pub enum SinkEvent {
    Accepted { session: u64, turn: u64, prompt_tokens: usize },
    Rejected { err: String },
    Turn(TurnEvent),
}

/// actor 主循环(独占线程内持 `&mut RunningEngine`;随 channel 关闭退出)
pub async fn run_actor<D: DeviceClient>(
    mut running: &mut RunningEngine<D>,
    mut rx: mpsc::Receiver<EngineReq>,
) {
    // turn_id → 事件回流(sink;Completed/Failed 后摘除)
    let mut sinks: HashMap<u64, mpsc::Sender<SinkEvent>> = HashMap::new();
    loop {
        // ① 阻塞等首个请求(通道关 = 引擎退役)
        let Some(req) = rx.recv().await else { return };
        accept(&mut running, &mut sinks, req).await;
        // ② 排空积压(pump 前收完,保证收发不相交)
        while let Ok(req) = rx.try_recv() {
            accept(&mut running, &mut sinks, req).await;
        }
        // ③ 泵到排空,事件按 turn 路由
        loop {
            match running.pump().await {
                Ok(TurnEvent::Idle) => break,
                Ok(ev) => dispatch(&mut sinks, ev).await,
                Err(e) => {
                    // 引擎错误(架子):广播 + 清桌。挂账:turn abort 原语
                    // 落地后此处应烧毁 active turn 而非滞留
                    eprintln!("[actor] pump 错误: {e}");
                    for (_, sink) in sinks.drain() {
                        let _ = sink
                            .send(SinkEvent::Turn(TurnEvent::Failed {
                                turn: 0,
                                err: format!("engine: {e}"),
                            }))
                            .await;
                    }
                    break;
                }
            }
        }
    }
}

/// 登记并提交(预算越界/空 prompt 在 submit 边界落地为 Rejected)
async fn accept<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<SinkEvent>>,
    req: EngineReq,
) {
    let EngineReq::Chat { session, prompt, max_new, sink } = req;
    match running.submit_session(session, prompt, max_new) {
        Ok((sid, tid)) => {
            // usage 真账:prompt token 数 = 队列/活跃 turn 的 prompt_ids
            // 长度(末块 prefill 不发 Prefill 事件,短 prompt 场景事件面
            // 拿不到 —— submit 后即刻取用,单源)
            let pt = running.turn_prompt_len(tid).unwrap_or(0);
            sinks.insert(tid, sink.clone());
            let _ = sink.send(SinkEvent::Accepted { session: sid, turn: tid, prompt_tokens: pt }).await;
        }
        Err(e) => {
            let _ = sink.send(SinkEvent::Rejected { err: e.to_string() }).await;
        }
    }
}

/// 事件路由:按 turn id 转发;终态事件后摘 sink
async fn dispatch(
    sinks: &mut HashMap<u64, mpsc::Sender<SinkEvent>>,
    ev: TurnEvent,
) {
    let (tid, done) = match &ev {
        TurnEvent::Prefill { turn, .. }
        | TurnEvent::Token { turn, .. }
        | TurnEvent::Completed { turn, .. }
        | TurnEvent::Failed { turn, .. } => (*turn, matches!(ev, TurnEvent::Completed { .. } | TurnEvent::Failed { .. })),
        TurnEvent::Idle => return,
    };
    if let Some(sink) = sinks.get(&tid) {
        let _ = sink.send(SinkEvent::Turn(ev)).await;
    }
    if done {
        // 摘除即 drop(mpsc Sender 无显式 close;对端 recv 尽后自然断流)
        sinks.remove(&tid);
    }
}
