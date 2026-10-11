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
        /// OpenAI 兼容扩展:忽略 EOS 跑满 max_new(测速口径)
        ignore_eos: bool,
        sink: mpsc::Sender<SinkEvent>,
    },
}

/// 事件投递(律 #1:不许静默丢弃可失败返回值)。
/// `send` 失败 = 请求方已断开 —— 属正常路径,但**首次**失败必须可见
/// (每事件都打会刷屏,故用一次性旗标;非错误语义见下)。
async fn deliver(sink: &mpsc::Sender<SinkEvent>, ev: SinkEvent) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static WARNED: AtomicBool = AtomicBool::new(false);
    if let Err(e) = sink.send(ev).await {
        if !WARNED.swap(true, Ordering::Relaxed) {
            eprintln!("[actor] 事件投递失败(请求方已断开,后续同类不再重复打印): {e}");
        }
    }
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
    running: &mut RunningEngine<D>,
    mut rx: mpsc::Receiver<EngineReq>,
) {
    // turn_id → 事件回流(sink;Completed/Failed 后摘除)
    let mut sinks: HashMap<u64, mpsc::Sender<SinkEvent>> = HashMap::new();
    loop {
        // ① 阻塞等首个请求(通道关 = 引擎退役)
        let Some(req) = rx.recv().await else { return };
        accept(running, &mut sinks, req).await;
        // ② 排空积压(pump 前收完,保证收发不相交)
        while let Ok(req) = rx.try_recv() {
            accept(running, &mut sinks, req).await;
        }
        // ③ 泵到排空,事件按 turn 路由
        loop {
            match running.pump().await {
                Ok(TurnEvent::Idle) => break,
                Ok(ev) => dispatch(&mut sinks, ev).await,
                Err(e) => {
                    // 引擎错误(架子):烧毁 active turn(E5 性能会战:
                    // 弃置滞留 = GDN 格泄漏 → 2 格两轮耗尽)+ 广播
                    eprintln!("[actor] pump 错误: {e}");
                    let failed = running.abandon(format!("engine: {e}")).await;
                    let failed_turn = match &failed {
                        TurnEvent::Failed { turn, .. } => *turn,
                        _ => 0,
                    };
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
    let EngineReq::Chat { session, prompt, max_new, ignore_eos, sink } = req;
    match running.submit_session_ex(session, prompt, max_new, ignore_eos) {
        Ok((sid, tid)) => {
            // usage 真账:prompt token 数 = 队列/活跃 turn 的 prompt_ids
            // 长度(末块 prefill 不发 Prefill 事件,短 prompt 场景事件面
            // 拿不到 —— submit 后即刻取用,单源)
            let pt = running.turn_prompt_len(tid).unwrap_or(0);
            sinks.insert(tid, sink.clone());
            deliver(&sink, SinkEvent::Accepted { session: sid, turn: tid, prompt_tokens: pt }).await;
        }
        Err(e) => {
            deliver(&sink, SinkEvent::Rejected { err: e.to_string() }).await;
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
        deliver(sink, SinkEvent::Turn(ev)).await;
    }
    if done {
        // 摘除即 drop(mpsc Sender 无显式 close;对端 recv 尽后自然断流)
        sinks.remove(&tid);
    }
}
