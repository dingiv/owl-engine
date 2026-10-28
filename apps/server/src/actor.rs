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
    /// 取消 turn(AGT-10:客户端断流 ⇒ 不再烧 GPU;**B5.1:必须指名**)。
    ///
    /// `turn` 已知(Accepted 帧之后断流)= 精确终结该 turn;
    /// 未知(断在受理帧之前)= 退化为按会话找(服务面已按会话串行 ⇒ 唯一)。
    /// 都不会动同批其它会话的 turn(旧实现走 `abandon` = 杀本步执行 turn,
    /// 合批档下会误杀别家)。
    Abort {
        session: u64,
        turn: Option<u64>,
        /// 回执(引擎已清完 ⇒ 服务面可释放会话闸;不发 = 延迟断言级 bug)
        ack: tokio::sync::oneshot::Sender<()>,
    },
    /// **tokenize 旁路**(G5③;`POST /tokenize`):不落 turn、不碰池/图。
    /// 应答走 oneshot(取消安全:请求方走了则发失败,不许静默)。
    Tokenize {
        text: String,
        /// 先套家族单轮包装(与裸文本提交同一律);false = 纯文本编码
        chat_wrap: bool,
        reply: tokio::sync::oneshot::Sender<Result<owl_engine::TokenizeReply, String>>,
    },
    /// **合批/队列事实旁路**(`POST /debug/load` 前后取差值;纯读,不落 turn)。
    /// 为什么单列一条而不是复用 `VramReport`:后者要取**块账房**快照(设备命令),
    /// 灌流测量要的是**几乎零成本**的事实口(前后各一次,不能污染读数)。
    BatchStats {
        reply: tokio::sync::oneshot::Sender<BatchStatsSnapshot>,
    },
    /// **store 账查询旁路**(`GET /debug/vram`;第 0/3 刀):不落 turn、不改状态。
    /// 应答 = 三层账 + **owner 维明细** + **事件账** + **块账房对账**(全等门)。
    VramReport {
        reply: tokio::sync::oneshot::Sender<VramSnapshot>,
    },
}

/// **`/debug/vram` 的一次快照**(三面事实:store 账 / 块账房 / 句柄计数)。
///
/// 为什么三面同屋:它们必须**同一时刻**取(引擎不在 pump 时)—— 分开取会让
/// "事件账 vs 块账房"的差集混入并发窗口的噪声,判据就不再是判据。
pub struct VramSnapshot {
    /// store 报告(三层账 + owner 维 + 事件账)
    pub report: owl_store::StoreReport,
    /// 块账房快照(逐 rank;`Err` = 后端不支持/读失败,如实带出,不静默)
    pub blocks: Vec<(u8, Result<Vec<owl_iface::contract::BlockLedgerRow>, String>)>,
    /// 记账失败计数(非零 = 有事件没入账)
    pub hook_fail: u64,
    /// 窗未收次数(图窗失败且去向未定)
    pub window_unclosed: u64,
    /// 窗内未归还块数
    pub open_window_blocks: usize,
    /// **状态腿淘汰序预览**(前 8 条;会话感知淘汰的可观测面)
    pub snap_evict: Vec<owl_store::SnapEvictPreview>,
    /// **默认池账(逐 rank)**:(used, used_high, reserved, reserved_high)。
    /// 与块账之差 = **池缓存**(已 free 仍占物理显存;`RELEASE_THRESHOLD=∞`)。
    pub pools: Vec<(u8, Result<(u64, u64, u64, u64), String>)>,
    /// **合批事实**(B5.1 服务面):上限 / 已走批步数 / 批图是否在场。
    /// 判据:开了 `max_batch>1` 却 `steps` 不涨 = 批门从未成立(要么并发不够,
    /// 要么被会话闸串行) —— 这正是服务面最常见的“以为在合批”假象。
    pub batch: BatchFacts,
    /// **设备显存实况**(`cuMemGetInfo`):`(free, total)`。`Err` = 后端不支持。
    /// `/debug/vram` 用它算驱动残差(`total − free − store − 池缓存`)。
    pub device: std::result::Result<(u64, u64), String>,
}

/// 灌流测量用的轻量事实(见 [`EngineReq::BatchStats`])
pub struct BatchStatsSnapshot {
    pub max_batch: usize,
    pub batch_steps: u64,
    pub has_batch_graph: bool,
    pub queue_len: usize,
    /// 当前活跃 turn 数(**多活跃**:合批档 >1)
    pub active: usize,
}

/// 合批事实(与 `/debug/vram` 同刻取;见 [`VramSnapshot::batch`])
pub struct BatchFacts {
    pub max_batch: usize,
    pub steps: u64,
    pub graph: bool,
    /// **准入容量快照**（store 事实 + 判决输出；判"还能进几个"的现场）
    pub free_pages: usize,
    pub cache_pages: usize,
    pub reserved_pages: usize,
    pub usable_pages: usize,
    pub n_slots: usize,
    pub n_mem: usize,
    pub n_eff: usize,
    /// 排队快照（最多 16，按等待降序）：`(turn, session, waited_ms)`
    pub queued: Vec<(u64, u64, u64)>,
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
        /// **实际生成预算**(准入夹紧后的值;与请求值不同 ⇒ 已如实夹紧)
        max_new: usize,
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
    shutdown: crate::shutdown::Shutdown,
    telemetry: std::sync::Arc<crate::telemetry::Telemetry>,
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
    // 遥测：容量快照降频（`movable_bytes` 要遍历前缀树，不能每拍）
    let mut last_capacity: Option<crate::telemetry::CapacitySnapshot> = None;
    let mut tele_tick: u64 = 0;
    loop {
        // ① 阻塞等首个请求(通道关 = 引擎退役)
        let Some(req) = rx.recv().await else { return };
        handle_req(running, &mut sinks, req).await;
        // ② 排空积压(pump 前收完,保证收发不相交)
        while let Ok(req) = rx.try_recv() {
            handle_req(running, &mut sinks, req).await;
        }
        // ③ 泵到排空,事件按 turn 路由
        let actor_prof = std::env::var_os("OWL_ACTOR_PROF").is_some();
        loop {
            // **泵间收件**(2026-10-20 真机定谳):旧实现泵到 Idle 才回 outer 收件 ⇒
            // 错峰到达的请求**永远等前一个 turn 跑完**才进引擎(实测:150ms 后到的 B
            // 在 A 的 96 步 decode 期间一步未进 ⇒ 合批/时间片全成摆设)。
            // 安全性:此处是**两拍之间**(pump 已完整返回),提交/取消都是 host 账操作,
            // 不打断设备侧半推进态;`dropped` 处置仍在下一条。
            while let Ok(req) = rx.try_recv() {
                handle_req(running, &mut sinks, req).await;
            }
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
            // **遥测快照**（每拍；容量每 32 拍）—— HTTP 侧零等待读
            {
                tele_tick = tele_tick.wrapping_add(1);
                if tele_tick.is_multiple_of(32) {
                    let (f, c) = running.capacity_probe();
                    last_capacity = Some(crate::telemetry::CapacitySnapshot {
                        free_pages: f.free_pages,
                        cache_pages: f.cache_pages,
                        reserved_pages: f.reserved_pages,
                        usable_pages: c.usable_pages,
                        n_slots: c.n_slots,
                        n_mem: c.n_mem,
                        n_eff: c.n_eff,
                    });
                }
                telemetry.publish(crate::telemetry::QueueSnapshot {
                    queue_len: running.queue_len(),
                    active: running.active_len(),
                    batch_steps: running.batch_steps(),
                    max_batch: running.max_batch(),
                    has_graph: running.has_batch_graph(),
                    queued: running.queued_snapshot(16),
                    capacity: last_capacity,
                    rejects: running.reject_snapshot(16),
                    wait_hist: running.wait_hist(),
                });
            }
            // P2.2 侦察(2026-10-19;env 门,临时诊断):拆"循环外 1.55ms/步" ——
            // pump(引擎侧)/dispatch(事件投递)/账务 三段。
            let t_l0 = std::time::Instant::now();
            match running.pump().await {
                Ok(TurnEvent::Idle) => break,
                Ok(ev) => {
                    let d_pump = t_l0.elapsed();
                    // turn 终了 = 性能账落盘点(每 turn 覆写,进程被 kill 也不丢)
                    let terminal = matches!(
                        ev,
                        TurnEvent::Completed { .. } | TurnEvent::Failed { .. }
                    );
                    // AGT-12:本 turn 首个事件之前先把真值 `cached_tokens` 发出去
                    // (块级前缀缓存命中只有 begin 后才知道;短 turn 无第二次机会)
                    //
                    // ⚠️ **次序纪律(2026-10-19 修)**:本段必须在**终态清理之前** ——
                    // 旧序把 `pending_cache.remove` 放在 terminal 分支里先跑,把
                    // "终态前一帧"静默删掉 ⇒ 客户端看到 submit 估值(新会话 = 0)而
                    // 引擎实际命中 9k(实测 9130-tok 冷会话整前缀:引擎日志 复用面
                    // m=285/285=9120,客户端 cached=0)。丢帧 = 已记账的 bug,顺序固定。
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
                    if terminal {
                        if let Some(turn) = event_turn(&ev) {
                            cache_reported.remove(&turn);
                            // 兜底:正常已被上面取走(留下的 = 号不匹配的脏项)
                            if pending_cache.remove(&turn).is_some() {
                                owl_shared::log_debug!(
                                    "actor",
                                    "turn {turn} 终态时仍有待发校正帧(已丢弃)"
                                );
                            }
                        }
                    }
                    let t_d0 = std::time::Instant::now();
                    if !dispatch(&mut sinks, ev).await {
                        // 投递失败 = 客户端已走 ⇒ 本 turn 弃置
                        dropped = true;
                    }
                    if actor_prof {
                        let d_disp = t_d0.elapsed();
                        owl_shared::log_info!(
                            "actor-prof",
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
                    // 引擎错误(架子):烧毁 active turn(E5 性能会战:
                    // 弃置滞留 = GDN 格泄漏 → 2 格两轮耗尽)+ 广播
                    owl_shared::log_warn!("actor", "pump 错误: {e}");
                    abandon_turn(running, &mut sinks, &format!("engine: {e}")).await;
                    // **缺口 3(2026-10-19)**:硬预算/池耗尽 ⇒ **不再直通致命**:
                    // 丢本 turn(客户端拿到 500/503,如实) + **立即收缩**
                    // (池排空 + 池缓存 trim + Maintain(Pressure))
                    // + **继续跑**。判据:引擎不退出,后续请求仍可服务;只有真正的
                    // 设备级错误(ServerClosed / 状态机失守)才走优雅关停。
                    if matches!(e, owl_iface::contract::ModelError::PoolExhausted { .. }) {
                        match running.relieve_pressure().await {
                            // **如实报数**:还缓存 0 块也照说(mid-turn 池空是常态 ——
                            // 失败那一刻的块都在使用中,不在池里)
                            Ok((freed, k, hkv, hsnap)) => owl_shared::log_warn!(
                                "actor",
                                "显存压力:已收缩(还缓存 {freed} 块 + 目标腾 {k} KV 槽 + {hkv} KV 宿主块 + {hsnap} 快照)⇒ 引擎继续(丢本 turn)"
                            ),
                            Err(pe) => owl_shared::log_warn!(
                                "actor",
                                "收缩失败(继续跑,下次再试): {pe}"
                            ),
                        }
                        continue;
                    }
                    if e.is_fatal() {
                        owl_shared::log_warn!("actor", "致命引擎错误({e})⇒ 优雅关停");
                        shutdown.graceful(&format!("引擎不可恢复错误:{e}"));
                    }
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

/// 客户端主动 abort:[`EngineReq::Abort`] ⇒ **只弃该会话/该 turn**。
/// 无活跃也无排队 = no-op(竞争下 turn 可能刚自然终了)。
async fn drop_turn<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<SinkEvent>>,
    session: u64,
    turn: Option<u64>,
) {
    let ev = match turn {
        Some(t) => running.cancel_turn(t).await,
        None => running.cancel_session(session).await,
    };
    let Some(ev) = ev else {
        owl_shared::log_debug!(
            "actor",
            "abort 未命中(session {session:#x} turn {turn:?}) —— 可能刚自然终了"
        );
        return;
    };
    // 只投递该 turn 的 sink;不广播(合批档下广播 = 给别家会话发终态)
    let tid = event_turn(&ev).unwrap_or(0);
    match sinks.remove(&tid) {
        Some(sink) => {
            if !deliver(&sink, SinkEvent::Turn(ev)).await {
                owl_shared::log_debug!(
                    "actor",
                    "abort 终态投递失败(turn {tid};请求方已断开 —— 终态本来就没人收)"
                );
            }
        }
        None => owl_shared::log_debug!("actor", "abort 终态无人接收(turn {tid})"),
    }
}

/// 单条请求处置:Abort = 弃置活跃 turn;Tokenize = 旁路(不落 turn);
/// Chat = 交 [`accept`]。
///
/// ⚠️ **Tokenize 在 actor 线程内跑**(与采样同线程 ⇒ 排在当前 turn 之后):
/// 这是“tokenizer 单一真源”(归属 `RunningEngine`)的代价;工量 = 毫秒级 CPU。
/// 若将来要“生成期间也能查”,把 tokenizer 克隆进 HTTP 层的前提是引擎侧
/// 先导出 `chat_wrap + try_encode` 的组合面(否则包装律两份)—— 另案。
async fn handle_req<D: DeviceClient>(
    running: &mut RunningEngine<D>,
    sinks: &mut HashMap<u64, mpsc::Sender<SinkEvent>>,
    req: EngineReq,
) {
    match req {
        EngineReq::Abort { session, turn, ack } => {
            drop_turn(running, sinks, session, turn).await;
            // 回执 = "该 turn 已从引擎账上消失"(会话闸据此释放;发送失败 =
            // 请求方已走,无损失,但留痕)
            if ack.send(()).is_err() {
                owl_shared::log_debug!("actor", "abort 回执投递失败(请求方已断开)");
            }
        }
        EngineReq::Tokenize { text, chat_wrap, reply } => {
            let out = running.tokenize(&text, chat_wrap);
            if reply.send(out).is_err() {
                // 请求方已断开(不许静默:这是可失败返回值)
                owl_shared::log_debug!("actor", "tokenize 应答投递失败(请求方已断开)");
            }
        }
        EngineReq::BatchStats { reply } => {
            let snap = BatchStatsSnapshot {
                max_batch: running.max_batch(),
                batch_steps: running.batch_steps(),
                has_batch_graph: running.has_batch_graph(),
                queue_len: running.queue_len(),
                active: running.active_len(),
            };
            if reply.send(snap).is_err() {
                owl_shared::log_debug!("actor", "batch stats 应答投递失败(请求方已断开)");
            }
        }
        EngineReq::VramReport { reply } => {
            // 纯读账(store_report 是只读投影)+ **块账房快照**(命令面只读口)。
            // 同一 turn 内取两面 ⇒ 无交错(引擎此刻不在 pump)⇒ "事件账 vs 块账房"
            // 的全等门才有意义。**本路零副作用**:不落 turn、不搬块、不建图。
            let (hook_fail, window_unclosed, open_window_blocks) = running.handle_probe();
            let blocks = running.block_ledger_probe().await;
            let pools = running.pool_probe().await;
            let report = running.store_report();
            let device = running.mem_info().await;
            let snap_evict = running.snap_evict_preview(8);
            let snap = VramSnapshot {
                report,
                blocks,
                hook_fail,
                window_unclosed,
                open_window_blocks,
                snap_evict,
                pools,
                device,
                batch: {
                    let (facts, cap) = running.capacity_probe();
                    BatchFacts {
                        max_batch: running.max_batch(),
                        steps: running.batch_steps(),
                        graph: running.has_batch_graph(),
                        free_pages: facts.free_pages,
                        cache_pages: facts.cache_pages,
                        reserved_pages: facts.reserved_pages,
                        usable_pages: cap.usable_pages,
                        n_slots: cap.n_slots,
                        n_mem: cap.n_mem,
                        n_eff: cap.n_eff,
                        queued: running.queued_snapshot(16),
                    }
                },
            };
            if reply.send(snap).is_err() {
                owl_shared::log_debug!("actor", "vram 账应答投递失败(请求方已断开)");
            }
        }
        req => accept(running, sinks, req).await,
    }
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
                SinkEvent::Accepted {
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