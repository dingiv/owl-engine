//! GPU server:设备执行权的唯一宿主(线程间 server;A3 每卡一线程)。
//!
//! ⚠️ 伪代码骨架(2026-09-23;async-runtime.md §三/§四)。
//! server 视角:收信封 → 状态机裁决 → 落到治理面 → 回信封。
//! server 不知道 client 的存在(没有"哪个任务发的"概念),只认信封。

// 伪代码:不编译;实施时 =
//   use crate::protocol::{Reply, Request, RequestReceiver};
//   use crate::{BackendError, MemPhase};
use crate::protocol::{Reply, Request, RequestReceiver, RequestId};
use crate::{BackendError, MemPhase};

// ============================================================================
// §1 状态机
// ============================================================================

/// server 生命周期三态(捕获窗的 MemPhase 由治理面自持,server 不重复管理)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    /// 线程启动中:bind_to_thread / 池初始化;工作请求一律拒绝
    Booting,
    /// 正常态:全部语义请求合法
    Ready,
    /// 收到 Close:排空在途 → 资源清算 → 线程退出
    Closing,
}

// ============================================================================
// §2 server 本体
// ============================================================================

/// 设备执行权的唯一宿主。
/// 内部构成(重写后从旧 src 收编为私有模块,不再是公共抽象):
/// - 池账房(旧 pool.rs 的 Arena/Ledger 机制)
/// - 相位机 + 延迟归还队列(旧 governor.rs)
/// - 哨兵③审计 + 捕获四步(旧 graph.rs)
/// - memx 流序 memcpy / nvrtc 编译缓存(旧 ffi/util)
pub(crate) struct GpuServer {
    state: State,
    rx: RequestReceiver,
    // dev_internals: DevInternals,   // ← 重写落定:治理面的私有聚合体
}

impl GpuServer {
    /// 线程入口:uuid 钉卡 → bind_to_thread → 池初始化 → Ready → 命令泵。
    // PSEUDO: pub fn serve(uuid: &str, pool_bytes: u64, rx: RequestReceiver) {
    // PSEUDO:     let mut me = Self::boot(uuid, pool_bytes, rx);
    // PSEUDO:     me.pump();
    // PSEUDO:     me.settle();   // Closing:资源清算(延迟归还队列排空/图销毁/账本对账)
    // PSEUDO: }

    /// Booting:一切失败 = Init 错误(此时还没有回执对象可通知,
    /// 通过 spawn 的入口 oneshot 报给创建者)。
    // PSEUDO: fn boot(uuid: &str, pool_bytes: u64, rx: RequestReceiver) -> Self;

    /// 命令泵:收信封 → 裁决 → 派发。永不 panic(错误过线,设计 §三)。
    // PSEUDO: fn pump(&mut self) {
    // PSEUDO:     while let Some(req) = self.rx.blocking_recv() {
    // PSEUDO:         match (self.state, &req.op) {
    // PSEUDO:             // Close 在任何态都合法(Closing 态重复 Close = Ok 幂等)
    // PSEUDO:             (_, Op::Close) => { self.reply(req.id, Reply::Ok); break; }
    // PSEUDO:             (State::Booting, _) | (State::Closing, _) =>
    // PSEUDO:                 self.reply(req.id, Reply::Err(law_violation("非 Ready 态"))),
    // PSEUDO:             (State::Ready, op) => self.dispatch(req.id, op),
    // PSEUDO:         }
    // PSEUDO:     }
    // PSEUDO: }

    /// 语义派发:一个变体一个 handler(全部阻塞消化在本线程)。
    // PSEUDO: fn dispatch(&mut self, id: RequestId, op: &Op) {
    // PSEUDO:     match op {
    // PSEUDO:         Op::Sync            => self.reply(id, Reply::Ok);          // ctx.synchronize
    // PSEUDO:         Op::Malloc { .. }   => self.reply(id, self.pools.malloc(..));   // → Reply::Block
    // PSEUDO:         Op::Htod { .. }     => self.reply(id, self.pools.htod(..));     // → Reply::Block
    // PSEUDO:         Op::Free { .. }     => self.reply(id, self.pools.free(..));
    // PSEUDO:         Op::Launch { .. }   => self.reply(id, self.kernels.launch(..)); // 提交即回
    // PSEUDO:         Op::Replay { .. }   => self.reply(id, self.graphs.replay(..));
    // PSEUDO:         Op::CaptureBegin{..} => { /* 治理面 begin;置 in_window 标记 */ }
    // PSEUDO:         Op::CaptureEnd      => { /* end+instantiate+audit → Reply::Graph */ }
    // PSEUDO:     }
    // PSEUDO: }

    /// 回执(信封回信;id 配对在 protocol 层)。
    // PSEUDO: fn reply(&self, id: RequestId, reply: Reply);

    /// Closing 清算:延迟归还队列排空 → 图全销毁 → 账本对账(bytes_alive==0,
    /// 不为 0 = 结构化泄漏报告,设计 §三"排空后退出")。
    // PSEUDO: fn settle(&mut self);
}

// ============================================================================
// §3 捕获事务(窗内协议面)
// ============================================================================
//
// 捕获是 server 上唯一一个"多条消息组成一个事务"的流程:
//
//   client: CaptureBegin ──► Launch* ──► CaptureEnd
//           (server:治理面 begin;窗内发射落 capture 流;end → instantiate → audit)
//
// 状态机细则(设计 §四契约 2):
// - CaptureBegin 仅 Ready 态合法;进入后 server 置 in_window 标记;
// - in_window 期间:Launch 合法(落捕获流);Malloc 合法(捕获安全分配);
//   Htod/CaptureBegin/Replay 非法(结构化报错——窗内零 memcpy 零旁路);
// - CaptureEnd:发射数 > 0 校验(坑 C)→ 治理面 end/instantiate/audit;
//   任一步失败 → 事务回滚(治理面 discard,相位回 Ready),错误过线;
// - 事务成功 → 相位回 Ready,图记入 server 图表,Reply::Graph 回执。
//
// 悬挂事务保护:client 崩了没发 CaptureEnd?——队列排空时 in_window 仍开
// = Closing 前 discard 回滚(结构化),不许带窗退出。

// ============================================================================
// §4 启动接线(伪代码;落点 = 新 lib.rs 的 spawn 入口)
// ============================================================================
//
// PSEUDO: pub fn spawn(uuid: &str, pool_bytes: u64) -> Result<Client, BackendError> {
// PSEUDO:     let (tx, rx) = channel::<Request>();       // protocol 层类型
// PSEUDO:     let (boot_tx, boot_rx) = oneshot();
// PSEUDO:     thread::Builder::new().name(format!("owl-gpu-{uuid}"))
// PSEUDO:         .spawn(move || GpuServer::serve(uuid, pool_bytes, rx))?;
// PSEUDO:     boot_rx.await?;                            // Booting → Ready 握手
// PSEUDO:     Ok(Client::new(ProtocolClient::new(tx)))   // client 不感知以上一切
// PSEUDO: }
