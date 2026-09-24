//! GPU actor server:线程间设备服务器(状态机 + 命令面 + 异步门面)。
//!
//! ⚠️ 本文件是【设计骨架/伪代码】(2026-09-23,async-runtime.md v0.1 的 A1 期
//! 接口框架)。只定义接口形状与契约注释,**不要求编译通过、不含实现体**。
//! 实施时按 A1 里程碑逐段落肉,并以 docs/arch/async-runtime.md 为唯一需求源。
//!
//! 结构总览(对应设计文档 §二~§五):
//!
//! ```text
//! GPUClient(async 门面,Clone)          ← 本文件对外唯一类型
//!   │ mpsc<Cmd> + oneshot 回执
//!   ▼
//! GpuServer(actor 线程,唯一执行权)
//!   │ 状态机 Booting→Ready→Closing
//!   ▼
//! CudaDevice(治理面:池/租约/哨兵/相位,不变)
//! ```

use crate::{BackendError, CudaDevice, DeviceGraph, MemPhase};
use std::sync::Arc;

// ============================================================================
// §1 状态机
// ============================================================================

/// actor 生命周期三态(design §三)。
///
/// 注意:捕获窗的 MemPhase(Idle/Capturing/...)由 CaptureSession 内部自持,
/// actor 不重复管理相位;actor 状态机只管"哪类命令此刻合法"。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ServerState {
    /// 线程已启动、bind_to_thread 进行中;一切工作命令非法
    Booting,
    /// 正常态:Run / Capture / Replay / Sync 全部合法
    Ready,
    /// 收到 Close,排空在途命令后线程退出;新命令一律拒绝
    Closing,
}

// ============================================================================
// §2 命令面
// ============================================================================

/// 回执通道:命令的完成通知(oneshot)。
/// 契约 4:回执用 oneshot,不用闭包 future——future 被遗漏 = UB 的雷
/// (async-cuda 教训)在本形态下结构性不存在。
type Ack<T> = oneshot::Sender<Result<T, BackendError>>;

/// actor 命令(每变体自带回执)。
enum Cmd {
    // ---- 档一:提交即回 ----

    /// 图回放:所有权回传 actor 线程,租约校验 + cuGraphLaunch。
    /// 提交成功即回执(发射异步于 GPU;完成通知走事件档,二期)。
    /// 仅 Ready 态合法。
    Replay {
        graph: DeviceGraph,
        ack: Ack<()>,
    },

    // ---- 档三:完成即回 ----

    /// 任意阻塞面 → 闭包(池分配/memcpy 装载/sync/cublas/eager 发射/查询)。
    /// 仅 Ready 态合法。闭包内禁止 .await(契约 1:Rust 异步不进门)。
    Run {
        job: Box<dyn FnOnce(&CudaDevice) -> Result<(), BackendError> + Send>,
        ack: Ack<()>,
    },

    /// 捕获事务(原子;契约 2:闭包面 = frame 发射能力,非 CaptureSession)。
    /// 仅 Ready 态合法;窗内发射数 > 0 由命令边界校验(坑 C 封死)。
    Capture {
        plan: CapturePlan,
        ack: Ack<Arc<DeviceGraph>>,
    },

    /// 关机:排空在途命令 → 线程退出。
    Close { ack: Ack<()> },
}

/// 捕获事务的声明面(契约 2:描述与执行分离的雏形;参考 cuTile DeviceOp)。
///
/// 捕获窗内只允许 frame 发射面(owl 治理的登记通道);本结构把"窗内要跑什么"
/// 声明为预绑定缓冲 + 一个 frame 闭包,杜绝旁路直发(坑 D 封死)。
struct CapturePlan {
    /// 窗口前预分配并钉住的绑定缓冲(租约登记走 session 正常面)。
    /// 句柄在本结构内存活到图销毁 —— 租约系统的自动保活对象。
    // bindings: GraphBindings,           // ← A2 期落地(seam 文档 §三)
    /// 档位(bs 精确档;GDN/mamba slot 不可 pad)
    // batch: usize,
    /// 实例化 flags(AUTO_FREE_ON_LAUNCH 在租约在场时不需要,见 seam §一)
    // flags: CUgraphInstantiate_flags,
    /// 窗内发射体:唯一合法的窗内代码。参数 = frame(发射面 + ctx)。
    // step: Box<dyn FnOnce(&CaptureFrame) -> Result<(), BackendError> + Send>,
    _placeholder: (),
}

// ============================================================================
// §3 actor 线程本体
// ============================================================================

/// 设备服务线程:唯一触碰 CudaDevice 执行权的线程(A3 每卡一线程)。
struct GpuServer {
    /// 设备执行权唯一持有者(**不外泄**;GPUClient 只持命令发送端)。
    dev: CudaDevice,
    rx: mpsc_receiver(),
    state: ServerState,
}

// 伪代码辅助标记(实施时删除)
#[allow(non_snake_case)]
fn mpsc_receiver() -> ReceiverPlaceholder {
    unimplemented!("骨架:tokio::sync::mpsc::Receiver<Cmd>")
}
struct ReceiverPlaceholder;

impl GpuServer {
    /// 线程主循环(状态机的执行器)。
    fn pump(mut self) {
        // Booting:bind_to_thread 一次到位(A3;此后本线程再无显式绑定)
        // PSEUDO: self.dev.ctx().bind_to_thread().expect(...);
        // PSEUDO: self.state = ServerState::Ready;

        // PSEUDO: while let Some(cmd) = self.rx.blocking_recv() {
        // PSEUDO:     match cmd {
        // PSEUDO:         Cmd::Run { job, ack }
        // PSEUDO:             if self.state == ServerState::Ready =>
        // PSEUDO:         {
        // PSEUDO:             let _ = ack.send(job(&self.dev));
        // PSEUDO:         }
        // PSEUDO:         Cmd::Replay { graph, ack }
        // PSEUDO:             if self.state == ServerState::Ready =>
        // PSEUDO:         {
        // PSEUDO:             // 租约校验在 DeviceGraph::launch 内部(P1 起全构建)
        // PSEUDO:             let _ = ack.send(graph.launch());
        // PSEUDO:             // 图生命周期:回放面持有(HashMap<GraphId, Arc<DeviceGraph>>)
        // PSEUDO:             //        由 server 记帐,close 时统一销毁
        // PSEUDO:         }
        // PSEUDO:         Cmd::Capture { plan, ack }
        // PSEUDO:             if self.state == ServerState::Ready =>
        // PSEUDO:         {
        // PSEUDO:             let _ = ack.send(self.run_capture_transaction(plan));
        // PSEUDO:         }
        // PSEUDO:         cmd => {
        // PSEUDO:             // 非 Ready 态收到工作命令 / Close:
        // PSEUDO:             //   Close → state = Closing,排空即 break;
        // PSEUDO:             //   其余 → 结构化 LawViolation 回执,不 panic(设计 §三)
        // PSEUDO:             let _ = self.reject(cmd, "server 不在该态");
        // PSEUDO:         }
        // PSEUDO:     }
        // PSEUDO: }
    }

    /// 捕获事务(原子;失败不污染状态机)。
    /// 流程 = seam 文档 §三 的三步,warmup 门禁由 CaptureSession 内部把关:
    /// PSEUDO:
    ///   1. self.dev.capture_session()?;
    ///   2. session.capture(plan.flags, |frame| {
    ///          // plan 绑定缓冲整档租入(自动记 touched);
    ///          // (plan.step)(&frame)          ← 窗内唯一代码,纯同步发射面
    ///      })?;
    ///   3. seal(measured) → Arc<DeviceGraph> 记帐返回
    fn run_capture_transaction(&mut self, plan: CapturePlan) -> Result<Arc<DeviceGraph>, BackendError> {
        let _ = plan;
        unimplemented!("骨架:A2 期落地")
    }

    /// 拒绝非法命令:结构化回执 + Close 的排空语义。
    fn reject(&mut self, cmd: Cmd, why: &str) {
        let _ = (cmd, why);
        unimplemented!("骨架:LawViolation 回执;Close → Closing + break")
    }
}

// ============================================================================
// §4 异步门面(对外唯一类型)
// ============================================================================

/// 异步设备句柄:Clone 便宜(mpsc sender);多任务共享同一 actor。
/// **不含 CudaDevice**——设备执行权在 server 线程内,门面只是排队凭证。
pub struct GPUClient {
    tx: TxPlaceholder,
}

// 伪代码辅助标记(实施时删除)
type TxPlaceholder = ();
type CudaStreamPlaceholder = ();
struct CaptureFramePlaceholder;

impl GPUClient {
    /// 启动 server(专用 OS 线程;UUID 钉卡;pool_bytes = 默认池容量,
    /// 构造参数指定)。
    ///
    /// 契约:不返回 Arc<CudaDevice>(设计 §二公理 1;执行权独占)。
    pub fn connect(uuid: &str, pool_bytes: u64) -> Result<Self, BackendError> {
        let _ = (uuid, pool_bytes);
        unimplemented!("骨架:建 dev → mpsc channel → thread::spawn(pump)")
    }

    /// 任意阻塞面 → async(design §四 run)。
    /// 闭包在 server 线程执行;await 侧零阻塞。泛型 R 经私有 oneshot 回传,
    /// 命令面保持非泛型(R 不进 Cmd 枚举)。
    pub async fn run<R, F>(&self, job: F) -> Result<R, BackendError>
    where
        R: Send + 'static,
        F: FnOnce(&CudaDevice) -> Result<R, BackendError> + Send + 'static,
    {
        let _ = job;
        unimplemented!("骨架:私有 oneshot 载 R + 命令 ack 载 ()")
    }

    /// 捕获事务(契约 2:plan 声明面,原子)。
    pub async fn capture(&self, plan: CapturePlan) -> Result<Arc<DeviceGraph>, BackendError> {
        let _ = plan;
        unimplemented!("骨架")
    }

    /// 图回放(档一:提交即回)。
    pub async fn replay(&self, graph: Arc<DeviceGraph>) -> Result<(), BackendError> {
        let _ = graph;
        unimplemented!("骨架")
    }

    /// 全设备同步(阻塞消化在 server 线程)。
    pub async fn sync(&self) -> Result<(), BackendError> {
        unimplemented!("骨架:self.run(|dev| dev.ctx().synchronize())")
    }

    /// 相位查询(只读面直达;无执行权,无竞争)。
    pub fn phase(&self) -> MemPhase {
        unimplemented!("骨架:经 Run 命令查,或缓存在门面(原子读)")
    }

    /// 关机:排空在途命令,server 线程退出。
    pub async fn close(&self) -> Result<(), BackendError> {
        unimplemented!("骨架")
    }
}

// ============================================================================
// §5 二期占位(不实现,占住语义位;design §五 档二)
// ============================================================================

/// 事件完成通知(档二):池流 record event → 专用 waiter 线程消化 → 唤醒
/// 对应 Waker。供 runner decode 循环使用。
///
/// 开放问题(design §十.2):waiter 线程归 actor 管辖还是独立。
// pub struct EventFuture { /* oneshot + waiter 注册表 */ }

/// Copy 流(bindings H2D 与计算重叠;design §八 性能条款):
/// server 命令面增加"目标流"字段 + 事件边登记 —— 二期流拓扑立项,非本期。
