//! 协议层(内部):client 与 GPU server 线程之间的线格式 + 关联机制。
//!
//! ⚠️ 伪代码骨架(2026-09-23;async-runtime.md §四命令面)。
//! 本模块是 server/client 之间**唯一的耦合点**:
//! - client 只见到语义方法(alloc/capture/replay/...),见不到 Request/Response;
//! - server 只见到 Request/Response,见不到 client 的 future/waker;
//! - 关联机制(request-id ↔ ack)封装在此,两侧都不手工管。
//!
//! 设计原则(design §二公理):
//! - 协议里**没有 Rust 泛型**——线格式封闭(枚举),泛型在 client 门面用
//!   私有 oneshot 桥接(骨架见 client.rs `bridge`);
//! - 协议里**没有 CudaDevice**——server 收到的是语义请求,自己决定怎么落到
//!   治理面;协议不感知池/租约/相位的内部形态。

// ============================================================================
// §1 请求关联
// ============================================================================

/// 请求标识:client 生成,server 原样带回。回执配对的唯一钥匙。
/// u64 单调递增即可(单队列,无乱序;多路复用在 §4 预留)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RequestId(u64);

// ============================================================================
// §2 语义操作(线格式的载荷;封闭枚举,不感知泛型)
// ============================================================================

/// 语义请求:一个"想对 GPU 做的事"。
/// 新能力 = 新变体;client 门面方法与变体一一对应(见 client.rs)。
pub(crate) enum Op {
    // ---- 生命周期 ----
    /// 全设备同步(排空):完成即回
    Sync,
    /// 关机:server 排空在途请求后退出
    Close,

    // ---- 内存面(server 内部池账房;client 见到的是 Bytes 句柄)----
    /// 池字节分配(清零;Capturing 相允许 = 捕获安全分配)
    Malloc { pool: PoolRef, bytes: u64 },
    /// host → device 写入式分配(权重装载;server 侧保证池流 async+sync)
    Htod { pool: PoolRef, src: HostBytes },
    /// 归还(显式;日常靠句柄 drop 的延迟归还,A1.2)
    Free { block: BlockRef },

    // ---- 发射面 ----
    /// kernel 发射(裸指针 + 标量参数;参数封在 KernelArgs 里,协议不拆)
    Launch {
        kernel: KernelRef,
        args: KernelArgs,
        grid: GridDim,
        block: BlockDim,
    },
    /// 图回放(档一:提交即回;租约校验在 server 侧 launch 内)
    Replay { graph: GraphRef },

    // ---- 捕获事务 ----
    /// 捕获开始(原子事务的第一条消息;窗内后续 = Launch 流)
    CaptureBegin { plan: CapturePlan },
    /// 捕获结束 + 实例化(返回图句柄;窗内发射数 > 0 由 server 校验,坑 C)
    CaptureEnd,

    // ---- A2 期占位(语义位,不实现)----
    // HtodPinned { pool: PoolRef, staging: PinnedRef },  // pinned 直填装载
    // CopyD2D { src: BlockRef, dst: BlockRef },          // bindings 装填
}

/// 语义应答。载荷 = 协议感知的最大粒度(句柄/字节/单位量)。
pub(crate) enum Reply {
    Ok,
    /// 分配/装载产生的池块句柄(client 侧张量的存储底座)
    Block(BlockRef),
    /// host 字节(D2H 读回)
    HostBytes(Vec<u8>),
    /// 捕获产物(图句柄)
    Graph(GraphRef),
    /// 结构化错误(违约/耗尽/非法态;server 永不 panic,错误过线)
    Err(BackendError),
}

// ============================================================================
// §3 线格式(envelope)
// ============================================================================

/// 请求信封:id + 载荷。server 按信封回信。
pub(crate) struct Request {
    pub(crate) id: RequestId,
    pub(crate) op: Op,
}

/// 应答信封:id 原样带回(配对钥匙)。
pub(crate) struct Response {
    pub(crate) id: RequestId,
    pub(crate) reply: Reply,
}

// ============================================================================
// §4 协议客户端(封装关联机制;client 门面的内部器官)
// ============================================================================

/// 协议端:管理"在途请求表"(id → 完成通知的桥)。
/// client 门面的每个语义方法 = 构造 Op + [`ProtocolClient::call`]。
pub(crate) struct ProtocolClient {
    /// 请求出线(server 消费)
    tx: RequestSender,
    /// 在途表:RequestId → 完成桥(oneshot)。
    /// 插入发生在 call() 出队前,拔除发生在响应配对处——
    /// 关机时未完成的在途项统一以 Err 拔除(design §三 Closing)。
    pending: PendingMap,
    next_id: u64,
}

impl ProtocolClient {
    /// 出队一个请求:登记在途表 → 发信封 → 返回应答接收端。
    /// (异步;server 侧什么时候消化与 client 无关。)
    // PSEUDO: fn send(&mut self, op: Op) -> ResponseFuture {
    // PSEUDO:     let id = RequestId(self.next_id); self.next_id += 1;
    // PSEUDO:     let (ack, rx) = oneshot::channel();
    // PSEUDO:     self.pending.insert(id, ack);
    // PSEUDO:     self.tx.send(Request { id, op });            // await 出队
    // PSEUDO:     ResponseFuture { rx, protocol: self.split() } // 响应到达时配对拔表
    // PSEUDO: }

    /// 关机时的对账:所有在途请求以 Err 拔除(调用方拿到结构化关闭错误,
    /// 而不是悬挂)。
    // PSEUDO: fn drain_pending_with_err(&mut self, why: &str) { ... }
}

/// 应答 future:响应到达 → 拔表 → 把 Reply 交给门面的类型桥。
pub(crate) struct ResponseFuture {
    /* rx: oneshot::Receiver<Response> */
}

// ============================================================================
// §5 传输端点(server 侧消费信封;类型是通道,不是语义)
// ============================================================================

pub(crate) type RequestSender = mpsc_sender();
pub(crate) type RequestReceiver = mpsc_receiver();

// 伪代码辅助标记(实施时替换为 tokio::sync::mpsc)
#[allow(non_snake_case)]
fn mpsc_sender() -> RequestSenderPlaceholder {
    unimplemented!()
}
#[allow(non_snake_case)]
fn mpsc_receiver() -> RequestSenderPlaceholder {
    unimplemented!()
}
struct RequestSenderPlaceholder;
struct RequestSenderPlaceholderAlias;

// ============================================================================
// §6 句柄词汇(协议感知的引用;具体类型随重写落定)
// ============================================================================

/// 池引用(命名池;server 内部账房键)
pub(crate) struct PoolRef(pub(crate) u64);
/// 池块句柄(client 侧张量的存储底座;drop = 延迟归还,A1.2)
pub(crate) struct BlockRef(pub(crate) u64);
/// 图句柄(捕获产物)
pub(crate) struct GraphRef(pub(crate) u64);
/// kernel 引用(nvrtc 编译产物;装载命令 A2 期补:LoadKernel)
pub(crate) struct KernelRef(pub(crate) u64);
/// kernel 参数(裸指针 + 标量;协议不拆,server 原样递给发射面)
pub(crate) struct KernelArgs(pub(crate) Vec<u64>);
pub(crate) struct GridDim(pub(crate) (u32, u32, u32));
pub(crate) struct BlockDim(pub(crate) (u32, u32, u32));
/// host 字节(契约 3:生产形态 = pinned 仓;一期 Vec 起步,落点见 loader)
pub(crate) struct HostBytes(pub(crate) Vec<u8>);
/// 捕获事务声明(窗内发射体 + 预绑定;详见 server.rs CapturePlan)
pub(crate) struct CapturePlan {
    _placeholder: (),
}
