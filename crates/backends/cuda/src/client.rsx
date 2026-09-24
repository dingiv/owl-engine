//! Client:GPU server 的异步门面(Tokio 面;对外唯一入口)。
//!
//! ⚠️ 伪代码骨架(2026-09-23;async-runtime.md §四门面 + §五完成语义)。
//!
//! 契约(设计 §二/§四):
//! - client **不感知 protocol**:语义方法 → 门面内部构造 Op(经 protocol 层);
//!   换传输/换编解码,本文件的业务签名一个都不变;
//! - client **不持有设备执行权**(Arc<CudaDevice> 不存在于此世界);
//! - 阻塞面全部经 server 消化;await 侧零阻塞;
//! - 泛型 R 不进协议:门面用私有 oneshot 桥(命令回执只承载"被消化")。
//!
//! 方法面 = 语义动词,与 protocol::Op 一一对应(新增能力 = 两处各一行)。

use crate::protocol::{
    BlockRef, CapturePlan, GraphRef, HostBytes, KernelArgs, KernelRef, PoolRef, ProtocolClient,
    RequestSender, Reply,
};
use owl_iface::BackendError;
use std::sync::Arc;

// ============================================================================
// §1 门面本体
// ============================================================================

/// 异步 GPU 门面:Clone 便宜;多任务共享同一 server(单队列)。
pub struct Client {
    proto: Arc<tokio_sync_mutex_placeholder::ProtocolClientInner>,
}

// 伪代码辅助标记(实施时删除):ProtocolClient 需要 &mut 调 send,
// 门面多任务共享 → 内包 tokio Mutex;队列本身就是串行点,锁不热。
mod tokio_sync_mutex_placeholder {
    pub type ProtocolClientInner = crate::protocol::ProtocolClient;
}

impl Client {
    // ======================================================================
    // §1.1 生命周期
    // ======================================================================

    /// 启动 server 并返回门面(uuid 钉卡;pool_bytes = 默认池容量)。
    /// Booting 握手完成(Ready)后才返回——调用方拿到的门面即刻可用。
    // PSEUDO: pub fn spawn(uuid: &str, pool_bytes: u64) -> Result<Self, BackendError>
    // PSEUDO:     /* = server.rs §4 启动接线;client 只见结果 */

    /// 关机:server 排空在途请求 → 资源清算 → 线程退出。
    /// 在途请求的调用方各收到结构化关闭错误(协议层 drain 兜底)。
    pub async fn close(&self) -> Result<(), BackendError> {
        self.proto_call(Op_::Close).await?.into_ok()
    }

    // ======================================================================
    // §1.2 内存面(句柄进出;字节面唯一,池去类型裁决)
    // ======================================================================

    /// 池字节分配(清零)。Capturing 相允许(捕获安全分配)。
    pub async fn malloc(&self, pool: PoolRef, bytes: u64) -> Result<BlockRef, BackendError> {
        self.proto_call(Op_::Malloc { pool, bytes })
            .await?
            .into_block()
    }

    /// host → device 写入式分配(权重装载;server 侧池流 async+sync)。
    pub async fn htod(&self, pool: PoolRef, src: HostBytes) -> Result<BlockRef, BackendError> {
        self.proto_call(Op_::Htod { pool, src })
            .await?
            .into_block()
    }

    /// 显式归还(日常靠 BlockRef drop 的延迟归还,A1.2)。
    pub async fn free(&self, block: BlockRef) -> Result<(), BackendError> {
        self.proto_call(Op_::Free { block }).await?.into_ok()
    }

    // ======================================================================
    // §1.3 发射面
    // ======================================================================

    /// kernel 发射(提交即回:server 发射成功即回执;完成通知 = 档二)。
    /// 参数为裸指针/标量(u64 面)——调用方保证指针活性(块句柄在侧)。
    pub async fn launch(
        &self,
        kernel: KernelRef,
        args: KernelArgs,
        grid: Grid,
        block: Block,
    ) -> Result<(), BackendError> {
        self.proto_call(Op_::Launch {
            kernel,
            args,
            grid: grid.into(),
            block: block.into(),
        })
        .await?
        .into_ok()
    }

    /// 图回放(提交即回;租约校验在 server 侧 launch 内,失效 = 结构化报错)。
    pub async fn replay(&self, graph: GraphRef) -> Result<(), BackendError> {
        self.proto_call(Op_::Replay { graph }).await?.into_ok()
    }

    // ======================================================================
    // §1.4 捕获事务(多消息事务的 client 侧形态:事务守卫)
    // ======================================================================

    /// 开启捕获事务:返回事务守卫。
    /// 守卫存活期间,launch 的落点自动是捕获流(守卫内可见性,非全局标记);
    /// Drop 时未 end = 自动回滚(discard)——悬挂事务保护(server.rs §三)。
    ///
    /// 窗内纪律(design §四契约 1):守卫方法全部同步,**无 await**;
    /// 窗内只有 frame 发射面,零 memcpy 零旁路。
    pub async fn capture_begin(&self, plan: CapturePlan) -> Result<CaptureGuard, BackendError> {
        self.proto_call(Op_::CaptureBegin { plan: () })
            .await?
            .into_ok()?;
        Ok(CaptureGuard {
            client: self.clone(),
            ended: false,
        })
    }

    /// 结束捕获事务:实例化 + 审计 → 图句柄。
    pub async fn capture_end(&self) -> Result<GraphRef, BackendError> {
        self.proto_call(Op_::CaptureEnd).await?.into_graph()
    }

    // ======================================================================
    // §1.5 同步/查询
    // ======================================================================

    /// 全设备同步(阻塞消化在 server 线程)。
    pub async fn sync(&self) -> Result<(), BackendError> {
        self.proto_call(Op_::Sync).await?.into_ok()
    }
}

/// 捕获事务守卫:窗内发射的唯一合法视野。
/// Drop 未 end → 自动向 server 发回滚(discard),杜绝悬挂窗。
pub struct CaptureGuard {
    client: Client,
    ended: bool,
}

impl CaptureGuard {
    /// 窗内发射(同步;参数面同 Client::launch;落点 = 捕获流)。
    pub fn launch(
        &mut self,
        kernel: KernelRef,
        args: KernelArgs,
        grid: Grid,
        block: Block,
    ) -> Result<(), BackendError> {
        let _ = (kernel, args, grid, block);
        unimplemented!("骨架:直发 server 的窗内发射通道")
    }

    /// 结束事务 → 图句柄(消费守卫)。
    pub async fn end(mut self) -> Result<GraphRef, BackendError> {
        self.ended = true;
        self.client.capture_end().await
    }
}

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        if !self.ended {
            // 骨架:自动回滚(server discard;阻塞回执在 drop 里可接受——
            // 守卫 drop 必须确定性地结束事务,异步化反而引入遗漏面)
        }
    }
}

// ============================================================================
// §2 内部:语义方法 → 协议调用(门面与协议的唯一接缝)
// ============================================================================

// 伪代码辅助标记(实施时删除/替换)
mod tokio_placeholder {
    pub enum Op {
        Close,
        Sync,
        Malloc { pool: PoolAlias, bytes: u64 },
        Htod { pool: PoolAlias, src: HostBytes },
        Free { block: BlockAlias },
        Launch { kernel: (), args: (), grid: (), block: () },
        Replay { graph: () },
        CaptureBegin { plan: () },
        CaptureEnd,
    }
    pub type PoolAlias = crate::protocol::PoolRef;
    pub type BlockAlias = crate::protocol::BlockRef;
    pub type HostBytes = crate::protocol::HostBytes;
}
use tokio_placeholder::Op as Op_;
struct Grid(pub (u32, u32, u32));
struct Block(pub (u32, u32, u32));
impl From<Grid> for crate::protocol::GridDim {
    fn from(g: Grid) -> Self {
        Self(g.0)
    }
}
impl From<Block> for crate::protocol::BlockDim {
    fn from(b: Block) -> Self {
        Self(b.0)
    }
}

impl Client {
    /// 唯一接缝:语义方法 → 协议调用 → Reply 拆包。
    /// **client 业务方法里禁止出现 Op 构造以外的 protocol 词汇。**
    // PSEUDO: async fn proto_call(&self, op: Op_) -> Result<Reply, BackendError> {
    // PSEUDO:     // proto 是 Arc<Mutex<ProtocolClient>>:锁 = 队列串行点(不热)
    // PSEUDO:     let mut proto = self.proto.lock().await;
    // PSEUDO:     proto.send(op).await   // ResponseFuture → Reply
    // PSEUDO: }
}

// Reply 拆包助手(into_ok/into_block/into_graph/into_host_bytes):
// 模式匹配 Reply 变体,错型 = LawViolation("应答型不匹配:期待 X 得 Y")。
// PSEUDO: impl Reply { fn into_ok(self) -> Result<(), BackendError>; ... }
