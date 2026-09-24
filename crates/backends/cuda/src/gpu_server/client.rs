//! GpuClient:客户端异步门面(对外唯一入口)。
//!
//! - 门面不感知设备内部形态(流注册表/账房/码头/编译缓存全在 server 侧);
//! - `submit` 把命令信封投进 server 队列,回执桥(Ack/Waiter)承接完成通知;
//! - server 侧 GPU API 全非阻塞 + 每流 reaper 阻塞收割,故 server 不因 GPU
//!   工作而停摆,客户端命令得以流水线化提交 —— await 的耗时 =
//!   「提交 + 该操作真实完成」,而非「提交 + 全队列串行」;
//! - **流管理**:`new_stream()` 要求 server 签发新流,此后原语带流 id
//!   (流内保序,流间并发;跨流依赖待事件边扩展)。

use crate::gpu_server::command::{Ack, Command, Waiter};
use owl_models::client::{Arg, Bytes, DeviceClient, GraphId, LaunchMsg};
use owl_models::{Dtype, ModelError};
use owl_models::shape::Shape;
use std::future::Future;
use std::sync::mpsc;

#[derive(Clone)]
pub struct GpuClient {
    tx: mpsc::Sender<Command>,
}

impl GpuClient {
    /// client 构造函数:**纯句柄**(管道的 client 端;管道由外部组装者
    /// 创建后分别传入 server 与 client,双方都不自创管道)。
    pub fn new(tx: mpsc::Sender<Command>) -> Self {
        Self { tx }
    }

    /// 投递信封,返回回执 Future(client 侧 `.await`)。
    ///
    /// Future 的 resolve 时机由 server 决定(见模块注释的语义分级):
    /// - launch/alloc:issue 即 send → 立即 resolve(fire-and-forget)
    /// - htod/dtoh/sync:host 回调/栅栏后 send → pending 至 GPU 真完成
    /// pending 期间 waker 挂起,不占执行线程。
    fn submit<T: Send + 'static>(
        &self,
        build: impl FnOnce(Ack<Result<T, ModelError>>) -> Command,
    ) -> Result<impl Future<Output = Result<T, ModelError>>, ModelError> {
        let (ack, waiter): (Ack<_>, Waiter<_>) = Ack::pair();
        self.tx.send(build(ack)).map_err(|_| ModelError::ServerClosed)?;
        Ok(waiter)
    }
}

/// 组装糖(便捷路径):一键建管道 + 起 server 线程 + boot 握手。
/// 手工组装等价于:
/// ```text
/// let (tx, rx) = mpsc::channel();
/// let server = GpuServer::new(rx, ordinal, Some(boot));
/// let client = GpuClient::new(tx);
/// thread::spawn(move || server.run());
/// ```
impl DeviceClient for GpuClient {
    async fn graph_begin(&mut self) -> Result<(), ModelError> {
        self.submit(move |ack| Command::GraphBegin { ack })?.await
    }

    async fn graph_end(&mut self) -> Result<GraphId, ModelError> {
        self.submit(move |ack| Command::GraphEnd { ack })?.await
    }

    async fn graph_launch(&mut self, graph: GraphId) -> Result<(), ModelError> {
        self.submit(move |ack| Command::GraphLaunch { graph, ack })?.await
    }

    async fn alloc(&mut self, n_bytes: usize) -> Result<Bytes, ModelError> {
        let n_elems = n_bytes / 4;
        self.submit(move |ack| Command::Alloc { n_elems, ack })?.await
    }

    async fn htod(
        &mut self,
        _dtype: Dtype,
        _shape: &Shape,
        src: &[u8],
    ) -> Result<Bytes, ModelError> {
        let data: Vec<f32> = src
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        self.submit(move |ack| Command::Htod { data, ack })?.await
    }

    async fn dtoh(&mut self, b: &Bytes, out: &mut [u8]) -> Result<(), ModelError> {
        let id = b.id;
        let want_elems = out.len() / 4;
        let bytes = self.submit(move |ack| Command::Dtoh { id, want_elems, ack })?.await?;
        out.copy_from_slice(&bytes);
        Ok(())
    }

    async fn launch(&mut self, msg: LaunchMsg) -> Result<Bytes, ModelError> {
        // 槽序契约前置校验(与 launch.rs 的"最后一个 Block"回传一致)
        debug_assert!(msg.args.iter().any(|a| matches!(a, Arg::Block { .. })));
        self.submit(move |ack| Command::Launch { msg, ack })?.await
    }

    async fn sync(&mut self) -> Result<(), ModelError> {
        self.submit(move |ack| Command::Sync { ack })?.await
    }
}
