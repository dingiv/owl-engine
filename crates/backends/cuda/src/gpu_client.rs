//! GpuClient:客户端异步门面(对外唯一入口)。
//!
//! - 门面不感知设备内部形态(流注册表/账房/码头/编译缓存全在 server 侧);
//! - `submit` 把命令信封投进 server 队列,回执桥(Ack/Waiter)承接完成通知;
//! - server 侧 GPU API 全非阻塞 + 每流 reaper 阻塞收割,故 server 不因 GPU
//!   工作而停摆,客户端命令得以流水线化提交 —— await 的耗时 =
//!   「提交 + 该操作真实完成」,而非「提交 + 全队列串行」;
//! - **流管理**:`new_stream()` 要求 server 签发新流,此后原语带流 id
//!   (流内保序,流间并发;跨流依赖待事件边扩展)。

use crate::command::{Ack, Command, Waiter};
use crate::server::GpuServer;
use crate::state::DeviceSelector;
use owl_iface::contract::{Arg, Bytes, DeviceClient, GraphId, LaunchMsg};
use owl_iface::contract::{Dtype, ModelError, Shape};
use std::future::Future;
use std::sync::mpsc;

#[derive(Clone)]
pub struct GpuClient {
    tx: mpsc::Sender<Command>,
    /// 诊断下沉口(与 GPU server 同一只;`sink()` 组装时同传双方)
    sink: owl_shared::log::LogSink,
}

/// 计时包封:记录 `submit → Ready` 的**等待**(分相 tag `srv.wait.<kind>`)。
/// 与 `srv.tx.<kind>`(入队)、`srv.op.<kind>`(服务端处理)三段拼起来
/// 就是客户端观测的那一段 —— 差值即唤醒延迟(见 perf-framework §十)。
struct TimedWaiter<T> {
    inner: Waiter<T>,
    t0: Option<std::time::Instant>,
    sink: owl_shared::log::LogSink,
    kind: &'static str,
}

impl<T: Send + 'static> Future for TimedWaiter<T> {
    type Output = T;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<T> {
        // SAFETY:inner 不被移动(结构体自持,pin 投影到字段)
        let this = unsafe { self.get_unchecked_mut() };
        let inner = unsafe { std::pin::Pin::new_unchecked(&mut this.inner) };
        match inner.poll(cx) {
            std::task::Poll::Ready(v) => {
                if let Some(t0) = this.t0.take() {
                    this.sink
                        .timer(&format!("srv.wait.{}", this.kind), t0.elapsed());
                }
                std::task::Poll::Ready(v)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

impl GpuClient {
    /// client 构造函数:**纯句柄**(管道的 client 端;管道由外部组装者
    /// 创建后分别传入 server 与 client,双方都不自创管道)。
    pub fn new(tx: mpsc::Sender<Command>) -> Self {
        Self { tx, sink: owl_shared::log::LogSink::none() }
    }

    /// 带下沉口构造(组装糖/spawn_with 用;两侧同 sink)
    pub fn with_sink(tx: mpsc::Sender<Command>, sink: owl_shared::log::LogSink) -> Self {
        Self { tx, sink }
    }

    /// **块账按名清单**(诊断查询面;2026-10-16 全量记名)。
    ///
    /// 纯读命令:不碰设备/不改块账,捕获期也放行。给 OOM 排障与
    /// 回归门用 —— "谁占着"的唯一直接答案(总量三账只能答"多满")。
    /// `top` = 列多少类名字(字节降序);未命名块单独计数并上屏。
    pub async fn ledger_report(&self, top: usize) -> Result<String, ModelError> {
        self.submit(move |ack| Command::LedgerReport { top, ack })?.await
    }

    /// **显存现场 JSON**(块账 + 家族 scratch;OOM 落盘同一序列化器)。
    /// 纯读、捕获期放行;给脚本/跨时间比对用。
    pub async fn ledger_json(&self) -> Result<String, ModelError> {
        self.submit(move |ack| Command::LedgerJson { ack })?.await
    }

    /// 投递信封,返回回执 Future(client 侧 `.await`)。
    ///
    /// Future 的 resolve 时机由 server 决定(见模块注释的语义分级):
    /// - launch/alloc:issue 即 send → 立即 resolve(fire-and-forget)
    /// - htod/dtoh/sync:host 回调/栅栏后 send → pending 至 GPU 真完成
    ///   pending 期间 waker 挂起,不占执行线程。
    fn submit<T: Send + 'static>(
        &self,
        build: impl FnOnce(Ack<Result<T, ModelError>>) -> Command,
    ) -> Result<impl Future<Output = Result<T, ModelError>>, ModelError> {
        let (ack, waiter): (Ack<_>, Waiter<_>) = Ack::pair();
        let cmd = build(ack);
        let kind = cmd.kind();
        // ① 入队(客户端侧):锁 mpsc + 唤醒服务端线程
        let t_tx = if self.sink.timing() { Some(std::time::Instant::now()) } else { None };
        self.tx.send(cmd).map_err(|_| ModelError::ServerClosed)?;
        if let Some(t0) = t_tx {
            self.sink.timer(&format!("srv.tx.{kind}"), t0.elapsed());
        }
        // ② 等回执(含服务端出队/处理/回执唤醒全程;`srv.wait − srv.op − srv.queue`
        //    = 回执侧的唤醒延迟)
        Ok(TimedWaiter {
            inner: waiter,
            t0: if self.sink.timing() { Some(std::time::Instant::now()) } else { None },
            sink: self.sink,
            kind,
        })
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
impl GpuClient {
    /// 组装糖(便捷路径):一键建管道 + 起 server 线程 + boot 握手。
    /// 手工组装等价于:
    /// ```text
    /// let (tx, rx) = mpsc::channel();
    /// let server = GpuServer::new(rx, selector, Some(boot));
    /// let client = GpuClient::new(tx);
    /// thread::spawn(move || server.run());
    /// ```
    pub fn spawn(selector: DeviceSelector) -> Result<Self, String> {
        Self::spawn_with(selector, crate::state::DiagOpts::default(), false, owl_shared::log::LogSink::none())
    }

    /// 带设备诊断选项组装(显式依赖律入口;engine 经 EngineKnobs 传入)
    pub fn spawn_with(
        selector: DeviceSelector,
        diag: crate::state::DiagOpts,
        debug: bool,
        sink: owl_shared::log::LogSink,
    ) -> Result<Self, String> {
        let (tx, rx) = mpsc::channel::<Command>();
        let (boot_tx, boot_rx) = mpsc::channel::<Result<(), String>>();
        let server = GpuServer::with_diag_sink(rx, selector, Some(boot_tx), diag, debug, sink);
        std::thread::Builder::new()
            .spawn(move || {
                // server 收摊 → 账房/图注册表随 ctx 析构,派发线程排空后退出。
                //
                // ⚠️ **不许丢弃 Err**(2026-10-15;AGENTS 纪律:可失败返回值不可丢弃):
                // 原为 `let _ = server.run();` —— 拒启/致命错误在此静默,叠加当时
                // "握手过早"缺陷后表现为"进程活着、无监听、日志无声"(A13 负例门
                // 实测:54 线程存活 / health 无响应)。现裸 eprintln(不经级别门,
                // 拒启是运维必须看见的事件)。
                if let Err(e) = server.run() {
                    eprintln!("[gpu-client] server 线程退出(Err): {e}");
                }
            })
            .map_err(|e| format!("server 线程启动失败: {e}"))?;
        boot_rx
            .recv()
            .map_err(|e| format!("server boot 通道关闭: {e}"))??;
        Ok(Self::with_sink(tx, sink))
    }

    // ── peer 面(TP2 S1 壳子;设计 `docs/arch/TP2-施工设计.md` §2)────────
    //
    // 客户端与 server 两侧对称落地;执行臂见 `server/peer.rs`(现为结构化桩)。
    // 客户端**不吞错**:桩路径会把 server 的结构化错误原样透传给调用方。

    /// **PeerShared 分配**(S1):VMM 拼块 + 本地 RW 授予;
    /// 返回 `(块句柄, 对端可导入凭据)`,凭据交给对端的 [`Self::peer_import`]。
    ///
    /// 单笔额度门在 server 侧(BAR1−64MiB;硬上限 192MiB)——**超限响亮拒**。
    pub async fn peer_alloc(
        &self,
        bytes: usize,
        name: impl Into<String>,
    ) -> Result<(Bytes, Vec<u8>), ModelError> {
        let name = name.into();
        self.submit(move |ack| Command::PeerAlloc { bytes, name, ack })?.await
    }

    /// **对端映射**(S1):把凭据导入为本卡 VA 别名(返回别名块句柄)。
    pub async fn peer_import(
        &self,
        credential: Vec<u8>,
        bytes: usize,
        peer_uuid: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<Bytes, ModelError> {
        let (peer_uuid, name) = (peer_uuid.into(), name.into());
        self.submit(move |ack| Command::PeerImport { credential, bytes, peer_uuid, name, ack })?
            .await
    }

    /// **peer 资源释放**(S1;D2 = 显式释放 + Drop 兜底)。
    pub async fn peer_release(&self, ids: Vec<u64>) -> Result<(), ModelError> {
        self.submit(move |ack| Command::PeerRelease { ids, ack })?.await
    }

    /// 优雅关机:server 进入 Closing —— 排空积压(结构化拒绝)+ 设备栅栏
    /// 后线程退出。本调用回执 = 关机流程已启动并完成排空。
    pub async fn close(&self) -> Result<(), ModelError> {
        self.submit(move |ack| Command::Close { ack })?.await
    }
}

impl DeviceClient for GpuClient {
    /// 执行环境提供(被动律:感知在本侧/boot,kernels 零探测)。
    /// arch 占位 Sm86 —— boot 时从 driver API/CC 查询接真感知,接口零改动。
    fn op_env(&self) -> Option<owl_kernels::driver::OpEnv> {
        Some(owl_kernels::driver::OpEnv {
            hw: owl_kernels::driver::Hw { arch: owl_kernels::Arch::Sm86 },
            page: 0, // 页项由解释器从页策略单源补齐(kv_paged_policy)
            // 几何事实由引擎 boot 从 ModelDims 合入(EnvProvider::op_env);
            // 后端自述面拿不到模型几何 ⇒ 0 = 未知(几何门跳过)
            head_dim: 0,
            grp: 0,
        })
    }

    /// 能力自述:GPU 后端可发射设备 kernel(RepackPath 能力否决项 = true)
    fn device_kernels(&self) -> bool {
        true
    }

    async fn graph_begin(&mut self) -> Result<(), ModelError> {
        self.submit(move |ack| Command::GraphBegin { ack })?.await
    }

    async fn graph_end(&mut self) -> Result<GraphId, ModelError> {
        self.submit(move |ack| Command::GraphEnd { ack })?.await
    }

    async fn graph_launch(&mut self, graph: GraphId) -> Result<(), ModelError> {
        self.submit(move |ack| Command::GraphLaunch { graph, ack })?.await
    }

    async fn graph_abort(&mut self) -> Result<(), ModelError> {
        self.submit(move |ack| Command::GraphAbort { ack })?.await
    }

    async fn graph_destroy(&mut self, graph: GraphId) -> Result<(), ModelError> {
        self.submit(move |ack| Command::GraphDestroy { graph, ack })?.await
    }

    async fn alloc(&mut self, dtype: Dtype, elems: usize, name: &str) -> Result<Bytes, ModelError> {
        let n_bytes = elems * dtype.size_bytes();
        // 事实面:句柄自带 dtype(阶数未定形 = 0,由调用点 `view` 或张量视图定)
        let nm = name.to_string();
        let mut b = self.submit(move |ack| Command::Alloc { n_bytes, elems, zero: true, name: nm, ack })?.await?;
        b.dtype = dtype;
        b.rank = 0;
        Ok(b)
    }

    async fn alloc_uninit(&mut self, dtype: Dtype, elems: usize, name: &str) -> Result<Bytes, ModelError> {
        let n_bytes = elems * dtype.size_bytes();
        let nm = name.to_string();
        let mut b = self.submit(move |ack| Command::Alloc { n_bytes, elems, zero: false, name: nm, ack })?.await?;
        b.dtype = dtype;
        b.rank = 0;
        Ok(b)
    }

    async fn alloc_pinned(
        &mut self,
        bytes: usize,
        name: &str,
    ) -> Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError> {
        let nm = name.to_string();
        self.submit(move |ack| Command::AllocPinned { bytes, name: nm, ack })?.await
    }

    async fn upload_pinned(
        &mut self,
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: &Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> Result<(), ModelError> {
        let dst = dst.clone();
        self.submit(move |ack| Command::UploadPinned {
            buf,
            dst,
            offset_bytes,
            len_bytes,
            ack,
        })?
        .await
    }

    async fn write_block_f32(
        &mut self,
        dst: &Bytes,
        offset_elems: usize,
        data: &[f32],
    ) -> Result<(), ModelError> {
        // f32 标量专用(slots/kv_lens 契约 5 恒 f32);字节化传输入口
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for f in data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        self.submit(move |ack| Command::HtodChunk {
            block: dst.id,
            offset_bytes: offset_elems * 4,
            data: bytes,
            ack,
        })?
        .await
    }

    /// 批量块内写入(S1):decode 每步多槽装填合一次命令/一次往返
    async fn write_blocks_f32(
        &mut self,
        writes: &[(&Bytes, usize, &[f32])],
    ) -> Result<(), ModelError> {
        let writes: Vec<(u64, usize, Vec<u8>)> = writes
            .iter()
            .map(|(b, off, data)| {
                let mut bytes = Vec::with_capacity(data.len() * 4);
                for f in *data {
                    bytes.extend_from_slice(&f.to_le_bytes());
                }
                (b.id, off * 4, bytes)
            })
            .collect();
        self.submit(move |ack| Command::HtodChunks { writes, ack })?
            .await
    }

    fn loader_faces(&self, k: usize) -> Option<Vec<Self>>
    where
        Self: Sized + Clone,
    {
        Some((0..k).map(|_| self.clone()).collect())
    }

    async fn copy_block_at(
        &mut self,
        src: &Bytes,
        src_off_bytes: usize,
        dst: &Bytes,
        dst_off_bytes: usize,
        len_bytes: usize,
    ) -> Result<(), ModelError> {
        let (src, dst) = (src.id, dst.id);
        self.submit(move |ack| {
            Command::CopyBlock { src, src_off_bytes, dst, dst_off_bytes, len_bytes, ack }
        })?
        .await
    }

    fn face_clone(&self) -> Option<Self> {
        Some(self.clone())
    }

    async fn copy_batch(
        &mut self,
        copies: &[(Bytes, usize, Bytes, usize, usize)],
    ) -> Result<(), ModelError> {
        let flat: Vec<(u64, usize, u64, usize, usize)> = copies
            .iter()
            .map(|(s, so, d, doff, l)| (s.id, *so, d.id, *doff, *l))
            .collect();
        self.submit(move |ack| Command::CopyBatch { copies: flat, ack })?.await
    }

    async fn memset_zero(&mut self, dst: &Bytes, len_bytes: usize) -> Result<(), ModelError> {
        let block = dst.id;
        self.submit(move |ack| Command::MemsetZero { block, offset_bytes: 0, len_bytes, ack })?.await
    }

    async fn memset_zero_at(
        &mut self,
        dst: &Bytes,
        offset_bytes: usize,
        len_bytes: usize,
    ) -> Result<(), ModelError> {
        let block = dst.id;
        self.submit(move |ack| {
            Command::MemsetZero { block, offset_bytes, len_bytes, ack }
        })?
        .await
    }

    async fn htod(
        &mut self,
        dtype: Dtype,
        shape: &Shape,
        src: &[u8],
        name: &str,
    ) -> Result<Bytes, ModelError> {
        // 字节直传(字节口径);Bytes.len = 元素数,按 shape 回填。
        //
        // ⚠️ **dtype 不再静默忽略**(2026-10-15 审查补):旧实现 `let _ = dtype;`
        // ⇒ 调用方传错 dtype(非 f16 基线,如 U32 打包权重)时 `Bytes.len`
        // 语义错而**无声**。这里做**交叉校验**(不拒非 F16 —— U32 是合法用法):
        // `src` 字节数必须 ≥ `elems × dtype 字节宽`(多给容许:池块可含尾料)。
        let elems: usize = shape.iter().product();
        let nm = name.to_string();
        let want = elems
            .checked_mul(dtype.size_bytes())
            .ok_or_else(|| ModelError::Msg(format!("htod: elems({elems})×dtype({dtype:?}) 溢出")))?;
        if src.len() < want {
            return Err(ModelError::Msg(format!(
                "htod: src {}B < elems({elems})×dtype({dtype:?})={want}B —— dtype/shape 与字节数不一致",
                src.len()
            )));
        }
        let mut b = self
            .submit(move |ack| Command::Htod { data: src.to_vec(), name: nm, ack })?
            .await?;
        // 事实面定形:服务端只回句柄(U8/未定形),dtype + 阶数由本面按 (dtype, shape) 回填
        b.len = elems;
        b.dtype = dtype;
        b.rank = shape.len() as u8;
        Ok(b)
    }

    async fn dtoh(&mut self, b: &Bytes, out: &mut [u8]) -> Result<(), ModelError> {
        let id = b.id;
        let want_bytes = out.len();
        let bytes = self.submit(move |ack| Command::Dtoh { id, want_bytes, ack })?.await?;
        out.copy_from_slice(&bytes);
        Ok(())
    }

    // TODO: 考虑改成同步调用, 因为 launch 不用等待, 待定
    async fn launch(&mut self, msg: LaunchMsg) -> Result<Bytes, ModelError> {
        // 槽序契约前置校验(与 launch.rs 的"最后一个 Block"回传一致)
        debug_assert!(msg.args.iter().any(|a| matches!(a, Arg::Block { .. })));
        self.submit(move |ack| Command::Launch { msg, ack })?.await
    }

    /// **块账本投影**(调试面;默认 trait 口"不支持" ⇒ 这里给 GPU 实现)
    async fn block_ledger_dump(
        &mut self,
    ) -> Result<Vec<owl_iface::contract::BlockLedgerRow>, ModelError> {
        self.submit(move |ack| Command::BlockLedgerDump { ack })?.await
    }

    async fn sync(&mut self) -> Result<(), ModelError> {
        self.submit(move |ack| Command::Sync { ack })?.await
    }

    /// **设备池回收**(装载收尾;默认 trait 口"不支持" ⇒ 这里给 GPU 实现)
    async fn mem_pool_trim(&mut self) -> Result<u64, ModelError> {
        self.submit(move |ack| Command::MemPoolTrim { ack })?.await
    }

    /// 中间块回收(E2a):投递 Free 命令,账房移除 Owned 块归池
    async fn free(&mut self, ids: &[u64]) -> Result<(), ModelError> {
        let ids = ids.to_vec();
        self.submit(move |ack| Command::Free { ids, ack })?.await
    }

    /// **上传并计算**:一次 pinned 搬运 + 一次发射,序由 server 内部保证。
    /// (E2a:线格式/校验/负门在位;接线 = E2b —— 目前 server 侧响亮拒。)
    async fn upload_and_launch(
        &mut self,
        dst: &Bytes,
        chunks: Vec<owl_iface::contract::UpChunk>,
        msg: LaunchMsg,
    ) -> Result<Bytes, ModelError> {
        let dst = dst.clone();
        self.submit(move |ack| Command::UploadAndLaunch { dst, chunks, msg, ack })?
            .await
    }

    /// **计算并回读**:一次发射 + 一次回读(`id` 必须是本 launch 的输出块;
    /// 该校验在 server 侧 = E2b)。
    async fn launch_and_harvest(
        &mut self,
        msg: LaunchMsg,
        id: u64,
        want_bytes: usize,
    ) -> Result<Vec<u8>, ModelError> {
        self.submit(move |ack| Command::LaunchAndHarvest { msg, id, want_bytes, ack })?
            .await
    }
}
