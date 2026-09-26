//! GpuServer:事件循环(单线程设备执行权唯一宿主)+ host 完成回调派发。
//!
//! 构造与启动分离:`new()` 是纯构造(零 CUDA 调用,不碰设备),
//! `run()` 在**调用它的线程上**上线设备(bind_to_thread)并进入事件循环
//! —— 因此 `run()` 必须跑在专职线程上(由外部组装者负责起线程)。
//!
//! 循环骨架:
//! ```text
//! actor 线程(唯一触碰设备)                派发线程(1 个,server 内部)
//!   loop {                                  loop {
//!     1. rx.recv() 阻塞等信封                 1. channel 收 finish 闭包
//!     2. dispatch:issue 相(全非阻塞,        2. 执行(放 pinned 码头、
//!        提交进流即返回)                          回执客户端;可自由调 CUDA)
//!     3. launch_host_function 排一个             }
//!        host 回调 → 前面工作全完成时
//!        由驱动线程触发 → 只投递 finish
//!   }
//! ```
//!
//! **完成通知 = 推(push)**:现代 API `cuLaunchHostFunc`(取代废弃的
//! `cudaStreamAddCallback`)——工作完成后**驱动主动调用**我们的回调,
//! 无轮询、无阻塞收割线程。回调跑在驱动线程,禁止调 CUDA/阻塞,所以只做
//! channel 投递;真正的 finish(含 free_host 等设备操作)由派发线程执行。
//!
//! **流语义**:客户端可要求签发新流(NewStream → StreamId);此后每条
//! 原语带流 id —— 流内保序(依赖维),流间并发。

use crate::ffi::{launch_host_function, memcpy_dtoh_async, memcpy_htod_async, memset_d8_async};
use crate::command::{Ack, Command};
use crate::launch::issue_launch;
use crate::state::{DeviceSelector, GpuCtx, KernelCache, Staging};
use owl_iface::contract::{Arg, Bytes, LaunchMsg};
use crate::state::{STREAM_COMPUTE, STREAM_D2H, STREAM_H2D};
use owl_iface::contract::ModelError;
use std::sync::mpsc;

type Finish = Box<dyn FnOnce() + Send>;

/// pinned 租约池(server 私有;Arc 随 dispatch 线程共持 ——
/// 生命周期严格罩在 CUDA ctx 之内,server 关闭即整池释放,
/// 杜绝跨 server 的悬垂页锁指针段错误)
#[derive(Default)]
pub(super) struct PinnedPool {
    free: std::sync::Mutex<Vec<Box<dyn owl_iface::contract::PinnedRegion + Send>>>,
}

impl PinnedPool {
    fn take(&self, min_elems: usize) -> Option<Box<dyn owl_iface::contract::PinnedRegion + Send>> {
        let mut v = self.free.lock().unwrap();
        let i = v.iter().position(|b| b.as_f32().len() >= min_elems)?;
        Some(v.swap_remove(i))
    }
    fn put(&self, b: Box<dyn owl_iface::contract::PinnedRegion + Send>) {
        self.free.lock().unwrap().push(b);
    }
}

pub struct GpuServer {
    rx: mpsc::Receiver<Command>,
    selector: DeviceSelector,
    pool: std::sync::Arc<PinnedPool>,
    /// 上线握手(可选;便捷组装路径用它回传设备上线结果)
    boot: Option<mpsc::Sender<Result<(), String>>>,
    ctx: Option<GpuCtx>,
    kernels: KernelCache,
    /// cuBLAS 封装(foreign-kernel 通道;算子之家 owl-kernels::cublas,懒初始化)
    blas: Option<owl_kernels::cublas::OwlCublas>,
    /// 完成派发出口(host 回调只投递;派发线程执行真正的 finish)
    dispatch: Option<mpsc::Sender<Finish>>,
}

impl GpuServer {
    /// server 构造函数:**纯构造,零 CUDA 调用**(设备上下文在 `run()` 时
    /// 于 run 所在线程上线 —— bind_to_thread 的线程亲和性要求如此)。
    ///
    /// - `rx`:命令管道的 server 端(管道由外部组装者创建,分别传入
    ///   server 与 client;server 不自创管道)
    /// - `selector`:设备选择器(UUID 钉卡推荐;数字序事故免疫)
    /// - `boot`:上线握手(可选;`None` = 外部不需要启动确认)。
    ///   用 mpsc 同步通道(boot 发生在 server 线程,组装者在另一线程同步等)。
    pub fn new(
        rx: mpsc::Receiver<Command>,
        selector: DeviceSelector,
        boot: Option<mpsc::Sender<Result<(), String>>>,
    ) -> Self {
        Self {
            rx,
            selector,
            pool: std::sync::Arc::new(PinnedPool::default()),
            boot,
            ctx: None,
            kernels: KernelCache::new(),
            blas: None,
            dispatch: None,
        }
    }

    /// 事件循环主入口(server 线程生命周期 = 循环生命周期)。
    /// 首行上线设备(本线程 bind_to_thread 一次到位)+ 起派发线程。
    pub fn run(mut self) -> Result<(), String> {
        let ctx = GpuCtx::new(&self.selector);
        match ctx {
            Ok(c) => {
                if let Some(boot) = self.boot.take() {
                    let _ = boot.send(Ok(()));
                }
                self.ctx = Some(c);
            }
            Err(e) => {
                if let Some(boot) = self.boot.take() {
                    let _ = boot.send(Err(e.clone()));
                }
                return Err(e);
            }
        }

        // 派发线程:host 回调只投递,真正 finish(含 free_host 等 CUDA 操作)在这跑
        let (dtx, drx) = mpsc::channel::<Finish>();
        std::thread::Builder::new()
            .name(format!("owl-gpu-{}-dispatch", self.selector_debug()))
            .spawn(move || {
                for finish in drx {
                    finish();
                }
            })
            .map_err(|e| format!("派发线程启动失败: {e}"))?;
        self.dispatch = Some(dtx);

        // 纯阻塞监听:零轮询。生命周期:
        // - Ready:正常 dispatch
        // - Close 命令 → Closing 态:此后一切命令结构化拒绝(ServerClosed),
        //   直到全部客户端离场(Disconnected)→ 设备栅栏 → 收摊
        // 在飞 finish 由派发线程排空(channel 关闭后自然收尾)。
        let mut closing = false;
        loop {
            let cmd = match self.rx.recv() {
                Ok(c) => c,
                Err(_) => break, // 全部客户端离场
            };
            match cmd {
                Command::Close { ack } if !closing => {
                    ack.send(Ok(()));
                    closing = true;
                }
                other if closing => {
                    // Closing 态:拒绝必须用 ServerClosed(客户端可程序化识别)
                    Self::reject_closed(other);
                }
                other => self.dispatch(other),
            }
        }
        self.shutdown_fence();
        Ok(())
    }

    fn selector_debug(&self) -> String {
        match &self.selector {
            DeviceSelector::Uuid(u) => format!("uuid-{:02x}{:02x}…", u[0], u[1]),
            DeviceSelector::Ordinal(o) => format!("ordinal-{o}"),
        }
    }

    /// Closing 收尾:设备栅栏(三条流全部落定,在飞搬运/kernel 不悬空)
    /// → 关派发通道(派发线程排完在飞 finish 后自然退出)。
    fn shutdown_fence(&mut self) {
        if let Some(ctx) = &self.ctx {
            for sid in [STREAM_H2D, STREAM_COMPUTE, STREAM_D2H] {
                if let Ok(s) = ctx.stream(sid) {
                    let _ = s.synchronize();
                }
            }
        }
        drop(self.dispatch.take());
    }

    fn ctx(&self) -> &GpuCtx {
        self.ctx.as_ref().expect("run() 已上线设备")
    }

    fn ctx_mut(&mut self) -> &mut GpuCtx {
        self.ctx.as_mut().expect("run() 已上线设备")
    }

    // ======================================================================
    // dispatch:issue 相(全非阻塞;GPU API 提交进流即返回)
    // ======================================================================

    fn dispatch(&mut self, cmd: Command) {
        // G2 护栏:捕获期间仅放行「Launch / Alloc(切 slab)」—— 二者由
        // 路由策略天然落在捕获流(COMPUTE);其余(搬运/同步/图操作)一律
        // 结构化拒绝 —— 它们或含非法捕获语义,或污染捕获拓扑。
        if self.ctx().capture_stream() {
            return match cmd {
                Command::Launch { msg, ack } => self.handle_launch(msg, ack),
                Command::Alloc { n_bytes, elems, ack } => self.handle_alloc(n_bytes, elems, ack),
                Command::GraphEnd { ack } => ack.send(self.ctx_mut().graph_end()),
                Command::Close { ack } => ack.send(Ok(())), // 防御性幂等回执
                other => Self::reject(other),
            };
        }

        match cmd {
            Command::Alloc { n_bytes, elems, ack } => self.handle_alloc(n_bytes, elems, ack),
            Command::Htod { data, ack } => self.handle_htod(data, ack),
            Command::HtodChunk { block, offset_bytes, data, ack } => {
                self.handle_htod_chunk(block, offset_bytes, data, ack)
            }
            Command::AllocPinned { elems, ack } => self.handle_alloc_pinned(elems, ack),
            Command::UploadPinned { buf, dst, offset_elems, elems, ack } => {
                self.handle_upload_pinned(buf, dst, offset_elems, elems, ack)
            }
            Command::Dtoh { id, want_bytes, ack } => self.handle_dtoh(id, want_bytes, ack),
            Command::Launch { msg, ack } => self.handle_launch(msg, ack),
            Command::Sync { ack } => self.handle_sync(ack),
            Command::GraphBegin { ack } => ack.send(self.ctx_mut().graph_begin()),
            Command::GraphEnd { ack } => ack.send(self.ctx_mut().graph_end()),
            Command::GraphLaunch { graph, ack } => {
                ack.send(self.ctx_mut().graph_launch(graph))
            }
            // 理论不可达:run() 顶层已截获 Close;防御性回执
            Command::Close { ack } => ack.send(Ok(())),
        }
    }

    /// 捕获期违规命令的结构化拒绝
    fn reject(cmd: Command) {
        Self::reject_with(cmd, "图捕获进行中:该操作被拒 —— 捕获期仅允许 \
             Launch/Alloc;同步/搬运/图操作会破坏图捕获".to_string());
    }

    /// Closing 态拒绝:每条命令的 ack 都必须回 ServerClosed
    /// (否则客户端 Future 永远 pending —— 本轮挂死根因)
    fn reject_closed(cmd: Command) {
        macro_rules! closed { ($ack:expr) => { $ack.send(Err(ModelError::ServerClosed)) } }
        match cmd {
            Command::Close { ack } => ack.send(Ok(())), // 幂等
            Command::Alloc { ack, .. } => closed!(ack),
            Command::Htod { ack, .. } => closed!(ack),
            Command::HtodChunk { ack, .. } => closed!(ack),
            Command::AllocPinned { ack, .. } => closed!(ack),
            Command::UploadPinned { ack, .. } => closed!(ack),
            Command::Dtoh { ack, .. } => closed!(ack),
            Command::Launch { ack, .. } => closed!(ack),
            Command::Sync { ack, .. } => closed!(ack),
            Command::GraphBegin { ack, .. } => closed!(ack),
            Command::GraphEnd { ack, .. } => closed!(ack),
            Command::GraphLaunch { ack, .. } => closed!(ack),
        }
    }

    fn reject_with(cmd: Command, why: String) {
        macro_rules! reject { ($ack:expr) => { $ack.send(Err(ModelError::Msg(why.clone()))) } }
        let name = |c: &Command| match c {
            Command::GraphBegin { .. } => "GraphBegin",
            Command::GraphEnd { .. } => "GraphEnd",
            Command::GraphLaunch { .. } => "GraphLaunch",
            Command::Alloc { .. } => "Alloc",
            Command::Htod { .. } => "Htod",
            Command::HtodChunk { .. } => "HtodChunk",
            Command::AllocPinned { .. } => "AllocPinned",
            Command::UploadPinned { .. } => "UploadPinned",
            Command::Dtoh { .. } => "Dtoh",
            Command::Sync { .. } => "Sync",
            Command::Launch { .. } => "Launch",
            Command::Close { .. } => "Close",
        };
        eprintln!("[owl-gpu] 捕获期拒绝: {} —— {why}", name(&cmd));
        match cmd {
            Command::GraphBegin { ack } => reject!(ack),
            Command::GraphEnd { ack } => reject!(ack),
            Command::GraphLaunch { ack, .. } => reject!(ack),
            Command::Alloc { ack, .. } => reject!(ack),
            Command::Htod { ack, .. } => reject!(ack),
            Command::HtodChunk { ack, .. } => reject!(ack),
            Command::AllocPinned { ack, .. } => reject!(ack),
            Command::UploadPinned { ack, .. } => reject!(ack),
            Command::Dtoh { ack, .. } => reject!(ack),
            Command::Sync { ack, .. } => reject!(ack),
            Command::Launch { ack, .. } => reject!(ack),
            Command::Close { ack } => ack.send(Ok(())), // 幂等
        }
    }

    fn handle_alloc(&mut self, n_bytes: usize, elems: usize, ack: Ack<Result<Bytes, ModelError>>) {
        // 捕获期:从 slab 切块(零 cudaMalloc;malloc 在捕获窗内非法)。
        // 切完必须流序清零(N4):块内容 = slab 残留,不清零则 Zeros 语义
        // / 部分写 kernel 静默踩垃圾;memset 可捕获 → replay 时重清零,
        // 语义恒成立。
        if self.ctx().capture_stream() {
            let result = (|| {
                let id = self.ctx_mut().carve_block(n_bytes)?;
                let stream = self.ctx().stream(STREAM_COMPUTE)?.clone();
                let (dptr, _) = self.ctx().block_ptr(id, &stream)?;
                unsafe { memset_d8_async(dptr, 0, n_bytes, stream.cu_stream()) }
                    .map_err(|e| ModelError::Msg(format!("carve memset: {e:?}")))?;
                Ok(Bytes::new(id, elems))
            })();
            return ack.send(result);
        }
        // 图外:路由 COMPUTE 流;alloc 后流序清零(契约:Zeros = 清零分配)
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let result = unsafe { stream.alloc::<u8>(n_bytes) }
            .map_err(|e| ModelError::Msg(format!("alloc: {e:?}")))
            .and_then(|mut slice| {
                stream
                    .memset_zeros(&mut slice)
                    .map_err(|e| ModelError::Msg(format!("alloc memset: {e:?}")))?;
                Ok(Bytes::new(self.ctx_mut().new_block(slice), elems))
            });
        ack.send(result);
    }

    fn handle_htod(&mut self, data: Vec<u8>, ack: Ack<Result<Bytes, ModelError>>) {
        eprintln!("[srv] Htod {} bytes", data.len());
        let mut ack = Some(ack);
        let n = data.len();
        let stream = match self.ctx().stream(STREAM_H2D) {
            Ok(s) => s.clone(),
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        match self.try_htod(&stream, &data) {
            Ok((id, staging)) => {
                // 码头随 finish 存活至搬运完成后由派发线程释放
                let mut cb_ack = ack.take();
                let finish: Finish = Box::new(move || {
                    drop(staging);
                    cb_ack.take().unwrap().send(Ok(Bytes::new(id, n)));
                });
                if let Err(e) = self.notify(&stream, finish) {
                    ack.take().unwrap().send(Err(e));
                }
            }
            Err(e) => ack.take().unwrap().send(Err(e)),
        }
    }

    /// htod issue 相(非阻塞):设备块 + pinned 码头 + 异步 memcpy。
    /// 返回 (块 id, 码头) —— 码头随 finish 存活至搬运完成。
    fn try_htod(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        data: &[u8],
    ) -> Result<(u64, Staging), ModelError> {
        let n = data.len();
        // 1. 设备块(流序 malloc;未初始化;字节口径)
        let dst = unsafe { stream.alloc::<u8>(n) }
            .map_err(|e| ModelError::Msg(format!("htod alloc: {e:?}")))?;
        let id = self.ctx_mut().new_block(dst);
        // 2. pinned 码头 + 异步 memcpy(流序;host 侧必须 pinned 才真异步)
        let mut staging = Staging::alloc(n)?;
        staging.slice_mut().copy_from_slice(data);
        let (dptr, _) = self.ctx().block_ptr(id, stream)?;
        unsafe { memcpy_htod_async(dptr, staging.slice(), stream.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("htod async: {e:?}")))?;
        Ok((id, staging))
    }

    /// 分配 pinned 租约:池优先,miss 才 cudaHostAlloc
    fn handle_alloc_pinned(
        &mut self,
        elems: usize,
        ack: Ack<Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError>>,
    ) {
        eprintln!("[srv] AllocPinned {elems}");
        // 池取(内部单次加锁;容量 ≥ 请求即命中,逻辑长度由调用方界定)
        if let Some(mut b) = self.pool.take(elems) {
            b.slice_mut()[..elems].fill(0.0);
            // 容量不可截,整块交出(调用方只用前 elems)
            ack.send(Ok(b));
            return;
        }
        // f32 租约口径(pinned 流水线仅剩 f32 装载路径;字节底层)
        match Staging::alloc(elems * 4) {
            Ok(s) => ack.send(Ok(Box::new(s))),
            Err(e) => ack.send(Err(e)),
        }
    }

    /// 上传租约:buf 所有权移入,DMA 到 dst+offset;完成回调把 buf 归还池
    fn handle_upload_pinned(
        &mut self,
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: Bytes,
        offset_elems: usize,
        elems: usize,
        ack: Ack<Result<(), ModelError>>,
    ) {
        eprintln!("[srv] UploadPinned block{} off{} elems{}", dst.id, offset_elems, elems);
        let stream = match self.ctx().stream(STREAM_H2D) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ack = Some(ack);
        let dptr_base = match self.ctx().block_ptr(dst.id, &stream) {
            Ok((p, _)) => p,
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        if let Err(e) = unsafe {
            memcpy_htod_async(
                dptr_base + (offset_elems as u64) * 4,
                &buf.as_f32()[..elems],
                stream.cu_stream(),
            )
        } {
            // 建场失败:未入队任何 DMA,finish 不得投递 —— 租约回池 + 回执错误
            let e = ModelError::Msg(format!("upload_pinned async: {e:?}"));
            self.pool.put(buf);
            ack.take().unwrap().send(Err(e));
            return;
        }
        let cell = std::sync::Arc::new(std::sync::Mutex::new(Some((buf, ack))));
        let cell2 = cell.clone();
        let pool = self.pool.clone();
        let pool_err = self.pool.clone();
        let finish: Finish = Box::new(move || {
            let (buf, mut ack) = cell2.lock().unwrap().take().unwrap();
            pool.put(buf);
            ack.take().unwrap().send(Ok(()));
        });
        if let Err(e) = self.notify(&stream, finish) {
            // notify 失败:finish 未投递,租约回收入池 + 回执错误
            if let Some((buf, mut ack)) = cell.lock().unwrap().take() {
                pool_err.put(buf);
                ack.take().unwrap().send(Err(e));
            }
        }
    }

    /// 分块写入已 alloc 的块(流式装载):pinned 码头 + memcpyAsync
    /// 到 dst+offset;完成回调回执
    fn handle_htod_chunk(
        &mut self,
        block: u64,
        offset_bytes: usize,
        data: Vec<u8>,
        ack: Ack<Result<(), ModelError>>,
    ) {
        let stream = match self.ctx().stream(STREAM_H2D) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ack = Some(ack);
        let dptr_base = match self.ctx().block_ptr(block, &stream) {
            Ok((p, _)) => p,
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        let mut staging = match Staging::alloc(data.len()) {
            Ok(s) => s,
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        staging.slice_mut().copy_from_slice(&data);
        unsafe {
            memcpy_htod_async(
                dptr_base + (offset_bytes as u64),
                staging.slice(),
                stream.cu_stream(),
            )
        }
        .map_err(|e| ModelError::Msg(format!("htod chunk async: {e:?}")))
        .map(|_| ())
        .map_err(|e| {
            if let Some(a) = ack.take() {
                a.send(Err(e));
            }
        })
        .err();
        // 码头随 finish 存活至搬运完成后由派发线程释放
        let mut cb_ack = ack.take();
        let finish: Finish = Box::new(move || {
            drop(staging);
            cb_ack.take().unwrap().send(Ok(()));
        });
        if let Err(e) = self.notify(&stream, finish) {
            ack.take().unwrap().send(Err(e));
        }
    }

    fn handle_dtoh(
        &mut self,
        id: u64,
        want_bytes: usize,
        ack: Ack<Result<Vec<u8>, ModelError>>,
    ) {
        // 路由:D2H 流(结果收割维)。先排空 COMPUTE:kernel 在 COMPUTE 流,
        // 本 memcpy 在 D2H 流 —— 跨流无保序,不排空则读块与产出 kernel 竞速
        // (实测:真模型单步中途 dtoh 读到全零块;sync 后重读同块数据完好

        // —— 2026-09-26 塔零案定谳,interpreter-tap.md)。收割路径本就阻塞
        // 等回调,排空无额外代价;htod 侧 async+sync 已内建,无对称问题。
        if let Err(e) = self
            .ctx()
            .stream(STREAM_COMPUTE)
            .and_then(|s| s.synchronize().map_err(|e| ModelError::Msg(format!("sync: {e:?}"))))
        {
            return ack.send(Err(e));
        }
        let stream = match self.ctx().stream(STREAM_D2H) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ack = Some(ack);
        match self.try_dtoh(&stream, id, want_bytes) {
            Ok(staging) => {
                // 完成 → 码头字节直出(传输面字节口径)→ 回执 → 放掉码头
                let mut cb_ack = ack.take();
                let finish: Finish = Box::new(move || {
                    let bytes = staging.slice().to_vec();
                    drop(staging);
                    cb_ack.take().unwrap().send(Ok(bytes));
                });
                if let Err(e) = self.notify(&stream, finish) {
                    ack.take().unwrap().send(Err(e));
                }
            }
            Err(e) => ack.take().unwrap().send(Err(e)),
        }
    }

    /// dtoh issue 相(非阻塞):账长校验 + 异步 memcpy 到 pinned 码头。
    /// 返回码头(随 finish 存活至搬运完成后由派发线程释放)。
    fn try_dtoh(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        id: u64,
        want_bytes: usize,
    ) -> Result<Staging, ModelError> {
        let n = self.ctx().block_len(id)?; // 块账本 = 字节(2026-09-26 f16 基线)
        if n != want_bytes {
            return Err(ModelError::Msg(format!(
                "dtoh: 块 {id} 字节 {n} != 收割 {want_bytes}"
            )));
        }
        let (dptr, _) = self.ctx().block_ptr(id, stream)?;
        let mut staging = Staging::alloc(n)?;
        unsafe { memcpy_dtoh_async(staging.slice_mut(), dptr, stream.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("dtoh async: {e:?}")))?;
        Ok(staging)
    }


    fn handle_launch(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        // foreign-kernel 通道(2026-09-26 合并:cuBLAS 不再另立命令,
        // 外部算子 = 虚拟核名走同一 Launch;谓词与槽序归 owl-kernels::cublas)
        if owl_kernels::cublas::is_foreign(&msg.kernel.name) {
            return self.handle_foreign_launch(msg, ack);
        }
        let mut ack = Some(ack);
        // 字段级解构:ctx(不可变)与 kernels(可变)借用不相交
        let Self { ctx, kernels, .. } = self;
        let ctx = match ctx.as_ref() {
            Some(c) => c,
            None => {
                return ack.take().unwrap().send(Err(ModelError::Msg(
                    "server 未上线".to_string(),
                )))
            }
        };
        let stream = match ctx.stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        // 火后不理:提交成功即回执(输出块句柄 issue 期已确定)。
        // 数据正确性由同流硬件保序保证;GPU 侧错误 sticky 延迟暴露
        // (在下次 dtoh/sync 收割)。不挂 host 回调 —— decode 百级 launch
        // 零回调,流水线不被驱动唤醒打断,并为 CUDA Graph 捕获铺路。
        match issue_launch(ctx, &stream, kernels, &msg) {
            Ok(out_id) => {
                self.ctx_mut().note_launch();
                ack.take()
                    .unwrap()
                    .send(Ok(Bytes::new(out_id, msg.out_elems)))
            }
            Err(e) => ack.take().unwrap().send(Err(e)),
        }
    }

    /// 外部核执行(cuBLAS 先行;marlin/FlashInfer 同通道后续接入)。
    /// 槽序契约见 owl_kernels::cublas::GEMM_F16 文档。
    fn handle_foreign_launch(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        // 捕获期拒绝(外部库 workspace 账外;capture 前须 warmup,README 口径)
        if self
            .ctx
            .as_ref()
            .map(|c| c.capture_stream())
            .unwrap_or(false)
        {
            return ack.send(Err(ModelError::Msg(format!(
                "foreign kernel {} 捕获期不支持(先 warmup;prefill eager 裁决)",
                msg.kernel.name
            ))));
        }
        if self.blas.is_none() {
            let stream = match self.ctx().stream(STREAM_COMPUTE) {
                Ok(s) => s.clone(),
                Err(e) => return ack.send(Err(e)),
            };
            match owl_kernels::cublas::OwlCublas::new(stream) {
                Ok(h) => self.blas = Some(h),
                Err(e) => return ack.send(Err(ModelError::Msg(e))),
            }
        }
        // 槽序:[T a, T b, T out, sz m, sz k, sz n, sz nt]
        let mut blocks: Vec<u64> = Vec::new();
        let mut scalars: Vec<u64> = Vec::new();
        for a in &msg.args {
            match a {
                Arg::Block { id } => blocks.push(*id),
                Arg::U64(v) => scalars.push(*v),
                _ => {
                    return ack.send(Err(ModelError::Msg(format!(
                        "foreign kernel {} 槽序违约:仅 Block/U64(见 cublas.rs 契约)",
                        msg.kernel.name
                    ))))
                }
            }
        }
        if blocks.len() != 3 || scalars.len() != 4 {
            return ack.send(Err(ModelError::Msg(format!(
                "foreign kernel {} 槽序违约:3 Block + 4 U64,得 {}B/{}S",
                msg.kernel.name,
                blocks.len(),
                scalars.len()
            ))));
        }
        let (m, k, n, nt) =
            (scalars[0] as usize, scalars[1] as usize, scalars[2] as usize, scalars[3] != 0);
        let blas = self.blas.as_ref().unwrap();
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let (a_ptr, _) = match self.ctx().block_ptr(blocks[0], &stream) {
            Ok(p) => p,
            Err(e) => return ack.send(Err(e)),
        };
        let (b_ptr, _) = match self.ctx().block_ptr(blocks[1], &stream) {
            Ok(p) => p,
            Err(e) => return ack.send(Err(e)),
        };
        let (out_ptr, _) = match self.ctx().block_ptr(blocks[2], &stream) {
            Ok(p) => p,
            Err(e) => return ack.send(Err(e)),
        };
        match blas.gemm_f16(a_ptr, b_ptr, out_ptr, m, k, n, nt) {
            Ok(()) => ack.send(Ok(Bytes::new(blocks[2], msg.out_elems))),
            Err(e) => ack.send(Err(ModelError::Msg(e))),
        }
    }

    fn handle_sync(&mut self, ack: Ack<Result<(), ModelError>>) {
        // 排空点:三条流全部 synchronize(阻塞 actor 至全部完成;语义栅栏)。
        // host 回调的 finish 由派发线程独立送达,与本栅栏互不依赖。
        let result = (|| {
            for sid in [STREAM_H2D, STREAM_COMPUTE, STREAM_D2H] {
                self.ctx()
                    .stream(sid)?
                    .synchronize()
                    .map_err(|e| ModelError::Msg(format!("sync: {e:?}")))?;
            }
            Ok(())
        })();
        ack.send(result);
    }

    // ======================================================================
    // 完成通知:host 回调(推式;cuLaunchHostFunc)
    // ======================================================================

    /// 在目标流排一个 host 回调:前面的工作全部完成后,驱动线程触发它,
    /// 它只把 finish 闭包投进派发 channel(驱动线程禁止调 CUDA/阻塞)。
    fn notify(&self, stream: &std::sync::Arc<cudarc::driver::CudaStream>, finish: Finish) -> Result<(), ModelError> {
        let dispatch = self
            .dispatch
            .as_ref()
            .ok_or_else(|| ModelError::Msg("notify: 派发线程未上线".to_string()))?
            .clone();
        // 双层 Box:外层薄指针过 FFI;内层闭包在派发线程执行(可调 CUDA)
        let inner: Box<dyn FnOnce() + Send> = Box::new(move || {
            // 驱动线程上:纯 host 动作(channel send 不阻塞、不调 CUDA)
            let _ = dispatch.send(Box::new(finish));
        });
        let boxed: Box<Box<dyn FnOnce() + Send>> = Box::new(inner);
        let raw = Box::into_raw(boxed);
        let rc = unsafe { launch_host_function(stream.cu_stream(), trampoline, raw as *mut _) };
        if let Err(e) = rc {
            // 收回闭包,避免泄漏
            unsafe { drop(Box::from_raw(raw)) };
            return Err(ModelError::Msg(format!("launch_host_function: {e:?}")));
        }
        Ok(())
    }
}

/// FFI 蹦床:还原闭包并执行(驱动线程;闭包体只做 channel 投递)
unsafe extern "C" fn trampoline(data: *mut std::ffi::c_void) {
    let boxed = Box::from_raw(data as *mut Box<dyn FnOnce() + Send>);
    boxed();
}
