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
use cudarc::driver::DevicePtr;
use crate::command::{Ack, Command};
use crate::launch::issue_launch;
use crate::state::{DeviceSelector, GpuCtx, KernelCache, Staging};
use owl_iface::contract::{Arg, Bytes, LaunchMsg};
use std::ffi::c_void;
use crate::state::{STREAM_COMPUTE, STREAM_D2H, STREAM_H2D};
use owl_iface::contract::ModelError;
use std::sync::mpsc;

type Finish = Box<dyn FnOnce() + Send>;

fn srv_timing_on() -> bool {
    std::env::var_os("OWL_SRV_TIMING").is_some()
}

/// pinned 租约池(server 私有;Arc 随 dispatch 线程共持 ——
/// 生命周期严格罩在 CUDA ctx 之内,server 关闭即整池释放,
/// 杜绝跨 server 的悬垂页锁指针段错误)
#[derive(Default)]
pub(super) struct PinnedPool {
    free: std::sync::Mutex<Vec<Box<dyn owl_iface::contract::PinnedRegion + Send>>>,
}

/// FlashInfer prefill 句柄(E1.5):设备 workspace(float/int)+ host
/// 暂冲 + 1 项 plan 缓存(全层同参;key = 形状七元组)。
struct FiState {
    float_ws: cudarc::driver::CudaSlice<u8>,
    int_ws: cudarc::driver::CudaSlice<u8>,
    /// 裸指针缓存(device_ptr 的 SyncOnDrop 每调用同步流 —— 发射期禁调;
    /// init 时流空零成本取一次)
    float_ws_ptr: u64,
    int_ws_ptr: u64,
    host_staging: Vec<u8>,
    plan: Option<FiPlanCache>,
}

/// plan 缓存项:形状键 + plan15 + tile/split
struct FiPlanCache {
    key: (usize, usize, usize, usize, usize, usize, usize),
    plan15: [i64; 15],
    cta_tile_q: i32,
    split_kv: i32,
}

impl PinnedPool {
    fn take(&self, min_bytes: usize) -> Option<Box<dyn owl_iface::contract::PinnedRegion + Send>> {
        let mut v = self.free.lock().unwrap();
        let i = v.iter().position(|b| b.as_bytes().len() >= min_bytes)?;
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
    /// FlashInfer prefill 句柄(foreign-kernel 通道;workspace + plan 缓存,
    /// 懒初始化 —— owl-kernels::flashinfer)
    fi: Option<FiState>,
    /// 完成派发出口(host 回调只投递;派发线程执行真正的 finish)
    dispatch: Option<mpsc::Sender<Finish>>,
    /// 逐命令计时账本(OWL_SRV_TIMING;name / count / total_ns / max_ns)
    timings: Vec<(&'static str, u64, u128, u128)>,
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
            fi: None,
            dispatch: None,
            timings: Vec::new(),
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
        // server 侧逐命令分相汇总(OWL_SRV_TIMING;与 [load] 客户端探针互补)
        if srv_timing_on() && !self.timings.is_empty() {
            eprintln!("[srv-timing] ===== handler 分相(count / total / max) =====");
            let mut rows = self.timings.clone();
            rows.sort_by_key(|(_, _, total, _)| std::cmp::Reverse(*total));
            for (name, count, total, max) in rows {
                eprintln!(
                    "[srv-timing]   {:<14} x{:<5} total={:>8.1}ms max={:>8.1}ms",
                    name,
                    count,
                    total as f64 / 1e6,
                    max as f64 / 1e6
                );
            }
        }
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
        // cublas 句柄先于上下文消亡(2026-09-26 定谳:字段声明序 ctx 先于
        // blas 掉,teardown 时 cublasDestroy 撞已拆上下文 → libcublasLt
        // SIGSEGV;显式 take = 句柄在活上下文内销毁,gdb bt 实证)
        drop(self.blas.take());
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
        // 逐命令计时(OWL_SRV_TIMING 门控;卸载时汇总 —— server 侧分相,
        // 与客户端 [load] 探针互补:handler 真实执行时间,含阻塞段)
        let timing = srv_timing_on();
        let (t0, name) = if timing {
            (Some(std::time::Instant::now()), Some(Self::cmd_name(&cmd)))
        } else {
            (None, None)
        };
        self.dispatch_inner(cmd);
        if let (Some(t0), Some(name)) = (t0, name) {
            let dt = t0.elapsed();
            let slot = if let Some(pos) = self.timings.iter().position(|(n, ..)| *n == name) {
                &mut self.timings[pos]
            } else {
                self.timings.push((name, 0u64, 0u128, 0u128));
                let last = self.timings.len() - 1;
                &mut self.timings[last]
            };
            slot.1 += 1;
            slot.2 += dt.as_nanos() as u128;
            slot.3 = slot.3.max(dt.as_nanos() as u128);
        }
    }

    fn cmd_name(c: &Command) -> &'static str {
        match c {
            Command::GraphBegin { .. } => "GraphBegin",
            Command::GraphEnd { .. } => "GraphEnd",
            Command::GraphLaunch { .. } => "GraphLaunch",
            Command::Alloc { .. } => "Alloc",
            Command::MemsetZero { .. } => "MemsetZero",
            Command::Htod { .. } => "Htod",
            Command::HtodChunk { .. } => "HtodChunk",
            Command::HtodChunks { .. } => "HtodChunks",
            Command::AllocPinned { .. } => "AllocPinned",
            Command::UploadPinned { .. } => "UploadPinned",
            Command::Dtoh { .. } => "Dtoh",
            Command::Free { .. } => "Free",
            Command::CopyBlock { .. } => "CopyBlock",
            Command::Sync { .. } => "Sync",
            Command::Launch { .. } => "Launch",
            Command::Close { .. } => "Close",
        }
    }

    fn dispatch_inner(&mut self, cmd: Command) {
        // G2 护栏:捕获期间仅放行「Launch / Alloc(切 slab)」—— 二者由
        // 路由策略天然落在捕获流(COMPUTE);其余(搬运/同步/图操作)一律
        // 结构化拒绝 —— 它们或含非法捕获语义,或污染捕获拓扑。
        if self.ctx().capture_stream() {
            return match cmd {
                Command::Launch { msg, ack } => self.handle_launch(msg, ack),
                Command::Alloc { n_bytes, elems, zero, ack } => self.handle_alloc(n_bytes, elems, zero, ack),
                Command::MemsetZero { block, offset_bytes, len_bytes, ack } => {
                    self.handle_memset_zero(block, offset_bytes, len_bytes, ack)
                }
                Command::GraphEnd { ack } => ack.send(self.ctx_mut().graph_end()),
                Command::Close { ack } => ack.send(Ok(())), // 防御性幂等回执
                other => Self::reject(other),
            };
        }

        match cmd {
            Command::Alloc { n_bytes, elems, zero, ack } => self.handle_alloc(n_bytes, elems, zero, ack),
            Command::MemsetZero { block, offset_bytes, len_bytes, ack } => {
                self.handle_memset_zero(block, offset_bytes, len_bytes, ack)
            }
            Command::Htod { data, ack } => self.handle_htod(data, ack),
            Command::HtodChunks { writes, ack } => self.handle_htod_chunks(writes, ack),
            Command::HtodChunk { block, offset_bytes, data, ack } => {
                self.handle_htod_chunk(block, offset_bytes, data, ack)
            }
            Command::AllocPinned { bytes, ack } => self.handle_alloc_pinned(bytes, ack),
            Command::UploadPinned { buf, dst, offset_bytes, len_bytes, ack } => {
                self.handle_upload_pinned(buf, dst, offset_bytes, len_bytes, ack)
            }
            Command::Dtoh { id, want_bytes, ack } => self.handle_dtoh(id, want_bytes, ack),
            Command::Free { ids, ack } => self.handle_free(ids, ack),
            Command::CopyBlock { src, src_off_bytes, dst, dst_off_bytes, len_bytes, ack } => {
                self.handle_copy_block(src, src_off_bytes, dst, dst_off_bytes, len_bytes, ack)
            }
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
            Command::MemsetZero { ack, .. } => closed!(ack),
            Command::Htod { ack, .. } => closed!(ack),
            Command::HtodChunk { ack, .. } => closed!(ack),
            Command::HtodChunks { ack, .. } => closed!(ack),
            Command::AllocPinned { ack, .. } => closed!(ack),
            Command::UploadPinned { ack, .. } => closed!(ack),
            Command::Dtoh { ack, .. } => closed!(ack),
            Command::Free { ack, .. } => closed!(ack),
            Command::CopyBlock { ack, .. } => closed!(ack),
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
            Command::MemsetZero { .. } => "MemsetZero",
            Command::Htod { .. } => "Htod",
            Command::HtodChunk { .. } => "HtodChunk",
            Command::HtodChunks { .. } => "HtodChunks",
            Command::AllocPinned { .. } => "AllocPinned",
            Command::UploadPinned { .. } => "UploadPinned",
            Command::Dtoh { .. } => "Dtoh",
            Command::Free { .. } => "Free",
            Command::CopyBlock { .. } => "CopyBlock",
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
            Command::MemsetZero { ack, .. } => reject!(ack),
            Command::Htod { ack, .. } => reject!(ack),
            Command::HtodChunk { ack, .. } => reject!(ack),
            Command::HtodChunks { ack, .. } => reject!(ack),
            Command::AllocPinned { ack, .. } => reject!(ack),
            Command::UploadPinned { ack, .. } => reject!(ack),
            Command::Dtoh { ack, .. } => reject!(ack),
            Command::Sync { ack, .. } => reject!(ack),
            Command::Free { ack, .. } => reject!(ack),
            Command::CopyBlock { ack, .. } => reject!(ack),
            Command::Launch { ack, .. } => reject!(ack),
            Command::Close { ack } => ack.send(Ok(())), // 幂等
        }
    }

    fn handle_alloc(
        &mut self,
        n_bytes: usize,
        elems: usize,
        zero: bool,
        ack: Ack<Result<Bytes, ModelError>>,
    ) {
        // 捕获期:从 slab 切块(零 cudaMalloc;malloc 在捕获窗内非法)。
        // zero = true 时切完流序清零(N4:Zeros 语义/部分写 kernel 防护;
        // memset 可捕获 → replay 重清零);zero = false(uninit scratch,
        // S1)免 memset —— kernel 全量覆写的输出块不烙 memset 节点
        // (1207 节点 launch/GPU 双税定谳,2026-09-30)
        if self.ctx().capture_stream() {
            let result = (|| {
                let id = self.ctx_mut().carve_block(n_bytes)?;
                if zero {
                    let stream = self.ctx().stream(STREAM_COMPUTE)?.clone();
                    let (dptr, _) = self.ctx().block_ptr(id, &stream)?;
                    unsafe { memset_d8_async(dptr, 0, n_bytes, stream.cu_stream()) }
                        .map_err(|e| ModelError::Msg(format!("carve memset: {e:?}")))?;
                }
                Ok(Bytes::new(id, elems))
            })();
            return ack.send(result);
        }
        // 图外:路由 COMPUTE 流;zero = true 时流序清零(契约:Zeros = 清零
        // 分配;uninit 免 memset —— eager 大量 scratch alloc 同税)
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let result = unsafe { stream.alloc::<u8>(n_bytes) }
            .map_err(|e| ModelError::Msg(format!("alloc({n_bytes}B / {:.1}MiB): {e:?}", n_bytes as f64 / 1048576.0)))
            .and_then(|mut slice| {
                if zero {
                    stream
                        .memset_zeros(&mut slice)
                        .map_err(|e| ModelError::Msg(format!("alloc memset: {e:?}")))?;
                }
                // 诊断开关(OWL_LAUNCH_SYNC=1):alloc/memset 后同步归因
                if std::env::var_os("OWL_LAUNCH_SYNC").is_some() {
                    if let Err(e) = stream.synchronize() {
                        return Err(ModelError::Msg(format!(
                            "alloc-sync(n={n_bytes}): {e:?}"
                        )));
                    }
                }
                Ok(Bytes::new(self.ctx_mut().new_block(slice), elems))
            });
        ack.send(result);
    }

    /// 块清零(memset_d8 异步入 COMPUTE 流;入队即回执 —— 后续同流
    /// launch 天然保序,读方由显式 sync/同流序兜底)
    fn handle_memset_zero(
        &mut self,
        block: u64,
        offset_bytes: usize,
        len_bytes: usize,
        ack: Ack<Result<(), ModelError>>,
    ) {
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let result = (|| {
            let (dptr, _) = self.ctx().block_ptr(block, &stream)?;
            unsafe {
                crate::ffi::memset_d8_async(dptr + offset_bytes as u64, 0, len_bytes, stream.cu_stream())
            }
            .map_err(|e| ModelError::Msg(format!("memset_zero: {e:?}")))
        })();
        ack.send(result);
    }

    fn handle_htod(&mut self, data: Vec<u8>, ack: Ack<Result<Bytes, ModelError>>) {
        let mut ack = Some(ack);
        let n = data.len();
        let stream = match self.ctx().stream(STREAM_H2D) {
            Ok(s) => s.clone(),
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        match self.try_htod(&stream, &data) {
            Ok((id, buf)) => {
                // 码头随 finish 存活至搬运完成后归还池(派发线程执行)
                let mut cb_ack = ack.take();
                let pool = self.pool.clone();
                let finish: Finish = Box::new(move || {
                    pool.put(buf);
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
    ) -> Result<(u64, Box<dyn owl_iface::contract::PinnedRegion + Send>), ModelError> {
        let n = data.len();
        // 1. 设备块(流序 malloc;未初始化;字节口径)
        let dst = unsafe { stream.alloc::<u8>(n) }
            .map_err(|e| ModelError::Msg(format!("htod alloc: {e:?}")))?;
        let id = self.ctx_mut().new_block(dst);
        // 2. pinned 码头 + 异步 memcpy(流序;host 侧必须 pinned 才真异步)。
        //    码头池化(2026-09-26 尾批:原每次 malloc_host 新分配,大权重
        //    锁页分配是装载耗时大头;finish 回调归还池)
        let mut buf: Box<dyn owl_iface::contract::PinnedRegion + Send> =
            match self.pool.take(n) {
                Some(b) => b,
                None => Box::new(Staging::alloc(n)?),
            };
        buf.slice_bytes_mut()[..n].copy_from_slice(data);
        let (dptr, _) = self.ctx().block_ptr(id, stream)?;
        // 池块容量 ≥ 请求:DMA 长度必须截到 n(越界写设备块 = INVALID_VALUE)
        unsafe { memcpy_htod_async(dptr, &buf.as_bytes()[..n], stream.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("htod async: {e:?}")))?;
        // 诊断开关(OWL_LAUNCH_SYNC=1):DMA 后同步归因(H2D 错误平时
        // 无条件 Ok 回执,sticky 晚冒 —— 2026-09-27 排查补)
        if std::env::var_os("OWL_LAUNCH_SYNC").is_some() {
            stream
                .synchronize()
                .map_err(|e| ModelError::Msg(format!("htod-sync(n={n}): {e:?}")))?;
        }
        Ok((id, buf))
    }

    /// 分配 pinned 租约：池优先，miss 才 cudaHostAlloc。
    /// 不填零（2026-09-26 装载定谳）：租约语义 = 调用方整块覆写后再按
    /// len_bytes DMA，填零纯属 server 单线程上的双倍带宽税。
    fn handle_alloc_pinned(
        &mut self,
        bytes: usize,
        ack: Ack<Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError>>,
    ) {
        // 池取（内部单次加锁；容量 ≥ 请求即命中，逻辑长度由调用方界定）
        if let Some(b) = self.pool.take(bytes) {
            // 容量不可截，整块交出（调用方只用前 bytes）
            ack.send(Ok(b));
            return;
        }
        match Staging::alloc(bytes) {
            Ok(s) => ack.send(Ok(Box::new(s))),
            Err(e) => ack.send(Err(e)),
        }
    }

    /// 上传租约:buf 所有权移入,DMA 到 dst+offset;完成回调把 buf 归还池
    fn handle_upload_pinned(
        &mut self,
        buf: Box<dyn owl_iface::contract::PinnedRegion + Send>,
        dst: Bytes,
        offset_bytes: usize,
        len_bytes: usize,
        ack: Ack<Result<(), ModelError>>,
    ) {
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
                dptr_base + (offset_bytes as u64),
                &buf.as_bytes()[..len_bytes],
                stream.cu_stream(),
            )
        } {
            // 建场失败:未入队任何 DMA,finish 不得投递 —— 租约回池 + 回执错误
            let e = ModelError::Msg(format!("upload_pinned async: {e:?}"));
            self.pool.put(buf);
            ack.take().unwrap().send(Err(e));
            return;
        }
        // 异步语义(F5 尾批,用户裁决):入队即回执 —— DMA 完成由调用方
        // 的显式 sync 栅栏兜底;finish 只归还租约入池(页锁定代价不再
        // 逐块支付,池真正转起来)。
        let cell = std::sync::Arc::new(std::sync::Mutex::new(Some(buf)));
        let cell2 = cell.clone();
        let pool = self.pool.clone();
        let pool_err = self.pool.clone();
        let finish: Finish = Box::new(move || {
            let buf = cell2.lock().unwrap().take().unwrap();
            pool.put(buf);
        });
        ack.take().unwrap().send(Ok(())); // 入队即回执(非阻塞)
        if let Err(e) = self.notify(&stream, finish) {
            // notify 失败:finish 未投递,租约回收入池(ack 已回,
            // DMA 未发生 —— 块内容缺失由调用方末端 sync 后自查/重试兜底)
            if let Some(buf) = cell.lock().unwrap().take() {
                pool_err.put(buf);
                eprintln!("[srv] upload notify 失败: {e:?}");
            }
        }
    }

    /// 分块写入已 alloc 的块(流式装载):pinned 码头 + memcpyAsync
    /// 到 dst+offset;完成回调回执
    /// 批量分块写入(S1):逐写池化码头 + H2D 流序 memcpy,末尾单
    /// notify —— 一命令一往返,保序协议不变(ack = 全部写完)
    fn handle_htod_chunks(
        &mut self,
        writes: Vec<(u64, usize, Vec<u8>)>,
        ack: Ack<Result<(), ModelError>>,
    ) {
        let mut ack = Some(ack);
        let stream = match self.ctx().stream(STREAM_H2D) {
            Ok(s) => s.clone(),
            Err(e) => return ack.take().unwrap().send(Err(e)),
        };
        let mut pending: Vec<Box<dyn owl_iface::contract::PinnedRegion + Send>> =
            Vec::with_capacity(writes.len());
        for (block, offset_bytes, data) in &writes {
            let dptr_base = match self.ctx().block_ptr(*block, &stream) {
                Ok((p, _)) => p,
                Err(e) => {
                    for b in pending {
                        self.pool.put(b);
                    }
                    return ack.take().unwrap().send(Err(e));
                }
            };
            let mut staging: Box<dyn owl_iface::contract::PinnedRegion + Send> =
                match self.pool.take(data.len()) {
                    Some(b) => b,
                    None => match Staging::alloc(data.len()) {
                        Ok(s) => Box::new(s),
                        Err(e) => {
                            for b in pending {
                                self.pool.put(b);
                            }
                            return ack.take().unwrap().send(Err(e));
                        }
                    },
                };
            staging.slice_bytes_mut()[..data.len()].copy_from_slice(data);
            if let Err(e) = unsafe {
                memcpy_htod_async(
                    dptr_base + (*offset_bytes as u64),
                    &staging.as_bytes()[..data.len()],
                    stream.cu_stream(),
                )
            }
            .map_err(|e| ModelError::Msg(format!("htod chunks async: {e:?}")))
            {
                pending.push(staging);
                for b in pending {
                    self.pool.put(b);
                }
                return ack.take().unwrap().send(Err(e));
            }
            pending.push(staging);
        }
        // 单 notify:全部写到位后一次性回执(保序 = H2D 流序)
        let mut cb_ack = ack.take();
        let pool = self.pool.clone();
        let finish: Finish = Box::new(move || {
            for b in pending {
                pool.put(b);
            }
            cb_ack.take().unwrap().send(Ok(()));
        });
        if let Err(e) = self.notify(&stream, finish) {
            // notify 失败 = 断链(ack 已随闭包移交;与 handle_htod_chunk 同语义)
            let _ = e;
        }
    }

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
        // 码头池化(S1,2026-09-30):write_block 是 decode 每步 5 连发的
        // 高频小写路径,per-call malloc_host/Free 实测 ~3.3ms/次(与
        // try_htod 09-26 同病史;池 = PinnedPool,take>=len / finish 归还)
        let mut staging: Box<dyn owl_iface::contract::PinnedRegion + Send> =
            match self.pool.take(data.len()) {
                Some(b) => b,
                None => match Staging::alloc(data.len()) {
                    Ok(s) => Box::new(s),
                    Err(e) => return ack.take().unwrap().send(Err(e)),
                },
            };
        staging.slice_bytes_mut()[..data.len()].copy_from_slice(&data);
        unsafe {
            memcpy_htod_async(
                dptr_base + (offset_bytes as u64),
                &staging.as_bytes()[..data.len()],
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
        // 码头随 finish 存活至搬运完成后归还池(派发线程执行)
        let mut cb_ack = ack.take();
        let pool = self.pool.clone();
        let finish: Finish = Box::new(move || {
            pool.put(staging);
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
                // 完成 → 码头字节直出(传输面字节口径)→ 回执 → 码头归还池
                let mut cb_ack = ack.take();
                let pool = self.pool.clone();
                // 字节口径 = want_bytes(池码头容量 ≥ n,直出整段会超发)
                let finish: Finish = Box::new(move || {
                    let bytes = staging.as_bytes()[..want_bytes].to_vec();
                    pool.put(staging);
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
    ) -> Result<Box<dyn owl_iface::contract::PinnedRegion + Send>, ModelError> {
        let n = self.ctx().block_len(id)?; // 块账本 = 字节(2026-09-26 f16 基线)
        if n != want_bytes {
            return Err(ModelError::Msg(format!(
                "dtoh: 块 {id} 字节 {n} != 收割 {want_bytes}"
            )));
        }
        let (dptr, _) = self.ctx().block_ptr(id, stream)?;
        // 码头池化(S1):收割路径同样高频(token 4B/步;logits 大块另计)
        let mut staging: Box<dyn owl_iface::contract::PinnedRegion + Send> =
            match self.pool.take(n) {
                Some(b) => b,
                None => Box::new(Staging::alloc(n)?),
            };
        // 拷贝长度钉死 n(池码头容量 ≥ n,整段拷会越读设备块)
        unsafe {
            memcpy_dtoh_async(&mut staging.slice_bytes_mut()[..n], dptr, stream.cu_stream())
        }
        .map_err(|e| ModelError::Msg(format!("dtoh async: {e:?}")))?;
        Ok(staging)
    }


    fn handle_launch(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        // GPU 逐核归因探针(OWL_GPU_PROF=1;2026-10-01 50tok/s 分账立案):
        // 每发射后同步 COMPUTE 流 → 本核隔离运行,host 计时 ≈ 核 GPU 纯时
        // (含 ~10µs sync 往返底噪);动态 tag `gpu.{核名}` 入 owl-shared
        // metrics(同进程线程,store 共享),测试侧 query(prefix "gpu.")
        // 出热点分账。捕获期跳过 —— 图回放是整图单发射,逐核归因由
        // OWL_NO_GRAPH eager 窗提供;与 OWL_LAUNCH_TIME(host 提交 µs)
        // 互补:本探针答「GPU 时花在哪」,彼答「host 提交多贵」。
        let gpu_prof = std::env::var_os("OWL_GPU_PROF").is_some();
        if gpu_prof && !self.ctx().capture_stream() {
            let name = msg.kernel.name.clone();
            let t0 = std::time::Instant::now();
            let r = self.handle_launch_inner(msg, ack);
            if let Ok(stream) = self.ctx().stream(STREAM_COMPUTE) {
                let _ = stream.synchronize(); // 归因同步(错误由 sticky 面另报)
            }
            let dt = t0.elapsed();
            owl_shared::metrics::with_metrics_store(|s| {
                s.timer_record_tag(&format!("gpu.{name}"), dt, file!(), 0);
            });
            return r;
        }
        // 逐发射计时(OWL_LAUNCH_TIME=1;捕获期 warmup = 全模型 eager 一遍,
        // 一次跑完全谱;capture 回放期自动关闭 —— 回放是整图一发射)
        if std::env::var_os("OWL_LAUNCH_TIME").is_some() && !self.ctx().capture_stream() {
            let t = std::time::Instant::now();
            let name = msg.kernel.name.clone();
            let r = self.handle_launch_inner(msg, ack);
            eprintln!("[launch-time] {:?} {}", t.elapsed(), name);
            return r;
        }
        self.handle_launch_inner(msg, ack)
    }

    fn handle_launch_inner(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        // foreign-kernel 通道(2026-09-26 合并:cuBLAS 不再另立命令,
        // 外部算子 = 虚拟核名走同一 Launch;谓词与槽序归 owl-kernels::cublas)
        if owl_kernels::is_foreign_op(&msg.kernel.name) {
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
        let capturing = ctx.capture_stream();
        match issue_launch(ctx, &stream, kernels, &msg) {
            Ok(out_id) => {
                self.ctx_mut().note_launch();
                // 诊断开关(OWL_LAUNCH_SYNC=1):逐发射同步,sticky 错误
                // 归属到具体核(2026-09-27 长 ctx ILLEGAL_ADDRESS 排查;
                // 捕获期禁 sync —— 跳过,捕获正确性由哨兵③/④另保)
                if std::env::var_os("OWL_LAUNCH_SYNC").is_some() && !capturing
                {
                    if let Err(e) = stream.synchronize() {
                        return ack.take().unwrap().send(Err(ModelError::Msg(format!(
                            "launch-sync({}): {e:?}",
                            msg.kernel.name
                        ))));
                    }
                }
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
        // 捕获期策略(2026-09-26 修订):cublasGemmEx/外部核本身可捕获
        // (kernel 入捕获流;workspace 需捕获前已分配 —— session warmup
        // 先于 graph_begin,句柄已建即满足)。唯一拒绝态 = 捕获内首次
        // 初始化(句柄创建含分配,捕获窗内非法)→ 结构化拒绝。
        if self
            .ctx
            .as_ref()
            .map(|c| c.capture_stream())
            .unwrap_or(false)
            && self.blas.is_none()
        {
            return ack.send(Err(ModelError::Msg(format!(
                "foreign kernel {} 捕获内首次初始化(需先 warmup 建句柄)",
                msg.kernel.name
            ))));
        }
        match msg.kernel.name.as_str() {
            name if name == owl_kernels::cublas::GEMM_F16 => self.handle_cublas_gemm(msg, ack),
            name if name == owl_kernels::marlin::GEMM_W4A16 => self.handle_marlin_gemm(msg, ack),
            name if name == owl_kernels::marlin::GEMM_W4A16_AWQ => {
                self.handle_marlin_gemm_awq(msg, ack)
            }
            name if name == owl_kernels::flashinfer::PREFILL_FI => self.handle_fi_prefill(msg, ack),
            other => ack.send(Err(ModelError::Msg(format!(
                "foreign kernel {other}: 无执行臂(owl_kernels::is_foreign_op 与分派表失配)"
            )))),
        }
    }

    /// cuBLAS f16 GEMM 臂(槽序:[T a, T b, T out, sz m, sz k, sz n, sz nt])
    fn handle_cublas_gemm(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
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
        if std::env::var_os("OWL_DEBUG").is_some() {
            eprintln!(
                "[dbg foreign] {} a=Block({}) b=Block({}) out=Block({}) m={n_out} k={k} n={n_tok} nt={nt}",
                msg.kernel.name, blocks[0], blocks[1], blocks[2], n_out = m, n_tok = n
            );
        }
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
        let capturing = self.ctx().capture_stream();
        match blas.gemm_f16(a_ptr, b_ptr, out_ptr, m, k, n, nt) {
            Ok(()) => {
                // 诊断开关(同 handle_launch;捕获期跳过)
                if std::env::var_os("OWL_LAUNCH_SYNC").is_some() && !capturing {
                    if let Err(e) = stream.synchronize() {
                        return ack.send(Err(ModelError::Msg(format!(
                            "gemm-sync(m={m},k={k},n={n},nt={nt}): {e:?}"
                        ))));
                    }
                }
                ack.send(Ok(Bytes::new(blocks[2], msg.out_elems)))
            }
            Err(e) => ack.send(Err(ModelError::Msg(e))),
        }
    }

    /// Marlin W4A16 臂(槽序:[T a, T b, T out, T scales, T ws, T c_tmp,
    /// sz m, sz k, sz n, sz groupsize];契约见 owl_kernels::marlin)。
    /// 无句柄状态(纯 FFI);stream/dev 由本 server 注入。
    fn handle_marlin_gemm(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        let mut blocks: Vec<u64> = Vec::new();
        let mut scalars: Vec<u64> = Vec::new();
        for a in &msg.args {
            match a {
                Arg::Block { id } => blocks.push(*id),
                Arg::U64(v) => scalars.push(*v),
                _ => {
                    return ack.send(Err(ModelError::Msg(format!(
                        "foreign kernel {} 槽序违约:仅 Block/U64(见 owl_kernels::marlin 契约)",
                        msg.kernel.name
                    ))))
                }
            }
        }
        if blocks.len() != 6 || scalars.len() != 4 {
            return ack.send(Err(ModelError::Msg(format!(
                "foreign kernel {} 槽序违约:6 Block + 4 U64,得 {}B/{}S",
                msg.kernel.name,
                blocks.len(),
                scalars.len()
            ))));
        }
        let (m, k, n, groupsize) =
            (scalars[0] as usize, scalars[1] as usize, scalars[2] as usize, scalars[3] as i32);
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ptrs = Vec::with_capacity(6);
        for b in &blocks {
            match self.ctx().block_ptr(*b, &stream) {
                Ok((p, _)) => ptrs.push(p),
                Err(e) => return ack.send(Err(e)),
            }
        }
        let dev = self.ctx().device_ordinal() as i32;
        // 排队即回执(fire-and-forget;marlin host launcher 入 COMPUTE 流)
        let r = unsafe {
            owl_kernels::marlin::gemm_v2_raw(
                ptrs[0] as *const u16,
                ptrs[1] as *const i32,
                ptrs[2] as *mut u16,
                ptrs[3] as *const u16,
                ptrs[5] as *const c_void,
                m as i32,
                n as i32,
                k as i32,
                ptrs[4] as *mut i32,
                groupsize,
                dev,
                stream.cu_stream() as usize,
            )
        };
        match r {
            Ok(()) => ack.send(Ok(Bytes::new(blocks[2], msg.out_elems))),
            Err(e) => ack.send(Err(ModelError::Msg(format!(
                "marlin gemm err {e}: {}",
                owl_kernels::marlin::v2_err_str(e)
            )))),
        }
    }


    /// FlashInfer paged prefill 臂(E1.5)。槽序契约:
    /// [T q, T kc_fi, T vc, T q_cu, T indices, T indptr, T last_len, T wr,
    ///  O out, sz total_rows, sz ctx_total, sz T, sz hq, sz hkv, sz hd,
    ///  sz page, sz sm_scale_bits, sz nb](8 Block + 8 sz;wr 依赖边)。
    /// plan(host,每 chunk 一次)1 项缓存;run 每层一次。批 = 1。
    /// 捕获窗拒绝态同 cublas(FI prefill 现役仅 eager;解码图不含它)。
    fn handle_fi_prefill(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        if self.fi.is_none() {
            if self.ctx.as_ref().map(|c| c.capture_stream()).unwrap_or(false) {
                return ack.send(Err(ModelError::Msg(
                    "flashinfer_prefill 捕获内首次初始化(需先 eager warmup)".into(),
                )));
            }
            let stream = match self.ctx().stream(STREAM_COMPUTE) {
                Ok(s) => s.clone(),
                Err(e) => return ack.send(Err(e)),
            };
            let float_ws = stream
                .alloc_zeros::<u8>(owl_kernels::flashinfer::FI_FLOAT_WS_BYTES)
                .map_err(|e| ModelError::Msg(format!("fi ws alloc: {e:?}")));
            let int_ws = stream
                .alloc_zeros::<u8>(owl_kernels::flashinfer::FI_INT_WS_BYTES)
                .map_err(|e| ModelError::Msg(format!("fi ws alloc: {e:?}")));
            let (float_ws, int_ws) = match (float_ws, int_ws) {
                (Ok(a), Ok(b)) => (a, b),
                (Err(e), _) | (_, Err(e)) => return ack.send(Err(e)),
            };
            use cudarc::driver::DevicePtr;
            // 空流快照:此时无排队内核,SyncOnDrop 守卫就地丢弃零成本
            let float_ws_ptr = {
                let (p, _g) = DevicePtr::<u8>::device_ptr(&float_ws, &stream);
                p
            };
            let int_ws_ptr = {
                let (p, _g) = DevicePtr::<u8>::device_ptr(&int_ws, &stream);
                p
            };
            self.fi = Some(FiState {
                float_ws,
                int_ws,
                float_ws_ptr,
                int_ws_ptr,
                host_staging: vec![0u8; owl_kernels::flashinfer::FI_HOST_STAGING_BYTES],
                plan: None,
            });
        }
        let mut blocks: Vec<u64> = Vec::new();
        let mut scalars: Vec<u64> = Vec::new();
        for a in &msg.args {
            match a {
                Arg::Block { id } => blocks.push(*id),
                Arg::U64(v) => scalars.push(*v),
                _ => {
                    return ack.send(Err(ModelError::Msg(format!(
                        "foreign kernel {} 槽序违约:仅 Block/U64(见 owl_kernels::flashinfer 契约)",
                        msg.kernel.name
                    ))))
                }
            }
        }
        if blocks.len() != 9 || scalars.len() != 8 {
            return ack.send(Err(ModelError::Msg(format!(
                "foreign kernel {} 槽序违约:9 Block + 8 sz,得 {}B/{}S",
                msg.kernel.name,
                blocks.len(),
                scalars.len()
            ))));
        }
        let (total_rows, ctx_total, t, hq, hkv, hd, page, sm_bits) = (
            scalars[0] as usize, scalars[1] as usize, scalars[2] as usize,
            scalars[3] as usize, scalars[4] as usize, scalars[5] as usize,
            scalars[6] as usize, scalars[7] as u32,
        );
        let sm_scale = f32::from_bits(sm_bits);
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ptrs = Vec::with_capacity(9);
        for b in &blocks {
            match self.ctx().block_ptr(*b, &stream) {
                Ok((p, _)) => ptrs.push(p),
                Err(e) => return ack.send(Err(e)),
            }
        }
        // workspace 指针/长度(init 时缓存的裸指针;发射期零同步)
        let (int_ws_ptr, int_ws_len, float_ws_ptr, float_ws_len, staging_ptr, staging_len) = {
            let fi = self.fi.as_mut().unwrap();
            (
                fi.int_ws_ptr as *mut c_void,
                fi.int_ws.len(),
                fi.float_ws_ptr as *mut c_void,
                fi.float_ws.len(),
                fi.host_staging.as_mut_ptr() as *mut c_void,
                fi.host_staging.len(),
            )
        };
        // plan 缓存(键 = 形状七元组 + total_rows)
        let key = (total_rows, ctx_total, t, hq, hkv, hd, page);
        let need_plan = self.fi.as_ref().unwrap().plan.as_ref().map(|c| c.key != key).unwrap_or(true);
        let fi_plan_t0 = std::time::Instant::now();
        let plan_result: Result<_, String> = (|| {
            let qo_indptr = [0i32, t as i32];
            let kv_indptr = [0i32, ctx_total as i32];
            let (mut plan15, mut cta_tile_q, mut split_kv) = ([0i64; 15], 0i32, 0i32);
            let r = unsafe {
                owl_kernels::flashinfer::owl_fi_prefill_plan(
                    float_ws_ptr,
                    float_ws_len,
                    int_ws_ptr,
                    int_ws_len,
                    staging_ptr,
                    staging_len,
                    plan15.as_mut_ptr(),
                    &mut cta_tile_q,
                    &mut split_kv,
                    qo_indptr.as_ptr(),
                    kv_indptr.as_ptr(),
                    total_rows as i32,
                    1, // batch = 1(owl 单会话)
                    hq as i32, hkv as i32, hd as i32, page as i32,
                    stream.cu_stream() as *mut c_void,
                )
            };
            if r != 0 {
                return Err(format!(
                    "flashinfer_prefill_plan err {r}(hd={hd} page={page} T={t} ctx={ctx_total})"
                ));
            }
            Ok((plan15, cta_tile_q, split_kv))
        })();
        owl_shared::metrics::with_metrics_store(|m| {
            m.timer_record_tag("fi.plan", fi_plan_t0.elapsed(), file!(), line!())
        });
        let (mut plan15, _cta, _split) = match plan_result {
            Ok(v) => v,
            Err(e) => return ack.send(Err(ModelError::Msg(e))),
        };
        if let Some(fi) = self.fi.as_mut() {
            fi.plan = Some(FiPlanCache { key, plan15, cta_tile_q: _cta, split_kv: _split });
        }
        let _ = (_cta, _split);
        let plan15 = self.fi.as_ref().unwrap().plan.as_ref().unwrap().plan15;
        let fi_run_t0 = std::time::Instant::now(); // v2
        let r = unsafe {
            owl_kernels::flashinfer::owl_fi_prefill_run(
                ptrs[0] as *const c_void,
                ptrs[1] as *const c_void,
                ptrs[2] as *const c_void,
                ptrs[8] as *mut c_void,
                ptrs[3] as *mut i32,
                ptrs[4] as *mut i32,
                ptrs[5] as *mut i32,
                ptrs[6] as *mut i32,
                plan15.as_ptr(),
                int_ws_ptr,
                int_ws_len,
                float_ws_ptr,
                float_ws_len,
                1, // batch
                hq as i32, hkv as i32, hd as i32, page as i32,
                total_rows as i32,
                sm_scale,
                stream.cu_stream() as *mut c_void,
            )
        };
        owl_shared::metrics::with_metrics_store(|m| {
            m.timer_record_tag("fi.run", fi_run_t0.elapsed(), file!(), line!())
        });
        match r {
            0 => ack.send(Ok(Bytes::new(blocks[8], msg.out_elems))),
            e => ack.send(Err(ModelError::Msg(format!("flashinfer_prefill_run err {e}")))),
        }
    }

    /// Marlin AWQ(kU4 非对称)臂(2026-10-01 cyankiwi g32 装载线)。
    /// 槽序:[T a, T b, T out, T scales, T zeros, T ws, T ctmp,
    /// sz m, sz k, sz n, sz groupsize](7 Block + 4 sz);
    /// zeros = pack_marlin_z 产物((k/g, n/8) i32)。
    fn handle_marlin_gemm_awq(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        let mut blocks: Vec<u64> = Vec::new();
        let mut scalars: Vec<u64> = Vec::new();
        for a in &msg.args {
            match a {
                Arg::Block { id } => blocks.push(*id),
                Arg::U64(v) => scalars.push(*v),
                _ => {
                    return ack.send(Err(ModelError::Msg(format!(
                        "foreign kernel {} 槽序违约:仅 Block/U64(见 owl_kernels::marlin 契约)",
                        msg.kernel.name
                    ))))
                }
            }
        }
        if blocks.len() != 7 || scalars.len() != 4 {
            return ack.send(Err(ModelError::Msg(format!(
                "foreign kernel {} 槽序违约:7 Block + 4 U64,得 {}B/{}S",
                msg.kernel.name,
                blocks.len(),
                scalars.len()
            ))));
        }
        let (m, k, n, groupsize) =
            (scalars[0] as usize, scalars[1] as usize, scalars[2] as usize, scalars[3] as i32);
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ptrs = Vec::with_capacity(7);
        for b in &blocks {
            match self.ctx().block_ptr(*b, &stream) {
                Ok((p, _)) => ptrs.push(p),
                Err(e) => return ack.send(Err(e)),
            }
        }
        let dev = self.ctx().device_ordinal() as i32;
        let r = unsafe {
            owl_kernels::marlin::gemm_v2_awq_raw(
                ptrs[0] as *const u16,
                ptrs[1] as *const i32,
                ptrs[2] as *mut u16,
                ptrs[3] as *const u16,
                ptrs[4] as *const i32,
                ptrs[6] as *const c_void,
                m as i32,
                n as i32,
                k as i32,
                ptrs[5] as *mut i32,
                groupsize,
                dev,
                stream.cu_stream() as usize,
            )
        };
        match r {
            Ok(()) => ack.send(Ok(Bytes::new(blocks[2], msg.out_elems))),
            Err(e) => ack.send(Err(ModelError::Msg(format!(
                "marlin awq gemm err {e}: {}",
                owl_kernels::marlin::v2_err_str(e)
            )))),
        }
    }

    /// 中间块回收(E2a):账房移除 Owned 块归池。调用方契约 = 已收割
    /// 所需数据(dtoh 回执即 COMPUTE 排空);图捕获期由 G2 拒绝臂拦截。
    fn handle_free(&mut self, ids: Vec<u64>, ack: Ack<Result<(), ModelError>>) {
        let freed = self.ctx_mut().free_blocks(&ids);
        if std::env::var_os("OWL_DEBUG").is_some() {
            eprintln!("[owl-gpu] free: {}/{} 块归池", freed, ids.len());
        }
        ack.send(Ok(()));
    }

    /// 设备内块→块拷贝(E2c 快照通道;COMPUTE 流序,fire-and-forget)
    #[allow(clippy::too_many_arguments)]
    fn handle_copy_block(
        &mut self,
        src: u64,
        src_off_bytes: usize,
        dst: u64,
        dst_off_bytes: usize,
        len_bytes: usize,
        ack: Ack<Result<(), ModelError>>,
    ) {
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let result = (|| {
            let (sptr, _) = self.ctx().block_ptr(src, &stream)?;
            let (dptr, _) = self.ctx().block_ptr(dst, &stream)?;
            unsafe {
                crate::ffi::memcpy_dtod_async(
                    dptr + dst_off_bytes as u64,
                    sptr + src_off_bytes as u64,
                    len_bytes,
                    stream.cu_stream(),
                )
            }
            .map_err(|e| ModelError::Msg(format!("copy_block: {e:?}")))
        })();
        ack.send(result);
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