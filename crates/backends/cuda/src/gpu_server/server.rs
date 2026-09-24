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

use crate::ffi::{launch_host_function, memcpy_dtoh_async, memcpy_htod_async};
use crate::gpu_server::command::{Ack, Command};
use crate::gpu_server::launch::issue_launch;
use crate::gpu_server::state::{GpuCtx, KernelCache, Staging};
use owl_models::client::{Bytes, LaunchMsg};
use crate::gpu_server::state::{STREAM_COMPUTE, STREAM_D2H, STREAM_H2D};
use owl_models::ModelError;
use std::sync::mpsc;

type Finish = Box<dyn FnOnce() + Send>;

pub struct GpuServer {
    rx: mpsc::Receiver<Command>,
    ordinal: usize,
    /// 上线握手(可选;便捷组装路径用它回传设备上线结果)
    boot: Option<mpsc::Sender<Result<(), String>>>,
    ctx: Option<GpuCtx>,
    kernels: KernelCache,
    /// 完成派发出口(host 回调只投递;派发线程执行真正的 finish)
    dispatch: Option<mpsc::Sender<Finish>>,
}

impl GpuServer {
    /// server 构造函数:**纯构造,零 CUDA 调用**(设备上下文在 `run()` 时
    /// 于 run 所在线程上线 —— bind_to_thread 的线程亲和性要求如此)。
    ///
    /// - `rx`:命令管道的 server 端(管道由外部组装者创建,分别传入
    ///   server 与 client;server 不自创管道)
    /// - `boot`:上线握手(可选;`None` = 外部不需要启动确认)。
    ///   用 mpsc 同步通道(boot 发生在 server 线程,组装者在另一线程同步等)。
    pub fn new(
        rx: mpsc::Receiver<Command>,
        ordinal: usize,
        boot: Option<mpsc::Sender<Result<(), String>>>,
    ) -> Self {
        Self { rx, ordinal, boot, ctx: None, kernels: KernelCache::new(), dispatch: None }
    }

    /// 事件循环主入口(server 线程生命周期 = 循环生命周期)。
    /// 首行上线设备(本线程 bind_to_thread 一次到位)+ 起派发线程。
    pub fn run(mut self) -> Result<(), String> {
        let ctx = GpuCtx::new(self.ordinal);
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
            .name(format!("owl-gpu-{}-dispatch", self.ordinal))
            .spawn(move || {
                for finish in drx {
                    finish();
                }
            })
            .map_err(|e| format!("派发线程启动失败: {e}"))?;
        self.dispatch = Some(dtx);

        // 纯阻塞监听:零轮询;客户端全部离场(Disconnected)即收摊。
        // 在飞 finish 由派发线程排空(它持有 channel 另一端,自然收尾)。
        while let Some(cmd) = self.rx.recv().ok() {
            self.dispatch(cmd);
        }
        Ok(())
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
                Command::Alloc { n_elems, ack } => self.handle_alloc(n_elems, ack),
                Command::GraphEnd { ack } => ack.send(self.ctx_mut().graph_end()),
                other => Self::reject(other),
            };
        }

        match cmd {
            Command::Alloc { n_elems, ack } => self.handle_alloc(n_elems, ack),
            Command::Htod { data, ack } => self.handle_htod(data, ack),
            Command::Dtoh { id, want_elems, ack } => self.handle_dtoh(id, want_elems, ack),
            Command::Launch { msg, ack } => self.handle_launch(msg, ack),
            Command::Sync { ack } => self.handle_sync(ack),
            Command::GraphBegin { ack } => ack.send(self.ctx_mut().graph_begin()),
            Command::GraphEnd { ack } => ack.send(self.ctx_mut().graph_end()),
            Command::GraphLaunch { graph, ack } => {
                ack.send(self.ctx_mut().graph_launch(graph))
            }
        }
    }

    /// 捕获期违规命令的结构化拒绝
    fn reject(cmd: Command) {
        let why = format!(
            "图捕获进行中:该操作被拒 —— 捕获期仅允许 Launch/Alloc;\
             同步/搬运/图操作会破坏图捕获"
        );
        macro_rules! reject { ($ack:expr) => { $ack.send(Err(ModelError::Msg(why.clone()))) } }
        let name = |c: &Command| match c {
            Command::GraphBegin { .. } => "GraphBegin",
            Command::GraphEnd { .. } => "GraphEnd",
            Command::GraphLaunch { .. } => "GraphLaunch",
            Command::Alloc { .. } => "Alloc",
            Command::Htod { .. } => "Htod",
            Command::Dtoh { .. } => "Dtoh",
            Command::Sync { .. } => "Sync",
            Command::Launch { .. } => "Launch",
        };
        eprintln!("[owl-gpu] 捕获期拒绝: {} —— {why}", name(&cmd));
        match cmd {
            Command::GraphBegin { ack } => reject!(ack),
            Command::GraphEnd { ack } => reject!(ack),
            Command::GraphLaunch { ack, .. } => reject!(ack),
            Command::Alloc { ack, .. } => reject!(ack),
            Command::Htod { ack, .. } => reject!(ack),
            Command::Dtoh { ack, .. } => reject!(ack),
            Command::Sync { ack, .. } => reject!(ack),
            Command::Launch { ack, .. } => reject!(ack),
        }
    }

    fn handle_alloc(&mut self, n_elems: usize, ack: Ack<Result<Bytes, ModelError>>) {
        // 捕获期:从 slab 切块(零 cudaMalloc;malloc 在捕获窗内非法)
        if self.ctx().capture_stream() {
            let result = self
                .ctx_mut()
                .carve_block(n_elems)
                .map(|id| Bytes::new(id, n_elems));
            return ack.send(result);
        }
        // 图外:路由 COMPUTE 流,流序 memset(非阻塞);立即回执
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let result = unsafe { stream.alloc::<f32>(n_elems) }
            .map_err(|e| ModelError::Msg(format!("alloc: {e:?}")))
            .map(|slice| Bytes::new(self.ctx_mut().new_block(slice), n_elems));
        ack.send(result);
    }

    fn handle_htod(&mut self, data: Vec<f32>, ack: Ack<Result<Bytes, ModelError>>) {
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
        data: &[f32],
    ) -> Result<(u64, Staging), ModelError> {
        let n = data.len();
        // 1. 设备块(流序 malloc;未初始化)
        let dst = unsafe { stream.alloc::<f32>(n) }
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

    fn handle_dtoh(
        &mut self,
        id: u64,
        want_elems: usize,
        ack: Ack<Result<Vec<u8>, ModelError>>,
    ) {
        // 路由:D2H 流(结果收割维)
        let stream = match self.ctx().stream(STREAM_D2H) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let mut ack = Some(ack);
        match self.try_dtoh(&stream, id, want_elems) {
            Ok(staging) => {
                // 完成 → 码头转 LE 字节 → 回执 → 放掉码头(派发线程跑)
                let mut cb_ack = ack.take();
                let finish: Finish = Box::new(move || {
                    let bytes: Vec<u8> =
                        staging.slice().iter().flat_map(|f| f.to_le_bytes()).collect();
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
        want_elems: usize,
    ) -> Result<Staging, ModelError> {
        let n = self.ctx().block_len(id)?;
        if n != want_elems {
            return Err(ModelError::Msg(format!(
                "dtoh: 块 {id} 元素 {n} != 收割 {want_elems}"
            )));
        }
        let (dptr, _) = self.ctx().block_ptr(id, stream)?;
        let mut staging = Staging::alloc(n)?;
        unsafe { memcpy_dtoh_async(staging.slice_mut(), dptr, stream.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("dtoh async: {e:?}")))?;
        Ok(staging)
    }

    fn handle_launch(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
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
