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

use crate::ffi::{
    launch_host_function, memcpy_dtoh_async, memcpy_htod_async, memset_d8_async,
    CudaContext, CudaSlice, CudaStream,
};
use std::sync::Arc;
use crate::command::{Ack, Command};
use crate::launch::issue_launch;
use crate::state::{DeviceSelector, GpuCtx, KernelCache, Staging};
use owl_iface::contract::{Arg, Bytes, Dtype, LaunchMsg};
use crate::state::{StreamId, STREAM_COMPUTE, STREAM_D2H, STREAM_H2D};
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
    fn take(&self, min_bytes: usize) -> Option<Box<dyn owl_iface::contract::PinnedRegion + Send>> {
        let mut v = self.free.lock().unwrap();
        // 2026-10-12 装载定谳:原策略 = 首个 `len >= 请求` 的桶 —— 小请求
        // (norm/小线性)会抢走 44MB 大桶,随后大权重请求无桶可用 → miss →
        // cudaHostAlloc(44MB) ≈ 300ms/次(装载期 ~100 次 miss = 30s)。
        // 改 best-fit:取满足条件的最小桶,大桶留给大请求。
        let i = v
            .iter()
            .enumerate()
            .filter(|(_, b)| b.as_bytes().len() >= min_bytes)
            .min_by_key(|(_, b)| b.as_bytes().len())
            .map(|(i, _)| i)?;
        Some(v.swap_remove(i))
    }
    fn put(&self, b: Box<dyn owl_iface::contract::PinnedRegion + Send>) {
        self.free.lock().unwrap().push(b);
    }
}

/// 刀1.6 结构清理(P0.3):server 探针开关,GpuServer 构造期一次性解析
/// (进程内不变)—— 热路径(每发射 2 次 var_os)清零。
#[derive(Clone, Copy, Debug, Default)]
struct ServerProbes {
    /// OWL_GPU_PROF(逐核归因;发射后同步)
    gpu_prof: bool,
    /// OWL_LAUNCH_TIME(host 提交计时)
    launch_time: bool,
    /// OWL_LAUNCH_SYNC(逐发射同步)
    launch_sync: bool,
    /// OWL_D2H_PROF(dtoh 分相)
    d2h_prof: bool,
    /// OWL_CAP_PROF(捕获期插桩)
    cap_prof: bool,
    /// OWL_SRV_TIMING(逐命令/图节点计时)
    srv_timing: bool,
    /// OWL_DEBUG(装配指针调试打印)
    debug: bool,
}

impl ServerProbes {
    /// **入口侧组装器**:设备旗标自 DiagOpts 一体搬运,debug 自引擎探针
    fn from_parts(diag: &crate::state::DiagOpts, debug: bool) -> Self {
        Self {
            gpu_prof: diag.gpu_prof,
            launch_time: diag.launch_time,
            launch_sync: diag.launch_sync,
            d2h_prof: diag.d2h_prof,
            cap_prof: diag.cap_prof,
            srv_timing: diag.srv_timing,
            debug,
        }
    }
}

mod harvest;

/// actor 编号(全局静态计数;DeadBlock 跨账本定谳用:每个 GpuServer = 一个
/// 独立账本,id 从 1 各自计数)
static ACTOR_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

pub struct GpuServer {
    /// 本 actor 的账本编号(打点/定谳用)
    pub(crate) actor_tag: u64,
    rx: mpsc::Receiver<Command>,
    selector: DeviceSelector,
    /// 设备诊断/调优选项(入口显式传入;Default = 全关。显式依赖律:
    /// 原 OWL_SRV_TIMING/CAP_PROF/LAUNCH_SYNC/FREE_LEGACY/GRAPH_FLAGS/
    /// CAPTURE_SLAB_MB/CUDA_HOME 散点直读 → DiagOpts 一体携带)
    diag: crate::state::DiagOpts,
    pool: std::sync::Arc<PinnedPool>,
    /// 上线握手(可选;便捷组装路径用它回传设备上线结果)
    boot: Option<mpsc::Sender<Result<(), String>>>,
    ctx: Option<GpuCtx>,
    kernels: KernelCache,
    /// 算子注册表(M4 cutover:foreign 分派唯一入口;boot 装配,P4 门)
    registry: Option<owl_kernels::registry::OpRegistry>,
    /// 生产算子白名单(dispatch.allowed_ops;None/空 = 门关;2026-10-13 P4)
    allowed_ops: Option<std::sync::Arc<Vec<String>>>,
    /// 执行引擎(kernels::device;LaunchVal 值表发射 + 装载缓存)
    exec: owl_kernels::device::Exec,
    /// **发射账**(cuda.trace_ring > 0 时建;见 crate::trace 模块头)
    trace: Option<std::sync::Arc<crate::trace::TraceRing>>,
    /// scratch 物理账本(家族 runtime 的 alloc 落此;append-only,
    /// 释放 = server 销毁 —— 家族 grave 是逻辑侧,这里是物理侧)
    scratch_ledger: Vec<(u64, CudaSlice<u8>)>,
    /// 完成派发出口(host 回调只投递;派发线程执行真正的 finish)
    dispatch: Option<mpsc::Sender<Finish>>,
    /// 逐命令计时账本(OWL_SRV_TIMING;name / count / total_ns / max_ns)
    timings: Vec<(&'static str, u64, u128, u128)>,
    /// 探针开关(构造期解析;热路径零 env 查询)
    probes: ServerProbes,
    /// 诊断下沉口(注入面;见 [`owl_shared::log::LogSink`]):server 用它
    /// 打日志,并在**发射/回收**路径记逐算子耗时。
    sink: owl_shared::log::LogSink,
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
        Self::with_diag(rx, selector, boot, crate::state::DiagOpts::default(), false)
    }

    /// 带设备诊断选项构造(显式依赖律入口;`new` = Default 便捷糖;
    /// debug = 引擎探针侧通用诊断旗标)
    pub fn with_diag(
        rx: mpsc::Receiver<Command>,
        selector: DeviceSelector,
        boot: Option<mpsc::Sender<Result<(), String>>>,
        diag: crate::state::DiagOpts,
        debug: bool,
    ) -> Self {
        Self::with_diag_sink(rx, selector, boot, diag, debug, owl_shared::log::LogSink::none())
    }

    /// 带**诊断下沉口**构造(注入面;GPU server 构造函数持 sink)。
    /// 实现方一般是 `owl_shared::log::sink()`(日志 + metrics 双出口);
    /// `LogSink::none()` = 全静默(测试/无宿主场景)。
    pub fn with_diag_sink(
        rx: mpsc::Receiver<Command>,
        selector: DeviceSelector,
        boot: Option<mpsc::Sender<Result<(), String>>>,
        diag: crate::state::DiagOpts,
        debug: bool,
        sink: owl_shared::log::LogSink,
    ) -> Self {
        Self {
            actor_tag: ACTOR_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            rx,
            selector,
            pool: std::sync::Arc::new(PinnedPool::default()),
            boot,
            ctx: None,
            kernels: KernelCache::new(),
// (diag 自身随 Self 移动,克隆 include 后置)
            registry: None,
            allowed_ops: diag.allowed_ops.clone(),
            exec: owl_kernels::device::Exec::new(),
            scratch_ledger: Vec::new(),
            dispatch: None,
            timings: Vec::new(),
            probes: ServerProbes::from_parts(&diag, debug),
            trace: {
                // 发射账:容量 > 0 才建;建好即**发布**给 app 侧查询面
                let r = crate::trace::TraceRing::new(diag.trace_ring);
                if r.enabled() {
                    crate::trace::publish(r.clone());
                    Some(r)
                } else {
                    None
                }
            },
            diag,
            sink,
        }
    }

    /// 分配落定入账(id + 设备地址;地址 = 生命周期判据的钥匙)
    fn trace_alloc_done(&self, id: u64) {
        let Some(tr) = self.trace.as_ref() else { return };
        let Ok(st) = self.ctx().stream(crate::state::STREAM_COMPUTE) else {
            return;
        };
        let ptr = self.ctx().block_ptr(id, st).map(|(p, _)| p).unwrap_or(0);
        let len = self.ctx().block_len(id).unwrap_or(0);
        tr.push(
            crate::trace::TraceEntry::new("alloc_done", "compute")
                .id(id)
                .ptrs([ptr])
                .bytes(len),
        );
    }

    /// 命令 → 发射账目(派发前调用;块 id 顺带解析成设备地址 —— 地址复用判据)
    fn trace_entry_of(&self, cmd: &Command) -> crate::trace::TraceEntry {
        use crate::trace::TraceEntry as E;
        let resolve = |ids: &[u64]| -> Vec<u64> {
            let Ok(st) = self.ctx().stream(crate::state::STREAM_COMPUTE) else {
                return Vec::new();
            };
            ids.iter()
                .map(|id| self.ctx().block_ptr(*id, st).map(|(p, _)| p).unwrap_or(0))
                .collect()
        };
        match cmd {
            Command::Launch { msg, .. } => {
                let ids: Vec<u64> = msg
                    .args
                    .iter()
                    .filter_map(|a| match a {
                        Arg::Block { id, .. } => Some(*id),
                        Arg::BlockSlice { id, .. } => Some(*id),
                        _ => None,
                    })
                    .collect();
                let ptrs = resolve(&ids);
                // 契约:最后一个 Block/BlockSlice = 输出(launch.rs 回传口径)
                let out_id = msg
                    .args
                    .iter()
                    .rev()
                    .find_map(|a| match a {
                        Arg::Block { id, .. } => Some(*id),
                        Arg::BlockSlice { id, .. } => Some(*id),
                        _ => None,
                    })
                    .unwrap_or(0);
                let mut e = E::new("launch", "compute")
                    .name(msg.kernel.name.clone())
                    .grid(msg.grid, msg.block, msg.shared_mem)
                    .bytes(msg.out_elems)
                    .ids(ids)
                    .ptrs(ptrs);
                e.out_id = out_id;
                e
            }
            Command::Htod { data, .. } => E::new("htod", "h2d").bytes(data.len()),
            Command::HtodChunk { block, data, .. } => {
                E::new("htod_chunk", "h2d").bytes(data.len()).id(*block)
            }
            Command::HtodChunks { writes, .. } => E::new("htod_chunks", "h2d")
                .bytes(writes.iter().map(|w| w.2.len()).sum())
                .ids(writes.iter().map(|w| w.0)),
            Command::UploadPinned { dst, len_bytes, .. } => {
                E::new("upload_pinned", "h2d").bytes(*len_bytes).id(dst.id)
            }
            Command::Dtoh { id, want_bytes, .. } => {
                E::new("dtoh", "d2h").bytes(*want_bytes).id(*id)
            }
            Command::Alloc { n_bytes, zero, .. } => E::new("alloc", "compute")
                .bytes(*n_bytes)
                .name(if *zero { "zero" } else { "uninit" }),
            Command::MemsetZero { block, len_bytes, .. } => {
                let ptrs = resolve(&[*block]);
                E::new("memset_zero", "compute").bytes(*len_bytes).id(*block).ptrs(ptrs)
            }
            Command::Free { ids, .. } => {
                // 回收前解析(此后账房已移除)⇒ free 条目自带地址
                let ptrs = resolve(ids);
                E::new("free", "").ids(ids.clone()).ptrs(ptrs)
            }
            Command::CopyBlock { src, dst, len_bytes, .. } => {
                E::new("copy_block", "compute").bytes(*len_bytes).ids([*src, *dst])
            }
            Command::CopyBatch { copies, .. } => {
                let ids: Vec<u64> = copies.iter().flat_map(|c| [c.0, c.2]).collect();
                let ptrs = resolve(&ids);
                E::new("copy_batch", "compute")
                    .bytes(copies.iter().map(|c| c.4).sum())
                    .ids(ids)
                    .ptrs(ptrs)
            }
            Command::Sync { .. } => E::new("sync", ""),
            Command::GraphBegin { .. } => E::new("graph_begin", ""),
            Command::GraphEnd { .. } => E::new("graph_end", ""),
            Command::GraphLaunch { .. } => E::new("graph_launch", "compute"),
            Command::GraphAbort { .. } => E::new("graph_abort", ""),
            Command::GraphDestroy { .. } => E::new("graph_destroy", ""),
            Command::AllocPinned { bytes, .. } => E::new("alloc_pinned", "").bytes(*bytes),
            Command::Close { .. } => E::new("close", ""),
        }
    }

    /// 事件循环主入口(server 线程生命周期 = 循环生命周期)。
    /// 首行上线设备(本线程 bind_to_thread 一次到位)+ 起派发线程。
    pub fn run(mut self) -> Result<(), String> {
        let ctx = GpuCtx::new(&self.selector, self.diag.clone(), self.actor_tag as u32, self.sink);
        match ctx {
            Ok(c) => {
                self.ctx = Some(c);
                // M4 cutover:算子注册表 boot 装配(P4 装配门;捕获窗前 ——
                // "捕获内首次初始化"事故类结构性消灭)。失败 = 拒启。
                //
                // ⚠️ **boot 握手点(2026-10-15 修正)**:原实现把 `boot.send(Ok)`
                // 放在"设备 ctx 上线"处 —— 而注册表装配(P4 门)**在它之后** ⇒
                // 门拒启时握手已 Ok,拒启被掩盖:`spawn_with` 返回成功、引擎继续
                // 启动、向已退出的 server 发命令 → 引擎线程卡住 → `ready` 永不发
                // → **主进程永挂且无监听**(实测:门触发时 54 线程存活 / health 无响应 /
                // warn 档下零输出)。现握手移到**boot 全链成功之后**。
                {
                    let mut res = GpuRes {
                        actor_tag: self.actor_tag,
                        ctx: self.ctx.as_mut().expect("ctx 上线"),
                        scratch: &mut self.scratch_ledger,
                    };
                    let mut exec = std::mem::take(&mut self.exec);
                    // 白名单硬门 + 池写集审计 + 装配期自检全在注册表 boot 内
                    // (设计 §八 P4/P5:装配门全链一个入口,失败 = 拒启)
                    let allowed: Option<Vec<&str>> = self
                        .allowed_ops
                        .as_ref()
                        .map(|v| v.iter().map(|s| s.as_str()).collect());
                    let r = owl_kernels::registry::OpRegistry::boot_with(
                        &mut res,
                        &mut exec,
                        allowed.as_deref(),
                    );
                    self.exec = exec;
                    match r {
                        Ok(reg) => self.registry = Some(reg),
                        Err(e) => {
                            // 拒启必须可见:LVL_ERROR(任何档位都在门内)+ **上握手**
                            // (否则 spawn_with 拿不到失败原因,只剩"通道关闭")
                            let msg = format!("算子注册表装配失败(拒启): {e}");
                            self.sink.logf(
                                owl_shared::log::LVL_ERROR,
                                "kernels",
                                format_args!("❌ {msg}"),
                            );
                            if let Some(boot) = self.boot.take() {
                                let _ = boot.send(Err(msg.clone()));
                            }
                            return Err(msg);
                        }
                    }
                }
                // ★ boot 全链成功 ⇒ 才发"就绪"(引擎侧此刻才可安全发命令)
                if let Some(boot) = self.boot.take() {
                    let _ = boot.send(Ok(()));
                }
            }
            Err(e) => {
                if let Some(boot) = self.boot.take() {
                    let _ = boot.send(Err(e.clone()));
                }
                return Err(e);
            }
        }

        // ── CUPTI 进程内归因(feature `cupti` + `perf level = gpu`)──
        // 只有显式武装时才装载采集器:默认 `host` 档 **根本不调**
        // `owl_cupti_start`(零开销,二进制路径不变)。图内逐核归因的唯一
        // 来源(见 perf-framework §十一;decode/verify 走图回放 ⇒ 解释器
        // 钩子与 gpu_prof 都看不见里面)。
        #[cfg(feature = "cupti")]
        if self.sink.gpu_armed() {
            match crate::cupti::start(self.sink) {
                Ok(()) => self.sink.logf(
                    owl_shared::log::LVL_INFO,
                    "cupti",
                    format_args!("CUPTI 归因已武装(逐核 GPU 时间 → gpu.kern.*/gpu.graph.*)"),
                ),
                Err(e) => self.sink.logf(
                    owl_shared::log::LVL_WARN,
                    "cupti",
                    format_args!("CUPTI 武装失败(继续跑,无逐核账): {e}"),
                ),
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
            // ② 队列等待 = 客户端入队 → 本线程被唤醒出队(tag `srv.queue.<kind>`)
            // 与 `srv.tx`/`srv.op`/`srv.wait` 三段拼出一次往返的完整分解。
            if self.sink.timing() {
                let kind = cmd.kind();
                self.sink
                    .timer(&format!("srv.queue.{kind}"), cmd.created_at().elapsed());
            }
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
        // CUPTI 收尾(先 flush 再停;排空线程在 stop 内 join)
        #[cfg(feature = "cupti")]
        {
            if crate::cupti::running() {
                crate::cupti::stop();
                let (tot, drop, batch, graph) = crate::cupti::stats();
                self.sink.logf(
                    owl_shared::log::LVL_INFO,
                    "cupti",
                    format_args!("CUPTI 收尾:记录 {tot}(图内 {graph})/丢 {drop}/排空批 {batch}"),
                );
            }
        }
        // server 侧逐命令分相汇总(OWL_SRV_TIMING;与 [load] 客户端探针互补)
        if self.probes.srv_timing && !self.timings.is_empty() {
            self.sink.logf(owl_shared::log::LVL_DEBUG, "srv-timing", format_args!("===== handler 分相(count / total / max) ====="));
            let mut rows = self.timings.clone();
            rows.sort_by_key(|(_, _, total, _)| std::cmp::Reverse(*total));
            for (name, count, total, max) in rows {
                self.sink.logf(owl_shared::log::LVL_DEBUG, "srv-timing", format_args!("  {:<14} x{:<5} total={:>8.1}ms max={:>8.1}ms",
                    name,
                    count,
                    total as f64 / 1e6,
                    max as f64 / 1e6
                ));
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
        // cublas 句柄先于上下文消亡(2026-09-26 定谳:teardown 时
        // cublasDestroy 撞已拆上下文 → libcublasLt SIGSEGV;显式 take =
        // 句柄在活上下文内销毁,gdb bt 实证)。M4 后句柄住 registry
        // (cublas runtime 内),同律 = registry 整体先于 ctx 掉。
        //
        // ⚠️ 收尾走**显式 `close()`**(析构唯一官方出口),不裸 drop:
        // 裸 drop 会让 FamilyGuard 的"未经 close() 即 drop"兜底告警**在每次
        // 正常退出时刷 N 条** ⇒ 告警疲劳,把真漏淹没(2026-10-13 复核实测)。
        // 次序不变:仍在 ctx 拆掉之前。
        if let Some(reg) = self.registry.take() {
            if let Err(e) = reg.close() {
                self.sink.logf(owl_shared::log::LVL_INFO, "kernels", format_args!("registry 收尾失败(已尽力逐族 close): {e}"));
            }
        }
        self.scratch_ledger.clear();
        if let Some(ctx) = &self.ctx {
            for sid in [STREAM_H2D, STREAM_COMPUTE, STREAM_D2H] {
                if let Ok(s) = ctx.stream(sid) {
                    let _ = s.synchronize();
                }
            }
            // E1:事件句柄在流栅栏之后销毁(收摊唯一出口)
            ctx.destroy_events();
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

    /// 命令的**目标流**(E1 记账表:哪些命令在 GPU 上落活、落在哪条流)。
    ///
    /// 与 `trace_entry_of` 的流标签同表(那里是给 trace 看的字符串;本表是
    /// 给事件记账看的 StreamId)。**扩充命令面时必须同步**(E2 复合命令按
    /// 自身语义定流)。
    fn cmd_stream(c: &Command) -> Option<StreamId> {
        match c {
            // H2D 流:一切 host→device 搬运
            Command::Htod { .. }
            | Command::HtodChunk { .. }
            | Command::HtodChunks { .. }
            | Command::UploadPinned { .. } => Some(STREAM_H2D),
            // COMPUTE 流:发射 / 清零分配 / 设备内拷贝 / 图回放
            Command::Launch { .. }
            | Command::Alloc { .. }
            | Command::MemsetZero { .. }
            | Command::CopyBlock { .. }
            | Command::CopyBatch { .. }
            | Command::GraphLaunch { .. } => Some(STREAM_COMPUTE),
            // D2H 流:一切 device→host 回读
            Command::Dtoh { .. } => Some(STREAM_D2H),
            // 无 GPU 活/图控制/同步/宿主面:不记账
            _ => None,
        }
    }

    fn dispatch(&mut self, cmd: Command) {
        // 逐命令计时(OWL_SRV_TIMING 门控;卸载时汇总 —— server 侧分相,
        // 与客户端 [load] 探针互补:handler 真实执行时间,含阻塞段)
        let timing = self.probes.srv_timing;
        let (t0, name) = if timing {
            (Some(std::time::Instant::now()), Some(Self::cmd_name(&cmd)))
        } else {
            (None, None)
        };
        // E1:本命令的落流(记账用;必须在 dispatch_inner 吞掉 cmd 之前取)
        let sid = Self::cmd_stream(&cmd);
        self.dispatch_inner(cmd);
        // **E1 纯记账**:把本命令的完成点记到该流尾事件(不等待任何东西;
        // 序消费在 E2/E3)。捕获期自动跳过(见 `record_tail`)。
        if let Some(sid) = sid {
            let t_ev = if timing { Some(std::time::Instant::now()) } else { None };
            if let Err(e) = self.ctx_mut().record_tail(sid) {
                // 记账面失败不改命令语义:响亮但不改回执
                self.sink.logf(
                    owl_shared::log::LVL_WARN,
                    "ev",
                    format_args!("尾事件记账失败(stream {sid})—— 不影响本命令: {e}"),
                );
            }
            if let Some(t1) = t_ev {
                self.timing_acc("srv.ev_record", t1.elapsed());
            }
        }
        if let (Some(t0), Some(name)) = (t0, name) {
            let dt = t0.elapsed();
            self.timing_acc(name, dt);
        }
    }

    /// 分相累加(count / total / max;OWL_SRV_TIMING 汇总表的唯一累加口)
    fn timing_acc(&mut self, name: &'static str, dt: std::time::Duration) {
        let slot = if let Some(pos) = self.timings.iter().position(|(n, ..)| *n == name) {
            &mut self.timings[pos]
        } else {
            self.timings.push((name, 0u64, 0u128, 0u128));
            let last = self.timings.len() - 1;
            &mut self.timings[last]
        };
        slot.1 += 1;
        slot.2 += dt.as_nanos();
        slot.3 = slot.3.max(dt.as_nanos());
    }

    /// 命令名(分相计时标签)。**单一真源 = `Command::kind()`**
    /// (2026-10-15 审查补:旧实现本地 match 出一张同名表,与 `kind()` 构成
    /// "命令名两处写"的漂移源 —— 与刚消灭的算子名两处写同构)。
    fn cmd_name(c: &Command) -> &'static str {
        c.kind()
    }

    fn dispatch_inner(&mut self, cmd: Command) {
        // **发射账**(2026-10-14):派发前记一条 —— 得到"host 视角的事实发射序
        // + 流归属 + 块地址"。关账(容量 0)= 此处一个 None 判定,零开销。
        if let Some(tr) = self.trace.as_ref() {
            let e = self.trace_entry_of(&cmd);
            tr.push(e);
        }
        // G2 护栏:捕获期间放行清单 = **Launch / Alloc(切 slab) / MemsetZero /
        // GraphEnd / Close** —— Launch 与 Alloc 由路由策略天然落在捕获流
        // (COMPUTE);**MemsetZero 有意放行**(它是合法图节点,GDN 逐 turn 状态
        // 重置靠它);GraphEnd/Close 是收尾幂等口。其余(Htod/Dtoh/Sync/
        // GraphLaunch)一律**结构化拒绝** —— 或含非法捕获语义,或污染捕获拓扑。
        // (注释原写"仅 Launch/Alloc",与代码漂移;2026-10-15 审查补正)
        if self.ctx().capture_stream() {
            return match cmd {
                Command::Launch { msg, ack } => self.handle_launch(msg, ack),
                Command::Alloc { n_bytes, elems, zero, ack } => self.handle_alloc(n_bytes, elems, zero, ack),
                Command::MemsetZero { block, offset_bytes, len_bytes, ack } => {
                    self.handle_memset_zero(block, offset_bytes, len_bytes, ack)
                }
                Command::GraphEnd { ack } => ack.send(self.ctx_mut().graph_end()),
                // 窗内出错/空窗的收尾手段 ⇒ 捕获期必须允许(否则只能等
                // 窗自然结束,slab 留尸)
                Command::GraphAbort { ack } => ack.send(self.ctx_mut().graph_abort()),
                // **M1:捕获窗内放行 Free** —— 锚定活性回收(reclaim_capture)
                // 的必要条件。窗内 `free_blocks` 三条路**零 CUDA 调用**:
                // ①跨流事件栅栏本就在 `capture.is_none()` 守卫下跳过;
                // ②Carved 块 → slab 内 arena 归还(纯 host 簿记);
                // ③Owned 块 → **防御性不释放**(见 free_blocks 内守卫;窗内
                //    不该出现 Owned 死块,释放会烙 MEM_FREE 节点)
                Command::Free { ids, ack } => self.handle_free(ids, ack),
                Command::Close { ack } => ack.send(Ok(())), // 防御性幂等回执
                other => Self::reject(other, self.sink),
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
            Command::CopyBatch { copies, ack } => self.handle_copy_batch(copies, ack),
            Command::Launch { msg, ack } => self.handle_launch(msg, ack),
            Command::Sync { ack } => self.handle_sync(ack),
            Command::GraphBegin { ack } => ack.send(self.ctx_mut().graph_begin()),
            Command::GraphEnd { ack } => ack.send(self.ctx_mut().graph_end()),
            Command::GraphAbort { ack } => ack.send(self.ctx_mut().graph_abort()),
            Command::GraphDestroy { graph, ack } => ack.send(self.ctx_mut().graph_destroy(graph)),
            Command::GraphLaunch { graph, ack } => {
                ack.send(self.ctx_mut().graph_launch(graph))
            }
            // 理论不可达:run() 顶层已截获 Close;防御性回执
            Command::Close { ack } => ack.send(Ok(())),
        }
    }

    /// 捕获期违规命令的结构化拒绝
    fn reject(cmd: Command, sink: owl_shared::log::LogSink) {
        Self::reject_with(cmd, "图捕获进行中:该操作被拒 —— 捕获期仅允许 \
             Launch/Alloc;同步/搬运/图操作会破坏图捕获".to_string(), sink);
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
            Command::CopyBatch { ack, .. } => closed!(ack),
            Command::Launch { ack, .. } => closed!(ack),
            Command::Sync { ack, .. } => closed!(ack),
            Command::GraphBegin { ack, .. } => closed!(ack),
            Command::GraphEnd { ack, .. } => closed!(ack),
            Command::GraphLaunch { ack, .. } => closed!(ack),
            Command::GraphAbort { ack, .. } => closed!(ack),
            Command::GraphDestroy { ack, .. } => closed!(ack),
        }
    }

    fn reject_with(cmd: Command, why: String, sink: owl_shared::log::LogSink) {
        macro_rules! reject { ($ack:expr) => { $ack.send(Err(ModelError::Msg(why.clone()))) } }
        let name = |c: &Command| match c {
            Command::GraphBegin { .. } => "GraphBegin",
            Command::GraphEnd { .. } => "GraphEnd",
            Command::GraphLaunch { .. } => "GraphLaunch",
            Command::GraphAbort { .. } => "GraphAbort",
            Command::GraphDestroy { .. } => "GraphDestroy",
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
            Command::CopyBatch { .. } => "CopyBatch",
            Command::Sync { .. } => "Sync",
            Command::Launch { .. } => "Launch",
            Command::Close { .. } => "Close",
        };
        sink.logf(owl_shared::log::LVL_WARN, "owl-gpu", format_args!("捕获期拒绝: {} —— {why}", name(&cmd)));
        match cmd {
            Command::GraphBegin { ack } => reject!(ack),
            Command::GraphEnd { ack } => reject!(ack),
            Command::GraphLaunch { ack, .. } => reject!(ack),
            Command::GraphAbort { ack, .. } => reject!(ack),
            Command::GraphDestroy { ack, .. } => reject!(ack),
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
            Command::CopyBatch { ack, .. } => reject!(ack),
            Command::Launch { ack, .. } => reject!(ack),
            Command::Close { ack } => ack.send(Ok(())), // 幂等
        }
    }

    /// alloc 段分相(与客户端 `interp.host.seg.alloc` 对账;差值 = 管道)
    fn handle_alloc(
        &mut self,
        n_bytes: usize,
        elems: usize,
        zero: bool,
        ack: Ack<Result<Bytes, ModelError>>,
    ) {
        let t0 = std::time::Instant::now();
        self.handle_alloc_inner(n_bytes, elems, zero, ack);
        self.sink.timer("srv.seg.alloc", t0.elapsed());
    }

    fn handle_alloc_inner(
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
                self.trace_alloc_done(id);
                if zero {
                    let stream = self.ctx().stream(STREAM_COMPUTE)?.clone();
                    let (dptr, _) = self.ctx().block_ptr(id, &stream)?;
                    if self.probes.cap_prof {
                        self.sink.logf(owl_shared::log::LVL_DEBUG, "cap-prof", format_args!("carve-memset {}B", n_bytes));
                    }
                    unsafe { memset_d8_async(dptr, 0, n_bytes, stream.cu_stream()) }
                        .map_err(|e| ModelError::Msg(format!("carve memset: {e:?}")))?;
                }
                // 句柄占位(事实面由客户端 `alloc(dtype, elems)` 侧定形)
                Ok(Bytes::new(id, elems, Dtype::U8, 0))
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
                // warmup 计量(slab 定量:owl_shared::slab_hint;仅真分配,
                // carve 分支不计 —— 捕获窗用 slab 不产生新账)
                owl_shared::slab_hint::meter_add(n_bytes as u64);
                // B6.5:live 分配音账(释放侧 = free_blocks)
                owl_shared::vram::live_add(n_bytes as i64);
                if zero {
                    stream
                        .memset_zeros(&mut slice)
                        .map_err(|e| ModelError::Msg(format!("alloc memset: {e:?}")))?;
                }
                // 诊断开关(sync_bit::ALLOC):alloc/memset 后同步归因
                if self.diag.sync_at(owl_shared::config::sync_bit::ALLOC) {
                    if let Err(e) = stream.synchronize() {
                        return Err(ModelError::Msg(format!(
                            "alloc-sync(n={n_bytes}): {e:?}"
                        )));
                    }
                }
                let id = self.ctx_mut().new_block(slice);
                // 发射账:**分配落定**记 id + 设备地址(trace 条目 alloc 本身
                // 无 id —— 生命周期判据要它才能把地址钉到块上)
                self.trace_alloc_done(id);
                // M1:warmup 影子 arena 登记(探针关 = 零开销;峰值口径的
                // 来源 —— `probe_free` 在 free_blocks 里配对)
                owl_shared::slab_hint::probe_alloc(id, n_bytes);
                // 句柄占位(事实面由客户端 `alloc(dtype, elems)` 侧定形)
                Ok(Bytes::new(id, elems, Dtype::U8, 0))
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
        if self.ctx().capture_stream() && self.probes.cap_prof {
            self.sink.logf(owl_shared::log::LVL_DEBUG, "cap-prof", format_args!("memset_zero {}B block{block}", len_bytes));
        }
        let result = (|| {
            // 边界校验(2026-10-15 补口):越界写会落到邻块 —— 设备内无保护
            self.ctx().check_range(block, offset_bytes, len_bytes, "memset")?;
            let (dptr, _) = self.ctx().block_ptr(block, &stream).map_err(|e| {
                // 带上块 id(谁在 memset 一个不存在的块;DeadBlock 定位)
                ModelError::Msg(format!("[memset-arg] id={block} len={len_bytes} :: {e}"))
            })?;
            unsafe {
                crate::ffi::memset_d8_async(dptr + offset_bytes as u64, 0, len_bytes, stream.cu_stream())
            }
            .map_err(|e| ModelError::Msg(format!("memset_zero: {e:?}")))
        })();
        ack.send(result);
    }
    /// 逐算子服务端总时(注入口:GPU server 构造函数注入的 sink)。
    ///
    /// 口径 = 命令进入 handler → 发射调用返回(含参数装配/取核/准入/发射);
    /// 与客户端 `interp.host.seg.launch`(18µs 级)**对账** ⇒ 差值 = 管道
    /// (actor 队列往返 + 回执),是「宿主链 93% 在 launch+alloc」的下一层分解。
    fn handle_launch(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        // **计时器关账 = 零开销**(2026-10-15;原 FIXME 落地):
        // 与 `srv.tx`(gpu_client)/`srv.queue`(本文件派发循环)同款谓词
        // `LogSink::timing()` = timer 面在场 ∧ `PERF_BUILD` ∧ level≥HOST ——
        // 先问再建,未武装就不做 `Instant::now` / `kernel.name.clone()` /
        // `format!`(本热路径的全部常驻税)。
        //
        // 语义对齐:`PERF_BUILD = debug_assertions ∨ feature=perf` ⇒
        // 非 perf 的 release 里 `srv.op.*` 与其余 SRV 段一致地关账
        // (要逐发射分解就 `--features perf`;原实现只漏了这一段 —— 三段
        // 里只有它无门,是 `srv.op 10.86%` 数字在非 perf 档下也吃 CPU 的原因)。
        // 不缓存到构造期:`/debug/perf/level` 可运行期升降。
        if !self.sink.timing() {
            return self.handle_launch_probed(msg, ack);
        }
        let t0 = std::time::Instant::now();
        let name = msg.kernel.name.clone();
        self.handle_launch_probed(msg, ack);
        self.sink.timer(&format!("srv.op.{name}"), t0.elapsed());
    }

    fn handle_launch_probed(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        // GPU 逐核归因探针(OWL_GPU_PROF=1;2026-10-01 50tok/s 分账立案):
        // 每发射后同步 COMPUTE 流 → 本核隔离运行,host 计时 ≈ 核 GPU 纯时
        // (含 ~10µs sync 往返底噪);动态 tag `gpu.{核名}` 入 owl-shared
        // metrics(同进程线程,store 共享),测试侧 query(prefix "gpu.")
        // 出热点分账。捕获期跳过 —— 图回放是整图单发射,逐核归因由
        // OWL_NO_GRAPH eager 窗提供;与 OWL_LAUNCH_TIME(host 提交 µs)
        // 互补:本探针答「GPU 时花在哪」,彼答「host 提交多贵」。
        let gpu_prof = self.probes.gpu_prof;
        if gpu_prof && !self.ctx().capture_stream() {
            let name = msg.kernel.name.clone();
            // 形状特化 tag:marlin 按 m×n×k 拆账(双峰形状定位,2026-10-03)
            let shape_tag = match msg.kernel.name.as_str() { // v2
                n if n == owl_kernels::family::marlin::GEMM_W4A16
                    || n == owl_kernels::family::marlin::GEMM_W4A16_AWQ =>
                {
                    let mut dims: Vec<u64> = Vec::new();
                    for a in &msg.args {
                        if let Arg::U64(v) = a {
                            dims.push(*v);
                        }
                    }
                    // 槽序:[T a, T b, T out, T (scales/zs/ws/ctmp)..., sz m, sz k, sz n, sz gs]
                    // m/k/n = 末 4 sz 的前三个
                    if dims.len() >= 3 {
                        let (m, k, nn) = (dims[dims.len() - 4], dims[dims.len() - 3], dims[dims.len() - 2]);
                        format!("gpu.marlin.{m}x{nn}x{k}")
                    } else {
                        format!("gpu.{name}")
                    }
                }
                _ => format!("gpu.{name}"),
            };
            let t0 = std::time::Instant::now();
            // 关账语义:本臂由 `gpu_prof` 布尔门控(未开 = 零动作;不需要宏
            // —— 装配期/入口期的 `handle_launch` 计时器已按
            // `LogSink::timing()` 先问再建,见其注)。开启档**本就要**逐核
            // `stream.synchronize()`(归因定义如此),那笔 ≫ 计时器本身。
            self.handle_launch_inner(msg, ack);
            if let Ok(stream) = self.ctx().stream(STREAM_COMPUTE) {
                let _ = stream.synchronize(); // 归因同步(错误由 sticky 面另报)
            }
            let dt: std::time::Duration = t0.elapsed();
            owl_shared::metrics::with_metrics_store(|s| {
                s.timer_record_tag(&shape_tag, dt, file!(), 0);
                s.timer_record_tag(&format!("gpu.{name}"), dt, file!(), 0);
            });
            return ;
        }
        // 逐发射计时(OWL_LAUNCH_TIME=1;捕获期 warmup = 全模型 eager 一遍,
        // 一次跑完全谱;capture 回放期自动关闭 —— 回放是整图一发射)
        if self.probes.launch_time && !self.ctx().capture_stream() {
            let t = std::time::Instant::now();
            let name = msg.kernel.name.clone();
            self.handle_launch_inner(msg, ack);
            self.sink.logf(owl_shared::log::LVL_DEBUG, "launch-time", format_args!("{:?} {}", t.elapsed(), name));
            return ;
        }
        self.handle_launch_inner(msg, ack)
    }

    /// **算子准入 L3 哨兵**(唯一调用点;2026-10-13 P3)。
    ///
    /// 覆盖**两条发射路**:原生 NVRTC 与家族 FFI —— 它们在这里汇合,
    /// 所以门放这里而不是 models(design §五「由上层统一调用」的落地,
    /// 实测唯一能同时管住两路的位置)。
    ///
    /// 未建表算子(`meta() == None`,处于"仅引入"状态)= 放行但无门 ——
    /// P4 的生产白名单硬门才会把"未建表进生产"变成拒启。
    fn admission_check(&self, msg: &LaunchMsg) -> Result<(), owl_kernels::contract::OpError> {
        use owl_kernels::bootcheck::{CheckCtx, Geom};
        // ① 白名单硬门(deny-by-default;门关时跳过)
        if let Some(list) = &self.allowed_ops {
            if !list.iter().any(|n| n == &msg.kernel.name) {
                return Err(owl_kernels::contract::OpError::Contract {
                    op: msg.kernel.name.clone(),
                    field: "allowed_ops",
                    expect: format!("∈ 本档白名单({} 项;dispatch.allowed_ops)", list.len()),
                    got: "未列入 —— 该算子未获准入进本生产档".into(),
                });
            }
        }
        // ② 算子自身准入声明(L3 哨兵;未建表 = 无声明 ⇒ 无检查)
        let Some(meta) = self.registry.as_ref().and_then(|r| r.meta(&msg.kernel.name)) else {
            return Ok(());
        };
        let geom = Geom {
            grid: msg.grid,
            block: msg.block,
            smem: msg.shared_mem,
            out_elems: msg.out_elems,
        };
        let ctx_ref = self.ctx.as_ref();
        // 账长探测:不在本账本/不在册 ⇒ None(哨兵跳过该槽)
        let len = |id: u64| ctx_ref.and_then(|c| c.block_len(id).ok());
        let ctx = CheckCtx { op: meta.entry, args: &msg.args, geom: &geom, len: &len };
        meta.check(&ctx)
    }

    fn handle_launch_inner(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
        // 准入门(foreign/native 汇合点;失败 = 指名道姓的 Contract 错)
        if let Err(e) = self.admission_check(&msg) {
            return ack.send(Err(ModelError::Msg(format!("[算子准入] {e}"))));
        }
        // foreign-kernel 通道(M4 cutover:算子注册表唯一入口 —— 名字分派
        // 在注册表内部,server 主循环零算子知识;旧 handle_foreign_launch
        // 已退役,见 ops/ 模块)。路由判定 = 注册表 knows(boot 表自洽,
        // 零第二份清单;旧 is_foreign_name 硬编码枚举已删,2026-10-12 review D 案)
        if self.registry.as_ref().is_some_and(|r| r.knows(&msg.kernel.name)) {
            // 家族(foreign)路分相:与 native 路的 srv.seg.prep/issue 并列,
            // 使 srv.op 的可分解性覆盖两路(P5 对账需要)
            let t_fam = std::time::Instant::now();
            let sink = self.sink;
            // 字段级解构:ctx(&mut,alloc_block 需要)、registry、exec 三者借用不相交
            let Self { ctx, registry, exec, scratch_ledger, actor_tag, .. } = self;
            let out = (|| -> Result<Bytes, String> {
                let ctx = ctx.as_mut().ok_or("server 未上线")?;
                let mut res = GpuRes { ctx, scratch: scratch_ledger, actor_tag: *actor_tag };
                registry
                    .as_mut()
                    .expect("knows 已查非 None")
                    .execute(&msg, &mut res, exec)
                    .map_err(|e| e.to_string())
            })()
            .map_err(ModelError::Msg);
            sink.timer("srv.seg.fam", t_fam.elapsed());
            return ack.send(out);
        }
        let mut ack = Some(ack);
        // 下沉口(值语义,先拷贝;下面字段级解构不再借 self)
        let sink = self.sink;
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
        if capturing && self.probes.cap_prof {
            self.sink.logf(owl_shared::log::LVL_DEBUG, "cap-prof", format_args!("native launch {} out={}", msg.kernel.name, msg.out_elems));
        }
        match issue_launch(ctx, &stream, kernels, &msg, self.probes.debug, sink) {
            Ok(out_id) => {
                self.ctx_mut().note_launch(&msg.kernel.name);
                // 诊断开关(OWL_LAUNCH_SYNC=1):逐发射同步,sticky 错误
                // 归属到具体核(2026-09-27 长 ctx ILLEGAL_ADDRESS 排查;
                // 捕获期禁 sync —— 跳过,捕获正确性由哨兵③/④另保)
                if self.diag.sync_at(owl_shared::config::sync_bit::LAUNCH) && !capturing {

                    if let Err(e) = stream.synchronize() {
                        return ack.take().unwrap().send(Err(ModelError::Msg(format!(
                            "launch-sync({}): {e:?}",
                            msg.kernel.name
                        ))));
                    }
                }
                // 出参句柄:事实面(声明 dtype/rank)由**线格式的 out 槽**承载,
                // 客户端回执后按它定形;此处只给句柄(未定形占位)
                ack.take()
                    .unwrap()
                    .send(Ok(Bytes::new(out_id, msg.out_elems, Dtype::U8, 0)))
            }
            Err(e) => ack.take().unwrap().send(Err(e)),
        }
    }

    /// 中间块回收(E2a):账房移除 Owned 块归池。调用方契约 = 已收割
    /// 所需数据(dtoh 回执即 COMPUTE 排空);图捕获期由 G2 拒绝臂拦截。
    fn handle_free(&mut self, ids: Vec<u64>, ack: Ack<Result<(), ModelError>>) {
        let t0 = std::time::Instant::now();
        let freed = self.ctx_mut().free_blocks(&ids);
        self.sink.timer("srv.seg.free", t0.elapsed());
        self.sink.count("srv.count.free_blocks", freed as u64);
        if self.probes.debug {
            self.sink.logf(owl_shared::log::LVL_WARN, "owl-gpu", format_args!("free: {}/{} 块归池", freed, ids.len()));
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
            // 边界校验(2026-10-15 补口):设备内 dtod 无保护,源/目标两端各查
            self.ctx().check_range(src, src_off_bytes, len_bytes, "copy-src")?;
            self.ctx().check_range(dst, dst_off_bytes, len_bytes, "copy-dst")?;
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

    /// 批量块拷贝(E5-M4):单命令逐条入 COMPUTE 流,单 ack 兜底;
    /// 任一失败即短路回执(账房错误语义与单条一致)
    fn handle_copy_batch(
        &mut self,
        copies: Vec<(u64, usize, u64, usize, usize)>,
        ack: Ack<Result<(), ModelError>>,
    ) {
        let stream = match self.ctx().stream(STREAM_COMPUTE) {
            Ok(s) => s.clone(),
            Err(e) => return ack.send(Err(e)),
        };
        let result = (|| {
            for (src, src_off, dst, dst_off, len) in &copies {
                // 边界校验(2026-10-15 补口;spec 快照 capture/restore 走本路)
                self.ctx().check_range(*src, *src_off, *len, "copyb-src")?;
                self.ctx().check_range(*dst, *dst_off, *len, "copyb-dst")?;
                let (sptr, _) = self.ctx().block_ptr(*src, &stream)?;
                let (dptr, _) = self.ctx().block_ptr(*dst, &stream)?;
                unsafe {
                    crate::ffi::memcpy_dtod_async(
                        dptr + *dst_off as u64,
                        sptr + *src_off as u64,
                        *len,
                        stream.cu_stream(),
                    )
                }
                .map_err(|e| ModelError::Msg(format!("copy_batch: {e:?}")))?;
            }
            Ok(())
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

// ============================================================================
// GpuRes —— DeviceRes 服务端实现(M4;状态 = GpuCtx 治理面 + scratch 账本)
// ============================================================================
//
/// 算子域资源面视图:ctx(不可变治理面)+ scratch 物理账本(可变)。
/// 家族 runtime 经 kernels::device::DeviceRes 使用;解析/分配/上传/
/// 捕获查询全部对准 GpuCtx 既有机制(与旧 handler 同一机制,零新面)。
pub(crate) struct GpuRes<'a> {
    /// 可变:`alloc_block`(自检面真账本块分配)要往账房插条目(2026-10-13 P5c)
    pub ctx: &'a mut crate::state::GpuCtx,
    pub scratch: &'a mut Vec<(u64, CudaSlice<u8>)>,
    /// 本 actor 账本编号(DeadBlock 定谳:跨账本时能看出"问的是哪个账房")
    pub actor_tag: u64,
}

const OP_RES: &str = "server.res";

use owl_kernels::contract::OpError;
use owl_kernels::device::ScratchBuf;

impl GpuRes<'_> {
    /// 块 id → (设备指针, 账长字节)。`resolve`/`resolve_len` **共用**它:
    /// 两个出口走同一条解析路径,错措辞只有一份(F2 尺寸自证的前提)。
    fn resolve_span(&self, id: u64) -> Result<(u64, usize), OpError> {
        let stream = <Self as owl_kernels::device::DeviceRes>::stream(self)?;
        self.ctx.block_ptr(id, &stream).map_err(|e| OpError::Contract {
            op: OP_RES.into(),
            field: "resolve",
            expect: format!("块在册(actor_tag={} 账本)", self.actor_tag),
            got: format!("id={id} / {e}"),
        })
    }

    /// 上传公共体(`upload` / `upload_invariant` 共用;`invariant` 只影响
    /// 捕获窗入境判定,不影响拷贝语义)
    fn upload_impl(&mut self, dst: u64, src: &[u8], invariant: bool) -> Result<(), OpError> {
        // 固有方法 ⇒ trait 方法须显式限定(resolve_span 同款先例)
        let stream = <Self as owl_kernels::device::DeviceRes>::stream(self)?;
        let capturing = self.ctx.capture_stream();
        // **捕获窗入境守卫**(§4.3e):裸切片窗内上传 = 源可能生命周期 < 图
        // (栈上/可变缓冲)⇒ 回放读到已释放内存(静默错值)。不变盒恒放行。
        if capturing && !invariant {
            self.ctx.sink.count("srv.count.fam_upload_in_window", 1);
            match crate::state::fam_window_verdict(
                crate::state::FamAct::Upload { invariant: false },
                crate::state::cap_strict(),
            ) {
                Ok(()) => self.ctx.sink.logf(
                    owl_shared::log::LVL_WARN,
                    "fam-upload-in-window",
                    format_args!("捕获窗内裸切片 H2D({}B @{dst:#x})—— 源必须生命周期 ≥ 图(不变盒),否则回放静默错值", src.len()),
                ),
                Err(detail) => {
                    return Err(OpError::Contract {
                        op: OP_RES.into(),
                        field: "upload",
                        expect: "捕获窗内上传须走 upload_invariant(InvariantBox)".into(),
                        got: detail,
                    })
                }
            }
        }
        // 入账(家族面唯一账线索;A13 藏一轮的直接原因)
        <Self as owl_kernels::device::DeviceRes>::note_device(
            self,
            "fam_upload",
            if capturing { "capture" } else { "eager" },
            src.len(),
            &[dst],
        );
        let (target, cap) = self
            .scratch
            .iter()
            .find(|(p, _)| *p == dst)
            .map(|(_, s)| (s.clone(), s.len()))
            .ok_or_else(|| OpError::Contract {
                op: OP_RES.into(),
                field: "upload",
                expect: "指针在 scratch 账本".into(),
                got: format!("{dst:#x}"),
            })?;
        // **边界校验(2026-10-15 审查补)**:账本命中只证明"基址在册",不证明
        // "写得下" —— 越界会静默落**邻块**(与命令面六条同族;设备内无保护)。
        if !crate::state::range_fits(cap, 0, src.len()) {
            return Err(OpError::Contract {
                op: OP_RES.into(),
                field: "upload",
                expect: format!("src.len ≤ scratch 缓冲长度({cap}B)"),
                got: format!("{}B @{dst:#x}", src.len()),
            });
        }
        if capturing {
            // 捕获窗:async 节点(回放重放 memcpy;不变盒语义由家族保证)
            // 捕获窗:async 节点(回放重放 memcpy;不变盒语义由家族保证)
            // SAFETY:dst ∈ scratch 账本(本 impl 分配并持有);src 不变盒
            // 2026-10-13(家规:可失败返回值不可丢弃):原 `;` 丢弃入队错误
            // → 入队失败会静默变成"节点不存在"(回放时少一次 memcpy)。现上抛。
            unsafe { crate::ffi::memcpy_htod_async(dst, src, stream.cu_stream()) }
                .map_err(|e| OpError::Launch { op: OP_RES.into(), stage: owl_kernels::contract::Stage::Store, detail: format!("upload async: {e:?}") })?;
            Ok(())
        } else {
            // 2026-10-09 崩案修复试刀:async memcpy_htod 静默不写(meta 表读回全零)
            // → 改同步 result API 直写
            //
            // ⚠️ **A13(2026-10-15,compute-sanitizer 定谳)**:上述诊断是
            // **回读侧** NULL 流无序的假象(`download` = `memcpy_dtoh_sync`,
            // 与 compute 流无跨流序 ⇒ 读回全零;与 gdn_chunked.rs 453 行记载
            // 同病史);改上传为同步反而**把写侧的序彻底抹掉** —— 目标地址
            // 来自 `stream.alloc`(流序异步分配),NULL 流同步拷贝与它无任何
            // 序,会写进**尚未落定**的内存(memcheck 判据:`Use-before-alloc`,
            // 命中点即本函数与 `upload_block`)。
            // 定谳实验(同 boot、同 prompt、只切一位):mask=0 → 末行 logits
            // `28`/gap 0.91 + 不发工具;mask=64(拷贝前排空)→ `760`/gap 4.33
            // + 真发 tool_calls;两者逐字节差 = 一位之差。
            if self.ctx.diag.sync_at(owl_shared::config::sync_bit::SCRATCH_SYNC_UP) {
                stream.synchronize().map_err(|e| OpError::Launch {
                    op: OP_RES.into(),
                    stage: owl_kernels::contract::Stage::Store,
                    detail: format!("upload pre-sync: {e:?}"),
                })?;
            }
            // 生产语义(本分支):**与分配同流的 async 拷贝** —— 分配 → 拷贝 →
            // 后续核 在 COMPUTE 流上严格有序(与捕获分支同一 API;原实现的
            // 不对称仅此而已)。代价 = 0(实测 TTFT 不变)。
            // `SCRATCH_UP_LEGACY` = 退回旧 NULL 流同步拷贝(**专门复现 A13**);
            // `SCRATCH_SYNC_UP` = 保留的归因位(拷贝前排空 COMPUTE)。
            if self.ctx.diag.sync_at(owl_shared::config::sync_bit::SCRATCH_UP_LEGACY) {
                unsafe { cudarc::driver::result::memcpy_htod_sync(dst, src) }.map_err(|e| {
                    OpError::Launch {
                        op: OP_RES.into(),
                        stage: owl_kernels::contract::Stage::Store,
                        detail: format!("upload sync(legacy NULL 流): {e:?}"),
                    }
                })?;
            } else {
                unsafe { crate::ffi::memcpy_htod_async(dst, src, stream.cu_stream()) }.map_err(
                    |e| OpError::Launch {
                        op: OP_RES.into(),
                        stage: owl_kernels::contract::Stage::Store,
                        detail: format!("upload async(same-stream): {e:?}"),
                    },
                )?;
            }
            let _target = target; // 保留账本句柄(析构语义不变)
            Ok(())
        }
    }
}


impl owl_kernels::device::DeviceRes for GpuRes<'_> {
    fn context(&self) -> Result<Arc<CudaContext>, OpError> {
        Ok(self.ctx.ctx.clone())
    }
    fn stream(&self) -> Result<Arc<CudaStream>, OpError> {
        self.ctx
            .stream(crate::state::STREAM_COMPUTE)
            .map_err(|e| OpError::Launch { op: OP_RES.into(), stage: owl_kernels::contract::Stage::Load, detail: format!("stream: {e:?}") })
            .cloned()
    }
    fn resolve(&self, id: u64) -> Result<u64, OpError> {
        Ok(self.resolve_span(id)?.0)
    }
    fn resolve_len(&self, id: u64) -> Result<usize, OpError> {
        Ok(self.resolve_span(id)?.1)
    }
    fn alloc(&mut self, bytes: usize, tag: &'static str) -> Result<ScratchBuf, OpError> {
        // 与旧 handler 同机制:stream.alloc_zeros(cudarc 直分配)。append-only
        // 账本,物理释放 = server 销毁。
        //
        // ⚠️ **捕获窗入境守卫(2026-10-15;设计 §4.3e)**:旧口径是"捕获窗内由
        // 家族 grave 律保证不再分配"(纯自律)。实测定谳(cuBLAS splitK 案)表明
        // 窗内 `cudaMallocAsync` **会被捕获**成 MEM_ALLOC/MEM_FREE 节点(回放期
        // 重分配 + 派发税),且 scratch 是裸指针 ⇒ **完全在 slab/账房之外**
        // (M1 峰值定尺看不见)。观察档计数 + 告警;`OWL_CAP_STRICT=1` 结构化拒。
        if self.ctx.capture_stream() {
            self.ctx.sink.count("srv.count.fam_alloc_in_window", 1);
            match crate::state::fam_window_verdict(
                crate::state::FamAct::ScratchAlloc,
                crate::state::cap_strict(),
            ) {
                Ok(()) => self.ctx.sink.logf(
                    owl_shared::log::LVL_WARN,
                    "fam-alloc-in-window",
                    format_args!("家族 scratch 分配发生在捕获窗(tag={tag}, {bytes}B)—— 会被烙成 MEM_ALLOC 节点且不进 slab/账房"),
                ),
                Err(detail) => {
                    return Err(OpError::Contract {
                        op: OP_RES.into(),
                        field: "alloc",
                        expect: "捕获窗内不得现建 scratch(装配期分配 / 不变盒 / 引擎真块三选一)".into(),
                        got: format!("{detail}(tag={tag}, {bytes}B)"),
                    })
                }
            }
        }
        let stream = self.stream()?;
        let d = stream.alloc_zeros::<u8>(bytes)
            .map_err(|e| OpError::Launch { op: OP_RES.into(), stage: owl_kernels::contract::Stage::Load, detail: format!("scratch alloc: {e:?}") })?;
        use cudarc::driver::DevicePtr;
        let ptr = d.device_ptr(&stream).0;
        // 入账:scratch 不在块账房(裸指针,不可 resolve)⇒ 账上只有这一条线索
        self.note_device("fam_alloc", tag, bytes, &[ptr]);
        self.scratch.push((ptr, d));
        Ok(ScratchBuf { ptr, bytes })
    }
    fn capturing(&self) -> Result<bool, OpError> {
        Ok(self.ctx.capture_stream())
    }
    /// **家族面设备动作入账**(2026-10-15 A13 补漏):家族 runtime 的设备动作
    /// 不产生 `Command` ⇒ 发射账(§十三)看不见它们 —— A13 藏一整轮的直接原因。
    /// 本覆写把家族面写进**同一个环**(命令面 `cmd` 与家族面 `kind` 并列可读),
    /// 且 `stream = "family"` 一眼可辨;地址一并落账(生命周期判据的钥匙)。
    ///
    /// 纪律:关账(容量 0)仅一次 `OnceLock` 判定;`ptrs` 为空时**不分配**内存。
    /// 入账面武装判定(关账 ⇒ 调用方跳过建载荷;`Exec::launch` 每发射一次)
    fn note_enabled(&self) -> bool {
        crate::trace::get().is_some_and(|r| r.enabled())
    }

    fn note_device(&mut self, kind: &'static str, name: &str, bytes: usize, ptrs: &[u64]) {
        let Some(tr) = crate::trace::get() else { return };
        if !tr.enabled() {
            return;
        }
        let mut e = crate::trace::TraceEntry::new(kind, "family").bytes(bytes);
        if !name.is_empty() {
            e = e.name(name);
        }
        if !ptrs.is_empty() {
            e = e.ptrs(ptrs.iter().copied());
        }
        tr.push(e);
    }
    fn record_capture(&mut self, note: owl_kernels::device::LaunchNote) -> Result<(), OpError> {
        // **A1.6 落地(2026-10-15)**:家族/外来核不发 `Command` ⇒ G2 白名单
        // 与原生发射计数都看不见它们。这里把每次家族发射喂给窗内登记表
        // (① 空窗哨兵不再误判"纯家族窗为空" ② "图里烘了什么"首次有账;
        // 见 state.rs `CaptureNodes`)。窗口外调用 = 无动作。
        if self.ctx.capture_stream() {
            self.ctx.note_launch(note.kernel);
        }
        Ok(())
    }
    fn upload(&mut self, dst: u64, src: &[u8]) -> Result<(), OpError> {
        self.upload_impl(dst, src, false)
    }
    fn upload_invariant(
        &mut self,
        dst: u64,
        src: &owl_kernels::contract::InvariantBox,
    ) -> Result<(), OpError> {
        self.upload_impl(dst, src.as_bytes(), true)
    }
    fn alloc_block(
        &mut self,
        bytes: usize,
        tag: &'static str,
    ) -> Result<owl_kernels::contract::Bytes, OpError> {
        // 装配期自检用:真账本块(可被 resolve)。**不计 slab/vram 音账** ——
        // 自检块是装配期一次性小额,且池预算在自检之前已按固定面定尺;
        // 计入 live 会让池预算随自检项数漂移(账目语义混乱 > 少记几十 KB)。
        let stream = self.stream()?;
        let slice = unsafe { stream.alloc::<u8>(bytes) }.map_err(|e| OpError::Launch {
            op: "selfcheck".into(),
            stage: owl_kernels::contract::Stage::Load,
            detail: format!("alloc_block[{tag}]({bytes}B): {e:?}"),
        })?;
        let id = self.ctx.new_block(slice);
        Ok(owl_kernels::contract::Bytes::new(id, bytes, Dtype::U8, 0))
    }

    fn upload_block(
        &mut self,
        b: &owl_kernels::contract::Bytes,
        src: &[u8],
    ) -> Result<(), OpError> {
        // 账本块上传(自检面):resolve 拿到块指针 → htod。
        // 与 `upload`(scratch-only 白名单)分开:这里写的是**账房在册块**。
        //
        // ⚠️ **A13 同族(2026-10-15)**:原为 `memcpy_htod_sync`(NULL 流),
        // 而目标块来自 `alloc_block` 的 `stream.alloc`(流序异步分配)——
        // compute-sanitizer 就在本函数上抓到 `Use-before-alloc`(写进尚未
        // 落定的内存 ⇒ 静默丢写)。改为**与分配同流的 async 拷贝**;
        // 调用方 `download` 已自带 COMPUTE 排空,回读序两侧都齐。
        let p = self.resolve(b.id)?;
        let stream = self.stream()?;
        // **边界校验(2026-10-15 审查补)**:账本块上传同样要验"写得下"
        self.ctx
            .check_range(b.id, 0, src.len(), "upload-block")
            .map_err(|e| OpError::Contract {
                op: OP_RES.into(),
                field: "upload_block",
                expect: "src.len ≤ block_len(账本)".into(),
                got: format!("{}B :: {e:?}", src.len()),
            })?;
        self.note_device("fam_upload_block", "", src.len(), &[p]);
        if self.ctx.diag.sync_at(owl_shared::config::sync_bit::SCRATCH_UP_LEGACY) {
            unsafe { crate::ffi::memcpy_htod_sync(p, src) }.map_err(|e| OpError::Launch {
                op: OP_RES.into(),
                stage: owl_kernels::contract::Stage::Store,
                detail: format!("upload_block(id={}, {}B, legacy sync): {e:?}", b.id, src.len()),
            })
        } else {
            unsafe { crate::ffi::memcpy_htod_async(p, src, stream.cu_stream()) }.map_err(|e| {
                OpError::Launch {
                    op: OP_RES.into(),
                    stage: owl_kernels::contract::Stage::Store,
                    detail: format!("upload_block(id={}, {}B, async): {e:?}", b.id, src.len()),
                }
            })
        }
    }

    fn download(&mut self, src: u64, dst: &mut [u8]) -> Result<(), OpError> {
        // **EagerOnly 守卫(2026-10-15 审查补)**:捕获窗内 `cudaStreamSynchronize`
        // **非法**(`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`)—— 本函数为回读补的
        // "排空 COMPUTE"在捕获期会把图捕获打坏。与 `Exec::sample_stats` 的既有
        // 守卫同款;捕获期可达的调用方(GDN diag 块)另加 `!capturing`。
        if self.ctx.capture_stream() {
            return Err(OpError::Contract {
                op: OP_RES.into(),
                field: "download",
                expect: "EagerOnly(捕获窗内禁 sync/D2H)".into(),
                got: "capturing=true".into(),
            });
        }
        // 装配期自检回读(同步语义:回执即落地;非热路径)
        //
        // ⚠️ **A13 同族(2026-10-15)**:`memcpy_dtoh_sync` 走 **NULL 流**,
        // 与 COMPUTE 流无序 ⇒ 直接回读会拿到半写入/旧值/全零(本仓
        // 2026-10-09 “meta 表读回全零”与 2026-10-13 同病史两次踩中,
        // `gdn_chunked.rs` 453 行曾就地补 sync 绕过)。此处补齐序:**先排空
        // COMPUTE**(生产方全在该流上),再做同步回读。
        self.stream()?.synchronize().map_err(|e| OpError::Launch {
            op: OP_RES.into(),
            stage: owl_kernels::contract::Stage::Load,
            detail: format!("download pre-sync: {e:?}"),
        })?;
        self.note_device("fam_download", "", dst.len(), &[src]);
        // SAFETY:dst 是 host 可写切片,src 是本账本块/scratch 的设备虚地址,
        // 长度由 dst 决定(调用方负责 src 侧可读范围 ≥ dst.len();自检面
        // 按自己的分配尺寸回读)。
        unsafe { crate::ffi::memcpy_dtoh_sync(dst, src) }.map_err(|e| {
            OpError::Launch {
                op: OP_RES.into(),
                stage: owl_kernels::contract::Stage::Load,
                detail: format!("dtoh(ptr={src:#x}, {}B): {e:?}", dst.len()),
            }
        })
    }

    fn device_ordinal(&self) -> Result<i32, OpError> {
        Ok(self.ctx.device_ordinal() as i32)
    }
}
