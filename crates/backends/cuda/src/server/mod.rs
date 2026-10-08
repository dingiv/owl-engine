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

use crate::ffi::{launch_host_function, memcpy_dtoh_async, memcpy_htod_async, memset_d8_async, PushKernelArg};
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

/// GDN chunked 句柄(FLA AOT cubin 五核 + f32 scratch + cast 模块)
struct GdnChunkedState {
    cumsum: std::sync::Arc<cudarc::driver::CudaFunction>,
    kkt: std::sync::Arc<cudarc::driver::CudaFunction>,
    /// solve_tril(BT=64 单核形态 = merge_16x16_to_64x64_inverse)
    merge: std::sync::Arc<cudarc::driver::CudaFunction>,
    wu: std::sync::Arc<cudarc::driver::CudaFunction>,
    h: std::sync::Arc<cudarc::driver::CudaFunction>,
    o: std::sync::Arc<cudarc::driver::CudaFunction>,
    /// 状态转置核(CAST nvrtc 同模块):owl 池 [HV,K,V] ↔ fork [HV,V,K]
    trans_kv_vk: std::sync::Arc<cudarc::driver::CudaFunction>,
    trans_vk_kv: std::sync::Arc<cudarc::driver::CudaFunction>,
    /// dtype 统一律(2026-10-11Ⅵ):bf16/f32 铸全在 handler 内,层侧纯 f16
    cast_f16_bf16: std::sync::Arc<cudarc::driver::CudaFunction>,
    cast_bf16_f16: std::sync::Arc<cudarc::driver::CudaFunction>,
    cast_f16_f32: std::sync::Arc<cudarc::driver::CudaFunction>,
    /// 输入 bf16 镜像(q/k/v/beta)+ g f32 镜像 + 输出 bf16 scratch
    in_q: cudarc::driver::CudaSlice<u16>,
    in_k: cudarc::driver::CudaSlice<u16>,
    in_v: cudarc::driver::CudaSlice<u16>,
    in_beta: cudarc::driver::CudaSlice<u16>,
    in_g: cudarc::driver::CudaSlice<f32>,
    o_b16: cudarc::driver::CudaSlice<u16>,
    // 2026-10-11 fork-bf16 全家桶中间量(g 恒 f32;q/k/v/beta bf16 层侧铸):
    /// g_cum [T*hv] f32(cumsum 出)
    g_cum: cudarc::driver::CudaSlice<f32>,
    /// A(kkt 出)/ Ai(merge 出)[T*hv*64] f32 ×2
    a_buf: cudarc::driver::CudaSlice<f32>,
    ai_buf: cudarc::driver::CudaSlice<f32>,
    /// w [T*hv*kd] / u [T*hv*vd] bf16(以 u16 承载)
    w: cudarc::driver::CudaSlice<u16>,
    u: cudarc::driver::CudaSlice<u16>,
    /// h_buf [nt*hv*vd*kd] bf16(fork 布局 V 行 K 列)
    h_buf: cudarc::driver::CudaSlice<u16>,
    /// v_new [T*hv*vd] bf16(h 核出,o 核入)
    v_new: cudarc::driver::CudaSlice<u16>,
    /// 转置后状态 [hv*kd*vd] f32(固定尺寸,init 分配);h0 入态
    state_t: cudarc::driver::CudaSlice<f32>,
    /// ht 出态(与 h0 分离 —— fork 探针姿势为两张量;别名行为未证)
    state_out_t: cudarc::driver::CudaSlice<f32>,
    /// varlen 元数据驻留(2026-10-11:vLLM 同款姿势 —— cu/idx/coff 按
    /// (T,NT) 键一次构建永驻,削掉每 call 3 个 pageable htod)
    meta_cache: std::collections::HashMap<(usize, usize), usize>,
    /// 驻留表仓库(键序:cu[2] + coff[2] + idx[NT*2];按 meta_cache 索引)
    meta_bufs: Vec<cudarc::driver::CudaSlice<i64>>,
    t_cap: usize,
    hv_dim: usize,
    kd_dim: usize,
    vd_dim: usize,
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

mod foreign;
mod harvest;

pub struct GpuServer {
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
    /// cuBLAS 封装(foreign-kernel 通道;算子之家 owl-kernels::cublas,懒初始化)
    blas: Option<owl_kernels::cublas::OwlCublas>,
    /// 刀1.5:server 私有 cublas 工作区(4MB;SetWorkspace 预绑,捕获期
    /// gemv splitK 不再走池分配烙 MEM_ALLOC/FREE 节点)。**每 server 独享**
    /// —— 全局单例在并行 server 下两句柄并发写同一缓冲 = ILLEGAL_ADDRESS
    /// (2026-10-03 实测);生命周期随 server(同 ctx,指针恒有效)。
    blas_ws: Option<std::sync::Arc<cudarc::driver::CudaSlice<u8>>>,
    /// FlashInfer prefill 句柄(foreign-kernel 通道;workspace + plan 缓存,
    /// 懒初始化 —— owl-kernels::flashinfer)
    fi: Option<FiState>,
    /// GDN chunked 句柄(FLA AOT cubin 五核;懒初始化)
    gdn_chunked: Option<GdnChunkedState>,
    /// GDN scalar 句柄(lmdeploy pre_sm90 port 单核;懒初始化)
    gdn_scalar: Option<GdnScalarState>,
    /// 完成派发出口(host 回调只投递;派发线程执行真正的 finish)
    dispatch: Option<mpsc::Sender<Finish>>,
    /// 逐命令计时账本(OWL_SRV_TIMING;name / count / total_ns / max_ns)
    timings: Vec<(&'static str, u64, u128, u128)>,
    /// 探针开关(构造期解析;热路径零 env 查询)
    probes: ServerProbes,
}

fn gdn_dptr<T>(s: &mut cudarc::driver::CudaSlice<T>, stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> u64 {
    use cudarc::driver::DevicePtr;
    let (p, _g) = s.device_ptr(stream);
    p
}

/// GDN chunked scratch 扩容(逐 slice alloc;Result 化供 ? 链)
#[allow(clippy::too_many_arguments)]
fn gdn_chunked_realloc(
    st: &mut GdnChunkedState,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    t: usize,
    nt: usize,
    hv: usize,
    nk: usize,
    kd: usize,
    vd: usize,
) -> Result<(), ModelError> {
    // fork-bf16 配方(2026-10-11):g 链 f32;q/k/v/beta/w/u/h/v_new bf16(u16)
    st.g_cum = stream.alloc_zeros::<f32>(t * hv).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.a_buf = stream.alloc_zeros::<f32>(t * hv * 64).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.ai_buf = stream.alloc_zeros::<f32>(t * hv * 64).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.w = stream.alloc_zeros::<u16>(t * hv * kd).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.u = stream.alloc_zeros::<u16>(t * hv * vd).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.h_buf = stream.alloc_zeros::<u16>(nt * hv * vd * kd).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.v_new = stream.alloc_zeros::<u16>(t * hv * vd).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    // dtype 统一律镜像(handler 内铸的 bf16/f32 副本)
    st.in_q = stream.alloc_zeros::<u16>(t * nk * kd).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.in_k = stream.alloc_zeros::<u16>(t * nk * kd).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.in_v = stream.alloc_zeros::<u16>(t * hv * vd).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.in_beta = stream.alloc_zeros::<u16>(t * hv).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.in_g = stream.alloc_zeros::<f32>(t * hv).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.o_b16 = stream.alloc_zeros::<u16>(t * hv * vd).map_err(|e| ModelError::Msg(format!("{e:?}")))?;
    st.t_cap = t;
    Ok(())
}

/// GDN scalar 臂状态(lmdeploy pre_sm90 port 单核):内核 + f16 cast + 私有
/// scratch(o_f32 / seq_off)。无中间张量 —— q/k/v/g/beta 直用层侧 cast 产物块。
struct GdnScalarState {
    k: std::sync::Arc<cudarc::driver::CudaFunction>,
    cast_f32_f16: std::sync::Arc<cudarc::driver::CudaFunction>,
    o_f32: cudarc::driver::CudaSlice<f32>,
    /// 捕获图安全(2026-10-11 立案即结案):旧代 o_f32 **坟场保活** ——
    /// 图持有捕获时指针,扩容换新块后旧块不可释放(A1.7 租约语义;史前
    /// 版本直接 drop → verify 图回放悬空指针 = ILLEGAL_ADDRESS,仅当
    /// scalar × 捕获图同跑时引爆)。容量单调涨,坟场总量有界(≤ 最大代)。
    o_f32_grave: Vec<cudarc::driver::CudaSlice<f32>>,
    /// soff htod staging 按 T 键保活:**栈临时发起的 memcpy 在捕获窗内
    /// = 回放读死栈**(同案第二违例);每 T 一份永不互改的 [0,T] 盒,
    /// 捕获节点与 eager 调用各自引用自己的不变源。
    soff_staging: std::collections::HashMap<usize, Box<[i32; 2]>>,
    soff: cudarc::driver::CudaSlice<i32>,
    t_cap: usize,
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
        Self {
            rx,
            selector,
            pool: std::sync::Arc::new(PinnedPool::default()),
            boot,
            ctx: None,
            kernels: KernelCache::new(diag.nvrtc_include.clone()),
// (diag 自身随 Self 移动,克隆 include 后置)
            blas: None,
            blas_ws: None,
            fi: None,
            gdn_chunked: None,
            gdn_scalar: None,
            dispatch: None,
            timings: Vec::new(),
            probes: ServerProbes::from_parts(&diag, debug),
            diag,
        }
    }

    /// 事件循环主入口(server 线程生命周期 = 循环生命周期)。
    /// 首行上线设备(本线程 bind_to_thread 一次到位)+ 起派发线程。
    pub fn run(mut self) -> Result<(), String> {
        let ctx = GpuCtx::new(&self.selector, self.diag.clone());
        match ctx {
            Ok(c) => {
                if let Some(boot) = self.boot.take() {
                    let _ = boot.send(Ok(()));
                }
                self.ctx = Some(c);
                // C1(2026-10-11):cuBLAS 句柄/工作区构造期急切预钉
                // (lm_head 量化后 warmup 期可能无 cublas 首发路径,惰性
                // 初始化的捕获窗前置条件不能靠碰运氻 warmup)
                if let Err(e) = self.ensure_blas() {
                    eprintln!("[blas-ws] 构造期预钉失败 {e:?}(回退惰性首初始化)");
                }
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
        if self.probes.srv_timing && !self.timings.is_empty() {
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
        let timing = self.probes.srv_timing;
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
            Command::CopyBatch { .. } => "CopyBatch",
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
            Command::CopyBatch { copies, ack } => self.handle_copy_batch(copies, ack),
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
            Command::CopyBatch { ack, .. } => closed!(ack),
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
            Command::CopyBatch { .. } => "CopyBatch",
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
            Command::CopyBatch { ack, .. } => reject!(ack),
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
                    if self.probes.cap_prof {
                        eprintln!("[cap-prof] carve-memset {}B", n_bytes);
                    }
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
                // 诊断开关(OWL_LAUNCH_SYNC=1):alloc/memset 后同步归因
                if self.probes.launch_sync {
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
        if self.ctx().capture_stream() && self.probes.cap_prof {
            eprintln!("[cap-prof] memset_zero {}B block{block}", len_bytes);
        }
        let result = (|| {
            let (dptr, _) = self.ctx().block_ptr(block, &stream)?;
            unsafe {
                crate::ffi::memset_d8_async(dptr + offset_bytes as u64, 0, len_bytes, stream.cu_stream())
            }
            .map_err(|e| ModelError::Msg(format!("memset_zero: {e:?}")))
        })();
        ack.send(result);
    }
    fn handle_launch(&mut self, msg: LaunchMsg, ack: Ack<Result<Bytes, ModelError>>) {
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
                n if n == owl_kernels::marlin::GEMM_W4A16
                    || n == owl_kernels::marlin::GEMM_W4A16_AWQ =>
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
            let r = self.handle_launch_inner(msg, ack);
            if let Ok(stream) = self.ctx().stream(STREAM_COMPUTE) {
                let _ = stream.synchronize(); // 归因同步(错误由 sticky 面另报)
            }
            let dt = t0.elapsed();
            owl_shared::metrics::with_metrics_store(|s| {
                s.timer_record_tag(&shape_tag, dt, file!(), 0);
                s.timer_record_tag(&format!("gpu.{name}"), dt, file!(), 0);
            });
            return r;
        }
        // 逐发射计时(OWL_LAUNCH_TIME=1;捕获期 warmup = 全模型 eager 一遍,
        // 一次跑完全谱;capture 回放期自动关闭 —— 回放是整图一发射)
        if self.probes.launch_time && !self.ctx().capture_stream() {
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
        if capturing && self.probes.cap_prof {
            eprintln!("[cap-prof] native launch {} out={}", msg.kernel.name, msg.out_elems);
        }
        match issue_launch(ctx, &stream, kernels, &msg, self.probes.debug) {
            Ok(out_id) => {
                self.ctx_mut().note_launch();
                // 诊断开关(OWL_LAUNCH_SYNC=1):逐发射同步,sticky 错误
                // 归属到具体核(2026-09-27 长 ctx ILLEGAL_ADDRESS 排查;
                // 捕获期禁 sync —— 跳过,捕获正确性由哨兵③/④另保)
                if self.probes.launch_sync && !capturing {

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

    /// 中间块回收(E2a):账房移除 Owned 块归池。调用方契约 = 已收割
    /// 所需数据(dtoh 回执即 COMPUTE 排空);图捕获期由 G2 拒绝臂拦截。
    fn handle_free(&mut self, ids: Vec<u64>, ack: Ack<Result<(), ModelError>>) {
        let freed = self.ctx_mut().free_blocks(&ids);
        if self.probes.debug {
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