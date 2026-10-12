//! server 状态面:设备上下文 + 流注册表(三固定流)+ 池块账房
//! + kernel 编译缓存 + pinned 码头 + 图捕获状态机/图注册表。
//!
//! 只被 actor 线程触碰(单线程所有权;零锁)。

use crate::ffi::{device_get_count, free_host, malloc_host, CudaContext, CudaFunction, CudaSlice, CudaStream, CAPTURE_MODE_THREAD_LOCAL};
use owl_iface::contract::GraphId;
use owl_iface::contract::ModelError;
use owl_iface::contract::{numel, Shape};
use std::collections::HashMap;
use std::sync::Arc;

/// 设备上下文诊断/调优选项(**显式依赖律,2026-10-10**:env 派生数据由
/// 入口/测试经本结构显式传入,设备路径零 env 读取)。
/// [`DiagOpts::from_config`] 是入口侧组装器(server config / 测试用例)。
#[derive(Clone, Debug, Default)]
pub struct DiagOpts {
    /// 逐命令/图节点计时(原 OWL_SRV_TIMING)
    pub srv_timing: bool,
    /// 捕获期插桩(原 OWL_CAP_PROF)
    pub cap_prof: bool,
    /// 图回放后同步归因(原 OWL_LAUNCH_SYNC;诊断用,性能杀手勿常开)
    pub launch_sync: bool,
    /// 旁路事件钉序(原 OWL_FREE_LEGACY;崩坏案 A/B 判别)
    pub free_legacy: bool,
    /// server 提交计时(server probes 面;DiagOpts 一体携带免双参)
    pub launch_time: bool,
    /// dtoh 分相(server probes 面)
    pub d2h_prof: bool,
    /// 逐核归因(server probes 面)
    pub gpu_prof: bool,
    /// 图实例化旗标(原 OWL_GRAPH_FLAGS;UPLOAD=2 / DEVICE_LAUNCH=4;
    /// 类型化位面经 OwlConfig.cuda.graph_flags → to_bits())
    pub graph_flags: u64,
    /// 捕获 slab 固定档 MiB(None = warmup 计量定量;原 OWL_CAPTURE_SLAB_MB
    /// —— 24G 贴顶的 27B + spec 双图场景,verify 图按需调小)
    pub capture_slab_mb: Option<usize>,
    /// **生产算子白名单**(dispatch.allowed_ops;None/空 = 门关)。
    /// `Arc` = 与 server 主循环共享,逐发射判定零克隆。
    pub allowed_ops: Option<std::sync::Arc<Vec<String>>>,
}

impl DiagOpts {
    /// **入口侧组装器**:自全量配置对象显式组装(设备路径零 env 读取)
    pub fn from_config(cfg: &owl_shared::config::OwlConfig) -> Self {
        let c = &cfg.cuda;
        Self {
            srv_timing: c.srv_timing,
            cap_prof: c.cap_prof,
            launch_sync: c.launch_sync,
            free_legacy: c.free_legacy,
            launch_time: c.launch_time,
            d2h_prof: c.d2h_prof,
            gpu_prof: c.gpu_prof,
            graph_flags: c.graph_flags.to_bits(),
            capture_slab_mb: c.capture_slab_mb,
            allowed_ops: cfg
                .dispatch
                .allowed_ops
                .as_ref()
                .filter(|v| !v.is_empty())
                .map(|v| std::sync::Arc::new(v.clone())),
        }
    }
}

// 流身份证(server 内部路由键;客户端不可见 —— 三固定流是 server 策略)
pub(super) type StreamId = u64;
pub(super) const STREAM_H2D: StreamId = 0;
pub(super) const STREAM_COMPUTE: StreamId = 1;
pub(super) const STREAM_D2H: StreamId = 2;

// ============================================================================
// 设备选择(UUID 钉卡;数字序事故免疫)
// ============================================================================

/// 设备选择器:优先 UUID(唯一稳定身份证);Ordinal 仅测试/单机便捷用。
#[derive(Debug, Clone)]
pub enum DeviceSelector {
    /// 16 字节设备 UUID(推荐;跨重启/跨机器枚举序稳定)
    Uuid([u8; 16]),
    /// CUDA ordinal(枚举序敏感;仅单卡/测试场景)
    Ordinal(usize),
}

impl DeviceSelector {
    /// 解析为 ordinal(UUID:遍历设备表比对;Ordinal:原样)
    pub(super) fn resolve(&self) -> Result<usize, String> {
        match self {
            Self::Ordinal(o) => Ok(*o),
            Self::Uuid(uuid) => {
                let count = device_get_count().map_err(|e| format!("device_get_count: {e:?}"))?;
                for o in 0..count as usize {
                    if crate::ffi::device_uuid(o).map_err(|e| format!("device_uuid({o}): {e:?}"))? == *uuid {
                        return Ok(o);
                    }
                }
                Err(format!(
                    "UUID {uuid:?} 不在设备表(count={count};检查 CUDA_VISIBLE_DEVICES)"
                ))
            }
        }
    }
}

// ============================================================================
// 池块账房
// ============================================================================

/// 块的两种形态:
/// - Owned:图外独立分配(cudaMallocAsync)
/// - Carved:捕获期从 capture slab 切出的切片(零分配;slab 随块保活)
pub(super) enum Block {
    Owned(CudaSlice<u8>),
    Carved { slab: Arc<CudaSlice<u8>>, off: usize, n: usize },
}

/// 账本内块 id 位编码(裁决 2,2026-10-13):单 u64 高位 ledger、低 40 位
/// 本账递增 local。线格式 `Bytes{id,len}` 不变 —— 只是 id 不再跨账本撞号
/// (pitfall §27 根因)。报错可同时打印两段。
const BLOCK_LEDGER_SHIFT: u32 = 40;
const BLOCK_LOCAL_MASK: u64 = (1u64 << BLOCK_LEDGER_SHIFT) - 1;

/// 打包块 id(分配点)。
fn pack_block_id(ledger: u32, local: u64) -> u64 {
    ((ledger as u64) << BLOCK_LEDGER_SHIFT) | (local & BLOCK_LOCAL_MASK)
}

/// 访问器:块 id → 所属账本(0 成本解码;诊断/断言用)。
pub(super) fn ledger_of(id: u64) -> u32 {
    (id >> BLOCK_LEDGER_SHIFT) as u32
}

/// 访问器:块 id → 账本内局部序号(测试/诊断)。
#[cfg(test)]
fn local_of(id: u64) -> u64 {
    id & BLOCK_LOCAL_MASK
}

/// 解析点断言:块 id 必须属本账本(跨账本 = §27 撞号案;六步压成一行)。
fn assert_ledger(id: u64, ledger: u32) -> Result<(), ModelError> {
    let owner = ledger_of(id);
    if owner != ledger {
        return Err(ModelError::CrossLedger { id, owner, reader: ledger });
    }
    Ok(())
}

/// 块账条目:块本体 + 账长 + 账本身份 + 布局/类型(§6.2 VramBlock 最小面)。
///
/// `dtype`/`shape` 当前从字节口径填充 —— cuda 池账是字节级分配
/// (`new_block` 只见 `slice.len()`),元素级 dtype/shape 待分配 API 携带
/// 元数据后细化;本轮先落 `ledger` 身份字段(撞号门)。
///
/// `BlockDtype` 是**本账局部** dtype token(不动线格式 `Dtype`;字节口径
/// 无元素宽语义,向共享 `Dtype` 加 U8 会连带 driver::DType / models 映射
/// 全链改,超出 P1 最小面)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BlockDtype {
    /// 字节口径(池块/元数据 blob)
    Bytes,
}

impl BlockDtype {
    fn size_bytes(self) -> usize {
        match self {
            BlockDtype::Bytes => 1,
        }
    }
}

pub(super) struct BlockEntry {
    pub(super) block: Block,
    /// 账长(字节口径;f16 基线)
    pub(super) bytes: usize,
    /// 分配它的账本(actor 序号;解析点断言 == 本账本)
    pub(super) ledger: u32,
    pub(super) dtype: BlockDtype,
    pub(super) shape: Shape,
}

// ============================================================================
// 设备上下文 + 流注册表
// ============================================================================

pub(super) struct GpuCtx {
    pub ctx: Arc<CudaContext>,
    /// 诊断/调优选项(入口显式传入;显式依赖律)
    pub(super) diag: DiagOpts,
    /// 流注册表:三固定流(H2D/COMPUTE/D2H;客户端不可自创)
    streams: HashMap<StreamId, Arc<CudaStream>>,
    /// 图捕获状态机(见 graph_begin/end 的护栏注释)
    ordinal: usize,
    capture: Option<CaptureState>,
    /// 图注册表:id → 实例化图(重放用)
    graphs: HashMap<GraphId, GraphHolder>,
    next_graph: GraphId,
    /// 块账房:id → 块条目(账长/账本身份/dtype/shape)
    blocks: HashMap<u64, BlockEntry>,
    next_block: u64,
    /// 本账本编号(= GpuServer actor 序号;块 id 高位编码此值)
    ledger: u32,
    /// 动态 smem 每块上限(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK
    /// _OPTIN;发射前硬顶守卫,超限结构化拒绝 —— 2026-09-27 v1 smem 事故)
    smem_optin: usize,
    /// 刀1.6:探针开关(boot 解析;热路径零 env 查询)
    pub(super) srv_timing: bool,
    pub(super) cap_prof: bool,
}

/// 捕获会话:目标流 + 切块 slab + 发射计数(空窗哨兵用)
struct CaptureState {
    stream: Arc<CudaStream>,
    slab: Arc<CudaSlice<u8>>,
    used: usize,
    launches: usize,
    /// 本图实际 slab 容量(warmup 计量定量或 env 固定档;carve 校验
    /// 与 graph_begin 分配必须同口 —— 旧实现 carve 校验读 env,与
    /// hint 定量的 slab 脱口,2026-10-10)
    cap: usize,
}

/// CudaGraph 非 Send(cudarc 未标注);我们把它钉死在 actor 线程单一所有权
/// 下使用(所有图操作都在 dispatch 线程序列化),跨线程移动 GpuServer 携带
/// 它是安全的 —— 与 cudarc "须外部串行化" 的要求一致。
/// 图持有(手搓实例化面;exec + 捕获流成对,Drop 双销毁。
/// 为何不走 cudarc safe CudaGraph:见 ffi.rs graph 定谳注释 —— 其
/// end_capture 强绑 AUTO_FREE_ON_LAUNCH,每 launch +4.45ms)
struct GraphHolder {
    exec: crate::ffi::sys::CUgraphExec,
    /// 捕获期 CUgraph(exec 实例化后独立,但随 holder 销毁,保持
    /// cudarc CudaGraph 同等资源语义)
    cu_graph: crate::ffi::sys::CUgraph,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}
unsafe impl Send for GraphHolder {}

impl Drop for GraphHolder {
    fn drop(&mut self) {
        unsafe {
            let _ = crate::ffi::graph_exec_destroy(self.exec);
            let _ = crate::ffi::graph_destroy(self.cu_graph);
        }
    }
}

impl GpuCtx {
    pub(super) fn new(selector: &DeviceSelector, diag: DiagOpts, ledger: u32) -> Result<Self, String> {
        let ordinal = selector.resolve()?;
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("{e:?}"))?;
        ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
        // B6.5:设备总显存落账(vram 治理面;一次)
        match cudarc::driver::result::device::get(ordinal as i32)
            .and_then(|dev| unsafe { cudarc::driver::result::device::total_mem(dev) })
        {
            Ok(bytes) => owl_shared::vram::total_set(bytes as u64),
            Err(e) => eprintln!("[boot] vram total 查询失败(治理禁用): {e:?}"),
        }
        if let Ok((free, _)) = cudarc::driver::result::mem_get_info() {
            owl_shared::vram::free_set(free as u64);
        }
        // G0 护栏:关 event-tracking(坑 A)。多流 + event tracking 会让
        // safe 层在块读写上插事件,污染图捕获(CAPTURE_ISOLATION/图内事件节点)
        unsafe { ctx.disable_event_tracking() };
        // 动态 smem 每块上限(boot 查一次缓存;发射前硬顶守卫用,
        // 不逐发射查 CUDA —— 热路径零额外调用)
        let smem_optin = ctx
            .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
            .map_err(|e| format!("查 MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: {e:?}"))?
            as usize;
        // ---- 装配期架构门(设计 §15.4)----
        // 设备 sm ∉ 构造期编出的代际集 ⇒ **现在**拒启,而不是等到某次发射
        // 报 "no kernel image is available for execution on the device"。
        // 资产集为空(`OWL_CUBIN_SKIP` 构建)= 门关(仅用于门的负例验证)。
        {
            let archs = owl_kernels::family::cubins::CUBIN_ARCHS;
            if !archs.is_empty() {
                let major = ctx
                    .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
                    .map_err(|e| format!("查 COMPUTE_CAPABILITY_MAJOR: {e:?}"))?;
                let minor = ctx
                    .attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
                    .map_err(|e| format!("查 COMPUTE_CAPABILITY_MINOR: {e:?}"))?;
                let dev = format!("sm_{major}{minor}");
                if !archs.contains(&dev.as_str()) {
                    return Err(format!(
                        "装配期架构门:设备 {dev}(ordinal {ordinal} )不在构建期 cubin 代际集 {archs:?} 内 ——\
                         重建时请加 `OWL_CUBIN_ARCHS=...,{dev}`(设计 §15.4)"
                    ));
                }
            }
            // ---- 零运行期编译覆盖率(boot 报告 + require 硬门)----
            let (covered, total, missing) = owl_kernels::family::cubins::coverage();
            eprintln!(
                "[aot] 构建期 cubin 代际 {archs:?} / 资产 {} 个(fp {}) —— 登记表覆盖 {covered}/{total}",
                owl_kernels::family::cubins::CUBIN_ASSETS.len(),
                owl_kernels::family::cubins::ASSET_FP
            );
            if !missing.is_empty() {
                let show: Vec<&str> = missing.iter().take(8).copied().collect();
                let total_show = missing.len();
                return Err(format!(
                    "零运行期编译硬门:{total_show} 个登记表核无构建期 cubin 资产 —— \
                     这些入口的 .cu 不在 build.rs 的 include_str! 闭包内(见 [aot] 行);\
                     前几个: {show:?}"
                ));
            }
        }
        let mut streams = HashMap::new();
        for (id, _role) in [(STREAM_H2D, "h2d"), (STREAM_COMPUTE, "compute"), (STREAM_D2H, "d2h")] {
            let s = ctx.new_stream().map_err(|e| format!("建流({_role}): {e:?}"))?;
            streams.insert(id, s);
        }
        Ok(Self {
            ctx,
            diag: diag.clone(),
            streams,
            ordinal,
            srv_timing: diag.srv_timing,
            cap_prof: diag.cap_prof,
            capture: None,
            graphs: HashMap::new(),
            next_graph: 1,
            blocks: HashMap::new(),
            next_block: 1,
            ledger,
            smem_optin,
        })
    }

    /// 动态 smem 每块上限(boot 缓存;issue_launch 硬顶守卫用)
    pub(super) fn smem_optin(&self) -> usize {
        self.smem_optin
    }

    /// 流反查(id → 流)
    pub(super) fn stream(&self, id: StreamId) -> Result<&Arc<CudaStream>, ModelError> {
        self.streams
            .get(&id)
            .ok_or_else(|| ModelError::Msg(format!("流 {id} 不存在(固定三流:H2D/COMPUTE/D2H)")))
    }

    // ======================================================================
    // 图捕获状态机(护栏所在)
    // ======================================================================

    /// 当前是否捕获中
        /// 设备 ordinal(foreign-kernel 需要:marlin SM 数查询的 dev 参数)
    pub(super) fn device_ordinal(&self) -> usize {
        self.ordinal
    }

    pub(super) fn capture_stream(&self) -> bool {
        self.capture.is_some()
    }

    /// 图捕获开始(G1 不可嵌套;slab 在捕获**前**分配 —— 捕获期零 cudaMalloc)。
    /// 捕获流恒为 COMPUTE(路由策略:Launch/Alloc 全在 COMPUTE)。
    pub(super) fn graph_begin(&mut self) -> Result<(), ModelError> {
        if self.capture.is_some() {
            return Err(ModelError::Msg(
                "graph_begin: 已有捕获进行中(不可嵌套/并发;先 graph_end)".to_string(),
            ));
        }
        let stream = self.stream(STREAM_COMPUTE)?.clone();
        // 捕获前必须排空该流(在飞任务会污染捕获窗)
        stream
            .synchronize()
            .map_err(|e| ModelError::Msg(format!("graph_begin: 预排空失败 {e:?}")))?;
        // slab 容量三档:diag 固定档(兼容);缺省 = warmup 计量
        // hint(见 owl_shared::slab_hint;满额 × 图数 = VRAM 爆的工程债);
        // hint 缺席 = 64MiB 兜底
        let env_mb = self.diag.capture_slab_mb.filter(|&v| v > 0);
        let slab_cap = match env_mb {
            Some(mb) => mb << 20,
            None => owl_shared::slab_hint::take_hint().unwrap_or(64 << 20),
        };
        let slab = unsafe { stream.alloc::<u8>(slab_cap) }
            .map_err(|e| ModelError::Msg(format!("graph_begin: 捕获 slab 分配失败({}MiB) {e:?}", slab_cap >> 20)))?;
        owl_shared::vram::live_add(slab_cap as i64); // B6.5(直配不走 handle_alloc)
        let slab = Arc::new(slab);
        stream
            .begin_capture(CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| ModelError::Msg(format!("graph_begin: {e:?}")))?;
        self.capture = Some(CaptureState { stream, slab, used: 0, launches: 0, cap: slab_cap });
        Ok(())
    }

    /// 图捕获结束(实例化 + 登记)。空捕获窗拒绝(哨兵③:发射数须 > 0)。
    pub(super) fn graph_end(&mut self) -> Result<GraphId, ModelError> {
        let cap = self
            .capture
            .take()
            .ok_or_else(|| ModelError::Msg("graph_end: 当前没有进行中的捕获".to_string()))?;
        // 哨兵③:空捕获窗(begin_capture 后一次发射都没有)
        if cap.launches == 0 {
            // 仍须 end_capture 退出捕获态;空图被驱动以 NULL 返回
            let g = unsafe { crate::ffi::stream_end_capture(cap.stream.cu_stream()) }
                .map_err(|e| ModelError::Msg(format!("graph_end: {e:?}")))?;
            if !g.is_null() {
                let _ = unsafe { crate::ffi::graph_destroy(g) };
            }
            return Err(ModelError::Msg(
                "graph_end: 空捕获窗(发射数 = 0;先在捕获期发射 ≥ 1 个 kernel)".to_string(),
            ));
        }
        // 裸端捕获 + 零旗标实例化(AUTO_FREE_ON_LAUNCH 定谳见 ffi.rs;
        // 图执行语义与 safe 版逐位一致 —— 同 cuGraphLaunch 入 COMPUTE 流)
        let cu_graph = unsafe { crate::ffi::stream_end_capture(cap.stream.cu_stream()) }
            .map_err(|e| ModelError::Msg(format!("graph_end: {e:?}")))?;
        if cu_graph.is_null() {
            return Err(ModelError::Msg(
                "graph_end: 空捕获窗(发射数 = 0;哨兵③)".to_string(),
            ));
        }
        let mut exec: crate::ffi::sys::CUgraphExec = std::ptr::null_mut();
        // 刀D:实例化旗标 A/B 门(UPLOAD=2 预热 / DEVICE_LAUNCH=4 设备侧
        // 派发;派发税 10.8µs/节点 vs vLLM ~1.4 立案的根因排查面)
        let flags: u64 = self.diag.graph_flags;
        unsafe { crate::ffi::sys::cuGraphInstantiateWithFlags(&mut exec, cu_graph, flags) }
            .result()
            .map_err(|e| ModelError::Msg(format!("graph_end: 实例化(flags={flags}): {e:?}")))?;
        // 节点数/类型观测(S1 launch 税立案:探针判 0.33µs/节点,引擎图
        // 实测 1.84µs/节点 —— 按类型分解找贵节点)
        if self.srv_timing {
            let mut n: usize = 0;
            unsafe {
                crate::ffi::sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut n);
            }
            let mut nodes = vec![std::ptr::null_mut(); n];
            let mut got: usize = n;
            unsafe {
                crate::ffi::sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut got);
            }
            let mut kinds: std::collections::HashMap<i32, usize> = Default::default();
            let mut alloc_sizes: std::collections::HashMap<usize, usize> = Default::default();
            let mut cpy_sizes: std::collections::HashMap<(&str, usize), usize> = Default::default();
            let mut kern_names: std::collections::HashMap<String, usize> = Default::default();
            for nd in &nodes[..got] {
                let mut t: crate::ffi::sys::CUgraphNodeType = unsafe { std::mem::zeroed() };
                unsafe {
                    crate::ffi::sys::cuGraphNodeGetType(*nd, &mut t);
                }
                *kinds.entry(t as i32).or_default() += 1;
                // 刀1 取证:MEM_ALLOC 节点尺寸分布(派发税定位;真枚举:
                // 0=EMPTY 1=KERNEL 2=MEMCPY 3=MEMSET 4=HOST 5=GRAPH
                // 10=MEM_ALLOC 11=MEM_FREE)
                if t as u32 == 0 {
                    // 刀D:内核名清单(图材料单;派发税定位)
                    let mut params: crate::ffi::sys::CUDA_KERNEL_NODE_PARAMS =
                        unsafe { std::mem::zeroed() };
                    let r = unsafe {
                        crate::ffi::sys::cuGraphKernelNodeGetParams_v2(*nd, &mut params)
                    };
                    if r == crate::ffi::sys::CUresult::CUDA_SUCCESS {
                        let mut name: *const std::ffi::c_char = std::ptr::null();
                        let rn = unsafe {
                            crate::ffi::sys::cuFuncGetName(&mut name, params.func)
                        };
                        if rn == crate::ffi::sys::CUresult::CUDA_SUCCESS {
                            let cname = unsafe { std::ffi::CStr::from_ptr(name) };
                            let sname = cname.to_string_lossy().to_string();
                            *kern_names.entry(sname).or_default() += 1;
                        }
                    }
                }
                if t as u32 == 10 {
                    let mut params: crate::ffi::sys::CUDA_MEM_ALLOC_NODE_PARAMS =
                        unsafe { std::mem::zeroed() };
                    let r = unsafe {
                        crate::ffi::sys::cuGraphMemAllocNodeGetParams(*nd, &mut params)
                    };
                    if r == crate::ffi::sys::CUresult::CUDA_SUCCESS {
                        *alloc_sizes.entry(params.bytesize).or_default() += 1;
                    }
                }
                if t as u32 == 1 {
                    let mut params: crate::ffi::sys::CUDA_MEMCPY_NODE_PARAMS =
                        unsafe { std::mem::zeroed() };
                    let r = unsafe {
                        crate::ffi::sys::cuGraphMemcpyNodeGetParams(*nd, &mut params.copyParams)
                    };
                    if r == crate::ffi::sys::CUresult::CUDA_SUCCESS {
                        // 3D 拷贝参数:宽度字节 = WidthInBytes;srcMemoryType/
                        // dstMemoryType 1=host 2=device(判方向)
                        let cp = &params.copyParams;
                        let kind = match (cp.srcMemoryType as u32, cp.dstMemoryType as u32) {
                            (1, 2) => "H2D",
                            (2, 1) => "D2H",
                            (2, 2) => "D2D",
                            _ => "?",
                        };
                        *cpy_sizes
                            .entry((kind, cp.WidthInBytes as usize))
                            .or_default() += 1;
                    }
                }
            }
            eprintln!("[gl-prof] 图节点 = {got}(类型 {kinds:?};真枚举 0=EMPTY 1=KERNEL 2=MEMCPY 3=MEMSET 4=HOST 10=MEM_ALLOC 11=MEM_FREE)");
            if !alloc_sizes.is_empty() {
                let mut rows: Vec<_> = alloc_sizes.into_iter().collect();
                rows.sort_by_key(|b| std::cmp::Reverse(b.0));
                for (sz, n) in rows.iter().take(12) {
                    eprintln!("[gl-prof]   MEM_ALLOC {sz}B × {n}");
                }
            }
            if !kern_names.is_empty() {
                let mut rows: Vec<_> = kern_names.into_iter().collect();
                rows.sort_by_key(|b| std::cmp::Reverse(b.1));
                for (name, n) in rows.iter().take(40) {
                    eprintln!("[gl-prof]   K {name} × {n}");
                }
            }
            if !cpy_sizes.is_empty() {
                let mut rows: Vec<_> = cpy_sizes.into_iter().collect();
                rows.sort_by(|a, b| b.0.cmp(&a.0));
                for (sz, n) in rows.iter().take(12) {
                    eprintln!("[gl-prof]   MEMCPY {:?} × {n}", sz);
                }
            }
        }
        let id = self.next_graph;
        self.next_graph += 1;
        self.graphs.insert(
            id,
            GraphHolder { exec, cu_graph, stream: cap.stream.clone() },
        );
        Ok(id)
    }

    /// 图重放(捕获期间拒绝;流序异步提交;路由 COMPUTE 流)
    pub(super) fn graph_launch(&mut self, gid: GraphId) -> Result<(), ModelError> {
        if self.capture.is_some() {
            return Err(ModelError::Msg(
                "graph_launch: 捕获进行中,不可回放(先 graph_end)".to_string(),
            ));
        }
        let stream = self.stream(STREAM_COMPUTE)?.clone();
        let prof = self.srv_timing;
        let t0 = prof.then(std::time::Instant::now);
        let (exec, cu_stream) = {
            let g = self
                .graphs
                .get(&gid)
                .ok_or_else(|| ModelError::Msg(format!("graph_launch: 图 {gid} 不存在")))?;
            (g.exec, g.stream.cu_stream())
        };
        unsafe { crate::ffi::graph_launch(exec, cu_stream) }
            .map_err(|e| ModelError::Msg(format!("graph_launch({gid}): {e:?}")))?;
        if prof {
            eprintln!("[gl-prof] gid={gid} launch={:?}", t0.unwrap().elapsed());
        }
        // 诊断开关(diag.launch_sync):回放后同步归因(烘焙指针消费者)
        if self.diag.launch_sync {
            let t1 = std::time::Instant::now();
            stream
                .synchronize()
                .map_err(|e| ModelError::Msg(format!("graph-sync({gid}): {e:?}")))?;
            eprintln!("[gl-prof] gid={gid} gpu-sync={:?}", t1.elapsed());
        }
        Ok(())
    }

    // ======================================================================
    // 池块账房
    // ======================================================================

    /// 登记新块(图外:独立分配)
    pub(super) fn new_block(&mut self, slice: CudaSlice<u8>) -> u64 {
        let bytes = slice.len();
        let local = self.next_block;
        self.next_block += 1;
        let id = pack_block_id(self.ledger, local);
        self.blocks.insert(
            id,
            BlockEntry {
                block: Block::Owned(slice),
                bytes,
                ledger: self.ledger,
                dtype: BlockDtype::Bytes,
                shape: vec![bytes],
            },
        );
        id
    }

    /// 捕获期切块(零 cudaMalloc;slab 线性碰撞分配)
    pub(super) fn carve_block(&mut self, n: usize) -> Result<u64, ModelError> {
        let cap = self.capture.as_mut().ok_or_else(|| {
            ModelError::Msg("carve_block: 仅限捕获期(内部护栏违例)".to_string())
        })?;
        // 256B 对齐切块:向量化核(uint4/float4 16B 访存、cublas、后续
        // FlashInfer)要求基底对齐;旧核全标量未暴露此缺陷(2026-09-27
        // v1 paged 核 MISALIGNED_ADDRESS 定谳)。无对齐需求的 2B 哑块也
        // 不会破坏后续块基底。
        const CARVE_ALIGN: usize = 256;
        if self.cap_prof {
            eprintln!("[cap-prof] carve {}B", n);
        }
        let used = cap.used.div_ceil(CARVE_ALIGN) * CARVE_ALIGN;
        let slab_cap = cap.cap;
        if used + n > slab_cap {
            return Err(ModelError::Msg(format!(
                "捕获 slab 耗尽:已用 {used} + 对齐后需 {n} > {slab_cap}(warmup 计量失准或 OWL_CAPTURE_SLAB_MB 固定档过小)"
            )));
        }
        let local = self.next_block;
        self.next_block += 1;
        let id = pack_block_id(self.ledger, local);
        let (off, slab) = (used, cap.slab.clone());
        cap.used = used + n;
        self.blocks.insert(
            id,
            BlockEntry {
                block: Block::Carved { slab, off, n },
                bytes: n,
                ledger: self.ledger,
                dtype: BlockDtype::Bytes,
                shape: vec![n],
            },
        );
        Ok(id)
    }

    /// 块设备指针 + 元素数(Owned 直取;Carved 切视图)
    pub(super) fn block_ptr(
        &self,
        id: u64,
        stream: &CudaStream,
    ) -> Result<(u64, usize), ModelError> {
        assert_ledger(id, self.ledger)?;
        use crate::ffi::DevicePtr;
        let e = self.blocks.get(&id).ok_or(ModelError::DeadBlock { id })?;
        // 块账自洽(哨兵;cfg! 展开 ⇒ 两档都编译,零热路径语义)
        debug_assert_eq!(e.ledger, self.ledger, "块条目账本与 id 账本不一致");
        debug_assert_eq!(numel(&e.shape), e.bytes, "块 shape/账长不一致");
        debug_assert!(e.dtype.size_bytes() >= 1, "块 dtype 非法");
        match &e.block {
            Block::Owned(slice) => {
                let (p, _) = slice.device_ptr(stream);
                Ok((p, slice.len()))
            }
            Block::Carved { slab, off, n } => {
                let view = slab.slice(*off..*off + *n);
                let (p, _) = view.device_ptr(stream);
                Ok((p, *n))
            }
        }
    }

    /// 捕获期发射计数(哨兵③:graph_end 空窗校验;Launch 恒在 COMPUTE = 捕获流)
    pub(super) fn note_launch(&mut self) {
        if let Some(cap) = &mut self.capture {
            cap.launches += 1;
        }
    }

    /// 块账长(字节口径;账长校验 + **准入哨兵**用 —— 哨兵侧
    /// `Result → Option` 降级:探测不该炸,真解析的拒绝仍在
    /// `block_ptr`/`assert_ledger`)
    pub(super) fn block_len(&self, id: u64) -> Result<usize, ModelError> {
        assert_ledger(id, self.ledger)?;
        Ok(self.blocks.get(&id).ok_or(ModelError::DeadBlock { id })?.bytes)
    }

    /// 中间块回收(E2a):移除 Owned 块(CudaSlice drop → 流序
    /// free_async 归还设备池);Carved 块(slab 切片)跳过不删;
    /// 未知 id 容错跳过(幂等)。返回实际回收数。
    /// **E5 跨流回收律(2026-10-08 崩坏案修正)**:free_async 落在块自身
    /// 分配流(htod 块 = H2D,队列浅,free 立即归池),而 COMPUTE 可能
    /// 还压着该块的在途读者(65 层深队列)——设备池把同址发给下一个
    /// alloc 即成跨流 UAF。旧注“流序安全”仅对同流复用成立。修法:drop
    /// 前 COMPUTE 记事件,H2D/D2H wait 之 —— free 被钉在读者排空后,任意
    /// 流复用皆安全;COMPUTE 同流复用本就保序不受影响。指纹全吻合:
    /// dtoh 排空掩蔽 / bisect 0/10 / GDN 臂无关 / 冷 boot 首选(与塔零
    /// 案同病,当时只修了 dtoh 侧)。捕获期跳过(事件会被烤进图)。
    pub(super) fn free_blocks(&mut self, ids: &[u64]) -> usize {
        let any_owned = ids
            .iter()
            .any(|id| self.blocks.get(id).is_some_and(|e| matches!(e.block, Block::Owned(_))));
        // diag.free_legacy:旁路事件钉序(崩坏案 A/B 判别;性能画像用)
        if any_owned && self.capture.is_none() && !self.diag.free_legacy {
            if let (Ok(comp), Ok(h2d), Ok(d2h)) = (
                self.stream(STREAM_COMPUTE),
                self.stream(STREAM_H2D),
                self.stream(STREAM_D2H),
            ) {
                if let Ok(ev) = comp.record_event(None) {
                    let _ = h2d.wait(&ev);
                    let _ = d2h.wait(&ev);
                } // 事件失败回旧语义(不让回收路径崩)
            }
        }
        let mut n = 0;
        let mut freed_bytes: i64 = 0;
        for id in ids {
            let owned = self
                .blocks
                .get(id)
                .is_some_and(|e| matches!(e.block, Block::Owned(_)));
            if owned {
                if let Some(e) = self.blocks.remove(id) {
                    if let Block::Owned(slice) = &e.block {
                        freed_bytes += slice.len() as i64; // B6.5 live 记账
                    }
                    drop(e); // free_async(块自身流;已按上律钉序)
                    n += 1;
                }
            }
        }
        owl_shared::vram::live_add(-freed_bytes);
        n
    }
}

// ============================================================================
// kernel 装载缓存(2026-10-13 载体切换:构建期 cubin 优先,零运行期编译)
// ============================================================================
//
// 两个平面:
// - **AOT 面**:`build.rs` 把每个 `.cu` 编成 fatbin(`$OUT_DIR/cubins/`),登记表
//   以**源文本哈希**为键(`owl_kernels::family::cubins`)。同一文件的核共享一个
//   模块(旧路径是**每核一个模块**,逐核 nvrtc + 重复装载)。
// - **无 NVRTC 面**(2026-10-13 裁决:零运行期编译 ⇒ NVRTC 已退役):
//   资产未命中 = **拒发**(不降级到运行期编译)。
//
// `aot_hits` / `misses` 是**机器判据面**:验收 = 整轮 e2e 日志 `aot-miss` 恒 0;
// 结构性判据 = `crates/kernels/tests/no_runtime_compile.rs`(生产路径零编译调用)。

pub(super) struct KernelCache {
    compiled: HashMap<String, CudaFunction>,
    /// AOT 资产名 → 已载模块(同 `.cu` 的核共享一个模块)
    modules: HashMap<&'static str, Arc<cudarc::driver::CudaModule>>,
    /// 资产的核数(诊断/验收判据)
    pub(super) aot_hits: usize,
    /// 无资产的核名(每名一次;= 拒启清单/机器判据面)
    pub(super) misses: Vec<String>,
}

impl KernelCache {
    pub(super) fn new() -> Self {
        Self { compiled: HashMap::new(), modules: HashMap::new(), aot_hits: 0, misses: Vec::new() }
    }

    /// 核取用:**构建期 cubin 资产唯一路径**(源哈希查表 → fatbin → 符号)。
    ///
    /// 未命中 = `Err`(拒发)。**没有 NVRTC 兜底**(2026-10-13 裁决:零运行期
    /// 编译);源码不在 `include_str!` 闭包内 = 该入口没有构建期资产 =
    /// 编程错误(要么漏建表,要么该入口已废弃应从发射链剔除)。
    pub(super) fn ensure_kernel(
        &mut self,
        ctx: &Arc<CudaContext>,
        name: &str,
        source: &str,
    ) -> Result<CudaFunction, ModelError> {
        if let Some(f) = self.compiled.get(name) {
            return Ok(f.clone());
        }
        let asset = owl_kernels::family::cubins::asset_of_source(source).ok_or_else(|| {
            // 机器判据面:整轮 e2e 日志里 `grep -c aot-miss` 应恒 0
            eprintln!("[aot-miss] {name} —— 无构建期 cubin 资产(源 {} B)", source.len());
            if !self.misses.iter().any(|m| m == name) {
                self.misses.push(name.to_string());
            }
            ModelError::Msg(format!(
                "{name} 无构建期 cubin 资产(源 {} B)—— 零运行期编译硬门:该 .cu 不在 build.rs 的 \
                 include_str! 闭包内(NVRTC 已退役,无兜底路径)",
                source.len()
            ))
        })?;
        let module = match self.modules.get(asset.asset) {
            Some(m) => m.clone(),
            None => {
                // fatbin 载入:driver API 与纯 cubin 完全相同(多代际由驱动
                // 按设备挑最匹配那份)
                let m = ctx
                    .load_module(cudarc::nvrtc::Ptx::from_binary(asset.bytes.to_vec()))
                    .map_err(|e| {
                        ModelError::Msg(format!(
                            "aot load_module({} / {name}): {e:?} —— 资产载入失败(代际不匹配?驱动过旧?)",
                            asset.asset
                        ))
                    })?;
                self.modules.insert(asset.asset, m.clone());
                m
            }
        };
        let func = module.load_function(name).map_err(|e| {
            ModelError::Msg(format!(
                "aot load_function({name}) in {}: {e:?} —— 符号不在资产里(源哈希命中但符号缺失 = 登记表/源失配)",
                asset.asset
            ))
        })?;
        self.aot_hits += 1;
        self.compiled.insert(name.to_string(), func.clone());
        Ok(func)
    }
}

// ============================================================================
// pinned host 码头(非阻塞搬运的 host 侧落点)
// ============================================================================

/// 页锁定 host 缓冲(自管生命周期)。
///
/// ⚠️ 生命周期纪律:pinned 内存在搬运完成前**不得释放**——本类型只允许
/// 被 move 进完成回调,随回调一起 drop。
pub(super) struct Staging {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for Staging {}

impl owl_iface::contract::PinnedRegion for Staging {
    fn slice_bytes_mut(&mut self) -> &mut [u8] {
        Staging::slice_mut(self)
    }
    fn as_bytes(&self) -> &[u8] {
        Staging::slice(self)
    }
}

impl Staging {
    /// 分配 n 元素的 pinned 缓冲(未初始化)
    pub(super) fn alloc(n: usize, label: &'static str) -> Result<Self, ModelError> {
        // 字节口径(f16 基线,2026-09-26):n = 字节数,DMA 按 byte 计数
        let ptr = unsafe { malloc_host(n, 0) }
            .map_err(|e| ModelError::Msg(format!("malloc_host(n={n}B): {e:?}")))? as *mut u8;
        // FIXME(2026-10-13):cudarc 的 `malloc_host` 对失败/零尺寸可能回 Ok(null)
        // ——这里是裸 assert,拿不到任何上下文。先带上尺寸与原因(下同)。
        assert!(
            !ptr.is_null(),
            "Staging::alloc[{label}]: malloc_host(n={n}B) 返回空指针(锁页内存耗尽?n=0 = 零长块写,上游形状错)"
        );
        Ok(Self { ptr, len: n })
    }

    pub(super) fn slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(super) fn slice_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        // 收尾失败无路可报(Drop 语义);sticky 错误会在后续流操作上暴露
        let _ = unsafe { free_host(self.ptr as *mut _) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 跨账本 id 解析 ⇒ 结构化 CrossLedger 错(含 id + 两个 ledger 号)。
    /// 纯解码/断言,不触 CUDA(可在无卡环境跑)。
    #[test]
    fn cross_ledger_id_is_rejected() {
        let id = pack_block_id(2, 11);
        assert_eq!(ledger_of(id), 2);
        assert_eq!(local_of(id), 11);
        let e = assert_ledger(id, 1).expect_err("跨账本必拒");
        let msg = format!("{e}");
        assert_eq!(
            msg,
            "跨账本块: 块 id=2199023255563 属 ledger 2,被 ledger 1 解析"
        );
        // 同账本放行
        assert!(assert_ledger(id, 2).is_ok());
    }

    /// 位拼接跨账本注入性:同 local 不同 ledger ⇒ id 不同(§27 撞号门根因消除)。
    #[test]
    fn pack_is_injective_across_ledgers() {
        assert_ne!(pack_block_id(1, 11), pack_block_id(2, 11));
        assert_eq!(ledger_of(pack_block_id(3, 99)), 3);
    }
}
