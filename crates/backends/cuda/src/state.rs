//! server 状态面:设备上下文 + 流注册表(三固定流)+ 池块账房
//! + kernel 编译缓存 + pinned 码头 + 图捕获状态机/图注册表。
//!
//! 只被 actor 线程触碰(单线程所有权;零锁)。

use crate::ffi::{device_get_count, free_host, malloc_host, CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, CAPTURE_MODE_THREAD_LOCAL, INSTANTIATE_AUTO_FREE};
use cudarc::nvrtc::safe::{compile_ptx_with_opts, CompileOptions};
use owl_models::client::GraphId;
use owl_models::ModelError;
use std::collections::HashMap;
use std::sync::Arc;

/// 捕获 slab 容量(64 MiB;图内 Alloc 从 slab 切块,零 cudaMalloc)
pub(super) const CAPTURE_SLAB_ELEMS: usize = (64 << 20) / 4;

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
    Owned(CudaSlice<f32>),
    Carved { slab: Arc<CudaSlice<f32>>, off: usize, n: usize },
}

// ============================================================================
// 设备上下文 + 流注册表
// ============================================================================

pub(super) struct GpuCtx {
    pub ctx: Arc<CudaContext>,
    /// 流注册表:三固定流(H2D/COMPUTE/D2H;客户端不可自创)
    streams: HashMap<StreamId, Arc<CudaStream>>,
    /// 图捕获状态机(见 graph_begin/end 的护栏注释)
    capture: Option<CaptureState>,
    /// 图注册表:id → 实例化图(重放用)
    graphs: HashMap<GraphId, GraphHolder>,
    next_graph: GraphId,
    /// 块账房:id → (块, 元素数)
    blocks: HashMap<u64, (Block, usize)>,
    next_block: u64,
}

/// 捕获会话:目标流 + 切块 slab + 发射计数(空窗哨兵用)
struct CaptureState {
    stream: Arc<CudaStream>,
    slab: Arc<CudaSlice<f32>>,
    used: usize,
    launches: usize,
}

/// CudaGraph 非 Send(cudarc 未标注);我们把它钉死在 actor 线程单一所有权
/// 下使用(所有图操作都在 dispatch 线程序列化),跨线程移动 GpuServer 携带
/// 它是安全的 —— 与 cudarc "须外部串行化" 的要求一致。
struct GraphHolder(CudaGraph);
unsafe impl Send for GraphHolder {}

impl GpuCtx {
    pub(super) fn new(selector: &DeviceSelector) -> Result<Self, String> {
        let ordinal = selector.resolve()?;
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("{e:?}"))?;
        ctx.bind_to_thread().map_err(|e| format!("{e:?}"))?;
        // G0 护栏:关 event-tracking(坑 A)。多流 + event tracking 会让
        // safe 层在块读写上插事件,污染图捕获(CAPTURE_ISOLATION/图内事件节点)
        unsafe { ctx.disable_event_tracking() };
        let mut streams = HashMap::new();
        for (id, _role) in [(STREAM_H2D, "h2d"), (STREAM_COMPUTE, "compute"), (STREAM_D2H, "d2h")] {
            let s = ctx.new_stream().map_err(|e| format!("建流({_role}): {e:?}"))?;
            streams.insert(id, s);
        }
        Ok(Self {
            ctx,
            streams,
            capture: None,
            graphs: HashMap::new(),
            next_graph: 1,
            blocks: HashMap::new(),
            next_block: 1,
        })
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
        let slab = unsafe { stream.alloc::<f32>(CAPTURE_SLAB_ELEMS) }
            .map_err(|e| ModelError::Msg(format!("graph_begin: 捕获 slab 分配失败 {e:?}")))?;
        let slab = Arc::new(slab);
        stream
            .begin_capture(CAPTURE_MODE_THREAD_LOCAL)
            .map_err(|e| ModelError::Msg(format!("graph_begin: {e:?}")))?;
        self.capture = Some(CaptureState { stream, slab, used: 0, launches: 0 });
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
            // 仍须 end_capture 退出捕获态;空图被驱动以 None 返回
            let _ = cap.stream.end_capture(INSTANTIATE_AUTO_FREE);
            return Err(ModelError::Msg(
                "graph_end: 空捕获窗(发射数 = 0;先在捕获期发射 ≥ 1 个 kernel)".to_string(),
            ));
        }
        let graph = cap
            .stream
            .end_capture(INSTANTIATE_AUTO_FREE)
            .map_err(|e| ModelError::Msg(format!("graph_end: {e:?}")))?
            .ok_or_else(|| {
                ModelError::Msg("graph_end: 空捕获窗(发射数 = 0;哨兵③)".to_string())
            })?;
        let id = self.next_graph;
        self.next_graph += 1;
        self.graphs.insert(id, GraphHolder(graph));
        Ok(id)
    }

    /// 图重放(捕获期间拒绝;流序异步提交;路由 COMPUTE 流)
    pub(super) fn graph_launch(&mut self, gid: GraphId) -> Result<(), ModelError> {
        if self.capture.is_some() {
            return Err(ModelError::Msg(
                "graph_launch: 捕获进行中,不可回放(先 graph_end)".to_string(),
            ));
        }
        self.stream(STREAM_COMPUTE)?;
        self.graphs
            .get(&gid)
            .ok_or_else(|| ModelError::Msg(format!("graph_launch: 图 {gid} 不存在")))?
            .0
            .launch()
            .map_err(|e| ModelError::Msg(format!("graph_launch({gid}): {e:?}")))
    }

    // ======================================================================
    // 池块账房
    // ======================================================================

    /// 登记新块(图外:独立分配)
    pub(super) fn new_block(&mut self, slice: CudaSlice<f32>) -> u64 {
        let n = slice.len();
        let id = self.next_block;
        self.next_block += 1;
        self.blocks.insert(id, (Block::Owned(slice), n));
        id
    }

    /// 捕获期切块(零 cudaMalloc;slab 线性碰撞分配)
    pub(super) fn carve_block(&mut self, n: usize) -> Result<u64, ModelError> {
        let cap = self.capture.as_mut().ok_or_else(|| {
            ModelError::Msg("carve_block: 仅限捕获期(内部护栏违例)".to_string())
        })?;
        if cap.used + n > CAPTURE_SLAB_ELEMS {
            return Err(ModelError::Msg(format!(
                "捕获 slab 耗尽:已用 {} + 需 {n} > {}(提高 CAPTURE_SLAB_ELEMS)",
                cap.used,
                CAPTURE_SLAB_ELEMS
            )));
        }
        let id = self.next_block;
        self.next_block += 1;
        let (off, slab) = (cap.used, cap.slab.clone());
        cap.used += n;
        self.blocks.insert(id, (Block::Carved { slab, off, n }, n));
        Ok(id)
    }

    /// 块设备指针 + 元素数(Owned 直取;Carved 切视图)
    pub(super) fn block_ptr(
        &self,
        id: u64,
        stream: &CudaStream,
    ) -> Result<(u64, usize), ModelError> {
        use crate::ffi::DevicePtr;
        match &self.blocks.get(&id).ok_or(ModelError::DeadBlock { id })?.0 {
            Block::Owned(slice) => {
                let (p, _) = slice.device_ptr(stream);
                Ok((p as u64, slice.len()))
            }
            Block::Carved { slab, off, n } => {
                let view = slab.slice(*off..*off + *n);
                let (p, _) = view.device_ptr(stream);
                Ok((p as u64, *n))
            }
        }
    }

    /// 捕获期发射计数(哨兵③:graph_end 空窗校验;Launch 恒在 COMPUTE = 捕获流)
    pub(super) fn note_launch(&mut self) {
        if let Some(cap) = &mut self.capture {
            cap.launches += 1;
        }
    }

    /// 块元素数(账长校验用)
    pub(super) fn block_len(&self, id: u64) -> Result<usize, ModelError> {
        Ok(self.blocks.get(&id).ok_or(ModelError::DeadBlock { id })?.1)
    }
}

// ============================================================================
// kernel 编译缓存(懒编译;name+source → CudaFunction)
// ============================================================================

pub(super) struct KernelCache {
    compiled: HashMap<String, CudaFunction>,
}

impl KernelCache {
    pub(super) fn new() -> Self {
        Self { compiled: HashMap::new() }
    }

    /// 懒编译:CUDA C 源 → nvrtc → PTX → module → function(命中缓存直返)
    pub(super) fn ensure_kernel(
        &mut self,
        ctx: &Arc<CudaContext>,
        name: &str,
        source: &str,
    ) -> Result<CudaFunction, ModelError> {
        if let Some(f) = self.compiled.get(name) {
            return Ok(f.clone());
        }
        let ptx = compile_ptx_with_opts(
            source,
            CompileOptions { arch: Some("compute_86"), ..Default::default() },
        )
        .map_err(|e| ModelError::Msg(format!("nvrtc({name}): {e:?}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| ModelError::Msg(format!("load_module({name}): {e:?}")))?;
        let func = module
            .load_function(name)
            .map_err(|e| ModelError::Msg(format!("load_function({name}): {e:?}")))?;
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
    ptr: *mut f32,
    len: usize,
}

unsafe impl Send for Staging {}

impl Staging {
    /// 分配 n 元素的 pinned 缓冲(未初始化)
    pub(super) fn alloc(n: usize) -> Result<Self, ModelError> {
        let ptr = unsafe { malloc_host(n * 4, 0) }
            .map_err(|e| ModelError::Msg(format!("malloc_host: {e:?}")))? as *mut f32;
        assert!(!ptr.is_null());
        Ok(Self { ptr, len: n })
    }

    pub(super) fn slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub(super) fn slice_mut(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for Staging {
    fn drop(&mut self) {
        // 收尾失败无路可报(Drop 语义);sticky 错误会在后续流操作上暴露
        let _ = unsafe { free_host(self.ptr as *mut _) };
    }
}
