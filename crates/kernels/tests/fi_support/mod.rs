//! FI 探针共享面(`fi_prefill_probe` 与 `fi_graph_replay_probe` 共用)。
//!
//! 抽出的理由:两份探针都要一个"最小 DeviceRes + 真 cudarc 上下文"的测试资源面
//! (与 `ops_gdn_chunked_chain` 同构);两份各写一份 = 两套事实源。**不放进 lib**:
//! TestRes 是测试域专用,lib 零依赖律不破。
#![allow(dead_code)] // 两份探针各用其中一部分

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use owl_kernels::contract::{Bytes, Dtype, OpError};
use owl_kernels::device::{DeviceRes, LaunchNote, ScratchBuf};

pub const PAGE: usize = 32;
pub const HKV: usize = 4;
pub const H: usize = 24;
pub const HD: usize = 256;

// ============================================================================
// 测试资源面(与 ops_gdn_chunked_chain 同构的最小 DeviceRes)
// ============================================================================
pub struct TestRes {
    dev_ordinal: i32,
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<cudarc::driver::CudaStream>,
    pub blocks: HashMap<u64, CudaSlice<u8>>,
    next_id: u64,
    /// 捕获窗句柄(P-6b/plan 自携探针用):置 true 即让家族以为自己在窗内
    /// —— production 后端由 `CaptureState` 置位,探针无捕获台账,手工当开关。
    /// 窗内路径的合法性(不许 sync/不许同步 H2D)由 [`TestRes::upload_invariant`]
    /// 的 async 实现对称保障。
    pub capture_flag: bool,
}

impl TestRes {
    pub fn new(dev: usize) -> Self {
        let ctx = CudaContext::new(dev).expect("ctx");
        ctx.bind_to_thread().expect("bind");
        // ⚠️ **必须自建流**:`ctx.default_stream()` 是 **legacy 默认流(stream 0)**,
        // 而 CUDA **不允许对 legacy 默认流开捕获**(会得到
        // `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` —— 报错措辞还很误导)。
        // 生产 server 同样是三固定流(COMPUTE/H2D/D2H),不走默认流。
        // **G0 护栏(与生产 `GpuCtx::new` 逐字同款)**:关 cudarc 的 event-tracking。
        // 开着它,`device_ptr()`/读写会在块操作上插 **event**;捕获窗内一插就
        // **invalidate capture**(实测:窗内 FI run 返回 901 =
        // `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`,而窗外的同一次调用完全正常)。
        // 生产侧把这条记为"坑 A",这里照抄 —— 探针必须与生产同姿态。
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().expect("new_stream");
        Self { dev_ordinal: dev as i32, ctx, stream, blocks: HashMap::new(), next_id: 1, capture_flag: false }
    }
    pub fn upload(&mut self, bytes: &[u8]) -> Bytes {
        let id = self.next_id;
        self.next_id += 1;
        let mut d = unsafe { self.stream.alloc::<u8>(bytes.len().max(1)) }.expect("alloc");
        self.stream.memcpy_htod(bytes, &mut d).expect("htod");
        self.blocks.insert(id, d);
        Bytes::new(id, bytes.len(), Dtype::F16, 2)
    }
    pub fn zeros(&mut self, bytes: usize) -> Bytes {
        let id = self.next_id;
        self.next_id += 1;
        let d = self.stream.alloc_zeros::<u8>(bytes.max(1)).expect("alloc0");
        self.blocks.insert(id, d);
        Bytes::new(id, bytes, Dtype::F16, 2)
    }
    /// 按 f32 读回一个块(状态/表件;测试面)
    pub fn take_f32(&self, id: u64) -> Vec<f32> {
        let s = self.blocks.get(&id).expect("块在册");
        let host = self.stream.clone_dtoh(s).expect("dtoh");
        host.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    pub fn take_u16(&self, id: u64) -> Vec<u16> {
        let s = self.blocks.get(&id).expect("块在册");
        let host = self.stream.clone_dtoh(s).expect("dtoh");
        host.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    /// 整块回读(u8;plan/workspace 取证面 —— 按尺寸找块见 `block_id_of_len`)
    pub fn take_bytes(&self, id: u64) -> Vec<u8> {
        let s = self.blocks.get(&id).expect("块在册");
        self.stream.clone_dtoh(s).expect("dtoh")
    }

    /// 按**块字节数**找一个块 id(探针取证:FiRuntime 的 int_ws = 32MiB、
    /// float_ws = 64MiB,尺寸在测试面唯一可辨)
    pub fn block_id_of_len(&self, bytes: usize) -> Option<u64> {
        self.blocks.iter().find(|(_, s)| s.len() == bytes).map(|(id, _)| *id)
    }
}

impl DeviceRes for TestRes {
    fn context(&self) -> Result<Arc<CudaContext>, OpError> {
        Ok(self.ctx.clone())
    }
    fn stream(&self) -> Result<Arc<cudarc::driver::CudaStream>, OpError> {
        Ok(self.stream.clone())
    }
    fn resolve(&self, id: u64) -> Result<u64, OpError> {
        self.blocks
            .get(&id)
            .map(|s| s.device_ptr(&self.stream).0)
            .ok_or_else(|| OpError::Contract {
                op: "fi-probe".into(),
                field: "resolve",
                expect: "块在册".into(),
                got: id.to_string(),
            })
    }
    fn resolve_len(&self, id: u64) -> Result<usize, OpError> {
        self.blocks.get(&id).map(|s| s.len()).ok_or_else(|| OpError::Contract {
            op: "fi-probe".into(),
            field: "resolve_len",
            expect: "块在册".into(),
            got: id.to_string(),
        })
    }
    fn alloc(&mut self, bytes: usize, _tag: &'static str) -> Result<ScratchBuf, OpError> {
        let d = self.stream.alloc_zeros::<u8>(bytes.max(1)).expect("alloc");
        let ptr = d.device_ptr(&self.stream).0;
        self.blocks.insert(self.next_id, d);
        self.next_id += 1;
        Ok(ScratchBuf { ptr, bytes })
    }
    fn alloc_block(&mut self, bytes: usize, _tag: &'static str) -> Result<Bytes, OpError> {
        Ok(self.zeros(bytes))
    }
    fn upload_block(&mut self, b: &Bytes, src: &[u8]) -> Result<(), OpError> {
        let stream = self.stream.clone();
        let s = self.blocks.get_mut(&b.id).ok_or_else(|| OpError::Contract {
            op: "fi-probe".into(),
            field: "upload_block",
            expect: "块在册".into(),
            got: format!("id={}", b.id),
        })?;
        stream.memcpy_htod(src, s).map_err(|e| OpError::Contract {
            op: "fi-probe".into(),
            field: "upload_block",
            expect: "htod".into(),
            got: format!("{e:?}"),
        })
    }
    fn capturing(&self) -> Result<bool, OpError> {
        Ok(self.capture_flag)
    }
    /// 不变盒上传:**async**(捕获窗内也能跑;production 同形)。
    /// 同步 `memcpy_htod` 在窗内会直接 invalidate 捕获(M2a 姿态坑 #2 同族)。
    fn upload_invariant(
        &mut self,
        dst: u64,
        src: &owl_kernels::contract::InvariantBox,
    ) -> Result<(), OpError> {
        let target = self
            .blocks
            .iter()
            .find(|(_, s)| s.device_ptr(&self.stream).0 == dst)
            .map(|(id, _)| *id);
        match target {
            Some(id) => {
                let v = self.blocks.get_mut(&id).expect("在册");
                let (ptr, _g) = v.device_ptr_mut(&self.stream);
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        ptr,
                        src.as_bytes(),
                        self.stream.cu_stream(),
                    )
                    .map_err(|e| OpError::Contract {
                        op: "fi-probe".into(),
                        field: "upload_invariant",
                        expect: "async htod".into(),
                        got: format!("{e:?}"),
                    })?;
                }
                Ok(())
            }
            None => Err(OpError::Contract {
                op: "fi-probe".into(),
                field: "upload_invariant",
                expect: "指针在册".into(),
                got: format!("{dst:#x}"),
            }),
        }
    }
    fn record_capture(&mut self, _note: LaunchNote) -> Result<(), OpError> {
        Ok(())
    }
    fn device_ordinal(&self) -> Result<i32, OpError> {
        Ok(self.dev_ordinal)
    }
    fn upload(&mut self, dst: u64, src: &[u8]) -> Result<(), OpError> {
        let target = self
            .blocks
            .iter()
            .find(|(_, s)| s.device_ptr(&self.stream).0 == dst)
            .map(|(id, _)| *id);
        match target {
            Some(id) => {
                let v = self.blocks.get_mut(&id).expect("在册");
                self.stream.memcpy_htod(src, v).expect("htod");
                Ok(())
            }
            None => Err(OpError::Contract {
                op: "fi-probe".into(),
                field: "upload",
                expect: "指针在册".into(),
                got: format!("{dst:#x}"),
            }),
        }
    }
    fn download(&mut self, src: u64, dst: &mut [u8]) -> Result<(), OpError> {
        let target = self
            .blocks
            .iter()
            .find(|(_, s)| s.device_ptr(&self.stream).0 == src)
            .map(|(id, _)| *id);
        match target {
            Some(id) => {
                let v = self.blocks.get(&id).expect("在册");
                let host = self.stream.clone_dtoh(v).expect("dtoh");
                let n = dst.len().min(host.len());
                dst[..n].copy_from_slice(&host[..n]);
                Ok(())
            }
            None => Err(OpError::Contract {
                op: "fi-probe".into(),
                field: "download",
                expect: "指针在册".into(),
                got: format!("{src:#x}"),
            }),
        }
    }
}


/// f16 → f32(探针读回口径)
pub fn f16_to_f32(b: u16) -> f32 {
    half::f16::from_bits(b).to_f32()
}

// ============================================================================
// 裸图捕获面(★ 只测**可捕获性/回放语义**;成本面走生产 ffi,不经这里)
// ============================================================================
//
// 为什么不用 cudarc 的 safe `CudaGraph`:
// ① `end_capture(flags)` 的旗标类型是**无零变体**的枚举,而生产用 `flags=0`
//    (AUTO_FREE_ON_LAUNCH 会让每 launch 多 4.45ms —— 见 ffi.rs 定谳);
// ② 本探针要对同一条已捕获图**反复回放**(换表件 / 换 mask 档),需要自己拿
//    exec 句柄。故走 sys 裸口(与 `crates/backends/cuda/src/ffi.rs` 同款)。
pub struct RawGraph {
    pub exec: cudarc::driver::sys::CUgraphExec,
    graph: cudarc::driver::sys::CUgraph,
}

impl RawGraph {
    /// 开捕获(流级;与生产的 CAPTURE_MODE_THREAD_LOCAL 同款)。
    /// ⚠️ 必须先 `bind_to_thread` —— cudarc 的 raw `result::stream::*` **不做**
    /// 上下文绑定(只有 safe 包装做),多上下文进程里漏绑会得到
    /// `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` 这种看不出真因的错。
    pub fn begin(
        ctx: &cudarc::driver::CudaContext,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), String> {
        ctx.bind_to_thread().map_err(|e| format!("bind: {e:?}"))?;
        unsafe {
            cudarc::driver::result::stream::begin_capture(
                stream.cu_stream(),
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
        }
        .map_err(|e| format!("begin_capture: {e:?}"))
    }

    /// 收捕获 + 实例化(零旗标)
    pub fn end(
        ctx: &cudarc::driver::CudaContext,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<Self, String> {
        ctx.bind_to_thread().map_err(|e| format!("bind: {e:?}"))?;
        let graph = unsafe { cudarc::driver::result::stream::end_capture(stream.cu_stream()) }
            .map_err(|e| format!("end_capture: {e:?}"))?;
        if graph.is_null() {
            return Err("end_capture: 空图".into());
        }
        let mut exec: cudarc::driver::sys::CUgraphExec = std::ptr::null_mut();
        let rc = unsafe { cudarc::driver::sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) };
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            unsafe { cudarc::driver::sys::cuGraphDestroy(graph) };
            return Err(format!("instantiate: {rc:?}"));
        }
        Ok(Self { exec, graph })
    }

    pub fn launch(
        &self,
        ctx: &cudarc::driver::CudaContext,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), String> {
        ctx.bind_to_thread().map_err(|e| format!("bind: {e:?}"))?;
        let rc = unsafe { cudarc::driver::sys::cuGraphLaunch(self.exec, stream.cu_stream()) };
        if rc != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Err(format!("graph_launch: {rc:?}"));
        }
        Ok(())
    }
}

impl Drop for RawGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::cuGraphExecDestroy(self.exec);
            let _ = cudarc::driver::sys::cuGraphDestroy(self.graph);
        }
    }
}

/// 设备侧改写一个已上传块的**内容**(地址不变)—— 图回放语义的核心动作:
/// "换表件/换输入槽"就是这一下(块 id 恒定 ⇒ 图里的指针恒定,内容变)
pub fn rewrite_block(res: &mut TestRes, b: &Bytes, bytes: &[u8]) {
    res.upload_block(b, bytes).expect("rewrite_block");
}
