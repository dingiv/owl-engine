//! cudarc 受控再导出(A4:精确到接口粒度,禁止整库 re-export)。
//! 这是上层(nn/未来的 rocnn)唯一可见的 cudarc 表面。
//! 扩充本清单 = 扩大 driver 依赖面,须过 backends/README 准入审核并登记。

// driver safe 层(逐项)
pub use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};

/// **NULL 流同步原语**(raw driver `cuMemcpy*_v2` 语义:与任何非默认流**无序**)
///
/// ⚠️ **A13 定谳(2026-10-15)**:`result::memcpy_*_sync` 是 **NULL 流同步拷贝** ——
/// 用它写一个由 `stream.alloc`(流序 `cudaMallocAsync`)刚给的地址,两者**零序**
/// ⇒ 写会落进尚未落定的内存并被**静默丢弃**(compute-sanitizer 判据:
/// `Use-before-alloc @ cuMemcpyHtoD ← cuMemAllocAsync`)。本仓**三次**踩中
/// (2026-10-09 “meta 表读回全零”/ `xqa.rs` / `gdn_chunked.rs:453`),每次都靠
/// 本地补 sync 绕过 —— 直到 A13 才追到写侧。
///
/// **纪律(只允许 `GpuRes` 用)**:
///   ① 目标是流序分配/要参与后续核 ⇒ 改用同流 `memcpy_htod_async(..., COMPUTE)`
///      (首选,零代价,分配→拷贝→核 同流严格有序);
///   ② 回读/自检类同步 D2H ⇒ **拷贝前先排空目标流**(`GpuRes::download` 已内建)。
///
/// 故此处**降级 `pub(crate)`**:不对 crate 外开放这条 NULL 流后门;
/// 外部要回读走 `DeviceRes::download`(自带排空),要上传走 `DeviceRes::upload`。
/// 机器判据:`crates/backends/cuda/tests/raw_sync_primitives_gate.rs`。
pub(crate) use cudarc::driver::result::{memcpy_dtoh_sync, memcpy_htod_sync};

/// result 图原语(M1 图治理 + 哨兵③;只返回句柄,自身零分配——
/// 准入铁律通过,证据:cudarc 0.19.9 result.rs:788/801/1494/1514/1522/1531)
pub use cudarc::driver::result::graph::{
    destroy as graph_destroy, exec_destroy as graph_exec_destroy, instantiate as graph_instantiate,
    launch as graph_launch,
};
pub use cudarc::driver::result::stream::{
    end_capture as stream_end_capture, is_capturing as stream_is_capturing,
};

/// **事件原语**(E1 三流事件序:`docs/arch/三流事件序-设计与迁移.md`)。
///
/// 准入铁律:只取句柄(零分配、零 host 阻塞),证据:cudarc 0.19.9
/// result.rs:1224(create)/1238(record)/1288(synchronize)/1302(destroy)。
/// `create` 是 safe 函数(需线程已 bind 当前上下文 —— 只允许 `GpuCtx::new`
/// 在 server 线程调用);其余为 unsafe raw 调用。
/// 纪律:事件**只住 server 线程**(A3 同步隔离),客户端永不可见。
pub use cudarc::driver::result::event::{
    create as event_create, destroy as event_destroy, record as event_record,
    synchronize as event_synchronize,
};

/// **跨流等待**(E2/E3 的序消费口):B 流等到 A 流的事件后才继续 —— 零 host
/// 往返(对比:host 全排空 ≈1.9ms/键)。证据:cudarc 0.19.9 result.rs:749
/// `stream::wait_event`(unsafe:流/事件须未释放)。
pub use cudarc::driver::result::stream::wait_event as stream_wait_event;

/// 裸 FFI(唯一低层出口;逐项白名单)
pub mod sys {
    // driver 侧:指针类型 + 显式拷贝(tensor 装载/回读)
    pub use cudarc::driver::sys::{
        CUdeviceptr, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuMemcpyDtoDAsync_v2, cuMemsetD8Async, cuMemsetD16Async, cuMemsetD32Async,
        // E1 事件序:句柄类型 + 创建标志(证据:cudarc 0.19.9 sys/mod.rs:423 CUevent /
        // :43 CUevent_flags(别名) / :3058 CUevent_flags_enum)
        CUevent, CUevent_flags, CUevent_wait_flags,
        // graph 捕获(M1 归入 graph 治理层)
        CUgraphInstantiate_flags, CUstreamCaptureMode,
        // 图实例化零旗标入口(S1;见上方 graph 定谳注释)
        cuGraphInstantiateWithFlags,
        // 图审计与枚举(哨兵③,M1;只读查询 API,零分配零懒状态——
        // 准入铁律通过,证据:cudarc 0.19.9 sys 绑定均为 load::<_F> 直通)
        CUDA_KERNEL_NODE_PARAMS, CUfunction, CUresult, CUgraph, CUgraphExec, CUgraphNode,
        CUgraphNodeType, CUkernel, cuGraphGetNodes, cuGraphKernelNodeGetParams_v2, cuGraphNodeGetType, cuGraphUpload,
        cuKernelGetParamInfo,
        // 刀1 取证:MEM_ALLOC/FREE/MEMCPY 节点参数(派发税立案;只读查询)
        CUDA_MEM_ALLOC_NODE_PARAMS, cuGraphMemAllocNodeGetParams, cuGraphMemFreeNodeGetParams,
        CUDA_MEMCPY_NODE_PARAMS, cuGraphMemcpyNodeGetParams,
        cuFuncGetName,
    };

    /// cuBLAS FFI(matmul/workspace/stream 重绑)
    pub mod cublas {
        pub use cudarc::cublas::sys::{
            cublasCreate_v2, cublasDestroy_v2, cublasHandle_t, cublasOperation_t,
            cublasSetStream_v2, cublasSetWorkspace_v2, cublasSgemm_v2, cublasStatus_t,
        };
    }
}

// ============================================================================
// 异步搬运 + host 完成回调(server 事件循环;A4 登记)
// ============================================================================

pub use cudarc::driver::DriverError;

/// pinned host 码头(自管;cuda 的 PinnedHostSlice 取裸指针会强制同步,不合用)
pub use cudarc::driver::result::{free_host, malloc_host};

/// 设备枚举(UUID 钉卡;数字序事故免疫 —— CUDA 序 ≠ nvidia-smi 序)
pub use cudarc::driver::result::device::{
    get as device_get, get_count as device_get_count, get_uuid as device_get_uuid,
};
/// 驱动初始化(result 层设备枚举的前置条件;幂等)
pub use cudarc::driver::result::init as driver_init;

/// 取指定 ordinal 的设备 UUID(内部自管 cuInit;UUID 钉卡的配套查询)
pub fn device_uuid(ordinal: usize) -> Result<[u8; 16], DriverError> {
    driver_init()?;
    let dev = device_get(ordinal as i32)?;
    let u = device_get_uuid(dev)?;
    Ok(u.bytes.map(|b| b as u8))
}

/// 图句柄 safe 面(捕获产物;非 Send —— 钉死 actor 线程使用)
pub use cudarc::driver::CudaGraph;
pub use cudarc::driver::sys::CUstreamCaptureMode;
pub use cudarc::driver::sys::CUgraphInstantiate_flags;

/// 捕获模式常量(ThreadLocal = 只锁本线程,不霸锁全进程)
pub const CAPTURE_MODE_THREAD_LOCAL: CUstreamCaptureMode =
    CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL;
/// 实例化走裸 `cuGraphInstantiateWithFlags`(S1,2026-09-30 定谳改判):
/// 旧档 AUTO_FREE_ON_LAUNCH 注释赌「只作用于图内分配节点,无影响」——
/// 实测每次 cuGraphLaunch 恒定 **+4.45ms** 驱动侧 auto-free 处理(decode
/// 40 tok/s 的主税)。cudarc safe end_capture 强绑该旗标(sys enum 无 0
/// 变体,transmute 被运行时校验拒绝),零旗标只能裸调。
/// 准入证据:cudarc 0.19.9 sys/mod.rs:10640(load 直通,与其他白名单项同款)
///
/// 流序清零(非阻塞;可捕获 —— 图内 memset 节点 = 每次 replay 重清零)
pub use cudarc::driver::result::memset_d8_async;

/// host 完成回调(现代 API;cuLaunchHostFunc,取代废弃的 cudaStreamAddCallback)。
/// ⚠️ 回调跑在驱动线程:**禁止调用任何 CUDA API、禁止阻塞** —— 只允许
/// 做纯 host 侧动作(如 channel send)。
pub use cudarc::driver::result::stream::launch_host_function;

/// 非阻塞搬运(流序;host 侧须为 pinned)
pub use cudarc::driver::result::{memcpy_dtoh_async, memcpy_htod_async};
/// 设备内块→块拷贝(DtoD,流序非阻塞;E2c GDN 快照拍摄/恢复通道)
pub use cudarc::driver::result::memcpy_dtod_async;

