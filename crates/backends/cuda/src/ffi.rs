//! cudarc 受控再导出(A4:精确到接口粒度,禁止整库 re-export)。
//! 这是上层(nn/未来的 rocnn)唯一可见的 cudarc 表面。
//! 扩充本清单 = 扩大 driver 依赖面,须过 backends/README 准入审核并登记。

// driver safe 层(逐项)
pub use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};

/// result 同步原语(逐项;当前仅 D2H 回读)
pub use cudarc::driver::result::memcpy_dtoh_sync;

/// result 图原语(M1 图治理 + 哨兵③;只返回句柄,自身零分配——
/// 准入铁律通过,证据:cudarc 0.19.9 result.rs:788/801/1494/1514/1522/1531)
pub use cudarc::driver::result::graph::{
    destroy as graph_destroy, exec_destroy as graph_exec_destroy, instantiate as graph_instantiate,
    launch as graph_launch,
};
pub use cudarc::driver::result::stream::{
    end_capture as stream_end_capture, is_capturing as stream_is_capturing,
};

/// 裸 FFI(唯一低层出口;逐项白名单)
pub mod sys {
    // driver 侧:指针类型 + 显式拷贝(tensor 装载/回读)
    pub use cudarc::driver::sys::{
        CUdeviceptr, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuMemcpyDtoDAsync_v2, cuMemsetD8Async, cuMemsetD16Async, cuMemsetD32Async,
        // graph 捕获(M1 归入 graph 治理层)
        CUgraphInstantiate_flags, CUstreamCaptureMode,
        // 图审计与枚举(哨兵③,M1;只读查询 API,零分配零懒状态——
        // 准入铁律通过,证据:cudarc 0.19.9 sys 绑定均为 load::<_F> 直通)
        CUDA_KERNEL_NODE_PARAMS, CUfunction, CUresult, CUgraph, CUgraphExec, CUgraphNode,
        CUgraphNodeType, CUkernel, cuGraphGetNodes, cuGraphKernelNodeGetParams_v2, cuGraphNodeGetType, cuGraphUpload,
        cuKernelGetParamInfo,
    };

    /// cuBLAS FFI(matmul/workspace/stream 重绑)
    pub mod cublas {
        pub use cudarc::cublas::sys::{
            cublasCreate_v2, cublasDestroy_v2, cublasHandle_t, cublasOperation_t,
            cublasSetStream_v2, cublasSetWorkspace_v2, cublasSgemm_v2, cublasStatus_t,
        };
    }
}

/// nvrtc 运行时编译(kernel 加载)
pub mod nvrtc {
    pub use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
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
/// 实例化 flag(cudarc 无 NONE 变体;AUTO_FREE 只作用于图内分配节点,
/// 我们 slab 图外预分配,无影响)
pub const INSTANTIATE_AUTO_FREE: CUgraphInstantiate_flags =
    CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;

/// host 完成回调(现代 API;cuLaunchHostFunc,取代废弃的 cudaStreamAddCallback)。
/// ⚠️ 回调跑在驱动线程:**禁止调用任何 CUDA API、禁止阻塞** —— 只允许
/// 做纯 host 侧动作(如 channel send)。
pub use cudarc::driver::result::stream::launch_host_function;

/// 非阻塞搬运(流序;host 侧须为 pinned)
pub use cudarc::driver::result::{memcpy_dtoh_async, memcpy_htod_async};

