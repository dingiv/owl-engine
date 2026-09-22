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
        CUdeviceptr, cuMemcpyDtoH_v2, cuMemcpyHtoD_v2, cuMemcpyDtoDAsync_v2, cuMemsetD8Async, cuMemsetD32Async,
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
