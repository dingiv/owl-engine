//! 缓冲类型:持久域(Persistent)/暂存域(Scratch)/远端视图(RemoteBuf)
//! 与 A2.8 VMM 独立缓冲(VmmBuf)。物理归账全部由 CudaPoolBuf/PoolBufInner
//! 负责;这里只做类型视图与租约接口。

use crate::pool::CudaPool;
use cudarc::driver::{CudaContext, sys};
use owl_iface::{DevBuf, MemValue};
use std::sync::Arc;

// ---- A2.8 VMM 独立缓冲 ----

/// A2.8 VMM 缓冲(独立入口;PeerShared 池内路径用 PoolBacking::Vmm)
pub struct VmmBuf {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) bytes: usize,
    pub(crate) chunk: sys::CUmemGenericAllocationHandle,
    pub(crate) ctx: Arc<CudaContext>,
    /// 字节账归池(账本是池的字段;2026-09-24 内联裁决)
    pub(crate) pool: Arc<CudaPool>,
}

impl VmmBuf {
    pub fn device_ptr(&self) -> sys::CUdeviceptr {
        self.ptr
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for VmmBuf {
    fn drop(&mut self) {
        let _ctx = Arc::clone(&self.ctx); // 保活至清理完成
        let _ = _ctx.bind_to_thread(); // P1-A:释放线程未必持本 ctx
        unsafe {
            use sys::{cuMemAddressFree, cuMemRelease, cuMemUnmap};
            if cuMemUnmap(self.ptr, self.bytes) != sys::CUresult::CUDA_SUCCESS {
                eprintln!("owl-cuda: VMM cuMemUnmap 失败,物理页泄漏");
            }
            if cuMemRelease(self.chunk) != sys::CUresult::CUDA_SUCCESS {
                eprintln!("owl-cuda: VMM cuMemRelease 失败,物理 handle 泄漏");
            }
            if cuMemAddressFree(self.ptr, self.bytes) != sys::CUresult::CUDA_SUCCESS {
                eprintln!("owl-cuda: VMM cuMemAddressFree 失败,VA 泄漏");
            }
        }
        self.pool.uncharge(self.bytes as u64);
    }
}

// ---- P2P 远端映射视图 ----

pub struct RemoteBuf<T: MemValue> {
    pub(crate) ptr: *mut T,
    pub(crate) len: usize,
    /// 对端卡 UUID(A2.6 窄口/诊断归因;M1 P2P 映射接线时消费)
    #[allow(dead_code)]
    pub(crate) peer_uuid: String,
}

unsafe impl<T: Send + 'static> Send for RemoteBuf<T> {}

impl<T: MemValue> DevBuf<T> for RemoteBuf<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn device_ptr(&self) -> *mut T {
        self.ptr
    }
}
