//! 缓冲类型:持久域(Persistent)/暂存域(Scratch)/远端视图(RemoteBuf)
//! 与 A2.8 VMM 独立缓冲(VmmBuf)。物理归账全部由 CudaPoolBuf/PoolBufInner
//! 负责;这里只做类型视图与租约接口。

use super::pool::CudaPoolBuf;
use crate::governor::Governor;
use cudarc::driver::{CudaContext, sys};
use owl_iface::{BufToken, DevBuf, MemValue};
use std::sync::Arc;

/// 持久域缓冲:P 阶段经池分配;drop 归账由 PoolBuf/CudaPoolBuf 负责。
#[derive(Clone)]
pub struct Persistent<T: MemValue> {
    pub(crate) buf: Option<CudaPoolBuf>,
    pub(crate) len: usize,
    pub(crate) token: Option<BufToken>,
    pub(crate) _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: MemValue> Persistent<T> {
    /// GraphLease:租约克隆(Arc +1)与令牌读取
    pub(crate) fn lease_parts(&self) -> (CudaPoolBuf, Option<BufToken>) {
        (self.buf.as_ref().expect("Persistent 未被 drop").clone(), self.token)
    }

    pub(crate) fn new(buf: CudaPoolBuf, len: usize) -> Self {
        let token = buf.token();
        Self {
            buf: Some(buf),
            len,
            token: Some(token),
            _marker: std::marker::PhantomData,
        }
    }

    /// 哨兵①:缓冲身份令牌(租约/审计/世代校验的入口;pub = 跨 crate
    /// 的依赖追踪词汇,与 Scratch::token 对称)
    pub fn token(&self) -> Option<BufToken> {
        self.token
    }
}

impl<T: MemValue> DevBuf<T> for Persistent<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn device_ptr(&self) -> *mut T {
        self.buf
            .as_ref()
            .expect("Persistent 未被 drop")
            .device_ptr() as *mut T
    }
}

/// 暂存域缓冲(允许 Capturing 相创建;归账同 Persistent)。
#[derive(Clone)]
pub struct Scratch<T: MemValue> {
    pub(crate) buf: Option<CudaPoolBuf>,
    pub(crate) len: usize,
    pub(crate) token: Option<BufToken>,
    pub(crate) _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: MemValue> Scratch<T> {
    pub(crate) fn new(buf: CudaPoolBuf, len: usize) -> Self {
        let token = buf.token();
        Self {
            buf: Some(buf),
            len,
            token: Some(token),
            _marker: std::marker::PhantomData,
        }
    }

    pub(crate) fn token(&self) -> Option<BufToken> {
        self.token
    }
}

impl<T: MemValue> DevBuf<T> for Scratch<T> {
    fn len(&self) -> usize {
        self.len
    }
    fn device_ptr(&self) -> *mut T {
        self.buf
            .as_ref()
            .expect("Scratch 未被 drop")
            .device_ptr() as *mut T
    }
}

// ---- A2.8 VMM 独立缓冲 ----

/// A2.8 VMM 缓冲(独立入口;PeerShared 池内路径用 PoolBacking::Vmm)
pub struct VmmBuf {
    pub(crate) ptr: sys::CUdeviceptr,
    pub(crate) bytes: usize,
    pub(crate) chunk: sys::CUmemGenericAllocationHandle,
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) gov: Arc<Governor>,
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
        unsafe {
            use sys::{cuMemAddressFree, cuMemRelease, cuMemUnmap};
            let _ = cuMemUnmap(self.ptr, self.bytes);
            let _ = cuMemRelease(self.chunk);
            let _ = cuMemAddressFree(self.ptr, self.bytes);
        }
        self.gov.uncharge(self.bytes as u64);
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
