//! 缓冲类型:持久域(Persistent)/暂存域(Scratch)/远端视图(RemoteBuf)
//! 与 A2.8 VMM 独立缓冲(VmmBuf)。物理归账全部由 CudaPoolBuf/PoolBufInner
//! 负责;这里只做类型视图与租约接口。

use super::pool::CudaPoolBuf;
use crate::pool::Ledger;
use cudarc::driver::{CudaContext, sys};
use owl_iface::{BufToken, DevBuf, MemValue};
use std::sync::Arc;

/// 持久域缓冲:P 阶段经池分配;drop 归账由 PoolBuf/CudaPoolBuf 负责。
/// Clone 为手写(字段全 Clone,无需 T: Clone —— derive 会保守地误加界)
pub struct Persistent<T: MemValue> {
    pub(crate) buf: Option<CudaPoolBuf>,
    pub(crate) len: usize,
    pub(crate) token: Option<BufToken>,
    pub(crate) _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: MemValue> Clone for Persistent<T> {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
            len: self.len,
            token: self.token,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: MemValue> Persistent<T> {
    /// GraphLease:租约克隆(Arc +1)与令牌读取
    pub(crate) fn lease_parts(&self) -> (CudaPoolBuf, Option<BufToken>) {
        (self.buf.as_ref().expect("Persistent 未被 drop").clone(), self.token)
    }

    /// 池块租约(Arc +1;类型视图无关;tensor 合并后的保活形态)。
    /// 仅借块不消费外壳:Persistent 本身生命周期照旧。
    pub fn block(&self) -> CudaPoolBuf {
        self.buf.as_ref().expect("Persistent 未被 drop").clone()
    }

    /// 字节块 → 类型视图装配(cast_persistent 的零成本底座;
    /// len = 元素数,块字节容量由调用方保证 ≥ len×size_of::<T>())
    pub(crate) fn from_block(buf: CudaPoolBuf, len: usize) -> Self {
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

/// 暂存域缓冲(允许 Capturing 相创建;归账同 Persistent)。手写 Clone 同上。
pub struct Scratch<T: MemValue> {
    pub(crate) buf: Option<CudaPoolBuf>,
    pub(crate) len: usize,
    pub(crate) token: Option<BufToken>,
    pub(crate) _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: MemValue> Clone for Scratch<T> {
    fn clone(&self) -> Self {
        Self {
            buf: self.buf.clone(),
            len: self.len,
            token: self.token,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: MemValue> Scratch<T> {
    /// 池块租约(同 [`Persistent::block`],暂存域)
    pub fn block(&self) -> CudaPoolBuf {
        self.buf.as_ref().expect("Scratch 未被 drop").clone()
    }

    /// 字节块 → 暂存域类型视图装配(同 [`Persistent::from_block`])
    pub(crate) fn from_block(buf: CudaPoolBuf, len: usize) -> Self {
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
    /// 字节账归默认池账本(2026-09-23 裁决:账本归池)
    pub(crate) ledger: Arc<Ledger>,
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
        self.ledger.uncharge(self.bytes as u64);
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
