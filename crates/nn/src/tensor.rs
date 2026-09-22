//! Tensor:池分配的张量句柄(分配 = P 阶段;本文件归 T1)。
//!
//! 裁决 5:分配与使用两阶段分离。本类型只在 **P 阶段**(启动/捕获前)
//! 经 [`owl_iface::Device`] 的池 API 出生;算子层(T3)只读它的视图,
//! 不可能经它分配。D2H 回读([`Tensor::to_vec`])标 [`crate::EagerOnly`]。
//!
//! port 出处:dtype/shape 语义参照 candle `WithDType`/`Shape`
//! (repos/candle-gb/candle-core/src/{dtype,shape}.rs),存储改为
//! owl-cuda 池缓冲(账本化)。

use crate::BufToken;
use owl_cuda::ffi::CudaContext;
use owl_cuda::CudaDevice;
use owl_iface::{BackendError, Device, DevBuf, MemValue, Pool};

/// 张量元素类型(一期 f32;f16 预留)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    /// 预留:M3 接入(f16 kernel 与 cublas 路径齐备后启用)
    F16,
}

/// 可入池标量:类型 ↔ Dtype 元数据桥
pub trait Scalar: MemValue {
    const DTYPE: Dtype;
}

impl Scalar for f32 {
    const DTYPE: Dtype = Dtype::F32;
}

/// 存储域:持久(权重/KV,Idle 外 drop 延迟)或暂存(捕获期可创建)。
/// 类型参数来自 Device 的关联类型——HAL 无泄漏(不点名具体后端类型)。
enum Storage<D: Device, T: MemValue> {
    Persistent(D::Persistent<T>),
    Scratch(D::Scratch<T>),
}

impl<D: Device, T: MemValue> DevBuf<T> for Storage<D, T> {
    fn len(&self) -> usize {
        match self {
            Storage::Persistent(p) => p.len(),
            Storage::Scratch(s) => s.len(),
        }
    }
    fn device_ptr(&self) -> *mut T {
        match self {
            Storage::Persistent(p) => p.device_ptr(),
            Storage::Scratch(s) => s.device_ptr(),
        }
    }
}

/// 池分配的张量句柄。**只在 P 阶段创建**;E 阶段算子只拿
/// `&Tensor` 读视图(device_ptr/shape),无任何分配入口。
pub struct Tensor<T: MemValue, D: Device> {
    storage: Storage<D, T>,
    shape: Vec<usize>,
    dtype: Dtype,
    /// 哨兵①:出生令牌(捕获期依赖记录的钥匙;后端无账本则 None)
    token: Option<BufToken>,
    /// D2H 绑定上下文用(仅 OwlCuda 路径;泛型设备时为 None)
    ctx: Option<std::sync::Arc<CudaContext>>,
    _dev: std::marker::PhantomData<D>,
}

impl<T: Scalar, D: Device> Tensor<T, D> {
    fn elems(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// (私有)由 Persistent 存储装配;token 由设备签发
    fn from_persistent(d: &D, storage: D::Persistent<T>, shape: &[usize]) -> Self {
        let token = d.persistent_token(&storage);
        Self {
            storage: Storage::Persistent(storage),
            shape: shape.to_vec(),
            dtype: T::DTYPE,
            token,
            ctx: None,
            _dev: std::marker::PhantomData,
        }
    }

    /// (私有)由 Scratch 存储装配
    fn from_scratch(d: &D, storage: D::Scratch<T>, shape: &[usize]) -> Self {
        let token = d.scratch_token(&storage);
        Self {
            storage: Storage::Scratch(storage),
            shape: shape.to_vec(),
            dtype: T::DTYPE,
            token,
            ctx: None,
            _dev: std::marker::PhantomData,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn device_ptr(&self) -> *mut T {
        self.storage.device_ptr()
    }

    /// 哨兵①:出生令牌(捕获期依赖记录;后端无账本则 None)
    pub fn token(&self) -> Option<BufToken> {
        self.token
    }
}

impl<T: Scalar, D: Device> DevBuf<T> for Tensor<T, D> {
    fn len(&self) -> usize {
        Self::elems(&self.shape)
    }
    fn device_ptr(&self) -> *mut T {
        self.storage.device_ptr()
    }
}

// ---- OwlCuda 专属:EagerOnly 的 D2H 回读 + GraphLease 租约登记 ----

impl<T: Scalar> Tensor<T, CudaDevice> {
    /// GraphLease:暴露底层持久存储,供捕获会话登记租约(哨兵①)。
    pub fn persistent(&self) -> Option<&owl_cuda::Persistent<T>> {
        match &self.storage {
            Storage::Persistent(p) => Some(p),
            _ => None,
        }
    }

    /// P 阶段:host → device 持久写入(权重装载)。(内部;经 PoolTensorOps)
    fn from_vec_cuda(
        d: &CudaDevice,
        pool: &<CudaDevice as Device>::Pool,
        shape: &[usize],
        src: Vec<T>,
    ) -> Result<Self, BackendError> {
        let storage = Storage::Persistent(d.htod_persistent_in::<T>(pool, src)?);
        let token = match &storage {
            Storage::Persistent(p) => d.persistent_token(p),
            _ => unreachable!("from_vec_cuda: 应为 Persistent"),
        };
        Ok(Self {
            token,
            storage,
            shape: shape.to_vec(),
            dtype: T::DTYPE,
            ctx: Some(std::sync::Arc::clone(d.ctx())),
            _dev: std::marker::PhantomData,
        })
    }

    /// **EagerOnly**:D2H 回读(同步;禁入捕获段)。
    pub fn to_vec(&self) -> Result<Vec<T>, BackendError> {
        use owl_cuda::ffi::sys;
        if let Some(ctx) = &self.ctx {
            ctx.bind_to_thread()
                .map_err(|e| BackendError::Init(format!("{e:?}")))?;
        }
        let bytes = self.len() * std::mem::size_of::<T>();
        let mut host: Vec<T> = Vec::with_capacity(self.len());
        let src = self.device_ptr() as sys::CUdeviceptr;
        // 安全:ptr 来自本设备的池缓冲,长度按元素字节数;同步 D2H 属
        // EagerOnly 路径(裁决 3③),调用方保证不在捕获段内。
        unsafe {
            sys::cuMemcpyDtoH_v2(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                src,
                bytes,
            )
            .result()
            .map_err(|e| BackendError::CopyFailed {
                dir: "dtoh",
                detail: format!("{e:?}"),
            })?;
        }
        unsafe { host.set_len(self.len()) };
        Ok(host)
    }
}

// EagerOnly 标记:D2H 回读禁入捕获段(裁决 3③)
impl<T: Scalar> crate::EagerOnly for Tensor<T, CudaDevice> {
    const EAGER_ONLY: bool = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    use owl_iface::{Pool, PoolConfig, PoolKind};

    fn dev() -> CudaDevice {
        CudaDevice::new(0).expect("需要 CUDA 设备")
    }

    fn scratch_pool(d: &CudaDevice, bytes: u64) -> <CudaDevice as Device>::Pool {
        d.create_pool(PoolConfig {
            name: "t1-test".into(),
            kind: PoolKind::Weights,
            bytes,
        })
        .expect("create_pool")
    }

    /// A5.4 冒烟:池耗尽 → PoolExhausted(分配在 P 阶段可失败)
    #[test]
    fn pool_exhausted_returns_error() {
        let d = dev();
        let pool = scratch_pool(&d, 1024 * 1024); // 1MiB
        let over = pool.zeros_tensor::<f32>(&[512 * 1024]); // 2MiB
        assert!(matches!(
            over,
            Err(BackendError::PoolExhausted { .. })
        ));
    }

    /// A5.2 账本:反复 alloc+drop,bytes_alive 回归基线(无漂移)
    #[test]
    fn ledger_stable_across_alloc_drop_cycles() {
        let d = dev();
        let pool = scratch_pool(&d, 1024 * 1024);
        let baseline = d.ledger().bytes_alive;
        for _ in 0..10 {
            let t = pool.zeros_tensor::<f32>(&[100]).unwrap();
            assert_eq!(DevBuf::<f32>::len(&t), 100);
            drop(t); // Idle 相:立即归账
            assert_eq!(d.ledger().bytes_alive, baseline);
        }
    }

    /// htod/dtoh 往返(EagerOnly 路径)数值一致
    #[test]
    fn htod_dtoh_roundtrip() {
        let d = dev();
        let pool = scratch_pool(&d, 64 * 1024);
        let src: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();
        let t = pool.from_vec_tensor(&[64], src.clone()).unwrap();
        assert_eq!(t.shape(), &[64]);
        assert_eq!(t.dtype(), Dtype::F32);
        assert_eq!(t.to_vec().unwrap(), src);
    }

    /// scratch 张量:捕获期可创建的域(此处仅验证创建/长度/device_ptr)
    #[test]
    fn scratch_tensor_basics() {
        let d = dev();
        let pool = d
            .create_pool(PoolConfig {
                name: "t1-scratch".into(),
                kind: PoolKind::Scratch,
                bytes: 64 * 1024,
            })
            .expect("create_pool");
        let t = pool.scratch_tensor::<f32>(&[16, 16]).unwrap();
        assert_eq!(t.shape(), &[16, 16]);
        assert_eq!(DevBuf::<f32>::len(&t), 256);
        assert!(!t.device_ptr().is_null());
    }
}

// ---- 池直连张量创建(裁决:分配入口在 Pool,Device 不出现在调用点)----

/// 池对象的张量工厂扩展。约束 `P::Dev::Pool = P`(池与设备的闭环),
/// 由 blanket impl 的 where 子句表达;调用点形如:
///
/// ```ignore
/// let w = pool.zeros_tensor::<f32>(&[M, K])?;      // 持久域
/// let s = pool.scratch_tensor::<f32>(&[M, N])?;    // 暂存域
/// let w = pool.from_vec_tensor(&[M, K], host)?;    // host 装载
/// ```
pub trait TensorPoolOps: Pool {
    /// P 阶段:池内持久分配(清零)。Weights/KvCache/Workspace 池语义。
    fn zeros_tensor<T: Scalar>(
        &self,
        shape: &[usize],
    ) -> Result<Tensor<T, Self::Dev>, BackendError>;

    /// P 阶段:池内暂存分配(清零;捕获期可创建)。Scratch 池语义。
    fn scratch_tensor<T: Scalar>(
        &self,
        shape: &[usize],
    ) -> Result<Tensor<T, Self::Dev>, BackendError>;

    /// P 阶段:host 装载(权重初始化)。shape 元素数须与 src 一致。
    fn from_vec_tensor<T: Scalar>(
        &self,
        shape: &[usize],
        src: Vec<T>,
    ) -> Result<Tensor<T, Self::Dev>, BackendError>;
}

impl<P, D> TensorPoolOps for P
where
    P: Pool<Dev = D>,
    D: Device<Pool = P>,
{
    fn zeros_tensor<T: Scalar>(&self, shape: &[usize]) -> Result<Tensor<T, D>, BackendError> {
        let dev = self.device();
        let n: usize = shape.iter().product();
        Ok(Tensor::from_persistent(&dev, dev.alloc_persistent_in::<T>(self, n)?, shape))
    }

    fn scratch_tensor<T: Scalar>(&self, shape: &[usize]) -> Result<Tensor<T, D>, BackendError> {
        let dev = self.device();
        let n: usize = shape.iter().product();
        Ok(Tensor::from_scratch(&dev, dev.alloc_scratch_in::<T>(self, n)?, shape))
    }

    fn from_vec_tensor<T: Scalar>(&self, shape: &[usize], src: Vec<T>) -> Result<Tensor<T, D>, BackendError> {
        let n: usize = shape.iter().product();
        if src.len() != n {
            return Err(BackendError::CopyFailed {
                dir: "htod",
                detail: format!("shape 元素数 {n} != src.len() {}", src.len()),
            });
        }
        let dev = self.device();
        let storage = Storage::Persistent(dev.htod_persistent_in::<T>(self, src)?);
        let token = match &storage {
            Storage::Persistent(p) => dev.persistent_token(p),
            Storage::Scratch(_) => unreachable!("from_vec_tensor: 应为 Persistent"),
        };
        Ok(Tensor {
            token,
            storage,
            shape: shape.to_vec(),
            dtype: T::DTYPE,
            ctx: None,
            _dev: std::marker::PhantomData,
        })
    }
}
