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

pub use crate::dtype::Dtype;
pub use crate::dtype::Scalar;

/// S2 broadcast 右对齐检查:从右往左逐维配对,相等或目标维为 1 可播,
/// 源维数不足左补 1。返回 Ok(()) 或结构化错误。
pub(crate) fn broadcast_check(from: &[usize], to: &[usize]) -> Result<(), BackendError> {
    if to.len() < from.len() {
        return Err(BackendError::Init(format!(
            "broadcast: 目标维数 {} < 源维数 {}(S2 右对齐)",
            to.len(),
            from.len()
        )));
    }
    let off = to.len() - from.len();
    for (i, &f) in from.iter().enumerate() {
        let t = to[off + i];
        if f != t && f != 1 {
            return Err(BackendError::Init(format!(
                "broadcast: 源维 {f} 不可播到目标维 {t}(S2:相等或源=1)"
            )));
        }
    }
    Ok(())
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

impl<T: Scalar, D: Device> Clone for Tensor<T, D> {
    /// 克隆 = 视图共享(底层存储引用计数 +1;对应 candle 浅拷贝语义,
    /// owl 下同时是租约克隆:活性由存储 Arc 兜底)。shape/dtype 元数据复制。
    fn clone(&self) -> Self {
        self.clone_shallow()
    }
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

    // ---- T1 签名层:shape 视图算子(语义表 SEMANTICS S3:无惰性布局,
    // 视图方法只做元数据;需要物理布局的 kernel 入口由算子层 assert)----

    /// 维数
    pub fn dims(&self) -> usize {
        self.shape.len()
    }

    /// 重塑(元素数不变;不触设备,纯元数据)
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, BackendError> {
        let n: usize = shape.iter().product();
        if n != Self::elems(&self.shape) {
            return Err(BackendError::Init(format!(
                "reshape: 元素数不变约束 {n} != {}",
                Self::elems(&self.shape)
            )));
        }
        let mut out = self.clone_shallow();
        out.shape = shape.to_vec();
        Ok(out)
    }

    /// 转置最后两维(元数据;materialize 延迟到 kernel 入口,S3)
    pub fn t2(&self) -> Result<Self, BackendError> {
        self.transpose(self.dims() - 2, self.dims() - 1)
    }

    /// 任意两维转置(元数据;形状重排,不触设备)
    pub fn transpose(&self, a: usize, b: usize) -> Result<Self, BackendError> {
        let d = self.dims();
        if a >= d || b >= d {
            return Err(BackendError::Init(format!("transpose: 维越界 {a}/{b}/{d}")));
        }
        let mut s = self.shape.clone();
        s.swap(a, b);
        let mut out = self.clone_shallow();
        out.shape = s;
        Ok(out)
    }

    /// 窄切(第 dim 维取 [start, start+len))(元数据 + 指针偏移待 kernel 回填;
    /// 编译口径:物理偏移视图在运行里程碑实现,当前只校验 shape)
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Self, BackendError> {
        let d = self.dims();
        if dim >= d || start + len > self.shape[dim] {
            return Err(BackendError::Init(format!(
                "narrow: dim={dim} start={start} len={len} shape={:?}",
                self.shape
            )));
        }
        let mut s = self.shape.clone();
        s[dim] = len;
        let mut out = self.clone_shallow();
        out.shape = s;
        Ok(out)
    }

    /// 广播到目标 shape(S2 右对齐规则;元数据)
    pub fn broadcast_as(&self, shape: &[usize]) -> Result<Self, BackendError> {
        crate::tensor::broadcast_check(&self.shape, shape)?;
        let mut out = self.clone_shallow();
        out.shape = shape.to_vec();
        Ok(out)
    }

    /// 紧凑化(S3 语义:owl Tensor 恒紧凑布局,无惰性 layout,本方法恒等;
    /// 返回共享同一存储的视图,不触设备)
    pub fn contiguous(&self) -> Result<Self, BackendError> {
        Ok(self.clone_shallow())
    }

    /// S3 语义:owl 无惰性布局,恒真
    pub fn is_contiguous(&self) -> bool {
        true
    }

    /// dtype 转换。编译先行口径:运行里程碑回填(转换 kernel 未移植),
    /// 签名先行锁定调用面(对应 candle `to_dtype`)。
    pub fn to_dtype<U: crate::dtype::Scalar>(&self) -> Result<Tensor<U, D>, BackendError> {
        Err(BackendError::Init(
            "to_dtype: 跨 dtype 转换 kernel 未回填(T1 签名层)".into(),
        ))
    }

    /// 与自身同 shape 的清零张量(对应 candle `zeros_like`)。
    /// 编译先行口径:T2 池直连工厂接通后回填(Tensor 暂不持有池引用)。
    pub fn zeros_like(&self) -> Result<Self, BackendError> {
        unimplemented!("T1 签名层:zeros_like 池引用接通待 T2")
    }

    // ---- T1 签名层:归约/组合算子(shape 校验真实,设备体待运行里程碑回填)----

    /// 沿 dim 维求和,结果 shape = self.shape 且 shape[dim] = 1(语义同 candle sum)
    pub fn sum(&self, dim: usize) -> Result<Self, BackendError> {
        let d = self.dims();
        if dim >= d {
            return Err(BackendError::Init(format!("sum: dim 越界 {dim}/{d}")));
        }
        let mut s = self.shape.clone();
        s[dim] = 1;
        let mut out = self.clone_shallow();
        out.shape = s;
        unimplemented!("T1 签名层:kernel 回填待运行里程碑");
    }

    /// 沿 dim 维取最大,结果 shape = self.shape 且 shape[dim] = 1(语义同 candle max)
    pub fn max(&self, dim: usize) -> Result<Self, BackendError> {
        let d = self.dims();
        if dim >= d {
            return Err(BackendError::Init(format!("max: dim 越界 {dim}/{d}")));
        }
        let mut s = self.shape.clone();
        s[dim] = 1;
        let mut out = self.clone_shallow();
        out.shape = s;
        unimplemented!("T1 签名层:kernel 回填待运行里程碑");
    }

    /// 沿 dim 维取最小,结果 shape = self.shape 且 shape[dim] = 1(语义同 candle min)
    pub fn min(&self, dim: usize) -> Result<Self, BackendError> {
        let d = self.dims();
        if dim >= d {
            return Err(BackendError::Init(format!("min: dim 越界 {dim}/{d}")));
        }
        let mut s = self.shape.clone();
        s[dim] = 1;
        let mut out = self.clone_shallow();
        out.shape = s;
        unimplemented!("T1 签名层:kernel 回填待运行里程碑");
    }

    /// 沿 dim 维拼接(静态方法;语义同 candle cat:除 dim 外各维须一致,dtype 须一致)
    pub fn cat(parts: &[&Self], dim: usize) -> Result<Self, BackendError> {
        if parts.is_empty() {
            return Err(BackendError::Init("cat: parts 为空".into()));
        }
        let ref_shape = parts[0].shape();
        let d = ref_shape.len();
        if dim >= d {
            return Err(BackendError::Init(format!("cat: dim 越界 {dim}/{d}")));
        }
        for p in parts {
            if p.shape().len() != d {
                return Err(BackendError::Init(format!(
                    "cat: 维数不一致 {:?} vs {:?}",
                    p.shape(),
                    ref_shape
                )));
            }
            if p.dtype() != parts[0].dtype() {
                return Err(BackendError::Init("cat: dtype 不一致(S1 禁隐式提升)".into()));
            }
            for i in 0..d {
                if i != dim && p.shape()[i] != ref_shape[i] {
                    return Err(BackendError::Init(format!(
                        "cat: 非拼接维 {i} 长度不一致 {:?} vs {:?}",
                        p.shape(),
                        ref_shape
                    )));
                }
            }
        }
        unimplemented!("T1 签名层:kernel 回填待运行里程碑");
    }

    /// 沿新 dim 维堆叠(静态方法;语义同 candle stack:各 part shape 须完全相同,
    /// dim 为新维插入位置,允许 dim = dims(右端追加))
    pub fn stack(parts: &[&Self], dim: usize) -> Result<Self, BackendError> {
        if parts.is_empty() {
            return Err(BackendError::Init("stack: parts 为空".into()));
        }
        let ref_shape = parts[0].shape();
        if dim > ref_shape.len() {
            return Err(BackendError::Init(format!(
                "stack: dim 越界 {dim}/{}(新维插入位置)",
                ref_shape.len()
            )));
        }
        for p in parts {
            if p.shape() != ref_shape {
                return Err(BackendError::Init(format!(
                    "stack: shape 不一致 {:?} vs {:?}",
                    p.shape(),
                    ref_shape
                )));
            }
            if p.dtype() != parts[0].dtype() {
                return Err(BackendError::Init("stack: dtype 不一致(S1 禁隐式提升)".into()));
            }
        }
        unimplemented!("T1 签名层:kernel 回填待运行里程碑");
    }

    /// 浅拷贝(共享同一底层存储;视图算子的实现基元。
    /// 活性由存储 Arc/租约兜底,克隆不触设备)
    fn clone_shallow(&self) -> Self {
        Self {
            storage: match &self.storage {
                Storage::Persistent(p) => Storage::Persistent(p.clone()),
                Storage::Scratch(s) => Storage::Scratch(s.clone()),
            },
            shape: self.shape.clone(),
            dtype: self.dtype,
            token: self.token,
            ctx: self.ctx.clone(),
            _dev: std::marker::PhantomData,
        }
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
    #[allow(dead_code)] // 通用版;当前调用点走 Device trait 的 from_vec_tensor 直配路径
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
        let mut host: Vec<T> = Vec::with_capacity(self.len());
        // P0-3:to_vec 无设备主流句柄,唯一可靠搭配 =
        // ctx.synchronize()(上下文级,覆盖 non-blocking 主流)+ NULL 流同步拷贝。
        // EagerOnly 路径(裁决 3③),调用方保证不在捕获段内。
        if let Some(ctx) = &self.ctx {
            ctx.synchronize()
                .map_err(|e| BackendError::Init(format!("device sync: {e:?}")))?;
        }
        unsafe {
            sys::cuMemcpyDtoH_v2(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                self.device_ptr() as sys::CUdeviceptr,
                self.len() * std::mem::size_of::<T>(),
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
    use owl_iface::{PoolConfig, PoolKind};

    fn dev() -> CudaDevice {
        CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备")
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
/// host 侧可迭代算术(T1 工厂专用:arange_tensor 在 host 算好 vec 再装载;
/// Bf16/F16 无算术语义(S5),不实现本 trait)
pub trait HostArith: Scalar + Copy + core::ops::Add<Output = Self> {}
impl HostArith for f32 {}
impl HostArith for u8 {}
impl HostArith for u32 {}
impl HostArith for i64 {}

pub trait TensorPoolOps: Pool {
    /// P 阶段:池内持久分配(清零)。Weights/KvCache/Workspace 池语义。
    fn zeros_tensor<T: Scalar>(
        &self,
        shape: &[usize],
    ) -> Result<Tensor<T, Self::Dev>, BackendError>;

    /// P 阶段:常量填充张量(复用 from_vec_tensor;对应 candle `full`)
    fn full_tensor<T: Scalar + Copy>(
        &self,
        shape: &[usize],
        fill: T,
    ) -> Result<Tensor<T, Self::Dev>, BackendError> {
        let n: usize = shape.iter().product();
        self.from_vec_tensor(shape, vec![fill; n])
    }

    /// P 阶段:等差序列 [start, start+step, ...)(长度 n,复用 from_vec_tensor;
    /// 对应 candle `arange`)。T 须 host 可迭代算术(HostArith)。
    fn arange_tensor<T: HostArith>(
        &self,
        start: T,
        step: T,
        n: usize,
    ) -> Result<Tensor<T, Self::Dev>, BackendError> {
        let mut v = Vec::with_capacity(n);
        let mut x = start;
        for _ in 0..n {
            v.push(x);
            x = x + step;
        }
        self.from_vec_tensor(&[n], v)
    }

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
