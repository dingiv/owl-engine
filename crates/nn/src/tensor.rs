//! Tensor:池分配张量(合并后唯一公共类型;分配 = P 阶段;本文件归 T1)。
//!
//! 2026-09-24 tensor 合并(candle 单类型模型,主 agent 裁决):
//! - `Tensor<D>` = dtype 擦除的公共句柄(吸收原 `DynTensor`,六臂
//!   DynKeepalive 枚举 → `block: Option<CudaPoolBuf>` 单一保活——池块
//!   本就无类型,iface 字节化后 TypedTensor 的 Storage 包装不再是保活
//!   载体);
//! - 类型只在算子内部:经 [`Tensor::view`](原 downcast)得
//!   [`TensorRef`](`S1` 运行期 dtype 检查,不匹配 = 结构化报错);
//! - `*mut u8` 算术收口进类型化访问器([`Tensor::f32_ptr_at`] 等,
//!   审计 P0-5:cat 案根因封死);
//! - `TypedTensor<T, D>` = 旧强类型面(过渡存续,消费点逐步收缩至删除)。
//!
//! 裁决 5:分配与使用两阶段分离。本类型只在 **P 阶段**(启动/捕获前)
//! 经 [`owl_iface::Device`] 的池 API 出生;算子层(T3)只读它的视图,
//! 不可能经它分配。D2H 回读([`Tensor::to_vec`] / [`Tensor::f32_at`])
//! 是 EagerOnly。
//!
//! port 出处:dtype/shape 语义参照 candle `WithDType`/`Shape`
//! (repos/candle-gb/candle-core/src/{dtype,shape}.rs),存储改为
//! owl-cuda 池缓冲(账本化)。

use crate::BufToken;
use owl_cuda::ffi::CudaContext;
use owl_cuda::ffi::sys;
use owl_cuda::CudaDevice;
use owl_iface::{BackendError, Device, DevBuf, MemValue, OpaqueDevBuf, Pool};

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

// ============================================================================
// 合并 Tensor:唯一公共类型(原 DynTensor 吸收;2026-09-24)
// ============================================================================

/// dtype 擦除的池张量句柄(同设备内;Clone = 租约克隆语义)。
///
/// 保活形态:`block: Option<D::Bytes>` —— 字节域池块句柄(iface
/// `Device::Bytes`;**单一形态**,取代六臂 DynKeepalive;池块本就无类型,
/// CUDA 下 = CudaPoolBuf,内即 Arc<PoolBufInner>)。
/// `None` = from_raw 裸视图:调用方保证缓冲存活。
#[derive(Clone)]
pub struct Tensor<D: Device> {
    /// 设备基址(唯一裸指针;构造时缓存。算术只许走 f32_ptr_at 收口)
    ptr: *mut u8,
    shape: Vec<usize>,
    dtype: Dtype,
    /// 按字节计的元素数缓存(shape 乘积 × size_bytes)
    len_bytes: usize,
    /// 哨兵①:出生令牌(捕获期依赖记录的钥匙;裸视图 = None)
    token: Option<BufToken>,
    /// 保活底层字节域池块(单一形态;from_raw = None)
    block: Option<D::Bytes>,
    _dev: std::marker::PhantomData<fn() -> D>,
}

impl<D: Device> Tensor<D> {
    /// dtype 标签构造:由既有字节域池块装配(纯元数据;块租约随行)。
    /// 容量守卫:shape 乘积 × dtype 宽度 > 池块字节 = 结构化报错
    /// (cat 案教训:偏移算术必须有长度边界)。
    pub fn from_bytes(dtype: Dtype, shape: &[usize], block: D::Bytes) -> Result<Self, BackendError> {
        let n: usize = shape.iter().product();
        let len_bytes = n * dtype.size_bytes();
        let cap = DevBuf::<u8>::len(&block);
        if len_bytes > cap {
            return Err(BackendError::Init(format!(
                "Tensor::from_bytes: 需 {len_bytes}B > 池块 {cap}B"
            )));
        }
        Ok(Self {
            ptr: DevBuf::<u8>::device_ptr(&block),
            shape: shape.to_vec(),
            dtype,
            len_bytes,
            token: Some(OpaqueDevBuf::token(&block)),
            block: Some(block),
            _dev: std::marker::PhantomData,
        })
    }

    /// 裸指针视图构造(无保活;shape = 逻辑形状)。
    /// 仅 bindings 类长期缓冲使用(存活方 = GraphPlan/SlotBank)。
    pub fn from_raw_bytes(ptr: *mut u8, dtype: Dtype, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            ptr,
            shape: shape.to_vec(),
            dtype,
            len_bytes: n * dtype.size_bytes(),
            token: None,
            block: None,
            _dev: std::marker::PhantomData,
        }
    }

    pub fn from_raw_u32(ptr: *mut u32, shape: &[usize]) -> Self {
        Self::from_raw_bytes(ptr as *mut u8, Dtype::U32, shape)
    }

    pub fn from_raw_f32(ptr: *mut f32, shape: &[usize]) -> Self {
        Self::from_raw_bytes(ptr as *mut u8, Dtype::F32, shape)
    }

    // ---- 元数据面 ----

    /// dtype 查询
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len_bytes(&self) -> usize {
        self.len_bytes
    }

    /// 哨兵①:令牌透传(捕获租约登记面)
    pub fn token(&self) -> Option<BufToken> {
        self.token
    }

    /// 裸设备指针(擦除层只许传地址给 kernel/记录,不许解引用算术;
    /// 类型化访问见 f32_ptr/f32_ptr_at)
    pub fn device_ptr(&self) -> *mut u8 {
        self.ptr
    }

    // ---- 类型化视图(S1;原 downcast 改名)----

    /// 下行转换:擦除 → 类型化只读视图(dtype 不匹配 = 结构化报错,
    /// S1 禁隐式提升)。活性由保活块/调用方兜底。
    pub fn view<T: crate::dtype::Scalar>(&self) -> Result<TensorRef<'_, T, D>, BackendError> {
        if self.dtype != T::DTYPE {
            return Err(BackendError::Init(format!(
                "Tensor::view: 擦除 dtype {} != 请求 {}(S1 禁隐式提升)",
                self.dtype,
                T::DTYPE
            )));
        }
        Ok(TensorRef {
            ptr: self.ptr as *mut T,
            shape: &self.shape,
            _marker: std::marker::PhantomData,
        })
    }

    /// [`Tensor::view`] 的旧名(过渡别名;Phase 3 随调用点迁移删除)
    #[doc(hidden)]
    pub fn downcast<T: crate::dtype::Scalar>(&self) -> Result<TensorRef<'_, T, D>, BackendError> {
        self.view::<T>()
    }

    // ---- P0-5:类型化访问器(*mut u8 算术唯一合法出口)----

    fn require_dtype(&self, want: Dtype, what: &str) -> Result<(), BackendError> {
        if self.dtype != want {
            return Err(BackendError::Init(format!(
                "Tensor::{what}: dtype {} != {want}(S1 结构化报错)",
                self.dtype
            )));
        }
        Ok(())
    }

    /// f32 基址指针(dtype≠F32 = 结构化报错)
    pub fn f32_ptr(&self) -> Result<*mut f32, BackendError> {
        self.require_dtype(Dtype::F32, "f32_ptr")?;
        Ok(self.ptr as *mut f32)
    }

    /// f32 元素偏移指针(elem_offset 单位 = 元素,字节换算与越界检查
    /// 在此收口;越界 = 结构化报错)
    pub fn f32_ptr_at(&self, elem_offset: usize) -> Result<*mut f32, BackendError> {
        self.require_dtype(Dtype::F32, "f32_ptr_at")?;
        let byte_off = elem_offset.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            BackendError::Init(format!("f32_ptr_at: 偏移 {elem_offset} 溢出"))
        })?;
        if byte_off + std::mem::size_of::<f32>() > self.len_bytes {
            return Err(BackendError::Init(format!(
                "f32_ptr_at: 偏移 {elem_offset} 越界(len_bytes={})",
                self.len_bytes
            )));
        }
        Ok(unsafe { self.ptr.add(byte_off) } as *mut f32)
    }

    /// **EagerOnly**:读单个 f32 元素(同步 D2H 4 字节;禁入捕获段)。
    /// dtype≠F32 = 结构化报错。
    pub fn f32_at(&self, elem_offset: usize) -> Result<f32, BackendError> {
        let p = self.f32_ptr_at(elem_offset)?;
        let mut out = 0f32;
        unsafe {
            sys::cuMemcpyDtoH_v2(
                &mut out as *mut f32 as *mut core::ffi::c_void,
                p as sys::CUdeviceptr,
                std::mem::size_of::<f32>(),
            )
        }
        .result()
        .map_err(|e| BackendError::CopyFailed {
            dir: "dtoh",
            detail: format!("f32_at: {e:?}"),
        })?;
        Ok(out)
    }

    // ---- 视图算子(元数据;真实现)----

    /// 重塑(元素数不变;S6 形参多态:切片/Vec/usize/元组均可)
    pub fn reshape(&self, shape: impl ShapeLike) -> Result<Self, BackendError> {
        let shape = shape.to_shape_vec();
        let n: usize = shape.iter().product();
        let elems: usize = self.shape.iter().product();
        if n != elems {
            let bt = std::backtrace::Backtrace::force_capture();
            eprintln!("[RDBG] Tensor::reshape 元素数 {n} != {elems} shape={:?}\n{bt}", self.shape);
            return Err(BackendError::Init(format!(
                "Tensor::reshape: 元素数 {n} != {elems}"
            )));
        }
        let mut out = self.clone();
        out.shape = shape;
        Ok(out)
    }

    /// 摊平(reshape 特例)
    pub fn flatten_all(&self) -> Result<Self, BackendError> {
        let n: usize = self.shape.iter().product();
        self.reshape(&[n])
    }

    /// 第 0 维窄切(基址偏移 + 保活克隆;连续块子集)。
    /// 非 0 维窄切需 stride 语义(S3),当前不支持 = 结构化报错。
    pub fn narrow_dim0(&self, start: usize, len: usize) -> Result<Self, BackendError> {
        if len == 0 || start + len > self.shape[0] {
            return Err(BackendError::Init(format!(
                "Tensor::narrow_dim0: start={start} len={len} shape={:?}",
                self.shape
            )));
        }
        let mut out = self.clone();
        let row: usize = self.shape[1..].iter().product::<usize>().max(1) * self.dtype.size_bytes();
        out.ptr = unsafe { self.ptr.add(start * row) };
        out.shape[0] = len;
        Ok(out)
    }

    // ---- EagerOnly 回读 ----

    /// **EagerOnly**:D2H 回读(同步;禁入捕获段)。
    /// S1:请求 `T::DTYPE` ≠ 擦除 dtype = 结构化报错(运行期 dtype 分发表)。
    pub fn to_vec<T: crate::dtype::Scalar>(&self) -> Result<Vec<T>, BackendError> {
        self.require_dtype(T::DTYPE, "to_vec")?;
        let n = self.len_bytes / std::mem::size_of::<T>();
        let mut host: Vec<T> = Vec::with_capacity(n);
        // EagerOnly 路径(裁决 3③):NULL 流同步拷贝;调用方保证不在捕获段
        unsafe {
            sys::cuMemcpyDtoH_v2(
                host.as_mut_ptr() as *mut core::ffi::c_void,
                self.ptr as sys::CUdeviceptr,
                self.len_bytes,
            )
        }
        .result()
        .map_err(|e| BackendError::CopyFailed {
            dir: "dtoh",
            detail: format!("to_vec: {e:?}"),
        })?;
        unsafe { host.set_len(n) };
        Ok(host)
    }
}

// EagerOnly 标记:D2H 回读禁入捕获段(裁决 3③)
impl<D: Device> crate::EagerOnly for Tensor<D> {
    const EAGER_ONLY: bool = true;
}

// ---- OwlCuda 专属:TypedTensor 擦除桥 + 块租约出口(泛型 Device 无法
// 具体化关联类型的内部块;擦除/租约路径本就 CUDA 具体)----

impl Tensor<CudaDevice> {
    /// 从强类型张量擦除(过渡桥,Phase 3 随调用点迁移删除;
    /// 块租约随行 = 计数 +1,活性语义不变)
    pub fn from_f32(t: &TypedTensor<f32, CudaDevice>) -> Self {
        Self::from_typed(t, Dtype::F32)
    }
    pub fn from_f16(t: &TypedTensor<crate::dtype::F16, CudaDevice>) -> Self {
        Self::from_typed(t, Dtype::F16)
    }
    pub fn from_bf16(t: &TypedTensor<crate::dtype::Bf16, CudaDevice>) -> Self {
        Self::from_typed(t, Dtype::BF16)
    }
    pub fn from_u8(t: &TypedTensor<u8, CudaDevice>) -> Self {
        Self::from_typed(t, Dtype::U8)
    }
    pub fn from_u32(t: &TypedTensor<u32, CudaDevice>) -> Self {
        Self::from_typed(t, Dtype::U32)
    }
    pub fn from_i64(t: &TypedTensor<i64, CudaDevice>) -> Self {
        Self::from_typed(t, Dtype::I64)
    }

    fn from_typed<T: Scalar>(t: &TypedTensor<T, CudaDevice>, dtype: Dtype) -> Self {
        let block = t.block();
        // 容量不可失败:强类型面块容量 ≥ 元素面(构造不变式),dtype 同源
        Self::from_bytes(dtype, t.shape(), block)
            .expect("TypedTensor 擦除:容量不变式破坏")
    }

    /// 块租约出口(图租约/手工登记用;裸视图 = None)
    pub fn lease_block(&self) -> Option<owl_cuda::CudaPoolBuf> {
        self.block.as_ref().cloned()
    }
}

/// 下行转换的只读视图(不拥有;活性由源 Tensor 的保活块兜底)
pub struct TensorRef<'a, T: MemValue, D: Device> {
    ptr: *mut T,
    shape: &'a [usize],
    _marker: std::marker::PhantomData<fn() -> (T, D)>,
}

impl<'a, T: MemValue, D: Device> TensorRef<'a, T, D> {
    pub fn device_ptr(&self) -> *mut T {
        self.ptr
    }
    pub fn shape(&self) -> &'a [usize] {
        self.shape
    }
}

/// 形状多态(S6 搬运友好:candle 调用点传切片/元组/Vec 均可)
pub trait ShapeLike {
    fn to_shape_vec(self) -> Vec<usize>;
}
impl ShapeLike for &[usize] {
    fn to_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}
impl ShapeLike for Vec<usize> {
    fn to_shape_vec(self) -> Vec<usize> {
        self
    }
}
impl ShapeLike for usize {
    fn to_shape_vec(self) -> Vec<usize> {
        vec![self]
    }
}
impl ShapeLike for (usize,) {
    fn to_shape_vec(self) -> Vec<usize> {
        vec![self.0]
    }
}
impl<const N: usize> ShapeLike for [usize; N] {
    fn to_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}
impl<const N: usize> ShapeLike for &[usize; N] {
    fn to_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}
macro_rules! shape_like_tuple {
    ($($t:ident),+; $($i:tt),+) => {
        impl ShapeLike for ($($t),+) {
            fn to_shape_vec(self) -> Vec<usize> {
                vec![$(self.$i),+]
            }
        }
    };
}
shape_like_tuple!(usize, usize; 0, 1);
shape_like_tuple!(usize, usize, usize; 0, 1, 2);
shape_like_tuple!(usize, usize, usize, usize; 0, 1, 2, 3);
shape_like_tuple!(usize, usize, usize, usize, usize; 0, 1, 2, 3, 4);

// ============================================================================
// TypedTensor:旧强类型面(过渡存续;消费点收缩至零后删除)
// ============================================================================

/// 存储域:持久(权重/KV,Idle 外 drop 延迟)或暂存(捕获期可创建)。
/// 类型参数来自 Device 的关联类型——HAL 无泄漏(不点名具体后端类型)。
// 2026-09-24 池去类型裁决:Storage(Persistent/Scratch 双臂)退役——
// 池块本就无类型,TypedTensor 直接持有字节块 D::Bytes + phantom T。

/// 池分配的强类型张量句柄(**只在 P 阶段创建**;E 阶段算子只拿
/// `&TypedTensor` 读视图,无任何分配入口)。
pub struct TypedTensor<T: MemValue, D: Device> {
    /// 池字节块(唯一存储形态;类型是 phantom 标注)
    pub(crate) block: D::Bytes,
    shape: Vec<usize>,
    dtype: Dtype,
    /// D2H 绑定上下文用(仅 OwlCuda 路径;泛型设备时为 None)
    ctx: Option<std::sync::Arc<CudaContext>>,
    _marker: std::marker::PhantomData<fn() -> (T, D)>,
}

impl<T: Scalar, D: Device> Clone for TypedTensor<T, D> {
    /// 克隆 = 视图共享(底层存储引用计数 +1;对应 candle 浅拷贝语义,
    /// owl 下同时是租约克隆:活性由存储 Arc 兜底)。shape/dtype 元数据复制。
    fn clone(&self) -> Self {
        self.clone_shallow()
    }
}

impl<T: Scalar, D: Device> TypedTensor<T, D> {
    fn elems(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// (私有)由池字节块装配(ctx 由 CudaPool 签发时绑定)
    fn from_block(block: D::Bytes, shape: &[usize], ctx: Option<std::sync::Arc<CudaContext>>) -> Self {
        Self {
            block,
            shape: shape.to_vec(),
            dtype: T::DTYPE,
            ctx,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn device_ptr(&self) -> *mut T {
        self.block.device_ptr() as *mut T
    }

    /// 哨兵①:出生令牌(捕获期依赖记录;池块的 OpaqueDevBuf 面)
    pub fn token(&self) -> Option<BufToken> {
        Some(owl_iface::OpaqueDevBuf::token(&self.block))
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
        broadcast_check(&self.shape, shape)?;
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
    pub fn to_dtype<U: crate::dtype::Scalar>(&self) -> Result<TypedTensor<U, D>, BackendError> {
        Err(BackendError::Init(
            "to_dtype: 跨 dtype 转换 kernel 未回填(T1 签名层)".into(),
        ))
    }

    /// 浅拷贝(共享同一底层存储;视图算子的实现基元。
    /// 活性由存储 Arc/租约兜底,克隆不触设备)
    fn clone_shallow(&self) -> Self {
        Self {
            block: self.block.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            ctx: self.ctx.clone(),
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Scalar, D: Device> DevBuf<T> for TypedTensor<T, D> {
    fn len(&self) -> usize {
        Self::elems(&self.shape)
    }
    fn device_ptr(&self) -> *mut T {
        self.block.device_ptr() as *mut T
    }
}

// ---- OwlCuda 专属:EagerOnly 的 D2H 回读 + GraphLease 租约登记 ----

impl<T: Scalar> TypedTensor<T, CudaDevice> {
    /// 保活池块克隆(捕获会话 lease_block 用;字节面唯一形态)
    pub fn block(&self) -> owl_cuda::CudaPoolBuf {
        owl_cuda::CudaPoolBuf::clone(&self.block)
    }

    /// **EagerOnly**:D2H 回读(同步;禁入捕获段)。
    pub fn to_vec(&self) -> Result<Vec<T>, BackendError> {
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
                host.as_mut_ptr() as *mut core::ffi::c_void,
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
impl<T: Scalar> crate::EagerOnly for TypedTensor<T, CudaDevice> {
    const EAGER_ONLY: bool = true;
}

#[cfg(test)]
mod tests {
    use super::*;
    

    fn dev() -> CudaDevice {
        CudaDevice::new(owl_cuda::test_device_ordinal(), owl_cuda::TEST_POOL_BYTES).expect("需要 CUDA 设备")
    }

    fn scratch_pool(d: &CudaDevice, _bytes: u64) -> std::sync::Arc<owl_cuda::CudaPool> {
        d.default_pool()
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
        let pool = d.default_pool();
        let t = pool.scratch_tensor::<f32>(&[16, 16]).unwrap();
        assert_eq!(t.shape(), &[16, 16]);
        assert_eq!(DevBuf::<f32>::len(&t), 256);
        assert!(!t.device_ptr().is_null());
    }

    /// 合并 Tensor:池工厂(dtype 标签)→ to_vec 往返 + S1 结构化报错
    #[test]
    fn merged_tensor_zeros_and_to_vec_roundtrip() {
        let d = dev();
        let pool = scratch_pool(&d, 64 * 1024);
        let t = pool.zeros(Dtype::F32, &[8, 4]).unwrap();
        assert_eq!(t.dtype(), Dtype::F32);
        assert_eq!(t.len_bytes(), 8 * 4 * 4);
        assert_eq!(t.to_vec::<f32>().unwrap(), vec![0f32; 32]);
        // S1:to_vec::<u32> 撞 F32 张量 = 结构化报错
        assert!(t.to_vec::<u32>().is_err());
    }

    /// 合并 Tensor:f32_ptr_at 收口 + f32_at 单元素读(P0-5)
    #[test]
    fn merged_tensor_f32_accessors() {
        let d = dev();
        let pool = scratch_pool(&d, 64 * 1024);
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let bytes: Vec<u8> = vals.iter().flat_map(|f| f.to_le_bytes()).collect();
        let t = pool.from_host_bytes(Dtype::F32, &[4, 4], &bytes).unwrap();
        unsafe {
            assert_eq!((*t.f32_ptr().unwrap()), 0.0);
            assert_eq!((*t.f32_ptr_at(5).unwrap()), 5.0);
        }
        assert_eq!(t.f32_at(7).unwrap(), 7.0);
        // 越界 = 结构化报错
        assert!(t.f32_ptr_at(16).is_err());
        // dtype≠F32 = 结构化报错
        let tu = pool.zeros(Dtype::U32, &[4]).unwrap();
        assert!(tu.f32_ptr().is_err());
        assert!(tu.f32_at(0).is_err());
    }

    /// 合并 Tensor:from_bytes 容量守卫(需 > 块 = 结构化报错)
    #[test]
    fn merged_tensor_from_bytes_capacity_guard() {
        let d = dev();
        let pool = scratch_pool(&d, 4 * 1024);
        let block = pool.zeros(Dtype::U8, &[16]).unwrap().lease_block().unwrap();
        let r = Tensor::<CudaDevice>::from_bytes(Dtype::F32, &[1024], block);
        assert!(r.is_err());
    }

    /// 合并 Tensor:块租约出口(lease_block)+ 视图克隆共享保活
    #[test]
    fn merged_tensor_lease_block_and_clone() {
        let d = dev();
        let pool = scratch_pool(&d, 64 * 1024);
        let t = pool.zeros(Dtype::F32, &[32]).unwrap();
        let blk = t.lease_block().expect("池块必有");
        let c = t.clone();
        assert_eq!(c.len_bytes(), t.len_bytes());
        assert_eq!(c.token(), t.token());
        drop(blk); // 块租约独立计数:张量仍活
        assert_eq!(c.to_vec::<f32>().unwrap(), vec![0f32; 32]);
    }
}

// ---- 池直连张量创建(裁决:分配入口在 Pool,Device 不出现在调用点)----

/// 池对象的张量工厂扩展。约束 `P::Dev::Pool = P`(池与设备的闭环),
/// 由 blanket impl 的 where 子句表达;调用点形如:
///
/// ```ignore
/// let w = pool.zeros_tensor::<f32>(&[M, K])?;      // 持久域(强类型,过渡)
/// let z = pool.zeros(Dtype::F32, &[M, K])?;        // 持久域(dtype 标签,合并)
/// let s = pool.scratch(Dtype::F32, &[M, N])?;      // 暂存域
/// let w = pool.from_host_bytes(Dtype::F32, &shape, bytes)?; // host 装载
/// ```
pub trait TensorPoolOps: Pool {
    /// P 阶段:池内持久分配(清零)。Weights/KvCache/Workspace 池语义。
    fn zeros_tensor<T: Scalar>(
        &self,
        shape: &[usize],
    ) -> Result<TypedTensor<T, Self::Dev>, BackendError>;

    /// P 阶段:常量填充张量(复用 from_vec_tensor;对应 candle `full`)
    fn full_tensor<T: Scalar + Copy>(
        &self,
        shape: &[usize],
        fill: T,
    ) -> Result<TypedTensor<T, Self::Dev>, BackendError> {
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
    ) -> Result<TypedTensor<T, Self::Dev>, BackendError> {
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
    ) -> Result<TypedTensor<T, Self::Dev>, BackendError>;

    /// P 阶段:host 装载(权重初始化)。shape 元素数须与 src 一致。
    fn from_vec_tensor<T: Scalar>(
        &self,
        shape: &[usize],
        src: Vec<T>,
    ) -> Result<TypedTensor<T, Self::Dev>, BackendError>;

    // ---- dtype 标签工厂(合并 Tensor;iface 字节化面的直接消费)----

    /// P 阶段:dtype 标签持久分配(清零)。合并 Tensor 的正规工厂。
    fn zeros(&self, dtype: Dtype, shape: &[usize]) -> Result<Tensor<Self::Dev>, BackendError>
    where
        Self::Dev: Device<Pool = Self>,
    {
        let n: usize = shape.iter().product();
        let block = self.malloc((n * dtype.size_bytes()) as u64)?;
        Tensor::from_bytes(dtype, shape, block)
    }

    /// P 阶段:dtype 标签暂存分配(清零;捕获期可创建)。
    fn scratch(&self, dtype: Dtype, shape: &[usize]) -> Result<Tensor<Self::Dev>, BackendError>
    where
        Self::Dev: Device<Pool = Self>,
    {
        let n: usize = shape.iter().product();
        let block = self.malloc((n * dtype.size_bytes()) as u64)?;
        Tensor::from_bytes(dtype, shape, block)
    }

    /// P 阶段:host 字节流装载(H2D;src.len() 须 = 元素数 × dtype 宽度)。
    fn from_host_bytes(
        &self,
        dtype: Dtype,
        shape: &[usize],
        src: &[u8],
    ) -> Result<Tensor<Self::Dev>, BackendError>
    where
        Self::Dev: Device<Pool = Self>,
    {
        let n: usize = shape.iter().product();
        let want = n * dtype.size_bytes();
        if src.len() != want {
            return Err(BackendError::CopyFailed {
                dir: "htod",
                detail: format!("shape 元素字节数 {want} != src.len() {}", src.len()),
            });
        }
        let block = self.htod(src)?;
        Tensor::from_bytes(dtype, shape, block)
    }
}

/// host 侧可迭代算术(T1 工厂专用:arange_tensor 在 host 算好 vec 再装载;
/// Bf16/F16 无算术语义(S5),不实现本 trait)
pub trait HostArith: Scalar + Copy + core::ops::Add<Output = Self> {}
impl HostArith for f32 {}
impl HostArith for u8 {}
impl HostArith for u32 {}
impl HostArith for i64 {}

impl<P, D> TensorPoolOps for P
where
    P: Pool<Dev = D>,
    D: Device<Pool = P>,
{
    fn zeros_tensor<T: Scalar>(&self, shape: &[usize]) -> Result<TypedTensor<T, D>, BackendError> {
        let n: usize = shape.iter().product();
        let block = self.malloc((n * std::mem::size_of::<T>()) as u64)?;
        Ok(TypedTensor::from_block(block, shape, None))
    }

    fn scratch_tensor<T: Scalar>(&self, shape: &[usize]) -> Result<TypedTensor<T, D>, BackendError> {
        // 池去类型后与 zeros 同路径(清零字节块);保留签名供调用面兼容
        self.zeros_tensor(shape)
    }

    fn from_vec_tensor<T: Scalar>(&self, shape: &[usize], src: Vec<T>) -> Result<TypedTensor<T, D>, BackendError> {
        let n: usize = shape.iter().product();
        if src.len() != n {
            return Err(BackendError::CopyFailed {
                dir: "htod",
                detail: format!("shape 元素数 {n} != src.len() {}", src.len()),
            });
        }
        let bytes = n * std::mem::size_of::<T>();
        // MemValue = POD,host 字节视图搬运(无别名风险)
        let host = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, bytes) };
        let block = self.htod(host)?;
        Ok(TypedTensor::from_block(block, shape, None))
    }
}
