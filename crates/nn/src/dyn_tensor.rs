//! DynTensor:dtype 擦除的张量句柄(T2-三 设计,主 agent 裁决)。
//!
//! 动机:KV cache / sampler 层持有 dtype 运行时可变的缓冲
//! (xinfer `Vec<(Tensor, Tensor)>` 擦除形态,35+ 处触点),而
//! [`crate::Tensor`] 是 `Tensor<T, D>` 强类型。本类型 = 同型
//! `MemValue` 面上的擦除视图:ptr/shape/dtype/token 保留,算术
//! 面不保留——需要计算时经 `downcast` 回强类型(运行期 Dtype
//! 检查,不匹配 = 结构化报错,S1 禁隐式提升在擦除层的对应面)。
//!
//! 搬运规则:candle `Tensor`(擦除形态)字段/参数 → 本类型;
//! candle 对擦除张量的方法调用 → 搬运点先 downcast 再调 owl 强
//! 类型算子(调用点显式,禁止在 DynTensor 上偷偷开算术面)。

use crate::dtype::Dtype;
use crate::tensor::Tensor;
use owl_iface::{BackendError, BufToken, Device, DevBuf, MemValue};



/// dtype 擦除的张量句柄(同设备内;Clone = 租约克隆语义,与 Tensor 一致)。
#[derive(Clone)]
pub struct DynTensor<D: Device> {
    ptr: *mut u8,
    shape: Vec<usize>,
    dtype: Dtype,
    /// 按字节计的元素数缓存(shape 乘积 × size_bytes)
    len_bytes: usize,
    token: Option<BufToken>,
    /// 保活底层存储(Arc 计数 = 租约计数;drop 链与强类型一致)
    _keepalive: DynKeepalive<D>,
}

enum DynKeepalive<D: Device> {
    /// 视图构造(from_raw):无保活——调用方保证缓冲存活
    /// (bindings 由 GraphPlan 持有,生命周期覆盖图)。
    Raw,
    F32(Tensor<f32, D>),
    F16(Tensor<crate::dtype::F16, D>),
    Bf16(Tensor<crate::dtype::Bf16, D>),
    U8(Tensor<u8, D>),
    U32(Tensor<u32, D>),
    I64(Tensor<i64, D>),
}

macro_rules! erase_ctor {
    ($fn_name:ident, $t:ty, $var:ident, $dt:expr) => {
        /// 从强类型张量擦除(保活克隆随行 = 租约计数 +1,活性语义不变)
        pub fn $fn_name(t: &Tensor<$t, D>) -> Self {
            let cloned = t.clone(); // 租约克隆
            Self {
                ptr: cloned.device_ptr() as *mut u8,
                shape: cloned.shape().to_vec(),
                dtype: $dt,
                len_bytes: cloned.len() * std::mem::size_of::<$t>(),
                token: cloned.token(),
                _keepalive: DynKeepalive::$var(cloned),
            }
        }
    };
}


/// typed_getter:拥有克隆下行转换(六型封闭;S1 结构化报错)
macro_rules! typed_getter {

    ($fn_name:ident, $t:ty, $var:ident, $dt:expr, $name:expr) => {
        impl<D: Device> DynTensor<D> {
            pub fn $fn_name(&self) -> Result<Tensor<$t, D>, BackendError> {
                if self.dtype != $dt {
                    return Err(BackendError::Init(format!(
                        "DynTensor::{}: 擦除 dtype {} != 请求 {}(S1 禁隐式提升)",
                        $name, self.dtype, $dt
                    )));
                }
                match &self._keepalive {
                    DynKeepalive::$var(t) => Ok(t.clone()),
                    _ => unreachable!("dtype 标注与 keepalive 变体不一致"),
                }
            }
        }
    };
}

typed_getter!(typed_f32, f32, F32, Dtype::F32, "typed_f32");
typed_getter!(typed_f16, crate::dtype::F16, F16, Dtype::F16, "typed_f16");
typed_getter!(typed_bf16, crate::dtype::Bf16, Bf16, Dtype::BF16, "typed_bf16");
typed_getter!(typed_u8, u8, U8, Dtype::U8, "typed_u8");
typed_getter!(typed_u32, u32, U32, Dtype::U32, "typed_u32");
typed_getter!(typed_i64, i64, I64, Dtype::I64, "typed_i64");

impl<D: Device> DynTensor<D> {
    /// R1:裸指针视图构造(无保活;shape = 逻辑形状)。
    /// 仅 bindings 类长期缓冲使用(存活方 = GraphPlan)。
    pub fn from_raw_u32(ptr: *mut u32, shape: &[usize]) -> Self {
        Self {
            ptr: ptr as *mut u8,
            shape: shape.to_vec(),
            dtype: Dtype::U32,
            len_bytes: shape.iter().product::<usize>() * 4,
            token: None,
            _keepalive: DynKeepalive::Raw,
        }
    }
    pub fn from_raw_f32(ptr: *mut f32, shape: &[usize]) -> Self {
        Self {
            ptr: ptr as *mut u8,
            shape: shape.to_vec(),
            dtype: Dtype::F32,
            len_bytes: shape.iter().product::<usize>() * 4,
            token: None,
            _keepalive: DynKeepalive::Raw,
        }
    }
    erase_ctor!(from_f32, f32, F32, Dtype::F32);
    erase_ctor!(from_f16, crate::dtype::F16, F16, Dtype::F16);
    erase_ctor!(from_bf16, crate::dtype::Bf16, Bf16, Dtype::BF16);
    erase_ctor!(from_u8, u8, U8, Dtype::U8);
    erase_ctor!(from_u32, u32, U32, Dtype::U32);
    erase_ctor!(from_i64, i64, I64, Dtype::I64);

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

    /// 裸设备指针(擦除层只许传地址给 kernel/记录,不许解引用算术)
    pub fn device_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// 下行转换:擦除 → 强类型(dtype 不匹配 = 结构化报错)
    pub fn downcast<T: crate::dtype::Scalar>(&self) -> Result<TensorRef<'_, T, D>, BackendError> {
        if self.dtype != T::DTYPE {
            return Err(BackendError::Init(format!(
                "downcast: 擦除 dtype {} != 请求 {}(S1 禁隐式提升)",
                self.dtype,
                T::DTYPE
            )));
        }
        // 借用视图;活性由 keepalive 兜底
        Ok(TensorRef {
            ptr: self.ptr as *mut T,
            shape: &self.shape,
            _marker: std::marker::PhantomData,
        })
    }


    /// 重塑(元数据;元素数不变,contiguous 保持——真实现)。
    /// 形参多态(S6 搬运友好):切片/Vec/usize/元组(1-5 元)均可。
    pub fn reshape(&self, shape: impl ShapeLike) -> Result<Self, BackendError> {
        let shape = shape.to_shape_vec();
        let n: usize = shape.iter().product();
        let elems: usize = self.shape.iter().product();
        if n != elems {
            let bt = std::backtrace::Backtrace::force_capture();
            eprintln!("[RDBG] DynTensor::reshape 元素数 {n} != {elems} shape={:?}\n{bt}", self.shape);
            return Err(BackendError::Init(format!(
                "DynTensor::reshape: 元素数 {n} != {elems}"
            )));
        }
        let mut out = self.clone();
        out.shape = shape;
        Ok(out)
    }

    /// 摊平(reshape 特例;真实现)
    pub fn flatten_all(&self) -> Result<Self, BackendError> {
        let n: usize = self.shape.iter().product();
        self.reshape(&[n])
    }

    /// 第 0 维窄切(真实现:基址偏移 + keepalive 克隆;连续块子集)。
    /// 非 0 维窄切需 stride 语义(S3),当前不支持 = 结构化报错。
    pub fn narrow_dim0(&self, start: usize, len: usize) -> Result<Self, BackendError> {
        if len == 0 || start + len > self.shape[0] {
            return Err(BackendError::Init(format!(
                "DynTensor::narrow_dim0: start={start} len={len} shape={:?}",
                self.shape
            )));
        }
        let mut out = self.clone();
        let row: usize = self.shape[1..].iter().product::<usize>().max(1) * self.dtype.size_bytes();
        out.ptr = unsafe { self.ptr.add(start * row) };
        out.shape[0] = len;
        Ok(out)
    }
}

/// 下行转换的只读视图(不拥有;活性由源 DynTensor 的 keepalive 兜底)
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

// 手写 Clone:keepalive 克隆 = Arc/租约计数 +1(与 Tensor 语义一致)
impl<D: Device> Clone for DynKeepalive<D> {
    fn clone(&self) -> Self {
        match self {
            DynKeepalive::Raw => DynKeepalive::Raw,
            DynKeepalive::F32(t) => DynKeepalive::F32(t.clone()),
            DynKeepalive::F16(t) => DynKeepalive::F16(t.clone()),
            DynKeepalive::Bf16(t) => DynKeepalive::Bf16(t.clone()),
            DynKeepalive::U8(t) => DynKeepalive::U8(t.clone()),
            DynKeepalive::U32(t) => DynKeepalive::U32(t.clone()),
            DynKeepalive::I64(t) => DynKeepalive::I64(t.clone()),
        }
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
