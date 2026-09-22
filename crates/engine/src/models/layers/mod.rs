//! models 层公共桩 + 张量算子扩展(T2-三 搬运地基,主 agent 设计)。
//!
//! 移植策略(编译先行口径):
//! - candle `Tensor`(dtype 擦除)→ `owl_nn::DynTensor<CudaDevice>`(别名 [`Tensor`]);
//! - candle 张量方法面(matmul/narrow/reshape/...)→ [`OwlTensor`] 扩展 trait,
//!   方法体 `unimplemented!("T3 kernel 回填")`——调用点语法保持不变,
//!   类型流(控制流骨架)完整保留;
//! - candle_nn 容器(VarBuilder/Linear/Conv/Embedding)→ 本模块桩结构,
//!   权重装载路径(T3 权重通道)回填前为 unimplemented;
//! - vendor attention_rs 的 kernel/缓存面(PagedAttention/MambaCache/moe/
//!   gdn/...)→ 占位模块,归宿裁决待定(design: vendor 三选一 = 自研 kernel
//!   进 owl-kernels / 保留依赖 / 换 flashinfer)。
//!
//! 纪律:本文件只做"类型面 + 控制流骨架",不引入任何真实 CUDA 语义;
//! 所有设备体在运行里程碑(T3)经 owl-kernels / marlin-ffi 回填。

#![allow(unused_variables)]

pub mod deepstack;
pub mod mask;
pub mod distributed;
pub mod linear;
pub mod mlp;
pub mod others;
pub mod rotary_emb;
pub mod wna16;

/// 权重名映射(GGUF 键名 vs HF 键名;= xinfer collect_key_map 直译)
pub fn collect_key_map<'a, const N: usize>(
    is_qvar_builder: bool,
    pairs: [(&'a str, &'a str); N],
) -> std::collections::HashMap<&'a str, &'a str> {
    if is_qvar_builder {
        pairs.into_iter().collect()
    } else {
        pairs.into_iter().map(|(key, _)| (key, key)).collect()
    }
}
pub mod attention;
pub mod deltanet;
pub mod moe;
//pub mod deepstack;
//pub mod deltanet;
//pub mod distributed;
//pub mod linear;
//pub mod mask;
//pub mod mlp;
//pub mod moe;
//pub mod others;
//pub mod rotary_emb;
//pub mod wna16;

use crate::error::{Error, Result};
use owl_iface::BackendError;
use owl_cuda::CudaDevice;
use owl_nn::DynTensor;
use owl_nn::erased;

pub mod ctx_scope;

// ============================================================================
// 基础类型别名(对应 candle_core 词汇面)
// ============================================================================

/// dtype 擦除张量(= candle `Tensor` 的 owl 对应物)。
pub type Tensor = DynTensor<CudaDevice>;

/// 设备句柄(= candle `Device` 的 owl 对应物;cuda-only 基座)。
pub type Device = CudaDevice;

/// dtype(= candle `DType`)。
pub type DType = owl_nn::Dtype;

/// 形状(= candle `Shape`;接受 &[usize]/Vec/元组的 Into 面)。
#[derive(Debug, Clone, Default)]
pub struct Shape(pub Vec<usize>);

impl From<&[usize]> for Shape {
    fn from(s: &[usize]) -> Self {
        Shape(s.to_vec())
    }
}
impl From<&Vec<usize>> for Shape {
    fn from(s: &Vec<usize>) -> Self {
        Shape(s.clone())
    }
}
impl From<Vec<usize>> for Shape {
    fn from(s: Vec<usize>) -> Self {
        Shape(s)
    }
}
impl From<usize> for Shape {
    fn from(s: usize) -> Self {
        Shape(vec![s])
    }
}
impl<const N: usize> From<[usize; N]> for Shape {
    fn from(s: [usize; N]) -> Self {
        Shape(s.to_vec())
    }
}

// candle 风格的元组形状 (a,) / (a, b) / ... 直译(显式 impl;宏在嵌套
// 字段访问上易踩 metavariable 解析坑,不用)
impl From<(usize,)> for Shape {
    fn from(s: (usize,)) -> Self {
        Shape(vec![s.0])
    }
}
impl From<(usize, usize)> for Shape {
    fn from(s: (usize, usize)) -> Self {
        Shape(vec![s.0, s.1])
    }
}
impl From<(usize, usize, usize)> for Shape {
    fn from(s: (usize, usize, usize)) -> Self {
        Shape(vec![s.0, s.1, s.2])
    }
}
impl From<(usize, usize, usize, usize)> for Shape {
    fn from(s: (usize, usize, usize, usize)) -> Self {
        Shape(vec![s.0, s.1, s.2, s.3])
    }
}
impl From<(usize, usize, usize, usize, usize)> for Shape {
    fn from(s: (usize, usize, usize, usize, usize)) -> Self {
        Shape(vec![s.0, s.1, s.2, s.3, s.4])
    }
}

impl Shape {
    pub fn ndim(&self) -> usize {
        self.0.len()
    }
    pub fn dim(&self, i: usize) -> Result<usize, crate::error::Error> {
        self.0
            .get(i)
            .copied()
            .ok_or_else(|| crate::error::Error::Msg(format!("dim {i} out of bounds")))
    }
    pub fn dims(&self) -> &[usize] {
        &self.0
    }
}

/// 负维枚举(= candle `D`;配合 [`Dim`] 用于 narrow/sum/... 的 dim 参数)。
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum D {
    Minus1,
    Minus2,
    Minus3,
}

/// 维度索引约束(= candle `Dim`;usize 与 D 均可作 dim 参数)。
pub trait Dim: Copy + Sized {
    fn resolve_dim(self, rank: usize) -> std::result::Result<usize, String>;
}
impl Dim for usize {
    fn resolve_dim(self, rank: usize) -> std::result::Result<usize, String> {
        if self < rank { Ok(self) } else { Err(format!("dim {self} 越界(rank={rank})")) }
    }
}
impl Dim for D {
    fn resolve_dim(self, rank: usize) -> std::result::Result<usize, String> {
        let off = match self {
            D::Minus1 => 1,
            D::Minus2 => 2,
            D::Minus3 => 3,
        };
        if rank >= off { Ok(rank - off) } else { Err(format!("D::{self:?} 越界(rank={rank})")) }
    }
}

// ============================================================================
// OwlTensor:张量算子扩展(方法体 T3 回填;调用点语法与 candle 一致)
// ============================================================================

/// 张量算子面。每个方法返回 `Result<Tensor>`;体为 `unimplemented!`
/// (T3 kernel 回填)。签名按 candle 对应方法取最常用形态。
pub trait OwlTensor: Sized {
    // ---- 形状/视图 ----
    fn reshape(&self, shape: impl Into<Shape>) -> Result<Tensor>;
    fn narrow(&self, dim: impl Dim, start: usize, len: usize) -> Result<Tensor>;
    fn transpose(&self, dim1: impl Dim, dim2: impl Dim) -> Result<Tensor>;
    fn t(&self) -> Result<Tensor>;
    fn t2(&self) -> Result<Tensor>;
    fn unsqueeze(&self, dim: impl Dim) -> Result<Tensor>;
    fn squeeze(&self, dim: impl Dim) -> Result<Tensor>;
    fn contiguous(&self) -> Result<Tensor>;
    fn broadcast_as(&self, shape: impl Into<Shape>) -> Result<Tensor>;
    fn chunk(&self, c: usize, dim: impl Dim) -> Result<Vec<Tensor>>;
    fn split(&self, s: usize, dim: impl Dim) -> Result<Vec<Tensor>>;
    fn select(&self, dim: impl Dim, i: usize) -> Result<Tensor>;

    // ---- 算术(broadcast,S2 右对齐)----
    fn add(&self, rhs: &Tensor) -> Result<Tensor>;
    fn sub(&self, rhs: &Tensor) -> Result<Tensor>;
    fn mul(&self, rhs: &Tensor) -> Result<Tensor>;
    fn div(&self, rhs: &Tensor) -> Result<Tensor>;
    fn broadcast_add(&self, rhs: &Tensor) -> Result<Tensor>;
    fn broadcast_mul(&self, rhs: &Tensor) -> Result<Tensor>;
    fn broadcast_div(&self, rhs: &Tensor) -> Result<Tensor>;
    fn broadcast_sub(&self, rhs: &Tensor) -> Result<Tensor>;
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor>;
    fn affine(&self, a: f64, b: f64) -> Result<Tensor>;
    fn where_cond(&self, t: &Tensor, f: &Tensor) -> Result<Tensor>;

    // ---- 归约 ----
    fn sum(&self, dim: impl Dim) -> Result<Tensor>;
    fn sum_all(&self) -> Result<Tensor>;
    fn max(&self, dim: impl Dim) -> Result<Tensor>;
    fn min(&self, dim: impl Dim) -> Result<Tensor>;
    fn argmax(&self, dim: impl Dim) -> Result<Tensor>;
    fn mean(&self, dim: impl Dim) -> Result<Tensor>;
    fn cumsum(&self, dim: impl Dim) -> Result<Tensor>;

    // ---- 逐元素 ----
    fn exp(&self) -> Result<Tensor>;
    fn sqrt(&self) -> Result<Tensor>;
    fn sin(&self) -> Result<Tensor>;
    fn cos(&self) -> Result<Tensor>;
    fn tanh(&self) -> Result<Tensor>;
    fn abs(&self) -> Result<Tensor>;
    fn neg(&self) -> Result<Tensor>;
    fn sqr(&self) -> Result<Tensor>;
    fn log(&self) -> Result<Tensor>;
    fn clamp(&self, min: f64, max: f64) -> Result<Tensor>;
    fn powf(&self, exp: f64) -> Result<Tensor>;
    fn reciprocal(&self) -> Result<Tensor>;
    fn gelu(&self) -> Result<Tensor>;
    fn silu(&self) -> Result<Tensor>;
    fn softmax(&self, dim: impl Dim) -> Result<Tensor>;
    fn softmax_last_dim(&self) -> Result<Tensor>;

    // ---- 索引/类型 ----
    fn index_select(&self, dim: impl Dim, idx: &Tensor) -> Result<Tensor>;
    fn gather(&self, dim: impl Dim, idx: &Tensor) -> Result<Tensor>;
    fn scatter(&self, dim: impl Dim, idx: &Tensor, src: &Tensor) -> Result<Tensor>;
    fn scatter_add(&self, idx: &Tensor, src: &Tensor, dim: impl Dim) -> Result<Tensor>;
    fn repeat(&self, shape: impl Into<Shape>) -> Result<Tensor>;
    fn repeat_interleave(&self, repeats: usize, dim: impl Dim) -> Result<Tensor>;
    fn to_dtype(&self, dtype: DType) -> Result<Tensor>;
    fn to_device(&self, _device: &Device) -> Result<Tensor>;
    // dtype() 由 DynTensor 固有方法提供,不在 trait 里重定义(避免递归)

    // ---- 元数据(编译先行:桩;T3 经 DynTensor 真值回填)----
    fn device(&self) -> Device;
    fn dim(&self, i: usize) -> Result<usize>;
    fn dims(&self) -> Result<Vec<usize>>;
    fn dims2(&self) -> Result<(usize, usize)>;
    fn flatten_all(&self) -> Result<Tensor>;
    fn to_scalar<T: Copy>(&self) -> Result<T>;

    // ---- rope(层内高频)----
    fn apply_rotary_emb_qkv(
        &self,
        cos: &Tensor,
        sin: &Tensor,
        n_rotations: usize,
    ) -> Result<Tensor>;

    // ---- 其余层件调用面(T2-三 波次补齐)----
    fn broadcast_left(&self, left: impl Into<Shape>) -> Result<Tensor>;
    fn is_contiguous(&self) -> bool;
    fn dims3(&self) -> Result<(usize, usize, usize)>;
    fn elem_count(&self) -> Result<usize>;
    fn permute(&self, order: &[usize]) -> Result<Tensor>;
    fn avg_pool2d_with_stride(&self, kernel: usize, stride: usize) -> Result<Tensor>;
    /// 量化张量解包(GGUF 路径;体 = marlin-ffi 路线,运行里程碑)
    fn dequantize(&self, device: &Device) -> Result<Tensor>;
}

/// T3 回填占位(编译先行口径;运行里程碑经 owl-kernels/marlin-ffi 替换为真实 kernel)
fn t3_unimpl(_name: &str) -> ! {
    unimplemented!("T3 kernel 回填")
}

impl OwlTensor for Tensor {
    // S6 垫片:candle 形态签名 → ctx_scope TLS 消费 (OpsCtx, NnBlas, DryKernels, ctx)
    // → owl_nn::erased 真实实现(f32)/ dry-kernels naive 核;元数据类不触设备。

    fn reshape(&self, shape: impl Into<Shape>) -> Result<Tensor> {
        let target = shape.into();
        let n: usize = target.dims().iter().product();
        let have: usize = self.shape().iter().product();
        if n != have {
            let bt = std::backtrace::Backtrace::force_capture();
        }
        Ok(DynTensor::reshape(self, &target.dims()[..])?)
    }
    fn narrow(&self, dim: impl Dim, start: usize, len: usize) -> Result<Tensor> {
        let d = resolve_dim(dim, self.shape().len())?;
        match d {
            0 => Ok(DynTensor::narrow_dim0(self, start, len)?),
            _ => ctx_scope::with_dry(|ctx, dry| {
                let s = self.shape();
                let outer: usize = s[..d].iter().product();
                let after: usize = s[d + 1..].iter().product();
                let src_dim = s[d] * after;
                let mut out_shape = s.to_vec();
                out_shape[d] = len;
                let out = ctx.scratch_tensor::<f32>(&out_shape)?;
                dry.narrow_strided_f32(
                    ctx.stream(),
                    self.device_ptr() as *const f32,
                    out.device_ptr(),
                    outer,
                    src_dim,
                    start * after,
                    len,
                )
                .map_err(BackendError::Init)?;
                Ok(DynTensor::from_f32(&out))
            }),
        }
    }
    fn transpose(&self, dim1: impl Dim, dim2: impl Dim) -> Result<Tensor> {
        let rank = self.shape().len();
        let (a, b) = (resolve_dim(dim1, rank)?, resolve_dim(dim2, rank)?);
        let mut s = self.shape().to_vec();
        s.swap(a, b);
        if self.shape()[a] == 1 || self.shape()[b] == 1 {
            // size-1 维换位 = 布局等价,纯 reshape
            eprintln!("[RS2] transpose have={:?} target={:?}", self.shape(), s);
            eprintln!("[RS3] squeeze have={:?} target={:?}", self.shape(), s);
        Ok(DynTensor::reshape(self, &s[..])?)
        } else if rank == 2 {
            // 2D 真转置:dry 转置核物化拷贝
            ctx_scope::with_dry(|ctx, dry| {
                let out = ctx.scratch_tensor::<f32>(&s[..])?;
                dry.transpose2d_f32(
                    ctx.stream(),
                    self.device_ptr() as *const f32,
                    out.device_ptr() as *mut f32,
                    self.shape()[0],
                    self.shape()[1],
                ).map_err(|e| crate::Error::Msg(format!("transpose2d: {e}")))?;
                Ok(DynTensor::from_f32(&out))
            })
        } else {
            Err(Error::from(BackendError::Init(format!(
                "transpose: >2D 非 size-1 换位需 stride 语义(shape={:?} a={a} b={b})",
                self.shape()
            ))))
        }
    }
    fn t(&self) -> Result<Tensor> { self.t2() }
    fn t2(&self) -> Result<Tensor> {
        let rank = self.shape().len();
        self.transpose(rank - 2, rank - 1)
    }
    fn unsqueeze(&self, dim: impl Dim) -> Result<Tensor> {
        let d = resolve_dim(dim, self.shape().len() + 1)?;
        let mut s = self.shape().to_vec();
        s.insert(d, 1);
        eprintln!("[RS3] squeeze have={:?} target={:?}", self.shape(), s);
        Ok(DynTensor::reshape(self, &s[..])?)
    }
    fn squeeze(&self, dim: impl Dim) -> Result<Tensor> {
        let d = resolve_dim(dim, self.shape().len())?;
        let mut s = self.shape().to_vec();
        if s[d] == 1 { s.remove(d); }
        eprintln!("[RS3] squeeze have={:?} target={:?}", self.shape(), s);
        Ok(DynTensor::reshape(self, &s[..])?)
    }
    fn contiguous(&self) -> Result<Tensor> { Ok(self.clone()) }
    fn broadcast_as(&self, _shape: impl Into<Shape>) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("broadcast_as: S2 广播 = P1(走 broadcast_* 算子)".to_string())))
    }
    fn chunk(&self, c: usize, dim: impl Dim) -> Result<Vec<Tensor>> {
        let d = resolve_dim(dim, self.shape().len())?;
        let full = self.shape()[d];
        if full % c != 0 {
            return Err(Error::from(BackendError::Init(format!("chunk: {full} 不可均分 {c}"))));
        }
        let seg = full / c;
        (0..c).map(|i| self.narrow(d, i * seg, seg)).collect()
    }
    fn split(&self, s: usize, dim: impl Dim) -> Result<Vec<Tensor>> {
        let d = resolve_dim(dim, self.shape().len())?;
        let full = self.shape()[d];
        let n = full.div_ceil(s);
        (0..n)
            .map(|i| {
                let start = i * s;
                let len = s.min(full - start);
                self.narrow(d, start, len)
            })
            .collect()
    }
    fn select(&self, dim: impl Dim, i: usize) -> Result<Tensor> {
        Ok(self.narrow(dim, i, 1)?.squeeze(dim)?)
    }

    fn add(&self, rhs: &Tensor) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| erased::add(ops, ctx, self, rhs).map_err(Into::into))
    }
    fn sub(&self, rhs: &Tensor) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| {
            let neg = scalar_dyn(ctx, -1.0)?;
            let nr = erased::broadcast_mul(ops, ctx, &neg, rhs)?;
            erased::add(ops, ctx, self, &nr).map_err(Into::into)
        })
    }
    fn mul(&self, rhs: &Tensor) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| erased::mul(ops, ctx, self, rhs).map_err(Into::into))
    }
    fn div(&self, rhs: &Tensor) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| {
            let inv = recip_dyn(ctx, rhs)?;
            erased::mul(ops, ctx, self, &inv).map_err(Into::into)
        })
    }
    fn broadcast_add(&self, rhs: &Tensor) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| erased::broadcast_add(ops, ctx, self, rhs).map_err(Into::into))
    }
    fn broadcast_mul(&self, rhs: &Tensor) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| erased::broadcast_mul(ops, ctx, self, rhs).map_err(Into::into))
    }
    fn broadcast_div(&self, rhs: &Tensor) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| {
            let inv = recip_dyn(ctx, rhs)?;
            erased::broadcast_mul(ops, ctx, self, &inv).map_err(Into::into)
        })
    }
    fn broadcast_sub(&self, rhs: &Tensor) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| {
            let neg = scalar_dyn(ctx, -1.0)?;
            let nr = erased::broadcast_mul(ops, ctx, &neg, rhs)?;
            erased::add(ops, ctx, self, &nr).map_err(Into::into)
        })
    }
    fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        ctx_scope::with_blas(|ops, ctx, blas| {
            erased::matmul(ops, ctx, blas, self, rhs).map_err(Into::into)
        })
    }
    fn affine(&self, a: f64, b: f64) -> Result<Tensor> {
        // dry-run 组合:mul 常量 + add 常量(标量经 [1] 张量广播)
        ctx_scope::with(|ops, ctx| {
            let const_mul = |ctx: &owl_nn::KernelCtx, v: f32| -> Result<Tensor> {
                let t = ctx.scratch_tensor::<f32>(&[1])?;
                owl_nn::erased::copy_d2d_to_raw(
                    ctx,
                    &owl_nn::DynTensor::from_f32(&owl_nn::TensorPoolOps::from_vec_tensor(
                        crate::models::layers::ctx_scope::weights_pool().as_ref(),
                        &[1],
                        vec![v],
                    )?),
                    owl_iface::DevBuf::device_ptr(&t) as *mut core::ffi::c_void,
                    4,
                )?;
                Ok(owl_nn::DynTensor::from_f32(&t))
            };
            let mul_t = const_mul(ctx, a as f32)?;
            let scaled = erased::broadcast_mul(ops, ctx, self, &mul_t)?;
            let add_t = const_mul(ctx, b as f32)?;
            erased::broadcast_add(ops, ctx, &scaled, &add_t).map_err(Into::into)
        })
    }
    fn where_cond(&self, _t: &Tensor, _f: &Tensor) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("where_cond: P1".to_string())))
    }

    fn sum(&self, dim: impl Dim) -> Result<Tensor> {
        let d = resolve_dim(dim, self.shape().len())?;
        ctx_scope::with(|ops, ctx| erased::sum_dim(ops, ctx, self, d).map_err(Into::into))
    }
    fn sum_all(&self) -> Result<Tensor> {
        eprintln!("[RS4] sum_all have={:?} target={}", self.shape(), self.len_bytes() / 4);
        let flat = DynTensor::reshape(self, &[self.len_bytes() / 4])?;
        ctx_scope::with(|ops, ctx| erased::sum_dim(ops, ctx, &flat, 0).map_err(Into::into))
    }
    fn max(&self, dim: impl Dim) -> Result<Tensor> {
        let d = resolve_dim(dim, self.shape().len())?;
        ctx_scope::with(|ops, ctx| erased::max_dim(ops, ctx, self, d).map_err(Into::into))
    }
    fn min(&self, _dim: impl Dim) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("min: P1".to_string())))
    }
    fn argmax(&self, _dim: impl Dim) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("argmax: P1(采样链 radix 接管)".to_string())))
    }
    fn mean(&self, dim: impl Dim) -> Result<Tensor> {
        let d = resolve_dim(dim, self.shape().len())?;
        let n = self.shape()[d] as f32;
        ctx_scope::with(|ops, ctx| {
            let s: DynTensor<owl_cuda::CudaDevice> = erased::sum_dim(ops, ctx, self, d).map_err(|e| crate::error::Error::from(e))?;
            let inv = scalar_dyn(ctx, 1.0 / n)?;
            erased::broadcast_mul(ops, ctx, &s, &inv).map_err(Into::into)
        })
    }
    fn cumsum(&self, _dim: impl Dim) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("cumsum: P1".to_string())))
    }

    fn exp(&self) -> Result<Tensor> { Err(Error::from(BackendError::Init("exp: P1".to_string()))) }
    fn sqrt(&self) -> Result<Tensor> { Err(Error::from(BackendError::Init("sqrt: P1".to_string()))) }
    fn sin(&self) -> Result<Tensor> {
        ctx_scope::with_dry(|ctx, dry| {
            let n = self.len_bytes() / 4;
            let out = ctx.scratch_tensor::<f32>(self.shape())?;
            dry.sin_f32(ctx.stream(), self.device_ptr() as *const f32, out.device_ptr(), n)
                .map_err(BackendError::Init)?;
            Ok(DynTensor::from_f32(&out))
        })
    }
    fn cos(&self) -> Result<Tensor> {
        ctx_scope::with_dry(|ctx, dry| {
            let n = self.len_bytes() / 4;
            let out = ctx.scratch_tensor::<f32>(self.shape())?;
            dry.cos_f32(ctx.stream(), self.device_ptr() as *const f32, out.device_ptr(), n)
                .map_err(BackendError::Init)?;
            Ok(DynTensor::from_f32(&out))
        })
    }
    fn tanh(&self) -> Result<Tensor> { Err(Error::from(BackendError::Init("tanh: P1(softcap 未用)".to_string()))) }
    fn abs(&self) -> Result<Tensor> { Err(Error::from(BackendError::Init("abs: P1".to_string()))) }
    fn neg(&self) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| {
            let neg = scalar_dyn(ctx, -1.0)?;
            erased::broadcast_mul(ops, ctx, &neg, self).map_err(Into::into)
        })
    }
    fn sqr(&self) -> Result<Tensor> { self.mul(self) }
    fn log(&self) -> Result<Tensor> { Err(Error::from(BackendError::Init("log: P1".to_string()))) }
    fn clamp(&self, _min: f64, _max: f64) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("clamp: P1".to_string())))
    }
    fn powf(&self, _exp: f64) -> Result<Tensor> { Err(Error::from(BackendError::Init("powf: P1".to_string()))) }
    fn reciprocal(&self) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| recip_dyn(ctx, self).map_err(Into::into))
    }
    fn gelu(&self) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("gelu: P1(hidden_act=Silu)".to_string())))
    }
    fn silu(&self) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| erased::silu(ops, ctx, self).map_err(Into::into))
    }
    fn softmax(&self, dim: impl Dim) -> Result<Tensor> {
        let d = resolve_dim(dim, self.shape().len())?;
        if d + 1 != self.shape().len() {
            return Err(Error::from(BackendError::Init("softmax: 一期 last 维".to_string())));
        }
        self.softmax_last_dim()
    }
    fn softmax_last_dim(&self) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| erased::softmax_last_dim(ops, ctx, self).map_err(Into::into))
    }

    fn index_select(&self, dim: impl Dim, idx: &Tensor) -> Result<Tensor> {
        let d = resolve_dim(dim, self.shape().len())?;
        ctx_scope::with(|ops, ctx| erased::index_select(ops, ctx, self, d, idx).map_err(Into::into))
    }
    fn gather(&self, dim: impl Dim, idx: &Tensor) -> Result<Tensor> {
        let d = resolve_dim(dim, self.shape().len())?;
        let _ = d;
        ctx_scope::with(|ops, ctx| erased::gather(ops, ctx, self, idx).map_err(Into::into))
    }
    fn scatter(&self, _dim: impl Dim, _idx: &Tensor, _src: &Tensor) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("scatter: P1".to_string())))
    }
    fn scatter_add(&self, idx: &Tensor, src: &Tensor, dim: impl Dim) -> Result<Tensor> {
        let d = resolve_dim(dim, self.shape().len())?;
        let _ = d;
        let num = self.len_bytes() / 4;
        ctx_scope::with(|ops, ctx| {
            erased::scatter_add(ops, ctx, num, src, idx).map_err(Into::into)
        })
    }
    fn repeat(&self, _shape: impl Into<Shape>) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("repeat: P1".to_string())))
    }
    fn repeat_interleave(&self, _repeats: usize, _dim: impl Dim) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("repeat_interleave: P1".to_string())))
    }
    fn to_dtype(&self, dtype: DType) -> Result<Tensor> {
        if dtype == self.dtype() {
            return Ok(self.clone());
        }
        // dry-run 数值转换:仅 U32→F32;构造期 = P 阶段,host 往返合法
        if self.dtype() == DType::U32 && dtype == DType::F32 {
            let dev = ctx_scope::with_device();
            use owl_cuda::ffi::sys;
            dev.ctx().bind_to_thread().map_err(|e| {
                Error::from(BackendError::Init(format!("bind_to_thread: {e:?}")))
            })?;
            let n = self.len_bytes() / 4;
            let mut host = vec![0u32; n];
            unsafe {
                sys::cuMemcpyDtoH_v2(
                    host.as_mut_ptr() as *mut std::ffi::c_void,
                    self.device_ptr() as sys::CUdeviceptr,
                    n * 4,
                )
                .result()
                .map_err(|e| {
                    Error::from(BackendError::CopyFailed { dir: "dtoh", detail: format!("{e:?}") })
                })?;
            }
            let conv: Vec<f32> = host.iter().map(|&b| b as f32).collect();
            let pool = ctx_scope::weights_pool();
            let t = owl_nn::TensorPoolOps::from_vec_tensor(pool.as_ref(), &[n], conv)?;
            return Ok(owl_nn::DynTensor::from_f32(&t));
        }
        Err(Error::from(BackendError::Init(format!(
            "to_dtype {}→{dtype}: 转换核未回填(dry-run 全 f32)",
            self.dtype()
        ))))
    }
    fn to_device(&self, _device: &Device) -> Result<Tensor> { Ok(self.clone()) }

    fn device(&self) -> Device {
        ctx_scope::with_device()
    }
    fn dim(&self, i: usize) -> Result<usize> {
        self.shape()
            .get(i)
            .copied()
            .ok_or_else(|| Error::from(BackendError::Init(format!("dim {i} 越界"))))
    }
    fn dims(&self) -> Result<Vec<usize>> { Ok(self.shape().to_vec()) }
    fn dims2(&self) -> Result<(usize, usize)> {
        let s = self.shape();
        match s.len() {
            2 => Ok((s[0], s[1])),
            _ => Err(Error::from(BackendError::Init(format!("dims2: rank {} ≠ 2", s.len())))),
        }
    }
    fn dims3(&self) -> Result<(usize, usize, usize)> {
        let s = self.shape();
        match s.len() {
            3 => Ok((s[0], s[1], s[2])),
            _ => Err(Error::from(BackendError::Init(format!("dims3: rank {} ≠ 3", s.len())))),
        }
    }
    fn elem_count(&self) -> Result<usize> { Ok(self.len_bytes() / 4) }
    fn flatten_all(&self) -> Result<Tensor> {
        eprintln!("[RS5] flatten_all have={:?} target={}", self.shape(), self.len_bytes() / 4);
        Ok(DynTensor::reshape(self, &[self.len_bytes() / 4])?)
    }
    fn broadcast_left(&self, _left: impl Into<Shape>) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("broadcast_left: P1".to_string())))
    }
    fn is_contiguous(&self) -> bool { true }
    fn permute(&self, order: &[usize]) -> Result<Tensor> {
        Err(Error::from(BackendError::Init(format!("permute {order:?}: P1(stride 语义)"))))
    }
    fn avg_pool2d_with_stride(&self, _k: usize, _s: usize) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("avg_pool2d: deltanet 专项".to_string())))
    }
    fn dequantize(&self, _device: &Device) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("dequantize: marlin-ffi 归属".to_string())))
    }
    fn to_scalar<T: Copy>(&self) -> Result<T> {
        Err(Error::from(BackendError::Init("to_scalar: D2H 面(采样层另行接)".to_string())))
    }

    fn apply_rotary_emb_qkv(&self, _cos: &Tensor, _sin: &Tensor, _n: usize) -> Result<Tensor> {
        Err(Error::from(BackendError::Init("apply_rotary_emb_qkv: rope 走 vendor::fused_rope".to_string())))
    }
}

/// 标量张量([1] f32;scratch + dry fill 核,捕获安全)
pub(crate) fn scalar_dyn(ctx: &owl_nn::KernelCtx, v: f32) -> Result<Tensor> {
    ctx_scope::with_dry(|ctx, dry| {
        let out = ctx.scratch_tensor::<f32>(&[1])?;
        dry.fill_f32(ctx.stream(), out.device_ptr(), v, 1).map_err(BackendError::Init)?;
        Ok(DynTensor::from_f32(&out))
    })
}

/// 逐元素倒数(dry recip 核;捕获安全)
pub(crate) fn recip_dyn(ctx: &owl_nn::KernelCtx, x: &Tensor) -> Result<Tensor> {
    ctx_scope::with_dry(|ctx, dry| {
        let n = x.len_bytes() / 4;
        let out = ctx.scratch_tensor::<f32>(x.shape())?;
        dry.recip_f32(ctx.stream(), x.device_ptr() as *const f32, out.device_ptr(), n)
            .map_err(BackendError::Init)?;
        Ok(DynTensor::from_f32(&out))
    })
}

/// 维度解析(usize 直通;D 负维回绕)
pub(crate) fn resolve_dim(d: impl Dim, rank: usize) -> Result<usize> {
    Ok(d.resolve_dim(rank).map_err(|m| Error::Msg(m))?)
}

/// 标量张量([1] f32,捕获安全:scratch + dry fill 核)


/// 逐元素倒数(dry recip 核;捕获安全)


/// 维度解析(usize 直通;D::Minus1/2/3 负维回绕)


// ============================================================================
// 自由算子(candle_nn::ops / 层内 rms_norm 等)
// ============================================================================

pub mod ops {
    use super::ctx_scope;
    use super::{DType, Result, Tensor};
    use owl_nn::erased;

    fn softmax_any_dim(x: &Tensor, dim: usize) -> Result<Tensor> {
        let rank = x.shape().len();
        let dim = if (dim as i32) < 0 { rank as i32 + dim as i32 } else { dim as i32 } as usize;
        // 末维 softmax = erased 真身;非末维 = transpose 到末维再回(元数据)
        ctx_scope::with(|ops, ctx| {
            if dim == rank - 1 {
                erased::softmax_last_dim(ops, ctx, x).map_err(Into::into)
            } else {
                let _ = x;
                unimplemented!("softmax 非末维(dry-run 走末维路径)")
            }
        })
    }

    pub fn silu(x: &Tensor) -> Result<Tensor> {
        ctx_scope::with(|ops, ctx| erased::silu(ops, ctx, x).map_err(Into::into))
    }
    pub fn gelu(_x: &Tensor) -> Result<Tensor> { unimplemented!("T3 kernel 回填: gelu") }
    pub fn softmax(x: &Tensor, dim: usize) -> Result<Tensor> { softmax_any_dim(x, dim) }
    pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> { softmax_any_dim(x, x.shape().len() - 1) }
    pub fn gelu_erf(_x: &Tensor) -> Result<Tensor> { unimplemented!("T3 kernel 回填: gelu_erf") }
    pub fn tanh(_x: &Tensor) -> Result<Tensor> { unimplemented!("T3 kernel 回填: tanh") }
    pub fn silu_and_mul(x: &Tensor) -> Result<Tensor> {
        // fused swiglu:out[r, :half] = silu(x[r, :half]) * x[r, half:](dry 核)
        ctx_scope::with_dry(|ctx, dry| {
            let rank = x.shape().len();
            let n = *x.shape().last().ok_or_else(|| crate::Error::Msg("silu_and_mul: 0 维".into()))?;
            let half = n / 2;
            let rows: usize = x.shape()[..rank - 1].iter().product();
            let out = ctx.scratch_tensor::<f32>(&[if rank == 1 { half } else { rows }, half])?;
            dry.silu_and_mul_f32(
                ctx.stream(),
                x.device_ptr() as *const f32,
                out.device_ptr() as *mut f32,
                half,
                rows,
            ).map_err(|e| crate::Error::Msg(format!("silu_and_mul: {e}")))?;
            Ok(super::DynTensor::from_f32(&out))
        })
    }
    pub fn sigmoid(_x: &Tensor) -> Result<Tensor> { unimplemented!("T3 kernel 回填: sigmoid") }
    pub fn cat(xs: &[Tensor], dim: usize) -> Result<Tensor> {
        ctx_scope::with(|_ops, ctx| erased::cat(ctx, xs, dim).map_err(Into::into))
    }
    pub fn full(_v: f64, _shape: &[usize], _like: &Tensor) -> Result<Tensor> {
        unimplemented!("T3 kernel 回填: full(dry-run 未见调用,撞上再接)")
    }
    pub fn _dt(_d: DType) {}
}

// ============================================================================
// 张量构造器(candle Tensor::new/zeros/arange/... → 池直连,T3 回填)
// ============================================================================

pub mod ctor {
    #![allow(unused_variables)]
    use super::{t3_unimpl, DType, Device, Result, Shape, Tensor};

    pub fn zeros(shape: impl Into<Shape>, _dtype: DType, _device: &Device) -> Result<Tensor> {
        t3_unimpl("zeros")
    }
    pub fn ones(shape: impl Into<Shape>, _dtype: DType, _device: &Device) -> Result<Tensor> {
        t3_unimpl("ones")
    }
    pub fn full(shape: impl Into<Shape>, _v: f64, _dtype: DType, _device: &Device) -> Result<Tensor> {
        t3_unimpl("full")
    }
    pub fn arange(start: usize, end: usize, _dtype: DType, _device: &Device) -> Result<Tensor> {
        // dry-run 装载基元:host 生成等差序列落池(u32 位型;rope 位置表用)
        let v: Vec<u32> = (start as u32..end as u32).collect();
        let shape = vec![v.len()];
        let pool = crate::models::layers::ctx_scope::weights_pool();
        let t = owl_nn::TensorPoolOps::from_vec_tensor(pool.as_ref(), &shape, v)?;
        Ok(owl_nn::DynTensor::from_u32(&t))
    }
    pub fn eye(n: usize, m: usize, _dtype: DType, _device: &Device) -> Result<Tensor> {
        let _ = (n, m);
        t3_unimpl("eye")
    }
    pub fn from_vec<T: Copy + 'static>(
        v: Vec<T>,
        shape: impl Into<Shape>,
        _device: &Device,
    ) -> Result<Tensor> {
        // dry-run 装载基元:借 ctx_scope rig 的 weights 池落池(账本化)。
        // FIX:u32 保真 U32 位型(原 f32::from_bits 强转 = 位型误解,
        // 索引/slot 全变垃圾 f32,S4 违约根源)。
        let shape = shape.into().dims().to_vec();
        let pool = crate::models::layers::ctx_scope::weights_pool();
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            let data: Vec<f32> = unsafe { std::mem::transmute::<Vec<T>, Vec<f32>>(v) };
            let t = owl_nn::TensorPoolOps::from_vec_tensor(pool.as_ref(), &shape, data)?;
            return Ok(owl_nn::DynTensor::from_f32(&t));
        }
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u32>() {
            let data: Vec<u32> = unsafe { std::mem::transmute::<Vec<T>, Vec<u32>>(v) };
            let t = owl_nn::TensorPoolOps::from_vec_tensor(pool.as_ref(), &shape, data)?;
            return Ok(owl_nn::DynTensor::from_u32(&t));
        }
        return Err(crate::Error::Msg(format!(
            "ctor::from_vec: dry-run 仅支持 f32/u32(收到 {})",
            std::any::type_name::<T>()
        )));
    }
    pub fn new<T: Copy>(
        _v: Vec<T>,
        _shape: impl Into<Shape>,
        _device: &Device,
    ) -> Result<Tensor> {
        t3_unimpl("new")
    }
    pub fn empty(shape: impl Into<Shape>, _dtype: DType, _device: &Device) -> Result<Tensor> {
        t3_unimpl("empty")
    }
}


/// safetensors 多文件查找索引(VarBuilderX safetensors 通道专用)。
///
/// 建索引时做两件事(零数据读取,只碰 header):
/// - **过滤**:`mtp.*`(MTP 草稿头)与 `model.visual.*`(视觉塔)不入索引,
///   查询返回不存在而非命中;
/// - **归一**:HF 多模态壳把文本模型嵌在 `model.language_model.`,checkpoint
///   键 `model.language_model.X` 追加别名 `model.X`(引擎请求路径;纯文本
///   checkpoint 无此嵌套时别名不产生,恒等查找)。
struct SafeIndex {
    files: Vec<crate::loader::safetensors::SafeTensorsFile>,
    /// 查找键 → (文件序号, 文件内实名)
    map: std::collections::HashMap<String, (usize, String)>,
}

impl SafeIndex {
    fn open(paths: &[std::path::PathBuf]) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        let mut map: std::collections::HashMap<String, (usize, String)> =
            std::collections::HashMap::new();
        for (fi, p) in paths.iter().enumerate() {
            let f = crate::loader::safetensors::SafeTensorsFile::open(p)?;
            let names: Vec<String> = f.names().map(String::from).collect();
            for name in names {
                if name.starts_with("mtp.") || name.starts_with("model.visual.") {
                    continue;
                }
                map.entry(name.clone()).or_insert((fi, name.clone()));
                if let Some(rest) = name.strip_prefix("model.language_model.") {
                    map.entry(format!("model.{rest}"))
                        .or_insert((fi, name.clone()));
                }
            }
            files.push(f);
        }
        Ok(Self { files, map })
    }

    fn resolve(&self, key: &str) -> Option<(&crate::loader::safetensors::SafeTensorsFile, &str)> {
        self.map
            .get(key)
            .and_then(|(fi, real)| self.files.get(*fi).map(|f| (f, real.as_str())))
    }

    fn contains(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    fn shape(&self, key: &str) -> Option<Vec<usize>> {
        self.resolve(key)
            .and_then(|(f, real)| f.info(real).ok().map(|i| i.shape.clone()))
    }

    fn tensor_f32(&self, key: &str) -> Result<Vec<f32>> {
        let (f, real) = self
            .resolve(key)
            .ok_or_else(|| Error::Msg(format!("safetensors: 无此张量 {key}")))?;
        f.tensor_f32(real)
    }
}

/// GGUF/safetensors 双路 var builder 包装(= xinfer `models::layers::VarBuilderX`)。
///
/// T2-三 接线(2026-09-22):GGUF 通道经 [`crate::loader::gguf::GGufVarBuilder`]
/// 全链真实化——元数据查询(has_key/contains_tensor/tensor_shape/get_no_shape)
/// 与 `get/get_with_hints_dtype`(元数据 shape 校验 → 反解 → 目标 dtype
/// 位型转换 → 池直连工厂落池 → 擦除张量)。分配语义遵守裁决 5:
/// 权重装载 = P 阶段,经注入的 Weights 池(`with_pool`);无池 = Host
/// 测试模式(get 系列结构化报错,host 产物走 [`Self::get_host`])。
#[derive(Clone)]
pub struct VarBuilderX {
    /// 模块路径(前缀)
    pub module_path: String,
    /// 底层权重文件列表
    pub weight_paths: Option<Vec<std::path::PathBuf>>,
    /// 是否 GGUF(Q) 路径
    pub is_gguf: bool,
    /// GGUF 内容句柄(全量元数据 + 分片解析;Arc 共享,pp 下钻零拷贝)
    gguf: Option<std::sync::Arc<crate::loader::gguf::GGufVarBuilder>>,
    /// R3 统一装载口(池由 allocator 持有;None = Host 测试模式)
    alloc: Option<std::sync::Arc<crate::loader::DeviceWeightAllocator<owl_cuda::CudaPool>>>,
    /// HF safetensors 通道(is_gguf=false 且 filenames 非空;Arc 共享,pp 下钻零拷贝)
    st: Option<std::sync::Arc<SafeIndex>>,
    // 设备句柄(device() 查询面)
    device: Option<Device>,
    // dry-run 假数据模式(from_fake;种子 = 键名哈希,确定性可复现)
    fake: bool,
}


impl VarBuilderX {
    /// GGUF 打开(全量元数据;数据按需读取)。`model_pathes.filenames`
    /// 为空或 is_gguf=false 时仅建空壳(Host 模式测试/纯元数据用途)。
    pub fn new(
        model_pathes: &crate::downloader::ModelPaths,
        is_gguf: bool,
        _dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let (gguf, st) = if is_gguf && !model_pathes.filenames.is_empty() {
            (
                Some(std::sync::Arc::new(
                    crate::loader::gguf::GGufVarBuilder::from_gguf_files(&model_pathes.filenames)?,
                )),
                None,
            )
        } else if !is_gguf && !model_pathes.filenames.is_empty() {
            // HF safetensors(单/多分片);header 全量校验在 open 内
            (None, Some(std::sync::Arc::new(SafeIndex::open(&model_pathes.filenames)?)))
        } else {
            (None, None)
        };
        Ok(Self {
            module_path: String::new(),
            weight_paths: Some(model_pathes.filenames.clone()),
            is_gguf,
            gguf,
            st,
            alloc: None,
            device: Some(device.clone()),
            fake: false,
        })
    }

    /// dry-run 假数据模式:不读文件,get 按请求 shape 产种子化确定性数据。
    pub fn from_fake(pool: std::sync::Arc<owl_cuda::CudaPool>, device: &Device) -> Result<Self> {
        let mut vb = Self::new(
            &crate::downloader::ModelPaths {
                tokenizer_filename: Default::default(),
                tokenizer_config_filename: Default::default(),
                config_filename: Default::default(),
                generation_config_filename: Default::default(),
                filenames: vec![],
                auxiliary_filenames: vec![],
                chat_template_filename: None,
            },
            false,
            DType::F32,
            device,
        )?;
        vb.fake = true;
        vb.alloc = Some(std::sync::Arc::new(
            crate::loader::DeviceWeightAllocator::new(pool),
        ));
        Ok(vb)
    }

    /// 注入装载目标池(P 阶段;R3 统一装载口:内部持 DeviceWeightAllocator)。
    pub fn with_pool(mut self, pool: std::sync::Arc<owl_cuda::CudaPool>) -> Self {
        self.alloc = Some(std::sync::Arc::new(
            crate::loader::DeviceWeightAllocator::new(pool),
        ));
        self
    }

    pub fn from_gguf_file_host<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let vb = crate::loader::gguf::GGufVarBuilder::from_gguf(path)?;
        Ok(Self {
            module_path: String::new(),
            weight_paths: None,
            is_gguf: true,
            gguf: Some(std::sync::Arc::new(vb)),
            st: None,
            alloc: None,
            device: None,
            fake: false,
        })
    }

    /// safetensors host 模式构造器(零设备依赖;装载链/对拍单测用)。
    pub fn from_safetensors_file_host<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let paths = vec![path.as_ref().to_path_buf()];
        let idx = SafeIndex::open(&paths)?;
        Ok(Self {
            module_path: String::new(),
            weight_paths: None,
            is_gguf: false,
            gguf: None,
            st: Some(std::sync::Arc::new(idx)),
            alloc: None,
            device: None,
            fake: false,
        })
    }

    pub fn is_gguf(&self) -> bool {
        self.is_gguf
    }
    pub fn is_var_builder(&self) -> bool {
        !self.is_gguf
    }
    pub fn is_qvar_builder(&self) -> bool {
        self.is_gguf
    }
    pub fn device(&self) -> Device {
        self.device
            .clone()
            .expect("VarBuilderX::device: 无设备句柄(Host 模式)")
    }

    /// 前缀下钻(路径拼接语义 = 真实,与 xinfer 一致)
    pub fn pp(&self, name: &str) -> VarBuilderX {
        let next = if self.module_path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.module_path, name)
        };
        VarBuilderX {
            module_path: next,
            weight_paths: self.weight_paths.clone(),
            is_gguf: self.is_gguf,
            gguf: self.gguf.clone(),
            st: self.st.clone(),
            alloc: self.alloc.clone(),
            device: self.device.clone(),
            fake: self.fake,
        }
    }

    pub fn aux(&self) -> Option<VarBuilderX> {
        None // 多文件 aux 权重(xinfer mm projector);本项目面未见,按需回填
    }
    pub fn gguf_path(&self) -> Option<&str> {
        self.gguf.as_ref().map(|g| g.gguf_path())
    }
    pub fn weight_paths(&self) -> Option<Vec<std::path::PathBuf>> {
        self.weight_paths.clone()
    }
    pub fn cpu_var_builder(&self) -> Option<VarBuilderX> {
        None // CPU offload 面本项目未见;按需回填
    }
    pub fn module_path(&self) -> &str {
        &self.module_path
    }
    pub fn has_key(&self, name: &str) -> bool {
        if !self.is_gguf {
            return self
                .st
                .as_ref()
                .map(|s| s.contains(&self.full_name(name)))
                .unwrap_or(false);
        }
        self.gguf
            .as_ref()
            .map(|g| g.contains_key(&self.full_name(name)))
            .unwrap_or(false)
    }
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.has_key(name)
    }
    pub fn get_no_shape(&self, name: &str) -> Result<Tensor> {
        // 无 shape 请求 = 按元数据原 shape 反解(safetensors 原生行主序;
        // GGUF 维序反转回 owl 行主序)
        let full = self.full_name(name);
        if !self.is_gguf {
            let shape = self
                .st
                .as_ref()
                .and_then(|s| s.shape(&full))
                .ok_or_else(|| Error::Msg(format!("VarBuilderX: 权重缺失 {full}")))?;
            return self.get_with_hints_dtype(shape, name, Shard::default(), DType::F32);
        }
        let shape = self
            .gguf
            .as_ref()
            .and_then(|g| g.tensor_shape(&full))
            .ok_or_else(|| Error::Msg(format!("VarBuilderX: 权重缺失 {full}")))?;
        // GGUF 元数据为反转维序
        let mut owl_shape = shape;
        owl_shape.reverse();
        self.get_with_hints_dtype(owl_shape, name, Shard::default(), DType::F32)
    }
    pub fn tensor_shape(&self, name: &str) -> Option<Vec<usize>> {
        if !self.is_gguf {
            return self.st.as_ref().and_then(|s| s.shape(&self.full_name(name)));
        }
        let mut s = self.gguf.as_ref().and_then(|g| g.tensor_shape(&self.full_name(name)))?;
        s.reverse(); // GGUF 反转维序 → owl 行主序
        Some(s)
    }

    fn full_name(&self, name: &str) -> String {
        if self.module_path.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.module_path, name)
        }
    }

    fn gguf_ref(&self) -> Result<std::sync::Arc<crate::loader::gguf::GGufVarBuilder>> {
        self.gguf
            .clone()
            .ok_or_else(|| Error::Msg("VarBuilderX: 非 GGUF 路径".into()))
    }

    fn st_ref(&self) -> Result<std::sync::Arc<SafeIndex>> {
        self.st
            .clone()
            .ok_or_else(|| Error::Msg("VarBuilderX: 非 safetensors 路径".into()))
    }

    /// R3 统一装载口尾段:f32 → 目标 dtype 字节流 → materialize_dyn 落池
    /// (账本化)→ 擦除句柄。
    fn materialize_f32_weight(
        &self,
        shape: Vec<usize>,
        f32s: Vec<f32>,
        dtype: DType,
    ) -> Result<Tensor> {
        let alloc = self.alloc_ref()?;
        let target = match dtype {
            DType::F32 => DType::F32,
            DType::BF16 => DType::BF16,
            DType::F16 => DType::F16,
            _ => {
                return Err(Error::Msg(format!(
                    "VarBuilderX: 目标 dtype {dtype} 权重装载未开(U8/U32/I64 经 kvcache/索引专用通道;F8 量化 = marlin-ffi 路线)"
                )))
            }
        };
        let bytes = crate::loader::f32_vec_to_dtype_bytes(target, &f32s)?;
        alloc.materialize_dyn(&shape, target, &bytes)
    }

    fn alloc_ref(&self) -> Result<std::sync::Arc<crate::loader::DeviceWeightAllocator<owl_cuda::CudaPool>>> {
        self.alloc
            .clone()
            .ok_or_else(|| Error::Msg(
                "VarBuilderX: 未注入装载 allocator(Host 测试模式;设备 get 需 with_pool)".into(),
            ))
    }

    pub fn get_with_hints_dtype(
        &self,
        s: impl Into<Shape>,
        name: &str,
        shard: Shard,
        dtype: DType,
    ) -> Result<Tensor> {
        if shard.world_size > 1 {
            unimplemented!("TP 分片装载(Row/Col)= T3 后续(当前单卡面)");
        }
        // dry-run 假数据模式:种子 = FNV(键名),值域 ±0.01(小值防 softmax 饱和)
        if self.fake {
            let shape = s.into().dims().to_vec();
            let n: usize = shape.iter().product();
            let mut h: u64 = 0xcbf29ce484222325;
            for b in name.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            let mut x = h | 1;
            let mut data = Vec::with_capacity(n);
            for _ in 0..n {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                data.push(((x >> 33) % 2001) as f32 / 100_000.0 - 0.01);
            }
            let pool = self.alloc.as_ref().ok_or_else(|| {
                Error::Msg("fake 模式需要 with_pool(装载口统一,R3)".into())
            })?;
            // f32 dry-run;dtype 面在运行里程碑接位型转换
            return Ok(pool.materialize_dyn(
                &shape,
                owl_nn::Dtype::F32,
                bytemuck::cast_slice(&data),
            )?);
        }
        let full = self.full_name(name);
        if !self.is_gguf {
            // safetensors 通道:shape 校验(原生行主序)→ 懒转 f32 → 落池
            let st = self.st_ref()?;
            let want = s.into().dims().to_vec();
            let shape = st
                .shape(&full)
                .ok_or_else(|| Error::Msg(format!("VarBuilderX: 权重缺失 {full}")))?;
            if shape != want {
                return Err(Error::Msg(format!(
                    "VarBuilderX: {full} shape 不符: 请求 {want:?}, 实际 {shape:?}"
                )));
            }
            let f32s = st.tensor_f32(&full)?;
            return self.materialize_f32_weight(shape, f32s, dtype);
        }
        let gg = self.gguf_ref()?;
        // shape 校验真实:请求(owl 行主序)→ GGUF 反转维序后比对
        let mut want = s.into().dims().to_vec();
        want.reverse();
        let raw = gg.get(&want, &full)?;
        let mut owl_shape = raw.shape.clone();
        owl_shape.reverse();

        // GGUF:反解到 f32(量化经 dequantize)后走统一尾段
        let f32s: Vec<f32> = if raw.dtype == crate::loader::gguf::GgmlDType::F32 {
            raw.raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else {
            crate::loader::gguf::dequantize_to_f32(raw.dtype, &raw.shape, &raw.raw)?
        };
        self.materialize_f32_weight(owl_shape, f32s, dtype)
    }
    pub fn get(&self, s: impl Into<Shape>, name: &str) -> Result<Tensor> {
        self.get_with_hints_dtype(s, name, Shard::default(), DType::F32)
    }
    pub fn get_with_hints(
        &self,
        s: impl Into<Shape>,
        name: &str,
        hints: Shard,
    ) -> Result<Tensor> {
        self.get_with_hints_dtype(s, name, hints, DType::F32)
    }

    /// Host 测试模式:反解产物留在内存(零设备依赖;供装载链单测)。
    pub fn get_host(&self, s: impl Into<Shape>, name: &str) -> Result<HostTensorBoxed> {
        let full = self.full_name(name);
        if !self.is_gguf {
            let st = self.st_ref()?;
            let want = s.into().dims().to_vec();
            let shape = st
                .shape(&full)
                .ok_or_else(|| Error::Msg(format!("VarBuilderX: 权重缺失 {full}")))?;
            if shape != want {
                return Err(Error::Msg(format!(
                    "VarBuilderX: {full} shape 不符: 请求 {want:?}, 实际 {shape:?}"
                )));
            }
            let f32s = st.tensor_f32(&full)?;
            let mut bytes = Vec::with_capacity(f32s.len() * 4);
            for f in &f32s {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
            return Ok(HostTensorBoxed { shape, data: bytes });
        }
        let gg = self.gguf_ref()?;
        let mut want = s.into().dims().to_vec();
        want.reverse();
        let raw = gg.get(&want, &full)?;
        let f32s = if raw.dtype == crate::loader::gguf::GgmlDType::F32 {
            raw.raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        } else {
            crate::loader::gguf::dequantize_to_f32(raw.dtype, &raw.shape, &raw.raw)?
        };
        let mut owl_shape = raw.shape.clone();
        owl_shape.reverse();
        let mut bytes = Vec::with_capacity(f32s.len() * 4);
        for f in &f32s {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        Ok(HostTensorBoxed {
            shape: owl_shape,
            data: bytes,
        })
    }
}

/// Host 模式反解产物(测试/对拍用;f32 LE)
pub struct HostTensorBoxed {
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

/// 分片参数(= candle_nn::var_builder::Shard;结构直译)
#[derive(Copy, Clone, Debug, Default)]
pub struct Shard {
    pub dim: usize,
    pub rank: usize,
    pub world_size: usize,
}

// ============================================================================
// candle_nn 容器桩(Linear/Conv/Embedding/Module/...)
// ============================================================================

/// 线性层真身在 linear::Linear(本桩已被其取代;re-export 统一词汇)
pub use linear::Linear;

/// 卷积层(= candle_nn::Conv;deltanet 用;权重 T3 回填)
#[derive(Clone)]
#[allow(dead_code, unused_variables)]
pub struct Conv {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub kernel: usize,
    pub stride: usize,
    pub groups: usize,
    pub padding: usize,
    pub dilation: usize,
    pub with_bias: bool,
}

/// 2D 卷积配置(= candle_nn::Conv2dConfig;类型面)
#[derive(Debug, Clone, Copy, Default)]
pub struct Conv2dConfig {
    pub padding: usize,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
}

impl Conv {
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        let kernel = weight.shape().last().copied().unwrap_or(1);
        let with_bias = bias.is_some();
        Self {
            weight,
            bias,
            kernel,
            stride: 1,
            groups: 1,
            padding: 0,
            dilation: 1,
            with_bias,
        }
    }
    pub fn conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let _ = &self.weight;
        unimplemented!("T3 kernel 回填: conv1d")
    }
    pub fn conv2d(&self, x: &Tensor) -> Result<Tensor> {
        let _ = &self.weight;
        unimplemented!("T3 kernel 回填: conv2d")
    }
    /// 统一前向入口(others::conv2d/Conv3dNoBias 调用面)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.conv2d(x)
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

/// 嵌入层(= candle_nn::Embedding;权重 T3 回填)
#[derive(Clone)]
pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(weight: Tensor) -> Self {
        Self { weight }
    }
    pub fn embed(&self, ids: &Tensor) -> Result<Tensor> {
        // dry-run:erased index_select(dim=0,U32 索引)→ [n_ids, d]
        crate::models::layers::ctx_scope::with(|ops, ctx| {
            let out = owl_nn::erased::index_select(ops, ctx, &self.weight, 0, ids)
                .map_err(crate::Error::from)?;
            // xinfer Embedding 语义:返回 [n_ids, d](调用点自行 reshape 扩维)
            Ok(out)
        })
    }
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        self.embed(ids)
    }
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn vocab_size(&self) -> usize {
        self.weight.shape().first().copied().unwrap_or(0)
    }
    pub fn dim(&self) -> usize {
        self.weight.shape().last().copied().unwrap_or(0)
    }
}

/// 模块 trait(= candle_nn::Module / candle_core::Module;forward 面)
pub trait Module {
    type Output;
    fn forward(&self, x: &Tensor) -> Result<Self::Output>;
}

/// 层归一化 / RMSNorm 容器(= candle_nn::{LayerNorm, RmsNorm})
#[derive(Clone)]
pub struct LayerNorm {
    pub weight: Option<Tensor>,
    pub bias: Option<Tensor>,
    pub eps: f64,
    pub affine: bool,
}

impl LayerNorm {
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self {
            weight: Some(weight),
            bias: Some(bias),
            eps,
            affine: true,
        }
    }
    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        Self {
            weight: Some(weight),
            bias: None,
            eps,
            affine: true,
        }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _ = x;
        unimplemented!("T3 kernel 回填: layer_norm")
    }
}

#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Option<Tensor>,
    pub eps: f64,
}

impl RmsNorm {
    pub fn new(weight: Tensor, eps: f64) -> Self {
        Self {
            weight: Some(weight),
            eps,
        }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let w = self.weight.as_ref().ok_or_else(|| {
            crate::Error::Msg("RmsNorm: 无 weight(dry-run 需 affine 形态)".into())
        })?;
        let eps = self.eps as f32;
        ctx_scope::with(|ops, ctx| erased::rmsnorm(ops, ctx, x, w, eps).map_err(Into::into))
    }
}

// ============================================================================
// vendor 占位(attention_rs / flashinfer;归宿裁决待定)
// ============================================================================

/// vendor kernel/缓存面占位。design:三选一(自研 kernel 进 owl-kernels /
/// 保留 vendor 依赖 / 换 flashinfer)在 T3 运行里程碑前裁决。
pub mod vendor {
    use super::{ctx_scope, erased, DType, Device, Error, OwlTensor, Result, Tensor};

    // fused rope(= attention_rs::fused_rope::FusedRope)
    pub mod fused_rope {
        use crate::Result;
        use super::super::{ctx_scope, erased, Tensor};
        use owl_nn::DynTensor;
        pub fn apply_inplace(
            q: &Tensor,
            k: &Tensor,
            cos: &Tensor,
            sin: &Tensor,
            positions: &Tensor,
            _is_rope_i: bool,
            num_heads: usize,
            num_kv_heads: usize,
            head_dim: usize,
        ) -> Result<()> {
            // naive rope_half dry 核(q/k 各 tokens×heads×head_dim 连续;写回 in-place)
            ctx_scope::with_dry(|ctx, dry| {
                let tokens = positions.shape()[0];
                let q_out = ctx.scratch_tensor::<f32>(q.shape())?;
                let k_out = ctx.scratch_tensor::<f32>(k.shape())?;
                dry.rope_half_f32(
                    ctx.stream(),
                    q.device_ptr() as *const f32,
                    k.device_ptr() as *const f32,
                    q_out.device_ptr() as *mut f32,
                    k_out.device_ptr() as *mut f32,
                    cos.device_ptr() as *const f32,
                    sin.device_ptr() as *const f32,
                    positions.device_ptr() as *const u32,
                    tokens,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                ).map_err(|e| crate::Error::Msg(format!("rope_half: {e}")))?;
                let q_bytes = q.len_bytes();
                let k_bytes = k.len_bytes();
                let qo = DynTensor::from_f32(&q_out);
                let ko = DynTensor::from_f32(&k_out);
                erased::copy_d2d_to_raw(ctx, &qo, q.device_ptr() as *mut core::ffi::c_void, q_bytes)?;
                erased::copy_d2d_to_raw(ctx, &ko, k.device_ptr() as *mut core::ffi::c_void, k_bytes)?;
                Ok(())
            })
        }
        pub fn apply_inplace_partial(
            _q: &Tensor,
            _k: &Tensor,
            _cos: &Tensor,
            _sin: &Tensor,
            _positions: &Tensor,
            _is_rope_i: bool,
            _rotary_dim: usize,
        ) -> Result<()> {
            Err(crate::Error::Msg("vendor fused_rope::apply_inplace_partial: T3 kernel 回填".into()))
        }
    }
    /// 分页注意力句柄(= attention_rs::PagedAttention)
    pub struct PagedAttention;
    /// Mamba/GDN 状态缓存(B3.2 真实现;= attention_rs::mamba_cache::MambaCache 收敛面)。
    /// 每层两张 F32 常驻表:conv [max_batch, d_conv, k-1]、recurrent [max_batch, nv, K, V]。
    /// P 阶段一次分配(preallocate → ctor::zeros,裁决 5);slot → 行寻址;
    /// prefill 行收集/回写 = D2D(erased::copy_d2d_to_raw);decode 直改全表(核内 slot 寻址)。
    pub struct MambaCache {
        conv_states: Vec<Tensor>,
        recurrent_states: Vec<Tensor>,
        max_batch: usize,
        /// 空闲 slot 栈(A9 槽分配;0 = 有效行,0xFFFFFFFF 无效哨兵与 GDN 核一致)
        free_slots: Vec<usize>,
        /// seq_id → slot
        owner_of: std::collections::HashMap<usize, usize>,
    }

    impl MambaCache {
        /// 空壳(runner 在 warmup 前 preallocate;容量参数在模型构造面才有)
        pub fn empty() -> Self {
            Self {
                conv_states: Vec::new(),
                recurrent_states: Vec::new(),
                max_batch: 0,
                free_slots: Vec::new(),
                owner_of: std::collections::HashMap::new(),
            }
        }

        /// P 阶段分配全层状态表(裁决 5:分配入口 ctor::zeros → 权重池)
        #[allow(clippy::too_many_arguments)]
        pub fn preallocate(
            &mut self,
            num_layers: usize,
            max_batch: usize,
            d_conv: usize,
            conv_len: usize,
            num_v_heads: usize,
            k_dim: usize,
            v_dim: usize,
            device: &Device,
        ) -> Result<()> {
            if !self.conv_states.is_empty() {
                return Ok(()); // 幂等
            }
            self.conv_states = (0..num_layers)
                .map(|_| super::ctor::zeros((max_batch, d_conv, conv_len), DType::F32, device))
                .collect::<Result<Vec<_>>>()?;
            self.recurrent_states = (0..num_layers)
                .map(|_| {
                    super::ctor::zeros(
                        (max_batch, num_v_heads, k_dim, v_dim),
                        DType::F32,
                        device,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            self.max_batch = max_batch;
            self.free_slots = (0..max_batch).rev().collect();
            Ok(())
        }

        pub fn is_allocated(&self) -> bool {
            !self.conv_states.is_empty()
        }
        pub fn max_batch(&self) -> usize {
            self.max_batch
        }
        pub fn conv_state(&self, layer: usize) -> &Tensor {
            &self.conv_states[layer]
        }
        pub fn conv_state_mut(&mut self, layer: usize) -> &mut Tensor {
            &mut self.conv_states[layer]
        }
        pub fn recurrent_state(&self, layer: usize) -> &Tensor {
            &self.recurrent_states[layer]
        }
        pub fn recurrent_state_mut(&mut self, layer: usize) -> &mut Tensor {
            &mut self.recurrent_states[layer]
        }

        /// A9:分配(或查既有)槽位;零层状态表(纯注意力模型)= 形式槽,
        /// 直接用 seq_id 作槽号(无表可寻址,与旧 dry-run 语义一致)
        pub fn ensure_slot(&mut self, seq_id: usize) -> Result<usize> {
            if let Some(&s) = self.owner_of.get(&seq_id) {
                return Ok(s);
            }
            if !self.is_allocated() {
                self.owner_of.insert(seq_id, seq_id);
                return Ok(seq_id);
            }
            let slot = self
                .free_slots
                .pop()
                .ok_or_else(|| Error::Msg(format!("MambaCache: 槽位耗尽(max_batch={})", self.max_batch)))?;
            self.owner_of.insert(seq_id, slot);
            Ok(slot)
        }
        /// A9:释放(seq 结束)
        pub fn free_slot(&mut self, seq_id: usize) {
            if let Some(slot) = self.owner_of.remove(&seq_id) {
                self.free_slots.push(slot);
            }
        }
        pub fn slot_of(&self, seq_id: usize) -> Option<usize> {
            self.owner_of.get(&seq_id).copied()
        }

        /// prefill 行收集:slots U32 设备张量 → D2H(急切路径合法)→ 逐行 D2D → [n, d, k-1]
        pub fn gather_conv_rows(&self, layer: usize, slots: &Tensor) -> Result<Tensor> {
            let slot_ids = read_u32_device(slots)?;
            let src = &self.conv_states[layer];
            let row: usize = src.shape()[1..].iter().product();
            ctx_scope::with(|_ops, ctx| {
                let n = slot_ids.len();
                let out = ctx.scratch_tensor::<f32>(&[n.max(1), row])?;
                let base = out.device_ptr();
                for (i, &s) in slot_ids.iter().enumerate() {
                    let view = src.narrow(0usize, s as usize, 1)?;
                    unsafe {
                        erased::copy_d2d_to_raw(
                            ctx,
                            &view,
                            base.add(i * row) as *mut core::ffi::c_void,
                            row * 4,
                        )?;
                    }
                }
                Ok(owl_nn::DynTensor::from_f32(&out))
            })
        }

        /// prefill 行回写(核已就地更新收集副本;按 slot 写回常驻表)
        pub fn scatter_conv_rows(&mut self, layer: usize, slots: &Tensor, rows: &Tensor) -> Result<()> {
            let slot_ids = read_u32_device(slots)?;
            let row: usize = self.conv_states[layer].shape()[1..].iter().product();
            ctx_scope::with(|_ops, ctx| {
                for (i, &s) in slot_ids.iter().enumerate() {
                    let piece = rows.narrow(0usize, i, 1)?;
                    let dst = self.conv_states[layer].narrow(0usize, s as usize, 1)?;
                    let dst_ptr = dst.device_ptr();
                    erased::copy_d2d_to_raw(ctx, &piece, dst_ptr as *mut core::ffi::c_void, row * 4)?;
                }
                Ok(())
            })
        }

        /// recurrent 行收集:slots U32 → [n, nv, K, V](prefill 递推逐序列取行)
        pub fn gather_rec_rows(&self, layer: usize, slots: &Tensor) -> Result<Tensor> {
            let slot_ids = read_u32_device(slots)?;
            let src = &self.recurrent_states[layer];
            let row: usize = src.shape()[1..].iter().product();
            ctx_scope::with(|_ops, ctx| {
                let n = slot_ids.len();
                let out = ctx.scratch_tensor::<f32>(&[n.max(1), row])?;
                let base = out.device_ptr();
                for (i, &s) in slot_ids.iter().enumerate() {
                    let view = src.narrow(0usize, s as usize, 1)?;
                    unsafe {
                        erased::copy_d2d_to_raw(
                            ctx,
                            &view,
                            base.add(i * row) as *mut core::ffi::c_void,
                            row * 4,
                        )?;
                    }
                }
                Ok(owl_nn::DynTensor::from_f32(&out))
            })
        }

        /// recurrent 行回写
        pub fn scatter_rec_rows(&mut self, layer: usize, slots: &Tensor, rows: &Tensor) -> Result<()> {
            let slot_ids = read_u32_device(slots)?;
            let row: usize = self.recurrent_states[layer].shape()[1..].iter().product();
            ctx_scope::with(|_ops, ctx| {
                for (i, &s) in slot_ids.iter().enumerate() {
                    let piece = rows.narrow(0usize, i, 1)?;
                    let dst = self.recurrent_states[layer].narrow(0usize, s as usize, 1)?;
                    let dst_ptr = dst.device_ptr();
                    erased::copy_d2d_to_raw(ctx, &piece, dst_ptr as *mut core::ffi::c_void, row * 4)?;
                }
                Ok(())
            })
        }
    }

    /// 设备 U32 张量 → host(急切 prefill 的槽位读取;捕获路径禁用——
    /// decode 核内自寻址不走此路)
    pub(crate) fn read_u32_device(t: &Tensor) -> Result<Vec<u32>> {
        if t.dtype() != DType::U32 {
            return Err(Error::Msg(format!(
                "read_u32_device: 槽张量应为 U32,实际 {}(S4)",
                t.dtype()
            )));
        }
        let dev = ctx_scope::with_device();
        let _ = dev.ctx().bind_to_thread();
        let n = t.len_bytes() / 4;
        let mut host = vec![0u32; n];
        use owl_cuda::ffi::sys;
        unsafe {
            sys::cuMemcpyDtoH_v2(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                t.device_ptr() as sys::CUdeviceptr,
                n * 4,
            )
            .result()
            .map_err(|e| Error::Msg(format!("dtoh slots: {e:?}")))?;
        }
        Ok(host)
    }

    /// 输入元数据(= attention_rs::InputMetadata;字段面 T3 按调用点补齐)
    #[derive(Clone, Default)]
    pub struct InputMetadata {
        pub seqlens: Vec<usize>,
        pub context_lens: Vec<usize>,
        /// prefill/decode 分派(attention/deltanet forward 入口分支)
        pub is_prefill: bool,
        /// MTP/DFlash2 verify 打包批次标记(快照区写入面)
        pub is_mtp_verify: bool,
        /// 变长 prefill 的序列偏移(U32 device 张量;A1.5 动态量设备化)
        pub cu_seqlens_q: Option<super::Tensor>,
        /// decode naive 路径的设备指针对(位型直读:i32 slots/kv_lens;
        /// u32 bindings 的位型一致)。None = 未提供(pa_shim 拒绝 decode)。
        pub decode_ptrs: Option<DecodePtrs>,
    }

    /// decode 设备指针对(slots/kv_lens;值域 <2^31,u32 位型等价 i32)
    #[derive(Clone, Copy)]
    pub struct DecodePtrs {
        pub slots: *const i32,
        pub kv_lens: *const i32,
    }
    /// MoE 算子面(= attention_rs::moe)
    pub mod moe {
        pub struct FusedMoe;
    }
    /// GDN 算子面(= attention_rs::gdn)
    pub mod gdn {}
    /// silu_and_mul(= attention_rs::silu_and_mul)
    pub mod silu_and_mul {}
    /// GGUF 量化线性(= attention_rs::gguf_linear)
    pub mod gguf_linear {
        pub fn gguf_iq_matmul() -> crate::error::Result<()> {
            unimplemented!("vendor 归宿裁决待定: gguf_iq_matmul")
        }
        pub fn is_iq_gguf_dtype() -> bool {
            false
        }
    }
    /// 排序算子(= attention_rs::sort::ArgSortOp)
    pub mod sort {
        pub struct ArgSortOp;
    }
    /// DeepSeek-V4 ATen-order RMSNorm(= attention_rs::deepseek_v4)
    pub mod deepseek_v4 {
        use crate::Result;
        use super::super::Tensor;
        pub fn rms_norm_v4(
            _x: &Tensor,
            _weight: &Tensor,
            _dim: usize,
            _eps: f32,
        ) -> Result<Tensor> {
            Err(crate::Error::Msg("vendor deepseek_v4::rms_norm_v4: T3 kernel 回填".into()))
        }
        pub fn rms_norm_v4_inplace(
            _x: &Tensor,
            _weight: &Tensor,
            _dim: usize,
            _eps: f32,
        ) -> Result<()> {
            Err(crate::Error::Msg("vendor deepseek_v4::rms_norm_v4_inplace: T3 kernel 回填".into()))
        }
    }
    /// MLX NVFP4 反量化(= attention_rs::nvfp4_linear)
    pub mod nvfp4_linear {
        use crate::Result;
        use super::super::{Tensor, DType};
        pub fn mlx_dequant_embedding(
            _w_u32: &Tensor,
            _scales: &Tensor,
            _vocab: usize,
            _hidden: usize,
            _out_dtype: DType,
        ) -> Result<Tensor> {
            Err(crate::Error::Msg("vendor nvfp4_linear: 量化 = marlin-ffi 路线".into()))
        }
    }
}

/// 量化 dtype(= candle_core::quantized::GgmlDType;类型面)
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[allow(non_camel_case_types)]
pub enum GgmlDType {
    #[default]
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ3_XXS,
    IQ3_S,
    IQ4_XS,
    IQ4_NL,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

/// 量化张量(= candle_core::quantized::QTensor;类型面)
#[derive(Clone)]
pub struct QTensor;

/// 存储抽象(= candle_core::{Storage, CudaStorage, CpuStorage};类型面)
#[derive(Clone)]
pub struct CudaStorage;
#[derive(Clone)]
pub struct CpuStorage;
#[derive(Clone)]
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
}

/// 布局(= candle_core::Layout;类型面)

#[derive(Clone, Debug)]
pub struct Layout {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

/// 自定义算子 trait(= candle_core::CustomOp1;类型面)
pub trait CustomOp1 {}

/// WithDType(= candle_core::WithDType;dtype() 面)
pub trait WithDType {
    fn dtype(&self) -> DType;
}
impl WithDType for Tensor {
    fn dtype(&self) -> DType {
        self.dtype()
    }
}
/// IndexOp(= candle_core::IndexOp;index_select/gather 面)
pub trait IndexOp: OwlTensor {}
impl IndexOp for Tensor {}

// ============================================================================
// VarBuilderX 装载链真机测试(T2-三 目标一验收)
// ============================================================================

#[cfg(test)]
mod varbuilder_tests {
    use super::*;

    /// 合成 GGUF(与 loader 测试同构造;自带 t_f32 [2,2] 与 t_q8_0 [32])
    fn write_synthetic_gguf() -> std::path::PathBuf {
        let mut w: Vec<u8> = Vec::new();
        w.extend_from_slice(&0x46554747u32.to_le_bytes());
        w.extend_from_slice(&3u32.to_le_bytes());
        w.extend_from_slice(&2u64.to_le_bytes());
        w.extend_from_slice(&1u64.to_le_bytes());
        w.extend_from_slice(&17u64.to_le_bytes());
        w.extend_from_slice(b"general.alignment");
        w.extend_from_slice(&4u32.to_le_bytes());
        w.extend_from_slice(&32u32.to_le_bytes());
        let name1 = b"t_f32";
        w.extend_from_slice(&(name1.len() as u64).to_le_bytes());
        w.extend_from_slice(name1);
        w.extend_from_slice(&2u32.to_le_bytes());
        w.extend_from_slice(&2u64.to_le_bytes());
        w.extend_from_slice(&2u64.to_le_bytes());
        w.extend_from_slice(&0u32.to_le_bytes());
        w.extend_from_slice(&0u64.to_le_bytes());
        let name2 = b"t_q8_0";
        w.extend_from_slice(&(name2.len() as u64).to_le_bytes());
        w.extend_from_slice(name2);
        w.extend_from_slice(&1u32.to_le_bytes());
        w.extend_from_slice(&32u64.to_le_bytes());
        w.extend_from_slice(&8u32.to_le_bytes());
        w.extend_from_slice(&16u64.to_le_bytes());
        while w.len() % 32 != 0 {
            w.push(0);
        }
        for f in [1.0f32, 2.0, 3.0, 4.0] {
            w.extend_from_slice(&f.to_le_bytes());
        }
        w.extend_from_slice(&0x4000u16.to_le_bytes());
        for i in 0..32u8 {
            w.push(i);
        }
        let p = std::env::temp_dir().join(format!(
            "owl-varbuilder-test-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&p, &w).unwrap();
        p
    }

    /// Host 模式:解析 + Q8_0 反解 + shape 校验,零设备依赖
    #[test]
    fn host_get_roundtrip_and_shape_guard() {
        let p = write_synthetic_gguf();
        let vb = VarBuilderX::from_gguf_file_host(&p).unwrap();
        // 命中:shape 正确 → f32 内容对拍
        let t = vb.get_host([2, 2], "t_f32").unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        let mut want = Vec::new();
        for f in [1.0f32, 2.0, 3.0, 4.0] {
            want.extend_from_slice(&f.to_le_bytes());
        }
        assert_eq!(t.data, want);
        // shape 不符 = 结构化报错(元数据 [2,2] vs 请求 [4,1])
        assert!(vb.get_host([4, 1], "t_f32").is_err());
        // 缺失 = 结构化报错
        assert!(vb.get_host([2, 2], "nope").is_err());
        // Q8_0 → f32 反解(d=2.0,qs=[0..32] 对称量化)
        let q = vb.get_host([32], "t_q8_0").unwrap();
        assert_eq!(q.data.len(), 32 * 4);
        std::fs::remove_file(&p).ok();
    }

    /// 设备模式:with_pool 注入 → get → DynTensor(F32)→ D2H 对拍
    #[test]
    fn device_get_roundtrip() {
        use owl_iface::Device as _;
        let dev = owl_cuda::CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA");
        let p = write_synthetic_gguf();
        let pool = std::sync::Arc::new(
            dev.create_pool(owl_iface::PoolConfig {
                name: format!("vb-test-{}", std::process::id()),
                kind: owl_iface::PoolKind::Weights,
                bytes: 1 << 20,
            })
            .unwrap(),
        );
        let vb = VarBuilderX::from_gguf_file_host(&p)
            .unwrap()
            .with_pool(pool);
        let t = vb.get([2, 2], "t_f32").unwrap();
        assert_eq!(t.dtype(), DType::F32);
        assert_eq!(t.shape(), &[2, 2]);
        // D2H 对拍:落池内容 = GGUF 原值
        let mut host = [0f32; 4];
        unsafe {
            owl_cuda::ffi::sys::cuMemcpyDtoH_v2(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                t.device_ptr() as owl_cuda::ffi::sys::CUdeviceptr,
                16,
            )
            .result()
            .unwrap();
        }
        assert_eq!(host, [1.0, 2.0, 3.0, 4.0]);
        std::fs::remove_file(&p).ok();
    }

    // ---- HF safetensors 通道(真文件;缺失时跳过——CI 无此文件)----

    const ST_MODEL: &str =
        "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B/model.safetensors-00001-of-00001.safetensors";

    fn st_host_vb() -> Option<VarBuilderX> {
        if !std::path::Path::new(ST_MODEL).exists() {
            eprintln!("skip: 真模型文件不在本机({ST_MODEL})");
            return None;
        }
        Some(VarBuilderX::from_safetensors_file_host(ST_MODEL).unwrap())
    }

    /// host 通道:别名归一 + checkpoint 过滤 + 元数据面 + get_host 数值抽查
    #[test]
    fn safetensors_host_channel_alias_filter_values() {
        let Some(vb) = st_host_vb() else { return };
        // 归一:model.language_model.X → model.X(引擎请求路径)
        assert!(vb.has_key("model.embed_tokens.weight"));
        assert!(vb.has_key("model.layers.0.linear_attn.conv1d.weight"));
        assert!(vb.has_key("model.layers.3.self_attn.q_proj.weight"));
        assert!(vb.contains_tensor("model.norm.weight"));
        // 过滤:mtp. / model.visual. 查询 = 不存在而非报错
        assert!(!vb.has_key("mtp.norm.weight"));
        assert!(!vb.has_key("mtp.fc.weight"));
        assert!(!vb.has_key("model.visual.patch_embed.proj.weight"));
        assert!(!vb.contains_tensor("model.visual.merger.linear_fc1.weight"));
        // 元数据 shape:原生行主序,无反转
        assert_eq!(
            vb.tensor_shape("model.layers.3.self_attn.q_proj.weight"),
            Some(vec![4096, 1024])
        );
        assert_eq!(
            vb.tensor_shape("model.layers.0.linear_attn.conv1d.weight"),
            Some(vec![6144, 1, 4])
        );

        // get_host 数值抽查:与直接 SafeTensorsFile 读数位型一致(BF16→F32 扩位精确)
        let direct =
            crate::loader::safetensors::SafeTensorsFile::open(ST_MODEL).unwrap();

        // ① embed_tokens(别名路径;首行 1024 元素全对拍)
        let host = vb
            .get_host([248320, 1024], "model.embed_tokens.weight")
            .unwrap();
        assert_eq!(host.shape, vec![248320, 1024]);
        assert_eq!(host.data.len(), 248320 * 1024 * 4);
        let want = direct
            .tensor_f32("model.language_model.embed_tokens.weight")
            .unwrap();
        for (i, c) in host.data[..1024 * 4].chunks_exact(4).enumerate() {
            let g = f32::from_le_bytes(c.try_into().unwrap());
            assert_eq!(g, want[i], "embed 第 {i} 元素不符");
        }

        // ② GDN 层 conv1d [6144,1,4](中段一点)
        let conv = vb
            .get_host([6144, 1, 4], "model.layers.0.linear_attn.conv1d.weight")
            .unwrap();
        assert_eq!(conv.shape, vec![6144, 1, 4]);
        let want_conv = direct
            .tensor_f32("model.language_model.layers.0.linear_attn.conv1d.weight")
            .unwrap();
        let mid = 6144 / 2;
        assert_eq!(
            f32::from_le_bytes(conv.data[mid * 4..mid * 4 + 4].try_into().unwrap()),
            want_conv[mid]
        );

        // ③ attn 层 q_proj [4096,1024] = 8 头×256×2(门控加倍;头数定谳依据)
        let q = vb
            .get_host([4096, 1024], "model.layers.3.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q.shape, vec![4096, 1024]);

        // shape 不符 = 结构化报错
        assert!(vb.get_host([1, 2], "model.layers.3.self_attn.q_proj.weight").is_err());
        // 缺失 = 结构化报错
        assert!(vb.get_host([2], "nope").is_err());
    }

    /// 设备通道:真文件 → with_pool 落池,三键(GDN conv1d / attn q_proj / norm)
    /// D2H 对拍 = 直接读数一致
    #[test]
    fn safetensors_device_channel_loads() {
        let Some(vb_host) = st_host_vb() else { return };
        use owl_iface::Device as _;
        let dev =
            owl_cuda::CudaDevice::new(owl_cuda::test_device_ordinal()).expect("需要 CUDA 设备");
        let pool = std::sync::Arc::new(
            dev.create_pool(owl_iface::PoolConfig {
                name: format!("vb-st-test-{}", std::process::id()),
                kind: owl_iface::PoolKind::Weights,
                bytes: 64 << 20,
            })
            .unwrap(),
        );
        let paths = vec![std::path::PathBuf::from(ST_MODEL)];
        let vb = VarBuilderX::new(
            &crate::downloader::ModelPaths {
                tokenizer_filename: Default::default(),
                tokenizer_config_filename: Default::default(),
                config_filename: Default::default(),
                generation_config_filename: Default::default(),
                filenames: paths,
                auxiliary_filenames: vec![],
                chat_template_filename: None,
            },
            false,
            DType::F32,
            &dev,
        )
        .unwrap()
        .with_pool(pool);
        assert!(!vb.is_qvar_builder());

        let d2h = |t: &Tensor| -> Vec<f32> {
            let n = t.shape().iter().product::<usize>();
            let mut host = vec![0f32; n];
            unsafe {
                owl_cuda::ffi::sys::cuMemcpyDtoH_v2(
                    host.as_mut_ptr() as *mut std::ffi::c_void,
                    t.device_ptr() as owl_cuda::ffi::sys::CUdeviceptr,
                    n * 4,
                )
                .result()
                .unwrap();
            }
            host
        };

        // ① GDN conv1d(中段一点)
        let conv = vb
            .get([6144, 1, 4], "model.layers.0.linear_attn.conv1d.weight")
            .unwrap();
        assert_eq!(conv.dtype(), DType::F32);
        assert_eq!(conv.shape(), &[6144, 1, 4]);
        let direct =
            crate::loader::safetensors::SafeTensorsFile::open(ST_MODEL).unwrap();
        let want_conv = direct
            .tensor_f32("model.language_model.layers.0.linear_attn.conv1d.weight")
            .unwrap();
        let got_conv = d2h(&conv);
        assert_eq!(got_conv[6144 / 2], want_conv[6144 / 2]);

        // ② attn q_proj(首 16 点)
        let q = vb
            .get([4096, 1024], "model.layers.3.self_attn.q_proj.weight")
            .unwrap();
        assert_eq!(q.shape(), &[4096, 1024]);
        let want_q = direct
            .tensor_f32("model.language_model.layers.3.self_attn.q_proj.weight")
            .unwrap();
        for (i, g) in d2h(&q).into_iter().take(16).enumerate() {
            assert_eq!(g, want_q[i]);
        }

        // ③ norm [1024](全量)
        let norm = vb.get([1024], "model.norm.weight").unwrap();
        assert_eq!(norm.shape(), &[1024]);
        let want_norm = direct
            .tensor_f32("model.language_model.norm.weight")
            .unwrap();
        assert_eq!(d2h(&norm), want_norm);

        // host 参照物仍存活(避免 unused 警告;别名面一致性)
        assert!(vb_host.has_key("model.embed_tokens.weight"));
    }
}
