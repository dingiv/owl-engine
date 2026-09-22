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

//pub mod attention;
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

use crate::error::Result;
use owl_cuda::CudaDevice;
use owl_nn::DynTensor;

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
pub trait Dim: Copy + Sized {}
impl Dim for usize {}
impl Dim for D {}

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
}

/// T3 回填占位(编译先行口径;运行里程碑经 owl-kernels/marlin-ffi 替换为真实 kernel)
fn t3_unimpl(_name: &str) -> ! {
    unimplemented!("T3 kernel 回填")
}

impl OwlTensor for Tensor {
    // 所有设备体 = T3 kernel 回填(编译先行口径);控制流骨架由调用方保留。
    fn reshape(&self, _shape: impl Into<Shape>) -> Result<Tensor> { t3_unimpl("reshape") }
    fn narrow(&self, _dim: impl Dim, _start: usize, _len: usize) -> Result<Tensor> {
        t3_unimpl("narrow")
    }
    fn transpose(&self, _d1: impl Dim, _d2: impl Dim) -> Result<Tensor> { t3_unimpl("transpose") }
    fn t(&self) -> Result<Tensor> { t3_unimpl("t") }
    fn t2(&self) -> Result<Tensor> { t3_unimpl("t2") }
    fn unsqueeze(&self, _dim: impl Dim) -> Result<Tensor> { t3_unimpl("unsqueeze") }
    fn squeeze(&self, _dim: impl Dim) -> Result<Tensor> { t3_unimpl("squeeze") }
    fn contiguous(&self) -> Result<Tensor> { t3_unimpl("contiguous") }
    fn broadcast_as(&self, _shape: impl Into<Shape>) -> Result<Tensor> { t3_unimpl("broadcast_as") }
    fn chunk(&self, _c: usize, _dim: impl Dim) -> Result<Vec<Tensor>> { t3_unimpl("chunk") }
    fn split(&self, _s: usize, _dim: impl Dim) -> Result<Vec<Tensor>> { t3_unimpl("split") }
    fn select(&self, _dim: impl Dim, _i: usize) -> Result<Tensor> { t3_unimpl("select") }
    fn add(&self, _r: &Tensor) -> Result<Tensor> { t3_unimpl("add") }
    fn sub(&self, _r: &Tensor) -> Result<Tensor> { t3_unimpl("sub") }
    fn mul(&self, _r: &Tensor) -> Result<Tensor> { t3_unimpl("mul") }
    fn div(&self, _r: &Tensor) -> Result<Tensor> { t3_unimpl("div") }
    fn broadcast_add(&self, _r: &Tensor) -> Result<Tensor> { t3_unimpl("broadcast_add") }
    fn broadcast_mul(&self, _r: &Tensor) -> Result<Tensor> { t3_unimpl("broadcast_mul") }
    fn broadcast_div(&self, _r: &Tensor) -> Result<Tensor> { t3_unimpl("broadcast_div") }
    fn broadcast_sub(&self, _r: &Tensor) -> Result<Tensor> { t3_unimpl("broadcast_sub") }
    fn matmul(&self, _r: &Tensor) -> Result<Tensor> { t3_unimpl("matmul") }
    fn affine(&self, _a: f64, _b: f64) -> Result<Tensor> { t3_unimpl("affine") }
    fn where_cond(&self, _t: &Tensor, _f: &Tensor) -> Result<Tensor> { t3_unimpl("where_cond") }
    fn sum(&self, _dim: impl Dim) -> Result<Tensor> { t3_unimpl("sum") }
    fn sum_all(&self) -> Result<Tensor> { t3_unimpl("sum_all") }
    fn max(&self, _dim: impl Dim) -> Result<Tensor> { t3_unimpl("max") }
    fn min(&self, _dim: impl Dim) -> Result<Tensor> { t3_unimpl("min") }
    fn argmax(&self, _dim: impl Dim) -> Result<Tensor> { t3_unimpl("argmax") }
    fn mean(&self, _dim: impl Dim) -> Result<Tensor> { t3_unimpl("mean") }
    fn cumsum(&self, _dim: impl Dim) -> Result<Tensor> { t3_unimpl("cumsum") }
    fn exp(&self) -> Result<Tensor> { t3_unimpl("exp") }
    fn sqrt(&self) -> Result<Tensor> { t3_unimpl("sqrt") }
    fn sin(&self) -> Result<Tensor> { t3_unimpl("sin") }
    fn cos(&self) -> Result<Tensor> { t3_unimpl("cos") }
    fn tanh(&self) -> Result<Tensor> { t3_unimpl("tanh") }
    fn abs(&self) -> Result<Tensor> { t3_unimpl("abs") }
    fn neg(&self) -> Result<Tensor> { t3_unimpl("neg") }
    fn sqr(&self) -> Result<Tensor> { t3_unimpl("sqr") }
    fn log(&self) -> Result<Tensor> { t3_unimpl("log") }
    fn clamp(&self, _min: f64, _max: f64) -> Result<Tensor> { t3_unimpl("clamp") }
    fn powf(&self, _e: f64) -> Result<Tensor> { t3_unimpl("powf") }
    fn reciprocal(&self) -> Result<Tensor> { t3_unimpl("reciprocal") }
    fn gelu(&self) -> Result<Tensor> { t3_unimpl("gelu") }
    fn silu(&self) -> Result<Tensor> { t3_unimpl("silu") }
    fn softmax(&self, _dim: impl Dim) -> Result<Tensor> { t3_unimpl("softmax") }
    fn softmax_last_dim(&self) -> Result<Tensor> { t3_unimpl("softmax_last_dim") }
    fn index_select(&self, _dim: impl Dim, _idx: &Tensor) -> Result<Tensor> {
        t3_unimpl("index_select")
    }
    fn gather(&self, _dim: impl Dim, _idx: &Tensor) -> Result<Tensor> { t3_unimpl("gather") }
    fn scatter(&self, _dim: impl Dim, _idx: &Tensor, _src: &Tensor) -> Result<Tensor> {
        t3_unimpl("scatter")
    }
    fn scatter_add(&self, _idx: &Tensor, _src: &Tensor, _dim: impl Dim) -> Result<Tensor> {
        t3_unimpl("scatter_add")
    }
    fn repeat(&self, _shape: impl Into<Shape>) -> Result<Tensor> { t3_unimpl("repeat") }
    fn repeat_interleave(&self, _repeats: usize, _dim: impl Dim) -> Result<Tensor> {
        t3_unimpl("repeat_interleave")
    }
    fn to_dtype(&self, _dtype: DType) -> Result<Tensor> { t3_unimpl("to_dtype") }
    fn to_device(&self, _device: &Device) -> Result<Tensor> { t3_unimpl("to_device") }
    fn device(&self) -> Device {
        t3_unimpl("device")
    }
    fn dim(&self, _i: usize) -> Result<usize> {
        t3_unimpl("dim")
    }
    fn dims(&self) -> Result<Vec<usize>> {
        t3_unimpl("dims")
    }
    fn dims2(&self) -> Result<(usize, usize)> {
        t3_unimpl("dims2")
    }
    fn flatten_all(&self) -> Result<Tensor> {
        t3_unimpl("flatten_all")
    }
    fn to_scalar<T: Copy>(&self) -> Result<T> {
        t3_unimpl("to_scalar")
    }
    fn apply_rotary_emb_qkv(
        &self,
        _cos: &Tensor,
        _sin: &Tensor,
        _n: usize,
    ) -> Result<Tensor> {
        t3_unimpl("apply_rotary_emb_qkv")
    }
}

// ============================================================================
// 自由算子(candle_nn::ops / 层内 rms_norm 等)
// ============================================================================

pub mod ops {
    #![allow(unused_variables)]
    use super::{t3_unimpl, DType, Result, Tensor};

    pub fn silu(x: &Tensor) -> Result<Tensor> { t3_unimpl("silu") }
    pub fn gelu(x: &Tensor) -> Result<Tensor> { t3_unimpl("gelu") }
    pub fn softmax(x: &Tensor, dim: impl super::Dim) -> Result<Tensor> { t3_unimpl("softmax") }
    pub fn softmax_last_dim(x: &Tensor) -> Result<Tensor> { t3_unimpl("softmax_last_dim") }
    pub fn gelu_erf(x: &Tensor) -> Result<Tensor> { t3_unimpl("gelu_erf") }
    pub fn tanh(x: &Tensor) -> Result<Tensor> { t3_unimpl("tanh") }
    pub fn sigmoid(x: &Tensor) -> Result<Tensor> { t3_unimpl("sigmoid") }
    pub fn silu_and_mul(x: &Tensor) -> Result<Tensor> { t3_unimpl("silu_and_mul") }
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
        let _ = (start, end);
        t3_unimpl("arange")
    }
    pub fn eye(n: usize, m: usize, _dtype: DType, _device: &Device) -> Result<Tensor> {
        let _ = (n, m);
        t3_unimpl("eye")
    }
    pub fn from_vec<T: Copy>(
        _v: Vec<T>,
        _shape: impl Into<Shape>,
        _device: &Device,
    ) -> Result<Tensor> {
        t3_unimpl("from_vec")
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

// ============================================================================
// VarBuilderX 桩(权重装载通道 T3 回填;类型面照抄 xinfer)
// ============================================================================

/// GGUF/safetensors 双路 var builder 包装(= xinfer `models::layers::VarBuilderX`)。
/// 编译先行:仅保留类型面与路径拼接语义;`get`/`get_with_hints_dtype` 为
/// T3 权重装载通道回填前的 unimplemented。
#[derive(Clone)]
pub struct VarBuilderX {
    /// 模块路径(前缀)
    pub module_path: String,
    /// 底层权重文件列表
    pub weight_paths: Option<Vec<std::path::PathBuf>>,
    /// 是否 GGUF(Q) 路径
    pub is_gguf: bool,
}

impl VarBuilderX {
    /// 占位构造(T3 权重装载通道回填:safetensors/GGUF 打开)
    pub fn new(
        _model_pathes: &crate::downloader::ModelPaths,
        is_gguf: bool,
        _dtype: DType,
        _device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            module_path: String::new(),
            weight_paths: None,
            is_gguf,
        })
    }

    pub fn from_gguf_file<P: AsRef<std::path::Path>>(_path: P, _device: &Device) -> Result<Self> {
        Ok(Self {
            module_path: String::new(),
            weight_paths: None,
            is_gguf: true,
        })
    }

    pub fn is_var_builder(&self) -> bool {
        !self.is_gguf
    }
    pub fn is_qvar_builder(&self) -> bool {
        self.is_gguf
    }
    pub fn device(&self) -> Device {
        unimplemented!("T3 权重装载通道回填: device 绑定")
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
        }
    }

    pub fn aux(&self) -> Option<VarBuilderX> {
        None // T3 回填
    }
    pub fn gguf_path(&self) -> Option<&str> {
        None // T3 回填
    }
    pub fn weight_paths(&self) -> Option<Vec<std::path::PathBuf>> {
        self.weight_paths.clone()
    }
    pub fn cpu_var_builder(&self) -> Option<VarBuilderX> {
        None // T3 回填
    }
    pub fn module_path(&self) -> &str {
        &self.module_path
    }
    pub fn has_key(&self, _name: &str) -> bool {
        false // T3 回填
    }
    pub fn tensor_shape(&self, _name: &str) -> Option<Vec<usize>> {
        None // T3 回填
    }

    pub fn get_with_hints_dtype(
        &self,
        _s: impl Into<Shape>,
        _name: &str,
        _shard: Shard,
        _dtype: DType,
    ) -> Result<Tensor> {
        unimplemented!("T3 权重装载通道回填: get_with_hints_dtype")
    }
    pub fn get(&self, _s: impl Into<Shape>, _name: &str) -> Result<Tensor> {
        unimplemented!("T3 权重装载通道回填: get")
    }
    pub fn get_with_hints(&self, _hints: &[&str], _s: impl Into<Shape>, _name: &str) -> Result<Tensor> {
        unimplemented!("T3 权重装载通道回填: get_with_hints")
    }
}

/// 分片枚举(= candle_nn::var_builder::Shard;类型面)
#[derive(Copy, Clone, Debug, Default)]
pub enum Shard {
    #[default]
    None,
    Row(usize),
    Col(usize),
}

// ============================================================================
// candle_nn 容器桩(Linear/Conv/Embedding/Module/...)
// ============================================================================

/// 线性层(= candle_nn::Linear;权重 T3 回填)
#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.weight).and_then(|y| {
            match &self.bias {
                Some(b) => y.broadcast_add(b),
                None => Ok(y),
            }
        })
    }
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
    pub fn n_in(&self) -> Option<usize> {
        None // T3 回填
    }
    pub fn n_out(&self) -> Option<usize> {
        None // T3 回填
    }
}

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

impl Conv {
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
        let _ = &self.weight;
        unimplemented!("T3 kernel 回填: embedding lookup")
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

#[derive(Clone)]
pub struct RmsNorm {
    pub weight: Option<Tensor>,
    pub eps: f64,
}

// ============================================================================
// vendor 占位(attention_rs / flashinfer;归宿裁决待定)
// ============================================================================

/// vendor kernel/缓存面占位。design:三选一(自研 kernel 进 owl-kernels /
/// 保留 vendor 依赖 / 换 flashinfer)在 T3 运行里程碑前裁决。
pub mod vendor {
    /// 分页注意力句柄(= attention_rs::PagedAttention)
    pub struct PagedAttention;
    /// Mamba/GDN 状态缓存(= attention_rs::mamba_cache::MambaCache)
    pub struct MambaCache;
    /// 输入元数据(= attention_rs::InputMetadata;字段面 T3 按调用点补齐)
    #[derive(Clone, Default)]
    pub struct InputMetadata {
        pub seqlens: Vec<usize>,
        pub context_lens: Vec<usize>,
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
