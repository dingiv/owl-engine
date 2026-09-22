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
    fn broadcast_left(&self, _left: impl Into<Shape>) -> Result<Tensor> {
        t3_unimpl("broadcast_left")
    }
    fn is_contiguous(&self) -> bool {
        true // S3:owl 无惰性布局,恒紧凑
    }
    fn dims3(&self) -> Result<(usize, usize, usize)> {
        t3_unimpl("dims3")
    }
    fn elem_count(&self) -> Result<usize> {
        t3_unimpl("elem_count")
    }
    fn permute(&self, order: &[usize]) -> Result<Tensor> {
        let _ = order;
        t3_unimpl("permute")
    }
    fn avg_pool2d_with_stride(&self, _kernel: usize, _stride: usize) -> Result<Tensor> {
        t3_unimpl("avg_pool2d_with_stride")
    }
    fn dequantize(&self, _device: &Device) -> Result<Tensor> {
        t3_unimpl("dequantize(量化 = marlin-ffi 路线)")
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
    pub fn silu_and_mul(x: &Tensor) -> Result<Tensor> { t3_unimpl("silu_and_mul") }
    pub fn sigmoid(x: &Tensor) -> Result<Tensor> { t3_unimpl("sigmoid") }
    pub fn cat(xs: &[Tensor], dim: usize) -> Result<Tensor> {
        let _ = (xs, dim);
        t3_unimpl("cat")
    }
    pub fn full(v: f64, shape: &[usize], _like: &Tensor) -> Result<Tensor> {
        let _ = (v, shape);
        t3_unimpl("full")
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
    /// 装载目标池(P 阶段注入;None = Host 测试模式)
    pool: Option<std::sync::Arc<owl_cuda::CudaPool>>,
    /// 设备句柄(device() 查询面)
    device: Option<Device>,
}

/// f32 → bf16 位型(RNE 舍入;host 位操作,无算术面)
fn f32_to_bf16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let lsb = (b >> 16) & 1;
    ((b + 0x7fff + lsb) >> 16) as u16
}

/// f32 → f16 位型(IEEE 754 half,RNE;查表-free 位算法)
fn f32_to_f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        // inf/nan
        return sign | 0x7c00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if e <= 0 {
        // 次正规(右移含隐藏位)
        if e < -10 {
            return sign;
        }
        let m = ((man | 0x0080_0000) >> (14 - e + 1)) as u16;
        let round = ((man >> (13 - e)) & 1) as u16;
        return sign | (m + round);
    }
    let half = ((e as u16) << 10) | ((man >> 13) as u16);
    let round = ((man >> 12) & 1) as u16;
    sign | (half + round)
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
        let gguf = if is_gguf && !model_pathes.filenames.is_empty() {
            Some(std::sync::Arc::new(
                crate::loader::gguf::GGufVarBuilder::from_gguf_files(&model_pathes.filenames)?,
            ))
        } else {
            None
        };
        Ok(Self {
            module_path: String::new(),
            weight_paths: Some(model_pathes.filenames.clone()),
            is_gguf,
            gguf,
            pool: None,
            device: Some(device.clone()),
        })
    }

    /// 注入装载目标池(P 阶段;裁决 5:分配入口在 Pool)。
    pub fn with_pool(mut self, pool: std::sync::Arc<owl_cuda::CudaPool>) -> Self {
        self.pool = Some(pool);
        self
    }

    pub fn from_gguf_file_host<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let vb = crate::loader::gguf::GGufVarBuilder::from_gguf(path)?;
        Ok(Self {
            module_path: String::new(),
            weight_paths: None,
            is_gguf: true,
            gguf: Some(std::sync::Arc::new(vb)),
            pool: None,
            device: None,
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
            pool: self.pool.clone(),
            device: self.device.clone(),
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
        self.gguf
            .as_ref()
            .map(|g| g.contains_key(&self.full_name(name)))
            .unwrap_or(false)
    }
    pub fn contains_tensor(&self, name: &str) -> bool {
        self.has_key(name)
    }
    pub fn get_no_shape(&self, name: &str) -> Result<Tensor> {
        // 无 shape 请求 = 按元数据原 shape 反解(GGUF 维序反转回 owl 行主序)
        let full = self.full_name(name);
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
            .ok_or_else(|| Error::Msg("VarBuilderX: 非 GGUF 路径(safetensors 通道 T3 回填)".into()))
    }

    fn pool_ref(&self) -> Result<std::sync::Arc<owl_cuda::CudaPool>> {
        self.pool
            .clone()
            .ok_or_else(|| Error::Msg(
                "VarBuilderX: 未注入装载池(Host 测试模式;设备 get 需 with_pool)".into(),
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
        let gg = self.gguf_ref()?;
        let full = self.full_name(name);
        // shape 校验真实:请求(owl 行主序)→ GGUF 反转维序后比对
        let mut want = s.into().dims().to_vec();
        want.reverse();
        let raw = gg.get(&want, &full)?;
        let mut owl_shape = raw.shape.clone();
        owl_shape.reverse();

        // 反解到 f32(除 F32 直通)→ 目标 dtype 位型 → 池直连
        let f32s = if raw.dtype == crate::loader::gguf::GgmlDType::F32 {
            raw.raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect::<Vec<f32>>()
        } else {
            crate::loader::gguf::dequantize_to_f32(raw.dtype, &raw.shape, &raw.raw)?
        };
        let pool = self.pool_ref()?;
        use owl_nn::TensorPoolOps;
        match dtype {
            DType::F32 => {
                let t = pool.from_vec_tensor::<f32>(&owl_shape, f32s)?;
                Ok(Tensor::from_f32(&t))
            }
            DType::BF16 => {
                let v: Vec<owl_nn::Bf16> =
                    f32s.iter().map(|&f| owl_nn::Bf16(f32_to_bf16_bits(f))).collect();
                let t = pool.from_vec_tensor::<owl_nn::Bf16>(&owl_shape, v)?;
                Ok(Tensor::from_bf16(&t))
            }
            DType::F16 => {
                let v: Vec<owl_nn::F16> =
                    f32s.iter().map(|&f| owl_nn::F16(f32_to_f16_bits(f))).collect();
                let t = pool.from_vec_tensor::<owl_nn::F16>(&owl_shape, v)?;
                Ok(Tensor::from_f16(&t))
            }
            _ => Err(Error::Msg(format!(
                "VarBuilderX: 目标 dtype {dtype} 权重装载未开(U8/U32/I64 经 kvcache/索引专用通道;F8 量化 = marlin-ffi 路线)"
            ))),
        }
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
        let gg = self.gguf_ref()?;
        let full = self.full_name(name);
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
        let _ = x;
        unimplemented!("T3 kernel 回填: rms_norm")
    }
}

// ============================================================================
// vendor 占位(attention_rs / flashinfer;归宿裁决待定)
// ============================================================================

/// vendor kernel/缓存面占位。design:三选一(自研 kernel 进 owl-kernels /
/// 保留 vendor 依赖 / 换 flashinfer)在 T3 运行里程碑前裁决。
pub mod vendor {
    // fused rope(= attention_rs::fused_rope::FusedRope)
    pub mod fused_rope {
        use crate::Result;
        use super::super::Tensor;
        pub fn apply_inplace(
            _q: &Tensor,
            _k: &Tensor,
            _cos: &Tensor,
            _sin: &Tensor,
            _positions: &Tensor,
            _is_rope_i: bool,
        ) -> Result<()> {
            Err(crate::Error::Msg("vendor fused_rope::apply_inplace: T3 kernel 回填".into()))
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
    /// Mamba/GDN 状态缓存(= attention_rs::mamba_cache::MambaCache)
    pub struct MambaCache;
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
}
