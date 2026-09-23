//! Linear 层族(= xinfer layers/linear.rs 直译,编译先行口径)。
//!
//! 路线分派:
//! - **未量化 Linear**:控制流骨架完整(broadcast/t/matmul/bias),
//!   设备体走 OwlTensor(T3 回填);
//! - **GGUF/ISQ**(QLinear::new/new_fused/from_linear_x*)、
//!   **FP8/MXFP4/NVFP4**(LnFp8/LnMxfp4/LnNvfp4)、**WNA16**:
//!   类型面 + 分派结构保留,体 = `unimplemented!`
//!   (量化 = marlin-ffi 路线,loader/权重通道 T3 回填)。

use super::wna16::WNA16;
use super::{DType, Device, Module, OwlTensor, Result, Shard, Shape, Tensor, VarBuilderX};
use crate::bail;
use crate::config::QuantConfig;
// GGUF VB 占位(loader::gguf 归 loader worker 所有;接线期换其真实类型)
#[derive(Clone)]
pub struct GgufVarBuilderStub;
use std::cell::Cell;

// FIXME: 不允许使用全局变量。包装在结构体里面。给一个构造函数，给外面让使用者来构造。
thread_local! {
    static LINEAR_IS_PREFILL: Cell<bool> = const { Cell::new(false) };
}

pub struct LinearPrefillGuard {
    prev: bool,
}

impl Drop for LinearPrefillGuard {
    fn drop(&mut self) {
        LINEAR_IS_PREFILL.with(|flag| flag.set(self.prev));
    }
}

pub fn set_linear_is_prefill(is_prefill: bool) -> LinearPrefillGuard {
    let prev = LINEAR_IS_PREFILL.with(|flag| {
        let prev = flag.get();
        flag.set(is_prefill);
        prev
    });
    LinearPrefillGuard { prev }
}

pub fn linear_is_prefill() -> bool {
    LINEAR_IS_PREFILL.with(|flag| flag.get())
}

pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard {
        dim,
        rank,
        world_size,
    }
}

#[derive(Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    type Output = Tensor;
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims()?;
        // w = seq_len>1 时 broadcast_left((b1, seq_len)) 后转置;否则直接转置
        let w = match dims.as_slice() {
            [b1, seq_len, _, _] if *seq_len > 1 => {
                self.weight.broadcast_left((*b1, *seq_len))?.t()?
            }
            [bsize, seq_len, _] if *seq_len > 1 => self.weight.broadcast_left(*bsize)?.t()?,
            _ => self.weight.t()?,
        };
        let x = match dims.as_slice() {
            [bsize, seq_len, dim1, dim2] if *seq_len > 1 => x.matmul(&w)?,
            [bsize, seq_len, _] if *seq_len > 1 => x.matmul(&w)?,
            [bsize, seq_len, dim] => {
                let wdim = w.dims()?[w.dims()?.len() - 1];
                x.reshape((bsize * seq_len, *dim))?
                    .matmul(&w)?
                    .reshape((*bsize, *seq_len, wdim))?
            }
            _ => x.matmul(&w)?,
        };

        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: &VarBuilderX,
    shard: Shard,
    dtype: DType,
) -> Result<Linear> {
    let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
    let weight = if weight.dtype() != dtype {
        weight.to_dtype(dtype)?
    } else {
        weight
    };
    Ok(Linear::new(weight, None))
}

pub fn linear_no_bias_merged(
    num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    vb: &VarBuilderX,
    shards: Shard,
    dtype: DType,
) -> Result<Linear> {
    let sd = shard(shards.dim + 1, shards.rank, shards.world_size);
    let weight = vb.get_with_hints((num_experts, out_dim, in_dim), "weight", sd)?;
    let weight = if weight.dtype() != dtype {
        weight.to_dtype(dtype)?
    } else {
        weight
    };
    Ok(Linear::new(weight, None))
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    vb: &VarBuilderX,
    shard: Shard,
    dtype: DType,
) -> Result<Linear> {
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", shard)?;
    let ws = if ws.dtype() != dtype {
        ws.to_dtype(dtype)?
    } else {
        ws
    };
    let bs = vb.get((out_dim,), "bias");
    let bs = match bs {
        Ok(bs) => {
            let bs = if shard.world_size > 1 {
                let dim_size = bs.dim(0)?;
                let start = shard.rank * (dim_size / shard.world_size);
                bs.narrow(0, start, dim_size / shard.world_size)?
                    .contiguous()?
            } else {
                bs
            };
            let bs = if bs.dtype() != dtype {
                bs.to_dtype(dtype)?
            } else {
                bs
            };
            Some(bs)
        }
        Err(_) => None,
    };

    Ok(Linear::new(ws, bs))
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: &VarBuilderX,
    shard: Shard,
    dtype: DType,
) -> Result<Linear> {
    if bias {
        linear(in_dim, out_dim, vb, shard, dtype)
    } else {
        linear_no_bias(in_dim, out_dim, vb, shard, dtype)
    }
}

// ---- 量化族:类型面 + 分派骨架(体 = marlin-ffi,T3 回填)----

#[derive(Clone)]
pub struct QLinear {
    pub inner: Option<QMatMul>,
    pub bias: Option<Tensor>,
    pub wna16: Option<WNA16>,
    pub dtype: DType,
}

/// QMatMul 双形态(= candle_core::quantized::QMatMul;类型面)
#[derive(Clone)]
pub enum QMatMul {
    /// 未量化回退(权重量化不可行时保持原张量)
    Tensor(Tensor),
    /// GGUF 量化权重(Arc<QTensor>;T3 经 loader::gguf 回填)
    QTensor(std::sync::Arc<super::QTensor>),
}

#[allow(dead_code)] // ISQ 分派面,marlin-ffi 接线期消费
impl QLinear {
    fn ggml_dtype_from_str(quant: &str) -> super::GgmlDType {
        match quant.to_lowercase().as_str() {
            "q40" | "q4_0" => super::GgmlDType::Q4_0,
            "q4" | "q41" | "q4_1" => super::GgmlDType::Q4_1,
            "q50" | "q5_0" => super::GgmlDType::Q5_0,
            "q5" | "q51" | "q5_1" => super::GgmlDType::Q5_1,
            "q8" | "q80" | "q8_0" => super::GgmlDType::Q8_0,
            _ => super::GgmlDType::Q4K,
        }
    }

    pub fn native_quantize_supported(
        _last_dim: usize,
        _quant: &str,
        _device: &Device,
    ) -> Result<bool> {
        // ISQ 原生量化路径:GGUF writer = loader T3 回填;编译口径恒 false
        Ok(false)
    }

    fn local_last_dim(in_dim: usize, shards: Shard) -> usize {
        if shards.world_size > 1 && shards.dim == 1 {
            in_dim / shards.world_size
        } else {
            in_dim
        }
    }

    pub fn new(
        _in_dim: usize,
        _out_dim: usize,
        _vb: &GgufVarBuilderStub,
        _shards: Shard,
        _dtype: DType,
    ) -> Result<Self> {
        unimplemented!("QLinear::new(GGUF 装载) = loader/gguf T3 回填")
    }

    pub fn new_fused(
        _num_experts: usize,
        _in_dim: usize,
        _out_dim: usize,
        _vb: &GgufVarBuilderStub,
        _shards: Shard,
        _dtype: DType,
    ) -> Result<Self> {
        unimplemented!("QLinear::new_fused(GGUF fused MoE 装载) = loader T3 回填")
    }

    pub fn from_qparts_x(_w: super::QTensor, _b: Option<Tensor>, _dtype: DType) -> Result<Self> {
        unimplemented!("QLinear::from_qparts_x = GGUF/QTensor T3 回填")
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        unimplemented!("QLinear::dequantize = marlin-ffi 反量化,T3 回填")
    }

    /// in-situ quantization(ISQ):量化 = marlin-ffi 路线,运行里程碑
    pub fn from_linear_x(_linear: Linear, _quant: String, _dtype: DType) -> Result<Self> {
        unimplemented!("QLinear::from_linear_x(ISQ)= marlin-ffi,T3 回填")
    }

    pub fn from_linear_x_on_device(
        _linear: Linear,
        _quant: String,
        _dtype: DType,
        _device: &Device,
    ) -> Result<Self> {
        unimplemented!("QLinear::from_linear_x_on_device(ISQ)= marlin-ffi,T3 回填")
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn bias_mut(&mut self) -> Option<&mut Tensor> {
        self.bias.as_mut()
    }
}

impl Module for QLinear {
    type Output = Tensor;
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if let Some(wna16) = &self.wna16 {
            wna16.forward(x)
        } else if self.inner.is_some() {
            unimplemented!("QLinear::forward(QMatMul)= GGUF kernel,T3 回填")
        } else {
            bail!("Invalid quantization type!")
        }
    }
}

impl QLinear {
    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        let _ = (x, ids);
        if self.inner.is_some() {
            unimplemented!("QLinear::indexed_moe_forward = GGUF moe kernel,T3 回填")
        } else {
            bail!("Invalid quantization type!")
        }
    }
}

#[derive(Clone)]
pub enum LinearX {
    Linear(Linear),
    QLinear(QLinear),
    LnFp8(LnFp8),
    LnMxfp4(LnMxfp4),
    LnNvfp4(LnNvfp4),
}

impl Module for LinearX {
    type Output = Tensor;
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(ln) => ln.forward(x),
            Self::QLinear(ln) => ln.forward(x),
            Self::LnMxfp4(ln) => ln.forward(x),
            Self::LnNvfp4(ln) => ln.forward(x),
            Self::LnFp8(ln) => ln.forward(x),
        }
    }
}

impl LinearX {
    /// S2.5-B:该 linear 是否为 W4A8(fused_silu_int8 prequant 可用)
    pub fn w4a8_active(&self) -> bool {
        match self {
            Self::QLinear(q) => q.wna16.as_ref().map(|w| w.is_w4a8()).unwrap_or(false),
            _ => false,
        }
    }

    /// S2.5-B:外部 prequant W4A8 matmul(见 wna16::WNA16::forward_prequant)
    pub fn forward_prequant(&self, xq: &Tensor, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::QLinear(q) => match &q.wna16 {
                Some(w) => w.forward_prequant(xq, xs),
                None => bail!("forward_prequant: QLinear without WNA16"),
            },
            _ => bail!("forward_prequant: non-W4A16 linear variant"),
        }
    }

    pub fn forward_dense_dtype_compatible(&self, x: &Tensor) -> Result<Tensor> {
        if let Self::Linear(linear) = self {
            let weight_dtype = linear.weight().dtype();
            if weight_dtype != x.dtype() {
                return linear
                    .forward(&x.to_dtype(weight_dtype)?)?
                    .to_dtype(x.dtype());
            }
        }
        self.forward(x)
    }

    pub fn fp8_weight_scale(&self) -> Option<(&Tensor, &Tensor)> {
        match self {
            Self::LnFp8(ln) => Some((&ln.weight, &ln.weight_scale)),
            _ => None,
        }
    }

    pub fn as_nvfp4(&self) -> Option<&LnNvfp4> {
        match self {
            Self::LnNvfp4(ln) => Some(ln),
            _ => None,
        }
    }

    pub fn dense_weight(&self) -> Result<&Tensor> {
        match self {
            Self::Linear(ln) => Ok(ln.weight()),
            _ => bail!("dense weight requested from a quantized linear layer"),
        }
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::Linear(_) => {
                panic!("No supported!")
            }
            Self::QLinear(ln) => ln.indexed_moe_forward(x, ids),
            Self::LnFp8(_) => panic!("LnFp8 does not support indexed_moe_forward yet"),
            Self::LnMxfp4(_) => panic!("LnMxfp4 does not support indexed_moe_forward yet"),
            Self::LnNvfp4(_) => panic!("LnNvfp4 does not support indexed_moe_forward yet"),
        }
    }

    pub fn new(weight: Tensor, bias: Option<Tensor>, quant: &Option<String>) -> Result<Self> {
        let dtype = weight.dtype();
        let ln = Linear::new(weight, bias);
        if let Some(quantized_type) = quant {
            Ok(Self::QLinear(QLinear::from_linear_x(
                ln,
                quantized_type.clone(),
                dtype,
            )?))
        } else {
            Ok(Self::Linear(ln))
        }
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        match self {
            Self::Linear(_) => {
                panic!("Unquantized tensor unable to be dequantized!")
            }
            Self::QLinear(ln) => ln.dequantize(),
            Self::LnFp8(_) => panic!("LnFp8 unable to be dequantized"),
            Self::LnMxfp4(_) => panic!("LnMxfp4 unable to be dequantized"),
            Self::LnNvfp4(_) => panic!("LnNvfp4 unable to be dequantized"),
        }
    }
}

// ---- scale 探测面(布尔逻辑真实,探测实现随 loader 回填)----

#[allow(dead_code)] // 量化分派树,marlin-ffi 接线期消费
fn has_fp4_scale_tensors(vb: &VarBuilderX, is_mlx_nvfp4: bool) -> bool {
    if is_mlx_nvfp4 {
        vb.contains_tensor("weight") && vb.contains_tensor("scales")
    } else {
        vb.contains_tensor("weight_scale")
            || vb.contains_tensor("weight_scale_2")
            || vb.contains_tensor("weight_global_scale")
            || vb.contains_tensor("weight_packed")
            || vb.contains_tensor("blocks")
    }
}

#[allow(dead_code)] // 同上
fn has_fp8_tensors(vb: &VarBuilderX) -> bool {
    vb.contains_tensor("weight_scale")
        || vb.contains_tensor("weight_scale_inv")
        || vb.contains_tensor("scale")
}

pub fn linear_x(
    in_dim: usize,
    out_dim: usize,
    vbx: &VarBuilderX,
    shards: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    if vbx.is_gguf() {
        unimplemented!("linear_x GGUF 分支 = loader/gguf T3 回填")
    }
    match quant_cfg {
        Some(cfg) if !cfg.quant_method.is_empty() => {
            // fp8/mxfp4/nvfp4/compressed-tensors:量化装载 = marlin-ffi,
            // 分派决策树结构保留(should_skip 语义 = config::should_skip_module)
            if cfg.should_skip_module(&vbx.module_path()) {
                let ln = linear_no_bias(in_dim, out_dim, vbx, shards, dtype)?;
                return Ok(LinearX::Linear(ln));
            }
            unimplemented!(
                "linear_x 量化分支({})= marlin-ffi 装载通道,T3 回填",
                cfg.quant_method
            )
        }
        _ => {
            let _ = quant;
            let ln = linear_no_bias(in_dim, out_dim, vbx, shards, dtype)?;
            Ok(LinearX::Linear(ln))
        }
    }
}

pub fn linear_no_bias_x(
    in_dim: usize,
    out_dim: usize,
    vbx: &VarBuilderX,
    shards: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    linear_x(in_dim, out_dim, vbx, shards, quant_cfg, quant, dtype)
}

pub fn linear_no_bias_merged_x(
    _num_experts: usize,
    in_dim: usize,
    out_dim: usize,
    vbx: &VarBuilderX,
    shards: Shard,
    _quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    let _ = (in_dim, out_dim, quant);
    if vbx.is_gguf() {
        unimplemented!("linear_no_bias_merged_x GGUF 分支 = loader T3 回填")
    }
    let ln = linear_no_bias_merged(_num_experts, in_dim, out_dim, vbx, shards, dtype)?;
    let _ = shards;
    Ok(LinearX::Linear(ln))
}

pub fn linear_b_x(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: &VarBuilderX,
    shard: Shard,
    quant_cfg: &Option<QuantConfig>,
    quant: &Option<String>,
    dtype: DType,
) -> Result<LinearX> {
    if bias {
        linear_x(in_dim, out_dim, vb, shard, quant_cfg, quant, dtype)
    } else {
        linear_no_bias_x(in_dim, out_dim, vb, shard, quant_cfg, quant, dtype)
    }
}

// ---- FP8 / MXFP4 / NVFP4:结构壳(体 = vendor fp8_linear 等,T3 回填)----

#[derive(Clone)]
pub struct LnFp8 {
    pub weight: Tensor,
    pub weight_scale: Tensor,
    pub weight_scale_cutlass: Option<Tensor>,
    /// Static ModelOpt FP8 activation scale (usually amax / 448).
    pub input_scale: Option<f32>,
    pub bias: Option<Tensor>,
    pub weight_block_size: Vec<usize>,
    pub sm_version: usize,
    pub ue8m0: bool,
}

impl Module for LnFp8 {
    type Output = Tensor;
    fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        unimplemented!("LnFp8::forward = vendor fp8_linear,T3 回填")
    }
}

#[derive(Clone)]
pub struct LnMxfp4 {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl LnMxfp4 {
    pub fn load(
        _in_dim: usize,
        _out_dim: usize,
        _vb: &VarBuilderX,
        _shards: Shard,
        _bias: bool,
    ) -> Result<Self> {
        unimplemented!("LnMxfp4::load = MLX NVFP4 量化装载,T3 回填")
    }

    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        unimplemented!("LnMxfp4::forward = vendor kernel,T3 回填")
    }
}

#[derive(Clone)]
pub struct LnNvfp4 {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl LnNvfp4 {
    pub fn load(
        _in_dim: usize,
        _out_dim: usize,
        _vb: &VarBuilderX,
        _shards: Shard,
        _bias: bool,
    ) -> Result<Self> {
        unimplemented!("LnNvfp4::load = NVFP4 量化装载,T3 回填")
    }

    pub fn load_mlx(
        _in_dim: usize,
        _out_dim: usize,
        _vb: &VarBuilderX,
        _shards: Shard,
        _bias: bool,
    ) -> Result<Self> {
        unimplemented!("LnNvfp4::load_mlx = MLX NVFP4 装载,T3 回填")
    }

    pub fn forward(&self, _x: &Tensor) -> Result<Tensor> {
        unimplemented!("LnNvfp4::forward = vendor kernel,T3 回填")
    }
}

// Shape 引用(保留类型导入面;编译口径)
#[allow(dead_code)]
type _ShapeUse = Shape;
