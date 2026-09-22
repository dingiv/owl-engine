//! 张量并行线性层包装(= xinfer layers/distributed.rs 直译,编译先行口径)。
//!
//! MVP 语义:单进程单实例(world_size=1)下 all_reduce = 恒等;
//! 多卡 = M2 comm crate(one-shot AR + VMM 通道),接线点已收敛在
//! [`Comm::all_reduce`],T3/M2 回填。列并行切分/行并行合并的
//! 分片装载随 loader 权重通道回填。

use super::wna16::WNA16;
use super::linear::{linear_b_x, linear_no_bias_x, LnNvfp4, LinearX};
use super::{DType, Module, OwlTensor, Result, Shard, Tensor, VarBuilderX};
use crate::config::QuantConfig;
use std::rc::Rc;

/// 量化配置引用别名(xinfer 侧为 &Option<QuantConfig>)
pub type QuantCfgRef<'a> = &'a Option<QuantConfig>;

/// 通信组(M2 comm 接线点;MVP = world_size 1)
pub struct Comm {
    pub world_size: usize,
    pub rank: usize,
}

impl Comm {
    pub fn new() -> Rc<Self> {
        Rc::new(Self {
            world_size: 1,
            rank: 0,
        })
    }

    /// all_reduce(A2.1 尺寸路由 + A2.2 可捕获性:decode 走 one-shot,
    /// prefill 走 two-shot;M2 回填。MVP 单实例 = 恒等)
    pub fn all_reduce(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

impl Default for Comm {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
        }
    }
}

/// 复制线性(无切分;装载走 dense 通道)
pub struct ReplicatedLinear {
    inner: LinearX,
}

impl ReplicatedLinear {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilderX,
        shard: Shard,
        quant_cfg: QuantCfgRef,
        quant: &Option<String>,
        dtype: DType,
        bias: bool,
    ) -> Result<Self> {
        let inner = linear_b_x(in_dim, out_dim, bias, vb, shard, quant_cfg, quant, dtype)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

/// 列并行(hidden 维切出;MVP = 装载全量,分片随 loader 回填)
pub struct TensorParallelColumnLinear {
    inner: LinearX,
}

impl TensorParallelColumnLinear {
    pub fn new(linear: LinearX) -> Self {
        Self { inner: linear }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_loaded(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilderX,
        shard: Shard,
        quant_cfg: QuantCfgRef,
        quant: &Option<String>,
        dtype: DType,
        bias: bool,
    ) -> Result<Self> {
        let inner = linear_b_x(in_dim, out_dim, bias, vb, shard, quant_cfg, quant, dtype)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }

    /// NVFP4 融合面(mlp merge 探测用;量化壳恒 None)
    pub fn as_nvfp4(&self) -> Option<&LnNvfp4> {
        self.inner.as_nvfp4()
    }

    /// S2.5-B prequant 通道
    pub fn forward_prequant(&self, xq: &Tensor, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward_prequant(xq, xs)
    }

    pub fn w4a8_active(&self) -> bool {
        self.inner.w4a8_active()
    }
}

/// 合并列并行(gate/up 融合;forward 返回切片向量)
pub struct MergedParallelColumnLinear {
    linears: Vec<TensorParallelColumnLinear>,
}

impl MergedParallelColumnLinear {
    pub fn new(linears: Vec<TensorParallelColumnLinear>) -> Self {
        Self { linears }
    }

    /// 分片装载(每片独立 linear_no_bias_x;loader 分片回填)
    #[allow(clippy::too_many_arguments)]
    pub fn new_loaded(
        in_dim: usize,
        out_dims: &[usize],
        vb: &VarBuilderX,
        shard: Shard,
        quant_cfg: QuantCfgRef,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let mut linears = Vec::with_capacity(out_dims.len());
        for (i, out) in out_dims.iter().enumerate() {
            let vbx = vb.pp(&format!("{i}"));
            let inner = linear_no_bias_x(*out, in_dim, &vbx, shard, quant_cfg, quant, dtype)?;
            linears.push(TensorParallelColumnLinear { inner });
        }
        Ok(Self { linears })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        self.linears.iter().map(|l| l.forward(x)).collect()
    }

    /// 前向到合并输出(切片后 cat;T3 融合 kernel 替换)
    pub fn forward_cat(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        let parts = self.forward(x)?;
        super::ops::cat(&parts, dim)
    }
}

/// 行并行(intermediate 维切出;输出过 all_reduce)
pub struct TensorParallelRowLinear {
    inner: LinearX,
    comm: Rc<Comm>,
    wna16: Option<WNA16>,
    dtype: DType,
}

impl TensorParallelRowLinear {
    pub fn new(comm: Rc<Comm>) -> Self {
        let _ = comm;
        // 零权占位(构造即用型;真实装载走 new_loaded)
        unimplemented!("TensorParallelRowLinear::new 零权占位 = loader T3")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_loaded(
        in_dim: usize,
        out_dim: usize,
        vb: &VarBuilderX,
        shard: Shard,
        quant_cfg: QuantCfgRef,
        quant: &Option<String>,
        dtype: DType,
        bias: bool,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        let inner = linear_b_x(in_dim, out_dim, bias, vb, shard, quant_cfg, quant, dtype)?;
        Ok(Self {
            inner,
            comm,
            wna16: None,
            dtype,
        })
    }

    pub fn new_with_bias(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: &VarBuilderX,
        shard: Shard,
        quant_cfg: QuantCfgRef,
        quant: &Option<String>,
        dtype: DType,
        comm: Rc<Comm>,
    ) -> Result<Self> {
        Self::new_loaded(
            in_dim,
            out_dim,
            vb,
            shard,
            quant_cfg,
            quant,
            dtype,
            bias,
            comm,
        )
    }

    /// 无 all_reduce 的本地前向(序列并行/预量化路径)
    pub fn forward_local(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }

    pub fn forward_prequant(&self, xq: &Tensor, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward_prequant(xq, xs)
    }

    /// F32 激活 + 权重自带 dtype 的归约前向(精度敏感路径)
    pub fn forward_f32_reduce(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.inner.forward(&x.to_dtype(DType::F32)?)?;
        self.comm.all_reduce(&out)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = self.inner.forward(x)?;
        self.comm.all_reduce(&out)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// wna16 权重直读(wna16 探测/融合用)
    pub fn set_wna16(&mut self, w: WNA16) {
        self.wna16 = Some(w);
    }
}
