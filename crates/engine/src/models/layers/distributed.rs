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

/// 分片参数构造(= xinfer layers::distributed::shard 直译)
pub fn shard(dim: usize, rank: usize, world_size: usize) -> Shard {
    Shard {
        dim,
        rank,
        world_size,
    }
}

/// KV 头切分(GQA;奇数头 = world_size 整除断言。= xinfer kv_head_shard 直译)
pub fn kv_head_shard(
    num_kv_heads: usize,
    rank: usize,
    world_size: usize,
) -> Result<(usize, Shard)> {
    if num_kv_heads % world_size != 0 {
        crate::bail!("kv heads {num_kv_heads} not divisible by TP {world_size}");
    }
    Ok((num_kv_heads / world_size, shard(0, rank, world_size)))
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

    /// 装载口(candle load_b 直译;量化壳透传)
    pub fn load_b(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilderX,
        dtype: DType,
    ) -> Result<Self> {
        Self::new(
            in_dim,
            out_dim,
            &vb,
            Shard::default(),
            &None,
            &None,
            dtype,
            bias,
        )
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

    /// 装载口(= candle load_with_hints;comm 内取 rank/world 构造 shard)
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_hints(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: QuantCfgRef,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let shard = shard(0, comm.rank, comm.world_size);
        Self::new_loaded(in_dim, out_dim, &vb, shard, quant_cfg, quant, dtype, bias)
    }

    /// 装载口(外部 shard;KV 头类切分)
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_shard(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: VarBuilderX,
        shard: Shard,
        quant_cfg: QuantCfgRef,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        Self::new_loaded(in_dim, out_dim, &vb, shard, quant_cfg, quant, dtype, bias)
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

    /// 打包权重本地构造(= xinfer from_packed_local)。
    /// dry-run 实装:weight [out_total, in_dim] 按 splits 逐段 narrow_dim0
    /// (DynTensor 指针级偏移视图)→ Linear 无偏线性;bias 切片同理。
    pub fn from_packed_local(
        weight: Tensor,
        bias: Option<Tensor>,
        splits: Vec<usize>,
    ) -> Result<Self> {
        let in_dim = *weight.shape().last().ok_or_else(|| {
            crate::Error::Msg("from_packed_local: 空 weight".into())
        })?;
        let mut linears = Vec::with_capacity(splits.len());
        let mut start = 0usize;
        for out in splits {
            let w_part = weight.narrow_dim0(start, out)?;
            let b_part = match &bias {
                Some(b) => Some(b.narrow_dim0(start, out)?),
                None => None,
            };
            let inner = crate::models::layers::linear::LinearX::Linear(
                crate::models::layers::linear::Linear::new(w_part, b_part),
            );
            linears.push(TensorParallelColumnLinear { inner });
            start += out;
        }
        Ok(Self { linears })
    }

    /// 分块合并装载(= xinfer load_merged_chunks;逐段 linear_b_x)
    #[allow(clippy::too_many_arguments)]
    pub fn load_merged_chunks(
        in_dim: usize,
        _out_dim_total: usize,
        chunks: Vec<usize>,
        vb: VarBuilderX,
        quant_cfg: QuantCfgRef,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let shard = Shard::default();
        let mut linears = Vec::with_capacity(chunks.len());
        for (i, clen) in chunks.iter().enumerate() {
            let vbx = vb.pp(&format!("{i}"));
            let inner = linear_b_x(*clen, in_dim, false, &vbx, shard, quant_cfg, quant, dtype)?;
            linears.push(TensorParallelColumnLinear { inner });
        }
        let _ = vb;
        Ok(Self { linears })
    }

    /// fp8 打包权重本地构造(量化体 = marlin-ffi,运行里程碑)
    #[allow(clippy::too_many_arguments)]
    pub fn from_packed_local_fp8(
        _weight: Tensor,
        _scale: Tensor,
        _bias: Option<Tensor>,
        _block_size: Vec<usize>,
        _sm_version: usize,
        _splits: Vec<usize>,
    ) -> Result<Self> {
        unimplemented!("T3: from_packed_local_fp8(marlin-ffi 路线)")
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

    /// 装载口(= candle load_with_hints;行并行 shard = dim 1)
    #[allow(clippy::too_many_arguments)]
    pub fn load_with_hints(
        in_dim: usize,
        out_dim: usize,
        vb: VarBuilderX,
        comm: Rc<Comm>,
        quant_cfg: QuantCfgRef,
        quant: &Option<String>,
        dtype: DType,
    ) -> Result<Self> {
        let shard = shard(1, comm.rank, comm.world_size);
        Self::new_loaded(in_dim, out_dim, &vb, shard, quant_cfg, quant, dtype, false, comm)
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
