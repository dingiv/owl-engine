//! MLP(gate/up/down;= xinfer layers/mlp.rs 直译,编译先行口径)。
//!
//! - Separate 路径(列并行 gate/up + 行并行 down)结构完整;
//! - Packed/Nvfp4Merged 分派保留,体 = 量化装载占位;
//! - 激活面走 config::Activation(engine 版),kernel 调用 = ops 桩。

use super::distributed::{
    Comm, MergedParallelColumnLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
};
use super::distributed::QuantCfgRef;
use super::OwlTensor as _;
use super::{ops, DType, Result, Tensor, VarBuilderX};
use crate::bail;
use crate::config::Activation;
use std::rc::Rc;

enum GateUpProjection {
    Separate {
        gate_proj: TensorParallelColumnLinear,
        up_proj: TensorParallelColumnLinear,
    },
    /// gate/up 已在 checkpoint 内合并为单矩阵(Loader 权重通道回填)
    Packed(MergedParallelColumnLinear),
}

pub struct MLP {
    gate_up_proj: GateUpProjection,
    down_proj: TensorParallelRowLinear,
    activation: Activation,
    swiglu_limit: f32,
}

impl MLP {
    /// xinfer 同签名;gate_up_merged = checkpoint 单矩阵开关
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        hidden_size: usize,
        intermediate_size: usize,
        activation: &Activation,
        quant_cfg: QuantCfgRef,
        quant: &Option<String>,
        gate_up_merged: bool,
        dtype: DType,
        _suffix: &str,
    ) -> Result<Self> {
        let shard = super::Shard::default();
        let gate_proj = TensorParallelColumnLinear::new_loaded(
            hidden_size,
            intermediate_size,
            vb,
            shard,
            quant_cfg,
            quant,
            dtype,
            false,
        )?;
        let up_vbx = vb.pp("up_proj");
        let up_proj = TensorParallelColumnLinear::new_loaded(
            hidden_size,
            intermediate_size,
            &up_vbx,
            shard,
            quant_cfg,
            quant,
            dtype,
            false,
        )?;
        let gate_up_proj = if gate_up_merged {
            GateUpProjection::Packed(MergedParallelColumnLinear::new(vec![gate_proj, up_proj]))
        } else {
            GateUpProjection::Separate { gate_proj, up_proj }
        };
        let down_vbx = vb.pp("down_proj");
        let down_proj = TensorParallelRowLinear::new_with_bias(
            intermediate_size,
            hidden_size,
            false,
            &down_vbx,
            shard,
            quant_cfg,
            quant,
            dtype,
            comm,
        )?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            activation: *activation,
            swiglu_limit: f32::INFINITY,
        })
    }

    pub fn with_swiglu_limit(mut self, limit: f32) -> Self {
        self.swiglu_limit = limit;
        self
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate_up = match &self.gate_up_proj {
            GateUpProjection::Separate { gate_proj, up_proj } => {
                let gate = gate_proj.forward(xs)?;
                let up = up_proj.forward(xs)?;
                (gate, up)
            }
            GateUpProjection::Packed(packed) => {
                let mut parts = packed.forward(xs)?.into_iter();
                let gate = parts.next().ok_or_else(|| crate::Error::Msg("MLP: empty packed gate/up".into()))?;
                let up = parts.next().ok_or_else(|| crate::Error::Msg("MLP: packed up missing".into()))?;
                (gate, up)
            }
        };
        let activated = self.apply_activation(gate_up.0, gate_up.1)?;
        self.down_proj.forward(&activated)
    }

    /// 激活分派(Silu/Gelu 走 ops;Qwen3.5 swiglu_limit 语义面保留)
    fn apply_activation(&self, gate: Tensor, up: Tensor) -> Result<Tensor> {
        match self.activation {
            Activation::Silu => {
                if self.swiglu_limit.is_finite() {
                    // Qwen3.5 swiglu-limit 变体(限制因子钳制;kernel 回填点)
                    bail!("swiglu_limit 变体:T3 kernel 回填")
                }
                ops::silu_and_mul(&ops::cat(&[gate.broadcast_mul(&up)?, up.clone()], 0)?)
            }
            Activation::Gelu => {
                let g = gate.gelu()?;
                g.broadcast_mul(&up)
            }
            Activation::NewGelu => {
                let g = ops::gelu(&gate)?;
                g.broadcast_mul(&up)
            }
            _ => bail!("MLP::apply_activation: 未支持的激活变体(搬运按需增补)"),
        }
    }
}

/// 朴素 MLP(单卡全量矩阵;测试/参考路径)
pub struct NaiveMLP {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
    activation: Activation,
}

impl NaiveMLP {
    pub fn new(
        gate_proj: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        activation: Activation,
    ) -> Self {
        Self {
            gate_proj,
            up_proj,
            down_proj,
            activation,
        }
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = xs.matmul(&self.gate_proj)?;
        let up = xs.matmul(&self.up_proj)?;
        let h = match self.activation {
            Activation::Silu => gate.silu()?.broadcast_mul(&up)?,
            _ => gate.gelu()?.broadcast_mul(&up)?,
        };
        h.matmul(&self.down_proj)
    }
}
