//! WNA16 权重量化层桩(= xinfer layers/wna16.rs 的类型面)。
//!
//! 真实路径 = owl marlin-ffi(W4A16/W4A8/moe_wna16,已验收件),
//! T3 权重装载通道回填;本文件只保 QLinear/LinearX 需要的签名面。

use super::{Result, Tensor};
use crate::config::QuantConfig;

#[derive(Debug, Clone)]
pub struct WNA16 {
    pub in_dim: usize,
    pub out_dim: usize,
    pub is_w4a8: bool,
}

impl WNA16 {
    /// 签名与 xinfer 对齐(12 参);量化装载 = marlin-ffi,运行里程碑回填
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_dim: usize,
        out_dim: usize,
        _vb: &super::VarBuilderX,
        _shards: crate::models::layers::Shard,
        _quant_cfg: &Option<QuantConfig>,
        _bias: bool,
        _dtype: crate::models::layers::DType,
        _is_gptq: bool,
        _module_path: &str,
        _cpu_vb: Option<&super::VarBuilderX>,
    ) -> Result<Self> {
        unimplemented!("WNA16::new = marlin-ffi 权重装载通道,T3 回填")
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _ = x;
        unimplemented!("WNA16::forward = marlin-ffi W4A16 kernel,T3 回填")
    }

    /// S2.5-B:外部 prequant W4A8 matmul
    pub fn forward_prequant(&self, xq: &Tensor, xs: &Tensor) -> Result<Tensor> {
        let _ = (xq, xs);
        unimplemented!("WNA16::forward_prequant = marlin-ffi W4A8,T3 回填")
    }

    pub fn is_w4a8(&self) -> bool {
        self.is_w4a8
    }
}
