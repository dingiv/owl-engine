//! RmsNorm:语义算子 Op::Rmsnorm(per-channel gamma [n] 广播;×w 语义,
//! 无 offset —— Qwen3.5 主干 norm 全系 use_offset=false)。
//! 容器 + LoaderOps 装载形态。

use crate::loader::{Loadable, LoaderCtx, LoaderOps, Weight};
use crate::module::{KernelCtx, Module};
use crate::TensorOps;

pub struct RmsNorm {
    /// gamma 槽 [n]
    w: Weight,
    eps: f32,
}

impl RmsNorm {
    /// 准备容器(`key` = 数据源槽键)
    pub fn new(key: &'static str, n: usize, eps: f32) -> RmsNorm {
        RmsNorm { w: Weight::new(key, vec![n]), eps }
    }


}

impl Module for RmsNorm {
    /// x [tokens, n] → 逐行 RMSNorm × gamma(未装载 → 毒值声明)
    fn forward(&self, xs: &TensorOps, _ctx: &KernelCtx) -> TensorOps {
        xs.rmsnorm(&self.w.decl(), self.eps, false)
    }
}

impl Loadable for RmsNorm {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.w.layout(ctx)
    }

}
