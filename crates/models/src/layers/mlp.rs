//! Mlp:SwiGLU 前馈(gate/up → silu(gate) × up → down;全语义算子,
//! 双 face(CPU/GPU)可对拍)。
//!
//! 融合优化(silu_and_mul kernel,port 自 dry_kernels.cu)留算子面
//! 立项 —— 语义组合(silu + mul)先行,正确性优先。
//! 容器 + LoaderOps 装载形态(子层描述 chain 聚合)。

use super::linear::Linear;
use crate::loader::{Loadable, LoaderCtx, LoaderOps};
use crate::module::{KernelCtx, Module};
use crate::TensorOps;

pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    /// 准备容器(hidden = 1024, intermediate = 3584 —— 0.8B 实测配置)
    pub fn new(hidden: usize, intermediate: usize) -> Mlp {
        Mlp {
            gate_proj: Linear::new("gate_proj", intermediate, hidden),
            up_proj: Linear::new("up_proj", intermediate, hidden),
            down_proj: Linear::new("down_proj", hidden, intermediate),
        }
    }


}

impl Module for Mlp {
    fn forward(&self, xs: &TensorOps, ctx: &KernelCtx) -> TensorOps {
        let gate = self.gate_proj.forward(xs, ctx);
        let up = self.up_proj.forward(xs, ctx);
        // silu(gate) * up(门控;同形逐元素)
        let h = gate.silu().mul(&up);
        self.down_proj.forward(&h, ctx)
    }
}

impl Loadable for Mlp {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.gate_proj.layout(ctx)
            .chain(self.up_proj.layout(ctx))
            .chain(self.down_proj.layout(ctx))
    }

}
