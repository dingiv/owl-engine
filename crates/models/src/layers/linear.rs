//! Linear:仿射层(容器 + LoaderOps 装载 + Matmul;无 bias —— Qwen3.5
//! 全系 attention/mlp 权重无 bias)。
//!
//! 权重槽 = 转置装载:数据源 [out, in] 行主序 → 声明 [in, out]
//! (forward 直 matmul,decode 零转置)。

use crate::module::{ForwardCtx, Loadable, LoaderCtx, LoaderOps, Module, Weight};
use crate::TensorOps;

pub struct Linear {
    /// 权重槽 [in, out](装载期已转置)
    w: Weight,
}

impl Linear {
    /// 准备容器(`key` = 数据源槽键;零数据零副作用)
    pub fn new(key: &'static str, out_dim: usize, in_dim: usize) -> Linear {
        Linear { w: Weight::new_transposed(key, out_dim, in_dim) }
    }

    /// 取出权重槽(单权重基本函数 `load_weight` 的入口;消费层容器)
    pub fn into_weight(self) -> Weight {
        self.w
    }
}

impl Module for Linear {
    /// y = x @ W([.., in] → [.., out];未装载 → 毒值声明)
    fn forward(&self, xs: &TensorOps, _ctx: &ForwardCtx) -> TensorOps {
        xs.matmul(&self.w.decl())
    }
}

impl Loadable for Linear {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.w.layout(ctx)
    }
}
