//! Embedding:词表查表(Kernel 注册表:`owl_embed_f32`)。
//! tied 权重:同一份 `[vocab, D]` 数据兼 embedding 与 lm_head
//! (lm_head 走 `lm_head_matmul`,转置槽第二份 —— 块复用待装载账立项)。
//! 容器 + LoaderOps 装载形态。

use crate::kernels;
use crate::loader::{Loadable, LoaderCtx, LoaderOps, Weight};
use crate::module::{KernelCtx, Module};
use crate::tensor::Dtype;
use crate::TensorOps;

pub struct Embedding {
    /// [vocab, D](原始布局;tied)
    w: Weight,
    /// lm_head 转置槽 [D, vocab](tied 同源数据)
    w_t: Weight,
    d_dim: usize,
}

impl Embedding {
    /// 准备容器
    pub fn new(vocab: usize, d_dim: usize) -> Embedding {
        Embedding {
            w: Weight::new("w", vec![vocab, d_dim]),
            w_t: Weight::new_transposed("w_t", vocab, d_dim),
            d_dim,
        }
    }

    /// 查表(Module 统一入口的实体;tokens 由 ctx 提供 —— 每步动态依赖)。
    /// grid = tokens(核内 blockIdx.x = token 行;发射配置声明期显式)。
    /// 槽序契约:kernel 签名 (w, ids, out, d_dim) → T 槽序 w、ids,输出块末尾
    /// (w = 物化块引用,eval 时 Block 叶子零操作)。
    pub fn embed(&self, ids: &TensorOps, tokens: usize) -> TensorOps {
        TensorOps::of(kernels::kernel_with(
            "owl_embed_f32",
            (tokens as u32, 1, 1),
            (1, 1, 1),
            0,
        ))
        .arg(&self.w.decl())
        .arg(ids)
        .arg_usize(self.d_dim)
        .with_shape(Dtype::F32, vec![tokens, self.d_dim])
    }

    /// lm_head(tied):hidden [.., D] → logits [.., vocab]
    pub fn lm_head_matmul(&self, hidden: &TensorOps) -> TensorOps {
        hidden.matmul(&self.w_t.decl())
    }
}

impl Loadable for Embedding {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.w.layout(ctx).chain(self.w_t.layout(ctx))
    }

}

impl Module for Embedding {
    /// 查表声明(tokens 由 ctx 提供 —— 每步动态依赖)
    fn forward(&self, ids: &TensorOps, ctx: &KernelCtx) -> TensorOps {
        self.embed(ids, ctx.tokens)
    }
}
