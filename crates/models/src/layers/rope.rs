//! Rope:旋转位置编码(mrope interleaved + partial;Kernel 注册表)。
//!
//! Qwen3.5-0.8B 实测配置:head_dim 256,partial_rotary_factor 0.25 →
//! rotary_dim 64(half 32);mrope_interleaved = true → 旋转对 = 相邻
//! (2i, 2i+1)(GPT-J 风格,非 rotate-half 的 (i, i+half));
//! 文本路径三 section 的 pos 相同(mrope 退化为一维 rope,t [11,11,10]
//! 分段仅在视觉多模态路径有意义 —— 本层只吃一份 pos)。
//!
//! cos/sin 表:new 期纯计算存于层内,经 `tables()` 容器内表源供执行器
//! 取数(不经外部数据源);forward 常驻块引用。
//! kernel 源 = 注册表 `owl_rope_interleaved_partial_f32`(owl-kernels cu/text)。

use crate::contract::ModelError;
use crate::kernel::Kernel;
use crate::kernel;
use crate::module::{Loadable, LoaderCtx, LoaderOps, Weight};
use crate::tensor::Dtype;
use crate::TensorOps;

pub struct Rope {
    /// cos 表(new 期纯计算;经 tables() 源供给执行器)
    cos_data: Vec<f32>,
    /// sin 表
    sin_data: Vec<f32>,
    /// cos 表槽 [max_pos, half](常驻块)
    cos: Weight,
    /// sin 表槽 [max_pos, half]
    sin: Weight,
    head_dim: usize,
    rotary_dim: usize,
}

impl Rope {
    /// `theta`(rope_theta = 1e7)、`rotary_dim` = head_dim × partial(0.25 → 64)。
    /// 配置校验是 new 唯一可败点(结构参数非法 = 容器语义不成立,非数据错)。
    pub fn new(
        max_pos: usize,
        head_dim: usize,
        rotary_dim: usize,
        theta: f32,
    ) -> Result<Rope, ModelError> {
        if rotary_dim % 2 != 0 || rotary_dim > head_dim {
            return Err(ModelError::Msg(format!(
                "Rope::new: rotary_dim {rotary_dim} 非法(须偶数且 ≤ head_dim {head_dim})"
            )));
        }
        let half = rotary_dim / 2;
        // inv_freq[i] = theta^(-2i / rotary_dim);interleaved:对 (2i, 2i+1) 用同一频率
        let cos_t: Vec<f32> = (0..max_pos)
            .flat_map(|p| {
                (0..half).map(move |i| {
                    ((p as f32) * theta.powf(-(2.0 * i as f32) / rotary_dim as f32)).cos()
                })
            })
            .collect();
        let sin_t: Vec<f32> = (0..max_pos)
            .flat_map(|p| {
                (0..half).map(move |i| {
                    ((p as f32) * theta.powf(-(2.0 * i as f32) / rotary_dim as f32)).sin()
                })
            })
            .collect();
        Ok(Rope {
            cos_data: cos_t,
            sin_data: sin_t,
            cos: Weight::new("cos_table", vec![max_pos, half]),
            sin: Weight::new("sin_table", vec![max_pos, half]),
            head_dim,
            rotary_dim,
        })
    }

    /// 容器内表源(执行器取数用;借用层内表,生命周期随层)
    pub fn tables(&self) -> crate::module::TableSource<'_> {
        crate::module::TableSource::new(vec![
            ("cos_table", &self.cos_data),
            ("sin_table", &self.sin_data),
        ])
    }

    fn launch_kernel(name: &'static str, tokens: usize) -> Kernel {
        kernel::kernel_with(name, (tokens as u32, 1, 1), (128, 1, 1), 0)
    }

    /// q 旋转:[T, Hq*HD] → [T, Hq*HD](前 rotary_dim 维转,余直通)
    pub fn forward_q(
        &self,
        q: &TensorOps,
        pos: &TensorOps,
        tokens: usize,
        q_heads: usize,
    ) -> TensorOps {
        TensorOps::of(Self::launch_kernel(
            "owl_rope_interleaved_partial_f32",
            tokens,
        ))
        .arg(q)
        .arg(&self.cos.decl())
        .arg(&self.sin.decl())
        .arg(pos)
        .arg_usize(q_heads)
        .arg_usize(self.head_dim)
        .arg_usize(self.rotary_dim / 2)
        .with_shape(Dtype::F32, vec![tokens, q_heads * self.head_dim])
    }

    /// k 旋转:[T, Hkv*HD] → [T, Hkv*HD]
    pub fn forward_k(
        &self,
        k: &TensorOps,
        pos: &TensorOps,
        tokens: usize,
        kv_heads: usize,
    ) -> TensorOps {
        TensorOps::of(Self::launch_kernel(
            "owl_rope_interleaved_partial_f32",
            tokens,
        ))
        .arg(k)
        .arg(&self.cos.decl())
        .arg(&self.sin.decl())
        .arg(pos)
        .arg_usize(kv_heads)
        .arg_usize(self.head_dim)
        .arg_usize(self.rotary_dim / 2)
        .with_shape(Dtype::F32, vec![tokens, kv_heads * self.head_dim])
    }
}

impl Loadable for Rope {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.cos.layout(ctx).chain(self.sin.layout(ctx))
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::f32b;
    use crate::tensor::Dtype;

    #[tokio::test]
    async fn declaration_is_wellformed() {
        let (head_dim, rotary_dim, heads) = (8usize, 4usize, 2usize);
        let mut face = owl_cpu::CpuFace::new();
        let rp = Rope::new(64, head_dim, rotary_dim, 10_000.0).expect("new");
        crate::interpreter::eval_load(&rp, &mut face, &rp.tables(), &Default::default())
            .await
            .expect("表物化");

        let pos = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[5.0]));
        let q = TensorOps::from_host(
            Dtype::F32,
            vec![1, heads * head_dim],
            &f32b(&vec![0.3; heads * head_dim]),
        );

        let out = rp.forward_q(&q, &pos, 1, heads);
        assert!(!out.is_poisoned(), "rope 声明不应有毒");
        assert_eq!(out.shape(), &[1, heads * head_dim]);

        assert!(Rope::new(64, head_dim, 7, 10_000.0).is_err(), "非偶 rotary_dim 应拒");
        assert!(Rope::new(64, head_dim, head_dim * 2, 10_000.0).is_err(), "超 head_dim 应拒");
    }

    /// Rope = 位置上下文机制(双输入 + pos),不进 Module(C4 后仍成立;
    /// attention 经 ForwardCtx 持 &Rope 调用其固有 forward_q/k)
    #[test]
    fn stays_outside_module_interface() {
        fn assert_impl<M: crate::module::Module>(_: &M) {}
        assert_impl(&crate::layers::mlp::Mlp::new(2, 4)); // Mlp 在 Module 内
        // Rope 无 forward(&TensorOps, &ForwardCtx) 签名 —— 编译期即证不在 trait 内
        let rp = Rope::new(64, 8, 4, 10_000.0).expect("new");
        let pos = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[5.0]));
        let _ = rp.forward_q(
            &TensorOps::from_host(Dtype::F32, vec![1, 16], &f32b(&vec![0.3; 16])),
            &pos,
            1,
            2,
        );
    }
}
