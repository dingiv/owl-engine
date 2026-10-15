//! LayerNorm:affine(weight + bias)行归一化 —— **Qwen3.5-VL 视觉塔专用**。
//!
//! 与主干 [`RmsNorm`](crate::layers::rmsnorm::RmsNorm) 的区别(不可互替):
//! - 去**均值**(主干 RMSNorm 不去);有 **bias**(主干没有);
//! - 无 `×(1+w)` 变体(塔的 LN 权重是普通仿射权重,`nn.LayerNorm` 直用)。
//!
//! 语义锚 = `torch.nn.LayerNorm(normalized_shape, eps=1e-6)`,即 HF
//! `Qwen3_5VisionBlock.norm1/norm2` 与 `Qwen3_5VisionPatchMerger.norm` 的实际
//! 实现(`modeling_qwen3_5.py`,见 `docs/arch/vision-tower-设计与CPU对拍.md` §1.2 #5)。
//!
//! 数值口径(two-pass,与 torch 同序):mean = Σx/n;var = Σ(x−mean)²/n(biased);
//! y = (x−mean)·rsqrt(var+eps)·w + b。

use crate::module::{Loadable, LoaderCtx, LoaderOps, Weight};
use crate::module::{ForwardCtx, Module};
use crate::TensorOps;

pub struct LayerNorm {
    /// weight 槽 [n]
    w: Weight,
    /// bias 槽 [n]
    b: Weight,
    eps: f32,
}

impl LayerNorm {
    /// 准备容器(`w_key`/`b_key` = 数据源槽键;零数据零副作用)
    pub fn new(w_key: &'static str, b_key: &'static str, n: usize, eps: f32) -> LayerNorm {
        LayerNorm { w: Weight::new(w_key, vec![n]), b: Weight::new(b_key, vec![n]), eps }
    }

    /// 装载完备性
    pub fn is_loaded(&self) -> bool {
        self.w.is_loaded() && self.b.is_loaded()
    }

    /// weight/bias 槽声明 / eps(塔装配期消费:rope/attn 接线先例同款)
    #[allow(dead_code)]
    pub(crate) fn weight_decl(&self) -> TensorOps {
        self.w.decl()
    }

    #[allow(dead_code)]
    pub(crate) fn bias_decl(&self) -> TensorOps {
        self.b.decl()
    }

    #[allow(dead_code)]
    pub(crate) fn eps(&self) -> f32 {
        self.eps
    }
}

impl Module for LayerNorm {
    /// x [rows, n] → LayerNorm(x)·w + b(未装载 → 毒值声明)
    fn forward(&self, xs: &TensorOps, _ctx: &ForwardCtx) -> TensorOps {
        xs.layernorm(&self.w.decl(), &self.b.decl(), self.eps)
    }
}

impl Loadable for LayerNorm {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.w.layout(ctx).chain(self.b.layout(ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Dtype;
    use crate::testkit::{assert_close, f32b, harvest, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path, Src};

    /// host 手算锚(two-pass;不依赖 python —— 平时 cron 跑的就这一条)
    #[tokio::test]
    async fn multirow_affine_matches_host() {
        let (rows, n) = (3usize, 5usize);
        let eps = 1e-6f32;
        let w = vec![0.5, 1.0, 1.5, 2.0, -0.25];
        let b = vec![0.1, -0.2, 0.3, 0.0, 0.7];
        let x: Vec<f32> = (0..rows * n).map(|i| (i as f32 * 0.37) - 1.3).collect();

        let mut face = owl_cpu::CpuFace::new();
        let ln = LayerNorm::new("w", "b", n, eps);
        let src = Src::from([("w".to_string(), w.clone()), ("b".to_string(), b.clone())]);
        crate::interpreters::eval_load(&ln, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");

        let xs = TensorOps::from_host(Dtype::F32, vec![rows, n], &f32b(&x));
        let got = harvest(&mut face, &ln.forward(&xs, &ForwardCtx::minimal(rows))).await;

        for r in 0..rows {
            let row = &x[r * n..(r + 1) * n];
            let mean = row.iter().sum::<f32>() / n as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..n {
                let want = (row[c] - mean) * inv * w[c] + b[c];
                assert!((got[r * n + c] - want).abs() < 1e-5, "[{r},{c}] {} vs {want}", got[r * n + c]);
            }
        }
    }

    /// HF/torch parity(`torch.nn.LayerNorm` 直用;门控 OWL_HF_PARITY=1)
    #[tokio::test]
    async fn parity_matches_torch() {
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (rows, n) = (4usize, 16usize);
        let eps = 1e-6f32;
        let w: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11) - 0.8).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07) - 0.3).collect();
        let x: Vec<f32> = (0..rows * n).map(|i| ((i as f32 * 0.29) - 2.0).sin()).collect();

        let inp = tmp_path("layernorm_in");
        let outp = tmp_path("layernorm_out");
        st_write(&inp, &[("x", &x, vec![rows, n]), ("weight", &w, vec![n]), ("bias", &b, vec![n])]);
        let contract = format!(r#"{{"eps": {eps}}}"#);
        let manifest = hf_python("layernorm.py", &[inp.to_str().unwrap(), outp.to_str().unwrap(), &contract]);
        eprintln!("[hf manifest] layernorm: {manifest}");
        let want = st_read(&outp, "y");

        let mut face = owl_cpu::CpuFace::new();
        let ln = LayerNorm::new("weight", "bias", n, eps);
        let src = Src::from([("weight".to_string(), w.clone()), ("bias".to_string(), b.clone())]);
        crate::interpreters::eval_load(&ln, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");
        let xs = TensorOps::from_host(Dtype::F32, vec![rows, n], &f32b(&x));
        let got = harvest(&mut face, &ln.forward(&xs, &ForwardCtx::minimal(rows))).await;
        assert_close(&got, &want, 1e-6, "layernorm");

        owl_shared::file_loader::remove_file(&inp).ok();
        owl_shared::file_loader::remove_file(&outp).ok();
    }
}
