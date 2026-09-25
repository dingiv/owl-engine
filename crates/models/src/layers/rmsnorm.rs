//! RmsNorm:语义算子 Op::Rmsnorm(per-channel gamma [n] 广播;×w 语义,
//! 无 offset —— Qwen3.5 主干 norm 全系 use_offset=false)。
//! 容器 + LoaderOps 装载形态。

use crate::module::{Loadable, LoaderCtx, LoaderOps, Weight};
use crate::module::{ForwardCtx, Module};
use crate::TensorOps;

pub struct RmsNorm {
    /// gamma 槽 [n]
    w: Weight,
    eps: f32,
    /// ×(1+w) 语义(qk-norm add_one;主干 norm 全系 false)
    w_off: bool,
}

impl RmsNorm {
    /// 准备容器(`key` = 数据源槽键;主干 norm ×w 语义)
    pub fn new(key: &'static str, n: usize, eps: f32) -> RmsNorm {
        RmsNorm { w: Weight::new(key, vec![n]), eps, w_off: false }
    }

    /// ×(1+w) 语义变体(Qwen3.5 qk-norm:per-head add_one)
    pub fn new_add_one(key: &'static str, n: usize, eps: f32) -> RmsNorm {
        RmsNorm { w: Weight::new(key, vec![n]), eps, w_off: true }
    }

}

impl Module for RmsNorm {
    /// x [tokens, n] → 逐行 RMSNorm × gamma(未装载 → 毒值声明)
    fn forward(&self, xs: &TensorOps, _ctx: &ForwardCtx) -> TensorOps {
        xs.rmsnorm(&self.w.decl(), self.eps, self.w_off)
    }
}

impl Loadable for RmsNorm {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.w.layout(ctx)
    }

}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, f32b, harvest, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path, Src};
    use crate::tensor::Dtype;

    #[tokio::test]
    async fn multirow_perchannel_matches_host() {
        let (rows, n) = (3usize, 4usize);
        let gamma = vec![0.5, 1.0, 1.5, 2.0];
        let x: Vec<f32> = (0..rows * n).map(|i| (i as f32 * 0.25) - 1.0).collect();

        let mut face = owl_cpu::CpuFace::new();
        let norm = RmsNorm::new("gamma", n, 1e-6);
        let src = Src::from([("gamma".to_string(), gamma.clone())]);
        crate::interpreter::eval_load(&norm, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");

        let xs = TensorOps::from_host(Dtype::F32, vec![rows, n], &f32b(&x));
        let got = harvest(&mut face, &norm.forward(&xs, &ForwardCtx::minimal(rows))).await;

        for r in 0..rows {
            let row = &x[r * n..(r + 1) * n];
            let ms = row.iter().map(|v| v * v).sum::<f32>() / n as f32;
            let inv = 1.0 / (ms + 1e-6).sqrt();
            for (c, v) in row.iter().enumerate() {
                let want = v * inv * gamma[c];
                assert!((got[r * n + c] - want).abs() < 1e-5, "[{r},{c}] {} vs {want}", got[r * n + c]);
            }
        }
    }

    #[tokio::test]
    async fn add_one_variant_matches_host() {
        let n = 4usize;
        let gamma = vec![0.5, 1.0, 1.5, 2.0];
        let x = vec![0.5, -1.0, 2.0, 0.25];
        let mut face = owl_cpu::CpuFace::new();
        let norm = RmsNorm::new_add_one("q_norm", n, 1e-6);
        let src = Src::from([("q_norm".to_string(), gamma.clone())]);
        crate::interpreter::eval_load(&norm, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");

        let xs = TensorOps::from_host(Dtype::F32, vec![1, n], &f32b(&x));
        let got = harvest(&mut face, &norm.forward(&xs, &ForwardCtx::minimal(1))).await;
        let ms = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for (c, g) in got.iter().enumerate() {
            let want = x[c] * inv * (gamma[c] + 1.0);
            assert!((g - want).abs() < 1e-5, "[{c}] {g} vs {want}");
        }
    }

    /// HF parity(三方锚:transformers 金标准;门控 OWL_HF_PARITY=1)
    #[tokio::test]
    async fn parity_matches_hf() {
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (rows, n) = (3usize, 8usize);
        let x: Vec<f32> = (0..rows * n).map(|i| ((i as f32 * 0.37) - 2.1).sin()).collect();
        let gamma: Vec<f32> = (0..n).map(|i| (i as f32 * 0.21) - 0.7).collect();
        let eps = 1e-6f32;
        let xs = TensorOps::from_host(Dtype::F32, vec![rows, n], &f32b(&x));
        let ctx = ForwardCtx::minimal(rows);

        for (tag, w_off) in [("main_xw", false), ("qknorm_x1pw", true)] {
            let inp = tmp_path(&format!("rmsnorm_in_{tag}"));
            let outp = tmp_path(&format!("rmsnorm_out_{tag}"));
            st_write(&inp, &[("x", &x, vec![rows, n]), ("weight", &gamma, vec![n])]);

            let contract = format!(r#"{{"eps": {eps}, "w_off": {}}}"#, w_off as i32);
            let manifest = hf_python(
                "rmsnorm.py",
                &[inp.to_str().unwrap(), outp.to_str().unwrap(), &contract],
            );
            eprintln!("[hf manifest] {tag}: {manifest}");
            let want = st_read(&outp, "y");

            let src = Src::from([("weight".to_string(), gamma.clone())]);
            let mut cpu = owl_cpu::CpuFace::new();
            let norm = if w_off {
                RmsNorm::new_add_one("weight", n, eps)
            } else {
                RmsNorm::new("weight", n, eps)
            };
            crate::interpreter::eval_load(&norm, &mut cpu, &src, &Default::default())
                .await
                .expect("eval_load(cpu)");
            let cpu_out = harvest(&mut cpu, &norm.forward(&xs, &ctx)).await;
            assert_close(&cpu_out, &want, 1e-5, tag);

            if crate::testkit::gpu_enabled() {
                let mut gpu = crate::testkit::gpu_client().await;
                let norm = if w_off {
                    RmsNorm::new_add_one("weight", n, eps)
                } else {
                    RmsNorm::new("weight", n, eps)
                };
                crate::interpreter::eval_load(&norm, &mut gpu, &src, &Default::default())
                    .await
                    .expect("eval_load(gpu)");
                let gpu_out = harvest(&mut gpu, &norm.forward(&xs, &ctx)).await;
                assert_close(&gpu_out, &want, 1e-5, &format!("{tag}-gpu"));
                gpu.close().await.expect("server 关机");
            }

            std::fs::remove_file(&inp).ok();
            std::fs::remove_file(&outp).ok();
        }
    }
}
