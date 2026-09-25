//! Mlp:SwiGLU 前馈(gate/up → silu(gate) × up → down;全语义算子,
//! 双 face(CPU/GPU)可对拍)。
//!
//! 融合优化(silu_and_mul kernel,port 自 dry_kernels.cu)留算子面
//! 立项 —— 语义组合(silu + mul)先行,正确性优先。
//! 容器 + LoaderOps 装载形态(子层描述 chain 聚合)。

use super::linear::Linear;
use crate::module::{Loadable, LoaderCtx, LoaderOps};
use crate::module::{ForwardCtx, Module};
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
    fn forward(&self, xs: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, f32b, harvest, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path, Src};
    use std::collections::HashMap;
    use crate::tensor::Dtype;

    #[tokio::test]
    async fn chain_matches_host_reference() {
        let (hidden, intermediate) = (4usize, 6usize);
        let gate_w: Vec<f32> = (0..intermediate * hidden).map(|i| (i as f32 * 0.01) - 0.1).collect();
        let up_w: Vec<f32> = (0..intermediate * hidden).map(|i| 0.2 - (i as f32 * 0.005)).collect();
        let down_w: Vec<f32> = (0..hidden * intermediate).map(|i| (i as f32 * 0.008) + 0.05).collect();
        let x = vec![0.5, -0.25, 1.0, 0.0];

        let mut face = owl_cpu::CpuFace::new();
        let layer = Mlp::new(hidden, intermediate);
        let src = Src::from([
            ("gate_proj".to_string(), gate_w.clone()),
            ("up_proj".to_string(), up_w.clone()),
            ("down_proj".to_string(), down_w.clone()),
        ]);
        crate::interpreter::eval_load(&layer, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");

        let xs = TensorOps::from_host(Dtype::F32, vec![1, hidden], &f32b(&x));
        let got = harvest(&mut face, &layer.forward(&xs, &ForwardCtx::minimal(1))).await;
        assert_eq!(got.len(), hidden);

        let mut want = vec![0.0f32; hidden];
        for o in 0..hidden {
            let mut acc = 0.0f32;
            for j in 0..intermediate {
                let mut g = 0.0f32;
                let mut u = 0.0f32;
                for i in 0..hidden {
                    g += x[i] * gate_w[j * hidden + i];
                    u += x[i] * up_w[j * hidden + i];
                }
                acc += (g / (1.0 + (-g).exp())) * u * down_w[o * intermediate + j];
            }
            want[o] = acc;
        }
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "{g} vs {w}");
        }
    }

    /// HF parity(Qwen3_5MLP 同式 SwiGLU;门控 OWL_HF_PARITY=1)
    #[tokio::test]
    async fn parity_matches_hf() {
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (tokens, hidden, inter) = (2usize, 4usize, 6usize);
        let x: Vec<f32> = (0..tokens * hidden).map(|i| ((i as f32 * 0.41) - 0.9).sin()).collect();
        let gate_w: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.013) - 0.3).collect();
        let up_w: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.027) - 0.2).collect();
        let down_w: Vec<f32> = (0..hidden * inter).map(|i| (i as f32 * 0.031) - 0.1).collect();
        let xs = TensorOps::from_host(Dtype::F32, vec![tokens, hidden], &f32b(&x));
        let ctx = ForwardCtx::minimal(tokens);

        let inp = tmp_path("mlp_in");
        let outp = tmp_path("mlp_out");
        st_write(&inp, &[
            ("x", &x, vec![tokens, hidden]),
            ("gate_w", &gate_w, vec![inter, hidden]),
            ("up_w", &up_w, vec![inter, hidden]),
            ("down_w", &down_w, vec![hidden, inter]),
        ]);
        let manifest = hf_python("mlp.py", &[
            inp.to_str().unwrap(), outp.to_str().unwrap(),
            &format!(r#"{{"hidden": {hidden}, "inter": {inter}}}"#),
        ]);
        eprintln!("[hf manifest] mlp: {manifest}");
        let want = st_read(&outp, "y");

        for (face_tag, on_gpu) in [("cpu", false), ("gpu", true)] {
            if on_gpu && !crate::testkit::gpu_enabled() {
                continue;
            }
            let (tag, out) = if !on_gpu {
                let mut face = owl_cpu::CpuFace::new();
                let layer = Mlp::new(hidden, inter);
                let src = HashMap::from([
                    ("gate_proj".to_string(), gate_w.clone()),
                    ("up_proj".to_string(), up_w.clone()),
                    ("down_proj".to_string(), down_w.clone()),
                ]);
                crate::interpreter::eval_load(&layer, &mut face, &src, &Default::default())
                    .await.expect("eval_load");
                ("cpu", harvest(&mut face, &layer.forward(&xs, &ctx)).await)
            } else {
                let mut gpu = crate::testkit::gpu_client().await;
                let layer = Mlp::new(hidden, inter);
                let src = HashMap::from([
                    ("gate_proj".to_string(), gate_w.clone()),
                    ("up_proj".to_string(), up_w.clone()),
                    ("down_proj".to_string(), down_w.clone()),
                ]);
                crate::interpreter::eval_load(&layer, &mut gpu, &src, &Default::default())
                    .await.expect("eval_load");
                let o = harvest(&mut gpu, &layer.forward(&xs, &ctx)).await;
                gpu.close().await.expect("server 关机");
                ("gpu", o)
            };
            assert_close(&out, &want, 1e-4, &format!("mlp-{face_tag}"));
        }
        std::fs::remove_file(&inp).ok();
        std::fs::remove_file(&outp).ok();
    }
}
