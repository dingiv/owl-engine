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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, f32b, harvest, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path, Src};
    use crate::tensor::Dtype;
    use std::collections::HashMap;

    #[tokio::test]
    async fn transposed_matmul_matches_host() {
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 源 [2,3] 行主序
        let x = vec![0.5, -1.0, 2.0];
        let mut want = vec![0.0f32; 2];
        for (o, wo) in want.iter_mut().enumerate() {
            *wo = (0..3).map(|i| x[i] * w[o * 3 + i]).sum();
        }

        let mut face = owl_cpu::CpuFace::new();
        let lin = Linear::new("w", 2, 3);
        let src = HashMap::from([("w".to_string(), w)]);
        crate::interpreter::eval_load(&lin, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");

        let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&x));
        let got = harvest(&mut face, &lin.forward(&xs, &ForwardCtx::minimal(1))).await;
        assert_eq!(got.len(), 2);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6, "{g} vs {w}");
        }
    }

    #[tokio::test]
    async fn unloaded_slot_becomes_poison_at_boundary() {
        let mut face = owl_cpu::CpuFace::new();
        let lin = Linear::new("w", 2, 3);
        let xs = TensorOps::from_host(Dtype::F32, vec![1, 3], &f32b(&[1.0, 2.0, 3.0]));

        let out = lin.forward(&xs, &ForwardCtx::minimal(1));
        assert!(out.is_poisoned(), "未装载槽的声明应立即带毒(随链流动)");
        let err = crate::interpreter::eval_ops(out.step(), &mut face)
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("未装载"), "{err:?}");

        let empty: Src = HashMap::new();
        let err = crate::interpreter::eval_load(&lin, &mut face, &empty, &Default::default())
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("缺键"), "{err:?}");

        let lin3 = Linear::new("w", 2, 3);
        let bad_len = HashMap::from([("w".to_string(), vec![1.0; 5])]);
        let err = crate::interpreter::eval_load(&lin3, &mut face, &bad_len, &Default::default())
            .await
            .unwrap_err();
        assert!(format!("{err:?}").contains("元素"), "{err:?}");
    }

    /// HF parity(F.linear = x@W^T;多行 tokens=2;门控 OWL_HF_PARITY=1)
    #[tokio::test]
    async fn parity_matches_hf() {
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (tokens, out_dim, in_dim) = (2usize, 2usize, 3usize);
        let x: Vec<f32> = (0..tokens * in_dim).map(|i| ((i as f32 * 0.53) - 1.2).cos()).collect();
        let w: Vec<f32> = (0..out_dim * in_dim).map(|i| (i as f32 * 0.29) - 0.5).collect();
        let xs = TensorOps::from_host(Dtype::F32, vec![tokens, in_dim], &f32b(&x));
        let ctx = ForwardCtx::minimal(tokens);

        let inp = tmp_path("linear_in");
        let outp = tmp_path("linear_out");
        st_write(&inp, &[("x", &x, vec![tokens, in_dim]), ("w", &w, vec![out_dim, in_dim])]);
        let manifest = hf_python("linear.py", &[
            inp.to_str().unwrap(), outp.to_str().unwrap(),
            &format!(r#"{{"out": {out_dim}, "in": {in_dim}}}"#),
        ]);
        eprintln!("[hf manifest] linear: {manifest}");
        let want = st_read(&outp, "y");

        for (face_tag, make) in [("cpu", None), ("gpu", Some(()))] {
            let (_, out) = if make.is_none() {
                let mut face = owl_cpu::CpuFace::new();
                let lin = Linear::new("w", out_dim, in_dim);
                let src = HashMap::from([("w".to_string(), w.clone())]);
                crate::interpreter::eval_load(&lin, &mut face, &src, &Default::default())
                    .await.expect("eval_load");
                ("cpu", harvest(&mut face, &lin.forward(&xs, &ctx)).await)
            } else {
                let mut gpu = crate::testkit::gpu_client().await;
                let lin = Linear::new("w", out_dim, in_dim);
                let src = HashMap::from([("w".to_string(), w.clone())]);
                crate::interpreter::eval_load(&lin, &mut gpu, &src, &Default::default())
                    .await.expect("eval_load");
                let o = harvest(&mut gpu, &lin.forward(&xs, &ctx)).await;
                gpu.close().await.expect("server 关机");
                ("gpu", o)
            };
            assert_close(&out, &want, 1e-5, &format!("linear-{face_tag}"));
        }
        std::fs::remove_file(&inp).ok();
        std::fs::remove_file(&outp).ok();
    }
}
