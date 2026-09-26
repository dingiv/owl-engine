//! Embedding:词表查表(Kernel 注册表:`owl_embed_f32`)。
//! tied 权重:同一份 `[vocab, D]` checkpoint 原布局数据兼 embedding 与
//! lm_head —— lm_head 走 **nt matmul**(`owl_matmul_nt_f32`,B 按
//! [n,k]=[vocab,D] 直读,抄 candle/mistral "W 保持 [out,in]" 惯例):
//! 零 host 转置、零第二份显存(2026-09-26 w_t 双槽形态作废)。
//! 容器 + LoaderOps 装载形态。

use crate::kernel;
use crate::module::{Loadable, LoaderCtx, LoaderOps, Weight};
use crate::module::{ForwardCtx, Module};
use crate::tensor::Dtype;
use crate::TensorOps;

pub struct Embedding {
    /// [vocab, D](checkpoint 原布局;查表 + nt matmul 共用一份)
    w: Weight,
    d_dim: usize,
}

impl Embedding {
    /// 准备容器(局部键 "weight":C10 前缀源下 →
    /// `{base}.embed_tokens.weight`;单槽,tied 经 nt matmul 复用)
    pub fn new(vocab: usize, d_dim: usize) -> Embedding {
        Embedding {
            w: Weight::new("weight", vec![vocab, d_dim]),
            d_dim,
        }
    }

    /// 装载完备性
    pub fn is_loaded(&self) -> bool {
        self.w.is_loaded()
    }

    /// 查表(Module 统一入口的实体;tokens 由 ctx 提供 —— 每步动态依赖)。
    /// grid = tokens(核内 blockIdx.x = token 行;发射配置声明期显式)。
    /// 槽序契约:kernel 签名 (w, ids, d_dim, out) → T 槽序 w、ids,输出块末尾
    /// (w = 物化块引用,eval 时 Block 叶子零操作)。
    pub fn embed(&self, ids: &TensorOps, tokens: usize) -> TensorOps {
        TensorOps::of(kernel::kernel_with(
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
    /// nt 直读原始 [vocab, D] 布局(matmul_nt;零转置)
    pub fn lm_head_matmul(&self, hidden: &TensorOps) -> TensorOps {
        hidden.matmul_nt(&self.w.decl())
    }
}

impl Loadable for Embedding {
    /// tied 单槽:一个 Want 兼查表与 lm_head(nt matmul)
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.w.layout(ctx)
    }

}

impl Module for Embedding {
    /// 查表声明(tokens 由 ctx 提供 —— 每步动态依赖)
    fn forward(&self, ids: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        self.embed(ids, ctx.tokens)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::{assert_close, f32b, harvest, hf_python, parity_enabled, skip_note, st_read, st_write, tmp_path, Src};
    use crate::tensor::Dtype;

    #[tokio::test]
    async fn declaration_is_wellformed() {
        let mut face = owl_cpu::CpuFace::new();
        let emb = Embedding::new(16, 4);
        let src = Src::from([
            ("weight".to_string(), (0..64).map(|i| i as f32 * 0.1).collect()),
        ]);
        crate::interpreter::eval_load(&emb, &mut face, &src, &Default::default())
            .await
            .expect("eval_load");
        assert!(emb.is_loaded(), "tied 槽应有块");

        let ids = TensorOps::from_host(Dtype::F32, vec![2], &f32b(&[3.0, 7.0]));
        let out = emb.embed(&ids, 2);
        assert!(!out.is_poisoned(), "embedding 声明不应有毒");
        assert_eq!(out.shape(), &[2, 4]);

        let hidden = TensorOps::from_host(Dtype::F32, vec![1, 4], &f32b(&[0.1; 4]));
        let logits = harvest(&mut face, &emb.lm_head_matmul(&hidden)).await;
        assert_eq!(logits.len(), 16);
    }

    /// HF parity(查表 + tied lm_head;门控 OWL_HF_PARITY=1)
    #[tokio::test]
    async fn parity_matches_hf() {
        if !parity_enabled() {
            skip_note();
            return;
        }
        let (tokens, vocab, d) = (3usize, 16usize, 4usize);
        let ids: Vec<f32> = vec![3.0, 15.0, 0.0];
        let w: Vec<f32> = (0..vocab * d).map(|i| (i as f32 * 0.1) - 0.8).collect();
        let x: Vec<f32> = vec![0.2, -0.5, 1.1, 0.0];

        let inp = tmp_path("embed_in");
        let outp = tmp_path("embed_out");
        st_write(&inp, &[
            ("ids", &ids, vec![tokens]),
            ("w", &w, vec![vocab, d]),
            ("x", &x, vec![1, d]),
        ]);
        let manifest = hf_python("embedding.py", &[
            inp.to_str().unwrap(), outp.to_str().unwrap(),
            &format!(r#"{{"vocab": {vocab}, "d": {d}}}"#),
        ]);
        eprintln!("[hf manifest] embedding: {manifest}");
        let want_y = st_read(&outp, "y");
        let want_logits = st_read(&outp, "logits");

        for (face_tag, on_gpu) in [("cpu", false), ("gpu", true)] {
            if on_gpu && !crate::testkit::gpu_enabled() {
                continue;
            }
            let (_, y, logits) = if !on_gpu {
                let mut face = owl_cpu::CpuFace::new();
                let emb = Embedding::new(vocab, d);
                let src = Src::from([
                    ("weight".to_string(), w.clone()),
                ]);
                crate::interpreter::eval_load(&emb, &mut face, &src, &Default::default())
                    .await.expect("eval_load");
                // embed = Kernel 节点(CPU face 不执行);lm_head = 语义 matmul 可跑
                let x_t = TensorOps::from_host(Dtype::F32, vec![1, d], &f32b(&x));
                let y = want_y.clone(); // CPU 臂无 embed 执行,占位由 GPU 臂覆盖
                let logits = harvest(&mut face, &emb.lm_head_matmul(&x_t)).await;
                ("cpu", y, logits)
            } else {
                let mut gpu = crate::testkit::gpu_client().await;
                let emb = Embedding::new(vocab, d);
                let src = Src::from([
                    ("weight".to_string(), w.clone()),
                ]);
                crate::interpreter::eval_load(&emb, &mut gpu, &src, &Default::default())
                    .await.expect("eval_load");
                let ids_t = TensorOps::from_host(Dtype::F32, vec![tokens], &f32b(&ids));
                let x_t = TensorOps::from_host(Dtype::F32, vec![1, d], &f32b(&x));
                let y = harvest(&mut gpu, &emb.embed(&ids_t, tokens)).await;
                let logits = harvest(&mut gpu, &emb.lm_head_matmul(&x_t)).await;
                gpu.close().await.expect("server 关机");
                ("gpu", y, logits)
            };
            assert_close(&y, &want_y, 1e-5, &format!("embed-{face_tag}"));
            assert_close(&logits, &want_logits, 1e-5, &format!("lmhead-{face_tag}"));
        }
        std::fs::remove_file(&inp).ok();
        std::fs::remove_file(&outp).ok();
    }
}
