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
        crate::interpreters::eval_load(&emb, &mut face, &src, &Default::default())
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
                crate::interpreters::eval_load(&emb, &mut face, &src, &Default::default())
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
                crate::interpreters::eval_load(&emb, &mut gpu, &src, &Default::default())
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

#[cfg(test)]
mod f16_tests {
    use super::*;
    use crate::testkit::{gpu_client, gpu_enabled};
    use crate::contract::DeviceClient as _;
    use crate::tensor::Dtype;

    fn half_le(f: f32) -> [u8; 2] {
        half::f16::from_f32(f).to_le_bytes()
    }

    /// GPU:f16 embed 查表 vs host f32(F3 收口;OWL_TEST_DEVICE 门控)
    #[tokio::test]
    async fn gpu_embed_f16_matches_host() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (vocab, d) = (8usize, 16usize);
        // 源值过 f16 量化(上载即 f16;查表正确性 = 对量化后参考逐位)
        let w: Vec<f32> = (0..vocab * d)
            .map(|i| half::f16::from_f32(((i as f32 * 0.37) - 3.0).sin()).to_f32())
            .collect();
        let ids = [2.0f32, 5.0f32, 7.0f32];
        let tokens = 3usize;

        let mut gpu = gpu_client().await;
        let dw = gpu.htod(Dtype::F16, &crate::contract::Shape::from(vec![vocab, d]),
            &w.iter().flat_map(|f| half_le(*f)).collect::<Vec<u8>>()).await.expect("htod w");
        let di = gpu.htod(Dtype::F32, &crate::contract::Shape::from(vec![tokens]),
            &crate::testkit::f32b(&ids)).await.expect("htod ids");

        // htod 块 → Block 叶子声明(f16 链不走 CpuFace 的 from_host)
        let w_decl = TensorOps::of_block(dw.id, Dtype::F16, vec![vocab, d]);
        let i_decl = TensorOps::of_block(di.id, Dtype::F32, vec![tokens]);
        let decl = TensorOps::of(crate::kernel::kernel_with(
            "owl_embed_f16", (tokens as u32, 1, 1), (128, 1, 1), 0,
        ))
        .arg(&w_decl)
        .arg(&i_decl)
        .arg_usize(d)
        .with_shape(Dtype::F16, vec![tokens, d]);
        let out = crate::interpreters::eval_ops(decl.step(), &mut gpu).await.expect("eval");
        let mut buf = vec![0u8; tokens * d * 2];
        gpu.dtoh(&out, &mut buf).await.expect("dtoh");

        for t in 0..tokens {
            let id = ids[t] as usize;
            for dd in 0..d {
                let got = half::f16::from_le_bytes([buf[(t * d + dd) * 2], buf[(t * d + dd) * 2 + 1]]).to_f32();
                assert_eq!(got, w[id * d + dd], "查表须逐位一致 [{t},{dd}]");
            }
        }
        gpu.close().await.expect("关机");
    }
}
