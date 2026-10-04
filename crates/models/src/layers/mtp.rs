//! MTP(Multi-Token Prediction)头 —— Qwen3.5/3.8 系检查点自带的单层草稿模块
//! (2026-10-05,E5-M0;cyankiwi/Qwen3.8-27B-AWQ-INT4 实测 15 键全 BF16)。
//!
//! 结构 = vLLM `Qwen3NextMultiTokenPredictor` 同构(逐键核对):
//! - `mtp.fc` [hidden, 2×hidden]:concat(embedding_norm, hidden_norm) → fc
//! - `mtp.layers.0`:完整 decoder layer(**门控注意力与 target 同款**
//!   q_proj [12288,5120] = q⊕gate 融合 —— owl Attention 原生)+ mlp
//! - `mtp.pre_fc_norm_hidden / pre_fc_norm_embedding / norm`:三 (1+w) 系
//!   rmsnorm
//! - embed / lm_head 与 target **共享**(不持有;由引擎喂入)
//!
//! 草稿语义(E5-M1):从 (anchor token, anchor 位 target 末层 hidden) 出发,
//! fc → layer → norm → lm_head → argmax 出草稿 token,hidden 链内自反馈
//! k 步(本文件 M0 范围 = 结构 + 装载 + fc 路径冒烟;链与 KV 属 M1)。
//!
//! 装载:键映射(Qwen35Convention::layer_key 同式,base="mtp",i=0)住
//! specs(qwen35.rs 的 `impl Loadable for MtpPredictor`)—— 家规:模型特有
//! 键约定住 specs,本组件只持结构与数学。

use crate::layers::decoder::DecoderLayer;
use crate::layers::embedding::Embedding;
use crate::layers::linear::Linear;
use crate::layers::rmsnorm::RmsNorm;
use crate::layers::rope::Rope;
use crate::module::{ForwardCtx, KvBuffers, Module, QuantPlan};
use crate::TensorOps;

pub struct MtpPredictor {
    /// fc 列半拆双子(E5-M2b;M0 挂账兑现):W [hidden, 2·hidden] 行主序
    /// 按 in 维前后半拆为两块连续 [hidden, hidden](awq 源派生键去交织,
    /// 装载一次性);fc_path 消每轮 2×50MB narrow 物化。n=hidden=5120 过
    /// marlin 谓词,marlin 化留二期。
    fc_e: Linear,
    fc_h: Linear,
    layer: DecoderLayer,
    pre_fc_norm_hidden: RmsNorm,
    pre_fc_norm_embedding: RmsNorm,
    norm: RmsNorm,
    hidden: usize,
}

impl MtpPredictor {
    /// 几何 = target full 层同款(27B:24H/4KV/256,hidden 5120,mlp 17408;
    /// eps 与 target 一致 1e-6)。plan 恒 F16(检查点 MTP 头为 BF16 未量化,
    /// 经源侧 bf16→f16 归一;marlin 化 = 二期)。
    pub fn new(hidden: usize, inter: usize, hq: usize, hkv: usize, hd: usize, eps: f32) -> Self {
        Self {
            fc_e: Linear::new("fc_e", hidden, hidden, QuantPlan::F16),
            fc_h: Linear::new("fc_h", hidden, hidden, QuantPlan::F16),
            layer: DecoderLayer::new_full(hq, hkv, hd, hidden, inter, eps, QuantPlan::F16),
            pre_fc_norm_hidden: RmsNorm::new_add_one("pre_fc_norm_hidden", hidden, eps),
            pre_fc_norm_embedding: RmsNorm::new_add_one("pre_fc_norm_embedding", hidden, eps),
            norm: RmsNorm::new_add_one("norm", hidden, eps),
            hidden,
        }
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// fc 路径:pre_fc_norm×2 → 双子 GEMM(y = We·e + Wh·h;W 列半拆
    /// 已在装载源完成 —— 权重连续 [hidden,hidden]×2,零运行期 narrow)。
    pub fn fc_path(
        &self,
        embedding: &TensorOps,
        target_hidden: &TensorOps,
        ctx: &crate::module::ForwardCtx,
    ) -> TensorOps {
        let e = self.pre_fc_norm_embedding.forward(embedding, ctx);
        let h = self.pre_fc_norm_hidden.forward(target_hidden, ctx);
        let we = self.fc_e.weight_decl();
        let wh = self.fc_h.weight_decl();
        e.matmul_nt(&we).add(&h.matmul_nt(&wh))
    }

    /// norm 出口(lm_head 前末 norm;M1 草稿链每步消费)
    pub fn norm(&self, x: &TensorOps, ctx: &crate::module::ForwardCtx) -> TensorOps {
        self.norm.forward(x, ctx)
    }

    /// 内层 decoder layer 访问器(M1:草稿链前向;M0 仅装载)
    pub(crate) fn layer(&self) -> &DecoderLayer {
        &self.layer
    }

    /// 单步 MTP 前向(fc_path → layer → norm;post-norm hidden 出)。
    /// ctx = 调用方构造的 decode 语境(pos/rope/mtp 链 KV;attention
    /// decode 臂先写后打分,单步自含)。target_hidden = 上一 token 位的
    /// target 末层 hidden(post-final-norm,即 lm_head 消费面)。
    pub fn forward_step(
        &self,
        token_embed: &TensorOps,
        target_hidden: &TensorOps,
        ctx: &ForwardCtx,
    ) -> TensorOps {
        let x = self.fc_path(token_embed, target_hidden, ctx);
        let x = self.layer().forward(&x, ctx);
        self.norm(&x, ctx)
    }

    /// 草稿链(E5-M1;k 步自反馈,全 SSA 单树 —— argmax 反馈为树内边,
    /// 零 host 往返):step i:(embed(tok_i), h_{i-1}) → logits_i →
    /// tok_{i+1} = argmax,h_i = post-norm hidden。tok_0 = anchor,
    /// h_0 = anchor_hidden(其采样 hidden = anchor 前一位的 target 末层
    /// hidden,DeepSeek MTP 语义;配对实证挂 M2 恒等门)。
    /// 返回 [k, 1] f32 草稿 token(行 i = 第 i 步)。
    #[allow(clippy::too_many_arguments)]
    pub fn propose(
        &self,
        anchor_token: &TensorOps,
        anchor_hidden: &TensorOps,
        k: usize,
        embed: &Embedding,
        rope: &Rope,
        mtp_kvs: &[KvBuffers],
        pos_t: &[TensorOps],
        vocab: usize,
        env: crate::env::EnvProvider,
    ) -> TensorOps {
        assert_eq!(mtp_kvs.len(), k, "mtp 链缓冲数 = k");
        assert_eq!(pos_t.len(), k, "pos 表数 = k");
        let mut tok = anchor_token.clone();
        let mut h = anchor_hidden.clone();
        let mut drafts: Vec<TensorOps> = Vec::with_capacity(k);
        for i in 0..k {
            let mut ctx = ForwardCtx::decode(1, &pos_t[i], &mtp_kvs[i], rope);
            ctx.env = env;
            let emb = embed.embed(&tok, 1);
            let out = self.forward_step(&emb, &h, &ctx);
            let logits = embed.lm_head_matmul(&out);
            let d = crate::ops::argmax_f32idx(&logits, vocab, 0);
            drafts.push(d.clone());
            tok = d;
            h = out;
        }
        let refs: Vec<&TensorOps> = drafts.iter().collect();
        crate::layers::concat_rows(&refs, 1, 1)
    }

    /// 草稿链 v2(E5-M2b 施工方案 §九;extend 批行 + 链步):
    /// - extend 行 i = (h_{F+i}, emb(t_{F+i+1})) @ 位置 F+i,i = 0..M
    ///   (M = tokens.len()-1 = 已接受数;vLLM 配对:token 左移一位、
    ///   hidden 原位,处理在 hidden 的位置)—— prefill 形 ctx 批行前向;
    /// - 末行输出 = 链种子 hidden',argmax = d1(预测 t_{F+M+2});
    /// - 链步 j = 1..k-1:(emb(d_j), h'_prev) @ 位置 F+M+j(decode 形
    ///   ctx,先写后打分)→ d_{j+1}。
    /// 返回 [k, 1] f32 草稿(行 i = d_{i+1})。调用方建表契约:
    /// extend slots/lens = [M+1](物理槽 F..F+M / kv_len F+1..F+M+1),
    /// chain 第 j 步 slots/lens = [1](物理槽 F+M+j / kv_len F+M+j+1)。
    #[allow(clippy::too_many_arguments)]
    pub fn propose_ext(
        &self,
        tokens: &TensorOps,
        hiddens: &TensorOps,
        extend_pos: &TensorOps,
        extend_kv: &KvBuffers,
        extend_slots: &TensorOps,
        extend_lens: &TensorOps,
        chain: &[(&TensorOps, &KvBuffers)],
        embed: &Embedding,
        rope: &Rope,
        vocab: usize,
        env: crate::env::EnvProvider,
    ) -> TensorOps {
        let m1 = tokens.shape[0];
        assert_eq!(hiddens.shape[0], m1, "extend 行数 = token 数");
        // extend 批行(prefill 形;fi=None → paged/naive,MTP 层专用)
        let mut ctx = crate::module::ForwardCtx::attn_prefill(
            m1, extend_pos, extend_kv, rope, extend_slots, extend_lens,
        );
        ctx.env = env;
        let emb = embed.embed(tokens, m1);
        let x = self.fc_path(&emb, hiddens, &ctx);
        let x = self.layer().forward(&x, &ctx);
        let x = self.norm(&x, &ctx);
        // 末行窄切(slice_view 连续视图,零拷贝)= 链种子;argmax = d1
        let mut h = x.slice_view((m1 - 1) * self.hidden, vec![1, self.hidden]);
        let logits = embed.lm_head_matmul(&h);
        let mut tok = crate::ops::argmax_f32idx(&logits, vocab, 0);
        let mut drafts = vec![tok.clone()];
        // 链步(decode 形;自反馈同 propose)
        for (pos_t, kv) in chain {
            let mut c = crate::module::ForwardCtx::decode(1, pos_t, kv, rope);
            c.env = env;
            let emb = embed.embed(&tok, 1);
            let out = self.forward_step(&emb, &h, &c);
            let lg = embed.lm_head_matmul(&out);
            tok = crate::ops::argmax_f32idx(&lg, vocab, 0);
            drafts.push(tok.clone());
            h = out;
        }
        let refs: Vec<&TensorOps> = drafts.iter().collect();
        crate::layers::concat_rows(&refs, 1, 1)
    }

    /// 装载面访问器(specs 侧 Loadable layout 用;pub(crate) 同 crate)
    pub(crate) fn fc_e(&self) -> &Linear {
        &self.fc_e
    }
    pub(crate) fn fc_h(&self) -> &Linear {
        &self.fc_h
    }
    pub(crate) fn pre_fc_norm_hidden(&self) -> &RmsNorm {
        &self.pre_fc_norm_hidden
    }
    pub(crate) fn pre_fc_norm_embedding(&self) -> &RmsNorm {
        &self.pre_fc_norm_embedding
    }
    pub(crate) fn norm_head(&self) -> &RmsNorm {
        &self.norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Loadable;
    use crate::tensor::Dtype;
    use crate::testkit::{assert_close, gpu_client, gpu_enabled, harvest_f16};
    use std::path::Path;

    /// E5-M0 验收:cyankiwi 检查点 MTP 头装载(15 键)+ fc 路径 host 对拍。
    /// 门控 OWL_TEST_DEVICE + OWL_AWQ27B_DIR(cyankiwi 目录)。
    /// host 参考 = 权重源直取(r16 对齐 f16 装载域)→ (1+w) rmsnorm →
    /// concat → fc matmul(全 f32)。
    #[tokio::test]
    async fn gpu_mtp_head_loads_from_awq27b() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let Ok(dir) = std::env::var("OWL_AWQ27B_DIR") else {
            eprintln!("skip: OWL_AWQ27B_DIR 未设(cyankiwi 检查点目录)");
            return;
        };
        let dir = Path::new(&dir);

        // ── host 参考:独立源直取(r16 = f16 装载域)──
        use crate::module::WeightSource;
        let mut src = crate::formats::awq::AwqSource::open_dir(dir).expect("源打开");
        let r16 = |x: f32| half::f16::from_f32(x).to_f32();
        let take = |src: &mut crate::formats::awq::AwqSource, k: &str| -> Vec<f32> {
            src.take(k).unwrap_or_else(|| panic!("键缺失 {k}"))
        };
        let w_fc = take(&mut src, "mtp.fc.weight").iter().map(|&x| r16(x)).collect::<Vec<_>>();
        let w_pfh = take(&mut src, "mtp.pre_fc_norm_hidden.weight")
            .iter()
            .map(|&x| r16(x))
            .collect::<Vec<_>>();
        let w_pfe = take(&mut src, "mtp.pre_fc_norm_embedding.weight")
            .iter()
            .map(|&x| r16(x))
            .collect::<Vec<_>>();

        let hidden = 5120usize;
        let eps = 1e-6f32;
        let gen = |n: usize, seed: f32| {
            (0..n).map(|i| r16(((i as f32 + seed) * 0.17).sin() * 0.8)).collect::<Vec<f32>>()
        };
        let emb = gen(hidden, 7.0);
        let th = gen(hidden, 8.0);

        let host_norm = |x: &[f32], w: &[f32]| -> Vec<f32> {
            let mean = x.iter().map(|&v| v * v).sum::<f32>() / x.len() as f32;
            let inv = 1.0 / (mean + eps).sqrt();
            x.iter().zip(w).map(|(&v, &g)| v * inv * (1.0 + g)).collect()
        };
        let e_n = host_norm(&emb, &w_pfe);
        let h_n = host_norm(&th, &w_pfh);
        // fc: y[o] = Σ_i W[o][i]·x[i],x = concat(e_n, h_n)
        let mut want = vec![0.0f32; hidden];
        for (o, row) in want.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for i in 0..hidden {
                acc += w_fc[o * 2 * hidden + i] * e_n[i];
                acc += w_fc[o * 2 * hidden + hidden + i] * h_n[i];
            }
            *row = acc;
        }

        // ── device:装载 + fc_path ──
        let mut gpu = gpu_client().await;
        let (mtp, manifest) = crate::specs::qwen35::load_27b_mtp(dir, &mut gpu)
            .await
            .expect("MTP 头装载");
        assert_eq!(mtp.hidden(), hidden);

        // 键集断言:16 键全 mtp.* 且逐键符合结构(fc 列半拆 → fc_e/fc_h)
        let mut keys: Vec<String> =
            manifest.entries().iter().map(|e| e.key.clone()).collect();
        keys.sort();
        assert_eq!(keys.len(), 16, "MTP 头键数 {keys:?}");
        for k in &keys {
            assert!(k.starts_with("mtp."), "键前缀 {k}");
        }
        for expect in [
            "mtp.fc_e.weight",
            "mtp.fc_h.weight",
            "mtp.norm.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
        ] {
            assert!(keys.iter().any(|k| k == expect), "缺键 {expect}");
        }

        // fc_path 对拍(embedding/target_hidden 全 f32 域 = 已 r16 输入)
        let ctx = ForwardCtx::minimal(1);
        let emb_t = TensorOps::from_host(Dtype::F16, vec![1, hidden], &f16bytes(&emb));
        let th_t = TensorOps::from_host(Dtype::F16, vec![1, hidden], &f16bytes(&th));
        let decl = mtp.fc_path(&emb_t, &th_t, &ctx);
        let got = harvest_f16(&mut gpu, &decl).await;
        assert_close(&got, &want, 2e-2, "mtp fc_path");
        gpu.close().await.expect("关机");
    }

    fn f16bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
    }
}

#[cfg(test)]
mod wiring_tests {
    use super::*;
    use crate::tensor::Dtype;
    use crate::module::{Loadable, LoaderCtx};
    use crate::testkit::{f32b, gpu_client, gpu_enabled};
    use std::collections::HashMap;

    /// E5-M1 接线金标(tiny 形状,GPU):propose(k=3) 逐位 ≡ 三次手工
    /// forward_step 组合(feedback:argmax → embed → fc;链内 ctx 逐步
    /// 递进 pos/kv_lens)。组件数学(fc/DecoderLayer)由 M0 金标与整模
    /// 套件另证,本测证链装配。
    #[tokio::test]
    async fn gpu_mtp_propose_chain_wiring() {
        if !gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let (hidden, inter, hq, hkv, hd, vocab) = (32usize, 32usize, 2usize, 1usize, 16usize, 11usize);
        let eps = 1e-6f32;
        let gen = |n: usize, seed: f32| {
            (0..n).map(|i| half::f16::from_f32(((i as f32 + seed) * 0.23).sin() * 0.7).to_f32()).collect::<Vec<f32>>()
        };
        let hb = |v: &[f32]| -> Vec<u8> {
            v.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect()
        };

        // 草稿头权重(mtp.* 键面;HashMap 源;fc 列半拆 = 两块 [h,h])
        let mut src: HashMap<String, Vec<f32>> = HashMap::new();
        src.insert("mtp.fc_e.weight".into(), gen(hidden * hidden, 1.0));
        src.insert("mtp.fc_h.weight".into(), gen(hidden * hidden, 101.0));
        for k in ["input_layernorm", "post_attention_layernorm"] {
            src.insert(format!("mtp.layers.0.{k}.weight"), gen(hidden, 2.0));
        }
        src.insert("mtp.layers.0.self_attn.q_proj.weight".into(), gen(2 * hq * hd * hidden, 3.0));
        src.insert("mtp.layers.0.self_attn.k_proj.weight".into(), gen(hkv * hd * hidden, 4.0));
        src.insert("mtp.layers.0.self_attn.v_proj.weight".into(), gen(hkv * hd * hidden, 5.0));
        src.insert("mtp.layers.0.self_attn.o_proj.weight".into(), gen(hidden * hq * hd, 6.0));
        src.insert("mtp.layers.0.self_attn.q_norm.weight".into(), gen(hd, 7.0));
        src.insert("mtp.layers.0.self_attn.k_norm.weight".into(), gen(hd, 8.0));
        src.insert("mtp.layers.0.mlp.gate_proj.weight".into(), gen(inter * hidden, 9.0));
        src.insert("mtp.layers.0.mlp.up_proj.weight".into(), gen(inter * hidden, 10.0));
        src.insert("mtp.layers.0.mlp.down_proj.weight".into(), gen(hidden * inter, 11.0));
        src.insert("mtp.norm.weight".into(), gen(hidden, 12.0));
        src.insert("mtp.pre_fc_norm_hidden.weight".into(), gen(hidden, 13.0));
        src.insert("mtp.pre_fc_norm_embedding.weight".into(), gen(hidden, 14.0));

        let mut gpu = gpu_client().await;
        let lctx = LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false };

        let mtp = MtpPredictor::new(hidden, inter, hq, hkv, hd, eps);
        crate::interpreters::eval_load(&mtp, &mut gpu, &src, &lctx).await.expect("mtp 装载");

        // 共享 embed/lm_head(tiny 独立方)
        let mut esrc: HashMap<String, Vec<f32>> = HashMap::new();
        esrc.insert("weight".into(), gen(vocab * hidden, 20.0));
        esrc.insert("lm_head.weight".into(), gen(vocab * hidden, 21.0));
        let embed = Embedding::new_untied(vocab, hidden);
        struct EmbWithHead<'a> {
            e: &'a Embedding,
        }
        impl crate::module::Loadable for EmbWithHead<'_> {
            fn layout(&self, ctx: &LoaderCtx) -> crate::module::LoaderOps {
                self.e.layout(ctx).chain(self.e.layout_lm_head(ctx).unwrap())
            }
        }
        crate::interpreters::eval_load(&EmbWithHead { e: &embed }, &mut gpu, &esrc, &lctx)
            .await
            .expect("embed 装载");

        let rope = Rope::new(64, hd, 4, 10_000.0).expect("rope");
        crate::interpreters::eval_load(&rope, &mut gpu, &rope.tables(), &lctx).await.expect("rope 表");

        // mtp 链 KV(paged;page 32 x 8;nb=1 恒等表;三步递进 lens 1/2/3)
        let (page, x, nb) = (32usize, 8usize, 1usize);
        let kc_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd / x, page, x]).step(), &mut gpu)
            .await.expect("kc");
        let vc_b = crate::interpreters::eval_ops(
            TensorOps::zeros(Dtype::F16, vec![nb, hkv, hd, page]).step(), &mut gpu)
            .await.expect("vc");
        let bt = TensorOps::from_host(Dtype::F32, vec![1, nb], &f32b(&(0..nb).map(|i| i as f32).collect::<Vec<_>>()));
        let mtp_kv = |slot: f32, len: f32| KvBuffers {
            k_cache: TensorOps::of_block(kc_b.id, Dtype::F16, vec![nb, hkv, hd / x, page, x]),
            v_cache: TensorOps::of_block(vc_b.id, Dtype::F16, vec![nb, hkv, hd, page]),
            slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot])),
            kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[len])),
            block_tables: bt.clone(),
        };
        let kvs: Vec<KvBuffers> = (0..3).map(|i| mtp_kv(i as f32, (i + 1) as f32)).collect();
        let pos_t: Vec<TensorOps> = (0..3)
            .map(|i| TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[i as f32])))
            .collect();

        // 锚:token 3,hidden 行(确定性)
        let anchor_token = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[3.0]));
        let anchor_hidden = TensorOps::from_host(Dtype::F16, vec![1, hidden], &hb(&gen(hidden, 30.0)));

        // propose(k=3)
        let drafts = mtp.propose(
            &anchor_token, &anchor_hidden, 3, &embed, &rope, &kvs, &pos_t, vocab,
            crate::env::EnvProvider::default(),
        );
        let got = harvest_drafts(&mut gpu, &drafts).await;
        assert_eq!(got.len(), 3, "草稿数");

        // 手工组合(同组件逐步;反馈链显式)
        let mut tok = anchor_token.clone();
        let mut h = anchor_hidden.clone();
        let mut want = Vec::new();
        for i in 0..3 {
            let ctx = ForwardCtx::decode(1, &pos_t[i], &kvs[i], &rope);
            let emb = embed.embed(&tok, 1);
            let out = mtp.forward_step(&emb, &h, &ctx);
            let logits = embed.lm_head_matmul(&out);
            let d = crate::ops::argmax_f32idx(&logits, vocab, 0);
            want.push(harvest_drafts(&mut gpu, &d).await.remove(0));
            tok = d;
            h = out;
        }
        assert_eq!(got, want, "propose 链 ≡ 手工组合(接线)");
        gpu.close().await.expect("关机");
    }

    async fn harvest_drafts(gpu: &mut owl_cuda::GpuClient, t: &TensorOps) -> Vec<f32> {
        crate::testkit::harvest(gpu, t).await
    }
}
