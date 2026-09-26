//! 通用解码器主干(批8,2026-09-26 拆分):embed + N×DecoderLayer
//! + final norm + tied lm_head —— **所有模型共有**的机制,凡本主干
//! 词汇表内(Full|Gdn 双 mixer)的 hybrid / dense 解码器共用。
//! 旧世界 `Qwen3_5ForCausalLM`(engine/src/models/qwen3_5.rs)的
//! 声明式重铸 —— 砍掉的是命令式缓存管理/MTP/DFlash 支线,留下的是
//! 主干数据流。
//!
//! **拆分律(2026-09-26 用户裁决)**:
//! - 本文件只住**共有机制**(主干结构/单树求值/序列 ctx 派生/C10
//!   分组件装载/fixture 测试),零模型特定参数;
//! - **模型特有**住 `specs/<model>.rs`(维度预设/层型表约定/检查点
//!   特例,纯参数事实零机制)—— 现存 [`crate::specs::qwen35`];
//! - 新模型 = 新 spec 文件;mixer 词汇超出 Full|Gdn(MoE/MLA/…)时
//!   另立 DecoderLayer 扩展口(需求基线:arch 分发表不写死),不动主干。
//!
//! ```text
//! ids ─→ embed ─→ layers[0] ─→ … ─→ layers[N-1] ─→ norm ─→ lm_head(tied) ─→ logits
//! ```
//!
//! ```text
//! ids ─→ embed ─→ layers[0] ─→ … ─→ layers[N-1] ─→ norm ─→ lm_head(tied) ─→ logits
//! ```
//!
//! 三条既有律在此收口:
//! - **C5 编排**:整模单树 —— 一次 `forward` = 一棵树 = 一次 eval;
//!   捕获/回放同一棵树(M-f 前提);
//! - **C10 装载**:Want 键恒层内短键,checkpoint 全局键由
//!   [`module::KeyConvention`] 改写(默认 [`module::LlamaFamily`];
//!   模型特有约定住 specs/<model>.rs);tied embedding 单源键双 Want;
//! - **CSE/DAG**:层内残差共享子树只求值一次;跨步状态(KV/GDN)
//!   靠常驻块 + 每步重建声明推进。

use crate::contract::Dtype;
use crate::layers::decoder::DecoderLayer;
use crate::layers::embedding::Embedding;
use crate::layers::rmsnorm::RmsNorm;
use crate::module::{ForwardCtx, KeyConvention, Loadable, LoaderCtx, LoaderOps, Module};
use crate::TensorOps;

// ============================================================================
// §1 ModelSpec:结构参数(声明期常量;纯数据零逻辑 —— 预设构造器
// 归 specs/<model>.rs,此处不认识任何具体模型)
// ============================================================================

/// 模型结构参数(旧世界 Config 的主干切片;全部声明期常量)
pub struct ModelSpec {
    pub vocab: usize,
    pub hidden: usize,
    pub inter: usize,
    /// full attention 头 (hq, hkv, hd)
    pub full_heads: (usize, usize, usize),
    /// GDN 头 (nk, hk_dim, nv, hv_dim)
    pub gdn_heads: (usize, usize, usize, usize),
    pub eps: f32,
    /// 层混型表(true = full_attention;dense 全 true;hybrid 按检查点
    /// layer_types 填)
    pub layer_types: Vec<bool>,
}

// ============================================================================
// §2 Model:主干容器(纯声明)
// ============================================================================

pub struct Model {
    /// tied:[vocab, hidden] 直读 + [hidden, vocab] 转置槽(单源键)
    pub embed: Embedding,
    pub layers: Vec<DecoderLayer>,
    /// final norm(checkpoint 键 `{base}.norm.weight`)
    pub norm: RmsNorm,
    /// 检查点键名约定(layout 期改写局部键;模型特有约定住 specs)
    keys: Box<dyn KeyConvention + Send + Sync>,
    hidden: usize,
    vocab: usize,
}

impl Model {
    /// 准备容器(纯元数据;零数据零副作用)。keys = 检查点键名约定
    /// (默认 [`LlamaFamily`];Qwen3.5 等特有约定由 specs 传入)。
    pub fn new(spec: &ModelSpec, keys: impl KeyConvention + Send + Sync + 'static) -> Model {
        let layers = spec
            .layer_types
            .iter()
            .map(|&full| {
                if full {
                    let (hq, hkv, hd) = spec.full_heads;
                    DecoderLayer::new_full(hq, hkv, hd, spec.hidden, spec.inter, spec.eps)
                } else {
                    let (nk, hk_dim, nv, hv_dim) = spec.gdn_heads;
                    DecoderLayer::new_gdn(nk, hk_dim, nv, hv_dim, spec.hidden, spec.inter, spec.eps)
                }
            })
            .collect();
        Model {
            embed: Embedding::new(spec.vocab, spec.hidden),
            layers,
            norm: RmsNorm::new_add_one("norm", spec.hidden, spec.eps),
            keys: Box::new(keys),
            hidden: spec.hidden,
            vocab: spec.vocab,
        }
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab
    }


    /// per-layer 子 ctx(整模 ctx → 单层 ctx;kvs/gdns 按混型游标取,
    /// Attention/Gdn 层读单引用零感知)
    pub(crate) fn layer_ctx<'a>(
        &self,
        ctx: &'a ForwardCtx<'a>,
        kvi: usize,
        gi: usize,
    ) -> ForwardCtx<'a> {
        ForwardCtx {
            tokens: ctx.tokens,
            pos: ctx.pos,
            kv: ctx.kvs.and_then(|s| s.get(kvi)),
            rope: ctx.rope,
            gdn: ctx.gdns.and_then(|s| s.get(gi)),
            kvs: None,
            gdns: None,
        }
    }

    /// 主干声明至 last hidden(pre-lm_head;MTP/深栈支线的挂点)
    pub fn last_hidden(&self, ids: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        let mut xs = self.embed.forward(ids, ctx);
        let (mut kvi, mut gi) = (0usize, 0usize);
        for layer in &self.layers {
            let sub = self.layer_ctx(ctx, kvi, gi);
            xs = layer.forward(&xs, &sub);
            if layer.is_full() {
                kvi += 1;
            } else {
                gi += 1;
            }
        }
        self.norm.forward(&xs, ctx)
    }
}

impl Module for Model {
    /// 整模单树(C5):embed → 层链 → final norm → tied lm_head
    fn forward(&self, ids: &TensorOps, ctx: &ForwardCtx) -> TensorOps {
        self.embed.lm_head_matmul(&self.last_hidden(ids, ctx))
    }
}

impl Loadable for Model {
    /// 整模装载需求(C10):embed / layers.{i} / norm 三段子清单 chain
    /// 聚合,局部键经 KeyConvention 改写为 checkpoint 全键 —— 改写
    /// 发生在声明期,Want 清单里已是最终键,解释器零约定感知。
    /// tied embedding 的双 Want(直读 + 转置)同键改写,天然保持同键。
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        let mut ops = self.embed.layout(ctx).map_keys(|k| self.keys.embed_key(&k));
        for (i, layer) in self.layers.iter().enumerate() {
            ops = ops.chain(layer.layout(ctx).map_keys(|k| self.keys.layer_key(i, &k)));
        }
        ops.chain(self.norm.layout(ctx).map_keys(|k| self.keys.norm_key(&k)))
    }
}

// 保持 Dtype 引用(层文件统一风格)
const _: Dtype = Dtype::F32;

// ============================================================================
// 测试:fixture 装载 + 声明良构(CPU)+ 两步 decode vs host 全链(GPU)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::gdn::GdnBuffers;
    use crate::module::LlamaFamily;
    use crate::layers::rope::Rope;
    use crate::module::KvBuffers;
    use crate::testkit::{assert_close, f32b, gpu_client, gpu_enabled, skip_note};
    use crate::tensor::Dtype;
    use std::collections::HashMap;

    // fixture 维度(缩水档;两 mixer + 3:1 周期一轮)
    const VOCAB: usize = 16;
    const HIDDEN: usize = 6;
    const INTER: usize = 5;
    const HQ: usize = 2;
    const HKV: usize = 1;
    const HD: usize = 4;
    const NK: usize = 2;
    const HK_DIM: usize = 4;
    const NV: usize = 2;
    const HV_DIM: usize = 4;
    const EPS: f32 = 1e-6;
    const MAX_SLOTS: usize = 4;
    const KV_ROWS: usize = 8;

    fn spec() -> ModelSpec {
        ModelSpec {
            vocab: VOCAB,
            hidden: HIDDEN,
            inter: INTER,
            full_heads: (HQ, HKV, HD),
            gdn_heads: (NK, HK_DIM, NV, HV_DIM),
            eps: EPS,
            layer_types: vec![false, false, false, true], // G,G,G,F
        }
    }

    fn gen(seed: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed) * 0.13).sin() * 0.5).collect()
    }

    /// checkpoint 全键源(经 LlamaFamily 约定走 C10 真路径)
    fn checkpoint_src() -> HashMap<String, Vec<f32>> {
        let base = "model.language_model";
        let mut m: HashMap<String, Vec<f32>> = HashMap::new();
        let mut put = |k: String, n: usize, seed: f32| {
            m.insert(k, gen(seed, n));
        };
        put(format!("{base}.embed_tokens.weight"), VOCAB * HIDDEN, 100.0);
        for i in 0..4 {
            let mut p = |k: &str, n: usize, seed: f32| {
                put(format!("{base}.layers.{i}.{k}.weight"), n, seed);
            };
            p("input_layernorm", HIDDEN, 1.0 + i as f32);
            p("post_attention_layernorm", HIDDEN, 2.0 + i as f32);
            if i < 3 {
                // GDN 层
                let key_dim = NK * HK_DIM;
                let value_dim = NV * HV_DIM;
                p("in_proj_qkv", (2 * key_dim + value_dim) * HIDDEN, 3.0);
                p("in_proj_z", value_dim * HIDDEN, 4.0);
                p("in_proj_b", NV * HIDDEN, 5.0);
                p("in_proj_a", NV * HIDDEN, 6.0);
                p("out_proj", HIDDEN * value_dim, 7.0);
                p("conv1d", (2 * key_dim + value_dim) * 4, 8.0);
                p("A_log", NV, 9.0);
                p("dt_bias", NV, 10.0);
                p("norm", HV_DIM, 11.0);
            } else {
                // full 层
                p("q_proj", HQ * HD * 2 * HIDDEN, 3.0);
                p("k_proj", HKV * HD * HIDDEN, 4.0);
                p("v_proj", HKV * HD * HIDDEN, 5.0);
                p("o_proj", HIDDEN * HQ * HD, 6.0);
                p("q_norm", HD, 7.0);
                p("k_norm", HD, 8.0);
            }
            p("gate_proj", INTER * HIDDEN, 12.0);
            p("up_proj", INTER * HIDDEN, 13.0);
            p("down_proj", HIDDEN * INTER, 14.0);
        }
        put(format!("{base}.norm.weight"), HIDDEN, 15.0);
        m
    }

    /// 零块(htod 清零;勿用裸 alloc —— 垃圾内存会进 conv/rec/KV 状态)
    async fn zero_block(client: &mut owl_cuda::GpuClient, n: usize, shape: Vec<usize>) -> TensorOps {
        let b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F32, vec![n], &f32b(&vec![0.0; n])).step(),
            client,
        )
        .await
        .expect("zero block");
        TensorOps::of_block(b.id, Dtype::F32, shape)
    }

    fn gdn_buf(gi: usize) -> GdnBuffers {
        // GPU 臂用零块构造;此构造器供 CPU 声明测试(零块声明即可)
        let key_dim = NK * HK_DIM;
        let value_dim = NV * HV_DIM;
        GdnBuffers {
            conv_q: TensorOps::zeros(Dtype::F32, vec![MAX_SLOTS, key_dim, 3]),
            conv_k: TensorOps::zeros(Dtype::F32, vec![MAX_SLOTS, key_dim, 3]),
            conv_v: TensorOps::zeros(Dtype::F32, vec![MAX_SLOTS, value_dim, 3]),
            rec: TensorOps::zeros(Dtype::F32, vec![MAX_SLOTS, NV, HK_DIM, HV_DIM]),
            slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[gi as f32])),
        }
    }

    /// CPU 臂:checkpoint 键形装载 + 声明良构(kvs/gdns 派生 + 双分支)
    #[tokio::test]
    async fn loads_checkpoint_keys_and_declares() {
        let mut face = owl_cpu::CpuFace::new();
        let model = Model::new(&spec(), LlamaFamily::new("model.language_model"));
        crate::interpreters::eval_load(&model, &mut face, &checkpoint_src(), &Default::default())
            .await
            .expect("model.eval_load(C10 声明路径)");

        // 装载完备性:embed 双槽 + final norm(all 层槽位由缺键 Err 兼容性覆盖)
        assert!(model.embed.is_loaded(), "embed 双槽");
        assert!(model.norm.is_loaded(), "final norm");

        // 声明良构:logits [1, vocab] 无毒(CPU face 不执行 Kernel,只验声明)
        let ids = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[3.0]));
        let pos = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0]));
        let kvs: Vec<KvBuffers> = (0..1)
            .map(|_| KvBuffers {
                k_cache: TensorOps::zeros(Dtype::F32, vec![KV_ROWS, HKV, HD]),
                v_cache: TensorOps::zeros(Dtype::F32, vec![KV_ROWS, HKV, HD]),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0])),
            })
            .collect();
        let gdns: Vec<GdnBuffers> = (0..3).map(gdn_buf).collect();
        let rp = Rope::new(64, HD, HD, 10_000.0).expect("rope");
        let ctx = ForwardCtx::model_decode(1, &pos, &kvs, &rp, &gdns);
        let logits = model.forward(&ids, &ctx);
        assert!(!logits.is_poisoned(), "整模声明不应有毒(缺依赖会沿分支毒化)");
        assert_eq!(logits.shape(), &[1, VOCAB]);
        assert_eq!(model.hidden(), HIDDEN);
        assert_eq!(model.vocab_size(), VOCAB);
    }

    // ======================================================================
    // GPU 两步 decode vs host 全链(门控 OWL_TEST_DEVICE)
    // ======================================================================

    /// host 全链参考(独立副本;内维显式传参 —— §四 16 律)。
    /// 两步跨步持状态:GDN conv/rec per-layer、full 层 KV cache 增行。
    struct HostModel {
        src: HashMap<String, Vec<f32>>,
        gdn_conv: Vec<[Vec<f32>; 3]>,       // per gdn 层 q/k/v 段 [slot, dim, 3]
        gdn_rec: Vec<Vec<f32>>,             // per gdn 层 [slot, NV, HK_DIM, HV_DIM]
        kv: Vec<(Vec<f32>, Vec<f32>)>,      // per full 层 (k, v) [row, hkv*hd]
    }

    impl HostModel {
        fn new(src: HashMap<String, Vec<f32>>) -> Self {
            let key_dim = NK * HK_DIM;
            let value_dim = NV * HV_DIM;
            let ngdn = 3;
            let nfull = 1;
            HostModel {
                gdn_conv: (0..ngdn)
                    .map(|_| {
                        [
                            vec![0.0; MAX_SLOTS * key_dim * 3],
                            vec![0.0; MAX_SLOTS * key_dim * 3],
                            vec![0.0; MAX_SLOTS * value_dim * 3],
                        ]
                    })
                    .collect(),
                gdn_rec: (0..ngdn)
                    .map(|_| vec![0.0; MAX_SLOTS * NV * HK_DIM * HV_DIM])
                    .collect(),
                kv: (0..nfull)
                    .map(|_| (vec![0.0; KV_ROWS * HKV * HD], vec![0.0; KV_ROWS * HKV * HD]))
                    .collect(),
                src,
            }
        }

        fn s(&self, i: usize, k: &str) -> &Vec<f32> {
            &self.src[&format!("model.language_model.layers.{i}.{k}.weight")]
        }

        fn lin(&self, x: &[f32], w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
            (0..out_dim)
                .map(|o| (0..in_dim).map(|j| x[j] * w[o * in_dim + j]).sum::<f32>())
                .collect()
        }

        fn ln(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
            let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
            let inv = 1.0 / (ms + EPS).sqrt();
            x.iter().zip(w).map(|(v, g)| v * inv * (g + 1.0)).collect()
        }

        fn rope_one(x: &[f32], heads: usize, p: f32) -> Vec<f32> {
            let half = HD / 2;
            let mut out = x.to_vec();
            for h in 0..heads {
                for i in 0..half {
                    let ang = p * 10_000f32.powf(-(2.0 * i as f32) / HD as f32);
                    let (c, sn) = (ang.cos(), ang.sin());
                    let (a, b) = (x[h * HD + i], x[h * HD + i + half]);
                    out[h * HD + i] = a * c - b * sn;
                    out[h * HD + i + half] = b * c + a * sn;
                }
            }
            out
        }

        fn rms_hd(&self, x: &[f32], w: &[f32]) -> Vec<f32> {
            let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / HD as f32;
            let inv = 1.0 / (ms + EPS).sqrt();
            x.iter().zip(w).map(|(v, g)| v * inv * (g + 1.0)).collect()
        }

        /// full mixer(单 token decode;cache 增行 + 窗 softmax)
        fn attn(&mut self, fi: usize, i: usize, n1: &[f32], pos: f32, slot: usize, kv_len: usize) -> Vec<f32> {
            let q_raw = self.lin(n1, self.s(i, "q_proj"), HQ * HD * 2, HIDDEN);
            let k_raw = self.lin(n1, self.s(i, "k_proj"), HKV * HD, HIDDEN);
            let v_raw = self.lin(n1, self.s(i, "v_proj"), HKV * HD, HIDDEN);
            let mut q = vec![0.0; HQ * HD];
            let mut gate = vec![0.0; HQ * HD];
            for h in 0..HQ {
                for d in 0..HD {
                    q[h * HD + d] = q_raw[h * HD * 2 + d];
                    gate[h * HD + d] = q_raw[h * HD * 2 + HD + d];
                }
            }
            let mut qn = vec![0.0f32; HQ * HD];
            let mut kn = vec![0.0f32; HKV * HD];
            for h in 0..HQ {
                qn[h * HD..(h + 1) * HD]
                    .copy_from_slice(&self.rms_hd(&q[h * HD..(h + 1) * HD], self.s(i, "q_norm")));
            }
            for h in 0..HKV {
                kn[h * HD..(h + 1) * HD]
                    .copy_from_slice(&self.rms_hd(&k_raw[h * HD..(h + 1) * HD], self.s(i, "k_norm")));
            }
            let q = Self::rope_one(&qn, HQ, pos);
            let k = Self::rope_one(&kn, HKV, pos);
            // 写 cache 第 slot 行
            let (kc, vc) = &mut self.kv[fi];
            kc[slot * HKV * HD..(slot + 1) * HKV * HD].copy_from_slice(&k);
            vc[slot * HKV * HD..(slot + 1) * HKV * HD].copy_from_slice(&v_raw);
            // 窗 [slot-kv_len+1, slot] 打分
            let kc = &self.kv[fi].0;
            let vc = &self.kv[fi].1;
            let scale = 1.0 / (HD as f32).sqrt();
            let mut y = vec![0.0f32; HQ * HD];
            for h in 0..HQ {
                let kvh = h / (HQ / HKV);
                let qs = &q[h * HD..(h + 1) * HD];
                let mut scores = Vec::new();
                for r in (slot + 1 - kv_len)..=slot {
                    let ks = &kc[r * HKV * HD + kvh * HD..r * HKV * HD + (kvh + 1) * HD];
                    scores.push(qs.iter().zip(ks).map(|(a, b)| a * b).sum::<f32>() * scale);
                }
                let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let es: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f32 = es.iter().sum();
                for d in 0..HD {
                    let mut acc = 0.0f32;
                    for (j, r) in ((slot + 1 - kv_len)..=slot).enumerate() {
                        acc += es[j] / sum * vc[r * HKV * HD + kvh * HD + d];
                    }
                    y[h * HD + d] = acc;
                }
            }
            for (yi, g) in y.iter_mut().zip(&gate) {
                *yi *= 1.0 / (1.0 + (-g).exp());
            }
            self.lin(&y, self.s(i, "o_proj"), HIDDEN, HQ * HD)
        }

        /// GDN mixer(单 token;per-layer conv/rec 状态跨步)
        fn gdn(&mut self, gi: usize, i: usize, n1: &[f32], slot: usize) -> Vec<f32> {
            let key_dim = NK * HK_DIM;
            let value_dim = NV * HV_DIM;
            let conv_dim = 2 * key_dim + value_dim;
            let qkv = self.lin(n1, self.s(i, "in_proj_qkv"), conv_dim, HIDDEN);
            let z = self.lin(n1, self.s(i, "in_proj_z"), value_dim, HIDDEN);
            let b_v = self.lin(n1, self.s(i, "in_proj_b"), NV, HIDDEN);
            let a_v = self.lin(n1, self.s(i, "in_proj_a"), NV, HIDDEN);
            let segs = [&qkv[0..key_dim], &qkv[key_dim..2 * key_dim], &qkv[2 * key_dim..]];
            let seg_dim = [key_dim, key_dim, value_dim];
            let seg_off = [0usize, key_dim, 2 * key_dim];
            let mut post = [vec![0.0f32; key_dim], vec![0.0f32; key_dim], vec![0.0f32; value_dim]];
            for (seg, inp) in segs.iter().enumerate() {
                let w = self.s(i, "conv1d").clone();
                let st = &mut self.gdn_conv[gi][seg];
                for ch in 0..seg_dim[seg] {
                    let sb = (slot * seg_dim[seg] + ch) * 3;
                    let wb = (seg_off[seg] + ch) * 4;
                    let hist = [st[sb], st[sb + 1], st[sb + 2]];
                    let mut sum = inp[ch] * w[wb + 3];
                    for kk in 0..3 {
                        sum += hist[kk] * w[wb + kk];
                    }
                    sum /= 1.0 + (-sum).exp();
                    st[sb] = hist[1];
                    st[sb + 1] = hist[2];
                    st[sb + 2] = inp[ch];
                    post[seg][ch] = sum;
                }
            }
            let l2 = |row: &[f32]| {
                let ss: f32 = row.iter().map(|v| v * v).sum();
                let inv = 1.0 / (ss.max(0.0) + 1e-6).sqrt();
                row.iter().map(|v| v * inv).collect::<Vec<f32>>()
            };
            let mut qn = vec![0.0f32; key_dim];
            let mut kn = vec![0.0f32; key_dim];
            for h in 0..NK {
                qn[h * HK_DIM..(h + 1) * HK_DIM]
                    .copy_from_slice(&l2(&post[0][h * HK_DIM..(h + 1) * HK_DIM]));
                kn[h * HK_DIM..(h + 1) * HK_DIM]
                    .copy_from_slice(&l2(&post[1][h * HK_DIM..(h + 1) * HK_DIM]));
            }
            let a_log = self.s(i, "A_log");
            let dtb = self.s(i, "dt_bias");
            let g_v: Vec<f32> = (0..NV)
                .map(|h| {
                    let x2 = a_v[h] + dtb[h];
                    let sp = if x2 < 20.0 { x2.exp().ln_1p() } else { x2 };
                    -a_log[h].exp() * sp
                })
                .collect();
            let beta_v: Vec<f32> = b_v.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
            let q_scale = 1.0 / (HK_DIM as f32).sqrt();
            let mut y = vec![0.0f32; value_dim];
            for vh in 0..NV {
                let kh = vh / (NV / NK);
                let decay = g_v[vh].exp();
                let sb = (slot * NV + vh) * HK_DIM * HV_DIM;
                let qoff = kh * HK_DIM;
                let voff = vh * HV_DIM;
                let mut kv_mem = vec![0.0f32; HV_DIM];
                for j in 0..HK_DIM {
                    for d in 0..HV_DIM {
                        let idx = sb + j * HV_DIM + d;
                        self.gdn_rec[gi][idx] *= decay;
                        kv_mem[d] += self.gdn_rec[gi][idx] * kn[qoff + j];
                    }
                }
                for d in 0..HV_DIM {
                    let delta = (post[2][voff + d] - kv_mem[d]) * beta_v[vh];
                    let mut acc = 0.0f32;
                    for j in 0..HK_DIM {
                        let idx = sb + j * HV_DIM + d;
                        self.gdn_rec[gi][idx] += kn[qoff + j] * delta;
                        acc += self.gdn_rec[gi][idx] * qn[qoff + j] * q_scale;
                    }
                    y[voff + d] = acc;
                }
            }
            let gamma = self.s(i, "norm");
            let mut gated = vec![0.0f32; value_dim];
            for vh in 0..NV {
                let base = vh * HV_DIM;
                let ms: f32 = (0..HV_DIM).map(|k| y[base + k] * y[base + k]).sum::<f32>() / HV_DIM as f32;
                let inv = 1.0 / (ms.max(0.0) + EPS).sqrt();
                for k in 0..HV_DIM {
                    let zv = z[base + k];
                    gated[base + k] = y[base + k] * inv * gamma[k] * (zv / (1.0 + (-zv).exp()));
                }
            }
            self.lin(&gated, self.s(i, "out_proj"), HIDDEN, value_dim)
        }

        /// 一步 decode(bs=1):ids/pos/slot/kv_len → logits [VOCAB]
        fn step(&mut self, id: f32, pos: f32, slot: usize, kv_len: usize) -> Vec<f32> {
            let w = self.src["model.language_model.embed_tokens.weight"].clone();
            let mut x = w[(id as usize) * HIDDEN..(id as usize + 1) * HIDDEN].to_vec();
            let (mut kvi, mut gi) = (0usize, 0usize);
            for (i, full) in [false, false, false, true].iter().enumerate() {
                let ln1 = self.ln(&x, self.s(i, "input_layernorm"));
                let mixed = if *full {
                    let m = self.attn(kvi, i, &ln1, pos, slot, kv_len);
                    kvi += 1;
                    m
                } else {
                    let m = self.gdn(gi, i, &ln1, slot);
                    gi += 1;
                    m
                };
                for (xv, mv) in x.iter_mut().zip(&mixed) {
                    *xv += mv;
                }
                let ln2 = self.ln(&x, self.s(i, "post_attention_layernorm"));
                let gate = self.lin(&ln2, self.s(i, "gate_proj"), INTER, HIDDEN);
                let up = self.lin(&ln2, self.s(i, "up_proj"), INTER, HIDDEN);
                let h: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                    .collect();
                let down = self.lin(&h, self.s(i, "down_proj"), HIDDEN, INTER);
                for (xv, dv) in x.iter_mut().zip(&down) {
                    *xv += dv;
                }
            }
            let f = self.ln(&x, &self.src["model.language_model.norm.weight"]);
            (0..VOCAB)
                .map(|v| f.iter().enumerate().map(|(d, fv)| fv * w[v * HIDDEN + d]).sum::<f32>())
                .collect()
        }
    }

    /// GPU 两步 decode vs host 全链(状态跨步:GDN conv/rec + KV 增行;
    /// 门控 OWL_TEST_DEVICE)
    #[tokio::test]
    async fn gpu_model_two_steps_match_host() {
        if !gpu_enabled() {
            skip_note();
            return;
        }
        let model = Model::new(&spec(), LlamaFamily::new("model.language_model"));
        let src = checkpoint_src();
        let mut gpu = gpu_client().await;
        crate::interpreters::eval_load(&model, &mut gpu, &src, &Default::default())
            .await
            .expect("model.eval_load");
        let rp = Rope::new(64, HD, HD, 10_000.0).expect("rope");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &Default::default())
            .await
            .expect("rope 表");

        // 常驻缓冲:1 full 层 KV + 3 gdn 层状态(全零起步)
        let kv_zero = vec![0.0f32; KV_ROWS * HKV * HD];
        let kvs: Vec<KvBuffers> = {
            let mut v = Vec::new();
            for _ in 0..1 {
                v.push(KvBuffers {
                    k_cache: zero_block(&mut gpu, kv_zero.len(), vec![KV_ROWS, HKV, HD]).await,
                    v_cache: zero_block(&mut gpu, kv_zero.len(), vec![KV_ROWS, HKV, HD]).await,
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0])),
                });
            }
            v
        };
        let key_dim = NK * HK_DIM;
        let value_dim = NV * HV_DIM;
        let gdns: Vec<GdnBuffers> = {
            let mut v = Vec::new();
            for _ in 0..3 {
                v.push(GdnBuffers {
                    conv_q: zero_block(&mut gpu, MAX_SLOTS * key_dim * 3, vec![MAX_SLOTS, key_dim, 3]).await,
                    conv_k: zero_block(&mut gpu, MAX_SLOTS * key_dim * 3, vec![MAX_SLOTS, key_dim, 3]).await,
                    conv_v: zero_block(&mut gpu, MAX_SLOTS * value_dim * 3, vec![MAX_SLOTS, value_dim, 3]).await,
                    rec: zero_block(&mut gpu, MAX_SLOTS * NV * HK_DIM * HV_DIM, vec![MAX_SLOTS, NV, HK_DIM, HV_DIM]).await,
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                });
            }
            v
        };

        // host 参考(独立副本)
        let mut host = HostModel::new(src);

        // 两步:step1 (id=5, pos=0, slot=0, kv_len=1) → step2 (id=9, pos=1, slot=1, kv_len=2)
        let steps: [(f32, f32, f32, f32); 2] = [(5.0, 0.0, 0.0, 1.0), (9.0, 1.0, 1.0, 2.0)];
        for (si, &(id, pos, slot, kv_len)) in steps.iter().enumerate() {
            let ids = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[id]));
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[pos]));
            // 每步重写动态标量块(slots/kv_lens 由 runner 步间改写 —— C7)
            let kvs_step: Vec<KvBuffers> = kvs
                .iter()
                .map(|kv| KvBuffers {
                    k_cache: kv.k_cache.clone(),
                    v_cache: kv.v_cache.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot])),
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[kv_len])),
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdns
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot])),
                })
                .collect();
            let ctx = ForwardCtx::model_decode(1, &pos_t, &kvs_step, &rp, &gdns_step);
            let logits = model.forward(&ids, &ctx);
            let got = crate::testkit::harvest(&mut gpu, &logits).await;
            let want = host.step(id, pos, slot as usize, kv_len as usize);
            assert_close(&got, &want, 1e-4, &format!("model-step{si}"));
        }
        gpu.close().await.expect("关机");
    }
}
