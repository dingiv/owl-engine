//! Qwen3.5 规格集(批8 拆分,2026-09-26):只住 **Qwen3.5 特有**的
//! 参数事实,零机制 —— 通用主干见 [`crate::model`]。
//!
//! 拆分律:specs/<model>.rs = 该模型的维度预设 / 层型表约定 / 检查点
//! 特例。Qwen3.5 特有项:
//! - hybrid 3:1 层型(`linear_attention` ×3 + `full_attention` ×1 周期;
//!   config.json layer_types 实测,24 层 = 18 GDN + 6 full);
//! - 0.8B 维度档(hidden 1024 / GDN 16×128 / full 8q2kv×256 /
//!   vocab 248320 tied / eps 1e-6)。
//!
//! 未来新档(4B/…)加构造函数;新模型(qwen3 dense 等)另起
//! `specs/<model>.rs`,勿混居。
//!
//! **M-e 装载面**:Qwen35Convention(键名约定)+ [`load_0_8b`] 快照
//! 目录装载入口 —— 模型特有胶水(子前缀/裸键/基座)全部封在本文件,
//! 通用层(model.rs/module.rs/loader.rs)零 Qwen3.5 知识。

//! 检查点实测(488 张量):BF16 为主 + `A_log`/`dt_bias` 两个 F32 裸键;
//! conv1d 为 3D `[6144,1,4]`(扁平直读)。visual/mtp 153 键不在最小路径。

use crate::contract::{DeviceClient, ModelError};
use crate::loader::SafeTensorsSource;
use crate::model::{Model, ModelSpec};
use crate::module::KeyConvention;
use std::path::Path;

// ============================================================================
// 检查点键名约定(488 张量实测,2026-09-26)
// ============================================================================

/// Qwen3.5 检查点键名约定 —— 与 Llama 系默认的三处差异:
/// 1. GDN 层 mixer 键带 `linear_attn.` 子前缀,Full 层带 `self_attn.`;
/// 2. mlp 键带 `mlp.` 子前缀;
/// 3. `A_log` / `dt_bias` 是**裸键**(无 `.weight` 后缀;全仓仅有的
///    两个 F32 键)。
///
/// 局部键名在 GDN/Full 两侧互斥,按名分派无歧义(无需知道层型)。
pub struct Qwen35Convention {
    base: String,
}

impl Qwen35Convention {
    /// base = 检查点基座(多模态仓 `model.language_model`,纯文本仓 `model`)
    pub fn new(base: impl Into<String>) -> Self {
        Self { base: base.into() }
    }
}

impl KeyConvention for Qwen35Convention {
    fn embed_key(&self, local: &str) -> String {
        format!("{}.embed_tokens.{local}", self.base)
    }
    fn norm_key(&self, local: &str) -> String {
        format!("{}.{}{}.weight", self.base, "", local)
    }
    fn layer_key(&self, i: usize, local: &str) -> String {
        let (sub, suffix) = match local {
            // 层共用件:双 norm 无子前缀
            "input_layernorm" | "post_attention_layernorm" => ("", ".weight"),
            // mlp
            "gate_proj" | "up_proj" | "down_proj" => ("mlp.", ".weight"),
            // GDN mixer(norm = 层内门控归一化,非 final norm)
            "in_proj_qkv" | "in_proj_z" | "in_proj_b" | "in_proj_a" | "out_proj" | "conv1d"
            | "norm" => ("linear_attn.", ".weight"),
            // GDN 裸键(全仓仅有的无 .weight 后缀特例)
            "A_log" | "dt_bias" => ("linear_attn.", ""),
            // Full mixer
            "q_proj" | "k_proj" | "v_proj" | "o_proj" | "q_norm" | "k_norm" => {
                ("self_attn.", ".weight")
            }
            other => unreachable!("Qwen35Convention: 未知层内局部键 {other}"),
        };
        format!("{}.layers.{i}.{}{local}{}", self.base, sub, suffix)
    }
}

// ============================================================================
// 装载入口(M-e)
// ============================================================================

/// 装载 Qwen3.5-0.8B(快照目录:`config.json` + `*.safetensors`)。
/// 目录内全部 safetensors 合并取源(BF16 → f32,visual/mtp 键无害
/// 常驻源内,只有主干键会被查到);装载走整模 Loadable 单清单。
/// 装载 Qwen3.5-0.8B(快照目录:`config.json` + `*.safetensors`)。
/// 目录内全部 safetensors 合并取源(BF16 → f32,visual/mtp 键无害
/// 常驻源内,只有主干键会被查到);装载走整模 Loadable 单清单。
pub async fn load_0_8b<D: DeviceClient>(
    dir: &Path,
    face: &mut D,
) -> Result<Model, ModelError> {
    let model = Model::new(&qwen3_5_0_8b(), Qwen35Convention::new("model.language_model"));
    let src = SafeTensorsSource::open_dir(dir)?;
    crate::interpreter::eval_load(&model, face, &src, &Default::default()).await?;
    Ok(model)
}

/// 3:1 周期(G,G,G,F)铺满 n 层(Qwen3.5 hybrid 惯例;
/// 24 层 → 18 GDN + 6 full,与 0.8B config.json 实测一致)
pub fn hybrid_3to1(n: usize) -> Vec<bool> {
    (0..n).map(|i| i % 4 == 3).collect()
}

/// Qwen3.5-0.8B 真实维度(M-e 灌真权重用;config.json 实测)
pub fn qwen3_5_0_8b() -> ModelSpec {
    ModelSpec {
        vocab: 248320,
        hidden: 1024,
        inter: 3584,
        full_heads: (8, 2, 256),
        gdn_heads: (16, 128, 16, 128),
        eps: 1e-6,
        layer_types: hybrid_3to1(24),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorOps;
    use crate::module::{ForwardCtx, Module};
    use crate::tensor::Dtype;
    use crate::testkit::{f32b, manifest_dir};

    #[test]
    fn hybrid_3to1_matches_config() {
        let lt = hybrid_3to1(24);
        assert_eq!(lt.len(), 24);
        assert_eq!(lt.iter().filter(|&&f| f).count(), 6, "6 full 层");
        assert_eq!(lt.iter().filter(|&&f| !f).count(), 18, "18 GDN 层");
        assert!(lt[3] && lt[7] && lt[23], "full 层落在 i%4==3");
        let s = qwen3_5_0_8b();
        assert_eq!(s.layer_types, lt);
        assert_eq!(s.vocab, 248320);
    }

    /// 键名约定 vs 检查点实测(488 张量逐键核对过的样例)
    #[test]
    fn convention_matches_checkpoint() {
        let c = Qwen35Convention::new("model.language_model");
        // GDN 层(linear_attn 子前缀)
        assert_eq!(
            c.layer_key(0, "in_proj_qkv"),
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        );
        assert_eq!(
            c.layer_key(0, "conv1d"),
            "model.language_model.layers.0.linear_attn.conv1d.weight"
        );
        assert_eq!(
            c.layer_key(0, "norm"),
            "model.language_model.layers.0.linear_attn.norm.weight"
        );
        // 裸键特例(无 .weight)
        assert_eq!(
            c.layer_key(0, "A_log"),
            "model.language_model.layers.0.linear_attn.A_log"
        );
        assert_eq!(
            c.layer_key(0, "dt_bias"),
            "model.language_model.layers.0.linear_attn.dt_bias"
        );
        // Full 层(self_attn 子前缀)
        assert_eq!(
            c.layer_key(3, "q_proj"),
            "model.language_model.layers.3.self_attn.q_proj.weight"
        );
        // mlp / 层共用件
        assert_eq!(
            c.layer_key(0, "gate_proj"),
            "model.language_model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            c.layer_key(5, "input_layernorm"),
            "model.language_model.layers.5.input_layernorm.weight"
        );
        // embed / final norm
        assert_eq!(c.embed_key("weight"), "model.language_model.embed_tokens.weight");
        assert_eq!(c.norm_key("norm"), "model.language_model.norm.weight");
    }


    // ==================================================================
    // 真权重装载 + 两步 decode 冒烟(门控 OWL_TEST_DEVICE;数值基准
    // 对比 transformers/vLLM 挂账)
    // ==================================================================

    const VOCAB: usize = 248_320;
    const HKV: usize = 2;
    const HD: usize = 256;
    const SLOTS: usize = 4;

    async fn zero_block(client: &mut owl_cuda::GpuClient, n: usize, shape: Vec<usize>) -> TensorOps {
        let b = crate::interpreter::eval_ops(
            TensorOps::from_host(Dtype::F32, vec![n], &f32b(&vec![0.0; n])).step(),
            client,
        )
        .await
        .expect("zero block");
        TensorOps::of_block(b.id, Dtype::F32, shape)
    }

    /// CPU face 真权重装载(非门控;验源/约定/内存,不验 CUDA)
    #[tokio::test]
    async fn cpu_real_weights_load() {
        let dir = manifest_dir().join("assets/Qwen3.5-0.8B");
        let mut face = owl_cpu::CpuFace::new();
        let model = load_0_8b(&dir, &mut face).await.expect("load_0_8b(CPU)");
        assert_eq!(model.layers.len(), 24);
        assert!(model.embed.is_loaded() && model.norm.is_loaded());
    }

    #[tokio::test]
    async fn gpu_real_weights_smoke() {
        use crate::layers::gdn::GdnBuffers;
        use crate::layers::rope::Rope;
        use crate::module::KvBuffers;
        // if !crate::testkit::gpu_enabled() {
        //     crate::testkit::skip_note();
        //     return;
        // }
        let dir = manifest_dir().join("assets/Qwen3.5-0.8B");
        eprintln!("[smoke] boot server...");
        let mut gpu = crate::testkit::gpu_client().await;
        eprintln!("[smoke] booted; open safetensors...");
        let t_load = std::time::Instant::now();
        let model = load_0_8b(&dir, &mut gpu).await.expect("load_0_8b(真权重)");
        let dt = t_load.elapsed().as_secs_f64();
        eprintln!(
            "[smoke] loaded 24 layers ({dt:.2}s,权重 3.9GB f32 → {:.1}GB/s)",
            3.9 / dt
        );
        assert_eq!(model.layers.len(), 24);
        assert_eq!(model.layers.iter().filter(|l| l.is_full()).count(), 6);

        // rope 表(theta 1e7 / partial 64 / max_pos 262144)
        let rp = Rope::new(262_144, HD, 64, 10_000_000.0).expect("rope");
        eprintln!("[smoke] rope 表...");
        crate::interpreter::eval_load(&rp, &mut gpu, &rp.tables(), &Default::default())
            .await
            .expect("rope 表");
        eprintln!("[smoke] rope ok; 分配常驻缓冲...");

        // 常驻缓冲:6 full 层 KV + 18 gdn 层状态(真维度;全零起步)
        let kv_len = SLOTS * HKV * HD;
        let kvs: Vec<KvBuffers> = {
            let mut v = Vec::new();
            for _ in 0..6 {
                v.push(KvBuffers {
                    k_cache: zero_block(&mut gpu, kv_len, vec![SLOTS, HKV, HD]).await,
                    v_cache: zero_block(&mut gpu, kv_len, vec![SLOTS, HKV, HD]).await,
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0])),
                });
            }
            v
        };
        let key_dim = 2048;
        let gdns: Vec<GdnBuffers> = {
            let mut v = Vec::new();
            for _ in 0..18 {
                v.push(GdnBuffers {
                    conv_q: zero_block(&mut gpu, SLOTS * key_dim * 3, vec![SLOTS, key_dim, 3]).await,
                    conv_k: zero_block(&mut gpu, SLOTS * key_dim * 3, vec![SLOTS, key_dim, 3]).await,
                    conv_v: zero_block(&mut gpu, SLOTS * key_dim * 3, vec![SLOTS, key_dim, 3]).await,
                    rec: zero_block(&mut gpu, SLOTS * 16 * 128 * 128, vec![SLOTS, 16, 128, 128]).await,
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                });
            }
            v
        };

        eprintln!("[smoke] buffers ok; 两步 decode...");
        // 两步 decode(tokenizer 未接,任意合法 id;验证数值健康与状态推进)
        let steps = [(985.0f32, 0.0f32, 0.0f32, 1.0f32), (4123.0, 1.0, 1.0, 2.0)];
        let mut prev_top: Option<usize> = None;
        for (si, &(id, pos, slot, kv_len)) in steps.iter().enumerate() {
            let ids = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[id]));
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[pos]));
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
            assert_eq!(logits.shape(), &[1, VOCAB]);
            let got = crate::testkit::harvest(&mut gpu, &logits).await;
            assert!(got.iter().all(|v| v.is_finite()), "step{si} logits 应全有限");
            let top = (0..VOCAB)
                .max_by(|&a, &b| got[a].abs().total_cmp(&got[b].abs()))
                .unwrap();
            println!("  [smoke] step{si}: id {id} → top@{top} (logit {:.4})", got[top]);
            if let Some(p) = prev_top {
                assert_ne!(p, top, "两步状态应推进(top1 不应不变)");
            }
            prev_top = Some(top);
        }
        gpu.close().await.expect("关机");
    }
}
