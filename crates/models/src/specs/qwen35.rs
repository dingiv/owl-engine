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
    // F16 直转装载(F5;权重 bf16 检查点 → f16 字节,不再 f32 设备中转)
    let ctx = crate::module::LoaderCtx { dtype: qwen3_5_0_8b().dtype, shard: 1 };
    crate::interpreters::eval_load(&model, face, &src, &ctx).await?;
    Ok(model)
}

/// Qwen3.5 tokenizer 装配:机制在 tokenizer.rs,事实在 spec 声明
/// (ModelSpec.tokenizer)—— 本函数只是两端的接线(jinja 全引擎挂账
/// serving 层)
pub fn load_tokenizer(dir: &Path) -> Result<crate::tokenizer::Tokenizer, ModelError> {
    crate::tokenizer::Tokenizer::from_spec(dir, &qwen3_5_0_8b().tokenizer)
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
        dtype: crate::contract::Dtype::F16,
        full_heads: (8, 2, 256),
        gdn_heads: (16, 128, 16, 128),
        eps: 1e-6,
        layer_types: hybrid_3to1(24),
        tokenizer: crate::tokenizer::TokenizerSpec {
            eos_tokens: vec!["<|im_end|>", "<|endoftext|>"],
            chat: crate::tokenizer::ChatFormat {
                prefix: "<|im_start|>user\n".into(),
                // 默认(非思考)模式:模板预填空 think 块(tokenizer_config
                // chat_template add_generation_prompt 分支实证);缺它模型需
                // 自己生成空 think 块,greedy 会紧跟 eos 答空(2026-09-26 实测)。
                suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into(),
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorOps;
    use crate::module::{ForwardCtx, Module};
    use crate::tensor::Dtype;
    use std::collections::HashMap;
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
        let b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F32, vec![n], &f32b(&vec![0.0; n])).step(),
            client,
        )
        .await
        .expect("zero block");
        TensorOps::of_block(b.id, Dtype::F32, shape)
    }

    /// f16 清零块(F5:KV cache;2B/元素)
    async fn zero_block_f16(client: &mut owl_cuda::GpuClient, n: usize, shape: Vec<usize>) -> TensorOps {
        let b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F16, vec![n], &vec![0u8; n * 2]).step(),
            client,
        )
        .await
        .expect("zero block f16");
        TensorOps::of_block(b.id, Dtype::F16, shape)
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


    /// 诊断(tap 单遍):真模型逐层 hidden rms 曲线(定位 step1 塌零层)。
    /// 整模单树单遍归约,StatsTap 只看层根 tag —— 零重放,状态每步只推进
    /// 一次(旧逐层 harvest 重放污染读数,已废;interpreter-tap.md §一)。
    #[tokio::test]
    async fn gpu_model_step_diag() -> Result<(), crate::contract::ModelError> {
        use crate::layers::gdn::GdnBuffers;
        use crate::layers::rope::Rope;
        use crate::module::KvBuffers;
        use crate::testkit::f32_of;
        if !crate::testkit::gpu_enabled() {
            crate::testkit::skip_note();
            return Ok(());
        }
        let dir = manifest_dir().join("assets/Qwen3.5-0.8B");
        let mut gpu = crate::testkit::gpu_client().await;
        let model = load_0_8b(&dir, &mut gpu).await?;
        eprintln!("[diag] loaded");
        let rp = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &Default::default()).await?;

        let (kvs_n, gdns_n) = (6usize, 18usize);
        let mut mk_kvs = Vec::new();
        let mut mk_gdns = Vec::new();
        for _ in 0..kvs_n {
            mk_kvs.push(KvBuffers {
                k_cache: zero_block_f16(&mut gpu, SLOTS * 2 * 256, vec![SLOTS, 2, 256]).await,
                v_cache: zero_block_f16(&mut gpu, SLOTS * 2 * 256, vec![SLOTS, 2, 256]).await,
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0])),
            });
        }
        for _ in 0..gdns_n {
            mk_gdns.push(GdnBuffers {
                conv_q: zero_block(&mut gpu, SLOTS * 2048 * 3, vec![SLOTS, 2048, 3]).await,
                conv_k: zero_block(&mut gpu, SLOTS * 2048 * 3, vec![SLOTS, 2048, 3]).await,
                conv_v: zero_block(&mut gpu, SLOTS * 2048 * 3, vec![SLOTS, 2048, 3]).await,
                rec: zero_block(&mut gpu, SLOTS * 16 * 128 * 128, vec![SLOTS, 16, 128, 128]).await,
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
            });
        }
        let kvs = mk_kvs;
        let gdns = mk_gdns;

        // manifest(装载即校验;320 块 checksum 另有专测 gpu_vram_manifest_checksum)
        let src_chk = SafeTensorsSource::open_dir(&dir)?;
        let _manifest =
            crate::interpreters::eval_load(&model, &mut gpu, &src_chk, &Default::default())
                .await?;
        let steps = [(985.0f32, 0.0f32, 0.0f32, 1.0f32), (4123.0, 1.0, 1.0, 2.0f32)];
        // 引用重读探针(竞速判决):记录层根块引用,步终 sync 后重读 ——
        // 窗口内读零 + 步终读非零 = D2H 流与 COMPUTE 流无同步的竞速读
        struct RefProbe {
            refs: Vec<(String, crate::contract::Bytes, usize)>,
        }
        impl crate::interpreters::Tap for RefProbe {
            fn on_node(
                &mut self,
                ev: &crate::interpreters::NodeEvent<'_>,
            ) -> crate::interpreters::Want {
                if let Some(t) = ev.tag {
                    let elems: usize = ev.shape.iter().product();
                    self.refs.push((
                        t.to_string(),
                        crate::contract::Bytes { id: ev.out.block_id, len: elems },
                        elems,
                    ));
                }
                crate::interpreters::Want::Quiet
            }
        }
        let mut probe = RefProbe { refs: Vec::new() };
        for (si, &(id, pos, slot, kv_len)) in steps.iter().enumerate() {
            eprintln!("[diag] === step{si} (pos {pos} slot {slot} kv_len {kv_len}) ===");
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
            let base_ctx = ForwardCtx::model_decode(1, &pos_t, &kvs_step, &rp, &gdns_step);
            // ── 律 25(候选):带状态观测禁重放 —— 整模单树**单遍**归约,
            // tap 层根曲线代替旧逐层 harvest(旧法每层整链重放,GDN 状态
            // 被观测行为污染,读数不可信 —— interpreter-tap.md §一)。
            let tree = model.forward(&ids, &base_ctx); // embed→…→norm→logits,层根已打标
            let logits = {
                let mut curve = crate::interpreters::StatsTap::curve(si);
                let mut taps = crate::interpreters::observe::TapChain(&mut curve, &mut probe);
                crate::interpreters::eval_ops_tap(tree.step(), &mut gpu, &mut taps).await?
            };
            // 步终判读:logits top-1(塌零 = top 落在 0 向量上)
            let mut buf = vec![0u8; model.vocab_size() * 4];
            gpu.dtoh(&logits, &mut buf).await?;
            let lh = f32_of(&buf);
            let (ti, tv) = lh.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| {
                    if v > a.1 { (i, v) } else { a }
                });
            eprintln!("[diag]   step{si} top@{ti} logit={tv:.4}");
            // ── 竞速判决:三流 sync 后重读 L0 层根与 final_norm(窗口内读的同一块)──
            gpu.sync().await?;
            for (tag, b, elems) in &probe.refs {
                if !matches!(tag.as_str(), "L0.gdn" | "final_norm") {
                    continue;
                }
                let mut buf = vec![0u8; elems * 4];
                gpu.dtoh(b, &mut buf).await?;
                let h = f32_of(&buf);
                let rms = (h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32).sqrt();
                eprintln!("[diag]   [sync后重读] {tag} rms={rms:.6} zeros={}", h.iter().filter(|v| **v == 0.0).count());
            }
            probe.refs.clear();
        }
        gpu.close().await?;
        Ok(())
    }


    /// ★ 核心校验:VRAM 块数据 vs 独立锚 逐位比较(门控 OWL_TEST_DEVICE)。
    /// 独立锚 = 自建文件读取(不经 SafeTensorsSource)+ 朴素 BF16 解码 +
    /// 朴素转置(不经 transpose_into_vec)—— 与被测装载链零共享。
    #[tokio::test]
    async fn gpu_vram_manifest_checksum() -> Result<(), crate::contract::ModelError> {
        use crate::module::Layout;
        if !crate::testkit::gpu_enabled() {
            crate::testkit::skip_note();
            return Ok(());
        }
        let dir = manifest_dir().join("assets/Qwen3.5-0.8B");
        let path = dir.join("model.safetensors-00001-of-00001.safetensors");

        // ── 独立锚:一次性自读文件 → 键 → (dtype, 字节区间视图)
        let raw_buf = std::fs::read(&path).map_err(|e| ModelError::Msg(format!("{e}")))?;
        let raw = std::sync::Arc::new(raw_buf);
        let st = safetensors::SafeTensors::deserialize(&raw)
            .map_err(|e| ModelError::Msg(format!("锚解析: {e}")))?;
        let mut anchor: HashMap<String, (safetensors::Dtype, usize, usize)> = HashMap::new();
        for (name, t) in st.iter() {
            let off = t.data().as_ptr() as usize - raw.as_ptr() as usize;
            anchor.insert(name.to_string(), (t.dtype(), off, t.data().len()));
        }
        let anchor_decode = |dtype: safetensors::Dtype,
                             bytes: &[u8],
                             out: &mut Vec<f32>| {
            match dtype {
                safetensors::Dtype::F32 => {
                    out.clear();
                    out.extend(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
                }
                safetensors::Dtype::BF16 => {
                    out.clear();
                    out.extend(bytes.chunks_exact(2).map(|c| {
                        f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
                    }));
                }
                _ => panic!("锚: 不支持的 dtype"),
            }
        };

        // ── 装载(被测链)
        let mut gpu = crate::testkit::gpu_client().await;
        let model = load_0_8b(&dir, &mut gpu).await?;
        // 复跑 load 生成 manifest(权重已装载,重复装载 = 重上传,结果等价)
        let manifest = {
            let src = SafeTensorsSource::open_dir(&dir)?;
            crate::interpreters::eval_load(&model, &mut gpu, &src, &Default::default()).await?
        };
        eprintln!("[chk] manifest {} 条", manifest.entries().len());

        // ── 逐条 dtoh 读回 + 朴素期望 + 逐位比较
        let mut bad = 0usize;
        for (i, e) in manifest.entries().iter().enumerate() {
            let n: usize = e.shape.iter().product();
            let (dtype, off, nbytes) = anchor[&e.key];
            let raw_slice = &raw[off..off + nbytes];
            let mut want = Vec::new();
            anchor_decode(dtype, raw_slice, &mut want);
            if e.layout == Layout::Transposed {
                // 源 [out, in] → 声明 [in, out]:朴素转置
                let (cols, rows) = (e.shape[0], e.shape[1]);
                let mut t = vec![0f32; n];
                for r in 0..rows {
                    for c in 0..cols {
                        t[c * rows + r] = want[r * cols + c];
                    }
                }
                want = t;
            }
            let mut got_bytes = vec![0u8; n * 4];
            gpu.dtoh(&e.block, &mut got_bytes).await?;
            let mism = got_bytes
                .chunks_exact(4)
                .zip(want.iter())
                .position(|(c, w)| {
                    let g = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                    g.to_bits() != w.to_bits()
                });
            match mism {
                None => {}
                Some(p) => {
                    bad += 1;
                    let g = f32::from_le_bytes([
                        got_bytes[p * 4],
                        got_bytes[p * 4 + 1],
                        got_bytes[p * 4 + 2],
                        got_bytes[p * 4 + 3],
                    ]);
                    eprintln!("[chk] ✗ #{i} {} 首个错位 {p}: gpu {g} vs 锚 {}", e.key, want[p]);
                }
            }
            if i % 100 == 0 {
                eprintln!("[chk] … {i}/{} 已核", manifest.entries().len());
            }
        }
        eprintln!("[chk] 完成: {} 条中 {bad} 条不符", manifest.entries().len());
        assert_eq!(bad, 0, "显存数据与独立锚逐位不一致");
        gpu.close().await?;
        Ok(())
    }


    /// ★ matmul 维度二分探针:同尺寸矩阵直调算子,host 参考逐元素对拍。
    /// 两种数据形态:from_host 叶子 / of_block 上传块(复刻 diag 场景)。
    #[tokio::test]
    async fn gpu_matmul_dim_probe() -> Result<(), crate::contract::ModelError> {
        use crate::contract::DeviceClient;
        if !crate::testkit::gpu_enabled() {
            crate::testkit::skip_note();
            return Ok(());
        }
        let mut gpu = crate::testkit::gpu_client().await;
        let cases: [(&str, usize, usize, usize); 5] = [
            ("fixture      m1-k6-n6", 1, 6, 6),
            ("k1024-n24   ", 1, 1024, 24),
            ("k1024-n384  ", 1, 1024, 384),
            ("k1024-n6144 ", 1, 1024, 6144),
            ("m4-k1024-n6144", 4, 1024, 6144),
        ];
        for (name, m, k, n) in cases {
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.125).collect();

            // 形态一:from_host 叶子
            let a_t = TensorOps::from_host(Dtype::F32, vec![m, k], &f32b(&a));
            let b_t = TensorOps::from_host(Dtype::F32, vec![k, n], &f32b(&b));
            let got = crate::testkit::harvest(&mut gpu, &a_t.matmul(&b_t)).await;

            // host 参考(朴素)
            let mut want = vec![0f32; m * n];
            for i in 0..m {
                for p in 0..k {
                    let av = a[i * k + p];
                    for j in 0..n {
                        want[i * n + j] += av * b[p * n + j];
                    }
                }
            }
            let maxdiff = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            let zeros = got.iter().filter(|v| **v == 0.0).count();
            eprintln!("[mm] {name} from_host : maxdiff={maxdiff:.6} zeros={zeros}/{}", got.len());

            // 形态二:of_block 上传块(htod_f32 后以 Block 叶子引用)
            let sa = vec![m, k];
            let sb = vec![k, n];
            let ba = gpu.htod_f32(&sa, a.clone()).await?;
            let bb = gpu.htod_f32(&sb, b.clone()).await?;
            let a_b = TensorOps::of_block(ba.id, Dtype::F32, vec![m, k]);
            let b_b = TensorOps::of_block(bb.id, Dtype::F32, vec![k, n]);
            let got2 = crate::testkit::harvest(&mut gpu, &a_b.matmul(&b_b)).await;
            let maxdiff2 = got2
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            let zeros2 = got2.iter().filter(|v| **v == 0.0).count();
            eprintln!("[mm] {name} of_block  : maxdiff={maxdiff2:.6} zeros={zeros2}/{}", got2.len());
            assert!(
                maxdiff < 1e-2 && maxdiff2 < 1e-2,
                "{name} matmul 错误: from_host {maxdiff} / of_block {maxdiff2}"
            );
        }
        gpu.close().await?;
        Ok(())
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
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &Default::default())
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
            // f16 logits(F5 整模切换):harvest_f16 解码回 f32(top 判读口径不变)
            let got = crate::testkit::harvest_f16(&mut gpu, &logits).await;
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
