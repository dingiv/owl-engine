//! Qwen3.5 dense 模型(移植自 xinfer `models/qwen3_5.rs`,984 行 → 本文件)。
//!
//! port 出处标注:结构/控制流 1:1;偏差全部登记在文末 [适配台账]。
//! 风格律:S6(functional-via-ctx)。层代码当前经 layers 桩
//! (OwlTensor 体 = T3 回填);模型自身不直接分配
//! (slot/verify 缓冲的 ctor 调用点标注回填面)。
//!
//! 本模型特性:hybrid 层(full_attention / linear_attention=GDN 交错),
//! mamba slot 状态管理,DFlash verify 逐层 hidden 捕获,MTP hidden 缓冲。

use super::layers::attention::Attention;
use super::layers::deltanet::GatedDeltaNet;
use super::layers::distributed::{Comm, TensorParallelRowLinear};
use super::layers::mask::get_attention_causal_mask;
use super::layers::mlp::MLP;
use super::layers::others::{embedding, rms_norm, NormX};
use super::layers::rotary_emb::{ApplyRotaryEmbedding, ScalingRotaryEmbedding};
use super::layers::vendor;
use super::layers::VarBuilderX;
use super::layers::{DType, Device, Embedding, OwlTensor, Tensor};
use crate::bail;
use crate::error::Result;
use crate::config::Config;
use crate::hybrid::resolve_qwen3_hybrid_config;
use parking_lot::{RwLock, RwLockWriteGuard};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

// =============================================================================
// 模型层输入元数据(适配:扩展 vendor::InputMetadata 的模型侧视图;
// 合并进 vendor::InputMetadata = 适配台账 A1)
// =============================================================================

/// qwen3_5 专用输入元数据:xinfer `attention_rs::InputMetadata` 的
/// 模型可见字段面。层调用时经 [`Self::vendor`] 投影为 vendor 类型。
#[derive(Clone, Default)]
pub struct InputMetadata {
    pub seqlens: Option<Vec<usize>>,
    pub is_prefill: bool,
    pub is_mtp_verify: bool,
    /// mamba slot 显式映射(I64;来源 runner)。None = 从 sequence_ids 解析。
    pub mamba_slot_mapping: Option<Tensor>,
    pub sequence_ids: Option<Vec<usize>>,
    /// decode naive 路径设备指针对(来源 runner/graphplan view;None = eager/prefill)
    pub decode_ptrs: Option<crate::models::layers::vendor::DecodePtrs>,
}

impl InputMetadata {
    /// 投影为层接受的 vendor 面(丢弃模型私有字段;decode 指针透传)。
    /// prefill 时从 seqlens 派生 cu_seqlens(U32 设备张量;GDN conv1d 与
    /// varlen 分派消费,M-Ⅰ 实测缺口,2026-09-23)。
    fn vendor(&self, device: &Device) -> vendor::InputMetadata {
        let seqlens = self.seqlens.clone().unwrap_or_default();
        let cu_seqlens_q = if self.is_prefill && !seqlens.is_empty() {
            let mut v = Vec::with_capacity(seqlens.len() + 1);
            let mut acc = 0u32;
            v.push(0u32);
            for s in &seqlens {
                acc += *s as u32;
                v.push(acc);
            }
            Some(super::layers::ctor::from_vec(v, (seqlens.len() + 1,), device).expect("cu_seqlens 落池"))
        } else {
            None
        };
        vendor::InputMetadata {
            seqlens,
            context_lens: Vec::new(),
            is_prefill: self.is_prefill,
            is_mtp_verify: self.is_mtp_verify,
            cu_seqlens_q,
            decode_ptrs: self.decode_ptrs,
        }
    }
}

// =============================================================================
// Hybrid decoder layer: either full attention or GatedDeltaNet
// =============================================================================

pub enum Qwen3_5AttnType {
    FullAttention(Attention),
    LinearAttention(GatedDeltaNet),
}

pub struct Qwen3_5DecoderLayer {
    attn: Qwen3_5AttnType,
    mlp: MLP,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
    rotary_emb: Option<Arc<ScalingRotaryEmbedding>>,
}

impl Qwen3_5DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        rotary_emb: Arc<ScalingRotaryEmbedding>,
        config: &Config,
        layer_type: &str,
        gdn_layer_idx: usize,
        dtype: DType,
    ) -> Result<Self> {
        let is_qvar_builder = vb.is_qvar_builder();
        // Qwen3.5 RMSNorm = ×(1+w)(HF modeling_qwen3_5.rs:854;xinfer 原语义忠实)。
        // 2026-09-23 曾误判 +1 为误植而改 false,被 m4 逐层对拍推翻:缺失 +1
        // 导致 GDN 层贡献缩水(层 4 起 7% 发散)。kernel 已支持 w_off 参数。
        let use_norm_offset = !is_qvar_builder
            && !config
                .quantization_config
                .as_ref()
                .is_some_and(|q| q.is_mlx_nvfp4);

        let attn = if layer_type == "full_attention" {
            Qwen3_5AttnType::FullAttention(Attention::new(
                if is_qvar_builder {
                    vb.clone()
                } else {
                    vb.pp("self_attn")
                },
                comm.clone(),
                config,
                None,
                config.sliding_window,
                dtype,
            )?)
        } else {
            Qwen3_5AttnType::LinearAttention(GatedDeltaNet::new(
                if is_qvar_builder {
                    vb.clone()
                } else {
                    vb.pp("linear_attn")
                },
                comm.clone(),
                config,
                gdn_layer_idx,
                dtype,
            )?)
        };

        let mlp = if is_qvar_builder {
            MLP::new(vb, comm.clone(), config.hidden_size, config.intermediate_size,
                &config.hidden_act, &config.quantization_config, &config.quant,
                false, dtype, "")?
        } else {
            let mlp_vb = vb.pp("mlp");
            MLP::new(&mlp_vb, comm.clone(), config.hidden_size, config.intermediate_size,
                &config.hidden_act, &config.quantization_config, &config.quant,
                false, dtype, "")?
        };

        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("attn_norm")
            } else {
                vb.pp("input_layernorm")
            },
            DType::F32,
            use_norm_offset,
        )?;

        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp("post_attention_norm")
            } else {
                vb.pp("post_attention_layernorm")
            },
            DType::F32,
            use_norm_offset,
        )?;

        let rotary = if layer_type == "full_attention" {
            Some(rotary_emb)
        } else {
            None
        };

        Ok(Self {
            attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            rotary_emb: rotary,
        })
    }

    /// S6 适配注记:层签名现无 ctx 参数(适配台账 A2);本 forward 保持
    /// xinfer 形态,`+` 已改 [`OwlTensor::add`](Result 化,无 panic 面)。
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Vec<Tensor>>,
        positions: &Tensor,
        cache: Option<(&Tensor, &Tensor)>,
        input_metadata: &vendor::InputMetadata,
        mamba_cache: &mut vendor::MambaCache,
        seq_slots: &Tensor,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;

        let attn_output = match &self.attn {
            Qwen3_5AttnType::FullAttention(attn) => {
                let rope: Arc<dyn ApplyRotaryEmbedding> = self.rotary_emb.as_ref().unwrap().clone();
                attn.forward(
                    &xs,
                    &Some(rope),
                    attention_mask,
                    positions,
                    cache,
                    input_metadata,
                )?
            }
            Qwen3_5AttnType::LinearAttention(gdn) => {
                gdn.forward(&xs, mamba_cache, input_metadata, seq_slots)?
            }
        };

        // layer_metrics barrier 面(xinfer lm::barrier)未移植:观测面,T3 随
        // metrics 里程碑接线(适配台账 A5)
        let xs = attn_output.add(residual)?;
        let residual = &xs;
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let mlp_output = self.mlp.forward(&xs)?;
        residual.add(&mlp_output)
    }

    pub fn is_full_attention(&self) -> bool {
        matches!(&self.attn, Qwen3_5AttnType::FullAttention(_))
    }
}

// =============================================================================
// Qwen3.5 causal LM (dense variant)
// =============================================================================

pub struct Qwen3_5ForCausalLM {
    embed_tokens: Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: NormX,
    /// 适配台账 A3:xinfer VocabParallelLinear → owl TensorParallelRowLinear
    /// (同为行并行 + all_reduce;vocab 分片语义等价)
    lm_head: TensorParallelRowLinear,
    mamba_cache: RwLock<vendor::MambaCache>,
    device: Device,
    config: Config,
    dtype: DType,
    vocab_size: usize,
    is_qvar_builder: bool,
    /// MTP 投机解码的预分配 hidden 缓冲(图池外;copy_ 入图安全)。
    /// 形状 (max_graph_bs, hidden_size),首次 decode forward 前分配。
    pub mtp_hidden_buffer: std::sync::Mutex<Option<Tensor>>,
    /// DFlash verify 的图安全逐层 hidden 缓冲,平行于 dflash_target_layer_ids,
    /// 形状 (max_verify_len, hidden_size)。
    pub dflash_verify_hidden_buffers: std::sync::Mutex<Option<Vec<Tensor>>>,
    pub dflash_target_layer_ids: std::sync::Mutex<Vec<usize>>,
}

/// 元素数(OwlTensor dims 的便捷积;T3 由 DynTensor 原生 elems 替代)
fn elems(t: &Tensor) -> Result<usize> {
    Ok(t.dims()?.iter().product())
}

impl Qwen3_5ForCausalLM {
    fn resolve_seq_slots(
        &self,
        input_metadata: &InputMetadata,
        token_count: usize,
    ) -> Result<Tensor> {
        if let Some(slot_mapping) = &input_metadata.mamba_slot_mapping {
            if slot_mapping.dtype() != DType::I64 {
                bail!(
                    "Qwen3.5 expects mamba_slot_mapping dtype I64, got {}",
                    slot_mapping.dtype()
                )
            }
            let slot_count = slot_mapping.dim(0)?;
            if slot_count == 0 {
                bail!("Qwen3.5 received empty mamba_slot_mapping")
            }
            if !input_metadata.is_prefill && slot_count != token_count {
                bail!(
                    "Qwen3.5 decode mamba_slot_mapping length mismatch: slots={slot_count} tokens={token_count}"
                )
            }
            return Ok(slot_mapping.clone());
        }

        let sequence_ids = input_metadata
            .sequence_ids
            .as_ref()
            .ok_or_else(|| crate::Error::Msg("Qwen3.5 requires sequence_ids".into()))?;
        if sequence_ids.is_empty() {
            bail!("Qwen3.5 received empty sequence_ids");
        }

        let slots = if input_metadata.is_prefill {
            self.ensure_mamba_slots_for_sequences(sequence_ids)?
        } else {
            self.get_mamba_slots_for_sequences(sequence_ids)?
        };
        if slots.is_empty() {
            bail!("Qwen3.5 resolved empty mamba slots from sequence_ids");
        }
        if !input_metadata.is_prefill && slots.len() != token_count {
            bail!(
                "Qwen3.5 decode mamba slot count mismatch: slots={} tokens={token_count}",
                slots.len()
            );
        }

        // S4:slot 语义 U32(I64 仅 GGUF 边界);full-attention 配置下仅形式参数
        let slots_u32 = slots.into_iter().map(|s| s as u32).collect::<Vec<_>>();
        let len = slots_u32.len();
        super::layers::ctor::from_vec(slots_u32, (len,), &self.device)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_prefix(vb, comm, config, dtype, is_rope_i, device, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_prefix(
        vb: &VarBuilderX,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_rope_i: bool,
        device: &Device,
        prefix: Option<String>,
    ) -> Result<Self> {
        let has_prefix = prefix.is_some();
        let mut prefix = prefix.unwrap_or("model.".to_string());
        let gguf_prefix = if has_prefix {
            prefix.clone()
        } else {
            "".to_string()
        };
        let key_map: HashMap<&str, &str> = [
            ("embed_tokens", "token_embd"),
            ("lm_head", "output"),
            ("norm", "output_norm"),
            ("layers", "blk"),
        ]
        .iter()
        .cloned()
        .collect();
        let is_qvar_builder = vb.is_qvar_builder();

        let tie_word_embeddings = if !is_qvar_builder
            && vb.has_key("embed_tokens.weight")
            && !vb.has_key(&format!("{}embed_tokens.weight", prefix))
        {
            // xinfer log_error! 面:引擎日志宏未移植,先 eprintln(适配台账 A6)
            eprintln!("This model does not support decoding!");
            prefix.clear();
            Some(true)
        } else {
            config.tie_word_embeddings
        };

        let (embed_tokens, vocab_size) = embedding(
            config.vocab_size,
            config.hidden_size,
            if is_qvar_builder {
                vb.pp(&format!("{gguf_prefix}{}", key_map["embed_tokens"]))
            } else {
                vb.pp(&format!("{}embed_tokens", prefix))
            },
            dtype,
        )?;

        let rotary_emb = Arc::new(ScalingRotaryEmbedding::new(
            if is_qvar_builder || config.higher_precision_required() {
                DType::F32
            } else {
                dtype
            },
            config,
            &vb.device(),
            is_rope_i,
            config.rope_theta,
        )?);

        let hybrid = resolve_qwen3_hybrid_config(config);
        let layer_types = &hybrid.layer_types;

        // Build layers, tracking GDN layer index separately
        let mut layers = Vec::new();
        let mut gdn_layer_idx = 0usize;

        for i in 0..config.num_hidden_layers {
            let layer_type = &layer_types[i];
            let current_gdn_idx = if layer_type == "linear_attention" {
                let idx = gdn_layer_idx;
                gdn_layer_idx += 1;
                idx
            } else {
                0 // unused for full attention
            };

            let layer = Qwen3_5DecoderLayer::new(
                &vb.pp(format!(
                    "{}.{}",
                    if is_qvar_builder {
                        format!("{gguf_prefix}{}", key_map["layers"])
                    } else {
                        format!("{}layers", prefix)
                    },
                    i
                )
                .as_str()),
                comm.clone(),
                rotary_emb.clone(),
                config,
                layer_type,
                current_gdn_idx,
                dtype,
            )?;
            layers.push(layer);
            // 进度上报面(xinfer ProgressLike)未移植:适配台账 A4
        }

        let _num_gdn_layers = gdn_layer_idx; // T3:MambaCache 形状元数据(A9 回填)

        let norm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            if is_qvar_builder {
                vb.pp(&format!("{gguf_prefix}{}", key_map["norm"]))
            } else {
                vb.pp(&format!("{}norm", prefix))
            },
            DType::F32,
            !is_qvar_builder
                && !config
                    .quantization_config
                    .as_ref()
                    .is_some_and(|q| q.is_mlx_nvfp4),
        )?;

        let is_mlx_nvfp4_tied = tie_word_embeddings.is_some_and(|x| x)
            && config
                .quantization_config
                .as_ref()
                .is_some_and(|q| q.is_mlx_nvfp4);
        let lm_head = if is_mlx_nvfp4_tied {
            // tied embeddings:复用 embedding 权重作 lm_head(零 bias)
            TensorParallelRowLinear::new_loaded(
                config.hidden_size,
                vocab_size,
                &vb.pp(""),
                super::layers::Shard::default(),
                &config.quantization_config,
                &None,
                dtype,
                false,
                comm.clone(),
            )?
        } else {
            let lm_head_vb = if tie_word_embeddings.is_some_and(|x| x) {
                if is_qvar_builder {
                    vb.pp(&format!("{gguf_prefix}{}", key_map["embed_tokens"]))
                } else {
                    vb.pp(&format!("{}embed_tokens", prefix))
                }
            } else if is_qvar_builder {
                vb.pp(key_map["lm_head"])
            } else {
                vb.pp("lm_head")
            };
            TensorParallelRowLinear::load_with_hints(
                config.hidden_size,
                vocab_size,
                lm_head_vb,
                comm.clone(),
                &config.quantization_config,
                &None,
                dtype,
            )?
        };

        // GDN 层 MambaCache:容量由 runner 预分配面接管(preallocate_mamba_cache)
        let world_size = comm.world_size;
        let num_v_heads = hybrid.num_v_heads;
        let num_k_heads = hybrid.num_k_heads;
        if num_v_heads % world_size != 0 || num_k_heads % world_size != 0 {
            bail!(
                "linear attention heads must be divisible by tensor parallel world_size (num_v_heads={num_v_heads}, num_k_heads={num_k_heads}, world_size={world_size})"
            );
        }

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            mamba_cache: RwLock::new(vendor::MambaCache::empty()),
            device: device.clone(),
            config: config.clone(),
            dtype,
            vocab_size,
            is_qvar_builder,
            mtp_hidden_buffer: std::sync::Mutex::new(None),
            dflash_verify_hidden_buffers: std::sync::Mutex::new(None),
            dflash_target_layer_ids: std::sync::Mutex::new(Vec::new()),
        })
    }

    pub fn embed_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.embed_tokens.forward(xs)?;
        if (self.is_qvar_builder || self.config.quant.is_some()) && xs.dtype() != DType::F32 {
            xs.to_dtype(DType::F32)
        } else {
            Ok(xs)
        }
    }

    pub fn embed_weight(&self) -> &Tensor {
        &self.embed_tokens.weight
    }

    /// MTP 取预缓冲第 0 行(MTP decode 恒 bs=1)。
    /// 适配台账 A7:OwlTensor 暂无 get(i);narrow+contiguous 表达。
    pub fn take_last_hidden_for_mtp(&self) -> Option<Tensor> {
        let guard = self.mtp_hidden_buffer.lock().ok()?;
        let buf = guard.as_ref()?;
        buf.narrow(0, 0, 1).ok()
    }

    /// DFlash verify 层 hidden 缓冲预分配(图池外;verify 图捕获前必须调用)。
    pub fn preallocate_dflash_verify_buffers(
        &self,
        target_layer_ids: &[usize],
        max_verify_len: usize,
    ) -> Result<()> {
        // 残差流 dtype:GGUF/ISQ/高精度 → F32;FP8 safetensors 保持模型 dtype
        let buf_dtype = if self.is_qvar_builder
            || self.config.quant.is_some()
            || self.config.higher_precision_required()
        {
            DType::F32
        } else {
            self.dtype
        };
        let mut buffers = Vec::with_capacity(target_layer_ids.len());
        for _ in target_layer_ids {
            buffers.push(super::layers::ctor::zeros(
                (max_verify_len, self.config.hidden_size),
                buf_dtype,
                &self.device,
            )?);
        }
        if let Ok(mut guard) = self.dflash_verify_hidden_buffers.lock() {
            *guard = Some(buffers);
        }
        if let Ok(mut guard) = self.dflash_target_layer_ids.lock() {
            *guard = target_layer_ids.to_vec();
        }
        Ok(())
    }

    /// 读取上一轮 `is_mtp_verify` forward/graph replay 写入的逐层 hidden。
    pub fn take_dflash_verify_hiddens(&self, num_tokens: usize) -> Option<Vec<Tensor>> {
        let ids = self.dflash_target_layer_ids.lock().ok()?;
        let guard = self.dflash_verify_hidden_buffers.lock().ok()?;
        let buffers = guard.as_ref()?;
        if buffers.len() != ids.len() || num_tokens == 0 {
            return None;
        }
        let mut out = Vec::with_capacity(buffers.len());
        for buf in buffers {
            if num_tokens > buf.dim(0).ok()? {
                return None;
            }
            out.push(buf.narrow(0, 0, num_tokens).ok()?.contiguous().ok()?);
        }
        Some(out)
    }

    /// DFlash verify:选中层 hidden 拷入预缓冲(图内 memcpy 节点;
    /// replay 时同址更新)。
    /// 适配台账 A8:OwlTensor 暂无 copy_;kernel 回填点收敛于此。
    fn copy_into_verify_buffer(
        &self,
        buf: &Tensor,
        xs: &Tensor,
        i: usize,
    ) -> Result<()> {
        let ids = self.dflash_target_layer_ids.lock().map_err(|_| {
            crate::Error::Msg("dflash_target_layer_ids 中毒".into())
        })?;
        let bufs = self.dflash_verify_hidden_buffers.lock().map_err(|_| {
            crate::Error::Msg("dflash_verify_hidden_buffers 中毒".into())
        })?;
        if let Some(buffers) = bufs.as_ref() {
            for (buf_idx, &layer_id) in ids.iter().enumerate() {
                if layer_id == i {
                    if let Some(b) = buffers.get(buf_idx) {
                        let n = xs.dim(0)?;
                        if n <= b.dim(0).unwrap_or(0) {
                            // T3 kernel 回填:copy_d2d(dst_view, src;dtype 提升禁,S1)
                            let _ = buf;
                            unimplemented!("T3 kernel 回填: verify hidden copy_d2d(含 dtype 分派)")
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_inner(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        visual_pos_masks: &Option<Tensor>,
        deepstack_visual_embeds: &Option<Vec<Tensor>>,
        return_hidden: bool,
        collect_layer_ids: Option<&[usize]>,
        collected_layers: &mut Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let seqlens = input_metadata.seqlens.clone().unwrap_or_default();

        let attention_mask = get_attention_causal_mask(
            &self.device,
            self.dtype,
            positions,
            seqlens.iter().map(|&x| x as u32).collect(),
            self.config.sliding_window,
            input_metadata.is_prefill,
        );

        let mut xs = if embeded_inputs {
            input_ids.clone()
        } else {
            self.embed_forward(input_ids)?
        };

        if collect_layer_ids.is_some() {
            *collected_layers = Some(vec![xs.clone()]);
        }

        let mut kv_cache_idx = 0usize;
        let seq_slots = self.resolve_seq_slots(input_metadata, xs.dim(0)?)?;
        // vendor 元数据(含 cu_seqlens)整个 forward 存活一份:逐层重建会经
        // 池块释放复用,与异步 kernel 写竞速被清零(2026-09-23 M-Ⅰ 实测)
        let vmeta = input_metadata.vendor(&self.device);
        let input_metadata = &vmeta;
        let mut mamba_cache = self.mamba_cache.write();

        for (i, layer) in self.layers.iter().enumerate() {
            let cache = if layer.is_full_attention() {
                kv_caches.map(|caches| {
                    let c = &caches[kv_cache_idx];
                    kv_cache_idx += 1;
                    (&c.0, &c.1)
                })
            } else {
                None
            };

            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                positions,
                cache,
                &vmeta,
                &mut mamba_cache,
                &seq_slots,
            )?;

            if let (Some(layer_ids), Some(layers)) = (collect_layer_ids, collected_layers.as_mut())
            {
                if layer_ids.contains(&i) {
                    layers.push(xs.clone());
                }
            }

            // Graph-safe DFlash verify:copy_ 被烙进图,replay 同址更新
            if input_metadata.is_mtp_verify {
                self.copy_into_verify_buffer(&xs, &xs, i)?;
            }

            if let (Some(pos_mask), Some(deepstacks)) = (visual_pos_masks, deepstack_visual_embeds)
            {
                use super::layers::deepstack::ApplyDeepStack;
                if i < deepstacks.len() {
                    xs = xs.apply_deep_stack(pos_mask, &deepstacks[i])?;
                }
            }
        }

        if collect_layer_ids.is_some() {
            let logits_xs = if !seqlens.is_empty() {
                let indices: Vec<u32> = seqlens.iter().map(|x| *x as u32 - 1).collect();
                let batch = indices.len();
                xs.index_select(
                    0,
                    &super::layers::ctor::from_vec(indices, (batch,), &self.device)?,
                )?
            } else {
                xs
            };
            let logits_xs = self.norm.forward(&logits_xs)?;
            return if self.is_qvar_builder {
                self.lm_head.forward(&logits_xs)
            } else {
                self.lm_head
                    .forward(&logits_xs.to_dtype(self.dtype)?)?
                    .to_dtype(DType::F32)
            };
        }

        if !seqlens.is_empty() && !return_hidden {
            let indices: Vec<u32> = seqlens.iter().map(|x| *x as u32 - 1).collect();
            let batch = indices.len();
            xs = xs.index_select(
                0,
                &super::layers::ctor::from_vec(indices, (batch,), &self.device)?,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        if return_hidden {
            xs.to_dtype(DType::F32)
        } else {
            // MTP:hidden 拷入预缓冲(copy_ 入图,replay 同址更新;
            // 适配台账 A8 同源)
            if let Ok(guard) = self.mtp_hidden_buffer.lock() {
                if let Some(buf) = guard.as_ref() {
                    if elems(&xs)? <= elems(buf)? {
                        let _ = self.copy_into_verify_buffer(buf, &xs, usize::MAX);
                    }
                }
            }
            let out = if self.is_qvar_builder {
                self.lm_head.forward(&xs)
            } else {
                self.lm_head
                    .forward(&xs.to_dtype(self.dtype)?)?
                    .to_dtype(DType::F32)
            };
            out
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        let mut none_layers = None;
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            &None,
            &None,
            false,
            None,
            &mut none_layers,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<Tensor> {
        let mut none_layers = None;
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            &None,
            &None,
            true,
            None,
            &mut none_layers,
        )
    }

    /// logits + hidden 双返回(MTP 投机解码 draft head 的骨干 hidden 源)。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_hidden(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
    ) -> Result<(Tensor, Tensor)> {
        let mut none_layers = None;
        let hidden = self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            &None,
            &None,
            true,
            None,
            &mut none_layers,
        )?;
        let logits = self.forward_lm_head(&hidden)?;
        Ok((logits, hidden))
    }

    /// logits + 中间层 hiddens(DFlash verify 目标层采集)。
    #[allow(clippy::too_many_arguments)]
    pub fn forward_collecting_layers(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        target_layer_ids: &[usize],
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut collected_layers = None;
        let logits = self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            &None,
            &None,
            false,
            Some(target_layer_ids),
            &mut collected_layers,
        )?;
        Ok((logits, collected_layers.unwrap_or_default()))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_deepstack(
        &self,
        input_ids: &Tensor,
        positions: &Tensor,
        kv_caches: Option<&Vec<(Tensor, Tensor)>>,
        input_metadata: &InputMetadata,
        embeded_inputs: bool,
        visual_pos_masks: &Option<Tensor>,
        deepstack_visual_embeds: &Option<Vec<Tensor>>,
    ) -> Result<Tensor> {
        let mut none_layers = None;
        self.forward_inner(
            input_ids,
            positions,
            kv_caches,
            input_metadata,
            embeded_inputs,
            visual_pos_masks,
            deepstack_visual_embeds,
            false,
            None,
            &mut none_layers,
        )
    }

    /// lm_head 单独施加(MTP drafting 用)。
    pub fn forward_lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        if self.is_qvar_builder {
            self.lm_head.forward(hidden)
        } else {
            self.lm_head
                .forward(&hidden.to_dtype(self.dtype)?)?
                .to_dtype(DType::F32)
        }
    }

    /// 观测面:full_attention 层数(hybrid 分派真跑验证用)
    pub fn full_attention_count(&self) -> usize {
        self.layers.iter().filter(|l| l.is_full_attention()).count()
    }

    /// 观测面:GDN(linear_attention)层数
    pub fn gdn_layer_count(&self) -> usize {
        self.layers.len() - self.full_attention_count()
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab_size
    }

    // ---- mamba slot 状态面(适配台账 A9:vendor::MambaCache 方法通道
    //      为 T3 回填;本组全部为类型面 stub,签名与 xinfer 同形)----

    pub fn release_sequence_state(&self, sequence_id: usize) {
        self.mamba_cache.write().free_slot(sequence_id);
    }

    pub fn ensure_mamba_slots_for_sequences(
        &self,
        sequence_ids: &[usize],
    ) -> Result<Vec<usize>> {
        let mut cache = self.mamba_cache.write();
        sequence_ids.iter().map(|&id| cache.ensure_slot(id)).collect()
    }

    pub fn get_mamba_slots_for_sequences(
        &self,
        sequence_ids: &[usize],
    ) -> Result<Vec<usize>> {
        // 惰性分配:decode 步序列可能未经本进程 prefill(重启恢复/直入 decode)
        let mut cache = self.mamba_cache.write();
        sequence_ids.iter().map(|&id| cache.ensure_slot(id)).collect()
    }

    pub fn lock_mamba_cache_for_graph(&self) -> RwLockWriteGuard<'_, vendor::MambaCache> {
        self.mamba_cache.write()
    }

    pub fn preallocate_mamba_cache(&self, max_num_seqs: usize) -> Result<()> {
        use crate::hybrid::resolve_qwen3_hybrid_config;
        let hybrid = resolve_qwen3_hybrid_config(&self.config);
        let num_gdn_layers = hybrid
            .layer_types
            .iter()
            .filter(|t| t.as_str() == "linear_attention")
            .count();
        if num_gdn_layers == 0 {
            return Ok(()); // 纯注意力模型:无状态表
        }
        // per-rank conv 通道宽:nk*K*2(q|k) + nv*V;world>1 时 hybrid 头数已含 TP 校验
        let world = 1usize; // 单卡面(A2.6 多实例另案;构造期已拦截不可整除)
        let nk = hybrid.num_k_heads / world;
        let nv = hybrid.num_v_heads / world;
        let kd = hybrid.key_head_dim;
        let vd = hybrid.value_head_dim;
        let d_conv = nk * kd * 2 + nv * vd;
        let conv_len = hybrid.conv_kernel_size - 1;
        self.mamba_cache.write().preallocate(
            num_gdn_layers,
            max_num_seqs,
            d_conv,
            conv_len,
            nv,
            kd,
            vd,
            &self.device,
        )
    }

    /// MTP hidden 缓冲预分配(warmup_capture 之前调用:缓冲须在常规显存,
    /// 不得在图池内出生)。
    pub fn preallocate_mtp_hidden_buffer(&self, max_batch_size: usize) -> Result<()> {
        let buf = super::layers::ctor::zeros(
            (max_batch_size, self.config.hidden_size),
            self.dtype,
            &self.device,
        )?;
        if let Ok(mut guard) = self.mtp_hidden_buffer.lock() {
            *guard = Some(buf);
        }
        Ok(())
    }

    pub fn set_mamba_prefix_cache_capacity(&self, _capacity: usize) {
        unimplemented!("T3: MambaCache::set_prefix_cache_capacity 通道回填")
    }

    pub fn capture_mamba_prefix_state(
        &self,
        _seq_id: usize,
        _hash: u64,
        _preserve: bool,
    ) -> Result<bool> {
        unimplemented!("T3: MambaCache::capture_prefix_state 通道回填(裁决 4:池缓冲重表达)")
    }

    pub fn has_mamba_prefix_state(&self, _hash: u64) -> bool {
        unimplemented!("T3: MambaCache::has_prefix_state 通道回填")
    }

    pub fn remove_mamba_prefix_state(&self, _hash: u64) -> bool {
        unimplemented!("T3: MambaCache::remove_prefix_state 通道回填")
    }

    pub fn restore_mamba_prefix_state(&self, _seq_id: usize, _hash: u64) -> Result<bool> {
        unimplemented!("T3: MambaCache::restore_prefix_state 通道回填")
    }

    pub fn mtp_rollback_mamba(&self, seq_id: usize, keep_tokens: usize) -> Result<bool> {
        self.mtp_rollback_mamba_at(seq_id, keep_tokens, 0)
    }

    pub fn mtp_rollback_mamba_at(
        &self,
        seq_id: usize,
        keep_tokens: usize,
        snapshot_offset: usize,
    ) -> Result<bool> {
        let mut mamba_cache = self.mamba_cache.write();
        let slots = self
            .get_mamba_slots_for_sequences(&[seq_id])?
            .into_iter()
            .map(|s| s as i64)
            .collect::<Vec<_>>();
        let seq_slots = super::layers::ctor::from_vec(slots, (1,), &self.device)?;
        for layer in &self.layers {
            if let Qwen3_5AttnType::LinearAttention(gdn) = &layer.attn {
                gdn.rollback_mtp_verify_at(
                    &mut mamba_cache,
                    &seq_slots,
                    keep_tokens,
                    snapshot_offset,
                )?;
            }
        }
        Ok(true)
    }

    pub fn reset_mamba_cache(&self) -> Result<()> {
        unimplemented!("T3: MambaCache::reset_all 通道回填")
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

// =============================================================================
// 适配台账(层/基建签名需求汇总;主 agent 批量合入 mod.rs/层文件)
// =============================================================================
//
// A1 vendor::InputMetadata 增字段:sequence_ids: Option<Vec<usize>>,
//    mamba_slot_mapping: Option<Tensor>(本文件局部结构体可整体退役)
// A2 层 forward 增 ctx: &KernelCtx 首参(S6 全量;本波层签名未带 ctx,
//    模型调用点保持 xinfer 形态,S6 scratch 通道自模型层启用)
// A3 VocabParallelLinear = TensorParallelRowLinear(已按此搬运;如后续
//    需要 vocab 专用分片对账面,在 distributed.rs 增 type alias)
// A4 ProgressLike/进度上报未移植(runner 波接线)
// A5 layer_metrics 面未移植(观测,随 metrics 里程碑)
// A6 log_error! 宏未移植(临时 eprintln)
// A7 OwlTensor 增 get(i)(行取视图)
// A8 OwlTensor 增 copy_d2d(dst_view, src) + dtype 分派(DFlash/MTP 双点)
// A9 MambaCache 方法通道:free_slot/ensure_slots_for_sequences/
//    get_slots_for_sequences/reserve_capacity/set_prefix_cache_capacity/
//    capture_prefix_state/has_prefix_state/remove_prefix_state/
//    restore_prefix_state/reset_all + 构造面(9 参数形状元数据)
// A10 ctor::from_vec 对 I64 的 dtype 面(S4 边界;slot 语义回填 U32 化
//     时同步裁剪)

// ============================================================================
// GraphForward 对接(T2-三 目标二;graph-model-seam.md §三)
// ============================================================================
