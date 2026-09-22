// src/utils/config.rs
use crate::error::Result;
use crate::transfer::PdConfig;
use llguidance::api::TopLevelGrammar;
use serde::de::value::SeqAccessDeserializer;
use serde::de::{Deserializer, Visitor};
use serde::ser::Error as _;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// 激活函数枚举(直译 candle_nn::Activation 的 serde 兼容面;
/// 变体集 = xinfer 代码库实际反序列化/匹配到的集合 + HF config.json
/// 常见别名,别名逐变体标注,与 candle 语义一致)。
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum Activation {
    /// 默认变体;HF 字符串 "gelu"
    #[default]
    #[serde(alias = "gelu")]
    Gelu,
    /// HF 字符串 "gelu_new"
    #[serde(alias = "gelu_new")]
    NewGelu,
    Relu,
    Relu2,
    Relu6,
    /// HF 字符串 "silu"(Qwen3.5 等小写原文)
    #[serde(alias = "silu", alias = "SiLU")]
    Silu,
    Sigmoid,
    HardSigmoid,
    Swiglu,
    Swish,
    HardSwish,
    Elu(f64),
    LeakyRelu(f64),
    /// HF 字符串 "gelu_pytorch_tanh"
    #[serde(alias = "gelu_pytorch_tanh")]
    GeluPytorchTanh,
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KvCacheDtype {
    Auto,
    Fp8,
    Turbo8,
    Turbo4,
    Turbo3,
    Nvfp4,
}

impl KvCacheDtype {
    pub fn is_turboquant(&self) -> bool {
        matches!(self, Self::Turbo8 | Self::Turbo4 | Self::Turbo3)
    }

    pub fn is_fp8_keys(&self) -> bool {
        matches!(self, Self::Fp8 | Self::Turbo8)
    }

    pub fn is_nvfp4(&self) -> bool {
        matches!(self, Self::Nvfp4)
    }

    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "auto" | "bf16" | "bfloat16" => Some(Self::Auto),
            "fp8" | "e4m3" => Some(Self::Fp8),
            "turbo8" | "k8v4" => Some(Self::Turbo8),
            "turbo4" | "4bit" => Some(Self::Turbo4),
            "turbo3" | "k3v4" => Some(Self::Turbo3),
            "nvfp4" | "ds_nvfp4" | "fp4" => Some(Self::Nvfp4),
            _ => None,
        }
    }
}

impl Default for KvCacheDtype {
    fn default() -> Self {
        Self::Auto
    }
}

impl std::fmt::Display for KvCacheDtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Fp8 => write!(f, "fp8"),
            Self::Turbo8 => write!(f, "turbo8"),
            Self::Turbo4 => write!(f, "turbo4"),
            Self::Turbo3 => write!(f, "turbo3"),
            Self::Nvfp4 => write!(f, "nvfp4"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl<'de> Deserialize<'de> for EosTokenId {
    fn deserialize<D>(deserializer: D) -> Result<EosTokenId, D::Error>
    where
        D: Deserializer<'de>,
    {
        if deserializer.is_human_readable() {
            // For JSON: deserialize as "untagged" using a visitor
            struct EosTokenIdVisitor;

            impl<'de> Visitor<'de> for EosTokenIdVisitor {
                type Value = EosTokenId;

                fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                    formatter.write_str("a u32 or a sequence of u32s")
                }

                // Handle a single number
                fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E> {
                    Ok(EosTokenId::Single(v as u32))
                }

                // Handle an array of numbers
                fn visit_seq<A>(self, seq: A) -> Result<Self::Value, A::Error>
                where
                    A: serde::de::SeqAccess<'de>,
                {
                    let vals = Vec::<u32>::deserialize(SeqAccessDeserializer::new(seq))?;
                    Ok(EosTokenId::Multiple(vals))
                }
            }

            deserializer.deserialize_any(EosTokenIdVisitor)
        } else {
            // For Bincode: deserialize as "tagged"
            let bincode_id = BincodeEosTokenId::deserialize(deserializer)?;
            let id = match bincode_id {
                BincodeEosTokenId::Single(v) => EosTokenId::Single(v),
                BincodeEosTokenId::Multiple(v) => EosTokenId::Multiple(v),
            };
            Ok(id)
        }
    }
}

impl Serialize for EosTokenId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if serializer.is_human_readable() {
            // For JSON: serialize as "untagged"
            match self {
                EosTokenId::Single(v) => v.serialize(serializer),
                EosTokenId::Multiple(v) => v.serialize(serializer),
            }
        } else {
            // For Bincode: serialize as "tagged"
            let bincode_id = match self {
                EosTokenId::Single(v) => BincodeEosTokenId::Single(*v),
                EosTokenId::Multiple(v) => BincodeEosTokenId::Multiple(v.clone()),
            };
            bincode_id.serialize(serializer)
        }
    }
}

impl EosTokenId {
    /// Merge `other` into `self`, returning the combined token set.
    /// - Single + Single => Multiple([a, b])
    /// - Single + Multiple => Multiple([a, ...])
    /// - Multiple + Single => Multiple([... , b])
    /// - Multiple + Multiple => Multiple([... , ...])
    pub fn merge(self, other: EosTokenId) -> EosTokenId {
        let mut out = self.into_vec();
        out.extend(other.into_vec());
        EosTokenId::Multiple(out)
    }

    /// Like merge, but de-duplicates while preserving first-seen order.
    pub fn merge_dedup(self, other: EosTokenId) -> EosTokenId {
        use std::collections::HashSet;

        let mut seen = HashSet::<u32>::new();
        let mut out = Vec::<u32>::new();

        for id in self.into_vec().into_iter().chain(other.into_vec()) {
            if seen.insert(id) {
                out.push(id);
            }
        }
        EosTokenId::Multiple(out)
    }

    pub fn to_vec(&self) -> Vec<u32> {
        match self {
            EosTokenId::Single(x) => vec![*x],
            EosTokenId::Multiple(v) => {
                // Deduplicate while preserving order
                let mut seen = HashSet::new();
                v.iter().filter(|&id| seen.insert(*id)).cloned().collect()
            }
        }
    }

    fn into_vec(self) -> Vec<u32> {
        match self {
            EosTokenId::Single(x) => vec![x],
            EosTokenId::Multiple(v) => v,
        }
    }
}

fn serialize_optional_grammar<S>(
    grammar: &Option<TopLevelGrammar>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if serializer.is_human_readable() {
        grammar.serialize(serializer)
    } else {
        let encoded = grammar
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(S::Error::custom)?;
        encoded.serialize(serializer)
    }
}

fn deserialize_optional_grammar<'de, D>(
    deserializer: D,
) -> Result<Option<TopLevelGrammar>, D::Error>
where
    D: Deserializer<'de>,
{
    if deserializer.is_human_readable() {
        Option::<TopLevelGrammar>::deserialize(deserializer)
    } else {
        let encoded = Option::<String>::deserialize(deserializer)?;
        encoded
            .map(|json| serde_json::from_str(&json).map_err(serde::de::Error::custom))
            .transpose()
    }
}

// To make the "tagged" logic work for bincode, we need a separate
// definition of the enum with derived traits. We keep it private inside this module.
#[derive(Serialize, Deserialize)]
enum BincodeEosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MoEConfig {
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: Option<usize>,
    #[serde(alias = "n_routed_experts", alias = "num_local_experts")]
    pub num_experts: Option<usize>,
    pub mlp_only_layers: Option<Vec<usize>>,
    pub decoder_sparse_step: Option<usize>,
    #[serde(default)]
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: Option<f64>,
    pub first_k_dense_replace: Option<usize>,
    pub n_shared_experts: Option<usize>,
    pub n_group: Option<usize>,
    pub topk_group: Option<usize>,
    #[serde(default)]
    pub scoring_func: Option<String>,
    #[serde(default)]
    pub topk_method: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RopeScalingValue {
    Bool(bool),
    Number(f64),
    NumberArray(Vec<f64>),
    String(String),
}

impl RopeScalingValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            RopeScalingValue::Number(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            RopeScalingValue::String(v) => Some(v),
            _ => None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Config {
    pub architectures: Option<Vec<String>>,
    pub head_dim: Option<usize>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub max_model_len: Option<usize>,
    #[serde(default, alias = "ffn_hidden_size", alias = "feed_forward_length")]
    pub intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: Option<usize>,
    pub rope_theta: Option<f64>,
    pub attention_bias: Option<bool>,
    pub qkv_bias: Option<bool>,
    pub attn_output_gate: Option<bool>,
    pub attn_logit_softcapping: Option<f64>,
    pub final_logit_softcapping: Option<f64>,
    pub tie_word_embeddings: Option<bool>,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<EosTokenId>,
    pub use_sliding_window: Option<bool>,
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub partial_rotary_factor: Option<f32>,
    #[serde(default)]
    pub output_gate_type: Option<String>,
    #[serde(alias = "hidden_activation")]
    pub hidden_act: Activation,
    #[serde(alias = "rope_parameters")]
    pub rope_scaling: Option<HashMap<String, RopeScalingValue>>,
    pub quant: Option<String>,
    pub moe_cfg: Option<MoEConfig>,
    #[serde(default)]
    pub kvcache_dtype: KvCacheDtype,
    pub quantization_config: Option<QuantConfig>,
    pub is_multi_model: Option<bool>,
    pub extra_config_json: Option<String>,
    #[serde(default)]
    pub is_f16_mode: bool,
    #[serde(default)]
    pub mtp_num_hidden_layers: Option<usize>,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: Option<bool>,
    #[serde(skip)]
    pub mtp_enabled: bool,
    /// Max packed verify tokens for GDN MTP/DFlash snapshot buffers
    /// (`max_num_seqs * (num_speculative_tokens + 1)`).
    #[serde(skip)]
    pub mtp_max_verify_tokens: usize,
    #[serde(default)]
    pub expert_dtype: Option<String>,
    // ---- Qwen3.5 hybrid(GDN 线性注意力)字段(字段名 = HF config.json 原文)----
    /// 逐层类型表("linear_attention" | "full_attention");None = 全 full_attention
    /// (兼容纯注意力旧模型)。alias 兼容 xinfer 命名 layers_block_type。
    #[serde(default, alias = "layers_block_type")]
    pub layer_types: Option<Vec<String>>,
    /// 无 layer_types 时的回退周期:第 (i+1)%interval==0 层为 full_attention
    #[serde(default)]
    pub full_attention_interval: Option<usize>,
    #[serde(default)]
    pub linear_num_value_heads: Option<usize>,
    #[serde(default)]
    pub linear_num_key_heads: Option<usize>,
    #[serde(default)]
    pub linear_key_head_dim: Option<usize>,
    #[serde(default)]
    pub linear_value_head_dim: Option<usize>,
    #[serde(default)]
    pub linear_conv_kernel_dim: Option<usize>,
}

impl fmt::Debug for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Config")
            .field("architectures", &self.architectures)
            .field("head_dim", &self.head_dim)
            .field("num_attention_heads", &self.num_attention_heads)
            .field("num_key_value_heads", &self.num_key_value_heads)
            .field("max_position_embeddings", &self.max_position_embeddings)
            .field("hidden_size", &self.hidden_size)
            .field("num_hidden_layers", &self.num_hidden_layers)
            .field("max_model_len", &self.max_model_len)
            .field("intermediate_size", &self.intermediate_size)
            .field("vocab_size", &self.vocab_size)
            .field("rope_theta", &self.rope_theta)
            .field("hidden_act", &self.hidden_act)
            .field("quant", &self.quant)
            .field("moe_cfg", &self.moe_cfg)
            .field("kvcache_dtype", &self.kvcache_dtype)
            .field("quantization_config", &self.quantization_config)
            .field("is_multi_model", &self.is_multi_model)
            .field("is_f16_mode", &self.is_f16_mode)
            .field("mtp_enabled", &self.mtp_enabled)
            .field("mtp_max_verify_tokens", &self.mtp_max_verify_tokens)
            .finish()
    }
}

impl Config {
    /// 从 config.json 原文反序列化。
    ///
    /// HF 多模态壳(architectures = Qwen3_5ForConditionalGeneration 等)把文本
    /// 模型字段嵌在 `text_config` 对象下;此处解壳:根对象与 text_config 合并,
    /// 同名键 text_config 优先(architectures / tie_word_embeddings /
    /// image_token_id 等根字段保留)。纯文本 config(无 text_config)原样反序列化。
    /// 另:`rope_parameters.{rope_theta,partial_rotary_factor}` 提升到顶层
    /// (Qwen3.5 把 rope 参数收在 rope_parameters 里;xinfer 装载分支同语义)。
    pub fn from_json_str(json: &str) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::error::Error::Msg(format!("config.json 解析失败: {e}")))?;
        let merged = match root.get("text_config") {
            Some(t) if t.is_object() => {
                let mut obj = root
                    .as_object()
                    .cloned()
                    .expect("root 为 JSON object(text_config 分支)");
                for (k, v) in t.as_object().expect("text_config 为 object") {
                    obj.insert(k.clone(), v.clone());
                }
                serde_json::Value::Object(obj)
            }
            _ => root,
        };
        // rope 参数提升(仅顶层缺省时;合并后的 rope_parameters 已入 rope_scaling)
        let mut merged = merged;
        if let Some(rp) = merged.get("rope_parameters").cloned() {
            if merged.get("rope_theta").and_then(serde_json::Value::as_f64).is_none() {
                if let Some(theta) = rp.get("rope_theta").and_then(serde_json::Value::as_f64) {
                    merged["rope_theta"] = serde_json::Value::from(theta);
                }
            }
            if merged
                .get("partial_rotary_factor")
                .and_then(serde_json::Value::as_f64)
                .is_none()
            {
                if let Some(prf) = rp
                    .get("partial_rotary_factor")
                    .and_then(serde_json::Value::as_f64)
                {
                    merged["partial_rotary_factor"] = serde_json::Value::from(prf as f32);
                }
            }
        }
        serde_json::from_value(merged)
            .map_err(|e| crate::error::Error::Msg(format!("config 反序列化失败: {e}")))
    }

    pub fn apply_generation_cfg(&mut self, generation_cfg: Option<&GenerationConfig>) {
        let Some(gcfg) = generation_cfg else { return };

        // BOS merge (fill if missing; config wins)
        if self.bos_token_id.is_none() {
            self.bos_token_id = gcfg.bos_token_id;
        }

        // EOS merge (combine if both present)
        self.eos_token_id = match (self.eos_token_id.take(), gcfg.eos_token_id.as_ref()) {
            (None, None) => None,
            (None, Some(e)) => Some(e.clone()),
            (Some(e), None) => Some(e),
            (Some(e), Some(other)) => Some(e.merge(other.clone())),
        };
    }

    pub fn higher_precision_required(&self) -> bool {
        let is_fp4_expert = self
            .expert_dtype
            .as_deref()
            .is_some_and(|dtype| matches!(dtype.to_ascii_lowercase().as_str(), "fp4" | "mxfp4"));
        self.is_f16_mode
            || self.quant.is_some()
            || self.quantization_config.as_ref().is_some_and(|cfg| {
                // Weight-only AWQ/GPTQ and compressed-tensors WNA16
                // models are especially sensitive to low-precision norm
                // statistics. Keep norm weights/reductions and Q/K
                // preparation in F32 while retaining the configured
                // activation dtype for the surrounding matmuls.
                cfg.is_compressed_tensors
                    || matches!(
                        cfg.quant_method.as_str(),
                        "awq"
                            | "gptq"
                            | "awq_marlin"
                            | "gptq_marlin"
                            | "compressed-tensors"
                            | "mxfp4"
                            | "nvfp4"
                    )
            })
            || is_fp4_expert
    }
}

pub const DEFAULT_PREFILL_CHUNK_SIZE: usize = 8 * 1024;
pub const MIN_PREFILL_CHUNK_SIZE: usize = 1024;
pub const MAX_PREFILL_CHUNK_SIZE: usize = 32 * 1024;

pub fn default_prefill_chunk_size() -> usize {
    DEFAULT_PREFILL_CHUNK_SIZE
}

pub fn normalize_prefill_chunk_size(size: usize) -> usize {
    let rounded = (size.saturating_add(MIN_PREFILL_CHUNK_SIZE / 2) / MIN_PREFILL_CHUNK_SIZE)
        * MIN_PREFILL_CHUNK_SIZE;
    rounded.clamp(MIN_PREFILL_CHUNK_SIZE, MAX_PREFILL_CHUNK_SIZE)
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EngineConfig {
    pub model_id: Option<String>,
    pub weight_path: Option<String>,
    pub weight_file: Option<String>,
    pub enforce_parser: Option<String>,
    pub hf_token: Option<String>,
    pub hf_token_path: Option<String>,
    pub num_blocks: usize,
    pub kv_fraction: Option<f32>, // After loading the model, the remaining percent of gpu used for kvcache
    pub mamba_fraction: Option<f32>, // Percent of cache budget reserved for hybrid mamba states
    pub cpu_mem_fold: Option<f32>, // the percentage of gpu kvcache: 0.1x to 10x, default 1.0x
    pub kvcache_memory_bytes: usize,
    #[serde(default)]
    pub mamba_memory_bytes: usize,
    #[serde(default)]
    pub mamba_slot_bytes: usize,
    #[serde(default)]
    pub mamba_cache_capacity: Option<usize>,
    pub block_size: usize,
    pub max_num_seqs: usize,
    /// User-requested request capacity. This is the allocation/preallocation
    /// limit; the scheduler uses `max_num_parallel_reqs` for active work.
    #[serde(default)]
    pub max_num_parallel_reqs: usize,
    pub max_num_batched_tokens: usize,
    #[serde(default)]
    pub max_kv_cache_tokens: usize,
    pub config_model_len: Option<usize>,
    pub max_model_len: Option<usize>,
    pub max_tokens: Option<usize>,
    pub isq: Option<String>,
    pub num_shards: Option<usize>,
    pub device_ids: Option<Vec<usize>>,
    pub generation_cfg: Option<GenerationConfig>,
    pub seed: Option<u64>,
    pub prefix_cache: Option<bool>,
    pub prefix_cache_max_tokens: Option<usize>,
    #[serde(default)]
    pub kvcache_dtype: KvCacheDtype,
    pub server_mode: Option<bool>,
    pub pd_config: Option<PdConfig>,
    pub mcp_command: Option<String>,
    pub mcp_config: Option<String>,
    pub mcp_args: Option<Vec<String>>,
    pub tool_prompt_template: Option<String>,
    pub pd_server_prefix_cache_ratio: Option<f32>,
    pub pd_client_prefix_cache_ratio: Option<f32>,
    pub yarn_scaling_factor: Option<f64>,
    #[serde(default)]
    pub disable_reasoning: bool,
    #[serde(default)]
    pub disable_cuda_graph: bool,
    #[serde(default = "default_prefill_chunk_size")]
    pub prefill_chunk_size: usize,
    #[serde(default = "default_num_nodes")]
    pub num_nodes: usize,
    #[serde(default)]
    pub node_rank: usize,
    pub master_addr: Option<String>,
    #[serde(default = "default_master_port")]
    pub master_port: u16,
    /// Number of speculative draft tokens per decode step.
    /// With built-in MTP heads (e.g. Qwen3.5), enables MTP decoding.
    /// With `--draft-model`, enables external DFlash2 decoding.
    #[serde(default)]
    pub num_speculative_tokens: Option<usize>,
    /// External DFlash2 draft model (HuggingFace id or local directory).
    #[serde(default)]
    pub draft_model: Option<String>,
    /// P5 草稿量化回退开关("off" = 强制 bf16 路径)。
    #[serde(default)]
    pub draft_quant: Option<String>,
    pub enable_tool_grammar: bool,
}

fn default_num_nodes() -> usize {
    1
}
fn default_master_port() -> u16 {
    29500
}

impl EngineConfig {
    pub fn effective_prefill_chunk_size(&self) -> usize {
        normalize_prefill_chunk_size(self.prefill_chunk_size)
    }

    pub fn with_num_speculative_tokens(mut self, tokens: Option<usize>) -> Self {
        self.num_speculative_tokens = tokens;
        self
    }
}

impl EngineConfig {
    pub fn new(
        model_id: Option<String>,
        weight_path: Option<String>,
        weight_file: Option<String>,
        hf_token: Option<String>,
        hf_token_path: Option<String>,
        enforce_parser: Option<String>,
        max_num_seqs: Option<usize>,
        config_model_len: Option<usize>,
        max_model_len: Option<usize>,
        max_tokens: Option<usize>,
        isq: Option<String>,
        num_shards: Option<usize>,
        device_ids: Option<Vec<usize>>,
        generation_cfg: Option<GenerationConfig>,
        seed: Option<u64>,
        disable_prefix_cache: bool,
        prefix_cache_max_tokens: Option<usize>,
        kvcache_dtype: Option<String>,
        server_mode: Option<bool>,
        cpu_mem_fold: Option<f32>,
        kv_fraction: Option<f32>,
        mamba_fraction: Option<f32>,
        pd_config: Option<PdConfig>,
        mcp_command: Option<String>,
        mcp_config: Option<String>,
        mcp_args: Option<Vec<String>>,
        tool_prompt_template: Option<String>,
        pd_server_prefix_cache_ratio: Option<f32>,
        pd_client_prefix_cache_ratio: Option<f32>,
        yarn_scaling_factor: Option<f64>,
        disable_reasoning: bool,
        disable_cuda_graph: bool,
        prefill_chunk_size: Option<usize>,
        num_nodes: usize,
        node_rank: usize,
        master_addr: Option<String>,
        master_port: u16,
        enable_tool_grammar: bool,
        num_speculative_tokens: Option<usize>,
        draft_model: Option<String>,
        draft_quant: Option<String>,
    ) -> Self {
        let mut device_ids = device_ids.unwrap_or_default();
        if device_ids.is_empty() {
            device_ids.push(0);
        }

        Self {
            model_id,
            weight_path,
            weight_file,
            hf_token,
            hf_token_path,
            enforce_parser,
            num_blocks: 128, //placeholder
            cpu_mem_fold,
            kv_fraction,
            mamba_fraction,
            kvcache_memory_bytes: 0, //placeholder
            mamba_memory_bytes: 0,
            mamba_slot_bytes: 0,
            mamba_cache_capacity: None,
            block_size: 64, // owl 仅 CUDA(与 xinfer 非-metal 分支一致)
            max_num_seqs: max_num_seqs.unwrap_or(32),
            max_num_parallel_reqs: max_num_seqs.unwrap_or(32), // placeholder; finalized after memory planning
            max_num_batched_tokens: max_num_seqs.unwrap_or(32) * 1024, //placeholder
            max_kv_cache_tokens: 0,
            config_model_len,
            max_model_len, //placeholder
            max_tokens,
            isq,
            num_shards,
            device_ids: Some(device_ids),
            generation_cfg,
            seed,
            prefix_cache: Some(!disable_prefix_cache),
            prefix_cache_max_tokens,
            kvcache_dtype: if let Some(ref s) = kvcache_dtype {
                KvCacheDtype::from_str_opt(s).unwrap_or(KvCacheDtype::Auto)
            } else {
                KvCacheDtype::Auto
            },
            server_mode,
            pd_config,
            mcp_command,
            mcp_config,
            mcp_args,
            tool_prompt_template,
            pd_server_prefix_cache_ratio,
            pd_client_prefix_cache_ratio,
            yarn_scaling_factor,
            disable_reasoning,
            disable_cuda_graph,
            prefill_chunk_size: normalize_prefill_chunk_size(
                prefill_chunk_size.unwrap_or(DEFAULT_PREFILL_CHUNK_SIZE),
            ),
            num_nodes,
            node_rank,
            master_addr,
            master_port,
            num_speculative_tokens,
            draft_model,
            draft_quant,
            enable_tool_grammar,
        }
    }
}

fn deserialize_token_field<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::de;
    struct TokenVisitor;
    impl<'de> de::Visitor<'de> for TokenVisitor {
        type Value = Option<String>;
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_str("a string, an AddedToken object with 'content', or null")
        }
        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }
        fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
            Ok(Some(v.to_owned()))
        }
        fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
            Ok(Some(v))
        }
        fn visit_map<A: de::MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
            let mut content: Option<String> = None;
            while let Some(key) = map.next_key::<String>()? {
                if key == "content" {
                    content = Some(map.next_value()?);
                } else {
                    let _ = map.next_value::<serde::de::IgnoredAny>()?;
                }
            }
            Ok(content)
        }
    }
    deserializer.deserialize_any(TokenVisitor)
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct TokenizerConfig {
    pub model_max_length: Option<f64>,
    pub add_bos_token: Option<bool>,
    pub add_eos_token: Option<bool>,
    pub chat_template: Option<String>,
    #[serde(default, deserialize_with = "deserialize_token_field")]
    pub bos_token: Option<String>,
    #[serde(default, deserialize_with = "deserialize_token_field")]
    pub eos_token: Option<String>,
    #[serde(default, deserialize_with = "deserialize_token_field")]
    pub pad_token: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub max_tokens: Option<usize>,
    pub ignore_eos: bool,
    pub top_k: Option<isize>,
    pub top_p: Option<f32>,
    pub session_id: Option<String>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(skip)]
    pub stop_token_ids: Option<Vec<Vec<u32>>>,
    #[serde(alias = "enable_thinking")]
    pub thinking: Option<bool>, // enable reasoning
    /// Tool mode for tool call handling.
    /// If Some(true), external tools are enabled and stream finishes at </tool_call>.
    #[serde(default)]
    pub mcp_mode: Option<bool>,
    /// Grammar constraint as TopLevelGrammar for RPC serialization
    #[serde(default)]
    #[serde(
        serialize_with = "serialize_optional_grammar",
        deserialize_with = "deserialize_optional_grammar"
    )]
    pub grammar: Option<TopLevelGrammar>,
    #[serde(default)]
    pub grammar_json: Option<String>,
    /// Reasoning effort level for OpenAI-compatible reasoning API
    #[serde(default)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Token IDs marking end of reasoning (e.g. </think>).
    /// When set, grammar constraints are deferred until after reasoning ends.
    #[serde(default)]
    pub guidance_reasoning_end_ids: Vec<u32>,
}

impl SamplingParams {
    pub fn new(
        temperature: Option<f32>,
        max_tokens: Option<usize>,
        ignore_eos: Option<bool>,
        top_k: Option<isize>,
        top_p: Option<f32>,
        session_id: Option<String>,
        frequency_penalty: Option<f32>,
        presence_penalty: Option<f32>,
        thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Self {
        Self {
            temperature,
            max_tokens,
            ignore_eos: ignore_eos.unwrap_or(false),
            top_k,
            top_p,
            session_id,
            frequency_penalty,
            presence_penalty,
            mcp_mode: None,
            stop_sequences: None,
            stop_token_ids: None,
            thinking,
            grammar: None,
            grammar_json: None,
            reasoning_effort,
            guidance_reasoning_end_ids: Vec::new(),
        }
    }

    pub fn new_with_max_tokens(max_tokens: usize) -> Self {
        Self {
            temperature: None,
            max_tokens: Some(max_tokens),
            ignore_eos: false,
            top_k: None,
            top_p: None,
            session_id: None,
            frequency_penalty: None,
            presence_penalty: None,
            mcp_mode: None,
            stop_sequences: None,
            stop_token_ids: None,
            thinking: None,
            grammar: None,
            grammar_json: None,
            reasoning_effort: None,
            guidance_reasoning_end_ids: Vec::new(),
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: None,
            max_tokens: Some(16384),
            ignore_eos: false,
            top_k: None,
            top_p: None,
            session_id: None,
            frequency_penalty: None,
            presence_penalty: None,
            mcp_mode: None,
            stop_sequences: None,
            stop_token_ids: None,
            thinking: None,
            grammar: None,
            grammar_json: None,
            reasoning_effort: None,
            guidance_reasoning_end_ids: Vec::new(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ModelType {
    Qwen3,
    Qwen3MoE,
    Qwen3_5,
    Qwen3_5MoE,
    LLaMa,
    Gemma,
    Gemma3,
    Gemma4,
    Phi,
    Phi4,
    Mistral,
    GLM4,
    GLM4MoE,
    GLM4MoeLite,
    Yi,
    StableLM,
    DeepSeek,
    DeepSeekV4,
    GLM5,
    Qwen4,
    Mistral3VL,
    Qwen3VL,
    LLaMa4,
    MiniMax,
}
impl ModelType {
    /// Convert architecture string to ModelType
    pub fn from_architectures(architectures: &[String]) -> Option<Self> {
        architectures.first().and_then(|arch| match arch.as_str() {
            "Qwen3ForCausalLM" => Some(ModelType::Qwen3),
            "Qwen3MoEForCausalLM" => Some(ModelType::Qwen3MoE),
            "Qwen3_5ForCausalLM" => Some(ModelType::Qwen3_5),
            "Qwen3_5MoEForCausalLM" => Some(ModelType::Qwen3_5MoE),
            "Qwen3NextForCausalLM" => Some(ModelType::Qwen3_5MoE),
            "Qwen4ExpForConditionalGeneration" | "Qwen4ExpForCausalLM" => Some(ModelType::Qwen4),
            "LlamaForCausalLM" => Some(ModelType::LLaMa),
            "GemmaForCausalLM" => Some(ModelType::Gemma),
            "Gemma3ForConditionalGeneration" => Some(ModelType::Gemma3),
            "Gemma4ForConditionalGeneration" => Some(ModelType::Gemma4),
            "Gemma4ForCausalLM" => Some(ModelType::Gemma4),
            "PhiForCausalLM" => Some(ModelType::Phi),
            "Phi4ForCausalLM" => Some(ModelType::Phi4),
            "MistralForCausalLM" => Some(ModelType::Mistral),
            "GLM4ForCausalLM" => Some(ModelType::GLM4),
            "GLM4MoEForCausalLM" => Some(ModelType::GLM4MoE),
            "YiForCausalLM" => Some(ModelType::Yi),
            "StableLmForCausalLM" => Some(ModelType::StableLM),
            "DeepSeekForCausalLM" => Some(ModelType::DeepSeek),
            "Mistral3VForConditionalGeneration" => Some(ModelType::Mistral3VL),
            "Qwen3VLMoEForConditionalGeneration" => Some(ModelType::Qwen3VL),
            _ => None,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Randomness of sampling.
    /// rec. default = 1
    pub temperature: Option<f32>,
    /// Cumulative prob of the top tokens to consider, must be in (0, 1]. Set 1 to consider all toks.  
    /// rec. default = 1    
    pub top_p: Option<f32>,
    /// Control the number of top tokens to consider, set -1 to consider all.
    /// rec. default = -1
    pub top_k: Option<isize>,

    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,

    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<EosTokenId>,
}

/// Match a module path against an ignore pattern.
/// Supports three pattern types:
///   - `re:` prefix → regex match
///   - Contains `*` → glob match (converted to regex: `*` becomes `.*`)
///   - Otherwise → literal suffix matching
pub fn match_ignore_pattern(module_path: &str, pattern: &str) -> bool {
    if let Some(re_pat) = pattern.strip_prefix("re:") {
        if let Ok(re) = regex::Regex::new(re_pat) {
            return re.is_match(module_path);
        }
        return false;
    }
    if pattern.contains('*') {
        let re_pat = format!("^{}$", regex::escape(pattern).replace(r"\*", ".*"));
        if let Ok(re) = regex::Regex::new(&re_pat) {
            return re.is_match(module_path);
        }
        return false;
    }
    let module_path = module_path.trim_end_matches(".weight");
    let item = pattern.trim_end_matches(".weight");
    module_path == item
        || module_path.ends_with(item)
        || module_path.ends_with(&format!(".{item}"))
        || item.ends_with(module_path)
        || item.ends_with(&format!(".{module_path}"))
}

#[derive(Serialize, Deserialize, PartialEq, Clone)]
pub struct QuantConfig {
    #[serde(default)]
    pub quant_method: String,
    #[serde(default)]
    pub bits: usize,
    #[serde(default)]
    pub group_size: i32,
    pub sym: Option<bool>,
    pub desc_act: Option<bool>,
    pub checkpoint_format: Option<String>,
    pub fmt: Option<String>,
    #[serde(default)]
    pub scale_fmt: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    pub weight_block_size: Option<Vec<usize>>,
    #[serde(default, alias = "ignore")]
    pub modules_to_not_convert: Vec<String>,
    #[serde(default)]
    pub config_groups: Option<serde_json::Value>,
    #[serde(default)]
    pub quant_algo: Option<String>,
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub quantized_layers: Option<serde_json::Value>,
    /// MLX NVFP4 uses U32-packed weights (8 nibbles per U32) instead of U8
    /// (2 nibbles per byte). Set during normalize_compressed_tensors when
    /// an MLX-style `"mode": "nvfp4"` config is detected.
    #[serde(default)]
    pub is_mlx_nvfp4: bool,
    /// compressed-tensors `pack-quantized` WNA16 weights.  This format is
    /// algorithm-neutral (AWQ/GPTQ recipes use the same packed tensors).
    #[serde(default)]
    pub is_compressed_tensors: bool,
}

impl QuantConfig {
    /// Normalizes a quantization config into a canonical quant_method string.
    ///
    /// Handles the following families:
    ///   1. `modelopt` with `quant_algo` == `NVFP4` / `FP4`
    ///   2. `compressed-tensors` with `format` containing `nvfp4` or `mxfp4`
    ///   3. `compressed-tensors` detected from `config_groups` content
    ///   4. MLX-style `"mode": "nvfp4"` — weights are U32-packed (8 nibbles per
    ///      U32) with FP8 E4M3 scales, no global_scale. Repacked to U8 at load.
    ///
    /// Also extracts group_size / bits from config_groups when present.
    pub fn normalize_compressed_tensors(&mut self) {
        if self.quant_method.is_empty() {
            if let Some(mode) = &self.mode {
                if mode.eq_ignore_ascii_case("nvfp4") {
                    self.quant_method = "nvfp4".to_string();
                    self.is_mlx_nvfp4 = true;
                    if self.group_size == 0 {
                        self.group_size = 16;
                    }
                    if self.bits == 0 {
                        self.bits = 4;
                    }
                    return;
                }
            }
        }
        // ModelOpt mixed checkpoints are published with both
        // `quant_method: "modelopt"` and `quant_method: "modelopt_mixed"`.
        // Treat the latter as an alias so the existing quantized-layer and
        // config-group detection selects the actual runtime format (NVFP4 or
        // FP8) instead of rejecting it in the config assertion.
        if self.quant_method.eq_ignore_ascii_case("modelopt_mixed") {
            self.quant_method = "modelopt".to_string();
        }

        // modelopt: {"quant_method": "modelopt", "quant_algo": "NVFP4"}
        if self.quant_method == "modelopt" {
            if let Some(algo) = &self.quant_algo {
                if algo.eq_ignore_ascii_case("NVFP4") || algo.eq_ignore_ascii_case("FP4") {
                    self.quant_method = "nvfp4".to_string();
                    self.extract_compressed_tensors_params();
                    if self.group_size == 0 {
                        self.group_size = 16;
                    }
                    if self.bits == 0 {
                        self.bits = 4;
                    }
                    return;
                }
                if algo.eq_ignore_ascii_case("FP8") {
                    self.quant_method = "fp8".to_string();
                    return;
                }
                if algo.eq_ignore_ascii_case("MIXED_PRECISION") {
                    if self.detect_nvfp4_from_config_groups()
                        || self.detect_nvfp4_from_quantized_layers()
                    {
                        self.quant_method = "nvfp4".to_string();
                        self.bits = 4;
                        self.group_size = 16;
                    } else if self.detect_fp8_from_quantized_layers() {
                        self.quant_method = "fp8".to_string();
                    }
                    return;
                }
            }
            if self.detect_nvfp4_from_config_groups() {
                self.quant_method = "nvfp4".to_string();
                self.extract_compressed_tensors_params();
                if self.group_size == 0 {
                    self.group_size = 16;
                }
                if self.bits == 0 {
                    self.bits = 4;
                }
                return;
            }
        }

        if self.quant_method != "compressed-tensors" {
            return;
        }

        // compressed-tensors: check format string for nvfp4 or mxfp4
        let format_str = self.format.as_deref().unwrap_or("");

        let is_nvfp4 = format_str.contains("nvfp4") || self.detect_nvfp4_from_config_groups();

        if is_nvfp4 {
            self.quant_method = "nvfp4".to_string();
            self.extract_compressed_tensors_params();
            if self.group_size == 0 {
                self.group_size = 16;
            }
            if self.bits == 0 {
                self.bits = 4;
            }
            return;
        }

        let is_mxfp4 = format_str.contains("mxfp4") || self.detect_mxfp4_from_config_groups();

        if is_mxfp4 {
            self.quant_method = "mxfp4".to_string();
            self.extract_compressed_tensors_params();
            return;
        }

        // llm-compressor's AWQ/GPTQ WNA16 checkpoints use the generic
        // compressed-tensors `pack-quantized` format.  Unlike legacy AWQ and
        // GPTQ files they contain weight_packed/weight_scale/weight_shape
        // (and optionally weight_zero_point), so preserve that distinction
        // for the loaders while exposing the common WNA16 parameters.
        if format_str == "pack-quantized" || self.detect_pack_quantized_from_config_groups() {
            self.quant_method = "compressed-tensors".to_string();
            self.is_compressed_tensors = true;
            self.extract_compressed_tensors_params();
            // P5-Q1b:对称性必须从 config_groups 读(weights.symmetric),
            // 非对称(AWQ zp)决定是否走 kU4 通路 —— 错报 true 会静默错载。
            if let Some(groups) = self.config_groups.as_ref() {
                if let Some(sym) = groups
                    .get("group_0")
                    .and_then(|g| g.get("weights"))
                    .and_then(|w| w.get("symmetric"))
                    .and_then(|v| v.as_bool())
                {
                    self.sym = Some(sym);
                }
            }
            if self.sym.is_none() {
                self.sym = Some(true);
            }
        }
    }

    fn detect_pack_quantized_from_config_groups(&self) -> bool {
        let groups = match &self.config_groups {
            Some(v) => v,
            None => return false,
        };
        groups.as_object().is_some_and(|obj| {
            obj.values().any(|group| {
                group
                    .get("format")
                    .and_then(|v| v.as_str())
                    .is_some_and(|fmt| fmt == "pack-quantized")
            })
        })
    }

    fn detect_nvfp4_from_config_groups(&self) -> bool {
        let groups = match &self.config_groups {
            Some(v) => v,
            None => return false,
        };
        if let Some(obj) = groups.as_object() {
            for (_key, group) in obj {
                // Check group-level format (e.g. "nvfp4-pack-quantized")
                if let Some(fmt) = group.get("format").and_then(|v| v.as_str()) {
                    if fmt.contains("nvfp4") {
                        return true;
                    }
                }
                if let Some(weights) = group.get("weights") {
                    // Check weights-level format
                    if let Some(fmt) = weights.get("format").and_then(|v| v.as_str()) {
                        if fmt.contains("nvfp4") {
                            return true;
                        }
                    }
                    // Detect by parameters: 4-bit float with group_size=16
                    if let Some(num_bits) = weights.get("num_bits").and_then(|v| v.as_u64()) {
                        if num_bits == 4 {
                            let is_float = weights
                                .get("type")
                                .and_then(|v| v.as_str())
                                .map(|t| t == "float")
                                .unwrap_or(false);
                            let gs = weights
                                .get("group_size")
                                .and_then(|v| v.as_u64())
                                .unwrap_or(0);
                            if is_float && gs == 16 {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn detect_mxfp4_from_config_groups(&self) -> bool {
        let groups = match &self.config_groups {
            Some(v) => v,
            None => return false,
        };
        if let Some(obj) = groups.as_object() {
            for (_key, group) in obj {
                if let Some(fmt) = group.get("format").and_then(|v| v.as_str()) {
                    if fmt.contains("mxfp4") {
                        return true;
                    }
                }
                if let Some(weights) = group.get("weights") {
                    if let Some(fmt) = weights.get("format").and_then(|v| v.as_str()) {
                        if fmt.contains("mxfp4") {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn detect_nvfp4_from_quantized_layers(&self) -> bool {
        let layers = match &self.quantized_layers {
            Some(v) => v,
            None => return false,
        };
        if let Some(obj) = layers.as_object() {
            for (_key, layer_info) in obj {
                if let Some(algo) = layer_info.get("quant_algo").and_then(|v| v.as_str()) {
                    if algo.contains("NVFP4") || algo.contains("nvfp4") {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn detect_fp8_from_quantized_layers(&self) -> bool {
        let layers = match &self.quantized_layers {
            Some(v) => v,
            None => return false,
        };
        if let Some(obj) = layers.as_object() {
            for (_key, layer_info) in obj {
                if let Some(algo) = layer_info.get("quant_algo").and_then(|v| v.as_str()) {
                    if algo == "FP8" || algo == "fp8" {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn extract_compressed_tensors_params(&mut self) {
        let groups = match &self.config_groups {
            Some(v) => v.clone(),
            None => return,
        };
        if let Some(obj) = groups.as_object() {
            for (_key, group) in obj {
                if let Some(weights) = group.get("weights") {
                    if self.group_size == 0 {
                        if let Some(gs) = weights.get("group_size").and_then(|v| v.as_i64()) {
                            self.group_size = gs as i32;
                        }
                    }
                    if self.bits == 0 {
                        if let Some(nb) = weights.get("num_bits").and_then(|v| v.as_u64()) {
                            self.bits = nb as usize;
                        }
                    }
                }
            }
        }
    }

    /// Check if a module path should be skipped for this quantization config.
    /// Supports literal paths and `re:` prefixed regex patterns in
    /// `modules_to_not_convert` / `ignore`.
    pub fn should_skip_module(&self, module_path: &str) -> bool {
        if module_path.is_empty() || self.modules_to_not_convert.is_empty() {
            return false;
        }
        self.modules_to_not_convert
            .iter()
            .any(|item| match_ignore_pattern(module_path, item))
    }
}

impl fmt::Debug for QuantConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QuantConfig")
            .field("quant_method", &self.quant_method)
            .field("is_compressed_tensors", &self.is_compressed_tensors)
            .field("bits", &self.bits)
            .field("group_size", &self.group_size)
            .field("sym", &self.sym)
            .field("desc_act", &self.desc_act)
            .field("checkpoint_format", &self.checkpoint_format)
            .field("fmt", &self.fmt)
            .field("scale_fmt", &self.scale_fmt)
            .field("format", &self.format)
            .field("weight_block_size", &self.weight_block_size)
            .field("is_mlx_nvfp4", &self.is_mlx_nvfp4)
            .finish()
    }
}

/// Reasoning effort level for grammar generation
/// Optimized for specific reasoning strategies based on current research (2024-2025)
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// No structured reasoning - direct output only
    None,
    /// Default model reasoning output as induced by opening a reasoning tag
    ModelDefault,
    /// Constrained single-paragraph reasoning (~150 chars max)
    Low,
    /// Standard multi-step Chain-of-Thought (CoT)
    Medium,
    /// Adversarial analysis with self-correction phases
    High,
    /// Best-of-breed Chain-of-Verification (CoVe) + Self-Critique
    ChainOfThought,
    /// Custom user-provided grammar template (non-Python builds only)
    Custom(String),
}

impl Default for ReasoningEffort {
    fn default() -> Self {
        ReasoningEffort::ModelDefault
    }
}

impl ReasoningEffort {
    /// Parse a string to ReasoningEffort
    pub fn from_str(s: String) -> Self {
        match s.to_lowercase().as_str() {
            "none" => Self::None,
            "model_default" | "default" => Self::ModelDefault,
            "low" => Self::Low,
            "normal" | "medium" => Self::Medium,
            "high" => Self::High,
            "xhigh" | "x_high" | "very_high" | "maximum" | "max" => Self::High,
            "chain_of_thought" | "cot" | "cove" => Self::ChainOfThought,
                    s if s.starts_with("custom:") => Self::Custom(s[7..].to_string()),
            _ => Self::None,
        }
    }

    /// Check if reasoning effort is enabled (not None)
    pub fn is_enabled(&self) -> bool {
        *self != ReasoningEffort::None
    }

    /// Convert the normalized effort to the value expected by a model's chat
    /// template. OpenAI-compatible `high` and `xhigh` both map to Qwen3.8's
    /// supported `xhigh` spelling; the enum intentionally keeps them unified
    /// for the rest of the inference stack.
    pub fn chat_template_value(&self) -> Option<String> {
        match self {
            ReasoningEffort::None | ReasoningEffort::ModelDefault => None,
            ReasoningEffort::Low => Some("low".to_string()),
            ReasoningEffort::Medium => Some("medium".to_string()),
            ReasoningEffort::High => Some("xhigh".to_string()),
            ReasoningEffort::ChainOfThought => Some("xhigh".to_string()),
                    ReasoningEffort::Custom(value) => Some(value.clone()),
        }
    }
}

/// Conversion to string for serialization
impl std::fmt::Display for ReasoningEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ReasoningEffort::None => write!(f, "none"),
            ReasoningEffort::ModelDefault => write!(f, "model_default"),
            ReasoningEffort::Low => write!(f, "low"),
            ReasoningEffort::Medium => write!(f, "medium"),
            ReasoningEffort::High => write!(f, "high"),
            ReasoningEffort::ChainOfThought => write!(f, "chain_of_thought"),
                    ReasoningEffort::Custom(_) => write!(f, "custom"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_ignore_literal_exact() {
        assert!(match_ignore_pattern("lm_head", "lm_head"));
        assert!(match_ignore_pattern(
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.q_proj"
        ));
    }

    #[test]
    fn test_match_ignore_literal_suffix() {
        assert!(match_ignore_pattern(
            "model.language_model.layers.0.linear_attn.out_proj",
            "model.language_model.layers.0.linear_attn.out_proj"
        ));
        assert!(match_ignore_pattern("model.lm_head.weight", "lm_head"));
    }

    #[test]
    fn test_match_ignore_regex() {
        assert!(match_ignore_pattern(
            "model.layers.5.self_attn.q_proj",
            "re:.*self_attn.*"
        ));
        assert!(match_ignore_pattern(
            "model.layers.10.linear_attn.in_proj_qkv",
            "re:.*linear_attn.*"
        ));
        assert!(match_ignore_pattern(
            "model.layers.3.mlp.gate",
            "re:.*.mlp.gate$"
        ));
        assert!(!match_ignore_pattern(
            "model.layers.3.mlp.gate_proj",
            "re:.*.mlp.gate$"
        ));
        assert!(match_ignore_pattern(
            "model.visual.blocks.0.attn.qkv",
            "re:.*visual.*"
        ));
        assert!(match_ignore_pattern("mtp.fc", "re:.*mtp.*"));
        assert!(match_ignore_pattern(
            "model.embed_tokens",
            "re:.*embed_tokens.*"
        ));
    }

    #[test]
    fn test_match_ignore_regex_no_false_positive() {
        assert!(!match_ignore_pattern(
            "model.layers.5.mlp.up_proj",
            "re:.*self_attn.*"
        ));
        assert!(!match_ignore_pattern(
            "model.layers.5.mlp.up_proj",
            "re:.*linear_attn.*"
        ));
        assert!(!match_ignore_pattern(
            "model.layers.5.mlp.up_proj",
            "re:.*lm_head.*"
        ));
    }

    #[test]
    fn test_should_skip_module() {
        let cfg = QuantConfig {
            quant_method: "mxfp4".to_string(),
            bits: 4,
            group_size: 32,
            sym: None,
            desc_act: None,
            checkpoint_format: None,
            fmt: None,
            scale_fmt: None,
            format: Some("mxfp4-pack-quantized".to_string()),
            weight_block_size: None,
            modules_to_not_convert: vec![
                "re:.*self_attn.*".to_string(),
                "re:.*linear_attn.*".to_string(),
                "re:.*.mlp.gate$".to_string(),
                "re:.*lm_head.*".to_string(),
                "re:.*embed_tokens.*".to_string(),
                "re:.*visual.*".to_string(),
                "re:.*mtp.*".to_string(),
            ],
            config_groups: None,
            quant_algo: None,
            mode: None,
            quantized_layers: None,
            is_mlx_nvfp4: false,
            is_compressed_tensors: false,
        };
        assert!(cfg.should_skip_module("model.layers.0.self_attn.q_proj"));
        assert!(cfg.should_skip_module("model.layers.5.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("model.layers.3.mlp.gate"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.gate_proj"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.up_proj"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.down_proj"));
        assert!(cfg.should_skip_module("lm_head"));
        assert!(cfg.should_skip_module("model.visual.blocks.0.attn.qkv"));
        assert!(cfg.should_skip_module("mtp.fc"));
    }

    #[test]
    fn test_normalize_compressed_tensors_regex_ignore() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "weights": {"num_bits": 4, "group_size": 32, "strategy": "group", "symmetric": true}
                }
            },
            "ignore": [
                "re:.*self_attn.*",
                "re:.*linear_attn.*",
                "re:.*.mlp.gate$",
                "re:.*lm_head.*",
                "re:.*embed_tokens.*",
                "re:.*visual.*",
                "re:.*mtp.*"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.modules_to_not_convert.len(), 7);
        assert!(cfg.should_skip_module("model.layers.5.self_attn.q_proj"));
        assert!(!cfg.should_skip_module("model.layers.5.mlp.up_proj"));
    }

    #[test]
    fn test_normalize_compressed_tensors_literal_ignore() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "weights": {"num_bits": 4, "group_size": 32}
                }
            },
            "ignore": [
                "model.layers.0.linear_attn.out_proj",
                "model.layers.0.linear_attn.in_proj_qkv",
                "lm_head"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert!(cfg.should_skip_module("model.layers.0.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("lm_head"));
        assert!(!cfg.should_skip_module("model.layers.1.mlp.up_proj"));
    }

    #[test]
    fn test_normalize_format_in_config_groups_only() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "weights": {"num_bits": 4, "group_size": 32, "type": "float", "strategy": "group", "symmetric": true}
                }
            },
            "ignore": ["lm_head"]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
    }

    #[test]
    fn test_olka_4b_config() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "group",
                        "group_size": 32,
                        "symmetric": true
                    }
                }
            },
            "ignore": [
                "model.language_model.embed_tokens",
                "model.language_model.layers.0.input_layernorm",
                "model.language_model.layers.0.linear_attn.conv1d",
                "model.language_model.layers.0.linear_attn.in_proj_a",
                "model.language_model.norm",
                "model.visual.blocks.0.attn.proj",
                "mtp.fc"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.should_skip_module("model.language_model.embed_tokens"));
        assert!(cfg.should_skip_module("model.language_model.layers.0.linear_attn.conv1d"));
        assert!(!cfg.should_skip_module("model.language_model.layers.0.mlp.up_proj"));
    }

    #[test]
    fn test_kaitchup_27b_config() {
        let json = r#"{
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "input_activations": null,
                    "output_activations": null,
                    "targets": ["Linear"],
                    "weights": {
                        "actorder": null,
                        "block_structure": null,
                        "dynamic": false,
                        "group_size": 32,
                        "num_bits": 4,
                        "observer": "memoryless_minmax",
                        "observer_kwargs": {},
                        "scale_dtype": "torch.uint8",
                        "strategy": "group",
                        "symmetric": true,
                        "type": "float",
                        "zp_dtype": null
                    }
                }
            },
            "format": "mxfp4-pack-quantized",
            "global_compression_ratio": null,
            "ignore": [
                "model.visual.blocks.0.attn.qkv",
                "model.language_model.layers.0.linear_attn.out_proj",
                "lm_head"
            ],
            "kv_cache_scheme": null,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
            "version": "0.14.5.dev54+g704d57b"
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.should_skip_module("model.language_model.layers.0.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("lm_head"));
    }

    #[test]
    fn test_122b_regex_config() {
        let json = r#"{
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "quantization_status": "compressed",
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "group",
                        "group_size": 32,
                        "symmetric": true,
                        "scale_dtype": "torch.uint8",
                        "dynamic": false,
                        "actorder": null,
                        "block_structure": null,
                        "observer": "minmax",
                        "observer_kwargs": {},
                        "zp_dtype": null
                    },
                    "targets": ["Linear"],
                    "input_activations": null,
                    "output_activations": null
                }
            },
            "ignore": [
                "re:.*self_attn.*",
                "re:.*linear_attn.*",
                "re:.*.mlp.gate$",
                "re:.*shared_expert_gate.*",
                "re:.*lm_head.*",
                "re:.*embed_tokens.*",
                "re:.*visual.*",
                "re:.*mtp.*"
            ],
            "kv_cache_scheme": null,
            "sparsity_config": {},
            "transform_config": {}
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.should_skip_module("model.layers.5.self_attn.q_proj"));
        assert!(cfg.should_skip_module("model.layers.5.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("model.layers.3.mlp.gate"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.gate_proj"));
        assert!(cfg.should_skip_module("model.layers.3.shared_expert_gate"));
        assert!(cfg.should_skip_module("lm_head"));
        assert!(cfg.should_skip_module("model.embed_tokens"));
        assert!(cfg.should_skip_module("model.visual.blocks.0.attn.qkv"));
        assert!(cfg.should_skip_module("mtp.fc"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.up_proj"));
        assert!(!cfg.should_skip_module("model.layers.3.mlp.down_proj"));
    }

    #[test]
    fn test_2imi9_9b_config() {
        let json = r#"{
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "input_activations": null,
                    "output_activations": null,
                    "targets": ["Linear"],
                    "weights": {
                        "actorder": null,
                        "block_structure": null,
                        "dynamic": false,
                        "group_size": 32,
                        "num_bits": 4,
                        "observer": "memoryless_minmax",
                        "observer_kwargs": {},
                        "scale_dtype": "torch.uint8",
                        "strategy": "group",
                        "symmetric": true,
                        "type": "float",
                        "zp_dtype": null
                    }
                }
            },
            "format": "mxfp4-pack-quantized",
            "global_compression_ratio": null,
            "ignore": [
                "model.layers.0.linear_attn.out_proj",
                "model.layers.0.linear_attn.in_proj_qkv",
                "lm_head"
            ],
            "kv_cache_scheme": null,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
            "version": "0.14.5.a20260310"
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "mxfp4");
        assert_eq!(cfg.group_size, 32);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.should_skip_module("model.layers.0.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("lm_head"));
        assert!(!cfg.should_skip_module("model.layers.1.mlp.up_proj"));
    }

    #[test]
    fn test_nvfp4_axionml_4b_config() {
        let json = r#"{
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["Linear"]
                }
            },
            "ignore": [
                "lm_head",
                "model.language_model.layers.0.linear_attn.conv1d",
                "model.visual*",
                "mtp.layers.0*"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert_eq!(cfg.group_size, 16);
        assert_eq!(cfg.bits, 4);
        assert!(cfg.should_skip_module("lm_head"));
        assert!(cfg.should_skip_module("model.visual.encoder.layers.0.self_attn"));
        assert!(cfg.should_skip_module("mtp.layers.0.mlp.gate_proj"));
        assert!(!cfg.should_skip_module("model.language_model.layers.1.mlp.up_proj"));
    }

    #[test]
    fn test_nvfp4_glob_wildcards() {
        let json = r#"{
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "ignore": [
                "lm_head",
                "*.mlp.shared_expert.*",
                "model.layers.0.self_attn*",
                "model.layers.92*"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert!(cfg.should_skip_module("lm_head"));
        assert!(cfg.should_skip_module("model.layers.5.mlp.shared_expert.gate_proj"));
        assert!(cfg.should_skip_module("model.layers.0.self_attn.q_proj"));
        assert!(cfg.should_skip_module("model.layers.0.self_attn.k_proj"));
        assert!(cfg.should_skip_module("model.layers.92.self_attn.q_proj"));
        assert!(!cfg.should_skip_module("model.layers.1.self_attn.q_proj"));
        assert!(!cfg.should_skip_module("model.layers.5.mlp.gate_proj"));
    }

    #[test]
    fn test_mlx_nvfp4_normalized() {
        // MLX-community models use U32-packed weights with FP8 E4M3 scales.
        // They should normalize to "nvfp4" with is_mlx_nvfp4 = true, and
        // the weight loader repacks U32 → U8 at load time.
        let json = r#"{
            "group_size": 16,
            "bits": 4,
            "mode": "nvfp4"
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.quant_method, "");
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert!(cfg.is_mlx_nvfp4, "MLX mode=nvfp4 must set is_mlx_nvfp4");
        assert_eq!(cfg.group_size, 16);
        assert_eq!(cfg.bits, 4);
    }

    #[test]
    fn test_nvfp4_compressed_tensors_format() {
        // RedHatAI/Qwen3.5-122B-A10B-NVFP4 style: compressed-tensors + nvfp4-pack-quantized
        let json = r#"{
            "quant_method": "compressed-tensors",
            "format": "nvfp4-pack-quantized",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                        "strategy": "tensor_group",
                        "symmetric": true,
                        "dynamic": false,
                        "scale_dtype": "torch.float8_e4m3fn"
                    },
                    "input_activations": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16,
                        "dynamic": "local",
                        "scale_dtype": "torch.float8_e4m3fn"
                    }
                }
            },
            "ignore": [
                "lm_head",
                "model.visual.blocks.0.attn.qkv",
                "model.language_model.layers.0.linear_attn.out_proj",
                "model.language_model.layers.0.mlp.gate",
                "model.language_model.layers.0.mlp.shared_expert_gate"
            ]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 16);
        assert!(cfg.should_skip_module("lm_head"));
        assert!(cfg.should_skip_module("model.visual.blocks.0.attn.qkv"));
        assert!(cfg.should_skip_module("model.language_model.layers.0.linear_attn.out_proj"));
        assert!(cfg.should_skip_module("model.language_model.layers.0.mlp.gate"));
        assert!(cfg.should_skip_module("model.language_model.layers.0.mlp.shared_expert_gate"));
        assert!(!cfg.should_skip_module("model.language_model.layers.0.mlp.gate_proj"));
        assert!(!cfg.should_skip_module("model.language_model.layers.0.mlp.down_proj"));
    }

    #[test]
    fn test_nvfp4_compressed_tensors_detect_from_groups() {
        // compressed-tensors without top-level format, detected from config_groups
        let json = r#"{
            "quant_method": "compressed-tensors",
            "config_groups": {
                "group_0": {
                    "format": "nvfp4-pack-quantized",
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "group_size": 16
                    }
                }
            }
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert_eq!(cfg.bits, 4);
        assert_eq!(cfg.group_size, 16);
    }

    #[test]
    fn test_mixed_precision_nvfp4_fp8_from_quantized_layers() {
        let json = r#"{
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "config_groups": {
                "group_0": {
                    "input_activations": {"dynamic": false, "num_bits": 8, "type": "float"},
                    "weights": {"dynamic": false, "num_bits": 8, "type": "float"},
                    "targets": ["model.layers.0.linear_attn.in_proj_qkv"]
                },
                "group_1": {
                    "input_activations": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "weights": {"dynamic": false, "num_bits": 4, "type": "float", "group_size": 16},
                    "targets": ["model.layers.0.mlp.experts"]
                }
            },
            "quantized_layers": {
                "model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
                "model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16}
            },
            "ignore": ["mtp*"]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert_eq!(cfg.group_size, 16);
        assert_eq!(cfg.bits, 4);
    }

    #[test]
    fn test_mixed_precision_fp8_only_from_quantized_layers() {
        let json = r#"{
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"},
                "model.layers.0.self_attn.k_proj": {"quant_algo": "FP8"}
            },
            "ignore": ["mtp*"]
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "fp8");
    }

    #[test]
    fn test_modelopt_mixed_alias_normalization() {
        let json = r#"{
            "quant_method": "modelopt_mixed",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.linear_attn.in_proj_qkv": {"quant_algo": "FP8"},
                "model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16}
            }
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "nvfp4");
        assert_eq!(cfg.group_size, 16);
        assert_eq!(cfg.bits, 4);
    }

    #[test]
    fn test_modelopt_fp8_normalization() {
        let json = r#"{
            "quant_method": "modelopt",
            "quant_algo": "FP8"
        }"#;
        let mut cfg: QuantConfig = serde_json::from_str(json).unwrap();
        cfg.normalize_compressed_tensors();
        assert_eq!(cfg.quant_method, "fp8");
    }

    #[test]
    fn test_reasoning_effort_from_str() {
        assert_eq!(
            ReasoningEffort::from_str("none".to_string()),
            ReasoningEffort::None
        );
        assert_eq!(
            ReasoningEffort::from_str("low".to_string()),
            ReasoningEffort::Low
        );
        assert_eq!(
            ReasoningEffort::from_str("medium".to_string()),
            ReasoningEffort::Medium
        );
        assert_eq!(
            ReasoningEffort::from_str("high".to_string()),
            ReasoningEffort::High
        );
        assert_eq!(
            ReasoningEffort::from_str("xhigh".to_string()),
            ReasoningEffort::High
        );
        assert_eq!(
            ReasoningEffort::from_str("chain_of_thought".to_string()),
            ReasoningEffort::ChainOfThought
        );
    }

    #[test]
    fn test_reasoning_effort_is_enabled() {
        assert!(!ReasoningEffort::None.is_enabled());
        assert!(ReasoningEffort::Low.is_enabled());
        assert!(ReasoningEffort::Medium.is_enabled());
        assert!(ReasoningEffort::High.is_enabled());
        assert!(ReasoningEffort::ChainOfThought.is_enabled());
    }

    #[test]
    fn test_reasoning_effort_chat_template_value() {
        assert_eq!(
            ReasoningEffort::from_str("xhigh".to_string()).chat_template_value(),
            Some("xhigh".to_string())
        );
        assert_eq!(
            ReasoningEffort::High.chat_template_value(),
            Some("xhigh".to_string())
        );
        assert_eq!(ReasoningEffort::ModelDefault.chat_template_value(), None);
    }

    // ---- 阶段二线 A:真 config.json(Qwen3.5-0.8B)反序列化 + text_config 解壳 ----

    /// 真模型 config(本地工作区;缺失时跳过——CI 无此文件)。
    const REAL_CONFIG: &str = "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B/config.json";

    #[test]
    fn from_json_str_unwraps_text_config_qwen35_08b() {
        if !std::path::Path::new(REAL_CONFIG).exists() {
            eprintln!("skip: 真模型 config 不在本机({REAL_CONFIG})");
            return;
        }
        let json = std::fs::read_to_string(REAL_CONFIG).unwrap();
        let cfg = Config::from_json_str(&json).expect("text_config 解壳后反序列化");

        // 层型分布:18 linear + 6 full,full 恒在 3,7,11,15,19,23
        let lt = cfg.layer_types.as_ref().expect("layer_types 存在");
        assert_eq!(lt.len(), 24);
        assert_eq!(lt.iter().filter(|t| t.as_str() == "linear_attention").count(), 18);
        assert_eq!(lt.iter().filter(|t| t.as_str() == "full_attention").count(), 6);
        for &i in &[3usize, 7, 11, 15, 19, 23] {
            assert_eq!(lt[i], "full_attention", "第 {i} 层应为 full_attention");
        }

        // GDN 线性层参数(config.json 原文核对)
        assert_eq!(cfg.linear_num_value_heads, Some(16));
        assert_eq!(cfg.linear_num_key_heads, Some(16));
        assert_eq!(cfg.linear_key_head_dim, Some(128));
        assert_eq!(cfg.linear_value_head_dim, Some(128));
        assert_eq!(cfg.linear_conv_kernel_dim, Some(4));

        // full-attention 侧:num_attention_heads=8 × head_dim 256
        // (q_proj [4096,1024] = 8×256×2,×2 来自 attn_output_gate 门控,非 16 头)
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.head_dim, Some(256));
        assert_eq!(cfg.num_key_value_heads, 2);
        assert_eq!(cfg.attn_output_gate, Some(true));
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.intermediate_size, 3584);
        assert_eq!(cfg.vocab_size, Some(248320));
        assert_eq!(cfg.max_position_embeddings, 262144);
        assert_eq!(cfg.rms_norm_eps, 1e-6);
        assert_eq!(cfg.tie_word_embeddings, Some(true));
        assert_eq!(cfg.mtp_num_hidden_layers, Some(1));

        // rope 参数提升:rope_parameters.{rope_theta,partial_rotary_factor} → 顶层
        assert_eq!(cfg.rope_theta, Some(10_000_000.0));
        assert_eq!(cfg.partial_rotary_factor, Some(0.25));
        // rope_scaling(alias rope_parameters)已接住 mrope 面与 rope_type
        let rs = cfg.rope_scaling.as_ref().expect("rope_scaling 存在");
        assert_eq!(rs.get("rope_type").and_then(|v| v.as_str()), Some("default"));
        assert!(matches!(
            rs.get("mrope_interleaved"),
            Some(crate::config::RopeScalingValue::Bool(true))
        ));
    }

    #[test]
    fn from_json_str_flat_config_still_works() {
        // 纯文本 config(无 text_config)原样反序列化(旧路径兼容)
        let cfg = Config::from_json_str(
            r#"{"architectures":["Qwen3ForCausalLM"],"num_attention_heads":4,
                "num_key_value_heads":2,"max_position_embeddings":40960,
                "hidden_size":256,"num_hidden_layers":2,"intermediate_size":512,
                "rms_norm_eps":1e-6,"vocab_size":1000,"hidden_act":"silu",
                "rope_theta":1000000.0}"#,
        )
        .expect("扁平 config 直接反序列化");
        assert_eq!(cfg.num_attention_heads, 4);
        assert!(cfg.layer_types.is_none(), "无 layer_types 字段时应为 None");
    }
}
