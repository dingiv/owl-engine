//! Qwen3 hybrid / DeepSeek V4 / Gemma4 逐层配置解析(T2-三 搬运)。
//!
//! port 出处:packages/xinfer/crates/core/src/utils/mod.rs
//! (Qwen3HybridRawConfig / Qwen3HybridConfig / 解析函数群)。
//! 纯配置面逻辑,零 candle 触点——原样搬运,仅路径改指。

use crate::config::Config;

#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct Qwen3HybridRawConfig {
    #[serde(alias = "layer_types")]
    pub layers_block_type: Option<Vec<String>>,
    #[serde(alias = "linear_conv_kernel_dim")]
    pub conv_kernel_size: Option<usize>,
    pub full_attention_interval: Option<usize>,
    pub linear_num_heads: Option<usize>,
    #[serde(alias = "linear_num_key_heads")]
    pub linear_num_key_heads: Option<usize>,
    #[serde(alias = "linear_num_value_heads")]
    pub linear_num_value_heads: Option<usize>,
    pub linear_num_key_value_heads: Option<usize>,
    pub linear_key_head_dim: Option<usize>,
    pub linear_value_head_dim: Option<usize>,
    pub mamba_ssm_dtype: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Qwen3HybridConfig {
    pub layer_types: Vec<String>,
    pub conv_kernel_size: usize,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub mamba_ssm_dtype: Option<String>,
}

pub fn is_qwen3_hybrid_arch_name(arch: &str) -> bool {
    matches!(
        arch,
        "Qwen3_5ForCausalLM"
            | "Qwen3_5MoeForCausalLM"
            | "Qwen3NextForCausalLM"
            | "Qwen3_5ForConditionalGeneration"
            | "Qwen3_5MoeForConditionalGeneration"
            | "Qwen3NextForConditionalGeneration"
            | "Qwen4ExpForConditionalGeneration"
            | "Qwen4ExpForCausalLM"
    )
}

pub fn is_deepseek_v4_arch_name(arch: &str) -> bool {
    matches!(arch, "DeepseekV4ForCausalLM" | "deepseek_v4" | "deepseek4")
}

fn is_qwen3_hybrid_arch(config: &Config) -> bool {
    let arch = config.architectures.as_ref().and_then(|a| a.first());
    arch.map(|a| is_qwen3_hybrid_arch_name(a)).unwrap_or(false)
}

fn qwen3_hybrid_raw_from_extra_config(config: &Config) -> Option<Qwen3HybridRawConfig> {
    if !is_qwen3_hybrid_arch(config) {
        return None;
    }
    let extra = config.extra_config_json.as_ref()?;
    let root = serde_json::from_str::<serde_json::Value>(extra).ok()?;
    let cfg = root.get("text_config").cloned().unwrap_or(root);
    serde_json::from_value::<Qwen3HybridRawConfig>(cfg).ok()
}

pub fn resolve_qwen3_hybrid_config(config: &Config) -> Qwen3HybridConfig {
    let raw_cfg = qwen3_hybrid_raw_from_extra_config(config).unwrap_or_default();

    let mut layer_types = if let Some(layer_types) = raw_cfg.layers_block_type {
        layer_types
    } else if let Some(interval) = raw_cfg.full_attention_interval {
        if interval > 0 {
            (0..config.num_hidden_layers)
                .map(|idx| {
                    if (idx + 1) % interval == 0 {
                        "full_attention".to_string()
                    } else {
                        "linear_attention".to_string()
                    }
                })
                .collect::<Vec<_>>()
        } else {
            vec!["full_attention".to_string(); config.num_hidden_layers]
        }
    } else {
        vec!["full_attention".to_string(); config.num_hidden_layers]
    };

    for layer_type in layer_types.iter_mut() {
        if layer_type == "attention" {
            *layer_type = "full_attention".to_string();
        }
    }
    if layer_types.len() != config.num_hidden_layers {
        crate::log_warn!(
            "Qwen3 hybrid layer_types length {} != num_hidden_layers {}, fallback to full_attention.",
            layer_types.len(),
            config.num_hidden_layers
        );
        layer_types = vec!["full_attention".to_string(); config.num_hidden_layers];
    }

    let num_v_heads = raw_cfg
        .linear_num_value_heads
        .or(raw_cfg.linear_num_heads)
        .unwrap_or(config.num_attention_heads);
    let num_k_heads = raw_cfg
        .linear_num_key_heads
        .or(raw_cfg.linear_num_key_value_heads)
        .unwrap_or(num_v_heads);
    let key_head_dim = raw_cfg.linear_key_head_dim.unwrap_or(
        config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads),
    );
    let value_head_dim = raw_cfg.linear_value_head_dim.unwrap_or(key_head_dim);
    let conv_kernel_size = raw_cfg.conv_kernel_size.unwrap_or(4);

    Qwen3HybridConfig {
        layer_types,
        conv_kernel_size,
        num_v_heads,
        num_k_heads,
        key_head_dim,
        value_head_dim,
        mamba_ssm_dtype: raw_cfg.mamba_ssm_dtype,
    }
}

pub fn qwen3_hybrid_layer_types(config: &Config) -> Option<Vec<String>> {
    if !is_qwen3_hybrid_arch(config) {
        return None;
    }
    Some(resolve_qwen3_hybrid_config(config).layer_types)
}

/// Gemma4 异构 head_dim(SWA=head_dim,full_attention=global_head_dim)模型的
/// 逐层 (num_kv_heads, head_dim) KV cache 配置。
pub fn gemma4_per_layer_cache_config(config: &Config) -> Option<Vec<(usize, usize)>> {
    let arch = config.architectures.as_ref()?.first()?;
    if !arch.contains("Gemma4") {
        return None;
    }
    let extra = config.extra_config_json.as_ref()?;
    let root: serde_json::Value = serde_json::from_str(extra).ok()?;
    let cfg = root.get("text_config").unwrap_or(&root);

    let get = |key: &str| -> Option<&serde_json::Value> { cfg.get(key).or_else(|| root.get(key)) };

    let layer_types: Vec<String> =
        get("layer_types").and_then(|value| serde_json::from_value(value.clone()).ok())?;
    if layer_types.len() != config.num_hidden_layers {
        crate::log_warn!(
            "Gemma4 layer_types length {} != num_hidden_layers {}; ignoring heterogeneous KV cache config.",
            layer_types.len(),
            config.num_hidden_layers
        );
        return None;
    }

    let swa_head_dim = get("swa_head_dim")
        .or_else(|| cfg.get("head_dim"))
        .or_else(|| root.get("head_dim"))
        .and_then(|v| v.as_u64())? as usize;
    let global_head_dim = get("global_head_dim").and_then(|v| v.as_u64())? as usize;
    let swa_kv_heads = get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(config.num_key_value_heads as u64) as usize;
    let global_kv_heads = get("num_global_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(swa_kv_heads as u64) as usize;

    let per_layer = layer_types
        .iter()
        .map(|lt| {
            if lt == "full_attention" {
                (global_kv_heads, global_head_dim)
            } else {
                (swa_kv_heads, swa_head_dim)
            }
        })
        .collect();
    Some(per_layer)
}
