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

/// A2 后 Config 已直解 hybrid 字段(from_json_str text_config 解壳归一):
/// config 字段优先,extra_config_json 作旧路径回退(xinfer 语义)。
fn qwen3_hybrid_raw_from_config(config: &Config) -> Qwen3HybridRawConfig {
    Qwen3HybridRawConfig {
        layers_block_type: config.layer_types.clone(),
        conv_kernel_size: config.linear_conv_kernel_dim,
        full_attention_interval: config.full_attention_interval,
        linear_num_heads: None,
        linear_num_key_heads: config.linear_num_key_heads,
        linear_num_value_heads: config.linear_num_value_heads,
        linear_num_key_value_heads: None,
        linear_key_head_dim: config.linear_key_head_dim,
        linear_value_head_dim: config.linear_value_head_dim,
        mamba_ssm_dtype: None,
    }
}

pub fn resolve_qwen3_hybrid_config(config: &Config) -> Qwen3HybridConfig {
    // 逐字段优先级:Config 直解字段 > extra_config_json(旧路径)> 缺省回退
    let mut raw_cfg = qwen3_hybrid_raw_from_config(config);
    if let Some(extra) = qwen3_hybrid_raw_from_extra_config(config) {
        if raw_cfg.layers_block_type.is_none() {
            raw_cfg.layers_block_type = extra.layers_block_type;
        }
        if raw_cfg.conv_kernel_size.is_none() {
            raw_cfg.conv_kernel_size = extra.conv_kernel_size;
        }
        if raw_cfg.full_attention_interval.is_none() {
            raw_cfg.full_attention_interval = extra.full_attention_interval;
        }
        if raw_cfg.linear_num_heads.is_none() {
            raw_cfg.linear_num_heads = extra.linear_num_heads;
        }
        if raw_cfg.linear_num_key_heads.is_none() {
            raw_cfg.linear_num_key_heads = extra.linear_num_key_heads;
        }
        if raw_cfg.linear_num_value_heads.is_none() {
            raw_cfg.linear_num_value_heads = extra.linear_num_value_heads;
        }
        if raw_cfg.linear_num_key_value_heads.is_none() {
            raw_cfg.linear_num_key_value_heads = extra.linear_num_key_value_heads;
        }
        if raw_cfg.linear_key_head_dim.is_none() {
            raw_cfg.linear_key_head_dim = extra.linear_key_head_dim;
        }
        if raw_cfg.linear_value_head_dim.is_none() {
            raw_cfg.linear_value_head_dim = extra.linear_value_head_dim;
        }
        if raw_cfg.mamba_ssm_dtype.is_none() {
            raw_cfg.mamba_ssm_dtype = extra.mamba_ssm_dtype;
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_CONFIG: &str = "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B/config.json";

    /// 最小必需字段骨架(required 字段:head 数/hidden/intermediate/eps/hidden_act)
    fn cfg_json(body: &str) -> Config {
        let json = format!(
            r#"{{
                "architectures": ["Qwen3_5ForCausalLM"],
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "max_position_embeddings": 4096,
                "hidden_size": 1024,
                "num_hidden_layers": 4,
                "intermediate_size": 256,
                "rms_norm_eps": 1e-6,
                "hidden_act": "silu"
                {body}
            }}"#
        );
        Config::from_json_str(&json).expect("测试 config 反序列化")
    }

    #[test]
    fn resolve_prefers_config_fields_real_qwen35_08b() {
        if !std::path::Path::new(REAL_CONFIG).exists() {
            eprintln!("skip: 真模型 config 不在本机({REAL_CONFIG})");
            return;
        }
        let json = std::fs::read_to_string(REAL_CONFIG).unwrap();
        let cfg = Config::from_json_str(&json).unwrap();
        let h = resolve_qwen3_hybrid_config(&cfg);
        assert_eq!(h.layer_types.len(), 24);
        assert_eq!(
            h.layer_types.iter().filter(|t| t.as_str() == "linear_attention").count(),
            18
        );
        assert_eq!(
            h.layer_types.iter().filter(|t| t.as_str() == "full_attention").count(),
            6
        );
        for &i in &[3usize, 7, 11, 15, 19, 23] {
            assert_eq!(h.layer_types[i], "full_attention", "第 {i} 层应为 full_attention");
        }
        assert_eq!(h.conv_kernel_size, 4);
        assert_eq!(h.num_v_heads, 16);
        assert_eq!(h.num_k_heads, 16);
        assert_eq!(h.key_head_dim, 128);
        assert_eq!(h.value_head_dim, 128);
    }

    #[test]
    fn resolve_config_layer_types_wins_over_extra_config_json() {
        let cfg = cfg_json(
            r#", "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"],
            "linear_num_value_heads": 16,
            "extra_config_json": "{\"layer_types\": [\"full_attention\", \"full_attention\", \"full_attention\", \"full_attention\"], \"linear_num_value_heads\": 8}""#,
        );
        let h = resolve_qwen3_hybrid_config(&cfg);
        assert_eq!(
            h.layer_types,
            vec![
                "linear_attention".to_string(),
                "full_attention".to_string(),
                "linear_attention".to_string(),
                "full_attention".to_string(),
            ]
        );
        assert_eq!(h.num_v_heads, 16, "config 字段应胜过 extra_config_json");
    }

    #[test]
    fn resolve_extra_config_json_path_still_works_when_config_fields_absent() {
        let cfg = cfg_json(
            r#", "extra_config_json": "{\"layer_types\": [\"linear_attention\", \"linear_attention\", \"linear_attention\", \"full_attention\"], \"linear_num_value_heads\": 8, \"linear_key_head_dim\": 96}""#,
        );
        let h = resolve_qwen3_hybrid_config(&cfg);
        assert_eq!(
            h.layer_types,
            vec![
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "linear_attention".to_string(),
                "full_attention".to_string(),
            ]
        );
        assert_eq!(h.num_v_heads, 8);
        assert_eq!(h.key_head_dim, 96);
    }

    #[test]
    fn resolve_interval_fallback_from_config_field() {
        let cfg = cfg_json(r#", "full_attention_interval": 2"#);
        let h = resolve_qwen3_hybrid_config(&cfg);
        assert_eq!(
            h.layer_types,
            vec![
                "linear_attention".to_string(),
                "full_attention".to_string(),
                "linear_attention".to_string(),
                "full_attention".to_string(),
            ]
        );
    }

    #[test]
    fn resolve_zero_gdn_fallback_unchanged() {
        // 无 layer_types/interval/extra:全 full_attention(纯注意力旧模型行为不变)
        let cfg = cfg_json("");
        let h = resolve_qwen3_hybrid_config(&cfg);
        assert_eq!(h.layer_types, vec!["full_attention".to_string(); 4]);
        assert_eq!(h.num_v_heads, 8, "缺省 = num_attention_heads");
        assert_eq!(h.key_head_dim, 128, "缺省 = hidden_size/num_attention_heads");
    }
}
