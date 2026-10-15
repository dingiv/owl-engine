//! **维度事实的起点读取**(`ModelSpec::from_ckpt`):config.json 优先 +
//! 张量形状兜底(解释器化设计 §2.3;用户裁决 2026-10-16)。
//!
//! 规矩(与 `formats::kind` 同族):
//! - **config.json 优先**,逐项与索引实测形状**复核**;不符 ⇒ **以形状为准**
//!   并记入 [`SpecReport::mismatches`](响亮:warn 级,不静默);
//! - **键基座由索引实测定位**(哪个基座命中 `embed_tokens` 就用哪个),
//!   不写死在代码里(纯文本仓 `model` / 多模态仓 `model.language_model`);
//! - 层型表(`layer_types`)以**索引键面**为准(有 `self_attn.*` = full,
//!   有 `linear_attn.*` = GDN)—— config 只作复核;
//! - tokenizer/chat 文本**不从维度表推**(模型族事实,调用方给;
//!   [`SpecSource::tokenizer`]),dtype 同(计算档事实)。

use crate::contract::{Dtype, ModelError};
use crate::model::ModelSpec;
use crate::tokenizer::TokenizerSpec;
use std::collections::HashMap;
use std::path::Path;

/// 起点侧"非维度"事实(调用方给:计算 dtype + tokenizer 声明)
pub struct SpecSource<'a> {
    pub dtype: Dtype,
    pub tokenizer: &'a TokenizerSpec,
}

/// 读取账(键基座 + config↔形状 失配清单;空 = 完全一致)
#[derive(Debug, Default, Clone)]
pub struct SpecReport {
    /// 实测键基座(`model.language_model` / `model` / …;含尾点)
    pub keys_base: String,
    /// 失配条目(`字段: config=X 形状=Y(采形状)`)
    pub mismatches: Vec<String>,
}

impl ModelSpec {
    /// 从检查点目录读维度事实(config.json 优先 + 形状兜底)。
    ///
    /// 终点侧(`Loadable::layout`)要什么形状由模型文件声明;本函数只负责
    /// **起点实测**,两者不符由装载器响亮拒(缺键/形状错归因)。
    pub fn from_ckpt(dir: &Path, src: SpecSource<'_>) -> Result<(ModelSpec, SpecReport), ModelError> {
        let cfg = owl_shared::file_loader::read(&dir.join("config.json"))
            .map_err(|e| ModelError::Msg(format!("from_ckpt: 读 {}/config.json: {e}", dir.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&cfg)
            .map_err(|e| ModelError::Msg(format!("from_ckpt: config.json 解析: {e}")))?;
        let text = v.get("text_config").unwrap_or(&v);
        let (_, raw) = crate::formats::mmap::open_raw_index(dir)?;
        let idx: HashMap<String, crate::formats::mmap::RawEntry> = raw;

        // ---- 键基座:哪个前缀命中 embed_tokens(实测,禁写死)----
        let base = idx
            .keys()
            .find_map(|k| k.strip_suffix("embed_tokens.weight").map(|b| b.to_string()))
            .ok_or_else(|| ModelError::Msg("from_ckpt: 索引里找不到 *.embed_tokens.weight(键基座无据)".into()))?;
        let mut rep = SpecReport { keys_base: base.clone(), mismatches: Vec::new() };

        // ---- 形状探针(指数级:`.weight` 直取;量化族 `.weight_packed` 配 `.weight_shape`)----
        let shape_of = |key: &str| -> Option<Vec<usize>> {
            if let Some(e) = idx.get(key) {
                return Some(e.shape.clone());
            }
            let shape_key = format!("{}_shape", key.strip_suffix(".weight")?);
            let mut s = idx.get(&shape_key)?.shape.clone();
            // weight_shape 语义 = [out, in] 或 [in, out](按条目判读);统一成 [rows, cols]
            if s.len() == 2 && s[0] < s[1] {
                s.swap(0, 1);
            }
            Some(s)
        };
        let first_layer = |suffix: &str| -> Option<Vec<usize>> {
            (0..256).find_map(|i| shape_of(&format!("{base}layers.{i}.{suffix}")))
        };
        let has = |suffix: &str| idx.keys().any(|k| k.ends_with(suffix));

        // ---- 层型表:索引键面为准 ----
        let mut layer_types: Vec<bool> = Vec::new();
        for i in 0..256 {
            let full = has(&format!("{base}layers.{i}.self_attn.q_proj.weight"))
                || has(&format!("{base}layers.{i}.self_attn.q_proj.weight_packed"));
            let gdn = has(&format!("{base}layers.{i}.linear_attn.in_proj_qkv.weight"))
                || has(&format!("{base}layers.{i}.linear_attn.in_proj_qkvz.weight"))
                || has(&format!("{base}layers.{i}.linear_attn.in_proj_qkv.weight_packed"))
                || has(&format!("{base}layers.{i}.linear_attn.in_proj_qkvz.weight_packed"));
            match (full, gdn) {
                (true, false) => layer_types.push(true),
                (false, true) => layer_types.push(false),
                (false, false) => break, // 连续层号断点 = 层数到顶
                (true, true) => {
                    return Err(ModelError::Msg(format!(
                        "from_ckpt: 层 {i} 同时带 self_attn 与 linear_attn 键(层型歧义)"
                    )))
                }
            }
        }
        let n_layers = layer_types.len();
        if n_layers == 0 {
            return Err(ModelError::Msg("from_ckpt: 索引里数不出任何解码层".into()));
        }
        let g = |k: &str| -> Option<usize> { text.get(k).and_then(|x| x.as_u64()).map(|x| x as usize) };
        let gf = |k: &str| -> Option<f32> { text.get(k).and_then(|x| x.as_f64()).map(|x| x as f32) };

        // 复核小器:config 值 vs 形状值(不符 ⇒ 采形状 + 记账)
        let mut pick = |field: &str, cfg_v: Option<usize>, shape_v: Option<usize>| -> Result<usize, ModelError> {
            match (cfg_v, shape_v) {
                (Some(c), Some(s)) if c != s => {
                    rep.mismatches.push(format!("{field}: config={c} 形状={s}(采形状)"));
                    Ok(s)
                }
                (Some(c), _) => Ok(c),
                (None, Some(s)) => Ok(s),
                (None, None) => Err(ModelError::Msg(format!("from_ckpt: {field} 既无 config 也无形状"))),
            }
        };

        let emb = shape_of(&format!("{base}embed_tokens.weight"));
        let vocab = pick("vocab_size", g("vocab_size"), emb.as_ref().map(|s| s[0]))?;
        let hidden = pick("hidden_size", g("hidden_size"), emb.as_ref().map(|s| s[1]))?;
        let inter = pick(
            "intermediate_size",
            g("intermediate_size"),
            first_layer("mlp.gate_proj.weight").map(|s| s[0]),
        )?;
        let hd = pick(
            "head_dim",
            g("head_dim"),
            first_layer("self_attn.q_norm.weight").map(|s| s[0]),
        )?;
        let hkv = match first_layer("self_attn.k_proj.weight") {
            Some(s) => pick("num_key_value_heads", g("num_key_value_heads"), Some(s[0] / hd))?,
            None => g("num_key_value_heads").ok_or_else(|| ModelError::Msg("from_ckpt: num_key_value_heads 无据".into()))?,
        };
        let gate_mul = text.get("attn_output_gate").and_then(|x| x.as_bool()).unwrap_or(false) as usize + 1;
        let hq = match first_layer("self_attn.q_proj.weight") {
            Some(s) => pick("num_attention_heads", g("num_attention_heads"), Some(s[0] / hd / gate_mul))?,
            None => g("num_attention_heads").ok_or_else(|| ModelError::Msg("from_ckpt: num_attention_heads 无据".into()))?,
        };
        // GDN 头:config 给头数+头维;索引只查**乘积**是否一致
        // (头维是 GDN 自己的 hk_dim,与 FA 的 hd 无关 —— 别拿 hd 去除)
        let nk = g("linear_num_key_heads")
            .ok_or_else(|| ModelError::Msg("from_ckpt: linear_num_key_heads 无据".into()))?;
        let nv = g("linear_num_value_heads")
            .ok_or_else(|| ModelError::Msg("from_ckpt: linear_num_value_heads 无据".into()))?;
        let mut gdn_pair = |field: &str, cfg_dim: Option<usize>, prod: Option<usize>, heads: usize| -> Result<usize, ModelError> {
            let cfg_dim = cfg_dim.ok_or_else(|| ModelError::Msg(format!("from_ckpt: {field} 无据")))?;
            let Some(p) = prod else { return Ok(cfg_dim) };
            if p == heads * cfg_dim {
                return Ok(cfg_dim);
            }
            rep.mismatches.push(format!(
                "{field}: config={cfg_dim}(×{heads}={}) 形状总宽={p}(采形状)",
                heads * cfg_dim
            ));
            if p % heads == 0 {
                Ok(p / heads)
            } else {
                rep.mismatches.push(format!("{field}: 形状总宽 {p} 不被头数 {heads} 整除(采 config)"));
                Ok(cfg_dim)
            }
        };
        // in_proj_qkv 行数 = 2·nk·hk_dim + nv·hv_dim(**三段融合:q|k|v**);
        // in_proj_z 行数 = nv·hv_dim(门)。故 v 段先定,再拿余量核 qk 段。
        let hv_dim = gdn_pair(
            "linear_value_head_dim",
            g("linear_value_head_dim"),
            first_layer("linear_attn.in_proj_z.weight").map(|s| s[0]),
            nv,
        )?;
        let hk_dim = gdn_pair(
            "linear_key_head_dim",
            g("linear_key_head_dim"),
            first_layer("linear_attn.in_proj_qkv.weight")
                .map(|s| s[0].saturating_sub(nv * hv_dim) / 2),
            nk,
        )?;
        // tied:config 为准;索引见 lm_head 而 config 说 tied ⇒ 记账(config 胜,
        // 与装载面同律:tied 复 embed 表,lm_head 键不装载)
        let tied = v
            .get("tie_word_embeddings")
            .and_then(|x| x.as_bool())
            .or_else(|| text.get("tie_word_embeddings").and_then(|x| x.as_bool()))
            .unwrap_or(false);
        if tied && has("lm_head.weight") {
            rep.mismatches.push("tie_word_embeddings: config=true 但索引有 lm_head.weight(采 config)".into());
        }
        let eps = gf("rms_norm_eps").or_else(|| gf("layer_norm_eps")).unwrap_or(1e-6);

        let spec = ModelSpec {
            vocab,
            hidden,
            inter,
            dtype: src.dtype,
            full_heads: (hq, hkv, hd),
            gdn_heads: (nk, hk_dim, nv, hv_dim),
            eps,
            layer_types,
            tokenizer: src.tokenizer.clone(),
            tied,
        };
        Ok((spec, rep))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(env: &str, default: &str) -> Option<std::path::PathBuf> {
        let p = std::path::PathBuf::from(std::env::var(env).unwrap_or_else(|_| default.to_string()));
        p.is_dir().then_some(p)
    }

    /// 对拍器:逐字段(数值 + 层型表),tokenizer/dtype 由调用方给不一致
    fn assert_same(got: &ModelSpec, want: &ModelSpec, tag: &str) {
        assert_eq!(got.vocab, want.vocab, "[{tag}] vocab");
        assert_eq!(got.hidden, want.hidden, "[{tag}] hidden");
        assert_eq!(got.inter, want.inter, "[{tag}] inter");
        assert_eq!(got.full_heads, want.full_heads, "[{tag}] full_heads");
        assert_eq!(got.gdn_heads, want.gdn_heads, "[{tag}] gdn_heads");
        assert_eq!((got.eps - want.eps).abs() < 1e-9, true, "[{tag}] eps");
        assert_eq!(got.layer_types, want.layer_types, "[{tag}] layer_types");
        assert_eq!(got.tied, want.tied, "[{tag}] tied");
    }

    /// 手写常量 == from_ckpt(0.8B / 27B AWQ 两档真检查点)
    #[test]
    fn matches_handwritten_specs() {
        for (env, default, want_fn, tag) in [
            (
                "OWL_CKPT_0_8B",
                "/home/div/Documents/codes/models/Qwen/Qwen3.5-0.8B",
                crate::specs::qwen35::qwen3_5_0_8b as fn() -> ModelSpec,
                "0.8B",
            ),
            (
                "OWL_CKPT_27B_AWQ",
                "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4",
                crate::specs::qwen35::qwen3_8_27b as fn() -> ModelSpec,
                "27B-AWQ",
            ),
        ] {
            let Some(d) = dir(env, default) else {
                eprintln!("skip: {default} 不在({env} 可指路)");
                continue;
            };
            let want = want_fn();
            let (got, rep) = ModelSpec::from_ckpt(&d, SpecSource { dtype: want.dtype, tokenizer: &want.tokenizer })
                .unwrap_or_else(|e| panic!("[{tag}] from_ckpt: {e}"));
            eprintln!("[{tag}] keys_base = {:?};失配 {} 条", rep.keys_base, rep.mismatches.len());
            for m in &rep.mismatches {
                eprintln!("[{tag}]   · {m}");
            }
            assert_same(&got, &want, tag);
        }
    }

    /// 4B-VL:from_ckpt 的文本 spec == `VlmCfg::from_json` 的 text_spec(同一份 config 两条读法)
    #[test]
    fn matches_vlm_text_spec() {
        let Some(d) = dir("OWL_CKPT_4B_VL", "/home/div/Documents/codes/models/Qwen/Qwen3.5-4B") else {
            eprintln!("skip: 4B-VL 检查点不在");
            return;
        };
        let raw = owl_shared::file_loader::read(&d.join("config.json")).expect("config.json");
        let cfg = crate::vlm::VlmCfg::from_json(&raw).expect("VlmCfg::from_json");
        let (got, rep) = ModelSpec::from_ckpt(&d, SpecSource {
            dtype: cfg.text_spec.dtype,
            tokenizer: &cfg.text_spec.tokenizer,
        })
        .expect("from_ckpt(4B)");
        eprintln!("[4B-VL] keys_base = {:?};失配 {} 条", rep.keys_base, rep.mismatches.len());
        assert!(rep.mismatches.is_empty(), "4B 应零失配: {:?}", rep.mismatches);
        assert_eq!(rep.keys_base, "model.language_model.");
        assert_same(&got, &cfg.text_spec, "4B-VL");
    }
}
