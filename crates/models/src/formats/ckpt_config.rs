//! **检查点 `config.json` 的类型化读取口**(2026-10-19;用户裁决立案)。
//!
//! # 为什么单独一个结构体
//!
//! 用户裁决(2026-10-19)三条,本条对应第 ①②:
//! > ① 给 loader 提供足够的参数 —— 从**配置文件**来,**不要 env**;
//! > ② 那个 `config.json` 里面**本来就有布局信息**,不许写死 —— 把它**读进来**。
//!
//! 装载面此前对"同名键的语义"是**键名启发式**(`plan::rules::rule_of` 按键尾猜):
//! `q_proj` 一律当**门控**的(Qwen3.5 主干 out = q ⊕ gate)。而 DFlash2 草稿的
//! `q_proj` **无门控**(out = hq·hd)⇒ 双 rank 装载在强校验处拒 / 设备重排臂自旋
//! (P2.1-a 立案)。**裁决依据本就在检查点里**:
//!
//! ```text
//! architectures: ["DFlash2DraftModel"]   ← 草稿 vs 主干的**声明事实**
//! num_attention_heads / head_dim / …     ← 形状事实(out 是否 = 2·hq·hd)
//! quantization_config.config_groups…     ← group_size / symmetric / num_bits / actorder
//! dflash_config                          ← block_size / selector_rank / target_layer_ids
//! ```
//!
//! 本模块只做一件事:**把这份文件读成类型化的结构**(零启发式、零默认猜测:
//! 缺字段 = `None`,由消费方决定"缺了要不要拒")。它**不改任何现有行为** ——
//! 现有 `ModelSpec::from_ckpt`(维度)与 `kind::quant_family`(量化族)继续走各自
//! 的读法;本结构是**布局事实**的收口面,后续由装载面按它裁决。
//!
//! # 消费方向(施工清单见路线图 §3「装载面参数化」)
//!
//! 1. `is_draft()` ⇒ `q_proj` 无门控(替代键名启发式);
//! 2. `pack_shift()` ⇒ 打包字宽(`num_bits = 4` ⇒ ÷8;非 4 **响亮拒**);
//! 3. `quant.actorder/block_structure` ⇒ 将来 k-major 轴选择(现为键名启发式);
//! 4. `[load]` 段(配置面)与 `LoadOpts` 的参数由**同一处**收口(见路线图)。

use std::path::Path;

use owl_iface::contract::ModelError;

/// `quantization_config.config_groups.<group>.weights`(vLLM/compressed-tensors 形态)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct QuantFacts {
    pub format: Option<String>,
    /// 分组尺度宽(`None` = 未声明)
    pub group_size: Option<usize>,
    pub num_bits: Option<usize>,
    pub symmetric: Option<bool>,
    /// `"group"` / `"channel"` / `"tensor"`(compressed-tensors 的 strategy)
    pub strategy: Option<String>,
    /// 重排线索(`actorder` / `block_structure`):k-major 轴选择的声明依据
    pub actorder: Option<String>,
    pub block_structure: Option<String>,
}

/// `dflash_config`(DFlash2 草稿家族自己的档)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DflashFacts {
    pub block_size: Option<usize>,
    pub conv_group_size: Option<usize>,
    pub conv_kernel_size: Option<usize>,
    pub selector_rank: Option<usize>,
    pub selector_top_k: Option<usize>,
    pub target_layer_ids: Vec<usize>,
}

/// 检查点 `config.json`(类型化;`text_config` 嵌套原样保留 —— VLM 形态)
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CkptConfig {
    pub architectures: Vec<String>,
    pub model_type: Option<String>,
    pub hidden_size: Option<usize>,
    pub num_hidden_layers: Option<usize>,
    pub num_attention_heads: Option<usize>,
    pub num_key_value_heads: Option<usize>,
    pub head_dim: Option<usize>,
    pub intermediate_size: Option<usize>,
    pub vocab_size: Option<usize>,
    pub tie_word_embeddings: Option<bool>,
    pub quant: Option<QuantFacts>,
    pub dflash: Option<DflashFacts>,
    /// VLM 形态(主模型 cyankiwi 的 `text_config`):同构嵌套
    pub text_config: Option<Box<CkptConfig>>,
}

fn u(v: &serde_json::Value, k: &str) -> Option<usize> {
    v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize)
}

fn s(v: &serde_json::Value, k: &str) -> Option<String> {
    v.get(k).and_then(|x| x.as_str()).map(|x| x.to_string())
}

impl CkptConfig {
    /// 从 JSON 值解析(纯函数;零 I/O —— 便于单测)
    pub fn from_json(v: &serde_json::Value) -> CkptConfig {
        let quant = v.get("quantization_config").and_then(|q| {
            let g = q
                .get("config_groups")
                .and_then(|c| c.as_object())
                .and_then(|o| o.values().next())?;
            let w = g.get("weights").unwrap_or(g);
            Some(QuantFacts {
                format: s(g, "format"),
                group_size: u(w, "group_size"),
                num_bits: u(w, "num_bits"),
                symmetric: w.get("symmetric").and_then(|x| x.as_bool()),
                strategy: s(w, "strategy"),
                actorder: w
                    .get("actorder")
                    .and_then(|x| x.as_str())
                    .map(|x| x.to_string()),
                block_structure: w
                    .get("block_structure")
                    .and_then(|x| x.as_str())
                    .map(|x| x.to_string()),
            })
        });
        let dflash = v.get("dflash_config").map(|d| DflashFacts {
            block_size: u(d, "block_size"),
            conv_group_size: u(d, "conv_group_size"),
            conv_kernel_size: u(d, "conv_kernel_size"),
            selector_rank: u(d, "selector_rank"),
            selector_top_k: u(d, "selector_top_k"),
            target_layer_ids: d
                .get("target_layer_ids")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as usize).collect())
                .unwrap_or_default(),
        });
        CkptConfig {
            architectures: v
                .get("architectures")
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_str()).map(|x| x.to_string()).collect())
                .unwrap_or_default(),
            model_type: s(v, "model_type"),
            hidden_size: u(v, "hidden_size"),
            num_hidden_layers: u(v, "num_hidden_layers"),
            num_attention_heads: u(v, "num_attention_heads"),
            num_key_value_heads: u(v, "num_key_value_heads"),
            head_dim: u(v, "head_dim"),
            intermediate_size: u(v, "intermediate_size"),
            vocab_size: u(v, "vocab_size"),
            tie_word_embeddings: v.get("tie_word_embeddings").and_then(|x| x.as_bool()),
            quant,
            dflash,
            text_config: v
                .get("text_config")
                .map(|t| Box::new(CkptConfig::from_json(t))),
        }
    }

    /// 读检查点目录的 `config.json`(唯一 I/O 入口;缺失/坏 JSON ⇒ 结构化错误)
    pub fn read(dir: &Path) -> Result<CkptConfig, ModelError> {
        let raw = owl_shared::file_loader::read(&dir.join("config.json"))
            .map_err(|e| ModelError::Msg(format!("ckpt config.json 读取({}): {e}", dir.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&raw)
            .map_err(|e| ModelError::Msg(format!("ckpt config.json 解析({}): {e}", dir.display())))?;
        Ok(CkptConfig::from_json(&v))
    }

    /// 形状事实取用:VLM 形态落到 `text_config`(语言主干)
    pub fn text(&self) -> &CkptConfig {
        self.text_config.as_deref().unwrap_or(self)
    }

    /// **草稿家族声明**(`architectures` 里带 `DFlash2DraftModel` / `Mtp`)
    /// —— 同名键语义裁决的**声明依据**(替代"按键尾猜")
    pub fn is_draft(&self) -> bool {
        self.architectures.iter().any(|a| {
            let a = a.to_ascii_lowercase();
            a.contains("dflash2draft") || a.contains("draft") || a.contains("mtp")
        })
    }

    /// **`q_proj` 是否门控**(out = 2·hq·hd):按**声明事实**裁决 ——
    /// 草稿家族 = 无门控;主模型(gated attention 家族)= 有。
    /// ⚠️ 无 `architectures` 声明时返回 `None`(调用方决定拒还是走旧启发式),
    /// **不猜**。
    pub fn q_proj_gated(&self) -> Option<bool> {
        if self.architectures.is_empty() {
            return None;
        }
        Some(!self.is_draft())
    }

    /// **打包字宽**(元素/`u32`):`num_bits = 4` ⇒ 8。非 4 ⇒ **响亮拒**
    /// (装载面的 `÷8` 是硬编码;声明与它不符时不许静默)
    pub fn pack_elems_per_word(&self) -> Result<usize, ModelError> {
        match self.quant.as_ref().and_then(|q| q.num_bits) {
            Some(4) => Ok(8),
            Some(other) => Err(ModelError::Msg(format!(
                "ckpt config.json: num_bits = {other} 未支持(装载面只实现 4bit 打包 = ÷8);\
                 拒绝静默按 8 处理"
            ))),
            None => Ok(8), // 非量化仓(f16/bf16 直读):该值不被消费
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_draft_family_facts() {
        // 真实草稿仓(syvai DFlash2 W4A16)的字段形状(值取自实测 config.json)
        let v: serde_json::Value = serde_json::json!({
            "architectures": ["DFlash2DraftModel"],
            "model_type": "qwen3",
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "tie_word_embeddings": false,
            "quantization_config": { "config_groups": { "group_0": {
                "format": "pack-quantized",
                "weights": { "group_size": 128, "num_bits": 4, "symmetric": true,
                             "strategy": "group", "actorder": null, "block_structure": null } } } },
            "dflash_config": { "block_size": 8, "conv_group_size": 16, "conv_kernel_size": 2,
                               "selector_rank": 256, "selector_top_k": 16,
                               "target_layer_ids": [5, 19, 33, 47, 61] }
        });
        let c = CkptConfig::from_json(&v);
        assert!(c.is_draft(), "architectures 声明草稿家族");
        assert_eq!(c.q_proj_gated(), Some(false), "草稿 q_proj 无门控");
        assert_eq!(c.pack_elems_per_word().unwrap(), 8);
        assert_eq!(c.quant.as_ref().unwrap().group_size, Some(128));
        assert_eq!(c.quant.as_ref().unwrap().symmetric, Some(true));
        assert_eq!(c.dflash.as_ref().unwrap().block_size, Some(8));
        assert_eq!(c.dflash.as_ref().unwrap().target_layer_ids, vec![5, 19, 33, 47, 61]);
        // out = hq·hd(无门控)⇒ 与实测装载日志 out=4096 一致
        assert_eq!(c.num_attention_heads.unwrap() * c.head_dim.unwrap(), 4096);
    }

    #[test]
    fn main_model_is_not_draft_and_nests_text_config() {
        let v: serde_json::Value = serde_json::json!({
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "tie_word_embeddings": false,
            "text_config": { "hidden_size": 5120, "num_attention_heads": 24,
                             "num_key_value_heads": 4, "head_dim": 256,
                             "quantization_config": { "config_groups": { "group_0": {
                                 "weights": { "group_size": 32, "num_bits": 4, "symmetric": false } } } } }
        });
        let c = CkptConfig::from_json(&v);
        assert!(!c.is_draft());
        assert_eq!(c.q_proj_gated(), Some(true), "主模型 gated attention");
        assert_eq!(c.text().hidden_size, Some(5120), "VLM 形态落到 text_config");
        assert_eq!(c.text().quant.as_ref().unwrap().group_size, Some(32));
        assert_eq!(c.text().quant.as_ref().unwrap().symmetric, Some(false));
    }

    #[test]
    fn refuses_unsupported_num_bits() {
        let v: serde_json::Value = serde_json::json!({
            "architectures": ["Whatever"],
            "quantization_config": { "config_groups": { "g": { "weights": { "num_bits": 8 } } } }
        });
        let c = CkptConfig::from_json(&v);
        let e = match c.pack_elems_per_word() {
            Err(e) => e.to_string(),
            Ok(_) => panic!("num_bits = 8 应响亮拒"),
        };
        assert!(e.contains("num_bits"), "拒因应指向 num_bits:{e}");
    }

    /// 无 `architectures` ⇒ **不猜**(`None`),由调用方决定
    #[test]
    fn no_architectures_means_no_guess() {
        let v: serde_json::Value = serde_json::json!({ "hidden_size": 8 });
        let c = CkptConfig::from_json(&v);
        assert_eq!(c.q_proj_gated(), None);
    }

    /// 真仓腿(有仓就读、没有就 skip;不设 env —— 走 workspace 缺省路径)
    #[test]
    fn reads_real_checkpoints_when_present() {
        for dir in [
            "/home/div/Documents/codes/models/syvai/Qwen3.8-27B-DFlash2-W4A16",
            "/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4",
        ] {
            let p = Path::new(dir);
            if !p.join("config.json").exists() {
                eprintln!("skip: {dir} 不在");
                continue;
            }
            let c = CkptConfig::read(p).expect("读 config.json");
            eprintln!(
                "[ckpt-config] {dir}: architectures={:?} draft={} gated={:?} group={:?} bits={:?} pack={}",
                c.architectures,
                c.is_draft(),
                c.q_proj_gated(),
                c.quant.as_ref().and_then(|q| q.group_size),
                c.quant.as_ref().and_then(|q| q.num_bits),
                c.pack_elems_per_word().map(|x| x.to_string()).unwrap_or_else(|e| e.to_string()),
            );
            assert!(!c.architectures.is_empty(), "真仓必有 architectures");
        }
    }
}
