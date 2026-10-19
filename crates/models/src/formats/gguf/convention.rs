//! **GGUF 键约定**(`KeyConvention` 第三套;2026-10-19 刀 4)。
//!
//! owl 的层只声明**层内局部键**(`q_proj` / `in_proj_qkvz` / `A_log` …),
//! 全键由 [`KeyConvention`] 在**声明期**改写(`Model::layout` 的 `map_keys`)——
//! 解释器零约定感知。本模块提供 GGUF 那一套(local → `blk.N.*` / 顶层键)。
//!
//! # 键面映射表(实测自 4B/27B 索引;机器门见本文件 `tests`)
//!
//! | owl 局部键 | GGUF | 备注 |
//! |---|---|---|
//! | `weight`(embed 表) | `token_embd.weight` | 行 = vocab,列 = hidden |
//! | `norm`(最终 norm) | `output_norm.weight` | **值 = HF 值 + 1**(见下) |
//! | `lm_head`(untied 头) | `output.weight` | **GgufConvention::head_key** 重写 |
//! | `input_layernorm` | `blk.N.attn_norm.weight` | +1 |
//! | `post_attention_layernorm` | `blk.N.post_attention_norm.weight` | +1 |
//! | `q_proj` / `k_proj` / `v_proj` / `o_proj` | `blk.N.attn_q`/`attn_k`/`attn_v`/`attn_output`.weight | |
//! | `q_norm` / `k_norm` | `blk.N.attn_q_norm` / `attn_k_norm`.weight | +1 |
//! | `gate_proj` / `up_proj` / `down_proj` | `blk.N.ffn_gate`/`ffn_up`/`ffn_down`.weight | |
//! | `in_proj_qkvz`(GDN 合并投影) | **虚拟键** `blk.N.attn_qkvz` | 拆自 `attn_qkv` ⊕ `attn_gate`(异型常态;见下) |
//! | `in_proj_a` / `in_proj_b` | `blk.N.ssm_alpha` / `ssm_beta`.weight | V-head 行重排 |
//! | `out_proj` | `blk.N.ssm_out.weight` | V-head **列**重排 |
//! | `conv1d` | `blk.N.ssm_conv1d.weight` | V 通道段重排 |
//! | `A_log` | `blk.N.ssm_a` | **裸键**;GGUF 存 `-exp(A_log)` |
//! | `dt_bias` | `blk.N.ssm_dt.bias` | **裸键** |
//! | `norm`(GDN 层内) | `blk.N.ssm_norm.weight` | **不加 1**(转换器例外;实测) |
//!
//! # 两条设计要点
//!
//! 1. **`in_proj_qkvz` 是虚拟键(合并语义)**:GGUF 里是 `attn_qkv` 与 `attn_gate`
//!    **两个**张量,而二者的 ggml 型**常常不同**(4B 实测 24/24 层:Q6_K vs Q4_K)
//!    ⇒ 无法合成单一同型张量。落地形态 = GGUF 档 GDN 声明**两个** Linear
//!    (刀 7),本约定同时支持两形态:传 `in_proj_qkvz` 时给虚拟键(源侧拆解),
//!    传 `in_proj_qkv`/`in_proj_z` 时给真实键。**异型合并门**:若走虚拟键,
//!    源必须验两段同型(否则拒)—— 本仓两靶位一律走**两 Linear**(不经虚拟键)。
//! 2. **`head_key` 是新增的第 4 类键**(embed/layer/norm 之外):untied lm_head
//!    在 GGUF 里叫 `output.weight`,而 HF 叫 `lm_head.weight` —— 它是**源的
//!    命名事实**,故必须由约定回答(原先 `Embedding::layout_lm_head` 硬编码
//!    `lm_head.weight`,对 GGUF 是错的)。

use crate::contract::ModelError;
use crate::module::KeyConvention;

/// GGUF 键约定(**无状态**;维度事实由源侧持有,这里只管命名)
#[derive(Debug, Clone, Copy, Default)]
pub struct GgufConvention;

impl GgufConvention {
    pub const fn new() -> Self {
        Self
    }
}

/// 量化后缀词表(与 `Qwen35Convention` 同表):GGUF 档**不接受**这些后缀 ——
/// GGUF 的量化块住**同一个** `.weight` 键里(没有 `.qweight` 一族)。
const QUANT_SUFFIXES: &[&str] = &[
    "qweight", "scales", "zeros", "ws", "marlin_ws", "marlin_ctmp", "packed_raw",
    "qweight_a8", "scales_a8", "zeros_a8", "a8_factor", "weight_packed", "weight_scale",
];

/// 层内局部键 → `blk.N.<gguf 名>`(无后缀路径;纯映射,零 I/O)
fn layer_local(i: usize, local: &str) -> Result<String, ModelError> {
    let name = match local {
        "input_layernorm" => "attn_norm.weight",
        "post_attention_layernorm" => "post_attention_norm.weight",
        "q_proj" => "attn_q.weight",
        "k_proj" => "attn_k.weight",
        "v_proj" => "attn_v.weight",
        "o_proj" => "attn_output.weight",
        "q_norm" => "attn_q_norm.weight",
        "k_norm" => "attn_k_norm.weight",
        "gate_proj" => "ffn_gate.weight",
        "up_proj" => "ffn_up.weight",
        "down_proj" => "ffn_down.weight",
        // GDN(**两形态**;见模块头 §1)
        "in_proj_qkvz" => "attn_qkvz", // 虚拟键(源侧拆 attn_qkv ⊕ attn_gate)
        "in_proj_qkv" => "attn_qkv.weight",
        "in_proj_z" => "attn_gate.weight",
        "in_proj_a" => "ssm_alpha.weight",
        "in_proj_b" => "ssm_beta.weight",
        "out_proj" => "ssm_out.weight",
        // ★ **虚拟键**(源生成、文件里没有):`out_proj` 的**激活侧列置换表**
        // (v-head 反重排;32 元素块粒度 = q8_1 块)。为什么要进键面:置换表
        // 必须**常驻显存**(轮内零 host 工作 ⇒ 可入捕获图),而“上卡”的唯一
        // 通道 = 权重槽 ⇒ 它像权重一样有键(源侧按名合成字节)。
        // **消费者** = GDN 层(声明 want + `owl_gguf_col_gather`)。
        "ssm_out_perm" => "ssm_out.perm",
        "conv1d" => "ssm_conv1d.weight",
        "A_log" => "ssm_a",
        "dt_bias" => "ssm_dt.bias",
        "norm" => "ssm_norm.weight",
        other => {
            return Err(ModelError::Msg(format!(
                "GgufConvention: 未知层内局部键 `{other}`(禁止静默兜底 —— 加映射或改布局声明)"
            )))
        }
    };
    Ok(format!("blk.{i}.{name}"))
}

impl KeyConvention for GgufConvention {
    /// 词表:`weight` → `token_embd.weight`
    fn embed_key(&self, local: &str) -> String {
        match local {
            "weight" => "token_embd.weight".to_string(),
            other => format!("token_embd.{other}"),
        }
    }

    /// 第 i 层:局部键 → `blk.i.<…>`;量化后缀 = 响亮拒(GGUF 无该族键)
    fn layer_key(&self, i: usize, local: &str) -> String {
        if let Some((_, s)) = local.rsplit_once('.') {
            if QUANT_SUFFIXES.contains(&s) {
                // 走到这里 = plan 选错了(GGUF 档不该声明 marlin/ct 键族)
                owl_shared::log_error!(
                    "gguf",
                    "键 `{local}` 带量化后缀 `.{}`,GGUF 起点不支持该族键(plan 选错?)",
                    s
                );
                return format!("blk.{i}.{local}");
            }
        }
        match layer_local(i, local) {
            Ok(k) => k,
            Err(e) => {
                // 约定层无 Result 出口(trait 定形);**响亮**:记 error 日志
                // + 返回明显不可能命中的键 ⇒ 装载期以"缺键"归因暴露
                // (比静默给个看似合法的键好:错键会命中别人)
                owl_shared::log_error!("gguf", "{e}");
                format!("blk.{i}.__unknown_local__.{local}")
            }
        }
    }

    /// 最终 norm:`norm` → `output_norm.weight`
    fn norm_key(&self, local: &str) -> String {
        match local {
            "norm" => "output_norm.weight".to_string(),
            other => format!("output_norm.{other}"),
        }
    }

    /// untied lm_head(第 4 类键;GGUF = `output.weight`)
    fn head_key(&self, local: &str) -> String {
        match local {
            "lm_head" => "output.weight".to_string(),
            other => format!("output.{other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::KeyConvention;

    /// 纯映射表(逐条;改表必须改这里)
    #[test]
    fn mapping_table_is_pinned() {
        let c = GgufConvention::new();
        assert_eq!(c.embed_key("weight"), "token_embd.weight");
        assert_eq!(c.norm_key("norm"), "output_norm.weight");
        assert_eq!(c.head_key("lm_head"), "output.weight");
        for (local, want) in [
            ("input_layernorm", "blk.7.attn_norm.weight"),
            ("post_attention_layernorm", "blk.7.post_attention_norm.weight"),
            ("q_proj", "blk.7.attn_q.weight"),
            ("k_proj", "blk.7.attn_k.weight"),
            ("v_proj", "blk.7.attn_v.weight"),
            ("o_proj", "blk.7.attn_output.weight"),
            ("q_norm", "blk.7.attn_q_norm.weight"),
            ("k_norm", "blk.7.attn_k_norm.weight"),
            ("gate_proj", "blk.7.ffn_gate.weight"),
            ("up_proj", "blk.7.ffn_up.weight"),
            ("down_proj", "blk.7.ffn_down.weight"),
            ("in_proj_qkv", "blk.7.attn_qkv.weight"),
            ("in_proj_z", "blk.7.attn_gate.weight"),
            ("in_proj_a", "blk.7.ssm_alpha.weight"),
            ("in_proj_b", "blk.7.ssm_beta.weight"),
            ("out_proj", "blk.7.ssm_out.weight"),
            ("conv1d", "blk.7.ssm_conv1d.weight"),
            ("A_log", "blk.7.ssm_a"),
            ("dt_bias", "blk.7.ssm_dt.bias"),
            ("norm", "blk.7.ssm_norm.weight"),
        ] {
            assert_eq!(c.layer_key(7, local), want, "局部键 {local}");
        }
        // 虚拟合并键(源侧拆解)
        assert_eq!(c.layer_key(7, "in_proj_qkvz"), "blk.7.attn_qkvz");
    }

    /// 未知局部键:**不许静默兜底**(返回不可能命中的键 ⇒ 装载期以缺键暴露)
    #[test]
    fn unknown_local_is_loud() {
        let c = GgufConvention::new();
        let k = c.layer_key(3, "definitely_not_a_key");
        assert!(k.contains("__unknown_local__"), "{k}");
        assert!(k.starts_with("blk.3."));
    }

    /// 量化后缀族:GGUF 起点不支持(`.qweight` 一族)—— 返回带后缀的键
    /// (必然缺键 ⇒ 装载期响亮拒,而不是静默命中 `attn_q`)
    #[test]
    fn quant_suffix_family_is_rejected_from_mapping() {
        let c = GgufConvention::new();
        for local in ["q_proj.qweight", "q_proj.scales", "in_proj_qkvz.qweight"] {
            let k = c.layer_key(2, local);
            assert!(k.ends_with(local), "带后缀键应原样保留以便缺键归因: {k}");
        }
    }

    // ========================================================================
    // **I-7 键面覆盖门**(设计 §六 I-7):模型**真实声明**的每一个键都必须在
    // GGUF 索引里。这是约定 + 维度事实 + 张量表的**三方合练**,且零设备:
    // 只跑 `Model::layout`(声明期),不碰一个字节。
    // ========================================================================
    #[test]
    fn declared_key_face_is_covered_by_gguf_index() {
        use crate::contract::Dtype;
        use crate::formats::gguf::{reader::Gguf, spec_from_gguf};
        use crate::model::Model;
        use crate::module::{Loadable, LoaderCtx, QuantPlan};
        use owl_shared::testkit::{resource, resource_ok};
        use std::path::PathBuf;

        for key in ["gguf:4b", "gguf:27b"] {
            if !resource_ok(key) {
                eprintln!("skip: 资源清单缺 \"{key}\"");
                continue;
            }
            let g = Gguf::open(&PathBuf::from(resource(key).expect("列了就有"))).expect("open");
            let (spec, _rep) = spec_from_gguf(&g).expect("spec");
            let model = Model::new(&spec, GgufConvention::new(), QuantPlan::F16, QuantPlan::F16)
                .expect("Model::new");
            let ctx = LoaderCtx {
                dtype: Dtype::F16,
                shard: 1,
                slice_id: 0,
                device_repack: false,
                verify: false,
                debug_tap: false,
                            staging_mb: 128,
                dump: false,
            };
            let ops = model.layout(&ctx);

            let mut miss: Vec<String> = Vec::new();
            let mut virt = 0usize;
            for w in ops.wants() {
                if g.index_of(&w.key).is_some() {
                    continue;
                }
                // 虚拟合并键(GDN 合并投影):两段皆在场即算覆盖
                if let Some(base) = w.key.strip_suffix("attn_qkvz") {
                    if g.index_of(&format!("{base}attn_qkv.weight")).is_some()
                        && g.index_of(&format!("{base}attn_gate.weight")).is_some()
                    {
                        virt += 1;
                        continue;
                    }
                }
                miss.push(w.key.clone());
            }
            assert!(
                miss.is_empty(),
                "[{key}] 模型声明了索引里没有的键 {} 个:{:?}",
                miss.len(),
                miss.iter().take(8).collect::<Vec<_>>()
            );
            // 反向面(信息):索引有而模型未声明的键 —— MTP 块/拓扑差异都在这
            // 声明集(含虚拟合并键的**组成键** —— 它们经虚拟键被消费)
            let mut decl: std::collections::HashSet<String> =
                ops.wants().iter().map(|w| w.key.clone()).collect();
            for w in ops.wants() {
                if let Some(base) = w.key.strip_suffix("attn_qkvz") {
                    decl.insert(format!("{base}attn_qkv.weight"));
                    decl.insert(format!("{base}attn_gate.weight"));
                }
            }
            let mut unused: Vec<&str> = g
                .tensors
                .iter()
                .map(|t| t.name.as_str())
                .filter(|n| !decl.contains(*n))
                .collect();
            unused.sort();
            // 未用键应**只**来自 MTP 块之外不存在的那部分 —— 判据(通用):
            // 凡 `blk.<i>.` 形式的未用键,i 必须 ≥ 解码层数(= MTP 块)
            let n_dec = spec.layer_types.len();
            let offenders: Vec<&&str> = unused
                .iter()
                .filter(|n| match n.strip_prefix("blk.").and_then(|r| r.split('.').next()) {
                    Some(idx) => idx.parse::<usize>().map(|i| i < n_dec).unwrap_or(true),
                    None => true, // 非 blk 键不该未声明
                })
                .collect();
            assert!(
                offenders.is_empty(),
                "[{key}] 解码层内有未声明的索引键(约定漏臂?):{:?}",
                offenders.iter().take(8).collect::<Vec<_>>()
            );
            eprintln!(
                "[{key}] 键面覆盖绿:声明 {} 件(其中虚拟合并 {virt} 件)· 未用 {} 件(均 nextn/MTP)",
                ops.wants().len(),
                unused.len()
            );
        }
    }
}
