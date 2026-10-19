//! **GGUF 超参 → `ModelSpec`**(维度事实;2026-10-19 刀 3;REQ-GGUF-02)。
//!
//! 与 HF 路径(`spec.rs::from_ckpt`)同律、不同源:
//!
//! | 事实 | HF 路径 | GGUF 路径(本模块) |
//! |---|---|---|
//! | 维度 | `config.json`(优先)+ 张量形状兜底 | **GGUF 超参 KV**(`qwen35.*`) |
//! | 复核 | config ↔ 形状,不符采形状 + warn | KV ↔ 形状,不符**响亮拒**(见下) |
//! | 层型表 | 索引键面(有 `self_attn.*` = full) | **索引键面**(同一律) |
//! | 词表大小 | `embed_tokens` 形状 | `token_embd` 形状 |
//!
//! # 为什么 GGUF 路径对「KV ↔ 形状」从严(拒,而不是采形状)
//!
//! HF 路径的"采形状 + warn"是**历史受害者**(cyankiwi requant 的 config.json
//! 与形状不符)。GGUF 没有这个包袱:**超参是 gguf 写入器从同一份 HF config 抄的**,
//! 且张量表与超参在**同一个文件**里 —— 两者不符 = 文件自相矛盾(坏文件或读错)。
//! 此时采任何一边都可能是静默错值 ⇒ 拒(与 `reader` 的截断/越界同款处置)。
//!
//! # 层型表(hybrid)怎么定 —— **键面是权威**(2026-10-19 4B 修正)
//!
//! 键面实测:该层有 `attn_q.weight` = full,有 `attn_qkv.weight` = GDN
//!(与 HF 路径同律:索引实测为准)。`full_attention_interval` **不是判据** ——
//! 4B 实测反例:块 32 是 full(带 `attn_q`)但它是 **MTP/NextN 头**(带
//! `nextn.*`),不属解码层;若拿 interval 硬推就会把 33 层全判错。
//! interval 只做**统计自检**(full 层数 vs `ceil(n/interval)`,差 > 1 记 issues)。
//!
//! # MTP(NextN)块不属解码层
//!
//! `block_count` 把 MTP 头也算一层(4B:33 = 32 解码 + 1 MTP;HF 侧 = `mtp.*`)。
//! 判据 = 该块带 `nextn.eh_proj.weight`。M1 不载 MTP ⇒ 排除 + 记 issues。

use crate::contract::{Dtype, ModelError};
use crate::model::ModelSpec;
use crate::tokenizer::{ChatFormat, TokenizerSpec};

use super::reader::Gguf;

/// 读取账(诊断;`issues` 非空 = 已按宽松项处理过)
#[derive(Debug, Clone, Default)]
pub struct GgufSpecReport {
    /// `general.architecture` 原文
    pub arch: String,
    /// 宽松项的说明(如"kv 未给 X,采张量形状")
    pub issues: Vec<String>,
}

/// 支持的 GGUF 架构(其余响亮拒)
const ARCH_QWEN35: &str = "qwen35";
/// 家族 eos token 名(max 2 项;与 qwen35 HF 预设同源)
const QWEN35_EOS: &[&str] = &["<|im_end|>", "<|endoftext|>"];

/// **GGUF 超参 → 维度事实**(唯一入口)。
pub fn spec_from_gguf(g: &Gguf) -> Result<(ModelSpec, GgufSpecReport), ModelError> {
    let arch = g.arch()?.to_string();
    if arch != ARCH_QWEN35 {
        return Err(ModelError::Msg(format!(
            "gguf spec: 架构 `{arch}` 未接通(本刀只接 `{ARCH_QWEN35}`;\
             `qwen35moe` = 设计 REQ-GGUF-09 挂账,`dflash` = 草稿线挂账)"
        )));
    }
    let kv = &g.kv;
    let m = |e: String| ModelError::Msg(format!("gguf spec({arch}): {e}"));
    let p = |k: &str| format!("{arch}.{k}");
    let mut rep = GgufSpecReport { arch: arch.clone(), issues: Vec::new() };

    // ---- 超参(必须在场)----
    let hidden = kv.usize(&p("embedding_length")).map_err(&m)?;
    let inter = kv.usize(&p("feed_forward_length")).map_err(&m)?;
    let hq = kv.usize(&p("attention.head_count")).map_err(&m)?;
    let hkv = kv.usize(&p("attention.head_count_kv")).map_err(&m)?;
    let hd = kv.usize(&p("attention.key_length")).map_err(&m)?;
    let eps = kv.f32(&p("attention.layer_norm_rms_epsilon")).map_err(&m)?;
    let interval = kv.usize(&p("full_attention_interval")).map_err(&m)?;
    if interval == 0 {
        return Err(m("full_attention_interval=0".into()));
    }
    // GDN 头四元组:KV 语义 → owl 语义(见模块头推导)
    let state = kv.usize(&p("ssm.state_size")).map_err(&m)?; // hk_dim
    let group = kv.usize(&p("ssm.group_count")).map_err(&m)?; // nk
    let nv = kv.usize(&p("ssm.time_step_rank")).map_err(&m)?; // nv
    let inner = kv.usize(&p("ssm.inner_size")).map_err(&m)?; // nv·hv_dim
    if nv == 0 || inner % nv != 0 {
        return Err(m(format!("ssm.inner_size={inner} 不被 time_step_rank={nv} 整除")));
    }
    let hv_dim = inner / nv;
    let gdn_heads = (group, state, nv, hv_dim);

    // ---- 形状事实(必须在场;逐条复核超参)----
    let tensor = |name: &str| -> Result<&super::reader::TensorInfo, ModelError> {
        g.get_tensor(name).ok_or_else(|| m(format!("缺张量 `{name}`(维度事实无法自证)")))
    };
    let emb = tensor("token_embd.weight")?;
    if emb.ne.len() != 2 {
        return Err(m(format!("`token_embd.weight` 维数 {} ≠ 2", emb.ne.len())));
    }
    // GGML 序:ne[0] = 最内维 = hidden
    let vocab = emb.ne[1] as usize;
    let shape_facts = [
        ("token_embd.weight ne[0](hidden)", hidden, emb.ne[0] as usize),
        (
            "blk.0.attn_qkv.weight ne[1](2·nk·hk+nv·hv)",
            2 * group * state + nv * hv_dim,
            tensor("blk.0.attn_qkv.weight")?.ne[1] as usize,
        ),
        (
            "blk.0.attn_gate.weight ne[1](in_proj_z = nv·hv)",
            nv * hv_dim,
            tensor("blk.0.attn_gate.weight")?.ne[1] as usize,
        ),
        (
            "blk.0.ssm_out.weight ne[0](nv·hv)",
            nv * hv_dim,
            tensor("blk.0.ssm_out.weight")?.ne[0] as usize,
        ),
    ];
    for (what, want, got) in shape_facts {
        if want != got {
            return Err(m(format!("超参↔形状不符:{what} 超参推出 {want},实测 {got}(拒:文件自相矛盾)")));
        }
    }
    // 首个 full 层的 q/k/v 形状(kv 头 × head_dim)
    let first_full = interval - 1;
    let q = tensor(&format!("blk.{first_full}.attn_q.weight"))?;
    let k = tensor(&format!("blk.{first_full}.attn_k.weight"))?;
    let v = tensor(&format!("blk.{first_full}.attn_v.weight"))?;
    // q = hq·hd·2(attn_output_gate:q ⊕ gate 融合;owl Attention 原生切分)
    for (what, want, got) in [
        ("attn_q ne[1](2·hq·hd)", 2 * hq * hd, q.ne[1] as usize),
        ("attn_k ne[1](hkv·hd)", hkv * hd, k.ne[1] as usize),
        ("attn_v ne[1](hkv·hd)", hkv * hd, v.ne[1] as usize),
        ("attn_q ne[0](hidden)", hidden, q.ne[0] as usize),
    ] {
        if want != got {
            return Err(m(format!("超参↔形状不符:{what} 超参推出 {want},实测 {got}")));
        }
    }
    // q_norm/k_norm 维数 = head_dim(二次复核)
    for name in [format!("blk.{first_full}.attn_q_norm.weight"), format!("blk.{first_full}.attn_k_norm.weight")] {
        let t = tensor(&name)?;
        if t.ne != vec![hd as u64] {
            return Err(m(format!("`{name}` ne={:?} ≠ [head_dim={hd}]", t.ne)));
        }
    }

    // ---- 层数:MTP(NextN)块 **不是解码层**(2026-10-19 4B 实证)----
    // `block_count` 把 MTP 头也算了一层(4B:33 = 32 解码 + 1 MTP)。
    // 判据 = 该块带 `nextn.*`(HF `mtp.*` 的对应物)。M1 不载 MTP ⇒ 排除 + 记账。
    let declared = kv.usize(&p("block_count")).map_err(&m)?;
    let mut n_layers = declared;
    for i in 0..declared {
        if g.index_of(&format!("blk.{i}.nextn.eh_proj.weight")).is_some() {
            n_layers = i;
            rep.issues.push(format!(
                "块 {i} 带 `nextn.*` = MTP/NextN 头(非解码层)⇒ 解码层数 = {i}(声明 {declared});M1 不载 MTP"
            ));
            break;
        }
    }
    if n_layers == 0 {
        return Err(m("数不出任何解码层".into()));
    }

    // ---- 层型表:**键面是权威**(与 HF 路径同律:索引实测为准)----
    // `full_attention_interval` 只作统计自检(见下)—— 它**不是**判据:
    // 末层/MTP 会让它失效(4B 的块 32 是 full 但不属解码层)。
    let mut layer_types = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        let has_full = g.index_of(&format!("blk.{i}.attn_q.weight")).is_some();
        let has_gdn = g.index_of(&format!("blk.{i}.attn_qkv.weight")).is_some();
        match (has_full, has_gdn) {
            (true, false) => layer_types.push(true),
            (false, true) => layer_types.push(false),
            _ => {
                return Err(m(format!(
                    "层 {i} 键面既有/两无 `attn_q`(full)与 `attn_qkv`(GDN)(层型歧义)"
                )))
            }
        }
    }
    // 统计自检:full 层数应与 ceil(n/interval) 相差 ≤ 1(末层/取整差允许)
    let n_full = layer_types.iter().filter(|f| **f).count();
    let expect_full = n_layers.div_ceil(interval);
    if n_full.abs_diff(expect_full) > 1 {
        rep.issues.push(format!(
            "full 层数 {n_full} 与 interval={interval} 推出的 {expect_full} 差 > 1(可疑,已采键面)"
        ));
    }

    // ---- tied:untied 的判据 = `output.weight` 在场 ----
    let tied = g.index_of("output.weight").is_none();
    if tied {
        rep.issues.push("无 `output.weight` ⇒ tied(lm_head 复用 token_embd)".into());
    }

    let spec = ModelSpec {
        vocab,
        hidden,
        inter,
        // 计算档恒 F16(激活域不因权重是 GGUF 块型而变;设计 REQ-PRE-04)
        dtype: Dtype::F16,
        full_heads: (hq, hkv, hd),
        gdn_heads,
        eps,
        layer_types,
        tokenizer: TokenizerSpec {
            // 家族 eos 名(HF 同源);GGUF **文件自带的 id** 是权威,
            // 由装载侧经 `GgufVocab` 直接注入(刀 7),此处只供家族路径使用
            eos_tokens: QWEN35_EOS.to_vec(),
            chat: family_chat_format(),
        },
        tied,
    };
    Ok((spec, rep))
}

/// qwen35 家族 ChatML 形态(`ChatFormat`)。
///
/// ⚠️ **这不是 GGUF 档的权威 prompt 路径**:权威 = 文件内嵌 Jinja 模板,由
/// serving 面的渲染器执行(`apps/server/chat_tpl.rs`,`ChatTpl::load_gguf`)。
/// 本函数只服务 engine 的**单轮便捷面**(`/tokenize?chat_wrap=true` 与裸文本
/// 提交),故退化为家族 ChatML 缺省形态(与 `qwen3_5_0_8b` 同源)。
fn family_chat_format() -> ChatFormat {
    ChatFormat {
        prefix: "<|im_start|>user\n".into(),
        suffix: "<|im_end|>\n<|im_start|>assistant\n".into(),
        im_open: "<|im_start|>".into(),
        im_close: "<|im_end|>\n".into(),
        assistant_open: "<|im_start|>assistant\n".into(),
        think_prefill: String::new(),
    }
}

/// **rope 事实**(GGUF KV;REQ-GGUF-02 单文件自给的一环)
///
/// | 事实 | KV 键 | 4B 实测 |
/// |---|---|---|
/// | 旋转维 | `qwen35.rope.dimension_count` | 64 |
/// | 频率基 | `qwen35.rope.freq_base` | 1e7 |
/// | 上下文长 | `qwen35.context_length` | 262144 |
///
/// 头维(`head_dim`)不在本结构里 —— 它就是 `full_heads.2`(`attention.key_length`),
/// 由 [`spec_from_gguf`] 一并给出(单一事实源,不重复读键)。
///
/// ⚠️ **mRoPE 分区**(`rope.dimension_sections`)不在此列:M1 只做文本
/// 单轴路径(与现役 27B 同构);多模态三轴走 caller 自己的表(REQ-GGUF-10 挂账)。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GgufRopeFacts {
    /// 旋转维(偶数;`Rope::new` 的 rotary_dim)
    pub rotary_dim: usize,
    /// 频率基(theta)
    pub theta: f32,
    /// 位置表长(模型上下文长;`Rope::new` 的 max_pos)
    pub max_pos: usize,
}

/// rope 事实读取(唯一入口;缺键/非法 ⇒ 响亮拒)
pub fn rope_facts(g: &Gguf) -> Result<GgufRopeFacts, ModelError> {
    let arch = g.arch()?;
    let p = |k: &str| format!("{arch}.{k}");
    let m = |e: String| ModelError::Msg(format!("gguf rope({arch}): {e}"));
    let kv = &g.kv;
    let rotary_dim = kv.usize(&p("rope.dimension_count")).map_err(&m)?;
    let theta = kv.f32(&p("rope.freq_base")).map_err(&m)?;
    let max_pos = kv.usize(&p("context_length")).map_err(&m)?;
    if rotary_dim == 0 || rotary_dim % 2 != 0 {
        return Err(m(format!("rope.dimension_count={rotary_dim} 非法(须非零偶数)")));
    }
    if !(theta.is_finite() && theta > 0.0) {
        return Err(m(format!("rope.freq_base={theta} 非法")));
    }
    if max_pos == 0 {
        return Err(m("context_length=0".into()));
    }
    Ok(GgufRopeFacts { rotary_dim, theta, max_pos })
}

#[cfg(test)]
mod tests {
    use super::*;
    use owl_shared::testkit::{resource, resource_ok};
    use std::path::PathBuf;

    fn res(key: &str) -> Option<PathBuf> {
        resource_ok(key).then(|| PathBuf::from(resource(key).expect("列了就有")))
    }

    /// **超参 ↔ 手写常量/形状 三方对拍**:9B 与 27B 两档。
    ///
    /// 判据 = 与 `formats/qwen35.rs` 的手写预设**逐字段相等**(那是经过真机
    /// 装载/对拍验证过的维度事实)——GGUF 读出的东西必须与之一致,否则
    /// 后续装载的每一维都错。
    #[test]
    fn gguf_spec_matches_family_presets() {
        // (资源键, 期望: hidden, inter, 层数, full_heads, gdn_heads, tied)
        let want: &[(&str, usize, usize, usize, (usize, usize, usize), (usize, usize, usize, usize), bool)] = &[
            // 4B:M1 靶位;**tied**(无 output.weight)+ MTP 块 32(非解码层)
            ("gguf:4b", 2560, 9216, 32, (16, 4, 256), (16, 128, 32, 128), true),
            // 27B 与手写预设同(24,4,256)/(16,128,48,128),层数 64
            ("gguf:27b", 5120, 17408, 64, (24, 4, 256), (16, 128, 48, 128), false),
        ];
        for (key, hidden, inter, n, fh, gh, tied) in want {
            let Some(path) = res(key) else {
                eprintln!("skip: 资源清单缺 \"{key}\"");
                continue;
            };
            let g = Gguf::open(&path).expect("open");
            let (spec, rep) = spec_from_gguf(&g).unwrap_or_else(|e| panic!("[{key}] {e}"));
            eprintln!(
                "[{key}] arch={} vocab={} hidden={} inter={} layers={} full={:?} gdn={:?} tied={} issues={:?}",
                rep.arch, spec.vocab, spec.hidden, spec.inter, spec.layer_types.len(),
                spec.full_heads, spec.gdn_heads, spec.tied, rep.issues
            );
            assert_eq!(spec.vocab, 248320, "[{key}] vocab");
            assert_eq!(spec.hidden, *hidden, "[{key}] hidden");
            assert_eq!(spec.inter, *inter, "[{key}] inter");
            assert_eq!(spec.layer_types.len(), *n, "[{key}] 层数");
            assert_eq!(spec.full_heads, *fh, "[{key}] full heads");
            assert_eq!(spec.gdn_heads, *gh, "[{key}] gdn heads");
            assert_eq!(spec.tied, *tied, "[{key}] tied(4B=untied 的反面:tied)");
            assert!((spec.eps - 1e-6).abs() < 1e-12, "[{key}] eps");
            assert_eq!(spec.layer_types, crate::formats::qwen35::hybrid_3to1(*n), "[{key}] 层型表");
            assert_eq!(spec.dtype, Dtype::F16);
            assert_eq!(spec.tokenizer.eos_tokens, QWEN35_EOS);
            // 27B 与手写预设逐字段同源(9B 无预设,查 27B)
            if *n == 64 {
                let p = crate::formats::qwen35::qwen3_8_27b();
                assert_eq!(spec.hidden, p.hidden);
                assert_eq!(spec.inter, p.inter);
                assert_eq!(spec.vocab, p.vocab);
                assert_eq!(spec.full_heads, p.full_heads);
                assert_eq!(spec.gdn_heads, p.gdn_heads);
                assert_eq!(spec.layer_types, p.layer_types);
                assert_eq!(spec.tied, p.tied);
            }
        }
    }

    /// 响亮拒:非 qwen35 架构(须点名架构;`qwen35moe`/`dflash` 是挂账项)
    #[test]
    fn rejects_other_arch() {
        // 造一个最小 GGUF:只有 architecture 键。用真实文件验证更省事:
        // 草稿 GGUF(arch=dflash)在三层挂账 ⇒ 必须拒
        if let Some(p) = res("gguf:draft-q4") {
            let g = Gguf::open(&p).expect("open");
            // `.unwrap_err()` 要 `T: Debug`(ModelSpec 未 derive)⇒ 走 match
            let e = match spec_from_gguf(&g) {
                Ok(_) => panic!("dflash 草稿架构必须拒(本刀未接)"),
                Err(e) => e.to_string(),
            };
            assert!(e.contains("未接通"), "应点名架构未接通: {e}");
            assert!(e.contains("dflash"), "{e}");
        } else {
            eprintln!("skip: 资源清单缺 gguf:draft-q4");
        }
    }
}
