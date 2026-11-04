//! **检查点声明面**(`CheckpointDecl`)—— 2026-11-04 clean-code 第一步。
//!
//! # 为什么
//!
//! `--check` 原先有**两条平行的校验路径**(`check_model_dir` 读 HF
//! `config.json` / `check_gguf_model` 读 GGUF KV + `rope_facts`),
//! 两份代码各自做「几何对账 + 视觉塔声明 + 一句话摘要」,字段名与错误文案
//! 各写一遍 ⇒ 新增一个源格式 = 再抄一份校验。这里把**声明事实**抽成一个
//! 源无关的接口:消费者(`--check`)只有一条路径,格式差异封在两个 impl 里。
//!
//! # 边界
//!
//! 本接口**只读元数据,不装权重**(纯 host 面,无设备、无 async)。
//! 权重装载路径(§2 第二步)复用同一套事实,但那是后续刀的范围。
//!
//! # 纪律
//!
//! - 事实**单源**:`DeclGeom` 由各源自己的解析器(`ModelSpec::from_ckpt` /
//!   `spec_from_gguf`)填充,本模块不重解析字节;
//! - 「超 max_pos」的处置**按源声明**(HF = 硬拒;GGUF = 软警告+YaRN 提示)
//!   —— 这是既有语义,不是本模块发明的策略;
//! - 未知事实用 `0` 表示(而不是猜),消费者按 `Option` 语义分支。

use std::path::Path;


/// 源无关的**几何声明**(检查点自己说了什么)。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeclGeom {
    pub hidden: usize,
    pub n_layers: usize,
    /// full-attention 层数(0 = 源未声明)
    pub n_full: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    /// 检查点声明的最大位置(0 = 未声明)
    pub max_pos: usize,
}

/// `max_pos` 被 `runtime.max_seq` 超过时的处置(**按源**:既有语义)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaxPosPolicy {
    /// 硬拒(HF:模型装不下)
    Hard,
    /// 软警告(GGUF:可靠 YaRN 外推)
    WarnYarn,
}

/// 视觉塔声明(已渲染成人读一段;消费者不再认识两套格式)。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionDecl {
    /// 例如 `../models/…/mmproj-F16.gguf（塔 27 层 / hidden 1152 / out 5120 / image_token 248056）`
    pub note: String,
}

/// 草稿(投机)声明面。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DraftDecl {
    /// 检查点自带 MTP 头层数(0 = 无)
    pub mtp_layers: usize,
}

/// **检查点声明**(两个源格式都答得出的问题)。
pub trait CheckpointDecl {
    /// 几何事实(单源;由各源自己的解析器填)
    fn geom(&self) -> &DeclGeom;
    /// 格式名(摘要用:`gguf` / `hf` …)
    fn format_name(&self) -> &'static str;
    /// 一句话摘要(格式 + 关键事实;**不含** profile 比对结论)
    fn summary(&self) -> String;
    /// `max_pos` 超标时的处置策略
    fn max_pos_policy(&self) -> MaxPosPolicy;
    /// 视觉塔声明:Some = 塔在场(渲染好的一段);None = 无塔。
    ///
    /// `vision_path` 覆盖只对 GGUF 有意义(单文件塔);HF 档给 path = 配置自相矛盾
    /// (由 impl 响亮拒,与装载期同律)。
    fn vision(&self, vision_path: Option<&Path>) -> Result<Option<VisionDecl>, String>;
    /// 草稿声明(MTP 头 / DFlash2 由独立目录声明,不在此列)
    fn draft(&self) -> DraftDecl;
}

/// **打开检查点声明**:按源格式分派**唯一一处**。
///
/// `gguf = true` ⇒ 单文件容器;否则 ⇒ HF/AWQ 目录(`config.json`)。
/// `Ok(None)` = 该源**文件不在**(HF 缺 `config.json`)—— 与历史行为一致:
/// `--check` 只提示"几何校验跳过",不判失败(真启动会失败)。
pub fn open_decl(path: &Path, gguf: bool) -> Result<Option<Box<dyn CheckpointDecl>>, String> {
    if gguf {
        return Ok(Some(Box::new(GgufDecl::open(path)?)));
    }
    if !path.join("config.json").is_file() {
        return Ok(None);
    }
    Ok(Some(Box::new(HfDecl::open(path)?)))
}

// ============================================================================
// HF / AWQ 目录(声明源 = `config.json` 的 `text_config.*`)
// ============================================================================

struct HfDecl {
    dir: std::path::PathBuf,
    geom: DeclGeom,
    /// `full_attention_interval`(0 = 未声明)
    full_interval: usize,
    /// `vision_config` 在场(塔权重在检查点目录 `model.visual.*`)
    has_vision_cfg: bool,
    has_image_token: bool,
}

impl HfDecl {
    fn open(dir: &Path) -> Result<Self, String> {
        let p = dir.join("config.json");
        let raw = owl_shared::file_loader::read_to_string(&p)
            .map_err(|e| format!("读取 {} 失败: {e}", p.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw)
            .map_err(|e| format!("{} 不是合法 JSON: {e}", p.display()))?;
        let g = v.get("text_config").unwrap_or(&v);
        let getu = |k: &str| g.get(k).and_then(|x| x.as_u64()).map(|n| n as usize);
        let n_layers = getu("num_hidden_layers").unwrap_or(0);
        let full_interval = getu("full_attention_interval").unwrap_or(0);
        let n_full = if full_interval > 0 { n_layers / full_interval } else { 0 };
        let geom = DeclGeom {
            hidden: getu("hidden_size").unwrap_or(0),
            n_layers,
            n_full,
            n_kv_heads: getu("num_key_value_heads").unwrap_or(0),
            head_dim: getu("head_dim").unwrap_or(0),
            vocab: getu("vocab_size").unwrap_or(0),
            max_pos: getu("max_position_embeddings").unwrap_or(0),
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            geom,
            full_interval,
            has_vision_cfg: v.get("vision_config").is_some(),
            has_image_token: v.get("image_token_id").and_then(|x| x.as_u64()).is_some(),
        })
    }
}

impl CheckpointDecl for HfDecl {
    fn geom(&self) -> &DeclGeom {
        &self.geom
    }
    fn format_name(&self) -> &'static str {
        "hf"
    }
    fn summary(&self) -> String {
        let g = &self.geom;
        format!(
            "{} ✓ hidden={} layers={}(full {}) kv_heads={} hd={} max_pos={}",
            self.dir.display(),
            g.hidden,
            g.n_layers,
            g.n_full,
            g.n_kv_heads,
            g.head_dim,
            g.max_pos
        )
    }
    fn max_pos_policy(&self) -> MaxPosPolicy {
        MaxPosPolicy::Hard
    }
    fn vision(&self, vision_path: Option<&Path>) -> Result<Option<VisionDecl>, String> {
        if vision_path.is_some() {
            return Err(format!(
                "{}:[vision] path 仅用于 format = \"gguf\"：HF/AWQ 档的视觉塔权重在检查点目录的 model.visual.*",
                self.dir.display()
            ));
        }
        if !self.has_vision_cfg {
            return Err(format!(
                "[vision] enabled = true,但 {} 的 config.json 无 vision_config(不是 VLM 检查点)",
                self.dir.display()
            ));
        }
        if !self.has_image_token {
            return Err(format!(
                "[vision] enabled = true,但 {} 的 config.json 缺 image_token_id",
                self.dir.display()
            ));
        }
        // HF 档的塔事实由装载期校验(此处只声明"在场")
        Ok(Some(VisionDecl { note: "model.visual.*".into() }))
    }
    fn draft(&self) -> DraftDecl {
        // MTP 头是否自带:由 `mtp.*` 张量索引声明(装载期读索引),此处不预读。
        DraftDecl::default()
    }
}

// ============================================================================
// GGUF 单文件(声明源 = 文件 KV + rope_facts)
// ============================================================================

struct GgufDecl {
    path: std::path::PathBuf,
    geom: DeclGeom,
    arch: String,
    /// 视觉塔事实(GGUF 的塔是独立单文件,声明在 `mmproj`)
    vision_note: Option<String>,
}

impl GgufDecl {
    fn open(path: &Path) -> Result<Self, String> {
        use crate::formats::gguf::{rope_facts, spec_from_gguf, Gguf};
        let g = Gguf::open(path).map_err(|e| format!("打开 GGUF {} 失败: {e}", path.display()))?;
        let (spec, rep) =
            spec_from_gguf(&g).map_err(|e| format!("{} GGUF 超参: {e}", path.display()))?;
        let rf = rope_facts(&g).map_err(|e| format!("{} GGUF rope: {e}", path.display()))?;
        let geom = DeclGeom {
            hidden: spec.hidden,
            n_layers: spec.layer_types.len(),
            n_full: spec.layer_types.iter().filter(|f| **f).count(),
            n_kv_heads: spec.full_heads.1,
            head_dim: spec.full_heads.2,
            vocab: spec.vocab,
            max_pos: rf.max_pos,
        };
        Ok(Self { path: path.to_path_buf(), geom, arch: rep.arch, vision_note: None })
    }
}

impl CheckpointDecl for GgufDecl {
    fn geom(&self) -> &DeclGeom {
        &self.geom
    }
    fn format_name(&self) -> &'static str {
        "gguf"
    }
    fn summary(&self) -> String {
        let g = &self.geom;
        format!(
            "{} ✓ GGUF {} · hidden={} layers={}(full {}) kv_heads={} hd={} vocab={} · ctx={}",
            self.path.display(),
            self.arch,
            g.hidden,
            g.n_layers,
            g.n_full,
            g.n_kv_heads,
            g.head_dim,
            g.vocab,
            g.max_pos
        )
    }
    fn max_pos_policy(&self) -> MaxPosPolicy {
        MaxPosPolicy::WarnYarn
    }
    fn vision(&self, vision_path: Option<&Path>) -> Result<Option<VisionDecl>, String> {
        use crate::formats::gguf::{resolve_mmproj_path, Gguf};
        // `--check` 面:path 缺席且同目录无候选 = 无塔(装载期才会拒)
        let Ok(mm) = resolve_mmproj_path(&self.path, vision_path) else {
            return Ok(None);
        };
        let text = Gguf::open(&self.path)
            .map_err(|e| format!("打开 GGUF {} 失败: {e}", self.path.display()))?;
        let mg = Gguf::open(&mm).map_err(|e| format!("打开 mmproj {} 失败: {e}", mm.display()))?;
        let (spec, _) = crate::formats::gguf::spec_from_gguf(&text)
            .map_err(|e| format!("{} GGUF 超参: {e}", self.path.display()))?;
        let vcfg = crate::vlm::VlmCfg::from_gguf(&text, &mg, spec).map_err(|e| e.to_string())?;
        let note = format!(
            "{}（塔 {} 层 / hidden {} / out {} / image_token {}）",
            mm.display(),
            vcfg.vision.depth,
            vcfg.vision.hidden,
            vcfg.vision.out_hidden,
            vcfg.image_token
        );
        Ok(Some(VisionDecl { note }))
    }
    fn draft(&self) -> DraftDecl {
        DraftDecl::default()
    }
}

/// **`--check` 的几何对账**(源无关):声明 vs 注册表 profile。
///
/// 返回 `Ok(())` = 通过;`Err` = 响亮拒(文案带 profile key 与字段名)。
/// 未知字段(声明里为 0/未给)按"跳过"处理 —— 与两条旧路径同律。
pub fn check_geom_against(
    decl: &dyn CheckpointDecl,
    profile: &owl_shared::config::intent::ModelProfile,
    max_seq: usize,
) -> Result<(), String> {
    let g = decl.geom();
    let want = profile.geom;
    for (what, got, want) in [
        ("hidden", g.hidden, want.hidden),
        ("layers", g.n_layers, want.n_layers),
        ("kv_heads", g.n_kv_heads, want.n_kv_heads),
        ("head_dim", g.head_dim, want.head_dim),
        ("vocab", g.vocab, want.vocab),
    ] {
        if got != 0 && got != want {
            return Err(format!(
                "{:?} 与 intent.model={:?} 不匹配:{what}={got} 期望 {want}\n\
                 —— 检查 [model].path / [site].model_dir 是否指向了别的检查点",
                decl.format_name(),
                profile.key
            ));
        }
    }
    // max_pos:按源策略(HF 硬拒 / GGUF 软警告,由调用方决定是否落进摘要)
    if g.max_pos != 0 {
        if max_seq > g.max_pos {
            match decl.max_pos_policy() {
                MaxPosPolicy::Hard => {
                    return Err(format!(
                        "max_seq={max_seq} 超过检查点 max_position_embeddings={}(模型装不下)",
                        g.max_pos
                    ))
                }
                MaxPosPolicy::WarnYarn => {} // 摘要里带"需 YaRN"提示
            }
        }
        if decl.max_pos_policy() == MaxPosPolicy::Hard && g.max_pos != want.max_pos {
            return Err(format!(
                "检查点 max_position_embeddings={} 与注册表 {:?} 的 {} 不一致 —— \
                 注册表需按该检查点重建(或换回注册表对应模型)",
                g.max_pos, profile.key, want.max_pos
            ));
        }
    }
    // full 层数(HF 由 interval 派生;GGUF 直接数层型)
    if g.n_full != 0 && g.n_full != want.n_full {
        return Err(format!(
            "n_full={} 与注册表 {:?} 的 {} 不一致 —— 几何漂移",
            g.n_full, profile.key, want.n_full
        ));
    }
    Ok(())
}

impl std::fmt::Debug for dyn CheckpointDecl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.format_name())
    }
}
