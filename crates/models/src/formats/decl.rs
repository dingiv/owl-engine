//! **检查点声明面**(`CheckpointDecl`)—— 2026-11-04 clean-code。
//!
//! # 为什么
//!
//! `--check` 原先有**两条平行的校验路径**(`check_model_dir` 读 HF
//! `config.json` / `check_gguf_model` 读 GGUF KV + `rope_facts`),
//! 两份代码各自做「几何对账 + 视觉塔声明 + 一句话摘要」,字段名与错误文案
//! 各写一遍 ⇒ 新增一个源格式 = 再抄一份校验。
//!
//! **第二刀(2026-11-04 深夜)**:两个 impl 曾经各持一套解析器(`HfDecl` 读
//! `config.json`、`GgufDecl` 读 KV),与装载器仍是两份事实。现在 impl 直接落在
//! **源的事实载体**上:
//!
//! | 声明实现 | 事实来源(与装载同一次解析) |
//! |---|---|
//! | [`HfFacts`](super::model_source::HfFacts) | `ModelSpec::from_ckpt` 的 `SpecReport` |
//! | [`GgufFacts`](super::model_source::GgufFacts) | `GgufSource` 的 `spec` + `spec_report` + `rope_facts` |
//!
//! ⇒ 第三套读取器删除;`--check` 看到的就是装载器将看到的那份事实。
//!
//! # 边界
//!
//! 本接口**只读元数据,不装权重**(纯 host 面,无设备、无 async)。
//! `--check` 构造的是**声明事实载体**(不是完整 `ModelSource`):不打开设备、
//! 不物化权重槽,也不产生装载日志。
//!
//! # 纪律
//!
//! - 事实**单源**:`DeclGeom` 由各源自己的解析器(`ModelSpec::from_ckpt` /
//!   `spec_from_gguf`)填充,本模块不重解析字节;
//! - 「超 max_pos」的处置**按源声明**(HF = 硬拒;GGUF = 软警告+YaRN 提示)
//!   —— 这是既有语义,不是本模块发明的策略;
//! - 未知事实用 `0` 表示(而不是猜),消费者按 `Option` 语义分支。

use std::path::Path;

use super::model_source::{GgufFacts, HfFacts};

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

/// **声明面的源族**(由装载臂决定,不再由 `path.is_file()` 猜)。
///
/// `--check` 与 boot 必须用**同一个**臂分派(`load_arm_of(format, quant)`):
/// 历史上按文件形态猜源 ⇒ `format=awq` + `.gguf` 路径可以"检查通过、启动失败"。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclSource {
    /// HF/AWQ/W4A16 目录(`config.json` + 索引)
    Hf,
    /// GGUF 单文件
    Gguf,
}

/// **打开检查点声明**:按 [`DeclSource`] 分派**唯一一处**。
///
/// `Ok(None)` = 该源**文件不在**(HF 缺 `config.json`)—— 与历史行为一致:
/// `--check` 只提示"几何校验跳过",不判失败(真启动会失败)。
///
/// 返回的实现是**事实载体**(`HfFacts` / `GgufFacts`):与装载路径同一份解析,
/// 但不开设备、不物化权重、不写装载日志。
pub fn open_decl(path: &Path, src: DeclSource) -> Result<Option<Box<dyn CheckpointDecl>>, String> {
    match src {
        DeclSource::Gguf => Ok(Some(Box::new(
            GgufFacts::open(path).map_err(|e| e.to_string())?,
        ))),
        DeclSource::Hf => {
            // 臂 = HF/AWQ,路径却是文件 ⇒ 与 boot 同律响亮拒(不再退化成
            // "config.json 缺失 ⇒ 跳过",那会让 `format=awq` + `.gguf` 检查通过)
            if path.is_file() {
                return Err(format!(
                    "HF/AWQ 档的 model.path 必须是**目录**(config.json + 索引),实得文件 {}",
                    path.display()
                ));
            }
            if !path.join("config.json").is_file() {
                return Ok(None);
            }
            // 声明面只需要族预设的 dtype/tokenizer 声明;几何全部来自检查点。
            let preset = super::qwen3_5_0_8b();
            Ok(Some(Box::new(
                HfFacts::open(path, &preset, (0, 1), false).map_err(|e| e.to_string())?,
            )))
        }
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
