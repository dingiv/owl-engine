//! **装载前预检**：检查点事实（`--check`；2026-11-16 业务下沉 P2）。
//!
//! ## 为什么在 engine
//!
//! `--check` 要答的问题（这个目录是不是 DFlash2？`quantization_config.format`
//! 是不是 `pack-quantized`？检查点带不带 `mtp.*`？几何对不对？）**全是装载器
//! 的同一份事实**：`load_arm_of` / `LoadIntent` / `open_decl`。规则一改（新增
//! 格式、草稿族、MTP 变体）却要在 server 同步一份 ⇒ 历史已发生
//! 「`format=awq` + `.gguf` 路径：检查通过、启动失败」。
//!
//! 所以本模块与装载**同源**：臂分派、声明解析、几何对账、草稿族判定都调引擎
//! 自己的接口；server 只拿 `(format, quant, head_quant, embed_quant, path,
//! max_context_length, vision)` 这些**配置字段**进来，把返回的文本按既有前缀打印。
//!
//! ## 接口纪律
//!
//! - **不 print、不 exit、不拼 `[check]` 前缀**：本模块只产 [`CheckItem`]，
//!   报告排版是 server 的事（`--check` / `--print-config` 的输出契约在那边）。
//! - 检查点层面的问题**收集**成 `Fail` 项（`--check` 一次看到全部问题）；
//!   只有配置级错误（臂分派失败）才走 `Err`。
//! - 字符串是**输出契约**：改动会动 `--check` 的逐字输出与 e2e 断言。

use std::path::Path;

use owl_models::formats::decl::{check_geom_against, open_decl, DeclSource, MaxPosPolicy};
use owl_shared::config::intent::ModelProfile;
use owl_shared::config::{EmbedQuant, HeadQuant, ModelFormat, QuantScheme};

use crate::{load_arm_of, LoadArm, LoadIntent};

/// 一条检查结论。
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CheckItem {
    /// 通过/提示（server 按位置加 `[check] model dir:` / `draft dir:` / `draft:` 前缀）
    Pass(String),
    /// 未通过（server 打 `[check] FAIL: …`）
    Fail(String),
}

impl CheckItem {
    pub fn ok(&self) -> bool {
        matches!(self, Self::Pass(_))
    }

    pub fn text(&self) -> &str {
        match self {
            Self::Pass(t) | Self::Fail(t) => t,
        }
    }
}

/// 一段检查的结论集合（顺序即报告顺序）。
#[derive(Clone, Debug, Default)]
pub struct CheckLines {
    pub items: Vec<CheckItem>,
}

impl CheckLines {
    pub fn pass(text: impl Into<String>) -> Self {
        Self {
            items: vec![CheckItem::Pass(text.into())],
        }
    }

    pub fn fail(text: impl Into<String>) -> Self {
        Self {
            items: vec![CheckItem::Fail(text.into())],
        }
    }

    /// 全部通过？（空集合 = 通过 —— 本模块不接受"没检查也没失败"的静默态，
    /// 调用点必须自己决定要不要产出行）
    pub fn ok(&self) -> bool {
        self.items.iter().all(CheckItem::ok)
    }

    fn push(&mut self, item: CheckItem) {
        self.items.push(item);
    }
}

/// **模型目录检查的输入**（server 只搬配置字段；臂/意图/声明解析在这里做）。
pub struct ModelCheckSpec<'a> {
    /// 模型目录或 `.gguf` 单文件
    pub path: &'a Path,
    pub format: ModelFormat,
    pub quant: QuantScheme,
    pub head_quant: HeadQuant,
    pub embed_quant: EmbedQuant,
    /// 运行时上下文（用于几何对账与 YaRN 软提示）
    pub max_context_length: usize,
    pub vision: bool,
    /// `[vision] path`（非空；`vision=false` 时给了它是配置自相矛盾）
    pub vision_path: Option<&'a Path>,
    /// 注册表目标档（`None` = 该 format 不在注册表 ⇒ 跳过逐项几何对账）
    pub profile: Option<&'a ModelProfile>,
}

/// **模型目录预检**（`--check` 的 `[check] model dir:` 段）。
///
/// `Err` 只用于**配置级**失败（装载臂分派）；检查点本身的拒绝是 `Fail` 项。
pub fn check_model_dir(spec: &ModelCheckSpec<'_>) -> Result<CheckLines, String> {
    let arm = load_arm_of(spec.format, spec.quant)
        .map_err(|e| format!("装载臂分派失败:{e}"))?;
    let intent = LoadIntent::of(arm, spec.head_quant, spec.embed_quant);
    let mut out = CheckLines::default();
    // 注册表里没有该 format ⇒ 只做声明摘要，不做逐项几何对账（仍然打开声明：
    // "不可读"本身也要报出来）
    if spec.profile.is_none() && arm.decl_source() == DeclSource::Hf {
        out.push(CheckItem::Pass("注册表无该 format ⇒ 几何校验跳过".into()));
    }
    match model_decl(spec, arm, &intent) {
        Ok(note) => out.push(CheckItem::Pass(note)),
        Err(e) => out.push(CheckItem::Fail(e)),
    }
    Ok(out)
}

/// 一条路径：臂 → 源族 → 声明事实；几何对账 / 视觉声明 / 摘要全部走
/// `CheckpointDecl` 接口（与 boot 同一份解析）。
fn model_decl(
    spec: &ModelCheckSpec<'_>,
    arm: LoadArm,
    intent: &LoadIntent,
) -> Result<String, String> {
    intent.check_arm(arm)?;
    let Some(decl) = open_decl(spec.path, arm.decl_source())? else {
        return Ok(format!(
            "{} 不可读(config.json 缺失)—— 几何校验跳过(真启动会失败)",
            spec.path.display()
        ));
    };
    if let Some(p) = spec.profile {
        check_geom_against(decl.as_ref(), p, spec.max_context_length)?;
    }
    let mut note = decl.summary();
    // 自带 MTP/NextN 头是检查点事实(同一份解析);0 = 不带,不打扰输出
    let draft = decl.draft();
    if draft.mtp_layers > 0 {
        note.push_str(&format!(" · mtp={}", draft.mtp_layers));
    }
    // max_pos 的**软**超标只在摘要里提示(HF 是硬拒,已在 check_geom_against 抛)
    let g = decl.geom();
    if decl.max_pos_policy() == MaxPosPolicy::WarnYarn && g.max_pos != 0 && spec.max_context_length > g.max_pos
    {
        note.push_str(&format!(
            "（runtime.max_context_length={} > 模型 ctx；需 YaRN）",
            spec.max_context_length
        ));
    }
    if spec.vision {
        if let Some(v) = decl.vision(spec.vision_path)? {
            note.push_str(&format!(" · vision={}", v.note));
        }
    } else if spec.vision_path.is_some() {
        return Err(format!(
            "{}:[vision] path 已设置但 [vision] enabled = false（配置自相矛盾）",
            spec.path.display()
        ));
    }
    Ok(note)
}

/// **草稿目录预检**（`--check` 的 `[check] draft dir:` 段）。
///
/// DFlash2 + W4A16 `pack-quantized`；GGUF 单文件只报张量数/架构。
pub fn check_draft_dir(dir: &Path) -> CheckLines {
    match draft_dir(dir) {
        Ok(note) => CheckLines::pass(note),
        Err(e) => CheckLines::fail(e),
    }
}

/// 草稿 `config.json` 校验(DFlash2 + W4A16 pack-quantized;BF16 源 = AL 0)
fn draft_dir(dir: &Path) -> Result<String, String> {
    // GGUF 草稿单文件(DFlash2-GGUF;2026-10-21 feat/gguf 线)
    if dir.is_file() {
        if dir.extension().and_then(|s| s.to_str()) != Some("gguf") {
            return Err(format!("草稿 {} 既不是目录也不是 .gguf 单文件", dir.display()));
        }
        let g = owl_models::formats::gguf::Gguf::open(dir)
            .map_err(|e| format!("打开草稿 GGUF {} 失败: {e}", dir.display()))?;
        return Ok(format!(
            "{} ✓ GGUF 草稿 / {} 张量(arch={})",
            dir.display(),
            g.tensors.len(),
            g.arch().map(|a| a.to_string()).unwrap_or_else(|_| "?".into())
        ));
    }
    let p = dir.join("config.json");
    if !p.is_file() {
        return Ok(format!(
            "{} 不可读(config.json 缺失)—— 校验跳过(真启动会失败)",
            dir.display()
        ));
    }
    let raw = owl_shared::file_loader::read_to_string(&p)
        .map_err(|e| format!("读取 {} 失败: {e}", p.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("{} 不是合法 JSON: {e}", p.display()))?;
    let arch: Vec<String> = v
        .get("architectures")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    if !arch.iter().any(|a| a.contains("DFlash2")) {
        return Err(format!(
            "草稿目录 {} 的 architectures={arch:?} 不是 DFlash2 —— 检查 [site].draft_dir",
            dir.display()
        ));
    }
    let fmt = v
        .get("quantization_config")
        .and_then(|q| q.get("format"))
        .and_then(|x| x.as_str());
    if fmt != Some("pack-quantized") {
        return Err(format!(
            "草稿目录 {} 不是 W4A16 pack-quantized(quantization_config.format={fmt:?})\n\
             —— BF16 源(z-lab 系)喂引擎 = 草稿全噪声 AL=0;生产用 syvai W4A16 系",
            dir.display()
        ));
    }
    let layers = v.get("num_hidden_layers").and_then(|x| x.as_u64());
    Ok(format!(
        "{} ✓ DFlash2 / pack-quantized{}",
        dir.display(),
        layers.map(|n| format!(" / {n} 层")).unwrap_or_default()
    ))
}

/// **MTP 头预检**（`--check` 的 `[check] draft:` 段；`spec.draft=mtp`）。
///
/// 目录看权重索引的 `mtp.*` 键；GGUF 单文件看 `blk.<n>.nextn.eh_proj.*`。
pub fn check_mtp_head(dir: &Path) -> CheckLines {
    match mtp_head(dir) {
        Ok(note) => CheckLines::pass(note),
        Err(e) => CheckLines::fail(e),
    }
}

fn mtp_head(dir: &Path) -> Result<String, String> {
    // GGUF 单文件:NextN 块 = `blk.<n>.nextn.eh_proj.weight`
    if dir.is_file() {
        if dir.extension().and_then(|s| s.to_str()) != Some("gguf") {
            return Err(format!(
                "MTP 头 {} 既不是目录也不是 .gguf 单文件",
                dir.display()
            ));
        }
        let g = owl_models::formats::gguf::Gguf::open(dir)
            .map_err(|e| format!("打开 GGUF {} 失败: {e}", dir.display()))?;
        let n = g
            .tensors
            .iter()
            .filter(|t| t.name.starts_with("blk.") && t.name.contains(".nextn.eh_proj."))
            .count();
        if n == 0 {
            return Err(format!(
                "spec.draft=mtp 但 {} 没有 blk.<n>.nextn.eh_proj.*（该 GGUF 不带 MTP 头）",
                dir.display()
            ));
        }
        return Ok(format!("{} ✓ GGUF MTP / {} 个 NextN 块", dir.display(), n));
    }
    let p = dir.join("model.safetensors.index.json");
    if !p.is_file() {
        return Ok(format!(
            "{} 无 model.safetensors.index.json —— 跳过 mtp.* 预检(装载期验证)",
            dir.display()
        ));
    }
    let raw = owl_shared::file_loader::read_to_string(&p)
        .map_err(|e| format!("读取 {} 失败: {e}", p.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)
        .map_err(|e| format!("{} 不是合法 JSON: {e}", p.display()))?;
    let n = v
        .get("weight_map")
        .and_then(|x| x.as_object())
        .map(|m| m.keys().filter(|k| k.starts_with("mtp.")).count())
        .unwrap_or(0);
    if n == 0 {
        return Err(format!(
            "spec.draft=mtp 但 {} 的权重索引没有 mtp.* 键（该检查点不是 MTP 档；请改用 dflash2 或关 spec）",
            dir.display()
        ));
    }
    Ok(format!("{} ✓ MTP 头 / {} 个 mtp.* 键", dir.display(), n))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!(
            "owl-preflight-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).expect("建临时目录");
        p
    }

    /// 缺 `config.json` = 提示跳过（不是 FAIL）：真启动会失败，但 `--check`
    /// 对假路径/CI 必须能跑。
    #[test]
    fn missing_config_json_is_a_hint_not_a_failure() {
        let dir = tmp_dir("missing-decl");
        let lines = check_model_dir(&ModelCheckSpec {
            path: &dir,
            format: ModelFormat::Safetensors,
            quant: QuantScheme::W4a16,
            head_quant: HeadQuant::Off,
            embed_quant: EmbedQuant::F16,
            max_context_length: 8192,
            vision: false,
            vision_path: None,
            profile: None,
        })
        .expect("臂分派");
        assert!(lines.ok());
        assert_eq!(lines.items.len(), 2, "{:?}", lines.items);
        // 第一行是"注册表无该 format"跳过硬答，第二行才是声明不可读
        assert!(
            lines
                .items
                .iter()
                .any(|i| i.text().contains("不可读")),
            "{:?}",
            lines.items
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// BF16 草稿（z-lab 系）= FAIL，文案要指出该换成 W4A16。
    #[test]
    fn bf16_draft_is_rejected_with_actionable_text() {
        let dir = tmp_dir("bf16-draft");
        std::fs::write(
            dir.join("config.json"),
            r#"{"architectures":["DFlash2ForCausalLM"],"quantization_config":{"format":"bf16"}}"#,
        )
        .expect("写 config.json");
        let lines = check_draft_dir(&dir);
        assert!(!lines.ok());
        let t = lines.items[0].text();
        assert!(t.contains("pack-quantized"), "{t}");
        assert!(t.contains("W4A16"), "{t}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 没有权重索引 = 提示跳过；有 `mtp.*` 键 = 通过。
    #[test]
    fn mtp_head_counts_index_keys() {
        let dir = tmp_dir("mtp");
        let lines = check_mtp_head(&dir);
        assert!(lines.ok());
        assert!(lines.items[0].text().contains("跳过 mtp.* 预检"));

        std::fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"mtp.fc.weight":"a.safetensors","model.embed.weight":"a.safetensors"}}"#,
        )
        .expect("写索引");
        let lines = check_mtp_head(&dir);
        assert!(lines.ok());
        assert!(lines.items[0].text().contains("1 个 mtp.* 键"), "{:?}", lines.items);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 非 `.gguf` 单文件不做 MTP 预检（响亮拒，不静默跳过）。
    #[test]
    fn non_gguf_file_is_rejected() {
        let dir = tmp_dir("plain-file");
        let f = dir.join("model.bin");
        std::fs::write(&f, b"x").expect("写文件");
        let lines = check_mtp_head(&f);
        assert!(!lines.ok());
        assert!(lines.items[0].text().contains("既不是目录也不是 .gguf"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
