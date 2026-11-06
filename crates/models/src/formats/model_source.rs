//! **模型源统一个口**(`ModelSource`)—— 2026-11-04 clean-code 第三步。
//!
//! # 为什么
//!
//! 装载器原先有六个 `load_*` 入口,每个都把同一串事实重新算一遍:
//! 维度(spec) · 键约定 · 计划(单值/逐键) · 词表 · rope · 视觉塔 · 草稿头。
//! 病史两条,都是"流程里各写一份"造的:
//! - GGUF 档的 `head_quant`/`embed_quant` 曾被**静默忽略**;
//! - DFlash2 草稿几何曾用目标 env 解析 ⇒ 静默错几何。
//!
//! 这里把**源能回答的事实**收成一个接口;装载流(`ModelLoader::load_source`)
//! 只认接口,不再认识格式。两个实现:
//!
//! | 实现 | 覆盖 | 事实来源 |
//! |---|---|---|
//! | [`HfDirSource`] | HF/AWQ/W4A16 **目录** | `config.json` + 索引(`ModelSpec::from_ckpt`)+ 目录 tokenizer |
//! | [`GgufFileSource`] | GGUF **单文件** | 文件 KV(`spec_from_gguf` / `rope_facts`)+ 内嵌词表 |
//!
//! **声明面同源**:`HfFacts` / `GgufFacts` 就是 `--check` 的 `CheckpointDecl`
//! 实现(几何/max_pos/视觉/MTP 全来自**同一次解析**),不再有第三套读取器。
//!
//! # 边界(纪律)
//!
//! - 本模块**纯 host**:不 alloc 设备块、不 eval;设备面只出现在
//!   [`crate::formats::load::load_component`](super::load::load_component) 之后。
//! - **计划不压平**:GGUF 的逐张量计划(`PlanSource::Keyed`)在
//!   [`ModelSource::build_model`] 内部就地构造、就地消费 —— 表生命周期不出
//!   本函数,`LoadedModel.quant_plan = None` 的语义因此天然成立。
//! - **能力由接口回答**:源不认的旋钮(如 GGUF 的 head/embed 覆盖)在这里
//!   **响亮拒**,不再靠调用方记得检查。
//! - 维度事实**已按本 rank 分片**([`shard_spec`] 是唯一换算点)。

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::contract::ModelError;
use crate::formats::decl::{CheckpointDecl, DeclGeom, DraftDecl, MaxPosPolicy, VisionDecl};
use crate::layers::rope::Rope;
use crate::model::{Model, ModelSpec};
use crate::module::{PlanSource, QuantPlan, SourceFacts, WeightSource};
use crate::tokenizer::Tokenizer;

use super::gguf::{
    build_vocab, rope_facts, Gguf, GgufConvention, GgufPlan, GgufRopeFacts, GgufSource, ShardSel,
};
use super::qwen35::{load_tokenizer, Qwen35Convention};
use super::spec::{SpecReport, SpecSource};
use super::{AwqSource, SafeTensorsSource, W4A16Source};

// ============================================================================
// §1 事实载体
// ============================================================================

/// 主/头/表三个计划(装载意图 → 源;**唯一换算入口**在引擎 `LoadIntent`)。
#[derive(Debug, Clone, Copy)]
pub struct PlanIntent {
    pub quant: QuantPlan,
    pub head: QuantPlan,
    pub embed: Option<QuantPlan>,
}

/// 源造好的主干模型(+ 装配面要的量化真值)。
pub struct SourceModel {
    pub model: Model,
    /// `Some` = 单值计划(HF 族);`None` = **逐张量**(GGUF —— 单一值表达不了)
    pub quant_plan: Option<QuantPlan>,
}

/// rope 三件套(HF 族;头维恒取 `spec.full_heads.2`)
#[derive(Debug, Clone, Copy)]
pub struct RopeFacts {
    pub max_pos: usize,
    pub rotary_dim: usize,
    pub theta: f32,
}

/// HF 目录的草稿策略(两族历史行为不同,原样保留)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftPolicy {
    /// 目录**无条件**交给 spec 运行时(27B AWQ:潜在 `mtp.*`)
    Directory,
    /// 探索引 `mtp.*`(safetensors;探针失败按"无头"处理,不阻断装载)
    Probe,
    /// 不提供
    Off,
}

/// HF 目录源族(AWQ / safetensors / W4A16)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfKind {
    SafeTensors,
    Awq,
    W4A16,
}

impl HfKind {
    pub const fn name(self) -> &'static str {
        match self {
            HfKind::SafeTensors => "safetensors",
            HfKind::Awq => "awq",
            HfKind::W4A16 => "w4a16",
        }
    }
}

/// `HfDirSource::open` 的入参(配置/引擎裁决显式传入,源不读全局)。
pub struct HfSourceCfg<'a> {
    /// 家族声明基座(非维度事实:dtype + tokenizer 声明)
    pub preset: &'a ModelSpec,
    pub kind: HfKind,
    /// `[spec]` 日志 tag(与既有输出保真)
    pub log_tag: &'static str,
    /// tokenizer 目录(`None` = 检查点目录;W4A16 开发档可不同)
    pub tokenizer_dir: Option<&'a Path>,
    /// `(slice_id, shard)`
    pub shard: (usize, usize),
    /// `runtime.tp_vocab_split`
    pub vocab_split: bool,
    pub rope: RopeFacts,
    pub draft: DraftPolicy,
    /// `[vision] enabled`
    pub vision: bool,
    /// `[vision] path`(非 GGUF 档给了 = 配置自相矛盾 ⇒ 拒)
    pub mmproj: Option<&'a Path>,
    /// 设备重排裁决(仅 AWQ 消费)
    pub device_repack: bool,
}

/// 词表装载结果(附带原始词表大小 —— GGUF 事实日志用)。
pub struct LoadedTokenizer {
    pub tokenizer: Tokenizer,
    pub n_tokens: usize,
}

/// 视觉塔来源(设备面装载仍由引擎的 loader 做)
pub enum VisionSource<'a> {
    /// 检查点目录自带 `model.visual.*`
    InCkpt,
    /// GGUF 独立 `mmproj-*.gguf`(文本容器由源持有)
    Mmproj(&'a Gguf),
}

/// 装载分相(观测面;GGUF 的 `[gguf] 装载分相` 由源自己打印)。
#[derive(Debug, Clone, Copy)]
pub struct LoadPhases {
    /// 源 open(容器/索引/维度事实)
    pub open: Duration,
    /// `build_model`(计划 + 构造)
    pub plan: Duration,
    /// 权重装载
    pub weights: Duration,
    /// 视觉塔
    pub vision: Duration,
    /// 全程(含 open;不含源自身释放)
    pub total: Duration,
}

// ============================================================================
// §2 接口
// ============================================================================

/// **一个源 = 声明 + 字节流 + 附属件**(纯 host;设备面在其后)。
pub trait ModelSource {
    /// 诊断名(`safetensors` / `awq` / `w4a16` / `gguf`)
    fn name(&self) -> &'static str;
    /// 源根(目录或单文件;日志/视觉装载用)
    fn root(&self) -> &Path;
    /// 维度事实(**已按本 rank 分片**)
    fn spec(&self) -> &ModelSpec;
    /// 造主干模型:计划换算 + 能力校验 + `Model::new*` 的唯一落点
    fn build_model(&self, intent: &PlanIntent) -> Result<SourceModel, ModelError>;
    /// 权重字节源(源自持)
    fn weights(&self) -> &dyn WeightSource;
    /// 词表(目录 / 内嵌)
    fn tokenizer(&self) -> Result<LoadedTokenizer, ModelError>;
    /// rope 表(HF 用自身事实;GGUF 吃 `[model.rope]` 覆盖)
    fn rope(&self, cfg: &owl_shared::config::RopeCfg, max_seq: usize) -> Result<Rope, ModelError>;
    /// 视觉塔来源(`None` = 纯文本)
    fn vision(&self) -> Option<VisionSource<'_>>;
    /// 草稿头所在(目录或单文件;`None` = 检查点不带)
    fn draft_dir(&self) -> Option<PathBuf>;
    /// 设备重排裁决(仅量化源为 true)
    fn device_repack(&self) -> bool {
        false
    }
    /// 源 open 耗时(分相日志用)
    fn open_duration(&self) -> Duration {
        Duration::ZERO
    }
    /// 源事实摘要日志(默认无;GGUF 打印模型/rope 一行)
    fn log_facts(&self, _rope: &Rope, _n_tokens: usize, _cfg: &owl_shared::config::RopeCfg) {}
    /// 装载分相日志(默认无;GGUF 有)
    fn log_phases(&self, _phases: &LoadPhases) {}
}

// ============================================================================
// §3 HF / AWQ / W4A16 目录源
// ============================================================================

/// **目录声明事实**(`--check` 与装载共用的同一次解析)。
///
/// 从 `config.json` + 索引推出:维度/键基座/max_pos/视觉声明/MTP 层数;
/// `CheckpointDecl` 直接实现给它 —— `--check` 不再有第二套读取器。
pub struct HfFacts {
    dir: PathBuf,
    /// 本 rank 分片后的维度(单卡检查 = 恒等)
    spec: ModelSpec,
    /// 起点读取账(键基座/失配/max_pos/视觉位/MTP 层数)
    report: SpecReport,
    /// 声明面几何(从 spec + report 派生一次)
    geom: DeclGeom,
}

impl HfFacts {
    /// 打开目录并读声明事实(**零日志、零设备**;`--check` 与装载同源)。
    pub(crate) fn open(
        dir: &Path,
        preset: &ModelSpec,
        shard: (usize, usize),
        vocab_split: bool,
    ) -> Result<Self, ModelError> {
        let (spec, report) = ModelSpec::from_ckpt(dir, SpecSource::family(preset))?;
        let spec = shard_spec(spec, shard, vocab_split)?;
        let geom = DeclGeom {
            hidden: spec.hidden,
            n_layers: spec.layer_types.len(),
            n_full: spec.layer_types.iter().filter(|f| **f).count(),
            n_kv_heads: spec.full_heads.1,
            head_dim: spec.full_heads.2,
            vocab: spec.vocab,
            max_pos: report.max_pos,
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            spec,
            report,
            geom,
        })
    }
}

impl CheckpointDecl for HfFacts {
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
        if !self.report.has_vision_cfg {
            return Err(format!(
                "[vision] enabled = true,但 {} 的 config.json 无 vision_config(不是 VLM 检查点)",
                self.dir.display()
            ));
        }
        if !self.report.has_image_token {
            return Err(format!(
                "[vision] enabled = true,但 {} 的 config.json 缺 image_token_id",
                self.dir.display()
            ));
        }
        Ok(Some(VisionDecl {
            note: "model.visual.*".into(),
        }))
    }
    fn draft(&self) -> DraftDecl {
        // 键面事实(`mtp.layers.<n>.*` 去重计数);装载期的 `draft_dir`
        // 另按家族策略(27B AWQ 历史语义 = 目录无条件交给 spec 运行时)
        DraftDecl {
            mtp_layers: self.report.mtp_layers,
        }
    }
}

/// 权重源三族(目录档;只在 open 时选一次)
enum HfWeights {
    St(SafeTensorsSource),
    Awq(AwqSource),
    W4a16(W4A16Source),
}

impl HfWeights {
    fn as_dyn(&self) -> &dyn WeightSource {
        match self {
            HfWeights::St(s) => s,
            HfWeights::Awq(s) => s,
            HfWeights::W4a16(s) => s,
        }
    }
}

/// HF/AWQ/W4A16 **目录**源(声明 = [`HfFacts`] + 权重字节源)。
pub struct HfDirSource {
    kind: HfKind,
    facts: HfFacts,
    tokenizer_dir: PathBuf,
    weights: HfWeights,
    rope: RopeFacts,
    vision: bool,
    draft: Option<PathBuf>,
    device_repack: bool,
    open: Duration,
}

impl HfDirSource {
    /// 打开目录并抬出全部声明事实(维度/键基座/词表/rope/视觉/草稿)。
    ///
    /// 设备重排裁决由调用方给(引擎按配置 + 能力自述解析)——
    /// 运算源在**首键物化前**必须知道这条,故在 open 期注入。
    pub fn open(dir: &Path, cfg: HfSourceCfg<'_>) -> Result<Self, ModelError> {
        let t0 = Instant::now();
        // `[vision] path` 只对 GGUF 单文件有意义(HF 的塔在检查点目录里)
        if cfg.mmproj.is_some() {
            return Err(ModelError::Msg(
                "[vision] path 仅用于 format = \"gguf\"：HF/AWQ 档的视觉塔权重在检查点目录的 model.visual.*"
                    .into(),
            ));
        }
        let facts = HfFacts::open(dir, cfg.preset, cfg.shard, cfg.vocab_split)?;
        log_spec_report(cfg.log_tag, &facts.report);
        let mut draft = None;
        let weights = match cfg.kind {
            HfKind::SafeTensors => {
                let src = SafeTensorsSource::open_dir(dir)?;
                // 同一目录自带 MTP 头 ⇒ 当草稿源(与 GGUF 探 `blk.N.nextn.*` 同律)
                if cfg.draft == DraftPolicy::Probe && matches!(src.has_mtp_head(), Ok(true)) {
                    owl_shared::log_info!(
                        "boot",
                        "检查点自带 MTP/NextN 头(mtp.*):spec.draft=mtp 时由同一目录装载"
                    );
                    draft = Some(dir.to_path_buf());
                }
                HfWeights::St(src)
            }
            HfKind::Awq => {
                let src = AwqSource::open_dir(dir)?;
                src.set_device_repack(cfg.device_repack);
                if cfg.draft == DraftPolicy::Directory {
                    // 27B 档历史语义:目录**无条件**交给 spec 运行时
                    draft = Some(dir.to_path_buf());
                }
                HfWeights::Awq(src)
            }
            HfKind::W4A16 => {
                let src = W4A16Source::open_dir(dir)?;
                // 与历史一致:0.8B 走 CPU 两步重排(量小,不值 GPU 往返)
                src.set_device_repack(false);
                HfWeights::W4a16(src)
            }
        };
        Ok(Self {
            kind: cfg.kind,
            facts,
            tokenizer_dir: cfg.tokenizer_dir.unwrap_or(dir).to_path_buf(),
            weights,
            rope: cfg.rope,
            vision: cfg.vision,
            draft,
            device_repack: cfg.device_repack,
            open: t0.elapsed(),
        })
    }
}

impl ModelSource for HfDirSource {
    fn name(&self) -> &'static str {
        self.kind.name()
    }

    fn root(&self) -> &Path {
        &self.facts.dir
    }

    fn spec(&self) -> &ModelSpec {
        &self.facts.spec
    }

    fn build_model(&self, intent: &PlanIntent) -> Result<SourceModel, ModelError> {
        if self.kind == HfKind::Awq {
            // 主干量化 plan:27B 档只认 W4A16Awq / W4A8Awq(其余 = 静默降级到
            // 全 f16,正是本仓禁止的那类错)
            match intent.quant {
                QuantPlan::W4A16Awq => {}
                QuantPlan::W4A8Awq => {
                    owl_shared::log_info!(
                        "boot",
                        "W4A8 档位开(model.quant = \"w4a8\"):权重 a8 布局 + 激活 per-token int8 + kS8×kU4 GEMM"
                    );
                }
                other => {
                    return Err(ModelError::Msg(format!(
                        "27B-AWQ 主干量化 plan 非法: {other:?} —— 只接受 W4A16Awq / W4A8Awq\
                         (engine::quant_plan_of 是唯一映射点)"
                    )));
                }
            }
        }
        // 键约定基座是**字符串前缀**(带尾点);约定无尾点
        let conv = Qwen35Convention::new(self.facts.report.keys_base.trim_end_matches('.'));
        let model = Model::new_with_embed(
            &self.facts.spec,
            conv,
            intent.quant,
            intent.head,
            intent.embed,
        )?;
        Ok(SourceModel {
            model,
            quant_plan: Some(intent.quant),
        })
    }

    fn weights(&self) -> &dyn WeightSource {
        self.weights.as_dyn()
    }

    fn tokenizer(&self) -> Result<LoadedTokenizer, ModelError> {
        let tokenizer = load_tokenizer(&self.tokenizer_dir, &self.facts.spec.tokenizer)?;
        Ok(LoadedTokenizer {
            tokenizer,
            n_tokens: self.facts.spec.vocab,
        })
    }

    fn rope(&self, _cfg: &owl_shared::config::RopeCfg, max_seq: usize) -> Result<Rope, ModelError> {
        build_hf_rope(self.rope, self.facts.spec.full_heads.2, max_seq)
    }

    fn vision(&self) -> Option<VisionSource<'_>> {
        self.vision.then_some(VisionSource::InCkpt)
    }

    fn draft_dir(&self) -> Option<PathBuf> {
        self.draft.clone()
    }

    fn device_repack(&self) -> bool {
        self.device_repack
    }

    fn open_duration(&self) -> Duration {
        self.open
    }
}

// ============================================================================
// §4 GGUF 单文件源
// ============================================================================

/// **GGUF 声明事实**(`--check` 与装载共用的同一次解析)。
///
/// 持有权重源(容器已开)⇒ 视觉声明不必二次开文本文件;几何/max_pos/arch/MTP
/// 全部来自 `GgufSource` 的同一次 `spec_from_gguf`。
pub struct GgufFacts {
    path: PathBuf,
    weights: GgufSource,
    /// 未分片的检查点全量维度(声明面对账用)
    spec: ModelSpec,
    geom: DeclGeom,
    arch: String,
    rf: GgufRopeFacts,
    /// 首个 NextN 块号(日志用)
    mtp_first: Option<usize>,
    mtp_layers: usize,
}

impl GgufFacts {
    /// 打开单文件并读声明事实(**零日志、零设备**;`--check` 用)。
    pub fn open(path: &Path) -> Result<Self, ModelError> {
        let weights = GgufSource::open(path)?;
        Self::from_source(path, weights)
    }

    fn from_source(path: &Path, weights: GgufSource) -> Result<Self, ModelError> {
        let spec = weights.spec().clone();
        let arch = weights.spec_report().arch.clone();
        let rf = rope_facts(weights.gguf())?;
        let (mtp_first, mtp_layers) = nextn_blocks(weights.gguf(), spec.layer_types.len());
        let geom = DeclGeom {
            hidden: spec.hidden,
            n_layers: spec.layer_types.len(),
            n_full: spec.layer_types.iter().filter(|f| **f).count(),
            n_kv_heads: spec.full_heads.1,
            head_dim: spec.full_heads.2,
            vocab: spec.vocab,
            max_pos: rf.max_pos,
        };
        Ok(Self {
            path: path.to_path_buf(),
            weights,
            spec,
            geom,
            arch,
            rf,
            mtp_first,
            mtp_layers,
        })
    }
}

impl CheckpointDecl for GgufFacts {
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
        // `--check` 面:path 缺席且同目录无候选 = 无塔(装载期才会拒)
        let Ok(mm) = super::gguf::resolve_mmproj_path(&self.path, vision_path) else {
            return Ok(None);
        };
        let mg = Gguf::open(&mm).map_err(|e| format!("打开 mmproj {} 失败: {e}", mm.display()))?;
        let vcfg = crate::vlm::VlmCfg::from_gguf(self.weights.gguf(), &mg, self.spec.clone())
            .map_err(|e| e.to_string())?;
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
        DraftDecl {
            mtp_layers: self.mtp_layers,
        }
    }
}

/// GGUF **单文件**源(声明 = 文件 KV;权重 = 块字节,零解量化)。
pub struct GgufFileSource {
    path: PathBuf,
    facts: GgufFacts,
    /// 本 rank 分片后的维度(装载面;`facts.spec` 保持全量给声明面)
    spec: ModelSpec,
    vision: bool,
    open: Duration,
}

impl GgufFileSource {
    /// 打开单文件(`vision` = `[vision] enabled`;`mmproj_given` 只用于
    /// 配置自洽校验 —— 路径解析仍由引擎按 `[vision] path` 做)。
    pub fn open(
        path: &Path,
        shard: (usize, usize),
        vocab_split: bool,
        vision: bool,
        mmproj_given: bool,
    ) -> Result<Self, ModelError> {
        let t0 = Instant::now();
        if !path.is_file() {
            return Err(ModelError::Msg(format!(
                "GGUF 档:model.dir 必须是 **.gguf 单文件**(实得 {})",
                path.display()
            )));
        }
        if !vision && mmproj_given {
            return Err(ModelError::Msg(
                "[vision] path 已设置但 [vision] enabled = false（配置自相矛盾）".into(),
            ));
        }
        // 源自担分片(块对齐列切 + v-head grouped 聚合;开箱即逐键验可分片性)
        let sel = (shard.1 > 1).then_some(ShardSel {
            id: shard.0,
            world: shard.1,
            head_split: vocab_split,
        });
        let weights = GgufSource::open_shard(path, sel)?;
        let facts = GgufFacts::from_source(path, weights)?;
        log_gguf_report(path, facts.weights.spec_report());
        let spec = shard_spec(facts.spec.clone(), shard, vocab_split)?;
        if let Some(b) = facts.mtp_first {
            owl_shared::log_info!(
                "boot",
                "GGUF 自带 MTP/NextN 块:blk.{b}.nextn.*（spec.draft=mtp 时由同一文件装载）"
            );
        }
        Ok(Self {
            path: path.to_path_buf(),
            facts,
            spec,
            vision,
            open: t0.elapsed(),
        })
    }
}

impl ModelSource for GgufFileSource {
    fn name(&self) -> &'static str {
        "gguf"
    }

    fn root(&self) -> &Path {
        &self.path
    }

    fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    fn build_model(&self, intent: &PlanIntent) -> Result<SourceModel, ModelError> {
        // 计划是**逐张量**事实 ⇒ head/embed 覆盖无意义(给了 = 配置在说谎)
        check_keyed_overrides(intent.head, intent.embed)?;
        let sel = self.facts.weights.shard();
        let plan = match sel {
            Some(s) => {
                GgufPlan::for_shard(self.facts.weights.gguf(), *self.facts.weights.dims(), s)
            }
            None => GgufPlan::new(self.facts.weights.gguf()),
        };
        // 源事实:TP 档 = 权重内维已是 grouped 序 ⇒ 不要 A2 表
        let facts = if sel.is_some() {
            SourceFacts::GGUF_TP
        } else {
            SourceFacts::GGUF
        };
        let conv = GgufConvention::new();
        let ps = PlanSource::keyed(&plan, &conv, facts);
        // 计划表生命周期不出本函数:能否压平成单一值是**语义**问题,不是风格
        let model = Model::new(&self.spec, GgufConvention::new(), ps, ps)?;
        Ok(SourceModel {
            model,
            quant_plan: None,
        })
    }

    fn weights(&self) -> &dyn WeightSource {
        &self.facts.weights
    }

    fn tokenizer(&self) -> Result<LoadedTokenizer, ModelError> {
        let vocab = build_vocab(self.facts.weights.gguf())?;
        let n_tokens = vocab.n_tokens;
        let eos = vocab.eos_ids();
        let fmt = self.spec.tokenizer.chat.clone();
        Ok(LoadedTokenizer {
            tokenizer: Tokenizer::from_parts(vocab.inner, eos, fmt),
            n_tokens,
        })
    }

    fn rope(&self, cfg: &owl_shared::config::RopeCfg, max_seq: usize) -> Result<Rope, ModelError> {
        build_gguf_rope(&self.facts.rf, cfg, max_seq, self.spec.full_heads.2)
    }

    fn vision(&self) -> Option<VisionSource<'_>> {
        self.vision
            .then(|| VisionSource::Mmproj(self.facts.weights.gguf()))
    }

    fn draft_dir(&self) -> Option<PathBuf> {
        (self.facts.mtp_layers > 0).then(|| self.path.clone())
    }

    fn open_duration(&self) -> Duration {
        self.open
    }

    fn log_facts(&self, rope: &Rope, n_tokens: usize, cfg: &owl_shared::config::RopeCfg) {
        let spec = &self.spec;
        let rf = &self.facts.rf;
        owl_shared::log_info!(
            "boot",
            "GGUF:{} · 层 {}(full {} / gdn {}) · 词表 {} · rope(rot {} / θ {:.3e} / max {} / 表 {} / yarn ×{}) · tied {}",
            self.path.display(),
            spec.layer_types.len(),
            spec.layer_types.iter().filter(|f| **f).count(),
            spec.layer_types.iter().filter(|f| !**f).count(),
            n_tokens,
            rf.rotary_dim,
            rf.theta,
            rf.max_pos,
            rope.max_pos(),
            if cfg.yarn_on() { cfg.yarn_factor } else { 1.0 },
            spec.tied,
        );
    }

    fn log_phases(&self, p: &LoadPhases) {
        let rest = p
            .total
            .saturating_sub(p.open)
            .saturating_sub(p.plan)
            .saturating_sub(p.weights)
            .saturating_sub(p.vision);
        owl_shared::log_info!(
            "gguf",
            "装载分相:容器+维度 {:.2}s + 计划/构造 {:.2}s + 权重 {:.2}s + 视觉 {:.2}s + 词表/rope {:.2}s = {:.2}s",
            p.open.as_secs_f32(),
            p.plan.as_secs_f32(),
            p.weights.as_secs_f32(),
            p.vision.as_secs_f32(),
            rest.as_secs_f32(),
            p.total.as_secs_f32()
        );
    }
}

// ============================================================================
// §5 共享助手(事实换算 / 能力门 / 日志 / rope)
// ============================================================================

/// **逐张量计划源不接受 head/embed 覆盖**(GGUF)——装载器与 `--check` 共用。
///
/// 为什么单独成函数:这条能力门有两个消费者(`GgufFileSource::build_model`
/// 与 server 的 `--check`);两处各写一份文案必然漂移(历史 bug:GGUF 档
/// `head_quant` 曾被静默忽略)。返回 `ModelError` 便于装载面直接上抛。
pub fn check_keyed_overrides(head: QuantPlan, embed: Option<QuantPlan>) -> Result<(), ModelError> {
    if head != QuantPlan::F16 || embed.is_some() {
        return Err(ModelError::Msg(format!(
            "GGUF 装载的 lm_head/embedding 计划由文件逐张量决定,\
             不接受覆盖:请把 [model] head_quant 保持 \"off\"、embed_quant 保持 \"f16\"\
             (当前 head={:?} embed={:?},来源 arm=gguf)",
            head, embed
        )));
    }
    Ok(())
}

/// **HF 档 rope 表构造**(事实硬编码,无 YaRN;表长必须覆盖 turn 预算)。
///
/// HF 家族档的 rope 事实由装载器显式给(family 参数),没有 GGUF 的
/// `[model.rope]` 覆盖面 ⇒ `max_seq` 超表长时**响亮拒**(绝不静默外推:
/// `owl_rope_*` 核按 `pos` 直接索引 cos/sin 表,越界 = 设备 OOB 读)。
fn build_hf_rope(facts: RopeFacts, head_dim: usize, max_seq: usize) -> Result<Rope, ModelError> {
    if max_seq > facts.max_pos {
        return Err(ModelError::Msg(format!(
            "max_seq={max_seq} > HF rope 表长={}(HF 档 rope 事实硬编码、不支持 YaRN;\
             请降 [runtime] max_seq 或改用 GGUF 档 + [model.rope] yarn_factor=…)",
            facts.max_pos
        )));
    }
    Rope::new(facts.max_pos, head_dim, facts.rotary_dim, facts.theta)
}

/// **形状分片换算**(唯一产地;`shard = 1` 恒等)。
///
/// 词表切(runtime.tp_vocab_split)与分片的组合门在这里:单卡没有"每 rank
/// 半表",静默忽略会让配置说谎。
pub fn shard_spec(
    mut spec: ModelSpec,
    (id, n): (usize, usize),
    vocab_split: bool,
) -> Result<ModelSpec, ModelError> {
    if vocab_split && n <= 1 {
        return Err(ModelError::Msg(
            "词表切(runtime.tp_vocab_split)需要 shard ≥ 2:单卡全量头应关本档".into(),
        ));
    }
    spec.head_split = vocab_split;
    if n <= 1 {
        return Ok(spec);
    }
    spec.sharded(crate::plan::tp::TpPlan::new(id, n).map_err(ModelError::Msg)?)
        .map_err(ModelError::Msg)
}

/// NextN 块扫描(`blk.<n>.nextn.*`):返回 (首个块号, 块数)。
fn nextn_blocks(g: &Gguf, n_layers: usize) -> (Option<usize>, usize) {
    let mut first = None;
    let mut n = 0usize;
    for i in 0..=n_layers {
        if g.index_of(&format!("blk.{i}.nextn.eh_proj.weight"))
            .is_some()
        {
            first.get_or_insert(i);
            n += 1;
        }
    }
    (first, n)
}

/// 起点读取账(响亮:键基座 + config↔形状失配逐条上屏)
fn log_spec_report(tag: &str, rep: &SpecReport) {
    owl_shared::log_info!(
        "boot",
        "[spec] {tag}: keys_base={:?} 失配 {} 条",
        rep.keys_base,
        rep.mismatches.len()
    );
    for m in &rep.mismatches {
        owl_shared::log_info!("boot", "[spec] {tag} · {m}");
    }
}

/// GGUF 读取账(与 HF 路径同律:宽松处理过的项逐条上屏)
fn log_gguf_report(path: &Path, rep: &super::gguf::GgufSpecReport) {
    owl_shared::log_info!(
        "boot",
        "[gguf] {}: arch={} · 宽松项 {}",
        path.display(),
        rep.arch,
        rep.issues.len()
    );
    for m in &rep.issues {
        owl_shared::log_info!("boot", "[gguf] · {m}");
    }
}

/// **GGUF rope 表构造**(单一产地;无 YaRN 时与历史逐位相同)。
///
/// 表长 = `max(模型 context, original×factor, max_seq)` ——
/// YaRN 的官方缓存长 = `original × factor`;我们再多看一眼 `runtime.max_seq`
/// (用户要 1M 而 factor 只给 2.0 时**响亮拒**,不静默外推)。
/// 未开 YaRN 且 `max_seq > 模型 context` ⇒ **响亮拒**(纯外推没有官方背书)。
fn build_gguf_rope(
    rf: &GgufRopeFacts,
    cfg: &owl_shared::config::RopeCfg,
    max_seq: usize,
    head_dim: usize,
) -> Result<Rope, ModelError> {
    if !cfg.yarn_on() {
        return if max_seq > rf.max_pos {
            Err(ModelError::Msg(format!(
                "max_seq={max_seq} > 模型 context={}(未开 YaRN):要么把 [runtime] max_seq ≤ {},要么开 [model.rope] yarn_factor=…(官方:2.0→524k / 4.0→1M)",
                rf.max_pos, rf.max_pos
            )))
        } else {
            Rope::new(rf.max_pos, head_dim, rf.rotary_dim, rf.theta)
        };
    }
    let original = if cfg.original_max == 0 {
        rf.max_pos
    } else {
        cfg.original_max
    };
    if original == 0 {
        return Err(ModelError::Msg(
            "YaRN:original_max=0 且模型 context 缺失".into(),
        ));
    }
    let scaled = (original as f64 * cfg.yarn_factor).round() as usize;
    let table = rf.max_pos.max(scaled).max(max_seq.max(1));
    let yarn = crate::layers::rope::YarnCfg {
        factor: cfg.yarn_factor as f32,
        original_max: original,
        beta_fast: cfg.beta_fast as f32,
        beta_slow: cfg.beta_slow as f32,
        attention_factor: cfg.attention_factor as f32,
    };
    Rope::new_with(table, head_dim, rf.rotary_dim, rf.theta, Some(yarn))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preset() -> ModelSpec {
        crate::formats::qwen3_5_0_8b()
    }

    fn rope() -> RopeFacts {
        RopeFacts {
            max_pos: 1024,
            rotary_dim: 64,
            theta: 1.0e7,
        }
    }

    /// 单卡 = 恒等(形状逐字段不变;词表切位保持关)
    #[test]
    fn shard_spec_single_is_identity() {
        let s = preset();
        let out = shard_spec(s.clone(), (0, 1), false).expect("单卡恒等");
        assert_eq!(out.full_heads, s.full_heads);
        assert_eq!(out.gdn_heads, s.gdn_heads);
        assert_eq!(out.inter, s.inter);
        assert!(!out.head_split);
    }

    /// 单卡 + 词表切 = 响亮拒(静默忽略会让配置说谎)
    #[test]
    fn shard_spec_rejects_vocab_split_on_single_card() {
        let Err(e) = shard_spec(preset(), (0, 1), true) else {
            panic!("单卡词表切必须拒");
        };
        assert!(
            format!("{e}").contains("tp_vocab_split"),
            "文案要点名旋钮:{e}"
        );
    }

    /// TP2:头数/中间维按 rank 切,hidden 与 dtype 不动
    #[test]
    fn shard_spec_shards_heads_and_inter() {
        let s = preset();
        let out = shard_spec(s.clone(), (1, 2), false).expect("TP2 可分");
        assert_eq!(out.full_heads.0, s.full_heads.0 / 2);
        assert_eq!(out.full_heads.1, s.full_heads.1 / 2);
        assert_eq!(out.full_heads.2, s.full_heads.2);
        assert_eq!(out.hidden, s.hidden);
        assert_eq!(out.inter, s.inter / 2);
        assert_eq!(out.dtype, s.dtype);
    }

    /// HF 目录档给 `[vision] path` = 配置自相矛盾;
    /// 自洽门在 `from_ckpt` 之前(目录不存在也先报这条)
    #[test]
    fn hf_source_rejects_mmproj_before_touching_disk() {
        let Err(e) = HfDirSource::open(
            Path::new("/nonexistent-owl-checkpoint"),
            HfSourceCfg {
                preset: &preset(),
                kind: HfKind::SafeTensors,
                log_tag: "test",
                tokenizer_dir: None,
                shard: (0, 1),
                vocab_split: false,
                rope: rope(),
                draft: DraftPolicy::Off,
                vision: false,
                mmproj: Some(Path::new("/tmp/mmproj.gguf")),
                device_repack: false,
            },
        ) else {
            panic!("HF 档给 mmproj 必须拒");
        };
        assert!(e.to_string().contains("仅用于 format"), "{e}");
    }

    /// HF 档 rope:表长硬编码;`max_seq` 越表 = 响亮拒(核按 pos 直索引 cos/sin)
    #[test]
    fn hf_rope_rejects_max_seq_beyond_table() {
        let f = RopeFacts {
            max_pos: 1024,
            rotary_dim: 64,
            theta: 1.0e7,
        };
        assert!(build_hf_rope(f, 128, 1024).is_ok(), "恰好等于表长应放行");
        assert!(build_hf_rope(f, 128, 0).is_ok(), "0 = 未定预算应放行");
        let e = match build_hf_rope(f, 128, 1025) {
            Ok(_) => panic!("越表必须拒"),
            Err(e) => e,
        };
        assert!(e.to_string().contains("rope 表长"), "{e}");
    }

    /// 逐张量源(gguf)的 head/embed 覆盖门:装载器与 `--check` 共用
    #[test]
    fn keyed_overrides_reject_non_default_plan() {
        assert!(check_keyed_overrides(QuantPlan::F16, None).is_ok());
        assert!(check_keyed_overrides(QuantPlan::I8W, None).is_err());
        assert!(check_keyed_overrides(QuantPlan::F16, Some(QuantPlan::I8W)).is_err());
    }

    /// GGUF 档:目录/缺文件 = 响亮拒(错误点名 `.gguf 单文件`)
    #[test]
    fn gguf_source_rejects_missing_file() {
        let Err(e) = GgufFileSource::open(
            Path::new("/nonexistent-owl.gguf"),
            (0, 1),
            false,
            false,
            false,
        ) else {
            panic!("缺文件必须拒");
        };
        assert!(e.to_string().contains(".gguf 单文件"), "{e}");
    }

    /// GGUF 档:`[vision] path` 而无 `enabled` = 自相矛盾(在开容器之前拒;
    /// 用一个存在但非 GGUF 的文件证明顺序 —— 报的是自洽门,不是容器错)
    #[test]
    fn gguf_source_rejects_mmproj_without_vision() {
        let any_file = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
        let Err(e) = GgufFileSource::open(&any_file, (0, 1), false, false, true) else {
            panic!("vision off + mmproj 必须拒");
        };
        assert!(e.to_string().contains("enabled = false"), "{e}");
    }
}
