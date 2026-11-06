//! 模型装载面(ModelLoader;2026-10-01 拆分自 engine.rs §1)。
//!
//! 面向上层的模型加载器(经引擎的 face;spec 声明驱动):权重 +
//! tokenizer + rope 按 spec 装载为 [`LoadedModel`],错误在 load 边界
//! 落地(装载即校验)。
//!
//! **2026-11-04 clean-code 第三步**:`load_*` 六个入口的十二步重复流程收成
//! 一条 [`ModelLoader::load_source`];格式差异(声明/计划/词表/rope/视觉/
//! 草稿)由 [`ModelSource`] 回答(见 `owl_models::formats::model_source`),
//! 本文件不再认识 HF `/` GGUF 细节。
//!
//! metrics 装载分相 = **直用 API 恒开**(装载一次性路径,release 有
//! 数据;区别于热路径宏 debug 展开/release 零开销的双轨纪律,见
//! owl-shared metrics 模块)。

use std::path::Path;
use std::sync::Arc;

use owl_iface::contract::{DeviceClient, ModelError};
use owl_models::formats::load::{load_component, load_from_dir, LoadOpts};
use owl_models::formats::model_source::{
    DraftPolicy, GgufFileSource, HfDirSource, HfKind, HfSourceCfg, LoadPhases, ModelSource,
    PlanIntent, RopeFacts, VisionSource,
};
use owl_models::layers::rope::Rope;
use owl_models::layers::vision::VisionTower;
use owl_models::model::Model;
use owl_models::model::ModelSpec;
use owl_models::module::QuantPlan;
use owl_models::tokenizer::Tokenizer;
use owl_models::vlm::{Vlm, VlmCfg};

type Result<T> = std::result::Result<T, ModelError>;

/// Qwen3.5/3.8 族 rope 事实(表长 262144 / rotary 64 / θ 1e7)。
const ROPE_QWEN35: RopeFacts = RopeFacts {
    max_pos: 262_144,
    rotary_dim: 64,
    theta: 10_000_000.0,
};

/// 已装载模型(权重在设备 + tokenizer 解析 + rope 表;[`Engine::run`]
/// 的输入 —— Engine 见 engine.rs)
pub struct LoadedModel {
    pub(crate) model: Arc<Model>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) rope: Rope,
    pub(crate) spec: ModelSpec,
    /// 量化方案(环境真值;`EnvProvider.quant` 注入源)。
    /// **`None` = 逐张量计划**(GGUF 档:单一值表达不了 —— 型是**逐键**事实,
    /// 见 `PlanSource::Keyed`/`GgufPlan`;不许拿一个代表值糊过去)。
    pub(crate) quant_plan: Option<owl_models::module::QuantPlan>,
    /// MTP 头检查点目录(E5-M2b;Some = 检查点自带 mtp.*,spec 可选真草稿)
    pub(crate) mtp_dir: Option<std::path::PathBuf>,
    /// **视觉塔装配**(`[vision] enabled`;Some = 本档是 VLM)。
    /// 文本主干与 `model` **同一份 Arc**;塔是额外权重(每 rank 一份)。
    pub(crate) vlm: Option<std::sync::Arc<Vlm>>,
}

/// **rank 侧模型件**(TP 装配面):`LoadedModel` 去掉 tokenizer ——
/// tokenizer 是 host 侧唯一物(两侧各留一份纯浪费);其余件每 rank 各一份。
pub(crate) struct RankModel {
    pub(crate) model: Arc<Model>,
    pub(crate) rope: Rope,
    /// **分片 spec**(形状单源;与设备池/层同源)
    pub(crate) spec: ModelSpec,
    /// 量化方案(环境真值;`EnvProvider.quant` 注入源;`None` = 逐张量,见 [`LoadedModel`])
    pub(crate) quant_plan: Option<owl_models::module::QuantPlan>,
    /// MTP 头检查点目录(E5-M2b;Some = 检查点自带 mtp.*)
    pub(crate) mtp_dir: Option<std::path::PathBuf>,
    /// 视觉塔装配(Some = VLM;文本主干与 `model` 同 Arc)
    pub(crate) vlm: Option<std::sync::Arc<Vlm>>,
}

impl LoadedModel {
    /// 量化方案(装配期护栏消费面;S6 2026-10-17:TP 档暂不支持量化档,
    /// 见 `tp.rs` 的 ④ 护栏与 `docs/arch/TP2-施工设计.md` §22)。
    pub fn quant_plan(&self) -> Option<owl_models::module::QuantPlan> {
        self.quant_plan
    }

    /// 拆件:`(rank 侧模型件, tokenizer)`。
    pub(crate) fn into_parts(self) -> (RankModel, Tokenizer) {
        let LoadedModel {
            model,
            tokenizer,
            rope,
            spec,
            quant_plan,
            mtp_dir,
            vlm,
        } = self;
        (
            RankModel {
                model,
                rope,
                spec,
                quant_plan,
                mtp_dir,
                vlm,
            },
            tokenizer,
        )
    }
}

/// 面向上层的模型加载器(经引擎的 face;spec 声明驱动)
pub struct ModelLoader<'a, D: DeviceClient> {
    face: &'a mut D,
    /// **装载旋钮**(`[load]` 段;从配置来 —— 用户裁决 2026-10-19:不读 env)
    load: owl_shared::config::LoadCfg,
    /// `(slice_id, shard)`;缺省 `(0, 1)` = 单卡全量(S4 ②;见 [`ModelLoader::slice`])
    slice: (usize, usize),
    /// **lm_head 词表切**(`runtime.tp_vocab_split`;P1.1) —— 装载前写入
    /// `ModelSpec::head_split`(形状单源;装载规则随之切 out 维)。
    vocab_split: bool,
    /// **rope 外推/缩放**(`[model.rope]`;缺省关。只对 GGUF 档生效)
    rope: owl_shared::config::RopeCfg,
    /// 单 turn 预算(`runtime.max_seq`;YaRN 表长至少覆盖它)
    max_seq: usize,
    /// 视觉塔(`[vision] enabled`;false = 纯文本路径逐字节不变)
    vision: bool,
    /// GGUF 视觉塔单文件(`[vision] path`;None = 同目录唯一候选)
    mmproj: Option<std::path::PathBuf>,
    /// MTP 头覆盖路径(目录或 `.gguf`;None = 检查点自带)
    mtp_override: Option<std::path::PathBuf>,
}

impl<D: DeviceClient + 'static> ModelLoader<'_, D> {
    /// 构造(工厂;face 由 [`Engine::loader`](crate::Engine::loader) 借出)
    pub(crate) fn new<'a>(
        face: &'a mut D,
        load: owl_shared::config::LoadCfg,
    ) -> ModelLoader<'a, D> {
        ModelLoader {
            face,
            load,
            slice: (0, 1),
            vocab_split: false,
            rope: owl_shared::config::RopeCfg::default(),
            max_seq: 0,
            vision: false,
            mmproj: None,
            mtp_override: None,
        }
    }

    /// MTP 头覆盖(`[model].mtp_dir`;None = 检查点自带)
    pub fn mtp_override(mut self, p: Option<std::path::PathBuf>) -> Self {
        self.mtp_override = p;
        self
    }

    /// **rope 外推档**(`[model.rope]` + `runtime.max_seq`;GGUF 装载用)。
    ///
    /// 显式依赖律:调用方(引擎)把配置**传进来**,本模块零 env 读取。
    pub fn rope(mut self, rope: owl_shared::config::RopeCfg, max_seq: usize) -> Self {
        self.rope = rope;
        self.max_seq = max_seq;
        self
    }

    /// **切片装载**(S4 ②;确定性分片协议):`(slice_id, shard)` —— 本 rank 只装
    /// 模型的 `1/shard`(形状与权重都按规则表切;零协调,多 rank 可并行装载)。
    /// `shard = 1`(缺省)= 单卡全量。
    pub fn slice(mut self, slice_id: usize, shard: usize) -> Self {
        self.slice = (slice_id, shard);
        self
    }

    /// **视觉塔装载**(`[vision] enabled`;2026-10-20 视觉接线第一刀)。
    /// 缺省 false = 纯文本路径逐字节不变。
    pub fn vision(mut self, on: bool) -> Self {
        self.vision = on;
        self
    }

    /// **GGUF 视觉塔单文件**(`[vision] path`;None = 同目录唯一
    /// `mmproj*.gguf`)。仅 GGUF 档消费;HF/AWQ 档写本键 = 响亮拒。
    pub fn mmproj(mut self, p: Option<std::path::PathBuf>) -> Self {
        self.mmproj = p;
        self
    }

    /// **lm_head 词表切档**(`runtime.tp_vocab_split`;P1.1 2026-10-19)
    ///
    /// 装载前把意图位写入 `ModelSpec::head_split` —— 形状/层/装载规则全部
    /// 从它派生(`ModelSpec::sharded` 是唯一真源)。与 `slice` 同律:
    /// **显式声明**,引擎内部零 env 读取。
    pub fn vocab_split(mut self, on: bool) -> Self {
        self.vocab_split = on;
        self
    }

    /// **设备重排裁决**(配置三态 + 能力自述;`RepackPath::resolve` 唯一收口)。
    ///
    /// 只有 AWQ 源消费(SafeTensors/GGUF 无重排面);在源 open 期注入,
    /// 因为源的首键物化就会用到这条。
    fn repack_choice(&self) -> Result<bool> {
        let auto =
            owl_models::module::RepackPath::resolve(None, self.face.device_kernels()).is_device();
        Ok(self
            .load
            .repack_device()
            .map_err(ModelError::Msg)?
            .unwrap_or(auto))
    }

    /// 装载选项(本 slice 的 `(slice_id, shard)` + `[load]` 旋钮)
    ///
    /// `[load]` 旋钮走 `LoadOpts::with_cfg`(唯一映射点);设备重排取**源已
    /// 裁决**的结果(AWQ 在源 open 期注入,其余源恒 false)。
    fn load_opts(
        &self,
        src: &dyn ModelSource,
        dtype: owl_models::contract::Dtype,
    ) -> Result<LoadOpts> {
        let (id, n) = self.slice;
        let mut opts = LoadOpts::plain(dtype).with_cfg(&self.load)?;
        opts.shard = n;
        opts.slice_id = id;
        opts.head_split = self.vocab_split;
        opts.device_repack = Some(src.device_repack());
        Ok(opts)
    }

    /// rope 表装载 ctx(同 slice 口径;rope 是 Replicate,但保持一致)
    fn shard_ctx(&self, dtype: owl_models::contract::Dtype) -> owl_models::module::LoaderCtx {
        let (id, n) = self.slice;
        owl_models::module::LoaderCtx {
            dtype,
            shard: n,
            slice_id: id,
            device_repack: false,
            verify: false,
            debug_tap: false,
            staging_mb: self.load.staging_mb,
            dump: false,
        }
    }

    /// **唯一装载流**(2026-11-04 clean-code):源 → 模型/权重/词表/rope/视觉。
    ///
    /// 十二步的差异全部下沉到 [`ModelSource`]:
    /// - 维度/分片:`src.spec()`;
    /// - 计划(单值/逐键)与能力校验:`src.build_model()`;
    /// - 词表/rope/视觉/草稿:`src.tokenizer()/rope()/vision()/draft_dir()`。
    ///
    /// 本函数只保留**设备面顺序**:建模 → 权重 → 词表 → rope → 视觉。
    async fn load_source(
        &mut self,
        src: &dyn ModelSource,
        intent: &crate::LoadIntent,
    ) -> Result<LoadedModel> {
        let t_flow = std::time::Instant::now();
        let _scope = self.face.scoped_ctx(
            owl_iface::handle::Owner::System,
            owl_iface::handle::AssetClass::Weight,
        );
        // metrics 早 init(幂等):`/debug/metrics` 的直用 API 记账需要 store 在场
        let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());

        let spec = src.spec().clone();
        let t_plan = std::time::Instant::now();
        let built = src.build_model(&PlanIntent {
            quant: intent.quant,
            head: intent.head,
            embed: intent.embed,
        })?;
        let t_plan = t_plan.elapsed();

        let opts = self.load_opts(src, spec.dtype)?;
        let t_weights = std::time::Instant::now();
        load_component(self.face, &built.model, src.weights(), opts).await?;
        let t_weights = t_weights.elapsed();
        let model = Arc::new(built.model);

        let tok = src.tokenizer()?;
        let rope = src.rope(&self.rope, self.max_seq)?;
        src.log_facts(&rope, tok.n_tokens, &self.rope);
        let ctx = self.shard_ctx(spec.dtype);
        owl_models::interpreters::eval_load(&rope, self.face, &rope.tables(), &ctx).await?;

        let t_vision = std::time::Instant::now();
        let vlm = match src.vision() {
            None => None,
            Some(VisionSource::InCkpt) => {
                self.load_vision_into(src.root(), &model, spec.dtype)
                    .await?
            }
            Some(VisionSource::Mmproj(text)) => {
                self.load_vision_gguf(text, &spec, &model, spec.dtype)
                    .await?
            }
        };
        let t_vision = t_vision.elapsed();

        src.log_phases(&LoadPhases {
            open: src.open_duration(),
            plan: t_plan,
            weights: t_weights,
            vision: t_vision,
            total: src.open_duration() + t_flow.elapsed(),
        });

        Ok(LoadedModel {
            model,
            tokenizer: tok.tokenizer,
            rope,
            spec,
            quant_plan: built.quant_plan,
            mtp_dir: self.mtp_override.clone().or_else(|| src.draft_dir()),
            vlm,
        })
    }

    /// **按装载臂装载 —— 唯一分派点**(2026-11-04 clean-code)。
    ///
    /// 三条臂的差异收敛在这里(而不是 server 的 `load_shard`):
    /// - 旋钮统一走 [`LoadIntent`];每臂**只取自己认的**,
    ///   不认的**响亮拒**(不再静默忽略 —— GGUF 的 head/embed 就是历史静默项);
    /// - `mtp_override` 是 loader 的构造期字段(引擎注入),与臂无关。
    pub async fn load_arm(
        &mut self,
        dir: &Path,
        arm: crate::LoadArm,
        intent: &crate::LoadIntent,
    ) -> Result<LoadedModel> {
        match arm {
            // AWQ pack-quantized 家族(cyankiwi 系;int4 已由产出方烘焙)
            crate::LoadArm::AwqInt4 | crate::LoadArm::AwqA8 => {
                self.load_awq(dir, dir, intent).await
            }
            // 普通 HF safetensors 目录(白名单下只有 nvfp4 / f16 基线两档)
            crate::LoadArm::NvFp4 | crate::LoadArm::F16Base => {
                self.load_safetensors_intent(dir, intent).await
            }
            // GGUF 单文件:超参/词表/rope/权重全从文件来 ⇒ head/embed 覆盖由
            // 源的 `build_model` 响亮拒(计划是**逐张量**事实)。
            crate::LoadArm::Gguf(_) => {
                let src = self.gguf_source(dir)?;
                self.load_source(&src, intent).await
            }
        }
    }

    /// GGUF 单文件源(配置自洽校验:vision/mmproj 组合)
    fn gguf_source(&self, path: &Path) -> Result<GgufFileSource> {
        GgufFileSource::open(
            path,
            self.slice,
            self.vocab_split,
            self.vision,
            self.mmproj.is_some(),
        )
    }

    /// **通用 Qwen3.5 HF 目录装载**(维度事实 = `config.json` + 形状兜底;
    /// `preset` = 家族声明基座)。
    ///
    /// 与 [`load_qwen35_0_8b`](Self::load_qwen35_0_8b) **同源**,只把"档位预设"
    /// 参数化。两个消费者:
    /// ① 任意 Qwen3.5 HF 档(4B/9B…)上引擎;
    /// ② **交叉源对拍**:同一模型的 HF 快照 vs GGUF(同 dims 两源 ⇒ 只比权重语义;
    ///    见 `tests_gguf::gpu_gguf_4b_vs_hf_4b`)。
    ///
    /// `rope` = (rotary_dim, theta, max_pos) 三件套 —— **必须显式给**:
    /// HF 的 rope 参数还没进 `ModelSpec`(家族档现在硬编码在各自装载器里),
    /// 显式入参让"交叉源对拍"能用**同一组** rope 事实(否则比的是两件事)。
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn load_qwen35_hf(
        &mut self,
        dir: &Path,
        preset: ModelSpec,
        rope: (usize, f32, usize),
    ) -> Result<LoadedModel> {
        if self.vision || self.mmproj.is_some() {
            return Err(ModelError::Msg(
                "load_qwen35_hf:通用 HF 档尚未接视觉塔（视觉塔请走 0.8B/27B/GGUF 专用装载器）"
                    .into(),
            ));
        }
        let src = HfDirSource::open(
            dir,
            HfSourceCfg {
                preset: &preset,
                kind: HfKind::SafeTensors,
                log_tag: "qwen35-hf",
                tokenizer_dir: None,
                shard: self.slice,
                vocab_split: self.vocab_split,
                rope: RopeFacts {
                    max_pos: rope.2,
                    rotary_dim: rope.0,
                    theta: rope.1,
                },
                draft: DraftPolicy::Off,
                vision: false,
                mmproj: None,
                device_repack: false,
            },
        )?;
        let intent = crate::LoadIntent {
            quant: QuantPlan::F16,
            head: QuantPlan::F16,
            embed: None,
        };
        self.load_source(&src, &intent).await
    }

    /// **GGUF 单文件装载**(2026-10-19 gguf 轮;设计 §四 L4 / REQ-GGUF-02)。
    ///
    /// 四件事实全从**同一个文件**取(单文件自给):
    /// 维度(`spec_from_gguf`)· 词表(`build_vocab` + `tokenizer.ggml.*`)·
    /// rope(`rope_facts`)· 权重(`GgufSource` —— 块字节原样,零解量化)。
    ///
    /// 三条与 HF 路径的差异(**设计使然**,不是遗漏):
    /// 1. **计划是逐张量的** ⇒ `PlanSource::Keyed`;构造期尾声取走未解析键
    ///    (非空 = 响亮拒;禁静默落 F16,设计 I-2);
    /// 2. **tokenizer 不经目录**(`from_parts`;HF 目录在场也只读打印);
    /// 3. **quant_plan = None**(逐张量;单一值表达不了)。
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn load_gguf(&mut self, path: &Path) -> Result<LoadedModel> {
        let src = self.gguf_source(path)?;
        // `quant` 不参与 GGUF 计划(逐键看文件);字面量只为满足接口,
        // head/embed 覆盖位空 = 单文件默认语义。
        let intent = crate::LoadIntent {
            quant: QuantPlan::F16,
            head: QuantPlan::F16,
            embed: None,
        };
        self.load_source(&src, &intent).await
    }

    /// **视觉塔装载**(`[vision] enabled`;两档共用同一份 —— 0.8B 与 27B 的
    /// 检查点都是 `Qwen3_5ForConditionalGeneration`,自带 `model.visual.*`)。
    ///
    /// 文本主干已装好(量化/分片/头计划)⇒ 只补视觉段;塔权重**每 rank 全量**
    /// (Replicate;TP2 不切视觉),故 opts 固定 `shard = 1`。
    ///
    /// **前置由源回答**:`VisionSource::InCkpt` 已经保证 `[vision] enabled` 且
    /// 未给 `[vision] path`(HF 档给了 path 在源 open 期就响亮拒)——
    /// 这里不再重复守卫(死分支,2026-11-04 收编)。
    async fn load_vision_into(
        &mut self,
        dir: &Path,
        model: &Arc<Model>,
        dtype: owl_iface::contract::Dtype,
    ) -> Result<Option<Arc<Vlm>>> {
        let raw = owl_shared::file_loader::read(dir.join("config.json")).map_err(|e| {
            ModelError::Msg(format!(
                "视觉装载:config.json 读取失败({}): {e}",
                dir.display()
            ))
        })?;
        let vcfg = VlmCfg::from_json(&raw)?;
        let tower = VisionTower::new(vcfg.vision);
        let mut topts = LoadOpts::new(self.face, dtype).with_cfg(&self.load)?;
        // 塔权重每 rank 全量(Replicate;TP2 不切视觉)
        topts.shard = 1;
        topts.slice_id = 0;
        load_from_dir(dir, self.face, &tower, topts).await?;
        let vlm = Vlm::from_model(Arc::clone(model), tower, vcfg)?;
        owl_shared::log_info!(
            "boot",
            "视觉塔已装载(model.visual.*;塔权重每 rank 全量,文本主干共用)"
        );
        Ok(Some(Arc::new(vlm)))
    }

    /// **GGUF 视觉塔装载**(2026-10-21):文本 `.gguf` 只提供 text/mRoPE/词表,
    /// 塔权重从同目录 `mmproj-*.gguf` 取。
    ///
    /// `mmproj` 显式路径优先;缺省 = 同目录 **唯一** 的 `mmproj*.gguf`
    /// (多个候选/没有 = 响亮拒,避免静默装错量化档)。
    ///
    /// **前置由源回答**:`VisionSource::Mmproj` 已经保证 `enabled` 且
    /// 不同时存在自相矛盾配置 —— 这里不再重复守卫(死分支,2026-11-04 收编)。
    async fn load_vision_gguf(
        &mut self,
        text: &owl_models::formats::gguf::Gguf,
        text_spec: &ModelSpec,
        model: &Arc<Model>,
        dtype: owl_iface::contract::Dtype,
    ) -> Result<Option<Arc<Vlm>>> {
        let path = self.resolve_mmproj(text.path.as_path())?;
        let mm = owl_models::formats::gguf::Gguf::open(&path).map_err(|e| {
            ModelError::Msg(format!("GGUF 视觉塔:打开 {} 失败: {e}", path.display()))
        })?;
        let vcfg = VlmCfg::from_gguf(text, &mm, text_spec.clone())?;
        let tower = VisionTower::new(vcfg.vision);
        let src = owl_models::formats::gguf::MmprojSource::from_gguf(mm)?;
        let mut opts = LoadOpts::plain(dtype).with_cfg(&self.load)?;
        // 塔权重每 rank 全量(Replicate);mmproj 无重排面
        opts.shard = 1;
        opts.slice_id = 0;
        load_component(self.face, &tower, &src, opts).await?;
        let vlm = Vlm::from_model(Arc::clone(model), tower, vcfg)?;
        owl_shared::log_info!(
            "boot",
            "GGUF 视觉塔已装载:{}（塔权重每 rank 全量，文本主干共用）",
            path.display()
        );
        Ok(Some(Arc::new(vlm)))
    }

    /// mmproj 路径解析:显式优先;缺省只在**同目录唯一候选**时自动选。
    fn resolve_mmproj(&self, text_path: &Path) -> Result<std::path::PathBuf> {
        owl_models::formats::gguf::resolve_mmproj_path(text_path, self.mmproj.as_deref())
    }

    /// Qwen3.5-0.8B(维度/键基座从**检查点**读:config.json 优先 + 形状
    /// 兜底;族预设只供 tokenizer 声明;装载走通用入口)
    // A3 同步隔离:引擎单线程模型;Arc 仅作共享句柄,不跨线程发送。
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn load_qwen35_0_8b(&mut self, dir: &Path) -> Result<LoadedModel> {
        self.load_qwen35_safetensors(dir, QuantPlan::F16, QuantPlan::F16, None)
            .await
    }

    /// **通用 HF safetensors 档(带量化计划)**(2026-10-28 装载管线 P2)。
    ///
    /// 与 [`Self::load_qwen35_0_8b`](Self::load_qwen35_0_8b) **同源**;差别只有
    /// 三个计划参数(主干 / 头 / 表)。为什么要这个入口:原函数把
    /// `QuantPlan::F16` **硬编码**在体内 ⇒ `[model] quant = "nvfp4"` 即使被
    /// 配置面接受也永远装不到(nvfp4 臂的声明根本不会产生 —— 实测 `quant=F16`
    /// 的 boot 日志就是那次静默失配的现场)。
    ///
    /// `quant_plan` 同时作为 `LoadedModel.quant_plan` 交给装配面:它决定
    /// `kernel-env` 的 tag 派生(没它就没有 `quant.nvfp4` ⇒ 算子被白名单拒)。
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn load_qwen35_safetensors(
        &mut self,
        dir: &Path,
        quant_plan: QuantPlan,
        head_plan: QuantPlan,
        embed_plan: Option<QuantPlan>,
    ) -> Result<LoadedModel> {
        let intent = crate::LoadIntent {
            quant: quant_plan,
            head: head_plan,
            embed: embed_plan,
        };
        self.load_safetensors_intent(dir, &intent).await
    }

    /// safetensors 目录源(装载意图已经算好;`load_arm` 与公开入口共用)
    async fn load_safetensors_intent(
        &mut self,
        dir: &Path,
        intent: &crate::LoadIntent,
    ) -> Result<LoadedModel> {
        let preset = owl_models::formats::qwen3_5_0_8b();
        let src = HfDirSource::open(
            dir,
            HfSourceCfg {
                preset: &preset,
                kind: HfKind::SafeTensors,
                log_tag: "safetensors",
                tokenizer_dir: None,
                shard: self.slice,
                vocab_split: self.vocab_split,
                rope: ROPE_QWEN35,
                draft: DraftPolicy::Probe,
                vision: self.vision,
                mmproj: self.mmproj.as_deref(),
                device_repack: false,
            },
        )?;
        self.load_source(&src, intent).await
    }

    /// W4A16 装载(E3;llm-compressor 产物目录):装载期重排到 marlin
    /// 布局,q_proj 等全部达标 Linear 走 foreign GEMM(REQ-PRE-01)
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn load_qwen35_0_8b_w4a16(
        &mut self,
        dir: &Path,
        tokenizer_dir: &Path,
    ) -> Result<LoadedModel> {
        if self.vision {
            return Err(ModelError::Msg(
                "[vision] enabled = true:0.8B 档尚未接视觉塔(仅 27B AWQ 档)".into(),
            ));
        }
        if self.mmproj.is_some() {
            return Err(ModelError::Msg(
                "[vision] path 仅用于 format = \"gguf\"（0.8B-W4A16 档无 mmproj 路径）".into(),
            ));
        }
        let preset = owl_models::formats::qwen3_5_0_8b();
        let src = HfDirSource::open(
            dir,
            HfSourceCfg {
                preset: &preset,
                kind: HfKind::W4A16,
                log_tag: "0.8B-W4A16",
                tokenizer_dir: Some(tokenizer_dir),
                shard: self.slice,
                vocab_split: self.vocab_split,
                rope: ROPE_QWEN35,
                draft: DraftPolicy::Off,
                vision: false,
                mmproj: None,
                device_repack: false,
            },
        )?;
        let intent = crate::LoadIntent {
            quant: QuantPlan::W4A16,
            head: QuantPlan::F16,
            embed: None,
        };
        self.load_source(&src, &intent).await
    }

    /// Qwen3.8-27B AWQ-INT4 装载(2026-10-01;cyankiwi g32-asym 检查点):
    /// marlin kU4(has_zp)内核;rope 与 0.8B 同参(theta 1e7 + rotary 64)。
    /// `head_plan`(C1,2026-10-11):untied lm_head 量化计划(唯一映射点
    /// owl_shared::config::HeadQuant → owl_models QuantPlan;量化方案
    /// = 配置枚举,loader 零硬编码)。
    ///
    /// `quant`(S2,2026-10-17;2026-10-17 由 `w4a8: bool` 升格为方案枚举):
    /// **主干**量化 plan —— `QuantPlan::W4A16Awq`(今天行为)/ `W4A8Awq`
    /// (a8 布局 + 激活 int8 + int8 张量核)。显式参数而非 env:量化方案改
    /// 语义(不是诊断旋钮),且入口(server boot)经 `engine::quant_plan_of`
    /// 从 `model.quant` 映射。**非主干两位 plan 在此响亮拒**(F16/W4A16
    /// 走 0.8B 线;馈错 = 静默降级到全部 f16,正是本仓禁止的那类错)。
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn load_qwen38_27b_awq(
        &mut self,
        dir: &Path,
        tokenizer_dir: &Path,
        head_plan: QuantPlan,
        quant: QuantPlan,
    ) -> Result<LoadedModel> {
        self.load_qwen38_27b_awq_with_embed(dir, tokenizer_dir, head_plan, quant, None)
            .await
    }

    /// **表计划覆盖入口**(P8,2026-10-22):`embed_plan = Some(I8W)` ⇒ embed 表
    /// int8(q/s 派生键;省 ~1.2G);`None` = 走检查点原表计划。
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn load_qwen38_27b_awq_with_embed(
        &mut self,
        dir: &Path,
        tokenizer_dir: &Path,
        head_plan: QuantPlan,
        quant: QuantPlan,
        embed_plan: Option<QuantPlan>,
    ) -> Result<LoadedModel> {
        let intent = crate::LoadIntent {
            quant,
            head: head_plan,
            embed: embed_plan,
        };
        self.load_awq(dir, tokenizer_dir, &intent).await
    }

    /// AWQ 目录源 + 装载计时(`load.27b.total` 是既有观测面,必须保留)
    async fn load_awq(
        &mut self,
        dir: &Path,
        tokenizer_dir: &Path,
        intent: &crate::LoadIntent,
    ) -> Result<LoadedModel> {
        let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
        owl_shared::metrics::with_metrics_store(|s| {
            s.timer_begin("load.27b.total", file!(), line!());
        });
        let preset = owl_models::formats::qwen3_8_27b();
        let device_repack = self.repack_choice()?;
        let src = HfDirSource::open(
            dir,
            HfSourceCfg {
                preset: &preset,
                kind: HfKind::Awq,
                log_tag: "27B-AWQ",
                tokenizer_dir: Some(tokenizer_dir),
                shard: self.slice,
                vocab_split: self.vocab_split,
                rope: ROPE_QWEN35,
                draft: DraftPolicy::Directory,
                vision: self.vision,
                mmproj: self.mmproj.as_deref(),
                device_repack,
            },
        )?;
        let out = self.load_source(&src, intent).await?;
        owl_shared::metrics::with_metrics_store(|s| {
            let _ = s.timer_end("load.27b.total", file!(), line!());
        });
        owl_shared::metrics::query_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("load."),
        );
        Ok(out)
    }
}
