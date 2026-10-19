//! 模型装载面(ModelLoader;2026-10-01 拆分自 engine.rs §1)。
//!
//! 面向上层的模型加载器(经引擎的 face;spec 声明驱动):权重 +
//! tokenizer + rope 按 spec 装载为 [`LoadedModel`],错误在 load 边界
//! 落地(装载即校验)。
//!
//! metrics 装载分相 = **直用 API 恒开**(装载一次性路径,release 有
//! 数据;区别于热路径宏 debug 展开/release 零开销的双轨纪律,见
//! owl-shared metrics 模块)。

use std::path::Path;
use std::sync::Arc;

use owl_iface::contract::{DeviceClient, ModelError};
use owl_models::layers::rope::Rope;
use owl_models::model::ModelSpec;
use owl_models::formats::load::{load_from_dir, LoadOpts};
use owl_models::formats::spec::{SpecReport, SpecSource};
use owl_models::formats::{load_tokenizer, qwen3_5_0_8b, qwen3_8_27b, Qwen35Convention};
use owl_models::tokenizer::Tokenizer;
use owl_models::model::Model;

type Result<T> = std::result::Result<T, ModelError>;

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
}

impl LoadedModel {
    /// 量化方案(装配期护栏消费面;S6 2026-10-17:TP 档暂不支持量化档,
    /// 见 `tp.rs` 的 ④ 护栏与 `docs/arch/TP2-施工设计.md` §22)。
    pub fn quant_plan(&self) -> Option<owl_models::module::QuantPlan> {
        self.quant_plan
    }

    /// 拆件:`(rank 侧模型件, tokenizer)`。
    pub(crate) fn into_parts(self) -> (RankModel, Tokenizer) {
        let LoadedModel { model, tokenizer, rope, spec, quant_plan, mtp_dir } = self;
        (RankModel { model, rope, spec, quant_plan, mtp_dir }, tokenizer)
    }
}

/// 面向上层的模型加载器(经引擎的 face;spec 声明驱动)
pub struct ModelLoader<'a, D: DeviceClient> {
    face: &'a mut D,
    /// `(slice_id, shard)`;缺省 `(0, 1)` = 单卡全量(S4 ②;见 [`ModelLoader::slice`])
    slice: (usize, usize),
}

impl<D: DeviceClient + 'static> ModelLoader<'_, D> {
    /// 构造(工厂;face 由 [`Engine::loader`](crate::Engine::loader) 借出)
    pub(crate) fn new<'a>(face: &'a mut D) -> ModelLoader<'a, D> {
        ModelLoader { face, slice: (0, 1) }
    }

    /// **切片装载**(S4 ②;确定性分片协议):`(slice_id, shard)` —— 本 rank 只装
    /// 模型的 `1/shard`(形状与权重都按规则表切;零协调,多 rank 可并行装载)。
    /// `shard = 1`(缺省)= 单卡全量。
    pub fn slice(mut self, slice_id: usize, shard: usize) -> Self {
        self.slice = (slice_id, shard);
        self
    }

    /// 分片 spec(形状单源 = [`ModelSpec::sharded`];`shard = 1` 恒等)
    fn shard_spec(&self, spec: ModelSpec) -> Result<ModelSpec> {
        let (id, n) = self.slice;
        if n <= 1 {
            return Ok(spec);
        }
        spec.sharded(owl_models::plan::tp::TpPlan::new(id, n).map_err(ModelError::Msg)?)
            .map_err(ModelError::Msg)
    }

    /// 装载选项(带上本 slice 的 (slice_id, shard))
    fn shard_opts(&self, dtype: owl_models::contract::Dtype) -> LoadOpts {
        let (id, n) = self.slice;
        LoadOpts { shard: n, slice_id: id, ..LoadOpts::plain(dtype) }
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
        }
    }

    /// **通用 Qwen3.5 HF 目录装载**(维度事实 = `config.json` + 形状兜底;
    /// `preset` = 家族声明基座)。
    ///
    /// 与 [`load_qwen35_0_8b`](Self::load_qwen35_0_8b) **同体**,只把"档位预设"
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
        let (spec, rep) = ModelSpec::from_ckpt(dir, SpecSource::family(&preset))?;
        log_spec_report("qwen35-hf", &rep);
        let spec = self.shard_spec(spec)?;
        let model = Model::new(
            &spec,
            Qwen35Convention::new(convention_base(&rep)),
            owl_models::module::QuantPlan::F16,
            owl_models::module::QuantPlan::F16,
        )?;
        load_from_dir(dir, self.face, &model, self.shard_opts(spec.dtype)).await?;
        let model = Arc::new(model);
        let tokenizer = load_tokenizer(dir, &spec.tokenizer)?;
        let (rotary_dim, theta, max_pos) = rope;
        let rope_tbl = Rope::new(max_pos, spec.full_heads.2, rotary_dim, theta)?;
        let quant_plan = Some(owl_models::module::QuantPlan::F16);
        let ctx = self.shard_ctx(spec.dtype);
        owl_models::interpreters::eval_load(&rope_tbl, self.face, &rope_tbl.tables(), &ctx).await?;
        Ok(LoadedModel {
            model,
            tokenizer,
            rope: rope_tbl,
            spec,
            quant_plan,
            mtp_dir: None,
        })
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
        use owl_models::formats::gguf::{
            build_vocab, rope_facts, spec_from_gguf, GgufConvention, GgufPlan, GgufSource,
        };
        use owl_models::module::{PlanSource, PlanTable, SourceFacts};
        if !path.is_file() {
            return Err(ModelError::Msg(format!(
                "GGUF 档:model.dir 必须是 **.gguf 单文件**(实得 {})",
                path.display()
            )));
        }
        // TP 分片(W-1,2026-10-19):**源自担** —— 块对齐列切 + v-head grouped
        // 聚合都在 `GgufSource` 里(`owns_shard`);开箱即逐键验可分片性
        // (不整除/块不对齐/按头不可分 ⇒ 响亮拒)。见设计 §14.6/§15。
        let sel = (self.slice.1 > 1)
            .then_some(owl_models::formats::gguf::ShardSel { id: self.slice.0, world: self.slice.1 });
        // metrics 早 init(与 27B 装载同律):`/debug/metrics` 的 `decode.step`
        // 等直用 API 记账需要 store 在场;server 入口已 init(幂等,`let _ =`)。
        let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
        // ① 容器 + 维度事实(同一索引;源自己持有 spec,不让两处推法漂移)
        let t_open = std::time::Instant::now();
        let src = GgufSource::open_shard(path, sel)?;
        let (spec, rep) = spec_from_gguf(src.gguf())?;
        log_gguf_report(path, &rep);
        let spec = self.shard_spec(spec)?;
        let t_meta = t_open.elapsed();
        // ② 计划表(逐键)+ 构造(构造期一步判死:缺键/未移植型/形态不符)
        // 分片档:计划表的形状门按角色把声明维**放大**到文件全量维
        let plan = match sel {
            Some(s) => GgufPlan::for_shard(src.gguf(), *src.dims(), s),
            None => GgufPlan::new(src.gguf()),
        };
        let keys = GgufConvention::new();
        // 源事实:TP 档 = 权重内维已是 grouped 序 ⇒ 不要 A2 表
        let facts = if sel.is_some() { SourceFacts::GGUF_TP } else { SourceFacts::GGUF };
        let ps = PlanSource::keyed(&plan, &keys, facts);
        let model = Model::new(&spec, GgufConvention::new(), ps, ps)?;
        let unresolved = plan.take_unresolved();
        if !unresolved.is_empty() {
            return Err(ModelError::Msg(format!(
                "GGUF 计划表有 {} 个未解析键(构造期响亮拒 —— 禁静默落 F16):{}{}",
                unresolved.len(),
                unresolved.iter().take(6).cloned().collect::<Vec<_>>().join(" | "),
                if unresolved.len() > 6 { " | …" } else { "" }
            )));
        }
        // ③ 权重装载(源 = 压缩块字节;与 HF 路径同一 eval_load 面)
        let mut opts = LoadOpts::plain(spec.dtype);
        opts.shard = self.slice.1;
        opts.slice_id = self.slice.0;
        opts.debug_tap = owl_shared::env_reader::flag("OWL_LOAD_DEBUG");
        opts.verify = owl_shared::env_reader::flag("OWL_LOAD_VERIFY");
        let t_plan = t_open.elapsed();
        load_from_dir(path, self.face, &model, opts).await?;
        let model = Arc::new(model);
        let t_load = t_open.elapsed();
        // ④ 词表(文件自给;eos 直接吃文件给的 id —— GGUF 给 id 不给名)
        let vocab = build_vocab(src.gguf())?;
        let eos = vocab.eos_ids();
        let fmt = spec.tokenizer.chat.clone();
        let tokenizer = Tokenizer::from_parts(vocab.inner, eos, fmt);
        // ⑤ rope(文件自给;头维 = full_heads.2 与 spec 同源)
        let rf = rope_facts(src.gguf())?;
        let rope = Rope::new(rf.max_pos, spec.full_heads.2, rf.rotary_dim, rf.theta)?;
        owl_shared::log_info!(
            "boot",
            "GGUF:{} · 层 {}(full {} / gdn {}) · 词表 {} · rope(rot {} / θ {:.3e} / max {}) · tied {}",
            path.display(),
            spec.layer_types.len(),
            spec.layer_types.iter().filter(|f| **f).count(),
            spec.layer_types.iter().filter(|f| !**f).count(),
            vocab.n_tokens,
            rf.rotary_dim,
            rf.theta,
            rf.max_pos,
            spec.tied,
        );
        let ctx = self.shard_ctx(spec.dtype);
        owl_models::interpreters::eval_load(&rope, self.face, &rope.tables(), &ctx).await?;
        // 装载分相(启动时间是生产指标:哪一相贵要看得出来;测试档也上屏)
        eprintln!(
            "[gguf] 装载分相:容器+维度 {:.2}s + 计划/构造 {:.2}s + 权重 {:.2}s + 词表/rope {:.2}s = {:.2}s",
            t_meta.as_secs_f32(),
            (t_plan - t_meta).as_secs_f32(),
            (t_load - t_plan).as_secs_f32(),
            (t_open.elapsed() - t_load).as_secs_f32(),
            t_open.elapsed().as_secs_f32()
        );
        Ok(LoadedModel {
            model,
            tokenizer,
            rope,
            spec,
            // 逐张量计划(GGUF):单一值表达不了 ⇒ None(见字段文档)
            quant_plan: None,
            mtp_dir: None,
        })
    }

    /// Qwen3.5-0.8B(维度/键基座从**检查点**读:config.json 优先 + 形状
    /// 兜底;族预设只供 tokenizer 声明;装载走通用入口)
    // A3 同步隔离:引擎单线程模型;Arc 仅作共享句柄,不跨线程发送。
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn load_qwen35_0_8b(&mut self, dir: &Path) -> Result<LoadedModel> {
        let (spec, rep) = ModelSpec::from_ckpt(dir, SpecSource::family(&qwen3_5_0_8b()))?;
        log_spec_report("0.8B", &rep);
        let spec = self.shard_spec(spec)?;
        let model = Model::new(
            &spec,
            Qwen35Convention::new(convention_base(&rep)),
            owl_models::module::QuantPlan::F16,
            owl_models::module::QuantPlan::F16,
        )?;
        load_from_dir(dir, self.face, &model, self.shard_opts(spec.dtype)).await?;
        let model = Arc::new(model);
        let tokenizer = load_tokenizer(dir, &spec.tokenizer)?;
        let rope = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        let quant_plan = Some(owl_models::module::QuantPlan::F16);
        let ctx = self.shard_ctx(spec.dtype);
        owl_models::interpreters::eval_load(&rope, self.face, &rope.tables(), &ctx).await?;
        Ok(LoadedModel { model, tokenizer, rope, spec, quant_plan, mtp_dir: None })

    }

    /// W4A16 装载(E3;llm-compressor 产物目录):装载期重排到 marlin
    /// 布局,q_proj 等全部达标 Linear 走 foreign GEMM(REQ-PRE-01)
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn load_qwen35_0_8b_w4a16(
        &mut self,
        dir: &Path,
        tokenizer_dir: &Path,
    ) -> Result<LoadedModel> {
        let (spec, rep) = ModelSpec::from_ckpt(dir, SpecSource::family(&qwen3_5_0_8b()))?;
        log_spec_report("0.8B-W4A16", &rep);
        let spec = self.shard_spec(spec)?;
        let model = Model::new(
            &spec,
            Qwen35Convention::new(convention_base(&rep)),
            owl_models::module::QuantPlan::W4A16,
            owl_models::module::QuantPlan::F16,
        )?;
        // 设备重排 = 关(与历史一致:0.8B 走 CPU 两步重排,量小不值 GPU 往返)
        load_from_dir(dir, self.face, &model, self.shard_opts(spec.dtype)).await?;
        let model = Arc::new(model);
        let tokenizer = load_tokenizer(tokenizer_dir, &spec.tokenizer)?;
        let rope = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        let quant_plan = Some(owl_models::module::QuantPlan::W4A16);
        let ctx = self.shard_ctx(spec.dtype);
        owl_models::interpreters::eval_load(&rope, self.face, &rope.tables(), &ctx).await?;
        Ok(LoadedModel { model, tokenizer, rope, spec, quant_plan, mtp_dir: None })

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
        head_plan: owl_models::module::QuantPlan,
        quant: owl_models::module::QuantPlan,
    ) -> Result<LoadedModel> {
        // metrics 装载分相(直用 API 恒开;装载一次性路径,release 有数据)
        let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
        owl_shared::metrics::with_metrics_store(|s| {
            s.timer_begin("load.27b.total", file!(), line!());
        });
        let (spec, rep) = ModelSpec::from_ckpt(dir, SpecSource::family(&qwen3_8_27b()))?;
        log_spec_report("27B-AWQ", &rep);
        let spec = self.shard_spec(spec)?;
        let quant_plan = match quant {
            owl_models::module::QuantPlan::W4A16Awq => quant,
            owl_models::module::QuantPlan::W4A8Awq => {
                owl_shared::log_info!(
                    "boot",
                    "W4A8 档位开(model.quant = \"w4a8\"):权重 a8 布局 + 激活 per-token int8 + kS8×kU4 GEMM"
                );
                quant
            }
            other => {
                return Err(ModelError::Msg(format!(
                    "27B-AWQ 主干量化 plan 非法: {other:?} —— 只接受 W4A16Awq / W4A8Awq\
                     (engine::quant_plan_of 是唯一映射点)"
                )));
            }
        };
        let model = Model::new(
            &spec,
            Qwen35Convention::new(convention_base(&rep)),
            quant_plan,
            head_plan,
        )?;
        // 旋钮:设备重排 = 能力自述裁决(RepackPath::resolve 唯一收口)
        let mut opts = LoadOpts::new(self.face, spec.dtype);
        // S4 ②:本 slice 的 (slice_id, shard)
        opts.shard = self.slice.1;
        opts.slice_id = self.slice.0;
        // 逐键分相账(OWL_LOAD_DEBUG=1;装载测速/瓶颈定位用 —— 与测试路同一开关)
        opts.debug_tap = owl_shared::env_reader::flag("OWL_LOAD_DEBUG");
        // 装载后整块回读校验(OWL_LOAD_VERIFY=1;纠错面 —— 装载数据面大改时的硬门)
        opts.verify = owl_shared::env_reader::flag("OWL_LOAD_VERIFY");
        load_from_dir(dir, self.face, &model, opts).await?;
        let model = Arc::new(model);
        let tokenizer = load_tokenizer(tokenizer_dir, &spec.tokenizer)?;
        let rope = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        let ctx = self.shard_ctx(spec.dtype);
        owl_models::interpreters::eval_load(&rope, self.face, &rope.tables(), &ctx).await?;
        owl_shared::metrics::with_metrics_store(|s| {
            let _ = s.timer_end("load.27b.total", file!(), line!());
        });
        owl_shared::metrics::query_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag_prefix("load."),
        );
        Ok(LoadedModel {
            model,
            tokenizer,
            rope,
            spec,
            quant_plan: Some(quant_plan),
            mtp_dir: Some(dir.to_path_buf()),
        })

    }
}

/// 键基座 → 键约定基座(`keys_base` 是**字符串前缀**(带尾点),约定无尾点;
/// 组件档空基座 = 原生命名直映,与 `Qwen35Convention::new("")` 同义)。
fn convention_base(rep: &SpecReport) -> &str {
    rep.keys_base.trim_end_matches('.')
}

/// 起点读取账(响亮:键基座 + config↔形状 失配逐条上屏;
/// 静默采形状违背“不静默”律)
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
fn log_gguf_report(path: &std::path::Path, rep: &owl_models::formats::gguf::GgufSpecReport) {
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
