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
    /// 量化方案(环境真值;EnvProvider.quant 注入源)
    pub(crate) quant_plan: owl_models::module::QuantPlan,
    /// MTP 头检查点目录(E5-M2b;Some = 检查点自带 mtp.*,spec 可选真草稿)
    pub(crate) mtp_dir: Option<std::path::PathBuf>,
}

/// 面向上层的模型加载器(经引擎的 face;spec 声明驱动)
pub struct ModelLoader<'a, D: DeviceClient> {
    face: &'a mut D,
}

impl<D: DeviceClient + 'static> ModelLoader<'_, D> {
    /// 构造(工厂;face 由 [`Engine::loader`](crate::Engine::loader) 借出)
    pub(crate) fn new<'a>(face: &'a mut D) -> ModelLoader<'a, D> {
        ModelLoader { face }
    }

    /// Qwen3.5-0.8B(维度/键基座从**检查点**读:config.json 优先 + 形状
    /// 兜底;族预设只供 tokenizer 声明;装载走通用入口)
    // A3 同步隔离:引擎单线程模型;Arc 仅作共享句柄,不跨线程发送。
    #[allow(clippy::arc_with_non_send_sync)]
    pub async fn load_qwen35_0_8b(&mut self, dir: &Path) -> Result<LoadedModel> {
        let (spec, rep) = ModelSpec::from_ckpt(dir, SpecSource::family(&qwen3_5_0_8b()))?;
        log_spec_report("0.8B", &rep);
        let model = Model::new(
            &spec,
            Qwen35Convention::new(convention_base(&rep)),
            owl_models::module::QuantPlan::F16,
            owl_models::module::QuantPlan::F16,
        );
        load_from_dir(dir, self.face, &model, LoadOpts::plain(spec.dtype)).await?;
        let model = Arc::new(model);
        let tokenizer = load_tokenizer(dir, &spec.tokenizer)?;
        let rope = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        let quant_plan = owl_models::module::QuantPlan::F16;
        let ctx = owl_models::module::LoaderCtx { dtype: spec.dtype, shard: 1, device_repack: false, verify: false, debug_tap: false };
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
        let model = Model::new(
            &spec,
            Qwen35Convention::new(convention_base(&rep)),
            owl_models::module::QuantPlan::W4A16,
            owl_models::module::QuantPlan::F16,
        );
        // 设备重排 = 关(与历史一致:0.8B 走 CPU 两步重排,量小不值 GPU 往返)
        load_from_dir(dir, self.face, &model, LoadOpts::plain(spec.dtype)).await?;
        let model = Arc::new(model);
        let tokenizer = load_tokenizer(tokenizer_dir, &spec.tokenizer)?;
        let rope = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        let quant_plan = owl_models::module::QuantPlan::W4A16;
        let ctx = owl_models::module::LoaderCtx { dtype: spec.dtype, shard: 1, device_repack: false, verify: false, debug_tap: false };
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
        );
        // 旋钮:设备重排 = 能力自述裁决(RepackPath::resolve 唯一收口)
        let mut opts = LoadOpts::new(self.face, spec.dtype);
        // 逐键分相账(OWL_LOAD_DEBUG=1;装载测速/瓶颈定位用 —— 与测试路同一开关)
        opts.debug_tap = owl_shared::env_reader::flag("OWL_LOAD_DEBUG");
        // 装载后整块回读校验(OWL_LOAD_VERIFY=1;纠错面 —— 装载数据面大改时的硬门)
        opts.verify = owl_shared::env_reader::flag("OWL_LOAD_VERIFY");
        load_from_dir(dir, self.face, &model, opts).await?;
        let model = Arc::new(model);
        let tokenizer = load_tokenizer(tokenizer_dir, &spec.tokenizer)?;
        let rope = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        let ctx = owl_models::module::LoaderCtx { dtype: spec.dtype, shard: 1, device_repack: false, verify: false, debug_tap: false };
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
            quant_plan,
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
