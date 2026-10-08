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
use owl_models::specs::{
    load_0_8b, load_0_8b_w4a16, load_27b_awq, load_tokenizer, qwen3_5_0_8b, qwen3_8_27b,
};
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

    /// Qwen3.5-0.8B(维度/键约定/分词器事实全在 specs/qwen35.rs 声明;
    /// 未来 `load(dir)` 按 config.json 分发,挂账)
    pub async fn load_qwen35_0_8b(&mut self, dir: &Path) -> Result<LoadedModel> {
        let spec = qwen3_5_0_8b();
        let model = Arc::new(load_0_8b(dir, self.face).await?);
        let tokenizer = load_tokenizer(dir)?;
        let rope = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        let quant_plan = owl_models::module::QuantPlan::F16;
        let ctx = owl_models::module::LoaderCtx { dtype: spec.dtype, shard: 1, device_repack: false, verify: false, debug_tap: false };
        owl_models::interpreters::eval_load(&rope, self.face, &rope.tables(), &ctx).await?;
        Ok(LoadedModel { model, tokenizer, rope, spec, quant_plan, mtp_dir: None })

    }

    /// W4A16 装载(E3;llm-compressor 产物目录):装载期重排到 marlin
    /// 布局,q_proj 等全部达标 Linear 走 foreign GEMM(REQ-PRE-01)
    pub async fn load_qwen35_0_8b_w4a16(
        &mut self,
        dir: &Path,
        tokenizer_dir: &Path,
    ) -> Result<LoadedModel> {
        let spec = qwen3_5_0_8b();
        let model = Arc::new(load_0_8b_w4a16(dir, self.face).await?);
        let tokenizer = load_tokenizer(tokenizer_dir)?;
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
    pub async fn load_qwen38_27b_awq(
        &mut self,
        dir: &Path,
        tokenizer_dir: &Path,
        head_plan: owl_models::module::QuantPlan,
    ) -> Result<LoadedModel> {
        let spec = qwen3_8_27b();
        // metrics 装载分相(直用 API 恒开;装载一次性路径,release 有数据)
        let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
        owl_shared::metrics::with_metrics_store(|s| {
            s.timer_begin("load.27b.total", file!(), line!());
        });
        let model = Arc::new(load_27b_awq(dir, self.face, head_plan).await?);
        let tokenizer = load_tokenizer(tokenizer_dir)?;
        let rope = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        let ctx = owl_models::module::LoaderCtx { dtype: spec.dtype, shard: 1, device_repack: false, verify: false, debug_tap: false };
        owl_models::interpreters::eval_load(&rope, self.face, &rope.tables(), &ctx).await?;
        let quant_plan = owl_models::module::QuantPlan::W4A16Awq;
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
