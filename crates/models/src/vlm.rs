//! VLM 装配(Qwen3.5 系:视觉塔 + 文本主干)—— **M4 注入文本模型**。
//!
//! 与 [`crate::model`] 的分工(同 `specs/` 拆分律):主干机制住 model,
//! **装配**住这里 —— 塔输出 [N, D] 怎么进文本流、位置号怎么建、
//! 检查点两段键怎么合,是本文件的全部职责,零新数学。
//!
//! 数据流(与 HF `Qwen3_5ForConditionalGeneration` 同位):
//!
//! ```text
//! pixel_values ─→ VisionTower ─→ merged [N, D]
//! input_ids ─→ embed [L, D] ─→ image_scatter(idx) ─→ layers ─→ norm ─→ lm_head
//!                       ▲                              ▲
//!                   host idx [L]                  host m-RoPE 表 [L, half]
//! ```
//!
//! 三个注入点各自显式(`ForwardCtx::with_vlm_image` / `with_vlm_rope` +
//! 本文件的 host 表),`None` 时 = 现役纯文本路径,**零节点零开销**。
//!
//! 装配律:
//! - 塔输出宽 = 文本 hidden(4B:2560)—— `Vlm::new` 期互证,不匹配响亮拒;
//! - 占位 token 数(ids 里 `image_token_id` 的个数)必须 = 塔输出行数,
//!   否则 [`Vlm::inputs`] 响亮拒(HF 在 `masked_scatter` 处静默错位);
//! - prefill 位置表 = `arange(L)` **rank 2 [L,1]**(现役 rope 核声明 rank 2,
//!   塔侧同款;m-RoPE 表按行号直取 ⇒ 行号即 token 号);
//! - decode 增量位 = `past + delta`(HF `mrope_position_deltas` 同义)。

use std::sync::Arc;

use crate::contract::{Dtype, ModelError};
use crate::layers::mrope::{split_video_frames, token_types, MropeCfg};
use crate::layers::vision::{GridThw, VisionCfg, VisionInputs, VisionTables, VisionTower};
use crate::model::{Model, ModelSpec};
use crate::module::{ForwardCtx, Loadable, LoaderCtx, LoaderOps, Module, PlanSource, QuantPlan};
use crate::formats::gguf::{mmproj_vision_cfg, rope_facts, token_id, Gguf};
use crate::formats::qwen35::Qwen35Convention;
use crate::TensorOps;

/// **一张图的预处理结果**(服务面 → 引擎的显式输入;值全为 host f32)。
///
/// `pixel_values` = `[n_patch, 3·temporal·16·16]`(processor 的 2×2 块序);
/// `grid` = `(t, h_patches, w_patches)`。**不在这里解码/缩放** —— 预处理是
/// 服务面(`vision_processor`)的职责;引擎只消费这个契约。
#[derive(Clone, Debug, PartialEq)]
pub struct VlmImage {
    pub pixel_values: Vec<f32>,
    pub grid: GridThw,
}

/// VLM 超参(`config.json` 单源;文本 + 塔 + 占位 token 三族)
pub struct VlmCfg {
    /// 塔超参
    pub vision: VisionCfg,
    /// m-RoPE 超参(sections / rotary_dim / theta / merge)
    pub mrope: MropeCfg,
    /// 文本 hidden(= 塔 out_hidden)
    pub hidden: usize,
    /// 文本层数
    pub layers: usize,
    /// 图像占位 token(248056)
    pub image_token: u32,
    /// 视频占位 token(248057;仅 `Some` 时参与位置号分段)
    pub video_token: Option<u32>,
    /// 检查点文本段前缀(`model.language_model`)
    pub text_prefix: String,
    /// 文本主干结构参数(config.json:text_config 直译;`Vlm::new` 消费)
    pub text_spec: ModelSpec,
}

impl VlmCfg {
    /// **从模型目录 `config.json` 读**(零手工抄数;Holo/4B 通吃)
    pub fn from_json(cfg_text: &[u8]) -> Result<VlmCfg, ModelError> {
        let v: serde_json::Value = serde_json::from_slice(cfg_text)
            .map_err(|e| ModelError::Msg(format!("vlm config.json 解析: {e}")))?;
        let text = v
            .get("text_config")
            .ok_or_else(|| ModelError::Msg("config.json 无 text_config".into()))?;
        let g = |k: &str| -> Result<usize, ModelError> {
            text.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| ModelError::Msg(format!("text_config 缺 {k}")))
        };
        let gp = |k: &str| -> Result<f32, ModelError> {
            text.get("rope_parameters")
                .and_then(|r| r.get(k))
                .and_then(|x| x.as_f64())
                .map(|x| x as f32)
                .ok_or_else(|| ModelError::Msg(format!("rope_parameters 缺 {k}")))
        };
        let vision = VisionCfg::from_json(cfg_text)?;
        let head_dim = g("head_dim")?;
        let partial = gp("partial_rotary_factor")?;
        let rotary_dim = (head_dim as f32 * partial).round() as usize;
        let sections = text
            .get("rope_parameters")
            .and_then(|r| r.get("mrope_section"))
            .and_then(|x| x.as_array())
            .map(|a| {
                // 元素必须是非负整数(2026-10-17 审查 §六轮:原 `as_u64().unwrap_or(0)`
                // 是静默回退 —— 非数值元素会变成 0 段,切分悄悄变形)
                a.iter()
                    .map(|x| {
                        x.as_u64().map(|v| v as usize).ok_or_else(|| {
                            ModelError::Msg(format!(
                                "mrope_section 元素须为非负整数,实得 {x}"
                            ))
                        })
                    })
                    .collect::<Result<Vec<usize>, ModelError>>()
            })
            .transpose()?
            .ok_or_else(|| ModelError::Msg("rope_parameters 缺 mrope_section".into()))?;
        if sections.len() != 3 {
            return Err(ModelError::Msg(format!(
                "mrope_section 需三网格 [T,H,W],实得 {sections:?}"
            )));
        }
        let hidden = g("hidden_size")?;
        if vision.out_hidden != hidden {
            return Err(ModelError::Msg(format!(
                "塔 out_hidden {} ≠ 文本 hidden {hidden}(装配不成立)",
                vision.out_hidden
            )));
        }
        let image_token = v
            .get("image_token_id")
            .and_then(|x| x.as_u64())
            .ok_or_else(|| ModelError::Msg("config.json 缺 image_token_id".into()))?
            as u32;
        let video_token = v
            .get("video_token_id")
            .and_then(|x| x.as_u64())
            .map(|x| x as u32);
        let layer_types = text
            .get("layer_types")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .map(|x| x.as_str() == Some("full_attention"))
                    .collect::<Vec<bool>>()
            })
            // 缺 `layer_types` ⇒ 按**家族规则**派生(3 GDN : 1 FA);
            // `num_hidden_layers` 缺失/为 0 = 层型表不可定 ⇒ **响亮拒**
            // (原 `unwrap_or(0)` 会静默生成空层型表 = "没有 GDN/FA 层" 的错模型)
            .map_or_else(
                || {
                    let n = g("num_hidden_layers")?;
                    if n == 0 {
                        return Err(ModelError::Msg(
                            "text_config.num_hidden_layers = 0(层型表不可定)".into(),
                        ));
                    }
                    Ok(crate::formats::qwen35::hybrid_3to1(n))
                },
                Ok,
            )?;
        // eos 字面量:Qwen3.5 家族共享(vocab 248320;`<|im_end|>` = 248044)
        let eos: Vec<&'static str> = if text.get("eos_token_id").is_some() {
            vec!["<|im_end|>"]
        } else {
            Vec::new()
        };
        let text_spec = ModelSpec {
            vocab: g("vocab_size")?,
            head_vocab: g("vocab_size")?, // VLM 不进 TP(v1):头 = 全量
            head_split: false,
            hidden,
            inter: g("intermediate_size")?,
            dtype: Dtype::F16,
            full_heads: (g("num_attention_heads")?, g("num_key_value_heads")?, head_dim),
            gdn_heads: (
                g("linear_num_key_heads")?,
                g("linear_key_head_dim")?,
                g("linear_num_value_heads")?,
                g("linear_value_head_dim")?,
            ),
            eps: gp("rms_norm_eps").unwrap_or_else(|_| {
                text.get("rms_norm_eps").and_then(|x| x.as_f64()).unwrap_or(1e-6) as f32
            }),
            layer_types,
            // FIXME(M5 服务面):chat 注入文本需按 4B 的 `chat_template.jinja`
            // 逐字校订(思考开关 / reasoning effort);本刀走 id 直喂
            // (处理器产 input_ids),不进这条链。
            tokenizer: crate::tokenizer::TokenizerSpec {
                eos_tokens: eos,
                chat: crate::tokenizer::ChatFormat {
                    prefix: String::new(),
                    suffix: String::new(),
                    im_open: String::new(),
                    im_close: String::new(),
                    assistant_open: String::new(),
                    think_prefill: String::new(),
                },
            },
            tied: v
                .get("tie_word_embeddings")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
        };
        Ok(VlmCfg {
            mrope: MropeCfg::new(
                [sections[0], sections[1], sections[2]],
                rotary_dim,
                gp("rope_theta")?,
                vision.merge,
            )?,
            vision,
            hidden,
            layers: g("num_hidden_layers")?,
            image_token,
            video_token,
            text_prefix: "model.language_model".into(),
            text_spec,
        })
    }

    /// **从 GGUF 双文件读 VLM 超参**(2026-10-21):文本 `.gguf` 提供
    /// text/mRoPE/占位 token,mmproj `.gguf` 提供视觉塔。
    ///
    /// 与 [`Self::from_json`] 同律:三处（塔宽 = 文本 hidden、mRoPE 段和、
    /// 图像占位 token 在场）缺一/不符 = 响亮拒。GGUF 没有 `image_token_id`
    /// 冗余键 ⇒ 从 `tokenizer.ggml.tokens` 按词面查 id（`<|image_pad|>`）。
    ///
    /// `text_spec` = 引擎已推导的文本 spec（含 MTP 块排除后的解码层数）。
    pub fn from_gguf(
        text: &Gguf,
        mmproj: &Gguf,
        text_spec: ModelSpec,
    ) -> Result<VlmCfg, ModelError> {
        let vision = mmproj_vision_cfg(mmproj)?;
        if vision.out_hidden != text_spec.hidden {
            return Err(ModelError::Msg(format!(
                "GGUF VLM:塔 out_hidden {} ≠ 文本 hidden {}（装配不成立）",
                vision.out_hidden, text_spec.hidden
            )));
        }
        let rf = rope_facts(text)?;
        let sections = text
            .kv
            .opt_u32s("qwen35.rope.dimension_sections")
            .map_err(|e| ModelError::Msg(format!("GGUF VLM mRoPE: {e}")))?
            .ok_or_else(|| {
                ModelError::Msg(
                    "GGUF VLM: 缺 `qwen35.rope.dimension_sections`（不是 Qwen3.5-VL？）".into(),
                )
            })?;
        let sec = match sections.as_slice() {
            [a, b, c] => [*a as usize, *b as usize, *c as usize],
            [a, b, c, 0] => [*a as usize, *b as usize, *c as usize],
            other => {
                return Err(ModelError::Msg(format!(
                    "GGUF VLM: rope.dimension_sections={other:?}（期望 [T,H,W] 或 [T,H,W,0]）"
                )))
            }
        };
        let mrope = MropeCfg::new(sec, rf.rotary_dim, rf.theta, vision.merge)?;
        let image_token = token_id(text, "<|image_pad|>")?.ok_or_else(|| {
            ModelError::Msg("GGUF VLM: 词表缺 `<|image_pad|>`（图像占位 token 不可定）".into())
        })?;
        let video_token = token_id(text, "<|video_pad|>")?;
        if video_token == Some(image_token) {
            return Err(ModelError::Msg(
                "GGUF VLM: `<|image_pad|>` 与 `<|video_pad|>` 映射到同一 id".into(),
            ));
        }
        let hidden = text_spec.hidden;
        let layers = text_spec.layer_types.len();
        if layers == 0 {
            return Err(ModelError::Msg("GGUF VLM: 文本层数为 0".into()));
        }
        Ok(VlmCfg {
            vision,
            mrope,
            hidden,
            layers,
            image_token,
            video_token,
            // GGUF 文本主干用 `GgufConvention`；本字段只服务 `Vlm::new` 的
            // HF 装配，`from_model` 路径不读它。
            text_prefix: String::new(),
            text_spec,
        })
    }
}

/// VLM 一步的 host 表(一次 `build` 出齐;`prefill` 原样消费)
pub struct VlmInputs {
    /// 位置表 [L, 1] f32 = arange(L)(m-RoPE 表按行号直取)
    pub pos: TensorOps,
    /// 行号表 [L] f32:≥ 0 = 取塔输出行;负 = 文本位保留 embed
    pub idx: TensorOps,
    /// 行号表 host 副本(断言/槽位映射用;与 `idx` 同源)
    pub idx_host: Vec<f32>,
    /// m-RoPE cos 行表 [L, half]
    pub cos: TensorOps,
    /// m-RoPE sin 行表 [L, half]
    pub sin: TensorOps,
    /// m-RoPE host 副本(chunk 切片用;元素 = f32)
    pub cos_host: Vec<f32>,
    pub sin_host: Vec<f32>,
    /// 每行 cos/sin 宽度(head_dim/2)
    pub half: usize,
    /// cos/sin 的打包 dtype(f16 = 塔/prefill 档)
    pub dtype: Dtype,
    /// 序列长
    pub tokens: usize,
    /// decode 增量位(HF `mrope_position_deltas`)
    pub delta: i64,
}

impl VlmInputs {
    /// **chunk 切片**(引擎 prefill 每块用):返回
    /// `(cos, sin, idx, pos, has_image)`。
    ///
    /// `base..base+t` 是 prompt 行号区间;行号表/余弦表按行切;`pos` 是
    /// **本 chunk 局部行号 0..n**(rope 核按 `pos[t]` 索引 cos/sin 行表,
    /// 不能传全局 `base+t`);`has_image` = 本块含 ≥1 个视觉占位(`idx >= 0`),
    /// 否则调用方跳过 `image_scatter`(省一次无意义的全 `-1` 扫描)。
    pub fn chunk_tables(
        &self,
        base: usize,
        t: usize,
    ) -> (TensorOps, TensorOps, TensorOps, TensorOps, bool) {
        let lo = base.min(self.tokens);
        let hi = base.saturating_add(t).min(self.tokens);
        let n = hi.saturating_sub(lo);
        let pack = |v: &[f32]| -> Vec<u8> {
            match self.dtype {
                Dtype::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                Dtype::F16 => v
                    .iter()
                    .flat_map(|x| half::f16::from_f32(*x).to_le_bytes())
                    .collect(),
                other => unreachable!("VlmInputs::chunk_tables: dtype {other:?}"),
            }
        };
        let cos = &self.cos_host[lo * self.half..hi * self.half];
        let sin = &self.sin_host[lo * self.half..hi * self.half];
        let idx = &self.idx_host[lo..hi];
        let has_image = idx.iter().any(|&x| x >= 0.0);
        let idx_bytes: Vec<u8> = idx.iter().flat_map(|x| x.to_le_bytes()).collect();
        // **rope 的 pos 必须相对本 chunk 的行表**(2026-10-23):
        // `chunk_tables` 的 cos/sin 是 `[n, half]` 行表，核按 `pos[t]` 取行；
        // 若 pos 仍传全局 `base+t`，第二个 chunk 起会越出本 chunk 表
        // （首 chunk 恰好 base=0 而正确，多 chunk 全错）。
        let pos_bytes: Vec<u8> = (0..n)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
        (
            TensorOps::from_host(self.dtype, vec![n, self.half], &pack(cos)),
            TensorOps::from_host(self.dtype, vec![n, self.half], &pack(sin)),
            TensorOps::from_host(Dtype::F32, vec![n], &idx_bytes),
            TensorOps::from_host(Dtype::F32, vec![n], &pos_bytes),
            has_image,
        )
    }
}

/// Qwen3.5 系 VLM = 视觉塔 + 文本主干(装配,零机制)
pub struct Vlm {
    tower: VisionTower,
    /// 文本主干(**与引擎共享**;`from_model` 用它接引擎已装好的量化/分片模型)
    model: Arc<Model>,
    cfg: VlmCfg,
}

impl Vlm {
    /// 装配(`spec` 从 `cfg.text_spec` 取 —— 单一真源)。
    ///
    /// 装配期互证:塔输出宽 = 文本 hidden(`VlmCfg::from_json` 已查,这里
    /// 再按实际 spec 查一遍 —— 两处入口不互相假定)。
    pub fn new<'a>(
        cfg: VlmCfg,
        plan: impl Into<PlanSource<'a>>,
        head_plan: impl Into<PlanSource<'a>>,
    ) -> Result<Vlm, ModelError> {
        let spec = &cfg.text_spec;
        if cfg.vision.out_hidden != spec.hidden {
            return Err(ModelError::Msg(format!(
                "Vlm::new: 塔 out_hidden {} ≠ 文本 hidden {}(装配不成立)",
                cfg.vision.out_hidden, spec.hidden
            )));
        }
        Ok(Vlm {
            tower: VisionTower::new(cfg.vision),
            model: Arc::new(Model::new(
                spec,
                Qwen35Convention::new(&cfg.text_prefix),
                plan,
                head_plan,
            )?),

            cfg,
        })
    }

    /// **用引擎已装好的文本模型装配**(2026-10-20 视觉接线第一刀)。
    ///
    /// 引擎的文本模型已经过 `ModelLoader`(分片/量化/头计划)⇒ 塔只补视觉段,
    /// 文本主干**不再重装**;`model_arc` 与引擎的 `RankDev.model` 同一份。
    /// 装配互证沿用 [`Self::new`] 的同一条(塔 out_hidden = 文本 hidden)。
    pub fn from_model(
        model: Arc<Model>,
        tower: VisionTower,
        cfg: VlmCfg,
    ) -> Result<Vlm, ModelError> {
        if cfg.vision.out_hidden != model.hidden() {
            return Err(ModelError::Msg(format!(
                "Vlm::from_model: 塔 out_hidden {} ≠ 文本 hidden {}(装配不成立)",
                cfg.vision.out_hidden,
                model.hidden()
            )));
        }
        Ok(Vlm { tower, model, cfg })
    }

    /// 引擎共享的文本主干(同一份 `Arc`;调用方不得重新 `Arc::new`)
    pub fn model_arc(&self) -> Arc<Model> {
        Arc::clone(&self.model)
    }

    /// 文本主干只读视图
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// **从模型目录整装**(两段键同源:文本 `model.language_model.*` 经
    /// [`Qwen35Convention`](GDN `linear_attn.` / FA `self_attn.` / `mlp.`
    /// 三处子前缀 + A_log/dt_bias 裸键),塔 `model.visual.*` 自带前缀;一份
    /// [`SafeTensorsSource`] 供给两边)。
    ///
    /// `dtype` = 计算 dtype(权重 bf16 检查点 → f16 字节直转,不经 f32 中转)。
    pub async fn load<D: crate::contract::DeviceClient + 'static>(
        dir: &std::path::Path,
        face: &mut D,
        dtype: Dtype,
    ) -> Result<Vlm, ModelError> {
        Self::load_ctx(dir, face, LoaderCtx { dtype, shard: 1, slice_id: 0, device_repack: false, verify: false, debug_tap: false, staging_mb: 128, dump: false }).await
    }

    /// 同上,装载 ctx 外给(分相探针 / 校验 / 分片调试面)
    pub async fn load_ctx<D: crate::contract::DeviceClient + 'static>(
        dir: &std::path::Path,
        face: &mut D,
        ctx: LoaderCtx,
    ) -> Result<Vlm, ModelError> {
        let raw = owl_shared::file_loader::read(&dir.join("config.json")).map_err(|e| {
            ModelError::Msg(format!("Vlm::load: config.json 读入 {}: {e}", dir.display()))
        })?;
        let cfg = VlmCfg::from_json(&raw)?;
        let vlm = Vlm::new(cfg, QuantPlan::F16, QuantPlan::F16)?;
        // 装载走通用入口(起点探测 → 开源 → 求值);ctx → LoadOpts 单一映射
        // (`shard` 是 TP 预留词,单卡恒 1,不进旋钮面)
        let opts = crate::formats::load::LoadOpts {
            dtype: ctx.dtype,
            shard: ctx.shard,
            slice_id: ctx.slice_id,
            device_repack: Some(ctx.device_repack),
            verify: ctx.verify,
            debug_tap: ctx.debug_tap,
            staging_mb: 128,
            dump: false,
            gguf_dequant: false,
        };
        crate::formats::load::load_from_dir(dir, face, &vlm, opts).await?;
        Ok(vlm)
    }

    pub fn cfg(&self) -> &VlmCfg {
        &self.cfg
    }

    pub fn hidden(&self) -> usize {
        self.model.hidden()
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    /// 塔叶(前向输入;`px` = [rows, patch_flat] host f32)。塔的 interp /
    /// axial rope / cu 三张 host 表由 [`VisionTables::build`] 按网格现算。
    pub fn tower_inputs(&self, grids: &[GridThw], px: &[f32], dtype: Dtype) -> Result<VisionInputs, ModelError> {
        VisionTables::build(grids, self.cfg.vision)?.inputs(px, dtype)
    }

    /// 塔前向(出 merged [N, D];只在 prefill 首步用)
    pub fn tower_forward(&self, tb: &VisionInputs) -> TensorOps {
        self.tower.forward(tb).merged
    }

    /// 备好一步请求(host 表 + 塔输出);调用方再用
    /// [`VlmRequest::bind`] 把它绑到 prefill ctx 上。
    pub fn prepare(
        &self,
        ids: &[u32],
        grids: &[GridThw],
        video_grids: &[GridThw],
        px: &[f32],
    ) -> Result<VlmRequest, ModelError> {
        let ins = self.inputs(ids, grids, video_grids, Dtype::F16)?;
        self.prepare_from_inputs(ins, grids, video_grids, px)
    }

    /// **显式位置的备料**(leave-gap;2026-10-25):塔表照旧,文本位置表由
    /// `positions` 直接生成。
    pub fn prepare_with_positions(
        &self,
        ids: &[u32],
        grids: &[GridThw],
        video_grids: &[GridThw],
        px: &[f32],
        positions: &[[i32; 3]],
    ) -> Result<VlmRequest, ModelError> {
        let ins = self.inputs_with_positions(ids, grids, video_grids, Dtype::F16, positions)?;
        self.prepare_from_inputs(ins, grids, video_grids, px)
    }

    fn prepare_from_inputs(
        &self,
        ins: VlmInputs,
        grids: &[GridThw],
        video_grids: &[GridThw],
        px: &[f32],
    ) -> Result<VlmRequest, ModelError> {
        // 纯文本请求(无网格):不进塔(塔表构造器拒空网格),注入面自然为空
        let merged = if grids.is_empty() && video_grids.is_empty() {
            None
        } else {
            let tb = self.tower_inputs(grids, px, Dtype::F16)?;
            let m = self.tower_forward(&tb);
            if m.shape.last().copied() != Some(self.model.hidden()) {
                return Err(ModelError::Msg(format!(
                    "Vlm::prepare: 塔输出宽 {:?} ≠ 文本 hidden {}",
                    m.shape,
                    self.model.hidden()
                )));
            }
            Some(m)
        };
        Ok(VlmRequest { ins, merged })
    }

    /// prefill 一步(根 = logits [L, vocab])。
    ///
    /// **响亮拒**:ctx 缺任一面注入 ⇒ 拒发(而不是静默走纯文本)。
    pub fn prefill_logits(&self, ids: &TensorOps, ctx: &ForwardCtx<'_>) -> Result<TensorOps, ModelError> {
        if ctx.vlm_rope.is_none() {
            return Err(ModelError::Msg(
                "Vlm::prefill_logits: ctx 缺 VLM 注入(需 VlmRequest::bind;宁拒不错)".into(),
            ));
        }
        Ok(self.model.forward(ids, ctx))
    }

    /// 文本一步(logits [L, vocab];decode 步同用 —— 注入面 None ✓)
    pub fn text_logits(&self, ids: &TensorOps, ctx: &ForwardCtx<'_>) -> TensorOps {
        self.model.forward(ids, ctx)
    }

    /// 一步的 host 表(位置号 + 行号 + m-RoPE 表)。
    ///
    /// **响亮拒绝**:占位 token 数与网格展开不符 / 网格与 token 类型不符 /
    /// 位置表宽 ≠ 塔 out_hidden —— 三处都是 HF 静默错位点。
    pub fn inputs(
        &self,
        ids: &[u32],
        grids: &[GridThw],
        video_grids: &[GridThw],
        dtype: Dtype,
    ) -> Result<VlmInputs, ModelError> {
        let ttypes = token_types(ids, self.cfg.image_token, self.cfg.video_token);
        let frames = split_video_frames(video_grids);
        let index = self.cfg.mrope.rope_index(ids, &ttypes, grids, &frames)?;
        self.inputs_from_positions(ids, grids, video_grids, dtype, index.pos, index.delta)
    }

    /// **显式位置输入**(leave-gap;2026-10-25):cos/sin 直接由调用方给的
    /// 逐 token 绝对位置生成,不再从 0 重算。行号/塔输出校验与 [`Self::inputs`]
    /// 同一条(单一产地)。
    pub fn inputs_with_positions(
        &self,
        ids: &[u32],
        grids: &[GridThw],
        video_grids: &[GridThw],
        dtype: Dtype,
        positions: &[[i32; 3]],
    ) -> Result<VlmInputs, ModelError> {
        if positions.len() != ids.len() {
            return Err(ModelError::Msg(format!(
                "Vlm::inputs_with_positions: 位置 {} ≠ token {}",
                positions.len(),
                ids.len()
            )));
        }
        let pos: Vec<[i64; 3]> = positions
            .iter()
            .map(|p| [p[0] as i64, p[1] as i64, p[2] as i64])
            .collect();
        let max_pos = pos.iter().flat_map(|p| p.iter()).copied().max().unwrap_or(0);
        let delta = max_pos + 1 - ids.len() as i64;
        self.inputs_from_positions(ids, grids, video_grids, dtype, pos, delta)
    }

    /// 位置表 → `VlmInputs`(两条入口的唯一实现)。
    fn inputs_from_positions(
        &self,
        ids: &[u32],
        grids: &[GridThw],
        video_grids: &[GridThw],
        dtype: Dtype,
        positions: Vec<[i64; 3]>,
        delta: i64,
    ) -> Result<VlmInputs, ModelError> {
        let l = ids.len();
        let ttypes = token_types(ids, self.cfg.image_token, self.cfg.video_token);
        let frames = split_video_frames(video_grids);
        if positions.len() != l {
            return Err(ModelError::Msg(format!(
                "Vlm::inputs_from_positions: 位置 {} ≠ token {l}",
                positions.len()
            )));
        }
        // 行号表:第 k 个视觉 token → 塔输出第 k 行(HF `masked_scatter` 同序)
        let mut idx = vec![-1.0f32; l];
        let mut n_vis = 0usize;
        for (t, &ty) in ttypes.iter().enumerate() {
            if ty != crate::layers::mrope::TYPE_TEXT {
                idx[t] = n_vis as f32;
                n_vis += 1;
            }
        }
        let n_merged: usize = grids
            .iter()
            .map(|&(t, h, w)| t * (h / self.cfg.vision.merge) * (w / self.cfg.vision.merge))
            .sum::<usize>()
            + frames
                .iter()
                .map(|&(t, h, w)| t * (h / self.cfg.vision.merge) * (w / self.cfg.vision.merge))
                .sum::<usize>();
        if n_vis != n_merged {
            return Err(ModelError::Msg(format!(
                "Vlm::inputs: 视觉占位 token {n_vis} ≠ 塔输出行数 {n_merged}(网格/占位不符)"
            )));
        }
        let (mcos, msin) = self.cfg.mrope.cos_sin(&positions);
        let half = self.cfg.mrope.half();
        let pack = |v: &[f32]| -> Vec<u8> {
            match dtype {
                Dtype::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                Dtype::F16 => v.iter().flat_map(|x| half::f16::from_f32(*x).to_le_bytes()).collect(),
                other => unreachable!("Vlm::inputs: dtype {other:?}(塔只走 f32/f16)"),
            }
        };
        let pos_host: Vec<f32> = (0..l).map(|t| t as f32).collect();
        Ok(VlmInputs {
            pos: TensorOps::from_host(Dtype::F32, vec![l, 1], &pack_f32(&pos_host)),
            idx: TensorOps::from_host(Dtype::F32, vec![l], &pack_f32(&idx)),
            idx_host: idx,
            cos: TensorOps::from_host(dtype, vec![l, half], &pack(&mcos)),
            sin: TensorOps::from_host(dtype, vec![l, half], &pack(&msin)),
            cos_host: mcos,
            sin_host: msin,
            half,
            dtype,
            tokens: l,
            delta,
        })
    }
}

/// f32 host 值 → 字节(位置/行号表恒 f32 过线)
fn pack_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// 一步请求(host 表 + 塔输出;生命周期 = 调用方本地)
pub struct VlmRequest {
    /// 位置号 / 行号 / m-RoPE 表
    pub ins: VlmInputs,
    /// 塔输出 [N, D](纯文本请求 = None ⇒ 不注入图像面)
    pub merged: Option<TensorOps>,
}

impl VlmRequest {
    /// 把两面注入 + 行号位置表绑到调用方建好的 prefill ctx 上(消费式;无 Clone 需求)。
    ///
    /// pos 由本函数接管(= `ins.pos` = arange 行号)⇒ 无“忘设 pos”这一错法。
    pub fn bind<'a>(&'a self, base: ForwardCtx<'a>) -> ForwardCtx<'a> {
        let ctx = base
            .with_pos(&self.ins.pos)
            .with_vlm_rope(&self.ins.cos, &self.ins.sin);
        match &self.merged {
            Some(m) => ctx.with_vlm_image(m, &self.ins.idx),
            None => ctx,
        }
    }
}

impl Loadable for Vlm {
    /// 两段键各归其位:文本走 [`KeyConvention`](`model.language_model.*`),
    /// 塔自带 `model.visual.*` 前缀(见 `VisionTower::layout`)。
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        self.model.layout(ctx).chain(self.tower.layout(ctx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// chunk 切片:m-RoPE/行号按行切,`has_image` 只在含视觉占位的块为真;
    /// **pos 必须是本 chunk 的局部行号**(cos/sin 是 chunk 行表)。
    #[tokio::test]
    async fn chunk_tables_slice_and_image_flag() {
        let f32b = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let ins = VlmInputs {
            pos: TensorOps::from_host(Dtype::F32, vec![4, 1], &f32b(&[0., 1., 2., 3.])),
            idx: TensorOps::from_host(Dtype::F32, vec![4], &f32b(&[-1., 0., -1., 1.])),
            idx_host: vec![-1., 0., -1., 1.],
            cos: TensorOps::from_host(Dtype::F32, vec![4, 2], &f32b(&[0.; 8])),
            sin: TensorOps::from_host(Dtype::F32, vec![4, 2], &f32b(&[0.; 8])),
            cos_host: vec![0.; 8],
            sin_host: vec![0.; 8],
            half: 2,
            dtype: Dtype::F32,
            tokens: 4,
            delta: 3,
        };
        let (_, _, _, pos, has) = ins.chunk_tables(1, 2);
        assert!(has, "chunk 1..3 含 idx=0");
        let mut face = owl_cpu::CpuFace::new();
        let got = harvest(&mut face, &pos).await;
        assert_eq!(got, vec![0.0, 1.0], "chunk pos 必须从 0 起(局部行号)");
        let (_, _, _, _, no) = ins.chunk_tables(0, 1);
        assert!(!no, "chunk 0..1 全文本");
        let (_, _, _, pos, tail) = ins.chunk_tables(3, 8);
        assert!(tail, "越界钳到 tokens");
        let got = harvest(&mut face, &pos).await;
        assert_eq!(got, vec![0.0], "尾块 pos 同样从 0 起");
    }

    /// **27B 真实 config.json 的 VLM 超参可解析**(纯 JSON,不装权重/不占卡)。
    /// 视觉接线第一刀的配置门:`vision_config` / `image_token_id` / mrope
    /// 三族缺一,`VlmCfg::from_json` 必须响亮拒。
    #[test]
    fn cfg_27b_vision_parses_from_real_config() {
        let Some(dir) = owl_shared::testkit::resource("ckpt:27b-awq") else {
            return; // 资源清单缺省 = skip(非失败)
        };
        let raw = owl_shared::file_loader::read(&std::path::Path::new(&dir).join("config.json"))
            .expect("27B config.json");
        let cfg = VlmCfg::from_json(&raw).expect("VlmCfg::from_json");
        assert_eq!(cfg.hidden, 5120, "文本 hidden");
        assert_eq!(cfg.vision.out_hidden, 5120, "塔 out_hidden 必须 = 文本 hidden");
        assert_eq!(cfg.image_token, 248056, "image_token_id");
    }
    use crate::testkit::{f32b, harvest};

    /// 4B 档模型目录(config.json 单源;可 `ckpt:qwen35-4b` 改指)
    fn ckpt_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(
            owl_shared::testkit::resource("ckpt:qwen35-4b")
                .unwrap_or_else(|| concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/Qwen/Qwen3.5-4B").into()),
        )
    }

    fn cfg4b() -> VlmCfg {
        let raw = owl_shared::file_loader::read(&ckpt_dir().join("config.json"))
            .expect("config.json 读入(ckpt:qwen35-4b 可指路)");
        VlmCfg::from_json(&raw).expect("VlmCfg::from_json")
    }

    /// config.json → 超参 + 文本 spec(零手工抄数)
    #[test]
    fn cfg_from_real_config() {
        let c = cfg4b();
        assert_eq!(c.hidden, 2560);
        assert_eq!(c.layers, 32);
        assert_eq!(c.image_token, 248056);
        assert_eq!(c.video_token, Some(248057));
        assert_eq!((c.vision.depth, c.vision.hidden, c.vision.out_hidden), (24, 1024, 2560));
        assert_eq!(c.mrope.half(), 32, "rotary_dim 64 → half 32");
        let s = &c.text_spec;
        assert_eq!(s.vocab, 248320);
        assert_eq!(s.hidden, 2560);
        assert_eq!(s.inter, 9216);
        assert_eq!(s.full_heads, (16, 4, 256));
        assert_eq!(s.gdn_heads, (16, 128, 32, 128));
        assert_eq!(s.layer_types.len(), 32);
        assert_eq!(s.layer_types.iter().filter(|&&f| f).count(), 8, "8 FA + 24 GDN");
        assert!(s.tied, "4B tie_word_embeddings = true");
    }

    /// host 表:行号 / 位置号 / m-RoPE 表 / delta(手算锚)+ 响亮拒
    #[test]
    fn inputs_hand_computed_and_loud() {
        let c = cfg4b();
        let vlm = match Vlm::new(c, QuantPlan::F16, QuantPlan::F16) {
            Ok(v) => v,
            Err(e) => panic!("装配: {e}"),
        };
        // [文本 3][图像 6][文本 2],网格 (1,4,6) → llm (1,2,3) = 6 行
        let mut ids = vec![10u32, 11, 12];
        ids.extend(vec![248056u32; 6]);
        ids.extend(vec![13u32, 14]);
        let ins = vlm
            .inputs(&ids, &[(1, 4, 6)], &[], Dtype::F16)
            .expect("inputs");
        assert_eq!(ins.tokens, 11);
        let want_idx = [-1.0, -1.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -1.0];
        for (i, (g, w)) in ins.idx_host.iter().zip(&want_idx).enumerate() {
            assert_eq!(g, w, "idx[{i}]");
        }
        assert_eq!(ins.pos.shape, vec![11, 1], "位置表 rank 2(核线契约)");
        assert_eq!(ins.cos.shape, vec![11, 32]);
        assert_eq!(ins.delta, -3, "delta = max+1-L = 7+1-11");
        // 占位 token 数 ≠ 网格展开 ⇒ 响亮拒(HF 在 masked_scatter 静默错位)
        let short = vec![248056u32; 5];
        assert!(vlm.inputs(&short, &[(1, 4, 6)], &[], Dtype::F16).is_err(), "占位 5 ≠ 6 应拒");
        assert!(vlm.inputs(&ids, &[(1, 4, 6), (1, 2, 2)], &[], Dtype::F16).is_err(), "多给网格应拒");
    }

    /// **相邻多图**:占位 token → 塔行号连续;m-RoPE 两张图分别推进。
    #[test]
    fn inputs_multiple_adjacent_images() {
        let c = cfg4b();
        let vlm = Vlm::new(c, QuantPlan::F16, QuantPlan::F16).expect("装配");
        let ids = vec![248056u32; 6 + 2];
        let ins = vlm
            .inputs(&ids, &[(1, 4, 6), (1, 2, 4)], &[], Dtype::F16)
            .expect("相邻两图 inputs");
        assert_eq!(ins.idx_host, (0..8).map(|i| i as f32).collect::<Vec<_>>());
        assert_eq!(ins.delta, -3, "两图最大位置 4 + 1 - 8");
    }

    /// image_scatter:CPU face vs host 手算锚 + 越界响亮拒(第三份独立副本)
    #[tokio::test]
    async fn image_scatter_matches_host_anchor() {
        let (l, n, dim) = (5usize, 2usize, 3usize);
        let h: Vec<f32> = (0..l * dim).map(|i| i as f32).collect();
        let merged: Vec<f32> = (0..n * dim).map(|i| 100.0 + i as f32).collect();
        let idx: Vec<f32> = vec![-1.0, 0.0, -1.0, 1.0, -1.0];

        let mut face = owl_cpu::CpuFace::new();
        let ht = TensorOps::from_host(Dtype::F32, vec![l, dim], &f32b(&h));
        let mt = TensorOps::from_host(Dtype::F32, vec![n, dim], &f32b(&merged));
        let it = TensorOps::from_host(Dtype::F32, vec![l], &f32b(&idx));
        let got = harvest(&mut face, &ht.image_scatter(&mt, &it)).await;

        for t in 0..l {
            let want: Vec<f32> = if idx[t] < 0.0 {
                h[t * dim..(t + 1) * dim].to_vec()
            } else {
                merged[(idx[t] as usize) * dim..(idx[t] as usize + 1) * dim].to_vec()
            };
            assert_eq!(&got[t * dim..(t + 1) * dim], &want[..], "行 {t}");
        }

        // 越界行号:CPU 面响亮拒(不许静默读飞)
        let bad = TensorOps::from_host(Dtype::F32, vec![l], &f32b(&[-1.0, 2.0, -1.0, -1.0, -1.0]));
        let out = ht.image_scatter(&mt, &bad);
        assert!(
            crate::interpreters::eval_ops(out.step(), &mut face).await.is_err(),
            "行号 2 ≥ merged 行数 2 ⇒ 应拒"
        );
    }

    // ========================================================================
    // 真权重面(4B;重,双门控 flag:vlm-real=1 + OWL_TEST_DEVICE)
    // ========================================================================

    /// 生产面常驻缓冲(f16 模型 ⇒ paged KV:页 32 + kNHD K 池 + 恒等块表;
    /// GDN 三段 conv + rec 恒 f32,槽数 = 序列槽 8(单序列用槽 0))
    async fn real_bufs(
        gpu: &mut owl_cuda::GpuClient,
        c: &VlmCfg,
        nb: usize,
        slots_n: usize,
    ) -> (Vec<crate::module::KvBuffers>, Vec<crate::layers::gdn::GdnBuffers>) {
        use crate::layers::gdn::GdnBuffers;
        use crate::module::KvBuffers;
        let (hq, hkv, hd) = c.text_spec.full_heads;
        let _ = hq;
        let _ = hd;
        let (nk, hk_dim, nv, hv_dim) = c.text_spec.gdn_heads;
        let key_dim = nk * hk_dim;
        let value_dim = nv * hv_dim;
        let bt_data: Vec<u8> = (0..nb).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let bt = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F32, vec![1, nb], &bt_data).step(),
            gpu,
        )
        .await
        .expect("恒等块表");
        let n_full = c.text_spec.layer_types.iter().filter(|&&f| f).count();
        let n_gdn = c.text_spec.layer_types.len() - n_full;
        let mut kvs = Vec::new();
        for _ in 0..n_full {
            kvs.push(KvBuffers {
                k_cache: zero_f16(gpu, nb * hkv * (hd / 8) * 32 * 8, vec![nb, hkv, hd / 8, 32, 8]).await,
                v_cache: zero_f16(gpu, nb * hkv * hd * 32, vec![nb, hkv, hd, 32]).await,
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0])),
                block_tables: TensorOps::of_block(bt.id, Dtype::F32, vec![1, nb]),
            });
        }
        let mut gdns = Vec::new();
        for _ in 0..n_gdn {
            gdns.push(GdnBuffers {
                conv_q: zero_f32(gpu, slots_n * key_dim * 3, vec![slots_n, key_dim, 3]).await,
                conv_k: zero_f32(gpu, slots_n * key_dim * 3, vec![slots_n, key_dim, 3]).await,
                conv_v: zero_f32(gpu, slots_n * value_dim * 3, vec![slots_n, value_dim, 3]).await,
                rec: zero_f32(gpu, slots_n * nv * hk_dim * hv_dim, vec![slots_n, nv, hk_dim, hv_dim]).await,
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
            });
        }
        (kvs, gdns)
    }

    async fn zero_f32(gpu: &mut owl_cuda::GpuClient, n: usize, shape: Vec<usize>) -> TensorOps {
        let b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F32, vec![n], &vec![0u8; n * 4]).step(),
            gpu,
        )
        .await
        .expect("零块 f32");
        TensorOps::of_block(b.id, Dtype::F32, shape)
    }

    async fn zero_f16(gpu: &mut owl_cuda::GpuClient, n: usize, shape: Vec<usize>) -> TensorOps {
        let b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F16, vec![n], &vec![0u8; n * 2]).step(),
            gpu,
        )
        .await
        .expect("零块 f16");
        TensorOps::of_block(b.id, Dtype::F16, shape)
    }

    /// v2 decode scratch(生产 decode 臂必需;v1 分页核无 kNHD 变体)
    async fn v2_scratch(
        gpu: &mut owl_cuda::GpuClient,
        hq: usize,
        hd: usize,
        nb: usize,
        page: usize,
    ) -> crate::module::AttnV2Scratch {
        let nparts = (nb * page).div_ceil(512);
        crate::module::AttnV2Scratch {
            exp_sums: zero_f32(gpu, hq * nparts, vec![1, hq * nparts]).await,
            max_logits: zero_f32(gpu, hq * nparts, vec![1, hq * nparts]).await,
            tmp_out: zero_f16(gpu, hq * nparts * hd, vec![hq * nparts * hd]).await,
            nparts,
            seqs: 1,
        }
    }

    /// 4B 真权重:**整装 + 文本 prefill 冒烟**(生产 f16 面;验两段键装载 /
    /// 缓冲账 / 单树前向)。门控:`flag:vlm-real=1` + `OWL_TEST_DEVICE`。
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gpu_real_weights_text_prefill_smoke() {
        use crate::layers::rope::Rope;
        if !owl_shared::testkit::resource_flag("flag:vlm-real") {
            owl_shared::testkit::report::skip("flag:vlm-real=1 未开(4B 整装 ~9.3G)");
            return;
        }
        if !crate::testkit::gpu_enabled() {
            crate::testkit::skip_note();
            return;
        }
        let dir = ckpt_dir();
        let t0 = std::time::Instant::now();
        let mut gpu = crate::testkit::gpu_client().await;
        let dbg = owl_shared::testkit::resource_flag("flag:vlm-load-debug");
        if dbg {
            // 分相账本需要 store 上线(测试进程默认未装)
            let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
        }
        let vlm = Vlm::load_ctx(&dir, &mut gpu, crate::module::LoaderCtx {
            dtype: Dtype::F16, shard: 1, slice_id: 0, device_repack: false, verify: false, debug_tap: dbg,
            staging_mb: 128,
            dump: false,
        }).await.expect("Vlm::load(4B)");
        eprintln!("[vlm] 4B 整装 {:.1}s", t0.elapsed().as_secs_f64());
        if dbg {
            let reps = owl_shared::metrics::collect_metrics(
                &owl_shared::metrics::MetricsFilter::new().limit(4096),
            );
            for r in reps.iter().filter(|r| r.tag.starts_with("srv.") || r.tag.starts_with("load.") || r.tag.starts_with("loadv.")) {
                eprintln!("[metrics] {:<28} n={:<8} total={:?}", r.tag, r.count, r.total);
            }
        }

        let c = vlm.cfg();
        let (hq, _hkv, hd) = c.text_spec.full_heads;
        let rp = Rope::new(262_144, hd, 64, 10_000_000.0).expect("rope");
        crate::interpreters::eval_load(
            &rp,
            &mut gpu,
            &rp.tables(),
            &crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1, slice_id: 0, device_repack: false, verify: false, debug_tap: false, staging_mb: 128, dump: false },
        )
        .await
        .expect("rope 表(f16)");

        // 文本 prompt(12 token;prefill 单块)
        let prompt: Vec<u32> = (0..12).map(|i| 1000 + i * 137).collect();
        let l = prompt.len();
        let nb = l.div_ceil(32);
        let (kvs, gdns) = real_bufs(&mut gpu, c, nb, 8).await;
        let ids = TensorOps::from_host(Dtype::F32, vec![l], &f32b(&prompt.iter().map(|&x| x as f32).collect::<Vec<f32>>()));
        // VLM host 表(纯文本:无网格 ⇒ 行号全 -1;m-RoPE 退化为 1D)
        let req = vlm.prepare(&prompt, &[], &[], &[]).expect("prepare(纯文本)");
        assert!(req.merged.is_none(), "纯文本不进塔");
        assert!(req.ins.idx_host.iter().all(|&x| x < 0.0), "纯文本无注入行");
        let slots = TensorOps::from_host(Dtype::F32, vec![l], &f32b(&(0..l).map(|t| t as f32).collect::<Vec<f32>>()));
        let lens = TensorOps::from_host(Dtype::F32, vec![l], &f32b(&(0..l).map(|t| (t + 1) as f32).collect::<Vec<f32>>()));
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0]));
        let base = ForwardCtx::model_prefill(l, &req.ins.pos, &kvs, &rp, &gdns, &slots, &lens, &gdn_slot, None, 0, None);
        let ctx = req.bind(base);
        let logits = vlm.prefill_logits(&ids, &ctx).expect("prefill(须带注入面)");
        let row = crate::testkit::harvest_f16(&mut gpu, &logits).await;
        let last = &row[(l - 1) * c.text_spec.vocab..l * c.text_spec.vocab];
        let (arg, maxv) = last.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a });
        assert!(last.iter().all(|v| v.is_finite()), "logits 必须有限(不许毒值/NaN)");
        eprintln!("[vlm] 文本 prefill 冒烟:argmax = {arg}, max logit = {maxv:.3}");
        assert!(arg < c.text_spec.vocab, "argmax 落词表内");

        // decode 两步(生产面:v2 scratch + qkv 融合插池)
        let scratch = v2_scratch(&mut gpu, hq, hd, nb.max(2), 32).await;
        let mut next = arg as u32;
        let mut gen = Vec::new();
        for step in 0..2usize {
            gen.push(next);
            let t = l + step;
            let ids1 = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[next as f32]));
            let pos1 = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(t as i64 + req.ins.delta) as f32]));
            let kvs_step: Vec<crate::module::KvBuffers> = kvs.iter().map(|kv| crate::module::KvBuffers {
                k_cache: kv.k_cache.clone(),
                v_cache: kv.v_cache.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(t + 1) as f32])),
                block_tables: kv.block_tables.clone(),
            }).collect();
            let gdns_step: Vec<crate::layers::gdn::GdnBuffers> = gdns.iter().map(|g| crate::layers::gdn::GdnBuffers {
                conv_q: g.conv_q.clone(),
                conv_k: g.conv_k.clone(),
                conv_v: g.conv_v.clone(),
                rec: g.rec.clone(),
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
            }).collect();
            let mut ctx = ForwardCtx::model_decode(1, &pos1, &kvs_step, &rp, &gdns_step);
            ctx.attn_v2 = Some(scratch.clone());
            ctx.env.attn.qkv_fuse = true;
            let logits = vlm.text_logits(&ids1, &ctx);
            let row = crate::testkit::harvest_f16(&mut gpu, &logits).await;
            assert!(row.iter().all(|v| v.is_finite()), "decode logits 必须有限");
            next = row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a }).0 as u32;
        }
        eprintln!("[vlm] decode 两步 id = {gen:?} → {next}");
        gpu.close().await.expect("关机");
    }

    /// **M4-d:owl 端到端识图**(塔 + 文本全链)vs HF 金标
    /// (`tools/vlm_e2e_golden.py` 产出)。
    ///
    /// 三层证据:① m-RoPE 位置号 vs **真模型** `get_rope_index`(整数逐元素;
    /// 闭合「shim vs 真身」缺口)② prefill 末位 logits 的 top-1 / top-10 交叠
    /// ③ 贪心 N 步**逐 token** 与 HF 一致 + 打印 owl 自己出的答案文本。
    ///
    /// 门控:`flag:vlm-real=1` + `OWL_TEST_DEVICE`(4B 整装 ~9G,占一卡)。
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gpu_real_image_e2e_matches_hf() {
        use crate::layers::rope::Rope;
        use crate::testkit::{gpu_client, gpu_enabled, skip_note, st_read};
        if !owl_shared::testkit::resource_flag("flag:vlm-real") {
            owl_shared::testkit::report::skip("flag:vlm-real=1 未开(4B 整装 ~9G)");
            return;
        }
        if !gpu_enabled() {
            skip_note();
            return;
        }
        let golden = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata/vlm_e2e.safetensors");
        if !golden.exists() {
            owl_shared::testkit::report::skip(format_args!("{} 不在(先跑 tools/vlm_e2e_golden.py)", golden.display()));
            return;
        }
        let dir = ckpt_dir();
        let mut gpu = gpu_client().await;
        let t0 = std::time::Instant::now();
        let vlm = Vlm::load(&dir, &mut gpu, Dtype::F16).await.expect("Vlm::load(4B)");
        eprintln!("[e2e] 4B 整装 {:.1}s", t0.elapsed().as_secs_f64());
        let c = vlm.cfg();
        let vocab = c.text_spec.vocab;
        let (hq, _hkv, hd) = c.text_spec.full_heads;
        let rp = Rope::new(262_144, hd, 64, 10_000_000.0).expect("rope");
        crate::interpreters::eval_load(
            &rp,
            &mut gpu,
            &rp.tables(),
            &crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1, slice_id: 0, device_repack: false, verify: false, debug_tap: false, staging_mb: 128, dump: false },
        )
        .await
        .expect("rope 表(f16)");
        let tok = crate::formats::qwen35::load_tokenizer(&dir, &c.text_spec.tokenizer).expect("tokenizer");

        let json = std::fs::read_to_string(golden.with_extension("json")).expect("金标 sidecar json");
        let meta: serde_json::Value = serde_json::from_str(&json).expect("sidecar 解析");
        let tags: Vec<String> = meta["images"]
            .as_object()
            .expect("images 段")
            .keys()
            .cloned()
            .collect();
        assert!(!tags.is_empty(), "金标无图");

        for tag in tags {
            let ids_h = st_read(&golden, &format!("in.{tag}.ids"));
            let px = st_read(&golden, &format!("in.{tag}.pixel_values"));
            let grid_h = st_read(&golden, &format!("in.{tag}.grid_thw"));
            let want_pos = st_read(&golden, &format!("hf.{tag}.pos_ids"));
            let want_gen = st_read(&golden, &format!("hf.{tag}.gen_ids"));
            let want_top = st_read(&golden, &format!("hf.{tag}.top_idx"));
            let ids: Vec<u32> = ids_h.iter().map(|&x| x as u32).collect();
            let l = ids.len();
            let grids = vec![(
                grid_h[0] as usize,
                grid_h[1] as usize,
                grid_h[2] as usize,
            )];

            // ① m-RoPE 位置号 vs HF 真模型(整数逐元素)
            let tt = crate::layers::mrope::token_types(&ids, c.image_token, c.video_token);
            let index = c.mrope.rope_index(&ids, &tt, &grids, &[]).expect("rope_index");
            assert_eq!(index.pos.len(), l, "[{tag}] 位置号长");
            let mut pos_ok = 0usize;
            for t in 0..l {
                for g in 0..3 {
                    assert_eq!(
                        index.pos[t][g] as f32,
                        want_pos[g * l + t],
                        "[{tag}] pos[{t}][{g}] vs HF 真模型"
                    );
                    pos_ok += 1;
                }
            }
            eprintln!("[e2e/{tag}] ① 位置号 {pos_ok} 个整数逐元素 == HF 真模型 ✓(delta={})", index.delta);

            // ② prefill(塔 + 文本)
            let req = vlm.prepare(&ids, &grids, &[], &px).expect("prepare(真图)");
            assert!(req.merged.is_some(), "[{tag}] 有图 ⇒ 必须出塔");
            let nb = (l + want_gen.len()).div_ceil(32);
            let (kvs, gdns) = real_bufs(&mut gpu, c, nb, 8).await;
            let ids_t = TensorOps::from_host(Dtype::F32, vec![l], &f32b(&ids.iter().map(|&x| x as f32).collect::<Vec<f32>>()));
            let slots = TensorOps::from_host(Dtype::F32, vec![l], &f32b(&(0..l).map(|t| t as f32).collect::<Vec<f32>>()));
            let lens = TensorOps::from_host(Dtype::F32, vec![l], &f32b(&(0..l).map(|t| (t + 1) as f32).collect::<Vec<f32>>()));
            let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0]));
            let base = ForwardCtx::model_prefill(l, &req.ins.pos, &kvs, &rp, &gdns, &slots, &lens, &gdn_slot, None, 0, None);
            let ctx = req.bind(base);
            let logits = vlm.prefill_logits(&ids_t, &ctx).expect("prefill");
            let row_all = crate::testkit::harvest_f16(&mut gpu, &logits).await;
            let last = &row_all[(l - 1) * vocab..l * vocab];
            assert!(last.iter().all(|v| v.is_finite()), "[{tag}] logits 有限");
            let (arg, _) = last.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a });
            let want_first = want_gen[0] as usize;
            // top-10 交叠(数值面:不苛求逐位,但集合要重合)
            let mut mine: Vec<(usize, f32)> = last.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            mine.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let hit = mine[..10].iter().filter(|(i, _)| want_top.iter().any(|&w| w as usize == *i)).count();
            eprintln!("[e2e/{tag}] ② prefill 末位 top-1 = {arg}(HF {want_first});top-10 交叠 {hit}/10");
            assert_eq!(arg, want_first, "[{tag}] 首 token argmax 必须 == HF");

            // ③ 贪心解码(生产 decode 面)
            let scratch = v2_scratch(&mut gpu, hq, hd, nb.max(2), 32).await;
            let mut gen: Vec<u32> = vec![arg as u32];
            for step in 1..want_gen.len() {
                let t = l + step - 1;
                let prev = gen[step - 1];
                let ids1 = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[prev as f32]));
                let pos1 = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(t as i64 + index.delta) as f32]));
                let kvs_step: Vec<crate::module::KvBuffers> = kvs.iter().map(|kv| crate::module::KvBuffers {
                    k_cache: kv.k_cache.clone(),
                    v_cache: kv.v_cache.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32])),
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[(t + 1) as f32])),
                    block_tables: kv.block_tables.clone(),
                }).collect();
                let gdns_step: Vec<crate::layers::gdn::GdnBuffers> = gdns.iter().map(|g| crate::layers::gdn::GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                }).collect();
                let mut dctx = ForwardCtx::model_decode(1, &pos1, &kvs_step, &rp, &gdns_step);
                dctx.attn_v2 = Some(scratch.clone());
                dctx.env.attn.qkv_fuse = true;
                let row = crate::testkit::harvest_f16(&mut gpu, &vlm.text_logits(&ids1, &dctx)).await;
                assert!(row.iter().all(|v| v.is_finite()), "[{tag}] decode logits 有限");
                let next = row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a }).0 as u32;
                gen.push(next);
            }
            let want_ids: Vec<u32> = want_gen.iter().map(|&x| x as u32).collect();
            let owl_ans = tok.decode(&gen);
            eprintln!("[e2e/{tag}] ③ owl 贪心: {:?}({} tok)", owl_ans, gen.len());
            let same = gen.iter().zip(&want_ids).take_while(|(a, b)| a == b).count();
            eprintln!(
                "[e2e/{tag}] ③ 与 HF 逐 token 前缀一致 {same}/{};HF: {:?}",
                want_ids.len(),
                tok.decode(&want_ids)
            );
            assert_eq!(gen, want_ids, "[{tag}] owl 贪心序列必须与 HF 相同");
        }
        gpu.close().await.expect("关机");
    }

    /// **装载分相微基准**(定位瓶颈用;门控 `flag:vlm-bench=1` + `OWL_TEST_DEVICE`)。
    ///
    /// 量三段各自的上限:① bf16→f16 CPU 转换(F16C)② mmap→host memcpy
    /// ③ pinned→device DMA(异步上传 + 尾 sync)。
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn bench_load_phases() {
        use crate::contract::DeviceClient as _;
        use crate::formats::safetensors::SafeTensorsSource;
        use crate::module::WeightSource;
        if !owl_shared::testkit::resource_flag("flag:vlm-bench") {
            owl_shared::testkit::report::skip("flag:vlm-bench=1 未开");
            return;
        }
        if !crate::testkit::gpu_enabled() {
            crate::testkit::skip_note();
            return;
        }
        let dir = ckpt_dir();
        let mut gpu = crate::testkit::gpu_client().await;
        let src = SafeTensorsSource::open_dir(&dir).expect("源");
        // 样本 = 四个大键(≈ 0.4GB;真权重面)
        let keys = [
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkvz.weight",
            "model.visual.blocks.0.mlp.linear_fc1.weight",
        ];
        // 合约检查:源必须知道首键元素数(2026-10-17 清 unused;原绑定 `n0` 无人读)
        let _ = src.elem_len(keys[0]).expect("源").expect("键");
        let tot_elems: usize = keys
            .iter()
            .map(|k| src.elem_len(k).expect("源").unwrap_or(0))
            .sum();
        let gb = tot_elems as f64 * 2.0 / 1e9;
        eprintln!("[bench] 样本 {gb:.3}GB(四键)");

        // ① CPU 转换 bf16→f16(单键 ×3 轮)
        for k in keys {
            let n = src.elem_len(k).expect("源").expect("键");
            let mut buf = vec![0u8; n * 2];
            let reps = 3;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                src.convert_chunk_into_bytes(k, 0, n, &mut buf, Dtype::F16).expect("源").expect("convert");
            }
            let dt = t0.elapsed().as_secs_f64();
            eprintln!("[bench] ① convert {n:>9} elem: {:.2} GB/s", n as f64 * 2.0 * reps as f64 / 1e9 / dt);
        }

        // ② host 读(mmap 在页缓存 → Vec):raw 路径的拷贝下限
        let mut raw_gb = 0f64;
        let mut raw_s = 0f64;
        for k in keys {
            let n = src.elem_len(k).expect("源").expect("键");
            let t0 = std::time::Instant::now();
            let v = src.take_range(k, 0, n).expect("源").expect("取数"); // 含转换;只作"有数据"对照
            std::hint::black_box(&v);
            raw_gb += n as f64 * 2.0 / 1e9;
            raw_s += t0.elapsed().as_secs_f64();
        }
        eprintln!("[bench] ② take_range(含转换)F32 面: {:.2} GB/s", raw_gb / raw_s);

        // ②' pinned 租约往返(热池):alloc + drop ×200 —— 摘掉 DMA 看纯命令延迟
        {
            let mut leases = Vec::new();
            for _ in 0..4 {
                leases.push(owl_iface::alloc_pinned!(gpu, 64 << 20).await.expect("warm"));
            }
            drop(leases); // 先放 4 个回池(池命中路径)
            let t0 = std::time::Instant::now();
            for _ in 0..200 {
                let l = owl_iface::alloc_pinned!(gpu, 64 << 20).await.expect("pinned");
                drop(l);
            }
            let dt = t0.elapsed().as_secs_f64();
            eprintln!("[bench] ②' alloc_pinned 往返(64MB,热池 ×200): {:.3} ms/次", dt * 1e3 / 200.0);
        }

        // ③ pinned → device DMA:单发大块(1GB)与多路并发(4 面)
        let n_big = 512usize << 20; // 1GB f16 = 512Mi 元素
        let mut big = vec![0u8; n_big * 2];
        for (i, c) in big.chunks_exact_mut(4096).enumerate() {
            c.fill((i & 0xff) as u8);
        }
        let blk = owl_iface::client_alloc!(gpu, Dtype::F16, n_big).await.expect("1GB 设备块");
        // 单发 1GB(分块 128MB)+ 尾 sync
        for reps in [1usize, 3] {
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                let mut off = 0usize;
                while off < n_big {
                    let len = (64usize << 20).min(n_big - off);
                    let mut lease = owl_iface::alloc_pinned!(gpu, len * 2).await.expect("pinned");
                    lease.slice_bytes_mut()[..len * 2].copy_from_slice(&big[off * 2..(off + len) * 2]);
                    gpu.upload_pinned(lease, &blk, off * 2, len * 2).await.expect("upload");
                    off += len;
                }
                gpu.sync().await.expect("sync");
            }
            let dt = t0.elapsed().as_secs_f64();
            eprintln!("[bench] ③ DMA 1GB/64MB 块 ×{reps}: {:.2} GB/s({dt:.2}s)", 1.0 * reps as f64 / dt);
        }
        // 4 面并发(装载工件同款:每面各自 256MB)
        let faces: Vec<_> = gpu.loader_faces(4).expect("并发面");
        let per = 128usize << 20; // 每面 256MB f16 = 128Mi 元素
        let t0 = std::time::Instant::now();
        let mut futs = Vec::new();
        for (fi, mut f) in faces.into_iter().enumerate() {
            let blk = owl_iface::client_alloc!(f, Dtype::F16, per).await.expect("分块");
            let src = big[..per * 2].to_vec();
            futs.push(async move {
                let mut off = 0usize;
                while off < per {
                    let len = (16usize << 20).min(per - off);
                    let mut lease = owl_iface::alloc_pinned!(f, len * 2).await.expect("pinned");
                    lease.slice_bytes_mut()[..len * 2].copy_from_slice(&src[off * 2..(off + len) * 2]);
                    f.upload_pinned(lease, &blk, off * 2, len * 2).await.expect("upload");
                    off += len;
                }
                f.sync().await.expect("sync");
                let _ = fi;
            });
        }
        futures_util::future::join_all(futs).await;
        let dt = t0.elapsed().as_secs_f64();
        eprintln!("[bench] ③ DMA 4 面并发(每面 256MB,16MB 块): {:.2} GB/s({dt:.2}s)", 0.25 * 4.0 / dt);
        gpu.close().await.expect("关机");
    }
}
