//! Qwen3.5 规格集(批8 拆分,2026-09-26):只住 **Qwen3.5 特有**的
//! 参数事实,零机制 —— 通用主干见 [`crate::model`]。
//!
//! 拆分律:specs/<model>.rs = 该模型的维度预设 / 层型表约定 / 检查点
//! 特例。Qwen3.5 特有项:
//! - hybrid 3:1 层型(`linear_attention` ×3 + `full_attention` ×1 周期;
//!   config.json layer_types 实测,24 层 = 18 GDN + 6 full);
//! - 0.8B 维度档(hidden 1024 / GDN 16×128 / full 8q2kv×256 /
//!   vocab 248320 tied / eps 1e-6)。
//!
//! 未来新档(4B/…)加构造函数;新模型(qwen3 dense 等)另起
//! `specs/<model>.rs`,勿混居。
//!
//! **M-e 装载面**:Qwen35Convention(键名约定)+ [`load_0_8b`] 快照
//! 目录装载入口 —— 模型特有胶水(子前缀/裸键/基座)全部封在本文件,
//! 通用层(model.rs/module.rs/loader.rs)零 Qwen3.5 知识。

//! 检查点实测(488 张量):BF16 为主 + `A_log`/`dt_bias` 两个 F32 裸键;
//! conv1d 为 3D `[6144,1,4]`(扁平直读)。visual/mtp 153 键不在最小路径。

use crate::contract::{DeviceClient, ModelError};
use crate::formats::safetensors::SafeTensorsSource;
use crate::model::{Model, ModelSpec};
use crate::module::{KeyConvention, Loadable, LoaderCtx, LoaderOps};
use std::path::Path;

// ============================================================================
// 检查点键名约定(488 张量实测,2026-09-26)
// ============================================================================

/// Qwen3.5 检查点键名约定 —— 与 Llama 系默认的三处差异:
/// 1. GDN 层 mixer 键带 `linear_attn.` 子前缀,Full 层带 `self_attn.`;
/// 2. mlp 键带 `mlp.` 子前缀;
/// 3. `A_log` / `dt_bias` 是**裸键**(无 `.weight` 后缀;全仓仅有的
///    两个 F32 键)。
///
/// 局部键名在 GDN/Full 两侧互斥,按名分派无歧义(无需知道层型)。
pub struct Qwen35Convention {
    base: String,
}

impl Qwen35Convention {
    /// base = 检查点基座(多模态仓 `model.language_model`,纯文本仓 `model`)
    pub fn new(base: impl Into<String>) -> Self {
        Self { base: base.into() }
    }
}

impl KeyConvention for Qwen35Convention {
    fn embed_key(&self, local: &str) -> String {
        format!("{}.embed_tokens.{local}", self.base)
    }
    fn norm_key(&self, local: &str) -> String {
        format!("{}.{}{}.weight", self.base, "", local)
    }
    fn layer_key(&self, i: usize, local: &str) -> String {
        // E3 量化后缀:local 可带 `.qweight/.scales/.zeros/.ws`(marlin 套件
        // —— AWQ 臂增 zeros)—— 剥离后按基名定子前缀,量化后缀替代 `.weight`
        let (local, qsuffix) = match local.rsplit_once('.') {
            Some((b, s))
                if matches!(
                    s,
                    "qweight" | "scales" | "zeros" | "ws" | "marlin_ws" | "marlin_ctmp"
                        | "packed_raw"
                ) =>
            {
                (b, Some(format!(".{s}")))
            }
            _ => (local, None),
        };
        let (sub, suffix) = match local {
            // 层共用件:双 norm 无子前缀
            "input_layernorm" | "post_attention_layernorm" => ("", ".weight"),
            // mlp
            "gate_proj" | "up_proj" | "down_proj" => ("mlp.", ".weight"),
            // GDN mixer(norm = 层内门控归一化,非 final norm;刀2:qkvz
            // 虚拟合并键 → linear_attn.in_proj_qkvz,由装载源合成)
            "in_proj_qkvz" | "in_proj_b" | "in_proj_a" | "out_proj" | "conv1d"
            | "norm" => ("linear_attn.", ".weight"),
            // GDN 裸键(全仓仅有的无 .weight 后缀特例)
            "A_log" | "dt_bias" => ("linear_attn.", ""),
            // Full mixer
            "q_proj" | "k_proj" | "v_proj" | "o_proj" | "q_norm" | "k_norm" => {
                ("self_attn.", ".weight")
            }
            other => unreachable!("Qwen35Convention: 未知层内局部键 {other}"),
        };
        // 量化后缀替代 `.weight`(裸键 A_log/dt_bias 不会被量化,互斥)
        let tail = qsuffix.unwrap_or_else(|| suffix.to_string());
        format!("{}.layers.{i}.{}{local}{tail}", self.base, sub)
    }
}

// ============================================================================
// 装载入口(M-e)
// ============================================================================

/// 装载 Qwen3.5-0.8B(快照目录:`config.json` + `*.safetensors`)。
/// 目录内全部 safetensors 合并取源(BF16 → f32,visual/mtp 键无害
/// 常驻源内,只有主干键会被查到);装载走整模 Loadable 单清单。
/// 装载 Qwen3.5-0.8B(快照目录:`config.json` + `*.safetensors`)。
/// 目录内全部 safetensors 合并取源(BF16 → f32,visual/mtp 键无害
/// 常驻源内,只有主干键会被查到);装载走整模 Loadable 单清单。
pub async fn load_0_8b<D: DeviceClient + 'static>(
    dir: &Path,
    face: &mut D,
) -> Result<Model, ModelError> {
    let model = Model::new(
        &qwen3_5_0_8b(),
        Qwen35Convention::new("model.language_model"),
        crate::module::QuantPlan::F16,
        crate::module::QuantPlan::F16,
    );
    let src = SafeTensorsSource::open_dir(dir)?;
    // F16 直转装载(F5;权重 bf16 检查点 → f16 字节,不再 f32 设备中转)
    let ctx = crate::module::LoaderCtx { dtype: qwen3_5_0_8b().dtype, shard: 1, device_repack: false, verify: false, debug_tap: false };
    crate::interpreters::eval_load(&model, face, &src, &ctx).await?;
    Ok(model)
}

/// 装载 Qwen3.5-0.8B **W4A16**(E3;llm-compressor 产物目录,
/// compressed-tensors pack-quantized)—— 装载期重排到 marlin 布局
/// (w4a16.rs 源;小线性自动反量化 f16)。
pub async fn load_0_8b_w4a16<D: DeviceClient + 'static>(
    dir: &Path,
    face: &mut D,
) -> Result<Model, ModelError> {
    let spec = qwen3_5_0_8b();
    // 量化计划构造期注入(禁 enable_* 可变后置):W4A16 臂在此定形,
    // 尺寸门控(marlin_eligible)在 Linear 构造期完成,小线性自动落 f16。
    let model = Model::new(
        &spec,
        Qwen35Convention::new("model.language_model"),
        crate::module::QuantPlan::W4A16,
        crate::module::QuantPlan::F16,
    );
    let src = crate::formats::w4a16::W4A16Source::open_dir(dir)?;
    let ctx = crate::module::LoaderCtx { dtype: crate::contract::Dtype::F16, shard: 1, device_repack: false, verify: false, debug_tap: false };
    crate::interpreters::eval_load(&model, face, &src, &ctx).await?;
    Ok(model)
}

/// 装载 **Qwen3.8-27B AWQ-INT4**(2026-10-01;cyankiwi 检查点,
/// compressed-tensors pack-quantized **g32-asym + zp**)—— 装载源
/// formats/awq.rs:eligible 线性走 marlin kU4(has_zp)内核
/// (GEMM_W4A16_AWQ),非门控线性/checkpoint ignore 项反量化或直读 f16。
///
/// 维度事实以**检查点实测**为准(config 的 24 头与张量形状不符):
/// - full attention:q 48 头 ×256(q_proj [12288, 5120])/ kv 4 头 ×256
///   (k/v_proj [1024, 5120]);
/// - GDN:qk 16 头 ×128(in_proj_qkv [10240, 5120] = 2048+2048+6144)/
///   v 48 头 ×128(in_proj_z [6144, 5120]);
/// - 64 层 3:1 hybrid(layers i%4==3 为 full);vocab 248320 不 tied。
/// MTP 头装载(E5-M0;cyankiwi 检查点自带 15 个 mtp.* 键全 BF16 ——
/// 结构 = vLLM Qwen3NextMultiTokenPredictor 同构,见 layers/mtp.rs)。
/// 返回 (组件, 装载清单);键集断言(15 键)由测试做。
pub async fn load_27b_mtp<D: DeviceClient + 'static>(
    dir: &Path,
    face: &mut D,
) -> Result<(crate::layers::mtp::MtpPredictor, crate::module::LoadManifest), ModelError> {
    let s = qwen3_8_27b();
    let mtp = crate::layers::mtp::MtpPredictor::new(
        s.hidden,
        s.inter,
        s.full_heads.0,
        s.full_heads.1,
        s.full_heads.2,
        s.eps,
    );
    let src = crate::formats::awq::AwqSource::open_dir(dir)?;
    let device_repack =
        crate::module::RepackPath::resolve(None, face.device_kernels()).is_device();
    let ctx = crate::module::LoaderCtx {
        dtype: crate::contract::Dtype::F16,
        shard: 1,
        device_repack,
        verify: false,
        debug_tap: false,
    };
    let manifest = crate::interpreters::eval_load(&mtp, face, &src, &ctx).await?;
    Ok((mtp, manifest))
}

/// 装载 **Qwen3.8-27B DFlash2 草稿**(2026-10-07;E5-DF1,
/// z-lab/Qwen3.8-27B-DFlash2,BF16 81 张量 1.92B —— 结构见
/// layers/dflash2.rs 头注;键约定 = 检查点原生命名直映)。
/// 全 F16 直转装载(BF16 源 → f16;fc 走 Transposed 臂,装载期 host
/// 转置 —— W^T 行连续 = 列块,消费面逐 tap slice_view;设备重排优化挂账)。
pub async fn load_27b_dflash2<D: DeviceClient + 'static>(
    dir: &Path,
    face: &mut D,
    kv_fp8: bool,
) -> Result<(crate::layers::dflash2::DFlash2Draft, crate::module::LoadManifest), ModelError> {
    // 源族自动探测(fc.weight_scale 在 = compressed-tensors W4A16 g128;
    // 缺 = BF16 原生)。W4A16 = 1.28GB(BF16 3.85GB)—— 24G 恒等门前置。
    // 裸索引探测(SafeTensorsSource::open_dir 的 F32/BF16 校验会拒收
    // W4A16 文件的 I32/F16 条目;open_raw_index 无过滤零拷贝)
    let (_, raw) = crate::formats::mmap::open_raw_index(dir)?;
    let is_q = raw.contains_key("fc.weight_packed");
    drop(raw);
    let ctx = crate::module::LoaderCtx {
        // BF16(E5-DF3 同日十四;sglang 对齐):草稿路径 BF16 全程 ——
        // 检查点 passthrough 权重(norms/conv 投影/codebooks)原生 BF16
        // 直载零转换;marlin scales 钉 F16(dtype_override);激活/池全 bf16
        dtype: crate::contract::Dtype::BF16,
        shard: 1,
        device_repack: false,
        verify: false,
        debug_tap: false,
    };
    if is_q {
        let draft = crate::layers::dflash2::DFlash2Draft::new_with_plan_dt(
            5120,
            17408,
            32,
            8,
            128,
            5,
            1e-6,
            248320,
            crate::module::QuantPlan::W4A16,
            crate::contract::Dtype::BF16,
        kv_fp8,
        );
        let src = crate::formats::w4a16::W4A16Source::open_dir(dir)?;
        let manifest = crate::interpreters::eval_load(&draft, face, &src, &ctx).await?;
        // E5 内存账:草稿装载分类账(按 dtype×layout 聚合元素数与字节)
        {
            let mut agg: std::collections::BTreeMap<String, (u64, u64)> = Default::default();
            for e in manifest.entries() {
                let bytes: u64 = e.shape.iter().map(|&d| d as u64).product::<u64>()
                    * match e.dtype {
                        crate::contract::Dtype::F16 | crate::contract::Dtype::BF16 => 2,
                        crate::contract::Dtype::F32 | crate::contract::Dtype::U32 => 4,
                    };
                let key = format!("{:?}/{:?}", e.dtype, e.layout);
                let e2 = agg.entry(key).or_insert((0, 0));
                e2.0 += 1;
                e2.1 += bytes;
            }
            for (k, (n, b)) in agg {
                eprintln!("[draft-mem] {k}: {n} 项 {b} ({:.2} GB)", b as f64 / 1e9);
            }
        }
        return Ok((draft, manifest));
    }
    // ⚠️ BF16 原生家族(z-lab)分支警示(2026-10-10 立案):本分支草稿
    // 在 27B+draft 服务形态实测全拒(AL=1.000 恒 m0,三域/双 dtype 主池
    // 同象)→ spec 退化为裸 decode−税(数学域 36 vs W4A16 家族 157 t/s)。
    // 生产一律用 W4A16(syvai);本分支死活另案(golden 参考/对照臂用途)。
    eprintln!(
        "[boot] ⚠️ DFlash2 草稿 = BF16 原生家族:实测草稿全拒(AL=1.0),\n         生产请用 W4A16 家族(syvai/...-DFlash2-W4A16);分支死活另案"
    );
    let draft = crate::layers::dflash2::DFlash2Draft::new_with_plan_dt(
        5120,
        17408,
        32,
        8,
        128,
        5,
        1e-6,
        248320,
        crate::module::QuantPlan::F16,
        crate::contract::Dtype::BF16,
        kv_fp8,
    );
    eprintln!("[boot] dflash draft-pool kv_fp8 = {kv_fp8}");
    let src = SafeTensorsSource::open_dir(dir)?;
    let manifest = crate::interpreters::eval_load(&draft, face, &src, &ctx).await?;
    Ok((draft, manifest))
}

/// fc 形态标记(Loadable 分派用;借用拆分)
pub(crate) enum FcShape {
    Q,
    F16T,
}

/// DFlash2 草稿键映射(检查点原生命名;conv 的 base_kernel 无 .weight
/// 后缀、codebook 两键亦然 —— 三特例在闭包内分流)。
impl crate::layers::dflash2::DFlash2Draft {
    pub(crate) fn fc_shape(&self) -> FcShape {
        if self.is_quant() {
            FcShape::Q
        } else {
            FcShape::F16T
        }
    }
}

impl Loadable for crate::layers::dflash2::DFlash2Draft {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        let w = |k: &str| format!("{k}.weight");
        // 键尾规则(双源通用):量化槽(qweight/scales/ws/marlin_ctmp)后缀
        // 已内含 → 前缀直拼;其余(norm 裸键 / f16 回落 Linear .weight)补
        // .weight(两源检查点的 norm 键均带 .weight)。
        let lin_tail = |k: &str| {
            if k.ends_with(".qweight")
                || k.ends_with(".scales")
                || k.ends_with(".ws")
                || k.ends_with(".marlin_ctmp")
                || k.ends_with(".marlin_ws")
            {
                k.to_string()
            } else {
                format!("{k}.weight")
            }
        };
        let mut ops = match self.fc_shape() {
            FcShape::Q => {
                let mut o = self.fc_q()[0].layout(ctx);
                for l in &self.fc_q()[1..] {
                    o = o.chain(l.layout(ctx));
                }
                o
            }
            FcShape::F16T => self.fc_t().layout(ctx).map_keys(|k| format!("{k}.weight")),
        };
        ops = ops.chain(self.hidden_norm().layout(ctx).map_keys(w));
        for (i, layer) in self.layers().iter().enumerate() {
            let conv_tail = |k: &str| {
                if k == "base_kernel" { k.to_string() } else { format!("{k}.weight") }
            };
            ops = ops
                .chain(
                    layer
                        .input_ln()
                        .layout(ctx)
                        .map_keys(|k| format!("layers.{i}.{k}.weight")),
                )
                .chain(
                    layer
                        .attn_conv()
                        .layout(ctx)
                        .map_keys(move |k| format!("layers.{i}.attention_conv.{}", conv_tail(k))),
                )
                .chain(
                    layer
                        .attn()
                        .layout(ctx)
                        .map_keys(|k| format!("layers.{i}.self_attn.{}", lin_tail(k))),
                )
                .chain(
                    layer
                        .post_ln()
                        .layout(ctx)
                        .map_keys(|k| format!("layers.{i}.{k}.weight")),
                )
                .chain({
                    let conv_tail = |k: &str| {
                        if k == "base_kernel" { k.to_string() } else { format!("{k}.weight") }
                    };
                    layer
                        .mlp_conv()
                        .layout(ctx)
                        .map_keys(move |k| format!("layers.{i}.mlp_conv.{}", conv_tail(k)))
                })
                .chain(
                    layer
                        .mlp()
                        .layout(ctx)
                        .map_keys(|k| format!("layers.{i}.mlp.{}", lin_tail(k))),
                );
        }
        ops.chain(self.norm_head().layout(ctx).map_keys(w)).chain(
            self.selector()
                .layout(ctx)
                .map_keys(|k| match k {
                    "hidden_projection" => "candidate_selector.hidden_projection.weight".into(),
                    other => format!("candidate_selector.{other}"),
                }),
        )
    }
}

/// MTP 头键映射(layout 期改写;与 Qwen35Convention::layer_key 同式,
/// base = "mtp",层序号恒 0 —— 键约定住 specs 家规的落点)。
impl Loadable for crate::layers::mtp::MtpPredictor {
    fn layout(&self, ctx: &LoaderCtx) -> LoaderOps {
        let keys = Qwen35Convention::new("mtp");
        self.fc_e()
            .layout(ctx)
            .map_keys(|k| format!("mtp.{k}.weight"))
            .chain(self.fc_h().layout(ctx).map_keys(|k| format!("mtp.{k}.weight")))
            .chain(self.layer().layout(ctx).map_keys(|k| keys.layer_key(0, &k)))
            .chain(
                self.pre_fc_norm_hidden()
                    .layout(ctx)
                    .map_keys(|k| format!("mtp.{k}.weight")),
            )
            .chain(
                self.pre_fc_norm_embedding()
                    .layout(ctx)
                    .map_keys(|k| format!("mtp.{k}.weight")),
            )
            .chain(self.norm_head().layout(ctx).map_keys(|k| format!("mtp.{k}.weight")))
    }
}

pub async fn load_27b_awq<D: DeviceClient + 'static>(
    dir: &Path,
    face: &mut D,
    head_plan: crate::module::QuantPlan,
) -> Result<Model, ModelError> {
    let model = Model::new(
        &qwen3_8_27b(),
        Qwen35Convention::new("model.language_model"),
        crate::module::QuantPlan::W4A16Awq,
        // C1 lm_head 头部量化计划(2026-10-11):量化方案由配置枚举映射
        // (HeadQuant → QuantPlan,engine loader 唯一点);W4A16Awq 下
        // bf16 裸 lm_head 由装载源 F16 兕底臂现场 RTN int4(g32 sym,
        // zp≡8;marlin kU4 通路)—— 零拼装,检查点不动。质量门:竖式/
        // gate greedy A/B 已过。
        head_plan,
    );
    let src = crate::formats::awq::AwqSource::open_dir(dir)?;
    // repack 路径裁决(唯一收口:module::RepackPath::resolve ——
    // 默认 GPU;host face 能力否决自动回退;OWL_LOAD_CPU_REPACK env 强制)
    let device_repack =
        crate::module::RepackPath::resolve(None, face.device_kernels()).is_device();
    src.set_device_repack(device_repack);
    let ctx = crate::module::LoaderCtx {
        dtype: crate::contract::Dtype::F16,
        shard: 1,
        device_repack,
        verify: false,
        debug_tap: false,
    };
    crate::interpreters::eval_load(&model, face, &src, &ctx).await?;
    Ok(model)
}

/// Qwen3.5 tokenizer 装配:机制在 tokenizer.rs,事实在 spec 声明
/// (ModelSpec.tokenizer)—— 本函数只是两端的接线(jinja 全引擎挂账
/// serving 层)
pub fn load_tokenizer(
    dir: &Path,
    spec: &crate::tokenizer::TokenizerSpec,
) -> Result<crate::tokenizer::Tokenizer, ModelError> {
    // 2026-10-12 两参化(原恒用 0.8B 档):chat 前后缀是模型档事实 ——
    // 27B froggeric(thinking)与 0.8B 的注入面完全不同(AL 案根因之二)
    crate::tokenizer::Tokenizer::from_spec(dir, spec)
}

/// 3:1 周期(G,G,G,F)铺满 n 层(Qwen3.5 hybrid 惯例;
/// 24 层 → 18 GDN + 6 full,与 0.8B config.json 实测一致)
pub fn hybrid_3to1(n: usize) -> Vec<bool> {
    (0..n).map(|i| i % 4 == 3).collect()
}

/// Qwen3.5-0.8B 真实维度(M-e 灌真权重用;config.json 实测)
pub fn qwen3_5_0_8b() -> ModelSpec {
    ModelSpec {
        vocab: 248320,
        hidden: 1024,
        inter: 3584,
        dtype: crate::contract::Dtype::F16,
        full_heads: (8, 2, 256),
        gdn_heads: (16, 128, 16, 128),
        eps: 1e-6,
        layer_types: hybrid_3to1(24),
        tokenizer: crate::tokenizer::TokenizerSpec {
            eos_tokens: vec!["<|im_end|>", "<|endoftext|>"],
            chat: crate::tokenizer::ChatFormat {
                prefix: "<|im_start|>user\n".into(),
                // 默认(非思考)模式:模板预填空 think 块(tokenizer_config
                // chat_template add_generation_prompt 分支实证);缺它模型需
                // 自己生成空 think 块,greedy 会紧跟 eos 答空(2026-09-26 实测)。
                suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into(),
            },
        },
        tied: true,
    }
}

/// Qwen3.8-27B 真实维度(2026-10-01;cyankiwi AWQ-INT4 检查点张量实测):
/// - full attention = **Qwen3Next 门控注意力**(vLLM qwen3_next.py 实证):
///   q_proj [12288, 5120] = q(24×256) ⊕ output_gate(24×256) per-head
///   [value|gate] 融合 —— owl Attention 层原生存(q_raw 2×Hq·HD 切分);
///   o_proj [5120, 6144] = 24×256 ✓;kv 4×256(k/v_proj [1024, 5120]);
/// - GDN:qk 16×128(in_proj_qkv [10240, 5120] = 2048+2048+6144)/
///   v 48×128(in_proj_z [6144, 5120]);
/// - 64 层 3:1 hybrid(i%4==3 为 full);vocab 248320 不 tied;
/// - rope:theta 1e7 + partial 0.25(rotary dim 64)与 0.8B 同。
pub fn qwen3_8_27b() -> ModelSpec {
    ModelSpec {
        vocab: 248320,
        hidden: 5120,
        inter: 17408,
        dtype: crate::contract::Dtype::F16,
        full_heads: (24, 4, 256),
        gdn_heads: (16, 128, 48, 128),
        eps: 1e-6,
        layer_types: hybrid_3to1(64),
        tokenizer: crate::tokenizer::TokenizerSpec {
            eos_tokens: vec!["<|im_end|>", "<|endoftext|>"],
            chat: crate::tokenizer::ChatFormat {
                // 27B(froggeric v22.5,thinking 模型)正确注入形态:
                // jinja add_generation_prompt 渲染实证(2026-10-12 AL 案)——
                // ① system 段 = reasoning effort 注入(模板默认 medium);
                // ② suffix 开 <think> 不预填空块(空块 = 非思考调法,模型
                //    在错误上下文生成 → 输出退化 + DFlash AL 1.45 vs vLLM 2.61)
                prefix: "<|im_start|>system\nReasoning effort is set to medium. Think through the task at a moderate depth: cover the key steps and verify the result, but keep the reasoning concise.<|im_end|>\n<|im_start|>user\n".into(),
                suffix: "<|im_end|>\n<|im_start|>assistant\n<think>\n".into(),
            },
        },
        tied: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TensorOps;
    use crate::module::{ForwardCtx, Module};
    use crate::tensor::Dtype;
    use std::collections::HashMap;
    use crate::testkit::{f32b, manifest_dir};

    #[test]
    fn hybrid_3to1_matches_config() {
        let lt = hybrid_3to1(24);
        assert_eq!(lt.len(), 24);
        assert_eq!(lt.iter().filter(|&&f| f).count(), 6, "6 full 层");
        assert_eq!(lt.iter().filter(|&&f| !f).count(), 18, "18 GDN 层");
        assert!(lt[3] && lt[7] && lt[23], "full 层落在 i%4==3");
        let s = qwen3_5_0_8b();
        assert_eq!(s.layer_types, lt);
        assert_eq!(s.vocab, 248320);
    }

    /// 键名约定 vs 检查点实测(488 张量逐键核对过的样例)
    #[test]
    fn convention_matches_checkpoint() {
        let c = Qwen35Convention::new("model.language_model");
        // GDN 层(linear_attn 子前缀)
        assert_eq!(
            c.layer_key(0, "in_proj_qkvz"),
            "model.language_model.layers.0.linear_attn.in_proj_qkvz.weight"
        );
        assert_eq!(
            c.layer_key(0, "conv1d"),
            "model.language_model.layers.0.linear_attn.conv1d.weight"
        );
        assert_eq!(
            c.layer_key(0, "norm"),
            "model.language_model.layers.0.linear_attn.norm.weight"
        );
        // 裸键特例(无 .weight)
        assert_eq!(
            c.layer_key(0, "A_log"),
            "model.language_model.layers.0.linear_attn.A_log"
        );
        assert_eq!(
            c.layer_key(0, "dt_bias"),
            "model.language_model.layers.0.linear_attn.dt_bias"
        );
        // Full 层(self_attn 子前缀)
        assert_eq!(
            c.layer_key(3, "q_proj"),
            "model.language_model.layers.3.self_attn.q_proj.weight"
        );
        // mlp / 层共用件
        assert_eq!(
            c.layer_key(0, "gate_proj"),
            "model.language_model.layers.0.mlp.gate_proj.weight"
        );
        assert_eq!(
            c.layer_key(5, "input_layernorm"),
            "model.language_model.layers.5.input_layernorm.weight"
        );
        // embed / final norm
        assert_eq!(c.embed_key("weight"), "model.language_model.embed_tokens.weight");
        assert_eq!(c.norm_key("norm"), "model.language_model.norm.weight");
    }


    // ==================================================================
    // 真权重装载 + 两步 decode 冒烟(门控 OWL_TEST_DEVICE;数值基准
    // 对比 transformers/vLLM 挂账)
    // ==================================================================

    const VOCAB: usize = 248_320;
    const HKV: usize = 2;
    const HD: usize = 256;
    const SLOTS: usize = 4;

    async fn zero_block(client: &mut owl_cuda::GpuClient, n: usize, shape: Vec<usize>) -> TensorOps {
        let b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F32, vec![n], &f32b(&vec![0.0; n])).step(),
            client,
        )
        .await
        .expect("zero block");
        TensorOps::of_block(b.id, Dtype::F32, shape)
    }

    /// f16 清零块(F5:KV cache;2B/元素)
    async fn zero_block_f16(client: &mut owl_cuda::GpuClient, n: usize, shape: Vec<usize>) -> TensorOps {
        let b = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F16, vec![n], &vec![0u8; n * 2]).step(),
            client,
        )
        .await
        .expect("zero block f16");
        TensorOps::of_block(b.id, Dtype::F16, shape)
    }

    /// CPU face 真权重装载(非门控;验源/约定/内存,不验 CUDA)
    #[tokio::test]
    async fn cpu_real_weights_load() {
        let dir = manifest_dir().join("assets/Qwen3.5-0.8B");
        let mut face = owl_cpu::CpuFace::new();
        let model = load_0_8b(&dir, &mut face).await.expect("load_0_8b(CPU)");
        assert_eq!(model.layers.len(), 24);
        assert!(model.embed.is_loaded() && model.norm.is_loaded());
    }


    /// 诊断(tap 单遍):真模型逐层 hidden rms 曲线(定位 step1 塌零层)。
    /// 整模单树单遍归约,StatsTap 只看层根 tag —— 零重放,状态每步只推进
    /// 一次(旧逐层 harvest 重放污染读数,已废;interpreter-tap.md §一)。
    #[tokio::test]
    async fn gpu_model_step_diag() -> Result<(), crate::contract::ModelError> {
        use crate::layers::gdn::GdnBuffers;
        use crate::layers::rope::Rope;
        use crate::module::KvBuffers;

        if !crate::testkit::gpu_enabled() {
            crate::testkit::skip_note();
            return Ok(());
        }
        let dir = manifest_dir().join("assets/Qwen3.5-0.8B");
        let mut gpu = crate::testkit::gpu_client().await;
        let model = load_0_8b(&dir, &mut gpu).await?;
        eprintln!("[diag] loaded");
        let rp = Rope::new(262_144, 256, 64, 10_000_000.0)?;
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &Default::default()).await?;

        let (kvs_n, gdns_n) = (6usize, 18usize);
        let mut mk_kvs = Vec::new();
        let mut mk_gdns = Vec::new();
        let bt_data: Vec<u8> = [0.0f32].iter().flat_map(|f| f.to_le_bytes()).collect();
        let bt0 = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F32, vec![1, 1], &bt_data).step(), &mut gpu)
            .await.expect("恒等块表");
        for _ in 0..kvs_n {
            mk_kvs.push(KvBuffers {
                k_cache: zero_block_f16(&mut gpu, 2 * 256 * 32, vec![1, 2, 256 / 8, 32, 8]).await,
                v_cache: zero_block_f16(&mut gpu, 2 * 256 * 32, vec![1, 2, 256, 32]).await,
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0])),
                block_tables: TensorOps::of_block(bt0.id, Dtype::F32, vec![1, 1]),
            });
        }
        for _ in 0..gdns_n {
            mk_gdns.push(GdnBuffers {
                conv_q: zero_block(&mut gpu, SLOTS * 2048 * 3, vec![SLOTS, 2048, 3]).await,
                conv_k: zero_block(&mut gpu, SLOTS * 2048 * 3, vec![SLOTS, 2048, 3]).await,
                conv_v: zero_block(&mut gpu, SLOTS * 2048 * 3, vec![SLOTS, 2048, 3]).await,
                rec: zero_block(&mut gpu, SLOTS * 16 * 128 * 128, vec![SLOTS, 16, 128, 128]).await,
                slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
            });
        }
        let kvs = mk_kvs;
        let gdns = mk_gdns;

        // (装载即校验已有专测 gpu_vram_manifest_checksum;此处二次装载
        //  在 f16 时代 = F32 ctx 污染权重 + 双倍耗时,F5-2 删)
        let steps = [(985.0f32, 0.0f32, 0.0f32, 1.0f32), (4123.0, 1.0, 1.0, 2.0f32)];
        // 引用重读探针(竞速判决):记录层根块引用与 dtype,步终 sync 后
        // 重读 —— 窗口内读零 + 步终读非零 = D2H 流与 COMPUTE 流无同步的
        // 竞速读。字节口径随 dtype 推导,禁硬编码(×4/×2 均为雷)
        struct RefProbe {
            refs: Vec<(String, crate::contract::Bytes, usize, Dtype)>,
        }
        impl crate::interpreters::Tap for RefProbe {
            fn on_node(
                &mut self,
                ev: &crate::interpreters::NodeEvent<'_>,
            ) -> crate::interpreters::Want {
                if let Some(t) = ev.tag {
                    let elems: usize = ev.shape.iter().product();
                    self.refs.push((
                        t.to_string(),
                        crate::contract::Bytes { id: ev.out.block_id, len: elems },
                        elems,
                        ev.dtype,
                    ));
                }
                crate::interpreters::Want::Quiet
            }
        }
        let mut probe = RefProbe { refs: Vec::new() };
        for (si, &(id, pos, slot, kv_len)) in steps.iter().enumerate() {
            eprintln!("[diag] === step{si} (pos {pos} slot {slot} kv_len {kv_len}) ===");
            let ids = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[id]));
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[pos]));
            let kvs_step: Vec<KvBuffers> = kvs
                .iter()
                .map(|kv| KvBuffers {
                    k_cache: kv.k_cache.clone(),
                    v_cache: kv.v_cache.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot])),
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[kv_len])),
                    block_tables: kv.block_tables.clone(),
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdns
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot])),
                })
                .collect();
            let base_ctx = ForwardCtx::model_decode(1, &pos_t, &kvs_step, &rp, &gdns_step);
            // ── 律 25(候选):带状态观测禁重放 —— 整模单树**单遍**归约,
            // tap 层根曲线代替旧逐层 harvest(旧法每层整链重放,GDN 状态
            // 被观测行为污染,读数不可信 —— interpreter-tap.md §一)。
            let tree = model.forward(&ids, &base_ctx); // embed→…→norm→logits,层根已打标
            let root = tree.step();
            // 步终判读:logits top-1(塌零 = top 落在 0 向量上)
            // 读回口径随声明链根 dtype 推导(f16 基线;禁硬编码)
            let logits_dt = root.dtype();
            let logits = {
                let mut curve = crate::interpreters::StatsTap::curve(si);
                let mut taps = crate::interpreters::observe::TapChain(&mut curve, &mut probe);
                crate::interpreters::eval_ops_tap(root, &mut gpu, &mut taps).await?
            };
            let mut buf = vec![0u8; model.vocab_size() * logits_dt.size_bytes()];
            gpu.dtoh(&logits, &mut buf).await?;
            let lh = crate::interpreters::eval::decode_host(logits_dt, &buf);
            let (ti, tv) = lh.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| {
                    if v > a.1 { (i, v) } else { a }
                });
            eprintln!("[diag]   step{si} top@{ti} logit={tv:.4}");
            // ── 竞速判决:三流 sync 后重读 L0 层根与 final_norm(窗口内读的同一块)──
            // 读回长度与解码均随块 dtype(RefProbe 记录),无硬编码
            gpu.sync().await?;
            for (tag, b, elems, dt) in &probe.refs {
                if !matches!(tag.as_str(), "L0.gdn" | "final_norm") {
                    continue;
                }
                let mut buf = vec![0u8; elems * dt.size_bytes()];
                gpu.dtoh(b, &mut buf).await?;
                let h = crate::interpreters::eval::decode_host(*dt, &buf);
                let rms = (h.iter().map(|v| v * v).sum::<f32>() / h.len() as f32).sqrt();
                eprintln!("[diag]   [sync后重读] {tag} rms={rms:.6} zeros={}", h.iter().filter(|v| **v == 0.0).count());
            }
            probe.refs.clear();
        }
        gpu.close().await?;
        Ok(())
    }


    /// ★ 核心校验:VRAM 块数据 vs 独立锚 逐位比较(门控 OWL_TEST_DEVICE)。
    /// 独立锚 = 自建文件读取(不经 SafeTensorsSource)+ 朴素 BF16 解码 +
    /// 朴素转置(不经 transpose_into_vec)—— 与被测装载链零共享。
    #[tokio::test]
    async fn gpu_vram_manifest_checksum() -> Result<(), crate::contract::ModelError> {
        use crate::module::Layout;
        if !crate::testkit::gpu_enabled() {
            crate::testkit::skip_note();
            return Ok(());
        }
        let dir = manifest_dir().join("assets/Qwen3.5-0.8B");
        let path = dir.join("model.safetensors-00001-of-00001.safetensors");

        // ── 独立锚:一次性自读文件 → 键 → (dtype, 字节区间视图)
        let raw_buf = owl_shared::file_loader::read(&path).map_err(|e| ModelError::Msg(format!("{e}")))?;
        let raw = std::sync::Arc::new(raw_buf);
        let st = safetensors::SafeTensors::deserialize(&raw)
            .map_err(|e| ModelError::Msg(format!("锚解析: {e}")))?;
        let mut anchor: HashMap<String, (safetensors::Dtype, usize, usize)> = HashMap::new();
        for (name, t) in st.iter() {
            let off = t.data().as_ptr() as usize - raw.as_ptr() as usize;
            anchor.insert(name.to_string(), (t.dtype(), off, t.data().len()));
        }
        let anchor_decode = |dtype: safetensors::Dtype,
                             bytes: &[u8],
                             out: &mut Vec<f32>| {
            match dtype {
                safetensors::Dtype::F32 => {
                    out.clear();
                    out.extend(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])));
                }
                safetensors::Dtype::BF16 => {
                    out.clear();
                    out.extend(bytes.chunks_exact(2).map(|c| {
                        f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
                    }));
                }
                _ => panic!("锚: 不支持的 dtype"),
            }
        };

        // ── 装载(被测链;单次装载保留 manifest —— F5-2 前为重跑 load
        //    生成 manifest,双倍上传;F5-2 删)
        let mut gpu = crate::testkit::gpu_client().await;
        let model = Model::new(
            &qwen3_5_0_8b(),
            Qwen35Convention::new("model.language_model"),
            crate::module::QuantPlan::F16,
            crate::module::QuantPlan::F16,
        );
        let manifest = {
            let src = SafeTensorsSource::open_dir(&dir)?;
            let ctx = crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false, verify: false, debug_tap: false };
            crate::interpreters::eval_load(&model, &mut gpu, &src, &ctx).await?
        };
        eprintln!("[chk] manifest {} 条", manifest.entries().len());

        // ── 逐条 dtoh 读回 + 朴素期望 + 逐位比较
        let mut bad = 0usize;
        for (i, e) in manifest.entries().iter().enumerate() {
            let n: usize = e.shape.iter().product();
            // 刀2:合并键锚 = 两子键解码行堆叠(检查点无合并条目)
            let merged_anchor: Option<Vec<f32>> = e
                .key
                .strip_suffix(".weight")
                .and_then(crate::formats::split_qkvz)
                .map(|(b1, b2)| {
                    let mut v = Vec::new();
                    for b in [format!("{b1}.weight"), format!("{b2}.weight")] {
                        let (dtype, off, nbytes) = anchor[&b];
                        let mut tmp = Vec::new();
                        anchor_decode(dtype, &raw[off..off + nbytes], &mut tmp);
                        v.extend(tmp);
                    }
                    v
                });
            let mut want = Vec::new();
            if let Some(v) = merged_anchor {
                want = v;
            } else {
                let (dtype, off, nbytes) = anchor[&e.key];
                let raw_slice = &raw[off..off + nbytes];
                anchor_decode(dtype, raw_slice, &mut want);
            }
            if e.layout == Layout::Transposed {
                // 源 [out, in] → 声明 [in, out]:朴素转置
                let (cols, rows) = (e.shape[0], e.shape[1]);
                let mut t = vec![0f32; n];
                for r in 0..rows {
                    for c in 0..cols {
                        t[c * rows + r] = want[r * cols + c];
                    }
                }
                want = t;
            }
            let esz = e.dtype.size_bytes();
            let mut got_bytes = vec![0u8; n * esz];
            gpu.dtoh(&e.block, &mut got_bytes).await?;
            // 期望链:锚 f32(含朴素转置)→ f16 位型(与装载链同一 half
            // 转换,位型应全等;f16 基线后块内字节 = f16)
            let mism = got_bytes
                .chunks_exact(esz)
                .zip(want.iter())
                .position(|(c, w)| {
                    let g = u16::from_le_bytes([c[0], c[1]]);
                    g != half::f16::from_f32(*w).to_bits()
                });
            match mism {
                None => {}
                Some(p) => {
                    bad += 1;
                    let g = half::f16::from_bits(u16::from_le_bytes([
                        got_bytes[p * 2],
                        got_bytes[p * 2 + 1],
                    ]));
                    eprintln!("[chk] ✗ #{i} {} 首个错位 {p}: gpu {g} vs 锚 f16({})", e.key, want[p]);
                }
            }
            if i % 100 == 0 {
                eprintln!("[chk] … {i}/{} 已核", manifest.entries().len());
            }
        }
        eprintln!("[chk] 完成: {} 条中 {bad} 条不符", manifest.entries().len());
        assert_eq!(bad, 0, "显存数据与独立锚逐位不一致");
        gpu.close().await?;
        Ok(())
    }


    /// ★ matmul 维度二分探针:同尺寸矩阵直调算子,host 参考逐元素对拍。
    /// 两种数据形态:from_host 叶子 / of_block 上传块(复刻 diag 场景)。
    #[tokio::test]
    async fn gpu_matmul_dim_probe() -> Result<(), crate::contract::ModelError> {
        use crate::contract::DeviceClient;
        if !crate::testkit::gpu_enabled() {
            crate::testkit::skip_note();
            return Ok(());
        }
        let mut gpu = crate::testkit::gpu_client().await;
        let cases: [(&str, usize, usize, usize); 5] = [
            ("fixture      m1-k6-n6", 1, 6, 6),
            ("k1024-n24   ", 1, 1024, 24),
            ("k1024-n384  ", 1, 1024, 384),
            ("k1024-n6144 ", 1, 1024, 6144),
            ("m4-k1024-n6144", 4, 1024, 6144),
        ];
        for (name, m, k, n) in cases {
            let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
            let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.125).collect();

            // 形态一:from_host 叶子
            let a_t = TensorOps::from_host(Dtype::F32, vec![m, k], &f32b(&a));
            let b_t = TensorOps::from_host(Dtype::F32, vec![k, n], &f32b(&b));
            let got = crate::testkit::harvest(&mut gpu, &a_t.matmul(&b_t)).await;

            // host 参考(朴素)
            let mut want = vec![0f32; m * n];
            for i in 0..m {
                for p in 0..k {
                    let av = a[i * k + p];
                    for j in 0..n {
                        want[i * n + j] += av * b[p * n + j];
                    }
                }
            }
            let maxdiff = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            let zeros = got.iter().filter(|v| **v == 0.0).count();
            eprintln!("[mm] {name} from_host : maxdiff={maxdiff:.6} zeros={zeros}/{}", got.len());

            // 形态二:of_block 上传块(htod_f32 后以 Block 叶子引用)
            let sa = vec![m, k];
            let sb = vec![k, n];
            let ba = gpu.htod_f32(&sa, a.clone()).await?;
            let bb = gpu.htod_f32(&sb, b.clone()).await?;
            let a_b = TensorOps::of_block(ba.id, Dtype::F32, vec![m, k]);
            let b_b = TensorOps::of_block(bb.id, Dtype::F32, vec![k, n]);
            let got2 = crate::testkit::harvest(&mut gpu, &a_b.matmul(&b_b)).await;
            let maxdiff2 = got2
                .iter()
                .zip(&want)
                .map(|(g, w)| (g - w).abs())
                .fold(0.0f32, f32::max);
            let zeros2 = got2.iter().filter(|v| **v == 0.0).count();
            eprintln!("[mm] {name} of_block  : maxdiff={maxdiff2:.6} zeros={zeros2}/{}", got2.len());
            assert!(
                maxdiff < 1e-2 && maxdiff2 < 1e-2,
                "{name} matmul 错误: from_host {maxdiff} / of_block {maxdiff2}"
            );
        }
        gpu.close().await?;
        Ok(())
    }

    /// 批P5 真模型验收:0.8B prefill T=8 vs 8×decode 逐步(单序列语义:
    /// GDN 状态格恒 0,KV 行随 token 走)+ 生成冒烟(prefill 喂 prompt,
    /// greedy 4 步,tokenizer 解码打印)。
    /// ⚠️ KV 槽本测独立 8 槽(测试常量 SLOTS=4 是 smoke 的两步口径)。
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gpu_real_weights_prefill() {
        use crate::layers::gdn::GdnBuffers;
        use crate::layers::rope::Rope;
        use crate::module::KvBuffers;
        let dir = manifest_dir().join("assets/Qwen3.5-0.8B");
        let t_len = 8usize;
        let slots_n = 8usize;
        let eprintln_skip = (); // 门控同 smoke(真权重测试默认必跑)
        let _ = eprintln_skip;

        let mut gpu = crate::testkit::gpu_client().await;
        let model = load_0_8b(&dir, &mut gpu).await.expect("load_0_8b(真权重)");
        let rp = Rope::new(262_144, HD, 64, 10_000_000.0).expect("rope");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(),
            &crate::module::LoaderCtx { dtype: Dtype::F16, shard: 1, device_repack: false, verify: false, debug_tap: false })
            .await.expect("rope 表(f16 正确姿势)");

        // 常驻缓冲 ×2 套(ref / prefill;KV 8 槽,GDN 状态格用 0)
        async fn mk(
            gpu: &mut owl_cuda::GpuClient, slots_n: usize, hd: usize, hkv: usize,
        ) -> (Vec<KvBuffers>, Vec<GdnBuffers>) {
            // paged 布局(页 32 = 层分派契约;恒等块表)
            let nb = (slots_n + 31) / 32;
            let kv_len = nb * hkv * hd * 32;
            let bt_data: Vec<u8> = (0..nb).flat_map(|i| (i as f32).to_le_bytes()).collect();
            let bt = crate::interpreters::eval_ops(
                TensorOps::from_host(Dtype::F32, vec![1, nb], &bt_data).step(), gpu)
                .await.expect("恒等块表");
            let mut kvs = Vec::new();
            for _ in 0..6 {
                kvs.push(KvBuffers {
                    k_cache: zero_block_f16(gpu, kv_len, vec![nb, hkv, hd / 8, 32, 8]).await,
                    v_cache: zero_block_f16(gpu, kv_len, vec![nb, hkv, hd, 32]).await,
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0])),
                    block_tables: TensorOps::of_block(bt.id, Dtype::F32, vec![1, nb]),
                });
            }
            let mut gdns = Vec::new();
            for _ in 0..18 {
                gdns.push(GdnBuffers {
                    conv_q: zero_block(gpu, slots_n * 2048 * 3, vec![slots_n, 2048, 3]).await,
                    conv_k: zero_block(gpu, slots_n * 2048 * 3, vec![slots_n, 2048, 3]).await,
                    conv_v: zero_block(gpu, slots_n * 2048 * 3, vec![slots_n, 2048, 3]).await,
                    rec: zero_block(gpu, slots_n * 16 * 128 * 128, vec![slots_n, 16, 128, 128]).await,
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                });
            }
            (kvs, gdns)
        }
        let (kvs_ref, gdns_ref) = mk(&mut gpu, slots_n, HD, HKV).await;
        let (kvs_pre, gdns_pre) = mk(&mut gpu, slots_n, HD, HKV).await;

        // prompt:8 token(合法 id;语义无关 —— 验的是路径等价)
        let prompt: Vec<f32> = [985.0, 4123.0, 711.0, 2023.0, 1546.0, 884.0, 3001.0, 4195.0]
            .iter().map(|v| *v).collect();

        // 参考:T 次 decode 步(单序列:gdn 恒 0,kv 行 = t)
        let mut ref_rows: Vec<Vec<f32>> = Vec::new();
        for t in 0..t_len {
            let ids = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[prompt[t]]));
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32]));
            let kvs_step: Vec<KvBuffers> = kvs_ref
                .iter()
                .map(|kv| KvBuffers {
                    k_cache: kv.k_cache.clone(),
                    v_cache: kv.v_cache.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32])),
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32 + 1.0])),
                    block_tables: kv.block_tables.clone(),
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdns_ref
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                })
                .collect();
            let ctx = ForwardCtx::model_decode(1, &pos_t, &kvs_step, &rp, &gdns_step);
            let logits = model.forward(&ids, &ctx);
            ref_rows.push(crate::testkit::harvest_f16(&mut gpu, &logits).await);
        }

        // 被测:一次 prefill T=8
        let ids_all = TensorOps::from_host(Dtype::F32, vec![t_len], &f32b(&prompt));
        let pos_all = TensorOps::from_host(Dtype::F32, vec![t_len],
            &f32b(&(0..t_len).map(|t| t as f32).collect::<Vec<_>>()));
        let slots_all = TensorOps::from_host(Dtype::F32, vec![t_len],
            &f32b(&(0..t_len).map(|t| t as f32).collect::<Vec<_>>()));
        let lens_all = TensorOps::from_host(Dtype::F32, vec![t_len],
            &f32b(&(0..t_len).map(|t| t as f32 + 1.0).collect::<Vec<_>>()));
        let gdn_slot = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0]));
        let ctx = ForwardCtx::model_prefill(t_len, &pos_all, &kvs_pre, &rp, &gdns_pre,
            &slots_all, &lens_all, &gdn_slot, 0, None);
        let all = crate::testkit::harvest_f16(&mut gpu, &model.forward(&ids_all, &ctx)).await;

        // 逐行等价(路径等价 = PF1a 契约):T 批 cuBLAS(m=8)vs 逐步
        // (m=1)核选型/浮序差 → 相对容差 5e-2 + top-1 逐行一致
        for t in 0..t_len {
            let row = &all[t * VOCAB..(t + 1) * VOCAB];
            for (i, (g, w)) in row.iter().zip(&ref_rows[t]).enumerate() {
                assert!(
                    (g - w).abs() <= 5e-2 * (1.0 + w.abs()),
                    "prefill 行{t}[{i}] {g} vs {w}"
                );
            }
            let (tp, tr) = (
                row.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a }).0,
                ref_rows[t].iter().enumerate().fold((0usize, f32::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a }).0,
            );
            assert_eq!(tp, tr, "prefill 行{t} top-1 不一致: {tp} vs {tr}");
        }

        // 生成冒烟:prefill 喂 prompt → 末行 greedy → decode 4 步 → 文本打印
        // 测试档 = 27B(两参化;0.8B 档另有调用点)
        let tok = load_tokenizer(&dir, &qwen3_8_27b().tokenizer).expect("tokenizer");
        let mut gen_ids: Vec<u32> = Vec::new();
        let mut next = all[(t_len - 1) * VOCAB..t_len * VOCAB]
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a })
            .0 as u32;
        for step in 0..4u32 {
            gen_ids.push(next);
            let t = t_len + step as usize;
            let ids1 = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[next as f32]));
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32]));
            let kvs_step: Vec<KvBuffers> = kvs_pre
                .iter()
                .map(|kv| KvBuffers {
                    k_cache: kv.k_cache.clone(),
                    v_cache: kv.v_cache.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32])),
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[t as f32 + 1.0])),
                    block_tables: kv.block_tables.clone(),
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdns_pre
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                })
                .collect();
            let ctx = ForwardCtx::model_decode(1, &pos_t, &kvs_step, &rp, &gdns_step);
            let logits = model.forward(&ids1, &ctx);
            let row = crate::testkit::harvest_f16(&mut gpu, &logits).await;
            next = row.iter().enumerate()
                .fold((0usize, f32::NEG_INFINITY), |a, (i, v)| if *v > a.1 { (i, *v) } else { a })
                .0 as u32;
        }
        let text = tok.decode(&gen_ids);
        eprintln!("[prefill-smoke] 生成: {text:?}");
        assert!(!text.is_empty(), "生成冒烟:非空");
        gpu.close().await.expect("关机");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gpu_real_weights_smoke() {
        use crate::layers::gdn::GdnBuffers;
        use crate::layers::rope::Rope;
        use crate::module::KvBuffers;
        // if !crate::testkit::gpu_enabled() {
        //     crate::testkit::skip_note();
        //     return;
        // }
        let dir = manifest_dir().join("assets/Qwen3.5-0.8B");
        eprintln!("[smoke] boot server...");
        let mut gpu = crate::testkit::gpu_client().await;
        eprintln!("[smoke] booted; open safetensors...");
        // 实际字节量从文件取(禁硬编码口径 —— 3.9GB f32 是 f32 时代残留,
        // bf16 检查点 1.7GB,f16 装载后同为 1.7GB)
        let mut ckpt_bytes = 0u64;
        for e in owl_shared::file_loader::read_dir(&dir).expect("读目录") {
            let p = e.expect("dir entry").path();
            if p.extension().is_some_and(|x| x == "safetensors") {
                ckpt_bytes += p.metadata().expect("meta").len();
            }
        }
        let t_load = std::time::Instant::now();
        let model = load_0_8b(&dir, &mut gpu).await.expect("load_0_8b(真权重)");
        let dt = t_load.elapsed().as_secs_f64();
        eprintln!(
            "[smoke] loaded 24 layers ({dt:.2}s,检查点 {:.2}GB → {:.2}GB/s)",
            ckpt_bytes as f64 / 1e9,
            ckpt_bytes as f64 / 1e9 / dt
        );
        assert_eq!(model.layers.len(), 24);
        assert_eq!(model.layers.iter().filter(|l| l.is_full()).count(), 6);

        // rope 表(theta 1e7 / partial 64 / max_pos 262144)
        let rp = Rope::new(262_144, HD, 64, 10_000_000.0).expect("rope");
        eprintln!("[smoke] rope 表...");
        crate::interpreters::eval_load(&rp, &mut gpu, &rp.tables(), &Default::default())
            .await
            .expect("rope 表");
        eprintln!("[smoke] rope ok; 分配常驻缓冲...");

        // 常驻缓冲:6 full 层 KV + 18 gdn 层状态(真维度;全零起步)
        let nb = (SLOTS + 31) / 32; // paged 池(页 32 = 层分派契约)
        let kv_len = nb * HKV * HD * 32;
        let bt_data: Vec<u8> = (0..nb).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let bt = crate::interpreters::eval_ops(
            TensorOps::from_host(Dtype::F32, vec![1, nb], &bt_data).step(), &mut gpu)
            .await.expect("恒等块表");
        let kvs: Vec<KvBuffers> = {
            let mut v = Vec::new();
            for _ in 0..6 {
                v.push(KvBuffers {
                    k_cache: zero_block(&mut gpu, kv_len, vec![nb, HKV, HD / 8, 32, 8]).await,
                    v_cache: zero_block(&mut gpu, kv_len, vec![nb, HKV, HD, 32]).await,
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0])),
                    block_tables: TensorOps::of_block(bt.id, Dtype::F32, vec![1, nb]),
                });
            }
            v
        };
        let key_dim = 2048;
        let gdns: Vec<GdnBuffers> = {
            let mut v = Vec::new();
            for _ in 0..18 {
                v.push(GdnBuffers {
                    conv_q: zero_block(&mut gpu, SLOTS * key_dim * 3, vec![SLOTS, key_dim, 3]).await,
                    conv_k: zero_block(&mut gpu, SLOTS * key_dim * 3, vec![SLOTS, key_dim, 3]).await,
                    conv_v: zero_block(&mut gpu, SLOTS * key_dim * 3, vec![SLOTS, key_dim, 3]).await,
                    rec: zero_block(&mut gpu, SLOTS * 16 * 128 * 128, vec![SLOTS, 16, 128, 128]).await,
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[0.0])),
                });
            }
            v
        };

        eprintln!("[smoke] buffers ok; 两步 decode...");
        // 两步 decode(tokenizer 未接,任意合法 id;验证数值健康与状态推进)
        let steps = [(985.0f32, 0.0f32, 0.0f32, 1.0f32), (4123.0, 1.0, 1.0, 2.0)];
        let mut prev_top: Option<usize> = None;
        for (si, &(id, pos, slot, kv_len)) in steps.iter().enumerate() {
            let ids = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[id]));
            let pos_t = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[pos]));
            let kvs_step: Vec<KvBuffers> = kvs
                .iter()
                .map(|kv| KvBuffers {
                    k_cache: kv.k_cache.clone(),
                    v_cache: kv.v_cache.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot])),
                    kv_lens: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[kv_len])),
                    block_tables: kv.block_tables.clone(),
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdns
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[slot])),
                })
                .collect();
            let ctx = ForwardCtx::model_decode(1, &pos_t, &kvs_step, &rp, &gdns_step);
            let logits = model.forward(&ids, &ctx);
            assert_eq!(logits.shape(), &[1, VOCAB]);
            // f16 logits(F5 整模切换):harvest_f16 解码回 f32(top 判读口径不变)
            let got = crate::testkit::harvest_f16(&mut gpu, &logits).await;
            assert!(got.iter().all(|v| v.is_finite()), "step{si} logits 应全有限");
            let top = (0..VOCAB)
                .max_by(|&a, &b| got[a].abs().total_cmp(&got[b].abs()))
                .unwrap();
            println!("  [smoke] step{si}: id {id} → top@{top} (logit {:.4})", got[top]);
            if let Some(p) = prev_top {
                assert_ne!(p, top, "两步状态应推进(top1 不应不变)");
            }
            prev_top = Some(top);
        }
        gpu.close().await.expect("关机");
    }
    /// E5-DF1 验收:DFlash2 检查点装载(81 键)+ fc 采样对拍 +
    /// encode/selector 真几何冒烟。门控 OWL_TEST_DEVICE + OWL_DFLASH2_DIR
    /// (z-lab/Qwen3.8-27B-DFlash2 目录)。
    #[tokio::test]
    async fn gpu_dflash2_27b_loads_and_smokes() {
        if !crate::testkit::gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        let Some(dir) = owl_shared::env_reader::str("OWL_DFLASH2_DIR") else {
            eprintln!("skip: OWL_DFLASH2_DIR 未设(DFlash2 检查点目录)");
            return;
        };
        let dir = std::path::Path::new(&dir);
        let mut gpu = crate::testkit::gpu_client().await;
        let (draft, manifest) = load_27b_dflash2(dir, &mut gpu, false)
            .await
            .expect("DFlash2 装载");
        assert_eq!(draft.hidden(), 5120);

        // 键集断言:BF16 = 81 键;W4A16 = 量化三槽 × 41 线性 + passthrough
        let keys: Vec<String> = manifest.entries().iter().map(|e| e.key.clone()).collect();
        let is_q = keys.iter().any(|k| k.ends_with(".qweight"));
        if is_q {
            assert!(
                keys.len() > 120 && keys.len() < 260,
                "W4A16 键数异常 {}",
                keys.len()
            );
        } else {
            assert_eq!(keys.len(), 81, "键数实际 {}", keys.len());
        }
        if is_q {
            for expect in [
                "fc_0.qweight",
                "fc_4.scales",
                "hidden_norm.weight",
                "norm.weight",
                "candidate_selector.hidden_projection.weight",
                "candidate_selector.predecessor_codebook",
                "candidate_selector.successor_codebook",
                "layers.0.attention_conv.base_kernel",
                "layers.0.self_attn.q_proj.qweight",
                "layers.4.mlp.down_proj.qweight",
                "layers.4.self_attn.k_norm.weight",
            ] {
                assert!(keys.iter().any(|k| k == expect), "缺键 {expect}");
            }
        } else {
            for expect in [
                "fc.weight",
                "hidden_norm.weight",
                "norm.weight",
                "candidate_selector.hidden_projection.weight",
                "candidate_selector.predecessor_codebook",
                "candidate_selector.successor_codebook",
                "layers.0.attention_conv.base_kernel",
                "layers.4.mlp_conv.kernel_projection.weight",
                "layers.4.self_attn.k_norm.weight",
                "layers.4.mlp.down_proj.weight",
            ] {
                assert!(keys.iter().any(|k| k == expect), "缺键 {expect}");
            }
        }

        // fc 采样对拍:y[o] = Σ_k tap[k]·W[o][k](W [5120, 25600] 行主序;
        // 采样 8 个输出元全量内积 —— 免 262M 全量 MAC)
        use crate::module::WeightSource;
        // host 参考:fc 反量化值(W4A16Source 反量化臂;BF16 源同键直读)
        let mut src = crate::formats::w4a16::W4A16Source::open_dir(dir).unwrap();
        let w_fc = src.dequant_linear("fc").expect("fc 源(反量化)");
        let hn = src.take("hidden_norm.weight").expect("hn 源");
        let fan = 5 * 5120usize;
        let (hidden_d, row_n) = (5120usize, 5120usize);
        let row: Vec<f32> = (0..row_n)
            .map(|i| half::f16::from_f32(((i as f32) * 0.13).sin() * 0.4).to_f32())
            .collect();
        let row_t = crate::TensorOps::from_host(
            crate::contract::Dtype::F16,
            vec![1, row_n],
            &row.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>(),
        );
        let ctx = crate::module::ForwardCtx::minimal(1);
        let mem = draft.project_memory(&[row_t.clone(), row_t.clone(), row_t.clone(), row_t.clone(), row_t.clone()], &ctx);
        let got = crate::testkit::harvest_f16(&mut gpu, &mem).await;
        let r16 = |x: f32| half::f16::from_f32(x).to_f32();
        // 完整行 host 重建(1 行 fc = 131M MAC;debug 逐元素循环偏慢属预期)
        let mut fc_out = vec![0f32; hidden_d];
        for (o, out) in fc_out.iter_mut().enumerate() {
            // 五 taps 恒等 → y[o] = Σ_{j<fan} row[j % 5120]·W[o][j](全列块)
            let mut acc = 0f32;
            for j in 0..fan {
                acc += row[j % row_n] * r16(w_fc[o * fan + j]);
            }
            *out = acc;
        }
        let ms: f32 = fc_out.iter().map(|&v| v * v).sum::<f32>() / 5120.0;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        let mut want = vec![0f32; 5120];
        for c in 0..5120 {
            want[c] = fc_out[c] * inv * (1.0 + r16(hn[c]));
        }
        crate::testkit::assert_close(&got, &want, 5e-2, "project_memory(fc+hidden_norm)");

        // selector 真几何冒烟:logits [7, 248320](host 随机)→ topk →
        // lattice + walk;锚 = 输出有限 + token 落词表内
        let vocab = 248320usize;
        let rows = crate::layers::dflash2::DEPTH;
        let logits: Vec<f32> = (0..rows * vocab)
            .map(|i| half::f16::from_f32(((i as f32) * 0.017).sin() * 8.0).to_f32())
            .collect();
        let logits_t = crate::TensorOps::from_host(
            crate::contract::Dtype::F16,
            vec![rows, vocab],
            &logits.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>(),
        );
        let topk = crate::TensorOps::call(crate::ops::SemanticKernel::Topk16)
            .aux(&[rows])
        .arg(&logits_t)
        .arg_i32(vocab as i32)
        .with_shape(crate::contract::Dtype::F32, vec![rows, 32]);
        let topk_full = crate::testkit::harvest(&mut gpu, &topk).await;
        let cand: Vec<f32> = (0..rows)
            .flat_map(|r| topk_full[r * 32 + 16..(r + 1) * 32].to_vec())
            .collect();
        let unary: Vec<f32> = (0..rows)
            .flat_map(|r| topk_full[r * 32..r * 32 + 16].to_vec())
            .collect();
        // proj = hidden_projection(hidden)[rows, 256];hidden 随机 f16
        let hidden: Vec<f32> = (0..(rows + 1) * 5120)
            .map(|i| half::f16::from_f32(((i as f32) * 0.019).cos() * 0.3).to_f32())
            .collect();
        let hidden_t = crate::TensorOps::from_host(
            crate::contract::Dtype::F16,
            vec![rows + 1, 5120],
            &hidden.iter().flat_map(|f| half::f16::from_f32(*f).to_le_bytes()).collect::<Vec<u8>>(),
        );
        let pred = hidden_t.slice_view(5120, vec![rows, 5120]);
        let proj = crate::interpreters::eval_ops(
            draft.selector().proj().forward(&pred, &ctx).step(), &mut gpu)
            .await
            .expect("proj eval");
        let proj_t = crate::TensorOps::of_block(proj.id, crate::contract::Dtype::F16, vec![rows, 256]);
        let anchor_t = crate::TensorOps::from_host(
            crate::contract::Dtype::F32, vec![1], &crate::testkit::f32b(&[42.0]),
        );
        let sel_out = crate::TensorOps::call(crate::ops::SemanticKernel::DflashSelect)
            .aux(&[0])
        .arg(&crate::TensorOps::from_host(crate::contract::Dtype::F32, vec![rows, 16], &crate::testkit::f32b(&cand)))
        .arg(&crate::TensorOps::from_host(crate::contract::Dtype::F32, vec![rows, 16], &crate::testkit::f32b(&unary)))
        .arg(&proj_t)
        .arg(&anchor_t)
        .arg(&draft.selector().a_code().decl())
        .arg(&draft.selector().b_code().decl())
        .arg_i32(rows as i32)
        .arg_i32(16)
        .arg_i32(256)
        .with_shape(crate::contract::Dtype::F32, vec![rows * (16 * 16 + 1)]);
        let sel_full = crate::testkit::harvest(&mut gpu, &sel_out).await;
        for (e, &tok) in sel_full[..rows].iter().enumerate() {
            assert!(tok >= 0.0 && tok < vocab as f32, "walk token 越界 e{e} = {tok}");
            assert!(tok.fract() == 0.0, "walk token 非整 e{e} = {tok}");
        }
        eprintln!("[dflash2] 27B 装载 + fc 对拍 + selector 冒烟全绿({} 键,W4A16 = {is_q})", keys.len());
        gpu.close().await.expect("关机");
    }

}
