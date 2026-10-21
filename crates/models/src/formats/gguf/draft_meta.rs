//! **GGUF 草稿 meta**(`dflash.*` KV → [`DraftSpec`];2026-10-20 DFlash2 线 P0)。
//!
//! 与主干 [`super::meta::spec_from_gguf`](crate::formats::gguf::spec_from_gguf)
//! **两条入口**:主干只认 `qwen35`(其余响亮拒);草稿只认 `dflash`。两个文件
//! 是两类东西 —— 草稿没有 embed/lm_head(共享 target),层型恒 full-attention,
//! 且自带 taps/conv/selector 家族常数。**绝不把草稿文件喂进主干路径**。
//!
//! # 三条硬门(全部来自实测,不靠推断)
//!
//! 1. **`target_layers` 是 1-based**(2026-10-20 实测):GGUF 写的是
//!    `[6,20,34,48,62]`,而 HF `dflash_config.target_layer_ids` 是
//!    `[5,19,33,47,61]` —— 转换器统一 `+1`(ggml 层号习惯)。**少减 1 会让
//!    taps 整体错一层 ⇒ AL 静默塌陷**(不报错、不崩)。故:
//!    `tap_ids = target_layers - 1`,并与 27B 家族常量**逐值对账**(G0-b)。
//! 2. **`mask_token_id` 未导出**:GGUF 没有这个 KV(HF config 有 248070)。
//!    v1 = 家族常量兜底 + **响亮点名**(不许静默);等转换器补 KV 后自动优先。
//! 3. **norm 未烘焙 `+1`**(2026-10-20 与 HF BF16 检查点逐元素实测:max|Δ|=0):
//!    草稿线的转换器**没有**做主干那套 `w+1`。⇒ 草稿源事实恒
//!    `SourceFacts::HF`(零中心 `×(1+w)` 语义),**不得**套用
//!    [`SourceFacts::GGUF`](crate::module::SourceFacts)。

use crate::contract::ModelError;
use crate::layers::dflash2::{BLOCK, DEPTH, DRAFT_TAP_IDS, GROUP, RANK, TAPS, TOP_K};

use super::reader::Gguf;

/// 草稿家族事实(逐字段来自 `dflash.*` KV;**不与 target 家族常量做等值绑定**,
/// 家族常量只作交叉校验)
#[derive(Debug, Clone, PartialEq)]
pub struct DraftSpec {
    pub hidden: usize,
    pub inter: usize,
    pub hq: usize,
    pub hkv: usize,
    pub hd: usize,
    pub layers: usize,
    pub eps: f32,
    /// 噪声块宽(= 1 锚 + depth)
    pub block: usize,
    /// 每轮草稿数(= `block - 1`)
    pub depth: usize,
    /// conv 卷积核长
    pub conv_taps: usize,
    /// conv 组内通道
    pub conv_group: usize,
    /// selector 码本秩
    pub rank: usize,
    /// selector 每 slot 候选数
    pub top_k: usize,
    /// 噪声块 mask token(**GGUF 未导出**:家族兜底 + 点名)
    pub mask_token: u32,
    /// target 层号(**0-based**;已由 `target_layers - 1` 归一)
    pub tap_ids: Vec<usize>,
    /// 草稿注意力窗(全层 true)
    pub sliding_window: Option<usize>,
    /// 是否因果(草稿 = false / ENCODER_ONLY)
    pub causal: bool,
    pub rope_max_pos: usize,
    pub rope_theta: f32,
}

/// target 侧事实(草稿**不自带**的东西:词表、层数、hidden —— 全部共享)
#[derive(Clone, Copy, Debug)]
pub struct TargetFacts {
    pub vocab: usize,
    pub layers: usize,
    pub hidden: usize,
}

impl DraftSpec {
    /// 27B 家族 taps(HF `dflash_config.target_layer_ids`,0-based)。
    /// 只作 **G0-b 交叉校验**:GGUF 声明必须归一到这一组,否则拒启。
    pub const FAMILY_27B_TAP_IDS: [usize; 5] = DRAFT_TAP_IDS;
    /// 27B 家族 mask token(GGUF 未导出;v1 兜底)
    pub const FAMILY_27B_MASK_TOKEN: u32 = 248070;

    /// `def` = 家族默认;`name` = 配置项名(上屏用)
    fn family_fallback(def: usize, name: &str) -> usize {
        owl_shared::log_warn!(
            "dflash",
            "GGUF 草稿未声明 `{name}`,取 27B 家族常量 {def}(文件转换器未导出该字段 —— \
             若换家族必须由转换器补 KV,不许靠默认值蒙)"
        );
        def
    }

    /// **唯一入口**:`dflash.*` KV → 声明事实。
    ///
    /// `target` = 主干事实(词表/层数/hidden);三处**跨文件一致性**在此判死:
    /// hidden 不等 ⇒ taps 张量对不上;taps 越界 ⇒ `tapped_hidden` 会在层循环
    /// 末尾 `expect` 崩(本仓纪律:启动期"响亮拒",不许让内部不变量失守)。
    pub fn from_gguf(g: &Gguf, target: TargetFacts) -> Result<Self, ModelError> {
        let arch = g.arch()?.to_string();
        if arch != "dflash" {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿 meta:架构 `{arch}` 不是草稿线(只接 `dflash`)—— \
                 主干文件请走 `spec_from_gguf`,两条入口不共用"
            )));
        }
        let kv = &g.kv;
        let p = |k: &str| format!("dflash.{k}");
        let m = |e: String| ModelError::Msg(format!("gguf 草稿 meta: {e}"));

        let hidden = kv.usize(&p("embedding_length")).map_err(&m)?;
        let inter = kv.usize(&p("feed_forward_length")).map_err(&m)?;
        let hq = kv.usize(&p("attention.head_count")).map_err(&m)?;
        let hkv = kv.usize(&p("attention.head_count_kv")).map_err(&m)?;
        let hd = kv.usize(&p("attention.key_length")).map_err(&m)?;
        let vd = kv.usize(&p("attention.value_length")).map_err(&m)?;
        let eps = kv.f32(&p("attention.layer_norm_rms_epsilon")).map_err(&m)?;
        let layers = kv.usize(&p("block_count")).map_err(&m)?;
        let block = kv.usize(&p("block_size")).map_err(&m)?;
        let rank = kv.usize(&p("selector_rank")).map_err(&m)?;
        let top_k = kv.usize(&p("selector_top_k")).map_err(&m)?;
        let conv_taps = kv.opt_usize(&p("conv_kernel_size")).map_err(&m)?.unwrap_or_else(|| Self::family_fallback(TAPS, "conv_kernel_size"));
        let conv_group = kv.opt_usize(&p("conv_group_size")).map_err(&m)?.unwrap_or_else(|| Self::family_fallback(GROUP, "conv_group_size"));
        let causal = kv.opt_bool(&p("attention.causal")).map_err(&m)?.unwrap_or(false);
        let sliding_window = kv.opt_usize(&p("attention.sliding_window")).map_err(&m)?;
        let rope_max_pos = kv.opt_usize(&p("context_length")).map_err(&m)?.unwrap_or(262_144);
        let rope_theta = kv.opt_f32(&p("rope.freq_base")).map_err(&m)?.unwrap_or(1.0e7);
        let mask_token = match kv.opt_u32("dflash.mask_token_id").map_err(&m)? {
            Some(v) => v,
            None => Self::FAMILY_27B_MASK_TOKEN,
        };

        // ---- 跨文件一致性(启动期判死) ----
        if hidden != target.hidden {
            return Err(m(format!(
                "草稿 hidden {hidden} ≠ target hidden {}(taps 张量对不上 —— 不是同一家族的草稿)",
                target.hidden
            )));
        }
        if hd != vd {
            return Err(m(format!("key_length {hd} ≠ value_length {vd}(草稿注意力两者必须同宽)")));
        }
        if layers == 0 || hkv == 0 || hq % hkv != 0 {
            return Err(m(format!("头/层几何不成立:layers {layers} · hq {hq} · hkv {hkv}")));
        }
        if block != BLOCK {
            return Err(m(format!(
                "block_size {block} ≠ 家族定形 {BLOCK}(DEPTH={DEPTH} 与 verify 槽/拼接核槽数绑定;改它要同步核面)"
            )));
        }
        if !causal {
            owl_shared::log_info!("dflash", "草稿注意力 = 非因果块(causal=false;ENCODER_ONLY)");
        } else {
            return Err(m(
                "`dflash.attention.causal = true`:草稿线只支持非因果块注意力(sglang 同构)"
                    .into(),
            ));
        }

        // ---- taps:**1-based → 0-based**(最高危的一条) ----
        let raw = kv.opt_u32s(&p("target_layers")).map_err(&m)?.ok_or_else(|| {
            m("缺键 `dflash.target_layers`(草稿的 taps 声明,无它无法对齐 target 层)"
                .to_string())
        })?;
        if raw.is_empty() {
            return Err(m("`dflash.target_layers` 为空".into()));
        }
        if raw.len() != layers {
            return Err(m(format!(
                "`target_layers` 声明 {} 项 ≠ 层数 {layers}(草稿层 = 一个 tap 对应一层)",
                raw.len()
            )));
        }
        let mut tap_ids = Vec::with_capacity(raw.len());
        for v in &raw {
            let v = *v as usize;
            let z = v.checked_sub(1).ok_or_else(|| {
                m(format!(
                    "`target_layers` 含 0(值 {v}):GGUF 侧是 1-based,0 无意义 —— \
                     文件可能已被归一(重复减 1 会整体错一层)"
                ))
            })?;
            tap_ids.push(z);
        }
        if tap_ids.windows(2).any(|w| w[0] >= w[1]) {
            return Err(m(format!(
                "归一后 tap_ids {tap_ids:?} 非严格递增(GGUF 原值 {raw:?})—— \
                 taps 顺序 = 注意力捕获顺序,乱序会静默垫错层"
            )));
        }
        if let Some(&last) = tap_ids.last() {
            if last >= target.layers {
                return Err(m(format!(
                    "归一后 tap_ids {tap_ids:?} 越界(target 只有 {} 层;GGUF 原值 {raw:?})",
                    target.layers
                )));
            }
        }
        // **G0-b**:家族对账(防 1-based 误读 / 转换器版本分叉)
        let fam = Self::FAMILY_27B_TAP_IDS;
        if tap_ids != fam {
            return Err(m(format!(
                "taps 与 27B 家族常量对不上:归一后 {tap_ids:?} vs 家族 {fam:?}(GGUF 原值 {raw:?})。\
                 两种可能:① 文件不是 27B 家族;② `target_layers` 不是 1-based ⇒ \
                 两种情况都**必须拒启**(错一层 = AL 静默塌陷,不报错不崩)"
            )));
        }
        if mask_token == 0 || mask_token as usize >= target.vocab {
            return Err(m(format!(
                "mask_token {mask_token} 越界(词表 {})",
                target.vocab
            )));
        }

        Ok(Self {
            hidden,
            inter,
            hq,
            hkv,
            hd,
            layers,
            eps,
            block,
            depth: block - 1,
            conv_taps,
            conv_group,
            rank,
            top_k,
            mask_token,
            tap_ids,
            sliding_window,
            causal,
            rope_max_pos,
            rope_theta,
        })
    }

    /// 家族定形常数是否与本仓层内常量一致(改核/改家族时的显式门)
    pub fn assert_family_constants(&self) -> Result<(), ModelError> {
        let bad = [
            ("conv_kernel_size", self.conv_taps, TAPS),
            ("conv_group_size", self.conv_group, GROUP),
            ("selector_rank", self.rank, RANK),
            ("selector_top_k", self.top_k, TOP_K),
        ]
        .into_iter()
        .filter(|(_, got, want)| got != want)
        .map(|(k, got, want)| format!("{k}: 文件 {got} ≠ 本仓 {want}"))
        .collect::<Vec<_>>();
        if !bad.is_empty() {
            return Err(ModelError::Msg(format!(
                "GGUF 草稿声明与本仓层内常量不符({})—— 层内常量是编期定形事实,\
                 换家族要先把常量搬进配置(见 dflash2.rs 顶部 FIXME),不许静默忽略",
                bad.join("; ")
            )));
        }
        Ok(())
    }
}
