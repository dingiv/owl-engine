//! **TagFilter** —— 配置派生出的选核算子过滤器（2026-10-20）。
//!
//! 语义（用户裁决“配置就是过滤条件，不显式列 allow_tags”）：
//!
//! - tag 按 **互斥 facet** 分组（`quant.weight` / `kv.main` / `kv.draft` / `feature.vision` …）；
//! - 同一 facet 内是**单选**：算子的该组 tag 必须全部 ∈ 配置允许集；
//! - 算子在某 facet **没有 tag** ⇒ 该 facet 对它是“通用”，放行；
//!   （所以 `quant.w4a16` 不会误杀 rmsnorm/add 这类与量化无关的核）
//! - facet 之间 AND：W4A16 + fp8 KV + vision 同时成立才放行；
//! - `*` / [`TagFilter::wildcard`] = 关闭过滤（诊断/兜底）。
//!
//! 用例：
//! - `model.kind=gguf` ⇒ 允许集含 `quant.gguf` ⇒ marlin/W4A16 核被拒；
//! - `model.quant=w4a16` ⇒ 允许集含 `quant.w4a16` ⇒ W4A8/GGUF 核被拒；
//! - `dispatch.kv_fp8` ⇒ 允许集含 `kv.main.fp8` ⇒ f16 主池核被拒；
//!   草稿池独立为 `kv.draft.*`，不会与主池 facet 混淆。
//! - `model.vision` ⇒ 允许集含 `feat.vision` ⇒ 视觉核放行；
//!   未开时视觉核被拒（配置声明与实际发射不一致会响亮暴露）。

use crate::spec::{KernelMeta, KernelTag, EXCLUSIVE_TAG_FACETS, KERNEL_TAGS};

/// 配置派生过滤器（Copy；随 `EnvProvider`/`DiagOpts` 值传递）
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TagFilter {
    allowed: u64,
    wildcard: bool,
}

impl TagFilter {
    /// 全放行（诊断/测试兜底；不参与生产配置派生）
    pub const fn wildcard() -> Self {
        Self { allowed: 0, wildcard: true }
    }

    /// 允许集构造（空 = 只放行“无互斥 facet tag”的通用算子）
    pub fn from_tags(tags: &[KernelTag]) -> Self {
        let mut allowed = 0u64;
        for t in tags {
            allowed |= t.bit();
        }
        Self { allowed, wildcard: false }
    }

    /// 由 tag id 构造；`"*"` = 全放行；未知 id = 结构化 Err（不静默）
    pub fn from_ids(ids: &[&str]) -> Result<Self, String> {
        let mut allowed = 0u64;
        for id in ids {
            if *id == "*" {
                return Ok(Self::wildcard());
            }
            let t = crate::spec::tag_by_id(id)
                .ok_or_else(|| format!("未知 kernel tag {id:?}（见 crates/kernels/tags.toml）"))?;
            allowed |= t.bit();
        }
        Ok(Self { allowed, wildcard: false })
    }

    /// 是否全放行
    pub const fn is_wildcard(&self) -> bool {
        self.wildcard
    }

    /// 允许位集
    pub const fn allowed_mask(&self) -> u64 {
        self.allowed
    }

    /// 命中的**首个不兼容 facet**（`None` = 放行；诊断用）
    pub fn denied_facet(&self, meta: &KernelMeta) -> Option<&'static str> {
        if self.wildcard {
            return None;
        }
        let have = meta.tag_mask();
        for facet in EXCLUSIVE_TAG_FACETS {
            let hit = have & facet.mask;
            // 该 facet 无 tag = 通用算子；有 tag 则必须全部被配置选中。
            if hit != 0 && (hit & !self.allowed) != 0 {
                return Some(facet.name);
            }
        }
        None
    }

    /// 算子是否放行
    pub fn allows(&self, meta: &KernelMeta) -> bool {
        self.denied_facet(meta).is_none()
    }

    /// 放行 tag id 列表（诊断日志）
    pub fn describe(&self) -> String {
        if self.wildcard {
            return "*".into();
        }
        let ids: Vec<&str> = KERNEL_TAGS
            .iter()
            .filter(|t| self.allowed & t.tag.bit() != 0)
            .map(|t| t.id)
            .collect();
        ids.join(",")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(entry: &str) -> &'static KernelMeta {
        crate::ops::meta(entry).unwrap_or_else(|| panic!("缺算子 {entry}"))
    }

    #[test]
    fn w4a16_filter_rejects_other_weight_formats_but_allows_generic() {
        let f = TagFilter::from_tags(&[KernelTag::QUANT_W4A16]);
        assert!(f.allows(meta("marlin_gemm_w4a16_awq")), "W4A16 核应放行");
        assert!(!f.allows(meta("marlin_gemm_w4a8_awq")), "W4A8 核应拒");
        assert!(!f.allows(meta("owl_gguf_gemm_q4k")), "GGUF 核应拒");
        // 与量化无关的通用核没有 quant.* tag ⇒ 放行
        assert!(f.allows(meta("owl_rmsnorm_f16")));
        assert!(f.allows(meta("cublas_gemm_f16")));
    }

    #[test]
    fn w4a8_filter_keeps_activation_quant_and_a8_gemm() {
        let f = TagFilter::from_tags(&[KernelTag::QUANT_W4A8]);
        assert!(f.allows(meta("marlin_gemm_w4a8_awq")));
        assert!(f.allows(meta("owl_quant_int8_row_f16")));
        assert!(f.allows(meta("owl_ct_repack_a8_u32")));
        assert!(!f.allows(meta("marlin_gemm_w4a16_awq")));
    }

    #[test]
    fn main_and_draft_kv_facets_are_independent() {
        let f = TagFilter::from_tags(&[
            KernelTag::KV_MAIN_FP8,
            KernelTag::KV_DRAFT_BF16,
            KernelTag::FEAT_DFLASH,
            KernelTag::FEAT_SPEC,
        ]);
        assert!(f.allows(meta("vllm_paged_attention_v2_fp8_knhd_hd256bs32")));
        assert!(f.allows(meta("owl_naive_attn_nc_bf16_v2")), "草稿 BF16 池核");
        assert!(!f.allows(meta("owl_naive_attn_nc_f16")), "草稿 F16 池核应拒");
        assert!(!f.allows(meta("vllm_reshape_and_cache_bf16_knhd")), "主 BF16 池核应拒");
        assert!(!f.allows(meta("vllm_paged_attention_v2_f16_knhd_hd256bs32")));
    }

    #[test]
    fn gguf_filter_rejects_marlin() {
        let f = TagFilter::from_tags(&[KernelTag::QUANT_GGUF]);
        assert!(f.allows(meta("owl_gguf_gemm_q4k")));
        assert!(!f.allows(meta("marlin_gemm_w4a16_awq")));
        assert!(f.allows(meta("owl_add_f16")));
    }

    #[test]
    fn kv_filter_and_feature_filter_are_and_ed() {
        let f = TagFilter::from_tags(&[
            KernelTag::QUANT_W4A16,
            KernelTag::KV_MAIN_FP8,
            KernelTag::FEAT_VISION,
        ]);
        assert!(f.allows(meta("vllm_paged_attention_v2_fp8_knhd_hd256bs32")));
        assert!(!f.allows(meta("vllm_paged_attention_v2_f16_knhd_hd256bs32")));
        assert!(f.allows(meta("owl_layernorm_f16")), "vision tag 命中");
        assert!(!f.allows(meta("owl_dflash_conv_bf16")), "dfalsh 未开 ⇒ 拒");
    }

    #[test]
    fn wildcard_and_unknown_id() {
        let f = TagFilter::from_ids(&["*"]).expect("wildcard");
        assert!(f.is_wildcard());
        assert!(f.allows(meta("owl_gguf_gemm_q4k")));
        assert!(TagFilter::from_ids(&["nope"]).is_err());
        let f = TagFilter::from_ids(&["quant.w4a16", "kv.main.fp8"]).expect("known");
        assert_eq!(f.describe(), "quant.w4a16,kv.main.fp8");
    }
}
