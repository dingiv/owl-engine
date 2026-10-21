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

use crate::spec::{KernelMeta, KernelTag, LayoutTag, EXCLUSIVE_TAG_FACETS, KERNEL_TAGS};

/// 配置派生过滤器（Copy；随 `EnvProvider`/`DiagOpts` 值传递）
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TagFilter {
    allowed: u64,
    wildcard: bool,
    /// 主池布局门（`None` = 不约束；生产档 = [`PROD_MAIN_KV_LAYOUT`]）
    main_layout: Option<LayoutTag>,
    /// 草稿池布局门（`None` = 不约束；生产档 = [`PROD_DRAFT_KV_LAYOUT`]）
    draft_layout: Option<LayoutTag>,
}

/// **`Default` = 全放行**（诊断/测试兜底）。
///
/// ⚠️ 2026-10-21 抓到的**既有红**:原为 `#[derive(Default)]` ⇒
/// `allowed = 0 ∧ wildcard = false` = **deny-all 互斥 facet** ——
/// `EngineKnobs::default().cuda`(全部引擎 e2e/单测档)因此**第一发 pool 写核
/// 就被拒**(实测 `vllm_reshape_and_cache_f16_knhd` / facet `kv.main`)。
/// 本文件头注写的缺省语义就是"`*` / Default = 全放行" —— 实现现在与文档一致。
impl Default for TagFilter {
    fn default() -> Self {
        Self::wildcard()
    }
}

/// **生产档主池布局**：P6-A 统一契约后主池只有 kNHD 寻址。
/// 声明面 `PoolAccess.layout` 是唯一真源；此常量只表达“本档选哪一套”。
pub const PROD_MAIN_KV_LAYOUT: LayoutTag = LayoutTag::Knhd;
/// **生产档草稿池布局**：DFlash2 自块直读仍走 classic 岛。
/// 注意：`*_knhd` NC 变体是主池 KnHD 的零调用点声明，不是草稿池。
pub const PROD_DRAFT_KV_LAYOUT: LayoutTag = LayoutTag::Classic;

impl TagFilter {
    /// 全放行（诊断/测试兜底；不参与生产配置派生）
    pub const fn wildcard() -> Self {
        Self {
            allowed: 0,
            wildcard: true,
            main_layout: None,
            draft_layout: None,
        }
    }

    /// 允许集构造（空 = 只放行“无互斥 facet tag”的通用算子）
    pub fn from_tags(tags: &[KernelTag]) -> Self {
        let mut allowed = 0u64;
        for t in tags {
            allowed |= t.bit();
        }
        Self {
            allowed,
            wildcard: false,
            main_layout: None,
            draft_layout: None,
        }
    }

    /// **生产档布局门**：主池 kNHD / 草稿池 classic。
    ///
    /// tag 只区分 `kv.main.fp8` / `kv.draft.bf16` 这类“池实例 dtype”；
    /// 同一个池实例还允许 classic/kNHD 两套声明并存（历史布局翻转面）。
    /// 若不过滤布局，两套写者会同时活跃 ⇒ E14 双寻址。布局的唯一真源是
    /// `PoolAccess.layout`，所以这里直接对声明面投影，不再手抄核名。
    pub const fn with_prod_kv_layouts(self) -> Self {
        self.with_kv_layouts(PROD_MAIN_KV_LAYOUT, PROD_DRAFT_KV_LAYOUT)
    }

    /// 显式布局门（测试/未来布局档用）
    pub const fn with_kv_layouts(mut self, main: LayoutTag, draft: LayoutTag) -> Self {
        self.main_layout = Some(main);
        self.draft_layout = Some(draft);
        self
    }

    /// 单个池访问是否满足布局门
    fn layout_denied(&self, pool: &'static str, layout: Option<LayoutTag>) -> bool {
        let allowed = match pool {
            crate::spec::pools::KV_MAIN => self.main_layout,
            crate::spec::pools::KV_DRAFT => self.draft_layout,
            _ => None,
        };
        match (allowed, layout) {
            (Some(want), Some(got)) => want != got,
            _ => false,
        }
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
        Ok(Self {
            allowed,
            wildcard: false,
            main_layout: None,
            draft_layout: None,
        })
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
        // 布局门先判：同一 dtype 的 classic/kNHD 两套声明只允许一套活跃。
        for p in meta.pool {
            if self.layout_denied(p.pool, p.layout) {
                return Some("pool.layout");
            }
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
    fn w4a8_main_plus_int4_head_allows_both_weight_facets() {
        // 头/草稿是**独立 plan**:主干 W4A8 与 aux W4A16 必须并集;
        // 同 facet 内每个算子仍只带一个 tag,故不会产生双路由。
        let f = TagFilter::from_tags(&[
            KernelTag::QUANT_W4A8,
            KernelTag::QUANT_AWQ,
            KernelTag::QUANT_W4A16,
        ]);
        assert!(f.allows(meta("marlin_gemm_w4a8_awq")));
        assert!(f.allows(meta("marlin_gemm_w4a16_awq")));
        assert!(f.allows(meta("marlin_gemm_w4a16_bf16")));
        assert!(!f.allows(meta("owl_gguf_gemm_q4k")));
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
    fn knhd_nc_variants_are_main_pool_readers() {
        // `*_knhd` NC 变体与主池 KnHD 写核配对；草稿池是 classic 布局。
        // 2026-10-21 262k-vlm 启动门抓到的漏配：这组曾被误打 kv.draft.*，
        // 导致装配门看到“草稿 KnHD/BF16 只有读者没有写者”而拒启。
        let main_bf16 = TagFilter::from_tags(&[
            KernelTag::KV_MAIN_BF16,
            KernelTag::FEAT_DFLASH,
            KernelTag::FEAT_SPEC,
        ]);
        assert!(main_bf16.allows(meta("owl_naive_attn_nc_bf16_knhd")));
        assert!(main_bf16.allows(meta("vllm_reshape_and_cache_bf16_knhd")));

        let draft_bf16 = TagFilter::from_tags(&[
            KernelTag::KV_DRAFT_BF16,
            KernelTag::FEAT_DFLASH,
            KernelTag::FEAT_SPEC,
        ]);
        assert!(!draft_bf16.allows(meta("owl_naive_attn_nc_bf16_knhd")));
        assert!(draft_bf16.allows(meta("vllm_reshape_and_cache_bf16")));
    }

    /// 装配门的不变量（纯 host 版）：每个活跃池实例的读者，必须至少有一个
    /// 活跃写者，且 `(pool, layout, dtype)` 三元组完全一致。boot 门在 GPU
    /// 装配前做同一件事；本测试让“tag 写错导致幽灵读者”在单测阶段就红。
    #[test]
    fn every_active_pool_reader_has_a_same_instance_writer() {
        // 262k-vlm 档的 tag 组合（配置唯一真源在 profile，测试只需覆盖语义）
        let f = TagFilter::from_tags(&[
            KernelTag::QUANT_W4A16,
            KernelTag::QUANT_AWQ,
            KernelTag::QUANT_INT8,
            KernelTag::FEAT_HEAD,
            KernelTag::KV_MAIN_FP8,
            KernelTag::FEAT_DFLASH,
            KernelTag::FEAT_SPEC,
            KernelTag::KV_DRAFT_BF16,
            KernelTag::FEAT_VISION,
            KernelTag::FEAT_TP,
        ]);
        let f = f.with_prod_kv_layouts();
        let active: Vec<&KernelMeta> =
            crate::ops::all().filter(|m| f.allows(m)).collect();
        let mut missing = Vec::new();
        for reader in &active {
            for rp in reader.pool.iter().filter(|p| p.is_reader()) {
                let has_writer = active.iter().any(|w| {
                    w.pool.iter().any(|wp| {
                        wp.is_writer()
                            && wp.pool == rp.pool
                            && wp.layout == rp.layout
                            && wp.dtype == rp.dtype
                    })
                });
                if !has_writer {
                    missing.push(format!(
                        "{} 读 {}/{:?}/{:?} 无写者",
                        reader.entry, rp.pool, rp.layout, rp.dtype
                    ));
                }
            }
        }
        assert!(missing.is_empty(), "活跃池读者缺写者: {missing:#?}");

        // E14：同一 (池, dtype) 的活跃写者布局必须唯一。
        let mut writers: std::collections::BTreeMap<
            (&'static str, Option<crate::contract::Dtype>),
            std::collections::BTreeSet<Option<LayoutTag>>,
        > = Default::default();
        for m in &active {
            for p in m.pool.iter().filter(|p| p.is_writer()) {
                writers.entry((p.pool, p.dtype)).or_default().insert(p.layout);
            }
        }
        let ambiguous: Vec<_> = writers
            .into_iter()
            .filter(|(_, layouts)| layouts.len() > 1)
            .collect();
        assert!(ambiguous.is_empty(), "活跃写者双寻址: {ambiguous:#?}");
    }

    #[test]
    fn draft_fp8_writers_are_draft_pool_only() {
        // draft_kv_fp8 档：classic 写核（K0WriteFp8/f16 输入与 bf16 输入）
        // 只写草稿池；主池仍走 KnHD。布局门必须让它们存活，同时不能把
        // 它们误当主池写者（E14）。
        let f = TagFilter::from_tags(&[
            KernelTag::QUANT_W4A16,
            KernelTag::QUANT_AWQ,
            KernelTag::KV_MAIN_FP8,
            KernelTag::KV_DRAFT_FP8,
            KernelTag::FEAT_DFLASH,
            KernelTag::FEAT_SPEC,
        ])
        .with_prod_kv_layouts();
        for entry in [
            "owl_reshape_and_cache_fp8kv",
            "owl_reshape_and_cache_fp8kv_bf16",
            "owl_naive_attn_nc_fp8kv_bf16",
        ] {
            assert!(f.allows(meta(entry)), "草稿 fp8 核应放行: {entry}");
        }
        assert!(f.allows(meta("owl_reshape_and_cache_fp8kv_bf16")), "草稿 classic bf16 写者应放行");

        let active: Vec<&KernelMeta> =
            crate::ops::all().filter(|m| f.allows(m)).collect();
        let mut missing = Vec::new();
        for reader in &active {
            for rp in reader.pool.iter().filter(|p| p.is_reader()) {
                let has_writer = active.iter().any(|w| {
                    w.pool.iter().any(|wp| {
                        wp.is_writer()
                            && wp.pool == rp.pool
                            && wp.layout == rp.layout
                            && wp.dtype == rp.dtype
                    })
                });
                if !has_writer {
                    missing.push(format!("{} 读 {:?}/{:?}", reader.entry, rp.pool, rp.dtype));
                }
            }
        }
        assert!(missing.is_empty(), "draft-fp8 活跃读者缺写者: {missing:#?}");
    }

    #[test]
    fn production_layout_gate_rejects_classic_main_writer() {
        let f = TagFilter::from_tags(&[
            KernelTag::KV_MAIN_FP8,
            KernelTag::KV_DRAFT_FP8,
            KernelTag::KV_DRAFT_BF16,
            KernelTag::FEAT_DFLASH,
            KernelTag::FEAT_SPEC,
        ])
        .with_prod_kv_layouts();
        assert!(f.allows(meta("owl_reshape_and_cache_fp8kv_knhd")));
        assert!(!f.allows(meta("owl_qknorm_rope_kv_insert_f16_fp8kv")), "主池 classic 应拒");
        assert!(f.allows(meta("owl_reshape_and_cache_fp8kv_bf16_knhd")), "主池 KnHD 应放行");
        assert!(f.allows(meta("owl_reshape_and_cache_fp8kv_bf16")), "草稿 classic 应放行");
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
