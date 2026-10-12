//! 生产算子元数据表(**P3**;2026-10-13)。
//!
//! 这里住的 = **单算子的声明式契约**(`impl KernelSpec`),按 driver 的域
//! 分模块排布(`ops::attn` ↔ `driver::attn`,依此类推)。**只给生产在用的
//! 算子建表**(审查依据见 `docs/arch/kernel-准入门槛-施工设计.md` §十四):
//! 未建表的入口继续走 [`crate::list`] 登记表(名字 → 源 + 手写槽序串),
//! 保持"仅引入"状态 —— 物理上仍可发射,但没有准入声明。
//!
//! # 为什么这些声明**不在** `server/` 下(与设计稿 §2.1 的一处偏离)
//!
//! 设计稿说"算子 struct 住 `server/<family>/<op>.rs`"。实测行不通:
//! `server` 是 `feature = "device"` 门控的(`KernelFamily` 签名引用
//! `device::Exec`),而 **models 以 `features = ["marlin"]` 依赖本 crate**
//! —— 若声明住 `server/`,models 就看不到 `KernelPick` 携带的元数据,
//! "按 KernelSpec 挑选算子"在编译期就断了。
//! 因此:**声明(本模块)= 零 feature 门**;`server/native.rs` = 原生算子的
//! 服务端面(注册/装配门,device 门内)。两处引用同一份 `KernelMeta`。
//!
//! # 每个算子一个**命名 static**(单一提升点)
//!
//! 域文件里 `pub static X: KernelMeta = meta_of::<T>();` 是唯一的提升点:
//! 域表(`ops::<域>::METAS`)与族表(`KernelFamily::ops()`)都引用同一个
//! `&X` —— 若两处各自 `&meta_of::<T>()`,就会得到**两份不同的 KernelMeta**
//! (内容相同、地址不同),"元数据同一性"比较随之失效。
//!
//! # 建表纪律
//!
//! 1. 参数名/序/类型**逐字对齐 `.cu` 签名**(`tests/op_tables.rs` 机器互证,
//!    对不上就红,不靠人眼);
//! 2. 槽序串由 [`crate::spec::token_of`] 从 `PARAMS` 派生 —— 本模块**没有**
//!    任何手写的 `"T,T,O,..."` 字符串;
//! 3. 池块 dtype 用 `Dtype::U8`(e4m3/字节池)如实标注,不许拿 U32 冒充。

use crate::spec::KernelMeta;

pub mod attn;
pub mod cublas;
pub mod dflash;
pub mod elems;
pub mod flashinfer;
pub mod fused;
pub mod gdn;
pub mod gdn_chunked;
pub mod ln;
pub mod load;
pub mod marlin;
pub mod oracles;
pub mod xqa;

/// 域表清单(每项 = 一个域的全部已建表算子;顺序 = 阅读序,无功能含义)
pub static TABLES: &[&[&KernelMeta]] = &[
    attn::METAS,
    gdn::METAS,
    gdn_chunked::METAS,
    elems::METAS,
    fused::METAS,
    ln::METAS,
    dflash::METAS,
    load::METAS,
    marlin::METAS,
    cublas::METAS,
    flashinfer::METAS,
    xqa::METAS,
];

/// 全部已建表算子(迭代序 = [`TABLES`] 序)
pub fn all() -> impl Iterator<Item = &'static KernelMeta> {
    TABLES.iter().flat_map(|t| t.iter().copied())
}

/// 按名查元数据(线性扫描;测试/工具面用 —— 热路径走 `OpRegistry` 的
/// boot 期索引,别在这条路上做逐发射查找)
pub fn meta(name: &str) -> Option<&'static KernelMeta> {
    all().find(|m| m.entry == name)
}

/// 已建表算子数(P3 进度条;boot 报告与 `--print-op-table` 共用)
pub fn built_count() -> usize {
    TABLES.iter().map(|t| t.len()).sum()
}

/// **未声明算子**(R1 进度表 / 未来执法面):登记表里有、但没有 `impl KernelSpec`
/// 的 native 入口。
///
/// 为什么单列一个函数:**"引入 vs 启用"的硬界要可数**。R1 规则(用户裁决
/// 2026-10-13)= 任何算子被发射前必须有声明 —— 本函数就是"还没达标"的清单,
/// boot 报告它、测试锁它;等它归零,就把"无声明 ⇒ 拒发射"打开。
pub fn undeclared() -> Vec<&'static str> {
    crate::list::REGISTRY
        .iter()
        .filter(|e| meta(e.name).is_none())
        .map(|e| e.name)
        .collect()
}

/// 登记表条数(声明覆盖率的分母)
pub fn registered_count() -> usize {
    crate::list::REGISTRY.len()
}

// ============================================================================
// 池写集穷举审计(装配期;设计 §5.4 —— E10/E14 的机器面)
// ============================================================================

/// **池写集穷举审计**(纯函数,零 GPU —— 可单测)。
///
/// 输入 = **生效算子集**(生产档白名单过滤后的元数据表)。
///
/// 三条规则:
/// - **R1 读必有人写**:每个被读的池实例 `(名字, 布局, dtype)` 至少要有一个
///   写者 —— 这就是 E10「池 dtype 改造漏写者 ⇒ 槽区永不落笔全零」的机器面;
/// - **R2 声明自洽**:同一算子的 `ShapeFamily.kv_dtype` 与它的 `PoolAccess.dtype`
///   必须一致(同一件事写两处 ⇒ 机器互证,不许各说各话);
/// - **R3 活跃写者不歧义**:同一池实例的活跃写者布局/dtype 必须一致
///   —— 即 E14「双寻址」(两种布局同时被当成权威)。
pub fn audit_pools(metas: &[&crate::spec::KernelMeta]) -> Result<(), crate::contract::OpError> {
    use crate::contract::OpError;
    use crate::spec::PoolAccess;
    use std::collections::BTreeMap;

    // R2:算子内部两处声明互证
    for m in metas {
        if let Some(d) = m.shape.kv_dtype {
            for p in m.pool.iter().filter(|p: &&PoolAccess| p.dtype.is_some()) {
                if p.dtype != Some(d) {
                    return Err(OpError::Contract {
                        op: m.entry.to_string(),
                        field: "shape.kv_dtype vs pool.dtype",
                        expect: format!("两处声明一致({d:?})"),
                        got: format!("ShapeFamily={d:?} / PoolAccess={:?}", p.dtype),
                    });
                }
            }
        }
    }

    /// 池实例键 = (池名, 布局, dtype)
    type InstKey = (&'static str, Option<crate::spec::LayoutTag>, Option<crate::contract::Dtype>);
    /// 实例账 = (写者, 读者)
    type InstOps = (Vec<&'static str>, Vec<&'static str>);

    let mut inst: BTreeMap<InstKey, InstOps> = BTreeMap::new();
    for m in metas {
        for p in m.pool {
            let e = inst.entry((p.pool, p.layout, p.dtype)).or_default();
            if p.is_writer() {
                e.0.push(m.entry);
            }
            if p.is_reader() {
                e.1.push(m.entry);
            }
        }
    }
    // R1
    for ((pool, layout, dtype), (writers, readers)) in &inst {
        if !readers.is_empty() && writers.is_empty() {
            return Err(OpError::Contract {
                op: pool.to_string(),
                field: "pool.writers",
                expect: "读该池实例的算子 ≥1 个写者(E10:漏写者 ⇒ 槽区全零)".into(),
                got: format!("无写者;读者 = {readers:?}(布局 {layout:?} dtype {dtype:?})"),
            });
        }
    }
    // R3:同一 (池, dtype) 下活跃写者的布局必须唯一
    /// 按 (池名, dtype) 归并的写者(布局一致性检查)
    type WriterKey = (&'static str, Option<crate::contract::Dtype>);
    type WriterList = Vec<(&'static str, Option<crate::spec::LayoutTag>)>;

    let mut by_name: BTreeMap<WriterKey, WriterList> = BTreeMap::new();
    for m in metas {
        for p in m.pool.iter().filter(|p| p.is_writer()) {
            by_name.entry((p.pool, p.dtype)).or_default().push((m.entry, p.layout));
        }
    }
    for ((pool, dtype), writers) in &by_name {
        let mut layouts: Vec<Option<crate::spec::LayoutTag>> =
            writers.iter().map(|(_, l)| *l).collect();
        layouts.dedup();
        let distinct: std::collections::BTreeSet<_> =
            writers.iter().map(|(_, l)| format!("{l:?}")).collect();
        if distinct.len() > 1 {
            return Err(OpError::Contract {
                op: pool.to_string(),
                field: "pool.layout",
                expect: "同一池实例的活跃写者布局唯一(E14:双寻址)".into(),
                got: format!("{writers:?}(dtype {dtype:?})"),
            });
        }
    }
    Ok(())
}

/// 按白名单过滤后审计(`None` = 全部已建表算子)
pub fn audit_active(
    allowed: Option<&[&str]>,
) -> Result<(), crate::contract::OpError> {
    let list: Vec<&crate::spec::KernelMeta> = match allowed {
        Some(a) => all().filter(|m| a.contains(&m.entry)).collect(),
        None => all().collect(),
    };
    audit_pools(&list)
}

// ============================================================================
// 生产算子集(实测台账;`docs/arch/kernel-准入门槛-施工设计.md` §14.2)
// ============================================================================
//
// 来源 = **真跑生产档 + `[cuda] launch_time` 逐发射记账**(不是代码阅读推断),
// 2026-10-13 实测。用途三件:
//   ① 白名单硬门的权威清单(flavor 的 `dispatch.allowed_ops` 必须与它一致 ——
//      由 `apps/server` 的白名单门测试锁死,两处不许漂移);
//   ② 池写集穷举审计的输入(`audit_active`);
//   ③ P3 建表范围的依据(生产集 100% 建表 = 本表 ⊆ `all()`)。

/// `--flavor spec-dflash` 实测发射集(44 个;27B AWQ + DFlash2 d7 + fp8 主池
/// + gdn_chunked + XQA verify/decode + int4 头)
pub const PROD_SPEC_DFLASH_2026_10_13: &[&str] = &[
    "cublas_gemm_bf16",
    "cublas_gemm_f16",
    "flashinfer_prefill_paged_fp8kv",
    "gdn_chunked_delta_rule_fwd",
    "marlin_gemm_w4a16_awq",
    "marlin_gemm_w4a16_bf16",
    "owl_add_bf16",
    "owl_add_f16",
    "owl_argmax_f32idx_f16",
    "owl_cast_bf16_f16",
    "owl_cast_f16_bf16",
    "owl_concat_rows_f16",
    "owl_concat_rows_f32",
    "owl_ct_repack_u32",
    "owl_dflash_conv_bf16",
    "owl_dflash_select_bf16",
    "owl_embed_f16",
    "owl_fused_add_rmsnorm_bf16",
    "owl_fused_add_rmsnorm_f16",
    "owl_gdn_conv_fwd_f16",
    "owl_gdn_conv_upd_dual_f16",
    "owl_gdn_decode_step_v2_f16",
    "owl_gdn_gating_g_f16",
    "owl_gdn_l2norm_f16",
    "owl_gdn_norm_act_f16",
    "owl_gdn_recurrence_varlen_gqa_f16",
    "owl_gemv_dual_f16",
    "owl_naive_attn_nc_bf16_v2",
    "owl_narrow_strided_f16",
    "owl_norm_rope_bf16",
    "owl_norm_rope_f16",
    "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",
    "owl_reshape_and_cache_fp8kv_knhd",
    "owl_rmsnorm_bf16",
    "owl_rmsnorm_f16",
    "owl_sigmoid_f16",
    "owl_sigmoid_gate_mul_f16",
    "owl_silu_and_mul_bf16",
    "owl_silu_and_mul_f16",
    "owl_topk16_f16",
    "owl_xqa_decode",
    "owl_xqa_verify",
    "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256",
    "vllm_reshape_and_cache_bf16",
];

/// `--flavor awq27b` 实测发射集(27 个;裸 27B AWQ + fp8 主池,无草稿/XQA)
pub const PROD_AWQ27B_2026_10_13: &[&str] = &[
    "cublas_gemm_f16",
    "flashinfer_prefill_paged_fp8kv",
    "marlin_gemm_w4a16_awq",
    "owl_add_f16",
    "owl_argmax_f32idx_f16",
    "owl_ct_repack_u32",
    "owl_embed_f16",
    "owl_fused_add_rmsnorm_f16",
    "owl_gdn_conv_fwd_f16",
    "owl_gdn_conv_upd_dual_f16",
    "owl_gdn_decode_step_v2_f16",
    "owl_gdn_gating_g_f16",
    "owl_gdn_l2norm_f16",
    "owl_gdn_norm_act_f16",
    "owl_gdn_recurrence_varlen_gqa_f16",
    "owl_gemv_dual_f16",
    "owl_narrow_strided_f16",
    "owl_norm_rope_f16",
    "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",
    "owl_reshape_and_cache_fp8kv_knhd",
    "owl_rmsnorm_f16",
    "owl_sigmoid_f16",
    "owl_sigmoid_gate_mul_f16",
    "owl_silu_and_mul_f16",
    "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256",
    "vllm_paged_attention_v2_fp8_knhd_hd256bs32",
    "vllm_paged_attention_v2_reduce_f16_hd256",
];

/// 两档并集(46 个;= P3 建表范围)
pub const PRODUCTION_UNION_2026_10_13: &[&str] = &[
    "cublas_gemm_bf16",
    "cublas_gemm_f16",
    "flashinfer_prefill_paged_fp8kv",
    "gdn_chunked_delta_rule_fwd",
    "marlin_gemm_w4a16_awq",
    "marlin_gemm_w4a16_bf16",
    "owl_add_bf16",
    "owl_add_f16",
    "owl_argmax_f32idx_f16",
    "owl_cast_bf16_f16",
    "owl_cast_f16_bf16",
    "owl_concat_rows_f16",
    "owl_concat_rows_f32",
    "owl_ct_repack_u32",
    "owl_dflash_conv_bf16",
    "owl_dflash_select_bf16",
    "owl_embed_f16",
    "owl_fused_add_rmsnorm_bf16",
    "owl_fused_add_rmsnorm_f16",
    "owl_gdn_conv_fwd_f16",
    "owl_gdn_conv_upd_dual_f16",
    "owl_gdn_decode_step_v2_f16",
    "owl_gdn_gating_g_f16",
    "owl_gdn_l2norm_f16",
    "owl_gdn_norm_act_f16",
    "owl_gdn_recurrence_varlen_gqa_f16",
    "owl_gemv_dual_f16",
    "owl_naive_attn_nc_bf16_v2",
    "owl_narrow_strided_f16",
    "owl_norm_rope_bf16",
    "owl_norm_rope_f16",
    "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",
    "owl_reshape_and_cache_fp8kv_knhd",
    "owl_rmsnorm_bf16",
    "owl_rmsnorm_f16",
    "owl_sigmoid_f16",
    "owl_sigmoid_gate_mul_f16",
    "owl_silu_and_mul_bf16",
    "owl_silu_and_mul_f16",
    "owl_topk16_f16",
    "owl_xqa_decode",
    "owl_xqa_verify",
    "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256",
    "vllm_paged_attention_v2_fp8_knhd_hd256bs32",
    "vllm_paged_attention_v2_reduce_f16_hd256",
    "vllm_reshape_and_cache_bf16",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{meta_of, pools, KernelSpec, LayoutTag, Param, PoolAccess};

    /// **生产集 100% 建表**(P3 的覆盖声明 → 机器锁):台账里每一个算子
    /// 都必须有 `impl KernelSpec` —— 否则"生产集全覆盖"只是文档里的一句话。
    #[test]
    fn production_set_is_fully_built() {
        let missing: Vec<&str> = PRODUCTION_UNION_2026_10_13
            .iter()
            .copied()
            .filter(|n| meta(n).is_none())
            .collect();
        assert!(missing.is_empty(), "生产集未建表:{missing:?}");
        // 反向:建表集必须覆盖两档清单
        for (flavor, list) in [
            ("spec-dflash", PROD_SPEC_DFLASH_2026_10_13),
            ("awq27b", PROD_AWQ27B_2026_10_13),
        ] {
            for n in list {
                assert!(meta(n).is_some(), "{flavor}: {n} 未建表");
            }
        }
        eprintln!(
            "[ops] 生产集 {} 个算子全部已建表(共 {} 张表)✓",
            PRODUCTION_UNION_2026_10_13.len(),
            built_count()
        );
    }

    /// **生产集审计必须全绿**(§14.2 实测清单:spec-dflash 44 + awq27b 27 的并集)
    #[test]
    fn audit_passes_on_production_set() {
        const PROD: &[&str] = &[
            "marlin_gemm_w4a16_awq", "marlin_gemm_w4a16_bf16", "owl_gdn_conv_fwd_f16",
            "owl_narrow_strided_f16", "owl_gdn_l2norm_f16", "owl_rmsnorm_f16",
            "owl_silu_and_mul_f16", "owl_fused_add_rmsnorm_f16", "owl_add_f16",
            "cublas_gemm_f16", "owl_gdn_recurrence_varlen_gqa_f16", "owl_ct_repack_u32",
            "owl_sigmoid_f16", "owl_gdn_norm_act_f16", "owl_gdn_gating_g_f16",
            "gdn_chunked_delta_rule_fwd", "owl_norm_rope_f16", "owl_gemv_dual_f16",
            "owl_sigmoid_gate_mul_f16", "owl_reshape_and_cache_fp8kv_knhd",
            "flashinfer_prefill_paged_fp8kv", "owl_norm_rope_bf16", "owl_gdn_decode_step_v2_f16",
            "owl_gdn_conv_upd_dual_f16", "owl_dflash_conv_bf16", "owl_cast_f16_bf16",
            "vllm_reshape_and_cache_bf16", "owl_add_bf16", "cublas_gemm_bf16",
            "owl_fused_add_rmsnorm_bf16", "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256",
            "owl_xqa_verify", "owl_xqa_decode", "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",
            "owl_argmax_f32idx_f16", "owl_silu_and_mul_bf16", "owl_naive_attn_nc_bf16_v2",
            "owl_embed_f16", "owl_rmsnorm_bf16", "owl_concat_rows_f16", "owl_topk16_f16",
            "owl_dflash_select_bf16", "owl_cast_bf16_f16", "owl_concat_rows_f32",
            "vllm_paged_attention_v2_fp8_knhd_hd256bs32", "vllm_paged_attention_v2_reduce_f16_hd256",
        ];
        audit_active(Some(PROD)).expect("生产集池审计必须通过");
        // 另一套**自洽**配置:classic 池档(主池 classic + 草稿池 classic)
        const CLASSIC_SET: &[&str] = &[
            "owl_reshape_and_cache_fp8kv",
            "owl_qknorm_rope_kv_insert_f16_fp8kv",
            "vllm_reshape_and_cache_bf16",
            "owl_naive_attn_nc_bf16_v2",
        ];
        audit_active(Some(CLASSIC_SET)).expect("classic 池档必须通过");
        // fp8 草稿池档(前缀池 e4m3)
        const FP8_DRAFT: &[&str] = &["owl_reshape_and_cache_fp8kv", "owl_naive_attn_nc_fp8kv_bf16"];
        audit_active(Some(FP8_DRAFT)).expect("fp8 草稿池档必须通过");
        // ⚠️ **knHD 与 classic 混在同一白名单 = 歧义**(池布局是配置选择,
        // 两种同时"活跃"就是 E14 双寻址的温床)⇒ 必须拒
        let mixed: Vec<&str> = PROD.iter().chain(["owl_reshape_and_cache_fp8kv"].iter()).copied().collect();
        let e = audit_active(Some(&mixed)).unwrap_err();
        assert!(e.to_string().contains("布局唯一"), "{e}");
    }

    // ---- 负例:三条规则各一 ----
    struct ReadOnly;
    impl KernelSpec for ReadOnly {
        const ENTRY: &'static str = "fake_reader";
        const SOURCE: &'static str = "";
        const PARAMS: &'static [Param] = &[];
        const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
            kv_dtype: Some(crate::contract::Dtype::U8),
            head_dim: &[],
            page: &[],
            grp: None,
            tile: None,
        };
        const POOL: &'static [PoolAccess] = &[PoolAccess::read(
            pools::KV_MAIN,
            LayoutTag::Knhd,
            crate::contract::Dtype::U8,
        )];
    }
    struct Mismatched;
    impl KernelSpec for Mismatched {
        const ENTRY: &'static str = "fake_mismatch";
        const SOURCE: &'static str = "";
        const PARAMS: &'static [Param] = &[];
        const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
            kv_dtype: Some(crate::contract::Dtype::U8),
            head_dim: &[],
            page: &[],
            grp: None,
            tile: None,
        };
        // 声明矛盾:ShapeFamily 说 U8,池声明说 F16
        const POOL: &'static [PoolAccess] = &[PoolAccess::write(
            pools::KV_MAIN,
            LayoutTag::Knhd,
            crate::contract::Dtype::F16,
        )];
    }
    struct WriteKnhd;
    impl KernelSpec for WriteKnhd {
        const ENTRY: &'static str = "fake_writer_knhd";
        const SOURCE: &'static str = "";
        const PARAMS: &'static [Param] = &[];
        const POOL: &'static [PoolAccess] = &[PoolAccess::write(
            pools::KV_MAIN,
            LayoutTag::Knhd,
            crate::contract::Dtype::U8,
        )];
    }
    struct WriteClassic;
    impl KernelSpec for WriteClassic {
        const ENTRY: &'static str = "fake_writer_classic";
        const SOURCE: &'static str = "";
        const PARAMS: &'static [Param] = &[];
        const POOL: &'static [PoolAccess] = &[PoolAccess::write(
            pools::KV_MAIN,
            LayoutTag::Classic,
            crate::contract::Dtype::U8,
        )];
    }

    /// R1:读者无写者 ⇒ 拒(E10 类)
    #[test]
    fn audit_rejects_reader_without_writer() {
        let m = [&meta_of::<ReadOnly>()];
        let e = audit_pools(&m).unwrap_err();
        let s = e.to_string();
        assert!(s.contains("kv.main") && s.contains("无写者"), "{s}");
    }

    /// R2:算子内部两处声明不一致 ⇒ 拒
    #[test]
    fn audit_rejects_self_inconsistent_declaration() {
        let m = [&meta_of::<Mismatched>()];
        let e = audit_pools(&m).unwrap_err();
        assert!(e.to_string().contains("两处声明一致"), "{e}");
    }

    /// R3:同一池实例两个活跃写者布局不同 ⇒ 拒(E14 双寻址)
    #[test]
    fn audit_rejects_ambiguous_writers() {
        let m = [&meta_of::<WriteKnhd>(), &meta_of::<WriteClassic>()];
        let e = audit_pools(&m).unwrap_err();
        assert!(e.to_string().contains("布局唯一"), "{e}");
    }

    /// 命名 static 是**单一提升点**:域表里的每一项都必须指向 `all()` 里
    /// 同一个 `KernelMeta`(地址相等)—— 否则"同一算子两份元数据"这个
    /// 隐蔽的漂移源就复活了。
    #[test]
    fn domain_tables_share_one_promotion_point() {
        for t in TABLES {
            for m in t.iter() {
                let found = meta(m.entry).expect("域表项不在 all() 里");
                assert!(
                    std::ptr::eq(*m, found),
                    "{}: 域表与 all() 指向不同 KernelMeta(重复提升)",
                    m.entry
                );
            }
        }
    }
}
