//! 生产算子元数据表(**P3**;2026-10-13)。
//!
//! 这里住的 = **单算子的声明式契约**(`impl KernelSpec`),按 driver 的域
//! 分模块排布(`ops::attn` ↔ `driver::attn`,依此类推)。**只给生产在用的
//! 算子建表**(审查依据见 `docs/arch/kernel-准入门槛-施工设计.md` §十四):
//! **未建表的入口 = "仅引入"**(登记表 F 轮已退役):物理上仍可发射,但没有
//! 准入声明 ⇒ 名字路由/白名单/池审计都看不到它(boot 报告会列"编出但未声明")。
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
//! build.rs 扫描这些 static 生成总表(`ops_table.rs`),族表
//! (`KernelFamily::ops()`)也引用同一个 `&X` —— 若两处各自
//! `&meta_of::<T>()`,就会得到**两份不同的 KernelMeta**(内容相同、地址不同),
//! "元数据同一性"比较随之失效。
//!
//! # 建表纪律
//!
//! 1. 参数名/序/类型**逐字对齐 `.cu` 签名**(`tests/op_tables.rs` 机器互证,
//!    对不上就红,不靠人眼);
//! 2. 槽序串由 [`crate::spec::token_of`] 从 `PARAMS` 派生 —— 本模块**没有**
//!    任何手写的 `"T,T,O,..."` 字符串;
//! 3. 池块 dtype **点名格式**:fp8 池 = `Dtype::F8E4M3`、int8 激活 = `Dtype::I8`
//!    (**不许**再用 `U8` 兜 —— `U8` 只留真正不透明的字节块;2026-10-16 收窄)。

use crate::spec::KernelMeta;

pub mod attn;
pub mod cublas;
pub mod dflash;
pub mod elems;
pub mod flashinfer;
pub mod fused;
pub mod gdn;
/// **GGUF 族算子声明**(2026-10-19;逐型一条 = H3 的"不许裸 int 路由")
pub mod gguf;
pub mod gdn_chunked;
pub mod gdn_scalar;
pub mod comm;
pub mod ln;
pub mod load;
pub mod marlin;
pub mod oracles;
pub mod vis;
pub mod xqa;

// 算子总表由构建期扫描 `pub static ... KernelMeta = meta_of::<T>();`
// 生成(build.rs;2026-10-20)。手写 `TABLES` / 各域 `METAS` 已退役 ——
// 宏生成的 impl 也在 static 提升点留下唯一一行,不会漏算子。
include!(concat!(env!("OUT_DIR"), "/ops_table.rs"));

/// 全部已建表算子(迭代序 = [`TABLES`] 序)
pub fn all() -> impl Iterator<Item = &'static KernelMeta> {
    TABLES.iter().flat_map(|t| t.iter().copied())
}

/// **按家族过滤已建表算子**(族 `ops()` 的派生源 —— 声明驱动路由,2026-10-15)。
///
/// 族侧不再手写 `const OPS: &[...]`:`KernelFamily::ops()` 的缺省实现就是
/// `of_family(self.id())` —— 单一真源 = 每个算子的 `KernelSpec::FAMILY`。
pub fn of_family(id: crate::contract::OpId) -> impl Iterator<Item = &'static KernelMeta> {
    all().filter(move |m| m.family == id)
}

/// 按名查元数据(线性扫描;测试/工具面用 —— 热路径走 `OpRegistry` 的
/// boot 期索引,别在这条路上做逐发射查找)
pub fn meta(name: &str) -> Option<&'static KernelMeta> {
    all().find(|m| m.entry == name)
}

/// **按入口名取 `.cu` 源码**(自托管的唯一读取口;探针 / 服务端装配面用)。
///
/// 解析序(**单一实现**住本函数;旧 `list::source` 兼容面已随登记表退役):
/// 1. **已建表** ⇒ `KernelMeta::SOURCE`(就地 `include_str!`,R5 —— 算子
///    自托管面是它的唯一住址);
///
/// **F 轮(2026-10-13)后无兜底**:登记表已退役,声明面是唯一真源 ⇒
/// 名字查不到 = 未准入 = 编程错误 —— 拼错名或漏建表,直接 panic 点名。
pub fn source(name: &str) -> &'static str {
    match meta(name) {
        Some(m) => m.src,
        None => panic!(
            "ops::source({name:?}): 无该算子的准入声明 —— 新核须先在 `cu/` 加源 \
             并在 `ops/<域>.rs` 建表(登记表已退役,见设计 §15.15)"
        ),
    }
}

/// 已建表算子数(P3 进度条;boot 报告与 `--print-op-table` 共用)
pub fn built_count() -> usize {
    TABLES.iter().map(|t| t.len()).sum()
}

/// **编出来但没有声明的核符号**(R1 判据,F 轮换源)。
///
/// 旧判据 = 登记表差额(110 − 已建表);F 轮登记表退役后换成
/// **构建期实际编出的符号集 ⊆ 声明集** —— 更贴 R1 本意:凡编出来的核都必须
/// 有 `impl KernelSpec`。**新增 `.cu` 而忘了建表**会被本函数当场抓住
/// (旧判据抓不到:它要求人先去登记表登记一条)。
/// **家族私有核**(编出来但**不走算子准入面**的辅助核:家族装配期按名装载,
/// 不经 `KernelSpec` 的发射路由)。R1 符号门据此豁免 —— **每条必须写理由**,
/// 且门会校验它确实在符号清单里(登错名 = 红)。
///
/// 与"算子"的界:算子有准入声明(参数表/池/律)且经 registry 路由;私有核
/// 只是某个家族自己的编排零件,消费者是那个家族的 `assemble`。
pub const FAMILY_PRIVATE: &[(&str, &str)] = &[
    // §15.7:gdn_scalar 家族的单核(FLA chunk_scalar;exec.load_aot 按名装载)
    ("cu/gdn/gdn_chunk_scalar.cu", "owl_gdn_chunk_scalar_f32"),
    // gdn_chunked 装配期的状态布局转置 [HV,K,V] ↔ [HV,V,K]
    ("cu/gdn/state_transpose.cu", "owl_state_kv_to_vk"),
    ("cu/gdn/state_transpose.cu", "owl_state_vk_to_kv"),
];

pub fn undeclared_symbols() -> Vec<(&'static str, &'static str)> {
    // 防呆:册上每条都必须在符号清单里(改名/删核后忘了清册 = 红)
    for (id, sym) in FAMILY_PRIVATE {
        assert!(
            crate::cu_sig::ASSET_SYMBOLS.iter().any(|(i, s)| i == id && s.contains(sym)),
            "FAMILY_PRIVATE 登记了不存在的符号:{id} :: {sym}(改名/删核后请同步清册)"
        );
    }
    undeclared_symbols_raw()
}

fn undeclared_symbols_raw() -> Vec<(&'static str, &'static str)> {
    crate::cu_sig::ASSET_SYMBOLS
        .iter()
        .flat_map(|(id, syms)| syms.iter().map(move |s| (*id, *s)))
        .filter(|(_, s)| meta(s).is_none())
        .filter(|(id, s)| !FAMILY_PRIVATE.iter().any(|(i, x)| i == id && x == s))
        .collect()
}

/// 声明表条数(唯一真源;旧"登记表条数"已退役)
pub fn declared_count() -> usize {
    TABLES.iter().map(|t| t.len()).sum()
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
// 生产算子集(历史实测台账;`docs/arch/kernel-准入门槛-施工设计.md` §14.2)
// ============================================================================
//
// **2026-10-20：运行时白名单已退役**——这些常量只保留为历史审计/回归证据；
// 生产准入由配置派生的 `TagFilter` 完成，profile 不再手抄核名。
//
// 来源 = **真跑生产档 + `[cuda] launch_time` 逐发射记账**(不是代码阅读推断),
// 2026-10-13 实测。用途三件:
//   ① 白名单硬门的权威清单(flavor 的 `dispatch.allowed_ops` 必须与它一致 ——
//      由 `apps/server` 的白名单门测试锁死,两处不许漂移);
//   ② 池写集穷举审计的输入(`audit_active`);
//   ③ P3 建表范围的依据(生产集 100% 建表 = 本表 ⊆ `all()`)。

/// `-p profiles/legacy/spec-dflash.toml` 实测发射集(**49 个**;27B AWQ + DFlash2 d7 + fp8 主池
/// + gdn_chunked + XQA verify/decode + int4 头;A5 后含 f16 主池族 4 项)
///
/// **2026-10-14 A5 补丁**:追加 **f16 主池族**(4 个)。原先本集只锁 fp8 池
/// (“XQA 前置 = fp8”),导致 `[dispatch] kv_fp8 = false` 的 f16 池档在
/// 白名单硬门处撞结构化拒(§7.6 二分实验只能拿 `allowed_ops = []` 绕)。
/// 现在两池族**共存于同一白名单**:`kv_fp8` 决定实际发射哪一族,
/// 池写集审计按 (池名, dtype) 分组 ⇒ 两族各成闭环、互不歧义(R3)。
pub const PROD_SPEC_DFLASH_2026_10_13: &[&str] = &[
    "cublas_gemm_bf16",
    "cublas_gemm_f16",
    "flashinfer_prefill_paged_f16", "flashinfer_prefill_paged_fp8kv",
    "gdn_chunked_delta_rule_fwd",
    "marlin_gemm_w4a16_awq",
    "marlin_gemm_w4a16_bf16",
    "owl_add_bf16",
    "owl_add_f16",
    "owl_argmax_f32idx_f16",
    "marlin_gemm_w4a8_awq",
    "owl_quant_int8_row_f16",
    "owl_ct_repack_a8_u32",
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
    "owl_gemv_i8w_f16",
    "owl_naive_attn_nc_bf16_v2",
    "owl_narrow_strided_f16",
    "owl_norm_rope_bf16",
    "owl_norm_rope_f16",
    "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",
    // ── f16 主池族(A5;`kv_fp8 = false` 时实际发射)──
    "owl_qknorm_rope_kv_insert_f16_knhd",
    "owl_reshape_and_cache_fp8kv_knhd",
    "owl_rmsnorm_bf16",
    "owl_rmsnorm_f16",
    "owl_sigmoid_f16",
    "owl_sigmoid_gate_mul_f16",
    "owl_silu_and_mul_bf16",
    "owl_silu_and_mul_f16",
    "owl_topk16_merge_f16",
    "owl_topk16_split_f16",
    "owl_xqa_decode",
    // ⚠️ 在白名单但**档位缺省不启用**(`dispatch.xqa_prefill=false`):
    // CUPTI 同形状 A/B = XQA 258.5ms vs FI 63.0ms(**慢 4.1×**,主路径禁用);
    // 它只接 FI 不覆盖的 t<128 尾块(那里 3.9× 于 chunked knhd 回落核)。
    // 留白名单 = 复验只改一行 config。
    "owl_xqa_prefill",
    "owl_xqa_verify",
    "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd256",
    "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256",
    "vllm_paged_attention_v2_f16_knhd_hd256bs32",
    // XQA 关臂(A9:decode_xqa/verify_xqa=false)所需的 v2-fp8 decode + reduce
    // (与 awq27b 集同项;不列 = XQA 关档在白名单硬门处结构化拒,实测踩过)
    "vllm_paged_attention_v2_fp8_knhd_hd256bs32",
    "vllm_paged_attention_v2_reduce_f16_hd256",
    "vllm_reshape_and_cache_bf16",
    "vllm_reshape_and_cache_f16_knhd",
];

/// `-p profiles/legacy/gguf4b.toml` 实测发射集(30 个;GGUF 4B Q4_K_M 单卡,无投机/无 XQA)
///
/// **来源 = 真实发射账**(不是手抄):`[cuda] trace_ring = 40000` 起机 →
/// 依次打 11 / ~512 / ~1024 / ~2048 / ~4096 / ~8192 token 的请求 →
/// 每轮 `GET /debug/launches/40000` 取 `cmd=="launch"` 的 `name` **取并集**
/// (⚠️ ring 会绕圈:只 dump 一次实测漏 5 个解码臂)→ 去重排序。
///
/// **判据**(见 `apps/server/tests/ops_whitelist.rs`):① 与 flavor 的
/// `dispatch.allowed_ops` 逐项一致;② 每项已建表;③ 过**池写集穷举审计**。
/// 换文件(不同量化型混布)/换臂(开 spec、改 kv_fp8)⇒ 重跑上面的收账流程。
pub const PROD_GGUF4B_2026_10_19: &[&str] = &[
    "cublas_gemm_f16",
    "flashinfer_prefill_paged_f16",
    "owl_add_f16",
    "owl_argmax_f32idx_f16",
    "owl_fused_add_rmsnorm_f16",
    "owl_gdn_conv_fwd_f16",
    "owl_gdn_conv_upd_dual_f16",
    "owl_gdn_decode_step_v2_f16",
    "owl_gdn_gating_g_f16",
    "owl_gdn_l2norm_f16",
    "owl_gdn_norm_act_f16",
    "owl_gdn_recurrence_varlen_gqa_f16",
    "owl_gemv_dual_f16",
    "owl_gguf_dequant_rows_q6k",
    "owl_gguf_gemm_q4k",
    "owl_gguf_gemm_q5k",
    "owl_gguf_gemm_q6k",
    "owl_gguf_gemm_q8_0",
    // **prefill 复用臂**(W-2,2026-10-19):`m >= 8` 的 GEMM 走这里
    // (与 MMVQ 位级相同,只是调度不同)
    "owl_gguf_gemm_tile_q4k",
    "owl_gguf_gemm_tile_q5k",
    "owl_gguf_gemm_tile_q6k",
    "owl_gguf_gemm_tile_q8_0",
    // **smem 复用臂**(W-2b):`m >= 64` 的 GEMM 走这里(权重窗解量化进 smem)
    "owl_gguf_gemm_smem_q4k",
    "owl_gguf_gemm_smem_q5k",
    "owl_gguf_gemm_smem_q6k",
    "owl_gguf_gemm_smem_q8_0",
    // **官方 MMQ**(W-2c;`m >= 32` 的 GEMM 走这里)
    "owl_gguf_mmq_q4k",
    "owl_gguf_mmq_q5k",
    "owl_gguf_mmq_q6k",
    "owl_gguf_mmq_q8_0",
    "owl_gguf_quant_act_f16",
    "owl_gguf_quant_act_f16_tiled",
    "owl_narrow_strided_f16",
    "owl_norm_rope_f16",
    "owl_qknorm_rope_kv_insert_f16_knhd",
    "owl_rmsnorm_f16",
    "owl_sigmoid_f16",
    "owl_sigmoid_gate_mul_f16",
    "owl_silu_and_mul_f16",
    "vllm_paged_attention_v2_f16_knhd_hd256bs32",
    "vllm_paged_attention_v2_reduce_f16_hd256",
    "vllm_reshape_and_cache_f16_knhd",
];

/// `-p profiles/legacy/awq27b.toml` 实测发射集(27 个;裸 27B AWQ + fp8 主池,无草稿/XQA)
pub const PROD_AWQ27B_2026_10_13: &[&str] = &[
    "cublas_gemm_f16",
    "flashinfer_prefill_paged_fp8kv",
    // 2026-10-20:`gdn_chunked` 定谳后转缺省 true(prefill 大 T 五核);
    // 档位不显式关就必须列它(否则首次 prefill 撞准入门)
    "gdn_chunked_delta_rule_fwd",
    "marlin_gemm_w4a16_awq",
    // W4A8 档(S2;env `OWL_W4A8=1` 才实际发射 —— 白名单超集无害)
    "marlin_gemm_w4a8_awq",
    "owl_quant_int8_row_f16",
    "owl_ct_repack_a8_u32",
    "owl_add_f16",
    "owl_argmax_f32idx_f16",
    // **装载域**(2026-10-16 补):checkpoint 的 `embed_tokens`/`lm_head` 是 BF16,
    // 引擎态是 F16 ⇒ 装载期走**复合命令**(`UploadAndLaunch`,段复合)里的
    // bf16→f16 铸边界。**本表重启时缺它**,而 `launch_time` 记账只覆盖
    // `handle_launch`(复合口不进那条路径)⇒ "真跑 + 逐发射记账"的生成方法
    // **系统性漏装载域算子**(现场:`-p profiles/legacy/awq27b.toml` + 草稿档启动即被准入拒:
    // `Weight 'model.language_model.embed_tokens.weight': 段复合 contract
    // owl_cast_bf16_f16 … 未列入`)。
    "owl_cast_bf16_f16",
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
    "owl_gemv_i8w_f16",
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
    "flashinfer_prefill_paged_f16", "flashinfer_prefill_paged_fp8kv",
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
    "owl_gemv_i8w_f16",
    "owl_naive_attn_nc_bf16_v2",
    "owl_narrow_strided_f16",
    "owl_norm_rope_bf16",
    "owl_norm_rope_f16",
    "owl_qknorm_rope_kv_insert_f16_fp8kv_knhd",
    // ── f16 主池族(A5/F5d;`kv_fp8 = false` 时实际发射)——
    // 2026-10-15:并集曾漏这 6 项(且多一个无主的 `owl_topk16_f16`)
    // ⇒ 新增机器门 `production_union_equals_flavor_union`:并集与
    // 两档清单的**集合相等**从此是红线(手抄清单不再靠注释自律)
    "owl_qknorm_rope_kv_insert_f16_knhd",
    "owl_quant_int8_row_f16",
    "owl_ct_repack_a8_u32",
    "marlin_gemm_w4a8_awq",
    "owl_topk16_merge_f16",
    "owl_topk16_split_f16",
    "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd256",
    "vllm_paged_attention_v2_f16_knhd_hd256bs32",
    "vllm_reshape_and_cache_f16_knhd",

    "owl_reshape_and_cache_fp8kv_knhd",
    "owl_rmsnorm_bf16",
    "owl_rmsnorm_f16",
    "owl_sigmoid_f16",
    "owl_sigmoid_gate_mul_f16",
    "owl_silu_and_mul_bf16",
    "owl_silu_and_mul_f16",
    "owl_xqa_decode",
    "owl_xqa_prefill",
    "owl_xqa_verify",
    "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256",
    "vllm_paged_attention_v2_fp8_knhd_hd256bs32",
    "vllm_paged_attention_v2_reduce_f16_hd256",
    "vllm_reshape_and_cache_bf16",
];

#[cfg(test)]
mod tests {
    /// **池面算子的源定义门**(E1,2026-10-17):从声明面 `POOL` 投影出"触池"的
    /// **原生**算子,逐个核 `.cu` 源里有定义 —— 这是 models 侧那份手抄核名清单
    /// 的**派生替身**(手抄清单会漏:新池核忘了写进去 ⇒ 门静默变窄,正是
    /// `kv_manifest_gate` 的 FIXME)。
    ///
    /// 外族(虚核:FI / XQA)不入此门 —— 它们的符号不在 AOT 资产索引里
    /// (实测 5 名:`flashinfer_prefill_paged_{f16,fp8kv}` / `owl_xqa_{decode,prefill,verify}`),
    /// 由各自族 runtime 自校验;本门只覆盖 `.cu` 编译进资产的原生核。
    #[test]
    fn pool_kernels_are_defined_in_source() {
        let mut checked = 0usize;
        let mut foreign = 0usize;
        for m in super::all() {
            if m.pool.is_empty() {
                continue;
            }
            if m.family != crate::contract::OpId(crate::spec::NATIVE) {
                foreign += 1;
                continue;
            }
            assert!(
                crate::cu_sig::source_defines(m.src, m.entry),
                "{}: 声明了池访问(POOL)但 .cu 源里无同名定义 —— 池写者/读者漏建,\
                 或声明面写错了源(池 dtype 改造漏网的机器面)",
                m.entry
            );
            checked += 1;
        }
        // 非空门:池面原生算子实测 **63** 名(2026-10-17 全 feature 档;
        // 外族 FI/XQA 7 名自理);投影断了(如 POOL 声明被清空)本行立刻红
        assert!(checked >= 30, "池面原生算子仅 {checked} 名 —— 投影异常(实测 63)");
        eprintln!("[pool-manifest] 原生池面算子 {checked} 名全部有源定义;外族自理 {foreign} 名");
    }

    /// **8 位族的声明纪律门**(2026-10-16):fp8 / int8 **必须点名格式**,
    /// 不许用 `U8` 兜底 —— `U8` 只留给真正不透明的字节块。
    ///
    /// 为什么要有这道门:F16/BF16 那一案(L3 只认线格式 tag ⇒ "声明 F16 而
    /// 内容是 bf16" 能静默过)在 8 位族同样存在;点名格式后,
    /// **语义面(kv_dtype / 池访问)与事实对不上就在此红**,不必等端到端。
    #[test]
    fn eight_bit_dtypes_are_named_not_u8() {
        use crate::contract::Dtype;
        let mut checked = 0usize;
        for m in super::all() {
            // ① 通则:语义槽(kv_dtype / 池访问)禁 U8
            if let Some(dt) = m.shape.kv_dtype {
                assert_ne!(
                    dt, Dtype::U8,
                    "{}: kv_dtype 不许用 U8 兜(fp8 请点名 F8E4M3/F8E5M2)", m.entry
                );
            }
            for pa in m.pool {
                if let Some(dt) = pa.dtype {
                    assert_ne!(
                        dt, Dtype::U8,
                        "{}: 池访问({})不许用 U8 兜(点名格式)", m.entry, pa.pool
                    );
                }
            }
            // ② fp8 算子 ⇒ 点名两种 fp8 之一
            if m.entry.contains("fp8") {
                if let Some(dt) = m.shape.kv_dtype {
                    assert!(
                        matches!(dt, Dtype::F8E4M3 | Dtype::F8E5M2),
                        "{}: fp8 算子的 kv_dtype 必须点名(e4m3/e5m2),实得 {dt:?}",
                        m.entry
                    );
                    checked += 1;
                }
            }
            // ③ int8/s8 算子 ⇒ 8 位张量槽必须是 I8
            if m.entry.contains("int8") || m.entry.contains("_s8") || m.entry.contains("kS8") {
                for p in m.params {
                    if let crate::spec::ParamType::Tensor { dtype, .. } = p.ty {
                        if dtype.size_bytes() == 1 {
                            assert_eq!(
                                dtype, Dtype::I8,
                                "{}::{}: int8 族的 1B 张量槽必须是 I8(有符号),实得 {dtype:?}",
                                m.entry, p.name
                            );
                            checked += 1;
                        }
                    }
                }
            }
        }
        assert!(checked > 10, "本法应该覆盖到十几处 8 位声明,实得 {checked}(门面失明?)");
        eprintln!("[8bit-gate] 点名格式 OK:覆盖 {checked} 处语义/张量槽");
    }

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

    /// **并集 = 两档清单的集合并**(机器门;2026-10-15 立)。
    ///
    /// 为什么:三张表(spec-dflash / awq27b / union)都是**手抄清单**,
    /// "union = sp ∪ awq" 此前只是注释里的一句话 ⇒ 实测已漂移:
    /// ① A5/F5d 给 spec-dflash 补的 f16 主池族 6 项并集没跟;
    /// ② 并集里还留着无主的 `owl_topk16_f16`(两档白名单都没有 ⇒
    ///    生产不会发射它)。
    /// 纪律 = **集合相等**(双向包含);顺序/分组不判。
    /// (反例:某分支把并集里的 `owl_naive_attn_nc_bf16_v2` 换成了未接线的
    ///  `v3` —— 本门会当场红,而当时它绿着。)
    #[test]
    fn production_union_equals_flavor_union() {
        use std::collections::BTreeSet;
        let set = |l: &[&'static str]| -> BTreeSet<&'static str> { l.iter().copied().collect() };
        let sp = set(PROD_SPEC_DFLASH_2026_10_13);
        let aw = set(PROD_AWQ27B_2026_10_13);
        let un = set(PRODUCTION_UNION_2026_10_13);
        let want: BTreeSet<&str> = sp.union(&aw).copied().collect();
        let missing: Vec<&&str> = want.difference(&un).collect();
        let extra: Vec<&&str> = un.difference(&want).collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "并集 != 两档并(spec-dflash ∪ awq27b):\n  并集缺:{missing:?}\n  并集多:{extra:?}"
        );
        // 两档清单自身无重复(手抄清单的常见笔误)
        for (name, l) in [
            ("spec-dflash", PROD_SPEC_DFLASH_2026_10_13),
            ("awq27b", PROD_AWQ27B_2026_10_13),
            ("union", PRODUCTION_UNION_2026_10_13),
        ] {
            let s = set(l);
            assert_eq!(s.len(), l.len(), "{name} 清单有重复项");
        }
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
        const SRC: &'static str = "";
        const PARAMS: &'static [Param] = &[];
        const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
            kv_dtype: Some(crate::contract::Dtype::F8E4M3),
            head_dim: &[],
            page: &[],
            grp: None,
            tile: None,
        };
        const POOL: &'static [PoolAccess] = &[PoolAccess::read(
            pools::KV_MAIN,
            LayoutTag::Knhd,
            crate::contract::Dtype::F8E4M3,
        )];
    }
    struct Mismatched;
    impl KernelSpec for Mismatched {
        const ENTRY: &'static str = "fake_mismatch";
        const SRC: &'static str = "";
        const PARAMS: &'static [Param] = &[];
        const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
            kv_dtype: Some(crate::contract::Dtype::F8E4M3),
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
        const SRC: &'static str = "";
        const PARAMS: &'static [Param] = &[];
        const POOL: &'static [PoolAccess] = &[PoolAccess::write(
            pools::KV_MAIN,
            LayoutTag::Knhd,
            crate::contract::Dtype::F8E4M3,
        )];
    }
    struct WriteClassic;
    impl KernelSpec for WriteClassic {
        const ENTRY: &'static str = "fake_writer_classic";
        const SRC: &'static str = "";
        const PARAMS: &'static [Param] = &[];
        const POOL: &'static [PoolAccess] = &[PoolAccess::write(
            pools::KV_MAIN,
            LayoutTag::Classic,
            crate::contract::Dtype::F8E4M3,
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

    /// **槽序回归锁(族算子;2026-10-15)**
    ///
    /// # 为什么需要
    ///
    /// 族侧 `client/<族>::parse` 与 `ins_args`/`scalar_args` 都**手写下标**
    /// (`blocks[0]`/`scalars[1]`),隐含一份槽序假设;`PARAMS` 是另一份。
    /// 改 `PARAMS` 里两个**同类型**槽时,L3(`check_from_params`)只看类型 ⇒
    /// 照放行,而解析器会把 `state` 当 `gate` 读 = **静默错值**。
    /// 本锁把两侧钉死:**改序 ⇒ 本测试红 ⇒ 作者必须同时审 parse 与 builder**。
    ///
    /// 与既有 `scalar_slot_order_matches_legacy_sig`(逐族、比旧 SIG 串)互补:
    /// 那些锁的是"与历史签名一致",本锁的是"**全部族算子一个不漏** + 名序列"。
    ///
    /// 覆盖率门:锁表必须恰好覆盖全部族算子(一个不漏、一个不多)。
    #[test]
    fn slot_order_lock_family_ops() {
        // (族身份常量, 槽名序列, 类型签名序列)。**两把锁一起**:名字序列防
        // 同型槽对调,类型签名防 dtype/rank 漂移(F16→BF16 / rank3→rank4 ——
        // 只锁名字时这两类变更**不触发任何门**;2026-10-15 审查定谳)。
        const EXPECT: &[(&str, &[&str], &[&str])] = &[
            (crate::family::gdn_chunked::GDN_CHUNKED_FWD, &["q", "k", "v", "beta", "gate", "state", "tables", "out", "t", "slot", "nv", "nk", "kd", "scale_bits", "nt"], &["F16x3", "F16x3", "F16x3", "F16x2", "F16x2", "F32x4", "U32x1", "F16x3", "U64", "U64", "U64", "U64", "U64", "F32Bits", "U64"]),
            (crate::family::gdn_scalar::GDN_SCALAR_FWD, &["q", "k", "v", "beta", "gate", "state", "out", "t", "slot", "ns", "nv", "nk", "kd", "scale_bits"], &["F32x3", "F32x3", "F32x3", "F32x2", "F32x2", "F32x4", "F16x3", "U64", "U64", "U64", "U64", "U64", "U64", "F32Bits"]),
            (crate::family::marlin::GEMM_W4A16_AWQ, &["a", "b", "out", "scales", "zeros", "ws", "c_tmp", "m", "k", "n", "groupsize"], &["F16x2", "U32x2", "F16x2", "F16x2", "U32x2", "U32x1", "U32x1", "U64", "U64", "U64", "U64"]),
            (crate::family::marlin::GEMM_W4A8_AWQ, &["a", "b", "out", "scales", "zeros", "a_scales", "ws", "c_tmp", "m", "k", "n", "groupsize"], &["I8x2", "U32x2", "F16x2", "F16x2", "U32x2", "F32x2", "U32x1", "U32x1", "U64", "U64", "U64", "U64"]),
            (crate::family::marlin::GEMM_W4A16_BF16, &["a", "b", "out", "scales", "ws", "c_tmp", "m", "k", "n", "groupsize"], &["BF16x2", "U32x2", "BF16x2", "BF16x2", "U32x1", "U32x1", "U64", "U64", "U64", "U64"]),
            (crate::family::marlin::GEMM_W4A16, &["a", "b", "out", "scales", "ws", "c_tmp", "m", "k", "n", "groupsize"], &["F16x2", "U32x2", "F16x2", "F16x2", "U32x1", "U32x1", "U64", "U64", "U64", "U64"]),
            (crate::family::cublas::GEMM_F16, &["a", "b", "out", "m", "k", "n", "nt"], &["F16x2", "F16x2", "F16x2", "U64", "U64", "U64", "U64"]),
            (crate::family::cublas::GEMM_BF16, &["a", "b", "out", "m", "k", "n", "nt"], &["BF16x2", "BF16x2", "BF16x2", "U64", "U64", "U64", "U64"]),
            // 写列窗臂(W-2d;独立 PARSE_SLOTS_DST;此前漏登本锁)
            (crate::family::cublas::GEMM_F16_DST, &["a", "b", "out", "m", "k", "n", "nt", "ldc", "off"], &["F16x2", "F16x2", "F16x2", "U64", "U64", "U64", "U64", "U64", "U64"]),
            (crate::family::flashinfer::PREFILL_FI_FP8KV, &["q", "k_cache", "v_cache", "q_cu", "indices", "indptr", "last_len", "wr", "out", "total_rows", "ctx_total", "t", "hq", "hkv", "hd", "page", "sm_scale_bits"], &["F16x2", "U32x1", "U32x1", "U32x1", "U32x1", "U32x1", "U32x1", "F16x1", "F16x2", "U64", "U64", "U64", "U64", "U64", "U64", "U64", "F32Bits"]),
            (crate::family::flashinfer::PREFILL_FI, &["q", "k_cache", "v_cache", "q_cu", "indices", "indptr", "last_len", "wr", "out", "total_rows", "ctx_total", "t", "hq", "hkv", "hd", "page", "sm_scale_bits"], &["F16x2", "F16x5", "F16x4", "U32x1", "U32x1", "U32x1", "U32x1", "F16x1", "F16x2", "U64", "U64", "U64", "U64", "U64", "U64", "U64", "F32Bits"]),
            // 视觉塔注意力(ragged 非因果;2026-10-15):槽位与 native
            // `vis::AttnFullF16` **逐字同构**(driver 按 dtype/图数在两臂间分流)
            (crate::family::flashinfer::ATTN_RAGGED_F16, &["q", "k", "v", "cu_seqlens", "out", "rows", "heads", "head_dim", "n_seq", "scale"], &["F16x3", "F16x3", "F16x3", "F32x1", "F16x3", "U64", "U64", "U64", "I32", "F32"]),
            (crate::family::xqa::VERIFY_XQA, &["q", "k_cache", "v_cache", "page_table", "seq_lens", "mask", "semaphores", "scratch", "wr", "out", "q_seq_len", "hkv", "hd", "page", "max_seq_len", "q_scale_bits", "sm_count", "stride_page", "stride_token", "stride_head"], &["F16x2", "U32x1", "U32x1", "F32x2", "F32x2", "F32x0", "F32x2", "F16x1", "F16x0", "F16x2", "U64", "U64", "U64", "U64", "U64", "F32Bits", "U64", "U64", "U64", "U64"]),
            (crate::family::xqa::DECODE_XQA, &["q", "k_cache", "v_cache", "page_table", "seq_lens", "mask", "semaphores", "scratch", "wr", "out", "q_seq_len", "hkv", "hd", "page", "max_seq_len", "q_scale_bits", "sm_count", "stride_page", "stride_token", "stride_head"], &["F16x2", "U32x1", "U32x1", "F32x2", "F32x2", "F32x0", "F32x2", "F16x1", "F16x0", "F16x2", "U64", "U64", "U64", "U64", "U64", "F32Bits", "U64", "U64", "U64", "U64"]),
            // prefill 臂:与 verify/decode **同一张 PARAMS 表**(同资产同槽序;
            // 差别只在 host 侧几何 ⇒ 不在这张表的表达面内)
            (crate::family::xqa::PREFILL_XQA, &["q", "k_cache", "v_cache", "page_table", "seq_lens", "mask", "semaphores", "scratch", "wr", "out", "q_seq_len", "hkv", "hd", "page", "max_seq_len", "q_scale_bits", "sm_count", "stride_page", "stride_token", "stride_head"], &["F16x2", "U32x1", "U32x1", "F32x2", "F32x2", "F32x0", "F32x2", "F16x1", "F16x0", "F16x2", "U64", "U64", "U64", "U64", "U64", "F32Bits", "U64", "U64", "U64", "U64"]),
            // 通信族(TP2 S2;零门 ⇒ 无条件入锁):stage / reduce 各三 dtype。
            // payload 槽 rank 0(阶数无关:AR 挂任意算子输出);staging 两槽 = U8 字节块。
            (crate::family::comm::asset::STAGE_F16, &["local", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"], &["F16x0", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            (crate::family::comm::asset::STAGE_BF16, &["local", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"], &["BF16x0", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            (crate::family::comm::asset::STAGE_F32, &["local", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"], &["F32x0", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            (crate::family::comm::asset::REDUCE_F16, &["out", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"], &["F16x0", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            (crate::family::comm::asset::REDUCE_BF16, &["out", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"], &["BF16x0", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            (crate::family::comm::asset::REDUCE_F32, &["out", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"], &["F32x0", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            // fp8 载荷档(§47 第二步;槽序/类型与常规档**逐字同形** —— 只有核名与载荷语义不同)
            (crate::family::comm::asset::STAGE_F8_F16, &["local", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"], &["F16x0", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            (crate::family::comm::asset::STAGE_F8_BF16, &["local", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"], &["BF16x0", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            (crate::family::comm::asset::REDUCE_F8_F16, &["out", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"], &["F16x0", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            (crate::family::comm::asset::REDUCE_F8_BF16, &["out", "own", "peer", "data", "bytes", "slot_bytes", "slab_off"], &["BF16x0", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            // 1-stage 融合核(§35;对称缓冲 —— 无 stage 段、零拷贝):槽 = out/own/peer/peer_data/bytes
            (crate::family::comm::asset::ONE_STAGE_F16, &["out", "src", "own", "peer", "peer_data", "bytes"], &["F16x0", "F16x0", "U8x1", "U8x1", "U8x1", "U64"]),
            (crate::family::comm::asset::ONE_STAGE_BF16, &["out", "src", "own", "peer", "peer_data", "bytes"], &["BF16x0", "BF16x0", "U8x1", "U8x1", "U8x1", "U64"]),
            (crate::family::comm::asset::ONE_STAGE_F32, &["out", "src", "own", "peer", "peer_data", "bytes"], &["F32x0", "F32x0", "U8x1", "U8x1", "U8x1", "U64"]),
            // v1 两核融合(§11.2;2026-10-19):8 槽 —— data_w/data_r 分开(方向不同)
            (crate::family::comm::asset::FUSED_F16, &["payload", "own", "peer", "data_w", "data_r", "bytes", "slot_bytes", "slab_off"], &["F16x0", "U8x1", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            (crate::family::comm::asset::FUSED_BF16, &["payload", "own", "peer", "data_w", "data_r", "bytes", "slot_bytes", "slab_off"], &["BF16x0", "U8x1", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
            (crate::family::comm::asset::FUSED_F32, &["payload", "own", "peer", "data_w", "data_r", "bytes", "slot_bytes", "slab_off"], &["F32x0", "U8x1", "U8x1", "U8x1", "U8x1", "U64", "U64", "U64"]),
        ];
        for (entry, want, want_ty) in EXPECT {
            let m = meta(entry).unwrap_or_else(|| panic!("锁表项 {entry} 不在声明表里"));
            assert_eq!(
                m.param_names(),
                *want,
                "{entry}: 槽序变了 —— 请同步审 client/<族> 的 parse 与 ins_args/scalar_args \
                 (同类型槽对调 = L3 放行 + 解析器静默错值)"
            );
            let got_ty: Vec<String> = m
                .params
                .iter()
                .map(|p| match p.ty {
                    crate::spec::ParamType::Tensor { dtype, rank } => format!("{dtype:?}x{rank}"),
                    crate::spec::ParamType::Scalar(s) => format!("{s:?}"),
                })
                .collect();
            assert_eq!(
                got_ty,
                *want_ty,
                "{entry}: 类型签名变了(dtype/rank)—— 名字锁拦不住这类漂移;\
                 请同步审各族 handler 的 dtype 假设与 client face 的 builder"
            );
        }
        // 覆盖率门:族算子一个不漏
        let fam: Vec<&str> = all().filter(|m| m.is_family()).map(|m| m.entry).collect();
        assert_eq!(fam.len(), EXPECT.len(), "族算子清单与锁表项数不一致:{fam:?}");
        for e in &fam {
            assert!(EXPECT.iter().any(|(k, _, _)| k == e), "族算子 {e} 未进锁表");
        }
        // 前置件自检:**按名寻址**可用(未来 parse/builder 按名取槽的基础)
        let g = meta("gdn_chunked_delta_rule_fwd").expect("gdn chunked");
        assert_eq!(g.slot_index("gate"), Some(4));
        assert_eq!(g.slot_index("state"), Some(5));
        assert_eq!(g.slot_index("tables"), Some(6), "P-6b 序列表槽");
        assert_eq!(g.slot_index("out"), Some(7));
        assert_eq!(g.slot_index("slot"), Some(9), "标量槽同表寻址");
        assert_eq!(g.slot_index("不存在"), None);
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


