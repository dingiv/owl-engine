//! attention 域生产算子(池写 / decode / 草稿 NC)。
//!
//! 参数名/序/类型 = `.cu` 签名的逐字抄录(机器互证见 `tests/op_tables.rs`)。

use crate::contract::{Dtype, Law};
use crate::spec::{
    meta_of, pools, CaptureEvidence, Determinism, KernelMeta, KernelSpec, LayoutTag, Param,
    PoolAccess,
};

// ============================================================================
// 草稿非因果块注意力(DFlash2 propose 的 NC 核 v2;2026-10-13 ③ 转正)
// ============================================================================

/// `owl_naive_attn_nc_bf16_v2` —— 非因果块注意力(自块直读 + 前缀池)。
///
/// v2 = 只换并行结构(lane→连续 dim + warp butterfly + split-K),语义/槽序
/// 与旧核逐字同一套;实测 prefix2048 2.238 → 1.201ms(1.86×),数值同档。
pub struct NcAttnBf16V2;

impl KernelSpec for NcAttnBf16V2 {
    const ENTRY: &'static str = "owl_naive_attn_nc_bf16_v2";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::BF16, 2),
        Param::t_in("k_self", Dtype::BF16, 3),
        Param::t_in("v_self", Dtype::BF16, 3),
        Param::t_in("kc", Dtype::BF16, 3),
        Param::t_in("vc", Dtype::BF16, 3),
        // 整请求 KV 长(设备侧标量,f32 张量;非 host 标量 —— 图内可读)
        Param::t_in("kv_len_p", Dtype::F32, 1),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        // ABI 占位形参(历史遗留:q_tokens 已由 grid.y 承担;签名不动)
        Param::i32("x"),
        Param::t_out("out", Dtype::BF16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_DRAFT, LayoutTag::Classic, Dtype::BF16)];
    const SELFCHECK: Option<&'static dyn crate::spec::BootCheck> =
        Some(&crate::ops::oracles::nc_bf16_v2::NcBf16V2Oracle);
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        // 草稿池(bfloat16;classic 布局 —— 布局声明在 PoolAccess)
        kv_dtype: Some(Dtype::BF16),
        head_dim: &[128, 256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}

// ============================================================================
// paged attention v2(kNHD + fp8 池;裸 decode 主核 + 分区归并)
// ============================================================================

/// `vllm_paged_attention_v2_fp8_knhd_hd256bs32` —— 裸 decode 主核
/// (bs32 = block 32 行;kNHD 寻址;KV 池 e4m3 1B/elem)。
pub struct PagedAttnV2Fp8KnhdHd256Bs32;

/// paged v2 主核共享参数表(18 参;f16/fp8 × classic/kNHD 同 ABI)
const PAGED_V2_FP8_KNHD_PARAMS: &[Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        // fp8 e4m3 池(1B/elem;.cu 以 uint16_t* 声明仅为 2 字节装载)
        Param::t_in("k_cache", Dtype::U8, 4),
        Param::t_in("v_cache", Dtype::U8, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 1),
        Param::t_in("alibi_slopes", Dtype::F32, 1),
        Param::t_in("exp_sums", Dtype::F32, 2),
        Param::t_in("max_logits", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("scale"),
        Param::i32("max_num_blocks_per_seq"),
        Param::i32("q_stride"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::f32("softscapping"),
        Param::i32("sliding_window"),
        Param::i32("use_alibi"),
        Param::t_out("tmp_out", Dtype::F16, 2),
];

impl KernelSpec for PagedAttnV2Fp8KnhdHd256Bs32 {
    const ENTRY: &'static str = "vllm_paged_attention_v2_fp8_knhd_hd256bs32";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = PAGED_V2_FP8_KNHD_PARAMS;
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::U8)];
}

/// `vllm_paged_attention_v2_reduce_f16_hd256` —— 分区 partial 归并
/// (exp_sums/max_logits + tmp_out → out;裸 decode 第二发)。
pub struct PagedAttnV2ReduceF16Hd256;

impl KernelSpec for PagedAttnV2ReduceF16Hd256 {
    const ENTRY: &'static str = "vllm_paged_attention_v2_reduce_f16_hd256";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("exp_sums", Dtype::F32, 2),
        Param::t_in("max_logits", Dtype::F32, 2),
        Param::t_in("tmp_out", Dtype::F16, 2),
        Param::t_in("context_lens", Dtype::F32, 1),
        Param::i32("max_num_partitions"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

// ============================================================================
// K0:KV 散写(kNHD + fp8 池)
// ============================================================================

/// `owl_reshape_and_cache_fp8kv_knhd` —— 批量 KV 写池(f16 入 → e4m3 池;
/// kNHD 寻址)。**池 dtype 改造的写者穷举律**要求本核的写集声明在案
/// (B6.3 全零案:E10)。
pub struct K0WriteFp8Knhd;

impl KernelSpec for K0WriteFp8Knhd {
    const ENTRY: &'static str = "owl_reshape_and_cache_fp8kv_knhd";
    const SRC: &'static str = "cu/attn/reshape_and_cache_fp8kv.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::F16, 3),
        Param::t_in("value", Dtype::F16, 3),
        Param::t_in("key_cache", Dtype::U8, 4),
        Param::t_in("value_cache", Dtype::U8, 4),
        Param::t_in("slot_mapping", Dtype::F32, 1),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        // ABI 占位形参(历史遗留;签名不动)
        Param::i32("x"),
        // 假输出(ABI 需要末位输出块;内核不写内容)
        Param::t_out("out", Dtype::F16, 1),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_MAIN, LayoutTag::Knhd, Dtype::U8)];
    const SELFCHECK: Option<&'static dyn crate::spec::BootCheck> =
        Some(&crate::ops::oracles::k0_fp8_knhd::K0Fp8KnhdOracle);
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8),
        head_dim: &[128, 256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}


// ============================================================================
// 池访问族补齐(2026-10-13 B1b):**生产里每一个池写者/读者都要有声明**
// (E10「漏写者 ⇒ 槽区全零」/ E14 双寻址 / E15 契约腐烂 的机器面)
// ============================================================================

/// `vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256` —— chunked prefill
/// 主核(fp8/kNHD 池**读**;短 prefill 尾段)。23 参 = 8 块 + 12 i32 + 2 f32 + 输出。
pub struct ChunkedPrefillFp8KnhdHd256;

impl KernelSpec for ChunkedPrefillFp8KnhdHd256 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::U8, 4),
        Param::t_in("v_cache", Dtype::U8, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 1),
        Param::t_in("query_start_len", Dtype::F32, 1),
        Param::t_in("alibi_slopes", Dtype::F32, 1),
        Param::t_in("sinks", Dtype::F32, 1),
        Param::i32("num_kv_heads"),
        Param::f32("sm_scale"),
        Param::i32("block_table_stride"),
        Param::i32("num_seqs"),
        Param::i32("num_query_heads"),
        Param::i32("num_query_tokens"),
        Param::f32("softscapping"),
        Param::i32("o_stride_tokens"),
        Param::i32("sliding_window"),
        Param::i32("total_num_blocks"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("use_alibi_flag"),
        Param::i32("use_sinks_flag"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::U8)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8),
        head_dim: &[256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}

/// `owl_reshape_and_cache_fp8kv` —— K0 写池(**classic** 寻址 + fp8 池)。
/// 与 kNHD 变体同参数表,只有寻址语义不同 —— 两者必须同时有声明,否则
/// "池布局翻转"时会有一侧无声。
pub struct K0WriteFp8Classic;

impl KernelSpec for K0WriteFp8Classic {
    const ENTRY: &'static str = "owl_reshape_and_cache_fp8kv";
    const SRC: &'static str = "cu/attn/reshape_and_cache_fp8kv.cu";
    const PARAMS: &'static [Param] = K0_PARAMS;
    const POOL: &'static [PoolAccess] = &[
        PoolAccess::write(pools::KV_MAIN, LayoutTag::Classic, Dtype::U8),
        PoolAccess::write(pools::KV_DRAFT, LayoutTag::Classic, Dtype::U8),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8),
        head_dim: &[128, 256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}

/// K0 双臂共用参数表(逐字 = `.cu` 签名;classic 与 kNHD 同 ABI)
const K0_PARAMS: &[Param] = &[
    Param::t_in("key", Dtype::F16, 3),
    Param::t_in("value", Dtype::F16, 3),
    Param::t_in("key_cache", Dtype::U8, 4),
    Param::t_in("value_cache", Dtype::U8, 4),
    Param::t_in("slot_mapping", Dtype::F32, 1),
    Param::i32("key_stride"),
    Param::i32("value_stride"),
    Param::i32("num_heads"),
    Param::i32("head_size"),
    Param::i32("block_size"),
    Param::i32("x"),
    Param::t_out("out", Dtype::F16, 1),
];

/// `vllm_reshape_and_cache_bf16` —— **草稿池**写者(bf16,classic 寻址)。
/// DFlash2 草稿池的权威写者;漏声明 = 草稿池静默全零(E10 同类)。
pub struct K0WriteBf16;

impl KernelSpec for K0WriteBf16 {
    const ENTRY: &'static str = "vllm_reshape_and_cache_bf16";
    const SRC: &'static str = "cu/attn/reshape_and_cache.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::BF16, 3),
        Param::t_in("value", Dtype::BF16, 3),
        Param::t_in("key_cache", Dtype::BF16, 4),
        Param::t_in("value_cache", Dtype::BF16, 4),
        Param::t_in("slot_mapping", Dtype::F32, 1),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_out("out", Dtype::BF16, 1),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_DRAFT, LayoutTag::Classic, Dtype::BF16)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::BF16),
        head_dim: &[128, 256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}

/// `owl_naive_attn_nc_fp8kv_bf16` —— 草稿 NC 注意力(**fp8 前缀池**变体;
/// 自块直读保持 bf16)。fp8 草稿池档的生产读核。
pub struct NcAttnFp8kvBf16;

impl KernelSpec for NcAttnFp8kvBf16 {
    const ENTRY: &'static str = "owl_naive_attn_nc_fp8kv_bf16";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::BF16, 2),
        Param::t_in("k_self", Dtype::BF16, 3),
        Param::t_in("v_self", Dtype::BF16, 3),
        Param::t_in("kc", Dtype::U8, 3),
        Param::t_in("vc", Dtype::U8, 3),
        Param::t_in("kv_len_p", Dtype::F32, 1),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::t_out("out", Dtype::BF16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_DRAFT, LayoutTag::Classic, Dtype::U8)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        // 草稿池 fp8 档(前缀池 e4m3 字节;自块直读保持 bf16)
        kv_dtype: Some(Dtype::U8),
        head_dim: &[128, 256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}


// ============================================================================
// norm + rope 融合(driver::attn::norm_rope)
// ============================================================================

/// `owl_norm_rope_f16` —— qk-norm(×(1+w)^{w_off})+ rotate-half partial rope
/// 三发合一(x 是 per-head [value|gate] 半段拼接)
pub struct NormRopeF16;

impl KernelSpec for NormRopeF16 {
    const ENTRY: &'static str = "owl_norm_rope_f16";
    const SRC: &'static str = "cu/attn/norm_rope.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("w", Dtype::F16, 1),
        Param::t_in("cos_t", Dtype::F16, 2),
        Param::t_in("sin_t", Dtype::F16, 2),
        Param::t_in("pos", Dtype::F32, 1),
        Param::f32("eps"),
        Param::sz("row_stride"),
        Param::sz("head_stride"),
        Param::sz("half"),
        Param::i32("w_off"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_norm_rope_bf16` —— 同上(x/w/out 为 BF16;cos/sin 共享 f16 表)
pub struct NormRopeBf16;

impl KernelSpec for NormRopeBf16 {
    const ENTRY: &'static str = "owl_norm_rope_bf16";
    const SRC: &'static str = "cu/attn/norm_rope.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::t_in("w", Dtype::BF16, 1),
        Param::t_in("cos_t", Dtype::F16, 2),
        Param::t_in("sin_t", Dtype::F16, 2),
        Param::t_in("pos", Dtype::F32, 1),
        Param::f32("eps"),
        Param::sz("row_stride"),
        Param::sz("head_stride"),
        Param::sz("half"),
        Param::i32("w_off"),
        Param::t_out("out", Dtype::BF16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

// ============================================================================
// 命名 static(**单一提升点**;域表与族表共用同一份 KernelMeta)
// ============================================================================

pub static NC_ATTN_BF16_V2: KernelMeta = meta_of::<NcAttnBf16V2>();
pub static PAGED_V2_FP8_KNHD_HD256BS32: KernelMeta = meta_of::<PagedAttnV2Fp8KnhdHd256Bs32>();
pub static PAGED_V2_REDUCE_F16_HD256: KernelMeta = meta_of::<PagedAttnV2ReduceF16Hd256>();
pub static K0_WRITE_FP8_KNHD: KernelMeta = meta_of::<K0WriteFp8Knhd>();
pub static CHUNKED_PREFILL_FP8_KNHD_HD256: KernelMeta = meta_of::<ChunkedPrefillFp8KnhdHd256>();
pub static K0_WRITE_FP8_CLASSIC: KernelMeta = meta_of::<K0WriteFp8Classic>();
pub static K0_WRITE_BF16: KernelMeta = meta_of::<K0WriteBf16>();
pub static NC_ATTN_FP8KV_BF16: KernelMeta = meta_of::<NcAttnFp8kvBf16>();
pub static NORM_ROPE_F16: KernelMeta = meta_of::<NormRopeF16>();
pub static NORM_ROPE_BF16: KernelMeta = meta_of::<NormRopeBf16>();


// ============================================================================
// R1 批次 1:attention 池读写变体族(用户裁决 2026-10-13:凡是用到都必须有声明)
// —— 共享参数表(同 ABI 只差 ENTRY/池 dtype/几何声明),逐表由 `.cu` 互证门对账
// ============================================================================

/// paged v2 主核参数表(f16/fp8 × classic/kNHD × hd128/256 同 ABI;18 参)
const PAGED_V2_PARAMS: &[Param] = PAGED_V2_FP8_KNHD_PARAMS;
/// reduce 参数表(6 参;hd128/hd256 同 ABI)
const PAGED_V2_REDUCE_PARAMS: &[Param] = &[
    Param::t_in("exp_sums", Dtype::F32, 2),
    Param::t_in("max_logits", Dtype::F32, 2),
    Param::t_in("tmp_out", Dtype::F16, 2),
    Param::t_in("context_lens", Dtype::F32, 1),
    Param::i32("max_num_partitions"),
    Param::t_out("out", Dtype::F16, 2),
];

/// paged v2 变体描述符(表里只写差异项;ABI 由共享参数表承担)
macro_rules! paged_v2_variant {
    ($ty:ident, $entry:literal, $pool_dt:expr, $hd:expr, $page:expr, $dims:expr) => {
        #[doc = concat!("`", $entry, "`(paged attention v2 变体;ABI = PAGED_V2_PARAMS)")]
        pub struct $ty;

        impl KernelSpec for $ty {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
            const PARAMS: &'static [Param] = PAGED_V2_PARAMS;
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
            const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
                kv_dtype: Some($pool_dt),
                head_dim: $dims,
                page: $page,
                grp: None,
                tile: None,
            };
            const POOL: &'static [PoolAccess] =
                &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, $pool_dt)];
        }
    };
}

paged_v2_variant!(PagedV2F16Hd128Bs32, "vllm_paged_attention_v2_f16_hd128bs32", Dtype::F16, 128, &[32], &[128]);
paged_v2_variant!(PagedV2F16Hd256Bs32, "vllm_paged_attention_v2_f16_hd256bs32", Dtype::F16, 256, &[32], &[256]);
paged_v2_variant!(PagedV2F16KnhdHd128Bs32, "vllm_paged_attention_v2_f16_knhd_hd128bs32", Dtype::F16, 128, &[32], &[128]);
paged_v2_variant!(PagedV2F16KnhdHd256Bs32, "vllm_paged_attention_v2_f16_knhd_hd256bs32", Dtype::F16, 256, &[32], &[256]);
paged_v2_variant!(PagedV2F16KnhdHd128, "vllm_paged_attention_v2_f16_knhd_hd128", Dtype::F16, 128, &[16], &[128]);
paged_v2_variant!(PagedV2F16KnhdHd256, "vllm_paged_attention_v2_f16_knhd_hd256", Dtype::F16, 256, &[16], &[256]);
paged_v2_variant!(PagedV2Fp8Hd128, "vllm_paged_attention_v2_fp8_hd128", Dtype::U8, 128, &[16], &[128]);
paged_v2_variant!(PagedV2Fp8Hd256, "vllm_paged_attention_v2_fp8_hd256", Dtype::U8, 256, &[16], &[256]);
paged_v2_variant!(PagedV2Fp8Hd128Bs32, "vllm_paged_attention_v2_fp8_hd128bs32", Dtype::U8, 128, &[32], &[128]);
paged_v2_variant!(PagedV2Fp8Hd256Bs32, "vllm_paged_attention_v2_fp8_hd256bs32", Dtype::U8, 256, &[32], &[256]);
paged_v2_variant!(PagedV2Fp8KnhdHd128Bs32, "vllm_paged_attention_v2_fp8_knhd_hd128bs32", Dtype::U8, 128, &[32], &[128]);
paged_v2_variant!(PagedV2Fp8KnhdHd256, "vllm_paged_attention_v2_fp8_knhd_hd256", Dtype::U8, 256, &[16], &[256]);
paged_v2_variant!(PagedV2Fp8KnhdHd128, "vllm_paged_attention_v2_fp8_knhd_hd128", Dtype::U8, 128, &[16], &[128]);

/// `vllm_paged_attention_v2_reduce_f16_hd128` —— 分区归并(hd128 变体)
pub struct PagedV2ReduceF16Hd128;

impl KernelSpec for PagedV2ReduceF16Hd128 {
    const ENTRY: &'static str = "vllm_paged_attention_v2_reduce_f16_hd128";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = PAGED_V2_REDUCE_PARAMS;
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


pub static PAGED_V2_F16_HD128BS32: KernelMeta = meta_of::<PagedV2F16Hd128Bs32>();
pub static PAGED_V2_F16_HD256BS32: KernelMeta = meta_of::<PagedV2F16Hd256Bs32>();
pub static PAGED_V2_F16_KNHD_HD128BS32: KernelMeta = meta_of::<PagedV2F16KnhdHd128Bs32>();
pub static PAGED_V2_F16_KNHD_HD256BS32: KernelMeta = meta_of::<PagedV2F16KnhdHd256Bs32>();
pub static PAGED_V2_F16_KNHD_HD128: KernelMeta = meta_of::<PagedV2F16KnhdHd128>();
pub static PAGED_V2_F16_KNHD_HD256: KernelMeta = meta_of::<PagedV2F16KnhdHd256>();
pub static PAGED_V2_FP8_HD128: KernelMeta = meta_of::<PagedV2Fp8Hd128>();
pub static PAGED_V2_FP8_HD256: KernelMeta = meta_of::<PagedV2Fp8Hd256>();
pub static PAGED_V2_FP8_HD128BS32: KernelMeta = meta_of::<PagedV2Fp8Hd128Bs32>();
pub static PAGED_V2_FP8_HD256BS32: KernelMeta = meta_of::<PagedV2Fp8Hd256Bs32>();
pub static PAGED_V2_FP8_KNHD_HD128BS32: KernelMeta = meta_of::<PagedV2Fp8KnhdHd128Bs32>();
pub static PAGED_V2_FP8_KNHD_HD256: KernelMeta = meta_of::<PagedV2Fp8KnhdHd256>();
pub static PAGED_V2_FP8_KNHD_HD128: KernelMeta = meta_of::<PagedV2Fp8KnhdHd128>();
pub static PAGED_V2_REDUCE_F16_HD128: KernelMeta = meta_of::<PagedV2ReduceF16Hd128>();


/// `owl_naive_decode_attn_f16` —— naive_decode_attn.cu(R1 补表;池声明见 POOL)
pub struct NaiveDecodeAttnF16;

impl KernelSpec for NaiveDecodeAttnF16 {
    const ENTRY: &'static str = "owl_naive_decode_attn_f16";
    const SRC: &'static str = "cu/attn/naive_decode_attn.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k", Dtype::F16, 2),
        Param::t_in("v", Dtype::F16, 2),
        Param::t_in("kc", Dtype::F16, 2),
        Param::t_in("vc", Dtype::F16, 2),
        Param::t_in("slots", Dtype::F32, 2),
        Param::t_in("kv_lens", Dtype::F32, 2),
        Param::sz("bs"),
        Param::sz("q_heads"),
        Param::sz("kv_heads"),
        Param::sz("d_dim"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_naive_decode_attn_f32` —— naive_decode_attn.cu(R1 补表;池声明见 POOL)
pub struct NaiveDecodeAttnF32;

impl KernelSpec for NaiveDecodeAttnF32 {
    const ENTRY: &'static str = "owl_naive_decode_attn_f32";
    const SRC: &'static str = "cu/attn/naive_decode_attn.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F32, 2),
        Param::t_in("k", Dtype::F32, 2),
        Param::t_in("v", Dtype::F32, 2),
        Param::t_in("kc", Dtype::F32, 2),
        Param::t_in("vc", Dtype::F32, 2),
        Param::t_in("slots", Dtype::F32, 2),
        Param::t_in("kv_lens", Dtype::F32, 2),
        Param::sz("bs"),
        Param::sz("q_heads"),
        Param::sz("kv_heads"),
        Param::sz("d_dim"),
        Param::t_in("out", Dtype::F32, 2),
    ];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_naive_attn_nc_f16` —— naive_attn_nc.cu(R1 补表;池声明见 POOL)
pub struct NaiveAttnNcF16;

impl KernelSpec for NaiveAttnNcF16 {
    const ENTRY: &'static str = "owl_naive_attn_nc_f16";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_self", Dtype::F16, 2),
        Param::t_in("v_self", Dtype::F16, 2),
        Param::t_in("kc", Dtype::F16, 2),
        Param::t_in("vc", Dtype::F16, 2),
        Param::t_in("kv_len_p", Dtype::F32, 2),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_DRAFT, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_naive_attn_nc_bf16` —— naive_attn_nc.cu(R1 补表;池声明见 POOL)
pub struct NaiveAttnNcBf16;

impl KernelSpec for NaiveAttnNcBf16 {
    const ENTRY: &'static str = "owl_naive_attn_nc_bf16";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::BF16, 2),
        Param::t_in("k_self", Dtype::BF16, 2),
        Param::t_in("v_self", Dtype::BF16, 2),
        Param::t_in("kc", Dtype::BF16, 2),
        Param::t_in("vc", Dtype::BF16, 2),
        Param::t_in("kv_len_p", Dtype::F32, 2),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::t_in("out", Dtype::BF16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_DRAFT, LayoutTag::Classic, Dtype::BF16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::BF16), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_naive_attn_nc_fp8kv_f16` —— naive_attn_nc.cu(R1 补表;池声明见 POOL)
pub struct NaiveAttnNcFp8KvF16;

impl KernelSpec for NaiveAttnNcFp8KvF16 {
    const ENTRY: &'static str = "owl_naive_attn_nc_fp8kv_f16";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_self", Dtype::F16, 2),
        Param::t_in("v_self", Dtype::F16, 2),
        Param::t_in("kc", Dtype::U8, 2),
        Param::t_in("vc", Dtype::U8, 2),
        Param::t_in("kv_len_p", Dtype::F32, 2),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_DRAFT, LayoutTag::Classic, Dtype::U8)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_reshape_and_cache_f16` —— reshape_and_cache.cu(R1 补表;池声明见 POOL)
pub struct VllmReshapeAndCacheF16;

impl KernelSpec for VllmReshapeAndCacheF16 {
    const ENTRY: &'static str = "vllm_reshape_and_cache_f16";
    const SRC: &'static str = "cu/attn/reshape_and_cache.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::F16, 2),
        Param::t_in("value", Dtype::F16, 2),
        Param::t_in("key_cache", Dtype::F16, 2),
        Param::t_in("value_cache", Dtype::F16, 2),
        Param::t_in("slot_mapping", Dtype::F32, 2),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128, 256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_reshape_and_cache_f16_knhd` —— reshape_and_cache.cu(R1 补表;池声明见 POOL)
pub struct VllmReshapeAndCacheF16Knhd;

impl KernelSpec for VllmReshapeAndCacheF16Knhd {
    const ENTRY: &'static str = "vllm_reshape_and_cache_f16_knhd";
    const SRC: &'static str = "cu/attn/reshape_and_cache.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::F16, 2),
        Param::t_in("value", Dtype::F16, 2),
        Param::t_in("key_cache", Dtype::F16, 2),
        Param::t_in("value_cache", Dtype::F16, 2),
        Param::t_in("slot_mapping", Dtype::F32, 2),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128, 256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_reshape_and_cache_bf16_knhd` —— reshape_and_cache.cu(R1 补表;池声明见 POOL)
pub struct VllmReshapeAndCacheBf16Knhd;

impl KernelSpec for VllmReshapeAndCacheBf16Knhd {
    const ENTRY: &'static str = "vllm_reshape_and_cache_bf16_knhd";
    const SRC: &'static str = "cu/attn/reshape_and_cache.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::BF16, 2),
        Param::t_in("value", Dtype::BF16, 2),
        Param::t_in("key_cache", Dtype::BF16, 2),
        Param::t_in("value_cache", Dtype::BF16, 2),
        Param::t_in("slot_mapping", Dtype::F32, 2),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_in("out", Dtype::BF16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_MAIN, LayoutTag::Knhd, Dtype::BF16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::BF16), head_dim: &[128, 256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_reshape_and_cache_fp8kv_bf16` —— reshape_and_cache_fp8kv.cu(R1 补表;池声明见 POOL)
pub struct ReshapeAndCacheFp8KvBf16;

impl KernelSpec for ReshapeAndCacheFp8KvBf16 {
    const ENTRY: &'static str = "owl_reshape_and_cache_fp8kv_bf16";
    const SRC: &'static str = "cu/attn/reshape_and_cache_fp8kv.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::BF16, 2),
        Param::t_in("value", Dtype::BF16, 2),
        Param::t_in("key_cache", Dtype::U8, 2),
        Param::t_in("value_cache", Dtype::U8, 2),
        Param::t_in("slot_mapping", Dtype::F32, 2),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_in("out", Dtype::BF16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_MAIN, LayoutTag::Classic, Dtype::U8)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8), head_dim: &[128, 256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_reshape_and_cache_fp8kv_bf16_knhd` —— reshape_and_cache_fp8kv.cu(R1 补表;池声明见 POOL)
pub struct ReshapeAndCacheFp8KvBf16Knhd;

impl KernelSpec for ReshapeAndCacheFp8KvBf16Knhd {
    const ENTRY: &'static str = "owl_reshape_and_cache_fp8kv_bf16_knhd";
    const SRC: &'static str = "cu/attn/reshape_and_cache_fp8kv.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::BF16, 2),
        Param::t_in("value", Dtype::BF16, 2),
        Param::t_in("key_cache", Dtype::U8, 2),
        Param::t_in("value_cache", Dtype::U8, 2),
        Param::t_in("slot_mapping", Dtype::F32, 2),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_in("out", Dtype::BF16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_MAIN, LayoutTag::Knhd, Dtype::U8)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8), head_dim: &[128, 256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_paged_attention_v1_f16_hd128` —— pagedattention_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmPagedAttentionV1F16Hd128;

impl KernelSpec for VllmPagedAttentionV1F16Hd128 {
    const ENTRY: &'static str = "vllm_paged_attention_v1_f16_hd128";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("scale"),
        Param::i32("max_num_blocks_per_seq"),
        Param::i32("q_stride"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::f32("softscapping"),
        Param::i32("sliding_window"),
        Param::i32("use_alibi"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_paged_attention_v1_f16_hd128bs32` —— pagedattention_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmPagedAttentionV1F16Hd128bs32;

impl KernelSpec for VllmPagedAttentionV1F16Hd128bs32 {
    const ENTRY: &'static str = "vllm_paged_attention_v1_f16_hd128bs32";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("scale"),
        Param::i32("max_num_blocks_per_seq"),
        Param::i32("q_stride"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::f32("softscapping"),
        Param::i32("sliding_window"),
        Param::i32("use_alibi"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_paged_attention_v1_f16_hd256` —— pagedattention_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmPagedAttentionV1F16Hd256;

impl KernelSpec for VllmPagedAttentionV1F16Hd256 {
    const ENTRY: &'static str = "vllm_paged_attention_v1_f16_hd256";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("scale"),
        Param::i32("max_num_blocks_per_seq"),
        Param::i32("q_stride"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::f32("softscapping"),
        Param::i32("sliding_window"),
        Param::i32("use_alibi"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_paged_attention_v1_f16_hd256bs32` —— pagedattention_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmPagedAttentionV1F16Hd256bs32;

impl KernelSpec for VllmPagedAttentionV1F16Hd256bs32 {
    const ENTRY: &'static str = "vllm_paged_attention_v1_f16_hd256bs32";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("scale"),
        Param::i32("max_num_blocks_per_seq"),
        Param::i32("q_stride"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::f32("softscapping"),
        Param::i32("sliding_window"),
        Param::i32("use_alibi"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_chunked_prefill_paged_attn_opt_f16_hd128` —— prefill_paged_attn_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmChunkedPrefillPagedAttnOptF16Hd128;

impl KernelSpec for VllmChunkedPrefillPagedAttnOptF16Hd128 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_f16_hd128";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 2),
        Param::t_in("query_start_len", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::t_in("sinks", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("sm_scale"),
        Param::i32("block_table_stride"),
        Param::i32("num_seqs"),
        Param::i32("num_query_heads"),
        Param::i32("num_query_tokens"),
        Param::f32("softscapping"),
        Param::i32("o_stride_tokens"),
        Param::i32("sliding_window"),
        Param::i32("total_num_blocks"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("use_alibi_flag"),
        Param::i32("use_sinks_flag"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_chunked_prefill_paged_attn_opt_f16_hd256` —— prefill_paged_attn_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmChunkedPrefillPagedAttnOptF16Hd256;

impl KernelSpec for VllmChunkedPrefillPagedAttnOptF16Hd256 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_f16_hd256";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 2),
        Param::t_in("query_start_len", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::t_in("sinks", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("sm_scale"),
        Param::i32("block_table_stride"),
        Param::i32("num_seqs"),
        Param::i32("num_query_heads"),
        Param::i32("num_query_tokens"),
        Param::f32("softscapping"),
        Param::i32("o_stride_tokens"),
        Param::i32("sliding_window"),
        Param::i32("total_num_blocks"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("use_alibi_flag"),
        Param::i32("use_sinks_flag"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd128` —— prefill_paged_attn_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmChunkedPrefillPagedAttnOptF16KnhdHd128;

impl KernelSpec for VllmChunkedPrefillPagedAttnOptF16KnhdHd128 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd128";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 2),
        Param::t_in("query_start_len", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::t_in("sinks", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("sm_scale"),
        Param::i32("block_table_stride"),
        Param::i32("num_seqs"),
        Param::i32("num_query_heads"),
        Param::i32("num_query_tokens"),
        Param::f32("softscapping"),
        Param::i32("o_stride_tokens"),
        Param::i32("sliding_window"),
        Param::i32("total_num_blocks"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("use_alibi_flag"),
        Param::i32("use_sinks_flag"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd256` —— prefill_paged_attn_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmChunkedPrefillPagedAttnOptF16KnhdHd256;

impl KernelSpec for VllmChunkedPrefillPagedAttnOptF16KnhdHd256 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd256";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 2),
        Param::t_in("query_start_len", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::t_in("sinks", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("sm_scale"),
        Param::i32("block_table_stride"),
        Param::i32("num_seqs"),
        Param::i32("num_query_heads"),
        Param::i32("num_query_tokens"),
        Param::f32("softscapping"),
        Param::i32("o_stride_tokens"),
        Param::i32("sliding_window"),
        Param::i32("total_num_blocks"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("use_alibi_flag"),
        Param::i32("use_sinks_flag"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_chunked_prefill_paged_attn_opt_fp8_hd128` —— prefill_paged_attn_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmChunkedPrefillPagedAttnOptFp8Hd128;

impl KernelSpec for VllmChunkedPrefillPagedAttnOptFp8Hd128 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_fp8_hd128";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 2),
        Param::t_in("query_start_len", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::t_in("sinks", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("sm_scale"),
        Param::i32("block_table_stride"),
        Param::i32("num_seqs"),
        Param::i32("num_query_heads"),
        Param::i32("num_query_tokens"),
        Param::f32("softscapping"),
        Param::i32("o_stride_tokens"),
        Param::i32("sliding_window"),
        Param::i32("total_num_blocks"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("use_alibi_flag"),
        Param::i32("use_sinks_flag"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::U8)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_chunked_prefill_paged_attn_opt_fp8_hd256` —— prefill_paged_attn_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmChunkedPrefillPagedAttnOptFp8Hd256;

impl KernelSpec for VllmChunkedPrefillPagedAttnOptFp8Hd256 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_fp8_hd256";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 2),
        Param::t_in("query_start_len", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::t_in("sinks", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("sm_scale"),
        Param::i32("block_table_stride"),
        Param::i32("num_seqs"),
        Param::i32("num_query_heads"),
        Param::i32("num_query_tokens"),
        Param::f32("softscapping"),
        Param::i32("o_stride_tokens"),
        Param::i32("sliding_window"),
        Param::i32("total_num_blocks"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("use_alibi_flag"),
        Param::i32("use_sinks_flag"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::U8)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd128` —— prefill_paged_attn_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmChunkedPrefillPagedAttnOptFp8KnhdHd128;

impl KernelSpec for VllmChunkedPrefillPagedAttnOptFp8KnhdHd128 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd128";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 2),
        Param::t_in("query_start_len", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F32, 2),
        Param::t_in("sinks", Dtype::F32, 2),
        Param::i32("num_kv_heads"),
        Param::f32("sm_scale"),
        Param::i32("block_table_stride"),
        Param::i32("num_seqs"),
        Param::i32("num_query_heads"),
        Param::i32("num_query_tokens"),
        Param::f32("softscapping"),
        Param::i32("o_stride_tokens"),
        Param::i32("sliding_window"),
        Param::i32("total_num_blocks"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("use_alibi_flag"),
        Param::i32("use_sinks_flag"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::U8)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_prefill_split_f16_hd256` —— prefill_split_f16.cu(R1 补表;池声明见 POOL)
pub struct PrefillSplitF16Hd256;

impl KernelSpec for PrefillSplitF16Hd256 {
    const ENTRY: &'static str = "owl_prefill_split_f16_hd256";
    const SRC: &'static str = "cu/attn/prefill_split_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("scr_out", Dtype::F16, 2),
        Param::t_in("scr_stat", Dtype::F32, 2),
        Param::t_in("k0_alibi", Dtype::F16, 2),
        Param::f32("scale"),
        Param::i32("hkv"),
        Param::i32("T"),
        Param::i32("ctx_base"),
        Param::i32("nparts"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("page"),
        Param::i32("hq"),
        Param::t_in("dummy", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_prefill_split_fp8kv_hd256` —— prefill_split_f16.cu(R1 补表;池声明见 POOL)
pub struct PrefillSplitFp8KvHd256;

impl KernelSpec for PrefillSplitFp8KvHd256 {
    const ENTRY: &'static str = "owl_prefill_split_fp8kv_hd256";
    const SRC: &'static str = "cu/attn/prefill_split_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("scr_out", Dtype::F16, 2),
        Param::t_in("scr_stat", Dtype::F32, 2),
        Param::t_in("k0_alibi", Dtype::F16, 2),
        Param::f32("scale"),
        Param::i32("hkv"),
        Param::i32("T"),
        Param::i32("ctx_base"),
        Param::i32("nparts"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("page"),
        Param::i32("hq"),
        Param::t_in("dummy", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::U8)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_prefill_split_f16_knhd_hd256` —— prefill_split_f16.cu(R1 补表;池声明见 POOL)
pub struct PrefillSplitF16KnhdHd256;

impl KernelSpec for PrefillSplitF16KnhdHd256 {
    const ENTRY: &'static str = "owl_prefill_split_f16_knhd_hd256";
    const SRC: &'static str = "cu/attn/prefill_split_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("scr_out", Dtype::F16, 2),
        Param::t_in("scr_stat", Dtype::F32, 2),
        Param::t_in("k0_alibi", Dtype::F16, 2),
        Param::f32("scale"),
        Param::i32("hkv"),
        Param::i32("T"),
        Param::i32("ctx_base"),
        Param::i32("nparts"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("page"),
        Param::i32("hq"),
        Param::t_in("dummy", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_prefill_split_fp8kv_knhd_hd256` —— prefill_split_f16.cu(R1 补表;池声明见 POOL)
pub struct PrefillSplitFp8KvKnhdHd256;

impl KernelSpec for PrefillSplitFp8KvKnhdHd256 {
    const ENTRY: &'static str = "owl_prefill_split_fp8kv_knhd_hd256";
    const SRC: &'static str = "cu/attn/prefill_split_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 2),
        Param::t_in("v_cache", Dtype::F16, 2),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("scr_out", Dtype::F16, 2),
        Param::t_in("scr_stat", Dtype::F32, 2),
        Param::t_in("k0_alibi", Dtype::F16, 2),
        Param::f32("scale"),
        Param::i32("hkv"),
        Param::i32("T"),
        Param::i32("ctx_base"),
        Param::i32("nparts"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("page"),
        Param::i32("hq"),
        Param::t_in("dummy", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::U8)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::U8), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


/// `owl_prefill_split_reduce_f16_hd256` —— prefill_split_f16.cu(R1 补表;池声明见 POOL)
pub struct PrefillSplitReduceF16Hd256;

impl KernelSpec for PrefillSplitReduceF16Hd256 {
    const ENTRY: &'static str = "owl_prefill_split_reduce_f16_hd256";
    const SRC: &'static str = "cu/attn/prefill_split_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("split_alibi", Dtype::F16, 2),
        Param::t_in("scr_out", Dtype::F16, 2),
        Param::t_in("scr_stat", Dtype::F32, 2),
        Param::i32("nparts"),
        Param::i32("hq"),
        Param::i32("hd"),
        Param::i32("T"),
        Param::t_in("out", Dtype::F16, 2),
    ];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}


pub static OWL_NAIVE_DECODE_ATTN_F16: KernelMeta = meta_of::<NaiveDecodeAttnF16>();
pub static OWL_NAIVE_DECODE_ATTN_F32: KernelMeta = meta_of::<NaiveDecodeAttnF32>();
pub static OWL_NAIVE_ATTN_NC_F16: KernelMeta = meta_of::<NaiveAttnNcF16>();
pub static OWL_NAIVE_ATTN_NC_BF16: KernelMeta = meta_of::<NaiveAttnNcBf16>();
pub static OWL_NAIVE_ATTN_NC_FP8KV_F16: KernelMeta = meta_of::<NaiveAttnNcFp8KvF16>();
pub static VLLM_RESHAPE_AND_CACHE_F16: KernelMeta = meta_of::<VllmReshapeAndCacheF16>();
pub static VLLM_RESHAPE_AND_CACHE_F16_KNHD: KernelMeta = meta_of::<VllmReshapeAndCacheF16Knhd>();
pub static VLLM_RESHAPE_AND_CACHE_BF16_KNHD: KernelMeta = meta_of::<VllmReshapeAndCacheBf16Knhd>();
pub static OWL_RESHAPE_AND_CACHE_FP8KV_BF16: KernelMeta = meta_of::<ReshapeAndCacheFp8KvBf16>();
pub static OWL_RESHAPE_AND_CACHE_FP8KV_BF16_KNHD: KernelMeta = meta_of::<ReshapeAndCacheFp8KvBf16Knhd>();
pub static VLLM_PAGED_ATTENTION_V1_F16_HD128: KernelMeta = meta_of::<VllmPagedAttentionV1F16Hd128>();
pub static VLLM_PAGED_ATTENTION_V1_F16_HD128BS32: KernelMeta = meta_of::<VllmPagedAttentionV1F16Hd128bs32>();
pub static VLLM_PAGED_ATTENTION_V1_F16_HD256: KernelMeta = meta_of::<VllmPagedAttentionV1F16Hd256>();
pub static VLLM_PAGED_ATTENTION_V1_F16_HD256BS32: KernelMeta = meta_of::<VllmPagedAttentionV1F16Hd256bs32>();
pub static VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_F16_HD128: KernelMeta = meta_of::<VllmChunkedPrefillPagedAttnOptF16Hd128>();
pub static VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_F16_HD256: KernelMeta = meta_of::<VllmChunkedPrefillPagedAttnOptF16Hd256>();
pub static VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_F16_KNHD_HD128: KernelMeta = meta_of::<VllmChunkedPrefillPagedAttnOptF16KnhdHd128>();
pub static VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_F16_KNHD_HD256: KernelMeta = meta_of::<VllmChunkedPrefillPagedAttnOptF16KnhdHd256>();
pub static VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_FP8_HD128: KernelMeta = meta_of::<VllmChunkedPrefillPagedAttnOptFp8Hd128>();
pub static VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_FP8_HD256: KernelMeta = meta_of::<VllmChunkedPrefillPagedAttnOptFp8Hd256>();
pub static VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_FP8_KNHD_HD128: KernelMeta = meta_of::<VllmChunkedPrefillPagedAttnOptFp8KnhdHd128>();
pub static OWL_PREFILL_SPLIT_F16_HD256: KernelMeta = meta_of::<PrefillSplitF16Hd256>();
pub static OWL_PREFILL_SPLIT_FP8KV_HD256: KernelMeta = meta_of::<PrefillSplitFp8KvHd256>();
pub static OWL_PREFILL_SPLIT_F16_KNHD_HD256: KernelMeta = meta_of::<PrefillSplitF16KnhdHd256>();
pub static OWL_PREFILL_SPLIT_FP8KV_KNHD_HD256: KernelMeta = meta_of::<PrefillSplitFp8KvKnhdHd256>();
pub static OWL_PREFILL_SPLIT_REDUCE_F16_HD256: KernelMeta = meta_of::<PrefillSplitReduceF16Hd256>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[
    &NC_ATTN_BF16_V2,
    &PAGED_V2_FP8_KNHD_HD256BS32,
    &PAGED_V2_REDUCE_F16_HD256,
    &K0_WRITE_FP8_KNHD,
    &CHUNKED_PREFILL_FP8_KNHD_HD256,
    &K0_WRITE_FP8_CLASSIC,
    &K0_WRITE_BF16,
    &NC_ATTN_FP8KV_BF16,
    &NORM_ROPE_F16,
    &NORM_ROPE_BF16,
    // ---- R1 批次 1:paged v2 变体族 ----
    &PAGED_V2_F16_HD128BS32,
    &PAGED_V2_F16_HD256BS32,
    &PAGED_V2_F16_KNHD_HD128BS32,
    &PAGED_V2_F16_KNHD_HD256BS32,
    &PAGED_V2_F16_KNHD_HD128,
    &PAGED_V2_F16_KNHD_HD256,
    &PAGED_V2_FP8_HD128,
    &PAGED_V2_FP8_HD256,
    &PAGED_V2_FP8_HD128BS32,
    &PAGED_V2_FP8_HD256BS32,
    &PAGED_V2_FP8_KNHD_HD128BS32,
    &PAGED_V2_FP8_KNHD_HD256,
    &PAGED_V2_FP8_KNHD_HD128,
    &PAGED_V2_REDUCE_F16_HD128,
    &OWL_NAIVE_DECODE_ATTN_F16,
    &OWL_NAIVE_DECODE_ATTN_F32,
    &OWL_NAIVE_ATTN_NC_F16,
    &OWL_NAIVE_ATTN_NC_BF16,
    &OWL_NAIVE_ATTN_NC_FP8KV_F16,
    &VLLM_RESHAPE_AND_CACHE_F16,
    &VLLM_RESHAPE_AND_CACHE_F16_KNHD,
    &VLLM_RESHAPE_AND_CACHE_BF16_KNHD,
    &OWL_RESHAPE_AND_CACHE_FP8KV_BF16,
    &OWL_RESHAPE_AND_CACHE_FP8KV_BF16_KNHD,
    &VLLM_PAGED_ATTENTION_V1_F16_HD128,
    &VLLM_PAGED_ATTENTION_V1_F16_HD128BS32,
    &VLLM_PAGED_ATTENTION_V1_F16_HD256,
    &VLLM_PAGED_ATTENTION_V1_F16_HD256BS32,
    &VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_F16_HD128,
    &VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_F16_HD256,
    &VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_F16_KNHD_HD128,
    &VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_F16_KNHD_HD256,
    &VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_FP8_HD128,
    &VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_FP8_HD256,
    &VLLM_CHUNKED_PREFILL_PAGED_ATTN_OPT_FP8_KNHD_HD128,
    &OWL_PREFILL_SPLIT_F16_HD256,
    &OWL_PREFILL_SPLIT_FP8KV_HD256,
    &OWL_PREFILL_SPLIT_F16_KNHD_HD256,
    &OWL_PREFILL_SPLIT_FP8KV_KNHD_HD256,
    &OWL_PREFILL_SPLIT_REDUCE_F16_HD256,
];

/// **谓词 ⇔ 声明一致性**(P4:收编 driver 谓词 —— 页配对律此前只住
/// `driver::attn::paged_decode_ok`,现在与算子表的 `ShapeFamily` 声明互证:
/// 谓词说"可跑"的 (hd,page) 必须被声明接受;反之亦然)。
#[cfg(test)]
mod predicate_lock {
    use crate::driver;
    use crate::spec::{meta_of, ShapeQuery};
    use super::PagedAttnV2Fp8KnhdHd256Bs32;

    #[test]
    fn paged_decode_predicate_agrees_with_declaration() {
        let m = meta_of::<PagedAttnV2Fp8KnhdHd256Bs32>();
        for hd in [128usize, 256] {
            for page in [16usize, 32] {
                let ok = driver::attn::paged_decode_ok(hd, page);
                let accepted = m
                    .shape
                    .accepts(&ShapeQuery { head_dim: hd, page, grp: 0, tile: 0 })
                    .is_ok();
                // 本表只声明 hd256/page32 ⇒ 谓词为真但表不接受 = 该 (hd,page)
                // 走的是**别的变体**(未建表),不算矛盾;谓词为假却表接受才是雷
                if !ok {
                    assert!(!accepted, "谓词说不可跑,但 {hd}/{page} 被声明接受");
                }
                if accepted {
                    assert!(ok, "声明接受 {hd}/{page},但页配对谓词说不可跑");
                }
            }
        }
        // 本表声明的组合必须被谓词接受
        assert!(driver::attn::paged_decode_ok(256, 32));
    }
}

/// 编译期自证:槽序由 PARAMS 派生(本模块零手写 `"T,T,..."` 字符串)。
#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::meta_of;

    #[test]
    fn nc_v2_slot_order_derives_from_params() {
        assert_eq!(
            meta_of::<NcAttnBf16V2>().args_str(),
            "T,T,T,T,T,T,i32,i32,i32,i32,i32,O"
        );
    }

    #[test]
    fn paged_v2_slot_order_derives_from_params() {
        assert_eq!(
            meta_of::<PagedAttnV2Fp8KnhdHd256Bs32>().args_str(),
            "T,T,T,T,T,T,T,T,i32,f32,i32,i32,i32,i32,f32,i32,i32,O"
        );
        assert_eq!(
            meta_of::<PagedAttnV2ReduceF16Hd256>().args_str(),
            "T,T,T,T,i32,O"
        );
    }

    #[test]
    fn k0_slot_order_derives_from_params() {
        assert_eq!(
            meta_of::<K0WriteFp8Knhd>().args_str(),
            "T,T,T,T,T,i32,i32,i32,i32,i32,i32,O"
        );
    }

    /// 形状族是**声明**的:池 dtype/头维/页大小(拾取期几何门消费);
    /// **布局不在此**(单一住址 = `PoolAccess.layout`)
    #[test]
    fn k0_declares_shape_family() {
        let m = meta_of::<K0WriteFp8Knhd>();
        assert_eq!(m.shape.kv_dtype, Some(Dtype::U8));
        assert_eq!(m.shape.head_dim, &[128, 256]);
        assert_eq!(m.shape.page, &[16, 32, 64, 128]);
    }

    /// `ShapeRule::NONE` 是显式"无约束",不是漏写
    #[test]
    fn shape_rule_none_is_explicit() {
        assert!(crate::spec::ShapeRule::NONE.dims.is_empty());
    }
}
