//! attention 域生产算子(池写 / decode / 草稿 NC)。
//!
//! 参数名/序/类型 = `.cu` 签名的逐字抄录(机器互证见 `tests/op_tables.rs`)。

use owl_macros::kernel_tags;
use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, pools, CaptureEvidence, Determinism, KernelMeta, KernelSpec, LayoutTag, Param, PoolAccess};

// ============================================================================
// 草稿非因果块注意力(DFlash2 propose 的 NC 核 v2;2026-10-13 ③ 转正)
// ============================================================================

/// `owl_naive_attn_nc_bf16_v2` —— 非因果块注意力(自块直读 + 前缀池)。
///
/// v2 = 只换并行结构(lane→连续 dim + warp butterfly + split-K),语义/槽序
/// 与旧核逐字同一套;实测 prefix2048 2.238 → 1.201ms(1.86×),数值同档。
///
/// **2026-10-18(propose 立案):再收 GQA 冗余** —— 同 kv-head 的 `group` 个
/// q-head 原各起一个 CTA(同一批 K/V 行读 group 遍;草稿档 268 MB/发、~225 GB/s
/// ⇒ 1.19 ms/发,占 propose 图 66%)。现 **grid (T, Hq) → (T, Hkv)**、CTA 内
/// `wq = warp % group` / `sub = warp / group`,同组 warp 走 L1 共享同一行。
/// 结构门:驱动件要求 `nwarp ≥ group` 且整除(否则回落旧核);核内 `__trap()` 兜底。
pub struct NcAttnBf16V2;

#[kernel_tags("dom.attn", "role.attn", "feat.dflash", "feat.spec", "phase.propose", "kv.draft.bf16")]
impl KernelSpec for NcAttnBf16V2 {
    const ENTRY: &'static str = "owl_naive_attn_nc_bf16_v2";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::BF16, 2),
        Param::t_in("k_self", Dtype::BF16, 3),
        Param::t_in("v_self", Dtype::BF16, 3),
        Param::t_in("kc", Dtype::BF16, 5),
        Param::t_in("vc", Dtype::BF16, 4),
        // 整请求 KV 长(设备侧标量,f32 张量;非 host 标量 —— 图内可读)
        Param::t_in("kv_len_p", Dtype::F32, 0),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        // ABI 占位形参(历史遗留:q_tokens 已由 grid.y 承担;签名不动)
        Param::i32("x"),
        Param::i32("ring_cap"),
        Param::t_out("out", Dtype::BF16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_DRAFT, LayoutTag::Classic, Dtype::BF16)];
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> =
        Some(&crate::ops::oracles::nc_bf16_v2::NcBf16V2Oracle);
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        // 草稿池(bfloat16;classic 布局 —— 布局声明在 PoolAccess)
        // head_dim:草稿族定形 128(z-lab/syvai 同族)。**2026-10-19 收窄**:
        // 原声明 `[128, 256]` 把"主模型几何被拿来解析草稿"这种错配放行了
        // (hd=256 是目标模型的数,草稿永远不是)⇒ 静默错几何;现在错配在
        // boot 期就响亮拒(草稿自己的 env 见 SpecRuntime::draft_env)。
        kv_dtype: Some(Dtype::BF16),
        head_dim: &[128],
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

/// paged v2 主核参数表 —— **fp8 池臂**(18 参;f16/fp8 × classic/kNHD 同 ABI,
/// 池句柄视图不同 ⇒ 2026-10-15 拆两表)
const PAGED_V2_FP8_KNHD_PARAMS: &[Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        // fp8 主池句柄:引擎叶子 = U32 扁平字节账(`bytes/4` 字;池的 e4m3
        // 字节语义由 `POOL` 的 `Dtype::F8E4M3` 承担)。旧声明 U8x4 描述的是核内
        // kNHD 字节视图,与喂进来的句柄不是同一层(2026-10-15 事实面对齐)
        Param::t_in("k_cache", Dtype::U32, 1),
        Param::t_in("v_cache", Dtype::U32, 1),
        Param::t_in("block_tables", Dtype::F32, 2),
        // 每序列 KV 长:视图随臂变(计划输入槽 rank1 / 会话表 rank2)⇒ 显式阶数无关
        Param::t_in("context_lens", Dtype::F32, 0),
        // 哑槽(use_alibi=0 不解引用):喂 K0 哑输出 = 本族激活 dtype(f16)
        Param::t_in("alibi_slopes", Dtype::F16, 0),
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
        // 引擎 scratch 叶子 = 扁平 `[hq·nparts·hd]`(rank1)
        Param::t_out("tmp_out", Dtype::F16, 1),
];

#[kernel_tags("dom.attn", "role.attn", "role.kv_read", "phase.decode", "kv.main.fp8")]
impl KernelSpec for PagedAttnV2Fp8KnhdHd256Bs32 {
    const ENTRY: &'static str = "vllm_paged_attention_v2_fp8_knhd_hd256bs32";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = PAGED_V2_FP8_KNHD_PARAMS;
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F8E4M3)];
}

/// `vllm_paged_attention_v2_reduce_f16_hd256` —— 分区 partial 归并
/// (exp_sums/max_logits + tmp_out → out;裸 decode 第二发)。
pub struct PagedAttnV2ReduceF16Hd256;

#[kernel_tags("dom.attn", "role.attn", "phase.decode")]
impl KernelSpec for PagedAttnV2ReduceF16Hd256 {
    const ENTRY: &'static str = "vllm_paged_attention_v2_reduce_f16_hd256";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("exp_sums", Dtype::F32, 2),
        Param::t_in("max_logits", Dtype::F32, 2),
        // 引擎 scratch 叶子 = 扁平 `[hq·nparts·hd]`(rank1)
        Param::t_in("tmp_out", Dtype::F16, 1),
        // 每序列 KV 长:视图随臂变(计划输入槽 rank1 / 会话表 rank2)⇒ 显式阶数无关
        Param::t_in("context_lens", Dtype::F32, 0),
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

#[kernel_tags("dom.attn", "role.attn", "role.kv_write", "phase.prefill", "kv.main.fp8")]
impl KernelSpec for K0WriteFp8Knhd {
    const ENTRY: &'static str = "owl_reshape_and_cache_fp8kv_knhd";
    const SRC: &'static str = "cu/attn/reshape_and_cache_fp8kv.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::F16, 3),
        Param::t_in("value", Dtype::F16, 3),
        Param::t_in("key_cache", Dtype::U32, 1),
        Param::t_in("value_cache", Dtype::U32, 1),
        Param::t_in("slot_mapping", Dtype::F32, 0),
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
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F8E4M3)];
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> =
        Some(&crate::ops::oracles::k0_fp8_knhd::K0Fp8KnhdOracle);
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3),
        head_dim: &[128, 256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}

// ============================================================================
// INT8 KV 专项 P0:int8 池写者(kNHD + per-token/head fp32 scale)
// ============================================================================

/// `owl_reshape_and_cache_int8_knhd` —— 批量 KV 写池(f16 入 → int8 池 +
/// fp32 scale 平面)。量化约定 = pth/vLLM `int8_per_token_head` writer 的镜像。
///
/// **FIXME(int8-kv)**:scale 目前作为独立 tensor 传入,尚未纳入池写集审计;
/// 接生产池时必须把 scale plane 加进 `POOL`/E10 写者穷举。
pub struct K0WriteInt8Knhd;

#[kernel_tags(
    "dom.attn",
    "role.attn",
    "role.kv_write",
    "phase.prefill",
    "kv.main.int8"
)]
impl KernelSpec for K0WriteInt8Knhd {
    const ENTRY: &'static str = "owl_reshape_and_cache_int8_knhd";
    const SRC: &'static str = "cu/attn/reshape_and_cache_int8.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::F16, 3),
        Param::t_in("value", Dtype::F16, 3),
        Param::t_in("key_cache", Dtype::I8, 1),
        Param::t_in("value_cache", Dtype::I8, 1),
        Param::t_in("k_scale", Dtype::F32, 1),
        Param::t_in("v_scale", Dtype::F32, 1),
        Param::t_in("slot_mapping", Dtype::F32, 0),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("num_tokens"),
        Param::t_out("out", Dtype::F16, 1),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(
        pools::KV_MAIN,
        LayoutTag::Knhd,
        Dtype::I8,
    )];
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> =
        Some(&crate::ops::oracles::k0_int8_knhd::K0Int8KnhdOracle);
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[128, 256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}

// ============================================================================
// INT8 KV 专项:4-warp int8 paged prefill(K int8 + V int8 prepacked + P uint8)
// ============================================================================

/// `owl_int8_prefill_paged` —— 4 warp / Bc=32 / Br=64 / D=256 的 int8 预填核。
/// K 走 int8 张量核;P 量化 uint8;V 走预打包 `[Hkv][D][L/2]` b16。
/// **无 V f16 shadow**。
///
/// **FIXME(int8-kv)**:V prepacked 布局尚无 `LayoutTag`;目前只声明 K 的主池读。
/// 接生产池前必须补齐 V/scale 的池访问声明与 writer。
pub struct Int8PrefillPaged;

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.int8")]
impl KernelSpec for Int8PrefillPaged {
    const ENTRY: &'static str = "owl_int8_prefill_paged";
    const SRC: &'static str = "cu/attn/int8_prefill.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::I8, 2),
        Param::t_in("qscale", Dtype::F32, 1),
        Param::t_in("k", Dtype::I8, 1),
        Param::t_in("kscale", Dtype::F32, 1),
        Param::t_in("vp", Dtype::I8, 1),
        Param::t_in("vscale", Dtype::F32, 1),
        Param::t_in("block_table", Dtype::F32, 2),
        Param::t_out("out", Dtype::F16, 2),
        Param::i32("T"),
        Param::i32("L"),
        Param::i32("Hq"),
        Param::i32("Hkv"),
        Param::i32("page"),
        Param::i32("base_pos"),
        Param::t_in("dep_k", Dtype::F16, 0),
        Param::t_in("dep_v", Dtype::F16, 0),
    ];
    const POOL: &'static [PoolAccess] =
        &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::I8)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

/// `owl_int8_prefill_paged_f16pv` —— int8 QK + f16 PV 的生产核:
/// P f16×2^8、V f16 shadow(token-major)、f16 accumulator。
pub struct Int8PrefillPagedF16Pv;

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.int8")]
impl KernelSpec for Int8PrefillPagedF16Pv {
    const ENTRY: &'static str = "owl_int8_prefill_paged_f16pv";
    const SRC: &'static str = "cu/attn/int8_prefill_f16pv.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::I8, 2),
        Param::t_in("qscale", Dtype::F32, 1),
        Param::t_in("k", Dtype::I8, 1),
        Param::t_in("kscale", Dtype::F32, 1),
        Param::t_in("v16", Dtype::F16, 1),
        Param::t_in("block_table", Dtype::F32, 2),
        Param::t_out("out", Dtype::F16, 2),
        Param::i32("T"),
        Param::i32("L"),
        Param::i32("Hq"),
        Param::i32("Hkv"),
        Param::i32("page"),
        Param::i32("base_pos"),
        Param::t_in("dep_k", Dtype::F16, 0),
        Param::t_in("dep_v", Dtype::F16, 0),
    ];
    const POOL: &'static [PoolAccess] =
        &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::I8)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

/// `owl_reshape_and_cache_int8_k_only_knhd` —— 只写 K 的 int8 writer。
///
/// int8 PV 的 V 走 per-page pack(见 `VPackInt8Prefill`/`VAppendInt8Paged`),
/// 因此 prefill/decode 都不应再让 K writer 顺带写一份 V int8(白写 + 白占池)。
pub struct K0WriteInt8KOnlyKnhd;

#[kernel_tags(
    "dom.attn",
    "role.attn",
    "role.kv_write",
    "phase.prefill",
    "kv.main.int8"
)]
impl KernelSpec for K0WriteInt8KOnlyKnhd {
    const ENTRY: &'static str = "owl_reshape_and_cache_int8_k_only_knhd";
    const SRC: &'static str = "cu/attn/reshape_and_cache_int8.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::F16, 3),
        Param::t_in("key_cache", Dtype::I8, 1),
        Param::t_in("k_scale", Dtype::F32, 1),
        Param::t_in("slot_mapping", Dtype::F32, 0),
        Param::i32("key_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("num_tokens"),
        Param::t_out("out", Dtype::F16, 1),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(
        pools::KV_MAIN,
        LayoutTag::Knhd,
        Dtype::I8,
    )];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

/// `owl_quant_int8_qk_f16` —— Q 的 per-(token, q_head) int8 量化。
/// 输出槽 `scales` 由调用方备块;`q` 是节点输出。
pub struct QuantInt8QkF16;

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "phase.decode")]
impl KernelSpec for QuantInt8QkF16 {
    const ENTRY: &'static str = "owl_quant_int8_qk_f16";
    const SRC: &'static str = "cu/attn/quant_int8_qk.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::i32("rows"),
        Param::i32("cols"),
        Param::t_in("scales", Dtype::F32, 1),
        Param::t_out("q", Dtype::I8, 2),
    ];
    // 输出是 int8:有限性律不适用(饱和由核内 clamp 保证)。
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

/// `owl_v_pack_int8_prefill` —— prefill 整页 V scale + 打包。
pub struct VPackInt8Prefill;

#[kernel_tags(
    "dom.attn",
    "role.attn",
    "role.kv_write",
    "phase.prefill",
    "kv.main.int8"
)]
impl KernelSpec for VPackInt8Prefill {
    const ENTRY: &'static str = "owl_v_pack_int8_prefill";
    const SRC: &'static str = "cu/attn/pack_v_int8_paged.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("v", Dtype::F16, 3),
        Param::t_in("slots", Dtype::F32, 0),
        Param::i32("T"),
        Param::i32("Hkv"),
        Param::i32("D"),
        Param::i32("page"),
        Param::t_in("vscale", Dtype::F32, 1),
        Param::t_in("vp", Dtype::I8, 1),
        Param::t_out("out", Dtype::F16, 1),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(
        pools::KV_MAIN,
        LayoutTag::Knhd,
        Dtype::I8,
    )];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

/// `owl_v_append_int8_paged` —— decode 单 token 追加 V。
pub struct VAppendInt8Paged;

#[kernel_tags(
    "dom.attn",
    "role.attn",
    "role.kv_write",
    "phase.decode",
    "kv.main.int8"
)]
impl KernelSpec for VAppendInt8Paged {
    const ENTRY: &'static str = "owl_v_append_int8_paged";
    const SRC: &'static str = "cu/attn/pack_v_int8_paged.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("v", Dtype::F16, 3),
        Param::t_in("slots", Dtype::F32, 0),
        Param::i32("T"),
        Param::i32("Hkv"),
        Param::i32("D"),
        Param::i32("page"),
        Param::t_in("vscale", Dtype::F32, 1),
        Param::t_in("vp", Dtype::I8, 1),
        Param::t_out("out", Dtype::F16, 1),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(
        pools::KV_MAIN,
        LayoutTag::Knhd,
        Dtype::I8,
    )];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

/// `owl_int8_decode_paged` —— 单 token decode(K int8 / V packed / per-page scale)。
pub struct Int8DecodePaged;

#[kernel_tags("dom.attn", "role.attn", "phase.decode", "kv.main.int8")]
impl KernelSpec for Int8DecodePaged {
    const ENTRY: &'static str = "owl_int8_decode_paged";
    const SRC: &'static str = "cu/attn/int8_decode.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::I8, 2),
        Param::t_in("qscale", Dtype::F32, 1),
        Param::t_in("k", Dtype::I8, 1),
        Param::t_in("kscale", Dtype::F32, 1),
        Param::t_in("vp", Dtype::I8, 1),
        Param::t_in("vscale", Dtype::F32, 1),
        Param::t_in("block_table", Dtype::F32, 2),
        Param::t_in("kv_lens", Dtype::F32, 0),
        Param::t_out("out", Dtype::F32, 2),
        Param::i32("B"),
        Param::i32("Hq"),
        Param::i32("Hkv"),
        Param::i32("page"),
        Param::i32("bt_stride"),
        Param::t_in("dep_k", Dtype::F16, 0),
        Param::t_in("dep_v", Dtype::F16, 0),
    ];
    const POOL: &'static [PoolAccess] =
        &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::I8)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

/// `owl_int8_decode_paged_split` —— decode 的 KV 分片 partial。
pub struct Int8DecodeSplit;

#[kernel_tags("dom.attn", "role.attn", "phase.decode", "kv.main.int8")]
impl KernelSpec for Int8DecodeSplit {
    const ENTRY: &'static str = "owl_int8_decode_paged_split";
    const SRC: &'static str = "cu/attn/int8_decode.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::I8, 2),
        Param::t_in("qscale", Dtype::F32, 1),
        Param::t_in("k", Dtype::I8, 1),
        Param::t_in("kscale", Dtype::F32, 1),
        Param::t_in("vp", Dtype::I8, 1),
        Param::t_in("vscale", Dtype::F32, 1),
        Param::t_in("block_table", Dtype::F32, 2),
        Param::t_in("kv_lens", Dtype::F32, 0),
        Param::t_in("partial_out", Dtype::F32, 1),
        Param::t_in("partial_stat", Dtype::F32, 1),
        Param::t_out("out_dummy", Dtype::F16, 1),
        Param::i32("B"),
        Param::i32("Hq"),
        Param::i32("Hkv"),
        Param::i32("page"),
        Param::i32("bt_stride"),
        Param::i32("nparts"),
        Param::t_in("dep_k", Dtype::F16, 0),
        Param::t_in("dep_v", Dtype::F16, 0),
    ];
    const POOL: &'static [PoolAccess] =
        &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::I8)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

/// `owl_int8_decode_gqa_tc_split` —— GQA 共享 K/V + tensor-core QK/PV 的 decode 分片。
pub struct Int8DecodeGqaTcSplit;

#[kernel_tags("dom.attn", "role.attn", "phase.decode", "kv.main.int8")]
impl KernelSpec for Int8DecodeGqaTcSplit {
    const ENTRY: &'static str = "owl_int8_decode_gqa_tc_split";
    const SRC: &'static str = "cu/attn/int8_decode.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::I8, 2),
        Param::t_in("qscale", Dtype::F32, 1),
        Param::t_in("k", Dtype::I8, 1),
        Param::t_in("kscale", Dtype::F32, 1),
        Param::t_in("vp", Dtype::I8, 1),
        Param::t_in("vscale", Dtype::F32, 1),
        Param::t_in("block_table", Dtype::F32, 2),
        Param::t_in("kv_lens", Dtype::F32, 0),
        Param::t_in("partial_out", Dtype::F32, 1),
        Param::t_in("partial_stat", Dtype::F32, 1),
        Param::t_out("out_dummy", Dtype::F16, 1),
        Param::i32("B"),
        Param::i32("Hq"),
        Param::i32("Hkv"),
        Param::i32("page"),
        Param::i32("bt_stride"),
        Param::i32("nparts"),
        Param::t_in("dep_k", Dtype::F16, 0),
        Param::t_in("dep_v", Dtype::F16, 0),
    ];
    const POOL: &'static [PoolAccess] =
        &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::I8)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

/// `owl_int8_decode_reduce` —— 合并 split partial(在线 softmax LSE)。
pub struct Int8DecodeReduce;

#[kernel_tags("dom.attn", "role.attn", "phase.decode")]
impl KernelSpec for Int8DecodeReduce {
    const ENTRY: &'static str = "owl_int8_decode_reduce";
    const SRC: &'static str = "cu/attn/int8_decode.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("partial_out", Dtype::F32, 1),
        Param::t_in("partial_stat", Dtype::F32, 1),
        Param::t_in("dep", Dtype::F16, 0),
        Param::t_out("out", Dtype::F16, 2),
        Param::i32("B"),
        Param::i32("Hq"),
        Param::i32("nparts"),
        Param::i32("D"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
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

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.fp8")]
impl KernelSpec for ChunkedPrefillFp8KnhdHd256 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd256";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::U32, 1),
        Param::t_in("v_cache", Dtype::U32, 1),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 0),
        Param::t_in("query_start_len", Dtype::F32, 0),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::t_in("sinks", Dtype::F32, 0),
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
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F8E4M3)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3),
        head_dim: &[256],
        page: &[16, 32, 64, 128],
        grp: None,
        tile: None,
    };
}

/// `owl_reshape_and_cache_fp8kv` —— K0 写池(**classic** 寻址 + fp8 池)。
/// 与 kNHD 变体同参数表,只有寻址语义不同。**生产归属 = 草稿池**
/// (`K0WriteFp8`,f16 输入);主池生产路径已统一 kNHD,classic 主池只作历史/
/// 测试面保留,不再同时声明为主池写者(E14 双寻址)。
pub struct K0WriteFp8Classic;

#[kernel_tags("dom.attn", "role.attn", "role.kv_write", "phase.prefill", "kv.draft.fp8")]
impl KernelSpec for K0WriteFp8Classic {
    const ENTRY: &'static str = "owl_reshape_and_cache_fp8kv";
    const SRC: &'static str = "cu/attn/reshape_and_cache_fp8kv.cu";
    const PARAMS: &'static [Param] = K0_PARAMS;
    // 主池生产路径已统一 KnHD；classic 臂只服务草稿池（K0WriteFp8）。
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(
        pools::KV_DRAFT,
        LayoutTag::Classic,
        Dtype::F8E4M3,
    )];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3),
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
    Param::t_in("key_cache", Dtype::F8E4M3, 4),
    Param::t_in("value_cache", Dtype::F8E4M3, 4),
    Param::t_in("slot_mapping", Dtype::F32, 2),
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

#[kernel_tags("dom.attn", "role.attn", "role.kv_write", "phase.prefill", "kv.draft.bf16")]
impl KernelSpec for K0WriteBf16 {
    const ENTRY: &'static str = "vllm_reshape_and_cache_bf16";
    const SRC: &'static str = "cu/attn/reshape_and_cache.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::BF16, 3),
        Param::t_in("value", Dtype::BF16, 3),
        Param::t_in("key_cache", Dtype::BF16, 5),
        Param::t_in("value_cache", Dtype::BF16, 4),
        Param::t_in("slot_mapping", Dtype::F32, 0),
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

#[kernel_tags("dom.attn", "role.attn", "feat.dflash", "feat.spec", "phase.propose", "kv.draft.fp8")]
impl KernelSpec for NcAttnFp8kvBf16 {
    const ENTRY: &'static str = "owl_naive_attn_nc_fp8kv_bf16";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::BF16, 2),
        Param::t_in("k_self", Dtype::BF16, 3),
        Param::t_in("v_self", Dtype::BF16, 3),
        Param::t_in("kc", Dtype::U32, 1),
        Param::t_in("vc", Dtype::U32, 1),
        Param::t_in("kv_len_p", Dtype::F32, 0),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::i32("ring_cap"),
        Param::t_out("out", Dtype::BF16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_DRAFT, LayoutTag::Classic, Dtype::F8E4M3)];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        // 草稿池 fp8 档(前缀池 e4m3 字节;自块直读保持 bf16)
        kv_dtype: Some(Dtype::F8E4M3),
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

#[kernel_tags("dom.attn", "role.attn", "role.norm", "role.rope")]
impl KernelSpec for NormRopeF16 {
    const ENTRY: &'static str = "owl_norm_rope_f16";
    const SRC: &'static str = "cu/attn/norm_rope.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("w", Dtype::F16, 1),
        Param::t_in("cos_t", Dtype::F16, 2),
        Param::t_in("sin_t", Dtype::F16, 2),
        Param::t_in("pos", Dtype::F32, 0),
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

#[kernel_tags("dom.attn", "role.attn", "role.norm", "role.rope")]
impl KernelSpec for NormRopeBf16 {
    const ENTRY: &'static str = "owl_norm_rope_bf16";
    const SRC: &'static str = "cu/attn/norm_rope.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::t_in("w", Dtype::BF16, 1),
        Param::t_in("cos_t", Dtype::F16, 2),
        Param::t_in("sin_t", Dtype::F16, 2),
        Param::t_in("pos", Dtype::F32, 0),
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

/// `owl_norm_rope_q_int8_f16` —— norm+rope 后直接 Q int8 量化。
pub struct NormRopeQInt8F16;

#[kernel_tags("dom.attn", "role.attn", "role.norm", "role.rope", "kv.main.int8")]
impl KernelSpec for NormRopeQInt8F16 {
    const ENTRY: &'static str = "owl_norm_rope_q_int8_f16";
    const SRC: &'static str = "cu/attn/norm_rope.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("w", Dtype::F16, 1),
        Param::t_in("cos_t", Dtype::F16, 2),
        Param::t_in("sin_t", Dtype::F16, 2),
        Param::t_in("pos", Dtype::F32, 0),
        Param::f32("eps"),
        Param::sz("row_stride"),
        Param::sz("head_stride"),
        Param::sz("half"),
        Param::i32("w_off"),
        Param::t_in("qscale", Dtype::F32, 1),
        Param::t_out("q", Dtype::I8, 2),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

/// `owl_norm_rope_k_int8_write_f16` —— norm+rope 后直接写 K int8 池 + scale。
pub struct NormRopeKInt8WriteF16;

#[kernel_tags(
    "dom.attn",
    "role.attn",
    "role.norm",
    "role.rope",
    "role.kv_write",
    "kv.main.int8"
)]
impl KernelSpec for NormRopeKInt8WriteF16 {
    const ENTRY: &'static str = "owl_norm_rope_k_int8_write_f16";
    const SRC: &'static str = "cu/attn/norm_rope.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 3),
        Param::t_in("w", Dtype::F16, 1),
        Param::t_in("cos_t", Dtype::F16, 2),
        Param::t_in("sin_t", Dtype::F16, 2),
        Param::t_in("pos", Dtype::F32, 0),
        Param::f32("eps"),
        Param::sz("row_stride"),
        Param::sz("head_stride"),
        Param::sz("half"),
        Param::i32("w_off"),
        Param::t_in("slots", Dtype::F32, 0),
        Param::t_in("k_cache", Dtype::I8, 1),
        Param::t_in("k_scale", Dtype::F32, 1),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::t_out("out", Dtype::F16, 1),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(
        pools::KV_MAIN,
        LayoutTag::Knhd,
        Dtype::I8,
    )];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::I8),
        head_dim: &[256],
        page: &[32],
        grp: None,
        tile: None,
    };
}

// ============================================================================
// 命名 static(**单一提升点**;域表与族表共用同一份 KernelMeta)
// ============================================================================

pub static NC_ATTN_BF16_V2: KernelMeta = meta_of::<NcAttnBf16V2>();
pub static NC_ATTN_TILED_BF16: KernelMeta = meta_of::<NcAttnTiledBf16>();
pub static NC_ATTN_TILED_F16: KernelMeta = meta_of::<NcAttnTiledF16>();
pub static NC_ATTN_TILED_FP8KV_BF16: KernelMeta = meta_of::<NcAttnTiledFp8kvBf16>();
pub static NC_ATTN_TILED_FP8KV_F16: KernelMeta = meta_of::<NcAttnTiledFp8kvF16>();
pub static PAGED_V2_FP8_KNHD_HD256BS32: KernelMeta = meta_of::<PagedAttnV2Fp8KnhdHd256Bs32>();
pub static PAGED_V2_REDUCE_F16_HD256: KernelMeta = meta_of::<PagedAttnV2ReduceF16Hd256>();
pub static K0_WRITE_FP8_KNHD: KernelMeta = meta_of::<K0WriteFp8Knhd>();
pub static K0_WRITE_INT8_KNHD: KernelMeta = meta_of::<K0WriteInt8Knhd>();
pub static K0_WRITE_INT8_K_ONLY_KNHD: KernelMeta = meta_of::<K0WriteInt8KOnlyKnhd>();
pub static QUANT_INT8_QK_F16: KernelMeta = meta_of::<QuantInt8QkF16>();
pub static V_PACK_INT8_PREFILL: KernelMeta = meta_of::<VPackInt8Prefill>();
pub static V_APPEND_INT8_PAGED: KernelMeta = meta_of::<VAppendInt8Paged>();
pub static INT8_DECODE_PAGED: KernelMeta = meta_of::<Int8DecodePaged>();
pub static INT8_DECODE_SPLIT: KernelMeta = meta_of::<Int8DecodeSplit>();
pub static INT8_DECODE_GQA_TC_SPLIT: KernelMeta = meta_of::<Int8DecodeGqaTcSplit>();
pub static INT8_DECODE_REDUCE: KernelMeta = meta_of::<Int8DecodeReduce>();
pub static OWL_INT8_PREFILL_PAGED: KernelMeta = meta_of::<Int8PrefillPaged>();
pub static OWL_INT8_PREFILL_PAGED_F16PV: KernelMeta = meta_of::<Int8PrefillPagedF16Pv>();
pub static CHUNKED_PREFILL_FP8_KNHD_HD256: KernelMeta = meta_of::<ChunkedPrefillFp8KnhdHd256>();
pub static K0_WRITE_FP8_CLASSIC: KernelMeta = meta_of::<K0WriteFp8Classic>();
pub static K0_WRITE_BF16: KernelMeta = meta_of::<K0WriteBf16>();
pub static NC_ATTN_FP8KV_BF16: KernelMeta = meta_of::<NcAttnFp8kvBf16>();
pub static NORM_ROPE_F16: KernelMeta = meta_of::<NormRopeF16>();
pub static NORM_ROPE_Q_INT8_F16: KernelMeta = meta_of::<NormRopeQInt8F16>();
pub static NORM_ROPE_K_INT8_WRITE_F16: KernelMeta = meta_of::<NormRopeKInt8WriteF16>();
pub static NORM_ROPE_BF16: KernelMeta = meta_of::<NormRopeBf16>();

// ============================================================================
// R1 批次 1:attention 池读写变体族(用户裁决 2026-10-13:凡是用到都必须有声明)
// —— 共享参数表(同 ABI 只差 ENTRY/池 dtype/几何声明),逐表由 `.cu` 互证门对账
// ============================================================================

/// paged v2 主核参数表 —— **f16 池臂**(池句柄 = kNHD 分页视图)
const PAGED_V2_F16_KNHD_PARAMS: &[Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        // f16 主池:引擎叶子 = [nb, hkv, hd/x, page, x](k)/[nb, hkv, hd, page](v)
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        // 每序列 KV 长:视图随臂变(计划输入槽 rank1 / 会话表 rank2)⇒ 显式阶数无关
        Param::t_in("context_lens", Dtype::F32, 0),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
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
        Param::t_out("tmp_out", Dtype::F16, 1),
];
/// reduce 参数表(6 参;hd128/hd256 同 ABI)
const PAGED_V2_REDUCE_PARAMS: &[Param] = &[
    Param::t_in("exp_sums", Dtype::F32, 2),
    Param::t_in("max_logits", Dtype::F32, 2),
    Param::t_in("tmp_out", Dtype::F16, 1),
    // 每序列 KV 长:视图随臂变(计划输入槽 rank1 / 会话表 rank2)⇒ 显式阶数无关
        Param::t_in("context_lens", Dtype::F32, 0),
    Param::i32("max_num_partitions"),
    Param::t_out("out", Dtype::F16, 2),
];

/// paged v2 变体描述符(表里只写差异项;ABI 由共享参数表承担)
macro_rules! paged_v2_variant {
    ($ty:ident, $entry:literal, $pool_dt:expr, $hd:expr, $page:expr, $dims:expr, $kv_tag:literal) => {
        #[doc = concat!("`", $entry, "`(paged attention v2 变体;ABI = PAGED_V2_PARAMS)")]
        pub struct $ty;

        #[kernel_tags("dom.attn", "role.attn", "role.kv_read", "phase.decode", $kv_tag)]
        impl KernelSpec for $ty {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
            // 池句柄视图随池 dtype 分家(fp8 = U32 扁平字节账;f16 = kNHD 分页视图)
            const PARAMS: &'static [Param] = match $pool_dt {
                Dtype::F8E4M3 => PAGED_V2_FP8_KNHD_PARAMS,
                _ => PAGED_V2_F16_KNHD_PARAMS,
            };
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

paged_v2_variant!(PagedV2F16Hd128Bs32, "vllm_paged_attention_v2_f16_hd128bs32", Dtype::F16, 128, &[32], &[128], "kv.main.f16");
paged_v2_variant!(PagedV2F16Hd256Bs32, "vllm_paged_attention_v2_f16_hd256bs32", Dtype::F16, 256, &[32], &[256], "kv.main.f16");
paged_v2_variant!(PagedV2F16KnhdHd128Bs32, "vllm_paged_attention_v2_f16_knhd_hd128bs32", Dtype::F16, 128, &[32], &[128], "kv.main.f16");
paged_v2_variant!(PagedV2F16KnhdHd256Bs32, "vllm_paged_attention_v2_f16_knhd_hd256bs32", Dtype::F16, 256, &[32], &[256], "kv.main.f16");
paged_v2_variant!(PagedV2F16KnhdHd128, "vllm_paged_attention_v2_f16_knhd_hd128", Dtype::F16, 128, &[16], &[128], "kv.main.f16");
paged_v2_variant!(PagedV2F16KnhdHd256, "vllm_paged_attention_v2_f16_knhd_hd256", Dtype::F16, 256, &[16], &[256], "kv.main.f16");
paged_v2_variant!(PagedV2Fp8Hd128, "vllm_paged_attention_v2_fp8_hd128", Dtype::F8E4M3, 128, &[16], &[128], "kv.main.fp8");
paged_v2_variant!(PagedV2Fp8Hd256, "vllm_paged_attention_v2_fp8_hd256", Dtype::F8E4M3, 256, &[16], &[256], "kv.main.fp8");
paged_v2_variant!(PagedV2Fp8Hd128Bs32, "vllm_paged_attention_v2_fp8_hd128bs32", Dtype::F8E4M3, 128, &[32], &[128], "kv.main.fp8");
paged_v2_variant!(PagedV2Fp8Hd256Bs32, "vllm_paged_attention_v2_fp8_hd256bs32", Dtype::F8E4M3, 256, &[32], &[256], "kv.main.fp8");
paged_v2_variant!(PagedV2Fp8KnhdHd128Bs32, "vllm_paged_attention_v2_fp8_knhd_hd128bs32", Dtype::F8E4M3, 128, &[32], &[128], "kv.main.fp8");
paged_v2_variant!(PagedV2Fp8KnhdHd256, "vllm_paged_attention_v2_fp8_knhd_hd256", Dtype::F8E4M3, 256, &[16], &[256], "kv.main.fp8");
paged_v2_variant!(PagedV2Fp8KnhdHd128, "vllm_paged_attention_v2_fp8_knhd_hd128", Dtype::F8E4M3, 128, &[16], &[128], "kv.main.fp8");

/// `vllm_paged_attention_v2_reduce_f16_hd128` —— 分区归并(hd128 变体)
pub struct PagedV2ReduceF16Hd128;

#[kernel_tags("dom.attn", "role.attn", "phase.decode")]
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

#[kernel_tags("dom.attn", "role.attn", "phase.decode", "kv.main.f16")]
impl KernelSpec for NaiveDecodeAttnF16 {
    const ENTRY: &'static str = "owl_naive_decode_attn_f16";
    const SRC: &'static str = "cu/attn/naive_decode_attn.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        // **契约(2026-10-16 B6 收敛):k/v r3 = [tokens, hkv, hd]。
        // 两条活路已统一到同一口径:层 decode 路 `reshape([tokens,hkv,hd])`;
        // prefill token-loop 路的窄视图也改成 `[1,hkv,hd]`(纯元数据;核按
        // 显式标量寻址,不吃 rank)。
        Param::t_in("k", Dtype::F16, 3),
        Param::t_in("v", Dtype::F16, 3),
        // kc/vc = **池叶**:形状随池表示而变(legacy r3 `[nb,hkv,hd]` / paged K r5
        // / paged V r4 —— 页=1 时内存等价),且核按显式标量寻址。
        // 契约单源 = 引擎 `StatePool::{k_shape,v_shape}`(池侧声明见 POOL),
        // 故本二槽不约束 rank(r0)—— 不是"两套口径",是"表示相关槽"。
        Param::t_in("kc", Dtype::F16, 0),
        Param::t_in("vc", Dtype::F16, 0),
        // 契约:slots/kv_lens r2([1, n] —— 池叶与狭窄视图同秩;2026-10-16 B6 收敛,
        // 秩以**实测线格式事实**定:5 处活调用点均为 r2)
        Param::t_in("slots", Dtype::F32, 2),
        Param::t_in("kv_lens", Dtype::F32, 2),
        Param::sz("bs"),
        Param::sz("q_heads"),
        Param::sz("kv_heads"),
        Param::sz("d_dim"),
        Param::t_out("out", Dtype::F16, 2),
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

#[kernel_tags("dom.attn", "role.attn", "phase.decode")]
impl KernelSpec for NaiveDecodeAttnF32 {
    const ENTRY: &'static str = "owl_naive_decode_attn_f32";
    const SRC: &'static str = "cu/attn/naive_decode_attn.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F32, 2),
        // 契约:k/v r3 = [tokens, hkv, hd](与 f16 臂同式;2026-10-16 B6 收敛,
        // 两条活路已统一 —— 旧声明 r2 与真调用不符)
        Param::t_in("k", Dtype::F32, 3),
        Param::t_in("v", Dtype::F32, 3),
        // kc/vc = 池叶(表示相关槽;契约单源同 f16 臂,见上)
        Param::t_in("kc", Dtype::F32, 0),
        Param::t_in("vc", Dtype::F32, 0),
        // 契约:slots/kv_lens r2([1, n];与 f16 臂同式)
        Param::t_in("slots", Dtype::F32, 2),
        Param::t_in("kv_lens", Dtype::F32, 2),
        Param::sz("bs"),
        Param::sz("q_heads"),
        Param::sz("kv_heads"),
        Param::sz("d_dim"),
        Param::t_out("out", Dtype::F32, 2),
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

#[kernel_tags("dom.attn", "role.attn", "feat.dflash", "feat.spec", "phase.propose", "kv.draft.f16")]
impl KernelSpec for NaiveAttnNcF16 {
    const ENTRY: &'static str = "owl_naive_attn_nc_f16";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_self", Dtype::F16, 3),
        Param::t_in("v_self", Dtype::F16, 3),
        Param::t_in("kc", Dtype::F16, 5),
        Param::t_in("vc", Dtype::F16, 4),
        Param::t_in("kv_len_p", Dtype::F32, 0),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::i32("ring_cap"),
        Param::t_out("out", Dtype::F16, 2),
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

#[kernel_tags("dom.attn", "role.attn", "feat.dflash", "feat.spec", "phase.propose", "kv.draft.bf16")]
impl KernelSpec for NaiveAttnNcBf16 {
    const ENTRY: &'static str = "owl_naive_attn_nc_bf16";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::BF16, 2),
        Param::t_in("k_self", Dtype::BF16, 3),
        Param::t_in("v_self", Dtype::BF16, 3),
        Param::t_in("kc", Dtype::BF16, 5),
        Param::t_in("vc", Dtype::BF16, 4),
        Param::t_in("kv_len_p", Dtype::F32, 0),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::i32("ring_cap"),
        Param::t_out("out", Dtype::BF16, 2),
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

#[kernel_tags("dom.attn", "role.attn", "feat.dflash", "feat.spec", "phase.propose", "kv.draft.fp8")]
impl KernelSpec for NaiveAttnNcFp8KvF16 {
    const ENTRY: &'static str = "owl_naive_attn_nc_fp8kv_f16";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_self", Dtype::F16, 3),
        Param::t_in("v_self", Dtype::F16, 3),
        Param::t_in("kc", Dtype::U32, 1),
        Param::t_in("vc", Dtype::U32, 1),
        Param::t_in("kv_len_p", Dtype::F32, 0),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::i32("ring_cap"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_DRAFT, LayoutTag::Classic, Dtype::F8E4M3)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

// ============================================================================
// 草稿 NC 注意力 v3(页粒度共享 K/V;2026-10-26 propose 刀1)
// ============================================================================

/// 四个入口共用一份表(与 v1/v2 同签名:PARAMS 一字不动)。
macro_rules! nc_tiled_spec {
    ($ty:ident, $entry:literal, $dt:ident, $kv:ident, $kr:literal, $vr:literal, $kvpool:ident,
     $tag:literal, $doc:literal) => {
        #[doc = $doc]
        pub struct $ty;

        #[kernel_tags("dom.attn", "role.attn", "feat.dflash", "feat.spec", "phase.propose", $tag)]
        impl KernelSpec for $ty {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
            const PARAMS: &'static [Param] = &[
                Param::t_in("q", Dtype::$dt, 2),
                Param::t_in("k_self", Dtype::$dt, 3),
                Param::t_in("v_self", Dtype::$dt, 3),
                // 线格式事实:classic 池句柄 = 5 阶(K)/4 阶(V);fp8 档 = 扁平字节账
                Param::t_in("kc", Dtype::$kv, $kr),
                Param::t_in("vc", Dtype::$kv, $vr),
                Param::t_in("kv_len_p", Dtype::F32, 0),
                Param::i32("q_heads"),
                Param::i32("kv_heads"),
                Param::i32("d_dim"),
                Param::i32("page"),
                Param::i32("x"),
                Param::i32("ring_cap"),
                Param::t_out("out", Dtype::$dt, 2),
            ];
            const POOL: &'static [PoolAccess] =
                &[PoolAccess::read(pools::KV_DRAFT, LayoutTag::Classic, Dtype::$kvpool)];
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
            const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
                kv_dtype: Some(Dtype::$kvpool),
                // v3 的 shared 契约按 hd=128 定形(驱动件只在该几何下选它;
                // hd=256 的 shared 预算要 opt-in,留 v2)
                head_dim: &[128],
                page: &[16, 32, 64, 128],
                grp: None,
                tile: None,
            };
        }
    };
}

nc_tiled_spec!(
    NcAttnTiledBf16,
    "owl_naive_attn_nc_tiled_bf16",
    BF16,
    BF16,
    5,
    4,
    BF16,
    "kv.draft.bf16",
    "`owl_naive_attn_nc_tiled_bf16` —— NC 注意力 v3(页粒度 shared;bf16 池)"
);
nc_tiled_spec!(
    NcAttnTiledF16,
    "owl_naive_attn_nc_tiled_f16",
    F16,
    F16,
    5,
    4,
    F16,
    "kv.draft.f16",
    "`owl_naive_attn_nc_tiled_f16` —— NC 注意力 v3(f16 池;MTP 链同族)"
);
nc_tiled_spec!(
    NcAttnTiledFp8kvBf16,
    "owl_naive_attn_nc_tiled_fp8kv_bf16",
    BF16,
    U32,
    1,
    1,
    F8E4M3,
    "kv.draft.fp8",
    "`owl_naive_attn_nc_tiled_fp8kv_bf16` —— NC 注意力 v3(fp8 前缀池;bf16 自块)"
);
nc_tiled_spec!(
    NcAttnTiledFp8kvF16,
    "owl_naive_attn_nc_tiled_fp8kv_f16",
    F16,
    U32,
    1,
    1,
    F8E4M3,
    "kv.draft.fp8",
    "`owl_naive_attn_nc_tiled_fp8kv_f16` —— NC 注意力 v3(fp8 前缀池;f16 自块)"
);

/// `vllm_reshape_and_cache_f16` —— reshape_and_cache.cu(R1 补表;池声明见 POOL)
pub struct VllmReshapeAndCacheF16;

#[kernel_tags("dom.attn", "role.attn", "role.kv_write", "phase.prefill", "kv.draft.f16")]
impl KernelSpec for VllmReshapeAndCacheF16 {
    const ENTRY: &'static str = "vllm_reshape_and_cache_f16";
    const SRC: &'static str = "cu/attn/reshape_and_cache.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::F16, 3),
        Param::t_in("value", Dtype::F16, 3),
        Param::t_in("key_cache", Dtype::F16, 5),
        Param::t_in("value_cache", Dtype::F16, 4),
        Param::t_in("slot_mapping", Dtype::F32, 0),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_out("out", Dtype::F16, 1),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_DRAFT, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128, 256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `vllm_reshape_and_cache_f16_knhd` —— reshape_and_cache.cu(R1 补表;池声明见 POOL)
pub struct VllmReshapeAndCacheF16Knhd;

#[kernel_tags("dom.attn", "role.attn", "role.kv_write", "phase.prefill", "kv.main.f16")]
impl KernelSpec for VllmReshapeAndCacheF16Knhd {
    const ENTRY: &'static str = "vllm_reshape_and_cache_f16_knhd";
    const SRC: &'static str = "cu/attn/reshape_and_cache.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::F16, 3),
        Param::t_in("value", Dtype::F16, 3),
        Param::t_in("key_cache", Dtype::F16, 5),
        Param::t_in("value_cache", Dtype::F16, 4),
        Param::t_in("slot_mapping", Dtype::F32, 0),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_out("out", Dtype::F16, 1),
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

#[kernel_tags("dom.attn", "role.attn", "role.kv_write", "phase.prefill", "kv.main.bf16")]
impl KernelSpec for VllmReshapeAndCacheBf16Knhd {
    const ENTRY: &'static str = "vllm_reshape_and_cache_bf16_knhd";
    const SRC: &'static str = "cu/attn/reshape_and_cache.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::BF16, 3),
        Param::t_in("value", Dtype::BF16, 3),
        Param::t_in("key_cache", Dtype::BF16, 5),
        Param::t_in("value_cache", Dtype::BF16, 4),
        Param::t_in("slot_mapping", Dtype::F32, 0),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_out("out", Dtype::BF16, 1),
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

#[kernel_tags("dom.attn", "role.attn", "role.kv_write", "phase.prefill", "kv.draft.fp8")]
impl KernelSpec for ReshapeAndCacheFp8KvBf16 {
    const ENTRY: &'static str = "owl_reshape_and_cache_fp8kv_bf16";
    const SRC: &'static str = "cu/attn/reshape_and_cache_fp8kv.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::BF16, 3),
        Param::t_in("value", Dtype::BF16, 3),
        Param::t_in("key_cache", Dtype::U32, 1),
        Param::t_in("value_cache", Dtype::U32, 1),
        Param::t_in("slot_mapping", Dtype::F32, 0),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_out("out", Dtype::BF16, 1),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_DRAFT, LayoutTag::Classic, Dtype::F8E4M3)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3), head_dim: &[128, 256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_reshape_and_cache_fp8kv_bf16_knhd` —— reshape_and_cache_fp8kv.cu(R1 补表;池声明见 POOL)
pub struct ReshapeAndCacheFp8KvBf16Knhd;

#[kernel_tags("dom.attn", "role.attn", "role.kv_write", "phase.prefill", "kv.main.fp8")]
impl KernelSpec for ReshapeAndCacheFp8KvBf16Knhd {
    const ENTRY: &'static str = "owl_reshape_and_cache_fp8kv_bf16_knhd";
    const SRC: &'static str = "cu/attn/reshape_and_cache_fp8kv.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("key", Dtype::BF16, 3),
        Param::t_in("value", Dtype::BF16, 3),
        Param::t_in("key_cache", Dtype::U32, 1),
        Param::t_in("value_cache", Dtype::U32, 1),
        Param::t_in("slot_mapping", Dtype::F32, 0),
        Param::i32("key_stride"),
        Param::i32("value_stride"),
        Param::i32("num_heads"),
        Param::i32("head_size"),
        Param::i32("block_size"),
        Param::i32("x"),
        Param::t_out("out", Dtype::BF16, 1),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::write(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F8E4M3)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3), head_dim: &[128, 256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `vllm_paged_attention_v1_f16_hd128` —— pagedattention_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmPagedAttentionV1F16Hd128;

#[kernel_tags("dom.attn", "role.attn", "role.kv_read", "phase.decode", "kv.main.f16")]
impl KernelSpec for VllmPagedAttentionV1F16Hd128 {
    const ENTRY: &'static str = "vllm_paged_attention_v1_f16_hd128";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::i32("num_kv_heads"),
        Param::f32("scale"),
        Param::i32("max_num_blocks_per_seq"),
        Param::i32("q_stride"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::f32("softscapping"),
        Param::i32("sliding_window"),
        Param::i32("use_alibi"),
        Param::t_out("out", Dtype::F16, 2),
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

#[kernel_tags("dom.attn", "role.attn", "role.kv_read", "phase.decode", "kv.main.f16")]
impl KernelSpec for VllmPagedAttentionV1F16Hd128bs32 {
    const ENTRY: &'static str = "vllm_paged_attention_v1_f16_hd128bs32";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::i32("num_kv_heads"),
        Param::f32("scale"),
        Param::i32("max_num_blocks_per_seq"),
        Param::i32("q_stride"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::f32("softscapping"),
        Param::i32("sliding_window"),
        Param::i32("use_alibi"),
        Param::t_out("out", Dtype::F16, 2),
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

#[kernel_tags("dom.attn", "role.attn", "role.kv_read", "phase.decode", "kv.main.f16")]
impl KernelSpec for VllmPagedAttentionV1F16Hd256 {
    const ENTRY: &'static str = "vllm_paged_attention_v1_f16_hd256";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::i32("num_kv_heads"),
        Param::f32("scale"),
        Param::i32("max_num_blocks_per_seq"),
        Param::i32("q_stride"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::f32("softscapping"),
        Param::i32("sliding_window"),
        Param::i32("use_alibi"),
        Param::t_out("out", Dtype::F16, 2),
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

#[kernel_tags("dom.attn", "role.attn", "role.kv_read", "phase.decode", "kv.main.f16")]
impl KernelSpec for VllmPagedAttentionV1F16Hd256bs32 {
    const ENTRY: &'static str = "vllm_paged_attention_v1_f16_hd256bs32";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::i32("num_kv_heads"),
        Param::f32("scale"),
        Param::i32("max_num_blocks_per_seq"),
        Param::i32("q_stride"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::f32("softscapping"),
        Param::i32("sliding_window"),
        Param::i32("use_alibi"),
        Param::t_out("out", Dtype::F16, 2),
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

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.f16")]
impl KernelSpec for VllmChunkedPrefillPagedAttnOptF16Hd128 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_f16_hd128";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 0),
        Param::t_in("query_start_len", Dtype::F32, 0),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::t_in("sinks", Dtype::F32, 0),
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

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.f16")]
impl KernelSpec for VllmChunkedPrefillPagedAttnOptF16Hd256 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_f16_hd256";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 0),
        Param::t_in("query_start_len", Dtype::F32, 0),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::t_in("sinks", Dtype::F32, 0),
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

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.f16")]
impl KernelSpec for VllmChunkedPrefillPagedAttnOptF16KnhdHd128 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd128";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 0),
        Param::t_in("query_start_len", Dtype::F32, 0),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::t_in("sinks", Dtype::F32, 0),
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

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.f16")]
impl KernelSpec for VllmChunkedPrefillPagedAttnOptF16KnhdHd256 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_f16_knhd_hd256";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 0),
        Param::t_in("query_start_len", Dtype::F32, 0),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::t_in("sinks", Dtype::F32, 0),
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

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.fp8")]
impl KernelSpec for VllmChunkedPrefillPagedAttnOptFp8Hd128 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_fp8_hd128";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::U32, 1),
        Param::t_in("v_cache", Dtype::U32, 1),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 0),
        Param::t_in("query_start_len", Dtype::F32, 0),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::t_in("sinks", Dtype::F32, 0),
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
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F8E4M3)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `vllm_chunked_prefill_paged_attn_opt_fp8_hd256` —— prefill_paged_attn_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmChunkedPrefillPagedAttnOptFp8Hd256;

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.fp8")]
impl KernelSpec for VllmChunkedPrefillPagedAttnOptFp8Hd256 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_fp8_hd256";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::U32, 1),
        Param::t_in("v_cache", Dtype::U32, 1),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 0),
        Param::t_in("query_start_len", Dtype::F32, 0),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::t_in("sinks", Dtype::F32, 0),
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
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F8E4M3)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd128` —— prefill_paged_attn_f16.cu(R1 补表;池声明见 POOL)
pub struct VllmChunkedPrefillPagedAttnOptFp8KnhdHd128;

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.fp8")]
impl KernelSpec for VllmChunkedPrefillPagedAttnOptFp8KnhdHd128 {
    const ENTRY: &'static str = "vllm_chunked_prefill_paged_attn_opt_fp8_knhd_hd128";
    const SRC: &'static str = "cu/attn/prefill_paged_attn_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::U32, 1),
        Param::t_in("v_cache", Dtype::U32, 1),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("seq_lens", Dtype::F32, 0),
        Param::t_in("query_start_len", Dtype::F32, 0),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
        Param::t_in("sinks", Dtype::F32, 0),
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
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F8E4M3)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3), head_dim: &[128], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_prefill_split_f16_hd256` —— prefill_split_f16.cu(R1 补表;池声明见 POOL)
pub struct PrefillSplitF16Hd256;

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.f16")]
impl KernelSpec for PrefillSplitF16Hd256 {
    const ENTRY: &'static str = "owl_prefill_split_f16_hd256";
    const SRC: &'static str = "cu/attn/prefill_split_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("scr_out", Dtype::F16, 2),
        Param::t_in("scr_stat", Dtype::F32, 2),
        Param::t_in("k0_alibi", Dtype::F16, 0),  // 调用方传 K0 写输出(out r1)⇒ 不约束 rank(2026-10-15 B6)
        Param::f32("scale"),
        Param::i32("hkv"),
        Param::i32("T"),
        Param::i32("ctx_base"),
        Param::i32("nparts"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("page"),
        Param::i32("hq"),
        Param::t_out("dummy", Dtype::F16, 0),  // 哨兵槽:调用方传 K0 写输出(r1)⇒ 不约束(2026-10-15 B6)
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

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.fp8")]
impl KernelSpec for PrefillSplitFp8KvHd256 {
    const ENTRY: &'static str = "owl_prefill_split_fp8kv_hd256";
    const SRC: &'static str = "cu/attn/prefill_split_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::U32, 1),
        Param::t_in("v_cache", Dtype::U32, 1),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("scr_out", Dtype::F16, 2),
        Param::t_in("scr_stat", Dtype::F32, 2),
        Param::t_in("k0_alibi", Dtype::F16, 0),  // 调用方传 K0 写输出(out r1)⇒ 不约束 rank(2026-10-15 B6)
        Param::f32("scale"),
        Param::i32("hkv"),
        Param::i32("T"),
        Param::i32("ctx_base"),
        Param::i32("nparts"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("page"),
        Param::i32("hq"),
        Param::t_out("dummy", Dtype::F16, 0),  // 哨兵槽:调用方传 K0 写输出(r1)⇒ 不约束(2026-10-15 B6)
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F8E4M3)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_prefill_split_f16_knhd_hd256` —— prefill_split_f16.cu(R1 补表;池声明见 POOL)
pub struct PrefillSplitF16KnhdHd256;

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.f16")]
impl KernelSpec for PrefillSplitF16KnhdHd256 {
    const ENTRY: &'static str = "owl_prefill_split_f16_knhd_hd256";
    const SRC: &'static str = "cu/attn/prefill_split_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("scr_out", Dtype::F16, 2),
        Param::t_in("scr_stat", Dtype::F32, 2),
        Param::t_in("k0_alibi", Dtype::F16, 0),  // 调用方传 K0 写输出(out r1)⇒ 不约束 rank(2026-10-15 B6)
        Param::f32("scale"),
        Param::i32("hkv"),
        Param::i32("T"),
        Param::i32("ctx_base"),
        Param::i32("nparts"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("page"),
        Param::i32("hq"),
        Param::t_out("dummy", Dtype::F16, 0),  // 哨兵槽:调用方传 K0 写输出(r1)⇒ 不约束(2026-10-15 B6)
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

#[kernel_tags("dom.attn", "role.attn", "phase.prefill", "kv.main.fp8")]
impl KernelSpec for PrefillSplitFp8KvKnhdHd256 {
    const ENTRY: &'static str = "owl_prefill_split_fp8kv_knhd_hd256";
    const SRC: &'static str = "cu/attn/prefill_split_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::U32, 1),
        Param::t_in("v_cache", Dtype::U32, 1),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("scr_out", Dtype::F16, 2),
        Param::t_in("scr_stat", Dtype::F32, 2),
        Param::t_in("k0_alibi", Dtype::F16, 0),  // 调用方传 K0 写输出(out r1)⇒ 不约束 rank(2026-10-15 B6)
        Param::f32("scale"),
        Param::i32("hkv"),
        Param::i32("T"),
        Param::i32("ctx_base"),
        Param::i32("nparts"),
        Param::i32("kv_block_stride"),
        Param::i32("kv_head_stride"),
        Param::i32("page"),
        Param::i32("hq"),
        Param::t_out("dummy", Dtype::F16, 0),  // 哨兵槽:调用方传 K0 写输出(r1)⇒ 不约束(2026-10-15 B6)
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F8E4M3)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3), head_dim: &[256], page: &[16, 32, 64, 128], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_prefill_split_reduce_f16_hd256` —— prefill_split_f16.cu(R1 补表;池声明见 POOL)
pub struct PrefillSplitReduceF16Hd256;

#[kernel_tags("dom.attn", "role.attn", "phase.prefill")]
impl KernelSpec for PrefillSplitReduceF16Hd256 {
    const ENTRY: &'static str = "owl_prefill_split_reduce_f16_hd256";
    const SRC: &'static str = "cu/attn/prefill_split_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("split_alibi", Dtype::F16, 0),  // 哨兵槽(=K0 写输出 r1)⇒ 不约束(2026-10-15 B6)
        Param::t_in("scr_out", Dtype::F16, 2),
        Param::t_in("scr_stat", Dtype::F32, 2),
        Param::i32("nparts"),
        Param::i32("hq"),
        Param::i32("hd"),
        Param::i32("T"),
        Param::t_out("out", Dtype::F16, 2),
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

/// `vllm_paged_attention_v2_f16_hd128` —— pagedattention_f16.cu(F 轮符号门抓出)
///
/// page=16 档(packing 走此名;`driver.rs` 依 page 拾取)—— **原为未登记的活路径**,F 轮符号门抓出
pub struct VllmPagedAttentionV2F16Hd128;

#[kernel_tags("dom.attn", "role.attn", "role.kv_read", "phase.decode", "kv.main.f16")]
impl KernelSpec for VllmPagedAttentionV2F16Hd128 {
    const ENTRY: &'static str = "vllm_paged_attention_v2_f16_hd128";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
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
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128], page: &[16], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `vllm_paged_attention_v2_f16_hd256` —— pagedattention_f16.cu(F 轮符号门抓出)
///
/// 同上
pub struct VllmPagedAttentionV2F16Hd256;

#[kernel_tags("dom.attn", "role.attn", "role.kv_read", "phase.decode", "kv.main.f16")]
impl KernelSpec for VllmPagedAttentionV2F16Hd256 {
    const ENTRY: &'static str = "vllm_paged_attention_v2_f16_hd256";
    const SRC: &'static str = "cu/attn/pagedattention_f16.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_cache", Dtype::F16, 5),
        Param::t_in("v_cache", Dtype::F16, 4),
        Param::t_in("block_tables", Dtype::F32, 2),
        Param::t_in("context_lens", Dtype::F32, 2),
        Param::t_in("alibi_slopes", Dtype::F16, 0),
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
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Classic, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[256], page: &[16], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_naive_attn_nc_f16_knhd` —— naive_attn_nc.cu(F 轮符号门抓出)
///
/// ⚠️ 当前 0 调用点(暂不可达;候选:从 .cu 删除变体 —— 先按 R1「编出即有裁决」建表)
/// **池归属**:`*_knhd` 变体对应**主池** KnHD(写者 = `vllm_reshape_and_cache_f16_knhd`);
/// 草稿池走 classic 变体 + `vllm_reshape_and_cache_bf16/f16`。
pub struct NaiveAttnNcF16Knhd;

#[kernel_tags("dom.attn", "role.attn", "feat.dflash", "feat.spec", "phase.propose", "kv.main.f16")]
impl KernelSpec for NaiveAttnNcF16Knhd {
    const ENTRY: &'static str = "owl_naive_attn_nc_f16_knhd";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_self", Dtype::F16, 3),
        Param::t_in("v_self", Dtype::F16, 3),
        Param::t_in("kc", Dtype::F16, 5),
        Param::t_in("vc", Dtype::F16, 4),
        Param::t_in("kv_len_p", Dtype::F32, 0),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::i32("ring_cap"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F16), head_dim: &[128], page: &[32], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_naive_attn_nc_bf16_knhd` —— naive_attn_nc.cu(F 轮符号门抓出)
///
/// 同上
pub struct NaiveAttnNcBf16Knhd;

#[kernel_tags("dom.attn", "role.attn", "feat.dflash", "feat.spec", "phase.propose", "kv.main.bf16")]
impl KernelSpec for NaiveAttnNcBf16Knhd {
    const ENTRY: &'static str = "owl_naive_attn_nc_bf16_knhd";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::BF16, 2),
        Param::t_in("k_self", Dtype::BF16, 3),
        Param::t_in("v_self", Dtype::BF16, 3),
        Param::t_in("kc", Dtype::BF16, 5),
        Param::t_in("vc", Dtype::BF16, 4),
        Param::t_in("kv_len_p", Dtype::F32, 0),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::i32("ring_cap"),
        Param::t_out("out", Dtype::BF16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::BF16)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::BF16), head_dim: &[128], page: &[32], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_naive_attn_nc_fp8kv_f16_knhd` —— naive_attn_nc.cu(F 轮符号门抓出)
///
/// 同上
pub struct NaiveAttnNcFp8KvF16Knhd;

#[kernel_tags("dom.attn", "role.attn", "feat.dflash", "feat.spec", "phase.propose", "kv.main.fp8")]
impl KernelSpec for NaiveAttnNcFp8KvF16Knhd {
    const ENTRY: &'static str = "owl_naive_attn_nc_fp8kv_f16_knhd";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::F16, 2),
        Param::t_in("k_self", Dtype::F16, 3),
        Param::t_in("v_self", Dtype::F16, 3),
        Param::t_in("kc", Dtype::U32, 1),
        Param::t_in("vc", Dtype::U32, 1),
        Param::t_in("kv_len_p", Dtype::F32, 0),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::i32("ring_cap"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F8E4M3)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3), head_dim: &[128], page: &[32], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_naive_attn_nc_fp8kv_bf16_knhd` —— naive_attn_nc.cu(F 轮符号门抓出)
///
/// 同上
pub struct NaiveAttnNcFp8KvBf16Knhd;

#[kernel_tags("dom.attn", "role.attn", "feat.dflash", "feat.spec", "phase.propose", "kv.main.fp8")]
impl KernelSpec for NaiveAttnNcFp8KvBf16Knhd {
    const ENTRY: &'static str = "owl_naive_attn_nc_fp8kv_bf16_knhd";
    const SRC: &'static str = "cu/attn/naive_attn_nc.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("q", Dtype::BF16, 2),
        Param::t_in("k_self", Dtype::BF16, 3),
        Param::t_in("v_self", Dtype::BF16, 3),
        Param::t_in("kc", Dtype::U32, 1),
        Param::t_in("vc", Dtype::U32, 1),
        Param::t_in("kv_len_p", Dtype::F32, 0),
        Param::i32("q_heads"),
        Param::i32("kv_heads"),
        Param::i32("d_dim"),
        Param::i32("page"),
        Param::i32("x"),
        Param::i32("ring_cap"),
        Param::t_out("out", Dtype::BF16, 2),
    ];
    const POOL: &'static [PoolAccess] = &[PoolAccess::read(pools::KV_MAIN, LayoutTag::Knhd, Dtype::F8E4M3)];
    const SHAPE: crate::spec::ShapeFamily = crate::spec::ShapeFamily {
        kv_dtype: Some(Dtype::F8E4M3), head_dim: &[128], page: &[32], grp: None, tile: None,
    };
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static VLLM_PAGED_ATTENTION_V2_F16_HD128: KernelMeta = meta_of::<VllmPagedAttentionV2F16Hd128>();
pub static VLLM_PAGED_ATTENTION_V2_F16_HD256: KernelMeta = meta_of::<VllmPagedAttentionV2F16Hd256>();
pub static NAIVE_ATTN_NC_F16_KNHD: KernelMeta = meta_of::<NaiveAttnNcF16Knhd>();
pub static NAIVE_ATTN_NC_BF16_KNHD: KernelMeta = meta_of::<NaiveAttnNcBf16Knhd>();
pub static NAIVE_ATTN_NC_FP8_KV_F16_KNHD: KernelMeta = meta_of::<NaiveAttnNcFp8KvF16Knhd>();
pub static NAIVE_ATTN_NC_FP8_KV_BF16_KNHD: KernelMeta = meta_of::<NaiveAttnNcFp8KvBf16Knhd>();


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
            "T,T,T,T,T,T,i32,i32,i32,i32,i32,i32,O"
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
        assert_eq!(m.shape.kv_dtype, Some(Dtype::F8E4M3));
        assert_eq!(m.shape.head_dim, &[128, 256]);
        assert_eq!(m.shape.page, &[16, 32, 64, 128]);
    }

    /// `ShapeRule::NONE` 是显式"无约束",不是漏写
    #[test]
    fn shape_rule_none_is_explicit() {
        assert!(crate::spec::ShapeRule::NONE.dims.is_empty());
    }
}
