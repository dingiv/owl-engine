//! FlashInfer prefill(foreign-kernel 通道;E1.5 工作包,2026-10-03 落地)。
//!
//! 溯源:adapter 预编 .a(预编源 `cu/flashinfer/owl_fi_attention.cu`,
//! ported from repos/attention.rs flashinfer_adapter_prefill.cu 非 SM90/FA2
//! 分支;guoqingbao/flashinfer fork @ 0f06c230,Apache-2.0)。.a 入库
//! (`cu/flashinfer/prebuilt/libowl_flashinfer.a`,marlin .a 先例;
//! FLASHINFER_FORCE_BUILD=1 重编,nvcc + fork include 面)。
//!
//! **为什么是 FlashInfer**(REQ-PRE-01 / REQ-DESIGN:禁自研核进热路径):
//! 旧 chunked_prefill_paged_attn_opt(vendor port)~6 TFLOPS(标量,1
//! thread/query 串行扫);自研 split(flash-decoding)同分块 O(N²) 形态,
//! 8k+ 才 +5.3%。FA2 级 tensor-core kernel 是社区标准答案,E2 分页底座
//! 落地后前置解除(2026-09-27 定盘)。
//!
//! **布局契约**(paged_kv_t QKVLayout::kNHD,页粒度;page.cuh
//! get_elem_offset = page·stride + h·hd + entry·(Hkv·hd) + d):
//! - k_fi / v_fi(K/V 影子池):`[num_blocks, page_size, Hkv, head_dim]`
//!   f16 —— 由 `owl_reshape_and_cache_dual_f16`(K0-dual)与 classic K/V
//!   池同发射写出(classic K = x-interleave、V = dim-major,均不匹配)
//! - 页表:`indices`(物理页 id 拼接)/ `indptr`([0, nb])/ `last_len`
//!   (末页有效数 = ctx − (nb−1)·page;整页 = page,非 %)—— i32 设备表,
//!   engine 每 chunk 构造(ForwardCtx.fi)
//!
//! **plan/run 分离**:plan = host 侧调度规划(PrefillPlan;split-kv 决策 +
//! tile 选择),每 chunk 一次,server 侧 1 项缓存(全层同参);run = 每
//! attention 层一次(BatchPrefillWithPagedKVCacheDispatched,causal,
//! chunked 语义 = q 对齐 kv 尾部,与 K0 先行的池读序配套)。
//!
//! **发射通道**:server foreign-kernel(虚拟核名 [`PREFILL_FI`];marlin/
//! cublas 同款,Launch 即唯一执行命令)。批 = 1(owl 单会话)。

use std::ffi::c_void;

/// foreign-kernel 虚拟核名(server 分派臂;f16 KV;**字面量住址 = 本模块**)
pub const PREFILL_FI: &str = "flashinfer_prefill_paged_f16";
/// fp8 KV 变体(e4m3 KV + half Q/Out;槽序同 PREFILL_FI)
pub const PREFILL_FI_FP8KV: &str = "flashinfer_prefill_paged_fp8kv";
/// **40 系原生 FP8 MMA 臂**(e4m3 Q×KV;仅 Ada/SM89 由 picker 选择)
pub const PREFILL_FI_FP8MMA: &str = "flashinfer_prefill_paged_fp8mma";

/// **视觉塔注意力**(2026-10-15):ragged(**非分页**)非因果 prefill。
/// 复用同一份 vendored FlashInfer FA2 家族核(`BatchPrefillWithRaggedKVCacheDispatched` + `MaskMode::kNone`),与 vLLM 的 ViT 走同路算法;胶水住
/// `cu/flashinfer/owl_fi_attn_ragged.cu`。**legacy 单图/多图 varlen**；
/// 2026-10-23 起视觉塔统一走 FA2，本常量保留但不再被 driver 选中。
pub const ATTN_RAGGED_F16: &str = "flashinfer_attn_ragged_f16";

/// **vLLM 原版 FA2 varlen forward**（2026-10-23）：视觉塔统一入口，
/// hd=64 → kHeadDim=64，hd=72 → kHeadDim=96；与 vLLM Qwen3.5-ViT 同路径。
/// 源在 `cu/flash_attn/vendor/`，本仓只写 host 胶水。
pub const FA2_VARLEN_F16: &str = "flash_attn_varlen_f16";
/// FA2 的 unpadded LSE 擦写区：`[heads, total_q] f32`（16384×16 → 1 MiB；
/// 8 MiB 覆盖到 131072 行，视觉远小于此）。
pub const FA2_LSE_BYTES: usize = 8 * 1024 * 1024;

/// 槽序契约(LaunchMsg.args):
/// `[T q, T k_fi, T v_fi, T q_cu_seqlens, T indices, T indptr, T last_len,
///   T wr(树序依赖边,K0-dual 先于读池;FI 不解引用),
///   O out, sz total_rows, sz ctx_total, sz T, sz hq, sz hkv, sz hd,
///   sz page, sz sm_scale_bits, sz nb]`
/// = 8 Block + 1 O(输出 = 槽序契约 4 变体,O 槽显式位)+ 8 sz;
/// sm_scale 以 `f32::to_bits` 过线,server 侧 `f32::from_bits` 还原。
pub const PREFILL_FI_SLOTS: &str =
    "[T q, T kc_fi, T vc, T q_cu, T indices, T indptr, T last_len, T wr, O out, \
     sz total_rows, sz ctx_total, sz T, sz hq, sz hkv, sz hd, sz page, \
     sz sm_scale_bits, sz nb]";

/// FI plan workspace 尺寸(设备 int;server 句柄懒分配)
pub const FI_INT_WS_BYTES: usize = 32 * 1024 * 1024;
/// **视觉塔 ragged 臂专用 int workspace**(2026-10-23):paged prefill 与视觉塔
/// 共用同一个 `int_ws` 时,每次文本 prefill 都会把 ragged plan 表覆盖掉;
/// ragged plan 按形状缓存后不会重刷 ⇒ **同一图片第二次起的注意力读到文本 plan**
/// (首图与后续图输出不同、且图像 token 语义错)。单独一份后 plan 常驻。
/// 8 MiB 对视觉 rows ≤ 16k 的 request/tile 表是数量级余量;不足时 plan 结构化拒。
pub const FI_RAGGED_INT_WS_BYTES: usize = 8 * 1024 * 1024;
/// FI plan workspace 尺寸(设备 float;split-kv tmp_v/tmp_s)
pub const FI_FLOAT_WS_BYTES: usize = 64 * 1024 * 1024;
/// FI page-locked/host 暂冲(H2D staging;PrefillPlan 用)
pub const FI_HOST_STAGING_BYTES: usize = 32 * 1024 * 1024;

/// foreign 分派谓词(server handle_launch 前置检查)
pub fn is_foreign(name: &str) -> bool {
    name == PREFILL_FI
        || name == PREFILL_FI_FP8KV
        || name == PREFILL_FI_FP8MMA
        || name == ATTN_RAGGED_F16
        || name == FA2_VARLEN_F16
        || name == PLAN_SYNC_FI
}

// ============================================================================
// plan 表同步(`flashinfer_fi_plan_sync`)—— 族的一个**虚拟算子**
// ============================================================================
//
// 为什么是算子而不是一条专属命令(2026-11-01 用户裁决):它本质上是
// **族内部的一次 host 调用 + 一次 H2D**(算调度表 + 刷进 int_ws),与 cuBLAS
// 封装同形 —— 那就该走**转发面**(算子声明 → 准入 → 路由 → 族 run),
// 不该在设备契约上开一条 flashinfer 专属命令。
//
// 形状约束:转发面要求发射**有一个块参与**(`GpuClient::exec` 的"至少一个
// Block"门)且 `out_elems ≥ 1`(几何通用律)⇒ 本算子带一个 **1 元素的哑输出**
// (契约 4 规定的 O 槽;调用方备块后复用,不逐次分配)。
//
// 捕获期禁令:本算子要算 host plan(**不可进图**)⇒ `CAPTURE = NeedsHostPrep`,
// 由准入面统一拒(原来靠 `Command::FiPlanSync` 的守卫表硬编码一条)。

/// 虚拟算子名(FI plan 表同步;槽序 = `[O out, sz t, sz ctx_total,
/// sz hq, sz hkv, sz hd, sz page]`)
pub const PLAN_SYNC_FI: &str = "flashinfer_fi_plan_sync";

/// **构造 plan 表同步的发射消息**(转发面唯一入口;引擎侧不手摩槽序)。
///
/// `t` = 本轮 q 行数(decode = 1;prefill 桶 = 桶 t);`ctx_total` = 本轮 KV
/// 总长(**必须给真值** —— plan 键的 ctx 维)。`out` = 调用方备的 1 元素哑输出块。
pub fn plan_sync_msg(
    out: crate::contract::Bytes,
    t: usize,
    ctx_total: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    page: usize,
) -> crate::contract::LaunchMsg {
    use crate::contract::{Arg, Dtype, KernelSource, LaunchMsg};
    LaunchMsg {
        kernel: KernelSource { name: PLAN_SYNC_FI.to_string(), source: String::new() },
        args: vec![
            Arg::block(out.id, Dtype::F16, 1),
            Arg::U64(t as u64),
            Arg::U64(ctx_total as u64),
            Arg::U64(hq as u64),
            Arg::U64(hkv as u64),
            Arg::U64(hd as u64),
            Arg::U64(page as u64),
        ],
        // 无核入队(纯 host 调用);几何给"非零最小"避免自持哨兵语义
        grid: (1, 1, 1),
        block: (1, 1, 1),
        shared_mem: 0,
        out_elems: 1,
    }
}

// ============================================================================
// 族装配参数(进程级单源;2026-11-01 分层修正)
// ============================================================================
//
// **为什么住这里**:这些是**算子族的装配参数**(plan 表是否图自携 / plan 用不用
// 图档瓦 / prefill 是否走 f16 影子),**不是设备能力**。此前它们挂在
// `DeviceClient` 上(缺省空实现),两宗罪:
//   (a) 每个后端都得背 flashinfer 的语义;
//   (b) **缺省空实现 = 静默托管** —— 后端没接就悄悄不生效,正是本仓禁的失败形态。
//
// 现在的形状:**写者唯一** = [`configure`](引擎装配期;引擎 = 配置持有者);
// **读侧** = `server::flashinfer`(feature 门内)+ C++ 适配器(它只认
// `FI_CUDA_GRAPH` env ⇒ 本函数**同一次**落 env,两条口径不可能漂移)。
//
// 放**零门**模块的理由:执行面受 `feature = "flashinfer"` 门控,而配置面必须对
// 引擎/后端**恒可见**(boot 期就要下发,不能因为 feature 关掉就编译不过)。
//
// 注:这里写 env 不是"用 env 当配置源"(禁),而是给**上游固定 ABI** 的适配器
// 递话(它只读 `getenv`);配置源恒为 `[dispatch]`/`[kernel]` 键。

/// FlashInfer 族装配参数(**整快照**:一次落全部字段,无局部更新语义)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashinferCfg {
    /// 捕获窗内是否把本 plan 的调度表烘成图内 restore 节点
    /// (`[dispatch] fi_plan_restore`)。`false` ⇒ host 是表唯一权威(每步刷)。
    pub plan_restore: bool,
    /// plan 用**图档瓦**(形状对 ctx 恒定;`[dispatch] fi_cuda_graph`)。
    pub cuda_graph: bool,
    /// prefill-only fp8→f16 影子(`[kernel] prefill_shadow`;仅 fp8 主池)。
    pub prefill_shadow: bool,
    /// 影子容量上界(token;`runtime.max_seq`):首层按它一次预留。
    pub prefill_shadow_cap: usize,
    /// **40 系原生 FP8 MMA 臂**(由引擎从运行期 `Arch::Sm89` 投影;非编译 feature)
    pub fp8_native: bool,
}

impl Default for FlashinferCfg {
    /// **生产缺省** = 各配置键的缺省值(探针/单测可
    /// `FlashinferCfg { cuda_graph: true, ..Default::default() }` 只改一项)。
    fn default() -> Self {
        Self {
            plan_restore: true,
            cuda_graph: false,
            prefill_shadow: false,
            prefill_shadow_cap: 0,
            fp8_native: false,
        }
    }
}

/// 图自携 plan restore(缺省 true = 图自带表)
static PLAN_RESTORE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
/// 图档 plan 瓦(缺省 false = eager 定瓦)
static CUDA_GRAPH: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// prefill f16 影子(缺省 false)
static PREFILL_SHADOW: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
/// 影子容量上限(token;0 = 未下发)
static PREFILL_SHADOW_CAP: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
/// 40 系原生 FP8 MMA(引擎 boot 从 `Arch::Sm89` 投影)
static FP8_NATIVE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// **族装配参数唯一写点**(引擎装配期一次;boot 期,任何 plan/捕获之前)。
///
/// 全部字段一次落:四个原子 + C++ 侧的 `FI_CUDA_GRAPH` env。
/// 自洽门:**影子开启必须有容量**(cap = 0 时首层按 0 预留 ⇒ 开关静默不生效,
/// 那正是要禁的静默失败);不满足则**结构化拒**,不做"看起来开了其实没开"。
pub fn configure(cfg: FlashinferCfg) -> Result<(), crate::contract::OpError> {
    if cfg.prefill_shadow && cfg.prefill_shadow_cap == 0 {
        return Err(crate::contract::OpError::Contract {
            op: PREFILL_FI.into(),
            field: "prefill_shadow_cap",
            expect: "影子开启 ⇒ cap > 0(容量上界,取 runtime.max_seq)".into(),
            got: "cap = 0(首层将按 0 预留,影子静默不生效)".into(),
        });
    }
    // 原生 FP8 MMA 只在 arch-sm89/120 构建里有设备符号；运行期 arch 由引擎
    // 投影，二者不一致 = 配置/构建错配，响亮拒（不回退到“看起来能用”）。
    if cfg.fp8_native && !cfg!(any(feature = "arch-sm89", feature = "arch-sm120")) {
        return Err(crate::contract::OpError::Contract {
            op: PREFILL_FI.into(),
            field: "fp8_native",
            expect: "arch-sm89/arch-sm120 构建的二进制".into(),
            got: "当前构建未编 native FP8 MMA".into(),
        });
    }
    // C++ 适配器(上游 ABI 不动)只认 env ⇒ 与本侧原子同一次写入
    std::env::set_var("FI_CUDA_GRAPH", if cfg.cuda_graph { "1" } else { "0" });
    PLAN_RESTORE.store(cfg.plan_restore, std::sync::atomic::Ordering::Relaxed);
    CUDA_GRAPH.store(cfg.cuda_graph, std::sync::atomic::Ordering::Relaxed);
    PREFILL_SHADOW.store(cfg.prefill_shadow, std::sync::atomic::Ordering::Relaxed);
    PREFILL_SHADOW_CAP.store(cfg.prefill_shadow_cap, std::sync::atomic::Ordering::Relaxed);
    FP8_NATIVE.store(cfg.fp8_native, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// 读:图自携 plan restore(热路径一次 relaxed load)
pub(crate) fn plan_restore() -> bool {
    PLAN_RESTORE.load(std::sync::atomic::Ordering::Relaxed)
}

/// 读:图档 plan 瓦(plan 期一次)
pub(crate) fn cuda_graph() -> bool {
    CUDA_GRAPH.load(std::sync::atomic::Ordering::Relaxed)
}

/// 读:prefill f16 影子开关
pub(crate) fn prefill_shadow_on() -> bool {
    PREFILL_SHADOW.load(std::sync::atomic::Ordering::Relaxed)
}

/// 读:影子容量上界(token)
pub(crate) fn prefill_shadow_cap() -> usize {
    PREFILL_SHADOW_CAP.load(std::sync::atomic::Ordering::Relaxed)
}
/// 读:40 系原生 FP8 MMA(引擎 boot 投影)
pub(crate) fn fp8_native() -> bool {
    FP8_NATIVE.load(std::sync::atomic::Ordering::Relaxed)
}

extern "C" {
    /// host 侧调度规划(见 cu/flashinfer/owl_fi_attention.cu 契约)
    pub fn owl_fi_prefill_plan(
        float_ws: *mut c_void, float_ws_size: usize,
        int_ws: *mut c_void, int_ws_size: usize,
        page_locked: *mut c_void, page_locked_size: usize,
        out_plan15: *mut i64,
        out_cta_tile_q: *mut i32,
        out_split_kv: *mut i32,
        // plan 表写入 int_ws 的有效字节数(不变盒快照尺寸;C++ 侧 = need_int)
        out_int_bytes: *mut usize,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        total_num_rows: i32,
        batch_size: i32,
        num_qo_heads: i32, num_kv_heads: i32,
        head_dim: i32, page_size: i32,
        stream: *mut c_void,
    ) -> i32;

    /// **f16 shadow 专用 plan**(显式 `kv_bytes`;2 = f16,不读 env)
    pub fn owl_fi_prefill_plan2(
        float_ws: *mut c_void, float_ws_size: usize,
        int_ws: *mut c_void, int_ws_size: usize,
        page_locked: *mut c_void, page_locked_size: usize,
        out_plan15: *mut i64,
        out_cta_tile_q: *mut i32,
        out_split_kv: *mut i32,
        out_int_bytes: *mut usize,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        total_num_rows: i32,
        batch_size: i32,
        num_qo_heads: i32, num_kv_heads: i32,
        head_dim: i32, page_size: i32,
        kv_bytes: i32,
        stream: *mut c_void,
    ) -> i32;

    /// **40/50 系原生 FP8 MMA 专用 plan**（arch-sm89/120 构建；force CTA16，split 沿用上游）
    #[cfg(any(feature = "arch-sm89", feature = "arch-sm120"))]
    pub fn owl_fi_prefill_plan_fp8mma(
        float_ws: *mut c_void, float_ws_size: usize,
        int_ws: *mut c_void, int_ws_size: usize,
        page_locked: *mut c_void, page_locked_size: usize,
        out_plan15: *mut i64,
        out_cta_tile_q: *mut i32,
        out_split_kv: *mut i32,
        out_int_bytes: *mut usize,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        total_num_rows: i32,
        batch_size: i32,
        num_qo_heads: i32, num_kv_heads: i32,
        head_dim: i32, page_size: i32,
        stream: *mut c_void,
    ) -> i32;

    /// **FP8→F16 shadow 反量化**(prefill-only;dense 逻辑页输出)
    pub fn owl_fi_dequant_fp8_to_f16(
        k_src: *const c_void,
        v_src: *const c_void,
        k_dst: *mut c_void,
        v_dst: *mut c_void,
        indices: *const i32,
        nb_pages: i32,
        hkv: i32,
        hd: i32,
        page: i32,
        stream: *mut c_void,
    ) -> i32;

    /// fp8 KV paged prefill(e4m3 KV + half Q/Out;scale 隐式 1.0)
    pub fn owl_fi_prefill_run_fp8kv(
        q_ptr: *const c_void,
        k_data: *const c_void,
        v_data: *const c_void,
        out_ptr: *mut c_void,
        q_cu_seqlens: *mut i32,
        indices: *mut i32,
        indptr: *mut i32,
        last_len: *mut i32,
        plan15: *const i64,
        int_ws: *mut c_void, int_ws_size: usize,
        float_ws: *mut c_void, float_ws_size: usize,
        batch_size: i32,
        num_qo_heads: i32, num_kv_heads: i32,
        head_dim: i32, page_size: i32,
        total_num_rows: i32,
        sm_scale: f32,
        // 尾部 CTA 掩码(P-5;null = 不用掩码)。内容 `i < tiles` 由 Rust 侧
        // 按 plan15 算好、**窗外**上传一次(桶级静态;见 server/flashinfer.rs)
        block_valid_mask: *const c_void,
        stream: *mut c_void,
    ) -> i32;

    /// paged FA2 prefill 发射(见 cu/flashinfer/owl_fi_attention.cu 契约)
    pub fn owl_fi_prefill_run(
        q_ptr: *const c_void,
        k_data: *const c_void,
        v_data: *const c_void,
        out_ptr: *mut c_void,
        q_cu_seqlens: *mut i32,
        indices: *mut i32,
        indptr: *mut i32,
        last_len: *mut i32,
        plan15: *const i64,
        int_ws: *mut c_void, int_ws_size: usize,
        float_ws: *mut c_void, float_ws_size: usize,
        batch_size: i32,
        num_qo_heads: i32, num_kv_heads: i32,
        head_dim: i32, page_size: i32,
        total_num_rows: i32,
        sm_scale: f32,
        // 尾部 CTA 掩码(P-5;null = 不用掩码)。内容 `i < tiles` 由 Rust 侧
        // 按 plan15 算好、**窗外**上传一次(桶级静态;见 server/flashinfer.rs)
        block_valid_mask: *const c_void,
        stream: *mut c_void,
    ) -> i32;

    /// **40/50 系原生 FP8 MMA**(2026-11-04;arch-sm89/120 构建)
    /// Q 已是 e4m3 的 paged prefill。参数表与 `owl_fi_prefill_run_fp8kv` 逐字相同。
    #[cfg(any(feature = "arch-sm89", feature = "arch-sm120"))]
    pub fn owl_fi_prefill_run_fp8mma(
        q_ptr: *const c_void,
        k_data: *const c_void,
        v_data: *const c_void,
        out_ptr: *mut c_void,
        q_cu_seqlens: *mut i32,
        indices: *mut i32,
        indptr: *mut i32,
        last_len: *mut i32,
        plan15: *const i64,
        int_ws: *mut c_void, int_ws_size: usize,
        float_ws: *mut c_void, float_ws_size: usize,
        batch_size: i32,
        num_qo_heads: i32, num_kv_heads: i32,
        head_dim: i32, page_size: i32,
        total_num_rows: i32,
        sm_scale: f32,
        block_valid_mask: *const c_void,
        stream: *mut c_void,
    ) -> i32;

    /// **Q f16 → e4m3**（原生 FP8 MMA 前置）
    #[cfg(any(feature = "arch-sm89", feature = "arch-sm120"))]
    pub fn owl_fi_quant_f16_to_fp8(
        src: *const c_void,
        dst: *mut c_void,
        n: i64,
        stream: *mut c_void,
    ) -> i32;

    /// 视觉塔 ragged plan(host;每 (rows,heads,hd,cu) 一次；多图按 cu 分段)
    pub fn owl_fi_attn_ragged_plan(
        float_ws: *mut std::ffi::c_void,
        float_ws_size: usize,
        int_ws: *mut std::ffi::c_void,
        int_ws_size: usize,
        page_locked: *mut std::ffi::c_void,
        page_locked_size: usize,
        out_plan15: *mut i64,
        out_cta_tile_q: *mut i32,
        out_cu_dev: *mut i32,
        cu_host: *const i32,
        n_seq: i32,
        rows: i32,
        num_heads: i32,
        head_dim: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// 视觉塔 ragged run(f16;非因果;单图/多图 varlen)
    pub fn owl_fi_attn_ragged_run_f16(
        q: *const std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        cu: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        plan15: *const i64,
        int_ws: *mut std::ffi::c_void,
        int_ws_size: usize,
        float_ws: *mut std::ffi::c_void,
        float_ws_size: usize,
        rows: i32,
        num_heads: i32,
        head_dim: i32,
        n_seq: i32,
        scale: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

    /// vLLM FA2 varlen forward（f16;hd=72;非因果;单图/多图 varlen）
    pub fn owl_fa2_varlen_fwd_f16(
        q: *const std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        out: *mut std::ffi::c_void,
        lse: *mut f32,
        cu_dev: *mut i32,
        batch: i32,
        total_q: i32,
        max_seqlen: i32,
        heads: i32,
        head_dim: i32,
        scale: f32,
        stream: *mut std::ffi::c_void,
    ) -> i32;

}

#[cfg(test)]
mod cfg_tests {
    use super::*;

    /// 缺省档 = 生产基线(plan 自携 true / eager 瓦 / 影子关)。
    #[test]
    fn 缺省等于生产档() {
        let d = FlashinferCfg::default();
        assert!(d.plan_restore, "图自携缺省 true(历史行为)");
        assert!(!d.cuda_graph, "图档瓦缺省关(eager 定瓦)");
        assert!(!d.prefill_shadow, "影子缺省关");
        assert_eq!(d.prefill_shadow_cap, 0);
    }

    /// **自洽门**:影子开启必须给容量 —— 否则首层按 0 预留、开关静默不生效。
    /// 红档不写任何状态(先判后写),故本测试不污染进程。
    #[test]
    fn 影子无容量响亮拒() {
        let e = configure(FlashinferCfg { prefill_shadow: true, ..Default::default() })
            .expect_err("cap = 0 时必须拒");
        assert!(format!("{e}").contains("prefill_shadow_cap"), "错误要点名字段: {e}");
    }

    /// **一次落全字段**(写者唯一的可观测后果);跑完还原缺省,不留给兄弟测试。
    #[test]
    fn 一次落全字段且可还原() {
        let restore = FlashinferCfg {
            plan_restore: plan_restore(),
            cuda_graph: cuda_graph(),
            prefill_shadow: prefill_shadow_on(),
            prefill_shadow_cap: prefill_shadow_cap(),
            fp8_native: fp8_native(),
        };
        configure(FlashinferCfg {
            plan_restore: false,
            cuda_graph: true,
            prefill_shadow: true,
            prefill_shadow_cap: 4096,
            fp8_native: false,
        })
        .expect("合法组合");
        assert!(!plan_restore());
        assert!(cuda_graph());
        assert!(prefill_shadow_on());
        assert_eq!(prefill_shadow_cap(), 4096);
        // C++ 适配器的唯一通道(env)必须与 Rust 原子同一次落
        assert_eq!(std::env::var("FI_CUDA_GRAPH").as_deref(), Ok("1"));
        configure(restore).expect("还原");
    }
}
