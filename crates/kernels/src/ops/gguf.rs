//! **GGUF 算子元数据**(2026-10-19 gguf 轮;设计 §四 L3)。
//!
//! 两条 H3 要求的落地:**逐型一条声明**(不许裸 int 路由)—— 每型一个
//! `KernelSpec`,`ENTRY` 取自 [`crate::family::gguf`] 的**唯一映射表**,
//! `SRC` 指向同一份 `.cu`(源里以宏实例出逐型入口)。
//!
//! 槽序 = `.cu` 实例行的形参序(`tests/op_tables.rs` 机器互证):
//!
//! ```text
//! gemm:      w[U8 块] , xq[U8 q8_1] , out[F16] , m,n,k,k_padded,out_off,out_ld
//! quant_act: x[F16/BF16] , xq[U8 q8_1] , kx, kx_padded
//! quant_act_tiled: x[F16/BF16] , tbl[U32] , xq[U8 q8_1] , kx, kx_padded
//! dequant_rows: w[U8 块] , ids[F32] , out[F16] , k , rows , vocab
//! ```
//!
//! `Dtype::U8` 是**正确选择**而非兜底:GGUF 块字节与 q8_1 块都是
//! "真正不透明的字节块"(2026-10-16 收窄纪律:fp8/int8 必须点名,其余
//! 不透明块才用 U8)。
//!
//! **`out_off`/`out_ld` 的存在理由**:GDN 的 `attn_qkv ⊕ attn_gate` 两段
//! **异型**(4B 实测 24/24 层 Q6_K vs Q4_K)⇒ 无法合成单一同型张量 ⇒ 分段
//! 发射,各写输出张量的不同列窗(设计 §十一 11.5)。

use crate::contract::Dtype;
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};

/// `owl_gguf_gemm_q8_0` —— GGUF Q8_0 dequant-in-kernel GEMM(MMVQ)
pub struct GgufGemmQ8_0;
/// `owl_gguf_gemm_q4k` —— Q4_K(超块 6bit scale/min 仿射)
pub struct GgufGemmQ4K;
/// `owl_gguf_gemm_q5k` —— Q5_K(5bit 权重 + 同套 scale/min)
pub struct GgufGemmQ5K;
/// `owl_gguf_gemm_q6k` —— Q6_K(6bit 权重;`d` 在**块尾**)
pub struct GgufGemmQ6K;

/// GEMM 公共参数表(四型同形;名字逐字对 `.cu` 实例行)
const GEMM_PARAMS: &[Param] = &[
    // 权重 = GGUF 块字节(**零解量化驻留**;行 = 权重的外维)
    Param::t_in("weights", Dtype::U8, 2),
    // 激活 = q8_1 块(`owl_gguf_quant_act_*` 的产物)
    Param::t_in("xq", Dtype::U8, 2),
    Param::t_out("out", Dtype::F16, 2),
    Param::i32("m"),
    Param::i32("n"),
    Param::i32("k"),
    Param::i32("k_padded"),
    Param::i32("out_off"),
    Param::i32("out_ld"),
];

macro_rules! gguf_gemm_spec {
    ($t:ty, $entry:expr) => {
        impl KernelSpec for $t {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = crate::family::gguf::GEMM_SRC;
            const PARAMS: &'static [Param] = GEMM_PARAMS;
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            // 纯读:总分配口在池(激活 q8_1 缓冲由调用方传入)⇒ 无 host 工作
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

use crate::contract::Law;

gguf_gemm_spec!(GgufGemmQ8_0, crate::family::gguf::GEMM_Q8_0);
gguf_gemm_spec!(GgufGemmQ4K, crate::family::gguf::GEMM_Q4K);
gguf_gemm_spec!(GgufGemmQ5K, crate::family::gguf::GEMM_Q5K);
gguf_gemm_spec!(GgufGemmQ6K, crate::family::gguf::GEMM_Q6K);

// ---- **复用臂(W-2;prefill)** ————————————————————————————————
//
// 与 MMVQ **同槽序**(`GEMM_PARAMS` 逐字相同)—— 两者是同一语义动作的两种
// 调度:MMVQ = 每 (行, token) 读一遍权重;复用臂 = 一行权重算 `TILE_MT` 个
// token。故声明面只差 `ENTRY`/`SRC`/`PARAMS` 之外的几何。

/// `owl_gguf_gemm_tile_q8_0` —— 权重复用 GEMM(Q8_0;W-2 prefill 臂)
pub struct GgufTileQ8_0;
/// 同上(Q4_K)
pub struct GgufTileQ4K;
/// 同上(Q5_K)
pub struct GgufTileQ5K;
/// 同上(Q6_K)
pub struct GgufTileQ6K;

macro_rules! gguf_tile_spec {
    ($t:ty, $entry:expr) => {
        impl KernelSpec for $t {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = crate::family::gguf::TILE_SRC;
            const PARAMS: &'static [Param] = GEMM_PARAMS;
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

gguf_tile_spec!(GgufTileQ8_0, crate::family::gguf::GEMM_TILE_Q8_0);
gguf_tile_spec!(GgufTileQ4K, crate::family::gguf::GEMM_TILE_Q4K);
gguf_tile_spec!(GgufTileQ5K, crate::family::gguf::GEMM_TILE_Q5K);
gguf_tile_spec!(GgufTileQ6K, crate::family::gguf::GEMM_TILE_Q6K);

/// `owl_gguf_quant_act_f16` —— 激活 → q8_1 块(f16 入;MMVQ 喂料手)
pub struct GgufQuantActF16;
/// `owl_gguf_quant_act_bf16` —— 同上,bf16 入
pub struct GgufQuantActBf16;

impl KernelSpec for GgufQuantActF16 {
    const ENTRY: &'static str = crate::family::gguf::QUANT_ACT_F16;
    const SRC: &'static str = crate::family::gguf::QUANT_SRC;
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_out("xq", Dtype::U8, 2),
        Param::i32("kx"),
        Param::i32("kx_padded"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

impl KernelSpec for GgufQuantActBf16 {
    const ENTRY: &'static str = crate::family::gguf::QUANT_ACT_BF16;
    const SRC: &'static str = crate::family::gguf::QUANT_SRC;
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::t_out("xq", Dtype::U8, 2),
        Param::i32("kx"),
        Param::i32("kx_padded"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static GEMM_Q8_0: KernelMeta = meta_of::<GgufGemmQ8_0>();
pub static GEMM_Q4K: KernelMeta = meta_of::<GgufGemmQ4K>();
pub static GEMM_Q5K: KernelMeta = meta_of::<GgufGemmQ5K>();
pub static GEMM_Q6K: KernelMeta = meta_of::<GgufGemmQ6K>();
pub static QUANT_ACT_F16: KernelMeta = meta_of::<GgufQuantActF16>();
pub static QUANT_ACT_BF16: KernelMeta = meta_of::<GgufQuantActBf16>();

// ---------------------------------------------------------------------------
// **激活量化的 tiled 变体**(v-head 列序;设计 §11.4 落点 A2)
//
// 为什么要它:GGUF 转换器把 `ssm_out` 的内维(v-head 轴)写成 ggml **tiled** 序
// (`conversion/qwen.py::_reorder_v_heads`),而 owl 的 GDN 核产出 **grouped** 序。
// 该置换**跨量化块**(`hv = 128` < `QK_K = 256`)⇒ 权重字节不可逆 ⇒ 在激活侧
// 做**等价置换**。落点选在本族(而非独立 gather 算子):量化核已在逐元素读
// 激活、按 32 元素块写 q8_1 —— 把“读哪个块”换成查表,零额外发射零额外趟数。
//
// 与“先 gather 再量化”**逐位等价**:置换单元 = 32 元素块,块内容不变 ⇒
// 量化(块内 max/和)不变。表 = U32[kx/32],由 host 侧 `transform::vhead_q8_blocks`
// 单一公式生成,以常驻显存形态由权重槽供给(`blk.N.ssm_out.perm` 虚拟键)。
// ---------------------------------------------------------------------------

/// `owl_gguf_quant_act_f16_tiled` —— 激活 → q8_1(f16 入;**按块表取源块**)
pub struct GgufQuantActF16Tiled;
/// 同上,bf16 入
pub struct GgufQuantActBf16Tiled;

impl KernelSpec for GgufQuantActF16Tiled {
    const ENTRY: &'static str = crate::family::gguf::QUANT_ACT_F16_TILED;
    const SRC: &'static str = crate::family::gguf::QUANT_SRC;
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        // 块序表(`kx / 32` 项;源侧虚拟键生成、常驻显存)
        Param::t_in("tbl", Dtype::U32, 1),
        Param::t_out("xq", Dtype::U8, 2),
        Param::i32("kx"),
        Param::i32("kx_padded"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

impl KernelSpec for GgufQuantActBf16Tiled {
    const ENTRY: &'static str = crate::family::gguf::QUANT_ACT_BF16_TILED;
    const SRC: &'static str = crate::family::gguf::QUANT_SRC;
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::t_in("tbl", Dtype::U32, 1),
        Param::t_out("xq", Dtype::U8, 2),
        Param::i32("kx"),
        Param::i32("kx_padded"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static QUANT_ACT_F16_TILED: KernelMeta = meta_of::<GgufQuantActF16Tiled>();
pub static QUANT_ACT_BF16_TILED: KernelMeta = meta_of::<GgufQuantActBf16Tiled>();

// ---------------------------------------------------------------------------
// 行解量化(查表臂;`token_embd` 是量化表 —— 设计 §四 L3 / §11.5 ②)
//
// 槽序 = `.cu` 实例行形参序: `w[U8 块] , ids[F32] , out[F16] , k , rows , vocab`
// 几何 = `family::gguf::rows_geom`(`grid = (tokens, k/qk)`、`block = (该型块线程数)`)
// -----------------------------------------------------------------------------

/// `owl_gguf_dequant_rows_q4k` —— 量化表行解量化(Q4_K)
pub struct GgufRowsQ4K;
/// `owl_gguf_dequant_rows_q5k` —— 同上(Q5_K)
pub struct GgufRowsQ5K;
/// `owl_gguf_dequant_rows_q6k` —— 同上(Q6_K)
pub struct GgufRowsQ6K;

/// 行解量化公共参数表(三型同形;名字逐字对 `.cu` 实例行)
const ROWS_PARAMS: &[Param] = &[
    // 量化表(块字节;行 = 词表项)
    Param::t_in("w", Dtype::U8, 2),
    // 词表 id(f32 承载设备侧索引 —— 与 `owl_embed_*` 同约定)
    Param::t_in("ids", Dtype::F32, 0),
    Param::t_out("out", Dtype::F16, 2),
    Param::i32("k"),
    Param::i32("rows"),
    Param::i32("vocab"),
];

macro_rules! gguf_rows_spec {
    ($t:ty, $entry:expr) => {
        impl KernelSpec for $t {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = crate::family::gguf::ROWS_SRC;
            const PARAMS: &'static [Param] = ROWS_PARAMS;
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

gguf_rows_spec!(GgufRowsQ4K, crate::family::gguf::ROWS_Q4K);
gguf_rows_spec!(GgufRowsQ5K, crate::family::gguf::ROWS_Q5K);
gguf_rows_spec!(GgufRowsQ6K, crate::family::gguf::ROWS_Q6K);

pub static ROWS_Q4K: KernelMeta = meta_of::<GgufRowsQ4K>();
pub static ROWS_Q5K: KernelMeta = meta_of::<GgufRowsQ5K>();
pub static ROWS_Q6K: KernelMeta = meta_of::<GgufRowsQ6K>();

// ---- 复用臂(W-2):四个 prefill 入口的元数据表项 ----
pub static TILE_Q8_0: KernelMeta = meta_of::<GgufTileQ8_0>();
pub static TILE_Q4K: KernelMeta = meta_of::<GgufTileQ4K>();
pub static TILE_Q5K: KernelMeta = meta_of::<GgufTileQ5K>();
pub static TILE_Q6K: KernelMeta = meta_of::<GgufTileQ6K>();

pub static METAS: &[&KernelMeta] = &[
    &GEMM_Q8_0,
    &GEMM_Q4K,
    &GEMM_Q5K,
    &GEMM_Q6K,
    &TILE_Q8_0,
    &TILE_Q4K,
    &TILE_Q5K,
    &TILE_Q6K,
    &QUANT_ACT_F16,
    &QUANT_ACT_BF16,
    &QUANT_ACT_F16_TILED,
    &QUANT_ACT_BF16_TILED,
    &ROWS_Q4K,
    &ROWS_Q5K,
    &ROWS_Q6K,
];
