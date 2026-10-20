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
use owl_macros::kernel_tags;

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
        #[kernel_tags("dom.gguf", "quant.gguf", "role.gemm")]
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

// ---- **head 版 int8-MMA MMQ**(W-2f;设计 §二十二)---------------------------
//
// 槽序与 `.cu` 实例逐参对位:
//   量化:`x[F16,2], xq[U8,2], n_tokens(i32), ne0(i32)`
//   GEMM:`weights[U8,2], xq[U8,2], out[F16,2], n(i32), m(i32), k(i32)`

/// `owl_mmq2_quant_q8_1_ds4`(激活 → q8_1_mmq;DS4 布局)
pub struct GgufMmq2QuantDs4;
/// `owl_mmq2_quant_q8_1_d4`(同上;D4 布局)
pub struct GgufMmq2QuantD4;

const MMQ2_QUANT_PARAMS: &[Param] = &[
    // 名字逐字对 `.cu` 实例(`vx`/`vy`;2026-10-20 互证门修正)
    Param::t_in("vx", Dtype::F16, 2),
    Param::t_out("vy", Dtype::U8, 2),
    Param::i32("n_tokens"),
    Param::i32("ne0"),
];

macro_rules! gguf_mmq2_quant_spec {
    ($t:ty, $entry:expr) => {
        #[kernel_tags("dom.gguf", "quant.gguf", "role.elem")]
        impl KernelSpec for $t {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = crate::family::gguf::MMQ2_SRC;
            const PARAMS: &'static [Param] = MMQ2_QUANT_PARAMS;
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

gguf_mmq2_quant_spec!(GgufMmq2QuantDs4, crate::family::gguf::MMQ2_QUANT_DS4);
gguf_mmq2_quant_spec!(GgufMmq2QuantD4, crate::family::gguf::MMQ2_QUANT_D4);

/// `owl_mmq2_q8_0_j128` 等 8 个(type × J)
pub struct GgufMmq2Q8_0J64;
/// 同上
pub struct GgufMmq2Q8_0J128;
/// 同上
pub struct GgufMmq2Q4KJ64;
/// 同上
pub struct GgufMmq2Q4KJ128;
/// 同上
pub struct GgufMmq2Q5KJ64;
/// 同上
pub struct GgufMmq2Q5KJ128;
/// 同上
pub struct GgufMmq2Q6KJ64;
/// 同上
pub struct GgufMmq2Q6KJ128;

const MMQ2_PARAMS: &[Param] = &[
    // 名字逐字对 `.cu` 实例(`vx`/`vy`;2026-10-20 互证门修正)
    Param::t_in("vx", Dtype::U8, 2),
    Param::t_in("vy", Dtype::U8, 2),
    Param::t_out("out", Dtype::F16, 2),
    Param::i32("n"),
    Param::i32("m"),
    Param::i32("k"),
];

macro_rules! gguf_mmq2_spec {
    ($t:ty, $entry:expr) => {
        #[kernel_tags("dom.gguf", "quant.gguf", "role.gemm")]
        impl KernelSpec for $t {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = crate::family::gguf::MMQ2_SRC;
            const PARAMS: &'static [Param] = MMQ2_PARAMS;
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

gguf_mmq2_spec!(GgufMmq2Q8_0J64, crate::family::gguf::MMQ2_Q8_0_J64);
gguf_mmq2_spec!(GgufMmq2Q8_0J128, crate::family::gguf::MMQ2_Q8_0_J128);
gguf_mmq2_spec!(GgufMmq2Q4KJ64, crate::family::gguf::MMQ2_Q4K_J64);
gguf_mmq2_spec!(GgufMmq2Q4KJ128, crate::family::gguf::MMQ2_Q4K_J128);
gguf_mmq2_spec!(GgufMmq2Q5KJ64, crate::family::gguf::MMQ2_Q5K_J64);
gguf_mmq2_spec!(GgufMmq2Q5KJ128, crate::family::gguf::MMQ2_Q5K_J128);
gguf_mmq2_spec!(GgufMmq2Q6KJ64, crate::family::gguf::MMQ2_Q6K_J64);
gguf_mmq2_spec!(GgufMmq2Q6KJ128, crate::family::gguf::MMQ2_Q6K_J128);

// ---- **行窗解量化**(W-2d;dequant+cuBLAS 的喂料手)———————————————
//
// 槽序:`w[U8 块], out[F16], n0(i32), rows(i32), k(i32)`
// —— 输出块是**调用方给的 scratch**(`Op::CallDst` 的 dest,不分配)。

/// `owl_gguf_dequant_range_q8_0`
pub struct GgufRangeQ8_0;
/// 同上(Q4_K)
pub struct GgufRangeQ4K;
/// 同上(Q5_K)
pub struct GgufRangeQ5K;
/// 同上(Q6_K)
pub struct GgufRangeQ6K;

const RANGE_PARAMS: &[Param] = &[
    // 名字逐字对 `.cu` 实例(`w`,不是 `weights`;2026-10-20 互证门修正)
    Param::t_in("w", Dtype::U8, 2),
    Param::t_out("out", Dtype::F16, 2),
    Param::i32("n0"),
    Param::i32("rows"),
    Param::i32("k"),
];

macro_rules! gguf_range_spec {
    ($t:ty, $entry:expr) => {
        #[kernel_tags("dom.gguf", "quant.gguf", "role.elem")]
        impl KernelSpec for $t {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = crate::family::gguf::SMEM_SRC;
            const PARAMS: &'static [Param] = RANGE_PARAMS;
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

gguf_range_spec!(GgufRangeQ8_0, crate::family::gguf::RANGE_Q8_0);
gguf_range_spec!(GgufRangeQ4K, crate::family::gguf::RANGE_Q4K);
gguf_range_spec!(GgufRangeQ5K, crate::family::gguf::RANGE_Q5K);
gguf_range_spec!(GgufRangeQ6K, crate::family::gguf::RANGE_Q6K);

// ---- **内维列序 gather**(W-2e;`ssm_out` tiled → grouped)------------------------
//
// 槽序与 `.cu` 实例同形:`x[F16,2], tbl[U32,1], out[F16,2], rows, k`
// (表 = 32 元素块序表;`out[.., 32·i] = x[.., 32·tbl[i]]`)

/// `owl_gather_cols_f16`
pub struct GgufGatherCols;

#[kernel_tags("dom.gguf", "quant.gguf", "role.elem")]
impl KernelSpec for GgufGatherCols {
    const ENTRY: &'static str = crate::family::gguf::GATHER_COLS_F16;
    const SRC: &'static str = crate::family::gguf::SMEM_SRC;
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("tbl", Dtype::U32, 1),
        Param::t_out("out", Dtype::F16, 2),
        Param::i32("rows"),
        Param::i32("k"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

// ---- **官方 MMQ**(W-2c;tensor-core prefill)———————————————
//
// 槽序 = 适配 TU `gguf_mmq.cu` 的形参序(6 参:3 张量 + 3 标量);几何由
// `family::gguf::mmq_geom` 按上游 ampere 档 `{x=4, y=32, nwarps=4}` 算。

/// `owl_gguf_mmq_q8_0`(llama.cpp `mul_mat_q<Q8_0>`)
pub struct GgufMmqQ8_0;
/// 同上(Q4_K)
pub struct GgufMmqQ4K;
/// 同上(Q5_K)
pub struct GgufMmqQ5K;
/// 同上(Q6_K)
pub struct GgufMmqQ6K;

const MMQ_PARAMS: &[Param] = &[
    Param::t_in("weights", Dtype::U8, 2),
    // 激活 q8_1(36B 块;**上游同代布局**,本仓 `GgufQuantAct*` 已产)
    Param::t_in("xq", Dtype::U8, 2),
    Param::t_out("out", Dtype::F16, 2),
    Param::i32("k"),
    Param::i32("n"),
    Param::i32("m"),
];

macro_rules! gguf_mmq_spec {
    ($t:ty, $entry:expr) => {
        #[kernel_tags("dom.gguf", "quant.gguf", "role.gemm")]
        impl KernelSpec for $t {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = crate::family::gguf::MMQ_SRC;
            const PARAMS: &'static [Param] = MMQ_PARAMS;
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

gguf_mmq_spec!(GgufMmqQ8_0, crate::family::gguf::MMQ_Q8_0);
gguf_mmq_spec!(GgufMmqQ4K, crate::family::gguf::MMQ_Q4K);
gguf_mmq_spec!(GgufMmqQ5K, crate::family::gguf::MMQ_Q5K);
gguf_mmq_spec!(GgufMmqQ6K, crate::family::gguf::MMQ_Q6K);

// ---- **smem 复用臂(W-2b;prefill 的算术杠杆)** ————————————————
//
// 与 MMVQ/复用臂**不同槽序**:本臂吃**原始 f16 激活**(不经 q8_1)——
// 少一发量化核,且比 q8_1 更准;权重仍是 GGUF 块字节。

/// `owl_gguf_gemm_smem_q8_0` —— smem 复用 GEMM(Q8_0)
pub struct GgufSmemQ8_0;
/// 同上(Q4_K)
pub struct GgufSmemQ4K;
/// 同上(Q5_K)
pub struct GgufSmemQ5K;
/// 同上(Q6_K)
pub struct GgufSmemQ6K;

/// smem 臂槽序(与 `gguf_smem.cu` 实例行逐参对位)
const SMEM_PARAMS: &[Param] = &[
    Param::t_in("weights", Dtype::U8, 2),
    Param::t_in("x", Dtype::F16, 2),
    Param::t_out("out", Dtype::F16, 2),
    Param::i32("m"),
    Param::i32("n"),
    Param::i32("k"),
    Param::i32("out_off"),
    Param::i32("out_ld"),
];

macro_rules! gguf_smem_spec {
    ($t:ty, $entry:expr) => {
        #[kernel_tags("dom.gguf", "quant.gguf", "role.gemm")]
        impl KernelSpec for $t {
            const ENTRY: &'static str = $entry;
            const SRC: &'static str = crate::family::gguf::SMEM_SRC;
            const PARAMS: &'static [Param] = SMEM_PARAMS;
            const LAWS: &'static [Law] = &[Law::OutputFinite];
            const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
            const DETERMINISM: Determinism = Determinism::Deterministic;
        }
    };
}

gguf_smem_spec!(GgufSmemQ8_0, crate::family::gguf::GEMM_SMEM_Q8_0);
gguf_smem_spec!(GgufSmemQ4K, crate::family::gguf::GEMM_SMEM_Q4K);
gguf_smem_spec!(GgufSmemQ5K, crate::family::gguf::GEMM_SMEM_Q5K);
gguf_smem_spec!(GgufSmemQ6K, crate::family::gguf::GEMM_SMEM_Q6K);

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
        #[kernel_tags("dom.gguf", "quant.gguf", "role.gemm")]
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

#[kernel_tags("dom.gguf", "quant.gguf", "role.elem")]
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

#[kernel_tags("dom.gguf", "quant.gguf", "role.elem")]
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

#[kernel_tags("dom.gguf", "quant.gguf", "role.elem")]
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

#[kernel_tags("dom.gguf", "quant.gguf", "role.elem")]
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
        #[kernel_tags("dom.gguf", "quant.gguf", "role.elem", "phase.load")]
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

// ---- 行窗解量化的元数据表项 ----
pub static RANGE_Q8_0: KernelMeta = meta_of::<GgufRangeQ8_0>();
pub static RANGE_Q4K: KernelMeta = meta_of::<GgufRangeQ4K>();
pub static RANGE_Q5K: KernelMeta = meta_of::<GgufRangeQ5K>();
pub static RANGE_Q6K: KernelMeta = meta_of::<GgufRangeQ6K>();

// ---- 内维列序 gather 的元数据表项(W-2e)----
pub static GATHER_COLS_F16: KernelMeta = meta_of::<GgufGatherCols>();

// ---- head 版 int8-MMA MMQ 的元数据表项(W-2f)----
pub static MMQ2_QUANT_DS4: KernelMeta = meta_of::<GgufMmq2QuantDs4>();
pub static MMQ2_QUANT_D4: KernelMeta = meta_of::<GgufMmq2QuantD4>();
pub static MMQ2_Q8_0_J64: KernelMeta = meta_of::<GgufMmq2Q8_0J64>();
pub static MMQ2_Q8_0_J128: KernelMeta = meta_of::<GgufMmq2Q8_0J128>();
pub static MMQ2_Q4K_J64: KernelMeta = meta_of::<GgufMmq2Q4KJ64>();
pub static MMQ2_Q4K_J128: KernelMeta = meta_of::<GgufMmq2Q4KJ128>();
pub static MMQ2_Q5K_J64: KernelMeta = meta_of::<GgufMmq2Q5KJ64>();
pub static MMQ2_Q5K_J128: KernelMeta = meta_of::<GgufMmq2Q5KJ128>();
pub static MMQ2_Q6K_J64: KernelMeta = meta_of::<GgufMmq2Q6KJ64>();
pub static MMQ2_Q6K_J128: KernelMeta = meta_of::<GgufMmq2Q6KJ128>();

// ---- MMQ 的元数据表项 ----
pub static MMQ_Q8_0: KernelMeta = meta_of::<GgufMmqQ8_0>();
pub static MMQ_Q4K: KernelMeta = meta_of::<GgufMmqQ4K>();
pub static MMQ_Q5K: KernelMeta = meta_of::<GgufMmqQ5K>();
pub static MMQ_Q6K: KernelMeta = meta_of::<GgufMmqQ6K>();

// ---- smem 复用臂的元数据表项 ----
pub static SMEM_Q8_0: KernelMeta = meta_of::<GgufSmemQ8_0>();
pub static SMEM_Q4K: KernelMeta = meta_of::<GgufSmemQ4K>();
pub static SMEM_Q5K: KernelMeta = meta_of::<GgufSmemQ5K>();
pub static SMEM_Q6K: KernelMeta = meta_of::<GgufSmemQ6K>();
