//! GGUF 量化型族(**格式词汇 + 块几何 + 内核码唯一映射**;2026-10-19 gguf 轮)。
//!
//! 三件事一个住址(**设计 H3:禁裸 int 路由**):
//!
//! 1. **格式词汇** [`GgmlTy`] —— 判别值 = GGUF 文件头里的 `ggml_type` 编码,
//!    权威源 = llama.cpp `enum ggml_type`;
//! 2. **块几何** [`GgmlTy::block`] = (每块元素数, 每块字节数)—— 解析器算张量
//!    字节数与行跨度的**唯一依据**;数值逐条对 `llama.cpp/gguf-py` 的
//!    `GGML_QUANT_SIZES`(见本文件 `tests`,改表必须连测试一起改);
//! 3. **内核码** [`GgmlTy::kernel_code`] ↔ `GgmlTy` 的**双射** ——
//!    `cu/gguf/gguf_gemm.cu` 的 `quant_type` 分派、`ops/gguf.rs` 的准入声明、
//!    装载侧的"能不能跑"判定,全部引用本表,**不许第二份**。
//!
//! **零门零依赖**(族身份面;`family/mod.rs` 纪律:声明面零门,FFI/封装面随
//! feature 门)。
//!
//! # 命名
//!
//! Rust 侧用驼峰(`Q4K`/`Iq4Xs`),`name()` 给回 **GGUF 官方拼法**
//! (`Q4_K`/`IQ4_XS`)—— 诊断/对拍/日志一律用后者(与 `gguf-py` 逐字可比)。

use crate::contract::Dtype;

/// `QK_K`:K-quant 超块元素数(固定 256)。**禁 `GGML_QKK_64` 变体** ——
/// 那会凭空多一倍模板实例,且 GGUF 分发件一律 256。
pub const QK_K: usize = 256;

/// 均匀块元素数(Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/IQ4_NL/MXFP4)
pub const QK_32: usize = 32;

/// ggml 张量型(**GGUF 格式词汇**;判别值 = 文件里的编码)。
///
/// 覆盖到"能解析、能被响亮拒"为止 —— **不求全部可跑**:可跑性由
/// [`GgmlTy::kernel_code`] 回答(设计:解析与执行能力分离,解析不因
/// 型未知而失败)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlTy {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K = 10,
    Q3K = 11,
    Q4K = 12,
    Q5K = 13,
    Q6K = 14,
    Q8K = 15,
    Iq2Xxs = 16,
    Iq2Xs = 17,
    Iq3Xxs = 18,
    Iq1S = 19,
    Iq4Nl = 20,
    Iq3S = 21,
    Iq2S = 22,
    Iq4Xs = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    Iq1M = 29,
    Bf16 = 30,
    Tq1_0 = 34,
    Tq2_0 = 35,
    Mxfp4 = 39,
    Nvfp4 = 40,
    Q1_0 = 41,
    Q2_0 = 42,
}

/// 全表(顺序 = 编码序;诊断/枚举用)
pub const ALL: &[GgmlTy] = &[
    GgmlTy::F32,
    GgmlTy::F16,
    GgmlTy::Q4_0,
    GgmlTy::Q4_1,
    GgmlTy::Q5_0,
    GgmlTy::Q5_1,
    GgmlTy::Q8_0,
    GgmlTy::Q8_1,
    GgmlTy::Q2K,
    GgmlTy::Q3K,
    GgmlTy::Q4K,
    GgmlTy::Q5K,
    GgmlTy::Q6K,
    GgmlTy::Q8K,
    GgmlTy::Iq2Xxs,
    GgmlTy::Iq2Xs,
    GgmlTy::Iq3Xxs,
    GgmlTy::Iq1S,
    GgmlTy::Iq4Nl,
    GgmlTy::Iq3S,
    GgmlTy::Iq2S,
    GgmlTy::Iq4Xs,
    GgmlTy::I8,
    GgmlTy::I16,
    GgmlTy::I32,
    GgmlTy::I64,
    GgmlTy::F64,
    GgmlTy::Iq1M,
    GgmlTy::Bf16,
    GgmlTy::Tq1_0,
    GgmlTy::Tq2_0,
    GgmlTy::Mxfp4,
    GgmlTy::Nvfp4,
    GgmlTy::Q1_0,
    GgmlTy::Q2_0,
];

impl GgmlTy {
    /// 文件编码 → 词汇;未知编码 = `None`(**调用方响亮拒**,点名数字)
    pub fn from_code(code: u32) -> Option<Self> {
        ALL.iter().copied().find(|t| t.code() == code)
    }

    /// 词汇 → 文件编码
    pub const fn code(self) -> u32 {
        self as u32
    }

    /// GGUF 官方拼法(诊断/对拍用;与 `gguf-py` 逐字可比)
    pub const fn name(self) -> &'static str {
        match self {
            GgmlTy::F32 => "F32",
            GgmlTy::F16 => "F16",
            GgmlTy::Q4_0 => "Q4_0",
            GgmlTy::Q4_1 => "Q4_1",
            GgmlTy::Q5_0 => "Q5_0",
            GgmlTy::Q5_1 => "Q5_1",
            GgmlTy::Q8_0 => "Q8_0",
            GgmlTy::Q8_1 => "Q8_1",
            GgmlTy::Q2K => "Q2_K",
            GgmlTy::Q3K => "Q3_K",
            GgmlTy::Q4K => "Q4_K",
            GgmlTy::Q5K => "Q5_K",
            GgmlTy::Q6K => "Q6_K",
            GgmlTy::Q8K => "Q8_K",
            GgmlTy::Iq2Xxs => "IQ2_XXS",
            GgmlTy::Iq2Xs => "IQ2_XS",
            GgmlTy::Iq3Xxs => "IQ3_XXS",
            GgmlTy::Iq1S => "IQ1_S",
            GgmlTy::Iq4Nl => "IQ4_NL",
            GgmlTy::Iq3S => "IQ3_S",
            GgmlTy::Iq2S => "IQ2_S",
            GgmlTy::Iq4Xs => "IQ4_XS",
            GgmlTy::I8 => "I8",
            GgmlTy::I16 => "I16",
            GgmlTy::I32 => "I32",
            GgmlTy::I64 => "I64",
            GgmlTy::F64 => "F64",
            GgmlTy::Iq1M => "IQ1_M",
            GgmlTy::Bf16 => "BF16",
            GgmlTy::Tq1_0 => "TQ1_0",
            GgmlTy::Tq2_0 => "TQ2_0",
            GgmlTy::Mxfp4 => "MXFP4",
            GgmlTy::Nvfp4 => "NVFP4",
            GgmlTy::Q1_0 => "Q1_0",
            GgmlTy::Q2_0 => "Q2_0",
        }
    }

    /// **块几何** = (每块元素数, 每块字节数)。
    ///
    /// 数值 = `gguf-py::GGML_QUANT_SIZES`(逐条;见 `tests::block_table_matches_gguf_py`)。
    /// 张量字节数 = `elems / block.0 * block.1`(元素数不整除块 = **调用方必须有为**,
    /// 本函数不兜)。
    pub const fn block(self) -> (usize, usize) {
        match self {
            // ---- 原生(逐元素)----
            GgmlTy::F32 => (1, 4),
            GgmlTy::F16 => (1, 2),
            GgmlTy::Bf16 => (1, 2),
            GgmlTy::F64 => (1, 8),
            GgmlTy::I8 => (1, 1),
            GgmlTy::I16 => (1, 2),
            GgmlTy::I32 => (1, 4),
            GgmlTy::I64 => (1, 8),
            // ---- 32 元素均匀块(scale ± min,或 4bit 索引)----
            GgmlTy::Q4_0 => (32, 2 + 16),
            GgmlTy::Q4_1 => (32, 2 + 2 + 16),
            GgmlTy::Q5_0 => (32, 2 + 4 + 16),
            GgmlTy::Q5_1 => (32, 2 + 2 + 4 + 16),
            GgmlTy::Q8_0 => (32, 2 + 32),
            GgmlTy::Q8_1 => (32, 4 + 4 + 32),
            GgmlTy::Iq4Nl => (32, 2 + 16),
            GgmlTy::Mxfp4 => (32, 1 + 16),
            // ---- 256 超块(K-quant:6bit scale/min;IQ:codebook 索引)----
            GgmlTy::Q2K => (QK_K, 2 + 2 + QK_K / 16 + QK_K / 4),
            GgmlTy::Q3K => (QK_K, 2 + QK_K / 4 + QK_K / 8 + 12),
            GgmlTy::Q4K => (QK_K, 2 + 2 + QK_K / 2 + 12),
            GgmlTy::Q5K => (QK_K, 2 + 2 + QK_K / 2 + QK_K / 8 + 12),
            GgmlTy::Q6K => (QK_K, 2 + QK_K / 2 + QK_K / 4 + QK_K / 16),
            GgmlTy::Q8K => (QK_K, 4 + QK_K + QK_K / 8),
            GgmlTy::Iq2Xxs => (QK_K, 2 + QK_K / 4),
            GgmlTy::Iq2Xs => (QK_K, 2 + QK_K / 4 + QK_K / 32),
            GgmlTy::Iq3Xxs => (QK_K, 2 + QK_K / 4 + QK_K / 8),
            GgmlTy::Iq1S => (QK_K, 2 + QK_K / 8 + QK_K / 16),
            GgmlTy::Iq3S => (QK_K, 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4),
            GgmlTy::Iq2S => (QK_K, 2 + QK_K / 4 + QK_K / 16),
            GgmlTy::Iq4Xs => (QK_K, 2 + 2 + QK_K / 2 + QK_K / 64),
            GgmlTy::Iq1M => (QK_K, QK_K / 8 + QK_K / 16 + QK_K / 32),
            GgmlTy::Tq1_0 => (QK_K, 2 + 4 * 13),
            GgmlTy::Tq2_0 => (QK_K, 2 + 64),
            // ---- 长尾(本仓无样本;几何仍如实登记,解析不因未知而崩)----
            GgmlTy::Nvfp4 => (64, 4 + 32),
            GgmlTy::Q1_0 => (128, 2 + 16),
            GgmlTy::Q2_0 => (64, 2 + 16),
        }
    }

    /// 原生(逐元素、无块结构) dtype —— 走 **元素装载路径**(f16 基线/cast 臂)。
    /// `None` = 量化块型(必须走 GGUF 内核族,禁元素路径)。
    pub const fn native_dtype(self) -> Option<Dtype> {
        match self {
            GgmlTy::F32 => Some(Dtype::F32),
            GgmlTy::F16 => Some(Dtype::F16),
            GgmlTy::Bf16 => Some(Dtype::BF16),
            _ => None,
        }
    }


    /// **M1 验收集**:有数值锚(本机存量模型 / 已下样本)的型。
    ///
    /// ⚠️ 与 [`Self::kernel_code`] 的区别:后者是"**代码**能跑",本项是
    /// "**我们有对拍证据**"。IQ 四型在两列都缺席的只有 `Iq4Xs`/`Iq3Xxs` 等
    /// (样本待下载;设计 §八 R3)⇒ 未在验收集的型**不得宣称支持**(本仓纪律:
    /// 无法验证 = 无法宣称)。
    pub const M1_VALIDATED: &'static [GgmlTy] = &[
        GgmlTy::F32,
        GgmlTy::F16,
        GgmlTy::Bf16,
        GgmlTy::Q8_0,
        GgmlTy::Q4K,
        GgmlTy::Q5K,
        GgmlTy::Q6K,
    ];
}

// ============================================================================
// §2 内核入口名(**唯一映射表**;H3:准入表与 driver 都引用这里)
// ============================================================================

/// 资产 id(`KernelSpec::SRC` 与 `load_aot` 同源)
pub const GEMM_SRC: &str = "cu/gguf/gguf_gemm.cu";
/// 激活量化源的资产 id
pub const QUANT_SRC: &str = "cu/gguf/gguf_quant.cu";

/// GEMM 入口(q8_0)
pub const GEMM_Q8_0: &str = "owl_gguf_gemm_q8_0";
/// GEMM 入口(Q4_K)
pub const GEMM_Q4K: &str = "owl_gguf_gemm_q4k";
/// GEMM 入口(Q5_K)
pub const GEMM_Q5K: &str = "owl_gguf_gemm_q5k";
/// GEMM 入口(Q6_K)
pub const GEMM_Q6K: &str = "owl_gguf_gemm_q6k";
/// GEMM 入口(IQ4_NL;32 元素块 + 16 项码本)
pub const GEMM_IQ4NL: &str = "owl_gguf_gemm_iq4nl";
/// GEMM 入口(IQ4_XS;256 超块 + 码本 + 4bit 尺度)
pub const GEMM_IQ4XS: &str = "owl_gguf_gemm_iq4xs";
/// GEMM 入口(IQ3_XXS;256 超块 + 4×i8 码本 + 符号表)
pub const GEMM_IQ3XXS: &str = "owl_gguf_gemm_iq3xxs";
/// GEMM 入口(IQ3_S;256 超块 + 512 项码本)
pub const GEMM_IQ3S: &str = "owl_gguf_gemm_iq3s";
/// GEMM 入口(Q2_K;27B UD 档)
pub const GEMM_Q2K: &str = "owl_gguf_gemm_q2k";
/// GEMM 入口(Q3_K;27B UD 档)
pub const GEMM_Q3K: &str = "owl_gguf_gemm_q3k";
/// GEMM 入口(IQ2_XXS;27B UD 档)
pub const GEMM_Iq2Xxs: &str = "owl_gguf_gemm_iq2xxs";
/// GEMM 入口(IQ2_XS;27B UD 档)
pub const GEMM_Iq2Xs: &str = "owl_gguf_gemm_iq2xs";
/// GEMM 入口(IQ2_S;27B UD 档)
pub const GEMM_Iq2S: &str = "owl_gguf_gemm_iq2s";
/// GEMM 入口(IQ1_S;27B UD 档)
pub const GEMM_Iq1S: &str = "owl_gguf_gemm_iq1s";

/// 激活 → q8_1 入口(f16 入)
pub const QUANT_ACT_F16: &str = "owl_gguf_quant_act_f16";
/// 激活 → q8_1 入口(bf16 入)
pub const QUANT_ACT_BF16: &str = "owl_gguf_quant_act_bf16";
/// **行解量化**入口(Q4_K;查表臂 —— `token_embd` 是量化表)
pub const ROWS_Q4K: &str = "owl_gguf_dequant_rows_q4k";
/// 行解量化入口(Q5_K)
pub const ROWS_Q5K: &str = "owl_gguf_dequant_rows_q5k";
/// 行解量化入口(Q6_K)
pub const ROWS_Q6K: &str = "owl_gguf_dequant_rows_q6k";
/// 行解量化入口(IQ4_NL)
pub const ROWS_IQ4NL: &str = "owl_gguf_dequant_rows_iq4nl";
/// 行解量化入口(IQ4_XS)
pub const ROWS_IQ4XS: &str = "owl_gguf_dequant_rows_iq4xs";
/// 行解量化入口(IQ3_XXS)
pub const ROWS_IQ3XXS: &str = "owl_gguf_dequant_rows_iq3xxs";
/// 行解量化入口(IQ3_S)
pub const ROWS_IQ3S: &str = "owl_gguf_dequant_rows_iq3s";
/// 行解量化入口(Q2_K)
pub const ROWS_Q2K: &str = "owl_gguf_dequant_rows_q2k";
/// 行解量化入口(Q3_K)
pub const ROWS_Q3K: &str = "owl_gguf_dequant_rows_q3k";
/// 行解量化入口(IQ2_XXS)
pub const ROWS_Iq2Xxs: &str = "owl_gguf_dequant_rows_iq2xxs";
/// 行解量化入口(IQ2_XS)
pub const ROWS_Iq2Xs: &str = "owl_gguf_dequant_rows_iq2xs";
/// 行解量化入口(IQ2_S)
pub const ROWS_Iq2S: &str = "owl_gguf_dequant_rows_iq2s";
/// 行解量化入口(IQ1_S)
pub const ROWS_Iq1S: &str = "owl_gguf_dequant_rows_iq1s";

/// 行解量化的资产 id
pub const ROWS_SRC: &str = "cu/gguf/gguf_rows.cu";

/// **权重复用 GEMM(prefill 臂)入口名**(W-2;设计 §十六)
pub const GEMM_TILE_Q8_0: &str = "owl_gguf_gemm_tile_q8_0";
/// 同上(Q4_K)
pub const GEMM_TILE_Q4K: &str = "owl_gguf_gemm_tile_q4k";
/// 同上(Q5_K)
pub const GEMM_TILE_Q5K: &str = "owl_gguf_gemm_tile_q5k";
/// 同上(Q6_K)
pub const GEMM_TILE_Q6K: &str = "owl_gguf_gemm_tile_q6k";
/// 复用臂的资产 id
pub const TILE_SRC: &str = "cu/gguf/gguf_tile.cu";

/// **复用臂的 token 窗宽**(= `gguf_tile.cu` 的编译期 `OWL_GGUF_TILE_MT`)。
///
/// 同一事实两处落地(`.cu` 的 `#define` 与这里)⇒ `cu_sig` 面单测互锁
/// (`tile_mt_matches_cu`):改一处不改另一处 = 网格与核内循环错位(静默错值)。
pub const TILE_MT: usize = 16;

/// **激活侧列置换**入口(v-head 反重排;设计 §11.4)
pub const QUANT_ACT_F16_TILED: &str = "owl_gguf_quant_act_f16_tiled";
/// 同上,bf16 入
pub const QUANT_ACT_BF16_TILED: &str = "owl_gguf_quant_act_bf16_tiled";

// ---- **head 版 int8-MMA MMQ**(W-2f;设计 §二十二)----------------------------
//
// 与 §十八 的 `../vendor/mmq.cu`(2024-05 同代,dp4a)根本差别:本臂走
// `mma.sync.m16n8k16.s8`(sm80+ 的 int8 张量核)⇒ 同机实测 103 TFLOPS 等效
// (llama.cpp 同形状),且**零权重搬运**(不吃 dequant→f16)。
pub const MMQ2_SRC: &str = "cu/gguf/gguf_mmq2.cu";

/// 激活量化 f16 → `block_q8_1_mmq`(DS4 布局:Q4_K/Q5_K —— 要 sum 抵 min)
pub const MMQ2_QUANT_DS4: &str = "owl_mmq2_quant_q8_1_ds4";
/// 同上(D4 布局:Q8_0/Q6_K)
pub const MMQ2_QUANT_D4: &str = "owl_mmq2_quant_q8_1_d4";

/// `owl_mmq2_q8_0_j64`(J = 每 CTA 的 token 窗)
pub const MMQ2_Q8_0_J64: &str = "owl_mmq2_q8_0_j64";
/// 同上(J=128)
pub const MMQ2_Q8_0_J128: &str = "owl_mmq2_q8_0_j128";
/// Q4_K / J=64
pub const MMQ2_Q4K_J64: &str = "owl_mmq2_q4k_j64";
/// Q4_K / J=128
pub const MMQ2_Q4K_J128: &str = "owl_mmq2_q4k_j128";
/// Q5_K / J=64
pub const MMQ2_Q5K_J64: &str = "owl_mmq2_q5k_j64";
/// Q5_K / J=128
pub const MMQ2_Q5K_J128: &str = "owl_mmq2_q5k_j128";
/// Q6_K / J=64
pub const MMQ2_Q6K_J64: &str = "owl_mmq2_q6k_j64";
/// Q6_K / J=128
pub const MMQ2_Q6K_J128: &str = "owl_mmq2_q6k_j128";

/// 每 CTA 的权重行数(上游 ampere 档 `I = 128`;分派门要求 `n % I == 0`)
pub const MMQ2_I: usize = 128;
/// 每 CTA 线程数(上游 `nthreads = 256`,occupancy 1)
pub const MMQ2_THREADS: usize = 256;
/// 激活块:`block_q8_1_mmq` 字节数(16B 头 + 128 int8)
pub const MMQ2_Q8_1_BLOCK_BYTES: usize = 144;
/// 激活块覆盖的 k 宽(元素)
pub const MMQ2_QK: usize = 128;
/// smem 的 X 面字节(X tile:I × sram_stride × 4;四种型的 sram_stride 都是 76)
pub const MMQ2_SMEM_X: usize = MMQ2_I * 76 * 4;

/// 型 → (量化入口, MMQ 入口 J64/J128);`None` = 本臂未移植
pub const fn mmq2_entry_for(ty: GgmlTy, m: usize) -> Option<&'static str> {
    let (e64, e128) = match ty {
        GgmlTy::Q8_0 => (MMQ2_Q8_0_J64, MMQ2_Q8_0_J128),
        GgmlTy::Q4K => (MMQ2_Q4K_J64, MMQ2_Q4K_J128),
        GgmlTy::Q5K => (MMQ2_Q5K_J64, MMQ2_Q5K_J128),
        GgmlTy::Q6K => (MMQ2_Q6K_J64, MMQ2_Q6K_J128),
        _ => return None,
    };
    Some(if m >= MMQ2_I { e128 } else { e64 })
}

/// 型 → 激活量化入口(DS4 = Q4_K/Q5_K;D4 = Q8_0/Q6_K)
pub const fn mmq2_quant_entry_for(ty: GgmlTy) -> Option<&'static str> {
    match ty {
        GgmlTy::Q4K | GgmlTy::Q5K => Some(MMQ2_QUANT_DS4),
        GgmlTy::Q8_0 | GgmlTy::Q6K => Some(MMQ2_QUANT_D4),
        _ => None,
    }
}

/// 本臂适用门(单一产地;`linear.rs` 与门/测试同读这一份):
/// ① 型已移植 ② `m ≥ 64`(小批仍给 MMVQ/复用臂)
/// ③ `n % 128 == 0`(上游 `I=128`,首版**不装 fallback**)
/// ④ `k % 128 == 0`(激活块 128 值)
pub const fn mmq2_ok(ty: GgmlTy, m: usize, n: usize, k: usize) -> bool {
    mmq2_entry_for(ty, m).is_some()
        && m >= 64
        && n % MMQ2_I == 0
        && k % MMQ2_QK == 0
}

/// 激活缓冲字节数(`[k/128][n_token]` × 144B)
pub const fn mmq2_xq_bytes(m: usize, k: usize) -> usize {
    (k / MMQ2_QK) * m * MMQ2_Q8_1_BLOCK_BYTES
}

/// **MMQ 发射几何**(上游 stream-k 支的"一 tile/block"形态):
/// `grid = (ceil(n/I)·ceil(m/J), 1, 1)`、`block = (32, THREADS/32)`、
/// `smem = J·4(ids) + X 面 + PAD(J·144, THREADS·4)`。
pub fn mmq2_geom(m: usize, n: usize, j: usize) -> ((u32, u32, u32), (u32, u32, u32), u32) {
    // ⚠️ 一维 grid:`ntiles = nty·ntx`(上游 stream-k 支在 tiles 效率 ≥90% 的形态 ——
    // 每 block 恰一个 tile、无 fixup)。核内用 blockIdx.x 走线性 tile 解码。
    let nty = n.div_ceil(MMQ2_I).max(1);
    let ntx = m.div_ceil(j).max(1);
    let gx = (nty * ntx) as u32;
    let gy = 1u32;
    let block = (32u32, (MMQ2_THREADS / 32) as u32, 1u32);
    ((gx, gy, 1), block, mmq2_smem(j) as u32)
}

/// smem 字节(与 `vendor2/mmq.cuh::mmq_get_nbytes_shared` 逐项同式;
/// 单测 `mmq2_smem_matches_formula` 钉住)
pub const fn mmq2_smem(j: usize) -> usize {
    let nbs_ids = j * 4;
    let nbs_y = j * MMQ2_Q8_1_BLOCK_BYTES;
    let pad_to = MMQ2_THREADS * 4;
    nbs_ids + MMQ2_SMEM_X + nbs_y.div_ceil(pad_to) * pad_to
}

/// J 档(分派用;与 `mmq2_entry_for` 同一判据)
pub const fn mmq2_j(m: usize) -> usize {
    if m >= MMQ2_I { 128 } else { 64 }
}

/// **激活量化的几何**:`grid = (n_tokens, ceil(k/(4·BLK)), 1)`、`block = (BLK,1,1)`
pub fn mmq2_quant_geom(m: usize, k: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    const BLK: usize = 128;
    let gy = k.div_ceil(4 * BLK).max(1) as u32;
    ((m.max(1) as u32, gy, 1), (BLK as u32, 1, 1))
}

// ---- **行窗解量化**(W-2d;dequant+cuBLAS 臂的喂料手)--------------------------
/// `owl_gguf_dequant_range_q8_0`
pub const RANGE_Q8_0: &str = "owl_gguf_dequant_range_q8_0";
/// 同上(Q4_K)
pub const RANGE_Q4K: &str = "owl_gguf_dequant_range_q4k";
/// 同上(Q5_K)
pub const RANGE_Q5K: &str = "owl_gguf_dequant_range_q5k";
/// 同上(Q6_K)
pub const RANGE_Q6K: &str = "owl_gguf_dequant_range_q6k";
/// 同上(IQ4_NL)
pub const RANGE_IQ4NL: &str = "owl_gguf_dequant_range_iq4nl";
/// 同上(IQ4_XS)
pub const RANGE_IQ4XS: &str = "owl_gguf_dequant_range_iq4xs";
/// 同上(IQ3_XXS)
pub const RANGE_IQ3XXS: &str = "owl_gguf_dequant_range_iq3xxs";
/// 同上(IQ3_S)
pub const RANGE_IQ3S: &str = "owl_gguf_dequant_range_iq3s";
/// 同上(Q2_K)
pub const RANGE_Q2K: &str = "owl_gguf_dequant_range_q2k";
/// 同上(Q3_K)
pub const RANGE_Q3K: &str = "owl_gguf_dequant_range_q3k";
/// 同上(IQ2_XXS)
pub const RANGE_Iq2Xxs: &str = "owl_gguf_dequant_range_iq2xxs";
/// 同上(IQ2_XS)
pub const RANGE_Iq2Xs: &str = "owl_gguf_dequant_range_iq2xs";
/// 同上(IQ2_S)
pub const RANGE_Iq2S: &str = "owl_gguf_dequant_range_iq2s";
/// 同上(IQ1_S)
pub const RANGE_Iq1S: &str = "owl_gguf_dequant_range_iq1s";


/// 型 → 行窗解量化入口(`None` = 未移植)
pub const fn range_entry_for(ty: GgmlTy) -> Option<&'static str> {
    match ty {
        GgmlTy::Q8_0 => Some(RANGE_Q8_0),
        GgmlTy::Q4K => Some(RANGE_Q4K),
        GgmlTy::Q5K => Some(RANGE_Q5K),
        GgmlTy::Q6K => Some(RANGE_Q6K),
        GgmlTy::Iq4Nl => Some(RANGE_IQ4NL),
        GgmlTy::Iq4Xs => Some(RANGE_IQ4XS),
        GgmlTy::Iq3Xxs => Some(RANGE_IQ3XXS),
        GgmlTy::Iq3S => Some(RANGE_IQ3S),
        GgmlTy::Q2K => Some(RANGE_Q2K),
        GgmlTy::Q3K => Some(RANGE_Q3K),
        GgmlTy::Iq2Xxs => Some(RANGE_Iq2Xxs),
        GgmlTy::Iq2Xs => Some(RANGE_Iq2Xs),
        GgmlTy::Iq2S => Some(RANGE_Iq2S),
        GgmlTy::Iq1S => Some(RANGE_Iq1S),
        _ => None,
    }
}

/// 行窗解量化几何(v2;`grid = ceil(rows·(k/GQ) / (NW·GPW))`、`block = (32, NW)`)。
///
/// **v1 → v2**(W-2e):v1 是 `grid = (rows, k/qk)`、`block = (32,1,1)`(一 CTA 一块,
/// 窄写),实测 Q8_0 388 / Q4_K 273 GB/s ⇒ 改成"组 = 256 元素 + 8 warp/CTA +
/// 16 B/lane 向量落盘"(详见 `.cu` 头注)。常量与 `cu/gguf/gguf_smem.cu`
/// `namespace owl_gguf_range` **逐项互锁**(单测 `range_geom_matches_cu`)。
pub const RANGE_NW: usize = 8;
/// 每 warp 组数
pub const RANGE_GPW: usize = 4;
/// 组宽(元素)= 512 B
pub const RANGE_GQ: usize = 256;

/// 本臂前置:`k % GQ == 0`(与 `.cu` 的 `gpr = k / GQ` 同一判据)。
pub const fn range_ok(ty: GgmlTy, k: usize) -> bool {
    range_entry_for(ty).is_some() && k % RANGE_GQ == 0
}

pub fn range_geom(ty: GgmlTy, rows: usize, k: usize) -> Option<((u32, u32, u32), (u32, u32, u32))> {
    let (be, _) = ty.block();
    if range_entry_for(ty).is_none() || be == 0 || k % be != 0 || k % RANGE_GQ != 0 {
        return None;
    }
    let total = (rows.max(1) * (k / RANGE_GQ)).max(1);
    let per_cta = RANGE_NW * RANGE_GPW;
    let gx = total.div_ceil(per_cta).max(1) as u32;
    Some(((gx, 1, 1), (32, RANGE_NW as u32, 1)))
}

// ---- **内维列序 gather**(W-2e;`ssm_out` 的 tiled → grouped)------------------
/// `owl_gather_cols_f16`(f16;`out[.., 32·i] = x[.., 32·tbl[i]]`)
pub const GATHER_COLS_F16: &str = "owl_gather_cols_f16";

/// gather 几何:`grid = (rows, 1, 1)`、`block = (256, 1, 1)`
pub fn gather_geom(rows: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    ((rows.max(1) as u32, 1, 1), (256, 1, 1))
}

// ---- **官方 MMQ**(W-2c;tensor-core prefill)--------------------------------
/// `owl_gguf_mmq_q8_0`(= llama.cpp `mul_mat_q<Q8_0>` 的 C 入口;见 `vendor/README.md`)
pub const MMQ_Q8_0: &str = "owl_gguf_mmq_q8_0";
/// 同上(Q4_K)
pub const MMQ_Q4K: &str = "owl_gguf_mmq_q4k";
/// 同上(Q5_K)
pub const MMQ_Q5K: &str = "owl_gguf_mmq_q5k";
/// 同上(Q6_K)
pub const MMQ_Q6K: &str = "owl_gguf_mmq_q6k";
/// MMQ 适配 TU 的资产 id(内核本体 = `vendor/mmq.cu`)
pub const MMQ_SRC: &str = "cu/gguf/gguf_mmq.cu";

/// MMQ 的 tile 档(**与 `vendor/common.cuh` 的 `OWL_MMQ_TENSOR_CORES` 互锁**;
/// 改一处必须改另一处 —— 网格与核内 tile 错位 = 静默错值,由数值门兜底)。
///
/// | 档 | Q8_0 | Q4_K/Q5_K | Q6_K | 备注 |
/// |---|---|---|---|---|
/// | **dp4a(本仓默认)** | {128,64,4} | {64,128,4} | {64,64,4} | 64-128 token/权重读 |
/// | 张量核 | {4,32,4} | {4,32,4} | {4,32,4} | 4 token/权重读 ⇒ 带宽墙 |
///
/// 实测(3090 Ti,27B,1001 tok):dp4a 档明显更快(见设计 §18.6)。
pub const MMQ_TENSOR_CORES: bool = false;

/// 型 → (mmq_x = 每 CTA token 数, mmq_y = 每 CTA 权重行数, nwarps)
pub const fn mmq_tile(ty: GgmlTy) -> (usize, usize, usize) {
    if MMQ_TENSOR_CORES {
        return (4, 32, 4);
    }
    match ty {
        GgmlTy::Q8_0 => (128, 64, 4),
        GgmlTy::Q4K | GgmlTy::Q5K => (64, 128, 4),
        GgmlTy::Q6K => (64, 64, 4),
        _ => (64, 64, 4),
    }
}

/// 型 → MMQ 入口名(`None` = 未移植/未 vendoring)
pub const fn mmq_entry_for(ty: GgmlTy) -> Option<&'static str> {
    match ty {
        GgmlTy::Q8_0 => Some(MMQ_Q8_0),
        GgmlTy::Q4K => Some(MMQ_Q4K),
        GgmlTy::Q5K => Some(MMQ_Q5K),
        GgmlTy::Q6K => Some(MMQ_Q6K),
        _ => None,
    }
}

/// **MMQ 发射几何**(唯一产地;与 `vendor/mmq.cu` 的 `arch_config` 同一契约):
/// `grid = (ceil(n / MMQ_Y), ceil(m / MMQ_X))`,`block = (32, MMQ_NW)`,`smem = 0`
/// (tile 全是 `__shared__` 静态数组)。
pub fn mmq_geom(ty: GgmlTy, m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let (x, y, nw) = mmq_tile(ty);
    let gx = n.div_ceil(y).max(1) as u32; // 权重行方向(grid.x)
    let gy = m.div_ceil(x).max(1) as u32; // token 方向(grid.y,按 x 分窗)
    ((gx, gy, 1), (32, nw as u32, 1))
}

// ---- **smem 复用臂**(W-2b;prefill 的算术杠杆)------------------------------
/// `owl_gguf_gemm_smem_q8_0`
pub const GEMM_SMEM_Q8_0: &str = "owl_gguf_gemm_smem_q8_0";
/// 同上(Q4_K)
pub const GEMM_SMEM_Q4K: &str = "owl_gguf_gemm_smem_q4k";
/// 同上(Q5_K)
pub const GEMM_SMEM_Q5K: &str = "owl_gguf_gemm_smem_q5k";
/// 同上(Q6_K)
pub const GEMM_SMEM_Q6K: &str = "owl_gguf_gemm_smem_q6k";
/// smem 臂的资产 id
pub const SMEM_SRC: &str = "cu/gguf/gguf_smem.cu";

/// 型 → **smem 复用臂**入口名(`None` = 未移植)
pub const fn smem_entry_for(ty: GgmlTy) -> Option<&'static str> {
    match ty {
        GgmlTy::Q8_0 => Some(GEMM_SMEM_Q8_0),
        GgmlTy::Q4K => Some(GEMM_SMEM_Q4K),
        GgmlTy::Q5K => Some(GEMM_SMEM_Q5K),
        GgmlTy::Q6K => Some(GEMM_SMEM_Q6K),
        _ => None,
    }
}

/// smem 臂的几何常量(**与 `cu/gguf/gguf_smem.cu` 的 `namespace` 互锁**;
/// 单测 `smem_geom_matches_cu` 逐项比对)。
pub const SMEM_MT: usize = 64;
/// 每 CTA 权重行窗
pub const SMEM_NT: usize = 64;
/// 每次 k 块(= 一个 K-quant 超块)
pub const SMEM_KC: usize = 256;
/// smem 行 padding(halfs;`STRIDE = KC + PAD`)
pub const SMEM_PAD: usize = 2;
/// 每 CTA warp 数(512 线程:1 CTA/SM 也有 16 warp 在飞)
pub const SMEM_NW: usize = 16;

/// smem 字节数(`(MT + NT) × (KC + PAD) × 2B` = 66048 > 48KB ⇒ 走 opt-in 通道)
pub const fn smem_bytes() -> usize {
    (SMEM_MT + SMEM_NT) * (SMEM_KC + SMEM_PAD) * 2
}

/// **smem 臂发射几何**:`grid = (ceil(n/NT), ceil(m/MT))`,`block = (32, NW)`,
/// `smem = smem_bytes()`(核内 `extern __shared__`)。
pub fn smem_geom(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let gx = (n.div_ceil(SMEM_NT)).max(1) as u32;
    let gy = (m.div_ceil(SMEM_MT)).max(1) as u32;
    ((gx, gy, 1), (32, SMEM_NW as u32, 1))
}

/// **已移植的型**(加型 = `.cu` 加一行实例 + 本表加一项 + `ops/gguf.rs` 加一条声明)
pub const PORTED: &[GgmlTy] = &[
    GgmlTy::Q8_0,
    GgmlTy::Q4K,
    GgmlTy::Q5K,
    GgmlTy::Q6K,
    // IQ 族(2026-10-20;REQ-GGUF-04):块解量化 + vec_dot 均在 `gguf.cuh`,
    // 本仓补的是入口/行窗/查表三条腿(见 ops/gguf.rs 的逐型声明)
    GgmlTy::Iq4Nl,
    GgmlTy::Iq4Xs,
    GgmlTy::Iq3Xxs,
    GgmlTy::Iq3S,
    GgmlTy::Q2K,
    GgmlTy::Q3K,
    GgmlTy::Iq2Xxs,
    GgmlTy::Iq2Xs,
    GgmlTy::Iq2S,
    GgmlTy::Iq1S,
];

/// **已移植「行解量化」的型**(查表臂 = `token_embd` 用)。
///
/// ⚠️ 与 [`PORTED`](GEMM 臂)**刻意分开**:两臂的能力集不是一回事 ——
/// Q8_0 有 GEMM(点积)入口,但本仓快照的 `gguf.cuh` **没有**
/// `dequantize_block_q8_0`(只有 `q2_K`..`q6_K` 五个块解量化)⇒ 量化表若是
/// Q8_0,**响亮拒**(设计纪律:解析得了 ≠ 跑得了)。
pub const ROWS_PORTED: &[GgmlTy] = &[
    GgmlTy::Q4K,
    GgmlTy::Q5K,
    GgmlTy::Q6K,
    // IQ 族:块解量化自带(32 线程),行解量化与 K-quant 同一条路
    GgmlTy::Iq4Nl,
    GgmlTy::Iq4Xs,
    GgmlTy::Iq3Xxs,
    GgmlTy::Iq3S,
    GgmlTy::Q2K,
    GgmlTy::Q3K,
    GgmlTy::Iq2Xxs,
    GgmlTy::Iq2Xs,
    GgmlTy::Iq2S,
    GgmlTy::Iq1S,
];

/// 型 → GEMM 入口名(**唯一映射**;`None` = 本仓未移植 ⇒ 调用方**响亮拒**)
pub const fn entry_for(ty: GgmlTy) -> Option<&'static str> {
    match ty {
        GgmlTy::Q8_0 => Some(GEMM_Q8_0),
        GgmlTy::Q4K => Some(GEMM_Q4K),
        GgmlTy::Q5K => Some(GEMM_Q5K),
        GgmlTy::Q6K => Some(GEMM_Q6K),
        GgmlTy::Iq4Nl => Some(GEMM_IQ4NL),
        GgmlTy::Iq4Xs => Some(GEMM_IQ4XS),
        GgmlTy::Iq3Xxs => Some(GEMM_IQ3XXS),
        GgmlTy::Iq3S => Some(GEMM_IQ3S),
        GgmlTy::Q2K => Some(GEMM_Q2K),
        GgmlTy::Q3K => Some(GEMM_Q3K),
        GgmlTy::Iq2Xxs => Some(GEMM_Iq2Xxs),
        GgmlTy::Iq2Xs => Some(GEMM_Iq2Xs),
        GgmlTy::Iq2S => Some(GEMM_Iq2S),
        GgmlTy::Iq1S => Some(GEMM_Iq1S),
        _ => None,
    }
}

/// 型 → **复用臂**入口名(`None` = 未移植 ⇒ 调用方回退 MMVQ,不静默)
pub const fn tile_entry_for(ty: GgmlTy) -> Option<&'static str> {
    match ty {
        GgmlTy::Q8_0 => Some(GEMM_TILE_Q8_0),
        GgmlTy::Q4K => Some(GEMM_TILE_Q4K),
        GgmlTy::Q5K => Some(GEMM_TILE_Q5K),
        GgmlTy::Q6K => Some(GEMM_TILE_Q6K),
        _ => None,
    }
}

/// **复用臂发射几何**(唯一产地;`cu/gguf/gguf_tile.cu` 头注 = 同一契约):
/// `grid = (ceil(n / n_warps), ceil(m / TILE_MT))`,`block = (32, n_warps, 1)`
pub const fn gemm_tile_geom(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let mut gx = (n as u32).div_ceil(N_WARPS);
    if gx == 0 {
        gx = 1;
    }
    let mut gy = (m as u32).div_ceil(TILE_MT as u32);
    if gy == 0 {
        gy = 1;
    }
    ((gx, gy, 1), (32, N_WARPS, 1))
}

/// MMVQ 发射几何(唯一产地;`cu/gguf/gguf_gemm.cu` 头注 = 同一契约):/// `grid = (ceil(n / n_warps), m)`,`block = (32, n_warps, 1)`。
///
/// `n_warps` 选 4:每 CTA 4 行输出(lane 覆盖 k_block 的步进 = `vdr*32/qi`);
/// 行数多时 CTA 数足够,行数少(小 n)也不至于一个 CTA 干太多行。
pub const N_WARPS: u32 = 4;

/// 几何(picker 用;纯函数,零设备)
pub const fn gemm_geom(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    // const 上下文里 `Ord::max` 不稳(2026-10-19 实测)⇒ 手写下界
    let mut gx = (n as u32).div_ceil(N_WARPS);
    if gx == 0 {
        gx = 1;
    }
    let mut gy = m as u32;
    if gy == 0 {
        gy = 1;
    }
    ((gx, gy, 1), (32, N_WARPS, 1))
}

/// 行解量化入口名(**唯一映射**;`None` = 本仓未移植 ⇒ 调用方**响亮拒**)
pub const fn rows_entry_for(ty: GgmlTy) -> Option<&'static str> {
    match ty {
        GgmlTy::Q4K => Some(ROWS_Q4K),
        GgmlTy::Q5K => Some(ROWS_Q5K),
        GgmlTy::Q6K => Some(ROWS_Q6K),
        GgmlTy::Iq4Nl => Some(ROWS_IQ4NL),
        GgmlTy::Iq4Xs => Some(ROWS_IQ4XS),
        GgmlTy::Iq3Xxs => Some(ROWS_IQ3XXS),
        GgmlTy::Iq3S => Some(ROWS_IQ3S),
        GgmlTy::Q2K => Some(ROWS_Q2K),
        GgmlTy::Q3K => Some(ROWS_Q3K),
        GgmlTy::Iq2Xxs => Some(ROWS_Iq2Xxs),
        GgmlTy::Iq2Xs => Some(ROWS_Iq2Xs),
        GgmlTy::Iq2S => Some(ROWS_Iq2S),
        GgmlTy::Iq1S => Some(ROWS_Iq1S),
        _ => None,
    }
}

/// 行解量化的 **CTA 线程数**(逐型精确值 —— **不是**上界)。
///
/// 为什么精确:`gguf.cuh` 的块解量化是**协作式**(`threadIdx.x` 直接参与分工:
/// Q4_K/Q5_K 用 `tid/8`、`tid%8` 覆盖 `64*il + 4*ir` 域 ⇒ 必须 32;
/// Q6_K 用 `ip = tid/32`(0/1)⇒ 必须 64)。多一线程 = 越界写。
pub const fn rows_threads(ty: GgmlTy) -> usize {
    match ty {
        GgmlTy::Q6K => 64,
        // Q2_K/Q3_K 的块解量化按 `tid/32`、`tid/4` 分工 ⇒ 需 64 线程
        GgmlTy::Q2K | GgmlTy::Q3K => 64,
        _ => 32,
    }
}

/// 行解量化几何(唯一产地;`cu/gguf/gguf_rows.cu` 头注 = 同一契约):
/// `grid = (tokens, k / qk)`,`block = (rows_threads(ty), 1, 1)`
///
/// 返 `None` = 形态/型不成立(未移植型 / `k` 不是超块整数倍)—— **响亮拒**
/// 的真相在调用方:本函数不 panic(声明面零失败路径),调用方按 `None` 报错。
pub const fn rows_geom(ty: GgmlTy, tokens: usize, k: usize) -> Option<((u32, u32, u32), (u32, u32, u32))> {
    if rows_entry_for(ty).is_none() {
        return None;
    }
    let (be, _) = ty.block();
    if be == 0 || k % be != 0 {
        return None;
    }
    let mut gx = tokens as u32;
    if gx == 0 {
        gx = 1;
    }
    let gy = (k / be) as u32;
    Some(((gx, gy, 1), (rows_threads(ty) as u32, 1, 1)))
}

/// **q8_1 块字节数 = 36**(CUDA/MMVQ 布局):`half2 ds`(`d`/`s` 各一个 half)+ `qs[32]`。
///
/// 两个**同名不同物**必须分清(2026-10-19 位级门实测抓出的坑):
///
/// | 布局 | 定义处 | 每块字节 | 用途 |
/// |---|---|---|---|
/// | **本仓激活缓冲** | `cu/gguf/gguf.cuh`(`ggml-common.h` 同源) | **36** | quant_act 写 / gemm 读 |
/// | GGUF *张量型* `Q8_1` | `gguf-py::GGML_QUANT_SIZES` | 40 | 文件里的同名张量型(本仓未用) |
///
/// `.cuh` 自带静态断言 `sizeof(block_q8_1) == 2*sizeof(ggml_fp16_t) + QK8_0`
/// (= `2*2 + 32`);本常量与它由单测 `q8_1_block_bytes_matches_cu_struct` 互锁。
/// **声明成 40 = 多占 11% 且留下未写的脏尾**(任何按声明形状读尾部的消费者会拿到垃圾)。
pub const Q8_1_BLOCK_BYTES: usize = 2 * 2 + QK_32;

/// 行解量化的**超块元素数**(核内 `nb = k / qk`;与 `rows_geom` 同一事实)
pub const fn rows_qk(ty: GgmlTy) -> Option<usize> {
    match ty {
        GgmlTy::Q4K | GgmlTy::Q5K | GgmlTy::Q6K => Some(QK_K),
        _ => None,
    }
}

/// 激活量化几何:`grid = (ceil(k_padded / 32), rows)`,`block = (32, 1, 1)`
pub const fn quant_act_geom(rows: usize, k_padded: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let mut gx = (k_padded as u32).div_ceil(32);
    if gx == 0 {
        gx = 1;
    }
    let mut gy = rows as u32;
    if gy == 0 {
        gy = 1;
    }
    ((gx, gy, 1), (32, 1, 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 编码 ↔ 词汇双射(表里不许有重码)
    #[test]
    fn codes_are_bijective() {
        for (i, t) in ALL.iter().enumerate() {
            assert_eq!(GgmlTy::from_code(t.code()), Some(*t), "{:?} 反查失败", t);
            assert_eq!(ALL.iter().filter(|x| x.code() == t.code()).count(), 1, "重码 {:?}", t);
            assert!(ALL[..i].iter().all(|x| x.code() != t.code()), "编码乱序/重复 {:?}", t);
        }
        assert_eq!(GgmlTy::from_code(4), None, "已移除的 Q4_2 必须查不到");
        assert_eq!(GgmlTy::from_code(31), None, "已移除的 Q4_0_4_4 必须查不到");
        assert_eq!(GgmlTy::from_code(999), None);
    }

    /// **可跑性映射的唯一性**:`entry_for` 与 `PORTED` 必须互证
    ///(逐型有名字 ∧ 名字不重复;未移植型 = `None` ⇒ 调用方响亮拒)
    #[test]
    fn runnable_mapping_is_bijective() {
        for t in ALL {
            match self::entry_for(*t) {
                Some(n) => {
                    assert!(PORTED.contains(t), "{:?} 有入口名却不在 PORTED", t);
                    assert!(
                        PORTED.iter().filter(|x| self::entry_for(**x) == Some(n)).count() == 1,
                        "入口名 `{n}` 被多型共用"
                    );
                }
                None => assert!(!PORTED.contains(t), "{:?} 在 PORTED 却无入口名", t),
            }
        }
        assert_eq!(PORTED.len(), PORTED.iter().filter(|t| entry_for(**t).is_some()).count());
    }

    /// **块几何表 = gguf-py `GGML_QUANT_SIZES`**(逐条硬编码;改表必须改这里)。
    ///
    /// 抄录自 `llama.cpp/gguf-py/gguf/constants.py`(2026-10-19 本机树)。
    #[test]
    fn block_table_matches_gguf_py() {
        let want: &[(GgmlTy, usize, usize)] = &[
            (GgmlTy::F32, 1, 4),
            (GgmlTy::F16, 1, 2),
            (GgmlTy::Q4_0, 32, 18),
            (GgmlTy::Q4_1, 32, 20),
            (GgmlTy::Q5_0, 32, 22),
            (GgmlTy::Q5_1, 32, 24),
            (GgmlTy::Q8_0, 32, 34),
            (GgmlTy::Q8_1, 32, 40),
            (GgmlTy::Q2K, 256, 84),
            (GgmlTy::Q3K, 256, 110),
            (GgmlTy::Q4K, 256, 144),
            (GgmlTy::Q5K, 256, 176),
            (GgmlTy::Q6K, 256, 210),
            (GgmlTy::Q8K, 256, 292),
            (GgmlTy::Iq2Xxs, 256, 66),
            (GgmlTy::Iq2Xs, 256, 74),
            (GgmlTy::Iq3Xxs, 256, 98),
            (GgmlTy::Iq1S, 256, 50),
            (GgmlTy::Iq4Nl, 32, 18),
            (GgmlTy::Iq3S, 256, 110),
            (GgmlTy::Iq2S, 256, 82),
            (GgmlTy::Iq4Xs, 256, 136),
            (GgmlTy::I8, 1, 1),
            (GgmlTy::I16, 1, 2),
            (GgmlTy::I32, 1, 4),
            (GgmlTy::I64, 1, 8),
            (GgmlTy::F64, 1, 8),
            (GgmlTy::Iq1M, 256, 56),
            (GgmlTy::Bf16, 1, 2),
            (GgmlTy::Tq1_0, 256, 54),
            (GgmlTy::Tq2_0, 256, 66),
            (GgmlTy::Mxfp4, 32, 17),
            (GgmlTy::Nvfp4, 64, 36),
            (GgmlTy::Q1_0, 128, 18),
            (GgmlTy::Q2_0, 64, 18),
        ];
        for (t, be, bb) in want {
            assert_eq!(t.block(), (*be, *bb), "{} 块几何不符", t.name());
        }
        assert_eq!(want.len(), ALL.len(), "全表条数不等(有型未登记)");
    }

    /// 实存两靶位的型都在可跑集;F32/Q6_K 等原生与 K-quant 各归其位
    #[test]
    fn targets_are_covered() {        for (n, t) in [
            ("Q4_K", GgmlTy::Q4K),
            ("Q8_0", GgmlTy::Q8_0),
            ("Q6_K", GgmlTy::Q6K),
            ("Q5_K", GgmlTy::Q5K),
        ] {
            assert!(entry_for(t).is_some(), "{n} 应在已移植集(PORTED)");
            assert!(range_entry_for(t).is_some(), "{n} 应有行窗解量化入口");
        }
        // IQ 族(2026-10-20;REQ-GGUF-04):三条腿全在(GEMM / 行解量化 / 行窗)
        for (n, t) in [
            ("IQ4_XS", GgmlTy::Iq4Xs),
            ("IQ4_NL", GgmlTy::Iq4Nl),
            ("IQ3_XXS", GgmlTy::Iq3Xxs),
            ("IQ3_S", GgmlTy::Iq3S),
        ] {
            assert!(entry_for(t).is_some(), "{n} 应在已移植集(PORTED)");
            assert!(rows_entry_for(t).is_some(), "{n} 应有行解量化入口");
            assert!(range_entry_for(t).is_some(), "{n} 应有行窗解量化入口");
        }
        assert!(GgmlTy::F32.native_dtype().is_some());
        assert!(GgmlTy::Q4K.native_dtype().is_none(), "量化型不得走元素路径");
        // 未移植者必须**无入口** ⇒ 可跑性响亮拒(设计 R3:IQ1/IQ2 与长尾型)
        for t in [GgmlTy::Iq1S, GgmlTy::Iq2Xxs, GgmlTy::Tq1_0, GgmlTy::Mxfp4, GgmlTy::Q2K] {
            assert!(entry_for(t).is_none(), "{:?} 不该有入口(未移植)", t);
        }
    }

    /// **行解量化臂**(查表)的能力集:逐型有名字、名字不重、几何对齐
    ///
    /// 两条判据(都是“错就静默错值/越界写”的雷):
    /// ① `rows_threads` 必须与块解量化的协作分工**精确对齐**;
    /// ② `rows_geom` 对未移植型 / `k` 非超块整数倍一律 `None`。
    #[test]
    fn rows_arm_mapping_and_geometry() {
        for t in ALL {
            match rows_entry_for(*t) {
                Some(n) => {
                    assert!(ROWS_PORTED.contains(t), "{:?} 有行入口却不在 ROWS_PORTED", t);
                    assert!(
                        ROWS_PORTED.iter().filter(|x| rows_entry_for(**x) == Some(n)).count() == 1,
                        "行入口名 `{n}` 被多型共用"
                    );
                }
                None => assert!(!ROWS_PORTED.contains(t), "{:?} 在 ROWS_PORTED 却无入口名", t),
            }
        }
        // ① 线程数(协作式分工的硬约束)
        assert_eq!(rows_threads(GgmlTy::Q4K), 32, "Q4_K: tid/8 与 tid%8 覆盖 256 元素需 32 线程");
        assert_eq!(rows_threads(GgmlTy::Q5K), 32);
        assert_eq!(rows_threads(GgmlTy::Q6K), 64, "Q6_K: ip = tid/32 需 64 线程");
        // ② 几何(4B token_embd = Q6_K: k=2560 ⇒ 10 块)
        let (g, b) = rows_geom(GgmlTy::Q6K, 5, 2560).expect("Q6_K 行几何");
        assert_eq!((g, b), ((5, 10, 1), (64, 1, 1)));
        assert!(rows_geom(GgmlTy::Q8_0, 5, 2560).is_none(), "Q8_0 无块解量化函数 ⇒ 行臂拒");
        assert!(rows_geom(GgmlTy::Q6K, 5, 2500).is_none(), "k 非超块整数倍 ⇒ 拒");
        assert_eq!(rows_qk(GgmlTy::Q4K), Some(QK_K));
        assert_eq!(rows_qk(GgmlTy::Q8_0), None);
    }

    /// **q8_1 激活块字节数**与 `.cuh` 的静态断言互锁(同名不同物的粒度钉子)
    #[test]
    fn q8_1_block_bytes_matches_cu_struct() {
        // 推导:`half2 ds`(d,s 各 half = 2+2)+ `qs[QK8_1]`(32)
        assert_eq!(Q8_1_BLOCK_BYTES, 2 * 2 + QK_32, "q8_1 激活块 = half2 + qs[32]");
        let t = crate::cu_sig::cu_text("cu/gguf/gguf.cuh");
        assert!(
            t.contains("sizeof(block_q8_1) == 2*sizeof(ggml_fp16_t) + QK8_0"),
            "`gguf.cuh` 的 block_q8_1 静态断言行变了 —— 同步 Q8_1_BLOCK_BYTES"
        );
        // 同名不同物:GGUF **张量型** Q8_1 仍是 40 B/块(gguf-py 表;本仓激活缓冲不用它)
        assert_eq!(GgmlTy::Q8_1.block(), (QK_32, 40), "文件格式的 Q8_1 是另一个布局");
    }

    /// 跨架构关键常量(QK_K 固定 256;块几何自洽:`block.1 * 8 >= block.0`(≥1bit/元素))
    #[test]
    fn geometry_is_sane() {
        assert_eq!(QK_K, 256);
        for t in ALL {
            let (be, bb) = t.block();
            assert!(be > 0 && bb > 0, "{:?} 零几何", t);
            assert!(bb * 8 >= be, "{:?} 每元素 <1 bit({be} 元素 / {bb} 字节)", t);
            // 字节数必须能被元素数整除后的位宽整除域成立(fp16 尺度对齐)
            assert!(bb >= be / 8, "{:?} 块字节数低于下界", t);
        }
    }
}
