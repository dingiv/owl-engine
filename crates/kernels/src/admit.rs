//! **算子族 × 代际的准入表** —— 「这种算子在这张卡上不应该跑」的**唯一出口**。
//!
//! 分工(2026-10-27 用户裁决):
//! - **权重格式决定算子族**(W4A16/W4A8 → Marlin;GGUF 块型 → MMQ;NVFP4 → CUTLASS 块缩放);
//! - **代际只做门**:本表回答"这族能不能在这代上跑",跑不了就给**原因 + 替代方案**。
//!
//! 为什么要有这张表(而不是散在各处的 `if arch == ...`):
//! 1. 选核是**三键**(arch × 权重格式 × m),散着写必然漏一处;表是单源,新族/新代际只改一处;
//! 2. 拒绝必须**响亮且可操作**(charter A9):返回值带原因串,直接进启动错误/预检,
//!    而不是运行期某次发射报 `no kernel image is available`;
//! 3. **实测结论**可以钉在这里当硬门 —— 例如 GDN chunked 在 sm_120 上 T>1 的数值 bug,
//!    在修好之前**根本不该被选中**,而不是等对拍发现。
//!
//! 纪律:
//! - 本模块**零依赖**(只 `core`)——`build.rs` 也能引(与 [`crate::arch`] 同风格);
//! - 判据一律走 [`crate::Caps`](ISA 事实),不写"我印象里 30 系可以"这种记忆;
//! - `Deny` 的串是**给人看的**(会进拒启信息),要写清"为什么 + 怎么办"。

use crate::arch::Arch;

/// 算子族 —— **按权重格式划分**,不是按显卡划分。
///
/// 同一族在不同代际上可能是"能用 / 要用 `a` 目标 / 根本不能用"([`Admit`])。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OpFamily {
    /// Marlin W4A16(权重 U4/U4B8 打包,激活 f16)
    MarlinW4A16,
    /// Marlin W4A8(权重 a8 布局,激活 per-token int8)
    MarlinW4A8,
    /// GGUF 块型(权重块字节原样驻留,MMQ2/MMVQ,激活 q8_1)
    GgufMmq,
    /// NVFP4 块缩放(e2m1 + 每 16 元素 e4m3),CUTLASS SM120 路径
    NvFp4Cutlass,
    /// FlashInfer paged prefill
    FiPrefill,
    /// XQA decode/verify(kNHD fp8 池)
    Xqa,
    /// GDN 线性注意力 chunked 臂(Triton AOT 六核)
    GdnChunked,
    /// GDN 线性注意力标量臂(数学参考实现)
    GdnScalar,
    /// cuBLAS f16 基线
    CublasF16,
}

impl OpFamily {
    /// 稳定短名(日志/错误串用)
    pub fn name(self) -> &'static str {
        match self {
            Self::MarlinW4A16 => "marlin-w4a16",
            Self::MarlinW4A8 => "marlin-w4a8",
            Self::GgufMmq => "gguf-mmq",
            Self::NvFp4Cutlass => "nvfp4-cutlass",
            Self::FiPrefill => "flashinfer-prefill",
            Self::Xqa => "xqa",
            Self::GdnChunked => "gdn-chunked",
            Self::GdnScalar => "gdn-scalar",
            Self::CublasF16 => "cublas-f16",
        }
    }

    /// 该族在 `ctx` 下的裁决。
    ///
    /// `NeedsArchSpecific` = 可跑,但**必须用 `a` 后缀编译目标**(`sm_120a` 等)出的 cubin;
    /// 构建面由 [`Arch::as_sm_a`] 负责,装配面由「设备 sm ∈ 构建期代际集」那道门负责。
    pub fn admit(self, ctx: AdmitCtx) -> Admit {
        let arch = ctx.arch;
        let caps = arch.caps();
        match self {
            // Marlin 设备码用 Ampere 原语(cp.async + ldmatrix + mma.sync)
            Self::MarlinW4A16 | Self::MarlinW4A8 => {
                if !caps.cp_async {
                    return Admit::Deny(
                        "Marlin 的设备码依赖 cp.async/ldmatrix(sm_80+);\
Turing(sm_75) 上跑不了,请换 f16 基线或换卡",
                    );
                }
                if matches!(self, Self::MarlinW4A8) && !caps.int8_mma {
                    return Admit::Deny(
                        "W4A8 需要 int8 张量核(sm_75+ 才有)",
                    );
                }
                Admit::Ok
            }
            // GGUF MMQ 用 int8 张量核(m16n8k32 s8);sm_75 起有
            Self::GgufMmq => {
                if caps.int8_mma {
                    Admit::Ok
                } else {
                    Admit::Deny(
                        "GGUF MMQ 需要 int8 张量核(sm_75+)",
                    )
                }
            }
            // NVFP4 是 arch-specific('a')指令集:除 FP4 张量核外还要求 compute_120a 目标
            Self::NvFp4Cutlass => {
                if caps.fp4_blockscale {
                    Admit::NeedsArchSpecific
                } else {
                    Admit::Deny(
                        "NVFP4 需要 FP4 张量核(`mxf4nvf4.block_scale`,sm_120a+);本代际没有 —— 该权重请走 W4A16/W4A8(Marlin)或 GGUF 块型",
                    )
                }
            }
            Self::FiPrefill | Self::Xqa => {
                if caps.cp_async {
                    Admit::Ok
                } else {
                    Admit::Deny(
                        "该注意力核依赖 cp.async(sm_80+)",
                    )
                }
            }
            // **实测硬门(2026-10-28 证伪并调转)**:sm_120 上跑坏的是**标量臂**,不是 chunked。
            // 证据(同 prompt 贪婪,`B/tok` = 字节/生成 token,>3 = 正常文本):
            //   spec off + scalar  → 正常(2.96~4.58 B/tok)
            //   spec d1/2/3/7 + scalar → **退化**("Thinking Process:\n" 后一路 "!",1.05~1.3 B/tok)
            //   spec d3 + chunked  → 正常(3.6~3.9 B/tok;d3 = 173 tok/s,接受 3.16/轮)
            // 前一轮把"字符少"当成"被截断"读反了 —— 415 字符/400 tok 才是退化那格。
            // 标量臂只在 verify(T>1) 形状坏:T=1 decode 与 prefill 都正常。
            Self::GdnChunked => {
                if !caps.cp_async {
                    return Admit::Deny(
                        "GDN chunked 依赖 cp.async(sm_80+)",
                    );
                }
                Admit::Ok
            }
            Self::GdnScalar => {
                // 标量臂(lmdeploy pre_sm90 chunked GDR 移植)在 Blackwell 上
                // 跑 spec verify 的 T=depth+1 形状时数值错 ⇒ 只允许 spec 关的档。
                if arch.cc().0 == 12 && ctx.spec_depth > 0 {
                    return Admit::Deny(
                        "GDN 标量臂在 sm_12x 上跑 spec verify(T>1)数值不对(2026-10-28 实测:spec 开即退化,1.05~1.3 B/tok;spec 关正常)。 二选一:关投机(spec.depth=0),或 `kernel.gdn_chunked = true, kernel.gdn_scalar = false`",
                    );
                }
                Admit::Ok
            }
            Self::CublasF16 => Admit::Ok,
        }
    }
}

/// 裁决上下文 —— 同一族在不同**用法**下合法性不同时用它表达
/// (典型:GDN chunked 只在 spec verify 的 T>1 形状上出错,单看代际无法判)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmitCtx {
    /// 设备代际(运行期探测值)
    pub arch: Arch,
    /// 投机深度(0 = 关闭;>0 表示会走 verify 的 T=depth+1 形状)
    pub spec_depth: usize,
}

impl AdmitCtx {
    pub fn of(arch: Arch) -> Self {
        Self { arch, spec_depth: 0 }
    }
    pub fn with_spec(mut self, depth: usize) -> Self {
        self.spec_depth = depth;
        self
    }
}

/// 准入裁决
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admit {
    /// 本代际可跑
    Ok,
    /// 可跑,但必须由 **arch-specific(`a`)编译目标**出的设备码
    NeedsArchSpecific,
    /// 本代际**不允许跑**;串 = 原因 + 替代方案(进拒启信息)
    Deny(&'static str),
}

impl Admit {
    pub fn is_ok(self) -> bool {
        matches!(self, Self::Ok | Self::NeedsArchSpecific)
    }

    /// 一行摘要(日志/预检)
    pub fn describe(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::NeedsArchSpecific => "ok(需 a 目标)",
            Self::Deny(_) => "denied",
        }
    }
}
