//! DFlash2 草稿特有算子元数据(conv / selector)。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};
use owl_macros::kernel_tags;

/// `owl_dflash_conv_bf16` —— 分组动态深度 2-tap 卷积
/// (delta [T,2,2,G] side-major + base [2,2,H];位置掩码 t%block;group=16 烘焙)
pub struct DflashConvBf16;

#[kernel_tags("dom.dflash", "feat.dflash", "phase.propose", "role.elem")]
impl KernelSpec for DflashConvBf16 {
    const ENTRY: &'static str = "owl_dflash_conv_bf16";
    const SRC: &'static str = "cu/dflash/dflash_conv.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::t_in("delta", Dtype::BF16, 2),
        Param::t_in("base", Dtype::BF16, 3),
        Param::i32("side"),
        Param::i32("block_size"),
        Param::i32("hidden"),
        Param::i32("groups"),
        Param::i32("rows"),
        Param::t_out("out", Dtype::BF16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_dflash_select_bf16` —— selector 格打分 + 贪心 walk 融合
/// (单缓冲 [S·(K·K+1)]:toks + scores;码本原生 BF16)
pub struct DflashSelectBf16;

#[kernel_tags("dom.dflash", "feat.dflash", "phase.propose", "role.sample")]
impl KernelSpec for DflashSelectBf16 {
    const ENTRY: &'static str = "owl_dflash_select_bf16";
    const SRC: &'static str = "cu/dflash/dflash_select.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("cand", Dtype::F32, 2),
        Param::t_in("unary", Dtype::F32, 2),
        Param::t_in("proj", Dtype::BF16, 2),
        Param::t_in("anchor", Dtype::F32, 0),
        Param::t_in("a_tab", Dtype::BF16, 2),
        Param::t_in("b_tab", Dtype::BF16, 2),
        Param::i32("slots"),
        Param::i32("k_top"),
        Param::i32("rank"),
        Param::t_out("out", Dtype::F32, 1),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static DFLASH_CONV_BF16: KernelMeta = meta_of::<DflashConvBf16>();
pub static DFLASH_SELECT_BF16: KernelMeta = meta_of::<DflashSelectBf16>();

// ============================================================================
// `dflash.rs` 域 R1 补表
// (R1 补表 2026-10-13:声明从登记表派生,签名由 `op_tables` 门逐参互证)
// ============================================================================

/// `owl_dflash_conv_f16` —— dflash_conv.cu(R1 补表)
pub struct DflashConvF16;

#[kernel_tags("dom.dflash", "feat.dflash", "phase.propose", "role.elem")]
impl KernelSpec for DflashConvF16 {
    const ENTRY: &'static str = "owl_dflash_conv_f16";
    const SRC: &'static str = "cu/dflash/dflash_conv.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::F16, 2),
        Param::t_in("delta", Dtype::F16, 2),
        // base = 核表 `base_kernel [2(side), TAPS, hidden]`(装载器同形)⇒ r3;
        // 旧声明 r2 与真算子/bf16 臂都不符(2026-10-15 B6 用例暴露)
        Param::t_in("base", Dtype::F16, 3),
        Param::i32("side"),
        Param::i32("block_size"),
        Param::i32("hidden"),
        Param::i32("groups"),
        Param::i32("rows"),
        Param::t_out("out", Dtype::F16, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_dflash_select_f16` —— dflash_select.cu(R1 补表)
pub struct DflashSelectF16;

#[kernel_tags("dom.dflash", "feat.dflash", "phase.propose", "role.sample")]
impl KernelSpec for DflashSelectF16 {
    const ENTRY: &'static str = "owl_dflash_select_f16";
    const SRC: &'static str = "cu/dflash/dflash_select.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("cand", Dtype::F32, 2),
        Param::t_in("unary", Dtype::F32, 2),
        Param::t_in("proj", Dtype::F16, 2),
        Param::t_in("anchor", Dtype::F32, 0),  // 调用方传 [1] 标量;r2 旧声明与真调用不符(2026-10-15 B6)
        Param::t_in("a_tab", Dtype::F16, 2),
        Param::t_in("b_tab", Dtype::F16, 2),
        Param::i32("slots"),
        Param::i32("k_top"),
        Param::i32("rank"),
        Param::t_out("out", Dtype::F32, 1),  // 生产/测试皆单缓冲 [2·rows·K] 扁平(r1)
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static OWL_DFLASH_CONV_F16: KernelMeta = meta_of::<DflashConvF16>();
pub static OWL_DFLASH_SELECT_F16: KernelMeta = meta_of::<DflashSelectF16>();
