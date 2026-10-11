//! DFlash2 草稿特有算子元数据(conv / selector)。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};
use crate::sources;

/// `owl_dflash_conv_bf16` —— 分组动态深度 2-tap 卷积
/// (delta [T,2,2,G] side-major + base [2,2,H];位置掩码 t%block;group=16 烘焙)
pub struct DflashConvBf16;

impl KernelSpec for DflashConvBf16 {
    const ENTRY: &'static str = "owl_dflash_conv_bf16";
    const SOURCE: &'static str = sources::owl::DFLASH2_F16;
    const PARAMS: &'static [Param] = &[
        Param::t_in("x", Dtype::BF16, 2),
        Param::t_in("delta", Dtype::BF16, 2),
        Param::t_in("base", Dtype::BF16, 2),
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

impl KernelSpec for DflashSelectBf16 {
    const ENTRY: &'static str = "owl_dflash_select_bf16";
    const SOURCE: &'static str = sources::owl::DFLASH2_F16;
    const PARAMS: &'static [Param] = &[
        Param::t_in("cand", Dtype::F32, 2),
        Param::t_in("unary", Dtype::F32, 2),
        Param::t_in("proj", Dtype::BF16, 2),
        Param::t_in("anchor", Dtype::F32, 2),
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

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[&DFLASH_CONV_BF16, &DFLASH_SELECT_BF16];
