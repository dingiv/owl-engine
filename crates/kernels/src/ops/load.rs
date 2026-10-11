//! 装载域生产算子元数据(ct packed → marlin B 设备重排)。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};
use crate::sources;

/// `owl_ct_repack_u32` —— AWQ 装载线的设备侧权重重排(**非轮内**;装载期一次)
pub struct CtRepackU32;

impl KernelSpec for CtRepackU32 {
    const ENTRY: &'static str = "owl_ct_repack_u32";
    const SOURCE: &'static str = sources::owl::CT_REPACK_U32;
    const PARAMS: &'static [Param] = &[
        Param::t_in("in", Dtype::U32, 2),
        Param::sz("rows"),
        Param::sz("cols"),
        Param::t_out("out", Dtype::U32, 2),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static CT_REPACK_U32: KernelMeta = meta_of::<CtRepackU32>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[&CT_REPACK_U32];
