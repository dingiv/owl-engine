//! Marlin W4A16 元数据(foreign:预编 `.a` + host 发射;无 `.cu` 源)。
//!
//! 槽序历史上有三处:`client/marlin.rs::SIG_AWQ`(权威串)、models 的
//! `foreign_sig`、server 的 parse 断言。现在**唯一真源 = 本表**。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};

/// `marlin_gemm_w4a16_awq` —— AWQ(kU4 非对称,has_zp)臂:
/// `[a, b(qpack), out, scales, zeros, ws, c_tmp, m, k, n, groupsize]`。
///
/// `b`/`zeros`/`ws`/`c_tmp` 都是 32 位打包/工作区块(`Dtype::U32` 承载);
/// `a`/`out`/`scales` 是 f16(激活/输出/尺度)。bf16 臂(`..._bf16`)同槽序,
/// 仅激活与输出换 BF16 —— 另立 struct(B2 批次)。
pub struct MarlinGemmW4a16Awq;

impl KernelSpec for MarlinGemmW4a16Awq {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::marlin::GEMM_W4A16);
    const ENTRY: &'static str = crate::family::marlin::GEMM_W4A16_AWQ;
    /// foreign(静态链接):无 `.cu` 源
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F16, 2),
        // 4bit 打包权重(32 位字承载)
        Param::t_in("b", Dtype::U32, 2),
        Param::t_out("out", Dtype::F16, 2),
        Param::t_in("scales", Dtype::F16, 2),
        // AWQ 零偏(kU4 has_zp;对称臂无此槽 —— 故本表只对 AWQ 生效)
        Param::t_in("zeros", Dtype::U32, 2),
        Param::t_in("ws", Dtype::U32, 1),
        Param::t_in("c_tmp", Dtype::U32, 1),
        Param::sz("m"),
        Param::sz("k"),
        Param::sz("n"),
        Param::sz("groupsize"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
    // 装配期 oracle 住 `cfg(feature = "marlin")` 模块(要用自家 packer 造输入)
    // —— 表(零门)与 oracle(门内)的接触点只有这一对互斥常量
    #[cfg(feature = "marlin")]
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> =
        Some(&crate::ops::oracles::marlin::MarlinAwqOracle);
    #[cfg(not(feature = "marlin"))]
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> = None;
}

/// `marlin_gemm_w4a16_bf16` —— BF16 激活/输出臂(**无 zeros 槽**:6 块 + 4 sz)。
/// DFlash2 草稿路径(权重原生 BF16)。
pub struct MarlinGemmW4a16Bf16;

impl KernelSpec for MarlinGemmW4a16Bf16 {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::marlin::GEMM_W4A16);
    const ENTRY: &'static str = crate::family::marlin::GEMM_W4A16_BF16;
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::BF16, 2),
        Param::t_in("b", Dtype::U32, 2),
        Param::t_out("out", Dtype::BF16, 2),
        Param::t_in("scales", Dtype::F16, 2),
        Param::t_in("ws", Dtype::U32, 1),
        Param::t_in("c_tmp", Dtype::U32, 1),
        Param::sz("m"),
        Param::sz("k"),
        Param::sz("n"),
        Param::sz("groupsize"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `marlin_gemm_w4a16` —— f16 激活臂(对称 kU4;6 块 + 4 sz)
pub struct MarlinGemmW4a16F16;

impl KernelSpec for MarlinGemmW4a16F16 {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::marlin::GEMM_W4A16);
    const ENTRY: &'static str = crate::family::marlin::GEMM_W4A16;
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F16, 2),
        Param::t_in("b", Dtype::U32, 2),
        Param::t_out("out", Dtype::F16, 2),
        Param::t_in("scales", Dtype::F16, 2),
        Param::t_in("ws", Dtype::U32, 1),
        Param::t_in("c_tmp", Dtype::U32, 1),
        Param::sz("m"),
        Param::sz("k"),
        Param::sz("n"),
        Param::sz("groupsize"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static W4A16_AWQ: KernelMeta = meta_of::<MarlinGemmW4a16Awq>();
pub static W4A16_BF16: KernelMeta = meta_of::<MarlinGemmW4a16Bf16>();
pub static W4A16_F16: KernelMeta = meta_of::<MarlinGemmW4a16F16>();

/// 本域全部已建表算子(**marlin 家族三名全建** ⇒ `names()` 可由本表派生)
pub static METAS: &[&KernelMeta] = &[&W4A16_AWQ, &W4A16_BF16, &W4A16_F16];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::meta_of;

    /// 表接管后与旧权威串(`client/marlin.rs::SIG_AWQ`,逐字抄录为锚)
    /// 一致 —— 漂移即红
    #[test]
    fn awq_slot_order_matches_legacy_sig() {
        assert_eq!(meta_of::<MarlinGemmW4a16Awq>().args_str(), "T,T,O,T,T,T,T,sz,sz,sz,sz");
    }

    /// 输出槽在第 2 位(**不是末位**)—— `O` 的角色标注必须能表达这点
    #[test]
    fn out_slot_is_not_last() {
        let m = meta_of::<MarlinGemmW4a16Awq>();
        assert_eq!(m.out_pos(), Some(2));
        assert_eq!(m.params.len(), 11);
    }
}
