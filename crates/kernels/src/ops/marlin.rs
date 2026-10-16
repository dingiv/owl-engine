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

/// `marlin_gemm_w4a8_awq` —— **W4A8 × AWQ 非对称**臂:int8 激活(已量化) ×
/// kU4 零偏权重。槽序 = AWQ 臂的 `zeros` 之后插 `a_scales`(per-token fp32 `[M,1]`,
/// 激活量化核产出、全局因子已折入)。权重侧与 W4A16 同布局(零 repack)。
pub struct MarlinGemmW4a8Awq;

impl KernelSpec for MarlinGemmW4a8Awq {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::marlin::GEMM_W4A16);
    const ENTRY: &'static str = crate::family::marlin::GEMM_W4A8_AWQ;
    /// foreign(静态链接):无 `.cu` 源
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        // A = int8 已量化激活(1B/elem ⇒ 按本仓惯例以 U8 字节承载声明;
        // 语义在核侧,与 fp8 KV 池同款约定)
        Param::t_in("a", Dtype::U8, 2),
        Param::t_in("b", Dtype::U32, 2),
        Param::t_out("out", Dtype::F16, 2),
        Param::t_in("scales", Dtype::F16, 2),
        Param::t_in("zeros", Dtype::U32, 2),
        // per-token 激活尺度(fp32;来自激活量化核)
        Param::t_in("a_scales", Dtype::F32, 2),
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
    // 装配期 oracle(S1d;与 AWQ 臂同律):自检**只在算子在白名单里时跑**
    // (`registry::run_selfchecks` 随白名单生效)⇒ S2 把 W4A8 纳入生产档那天,
    // 它自动获得启动期对拍门;在那之前零启动成本。
    #[cfg(feature = "marlin")]
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> =
        Some(&crate::ops::oracles::marlin::MarlinW4a8AwqOracle);
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
pub static W4A8_AWQ: KernelMeta = meta_of::<MarlinGemmW4a8Awq>();

/// 本域全部已建表算子(**marlin 家族全建** ⇒ `names()` 可由本表派生)
pub static METAS: &[&KernelMeta] = &[&W4A16_AWQ, &W4A16_BF16, &W4A16_F16, &W4A8_AWQ];

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

    /// W4A8 臂:8 Block + 4 sz;槽序 = AWQ 臂的 `zeros` 之后插 `a_scales`
    /// (逐字锁住旧权威串锚 —— 漂移即红)
    #[test]
    fn w4a8_slot_order_matches_legacy_sig() {
        let m = meta_of::<MarlinGemmW4a8Awq>();
        assert_eq!(m.args_str(), "T,T,O,T,T,T,T,T,sz,sz,sz,sz");
        assert_eq!(m.out_pos(), Some(2));
        assert_eq!(m.params.len(), 12);
    }

    /// face 串(`client::marlin::SIG_W4A8`;models 的 `foreign_sig` 消费它)
    /// 与表派生串逐字一致
    #[cfg(feature = "marlin")]
    #[test]
    fn w4a8_slot_order_matches_face_sig() {
        assert_eq!(
            meta_of::<MarlinGemmW4a8Awq>().args_str(),
            crate::client::marlin::SIG_W4A8
        );
    }

    /// W4A8 臂的**激活槽 = 字节承载**(U8 线事实):L3 事实门按声明表核线槽
    /// dtype ⇒ 声明写错(F16)就再也拦不住 int8/f16 的混用
    #[test]
    fn w4a8_activation_slot_is_byte_carried() {
        use crate::spec::ParamType;
        let dt_of = |i: usize| match meta_of::<MarlinGemmW4a8Awq>().params[i].ty {
            ParamType::Tensor { dtype, .. } => dtype,
            other => panic!("槽 {i} 不是张量槽: {other:?}"),
        };
        assert_eq!(meta_of::<MarlinGemmW4a8Awq>().params[0].name, "a");
        assert_eq!(dt_of(0), Dtype::U8, "int8 激活 = 字节承载(与 fp8 池同惯例)");
        // a_scales = f32(与同形的 U32 槽可区分 —— 同型槽对调由事实门拦)
        assert_eq!(dt_of(5), Dtype::F32);
        assert_eq!(meta_of::<MarlinGemmW4a8Awq>().params[5].name, "a_scales");
    }
}
