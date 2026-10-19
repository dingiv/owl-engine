//! cuBLAS GEMM 元数据(foreign:FFI 直调;无 `.cu` 源)。
//!
//! 槽序此前只住两处:models 的 `lower_gemm` 手摆 `Vec<Arg>` 与
//! `client::cublas::parse_gemm` 的解析断言。现在**唯一真源 = 本表**
//! (槽序由 `PARAMS` 派生),server 侧 `check` 同时校验发射端。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};

/// `cublas_gemm_f16` —— f16 GEMM(lm_head / 非量化线性层)
pub struct CublasGemmF16;

impl KernelSpec for CublasGemmF16 {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::cublas::GEMM_F16);
    const ENTRY: &'static str = crate::family::cublas::GEMM_F16;
    /// foreign(FFI):无 `.cu` 源
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F16, 2),
        Param::t_in("b", Dtype::F16, 2),
        Param::t_out("out", Dtype::F16, 2),
        Param::sz("m"),
        Param::sz("k"),
        Param::sz("n"),
        // 转置标志(**不是变体选择**;cublas nt 形参)
        Param::sz("nt"),
    ];
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> =
        Some(&crate::ops::oracles::cublas_gemm::CublasGemmOracle);
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::NonDeterministic;
}

/// `cublas_gemm_bf16` —— BF16 GEMM(DFlash2 草稿投影)
pub struct CublasGemmBf16;

impl KernelSpec for CublasGemmBf16 {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::cublas::GEMM_F16);
    const ENTRY: &'static str = crate::family::cublas::GEMM_BF16;
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::BF16, 2),
        Param::t_in("b", Dtype::BF16, 2),
        Param::t_out("out", Dtype::BF16, 2),
        Param::sz("m"),
        Param::sz("k"),
        Param::sz("n"),
        Param::sz("nt"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::NonDeterministic;
}

/// `cublas_gemm_f16_dst` —— **写列窗** f16 GEMM(W-2d;GGUF dequant+cuBLAS 臂)
pub struct CublasGemmF16Dst;

impl KernelSpec for CublasGemmF16Dst {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::cublas::GEMM_F16);
    const ENTRY: &'static str = crate::family::cublas::GEMM_F16_DST;
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        Param::t_in("a", Dtype::F16, 2),
        Param::t_in("b", Dtype::F16, 2),
        // out = **既有块**(`Op::CallDst` 的 dest);本核只写它的 `[off, off+n)` 列窗
        Param::t_out("out", Dtype::F16, 2),
        Param::sz("m"),
        Param::sz("k"),
        Param::sz("n"),
        Param::sz("nt"),
        Param::sz("ldc"),
        Param::sz("off"),
    ];
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::NonDeterministic;
}

pub static GEMM_F16: KernelMeta = meta_of::<CublasGemmF16>();
pub static GEMM_F16_DST: KernelMeta = meta_of::<CublasGemmF16Dst>();
pub static GEMM_BF16: KernelMeta = meta_of::<CublasGemmBf16>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[&GEMM_F16, &GEMM_BF16, &GEMM_F16_DST];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::meta_of;

    /// 槽序与 models `lower_gemm` 的手摆序(逐字抄录为锚)一致
    #[test]
    fn gemm_slot_order_matches_handwritten_lower() {
        assert_eq!(meta_of::<CublasGemmF16>().args_str(), "T,T,O,sz,sz,sz,sz");
        assert_eq!(meta_of::<CublasGemmBf16>().args_str(), "T,T,O,sz,sz,sz,sz");
    }
}
