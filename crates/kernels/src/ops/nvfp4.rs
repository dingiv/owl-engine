//! `nvfp4_gemm` —— NVFP4(e2m1 权重/激活 + e4m3/16 块缩放)GEMM 元数据。
//!
//! foreign(FFI 直调 `libnvfp4.a`,CUTLASS SM120 blockscaled mainloop):无 `.cu` 源。
//!
//! ## 适用面
//! - 硬件:必须 `Caps::fp4_blockscale`(sm_12x;准入表 [`crate::admit`] 管)
//! - 形状:`k % 16 == 0`(块缩放粒度)、`n % 8 == 0`(epilogue 向量化);
//!   **M 不设门**(2026-10-28 用户裁决:统一臂,decode 的 m=1 也走这条)
//! - **可进图**(2026-10-28):alpha 在设备侧算(`dg` 槽 = `alpha = g·gb`),
//!   发射路径零 host 工作/零同步 ⇒ decode / propose / verify 三张图都能捕

use owl_macros::kernel_tags;
use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};

/// `nvfp4_gemm` —— NVFP4 GEMM(prefill 臂;`m ≥ 128`)
pub struct NvFp4Gemm;

#[kernel_tags("dom.nvfp4", "role.gemm", "quant.nvfp4")]
impl KernelSpec for NvFp4Gemm {
    /// 执行家族(声明驱动路由;见 `KernelSpec::FAMILY`)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::nvfp4::GEMM);
    const ENTRY: &'static str = crate::family::nvfp4::GEMM;
    /// foreign(FFI + 设备量化器):无单一 `.cu` 源(两 TU 在 `cu/nvfp4/`)
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        // 激活 f16 [m,k]
        Param::t_in("a", Dtype::F16, 2),
        // 权重 fp4 [n, k/2](装载期打包一次)
        Param::t_in("bq", Dtype::U8, 2),
        // 权重块缩放 [n*k/16](CUTLASS 交错布局)
        Param::t_in("bsf", Dtype::U8, 1),
        // 输出 f16 [m,n]
        Param::t_out("out", Dtype::F16, 2),
        // 激活 fp4 [m, k/2] / 块缩放 / **f32[2] 标量槽**(`[0]` = amax→g,`[1]` = alpha = g·gb;
        // 由收尾核写定,之后 `[1]` 就是 GEMM 的 `alpha_ptr` —— 零 D2H ⇒ 可捕获)
        Param::t_in("aq", Dtype::U8, 2),
        Param::t_in("asf", Dtype::U8, 1),
        Param::t_in("dg", Dtype::F32, 1),
        Param::sz("m"),
        Param::sz("k"),
        Param::sz("n"),
        // 权重 per-tensor 全局缩放(host f32)
        Param::f32("gb"),
    ];
    /// 装配期数值门(每次 boot 跑一发真 GEMM,对 host 同规则参考)
    ///
    /// `cfg` 与 oracle 本体同门:未编入 `nvfp4` 时这条算子本就不可能发射
    /// (准入表先拒 `NvFp4Cutlass`),但**清单仍然列出**它 —— 声明面与实现面
    /// 解耦是既定的(选不选由 `driver`/`admit` 说话),所以这里只是把数值门
    /// 跟着它自己的库一起编。
    #[cfg(feature = "nvfp4")]
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> =
        Some(&crate::ops::oracles::nvfp4::NvFp4GemmOracle);
    #[cfg(not(feature = "nvfp4"))]
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> = None;
    const LAWS: &'static [Law] = &[Law::OutputFinite];
    /// 捕获证据:**可进图**(`NoHostWork`:无 host 工作 / 无同步 / 无 D2H)。
    ///
    /// 依据(2026-10-28 进图刀):
    /// - `alpha = g·gb` 在收尾核里算并写回 `dg` 槽(**设备侧**);
    /// - GEMM 的 epilogue 用 `fusion::LinearCombination`,**alpha 从 `alpha_ptr` 读**;
    /// - `gb` 是装载期常量(host 参数,捕获时烘进 kernel 参 — 图内合法)。
    ///
    /// 影史:上一版是 `NeedsHostPrep`(发射前回读 4B `ga` + 一次 stream 排空),
    /// 因此只能 eager —— decode/propose/verify 三张图全建不了(引擎在 boot 期
    /// 自降级并 WARN)。现在改为可捕。
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::NonDeterministic;
}

pub static GEMM: KernelMeta = meta_of::<NvFp4Gemm>();
