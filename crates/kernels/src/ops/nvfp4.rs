//! `nvfp4_gemm` —— NVFP4(e2m1 权重/激活 + e4m3/16 块缩放)GEMM 元数据。
//!
//! 本域有**两条**算子(同族同 plan):
//! - [`NvFp4Gemm`] —— 运行期 GEMM(激活设备量化 + blockscaled MMA,**可进图**);
//! - [`NvFp4PackWeight`] —— **装载期**权重打包(源 `f16/bf16` 直上卡 → `q`+`sf`+`g`,
//!   与 AWQ `ct_repack` 同律:装载域的重排/量化也住在「按名选核」这条路上)。
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
    /// **确定**：amax 走 `atomicMax`（非负 f32 的位序 = 整型序，max 与归约顺序无关），
    /// 量化与 GEMM 都是纯前向、无原子累加 ⇒ **同输入位相输出**（跨进程同一 prompt
    /// 输出逐字节相同，已在 4B/9B 两档实测）。旧声明写 `NonDeterministic` 是把
    /// “用了 atomicMax”当成了“不确定” —— 它只对**加法**成立。
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static GEMM: KernelMeta = meta_of::<NvFp4Gemm>();

/// `nvfp4_pack_weight` —— **装载期设备侧权重打包**(2026-10-28 M7)。
///
/// 源 `f16/bf16 [N,K]`(DMA 上卡)→ 设备上 amax → `g = amax/(6·448)` →
/// `q`(e2m1)+ `sf`(e4m3/16,SFB 交错)+ `d_g[0] = g`。数学与 host
/// `pack_weight_b` 逐条相同(一次性对拍:q/sf 逐字节、g 位相同)。
///
/// **为什么它是一个正式算子而不是装载器里的私货**:与 AWQ 的 `ct_repack` 同律 ——
/// 装载域「DMA 上卡 + GPU 就地重排」这件事只能经 `KernelPick` 选核(用户裁决),
/// 装载器不自带第二套执行面。
///
/// **不是 GEMM**:不建 MMA、不进图、无 workspace;唯一宿主是装载相位。
pub struct NvFp4PackWeight;

#[kernel_tags("dom.load", "phase.load", "role.elem", "quant.nvfp4")]
impl KernelSpec for NvFp4PackWeight {
    /// 与 [`NvFp4Gemm`] **同族**(共用 plan 的 SFB 布局;族 runtime 按名分派)
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::nvfp4::GEMM);
    const ENTRY: &'static str = crate::family::nvfp4::PACK_WEIGHT;
    /// foreign(FFI + 设备量化器):无单一 `.cu` 源(两 TU 在 `cu/nvfp4/`)
    const SRC: &'static str = "";
    const PARAMS: &'static [Param] = &[
        // 源字节(**不透明**:f16 或 bf16,由 `src_bf16` 决定解释;`[N,K]` 行主序)
        Param::t_in("src", Dtype::U8, 2),
        // fp4 权重 [N, K/2](主产物)
        Param::t_out("q", Dtype::U8, 2),
        // 权重块缩放 [sfb_bytes](CUTLASS 交错布局;尺寸 = `nvfp4::sf_bytes`)
        Param::t_in("sf", Dtype::U8, 1),
        // per-tensor 全局缩放(f32[1],**装载面回读 4 字节**;不在热路径)
        Param::t_in("g", Dtype::F32, 1),
        Param::sz("n"),
        Param::sz("k"),
        // 源精度开关:0 = f16,非 0 = bf16
        Param::sz("src_bf16"),
    ];
    /// 无数值门:本算子**没有模型可见的输出**(权重字节),其数值正确性由
    /// 装载面在 `OWL_LOAD_VERIFY` 档**与 host 打包逐字节对拍**证明
    /// (`formats::load::load_nvfp4_device`),而不是再建一份 oracle ——
    /// 同一件事两份参考实现 = 两份会漂移的真值。
    const SELFCHECK: Option<&'static dyn crate::bootcheck::BootCheck> = None;
    const LAWS: &'static [Law] = &[];
    /// 装载相位专用(不在图内);发射路径本身无 host 工作/无同步
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    /// amax 走 `atomicMax`(非负 f32 位序 = 整型序),量化无原子累加 ⇒ 位一致
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static PACK_WEIGHT: KernelMeta = meta_of::<NvFp4PackWeight>();
