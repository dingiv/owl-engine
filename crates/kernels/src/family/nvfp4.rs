//! NVFP4 块缩放 GEMM 家族底座(名 / 槽序契约 / foreign 谓词)。
//!
//! 执行面在 [`crate::server::nvfp4`](feature `device`)与 `cu/nvfp4/*.cu`(FFI);
//! 声明面在 [`crate::ops::nvfp4`]。三者共用本模块的字面量 —— 名字只写一次。
//!
//! ## 槽序(唯一真源 = `ops::nvfp4::PARAMS`;本常量是它的可读投影)
//! ```text
//! [T a, T bq, T bsf, O out, T aq, T asf, T dg, sz m, sz k, sz n, f32 gb]
//! ```
//! - `a`   = 激活 f16 `[m, k]`(块型)
//! - `bq`  = 权重 fp4 `[n, k/2]`(u8;装载期 `pack_b` 一次常驻)
//! - `bsf` = 权重块缩放 `[n*k/16]`(u8;CUTLASS 交错布局)
//! - `out` = 输出 f16 `[m, n]`
//! - `aq/asf/dg` = **本次调用的量化产物**(u8/u8/f32[1];层侧每轮瞬态)
//! - `gb`  = 权重 per-tensor 全局缩放(host f32;`ga` 由设备算完回读)

/// foreign-kernel 核名(server 分派臂;**字面量住址 = 本模块**)
pub const GEMM: &str = "nvfp4_gemm";

/// **装载期设备侧权重打包**核名(2026-10-28 M7)。
///
/// 与 [`GEMM`] 同族同 plan(`sfb` 步长只依赖 `N,K`),但**不是 GEMM**:
/// 源 `f16/bf16 [N,K]` → `q` + `sf`(SFB)+ `g`(f32,回读)。装载面按它
/// 把「DMA 上卡 + GPU 就地打包」与 AWQ 的 `ct_repack` 对齐(同一个落点)。
/// 不发射 MMA —— 只有三发量化核(`cu/nvfp4/nvfp4_quant_act.cu`)。
pub const PACK_WEIGHT: &str = "nvfp4_pack_weight";

/// foreign 分派谓词
pub fn is_foreign(name: &str) -> bool {
    name == GEMM || name == PACK_WEIGHT
}

/// 槽序契约(LaunchMsg.args;与 `cu/nvfp4/nvfp4_gemm.cu` 的 `run` 形参对齐)
pub const GEMM_SLOTS: &str =
    "[T a, T bq, T bsf, O out, T aq, T asf, T dg, sz m, sz k, sz n, f32 gb]";

/// 槽序契约(权重打包;与 `cu/nvfp4/nvfp4_quant_act.cu::owl_nvfp4_pack_b_dev` 对齐)
///
/// `src` 是**不透明源字节**(f16 或 bf16,由 `src_bf16` 决定解释)—— 用 `U8`
/// 是诚实声明(同一个核吃两种源精度,声明成 F16 会在 bf16 源上说谎)。
pub const PACK_WEIGHT_SLOTS: &str = "[T src, O q, T sf, T g, sz n, sz k, sz src_bf16]";
