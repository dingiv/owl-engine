//! 家族底座 —— 资产/FFI/ABI(两平面共用的单源)。
//!
//! - [`gdn_chunked`]/[`gdn_scalar`]:AOT cubin 声明 + 发射常量 + ABI 注记
//! - [`cublas`]:OwlCublas 封装(handle/工作区/gemm)
//! - [`marlin`]:FFI + repack(装载域 weight packing)
//! - [`flashinfer`]:FFI + workspace 常量
//! - [`xqa`]:XQA 契约 + FFI(Ampere spec-verify;2026-10-13)
//!
//! **门控(2026-10-15 起)**:各族的**身份字面量/契约面零门**(声明面
//! `ops/<域>.rs` 与 models 要引用族名);**FFI/封装面随各自 feature 门**
//! (`#[cfg]` 落在文件内的门控项上)。

/// 通信族(TP2 S2:allreduce stage/reduce;族身份 + 资产,零门)
pub mod comm;
/// **构建期 cubin 资产登记表**(build.rs 生成;源文本哈希 → fatbin 字节)。
/// 门 = `device` —— 只有执行面(backends/cuda)按源查资产;models 侧零成本。
#[cfg(feature = "device")]
pub mod cubins;
/// cuBLAS:族身份零门;**OwlCublas 封装**随 feature=cublas(cudarc 绑定)
pub mod cublas;
/// FlashInfer:族身份 + workspace 常量 + FFI 声明(零依赖,零门)
pub mod flashinfer;
/// GDN chunked AOT cubin + 发射常量(零门)
pub mod gdn_chunked;
/// GDN scalar AOT cubin(零门)
pub mod gdn_scalar;
/// **GGUF 族**(2026-10-19):格式词汇 `GgmlTy` + 块几何 + 内核码唯一映射。
/// 零门零依赖(声明面;`cu/gguf/` 执行面随 `device` 门)。
pub mod gguf;
/// Marlin:族身份 + FFI 声明(零门);**repack** 随 feature=marlin(half/rayon)
pub mod marlin;
/// **NVFP4 块缩放 GEMM**(2026-10-28):名/槽序契约/foreign 谓词。
/// **族身份零门**(与 `family::cublas` 同律 —— `driver::resolve` 的表要它);
/// 执行面 `server::nvfp4` 与源 `cu/nvfp4/` 才是 feature 门槛的。
pub mod nvfp4;

/// **声明恒在、执行面随 feature 门控**的家族 id(装配门例外名单)。
///
/// 只有"声明面必须无条件存在、执行面由 feature 决定"的族才进这里;
/// 孤儿门对它们豁免(否则 `tag_wildcard` 测试档会把"本构建没有该执行面"
/// 误判成"家族遗漏")。真正不可跑仍由 [`crate::admit`] 拒绝。
pub const BUILD_GATED_FAMILIES: &[&str] = &[nvfp4::FAMILY_ID];

/// 该家族是否"构建期可选"(见 [`BUILD_GATED_FAMILIES`])。
#[must_use]
pub fn is_build_gated_family(id: &str) -> bool {
    BUILD_GATED_FAMILIES.contains(&id)
}
/// XQA 契约 + FFI(feature=xqa 时链接;契约面零门 —— 层侧 sig 查询要它)
pub mod xqa;
