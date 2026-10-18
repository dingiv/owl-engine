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

/// cuBLAS:族身份零门;**OwlCublas 封装**随 feature=cublas(cudarc 绑定)
pub mod cublas;
/// XQA 契约 + FFI(feature=xqa 时链接;契约面零门 —— 层侧 sig 查询要它)
pub mod xqa;
/// FlashInfer:族身份 + workspace 常量 + FFI 声明(零依赖,零门)
pub mod flashinfer;
/// GDN chunked AOT cubin + 发射常量(零门)
pub mod gdn_chunked;
/// GDN scalar AOT cubin(零门)
pub mod gdn_scalar;
/// 通信族(TP2 S2:allreduce stage/reduce;族身份 + 资产,零门)
pub mod comm;
/// Marlin:族身份 + FFI 声明(零门);**repack** 随 feature=marlin(half/rayon)
pub mod marlin;
/// **构建期 cubin 资产登记表**(build.rs 生成;源文本哈希 → fatbin 字节)。
/// 门 = `device` —— 只有执行面(backends/cuda)按源查资产;models 侧零成本。
#[cfg(feature = "device")]
pub mod cubins;
