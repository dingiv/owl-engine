//! 客户端平面 —— 类型化调用框架(models 视角;两平面之一)。
//!
//! 每家族一文件:face = 名字常量/槽序单源/builder 校验/线格式。
//! models 只 `use` 本平面,不拼名字字符串、不摆参数槽。
//! 对偶面 = [`crate::server`](家族 runtime;DeviceRes 由后端实现)。

/// native 语义算子胖算子(样例:NarrowStrided;值域校验内嵌)
pub mod native;
/// GDN chunked(face;零门)
pub mod gdn_chunked;
/// GDN scalar(face;零门)
pub mod gdn_scalar;
/// cuBLAS GEMM face(feature=cublas,随底座)
#[cfg(feature = "cublas")]
pub mod cublas;
/// Marlin face(feature=marlin,随底座)
#[cfg(feature = "marlin")]
pub mod marlin;
/// FlashInfer 槽序签名 + scale 编码(ungated:models 零 feature 消费方
/// 也引用;解析面仍在 feature = "flashinfer" 的 flashinfer.rs)
pub mod fi_sig {
    /// 槽序签名(9 Block + 8 sz,O = 输出槽;槽序单源,models 禁止手撸 ——
    /// 2026-10-12 review H 案)
    pub const SIG: &str = "T,T,T,T,T,T,T,T,O,sz,sz,sz,sz,sz,sz,sz,sz";

    /// attention scale 位型编码(f32 bits 过 U64 槽,server 面 from_bits 还原;
    /// 编码单源 —— models 调用点禁止手撸 to_bits)
    pub fn scale_bits(hd: usize) -> u64 {
        (1.0f32 / (hd as f32).sqrt()).to_bits() as u64
    }
}

/// FlashInfer prefill 解析面(feature=flashinfer,随底座)
#[cfg(feature = "flashinfer")]
pub mod flashinfer;
