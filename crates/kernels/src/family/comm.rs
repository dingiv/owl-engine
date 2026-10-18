//! 通信算子族 —— allreduce stage/reduce(TP2 S2)。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §3(协议 / 两核 / 不变量 I1-I5);
//! 协议与 header 布局的**权威注记**在 `cu/comm/ar_stage.cu` 头注(与
//! `owl_comm::ar` 的常量对望,Rust 侧改 = 双侧同改 + 门重跑)。
//!
//! 两族(不是一族两核):一次 `LaunchMsg` 只承载一个核 ⇒ 各持自己的
//! `OpId`/装配面;`is_foreign` 供 server 分派谓词共用。

/// foreign-kernel 虚拟核名(server 分派臂;stage)
pub const COMM_AR_STAGE: &str = "comm_ar_stage";
/// foreign-kernel 虚拟核名(reduce)
pub const COMM_AR_REDUCE: &str = "comm_ar_reduce";

/// foreign 分派谓词(server `handle_launch` 前置检查)
pub fn is_foreign(name: &str) -> bool {
    name == COMM_AR_STAGE || name == COMM_AR_REDUCE
}

/// payload dtype → stage 发射名(**单源**;client 与 driver 共读)。
///
/// 非法 dtype = **响亮 panic**(driver 是 host 侧图构建/装配面 —— 声明了
/// 不支持的 dtype 是编程错,与 `driver.rs` 其他档同律)。
pub fn stage_entry(dt: crate::driver::DType) -> &'static str {
    use crate::driver::DType;
    match dt {
        DType::F32 => asset::STAGE_F32,
        DType::F16 => asset::STAGE_F16,
        DType::BF16 => asset::STAGE_BF16,
        other => panic!("comm.ar_stage: 不支持 dtype {other:?}(f32/f16/bf16 三实例)"),
    }
}

/// payload dtype → reduce 发射名(单源)
pub fn reduce_entry(dt: crate::driver::DType) -> &'static str {
    use crate::driver::DType;
    match dt {
        DType::F32 => asset::REDUCE_F32,
        DType::F16 => asset::REDUCE_F16,
        DType::BF16 => asset::REDUCE_BF16,
        other => panic!("comm.ar_reduce: 不支持 dtype {other:?}(f32/f16/bf16 三实例)"),
    }
}

/// 设备码资产 + 发射常量(源在树 ⇒ build.rs 目录扫描自动纳入;
/// 资产 id 与 `ops::comm` 的 `SRC` **同值** —— 改一处必改另一处,门会红)。
pub mod asset {
    pub const STAGE_SRC: &str = "cu/comm/ar_stage.cu";
    pub const REDUCE_SRC: &str = "cu/comm/ar_reduce.cu";

    pub const STAGE_F32: &str = "owl_comm_ar_stage_f32";
    pub const STAGE_F16: &str = "owl_comm_ar_stage_f16";
    pub const STAGE_BF16: &str = "owl_comm_ar_stage_bf16";
    pub const REDUCE_F32: &str = "owl_comm_ar_reduce_f32";
    pub const REDUCE_F16: &str = "owl_comm_ar_reduce_f16";
    pub const REDUCE_BF16: &str = "owl_comm_ar_reduce_bf16";

    /// 发射几何(v1 单 block 256 线程;设计 §3.2 —— 单写者游标的前提)
    pub const BLOCK: u32 = 256;

    /// staging header 字节数(与 `owl_comm::ar::HEADER_BYTES` 同值;
    /// 权威注记在 `cu/comm/ar_stage.cu` 头 —— 三处同改 + 门重跑)
    pub const AR_HDR_BYTES: u64 = 64;
}
