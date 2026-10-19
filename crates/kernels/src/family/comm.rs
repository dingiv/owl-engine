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
/// foreign-kernel 虚拟核名(**1-stage 融合核**;对称缓冲版,`TP2-施工设计.md` §35)
pub const COMM_AR_1STAGE: &str = "comm_ar_1stage";
/// foreign-kernel 虚拟核名(**v1 两核融合**;§11.2 2026-10-19 —— 一核做完
/// stage+reduce 同一协议,载荷仍是私有块)
pub const COMM_AR_FUSED: &str = "comm_ar_fused";

/// foreign 分派谓词(server `handle_launch` 前置检查)
pub fn is_foreign(name: &str) -> bool {
    name == COMM_AR_STAGE
        || name == COMM_AR_REDUCE
        || name == COMM_AR_1STAGE
        || name == COMM_AR_FUSED
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
/// **fp8 载荷档的发射名**(`f8 = false` ⇒ 逐字返回常规档 ⇒ 零行为变化)
pub fn stage_entry_f8(dt: crate::driver::DType, f8: bool) -> &'static str {
    use crate::driver::DType;
    if !f8 {
        return stage_entry(dt);
    }
    match dt {
        DType::F16 => asset::STAGE_F8_F16,
        DType::BF16 => asset::STAGE_F8_BF16,
        other => panic!("comm.ar_stage(f8): 载荷量化只支持 f16/bf16,实得 {other:?}"),
    }
}

/// 同上(reduce 面)
pub fn reduce_entry_f8(dt: crate::driver::DType, f8: bool) -> &'static str {
    use crate::driver::DType;
    if !f8 {
        return reduce_entry(dt);
    }
    match dt {
        DType::F16 => asset::REDUCE_F8_F16,
        DType::BF16 => asset::REDUCE_F8_BF16,
        other => panic!("comm.ar_reduce(f8): 载荷量化只支持 f16/bf16,实得 {other:?}"),
    }
}

pub fn reduce_entry(dt: crate::driver::DType) -> &'static str {
    use crate::driver::DType;
    match dt {
        DType::F32 => asset::REDUCE_F32,
        DType::F16 => asset::REDUCE_F16,
        DType::BF16 => asset::REDUCE_BF16,
        other => panic!("comm.ar_reduce: 不支持 dtype {other:?}(f32/f16/bf16 三实例)"),
    }
}

/// payload dtype → 1-stage 发射名(单源)
pub fn one_stage_entry(dt: crate::driver::DType) -> &'static str {
    use crate::driver::DType;
    match dt {
        DType::F32 => asset::ONE_STAGE_F32,
        DType::F16 => asset::ONE_STAGE_F16,
        DType::BF16 => asset::ONE_STAGE_BF16,
        other => panic!("comm.ar_1stage: 不支持 dtype {other:?}(f32/f16/bf16 三实例)"),
    }
}

/// payload dtype → **fused** 发射名(单源;§11.2 2026-10-19)
/// **只非 fp8 载荷** —— fp8 腿仍走两核(见 `cu/comm/ar_fused.cu` 头注)。
pub fn fused_entry(dt: crate::driver::DType) -> &'static str {
    use crate::driver::DType;
    match dt {
        DType::F32 => asset::FUSED_F32,
        DType::F16 => asset::FUSED_F16,
        DType::BF16 => asset::FUSED_BF16,
        other => panic!("comm.ar_fused: 不支持 dtype {other:?}(f32/f16/bf16 三实例)"),
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

    /// **fp8 载荷档**(§47 第二步;2026-10-19):同协议/同几何,搬运面换成
    /// "量化写 + 反量化读"(载荷 1B/元素 + 64 元素/尺度 ⇒ 1.031× 原字节)。
    pub const STAGE_F8_F16: &str = "owl_comm_ar_stage_f8_f16";
    pub const STAGE_F8_BF16: &str = "owl_comm_ar_stage_f8_bf16";
    pub const REDUCE_F8_F16: &str = "owl_comm_ar_reduce_f8_f16";
    pub const REDUCE_F8_BF16: &str = "owl_comm_ar_reduce_f8_bf16";

    /// **1-stage 与 reduce 同源文件**(同族两种形态;共享 dtype 桥与向量循环)
    pub const ONE_STAGE_SRC: &str = "cu/comm/ar_reduce.cu";
    pub const ONE_STAGE_F32: &str = "owl_comm_ar_1stage_f32";
    pub const ONE_STAGE_F16: &str = "owl_comm_ar_1stage_f16";
    pub const ONE_STAGE_BF16: &str = "owl_comm_ar_1stage_bf16";

    /// **v1 两核融合**(§11.2;2026-10-19):helper **逐字复制**自
    /// `ar_stage.cu`(拷贝)/`ar_reduce.cu`(读加)—— 遵本仓纪律:`.cu` 之间
    /// 不 include(构建期增量指纹只覆盖 `.cu` 自身);改 helper = **三处同改**。
    pub const FUSED_SRC: &str = "cu/comm/ar_fused.cu";
    pub const FUSED_F32: &str = "owl_comm_ar_fused_f32";
    pub const FUSED_F16: &str = "owl_comm_ar_fused_f16";
    pub const FUSED_BF16: &str = "owl_comm_ar_fused_bf16";

    /// 发射几何(v1 单 block 256 线程;设计 §3.2 —— 单写者游标的前提)
    ///
    /// ★ **2026-10-17 MLP 律(§25)**:单 block 能输满 PCIe 的**唯一**条件是
    /// 核内**在飞访存数 ≫ 1** —— 载荷 >~256KB 时数据面必须走
    /// "16B 向量 + 8 路展开"(`cu/comm/ar_*.cu`);若有人改回逐元素循环,
    /// 带宽会回落到 **~360MB/s**(实测病态:每元素一趟 PCIe 往返 1.41µs)
    /// 而**不是**回落到 6-13GB/s。要改多 block 先看 §25.7(游标协议要动)。
    /// ★ **2026-10-17 扫块内线程数**:512(256 → 512,`AR(5.24MB)` 608.6 → **590.2µs**,
    /// −3.0%;1024 ≈ 512 但在 10.49MB 上略差 1144.9 vs 1142.3 ⇒ 取 512)。
    /// 对照:NCCL 官方样例 `__launch_bounds__(1024)`、vLLM 512 —— 我们原先 256 是
    /// **漏扫的一维**(块数扫过 32/64,块内线程数一直没动)。
    /// 在飞线程总数上限见 `server::comm::MAX_THREADS`(= 8192):块数 × 块内线程数
    /// 才是那一维,所以本值改动要同步看 grid 选择。
    pub const BLOCK: u32 = 512;

    /// staging header 字节数(与 `owl_comm::ar::HEADER_BYTES` 同值;
    /// 权威注记在 `cu/comm/ar_stage.cu` 头 —— 三处同改 + 门重跑)
    pub const AR_HDR_BYTES: u64 = 64;
}
