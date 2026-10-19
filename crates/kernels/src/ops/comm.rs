//! 通信算子族元数据 —— allreduce stage/reduce(TP2 S2)。
//!
//! 声明面承载三件事(与其他族同律):
//! 1. **槽序唯一真源**(`PARAMS`;`client::comm` 的 parse 按名取址);
//! 2. **表 ⇔ `.cu` 签名互证**(`tests/op_tables.rs` 逐参比对);
//! 3. **族归属**(`FAMILY` ⇒ `KernelFamily::ops()` 派生,server 按名路由)。
//!
//! dtype 分档 = 三个实例(`f32/f16/bf16`);stage 与 reduce 各一表(核不同)。
//! payload 槽 rank 声明 0(显式阶数无关:AR 挂在任意算子输出上);
//! staging 两槽(own/peer)是**字节块**(U8)—— 它们是 VMM 分配的裸块。

use crate::contract::{Dtype, Law};
use crate::spec::{meta_of, CaptureEvidence, Determinism, KernelMeta, KernelSpec, Param};

// ============================================================================
// stage(本端 payload → 本端 staging slot;发布 produced)
// ============================================================================

/// `owl_comm_ar_stage_f16`
pub struct CommArStageF16;
impl KernelSpec for CommArStageF16 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_STAGE);
    const ENTRY: &'static str = "owl_comm_ar_stage_f16";
    const SRC: &'static str = "cu/comm/ar_stage.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("local", Dtype::F16, 0),
        Param::t_out("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("data", Dtype::U8, 1),
        Param::sz("bytes"),
        Param::sz("slot_bytes"),
        Param::sz("slab_off"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_comm_ar_stage_bf16`
pub struct CommArStageBf16;
impl KernelSpec for CommArStageBf16 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_STAGE);
    const ENTRY: &'static str = "owl_comm_ar_stage_bf16";
    const SRC: &'static str = "cu/comm/ar_stage.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("local", Dtype::BF16, 0),
        Param::t_out("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("data", Dtype::U8, 1),
        Param::sz("bytes"),
        Param::sz("slot_bytes"),
        Param::sz("slab_off"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_comm_ar_stage_f8_f16` —— **fp8 载荷档**(§47 第二步;2026-10-19)
/// 
/// 与常规档同协议/同几何:数据面换成 e4m3(1B/元素 + 64 元素/尺度 f16)。
pub struct CommArStageF8F16;
impl KernelSpec for CommArStageF8F16 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_STAGE);
    const ENTRY: &'static str = "owl_comm_ar_stage_f8_f16";
    const SRC: &'static str = "cu/comm/ar_stage.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("local", Dtype::F16, 0),
        Param::t_out("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("data", Dtype::U8, 1),
        Param::sz("bytes"),
        Param::sz("slot_bytes"),
        Param::sz("slab_off"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
/// `owl_comm_ar_stage_f8_bf16` —— **fp8 载荷档**(§47 第二步;2026-10-19)
/// 
/// 与常规档同协议/同几何:数据面换成 e4m3(1B/元素 + 64 元素/尺度 f16)。
pub struct CommArStageF8Bf16;
impl KernelSpec for CommArStageF8Bf16 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_STAGE);
    const ENTRY: &'static str = "owl_comm_ar_stage_f8_bf16";
    const SRC: &'static str = "cu/comm/ar_stage.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("local", Dtype::BF16, 0),
        Param::t_out("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("data", Dtype::U8, 1),
        Param::sz("bytes"),
        Param::sz("slot_bytes"),
        Param::sz("slab_off"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_comm_ar_stage_f32`
pub struct CommArStageF32;
impl KernelSpec for CommArStageF32 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_STAGE);
    const ENTRY: &'static str = "owl_comm_ar_stage_f32";
    const SRC: &'static str = "cu/comm/ar_stage.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_in("local", Dtype::F32, 0),
        Param::t_out("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("data", Dtype::U8, 1),
        Param::sz("bytes"),
        Param::sz("slot_bytes"),
        Param::sz("slab_off"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

// ============================================================================
// reduce(原地累加对端 slot;发布 consumed)
// ============================================================================

/// `owl_comm_ar_reduce_f16`
pub struct CommArReduceF16;
impl KernelSpec for CommArReduceF16 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_REDUCE);
    const ENTRY: &'static str = "owl_comm_ar_reduce_f16";
    const SRC: &'static str = "cu/comm/ar_reduce.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_out("out", Dtype::F16, 0),
        Param::t_in("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("data", Dtype::U8, 1),
        Param::sz("bytes"),
        Param::sz("slot_bytes"),
        Param::sz("slab_off"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_comm_ar_reduce_bf16`
pub struct CommArReduceBf16;
impl KernelSpec for CommArReduceBf16 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_REDUCE);
    const ENTRY: &'static str = "owl_comm_ar_reduce_bf16";
    const SRC: &'static str = "cu/comm/ar_reduce.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_out("out", Dtype::BF16, 0),
        Param::t_in("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("data", Dtype::U8, 1),
        Param::sz("bytes"),
        Param::sz("slot_bytes"),
        Param::sz("slab_off"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_comm_ar_reduce_f32`
pub struct CommArReduceF32;
impl KernelSpec for CommArReduceF32 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_REDUCE);
    const ENTRY: &'static str = "owl_comm_ar_reduce_f32";
    const SRC: &'static str = "cu/comm/ar_reduce.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_out("out", Dtype::F32, 0),
        Param::t_in("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("data", Dtype::U8, 1),
        Param::sz("bytes"),
        Param::sz("slot_bytes"),
        Param::sz("slab_off"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_comm_ar_reduce_f8_f16` —— **fp8 载荷档**(§47 第二步;2026-10-19)
/// 
/// 与常规档同协议/同几何:数据面换成 e4m3(1B/元素 + 64 元素/尺度 f16)。
pub struct CommArReduceF8F16;
impl KernelSpec for CommArReduceF8F16 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_REDUCE);
    const ENTRY: &'static str = "owl_comm_ar_reduce_f8_f16";
    const SRC: &'static str = "cu/comm/ar_reduce.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_out("out", Dtype::F16, 0),
        Param::t_in("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("data", Dtype::U8, 1),
        Param::sz("bytes"),
        Param::sz("slot_bytes"),
        Param::sz("slab_off"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}
/// `owl_comm_ar_reduce_f8_bf16` —— **fp8 载荷档**(§47 第二步;2026-10-19)
/// 
/// 与常规档同协议/同几何:数据面换成 e4m3(1B/元素 + 64 元素/尺度 f16)。
pub struct CommArReduceF8Bf16;
impl KernelSpec for CommArReduceF8Bf16 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_REDUCE);
    const ENTRY: &'static str = "owl_comm_ar_reduce_f8_bf16";
    const SRC: &'static str = "cu/comm/ar_reduce.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_out("out", Dtype::BF16, 0),
        Param::t_in("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("data", Dtype::U8, 1),
        Param::sz("bytes"),
        Param::sz("slot_bytes"),
        Param::sz("slab_off"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

// ============================================================================
// 1-stage(对称缓冲 ⇒ 融合核;TP2-施工设计.md §35)—— 无 stage 段、零拷贝
// ============================================================================

/// `owl_comm_ar_1stage_f16`(out 就是本端槽;peer_data 是对端槽的别名)
pub struct CommAr1StageF16;
impl KernelSpec for CommAr1StageF16 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_1STAGE);
    const ENTRY: &'static str = "owl_comm_ar_1stage_f16";
    const SRC: &'static str = "cu/comm/ar_reduce.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_out("out", Dtype::F16, 0),
        Param::t_in("src", Dtype::F16, 0),
        Param::t_in("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("peer_data", Dtype::U8, 1),
        Param::sz("bytes"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_comm_ar_1stage_bf16`
pub struct CommAr1StageBf16;
impl KernelSpec for CommAr1StageBf16 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_1STAGE);
    const ENTRY: &'static str = "owl_comm_ar_1stage_bf16";
    const SRC: &'static str = "cu/comm/ar_reduce.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_out("out", Dtype::BF16, 0),
        Param::t_in("src", Dtype::BF16, 0),
        Param::t_in("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("peer_data", Dtype::U8, 1),
        Param::sz("bytes"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

/// `owl_comm_ar_1stage_f32`
pub struct CommAr1StageF32;
impl KernelSpec for CommAr1StageF32 {
    const FAMILY: crate::contract::OpId = crate::contract::OpId(crate::family::comm::COMM_AR_1STAGE);
    const ENTRY: &'static str = "owl_comm_ar_1stage_f32";
    const SRC: &'static str = "cu/comm/ar_reduce.cu";
    const PARAMS: &'static [Param] = &[
        Param::t_out("out", Dtype::F32, 0),
        Param::t_in("src", Dtype::F32, 0),
        Param::t_in("own", Dtype::U8, 1),
        Param::t_in("peer", Dtype::U8, 1),
        Param::t_in("peer_data", Dtype::U8, 1),
        Param::sz("bytes"),
    ];
    const LAWS: &'static [Law] = &[];
    const CAPTURE: CaptureEvidence = CaptureEvidence::NoHostWork;
    const DETERMINISM: Determinism = Determinism::Deterministic;
}

pub static AR_STAGE_F16: KernelMeta = meta_of::<CommArStageF16>();
pub static AR_STAGE_BF16: KernelMeta = meta_of::<CommArStageBf16>();
pub static AR_STAGE_F32: KernelMeta = meta_of::<CommArStageF32>();
pub static AR_REDUCE_F16: KernelMeta = meta_of::<CommArReduceF16>();
pub static AR_REDUCE_BF16: KernelMeta = meta_of::<CommArReduceBf16>();
pub static AR_REDUCE_F32: KernelMeta = meta_of::<CommArReduceF32>();
pub static AR_1STAGE_F16: KernelMeta = meta_of::<CommAr1StageF16>();
pub static AR_1STAGE_BF16: KernelMeta = meta_of::<CommAr1StageBf16>();
pub static AR_1STAGE_F32: KernelMeta = meta_of::<CommAr1StageF32>();
// **fp8 载荷档**(§47 第二步;2026-10-19)—— 登记进表才进 `ops::all()` 的预加载面
pub static AR_STAGE_F8_F16: KernelMeta = meta_of::<CommArStageF8F16>();
pub static AR_STAGE_F8_BF16: KernelMeta = meta_of::<CommArStageF8Bf16>();
pub static AR_REDUCE_F8_F16: KernelMeta = meta_of::<CommArReduceF8F16>();
pub static AR_REDUCE_F8_BF16: KernelMeta = meta_of::<CommArReduceF8Bf16>();

/// 本域全部已建表算子
pub static METAS: &[&KernelMeta] = &[
    &AR_STAGE_F8_F16,
    &AR_STAGE_F8_BF16,
    &AR_REDUCE_F8_F16,
    &AR_REDUCE_F8_BF16,
    &AR_STAGE_F16,
    &AR_STAGE_BF16,
    &AR_STAGE_F32,
    &AR_REDUCE_F16,
    &AR_REDUCE_BF16,
    &AR_REDUCE_F32,
    &AR_1STAGE_F16,
    &AR_1STAGE_BF16,
    &AR_1STAGE_F32,
];
