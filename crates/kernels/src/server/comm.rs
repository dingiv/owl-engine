//! 通信算子族 runtime(TP2 S2)—— allreduce stage/reduce 两族。
//!
//! 每族持三个 dtype 核(f32/f16/bf16;同资产同槽序,按线格式 dtype 分派)。
//! 几何自持:单 block × [`asset::BLOCK`](v1 单写者游标的前提;设计 §3.2)。
//!
//! **F2 尺寸自证**(家族直发前):payload 与 staging 两块都按账长校验
//! "请求落在块内" —— 内核越界不会当场报错(写进邻块),必须在这里结构化拒。

use crate::client::comm::{ArReduceCall, ArStageCall};
use crate::contract::{Bytes, Dtype, LaunchMsg, OpError, OpId};
use crate::device::{Exec, LaunchVal};
use crate::family::comm::asset;
use crate::registry::{FamilyGuard, KernelFamily};
use std::sync::Arc;

type K = Arc<cudarc::driver::CudaFunction>;

/// 按线格式 dtype 选核(f32/f16/bf16;其余 = 结构化拒)
fn pick<'a>(op: OpId, dt: Dtype, k32: &'a Option<K>, k16: &'a Option<K>, kbf: &'a Option<K>) -> Result<&'a K, OpError> {
    let slot = match dt {
        Dtype::F32 => k32,
        Dtype::F16 => k16,
        Dtype::BF16 => kbf,
        other => {
            return Err(OpError::Contract {
                op: op.0.to_string(),
                field: "payload dtype",
                expect: "F32/F16/BF16(三实例)".into(),
                got: format!("{other:?}"),
            })
        }
    };
    slot.as_ref().ok_or_else(|| OpError::Asset {
        op: op.0.to_string(),
        detail: "该 dtype 核未装载(assemble 缺失?)".into(),
    })
}

/// 块尺寸自证(payload 侧:`slab_off + bytes ≤ 账长`)
fn check_payload_span(op: OpId, what: &'static str, len: usize, off: u64, bytes: u64) -> Result<(), OpError> {
    let end = off.checked_add(bytes).ok_or_else(|| OpError::Contract {
        op: op.0.to_string(),
        field: what,
        expect: "slab_off + bytes 不溢出".into(),
        got: format!("off={off} bytes={bytes}"),
    })?;
    if end > len as u64 {
        return Err(OpError::Contract {
            op: op.0.to_string(),
            field: what,
            expect: format!("slab_off + bytes ≤ 账长({len}B)"),
            got: format!("off={off} + bytes={bytes} = {end}"),
        });
    }
    Ok(())
}

/// 块尺寸自证(staging 侧:`header + slot_bytes ≤ 账长`;slot 数据的容器)
fn check_staging(op: OpId, what: &'static str, len: usize, slot_bytes: u64) -> Result<(), OpError> {
    let need = asset::AR_HDR_BYTES.checked_add(slot_bytes).ok_or_else(|| OpError::Contract {
        op: op.0.to_string(),
        field: what,
        expect: "header + slot_bytes 不溢出".into(),
        got: format!("slot_bytes={slot_bytes}"),
    })?;
    if need > len as u64 {
        return Err(OpError::Contract {
            op: op.0.to_string(),
            field: what,
            expect: format!("header(64) + slot_bytes ≤ staging 账长({len}B)"),
            got: format!("needs={need} slot_bytes={slot_bytes}"),
        });
    }
    Ok(())
}

/// 公共发射参数(两族同型:6 槽)
#[allow(clippy::too_many_arguments)]
fn launch_ar(
    op: OpId,
    kernel: &'static str,
    res: &mut dyn crate::device::DeviceRes,
    exec: &mut Exec,
    k: &K,
    a: u64,
    own: u64,
    peer: u64,
    bytes: u64,
    slot_bytes: u64,
    slab_off: u64,
) -> Result<(), OpError> {
    exec.launch(
        res,
        op,
        k,
        kernel,
        (1, 1, 1),
        (asset::BLOCK, 1, 1),
        0,
        &[
            LaunchVal::Ptr(a),
            LaunchVal::Ptr(own),
            LaunchVal::Ptr(peer),
            LaunchVal::U64(bytes),
            LaunchVal::U64(slot_bytes),
            LaunchVal::U64(slab_off),
        ],
    )
}

// ============================================================================
// stage 族
// ============================================================================

/// allreduce stage runtime(`comm_ar_stage`)
pub struct CommArStageRuntime {
    k_f32: Option<K>,
    k_f16: Option<K>,
    k_bf16: Option<K>,
}

impl CommArStageRuntime {
    /// 族身份(声明面 `ops::comm::*::FAMILY` 引用同一字面量)
    pub const ID: OpId = OpId(crate::family::comm::COMM_AR_STAGE);
}

impl KernelFamily for CommArStageRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }

    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        let me = CommArStageRuntime {
            k_f32: Some(exec.load_aot(res, asset::STAGE_SRC, asset::STAGE_F32, Self::ID)?),
            k_f16: Some(exec.load_aot(res, asset::STAGE_SRC, asset::STAGE_F16, Self::ID)?),
            k_bf16: Some(exec.load_aot(res, asset::STAGE_SRC, asset::STAGE_BF16, Self::ID)?),
        };
        Ok(FamilyGuard::boxed(me))
    }

    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut Exec,
    ) -> Result<Bytes, OpError> {
        let (call, out) = ArStageCall::parse(msg)?;
        let k = pick(Self::ID, call.local().dtype, &self.k_f32, &self.k_f16, &self.k_bf16)?;
        // ── F2 尺寸自证(两块账长) ──
        let local_len = res.resolve_len(call.local().id)?;
        let own_len = res.resolve_len(call.own().id)?;
        let _ = res.resolve_len(call.peer().id)?; // peer 必须在册(跨卡别名块)
        check_payload_span(Self::ID, "local", local_len, call.slab_off(), call.bytes())?;
        check_staging(Self::ID, "own", own_len, call.slot_bytes())?;
        // ── 发射 ──
        let local_p = res.resolve(call.local().id)?;
        let own_p = res.resolve(call.own().id)?;
        let peer_p = res.resolve(call.peer().id)?;
        launch_ar(
            Self::ID, "ar_stage", res, exec, k, local_p, own_p, peer_p, call.bytes(),
            call.slot_bytes(), call.slab_off(),
        )?;
        Ok(out)
    }
}

// ============================================================================
// reduce 族
// ============================================================================

/// allreduce reduce runtime(`comm_ar_reduce`)
pub struct CommArReduceRuntime {
    k_f32: Option<K>,
    k_f16: Option<K>,
    k_bf16: Option<K>,
}

impl CommArReduceRuntime {
    pub const ID: OpId = OpId(crate::family::comm::COMM_AR_REDUCE);
}

impl KernelFamily for CommArReduceRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }

    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        let me = CommArReduceRuntime {
            k_f32: Some(exec.load_aot(res, asset::REDUCE_SRC, asset::REDUCE_F32, Self::ID)?),
            k_f16: Some(exec.load_aot(res, asset::REDUCE_SRC, asset::REDUCE_F16, Self::ID)?),
            k_bf16: Some(exec.load_aot(res, asset::REDUCE_SRC, asset::REDUCE_BF16, Self::ID)?),
        };
        Ok(FamilyGuard::boxed(me))
    }

    fn run(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
        exec: &mut Exec,
    ) -> Result<Bytes, OpError> {
        let (call, out) = ArReduceCall::parse(msg)?;
        let k = pick(Self::ID, call.out().dtype, &self.k_f32, &self.k_f16, &self.k_bf16)?;
        let out_len = res.resolve_len(call.out().id)?;
        let own_len = res.resolve_len(call.own().id)?;
        let _ = res.resolve_len(call.peer().id)?;
        check_payload_span(Self::ID, "out", out_len, call.slab_off(), call.bytes())?;
        check_staging(Self::ID, "own", own_len, call.slot_bytes())?;
        let out_p = res.resolve(call.out().id)?;
        let own_p = res.resolve(call.own().id)?;
        let peer_p = res.resolve(call.peer().id)?;
        launch_ar(
            Self::ID, "ar_reduce", res, exec, k, out_p, own_p, peer_p, call.bytes(),
            call.slot_bytes(), call.slab_off(),
        )?;
        Ok(out)
    }
}
