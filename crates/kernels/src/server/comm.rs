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

/// **reduce 的 grid 选择**(2026-10-17 §27.7;单 block 打不满本地访存 ——
/// 实测本地 RMW 37GB/s vs stage 核(copy)114GB/s vs 硬件 500+GB/s)。
///
/// 判据 = "每线程约 8 个 16B 向量"(`OWL_AR_VEC_PER_THREAD`),再夹到
/// `[1, MAX_BLOCKS]`:
/// - 小载荷(decode 10KB = 640 向量)⇒ grid 仍为 1(多块只添两级等待的钱);
/// - 大载荷(5.24MB = 327680 向量 / 256 线程 / 8)⇒ 夹到 MAX_BLOCKS。
///
/// **那一维 = 块数 × 块内线程数(在飞线程总数)**,两者都扫过:
/// - 块数:32 块 = 600.5µs / 64 块 = 627.8µs(`L dev` 6.23 vs 16.94)⇒ **32 更好**,
///   与 vLLM "too many SMs will cause contention on the bus"(它取 36)同向;
/// - 块内线程数:`asset::BLOCK` = **512**(256 → 512:`AR(5.24MB)` 608.6 → 590.2µs);
/// ⇒ 故上限记成**在飞线程总数** `MAX_THREADS`,grid = 它 / BLOCK(256 → 32 块;
///   512 → 16 块;1024 → 8 块 —— 三者扫出来 512 最好)。

fn ar_reduce_grid(dt: Dtype, bytes: u64) -> (u32, u32, u32) {
    /// 每线程的 16B 向量数
    const VEC_PER_THREAD: u64 = 8;
    let per_vec = 16u64 / (dt.size_bytes().max(1) as u64).min(16); // 16B/元素宽
    let n_vec = (bytes / (dt.size_bytes().max(1) as u64)).max(1) / per_vec.max(1);
    let blk = asset::BLOCK as u64;
    let want = n_vec.div_ceil(blk * VEC_PER_THREAD).max(1);
    let cap = (MAX_THREADS / blk).max(1);
    (want.min(cap) as u32, 1, 1)
}

/// **在飞线程总数上限**(见上注:块数×块内线程数才是那一维)。
pub(crate) const MAX_THREADS: u64 = 32 * 256; // = 8192;块数甜点 32 × 块内 256


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
    grid: (u32, u32, u32),
) -> Result<(), OpError> {
    exec.launch(
        res,
        op,
        k,
        kernel,
        grid,
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
            ar_reduce_grid(call.local().dtype, call.bytes()),
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
        let grid = ar_reduce_grid(call.out().dtype, call.bytes());
        launch_ar(
            Self::ID, "ar_reduce", res, exec, k, out_p, own_p, peer_p, call.bytes(),
            call.slot_bytes(), call.slab_off(), grid,
        )?;
        Ok(out)
    }
}
