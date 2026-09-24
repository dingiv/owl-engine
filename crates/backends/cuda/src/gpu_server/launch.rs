//! kernel 发射装配:LaunchMsg 参数槽 → 设备指针/类型化标量 → builder.arg。
//!
//! server 是哑执行器:本模块不做任何算子语义判断,只按类型化槽序对位。

use crate::ffi::{CudaStream, LaunchConfig, PushKernelArg};
use crate::gpu_server::state::{GpuCtx, KernelCache};
use owl_models::client::{Arg, LaunchMsg};
use owl_models::ModelError;
use std::sync::Arc;

/// 参数槽(发射前装配;与 kernel 形参宽度严格对位 —— 坑 I)
enum Slot {
    Ptr(u64),
    U(u64),
    I(i32),
    F(f32),
}

/// 发射一单(非阻塞;发射即返回,完成由事件跟踪)。
/// 返回输出块句柄(槽序契约:输出块 = 最后一个 Block 参数)。
pub(super) fn issue_launch(
    ctx: &GpuCtx,
    stream: &Arc<CudaStream>,
    kernels: &mut KernelCache,
    msg: &LaunchMsg,
) -> Result<u64, ModelError> {
    // 1. 参数槽装配:Block → 设备指针;标量按类型入槽
    let mut slots: Vec<Slot> = Vec::with_capacity(msg.args.len());
    for a in &msg.args {
        match a {
            Arg::Block { id } => {
                let (p, _len) = ctx.block_ptr(*id, stream)?;
                slots.push(Slot::Ptr(p));
            }
            Arg::U64(v) => slots.push(Slot::U(*v)),
            Arg::I32(v) => slots.push(Slot::I(*v)),
            Arg::F32(v) => slots.push(Slot::F(*v)),
        }
    }

    if std::env::var_os("OWL_DEBUG").is_some() {
        eprintln!(
            "[dbg launch] {} slots = {:?}",
            msg.kernel.name,
            msg.args
                .iter()
                .map(|a| match a {
                    Arg::Block { id } => format!("Block({id})"),
                    Arg::U64(v) => format!("U64({v})"),
                    Arg::I32(v) => format!("I32({v})"),
                    Arg::F32(v) => format!("F32({v})"),
                })
                .collect::<Vec<_>>()
        );
    }

    // 2. 懒编译(缓存命中直返)
    let func = kernels.ensure_kernel(&ctx.ctx, &msg.kernel.name, &msg.kernel.source)?;

    // 3. 发射(非阻塞:提交进流即返回)
    unsafe {
        let mut builder = stream.launch_builder(&func);
        for slot in &slots {
            match slot {
                Slot::Ptr(v) => builder.arg(v),
                Slot::U(v) => builder.arg(v),
                Slot::I(v) => builder.arg(v),
                Slot::F(v) => builder.arg(v),
            };
        }
        builder
            .launch(LaunchConfig {
                grid_dim: msg.grid,
                block_dim: msg.block,
                shared_mem_bytes: msg.shared_mem,
            })
            .map_err(|e| ModelError::Msg(format!("launch({}): {e}", msg.kernel.name)))?;
    }

    // 4. 输出块句柄(槽序契约:最后一个 Block)
    let out_id = msg
        .args
        .iter()
        .rev()
        .find_map(|a| match a {
            Arg::Block { id } => Some(*id),
            Arg::U64(_) | Arg::I32(_) | Arg::F32(_) => None,
        })
        .ok_or_else(|| ModelError::Msg("launch: args 中无输出块".to_string()))?;
    Ok(out_id)
}
