//! kernel 发射装配:LaunchMsg 参数槽 → 设备指针/类型化标量 → builder.arg。
//!
//! server 是哑执行器:本模块不做任何算子语义判断,只按类型化槽序对位。

use crate::ffi::{CudaStream, LaunchConfig, PushKernelArg};
use crate::state::{GpuCtx, KernelCache};
use owl_iface::contract::{Arg, LaunchMsg};
use owl_iface::contract::ModelError;
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
        let addrs: Vec<String> = slots.iter().map(|s| match s {
            Slot::Ptr(v) => format!("ptr {v:#x}"),
            Slot::U(v) => format!("u64 {v}"),
            Slot::I(v) => format!("i32 {v}"),
            Slot::F(v) => format!("f32 {v}"),
        }).collect();
        eprintln!("[dbg launch-ptr] {} = {addrs:?}", msg.kernel.name);
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

    // 2.4 硬顶守卫:动态 smem 超设备 opt-in 上限 → 结构化拒绝(提前到
    // 发射前,而非驱动层泛化 INVALID_VALUE)。开销 = 一次字段比较
    // (~ns;boot 时查一次设备属性缓存,不逐发射调 CUDA)。decode 热路径
    // 走图回放不经此处;捕获期发射会被本守卫覆盖一次。
    if msg.shared_mem as usize > ctx.smem_optin() {
        return Err(ModelError::Msg(format!(
            "launch({}): 动态 smem {}B 超设备 opt-in 上限 {}B —— \
             检查 smem 契约推导(长 ctx 候选:v2 分块核,vLLM 同款弃 v1)",
            msg.kernel.name, msg.shared_mem, ctx.smem_optin()
        )));
    }

    // 2.5 opt-in 通道(2026-10-02 接通,原挂账):动态 smem > 默认顶 48KB
    // 时设 MAX_DYNAMIC_SHARED_SIZE_BYTES(守卫已保证 ≤ 设备 opt-in 上限)。
    // 长 ctx paged v1 的 logits smem 契约(8B/token)在 ctx > 6k 后即越
    // 48KB;属性按函数持久,重复 set 幂等(仅非热路径发射经过此处,
    // 图回放不经)
    if msg.shared_mem as usize > 49152 {
        use cudarc::driver::sys::CUfunction_attribute_enum as Attr;
        let r = func.set_attribute(Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            msg.shared_mem as i32);
        eprintln!("[dbg optin] {} smem={} set_attr={r:?} max_dyn_attr={:?}",
            msg.kernel.name, msg.shared_mem,
            func.get_attribute(Attr::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES));
        r.map_err(|e| ModelError::Msg(format!(
            "launch({}): cudaFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE, {}) 失败: {e:?}",
            msg.kernel.name, msg.shared_mem)))?;
    }

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
