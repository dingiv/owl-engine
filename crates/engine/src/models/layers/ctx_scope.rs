//! 模型层 ctx 作用域(S6 的执行凭证传递)。
//! ⚠️ 临时脚手架:dry-run 专用桥(OwlTensor candle 形态签名的权宜实现,
//! Rig/TLS 栈 = 全局作用域反模式的隔离仓)。T3 接线改"OwlTensor 签名
//! 带 ctx"正解后,本模块整体拆除。
#![allow(dead_code)]
//!
//! OwlTensor 方法签名无 ctx(保持 candle 形态);rig(ops/blas/dry/device)
//! 经进程级 OnceLock 安装(dry-run 单卡单实例;A2.6 多实例 = 每 rig 一线程,
//! TLS 栈本就线程私有),KernelCtx 每次 forward 压 TLS 栈,guard drop 弹出。

use crate::Result;
use owl_iface::MemPhase;
use owl_cuda::CudaDevice;
use owl_nn::cublas::NnBlas;
use owl_nn::{KernelCtx, OpsCtx};
use std::cell::RefCell;
use std::sync::{Arc, Mutex, OnceLock};

/// forward 执行资源集(dry-run 装配;真 runner 接线后同构替换)
pub struct Rig {
    pub ops: Mutex<OpsCtx>,
    pub blas: NnBlas,
    pub dry: Mutex<crate::models::dry_kernels::DryKernels>,
    pub device: CudaDevice,
    /// S6 共享激活池(捕获 ctx 也注入——frame ctx 本身无池)
    pub scratch: Arc<owl_cuda::CudaPool>,
    /// dry-run 装载池(Weights 语义;from_vec/表类持久数据落此)
    pub weights_pool: Arc<owl_cuda::CudaPool>,
}

static RIG: OnceLock<Arc<Rig>> = OnceLock::new();

/// 安装 rig(P 阶段,进程一次)
pub fn install(
    ops: OpsCtx,
    blas: NnBlas,
    dry: crate::models::dry_kernels::DryKernels,
    scratch: Arc<owl_cuda::CudaPool>,
    weights_pool: Arc<owl_cuda::CudaPool>,
    device: &CudaDevice,
) {
    let _ = RIG.set(Arc::new(Rig {
        ops: Mutex::new(ops),
        blas,
        dry: Mutex::new(dry),
        device: device.clone(),
        scratch,
        weights_pool,
    }));
}

fn rig() -> Arc<Rig> {
    RIG.get().expect("ctx_scope: rig 未安装(ctx_scope::install)").clone()
}

thread_local! {
    static STACK: RefCell<Vec<KernelCtx>> = const { RefCell::new(Vec::new()) };
}

/// 压栈 guard(drop 弹栈)
pub struct Guard {
    _priv: (),
}

/// rig 的装载池(Weights 语义;cos/sin 等持久表落此)
pub fn weights_pool() -> Arc<owl_cuda::CudaPool> {
    Arc::clone(&rig().weights_pool)
}

/// 压入 ctx(forward 入口;adapter 与 eager warmup 各自调用)
pub fn push(ctx: &KernelCtx) -> Guard {
    STACK.with(|s| s.borrow_mut().push(ctx.clone()));
    Guard { _priv: () }
}

fn with_stack<R>(f: impl FnOnce(&KernelCtx) -> Result<R>) -> Result<R> {
    let rig = rig();
    // 栈空 = 构造期(P 阶段)调用:自动回退 eager ctx(带 S6 scratch)
    let mut ctx = match STACK.with(|s| s.borrow().last().cloned()) {
        Some(c) => c,
        None => eager_ctx(MemPhase::Live)?,
    };
    // 捕获 frame ctx 不带池:rig 的 S6 共享池无条件注入(幂等)
    ctx = ctx.with_scratch(Arc::clone(&rig.scratch));
    f(&ctx)
}

/// 以栈顶 ctx + rig.ops 执行 f
pub fn with<R>(f: impl FnOnce(&mut OpsCtx, &KernelCtx) -> Result<R>) -> Result<R> {
    let rig = rig();
    with_stack(|ctx| {
        let mut ops = rig.ops.lock().map_err(|_| crate::Error::Msg("rig ops 锁中毒".into()))?;
        f(&mut ops, ctx)
    })
}

/// 带 blas(cublas 链)
pub fn with_blas<R>(
    f: impl FnOnce(&mut OpsCtx, &KernelCtx, &NnBlas) -> Result<R>,
) -> Result<R> {
    let rig = rig();
    with_stack(|ctx| {
        let mut ops = rig.ops.lock().map_err(|_| crate::Error::Msg("rig ops 锁中毒".into()))?;
        f(&mut ops, ctx, &rig.blas)
    })
}

/// 带 dry kernel 模块(naive attention/rope/embed/fill)
pub fn with_dry<R>(
    f: impl FnOnce(&KernelCtx, &mut crate::models::dry_kernels::DryKernels) -> Result<R>,
) -> Result<R> {
    let rig = rig();
    with_stack(|ctx| {
        let mut dry = rig.dry.lock().map_err(|_| crate::Error::Msg("rig dry 锁中毒".into()))?;
        f(ctx, &mut dry)
    })
}

/// 设备句柄(OwlTensor::device 元数据)
pub fn with_device() -> CudaDevice {
    rig().device.clone()
}

/// eager 上下文(rig.ops 构造;warmup/回退路径)
pub fn eager_ctx(phase: owl_iface::MemPhase) -> Result<KernelCtx> {
    let rig = rig();
    let c = {
        let ops = rig.ops.lock().map_err(|_| crate::Error::Msg("rig ops 锁中毒".into()))?;
        ops.ctx(phase)
    };
    // scratch 池注入(S6;ctx 克隆携带)
    Ok(c.with_scratch(Arc::clone(&rig.scratch)))
}

