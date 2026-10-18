//! owl-cuda —— GPU server 后端:实现 `owl-iface::contract::DeviceClient` 契约。
//!
//! 分层:owl-kernels(kernel 描述)→ owl-cuda(actor 执行)→ owl-iface::contract(契约)。
//! server 是**哑执行器**:不认识具体算子,只认 LaunchMsg / alloc / htod / dtoh / sync。
//!
//! 模块地图:
//! - [`ffi`]:cudarc 受控再导出(A4;上层唯一可见的 driver 表面)
//! - [`command`]:命令信封(封闭枚举)+ Ack/Waiter 回执桥(公开;手工组装用)
//! - [`state`]:GpuCtx(三固定流/池块账房/图捕获状态机)+ KernelCache + Staging
//! - [`launch`]:LaunchMsg 装配发射(哑执行器;零算子语义)
//! - [`server`]:GpuServer 事件循环(rx 阻塞监听 + issue 相)+ 完成派发线程
//! - [`gpu_client`]:GpuClient 异步门面(DeviceClient 实现)
//!
//! 组装形态(构造/管道/启动三者分离):
//! ```text
//! let (tx, rx) = mpsc::channel::<Command>();       // 管道:外部创建
//! let server = GpuServer::new(rx, selector, None); // server 构造(纯构造,零 CUDA)
//! let client = GpuClient::new(tx);                 // client 构造(纯句柄)
//! thread::spawn(move || server.run());             // 启动(专用线程)
//! ```
//!
//! 异步模型:GPU API 全走非阻塞形态(launch/memset/异步 memcpy 提交进流即返回);
//! 完成通知用现代回调 API `cuLaunchHostFunc`(host 回调随流序由**驱动主动触发**,
//! 零轮询零阻塞收割)—— 回调跑在驱动线程,只做 channel 投递;真正的 finish
//! (含 free_host 等 CUDA 操作)由 server 内部的派发线程执行,结果经 Ack
//! 回执客户端,以此构成对外的异步 API(命令流水线化;server 不因 GPU 工作而停摆)。

mod command;
mod gpu_client;
mod launch;
mod server;
mod state;

pub mod ffi;
pub mod trace;

/// CUPTI 进程内归因(仅 feature `cupti`;见 perf-framework §十一)
#[cfg(feature = "cupti")]
pub mod cupti;

pub use command::Command;
pub use gpu_client::GpuClient;
pub use state::DiagOpts;
pub use server::GpuServer;
pub use state::DeviceSelector;

// 线格式与词汇再导出(契约权威 = owl-iface::contract,前后端共同依赖;
// 2026-09-25 解除 owl-cuda → owl-models 反向依赖,客户只依赖 owl-cuda
// 即可组装命令,不必直连任何前端 crate)
pub use owl_iface::contract::{
    Arg, BlockLedgerRow, Bytes, ConnSlot, Connector, DeviceClient, GraphId, KernelSource, LaunchMsg,
    ModelError, SwapDir, SwapPiece, UpChunk,
};
pub use owl_iface::contract::{Dtype, Shape};

/// 测试设备门控值(`OWL_TEST_DEVICE`;**唯一形态 = BDF**,如 `41:00.0`)。
/// 未设/空串 = None(GPU 用例据此 skip)。
pub fn test_device_bdf() -> Option<String> {
    owl_shared::device::test_device_bdf()
}

/// **BDF → 本进程 CUDA ordinal**(枚举设备表 + `DeviceSpec` 同一套归一匹配)。
///
/// 显式拒绝纯数字(旧的 CUDA 序号写法);找不到 = Err 带**可用 BDF 表**。
pub fn resolve_ordinal_by_bdf(bdf: &str) -> Result<usize, String> {
    if bdf.chars().all(|c| c.is_ascii_digit()) {
        return Err(format!(
            "OWL_TEST_DEVICE={bdf:?} 是纯数字 —— **CUDA 序号已禁用**,请用 BDF(如 `41:00.0`)"
        ));
    }
    let spec = owl_shared::device::DeviceSpec { bdf: Some(bdf.to_string()), ..Default::default() };
    let rows = crate::ffi::device_rows()?;
    let hit = spec.select(&rows);
    match hit.first() {
        Some(r) => Ok(r.ordinal),
        None => {
            let table: Vec<String> = rows
                .iter()
                .map(|r| format!("{}[ordinal={} {}]", r.bdf, r.ordinal, r.name))
                .collect();
            Err(format!(
                "OWL_TEST_DEVICE={bdf:?} 未命中任何设备 —— 可用:{table:?}"
            ))
        }
    }
}

/// 同 [`test_device_ordinal`] 的**可失败形态**:未设 = `None`(调用点据此 skip);
/// 设了但不是合法 BDF / 未命中 = **panic**(不静默当未设)。
pub fn test_device_ordinal_opt() -> Option<usize> {
    test_device_bdf().map(|b| {
        resolve_ordinal_by_bdf(&b).unwrap_or_else(|e| panic!("[owl-cuda] {e}"))
    })
}

/// 测试/示例的设备序(**测试域唯一入口**):`OWL_TEST_DEVICE`(BDF)→ ordinal。
/// 未设/非法 = **panic**(调用点必须先过 GPU 门;测试面拒绝隐式 CUDA 序号)。
pub fn test_device_ordinal() -> usize {
    match test_device_bdf() {
        Some(bdf) => {
            resolve_ordinal_by_bdf(&bdf).unwrap_or_else(|e| panic!("[owl-cuda] {e}"))
        }
        None => panic!(
            "[owl-cuda] OWL_TEST_DEVICE 未设:测试钉卡唯一形态 = BDF(如 `41:00.0`);\
             用例应先过 GPU 门(未设 = skip)"
        ),
    }
}
