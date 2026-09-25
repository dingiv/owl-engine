//! Client:对 GPU server 的**能力期望** + 每步动态上下文。
//!
//! 职责定稿(2026-09-23 四层架构;2026-09-25 契约下沉 + 解释器归位):
//! - server(硬件层)= 哑执行器:只认 LaunchMsg / Alloc / Htod / Dtoh / Sync;
//! - 解释器 = 把声明树翻译成原语消息(见 [`crate::interpreter`]:
//!   eval 异步解释器,face 注入 CpuFace/GpuClient);
//! - 上层(TensorOps/Module)= 纯声明,零设备知识。
//!
//! **线格式与 DeviceClient 契约的权威在 `owl-iface::contract`**
//! (前后端共同依赖;本模块 re-export 保持 `crate::client::` 路径稳定),
//! 此处只保留模型层语义:`KvCtx`。

pub use owl_iface::contract::{
    Arg, Bytes, DeviceClient, GraphId, KernelSpec, LaunchMsg,
};

/// KV 动态上下文:每步由 runner 构造。
#[derive(Clone, Debug)]
pub struct KvCtx {
    pub step: u64,
    pub slots: Vec<u32>,
}
