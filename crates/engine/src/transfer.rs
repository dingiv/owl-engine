//! PD 分离(预填充/解码分离)配置类型——从 xinfer `crates/core/src/transfer/mod.rs`
//! 只搬 `PdRole`/`PdMethod`/`PdConfig` 三类型(及其直接依赖),
//! comm/cuda_remote 等传输实现随 T4 transfer 模块整体搬运时补齐。

use serde::{Deserialize, Serialize};

/// Defines the role of the current inference engine instance.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PdRole {
    /// The main instance, handles decoding and orchestrates prefills.
    Client = 1,
    /// A worker instance, dedicated to executing prefills.
    Server = 2,
}

/// The mechanism used to transfer KV cache data.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum PdMethod {
    /// Use CUDA IPC handles for D2D transfer (fastest, local machine only).
    LocalIpc = 1,
    /// Use TCP for remote transfer (inter-machine).
    RemoteTcp = 2,
}

/// Configuration for the Transfer sub-system.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PdConfig {
    /// Is this instance a Client or a PDServer?
    pub role: PdRole,
    /// The chosen transfer method.
    pub method: PdMethod,
    // The network address for the PD Server or client to listen on/connect to (e.g., "0.0.0.0:9000")
    pub url: Option<String>,
}
