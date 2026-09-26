//! owl-engine —— 编排/内存/图治理层(M0 骨架,2026-09-26 立;旧 engine
//! 移至 engine-bak 存档,25k 行产品化残留不复用为主干,复用审计见下)。
//!
//! 唯一设备标准 = `owl_iface::contract::DeviceClient`(线格式/回执分级/
//! 顺序语义);模型 = owl-models 声明层(层/Model 纯描述 + 解释器执行)。
//! engine 的存在理由:把「槽管理 + 步循环 + 图三态 + 内存预算」从每个
//! 上层调用者手里收编为**治理对象**(charter A1 / session-plan.md)。
//!
//! # 模块地图(M0)
//!
//! - [`session`]:Session 编排面(P0 **eager 闭环**:槽装填 + 闭包纯声明
//!   + 计算解释器执行;捕获/回放三态 M1 接线)
//!
//! # 复用审计(engine-bak → 本 crate)
//!
//! | 老模块 | 处置 |
//! |---|---|
//! | `session.rs`(464 行流程纪律) | ✅ **移植改写**:warmup 门禁/槽装填/eager 直发保留;ETensor/KernelCtx/CudaPool → TensorOps/DeviceClient |
//! | graphplan 捕获循环(姿势 6/Governor/租约/哨兵③) | ✅ M1 移植(经 DeviceClient graph_begin/end/launch 泛化) |
//! | `kvcache/allocator.rs` + `block_manager` | 参考重写 M1(页表/槽位治理;数据面已换 KV 直排) |
//! | `runner` / `scheduler` / `sequence` | 挂账 M2(每卡一线程 A2.6 + continuous batching) |
//! | `sampler` | M1(host 贪心已在 models::interpreters::generate;设备采样随 Argmax kernel) |
//! | `loader`(gguf/safetensors/load_cache) | ❌ 废弃(owl-models loader + specs 流式装载取代) |
//! | `models/`(candle 旧世界) | ❌ 废弃(owl-models 取代;移植母本价值已录 qwen3-mini-demo §七) |
//! | `server` | ❌ 废弃(backends/cuda GpuClient actor 取代) |
//! | `downloader`/`image`/`multi_node`/`transfer`/`distributed`/`env` | ❌ 产品化残留废弃(需要时按 charter 重立) |
//!
//! 里程碑对齐 charter:M0 骨架 → M1 图治理最小闭环 → M2 TP2 comm →
//! M3 模型 + DFlash2 对齐线 → M4 冲刺线。

pub mod engine;
pub mod session;
pub mod turn;

pub use engine::{Engine, EngineConfig, LoadedModel, ModelLoader, RunningEngine};
pub use turn::{TurnEvent, TurnSpec};

/// 本 crate 的结果别名:错误权威 = iface `ModelError`(线上一族,不另立)
pub type Result<T> = std::result::Result<T, owl_iface::contract::ModelError>;
