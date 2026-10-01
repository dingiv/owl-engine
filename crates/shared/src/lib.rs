//! owl-shared —— 共用基建 crate(零后端依赖;后端契约经 owl-iface)。
//!
//! - [`signal`]:后端无关的依赖追踪(Vue 语义映射;原 owl-signal);
//! - [`testkit`]:单算子对拍测试框架(原 owl-testkit,设计见 README.md);
//! - [`metrics`]:统一性能 metrics 框架(全局 store + timer/counter 宏,
//!   debug 展开 / release 零开销)—— 2026-10-01 自 owl-metrics 独立
//!   crate 收编回本 crate(减 crate 面;宏经 `#[macro_export]` 落本
//!   crate 根,`owl_shared::timer_start!` 调用路径不变)。

pub use owl_iface::signal;

pub mod metrics;
pub mod testkit;
