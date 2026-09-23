//! owl-shared —— 共用基建 crate。
//!
//! - [`signal`]:后端无关的依赖追踪(Vue 语义映射;原 owl-signal);
//! - [`testkit`]:单算子对拍测试框架(原 owl-testkit,设计见 README.md)。

pub use owl_iface::signal;

pub mod testkit;
