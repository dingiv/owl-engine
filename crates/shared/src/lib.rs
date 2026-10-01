//! owl-shared —— 共用基建 crate。
//!
//! - [`signal`]:后端无关的依赖追踪(Vue 语义映射;原 owl-signal);
//! - [`testkit`]:单算子对拍测试框架(原 owl-testkit,设计见 README.md);
//! - [`metrics`]:统一性能 metrics 框架(全局 store + timer/counter 宏,
//!   debug 展开 / release 零开销)—— 实体住 owl-metrics(零依赖横切
//!   crate,models/engine 直依赖);本 crate 仅 re-export 保持调用路径。

pub use owl_iface::signal;

pub use owl_metrics as metrics;
pub mod testkit;

pub use owl_metrics::{
    counter_add, counter_inc, event, timer_end, timer_record, timer_scope, timer_start,
};
