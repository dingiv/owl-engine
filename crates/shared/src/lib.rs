//! owl-shared —— 共用基建 crate(零后端依赖;后端契约经 owl-iface)。
//!
//! - [`signal`]:后端无关的依赖追踪(Vue 语义映射;原 owl-signal);
//! - [`testkit`]:单算子对拍测试框架(原 owl-testkit,设计见 README.md);
//! - [`metrics`]:统一性能 metrics 框架(全局 store + timer/counter 宏,
//!   debug 展开 / release 零开销)—— 2026-10-01 自 owl-metrics 独立
//!   crate 收编回本 crate(减 crate 面;宏经 `#[macro_export]` 落本
//!   crate 根,`owl_shared::timer_start!` 调用路径不变)。
//! - [`env_reader`]:环境变量统一读取面(全 workspace 唯一 std::env
//!   访问口;解析失败降级可见 + EnvGuard 测试防踩踏;build.rs 例外);
//! - [`file_loader`]:文件读写统一入口(std::fs 薄包装,未来统一
//!   控制点留口;应用运行时文件 I/O 一律经此;build.rs 例外)。

pub use owl_iface::signal;

pub mod env_reader;
pub mod file_loader;
pub mod metrics;
pub mod slab_hint;
pub mod vram;
pub mod testkit;
