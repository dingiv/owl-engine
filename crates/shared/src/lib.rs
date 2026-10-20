//! owl-shared —— 共用基建 crate(零后端依赖;后端契约经 owl-iface)。
//!
//! - [`signal`]:后端无关的依赖追踪(Vue 语义映射;原 owl-signal);
//! - [`testkit`]:单算子对拍测试框架(原 owl-testkit,设计见 README.md);
//! - [`metrics`]:统一性能 metrics 框架(全局 store + timer/counter 宏,
//!   debug 展开 / release 零开销)—— 2026-10-01 自 owl-metrics 独立
//!   crate 收编回本 crate(减 crate 面;宏经 `#[macro_export]` 落本
//!   crate 根,`owl_shared::timer_start!` 调用路径不变)。
//! - [`config`]:**全量配置总账 + 统一 loader**(OwlConfig 十 module
//!   强类型;枚举严格 fail-fast;入口一次加载,消费方 from_config 零
//!   env 直读 —— 显式依赖律终点,2026-10-10);
//! - [`env_reader`]:环境变量读取原语(flag/str/parse + EnvGuard;
//!   config loader 的底层;应用代码勿直用 —— 走 [`config`]);
//! - [`file_loader`]:文件读写统一入口(std::fs 薄包装,未来统一
//!   控制点留口;应用运行时文件 I/O 一律经此;build.rs 例外)。
//! - [`log`]:**logger 框架**(2026-10-14;与 metrics 同风格、不同定位 ——
//!   事件流 vs 数值账;宏风格 log_info!/… + feature `log` 编译期门 + 运行期
//!   级别/tag 过滤 + tee 落盘与 rotation)。
//! - [`stepgate`]:**单步/粘滞调试门**(2026-10-14;与 metrics/log 同纪律、
//!   不同用途 —— 干预而非观察):`step_gate!("tag").await` 检查点宏 +
//!   feature `stepgate` 编译期门(不带 = 空展开零开销)+ 运行期档位
//!   off/sticky/single + tx/rx 外部 step 信号(HTTP `/debug/step*`)。

pub use owl_iface::signal;

pub mod profiles;
pub mod config;
pub mod device;
pub mod env_reader;
pub mod file_loader;
pub mod gpu_lock;
pub mod log;
pub mod metrics;
pub mod stepgate;
pub mod slab_hint;
pub mod vram;
pub mod testkit;
