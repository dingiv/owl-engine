//! **内存 / 存储域**(三层:GPU 池 → host pinned → 磁盘)。
//!
//! - `state`:`StatePool` —— 引擎全部设备侧状态块的治理对象(唯一块句柄出口);
//! - `store_hook`:Store 三钩子契约(Acquire / Release / Maintain)的**请求与回执**
//!   (执行臂在 `owl-store`,不在本 crate);
//! - `disk_io`:磁盘层的**执行面**(S6):store 出计划,这里做真 I/O。
//!
//! 预算/布局纪律见 `docs/devlog/memory/store.md` 与 charter A5/A6。

pub(super) mod disk_io;
pub(super) mod state;
pub(super) mod store_hook;
