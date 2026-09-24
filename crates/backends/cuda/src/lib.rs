//! owl-cuda —— cudarc **官方版**(crates.io)的治理封装层,实现
//! `owl_iface` 的 `Backend`/`Device`/`Pool` 三层契约(charter A1/A4/A5)。
//!
//! 选型注记(2026-09-22):官方 cudarc 0.19 已原生覆盖 fork 的图安全增量
//! (CudaGraph/stream-ordered 分配/13.x 绑定),按 A4"语义分歧不进第三方
//! 库"选官方。cudarc 仅在本 crate 导入;上层经 `ffi` 模块的精确白名单
//! 使用(整库 re-export 禁止)。
//!
//! 分配架构(裁决 5 + A5.2):
//! - **Pool 是分配者**:`CudaPool::malloc_*` 是 P 阶段唯一分配入口,
//!   校验链 = kind 语义 → 池余量(A5.4)→ 全局预算(A5.4)→ 物理分配;
//! - 物理路径按池类型路由:`PeerShared` → VMM(cuMemCreate,2MiB 粒度,
//!   A2.8);其余 → stream-ordered(捕获安全);
//! - 所有缓冲 drop 时自动归池账 + 全局账(非 Idle 相延迟到净空窗口,
//!   A1.2);BufToken/世代校验 = 哨兵①(结构化报错替代 Xid 盲死);
//! - Device 的 `alloc_*_in` 只是 `pool.malloc_*` 的类型化薄封装。

pub mod ffi;

