//! owl-cpu —— CPU 后端:`DeviceClient` 契约的**进程内同步实现**。
//!
//! 与 owl-cuda 的形态对照(同一契约,两种执行体):
//!
//! ```text
//! owl-cuda  客户端 ──mpsc 管道──► GPU actor 线程(绑卡/三流/账房)──► 硬件
//! owl-cpu   客户端 ──函数调用──►  直接执行(朴素算子,Vec<f32>)    ──► CPU
//! ```
//!
//! **无 server**:CPU 没有异构执行域,launch/memcpy 本身就是同步完成的,
//! 不需要 actor/管道/回执桥 —— `submit` 即执行,回执即返回。
//! 这就是"后端是否需要 server"的分界:server 的存在理由是跨执行域
//! 序列化设备访问,CPU 后端没有这个跨域问题。
//!
//! 模块地图:
//! - [`value`]:f32 值块(CPU 后端的数据平面;host 真值)
//! - [`ops`]:朴素算子实现(matmul/add/silu/rmsnorm;对拍锚的独立副本)
//! - [`face`]:CpuFace(DeviceClient 实现;客户端直接持有的后端句柄)
//!
//! 纪律:本 crate 只依赖 owl-iface(契约),**不依赖任何前端 crate**
//! (2026-09-25 后端不依赖前端裁决;与 owl-cuda 同律)。

mod face;
mod ops;
mod value;

pub use face::CpuFace;
pub use value::Value;
