//! 标注词汇:dtype + shape。client 侧元数据;server(字节世界)不感知。
//!
//! 词汇权威:2026-09-25 下沉 `owl-iface::contract`(线格式的元数据维,
//! 前后端共同依赖);本模块为 re-export 壳(保持 `crate::shape::` 路径稳定)。

pub use owl_iface::contract::{numel, Dtype, Shape};
