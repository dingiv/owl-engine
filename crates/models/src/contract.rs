//! 契约门面:模型层对外部契约词汇的统一出口。
//!
//! **权威在 `owl-iface::contract`**(线格式 + 能力契约,前后端共同依赖);
//! 本模块是 models 内的 re-export 门面(路径稳定面),语义零增量:
//! - 线格式:LaunchMsg / KernelSpec / Arg / Bytes / GraphId;
//! - 能力契约:DeviceClient(server 照此实现,owl-cpu/owl-cuda 各一);
//! - 标注词汇:Dtype / Shape / numel(server 是字节世界,不感知);
//! - 错误:ModelError(执行层资源错,直接 Err 过线)。
//!
//! 描述层毒值 [`crate::tensor::LazyError`] 不在本门面 —— 它是声明树
//! 的元数据,住在 tensor 模块。
//!
//! (2026-09-26 文件重组:吸收原 client.rs / types.rs(重复物)/
//! shape.rs / error.rs 的 re-export 部分,四个碎壳合一。)

pub use owl_iface::contract::{
    Arg, Bytes, DeviceClient, Dtype, GraphId, KernelSpec, LaunchMsg, ModelError, Shape, numel,
};
