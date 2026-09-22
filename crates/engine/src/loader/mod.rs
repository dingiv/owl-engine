//! 模型权重装载器(T2-三,主 agent 设计 + 搬运)。
//!
//! ## 架构:装载与分配解耦(用户裁决)
//!
//! Loader 本体只做两件事:**读文件/解量化 → 得到 host 原始字节**,
//! 然后把字节交给 [`WeightAllocator`]。实例化 loader 时注入 allocator:
//!
//! - [`DeviceWeightAllocator`]:指向显存(owl Weights 池,账本化,
//!   裁决 5 分配入口在 Pool);
//! - [`HostWeightAllocator`]:指向内存(host `Vec<u8>`),**测试用**——
//!   装载全链(解析/分片/转置/路径拼接)可在无卡环境自测。
//!
//! 禁止 loader 内部绕过 allocator 直接触碰 CUDA(A4);GGUF 量化权重
//! 的反解在 host 侧完成后过 allocator,设备上不做量化算术(S5)。

pub mod alloc;
pub mod gguf;
pub mod gguf_helper;
pub mod load_cache;
pub mod load_stats;

pub use alloc::{DeviceWeightAllocator, HostWeightAllocator, HostTensor, WeightAllocator};
