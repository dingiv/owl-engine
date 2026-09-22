//! KV cache 后端类型 + 分配器(T2-三 搬运)。
//!
//! port 出处:packages/xinfer/crates/core/src/utils/{kv_backend.rs,
//! kvcache_allocator.rs}(只读参照源)。
//!
//! 翻译要点:
//! - candle `Tensor`(擦除形态)→ `owl_nn::DynTensor<CudaDevice>`(GPU 侧);
//! - CPU swap 镜像 → host `Vec<u8>` 零拷贝段(装载时上设备,T3 传输层接线);
//! - engine 是 cuda-only 基座:xinfer 的 `#[cfg(not(feature = "cuda"))]`
//!   sysinfo/metal 分支不搬运(编译面直接消失,非运行期分支);
//! - DeepSeek V4 混合页池 = T3 占位(类型层保留,语义回填)。

pub mod allocator;

use owl_cuda::CudaDevice;
use owl_nn::DynTensor;
use parking_lot::Mutex;
use std::sync::Arc;

/// 引擎分配的物理 KV 布局种类。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheBackend {
    /// 标准 paged `(K, V)`(FlashInfer / FlashAttention)。
    Flash,
    /// 经典 MLA `(ckv, kpe)`(DeepSeek V3 / GLM MoE DSA 等)。
    Mla,
    /// DeepSeek V4 混合页(SWA + 压缩 + 残差 + indexer)。
    DeepSeekV4,
}

impl KvCacheBackend {
    pub fn is_mla(self) -> bool {
        matches!(self, Self::Mla)
    }

    pub fn is_deepseek_v4(self) -> bool {
        matches!(self, Self::DeepSeekV4)
    }

    pub fn is_flash(self) -> bool {
        matches!(self, Self::Flash)
    }
}

/// T3 占位:DeepSeek V4 混合页池(ds_v4 层搬运期回填真实布局)。
#[allow(dead_code)] // T3 ds_v4 搬运期回填
pub struct V4HybridPagePool {
    /// 参与的层 id 列表(类型层先占位)
    pub layers: Vec<usize>,
}

/// T3 占位:V4 层缓存规格(ds_v4 搬运期回填)。
#[allow(dead_code)] // T3 ds_v4 搬运期回填
#[derive(Debug, Clone)]
pub struct V4LayerCacheSpec;

/// 图捕获批次规划(T3 占位):runner 的 GraphCapturer 改驱动 owl
/// CaptureSession 时回填(vLLM 语义:1..=max_num_seqs 的精确批次表,
/// 截断到捕获上限)。
pub fn planned_graph_capture_batches(max_num_seqs: usize) -> Vec<usize> {
    // 编译先行口径:返回 1..=max_num_seqs(语义待 T3 定稿)
    (1..=max_num_seqs.max(1)).collect()
}

/// GPU 常驻 KV 张量(引擎持有物理缓冲;模型消费匹配变体)。
#[allow(clippy::large_enum_variant)] // T3 运行期按模型类型实际只活一个变体
pub enum GpuKvCache {
    /// Flash / MLA:每层 (K,V) 或 (ckv,kpe) 对
    Flash(Vec<(DynTensor<CudaDevice>, DynTensor<CudaDevice>)>),
    Mla(Vec<(DynTensor<CudaDevice>, DynTensor<CudaDevice>)>),
    /// DeepSeek V4:层与引擎共享同一个池 Arc
    DeepSeekV4(Arc<Mutex<Option<V4HybridPagePool>>>),
}

impl GpuKvCache {
    pub fn backend(&self) -> KvCacheBackend {
        match self {
            Self::Flash(_) => KvCacheBackend::Flash,
            Self::Mla(_) => KvCacheBackend::Mla,
            Self::DeepSeekV4(_) => KvCacheBackend::DeepSeekV4,
        }
    }

    /// 层 `(K,V)` 或 `(ckv,kpe)` 对 —— 仅 Flash / MLA
    pub fn as_pairs(&self) -> Option<&Vec<(DynTensor<CudaDevice>, DynTensor<CudaDevice>)>> {
        match self {
            Self::Flash(v) | Self::Mla(v) => Some(v),
            Self::DeepSeekV4(_) => None,
        }
    }

    pub fn as_pairs_mut(
        &mut self,
    ) -> Option<&mut Vec<(DynTensor<CudaDevice>, DynTensor<CudaDevice>)>> {
        match self {
            Self::Flash(v) | Self::Mla(v) => Some(v),
            Self::DeepSeekV4(_) => None,
        }
    }

    pub fn as_v4_pool(&self) -> Option<&Arc<Mutex<Option<V4HybridPagePool>>>> {
        match self {
            Self::DeepSeekV4(p) => Some(p),
            _ => None,
        }
    }

    /// 表达的注意力层数(对数,或 V4 层数)
    pub fn num_layers(&self) -> usize {
        match self {
            Self::Flash(v) | Self::Mla(v) => v.len(),
            Self::DeepSeekV4(p) => p.lock().as_ref().map(|pool| pool.layers.len()).unwrap_or(0),
        }
    }
}

/// CPU swap 镜像(Flash / MLA)。host `Vec<u8>` 零拷贝段:
/// 尺寸 = shape 元素数 × dtype 字节宽,装载/交换时经传输层上设备
/// (T3 接线;V4 的 CPU swap 延后)。
#[allow(clippy::type_complexity)]
pub enum CpuKvCache {
    Flash(Vec<(Vec<u8>, Vec<u8>)>),
    Mla(Vec<(Vec<u8>, Vec<u8>)>),
    DeepSeekV4,
}

impl CpuKvCache {
    pub fn as_pairs(&self) -> Option<&Vec<(Vec<u8>, Vec<u8>)>> {
        match self {
            Self::Flash(v) | Self::Mla(v) => Some(v),
            Self::DeepSeekV4 => None,
        }
    }

    pub fn as_pairs_mut(&mut self) -> Option<&mut Vec<(Vec<u8>, Vec<u8>)>> {
        match self {
            Self::Flash(v) | Self::Mla(v) => Some(v),
            Self::DeepSeekV4 => None,
        }
    }
}

/// CPU TurboQuant swap 层缓存(T3 占位:vendor TurboQuant 归宿裁决待定)。
#[allow(dead_code)] // T3 运行里程碑回填(vendor 归宿)
pub struct CpuTqLayerCache {
    pub k_absmax: Option<Vec<u8>>,
    pub k_quant: Option<Vec<u8>>,
    pub v_absmax: Option<Vec<u8>>,
    pub v_quant: Option<Vec<u8>>,
}

/// 分配器导出(对齐 xinfer `crate::utils` 出口面:runner/scheduler
/// 经 `crate::kvcache::...` 引用)
pub use allocator::{GpuMemoryBudget, KVCacheAllocation, KVCacheAllocator, KVCacheError};
