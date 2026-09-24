//! KVCache 分配器(T2-三 搬运)。
//!
//! port 出处:packages/xinfer/crates/core/src/utils/kvcache_allocator.rs
//! (1973 行;只读参照源)。
//!
//! 翻译规则(见 candle-api-mapping.md / xinfer-transplant.md §四):
//! - `Tensor::empty/zeros(shape, dtype, &device)` → [`alloc_gpu_buffer`]
//!   (owl 池直连工厂,裁决 5:分配入口在 Pool;六 dtype 分派,无 unsafe);
//! - `Tensor::zeros(.., &Device::Cpu)` → host `Vec<u8>` 零拷贝段
//!   (装载/交换时上设备,T3 传输层接线);
//! - cudarc 直触段(get_rank_available/free)→ `owl_cuda::CudaDevice`
//!   + `mem_get_info`(A4:FFI 只在 owl-cuda);
//! - engine 是 cuda-only 基座:xinfer 非 CUDA 平台分支(sysinfo/metal)
//!   不搬运;`cfg!(feature = "cutlass"/"flash"/...)` 保留原样
//!   (engine 未声明这些 feature = 恒 false = 行为与 xinfer 默认构建一致);
//! - TurboQuant(vendor attention_rs)= T3 占位:vendor 归宿裁决待定,
//!   函数签名保留,体 `unimplemented!()`。

use crate::config::{Config, EngineConfig, KvCacheDtype};
use crate::error::Result;
use crate::hybrid::{
    gemma4_per_layer_cache_config, is_deepseek_v4_arch_name, qwen3_hybrid_layer_types,
    resolve_qwen3_hybrid_config,
};
use crate::kvcache::{
    CpuKvCache, CpuTqLayerCache, GpuKvCache, KvCacheBackend, V4HybridPagePool, V4LayerCacheSpec,
    planned_graph_capture_batches,
};
use crate::transfer::PdRole;
use owl_cuda::CudaDevice;
use owl_nn::{DynTensor, Dtype, TensorPoolOps};
use std::fmt;

/// 保留内存常量——分配后警告用
const CUDA_RESERVED_BYTES: u64 = 512 * 1024 * 1024; // 512 MB 建议最低剩余
/// 模型感知计算得出极小值时的激活最低预留。
const MIN_ACTIVATION_RESERVE_BYTES: u64 = 256 * 1024 * 1024; // 256 MB 下限
const SIZE_IN_MB: f64 = (1024 * 1024) as f64;
const SIZE_IN_GB: f64 = 1024.0 * 1024.0 * 1024.0;
const DEFAULT_HYBRID_MAMBA_FRACTION: f64 = 0.20;
const MAX_HYBRID_MAMBA_FRACTION: f64 = 0.35;
/// CUDA 图捕获按 1..=32 每个精确批次规划;混合 GDN/Mamba 状态容量与之对齐。
pub(crate) const HYBRID_MAMBA_GRAPH_CAPTURE_MAX_BATCH: usize = 32;

pub(crate) fn hybrid_mamba_graph_capture_max_batch(max_num_seqs: usize) -> usize {
    max_num_seqs.max(1).min(HYBRID_MAMBA_GRAPH_CAPTURE_MAX_BATCH)
}

/// 池直连分配:按 dtype 六型分派到 owl 池工厂,擦除为 DynTensor。
///
/// 2026-09-24 池随设备出生裁决:KV 缓冲走设备默认池(容量闸由设备级
/// A5 预算承担);块租约语义不变。
fn alloc_gpu_buffer(device: &CudaDevice, shape: &[usize], dtype: Dtype) -> Result<DynTensor<CudaDevice>> {
    let _n: usize = shape.iter().product();
    match dtype {
        Dtype::F32 => {
            let pool = device.default_pool();
            let t = pool.zeros_tensor::<f32>(shape)?;
            Ok(DynTensor::from_f32(&t))
        }
        Dtype::F16 => {
            let pool = device.default_pool();
            let t = pool.zeros_tensor::<owl_nn::F16>(shape)?;
            Ok(DynTensor::from_f16(&t))
        }
        Dtype::BF16 => {
            let pool = device.default_pool();
            let t = pool.zeros_tensor::<owl_nn::Bf16>(shape)?;
            Ok(DynTensor::from_bf16(&t))
        }
        Dtype::U8 => {
            let pool = device.default_pool();
            let t = pool.zeros_tensor::<u8>(shape)?;
            Ok(DynTensor::from_u8(&t))
        }
        Dtype::U32 => {
            let pool = device.default_pool();
            let t = pool.zeros_tensor::<u32>(shape)?;
            Ok(DynTensor::from_u32(&t))
        }
        other => {
        crate::bail!("alloc_gpu_buffer: KV 缓冲 dtype {} 无池工厂路径(编译口径)", other)
        }
    }
}

/// host 零拷贝段:元素数 × 字节宽,全零(装载时上设备,T3 接线)
fn host_zeros(shape: &[usize], dtype: Dtype) -> Vec<u8> {
    let n: usize = shape.iter().product();
    vec![0u8; n * dtype.size_bytes()]
}

/// 按模型配置确定性地计算分类 GPU 内存预算。
#[derive(Debug, Clone)]
pub struct GpuMemoryBudget {
    /// FlashInfer float+int 工作区(固定;禁用为 0)
    pub flashinfer_bytes: u64,
    /// CUTLASS 工作区(固定;禁用为 0)
    pub cutlass_bytes: u64,
    /// MoE 激活池峰值估算
    pub moe_pool_bytes: u64,
    /// 逐层 Flash split-K 工作区合计
    pub flash_splitk_bytes: u64,
    /// 瞬态激活开销(最大 forward 中间量)
    pub transient_bytes: u64,
    /// 每 token 保留的 MLA prefill 张量(查询吸收与注意力输出)
    pub mla_attention_bytes_per_token: u64,
    /// 工作区保留合计(以上之和)
    pub total_bytes: u64,
}

impl GpuMemoryBudget {
    pub fn report(&self, min_available_before: u64) {
        let mut parts = Vec::new();
        if self.flashinfer_bytes > 0 {
            parts.push(format!(
                "FlashInfer {:.0}M",
                self.flashinfer_bytes as f64 / SIZE_IN_MB
            ));
        }
        if self.cutlass_bytes > 0 {
            parts.push(format!(
                "CUTLASS {:.0}M",
                self.cutlass_bytes as f64 / SIZE_IN_MB
            ));
        }
        if self.moe_pool_bytes > 0 {
            parts.push(format!(
                "MoE pool {:.0}M",
                self.moe_pool_bytes as f64 / SIZE_IN_MB
            ));
        }
        if self.flash_splitk_bytes > 0 {
            parts.push(format!(
                "SplitK {:.0}M",
                self.flash_splitk_bytes as f64 / SIZE_IN_MB
            ));
        }
        parts.push(format!(
            "Transient {:.0}M",
            self.transient_bytes as f64 / SIZE_IN_MB
        ));
        if self.mla_attention_bytes_per_token > 0 {
            parts.push(format!(
                "MLA prefill {:.1}K/token",
                self.mla_attention_bytes_per_token as f64 / 1024.0
            ));
        }
        crate::log_warn!(
            "GPU Memory Budget: {:.2} GB available → {:.2} GB workspace reserve ({}) → {:.2} GB for caches",
            min_available_before as f64 / SIZE_IN_GB,
            self.total_bytes as f64 / SIZE_IN_GB,
            parts.join(" + "),
            (min_available_before.saturating_sub(self.total_bytes)) as f64 / SIZE_IN_GB
        );
    }
}

/// KVCache 分配规划结果
#[derive(Debug, Clone)]
pub struct KVCacheAllocation {
    /// KVCache 的 GPU block 数
    pub num_gpu_blocks: usize,
    /// KVCache swap 的 CPU block 数
    pub num_cpu_blocks: usize,
    /// 最大并发序列数
    pub max_num_seqs: usize,
    /// 最大模型上下文长度
    pub max_model_len: usize,
    /// 分配给 KVCache 的 GPU 内存总字节
    pub kvcache_memory_bytes: usize,
    /// 每步调度器 token 预算(vLLM `max_num_batched_tokens`)
    pub max_num_batched_tokens: usize,
    /// KV 缓存总 token 容量(`num_gpu_blocks * block_size`)
    pub max_kv_cache_tokens: usize,
}

/// KVCache 分配错误类型
#[derive(Debug, Clone)]
pub enum KVCacheError {
    /// GPU 内存不足以分配 KVCache
    InsufficientGpuMemory {
        available_mb: f64,
        required_mb: f64,
        reserved_mb: f64,
    },
    /// 无效配置参数
    InvalidConfiguration { message: String },
    /// 平台错误(如 CUDA 不可用)
    PlatformError { message: String },
}

impl fmt::Display for KVCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KVCacheError::InsufficientGpuMemory {
                available_mb,
                required_mb,
                reserved_mb,
            } => {
                write!(
                    f,
                    "Insufficient GPU memory for KVCache allocation.\n\
                     Available: {:.2} MB, Required: {:.2} MB, Reserved: {:.2} MB.\n\
                     Tips: Try reducing --max-model-len or --max-num-seqs, \
                     or free GPU resources.",
                    available_mb, required_mb, reserved_mb
                )
            }
            KVCacheError::InvalidConfiguration { message } => {
                write!(f, "Invalid KVCache configuration: {}", message)
            }
            KVCacheError::PlatformError { message } => {
                write!(f, "Platform error: {}", message)
            }
        }
    }
}

impl std::error::Error for KVCacheError {}

/// 平台感知的 KVCache 内存规划主分配器(cuda-only 基座)
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct KVCacheAllocator {
    // 模型参数
    num_hidden_layers: usize,
    /// 需要 KV cache 的层数(不含 GDN/Mamba 层)
    num_kv_layers: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_shards: usize,
    block_size: usize,
    // 用户约束(None = 自动决定)
    user_max_model_len: Option<usize>,
    user_max_num_seqs: Option<usize>,
    config_model_len: usize,
    kv_fraction: f64,
    kvcache_dtype: KvCacheDtype,
    cpu_mem_fold: f32,
    dtype_size: usize,
    model_dtype_size: usize,
    hybrid_mamba_slot_bytes: Option<usize>,
    hybrid_num_gdn_layers: usize,
    is_mla: bool,
    mla_kv_lora_rank: usize,
    mla_qk_rope_head_dim: usize,
    /// 一等 KV 后端(Flash / Mla / DeepSeekV4)
    kv_backend: KvCacheBackend,
    /// DeepSeek V4:预算混合页(SWA+压缩+残差)
    is_deepseek_v4: bool,
    /// 跨全部 V4 层的每原生页字节(非 V4 为 0)
    v4_bytes_per_native_page: usize,
    /// 构造 [`V4HybridPagePool`] 的层规格(仅 V4;T3 占位)
    v4_cache_specs: Option<Vec<V4LayerCacheSpec>>,
    /// 逐层 KV cache 配置:每个 KV 层的 (num_kv_heads, head_dim)
    /// 设置后覆盖统一的 num_kv_heads/head_dim 用于 cache 分配。
    per_layer_cache_config: Option<Vec<(usize, usize)>>,
    // MoE 配置(工作区预算计算用)
    num_attention_heads: usize,
    hidden_size: usize,
    moe_intermediate_size: usize,
    moe_num_experts_per_tok: usize,
    is_moe: bool,
    prefill_chunk_size: usize,
    /// FlashInfer GPU 工作区是否实际在运行期初始化
    use_flashinfer_workspace: bool,
    /// 每 rank draft 权重字节(保守上界 = safetensors 总尺寸)。
    /// draft 在池规划后才载入,规划期每次自由内存读取必须预留这些
    /// 字节(p5-Q2 §6.2 定谳)。
    draft_weight_bytes: u64,
}

/// 递归统计 `path` 下所有 `*.safetensors` 尺寸之和(每 rank 保守上界:
/// draft 跨 rank 大体复制,TP 只分片 fc 投影;p5-Q2 §6.2)
fn estimate_draft_weight_bytes(path: Option<&str>) -> u64 {
    let Some(path) = path else { return 0 };
    fn walk(dir: &std::path::Path, out: &mut u64, depth: usize) {
        if depth > 4 {
            return;
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() {
                    walk(&p, out, depth + 1);
                } else if p.extension().is_some_and(|x| x == "safetensors") {
                    if let Ok(meta) = entry.metadata() {
                        *out = out.saturating_add(meta.len());
                    }
                }
            }
        }
    }
    let mut total = 0u64;
    walk(std::path::Path::new(path), &mut total, 0);
    total
}

/// 对应 `core/runner.rs` 的 `skip_flashinfer_init`。
/// engine 无 flashinfer feature → 恒 false(与 xinfer 默认构建一致)
fn uses_flashinfer_workspace(econfig: &EngineConfig, config: &Config) -> bool {
    let _ = (econfig, config);
    false
}

impl KVCacheAllocator {
    fn kv_heads_per_shard_for(&self, num_kv_heads: usize) -> usize {
        if self.num_shards == 0 {
            return 1;
        }
        if num_kv_heads >= self.num_shards {
            num_kv_heads / self.num_shards
        } else {
            1
        }
    }

    fn kv_heads_per_shard(&self) -> usize {
        self.kv_heads_per_shard_for(self.num_kv_heads)
    }

    fn layer_kv_config(&self, layer_idx: usize) -> (usize, usize) {
        self.per_layer_cache_config
            .as_ref()
            .and_then(|configs| configs.get(layer_idx).copied())
            .unwrap_or((self.num_kv_heads, self.head_dim))
    }

    fn layer_flash_key_value_block_shape(&self, layer_idx: usize) -> (usize, usize, usize) {
        let (num_kv_heads, head_dim) = self.layer_kv_config(layer_idx);
        (
            self.block_size,
            self.kv_heads_per_shard_for(num_kv_heads),
            head_dim,
        )
    }

    fn layer_key_block_shape(
        &self,
        layer_idx: usize,
        cache_dtype: Dtype,
    ) -> (usize, usize, usize, usize) {
        let (_, kv_heads, head_dim) = self.layer_flash_key_value_block_shape(layer_idx);
        let element_size = cache_dtype.size_bytes();
        let x = 16 / element_size;
        (kv_heads, head_dim / x, self.block_size, x)
    }

    fn layer_value_block_shape(&self, layer_idx: usize) -> (usize, usize, usize) {
        let (_, kv_heads, head_dim) = self.layer_flash_key_value_block_shape(layer_idx);
        (kv_heads, head_dim, self.block_size)
    }

    /// 从引擎与模型配置创建 KVCacheAllocator
    pub fn new(econfig: &EngineConfig, config: &Config, dtype: Dtype) -> Self {
        let configured_num_shards = econfig.num_shards.unwrap_or(1);
        if configured_num_shards == 0 {
            crate::log_warn!(
                "EngineConfig.num_shards=0 is invalid; defaulting to 1 for KVCache allocation"
            );
        }
        let num_shards = configured_num_shards.max(1);
        let head_dim = config
            .head_dim
            .unwrap_or(config.hidden_size / config.num_attention_heads);

        let fp8_kvcache = econfig.kvcache_dtype.is_fp8_keys();
        let dtype_size = if fp8_kvcache { 1 } else { dtype.size_bytes() };
        let model_dtype_size = dtype.size_bytes();

        let kv_fraction = econfig
            .kv_fraction
            .unwrap_or(if cfg!(feature = "flashattn") {
                0.7
            } else {
                if cfg!(feature = "cuda") {
                    0.6
                } else {
                    0.4
                }
            }) as f64;

        let config_model_len = econfig
            .config_model_len
            .unwrap_or(config.max_position_embeddings);

        // 混合模型(如 Qwen3.5)只计全注意力层
        let num_kv_layers = if let Some(block_types) = qwen3_hybrid_layer_types(config) {
            block_types
                .iter()
                .filter(|t| t.as_str() == "full_attention")
                .count()
        } else {
            config.num_hidden_layers
        };

        let (hybrid_mamba_slot_bytes, hybrid_num_gdn_layers) =
            if let Some(block_types) = qwen3_hybrid_layer_types(config) {
                let num_gdn_layers = block_types
                    .iter()
                    .filter(|t| t.as_str() == "linear_attention")
                    .count();
                if num_gdn_layers == 0 {
                    (None, 0)
                } else {
                    let hybrid = resolve_qwen3_hybrid_config(config);
                    let shard_count = num_shards.max(1);
                    if hybrid.num_v_heads % shard_count != 0 || hybrid.num_k_heads % shard_count != 0 {
                        crate::log_warn!(
                            "Hybrid mamba heads are not divisible by num_shards (v_heads={}, k_heads={}, shards={}); memory estimate uses floor division.",
                            hybrid.num_v_heads,
                            hybrid.num_k_heads,
                            shard_count
                        );
                    }
                    let num_v_heads = std::cmp::max(1, hybrid.num_v_heads / shard_count);
                    let num_k_heads = std::cmp::max(1, hybrid.num_k_heads / shard_count);
                    let conv_window = hybrid.conv_kernel_size.saturating_sub(1);
                    let d_conv = num_k_heads
                        .saturating_mul(hybrid.key_head_dim)
                        .saturating_mul(2)
                        .saturating_add(num_v_heads.saturating_mul(hybrid.value_head_dim));
                    let per_layer_conv_bytes = d_conv
                        .saturating_mul(conv_window)
                        .saturating_mul(model_dtype_size);
                    let per_layer_recurrent_bytes = num_v_heads
                        .saturating_mul(hybrid.key_head_dim)
                        .saturating_mul(hybrid.value_head_dim)
                        .saturating_mul(Dtype::F32.size_bytes());
                    let per_slot_bytes = num_gdn_layers
                        .saturating_mul(per_layer_conv_bytes.saturating_add(per_layer_recurrent_bytes));
                    if per_slot_bytes == 0 {
                        (None, num_gdn_layers)
                    } else {
                        (Some(per_slot_bytes), num_gdn_layers)
                    }
                }
            } else {
                (None, 0)
            };

        let (
            is_mla,
            mla_kv_lora_rank,
            mla_qk_rope_head_dim,
            is_deepseek_v4,
            v4_bytes_per_native_page,
            v4_cache_specs,
            kv_backend,
        ) = {
            let extra: Option<serde_json::Value> = config
                .extra_config_json
                .as_ref()
                .and_then(|s| serde_json::from_str(s).ok());
            if let Some(ref extra) = extra {
                let kv_lora_rank = extra.get("kv_lora_rank").and_then(|v| v.as_u64());
                if let Some(rank) = kv_lora_rank {
                    let rope_dim = extra
                        .get("qk_rope_head_dim")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(64) as usize;
                    (
                        true,
                        rank as usize,
                        rope_dim,
                        false,
                        0,
                        None,
                        KvCacheBackend::Mla,
                    )
                } else {
                    let model_type = extra.get("model_type").and_then(|v| v.as_str());
                    let arch = config
                        .architectures
                        .as_ref()
                        .and_then(|a| a.first())
                        .map(|s| s.as_str())
                        .unwrap_or("");
                    if model_type == Some("deepseek_v4") || is_deepseek_v4_arch_name(arch) {
                        // T3 占位:ds_v4 层搬运期回填真实规格与页字节
                        let _specs: Vec<V4LayerCacheSpec> = Vec::new();
                        let page_bytes = 0usize;
                        (
                            false,
                            0,
                            0,
                            true,
                            page_bytes,
                            Some(_specs),
                            KvCacheBackend::DeepSeekV4,
                        )
                    } else {
                        (false, 0, 0, false, 0, None, KvCacheBackend::Flash)
                    }
                }
            } else {
                (false, 0, 0, false, 0, None, KvCacheBackend::Flash)
            }
        };

        // V4 逻辑单位 = 256 原生 token(vLLM);强制引擎 block size。
        // T3 占位:V4_NATIVE_BLOCK_SIZE 随 ds_v4 搬运回填;当前按 256 定值。
        let v4_native_block_size: usize = 256;
        let block_size = if is_deepseek_v4 {
            v4_native_block_size
        } else {
            econfig.block_size
        };
        if is_deepseek_v4 && econfig.block_size != block_size {
            crate::log_warn!(
                "DeepSeek V4: forcing engine block_size {} → {} (native hybrid page unit)",
                econfig.block_size,
                block_size
            );
        }

        let kvcache_dtype = if is_mla && econfig.kvcache_dtype.is_turboquant() {
            crate::log_warn!(
                "TurboQuant ({:?}) is not supported for MLA models (kv_lora_rank={}). \
                 MLA uses compressed KV cache layout incompatible with TurboQuant. \
                 Falling back to auto KV cache dtype.",
                econfig.kvcache_dtype,
                mla_kv_lora_rank,
            );
            KvCacheDtype::Auto
        } else if is_mla && econfig.kvcache_dtype.is_nvfp4() {
            crate::log_warn!(
                "KV cache dtype nvfp4 selected for MLA/V4. \
                 Using fp8-compatible allocator sizing for now; expand kernels TBD."
            );
            KvCacheDtype::Fp8
        } else {
            econfig.kvcache_dtype
        };

        let per_layer_cache_config = match gemma4_per_layer_cache_config(config) {
            Some(configs) if configs.len() == num_kv_layers => Some(configs),
            Some(_) => {
                crate::log_warn!(
                    "Ignoring Gemma4 heterogeneous KV cache config because it does not match num_kv_layers={}.",
                    num_kv_layers
                );
                None
            }
            None => None,
        };

        let is_moe = config.moe_cfg.is_some()
            && config
                .moe_cfg
                .as_ref()
                .is_some_and(|m| m.num_experts.unwrap_or(0) > 1);
        let moe_intermediate_size = config
            .moe_cfg
            .as_ref()
            .map(|m| m.moe_intermediate_size)
            .unwrap_or(0);
        let moe_num_experts_per_tok = config
            .moe_cfg
            .as_ref()
            .map(|m| m.num_experts_per_tok)
            .unwrap_or(0);

        Self {
            num_hidden_layers: config.num_hidden_layers,
            num_kv_layers,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            num_shards,
            block_size,
            user_max_model_len: econfig.max_model_len,
            user_max_num_seqs: Some(econfig.max_num_seqs.max(1)),
            config_model_len,
            kv_fraction: if econfig.max_model_len.is_some() && econfig.kv_fraction.is_none() {
                if cfg!(feature = "cuda") {
                    0.8
                } else {
                    0.6
                }
            } else {
                kv_fraction
            },
            kvcache_dtype,
            cpu_mem_fold: econfig.cpu_mem_fold.unwrap_or(0.5),
            dtype_size,
            model_dtype_size,
            hybrid_mamba_slot_bytes,
            hybrid_num_gdn_layers,
            is_mla,
            mla_kv_lora_rank,
            mla_qk_rope_head_dim,
            kv_backend,
            is_deepseek_v4,
            v4_bytes_per_native_page,
            v4_cache_specs,
            per_layer_cache_config,
            num_attention_heads: config.num_attention_heads,
            hidden_size: config.hidden_size,
            moe_intermediate_size,
            moe_num_experts_per_tok,
            is_moe,
            prefill_chunk_size: econfig.effective_prefill_chunk_size(),
            use_flashinfer_workspace: uses_flashinfer_workspace(econfig, config),
            draft_weight_bytes: estimate_draft_weight_bytes(econfig.draft_model.as_deref()),
        }
    }

    pub fn resolved_kvcache_dtype(&self) -> KvCacheDtype {
        self.kvcache_dtype
    }

    pub fn kv_backend(&self) -> KvCacheBackend {
        self.kv_backend
    }

    /// 从模型/引擎配置计算确定性工作区预算。
    pub fn compute_workspace_budget(&self) -> GpuMemoryBudget {
        // FlashInfer 工作区:512 MiB float + 128 MiB int (GPU) + 128 MiB 固定 host(非 GPU)
        let flashinfer_bytes: u64 = if self.use_flashinfer_workspace {
            (512 + 128) * 1024 * 1024
        } else {
            0
        };

        let cutlass_bytes: u64 = if cfg!(feature = "cutlass") {
            512 * 1024 * 1024
        } else {
            0
        };

        let moe_pool_bytes: u64 =
            if cfg!(feature = "cutlass") && self.is_moe && self.moe_num_experts_per_tok > 0 {
                let topk = self.moe_num_experts_per_tok;
                let size_m = self.prefill_chunk_size * topk;
                let hidden = self.hidden_size / self.num_shards.max(1);
                let inter = self.moe_intermediate_size / self.num_shards.max(1);
                let largest_dim = hidden.max(2 * inter);
                let gathered = size_m * hidden * self.model_dtype_size;
                let rep_out = size_m * inter * self.model_dtype_size;
                let act_packed = size_m * hidden / 2;
                let act_scales = size_m * (hidden / 16 + 128);
                let pool_total = gathered + rep_out + act_packed + act_scales;
                let transient_output = size_m * largest_dim * self.model_dtype_size;
                (pool_total + transient_output) as u64
            } else {
                0
            };

        let flash_splitk_bytes: u64 = if cfg!(feature = "flash") || cfg!(feature = "flashattn") {
            let q_heads_per_shard = self.num_attention_heads / self.num_shards.max(1);
            let splits = 8usize; // flash::NUM_SPLITS
            let per_layer = 64 * q_heads_per_shard * splits * (self.head_dim + 2) * 4;
            (per_layer * self.num_kv_layers) as u64
        } else {
            0
        };

        let transient_bytes =
            (2 * self.prefill_chunk_size * self.hidden_size * self.model_dtype_size) as u64;

        let mla_attention_bytes_per_token = if self.is_mla {
            let heads_per_shard = self.num_attention_heads / self.num_shards.max(1);
            (3 * heads_per_shard
                * (self.mla_kv_lora_rank + self.mla_qk_rope_head_dim)
                * self.model_dtype_size) as u64
        } else {
            0
        };
        let mla_attention_bytes =
            mla_attention_bytes_per_token.saturating_mul(self.prefill_chunk_size.max(1) as u64);

        let total_bytes = flashinfer_bytes
            + cutlass_bytes
            + moe_pool_bytes
            + flash_splitk_bytes
            + transient_bytes
            + mla_attention_bytes;
        let total_bytes = total_bytes.max(MIN_ACTIVATION_RESERVE_BYTES);

        GpuMemoryBudget {
            flashinfer_bytes,
            cutlass_bytes,
            moe_pool_bytes,
            flash_splitk_bytes,
            transient_bytes,
            mla_attention_bytes_per_token,
            total_bytes,
        }
    }

    fn fixed_workspace_bytes(&self, workspace: &GpuMemoryBudget) -> u64 {
        workspace
            .flashinfer_bytes
            .saturating_add(workspace.cutlass_bytes)
            .saturating_add(workspace.flash_splitk_bytes)
            .saturating_add(MIN_ACTIVATION_RESERVE_BYTES)
    }

    fn prefill_bytes_per_token(&self, workspace: &GpuMemoryBudget) -> u64 {
        let chunk = self.prefill_chunk_size.max(1) as u64;
        workspace
            .transient_bytes
            .saturating_add(workspace.moe_pool_bytes)
            .saturating_add(
                workspace
                    .mla_attention_bytes_per_token
                    .saturating_mul(chunk),
            )
            .saturating_add(chunk - 1)
            / chunk
    }

    /// 估算 decode 批次图捕获保留的缓冲。
    ///
    /// OWL_GRAPH_RESERVE_BYTES(兼容 XINFER_GRAPH_RESERVE_BYTES)整体覆盖;
    /// draft 模式默认预留 4GiB(p5-Q2 §八 点火配方实测值)。
    fn cuda_graph_bytes(&self, graph_batch: usize, max_model_len: usize) -> u64 {
        if graph_batch == 0 {
            return 0;
        }
        if let Some(v) = std::env::var("OWL_GRAPH_RESERVE_BYTES")
            .or_else(|_| std::env::var("XINFER_GRAPH_RESERVE_BYTES"))
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .filter(|v| *v > 0)
        {
            return v;
        }
        if self.draft_weight_bytes > 0 {
            return 4 * 1024 * 1024 * 1024;
        }
        let max_blocks = max_model_len.div_ceil(self.block_size.max(1));
        planned_graph_capture_batches(graph_batch)
            .into_iter()
            .map(|batch| {
                let batch = batch as u64;
                let metadata = batch * (max_blocks as u64 * 8 + 8 * 1024 * 1024);
                let hidden = batch * self.hidden_size as u64 * self.model_dtype_size as u64 * 8;
                let logits = batch
                    * self.num_attention_heads.max(1) as u64
                    * self.model_dtype_size as u64
                    * 4;
                metadata.saturating_add(hidden).saturating_add(logits)
            })
            .sum()
    }

    fn runtime_reserve_bytes(&self, parallel_reqs: usize, max_model_len: usize) -> u64 {
        let workspace = self.compute_workspace_budget();
        let fixed = self.fixed_workspace_bytes(&workspace);
        let per_token = self.prefill_bytes_per_token(&workspace);
        let prefill_tokens = parallel_reqs
            .max(1)
            .saturating_mul(self.prefill_chunk_size.max(1));
        let graph_batch = hybrid_mamba_graph_capture_max_batch(parallel_reqs);
        fixed
            .saturating_add(per_token.saturating_mul(prefill_tokens as u64))
            .saturating_add(self.cuda_graph_bytes(graph_batch, max_model_len))
        // 注:draft 权重不在此处追加——get_free_memory/get_rank_available_memory
        // 已在规划期 netting,这里再算会双计(p5-Q2 §八)。
    }

    fn max_parallel_reqs(
        &self,
        available_memory: u64,
        kv_cache_bytes: u64,
        mamba_bytes: u64,
        max_kv_cache_tokens: usize,
        max_num_reqs: usize,
        max_model_len: usize,
        mamba_active_limit: usize,
    ) -> usize {
        let max_num_reqs = max_num_reqs.max(1);
        let remaining = available_memory
            .saturating_sub(kv_cache_bytes)
            .saturating_sub(mamba_bytes);

        let workspace = self.compute_workspace_budget();
        let fixed_runtime = self.fixed_workspace_bytes(&workspace);
        let per_request_prefill = self
            .prefill_bytes_per_token(&workspace)
            .saturating_mul(self.prefill_chunk_size.max(1) as u64)
            .max(1);
        let kv_parallel_limit = max_kv_cache_tokens / self.prefill_chunk_size.max(1);
        let memory_parallel_limit = remaining
            .saturating_sub(fixed_runtime)
            .checked_div(per_request_prefill)
            .unwrap_or(0) as usize;
        let parallel_limit = kv_parallel_limit.min(memory_parallel_limit);
        // GraphCapturer 捕获 1..=32 的每个精确批次
        let parallel_limit = parallel_limit.min(HYBRID_MAMBA_GRAPH_CAPTURE_MAX_BATCH);

        if max_num_reqs > parallel_limit {
            let graph_capacity = if self.hybrid_mamba_slot_bytes.is_some() {
                hybrid_mamba_graph_capture_max_batch(max_num_reqs)
            } else {
                max_num_reqs
            };
            return if graph_capacity <= mamba_active_limit
                && self.runtime_reserve_bytes(max_num_reqs, max_model_len) <= remaining
            {
                max_num_reqs
            } else {
                0
            };
        }

        for requested in (1..=parallel_limit).rev() {
            let graph_capacity = if self.hybrid_mamba_slot_bytes.is_some() {
                hybrid_mamba_graph_capture_max_batch(requested)
            } else {
                requested
            };
            if graph_capacity > mamba_active_limit {
                continue;
            }
            if self.runtime_reserve_bytes(requested, max_model_len) <= remaining {
                return requested;
            }
        }
        0
    }

    /// 每步调度器 token 预算(vLLM `max_num_batched_tokens`)
    pub fn compute_scheduling_token_budget(
        &self,
        max_num_seqs: usize,
        activation_reserve: u64,
    ) -> usize {
        let chunk = self.prefill_chunk_size;
        let workspace = self.compute_workspace_budget();
        let per_chunk_activation = self
            .prefill_bytes_per_token(&workspace)
            .saturating_mul(chunk.max(1) as u64)
            .max(1);
        let available = activation_reserve.saturating_sub(self.fixed_workspace_bytes(&workspace));
        let possible = (available / per_chunk_activation) as usize;
        possible
            .min(max_num_seqs.max(1))
            .max(1)
            .saturating_mul(chunk)
    }

    /// 设置异构 head_dim 模型(如 Gemma4)的逐层 KV cache 配置
    pub fn set_per_layer_cache_config(&mut self, configs: Vec<(usize, usize)>) {
        assert_eq!(configs.len(), self.num_kv_layers);
        self.per_layer_cache_config = Some(configs);
    }

    pub fn plan(&self, device_ids: &[usize], econfig: &mut EngineConfig) -> Result<()> {
        match self.get_available_memory(device_ids) {
            Ok(available_before_reserve) => {
                let workspace_budget = self.compute_workspace_budget();
                workspace_budget.report(available_before_reserve);
                let runtime_available = self.get_free_memory(device_ids)?;
                // P5 产品化(p5-Q2 §八):draft 模式预算封顶
                let available_before_reserve = if self.draft_weight_bytes > 0
                    && econfig.kv_fraction.is_none()
                {
                    const DRAFT_GRAPH_RESERVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
                    const DRAFT_PLANNING_MARGIN_BYTES: u64 = 768 * 1024 * 1024;
                    let draft_cap = runtime_available
                        .saturating_sub(DRAFT_GRAPH_RESERVE_BYTES)
                        .saturating_sub(DRAFT_PLANNING_MARGIN_BYTES);
                    let capped = available_before_reserve.min(draft_cap);
                    if capped < available_before_reserve {
                        crate::log_warn!(
                            "draft mode: KV/mamba budget capped to {:.2} GB (graph reserve 4G + planning margin)",
                            capped as f64 / SIZE_IN_GB
                        );
                    }
                    capped
                } else {
                    available_before_reserve
                };
                let mut kv_budget =
                    available_before_reserve.saturating_sub(workspace_budget.total_bytes);
                let mut mamba_budget = 0u64;
                let mut mamba_budget_slots = 0usize;
                let mut mamba_budget_enabled = false;

                if let Some(slot_bytes) = self.hybrid_mamba_slot_bytes {
                    let requested_fraction = econfig
                        .mamba_fraction
                        .map(|f| f as f64)
                        .unwrap_or(DEFAULT_HYBRID_MAMBA_FRACTION)
                        .clamp(0.0, MAX_HYBRID_MAMBA_FRACTION);
                    if requested_fraction > 0.0 {
                        mamba_budget_enabled = true;

                        let mut target_budget =
                            ((available_before_reserve as f64) * requested_fraction) as u64;
                        let min_one_slot = slot_bytes as u64;
                        if target_budget > 0 && target_budget < min_one_slot {
                            crate::log_warn!(
                                "Hybrid mamba budget {:.2} MB is smaller than one slot {:.2} MB; bumping to one slot.",
                                target_budget as f64 / SIZE_IN_MB,
                                min_one_slot as f64 / SIZE_IN_MB
                            );
                            target_budget = min_one_slot;
                        }

                        if target_budget >= available_before_reserve {
        crate::bail!(
                                "Hybrid mamba budget ({:.2} GB) leaves no memory for KV cache. Reduce mamba_fraction.",
                                target_budget as f64 / SIZE_IN_GB,
                            );
                        }

                        mamba_budget = target_budget;
                        kv_budget = kv_budget.saturating_sub(mamba_budget);
                        mamba_budget_slots = if slot_bytes == 0 {
                            0
                        } else {
                            (mamba_budget as usize / slot_bytes).max(1)
                        };
                    }
                }

                match self.plan_allocation(kv_budget, device_ids.len()) {
                    Ok(allocation) => {
                        self.apply_to_config(&allocation, econfig);

                        if let Some(slot_bytes) = self.hybrid_mamba_slot_bytes {
                            if !mamba_budget_enabled {
                                econfig.mamba_slot_bytes = 0;
                                econfig.mamba_memory_bytes = 0;
                                econfig.mamba_cache_capacity = None;
                            } else {
                                econfig.mamba_slot_bytes = slot_bytes;
                                econfig.mamba_memory_bytes = mamba_budget as usize;
                            }
                        } else {
                            econfig.mamba_slot_bytes = 0;
                            econfig.mamba_memory_bytes = 0;
                            econfig.mamba_cache_capacity = None;
                        }

                        let prefix_cache_enabled = econfig.prefix_cache.unwrap_or(false);
                        let mamba_active_limit =
                            if self.hybrid_mamba_slot_bytes.is_some() && mamba_budget_enabled {
                                if prefix_cache_enabled {
                                    (mamba_budget_slots / 2).max(1)
                                } else {
                                    mamba_budget_slots.max(1)
                                }
                            } else {
                                usize::MAX
                            };
                        let max_parallel = self.max_parallel_reqs(
                            runtime_available,
                            allocation.kvcache_memory_bytes as u64,
                            mamba_budget,
                            allocation.max_kv_cache_tokens,
                            econfig.max_num_seqs,
                            allocation.max_model_len,
                            mamba_active_limit,
                        );
                        if max_parallel == 0 {
        crate::bail!(
                                "GPU memory is insufficient for one prefill chunk after KV-cache, Mamba, attention, and graph buffers; reduce max_model_len, max_num_seqs, or prefill_chunk_size."
                            );
                        }
                        econfig.max_num_parallel_reqs = max_parallel;
                        econfig.max_num_batched_tokens = max_parallel
                            .saturating_mul(self.prefill_chunk_size.max(1))
                            .min(allocation.max_kv_cache_tokens)
                            .max(1);
                        if self.hybrid_mamba_slot_bytes.is_some() && mamba_budget_enabled {
                            let graph_capture_capacity =
                                hybrid_mamba_graph_capture_max_batch(max_parallel);
                            econfig.mamba_cache_capacity = Some(graph_capture_capacity);
                            let prefix_budget_slots =
                                mamba_budget_slots.saturating_sub(graph_capture_capacity);
                            crate::log_warn!(
                                "Hybrid Mamba Allocation: {} active slot(s), {} prefix slot budget, {} total slot budget, {:.2} GB budget, {:.2} MB/slot, {} linear-attention layer(s), model dtype {} bytes",
                                graph_capture_capacity,
                                prefix_budget_slots,
                                mamba_budget_slots,
                                mamba_budget as f64 / SIZE_IN_GB,
                                self.hybrid_mamba_slot_bytes.unwrap_or(0) as f64 / SIZE_IN_MB,
                                self.hybrid_num_gdn_layers,
                                self.model_dtype_size
                            );
                        }
                        crate::log_warn!(
                            "Per-step scheduling budget: {} batched tokens (prefill chunk {}). KV-cache pool: {} tokens.",
                            econfig.max_num_batched_tokens,
                            econfig.effective_prefill_chunk_size(),
                            econfig.max_kv_cache_tokens
                        );
                        Ok(())
                    }
                    Err(e) => {
                        crate::log_error!("KVCache allocation failed: {}", e);
        crate::bail!("KVCache allocation failed: {}", e)
                    }
                }
            }
            Err(e) => {
                crate::log_error!("Failed to get available memory: {:?}", e);
                Err(e)
            }
        }?;

        Ok(())
    }

    /// 计算每 block 内存字节数。
    pub fn per_block_bytes(&self) -> usize {
        let tq_full = matches!(
            self.kvcache_dtype,
            KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3
        );

        let base = if tq_full {
            // turbo4/turbo3:标准 K/V cache 不按 block 分配
            0
        } else if self.kv_backend.is_deepseek_v4() {
            self.v4_bytes_per_native_page.max(1)
        } else if self.is_mla {
            self.block_size
                * (self.mla_kv_lora_rank + self.mla_qk_rope_head_dim)
                * self.dtype_size
                * self.num_kv_layers
        } else if let Some(ref configs) = self.per_layer_cache_config {
            let mut total = 0usize;
            for &(kv_heads, hd) in configs {
                let heads_per_shard = self.kv_heads_per_shard_for(kv_heads);
                total += self.block_size * heads_per_shard * hd * self.dtype_size * 2;
            }
            total
        } else {
            self.block_size
                * self.kv_heads_per_shard()
                * self.head_dim
                * self.dtype_size
                * 2 // K 和 V
                * self.num_kv_layers
        };

        let tq_extra = match self.kvcache_dtype {
            KvCacheDtype::Turbo8 => {
                let per_layer = |heads: usize, hd: usize| {
                    self.block_size * heads * 4          // v_absmax: f32
                    + self.block_size * heads * (hd / 2) // v_quant: packed 4-bit
                };
                if let Some(ref configs) = self.per_layer_cache_config {
                    configs
                        .iter()
                        .map(|&(kv_heads, hd)| per_layer(self.kv_heads_per_shard_for(kv_heads), hd))
                        .sum()
                } else {
                    per_layer(self.kv_heads_per_shard(), self.head_dim) * self.num_kv_layers
                }
            }
            KvCacheDtype::Turbo4 => {
                let per_layer = |heads: usize, hd: usize| {
                    self.block_size * heads * 4 * 2          // k_absmax + v_absmax
                    + self.block_size * heads * (hd / 2) * 2 // k_quant + v_quant
                };
                if let Some(ref configs) = self.per_layer_cache_config {
                    configs
                        .iter()
                        .map(|&(kv_heads, hd)| per_layer(self.kv_heads_per_shard_for(kv_heads), hd))
                        .sum()
                } else {
                    per_layer(self.kv_heads_per_shard(), self.head_dim) * self.num_kv_layers
                }
            }
            KvCacheDtype::Turbo3 => {
                let per_layer = |heads: usize, hd: usize| {
                    self.block_size * heads * 4 * 2                    // k_absmax + v_absmax
                    + self.block_size * heads * ((hd * 3 + 7) / 8)     // k_quant (3-bit)
                    + self.block_size * heads * (hd / 2)               // v_quant (4-bit)
                };
                if let Some(ref configs) = self.per_layer_cache_config {
                    configs
                        .iter()
                        .map(|&(kv_heads, hd)| per_layer(self.kv_heads_per_shard_for(kv_heads), hd))
                        .sum()
                } else {
                    per_layer(self.kv_heads_per_shard(), self.head_dim) * self.num_kv_layers
                }
            }
            _ => 0,
        };

        base + tq_extra
    }

    /// 给定参数下的所需内存
    pub fn calculate_required_memory(&self, num_seqs: usize, model_len: usize) -> usize {
        let blocks_per_seq = (model_len + self.block_size - 1) / self.block_size;
        let num_blocks = num_seqs * blocks_per_seq;
        num_blocks * self.per_block_bytes()
    }

    fn draft_reserve_usize(&self) -> usize {
        usize::try_from(self.draft_weight_bytes).unwrap_or(usize::MAX)
    }

    /// 查询单设备可用 GPU 内存(应用 kv_fraction 后)
    pub fn get_rank_available_memory(&self, device_id: usize) -> Result<u64> {
        // A4:FFI 只在 owl-cuda;UUID 钉卡纪律下按 ordinal 打开
        let dev = CudaDevice::new(device_id, owl_cuda::TEST_POOL_BYTES)
            .map_err(|e| crate::Error::Msg(format!("CudaDevice::new({device_id}): {e:?}")))?;
        let (free, total) = dev
            .mem_get_info()
            .map_err(|e| crate::Error::Msg(format!("mem_get_info: {e:?}")))?;

        // P5 产品化:先预留 draft 权重(bytes)再取 fraction(p5-Q2 §6.2 定谳)
        let free = free.saturating_sub(self.draft_reserve_usize());
        let usable = (free as f64 * self.kv_fraction) as u64;

        crate::log_warn!(
            "GPU {}: total {:.2} GB, free {:.2} GB (draft reserve {:.2} GB), kv_fraction {:.0}%, Max usable cache budget {:.2} GB",
            device_id,
            total as f64 / SIZE_IN_GB,
            free as f64 / SIZE_IN_GB,
            self.draft_weight_bytes as f64 / SIZE_IN_GB,
            self.kv_fraction * 100.0,
            usable as f64 / SIZE_IN_GB
        );

        Ok(usable)
    }

    /// 查询单设备原始自由 GPU 内存(未应用 kv_fraction)
    pub fn get_rank_free_memory(&self, device_id: usize) -> Result<u64> {
        let dev = CudaDevice::new(device_id, owl_cuda::TEST_POOL_BYTES)
            .map_err(|e| crate::Error::Msg(format!("CudaDevice::new({device_id}): {e:?}")))?;
        let (free, _total) = dev
            .mem_get_info()
            .map_err(|e| crate::Error::Msg(format!("mem_get_info: {e:?}")))?;
        Ok(free.saturating_sub(self.draft_reserve_usize()) as u64)
    }

    /// 跨全部给定 device_ids 查询可用 GPU 内存(取最小值)
    pub fn get_available_memory(&self, device_ids: &[usize]) -> Result<u64> {
        let mut min_memory: Option<u64> = None;
        for &device_id in device_ids {
            let mem = self.get_rank_available_memory(device_id)?;
            min_memory = Some(min_memory.map_or(mem, |m| std::cmp::min(m, mem)));
        }
        min_memory.ok_or_else(|| crate::Error::Msg("No device IDs provided".into()))
    }

    /// 返回跨 rank 的最小原始自由内存
    pub fn get_free_memory(&self, device_ids: &[usize]) -> Result<u64> {
        let mut min_memory: Option<u64> = None;
        for &device_id in device_ids {
            let mem = self.get_rank_free_memory(device_id)?;
            min_memory = Some(min_memory.map_or(mem, |m| std::cmp::min(m, mem)));
        }
        min_memory.ok_or_else(|| crate::Error::Msg("No device IDs provided".into()))
    }

    /// 内存预算内自动决定最优 (max_num_seqs, max_model_len)
    fn auto_decide_params(
        &self,
        available_memory: u64,
    ) -> std::result::Result<(usize, usize), KVCacheError> {
        let per_block = self.per_block_bytes();
        let total_blocks = available_memory as usize / per_block;
        let total_tokens = total_blocks * self.block_size;

        if total_blocks == 0 {
            return Err(KVCacheError::InsufficientGpuMemory {
                available_mb: available_memory as f64 / SIZE_IN_MB,
                required_mb: per_block as f64 / SIZE_IN_MB,
                reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
            });
        }

        if total_tokens <= self.config_model_len {
            let requested = self.user_max_num_seqs.unwrap_or(1).max(1);
            if requested > 1 {
                return Ok((requested, (total_tokens / requested).max(1)));
            }
            return Ok((1, total_tokens));
        }

        let candidates = [
            self.config_model_len,
            self.config_model_len / 2,
            self.config_model_len / 4,
            self.config_model_len / 8,
            16 * 1024,
            8 * 1024,
            4 * 1024,
            1024,
        ];

        for &max_len in candidates.iter() {
            if max_len == 0 {
                continue;
            }
            let blocks_per_seq = (max_len + self.block_size - 1) / self.block_size;
            if total_blocks >= blocks_per_seq {
                let max_possible_seqs = total_blocks / blocks_per_seq;
                let max_seqs = std::cmp::min(
                    max_possible_seqs,
                    self.user_max_num_seqs.unwrap_or(8).max(1),
                );
                return Ok((max_seqs, max_len));
            }
        }

        Ok((1, total_tokens))
    }

    /// 给定跨 rank 最小可用内存,计算分配规划
    pub fn plan_allocation(
        &self,
        min_available_memory: u64,
        num_shards: usize,
    ) -> std::result::Result<KVCacheAllocation, KVCacheError> {
        let per_block = self.per_block_bytes();
        if per_block == 0 {
            return Err(KVCacheError::InvalidConfiguration {
                message: format!(
                    "Invalid KVCache block size: per_block=0 (num_kv_heads={}, num_shards={}, head_dim={}, block_size={}, dtype_size={}, num_kv_layers={})",
                    self.num_kv_heads,
                    self.num_shards,
                    self.head_dim,
                    self.block_size,
                    self.dtype_size,
                    self.num_kv_layers
                ),
            });
        }

        let (max_num_seqs, max_model_len) = if let (Some(max_num_seqs), Some(max_model_len)) =
            (self.user_max_num_seqs, self.user_max_model_len)
        {
            let required_bytes = self.calculate_required_memory(max_num_seqs, max_model_len);
            if required_bytes as u64 > min_available_memory {
                return Err(KVCacheError::InsufficientGpuMemory {
                    available_mb: min_available_memory as f64 / SIZE_IN_MB,
                    required_mb: required_bytes as f64 / SIZE_IN_MB,
                    reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
                });
            }
            (max_num_seqs, max_model_len)
        } else {
            self.auto_decide_params(min_available_memory)?
        };

        let mut num_gpu_blocks = min_available_memory as usize / per_block;

        // P5-piecewise:opt-in 池上限(OWL_KV_POOL_MAX_TOKENS,兼容 XINFER_)
        if let Some(cap_tokens) = std::env::var("OWL_KV_POOL_MAX_TOKENS")
            .or_else(|_| std::env::var("XINFER_KV_POOL_MAX_TOKENS"))
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|t| *t > 0)
        {
            let cap_blocks = cap_tokens.div_ceil(self.block_size.max(1));
            num_gpu_blocks = num_gpu_blocks.min(cap_blocks);
        }

        if num_gpu_blocks == 0 {
            return Err(KVCacheError::InsufficientGpuMemory {
                available_mb: min_available_memory as f64 / SIZE_IN_MB,
                required_mb: per_block as f64 / SIZE_IN_MB,
                reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
            });
        }

        let max_kv_cache_tokens = num_gpu_blocks * self.block_size;
        let kvcache_memory_bytes = num_gpu_blocks * per_block;
        let max_num_batched_tokens = 0;

        // CPU swap block
        let num_cpu_blocks = (num_gpu_blocks as f32 * self.cpu_mem_fold) as usize;

        if num_gpu_blocks == 0 || max_num_seqs == 0 {
            return Err(KVCacheError::InsufficientGpuMemory {
                available_mb: min_available_memory as f64 / SIZE_IN_MB,
                required_mb: per_block as f64 / SIZE_IN_MB,
                reserved_mb: CUDA_RESERVED_BYTES as f64 / SIZE_IN_MB,
            });
        }

        let allocation = KVCacheAllocation {
            num_gpu_blocks,
            num_cpu_blocks,
            max_num_seqs,
            max_model_len,
            kvcache_memory_bytes,
            max_num_batched_tokens,
            max_kv_cache_tokens,
        };

        crate::log_warn!(
            "KVCache Allocation: {} GPU blocks ({:.2} GB x {}), max KV-cache tokens {} ({}k bytes per token), scheduling limits [{} seqs x {} tokens]",
            num_gpu_blocks,
            kvcache_memory_bytes as f64 / SIZE_IN_GB,
            num_shards,
            max_kv_cache_tokens,
            per_block / 1024 / self.block_size,
            max_num_seqs,
            max_model_len
        );

        Ok(allocation)
    }

    /// 将分配结果应用到 EngineConfig
    pub fn apply_to_config(&self, allocation: &KVCacheAllocation, econfig: &mut EngineConfig) {
        econfig.num_blocks = allocation.num_gpu_blocks;
        econfig.max_num_seqs = allocation.max_num_seqs;
        econfig.max_model_len = Some(allocation.max_model_len);
        econfig.kvcache_memory_bytes = allocation.kvcache_memory_bytes;
        econfig.max_kv_cache_tokens = allocation.max_kv_cache_tokens;
        econfig.max_num_batched_tokens = allocation.max_num_batched_tokens;
        if self.is_deepseek_v4 {
            econfig.block_size = self.block_size;
        }
    }

    /// 是否需要 auto-decide 模式
    pub fn needs_auto_decide(&self) -> bool {
        self.user_max_model_len.is_none()
    }

    //==========================================================================
    // 张量分配方法
    //==========================================================================

    /// 在 GPU 与 CPU 上初始化 KV cache 张量。
    ///
    /// 返回一等 [`GpuKvCache`] / [`CpuKvCache`] 后端
    /// (Flash / Mla / DeepSeekV4 混合页)。
    pub fn init_kv_cache(
        &self,
        allocation: &KVCacheAllocation,
        dtype: Dtype,
        device: &CudaDevice,
        pd_config: Option<&crate::transfer::PdConfig>,
    ) -> Result<(GpuKvCache, CpuKvCache)> {
        let num_gpu_blocks = allocation.num_gpu_blocks;
        let num_cpu_blocks = allocation.num_cpu_blocks;

        // sync_alloc:PD Server 角色的同步分配标记(engine cuda-only,恒生效)
        #[allow(unused)]
        let sync_alloc = pd_config
            .map(|p_cfg| matches!(p_cfg.role, PdRole::Server))
            .unwrap_or(false);

        let cache_dtype = if self.kvcache_dtype.is_fp8_keys() {
            Dtype::U8
        } else {
            dtype
        };
        if self.kvcache_dtype.is_turboquant() {
            crate::log_warn!(
                "TurboQuant mode: {:?}, standard cache dtype {:?} (dummy for turbo4/turbo3)",
                self.kvcache_dtype,
                cache_dtype
            );
        } else {
            crate::log_warn!(
                "KV cache dtype: {:?}, cache dtype {:?}, backend {:?}",
                self.kvcache_dtype,
                cache_dtype,
                self.kv_backend
            );
        }

        if self.kv_backend == KvCacheBackend::DeepSeekV4 {
            let _specs = self.v4_cache_specs.as_ref().ok_or_else(|| {
                crate::Error::Msg("DeepSeek V4 KV backend missing cache specs at init".into())
            })?;
            crate::log_warn!(
                "DeepSeek V4: allocating engine-owned hybrid page pool ({} native pages, {:.2} GB)",
                num_gpu_blocks,
                (num_gpu_blocks * self.v4_bytes_per_native_page) as f64 / SIZE_IN_GB
            );
            let pool = V4HybridPagePool {
                layers: (0.._specs.len()).collect(),
            };
            let pool = std::sync::Arc::new(parking_lot::Mutex::new(Some(pool)));
            return Ok((GpuKvCache::DeepSeekV4(pool), CpuKvCache::DeepSeekV4));
        }

        if self.kv_backend == KvCacheBackend::Mla {
            // MLA cache:每层 (ckv_cache, kpe_cache)
            let mut gpu_cache = Vec::new();
            let mut cpu_cache = Vec::new();
            let ckv_shape = [num_gpu_blocks, self.block_size, 1, self.mla_kv_lora_rank];
            let kpe_shape = [
                num_gpu_blocks,
                self.block_size,
                1,
                self.mla_qk_rope_head_dim,
            ];
            let ckv_cpu = [num_cpu_blocks, self.block_size, 1, self.mla_kv_lora_rank];
            let kpe_cpu = [
                num_cpu_blocks,
                self.block_size,
                1,
                self.mla_qk_rope_head_dim,
            ];
            for _ in 0..self.num_kv_layers {
                let ckv_blocks = alloc_gpu_buffer(device, &ckv_shape, cache_dtype)?;
                let kpe_blocks = alloc_gpu_buffer(device, &kpe_shape, cache_dtype)?;
                let _ = sync_alloc; // stream-ordered 分配自带排序(T3 运行期按角色细化)
                gpu_cache.push((ckv_blocks, kpe_blocks));
            }
            for _ in 0..self.num_kv_layers {
                let ckv_blocks = host_zeros(&ckv_cpu, cache_dtype);
                let kpe_blocks = host_zeros(&kpe_cpu, cache_dtype);
                cpu_cache.push((ckv_blocks, kpe_blocks));
            }
            return Ok((GpuKvCache::Mla(gpu_cache), CpuKvCache::Mla(cpu_cache)));
        }

        // Flash 路径
        let (gpu_cache, cpu_cache) = self.init_flash_kv_cache(
            num_gpu_blocks,
            num_cpu_blocks,
            cache_dtype,
            device,
            sync_alloc,
        )?;
        Ok((GpuKvCache::Flash(gpu_cache), CpuKvCache::Flash(cpu_cache)))
    }

    fn init_flash_kv_cache(
        &self,
        num_gpu_blocks: usize,
        num_cpu_blocks: usize,
        cache_dtype: Dtype,
        device: &CudaDevice,
        sync_alloc: bool,
    ) -> Result<(
        Vec<(DynTensor<CudaDevice>, DynTensor<CudaDevice>)>,
        Vec<(Vec<u8>, Vec<u8>)>,
    )> {
        if cfg!(feature = "flash")
            || cfg!(feature = "flashinfer")
            || cfg!(feature = "flashattn")
            || cfg!(feature = "metal")
        {
            // turbo4/turbo3:标准 K/V cache 不使用——分配 1-block 哑块
            let tq_full = matches!(
                self.kvcache_dtype,
                KvCacheDtype::Turbo4 | KvCacheDtype::Turbo3
            );
            let std_gpu_blocks = if tq_full { 1 } else { num_gpu_blocks };
            let std_cpu_blocks = if tq_full { 1 } else { num_cpu_blocks };

            let mut gpu_cache = Vec::new();
            let mut cpu_cache = Vec::new();
            for layer_idx in 0..self.num_kv_layers {
                let (_, kv_heads, hd) = self.layer_flash_key_value_block_shape(layer_idx);
                let shape = [std_gpu_blocks, self.block_size, kv_heads, hd];
                let key_blocks = alloc_gpu_buffer(device, &shape, cache_dtype)?;
                let value_blocks = alloc_gpu_buffer(device, &shape, cache_dtype)?;
                let _ = sync_alloc;
                gpu_cache.push((key_blocks, value_blocks));
            }
            for layer_idx in 0..self.num_kv_layers {
                let (_, kv_heads, hd) = self.layer_flash_key_value_block_shape(layer_idx);
                let shape = [std_cpu_blocks, self.block_size, kv_heads, hd];
                let key_blocks = host_zeros(&shape, cache_dtype);
                let value_blocks = host_zeros(&shape, cache_dtype);
                cpu_cache.push((key_blocks, value_blocks));
            }

            if self.kvcache_dtype.is_turboquant() {
                self.init_turboquant_cache(num_gpu_blocks, device, sync_alloc)?;
            }

            Ok((gpu_cache, cpu_cache))
        } else {
            let mut gpu_cache = Vec::new();
            let mut cpu_cache = Vec::new();
            for layer_idx in 0..self.num_kv_layers {
                let kshape = self.layer_key_block_shape(layer_idx, cache_dtype);
                let vshape = self.layer_value_block_shape(layer_idx);
                let k_arr = [kshape.0, kshape.1, kshape.2, kshape.3];
                let v_arr = [vshape.0, vshape.1, vshape.2];
                let key_blocks = alloc_gpu_buffer(device, &k_arr, cache_dtype)?;
                let value_blocks = alloc_gpu_buffer(device, &v_arr, cache_dtype)?;
                let _ = sync_alloc;
                gpu_cache.push((key_blocks, value_blocks));
            }
            for layer_idx in 0..self.num_kv_layers {
                let kshape = self.layer_key_block_shape(layer_idx, cache_dtype);
                let vshape = self.layer_value_block_shape(layer_idx);
                let k_arr = [kshape.0, kshape.1, kshape.2, kshape.3];
                let v_arr = [vshape.0, vshape.1, vshape.2];
                let key_blocks = host_zeros(&k_arr, cache_dtype);
                let value_blocks = host_zeros(&v_arr, cache_dtype);
                cpu_cache.push((key_blocks, value_blocks));
            }
            Ok((gpu_cache, cpu_cache))
        }
    }

    /// TurboQuant GPU 缓存(T3 占位:vendor attention_rs 归宿裁决待定;
    /// 类型层保留,运行里程碑回填)
    fn init_turboquant_cache(
        &self,
        num_gpu_blocks: usize,
        device: &CudaDevice,
        sync_alloc: bool,
    ) -> Result<()> {
        let _ = (num_gpu_blocks, device, sync_alloc);
        unimplemented!("T3 运行里程碑回填:TurboQuant vendor 归宿裁决待定(attention_rs TurboquantMode/init_turboquant_cache)")
    }

    /// 分配 CPU 侧 TQ 缓冲用于 swap。非 TQ 模式返回 None(T3 占位同上)
    pub fn init_cpu_tq_cache(
        &self,
        num_cpu_blocks: usize,
    ) -> Result<Option<Vec<CpuTqLayerCache>>> {
        let _ = num_cpu_blocks;
        unimplemented!("T3 运行里程碑回填:TurboQuant vendor 归宿裁决待定(CpuTqLayerCache host 段)")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hybrid_mamba_graph_capture_max_batch_scales_with_max_num_seqs() {
        assert_eq!(hybrid_mamba_graph_capture_max_batch(1), 1);
        assert_eq!(hybrid_mamba_graph_capture_max_batch(8), 8);
        assert_eq!(hybrid_mamba_graph_capture_max_batch(17), 17);
        assert_eq!(hybrid_mamba_graph_capture_max_batch(32), 32);
        assert_eq!(hybrid_mamba_graph_capture_max_batch(64), 32);
    }
}
