# T2 侦察:xinfer 六文件 candle 触点逐行清单(只读普查,无翻译方案)

> 普查对象:`packages/xinfer/crates/core/src` 下
> `utils/config.rs`、`utils/kv_backend.rs`、`utils/kvcache_allocator.rs`、
> `utils/logits_processor.rs`、`core/scheduler.rs`、`core/sequence.rs`。
> 归类口径:【Tensor 方法】【Tensor 构造】【DType】【Device】
> 【Result/Error】【bail!/宏】【其他】。
> 2026-09-22 千问侦察兵产出(思考 low,纯罗列)。

---

## 1. utils/config.rs(2144 行)

### imports(行 2-11)

```
2:  use crate::transfer::PdConfig;
3:  use llguidance::api::TopLevelGrammar;
4:  #[cfg(feature = "python")]
5:  use pyo3::pyclass;
6:  use serde::de::value::SeqAccessDeserializer;
7:  use serde::de::{Deserializer, Visitor};
8:  use serde::ser::Error as _;
9:  use serde::{Deserialize, Serialize, Serializer};
10: use std::collections::{HashMap, HashSet};
11: use std::fmt;
```

### candle 触点

| 行号 | 触点原文 | 归类 |
|---|---|---|
| 306 | `pub hidden_act: candle_nn::Activation,` | 【其他】(candle_nn 枚举字段,非 candle_core) |

> 全文件仅此 1 处 candle 触点。其余 `Result` 行(56/75/86/91/96/119/189/207/333/
> 747/755/758/761/764/767/770/1399/1486)全部是 serde 序列化器上下文里的
> std `Result`(如 `Result<S::Ok, S::Error>`、`fmt::Result`),**不是 candle**。

### 结构体/枚举清单

| 名 | 行范围 | 字段/变体数 | 标注 |
|---|---|---|---|
| KvCacheDtype | 14-21 | 6 变体 | 自包含(可原样搬) |
| EosTokenId | 69-72 | 2 变体 | 自包含(可原样搬) |
| BincodeEosTokenId(私有) | 224-229 | - | 自包含(可原样搬) |
| MoEConfig | 230-249 | 14 字段 | 自包含(可原样搬) |
| RopeScalingValue | 253-258 | 4 变体 | 自包含(可原样搬) |
| Config | 277-330 | 39 字段 | **含 candle 字段**(行 306 `candle_nn::Activation`) |
| EngineConfig | 422-495 | 50 字段 | 自包含(可原样搬)(cfg 双定义之一) |
| EngineConfig | 508-616 | 50 字段 | 自包含(可原样搬)(cfg 双定义之二,同上结构) |
| TokenizerConfig | 786-797 | 7 字段 | 自包含(可原样搬) |
| SamplingParams | 801-836 | 16 字段 | 自包含(可原样搬)(cfg 双定义之一) |
| SamplingParams | 841-886 | 16 字段 | 自包含(可原样搬)(cfg 双定义之二,同上结构) |
| ModelType | 968-993 | 22 变体 | 自包含(可原样搬) |
| GenerationConfig | 1026-1042 | 7 字段 | 自包含(可原样搬) |
| QuantConfig | 1073-1108 | 17 字段 | 自包含(可原样搬) |
| ReasoningEffort | 1421-1437 | 7 变体 | 自包含(可原样搬) |

---

## 2. utils/kv_backend.rs(107 行)

### imports(行 6-9)

```
6:  use crate::models::layers::ds_v4::V4HybridPagePool;
7:  use candle_core::Tensor;
8:  use parking_lot::Mutex;
9:  use std::sync::Arc;
```

### candle 触点

| 行号 | 触点原文 | 归类 |
|---|---|---|
| 7 | `use candle_core::Tensor;` | 【Tensor 构造】(import) |
| 41 | `Flash(Vec<(Tensor, Tensor)>),` | 【Tensor 构造】(枚举字段) |
| 42 | `Mla(Vec<(Tensor, Tensor)>),` | 【Tensor 构造】(枚举字段) |
| 56 | `pub fn as_pairs(&self) -> Option<&Vec<(Tensor, Tensor)>> {` | 【Tensor 构造】(签名) |
| 58 | `Self::Flash(v) \| Self::Mla(v) => Some(v),` | 【其他】(模式,无直接触点) |
| 63 | `pub fn as_pairs_mut(&mut self) -> Option<&mut Vec<(Tensor, Tensor)>> {` | 【Tensor 构造】(签名) |
| 81 | `Self::DeepSeekV4(p) => p.lock().as_ref().map(\|pool\| pool.layers.len())...` | 【其他】(无直接触点) |
| 88 | `Flash(Vec<(Tensor, Tensor)>),` | 【Tensor 构造】(枚举字段) |
| 89 | `Mla(Vec<(Tensor, Tensor)>),` | 【Tensor 构造】(枚举字段) |
| 94 | `pub fn as_pairs(&self) -> Option<&Vec<(Tensor, Tensor)>> {` | 【Tensor 构造】(签名) |
| 101 | `pub fn as_pairs_mut(&mut self) -> Option<&mut Vec<(Tensor, Tensor)>> {` | 【Tensor 构造】(签名) |

> 全文件无 Tensor 方法调用(只持有/传递 `(Tensor, Tensor)` 对)。
> `V4HybridPagePool` 是 crate 内类型(`models::layers::ds_v4`),非 candle。

### 结构体/枚举清单

| 名 | 行范围 | 字段/变体数 | 标注 |
|---|---|---|---|
| KvCacheBackend | 13-20 | 3 变体 | 自包含(可原样搬) |
| GpuKvCache | 40-44 | 3 变体 | **含 candle 字段**(`Vec<(Tensor, Tensor)>`) |
| CpuKvCache | 87-91 | 3 变体 | **含 candle 字段**(`Vec<(Tensor, Tensor)>`) |

---

## 3. utils/kvcache_allocator.rs(1973 行)

### imports(行 19-22)

```
19: use super::{gemma4_per_layer_cache_config, qwen3_hybrid_layer_types, resolve_qwen3_hybrid_config};
20: use crate::utils::config::{Config, EngineConfig};
21: use candle_core::{DType, Device, Result, Tensor};
22: use std::fmt;
```

(行 1-18 为模块 doc 注释。)

### candle 触点

**import / 结构字段 / 签名**

| 行号 | 触点原文 | 归类 |
|---|---|---|
| 21 | `use candle_core::{DType, Device, Result, Tensor};` | 【DType】【Device】【Result/Error】【Tensor 构造】(import) |
| 331 | `cache_dtype: DType,` | 【DType】(方法参数) |
| 334 | `let element_size = cache_dtype.size_in_bytes();` | 【DType】(方法调用) |
| 345 | `pub fn new(econfig: &EngineConfig, config: &Config, dtype: DType) -> Self {` | 【DType】(签名) |
| 424 | `.saturating_mul(DType::F32.size_in_bytes());` | 【DType】(方法调用) |
| 926 | `pub fn plan(&self, device_ids: &[usize], econfig: &mut EngineConfig) -> Result<()> {` | 【Result/Error】 |
| 1203 | `pub fn get_rank_available_memory(&self, device_id: usize) -> Result<u64> {` | 【Result/Error】 |
| 1204 | `use candle_core::backend::BackendDevice;` | 【Device】(import) |
| 1205 | `use candle_core::cuda_backend::cudarc::driver::sys;` | 【Device】(import,直触 cudarc) |
| 1206 | `use candle_core::cuda_backend::CudaDevice;` | 【Device】(import) |
| 1209 | `let _ = CudaDevice::new(device_id)?;` | 【Device】 |
| 1218 | `.map_err(\|e\| candle_core::Error::Msg(format!("cuMemGetInfo_v2 failed: {e:?}")))?;` | 【Result/Error】 |
| 1244 | `pub fn get_rank_free_memory(&self, device_id: usize) -> Result<u64> {` | 【Result/Error】 |
| 1245 | `use candle_core::backend::BackendDevice;` | 【Device】(import) |
| 1246 | `use candle_core::cuda_backend::cudarc::driver::sys;` | 【Device】(import,直触 cudarc) |
| 1247 | `use candle_core::cuda_backend::CudaDevice;` | 【Device】(import) |
| 1249 | `let _ = CudaDevice::new(device_id)?;` | 【Device】 |
| 1256 | `.map_err(\|e\| candle_core::Error::Msg(format!("cuMemGetInfo_v2 failed: {e:?}")))?;` | 【Result/Error】 |
| 1266 | `pub fn get_rank_available_memory(&self, _device_id: usize) -> Result<u64> {` | 【Result/Error】(metal 分支) |
| 1298 | `pub fn get_rank_free_memory(&self, _device_id: usize) -> Result<u64> {` | 【Result/Error】(metal 分支) |
| 1320 | `pub fn get_available_memory(&self, device_ids: &[usize]) -> Result<u64> {` | 【Result/Error】 |
| 1328 | `min_memory.ok_or_else(\|\| candle_core::Error::msg("No device IDs provided"))` | 【bail!/宏】(Error 构造) |
| 1332 | `pub fn get_free_memory(&self, device_ids: &[usize]) -> Result<u64> {` | 【Result/Error】 |
| 1340 | `min_memory.ok_or_else(\|\| candle_core::Error::msg("No device IDs provided"))` | 【bail!/宏】(Error 构造) |
| 1352 | `) -> std::result::Result<(usize, usize), KVCacheError> {` | 【其他】(显式 std Result) |
| 1415 | `) -> std::result::Result<KVCacheAllocation, KVCacheError> {` | 【其他】(显式 std Result) |
| 1549 | `dtype: DType,` | 【DType】(方法参数) |
| 1550 | `device: &Device,` | 【Device】(方法参数) |
| 1552 | `) -> Result<(crate::utils::GpuKvCache, crate::utils::CpuKvCache)> {` | 【Result/Error】 |
| 1589 | `DType::U8` | 【DType】 |
| 1631 | `let ckv_blocks = Tensor::empty(` | 【Tensor 构造】 |
| 1637 | `let kpe_blocks = Tensor::empty(` | 【Tensor 构造】 |
| 1651 | `let ckv_blocks = Tensor::zeros(` | 【Tensor 构造】 |
| 1654 | `&Device::Cpu,` | 【Device】 |
| 1656 | `let kpe_blocks = Tensor::zeros(` | 【Tensor 构造】 |
| 1664 | `&Device::Cpu,` | 【Device】 |
| 1687 | `cache_dtype: DType,` | 【DType】(方法参数) |
| 1688 | `device: &Device,` | 【Device】(方法参数) |
| 1690 | `) -> Result<(Vec<(Tensor, Tensor)>, Vec<(Tensor, Tensor)>)> {` | 【Result/Error】【Tensor 构造】(签名) |
| 1711 | `let key_blocks = Tensor::empty(` | 【Tensor 构造】 |
| 1717 | `let value_blocks = Tensor::empty(` | 【Tensor 构造】 |
| 1727 | `let key_blocks = Tensor::zeros(` | 【Tensor 构造】 |
| 1730 | `&Device::Cpu,` | 【Device】 |
| 1732 | `let value_blocks = Tensor::zeros(` | 【Tensor 构造】 |
| 1735 | `&Device::Cpu,` | 【Device】 |
| 1751 | `let key_blocks = Tensor::empty(` | 【Tensor 构造】 |
| 1757 | `let value_blocks = Tensor::empty(` | 【Tensor 构造】 |
| 1768 | `let key_blocks = Tensor::zeros(` | 【Tensor 构造】 |
| 1771 | `&Device::Cpu,` | 【Device】 |
| 1773 | `let value_blocks = Tensor::zeros(` | 【Tensor 构造】 |
| 1776 | `&Device::Cpu,` | 【Device】 |
| 1787 | `device: &Device,` | 【Device】(方法参数) |
| 1789 | `) -> candle_core::Result<()> {` | 【Result/Error】 |
| 1804 | `let v_absmax = Tensor::empty(` | 【Tensor 构造】 |
| 1806 | `candle_core::DType::F32,` | 【DType】 |
| 1810 | `let v_quant = Tensor::empty(` | 【Tensor 构造】 |
| 1812 | `candle_core::DType::U8,` | 【DType】 |
| 1819 | `let ka = Tensor::empty(` | 【Tensor 构造】 |
| 1821 | `candle_core::DType::F32,` | 【DType】 |
| 1825 | `let kq = Tensor::empty(` | 【Tensor 构造】 |
| 1827 | `candle_core::DType::U8,` | 【DType】 |
| 1834 | `let ka = Tensor::empty(` | 【Tensor 构造】 |
| 1836 | `candle_core::DType::F32,` | 【DType】 |
| 1841 | `let kq = Tensor::empty(` | 【Tensor 构造】 |
| 1843 | `candle_core::DType::U8,` | 【DType】 |
| 1874 | `) -> candle_core::Result<Option<Vec<crate::core::runner::CpuTqLayerCache>>> {` | 【Result/Error】 |
| 1892 | `let v_absmax = Tensor::zeros(` | 【Tensor 构造】 |
| 1894 | `candle_core::DType::F32,` | 【DType】 |
| 1895 | `&Device::Cpu,` | 【Device】 |
| 1897 | `let v_quant = Tensor::zeros(` | 【Tensor 构造】 |
| 1899 | `candle_core::DType::U8,` | 【DType】 |
| 1900 | `&Device::Cpu,` | 【Device】 |
| 1905 | `let ka = Tensor::zeros(` | 【Tensor 构造】 |
| 1907 | `candle_core::DType::F32,` | 【DType】 |
| 1908 | `&Device::Cpu,` | 【Device】 |
| 1910 | `let kq = Tensor::zeros(` | 【Tensor 构造】 |
| 1912 | `candle_core::DType::U8,` | 【DType】 |
| 1913 | `&Device::Cpu,` | 【Device】 |
| 1918 | `let ka = Tensor::zeros(` | 【Tensor 构造】 |
| 1920 | `candle_core::DType::F32,` | 【DType】 |
| 1921 | `&Device::Cpu,` | 【Device】 |
| 1924 | `let kq = Tensor::zeros(` | 【Tensor 构造】 |
| 1926 | `candle_core::DType::U8,` | 【DType】 |
| 1927 | `&Device::Cpu,` | 【Device】 |
| 993 | `candle_core::bail!(` | 【bail!/宏】 |
| 1049 | `candle_core::bail!(` | 【bail!/宏】 |
| 1088 | `candle_core::bail!("KVCache allocation failed: {}", e)` | 【bail!/宏】 |
| 1610 | `candle_core::Error::Msg("DeepSeek V4 KV backend missing cache specs at init".into())` | 【Result/Error】 |

> 注意:行 1274/1306 的 `metal::Device::system_default()` 是 Metal 后端,
> 非 candle【Device】。Tensor 只有 `empty`/`zeros` 构造,**无方法调用**
> (无 narrow/to_device 等)。

### 结构体/枚举清单

| 名 | 行范围 | 字段/变体数 | 标注 |
|---|---|---|---|
| GpuMemoryBudget | 51-66 | 7 字段 | 自包含(可原样搬) |
| KVCacheAllocation | 117-132 | 7 字段 | 自包含(可原样搬) |
| KVCacheError | 136-147 | 3 变体 | 自包含(可原样搬) |
| KVCacheAllocator | 181-230 | 32 字段 | 自包含(可原样搬)(字段均非 candle 类型,`KvCacheDtype`/`KvCacheBackend` 是 crate 内类型) |

---

## 4. utils/logits_processor.rs(747 行)

### imports(行 1-10)

```
1:  use super::config::SamplingParams;
2:  #[cfg(feature = "cuda")]
3:  use attention_rs::sort::ArgSortOp; //Use our custom sort kernel, fix kernel crash on A100
4:  use candle_core::D;
5:  use candle_core::{DType, Error, Result, Tensor};
6:  use parking_lot::Mutex;
7:  use rand::{distr::Distribution, SeedableRng};
8:  use rayon::iter::IntoParallelIterator;
9:  use rayon::iter::ParallelIterator;
10: use std::sync::Arc;
```

### candle 触点

| 行号 | 触点原文 | 归类 |
|---|---|---|
| 4 | `use candle_core::D;` | 【Tensor 方法】(D 轴) |
| 5 | `use candle_core::{DType, Error, Result, Tensor};` | 【DType】【Result/Error】【Tensor 构造】(import) |
| 67 | `fn sample_argmax(&self, logits: &Tensor) -> Result<Vec<u32>> {` | 【Tensor 构造】【Result/Error】(签名) |
| 68 | `let next_tokens = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;` | 【Tensor 方法】 |
| 72 | `fn sample_multinomial(&self, prs: &Vec<f32>) -> Result<u32> {` | 【Result/Error】(签名) |
| 73 | `let distr = rand::distr::weighted::WeightedIndex::new(prs).map_err(Error::wrap)?;` | 【Result/Error】 |
| 82 | `fn sample_topp(&self, logits: &Tensor, top_p: f32) -> Result<Vec<u32>> {` | 【Tensor 构造】【Result/Error】(签名) |
| 84 | `let asort = logits.arg_sort(false)?;` | 【Tensor 方法】(cuda 分支) |
| 87 | `.to_device(&candle_core::Device::Cpu)?` | 【Device】(非 cuda 分支) |
| 88 | `.arg_sort_last_dim(false)?;` | 【Tensor 方法】(非 cuda 分支) |
| 89 | `let asort: Vec<Vec<u32>> = asort.to_vec2()?;` | 【Tensor 方法】 |
| 90 | `let sorted: Vec<Vec<f32>> = logits.to_vec2()?;` | 【Tensor 方法】 |
| 91 | `let batch = logits.layout().dims()[0];` | 【Tensor 方法】 |
| 92 | `let vec_ret: Result<Vec<u32>> = (0..batch)` | 【Result/Error】 |
| 95 | `let indices: Vec<u32> = asort[b].to_vec();` | 【其他】(Vec 的 to_vec,非 Tensor) |
| 96 | `let mut prs: Vec<f32> = sorted[b].to_vec();` | 【其他】(Vec 的 to_vec) |
| 114 | `fn sample_topk(&self, logits: &Tensor, top_k: usize) -> Result<Vec<u32>> {` | 【Tensor 构造】【Result/Error】(签名) |
| 116 | `let (sorted, asort) = logits.sort(false)?;` | 【Tensor 方法】(cuda 分支) |
| 119 | `.to_device(&candle_core::Device::Cpu)?` | 【Device】(非 cuda 分支) |
| 120 | `.sort_last_dim(false)?;` | 【Tensor 方法】(非 cuda 分支) |
| 122 | `.narrow(candle_core::D::Minus1, 0, top_k)?` | 【Tensor 方法】 |
| 125 | `.narrow(candle_core::D::Minus1, 0, top_k)?` | 【Tensor 方法】 |
| 128 | `let asort: Vec<Vec<u32>> = asort.to_vec2()?;` | 【Tensor 方法】 |
| 129 | `let sorted: Vec<Vec<f32>> = sorted.to_vec2()?;` | 【Tensor 方法】 |
| 131 | `let vec_ret: Result<Vec<u32>> = (0..batch)` | 【Result/Error】 |
| 145 | `fn sample_topk_topp(&self, logits: &Tensor, top_k: usize, top_p: f32) -> Result<Vec<u32>> {` | 【Tensor 构造】【Result/Error】(签名) |
| 147 | `let (sorted, asort) = logits.sort(false)?;` | 【Tensor 方法】(cuda 分支) |
| 150 | `.to_device(&candle_core::Device::Cpu)?` | 【Device】(非 cuda 分支) |
| 151 | `.sort_last_dim(false)?;` | 【Tensor 方法】(非 cuda 分支) |
| 154 | `.narrow(candle_core::D::Minus1, 0, top_k)?` | 【Tensor 方法】 |
| 157 | `.narrow(candle_core::D::Minus1, 0, top_k)?` | 【Tensor 方法】 |
| 160 | `let asort: Vec<Vec<u32>> = asort.to_vec2()?;` | 【Tensor 方法】 |
| 161 | `let sorted: Vec<Vec<f32>> = sorted.to_vec2()?;` | 【Tensor 方法】 |
| 165 | `let vec_ret: Result<Vec<u32>> = (0..batch)` | 【Result/Error】 |
| 201 | `pub fn sample_with_strategy_sort(` | 【其他】(方法定义行) |
| 203 | `logits: &Tensor,` | 【Tensor 构造】(参数) |
| 205 | `) -> Result<Option<Vec<u32>>> {` | 【Result/Error】 |
| 222 | `let prs = candle_nn::ops::softmax_last_dim(&prs)?;` | 【Tensor 方法】(candle_nn 自由函数) |
| 226 | `let asort = prs.arg_sort(false)?;` | 【Tensor 方法】 |
| 227 | `let sorted_v = prs.gather(&asort, candle_core::D::Minus1)?;` | 【Tensor 方法】 |
| 228 | `let v = sorted_v.layout().dims()[1];` | 【Tensor 方法】 |
| 231 | `.narrow(candle_core::D::Minus1, 0, k_eff)?` | 【Tensor 方法】 |
| 234 | `.narrow(candle_core::D::Minus1, 0, k_eff)?` | 【Tensor 方法】 |
| 236 | `let sorted_cpu = sorted.to_device(&candle_core::Device::Cpu)?.to_vec2::<f32>()?;` | 【Device】【Tensor 方法】 |
| 237 | `let asort_cpu = asort.to_device(&candle_core::Device::Cpu)?.to_vec2::<u32>()?;` | 【Device】【Tensor 方法】 |
| 273 | `logits: &Tensor,` | 【Tensor 构造】(参数) |
| 275 | `) -> Result<Option<Vec<u32>>> {` | 【Result/Error】 |
| 292 | `candle_core::Device::Cuda(d) => d.clone(),` | 【Device】 |
| 298 | `let out_idx = Tensor::zeros((1, k_eff), DType::U32, &device)?;` | 【Tensor 构造】【DType】【Device】 |
| 309 | `let gathered = logits.gather(&out_idx, D::Minus1)?;` | 【Tensor 方法】 |
| 310 | `let vals_cpu = gathered.to_device(&candle_core::Device::Cpu)?.to_vec2::<f32>()?;` | 【Device】【Tensor 方法】 |
| 312 | `let asort_cpu = out_idx.to_device(&candle_core::Device::Cpu)?.to_vec2::<u32>()?;` | 【Device】【Tensor 方法】 |
| 371 | `logits: &Tensor,` | 【Tensor 构造】(参数) |
| 373 | `) -> Result<Option<Vec<u32>>> {` | 【Result/Error】 |
| 390 | `candle_core::Device::Cuda(d) => d.clone(),` | 【Device】 |
| 397 | `let prs = candle_nn::ops::softmax_last_dim(&prs)?;` | 【Tensor 方法】(candle_nn 自由函数) |
| 398 | `let prs_c = prs.to_dtype(DType::F32)?.contiguous()?;` | 【Tensor 方法】【DType】 |
| 401 | `prs_c.narrow(1, 0, 1).ok().and_then(\|x\| x.to_device(&candle_core::Device::Cpu).ok())` | 【Tensor 方法】【Device】 |
| 407 | `let out_idx = Tensor::zeros((1, k_eff), DType::U32, &device)?;` | 【Tensor 构造】【DType】【Device】 |
| 416 | `let sorted_v = prs.gather(&out_idx, D::Minus1)?;` | 【Tensor 方法】 |
| 417 | `let sorted_cpu = sorted_v.to_device(&candle_core::Device::Cpu)?.to_vec2::<f32>()?;` | 【Device】【Tensor 方法】 |
| 419 | `let asort_cpu = out_idx.to_device(&candle_core::Device::Cpu)?.to_vec2::<u32>()?;` | 【Device】【Tensor 方法】 |
| 454 | `pub fn sample_with_strategy(&self, logits: &Tensor, sampling: &Sampling) -> Result<Vec<u32>> {` | 【Tensor 构造】【Result/Error】(签名) |
| 488 | `let logits = logits.to_dtype(DType::F32)?;` | 【Tensor 方法】【DType】 |
| 490 | `let prs = \|temperature: f64\| -> Result<Tensor> {` | 【Result/Error】【Tensor 构造】(闭包签名) |
| 491 | `let logits = (&logits / temperature)?;` | 【Tensor 方法】(算术 op) |
| 492 | `let prs = candle_nn::ops::softmax_last_dim(&logits)?;` | 【Tensor 方法】(candle_nn 自由函数) |
| 497 | `Sampling::ArgMax => self.sample_argmax(&logits)?,` | 【其他】(内部调用) |
| 499 | `let prs = prs(*temperature as f64)?.to_vec2()?;` | 【Tensor 方法】 |
| 508 | `let prs = prs.to_vec2()?;` | 【Tensor 方法】 |
| 531 | `logits: &Tensor,` | 【Tensor 构造】(参数) |
| 533 | `) -> Result<Vec<u32>> {` | 【Result/Error】 |
| 565 | `logits: &Tensor,` | 【Tensor 构造】(参数) |
| 569 | `) -> Result<Tensor> {` | 【Result/Error】【Tensor 构造】 |
| 573 | `let logits: Vec<Vec<f32>> = if logits.dtype() != candle_core::DType::F32 {` | 【Tensor 方法】【DType】 |
| 574 | `logits.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?` | 【Tensor 方法】【DType】 |
| 576 | `logits.to_vec2::<f32>()?` | 【Tensor 方法】 |
| 598 | `Tensor::from_vec(logits, (batch, logits_len), device)` | 【Tensor 构造】 |
| 723 | `fn f32_dev_ptr(t: &Tensor) -> Result<*const f32> {` | 【Tensor 构造】【Result/Error】(签名) |
| 724 | `use candle_core::cuda_backend::cudarc::driver::DevicePtr;` | 【Device】(import,直触 cudarc) |
| 727 | `candle_core::Storage::Cuda(p) => {` | 【其他】(Storage 枚举模式) |
| 731 | `_ => candle_core::bail!("vllm sampler: logits must be cuda f32"),` | 【bail!/宏】 |
| 737 | `fn u32_dev_ptr(t: &Tensor) -> Result<*mut u32> {` | 【Tensor 构造】【Result/Error】(签名) |
| 738 | `use candle_core::cuda_backend::cudarc::driver::DevicePtr;` | 【Device】(import,直触 cudarc) |
| 741 | `candle_core::Storage::Cuda(p) => {` | 【其他】(Storage 枚举模式) |
| 745 | `_ => candle_core::bail!("vllm sampler: index buffer must be cuda u32"),` | 【bail!/宏】 |

### 结构体/枚举清单

| 名 | 行范围 | 字段/变体数 | 标注 |
|---|---|---|---|
| Sampling | 12-18 | 5 变体(ArgMax / All / TopK / TopP / TopKThenTopP) | 自包含(可原样搬) |
| LogitsProcessor | 20-25 | 3 字段 | **含 candle 字段**(`fast_sampler: Arc<Mutex<attention_rs::sampler::Sampler>>`,cuda feature;`rand`/`parking_lot` 字段自包含) |

> 备注:`attention_rs::sort::ArgSortOp`(行 3)与 `attention_rs::sampler::Sampler`
> 是 vendor crate(非 candle_core),按【其他】处理,但搬运时同样要决定
> 归宿(owl 自有采样/排序 kernel 或保留 vendor 依赖)。

---

## 5. core/scheduler.rs(1518 行)

### imports(行 2-16)

```
2:  use super::runner::RunnerType;
3:  use super::{
4:      block_manager::BlockManager,
5:      prefix_cache::PrefixCacheConfig,
6:      sequence::{Sequence, SequenceStatus},
7:  };
8:  use crate::transfer::{PdConfig, PdRole};
9:  use crate::utils::config::{Config, EngineConfig, EosTokenId};
10: use candle_core::Result;
11: use parking_lot::RwLock;
12: use regex::Regex;
13: use std::collections::{HashMap, VecDeque};
14: use std::sync::Arc;
15: use std::time::{SystemTime, UNIX_EPOCH};
16: use tokenizers::Tokenizer;
```

### candle 触点

| 行号 | 触点原文 | 归类 |
|---|---|---|
| 10 | `use candle_core::Result;` | 【Result/Error】(import) |
| 236 | `pub fn schedule(&mut self) -> Result<(Vec<usize>, bool)> {` | 【Result/Error】 |
| 420 | `) -> Result<usize> {` | 【Result/Error】 |
| 498 | `pub fn get_seq_token_usage(&self, seq_id: usize) -> Result<usize> {` | 【Result/Error】 |
| 1156 | `pub fn try_receive_kvcache(&mut self) -> Result<()> {` | 【Result/Error】 |

> 全文件**只有 candle Result** 一个触点面。无 Tensor/DType/Device 触点。
> 依赖的 crate 内类型:`BlockManager`、`PrefixCacheConfig`、`Sequence`、
> `SequenceStatus`、`PdConfig`、`PdRole`、`Config`、`EngineConfig`、
> `EosTokenId`、`RunnerType`(均需在各自模块先行/同步搬运)。

### 结构体/枚举清单

| 名 | 行范围 | 字段/变体数 | 标注 |
|---|---|---|---|
| Scheduler | 17-45 | 17 字段 | 自包含(可原样搬)(字段类型全为 std/crate 内类型;`block_manager: BlockManager` 与 `cfg: EngineConfig` 是 crate 内依赖,非 candle) |

---

## 6. core/sequence.rs(402 行)

### imports(行 2-6)

```
2:  use crate::utils::config::SamplingParams;
3:  use crate::utils::image::ImageData;
4:  use serde::{Deserialize, Serialize};
5:  use std::fmt;
6:  use std::time::{SystemTime, UNIX_EPOCH};
```

### candle 触点

**无。** 全文件 0 处 candle/candle_core/cudarc 触点。

### 结构体/枚举清单

| 名 | 行范围 | 字段/变体数 | 标注 |
|---|---|---|---|
| SequenceStatus | 8-15 | 6 变体(Waiting/Running/Finished/Cached/Swapped/FinishSwapped) | 自包含(可原样搬) |
| Sequence | 32-55 | 19 字段 | 自包含(可原样搬)(依赖 `SamplingParams`、`ImageData`,均 crate 内类型) |
| DecodeSequence | 58-66 | 7 字段 | 自包含(可原样搬)(依赖 `SamplingParams`) |

---

## 7. 总表:六文件搬运顺序建议(依赖少的先)

| 顺序 | 文件 | candle 触点数 | 理由(依赖少的先) |
|---|---|---|---|
| 1 | core/sequence.rs | 0 | 零 candle;只依赖 config 的 SamplingParams 与 image 的 ImageData,可最先原样搬 |
| 2 | utils/config.rs | 1(行 306 `candle_nn::Activation`) | 近乎自包含;唯一触点是一个枚举字段,替换成本低;是 sequence/scheduler/kvcache_allocator 的共同依赖,第二搬可解锁面最大 |
| 3 | core/scheduler.rs | 5(全部是 `candle_core::Result`) | 无 Tensor/Device,只需一个 Result 别名翻译;依赖 sequence+config+block_manager/prefix_cache(后两者另行处理) |
| 4 | utils/kv_backend.rs | ~9(全为 `Tensor` 类型持有) | 无方法调用,只需 Tensor 类型替换(owl 池化张量);依赖 models/layers/ds_v4 的 V4HybridPagePool(跨层依赖,可先以类型占位) |
| 5 | utils/kvcache_allocator.rs | ~65(构造/DType/Device/Result 混合,含 cudarc 直触) | 触点面最大但结构自包含(4 个类型全自包含);依赖 config+kv_backend,放第五;Tensor 只有 empty/zeros 构造,翻译集中 |
| 6 | utils/logits_processor.rs | ~75(Tensor 方法密集:to_device/to_vec2/narrow/gather/sort/arg_sort/to_dtype/softmax_last_dim,含 cudarc DevicePtr 直触) | 方法调用最密,且含 attention_rs 的 Sampler/ArgSortOp 两个 vendor 依赖;依赖 config(Sampling)+ 自有 Sampling 枚举,放最后 |

### 附:跨文件依赖备注(非 candle,但搬运时会碰到)

- `sequence.rs` → `config::SamplingParams`、`image::ImageData`
- `scheduler.rs` → `sequence::{Sequence, SequenceStatus}`、`block_manager::BlockManager`、`prefix_cache::PrefixCacheConfig`、`config::{Config, EngineConfig, EosTokenId}`、`transfer::{PdConfig, PdRole}`、`runner::RunnerType`、`tokenizers::Tokenizer`、`regex::Regex`
- `kv_backend.rs` → `models::layers::ds_v4::V4HybridPagePool`(跨 models 层,搬运时需类型占位或先搬该类型)
- `kvcache_allocator.rs` → `config::{Config, EngineConfig, KvCacheDtype}`、`super::{gemma4_per_layer_cache_config, qwen3_hybrid_layer_types, resolve_qwen3_hybrid_config}`、`kv_backend::{KvCacheBackend, GpuKvCache, CpuKvCache}`、`core::runner::{CpuTqLayerCache}`、`models::layers::ds_v4::{V4LayerCacheSpec, V4HybridPagePool}`
- `logits_processor.rs` → `config::SamplingParams`、`attention_rs::sort::ArgSortOp`(cuda)、`attention_rs::sampler::Sampler`(cuda)、`rand`、`rayon`
