//! **GGUF 起点族**(2026-10-19 gguf 轮;设计见
//! `docs/arch/gguf-格式支持-设计与施工.md`)。
//!
//! 与既有起点族(`safetensors` / `w4a16` / `awq`)的关系:**同一条契约** ——
//! 「键 → 字节」,`源不懂设备、执行不懂格式`。GGUF 的特殊性全在**自带的东西多**:
//!
//! | 面 | 模块 | 说明 |
//! |---|---|---|
//! | 容器 | [`reader`] | header / KV / tensor table / 对齐 / mmap(**本刀落点**) |
//! | 维度事实 | `meta`(刀 3) | `qwen35.*` / `qwen35moe.*` / `dflash.*` KV → `ModelSpec` |
//! | 词表 | `tokenizer`(刀 2) | 内嵌 `tokenizer.ggml.*` → `tokenizers::Tokenizer` |
//! | 键约定 | `convention`(刀 4) | `blk.N.*` ↔ owl `layers.N.*` |
//! | 权重字节 | `source`(刀 4) | `GgufSource: WeightSource`,**零解量化**(块字节原样上卡) |
//! | TP 分片 | [`shard`](shard)(W-1) | 角色表 + 块对齐列切 + v-head grouped 聚合(**源自己承担分片**) |
//!
//! 两条跨模块律(设计 §六 I-1 / I-5):
//! - **形状翻转律**:GGML `ne[0]` = 最内维(= Linear 的 `in_dim`)⇒ 报给装载面
//!   的 `[out, in]` 必须在**源侧**翻转一次(safetensors 是 `[out, in]`);
//! - **块布局 = 内核读序**:零重排,装载字节区间 ≡ `gguf-py` 报的 offset/length。

pub mod convention;
pub mod dequant;
pub mod kernel_gate;
pub mod load_gate;
pub mod meta;
pub mod plan;
pub mod reader;
pub mod shard;
pub mod tokenizer;
pub mod source;
pub mod transform;

pub use convention::GgufConvention;
pub use dequant::{dequant_all, dequant_row};
/// 量化型词表(块几何 + 可跑性映射在同一处;`owl_kernels` 为权威住址)
pub use owl_kernels::family::gguf::{GgmlTy, Q8_1_BLOCK_BYTES};
pub use meta::{rope_facts, spec_from_gguf, GgufRopeFacts, GgufSpecReport};
pub use plan::{require_portable, ty_histogram, GgufPlan};
pub use shard::{geom_of, role_of, Cols, KeyGeom, Role, ShardSel, Shape, TensorFacts};
pub use source::GgufSource;
pub use reader::{Gguf, GgufValue, KvTable, TensorInfo, MAGIC};
pub use tokenizer::{build_vocab, GgufVocab, SpecialIds, VocabModel};
pub use transform::{classify, xf_for, GgufDims, KeyClass, KeyXf, RowPerm, ValueXf};
