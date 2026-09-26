//! 模型规格集(每档一文件;批8 拆分,2026-09-26 用户裁决)。
//!
//! **拆分律**:此处只住**模型特有**的参数事实(维度预设/层型表约定/
//! 检查点特例),零机制;所有模型共有的主干机制住 [`crate::model`]。
//! mixer 词汇超出 Full|Gdn(MoE/MLA/…)时另立 DecoderLayer 扩展口,
//! 不动主干(需求基线:arch 分发表不写死)。
//!
//! 文件名 = 模型档位:`qwen35` = Qwen3.5 文本主干。新档(如 qwen3
//! dense)另起文件,禁混居。

pub mod qwen35;

pub use qwen35::{hybrid_3to1, load_0_8b, qwen3_5_0_8b, Qwen35Convention};
