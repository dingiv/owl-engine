//! 模型加载域(**2026-09-30 重组**:格式源 + 装载执行收拢一域)。
//!
//! | 模块 | 职责 | 入口 |
//! |---|---|---|
//! | [`mmap`] | 只读映射原语 + 全 dtype 条目索引(两源共享底座;零数据拷贝) | [`mmap::open_raw_index`] |
//! | [`safetensors`] | f16 基线流式源(mmap 直解码/F16C/DONTNEED;F32/BF16) | [`safetensors::SafeTensorsSource`] |
//! | [`w4a16`] | W4A16 量化源(ct pack-quantized → marlin 懒物化;g128-sym) | [`w4a16::W4A16Source`] |
//! | [`kind`] | 起点格式探测(目录嗅探,零数据拷贝) | [`kind::detect`] |
//! | [`spec`] | 起点维度事实(config.json 优先 + 形状兜底) | [`spec::ModelSpec`](crate::model::ModelSpec) |
//! | [`qwen35`] | 模型档事实(Qwen3.5/3.8 族:维度预设 + 键约定 + 检查点特例) | [`qwen35::Qwen35Convention`] |
//! | [`load`] | 装载域解释器(Loadable::layout → 取数/变换/物化/流水) | [`load::load_from_dir`] |
//! | [`awq_build`] | AWQ 源的**字节物化面**(缓存取用口/打包器;自 awq.rs 拆出) | [`awq_build`] |
//! | [`util`] | **格式无关共享助手**(marlin 谓词 / 位型转换;两源同用,不得寄生于某一源) | [`util::marlin_eligible`] |
//!
//! 分层纪律:**源(safetensors/w4a16)只管「键 → 字节」**,执行(load)
//! 只管「字节 → 设备块」—— 源不懂设备,执行不懂格式。新格式(如
//! AWQ g32-asym、W4A8)按 [`WeightSource`](crate::module::WeightSource)
//! 契约另起子模块,勿塞进既有源。
//!
//! 兼容出口:crate 根再导出 `SafeTensorsSource`/`W4A16Source`/
//! `marlin_eligible`/`marlin_n_pack`;`crate::interpreters::eval_load`
//! 经 interpreters 模块 shim 转发(历史调用面不动)。

pub mod awq;
pub(crate) mod awq_build;
pub mod kind;
pub mod load;
pub mod mmap;
pub mod qwen35;
pub mod safetensors;
/// **装载面分片化**(S4 ②;确定性分片协议:pass + 源包装)
pub mod shard;
pub mod spec;
pub mod util;
pub mod w4a16;

/// 刀2 虚拟合并键:{base}in_proj_qkvz = row-stack({base}in_proj_qkv,
/// {base}in_proj_z)。装载期列拼接(vLLM 同款 in_proj_qkvz 单投影);
/// 两源同 k,行堆叠 = 字节拼接,零数值风险。各源(awq/safetensors)
/// 在自身键面上合成,层侧声明单合并 Linear。
pub(crate) fn split_qkvz(base: &str) -> Option<(String, String)> {
    let prefix = base.strip_suffix("in_proj_qkvz")?;
    Some((format!("{prefix}in_proj_qkv"), format!("{prefix}in_proj_z")))
}

pub use awq::AwqSource;
pub use kind::{detect, CheckpointKind, Detection, QuantScheme};
pub use load::{eval_load, load_component, load_from_dir, LoadOpts};
pub use qwen35::{hybrid_3to1, load_tokenizer, qwen3_5_0_8b, qwen3_8_27b, Qwen35Convention};
pub use safetensors::SafeTensorsSource;
pub use util::{marlin_eligible, marlin_n_pack};
pub use w4a16::W4A16Source;
