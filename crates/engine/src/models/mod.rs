//! models(T2-三 搬运):Qwen3.5 dense/MoE/MTP + DFlash draft 四件套。
//! 上层业务逻辑照搬 xinfer,算子面统一翻译为 owl-nn/owl-cuda 契约。
//! 编译先行口径:控制流骨架保留,设备体 T3 kernel 回填。

pub mod layers;
pub mod qwen3_5;
