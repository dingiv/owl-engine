//! **引擎侧 VLM 请求态**(2026-10-20 视觉接线 V2)。
//!
//! 模型层 [`owl_models::vlm::Vlm`] 负责塔/表/注入;本模块只持**一个 turn
//! 的已备料**:host 表(m-RoPE/行号)+ 塔输出的设备叶块。
//!
//! 生命周期 = `ActiveTurn`(turn 终了随 `Owner::Turn` 回收);decode 只用
//! `ins.delta` 修正 rope 位置,不再碰塔。

use owl_iface::contract::Bytes;
use owl_models::vlm::VlmInputs;
use owl_models::TensorOps;

/// 一个 turn 的 VLM 备料(`ActiveTurn.vlm`)
pub(crate) struct VlmPrepared {
    /// 位置号/行号/ m-RoPE 表(host 副本 + 设备张量;chunk 切片用)
    pub(crate) ins: VlmInputs,
    /// 塔输出 `merged [N,D]` 的设备叶块(所有权 = turn;叶子只借 id)
    pub(crate) merged: Option<(Bytes, Vec<u64>, TensorOps)>,
}
