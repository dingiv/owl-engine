//! **E4 接入面(声明;未实现)** —— 引擎域"少一次 host 往返 / 少一次排空"的接口位。
//!
//! ⚠️ 本模块的函数**有意**处于未接线态(声明面先就位;实现挂 E4 轮)
//! ⇒ 模块级 `dead_code` 豁免:此处的 "never used" 是**设计事实**(见下方纪律),
//! 不是遗漏。接线后应删掉本豁免,让编译器继续盯新的死代码。
#![allow(dead_code)]
//!
//! 设计出处:`docs/arch/三流事件序-设计与迁移.md` §3.2(复合口)/ §6(E4 行)。
//! 一句话:**引擎域今天的三处隐式同步**,各有对应的显式接口位。
//!
//! | # | 靶面 | 今天 | E4 形态 | 收益预估 |
//! |---|---|---|---|---|
//! | **E4-1** | **采样回读** | `phases.rs`:launch(采样核) → `face.dtoh(logits)`(~1MB) → host argmax(两次 actor 往返) | `LaunchAndHarvest`:**一次**往返 + **一次**设备端序(采样核 → 回读) | 每 step 省一次往返(百级 step/轮) |
//! | **E4-2** | **KV 装填** | 每步 `htod_chunks`;块复用/清零序由 server 兜 | server 侧清零序已修(`GpuCtx::wait_zeroing`)**⇒ 引擎侧零改动**;若将来有"池块跨轮复用"的显式路径,复用同一事件面 | 0(已由 b7ff082 覆盖) |
//! | **E4-3** | **spec 快照** | turn-end `copy_batch` + 排空 | 快照"拍摄/恢复"与 decode 重叠(设备端序替代排空) | 待量 |
//!
//! **纪律**(AGENTS:不许把可失败逻辑藏在 `()` 或静默里):
//! 本模块的函数**全部返回 `Err(未实现)`** —— 声明面就位、接线未做时**调用即响亮拒**,
//! 绝不"看起来能用"。实现落地时:① 换掉 `Err` 的实现;② 跑 §6 的 E4 门
//! (decode tok/s 不退化 + 长跑稳定 + 数值门);③ 把本注释的"今天"列改成历史。

use owl_iface::contract::{Bytes, DeviceClient, ModelError};

/// **E4-1:采样并收割 logits**(声明;未实现)。
///
/// 形态 = `LaunchAndHarvest`:一次命令把「采样核发射 + 回读其输出块」串成
/// 一次 actor 往返,序由 server 内部保证(设备端事件;客户端不出现流/事件/槽)。
///
/// 调用方(未来):`crates/engine/src/exec/phases.rs` 的 decode_step —— 今天
/// 那里是 `launch(...)` + `face.dtoh(&logits, &mut buf)`,两次往返、两次序。
///
/// # Errors
/// **未实现**:恒返回 [`ModelError::Msg`](响亮;绝不静默假装成功)。
pub(crate) async fn launch_and_harvest_logits<D: DeviceClient>(
    _face: &mut D,
    _launch: owl_iface::contract::LaunchMsg,
    _logits: &Bytes,
    _out: &mut [u8],
) -> Result<(), ModelError> {
    Err(ModelError::Msg(
        "E4-1(launch_and_harvest_logits)未实现:声明面已就位,接线属 E4 另轮;\
         见 docs/arch/三流事件序-设计与迁移.md §6"
            .into(),
    ))
}

/// **E4-3:快照拍摄/恢复与 decode 重叠**(声明;未实现)。
///
/// 今天:turn-end 的 `copy_batch` + 排空;**E4 形态**:用同一事件面把"快照落定"
/// 变成一条可等待的依赖,不再整流排空(收益待量,风险 = 与图回放的交界)。
///
/// # Errors
/// **未实现**:恒返回 [`ModelError::Msg`](响亮)。
pub(crate) async fn snapshot_overlapped<D: DeviceClient>(
    _face: &mut D,
    _dir: SnapshotDir,
) -> Result<(), ModelError> {
    Err(ModelError::Msg(
        "E4-3(snapshot_overlapped)未实现:声明面已就位;见设计文档 §6(E4)".into(),
    ))
}

/// 快照方向(拍摄 = 会话 → 快照池;恢复 = 快照池 → 会话格)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SnapshotDir {
    Capture,
    Restore,
}
