//! **算子族装配参数的下发面**(2026-11-01 分层修正)。
//!
//! # 为什么不在 `DeviceClient` 上
//!
//! `DeviceClient` 是**设备能力契约**(执行 / 搬运 / 图 / 流 / 分配)。
//! "prefill 用不用 f16 影子""plan 表是否图自携""plan 用不用图档瓦"是
//! **算子族的装配参数**,与设备能力无关。摆在 trait 上会带来两宗罪:
//!
//! 1. **每个后端都得背 flashinfer 的语义** —— 设备抽象层被迫知道某个族的
//!    内核长什么样;
//! 2. **缺省空实现 = 静默托管** —— 后端没接就悄悄不生效(引擎以为开了、
//!    族按缺省跑),正是本仓明令禁止的失败形态。
//!
//! # 现在的形状
//!
//! - 配置面归**族自己**:`owl_kernels::family::flashinfer::FlashinferCfg` +
//!   `configure()`(整快照,含自洽门;写原子与 C++ 侧 env 一次落)。
//! - **写者唯一 = 本模块**(引擎装配期 = 配置持有者),读侧 = 族 runtime。
//! - 位置纪律:必须在**任何 plan / 图捕获 / warmup 之前**(`boot_knobs_check` 内)。
//!
//! 加新的族装配参数时:配置键 → [`EngineKnobs`](crate::EngineKnobs) →
//! 本函数 → 族 `configure`,不要回到 trait 上加方法。

use owl_iface::contract::ModelError;

use crate::engine::EngineKnobs;

/// 把引擎旋钮里的**族装配参数**下发给各族(新的族 `configure` 在此排队)。
pub(crate) fn apply(knobs: &EngineKnobs, max_seq_tokens: usize) -> Result<(), ModelError> {
    // ── FlashInfer 族 ──
    // 影子容量只在开启时给值:自洽门要求 `prefill_shadow ⇒ cap > 0`
    // (cap = 0 会让首层按 0 预留 ⇒ 开关静默不生效);关时给 0 以免误导。
    let cfg = owl_kernels::family::flashinfer::FlashinferCfg {
        plan_restore: knobs.fi_plan_restore,
        cuda_graph: knobs.fi_cuda_graph,
        prefill_shadow: knobs.prefill_shadow,
        prefill_shadow_cap: if knobs.prefill_shadow { max_seq_tokens } else { 0 },
    };
    owl_kernels::family::flashinfer::configure(cfg)
        .map_err(|e| ModelError::Msg(format!("算子族装配参数下发失败(flashinfer): {e}")))?;
    // 响亮留痕:配置面 → 族 的实际落地值(无法从配置键回推的东西就靠这一行;
    // 说是“开了”而实际没开 = 最难查的一类)
    owl_shared::log_info!(
        "kernel_cfg",
        "flashinfer: plan_restore={} cuda_graph={} prefill_shadow={} cap={}",
        cfg.plan_restore,
        cfg.cuda_graph,
        cfg.prefill_shadow,
        cfg.prefill_shadow_cap
    );
    Ok(())
}
