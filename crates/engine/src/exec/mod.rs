//! 执行器(设备侧照办;2026-10-01 拆分自 engine.rs §3;2026-10-10 再拆
//! 六模块:phases 调度三相位/spec 轮机器/draft 草稿器面/prefill_chunk
//! 块式预填/turn_end 终局)。S0 决策/执行分离的**执行半**:调度面
//! (scheduler.rs)产出 SchedulerOutput,本模块无决策权。
//! 状态块触碰一律经 StatePool(state.rs),不直摸块句柄。

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_models::interpreters::eval_ops_scoped_env;
use owl_models::layers::gdn::GdnBuffers;
use owl_models::module::{FiPrefillCtx, ForwardCtx, KvBuffers, Module};
use owl_models::tokenizer::Tokenizer;
use owl_models::TensorOps;

use crate::graph_plan::f32b;
use crate::scheduler::BeginPlan;

pub mod phases;
pub mod spec;
pub mod draft;
pub mod prefill_chunk;
pub mod turn_end;

pub(crate) fn spec_degrade_transition(
    streak: usize,
    degraded: bool,
    probe_every: usize,
    m: usize,
    threshold: usize,
    probe_floor: usize,
) -> (usize, bool, usize) {
    if m >= 1 {
        return (0, false, probe_floor);
    }
    let streak = streak + 1;
    if degraded {
        (streak, true, (probe_every * 2).min(512))
    } else if threshold > 0 && streak >= threshold {
        (streak, true, probe_floor)
    } else {
        (streak, false, probe_every)
    }
}
use crate::turn::TurnEvent;


/// E5-M5 spec 分相探针(owl-metrics 直用 API:release 恒开,热路径每轮
/// ~8 条环形推入 ≈ µs 级;/debug/metrics 查询。宏是 debug-only,装载域
/// 「直用 API」双轨纪律的 server 侧复刻)
fn mrec(tag: &str, dur: std::time::Duration) {
    owl_shared::metrics::with_metrics_store(|s| {
        s.timer_record_tag(tag, dur, file!(), line!());
    });
}
fn mcnt(tag: &str, n: u64) {
    owl_shared::metrics::with_metrics_store(|s| s.counter_add(tag, n, file!(), line!()));
}
/// 事实计数器(E5-DF4 崩坏排查):per-boot 命名空间记事实值
/// (token id / m / pos……counter 累计语义下每 tag 只记一次 = 事实值)
fn mfact(boot: u64, fact: &str, v: u64) {
    owl_shared::metrics::with_metrics_store(|s| {
        s.counter_add(&format!("b{boot}.{fact}"), v, file!(), line!());
    });
}
/// FNV-1a 64(逐层 checksum;与装载校验同款字批量混入)
fn fnv_bytes(data: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut chunks = data.chunks_exact(8);
    for c in &mut chunks {
        let w = u64::from_le_bytes(c.try_into().unwrap());
        h = (h ^ w).wrapping_mul(0x100000001b3);
    }
    for b in chunks.remainder() {
        h = (h ^ (*b as u64)).wrapping_mul(0x100000001b3);
    }
    h
}

type Result<T> = std::result::Result<T, ModelError>;


/// 全文重解兜底,流式末字符可能延后一笔)
pub(crate) fn decode_delta(tok: &Tokenizer, out: &[u32], decoded: &mut String) -> String {
    const REPL: &str = "\u{FFFD}";
    let full = tok.decode(out);
    let common = decoded
        .as_bytes()
        .iter()
        .zip(full.as_bytes())
        .take_while(|(a, b)| a == b)
        .count();
    let mut cut = common;
    while cut > 0 && (!full.is_char_boundary(cut) || !decoded.is_char_boundary(cut)) {
        cut -= 1;
    }
    let mut emit = full[cut..].to_string();
    if emit.ends_with(REPL) {
        emit.truncate(emit.len() - REPL.len());
    }
    *decoded = full[..cut + emit.len()].to_string();
    emit
}

