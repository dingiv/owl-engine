//! 执行器(设备侧照办;2026-10-01 拆分自 engine.rs §3;2026-10-10 再拆
//! 六模块:phases 调度三相位/spec 轮机器/draft 草稿器面/prefill_chunk
//! 块式预填/turn_end 终局)。S0 决策/执行分离的**执行半**:调度面
//! (scheduler.rs)产出 SchedulerOutput,本模块无决策权。
//! 状态块触碰一律经 StatePool(state.rs),不直摸块句柄。

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_models::interpreters::eval_ops_scoped_env;
use owl_models::layers::gdn::GdnBuffers;
use owl_models::module::{FiPrefillCtx, ForwardCtx, KvBuffers};
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

/// 增量解码(全量重解后取后缀差分;多字节字符跨 token 边界时以
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

/// **采样接受的前缀长度**(A11,2026-10-14):首个 `drafts[i] != sampled[i]`
/// 的 `i`。语义 = 发射 `sampled[0..=m]`(前 m 个恰等于 drafts,末位是
/// 在拒绝点的重抽)。
///
/// 分布正确性(逐位归纳):发 x 的概率 =
///   P(s=draft)·1[x=draft] + P(s≠draft)·P(s=x | s≠draft) = p_target(x)
/// —— 即"抽-比"式投机采样,**不依赖草稿分布**(DFlash2 选择器只有 lattice
/// 码本、无全词表分布 ⇒ 无法用 p/q 比值式)。
pub(crate) fn spec_sampled_accept_prefix(drafts: &[u32], sampled: &[u32]) -> usize {
    drafts
        .iter()
        .zip(sampled.iter())
        .take_while(|(d, s)| d == s)
        .count()
}

/// 字节偏移向下取 char 边界(超界 = 全长;切 `&str` 前必须过一道)
pub(crate) fn floor_boundary(s: &str, i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    let mut i = i;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// **stop 尾缓冲上界**:返回最大的 `end` 使 `s[..end]` 的尾部不包含任一停止串的
/// **真前缀**(否则半截停止串会被当成正文流给客户端,命中时收不回来)。
/// 空 `stops` = 全长(零开销)。
pub(crate) fn safe_end(s: &str, stops: &[String]) -> usize {
    let mut end = s.len();
    for st in stops {
        // 尾部 k 字节(1 ≤ k < |st|)若恰为 st 的前缀 ⇒ 回退到该起点
        let maxk = (st.len() - 1).min(s.len());
        for k in (1..=maxk).rev() {
            let cut = s.len() - k;
            if s.is_char_boundary(cut) && st.starts_with(&s[cut..]) && cut < end {
                end = cut;
                break;
            }
        }
    }
    end
}


#[cfg(test)]
mod spec_accept_tests {
    use super::spec_sampled_accept_prefix;

    /// 全同 ⇒ 全接受(深度 = drafts 长度)
    #[test]
    fn all_match_accepts_everything() {
        assert_eq!(spec_sampled_accept_prefix(&[1, 2, 3], &[1, 2, 3, 9]), 3);
    }

    /// 首个不等处停(末位是重抽,不参与接受)
    #[test]
    fn first_mismatch_stops() {
        assert_eq!(spec_sampled_accept_prefix(&[1, 2, 3], &[1, 7, 3, 4]), 1);
        assert_eq!(spec_sampled_accept_prefix(&[1, 2, 3], &[9, 2, 3, 4]), 0);
    }

    /// 边界:空草稿 / 采样表短于草稿(取较短者,不越界)
    #[test]
    fn bounds_are_safe() {
        assert_eq!(spec_sampled_accept_prefix(&[], &[1]), 0);
        assert_eq!(spec_sampled_accept_prefix(&[1, 2], &[1]), 1);
    }
}

#[cfg(test)]
mod stop_tests {
    use super::{floor_boundary, safe_end};

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    /// 空 stop 列表 = 零回吐(全长)
    #[test]
    fn no_stops_returns_full_len() {
        assert_eq!(safe_end("abc", &[]), 3);
    }

    /// 尾部是停止串的**真前缀** ⇒ 回退(半截不许流给客户端)
    #[test]
    fn holdback_on_partial_prefix() {
        let stops = s(&["</tool_call>"]);
        assert_eq!(safe_end("abc</", &stops), 3, "`</` 是前缀 ⇒ 回吐");
        assert_eq!(safe_end("abc</to", &stops), 3);
        assert_eq!(safe_end("abc", &stops), 3, "无前缀 ⇒ 原样");
        assert_eq!(safe_end("abc<x", &stops), 5, "`<x` 不是前缀 ⇒ 放行");
        // 完整串不算"部分前缀":命中由调用方的 `find` 单独判定(本函数只
        // 负责把**半截**留住)。故此处 = 全长
        assert_eq!(safe_end("abc</tool_call>", &stops), 15);
    }

    /// 多串 / 长串前缀关系(短串是长串前缀时按最长回吐)
    #[test]
    fn holdback_multiple_stops() {
        let stops = s(&["\n\n", "STOP", "<|im_end|>"]);
        assert_eq!(safe_end("hi\n", &stops), 2);
        assert_eq!(safe_end("hi<|im_e", &stops), 2);
        assert_eq!(safe_end("hiSTO", &stops), 2);
        assert_eq!(safe_end("hi XX", &stops), 5);
    }

    /// floor_boundary:绝不切进多字节字符中间(中文/emoji)
    #[test]
    fn boundary_is_char_safe() {
        assert_eq!(floor_boundary("中文", 1), 0);
        assert_eq!(floor_boundary("中文", 3), 3);
        assert_eq!(floor_boundary("中文", 99), 6);
        let e = "a😀b";
        assert_eq!(floor_boundary(e, 2), 1, "emoji 中间回退到 1");
        assert_eq!(floor_boundary(e, 5), 5);
    }
}
