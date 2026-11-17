//! **首 token 门（TTFT gate）** —— 按「新增 token」预估首 token 期限，超时强停。
//!
//! 为什么归引擎：prefill 卡死/工作区 OOM 会让**整个泵**停摆（客户端只看到超时），
//! 而“这个 turn 还要多久”只有引擎自己知道（新增 token = prompt − 已命中前缀）。
//! 服务面拿不到设备侧进度，只能拍一个更大的 HTTP 超时。
//!
//! 口径（2026-11-16 用户裁决）：
//! - 预算 = `BASE + new_tokens × MS_PER_NEW_TOKEN`，夹到 `[MIN, MAX]`；
//! - 起点 = `ActiveTurn.started_ms`（准入起跑；排队由准入姿态管，不计入）；
//! - 到点仍无 `first_token_ms` ⇒ 引擎强停该 turn（与客户端 cancel 同一条清理
//!   路径），日志点名 turn / 新增 token / 预算 / 实等时长。
//!
//! 常数依据：本机 3080 TP2 实测 prefill 1473–2344 tok/s（193k / 44k prompt）
//! ⇒ `BASE=2000ms + 1ms/token` 约等于 1000 tok/s 的下限，健康路径不误杀；
//! `MAX=240s` 低于 pi 的 300s 请求超时（否则客户端先断，引擎拿不到诊断窗口）。

pub(crate) const BASE_MS: u64 = 2_000;
pub(crate) const MS_PER_NEW_TOKEN: u64 = 1;
pub(crate) const MIN_MS: u64 = 5_000;
pub(crate) const MAX_MS: u64 = 240_000;

/// 新增 token（真正要算的量）= prompt − 已命中前缀。
pub(crate) fn new_tokens(prompt_tokens: usize, cached_tokens: usize) -> usize {
    prompt_tokens.saturating_sub(cached_tokens)
}

/// 首 token 预算（毫秒）。
pub(crate) fn budget_ms(prompt_tokens: usize, cached_tokens: usize) -> u64 {
    let new = new_tokens(prompt_tokens, cached_tokens) as u64;
    BASE_MS
        .saturating_add(new.saturating_mul(MS_PER_NEW_TOKEN))
        .clamp(MIN_MS, MAX_MS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tokens_is_prompt_minus_cached() {
        assert_eq!(new_tokens(1000, 200), 800);
        assert_eq!(new_tokens(1000, 1000), 0);
        assert_eq!(new_tokens(1000, 1200), 0, "缓存数不许把新增算成负数");
    }

    #[test]
    fn budget_is_monotone_and_clamped() {
        assert_eq!(budget_ms(0, 0), MIN_MS);
        assert_eq!(budget_ms(1_000, 0), MIN_MS, "小 prompt 也保底 MIN");
        assert_eq!(budget_ms(28_000, 0), 30_000);
        assert_eq!(budget_ms(193_000, 0), 195_000);
        assert_eq!(budget_ms(400_000, 0), MAX_MS, "超长 prompt 夹到 MAX");
        let mut prev = 0;
        for n in [0usize, 1_000, 5_000, 20_000, 100_000, 200_000, 500_000] {
            let b = budget_ms(n, 0);
            assert!(b >= prev, "新增 token 增加不许让预算下降（{prev} → {b}）");
            prev = b;
        }
    }

    #[test]
    fn cached_prefix_shortens_budget() {
        // 44k prompt、命中 40k ⇒ 只按 4k 新增算（12k 预算，不是 46k）
        assert_eq!(budget_ms(44_000, 40_000), 6_000);
    }
}
