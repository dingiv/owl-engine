//! DFlash2 lookup-augmented drafting(LABD)—— 宿主侧后缀查找 + 草稿头融合。
//!
//! 语义锚 = vLLM `v1/worker/gpu/spec_decode/dflash2/lookup.py`(2026-10-22):
//! 把「当前 token 历史的后缀」在历史里最近一次出现的位置找出来,用那次出现
//! **后面跟着的 token** 覆盖草稿头。与 vLLM 的差异:
//!
//! - 这里只做 **head**(草稿器自己的 7 个位置),不做 tail 扩展(verify 图
//!   仍是 depth+1 定形;tail 需要更大的 verify 桶,另案);
//! - 宿主侧纯函数扫描(会话 token 本来就在 host),不新增 CUDA kernel;
//! - greedy spec 路径无需 point-mass/rejection 重写(草稿仍由 target verify)。
//!
//! 匹配规则(与 vLLM kernel 同式):
//! - 候选 = 历史中任意结束位 `e < total-1`,比较「以 e 结尾的 nmin 个 token」
//!   与「以 total-1 结尾的 nmin 个 token」;命中后向更早扩展,最多 nmax 个;
//! - 取最长匹配;等长取 e 更大(更近);
//! - 允许候选与后缀重叠(周期重复,如 `- item` 每行);
//! - 返回 `min(k, total-1-e)` 个后继 token(不足 k 个就只覆盖前 valid 个)。
//!
//! 融合规则(与 vLLM `_fuse_draft_kernel` 的 head 分支同式):
//! - `match_len >= nstrong` ⇒ 直接覆盖;
//! - 或 `match_len >= nmin && agree >= agree_min` ⇒ 覆盖(agree = 草稿与
//!   lookup 的前缀一致长度,按 valid 截断);
//! - 否则保留草稿。默认 `nstrong = nmin = 6`、`agree_min = 0` ⇒ 任意
//!   6-token 匹配都覆盖(与 vLLM 缺省一致)。

/// lookup 阈值(从 [`crate::spec_runtime::SpecRuntime`] 快照,避免热路径借 self)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LookupCfg {
    pub nmin: usize,
    pub nmax: usize,
    pub nstrong: usize,
    pub agree: usize,
}

impl LookupCfg {
    pub(crate) fn from_runtime(rt: &crate::spec_runtime::SpecRuntime) -> Self {
        Self {
            nmin: rt.lookup_nmin,
            nmax: rt.lookup_nmax,
            nstrong: rt.lookup_nstrong,
            agree: rt.lookup_agree,
        }
    }
}

/// 一次后缀查找的产物。
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LookupProposal {
    /// 匹配点之后的 token(长度 = valid)
    pub tokens: Vec<u32>,
    /// 匹配到的后缀长度(nmin..=nmax)
    pub match_len: usize,
    /// 实际可供覆盖的 token 数(<= k;匹配点太靠近末尾时不足 k)
    pub valid: usize,
}

/// 在 `hist` 里找「当前后缀」的最近一次出现,返回其后继 token。
///
/// `hist` 必须**包含当前最后喂入的 token**(即待预测位置的前一个 token);
/// `k` = 草稿头宽度(通常 7)。无匹配返回 `None`。
pub(crate) fn suffix_lookup(hist: &[u32], k: usize, cfg: LookupCfg) -> Option<LookupProposal> {
    let total = hist.len();
    // 与 vLLM 同门:至少要有 nmin 个后缀 + 1 个候选位。
    if k == 0 || cfg.nmin == 0 || cfg.nmax < cfg.nmin || total < cfg.nmin + 2 {
        return None;
    }
    let end_of_suffix = total - 1;
    let lo = cfg.nmin - 1;
    let suffix = &hist[end_of_suffix + 1 - cfg.nmin..=end_of_suffix];
    let mut best: Option<(usize, usize)> = None; // (match_len, end)

    // 从最近到最老扫候选结束位;等长时保留先遇到的(更近)。
    for e in (lo..end_of_suffix).rev() {
        let cand = &hist[e + 1 - cfg.nmin..=e];
        if cand != suffix {
            continue;
        }
        let mut n = cfg.nmin;
        while n < cfg.nmax {
            let (Some(ci), Some(si)) = (e.checked_sub(n), end_of_suffix.checked_sub(n)) else {
                break;
            };
            if hist[ci] != hist[si] {
                break;
            }
            n += 1;
        }
        let better = match best {
            None => true,
            Some((bn, be)) => n > bn || (n == bn && e > be),
        };
        if better {
            best = Some((n, e));
        }
    }

    let (match_len, end) = best?;
    // vLLM: valid = min(k, end_of_suffix - end);后继 token 从 end+1 起。
    let valid = k.min(end_of_suffix - end);
    if valid == 0 {
        return None;
    }
    let tokens = hist[end + 1..end + 1 + valid].to_vec();
    Some(LookupProposal {
        tokens,
        match_len,
        valid,
    })
}

/// 把 lookup 提案融合进草稿头(in-place)。返回是否发生了覆盖。
///
/// 只处理 head:调用方传入的 `drafts` 就是草稿器自己的 7 个位置。
pub(crate) fn fuse_lookup(
    drafts: &mut [u32],
    proposal: &LookupProposal,
    cfg: LookupCfg,
) -> bool {
    let valid = proposal
        .valid
        .min(proposal.tokens.len())
        .min(drafts.len());
    if valid == 0 {
        return false;
    }
    // agree = 草稿与 lookup 的前缀一致长度(按 valid 截断;同 vLLM)。
    let agree = (0..valid)
        .take_while(|&i| drafts[i] == proposal.tokens[i])
        .count();
    let take = proposal.match_len >= cfg.nstrong
        || (proposal.match_len >= cfg.nmin && agree >= cfg.agree);
    if !take {
        return false;
    }
    drafts[..valid].copy_from_slice(&proposal.tokens[..valid]);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(nmin: usize, nmax: usize) -> LookupCfg {
        LookupCfg {
            nmin,
            nmax,
            nstrong: nmin,
            agree: 0,
        }
    }

    #[test]
    fn repeats_earlier_continuation() {
        // 当前后缀 [A,B,C,D] 在历史前部出现过,后面跟着 X,Y,Z。
        let a = 10;
        let b = 20;
        let c = 30;
        let d = 40;
        let x = 50;
        let y = 60;
        let z = 70;
        let hist = [a, b, c, d, x, y, z, a, b, c, d];
        let p = suffix_lookup(&hist, 7, cfg(4, 8)).expect("命中");
        assert_eq!(p.match_len, 4);
        assert_eq!(p.valid, 7);
        assert_eq!(p.tokens, vec![x, y, z, a, b, c, d]);
    }

    #[test]
    fn allows_overlap_periodic() {
        // 周期 3 的重复;候选与后缀重叠,仍应提出 [1,2,3]。
        let hist = [1, 2, 3, 1, 2, 3, 1, 2, 3];
        let p = suffix_lookup(&hist, 4, cfg(3, 6)).expect("周期命中");
        assert!(p.match_len >= 5, "应扩到 >=5,得 {}", p.match_len);
        assert_eq!(p.valid, 3);
        assert_eq!(p.tokens, vec![1, 2, 3]);
    }

    #[test]
    fn no_match_returns_none() {
        let hist = [1, 2, 3, 4, 5, 6];
        assert!(suffix_lookup(&hist, 7, cfg(3, 8)).is_none());
        // 太短(不足 nmin+2)也拒。
        assert!(suffix_lookup(&[1, 2, 3, 4], 7, cfg(3, 8)).is_none());
    }

    #[test]
    fn prefers_longest_then_recent() {
        // 当前后缀 = [P,Q,A,B,C,D](末 6 个);老候选 6-token,近候选 4-token。
        let hist = [
            1, 2, 3, 4, 5, 6, 9, 3, 4, 5, 6, 8, 1, 2, 3, 4, 5, 6,
        ];
        let p = suffix_lookup(&hist, 7, cfg(4, 8)).expect("命中");
        assert_eq!(p.match_len, 6);
        assert_eq!(p.valid, 7);
        assert_eq!(p.tokens, vec![9, 3, 4, 5, 6, 8, 1]);
    }

    #[test]
    fn fuse_replaces_when_strong_or_agreeing() {
        let proposal = LookupProposal {
            tokens: vec![7, 8, 9],
            match_len: 6,
            valid: 3,
        };
        let mut drafts = vec![1, 2, 3, 4];
        assert!(fuse_lookup(&mut drafts, &proposal, cfg(6, 12)));
        assert_eq!(drafts, vec![7, 8, 9, 4]);

        // 弱匹配(低于 nmin)不覆盖。
        let weak = LookupProposal {
            tokens: vec![7, 8],
            match_len: 3,
            valid: 2,
        };
        let mut drafts = vec![1, 2, 3, 4];
        assert!(!fuse_lookup(&mut drafts, &weak, cfg(6, 12)));
        assert_eq!(drafts, vec![1, 2, 3, 4]);
    }

    #[test]
    fn fuse_short_match_requires_agreement() {
        let proposal = LookupProposal {
            tokens: vec![1, 2, 9],
            match_len: 6,
            valid: 3,
        };
        let mut cfg = cfg(6, 12);
        cfg.nstrong = 12;
        cfg.agree = 3;
        // 草稿前两位与 lookup 一致 => agree=2 < 3,不覆盖。
        let mut drafts = vec![1, 2, 3, 4];
        assert!(!fuse_lookup(&mut drafts, &proposal, cfg));
        assert_eq!(drafts, vec![1, 2, 3, 4]);
        // 三位全一致 => 覆盖。
        let mut drafts = vec![1, 2, 9, 4];
        assert!(fuse_lookup(&mut drafts, &proposal, cfg));
        assert_eq!(drafts, vec![1, 2, 9, 4]);
    }

    #[test]
    fn valid_caps_at_history_tail() {
        // 候选结束位 e=5、历史末位=8 => 只余 3 个后继 token。
        let hist = [9, 3, 1, 2, 3, 1, 2, 3, 1];
        let p = suffix_lookup(&hist, 7, cfg(4, 8)).expect("命中");
        assert_eq!(p.match_len, 5);
        assert_eq!(p.valid, 3);
        assert_eq!(p.tokens, vec![2, 3, 1]);
    }
}
