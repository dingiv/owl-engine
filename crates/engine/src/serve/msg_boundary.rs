//! 消息边界/快照布局（纯 host 计算；2026-11-18 v4）。
//!
//! 目的：agent 请求的历史前缀往往在**消息结束**处分叉（system → 不同 user，
//! tool 结果 → 下一次工具调用），而这些点通常不落在 `prefill_chunk` 末尾。
//! 这里把「消息开启标记」在 prompt token 流中的位置翻成**页对齐的 GDN 快照点**，
//! 再按 v4 规则切成 `Tail`（最后 12 张连续）与 `History`（前缀 18 张均匀）。
//!
//! **不写死分割串**：`open_ids` 是调用方传入的 token 序列 ——
//! 来源 = `[store].snap_msg_delim` 配置（非空）或模型的
//! `ChatFormat.im_open`（Qwen 系 `<|im_start|>`）。函数本身只认 token 序列；
//! 换模板 = 换配置/模型声明，不改这里。
//!
//! 快照 key 的粒度是**页**（32 token），所以边界只能取 `floor(end/page)*page`；
//! 损失 < 1 页，换来 `reuse_face` 能命中。

/// 从 prompt token 流中提取**全部页对齐的消息结束候选**。
///
/// 语义：
/// - 每个 `open_ids` 出现处的前一 token 是消息边界候选（开启标记 = 上一条
///   消息结束；关闭标记 = 本条消息结束）；
/// - 页下界为 0 的候选（首条消息之前）自然丢弃；
/// - `end` 向下取整到页边界（快照 key 的粒度）；
/// - 同一页内的多个边界去重；
/// - **不再按 `min_gap/recent` 过滤**：保留哪些点由
///   [`plan_snap_layout`] 的 Tail/History 布局决定。
///
/// `open_ids` 为空 / `page == 0` ⇒ 返回空（调用方回退到粗锚点策略）。
pub(crate) fn message_snap_points(prompt_ids: &[u32], open_ids: &[u32], page: usize) -> Vec<usize> {
    if page == 0 || open_ids.is_empty() || prompt_ids.len() < open_ids.len() {
        return Vec::new();
    }
    let mut all: Vec<usize> = Vec::new();
    for i in 0..=prompt_ids.len() - open_ids.len() {
        if &prompt_ids[i..i + open_ids.len()] != open_ids {
            continue;
        }
        let boundary = i / page * page;
        if boundary == 0 || all.last().copied() == Some(boundary) {
            continue;
        }
        all.push(boundary);
    }
    all
}

/// **历史点均匀选点**（v4；纯函数）。
///
/// 输入按 token offset 升序（同页去重由上游负责；本函数仍会排序/去重兜底）。
/// 选 `target` 个点，使首尾覆盖、按 token 距离尽量均匀。
///
/// `target == 1` 时选最后一个历史点（离 Tail 最近，深回撤收益最大）；
/// `target >= len` 时全选。
pub(crate) fn uniform_select(points: &[usize], target: usize) -> Vec<usize> {
    if target == 0 || points.is_empty() {
        return Vec::new();
    }
    let mut cands = points.to_vec();
    cands.sort_unstable();
    cands.dedup();
    if cands.len() <= target {
        return cands;
    }
    if target == 1 {
        return vec![*cands.last().expect("非空已判")];
    }
    let last = *cands.last().expect("非空已判") as u128;
    let denom = (target - 1) as u128;
    let mut used = vec![false; cands.len()];
    let mut out = Vec::with_capacity(target);
    for j in 0..target {
        let want = (last * j as u128 / denom) as usize;
        let mut best: Option<(usize, usize)> = None; // (距离, 下标)
        for (i, &p) in cands.iter().enumerate() {
            if used[i] {
                continue;
            }
            let dist = p.abs_diff(want);
            let better = match best {
                None => true,
                Some((bd, bi)) => dist < bd || (dist == bd && i < bi),
            };
            if better {
                best = Some((dist, i));
            }
        }
        let (_, i) = best.expect("target < len ⇒ 必有未选点");
        used[i] = true;
        out.push(cands[i]);
    }
    out.sort_unstable();
    out
}

/// **v4 快照布局**：Tail 连续尾段 + History 均匀前缀。
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct SnapLayout {
    /// 目标快照点（升序；Tail + History）。
    pub points: Vec<usize>,
    /// 与 `points` 同序的角色。
    pub roles: Vec<owl_store::SnapRole>,
}

/// 计算 Tail/History 布局（纯函数）。
///
/// - `cap == 0`：旧行为，不压缩候选；最后 `tail_count` 个为 Tail。
/// - `cap > 0`：总点数 ≤ cap，最后 `min(tail_count, cap)` 个连续点为 Tail；
///   Tail 之前的候选按 token 位置均匀选 `cap - tail_n` 个 History。
/// - `tail_count > cap`：Tail 只保留最后 cap 张，History 为空。
pub(crate) fn plan_snap_layout(candidates: &[usize], tail_count: usize, cap: usize) -> SnapLayout {
    use owl_store::SnapRole;
    let mut cands = candidates.to_vec();
    cands.sort_unstable();
    cands.dedup();
    if cands.is_empty() {
        return SnapLayout::default();
    }
    let (points, tail_n) = if cap == 0 {
        let tail_n = tail_count.min(cands.len());
        (cands, tail_n)
    } else {
        let tail_n = tail_count.min(cap).min(cands.len());
        let history_n = cap.saturating_sub(tail_n);
        let split = cands.len() - tail_n;
        let mut points = uniform_select(&cands[..split], history_n);
        points.extend_from_slice(&cands[split..]);
        (points, tail_n)
    };
    let history_n = points.len().saturating_sub(tail_n);
    let mut roles = vec![SnapRole::History; history_n];
    roles.resize(points.len(), SnapRole::Tail);
    SnapLayout { points, roles }
}

#[cfg(test)]
mod tests {
    use super::message_snap_points;

    /// 多 token 分割串 + 页对齐 + 同页去重。
    #[test]
    fn scans_multi_token_delimiters_and_page_aligns() {
        // 页 = 4;open 在 2..4、6..8、10..12、14..16。
        // 结束点 = 2/4→0(丢)、6/4→4、10/4→8、14/4→12。
        let prompt = [9, 9, 1, 2, 10, 11, 1, 2, 20, 21, 1, 2, 30, 31, 1, 2, 40];
        let open = [1, 2];
        assert_eq!(message_snap_points(&prompt, &open, 4), vec![4, 8, 12]);
    }

    /// 标记不一定在 prompt 开头：首个页下界非 0 的标记也要算边界
    /// （因此 `snap_msg_delim` 也可以配成“消息关闭标记”）。
    #[test]
    fn first_nonzero_marker_is_a_boundary() {
        // 页 = 4;标记在 4、10 ⇒ 页下界 4、8。
        let prompt = [9, 9, 9, 9, 1, 0, 9, 9, 9, 9, 1, 0];
        assert_eq!(message_snap_points(&prompt, &[1], 4), vec![4, 8]);
    }

    /// 均匀选点：数量、首尾、升序、无重复。
    #[test]
    fn uniform_select_covers_ends_and_count() {
        let pts: Vec<usize> = (0..100).map(|i| i * 32).collect();
        for (m, want_n) in [(0, 0), (1, 1), (18, 18), (100, 100), (101, 100)] {
            let out = super::uniform_select(&pts, m);
            assert_eq!(out.len(), want_n, "m={m}");
            assert!(out.windows(2).all(|w| w[0] < w[1]), "升序无重复: {out:?}");
            if want_n > 1 {
                assert_eq!(out[0], pts[0], "首点覆盖");
            }
            if want_n > 0 {
                assert_eq!(*out.last().unwrap(), *pts.last().unwrap(), "尾点覆盖");
            }
        }
        assert_eq!(super::uniform_select(&pts, 1), vec![*pts.last().unwrap()]);
        assert!(super::uniform_select(&[], 3).is_empty());
    }

    /// 删除/缩容后重算仍均匀：最大 gap 有界，不产生局部空洞。
    #[test]
    fn uniform_select_keeps_gap_bounded_after_shrink() {
        let pts: Vec<usize> = (0..100).map(|i| i * 32).collect();
        for m in [18, 17, 12, 6] {
            let out = super::uniform_select(&pts, m);
            let max_gap = out.windows(2).map(|w| w[1] - w[0]).max().unwrap();
            let avg = pts.last().unwrap() / (m - 1);
            assert!(
                max_gap <= avg * 2,
                "m={m} max_gap={max_gap} avg={avg} out={out:?}"
            );
        }
    }

    /// v4 布局：最后 12 张 Tail，前面 18 张均匀 History。
    #[test]
    fn layout_tail12_history18() {
        let pts: Vec<usize> = (0..100).map(|i| i * 32).collect();
        let layout = super::plan_snap_layout(&pts, 12, 30);
        assert_eq!(layout.points.len(), 30);
        assert_eq!(
            layout.points[18..],
            pts[pts.len() - 12..],
            "Tail 必须是最后 12 个连续点"
        );
        assert_eq!(layout.roles[..18], vec![owl_store::SnapRole::History; 18]);
        assert_eq!(layout.roles[18..], vec![owl_store::SnapRole::Tail; 12]);
        assert!(layout.points[..18].windows(2).all(|w| w[0] < w[1]));
        assert_eq!(layout.points[0], pts[0], "History 首点覆盖");
        assert!(
            layout.points[..18].last().unwrap() < &layout.points[18],
            "History 与 Tail 不重叠"
        );
    }

    /// cap < tail：Tail 夹紧到 cap，History 为空。
    #[test]
    fn layout_clamps_tail_below_cap() {
        let pts: Vec<usize> = (0..20).map(|i| i * 32).collect();
        let layout = super::plan_snap_layout(&pts, 12, 5);
        assert_eq!(layout.points, pts[15..].to_vec());
        assert_eq!(layout.roles, vec![owl_store::SnapRole::Tail; 5]);
    }

    /// cap=0：不压缩，全部保留；最后 tail_count 为 Tail。
    #[test]
    fn layout_cap_zero_keeps_all_candidates() {
        let pts: Vec<usize> = (0..40).map(|i| i * 32).collect();
        let layout = super::plan_snap_layout(&pts, 12, 0);
        assert_eq!(layout.points, pts);
        assert_eq!(layout.roles[..28], vec![owl_store::SnapRole::History; 28]);
        assert_eq!(layout.roles[28..], vec![owl_store::SnapRole::Tail; 12]);
    }

    /// 候选不足 cap：全选，尾部仍标 Tail。
    #[test]
    fn layout_small_candidate_set_is_all_tail() {
        let pts = vec![32, 64, 96];
        let layout = super::plan_snap_layout(&pts, 12, 30);
        assert_eq!(layout.points, pts);
        assert_eq!(layout.roles, vec![owl_store::SnapRole::Tail; 3]);
    }
}
