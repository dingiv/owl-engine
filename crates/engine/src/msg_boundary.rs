//! 消息边界快照点(纯 host 计算;2026-10-24)。
//!
//! 目的:agent 请求的历史前缀往往在**消息结束**处分叉(system → 不同 user,
//! tool 结果 → 下一次工具调用),而这些点通常不落在 `prefill_chunk` 末尾。
//! 这里把「消息开启标记」在 prompt token 流中的位置翻成**页对齐的 GDN 快照点**,
//! 供 `prefill_step_chunk_msg` 切块与 `capture_gdn_snap` 的 `snap_keep` 共用。
//!
//! **不写死分割串**:`open_ids` 是调用方传入的 token 序列 ——
//! 来源 = `[pool].snap_msg_delim` 配置(非空)或模型的
//! `ChatFormat.im_open`(Qwen 系 `<|im_start|>`)。函数本身只认 token 序列;
//! 换模板 = 换配置/模型声明,不改这里。
//!
//! 快照 key 的粒度是**页**(32 token),所以边界只能取 `floor(end/page)*page`;
//! 损失 < 1 页,换来 `reuse_face` 能命中。`min_gap` 控制后续边界稀疏度:
//! 第一条边界永远保留(通常是 system 结束),之后距上一条不足 `min_gap` 的跳过,
//! 避免 10K 多消息场景一次拍出几十张 152 MiB 的 27B 快照。

/// 从 prompt token 流中提取**页对齐的消息结束点**。
///
/// 语义:
/// - 每个 `open_ids` 出现处的前一 token 是消息边界候选（开启标记 = 上一条
///   消息结束；关闭标记 = 本条消息结束）；
/// - 页下界为 0 的候选（首条消息之前）自然丢弃；
/// - `end` 向下取整到页边界(快照 key 的粒度);
/// - 第一条边界无条件保留;后续边界要求距上一条 `>= min_gap` token;
/// - 同一页内的多个边界去重。
///
/// `open_ids` 为空 / `page == 0` ⇒ 返回空(调用方回退到旧的粗锚点策略)。
pub(crate) fn message_snap_points(
    prompt_ids: &[u32],
    open_ids: &[u32],
    page: usize,
    min_gap: usize,
) -> Vec<usize> {
    if page == 0 || open_ids.is_empty() || prompt_ids.len() < open_ids.len() {
        return Vec::new();
    }
    let mut out: Vec<usize> = Vec::new();
    for i in 0..=prompt_ids.len() - open_ids.len() {
        if &prompt_ids[i..i + open_ids.len()] != open_ids {
            continue;
        }
        let boundary = i / page * page;
        if boundary == 0 || out.last().copied() == Some(boundary) {
            continue;
        }
        let far_enough = match out.last().copied() {
            None => true,
            Some(last) => boundary.saturating_sub(last) >= min_gap,
        };
        if far_enough {
            out.push(boundary);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::message_snap_points;

    /// 多 token 分割串 + 页对齐 + 同页去重。
    #[test]
    fn scans_multi_token_delimiters_and_page_aligns() {
        // 页 = 4;open 在 2..4、6..8、10..12、14..16。
        // 结束点 = 2/4→0(丢)、6/4→4、10/4→8、14/4→12。
        let prompt = [
            9, 9, 1, 2, 10, 11, 1, 2, 20, 21, 1, 2, 30, 31, 1, 2, 40,
        ];
        let open = [1, 2];
        assert_eq!(message_snap_points(&prompt, &open, 4, 0), vec![4, 8, 12]);
    }

    /// 第一条边界永远保留;后续按 min_gap 稀疏。
    #[test]
    fn first_boundary_survives_min_gap() {
        let prompt = [
            9, 9, 1, 2, 10, 11, 1, 2, 20, 21, 1, 2, 30, 31, 1, 2, 40,
        ];
        let open = [1, 2];
        assert_eq!(message_snap_points(&prompt, &open, 4, 4), vec![4, 8, 12]);
        assert_eq!(message_snap_points(&prompt, &open, 4, 5), vec![4, 12]);
        assert_eq!(message_snap_points(&prompt, &open, 4, 9), vec![4]);
    }

    /// 标记不一定在 prompt 开头：首个页下界非 0 的标记也要算边界
    /// （因此 `snap_msg_delim` 也可以配成“消息关闭标记”）。
    #[test]
    fn first_nonzero_marker_is_a_boundary() {
        // 页 = 4;标记在 4、10 ⇒ 页下界 4、8。
        let prompt = [9, 9, 9, 9, 1, 0, 9, 9, 9, 9, 1, 0];
        assert_eq!(message_snap_points(&prompt, &[1], 4, 0), vec![4, 8]);
    }

    /// 同页多个边界只保留一个;空分割串/页宽 0 = 关闭。
    #[test]
    fn dedups_same_page_and_disables_cleanly() {
        // 页 = 8;标记在 9、12、14 ⇒ 三者的页下界都是 8,去重后只留一个。
        let mut prompt = vec![9u32; 20];
        prompt[9] = 1;
        prompt[12] = 1;
        prompt[14] = 1;
        assert_eq!(message_snap_points(&prompt, &[1], 8, 0), vec![8]);
        assert!(message_snap_points(&prompt, &[], 8, 0).is_empty());
        assert!(message_snap_points(&prompt, &[1], 0, 0).is_empty());
    }
}
