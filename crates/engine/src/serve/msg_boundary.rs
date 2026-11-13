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
//! 损失 < 1 页,换来 `reuse_face` 能命中。
//!
//! **密度 = 稠密尾部 + 稀疏中段**(2026-11-12):
//! - 最近 `recent` 条消息边界**无条件保留**(dense tail)——前端追加/回撤的
//!   TTFT 敏感区,短工具结果不能被 `min_gap` 跳过;
//! - 更早的边界按 `min_gap` 稀疏(分叉点);
//! - 第一条边界(通常 system 结束)保持无条件(旧行为)。
//! 这样 10K 多消息场景不会一次拍出几十张 152 MiB 快照,同时最近几条
//! 100% 有锚点。

/// 从 prompt token 流中提取**页对齐的消息结束点**。
///
/// 语义:
/// - 每个 `open_ids` 出现处的前一 token 是消息边界候选（开启标记 = 上一条
///   消息结束；关闭标记 = 本条消息结束）；
/// - 页下界为 0 的候选（首条消息之前）自然丢弃；
/// - `end` 向下取整到页边界(快照 key 的粒度);
/// - **最近 `recent` 条边界无条件保留**（dense tail；`0` = 关闭）；
/// - 更早的边界要求距上一条**已保留**边界 `>= min_gap` token；
/// - 第一条边界无条件保留（旧行为）；
/// - 同一页内的多个边界去重。
///
/// `open_ids` 为空 / `page == 0` ⇒ 返回空(调用方回退到旧的粗锚点策略)。
pub(crate) fn message_snap_points(
    prompt_ids: &[u32],
    open_ids: &[u32],
    page: usize,
    min_gap: usize,
    recent: usize,
) -> Vec<usize> {
    if page == 0 || open_ids.is_empty() || prompt_ids.len() < open_ids.len() {
        return Vec::new();
    }
    // 1) 全部候选（页对齐、同页去重、升序）。
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
    if all.is_empty() {
        return all;
    }
    // 2) 稠密尾部 + 稀疏中段。第一条始终保留（system 结束的跨会话锚点）。
    let dense_from = all.len().saturating_sub(recent);
    let mut out: Vec<usize> = Vec::with_capacity(dense_from.min(all.len()) + recent);
    let mut last_kept: Option<usize> = None;
    for (i, &boundary) in all.iter().enumerate() {
        let dense = i >= dense_from;
        let keep = i == 0
            || dense
            || match last_kept {
                None => true,
                Some(prev) => boundary.saturating_sub(prev) >= min_gap,
            };
        if keep {
            out.push(boundary);
            last_kept = Some(boundary);
        }
    }
    out
}

/// **淘汰优先级分类**(2026-11-12;不是 pin)。
///
/// 输入 = `message_snap_points` 选出的升序边界,输出同序的
/// [`owl_store::SnapClass`]:
/// - 最后 `dense_recent` 条 = `Dense`(最近 K 条,append/浅回撤的锚点);
/// - 每个二分点 `P/2^k` 取 **≤ 且最近** 的边界 = `Binary(k)`
///   (k 越大越老;淘汰时老节点先走);
/// - 其余 = `Extra`(优先淘汰)。
///
/// 这是**淘汰顺序**,不是保留集合:任何档位在空间不足时都会被逐出,
/// `Dense` 只是最后被选中。
pub(crate) fn classify_snap_points(
    points: &[usize],
    prompt_len: usize,
    dense_recent: usize,
) -> Vec<owl_store::SnapClass> {
    use owl_store::SnapClass;
    let mut classes = vec![SnapClass::Extra; points.len()];
    if points.is_empty() {
        return classes;
    }
    let dense_from = points.len().saturating_sub(dense_recent);
    for c in classes.iter_mut().skip(dense_from) {
        *c = SnapClass::Dense;
    }
    let mut k = 1u32;
    while k <= 32 {
        let target = prompt_len >> k;
        if target == 0 {
            break;
        }
        if let Some(i) = points.iter().rposition(|&b| b <= target) {
            if classes[i] == SnapClass::Extra {
                classes[i] = SnapClass::Binary(k as u8);
            }
        }
        k += 1;
    }
    classes
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
        assert_eq!(message_snap_points(&prompt, &open, 4, 0, 0), vec![4, 8, 12]);
    }

    /// 第一条边界永远保留;后续按 min_gap 稀疏。
    #[test]
    fn first_boundary_survives_min_gap() {
        let prompt = [
            9, 9, 1, 2, 10, 11, 1, 2, 20, 21, 1, 2, 30, 31, 1, 2, 40,
        ];
        let open = [1, 2];
        assert_eq!(message_snap_points(&prompt, &open, 4, 4, 0), vec![4, 8, 12]);
        assert_eq!(message_snap_points(&prompt, &open, 4, 5, 0), vec![4, 12]);
        assert_eq!(message_snap_points(&prompt, &open, 4, 9, 0), vec![4]);
    }

    /// 标记不一定在 prompt 开头：首个页下界非 0 的标记也要算边界
    /// （因此 `snap_msg_delim` 也可以配成“消息关闭标记”）。
    #[test]
    fn first_nonzero_marker_is_a_boundary() {
        // 页 = 4;标记在 4、10 ⇒ 页下界 4、8。
        let prompt = [9, 9, 9, 9, 1, 0, 9, 9, 9, 9, 1, 0];
        assert_eq!(message_snap_points(&prompt, &[1], 4, 0, 0), vec![4, 8]);
    }

    /// **稠密尾部**：最近 `recent` 条边界无条件保留，不受 min_gap 约束。
    #[test]
    fn dense_tail_keeps_last_k_regardless_of_gap() {
        // 页 = 4；open 在 2/6/10/14/18 ⇒ 边界 0(丢)/4/8/12/16。
        let prompt = [
            9, 9, 1, 2, 10, 11, 1, 2, 20, 21, 1, 2, 30, 31, 1, 2, 40, 41, 1, 2, 50,
        ];
        let open = [1, 2];
        // 旧行为：first + min_gap(100) ⇒ 只有 4。
        assert_eq!(message_snap_points(&prompt, &open, 4, 100, 0), vec![4]);
        // dense=1：first + 最后 1 条（16）；8/12 仍被 gap 跳过。
        assert_eq!(message_snap_points(&prompt, &open, 4, 100, 1), vec![4, 16]);
        // dense=2：first + 最后 2 条（12/16）；8 被跳过。
        assert_eq!(message_snap_points(&prompt, &open, 4, 100, 2), vec![4, 12, 16]);
        // dense=3：8/12/16 都是 dense ⇒ 全部保留。
        assert_eq!(
            message_snap_points(&prompt, &open, 4, 100, 3),
            vec![4, 8, 12, 16]
        );
        // dense >= 4：全部保留。
        assert_eq!(
            message_snap_points(&prompt, &open, 4, 100, 8),
            vec![4, 8, 12, 16]
        );
    }

    /// **淘汰优先级分类**：dense 尾部 + 二分节点 + extra。
    #[test]
    fn classifies_dense_tail_and_binary_nodes() {
        use owl_store::SnapClass;
        // P=17；points [4,8,12]；dense=1 ⇒ 12 Dense；
        // k=1 target=8 → 8 Binary(1)；k=2 target=4 → 4 Binary(2)。
        assert_eq!(
            super::classify_snap_points(&[4, 8, 12], 17, 1),
            vec![
                SnapClass::Binary(2),
                SnapClass::Binary(1),
                SnapClass::Dense
            ]
        );
        // dense=3 ⇒ 全部 Dense（dense 优先于 binary）。
        assert_eq!(
            super::classify_snap_points(&[4, 8, 12], 17, 3),
            vec![SnapClass::Dense, SnapClass::Dense, SnapClass::Dense]
        );
        // dense=0 ⇒ 12 落 Extra（没有二分点命中它）。
        assert_eq!(
            super::classify_snap_points(&[4, 8, 12], 17, 0),
            vec![
                SnapClass::Binary(2),
                SnapClass::Binary(1),
                SnapClass::Extra
            ]
        );
    }

    /// 同页多个边界只保留一个;空分割串/页宽 0 = 关闭。
    #[test]
    fn dedups_same_page_and_disables_cleanly() {
        // 页 = 8;标记在 9、12、14 ⇒ 三者的页下界都是 8,去重后只留一个。
        let mut prompt = vec![9u32; 20];
        prompt[9] = 1;
        prompt[12] = 1;
        prompt[14] = 1;
        assert_eq!(message_snap_points(&prompt, &[1], 8, 0, 0), vec![8]);
        assert!(message_snap_points(&prompt, &[], 8, 0, 0).is_empty());
        assert!(message_snap_points(&prompt, &[1], 0, 0, 0).is_empty());
    }
}
