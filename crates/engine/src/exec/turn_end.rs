//! turn 终局(abandon / complete / cancel)
use super::*;
use owl_iface::contract::ModelError;

/// 弃置 turn 时**是否发布已写入前缀**(`Release(Abort)` 的 H3 载荷选择)。
///
/// 两种弃置的账面可信度不同:
/// - 客户端中断(断流/取消):KV `[0, fed)` 是**算完的真内容** ⇒ 发布;
/// - 引擎错误:账面不可信 ⇒ 不发布(turn 账不可信，但**会话保留**)。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AbandonPublish {
    /// 把**已写入**的前缀写回会话账并交给 `Release(Abort)` 登记链身份。
    WrittenPrefix,
    /// 只释放资源,不动会话账。
    Skip,
}

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// 把当前游标的最后完整页加入活跃 turn 的布局点（幂等）。
    ///
    /// Tail/History 布局按 token offset 计算；turn 结束/中断时生成段末尾
    /// 可能尚未出现在 submit 的候选里，必须先把 `at` 纳入再拍 Tail。
    fn add_active_snap_point(&mut self, at: usize) -> Result<()> {
        if at == 0 {
            return Ok(());
        }
        let act = self.turns.active_mut()?;
        if let Err(pos) = act.msg_snaps.binary_search(&at) {
            act.msg_snaps.insert(pos, at);
            act.msg_snap_roles.insert(pos, owl_store::SnapRole::Tail);
            act.msg_snap_keys.insert(pos, None);
        }
        Ok(())
    }

    /// 按 v4 规则重算活跃 turn 的 Tail/History 布局，删除不在目标集合里的
    /// 旧持久键（当前正在拍的最新尾已加入布局，不会被删）。
    fn reconcile_active_snap_layout(&mut self) -> Result<usize> {
        let (old_points, old_keys, old_tail) = {
            let act = self.turns.active()?;
            (
                act.msg_snaps.clone(),
                act.msg_snap_keys.clone(),
                act.last_snap_at.zip(act.last_snap),
            )
        };
        let layout = crate::msg_boundary::plan_snap_layout(
            &old_points,
            self.boot.cfg.knobs.snap_tail,
            self.boot.cfg.knobs.snap_session_cap,
        );
        let mut new_keys =
            crate::serve::session::carry_msg_snap_keys(&old_points, &old_keys, &layout.points);
        if let Some((at, key)) = old_tail {
            if let Ok(i) = layout.points.binary_search(&at) {
                if new_keys[i].is_none() {
                    new_keys[i] = Some(key);
                }
            }
        }
        let mut reclassified = 0usize;
        for i in 0..new_keys.len() {
            let Some(key) = new_keys[i] else { continue };
            if self
                .bank
                .parts_mut()
                .rel
                .reclassify_snap(key, layout.roles[i])
            {
                reclassified += 1;
            } else {
                new_keys[i] = None;
            }
        }
        let keep: std::collections::BTreeSet<owl_store::SnapKey> =
            new_keys.iter().flatten().copied().collect();
        let mut drop_keys: std::collections::BTreeSet<owl_store::SnapKey> =
            std::collections::BTreeSet::new();
        for key in old_keys.iter().flatten().copied() {
            if !keep.contains(&key) {
                drop_keys.insert(key);
            }
        }
        if let Some((_, key)) = old_tail {
            if !keep.contains(&key) {
                drop_keys.insert(key);
            }
        }
        let mut dropped = 0usize;
        for key in drop_keys {
            if self.bank.parts().rel.state_locate(key).is_none() {
                continue;
            }
            self.drop_snap_asset_for_admit(key)?;
            dropped += 1;
        }
        // 最新实际在账的 Tail 才是尾锚点；布局中尚未拍到的尾部点不回填
        // `last_snap`（否则会指向已删/不存在的键）。
        let mut last_actual: Option<(usize, owl_store::SnapKey)> = None;
        for i in (0..new_keys.len()).rev() {
            if let Some(key) = new_keys[i] {
                last_actual = Some((layout.points[i], key));
                break;
            }
        }
        {
            let act = self.turns.active_mut()?;
            act.msg_snaps = layout.points;
            act.msg_snap_roles = layout.roles;
            act.msg_snap_keys = new_keys;
            act.last_snap = last_actual.map(|(_, key)| key);
            act.last_snap_at = last_actual.map(|(at, _)| at);
        }
        if reclassified > 0 || dropped > 0 {
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("snap.layout.recomputed", 1, file!(), line!());
                m.counter_add("snap.history.evicted", dropped as u64, file!(), line!());
            });
        }
        Ok(dropped)
    }

    /// **turn 末释放设备表**：表出账归还（引用减一）；历史 `tokens` 保留。
    ///
    /// 不在 turn 之间保留任何 block_table。所有块回到 store 缓存面，由
    /// 引用计数 + LRU/过期策略回收；下一轮可重新按内容命中，命中不了就重建。
    /// 这样不存在“空闲会话仍持有、store 不能动”的块。
    // Phase 3：H6 执行臂。唯一调用者是 `StoreHook::Release`；
    // engine 其它位置不得直呼。
    pub(crate) fn release_turn_table(&mut self, session_id: u64) -> Result<()> {
        // 表归还 + 会话级前缀账归零由 store 同屋完成（2026-11-15 下沉）。
        self.bank.release_turn_table(session_id)
    }

    pub async fn abandon(&mut self, err: String) -> TurnEvent {
        // **归属只从显式选择器取**（`schedule` 为本步选中的单 turn）；
        // 无选择器时才退回“活跃首”（批动作失败 = 整批失败，弃行首让泵
        // 继续 —— B5.1 的既有妥协）。⚠️ 启动失败**不得**走这里：那种 turn
        // 从没进过 `active`，猜中谁都错（见 `schedule` 的 `start_one` 失败臂）。
        let exec = self.turns.executing;
        let turn_id = exec
            .or_else(|| self.turns.active.first().map(|a| a.id))
            .unwrap_or(0);
        if exec.is_none() && !self.turns.active.is_empty() {
            owl_shared::log_warn!(
                "sched",
                "abandon 无执行选择器（批动作/循环头错误）⇒ 按活跃首弃置 turn {turn_id}"
            );
        }
        // B5.1:只弃**本步执行 turn**(批步错 = 该行失败;其余活跃行留场,
        // 由服务面继续泵 —— 全清会把无辜会话连坐)。执行选择器缺省 = 活跃首
        // (批动作失败时未选定)。
        if let Some(act) = self.turns.take_exec() {
            // 引擎错误：账面不可信 ⇒ **不发布**已写入前缀（会话保留，前缀守卫失配自会回退全量）
            self.abandon_active(act, turn_id, AbandonPublish::Skip)
                .await;
        }
        // 准入账同步(S3b §23.2):turn 生死 = 保留额生死
        self.close_admission(turn_id);
        // **孤儿收口恒开**（不只 `turn_id == 0`）：准入账里有、活跃集里没有的
        // 条目只可能来自启动半态，收掉它是幂等且安全的（在册 turn 不被动）；
        // 恒开 = 任何没被本函数归属到的半态都能自愈，而不是等下一次巧合同形。
        let live: Vec<u64> = self.turns.active.iter().map(|a| a.id).collect();
        match self.bank.parts_mut().rel.finish_orphan_turns(&live) {
            Ok(reps) if !reps.is_empty() => {
                for (t, r) in reps {
                    owl_shared::log_warn!(
                        "store",
                        "孤儿 turn {t} 已收口（准入={} Grant={} 行={:?}）",
                        r.admission,
                        r.grants,
                        r.row
                    );
                }
                if let Err(e) = self.sync_pool_account() {
                    owl_shared::log_warn!("store", "孤儿收口后账同步失败：{e}");
                }
            }
            Ok(_) => {}
            Err(e) => owl_shared::log_warn!("store", "孤儿 turn 收口失败：{e}"),
        }
        TurnEvent::Failed {
            turn: turn_id,
            err: err.into(),
        }
    }

    /// **按 turn 精确终结**(B5.1 服务面契约:合批档下"取消"必须指名 turn,
    /// 否则会误杀同批的别家会话)。
    ///
    /// - 活跃:走与 `abandon` 同一条清理(块链归还/还 GDN 行/N4 节点),只弃该行;
    /// - **排队中**:只出队 + 准入账收口 + 丢弃该 turn 的残留事件(不碰设备);
    /// - 不在账:`None`(调用方如实处理 —— 竞态下 turn 可能刚自然终了)。
    pub async fn cancel_turn(&mut self, turn: u64) -> Option<TurnEvent> {
        self.cancel_turn_reason(turn, "客户端取消").await
    }

    /// 同 [`Self::cancel_turn`]，但终态文案可归因（首 token 门 = "TTFT 超时"）。
    pub async fn cancel_turn_reason(&mut self, turn: u64, reason: &str) -> Option<TurnEvent> {
        // 残留事件先清(否则服务面会在 Failed 之后还收到该 turn 的 Token 帧)
        self.turns.pending_events.retain(|e| match e {
            TurnEvent::Prefill { turn: t, .. }
            | TurnEvent::Token { turn: t, .. }
            | TurnEvent::Completed { turn: t, .. }
            | TurnEvent::Failed { turn: t, .. } => *t != turn,
            TurnEvent::Idle => true,
        });
        // **中断在 prefill 中段 ⇒ 先把"已算完的边界"钉成快照**(2026-11-03)。
        // 必须在 `take_active` **之前**:补拍要用活跃 turn 的 GDN 行与位置。
        if self.turns.active_of(turn).is_some() {
            if let Err(e) = self.capture_abort_snap(turn).await {
                // 补拍失败不挡收摊(它只是"让已算完的前缀可复用"的加速面);
                // 但必须响亮 —— 否则下一轮会静默变成全量重算。
                owl_shared::log_warn!("snap", "中断补拍失败(turn {turn};下一轮可能全量重算): {e}");
            }
        }
        if let Some(act) = self.turns.take_active(turn) {
            // 客户端中断（断流/取消）：**已写入**的 KV 是真内容 ⇒ 发布链身份，
            // 否则下一个同前缀请求内容寻址命中 0 ⇒ 整段重算（见 `ledger_written`）
            self.abandon_active(act, turn, AbandonPublish::WrittenPrefix)
                .await;
            self.close_admission(turn);
            return Some(TurnEvent::Failed {
                turn,
                err: format!("{reason}(活跃 turn)").into(),
            });
        }
        if let Some(pos) = self.turns.queue.iter().position(|t| t.id == turn) {
            if let Some(t) = self.turns.queue.remove(pos) {
                // 排队 turn 未占设备资源(块链/GDN 行都在起 turn 时借)⇒ 只清账。
                // **会话保留**：它是 submit 时按内容身份建的持久账（F-18 的
                // "临时会话漏焚"已随 ephemeral 概念一起退休 —— 2026-11-16）。
                self.drop_queued_turn(&t);
                return Some(TurnEvent::Failed {
                    turn,
                    err: format!("{reason}(排队中)").into(),
                });
            }
        }
        None
    }

    /// **按会话终结**(服务面 `EngineReq::Abort` 的语义面):该会话的活跃 turn
    /// 优先,其次排队首;都没有 = `None`。
    ///
    /// 为什么按会话而不是按"执行 turn":合批档下活跃集有 B 个 turn,
    /// `take_exec()` 只认本步执行者 —— 客户端 A 断流时可能杀掉 B 的 turn。
    /// 服务面已按会话串行(每会话至多一个 turn 在场)⇒ 该会话的 turn 唯一。
    pub async fn cancel_session(&mut self, session: u64) -> Option<TurnEvent> {
        let turn = self
            .turns
            .active
            .iter()
            .find(|a| a.session_id == session)
            .map(|a| a.id)
            .or_else(|| {
                self.turns
                    .queue
                    .iter()
                    .find(|t| t.session_id == session)
                    .map(|t| t.id)
            })?;
        self.cancel_turn(turn).await
    }

    /// 单个活跃 turn 的弃置清理(块链归还 / 还 GDN 行 / N4 节点)。
    /// `turn_id` = 事件与准入口径的 turn id(与 `act.id` 同源,显式传入
    /// 以免调用方各自猜)。
    pub(crate) async fn abandon_active(
        &mut self,
        act: crate::turn::ActiveTurn,
        turn_id: u64,
        publish: AbandonPublish,
    ) {
        debug_assert_eq!(act.id, turn_id, "abandon_active: turn id 与 act 不同源");
        // **丢弃未 drain 的 DFlash tap 根块**（2026-11-13）：中断可能发生在
        // 首 token 之前，encode 还没补跑；根块不能随 turn 账被静默吊住。
        if let Err(e) = self.dev.discard_pending_dflash().await {
            owl_shared::log_warn!("dflash", "turn 中断:丢弃 pending tap 失败(继续收尾): {e}");
        }
        if let Some(p) = self.peer.as_mut() {
            if let Err(e) = p.discard_pending_dflash().await {
                owl_shared::log_warn!("dflash", "turn 中断(peer):丢弃 pending tap 失败: {e}");
            }
        }
        // 墙钟 EWMA（`Retry-After` 估算用；所有终局路径同源）
        self.turns
            .note_turn_ms(crate::running::now_ms().saturating_sub(act.started_ms));
        // **中断落账**(必须在 `Release(Abort)` 之前:链登记读会话账,且
        // 设备表此刻还没归还 —— 树要在这个窗口里加引用,内容才沉得住)
        if publish == AbandonPublish::WrittenPrefix {
            if let Err(e) = self.ledger_written(&act) {
                owl_shared::log_warn!("prefix", "中断落账失败(跳过链发布): {e}");
            }
        }
        // 草稿账随 act(TurnSpecState)销毁(生命周期收口)。
        // H6 / GDN 行 / 归属下行 / N4 全部由 `Release(Abort)` 统一执行；
        // 弃置路径失败只 WARN，不掩盖原始错误。
        if let Err(e) = self
            .store_hook(crate::store_hook::StoreHook::abort(
                act.id,
                act.session_id,
                act.gdn_slot,
                publish == AbandonPublish::WrittenPrefix,
            ))
            .await
        {
            owl_shared::log_warn!("store", "Release(Abort) 失败(继续收尾): {e}");
        }
    }

    /// **中断点补一张 Tail 快照**（v4：prefill 与 generation 都覆盖）。
    ///
    /// 判据：
    /// - `fed > 0`，且至少有一个完整页（快照 key 只能挂页边界）；
    /// - 前缀缓存与快照池都在场（否则快照没有消费者）；
    /// - 取 `floor(fed/page)*page` = 最后完整页边界；generation 中断同样适用。
    async fn capture_abort_snap(&mut self, turn: u64) -> Result<()> {
        let Some(act) = self.turns.active_of(turn) else {
            return Ok(());
        };
        let (sid, fed, prompt_len) = (act.session_id, act.fed, act.prompt_ids.len());
        let page = self.dev.pool.geom.page.max(1);
        let at = self
            .bank
            .parts()
            .sessions
            .get(sid)
            .map(|s| s.chain.blocks() * page)
            .unwrap_or(0);
        if at == 0 {
            return Ok(());
        }
        if !self.boot.cfg.knobs.prefix_cache || self.dev.pool.snap_capacity() == 0 {
            return Ok(());
        }
        self.add_active_snap_point(at)?;
        let captured = self
            .capture_snap_at_cursor(
                sid,
                owl_store::SnapDurability::Durable,
                owl_store::SnapRole::Tail,
            )
            .await?;
        self.reconcile_active_snap_layout()?;
        if captured {
            let tag = if fed >= prompt_len {
                "snap.capture.abort_generation"
            } else {
                "snap.capture.abort_prefill"
            };
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add(tag, 1, file!(), line!());
            });
        }
        Ok(())
    }

    /// **中断落账**(2026-11-03 前缀缓存命中修复):把**已确认的 prompt 前缀**
    /// 写回会话账。
    ///
    /// # 为什么必须落
    ///
    /// `Release(Abort)` 的链登记以**会话账**为 token 源
    /// (`register_chain_owned(&s.tokens, &s.block_table, ..)`)。客户端中断时
    /// `commit_turn` 不会跑 ⇒ 账还停在上一轮 ⇒ 本 turn 算出来的整段前缀
    /// **没有内容身份** ⇒ 下一个同前缀请求 `match_prefix` 命中 0 ⇒ 整段重算
    /// (2026-11-03 真机实测:5850 tok prompt 中断后再提交,`cache=32/5865`,
    /// prefill 5833 tok / 2.39 s —— 本应只 prefill 新增的 41 tok)。
    ///
    /// # 记账口径 = **已算完的 prompt 前缀**,不含生成段
    ///
    /// - `p = min(fed, prompt_len)`:`fed` 是 KV 推进指针(KV 有效域恒为
    ///   `[0, fed)`),prefill 每 chunk / decode 每步都推进它 ⇒ 中断在
    ///   prefill 中段也只登记真的算完的那一截;
    /// - **生成段故意不入账**:它没交付给客户端(中断即被前端丢弃),
    ///   而一旦写进账本,下一轮 `positions_after_splice` 会看到
    ///   "旧账尾巴 ≠ 新 prompt" ⇒ 位置账/复用面错位。
    /// - 位置账/图像段账**同源截短** —— 否则下一轮 `unpad`/leave-gap 会按
    ///   错坐标算(`apply_image_leave_gap` 的中途落账同律)。
    ///
    /// 链登记本身由 `Release(Abort)` 统一做(单一产地:H3 只在一处);
    /// 本函数只写账,不碰设备也不碰树。
    fn ledger_written(&mut self, act: &crate::turn::ActiveTurn) -> Result<()> {
        // 无块表(paged 关)= 无链身份可发布,写账只会让 `guard()` 更保守
        if !self.dev.pool.geom.paged {
            return Ok(());
        }
        let written = act.fed.min(act.prompt_ids.len());
        if written == 0 {
            return Ok(());
        }
        let generated = act.fed.saturating_sub(act.prompt_ids.len());
        let s = self.bank.parts_mut().sessions.session_mut(act.session_id)?;
        s.tokens.clear();
        s.tokens.extend_from_slice(&act.prompt_ids[..written]);
        // 内容身份的参考边界 = 已写入的 prompt 段（中断时 generation 尚未落账）
        // 历史长度唯一真源 = `tokens.len()`（2026-11-16 收口，无 cached_len）
        s.prompt_boundary = written;
        // 位置账:长度对不上 = 结构错(写错位置比不写更坏 ⇒ 清空后
        // leave-gap 自会按"位置账缺失"跳过,而不是拿错坐标算)
        if act.positions.len() >= written {
            s.set_positions(&act.positions[..written])?;
        } else {
            owl_shared::log_warn!(
                "prefix",
                "中断落账:位置账 {} < prompt 前缀 {} ⇒ 位置账清空(session={:#x} turn={})",
                act.positions.len(),
                written,
                act.session_id,
                act.id
            );
            s.positions.clear();
        }
        let (spans, pads) = crate::img_evict::ledger_prefix(&act.img_spans, &act.img_pads, written);
        s.set_img_spans(spans, pads);
        // **L1 资产：Tail 尾锚点 + 布局跨轮保留（v4）**。
        //
        // 布局点只保留 `<= written` 的 prompt 边界；生成段不入 token 账，
        // 但 generation abort 的最后页 Tail 仍通过 `tail_snap/tail_at`
        // 保留。下一轮若客户端带回这段文本，可按 offset 重新纳入布局。
        if act.last_snap.is_some() {
            s.tail_snap = act.last_snap;
            s.tail_at = act.last_snap_at;
        }
        let kept: Vec<(usize, owl_store::SnapRole, Option<owl_store::SnapKey>)> = act
            .msg_snaps
            .iter()
            .copied()
            .zip(act.msg_snap_roles.iter().copied())
            .zip(act.msg_snap_keys.iter().copied())
            .filter(|((b, _), _)| *b <= written)
            .map(|((b, role), key)| (b, role, key))
            .collect();
        s.msg_snaps = kept.iter().map(|(b, _, _)| *b).collect();
        s.msg_snap_roles = kept.iter().map(|(_, role, _)| *role).collect();
        s.msg_snap_keys = kept.into_iter().map(|(_, _, key)| key).collect();
        owl_shared::log_info!(
            "prefix",
            "中断落账:session={:#x} turn={} prompt 前缀 {} tok 入账(生成段 {generated} tok 不入缓存)⇒ 交给 Release(Abort) 发布链身份",
            act.session_id,
            act.id,
            written
        );
        Ok(())
    }

    /// **准入账收口**(S3b §23.2):turn 终了 ⇒ 还保留额 + 重算可动面。
    /// 账里没有该 turn(如已拒/已清)= 幂等(不报错也不铎）。
    pub(crate) fn close_admission(&mut self, turn_id: u64) {
        if turn_id == 0 {
            return;
        }
        // 准入状态机归还预留(幂等;complete/abandon/腿终局全走这里)
        self.turns.admission.note_finished(turn_id);
        // **store 单一围栏**：Grant + 准入账 + GDN 行一次归还（幂等）。
        // 不在这里逐项手还 —— 三本账同源，拆开就必然有半态（2026-11-06 真机）。
        match self.bank.parts_mut().rel.finish_turn(turn_id) {
            Ok(r) => {
                if r.admission || r.grants > 0 || r.row.is_some() {
                    owl_shared::log_debug!(
                        "store",
                        "turn {turn_id} 终局归还：准入={} Grant={} 行={:?}",
                        r.admission,
                        r.grants,
                        r.row
                    );
                }
            }
            Err(e) => {
                owl_shared::log_warn!("store", "turn {turn_id} 终局归还失败（继续收尾）：{e}")
            }
        }
        if let Err(e) = self.sync_pool_account() {
            owl_shared::log_warn!("store", "准入账同步失败：{e}");
        }
    }

    pub(crate) async fn complete(
        &mut self,
        reason: crate::turn::FinishReason,
    ) -> Result<TurnEvent> {
        // **Turn 末 Tail 尾锚点（v4）**：生成段最后一页边界默认是
        // Transient（最先淘汰）；这里把该 offset 纳入布局并升级为 Tail。
        // 只改角色，不 pin：空间不足时 Tail 仍可被逐出。
        if self.boot.cfg.knobs.prefix_cache && self.dev.pool.snap_capacity() > 0 {
            if let Ok(act) = self.turns.active() {
                let sid = act.session_id;
                let page = self.dev.pool.geom.page.max(1);
                let at = self
                    .bank
                    .parts()
                    .sessions
                    .get(sid)
                    .map(|s| s.chain.blocks() * page)
                    .unwrap_or(0);
                if at > 0 {
                    if let Err(e) = self.add_active_snap_point(at) {
                        owl_shared::log_warn!("snap", "turn 末布局点追加失败: {e}");
                    } else {
                        match self
                            .capture_snap_at_cursor(
                                sid,
                                owl_store::SnapDurability::Durable,
                                owl_store::SnapRole::Tail,
                            )
                            .await
                        {
                            Ok(true) => {
                                owl_shared::metrics::with_metrics_store(|m| {
                                    m.counter_add("snap.capture.complete", 1, file!(), line!());
                                });
                            }
                            Ok(false) => {}
                            Err(e) => owl_shared::log_warn!(
                                "snap",
                                "turn 末 Tail 尾锚点补拍失败(继续收口): {e}"
                            ),
                        }
                        if let Err(e) = self.reconcile_active_snap_layout() {
                            owl_shared::log_warn!("snap", "turn 末布局压缩失败: {e}");
                        }
                    }
                }
            }
        }
        let act = self
            .turns
            .take_exec()
            .ok_or_else(|| ModelError::EngineInvariant {
                what: "turn_end:无活跃 turn".into(),
            })?;
        // **单 turn 诊断账**:在 take_exec 之后、任何收口动作之前取快照。
        // `started_ms` = start_one 起跑;首 token 时间由 `sample_and_emit` 写。
        let now = crate::running::now_ms();
        let first_token_ms = act.first_token_ms.unwrap_or(now);
        let prompt_tokens = act.prompt_ids.len();
        let cached_tokens = act.cached_start.min(prompt_tokens);
        let stats = crate::turn::TurnStats {
            prefill_ms: first_token_ms.saturating_sub(act.started_ms),
            decode_ms: now.saturating_sub(first_token_ms),
            prompt_tokens,
            cached_tokens,
            prefill_tokens: prompt_tokens.saturating_sub(cached_tokens),
            decode_tokens: if act.first_token_ms.is_some() {
                act.out.len().saturating_sub(1)
            } else {
                0
            },
            spec_rounds: act.spec.spec_rounds,
            spec_accepted: act.spec.spec_accepted,
            spec_emitted: act.spec.spec_emitted,
            spec_degraded_steps: act.spec.spec_degraded_steps,
        };
        // B4 分母保鲜回写(下一 turn 种入;见 RunningEngine::bare_cost_carry_ms)
        self.dev.spec_rt.carry_ms = act.spec.bare_cost_ewma_ms;
        self.dev.spec_rt.carry_ctx = act.spec.bare_cost_ctx;
        // 取证(B6.3 后泄漏案):每 turn 设备余量打点(free 账由 cuda 侧 dtoh
        // 收割刷新;泄漏 = 线性下降)
        owl_shared::log_debug!(
            "vram-trace",
            "turn end: free={:.0}MiB out={} rope_gap={}",
            owl_shared::vram::free_bytes() as f64 / 1024.0 / 1024.0,
            act.out.len(),
            act.rope_gap
        );
        // spec 轮机器状态随 act(TurnSpecState)销毁 —— 跨 turn 残留
        // 结构性消失(2026-10-10 生命周期收口;原手动 spec_drafts_host 清)
        // 会话账落地(S1 收口):KV 此刻 = prompt 全量 + 生成段;GDN 状态
        // 跨 turn 连续。临时会话终了即焚(表不随 turn 数无界增长)
        if let Some(s) = self.bank.parts_mut().sessions.get_mut(act.session_id) {
            s.commit_turn(&act.prompt_ids, &act.out, &act.positions)?;
            // **图像段账写回**(前缀无感删除;2026-10-23):与 `tokens` 同一次
            // 收口 —— 两处各写一份 = 下一轮删除识别拿错坐标。
            s.set_img_spans(act.img_spans.clone(), act.img_pads.clone());
            // **L1 资产：Tail 尾锚点 + Tail/History 布局跨轮保留**。
            // 空闲会话的 tail_snap 保持 Tail 角色（最后淘汰，不是 pin）。
            if act.last_snap.is_some() {
                s.tail_snap = act.last_snap;
                s.tail_at = act.last_snap_at;
            }
            s.msg_snaps = act.msg_snaps.clone();
            s.msg_snap_roles = act.msg_snap_roles.clone();
            s.msg_snap_keys = act.msg_snap_keys.clone();
            // 链登记 / 盘索引 / 表释放 / GDN 行 / 归属下行 / N4 全部由
            // `StoreHook::Release(TurnEnd)` 统一执行（Phase 3）。
        }
        // **收口顺序**：先跑 Release 钩子（块链/行/N4），再无条件走准入围栏，
        // 最后才把钩子错误上抛 —— `store_hook(...)?` 直接早退会跳过收口，
        // 一次收尾失败就会把 turn 永久留在准入账里（2026-11-06 真机）。
        let released = self
            .store_hook(crate::store_hook::StoreHook::turn_end(
                act.id,
                act.session_id,
                act.gdn_slot,
            ))
            .await;
        // 准入账同步(S3b §23.2):turn 生死 = 保留额生死
        self.close_admission(act.id);
        released?;
        // **B-3d**:选择层视图态随 turn 收口(视图是 turn 域资源 —— 下一 turn
        // 的召回点重新产出;不清 = 用旧视图读新会话的块,静默错值)
        self.sel.clear_view();
        let mut text = self.boot.tok.decode(&act.out);
        // stop 串命中(AGT-06):终了文本较交付给客户端的输出多了一截停止串 ⇒
        // 截回命中位置(char 边界安全,不命中则原样)
        if let Some(n) = act.trim_text_to {
            let n = crate::exec::floor_boundary(&text, n);
            text.truncate(n);
        }
        Ok(TurnEvent::Completed {
            turn: act.id,
            text,
            finish: reason,
            stats,
        })
    }
}
