//! turn 终局(burn_ephemeral / abandon / complete)
use super::*;
use owl_iface::contract::ModelError;

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// **turn 末释放设备表**(S2 引用拆分;§25.1①):表出账归还(引用减一),
    /// `cached_len` 归到**块边界**(缓存里确实还有的整块数 —— 尾块随表释放,
    /// 内容丢,下次重算 ≤ 1 页)。
    // Phase 3：H6 执行臂。唯一调用者是 `StoreHook::Release`；
    // engine 其它位置不得直呼。
    pub(crate) fn release_turn_table(&mut self, session_id: u64) -> Result<()> {
        let page = self.dev.pool.geom.page.max(1);
        let mut table = Vec::new();
        if let Some(s) = self.bank.sessions.get_mut(session_id) {
            table = std::mem::take(&mut s.block_table);
            s.cached_len = (s.tokens.len() / page) * page;
        }
        self.bank.rel.release_table(&mut table)?;
        Ok(())
    }

    /// turn 烧毁(E5 性能会战:server 泵错误路径的原先弃置会让 active
    /// 会话滞留 → GDN 格泄漏 → 2 格两轮耗尽)。与 complete 同构的清理,
    /// 但**不 commit_turn/不进前缀缓存**(引擎错误后账面不可信;ephemeral
    /// 直接焚,键控会话保留账本 —— 前缀守卫失配自动回退全量重算,安全)。
    /// E1:ephemeral 会话终了焚毁(块链/MTP 链归账房 + 表移除释放格)。
    /// complete 与 abandon 共用 —— 两路径的清理原 ~20 行双写。
    pub(crate) fn burn_ephemeral(&mut self, session_id: u64) {
        self.bank.burn(session_id);
        // **[N5 SessionClose]**:会话关闭 ⇒ 归属回 `System`(会话面已交还给
        //    缓存/池;后续分配没有会话归属可谈)
        self.set_dev_ctx(
            owl_iface::handle::Owner::System,
            owl_iface::handle::AssetClass::Unclassified,
        );
    }
    pub async fn abandon(&mut self, err: String) -> TurnEvent {
        let turn_id = self
            .turns
            .executing
            .or_else(|| self.turns.active.first().map(|a| a.id))
            .unwrap_or(0);
        // B5.1:只弃**本步执行 turn**(批步错 = 该行失败;其余活跃行留场,
        // 由服务面继续泵 —— 全清会把无辜会话连坐)。执行选择器缺省 = 活跃首
        // (批动作失败时未选定)。
        if let Some(act) = self.turns.take_exec() {
            self.abandon_active(act, turn_id).await;
        }
        // 准入账同步(S3b §23.2):turn 生死 = 保留额生死
        self.close_admission(turn_id);
        TurnEvent::Failed { turn: turn_id, err }
    }

    /// **按 turn 精确终结**(B5.1 服务面契约:合批档下"取消"必须指名 turn,
    /// 否则会误杀同批的别家会话)。
    ///
    /// - 活跃:走与 `abandon` 同一条清理(块链归还/还 GDN 行/N4 节点),只弃该行;
    /// - **排队中**:只出队 + 准入账收口 + 丢弃该 turn 的残留事件(不碰设备);
    /// - 不在账:`None`(调用方如实处理 —— 竞态下 turn 可能刚自然终了)。
    pub async fn cancel_turn(&mut self, turn: u64) -> Option<TurnEvent> {
        // 残留事件先清(否则服务面会在 Failed 之后还收到该 turn 的 Token 帧)
        self.turns.pending_events.retain(|e| match e {
            TurnEvent::Prefill { turn: t, .. }
            | TurnEvent::Token { turn: t, .. }
            | TurnEvent::Completed { turn: t, .. }
            | TurnEvent::Failed { turn: t, .. } => *t != turn,
            TurnEvent::Idle => true,
        });
        if let Some(act) = self.turns.take_active(turn) {
            self.abandon_active(act, turn).await;
            self.close_admission(turn);
            return Some(TurnEvent::Failed {
                turn,
                err: "客户端取消(活跃 turn)".into(),
            });
        }
        if let Some(pos) = self.turns.queue.iter().position(|t| t.id == turn) {
            if let Some(t) = self.turns.queue.remove(pos) {
                // 排队 turn 未占设备资源(块链/GDN 行都在起 turn 时借)⇒ 只清账。
                if self.turns.admit_wait.map(|(id, _)| id) == Some(t.id) {
                    self.turns.admit_wait = None;
                }
                self.close_admission(turn);
                return Some(TurnEvent::Failed {
                    turn,
                    err: "客户端取消(排队中)".into(),
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
    pub(crate) async fn abandon_active(&mut self, act: crate::turn::ActiveTurn, turn_id: u64) {
        debug_assert_eq!(act.id, turn_id, "abandon_active: turn id 与 act 不同源");
        // 墙钟 EWMA（`Retry-After` 估算用；所有终局路径同源）
        self.turns
            .note_turn_ms(crate::running::now_ms().saturating_sub(act.started_ms));
        // 草稿账随 act(TurnSpecState)销毁(生命周期收口)。
        // H6 / GDN 行 / 归属下行 / N4 全部由 `Release(Abort)` 统一执行；
        // 弃置路径失败只 WARN，不掩盖原始错误。
        if let Err(e) = self
            .store_hook(crate::store_hook::StoreHook::abort(
                act.id,
                act.session_id,
                act.gdn_slot,
                act.ephemeral,
            ))
            .await
        {
            owl_shared::log_warn!("store", "Release(Abort) 失败(继续收尾): {e}");
        }
    }

    /// **准入账收口**(S3b §23.2):turn 终了 ⇒ 还保留额 + 重算可动面。
    /// 账里没有该 turn(如已拒/已清)= 幂等(不报错也不铎）。
    pub(crate) fn close_admission(&mut self, turn_id: u64) {
        if turn_id == 0 {
            return;
        }
        // 准入状态机归还预留(幂等;complete/abandon/腿终局全走这里)
        self.turns.admission.note_finished(turn_id);
        // **P0-4**：Grant 是 store 的账，准入收口也必须归还（store_hook::Release
        // 正常路径已还；这里兜底排队取消 / 拒绝 / 早退路径，幂等）。
        let released = self.bank.rel.release_turn_grants(turn_id);
        if !released.is_empty() {
            owl_shared::log_debug!(
                "store",
                "准入收口归还 Grant：turn {turn_id} {} 张",
                released.len()
            );
        }
        if let Err(e) = self.bank.rel.admission.release(turn_id) {
            owl_shared::log_debug!("store", "准入账收口：turn {turn_id} 不在账上（{e}）");
        }
        if let Err(e) = self.sync_pool_account() {
            owl_shared::log_warn!("store", "准入账同步失败：{e}");
        }
    }

    pub(crate) async fn complete(
        &mut self,
        reason: crate::turn::FinishReason,
    ) -> Result<TurnEvent> {
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
        };
        // B4 分母保鲜回写(下一 turn 种入;见 RunningEngine::bare_cost_carry_ms)
        self.dev.spec_rt.carry_ms = act.spec.bare_cost_ewma_ms;
        self.dev.spec_rt.carry_ctx = act.spec.bare_cost_ctx;
        // 取证(B6.3 后泄漏案):每 turn 设备余量打点(free 账由 cuda 侧 dtoh
        // 收割刷新;泄漏 = 线性下降)
        owl_shared::log_debug!(
            "vram-trace",
            "turn end: free={:.0}MiB out={}",
            owl_shared::vram::free_bytes() as f64 / 1024.0 / 1024.0,
            act.out.len()
        );
        // spec 轮机器状态随 act(TurnSpecState)销毁 —— 跨 turn 残留
        // 结构性消失(2026-10-10 生命周期收口;原手动 spec_drafts_host 清)
        // 会话账落地(S1 收口):KV 此刻 = prompt 全量 + 生成段;GDN 状态
        // 跨 turn 连续。临时会话终了即焚(表不随 turn 数无界增长)
        if let Some(s) = self.bank.sessions.get_mut(act.session_id) {
            s.commit_turn(&act.prompt_ids, &act.out, &act.positions, act.rope_gap)?;
            // **图像段账写回**(前缀无感删除;2026-10-23):与 `tokens` 同一次
            // 收口 —— 两处各写一份 = 下一轮删除识别拿错坐标。
            s.set_img_spans(act.img_spans.clone(), act.img_pads.clone());
            // 链登记 / 盘索引 / 表释放 / GDN 行 / 归属下行 / N4 全部由
            // `StoreHook::Release(TurnEnd)` 统一执行（Phase 3）。
        }
        self.store_hook(crate::store_hook::StoreHook::turn_end(
            act.id,
            act.session_id,
            act.gdn_slot,
            act.ephemeral,
        ))
        .await?;
        // 准入账同步(S3b §23.2):turn 生死 = 保留额生死
        self.close_admission(act.id);
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
