//! turn 终局(burn_ephemeral / abandon / complete)
use super::*;
use owl_iface::contract::ModelError;

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// **turn 末释放设备表**(S2 引用拆分;§25.1①):表出账归还(引用减一),
    /// `cached_len` 归到**块边界**(缓存里确实还有的整块数 —— 尾块随表释放,
    /// 内容丢,下次重算 ≤ 1 页)。
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
        // 草稿账随 act(TurnSpecState)销毁(生命周期收口)
        if act.ephemeral {
            self.burn_ephemeral(act.session_id);
        } else {
            // S2 引用拆分:弃置的 turn 也不留设备表(半途的链**不进缓存**
            // —— 没有 `register_chain` ⇒ 下次内容寻址自然不命中,安全)
            if let Err(e) = self.release_turn_table(act.session_id) {
                owl_shared::log_warn!("session", "弃置 turn 释放设备表失败: {e}");
            }
        }
        // S2b:还 GDN 行(半途状态**不落快照** —— 弃置的 turn 账面不可信)
        if let Err(e) = self.bank.rel.state.return_row(act.gdn_slot, None) {
            owl_shared::log_warn!("session", "弃置 turn {} 还 GDN 行失败: {e}", act.id);
        }
        // **[N4 TurnEnd]**(弃置路径同款:归属下行;须在 act 还在手时取会话事实)
        self.set_ctx_after_turn(act.session_id, act.ephemeral);
        // abandon 返回 TurnEvent(非 Result)⇒ 节点失败**响亮留痕**但不掩盖原错
        if let Err(e) = self.node(crate::running::Node::TurnEnd(act.id)).await {
            owl_shared::log_warn!("store", "N4 节点失败(继续收尾): {e}");
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
        if let Err(e) = self.bank.rel.admission.release(turn_id) {
            owl_shared::log_debug!("store", "准入账收口：turn {turn_id} 不在账上（{e}）");
        }
        if let Err(e) = self.sync_pool_account() {
            owl_shared::log_warn!("store", "准入账同步失败：{e}");
        }
    }

    pub(crate) async fn complete(&mut self, reason: crate::turn::FinishReason) -> Result<TurnEvent> {
        let act = self.turns.take_exec().ok_or_else(|| ModelError::EngineInvariant { what: "turn_end:无活跃 turn".into() })?;
        // B4 分母保鲜回写(下一 turn 种入;见 RunningEngine::bare_cost_carry_ms)
        self.dev.spec_rt.carry_ms = act.spec.bare_cost_ewma_ms;
        self.dev.spec_rt.carry_ctx = act.spec.bare_cost_ctx;
        // 取证(B6.3 后泄漏案):每 turn 设备余量打点(free 账由 cuda 侧 dtoh
        // 收割刷新;泄漏 = 线性下降)
        owl_shared::log_debug!("vram-trace", "turn end: free={:.0}MiB out={}",
                  owl_shared::vram::free_bytes() as f64 / 1024.0 / 1024.0, act.out.len());
        // spec 轮机器状态随 act(TurnSpecState)销毁 —— 跨 turn 残留
        // 结构性消失(2026-10-10 生命周期收口;原手动 spec_drafts_host 清)
        // 会话账落地(S1 收口):KV 此刻 = prompt 全量 + 生成段;GDN 状态
        // 跨 turn 连续。临时会话终了即焚(表不随 turn 数无界增长)
        if let Some(s) = self.bank.sessions.get_mut(act.session_id) {
            s.commit_turn(&act.prompt_ids, &act.out);
            // E2c:块链登记前缀缓存(缓存持引用;会话释放后块仍驻留)。
            // S3-2:登记走 store 的块腿(引用配对在腿内自洽)
            if self.dev.pool.geom.paged {
                // **会话感知**(2026-10-19):链块记到会话名下(淘汰时按会话从末端削)
                let up = self.bank.rel.leg.register_chain_owned(
                    &s.tokens,
                    &s.block_table,
                    act.session_id,
                )?;
                if self.boot.cfg.knobs.probes.debug {
                    owl_shared::log_debug!(
                        "dbg prefix",
                        "register_chain: 块表={} 新入={} 逐出={}",
                        s.block_table.len(),
                        up.inserted.len(),
                        up.evicted.len()
                    );
                }
            }
        }
        // **S6.1:盘上链索引**(turn 收口一次;内容可能还没溢写 ⇒ boot 按内容截断)。
        // 放在 `ephemeral` 焚毁之前 —— 会话 tokens 此刻还在。
        if self.dev.pool.geom.paged && self.bank.rel.disk_enabled() {
            let toks = self
                .bank
                .sessions
                .get(act.session_id)
                .map(|s| s.tokens.clone())
                .unwrap_or_default();
            if !toks.is_empty() {
                self.write_chain_index(&toks).await?;
            }
        }
        if act.ephemeral {
            // 终了即焚:块链归还账房 + 会话表移除(GDN 格随之释放)
            self.burn_ephemeral(act.session_id);
        } else {
            // **S2 引用拆分(§25.1①)**:idle 会话**不持设备表** ——
            // 历史此刻已进前缀缓存(上面的 `register_chain`),设备表的引用随 turn 死;
            // 下一个 turn 由**内容寻址重建**(`schedule` 的前缀匹配路径)。
            // 于是:①idle 会话对设备零持有(历史 = 纯缓存 ⇒ 可动/可逐/可下沉)
            //       ②`Admission.reserved` = 真·在飞工作集(§24.5② 收口)
            self.release_turn_table(act.session_id)?;
        }
        // **S2b(§28):还 GDN 行**(行 = turn 域资源;状态已由边界快照承担)
        if let Err(e) = self.bank.rel.state.return_row(act.gdn_slot, None) {
            owl_shared::log_warn!("session", "turn {} 还 GDN 行失败: {e}", act.id);
        }
        // **[N4 TurnEnd]**:归属下行 —— 会话仍活 ⇒ `Session(s)`,焚毁 ⇒ `System`
        self.set_ctx_after_turn(act.session_id, act.ephemeral);
        // **[N4 TurnEnd]**(节点入口):池清点(改归属 Cache)+ 未归还清单(响亮)
        self.node(crate::running::Node::TurnEnd(act.id)).await?;
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
        Ok(TurnEvent::Completed { turn: act.id, text, finish: reason })
    }
}

