//! turn 终局(burn_ephemeral / abandon / complete)
use super::*;
use owl_iface::contract::ModelError;

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// **turn 末释放设备表**(S2 引用拆分;§25.1①):表出账归还(引用减一),
    /// `cached_len` 归到**块边界**(缓存里确实还有的整块数 —— 尾块随表释放,
    /// 内容丢,下次重算 ≤ 1 页)。
    pub(crate) fn release_turn_table(&mut self, session_id: u64) -> Result<()> {
        let page = self.dev.pool.page.max(1);
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
        let turn_id = self.turns.active.as_ref().map(|a| a.id).unwrap_or(0);
        if let Some(act) = self.turns.active.take() {
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
            self.n4_pool_sweep(act.id);
            self.n4_report_unreturned(act.id);
        }
        // 准入账同步(S3b §23.2):turn 生死 = 保留额生死
        self.close_admission(turn_id);
        TurnEvent::Failed { turn: turn_id, err }
    }

    /// **准入账收口**(S3b §23.2):turn 终了 ⇒ 还保留额 + 重算可动面。
    /// 账里没有该 turn(如已拒/已清)= 幂等(不报错也不铎）。
    pub(crate) fn close_admission(&mut self, turn_id: u64) {
        if turn_id == 0 {
            return;
        }
        if let Err(e) = self.bank.rel.admission.release(turn_id) {
            owl_shared::log_debug!("store", "准入账收口：turn {turn_id} 不在账上（{e}）");
        }
        if let Err(e) = self.sync_pool_account() {
            owl_shared::log_warn!("store", "准入账同步失败：{e}");
        }
    }

    pub(crate) async fn complete(&mut self, reason: crate::turn::FinishReason) -> Result<TurnEvent> {
        let act = self.turns.active.take().ok_or_else(|| ModelError::EngineInvariant { what: "turn_end:无活跃 turn".into() })?;
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
            if self.dev.pool.paged {
                let up = self.bank.rel.leg.register_chain(&s.tokens, &s.block_table)?;
                if self.cfg.knobs.probes.debug {
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
        // 缺口 4:先把本 turn 已释放的临时块入**复用池**(改归属 Cache)
        self.n4_pool_sweep(act.id);
        // 归还清单(store 立案 §缺口1):非空 = 有临时面没在节点上归还 ⇒ 响亮留痕
        self.n4_report_unreturned(act.id);
        // 准入账同步(S3b §23.2):turn 生死 = 保留额生死
        self.close_admission(act.id);
        // **B-3d**:选择层视图态随 turn 收口(视图是 turn 域资源 —— 下一 turn
        // 的召回点重新产出;不清 = 用旧视图读新会话的块,静默错值)
        self.sel_view = None;
        let mut text = self.tok.decode(&act.out);
        // stop 串命中(AGT-06):终了文本较交付给客户端的输出多了一截停止串 ⇒
        // 截回命中位置(char 边界安全,不命中则原样)
        if let Some(n) = act.trim_text_to {
            let n = crate::exec::floor_boundary(&text, n);
            text.truncate(n);
        }
        Ok(TurnEvent::Completed { turn: act.id, text, finish: reason })
    }
}

