//! turn 终局(burn_ephemeral / abandon / complete)
use super::*;

impl<D: DeviceClient> crate::running::RunningEngine<D> {
    /// turn 烧毁(E5 性能会战:server 泵错误路径的原先弃置会让 active
    /// 会话滞留 → GDN 格泄漏 → 2 格两轮耗尽)。与 complete 同构的清理,
    /// 但**不 commit_turn/不进前缀缓存**(引擎错误后账面不可信;ephemeral
    /// 直接焚,键控会话保留账本 —— 前缀守卫失配自动回退全量重算,安全)。
    /// E1:ephemeral 会话终了焚毁(块链/MTP 链归账房 + 表移除释放格)。
    /// complete 与 abandon 共用 —— 两路径的清理原 ~20 行双写。
    pub(crate) fn burn_ephemeral(&mut self, session_id: u64) {
        self.bank.burn(session_id);
    }
    pub async fn abandon(&mut self, err: String) -> TurnEvent {
        let turn_id = self.turns.active.as_ref().map(|a| a.id).unwrap_or(0);
        if let Some(act) = self.turns.active.take() {
            // 草稿账随 act(TurnSpecState)销毁(生命周期收口)
            if act.ephemeral {
                self.burn_ephemeral(act.session_id);
            }
        }
        TurnEvent::Failed { turn: turn_id, err }
    }

    pub(crate) async fn complete(&mut self, reason: crate::turn::FinishReason) -> Result<TurnEvent> {
        let act = self.turns.active.take().expect("active");
        // B4 分母保鲜回写(下一 turn 种入;见 RunningEngine::bare_cost_carry_ms)
        self.spec.carry_ms = act.spec.bare_cost_ewma_ms;
        self.spec.carry_ctx = act.spec.bare_cost_ctx;
        // 取证(B6.3 后泄漏案):每 turn 设备余量打点(free 账由 cuda 侧 dtoh
        // 收割刷新;泄漏 = 线性下降)
        eprintln!("[vram-trace] turn end: free={:.0}MiB out={}",
                  owl_shared::vram::free_bytes() as f64 / 1024.0 / 1024.0, act.out.len());
        // spec 轮机器状态随 act(TurnSpecState)销毁 —— 跨 turn 残留
        // 结构性消失(2026-10-10 生命周期收口;原手动 spec_drafts_host 清)
        // 会话账落地(S1 收口):KV 此刻 = prompt 全量 + 生成段;GDN 状态
        // 跨 turn 连续。临时会话终了即焚(表不随 turn 数无界增长)
        if let Some(s) = self.bank.sessions.get_mut(act.session_id) {
            s.commit_turn(&act.prompt_ids, &act.out);
            // E2c:块链登记前缀缓存(缓存持引用;会话释放后块仍驻留)
            if self.pool.paged {
                self.bank.blocks_m.cache_seq(&s.tokens, &s.block_table);
            }
        }
        if act.ephemeral {
            // 终了即焚:块链归还账房 + 会话表移除(GDN 格随之释放)
            self.burn_ephemeral(act.session_id);
        }
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
