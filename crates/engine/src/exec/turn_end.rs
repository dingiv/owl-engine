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
        if let Some(s) = self.sessions.get_mut(session_id) {
            let mut table = std::mem::take(&mut s.block_table);
            self.blocks_m.release_table(&mut table);
            // MTP 链第二链同步归还(E5-M2b;DFlash2 模式表空归零账)
            let mut mtp_table = std::mem::take(&mut s.mtp_block_table);
            self.blocks_mtp.release_table(&mut mtp_table);
        }
        self.sessions.close(session_id).ok();
    }
    pub async fn abandon(&mut self, err: String) -> TurnEvent {
        let turn_id = self.active.as_ref().map(|a| a.id).unwrap_or(0);
        if let Some(act) = self.active.take() {
            // 草稿账随 act(TurnSpecState)销毁(生命周期收口)
            if act.ephemeral {
                self.burn_ephemeral(act.session_id);
            }
        }
        TurnEvent::Failed { turn: turn_id, err }
    }

    pub(crate) async fn complete(&mut self) -> Result<TurnEvent> {
        let act = self.active.take().expect("active");
        // 取证(B6.3 后泄漏案):每 turn 设备余量打点(free 账由 cuda 侧 dtoh
        // 收割刷新;泄漏 = 线性下降)
        eprintln!("[vram-trace] turn end: free={:.0}MiB out={}",
                  owl_shared::vram::free_bytes() as f64 / 1024.0 / 1024.0, act.out.len());
        // spec 轮机器状态随 act(TurnSpecState)销毁 —— 跨 turn 残留
        // 结构性消失(2026-10-10 生命周期收口;原手动 spec_drafts_host 清)
        // 会话账落地(S1 收口):KV 此刻 = prompt 全量 + 生成段;GDN 状态
        // 跨 turn 连续。临时会话终了即焚(表不随 turn 数无界增长)
        if let Some(s) = self.sessions.get_mut(act.session_id) {
            s.commit_turn(&act.prompt_ids, &act.out);
            // E2c:块链登记前缀缓存(缓存持引用;会话释放后块仍驻留)
            if self.pool.paged {
                self.blocks_m.cache_seq(&s.tokens, &s.block_table);
            }
        }
        if act.ephemeral {
            // 终了即焚:块链归还账房 + 会话表移除(GDN 格随之释放)
            self.burn_ephemeral(act.session_id);
        }
        let text = self.tok.decode(&act.out);
        Ok(TurnEvent::Completed { turn: act.id, text })
    }
}
