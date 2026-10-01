//! RunningEngine:执行态引擎(请求入口;2026-10-01 拆分自 engine.rs §3)。
//!
//! 职责 = turn 生命周期账(submit/close/事件产出)+ pump 主循环
//! (调度/执行分派)。分工:
//! - 决策逻辑 = scheduler.rs(`RunningEngine::schedule`,纯主机侧);
//! - 设备执行 = exec.rs(execute_*/prefill_chunk/采样收口);
//! - 状态块治理 = state.rs(StatePool:KV 池/GDN 格/快照/块表)。
//!
//! 并发:M0.5 单槽串行(一次一个 turn 占 0 号槽;GDN 状态 per-turn 零化,
//! KV 由「kv_len 窗口 + 先写后打分」语义天然隔离);真并发随调度层铺开
//! S3 拼批换入,submit/pump 形状不变。

use std::collections::VecDeque;
use std::sync::Arc;

use owl_iface::contract::{DeviceClient, ModelError};
use owl_models::layers::rope::Rope;
use owl_models::model::Model;
use owl_models::tokenizer::Tokenizer;

use crate::blocks::BlockManager;
use crate::engine::EngineConfig;
use crate::graph_plan::GraphPlan;
use crate::scheduler::{SchedulerOutput, StepAction};
use crate::session::SessionTable;
use crate::state::{GDN_SLOTS, StatePool};
use crate::turn::{TurnEvent, TurnSpec};

type Result<T> = std::result::Result<T, ModelError>;

pub(crate) struct Turn {
    pub(crate) id: u64,
    /// 归属会话(已解析 id;None 提交在 submit 时已建临时账)
    pub(crate) session_id: u64,
    /// 临时会话(终了即焚;行为同 M0.5 每 turn 全量重算)
    pub(crate) ephemeral: bool,
    pub(crate) spec: TurnSpec,
    pub(crate) prompt_ids: Vec<u32>,
}

pub(crate) struct ActiveTurn {
    pub(crate) id: u64,
    pub(crate) session_id: u64,
    pub(crate) ephemeral: bool,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) max_new: usize,
    pub(crate) out: Vec<u32>,
    /// KV/prompt 推进指针(会话 turn 从 `cached_len` 起步 = 增量 prefill)
    pub(crate) fed: usize,
    pub(crate) decoded: String,
}

/// 执行态引擎(请求入口)
pub struct RunningEngine<D: DeviceClient> {
    pub(crate) session: GraphPlan<D>,
    pub(crate) tok: Tokenizer,
    pub(crate) cfg: EngineConfig,
    /// 捕获三态结果(Captured = 回放态;EagerFallback = 同闭包直发)
    pub capture_outcome: crate::graph_plan::PlanOutcome,
    /// 设备状态块池(KV paged 池/GDN 格/快照池/持久块表;state.rs)
    pub(crate) pool: StatePool,
    pub(crate) queue: VecDeque<Turn>,
    pub(crate) active: Option<ActiveTurn>,
    pub(crate) next_id: u64,
    /// 会话账(S0):同会话跨 turn 复用 KV/GDN,增量 prefill
    pub(crate) sessions: SessionTable,
    /// prefill 块式喂入(W1)的模型面:树根构造
    pub(crate) model: Arc<Model>,
    pub(crate) rope: Rope,
    /// KV 物理块账房(E2b;块链按会话分派,池内 ref 计数)
    pub(crate) blocks_m: BlockManager,
}

impl<D: DeviceClient> RunningEngine<D> {
    /// 提交 turn(临时会话:每 turn 独立,终了即焚;行为同 M0.5)。
    /// 预算越界 fail-fast:prompt + 生成 ≤ max_seq_tokens(= KV 槽位)。
    pub fn submit(&mut self, prompt: impl Into<String>, max_new: usize) -> Result<u64> {
        self.submit_session(None, prompt, max_new).map(|(_, t)| t)
    }

    /// 会话化提交(S0):`session = None` 自动开临时会话;`Some(id)` =
    /// 连续会话 —— prompt 语义为**全量重发**(历史 + 新消息,客户端保证
    /// 前缀一致):引擎按 token 账前缀守卫,命中则跳过 GDN 重置、只
    /// prefill `cached_len..` 增量段;失配回退全量重算(S1,正确性优先)。
    /// 返回 (session_id, turn_id)。
    pub fn submit_session(
        &mut self,
        session: impl Into<Option<u64>>,
        prompt: impl Into<String>,
        max_new: usize,
    ) -> Result<(u64, u64)> {
        let session = session.into();
        let prompt = prompt.into();
        let prompt_ids = {
            let wrapped = self.tok.chat_wrap(&prompt);
            self.tok.encode(&wrapped)
        };
        if prompt_ids.is_empty() {
            return Err(ModelError::Msg("submit: 空 prompt".into()));
        }
        // 预算(会话感知):命中账本的 turn 按 max(prompt, cached) 记账
        // (增量段写 `[cached, prompt)` 行,生成续写其后;失配回退全量时
        // prompt 为准,同一上界)
        let cached_hit = session
            .and_then(|id| self.sessions.get(id).map(|s| (s.cached_len, s.guard(&prompt_ids))))
            .filter(|&(_, guard)| guard)
            .map(|(cached, _)| cached)
            .unwrap_or(0);
        let need = prompt_ids.len().max(cached_hit) + max_new;
        if need > self.cfg.max_seq_tokens {
            return Err(ModelError::Msg(format!(
                "submit: prompt {} + 生成 {} 超预算 {}",
                prompt_ids.len(),
                max_new,
                self.cfg.max_seq_tokens
            )));
        }
        let (session_id, ephemeral) = match session {
            Some(id) => (self.sessions.get_or_create(Some(id), GDN_SLOTS)?.0, false),
            None => (self.sessions.get_or_create(None, GDN_SLOTS)?.0, true),
        };
        let id = self.next_id;
        self.next_id += 1;
        self.queue.push_back(Turn {
            id,
            session_id,
            ephemeral,
            spec: TurnSpec { prompt, max_new },
            prompt_ids,
        });
        Ok((session_id, id))
    }

    /// 显式关会话(客户端声明不再续;账本清票,块链归还账房)
    pub fn close_session(&mut self, id: u64) -> Result<()> {
        if let Some(s) = self.sessions.get_mut(id) {
            let mut table = std::mem::take(&mut s.block_table);
            self.blocks_m.release_table(&mut table);
        }
        self.sessions.close(id)
    }

    /// 会话账观测(已入 KV 的 token 数;None = 会话不存在)
    pub fn session_len(&self, id: u64) -> Option<usize> {
        self.sessions.get(id).map(|s| s.cached_len)
    }

    /// 推进到下一个事件点(每调用 = 活跃 turn 的一个采样步,或 turn 的
    /// 启动切换;`Idle` = 队列排空且无活跃,调用方可安全挂起等新提交)。
    /// **S0 调度/执行分离**:本函数 = 调度(纯主机侧决策,零设备 IO)
    /// + 执行(设备侧照办)—— 决策面可单测,执行器无决策权
    /// (调度层铺开 §S0;对标 vLLM v1 schedule/execute 分离)。
    pub async fn pump(&mut self) -> Result<TurnEvent> {
        let prof = std::env::var_os("OWL_STEP_PROFILE").is_some();
        // metrics 框架接线(2026-10-01 合并示范):schedule 分相计时入
        // owl_shared 全局 store(debug 展开/release 零开销);prof 时查询
        // 打印。其余 STEP_PROFILE 打点保留原样(gl-prof/srv-timing 工具链
        // 依赖其输出格式,逐点收编挂账)。
        owl_shared::timer_start!("engine.pump.schedule");
        let out = self.schedule()?;
        owl_shared::timer_end!("engine.pump.schedule");
        if prof {
            owl_shared::metrics::query_metrics(&owl_shared::metrics::MetricsFilter::new()
                .tag_prefix("engine.pump."));
        }
        match out {
            SchedulerOutput::Idle => Ok(TurnEvent::Idle),
            SchedulerOutput::Step { begin, action } => {
                if let Some(b) = begin {
                    self.execute_begin(b).await?;
                }
                match action {
                    StepAction::Prefill { chunk_ids, base, is_last } => {
                        self.execute_prefill(chunk_ids, base, is_last).await
                    }
                    StepAction::Decode { token, pos, kv_slot, gdn_slot, grew } => {
                        self.execute_decode(token, pos, kv_slot, gdn_slot, grew).await
                    }
                }
            }
        }
    }

    /// 便捷入口:提交一个 turn 并泵到其完成,返回全文
    /// (多 turn 并发场景用 submit + pump 手动驱动)
    pub async fn generate(&mut self, prompt: impl Into<String>, max_new: usize) -> Result<String> {
        let id = self.submit(prompt, max_new)?;
        loop {
            match self.pump().await? {
                TurnEvent::Completed { turn, text } if turn == id => return Ok(text),
                TurnEvent::Idle => {
                    return Err(ModelError::Msg(format!("generate: turn {id} 队列丢失")))
                }
                _ => {}
            }
        }
    }
}
