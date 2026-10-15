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

use std::sync::Arc;

use owl_iface::contract::{DeviceClient, ModelError};
use owl_models::layers::rope::Rope;
use owl_models::model::Model;
use owl_models::tokenizer::Tokenizer;

use crate::engine::EngineConfig;
use crate::graph_fleet::GraphFleet;
use crate::scheduler::{SchedulerOutput, StepAction};
use crate::session::SessionBank;
use crate::spec_runtime::SpecRuntime;
use crate::state::StatePool;
use crate::turn::{Turn, TurnEvent, TurnLedger, TurnSpec, TurnSpecState};

type Result<T> = std::result::Result<T, ModelError>;

/// 步级探针开关(诊断面;结构纯数据,Default = 全关)。
/// **入口侧构造器** = [`StepProbes::from_env`](server config/测试用例
/// 专用);引擎内部零 env 读取,经 EngineKnobs.probes 显式传入。
/// 新增探针在此登记,禁止 exec 每步读 env。
#[derive(Clone, Copy, Debug, Default)]
pub struct StepProbes {
    /// OWL_STEP_PROFILE(逐相计时)
    pub step_profile: bool,
    /// OWL_GDN_DUMP(GDN 状态格画像)
    pub gdn_dump: bool,
    /// OWL_DEBUG(通用诊断:kv-dump/logits-dump/sample-dump)
    pub debug: bool,
    /// OWL_PREFILL_CKSUM(prefill 校验和)
    pub prefill_cksum: bool,
    /// OWL_FACT_PROBE(逐轮事实值 mfact;UAF 案取证遗留 —— 无门控时
    /// 每轮 ×9 个动态 tag 无限增殖(metrics tag 爆炸 + 每轮 host 开销),
    /// 案结后默认关,2026-10-10 A1 复测踩中)
    pub fact_probe: bool,
    /// B4:OWL_SPEC_DEGRADE_AFTER(m=0 连击降级阈值;0 = 禁用)
    pub degrade_after: usize,
    /// B4:OWL_SPEC_PROBE_EVERY(降级期探测周期 token 数,失败 ×2 退避)
    pub probe_every: usize,
    /// GDN 分派门控打印(原 OWL_TRACE_GATE 层内直读 → ForwardCtx 注入)
    pub trace_gate: bool,
    /// E3 收编(原热路径散落 var_os;boot 解析一次):DFlash2 诊断/回退族
    pub dflash_probe: bool,
    pub propose_eager: bool,
    pub dflash_eager: bool,
    pub dflash_noencode: bool,
    pub dflash_dumb: bool,
}

impl StepProbes {
    /// **入口侧组装器**:自全量配置对象显式组装(引擎内部零 env 读取)
    pub fn from_config(cfg: &owl_shared::config::OwlConfig) -> Self {
        let p = &cfg.probes;
        Self {
            step_profile: p.step_profile,
            gdn_dump: p.gdn_dump,
            debug: p.debug,
            prefill_cksum: p.prefill_cksum,
            fact_probe: p.fact_probe,
            degrade_after: cfg.spec.degrade_after,
            probe_every: cfg.spec.probe_every,
            trace_gate: p.trace_gate,
            dflash_probe: p.dflash_probe,
            propose_eager: p.propose_eager,
            dflash_eager: p.dflash_eager,
            dflash_noencode: p.dflash_noencode,
            dflash_dumb: p.dflash_dumb,
        }
    }
}

/// 执行态引擎(请求入口)。
///
/// # 结构(2026-10-14 拆分:35 字段万能袋 → 状态域组合)
///
/// 按状态域拆四户入组合,本体只留**跨域核心资产**(设备面/配置/
/// 模型/池/环境/探针):
/// - [`crate::turn::TurnLedger`]:排队/活跃/id 发号/事件队列
/// - [`crate::session::SessionBank`]:会话账 + 双块账房(target/MTP)
/// - [`crate::graph_fleet::GraphFleet`]:主会话图 + spec 三族图阵
/// - [`crate::spec_runtime::SpecRuntime`]:spec 模式/草稿器/统计/B4 保鲜
///
/// 生命周期分层:turn 域状态住 ActiveTurn(随 turn 生死);引擎级
/// spec 状态住 SpecRuntime(跨 turn)—— 两层不混居(2026-10-10
/// 生命周期收口教训)。
pub struct RunningEngine<D: DeviceClient> {
    /// 步级探针(boot 解析;不可变)
    pub(crate) probes: StepProbes,
    pub(crate) cfg: EngineConfig,
    pub(crate) tok: Tokenizer,
    /// 设备状态块池(KV paged 池/GDN 格/快照池/持久块表;state.rs)
    pub(crate) pool: StatePool,
    /// prefill 块式喂入(W1)的模型面:树根构造
    pub(crate) model: Arc<Model>,
    pub(crate) rope: Rope,
    /// 解释器执行环境(EnvProvider;逐 chunk/step 注入 ctx)
    pub(crate) env: owl_models::env::EnvProvider,
    /// boot 序号(metrics 打点命名空间;进程内自增,b1/b2…)
    pub(crate) boot_seq: u64,
    /// E5 性能:schedule 累计账(轮间 host 成本剖析)
    pub(crate) sched_acc_ns: u64,
    pub(crate) sched_cnt: usize,
    /// turn 运行账(turn.rs)
    pub(crate) turns: TurnLedger,
    /// 会话 + 块账房(session.rs)
    pub(crate) bank: SessionBank,
    /// 图族(graph_fleet.rs:主会话图 + verify/fold/propose/dflash)
    pub(crate) graphs: GraphFleet<D>,
    /// 引擎级 spec 运行态(spec_runtime.rs;turn 域七件套住 ActiveTurn)
    pub(crate) spec: SpecRuntime,
}

impl<D: DeviceClient> RunningEngine<D> {
    /// 捕获三态结果(Captured = 回放态;EagerFallback = 同闭包直发)
    pub fn capture_outcome(&self) -> &crate::graph_plan::PlanOutcome {
        &self.graphs.capture_outcome
    }

    /// turn 域 spec 状态(turn 生命周期;仅活跃 turn 可触 —— 轮执行期
    /// active 恒在;turn 外触碰 = 调用序错误)
    pub(crate) fn tspec(&mut self) -> &mut TurnSpecState {
        self.turns.tspec()
    }
    pub(crate) fn tspec_ref(&self) -> &TurnSpecState {
        self.turns.tspec_ref()
    }

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
        self.submit_session_ex(session, prompt, max_new, false)
    }

    /// 同 [`Self::submit_session`],多一个 `ignore_eos`(OpenAI 兼容扩展;
    /// true = 忽略 EOS 跑满 `max_new` —— 测速口径用)
    pub fn submit_session_ex(
        &mut self,
        session: impl Into<Option<u64>>,
        prompt: impl Into<String>,
        max_new: usize,
        ignore_eos: bool,
    ) -> Result<(u64, u64)> {
        self.submit_session_opts(
            session,
            prompt,
            max_new,
            crate::turn::TurnOpts { ignore_eos, ..Default::default() },
        )
    }

    /// 同 [`Self::submit_session_ex`],但携带 per-turn 可选项(采样参数/停止串/种子;
    /// AGT-06)。`ignore_eos` 在 `opts` 内。
    pub fn submit_session_opts(
        &mut self,
        session: impl Into<Option<u64>>,
        prompt: impl Into<String>,
        max_new: usize,
        opts: crate::turn::TurnOpts,
    ) -> Result<(u64, u64)> {
        let session = session.into();
        let prompt = prompt.into();
        let prompt_ids = {
            // 诊断开关(knobs.raw_completion):裸续写,绕过 chat 模板 ——
            // 模型健康度鉴别(模板态病 vs 权重病)的 A/B 臂
            // §三十二:含 im_start = 上层已渲染完整 chat 模板(OpenAI 多轮
            // 面),跳过单轮 chat_wrap(整段历史塞单 user 消息 = 多轮崩根因)
            let wrapped = if self.cfg.knobs.raw_completion || prompt.contains("<|im_start|>") {
                prompt.clone()
            } else {
                self.tok.chat_wrap(&prompt)
            };
            // 诊断:最终 prompt 落盘(OWL_PROMPT_DUMP=1;§三十六 多轮"无记忆"
            // 幻觉定位面 —— dump 渲染+wrap 后送模型的完整 prompt 字节)
            if std::env::var("OWL_PROMPT_DUMP").is_ok() {
                let ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    // 系统时钟回拨兜底 → 0(非失败路径;仅用于 dump 文件名)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let path = format!("/tmp/owl-prompt-{ms}.txt");
                match std::fs::write(&path, &wrapped) {
                    Ok(_) => eprintln!("[prompt-dump] {path} ({}B)", wrapped.len()),
                    Err(e) => eprintln!("[prompt-dump] 落盘失败 {e}"),
                }
            }
            self.tok.encode(&wrapped)
        };
        if prompt_ids.is_empty() {
            return Err(ModelError::Msg("submit: 空 prompt".into()));
        }
        // 预算(会话感知):命中账本的 turn 按 max(prompt, cached) 记账
        // (增量段写 `[cached, prompt)` 行,生成续写其后;失配回退全量时
        // prompt 为准,同一上界)
        let cached_hit = session
            .and_then(|id| self.bank.sessions.get(id).map(|s| (s.cached_len, s.guard(&prompt_ids))))
            .filter(|&(_, guard)| guard)
            // 无会话 / 守卫未命中 → cached 0(语义默认,非失败)
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
            Some(id) => (self.bank.get_or_create_evicting(Some(id), self.cfg.knobs.gdn_slots)?, false),
            None => (self.bank.get_or_create_evicting(None, self.cfg.knobs.gdn_slots)?, true),
        };
        let id = self.turns.alloc_id();
        self.turns.queue.push_back(Turn {
            id,
            session_id,
            ephemeral,
            spec: TurnSpec { prompt, max_new, ignore_eos: opts.ignore_eos },
            prompt_ids,
            opts,
            cached: cached_hit,
        });
        Ok((session_id, id))
    }

    /// 显式关会话(客户端声明不再续;账本清票,块链归还账房)
    pub fn close_session(&mut self, id: u64) -> Result<()> {
        self.bank.close(id)
    }

    /// 会话账观测(已入 KV 的 token 数;None = 会话不存在)
    pub fn session_len(&self, id: u64) -> Option<usize> {
        self.bank.sessions.get(id).map(|s| s.cached_len)
    }

    /// 提交后的 prompt token 数(turn 尚在队列/活跃期内有效;usage 真账
    /// 单源 —— 服务层在 submit 后即刻取用,终了后 turn 已焚,返回 None)
    pub fn turn_prompt_len(&self, turn: u64) -> Option<usize> {
        self.turns
            .queued(turn)
            .map(|t| t.prompt_ids.len())
            .or_else(|| self.turns.active_of(turn).map(|a| a.prompt_ids.len()))
    }

    /// 提交时**前缀命中量**(usage `cached_tokens` 真源;AGT-12):
    /// 会话守卫命中 ⇒ = 会话已缓存 token 数;未命中/无会话 ⇒ 0。
    /// **真值**:`prompt_tokens - cached_tokens` = 本 turn 实际 prefill 量。
    pub fn turn_cached_len(&self, turn: u64) -> Option<usize> {
        self.turns.queued(turn).map(|t| t.cached)
    }

    /// 活跃 turn 起始前缀命中量(AGT-12 真源;`turn` 不匹配 = None)
    pub fn active_cached_len(&self, turn: u64) -> Option<usize> {
        self.turns.active_of(turn).map(|a| a.cached_start)
    }

    /// 当前活跃 turn id(actor 侧"该发缓存纠正帧了吗"判据)
    pub fn active_turn_id(&self) -> Option<u64> {
        self.turns.active.as_ref().map(|a| a.id)
    }

    /// 推进到下一个事件点(每调用 = 活跃 turn 的一个采样步,或 turn 的
    /// 启动切换;`Idle` = 队列排空且无活跃,调用方可安全挂起等新提交)。
    /// **S0 调度/执行分离**:本函数 = 调度(纯主机侧决策,零设备 IO)
    /// + 执行(设备侧照办)—— 决策面可单测,执行器无决策权
    ///   (调度层铺开 §S0;对标 vLLM v1 schedule/execute 分离)。
    ///
    /// 相位定相:**引擎 env + 各 GraphPlan 的 env 快照**同步。
    /// 相位是 perf tag 维度(`interp.host.node.<phase>.<核>`),GraphPlan
    /// 构造期只拿到 env 快照 ⇒ 逐相位显式下发(perf-framework.md §十.3)。
    pub(crate) fn set_phase(&mut self, p: owl_models::env::EvalPhase) {
        self.env.phase = p;
        self.graphs.set_phase(p);
    }

    pub async fn pump(&mut self) -> Result<TurnEvent> {
        // E5-M2:spec 轮多 token 事件队列优先排空(schedule 前 —— 事件序
        // 不可与下一轮调度交织)
        if let Some(ev) = self.turns.pending_events.pop_front() {
            return Ok(ev);
        }
        let prof = self.cfg.knobs.probes.step_profile;
        // metrics 框架接线(2026-10-01 合并示范):schedule 分相计时入
        // owl_shared 全局 store(debug 展开/release 零开销);prof 时查询
        // 打印。其余 STEP_PROFILE 打点保留原样(gl-prof/srv-timing 工具链
        // 依赖其输出格式,逐点收编挂账)。
        owl_shared::timer_start!("engine.pump.schedule");
        let t_sched = (prof || true).then(std::time::Instant::now);
        let out = self.schedule()?;
        if let Some(ts) = t_sched {
            let d = ts.elapsed();
            self.sched_acc_ns += d.as_nanos() as u64;
            self.sched_cnt += 1;
            if self.sched_cnt.is_multiple_of(16) {
                eprintln!(
                    "[sched] n={} avg={:.2}ms last={:.2}ms",
                    self.sched_cnt,
                    self.sched_acc_ns as f64 / self.sched_cnt as f64 / 1e6,
                    d.as_secs_f64() * 1e3
                );
            }
        }
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
                        // 性能面:相位定相 + 包络 span(perf-framework.md §四)。
                        // 引擎 actor = current_thread 运行时 ⇒ 单线程栈配对安全;
                        // 解释器节点钩子按 env.phase 打 tag,两者同源。
                        let saved = self.env.phase;
                        self.set_phase(owl_models::env::EvalPhase::Prefill);
                        owl_shared::timer_start!("engine.phase.prefill");
                        let r = self.execute_prefill(chunk_ids, base, is_last).await;
                        owl_shared::timer_end!("engine.phase.prefill");
                        self.set_phase(saved);
                        r
                    }
                    StepAction::Decode { token, pos, kv_slot, gdn_slot, grew } => {
                        let saved = self.env.phase;
                        self.set_phase(owl_models::env::EvalPhase::Decode);
                        owl_shared::timer_start!("engine.phase.decode");
                        let r = self.execute_decode(token, pos, kv_slot, gdn_slot, grew).await;
                        owl_shared::timer_end!("engine.phase.decode");
                        self.set_phase(saved);
                        r
                    }
                    StepAction::SpecRound { token, pos, kv_slots, gdn_slot, grew } => {
                        let saved = self.env.phase;
                        self.set_phase(owl_models::env::EvalPhase::Spec);
                        owl_shared::timer_start!("engine.phase.spec");
                        let r =
                            self.execute_spec_round(token, pos, kv_slots, gdn_slot, grew).await;
                        owl_shared::timer_end!("engine.phase.spec");
                        self.set_phase(saved);
                        r
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
                TurnEvent::Completed { turn, text, .. } if turn == id => return Ok(text),
                TurnEvent::Idle => {
                    return Err(ModelError::Msg(format!("generate: turn {id} 队列丢失")))
                }
                _ => {}
            }
        }
    }
}
