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
    /// 渲染后 prompt 落盘诊断(原 OWL_PROMPT_DUMP;2026-10-15 收编)
    pub prompt_dump: bool,
    /// reset_gdn 后 GDN rec 格画像(原 OWL_GDN_DBG;§三十二)
    pub gdn_dbg: bool,
    /// prefill H2D 叶子后流栅栏(A13 竞态排障;原 OWL_PF_FENCE)
    pub pf_fence: bool,
    /// KV 页差分落点(原 OWL_KV_DUMP,同上收编)
    pub kv_dump_at: Option<usize>,
}

impl StepProbes {
    /// **入口侧组装器**:自全量配置对象显式组装(引擎内部零 env 读取)
    pub fn from_config(cfg: &owl_shared::config::OwlConfig) -> Self {
        let p = &cfg.probes;
        Self {
            step_profile: p.step_profile,
            gdn_dump: p.gdn_dump,
            // 既有命名:OWL_DEBUG(通用诊断;preset 的 [probes] debug 可覆盖)
            debug: p.debug || owl_shared::env_reader::flag("OWL_DEBUG"),
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
            prompt_dump: p.prompt_dump,
            gdn_dbg: p.gdn_dbg,
            pf_fence: p.pf_fence,
            kv_dump_at: p
                .kv_dump_at
                .or_else(|| owl_shared::env_reader::parse::<usize>("OWL_KV_DUMP")),
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

/// **生成预算的最小保留量**(tok):剩余额度低于它 = 真溢出(连一个像样的
/// 工具调用/短答都放不下)⇒ 拒绝,让客户端压缩上下文。
///
/// 取 1024:pi 的工具调用单轮实测 200-500 tok(含 reasoning 可到 1k+),1k 是
/// "还能干活"的下限;低于它就算跑起来也是半截输出。
pub const _MIN_GEN_TOKENS: usize = 1024;

/// **预算收缩**(纯函数;2026-10-15;同日修复回归:下限检查误杀小预算档):
/// `base + max_new ≤ max_seq` 是 KV 槽位硬约束,但**不因此拒绝** ——
/// 先按剩余额度收缩生成长度。
///
/// 语义两段(顺序有义,勿倒):
/// ① **请求本来就装得下 ⇒ 原样放行**(下限 `_MIN_GEN_TOKENS` 只约束
///    "需要收缩"的场景;曾无条件先查下限 ⇒ max_seq < 1024 的档位
///    任何 submit 必拒 —— 全部 spec e2e 小预算门被误杀的根因);
/// ② 装不下 ⇒ 收缩:`room = max_seq − base`;`room < _MIN_GEN_TOKENS`
///    ⇒ `Err((base, max_new))`(真溢出,客户端压缩上下文),否则
///    `Ok(room)`(此时 room < max_new,即收缩后额度)。
pub fn budget_clamp(
    base: usize,
    max_new: usize,
    max_seq: usize,
) -> std::result::Result<usize, (usize, usize)> {
    if base.saturating_add(max_new) <= max_seq {
        return Ok(max_new);
    }
    let room = max_seq.saturating_sub(base);
    if room < _MIN_GEN_TOKENS {
        return Err((base, max_new));
    }
    Ok(room)
}

impl<D: DeviceClient> RunningEngine<D> {
    /// 捕获三态结果(Captured = 回放态;EagerFallback = 同闭包直发)
    /// **face 句柄克隆**(HTTP 调试投影面出口;见 `apps/server` 的 `/debug/blocks`)
    pub fn face_clone(&self) -> Option<D> {
        self.graphs.face_clone()
    }

    pub fn capture_outcome(&self) -> &crate::graph_plan::PlanOutcome {
        &self.graphs.capture_outcome
    }

    /// 提交 turn(临时会话:每 turn 独立,终了即焚;行为同 M0.5)。
    /// 预算:prompt + 生成 ≤ max_seq_tokens(= KV 槽位);超出先收缩生成
    /// (见 [`budget_clamp`]),只剩 < 1k 才拒。
    /// turn 域 spec 状态(turn 生命周期;仅活跃 turn 可触 —— 轮执行期
    /// active 恒在;turn 外触碰 = 调用序错误)
    /// 活跃 turn 的 spec 状态(**显式 Result**;见 [`TurnLedger::tspec`])。
    pub(crate) fn tspec(&mut self) -> Result<&mut TurnSpecState> {
        self.turns.tspec()
    }
    /// 只读版(同上)。
    pub(crate) fn tspec_ref(&self) -> Result<&TurnSpecState> {
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
            // 诊断:最终 prompt 落盘([probes] prompt_dump;§三十六 多轮"无记忆"
            // 幻觉定位面 —— dump 渲染+wrap 后送模型的完整 prompt 字节)
            if self.probes.prompt_dump {
                let ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    // 系统时钟回拨兜底 → 0(非失败路径;仅用于 dump 文件名)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let path = format!("/tmp/owl-prompt-{ms}.txt");
                match std::fs::write(&path, &wrapped) {
                    Ok(_) => owl_shared::log_debug!("prompt-dump", "{path} ({}B)", wrapped.len()),
                    Err(e) => owl_shared::log_debug!("prompt-dump", "落盘失败 {e}"),
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
        // 预算(会话感知):base = max(prompt, cached,命中增量段时前缀仍占槽)
        // 不足 ⇒ **先收缩生成长度**(不是直接拒):agent 客户端(pi)每轮重发
        // 系统提示 + 工具 schema + 历史,工作区 `AGENTS.md`+skills 一注入
        // prompt 就 26k+,而它按 `models.json.maxTokens` 要 8192 ⇒ 旧行为
        // 直接 400(实测:工作区里一句"1+1=?"都发不出去)。收缩后仍能跑,
        // `finish_reason=length` 由客户端自行续写/压缩。
        let base = prompt_ids.len().max(cached_hit);
        let max_new = match budget_clamp(base, max_new, self.cfg.max_seq_tokens) {
            Ok(m) => {
                if m != max_new {
                    owl_shared::log_warn!(
                        "chat",
                        "预算收缩: prompt {} + 生成 {} > max_seq {} ⇒ 生成长度收缩到 {} tokens",
                        prompt_ids.len(),
                        max_new,
                        self.cfg.max_seq_tokens,
                        m
                    );
                }
                m
            }
            Err(_) => {
                // 连 [_MIN_GEN_TOKENS] 都放不下 ⇒ 真溢出:客户端该压缩上下文。
                // 报文保留"超预算"字样(server 面据此归一化为 context-overflow
                // 语义,客户端才会自动 compact;见 apps/server/src/openai.rs)
                return Err(ModelError::Msg(format!(
                    "submit: prompt {} + 生成 {} 超预算 {}",
                    prompt_ids.len(),
                    max_new,
                    self.cfg.max_seq_tokens
                )));
            }
        };
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

    /// **本步的设备槽需求**(§12.4 需求驱动保洁的**真源**)。
    ///
    /// 口径 = 与 `schedule()` 里 `ensure_for_len` 的 target **同一份算式**
    /// (块长走 [`crate::scheduler::RunningEngine::prefill_step_chunk`] 单源):
    /// - 队列里有待准入 turn ⇒ 目标 = 该 prompt 全长(准入那一刻就要覆盖);
    /// - 活跃 turn 在 prefill ⇒ 目标 = `fed + chunk`;
    /// - 活跃 turn 在 decode ⇒ 目标 = `pos + spec 深度上界 + 1`
    ///   (**上界是安全方向**:多算只是多腾几个槽(内容不丢),
    ///   少算才会退回 `ensure_for_len` 的同步驱逐 —— 那正是要消灭的);
    /// - 无活跃且队列空 ⇒ **零需求**(§12.4:不用固定水位预搬;
    ///   预搬是显式策略,留给将来 预热/B 轴)。
    ///
    /// 返回的是"**还要长多少槽**"(已扣当前表长),不是总量。
    ///
    /// 准入分支要查一次命中面(纯 host、与 `schedule` 同一条匹配路):
    /// Δ_j = `ceil(N_j/page) − 设备已驻留的命中块数`(§12.1)—— **host 命中的
    /// 块不算省槽**(回灌时照样占槽,少算就是回灌 Exhausted)。
    fn step_need(&mut self) -> Result<owl_store::Need> {
        use owl_store::NeedWhy;
        let page = self.pool.page.max(1);
        let slots_for = |tokens: usize| tokens.div_ceil(page);
        // ① 待准入(队列 front):准入即要覆盖整个 prompt
        if let Some(t) = self.turns.queue.front() {
            let (sid, plen) = (t.session_id, t.prompt_ids.len());
            let dev = {
                let hit = self.bank.leg_target.match_prefix(&t.prompt_ids)?;
                Self::reuse_face(&self.pool, &hit, plen, page).1
            };
            let have = self.bank.sessions.get(sid).map(|s| s.block_table.len()).unwrap_or(0);
            return Ok(owl_store::Need {
                slots: slots_for(plen).saturating_sub(dev).saturating_sub(have),
                why: NeedWhy::Admit,
            });
        }
        // ② 活跃 turn:prefill 用单源块长;decode 用 spec 深度上界
        if let Some(act) = self.turns.active.as_ref() {
            let target = match Self::prefill_step_chunk(
                act.fed,
                act.prompt_ids.len(),
                self.cfg.prefill_chunk,
            ) {
                Some(chunk) => act.fed + chunk,
                None => act.fed + self.spec.depth + 1, // 生成相位(act.fed = pos)
            };
            let have = self
                .bank
                .sessions
                .get(act.session_id)
                .map(|s| s.block_table.len())
                .unwrap_or(0);
            return Ok(owl_store::Need {
                slots: slots_for(target).saturating_sub(have),
                why: NeedWhy::Grow,
            });
        }
        // ③ 空转:零需求(不预搬)
        Ok(owl_store::Need { slots: 0, why: NeedWhy::Idle })
    }




    /// **决策单执行**(§21 裁决:**搬运由 engine 编排**;store 只出计划与记账)。
    ///
    /// 逐条:取计划(纯 host)→ **engine 自己发 `face.swap`** → 回执取台 → 记账。
    /// - `Busy`(判与搬之间状态变了)= **跳过该条**(不是错误);
    /// - 搬运失败 ⇒ `abort_*`(内容已随命令消失 ⇒ 条目删,不留假账)后上抛;
    /// - `Drop` 是纯 host(store 直接删条目),无需 face。
    async fn apply_verdict(
        &mut self,
        verdict: &owl_store::Verdict,
    ) -> Result<owl_store::Applied> {
        use owl_store::Sink;
        let mut a = owl_store::Applied::default();
        // ① 换出/驱逐
        for item in &verdict.out {
            match item.sink {
                Sink::Host => {
                    let (blocks, pieces, table) = {
                        match owl_store::plan_offload(&mut self.bank.leg_target, item.chain) {
                            Ok(p) => (p.blocks, p.pieces, p.table),
                            Err(owl_store::StoreError::Busy(_)) => {
                                a.skipped_busy += 1;
                                continue;
                            }
                            Err(e) => return Err(e.into()),
                        }
                    };
                    let face = self.graphs.session.face_mut();
                    let conn = match face
                        .swap(owl_iface::contract::SwapDir::Out, &pieces, Box::new(table))
                        .await
                    {
                        Ok(c) => c,
                        Err(e) => {
                            owl_store::abort_offload(&mut self.bank.leg_target, &blocks);
                            return Err(owl_iface::contract::ModelError::Msg(format!(
                                "换出 swap 失败(engine 编排):{e}"
                            )));
                        }
                    };
                    let table = owl_store::reclaim_slot_table(conn)?;
                    a.offloaded +=
                        owl_store::commit_offload(&mut self.bank.leg_target, &blocks, table)?;
                }
                Sink::Drop => match self.bank.leg_target.drop_cached(item.chain) {
                    Ok(()) => a.dropped += 1,
                    Err(owl_store::StoreError::Busy(_)) => a.skipped_busy += 1,
                    Err(e) => return Err(e.into()),
                },
            }
        }
        // ② 回灌(Verdict.in_:逐条一条命令;H4 的批量形态在 execute_begin)
        for chain in &verdict.in_ {
            let (cs, slots, pieces, table) = {
                match owl_store::plan_onload(&mut self.bank.leg_target, std::slice::from_ref(chain)) {
                    Ok(p) => (p.chains, p.slots, p.pieces, p.table),
                    Err(e) => return Err(e.into()),
                }
            };
            let face = self.graphs.session.face_mut();
            let conn = match face
                .swap(owl_iface::contract::SwapDir::In, &pieces, Box::new(table))
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    owl_store::abort_onload(&mut self.bank.leg_target, &cs, &slots);
                    return Err(owl_iface::contract::ModelError::Msg(format!(
                        "回灌 swap 失败(engine 编排):{e}"
                    )));
                }
            };
            let table = owl_store::reclaim_slot_table(conn)?;
            let _ = owl_store::commit_onload(&mut self.bank.leg_target, &cs, &slots, table)?;
            a.onloaded += 1;
        }
        Ok(a)
    }

    /// **池压力保洁**(§14 钩子 H1/H2;段边界调用)—— store 的插入语义:
    /// "给你一个需求,谁该离开、谁该进来?" 本函数只负责**问**与**搬**,
    /// 判定内容全在 store(`owl_store::plan` / `apply`)。
    ///
    /// §12.4 口径:**需求驱动** —— `need` 由 [`Self::step_need`] 与 `schedule()`
    /// 的同一份算式给出;空转(零需求)不搬。`need.slots == 0` = 直接返回
    /// (老路径零变化:宿主面关闭时同样直接返回)。
    /// 判定单里的 `deny` = **"即使搬空也不够"的真容量不足预警** ⇒ 响亮 WARN,
    /// 但 `out` 里能搬的照搬(下沉不丢内容);真拒绝由 `ensure_for_len` 兜底。
    async fn pool_housekeeping(&mut self, need: owl_store::Need) -> Result<()> {
        if !self.bank.leg_target.host_enabled() || need.slots == 0 {
            return Ok(());
        }
        let verdict = owl_store::plan(&self.bank.leg_target, need, &[]);
        if owl_shared::env_reader::flag("OWL_STORE_HOOK_DEBUG") {
            eprintln!(
                "[hook] {:?} need={} free={}/{} 判 out={} in={} deny={:?}",
                need.why,
                need.slots,
                self.bank.leg_target.slots_free(),
                self.bank.leg_target.slots_total(),
                verdict.out.len(),
                verdict.in_.len(),
                verdict.deny
            );
        }
        if verdict.is_noop() {
            return Ok(());
        }
        let (n_out, n_in) = (verdict.out.len(), verdict.in_.len());
        if let Some(why) = &verdict.deny {
            // 凑不够 = 真容量不足预警:照搬能搬的(下沉不丢内容),响亮留痕
            owl_shared::log_warn!("store", "钩子:需求凑不够(真容量不足预警):{why}");
        }
        let applied = self.apply_verdict(&verdict).await?;
        owl_shared::log_debug!(
            "store",
            "钩子({:?}) 判 out={n_out} in={n_in} ⇒ 搬:下沉 {} 驱逐 {} 回灌 {} 跳过活跃 {}",
            need.why,
            applied.offloaded,
            applied.dropped,
            applied.onloaded,
            applied.skipped_busy
        );
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.hook.out", applied.offloaded as u64, file!(), line!());
            m.counter_add("store.hook.drop", applied.dropped as u64, file!(), line!());
            m.counter_add("store.hook.in", applied.onloaded as u64, file!(), line!());
            m.counter_add("store.hook.busy", applied.skipped_busy as u64, file!(), line!());
        });
        Ok(())
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
        // **§14 钩子(段边界)**:需求驱动保洁(§12.4)—— 给 store 一个**真需求**
        // (本步要长的槽数,与 `schedule` 同一份算式),它答"谁走谁进"。
        //  · 有活跃/待准入 ⇒ H1(H1 准入)/ H2(增长),need = 真 delta;
        //  · 空转 ⇒ **零需求,不搬**(§12.4 废弃了"总槽/2"这类固定水位:
        //    预搬是显式策略,不是默认行为)。
        // 保洁**先于** `schedule` 里的 `ensure_for_len` ⇒ 正常路径下同步驱逐恒为空
        // (报告面 `ops.ensure_evicted` 可查)。
        let need = self.step_need()?;
        self.pool_housekeeping(need).await?;
        owl_shared::timer_start!("engine.pump.schedule");
        let t_sched = (prof || true).then(std::time::Instant::now);
        let out = self.schedule()?;
        if let Some(ts) = t_sched {
            let d = ts.elapsed();
            self.sched_acc_ns += d.as_nanos() as u64;
            self.sched_cnt += 1;
            if self.sched_cnt.is_multiple_of(16) {
                owl_shared::log_info!("sched", "n={} avg={:.2}ms last={:.2}ms",
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

#[cfg(test)]
mod budget_tests {
    //! 预算收缩单测(纯;工作区 pi 场景的实际数字)
    use super::{budget_clamp, _MIN_GEN_TOKENS};

    #[test]
    fn clamps_instead_of_rejecting() {
        // 工作区实测:prompt 26446 + 请求 8192 ⇒ 32768 装不下 → 收缩到 6322
        assert_eq!(budget_clamp(26446, 8192, 32768), Ok(32768 - 26446));
        // 装得下 ⇒ 原样(不动客户端请求)
        assert_eq!(budget_clamp(12723, 8192, 32768), Ok(8192));
        // 边界:刚好 = max_seq ⇒ 可用 0 …… 但 0 < 下限 ⇒ 拒
        assert_eq!(budget_clamp(32768, 8192, 32768), Err((32768, 8192)));
        // 剩余恰好 = 下限 ⇒ 放行(边界语义:≥ 下限即可)
        assert_eq!(budget_clamp(32768 - _MIN_GEN_TOKENS, 8192, 32768), Ok(_MIN_GEN_TOKENS));
        // 剩余 = 下限 − 1 ⇒ 拒(真溢出,客户端该压缩)
        assert_eq!(
            budget_clamp(32768 - _MIN_GEN_TOKENS + 1, 8192, 32768),
            Err((32768 - _MIN_GEN_TOKENS + 1, 8192))
        );
        // base 已 ≥ max_seq(saturating)⇒ 拒,不 panic
        assert!(budget_clamp(99999, 8192, 32768).is_err());
        // 回归门(2026-10-15):**装得下 ⇒ 原样放行**,下限不倒查 ——
        // 小预算档(max_seq < 1024)曾被无条件下限检查误杀,全部
        // spec e2e 门(spec_identity_gate 128 / awq27b 1024 / …)假红
        assert_eq!(budget_clamp(18, 48, 128), Ok(48));
        assert_eq!(budget_clamp(52, 16, 1024), Ok(16));
        assert_eq!(budget_clamp(0, 48, 64), Ok(48));
        // 装不下且 room < 下限 ⇒ 仍拒(round4 真溢出语义不变)
        assert_eq!(budget_clamp(18, 4096, 128), Err((18, 4096)));
    }
}
