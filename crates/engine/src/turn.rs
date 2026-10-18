//! 请求面词汇:Session / Turn / Step 三层(2026-09-26 用户裁定)。
//!
//! - **Session**([`crate::session`]):agent 连续会话(多轮 prompt);
//!   引擎侧感知 = KV cache 区域 + 会话 token 账;
//! - **Turn**(本模块):用户发来一条新消息(全量 prompt)→ 一个推理
//!   生命周期 —— 提交后入队,引擎推进至产出文本或工具调用;
//! - **Step**:turn 内一次**推理段** —— 模型推理出工具调用 → 客户端
//!   执行 → 结果回传 → 继续推理 = 下一个 step(工具循环轮次)。
//!
//! 引擎内部还有更细的推进单位 —— **调度步**(pump 一次 = 一个采样步
//! 或一个 prefill 块),属实现细节不上层词汇。层级:Session ⊃ Turn ⊃
//! Step ⊃ 调度步。
//!
//! 并发形态:M0.5 = 单槽串行(一次一个 turn 占槽;submit 非阻塞入队,
//! pump 按事件点推进);真并发(多 turn 共 batch)随 M2 continuous
//! batching 换入 —— 词汇与 API 形状不变,只换引擎内部。

use std::collections::VecDeque;

use owl_iface::contract::Bytes;

use crate::sampler::SamplerCfg;

/// 提交面:一个用户请求(前端一个 turn)
#[derive(Clone, Debug)]
pub struct TurnSpec {
    pub prompt: String,
    /// 生成 token 上限(不含 prompt)
    pub max_new: usize,
    /// OpenAI 兼容 `ignore_eos`:true = 忽略 EOS 强制跑满 `max_new`
    /// (2026-10-13 落地)。**用途 = 测速口径**:decode 吞吐必须在不被
    /// "模型想不想说话"干扰的足量生成上测(词汤 OOD 输入下模型常吐
    /// 2 token 就 EOS → 旧口径读数是伪影,见坑册 §23/§24)
    pub ignore_eos: bool,
}

/// per-turn 采样**覆写**(部分字段;`None` 字段 = 继承引擎 `knobs.sampler`)。
/// 为什么不直接传整块 `SamplerCfg`:请求只带 `temperature/top_p/top_k`,
/// 而 `rep_penalty` 等档位旋钮住引擎侧 —— 整块传会**静默丢掉**后者。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TurnSampler {
    /// OpenAI `temperature`;0 = 本 turn 贪心
    pub temp: Option<f32>,
    /// OpenAI `top_k`
    pub topk: Option<usize>,
    /// OpenAI `top_p`
    pub topp: Option<f32>,
}

impl TurnSampler {
    /// 三字段全缺 = 请求没表达采样意图 ⇒ 完全走引擎缺省
    pub fn is_empty(&self) -> bool {
        self.temp.is_none() && self.topk.is_none() && self.topp.is_none()
    }
}

/// 提交面可选项(per-turn;AGT-06:**OpenAI 协议的采样意图逐请求生效**)。
/// 历史教训:这些参数曾被服务面"收下 + 告警但未生效" —— 客户端发来的
/// 采样意图被全局单源静默覆盖,是"模型看起来不听话"的头号制造机。
#[derive(Clone, Debug, Default)]
pub struct TurnOpts {
    /// 忽略 EOS 跑满预算([`TurnSpec::ignore_eos`] 同义;测速口径)
    pub ignore_eos: bool,
    /// 采样覆写(见 [`TurnSampler`])
    pub sampler: Option<TurnSampler>,
    /// 停止串(命中即停,输出截断于该串之前;空 = 不启用)
    pub stop: Vec<String>,
    /// RNG 种子覆写(`None` = turn id 派生;同 turn 重放仍确定)
    pub seed: Option<u64>,
    /// **直接显存地板**(设计卷 §2.4;`None` = 走引擎 `knobs.dense_floor`)。
    ///
    /// 语义 = 对 KVMem 式损失的耐受度:`Strict` = 全量 dense 驻留(禁选择层)。
    /// **本刀只实做 `Strict`** —— 非 `Strict` 在提交边界**响亮拒**(选择层
    /// B 轴未落地,地板语义无从兼现,§23.5)。
    pub dense_floor: Option<owl_store::TurnFloor>,
}

/// turn 终了原因(OpenAI `finish_reason` 真源)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// 模型自吐 EOS(自然停)
    Stop,
    /// 触到 max_new 预算(含 ignore_eos 强制跑满)
    Length,
}

/// 步级事件(pump 回吐;前端按此流式渲染)。**每 pump 必得一事件**
/// (Idle = 队列排空;Prefill = prompt 中段状态步),调用方循环 pump
/// 至 Idle 即引擎排空。
#[derive(Clone, Debug)]
pub enum TurnEvent {
    /// 引擎空闲(队列空且无活跃 turn;调用方可安全等待新提交)
    Idle,
    /// prompt 段推进中(块式喂入后 = 一个 chunk 落定;无文本产出)
    Prefill { turn: u64, fed: usize, total: usize },
    /// 采样步产出一个 token(delta = 解码后文本增量;多字节字符跨
    /// token 由全量重解差分兜住)
    Token { turn: u64, delta: String },
    /// turn 终了(text = 生成全文,eos 不入文;finish = 终了原因)
    Completed { turn: u64, text: String, finish: FinishReason },
    /// turn 失败(槽/执行错误;槽已释放,队列继续)
    Failed { turn: u64, err: String },
}

// ===========================================================================
// turn 运行账(2026-10-14 拆分自 running.rs:RunningEngine 万能结构体
// 按状态域拆解 —— 本节收编**排队/活跃/ id 发号/多 token 事件队列**)
// ===========================================================================

/// 队列中待跑的 turn(提交面快照)
pub(crate) struct Turn {
    pub(crate) id: u64,
    /// 归属会话(已解析 id;None 提交在 submit 时已建临时账)
    pub(crate) session_id: u64,
    /// 临时会话(终了即焚;行为同 M0.5 每 turn 全量重算)
    pub(crate) ephemeral: bool,
    pub(crate) spec: TurnSpec,
    pub(crate) prompt_ids: Vec<u32>,
    /// per-turn 可选项(AGT-06)
    pub(crate) opts: TurnOpts,
    /// 提交时前缀命中量(usage `cached_tokens` 真源;AGT-12)
    pub(crate) cached: usize,
}

/// 活跃 turn(占用执行位的完整运行态)
pub(crate) struct ActiveTurn {
    pub(crate) id: u64,
    pub(crate) session_id: u64,
    /// **GDN 行号(S2b:turn 域资源)** —— turn 起从 `StateStore::borrow_row()`
    /// 借,终了归还(`return_row`);跨 turn 的状态连续性由**边界快照**承担
    /// (§25.2),不再靠"会话钉一行"。idle 会话因此对设备**零占用**。
    pub(crate) gdn_slot: usize,
    pub(crate) ephemeral: bool,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) max_new: usize,
    /// 忽略 EOS 跑满预算(TurnSpec 透传;测速口径)
    pub(crate) ignore_eos: bool,
    /// 本 turn 采样参数(AGT-06;**提交时已与 knobs 合并**,见 scheduler)
    pub(crate) sampler: SamplerCfg,
    /// 本 turn 是否走采样路径(temp=0 或 flavor greedy 档 = 否)
    pub(crate) sampler_on: bool,
    /// 停止串(AGT-06;空 = 不启用)
    pub(crate) stop: Vec<String>,
    /// 已发射文本字节数(`decoded` 为全量;stop 回吐 holdback 用)
    pub(crate) emitted: usize,
    /// stop 命中位置(字节偏移;终了时据此截断 `Completed.text`)
    pub(crate) trim_text_to: Option<usize>,
    /// RNG 种子覆写(AGT-06 `seed`;None = turn id 派生)
    pub(crate) seed: Option<u64>,
    /// 本 turn **起始 fed**(= 真实前缀命中量:会话守卫命中或块级前缀
    /// 缓存命中;AGT-12 `cached_tokens` 真源 —— 后者在 submit 时尚未
    /// 决定,只有 begin 定相后才知)
    pub(crate) cached_start: usize,
    pub(crate) out: Vec<u32>,
    /// KV/prompt 推进指针(会话 turn 从 `cached_len` 起步 = 增量 prefill)
    pub(crate) fed: usize,
    pub(crate) decoded: String,
    /// spec 轮机器状态(**turn 生命周期**:随 ActiveTurn 创建、随 turn
    /// 终了销毁)。历史教训:这七件曾寄存引擎级,跨 turn 残留 ——
    /// spec_snap_valid 残留 true = 新 turn 用上 turn 快照折叠(GDN 污染,
    /// AL 2.97→0.49);B4 降级三态残留 = 下 turn 带病起步。生命周期卡对
    /// 后此类跨 turn 污染结构性消失(2026-10-10)。
    pub(crate) spec: TurnSpecState,
}

/// turn 域 spec 状态(生命周期 = ActiveTurn;见上)
#[derive(Default)]
pub(crate) struct TurnSpecState {
    /// spec 快照有效性(轮首拍;fold 后失效重拍;全接受轮保持)
    pub(crate) spec_snap_valid: bool,
    /// 已 propose 草稿(host 账;轮末产出轮首消费)
    pub(crate) spec_drafts_host: Option<Vec<u32>>,
    /// B4:m=0 连击计数(成功轮归零)
    pub(crate) spec_zero_streak: usize,
    /// B4:降级态(true = 调度发裸 Decode;草稿池逐 token 同步)
    pub(crate) spec_degraded: bool,
    /// B4:降级以来 decode 步数(探测周期判定)
    pub(crate) spec_steps_degraded: usize,
    /// B4:当前探测周期(失败 ×2 退避,上限 512)
    pub(crate) spec_probe_every: usize,
    /// B4 滚动 AL 窗口(2026-10-12):轮数 / 产出 token 数 —— 连击 streak
    /// 会被"偶发 m=1"破解(随机域 AL≈1.1 时 9.3 t/s 案),窗口均值
    /// < 盈亏线(3.6 tok/轮)即降级,不看连击
    pub(crate) recent_rounds: u32,
    pub(crate) recent_tokens: u32,
    /// B4 窗口内累计轮时(ms)—— ①**时间门**:超长轮下等满 16 轮太久
    /// (44ms×16=0.7s、150ms×16=2.4s)②**绝对吞吐**读数(2026-10-14)
    pub(crate) recent_ms: f32,
    /// B4 分母(`bare_cost_ewma_ms`)采样时的 ctx(pos)—— ctx 涨过 25%
    /// 即作废分母(换档后旧采样失真 = 判据偏软的主因,2026-10-14)
    pub(crate) bare_cost_ctx: usize,
    /// B4 动态盈亏线(2026-10-09):滚动轮成本 / 裸步成本 ewma(ms)。
    /// 盈亏平衡 avg = round_cost/bare_cost(常数 1.2 曾远低于真线 ~4,
    /// AL 1.2~4 区间亏本跑 spec = 9.4 t/s 案)
    pub(crate) round_cost_ewma_ms: f32,
    pub(crate) bare_cost_ewma_ms: f32,
    /// B4 分母**播种**(2026-10-17):`bare_cost_ewma_ms` 未测时(新进程首个 turn,
    /// 跨 turn 未保鲜过)先跑 [`SPEC_SEED_SAMPLES`](crate::turn::SPEC_SEED_SAMPLES) 步
    /// **裸 decode** —— 合法产出、不回滚,只为让盈亏线的分母是**实测**而不是兜底
    /// 常量(32k 实测:兜底 2.5 把 AL 2.25 的盈利档误判成亏本 ⇒ 整会话降级;
    /// 见 `roadmap.local/26-10-17-verify核级账-投机盈亏线.md` §3)。
    /// 非零 = 还欠的播种采样数
    pub(crate) bare_seed_left: usize,
    /// **本步是播种裸步**(scheduler 置 → `exec/phases.rs` 裸步臂消费清零)。
    /// 与降级裸步分开计账:两者执行同构(都要同步草稿池 + 计时进分母),
    /// 但播种步**不置** `spec_degraded`(会话仍是投机态)。
    pub(crate) bare_seed_step: bool,
    /// prefill 末行 hidden(首轮 propose 的 anchor_hidden;消费后置 None)
    pub(crate) spec_seed_hidden: Option<Bytes>,
}

/// 播种采样数(盈亏线分母;见 [`TurnSpecState::bare_seed_left`])。
/// 2 个样本 ≈ 首次窗口前把分母定住(轮/裸 比在 32k 与 2.9k 都是 ~1.9-2.0,
/// 逐步 jitter 由 α=0.15 的 EWMA 在后续降级期继续吸收)
pub(crate) const SPEC_SEED_SAMPLES: usize = 2;

/// turn 运行账(组合叶;RunningEngine 持有)。字段 pub(crate) 供
/// scheduler/exec 直读,turn 生命周期方法收拢在此。
pub(crate) struct TurnLedger {
    /// 待跑队列(pump 逐个取首)
    pub(crate) queue: VecDeque<Turn>,
    /// 活跃 turn(M0.5 单槽:None = 空位)
    pub(crate) active: Option<ActiveTurn>,
    /// turn id 发号器(进程内单调)
    pub(crate) next_id: u64,
    /// 多 token 轮的事件队列(pump 逐个出;SpecRound 一轮 m+1 token)
    pub(crate) pending_events: VecDeque<TurnEvent>,
    /// **准入等待**(§23.4 `Queue` 姿态):(turn, 起始时刻)。
    ///
    /// 单线程无定时器 ⇒ 重判时机 = **下一次 pump**(通常由"活跃 turn 终了"
    /// 或新提交驱动);超 `max_wait_ms` 即拒。这一限制已写进设计卷 §23.4。
    pub(crate) admit_wait: Option<(u64, std::time::Instant)>,
    /// 最近终了 turn 的 prompt token 数(诊断/无卡门校准;不参与语义)。
    pub(crate) last_prompt_len: Option<usize>,
}

impl TurnLedger {
    /// **活跃 turn**(显式 `Result`;2026-10-17 用户裁决:引擎内部不变量失守
    /// **不许 panic** —— 一路上抛到泵循环,由它打印 + 丢 turn + 优雅关停)。
    ///
    /// 判据:调用点若在"必然有活跃 turn"的相位之外被触发(或相位判定漂移),
    /// 这里给出**可打印、可归因**的错误,而不是把引擎线程打死。
    pub(crate) fn active(
        &self,
    ) -> std::result::Result<&ActiveTurn, owl_iface::contract::ModelError> {
        self.active.as_ref().ok_or_else(|| owl_iface::contract::ModelError::EngineInvariant {
            what: "无活跃 turn(调用点落在 turn 生命周期之外)".into(),
        })
    }

    /// 同上(可变)。
    pub(crate) fn active_mut(
        &mut self,
    ) -> std::result::Result<&mut ActiveTurn, owl_iface::contract::ModelError> {
        self.active.as_mut().ok_or_else(|| owl_iface::contract::ModelError::EngineInvariant {
            what: "无活跃 turn(调用点落在 turn 生命周期之外)".into(),
        })
    }

    pub(crate) fn new() -> Self {
        Self {
            queue: VecDeque::new(),
            active: None,
            next_id: 1,
            pending_events: VecDeque::new(),
            admit_wait: None,
            last_prompt_len: None,
        }
    }

    /// 号 +1
    pub(crate) fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// 活跃 turn 域 spec 状态。
    ///
    /// **显式 Result**(2026-10-17 用户裁决):无活跃 turn = 调用序错误,但**不许
    /// panic** —— 一路上抛到泵循环(打印 + 丢 turn + `shutdown.graceful()`)。
    pub(crate) fn tspec(
        &mut self,
    ) -> std::result::Result<&mut TurnSpecState, owl_iface::contract::ModelError> {
        self.active_mut().map(|a| &mut a.spec)
    }

    /// 只读版(同上)。
    pub(crate) fn tspec_ref(
        &self,
    ) -> std::result::Result<&TurnSpecState, owl_iface::contract::ModelError> {
        self.active().map(|a| &a.spec)
    }

    /// 排队中的 turn(按 id;usage 真账取用面)
    pub(crate) fn queued(&self, turn: u64) -> Option<&Turn> {
        self.queue.iter().find(|t| t.id == turn)
    }

    /// 活跃 turn(按 id 过滤)
    pub(crate) fn active_of(&self, turn: u64) -> Option<&ActiveTurn> {
        self.active.as_ref().filter(|a| a.id == turn)
    }
}
