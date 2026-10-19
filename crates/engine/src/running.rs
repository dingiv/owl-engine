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


use owl_iface::contract::{DeviceClient, ModelError};
use owl_models::tokenizer::Tokenizer;

use crate::engine::{EngineConfig, RankDev};
use crate::scheduler::{SchedulerOutput, StepAction};
use crate::session::SessionBank;
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
    /// OWL_GDN_DUMP(GDN 行画像)
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

/// `/tokenize` 应答面(G5③;引擎侧唯一产地:token 序列 + 上下文上限)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TokenizeReply {
    pub tokens: Vec<u32>,
    /// 本档正文长度上限(token;服务面字段名 `max_model_len`)
    pub max_seq_tokens: usize,
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
pub struct RunningEngine<D: DeviceClient> {    /// 步级探针(boot 解析;不可变)
    pub(crate) probes: StepProbes,
    pub(crate) cfg: EngineConfig,
    pub(crate) tok: Tokenizer,
    /// boot 序号(metrics 打点命名空间;进程内自增,b1/b2…)
    pub(crate) boot_seq: u64,
    /// E5 性能:schedule 累计账(轮间 host 成本剖析)
    pub(crate) sched_acc_ns: u64,
    pub(crate) sched_cnt: usize,
    /// turn 运行账(turn.rs)
    pub(crate) turns: TurnLedger,
    /// 会话 + 块账房(session.rs;host 侧 —— 两 rank 共用一份)
    pub(crate) bank: SessionBank,
    /// **rank0 设备束**(池/图/模型/env/spec 运行态/tp;S4 ⑤-K3 收编)
    pub(crate) dev: RankDev<D>,
    /// **TP peer 束(rank1;`None` = 单卡)** —— 设备侧程序镜像的目标
    /// (S4 ⑤-K3:同一份设备侧代码,换装到 peer 束上再跑一遍)
    pub(crate) peer: Option<RankDev<D>>,
    /// **B 轴覆盖率探针**待分析面(Q 已回读;等块边界事件补齐索引账)
    /// —— `OWL_COVERAGE_PROBE`;缺省 `None`(生产零开销)。
    pub(crate) cov_pending: Option<crate::exec::coverage::CoveragePending>,
    /// 覆盖率探针最近一次观测(测试/调试读数面;`None` = 未观测过)
    pub(crate) cov_last: Option<crate::exec::coverage::CoverageReport>,
    /// **B-3d 选择层视图态**(召回点产出;`select_decode` 档每步展开成视图页表)
    pub(crate) sel_view: Option<SelectView>,
    /// 视图**生效**的 decode 步数(证据面:该臂真的走了视图)
    pub(crate) select_view_steps: usize,
    /// 视图输入槽**回退为全量链**的步数(视图态缺位;与上者和 = decode 步数)
    pub(crate) sel_view_fallback: usize,
    /// **B-3e stage-in 实测账**(回灌延迟定档;见 `StageStats`)
    pub(crate) stage: StageStats,
    /// **B-3e-2 部分驻留账**(召回点下沉/回灌/不可沉计数)
    pub(crate) partial: PartialStats,
    /// **B 轴相位账**(性能面;纯计时零语义;`PartialStats` 同款读数面)
    pub(crate) prof: BAxisProfile,
    /// **上次完整召回的选择**(内容身份,时间序)。
    ///
    /// 用途:**廉价边界**的驻留调和用**冻结名单**,而不是重新打分 ——
    /// 也**不能是"什么都不做"**:`ensure_for_len` 驱逐不了会话自己持有的页,
    /// 链每 chunk 要新增 `chunk/page` 页,不腾就必然 `Exhausted`
    /// (实测:池 96 页,链要到 100 页 ⇒ 耗尽)。
    pub(crate) sel_keep: Vec<owl_store::ChainHash>,
}

/// **stage-in 实测账**(B-3e;宿主 → 设备回灌的块数/字节/耗时)。
///
/// 用途:把"恢复延迟"从论文口号变成我们的数字(论文 174ms@256K ⇒ 我们的
/// **选中集**回灌:W 块 × 每块字节)。
#[derive(Debug, Clone, Copy, Default)]
pub struct StageStats {
    /// 回灌块数 / 字节
    pub blocks: usize,
    pub bytes: usize,
    /// 回灌墙钟(ns;**含** `face.swap` 的发射与回执)
    pub nanos: u64,
    /// 来源拆分:召回前置(prefill 段边界)vs 决策单(verdict.in_)
    pub from_prefill: usize,
    pub from_verdict: usize,
}

#[cfg(any(test, feature = "e2e"))] // e2e 场景面也消费(渲染行/吞吐)
impl StageStats {
    /// 每块字节(0 = 无样本)。
    pub fn bytes_per_block(&self) -> usize {
        if self.blocks == 0 { 0 } else { self.bytes / self.blocks }
    }
    /// 吞吐(MB/s;0 = 无样本)。
    pub fn mib_per_s(&self) -> f64 {
        if self.nanos == 0 {
            0.0
        } else {
            self.bytes as f64 / 1048576.0 / (self.nanos as f64 / 1e9)
        }
    }
    /// 人类可读一行。
    pub fn line(&self) -> String {
        format!(
            "stage-in:{} 块 / {:.1}MiB / {:.1}ms(每块 {:.0}KiB · {:.0}MiB/s;前置 {} 决策 {} )",
            self.blocks,
            self.bytes as f64 / 1048576.0,
            self.nanos as f64 / 1e6,
            self.bytes_per_block() as f64 / 1024.0,
            self.mib_per_s(),
            self.from_prefill,
            self.from_verdict
        )
    }
}

/// **B-3e-2 部分驻留账**(召回点下沉 / 回灌 / 不可沉)。
#[derive(Debug, Clone, Copy, Default)]
pub struct PartialStats {
    /// 召回点下沉的块数(会话放手 → 缓存独占 → 宿主)
    pub sunk: usize,
    /// 召回点回灌的块数(选中但内容在宿主)
    pub staged: usize,
    /// **本轮不可沉**的块数(还没进缓存树 ⇒ 无宿主键;见 §8.19 第 2b 项)
    pub not_cacheable: usize,
    /// **prefill 子集表**生效的 chunk 数(= 链上有洞后才换表的次数;第 3 项证据)
    pub prefill_subset: usize,
}

/// **B 轴相位账**(性能面;**纯计时,零语义**)——
///
/// 为什么单独立账:这块是 host 纯逻辑(索引/打分/选择/视图),它不在任何
/// 设备核的计时里,却直接吃掉 prefill 的墙钟。相位口径与
/// `coverage_analyze` / `collect_index_for_block` 的段落一一对应:
///
/// | 相位 | 含义 |
/// |---|---|
/// | `index_ms` | 索引采集(逐块:设备 K 回读 + host 逆 RoPE + 累加) |
/// | `q_face_ms` | query 面取回(块边界 Q dtoh + 逐行逆 RoPE) |
/// | `mass_ms` | 重诊断采样段(K/V 全量回读 + host 逐元素解码) |
/// | `score_ms` | 打分(**生产面**;快路口径) |
/// | `paper_ms` | 打分(论文口径保真对照;**诊断面**) |
/// | `select_ms` | 选择(预算 + sink/recent) |
/// | `view_ms` | 视图装配(页表写 + 引擎视图态) |
/// | `reconcile_ms` | 召回点驻留调和(下沉/回灌) |
#[derive(Debug, Clone, Copy, Default)]
pub struct BAxisProfile {
    /// 走过的块边界数(每个边界 = 一次召回点)。
    pub boundaries: usize,
    /// 其中**完整召回**(打分+按分选择)的次数。
    pub recall_full: usize,
    /// 其中**廉价召回**(无打分近期选择)的次数。
    pub recall_cheap: usize,
    pub index_ms: f64,
    pub q_face_ms: f64,
    /// 查询面细分:设备往返(含读语义的 COMPUTE 排空)
    pub q_toh_ms: f64,
    /// 查询面细分:host 解码(f16→f32)
    pub q_dec_ms: f64,
    /// 查询面细分:Q 块 `free`(释放命令墙钟;从 `q_dec_ms` 窗口里剔出单记 ——
    /// 原口径把释放命令算进"host 解码",读数虚高)
    pub q_free_ms: f64,
    /// 查询面细分:host 逆 RoPE(逐行)
    pub q_derope_ms: f64,
    /// 查询面细分:**真实质量**(诊断面;逐 query 行 × 层 × 头 扫全 ctx)
    pub q_mass_ms: f64,
    pub mass_ms: f64,
    pub score_ms: f64,
    pub paper_ms: f64,
    pub select_ms: f64,
    pub view_ms: f64,
    pub reconcile_ms: f64,
    pub total_ms: f64,
}

#[cfg(any(test, feature = "e2e"))]
impl BAxisProfile {
    /// 人类可读一行(各相位 + 占比)。
    pub fn line(&self) -> String {
        let pct = |v: f64| {
            if self.total_ms > 0.0 { v / self.total_ms * 100.0 } else { 0.0 }
        };
        format!(
            "B 轴相位 ×{} 边界:合计 {:.1}ms | 索引 {:.1}({:.0}%) 查询面 {:.1}({:.0}%) \
             重诊断 {:.1}({:.0}%) 打分 {:.1}({:.0}%) 论文口径 {:.1}({:.0}%) \
             选择 {:.1}({:.0}%) 视图 {:.1}({:.0}%) 调和 {:.1}({:.0}%) \
             ‖ 查询面拆:{:.1} 设备往返 + {:.1} host 解码 + {:.1} 释放 {:.1} 逆RoPE + {:.1} 真实质量 \
             ‖ 召回档:完整 {} · 廉价 {}",
            self.boundaries,
            self.total_ms,
            self.index_ms, pct(self.index_ms),
            self.q_face_ms, pct(self.q_face_ms),
            self.mass_ms, pct(self.mass_ms),
            self.score_ms, pct(self.score_ms),
            self.paper_ms, pct(self.paper_ms),
            self.select_ms, pct(self.select_ms),
            self.view_ms, pct(self.view_ms),
            self.reconcile_ms, pct(self.reconcile_ms),
            self.q_toh_ms,
            self.q_dec_ms - self.q_free_ms,
            self.q_free_ms,
            self.q_derope_ms,
            self.q_mass_ms,
            self.recall_full,
            self.recall_cheap
        )
    }
}

#[cfg(any(test, feature = "e2e"))]
impl PartialStats {
    /// 人类可读一行。
    pub fn line(&self) -> String {
        format!(
            "部分驻留:下沉 {} 块 · 回灌 {} 块 · 不可沉(未入缓存树){} 块 · prefill 子集表 {} 次",
            self.sunk, self.staged, self.not_cacheable, self.prefill_subset
        )
    }
}

/// **选择层视图态**(B-3d;方案 A = 原位置,不做 re-RoPE)。
///
/// 视图 = **选中块(时间序)++ 召回点之后新生成的尾部**:
/// - 尾部必须带(否则模型看不见自己刚写的 token);
/// - 尾部在会话链里天然连续,且召回点落在**块边界** ⇒ 尾部从整页起;
/// - `kv_lens`(注意力上下文长)= 选中 token 数 + `pos+1 − recall_pos`。
#[derive(Debug, Clone)]
pub struct SelectView {
    /// 选中块的物理页(时间序)。
    pub pages: Vec<u32>,
    /// 选中块的 token 数(= `pages.len() × page`)。
    pub tokens: usize,
    /// 召回点绝对位置(块边界;尾部从这里起)。
    pub recall_pos: usize,
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
    /// 本档 prefill 块长(探针排版用;与调度器取**同一份配置** ⇒ 不会对不上)
    #[cfg(any(test, feature = "e2e"))]
    pub(crate) fn cfg_prefill_chunk(&self) -> usize {
        self.cfg.prefill_chunk
    }

    /// 捕获三态结果(Captured = 回放态;EagerFallback = 同闭包直发)
    /// **face 句柄克隆**(HTTP 调试投影面出口;见 `apps/server` 的 `/debug/blocks`)
    pub fn face_clone(&self) -> Option<D> {
        self.dev.graphs.face_clone()
    }

    pub fn capture_outcome(&self) -> &crate::graph_plan::PlanOutcome {
        &self.dev.graphs.capture_outcome
    }

    /// 是否带 TP peer 束(双 rank 执行态;单卡 = false)
    pub fn has_peer(&self) -> bool {
        self.peer.is_some()
    }

    /// **诊断出口**:读主 rank(rank0)会话图的命名输出(如 `logits`/`token`)。
    /// `None` = 该图无此输出槽。G8 对拍/数值归因用(S4 ⑤-K3)。
    pub async fn read_output_f32(&mut self, name: &str) -> Result<Option<Vec<f32>>> {
        if !self.dev.graphs.session.has_output(name) {
            return Ok(None);
        }
        Ok(Some(self.dev.graphs.session.read_output_f32(name).await?))
    }

    /// **诊断出口**:读 peer(rank1)会话图输出(TP 对拍;**单卡 = None**)。
    /// 用途:G8 的"两 rank 同值"实证(AR 后两 rank 应**逐位相同**)。
    pub async fn read_peer_output_f32(&mut self, name: &str) -> Result<Option<Vec<f32>>> {
        let Some(p) = self.peer.as_mut() else { return Ok(None) };
        if !p.graphs.session.has_output(name) {
            return Ok(None);
        }
        Ok(Some(p.graphs.session.read_output_f32(name).await?))
    }

    /// **会话 token 账**(诊断/门面:prompt + 已生成,逐 token;S4 ⑤-K3 的 G8
    /// 判据面 —— 与 `gpu_spec_identity_gate` 同一条 token 级硬判)。`None` = 无此会话。
    pub fn session_token_ids(&self, sid: u64) -> Option<Vec<u32>> {
        self.bank.sessions.get(sid).map(|s| s.tokens.clone())
    }

    /// **tokenize 出口**(G5③:服务面 `POST /tokenize`):纯 CPU,不落 turn、
    /// 不碰池/图。`chat_wrap = true` = 先套家族单轮包装(与裸文本提交同一律,
    /// 见 [`owl_models::tokenizer::Tokenizer::chat_wrap`]);否则纯文本编码。
    /// 编码失败**显式上抛**(不用会吞错误的 `encode` —— 空序列 ≡ “文本是空的”)。
    pub fn tokenize(&self, text: &str, chat_wrap: bool) -> std::result::Result<TokenizeReply, String> {
        let tokens = if chat_wrap {
            self.tok.try_encode(&self.tok.chat_wrap(text))?
        } else {
            self.tok.try_encode(text)?
        };
        Ok(TokenizeReply { tokens, max_seq_tokens: self.cfg.max_seq_tokens })
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
        // 准入地板(S3b §23.5 / A7-bis 接线 2026-10-19):
        //
        // 非 Strict 地板 = "允许选择层按需物化" ⇒ **必须有执行面**才收:
        // 判据 = 引擎的**虚拟化两扇门**(`select_decode` + `select_partial`,
        // 唯一产地 `kvmem_virtualization`)。没开就响亮拒(接受=静默降级)。
        // 开着的档位:地板进 `admission_gate`(准入判据 + 实得地板回吐),
        // 页数上限在准入处按**本档工作集**再校一遍(ctx 长度那时才知道)。
        if let Some(f) = opts.dense_floor {
            if !f.is_strict() && !self.cfg.knobs.kvmem_virtualization {
                return Err(ModelError::Msg(
                    "submit: per-turn 非 Strict 地板需要**虚拟化两扇门**\
                     ([kvmem].select_decode = true **且** select_partial = true)——\
                     地板语义只有选择层在场才能兜住(设计卷 §2.4/§23.5);\
                     两者任一为假 = 响亮拒,不静默降级"
                        .into(),
                ));
            }
        }
        // 预算(会话感知):命中账本的 turn 按 max(prompt, cached) 记账
        // (增量段写 `[cached, prompt)` 行,生成续写其后;失配回退全量时
        // prompt 为准,同一上界)
        let cached_hit = {
            let probe = session
                .and_then(|id| self.bank.sessions.get(id).map(|s| (id, s.cached_len, s.guard(&prompt_ids))));
            match probe {
                Some((_, cached, true)) => cached,
                // **守卫失配取证**(`[probes] debug`;§4.3):报首个分叉的**位置 +
                // 两侧 id 窗口**。渲染对称律的判据就是这一行:分叉在
                // "assistant 轮边界" = 渲染/切分不对称;分叉在别处 = 客户端回填不同。
                Some((id, cached, false)) => {
                    if self.probes.debug {
                        let s = self.bank.sessions.get(id).expect("刚查过");
                        match s.first_divergence(&prompt_ids) {
                            Some(i) => {
                                let lo = i;
                                let hi_old = (i + 8).min(s.tokens.len());
                                let hi_new = (i + 8).min(prompt_ids.len());
                                let (a, b) = (&s.tokens[lo..hi_old], &prompt_ids[lo..hi_new]);
                                eprintln!(
                                    "[dbg prefix] 守卫失配 session={id} cached_len={cached} 首个分叉 idx={i} \
                                     旧(已入 KV)={a:?} ⇒ {:?} ‖ 新(本轮)={b:?} ⇒ {:?}",
                                    self.tok.decode(a),
                                    self.tok.decode(b)
                                );
                            }
                            None => eprintln!(
                                "[dbg prefix] 守卫失配 session={id} cached_len={cached} 但前缀全同\
                                 (长度不足?prompt={} )",
                                prompt_ids.len()
                            ),
                        }
                    }
                    0
                }
                // 无会话 → cached 0(语义默认,非失败)
                None => 0,
            }
        };
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
            // S2b:建会话不再占 GDN 行(行随 turn 借还)⇒ 无逐出重试
            Some(id) => (self.bank.get_or_create(Some(id))?, false),
            None => (self.bank.get_or_create(None)?, true),
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

    /// **最近一次终了 turn 的 prompt token 数**(无卡门/诊断用:turn 终了后
    /// `turn_prompt_len` 已查不到,而测试要按"prompt 是否 32 对齐"做校准)。
    pub fn last_turn_prompt_len(&self) -> Option<usize> {
        self.turns.last_prompt_len
    }

    /// 队列长度(待跑 turn 数;服务面诊断/准入门读)。
    pub fn queue_len(&self) -> usize {
        self.turns.queue.len()
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
    /// **设备归属/类别上下文**(生命周期节点 N1/N4/N5)。
    ///
    /// 两 rank 一起设(上下文各引擎一份、账本共用):只设 rank0 会让 rank1 的
    /// 分配归错 owner —— 而这正是"谁占着多少"要回答的问题。
    pub(crate) fn set_dev_ctx(&mut self, owner: owl_iface::handle::Owner, class: owl_iface::handle::AssetClass) {
        for dev in std::iter::once(&mut self.dev).chain(self.peer.as_mut()) {
            let f = dev.graphs.dev_ctx_face_mut();
            f.set_owner(owner);
            f.set_class(class);
        }
    }

    /// **N6 压力探测**(缺口 3;2026-10-19):驱动余量逼近 reserve ⇒ 返回"目标腾出槽数"。
    ///
    /// 判据(全部实测量,零猜测):`free < reserve` 才报压力;目标 = 把 free 拉回
    /// **2×reserve**(留一次呼吸)所需的块数,按池块字节折算,夹在 `[1, 64]`
    /// (一次 N6 是"整理"不是"搬家";不够就下一拍再来)。无设备账(CPU mock)⇒ None。
    fn pressure_probe(&self) -> Option<usize> {
        let free = owl_shared::vram::free_bytes();
        if owl_shared::vram::total_bytes() == 0 {
            return None;
        }
        let reserve = (self.cfg.knobs.vram_reserve_mb as u64) << 20;
        let target = reserve.saturating_mul(2).max(1 << 30);
        if free >= target {
            return None;
        }
        let per = self.dev.pool.block_bytes().max(1) as u64;
        let want = ((target - free) / per) as usize;
        Some(want.clamp(1, 64))
    }

    /// **句柄计数三元组**(诊断面;`/debug/vram`):`(hook_fail, window_unclosed, 窗内块数)`
    ///
    /// 判据:三者非零 = 有事件没入账 / 有窗未收 —— **可见**比"账看起来平"更重要。
    pub fn handle_probe(&self) -> (u64, u64, usize) {
        self.dev.graphs.dev_ctx_face().ledger_probe()
    }

    /// **块账房快照(逐 rank)**:`/debug/vram` 与事件账做**全等门**的右值。
    ///
    /// 返回逐 rank 的 `Result`(不合并成败):"后端不支持"与"真出错"都如实带出文本,
    /// 由报告面分别标——**不静默降级成 None**。
    pub async fn block_ledger_probe(
        &mut self,
    ) -> Vec<(u8, std::result::Result<Vec<owl_iface::contract::BlockLedgerRow>, String>)> {
        #[allow(clippy::type_complexity)]
        let mut out: Vec<(u8, std::result::Result<Vec<owl_iface::contract::BlockLedgerRow>, String>)> =
            Vec::new();
        let ranks: Vec<(u8, &mut RankDev<D>)> = {
            let mut v: Vec<(u8, &mut RankDev<D>)> = vec![(0, &mut self.dev)];
            if let Some(p) = self.peer.as_mut() {
                v.push((1, p));
            }
            v
        };
        for (rank, dev) in ranks {
            let r = dev
                .graphs
                .dev_ctx_face_mut()
                .block_ledger_dump()
                .await
                .map_err(|e| e.to_string());
            out.push((rank, r));
        }
        out
    }

    /// **N4 TurnEnd**:turn 收尾后的归属 —— 会话仍活 ⇒ `Session(s)`(它持有 KV/快照
    /// 与 GDN 行,跨 turn),临时会话焚毁 ⇒ `System`;类别回 `Unclassified`
    /// (turn 相位结束,后续分配未分类就**可见地**落 Unclassified 桶)。
    pub(crate) fn set_ctx_after_turn(&mut self, session_id: u64, burned: bool) {
        let owner = if burned {
            owl_iface::handle::Owner::System
        } else {
            owl_iface::handle::Owner::Session(session_id)
        };
        self.set_dev_ctx(owner, owl_iface::handle::AssetClass::Unclassified);
    }

    pub(crate) fn set_phase(&mut self, p: owl_models::env::EvalPhase) {
        self.dev.env.phase = p;
        self.dev.graphs.set_phase(p);
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
        let page = self.dev.pool.page.max(1);
        let slots_for = |tokens: usize| tokens.div_ceil(page);
        // ① 待准入(队列 front):准入即要覆盖整个 prompt
        if let Some(t) = self.turns.queue.front() {
            let (sid, plen) = (t.session_id, t.prompt_ids.len());
            let dev = {
                let (hit, recall) = self.bank.rel.match_prefix(&t.prompt_ids)?;
                Self::reuse_face(&self.bank.rel, &hit, &recall, plen, page).1
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
                self.cfg.knobs.prefill_tail_merge,
                Self::prefill_align_page(
                    self.cfg.knobs.prefill_align_tail,
                    self.cfg.knobs.prefix_cache,
                    self.dev.pool.page,
                ),
            ) {
                Some(chunk) => act.fed + chunk,
                None => act.fed + self.dev.spec_rt.depth + 1, // 生成相位(act.fed = pos)
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

    /// **N4 池清点**(缺口 4 的节点面):把**已在 store 池里**的 `Turn(t)` 名下块
    /// 改归属 `Cache`(池 = 共享复用面,不再是某 turn 的临时面),返回 `(块数, 字节)`。
    ///
    /// 安全性:池成员**按构造**无活持有者(只在 `free` 路径入池)⇒ 换归属不改任何
    /// 持有事实;而"真没归还"的块(不在池里)**不换** —— 它们继续归 `Turn(t)`,
    /// 继续在 N4 清单里点名 ⇒ **"池内复用"与"泄漏"从此可分**。
    pub(crate) fn n4_pool_sweep(&self, turn: u64) -> (usize, usize) {
        let cell = self.bank.rel.ledger_cell();
        let mut b = owl_store::book_lock(&cell);
        let swept = b.pool_sweep(
            owl_iface::handle::Owner::Turn(turn),
            owl_iface::handle::Owner::Cache,
        );
        if swept.0 > 0 {
            let (n, bytes) = b.pool_stats();
            let (hits, misses, _) = b.pool_counters();
            owl_shared::log_info!(
                "store",
                "N4 池清点:turn {turn} 归还 {} 块 / {} KiB 入复用池 ⇒ 池 {} 块 / {} KiB(命中 {hits} / 未中 {misses})",
                swept.0,
                swept.1 / 1024,
                n,
                bytes / 1024
            );
        }
        swept
    }

    /// **N4 未归还清单**(store 立案 §缺口1 的第一半;2026-10-19)。
    ///
    /// turn 收尾时按**块**清点 `Turn(t)` 名下**仍开着**的块:非空 ⇒ 响亮 WARN,
    /// 名字直接指向分配点(哪位调用点要的、要了几块、共多少字节)。这就是
    /// "为什么每轮 +N"的现场 —— 没有它,泄漏只有 `/debug/vram` 的聚合数字,
    /// 定位就得开 `trace_ring` 拼发射账。
    pub(crate) fn n4_report_unreturned(&self, turn: u64) {
        let cell = self.bank.rel.ledger_cell();
        let loans = owl_store::book_lock(&cell)
            .open_loans(|o| o == owl_iface::handle::Owner::Turn(turn));
        if loans.is_empty() {
            return;
        }
        let mut agg: std::collections::BTreeMap<String, (usize, usize)> =
            std::collections::BTreeMap::new();
        for (_, l) in &loans {
            let e = agg.entry(l.name.clone()).or_insert((0, 0));
            e.0 += 1;
            e.1 += l.bytes;
        }
        let total_kib: usize = agg.values().map(|(_, b)| *b).sum::<usize>() / 1024;
        let detail: Vec<String> = agg
            .iter()
            .map(|(n, (c, b))| format!("{n} ×{c} / {}B", b))
            .collect();
        owl_shared::log_warn!(
            "store",
            "N4 未归还清单:turn {turn} 名下仍有 {} 块 / {total_kib} KiB 开着 —— {}",
            loans.len(),
            detail.join(" ｜ ")
        );
    }

    /// **store 报告面**(设计卷 §2.4 / §13.4 对账口):三层 + 两腿 +
    /// 保留/弹性/可动面。服务面 `/debug` 与准入门读这里(单一真源)。
    pub fn store_report(&self) -> owl_store::StoreReport {
        self.bank.rel.report()
    }

    /// **KV 腿报告**(槽/缓存/宿主/搬运计数 + 可动面;§13.4 对账口)。
    pub fn leg_stats(&self) -> owl_store::LegStats {
        self.bank.rel.leg.report()
    }

    /// **池账同步**(S3b §23.2):把池实占按"活跃面 / 可动面"拆给准入账 ——
    /// 真源 = 块腿报告的设备驻留字节 + 活跃 turn 的块表长(引擎知道)。
    ///
    /// 恒等式(必须成立,否则判据就偏):
    /// `reserved(活跃实占) + cache(其余池实占) = 池实占`。
    /// 单一活跃 turn ⇒ O(1);将来多活跃 = 求和(算式不变)。
    pub(crate) fn sync_pool_account(&mut self) -> Result<()> {
        let block_bytes = self.dev.pool.block_bytes();
        // 池实占 = **槽账**真源(已用槽 × 块字节)—— 不看树绑定面:
        // 尾部不满一块的前缀不进树,但槽已经占了(实测:8 会话占满 8 槽,
        // 绑定面只报 6 块 ⇒ 按绑定面判就会多放一个 turn 进来)。
        let occupied = self.bank.rel.leg.occupied_bytes();
        // 可动面 = 腿的**引用账**真源(refs = 1 的设备块;§23.2)——
        // 不用"池实占 − 活跃实占"那种字节相减:那会把 idle 会话持着的
        // 历史算成可动,准入就说满了话而搬不动(Busy ⇒ 退回 Exhausted)。
        let movable = self.bank.rel.leg.movable_bytes();
        let active = self.turns.active.as_ref().map(|a| (a.id, a.session_id));
        let held = match active {
            Some((_, sid)) => {
                self.bank.sessions.get(sid).map_or(0, |s| s.block_table.len()) * block_bytes
            }
            None => 0,
        }
        .min(occupied.saturating_sub(movable));
        let ad = &mut self.bank.rel.admission;
        ad.set_cache_bytes(movable); // 可动面(可下沉/可驱逐)
        ad.set_session_held(occupied.saturating_sub(movable).saturating_sub(held)); // 会话持有面
        if let Some((id, _)) = active {
            ad.set_floor(id, held)?; // 活跃 turn 保留额(G2b)
        }
        Ok(())
    }

    /// **实得地板回吐**(设计卷 §2.4 的"报告面"):一行说清
    /// **请求页 / 实得页 / 上下文页**——实得 < 请求 = 有耐受档的合法压缩
    /// (差额推给三层),但**必须看得见**(不许静默)。
    fn log_floor_grant(
        &self,
        turn_id: u64,
        floor: owl_store::TurnFloor,
        ctx_tokens: usize,
        page: usize,
        page_bytes: usize,
        granted_bytes: usize,
    ) {
        let req_pages = floor.floor_pages(ctx_tokens, page);
        let got_pages = granted_bytes.div_ceil(page_bytes.max(1));
        let ctx_pages = ctx_tokens.div_ceil(page.max(1));
        owl_shared::log_info!(
            "store",
            "准入：turn {turn_id} 地板请求 {req_pages} 页 / **实得 {got_pages} 页**\
             (={} KiB)‖ ctx {ctx_tokens} tok = {ctx_pages} 页 ‖ 档 {}",
            granted_bytes >> 10,
            if floor.is_strict() { "Strict" } else { "耐受" }
        );
    }

    /// **H1 准入判决**(S3b §23.4):待准入 turn 过 `Rel::check` ⇒
    /// 调需求(EvictThen)/ 拒(Reject)/ 留队(Queue)。
    ///
    /// 为什么先于 `ensure_for_len`:那条路是"**占了槽**才发现不够"(Exhausted,
    /// 中后段才炸,且已写半截账);本判决是"**判在前**"——不够就先腾,
    /// 腾不够就带数字拒。
    fn admission_gate(&mut self, need: &mut owl_store::Need) -> Result<()> {
        use owl_store::AdmissionVerdict;
        if need.why != owl_store::NeedWhy::Admit {
            return Ok(());
        }
        // **只在真的能起 turn 时判**(无活跃 turn)。有活跃 turn 时队首只是在
        // 等位 —— 此刻"池被活跃面占着"是常态(它马上会退成缓存),照判会
        // 把正常排队的第二个请求误拒。准入 = "谁能**开始**"。
        if self.turns.active.is_some() {
            return Ok(());
        }
        self.sync_pool_account()?;
        let Some((turn_id, ctx_tokens, floor)) = self.turns.queue.front().map(|t| {
            (
                t.id,
                // 上界 = prompt + 生成长度 + spec 深度(安全方向:多算只是多腾几个槽)
                t.prompt_ids.len() + t.spec.max_new + self.dev.spec_rt.depth,
                t.opts.dense_floor.unwrap_or(self.cfg.knobs.dense_floor),
            )
        }) else {
            return Ok(());
        };
        // **per-turn 地板的上限校验**(A7-bis):视图只能保持"本档工作集"那么多页
        // 常驻 ⇒ 请求地板 > 工作集 = 承诺无法兑现 ⇒ **响亮拒**(不静默夹）。
        // 只有 ctx 长度到了这里才知道,所以校在这里而不是 submit 处。
        let page = self.dev.pool.page.max(1);
        let page_bytes = self.dev.pool.block_bytes().max(1);
        let ws_pages = self.cfg.knobs.select_blocks.max(1);
        let req_pages = floor.floor_pages(ctx_tokens, page);
        if !floor.is_strict() && req_pages > ws_pages {
            let reason = format!(
                "per-turn 地板 {req_pages} 页 > 本档工作集 {ws_pages} 页(视图只能保这么多常驻)\
                 —— 调大 [kvmem].workspace_per_req 或调小请求地板"
            );
            self.reject_queued_turn(turn_id, format!("准入拒绝：{reason}"))?;
            return Ok(());
        }
        match self.bank.rel.check(floor, ctx_tokens) {
            AdmissionVerdict::Admit { floor_bytes } => {
                self.turns.admit_wait = None;
                // **实得地板回吐**(设计卷 §2.4 报告面):请求页 / 实得页 / 上下文页。
                self.log_floor_grant(turn_id, floor, ctx_tokens, page, page_bytes, floor_bytes);
            }
            AdmissionVerdict::EvictThen { need_bytes, floor_bytes } => {
                // 抬需求到"腾够就能进":同一次段边界搬完(下沉不丢内容)
                let want = need_bytes.div_ceil(self.dev.pool.block_bytes().max(1));
                need.slots = need.slots.max(want);
                self.log_floor_grant(
                    turn_id, floor, ctx_tokens, page, page_bytes, floor_bytes,
                );
            }
            AdmissionVerdict::Queue { reason } => {
                let started = match self.turns.admit_wait {
                    Some((id, t0)) if id == turn_id => t0,
                    _ => {
                        self.turns.admit_wait = Some((turn_id, std::time::Instant::now()));
                        owl_shared::log_info!("store", "准入：turn {turn_id} 排队等待 — {reason}");
                        std::time::Instant::now()
                    }
                };
                let waited = started.elapsed().as_millis() as u64;
                // `>=`:上限 0 = 立刻按拒处理(不许"等 0 毫秒"变成再等一轮)
                if waited >= self.cfg.knobs.admission_max_wait_ms() {
                    self.reject_queued_turn(
                        turn_id,
                        format!("准入拒绝：排队 {waited}ms 仍无法腾位（{reason}）"),
                    )?;
                }
            }
            AdmissionVerdict::Reject { reason } => {
                self.reject_queued_turn(turn_id, format!("准入拒绝：{reason}"))?;
            }
        }
        Ok(())
    }

    /// **拒一个待准入 turn**(§23.4):出队 + 临时会话焚毁 + `Failed` 事件入队。
    ///
    /// 不掉 `ensure_for_len` 的 `Exhausted` —— 拒的文案不带 "超预算" 字样:
    /// 那是"prompt + max_tokens > max_seq"的专用标记(服务面据此归一化成
    /// `context_length_exceeded` 触发客户端自动 compact);池墙拒是另一类,
    /// 客户端该做的是降并发/重试。
    fn reject_queued_turn(&mut self, turn_id: u64, why: String) -> Result<()> {
        let Some(pos) = self.turns.queue.iter().position(|t| t.id == turn_id) else {
            return Ok(()); // 已被别处取走(幂等)
        };
        let t = self.turns.queue.remove(pos).expect("刚定位过");
        if self.turns.admit_wait.map(|(id, _)| id) == Some(turn_id) {
            self.turns.admit_wait = None;
        }
        if t.ephemeral {
            self.bank.burn(t.session_id); // 临时会话：历史不落地，链/格随 turn 死
        }
        owl_shared::log_warn!("store", "{why}（turn {turn_id}）");
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.admit.reject", 1, file!(), line!());
        });
        self.turns.pending_events.push_back(TurnEvent::Failed { turn: turn_id, err: why });
        self.sync_pool_account()?;
        Ok(())
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
                        match owl_store::plan_offload(&mut self.bank.rel.leg, item.chain) {
                            Ok(p) => (p.blocks, p.pieces, p.table),
                            Err(owl_store::StoreError::Busy(_)) => {
                                a.skipped_busy += 1;
                                continue;
                            }
                            Err(e) => return Err(e.into()),
                        }
                    };
                    let face = self.dev.graphs.session.face_mut();
                    let conn = match face
                        .swap(owl_iface::contract::SwapDir::Out, &pieces, Box::new(table))
                        .await
                    {
                        Ok(c) => c,
                        Err(e) => {
                            owl_store::abort_offload(&mut self.bank.rel.leg, &blocks);
                            return Err(owl_iface::contract::ModelError::Msg(format!(
                                "换出 swap 失败(engine 编排):{e}"
                            )));
                        }
                    };
                    let table = owl_store::reclaim_slot_table(conn)?;
                    a.offloaded +=
                        owl_store::commit_offload(&mut self.bank.rel.leg, &blocks, table)?;
                }
                Sink::Drop => match self.bank.rel.leg.drop_cached(item.chain) {
                    Ok(()) => a.dropped += 1,
                    Err(owl_store::StoreError::Busy(_)) => a.skipped_busy += 1,
                    Err(e) => return Err(e.into()),
                },
            }
        }
        // ② 回灌(Verdict.in_:逐条一条命令;H4 的批量形态在 execute_begin)
        let t_stage = std::time::Instant::now();
        for chain in &verdict.in_ {
            let (cs, slots, pieces, table) = {
                match owl_store::plan_onload(&mut self.bank.rel.leg, std::slice::from_ref(chain)) {
                    Ok(p) => (p.chains, p.slots, p.pieces, p.table),
                    Err(e) => return Err(e.into()),
                }
            };
            let face = self.dev.graphs.session.face_mut();
            let conn = match face
                .swap(owl_iface::contract::SwapDir::In, &pieces, Box::new(table))
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    owl_store::abort_onload(&mut self.bank.rel.leg, &cs, &slots);
                    return Err(owl_iface::contract::ModelError::Msg(format!(
                        "回灌 swap 失败(engine 编排):{e}"
                    )));
                }
            };
            let table = owl_store::reclaim_slot_table(conn)?;
            let _ = owl_store::commit_onload(&mut self.bank.rel.leg, &cs, &slots, table)?;
            self.stage.blocks += cs.len();
            self.stage.bytes += cs.len() * self.dev.pool.block_bytes();
            self.stage.from_verdict += cs.len();
            a.onloaded += 1;
        }
        self.stage.nanos += t_stage.elapsed().as_nanos() as u64;
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
        if !self.bank.rel.leg.host_enabled() || need.slots == 0 {
            return Ok(());
        }
        let verdict = owl_store::plan(&self.bank.rel.leg, need, &[]);
        if owl_shared::env_reader::flag("OWL_STORE_HOOK_DEBUG") {
            eprintln!(
                "[hook] {:?} need={} free={}/{} 判 out={} in={} deny={:?}",
                need.why,
                need.slots,
                self.bank.rel.leg.slots_free(),
                self.bank.rel.leg.slots_total(),
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
            // 凑不够 = 真容量不足预警:照搬能搬的(下沉不丢内容),响亮留痕。
            // **不致命**(缺口 3):记为一次 `degrade`(服务面/报告可查),由调用方
            // (准入/调度)决定收缩还是降级;真正的设备级错才走关停。
            owl_shared::log_warn!("store", "钩子:需求凑不够(真容量不足预警):{why}");
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("store.degrade", 1, file!(), line!());
            });
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
        let mut need = need;
        self.admission_gate(&mut need)?;
        self.pool_housekeeping(need).await?;
        // **[N6 Pressure]**(缺口 3):余量告急 ⇒ **零缺口也搬**(先下沉、再驱逐);
        // 这是"压力路径第一步":先把自己能腾的腾出来,而不是等 `ensure_for_len` 撞墙。
        if let Some(k) = self.pressure_probe() {
            let free = owl_shared::vram::free_bytes();
            owl_shared::log_warn!(
                "store",
                "N6 压力节点:驱动余量 {:.0}MiB < 目标 ⇒ 目标腾 {k} 槽(先下沉后驱逐)",
                free as f64 / 1048576.0
            );
            self.pool_housekeeping(owl_store::Need {
                slots: k,
                why: owl_store::NeedWhy::Pressure,
            })
            .await?;
        }
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
            SchedulerOutput::Idle => {
                // 准入拒(S3b §23.4)等 host 侧即时终态:先回吐再报 Idle ——
                // 否则 Idle 会让服务面的泵循环在此断开,被拒 turn 的事件
                // 要等下一次提交才发得出去。
                if let Some(ev) = self.turns.pending_events.pop_front() {
                    Ok(ev)
                } else {
                    Ok(TurnEvent::Idle)
                }
            }
            SchedulerOutput::Step { begin, action } => {
                if let Some(b) = begin {
                    self.execute_begin(b).await?;
                }
                match action {
                    StepAction::Prefill { chunk_ids, base, is_last } => {
                        // 性能面:相位定相 + 包络 span(perf-framework.md §四)。
                        // 引擎 actor = current_thread 运行时 ⇒ 单线程栈配对安全;
                        // 解释器节点钩子按 env.phase 打 tag,两者同源。
                        let saved = self.dev.env.phase;
                        self.set_phase(owl_models::env::EvalPhase::Prefill);
                        owl_shared::timer_start!("engine.phase.prefill");
                        let r = self.execute_prefill(chunk_ids, base, is_last).await;
                        owl_shared::timer_end!("engine.phase.prefill");
                        self.set_phase(saved);
                        r
                    }
                    StepAction::Decode { token, pos, kv_slot, gdn_slot, grew } => {
                        let saved = self.dev.env.phase;
                        self.set_phase(owl_models::env::EvalPhase::Decode);
                        owl_shared::timer_start!("engine.phase.decode");
                        let r = self.execute_decode(token, pos, kv_slot, gdn_slot, grew).await;
                        owl_shared::timer_end!("engine.phase.decode");
                        self.set_phase(saved);
                        r
                    }
                    StepAction::SpecRound { token, pos, kv_slots, gdn_slot, grew } => {
                        let saved = self.dev.env.phase;
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
