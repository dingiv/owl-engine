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

/// **引擎节点**(store 立案 §三.A / §五.2 的"节点即调用点";步 4 收编 2026-10-19)。
///
/// 每个节点 = **一次 `store.plan` 决策 + 一次搬运执行 + 一次记账**(N4/N5 例外:
/// 它们不按"槽需求"决策,而是"归还清单/归属复位")。所有保洁调用都从
/// [`RunningEngine::node`] 走 —— 那正是"散落 ad-hoc 驱逐/下沉归零"的判据面。
#[derive(Clone, Copy, Debug)]
pub(crate) enum Node {
    /// **N1/N2/N3**:准入 / prefill chunk 增长 / decode 步增长(需求算式在 `step_need`)
    Need(owl_store::Need),
    /// **N4 TurnEnd**:池清点(池内块改归 `Cache`)+ 未归还清单(按名 WARN)
    TurnEnd(u64),
    /// **N5 SessionClose**:会话关闭(链已由会话表经 store API 归还;节点只记账/复位)
    SessionClose(u64),
    /// **N6 Pressure**:余量告急 ⇒ 零缺口也搬(先下沉、再驱逐)
    Pressure(usize),
    /// **H5 Idle**:空转整理(无活跃 turn;每 N 拍一次)
    Idle(usize),
}

impl Node {
    /// 打点名(`store.node.*`)。
    pub(crate) fn label(&self) -> &'static str {
        match self {
            Node::Need(need) => match need.why {
                owl_store::NeedWhy::Admit => "admit",
                owl_store::NeedWhy::Grow => "grow",
                owl_store::NeedWhy::Idle => "idle",
                owl_store::NeedWhy::Pressure => "pressure",
            },
            Node::TurnEnd(_) => "turn_end",
            Node::SessionClose(_) => "session_close",
            Node::Pressure(_) => "pressure",
            Node::Idle(_) => "idle",
        }
    }
}


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
    /// H5 空转整理节拍(空转 pump 计数;每 `IDLE_HK_EVERY` 拍整理一次)
    pub(crate) idle_pumps: u64,
    /// **默认池缓存回收的节拍**(§23.9:turn 收尾之后才 free 的页要在空转期再清一次;
    /// 每 `IDLE_TRIM_EVERY` 一次,失败只 WARN 不打断空转)。
    pub(crate) last_trim: Option<std::time::Instant>,
    /// N6/保洁 deny 的**累计**次数(限流日志用:首条 + 每 64 条打一次;
    /// 计数本身**不丢** —— 机读面 = `store.degrade` metrics)
    pub(crate) degrade_seen: u64,
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
    /// **合批 decode 已执行步数**(B5.1 事实位;门用它证明批路径真被走到)
    pub(crate) batch_steps_n: u64,
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
    /// **懒沉保留**的块数(池仍有余量 ⇒ 本轮不沉、整链常驻;2026-10-20)
    pub lazy_keep: usize,
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
            "部分驻留:下沉 {} 块 · 回灌 {} 块 · 不可沉(未入缓存树){} 块 · 懒沉保留 {} 块 · prefill 子集表 {} 次",
            self.sunk, self.staged, self.not_cacheable, self.lazy_keep, self.prefill_subset
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
/// 单调毫秒（`SystemTime`；时钟回拨兜底为 0 —— 只用于排队时长/诊断，不参与语义）
pub(crate) fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// 预算收缩
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

/// **半表行拼接**（P1.1 词表切；纯函数 —— 判据可单测）：
/// `[rows, half] × 2 → [rows, 2·half]`。
///
/// 列序 = **全局词表序**：rank0 持低半、rank1 持高半
/// （`TpPlan::shard_range` 语义 ⇒ `build_shard_pass` 给 rank0 的 span 从 0 起），
/// 故逐行 `a ++ b`。
///
/// 响亮拒：两半长度不等 / 不是半表宽的整倍（形状不符就拼接了
/// = 静默错词，比拒启难查一百倍）。
pub(crate) fn concat_head_rows(
    a: &[f32],
    b: &[f32],
    half: usize,
) -> crate::Result<Vec<f32>> {
    if half == 0 || a.len() != b.len() || !a.len().is_multiple_of(half) {
        return Err(owl_iface::contract::ModelError::EngineInvariant {
            what: format!(
                "词表切拼接：形状不符（半表宽 {half} / 本端 {} / 对端 {}）",
                a.len(),
                b.len()
            ),
        });
    }
    let rows = a.len() / half;
    let mut out = Vec::with_capacity(a.len() * 2);
    for r in 0..rows {
        out.extend_from_slice(&a[r * half..(r + 1) * half]);
        out.extend_from_slice(&b[r * half..(r + 1) * half]);
    }
    Ok(out)
}

/// **候选合并**（P1.1 × P2.1；纯函数 —— 判据可单测）：两 rank 各 `[rows, K]`
/// 值/索引 → **全局 top-K**。
///
/// - rank1（peer）的索引是**半表域** ⇒ 全局 id = 局部 id + `half`（rank0 低半）；
/// - 排序律与核同：**值降序、平局取小 id**（`topk` 族的既有语义）；
/// - 正确性依据：全局 top-K ⊆ 两半各自 top-K 的并集（截断性质）。
///
/// 响亮拒：形状不符（行数/K 不一致 / 长度不是 rows*K）。
pub(crate) fn merge_topk_two_halves(
    vals_a: &[f32],
    ids_a: &[f32],
    vals_b: &[f32],
    ids_b: &[f32],
    rows: usize,
    k: usize,
    half: usize,
) -> crate::Result<(Vec<f32>, Vec<u32>)> {
    let want = rows * k;
    if k == 0 || vals_a.len() != want || vals_b.len() != want || ids_a.len() != want || ids_b.len() != want {
        return Err(owl_iface::contract::ModelError::EngineInvariant {
            what: format!(
                "候选合并：形状不符（rows {rows} k {k} / a {}/{} b {}/{}）",
                vals_a.len(),
                ids_a.len(),
                vals_b.len(),
                ids_b.len()
            ),
        });
    }
    let mut out_v = Vec::with_capacity(want);
    let mut out_i = Vec::with_capacity(want);
    let mut cand: Vec<(f32, u32)> = Vec::with_capacity(2 * k);
    for r in 0..rows {
        cand.clear();
        for j in 0..k {
            let ia = ids_a[r * k + j];
            if ia.is_finite() {
                cand.push((vals_a[r * k + j], ia as u32));
            }
            let ib = ids_b[r * k + j];
            if ib.is_finite() {
                cand.push((vals_b[r * k + j], ib as u32 + half as u32));
            }
        }
        // 值降序；平局取小 id（与核的 topk 语义同律）
        cand.sort_by(|x, y| {
            y.0.partial_cmp(&x.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(x.1.cmp(&y.1))
        });
        for &(v, i) in cand.iter().take(k) {
            out_v.push(v);
            out_i.push(i);
        }
    }
    Ok((out_v, out_i))
}

impl<D: DeviceClient> RunningEngine<D> {
    /// 本档 prefill 块长（探针排版用；与调度器取**同一份配置** ⇒ 不会对不上）
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

    /// 合批 decode 已执行的步数(B5.1 事实位;门用 —— 0 = 批路径没被走到)
    pub fn batch_steps(&self) -> u64 {
        self.batch_steps_n
    }

    /// 合批图是否在场(B5.1 M1b 事实位;false = 执行器走 eager 臂)
    /// 合批上限(`[runtime] max_batch`;1 = 串行现状)。服务面诊断/门用。
    pub fn max_batch(&self) -> usize {
        self.cfg.knobs.max_batch
    }

    pub fn has_batch_graph(&self) -> bool {
        self.dev.graphs.batch.is_some()
    }

    /// **诊断出口**：读主 rank（rank0）会话图的命名输出（如 `logits`/`token`）。
    /// `None` = 该图无此输出槽。G8 对拍/数值归因用（S4 ⑤-K3）。
    /// ⚠️ **P1.1 词表切档下 `logits` 是半表**（本 rank 宽度 = `logits_vocab()`）；
    /// 要全表请用 [`Self::read_full_logits`]（唯一的“半表→全表”出口）。
    pub async fn read_output_f32(&mut self, name: &str) -> Result<Option<Vec<f32>>> {
        if !self.dev.graphs.session.has_output(name) {
            return Ok(None);
        }
        Ok(Some(self.dev.graphs.session.read_output_f32(name).await?))
    }

    /// **P1.1 词表切：logits 的“全表”读出口**（采样/诊断/接受判据的**唯一**入口）。
    ///
    /// - 未切档（`head_split = false`）：单次读 —— 与旧
    ///   `session.read_output_f32("logits")` **逐字节等价**（零开销）；
    /// - 词表切档：本 rank 半表 + peer 半表在 **host 拼接**（两 rank 同一进程
    ///   ⇒ 无新传输设施；列序 = 全局词表序：rank0 低半 + rank1 高半）。
    ///
    /// 形状 = `[rows, 2·half]`（`rows` 由半表长度推出：decode = 1，verify = depth+1）。
    /// **词表切实效判据**（P1.1）：本 rank 的 logits 宽 < 全量词表。
    /// **不读意图位**（`spec.head_split`）—— 意图可能被模型结构否定（tied 表
    /// 装载期已响亮拒；但“事实在模型里”才是拼接面的唯一判据）。
    pub(crate) fn head_split_active(&self) -> bool {
        self.dev.model.logits_vocab() != self.dev.model.vocab_size()
    }

    pub(crate) async fn read_full_logits(&mut self, name: &str) -> Result<Vec<f32>> {
        let a = self.dev.graphs.session.read_output_f32(name).await?;
        if !self.head_split_active() {
            return Ok(a);
        }
        let half = self.dev.model.logits_vocab();
        let p = self
            .peer
            .as_mut()
            .ok_or_else(|| ModelError::EngineInvariant {
                what: "词表切档缺 peer：半表拼接无对端（装配不一致）".into(),
            })?;
        let b = p.graphs.session.read_output_f32(name).await?;
        concat_head_rows(&a, &b, half)
    }

    /// **诊断出口**：读 peer（rank1）会话图输出（TP 对拍；**单卡 = None**）。
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
        // **在飞总数上限**（P0-2）：批位 + 队列 都有界。
        // 为什么放在 submit 而不是 `schedule`：槽满时判决根本不跑（
        // `active.len() < max_batch` 的闸先关），队列会无界膨胀；submit 是
        // 唯一能看见"活跃 + 排队"总数的地方。0 = 无界（旧行为）。
        let max_queue = self.cfg.knobs.max_queue;
        if max_queue > 0 {
            let in_flight = self.turns.active.len() + self.turns.queue.len();
            let cap = self.cfg.knobs.max_batch.max(1).saturating_add(max_queue);
            if in_flight >= cap {
                // 退避建议（P3-c）：EWMA × 需要的波数（含自己）
                let eta = self
                    .turns
                    .eta_ms(self.cfg.knobs.max_batch.max(1), self.turns.queue.len());
                self.turns.note_reject(session.unwrap_or(0), "no_slot");
                return Err(ModelError::Msg(format!(
                    "admission:no_slot retry_ms={eta} —— 在飞 {in_flight} 已达上限 {cap}\
                     （批位 {} + 队列 {max_queue}）",
                    self.cfg.knobs.max_batch.max(1)
                )));
            }
        }
        // **同会话并发守卫**(B5.1):合批档下两个同会话 turn 会共享同一份
        // KV/GDN 状态 ⇒ 结构性互踩。亮拒(上层应按会话串行)。
        if self.cfg.knobs.max_batch > 1 {
            if let Some(id) = session {
                if self.turns.active.iter().any(|a| a.session_id == id)
                    || self.turns.queue.iter().any(|t| t.session_id == id)
                {
                    return Err(ModelError::Msg(format!(
                        "submit: 会话 {id} 已有活跃/排队 turn —— 合批档不支持同会话并发"
                    )));
                }
            }
        }
        let prompt = prompt.into();
        // **VLM 输入门**(2026-10-20 视觉接线 V5):单图、非合批;图像 × 投机
        // 已接(verify 的 rope pos 加 m-RoPE delta,slots/lens 仍用真实位置)。
        // 图像路径的塔/表/prefill 见 execute_prefill;这里只做提交边界拒。
        if !opts.images.is_empty() {
            if self.dev.vlm.is_none() {
                return Err(ModelError::Msg(
                    "submit: 请求带图,但本档 [model].vision != true(引擎无视觉塔)".into(),
                ));
            }
            if self.cfg.knobs.max_batch > 1 {
                return Err(ModelError::Msg(
                    "submit: VLM 图像首版不支持合批(max_batch > 1)".into(),
                ));
            }

            if opts.images.len() != 1 {
                return Err(ModelError::Msg(format!(
                    "submit: 图像首版只支持单图(来 {} 张)",
                    opts.images.len()
                )));
            }
        }
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
        // **准入封顶**（`sched_policy::cap_for`）：预留 = prompt + min(max_new, cap)。
        // 不夹的话，turn 会生成超出预留 ⇒ 撞墙/重灌（与准入账打架）。
        let max_new = {
            let l = self.admission_limits();
            let s = self.store_facts();
            let facts = crate::sched_policy::TurnFacts {
                turn: 0, // 尚未发号；`cap_for` 只用到 prompt/max_new/oversub
                session: session_id,
                prompt_tokens: prompt_ids.len(),
                max_new,
                fed: 0,
                out_len: 0,
                phase: crate::sched_policy::PhaseSz::Prefill,
                spec: false,
            };
            let cap = self.turns.admission.cap_for(&l, &s, &facts);
            if cap < max_new {
                owl_shared::log_warn!(
                    "admit",
                    "生成预算夹紧：max_new {max_new} → {cap}（会话 {session_id:#x}；                     在位者优先 + 给新人留头寸；客户端会看到 finish_reason=length）"
                );
                cap.max(1)
            } else {
                max_new
            }
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
            submitted_ms: now_ms(),
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

    /// 单 turn 的**实际生成预算**（submit 夹紧后的值；`Accepted` 如实带出）。
    /// 排队/活跃两侧都查；不存在 = `None`。
    pub fn turn_max_new(&self, turn: u64) -> Option<usize> {
        self.turns
            .queue
            .iter()
            .find(|t| t.id == turn)
            .map(|t| t.spec.max_new)
            .or_else(|| self.turns.active.iter().find(|a| a.id == turn).map(|a| a.max_new))
    }

    /// 拒绝分桶快照（会话, 码, 次数；按次数降序，最多 `limit` 条）
    pub fn reject_snapshot(&self, limit: usize) -> Vec<(u64, String, u64)> {
        let mut v: Vec<(u64, String, u64)> = self
            .turns
            .reject_stats
            .iter()
            .map(|((s, c), n)| (*s, c.clone(), *n))
            .collect();
        v.sort_by(|a, b| b.2.cmp(&a.2));
        v.truncate(limit);
        v
    }

    /// 排队时长直方图（`<50ms / <200ms / <1s / <5s / ≥5s`）
    pub fn wait_hist(&self) -> [u64; 5] {
        self.turns.wait_hist
    }

    /// 排队快照（`/debug/vram → batch.queued`；最多 16 条，按等待降序）
    pub fn queued_snapshot(&self, limit: usize) -> Vec<(u64, u64, u64)> {
        let now = now_ms();
        let mut v: Vec<(u64, u64, u64)> = self
            .turns
            .queue
            .iter()
            .map(|t| (t.id, t.session_id, now.saturating_sub(t.submitted_ms)))
            .collect();
        v.sort_by(|a, b| b.2.cmp(&a.2));
        v.truncate(limit);
        v
    }

    /// 队列长度(待跑 turn 数;服务面诊断/准入门读)。
    pub fn queue_len(&self) -> usize {
        self.turns.queue.len()
    }

    /// 活跃 turn 数(B5.1 多活跃;服务面灌流测量读 —— 并发是否真进引擎)。
    pub fn active_len(&self) -> usize {
        self.turns.active.len()
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
        // B5.1:多活跃时给“本步执行 turn”;批动作(未选)= 活跃首
        self.turns
            .executing
            .or_else(|| self.turns.active.first().map(|a| a.id))
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

    /// **H5 空转整理的触发节拍**(每 N 次空转 pump 整理一次)。
    ///
    /// 常量而非 env:引擎零 env 读取纪律(env 是隐式依赖);将来若要可调,
    /// 走 `EngineKnobs` 显式配置字段。
    const IDLE_HK_EVERY: u64 = 16;

    /// 空转期池缓存回收的最小间隔(空转 pump 很密;500ms 足够让 turn 收尾的
    /// 大额临时块真正 free 之后再清一次)。
    const IDLE_TRIM_EVERY: std::time::Duration = std::time::Duration::from_millis(500);

    /// **统一节点入口**(store 立案 §三.A / §五.2;步 4 收编)。
    ///
    /// 纪律:节点只做三件事 —— ①按节点算需求 ②`store.plan` 取计划 ③`apply_verdict`
    /// 执行 + 记账。**决策唯一产地 = store**;本函数只负责"何时问、搬、记"。
    /// **判据(grep 可证)**:engine 内 `store::plan(` / `apply_verdict(` 只应出现在
    /// `pool_housekeeping`(本入口的唯一执行臂)里 —— 散落的 ad-hoc 驱逐/下沉归零。
    pub(crate) async fn node(&mut self, n: Node) -> Result<()> {
        let name = n.label();
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add(&format!("store.node.{name}"), 1, file!(), line!());
        });
        match n {
            // N1/N2/N3:准入/增长(需求算式在 `step_need`,唯一边界由它给)
            Node::Need(need) => self.pool_housekeeping(need).await,
            // N6:余量告急 ⇒ 零缺口也搬
            Node::Pressure(k) => {
                self.pool_housekeeping(owl_store::Need::kv(k, owl_store::NeedWhy::Pressure))
                    .await
            }
            // H5:空转整理(无活跃 turn;冷面下沉/清池,不碰活跃面)
            // 状态腿目标 = **一条快照的字节**(27B ≈162MB):空转期慢慢收,
            // 每 N 拍整理一条;KV 侧照旧是 1/8 池(两条腿目标各自独立)。
            Node::Idle(k) => {
                // 状态腿目标 = **一条快照**(宿主**有位**才要:空转整理不是为了丢内容;
                // 宿主分片小/关时给 0 ⇒ 不产生永远凑不够的 deny)
                let snap = if self.bank.rel.snap_host_free_slots() > 0 {
                    self.bank.rel.snap_bytes()
                } else {
                    0
                };
                self.pool_housekeeping(owl_store::Need {
                    slots: k,
                    snap_bytes: snap,
                    why: owl_store::NeedWhy::Idle,
                })
                .await
            }
            // N4:turn 收尾 —— 池清点(改归属 Cache)+ 未归还清单(响亮)
            Node::TurnEnd(turn) => {
                self.n4_pool_sweep(turn);
                self.n4_report_unreturned(turn);
                // **兜底释放**(2026-10-20;§78 P1):清点只报告、不处置 ⇒ 有调用点
                // 漏 free 时每 turn 掉一截显存(4k prefill ≈ 1.6 GiB),长 ctx 撞
                // `vram-hard-budget`。这里把 `Turn(t)` 名下"仍开着、不在池里"的块
                // **强制归还驱动**(决策在 store `stale_of`,执行在句柄)。
                self.n4_force_release(turn).await?;
                // **池缓存回收**(2026-10-20 §23.9):`RELEASE_THRESHOLD=∞` ⇒ turn 内
                // 的峰值缓存**永不**还 OS(peer 侧实测 3.3 GiB 长期挂着,长 prefill
                // 因此提前撞 `vram-hard-budget`)。turn 收尾清一次:块账不变,
                // 只把"已 free 的缓存页"还给驱动。
                let freed = self.trim_pools().await;
                // 让**紧随其后的 Idle 出口**再清一次:turn 收尾路径里还有一批
                // 大额临时块是在 TurnEnd 之后才 free 的(实测 peer 3.0 GiB)。
                self.last_trim = None;
                for (rank, r) in freed {
                    match r {
                        Ok(0) => {}
                        Ok(n) => owl_shared::log_info!(
                            "store",
                            "池缓存回收:turn {turn} rank{rank} 释放 {:.1} MiB(已 free 的池页还给驱动)",
                            n as f64 / 1048576.0
                        ),
                        // 收尾路径的可失败调用:如实 WARN(不静默;也不升级为 turn 失败 ——
                        // 显存多占一点不影响正确性,下一 turn 的压力旁路还会再试)
                        Err(e) => owl_shared::log_warn!(
                            "store",
                            "池缓存回收失败(turn {turn} rank{rank}): {e}"
                        ),
                    }
                }
                Ok(())
            }
            // N5:会话关闭 —— 链已由会话表归还(store API),本节点只记账/归属复位
            Node::SessionClose(_s) => Ok(()),
        }
    }

    /// **压力缓解口**(缺口 3;actor 在 `PoolExhausted` 后调;2026-10-19)。
    ///
    /// 次序:**先还缓存**(池排空 —— 真还驱动),**再腾结构**(N6 节点:先下沉后驱逐)。
    /// 返回 `(池排空块数, 目标腾出槽数)` —— **如实报数**(0 就是 0:
    /// 实测"mid-turn 池为空"是常态:失败那一刻的块都还在**使用中**,不在池里);
    /// 失败响亮上抛(调用方只记日志继续)。
    pub async fn relieve_pressure(&mut self) -> Result<(usize, usize)> {
        let freed = {
            let f = self.dev.graphs.dev_ctx_face_mut();
            f.drain_temp_pool().await
        };
        if freed > 0 {
            owl_shared::log_warn!(
                "store",
                "压力缓解:复用池排空 {freed} 块后归还驱动({:.0}MiB free)",
                owl_shared::vram::free_bytes() as f64 / 1048576.0
            );
        }
        let k = self.pressure_probe().unwrap_or(0);
        if k > 0 {
            self.node(Node::Pressure(k)).await?;
        }
        Ok((freed, k))
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

    /// **H5 空转制的节拍推进**(Idle pump 计数;每 N 拍调 `node(Node::Idle)` 一次)
    fn idle_tick(&mut self) -> Option<usize> {
        self.idle_pumps = self.idle_pumps.wrapping_add(1);
        if !self.idle_pumps.is_multiple_of(Self::IDLE_HK_EVERY) {
            return None;
        }
        let total = self.bank.rel.leg.slots_total();
        if total == 0 {
            return None;
        }
        // 目标 = 总量的 1/8(至少 1 槽、至多 16):空转整理是"慢慢收",不是搬家
        Some((total / 8).clamp(1, 16))
    }

    /// **句柄计数三元组**(诊断面;`/debug/vram`):`(hook_fail, window_unclosed, 窗内块数)`
    ///
    /// 判据:三者非零 = 有事件没入账 / 有窗未收 —— **可见**比"账看起来平"更重要。
    pub fn handle_probe(&self) -> (u64, u64, usize) {
        self.dev.graphs.dev_ctx_face().ledger_probe()
    }

    /// **默认池账(逐 rank;2026-10-20 §23.9)**:`/debug/vram` 的 `pools` 行。
    ///
    /// 为什么要它:`RELEASE_THRESHOLD=∞` 下**已 free 的块仍占物理显存**(池缓存);
    /// 块账只见 live ⇒ "物理 − live" 这部分在双 rank 上只长在 peer 一侧
    /// (实测 +3.3 GiB),不读数就只能瞎猜。
    pub async fn pool_probe(
        &mut self,
    ) -> Vec<(u8, std::result::Result<(u64, u64, u64, u64), String>)> {
        let mut out: Vec<(u8, std::result::Result<(u64, u64, u64, u64), String>)> = Vec::new();
        let r = self
            .dev
            .graphs
            .session
            .face_mut()
            .mem_pool_stats()
            .await
            .map_err(|e| e.to_string());
        out.push((0, r));
        if let Some(p) = self.peer.as_mut() {
            let r = p
                .graphs
                .session
                .face_mut()
                .mem_pool_stats()
                .await
                .map_err(|e| e.to_string());
            out.push((1, r));
        }
        out
    }

    /// **默认池缓存回收(逐 rank)** —— turn 收尾 / 压力旁路调。
    ///
    /// 语义:`cuMemPoolTrimTo(0)` 把池内**已 free 的缓存页**还给 OS(不改块账)。
    /// 返回逐 rank 的 `(rank, Ok(释放字节)/Err(原因))` —— 失败**如实带出**
    /// (调用方决定是否响亮;本函数不吞)。
    pub async fn trim_pools(
        &mut self,
    ) -> Vec<(u8, std::result::Result<u64, String>)> {
        let mut out: Vec<(u8, std::result::Result<u64, String>)> = Vec::new();
        let r = self
            .dev
            .graphs
            .session
            .face_mut()
            .mem_pool_trim()
            .await
            .map_err(|e| e.to_string());
        out.push((0, r));
        if let Some(p) = self.peer.as_mut() {
            let r = p
                .graphs
                .session
                .face_mut()
                .mem_pool_trim()
                .await
                .map_err(|e| e.to_string());
            out.push((1, r));
        }
        out
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
        // ① 待准入(队列 front)**且无活跃**:准入即要覆盖整个 prompt。
        // (有活跃时它只是排队 —— 准入需求留给 admission_gate,不许把
        //  “别人的 prompt”算成当前批的保洁缺口。)
        if self.turns.active.is_empty() {
            if let Some(t) = self.turns.queue.front() {
                let (sid, plen) = (t.session_id, t.prompt_ids.len());
                let dev = {
                    let (hit, recall) = self.bank.rel.match_prefix(&t.prompt_ids)?;
                    Self::reuse_face(&self.bank.rel, &hit, &recall, plen, page).1
                };
                let have = self.bank.sessions.get(sid).map(|s| s.block_table.len()).unwrap_or(0);
                return Ok(owl_store::Need::kv(
                    slots_for(plen).saturating_sub(dev).saturating_sub(have),
                    NeedWhy::Admit,
                ));
            }
            return Ok(owl_store::Need::kv(0, NeedWhy::Idle));
        }
        // ② 活跃集:**逐 turn 求和**(B5.1 —— 批内每行都会长块链;只算
        // 首 turn 会少腾槽 ⇒ 批步中途 Exhausted 把整批打掉)
        let mut need = 0usize;
        for act in self.turns.actives() {
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
            need += slots_for(target).saturating_sub(have);
        }
        Ok(owl_store::Need::kv(need, NeedWhy::Grow))
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
        // **P2-1**:池指标入 metrics(时序可见;每 turn 一次,量级 = 计数快照)
        {
            let (n, bytes) = b.pool_stats();
            let (hits, misses, drained) = b.pool_counters();
            let evicted = b.pool_evicted();
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("store.pool.entries", n as u64, file!(), line!());
                m.counter_add("store.pool.bytes", bytes as u64, file!(), line!());
                m.counter_add("store.pool.hits", hits, file!(), line!());
                m.counter_add("store.pool.misses", misses, file!(), line!());
                m.counter_add("store.pool.evicted", evicted, file!(), line!());
                m.counter_add("store.pool.drained", drained, file!(), line!());
            });
        }
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
        // **池内块不算"未归还"**(2026-10-20):`pool_release` 入池时 loan 仍开着
        // (`pooled=true`)—— 它已可零命令复用,是**缓存**不是泄漏。旧报告把两者
        // 混在一起 ⇒ 每 turn 都喊几百 MiB,真泄漏反而被噪声淹没。
        let (pooled, open): (Vec<_>, Vec<_>) = loans.iter().partition(|(_, l)| l.pooled);
        let agg_of = |xs: &[&(u64, owl_store::Loan)]| {
            let mut agg: std::collections::BTreeMap<String, (usize, usize)> =
                std::collections::BTreeMap::new();
            for (_, l) in xs {
                let e = agg.entry(l.name.clone()).or_insert((0, 0));
                e.0 += 1;
                e.1 += l.bytes;
            }
            agg
        };
        if open.is_empty() {
            let pool_kib: usize = pooled.iter().map(|(_, l)| l.bytes).sum::<usize>() / 1024;
            owl_shared::log_debug!(
                "store",
                "N4 池内复用:turn {turn} 名下 {} 块 / {pool_kib} KiB 仍在复用池(非泄漏)",
                pooled.len()
            );
            return;
        }
        let agg = agg_of(&open);
        let total_kib: usize = agg.values().map(|(_, b)| *b).sum::<usize>() / 1024;
        let detail: Vec<String> = agg
            .iter()
            .map(|(n, (c, b))| format!("{n} ×{c} / {}B", b))
            .collect();
        let pool_kib: usize = pooled.iter().map(|(_, l)| l.bytes).sum::<usize>() / 1024;
        owl_shared::log_warn!(
            "store",
            "N4 未归还清单:turn {turn} 名下仍有 {} 块 / {total_kib} KiB 开着(另有池内复用 {} 块 / {pool_kib} KiB)—— {}",
            open.len(),
            pooled.len(),
            detail.join(" ｜ ")
        );
    }

    /// **N4 兜底释放**(2026-10-20;§78 P1 的收口)。
    ///
    /// **为什么需要**:`n4_report_unreturned` 只**报告**不处置。实测 27B 4k prefill
    /// 每 turn 有 ~1000 块挂在 `Turn(t)` 名下从不归还(名单 = 整层前向中间量)
    /// ⇒ 第 2 个长 turn 就撞 `vram-hard-budget`(要 2 KiB 也发不出)。
    ///
    /// **分工**(与 N6 池排空同律):决策+结清在 store(`drain_owner`:删借出登记 +
    /// 出三层账 + 摘池索引),执行在这里(`free_settled`:只发驱动 free,不补事件)。
    ///
    /// **纪律**:
    /// - 释放前 WARN 已由 `n4_report_unreturned` 打过(**名字/字节** = 定位材料),
    ///   这里补一条 INFO 的"已收口"账(不许因为兜住了就静默);
    /// - 驱动 free 失败 = **上抛**(收尾路径会响亮留痕;账已在 store 结清,记得
    ///   显存已经回不来了 —— 这是硬错,不是可重试的软错);
    /// - 只碰 `Turn(t)` 归属 —— 图/池/会话各有归属,不会被误伤。
    async fn n4_force_release(&mut self, turn: u64) -> Result<()> {
        let (n, bytes) = self.drain_turn_stale(turn, "turn_end").await?;
        if n == 0 {
            return Ok(());
        }
        owl_shared::log_info!(
            "store",
            "N4 兜底释放:turn {turn} {n} 块 / {} KiB 结清并归还驱动(逐 rank 归各自面)",
            bytes / 1024
        );
        Ok(())
    }

    /// **提前释放(段边界)**:把 `Turn(t)` 名下**仍开着**的块逐 rank 结清。
    ///
    /// 为什么要有"提前"这一档(2026-10-20 用户裁决):turn 收尾才收口 = 长 prefill
    /// 的残块在**整个 turn 内**累积(27B TP 实测 peer 每 chunk ≈94 MB,128k 就把
    /// 12 GB 余量吃光)。客户端 `DeviceHandle` 是**唯一咽喉** —— 每次分配都进了
    /// store 账(带 rank/owner),所以"该 chunk 的中间量已经死了"这件事客户端全知:
    /// 在**下一个 chunk 起点**把上一 chunk 遗留的 `Turn(t)` 块结清即可。
    ///
    /// 判据(不动长命面):只碰 `Owner::Turn(t)` —— 权重/池/图/会话各有归属,
    /// 且调用点保证此刻上一 chunk 的前向已全部入队(同流 free 保序)。
    ///
    /// 返回 `(块数, 字节)`。
    pub(crate) async fn drain_turn_stale(
        &mut self,
        turn: u64,
        tag: &'static str,
    ) -> Result<(usize, usize)> {
        let owner = owl_iface::handle::Owner::Turn(turn);
        // **例外集:本 turn 内跨相位存活的块**。
        //
        // 今天只有一件:VLM 塔输出 `merged`(首 chunk 评出,逐 chunk 当 scatter
        // 的输入复用)。它同样是 `Owner::Turn(t)`,但**不是**"上一 chunk 的残块"——
        // 清掉它 = 下一 chunk 的 `image_scatter` 拿到 `DeadBlock`
        // (2026-10-21 VLM 单图基线实测:**首个 chunk 后**塔输出即被此处 free)。
        // 将来若再出现"跨 chunk 长命"的 turn 块,一律加到这里(单点登记)。
        let keep: Vec<u64> = self
            .turns
            .active()
            .ok()
            .map(|a| {
                [a.vlm.as_ref(), a.vlm_peer.as_ref()]
                    .into_iter()
                    .flatten()
                    .filter_map(|v| v.merged.as_ref().map(|(b, _, _)| b.id))
                    .collect()
            })
            .unwrap_or_default();
        // **逐 rank 分账**:两 rank 共用一本账,但块在**各自的 server** 块表里 ——
        // 把 rank1 的 id 交给 rank0 的 server = 找不到 ⇒ 静默漏(实测 434/1010)。
        let mut total = (0usize, 0usize);
        for rank in 0..2u8 {
            let (ids, bytes) = {
                let cell = self.bank.rel.ledger_cell();
                let mut b = owl_store::book_lock(&cell);
                b.drain_owner_except(owner, Some(rank), &keep)
            };
            if ids.is_empty() {
                continue;
            }
            let face = if rank == 0 {
                Some(self.dev.graphs.session.face_mut())
            } else {
                self.peer.as_mut().map(|p| p.graphs.session.face_mut())
            };
            let Some(face) = face else {
                // 单卡档不该有 rank1 的账;有 = 账目串了 ⇒ 响亮(不静默丢)
                owl_shared::log_error!(
                    "store",
                    "N4 兜底释放:rank{rank} 有 {} 块待还,但本引擎**无 rank1 束** —— 账目串了",
                    ids.len()
                );
                return Err(ModelError::Msg(format!(
                    "N4 兜底释放:rank{rank} 无对应设备面(ids={})",
                    ids.len()
                )));
            };
            let n = ids.len();
            face.free_settled(&ids).await?;
            total.0 += n;
            total.1 += bytes;
        }
        if total.0 == 0 {
            return Ok((0, 0));
        }
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add(&format!("store.{tag}.stale_blocks"), total.0 as u64, file!(), line!());
            m.counter_add(&format!("store.{tag}.stale_bytes"), total.1 as u64, file!(), line!());
        });
        Ok(total)
    }

    /// **store 报告面**(设计卷 §2.4 / §13.4 对账口):三层 + 两腿 +
    /// 保留/弹性/可动面。服务面 `/debug` 与准入门读这里(单一真源)。
    pub fn store_report(&self) -> owl_store::StoreReport {
        self.bank.rel.report()
    }

    /// **状态腿淘汰序预览**(诊断面;与容量调度同一份判据)。
    pub fn snap_evict_preview(&self, limit: usize) -> Vec<owl_store::SnapEvictPreview> {
        self.bank.rel.snap_evict_preview(limit)
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
        // B5.1:多活跃 ⇒ 逐 turn 记保留额(各自会话块链实占)
        let active_holds: Vec<(u64, usize)> = self
            .turns
            .actives()
            .map(|a| {
                let held = self
                    .bank
                    .sessions
                    .get(a.session_id)
                    .map_or(0, |s| s.block_table.len())
                    * block_bytes;
                (a.id, held)
            })
            .collect();
        let held = active_holds
            .iter()
            .map(|(_, h)| *h)
            .sum::<usize>()
            .min(occupied.saturating_sub(movable));
        let ad = &mut self.bank.rel.admission;
        ad.set_cache_bytes(movable); // 可动面(可下沉/可驱逐)
        ad.set_session_held(occupied.saturating_sub(movable).saturating_sub(held)); // 会话持有面
        for (id, h) in &active_holds {
            ad.set_floor(*id, *h)?; // 活跃 turn 保留额(G2b)
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
        if self.turns.active.len() >= self.cfg.knobs.max_batch.max(1) {
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




    // ─────────────────────── 磁盘层执行面(S6;store 出计划 / 这里做 I/O) ───────────────────────

    /// **KV 溢写**(宿主 → 磁盘;内容移动)。返回 `true` = 真搬了。
    ///
    /// - 不在宿主(已在盘/在设备)= 幂等跳过;
    /// - 盘满 ⇒ 先按 store 的驱逐计划腾位(删文件 + 出账),再试一次;
    /// - 写盘失败 ⇒ **内容放回宿主**(不丢) + 响亮打点 ⇒ `Ok(false)`(本单跳过该条)。
    async fn spill_kv_to_disk(&mut self, chain: owl_store::ChainHash) -> Result<bool> {
        if self.bank.rel.disk.is_none() {
            return Ok(false);
        }
        if self.bank.rel.disk.as_ref().is_some_and(|d| d.contains(chain)) {
            return Ok(false);
        }
        let buf = match self.bank.rel.leg.host_take(chain) {
            Ok(b) => b,
            Err(owl_store::StoreError::Unknown(_)) => return Ok(false), // 不在宿主
            Err(e) => return Err(e.into()),
        };
        let plan = match self.disk_spill_plan_kv(chain).await? {
            Some(p) => p,
            None => {
                // 盘装不下(腾完也不够)⇒ 内容还回宿主:本单该条不出票
                self.bank.rel.leg.host_put(chain, buf)?;
                return Ok(false);
            }
        };
        match crate::disk_io::write_block(plan.path.clone(), plan.tmp.clone(), buf.into_bytes()).await
        {
            Ok(_) => {
                self.bank.rel.disk.as_mut().expect("上面判过").commit_spill(&plan)?;
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.spill", 1, file!(), line!());
                });
                Ok(true)
            }
            Err(f) => {
                // 内容所有权被带回 ⇒ 放回宿主(溢写失败 ≠ 丢内容)
                self.bank.rel
                    .leg
                    .host_put(chain, owl_store::HostBlock::from_bytes(f.bytes))?;
                owl_shared::log_warn!("store", "KV 溢写失败(内容已回宿主):{}", f.err);
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.spill_fail", 1, file!(), line!());
                });
                Ok(false)
            }
        }
    }

    /// 取 KV 溢写计划(盘满则先腾位重试一次;`None` = 腾不出)。
    async fn disk_spill_plan_kv(
        &mut self,
        chain: owl_store::ChainHash,
    ) -> Result<Option<owl_store::SpillPlanOf<owl_store::ChainHash>>> {
        for attempt in 0..2 {
            let r = self.bank.rel.disk.as_ref().expect("调用方判过").plan_spill(chain);
            match r {
                Ok(p) => return Ok(Some(p)),
                Err(owl_store::StoreError::Exhausted(_)) if attempt == 0 => {
                    let need = self.bank.rel.disk.as_ref().expect("判过").block_bytes();
                    let evp = self.bank.rel.disk.as_ref().expect("判过").plan_evict(need, &[]);
                    if evp.victims.is_empty() {
                        return Ok(None);
                    }
                    let paths: Vec<std::path::PathBuf> =
                        evp.victims.iter().map(|(_, p)| p.clone()).collect();
                    let keys: Vec<owl_store::ChainHash> =
                        evp.victims.iter().map(|(k, _)| *k).collect();
                    let n = crate::disk_io::remove_files(paths).await;
                    let freed = self.bank.rel.disk.as_mut().expect("判过").commit_evict(&keys);
                    owl_shared::log_info!(
                        "store",
                        "磁盘腾位:删 {n} 文件 / 出账 {freed}B(盘按 LRU 逐最老)"
                    );
                    owl_shared::metrics::with_metrics_store(|m| {
                        m.counter_add("store.disk.evict", freed as u64, file!(), line!());
                    });
                }
                Err(owl_store::StoreError::Exhausted(m)) => {
                    owl_shared::log_warn!("store", "磁盘装不下:{m}");
                    return Ok(None);
                }
                Err(e) => return Err(e.into()),
            }
        }
        Ok(None)
    }

    /// **写盘上链索引**(S6.1;turn 收口一次,同 tail 幂等)。
    ///
    /// 内容可能**还没溢写到盘** —— 索引先写;`boot` 以"内容文件在场"截断到连续前缀
    /// (没有内容的索引 = 孤儿,boot 会清掉内容/跳过)。8k token 的链 ≈ 32 KB。
    pub(crate) async fn write_chain_index(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() || !self.bank.rel.disk_enabled() {
            return Ok(());
        }
        let page = self.dev.pool.page.max(1);
        let full = tokens.len() / page;
        if full == 0 {
            return Ok(());
        }
        let Some(tail) = self.bank.rel.leg.tail_hash(tokens, full) else {
            return Ok(());
        };
        let Some(root) = self.bank.rel.disk.as_ref().map(|d| d.root().to_path_buf()) else {
            return Ok(());
        };
        let stem = owl_store::DiskKey::stem(tail);
        let path = root.join(format!("{stem}.idx"));
        if crate::disk_io::exists(&path) {
            return Ok(()); // 同链已写过(幂等;链尾相同 = 前缀相同)
        }
        let tmp = root.join(format!("{stem}.idx.tmp"));
        let idx = match owl_store::ChainIndex::new(page, tokens.to_vec()) {
            Ok(i) => i,
            Err(e) => {
                owl_shared::log_warn!("store", "链索引构造失败(tail={tail:?}):{e}");
                return Ok(());
            }
        };
        match crate::disk_io::write_block(path, tmp, idx.encode()).await {
            Ok(_) => {
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.index.write", 1, file!(), line!());
                });
            }
            Err(f) => {
                owl_shared::log_warn!("store", "链索引写盘失败:{}(索引缺失 = 该链跨重启不可复用)", f.err);
            }
        }
        Ok(())
    }

    /// **回灌前把盘上的块读回宿主**(S6;两个回灌口共用)。
    ///
    /// 为什么必须先过宿主:swap 命令的语义是"设备 ↔ 宿主槽台";
    /// 盘上的块要先落回宿主缓冲,才能作为槽台交给 `plan_onload`。
    /// 返回真读回的块数(诊断/打点)。
    pub(crate) async fn ensure_host_resident(
        &mut self,
        chains: &[owl_store::ChainHash],
    ) -> Result<usize> {
        let mut n = 0usize;
        for c in chains {
            if self.fill_kv_from_disk(*c).await? {
                n += 1;
            }
        }
        Ok(n)
    }

    /// **KV 回填**(磁盘 → 宿主;S6)。返回 `true` = 真读了盘。
    async fn fill_kv_from_disk(&mut self, chain: owl_store::ChainHash) -> Result<bool> {
        let Some(d) = self.bank.rel.disk.as_ref() else { return Ok(false) };
        if !d.contains(chain) {
            return Ok(false);
        }
        if self.bank.rel.leg.host_contains(chain) {
            let _ = self.bank.rel.disk.as_mut().expect("判过").touch(chain);
            return Ok(false); // 已在宿主(盘副本仍在)
        }
        let plan = self.bank.rel.disk.as_ref().expect("判过").plan_fill(chain)?;
        let bytes = crate::disk_io::read_block(plan.path.clone(), plan.bytes)
            .await
            .map_err(|e| owl_iface::contract::ModelError::Msg(e))?;
        // 宿主满 ⇒ 先逐最冷再落(与 offload 同一政策;裸 `host_put` 会在
        // 回填前撞 `Exhausted` ⇒ onload 整单失败)
        let r0 = self.bank.rel.leg.host_relief_blocks();
        self.bank
            .rel
            .host_fill(chain, owl_store::HostBlock::from_bytes(bytes))?;
        let relief = self.bank.rel.leg.host_relief_blocks().saturating_sub(r0);
        if relief > 0 {
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("store.host.relief_blocks", relief, file!(), line!());
            });
            owl_shared::log_warn!(
                "store",
                "宿主让位:磁盘回填前先逐最冷 {relief} 块(链 {chain:?})"
            );
        }
        self.bank.rel.disk.as_mut().expect("判过").commit_fill(chain)?;
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.disk.fill", 1, file!(), line!());
        });
        Ok(true)
    }

    /// **快照溢写**(状态腿宿主 → 磁盘)。语义同 [`Self::spill_kv_to_disk`]。
    async fn spill_snap_to_disk(&mut self, key: owl_store::SnapKey) -> Result<bool> {
        if self.bank.rel.snap_disk.is_none() {
            return Ok(false);
        }
        if self.bank.rel.snap_disk.as_ref().is_some_and(|d| d.contains(key)) {
            return Ok(false);
        }
        let buf = match self.bank.rel.snap_host.take(key) {
            Ok(b) => b,
            Err(owl_store::StoreError::Unknown(_)) => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        let mut plan = None;
        for attempt in 0..2 {
            let r = self.bank.rel.snap_disk.as_ref().expect("判过").plan_spill(key);
            match r {
                Ok(p) => {
                    plan = Some(p);
                    break;
                }
                Err(owl_store::StoreError::Exhausted(_)) if attempt == 0 => {
                    let need = self.bank.rel.snap_disk.as_ref().expect("判过").block_bytes();
                    let evp = self.bank.rel.snap_disk.as_ref().expect("判过").plan_evict(need, &[]);
                    if evp.victims.is_empty() {
                        break;
                    }
                    let paths: Vec<std::path::PathBuf> =
                        evp.victims.iter().map(|(_, p)| p.clone()).collect();
                    let keys: Vec<owl_store::SnapKey> = evp.victims.iter().map(|(k, _)| *k).collect();
                    crate::disk_io::remove_files(paths).await;
                    self.bank.rel.snap_disk.as_mut().expect("判过").commit_evict(&keys);
                }
                Err(_) => break,
            }
        }
        let Some(plan) = plan else {
            self.bank.rel.snap_host.put(key, buf)?;
            return Ok(false);
        };
        match crate::disk_io::write_block(plan.path.clone(), plan.tmp.clone(), buf.into_bytes()).await
        {
            Ok(_) => {
                self.bank.rel.snap_disk.as_mut().expect("判过").commit_spill(&plan)?;
                self.bank.rel.state.set_tier(key, owl_store::Tier::Disk)?;
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.snap.spill", 1, file!(), line!());
                });
                Ok(true)
            }
            Err(f) => {
                self.bank.rel
                    .snap_host
                    .put(key, owl_store::HostBlock::from_bytes(f.bytes))?;
                owl_shared::log_warn!("store", "快照溢写失败(内容已回宿主):{}", f.err);
                Ok(false)
            }
        }
    }

    /// **快照回填**(磁盘 → 宿主;S6)。
    pub(crate) async fn fill_snap_from_disk(&mut self, key: owl_store::SnapKey) -> Result<bool> {
        let Some(d) = self.bank.rel.snap_disk.as_ref() else { return Ok(false) };
        if !d.contains(key) {
            return Ok(false);
        }
        if self.bank.rel.snap_host.get(key).is_some() {
            let _ = self.bank.rel.snap_disk.as_mut().expect("判过").touch(key);
            return Ok(false);
        }
        let plan = self.bank.rel.snap_disk.as_ref().expect("判过").plan_fill(key)?;
        let bytes = crate::disk_io::read_block(plan.path.clone(), plan.bytes)
            .await
            .map_err(|e| owl_iface::contract::ModelError::Msg(e))?;
        self.bank.rel
            .snap_host
            .put(key, owl_store::HostBlock::from_bytes(bytes))?;
        self.bank.rel.snap_disk.as_mut().expect("判过").commit_fill(key)?;
        self.bank.rel.state.set_tier(key, owl_store::Tier::Host)?;
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.disk.snap.fill", 1, file!(), line!());
        });
        Ok(true)
    }

    /// **宿主腿一条 swap 落到每个 rank**(2026-10-20;双卡 KVMem 立案 §23.8)。
    ///
    /// `pieces[p]` = store 用**第 p 个 rank 自己的块 id** 组的片(两侧池句柄是两套,
    /// 不能靠偏移平移);宿主槽 = `world` 个平面,偏移由 store 的 `out/in_pieces`
    /// 按 `plane × 单页字节` 落好。
    /// 槽台 = **一次借台**,两条命令之间原箱传递 ⇒ 一次落账、一条链一份预算。
    /// 单卡(`world = 1`)= 与历史逐字相同。
    pub(crate) async fn swap_all_ranks(
        &mut self,
        dir: owl_iface::contract::SwapDir,
        pieces: &[Vec<owl_iface::contract::SwapPiece>],
        table: owl_store::SlotTable,
    ) -> Result<owl_store::SlotTable> {
        let world = self.bank.rel.world().max(1);
        if pieces.len() != world {
            return Err(owl_iface::contract::ModelError::EngineInvariant {
                what: format!(
                    "宿主腿 swap({dir:?}): 片组数 {} ≠ world {world}(store 组片与装配不一致)",
                    pieces.len()
                ),
            });
        }
        let mut table = table;
        for p in 0..world {
            let pcs = &pieces[p];
            let conn = match p {
                0 => {
                    self.dev
                        .graphs
                        .session
                        .face_mut()
                        .swap(dir, pcs, Box::new(table))
                        .await
                }
                _ => match self.peer.as_mut() {
                    Some(pr) => pr.graphs.session.face_mut().swap(dir, pcs, Box::new(table)).await,
                    None => {
                        return Err(owl_iface::contract::ModelError::EngineInvariant {
                            what: format!(
                                "宿主腿 swap({dir:?}): world = {world} 但 peer 束缺席(装配不一致)"
                            ),
                        })
                    }
                },
            };
            let conn = conn.map_err(|e| {
                owl_iface::contract::ModelError::Msg(format!(
                    "宿主腿 swap({dir:?}) rank{p} 失败(engine 编排): {e}"
                ))
            })?;
            table = owl_store::reclaim_slot_table(conn)?;
        }
        Ok(table)
    }

    /// **视图页表写两 rank**(TP;槽号两侧一致 ⇒ 同一份页表)。
    ///
    /// 只写 rank0 的后果(实测):peer 的 attention 读捕获期旧视图 ⇒ ~8k 起输出退化。
    pub(crate) async fn write_select_view_all(&mut self, pages: &[u32]) -> Result<()> {
        {
            let face = self.dev.graphs.session.face_mut();
            self.dev.pool.write_select_view(face, pages).await?;
        }
        if let Some(pr) = self.peer.as_mut() {
            let face = pr.graphs.session.face_mut();
            pr.pool.write_select_view(face, pages).await?;
        }
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
        // ① 换出/驱逐(**两腿同单**:KV 链/叶 + GDN 快照;§30)
        for item in &verdict.out {
            match item {
                owl_store::EvictOut::Kv { chain, sink, .. } => match sink {
                    Sink::Host => {
                        let relief0 = self.bank.rel.leg.host_relief_blocks();
                        let (blocks, pieces, table) = {
                            match self.bank.rel.plan_kv_offload(*chain) {
                                Ok(p) => (p.blocks, p.pieces, p.table),
                                Err(owl_store::StoreError::Busy(_)) => {
                                    a.skipped_busy += 1;
                                    continue;
                                }
                                Err(e) => return Err(e.into()),
                            }
                        };
                        // 宿主满时 `plan_kv_offload` 会**先逐最冷**再借台 ——
                        // 这不是错，但说明计划面没预见到(盘写失败/账漂);
                        // 如实上账，别让"计划曾经乐观"隐身。
                        let relief = self
                            .bank
                            .rel
                            .leg
                            .host_relief_blocks()
                            .saturating_sub(relief0);
                        if relief > 0 {
                            owl_shared::metrics::with_metrics_store(|m| {
                                m.counter_add(
                                    "store.host.relief_blocks",
                                    relief,
                                    file!(),
                                    line!(),
                                );
                            });
                            owl_shared::log_warn!(
                                "store",
                                "宿主让位:offload 前先逐最冷 {relief} 块(计划面未预见;链 {chain:?})"
                            );
                        }
                        let table = match self
                            .swap_all_ranks(owl_iface::contract::SwapDir::Out, &pieces, table)
                            .await
                        {
                            Ok(t) => t,
                            Err(e) => {
                                owl_store::abort_offload(&mut self.bank.rel.leg, &blocks);
                                return Err(e);
                            }
                        };
                        a.offloaded +=
                            owl_store::commit_offload(&mut self.bank.rel.leg, &blocks, table)?;
                    }
                    Sink::Drop => match self.bank.rel.leg.drop_cached(*chain) {
                        Ok(()) => a.dropped += 1,
                        Err(owl_store::StoreError::Busy(_)) => a.skipped_busy += 1,
                        Err(e) => return Err(e.into()),
                    },
                    // **第三层(S6)**:宿主 → 磁盘(内容不丢)。
                    // 票是**链级**的(`chain` = 链尾,S6.1):整链里所有**在宿主**的块
                    // 逐块写盘 —— 只挪尾部会拆散链,重启重建不出从根连续的前缀。
                    Sink::Disk => {
                        let blocks = self.bank.rel.leg.chain_blocks(*chain);
                        let mut n = 0usize;
                        for b in blocks {
                            if self.spill_kv_to_disk(b).await? {
                                n += 1;
                            }
                        }
                        if n > 0 {
                            a.spilled += n;
                        } else {
                            a.skipped_busy += 1;
                        }
                    }
                },
                owl_store::EvictOut::Snap { key, sink } => match sink {
                    Sink::Host => {
                        // 不在设备槽 = 判与搬之间已被移走/已在宿主 ⇒ 跳过(幂等)
                        let Some(slot) = self.dev.pool.snap_slot_of(*key) else {
                            a.skipped_busy += 1;
                            continue;
                        };
                        self.snap_offload(*key, slot).await?;
                        self.dev.pool.snap_clear_slot(slot);
                        if let Some(pr) = self.peer.as_mut() {
                            pr.pool.snap_clear_slot(slot);
                        }
                        a.snap_offloaded += 1;
                    }
                    Sink::Drop => {
                        let slot = self.dev.pool.snap_slot_of(*key);
                        self.bank.rel.drop_snap_host(*key)?;
                        match self.bank.rel.state.drop_snap(*key) {
                            Ok(_) => {}
                            // 已不在账 = 幂等成功(判与搬之间被别处出账)
                            Err(owl_store::StoreError::Unknown(_)) => {}
                            Err(e) => return Err(e.into()),
                        }
                        if let Some(slot) = slot {
                            self.dev.pool.snap_clear_slot(slot);
                            if let Some(pr) = self.peer.as_mut() {
                                pr.pool.snap_clear_slot(slot);
                            }
                        }
                        a.snap_dropped += 1;
                    }
                    // **第三层(S6)**:快照宿主 → 磁盘(Transient 不会走到这里 —— 判据已排除)
                    Sink::Disk => {
                        if self.spill_snap_to_disk(*key).await? {
                            a.snap_spilled += 1;
                        } else {
                            a.skipped_busy += 1;
                        }
                    }
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
            let table = match self
                .swap_all_ranks(owl_iface::contract::SwapDir::In, &pieces, table)
                .await
            {
                Ok(t) => t,
                Err(e) => {
                    owl_store::abort_onload(&mut self.bank.rel.leg, &cs, &slots);
                    return Err(e);
                }
            };
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
    /// 判定内容全在 store([`Rel::plan_need`](owl_store::Rel::plan_need),唯一产地)。
    ///
    /// §12.4 口径:**需求驱动** —— `need` 由 [`Self::step_need`] 与 `schedule()`
    /// 的同一份算式给出;零需求不搬。宿主两腿都关 = 直接返回(无下沉面;
    /// 与接线前行为一致)。
    /// 判定单里的 `deny` = **"即使搬空也不够"的真容量不足预警** ⇒ 响亮 WARN,
    /// 但 `out` 里能搬的照搬(下沉不丢内容);真拒绝由 `ensure_for_len` 兜底。
    pub(crate) async fn pool_housekeeping(&mut self, need: owl_store::Need) -> Result<()> {
        // 两条腿各自"宿主关 ⇒ 该腿目标清零"(旧行为:宿主关 = 不换出,**也不做无谓驱逐**;
        // 两腿独立,不能因为状态腿开就顺手把 KV 面 Drop 掉)
        let mut need = need;
        if !self.bank.rel.leg.host_enabled() {
            need.slots = 0;
        }
        if self.bank.rel.snap_host_budget() == 0 {
            need.snap_bytes = 0;
        }
        if need.is_zero() {
            return Ok(());
        }
        let verdict = self.bank.rel.plan_need(need, &owl_store::Protected::default());
        if owl_shared::env_reader::flag("OWL_STORE_HOOK_DEBUG") {
            eprintln!(
                "[hook] {:?} need={}槽/{}B free={}/{} host_free={} disk_free={} 整链候选={} 判 out={} in={} deny={:?}",
                need.why,
                need.slots,
                need.snap_bytes,
                self.bank.rel.leg.slots_free(),
                self.bank.rel.leg.slots_total(),
                self.bank.rel.leg.host_free_slots(),
                self.bank.rel.disk.as_ref().map(|d| d.free_slots()).unwrap_or(0),
                self.bank.rel.leg.host_chain_candidates(&[]).len(),
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
            //
            // **日志限流**(2026-10-19;压力节点每拍都跑 ⇒ 不限流必刷屏):
            // 首条 + 每 64 条打一次,带累计数;计数本身逐次入 metrics,不丢。
            self.degrade_seen = self.degrade_seen.saturating_add(1);
            let n = self.degrade_seen;
            if n == 1 || n.is_multiple_of(64) {
                owl_shared::log_warn!(
                    "store",
                    "钩子:需求凑不够(真容量不足预警,累计 {n} 次):{why}"
                );
            }
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("store.degrade", 1, file!(), line!());
            });
        }
        let applied = self.apply_verdict(&verdict).await?;
        owl_shared::log_debug!(
            "store",
            "钩子({:?}) 判 out={n_out} in={n_in} ⇒ 搬:KV 下沉 {} 驱逐 {} 溢盘 {} 回灌 {} / 快照 下沉 {} 溢盘 {} 丢 {} / 跳过 {}",
            need.why,
            applied.offloaded,
            applied.dropped,
            applied.spilled,
            applied.onloaded,
            applied.snap_offloaded,
            applied.snap_spilled,
            applied.snap_dropped,
            applied.skipped_busy
        );
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.hook.out", applied.offloaded as u64, file!(), line!());
            m.counter_add("store.hook.drop", applied.dropped as u64, file!(), line!());
            m.counter_add("store.hook.in", applied.onloaded as u64, file!(), line!());
            m.counter_add("store.hook.snap.out", applied.snap_offloaded as u64, file!(), line!());
            m.counter_add("store.hook.snap.drop", applied.snap_dropped as u64, file!(), line!());
            m.counter_add("store.hook.spill", applied.spilled as u64, file!(), line!());
            m.counter_add("store.hook.snap.spill", applied.snap_spilled as u64, file!(), line!());
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
        // P2.2 侦察(2026-10-19):pump 内分段墙钟(pop-events / 头 / sched / exec / 尾)
        let t_ph0 = prof.then(std::time::Instant::now);
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
        // P0.3 定谳(2026-10-19):**循环头**耗时(需求驱动保洁:need/gate/housekeeping)
        // —— 它是"步墙钟 − 设备 − fill/launch − 采样"的缺口候选(此前未打点)。
        let t_loophead = (prof || true).then(std::time::Instant::now);
        let need = self.step_need()?;
        let mut need = need;
        self.admission_gate(&mut need)?;
        self.node(Node::Need(need.clone())).await?;
        // **[N6 Pressure]**(缺口 3):余量告急 ⇒ **零缺口也搬**(先下沉、再驱逐);
        // 这是"压力路径第一步":先把自己能腾的腾出来,而不是等 `ensure_for_len` 撞墙。
        if let Some(k) = self.pressure_probe() {
            let free = owl_shared::vram::free_bytes();
            owl_shared::log_warn!(
                "store",
                "N6 压力节点:驱动余量 {:.0}MiB < 目标 ⇒ 目标腾 {k} 槽(先下沉后驱逐)",
                free as f64 / 1048576.0
            );
            self.node(Node::Pressure(k)).await?;
        }
        let loophead_ms = t_loophead.map(|t| t.elapsed().as_secs_f64() * 1e3).unwrap_or(0.0);
        let t_lh1 = prof.then(std::time::Instant::now); // 循环头之后
        owl_shared::timer_start!("engine.pump.schedule");
        let t_sched = (prof || true).then(std::time::Instant::now);
        let out = self.schedule()?;
        let t_sc1 = prof.then(std::time::Instant::now); // schedule 之后
        if let Some(ts) = t_sched {
            let d = ts.elapsed();
            self.sched_acc_ns += d.as_nanos() as u64;
            self.sched_cnt += 1;
            if self.sched_cnt.is_multiple_of(16) {
                owl_shared::log_info!("sched", "n={} avg={:.2}ms last={:.2}ms loophead={:.3}ms",
                    self.sched_cnt,
                    self.sched_acc_ns as f64 / self.sched_cnt as f64 / 1e6,
                    d.as_secs_f64() * 1e3,
                    loophead_ms
                );
            }
        }
        owl_shared::timer_end!("engine.pump.schedule");
        if prof {
            // ⚠️ **降频**(P2.2 立案;2026-10-19):原**每步**查一次 metrics ⇒
            // 实测 **1.63ms/步**（"探针观测即污染"—— 它把 `step_profile` 档的
            // 读数整体抬高 ~8%,还让"循环外 1.55ms"成为 **探针假象**:分段打点
            // 显示 `sched→exec` 1.63ms、而 schedule 本身 ≈0 ⇒ 靶就是这一句）。
            // 改成每 64 步一次(泵循环探针面足够;静态计数零结构改动)。
            static PUMP_METRIC_TICK: std::sync::atomic::AtomicU64 =
                std::sync::atomic::AtomicU64::new(0);
            if PUMP_METRIC_TICK
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                .is_multiple_of(64)
            {
                owl_shared::metrics::query_metrics(&owl_shared::metrics::MetricsFilter::new()
                    .tag_prefix("engine.pump."));
            }
        }
        match out {
            SchedulerOutput::Idle => {
                // **[H5 Idle]**(步 4 入档):空转期每 N 拍做一次整理(冷面下沉/清池),
                // 决策仍全在 store(`NeedWhy::Idle` = 零缺口也搬);无活跃 turn ⇒
                // 只动缓存面,绝不碰活跃面。
                // **池缓存回收(空转出口;§23.9)**:server 的泵循环 `Idle` 即断,
                // `Node::Idle`(每 16 拍)到不了 ⇒ 在这里按时间节流清一次。
                // `RELEASE_THRESHOLD=∞` 下 turn 收尾后 free 的 GB 级临时块
                // 不主动还驱动就永远占着(实测 peer 侧 3.0 GiB)。
                let due = self
                    .last_trim
                    .is_none_or(|t| t.elapsed() >= Self::IDLE_TRIM_EVERY);
                if due {
                    self.last_trim = Some(std::time::Instant::now());
                    // 先排空再清:异步 free 要等流上事件落地才进池的"空闲页"表
                    for (rank, r) in [
                        (0u8, self.dev.graphs.session.face_mut().sync().await),
                        match self.peer.as_mut() {
                            Some(p) => (1u8, p.graphs.session.face_mut().sync().await),
                            None => (1u8, Ok(())),
                        },
                    ] {
                        if let Err(e) = r {
                            owl_shared::log_warn!("store", "空转排空失败(rank{rank}): {e}");
                        }
                    }
                    let freed = self.trim_pools().await;
                    let total: u64 =
                        freed.iter().filter_map(|(_, r)| r.as_ref().ok()).sum();
                    if total > 0 {
                        owl_shared::log_info!(
                            "store",
                            "空转池缓存回收 {:.1} MiB(逐 rank: {:?})",
                            total as f64 / 1048576.0,
                            freed
                                .iter()
                                .map(|(r, x)| (*r, x.as_ref().map(|n| n >> 20).unwrap_or(0)))
                                .collect::<Vec<_>>()
                        );
                    }
                    for (rank, r) in freed {
                        if let Err(e) = r {
                            owl_shared::log_warn!(
                                "store",
                                "空转池缓存回收失败(rank{rank}): {e}"
                            );
                        }
                    }
                }
                if let Some(k) = self.idle_tick() {
                    self.node(Node::Idle(k)).await?;
                }
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
                        let t_ex0 = t_ph0.map(|_| std::time::Instant::now());
                        let r = self.execute_decode(token, pos, kv_slot, gdn_slot, grew).await;
                        if let (Some(p0), Some(ex0)) = (t_ph0, t_ex0) {
                            let head = ex0.duration_since(p0);
                            let exec = ex0.elapsed();
                            let lh = t_lh1.map(|t| t.duration_since(p0)).unwrap_or_default();
                            let sc = t_sc1.map(|t| t.duration_since(t_lh1.unwrap_or(t))).unwrap_or_default();
                            let tail_head = t_sc1.map(|t| ex0.duration_since(t)).unwrap_or_default();
                            owl_shared::log_info!(
                                "pump-phases",
                                "head={head:?} [entry→lh={lh:?} lh→sched={sc:?} sched→exec={tail_head:?}] exec={exec:?}"
                            );
                        }
                        owl_shared::timer_end!("engine.phase.decode");
                        self.set_phase(saved);
                        r
                    }
                    StepAction::BatchDecode { rows } => {
                        let saved = self.dev.env.phase;
                        self.set_phase(owl_models::env::EvalPhase::Decode);
                        owl_shared::timer_start!("engine.phase.batch");
                        let n = rows.len();
                        let r = self.execute_batch_decode(rows).await;
                        owl_shared::timer_end!("engine.phase.batch");
                        self.set_phase(saved);
                        if let Ok(ev) = &r {
                            owl_shared::log_debug!("batch", "批步完成 n={n} first={ev:?}");
                        }
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
mod topk_merge_tests {
    //! P1.1 候选合并单测（纯）
    use super::merge_topk_two_halves;

    /// rank0 低半 ++ rank1 高半：跨半的全局前三正确（含 id 偏移）
    #[test]
    fn merges_across_halves_with_offset() {
        // rows=1, k=2, half=100
        let va = [5.0, 1.0];
        let ia = [3.0, 4.0];
        let vb = [9.0, 2.0];
        let ib = [7.0, 8.0];
        let (v, i) = merge_topk_two_halves(&va, &ia, &vb, &ib, 1, 2, 100).unwrap();
        assert_eq!(i, vec![107, 3], "全局 id = 局部 + half（rank1）");
        assert_eq!(v, vec![9.0, 5.0]);
    }

    /// 平局取小 id（与核同律）
    #[test]
    fn ties_prefer_smaller_id() {
        let va = [1.0];
        let ia = [5.0];
        let vb = [1.0];
        let ib = [2.0]; // 全局 102
        let (_, i) = merge_topk_two_halves(&va, &ia, &vb, &ib, 1, 1, 100).unwrap();
        assert_eq!(i, vec![5], "平局 ⇒ 小 id 在前");
    }

    /// 非有限值（空候选槽）被跳过
    #[test]
    fn skips_non_finite() {
        let va = [f32::NEG_INFINITY, 3.0];
        let ia = [0.0, 9.0];
        let vb = [1.0, 2.0];
        let ib = [0.0, 1.0];
        let (_, i) = merge_topk_two_halves(&va, &ia, &vb, &ib, 1, 2, 100).unwrap();
        assert_eq!(i, vec![9, 101]);
    }

    /// 多行各自独立 + 形状不符响亮拒
    #[test]
    fn rows_independent_and_rejects_malformed() {
        let (va, ia) = ([3.0, 2.0], [1.0, 2.0]);
        let (vb, ib) = ([4.0, 1.0], [1.0, 2.0]);
        let (v, _) = merge_topk_two_halves(&va, &ia, &vb, &ib, 2, 1, 10).unwrap();
        assert_eq!(v, vec![4.0, 2.0], "行 0 取 B/行 1 取 A");
        assert!(merge_topk_two_halves(&va[..1], &ia, &vb, &ib, 2, 1, 10).is_err());
        assert!(merge_topk_two_halves(&va, &ia, &vb, &ib, 2, 0, 10).is_err());
    }
}

#[cfg(test)]
mod head_concat_tests {
    //! P1.1 半表拼接单测（纯）
    use super::concat_head_rows;

    /// 单行：rank0 低半 + rank1 高半 = 全局词表序
    #[test]
    fn single_row_low_then_high() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        assert_eq!(concat_head_rows(&a, &b, 3).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    /// 多行（verify 形状）：每行各自拼接，不串行
    #[test]
    fn rows_concat_independently() {
        // rows = 2, half = 2
        let a = [1.0, 2.0, 3.0, 4.0]; // 行0=[1,2] 行1=[3,4]
        let b = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(concat_head_rows(&a, &b, 2).unwrap(), vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0]);
    }

    /// 响亮拒：长度不等 / 非整倍 / 半表宽 0
    #[test]
    fn rejects_malformed() {
        assert!(concat_head_rows(&[1.0, 2.0], &[3.0], 2).is_err());
        assert!(concat_head_rows(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], 2).is_err());
        assert!(concat_head_rows(&[], &[], 0).is_err());
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
