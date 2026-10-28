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
use crate::housekeeping::Housekeeping;
use crate::scheduler::{SchedulerOutput, StepAction};
use crate::select::SelectRuntime;
use crate::session::SessionBank;
use crate::store_hook::{
    Acquire, LoadRequest, Maintain, Release, ReleaseKind, StateAcquire, StoreHook, StoreOutcome,
};
use crate::turn::{Turn, TurnEvent, TurnLedger, TurnSpec, TurnSpecState};

type Result<T> = std::result::Result<T, ModelError>;

// Store 钩子的请求/回执类型在 `store_hook.rs`；唯一执行臂是
// [`RunningEngine::store_hook`]。旧 `Node` 四变体已于 2026-10-25 Phase 1
// 收编为 `StoreHook::{Acquire, Release, Maintain}`。

/// 步级探针开关(诊断面;结构纯数据,Default = 全关)。
/// **入口侧构造器** = [`StepProbes::from_config`](server config/测试用例
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
    /// B4:OWL_SPEC_DEGRADE_AFTER(m=0 连击降级阈值;0 = 禁用)
    pub degrade_after: usize,
    /// B4:OWL_SPEC_PROBE_EVERY(降级期探测周期 token 数,失败 ×2 退避)
    pub probe_every: usize,
    /// GDN 分派门控打印(原 OWL_TRACE_GATE 层内直读 → ForwardCtx 注入)
    pub trace_gate: bool,
    /// E3 收编(原热路径散落 var_os;boot 解析一次):DFlash2 诊断/回退族
    pub dflash_probe: bool,
    /// 渲染后 prompt 落盘诊断(原 OWL_PROMPT_DUMP;2026-10-15 收编)
    pub prompt_dump: bool,
    /// KV 页差分落点(原 OWL_KV_DUMP,同上收编)
    pub kv_dump_at: Option<usize>,
    /// 解释器活性收尾归因打印(原 `OWL_LIVE_DIAG`)
    pub live_diag: bool,
    // 2026-10-27 配置面剪枝:一次性排障旗标(ck/mfact/view-hash/propose-eager/
    // dflash-*/gdn_dbg/pf_fence/B 轴六件)已随 `[probes]` 一起退役。
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
            degrade_after: cfg.spec.degrade_after,
            probe_every: cfg.spec.probe_every,
            trace_gate: p.trace_gate,
            dflash_probe: p.dflash_probe,
            prompt_dump: p.prompt_dump,
            // ⚠️ `0` = 关(与 full.toml 的文档口径一致):`Option` 会把 toml 的
            // `kv_dump_at = 0` 解析成 **Some(0)** —— 那意味着"每个请求 pos 0
            // 各 d2h 一页",是纯税(2026-10-27 配置面 review 抓到的现役 bug)。
            kv_dump_at: p.kv_dump_at.filter(|v| *v > 0).or_else(|| {
                owl_shared::env_reader::parse::<usize>("OWL_KV_DUMP").filter(|v| *v > 0)
            }),
            live_diag: p.live_diag,
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
/// # 结构(2026-10-14 拆万能袋 → 2026-10-21 收户)
///
/// 本体只留**跨域协调**要抓的把手,状态按域入组合:
/// - [`BootCtx`]:boot 期定型、**运行期只读**(配置/分词器/探针/序号)
/// - [`crate::turn::TurnLedger`]:排队/活跃/id 发号/事件队列
/// - [`crate::session::SessionBank`]:会话账 + 双块账房(target/MTP)
/// - [`crate::select::SelectRuntime`]:B 轴视图/覆盖率探针/驻留账
/// - [`crate::housekeeping::Housekeeping`]:节拍与计数(调度计时/空转/回收/降级)
/// - [`RankDev`]:rank0 设备束(池/图/模型/env/spec 运行态/tp)
///
/// 纪律:各户字段**私有**,外部只经方法读写 —— 状态有守门人,
/// 方法离它操作的状态近(收户前的反例:`self.dev` 被解引 300+ 次、
/// B 轴 9 字段的写者散在四个文件)。
///
/// 生命周期分层:turn 域状态住 ActiveTurn(随 turn 生死);引擎级
/// spec 状态住 SpecRuntime(跨 turn)—— 两层不混居(2026-10-10
/// 生命周期收口教训)。
pub struct RunningEngine<D: DeviceClient> {
    /// boot 期事实(只读;写入只在构造期)
    pub(crate) boot: BootCtx,
    /// **前缀无感删除的观测账**(leave-gap):决策事实(删除区间/保留块/
    /// 快照重挂数)。落账在 `scheduler::apply_image_leave_gap`;门与运维
    /// 共用这一份(**不许再从日志里反推**)。
    pub(crate) img_evict: ImgEvictLedger,
    /// turn 运行账(turn.rs)
    pub(crate) turns: TurnLedger,
    /// 会话 + 块账房(session.rs;host 侧 —— 两 rank 共用一份)
    pub(crate) bank: SessionBank,
    /// B 轴运行态(select.rs;字段私有,只经方法访问)
    pub(crate) sel: SelectRuntime,
    /// 保洁/遥测账(housekeeping.rs;字段私有,只经方法访问)
    pub(crate) hk: Housekeeping,
    /// **rank0 设备束**(池/图/模型/env/spec 运行态/tp;S4 ⑤-K3 收编)
    pub(crate) dev: RankDev<D>,
    /// **TP peer 束(rank1;`None` = 单卡)** —— 设备侧程序镜像的目标
    /// (S4 ⑤-K3:同一份设备侧代码,换装到 peer 束上再跑一遍)
    pub(crate) peer: Option<RankDev<D>>,
    /// **视觉塔逐图 merged 缓存**(每 rank 一份;跨 turn,按像素内容指纹命中)。
    /// 索引 0 = rank0,1 = peer;缓存块归 `Owner::Cache`,不随 turn 回收。
    pub(crate) vlm_cache: [crate::vlm::VlmMergedCache; 2],
    /// **在飞的 stage-out**(2026-10-27):下沉命令已提交、回执未收。
    /// 安全点(下一个 chunk 已入队、GPU 正在算)由 [`Self::drain_pending_out`] 收口。
    pub(crate) pending_out: Option<PendingOut>,
}

/// **在飞的下沉命令**(分离式换出)。
///
/// 为什么可以"发出去先不管":
/// - `plan_offload_blocks` 已把块票占住 ⇒ 在 commit 之前**没有任何路径能为
///   这些槽签发新 lease**(在飞槽不可复用的结构保证);
/// - 被沉的块**不在**选择集里 ⇒ 本 chunk 的视图与下一个 chunk 的 `past_view`
///   都不读它们(表已写占位符);
/// - `SlotTable` 单主零锁 ⇒ 同一时刻只允许一条 out 在飞(`drain` 后清)。
///
/// 本结构尚在 = 账尚未结 — 不许推迟到 turn 收口后(那吋池可能已在别处填满)。
/// `since` 给诊断面(在飞时长)。
pub(crate) struct PendingOut {
    /// 会话(表写回/恢复用)
    pub(crate) sid: u64,
    /// (表内下标, 身份, 原设备槽):drain **失败**时按它把表恢复成原槽
    pub(crate) blocks: Vec<(usize, owl_store::ChainHash, u32)>,
    /// 每个 rank 一个回执句柄(单卡 ⇒ 1 个;分离式目前只开在单卡)
    pub(crate) waiters: Vec<owl_iface::contract::SwapWaiter>,
    /// 提交时刻(诊断:在飞时长)
    pub(crate) since: std::time::Instant,
}

/// **前缀无感删除的观测账**(进程内;`Default` = 从未吸收过删除)。
#[derive(Default, Debug, Clone)]
pub struct ImgEvictLedger {
    /// 累计吸收次数(一次 turn 最多一次)
    pub absorbed: u64,
    /// 最近一次的决策事实(`None` = 还没吸收过)
    pub last: Option<ImgEvictTrace>,
}

/// 最近一次删除的事实(G1 机械门 / 运维取证用)。
///
/// 字段目前只被引擎门与未来 `/debug/*` 面板读取;生产热路径不读 ⇒ 这里显式
/// 允许 dead_code(它是**观测面**的事实契约,不是死代码)。
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ImgEvictTrace {
    /// 删除区间(旧 token 坐标;页对齐)
    pub a: usize,
    /// 删除区间终点(不含;整页数)
    pub b: usize,
    /// 复用后缀的整块数
    pub keep_blocks: usize,
    /// 快照重挂条数(设备 + 宿主;盘层异步另计)
    pub rekeyed: usize,
}

/// **boot 期定型、运行期只读**的事实。
///
/// 为什么单独立户:配置/分词器/探针/boot 序号原来各占一个字段,被 ~150 个
/// 方法随手读 —— 但它们**没有写入者**(`cfg`/`tok`/`probes` 在运行期零赋值)。
/// 收成只读结构后,"谁能改运行态"的边界在类型上就看得见:
/// 要 `&mut` 的只有 turns / bank / sel / hk / dev / peer。
pub(crate) struct BootCtx {
    pub(crate) cfg: EngineConfig,
    pub(crate) tok: Tokenizer,
    pub(crate) probes: StepProbes,
    /// 消息开启标记的 token 序列（`[pool].snap_msg_delim` 或模型
    /// `ChatFormat.im_open` 在 boot 期 tokenize 的结果；只读）。
    pub(crate) msg_open_ids: Vec<u32>,
    /// boot 序号(metrics 打点命名空间;进程内自增,b1/b2…)
    pub(crate) seq: u64,
}

/// **紧急压力单次腾槽上限**（`PoolExhausted` 后的收缩；原 N6 改名）：
/// 一次最多腾这么多槽，避免"一口气搬空"把正在跑的 turn 的工作集也冲掉。
const PRESSURE_MAX_SLOTS: usize = 64;

/// 空转 compact 单次腾槽上限（显式 `idle_compact=true` 才走）。
const IDLE_COMPACT_MAX_SLOTS: usize = 16;

/// 单次 HostPressure 最多补多少宿主槽（防止低水位目标过大时一次搬空）。
const HOST_PRESSURE_MAX_SLOTS: usize = 64;

/// **压力信号快照**（P1-3）：把"池/宿主/磁盘/驱动"的余量一次读齐。
///
/// 读的时候全部是纯 host 账；不触发 I/O，也不改任何状态。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PressureSignals {
    pub(crate) pool_free_pages: usize,
    pub(crate) pool_total_pages: usize,
    /// KV 宿主腿 + 状态腿的剩余槽（合计；用于判断是否还能下沉）。
    pub(crate) host_free_slots: usize,
    /// 分腿余量/开关：启用的腿低于低水位 ⇒ 需要 HostPressure。
    pub(crate) host_kv_enabled: bool,
    pub(crate) host_snap_enabled: bool,
    pub(crate) host_kv_total_slots: usize,
    pub(crate) host_snap_total_slots: usize,
    pub(crate) host_kv_free_slots: usize,
    pub(crate) host_snap_free_slots: usize,
    /// KV 盘腿 + 状态盘腿的剩余槽。
    pub(crate) disk_free_slots: usize,
    pub(crate) disk_kv_free_slots: usize,
    pub(crate) disk_snap_free_slots: usize,
    /// 最紧一张卡的驱动 free（TP2 取两 rank 的 min）。
    pub(crate) driver_free_bytes: u64,
    pub(crate) reserve_bytes: u64,
}

impl PressureSignals {
    /// 驱动余量已低于 boot 预留。
    pub(crate) fn driver_below_reserve(&self) -> bool {
        self.driver_free_bytes < self.reserve_bytes
    }

    /// 一行诊断（日志/报告；不参与判据）。
    pub(crate) fn summary(&self) -> String {
        format!(
            "pool {}/{} 页 · host {} 槽 · disk {} 槽 · driver {:.0}MiB/reserve {:.0}MiB{}",
            self.pool_free_pages,
            self.pool_total_pages,
            self.host_free_slots,
            self.disk_free_slots,
            self.driver_free_bytes as f64 / 1048576.0,
            self.reserve_bytes as f64 / 1048576.0,
            if self.driver_below_reserve() { " · 驱动低于预留" } else { "" }
        )
    }
}

/// **压力水位策略**：低水位触发、高水位目标（滞回）。
///
/// 百分比是主池页数的占比，不是驱动字节；固定池档的池墙才是主判据。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PressurePolicy {
    low_pct: usize,
    high_pct: usize,
    host_low_pct: usize,
    host_high_pct: usize,
}

impl Default for PressurePolicy {
    fn default() -> Self {
        Self {
            low_pct: 12,
            high_pct: 25,
            host_low_pct: 12,
            host_high_pct: 25,
        }
    }
}

impl PressurePolicy {
    pub(crate) fn new(
        low_pct: usize,
        high_pct: usize,
        host_low_pct: usize,
        host_high_pct: usize,
    ) -> Self {
        let low = low_pct.clamp(1, 100);
        let high = high_pct.clamp(low, 100);
        let host_low = host_low_pct.clamp(1, 100);
        let host_high = host_high_pct.clamp(host_low, 100);
        Self {
            low_pct: low,
            high_pct: high,
            host_low_pct: host_low,
            host_high_pct: host_high,
        }
    }

    /// 宿主腿低水位（按该腿总槽数）。
    pub(crate) fn host_low_slots(&self, total: usize) -> usize {
        Self::slots_at(total, self.host_low_pct)
    }

    /// 宿主腿目标水位（触发后补到该值；≥ 低水位）。
    pub(crate) fn host_high_slots(&self, total: usize) -> usize {
        Self::slots_at(total, self.host_high_pct).max(self.host_low_slots(total))
    }

    fn slots_at(total: usize, pct: usize) -> usize {
        if total == 0 {
            return 0;
        }
        (total.saturating_mul(pct) / 100).clamp(1, total)
    }

    /// 触发当前压力所需的目标。`free >= low` 返回 0（不搬）。
    pub(crate) fn target_slots(&self, total: usize, free: usize, cap: usize) -> usize {
        let low = Self::slots_at(total, self.low_pct);
        if free >= low {
            return 0;
        }
        let high = Self::slots_at(total, self.high_pct).max(low);
        high.saturating_sub(free).min(cap)
    }
}

/// 生成相位 spec 档位（只用于 `generation_target_for` 的纯算式）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GenerationKind {
    None,
    Mtp,
    DFlash2,
}

/// **一个 pump 步内生成相位可能长到的最远 token 位**（纯函数；Phase 2）。
///
/// - 裸 decode / Dumb：`pos + d + 1`（verify 行）；
/// - MTP：`pos + 2d`（verify + 轮末 propose `pos+m+d`，`m ≤ d`）；
/// - DFlash2：`pos + 2d + 2`（propose `fp+block`，`fp=pos+m+1`、
///   `block=d+1`、`m ≤ d`）。
fn generation_target_for(pos: usize, depth: usize, kind: GenerationKind) -> usize {
    if depth == 0 {
        return pos.saturating_add(1);
    }
    match kind {
        GenerationKind::DFlash2 => pos
            .saturating_add(depth.saturating_mul(2))
            .saturating_add(2),
        GenerationKind::Mtp => pos.saturating_add(depth.saturating_mul(2)),
        GenerationKind::None => pos.saturating_add(depth).saturating_add(1),
    }
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
pub(crate) fn concat_head_rows(a: &[f32], b: &[f32], half: usize) -> crate::Result<Vec<f32>> {
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
    if k == 0
        || vals_a.len() != want
        || vals_b.len() != want
        || ids_a.len() != want
        || ids_b.len() != want
    {
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
        self.boot.cfg.prefill_chunk
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
        self.hk.batch_steps()
    }

    /// 合批图是否在场(B5.1 M1b 事实位;false = 执行器走 eager 臂)
    /// 合批上限(`[runtime] max_batch`;1 = 串行现状)。服务面诊断/门用。
    pub fn max_batch(&self) -> usize {
        self.boot.cfg.knobs.max_batch
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
        self.dev.boot.model.logits_vocab() != self.dev.boot.model.vocab_size()
    }

    /// **候选采样读面**(2026-10-26):裸 decode 的 `cand` 两半表 → 每行候选并集
    /// (全局索引升序)。`None` = 图没有候选槽(未切档 / greedy 档 flavor)⇒
    /// 调用方回退 `read_full_logits`。
    pub(crate) async fn read_session_cands(&mut self) -> Result<Option<Vec<Vec<(f32, u32)>>>> {
        if !self.dev.graphs.session.has_output("cand") {
            return Ok(None);
        }
        let half = self.dev.boot.model.logits_vocab();
        let a = self.dev.graphs.session.read_output_f32("cand").await?;
        let p = self
            .peer
            .as_mut()
            .ok_or_else(|| ModelError::EngineInvariant {
                what: "候选采样:词表切档缺 peer（装配不一致）".into(),
            })?;
        let b = p.graphs.session.read_output_f32("cand").await?;
        Ok(Some(crate::exec::merge_sample_cands(&a, &b, 1, half)?))
    }

    pub(crate) async fn read_full_logits(&mut self, name: &str) -> Result<Vec<f32>> {
        let a = self.dev.graphs.session.read_output_f32(name).await?;
        if !self.head_split_active() {
            return Ok(a);
        }
        let half = self.dev.boot.model.logits_vocab();
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
        let Some(p) = self.peer.as_mut() else {
            return Ok(None);
        };
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
    pub fn tokenize(
        &self,
        text: &str,
        chat_wrap: bool,
    ) -> std::result::Result<TokenizeReply, String> {
        let tokens = if chat_wrap {
            self.boot.tok.try_encode(&self.boot.tok.chat_wrap(text))?
        } else {
            self.boot.tok.try_encode(text)?
        };
        Ok(TokenizeReply {
            tokens,
            max_seq_tokens: self.boot.cfg.max_seq_tokens,
        })
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
            crate::turn::TurnOpts {
                ignore_eos,
                ..Default::default()
            },
        )
    }

    /// **本轮 prompt 的逐 token 绝对位置账**(单一产地)。
    ///
    /// - 会话已有位置账:删除段 `[a,b)` 的幸存后缀保留旧 position,新尾巴
    ///   从 `max_pos+1` 追加;普通追加同样保留旧前缀;
    /// - 首轮 / 无位置账:从 0 起紧凑重算。
    ///
    /// 返回 `(positions, positions_gap)`;`positions_gap=true` 表示位置已
    /// 离开紧凑坐标,调用方须禁用跨会话内容缓存。
    fn prompt_positions(
        &self,
        session: Option<u64>,
        prompt_ids: &[u32],
        spans: &[crate::img_evict::ImgSpan],
        deletion: Option<crate::img_evict::TextDelete>,
    ) -> Result<(Vec<crate::img_evict::Pos3>, bool)> {
        let Some(vlm) = self.dev.boot.vlm.as_ref() else {
            // 纯文本档:位置 = 标量下标(逐字节旧行为)。
            return Ok((
                (0..prompt_ids.len()).map(|i| [i as i32; 3]).collect(),
                false,
            ));
        };
        let image_token = vlm.cfg().image_token;
        let mrope = &vlm.cfg().mrope;
        if let Some(sid) = session {
            if let Some(s) = self.bank.sessions.get(sid) {
                if !s.positions.is_empty() && s.positions.len() == s.tokens.len() {
                    let (pos, _max, mut gap) = crate::img_evict::positions_after_splice(
                        &s.tokens,
                        &s.positions,
                        prompt_ids,
                        spans,
                        deletion,
                        image_token,
                        mrope,
                    )
                    .map_err(ModelError::Msg)?;
                    // 已 positioned 的会话即使本轮纯追加也仍带位置洞；
                    // 标记必须跨 turn 保持（缓存安全门）。
                    gap |= s.positioned;
                    owl_shared::log_debug!(
                        "img-evict",
                        "位置账:session={sid:#x} old={} new={} deletion={} gap={} max_pos={}",
                        s.tokens.len(),
                        prompt_ids.len(),
                        deletion.is_some(),
                        gap,
                        s.max_pos
                    );
                    return Ok((pos, gap));
                }
            }
        }
        let (pos, _) = crate::img_evict::compact_positions(prompt_ids, spans, image_token, mrope)
            .map_err(ModelError::Msg)?;
        Ok((pos, false))
    }

    /// **页对齐 pad 词**(单源;`unpad` 按区间剥 ⇒ 内容对客户端不可见)。
    ///
    /// 只要"恰 1 token",顺序 = **模型最不敏感者优先**:`。`(正常标点)→ ` ` → `\n`。
    /// 依据(27B/TP2,keep=99 规整多图历史):`\n` 从第 3 张**标记**图起就让回答开头
    /// 吐 `< >`;换成 ` ` 后阈值推到 ≥7 张(见设计卷 §5duodecies)。与旋转/删除无关。
    pub(crate) fn image_pad_token(&self) -> Option<u32> {
        // 显式配置优先(boot 门已验"恰 1 token");空串 = 自动挑。
        let conf = self.boot.cfg.knobs.image_evict_pad.as_str();
        if !conf.is_empty() {
            return self
                .boot
                .tok
                .try_encode(conf)
                .ok()
                .filter(|v| v.len() == 1)
                .map(|v| v[0]);
        }
        for cand in ["。", " ", "\n"] {
            if let Ok(v) = self.boot.tok.try_encode(cand) {
                if v.len() == 1 {
                    return Some(v[0]);
                }
            }
        }
        None
    }

    /// **页对齐 pad 律 + 删除重建**(前缀无感删除;2026-10-23)。
    ///
    /// 三件事(纯 host、零设备 IO):
    /// 1. 若会话账里有上轮的图像段:把 `tokens` 剥掉 pad 得到"客户端可见序列",
    ///    在**文本坐标**里识别"这一轮正好删了一段已标记图像";
    /// 2. 命中 ⇒ **重建**:`old[..a]`(含旧 pad 块)++ 客户端删点之后的文本
    ///    —— 删点之前逐字节不变是**构造性保证**(纯 diff 做不到,见模块头注);
    /// 3. 对整条序列跑 pad 律(继承旧 pad 区间,幂等)。
    ///
    /// 未开档 ⇒ 只记 spans(网格/诊断用),**一个 pad 都不加**(逐字节旧行为)。
    /// 返回 `(ids, spans, pads, deletion)`;`deletion` 交给调度面做取链/拼接。
    fn image_pad_law(
        &self,
        session: Option<u64>,
        ids: Vec<u32>,
        images: &[crate::turn::TurnImage],
    ) -> Result<(
        Vec<u32>,
        Vec<crate::img_evict::ImgSpan>,
        Vec<(usize, usize)>,
        Option<crate::img_evict::TextDelete>,
    )> {
        // 本轮**一张图都没有**也要往下走:删掉历史里**最后一张**图时,新 prompt
        // 就是纯文本 —— 吸收路径仍必须能识别(否则"删最后一张"永远重算)。
        let Some(vlm) = self.dev.boot.vlm.as_ref() else {
            return if images.is_empty() {
                Ok((ids, Vec::new(), Vec::new(), None))
            } else {
                Err(ModelError::Msg(
                    "image_pad_law: 请求带图但本 rank 无视觉档(提交门应已拒)".into(),
                ))
            };
        };
        let image_token = vlm.cfg().image_token;
        let grids: Vec<owl_models::layers::vision::GridThw> =
            images.iter().map(|i| i.image.grid).collect();
        let marks: Vec<bool> = images.iter().map(|i| i.evict).collect();
        if images.is_empty() && !self.boot.cfg.knobs.image_evict {
            return Ok((ids, Vec::new(), Vec::new(), None));
        }
        let page = if self.boot.cfg.knobs.image_evict {
            self.dev.pool.geom.page
        } else {
            0
        };
        let one = |text: &str| -> Option<u32> {
            match self.boot.tok.try_encode(text) {
                Ok(v) if v.len() == 1 => Some(v[0]),
                _ => None,
            }
        };
        let (vs, ve, pad) = if page == 0 {
            (None, None, 0u32)
        } else {
            // **pad 词** = [`Self::image_pad_token`] 单源(恰 1 token、模型最不敏感者)。
            let pad = self.image_pad_token().ok_or_else(|| {
                ModelError::Msg(
                    "image_pad_law: 找不到\"恰 1 token\"的 pad 词(页对齐 pad 律要求)".into(),
                )
            })?;
            (one("<|vision_start|>"), one("<|vision_end|>"), pad)
        };

        // ── ① 删除识别 + 重建(只在"档位开 + 会话账里有上轮图像段") ──
        let mut deletion = None;
        let mut input = ids;
        let mut inherited: Vec<(usize, usize)> = Vec::new();
        if page > 0 {
            if let Some(sid) = session {
                if let Some(s) = self.bank.sessions.get(sid) {
                    if !s.tokens.is_empty() && !s.img_spans.is_empty() {
                        let old_text =
                            crate::img_evict::unpad(&s.tokens, &s.pads).map_err(ModelError::Msg)?;
                        let detect =
                            crate::img_evict::detect_text_delete(&old_text, &input, &s.img_spans);
                        owl_shared::log_debug!(
                            "img-evict",
                            "detect: session={sid:#x} old={} raw={} spans={:?} ⇒ {}",
                            old_text.len(),
                            input.len(),
                            s.img_spans
                                .iter()
                                .map(|x| (x.start, x.end, x.text_start, x.text_end, x.evict))
                                .collect::<Vec<_>>(),
                            detect
                                .map(|h| format!(
                                    "hit text=[{},{})=padded[{},{})",
                                    h.text_a, h.text_b, h.a, h.b
                                ))
                                .unwrap_or_else(|| "none".to_string())
                        );
                        if detect.is_none() {
                            let n = old_text.len().min(input.len());
                            let first_diff = (0..n).find(|&i| old_text[i] != input[i]);
                            if let Some(i) = first_diff {
                                let lo = i.saturating_sub(24);
                                let hi = (i + 40).min(n);
                                owl_shared::log_debug!(
                                    "img-evict",
                                    "first_diff={i} old_win={:?} raw_win={:?}",
                                    self.boot.tok.decode(&old_text[lo..hi]),
                                    self.boot.tok.decode(&input[lo..hi])
                                );
                            }
                            for sp in s.img_spans.iter().filter(|x| x.evict) {
                                let ta = sp.text_start;
                                let tb = sp.text_end;
                                let pre = ta <= old_text.len()
                                    && ta <= input.len()
                                    && old_text[..ta] == input[..ta];
                                let k = if tb <= old_text.len() && ta <= input.len() {
                                    old_text[tb..]
                                        .iter()
                                        .zip(&input[ta..])
                                        .take_while(|(x, y)| x == y)
                                        .count()
                                } else {
                                    0
                                };
                                let old_after: Vec<u32> = old_text
                                    .get(tb..tb.saturating_add(12))
                                    .unwrap_or(&[])
                                    .to_vec();
                                let raw_after: Vec<u32> =
                                    input.get(ta..ta.saturating_add(12)).unwrap_or(&[]).to_vec();
                                owl_shared::log_debug!(
                                    "img-evict",
                                    "detect miss: span text=[{ta},{tb}) padded=[{},{}) pre={pre} k={k} old_after={:?} raw_after={:?}",
                                    sp.start,
                                    sp.end,
                                    self.boot.tok.decode(&old_after),
                                    self.boot.tok.decode(&raw_after)
                                );
                            }
                        }
                        if detect.is_none() {
                            // **沿用上一轮 padded 前缀**(2026-10-26)。
                            //
                            // leave-gap 删除后,删除段前面的 `pad_before` 块
                            // 仍在前缀里、并且已经进了 KV;但对应图像已经不在。
                            // 下一轮若从裸文本重算 pad,这个"孤儿 pad 块"不会
                            // 被重新插入 ⇒ 前缀在删点前的 pad 块处分叉,缓存
                            // 退到删点之前(实测 turn36→37)。
                            //
                            // 客户端追加/只改尾部时,`old_text` 与 `input` 有
                            // 长公共前缀。把这段公共前缀映射成 padded 切点
                            // (吞掉切点后的 pad),再接客户端新尾部 ⇒ 历史 pad
                            // 全部保留,新尾部照常走 pad 律。
                            //
                            // 切点落在图像内部 / 客户端缩短历史 = 不做(交回
                            // 裸文本重算),避免把半个图像拼进来。
                            let common = old_text
                                .iter()
                                .zip(&input)
                                .take_while(|(a, b)| a == b)
                                .count();
                            let cuts_image = s
                                .img_spans
                                .iter()
                                .any(|sp| sp.text_start < common && common < sp.text_end);
                            if common > 0 && input.len() > common && !cuts_image {
                                let p = crate::img_evict::padded_prefix_for_raw(
                                    &s.tokens, &s.pads, common,
                                )
                                .map_err(ModelError::Msg)?;
                                let mut merged = s.tokens[..p].to_vec();
                                merged.extend_from_slice(&input[common..]);
                                // 拼接后图像游程数必须与当前请求的 marks 对齐;
                                // 不齐 = 切点切进了图像/账不平 ⇒ 放弃复用,走裸文本重算,
                                // 不许把半个图拼进来后让下游 silent 错位。
                                let mut runs = 0usize;
                                let mut prev = false;
                                for &t in &merged {
                                    let cur = t == image_token;
                                    if cur && !prev {
                                        runs += 1;
                                    }
                                    prev = cur;
                                }
                                if runs == marks.len() {
                                    input = merged;
                                    inherited =
                                        s.pads.iter().copied().filter(|(_, hi)| *hi <= p).collect();
                                    owl_shared::log_debug!(
                                        "img-evict",
                                        "沿用 padded 前缀:session={sid:#x} common={common} padded_cut={p} old={} new={} pads={}",
                                        s.tokens.len(),
                                        input.len(),
                                        inherited.len()
                                    );
                                }
                            }
                        }
                        if let Some(hit) = detect {
                            let imgs = s.img_spans.len();
                            let marked = s.img_spans.iter().filter(|x| x.evict).count();
                            owl_shared::log_info!(
                                "img-evict",
                                "删除识别: session={sid:#x} old={} raw={} images={imgs}(marked={marked}) \
                                 ⇒ 删 text=[{},{}), padded=[{},{}), {} tok/{} 页",
                                old_text.len(),
                                input.len(),
                                hit.text_a,
                                hit.text_b,
                                hit.a,
                                hit.b,
                                hit.b - hit.a,
                                (hit.b - hit.a) / page
                            );
                            let rebuilt =
                                crate::img_evict::rebuild_after_delete(&s.tokens, &input, &hit)
                                    .map_err(ModelError::Msg)?;
                            // 只继承**删点之前**的 pad 块(删点之后由本轮的 pad 律重算)
                            inherited = s
                                .pads
                                .iter()
                                .copied()
                                .filter(|(_, hi)| *hi <= hit.a)
                                .collect();
                            if hit.a % page != 0 {
                                owl_shared::log_warn!(
                                    "img-evict",
                                    "识别命中但删点 {} 不落页边界(页 {})⇒ 不吸收",
                                    hit.a,
                                    page
                                );
                            } else {
                                input = rebuilt;
                                deletion = Some(hit);
                            }
                        }
                    } else {
                        owl_shared::log_debug!(
                            "img-evict",
                            "detect skipped: session={sid:#x} tokens={} spans={}",
                            s.tokens.len(),
                            s.img_spans.len()
                        );
                    }
                } else {
                    owl_shared::log_debug!("img-evict", "detect skipped: session={sid:#x} 不在表");
                }
            } else {
                owl_shared::log_debug!("img-evict", "detect skipped: 无 session(ephemeral)");
            }
        }

        // ── ② pad 律(幂等;继承旧 pad 区间) ──
        let seq = crate::img_evict::pad_image_spans(
            &input,
            &marks,
            &grids,
            image_token,
            vs,
            ve,
            page,
            pad,
            &inherited,
        )
        .map_err(ModelError::Msg)?;
        Ok((seq.ids, seq.spans, seq.pads, deletion))
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
        let max_queue = self.boot.cfg.knobs.max_queue;
        if max_queue > 0 {
            let in_flight = self.turns.active.len() + self.turns.queue.len();
            let cap = self
                .boot
                .cfg
                .knobs
                .max_batch
                .max(1)
                .saturating_add(max_queue);
            if in_flight >= cap {
                // 退避建议（P3-c）：EWMA × 需要的波数（含自己）
                let eta = self
                    .turns
                    .eta_ms(self.boot.cfg.knobs.max_batch.max(1), self.turns.queue.len());
                self.turns.note_reject(session.unwrap_or(0), "no_slot");
                return Err(ModelError::Msg(format!(
                    "admission:no_slot retry_ms={eta} —— 在飞 {in_flight} 已达上限 {cap}\
                     （批位 {} + 队列 {max_queue}）",
                    self.boot.cfg.knobs.max_batch.max(1)
                )));
            }
        }
        // **同会话并发守卫**(B5.1):合批档下两个同会话 turn 会共享同一份
        // KV/GDN 状态 ⇒ 结构性互踩。亮拒(上层应按会话串行)。
        if self.boot.cfg.knobs.max_batch > 1 {
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
        // **VLM 输入门**(2026-10-20 视觉接线 V5;2026-10-23 多图):
        // 单请求可带多图(按内容序),但图像 × 合批仍拒;图像 × 投机已接
        // (verify 的 rope pos 加 m-RoPE delta,slots/lens 用真实位置)。
        // 邻接多图的 m-RoPE 由 `MropeCfg::rope_index` 按 grid 顺序分段。
        if !opts.images.is_empty() {
            if self.dev.boot.vlm.is_none() {
                return Err(ModelError::Msg(
                    "submit: 请求带图,但本档 [model].vision != true(引擎无视觉塔)".into(),
                ));
            }
            if self.boot.cfg.knobs.max_batch > 1 {
                return Err(ModelError::Msg(
                    "submit: VLM 图像首版不支持合批(max_batch > 1)".into(),
                ));
            }
            // **可删标记必须有人接**(前缀无感删除;2026-10-23):客户端打了
            // `owl_evict` 而引擎档没开 = 静默失约(用户以为删了不重算,
            // 实际每轮全量重灌)⇒ 提交边界**响亮拒**,不静默忽略。
            if !self.boot.cfg.knobs.image_evict {
                if opts.images.iter().any(|i| i.evict) {
                    return Err(ModelError::Msg(
                        "submit: 图像带 evict(可删)标记,但本档 [pool] image_evict = false\
                         —— 要么把档位打开,要么去掉标记(不许静默忽略)"
                            .into(),
                    ));
                }
            }
        }
        let raw_ids = {
            // 诊断开关(knobs.raw_completion):裸续写,绕过 chat 模板 ——
            // 模型健康度鉴别(模板态病 vs 权重病)的 A/B 臂
            // §三十二:含 im_start = 上层已渲染完整 chat 模板(OpenAI 多轮
            // 面),跳过单轮 chat_wrap(整段历史塞单 user 消息 = 多轮崩根因)
            let wrapped = if self.boot.cfg.knobs.raw_completion || prompt.contains("<|im_start|>") {
                prompt.clone()
            } else {
                self.boot.tok.chat_wrap(&prompt)
            };
            // 诊断:最终 prompt 落盘([probes] prompt_dump;§三十六 多轮"无记忆"
            // 幻觉定位面 —— dump 渲染+wrap 后送模型的完整 prompt 字节)
            if self.boot.probes.prompt_dump {
                let ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    // 系统时钟回拨兜底 → 0(非失败路径;仅用于 dump 文件名)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                let path = crate::probe_dump::probe_path(&format!("owl-prompt-{ms}.txt"));
                match std::fs::write(&path, &wrapped) {
                    Ok(_) => owl_shared::log_debug!("prompt-dump", "{path} ({}B)", wrapped.len()),
                    Err(e) => owl_shared::log_debug!("prompt-dump", "落盘失败 {e}"),
                }
            }
            self.boot.tok.encode(&wrapped)
        };
        // **前缀无感删除的页对齐 pad 律**(2026-10-23):只对**带标记**的图像段
        // 做(且只在 `[pool] image_evict` 开时)—— 未标记/未开档 = 逐字节旧行为。
        // 产出的 `spans` 随 turn 走到收口,写回会话账(下一轮删除识别的唯一输入)。
        let (prompt_ids, img_spans, img_pads, deletion) =
            self.image_pad_law(session, raw_ids, &opts.images)?;
        if prompt_ids.is_empty() {
            return Err(ModelError::Msg("submit: 空 prompt".into()));
        }
        let (positions, positions_gap) =
            self.prompt_positions(session, &prompt_ids, &img_spans, deletion)?;
        // **消息边界快照点**(2026-10-24):页对齐 token offset，供
        // `prefill_step_chunk_msg` 切块与 `capture_gdn_snap` 的 snap_keep
        // 共用。`open_ids` 来自配置/模型声明，不在扫描函数里写死。
        let msg_snaps = if self.boot.cfg.knobs.snap_msg_boundary
            && self.boot.cfg.knobs.prefix_cache
            && self.dev.pool.geom.paged
            && self.dev.pool.snap_capacity() > 0
        {
            crate::msg_boundary::message_snap_points(
                &prompt_ids,
                &self.boot.msg_open_ids,
                self.dev.pool.geom.page,
                self.boot.cfg.knobs.snap_msg_min_gap,
            )
        } else {
            Vec::new()
        };
        // 准入地板(S3b §23.5 / A7-bis 接线 2026-10-19):
        //
        // 非 Strict 地板 = "允许选择层按需物化" ⇒ **必须有执行面**才收:
        // 判据 = 引擎的**虚拟化两扇门**(`select_decode` + `select_partial`,
        // 唯一产地 `kvmem_virtualization`)。没开就响亮拒(接受=静默降级)。
        // 开着的档位:地板进 `admission_gate`(准入判据 + 实得地板回吐),
        // 页数上限在准入处按**本档工作集**再校一遍(ctx 长度那时才知道)。
        if let Some(f) = opts.dense_floor {
            if !f.is_strict() && !self.boot.cfg.knobs.kvmem_virtualization {
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
            let probe = session.and_then(|id| {
                self.bank
                    .sessions
                    .get(id)
                    .map(|s| (id, s.cached_len, s.guard(&prompt_ids)))
            });
            match probe {
                Some((_, cached, true)) => cached,
                // **守卫失配取证**(`[probes] debug`;§4.3):报首个分叉的**位置 +
                // 两侧 id 窗口**。渲染对称律的判据就是这一行:分叉在
                // "assistant 轮边界" = 渲染/切分不对称;分叉在别处 = 客户端回填不同。
                Some((id, cached, false)) => {
                    if self.boot.probes.debug {
                        let s = self.bank.sessions.get(id).expect("刚查过");
                        match s.first_divergence(&prompt_ids) {
                            Some(i) => {
                                let lo = i;
                                let hi_old = (i + 8).min(s.tokens.len());
                                let hi_new = (i + 8).min(prompt_ids.len());
                                let (a, b) = (&s.tokens[lo..hi_old], &prompt_ids[lo..hi_new]);
                                owl_shared::log_debug!(
                                    "prefix",
                                    "守卫失配 session={id} cached_len={cached} 首个分叉 idx={i} \
                                     旧(已入 KV)={a:?} ⇒ {:?} ‖ 新(本轮)={b:?} ⇒ {:?}",
                                    self.boot.tok.decode(a),
                                    self.boot.tok.decode(b)
                                );
                            }
                            None => owl_shared::log_debug!(
                                "prefix",
                                "守卫失配 session={id} cached_len={cached} 但前缀全同\
                                 (长度不足?prompt={})",
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
        let max_new = match budget_clamp(base, max_new, self.boot.cfg.max_seq_tokens) {
            Ok(m) => {
                if m != max_new {
                    owl_shared::log_warn!(
                        "chat",
                        "预算收缩: prompt {} + 生成 {} > max_seq {} ⇒ 生成长度收缩到 {} tokens",
                        prompt_ids.len(),
                        max_new,
                        self.boot.cfg.max_seq_tokens,
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
                    self.boot.cfg.max_seq_tokens
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
            spec: TurnSpec {
                prompt,
                max_new,
                ignore_eos: opts.ignore_eos,
            },
            prompt_ids,
            msg_snaps,
            img_spans,
            img_pads,
            positions,
            positions_gap,
            deletion,
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
            .or_else(|| {
                self.turns
                    .active
                    .iter()
                    .find(|a| a.id == turn)
                    .map(|a| a.max_new)
            })
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
    pub(crate) fn set_dev_ctx(
        &mut self,
        owner: owl_iface::handle::Owner,
        class: owl_iface::handle::AssetClass,
    ) {
        for dev in std::iter::once(&mut self.dev).chain(self.peer.as_mut()) {
            let f = dev.graphs.dev_ctx_face_mut();
            f.set_owner(owner);
            f.set_class(class);
        }
    }

    /// **store 唯一入口**（三语义钩子；2026-10-25 Phase 1）。
    ///
    /// 纪律：engine **生产位置**不直接调 `plan_need` / `plan_onload` / `snap_*`；
    /// 先构造 [`StoreHook`]，由本函数统一走 `plan → apply` 与记账。
    /// 旧 `Node` 四变体已删除，调用点收敛为：
    /// - `Acquire::Capacity`（准入/增长）→ `pool_housekeeping`；
    /// - `Acquire::Load`（前缀回灌）→ `store_load`；
    /// - `Acquire::State`（选槽/就位）→ `store_state_slot` / `store_state_resident`；
    /// - `Release::{TurnEnd,Abort}` → `store_release`（H3/H6/N4）；
    /// - `Maintain::{Idle,Pressure,HostPressure}` → `store_maintain`。
    pub(crate) async fn store_hook(&mut self, hook: StoreHook) -> Result<StoreOutcome> {
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add(&format!("store.hook.{}", hook.label()), 1, file!(), line!());
        });
        match hook {
            StoreHook::Acquire(a) => self.store_acquire(a).await,
            StoreHook::Release(r) => self.store_release(r).await,
            StoreHook::Maintain(m) => self.store_maintain(m).await,
        }
    }

    async fn store_acquire(&mut self, acquire: Acquire) -> Result<StoreOutcome> {
        match acquire {
            Acquire::Capacity { need, reserve } => {
                match need.why {
                    owl_store::NeedWhy::Admit | owl_store::NeedWhy::Grow => {}
                    other => {
                        return Err(ModelError::EngineInvariant {
                            what: format!(
                                "store_hook: Acquire::Capacity 只接 Admit/Grow，收到 {other:?}"
                            ),
                        });
                    }
                }
                if reserve.is_some() && need.why != owl_store::NeedWhy::Admit {
                    return Err(ModelError::EngineInvariant {
                        what: "store_hook: 只有 Admit 能新开 Grant".into(),
                    });
                }
                let applied = self.pool_housekeeping(need, &[], None).await?;
                let grant = match reserve {
                    Some(r) => Some(self.bank.rel.reserve_capacity(
                        r.turn,
                        r.slots,
                        r.alloc_slots,
                        r.kind,
                    )?),
                    None => None,
                };
                Ok(StoreOutcome {
                    applied,
                    grant,
                    ..StoreOutcome::default()
                })
            }
            Acquire::Load(req) => self.store_load(req).await,
            Acquire::State(StateAcquire::Resident { key }) => self.store_state_resident(key).await,
            Acquire::State(StateAcquire::Slot { key }) => {
                // capture 路径：本 turn 的旧边界会被新边界替代，允许它成为 victim；
                // 其它活跃 turn 的最新边界仍受保护。
                let skip_turn = self.turns.active().ok().map(|a| a.id);
                let slot = self.store_state_slot(key, &[], skip_turn).await?;
                Ok(StoreOutcome {
                    state_slot: Some(slot),
                    ..StoreOutcome::default()
                })
            }
        }
    }

    /// **状态腿就位**（原 H7 三段直呼）：Gpu 直接命中；Host/Disk 先找槽再回灌。
    /// 返回的槽号经 `StoreOutcome.state_slot` 交给调用方。
    async fn store_state_resident(&mut self, key: owl_store::SnapKey) -> Result<StoreOutcome> {
        let slot = match self.bank.rel.state.locate(key) {
            Some(owl_store::Tier::Gpu) => {
                let slot = self.dev.pool.snap_slot_of(key).ok_or_else(|| {
                    ModelError::Msg(format!("快照 {key:?} 账在设备层但池里找不到槽(账目不平)"))
                })?;
                // 使用即触达(store 出序的真源)
                self.bank.rel.state.touch(key);
                slot
            }
            Some(owl_store::Tier::Host) => {
                let slot = self.store_state_slot(key, &[key], None).await?;
                self.snap_onload(key, slot).await?;
                slot
            }
            Some(owl_store::Tier::Disk) => {
                // **S6**:盘 → 宿主 → 设备(两跳);读回后层账回 Host,再走 Host 分支
                self.fill_snap_from_disk(key).await?;
                let slot = self.store_state_slot(key, &[key], None).await?;
                self.snap_onload(key, slot).await?;
                slot
            }
            None => return Err(ModelError::Msg(format!("快照 {key:?} 不在账(已被逐出)"))),
        };
        Ok(StoreOutcome {
            state_slot: Some(slot),
            ..StoreOutcome::default()
        })
    }

    /// **快照槽选择**（原 `spec::snap_ensure_slot`）：池满时先问 store 的
    /// 同一张容量票（`plan_capacity`），把“谁走、去宿主还是磁盘”的判据
    /// 收回收敛点；池只做 key→槽物理映射。
    ///
    /// 为什么要循环：设备快照不能直连磁盘（层级律要求 GPU → Host → Disk）。
    /// 宿主满时第一张票可能只把**宿主里的冷快照**挪到盘（腾宿主位），
    /// 设备槽要下一拍才空出来；因此最多重试 3 次（搬盘 → 设备下沉 → 取槽）。
    async fn store_state_slot(
        &mut self,
        key: owl_store::SnapKey,
        extra_snaps: &[owl_store::SnapKey],
        skip_turn: Option<u64>,
    ) -> Result<u32> {
        use crate::state::SnapSlotPick;
        // **Snap 票**：快照设备槽也进同一本 ReservationLedger（turn 生命周期）。
        // 无活跃 turn（测试/纯 store 路径）⇒ 不发；正常 capture/restore 都有 turn。
        if let Ok(act) = self.turns.active() {
            let grant = self.bank.rel.reserve_snap(act.id)?;
            self.bank
                .rel
                .consume_capacity(grant.id, 1, self.bank.rel.snap_bytes())?;
        }
        // **P1-2**：出序与容量票必须看到同一份保护面。
        let protect = self.snap_guard(extra_snaps, skip_turn);
        for _ in 0..3 {
            // **判据 = store(唯一产地;§30.6-②)**:出序(事实档 + EvictPolicy)从
            // `Rel::snap_victim_order` 拿;池只做 key→槽映射。
            let order = self.bank.rel.snap_victim_order(&protect);
            let pick = self.dev.pool.snap_pick_slot(key, &order).ok_or_else(|| {
                ModelError::Msg("GDN 快照池为空(snap_max = 0):调用方应跳过 capture".into())
            })?;
            match pick {
                SnapSlotPick::Hit(slot) | SnapSlotPick::Free(slot) => return Ok(slot),
                SnapSlotPick::Victim(slot, victim) => {
                    // 池内 LRU 兜底可能越过 store 序；保护面是最后一道闸。
                    if protect.contains(&victim) {
                        return Err(ModelError::Msg(format!(
                            "快照槽选择把受保护的 {victim:?} 当 victim（池序回退越过保护面）"
                        )));
                    }
                    // **P0-2**:不再走“只搬 host、失败就 drop”的旁路；
                    // 用状态腿的同一张票（snap_bytes = 一条快照），
                    // host 满时会出 Disk 票，盘层因此可达。
                    let snap_bytes = self.bank.rel.snap_bytes();
                    if snap_bytes > 0 {
                        let applied = self
                            .pool_housekeeping(
                                owl_store::Need {
                                    slots: 0,
                                    snap_bytes,
                                    host_kv_slots: 0,
                                    host_snap_slots: 0,
                                    why: owl_store::NeedWhy::Grow,
                                },
                                extra_snaps,
                                skip_turn,
                            )
                            .await?;
                        let progress =
                            applied.snap_offloaded + applied.snap_spilled + applied.snap_dropped;
                        if progress > 0 {
                            continue;
                        }
                    }

                    // 最后兜底（store 判据无票可出 / 宿主+盘都关）：
                    // 丢最冷一条，保证本次 capture 能拿到槽；不静默。
                    owl_shared::log_debug!(
                        "store",
                        "快照槽满且 store 无票可出 ⇒ 丢弃最冷快照 {victim:?}(该边界将不可复用)"
                    );
                    self.bank.rel.drop_snap_host(victim)?;
                    if let Err(e) = self.bank.rel.state.drop_snap(victim) {
                        owl_shared::log_debug!("store", "快照 {victim:?} 出账: {e}");
                    }
                    owl_shared::metrics::with_metrics_store(|m| {
                        m.counter_add("store.snap.drop", 1, file!(), line!());
                    });
                    self.dev.pool.snap_clear_slot(slot);
                    if let Some(pr) = self.peer.as_mut() {
                        pr.pool.snap_clear_slot(slot);
                    }
                    return Ok(slot);
                }
            }
        }
        Err(ModelError::Msg(
            "快照槽选择重试 3 次仍无空位（store 票未兑现；检查 host/disk 预算与快照账）".into(),
        ))
    }

    /// **前缀命中回灌**（原 H4 双旁路）：槽不足先问 store，再一条命令搬 N 块，
    /// 最后把新槽写回会话块表。
    async fn store_load(&mut self, req: LoadRequest) -> Result<StoreOutcome> {
        if req.blocks.is_empty() {
            return Ok(StoreOutcome::default());
        }
        let t_stage = std::time::Instant::now();
        let chains: Vec<owl_store::ChainHash> =
            req.blocks.iter().map(|(_, chain)| *chain).collect();
        // 盘上的块先读回宿主（盘 → 设备要两跳；判据与账都在 store）
        self.ensure_host_resident(&chains).await?;
        let need_slots = chains.len();
        let block_bytes = self.dev.pool.block_bytes();
        let load_bytes = need_slots
            .checked_mul(block_bytes)
            .ok_or_else(|| ModelError::Msg("Load Grant: 字节溢出".into()))?;
        // **Load / Temp 票**（同一本 ReservationLedger）：
        // - Load = 回灌块占用的设备槽；consume 标记已兑现；
        // - Temp = 本次回灌的周转额度；turn 收尾统一归还。
        // 两者都不进 ActiveFloor 的 KV 准入面（`Rel::reserved_slots` 按 kind 过滤）。
        let load_grant = self.bank.rel.reserve_load(req.turn, need_slots)?;
        self.bank
            .rel
            .consume_capacity(load_grant.id, need_slots, load_bytes)?;
        let temp_grant = self
            .bank
            .rel
            .reserve_temp(req.turn, need_slots, load_bytes)?;
        self.bank
            .rel
            .consume_capacity(temp_grant.id, need_slots, load_bytes)?;
        let mut applied = owl_store::Applied::default();
        // 设备槽先腾够：`Need.slots` 是需求，`plan_capacity` 会按当前 free 折算缺口。
        if self.bank.rel.leg.slots_free() < need_slots {
            applied = self
                .pool_housekeeping(
                    owl_store::Need::kv(need_slots, owl_store::NeedWhy::Grow),
                    &[],
                    None,
                )
                .await?;
        }
        let (cs, new_slots, pieces, table) = {
            let p = owl_store::plan_onload(&mut self.bank.rel.leg, &chains)?;
            (p.chains, p.slots, p.pieces, p.table)
        };
        let table = match self
            .swap_all_ranks(owl_iface::contract::SwapDir::In, &pieces, table)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                owl_store::abort_onload(&mut self.bank.rel.leg, &cs, &new_slots);
                return Err(e);
            }
        };
        let slots = owl_store::commit_onload(&mut self.bank.rel.leg, &cs, &new_slots, table)?;
        let loaded_blocks = cs.len();
        let moved_bytes = loaded_blocks * self.dev.pool.block_bytes();
        let s = self
            .bank
            .sessions
            .get_mut(req.session)
            .ok_or_else(|| ModelError::Msg(format!("回灌:会话 {} 不在账", req.session)))?;
        for ((idx, _), slot) in req.blocks.iter().zip(&slots) {
            if *idx >= s.block_table.len() {
                return Err(ModelError::Msg(format!(
                    "回灌下标 {idx} 越界(表长 {})",
                    s.block_table.len()
                )));
            }
            s.block_table[*idx] = *slot;
        }
        self.sel.note_stage_in(
            loaded_blocks,
            moved_bytes,
            t_stage.elapsed().as_nanos() as u64,
            crate::select::StageSrc::Prefill,
        );
        Ok(StoreOutcome {
            applied,
            loaded_blocks,
            ..StoreOutcome::default()
        })
    }

    /// **生命周期收尾**（Phase 3）：H3 链登记/盘索引 + H6 表/引用归零 +
    /// GDN 行归还 + 归属下行 + N4 清点/force release/trim。
    async fn store_release(&mut self, release: Release) -> Result<StoreOutcome> {
        let turn = release.turn;
        // **P0-4：Release 归还本 turn 的全部 Grant**（幂等；准入收口也会再调一次）。
        let released = self.bank.rel.release_turn_grants(turn);
        if !released.is_empty() {
            owl_shared::log_debug!(
                "store",
                "Release 归还 Grant：turn {turn} {} 张 / {} 槽",
                released.len(),
                released.iter().map(|g| g.slots).sum::<usize>()
            );
        }
        // ── H3：正常 turn 收尾才发布链；abort 的半途链不进缓存 ----------
        if release.kind == ReleaseKind::TurnEnd {
            if self.dev.pool.geom.paged {
                if let Some(s) = self.bank.sessions.get(release.session) {
                    let up = self.bank.rel.leg.register_chain_owned(
                        &s.tokens,
                        &s.block_table,
                        release.session,
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
            // **S6.1：盘上链索引**（内容可能还没溢写 ⇒ boot 按内容截断）。
            // 放在 ephemeral 焚毁之前 —— 会话 tokens 此刻还在。
            if self.dev.pool.geom.paged && self.bank.rel.disk_enabled() {
                let toks = self
                    .bank
                    .sessions
                    .get(release.session)
                    .map(|s| s.tokens.clone())
                    .unwrap_or_default();
                if !toks.is_empty() {
                    self.write_chain_index(&toks).await?;
                }
            }
            // **索引暂存排空**：正常 prefill 的末边界已清空；这里兜底
            // abort/短轮 —— 必须在 `release_turn_table` 之前采，物理页还在。
            if self.boot.cfg.knobs.index_probe {
                let n_pending = self.sel.pending_index_len();
                let pending = self.sel.take_index_pending();
                if !pending.is_empty() {
                    owl_shared::log_debug!("idx", "turn 收口排空索引暂存:{n_pending} 块");
                    self.collect_index_for_blocks(&pending).await?;
                }
            }
        }

        // ── H6：表/引用归零（ephemeral = 焚毁；否则只释放设备表） ----------
        if release.ephemeral {
            self.burn_ephemeral(release.session);
        } else {
            let r = self.release_turn_table(release.session);
            match (release.kind, r) {
                (_, Ok(())) => {}
                // 正常收尾：表释放失败 = turn 失败（当前 complete 语义）
                (ReleaseKind::TurnEnd, Err(e)) => return Err(e),
                // 弃置：只 WARN，不能让一次表释放失败挡住后面的 GDN/N4 收尾
                (ReleaseKind::Abort, Err(e)) => {
                    owl_shared::log_warn!("session", "弃置 turn 释放设备表失败: {e}");
                }
            }
        }

        // ── GDN 行归还（行 = turn 域资源；失败只 WARN） --------------------
        if let Err(e) = self.bank.rel.state.return_row(release.gdn_slot, None) {
            owl_shared::log_warn!("session", "turn {} 还 GDN 行失败: {e}", release.turn);
        }

        // ── 归属下行（会话仍活 ⇒ Session(s)，焚毁 ⇒ System） ---------------
        self.set_ctx_after_turn(release.session, release.ephemeral);

        // ── N4：池清点 + 未归还清单 + 兜底释放 + trim ---------------------
        self.n4_pool_sweep(turn);
        self.n4_report_unreturned(turn);
        let force = self.n4_force_release(turn).await;
        match (release.kind, force) {
            (_, Ok(())) => {}
            (ReleaseKind::TurnEnd, Err(e)) => return Err(e),
            (ReleaseKind::Abort, Err(e)) => {
                owl_shared::log_warn!("store", "N4/Release(Abort) 失败(继续收尾): {e}");
                return Ok(StoreOutcome::default());
            }
        }
        // **池缓存回收**（2026-10-20 §23.9）：`RELEASE_THRESHOLD=∞` ⇒ turn 内
        // 的峰值缓存**永不**还 OS（peer 侧实测 3.3 GiB 长期挂着，长 prefill
        // 因此提前撞 `vram-hard-budget`）。turn 收尾清一次：块账不变，
        // 只把"已 free 的缓存页"还给驱动。
        let freed = self.trim_pools().await;
        // 让**紧随其后的 Idle 出口**再清一次：turn 收尾路径里还有一批
        // 大额临时块是在 TurnEnd 之后才 free 的（实测 peer 3.0 GiB）。
        self.hk.expire_trim();
        for (rank, r) in freed {
            match r {
                Ok(0) => {}
                Ok(n) => owl_shared::log_info!(
                    "store",
                    "池缓存回收:turn {turn} rank{rank} 释放 {:.1} MiB(已 free 的池页还给驱动)",
                    n as f64 / 1048576.0
                ),
                // 收尾路径的可失败调用：如实 WARN（不静默；也不升级为 turn 失败 ——
                // 显存多占一点不影响正确性，下一 turn 的压力旁路还会再试）
                Err(e) => {
                    owl_shared::log_warn!("store", "池缓存回收失败(turn {turn} rank{rank}): {e}")
                }
            }
        }
        Ok(StoreOutcome::default())
    }

    /// **后场维护**：Idle 的状态腿目标由宿主余位补；Pressure 只带显式 snap_bytes。
    async fn store_maintain(&mut self, maintain: Maintain) -> Result<StoreOutcome> {
        let need = match maintain {
            Maintain::Idle(slots) => {
                // 状态腿目标 = 一条快照（宿主有位才要；宿主关/满时给 0，
                // 不产生永远凑不够的 deny）
                let snap_bytes = if self.bank.rel.snap_host_free_slots() > 0 {
                    self.bank.rel.snap_bytes()
                } else {
                    0
                };
                owl_store::Need {
                    slots,
                    snap_bytes,
                    host_kv_slots: 0,
                    host_snap_slots: 0,
                    why: owl_store::NeedWhy::Idle,
                }
            }
            Maintain::Pressure { slots, snap_bytes } => owl_store::Need {
                slots,
                snap_bytes,
                host_kv_slots: 0,
                host_snap_slots: 0,
                why: owl_store::NeedWhy::Pressure,
            },
            Maintain::HostPressure { kv_slots, snap_slots } => {
                owl_store::Need::host(kv_slots, snap_slots)
            }
        };
        let applied = self.pool_housekeeping(need, &[], None).await?;
        Ok(StoreOutcome {
            applied,
            ..StoreOutcome::default()
        })
    }

    /// **压力缓解口**（actor 在 `PoolExhausted` 后调）。
    ///
    /// 次序：**先还缓存**（复用池排空 + 池缓存 trim —— 真还驱动），
    /// **先腾宿主**（`Maintain::HostPressure`：只出 Disk 票），
    /// **再做池内收缩**（`Maintain::Pressure`：KV 槽 + 状态腿目标）。
    /// 返回 `(池排空块数, 目标腾出 KV 槽数, 目标腾出 KV 宿主块数, 目标腾出快照数)`；失败响亮上抛
    /// （调用方只记日志继续）。
    ///
    /// 为什么不再用“驱动 free < 1GiB”触发：固定池把块沉宿主**不增加驱动 free**，
    /// 却会制造每拍 churn。`pump` 不再跑 N6；只有真正 `PoolExhausted` 才收缩。
    pub async fn relieve_pressure(&mut self) -> Result<(usize, usize, usize, usize)> {
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
        // 池缓存 pages 真还驱动（不增池槽，但能缓解驱动硬墙）
        let trimmed = self.trim_pools().await;
        let mut trim_bytes = 0u64;
        for (rank, r) in trimmed {
            match r {
                Ok(n) => trim_bytes += n as u64,
                Err(e) => {
                    owl_shared::log_warn!("store", "压力缓解:池缓存回收失败(rank{rank}): {e}")
                }
            }
        }
        if trim_bytes > 0 {
            owl_shared::log_warn!(
                "store",
                "压力缓解:池缓存回收 {:.1} MiB",
                trim_bytes as f64 / 1048576.0
            );
        }
        let signals = self.pressure_signals();
        if signals.driver_below_reserve() {
            owl_shared::log_warn!(
                "store",
                "压力信号:驱动 free 已低于预留（{}）；先排空/回收，池收缩仍按池水位",
                signals.summary()
            );
        }
        let (slots, snap_bytes, host_kv, host_snap) = self.pressure_need(&signals);
        if host_kv > 0 || host_snap > 0 {
            // **先腾宿主**：宿主腿满时若先压池，轮一会把设备块继续往宿主塞。
            owl_shared::log_warn!(
                "store",
                "压力缓解目标:宿主 {host_kv} KV 块 + {host_snap} 快照 → 磁盘（跳过 GPU→Host）; {}",
                signals.summary()
            );
            self.store_hook(StoreHook::host_pressure(host_kv, host_snap))
                .await?;
            self.hk.mark_host_pressure();
        }
        if slots > 0 || snap_bytes > 0 {
            owl_shared::log_warn!(
                "store",
                "压力缓解目标:KV {slots} 槽 / 状态腿 {:.1} MiB（低水位触发 → 高水位目标）; {}",
                snap_bytes as f64 / 1048576.0,
                signals.summary()
            );
            self.store_hook(StoreHook::pressure_with_snap(slots, snap_bytes))
                .await?;
        }
        Ok((freed, slots, host_kv, host_snap))
    }

    /// **压力信号快照**（P1-3；纯 host 账，零 I/O）。
    fn pressure_signals(&self) -> PressureSignals {
        let leg = &self.bank.rel.leg;
        let mut driver_free = self.dev.graphs.dev_ctx_face().vram_snapshot().free;
        if let Some(p) = self.peer.as_ref() {
            driver_free = driver_free.min(p.graphs.dev_ctx_face().vram_snapshot().free);
        }
        PressureSignals {
            pool_free_pages: leg.slots_free(),
            pool_total_pages: leg.slots_total(),
            host_free_slots: leg.host_free_slots() + self.bank.rel.snap_host_free_slots(),
            host_kv_enabled: leg.host_enabled(),
            host_snap_enabled: self.bank.rel.snap_host_budget() > 0,
            host_kv_total_slots: leg.host_total_slots(),
            host_snap_total_slots: self.bank.rel.snap_host_total_slots(),
            host_kv_free_slots: leg.host_free_slots(),
            host_snap_free_slots: self.bank.rel.snap_host_free_slots(),
            disk_free_slots: self
                .bank
                .rel
                .disk
                .as_ref()
                .map(|d| d.free_slots())
                .unwrap_or(0)
                + self
                    .bank
                    .rel
                    .snap_disk
                    .as_ref()
                    .map(|d| d.free_slots())
                    .unwrap_or(0),
            disk_kv_free_slots: self
                .bank
                .rel
                .disk
                .as_ref()
                .map(|d| d.free_slots())
                .unwrap_or(0),
            disk_snap_free_slots: self
                .bank
                .rel
                .snap_disk
                .as_ref()
                .map(|d| d.free_slots())
                .unwrap_or(0),
            driver_free_bytes: driver_free,
            reserve_bytes: self.boot.cfg.knobs.vram_reserve_mb << 20,
        }
    }

    fn pressure_policy(&self) -> PressurePolicy {
        PressurePolicy::new(
            self.boot.cfg.knobs.pressure_low_pct,
            self.boot.cfg.knobs.pressure_high_pct,
            self.boot.cfg.knobs.host_pressure_low_pct,
            self.boot.cfg.knobs.host_pressure_high_pct,
        )
    }

    /// HostPressure 每请求目标（KV 前缀块 / 快照条数；分腿裁剪）。
    fn request_host_pressure(&self, signals: &PressureSignals) -> (usize, usize) {
        let (planned_kv, planned_snap) = host_pressure_targets(signals, &self.pressure_policy());
        clamp_host_pressure(
            planned_kv,
            planned_snap,
            self.boot.cfg.knobs.host_pressure_request_slots,
            self.boot.cfg.knobs.host_pressure_request_snaps,
        )
    }

    /// **紧急池压力目标**（`PoolExhausted` 后；不是每拍 N6）。
    ///
    /// 判据 = **压力信号 + 低/高双水位**：自由页低于 low 才触发，目标补到 high
    /// （滞回，避免阈值附近 churn）。状态腿在宿主有位时带一条快照。
    fn pressure_need(&self, signals: &PressureSignals) -> (usize, usize, usize, usize) {
        let slots = self.pressure_policy().target_slots(
            signals.pool_total_pages,
            signals.pool_free_pages,
            PRESSURE_MAX_SLOTS,
        );
        let snap_bytes = if slots > 0
            && signals.host_free_slots > 0
            && self.bank.rel.state.stats().snap_gpu.0 > 0
        {
            self.bank.rel.snap_bytes()
        } else {
            0
        };
        let (host_kv, host_snap) = self.request_host_pressure(signals);
        (slots, snap_bytes, host_kv, host_snap)
    }

    /// **H5 空转整理节拍**（Idle pump 计数；每 N 拍问一次）。
    ///
    /// **默认不预搬**：`runtime.idle_compact=false` 时只让 `pump` 走池缓存
    /// 回收（`trim_pools`），不触发 `Maintain::Idle`。显式开启后也只按
    /// 压力信号的低水位触发、补到高水位，不再用固定 `pool/8` 目标。
    fn idle_tick(&mut self) -> Option<(usize, usize, usize)> {
        if !self.hk.note_idle_pump() {
            return None;
        }
        if !self.boot.cfg.knobs.idle_compact {
            return None;
        }
        let signals = self.pressure_signals();
        let slots = self.pressure_policy().target_slots(
            signals.pool_total_pages,
            signals.pool_free_pages,
            IDLE_COMPACT_MAX_SLOTS,
        );
        let (mut host_kv, mut host_snap) = self.request_host_pressure(&signals);
        if host_kv > 0 || host_snap > 0 {
            let interval = std::time::Duration::from_millis(
                self.boot.cfg.knobs.host_pressure_interval_ms as u64,
            );
            if self.hk.host_pressure_due(interval) {
                self.hk.mark_host_pressure();
            } else {
                host_kv = 0;
                host_snap = 0;
            }
        }
        (slots > 0 || host_kv > 0 || host_snap > 0).then_some((slots, host_kv, host_snap))
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

    /// **设备显存实况**(只读;`/debug/vram` 的驱动残差):`(free, total)`。
    pub async fn mem_info(&mut self) -> std::result::Result<(u64, u64), String> {
        self.dev
            .graphs
            .session
            .face_mut()
            .mem_info()
            .await
            .map_err(|e| e.to_string())
    }

    /// **默认池缓存回收(逐 rank)** —— turn 收尾 / 压力旁路调。
    ///
    /// 语义:`cuMemPoolTrimTo(0)` 把池内**已 free 的缓存页**还给 OS(不改块账)。
    /// 返回逐 rank 的 `(rank, Ok(释放字节)/Err(原因))` —— 失败**如实带出**
    /// (调用方决定是否响亮;本函数不吞)。
    pub async fn trim_pools(&mut self) -> Vec<(u8, std::result::Result<u64, String>)> {
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
    ) -> Vec<(
        u8,
        std::result::Result<Vec<owl_iface::contract::BlockLedgerRow>, String>,
    )> {
        #[allow(clippy::type_complexity)]
        let mut out: Vec<(
            u8,
            std::result::Result<Vec<owl_iface::contract::BlockLedgerRow>, String>,
        )> = Vec::new();
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

    /// 当前活跃 turn 的容量票（P0-4；exec 子模块的块表增长入口用）。
    pub(crate) fn grant_of_session(&self, session: u64) -> Result<owl_store::ReservationId> {
        self.turns
            .active
            .iter()
            .find(|a| a.session_id == session)
            .map(|a| a.grant)
            .ok_or_else(|| ModelError::EngineInvariant {
                what: format!("会话 {session:#x} 无活跃 turn 的 Grant"),
            })
    }

    pub(crate) fn set_phase(&mut self, p: owl_models::env::EvalPhase) {
        self.dev.set_phase(p);
        self.dev.graphs.set_phase(p);
    }

    /// **本步的设备槽需求**(§12.4 需求驱动保洁的**真源**)。
    ///
    /// 口径 = 与 `schedule()` 里 `ensure_for_len` 的 target **同一份算式**
    /// (块长走 [`crate::scheduler::RunningEngine::prefill_step_chunk`] 单源):
    /// - 队列里有待准入 turn ⇒ 目标 = 该 prompt 全长(准入那一刻就要覆盖);
    /// - 活跃 turn 在 prefill ⇒ 目标 = `fed + chunk`;
    /// - 活跃 turn 在 decode ⇒ 目标 = `pos + 本档 spec 轮总增长上界`
    ///   (裸 decode / MTP / DFlash2 的差别见 [`Self::generation_target`];
    ///   **上界是安全方向**:多算只是多腾几个槽(内容不丢),少算会在
    ///   纯分配 `ensure_for_len` 处响亮拒);
    /// - 无活跃且队列空 ⇒ **零需求**(§12.4:不用固定水位预搬;
    ///   预搬是显式策略,留给将来 预热/B 轴)。
    ///
    /// 返回的是"**还要长多少槽**"(已扣当前表长),不是总量。
    ///
    /// 准入分支要查一次命中面(纯 host、与 `schedule` 同一条匹配路):
    /// Δ_j = `ceil(N_j/page) − 设备已驻留的命中块数`(§12.1)—— **host 命中的
    /// 块不算省槽**(回灌时照样占槽,少算就是回灌 Exhausted)。
    // H1(Admit) 与 H2(Grow) 共用一个 Need；两层判决是**有意保留**的：
    // - `scheduler::admission.decide` = 批量准入策略（会话守卫/槽位/生成预算/退避）；
    // - `admission_gate` 的 `Rel::check` = 单 turn VRAM 地板（dense floor/EvictThen）。
    // 两者共享 store facts，但判据不同；合并会把预算内部暴露给策略状态机。
    //
    // **Phase 2**:生成相位的目标必须覆盖**本 pump 内 spec 轮的 propose**,
    // 不能只算 verify 的 `pos+depth+1`;否则 ⑦ 轮末 propose 的 `ensure_for_len`
    // 会在纯分配处撞墙。
    fn step_need(&mut self) -> Result<owl_store::Need> {
        use owl_store::NeedWhy;
        let page = self.dev.pool.geom.page.max(1);
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
                let have = self
                    .bank
                    .sessions
                    .get(sid)
                    .map(|s| s.block_table.len())
                    .unwrap_or(0);
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
            let target = match Self::prefill_step_chunk_msg(
                act.fed,
                act.prompt_ids.len(),
                self.boot.cfg.prefill_chunk,
                self.boot.cfg.knobs.prefill_tail_merge,
                Self::prefill_align_page(
                    self.boot.cfg.knobs.prefill_align_tail,
                    self.boot.cfg.knobs.prefix_cache,
                    self.dev.pool.geom.page,
                ),
                &act.msg_snaps,
                self.boot.cfg.knobs.snap_msg_min_span,
            ) {
                Some(chunk) => act.fed + chunk,
                None => self.generation_target(act.fed), // 生成相位(act.fed = pos)
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

    /// **生成相位一个 pump 步内可能长到的最远 token 位**（Phase 2 预留上界）。
    ///
    /// 一个 spec 轮会连续做两段增长：
    /// 1. verify 行 `pos..pos+depth`（scheduler 已按 `pos+spec+1` 增长）；
    /// 2. 轮末 propose 的草稿链（⑦，发生在 `act.fed = pos+m+1` 之后）。
    ///
    /// 因此 `step_need` 必须按**本轮最终位**预留，而不是只按 verify 行：
    /// - 裸 decode / Dumb：`pos + depth + 1`；
    /// - MTP：`pos + m + depth`（propose 链），`m ≤ depth` ⇒ `pos + 2·depth`；
    /// - DFlash2：`pos + m + 1 + (depth+1)`（`fp + block`），`m ≤ depth`
    ///   ⇒ `pos + 2·depth + 2`。
    ///
    /// 这里用配置的最大 `depth`（自适应降档只会更小），宁多勿少。
    fn generation_target(&self, pos: usize) -> usize {
        let kind = match self.dev.spec_rt.drafter.as_ref() {
            Some(crate::spec_runtime::Drafter::DFlash2(_)) => GenerationKind::DFlash2,
            Some(crate::spec_runtime::Drafter::Mtp(_)) => GenerationKind::Mtp,
            None => GenerationKind::None,
        };
        generation_target_for(pos, self.dev.spec_rt.depth, kind)
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
        let loans =
            owl_store::book_lock(&cell).open_loans(|o| o == owl_iface::handle::Owner::Turn(turn));
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
        // **噪声闸门**(2026-10-27):小量残块(≤4 块且 ≤64 KiB)降为 debug。
        //
        // 依据不是"看着烦",而是**它们不会累积**:`n4_force_release` 紧随其后
        // 用同一份账兜底结清(`drain_owner` 与 `open_loans` 同源),且实测
        // 13 个 turn 的池水位稳定(used 21.1~22.8 GiB 波动 = 正常周转)。
        // 典型样本 = 每个 turn 1 块 `owl_narrow_strided_f16` / 10240 B
        // (GDN `project` 的 z 窄切中间量)。真正的泄漏形态是"每轮 +N 且 N 大"
        // —— 那条仍然响亮 WARN(阈值之上的老口径一字未改)。
        let small = open.len() <= 4 && total_kib <= 64;
        if small {
            owl_shared::log_debug!(
                "store",
                "N4 残块(兜底将清):turn {turn} {} 块 / {total_kib} KiB —— {}",
                open.len(),
                detail.join(" ｜ ")
            );
            return;
        }
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
            m.counter_add(
                &format!("store.{tag}.stale_blocks"),
                total.0 as u64,
                file!(),
                line!(),
            );
            m.counter_add(
                &format!("store.{tag}.stale_bytes"),
                total.1 as u64,
                file!(),
                line!(),
            );
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
    /// 真源 = 块腿报告的设备驻留字节 + 活跃 turn 的块表(**按槽号去重**)。
    ///
    /// 恒等式(必须成立,否则判据就偏):
    /// `reserved(唯一活跃实占) + session_held + cache = 池实占`。
    /// 共享前缀出现在多个 turn 的块表里时,保留面只计一次。
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
        // B5.1:多活跃 ⇒ 逐 turn 记块表快照;共享块的槽号相同,
        // `Admission::reserved_bytes` 取并集(共享前缀只算一次)。
        let active_slots: Vec<(u64, Vec<u32>)> = self
            .turns
            .actives()
            .map(|a| {
                let slots: Vec<u32> = self
                    .bank
                    .sessions
                    .get(a.session_id)
                    .map(|s| {
                        s.block_table
                            .iter()
                            .copied()
                            .filter(|b| !owl_store::is_host_placeholder(*b))
                            .collect()
                    })
                    .unwrap_or_default();
                (a.id, slots)
            })
            .collect();
        let ad = &mut self.bank.rel.admission;
        ad.set_cache_bytes(movable); // 可动面(可下沉/可驱逐)
        for (id, slots) in &active_slots {
            ad.set_active_slots(*id, slots.iter().copied(), block_bytes)?;
        }
        let held = ad.reserved_bytes();
        ad.set_session_held(occupied.saturating_sub(movable).saturating_sub(held)); // 会话持有面
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
            if floor.is_strict() {
                "Strict"
            } else {
                "耐受"
            }
        );
    }

    /// **H1 准入判决**(S3b §23.4):待准入 turn 过 `Rel::check` ⇒
    /// 调需求(EvictThen)/ 拒(Reject)/ 留队(Queue)。
    ///
    /// 为什么先于 `ensure_for_len`:那条路是"**占了槽**才发现不够"(Exhausted,
    /// 中后段才炸,且已写半截账);本判决是"**判在前**"——不够就先腾,
    /// 腾不够就带数字拒。
    // H1 的地板层（有意保留）：scheduler 的 `turns.admission.decide` 管批量策略，
    // 本函数只管单 turn VRAM 地板；两层共享 store facts，不合并。
    fn admission_gate(&mut self, need: &mut owl_store::Need) -> Result<()> {
        use owl_store::AdmissionVerdict;
        if need.why != owl_store::NeedWhy::Admit {
            return Ok(());
        }
        // **只在真的能起 turn 时判**(无活跃 turn)。有活跃 turn 时队首只是在
        // 等位 —— 此刻"池被活跃面占着"是常态(它马上会退成缓存),照判会
        // 把正常排队的第二个请求误拒。准入 = "谁能**开始**"。
        if self.turns.active.len() >= self.boot.cfg.knobs.max_batch.max(1) {
            return Ok(());
        }
        self.sync_pool_account()?;
        let Some((turn_id, ctx_tokens, floor)) = self.turns.queue.front().map(|t| {
            (
                t.id,
                // 上界 = prompt + 生成长度 + spec 深度(安全方向:多算只是多腾几个槽)
                t.prompt_ids.len() + t.spec.max_new + self.dev.spec_rt.depth,
                t.opts
                    .dense_floor
                    .unwrap_or(self.boot.cfg.knobs.dense_floor),
            )
        }) else {
            return Ok(());
        };
        // **per-turn 地板的上限校验**(A7-bis):视图只能保持"本档工作集"那么多页
        // 常驻 ⇒ 请求地板 > 工作集 = 承诺无法兑现 ⇒ **响亮拒**(不静默夹）。
        // 只有 ctx 长度到了这里才知道,所以校在这里而不是 submit 处。
        let page = self.dev.pool.geom.page.max(1);
        let page_bytes = self.dev.pool.block_bytes().max(1);
        let ws_pages = self.boot.cfg.knobs.select_blocks.max(1);
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
            AdmissionVerdict::EvictThen {
                need_bytes,
                floor_bytes,
            } => {
                // 抬需求到"腾够就能进":同一次段边界搬完(下沉不丢内容)
                let want = need_bytes.div_ceil(self.dev.pool.block_bytes().max(1));
                need.slots = need.slots.max(want);
                self.log_floor_grant(turn_id, floor, ctx_tokens, page, page_bytes, floor_bytes);
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
                if waited >= self.boot.cfg.knobs.admission_max_wait_ms() {
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
        self.turns.pending_events.push_back(TurnEvent::Failed {
            turn: turn_id,
            err: why,
        });
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
        if self
            .bank
            .rel
            .disk
            .as_ref()
            .is_some_and(|d| d.contains(chain))
        {
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
        match crate::disk_io::write_block(plan.path.clone(), plan.tmp.clone(), buf.into_bytes())
            .await
        {
            Ok(_) => {
                self.bank
                    .rel
                    .disk
                    .as_mut()
                    .expect("上面判过")
                    .commit_spill(&plan)?;
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.spill", 1, file!(), line!());
                });
                Ok(true)
            }
            Err(f) => {
                // 内容所有权被带回 ⇒ 放回宿主(溢写失败 ≠ 丢内容)
                self.bank
                    .rel
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
            let r = self
                .bank
                .rel
                .disk
                .as_ref()
                .expect("调用方判过")
                .plan_spill(chain);
            match r {
                Ok(p) => return Ok(Some(p)),
                Err(owl_store::StoreError::Exhausted(_)) if attempt == 0 => {
                    let need = self.bank.rel.disk.as_ref().expect("判过").block_bytes();
                    let evp = self
                        .bank
                        .rel
                        .disk
                        .as_ref()
                        .expect("判过")
                        .plan_evict(need, &[]);
                    if evp.victims.is_empty() {
                        return Ok(None);
                    }
                    let paths: Vec<std::path::PathBuf> =
                        evp.victims.iter().map(|(_, p)| p.clone()).collect();
                    let keys: Vec<owl_store::ChainHash> =
                        evp.victims.iter().map(|(k, _)| *k).collect();
                    let n = crate::disk_io::remove_files(paths).await;
                    let freed = self
                        .bank
                        .rel
                        .disk
                        .as_mut()
                        .expect("判过")
                        .commit_evict(&keys);
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
        let page = self.dev.pool.geom.page.max(1);
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
                owl_shared::log_warn!(
                    "store",
                    "链索引写盘失败:{}(索引缺失 = 该链跨重启不可复用)",
                    f.err
                );
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
        let Some(d) = self.bank.rel.disk.as_ref() else {
            return Ok(false);
        };
        if !d.contains(chain) {
            return Ok(false);
        }
        if self.bank.rel.leg.host_contains(chain) {
            let _ = self.bank.rel.disk.as_mut().expect("判过").touch(chain);
            return Ok(false); // 已在宿主(盘副本仍在)
        }
        let plan = self
            .bank
            .rel
            .disk
            .as_ref()
            .expect("判过")
            .plan_fill(chain)?;
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
        self.bank
            .rel
            .disk
            .as_mut()
            .expect("判过")
            .commit_fill(chain)?;
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
        if self
            .bank
            .rel
            .snap_disk
            .as_ref()
            .is_some_and(|d| d.contains(key))
        {
            return Ok(false);
        }
        let buf = match self.bank.rel.snap_host.take(key) {
            Ok(b) => b,
            Err(owl_store::StoreError::Unknown(_)) => return Ok(false),
            Err(e) => return Err(e.into()),
        };
        let mut plan = None;
        for attempt in 0..2 {
            let r = self
                .bank
                .rel
                .snap_disk
                .as_ref()
                .expect("判过")
                .plan_spill(key);
            match r {
                Ok(p) => {
                    plan = Some(p);
                    break;
                }
                Err(owl_store::StoreError::Exhausted(_)) if attempt == 0 => {
                    let need = self
                        .bank
                        .rel
                        .snap_disk
                        .as_ref()
                        .expect("判过")
                        .block_bytes();
                    let evp = self
                        .bank
                        .rel
                        .snap_disk
                        .as_ref()
                        .expect("判过")
                        .plan_evict(need, &[]);
                    if evp.victims.is_empty() {
                        break;
                    }
                    let paths: Vec<std::path::PathBuf> =
                        evp.victims.iter().map(|(_, p)| p.clone()).collect();
                    let keys: Vec<owl_store::SnapKey> =
                        evp.victims.iter().map(|(k, _)| *k).collect();
                    crate::disk_io::remove_files(paths).await;
                    self.bank
                        .rel
                        .snap_disk
                        .as_mut()
                        .expect("判过")
                        .commit_evict(&keys);
                }
                Err(_) => break,
            }
        }
        let Some(plan) = plan else {
            self.bank.rel.snap_host.put(key, buf)?;
            return Ok(false);
        };
        match crate::disk_io::write_block(plan.path.clone(), plan.tmp.clone(), buf.into_bytes())
            .await
        {
            Ok(_) => {
                self.bank
                    .rel
                    .snap_disk
                    .as_mut()
                    .expect("判过")
                    .commit_spill(&plan)?;
                self.bank.rel.state.set_tier(key, owl_store::Tier::Disk)?;
                owl_shared::metrics::with_metrics_store(|m| {
                    m.counter_add("store.disk.snap.spill", 1, file!(), line!());
                });
                Ok(true)
            }
            Err(f) => {
                self.bank
                    .rel
                    .snap_host
                    .put(key, owl_store::HostBlock::from_bytes(f.bytes))?;
                owl_shared::log_warn!("store", "快照溢写失败(内容已回宿主):{}", f.err);
                Ok(false)
            }
        }
    }

    /// **快照回填**(磁盘 → 宿主;S6)。
    pub(crate) async fn fill_snap_from_disk(&mut self, key: owl_store::SnapKey) -> Result<bool> {
        let Some(d) = self.bank.rel.snap_disk.as_ref() else {
            return Ok(false);
        };
        if !d.contains(key) {
            return Ok(false);
        }
        if self.bank.rel.snap_host.get(key).is_some() {
            let _ = self.bank.rel.snap_disk.as_mut().expect("判过").touch(key);
            return Ok(false);
        }
        let plan = self
            .bank
            .rel
            .snap_disk
            .as_ref()
            .expect("判过")
            .plan_fill(key)?;
        let bytes = crate::disk_io::read_block(plan.path.clone(), plan.bytes)
            .await
            .map_err(|e| owl_iface::contract::ModelError::Msg(e))?;
        self.bank
            .rel
            .snap_host
            .put(key, owl_store::HostBlock::from_bytes(bytes))?;
        self.bank
            .rel
            .snap_disk
            .as_mut()
            .expect("判过")
            .commit_fill(key)?;
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
                    Some(pr) => {
                        pr.graphs
                            .session
                            .face_mut()
                            .swap(dir, pcs, Box::new(table))
                            .await
                    }
                    None => {
                        return Err(owl_iface::contract::ModelError::EngineInvariant {
                            what: format!(
                                "宿主腿 swap({dir:?}): world = {world} 但 peer 束缺席(装配不一致)"
                            ),
                        });
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

    /// **本档能否走分离式下沉**(单卡 + 后端自述支持)。
    ///
    /// 先探能力再动 `SlotTable`:表格是单主零锁的,一旦交给 `swap_begin`
    /// 就只能等回执拿回 —— 没有"试一下不行就换回去"这条路。
    pub(crate) fn can_defer_out(&self, rank_groups: usize) -> bool {
        self.bank.rel.world().max(1) == 1
            && rank_groups == 1
            && self.dev.graphs.session.face_ref().supports_swap_begin()
    }

    /// **发一条分离式 stage-out**(2026-10-27;调用前必须 `can_defer_out`)。
    pub(crate) fn swap_out_begin(
        &mut self,
        pieces: &[Vec<owl_iface::contract::SwapPiece>],
        table: owl_store::SlotTable,
    ) -> Result<owl_iface::contract::SwapWaiter> {
        let w = self.dev.graphs.session.face_mut().swap_begin(
            owl_iface::contract::SwapDir::Out,
            &pieces[0],
            Box::new(table),
        )?;
        Ok(w)
    }

    /// **boot 后预热 pinned staging 池**(2026-10-27)。
    ///
    /// 预算 = 池总量(`slots_total × block_bytes`):选择层一次下沉最多搬
    /// "池内全部页"(64k 档实测 1881 MiB),而不预热的代价是
    /// `cudaHostAlloc(1.8 GiB) ≈ 600 ms` **占着 server 命令线程** —— 当时排队的
    /// 视图页表写/下一个 chunk 入队全部干等(实测 `select.view` 965 ms)。
    /// 稳态发生第一次越池时再付这笔 = 单次 TTFT 多 0.6 s。
    ///
    /// 只在"部分驻留档"开(其它档没有下沉腿);默认无操作面也会返回 (0,0)
    /// —— 日志把数报出来,不静默。
    pub(crate) async fn prewarm_pinned_staging(&mut self) -> Result<()> {
        if !self.boot.cfg.knobs.select_partial {
            return Ok(());
        }
        let st = self.bank.rel.leg.report();
        let bytes = st.slots_total.saturating_mul(st.block_bytes).min(4usize << 30);
        if bytes == 0 {
            return Ok(());
        }
        let (made, have) = self.dev.graphs.session.face_mut().prewarm_pinned(bytes).await?;
        owl_shared::log_warn!(
            "store",
            "pinned staging 预热:目标 {} MiB → 新建 {made} 桶 / 已有 {have} 桶",
            bytes >> 20
        );
        Ok(())
    }

    /// **收口在飞的下沉**(2026-10-27)。
    ///
    /// 调用点必须落在"下一个 chunk 已入队、GPU 正在算"的窗口:这里 await 的
    /// 是 D2H + 宿主拷贝 + 首次 pinned 分配,实测 1.1 s/次 —— 让它与计算**并行**
    /// 才是收益来源(而不是把等待搬到别处)。
    ///
    /// 失败 = 整条失败(与同步路径同语义):`abort_offload` 回滚块票 + 会话表
    /// 恢复成原槽 —— **不许留下"表说在宿主、实际在设备"的半态**。
    pub(crate) async fn drain_pending_out(&mut self) -> Result<()> {
        let Some(p) = self.pending_out.take() else { return Ok(()) };
        // 在飞时长(诊断面):它应该 ≈ 一个 chunk 的计算时间(完全被盖住),
        // 而不是被白白串行掉。
        let inflight_ms = p.since.elapsed().as_secs_f64() * 1000.0;
        owl_shared::metrics::with_metrics_store(|m| {
            m.timer_record_tag(
                "select.rec.out_inflight",
                std::time::Duration::from_secs_f64(inflight_ms / 1000.0),
                file!(),
                line!(),
            );
        });
        let mut table: Option<owl_store::SlotTable> = None;
        for w in p.waiters {
            match w.await {
                Ok(conn) => table = Some(owl_store::reclaim_slot_table(conn)?),
                Err(e) => {
                    let pairs: Vec<(owl_store::ChainHash, u32)> =
                        p.blocks.iter().map(|(_, h, s)| (*h, *s)).collect();
                    owl_store::abort_offload(&mut self.bank.rel.leg, &pairs);
                    if let Some(s) = self.bank.sessions.get_mut(p.sid) {
                        for (i, _, slot) in &p.blocks {
                            if let Some(cell) = s.block_table.get_mut(*i) {
                                *cell = *slot;
                            }
                        }
                    }
                    return Err(owl_iface::contract::ModelError::Msg(format!(
                        "在飞下沉失败(已回滚 {} 块): {e}",
                        p.blocks.len()
                    )));
                }
            }
        }
        let Some(t) = table else { return Ok(()) };
        let pairs: Vec<(owl_store::ChainHash, u32)> =
            p.blocks.iter().map(|(_, h, s)| (*h, *s)).collect();
        let n = owl_store::commit_offload(&mut self.bank.rel.leg, &pairs, t)?;
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("select.rec.out_committed", n as u64, file!(), line!());
        });
        Ok(())
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
    async fn apply_verdict(&mut self, verdict: &owl_store::Verdict) -> Result<owl_store::Applied> {
        use owl_store::Sink;
        let mut a = owl_store::Applied::default();
        // ① 换出/驱逐(**两腿同单**:KV 链/叶 + GDN 快照;§30)
        for item in &verdict.out {
            match item {
                owl_store::EvictOut::Kv { chain, sink, span } => match sink {
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
                                m.counter_add("store.host.relief_blocks", relief, file!(), line!());
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
                    // **整枝塌缩**:一张票可带 `span` 块(叶→根,到分叉点为止);
                    // 逐块校验引用(活跃面停),丢掉的块数如实记账。
                    Sink::Drop => match self.bank.rel.leg.prune_branch(*chain, (*span).max(1)) {
                        Ok(n) => a.dropped += n,
                        Err(owl_store::StoreError::Busy(_)) => a.skipped_busy += 1,
                        Err(e) => return Err(e.into()),
                    },
                    // **第三层(S6)**:宿主 → 磁盘(内容不丢)。
                    // 票是**链级**的(`chain` = 链尾,S6.1):整链里所有**在宿主**的块
                    // 逐块写盘 —— 只挪尾部会拆散链,重启重建不出从根连续的前缀。
                    Sink::Disk => {
                        // `span` = 从根起的前缀块数(HostPressure 可只搬一段;
                        // 普通整链票的 span = 链长)。盘上仍是 root 连续前缀。
                        let blocks = self.bank.rel.leg.chain_blocks(*chain);
                        let mut n = 0usize;
                        for b in blocks.into_iter().take((*span).max(1)) {
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
        let mut moved_blocks = 0usize;
        let mut moved_bytes = 0usize;
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
            moved_blocks += cs.len();
            moved_bytes += cs.len() * self.dev.pool.block_bytes();
            a.onloaded += 1;
        }
        self.sel.note_stage_in(
            moved_blocks,
            moved_bytes,
            t_stage.elapsed().as_nanos() as u64,
            crate::select::StageSrc::Verdict,
        );
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
    /// 但 `out` 里能搬的照搬(下沉不丢内容);纯分配不足会在
    /// `ensure_for_len` 处响亮拒（Phase 2 后不再有同步驱逐）。
    // 唯一调用者 = `store_hook`（Acquire::Capacity / Load / Maintain）。
    // Phase 3 后 spec `snap_*` / H3/H6 均已收口；仅剩 coverage 诊断直呼。
    /// **活跃 turn 的显式快照保护面**(P1-2)。
    ///
    /// 快照面没有引用计数(不像 KV 由 `SlotPool.refs` 兜底),因此必须由
    /// engine 把「本 turn 最新边界」与「正在回滚/就位的 key」显式交给 store。
    ///
    /// `skip_turn` 只给 capture 路径:新边界会替代本 turn 的旧边界,旧边界
    /// 可以成为 victim;其它活跃 turn 的最新边界仍然受保护。
    fn snap_guard(
        &self,
        extra: &[owl_store::SnapKey],
        skip_turn: Option<u64>,
    ) -> Vec<owl_store::SnapKey> {
        snap_guard_keys(
            self.turns.active.iter().map(|a| (a.id, a.last_snap)),
            extra,
            skip_turn,
        )
    }

    async fn pool_housekeeping(
        &mut self,
        need: owl_store::Need,
        extra_snaps: &[owl_store::SnapKey],
        skip_turn: Option<u64>,
    ) -> Result<owl_store::Applied> {
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
            return Ok(owl_store::Applied::default());
        }
        let protect = owl_store::Protected {
            snaps: self.snap_guard(extra_snaps, skip_turn),
            ..owl_store::Protected::default()
        };
        let verdict = self.bank.rel.plan_need(need, &protect);
        if verdict.is_noop() {
            return Ok(owl_store::Applied::default());
        }
        let (n_out, n_in) = (verdict.out.len(), verdict.in_.len());
        if let Some(why) = &verdict.deny {
            // 凑不够 = 真容量不足预警:照搬能搬的(下沉不丢内容),响亮留痕。
            // **不致命**(缺口 3):记为一次 `degrade`(服务面/报告可查),由调用方
            // (准入/调度)决定收缩还是降级;真正的设备级错才走关停。
            //
            // **日志限流**(2026-10-19;压力节点每拍都跑 ⇒ 不限流必刷屏):
            // 首条 + 每 64 条打一次,带累计数;计数本身逐次入 metrics,不丢。
            let n = self.hk.note_degrade();
            if n == 1 || n.is_multiple_of(64) {
                owl_shared::log_warn!("store", "钩子:需求凑不够(真容量不足预警,累计 {n} 次):{why}");
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
            m.counter_add(
                "store.hook.snap.out",
                applied.snap_offloaded as u64,
                file!(),
                line!(),
            );
            m.counter_add(
                "store.hook.snap.drop",
                applied.snap_dropped as u64,
                file!(),
                line!(),
            );
            m.counter_add("store.hook.spill", applied.spilled as u64, file!(), line!());
            m.counter_add(
                "store.hook.snap.spill",
                applied.snap_spilled as u64,
                file!(),
                line!(),
            );
            m.counter_add(
                "store.hook.busy",
                applied.skipped_busy as u64,
                file!(),
                line!(),
            );
        });
        Ok(applied)
    }

    pub async fn pump(&mut self) -> Result<TurnEvent> {
        // E5-M2:spec 轮多 token 事件队列优先排空(schedule 前 —— 事件序
        // 不可与下一轮调度交织)
        if let Some(ev) = self.turns.pending_events.pop_front() {
            return Ok(ev);
        }
        let prof = self.boot.cfg.knobs.probes.step_profile;
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
        // 保洁**先于** `schedule` 里的 `ensure_for_len`；Phase 2 起后者只纯分配，
        // 不再有同步驱逐。`ops.ensure_evicted` 是回归 canary（应恒 0）。
        // P0.3 定谳(2026-10-19):**循环头**耗时(需求驱动保洁:need/gate/housekeeping)
        // —— 它是"步墙钟 − 设备 − fill/launch − 采样"的缺口候选(此前未打点)。
        let t_loophead = (prof || true).then(std::time::Instant::now);
        let need = self.step_need()?;
        let mut need = need;
        self.admission_gate(&mut need)?;
        // `step_need` 在"无活跃 + 队列空"时返回 NeedWhy::Idle（零需求）。
        // 这不是 Acquire::Capacity 的合法 kind；Idle 的整理由下方
        // H5 `idle_tick` / Maintain::{Idle,HostPressure} 负责，不能在这里误报不变量。
        // Phase 4 已删除 pump 的 N6（驱动 free 触发只制造 churn）；
        // 压力整理只在 PoolExhausted 后走 `relieve_pressure`。
        if !matches!(need.why, owl_store::NeedWhy::Idle) {
            self.store_hook(StoreHook::capacity(need)).await?;
        }
        let loophead_ms = t_loophead
            .map(|t| t.elapsed().as_secs_f64() * 1e3)
            .unwrap_or(0.0);
        let t_lh1 = prof.then(std::time::Instant::now); // 循环头之后
        owl_shared::timer_start!("engine.pump.schedule");
        let t_sched = (prof || true).then(std::time::Instant::now);
        let out = self.schedule().await?;
        let t_sc1 = prof.then(std::time::Instant::now); // schedule 之后
        if let Some(ts) = t_sched {
            let d = ts.elapsed();
            if let Some((n, avg_ms, last_ms)) = self.hk.note_sched(d) {
                owl_shared::log_info!(
                    "sched",
                    "n={} avg={:.2}ms last={:.2}ms loophead={:.3}ms",
                    n,
                    avg_ms,
                    last_ms,
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
                owl_shared::metrics::query_metrics(
                    &owl_shared::metrics::MetricsFilter::new().tag_prefix("engine.pump."),
                );
            }
        }
        match out {
            SchedulerOutput::Idle => {
                // **空转整理**(H5;`runtime.idle_compact` 缺省 **false**)。
                // false = 只做下方池缓存回收，**不预搬**；true = 每 N 拍按低水位
                // 触发、补到高水位（决策仍全在 store；无活跃 turn ⇒ 只动缓存面）。
                // **池缓存回收(空转出口;§23.9)**:server 的泵循环 `Idle` 即断,
                // `Maintain::Idle`(显式 compact 时才每 16 拍)到不了 ⇒ 在这里按时间节流清一次。
                // `RELEASE_THRESHOLD=∞` 下 turn 收尾后 free 的 GB 级临时块
                // 不主动还驱动就永远占着(实测 peer 侧 3.0 GiB)。
                let due = self.hk.trim_due();
                if due {
                    self.hk.mark_trim();
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
                    let total: u64 = freed.iter().filter_map(|(_, r)| r.as_ref().ok()).sum();
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
                            owl_shared::log_warn!("store", "空转池缓存回收失败(rank{rank}): {e}");
                        }
                    }
                }
                if let Some((slots, host_kv, host_snap)) = self.idle_tick() {
                    if host_kv > 0 || host_snap > 0 {
                        self.store_hook(StoreHook::host_pressure(host_kv, host_snap))
                            .await?;
                    }
                    if slots > 0 {
                        self.store_hook(StoreHook::idle(slots)).await?;
                    }
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
                    StepAction::Prefill {
                        chunk_ids,
                        base,
                        is_last,
                    } => {
                        // 性能面:相位定相 + 包络 span(perf-framework.md §四)。
                        // 引擎 actor = current_thread 运行时 ⇒ 单线程栈配对安全;
                        // 解释器节点钩子按 env.phase 打 tag,两者同源。
                        let saved = self.dev.phase();
                        self.set_phase(owl_models::env::EvalPhase::Prefill);
                        owl_shared::timer_start!("engine.phase.prefill");
                        let r = self.execute_prefill(chunk_ids, base, is_last).await;
                        owl_shared::timer_end!("engine.phase.prefill");
                        self.set_phase(saved);
                        r
                    }
                    StepAction::Decode {
                        token,
                        pos,
                        kv_slot,
                        gdn_slot,
                        grew,
                    } => {
                        let saved = self.dev.phase();
                        self.set_phase(owl_models::env::EvalPhase::Decode);
                        owl_shared::timer_start!("engine.phase.decode");
                        let t_ex0 = t_ph0.map(|_| std::time::Instant::now());
                        let r = self
                            .execute_decode(token, pos, kv_slot, gdn_slot, grew)
                            .await;
                        if let (Some(p0), Some(ex0)) = (t_ph0, t_ex0) {
                            let head = ex0.duration_since(p0);
                            let exec = ex0.elapsed();
                            let lh = t_lh1.map(|t| t.duration_since(p0)).unwrap_or_default();
                            let sc = t_sc1
                                .map(|t| t.duration_since(t_lh1.unwrap_or(t)))
                                .unwrap_or_default();
                            let tail_head =
                                t_sc1.map(|t| ex0.duration_since(t)).unwrap_or_default();
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
                        let saved = self.dev.phase();
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
                    StepAction::SpecRound {
                        token,
                        pos,
                        kv_slots,
                        gdn_slot,
                        grew,
                        depth,
                    } => {
                        let saved = self.dev.phase();
                        self.set_phase(owl_models::env::EvalPhase::Spec);
                        owl_shared::timer_start!("engine.phase.spec");
                        let r = self
                            .execute_spec_round(token, pos, kv_slots, gdn_slot, grew, depth)
                            .await;
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
                    return Err(ModelError::Msg(format!("generate: turn {id} 队列丢失")));
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
        assert_eq!(
            concat_head_rows(&a, &b, 3).unwrap(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
    }

    /// 多行（verify 形状）：每行各自拼接，不串行
    #[test]
    fn rows_concat_independently() {
        // rows = 2, half = 2
        let a = [1.0, 2.0, 3.0, 4.0]; // 行0=[1,2] 行1=[3,4]
        let b = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(
            concat_head_rows(&a, &b, 2).unwrap(),
            vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0]
        );
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
        assert_eq!(
            budget_clamp(32768 - _MIN_GEN_TOKENS, 8192, 32768),
            Ok(_MIN_GEN_TOKENS)
        );
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

#[cfg(test)]
mod generation_target_tests {
    use super::{generation_target_for, GenerationKind};

    /// Phase 2 的预留上界必须覆盖“verify + 轮末 propose”两段增长；
    /// 这些数字与 `draft.rs` 的 `fp/block`、`pos+m+depth` 同源。
    #[test]
    fn pins_spec_round_growth_upper_bounds() {
        let pos = 100usize;
        let d = 7usize;
        assert_eq!(
            generation_target_for(pos, 0, GenerationKind::DFlash2),
            pos + 1,
            "depth=0 ⇒ 只有当前 token"
        );
        assert_eq!(
            generation_target_for(pos, d, GenerationKind::None),
            pos + d + 1,
            "裸 decode / Dumb = verify 行"
        );
        assert_eq!(
            generation_target_for(pos, d, GenerationKind::Mtp),
            pos + 2 * d,
            "MTP = verify + propose pos+m+d"
        );
        assert_eq!(
            generation_target_for(pos, d, GenerationKind::DFlash2),
            pos + 2 * d + 2,
            "DFlash2 = fp+block，m≤d"
        );
    }
}

/// **P1-2 快照保护面合并**（纯函数；测试与 `snap_guard` 共用）。
///
/// `active` = 活跃 turn 的 `(turn, last_snap)`；`extra` = 本次操作点名的
/// key；`skip_turn` = capture 时允许被新边界替代的本 turn。
fn snap_guard_keys<I>(
    active: I,
    extra: &[owl_store::SnapKey],
    skip_turn: Option<u64>,
) -> Vec<owl_store::SnapKey>
where
    I: IntoIterator<Item = (u64, Option<owl_store::SnapKey>)>,
{
    let mut out: Vec<owl_store::SnapKey> = Vec::new();
    for (turn, key) in active {
        if Some(turn) == skip_turn {
            continue;
        }
        if let Some(key) = key {
            if !out.contains(&key) {
                out.push(key);
            }
        }
    }
    for key in extra {
        if !out.contains(key) {
            out.push(*key);
        }
    }
    out
}

#[cfg(test)]
mod snap_guard_tests {
    use super::snap_guard_keys;
    use owl_store::{ChainHash, SnapKey};

    fn key(n: u8) -> SnapKey {
        SnapKey::at_boundary(ChainHash::new(n as u64))
    }

    #[test]
    fn keeps_active_latest_and_restore_key_without_duplicates() {
        let (a, b, restore) = (key(1), key(2), key(3));
        assert_eq!(
            snap_guard_keys([(10, Some(a)), (20, Some(b))], &[restore], None),
            vec![a, b, restore]
        );
        assert_eq!(
            snap_guard_keys([(10, Some(a)), (20, Some(a))], &[], None),
            vec![a],
            "同一 key 在多个 turn 上只保留一次"
        );
        assert_eq!(
            snap_guard_keys([(10, Some(a)), (20, Some(a))], &[a], None),
            vec![a],
            "extra 与活跃面重复时不得双计"
        );
    }

    #[test]
    fn capture_skips_only_its_own_previous_boundary() {
        let (own, other, new_key) = (key(1), key(2), key(3));
        assert_eq!(
            snap_guard_keys([(10, Some(own)), (20, Some(other))], &[new_key], Some(10)),
            vec![other, new_key],
            "本 turn 旧边界可被新 capture 替代；其它 turn 与本次目标仍受保护"
        );
    }
}

/// 宿主腿低于低水位且盘有位 ⇒ 把该腿补回**高水位**（未裁剪的期望值）。
fn host_pressure_targets(
    signals: &PressureSignals,
    policy: &PressurePolicy,
) -> (usize, usize) {
    let kv = if signals.host_kv_enabled
        && signals.host_kv_free_slots < policy.host_low_slots(signals.host_kv_total_slots)
        && signals.disk_kv_free_slots > 0
    {
        policy
            .host_high_slots(signals.host_kv_total_slots)
            .saturating_sub(signals.host_kv_free_slots)
    } else {
        0
    };
    let snap = if signals.host_snap_enabled
        && signals.host_snap_free_slots < policy.host_low_slots(signals.host_snap_total_slots)
        && signals.disk_snap_free_slots > 0
    {
        policy
            .host_high_slots(signals.host_snap_total_slots)
            .saturating_sub(signals.host_snap_free_slots)
    } else {
        0
    };
    (kv, snap)
}

/// 每请求分腿配额：KV 槽 / 快照条数分别裁剪（0 = 该腿不设每请求上限）。
fn clamp_host_pressure(
    planned_kv: usize,
    planned_snap: usize,
    request_slots: usize,
    request_snaps: usize,
) -> (usize, usize) {
    let kv = if request_slots == 0 {
        planned_kv.min(HOST_PRESSURE_MAX_SLOTS)
    } else {
        planned_kv.min(request_slots).min(HOST_PRESSURE_MAX_SLOTS)
    };
    let snap = if request_snaps == 0 {
        planned_snap
    } else {
        planned_snap.min(request_snaps)
    };
    (kv, snap)
}

#[cfg(test)]
mod pressure_tests {
    use super::{
        clamp_host_pressure, host_pressure_targets, PressurePolicy, PressureSignals,
    };

    #[test]
    fn low_watermark_triggers_and_high_is_the_target() {
        let p = PressurePolicy::new(12, 25, 12, 25);
        assert_eq!(p.target_slots(100, 11, 64), 14, "低水位下补到高水位");
        assert_eq!(p.target_slots(100, 12, 64), 0, "等于低水位不触发");
        assert_eq!(p.target_slots(100, 0, 64), 25, "空池补到高水位");
        assert_eq!(p.target_slots(100, 0, 8), 8, "单次上限生效");
        assert_eq!(p.target_slots(0, 0, 64), 0, "无池不触发");
    }

    #[test]
    fn small_pool_and_bad_percentages_are_clamped() {
        let p = PressurePolicy::new(0, 0, 0, 0);
        assert_eq!(p.target_slots(4, 0, 64), 1, "1% 至少取 1 页");
        let p = PressurePolicy::new(80, 10, 12, 25);
        assert_eq!(p.target_slots(100, 79, 64), 1, "high < low 时按 low 封底");
    }

    #[test]
    fn host_pressure_uses_independent_low_watermark_and_disk_room() {
        let p = PressurePolicy::new(12, 25, 20, 40);
        let mut s = PressureSignals::default();
        s.host_kv_enabled = true;
        s.host_snap_enabled = true;
        s.host_kv_total_slots = 100; // low = 20, high = 40
        s.host_kv_free_slots = 19;
        s.disk_kv_free_slots = 100;
        assert_eq!(
            host_pressure_targets(&s, &p),
            (21, 0),
            "低于低水位补到高水位"
        );
        s.host_kv_free_slots = 0;
        assert_eq!(host_pressure_targets(&s, &p), (40, 0), "空宿主补到高水位");
        s.host_kv_free_slots = 20;
        assert_eq!(host_pressure_targets(&s, &p), (0, 0), "等于低水位不触发");

        s.host_snap_total_slots = 10; // low = 2, high = 4
        s.host_snap_free_slots = 0;
        s.disk_snap_free_slots = 10;
        assert_eq!(host_pressure_targets(&s, &p), (0, 4), "状态腿独立高水位");
        s.disk_kv_free_slots = 0;
        assert_eq!(host_pressure_targets(&s, &p), (0, 4), "KV 无盘位不影响快照");
        s.host_kv_free_slots = 0; // KV 仍低但无盘位
        assert_eq!(host_pressure_targets(&s, &p), (0, 4));
        s.host_snap_free_slots = 4;
        assert_eq!(host_pressure_targets(&s, &p), (0, 0));
    }

    #[test]
    fn request_quota_clamps_kv_and_snapshot_independently() {
        assert_eq!(
            clamp_host_pressure(40, 3, 16, 1),
            (16, 1),
            "KV/快照各自裁剪"
        );
        assert_eq!(clamp_host_pressure(40, 3, 0, 0), (40, 3), "0 = 不设每请求上限");
        assert_eq!(
            clamp_host_pressure(100, 2, 0, 0),
            (64, 2),
            "KV 仍受绝对 cap 64"
        );
        assert_eq!(clamp_host_pressure(0, 5, 16, 1), (0, 1));
    }

    #[test]
    fn driver_headroom_is_part_of_the_signal() {
        let mut s = PressureSignals::default();
        s.driver_free_bytes = 100;
        s.reserve_bytes = 200;
        assert!(s.driver_below_reserve());
        assert!(s.summary().contains("驱动低于预留"));
        s.driver_free_bytes = 300;
        assert!(!s.driver_below_reserve());
    }
}
