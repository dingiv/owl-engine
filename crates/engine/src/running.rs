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

use crate::spec_runtime::DraftHead as _;
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

/// **KV 磁盘 checkpoint 的一次结果**（S6.2；诊断/日志面）。
///
/// 三数分开记是必要的：`dev_blocks` 走 D2H（贵，受 PCIe 限），`host_blocks`
/// 只是内存拷贝（便宜）——"为什么这次慢"只能从这两个数的比例看出来。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct CheckpointStats {
    /// 从设备只读 D2H 后写盘的块数
    pub dev_blocks: usize,
    /// 从宿主副本直接写盘的块数
    pub host_blocks: usize,
    /// 实际写入字节（KV 块；快照另记）
    pub bytes: usize,
    /// 尾部边界快照是否落盘（0/1）
    pub snap_disk: bool,
}

impl CheckpointStats {
    /// 本次写入的块数。
    pub fn blocks(&self) -> usize {
        self.dev_blocks + self.host_blocks
    }
}

/// **主动 checkpoint 的内部节奏**（不是配置键）。
///
/// 磁盘层一旦启用（`disk_cache_mb>0` + `disk_dir`），空闲拍就按本策略
/// 自动复制已算好的 KV，供重启召回。`step_bytes` 只限制**单次空闲拍**
/// 的复制量，避免长链一次写几 GiB 占住主泵；`min_gap` 限制空闲拍的
/// 磁盘写频率。二者都不参与准入/压力卸载。
///
/// `step_bytes = 0` 仅供测试关闭该路径；生产缺省 = 64 MiB / 5s。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CheckpointPolicy {
    /// 单次空闲拍最多复制的字节数。
    pub step_bytes: usize,
    /// 两次主动 checkpoint 的最小间隔。
    pub min_gap: std::time::Duration,
}

impl Default for CheckpointPolicy {
    fn default() -> Self {
        Self {
            step_bytes: 64 << 20,
            min_gap: std::time::Duration::from_secs(5),
        }
    }
}

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
    /// 草稿逐位追踪（只打已在关键路径读回的 draft/accept 账；TP2 可用）
    pub draft_trace: bool,
    /// 渲染后 prompt 落盘诊断(原 OWL_PROMPT_DUMP;2026-10-15 收编)
    pub prompt_dump: bool,
    /// KV 页差分落点(原 OWL_KV_DUMP,同上收编)
    pub kv_dump_at: Option<usize>,
    /// 解释器活性收尾归因打印(原 `OWL_LIVE_DIAG`)
    pub live_diag: bool,
    /// 层间 prefill bisect(诊断;只回读 token 512 行)
    pub prefill_bisect: bool,
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
            // 配置单源（F-03）：`[probes] debug`；不再读 `OWL_DEBUG`。
            debug: p.debug,
            degrade_after: cfg.spec.degrade_after,
            probe_every: cfg.spec.probe_every,
            trace_gate: p.trace_gate,
            dflash_probe: p.dflash_probe,
            draft_trace: p.draft_trace,
            prompt_dump: p.prompt_dump,
            // ⚠️ `0` = 关(与 full.toml 的文档口径一致):`Option` 会把 toml 的
            // `kv_dump_at = 0` 解析成 **Some(0)** —— 那意味着"每个请求 pos 0
            // 各 d2h 一页",是纯税(2026-10-27 配置面 review 抓到的现役 bug)。
            // 配置单源（F-03）：`[probes] kv_dump_at`；不再读 `OWL_KV_DUMP`。
            kv_dump_at: p.kv_dump_at.filter(|v| *v > 0),
            live_diag: p.live_diag,
            prefill_bisect: p.prefill_bisect,
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
/// - `BootCtx`:boot 期定型、**运行期只读**(配置/分词器/探针/序号)
/// - `crate::turn::TurnLedger`:排队/活跃/id 发号/事件队列
/// - `crate::session::SessionBank`:会话账 + 双块账房(target/MTP)
/// - `crate::select::SelectRuntime`:B 轴视图/覆盖率探针/驻留账
/// - `crate::housekeeping::Housekeeping`:节拍与计数(调度计时/空转/回收/降级)
/// - `RankDev`:rank0 设备束(池/图/模型/env/spec 运行态/tp)
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
    /// **磁盘 I/O 运行层**（S6）：配额 + 溢写/回填编排；store 只出计划/记账。
    pub(crate) disk: crate::mem::disk_runtime::DiskRuntime,
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
    /// 消息开启标记的 token 序列（`[store].snap_msg_delim` 或模型
    /// `ChatFormat.im_open` 在 boot 期 tokenize 的结果；只读）。
    pub(crate) msg_open_ids: Vec<u32>,
    /// boot 序号(metrics 打点命名空间;进程内自增,b1/b2…)
    pub(crate) seq: u64,
}

/// **压力信号**（store 自持的类型；这里只做别名，避免调用点全改）。
pub(crate) use owl_store::PressureSignals;

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

/// **准入 alloc 面的 token 上界**(纯函数;与 `step_need`/`generation_target` **同一口径**)。
///
/// 为什么不能只算 `prompt + max_new`(2026-10-28 实测 `grant 额度不足` 的根因):
/// 产出预算约束的是**发射**位置(`pos ≤ prompt + max_new − 1`),但块表在
/// **最后一个 spec 轮**还要覆盖 verify 行与轮末 propose 链 —— 它们的位号
/// 翻过产出预算继续长(`pos + 2·depth + 2`,DFlash2)。旧 alloc 面按
/// `prompt + max_new` 折算页宽,恰好在"块边界对齐"的 prompt 上少算 1 页:
/// 111340 + 16 tok ⇒ 旧 3480 槽,实需 3481 槽 ⇒ 生成到第 12 个 token 时
/// 撞 `grant 额度不足` 而**整发 500**(台账 `26-10-28-位置定谳与三修` 同卷日志)。
///
/// 因此 alloc 面 = `generation_target_for(最后一个产出位)`，让"预留面"与
/// "增长面"共用同一个上界函数 —— 两处各算一次就是下一次同类事故。
fn chain_alloc_tokens(
    prompt_tokens: usize,
    max_new: usize,
    depth: usize,
    kind: GenerationKind,
) -> usize {
    let last_pos = prompt_tokens.saturating_add(max_new).saturating_sub(1);
    generation_target_for(last_pos, depth, kind)
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

/// **位置压实判定**（图片 leave-gap 的位置空洞用尽）。
///
/// `pos_hi` = 本 turn 已用到的最大 rope 位置（0 基）。本 turn 还要生成
/// `max_new` 枚 token ⇒ 会用到 `pos_hi+1 ..= pos_hi+max_new`；rope 表合法下标
/// 是 `0..pos_cap-1` ⇒ 安全条件 `pos_hi + max_new < pos_cap`。
///
/// 违反 ⇒ **本 turn 必须先压实**（位置账归零重算），否则核会越界读 rope 表
/// （`rope_f32.cu` 对 pos 无边界检查）。正常（非 gap）会话不可能触发：
/// token 预算保证 `pos_hi = n_tokens ≤ max_seq < pos_cap`。
pub fn needs_consolidation(pos_hi: i32, max_new: usize, pos_cap: usize) -> bool {
    let pos_hi = pos_hi.max(0) as usize;
    pos_hi.saturating_add(max_new) >= pos_cap
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

/// **提交边界的错误面**（2026-11-16 P3）。
///
/// 为什么需要：服务面要按「是不是准入拒绝」选 HTTP 语义（429 + `Retry-After`
/// vs 400/500）。旧做法 = 解析 `admission:<code> retry_ms=<n>` 错误串 —— 那是
/// 人读文本，改一个字就静默改变协议行为。现在拒绝在产生点结构化。
#[derive(Clone, Debug)]
pub enum SubmitError {
    /// 准入拒绝（服务面翻 429）
    Rejected(crate::turn::AdmissionReject),
    /// 其余错误（原样；`Display` 与旧口径一致）
    Engine(ModelError),
}

impl SubmitError {
    /// 退回旧错误面（`ModelError::Msg`，文案 = 结构化拒绝的 `Display`）。
    pub fn into_model_error(self) -> ModelError {
        match self {
            Self::Rejected(r) => ModelError::Msg(r.to_string()),
            Self::Engine(e) => e,
        }
    }

    /// 转成 turn 事件面（`Rejected`/`Failed` 的 `err` 字段）。
    pub fn into_turn_error(self) -> crate::turn::TurnError {
        match self {
            Self::Rejected(r) => crate::turn::TurnError::rejected(r),
            Self::Engine(e) => crate::turn::TurnError::new(e.to_string()),
        }
    }

    /// 结构化拒绝（非准入错误 = `None`）。
    pub fn reject(&self) -> Option<&crate::turn::AdmissionReject> {
        match self {
            Self::Rejected(r) => Some(r),
            Self::Engine(_) => None,
        }
    }
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rejected(r) => f.write_str(&r.to_string()),
            Self::Engine(e) => std::fmt::Display::fmt(e, f),
        }
    }
}

impl From<ModelError> for SubmitError {
    fn from(e: ModelError) -> Self {
        Self::Engine(e)
    }
}

impl From<owl_store::StoreError> for SubmitError {
    fn from(e: owl_store::StoreError) -> Self {
        // store 侧已有 `From<StoreError> for ModelError`(唯一换算在 store)
        Self::Engine(e.into())
    }
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

    /// **候选采样读面**(2026-10-26;**2026-10-31 扩到单卡**):裸 decode 的
    /// `cand` 表 → 每行候选并集(全局索引升序)。`None` = 图没有候选槽
    /// (greedy 档 flavor / 未开采样)⇒ 调用方回退 `read_full_logits`。
    ///
    /// 两种拓扑:
    /// - **词表切**:本 rank 半表 + peer 半表,索引加 `/half_vocab` 偏移后 host 合并;
    /// - **单卡**(2026-10-31):本设备已经是**全词表行** ⇒ 直接解析 `[2·rows·K]`
    ///   (与 merge 同一布局,索引已全局)。此前单卡没有这条路 ⇒ 采样档每步回读
    ///   整行(verify 是 `depth1` 行)logits,实测 **-25%** 解码。
    pub(crate) async fn read_session_cands(&mut self) -> Result<Option<Vec<Vec<(f32, u32)>>>> {
        if !self.dev.graphs.session.has_output("cand") {
            return Ok(None);
        }
        let a = self.dev.graphs.session.read_output_f32("cand").await?;
        let Some(p) = self.peer.as_mut() else {
            // 单卡:全词表行的 top-K,索引已全局
            return Ok(Some(crate::exec::parse_local_cands(&a)));
        };
        let half = self.dev.boot.model.logits_vocab();
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
        self.bank.parts().sessions.get(sid).map(|s| s.tokens.clone())
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
    /// 前缀一致):`tokens` 只做历史守卫/诊断；实际可复用面由 store 内容树
    /// 回答，失配回退全量重算(S1,正确性优先)。
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
    pub(crate) fn prompt_positions(
        &self,
        session: Option<u64>,
        prompt_ids: &[u32],
        spans: &[crate::img_evict::ImgSpan],
        deletion: Option<&crate::img_evict::DeletionPlan>,
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
            if let Some(s) = self.bank.parts().sessions.get(sid) {
                if !s.positions.is_empty() && s.positions.len() == s.tokens.len() {
                    // **只有与新 prompt 有传承关系时才沿用旧位置账**：
                    // 纯换分支（公共前缀为 0）若仍从旧 `max_pos+1` 起算，
                    // 会把上一条链的位置洞带进一条毫不相干的新链。删除计划
                    // 例外：它本身就是按旧序列的幸存段构造的，必须沿用旧位。
                    let common = s
                        .tokens
                        .iter()
                        .zip(prompt_ids.iter())
                        .take_while(|(a, b)| a == b)
                        .count();
                    if common == 0 && deletion.is_none() {
                        owl_shared::log_debug!(
                            "img-evict",
                            "位置账:session={sid:#x} 新 prompt 与旧链无公共前缀 ⇒ 紧凑重算（不继承旧洞）"
                        );
                    } else {
                        let (pos, _max, gap) = match deletion {
                            Some(plan) if plan.ranges.len() > 1 => {
                                crate::img_evict::positions_after_deletes(
                                    &s.tokens,
                                    &s.positions,
                                    prompt_ids,
                                    spans,
                                    plan,
                                    image_token,
                                    mrope,
                                )
                            }
                            Some(plan) => crate::img_evict::positions_after_splice(
                                &s.tokens,
                                &s.positions,
                                prompt_ids,
                                spans,
                                plan.first(),
                                image_token,
                                mrope,
                            ),
                            None => crate::img_evict::positions_after_splice(
                                &s.tokens,
                                &s.positions,
                                prompt_ids,
                                spans,
                                None,
                                image_token,
                                mrope,
                            ),
                        }
                        .map_err(ModelError::Msg)?;
                        // 位置洞由本次 splice/deletes 的返回值和 turn 账携带，
                        // 不再在会话上保留额外状态位。
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
    // 结构性豁免(2026-11-02 clippy 清理):类型面 = 派发/表项签名;起别名会藏住"这一串到底是什么"
    #[allow(clippy::type_complexity)]
    /// 单 token 是否为**可见空白**（特殊 token 解码为空 ⇒ false）。
    ///
    /// 只给删除边界复原用：`decode(&[t])` 为空串的 token 可能是任何特殊标记
    /// （`<|im_end|>` 等），绝不能当成“可省略的空白”。
    fn token_is_visible_whitespace(&self, t: u32) -> bool {
        let s = self.boot.tok.decode(&[t]);
        !s.is_empty() && s.trim().is_empty()
    }

    fn image_pad_law(
        &self,
        session: Option<u64>,
        ids: Vec<u32>,
        images: &[crate::turn::TurnImage],
    ) -> Result<(
        Vec<u32>,
        Vec<crate::img_evict::ImgSpan>,
        Vec<(usize, usize)>,
        Option<crate::img_evict::DeletionPlan>,
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
        // 图像内容指纹只算一次：删除识别用它消歧相邻同 token 图像，
        // ②/③ 再把它盖到 spans 上（旧实现到 ③ 才逐图重算）。
        let new_tags: Vec<u64> = images
            .iter()
            .map(|i| crate::vlm::VlmImageKey::of(&i.image).content_tag())
            .collect();
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
        let mut deletion: Option<crate::img_evict::DeletionPlan> = None;
        let mut input = ids;
        let mut inherited: Vec<(usize, usize)> = Vec::new();
        if page > 0 {
            if let Some(sid) = session {
                if let Some(s) = self.bank.parts().sessions.get(sid) {
                    if !s.tokens.is_empty() && !s.img_spans.is_empty() {
                        let old_text =
                            crate::img_evict::unpad(&s.tokens, &s.pads).map_err(ModelError::Msg)?;
                        // **模板 trim 边界复原**（2026-11-15）：Qwen 快照模板对
                        // user content 做 `|trim`，删图后“图片前最后一段文本”的
                        // `\n` 会被剪掉（服务端补的边界 token 只在内部 part 幸存）。
                        // 若 raw 与旧文本的唯一差异是**删点前的尾部空白**，把旧
                        // 前缀的空白插回 raw —— 后续识别/重建/位置账全部走原路径。
                        let restored = crate::img_evict::restore_trimmed_prefix(
                            &old_text,
                            &mut input,
                            &s.img_spans,
                            |t| self.token_is_visible_whitespace(t),
                        );
                        if let Some(k) = restored {
                            owl_shared::log_debug!(
                                "img-evict",
                                "模板 trim 边界复原:session={sid:#x} 补回 {k} 个空白 token(old={} raw={})",
                                old_text.len(),
                                input.len()
                            );
                        }
                        // 旧账 tag=0（升级前拍的 spans）没有内容指纹，退回
                        // 只能整游程识别的旧口；新账一律走带指纹的精确口。
                        let tags_ready = s.img_spans.iter().all(|sp| sp.img_tag != 0);
                        let detect = if tags_ready {
                            crate::img_evict::detect_text_delete_tagged(
                                &old_text,
                                &input,
                                &s.img_spans,
                                &new_tags,
                            )
                        } else {
                            crate::img_evict::detect_text_delete(
                                &old_text,
                                &input,
                                &s.img_spans,
                            )
                        };
                        // 单段没命中且旧图全部被换掉时，再试“多段全删”：
                        // 旧图可能分散在多个 tool result，新图追加在尾部。
                        let multi = if detect.is_none() && tags_ready {
                            crate::img_evict::detect_all_images_deleted_tagged(
                                &old_text,
                                &input,
                                &s.img_spans,
                                &new_tags,
                            )
                        } else {
                            None
                        };
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
                        if detect.is_none() && multi.is_none() {
                            let n = old_text.len().min(input.len());
                            let first_diff = (0..n).find(|&i| old_text[i] != input[i]);
                            if let Some(i) = first_diff {
                                let lo = i.saturating_sub(24);
                                let hi = (i + 40).min(n);
                                // 删点边界诊断：逐 token id + 单 token 文本。BPE 在
                                // “文本→图片/结束标记”处的合并差异只有这里看得见。
                                let w = i.saturating_sub(3)..(i + 4).min(n);
                                owl_shared::log_debug!(
                                    "img-evict",
                                    "first_diff={i} old_win={:?} raw_win={:?}",
                                    self.boot.tok.decode(&old_text[lo..hi]),
                                    self.boot.tok.decode(&input[lo..hi])
                                );
                                owl_shared::log_debug!(
                                    "img-evict",
                                    "first_diff={i} old_ids={:?} raw_ids={:?} old_tok={:?} raw_tok={:?}",
                                    &old_text[w.clone()],
                                    &input[w.clone()],
                                    old_text[w.clone()]
                                        .iter()
                                        .map(|t| self.boot.tok.decode(&[*t]))
                                        .collect::<Vec<_>>(),
                                    input[w].iter()
                                        .map(|t| self.boot.tok.decode(&[*t]))
                                        .collect::<Vec<_>>()
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
                        if detect.is_none() && multi.is_none() {
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
                                (hit.b - hit.a).checked_div(page).unwrap_or(0)
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
                                deletion = Some(crate::img_evict::DeletionPlan {
                                    ranges: vec![hit],
                                });
                            }
                        } else if let Some(plan) = multi {
                            let marked = s.img_spans.iter().filter(|x| x.evict).count();
                            let all_page_aligned = plan
                                .ranges
                                .iter()
                                .all(|r| r.a % page == 0 && r.b % page == 0);
                            if !all_page_aligned {
                                owl_shared::log_warn!(
                                    "img-evict",
                                    "多段删除识别命中但存在非页边界区间 ⇒ 不吸收"
                                );
                            } else {
                                // 多段删除必须保留旧 **padded** 幸存序列（含删点
                                // 前/段间的 orphan pad）；客户端 raw 是 unpadded，
                                // 只接到幸存文本的共同前缀之后。否则 pad 块会在
                                // 删点处与客户端 raw 分叉，整段缓存只能到图片前。
                                let mut survivors = Vec::with_capacity(s.tokens.len());
                                let mut cursor = 0usize;
                                for r in &plan.ranges {
                                    survivors.extend_from_slice(&s.tokens[cursor..r.a]);
                                    cursor = r.b;
                                }
                                survivors.extend_from_slice(&s.tokens[cursor..]);
                                // 旧 pad 里落在删除区间内的丢掉；区间外的 orphan pad
                                // 按删页数左移，继续作为 inherited 搬进新序列。
                                let mut surv_pads = Vec::new();
                                for (lo, hi) in &s.pads {
                                    let mut shift = 0usize;
                                    let mut inside = false;
                                    for r in &plan.ranges {
                                        if *lo < r.b && *hi > r.a {
                                            inside = true;
                                            break;
                                        }
                                        if r.b <= *lo {
                                            shift += r.b - r.a;
                                        }
                                    }
                                    if !inside {
                                        surv_pads.push((lo - shift, hi - shift));
                                    }
                                }
                                let mut surv_text = Vec::with_capacity(old_text.len());
                                let mut tcur = 0usize;
                                for r in &plan.ranges {
                                    surv_text.extend_from_slice(&old_text[tcur..r.text_a]);
                                    tcur = r.text_b;
                                }
                                surv_text.extend_from_slice(&old_text[tcur..]);
                                let common = surv_text
                                    .iter()
                                    .zip(&input)
                                    .take_while(|(a, b)| a == b)
                                    .count();
                                if common > 0 {
                                    let cut = crate::img_evict::padded_prefix_for_raw(
                                        &survivors,
                                        &surv_pads,
                                        common,
                                    )
                                    .map_err(ModelError::Msg)?;
                                    let mut merged = survivors[..cut].to_vec();
                                    merged.extend_from_slice(&input[common..]);
                                    input = merged;
                                    inherited = surv_pads
                                        .into_iter()
                                        .filter(|(_, hi)| *hi <= cut)
                                        .collect();
                                    let n_ranges = plan.ranges.len();
                                    let n_deleted = plan
                                        .ranges
                                        .iter()
                                        .map(|r| r.text_b - r.text_a)
                                        .sum::<usize>();
                                    deletion = Some(plan);
                                    owl_shared::log_info!(
                                        "img-evict",
                                        "多段删除识别: session={sid:#x} old={} raw={} images={}(marked={}) ⇒ 删 {} 段 / {} tok；保留旧 padded 前缀 {} tok + 新尾巴",
                                        old_text.len(),
                                        input.len(),
                                        s.img_spans.len(),
                                        marked,
                                        n_ranges,
                                        n_deleted,
                                        cut
                                    );
                                } else {
                                    owl_shared::log_warn!(
                                        "img-evict",
                                        "多段删除识别命中但幸存文本无公共前缀 ⇒ 不吸收"
                                    );
                                }
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
                owl_shared::log_debug!("img-evict", "detect skipped: 无 session(新建会话)");
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
        // ── ③ **图像内容指纹盖章**(2026-11-03 定谳的陈旧图像 KV):
        // token 流里图像只是字面相同的占位符 ⇒ 块身份必须另带图像内容指纹,
        // 否则"同文本 + 换截图"会整条命中上一条请求的 KV(模型对着旧图答)。
        // 指纹 = 预处理后 pixel_values 的逐位哈希(`VlmImageKey`),与塔输出
        // 缓存同源 ⇒ 同图同值、异图必异。
        let mut spans = seq.spans;
        if spans.len() != images.len() {
            return Err(ModelError::Msg(format!(
                "图像指纹盖章:段 {} ≠ 图 {}(账不平;宁可拒也不错位)",
                spans.len(),
                images.len()
            )));
        }
        for (sp, tag) in spans.iter_mut().zip(&new_tags) {
            sp.img_tag = *tag;
        }
        Ok((seq.ids, spans, seq.pads, deletion))
    }

    /// **本轮块指纹**(`owl_store::BlockTag` 的引擎侧唯一入口)。
    ///
    /// 优先取**活跃 turn** 的 `img_spans`(submit 已按像素盖章);会话账只在
    /// turn 不活跃时当退路(收口/空转拍)。
    ///
    /// ⚠️ 这与"turn 进行中却读会话账"是两回事:会话账在 turn 收口才写回,
    /// 轮内读它拿的是**上一轮**的 spans(图像块会被判成纯文本身份)⇒ 游标/快照键
    /// 与命中面身份不一致 ⇒ 复用面被腰斩(2026-11-03 实测只命中 3 块)。
    pub(crate) fn turn_block_tags(&self, sid: u64, page: usize) -> Vec<owl_store::BlockTag> {
        if let Some(act) = self.turns.active.iter().find(|a| a.session_id == sid) {
            // **已生成 token 也在链上**（2026-11-13 修）：decode 边界的链游标
            // 走到 `prompt + out`，旧实现只按 prompt 长度铸 tags ⇒ 最后一个
            // 跨 prompt/生成边界的块缺 tag，`ChainCursor::advance` 直接拒
            // （实测 `链哈希游标推进失败 ... fed=36704 start=1115 新增 32 块`）。
            // `img_spans` 只覆盖 prompt 区，`block_tags` 对超出部分自然给 0。
            // 位置签名必须同源：非紧凑链（leave-gap）的 K 只在同一 position
            // 基下有效，必须折进块身份，否则跨会话会按 token 撞链。
            let total = act.prompt_ids.len() + act.out.len();
            return crate::img_evict::block_tags_pos(
                total,
                page,
                &act.img_spans,
                &act.positions,
            );
        }
        match self.bank.parts().sessions.get(sid) {
            Some(s) => crate::img_evict::block_tags_pos(
                s.tokens.len(),
                page,
                &s.img_spans,
                &s.positions,
            ),
            None => Vec::new(),
        }
    }

    /// 同 [`Self::turn_block_tags`]，但把 **spec 轮内已提交、尚未进
    /// `act.out`** 的尾 token 也算进块身份。
    ///
    /// 这些 token 的位置按 `act.next_rope_pos` 连续推进；如果只用
    /// `prompt+out` 造 tag，后面真正提交时会得到另一套位置签名 ⇒
    /// 链游标/快照键与发布身份分叉。
    pub(crate) fn turn_block_tags_with_tail(
        &self,
        sid: u64,
        page: usize,
        tail: &[u32],
    ) -> Vec<owl_store::BlockTag> {
        let Some(act) = self.turns.active.iter().find(|a| a.session_id == sid) else {
            return self.turn_block_tags(sid, page);
        };
        let total = act.prompt_ids.len() + act.out.len() + tail.len();
        let mut positions = act.positions.clone();
        if positions.len() < total {
            let base = act.next_rope_pos;
            positions.extend((0..total - positions.len()).map(|i| {
                [base.saturating_add(i as i32); 3]
            }));
        }
        crate::img_evict::block_tags_pos(total, page, &act.img_spans, &positions)
    }

    /// 同 [`Self::submit_session_ex`],但携带 per-turn 可选项(采样参数/停止串/种子;
    /// AGT-06)。`ignore_eos` 在 `opts` 内。
    ///
    /// 只保留**旧错误面**(`ModelError`);需要结构化准入拒绝的服务面请用
    /// [`Self::submit_session_checked`](后者是本函数的实现体)。
    pub fn submit_session_opts(
        &mut self,
        session: impl Into<Option<u64>>,
        prompt: impl Into<String>,
        max_new: usize,
        opts: crate::turn::TurnOpts,
    ) -> Result<(u64, u64)> {
        self.submit_session_checked(session, prompt, max_new, opts)
            .map_err(SubmitError::into_model_error)
    }

    /// **提交（结构化拒绝版）**：准入拒绝以 [`SubmitError::Rejected`] 返回，
    /// 其余错误原样包在 [`SubmitError::Engine`] 里。
    ///
    /// 服务面（`service::pump::accept`）用这个：HTTP 语义（429 + `Retry-After`）
    /// 由结构化字段决定，不再解析错误串（2026-11-16 P3）。
    pub fn submit_session_checked(
        &mut self,
        session: impl Into<Option<u64>>,
        prompt: impl Into<String>,
        max_new: usize,
        opts: crate::turn::TurnOpts,
    ) -> std::result::Result<(u64, u64), SubmitError> {
        let session = session.into();
        // **同会话并发（2026-11-16 P4：引擎排队，不再硬拒）**
        //
        // 同会话两 turn 会共享同一份 KV/GDN 状态 ⇒ 结构性互踩，**必须串行**。
        // 旧做法 = submit 边界硬拒（服务面用 `session_gate` 补一层串行，结果
        // 串行契约有两个真源）。现在引擎自己保证：`startable_queue_idx()` 只
        // 放行"会话无活跃 turn"的队首候选，同会话后继 turn 就排在队里等前一个
        // 终了 —— 顺序优先，不插队，也不 429。
        let prompt = prompt.into();
        // **VLM 输入门**(2026-10-20 视觉接线 V5;2026-10-23 多图):
        // 单请求可带多图(按内容序),但图像 × 合批仍拒;图像 × 投机已接
        // (verify 的 rope pos 加 m-RoPE delta,slots/lens 用真实位置)。
        // 邻接多图的 m-RoPE 由 `MropeCfg::rope_index` 按 grid 顺序分段。
        if !opts.images.is_empty() {
            if self.dev.boot.vlm.is_none() {
                return Err(SubmitError::Engine(ModelError::Msg(
                    "submit: 请求带图,但本档 [vision] enabled != true(引擎无视觉塔)".into(),
                )));
            }
            // 图像 × 并发：**允许提交**；批调度门会排除带图的活跃 turn
            // （图像走单 turn 路径，文本轮仍可批）。旧实现提交边界硬拒 =
            // 27B 档一开并发就丢视觉，agent 截图请求直接 400。
            // **可删标记必须有人接**(前缀无感删除;2026-10-23):客户端打了
            // `owl_evict` 而引擎档没开 = 静默失约(用户以为删了不重算,
            // 实际每轮全量重灌)⇒ 提交边界**响亮拒**,不静默忽略。
            if !self.boot.cfg.knobs.image_evict && opts.images.iter().any(|i| i.evict) {
                return Err(SubmitError::Engine(ModelError::Msg(
                    "submit: 图像带 evict(可删)标记,但本档 [vision] image_evict = false\
                         —— 要么把档位打开,要么去掉标记(不许静默忽略)"
                        .into(),
                )));
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
        // **会话身份 = 内容身份**（原始算法 §32；2026-11-16）：
        // - 显式 `session` 键 = **粘性提示**（保留兼容；客户端声明它是哪条对话）；
        // - 无键 ⇒ 用 prompt 原文在**空闲**会话里找最长公共前缀（≥ 一页）——
        //   "同一条历史对话的续写"由内容本身认定，不再靠客户端身份。
        //
        // 位置：必须在 tokenize 之后（匹配要 token 原文）、在 `image_pad_law`
        // 之前（删除检测/pad 律要先知道是哪条会话的账）。
        let session = match session {
            Some(id) => Some(id),
            None => self.match_idle_session(&raw_ids),
        };
        // **前缀无感删除的页对齐 pad 律**(2026-10-23):只对**带标记**的图像段
        // 做(且只在 `[vision] image_evict` 开时)—— 未标记/未开档 = 逐字节旧行为。
        // 产出的 `spans` 随 turn 走到收口,写回会话账(下一轮删除识别的唯一输入)。
        let (prompt_ids, img_spans, img_pads, deletion) =
            self.image_pad_law(session, raw_ids, &opts.images)?;
        if prompt_ids.is_empty() {
            return Err(SubmitError::Engine(ModelError::Msg("submit: 空 prompt".into())));
        }
        let (positions, positions_gap) =
            self.prompt_positions(session, &prompt_ids, &img_spans, deletion.as_ref())?;
        // **消息边界快照点**(2026-10-24；2026-11-16 键退休 = **恒开**):页对齐
        // token offset，供 `prefill_step_chunk_msg` 切块与 `capture_gdn_snap` 的
        // snap_keep 共用。门 = 消费者存在；`open_ids` 来自配置/模型声明。
        let msg_snaps = if self.boot.cfg.knobs.prefix_cache
            && self.dev.pool.geom.paged
            && self.dev.pool.snap_capacity() > 0
        {
            crate::msg_boundary::message_snap_points(
                &prompt_ids,
                &self.boot.msg_open_ids,
                self.dev.pool.geom.page,
                crate::engine::SNAP_MSG_MIN_GAP,
                self.boot.cfg.knobs.snap_msg_recent,
            )
        } else {
            Vec::new()
        };
        // 淘汰优先级档(2026-11-12):Dense(最近 K 条) / Binary(二分节点) /
        // Extra。不是 pin —— 只决定 shell 淘汰顺序(§二分稀疏淘汰)。
        let msg_snap_classes = if msg_snaps.is_empty() {
            Vec::new()
        } else {
            crate::msg_boundary::classify_snap_points(
                &msg_snaps,
                prompt_ids.len(),
                self.boot.cfg.knobs.snap_msg_recent,
            )
        };
        // **可删图前的硬快照点**（2026-11-13）：截图在 prompt 尾部时，预填充末
        // 快照的键落在被删页上 ⇒ 删除发生时幸存页没有任何边界快照，只能退回
        // fresh。锚点 = **每张可删图段起点**（同游程也各拍）；预填充必须在这些
        // 边界**强制切块**才能拍出 GDN 态（见 `prefill_step_chunk_msg` 的
        // `hard_snaps`）。旧图的锚点键是内容身份，前缀命中时不会重拍。
        // 2026-11-16 键退休 = **恒开**（门 = 消费者存在）。
        let evict_snaps: Vec<usize> = if self.boot.cfg.knobs.image_evict
            && self.boot.cfg.knobs.prefix_cache
            && self.dev.pool.geom.paged
            && self.dev.pool.snap_capacity() > 0
        {
            crate::img_evict::evict_snap_points(
                &img_spans,
                prompt_ids.len(),
                self.dev.pool.geom.page,
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
                return Err(SubmitError::Engine(ModelError::Msg(
                    "submit: per-turn 非 Strict 地板需要**虚拟化两扇门**\
                     ([kvmem].select_decode = true **且** select_partial = true)——\
                     地板语义只有选择层在场才能兜住(设计卷 §2.4/§23.5);\
                     两者任一为假 = 响亮拒,不静默降级"
                        .into(),
                )));
            }
        }
        // 历史守卫只做**诊断/旧键降级**：历史长度唯一真源 = tokens.len()；
        // 真正可复用面由 store 内容树回答，不能把历史长度当设备驻留长度。
        if let Some((id, false)) = session.and_then(|id| {
            self.bank.parts().sessions.get(id)
                .map(|s| (id, s.guard(&prompt_ids)))
        }) {
            // **守卫失配取证**(`[probes] debug`;§4.3):报首个分叉的**位置 +
            // 两侧 id 窗口**。渲染对称律的判据就是这一行:分叉在
            // "assistant 轮边界" = 渲染/切分不对称;分叉在别处 = 客户端回填不同。
            if self.boot.probes.debug {
                if let Some(s) = self.bank.parts().sessions.get(id) {
                    match s.first_divergence(&prompt_ids) {
                        Some(i) => {
                            let lo = i;
                            let hi_old = (i + 8).min(s.tokens.len());
                            let hi_new = (i + 8).min(prompt_ids.len());
                            let (a, b) = (&s.tokens[lo..hi_old], &prompt_ids[lo..hi_new]);
                            owl_shared::log_debug!(
                                "prefix",
                                "历史失配 session={id} history_len={} 首个分叉 idx={i} \
                                 旧(已入 KV)={a:?} ⇒ {:?} ‖ 新(本轮)={b:?} ⇒ {:?}",
                                s.tokens.len(),
                                self.boot.tok.decode(a),
                                self.boot.tok.decode(b)
                            );
                        }
                        None => owl_shared::log_debug!(
                            "prefix",
                            "历史失配 session={id} history_len={} 但前缀全同\
                             (长度不足?prompt={})",
                            s.tokens.len(),
                            prompt_ids.len()
                        ),
                    }
                }
            }
        }
        // 历史命中量**不直接当驻留量**；下面的 store_cached 才是准入依据。
        let cached_hit = 0usize;
        // 会话账只是“历史提示”；真正的可复用面由 store 的内容树/层账回答。
        // 表在 turn 末已释放，但 store 仍可能命中 host/disk；
        // 用空表把命中面归零，会让 admission 按全量 ctx 腾位，反而把刚释放
        // 的缓存整链下沉到 host，下一拍再整链 swap-in（TTFT 异常）。
        let page = self.dev.pool.geom.page.max(1);
        // **只信本会话的 token 账**：store 的内容树是跨会话的，直接拿
        // `match_prefix` 当 admission 的 cached_hit 会把别的会话（或
        // leave-gap 的另一套位置）算成本轮已驻留，导致少腾槽/复用错链。
        // 先取本会话 tokens 与本轮 prompt 的公共前缀，再用 store 命中面封顶。
        let session_prefix = session
            .and_then(|id| self.bank.parts().sessions.get(id))
            .map(|s| {
                s.tokens
                    .iter()
                    .zip(prompt_ids.iter())
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .unwrap_or(0);
        let store_cached = if self.boot.cfg.knobs.prefix_cache
            && self.dev.pool.geom.paged
            && deletion.is_none()
            && !positions_gap
            && session_prefix > 0
        {
            let tags = crate::img_evict::block_tags_pos(
                prompt_ids.len(),
                page,
                &img_spans,
                &positions,
            );
            let (hit, recall) = self
                .bank.parts_mut().rel.match_prefix_tagged(&prompt_ids, &tags)?;
            let (m, _dev) = self
                .bank.parts().rel.reuse_face(&hit, &recall, prompt_ids.len(), page);
            m.saturating_mul(page)
                .min(session_prefix)
                .min(prompt_ids.len())
        } else {
            0
        };
        // **删除轮的复用面下界**（2026-11-15 真机定谳）：
        //
        // leave-gap 的复用决策在 `schedule` 之后才落地；submit 侧的守卫
        // （`s.guard(&prompt_ids)`）对“旧 padded 序列 + 新 gapped 序列”必然
        // 失配，`store_cached` 也因 `deletion.is_some()/positions_gap` 被跳过
        // ⇒ `cached_hit=0` ⇒ **admission 按全量 ctx 预留**。真机签名（30k
        // 会话、每轮删一张旧截图）：正常轮 1036 页，删图轮 **1952 页 / 复用 0**
        // （多留 ~1GB），资源充足时也会逼出无谓的腾位/下沉。
        //
        // 这里只取**第一段删除之前**的旧前缀（删除检测已保证逐 token 相同），
        // 用它在内容树里的**设备驻留块数**作下界：真正的复用面（含后缀页 +
        // 快照恢复）只会 ≥ 它，所以 admission 不会少留；日志/usage 也不再
        // 把删图轮报成 0。
        let deletion_cached = match deletion.as_ref().and_then(|p| p.first()) {
            Some(first) => {
                let a = first.a;
                if a == 0 || !a.is_multiple_of(page) {
                    0
                } else {
                    let prefix_tags = session
                        .and_then(|id| self.bank.parts().sessions.get(id))
                        .filter(|s| a <= s.tokens.len() && a <= prompt_ids.len() && s.tokens[..a] == prompt_ids[..a])
                        .map(|s| {
                            (
                                s.tokens[..a].to_vec(),
                                crate::img_evict::block_tags_pos(
                                    a,
                                    page,
                                    &s.img_spans,
                                    &s.positions,
                                ),
                            )
                        });
                    match prefix_tags {
                        Some((prefix, tags)) => {
                            let (hit, _) =
                                self.bank.parts_mut().rel.match_prefix_tagged(&prefix, &tags)?;
                            let resident = hit
                                .slots
                                .iter()
                                .take(hit.matched_blocks)
                                .filter(|slot| slot.is_some())
                                .count();
                            resident.saturating_mul(page).min(a)
                        }
                        None => 0,
                    }
                }
            }
            None => 0,
        };
        // **同会话串行排队期（P4）**：本会话已有活跃/排队 turn ⇒ 会话仍在写，
        // 此刻的账（tokens / 内容树）是"未来态"——命中估计不可作预算依据。
        // 记 0（保守 = 按全量预留）；真正的命中量在起跑定相后由 `CachedTokens`
        // 帧校正（AGT-12），正确性不依赖这个估值。
        let session_busy = session.is_some_and(|id| {
            self.turns.active.iter().any(|a| a.session_id == id)
                || self.turns.queue.iter().any(|t| t.session_id == id)
        });
        let cached_hit = if session_busy {
            0
        } else {
            cached_hit
                .max(store_cached)
                .max(deletion_cached)
                .min(prompt_ids.len())
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
                return Err(SubmitError::Engine(ModelError::Msg(format!(
                    "submit: prompt {} + 生成 {} 超预算 {}",
                    prompt_ids.len(),
                    max_new,
                    self.boot.cfg.max_seq_tokens
                ))));
            }
        };
        // 会话**一律持久**（2026-11-16 原始算法 §32）：无论显式键还是内容身份，
        // turn 终局只还设备表，不焚会话（idle 会话的快照/前缀账是下一轮命中的
        // 依据，焚掉等于内容身份失效）。
        let session_id = self.bank.get_or_create(session)?;
        // **L1 快照键沿用 + 重分类**(2026-11-12 二刀)
        //
        // 纯 append 时位置对齐(前缀守卫生效)⇒ 直接沿用上轮 `msg_snap_keys`;
        // 然后按**本轮**的 `msg_snap_classes` 重分类已存在的快照(不搬内容)。
        // 旧 dense 尾巴降级为 Binary/Extra —— 否则 Dense 会跨轮无限积累。
        // 截短/前缀失配时,不再属于当前链的旧锚点统一降为 Extra。
        let mut msg_snap_keys: Vec<Option<owl_store::SnapKey>> = vec![None; msg_snaps.len()];
        let mut demote_keys: std::collections::BTreeSet<owl_store::SnapKey> =
            std::collections::BTreeSet::new();
        if let Some(s) = self.bank.parts().sessions.get(session_id) {
            if cached_hit > 0 {
                msg_snap_keys = crate::serve::session::carry_msg_snap_keys(
                    &s.msg_snaps,
                    &s.msg_snap_keys,
                    &msg_snaps,
                );
                // 旧边界不在新 prompt 里(截短/回撤)⇒ 降为 Extra。
                let carried: Vec<owl_store::SnapKey> =
                    msg_snap_keys.iter().flatten().copied().collect();
                for (at, key) in s.msg_snaps.iter().zip(s.msg_snap_keys.iter().copied()) {
                    if let Some(key) = key {
                        if msg_snaps.binary_search(at).is_err() && !carried.contains(&key) {
                            demote_keys.insert(key);
                        }
                    }
                }
                // 截短:旧 turn 末 cursor 在新 prompt 之外,不能再作为起点。
                if prompt_ids.len() < s.tokens.len() {
                    if let Some(tail) = s.tail_snap {
                        demote_keys.insert(tail);
                    }
                }
            } else {
                // 前缀失配/新链:旧边界与旧 tail 都不再是当前链的 Dense 锚点。
                demote_keys.extend(s.msg_snap_keys.iter().flatten().copied());
                if let Some(tail) = s.tail_snap {
                    demote_keys.insert(tail);
                }
            }
        }
        let mut reclassified = 0usize;
        let mut reclass_miss = 0usize;
        debug_assert_eq!(
            msg_snap_keys.len(),
            msg_snap_classes.len(),
            "L1 keys/classes 必须同序等长"
        );
        for i in 0..msg_snap_keys.len() {
            let Some(key) = msg_snap_keys[i] else { continue };
            let Some(class) = msg_snap_classes.get(i).copied() else {
                continue;
            };
            if class == owl_store::SnapClass::Transient {
                continue;
            }
            if self.bank.parts_mut().rel.reclassify_snap(key, class) {
                reclassified += 1;
            } else {
                // 已被逐出是正常事件(不是错误);同时清掉 L1 残影,避免
                // 每轮 submit 都重试一个不存在的键。
                msg_snap_keys[i] = None;
                reclass_miss += 1;
            }
        }
        for key in demote_keys {
            // 仍在当前消息边界表里的键由上面的分类循环管,别降级。
            if msg_snap_keys.iter().flatten().any(|k| *k == key) {
                continue;
            }
            if self
                .bank.parts_mut().rel.reclassify_snap(key, owl_store::SnapClass::Extra)
            {
                reclassified += 1;
            } else {
                reclass_miss += 1;
            }
        }
        if reclassified > 0 || reclass_miss > 0 {
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("snap.reclassify", reclassified as u64, file!(), line!());
                m.counter_add("snap.reclassify.miss", reclass_miss as u64, file!(), line!());
            });
        }
        // **生成预算 = 上下文 + 本请求的输出上限**（2026-11-17 裁决 A）。
        //
        // 预留口径 = 这轮**实际可能生成的长度**：`max_new`（请求值，先被
        // `max_new_cap` 与上下文窗口 `max_seq − prompt` 夹过）。
        //
        // 为什么不再预留整档 cap（2026-11-16 旧口径）：客户端只要 13k 时，
        // 多留到 32k 的那 19k 永远不可能被本 turn 兑现；而 `prompt + cap`
        // 在近窗时正好 = 整窗 ⇒ 触发 `reserve_pages` 安全边 ⇒ 235k/241k
        // 这类 "prompt + 请求值明明装得下" 的请求被永久 `no_memory`。
        // 在跑 turn 仍按自己的 `max_new` 全额预留，新人挤不动它。
        let window_left = self
            .boot
            .cfg
            .max_seq_tokens
            .saturating_sub(prompt_ids.len());
        if window_left == 0 {
            return Err(SubmitError::Engine(ModelError::Msg(format!(
                "submit: prompt {} 已占满上下文窗口（max_seq {}）—— 连 1 token 的输出预算都没有",
                prompt_ids.len(),
                self.boot.cfg.max_seq_tokens
            ))));
        }
        let max_new = max_new
            .min(self.boot.cfg.knobs.max_new_cap)
            .min(window_left);
        let reserve_new = max_new;
        let id = self.turns.alloc_id();
        self.turns.queue.push_back(Turn {
            id,
            session_id,
            spec: TurnSpec {
                prompt,
                max_new,
                ignore_eos: opts.ignore_eos,
            },
            reserve_new,
            prompt_ids,
            msg_snaps,
            msg_snap_classes,
            msg_snap_keys,
            evict_snaps,
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
        // 快照归属由 `SessionBank::close` 同屋收口（会话 id 可复用，
        // 旧归属不能串到新会话）。
        self.bank.close(id)
    }

    /// 会话账观测(已入 KV 的 token 数;None = 会话不存在)
    pub fn session_len(&self, id: u64) -> Option<usize> {
        self.bank.parts().sessions.get(id).map(|s| s.tokens.len())
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
    // 结构性豁免(2026-11-02 clippy 清理):排序键是复合比较(多字段/方向混合),抽 key 会丢掉可读性
    #[allow(clippy::unnecessary_sort_by)]
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
    // 结构性豁免(2026-11-02 clippy 清理):排序键是复合比较(多字段/方向混合),抽 key 会丢掉可读性
    #[allow(clippy::unnecessary_sort_by)]
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

    /// **准入排队中的候选 turn**（F-19）：`Some(id)` 表示泵循环不应因
    /// `Idle` 断流 —— `admission_on_full = Queue` 的重判与超时都靠下一次
    /// `pump`，没有外部事件时必须由服务面定时重试。
    pub fn admission_pending(&self) -> Option<u64> {
        let (id, _) = self.turns.admit_wait?;
        let cand = self.startable_queue_idx()?;
        (self.turns.queue[cand].id == id).then_some(id)
    }

    /// 本会话的 turn 集（**活跃在前，其后按排队序**）。
    ///
    /// 用途（P4）：同会话现在可以"活跃 1 条 + 排队 N 条"，而断流取消可能只带
    /// session（客户端在 `Accepted` 之前就断了）。调用方据此挑**接收方已关**的
    /// 那一条，而不是盲取消活跃 turn（否则会误杀上一条请求）。
    pub fn session_turns(&self, session: u64) -> Vec<u64> {
        self.turns
            .active
            .iter()
            .filter(|a| a.session_id == session)
            .map(|a| a.id)
            .chain(
                self.turns
                    .queue
                    .iter()
                    .filter(|t| t.session_id == session)
                    .map(|t| t.id),
            )
            .collect()
    }

    /// **本拍可起跑的候选** = 队列里第一个"会话无活跃 turn"的排队项。
    ///
    /// 2026-11-16 P4：同会话串行的**唯一判据**。同会话后继 turn 排在队里等
    /// 前一个终了（`Reason::SessionBusy` 因此只在 `sched_policy` 单测里作为
    /// 不变量出现）；不同会话按 FIFO，不插队。
    ///
    /// 为什么"按位置取第一个"就够：同一会话的多个排队 turn 里，最靠前的那个
    /// 总是先被选中（前面的位置优先），所以不存在"后一个抢跑"。
    pub(crate) fn startable_queue_idx(&self) -> Option<usize> {
        self.turns.queue.iter().position(|t| {
            !self
                .turns
                .active
                .iter()
                .any(|a| a.session_id == t.session_id)
        })
    }

    /// **无键请求的会话解析**（内容身份；原始算法 §32；2026-11-16）。
    ///
    /// 活跃/排队会话不参与匹配：它们的账是"未来态"（尾部正在写）。
    /// 匹配下限 = 一页：短于它的公共前缀没有区分度（所有 agent 会话都共享
    /// 同一段系统提示）。
    fn match_idle_session(&self, prompt: &[u32]) -> Option<u64> {
        if !self.boot.cfg.knobs.prefix_cache {
            return None;
        }
        let page = self.dev.pool.geom.page.max(1);
        let busy: std::collections::HashSet<u64> = self
            .turns
            .active
            .iter()
            .map(|a| a.session_id)
            .chain(self.turns.queue.iter().map(|t| t.session_id))
            .collect();
        let hit = self.bank.parts().sessions.match_prefix(prompt, page, &busy);
        if let Some((id, matched)) = hit {
            owl_shared::log_debug!(
                "session",
                "内容身份命中：session={id:#x} 公共前缀 {matched} tok（无键请求）"
            );
        }
        hit.map(|(id, _)| id)
    }

    /// 活跃 turn 数(B5.1 多活跃;服务面灌流测量读 —— 并发是否真进引擎)。
    /// **活跃 session 集合**（盘层 session 感知淘汰输入；纯 host）。
    fn active_session_ids(&self) -> Vec<u64> {
        self.turns.actives().map(|a| a.session_id).collect()
    }

    pub fn active_len(&self) -> usize {
        self.turns.active.len()
    }

    /// **首 token 超期清单**（服务面泵循环每拍调用；纯 host 账，零设备）。
    ///
    /// 预算在起跑时按新增 token 定（`crate::ttft_gate`）；到点仍无首 token
    /// ⇒ 调用方打日志并 `cancel_turn_reason(..., "TTFT 超时")`。
    pub(crate) fn ttft_expired(&self, now_ms: u64) -> Vec<crate::turn::TtftExpired> {
        self.turns.ttft_expired(now_ms)
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
            // 生命周期节点 = 图面可能已换/已回收:哑输出块按 face 缓存,
            // 跨 turn 复用会撞"块不在本账本"(实测 turn2 起每轮被拒)。
            dev.fi_plan_out.clear();
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
                let applied = self.pool_housekeeping(need, &[]).await?;
                let grant = match reserve {
                    Some(r) => Some(self.bank.parts_mut().rel.reserve_capacity(
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
                // capture 路径：快照按统一优先级入队，不按活跃会话点名保护。
                let slot = self.store_state_slot(key, &[]).await?;
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
        let slot = match self.bank.parts().rel.state_locate(key) {
            Some(owl_store::Tier::Gpu) => {
                let slot = self.dev.pool.snap_slot_of(key).ok_or_else(|| {
                    ModelError::Msg(format!("快照 {key:?} 账在设备层但池里找不到槽(账目不平)"))
                })?;
                // 使用即触达(store 出序的真源)
                self.bank.parts_mut().rel.touch_snap(key);
                slot
            }
            Some(owl_store::Tier::Host) => {
                let slot = self.store_state_slot(key, &[key]).await?;
                self.snap_onload(key, slot).await?;
                slot
            }
            Some(owl_store::Tier::Disk) => {
                // **S6**:盘 → 宿主 → 设备(两跳);读回后层账回 Host,再走 Host 分支
                let rel = self.bank.parts_mut().rel;
                self.disk.fill_snap(rel, key).await?;
                let slot = self.store_state_slot(key, &[key]).await?;
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
    ) -> Result<u32> {
        use crate::state::SnapSlotPick;
        // **Snap 票**：快照设备槽也进同一本 ReservationLedger（turn 生命周期）。
        // 无活跃 turn（测试/纯 store 路径）⇒ 不发；正常 capture/restore 都有 turn。
        if let Ok(act) = self.turns.active() {
            let bank = self.bank.parts_mut();
            let grant = bank.rel.reserve_snap(act.id)?;
            bank.rel
                .consume_capacity(grant.id, 1, bank.rel.snap_file_bytes())?;
        }
        // 只保护**本次操作正在载入的 key**（不是活跃会话的特权）：
        // 其余快照全部进入统一优先级队列。
        let protect = extra_snaps.to_vec();
        for _ in 0..3 {
            // **判据 = store(唯一产地;§30.6-②)**:出序(事实档 + EvictPolicy)从
            // `Rel::snap_victim_order` 拿;池只做 key→槽映射。
            let order = self.bank.parts().rel.snap_victim_order(&protect);
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
                    let snap_bytes = self.bank.parts().rel.snap_file_bytes();
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
                    self.bank.parts_mut().rel.drop_snap_host(victim)?;
                    if let Err(e) = self.bank.parts_mut().rel.drop_snap_account(victim) {
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
        // 盘上的块先读回宿主(盘 → 设备要两跳;判据与账都在 store)
        //
        // **分相计时**(2026-11-01 P1 回灌提速):整链回灌的天花板在哪一段 ——
        // 盘→宿主(读+拷贝) / plan / swap(主机→设备) / commit。每次 ≥64 块打一行,
        // 千块级回灌的调优全靠这三个数。
        // **窗口式回灌**(2026-11-01 P1 提速):整链一次性回灌时,
        // ① 盘→宿主是每块一次线程跳(0.96 GB/s)、② pinned staging 一次
        // `cudaHostAlloc` 要 2 GiB 级(~1.3 s/rank)、③ peak 宿主/显存都顶满。
        // 切成窗口后:staging 只有 `W × block_bytes`,第一次分配后**池内复用**,
        // 读盘走批量口,三笔一起降。W 取 1024 块(≈512 MiB/rank 档)。
        // 死代码豁免(2026-11-02 clippy 清理):盘→宿主回灌的分段宽度档位;批量并行读上线后
        // 由 `disk_io::read_blocks` 内部定窗,这里留作对照/回归的显式口径。
        #[allow(dead_code)]
        const LOAD_WINDOW_BLOCKS: usize = 1024;
        let t_fill = std::time::Instant::now();
        // 批量读(整窗一次线程跳 + 4 路并行;逐块口每块一次 spawn_blocking,
        // 热页缓存也只有 0.96 GB/s)
        let bank = self.bank.parts_mut();
        let filled = self.disk.fill_kv_window(bank.rel, &chains).await?;
        if filled == 0 {
            // 全在宿主/设备 ⇒ 走老口(它还会 touch 盘副本,保持 LRU 语义)
            self.disk.ensure_host_resident(bank.rel, &chains).await?;
        }
        let d_fill = t_fill.elapsed();
        let need_slots = chains.len();
        let block_bytes = self.dev.pool.block_bytes();
        let load_bytes = need_slots
            .checked_mul(block_bytes)
            .ok_or_else(|| ModelError::Msg("Load Grant: 字节溢出".into()))?;
        // **Load / Temp 票**（同一本 ReservationLedger）：
        // - Load = 回灌块占用的设备槽；consume 标记已兑现；
        // - Temp = 本次回灌的周转额度；turn 收尾统一归还。
        // 两者都不进 ActiveFloor 的 KV 准入面（`Rel::reserved_slots` 按 kind 过滤）。
        let load_grant = self.bank.parts_mut().rel.reserve_load(req.turn, need_slots)?;
        self.bank.parts_mut().rel.consume_capacity(load_grant.id, need_slots, load_bytes)?;
        let temp_grant = self
            .bank.parts_mut().rel.reserve_temp(req.turn, need_slots, load_bytes)?;
        self.bank.parts_mut().rel.consume_capacity(temp_grant.id, need_slots, load_bytes)?;
        let mut applied = owl_store::Applied::default();
        // 设备槽先腾够：`Need.slots` 是需求，`plan_capacity` 会按当前 free 折算缺口。
        if self.bank.parts().rel.slots_free() < need_slots {
            applied = self
                .pool_housekeeping(
                    owl_store::Need::kv(need_slots, owl_store::NeedWhy::Grow),
                    &[],
                )
                .await?;
        }
        let t_plan = std::time::Instant::now();
        let (cs, new_slots, pieces, table) = {
            let p = self.bank.parts_mut().rel.plan_onload(&chains)?;
            (p.chains, p.slots, p.pieces, p.table)
        };
        let d_plan = t_plan.elapsed();
        let t_swap = std::time::Instant::now();
        let table = match self
            .swap_all_ranks(owl_iface::contract::SwapDir::In, &pieces, table)
            .await
        {
            Ok(t) => t,
            Err(e) => {
                self.bank.parts_mut().rel.abort_onload(&cs, &new_slots);
                return Err(e);
            }
        };
        let slots = self.bank.parts_mut().rel.commit_onload(&cs, &new_slots, table)?;
        let d_swap = t_swap.elapsed();
        let loaded_blocks = cs.len();
        let moved_bytes = loaded_blocks * self.dev.pool.block_bytes();
        if loaded_blocks >= 64 {
            let gbs = |b: usize, d: std::time::Duration| {
                if d.as_secs_f64() > 0.0 {
                    b as f64 / d.as_secs_f64() / 1e9
                } else {
                    0.0
                }
            };
            owl_shared::log_info!(
                "store",
                "回灌 {loaded_blocks} 块 / {:.0} MiB:盘→宿主 {:.0} ms({:.2} GB/s) · \
                 plan {:.0} ms · 宿主→设备 {:.0} ms({:.2} GB/s) · 合计 {:.0} ms",
                moved_bytes as f64 / 1048576.0,
                d_fill.as_secs_f64() * 1e3,
                gbs(moved_bytes, d_fill),
                d_plan.as_secs_f64() * 1e3,
                d_swap.as_secs_f64() * 1e3,
                gbs(moved_bytes, d_swap),
                (d_fill + d_plan + d_swap).as_secs_f64() * 1e3
            );
        }
        let s = self
            .bank.parts_mut().sessions.get_mut(req.session)
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
    /// **H3 链登记**（TurnEnd/Abort 都发）。
    ///
    /// **位置空洞会话的发布**（2026-11-13 修正）：链身份现在混入非紧凑
    /// position 签名（[`crate::img_evict::block_tags_pos`]）⇒ gapped 链与
    /// 紧凑链不会撞哈希，发布是安全的，也能让同会话跨 turn 继续复用。
    ///
    /// 位置账与 token 账必须等长；不齐 = 结构错，拒绝发布（宁可下轮重算，
    /// 不许把无位置签名/错位置的链写进共享树）。
    fn publish_turn_chain(
        &mut self,
        session: u64,
        turn: u64,
        kind: ReleaseKind,
    ) -> Result<()> {
        let bank = self.bank.parts_mut();
        let Some(s) = bank.sessions.get(session) else {
            return Ok(());
        };
        if s.positions.len() != s.tokens.len() {
            owl_shared::log_warn!(
                "prefix",
                "链发布跳过：session={session:#x} turn={turn} 位置账 {} ≠ token {}（位置签名不可信）",
                s.positions.len(),
                s.tokens.len()
            );
            return Ok(());
        }
        // 指纹 = 会话账的 pad 后坐标(span 已在 submit 侧按像素盖章)，
        // 并混入位置签名：leave-gap 的非紧凑链不能与紧凑链共享身份。
        let tags = crate::img_evict::block_tags_pos(
            s.tokens.len(),
            self.dev.pool.geom.page.max(1),
            &s.img_spans,
            &s.positions,
        );
        let up = bank.rel.register_chain_owned_tagged(
            &s.tokens,
            &s.block_table,
            session,
            &tags,
        )?;
        if self.boot.cfg.knobs.probes.debug {
            owl_shared::log_debug!(
                "dbg prefix",
                "register_chain({kind:?}): 块表={} 新入={} 逐出={}",
                s.block_table.len(),
                up.inserted.len(),
                up.evicted.len()
            );
        }
        Ok(())
    }

    async fn store_release(&mut self, release: Release) -> Result<StoreOutcome> {
        let turn = release.turn;
        // **P0-4：Release 归还本 turn 的全部 Grant**（幂等；准入收口也会再调一次）。
        let released = self.bank.parts_mut().rel.release_turn_grants(turn);
        if !released.is_empty() {
            owl_shared::log_debug!(
                "store",
                "Release 归还 Grant：turn {turn} {} 张 / {} 槽",
                released.len(),
                released.iter().map(|g| g.slots).sum::<usize>()
            );
        }
        // ── H3：链登记（两种终局都发）────────────────────────────
        // **Abort 也发**(2026-11-03):客户端中断时 KV 里已经有 `[0, fed)`
        // 的真内容,它的内容身份必须进树 —— 否则下一个同前缀请求按内容
        // 寻址命中 0 ⇒ 整段重算(真机实测 5850 tok 白算 2.39 s)。
        // 账源由 `exec/turn_end.rs::ledger_written` 在调本钩子前写好
        // (中断路径已把"已写入"写回会话账)。
        // 盘索引/空转拍仍只给 TurnEnd:中断路径要便宜。
        if release.publish && self.dev.pool.geom.paged {
            self.publish_turn_chain(release.session, turn, release.kind)?;
        }
        // ── H3b：盘索引 / 空转拍（只给正常收尾）─────────────────
        if release.kind == ReleaseKind::TurnEnd {
            // **S6.1：盘上链索引**（内容可能还没溢写 ⇒ boot 按内容截断）。
            // 会话账此刻仍在（turn 终局不再抹账）。
            if self.dev.pool.geom.paged && self.bank.parts().rel.disk_enabled() {
                let toks = self
                    .bank.parts().sessions.get(release.session)
                    .map(|s| s.tokens.clone())
                    .unwrap_or_default();
                if !toks.is_empty() {
                    let tags = self
                        .bank.parts().sessions.get(release.session)
                        .map(|s| {
                            crate::img_evict::block_tags_pos(
                                s.tokens.len(),
                                self.dev.pool.geom.page.max(1),
                                &s.img_spans,
                                &s.positions,
                            )
                        })
                        .unwrap_or_default();
                    let bank = self.bank.parts_mut();
                    self.disk
                        .write_chain_index(bank.rel, &toks, &tags, self.dev.pool.geom.page)
                        .await?;
                }
                // **S6.2 主动落盘**:turn 收口只**登记**(2026-11-01 收尾)——
                // 真正落盘挪到空转拍。理由:一次配额写 2400 MiB 是 5 s 量级,
                // 挂在收口上直接变成 agent 的尾延迟;而空转拍没有人在等回包。
                // 会话 tokens 在空转拍仍可取(会话未出表),不影响链取址。
                let tags = self
                    .bank.parts().sessions.get(release.session)
                    .map(|s| {
                        crate::img_evict::block_tags_pos(
                            s.tokens.len(),
                            self.dev.pool.geom.page.max(1),
                            &s.img_spans,
                            &s.positions,
                        )
                    })
                    .unwrap_or_default();
                self.hk.arm_kv_checkpoint(release.session, toks, tags);
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

        // ── H6：表/引用归零（会话账留着；只还设备表） ----------------------
        {
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
        if let Err(e) = self.bank.parts_mut().rel.return_state_row(release.gdn_slot, None) {
            owl_shared::log_warn!("session", "turn {} 还 GDN 行失败: {e}", release.turn);
        }

        // ── 归属下行（会话恒活 ⇒ Session(s)） ----------------------------
        self.set_ctx_after_turn(release.session);

        // ── N4：池清点 + 未归还清单 + 兜底释放 + trim ---------------------
        // **先交还 eager 档的瞬态输出**(2026-10-28):那批块是 plan 自己持有的
        // 句柄(要到下一步开头才回收);若让 N4 兼底释放先跑(它会把借出登记
        // 删掉并真 free),下一 turn 头部就会对同一批 id 双 free —— 实测
        // `幽灵归还 —— 账上无 id 的在借登记`,第二次请求必失败(与量化方案
        // 无关;F16 档同形)。捕获档里这个调用是 no-op(输出槽由图持有)。
        if let Err(e) = self.dev.graphs.session.release_eager_outputs().await {
            owl_shared::log_warn!("store", "turn {turn} 交还 eager 输出失败(继续收尾): {e}");
        }
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
                let snap_bytes = if self.bank.parts().rel.snap_host_free_slots() > 0 {
                    self.bank.parts().rel.snap_file_bytes()
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
        let applied = self.pool_housekeeping(need, &[]).await?;
        Ok(StoreOutcome {
            applied,
            ..StoreOutcome::default()
        })
    }

    /// **压力缓解口**（actor 在 `PoolExhausted` 后调）。
    ///
    /// 次序：**先还缓存**（GPU 默认池 trim —— 真还驱动），
    /// **先腾宿主**（`Maintain::HostPressure`：只出 Disk 票），
    /// **再做池内收缩**（`Maintain::Pressure`：KV 槽 + 状态腿目标）。
    /// 返回 `(GPU 池缓存回收字节, 目标腾出 KV 槽数, 目标腾出 KV 宿主块数, 目标腾出快照数)`；失败响亮上抛
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
                "压力缓解:GPU 池缓存回收 {:.1}MiB 后归还驱动({:.0}MiB free)",
                freed as f64 / 1048576.0,
                owl_shared::vram::free_bytes() as f64 / 1048576.0
            );
        }
        // 池缓存 pages 真还驱动（不增池槽，但能缓解驱动硬墙）
        let trimmed = self.trim_pools().await;
        let mut trim_bytes = 0u64;
        for (rank, r) in trimmed {
            match r {
                Ok(n) => trim_bytes += n,
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
            // 记录收缩前的自由页：`slots` 是"缺口到高水位",不是绝对目标。
            let free_before = self.bank.parts().rel.slots_free();
            self.store_hook(StoreHook::pressure_with_snap(slots, snap_bytes))
                .await?;
            // **第四级：整会话淘汰**（原始算法 §30/§32；2026-11-16）。
            // 逐块/逐快照收缩之后仍不够 ⇒ 从**最冷空闲会话**的末端向内丢掉
            // 它挂着的缓存独占块，并把它从会话表里关掉（快照归属降 Extra）。
            // 不动活跃/排队会话；被别家共用的前缀块引用不为 1 ⇒ `prune_branch`
            // 自然停手（内容不丢）。
            if slots > 0 {
                let want_free = free_before.saturating_add(slots);
                let (n, pages) = self.evict_cold_sessions(want_free);
                if n > 0 {
                    owl_shared::log_warn!(
                        "store",
                        "压力缓解:整会话淘汰 {n} 个（LRU 最冷）腾 {pages} 页（目标 {want_free}，现有 {}）",
                        self.bank.parts().rel.slots_free()
                    );
                }
            }
        }
        Ok((freed, slots, host_kv, host_snap))
    }

    /// **整会话淘汰到自由页目标**（原始算法 §30/§32；2026-11-16）。
    ///
    /// 次序（每淘汰一个会话）：
    /// 1. 先取**链尾身份**（关账后会话账里就取不到了）；
    /// 2. `close_session` —— 还双链引用 + 快照归属降 `Extra`（内容仍在树上）；
    /// 3. `prune_branch(tail, 整链块数)` —— **从叶向内**逐掉引用为 1 的
    ///    缓存独占块（别的会话还在用的共享前缀引用 > 1，自动停住）。
    ///
    /// 返回 `(淘汰会话数, 腾出的池页数)`。活跃/排队会话不参与；淘汰失败只
    /// WARN 并拉黑该候选（不静默、不死循环 —— 压力路径不能把 turn 拖死）。
    pub(crate) fn evict_cold_sessions(&mut self, want_free: usize) -> (usize, usize) {
        let mut busy: std::collections::HashSet<u64> = self
            .turns
            .active
            .iter()
            .map(|a| a.session_id)
            .chain(self.turns.queue.iter().map(|t| t.session_id))
            .collect();
        let free0 = self.bank.parts().rel.slots_free();
        let mut evicted = 0usize;
        while self.bank.parts().rel.slots_free() < want_free {
            let Some(victim) = self.bank.parts().sessions.lru_victim_excluding(&busy) else {
                break; // 只剩忙会话 ⇒ 压力只能由后续 turn 自行消化
            };
            busy.insert(victim);
            let tail = self.session_tail_hash(victim);
            if let Err(e) = self.close_session(victim) {
                owl_shared::log_warn!(
                    "session",
                    "压力缓解:关会话 {victim:#x} 失败（继续下一个）：{e}"
                );
                continue;
            }
            if let Some(tail) = tail {
                let span = self.bank.parts().rel.chain_blocks(tail).len();
                if span > 0 {
                    match self.bank.parts_mut().rel.prune_branch(tail, span) {
                        Ok(_) => {}
                        Err(e) => owl_shared::log_warn!(
                            "store",
                            "压力缓解:会话 {victim:#x} 整链回退失败（继续）：{e}"
                        ),
                    }
                }
            }
            evicted += 1;
        }
        (evicted, self.bank.parts().rel.slots_free().saturating_sub(free0))
    }

    /// 会话当前链尾的内容身份（压力淘汰用；`None` = 不足一整块/不在树上）。
    fn session_tail_hash(&self, id: u64) -> Option<owl_store::ChainHash> {
        let s = self.bank.parts().sessions.get(id)?;
        let page = self.dev.pool.geom.page.max(1);
        let full = s.tokens.len() / page;
        if full == 0 {
            return None;
        }
        let tags = crate::img_evict::block_tags_pos(
            s.tokens.len(),
            page,
            &s.img_spans,
            &s.positions,
        );
        self.bank.parts().rel.tail_hash_tagged(&s.tokens, full, &tags)
    }

    /// **压力信号快照**（P1-3；纯 host 账，零 I/O）。
    fn pressure_signals(&self) -> PressureSignals {
        let rel = self.bank.parts().rel;
        let mut driver_free = self.dev.graphs.dev_ctx_face().vram_snapshot().free;
        if let Some(p) = self.peer.as_ref() {
            driver_free = driver_free.min(p.graphs.dev_ctx_face().vram_snapshot().free);
        }
        PressureSignals {
            pool_free_pages: rel.slots_free(),
            pool_total_pages: rel.slots_total(),
            host_free_slots: rel.host_free_slots() + self.bank.parts().rel.snap_host_free_slots(),
            host_kv_enabled: rel.host_enabled(),
            host_snap_enabled: self.bank.parts().rel.snap_host_budget() > 0,
            host_kv_total_slots: rel.host_total_slots(),
            host_snap_total_slots: self.bank.parts().rel.snap_host_total_slots(),
            host_kv_free_slots: rel.host_free_slots(),
            host_snap_free_slots: self.bank.parts().rel.snap_host_free_slots(),
            disk_free_slots: self.bank.parts().rel.kv_disk_free_slots()
                + self.bank.parts().rel.snap_disk_free_slots(),
            disk_kv_free_slots: self.bank.parts().rel.kv_disk_free_slots(),
            disk_snap_free_slots: self.bank.parts().rel.snap_disk_free_slots(),
            driver_free_bytes: driver_free,
            reserve_bytes: self
                .boot
                .cfg
                .knobs
                .vram_reserve_mb
                .saturating_sub(owl_shared::config::POOL_GRAPH_WARMUP_RESERVE_MB)
                << 20,
        }
    }

    /// **紧急池压力目标**（`PoolExhausted` 后；不是每拍）。
    ///
    /// 判据/水位/上限**全在 `owl_store::PressurePolicy`**（2026-11-16 裁决：
    /// 这些不是操作员旋钮）；这里只取现场数值并执行。状态腿在宿主有位时
    /// 带一条快照 —— 字节数只有 `Rel` 知道（`snap_bytes()`）。
    fn pressure_need(&self, signals: &PressureSignals) -> (usize, usize, usize, usize) {
        let p = owl_store::PressurePolicy::new();
        let slots = p.relief_pool_slots(signals.pool_total_pages, signals.pool_free_pages);
        let snap_bytes = if slots > 0
            && signals.host_free_slots > 0
            && self.bank.parts().rel.state_stats().snap_gpu.0 > 0
        {
            self.bank.parts().rel.snap_bytes()
        } else {
            0
        };
        let (host_kv, host_snap) = p.host_targets(signals);
        (slots, snap_bytes, host_kv, host_snap)
    }

    /// **H5 空转整理节拍**（Idle pump 计数；每 N 拍问一次）。
    ///
    /// **要不要整理由 store 回答**（2026-11-16 裁决，`runtime.idle_compact`
    /// 键已删）：低水位才触发、补到高水位；余量健康时目标恒 0 ⇒ 空转不做
    /// 任何搬运。这里只负责"到点了吗 + 把目标交给执行臂"。
    fn idle_tick(&mut self) -> Option<(usize, usize, usize)> {
        if !self.hk.note_idle_pump() {
            return None;
        }
        let p = owl_store::PressurePolicy::new();
        let signals = self.pressure_signals();
        let slots = p.idle_pool_slots(signals.pool_total_pages, signals.pool_free_pages);
        let (mut host_kv, mut host_snap) = p.host_targets(&signals);
        if host_kv > 0 || host_snap > 0 {
            if self.hk.host_pressure_due(p.host_interval()) {
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
    // 结构性豁免(2026-11-02 clippy 清理):类型面 = 派发/表项签名;起别名会藏住"这一串到底是什么"
    #[allow(clippy::type_complexity)]
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

    /// **逐 rank 设备显存实况**(只读;`/debug/vram` 的驱动残差/闭合账):
    /// `(free, total)`。TP2 必须逐卡取 —— 只读 rank0 会把另一张卡算漏。
    pub async fn mem_info_all(
        &mut self,
    ) -> Vec<(u8, std::result::Result<(u64, u64), String>)> {
        let mut out: Vec<(u8, std::result::Result<(u64, u64), String>)> = Vec::new();
        let r = self
            .dev
            .graphs
            .session
            .face_mut()
            .mem_info()
            .await
            .map_err(|e| e.to_string());
        out.push((0, r));
        if let Some(p) = self.peer.as_mut() {
            let r = p
                .graphs
                .session
                .face_mut()
                .mem_info()
                .await
                .map_err(|e| e.to_string());
            out.push((1, r));
        }
        out
    }

    /// **设备显存实况**(rank0;兼容旧调用面):`(free, total)`。
    pub async fn mem_info(&mut self) -> std::result::Result<(u64, u64), String> {
        self.mem_info_all()
            .await
            .into_iter()
            .next()
            .map(|(_, r)| r)
            .unwrap_or_else(|| Err("mem_info: 无 rank 读数".into()))
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
    pub(crate) fn set_ctx_after_turn(&mut self, session_id: u64) {
        self.set_dev_ctx(
            owl_iface::handle::Owner::Session(session_id),
            owl_iface::handle::AssetClass::Unclassified,
        );
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
                    let tags = crate::img_evict::block_tags_pos(
                        t.prompt_ids.len(),
                        page,
                        &t.img_spans,
                        &t.positions,
                    );
                    let (hit, recall) = self.bank.parts_mut().rel.match_prefix_tagged(&t.prompt_ids, &tags)?;
                    self.bank.parts().rel.reuse_face(&hit, &recall, plen, page).1
                };
                let have = self
                    .bank.parts().sessions.get(sid)
                    .map(|s| s.block_table.len())
                    .unwrap_or(0);
                // `dev`（复用面里在设备的块）与 `have`（会话当前表）可能指向
                // **同一批前缀**：直接相减会双重扣减（2026-11-06 真机：need=0
                // ⇒ 一拍不腾位 ⇒ ensure 撞墙）。取两者上限（安全下界；精确缺口
                // 由 `start_one → ensure_start_chain` 在 plan 之后补算）。
                return Ok(owl_store::Need::kv(
                    slots_for(plen).saturating_sub(dev.max(have)),
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
                crate::engine::PREFILL_TAIL_MERGE,
                Self::prefill_align_page(
                    self.boot.cfg.knobs.prefix_cache,
                    self.dev.pool.geom.page,
                ),
                &act.msg_snaps,
                &act.evict_snaps,
                crate::engine::SNAP_MSG_MIN_SPAN,
            ) {
                Some(chunk) => act.fed + chunk,
                None => self.generation_target(act.fed), // 生成相位(act.fed = pos)
            };
            let have = self
                .bank.parts().sessions.get(act.session_id)
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
        generation_target_for(pos, self.dev.spec_rt.depth, self.generation_kind())
    }

    /// 本进程的草稿档位（`generation_target_for` 的档位入参真源）。
    fn generation_kind(&self) -> GenerationKind {
        // 族 = 数据(trait 单点);这里只做三态映射,不再模式匹配变体
        match self.dev.spec_rt.drafter.as_ref().map(|d| d.family()) {
            Some(crate::spec_runtime::SpecMode::DFlash2) => GenerationKind::DFlash2,
            Some(crate::spec_runtime::SpecMode::Mtp) => GenerationKind::Mtp,
            _ => GenerationKind::None,
        }
    }

    /// **准入 alloc 面（块表可消费总槽）** —— 单源：
    /// [`chain_alloc_tokens`] 的 token 上界按页宽折算。
    /// 超卖档的准入面只是工作集，但块表仍会长到整条链 ⇒ alloc 必须覆盖全程。
    pub(crate) fn chain_alloc_slots(&self, prompt_tokens: usize, max_new: usize) -> usize {
        let page = self.dev.pool.geom.page.max(1);
        let tokens = chain_alloc_tokens(
            prompt_tokens,
            max_new,
            self.dev.spec_rt.depth,
            self.generation_kind(),
        );
        tokens.div_ceil(page).max(1)
    }

    /// **N4 池清点**(缺口 4 的节点面):把**已在 store 池里**的 `Turn(t)` 名下块
    /// 改归属 `Cache`(池 = 共享复用面,不再是某 turn 的临时面),返回 `(块数, 字节)`。
    ///
    /// 安全性:池成员**按构造**无活持有者(只在 `free` 路径入池)⇒ 换归属不改任何
    /// 持有事实;而"真没归还"的块(不在池里)**不换** —— 它们继续归 `Turn(t)`,
    /// 继续在 N4 清单里点名 ⇒ **"池内复用"与"泄漏"从此可分**。
    /// **N4 未归还清单**(turn 收尾按块清点 `Turn(t)` 名下仍开着的块)。
    ///
    /// 2026-11-15 起 store 不再有临时复用池：任何仍开着的块都是**真未归还**，
    /// 不再有 "pooled=true 非泄漏" 的豁免分支。
    pub(crate) fn n4_report_unreturned(&self, turn: u64) {
        let cell = self.bank.parts().rel.ledger_cell();
        let loans =
            owl_store::book_lock(&cell).open_loans(|o| o == owl_iface::handle::Owner::Turn(turn));
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
        let small = loans.len() <= 4 && total_kib <= 64;
        if small {
            owl_shared::log_debug!(
                "store",
                "N4 残块(兜底将清):turn {turn} {} 块 / {total_kib} KiB —— {}",
                loans.len(),
                detail.join(" ｜ ")
            );
            return;
        }
        owl_shared::log_warn!(
            "store",
            "N4 未归还清单:turn {turn} 名下仍有 {} 块 / {total_kib} KiB 开着 —— {}",
            loans.len(),
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
        let mut keep: Vec<u64> = self
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
        // **DFlash2 延迟 encode 的 tap 根块**同样跨 chunk 存活（2026-11-13）：
        // 它们 Owner::Turn(t)，但 encode 挪到首 token 之后 ⇒ 不能在本边界被
        // 当“上一 chunk 残块”清掉。双 rank 的 id 都进 keep（按 rank 过滤在
        // drain_owner_except 内做）。
        keep.extend(self.dev.pending_dflash.iter().map(|p| p.root.id));
        if let Some(p) = self.peer.as_ref() {
            keep.extend(p.pending_dflash.iter().map(|p| p.root.id));
        }
        // **逐 rank 分账**:两 rank 共用一本账,但块在**各自的 server** 块表里 ——
        // 把 rank1 的 id 交给 rank0 的 server = 找不到 ⇒ 静默漏(实测 434/1010)。
        let mut total = (0usize, 0usize);
        for rank in 0..2u8 {
            let (ids, bytes) = {
                let cell = self.bank.parts().rel.ledger_cell();
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
        self.bank.parts().rel.report()
    }

    /// **状态腿淘汰序预览**(诊断面;与容量调度同一份判据)。
    pub fn snap_evict_preview(&self, limit: usize) -> Vec<owl_store::SnapEvictPreview> {
        self.bank.parts().rel.snap_evict_preview(limit)
    }

    /// **L1 会话资产只读投影**(2026-11-12 三层架构):每会话 KV 前缀长度 /
    /// 尾锚点快照 / 消息边界。`/debug/vram` 的 `sessions` 面;零副作用。
    pub fn session_assets(&self) -> Vec<crate::SessionAssets> {
        self.bank.parts().sessions.assets()
    }

    /// **KV 腿报告**(槽/缓存/宿主/搬运计数 + 可动面;§13.4 对账口)。
    pub fn leg_stats(&self) -> owl_store::LegStats {
        self.bank.parts().rel.leg_stats()
    }

    /// **池账同步**(S3b §23.2):把池实占按"活跃面 / 可动面"拆给准入账 ——
    /// 真源 = 块腿报告的设备驻留字节 + 活跃 turn 的块表(**按槽号去重**)。
    ///
    /// 恒等式(必须成立,否则判据就偏):
    /// `reserved(唯一活跃实占) + session_held + cache = 池实占`。
    /// 共享前缀出现在多个 turn 的块表里时,保留面只计一次。
    pub(crate) fn sync_pool_account(&mut self) -> Result<()> {
        // 会话账/块表/准入账同屋（2026-11-15 下沉）：engine 只给活跃 turn
        // 的 `(turn, session)` 事实，slots/持有面/可动面全由 store 从自己的
        // 会话账与块腿报告推导。
        let active: Vec<(u64, u64)> = self
            .turns
            .actives()
            .map(|a| (a.id, a.session_id))
            .collect();
        self.bank.sync_pool_account(&active)?;
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
        cached_tokens: usize,
        page: usize,
        page_bytes: usize,
        granted_bytes: usize,
    ) {
        let new_ctx = ctx_tokens.saturating_sub(cached_tokens);
        let req_pages = floor.floor_pages(new_ctx, page);
        let got_pages = granted_bytes.div_ceil(page_bytes.max(1));
        let ctx_pages = ctx_tokens.div_ceil(page.max(1));
        owl_shared::log_info!(
            "store",
            "准入：turn {turn_id} 地板请求 {req_pages} 页 / **实得 {got_pages} 页**\
             (={} KiB)‖ ctx {ctx_tokens} tok（复用 {cached_tokens} / 新增 {new_ctx}）\
             = {ctx_pages} 页 ‖ 档 {}",
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
    /// **准入第一段：地板/腾位判决**（F-20 的分工注释）。
    ///
    /// 两套判决并存且顺序固定：
    /// 1. 本函数（pump 循环头）= `rel.check`：**per-turn 显存地板**
    ///    （Admit / EvictThen / Queue / Reject）；
    /// 2. `sched_policy::Admission::decide`（`schedule` 内）= **槽位/同会话/
    ///    退避**（Admit / Hold / Reject）。
    ///
    /// 本函数 Reject/Queue 后本拍不再进 `decide`；本函数 Admit 只表示
    /// "地板可兑现"，槽位仍可能 Hold。合并两套状态机是挂账（F-20），
    /// 在合并前不许在别处再加第三处准入判据。
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
        // 判**本拍可起跑的候选**（不是队首 —— 队首可能被同会话串行挡住；
        // 判错对象会把一个还不能跑的 turn 误拒/误腾位）。P4。
        let Some((turn_id, ctx_tokens, cached_tokens, floor)) =
            self.startable_queue_idx().map(|i| {
                let t = &self.turns.queue[i];
                (
                    t.id,
                    // 上界 = prompt + 生成长度 + spec 深度(安全方向:多算只是多腾几个槽)
                    t.prompt_ids.len() + t.spec.max_new + self.dev.spec_rt.depth,
                    // 已驻留的会话前缀：它已经在 `session_held` 里，地板只算新增段。
                    t.cached.min(t.prompt_ids.len()),
                    t.opts
                        .dense_floor
                        .unwrap_or(self.boot.cfg.knobs.dense_floor),
                )
            })
        else {
            return Ok(());
        };
        // **per-turn 地板的上限校验**(A7-bis):视图只能保持"本档工作集"那么多页
        // 常驻 ⇒ 请求地板 > 工作集 = 承诺无法兑现 ⇒ **响亮拒**(不静默夹）。
        // 只有 ctx 长度到了这里才知道,所以校在这里而不是 submit 处。
        let page = self.dev.pool.geom.page.max(1);
        let page_bytes = self.dev.pool.block_bytes().max(1);
        let ws_pages = self.boot.cfg.knobs.select_blocks.max(1);
        let new_ctx_tokens = ctx_tokens.saturating_sub(cached_tokens);
        let req_pages = floor.floor_pages(new_ctx_tokens, page);
        if !floor.is_strict() && req_pages > ws_pages {
            let reason = format!(
                "per-turn 地板 {req_pages} 页 > 本档工作集 {ws_pages} 页(视图只能保这么多常驻)\
                 —— 调大 [kvmem].workspace_per_req 或调小请求地板"
            );
            self.reject_queued_turn(
                turn_id,
                "no_memory",
                format!("准入拒绝：{reason}"),
            )?;
            return Ok(());
        }
        match self.bank.parts().rel.check(floor, new_ctx_tokens) {
            AdmissionVerdict::Admit { floor_bytes } => {
                self.turns.admit_wait = None;
                // **实得地板回吐**(设计卷 §2.4 报告面):请求页 / 实得页 / 上下文页。
                self.log_floor_grant(
                    turn_id,
                    floor,
                    ctx_tokens,
                    cached_tokens,
                    page,
                    page_bytes,
                    floor_bytes,
                );
            }
            AdmissionVerdict::EvictThen {
                need_bytes,
                floor_bytes,
            } => {
                // 抬需求到"腾够就能进":同一次段边界搬完(下沉不丢内容)
                //
                // ⚠️ `need_bytes` 是对 `face` 的**净值**(已把空闲面算进抵扣),
                // 而 `plan_capacity` 非 force 分支会再减 `slots_free` ⇒ 直接塞
                // `want` 会少腾 `slots_free` 个槽(准入反复判 EvictThen、每次都不够,
                // 拖到 `admission_max_wait_ms` 变 429)。这里换回"想要多少空闲槽"
                // 的口径(与 `step_need` / `ensure_start_chain` 同源)。
                let want = need_bytes.div_ceil(self.dev.pool.block_bytes().max(1));
                need.slots = need
                    .slots
                    .max(want.saturating_add(self.bank.parts().rel.slots_free()));
                self.log_floor_grant(
                    turn_id,
                    floor,
                    ctx_tokens,
                    cached_tokens,
                    page,
                    page_bytes,
                    floor_bytes,
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
                if waited >= self.boot.cfg.knobs.admission_max_wait_ms() {
                    self.reject_queued_turn(
                        turn_id,
                        "queue_timeout",
                        format!("准入拒绝：排队 {waited}ms 仍无法腾位（{reason}）"),
                    )?;
                }
            }
            AdmissionVerdict::Reject { reason } => {
                self.reject_queued_turn(
                    turn_id,
                    "no_memory",
                    format!("准入拒绝：{reason}"),
                )?;
            }
        }
        Ok(())
    }

    /// **排队 turn 的统一终局清理**（F-18）：排队 turn 没占设备资源
    /// （块链/GDN 行都在 `start_one` 才借），也不动会话账 —— 会话是持久资产
    /// （2026-11-16 内容身份），一条排队请求被取消/拒收不应抹掉历史。
    /// 三条终局（准入拒/准入超时/客户端取消）必须走同一处，否则又是一套语义。
    pub(crate) fn drop_queued_turn(&mut self, t: &crate::turn::Turn) {
        if self.turns.admit_wait.map(|(id, _)| id) == Some(t.id) {
            self.turns.admit_wait = None;
        }
        self.close_admission(t.id);
    }

    /// **拒一个待准入 turn**(§23.4):出队 + 临时会话焚毁 + `Failed` 事件入队。
    ///
    /// 不掉 `ensure_for_len` 的 `Exhausted` —— 拒的文案不带 "超预算" 字样:
    /// 那是"prompt + max_tokens > max_seq"的专用标记(服务面据此归一化成
    /// `context_length_exceeded` 触发客户端自动 compact);池墙拒是另一类,
    /// 客户端该做的是降并发/重试。
    fn reject_queued_turn(
        &mut self,
        turn_id: u64,
        // 码是静态词表（与调度判决 `Reason::code()` 同一套）；调用点全是字面量
        code: &'static str,
        why: String,
    ) -> Result<()> {
        let Some(pos) = self.turns.queue.iter().position(|t| t.id == turn_id) else {
            return Ok(()); // 已被别处取走(幂等)
        };
        let Some(t) = self.turns.queue.remove(pos) else {
            return Ok(()); // 竞态下已被别处取走（幂等）
        };
        self.drop_queued_turn(&t);
        owl_shared::log_warn!("store", "{why}（turn {turn_id}）");
        owl_shared::metrics::with_metrics_store(|m| {
            m.counter_add("store.admit.reject", 1, file!(), line!());
        });
        // 服务面按 `admission:<code>` 翻 429 + Retry-After；缺这个前缀会
        // 落进普通 500/SSE 错误路径，agent 客户端看起来就像“请求卡死”。
        self.turns.pending_events.push_back(TurnEvent::Failed {
            turn: turn_id,
            err: crate::turn::TurnError::rejected(crate::turn::AdmissionReject::new(
                code, None, why,
            )),
        });
        self.sync_pool_account()?;
        Ok(())
    }

    // ─────────────────────── 磁盘层执行面(S6;store 出计划 / 这里做 I/O) ───────────────────────

    /// **KV 主动 checkpoint（S6.2）**：把活跃会话已算出的 KV 块**复制**一份到盘层。
    ///
    /// 与旧路径（`spill_kv_to_disk`）的区别只有**驱动条件**：
    /// - 旧 = 宿主压力票 ⇒ 宿主够大时永不落盘 ⇒ 重启后索引在、内容不在（召回=0）；
    /// - 本 = **定期主动**（磁盘层启用后按 [`CheckpointPolicy`] 自动）。
    ///
    /// 三条纪律：
    /// 1. **复制语义**：设备块走只读 D2H（`plan_read_blocks`）后归还缓冲、不解绑；
    ///    宿主块只 clone 字节 —— 运行期驻留与账**一律不动**；
    /// 2. **只写连续前缀**：boot 的 `adopt_disk_chains` 以"内容在场"截断到连续前缀，
    ///    所以遇到"既不在设备也不在宿主"的洞（未算/已丢）立即停，**绝不跳块**；
    /// 3. **配额小步**：一次最多写 `quota_bytes`（按块向下取整），中断即停 ——
    ///    已写部分仍是合法前缀，多轮小步自然铺满整链。
    async fn checkpoint_session_kv(
        &mut self,
        tokens: &[u32],
        tags: &[owl_store::BlockTag],
        quota_bytes: usize,
    ) -> Result<CheckpointStats> {
        let active_sessions = self.active_session_ids();
        let mut out = CheckpointStats::default();
        if quota_bytes == 0 || !self.bank.parts().rel.disk_enabled() {
            return Ok(out);
        }
        // **索引与内容成对**（幂等补写；空转拍内容写盘时索引可能还没落）。
        {
            let bank = self.bank.parts_mut();
            if let Err(e) = self
                .disk
                .write_chain_index(bank.rel, tokens, tags, self.dev.pool.geom.page)
                .await
            {
                owl_shared::log_warn!("store", "checkpoint 侧索引补写失败(内容仍写):{e}");
            }
        }
        // **身份来源 = tokens**（store 侧统一重算）：盘上内容的键必须与
        // `ChainIndex` 同源；store 同时负责跳过盘上已有前缀、按真实 carrier
        // 分出 D2H / host clone，并在盘上-only 处截断（不再由 engine 拿
        // `KvStore` 的二值 `locate` 猜层）。
        let Some(plan) = self
            .bank
            .parts()
            .rel
            .plan_checkpoint_kv(tokens, tags, quota_bytes)?
        else {
            return Ok(out);
        };
        let blk = plan.block_bytes;
        let mut spill = crate::mem::disk_runtime::SpillStats::default();
        // ④ 宿主副本：clone 字节写盘（缓冲不动 = 复制语义）
        {
            let bank = self.bank.parts_mut();
            self.disk
                .spill_host_blocks(bank.rel, &plan.host, &active_sessions, &mut spill)
                .await?;
        }
        // ⑤ 设备块：批量只读 D2H → 交给 DiskRuntime 逐槽写盘 → 归还缓冲
        if !plan.dev.is_empty() {
            let owl_store::ReadPlan {
                chains,
                pieces,
                table,
                ..
            } = self.bank.parts_mut().rel.plan_read_blocks(&plan.dev)?;
            if !chains.is_empty() {
                let t = self
                    .swap_all_ranks(owl_iface::contract::SwapDir::Out, &pieces, table)
                    .await?;
                {
                    let bank = self.bank.parts_mut();
                    self.disk
                        .spill_device_table(
                            bank.rel,
                            &chains,
                            &t,
                            blk,
                            &active_sessions,
                            &mut spill,
                        )
                        .await?;
                }
                self.bank.parts_mut().rel.recycle_read_plan(t);
            }
        }
        out.dev_blocks = spill.dev_blocks;
        out.host_blocks = spill.host_blocks;
        let wrote = out.dev_blocks + out.host_blocks;
        out.bytes = wrote * blk;
        // ⑥ **尾部边界快照必须一起落盘**（否则召回等于 0）。
        //
        // 为什么：`reuse_face` 要求复用面的**尾边界有 GDN 快照**（混合模型的
        // 状态不能只靠 KV 块重建）；快照不在盘上 ⇒ `state.locate == None` ⇒
        // m 一路裁到 0 ⇒ `plan_fresh`（**实测：只 checkpoint KV 时，重启后
        // `cache=0/4753`，全量重算 2.4 s**）。
        //
        // 语义上快照用**搬家**（设备/宿主 → 盘），不是复制：一份 148 MiB，
        // 复制会白吃宿主；丢掉的快照是可重建的锚点（复用点变近，不是错值），
        // 下次命中由 `fill_snap_from_disk` 回填。
        // **窗口内所有 Durable 锚点都落盘**(2026-11-01 P1 收尾)。
        //
        // 旧实现只落**链尾那一张** ⇒ `reuse_face` 从命中面尾部往回找快照时,
        // 只有尾边界可用/或者干脆没有(实测 `m_raw=1398/3072`,尾快照层=None);
        // 复用面被削到 1/3。这里按"已写进盘的块前缀"逐个边界落盘,
        // 盘不够就停(配额由 snap_disk 管,谁先到顶谁截断)。
        // 部分写失败时只能保证 `already + 实际写成功` 的前缀；完整成功才
        // 用 store 计划的全链（含之前已在盘上的前缀）。
        let planned = plan.dev.len() + plan.host.len();
        let prefix_end = if wrote == planned {
            plan.chain.len()
        } else {
            (plan.already_on_disk + wrote).min(plan.chain.len())
        };
        out.snap_disk = self
            .checkpoint_prefix_snaps(&plan.chain[..prefix_end])
            .await?;
        Ok(out)
    }

    /// **turn 收口的 checkpoint 调度**（门 + 节拍 + 日志/指标）。
    async fn maybe_checkpoint_kv(
        &mut self,
        sid: u64,
        tokens: &[u32],
        tags: &[owl_store::BlockTag],
    ) -> Result<()> {
        let policy = self.boot.cfg.knobs.checkpoint_policy;
        // 策略是内部维护：没有配置开关；磁盘层没接线或测试显式关路径才跳过。
        if policy.step_bytes == 0 || !self.bank.parts().rel.disk_enabled() {
            return Ok(());
        }
        if !policy.min_gap.is_zero() && !self.hk.kv_checkpoint_due(policy.min_gap) {
            return Ok(());
        }
        self.hk.mark_kv_checkpoint();
        let t0 = std::time::Instant::now();
        let st = self
            .checkpoint_session_kv(tokens, tags, policy.step_bytes)
            .await?;
        if st.blocks() > 0 || st.snap_disk {
            owl_shared::log_info!(
                "store",
                "KV checkpoint:session={sid:#x} 写 {} 块 / {} MiB(设备 {} · 宿主 {})· 锚点快照 {} in {:?}",
                st.blocks(),
                st.bytes >> 20,
                st.dev_blocks,
                st.host_blocks,
                if st.snap_disk { "已落盘" } else { "无(该链尚无 Durable 锚点)" },
                t0.elapsed()
            );
            owl_shared::metrics::with_metrics_store(|m| {
                m.counter_add("store.disk.checkpoint.blocks", st.blocks() as u64, file!(), line!());
                m.counter_add("store.disk.checkpoint.bytes", st.bytes as u64, file!(), line!());
                if st.snap_disk {
                    m.counter_add("store.disk.checkpoint.snap", 1, file!(), line!());
                }
            });
        }
        Ok(())
    }

    /// **把已 checkpoint 前缀里的所有 Durable 锚点落盘**(2026-11-01)。
    ///
    /// 为什么不能只落尾:`reuse_face` 要求复用面的**尾边界有快照**,
    /// 而它从命中面尾部往回走到最近一个在场锚点 —— 只有尾锚点时复用面
    /// 会被削到口径之外(实测 3072 → 1398 块)。逐边界落盘 ⇒ 每个
    /// 页对齐的锚点都能当复用点。
    ///
    /// 顺序:自尾向头(近端优先,盘配额先给最近用得上的);
    /// 配额到顶即停(`spill_snap_to_disk` 返回 false),不静默扩容。
    async fn checkpoint_prefix_snaps(&mut self, prefix: &[owl_store::ChainHash]) -> Result<bool> {
        use owl_store::state::SnapDurability;
        let active_sessions = self.active_session_ids();
        use owl_store::Tier;
        if !self.bank.parts().rel.snap_disk_enabled() {
            return Ok(false);
        }
        let mut any = false;
        for h in prefix.iter().rev() {
            let key = owl_store::SnapKey::at_boundary(*h);
            if self.bank.parts().rel.snap_durability(key) != Some(SnapDurability::Durable) {
                continue;
            }
            let moved = match self.bank.parts().rel.state_locate(key) {
                Some(Tier::Disk) => true, // 已在盘(=可当复用点)
                Some(Tier::Host) => {
                    let rel = self.bank.parts_mut().rel;
                    self.disk.spill_snap(rel, key, &active_sessions).await?
                }
                Some(Tier::Gpu) => {
                    let Some(slot) = self.dev.pool.snap_slot_of(key) else {
                        continue;
                    };
                    self.snap_offload(key, slot).await?;
                    self.dev.pool.snap_clear_slot(slot);
                    if let Some(pr) = self.peer.as_mut() {
                        pr.pool.snap_clear_slot(slot);
                    }
                    let rel = self.bank.parts_mut().rel;
                    self.disk.spill_snap(rel, key, &active_sessions).await?
                }
                None => continue, // 已被取代/逐出
            };
            if moved {
                any = true;
            } else {
                // 配额到顶(或该张已不 Durable)⇒ 近端已尽力,停
                owl_shared::log_info!(
                    "store",
                    "锚点落盘在边界 {:?} 停(盘配额到顶,前缀里其余锚点下次再落)",
                    h
                );
                break;
            }
        }
        Ok(any)
    }

    /// **写盘上链索引**(S6.1;turn 收口一次,同 tail 幂等)。
    ///
    /// 内容可能**还没溢写到盘** —— 索引先写;`boot` 以"内容文件在场"截断到连续前缀
    /// (没有内容的索引 = 孤儿,boot 会清掉内容/跳过)。8k token 的链 ≈ 32 KB。
    /// **宿主腿一条 swap 落到每个 rank**(2026-10-20;双卡 KVMem 立案 §23.8)。
    ///
    /// `pieces[p]` = store 用**第 p 个 rank 自己的块 id** 组的片(两侧池句柄是两套,
    /// 不能靠偏移平移);宿主槽 = `world` 个平面,偏移由 store 的 `out/in_pieces`
    /// 按 `plane × 单页字节` 落好。
    /// 槽台 = **一次借台**,两条命令之间原箱传递 ⇒ 一次落账、一条链一份预算。
    /// 单卡(`world = 1`)= 与历史逐字相同。
    // 结构性豁免(2026-11-02 clippy 清理):循环变量是**维度序**(多数组同索引 / 手工归约要显式序);zip 会藏住对应关系
    #[allow(clippy::needless_range_loop)]
    pub(crate) async fn swap_all_ranks(
        &mut self,
        dir: owl_iface::contract::SwapDir,
        pieces: &[Vec<owl_iface::contract::SwapPiece>],
        table: owl_store::SlotTable,
    ) -> Result<owl_store::SlotTable> {
        let world = self.bank.parts().rel.world().max(1);
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
        self.bank.parts().rel.world().max(1) == 1
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

    /// **装配后置失败时的收尾**（F-05）：只关图族（staging 的 PeerRelease
    /// 需要 `TpEngine` 的 GpuClient 特化面，不在此处）。给 `Engine::run`
    /// 在 prewarm 失败时用，避免已捕获的图族被无声 Drop。
    pub(crate) async fn close_graphs_on_boot_failure(mut self) {
        if let Err(e) = self.dev.graphs.close().await {
            owl_shared::log_warn!("boot", "装配失败收尾:rank0 图族关闭失败: {e}");
        }
        if let Some(peer) = self.peer.as_mut() {
            if let Err(e) = peer.graphs.close().await {
                owl_shared::log_warn!("boot", "装配失败收尾:rank1 图族关闭失败: {e}");
            }
        }
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
        let st = self.bank.parts().rel.leg_stats();
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
                    self.bank.parts_mut().rel.abort_offload(&pairs);
                    if let Some(s) = self.bank.parts_mut().sessions.get_mut(p.sid) {
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
        let n = self.bank.parts_mut().rel.commit_offload(&pairs, t)?;
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
        let active_sessions = self.active_session_ids();
        use owl_store::Sink;
        let mut a = owl_store::Applied::default();
        // ① 换出/驱逐(**两腿同单**:KV 链/叶 + GDN 快照;§30)
        for item in &verdict.out {
            match item {
                owl_store::EvictOut::Kv { chain, sink, span } => match sink {
                    Sink::Host => {
                        let relief0 = self.bank.parts().rel.host_relief_blocks();
                        let (blocks, pieces, table) = {
                            match self.bank.parts_mut().rel.plan_kv_offload(*chain) {
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
                            .bank.parts().rel.host_relief_blocks()
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
                                self.bank.parts_mut().rel.abort_offload(&blocks);
                                return Err(e);
                            }
                        };
                        a.offloaded +=
                            self.bank.parts_mut().rel.commit_offload(&blocks, table)?;
                    }
                    // **整枝塌缩**:一张票可带 `span` 块(叶→根,到分叉点为止);
                    // 逐块校验引用(活跃面停),丢掉的块数如实记账。
                    Sink::Drop => match self.bank.parts_mut().rel.prune_branch(*chain, (*span).max(1)) {
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
                        let blocks = self.bank.parts().rel.chain_blocks(*chain);
                        let mut n = 0usize;
                        for b in blocks.into_iter().take((*span).max(1)) {
                            let rel = self.bank.parts_mut().rel;
                            if self.disk.spill_kv(rel, b, &active_sessions).await? {
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
                        self.bank.parts_mut().rel.drop_snap_host(*key)?;
                        match self.bank.parts_mut().rel.drop_snap_account(*key) {
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
                        let rel = self.bank.parts_mut().rel;
                        if self.disk.spill_snap(rel, *key, &active_sessions).await? {
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
                match self.bank.parts_mut().rel.plan_onload(std::slice::from_ref(chain)) {
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
                    self.bank.parts_mut().rel.abort_onload(&cs, &slots);
                    return Err(e);
                }
            };
            let _ = self.bank.parts_mut().rel.commit_onload(&cs, &slots, table)?;
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
    pub(crate) async fn pool_housekeeping(
        &mut self,
        need: owl_store::Need,
        extra_snaps: &[owl_store::SnapKey],
    ) -> Result<owl_store::Applied> {
        // 两条腿各自"宿主关 ⇒ 该腿目标清零"(旧行为:宿主关 = 不换出,**也不做无谓驱逐**;
        // 两腿独立,不能因为状态腿开就顺手把 KV 面 Drop 掉)
        let mut need = need;
        if !self.bank.parts().rel.host_enabled() {
            need.slots = 0;
        }
        if self.bank.parts().rel.snap_host_budget() == 0 {
            need.snap_bytes = 0;
        }
        if need.is_zero() {
            return Ok(owl_store::Applied::default());
        }
        // 没有“活跃会话快照”特权；只有本次操作正在载入的 key 不入 victim。
        let protect = owl_store::Protected {
            snaps: extra_snaps.to_vec(),
            ..owl_store::Protected::default()
        };
        let verdict = self.bank.parts().rel.plan_need(need, &protect);
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
        // 恒开（P0.3 定谳：纳秒级打点；`prof` 只控制下面的分相打印）——
        // 不用 `(prof || true)` 那种让人误以为有条件的形式。
        let t_loophead = Some(std::time::Instant::now());
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
        let t_sched = Some(std::time::Instant::now());
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
                // **待落盘的 KV checkpoint**(2026-11-01 收尾):空转拍执行,
                // 一拍一个(不排队)。失败只 WARN —— 落盘是尽力而为,
                // 不许把空转打断(下一拍还有机会,内部 `CheckpointPolicy.min_gap` 另管节拍)。
                if let Some((sid, toks, tags)) = self.hk.take_kv_checkpoint() {
                    if let Err(e) = self.maybe_checkpoint_kv(sid, &toks, &tags).await {
                        owl_shared::log_warn!("store", "空转 checkpoint 失败(session={sid:#x}): {e}");
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
        let id: u64 = self.submit(prompt, max_new)?;
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

    /// **位置压实触发边界**（2026-11-08 图片 gap 预留）：
    /// 安全条件 = `pos_hi + max_new < pos_cap`；`==` 已越界。
    #[test]
    fn consolidates_only_when_positions_would_overflow() {
        use super::needs_consolidation;
        // 262144 ctx + 0.5 预留 = 393216；leading gap 用到 393200 时：
        assert!(needs_consolidation(393_200, 16, 393_216), "末位 393216 越界");
        assert!(!needs_consolidation(393_200, 15, 393_216), "末位 393215 安全");
        // 边界：末位 = pos_cap-1 安全；= pos_cap 触发
        assert!(!needs_consolidation(98, 1, 100), "末位 99 安全");
        assert!(needs_consolidation(99, 1, 100), "末位 100 越界");
        assert!(!needs_consolidation(99, 0, 100), "prompt 末位 99 安全");
        assert!(needs_consolidation(100, 0, 100), "prompt 本身已越界");
        // 非 gap 会话：pos_hi = n_tokens ≤ max_seq < pos_cap ⇒ 永不触发
        assert!(!needs_consolidation(262_144, 0, 393_216));
        // 负数/异常账不得 panic，按 0 处理
        assert!(!needs_consolidation(-5, 0, 10));
    }

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
    use super::{chain_alloc_tokens, generation_target_for, GenerationKind};

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

    /// **alloc 面必须覆盖 spec 滑越**（2026-10-28 `grant 额度不足` 回归门）。
    ///
    /// 实测失败现场：`prompt=111340 / max_new=16 / depth=7 / DFlash2 /
    /// page=32`。最后一个产出位 = 111355，最后一个 spec 轮还要滑到
    /// `111355 + 16 = 111371` ⇒ 实需 `ceil(111371/32) = 3481` 槽；
    /// 旧式 `ceil((prompt+max_new)/32)` 只给 3480 ⇒ 生成到第 12 个 token 撞墙。
    #[test]
    fn alloc_face_covers_spec_round_overflow() {
        let (prompt, max_new, d, page) = (111_340usize, 16usize, 7usize, 32usize);
        let naive = prompt
            .saturating_add(max_new)
            .div_ceil(page);
        let alloc = chain_alloc_tokens(prompt, max_new, d, GenerationKind::DFlash2).div_ceil(page);
        assert_eq!(naive, 3480, "旧式（少一页）");
        assert_eq!(alloc, 3481, "上界必须含 verify+propose 滑越");
        // 裸 decode（depth = 0）不滑越：与旧式**逐位相等**（不得白涨池占用）。
        for (p, m) in [(0usize, 1usize), (32, 32), (4096, 512), (111_340, 16)] {
            let alloc0 = chain_alloc_tokens(p, m, 0, GenerationKind::None).div_ceil(page);
            assert_eq!(
                alloc0,
                p.saturating_add(m).div_ceil(page).max(1),
                "depth=0 时 alloc 面 = 旧式（{p}+{m}）"
            );
        }
    }
}

