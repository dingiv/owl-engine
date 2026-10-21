//! FlashInfer paged prefill runtime(FFI 直调链路;plan/run 分离)。
//!
//! init:workspace 预钉(float 64MB + int 32MB;构造期,捕获内首次初始化
//! 事故类结构性消灭)。run:plan(每形状缓存,键 = 形状七元组)→ run
//! (f16/fp8kv 按名路由)。

use std::collections::HashMap;

use crate::contract::{Bytes, Dtype, InvariantBox, LaunchMsg, OpError, OpId};
use crate::device::ScratchBuf;

use crate::client::flashinfer::{parse_prefill, PREFILL_FI};
use crate::family::flashinfer::{
    owl_fi_prefill_plan, owl_fi_prefill_run, owl_fi_prefill_run_fp8kv,
    FI_FLOAT_WS_BYTES, FI_HOST_STAGING_BYTES, FI_INT_WS_BYTES,
};
use crate::registry::{FamilyGuard, KernelFamily};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct PlanKey {
    total_rows: usize,
    /// **ctx 恒进键**(2026-10-18 定谳;原实现在这里埋了雷):
    ///
    /// 旧写法是 `ctx_total: if split { ctx } else { 0 }`,其中 `split` 读
    /// env `FI_SPLIT_KV` 且**缺省判 false**,而核侧 `owl_fi_split_kv_enabled()`
    /// **缺省判 true**(只有 `=0` 才关)⇒ **两把尺不同源**:引擎以为"ctx 无关"
    /// 而 planner 实际按 ctx 分块拆分 ⇒ 键把 ctx 抹平后,**第一次算的那张 plan
    /// (boot warmup 的 ctx=1/nb_pages=1,连 split 都无从谈起)被全生命周期复用**。
    /// 实测(单 3090 Ti,27B,fp8 池,512→32k):默认档 32k 裸 decode **38.17 ms/tok**,
    /// 与显式 `FI_SPLIT_KV=0` 档(38.13)**逐字同值** = 默认档等于"无 split 档";
    /// 正确键控(ctx 进键)后 32k **23.61 ms/tok ⇒ 1.62×**。
    ///
    /// 现口径:**不看 split 开关,ctx 一律进键**(桶化后的值,见 `plan_bucket`)。
    /// split 语义唯一来源 = 核侧 `owl_fi_split_kv_enabled()`(本侧不再复刻)。
    ctx_total: usize,
    /// **图档 plan**(`FI_CUDA_GRAPH`;见 common.h `owl_fi_cuda_graph`)
    /// —— 数组与 `padded_batch_size` 都与 eager 档不同,必须分键,否则
    /// 一次 eager 调用会把图档 plan 挤掉(mask 内容随之失配)
    graph: bool,
    t: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    page: usize,
}

/// 图档 plan 开关(**与 C++ `owl_fi_cuda_graph` 同源**)。
///
/// 2026-10-21 收编:C++ 适配器只认 `getenv("FI_CUDA_GRAPH")`(上游 ABI 不动),
/// 而本侧**不再自己读 env** —— 唯一写点 = 后端 [`set_plan_mode`]
/// (C++ env 与本地原子同一次写入,两份口径不可能漂移)。
/// plan 期读一次(非热路径)。
static CUDA_GRAPH: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// 写图档 plan 开关(与 C++ env 同值一次落;boot 期调用)
pub fn set_plan_mode(cuda_graph: bool) {
    CUDA_GRAPH.store(cuda_graph, std::sync::atomic::Ordering::Relaxed);
}

fn fi_cuda_graph() -> bool {
    CUDA_GRAPH.load(std::sync::atomic::Ordering::Relaxed)
}

/// **plan 键归一化(纯函数;可单测)**:ctx 恒进键。
///
/// 桶化(把 ctx 上取整到桶再进键)**不在这里做** —— 值由引擎侧写进
/// 宿主标量槽(`[dispatch] fi_plan_bucket`;见 `exec/phases.rs` 的 `plan_ctx`),
/// 核侧拿到的 `call.ctx_total` 已是桶化值:单一事实源,本侧零 env。
fn plan_key(
    total_rows: usize,
    ctx_total: usize,
    t: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    page: usize,
    graph: bool,
) -> PlanKey {
    PlanKey {
        total_rows,
        ctx_total,
        t,
        hq,
        hkv,
        hd,
        page,
        graph,
    }
}

/// **图自携 plan restore** 的进程级开关(缺省 `true` = 历史行为)。
///
/// 语义(2026-10-18 W2 定稿;**只此一处**,不再是 env):
///   - `true`  ⇒ 捕获窗内把"本 plan 的表 → int_ws"烘成一条 memcpy 节点
///                (＝**图自携表**;prefill 桶图靠它);
///   - `false` ⇒ 窗内不烘 ⇒ **host 成为表的唯一权威**:引擎每次图 launch 之前
///                用 [`FiRuntime::plan_sync`] 走窗外路径刷当步精确表。
///
/// ⚠️ 射程曾经是错的:旧实现(env `OWL_FI_NO_PLAN_RESTORE`)为**取证档**,
/// 语义 = "关自携 ⇒ 切换后不重刷",在 `activate` 里**无条件 return** ⇒
/// 把**窗外重刷**也一起掐了 ⇒ 单图档下表永不上传(实测:输出 64 个空 token)。
/// 现口径:**只影响窗内那一烘**,窗外路径不受它管。
///
/// 写者唯一 = 后端 boot(配置 `[dispatch] fi_plan_restore` → `DiagOpts` →
/// `set_plan_restore`);本 crate 的 lib 零依赖律 ⇒ 不经 env。
static PLAN_RESTORE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// 设置"图自携 plan restore"开关(仅 boot 期调用一次;见 [`PLAN_RESTORE`])。
pub fn set_plan_restore(on: bool) {
    PLAN_RESTORE.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// 读当前开关(热路径:一次 relaxed load)
fn plan_restore() -> bool {
    PLAN_RESTORE.load(std::sync::atomic::Ordering::Relaxed)
}

impl FiRuntime {
    /// **本族身份**(族名 = 线契约名;字面量住 `family::flashinfer`,零门)
    pub const ID: OpId = OpId(crate::family::flashinfer::PREFILL_FI);

    // 本族算子表:由 `KernelSpec::FAMILY` 派生(f16 + fp8 两臂)
}

/// plan 条目(M2b,2026-10-15):一个 `PlanKey` 一份。
///
/// **为什么从"单条缓存"升级成"条目表"**:多桶(prefill 图族)共存时,int_ws
/// 是**唯一共享区** —— 每次 `owl_fi_prefill_plan` 都会把调度表 memcpy 进同一个
/// int_ws,而图里烘的是 `int_ws + offset` 指针 ⇒ "最后一个 plan"决定所有图读到
/// 什么。条目表配 `staged` 不变盒后,**图自携 plan**:回放前先把 int_ws 刷回本
/// plan 的表(窗内那次刷 = 图里一个 memcpy 节点;见 [`FiRuntime::activate`])。
struct PlanEntry {
    plan15: [i64; 15],
    /// **不变盒:plan 表快照**(共享 staging 有效前缀的冻结拷贝)。
    /// 为什么不能直接用共享 `host_staging`:plan 的 H2D 源是**同一个可变缓冲**
    /// (换形状即重写)⇒ 窗内烘该 memcpy 的回放期会读到"最后一次 plan"的内容
    /// (错调度表;设计 §4.6 那条雷的同一族)。盒 = 构造即冻结、生命周期 ≥ 图。
    staged: InvariantBox,
    /// 尾部 CTA 掩码块(P-5)+ 内容 `i < tiles`;**`None` = 无需掩码**
    /// (`padded == tiles` ⇒ 内容恒全 1 = 空操作,传 null;核侧契约允许)。
    /// 为什么与 plan 同车:掩码是 `tiles/padded` 的函数;共用一条缓冲会被换形
    /// 调用改写 ⇒ 回放旧图掩码失配 ⇒ **真行漏算**(静默错值)。
    mask: Option<ScratchBuf>,
    /// 真 tile 数(诊断/日志)
    tiles: usize,
    /// 入表序号(容量治理用;见 [`FiRuntime::evict_stale`])
    born: u64,
    /// **图归属**(该 plan 在捕获窗内被烘进图 ⇒ 掩码块/表快照被图引用 ⇒
    /// 永不可驱逐;见 [`FiRuntime::activate`])
    graph_owned: bool,
}

pub struct FiRuntime {
    float_ws: Option<ScratchBuf>,
    int_ws: Option<ScratchBuf>,
    /// plan **共享** staging(plan 内部用;只是计量/复制的中转 —— 不要指望它
    /// 的内容在半途稳定,快照见 [`PlanEntry::staged`])
    host_staging: Vec<u8>,
    /// plan 条目表(key = 形状七元组 + 两开关位;桶数量级)
    plans: HashMap<PlanKey, PlanEntry>,
    /// **`active` = 上一次 *窗外* 显式上传的 key**(None = 未知/已失效)。
    ///
    /// ⚠️ **它不是"int_ws 现内容"的当前事实**(2026-10-17 定谳):图回放会用
    /// 窗内烘的 restore 节点把 int_ws 刷成**图自己的** plan,而那次改写 host
    /// 侧看不见 ⇒ 回放之后本字段是**旧账**。故它的唯一用途 = 窗内去重
    /// (避免同一窗烘冗余节点);**窗外调用不得凭它跳过重刷**。
    active: Option<PlanKey>,
    /// 捕获窗上升沿检测(见 `activate` 的调用点注释)
    last_capturing: bool,
    /// 入表序号发生器(容量治理)
    stamp: u64,
    /// 视觉塔 ragged 臂的 plan 缓存(2026-10-15;键 = (rows, heads, head_dim))
    plan_ragged: Option<(RaggedKey, [i64; 15])>,
    /// 视觉塔 ragged 的**规范 int32 段界**([0, rows];plan 期写一次,回放期只读)
    ragged_cu: Option<ScratchBuf>,
}
/// 视觉塔 ragged plan 键(单图契约 ⇒ 三件即可)
#[derive(Clone, Copy, PartialEq)]
struct RaggedKey {
    rows: usize,
    heads: usize,
    head_dim: usize,
}


/// plan 条目表容量软上限(超过就驱逐**非图归属**的最旧条目)
const PLAN_ENTRIES_SOFT_MAX: usize = 32;
/// 驱逐后水位(避免每次插入都驱逐)
const PLAN_ENTRIES_KEEP: usize = 16;

impl FiRuntime {
    /// plan 条目表**容量治理**:eager 路的 chunk 长逐值不同 ⇒ 键会缓慢堆积
    /// (每键 = plan15 + 表快照(~KB 主机)+ 掩码块(~百字节设备))。超过软上限
    /// 就驱逐**非图归属**的最旧条目(图引用的掩码/快照**永不动** —— 这是
    /// [`PlanEntry::graph_owned`] 的全部意义;驱逐错一个 = 回放读到已释放块)。
    fn evict_stale(&mut self) {
        if self.plans.len() <= PLAN_ENTRIES_SOFT_MAX {
            return;
        }
        let mut victims: Vec<(u64, PlanKey)> = self
            .plans
            .iter()
            .filter(|(k, e)| !e.graph_owned && Some(**k) != self.active)
            .map(|(k, e)| (e.born, *k))
            .collect();
        victims.sort_unstable_by_key(|(born, _)| *born);
        let drop_n = self.plans.len().saturating_sub(PLAN_ENTRIES_KEEP);
        for (_, k) in victims.into_iter().take(drop_n) {
            self.plans.remove(&k);
        }
    }

    /// 把 int_ws 刷成 `key` 的 plan 表(必要时)。
    ///
    /// 窗内调用 = 图里多一个 memcpy 节点 ⇒ **图自携 plan**(多桶/外部 plan
    /// 切换都不再能静默换掉它读的表);窗外调用 = 立即生效(便宜的一次 H2D)。
    fn activate(
        &mut self,
        key: PlanKey,
        res: &mut dyn crate::device::DeviceRes,
    ) -> Result<(), OpError> {
        let capturing = res.capturing().unwrap_or(false);
        // **关自携档**(2026-10-18 W2 的正式用途;原为取证档):
        //   - **窗内**:不烘 restore memcpy(⇒ 图不携带自己的表);
        //   - **窗外**:**照旧重刷**(这半边是 W3 的命脉 —— 引擎每步用
        //     `plan_sync` 走窗外路径把当步精确表刷进 int_ws;旧写法在此处
        //     也 `return`,把刷新一起掐了 ⇒ 表永不上传,实测输出 64 个空 token)。
        if !plan_restore() && capturing {
            // ⚠️ **图归属仍要登记**（2026-10-18）：本档图不带自携表，但图里烘的
            // `block_valid_mask` 指针仍可能指向本条目的掩码块（split 关时用 Rust
            // 侧那块）⇒ 条目被驱逐 = 图引用已释放块。登记 = 永不驱逐
            // （代价 = 每图常驻一条条目，数量 = 图数）。
            if let Some(e) = self.plans.get_mut(&key) {
                e.graph_owned = true;
            }
            self.active = Some(key);
            return Ok(());
        }
        // 窗内去重(同一捕获窗内同 key 只烘一个 restore 节点)。
        //
        // ⚠️ **窗外一律重刷**(`capturing == false` ⇒ 不跳过;2026-10-17 定谳):
        // 旧行为是`self.active == Some(key)` 就跳过 —— 该判据**只在窗外成立**,
        // 但**图回放会把 int_ws 刷成图自己的 plan**(窗内烘的 memcpy 节点),
        // 而 host 侧 `active` 对那次改写**不知情** ⇒ 回放之后同 key 的首次窗外
        // 调用命中旧账、跳过重刷 ⇒ 核读到**上一个图的调度表**。
        // 实测(重录档 = 1 格 + geobuckets + 同会话复用格):桶回放后的
        // **每一步 decode** 都拿桶(T=64/padded=6/tiles=6)的表当 T=1 的表用
        // ⇒ 整段 decode 崩坏(前 1 发正常 —— 因为窗后首个 decode 的 key
        // 与 boot 末次激活不同,恰好重刷过一次)。X1 对照:第 1 发只出 1 token
        // (无 decode 步)⇒ 两发都正常 ⇒ 归因闭合。
        // 代价 = 每次窗外 FI 发射多一次 ~60…600B 的 COMPUTE 流 H2D
        // (与消费它的核**同流 FIFO**;`upload_impl` 已走 Compute)。
        if self.active == Some(key) && capturing {
            return Ok(());
        }
        let int_ws = self.int_ws.as_ref().ok_or_else(|| OpError::Asset {
            op: PREFILL_FI.into(),
            detail: "workspace 未装配".into(),
        })?;
        let entry = self.plans.get_mut(&key).ok_or_else(|| OpError::Contract {
            op: PREFILL_FI.into(),
            field: "plan",
            expect: "激活的 plan 必在条目表".into(),
            got: format!("{key:?}"),
        })?;
        if capturing {
            entry.graph_owned = true;
        }
        res.upload_invariant(int_ws.ptr, &entry.staged)?;
        self.active = Some(key);
        crate::klog_debug!(
            "owl-fi",
            "[plan] int_ws 激活(t={} padded={} tiles={} cap={})",
            key.t,
            entry.plan15[0].max(0),
            entry.tiles,
            capturing,
        );
        Ok(())
    }}

impl Default for FiRuntime {
    fn default() -> Self {
        Self {
            float_ws: None,
            int_ws: None,
            host_staging: vec![0u8; FI_HOST_STAGING_BYTES],
            plans: HashMap::new(),
            active: None,
            last_capturing: false,
            stamp: 0,plan_ragged: None,
            ragged_cu: None,
        }
    }
}


impl KernelFamily for FiRuntime {
    fn id(&self) -> OpId {
        Self::ID
    }

    /// 本族**已建表**算子(fp8 臂;f16 臂未建表 —— 它由族主名承担,
    /// 见 `KernelFamily::names()` 的 id ∪ ops 律)


    /// 构造函数:workspace 预钉(float 64MB + int 32MB;构造期,
    /// 捕获内首次初始化事故类结构性消灭)
    fn assemble(
        res: &mut dyn crate::device::DeviceRes,
        _exec: &mut crate::device::Exec,
    ) -> Result<Box<dyn KernelFamily>, OpError> {
        let me = FiRuntime {
            float_ws: Some(res.alloc(FI_FLOAT_WS_BYTES, "fi.float_ws")?),
            int_ws: Some(res.alloc(FI_INT_WS_BYTES, "fi.int_ws")?),
            host_staging: vec![0u8; FI_HOST_STAGING_BYTES],
            plans: HashMap::new(),
            active: None,
            last_capturing: false,
            stamp: 0,plan_ragged: None,
            ragged_cu: Some(res.alloc(64, "fi.ragged_cu")?),
        };
        Ok(FamilyGuard::boxed(me))
    }

    /// **FI plan 表同步**(W3):把契约层的 [`owl_iface::contract::FiPlanSpec`]
    /// 映射成本族的 [`FiPlanGeom`] 后走 [`FiRuntime::plan_sync`]。
    /// 几何非页对齐/零值等非法输入 = 结构化拒(不静默算错的表)。
    #[allow(clippy::too_many_arguments)]
    fn plan_sync(
        &mut self,
        t: usize,
        ctx_total: usize,
        hq: usize,
        hkv: usize,
        hd: usize,
        page: usize,
        res: &mut dyn crate::device::DeviceRes,
    ) -> Result<(), OpError> {
        if t == 0 || ctx_total == 0 || page == 0 || hkv == 0 || hd == 0 || hq == 0 {
            return Err(OpError::Contract {
                op: PREFILL_FI.into(),
                field: "plan_sync",
                expect: "t/ctx_total/page/hkv/hd/hq 均 > 0".into(),
                got: format!("t={t} ctx={ctx_total} hq={hq} hkv={hkv} hd={hd} page={page}"),
            });
        }
        let geom = FiPlanGeom { total_rows: t, t, hq, hkv, hd, page };
        self.plan_sync(&geom, ctx_total, res)
    }

    fn run(&mut self, msg: &LaunchMsg, res: &mut dyn crate::device::DeviceRes, _exec: &mut crate::device::Exec) -> Result<Bytes, OpError> {
        // 视觉塔注意力(ragged 非因果;2026-10-15)—— 独立契约,先分流
        if msg.kernel.name == crate::family::flashinfer::ATTN_RAGGED_F16 {
            return self.run_attn_ragged(msg, res);
        }
        let call = parse_prefill(msg)?;
        let r = |b: &crate::client::flashinfer::BlockRef| {
            res.resolve(b.id).map(|p| p + b.byte_offset)
        };
        let q = r(&call.q)?;
        let kc = r(&call.kc_fi)?;
        let vc = r(&call.vc)?;
        let q_cu = r(&call.q_cu)?;
        let indices = r(&call.indices)?;
        let indptr = r(&call.indptr)?;
        let last_len = r(&call.last_len)?;
        let out_ref = crate::client::flashinfer::BlockRef { id: call.out.id, byte_offset: 0 };
        let out_ptr = r(&out_ref)?;
        // 就地解包( locals 贯穿 plan/run;None = init 未跑,Asset 拒)
        // 注:只取 (ptr, bytes) 两个标量 —— 后面要 `self.activate(&mut self)`,
        // 持 `&self.float_ws` 的引会与它撞借用。
        let (float_ws_ptr, float_ws_bytes) = {
            let w = self.float_ws.as_ref().ok_or_else(|| OpError::Asset {
                op: PREFILL_FI.into(),
                detail: "workspace 未装配".into(),
            })?;
            (w.ptr, w.bytes)
        };
        let (int_ws_ptr, int_ws_bytes) = {
            let w = self.int_ws.as_ref().ok_or_else(|| OpError::Asset {
                op: PREFILL_FI.into(),
                detail: "workspace 未装配".into(),
            })?;
            (w.ptr, w.bytes)
        };
        let stream = res.stream()?;

        // plan(host,每形状一次;键 = 形状七元组 + **桶化 ctx**)
        let key = plan_key(
            call.total_rows,
            call.ctx_total,
            call.t,
            call.hq,
            call.hkv,
            call.hd,
            call.page,
            fi_cuda_graph(),
        );
        // ⚠️ **plan 必须在捕获窗之外算**(2026-10-14 实测 + 结构性守卫;自
        // `feat/prefill-graph` 合入)。
        //
        // 事实(本机实测,`cudaStreamCaptureModeGlobal`,sm_86):
        // `cudaMemcpyAsync` 的 **H2D/D2H × pinned/pageable 四组合全部可被捕获**
        // (各成 1 个 memcpy 节点,回放 3 次值正确);不可捕获的只有**同步**
        // `cudaMemcpy` 与 `cudaMalloc/Free`。⇒ plan 内部的
        // `cudaMemcpyAsync(int_ws ← page_locked staging)` **会被烘进图**,而
        // `host_staging` 是**可变**缓冲(换形状会重写)—— 一旦 plan 在窗内算,
        // 图里就留下"从可变 host 缓冲拷贝"的节点,回放期取**当时**的内容
        // ⇒ 错调度表/越界,且**不报错**。
        //
        // 现状为何没炸:图捕获前有一次 warmup eager(`GraphPlan::plan`),plan 已
        // 缓存 ⇒ 捕获期不重算。但那是"时序恰好",不是不变量 —— 故此处立硬门:
        // 捕获期若 plan 未命中,**结构化拒**(→ 上层 `Ok(None)` → eager 原路),
        // 不静默、不把错节点烘进图。
        // (更强做法:每形状一份独立 staging 缓冲 ⇒ 节点源恒定;待有需求再上。)
        // **计划块**(抽出成 `ensure_plan` —— 单一事实源;`plan_sync` 同用)
        self.ensure_plan(
            &FiPlanGeom {
                total_rows: call.total_rows,
                t: call.t,
                hq: call.hq,
                hkv: call.hkv,
                hd: call.hd,
                page: call.page,
            },
            call.ctx_total,
            res,
            (float_ws_ptr, float_ws_bytes),
            (int_ws_ptr, int_ws_bytes),
            &stream,
            &msg.kernel.name,
        )?;
        // ── int_ws 激活(多 plan 共存;M2b 2026-10-15)──
        //
        // 窗内那次激活 = 图里一个 memcpy 节点 ⇒**图自携 plan**(多桶共存/
        // 外部换形都不再能把图读的表换掉)。窗上升沿必须强制重激活:上一次
        // 窗内烘过之后,int_ws 可能已被窗外调用刷成别的 plan,而**本窗的首个
        // FI 调用**是唯一能再把它写进图的位置(窗内后续调用会命中 active).
        let capturing = res.capturing().unwrap_or(false);
        if capturing && !self.last_capturing {
            self.active = None;
        }
        self.last_capturing = capturing;
        self.activate(key, res)?;
        let (plan15, mask_ptr) = {
            let e = self.plans.get(&key).ok_or_else(|| OpError::Contract {
                op: PREFILL_FI.into(),
                field: "plan",
                expect: "条目刚插入/命中".into(),
                got: format!("{key:?}"),
            })?;
            (e.plan15, e.mask.as_ref().map(|m| m.ptr).unwrap_or(0))
        };

        let run = if call.fp8kv {
            owl_fi_prefill_run_fp8kv
        } else {
            owl_fi_prefill_run
        };
        // 家族面入账(A13 补漏):FlashInfer = **直连 FFI 发射**(plan/run 均不经
        // `Exec::launch`),账上本无痕;ptrs = q/kc/vc/out 四块地址。
        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[q, kc, vc, out_ptr]);
        let rc = unsafe {
            run(
                q as *const std::ffi::c_void,
                kc as *const std::ffi::c_void,
                vc as *const std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                q_cu as *mut i32,
                indices as *mut i32,
                indptr as *mut i32,
                last_len as *mut i32,
                plan15.as_ptr(),
                int_ws_ptr as *mut std::ffi::c_void,
                int_ws_bytes,
                float_ws_ptr as *mut std::ffi::c_void,
                float_ws_bytes,
                1, // batch
                call.hq as i32,
                call.hkv as i32,
                call.hd as i32,
                call.page as i32,
                call.total_rows as i32,
                call.sm_scale,
                mask_ptr as *const std::ffi::c_void,
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(OpError::Launch {
                op: msg.kernel.name.clone(),
                stage: crate::contract::Stage::Compute,
                detail: format!("run err {rc}"),
            });
        }
        Ok(Bytes::new(call.out.id, msg.out_elems, Dtype::F16, 2))
    }
}

// ============================================================================
// 视觉塔注意力:ragged **非因果** prefill(FlashInfer FA2 家族;2026-10-15)
// ============================================================================

impl FiRuntime {
    /// 一次调用 = 一层注意力(q/k/v [rows, heads, head_dim] f16 连续)。
    /// plan 每形状一次(窗口外;捕获期命中 ⇒ 与 paged 臂同律),**单图契约**。
    fn run_attn_ragged(
        &mut self,
        msg: &LaunchMsg,
        res: &mut dyn crate::device::DeviceRes,
    ) -> Result<Bytes, OpError> {
        let call = crate::client::flashinfer::parse_attn_ragged(msg)?;
        let r = |b: &crate::client::flashinfer::BlockRef| {
            res.resolve(b.id).map(|p| p + b.byte_offset)
        };
        let q = r(&call.q)?;
        let k = r(&call.k)?;
        let v = r(&call.v)?;
        let _ = r(&call.cu)?; // 保留解析/账面对位(FI 臂用 plan 期写的规范 int32 段界)
        let out = r(&crate::client::flashinfer::BlockRef { id: call.out.id, byte_offset: 0 })?;
        let float_ws = self.float_ws.as_ref().ok_or_else(|| OpError::Asset {
            op: crate::family::flashinfer::ATTN_RAGGED_F16.into(),
            detail: "workspace 未装配".into(),
        })?;
        let int_ws = self.int_ws.as_ref().ok_or_else(|| OpError::Asset {
            op: crate::family::flashinfer::ATTN_RAGGED_F16.into(),
            detail: "workspace 未装配".into(),
        })?;
        let stream = res.stream()?;
        if call.n_seq != 1 {
            return Err(OpError::Contract {
                op: crate::family::flashinfer::ATTN_RAGGED_F16.into(),
                field: "n_seq",
                expect: "1(单图契约;多图走 native vis.attn_full 臂)".into(),
                got: format!("{}", call.n_seq),
            });
        }
        let key = RaggedKey { rows: call.rows, heads: call.heads, head_dim: call.head_dim };
        let cu_dev_ptr = self
            .ragged_cu
            .as_ref()
            .ok_or_else(|| OpError::Asset {
                op: crate::family::flashinfer::ATTN_RAGGED_F16.into(),
                detail: "ragged cu scratch 未装配".into(),
            })?
            .ptr as *const std::ffi::c_void;
        // 与 paged 臂同律:**plan 必须在捕获窗外算**(内含 H2D)
        if self.plan_ragged.as_ref().map(|(k, _)| *k != key).unwrap_or(true) {
            if res.capturing().unwrap_or(false) {
                return Err(OpError::Launch {
                    op: msg.kernel.name.clone(),
                    stage: crate::contract::Stage::Load,
                    detail: "FI ragged plan 未缓存而处于捕获窗(plan 内含 H2D)⇒ 拒捕回 eager"
                        .to_string(),
                });
            }
            let (mut plan15, mut cta_tile_q) = ([0i64; 15], 0i32);
            let cu_dev = self.ragged_cu.as_ref().ok_or_else(|| OpError::Asset {
                op: crate::family::flashinfer::ATTN_RAGGED_F16.into(),
                detail: "ragged cu scratch 未装配".into(),
            })?;
            let rc = unsafe {
                crate::family::flashinfer::owl_fi_attn_ragged_plan(
                    float_ws.ptr as *mut std::ffi::c_void,
                    float_ws.bytes,
                    int_ws.ptr as *mut std::ffi::c_void,
                    int_ws.bytes,
                    self.host_staging.as_mut_ptr() as *mut std::ffi::c_void,
                    self.host_staging.len(),
                    plan15.as_mut_ptr(),
                    &mut cta_tile_q,
                    cu_dev.ptr as *mut i32,
                    call.rows as i32,
                    call.heads as i32,
                    call.head_dim as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(OpError::Launch {
                    op: msg.kernel.name.clone(),
                    stage: crate::contract::Stage::Load,
                    detail: format!("ragged plan err {rc}(rows={} heads={} hd={})", call.rows, call.heads, call.head_dim),
                });
            }
            self.plan_ragged = Some((key, plan15));
        }
        let plan15 = self.plan_ragged.as_ref().map(|(_, p)| *p).expect("plan 缓存");
        res.note_device("fam_ffi_launch", &msg.kernel.name, 0, &[q, k, v, out]);
        let rc = unsafe {
            crate::family::flashinfer::owl_fi_attn_ragged_run_f16(
                q as *const std::ffi::c_void,
                k as *const std::ffi::c_void,
                v as *const std::ffi::c_void,
                cu_dev_ptr,
                out as *mut std::ffi::c_void,
                plan15.as_ptr(),
                int_ws.ptr as *mut std::ffi::c_void,
                int_ws.bytes,
                float_ws.ptr as *mut std::ffi::c_void,
                float_ws.bytes,
                call.rows as i32,
                call.heads as i32,
                call.head_dim as i32,
                call.n_seq as i32,
                call.scale,
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(OpError::Launch {
                op: msg.kernel.name.clone(),
                stage: crate::contract::Stage::Compute,
                detail: format!("ragged run err {rc}"),
            });
        }
        Ok(call.out)
    }
}

#[cfg(test)]
mod plan_key_tests {
    use super::plan_key;

    /// **P-1 改写为 ctx 恒分键(2026-10-18)**:桶化前 ctx 每变一次 = 一条 plan
    /// (实测 32k 单请求 186 次重算);桶化后只跨桶才新键。图档位单列。
    #[test]
    fn ctx_always_enters_key_before_bucketing() {
        let k = |rows: usize, ctx: usize, graph: bool| plan_key(rows, ctx, rows, 24, 4, 256, 32, graph);
        assert_ne!(k(512, 512, true), k(512, 2048, true), "ctx 必须分键(桶化由引擎侧做)");
        // 图档位:同形状不同档 ⇒ 不同键(数组与 padded_batch_size 都不同)
        assert_ne!(k(512, 0, false), k(512, 0, true));
        // 形状位:行数/页 仍分键
        assert_ne!(k(512, 0, true), k(511, 0, true));
        assert_ne!(
            plan_key(512, 0, 512, 24, 4, 256, 32, true),
            plan_key(512, 0, 512, 24, 4, 256, 64, true),
            "page 必须分键"
        );    }
}

// ============================================================================
// §3.1 **计划块**(单一事实源;`run` 与 `plan_sync` 共用)
// ============================================================================

/// FI 调用的**形状**(进 plan 键的那几维;与 `run` 的调用点同源)
#[derive(Clone, Copy, Debug)]
pub struct FiPlanGeom {
    pub total_rows: usize,
    pub t: usize,
    pub hq: usize,
    pub hkv: usize,
    pub hd: usize,
    pub page: usize,
}

impl FiRuntime {
    /// **确保 (geom, ctx) 的 plan 在场**(未命中则调 planner + 建掩码 + 冻结表快照)。
    ///
    /// 抽出来的理由(2026-10-18 W3):**表刷新**要能独立于"发射"发生 ——
    /// FI 核进捕获图后,`run` 在回放期不再被调用,而表必须每步换成"当步精确
    /// ctx"那张(否则运行期 chunk 数 ≠ 表里工作项数 ⇒ partial 行 stride 错,
    /// 实测:输出 64 个空 token)。
    fn ensure_plan(
        &mut self,
        geom: &FiPlanGeom,
        ctx_total: usize,
        res: &mut dyn crate::device::DeviceRes,
        float_ws: (u64, usize),
        int_ws: (u64, usize),
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        op_name: &str,
    ) -> Result<(), OpError> {
        let key = plan_key(
            geom.total_rows,
            ctx_total,
            geom.t,
            geom.hq,
            geom.hkv,
            geom.hd,
            geom.page,
            fi_cuda_graph(),
        );
        if self.plans.contains_key(&key) {
            return Ok(());
        }
        if res.capturing().unwrap_or(false) {
            return Err(OpError::Launch {
                op: op_name.to_string(),
                stage: crate::contract::Stage::Load,
                detail: "FI plan 未缓存而处于捕获窗:plan 内含 H2D(可被捕获),\
                         必须在窗口外算 —— 拒捕回 eager".to_string(),
            });
        }
        let qo_indptr = [0i32, geom.t as i32];
        // ⚠️ **kv_indptr 是 page 级**(2026-10-15 V1 定谳)—— plan 这套数组
        // (`kv_len_arr` → `effective_kv_len_arr` → `num_chunks_kv` → 二分搜索的
        // `min_kv_chunk_size = 128/page`)整体按 **页** 计量;而核侧的
        // `paged_kv.indptr` 也是页级 ⇒ 两侧同尺。此前传 token 级([0, ctx_total])
        // ⇒ split 档 plan 算出 34 chunk、核按 2912 token/chunk 算 2 ⇒ partial
        // 行的 `o_stride_n` 缩放与 `merge_indptr` 口径打架 ⇒ 除第 0 行全零
        // (实测 T=8/ctx=3072:零值 43008/49152)。split 关时该数组不参与任何
        // 决策(plan 与 ctx 无关),所以这个单位错一直隐身。
        let nb_pages = ctx_total.div_ceil(geom.page.max(1)) as i32;
        // ⚙ 诊断(level=debug 或 tag `owl-fi`):plan 重算时的**本轮** ctx_total
        // (宿主标量槽是否真按步送到 —— piecewise 断点面取证)。
        // 2026-10-21:原 `[probes] fi_debug` 纯打印开关退役 → 可见性交给日志门。
        crate::klog_debug!(
            "owl-fi",
            "[fi-plan] ctx_total={} nb_pages={nb_pages} T={} page={}",
            ctx_total,
            geom.total_rows,
            geom.page
        );
        let kv_indptr = [0i32, nb_pages];
        let (mut plan15, mut cta_tile_q, mut split_kv) = ([0i64; 15], 0i32, 0i32);
        let mut int_bytes: usize = 0;
        let rc = unsafe {
            owl_fi_prefill_plan(
                float_ws.0 as *mut std::ffi::c_void,
                float_ws.1,
                int_ws.0 as *mut std::ffi::c_void,
                int_ws.1,
                self.host_staging.as_mut_ptr() as *mut std::ffi::c_void,
                self.host_staging.len(),
                plan15.as_mut_ptr(),
                &mut cta_tile_q,
                &mut split_kv,
                &mut int_bytes,
                qo_indptr.as_ptr(),
                kv_indptr.as_ptr(),
                geom.total_rows as i32,
                1, // batch = 1(owl 单会话)
                geom.hq as i32,
                geom.hkv as i32,
                geom.hd as i32,
                geom.page as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            )
        };
        if rc != 0 {
            return Err(OpError::Launch {
                op: op_name.to_string(),
                stage: crate::contract::Stage::Load,
                detail: format!(
                    "plan err {rc}(hd={} page={} T={} ctx={})",
                    geom.hd, geom.page, geom.t, ctx_total
                ),
            });
        }
        // ── **P-5:尾部 CTA 掩码**(2026-10-15;设计 §4.1c F-6/F-7/F-8)
        //
        // 图档 plan 的 `padded_batch_size = max(2·SM/hkv, tiles)` ⇒ 尾部
        // CTA 数可以远超真 tile 数(t_pad=128/3090:41 vs 12)。不掩码时
        // 它们读 plan 表的**残留表项**(行为取决于上一次 plan 写了什么 =
        // 静默);掩码后核侧 `if (!mask[bx]) return;` 立即退。
        //
        // 内容 `i < tiles` 由**本侧**算(唯一事实源 = plan15):
        //   tiles = ceil_div(packed_qo_len, cta_tile_q),packed = t·gqa
        // (`batch=1`;多请求时须逐请求求和 —— 本仓 FI 调用恒 batch=1,
        //  非 1 在此结构化拒,不给"算错了但看着对"的余地)
        let gqa = geom.hq / geom.hkv.max(1);
        let (padded, cta_tile_q) = (plan15[0].max(0) as usize, plan15[3].max(0) as usize);
        if gqa == 0 || cta_tile_q == 0 {
            return Err(OpError::Contract {
                op: PREFILL_FI.into(),
                field: "plan15",
                expect: "cta_tile_q > 0 且 gqa ≥ 1".into(),
                got: format!("gqa={gqa} cta_tile_q={cta_tile_q}"),
            });
        }
        let tiles = (geom.total_rows * gqa).div_ceil(cta_tile_q).max(1);
        // **掩码按需分配**(2026-10-18):`padded > tiles` 才需要(split 档 /
        // 图档真会派超出真 tile 数的 CTA);`padded == tiles` 时 mask 内容
        // 恒全 1 = 空操作 ⇒ 传 null(核侧契约:null = 不用掩码)。
        // 旧写法无条件每 plan 分配一块(进 append-only scratch 账本、
        // 永不释放)⇒ 桶化后每个跨桶都多一笔设备分配 + 一次线性扫描。
        let mask = if padded > tiles {
            let mask = res.alloc(padded.max(1), "fi.mask")?;
            let mut m = vec![0u8; padded.max(1)];
            for (i, v) in m.iter_mut().enumerate() {
                *v = u8::from(i < tiles);
            }
            res.upload(mask.ptr, &m)?;
            Some(mask)
        } else {
            None
        };
        if padded > tiles {
            crate::klog_debug!(
                "owl-fi",
                "[mask] 图档 plan:T={} ctx={} padded={} tiles={} ⇒ 掩 {} 个尾部 CTA",
                geom.t,
                ctx_total,
                padded,
                tiles,
                padded - tiles
            );
        }
        // 不变盒快照(plan 表有效前缀;源 = 共享 staging —— 只在此刻拷一次,
        // 此后 int_ws 的每次重刷都从盒子来,见 activate)
        if int_bytes == 0 || int_bytes > self.host_staging.len() {
            return Err(OpError::Contract {
                op: PREFILL_FI.into(),
                field: "int_bytes",
                expect: format!("1..={}(plan 表有效长度)", self.host_staging.len()),
                got: format!("{int_bytes}"),
            });
        }
        let staged = InvariantBox::freeze(self.host_staging[..int_bytes].to_vec());
        let born = self.stamp;
        self.stamp += 1;
        self.plans.insert(key, PlanEntry { plan15, staged, mask, tiles, born, graph_owned: false });
        // ⚠️ **不再信任 C++ 侧那次写**(2026-10-17):`owl_fi_prefill_plan` 的
        // `int_ws ← page_locked staging` memcpy 在**C++ 适配器内部**发起
        // (自有/NULL 流),与本 server 的 COMPUTE 流**无跨流序**;而 FI 核
        // 在 COMPUTE 流 —— H2D 流繁忙时核可能读到**未落地的调度表**
        // (实测:同格复用档 decode 中段分叉;纯栅栏即愈的同一族)。
        // ⇒ 认知置空:紧接着的 `activate` 会从**冻结盒**在本侧(COMPUTE 流)
        // 重刷一遍 —— 与消费它的核**同流 FIFO**,结构性有序。
        self.active = None;
        self.evict_stale();
        Ok(())
    }

    /// **plan 表同步**(W3):保证 (geom, ctx) 的 plan 在场,并把它的表刷进
    /// `int_ws`(窗外路径 ⇒ `activate` 必刷)。**调用方 = 引擎**(每步图 launch
    /// 之前),用于"FI 核在图内 + 图不烘 restore"的档。
    pub fn plan_sync(
        &mut self,
        geom: &FiPlanGeom,
        ctx_total: usize,
        res: &mut dyn crate::device::DeviceRes,
    ) -> Result<(), OpError> {
        let float_ws = self.float_ws.as_ref().ok_or_else(|| OpError::Asset {
            op: PREFILL_FI.into(),
            detail: "workspace 未装配(float)".into(),
        })?;
        let int_ws = self.int_ws.as_ref().ok_or_else(|| OpError::Asset {
            op: PREFILL_FI.into(),
            detail: "workspace 未装配(int)".into(),
        })?;
        let (fp, fb) = (float_ws.ptr, float_ws.bytes);
        let (ip, ib) = (int_ws.ptr, int_ws.bytes);
        let stream = res.stream()?;
        self.ensure_plan(geom, ctx_total, res, (fp, fb), (ip, ib), &stream, "plan-sync")?;
        let key = plan_key(
            geom.total_rows,
            ctx_total,
            geom.t,
            geom.hq,
            geom.hkv,
            geom.hd,
            geom.page,
            fi_cuda_graph(),
        );
        self.activate(key, res)
    }
}
