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
    /// **split 关时恒 0**(归一化):源码级 F-5 —— split 关时 plan 的数组
    /// 只由 qo 长度定,ctx 三条链(`kv_len_arr`/`effective_kv_len_arr`/
    /// `kv_chunk_size`)只在 split 分支被消费 ⇒ 键去 ctx 后,**一个桶一份 plan
    /// 跨 ctx 复用**(M2b 的 P-1)。split 开时必须留着(那时 ctx 真参与)。
    ctx_total: usize,
    /// **图档 plan**(`FI_CUDA_GRAPH`;见 common.h `owl_fi_cuda_graph`)
    /// —— 数组与 `padded_batch_size` 都与 eager 档不同,必须分键,否则
    /// 一次 eager 调用会把图档 plan 挤掉(mask 内容随之失配)
    graph: bool,
    /// split-kv 开关(`FI_SPLIT_KV`;与 C++ `owl_fi_split_kv_enabled` 同源)——
    /// 决定 ctx 是否进键(见上),必须显式入键,否则切档后复用错 plan
    split: bool,
    t: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    page: usize,
}

/// 图档 plan 开关(与 C++ `owl_fi_cuda_graph` 同源:env `FI_CUDA_GRAPH`)。
/// plan 期读一次(非热路径);两处必须一致,否则 PlanKey 与真数组错档。
fn fi_cuda_graph() -> bool {
    std::env::var_os("FI_CUDA_GRAPH").is_some_and(|v| !v.is_empty() && v != "0")
}

/// split-kv 开关(与 C++ `owl_fi_split_kv_enabled` 同源:env `FI_SPLIT_KV`)
fn fi_split_kv() -> bool {
    std::env::var_os("FI_SPLIT_KV").is_some_and(|v| !v.is_empty() && v != "0")
}

/// **plan 键归一化(纯函数;可单测)**:split 关 ⇒ 抹掉 `ctx_total`
/// (源码级 F-5:那三条 ctx 链只在 split 分支被消费)⇒ 一个桶一份 plan 跨 ctx
/// 复用(M2b 的 P-1);split 开 ⇒ ctx 真参与定瓦,必须进键。
fn plan_key(
    total_rows: usize,
    ctx_total: usize,
    t: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    page: usize,
    graph: bool,
    split: bool,
) -> PlanKey {
    PlanKey {
        total_rows,
        ctx_total: if split { ctx_total } else { 0 },
        t,
        hq,
        hkv,
        hd,
        page,
        graph,
        split,
    }
}

/// **取证档**(探针④红绿对照):`OWL_FI_NO_PLAN_RESTORE=1` ⇒ 关掉 int_ws
/// 自携(plan 切换后不重刷)⇒ 多 plan 共存时应出现"回放读到别人的调度表"。
/// 生产不得设此档(它存在的唯一目的是让探针能把机制证成承重件)。
fn no_plan_restore() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var_os("OWL_FI_NO_PLAN_RESTORE").is_some_and(|v| !v.is_empty() && v != "0")
    })
}

/// 诊断打印开关(与 C++ `owl_fi_debug` 同源:env `FI_DEBUG`)
fn fi_debug() -> bool {
    std::env::var_os("FI_DEBUG").is_some_and(|v| !v.is_empty() && v != "0")
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
    /// 尾部 CTA 掩码块(P-5)+ 内容 `i < tiles`。为什么与 plan 同车:掩码是
    /// `tiles/padded` 的函数;共用一条缓冲会被换形调用改写 ⇒ 回放旧图掩码失配
    /// ⇒ **真行漏算**(静默错值)。
    mask: ScratchBuf,
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
    /// **int_ws 当前内容**对应的 key(None = 未知/已失效)
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
        // 取证档(探针④红绿对照;生产不得设):关掉自携 ⇒ 应退回"读到别人的
        // 调度表"。
        if no_plan_restore() {
            self.active = Some(key);
            return Ok(());
        }
        if self.active == Some(key) {
            return Ok(());
        }
        let int_ws = self.int_ws.as_ref().ok_or_else(|| OpError::Asset {
            op: PREFILL_FI.into(),
            detail: "workspace 未装配".into(),
        })?;
        let capturing = res.capturing().unwrap_or(false);
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
        if fi_debug() {
            eprintln!(
                "[owl-fi][plan] int_ws 激活(t={} padded={} tiles={} cap={})",
                key.t,
                entry.plan15[0].max(0),
                entry.tiles,
                res.capturing().unwrap_or(false),
            );
        }
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

        // plan(host,每形状一次;键 = 形状七元组)
        let key = plan_key(
            call.total_rows,
            call.ctx_total,
            call.t,
            call.hq,
            call.hkv,
            call.hd,
            call.page,
            fi_cuda_graph(),
            fi_split_kv(),
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
        if !self.plans.contains_key(&key) {
            if res.capturing().unwrap_or(false) {
                return Err(OpError::Launch {
                    op: msg.kernel.name.clone(),
                    stage: crate::contract::Stage::Load,
                    detail: "FI plan 未缓存而处于捕获窗:plan 内含 H2D(可被捕获),\
                             必须在窗口外算 —— 拒捕回 eager".to_string(),
                });
            }
            let qo_indptr = [0i32, call.t as i32];
            // ⚠️ **kv_indptr 是 page 级**(2026-10-15 V1 定谳)—— plan 这套数组
            // (`kv_len_arr` → `effective_kv_len_arr` → `num_chunks_kv` → 二分搜索的
            // `min_kv_chunk_size = 128/page`)整体按 **页** 计量;而核侧的
            // `paged_kv.indptr` 也是页级 ⇒ 两侧同尺。此前传 token 级([0, ctx_total])
            // ⇒ split 档 plan 算出 34 chunk、核按 2912 token/chunk 算 2 ⇒ partial
            // 行的 `o_stride_n` 缩放与 `merge_indptr` 口径打架 ⇒ 除第 0 行全零
            // (实测 T=8/ctx=3072:零值 43008/49152)。split 关时该数组不参与任何
            // 决策(plan 与 ctx 无关),所以这个单位错一直隐身。
            let nb_pages = call.ctx_total.div_ceil(call.page.max(1)) as i32;
            // ⚙ 诊断(OWL_FI_DEBUG=1):plan 重算时的**本轮** ctx_total(宿主标量槽
            // 是否真按步送到 —— piecewise 断点面取证;缺省零输出)
            if std::env::var_os("OWL_FI_DEBUG").is_some_and(|v| v != "0") {
                eprintln!(
                    "[fi-plan] ctx_total={} nb_pages={nb_pages} T={} page={}",
                    call.ctx_total, call.total_rows, call.page
                );
            }
            let kv_indptr = [0i32, nb_pages];
            let (mut plan15, mut cta_tile_q, mut split_kv) = ([0i64; 15], 0i32, 0i32);
            let mut int_bytes: usize = 0;
            let rc = unsafe {
                owl_fi_prefill_plan(
                    float_ws_ptr as *mut std::ffi::c_void,
                    float_ws_bytes,
                    int_ws_ptr as *mut std::ffi::c_void,
                    int_ws_bytes,
                    self.host_staging.as_mut_ptr() as *mut std::ffi::c_void,
                    self.host_staging.len(),
                    plan15.as_mut_ptr(),
                    &mut cta_tile_q,
                    &mut split_kv,
                    &mut int_bytes,
                    qo_indptr.as_ptr(),
                    kv_indptr.as_ptr(),
                    call.total_rows as i32,
                    1, // batch = 1(owl 单会话)
                    call.hq as i32,
                    call.hkv as i32,
                    call.hd as i32,
                    call.page as i32,
                    stream.cu_stream() as *mut std::ffi::c_void,
                )
            };
            if rc != 0 {
                return Err(OpError::Launch {
                    op: msg.kernel.name.clone(),
                    stage: crate::contract::Stage::Load,
                    detail: format!(
                        "plan err {rc}(hd={} page={} T={} ctx={})",
                        call.hd, call.page, call.t, call.ctx_total
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
            let gqa = call.hq / call.hkv.max(1);
            let (padded, cta_tile_q) = (plan15[0].max(0) as usize, plan15[3].max(0) as usize);
            if gqa == 0 || cta_tile_q == 0 {
                return Err(OpError::Contract {
                    op: PREFILL_FI.into(),
                    field: "plan15",
                    expect: "cta_tile_q > 0 且 gqa ≥ 1".into(),
                    got: format!("gqa={gqa} cta_tile_q={cta_tile_q}"),
                });
            }
            let tiles = (call.total_rows * gqa).div_ceil(cta_tile_q).max(1);
            let mask = res.alloc(padded.max(1), "fi.mask")?;
            let mut m = vec![0u8; padded.max(1)];
            for (i, v) in m.iter_mut().enumerate() {
                *v = u8::from(i < tiles);
            }
            res.upload(mask.ptr, &m)?;
            if padded > tiles && fi_debug() {
                eprintln!(
                    "[owl-fi][mask] 图档 plan:T={} ctx={} padded={} tiles={} ⇒ 掩 {} 个尾部 CTA",
                    call.t,
                    call.ctx_total,
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
            // C++ plan 自己已把表写进 int_ws ⇒ 内容认知同步(activate 不重刷)
            self.active = Some(key);
            self.evict_stale();
        }
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
            (e.plan15, e.mask.ptr)
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

    /// **P-1 判据**:split 关 ⇒ ctx 不进键(换 ctx 复用同一份 plan);
    /// split 开 ⇒ ctx 进键(不得跨 ctx 复用)。图档位单列(数组/padded 都不同)。
    #[test]
    fn ctx_enters_key_only_when_split_on() {
        let k = |rows: usize, ctx: usize, graph: bool, split: bool| {
            plan_key(rows, ctx, rows, 24, 4, 256, 32, graph, split)
        };
        let a = k(512, 512, true, false);
        let b = k(512, 2048, true, false);
        assert_eq!(a, b, "split 关:ctx 必须被归一化掉(F-5)");
        assert_eq!(a.ctx_total, 0);

        let c = k(512, 512, true, true);
        let d = k(512, 2048, true, true);
        assert_ne!(c, d, "split 开:ctx 必须分键");

        // 图档位:同形状不同档 ⇒ 不同键(数组与 padded_batch_size 都不同)
        assert_ne!(k(512, 0, false, false), k(512, 0, true, false));
        // 形状位:行数/页 仍分键
        assert_ne!(k(512, 0, true, false), k(511, 0, true, false));
        assert_ne!(
            plan_key(512, 0, 512, 24, 4, 256, 32, true, false),
            plan_key(512, 0, 512, 24, 4, 256, 64, true, false),
            "page 必须分键"
        );    }
}
