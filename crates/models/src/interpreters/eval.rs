//! 计算(推理)解释器:声明树 DAG → 后端原语调用。
//!
//! 职责(单域):`TensorOps` 树的自叶向根归约 —— 逐节点翻译为
//! alloc / lower / launch,带 CSE 备忘录(同 id 单次求值)+ 毒值落地
//! (结构化报错 + depth 归因)+ C1 维度断言。
//!
//! 变体定位:这是**计算域**解释器;装载域见 [`super::load`]
//! (Loadable::layout → Want 清单 → 分桶流水)。二者共享 face 注入
//! 形态(`owl-iface::contract::DeviceClient`)与"声明/执行分离"范式,
//! 但互不依赖 —— 新变体(图捕获回放、量化装载、其他平台)另起文件。

use crate::contract::{Bytes, Dtype, ModelError};
use owl_iface::contract::Arg;
use crate::ops::Op;
use crate::tensor::TensorOps;
use std::future::Future;
use std::pin::Pin;

use super::observe::{BlockRef, BlockStats, NodeEvent, Tap, Want};
use super::planner::BlockPool;

// ============================================================================
// §0 性能钩子(perf-framework.md §四)
// ============================================================================
//
// 解释器是全部节点的唯一执行口 ⇒ 「每个算子的宿主耗时 + 调用树」一处声明、
// 全图生效。钩子≈只读:不改变计算、不新增 launch/alloc。
//
// 两道门:①编译期(debug 或 feature `perf`;不带 = 空语句零成本)
//         ②运行期 level(eval 入口读一次存入 [`PerfCtx`],热路径零全局查询)
//
// tag 律:`interp.host.<面>.<phase>.<class>`,class = 核名(登记表有界集)
// 或内建类;phase 由 [`crate::env::EvalPhase`] 定相。

/// 热路径性能面(值语义;入口一次性组装)
#[derive(Clone, Copy)]
struct PerfCtx {
    /// 运行期分级(见 `metrics::level`)
    level: u8,
    /// 阶段 tag 段(`unk`/`prefill`/`decode`/`spec`/`load`)
    phase: &'static str,
}

impl Default for PerfCtx {
    fn default() -> Self {
        Self { level: owl_shared::metrics::level::OFF, phase: "unk" }
    }
}

impl PerfCtx {
    fn of(env: &crate::env::EnvProvider) -> Self {
        // 非 perf 构建:`PERF_BUILD = false` 常量折叠 → 本函数空转
        if !owl_shared::metrics::PERF_BUILD {
            return Self::default();
        }
        Self { level: owl_shared::metrics::level(), phase: env.phase.tag() }
    }

    /// HOST 级以上(计时准入;非 perf 构建 = 常量假)
    #[inline]
    fn host(self) -> bool {
        owl_shared::metrics::PERF_BUILD && self.level >= owl_shared::metrics::level::HOST
    }
}

/// 样本落账(直用 API:不经宏,故 release 亦有效)
#[inline]
fn perf_rec(tag: &str, dur: std::time::Duration) {
    owl_shared::metrics::with_metrics_store(|s| s.timer_record_tag(tag, dur, file!(), line!()));
}

/// 计数落账
#[inline]
fn perf_cnt(tag: &str, n: u64) {
    owl_shared::metrics::with_metrics_store(|s| s.counter_add(tag, n, file!(), line!()));
}

/// 块池预算(字节;**0 / 缺省 = 关**)。P1″ 2026-10-14。
///
/// `OWL_NO_RECLAIM=1` ⇒ 关掉活性中途回收(诊断/对照臂)。
///
/// **OnceLock 记忆化**(2026-10-17 审查):原为每个 `eval_ops` 现场
/// `std::env::var`(取进程 env 锁 + 分配 String;求值入口 = 逐算子),而本文件
/// 的兄弟旋钮(`block_pool_budget`)早有 OnceLock 先例 ⇒ 口径归位。语义:该旋钮
/// **boot 期定谳**(进程内改 env 不再生效;诊断档本就如此用)。
fn no_reclaim() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // env **单一访问口**(2026-10-17 审查 §六轮:原先直读 `std::env::var_os`,
    // 绕过 `env_reader` —— 与 J1/J4 的收口纪律相悖,且测试面的 EnvGuard 不再生效)
    *V.get_or_init(|| owl_shared::env_reader::flag("OWL_NO_RECLAIM"))
}

/// 门 = env `OWL_BLOCK_POOL`(与本文件的 `OWL_NO_RECLAIM` 同一「诊断/调优门」
/// 先例),**`OnceLock` 缓存一次** ⇒ 热路径零 env 查询。
/// 引擎侧后续应升格为 `OwlConfig` 键(本 lane 文件面不含 engine/,见报告)。
fn block_pool_budget() -> usize {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| owl_shared::env_reader::parse::<usize>("OWL_BLOCK_POOL").unwrap_or(0))
}

/// 分段秒表(`Call` 臂宿主链分段:resolve/build/alloc/lower/launch)
struct LapTimer {
    at: std::time::Instant,
}

impl LapTimer {
    #[inline]
    fn start() -> Self {
        Self { at: std::time::Instant::now() }
    }
    /// 落上一段账并重置
    #[inline]
    fn lap(&mut self, tag: &str) {
        let now = std::time::Instant::now();
        perf_rec(tag, now.saturating_duration_since(self.at));
        self.at = now;
    }
}

/// 算子类名(有界词表;`Call` 臂在 resolve 后用核名覆写)
fn builtin_class(op: &Op) -> &'static str {
    match op {
        Op::Htod { .. } => "htod",
        Op::Zeros => "zeros",
        Op::Block { .. } => "block",
        Op::Reshape | Op::SliceView { .. } => "view",
        Op::SlotWrite => "slot_write",
        Op::Matmul | Op::MatmulNt => "gemm",
        Op::Call { .. } => "call",
        Op::Uninit => "uninit",
        Op::CallDst { .. } => "call_dst",
        #[allow(unreachable_patterns)]
        _ => "other",
    }
}

// ============================================================================
// §1 异步解释器:eval(层入口)/ eval_ops(树归约;引擎主路径)
// ============================================================================

/// 计算执行(层级入口):驱动层的 `Module::forward` 声明并归约。
/// 解释器自己调用层钩子并为它传递 ctx —— 使用者只给层与输入。
///
/// ```rust,ignore
/// let out = eval(&mlp, &xs, face, &ForwardCtx::minimal(1)).await?;
/// ```
pub async fn eval<M, D>(
    layer: &M,
    xs: &TensorOps,
    face: &mut D,
    ctx: &crate::module::ForwardCtx<'_>,
) -> Result<Bytes, ModelError>
where
    M: crate::module::Module,
    D: crate::contract::DeviceClient,
{
    let ops = layer.forward(xs, ctx);   // 层产出声明(ctx 引用透传)
    eval_ops(ops.step(), face).await    // 解释器归约
}

/// 声明树求值(内部件;供手工子树/非层根的归约):自叶向根,
/// 逐节点翻译为原语调用(face = 后端句柄)。
/// 块字节 → host f32(按块 dtype;f16 基线后观测面统一走此解码。
/// 字节口径纪律:读回长度一律 `elem 数 × dtype.size_bytes()`,禁硬编码)
pub(crate) fn decode_host(dtype: Dtype, buf: &[u8]) -> Vec<f32> {
    match dtype {
        Dtype::F16 => buf
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => buf
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F32 => buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        // U32(frontier 等 id 块):位型直释(观测统计只对浮点块有意义)
        Dtype::U32 => buf
            .chunks_exact(4)
            .map(|c| f32::from_bits(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
        // U8(不透明字节)/ F8 两种(字节口径)/ **I8(按有符号解!)** —— 观测面
        // 只用于统计量级(池内容不是激活);`I8` 若按无符号读会系统性偏 +128
        Dtype::U8 | Dtype::F8E4M3 | Dtype::F8E5M2 => buf.iter().map(|b| *b as f32).collect(),
        Dtype::I8 => buf.iter().map(|b| (*b as i8) as f32).collect(),
    }
}

pub fn eval_ops<'a, D>(
    t: &'a TensorOps,
    face: &'a mut D,
) -> Pin<Box<dyn Future<Output = Result<Bytes, ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    eval_ops_env(t, face, crate::env::EnvProvider::default(), None)
}

/// 带环境的求值(EnvProvider;engine 供给,Call 臂按 hw/page/trace 执行)。
/// `tp` = 张量并行上下文(**S4 ⑤ 注入口**;`None` = 单卡,零分支代价)。
pub fn eval_ops_env<'a, D>(
    t: &'a TensorOps,
    face: &'a mut D,
    env: crate::env::EnvProvider,
    tp: Option<&'a crate::module::TpCtx>,
) -> Pin<Box<dyn Future<Output = Result<Bytes, ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    Box::pin(async move {
        let perf = PerfCtx::of(&env);
        let mut ctx =
            EvalCtx { face, memo: std::collections::HashMap::new(), tap: None, arena: Vec::new(),
                arena_set: std::collections::HashSet::new(),
                block_users: std::collections::HashMap::new(),
                pending: std::collections::HashMap::new(), reclaim: false, perf, env, pool: None,
            tp: tp.cloned(),
        };
        let root = _eval_rec(t, &mut ctx).await?;
        // 非 scoped 口径:竞技场原样丢弃(块常驻,历史行为不变)
        Ok(root)
    })
}

/// 带竞技场的求值(E2a,2026-09-27):返回 (根块, 中间块候选表)。
/// 竞技场 = 本次求值新建的全部块 id(Htod/Alloc/launch 输出;CSE 去重);
/// 候选表 = 竞技场 - 根。调用方在收割根数据后 `face.free(&candidates)`
/// 归还设备池 —— 不再引用即焚,消除 eval 中间块零回收的线性累积
/// (4k 双 turn ~24GB OOM 立案见 roadmap E2)。安全前提:free 前根
/// 已 dtoh(COMPUTE 排空);图捕获期禁用(捕获块 = slab 切片)。
///
/// **活性中途回收(2026-10-02;scoped 独占)**:整 chunk SSA 图宽问题
/// —— 27B prefill chunk 的全 64 层中间块同时存活(峰值 ~4GB),chunk
/// 被内存封顶在 128,发射摊薄锁死(prefill 差 vLLM 1.76× 主根因)。本
/// 口径在预过波里给每节点记剩余消费者数(+1 = 调用方持根),每个节点
/// 执行完后递减各父节点;计零 = 节点死亡 → 块归属减一,归零且竞技场
/// 原生 → 立即 `face.free`(流序 free:消费 kernel 已入队,同流保序。
/// 透传臂 Reshape/SlotWrite 与父共享块 → 块归属按块计数,不按节点。
/// scoped 结束时候选表 = 仍存活者(通常仅根外的零星),逐层收割后
/// 图宽 = 峰值活跃集,chunk 可提至 ≥1024。)
#[allow(clippy::type_complexity)] // 手写 boxed async fn 返回型
pub fn eval_ops_scoped<'a, D>(
    t: &'a TensorOps,
    face: &'a mut D,
) -> Pin<Box<dyn Future<Output = Result<(Bytes, Vec<u64>), ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    eval_ops_scoped_env(t, face, crate::env::EnvProvider::default(), None)
}

/// 带环境的 scoped 求值(EnvProvider;engine 供给;`tp` 注入口同 [`eval_ops_env`])
#[allow(clippy::type_complexity)] // 手写 boxed async fn 返回型
pub fn eval_ops_scoped_env<'a, D>(
    t: &'a TensorOps,
    face: &'a mut D,
    env: crate::env::EnvProvider,
    tp: Option<&'a crate::module::TpCtx>,
) -> Pin<Box<dyn Future<Output = Result<(Bytes, Vec<u64>), ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    eval_ops_scoped_env_pool(t, face, env, tp, block_pool_budget())
}

/// scoped 求值 + **显式块池预算**(0 = 关;P1″)。
/// 测试面直传预算 ⇒ 不依赖进程级 env;生产入口走 [`block_pool_budget`]。
#[allow(clippy::type_complexity)] // 手写 boxed async fn 返回型
pub(crate) fn eval_ops_scoped_env_pool<'a, D>(
    t: &'a TensorOps,
    face: &'a mut D,
    env: crate::env::EnvProvider,
    tp: Option<&'a crate::module::TpCtx>,
    pool_budget: usize,
) -> Pin<Box<dyn Future<Output = Result<(Bytes, Vec<u64>), ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    Box::pin(async move {
        let perf = PerfCtx::of(&env);
        let mut ctx =
            EvalCtx { face, memo: std::collections::HashMap::new(), tap: None, arena: Vec::new(),
                arena_set: std::collections::HashSet::new(),
                block_users: std::collections::HashMap::new(),
                pending: count_pending(t), reclaim: !no_reclaim(), perf, env,
                pool: (pool_budget > 0).then(|| BlockPool::new(pool_budget)),
            tp: tp.cloned(),
        };
        let r = _eval_rec(t, &mut ctx).await;
        // 池**无论成败都要抽干**:错误路径不抽 = 缓存块被遗弃而 server 仍认
        // 它们 owned(真漏显存)。主路错优先(池错只降级不掩盖)。
        let flushed = ctx.flush_pool().await;
        let root = match (r, flushed) {
            (Ok(root), Ok(())) => root,
            (Ok(_), Err(e)) => return Err(e),
            (Err(e), _) => return Err(e),
        };
        let mut candidates = std::mem::take(&mut ctx.arena);
        candidates.retain(|id| *id != root.id && ctx.arena_set.contains(id));
        Ok((root, candidates))
    })
}

/// 多根共享 memo 求值(刀1.6,2026-10-03):**捕获窗单次执行多输出树**
/// —— 修复「每输出一棵独立整模重执行」的图内重复(捕获图含 2.64×
/// 整模内核,回放税翻倍的历史包袱一次性清除)。roots 各自归约,共享
/// CSE 备忘录(同 id 单次求值);语义与逐根 eval_ops 完全一致,
/// 仅消除跨根重复执行。竞技场/回收 = plain 口径(块常驻,图捕获用)。
pub fn eval_ops_multi<'a, D>(
    roots: &'a [&'a TensorOps],
    face: &'a mut D,
    tp: Option<&'a crate::module::TpCtx>,
) -> Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    eval_ops_multi_env(roots, face, crate::env::EnvProvider::default(), tp)
}

/// 多根 + 环境(E5-DF2;prefill taps 同树多根,层声明 env 与执行 env 双通;
/// `tp` 注入口同 [`eval_ops_env`])
pub fn eval_ops_multi_env<'a, D>(
    roots: &'a [&'a TensorOps],
    face: &'a mut D,
    env: crate::env::EnvProvider,
    tp: Option<&'a crate::module::TpCtx>,
) -> Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    Box::pin(async move {
        let perf = PerfCtx::of(&env);
        let mut ctx =
            EvalCtx { face, memo: std::collections::HashMap::new(), tap: None, arena: Vec::new(),
                arena_set: std::collections::HashSet::new(),
                block_users: std::collections::HashMap::new(),
                pending: std::collections::HashMap::new(), reclaim: false, perf, env, pool: None,
            tp: tp.cloned(),
        };
        let mut out = Vec::with_capacity(roots.len());
        for r in roots {
            out.push(_eval_rec(r, &mut ctx).await?);
        }
        Ok(out)
    })
}

/// 多根 + 竞技场回收(B6.3 后泄漏案,2026-10-10):roots 各自归约共享
/// CSE;pending = 各根 count_pending 合并(共享子树用户数累加);求值
/// 完成后 candidates = arena − 根集就地 face.free(流序 free:消费
/// kernel 已入队,同流保序 —— 与 scoped 单根同一法律)。
/// 用途:运行期副作用树(K0 写哑根/encode 等)——roots 仅触发执行,
/// 中间块不再被引用;捕获窗仍必须用 plain eval_ops_multi_env(块常驻)。
pub fn eval_ops_multi_scoped_env<'a, D>(
    roots: &'a [&'a TensorOps],
    face: &'a mut D,
    env: crate::env::EnvProvider,
    tp: Option<&'a crate::module::TpCtx>,
) -> Pin<Box<dyn Future<Output = Result<Vec<Bytes>, ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    Box::pin(async move {
        // ⚠️ **多根边只数一次**(2026-10-19 §8.31):旧口径逐根相加,在"根套根"
        // (Q 槽 / DFlash2 taps:根节点在主根子树里)时把共享子树的边数 k 次 ⇒
        // `pending` 恒多算 ⇒ 中途活性回收整体失效(真回收 14/新块 1431 实测)。
        let pending = count_pending_multi(roots);
        let perf = PerfCtx::of(&env);
        let pool_budget = block_pool_budget();
        let mut ctx =
            EvalCtx { face, memo: std::collections::HashMap::new(), tap: None, arena: Vec::new(),
                arena_set: std::collections::HashSet::new(),
                block_users: std::collections::HashMap::new(),
                pending, reclaim: true, perf, env,
                pool: (pool_budget > 0).then(|| BlockPool::new(pool_budget)),
            tp: tp.cloned(),
        };
        let mut out = Vec::with_capacity(roots.len());
        for r in roots {
            // **错误路径也必须抽干竞技场**(2026-10-16 实测定谳):早退(求值途中
            // 任一节点失败)会让本次求值的**全部活块整套泄漏**(下面的末尾清扫根本
            // 跑不到)。27B prefill 一次就是 GB 级 —— 桶族失败链就是这样被毒死的:
            // 第一个桶 warmup 失败 → 泄漏 ~GB → 后面每个桶(含只需 5MiB 的)都跟着失败。
            match _eval_rec(r, &mut ctx).await {
                Ok(b) => out.push(b),
                Err(e) => {
                    let _ = ctx.flush_pool().await;
                    let mut cand: Vec<u64> = std::mem::take(&mut ctx.arena);
                    cand.retain(|id| ctx.arena_set.contains(id));
                    if !cand.is_empty() {
                        let _ = ctx.face.free(&cand).await; // 回收失败不掩盖原错
                    }
                    return Err(e);
                }
            }
        }
        ctx.flush_pool().await?;
        // **活性收尾归因**(门:`OWL_LIVE_DIAG`;2026-10-19):
        // "真回收几乎为 0"时唯一要问的 = **memo 里还剩什么、靠什么吊着**:
        // 节点的 `pending`(预波消费者计数未归零)还是块的 `users`(别名/持有者)。
        // 打印前 6 大残块的 (bytes, users, pending, name) + 直方图 ⇒ 一眼定位。
        if std::env::var_os("OWL_LIVE_DIAG").is_some() {
            let mut hist = [0usize; 4]; // pending==0 / ==1 / ==2 / >=3
            let mut users_hist = [0usize; 4];
            let mut live_bytes = 0usize;
            let mut rows: Vec<(usize, u64, usize, usize, String)> = Vec::new();
            for (nid, b) in ctx.memo.iter() {
                let p = ctx.pending.get(nid).copied().unwrap_or(0);
                let u = ctx.block_users.get(&b.id).copied().unwrap_or(0);
                hist[p.min(3)] += 1;
                users_hist[u.min(3)] += 1;
                let bytes = b.len.saturating_mul(b.dtype.size_bytes());
                live_bytes += bytes;
                rows.push((bytes, *nid, u, p, format!("{:?}", b.id)));
            }
            rows.sort_unstable_by_key(|r| std::cmp::Reverse(r.0));
            let top: Vec<String> = rows
                .iter()
                .take(6)
                .map(|(by, nid, u, p, bid)| format!("{:.1}MB node{nid} block{bid} u={u} p={p}", *by as f64 / 1048576.0))
                .collect();
            eprintln!(
                "[live-end] memo 剩 {} 节点 / {} MiB ‖ 节点 pending 直方图 [=0 {}, =1 {}, =2 {}, ≥3 {}] ‖ 块 users 直方图 [=0 {}, =1 {}, =2 {}, ≥3 {}] ‖ 最大残余: {}",
                ctx.memo.len(),
                live_bytes >> 20,
                hist[0], hist[1], hist[2], hist[3],
                users_hist[0], users_hist[1], users_hist[2], users_hist[3],
                top.join(" ｜ ")
            );
        }
        let root_ids: std::collections::HashSet<u64> = out.iter().map(|b| b.id).collect();
        let mut candidates: Vec<u64> = std::mem::take(&mut ctx.arena);
        candidates.retain(|id| !root_ids.contains(id) && ctx.arena_set.contains(id));
        if !candidates.is_empty() {
            ctx.face.free(&candidates).await?;
        }
        Ok(out)
    })
}

// ============================================================================
// 段求值面 —— **已随 piecewise 退役**(2026-10-18,W5)
// ============================================================================
//
// 原址:`BreakDecl`(断点登记)/ `EvalCarry`(跨段接力棒)/ `SegMode`(段求值口径)/
// `eval_segment`(分段求值 + 钉块覆盖 + 宿主标量槽补丁)。
//
// **为什么退役**:W3/W4 之后 FI 核留在捕获图内、调度表由窗外 `fi_plan_sync` 刷
// ⇒ 断点面零消费者;而它正是"接力棒每步只增(新节点 id)⇒ 段末 `memo.retain`
// 每步 16×O(N) ⇒ 端到端 O(步数²)"那条慢化病的宿主(设计卷 §3.3)。
// 需要分段求值时从 git 历史取回(`docs/piecewise-断点图-设计与实现.md` 已标退役)。

/// 带观测的求值(interpreter-tap.md §4.2):归约语义与 [`eval_ops`]
/// 完全一致,仅在每个节点 After 窗口发事件 —— 数据读取(Stats/Bytes)
/// 由解释器代执行,tap 零执行权(观测窗口律,候选 §四 律 25)。
pub fn eval_ops_tap<'a, D>(
    t: &'a TensorOps,
    face: &'a mut D,
    tap: &'a mut dyn Tap,
) -> Pin<Box<dyn Future<Output = Result<Bytes, ModelError>> + Send + 'a>>
where
    D: crate::contract::DeviceClient + 'a,
{
    Box::pin(async move {
        let env = crate::env::EnvProvider::default();
        let perf = PerfCtx::of(&env);
        let mut ctx =
            EvalCtx { face, memo: std::collections::HashMap::new(), tap: Some(tap), arena: Vec::new(), env,
                arena_set: std::collections::HashSet::new(),
                block_users: std::collections::HashMap::new(),
                pending: std::collections::HashMap::new(), reclaim: false, perf, pool: None,
            tp: None,
        };
        _eval_rec(t, &mut ctx).await
    })
}

/// 求值上下文(CSE 备忘录 + 后端句柄 + 观测者)
struct EvalCtx<'a, D> {
    face: &'a mut D,
    memo: std::collections::HashMap<u64, Bytes>,
    /// 观测面(interpreter-tap.md;None = 零开销直通)
    tap: Option<&'a mut dyn Tap>,
    /// 本求值新建块 id 表(E2a 竞技场;Htod/Alloc/launch 输出在创建时登记,
    /// CSE 命中/Reshape 透传/Block 叶不登记)。scoped 口径下回收候选源。
    arena: Vec<u64>,
    /// 竞技场成员速查(中途回收/free 守卫;与 arena 同内容)
    arena_set: std::collections::HashSet<u64>,
    /// 每块被存活节点引用计数(track_new=1;透传臂 +1;节点死亡 -1,
    /// 归零且竞技场原生 → 中途 free)
    block_users: std::collections::HashMap<u64, usize>,
    /// 每节点剩余消费者数(预波;scoped 独占,reclaim=false 时为空)
    pending: std::collections::HashMap<u64, usize>,
    /// 活性中途回收开关(scoped = true;plain/tap 保历史语义)
    reclaim: bool,
    /// 性能面(level 入口读一次;热路径零全局查询)
    perf: PerfCtx,
    /// 解释器执行环境(EnvProvider;默认 = Cpu + 生产基线 —— Call 臂
    /// 对 Cpu 回退 face.op_env() 旧径,存量调用点零迁移)
    env: crate::env::EnvProvider,
    /// **块池**(P1″;`None` = 存量逐节点 alloc/free 路径不变)
    /// 只在 scoped(reclaim=true)入口装载 —— plain/multi/tap 口径
    /// (图捕获面)零改动:捕获窗分配会烙进图,不能改分配模式。
    pool: Option<BlockPool>,
    /// **张量并行上下文**(S4 ③';`None` = 单卡)。解释器在求值尾部按节点的
    /// [`TensorOps::tp_role`](crate::tensor::TensorOps::tp_role) 标记统一追加
    /// allreduce(层只给标记,不构造通信节点 —— 用户裁决 2026-10-17)。
    tp: Option<crate::module::TpCtx>,
}

impl<'a, D> EvalCtx<'a, D>
where
    D: crate::contract::DeviceClient,
{
    /// 新块登记:竞技场 + 成员速查 + 块归属 = 1。
    /// 池命中会复用**已登记**的 id(它本就在 arena 里)⇒ 只在首次见到该 id
    /// 时推 arena,避免候选表出现重复项(重复项不会错值,但会令
    /// `srv.count.free_blocks` / 候选表口径失真)。
    fn track_new(&mut self, id: u64) {
        owl_shared::slab_hint::live_diag::new_block();
        if self.arena_set.insert(id) {
            self.arena.push(id);
        } else {
            // **同一块被登记两次**(记账重复 ⇒ users 恒多 1 ⇒ 永不归还;诊断计数)
            owl_shared::slab_hint::live_diag::double_new(id, id, 0, self.memo.len());
        }
        let u = self.block_users.entry(id).or_insert(0);
        *u += 1;
        // 引用计数推送(诊断;"它凭什么是活的"面;探针关 = 零动作)
        owl_shared::slab_hint::probe_refs(id, *u, self.pending.get(&id).copied().unwrap_or(0));
    }

    /// **memo 写入单点**:登记节点值。
    fn memo_put(&mut self, node_id: u64, b: Bytes) {
        self.memo.insert(node_id, b);
    }

    /// 透传别名登记(Reshape/SlotWrite:节点共享父块;仅竞技场原生块)
    fn track_alias(&mut self, id: u64) {
        if self.arena_set.contains(&id) {
            let e = self.block_users.entry(id).or_insert(0);
            if *e > 0 {
                owl_shared::slab_hint::live_diag::alias_multi();
            }
            *e += 1;
            let u = *e;
            owl_shared::slab_hint::probe_refs(
                id,
                u,
                self.pending.get(&id).copied().unwrap_or(0),
            );
        }
    }

    /// 输出块获取(**块池面**;P1″):命中池 = **零命令**复用死块;
    /// 未命中 = 落 server 真分配并登记容量。
    /// `zeroed`(`Op::Zeros` 语义)在命中时必须补一次设备侧 memset ——
    /// 池块内容 = 上一位使用者的旧值(契约:Zeros 即零化)。
    /// **不在此登记竞技场**:`track_new` 由调用方统一调用(单一拼写;
    /// 命中/未命中在调用方眼里都是「一个新输出块」)。
    /// ★ **1-stage 的输出槽**(2026-10-18;`TP2-施工设计.md` §35):
    /// 把"要做 AR 的那一个输出"**直接放进对称槽** —— 上游 GEMM 写它 ⇒ 对端就
    /// 能读它 ⇒ 融合核(`owl_comm_ar_1stage_*`)原地加,**stage 段整段消失**。
    ///
    /// 返回 `None` 的两种情况(**都退回 v1 两段**,不静默错):
    /// ① 引擎没开 arena(老路径 / 单卡);② 槽装不下本次载荷。
    /// 槽选择 = `TpArenaSlots::next_pair()`(乒乓 + 建图期递增序号)。
    ///
    /// ⚠️ 返回值是**重新构造**的 `Bytes`(元素数 = `n_elems`)—— 因为槽本身是
    /// `U8` 块(`len` 是字节数),直接 `view` 会让 `len` 单位与 dtype 错位。
    fn symmetric_out_slot(&self, is_ar: bool, dtype: Dtype, n_elems: usize) -> Option<Bytes> {
        if !is_ar {
            return None;
        }
        let a = self.tp.as_ref()?.arena.as_ref()?;
        let need = n_elems * dtype.size_bytes();
        if (a.slot_bytes as usize) < need {
            return None; // 槽装不下 ⇒ 回退 v1(载荷 > slot_bytes 在注入点另有响亮拒)
        }
        let (own, _peer) = a.next_pair();
        // ⚠️ rank 留 0:由调用点 `.view(dtype, shape.len())` 定形 —— 与私有块路径
        // **逐参同款**(算子准入按声明校验 rank;2026-10-18 这里给 1 撞过
        // `cublas_gemm_f16: out 期望 rank = 2 实得 1`)。
        Some(Bytes::new(own.id, n_elems, dtype, 0))
    }

    /// 块获取(真分配或池命中)。
    ///
    /// `name` = **分配点点名**(全量记名纪律;传给 `face.alloc` ⇒ 服务端块账房
    /// `BlockEntry.name` ⇒ OOM 现场按名清单)。池**命中**分支无需记名:块 id 不变
    /// ⇒ 账房里的名字仍是该块**首次出生**时那一次(块身份与"这个槽位是干什么的"
    /// 语义未变 —— 同一张图的同一个槽)。
    async fn alloc_out(
        &mut self,
        dtype: Dtype,
        n_elems: usize,
        zeroed: bool,
        name: &str,
    ) -> Result<Bytes, ModelError> {
        let n_bytes = n_elems * dtype.size_bytes();
        let hit = self.pool.as_mut().and_then(|p| p.take(n_bytes));
        if let Some(id) = hit {
            // 池命中**必须照样计入 slab_hint 计量**(与 server 真分配侧同口径):
            // 捕获 slab 的定量靠 GraphPlan warmup dry-run 计量闭包需求,而捕获窗
            // 走 plain 入口(pool=None)⇒ 若命中不计,量得的 slab 会**偏小**,
            // 捕获当场炸「捕获 slab 耗尽」(2026-10-14 实测:27B spec-dflash 开池
            // 即 boot 失败,slab 7.69MB 差 81KB)。计量语义 = 闭包需求,与"是否复用"无关。
            owl_shared::slab_hint::meter_add(n_bytes as u64);
            // 块尚未按形状定形:rank = 0(未定形;dtype 已知)——
            // 视图事实由调用点(张量节点)在发射时给出
            let b = Bytes::new(id, n_elems, dtype, 0);
            if zeroed {
                self.face.memset_zero(&b, n_bytes).await?;
            }
            return Ok(b);
        }
        let b = if zeroed {
            self.face.alloc(dtype, n_elems, name).await?
        } else {
            self.face.alloc_uninit(dtype, n_elems, name).await?
        };
        if let Some(p) = self.pool.as_mut() {
            p.register(b.id, n_bytes);
        }
        Ok(b)
    }

    /// 归还一个已死块(DAG 上再无消费者):入池 = **零命令**;否则真 free。
    async fn release_block(&mut self, id: u64) -> Result<(), ModelError> {
        if self.pool.as_mut().is_some_and(|p| p.give_back(id)) {
            return Ok(());
        }
        owl_shared::slab_hint::live_diag::freed();
        self.face.free(&[id]).await
    }

    /// 收尾:抽干池,**一条批量 free**(替代 N 条逐块 free)。
    async fn flush_pool(&mut self) -> Result<(), ModelError> {
        let Some(mut p) = self.pool.take() else {
            return Ok(());
        };
        let ids = p.drain();
        let st = p.stats();
        if st.hit + st.miss > 0 {
            // 池账落 metrics(counters 级;`/debug/metrics/interp.pool.`)
            perf_cnt("interp.pool.hit", st.hit);
            perf_cnt("interp.pool.miss", st.miss);
            perf_cnt("interp.pool.in_pool", st.in_pool);
            perf_cnt("interp.pool.spilled", st.spilled);
            perf_cnt("interp.pool.flush_blocks", st.flush_blocks);
            perf_cnt("interp.pool.peak_cached_bytes", st.peak_cached_bytes as u64);
        }
        if !ids.is_empty() {
            self.face.free(&ids).await?;
        }
        Ok(())
    }
}

/// 预波:每节点剩余消费者计数(每条入边 +1;根 +1 = 调用方持根)。
/// 迭代显式栈(DAG 可数千节点深,避开递归深度);visited 保证每节点
/// 处理一次,每条边在其消费节点的访问时计一次。
/// **多根 pending(边只数一次)**(2026-10-19 定谳;设计卷 §8.31)。
///
/// 旧口径 = 逐根 `count_pending(r)` 后**相加**。它对"副作用树"(roots 互不相交:
/// K0 写哑根 / encode,2026-10-10 的立项场景)是对的,但**对"根套根"错**:
/// `eval_root_and_q` 的 roots = `[hidden, q₁..qₙ]`,而 Q/tap 节点**就在 hidden 的
/// 子树里** ⇒ 共享子树的每条边被数 k 次,而执行只发生一次(节点求值 + memo)
/// ⇒ `pending` 恒定多算 ⇒ 活性回收的"归零"永不成立 ⇒ **中途一块都不还**。
///
/// 实测(B 轴档 27B prefill chunk 128):真回收 **14** 块 / 新块 1431;影子 arena
/// 峰值 2064 MiB ≈ 分配总量 2.0 GB(逐名同比例)⇒ 单 chunk 激活面 ≈ 48 层全摊。
/// 修后同一档峰值应回到"峰值活跃集"量级(1–3 层)。
///
/// 不变量:**一张图数边一次**(每个节点只贡献自己的 `parents`),每个根另 +1
/// (调用方持根 ⇒ 根永不中途回收 —— Q 槽/回灌面在 eval 结束后才读它们)。
fn count_pending_multi(roots: &[&TensorOps]) -> std::collections::HashMap<u64, usize> {
    let mut pending: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut visited = std::collections::HashSet::new();
    for r in roots {
        // 调用方持有的根:带 +1(与单根口径同律)
        *pending.entry(r.id).or_insert(0) += 1;
        let mut stack = vec![*r];
        while let Some(n) = stack.pop() {
            // **跨根共享 visited** ⇒ 同一节点只贡献一次边(修的就是这里)
            if !visited.insert(n.id) {
                continue;
            }
            for p in &n.parents {
                *pending.entry(p.id).or_insert(0) += 1;
                if !visited.contains(&p.id) {
                    stack.push(p);
                }
            }
        }
    }
    pending
}

/// 单根 pending(lease:多根走 [`count_pending_multi`])
#[allow(dead_code)]
fn count_pending(root: &TensorOps) -> std::collections::HashMap<u64, usize> {
    let mut pending: std::collections::HashMap<u64, usize> = std::collections::HashMap::new();
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![root];
    *pending.entry(root.id).or_insert(0) += 1;
    while let Some(n) = stack.pop() {
        if !visited.insert(n.id) {
            continue;
        }
        for p in &n.parents {
            *pending.entry(p.id).or_insert(0) += 1;
            if !visited.contains(&p.id) {
                stack.push(p);
            }
        }
    }
    pending
}

/// DAG 求值入口(装箱:async 递归要求;'b 短借用 reborrow)。
fn _eval_rec<'a, 'b, D>(
    t: &'a TensorOps,
    ctx: &'b mut EvalCtx<'a, D>,
) -> Pin<Box<dyn Future<Output = Result<Bytes, ModelError>> + Send + 'b>>
where
    D: crate::contract::DeviceClient + 'a,
    D: Send,
{
    Box::pin(_eval_node(t, ctx))
}

async fn _eval_node<'a, D>(
    t: &'a TensorOps,
    ctx: &mut EvalCtx<'a, D>,
) -> Result<Bytes, ModelError>
where
    D: crate::contract::DeviceClient,
{
    // **单步/粘滞调试门**(2026-10-14;`step_gate!` 不带 feature 时为空展开):
    // 放在 CSE 判定**之前** ⇒ "一步 = 一次节点访问"(含命中),行为可预测。
    // 见 owl_shared::stepgate 模块头(off/sticky/single + tx/rx 外部 step 信号)。
    owl_shared::step_gate!("interp.node").await;
    // CSE / DAG 求值(2026-09-26 定案):共享子树(如 DecoderLayer 的残差
    // h:既喃 ln2 又喃残差加)只求值一次 —— 无 CSE 则 mixer 的状态副作用
    // kernel(conv/delta)执行两遍,第二遍读到滑过的状态 → 语义破坏。
    // memo 键 = 节点 id(全局自增,克隆保留 → 同节点 = 同值)。
    if let Some(b) = ctx.memo.get(&t.id) {
        // 性能面:CSE 命中 = 白赚的节点(无计时样本 → counter 记)
        if owl_shared::metrics::PERF_BUILD
            && ctx.perf.level >= owl_shared::metrics::level::COUNTERS
        {
            perf_cnt("interp.count.cse_hit", 1);
        }
        return Ok(b.clone());
    }
    if let Some(e) = &t.err {
        let detail = format!("[毒值落地 @depth {}] {}", e.at_depth, e.detail);
        // Tap:Poison 事件(毒值节点无输出块,无 After;归约照旧 Err)
        if let Some(tap) = ctx.tap.as_deref_mut() {
            tap.on_poison(t.id, &detail);
        }
        return Err(ModelError::Msg(detail));
    }
    let mut vals: Vec<Bytes> = Vec::with_capacity(t.parents.len());
    let mut ins: Vec<Arg> = Vec::with_capacity(t.parents.len());
    for p in &t.parents {
        let b = _eval_rec(p, ctx).await?;
        // 刀1:切片视图父 → Arg::BlockSlice(ptr = 块首 + offset·esz);
        // 其余 → Arg::Block。偏移在节点声明期已含父链合成。
        let a = match &p.op {
            Op::SliceView { offset_elems } => Arg::block_slice(
                b.id,
                (offset_elems * p.dtype.size_bytes()) as u64,
                p.shape.iter().product::<usize>() as u64,
                p.dtype,
                p.shape.len() as u8,
            ),
            // 事实 = **本父节点的视图**(dtype + shape 阶数),不是块句柄的标注 ——
            // 块句柄经 Reshape/SliceView 透传时会保留**底块**的标注(下面两臂已重定形)
            _ => Arg::block(b.id, p.dtype, p.shape.len() as u8),
        };
        vals.push(b);
        ins.push(a);
    }

    // C1 边界断言:非 Block 父节点的块账长 = 声明元素数(Block 叶子 len=0 无
    // 语义;SliceView 父 = 块首切片,账长 = 父块全长,不在此检)。维度推导
    // 一律不用 len(见下方各分支)。
    //
    // 门谓词 = 仓内统一 `PERF_BUILD`(2026-10-17 审查 §六轮:原 `cfg!(debug_assertions)`
    // 是**第二套谓词**;`PERF_BUILD = debug_assertions ∨ feature=perf` ⇒ 非 perf 的
    // release 里整段常量折叠(逐字节零成本),而 `--features perf` 能在 release 档
    // 打开这类账本检查 —— 与 `cuda/state.rs` 的既有用法同源)。
    if owl_shared::metrics::PERF_BUILD {
        for (i, (p, b)) in t.parents.iter().zip(&vals).enumerate() {
            if matches!(p.op, Op::SliceView { .. }) {
                continue;
            }
            let want: usize = p.shape.iter().product();
            if b.len != 0 && b.len != want {
                panic!(
                    "[C1] 块账长 != 声明元素数: node {} op {:?} parent#{i} id {} op {:?} 声明 {:?} 账长 {}",
                    t.id, t.op, p.id, p.op, p.shape, b.len
                );
            }
        }
    }

    let dtype = t.dtype;
    let shape = t.shape.clone();
    let n_elems: usize = shape.iter().product::<usize>();
    let n_bytes = n_elems * dtype.size_bytes();

    // ── 性能面:本节点**排他**宿主计时 ─────────────────────────────
    // 起表点在本行(父递归/ins 构造已完成)⇒ 样本 = 该节点自身的宿主成本;
    // 包住子节点的账留给外层(调用树由报告按 tag 层次承担)。
    let perf = ctx.perf;
    let t_node = if perf.host() { Some(std::time::Instant::now()) } else { None };
    // 类名先按内建词表;`Call` 臂 resolve 后用**核名**覆写(登记表有界集)
    let mut class: &'static str = builtin_class(&t.op);

    let out: Bytes = match &t.op {
        Op::Htod { bytes } => {
            // **参数面走 COMPUTE 流**(2026-10-17;用户裁决):图/核的输入叶子
            // (pos/slots/lens/id/gdn_slot/fi_* 等)与消费它的发射同流 FIFO ⇒
            // 结构上不再依赖"等回执"的时序运气(H2D 流忙时必翻车的那类竞态)。
            // 大批量(权重/装载材料)仍走 H2D:不占计算流。
            const ARG_PLANE_MAX_BYTES: usize = 64 << 10;
            let stream = if n_bytes <= ARG_PLANE_MAX_BYTES {
                owl_iface::contract::UpStream::Compute
            } else {
                owl_iface::contract::UpStream::H2d
            };
            let b = ctx.face.htod_on(stream, dtype, &shape, bytes, class).await?;
            ctx.track_new(b.id);
            b
        }
        Op::Zeros => {
            let b = ctx.alloc_out(dtype, n_elems, true, class).await?.view(dtype, shape.len() as u8);
            owl_shared::slab_hint::probe_name(b.id, class);
            ctx.track_new(b.id);
            b
        }
        // 无清零分配(scratch;内容未定义 —— 契约见 `Op::Uninit`)
        Op::Uninit => {
            let b = ctx.alloc_out(dtype, n_elems, false, class).await?.view(dtype, shape.len() as u8);
            owl_shared::slab_hint::probe_name(b.id, class);
            ctx.track_new(b.id);
            b
        }
        // 裸块叶:事实取本节点声明(dtype + 阶数)
        Op::Block { id } => Bytes::new(*id, 0, dtype, shape.len() as u8),
        // 纯元数据视图:透传父块(零拷贝)—— 但**事实面按本节点视图重定形**
        // (否则下游按父块标注发射 ⇒ rank/dtype 错,2026-10-15 事实面门实测)
        Op::Reshape => {
            ctx.track_alias(vals[0].id);
            vals[0].clone().view(dtype, shape.len() as u8)
        }
        // 刀1:连续切片视图(与 Reshape 同族透传;偏移在消费面由节点 op
        // 合成 Arg::BlockSlice,此处块句柄 = 父块全长 —— CSE/回收按块计)
        Op::SliceView { .. } => {
            ctx.track_alias(vals[0].id);
            vals[0].clone().view(dtype, shape.len() as u8)
        }
        Op::Matmul | Op::MatmulNt => {
            let (m, n) = (shape[0], shape[1]);
            // k = 内维 = x 声明 shape 末维(C1:只读声明 shape;原取 ins[0].len
            // 在多行 [m,k] 时越界读 —— 此 bug 被"历届测试都单行"掩盖)
            // k = 内维 = x 声明 shape 末维(C1:只读声明 shape;原取 ins[0].len
            // 在多行 [m,k] 时越界读 —— 此 bug 被"历届测试都单行"掩盖)。
            // 空形状此处**不可达**(2026-10-17 审查 §六轮定判)两点:
            // ①`TensorOps::matmul/matmul_nt` 声明面已按 rank ≥ 1 律毒化;
            // ②本函数在执行本臂前先经父值求值,父节点带毒 ⇒ 更早以
            // "[毒值落地]" 返回 Err(见上方 `t.err` 门)。故 `unwrap_or(0)` 只是
            // 形式默认,不会把 0 送进几何。
            let k = t.parents[0].shape.last().copied().unwrap_or(0);
            // ★ **1-stage(§35)**:Matmul 的输出就是 AR 的 payload(最常见的那条)
            // —— 必须也走对称槽,否则融合核永远拿不到 out。
            // ⚠️ 2026-10-18 踩过:只改了 `Op::Call` 的 alloc 点 ⇒ 1-stage **从未执行**
            // (融合核 0 次、两段核照跑 1344 次),而门全绿 ⇒ "中性"结论是假的。
            let out = match ctx.symmetric_out_slot(
                t.tp_role == crate::plan::rules::WeightRule::UniformRow,
                dtype,
                m * n,
            ) {
                Some(b) => b.view(dtype, shape.len() as u8),
                None => ctx
                    .alloc_out(dtype, m * n, false, class)
                    .await?
                    .view(dtype, shape.len() as u8),
            };
            owl_shared::slab_hint::probe_name(out.id, class);
            ctx.track_new(out.id);
            if dtype == Dtype::F16 || dtype == Dtype::BF16 {
                // f16/bf16:foreign-kernel 通道(cuBLAS;nt = owl Linear 惯例)。
                // gemm 侧 cm 约定:m = n_out(权重行)/ n = T(2026-09-26 修正:
                // 原误传 (m=T, n=n_out),T=1 即 B 操作数越界读 → ILLEGAL_ADDRESS)
                // bf16 臂(E5-DF3 同日十四;DFlash2 草稿非量化投影)
                let nt = matches!(t.op, Op::MatmulNt);
                let msg = crate::ops::lower_gemm(&ins, &out, n, k, m, nt, dtype == Dtype::BF16);
                ctx.face.launch(msg).await?;
            } else {
                let msg = if matches!(t.op, Op::MatmulNt) {
                    crate::ops::lower_matmul_nt(&ins, &out, m, k, n)
                } else {
                    crate::ops::lower_matmul(&ins, &out, m, k, n)
                };
                ctx.face.launch(msg).await?;
            }
            out
        }
        Op::Call { op, aux } | Op::CallDst { op, aux } => {
            // **CallDst**:parents[0] = dest(写入既有块;不新分配)。余下父 = 核输入。
            let dst_in = if matches!(t.op, Op::CallDst { .. }) { Some(vals[0].clone()) } else { None };
            let is_dst = dst_in.is_some();
            let slot_ins: &[crate::contract::Arg] =
                if dst_in.is_some() { &ins[1..] } else { &ins[..] };
            // 性能面:宿主链分段(选核 / 构件 / 分配 / 线格式构造 / 发射)
            let mut lap = if perf.host() { Some(LapTimer::start()) } else { None };
            // 硬件感知拾取(被动律):环境必传 —— hw 自 face(引擎感知),
            // page 自页策略单源;kernels/解释器零探测。CPU face 无环境 =
            // 结构化"需 GPU server"(与 Kernel 节点 CPU 口径同)。
            // 环境必传(被动律);face 无环境(CPU face,测试域)= Sm86
            // 占位 —— 纯语义臂(add/mul/silu/sigmoid/rmsnorm)不消费 env,
            // 硬件感知臂在 CPU face 上本就不可达(launch 面结构化拒)
            let mut env = ctx.face.op_env().unwrap_or(owl_kernels::driver::OpEnv {
                hw: owl_kernels::driver::Hw { arch: owl_kernels::Arch::Sm86 },
                page: 0,
                head_dim: 0,
                grp: 0,
            });
            env.page = crate::module::kv_paged_policy(dtype).map(|p| p.page).unwrap_or(0);
            // 几何事实自 env 单源补齐(与页同律:解释器 = env 注入点)
            env.head_dim = ctx.env.kv.head_dim;
            env.grp = ctx.env.kv.grp;
            let skip = if dst_in.is_some() { 1 } else { 0 };
            let shapes: Vec<Vec<usize>> =
                t.parents.iter().skip(skip).map(|p| p.shape.clone()).collect();
            let scalars: Vec<i64> = t.args.iter().filter_map(|a| match a {
                crate::ops::KernelArg::Bits(v) => Some(*v as i64),
                crate::ops::KernelArg::I32(v) => Some(*v as i64),
                _ => None,
            }).collect();
            let pick = owl_kernels::driver::resolve(owl_kernels::driver::OpReq {
                op: op.op_id(),
                env: &env,
                dt: crate::ops::ddt(dtype)?,
                shapes: &shapes,
                aux,
                scalars: &scalars,
            });
            if let Some(l) = lap.as_mut() {
                l.lap("interp.host.seg.resolve");
            }
            class = pick.name; // 类名 = 核名(报告里与图材料单同词表)
            if ctx.env.diag.resolve_trace {
                eprintln!("[call] {:?} -> {} grid={:?} block={:?} smem={} n_elems={}",
                    op, pick.name, pick.shape.grid, pick.shape.block, pick.shape.smem, n_elems);
            }
            // 统一构造(2026-10-13 P3 算子准入):
            // ① 已建表算子(meta 在场)⇒ 名字/源码/槽序全取算子元数据,
            //    foreign 家族臂的 SOURCE 为空 ⇒ server 走家族 FFI;
            // ② 未建表 foreign 臂 ⇒ 家族线契约 sig(face 单源);
            // ③ 未建表 native ⇒ 登记表(仅引入)
            let kernel = if pick.meta().is_some() {
                crate::kernel::with_pick(pick)
            } else {
                match crate::ops::foreign_sig(pick.name) {
                    Some(sig) => crate::kernel::Kernel::new(pick.name, "").with_sig(sig),
                    None => crate::kernel::with_pick(pick),
                }
            };
            // 以下 = 统一发射面(登记表 dtype 守门 + alloc + lower + launch)
            // dtype 守门(F 轮):口径 = **算子声明的输出槽 dtype**(原登记表
            // `Entry.dtype` 的等价物,已从声明面派生)
            if let Some(od) = crate::kernel::lookup(kernel.name).and_then(|m| m.out_dtype()) {
                if od != dtype {
                    return Err(ModelError::Msg(format!(
                        "[dtype 守门] kernel \"{}\" 声明输出为 {:?},实得 {:?}\n{}",
                        kernel.name, od, dtype,
                        std::backtrace::Backtrace::force_capture()
                    )));
                }
            }
            if let Some(l) = lap.as_mut() {
                l.lap("interp.host.seg.build");
            }
            // ★ **1-stage(§35)**:要做 AR 的输出直接落进对称槽(GEMM 写它 ⇒ 对端可读)。
            // 无 arena / 槽装不下 = 退回私有块 + v1 两段(不静默错)。
            let out = match dst_in {
                // dest = 既有块(同块别名;不分配、不 memset)
                Some(b) => {
                    ctx.track_alias(b.id);
                    b
                }
                None => match ctx.symmetric_out_slot(
                    t.tp_role == crate::plan::rules::WeightRule::UniformRow,
                    dtype,
                    n_elems,
                ) {
                    Some(b) => b.view(dtype, shape.len() as u8),
                    None => ctx
                        .alloc_out(dtype, n_elems, false, class)
                        .await?
                        .view(dtype, shape.len() as u8),
                },
            };
            // 峰值归因:本输出块挂上核名(`class` 已在上面 resolve 处覆写为核名)
            owl_shared::slab_hint::probe_name(out.id, class);
            if let Some(l) = lap.as_mut() {
                l.lap("interp.host.seg.alloc");
            }
            // dest 复用既有块 ⇒ **不**新记一次(否则同一块被当两个 node 记账,
            // live 账虚高、槽位判断误伤;实测 27B 触发 `live-dup` 与池耗尽)
            if !is_dst {
                ctx.track_new(out.id);
            }
            let msg = crate::ops::lower_kernel(&kernel, &t.args, slot_ins, &out, n_elems);
            if let Some(l) = lap.as_mut() {
                l.lap("interp.host.seg.lower");
            }
            ctx.face.launch(msg).await?;
            if let Some(l) = lap.as_mut() {
                l.lap("interp.host.seg.launch");
            }
            out
        }
        Op::SlotWrite => { ctx.track_alias(vals[0].id); vals[0].clone() }
        #[allow(unreachable_patterns)] // 防御臂:Op 新增变体时给运行时错值而非静默
        other => return Err(ModelError::Msg(format!("eval 未覆盖: {other:?}"))),
    };
    // ★ **S4 ③':TP 行并行 ⇒ 解释器统一追加 allreduce**(层只给标记;
    //   用户裁决 2026-10-17:感知下沉到解释器层)。
    //   标记来源 = 节点上的 `tp_role`(`Linear::new` 从规则表算好透传);
    //   语义 = payload 原地累加(输出仍是 `out` 的同块别名)。
    let out: Bytes = if t.tp_role == crate::plan::rules::WeightRule::UniformRow {
        match ctx.tp.as_ref() {
            Some(tp) => {
                if n_bytes as u64 > tp.slot_bytes {
                    return Err(ModelError::Msg(format!(
                        "TP allreduce: 载荷 {n_bytes}B > slot {}B(v1 单块;多块是 S5 ——                          调大 staging 或改计划)",
                        tp.slot_bytes
                    )));
                }
                // ★ **rank 标记点(K1;用户裁决"注入点打标")** —— 层键不在
                //   节点上(layer 零改动的代价),故标记 = 静态 tag 计数(per
                //   rank;store 缺席 = 静默跳过,纯观测面)+ debug 日志(节点
                //   id/label/字节数;log 未开 = 编译期零成本)。
                //   位置 = 决策点(发射之前):即便发射失败,标记也如实记下
                //   "此处该 AR"(观测与执行解耦)。
                let rank = tp.plan.rank();
                let tag = match rank {
                    0 => "tp.ar.r0",
                    1 => "tp.ar.r1",
                    _ => "tp.ar.rN",
                };
                owl_shared::metrics::with_metrics_store(|s| {
                    s.counter_add(tag, 1, file!(), line!())
                });
                owl_shared::log_debug!(
                    "tp",
                    "AR 注入 rank={rank} node={} label={:?} bytes={n_bytes} slot={}",
                    t.id(),
                    t.label(),
                    tp.slot_bytes
                );
                // ★ **1-stage(§35)**:本次输出落在对称槽里 ⇒ 融合核**原地加对端槽**
                // (没有 stage 段、没有拷贝、少一个图节点)。判据 = `out.id` 是不是
                // arena 的槽(alloc 时选的那一个)—— 不需额外字段/标志。
                // ★ **路径存在性判据**(2026-10-18):`tp.ar.1stage` / `tp.ar.v1`
                // 两个 counter —— 门断言"开 1-stage 时前者 > 0 且后者 == 0"。
                // 动机是现场教训:融合核写好了、门全绿,但它**一次都没被走到**
                // (只改了 `Op::Call` 的 alloc,payload 却来自 `Op::Matmul`)。
                let fused = match tp.arena.as_ref() {
                    Some(a) => match a.own_slots.iter().position(|s| s.id == out.id) {
                        Some(i) => {
                            let call = owl_kernels::client::comm::Ar1StageCall::new(
                                out.clone(),
                                a.own_hdr.clone(),
                                a.peer_hdr.clone(),
                                a.peer_slots[i].clone(),
                                n_bytes as u64,
                            )
                            .to_launch();
                            ctx.face.launch(call).await?;
                            owl_shared::metrics::with_metrics_store(|s| {
                                s.counter_add("tp.ar.1stage", 1, file!(), line!())
                            });
                            true
                        }
                        None => false,
                    },
                    None => false,
                };
                if !fused {
                    // ★ **push 档(§36.6/§36.7)**:两核把 `own`/`peer` **实参对调** ——
                    //   v1 : stage 写**本端**槽 → reduce 读**对端**槽（数据腿 = 读对端）
                    //   push: stage 写**对端**槽 → reduce 读**本端**槽（数据腿 = 写对端）
                    // 协议（credit / produced / rendezvous / 两级等待）**逐字不变** ——
                    // 两个游标的相对比较在两种绑法下都成立（推导见 §36.6）。
                    // ★ **push 档(§36.6/§36.7)**:header(`own`/`peer`)两档**都不变**;
                    // 变的只是**数据块** —— stage 写哪里 / reduce 读哪里。
                    //   v1 : stage 写**本端**槽 → reduce 读**对端**槽(数据腿 = 读对端)
                    //   push: stage 写**对端**槽 → reduce 读**本端**槽(数据腿 = 写对端)
                    // 为什么 header 不动:`wait_done`/`blocks_done` 是**本核内部的块间
                    // 令牌** —— 把它们搬到对端会让 15 个 block 紧自旋**远端读**,实测把
                    // AR 从 601µs 拉到 1225µs(§36.8)。协议(credit/produced/rendezvous)
                    // 在两种绑法下都成立(推导见 §36.6)。
                    let (stage_data, reduce_data) = if tp.push {
                        owl_shared::metrics::with_metrics_store(|s| {
                            s.counter_add("tp.ar.push", 1, file!(), line!())
                        });
                        (tp.peer.clone(), tp.own.clone())
                    } else {
                        owl_shared::metrics::with_metrics_store(|s| {
                            s.counter_add("tp.ar.v1", 1, file!(), line!())
                        });
                        (tp.own.clone(), tp.peer.clone())
                    };
                    // 两段(私有 payload ⇒ 先 stage 到 staging,再归约对端/本端 slot)
                    let stage = owl_kernels::client::comm::ArStageCall::new(
                        out.clone(),
                        tp.own.clone(),
                        tp.peer.clone(),
                        stage_data,
                        n_bytes as u64,
                        tp.slot_bytes,
                        0,
                    )
                    .to_launch();
                    ctx.face.launch(stage).await?;
                    let reduce = owl_kernels::client::comm::ArReduceCall::new(
                        out.clone(),
                        tp.own.clone(),
                        tp.peer.clone(),
                        reduce_data,
                        n_bytes as u64,
                        tp.slot_bytes,
                        0,
                    )
                    .to_launch();
                    ctx.face.launch(reduce).await?;
                }
                out // 别名:payload 原地累加
            }
            None => out, // 单卡:无 TP 上下文
        }
    } else {
        out
    };
    ctx.memo_put(t.id, out.clone());
    // ── 性能面:本节点账落地(排他宿主时;tag = interp.host.node.<phase>.<class>)──
    if let Some(t0) = t_node {
        perf_rec(
            &format!("interp.host.node.{}.{}", perf.phase, class),
            t0.elapsed(),
        );
    }
    // ── 活性中途回收(2026-10-02;scoped 独占;在 Tap 窗口后 —— 观测窗
    // 律要求父输入窗口内存活)── 逐父递减剩余消费者数;计零 = 节点死亡
    // → memo 摘除 + 块归属减一;归零且竞技场原生且非本节点输出(透传
    // 别名防误焚)→ 立即 free。流序安全:消费 kernel 已入队,free 同流
    // 排在其后;根带 +1 永不中途回收。
    if ctx.reclaim {
        for (p, b) in t.parents.iter().zip(vals.iter()) {
            let cnt = ctx.pending.entry(p.id).or_insert(0);
            // 注:此处 cnt 尚未减,pending 推送在减完后(见下方;与 users 同点)
            if *cnt > 0 {
                *cnt -= 1;
            }
            if *cnt == 0 {
                ctx.memo.remove(&p.id);
                let bid = b.id;
                if let Some(u) = ctx.block_users.get_mut(&bid) {
                    if *u > 0 {
                        *u -= 1;
                    }
                    let left_u = *u; // ← 取完即放(下面要调 ctx.* ⇒ 不能揣着可变借)
                    // 引用计数推送(诊断):此刻 users 已减、pending 已减(上面那行)
                    owl_shared::slab_hint::probe_refs(
                        bid,
                        left_u,
                        ctx.pending.get(&bid).copied().unwrap_or(0),
                    );
                    if left_u == 0 {
                        // 回收出口四个判定各记一笔(诊断;见 `slab_hint::live_diag`)
                        use owl_shared::slab_hint::live_diag::{outcome, refuse, Refuse};
                        // **决定点落账**(按名分账:字节/次数/残余 users)——
                        // 尺寸口径 = 视图元素数 × dtype(别名块可能小于底层块,量级够用)
                        let bytes = b.len.saturating_mul(b.dtype.size_bytes());
                        if bid == out.id {
                            refuse(Refuse::Alias);
                            outcome(bid, bytes, Some(Refuse::Alias), left_u);
                        } else if !ctx.arena_set.contains(&bid) {
                            refuse(Refuse::NotArena);
                            outcome(bid, bytes, Some(Refuse::NotArena), left_u);
                        } else {
                            outcome(bid, bytes, None, 0);
                            ctx.arena_set.remove(&bid);
                            ctx.release_block(bid).await?;
                        }
                    } else {
                        owl_shared::slab_hint::live_diag::refuse(
                            owl_shared::slab_hint::live_diag::Refuse::Users,
                        );
                        // 决定点被 users 拦下:落账(残余 users 一并记下)
                        let bytes = b.len.saturating_mul(b.dtype.size_bytes());
                        owl_shared::slab_hint::live_diag::outcome(
                            bid,
                            bytes,
                            Some(owl_shared::slab_hint::live_diag::Refuse::Users),
                            left_u,
                        );
                    }
                } else {
                    owl_shared::slab_hint::live_diag::refuse(
                        owl_shared::slab_hint::live_diag::Refuse::NotTracked,
                    );
                }
            } else {
                owl_shared::slab_hint::live_diag::refuse(
                    owl_shared::slab_hint::live_diag::Refuse::Pending,
                );
            }
        }
    }
    // ── Tap:After 窗口(输出已入 memo;父输入仍在 memo 存活)──
    // 观测窗口律:数据读取由解释器代执行并在窗口内完成;窗口外读块
    // 未定义(scratch 块 eval 结束后可被池复用)。tap 无 launch/alloc
    // 权 —— 观测在结构上不可能改变计算(对照 harvest 重放污染)。
    // f16 基线(F5):观测读回按块 dtype 解码(f32 硬解码会读歪 rms)
    if let Some(tap) = ctx.tap.as_deref_mut() {
        let ev = NodeEvent {
            id: t.id,
            op: &t.op,
            dtype,
            shape: &t.shape,
            depth: t.depth,
            tag: t.label.as_deref(),
            out: BlockRef { block_id: out.id, len: out.len },
        };
        match tap.on_node(&ev) {
            Want::Quiet | Want::Keep => {} // Keep:MVP 未实施,按 Quiet 落空
            Want::Stats => {
                let mut buf = vec![0u8; n_bytes];
                ctx.face.dtoh(&out, &mut buf).await?;
                let host = decode_host(dtype, &buf);
                tap.on_stats(t.id, BlockStats::of(&host));
            }
            Want::Bytes => {
                let mut buf = vec![0u8; n_bytes];
                ctx.face.dtoh(&out, &mut buf).await?;
                let host = decode_host(dtype, &buf);
                tap.on_bytes(&ev, &host);
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tp_injection_tests {
    //! **S4 ⑤-K1 注入口门**:解释器尾部按 `ctx.tp` 注入 allreduce ——
    //! ① 单卡(`tp = None`)零注入(求值照常成功);
    //! ② `tp = Some` ⇒ 真走到 comm 发射(CpuFace 无该核 ⇒ 响亮拒,
    //!    错误文本必须点名 **stage 核** —— 证明注入口生效且顺序不是乱发);
    //! ③ 标记:per-rank 计数进 metrics(决策点打标;静态 tag)。
    use super::*;
    use crate::contract::DeviceClient as _;
    use crate::plan::rules::WeightRule;
    use crate::plan::tp::TpPlan;
    use crate::tensor::TensorOps;

    fn row_node() -> TensorOps {
        TensorOps::zeros(Dtype::F32, vec![1, 2]).with_tp_role(WeightRule::UniformRow)
    }

    /// 单卡:tap 未开、无 TP 上下文 ⇒ 零发射(历史行为不变)
    #[tokio::test]
    async fn no_tp_means_no_injection() {
        let mut face = owl_cpu::CpuFace::new();
        let r = eval_ops_scoped_env(&row_node(), &mut face, crate::env::EnvProvider::default(), None).await;
        assert!(r.is_ok(), "tp = None 不应触发任何 comm 发射:{:?}", r.err());
    }

    /// TP:注入 stage→reduce;CpuFace 在 stage 核上响亮拒(且名字可断言)
    #[tokio::test]
    async fn tp_injects_ar_and_marks_rank() {
        let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
        let mut face = owl_cpu::CpuFace::new();
        let own = face.alloc(Dtype::F32, 16, "tp.own").await.expect("own staging 块(64B)");
        let peer = face.alloc(Dtype::F32, 16, "tp.peer").await.expect("peer staging 块(64B)");
        let tp = crate::module::TpCtx {
            plan: TpPlan::new(1, 2).expect("rank 1/2"),
            own,
            peer,
            slot_bytes: 64,
            arena: None,
            push: false,
        };
        let err = eval_ops_scoped_env(
            &row_node(),
            &mut face,
            crate::env::EnvProvider::default(),
            Some(&tp),
        )
        .await
        .expect_err("tp = Some 必须走到 comm 发射(CpuFace 无此核 ⇒ 响亮拒)");
        let msg = format!("{err}");
        assert!(
            msg.contains("owl_comm_ar_stage"),
            "注入的应是 stage 核(顺序：stage → reduce);实得: {msg}"
        );
        // 标记面:rank=1 的静态 tag 计数 ≥ 1(注入口在决策点打标)
        let hits: u64 = owl_shared::metrics::query_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag("tp.ar.r1"),
        )
        .iter()
        .map(|r| r.counter)
        .sum();
        assert!(hits >= 1, "rank 标记缺失(metrics tp.ar.r1;实得 {hits})");
    }

    /// **载荷超 slot ⇒ 响亮拒**(v1 单块;多块留给 S5)
    #[tokio::test]
    async fn payload_over_slot_is_loud() {
        let mut face = owl_cpu::CpuFace::new();
        let own = face.alloc(Dtype::F32, 4, "tp.own").await.expect("own(16B)");
        let peer = face.alloc(Dtype::F32, 4, "tp.peer").await.expect("peer(16B)");
        let tp = crate::module::TpCtx {
            plan: TpPlan::new(0, 2).expect("rank 0/2"),
            own,
            peer,
            slot_bytes: 4, // < 载荷 8B
            arena: None,
            push: false,
        };
        let err = eval_ops_scoped_env(
            &row_node(),
            &mut face,
            crate::env::EnvProvider::default(),
            Some(&tp),
        )
        .await
        .expect_err("超 slot 必须响亮拒");
        assert!(
            format!("{err}").contains("TP allreduce"),
            "拒收消息须点名 TP allreduce;实得: {err}"
        );
    }
}

#[cfg(test)]
mod liveness_tests {
    use super::*;
    use crate::contract::DeviceClient as _;
    use crate::tensor::TensorOps;
    use crate::testkit::f32b;

    /// 深链(128 级 Add):活性中途回收 → 中间节点逐个死亡,
    /// scoped 候选表应空(历史行为 = 128 块全落候选);根数据收割
    /// 不受影响(值 = 1+128)。CpuFace::free = 无操作,验的是记账面
    /// (候选裁剪 / 归属归零),GPU 域真 free 由引擎 prefill 链路覆盖。
    #[tokio::test]
    async fn scoped_liveness_chain_reclaimed() {
        let mut face = owl_cpu::CpuFace::new();
        let mut x = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0]));
        let c = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[1.0]));
        for _ in 0..128 {
            x = x.add(&c);
        }
        let (root, candidates) = eval_ops_scoped(&x, &mut face).await.expect("scoped eval");
        assert!(
            candidates.is_empty(),
            "中途回收后候选表应为空,得 {} 块",
            candidates.len()
        );
        let mut buf = vec![0u8; 4];
        face.dtoh(&root, &mut buf).await.expect("根 dtoh");
        let v = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert!((v - 129.0).abs() < 1e-5, "链值应 129.0,得 {v}");
    }


    /// **引用计数推送**(2026-10-16):求值器在改动点把 `(users, pending)` 推给
    /// 探针 —— 这是"它凭什么是活的"数据源。
    ///
    /// 判据用**推送计数器**(`slab_hint::refs_pushed`),不用报告文本:
    /// 报告渲染受日志级别影响,而"推了几次"是机械事实。CPU 面跑不了真块叶
    /// (无块账房 ⇒ DeadBlock),所以这里只验"推了且计数随引用增长";
    /// 真实图路径的端到端断言在 engine 侧(probe 窗口内 `refs_pushed > 0`)。
    /// **多根"根套根"的活性回收**(2026-10-19;设计卷 §8.31 立案):
    /// Q 槽(`eval_root_and_q` 的 `[hidden, q₁..qₙ]`)与 DFlash2/MTP taps 正是这种
    /// 形态 —— **根节点在主根子树里**。旧口径逐根 `count_pending` 相加 ⇒ 共享子树的
    /// 边被数 k 次、执行只减一次 ⇒ `pending` 永不归零 ⇒ 中途一块不还
    /// (峰值 = 全图同摊:实测 27B 单 chunk 2064 MiB)。
    ///
    /// 判据(**紧**):活块按**块尺寸**定标 —— 每节点 16 KiB,回收正常时峰值活跃集
    /// 只有个位数块(≤ `8 × NODE`);旧口径会把内层根那半条链(≈48 块)全吊住。
    /// 过激回收会立刻在值上爆(两根的值都验)。
    #[tokio::test]
    async fn multi_root_nested_liveness_reclaimed() {
        use owl_shared::slab_hint;
        const N: usize = 4096; // 每节点 16 KiB
        const NODE: usize = N * 4;
        const GATE: usize = 8 * NODE; // 峰值活跃集上界(个位数块)
        let mut face = owl_cpu::CpuFace::new();
        let c = TensorOps::from_host(Dtype::F32, vec![N], &f32b(&vec![1.0f32; N]));
        let mut x = TensorOps::from_host(Dtype::F32, vec![N], &f32b(&vec![0.0f32; N]));
        let mut inner = None;
        for i in 0..64 {
            x = x.add(&c);
            if i == 48 {
                inner = Some(x.clone()); // 内层根:深处(在主根子树里)
            }
        }
        let inner = inner.expect("内层根");

        // ① 基线:单根口径(既有中途回收)
        slab_hint::probe_on(true);
        let (r1, cand1) = eval_ops_scoped(&x, &mut face).await.expect("单根 scoped");
        let peak_single = slab_hint::probe_end();
        assert!(cand1.is_empty(), "单根档候选表应为空,得 {}", cand1.len());
        assert!(
            peak_single <= GATE,
            "单根基线峰值 {peak_single}B 就该是个位数块(门 {GATE}B)⇒ 基线本身坏了"
        );

        // ② 被验档:多根嵌套(两根:[主根, 内层根])
        slab_hint::probe_on(true);
        let roots = [&x, &inner];
        let outs =
            eval_ops_multi_scoped_env(&roots, &mut face, crate::env::EnvProvider::default(), None)
                .await
            .expect("多根 scoped");
        let peak_multi = slab_hint::probe_end();
        assert_eq!(outs.len(), 2, "两根本都必须拿到值");

        // 值校验(过激回收 = 读到已还块 ⇒ 立刻错值;`dtoh` 读**整块**)
        let mut buf = vec![0u8; NODE];
        face.dtoh(&r1, &mut buf).await.expect("单根 dtoh");
        let v1 = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert!((v1 - 64.0).abs() < 1e-4, "单根链值应 64.0,得 {v1}");
        for (i, b) in outs.iter().enumerate() {
            let mut bb = vec![0u8; NODE];
            face.dtoh(b, &mut bb).await.expect("多根 dtoh");
            let v = f32::from_le_bytes([bb[0], bb[1], bb[2], bb[3]]);
            let want = if i == 0 { 64.0 } else { 49.0 };
            assert!((v - want).abs() < 1e-4, "多根根#{i} 值应 {want},得 {v}");
        }

        // ③ 门:多根嵌套**不许**把峰值抬到图宽同阶(旧口径 = 内层根那半条链被吊住)
        assert!(
            peak_multi <= GATE,
            "多根嵌套活性峰值 {peak_multi}B ≈ {} 块(门 {GATE}B)⇒ 共享子树的 pending \
             被重复计数(§8.31);单根基线 {peak_single}B",
            peak_multi / NODE
        );
    }

    #[tokio::test]
    async fn eval_pushes_ref_counts_to_probe() {
        let mut face = owl_cpu::CpuFace::new();
        owl_shared::slab_hint::probe_on(true);
        // 菱形:同一块被三处引用(两条边 + 一次透传别名)
        let x = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[3.0]));
        let sq = x.mul(&x);
        let r = x.reshape(vec![1]);
        let root = sq.mul(&r);
        // CPU 面执行可能结构化拒(块账房/语义面限制);本测只关心**推送**
        let _ = eval_ops_scoped(&root, &mut face).await;
        let pushed = owl_shared::slab_hint::refs_pushed();
        assert!(
            pushed > 0,
            "求值必须推送引用计数(0 次 = 推送面断了)"
        );
        owl_shared::slab_hint::probe_on(false);
        // 关探针后不再推(探针关 = 零动作)
        let before = owl_shared::slab_hint::refs_pushed();
        let _ = eval_ops_scoped(&root, &mut face).await;
        assert_eq!(owl_shared::slab_hint::refs_pushed(), before, "探针关必须零推送");
    }

    /// 菱形 + 透传别名:x 被四方引用(mul 两参 + reshape 透传 + mul 右参)
    /// → 块归属按块计数;候选空且无双重 free(若误焚,host 值读回即花);
    /// 值 = x⁴ = 81(x=3:mul(x,x)=9;reshape 透传;9×3=27… 改:root =
    /// sq.mul(&r) = 9×3 = 27;r 的块 = x 的块(透传),值 3 → 27)。
    #[tokio::test]
    async fn scoped_liveness_diamond_and_alias() {
        let mut face = owl_cpu::CpuFace::new();
        let x = TensorOps::from_host(Dtype::F32, vec![1], &f32b(&[3.0]));
        let sq = x.mul(&x); // x 两次出现
        let r = x.reshape(vec![1]); // 透传别名(与 x 同块)
        let root = sq.mul(&r); // x 第三次出现
        let (root, candidates) = eval_ops_scoped(&root, &mut face).await.expect("scoped eval");
        assert!(candidates.is_empty(), "菱形/别名候选表应为空");
        let mut buf = vec![0u8; 4];
        face.dtoh(&root, &mut buf).await.expect("根 dtoh");
        let v = f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        assert!((v - 27.0).abs() < 1e-5, "应 27.0,得 {v}");
    }

    /// **块池命中 + `Zeros` 补零的确定性见证**(P1″)。
    ///
    /// 构造一个确定性求值序:先释放一个**非零**块,紧接着一个 `Zeros`
    /// 节点取块 ——
    ///   `a = (x·c) + one` ⇒ `x·c` 的输出块在 a 节点结束处被释放(16B,非零);
    ///   `v = a + zeros(4)` ⇒ zeros 节点在 a **之后**求值,best-fit/LIFO 恰好
    ///   取到那个非零块。
    /// 若命中后漏掉 `memset_zero`,zeros 块会带着 18.0 的真值 ⇒ v = 37 而非 19。
    /// (CpuFace 的 `alloc` 恒零化,故「池关」口径 = 金标;两口径逐位对比。)
    #[tokio::test]
    async fn pool_hit_zeroed_node_is_really_zeroed() {
        for budget in [0usize, 1 << 20] {
            let mut face = owl_cpu::CpuFace::new();
            let c = TensorOps::from_host(Dtype::F32, vec![4], &f32b(&[2.0; 4]));
            let one = TensorOps::from_host(Dtype::F32, vec![4], &f32b(&[1.0; 4]));
            let x = TensorOps::from_host(Dtype::F32, vec![4], &f32b(&[9.0; 4]));
            let a = x.mul(&c).add(&one);
            let v = a.add(&TensorOps::zeros(Dtype::F32, vec![4]));
            let (root, cands) = eval_ops_scoped_env_pool(
                &v, &mut face, crate::env::EnvProvider::default(), None, budget,
            )
            .await
            .expect("scoped eval");
            let mut buf = vec![0u8; 16];
            face.dtoh(&root, &mut buf).await.expect("根 dtoh");
            let got = decode_host(Dtype::F32, &buf);
            for g in &got {
                assert!(
                    (g - 19.0).abs() < 1e-5,
                    "budget={budget}: 期望 19(=9×2+1),得 {got:?}(38 则 = 漏了命中后的 memset)"
                );
            }
            // 池收尾已抽干 ⇒ 候选表不应包含池内块(两口径都不重复交还)
            assert!(cands.is_empty(), "budget={budget}: 候选表应空,得 {}", cands.len());
        }
    }

    /// **逐层同形循环:池开 / 池关两口径逐位一致**(复用不得改值)。
    /// 形状 = prefill 的真实分配形状(每层同形 → 池跨层命中)。
    #[tokio::test]
    async fn pool_reuse_matches_pool_off_bit_exact() {
        let build = || {
            let c = TensorOps::from_host(Dtype::F32, vec![8], &f32b(&(0..8).map(|i| i as f32 * 0.5).collect::<Vec<f32>>()));
            let b = TensorOps::from_host(Dtype::F32, vec![8], &f32b(&(0..8).map(|i| 1.0 + i as f32).collect::<Vec<f32>>()));
            let mut x = TensorOps::from_host(Dtype::F32, vec![8], &f32b(&[1.5; 8]));
            for _ in 0..6 {
                x = x.mul(&c).add(&b);
            }
            x
        };
        let mut f_off = owl_cpu::CpuFace::new();
        let (r_off, _) = eval_ops_scoped_env_pool(
            &build(), &mut f_off, crate::env::EnvProvider::default(), None, 0,
        )
        .await
        .expect("池关");
        let mut off = vec![0u8; 32];
        f_off.dtoh(&r_off, &mut off).await.expect("dtoh off");
        let mut f_on = owl_cpu::CpuFace::new();
        let (r_on, cands_on) = eval_ops_scoped_env_pool(
            &build(), &mut f_on, crate::env::EnvProvider::default(), None, 1 << 20,
        )
        .await
        .expect("池开");
        let mut on = vec![0u8; 32];
        f_on.dtoh(&r_on, &mut on).await.expect("dtoh on");
        assert_eq!(on, off, "池开/池关两口径必须逐位一致");
        assert!(cands_on.is_empty(), "候选表应空(池块已入池、不减 arena)");
    }
}

#[cfg(test)]
mod dtype_guard_tests {
    use super::*;
    use crate::contract::{DeviceClient, Dtype};
    use crate::ops::SemanticKernel;
    use crate::tensor::TensorOps;

    /// CPU 面:f16 语义算子结构化报错(CPU 先不搞,裁决)——不许静默
    #[tokio::test]
    async fn cpu_f16_semantic_unsupported() {
        let mut face = owl_cpu::CpuFace::new();
        // Block 叶子(CPU htod 对 f16 本就拒;守门测的是声明 dtype 对账)
        let a = TensorOps::of_block(901, Dtype::F16, vec![2]);
        let b = TensorOps::of_block(902, Dtype::F16, vec![2]);
        let y = a.add(&b);
        let err = eval_ops(y.step(), &mut face).await.unwrap_err();
        let msg = format!("{err}");
        // CPU 拒绝点随面能力漂移(alloc 拒 / 未注册动作 / DeadBlock /
        // dtype 守门),唯一不变量 = **显式报错,绝不静默错值** —— 这才是
        // 本测试守护的性质
        assert!(
            !(msg.is_empty()),
            "f16 语义算子在 CPU 面必须显式报错,得静默通过: {msg}"
        );
    }

    /// Call 通道的 dtype 错位防护(2026-10-12 瘦身后语义升级):名字由
    /// driver 按声明 dt 单源(f16 → f16 核)—— 错位类错误在源头结构性
    /// 消失;CPU 面不能执行 → 必须结构化拒绝(绝不静默错值)
    #[tokio::test]
    async fn f16_registered_kernel_is_rejected() {
        let mut face = owl_cpu::CpuFace::new();
        let src = TensorOps::of_block(903, Dtype::F16, vec![4]);
        let decl = TensorOps::call(SemanticKernel::Narrow)
        .arg(&src)
        .arg_usize(1)
        .arg_usize(4)
        .arg_usize(0)
        .arg_usize(2)
        .with_shape(Dtype::F16, vec![2]);
        assert!(
            eval_ops(decl.step(), &mut face).await.is_err(),
            "CPU 面执行 f16 narrow 必须显式报错,不得静默"
        );
    }

    /// GPU:f16 语义五算子 vs host f32 参考(F2 收口锚;OWL_TEST_DEVICE 门控)
    #[tokio::test]
    async fn gpu_f16_semantic_matches_host() {
        if !crate::testkit::gpu_enabled() {
            eprintln!("skip: OWL_TEST_DEVICE 未设");
            return;
        }
        use crate::testkit::gpu_client;
        let n = 1024usize;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.21) - 5.0).sin() * 2.0).collect();
        let yv: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.13) - 1.0).cos() * 1.5).collect();
        let alpha: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - 0.5).collect();
        let xb: Vec<u8> = x.iter().flat_map(|f| half_le(*f)).collect();

        let mut gpu = gpu_client().await;
        let shape = crate::contract::Shape::from(vec![n]);
        let dx_b = owl_iface::htod!(gpu, Dtype::F16, &shape, &xb).await.expect("htod x");
        let dy_b = owl_iface::htod!(gpu, Dtype::F16, &shape, &yv.iter().flat_map(|f| half_le(*f)).collect::<Vec<u8>>()).await.expect("htod y");
        let da_b = owl_iface::htod!(gpu, Dtype::F16, &shape, &alpha.iter().flat_map(|f| half_le(*f)).collect::<Vec<u8>>()).await.expect("htod alpha");
        // 契约:rmsnorm x/out 皆 r2(行视图)
        let dx = TensorOps::of_block(dx_b.id, Dtype::F16, vec![1, n]);
        let dy = TensorOps::of_block(dy_b.id, Dtype::F16, vec![1, n]);
        let da = TensorOps::of_block(da_b.id, Dtype::F16, vec![n]);

        // y = silu(x + y) * sigmoid(x);z = rmsnorm(y ×w_off, alpha)
        let s1 = dx.add(&dy);
        let s2 = s1.silu();
        let g = dx.sigmoid();
        let m = s2.mul(&g);
        let z = m.rmsnorm(&da, 1e-6, true);
        let bytes = eval_ops(z.step(), &mut gpu).await.expect("eval f16 chain");
        let mut buf = vec![0u8; n * 2];
        gpu.dtoh(&bytes, &mut buf).await.expect("dtoh");
        let got: Vec<f32> = buf
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect();

        // host f32 参考
        let mut max_diff = 0f32;
        for i in 0..n {
            let s = x[i] + yv[i];
            let silu = s / (1.0 + (-s).exp());
            let sig = 1.0 / (1.0 + (-x[i]).exp());
            let m_val = silu * sig;
            // rmsnorm 行内需全行 rms —— 算全行
            let _ = max_diff;
            let _ = m_val;
        }
        let mut sumsq = 0f32;
        for i in 0..n {
            let s = x[i] + yv[i];
            let silu = s / (1.0 + (-s).exp());
            let sig = 1.0 / (1.0 + (-x[i]).exp());
            let mv = silu * sig;
            sumsq += mv * mv;
        }
        let inv = 1.0 / (sumsq / n as f32 + 1e-6).sqrt();
        for (i, g_val) in got.iter().enumerate() {
            let s = x[i] + yv[i];
            let silu = s / (1.0 + (-s).exp());
            let sig = 1.0 / (1.0 + (-x[i]).exp());
            let want = silu * sig * inv * (alpha[i] + 1.0);
            let diff = (g_val - want).abs();
            max_diff = max_diff.max(diff);
            assert!(diff < 5e-2 * (1.0 + want.abs()), "[{i}] f16 {g_val} vs host {want}");
        }
        eprintln!("[f16 语义链] max_diff = {max_diff:.5}");
        gpu.close().await.expect("关机");
    }

    fn half_le(f: f32) -> [u8; 2] {
        half::f16::from_f32(f).to_le_bytes()
    }

    /// f32 正常路径不受守门影响(回归哨)
    #[tokio::test]
    async fn f32_path_unaffected() {
        let mut face = owl_cpu::CpuFace::new();
        let a = TensorOps::from_host(Dtype::F32, vec![2], &crate::testkit::f32b(&[1.0, 2.0]));
        let b = TensorOps::from_host(Dtype::F32, vec![2], &crate::testkit::f32b(&[3.0, 4.0]));
        let y = a.add(&b);
        let bytes = eval_ops(y.step(), &mut face).await.expect("f32 add");
        let mut buf = vec![0u8; 8];
        face.dtoh(&bytes, &mut buf).await.expect("dtoh");
        let got: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, vec![4.0, 6.0]);
    }
}
