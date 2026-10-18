//! **prefill 桶图(M2b)** —— 设计 `docs/arch/prefill-graph-设计与实现.md` §4–§6。
//!
//! # 形态
//! **一个桶(t_pad)一张图**,图内**含注意力**(FI)与 KV 写:
//! ```text
//! 捕获(boot 静止点;每桶一次):
//!   embed → [64 层:norm/qkv/o/MLP/GDN + KV 写(t_pad 格) + FI 注意力] → 全行 hidden + taps
//! 回放(每 chunk):
//!   写输入槽(全 t_pad 长度;真行真值 + 假行落 dump 页)→ 1 条 graph_launch
//!   → 出图尾(eager:末行窄切 + lm_head + argmax;taps/extend 按真行)
//! ```
//!
//! # 为什么成立(前置件全部已实测;逐条见设计 §4)
//! - **FI plan 与 ctx 无关**(F-5;split 关)⇒ 一个桶一份 plan 跨 ctx 复用;
//!   真 ctx 由**设备表**(indices/indptr/last_len)+ **设备 q_cu** 在回放期读
//!   ⇒ `qo_len`/`kv_len` 都是真值(因果窗口与 eager 逐字同式,`§4.1c` 推论 A)。
//! - **三律(§4.2)照进桶几何**:假行不进序列(`seq_cu=[0,t_real]` + GDN 序列表
//!   按 t_real 铸)、假行池写落 dump 页(不在页链 ⇒ 读不到)、采样读真行(出图尾)。
//! - **内存四原语 + M1 峰值口径**(§4.3d)让大图能捕上来;**家族面三正路**
//!   (§4.3e)保证窗内不现建 scratch。
//!
//! # 两条**硬纪律**(写进代码免得后人踩)
//! 1. **静止点律**(§4.5):`GraphPlan::plan` 的 warmup 会真跑一遍 eager(prefill
//!    有 KV/GDN 副作用)⇒ 只在 **boot** 捕获;turn 内只回放。失败桶永久
//!    `EagerFallback`(warn 一次,不重试)。
//! 2. **降序捕获**:家族 scratch(`ensure_one`/`ensure 律`)只按需**增大**,
//!    已捕图烘的是当前指针 ⇒ 先捕大桶(t_pad 降序)保证后续小桶复用同一批缓冲,
//!    绝不触发重分配(§4.3e 的 `ensure_one` 律)。
//!
//! # GDN 状态槽口径(本轮的边界,写清楚)
//! - **varlen 臂**:状态走**设备槽张量**(`ctx.gdn_slot`)⇒ 图对任意会话槽可用 ✓
//! - **chunked/scalar 臂**(t_pad ≥ 64;27B 生产档):状态地址 = **host 槽标量**
//!   (A13 面:`StateSlot{pool, slot}` 在 host 算指针)⇒ 图只能烘一个槽
//!   ⇒ 本档**只对 `gdn_slot == baked` 的会话可用**,其余回 eager(正确性优先;
//!   该槽经 `BucketGraph::usable` 结构化判定,**绝不静默串台**)。
//!   彻底解 = 把槽做成设备面(P-6c:自建 transpose 核吃设备 slot)—— 另案。

use std::sync::Arc;

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_models::env::EnvProvider;
use owl_models::layers::gdn::GdnBuffers;
use owl_models::layers::rope::Rope;
use owl_models::model::Model;
use owl_models::module::{FiPrefillCtx, ForwardCtx, GdnTableSlot, KvBuffers, TpCtx};
use owl_models::TensorOps;

use crate::exec::prefill_chunk::ChunkTables;
use crate::graph_plan::{GraphPlan, GraphPlanDesc, InputSlot, OutputSlot, PlanCtx, PlanOutcome};
use crate::state::{ModelDims, StatePool};

type Result<T> = std::result::Result<T, ModelError>;

// ============================================================================
// §1 档位 → 桶阶梯(纯函数面;可单测)
// ============================================================================

/// 档位 → 桶阶梯(**t_pad 升序**;空 = 不建桶 = 今天行为)。
///
/// ⚠️ 显存纪律(§4.3b):全 5 桶的 slab 之和 ≈ 374 MiB ≈ 0.69 × KV 池
/// (fp8/27B);`t512` 一桶 ≈ 193 MiB。S-1/S-2/S-3 的取舍见设计 §4.3b。
pub(crate) fn ladder(b: owl_shared::config::PrefillBucket) -> &'static [usize] {
    use owl_shared::config::PrefillBucket as B;
    match b {
        B::Off => &[],
        B::T512 => &[512],
        B::Geobuckets => &[32, 64, 128, 256, 512],
    }
}

/// 桶命中(纯函数):**向上取整**到最小可用桶;None = 走 eager 原路。
pub(crate) fn pick_bucket(ladder: &[usize], t: usize) -> Option<usize> {
    ladder.iter().copied().filter(|&b| b >= t).min()
}

/// **GDN host 槽臂判定**(纯函数;见模块头注 §GDN 状态槽口径)。
/// true ⇒ 图把某个槽烘死 ⇒ 只对该槽的会话可用(`BucketGraph::usable` 守门)。
pub(crate) fn gdn_host_slot_arm(env: &EnvProvider, dims: &ModelDims, t_pad: usize) -> bool {
    if t_pad < 64 {
        return false; // 小 T 恒走 varlen(层侧同门:`tokens >= 64`)
    }
    // 形状判据**单源在核的家**(`kernels::family::gdn_chunked`:验证清单 + 谓词;
    // engine 零数字 —— 2026-10-17 用户裁决"别写成硬编码机制")。
    // `nk/nv` 是**每 rank 的头数**(TP 分片后减半)⇒ 清单里含分片档。
    let chunked_shape_ok = dims.hk == 128
        && dims.hv == 128
        && owl_models::kernel::host_slot_arm_shape_ok(dims.nk, dims.nv);
    let scalar_shape_ok = dims.hk == 128 && dims.hv == 128;
    (env.gdn.scalar && !env.gdn.chunked && scalar_shape_ok)
        || (env.gdn.chunked && chunked_shape_ok)
}

// ============================================================================
// §2 回放产物(真行口径;与 eager 的根同形同布局)
// ============================================================================

/// 桶回放产物。**真行视图**:hidden 取前 `t_real` 行;tap 取布局第 `(i+1)`
/// 个 `t_pad` 区的前 `t_real` 行(块内偏移 = 布局 stride —— 这是"布局 t /
/// 语义 t 分离"的落地点,§4.2)。
pub(crate) struct BucketOut {
    /// 图输出根块(**布局**行数 = `(taps+1)·t_pad`;调用方按 `t_real` 切)
    pub(crate) root: Bytes,
    /// taps 真行视图([t_real, dd];空 = 无草稿面)
    pub(crate) taps: Vec<TensorOps>,
}

// ============================================================================
// §3 单桶图
// ============================================================================

pub(crate) struct BucketGraph<D: DeviceClient> {
    /// 布局行数(= 图定形;真行 ≤ 它)
    pub(crate) t_pad: usize,
    /// 回放态门(**只有真捕获成功才允许 `run`**;否则 `step()` 会再跑 eager
    /// ⇒ 与外层 eager 双执行 ⇒ GDN 状态双推进 —— 设计 §4.6 的"双执行"雷)
    pub(crate) captured: bool,
    /// 捕获结果说明(warn 一次的证据面)
    pub(crate) outcome: String,
    pub(crate) dd: usize,
    pub(crate) taps: usize,
    pub(crate) dtype: Dtype,
    /// GDN 状态槽口径:Some(s) = 本图把槽 s 烘死(host 槽臂)⇒ 只对槽 s 可用
    pub(crate) gdn_slot_baked: Option<usize>,
    /// FI 页链槽的声明容量(= 池管理面页数;`fi_indices` 槽长度)
    fi_nb: usize,
    /// 回放前是否要宿主刷 FI plan 表(= `!fi_plan_restore`;图不自携表时必需)
    pub(crate) needs_plan_sync: bool,
    /// 回放次数(证据面:桶被真用了吗;测试/运维读取)
    replays: usize,
    pub(crate) plan: GraphPlan<D>,
}

impl<D: DeviceClient> BucketGraph<D> {
    /// **本桶此刻可用吗**(纯判定;可单测)。`false` ⇒ 调用方走 eager 原路。
    ///
    /// 判据:①已捕获;②t 在桶内;③**槽号 = 烘焙槽**。
    ///
    /// ③ 是结构性约束:host 槽臂把**格地址**烘进图 ⇒ 槽号不同 = 图读写别人的格。
    ///
    /// ⚠️ 曾另有第 4 条「格世代相等」(2026-10-17 临时防线):当时"同格复用即静默
    /// 错值"的**根因未点名**(记为"另一半跨流序")。根因同日定谳:**FI plan 的共享
    /// `int_ws` 调度表被图回放改写,而 host 侧 `active` 旧账不知情** ⇒ 复用发的
    /// decode 读到桶的调度表(修于 `FiRuntime::activate`:窗外一律重刷;
    /// 结案见 `roadmap.local/26-10-17-prefill桶图-二次会话污染-定位实录.md`)。
    /// 真因修好后世代与桶无关 ⇒ 第 4 条已撤(格换主后回放实测正常)。
    pub(crate) fn usable(&self, t_real: usize, gdn_slot: usize) -> bool {
        bucket_usable(
            self.captured,
            t_real,
            self.t_pad,
            self.gdn_slot_baked,
            gdn_slot,
        )
    }

    /// **回放**:写输入槽(全 `t_pad` 长度)→ 1 条 `graph_launch` → 真行视图。
    ///
    /// 槽内容由 [`ChunkTables::build_bucket`] 给出(真行真值 + 假行 dump 槽;
    /// FI 四件/GDN 三表按 `t_real` 铸)。`fi_indices` 槽按声明容量对齐(核只读
    /// `indptr` 划定的前 `nb_used` 项;尾部补 0 —— 结构上到不了)。
    pub(crate) async fn replay(
        &mut self,
        tb: &ChunkTables,
        t_real: usize,
    ) -> Result<Option<BucketOut>> {
        if !self.captured {
            return Ok(None); // EagerFallback ⇒ 必须走外层 eager(否则双执行)
        }
        let Some(ft) = tb.fi.as_ref() else {
            return Ok(None); // 非 FI 档(FI 是桶图的前置面)
        };
        let bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|x| f32::from_bits(*x)).collect() };
        // 页链槽:**声明容量对齐**(零点 = 页 0;indptr 划定的前缀之外结构上不读)
        let mut idx = ft.indices.clone();
        if idx.len() > self.fi_nb {
            return Err(ModelError::Msg(format!(
                "prefill 桶 t={}:页链 {} 超槽容量 {}(池几何或 ctx 配置不一致)",
                self.t_pad,
                idx.len(),
                self.fi_nb
            )));
        }
        idx.resize(self.fi_nb, 0);
        let fills: [(&str, Vec<f32>); 10] = [
            ("tl", tb.ids.clone()),
            ("pos", tb.pos.clone()),
            ("kv_slots", tb.slots.clone()),
            ("kv_lens", tb.lens.clone()),
            ("gdn_slot", vec![tb.gdn_slot]),
            ("seq_cu", tb.seq_cu.clone()),
            ("fi_q_cu", bits(&ft.q_cu)),
            ("fi_indices", bits(&idx)),
            ("fi_indptr", bits(&ft.indptr)),
            ("fi_last_len", bits(&ft.last_len)),
        ];
        let refs: Vec<(&str, &[f32])> = fills.iter().map(|(n, v)| (*n, v.as_slice())).collect();
        self.plan.step(&refs).await?;
        self.replays += 1;
        let Some(root) = self.plan.output_block("root") else {
            return Ok(None);
        };
        let dd = self.dd;
        let t_pad = self.t_pad;
        let taps = (0..self.taps)
            .map(|i| {
                TensorOps::of_block(root.id, self.dtype, vec![(self.taps + 1) * t_pad, dd])
                    .slice_view((i + 1) * t_pad * dd, vec![t_real, dd])
            })
            .collect();
        Ok(Some(BucketOut { root, taps }))
    }

    /// 相位扇出(perf tag;GraphFleet::set_phase 的统一入口)
    pub(crate) fn set_phase(&mut self, p: owl_models::env::EvalPhase) {
        self.plan.set_phase(p);
    }
}

// ============================================================================
// §4 桶族(捕获面)
// ============================================================================

/// **桶可用性纯函数**(单一事实源 = [`BucketGraph::usable`];提出来是为了单测)。
///
/// 判据三条(见 `usable` 注):已捕获 / t 在桶内 / 槽号 = 烘焙槽。
pub(crate) fn bucket_usable(
    captured: bool,
    t_real: usize,
    t_pad: usize,
    baked_slot: Option<usize>,
    gdn_slot: usize,
) -> bool {
    captured
        && t_real >= 1
        && t_real <= t_pad
        && baked_slot.is_none_or(|s| s == gdn_slot)
}

/// 捕获期 boot 事实(显式依赖;零隐式读取)
pub(crate) struct BucketBoot<'a> {
    pub(crate) env: EnvProvider,
    /// **张量并行上下文(S4 ⑤;`None` = 单卡)** —— 与 env 同律：
    /// 每桶图装配期烘进求值路(eager/捕获两路都带)。
    pub(crate) tp: Option<TpCtx>,
    pub(crate) model: Arc<Model>,
    pub(crate) rope: Rope,
    pub(crate) pool: &'a StatePool,
    pub(crate) dims: ModelDims,
    /// DFlash2 tap 数(0 = 无草稿面)
    pub(crate) taps: usize,
    /// 桶阶梯(t_pad 升序;档位 → `ladder()`)
    pub(crate) ladder: &'static [usize],
    /// `[dispatch] fi_plan_restore`(2026-10-19):false ⇒ 桶图**不自携** FI plan 表,
    /// 回放前由引擎窗外 `fi_plan_sync` 刷(与 decode/verify 同律)。
    pub(crate) fi_plan_restore: bool,
}

/// 桶族(RunningEngine 持有;空 = 档位 off/未启用)
pub(crate) struct PrefillBuckets<D: DeviceClient> {
    /// t_pad 升序(与阶梯同序)
    buckets: Vec<BucketGraph<D>>,
}

impl<D: DeviceClient> PrefillBuckets<D> {
    pub(crate) fn empty() -> Self {
        Self { buckets: Vec::new() }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// 桶族 slab 占用(Σ 已捕获桶;图显存总账账源)
    pub(crate) fn slab_bytes(&self) -> usize {
        self.buckets
            .iter()
            .filter(|b| b.captured)
            .map(|b| b.plan.slab_bytes())
            .sum()
    }

    /// **桶族退休**(与 `GraphFleet::close` 同律:逐图卸图 + 归槽,
    /// 失败仅告警 + 记首错不短路)
    pub(crate) async fn close(&mut self) -> crate::Result<()> {
        let mut first: Option<owl_iface::contract::ModelError> = None;
        for b in self.buckets.iter_mut() {
            crate::graph_fleet::record_close(&mut first, b.plan.close().await);
        }
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// 桶族一行摘要(boot 日志/运维;空 = "")
    pub(crate) fn summary(&self) -> String {
        self.buckets
            .iter()
            .map(|b| {
                format!(
                    "t{}{}{}",
                    b.t_pad,
                    if b.captured { "✓" } else { "✗" },
                    b.gdn_slot_baked.map(|s| format!("(slot{s})")).unwrap_or_default()
                )
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// 命中查询:t **向上取整**到最小可用桶;该桶不可用(未捕获/槽不匹配)
    /// ⇒ None(走 eager 原路)。纯 host 判定(零分配、零设备往返)。
    /// (回放面用可变版;只读版不设 —— 命中判定单源在 [`BucketGraph::usable`])
    pub(crate) fn get_mut(&mut self, t: usize, gdn_slot: usize) -> Option<&mut BucketGraph<D>> {
        let t_pad = self.t_pad_for(t)?;
        self.buckets
            .iter_mut()
            .find(|b| b.t_pad == t_pad)
            .filter(|b| b.usable(t, gdn_slot))
    }

    /// 向上取整命中(阶梯来自本族桶表;纯函数同 [`pick_bucket`])
    fn t_pad_for(&self, t: usize) -> Option<usize> {
        let ladder: Vec<usize> = self.buckets.iter().map(|b| b.t_pad).collect();
        pick_bucket(&ladder, t)
    }

    /// 相位扇出(唯一入口;图族纪律同 GraphFleet)
    pub(crate) fn set_phase(&mut self, p: owl_models::env::EvalPhase) {
        for b in self.buckets.iter_mut() {
            b.set_phase(p);
        }
    }

    /// 已捕获桶的 t_pad(测试取证面)
    #[cfg(test)]
    pub(crate) fn captured_t_pads(&self) -> Vec<usize> {
        self.buckets.iter().filter(|b| b.captured).map(|b| b.t_pad).collect()
    }

    /// 逐桶回放次数(测试取证面:桶真被用了吗;**非零 = 图路确实跑了**)
    #[cfg(test)]
    pub(crate) fn replay_counts(&self) -> Vec<(usize, usize)> {
        self.buckets.iter().map(|b| (b.t_pad, b.replays)).collect()
    }
}

impl<D: DeviceClient> PrefillBuckets<D> {
    /// **boot 静止点捕获**(降序;失败桶永久 eager,不重试)。
    ///
    /// `face` = 主 face 句柄(**借用**;每桶 `face_clone()` 一个 —— 句柄共享同一
    /// actor / 同一家族运行时,这正是 FI plan 必须**自携**的原因,见
    /// `crates/kernels/src/server/flashinfer.rs::PlanEntry`)。
    pub(crate) async fn capture(face: &D, boot: &BucketBoot<'_>) -> Self {
        let mut buckets = Vec::new();
        // 降序:家族 scratch 只增不减 ⇒ 先捕大桶,后续小桶复用同一批缓冲,
        // 已捕图烘的指针绝不会被重分配(模块头注硬纪律 #2)。
        for &t_pad in boot.ladder.iter().rev() {
            let Some(f) = face.face_clone() else {
                owl_shared::log_warn!(
                    "prefill-graph",
                    "face_clone 不可用 ⇒ 桶族止步于 t_pad={t_pad}(未建桶全走 eager)"
                );
                break;
            };
            match Self::capture_one(f, boot, t_pad).await {
                Ok(b) => {
                    owl_shared::log_info!(
                        "prefill-graph",
                        "桶 t_pad={t_pad} taps={} 捕获结果={}",
                        boot.taps,
                        b.outcome
                    );
                    buckets.push(b);
                }
                Err(e) => {
                    // 失败 ⇒ 该桶永久 eager(不重试);warn 一次是**证据面**
                    // (静默降级 = 隐瞑,本仓禁止)
                    owl_shared::log_warn!(
                        "prefill-graph",
                        "桶 t_pad={t_pad} 捕获失败 ⇒ 该桶永久 eager 回退:{e}"
                    );
                }
            }
        }
        buckets.sort_by_key(|b| b.t_pad);
        if !buckets.is_empty() {
            let s: Vec<String> = buckets
                .iter()
                .map(|b| {
                    format!(
                        "t{}:{}{}",
                        b.t_pad,
                        b.outcome,
                        if b.captured { "" } else { "(不可回放)" }
                    )
                })
                .collect();
            owl_shared::log_info!(
                "prefill-graph",
                "桶族就位({} 桶):{}",
                buckets.len(),
                s.join(" ")
            );
        }
        Self { buckets }
    }

    /// 单桶:建槽面 + 声明树 + `GraphPlan::plan`(warmup + 捕获)
    async fn capture_one(face: D, boot: &BucketBoot<'_>, t_pad: usize) -> Result<BucketGraph<D>> {
        let pool = boot.pool;
        let dims = boot.dims;
        let dd = dims.hidden;
        let dtype = dims.dtype;
        let taps = boot.taps;
        // ── 槽面(与 eager 的 `ChunkTables` 同名同义;长度 = 布局 t_pad)──
        let fi_blocks = pool.fi_table_blocks().ok_or_else(|| {
            ModelError::Msg("prefill 桶:无 FI 表件槽(非 FI 档不建桶)".into())
        })?;
        let nb = fi_blocks.nb_cap;
        let inputs = vec![
            InputSlot::f32("tl", t_pad).rank1(),
            InputSlot::f32("pos", t_pad)
                .rank1()
                .init((0..t_pad).map(|i| i as f32).collect()),
            InputSlot::f32("kv_slots", t_pad).rank1(),
            InputSlot::f32("kv_lens", t_pad)
                .rank1()
                .init((0..t_pad).map(|i| (i + 1) as f32).collect()),
            InputSlot::f32("gdn_slot", 1).rank1(),
            InputSlot::f32("seq_cu", 2).rank1().init(vec![0.0, t_pad as f32]),
            // ⚠️ 槽初值必须**自洽**:warmup = 真跑一遍 eager ⇒ FI 的
            // `kv_len ≥ qo_len` 条(causal)会校验设备表。这里给"整段上下文 =
            // 本 chunk"的合法表(单页 + 末页长 = t_pad)。回放期引擎会写**真**
            // 口径(§4.1c 推论 A:`qo_len`/`kv_len` 都从设备表读 ⇒ 与 eager 逐字同式)。
            // U32 位型槽按 `f32::from_bits` 承写(见 `InputSlot::u32`)。
            InputSlot::u32("fi_q_cu", 2).init(vec![
                f32::from_bits(0),
                f32::from_bits(t_pad as u32),
            ]),
            InputSlot::u32("fi_indices", nb), // 页链槽(容量 = 池页数)
            InputSlot::u32("fi_indptr", 2).init(vec![f32::from_bits(0), f32::from_bits(1)]),
            InputSlot::u32("fi_last_len", 1).init(vec![f32::from_bits(t_pad as u32)]),
        ];
        let rows = if taps > 0 { (taps + 1) * t_pad } else { t_pad };
        let outputs = vec![OutputSlot {
            name: "root".into(),
            shape: vec![rows, dd],
            dtype,
        }];

        // ── 闭包捕获的设备句柄(池件;地址恒定 = 图可烘)──
        let kv_caches = pool.kv_caches();
        let fi_leaves = pool.kv_fi_leaves();
        let gdn_leaves = pool.gdn_caches();
        let bt = pool.bt_leaf_flat();
        // GDN 序列表:**布局口径 = ceil(t_pad/BT)**(图内 `nt` 是标量,烘死;
        // 表内容由引擎按 t_real 写 —— cu 决定行区间与递推落点,§4.3e P-6b)
        let gdn_nt = t_pad.div_ceil(owl_models::kernel::GDN_BT).max(1);
        let gdn_table = match pool.gdn_table_blocks() {
            Some(gt) => {
                if gdn_nt > gt.nt_cap {
                    return Err(ModelError::Msg(format!(
                        "prefill 桶 t_pad={t_pad}:GDN 序列表超槽容量 nt={gdn_nt} > {}",
                        gt.nt_cap
                    )));
                }
                Some(TensorOps::of_block(
                    gt.table.id,
                    Dtype::U32,
                    vec![owl_models::kernel::gdn_table_bytes(gdn_nt) / 4],
                ))
            }
            None => None,
        };
        // GDN 状态槽口径(host 槽臂 ⇒ 只对该槽可用;见模块头注)
        let slot_baked = gdn_host_slot_arm(&boot.env, &dims, t_pad).then_some(0usize);
        let env = boot.env;
        let model = Arc::clone(&boot.model);
        let rope = boot.rope.clone();
        let forward = move |sc: &PlanCtx| -> Result<()> {
            let tl = sc.input("tl")?;
            let pos = sc.input("pos")?;
            let slots = sc.input("kv_slots")?;
            let lens = sc.input("kv_lens")?;
            let gdn_slot = sc.input("gdn_slot")?;
            let seq_cu = sc.input("seq_cu")?;
            let (q_cu, indices, indptr, last_len) = (
                sc.input("fi_q_cu")?,
                sc.input("fi_indices")?,
                sc.input("fi_indptr")?,
                sc.input("fi_last_len")?,
            );
            let kvs: Vec<KvBuffers> = kv_caches
                .iter()
                .map(|(k, v)| KvBuffers {
                    k_cache: k.clone(),
                    v_cache: v.clone(),
                    slots: slots.clone(),
                    kv_lens: lens.clone(),
                    block_tables: bt.clone(),
                })
                .collect();
            let gdns: Vec<GdnBuffers> = gdn_leaves
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: gdn_slot.clone(),
                })
                .collect();
            let (kcs, vcs): (Vec<TensorOps>, Vec<TensorOps>) =
                fi_leaves.iter().cloned().unzip();
            let fi = Some(FiPrefillCtx {
                kcs: &kcs,
                vcs: &vcs,
                q_cu: &q_cu,
                indices: &indices,
                indptr: &indptr,
                last_len: &last_len,
            });
            let mut ctx = ForwardCtx::model_prefill(
                t_pad,
                &pos,
                &kvs,
                &rope,
                &gdns,
                &slots,
                &lens,
                &gdn_slot,
                slot_baked, // host 槽臂:Some(0);varlen 臂:None(状态走设备张量)
                0,          // ctx_base = 0(桶只服务"本 chunk 即整段"的布局;真 ctx 走设备表)
                fi,
            );
            ctx.seq_cu = Some(&seq_cu);
            if let Some(tbl) = &gdn_table {
                ctx.gdn_tables = Some(GdnTableSlot { table: tbl, nt: gdn_nt });
            }
            ctx.env = env;
            // 根:与 eager 逐字同构(单根 concat;taps 行 = 布局 t_pad)
            let (h, tps) = if taps > 0 {
                model.tapped_hidden(&tl, &ctx, &[5, 19, 33, 47, 61])
            } else {
                (model.last_hidden(&tl, &ctx), Vec::new())
            };
            if taps > 0 {
                let mut c = TensorOps::call(owl_models::ops::SemanticKernel::Concat).arg(&h);
                for tp in &tps {
                    c = c.arg(tp);
                }
                for _ in tps.len()..7 {
                    c = c.arg(&h);
                }
                let n = tps.len() + 1;
                let root = c
                    .arg_usize(n)
                    .arg_usize(t_pad)
                    .arg_usize(dd)
                    .with_shape(dtype, vec![n * t_pad, dd]);
                sc.output("root", &root)
            } else {
                sc.output("root", &h)
            }
        };
        let desc = GraphPlanDesc {
            inputs,
            outputs,
            capture: true,
            slab_fixed: false,
            debug_probe: false,
            // 捕获窗锚定活性回收(§4.3d M1):prefill 图族开 ⇒ slab 容量 =
            // 峰值活跃集(总和口径会让大图捕不上:27B/512-tok 实测 9362 MiB)
            reclaim_capture: true,
            warmup: true,
        };
        let (plan, outcome) = GraphPlan::plan(face, boot.env, boot.tp.clone(), desc, forward).await?;
        let captured = matches!(outcome, PlanOutcome::Captured);
        Ok(BucketGraph {
            t_pad,
            captured,
            outcome: format!("{outcome:?}"),
            dd,
            taps,
            dtype,
            gdn_slot_baked: slot_baked,
            fi_nb: nb,
            needs_plan_sync: !boot.fi_plan_restore,
            replays: 0,
            plan,
        })
    }
}

// (无外部辅助:face 句柄由 `face_clone` 逐桶派生)

#[cfg(test)]
mod tp_shard_arm_tests {
    //! S4 ④:GDN 图档判据的**分片档**(保守白名单:world=2 已列,未验证档不放行)
    use super::*;

    fn dims(nk: usize, nv: usize) -> ModelDims {
        ModelDims {
            hq: 12, hkv: 2, hd: 256, nk, hk: 128, nv, hv: 128,
            hidden: 5120, dtype: owl_models::tensor::Dtype::F16,
        }
    }

    #[test]
    fn gdn_host_slot_arm_accepts_full_and_tp2_shard_only() {
        let mut env = owl_models::env::EnvProvider::default();
        env.gdn.chunked = true;
        env.gdn.scalar = false;
        assert!(gdn_host_slot_arm(&env, &dims(16, 48), 128), "全量档 (16,48) 应放行");
        assert!(gdn_host_slot_arm(&env, &dims(8, 24), 128), "TP=2 分片档 (8,24) 应放行");
        assert!(!gdn_host_slot_arm(&env, &dims(4, 12), 128), "未验证档 (4,12) 不放行");
        assert!(!gdn_host_slot_arm(&env, &dims(16, 48), 32), "小 T 恒走 varlen");
    }
}
