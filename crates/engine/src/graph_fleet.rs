//! 图族域(2026-10-14 拆分自 running.rs:RunningEngine 万能结构体按
//! 状态域拆解 —— 本模块收编**全部 GraphPlan 实例**:主会话图 + spec
//! 三族图桶形阵 + 捕获三态结果)。
//!
//! 2026-10-15 第二刀:`Engine::run()` 的四套图装配(主会话图 forward
//! 闭包 / verify 桶形图 / fold+propose 族 / DFlash2 propose 单图)整体
//! 下沉到本模块 —— run() 只留编排,图拓扑知识全部封装在图族域。
//!
//! 图生命周期纪律(charter A1):实例化后禁 trim/池属性变更;相位
//! 定相(perf tag 维度)须同步全部图 —— [`GraphFleet::set_phase`] 是
//! 唯一扇出点,禁止散点逐图设置(漏一张 = perf 归因错相)。

use std::sync::Arc;

use owl_iface::contract::{Bytes, DeviceClient, Dtype, ModelError};
use owl_models::env::EnvProvider;
use owl_models::layers::gdn::GdnBuffers;
use owl_models::module::{FiPrefillCtx, ForwardCtx, KvBuffers, Module, TpCtx};
use owl_models::ops::PlanNode; // `TensorOps::op()`(槽叶 → 块 id)
use owl_models::model::Model;
use owl_models::tensor::TensorOps;

use crate::engine::EngineKnobs;
use crate::graph_plan::{GraphPlan, GraphPlanDesc, InputSlot, OutputSlot, PlanCtx, PlanOutcome};
use crate::sampler::{REP_HISTORY_SENTINEL, REP_HISTORY_WINDOW};
use crate::spec_runtime::{Drafter, SpecRuntime};
use crate::state::{ModelDims, StatePool};
use crate::Result;

type T = std::time::Instant;

/// 拆卸口径的小帮手(图示/桶族共用):记首错 + 告警,不短路。
pub(crate) fn record_close(slot: &mut Option<owl_iface::contract::ModelError>, r: crate::Result<()>) {
    if let Err(e) = r {
        owl_shared::log_warn!("graph-fleet", "图退役失败(继续拆卸):{e}");
        if slot.is_none() {
            *slot = Some(e);
        }
    }
}

/// **一个 active depth 的完整图族**（自适应 depth 的容器单元）。
///
/// 为什么 fold/propose 也要按 depth 分：propose 图的链步数/输出宽度由
/// `depth` 定形，fold 图又烘焙了**本 depth verify 图**的 `r{i}` 输出块；
/// 只补 verify 两张图而共用 propose/fold ⇒ d3 轮仍付 d7 propose 税，
/// 且 fold 会读到另一张 verify 图的陈旧记录。
pub(crate) struct DepthGraphs<D: DeviceClient> {
    pub depth: usize,
    /// verify 桶形图(T = depth+1;None = eager fallback)
    pub verify: Option<GraphPlan<D>>,
    /// fold 桶形图族(index = m,0..depth)
    pub fold: Vec<GraphPlan<D>>,
    /// propose 桶形图族(index = m,0..=depth)
    pub propose: Vec<GraphPlan<D>>,
}

/// 图族(组合叶;RunningEngine 持有)。字段 pub(crate) 供 exec 直读
/// (`graphs.session.face_mut()` 等),相位扇出走方法。
pub(crate) struct GraphFleet<D: DeviceClient> {
    /// 主会话图(prefill/decode 主路径;槽 = frontier/pos/kv_len/…)
    pub session: GraphPlan<D>,
    /// 捕获三态结果(Captured = 回放态;EagerFallback = 同闭包直发)
    pub capture_outcome: PlanOutcome,
    /// 按 depth 的 spec 图族（升序；无自适应 = 单元素）
    pub depths: Vec<DepthGraphs<D>>,
    /// **合批 decode 图**(B5.1 M1b;T = max_batch 定形,行 = 会话;None = eager)
    pub batch: Option<GraphPlan<D>>,
    /// DFlash2 propose 单图(E5-DF4;encode 全 W 行 + 噪声块 + selector,
    /// W=K+1 运行时定形;None = eager 回退)
    pub dflash: Option<GraphPlan<D>>,
    /// **切档 propose 收割图**(2026-10-26 propose 刀3;仅 rank0 建):
    /// 两半候选合并核 + selector 全在设备侧,输出 = select 单缓冲(前 K = drafts)。
    /// None = 回落 host 合并 + eager select(`dflash_select_merged`)。
    pub dflash_select: Option<GraphPlan<D>>,
}
/// **臂事实**(2026-10-19 审查):boot 自证行原先只报"**面**登记"(`FI 节点登记=true`),
/// 而**臂**由 models 侧顺序决定 —— 两者可以不一致(实测:flavor `verify_fi=true`
/// 但 `verify_v2` 缺省 true 抢前 ⇒ 真跑 v2 核)。这里读 models 记的 `attn.arm.*`
/// 计数(每图 warmup 跑过一遍 ⇒ 生效臂计数 > 0),让报告**名实一致**。
fn arm_facts(prefix: &str) -> String {
    let rows = owl_shared::metrics::query_metrics(
        &owl_shared::metrics::MetricsFilter::new().tag_prefix(prefix),
    );
    let mut v: Vec<String> = rows
        .iter()
        .filter(|r| r.counter > 0)
        .map(|r| format!("{}={}", r.tag.trim_start_matches(prefix), r.counter))
        .collect();
    v.sort();
    if v.is_empty() { "无(warmup 未记录)".to_string() } else { v.join(" ") }
}


impl<D: DeviceClient> GraphFleet<D> {
    fn depth_slot(&self, depth: usize) -> Result<usize> {
        self.depths
            .iter()
            .position(|d| d.depth == depth)
            .ok_or_else(|| ModelError::EngineInvariant {
                what: format!(
                    "spec depth {depth} 图族缺席（已捕获: {:?}）",
                    self.depths.iter().map(|d| d.depth).collect::<Vec<_>>()
                ),
            })
    }

    /// 只读 verify 面（boot 检查/装配期用；None = 该档捕获降级）。
    pub(crate) fn verify_of(&self, depth: usize) -> Option<&GraphPlan<D>> {
        self.depths
            .iter()
            .find(|d| d.depth == depth)
            .and_then(|d| d.verify.as_ref())
    }

    /// 该档是否有 propose 图族（空 = eager propose）。
    pub(crate) fn has_propose(&self, depth: usize) -> bool {
        self.depths
            .iter()
            .find(|d| d.depth == depth)
            .is_some_and(|d| !d.propose.is_empty())
    }

    /// 该档是否有 fold 图族（空 = eager fold/replay）。
    pub(crate) fn has_fold(&self, depth: usize) -> bool {
        self.depths
            .iter()
            .find(|d| d.depth == depth)
            .is_some_and(|d| !d.fold.is_empty())
    }

    /// 热路径 verify 面（必须存在；缺失 = 装配不变量失守）。
    pub(crate) fn verify_mut(&mut self, depth: usize) -> Result<&mut GraphPlan<D>> {
        let i = self.depth_slot(depth)?;
        self.depths[i]
            .verify
            .as_mut()
            .ok_or_else(|| ModelError::EngineInvariant {
                what: format!("spec depth {depth} verify 图缺席（捕获降级）"),
            })
    }

    /// fold 桶（rows = 1..=depth；桶下标 = rows-1）。
    pub(crate) fn fold_mut(&mut self, depth: usize, rows: usize) -> Result<&mut GraphPlan<D>> {
        if rows == 0 || rows > depth {
            return Err(ModelError::EngineInvariant {
                what: format!("fold rows={rows} 越界（active depth={depth}）"),
            });
        }
        let i = self.depth_slot(depth)?;
        // 单图(按最大 rows=depth 捕获)覆盖全部 rows;行数由 `cu` 输入在回放时给。
        self.depths[i]
            .fold
            .first_mut()
            .ok_or_else(|| ModelError::EngineInvariant {
                what: format!("spec depth {depth} fold 图缺席（eager 路径应由 has_fold 拦）"),
            })
    }

    /// propose 桶（m = 0..=depth）。
    pub(crate) fn propose_mut(&mut self, depth: usize, m: usize) -> Result<&mut GraphPlan<D>> {
        if m > depth {
            return Err(ModelError::EngineInvariant {
                what: format!("propose m={m} 越界（active depth={depth}）"),
            });
        }
        let i = self.depth_slot(depth)?;
        self.depths[i]
            .propose
            .get_mut(m)
            .ok_or_else(|| ModelError::EngineInvariant {
                what: format!("spec depth {depth} propose 图桶 m={m} 缺席"),
            })
    }

    /// **整族退休**(TP 双 rank 装配面的收尾;S4 ⑤-K2):逐图 `GraphPlan::close`
    /// (销毁图 + 归还输入槽)。
    ///
    /// 次序:图在时其钉块(staging/槽)被**图租约**钉住 ⇒ `PeerRelease`/
    /// `free_blocks` 一律被响亮拒 —— 所以必须先卸图,后释放外部资源。
    /// 拆卸面不允许把控制流带跑:失败仅告警 + 记首错,不短路(全量尝试)。
    pub(crate) async fn close(&mut self) -> Result<()> {
        let mut first: Option<owl_iface::contract::ModelError> = None;
        record_close(&mut first, self.session.close().await);
        for d in self.depths.iter_mut() {
            if let Some(g) = d.verify.as_mut() {
                record_close(&mut first, g.close().await);
            }
            for g in d.fold.iter_mut() {
                record_close(&mut first, g.close().await);
            }
            for g in d.propose.iter_mut() {
                record_close(&mut first, g.close().await);
            }
        }
        if let Some(g) = self.batch.as_mut() {
            record_close(&mut first, g.close().await);
        }
        if let Some(g) = self.dflash.as_mut() {
            record_close(&mut first, g.close().await);
        }
        match first {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// 相位定相扇出(唯一入口;引擎 env 快照与各图 env 同步 ——
    /// GraphPlan 构造期只拿到 env 快照 ⇒ 逐相位显式下发,
    /// perf-framework.md §十.3)。
    pub(crate) fn set_phase(&mut self, p: owl_models::env::EvalPhase) {
        self.session.set_phase(p);
        for d in self.depths.iter_mut() {
            if let Some(g) = d.verify.as_mut() {
                g.set_phase(p);
            }
            for g in d.fold.iter_mut() {
                g.set_phase(p);
            }
            for g in d.propose.iter_mut() {
                g.set_phase(p);
            }
        }
        if let Some(g) = self.batch.as_mut() {
            g.set_phase(p);
        }
        if let Some(g) = self.dflash.as_mut() {
            g.set_phase(p);
        }
    }
}

// ===========================================================================
// 装配面(run() 委托;2026-10-15 自 engine.rs 下沉)
// ===========================================================================

/// 装配上下文(boot 侧只读事实的显式快照 —— 图装配零隐式依赖;
/// `pool` 借用:装配完成后 run() 仍要把 pool 移交 RunningEngine)
pub(crate) struct FleetBoot<'a> {
    /// 解释器执行环境(捕获期烘焙进各图闭包;EnvProvider: Copy)
    pub(crate) env: EnvProvider,
    /// **张量并行上下文(S4 ⑤;`None` = 单卡)** —— 与 env 同律：装配期
    /// 烘进本族每张图(eager/捕获/piecewise 三路求值都带上);`clone` 为
    /// 轻句柄(每个 GraphPlan/桶各自持一份)。
    pub(crate) tp: Option<TpCtx>,
    /// 模型几何(hq/hkv/…/hidden/dtype;Copy)
    pub(crate) dims: ModelDims,
    pub(crate) pool: &'a StatePool,
    pub(crate) model: Arc<Model>,
    pub(crate) rope: owl_models::layers::rope::Rope,
    /// 引擎级 spec 运行态(模式/深度/草稿器/草稿 rope/tap 开关)
    pub(crate) spec: &'a SpecRuntime,
    /// boot 旋钮(no_graph/ts_probe/capture_slab/debug)
    pub(crate) knobs: &'a EngineKnobs,
}

impl<D: DeviceClient> GraphFleet<D> {
    /// **图族总装配**（`extra_faces` 是 pop 消费桶；顺序有语义）：
    /// ① 主会话图（warmup+捕获）→ ② 每档 verify（spec 且 ① Captured）
    /// → ③ 每档 fold/propose（依赖本档 verify 输出块）→ ④ DFlash2 单图。
    /// adaptive 档 = 多档循环；单档 = 历史顺序。
    ///
    /// 降级语义(A1.4):任何一环捕获失败只降级该环(对应回 eager 路),
    /// 不连锁;`capture_outcome` 只记主会话图(它决定回放态/直发态)。
    pub(crate) async fn assemble(
        face: D,
        vface: Option<D>,
        mut extra_faces: Vec<D>,
        boot: FleetBoot<'_>,
        t0: T,
    ) -> Result<Self> {
        // ① 主会话图
        // 合批 decode 图(B5.1):独立 face;捕获失败 ⇒ None
        let bface_src = if boot.knobs.max_batch > 1 && !boot.knobs.no_graph {
            face.face_clone()
        } else {
            None
        };
        // V1c:decode_fi 开 ⇒ 同图走 piecewise,断点挂 decode 核
        let (mut session, capture_outcome, decode_fi_node) =
            Self::plan_session(face, &boot, t0).await?;
        let mut phase = crate::engine::VramPhase::new();
        phase.mark_info("graph:session", session.face_mut().mem_info().await);
        // ② verify 桶形图（自适应档可能有多张：每个 active depth 一张）。
        // 先为每档领一张 face（第一张 = vface，其余从 extra_faces 头部取），
        // 再逐档建 verify → fold/propose；图族不能跨 depth 复用（见 DepthGraphs 注）。
        let depth_candidates = boot.spec.depth_candidates();
        let mut verify_faces: Vec<D> = Vec::new();
        if boot.spec.depth > 0
            && matches!(capture_outcome, PlanOutcome::Captured)
        {
            if let Some(v) = vface {
                verify_faces.push(v);
            }
            while verify_faces.len() < depth_candidates.len() {
                let Some(f) = extra_faces.pop() else {
                    return Err(ModelError::EngineInvariant {
                        what: format!(
                            "verify face 不足（需要 {} 档，实得 {}；检查 fleet_faces 预算）",
                            depth_candidates.len(),
                            verify_faces.len()
                        ),
                    });
                };
                verify_faces.push(f);
            }
        }
        let mut depth_graphs: Vec<DepthGraphs<D>> = Vec::new();
        let mut verify_fi_node = false;
        if boot.spec.depth > 0 && !verify_faces.is_empty() {
            let mut faces = verify_faces.into_iter();
            for depth in depth_candidates {
                let vface = faces.next().ok_or_else(|| ModelError::EngineInvariant {
                    what: format!("verify face 缺席（depth={depth}）"),
                })?;
                let (verify, fi_node) = Self::plan_verify(vface, &boot, t0, depth).await?;
                phase.mark_info(
                    &format!("graph:verify(depth={depth})"),
                    session.face_mut().mem_info().await,
                );
                verify_fi_node |= fi_node;
                let Some(vg) = verify.as_ref() else {
                    if boot.spec.adaptive_depth {
                        return Err(ModelError::Msg(format!(
                            "[spec] adaptive_depth 要求 depth={depth} 的 verify 图捕获成功，\
                             但该档降级为 eager；请查图捕获日志/显存预算"
                        )));
                    }
                    depth_graphs.push(DepthGraphs {
                        depth,
                        verify: None,
                        fold: Vec::new(),
                        propose: Vec::new(),
                    });
                    continue;
                };
                let fold = Self::plan_fold_family(vg, &mut extra_faces, &boot, t0, depth).await?;
                let propose =
                    Self::plan_propose_family(vg, &mut extra_faces, &boot, t0, depth).await?;
                phase.mark_info(
                    &format!("graph:fold+propose(depth={depth})"),
                    session.face_mut().mem_info().await,
                );
                if boot.spec.adaptive_depth {
                    let want_fold = depth;
                    let want_propose = depth + 1;
                    if fold.len() != want_fold || propose.len() != want_propose {
                        return Err(ModelError::Msg(format!(
                            "[spec] adaptive_depth depth={depth} 图族不完整：\
                             fold {}/{want_fold}，propose {}/{want_propose}；\
                             这是启动期硬失败，避免运行期半档回退",
                            fold.len(),
                            propose.len()
                        )));
                    }
                }
                depth_graphs.push(DepthGraphs { depth, verify, fold, propose });
            }
        }
        if boot.spec.adaptive_depth
            && depth_graphs.len() != boot.spec.depth_candidates().len()
        {
            return Err(ModelError::Msg(format!(
                "[spec] adaptive_depth 需要全部 depth 图族，实得 {} / {}（主会话图未捕获？）",
                depth_graphs.len(),
                boot.spec.depth_candidates().len()
            )));
        }
        // ③b DFlash2 propose 单图（固定 depth = spec.depth；adaptive 只服务 MTP）。
        let dflash = if let Some(vg) = depth_graphs
            .iter()
            .find(|g| g.depth == boot.spec.depth)
            .and_then(|g| g.verify.as_ref())
        {
            Self::plan_dflash(vg, &mut extra_faces, &boot, t0).await?
        } else {
            None
        };
        // ③c **切档 propose 收割图**(刀3;仅 rank0 且词表切档):
        // 吃 dflash 图的 topk/sel_proj 输出块(刀2 已验证 of_block 这条路)
        // + peer 候选输入槽 ⇒ 图内合并 + selector,输出 select 单缓冲。
        let dflash_select = match (dflash.as_ref(), boot.spec.drafter.as_ref()) {
            (Some(dg), Some(Drafter::DFlash2(_)))
                if boot.model.logits_vocab() != boot.model.vocab_size()
                    && boot.tp.as_ref().map(|t| t.plan.rank()).unwrap_or(0) == 0 =>
            {
                Self::plan_dflash_select(dg, &mut extra_faces, &boot, t0).await?
            }
            _ => None,
        };
        // ③.5 **合批 decode 图**(B5.1 M1b):T = max_batch 定形,行 = 会话。
        // 捕获失败只降级本环(eager 臂仍在执行器里),不连锁。
        let batch = if boot.knobs.max_batch > 1 {
            match bface_src {
                None => {
                    owl_shared::log_warn!(
                        "batch",
                        "合批图:face_clone 不可用 ⇒ 走 eager 臂"
                    );
                    None
                }
                Some(bf) => match Self::plan_batch_decode(bf, &boot, t0).await {
                    Ok((g, outcome)) => {
                        if let PlanOutcome::EagerFallback { reason } = &outcome {
                            owl_shared::log_warn!("batch", "合批图捕获降级为 eager:{reason}");
                        }
                        Some(g)
                    }
                    Err(e) => {
                        owl_shared::log_warn!("batch", "合批图装配失败(回 eager):{e}");
                        None
                    }
                },
            }
        } else {
            None
        };
        phase.mark_info("graph:batch", session.face_mut().mem_info().await);
        // **现役臂自证**(2026-10-16 立;2026-10-18 改**事实位**)。
        // ⚠️ 判据 = 建树期"FI 面是否真登记"(`*_fi_node`),**不是旋钮** ——
        // 按旋钮猜曾把"FI 根本没进图、跑的是回落臂"报成 "FI(单图)"(坑册 §58)。
        {
            // 报法 = **请求臂 + 节点事实**:`fi(图内FI)` = 臂是 fi 且图里真有 FI 槽;
            // 若臂是 fi 但没有 `(图内FI)` 后缀 ⇒ 图/层两套真源打架(单键化前的
            // 静默坑),这一行直接暴露。
            let arm = |fi_node: bool, want: &str| -> String {
                if fi_node {
                    format!("{want}(图内FI)")
                } else {
                    want.to_string()
                }
            };
            owl_shared::log_info!(
                "boot",
                "现役臂: prefill={} | verify={} | decode={} | spec=depth{} | kv_fp8={}",
                if boot.knobs.env.attn.fi { "FI" } else { "自研读核" },
                arm(verify_fi_node, boot.knobs.env.attn.verify_arm.as_str()),
                arm(decode_fi_node, boot.knobs.env.attn.decode_arm.as_str()),
                boot.spec.depth,
                boot.pool.geom.kv_fp8,
            );
        }
        // **图显存总账**(2026-10-16;user 令"先归因"):所有已捕获图的 slab
        // 容量合计 —— 此前只有非 piecewise 的图能对上账(每条 `slab 定尺`),
        // 而 piecewise 的 34 条段 slab 在日志里完全不可见 ⇒ 归因靠猜。
        // 口径 = 各图 `graph_begin` 真分配容量(定尺 hint)之和。
        {
            let mb = |v: usize| v >> 20;
            let s = session.slab_bytes();
            let v: usize = depth_graphs
                .iter()
                .filter_map(|g| g.verify.as_ref())
                .map(|g| g.slab_bytes())
                .sum();
            let fo: usize = depth_graphs
                .iter()
                .flat_map(|g| g.fold.iter())
                .map(|g| g.slab_bytes())
                .sum();
            let pr: usize = depth_graphs
                .iter()
                .flat_map(|g| g.propose.iter())
                .map(|g| g.slab_bytes())
                .sum();
            let n_fold: usize = depth_graphs.iter().map(|g| g.fold.len()).sum();
            let n_propose: usize = depth_graphs.iter().map(|g| g.propose.len()).sum();
            let df = dflash.as_ref().map(|g| g.slab_bytes()).unwrap_or(0);
            let dfs = dflash_select.as_ref().map(|g| g.slab_bytes()).unwrap_or(0);
            let total = s + v + fo + pr + df + dfs;
            owl_shared::log_info!(
                "boot",
                "slab 总账:session {}MiB + verify {}MiB + fold {}MiB({}) + propose {}MiB({}) + dflash {}MiB = {}MiB(Σ 已捕获图)‖ 引擎账 vram::live={}MiB(含池/权重/全部真分配)",
                mb(s),
                mb(v),
                mb(fo),
                n_fold,
                mb(pr),
                n_propose,
                mb(df + dfs),
                mb(total),
                owl_shared::vram::live_bytes() >> 20,
            );
            // **预算对账**(B1,2026-10-17):预算里的图项是个**校准常数**
            // (`GRAPH_SLAB_BUDGET`,取自 spec-dflash 档实测),而池上限在**捕获之前**
            // 就已按它定死 ⇒ 图族/臂换代后若实测变大,池上限就会偏大(请求级 OOM
            // 的典型形状)。此处把实测 Σ 与预算项摆在一起:超了打 ⚠️。
            let budget = crate::state::GRAPH_SLAB_BUDGET;
            let mbf = |v: usize| v as f64 / 1048576.0;
            if total > budget {
                owl_shared::log_warn!(
                    "boot",
                    "⚠️ 图 slab 预算对账:实测 {:.1}MiB > 预算假设 {}MiB(+{:.1}MiB,图族/臂换代后预算偏小)⇒ 池上限可能偏大(请求级 OOM 风险):把 GRAPH_SLAB_BUDGET 校准为新实测值(engine/src/state.rs),或下调 max_seq / per_tok",
                    mbf(total),
                    mb(budget),
                    mbf(total - budget),
                );
            } else {
                owl_shared::log_info!(
                    "boot",
                    "图 slab 预算对账:实测 {:.1}MiB ≤ 预算假设 {}MiB(余量 {:.1}MiB ✓)",
                    mbf(total),
                    mb(budget),
                    mbf(budget - total),
                );
            }
        }
        Ok(Self {
            session,
            capture_outcome,
            depths: depth_graphs,
            batch,
            dflash,
            dflash_select,
        })
    }


    /// ① 主会话图(槽 → 整模单树;状态句柄捕获,模型 Arc 共享)。
    /// 捕获期禁 Htod:常量标量走持久槽;KV/GDN 双槽分立 ——
    /// KV 槽 = 本 token 自己的格子(窗 = [slot-kv_len+1, slot],随步进;
    /// 2026-09-26 窗口语义探针定谳:slots 恒 0 读块外垃圾行);
    /// GDN 槽 = 序列状态格,turn 内恒 0(递推状态按序列累积)。
    /// **face 句柄克隆**(诊断/投影面出口:`GET /debug/blocks` 的 GPU 命令通道;
    /// 取主会话图的 face —— 克隆 = 同一个 actor 的另一个发送端)
    pub fn face_clone(&self) -> Option<D> {
        self.session.face_clone()
    }

    /// **设备上下文面**(生命周期节点 N1/N4/N5 用):主会话图的 face。
    ///
    /// 为什么改这一个就够:`DeviceHandle` 的 owner/class 上下文是 `Arc<Mutex<..>>`
    /// **共享**的(face_clone 出来的句柄与本体同源)⇒ 图族每张图各持一个句柄,
    /// 但设一次即全族生效(逐个设 = 漏设的温床)。
    pub(crate) fn dev_ctx_face_mut(&mut self) -> &mut D {
        self.session.face_mut()
    }

    /// 主会话图 face 的只读面(诊断计数:`hook_fail`/窗栈)
    pub(crate) fn dev_ctx_face(&self) -> &D {
        self.session.face_ref()
    }

    /// main 侧签名(V1c 宿主标量槽):额外返回 FI 断点臂的 `ctx_total` 槽
    async fn plan_session(
        face: D,
        boot: &FleetBoot<'_>,
        t0: T,
    ) -> Result<(GraphPlan<D>, PlanOutcome, bool)> {
        let env = boot.env;
        let tp = boot.tp.clone();
        let pool = boot.pool;
        let dims = boot.dims;
        let model = Arc::clone(&boot.model);
        let rp = boot.rope.clone();
        // **本 rank 的 logits 宽度**(P1.1 词表切档 = 半表;未切档 = 全词表):
        // decode 图的 logits 槽宽、设备 argmax 的 stride 都读它 ——
        // `vocab_size()` 仍是词表全量(embed/tokenizer 面用)。
        let vocab = boot.model.logits_vocab();
        let knobs = boot.knobs;
        // **词表切面**(2026-10-26):decode 侧候选采样只对"两半表合并"有意义。
        // 候选槽**恒随词表切**捕获(不再看 flavor 的 sampling/greedy):topk32
        // 每轮 ~60µs/rank(0.2–0.4%),换来"请求级 temp>0 在 greedy 档也不再
        // 回退 4MB logits"——用固定小税消掉一个静默陷阱。
        let head_split = vocab != boot.model.vocab_size();
        let cand_on = head_split;
        // 本 rank 半表在全局词表里的起点(rank0=0 / rank1=half)。候选索引
        // 保持 local(host 合并照旧加偏移),只有设备端惩罚查历史时用全局 id。
        let vocab_offset = if head_split {
            vocab * boot.tp.as_ref().map(|t| t.plan.rank()).unwrap_or(0)
        } else {
            0
        };
        // taps 层号表(闭包 'static ⇒ Arc 持有;单一事实源 = SpecRuntime)
        let tap_ids = std::sync::Arc::clone(&boot.spec.dflash_tap_ids);
        let kv_caches = pool.kv_caches();
        let gdn_caches = pool.gdn_caches();
        let bt_leaf = pool.bt_leaf();
        let attn_v2 = pool.attn_v2_scratch();
        // XQA 裸 decode 面(2026-10-13 ②):池常量件 + 两个图输入槽
        // (`xqa_seq` = kv_len 的 **u32 位型** —— 核按 `const uint32_t*` 读,
        //  直接写 f32 数值会拿到浮点位型;`xqa_sem` = 每轮写零)。
        // 槽只在 `decode_xqa` 开时才加 ⇒ 关时图拓扑与历史一致(零影响)。
        let xqa_decode_on = matches!(knobs.env.attn.decode_arm, owl_shared::config::DecodeArm::Xqa);
        let xqa_decode_ctx = if xqa_decode_on { pool.xqa_ctx() } else { None };
        let xqa_dec_sem_n = if xqa_decode_on { pool.xqa_sem_n() } else { 0 };
        // B4:decode 图 taps(DFlash2 降级期逐 token 草稿同步的取数面;
        // tapped_hidden 零额外计算,
        // 仅多 5 个 [1,hidden] 输出槽写;非 dflash 走原 forward(拓扑不变)
        let decode_taps = boot.spec.dflash_tap_count > 0;
        let fi_kcv: (Vec<TensorOps>, Vec<TensorOps>) = kv_caches.iter().cloned().unzip();
        // **图档单图烘 plan 档**(2026-10-18;`[dispatch] fi_plan_bake`):
        // Some(ctx) = decode **不进 piecewise**,FI 核留图内,且喂 FI 的
        // `ctx_total` 用该上界烘死 ⇒ 烘出的调度表按上界枚举工作项(运行期
        // `chunk_start/end = min(.., kv_len)` 夹取;T=1 实测与逐 ctx 表同误差带)。
        // 上界必须 ≥ 任何可能出现的 ctx(否则表的工作项数 < 运行期 chunk 数 ⇒
        // V1 那个 stride 错),由配置校验拒启兜底。
        let plan_bake: Option<usize> = knobs.fi_plan_bake;
        // **单键臂 → 闭包常量**(2026-10-21 合并):闭包捕获 'static ⇒ 先取值
        let dfi_on = knobs.env.attn.decode_arm.uses_fi();
        let fi_nb = pool.geom.nb;
        // **臂事实位**(2026-10-18,同 verify 律):decode 侧 FI 面是否真登记。
        let dfi_node_seen = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dfi_node_seen_c = dfi_node_seen.clone();
        let forward = move |sc: &PlanCtx| -> Result<()> {
            // FI 面持有处(活得比 ctx 长:FiPrefillCtx 借它们)
            let fi_tensors: (TensorOps, TensorOps, TensorOps, TensorOps);
            let ids = sc.input("frontier")?;
            let pos = sc.input("pos")?;
            let kv_len = sc.input("kv_len")?;
            let kv_slot = sc.input("kv_slot")?;
            let gdn_slot = sc.input("gdn_slot")?;
            let ts_buf = sc.input_opt("ts_buf")?;
            // **B-3d 选择层视图**(`select_decode`;方案 A):页表与上下文长走
            // **图输入槽** —— 每轮由引擎写"选中页 ++ 新生成尾部"。
            // 为何必须走输入槽(而不是池里那块视图页表):图烘的是**指针**,
            // 池块指针虽稳,但 xqa_pt 的教训是「页表内容必须每轮可换」——
            // 输入槽正是为此存在(§7.6)。
            let sel_pt = sc.input_opt("sel_pt")?;
            let sel_len = sc.input_opt("sel_len")?;
            let (bt_step, len_step) = match (sel_pt, sel_len) {
                (Some(pt), Some(l)) => (pt, l),
                _ => (bt_leaf.clone(), kv_len.clone()),
            };
            let kvs_step: Vec<KvBuffers> = kv_caches
                .iter()
                .map(|(k, v)| KvBuffers {
                    k_cache: k.clone(),
                    v_cache: v.clone(),
                    slots: kv_slot.clone(),
                    kv_lens: len_step.clone(),
                    block_tables: bt_step.clone(),
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdn_caches
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: gdn_slot.clone(),
                })
                .collect();
            let mut ctx = ForwardCtx::model_decode(1, &pos, &kvs_step, &rp, &gdns_step);
            ctx.env = env; // 捕获期烘焙(图闭包 env 定格;回放沿用)
            ctx.ts_buf = ts_buf; // 刀D 时间戳探针(None = 关)
            ctx.attn_v2 = attn_v2.clone(); // v2 分页 decode scratch(None = v1)
            // XQA 裸 decode 臂(2026-10-13 ②):面 = pool 常量件 + 图输入 seq/sem
            // (sem 走输入槽的正解见工单 §2.7 —— 禁跨账本 memset 命令)
            if let Some(x) = xqa_decode_ctx.as_ref() {
                let seq_in = sc.input("xqa_seq")?;
                let pt_in = sc.input("xqa_pt")?;
                let sem_in = sc.input("xqa_sem")?;
                ctx.xqa = Some(owl_models::module::XqaVerifyCtx {
                    // 页表 = 输入槽(由核侧每轮写会话真链);verify 图同款
                    page_table: pt_in,
                    seq_lens: seq_in,
                    mask: x.mask.clone(),
                    semaphores: sem_in,
                    scratch: x.scratch.clone(),
                    q_seq_len: 1,
                    max_seq_len: x.max_seq_len,
                    sm_count: x.sm_count,
                });
            }
            // **FI 图内臂**(session/T=1;W3 转正 + W5 收口):表四件套槽 → U32
            // 视图 + FiPrefillCtx + `ctx_total` 烘上界;当步**精确**表由引擎在
            // 窗外 `fi_plan_sync` 刷(图不烘表)。
            // ⚠️ 门 = **两件套都在场**(`dfi_on` + `plan_bake`):缺上界 ⇒ 本树
            // **不**换 FI 臂(回落 v2/XQA),而不是拿占位 ctx 烘表 —— 后者是
            // "文本连贯但错"的历史来源;校验层另有"FI 臂 ⇒ bake 在场"拒启门。
            if let (true, Some(bake)) = (dfi_on, plan_bake) {
                ctx.fi_plan_ctx = Some(bake);
                let blk_id = |t: &TensorOps| -> Result<u64> {
                    match t.op() {
                        owl_models::ops::Op::Block { id } => Ok(*id),
                        _ => Err(owl_iface::contract::ModelError::Msg(
                            "FI 表槽非块叶(内部契约违例)".into(),
                        )),
                    }
                };
                let (q_cu_t, idx_t, ind_t, ll_t) = (
                    TensorOps::of_block(blk_id(&sc.input("fi_q_cu")?)?, Dtype::U32, vec![2]),
                    TensorOps::of_block(blk_id(&sc.input("fi_indices")?)?, Dtype::U32, vec![fi_nb]),
                    TensorOps::of_block(blk_id(&sc.input("fi_indptr")?)?, Dtype::U32, vec![2]),
                    TensorOps::of_block(blk_id(&sc.input("fi_last_len")?)?, Dtype::U32, vec![1]),
                );
                fi_tensors = (q_cu_t, idx_t, ind_t, ll_t);
                ctx.fi = Some(FiPrefillCtx {
                    kcs: &fi_kcv.0,
                    vcs: &fi_kcv.1,
                    q_cu: &fi_tensors.0,
                    indices: &fi_tensors.1,
                    indptr: &fi_tensors.2,
                    last_len: &fi_tensors.3,
                });
                dfi_node_seen_c.store(true, std::sync::atomic::Ordering::Relaxed);
            }
            // B4:降级期取数面 —— tapped_hidden 拿末层 hidden + 5 层 taps
            // (零额外前向;l m_head 与 forward 同式)。非 dflash 原路(位等价)
            let tree = if decode_taps {
                let (hidden, taps) = model.tapped_hidden(&ids, &ctx, tap_ids.as_slice());
                for (i, t) in taps.iter().enumerate() {
                    sc.output(format!("tap{i}"), t)?;
                }
                model.embed.lm_head_matmul(&hidden, &ctx.env.gguf)
            } else {
                model.forward(&ids, &ctx)
            };
            // E3 设备采样:token = argmax(logits)(f32 数值过线,契约 5)
            // ⚠️ P1.1:stride = **本 rank 的 logits 宽度**(词表切档 = 半表);
            // 用全量 `vocab_size()` 会跨行读(越界)。半表档的 token 只是**局部**
            // argmax —— 消费面走 host 拼接(见 `RunningEngine::read_full_logits`)。
            let tok = owl_models::ops::argmax_f32idx(&tree, vocab, 0);
            sc.output("token", &tok)?;
            sc.output("logits", &tree)?;
            // **候选采样槽**(2026-10-26):裸 decode 采样不再回读整行 logits。
            // 槽恒随词表切捕获(请求级 temp 可在 greedy flavor 上打开采样);
            // 设备端 rep_penalty 也吃同一切档门,固定 hist/penalty 槽。
            if cand_on {
                let hist = sc.input("rep_hist")?;
                let rep = sc.input("rep_penalty")?;
                let cand = owl_models::ops::topk32_of_logits(
                    &tree,
                    &hist,
                    &rep,
                    1,
                    owl_models::layers::dflash2::TOPK_NPARTS,
                    vocab_offset,
                );
                sc.output("cand", &cand)?;
            }
            Ok(())
        };

        let desc = GraphPlanDesc {
                inputs: {
                    let mut v = vec![
                        InputSlot::f32("frontier", 1),
                        InputSlot::f32("pos", 1),
                        InputSlot::f32("kv_len", 1),
                        InputSlot::f32("kv_slot", 1),
                        InputSlot::f32("gdn_slot", 1).init(vec![0.0]),
                    ];
                    if xqa_dec_sem_n > 0 {
                        v.push(InputSlot::f32("xqa_seq", 1).init(vec![0.0]));
                        v.push(InputSlot::f32("xqa_sem", xqa_dec_sem_n).init(vec![0.0; xqa_dec_sem_n]));
                        // XQA **页表必须走输入槽**(每轮写会话真页链):
                        // 常量恒等链 0..nb 只在"单会话独占池(块从 0 起)"时成立;
                        // 多会话(agent:每请求新会话,块链从空闲链续)会让 XQA
                        // 整段读到**别家 KV** —— 实测"答上一题"(§7.6 定谳,
                        // 2026-10-14)。init 保留恒等链(单会话行为不变);
                        // 槽尺寸 = 池 nb(与会话链上界一致,尾部补零)
                        v.push(InputSlot::f32("xqa_pt", pool.geom.nb).init(
                            (0..pool.geom.nb).map(|i| f32::from_bits(i as u32)).collect(),
                        ));
                    }
                    if knobs.select_decode {
                        // **B-3d 选择层视图**:`sel_pt` = 视图页链(nb 槽,尾补零),
                        // `sel_len` = 视图内 token 数(≤ 池 token 容量)。
                        // init = 恒等链 + 全量长(单会话且视图未启用时行为不变)
                        // 编码 = **数值 f32**(与 `write_bt` 同源:block_tables 在本族
                        // 声明为 F32 rank2;XQA 的位型编码是另一套,别混)
                        v.push(InputSlot::f32("sel_pt", pool.geom.nb).init(
                            (0..pool.geom.nb).map(|i| i as f32).collect(),
                        ));
                        v.push(InputSlot::f32("sel_len", 1).init(vec![1.0]));
                    }
                    if knobs.ts_probe {
                        v.push(InputSlot::f32("ts_buf", 4096));
                    }
                    // FI 断点臂(session/T=1;`decode_fi`):表四件套(位型过线,
                    // u32 → f32)。init 与占位 ctx_total(=1 行)自洽:nb_pages=1
                    if knobs.env.attn.decode_arm.uses_fi() {
                        v.push(InputSlot::f32("fi_q_cu", 2).init(vec![0.0, f32::from_bits(1)]));
                        v.push(InputSlot::f32("fi_indices", pool.geom.nb)
                            .init((0..pool.geom.nb).map(|i| f32::from_bits(i as u32)).collect()));
                        v.push(InputSlot::f32("fi_indptr", 2).init(vec![0.0, f32::from_bits(1)]));
                        v.push(InputSlot::f32("fi_last_len", 1).init(vec![f32::from_bits(1)]));
                    }
                    if cand_on {
                        // 设备端重复惩罚的固定输入槽(形状不随内容变 => 不建桶)。
                        // init = 全哨兵 + 1.0,与"惩罚关"的旧路径逐位一致。
                        v.push(InputSlot::u32("rep_hist", REP_HISTORY_WINDOW).init(
                            vec![f32::from_bits(REP_HISTORY_SENTINEL); REP_HISTORY_WINDOW],
                        ));
                        v.push(InputSlot::f32("rep_penalty", 1).rank1().init(vec![1.0]));
                    }
                    v
                },
                outputs: {
                    let mut o = vec![
                        OutputSlot { name: "logits".into(), shape: vec![1, vocab], dtype: dims.dtype },
                        OutputSlot { name: "token".into(), shape: vec![1], dtype: Dtype::F32 },
                    ];
                    if cand_on {
                        o.push(OutputSlot {
                            name: "cand".into(),
                            shape: vec![64],
                            dtype: Dtype::F32,
                        });
                    }
                    if decode_taps {
                        for i in 0..5 {
                            o.push(OutputSlot {
                                name: format!("tap{i}"),
                                shape: vec![1, dims.hidden],
                                dtype: dims.dtype,
                            });
                        }
                    }
                    o
                },
                // 逃生开关(C1 禁 graph 裁决配套):no_graph → eager
                // 直发(decode 逐步 eval,无捕获回放)—— 图内/图外行为
                // A/B 的对照臂(2026-10-01 paged decode 质量案)
                capture: !knobs.no_graph,
                slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                debug_probe: knobs.probes.debug,
                // 锚定活性回收(2026-10-15 归因之修②;理由同族桶形图):
                // plain = 分配总和 ⇒ 开 scoped,容量降到峰值活跃集;恒等门同族。
                reclaim_capture: true,
                warmup: true,
        };
        // **单图**(W3/W5):FI 在图内 + `ctx_total` 烘上界;段驱动件与断点面
        // 整体退役(它们曾在此承载 decode 的"每步按真 ctx 重算 plan")。
        let single_baked = plan_bake.is_some() && dfi_on;
        let (session, capture_outcome) = {
            let (g, o) = GraphPlan::plan(face, env, tp, desc, forward).await?;
            if single_baked {
                owl_shared::log_info!(
                    "boot",
                    "session 单图捕获(FI 图内 + plan 上界 {} 烘死) +{:.2}s",
                    plan_bake.unwrap_or(0),
                    t0.elapsed().as_secs_f32()
                );
            }
            (g, o)
        };
        if let PlanOutcome::EagerFallback { reason } = &capture_outcome {
            owl_shared::log_warn!("engine", "捕获降级为 eager:{reason}");
        }
        owl_shared::log_info!("boot", "GraphPlan plan(warmup+捕获) {:.2}s", t0.elapsed().as_secs_f32());
        owl_shared::log_info!(
            "boot",
            "decode 面事实:FI 节点登记={} | single_baked={} bake={:?} | 臂事实(attn.arm.decode.*):{}",
            dfi_node_seen.load(std::sync::atomic::Ordering::Relaxed),
            single_baked,
            plan_bake,
            arm_facts("attn.arm.decode.")
        );
        Ok((session, capture_outcome, dfi_node_seen.load(std::sync::atomic::Ordering::Relaxed)))
    }

    /// **合批 decode 图**(B5.1 M1b;2026-10-20):T = `max_batch` 定形,
    /// 行 = 会话。槽面全部按行(`frontier/pos/kv_len/kv_slot/gdn_slot` [T]),
    /// 页表 = 持久 `bt_batch [T,nb]`(每步重写),注意力 = v2 seq 臂
    /// (`seqs=T`),输出 = 逐行 argmax + logits [T,vocab]。
    ///
    /// 捕获失败 ⇒ 执行器回 eager 臂(`graphs.batch = None`),语义不变。
    async fn plan_batch_decode(
        face: D,
        boot: &FleetBoot<'_>,
        t0: T,
    ) -> Result<(GraphPlan<D>, PlanOutcome)> {
        let env = boot.env;
        let tp = boot.tp.clone();
        let pool = boot.pool;
        let dims = boot.dims;
        let model = Arc::clone(&boot.model);
        let rp = boot.rope.clone();
        let batch = boot.knobs.max_batch;
        let vocab = boot.model.logits_vocab();
        let kv_caches = pool.kv_caches();
        let gdn_caches = pool.gdn_caches();
        let scratch = pool.batch_attn_v2_scratch().ok_or_else(|| {
            ModelError::EngineInvariant {
                what: "合批图:batch_attn_v2 未分配(max_batch 档位不一致)".into(),
            }
        })?;
        let bt8 = pool.bt_batch_leaf().ok_or_else(|| ModelError::EngineInvariant {
            what: "合批图:bt_batch 未分配(max_batch 档位不一致)".into(),
        })?;
        let desc = GraphPlanDesc {
            inputs: vec![
                InputSlot::f32("frontier", batch),
                InputSlot::f32("pos", batch),
                InputSlot::f32("kv_len", batch),
                InputSlot::f32("kv_slot", batch),
                InputSlot::f32("gdn_slot", batch),
            ],
            outputs: vec![
                OutputSlot { name: "logits".into(), shape: vec![batch, vocab], dtype: dims.dtype },
                // Concat 核固定 8 槽(verify 图同款)⇒ token 输出恒 [8,1];
                // 执行器只取前 batch 行(batch ≤ 8 由配置门保证)
                OutputSlot { name: "token".into(), shape: vec![8, 1], dtype: Dtype::F32 },
            ],
            capture: true,
            slab_fixed: boot.knobs.cuda.capture_slab_mb.is_some(),
            debug_probe: boot.knobs.probes.debug,
            reclaim_capture: true,
            warmup: true,
        };
        let forward = move |sc: &PlanCtx| -> Result<()> {
            let ids = sc.input("frontier")?;
            let pos = sc.input("pos")?;
            let kv_len = sc.input("kv_len")?;
            let kv_slot = sc.input("kv_slot")?;
            let gdn_slot = sc.input("gdn_slot")?;
            let kvs_step: Vec<KvBuffers> = kv_caches
                .iter()
                .map(|(k, v)| KvBuffers {
                    k_cache: k.clone(),
                    v_cache: v.clone(),
                    slots: kv_slot.clone(),
                    kv_lens: kv_len.clone(),
                    block_tables: bt8.clone(),
                })
                .collect();
            let gdns_step: Vec<GdnBuffers> = gdn_caches
                .iter()
                .map(|g| GdnBuffers {
                    conv_q: g.conv_q.clone(),
                    conv_k: g.conv_k.clone(),
                    conv_v: g.conv_v.clone(),
                    rec: g.rec.clone(),
                    slots: gdn_slot.clone(),
                })
                .collect();
            let mut ctx = ForwardCtx::model_decode(batch, &pos, &kvs_step, &rp, &gdns_step);
            ctx.env = env;
            ctx.attn_v2 = Some(scratch.clone());
            ctx.v2_bt8 = Some(&bt8);
            let hidden = model.last_hidden(&ids, &ctx);
            let logits = model.embed.lm_head_matmul(&hidden, &ctx.env.gguf);
            sc.output("logits", &logits)?;
            let rows: Vec<TensorOps> = (0..batch)
                .map(|r| owl_models::ops::argmax_f32idx(&logits, vocab, r * vocab))
                .collect();
            let mut root = TensorOps::call(owl_models::ops::SemanticKernel::Concat);
            for i in 0..owl_models::kernel::CONCAT_ROWS_SLOTS {
                root = root.arg(rows.get(i).unwrap_or(&rows[0]));
            }
            let root = root
                .arg_usize(8)
                .arg_usize(1)
                .arg_usize(1)
                .with_shape(Dtype::F32, vec![8, 1]);
            sc.output("token", &root)?;
            Ok(())
        };
        let (g, o) = GraphPlan::plan(face, env, tp, desc, forward).await?;
        owl_shared::log_info!(
            "boot",
            "合批 decode 图计划(T={} / vocab={})完成 +{:.2}s outcome={:?}",
            batch,
            vocab,
            t0.elapsed().as_secs_f32(),
            o
        );
        Ok((g, o))
    }

    /// ② verify 桶形图(E5-M4;T = depth+1 定形;spec 模式专用)—— 消
    /// 投机轮的整模 eager 发射税(~830 发 × ~25µs)。FI **在图内**(W4 转正;
    /// plan ctx 烘上界 + 引擎每轮窗外刷精确表);输出 = tok/hid +
    /// **GDN fold 记录**(每层 8 件,tap 收集 → 多根共享 memo 单次归约,
    /// 刀 1.6 语义)。捕获失败 → None → 执行器落旧 eager verify+重放路。
    async fn plan_verify(
        vface: D,
        boot: &FleetBoot<'_>,
        t0: T,
        depth: usize,
    ) -> Result<(Option<GraphPlan<D>>, bool)> {
        let pool = boot.pool;
        let dims = boot.dims;
        let env = boot.env;
        let tp = boot.tp.clone();
        let knobs = boot.knobs;
        let depth1 = depth + 1;
        let tap_ids = std::sync::Arc::clone(&boot.spec.dflash_tap_ids);
        let model_v = Arc::clone(&boot.model);
        let rp_v = boot.rope.clone();
        let kv_caches = pool.kv_caches();
        let gdn_caches = pool.gdn_caches();
        let bt_leaf = pool.bt_leaf_flat();
        let bt_leaf_nb = *bt_leaf.shape().last().unwrap_or(&1);
        let verify_attn_v2 = pool.verify_attn_v2_scratch();
        // XQA 面(2026-10-13 D3):池常量件 + 图槽 init(i32/u32 位型走私进
        // f32 账叶 —— 与 sm_scale/fi_scale_bits 同款约定)
        let xqa_pool_ctx = pool.xqa_ctx();
        let xqa_pt_nb = bt_leaf_nb;
        let xqa_pt_init: Vec<f32> = (0..xqa_pt_nb)
            .map(|i| f32::from_bits(i as u32))
            .collect();
        let xqa_sem_n = xqa_pool_ctx.as_ref().map(|x| x.semaphores.shape().iter().product::<usize>()).unwrap_or(0);
        let xqa_mask_init: Vec<f32> = (0..depth1)
            .map(|i| f32::from_bits((1u32 << (i + 1)) - 1))
            .collect();
        let key_dim = dims.nk * dims.hk;
        let value_dim = dims.nv * dims.hv;
        // piecewise(可断捕获图;2026-10-15 perf 会话立项):verify 图要把
        // **注意力核**移出捕获图(核的 plan 在 split 档依赖 ctx ⇒ 单图只对
        // 一个 ctx 有效)。开关 = `dispatch.verify_piecewise`;设计见
        // `docs/piecewise-断点图-设计与实现.md`。
        //
        // **verify 单图**(W4 转正 + W5 收口):FI 核留图内 + `ctx_total` 烘上界,
        // **表由引擎每轮 `fi_plan_sync` 刷当轮精确值**(图不烘自携 restore;
        // 配置层已校验二者配套)。verify 恒随 `[dispatch] fi_plan_bake`
        // (`fi_plan_bake_verify` 旋钮已随断点面一并退役)。
        let vfi_bake: Option<usize> = knobs.fi_plan_bake;
        // **臂 = 单值**(2026-10-21):verify 面是否带 FI 四件套槽,只看
        // `verify_arm`;从前是 `verify_fi` 布尔 + 层内 `verify_v2` 抢前的隐性组合。
        let fi_want = knobs.env.attn.verify_arm.uses_fi();
        let mut vouts = vec![
            OutputSlot { name: "tok".into(), shape: vec![depth1, 1], dtype: Dtype::F32 },

            OutputSlot { name: "hid".into(), shape: vec![depth1, dims.hidden], dtype: dims.dtype },
            // A11(2026-10-14):per-row logits 出槽 —— 投机轮的**采样接受**
            // 需要各行目标分布(原先只有 argmax ⇒ spec 臂不采样 =
            // 请求温度失效;A7/A8 根因,见 agent 专项 §7.11)。
            // 贪心档不回读(零额外 D2H);仅采样档付 ~2.4MB/轮。
            OutputSlot { name: "logits".into(), shape: vec![depth1, model_v.logits_vocab()], dtype: Dtype::F16 },
        ];
        // **词表切面**(2026-10-26):只有 `logits_vocab < vocab` 才需要
        // "两半表合并"的分布式通路(tokv / cand)。未切档(单卡 / 关
        // `tp_vocab_split`)**零额外节点零额外槽** —— 不许为不存在的合并付税。
        let head_split = model_v.logits_vocab() != model_v.vocab_size();
        let vocab_offset = if head_split {
            model_v.logits_vocab()
                * boot.tp.as_ref().map(|t| t.plan.rank()).unwrap_or(0)
        } else {
            0
        };
        if head_split {
            vouts.push(OutputSlot {
                name: "tokv".into(),
                shape: vec![depth1, 2],
                dtype: Dtype::F32,
            });
        }
        // **分布式采样候选槽**:词表切 + "采样档 flavor"才捕获(topk32 每轮
        // 有 ~50µs 成本;纯 greedy 档不该付)。请求级 temp>0 落在 greedy flavor
        // 上时没有该槽 ⇒ spec.rs 自动回退 logits 老路(见 `use_cand` 门)。
        let cand_on = head_split;
        if cand_on {
            vouts.push(OutputSlot {
                name: "cand".into(),
                shape: vec![2 * depth1 * 32],
                dtype: Dtype::F32,
            });
        }
        // DFlash2 target taps(E5-DF2;层输出残差流,sglang capture 同语义)
        // —— DFlash2 模式的主路径(子路径二分开关已于 2026-10-27 退役)
        let dflash_taps = boot.spec.dflash_tap_count > 0;
        let dflash_tap_decl = dflash_taps;
        if dflash_taps {
            for i in 0..5 {
                vouts.push(OutputSlot { name: format!("tap{i}"), shape: vec![depth1, dims.hidden], dtype: dims.dtype });
            }
        }
        for i in 0..gdn_caches.len() * owl_shared::config::SPEC_FOLD_RECORDS_PER_LAYER {
            let (_gi, f) = (i / owl_shared::config::SPEC_FOLD_RECORDS_PER_LAYER, i % owl_shared::config::SPEC_FOLD_RECORDS_PER_LAYER);
            let shape = match f {
                0 | 1 => vec![depth1, key_dim],
                2 => vec![depth1, value_dim],
                3 | 4 => vec![depth1, dims.nk, dims.hk],
                5 => vec![depth1, dims.nv, dims.hv],
                _ => vec![depth1, dims.nv],
            };
            vouts.push(OutputSlot { name: format!("r{i}") , shape, dtype: dims.dtype });
        }
        let desc = GraphPlanDesc {
                inputs: {
                    let mut v = vec![
                    InputSlot::f32("ids", depth1),
                    InputSlot::f32("pos", depth1),
                    InputSlot::f32("kv_slots", depth1),
                    InputSlot::f32("kv_lens", depth1),
                    InputSlot::f32("gdn_slot", 1),
                    InputSlot::f32("gdn_cu", 2).init(vec![0.0, depth1 as f32]),
                    InputSlot::f32("bt8", depth1 * bt_leaf_nb),
                    // XQA 面(2026-10-13 D3):seq_lens 逐轮变(u32 位型走私);
                    // 页表/掩码是常量,在槽 init 里填一次(捕获前 Htod 合法)
                    // ⚠️ **零长槽地雷**:`xqa_sem_n` 在"无 XQA 面"时为 0,
                    // 无条件声明就是零长槽 ⇒ 每轮 step 写 0 字节 ⇒ 后端
                    // `malloc_host(0)` 空指针崩(2026-10-13 挂账 #7 定谳:
                    // 0.8B spec 档踩的就是它 —— 之前只跑 fp8 门所以未暴露)。
                    // 取 `.max(1)`:拓扑恒常(不做条件声明),多写 4 字节无害。
                    InputSlot::f32("xqa_seq", 1).init(vec![0.0]),
                    InputSlot::f32("xqa_pt", xqa_pt_nb).init(xqa_pt_init.clone()),
                    InputSlot::f32("xqa_mask", depth1).init(xqa_mask_init.clone()),
                    // semaphores 走**图输入槽**(每轮 step 写零):
                    // ⚠️ 曾用 `memset_zero` 命令 → 跨账本解析踩雷(见工单
                    // §2.7 定谳);输入槽与 xqa_seq 同一条已验证通路
                    InputSlot::f32("xqa_sem", xqa_sem_n.max(1))
                        .init(vec![0.0; xqa_sem_n.max(1)]),
                    // FI 表四件套(位型过线:u32 → f32;核按 i32* 读)。init 与
                    // 占位 ctx_total(=depth1)自洽:nb_pages = ceil(T/page) = 1
                    InputSlot::f32("fi_q_cu", 2).init(vec![0.0, f32::from_bits(depth1 as u32)]),
                    InputSlot::f32("fi_indices", pool.geom.nb)
                        .init((0..pool.geom.nb).map(|i| f32::from_bits(i as u32)).collect()),
                    InputSlot::f32("fi_indptr", 2).init(vec![0.0, f32::from_bits(1)]),
                    InputSlot::f32("fi_last_len", 1).init(vec![f32::from_bits(depth1 as u32)]),
                    ];
                    // 设备端重复惩罚槽只随词表切存在(未切档零额外槽)。
                    if head_split {
                        v.push(InputSlot::u32("rep_hist", REP_HISTORY_WINDOW).init(
                            vec![f32::from_bits(REP_HISTORY_SENTINEL); REP_HISTORY_WINDOW],
                        ));
                        v.push(InputSlot::f32("rep_penalty", 1).rank1().init(vec![1.0]));
                    }
                    v
                },
                outputs: vouts,
                capture: !knobs.no_graph,
                slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                debug_probe: knobs.probes.debug,
                // 锚定活性回收(2026-10-15 归因之修②;理由同族桶形图):
                // plain = 分配总和 ⇒ 开 scoped,容量降到峰值活跃集;恒等门同族。
                reclaim_capture: true,
                warmup: true,
        };
        let fi_want_c = fi_want;
        // kcs/vcs = 主池叶子(FI 与其余臂同池;与 kv_caches 同源)
        let fi_kcv: (Vec<TensorOps>, Vec<TensorOps>) = kv_caches.iter().cloned().unzip();
        let fi_nb = pool.geom.nb; // 闭包 'static:借 boot 的字段必须先取值(同 kv_caches 纪律)
        // (2026-10-21 单键化后本闭包不再分 decode FI 臂 —— 此处曾取
        //  `dfi_on`,已随 decode 路径收口删除;prefill 面走 fi_want/vfi_bake)
        // 图内臂把 FI plan 的 ctx 烘成上界(表由引擎每轮 sync 成精确值)
        let vfi_bake_c = vfi_bake;
        // **臂事实位**(2026-10-18,同 "槽在场≠节点在场" 律):FI 面是否真登记进本图
        // —— 臂自证不许再按旋钮猜(本轮假绿的一半就是它)。
        let fi_node_seen = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let fi_node_seen_c = fi_node_seen.clone();
        let forward = move |sc: &PlanCtx| -> Result<()> {
                // FI 面持有处(活得比 ctx 长:FiPrefillCtx 借它们)
                let fi_tensors: (TensorOps, TensorOps, TensorOps, TensorOps);
                let ids = sc.input("ids")?;
                let pos = sc.input("pos")?;
                let slots_in = sc.input("kv_slots")?;
                let lens_in = sc.input("kv_lens")?;
                let gdn_slot = sc.input("gdn_slot")?;
                let gdn_cu = sc.input("gdn_cu")?;
                let bt8_in = sc.input("bt8")?;
                // §三十一:v2 臂在(seqs=depth1)→ 页表 = bt8([8×nb]);
                // 回退(无 scratch)→ 单表 bt_leaf(split/chunked 语义不变)
                let kv_bt = if verify_attn_v2.is_some() { bt8_in.clone() } else { bt_leaf.clone() };
                let kvs_step: Vec<KvBuffers> = kv_caches
                    .iter()
                    .map(|(k, v)| KvBuffers {
                        k_cache: k.clone(),
                        v_cache: v.clone(),
                        slots: slots_in.clone(),
                        kv_lens: lens_in.clone(),
                        block_tables: kv_bt.clone(),
                    })
                    .collect();
                let gdns_step: Vec<GdnBuffers> = gdn_caches
                    .iter()
                    .map(|g| GdnBuffers {
                        conv_q: g.conv_q.clone(),
                        conv_k: g.conv_k.clone(),
                        conv_v: g.conv_v.clone(),
                        rec: g.rec.clone(),
                        slots: gdn_slot.clone(),
                    })
                    .collect();
                let mut ctx = ForwardCtx::model_prefill(
                    depth1, &pos, &kvs_step, &rp_v, &gdns_step, &slots_in, &lens_in,
                    &gdn_slot,
                    // 捕获图:槽号是回放期输入(捕获期无真值)—— T=depth1
                    // ≤ 16 < 64,chunked/scalar 臂结构上不可能启用,故
                    // None 安全;若将来在此捕 ≥64 图,层侧会拒发(poisoned)
                    // 而不是静默写槽 0(显式依赖律)
                    None,
                    0, None,
                );
                // **FI 图内臂**(四件套槽 + `ctx_total` 烘上界;表由引擎窗外刷):
                // 门 = 两件套(`verify_fi` 意图 + `fi_plan_bake` 上界)都在场。
                if let (true, Some(bake)) = (fi_want_c, vfi_bake_c) {
                    // 槽叶按块叶取 **U32 视图**(准入按声明 dtype 校验输入:
                    // FI 表是 u32;值由引擎逐轮以位型写过线)
                    let blk_id = |t: &TensorOps| -> Result<u64> {
                        match t.op() {
                            owl_models::ops::Op::Block { id } => Ok(*id),
                            _ => Err(owl_iface::contract::ModelError::Msg(
                                "FI 表槽非块叶(内部契约违例)".into(),
                            )),
                        }
                    };
                    let (q_cu_t, idx_t, ind_t, ll_t) = (
                        TensorOps::of_block(blk_id(&sc.input("fi_q_cu")?)?, Dtype::U32, vec![2]),
                        TensorOps::of_block(blk_id(&sc.input("fi_indices")?)?, Dtype::U32, vec![fi_nb]),
                        TensorOps::of_block(blk_id(&sc.input("fi_indptr")?)?, Dtype::U32, vec![2]),
                        TensorOps::of_block(blk_id(&sc.input("fi_last_len")?)?, Dtype::U32, vec![1]),
                    );
                    fi_tensors = (q_cu_t, idx_t, ind_t, ll_t);
                    ctx.fi = Some(FiPrefillCtx {
                        kcs: &fi_kcv.0,
                        vcs: &fi_kcv.1,
                        q_cu: &fi_tensors.0,
                        indices: &fi_tensors.1,
                        indptr: &fi_tensors.2,
                        last_len: &fi_tensors.3,
                    });
                    ctx.fi_plan_ctx = Some(bake);
                    fi_node_seen_c.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                ctx.seq_cu = Some(&gdn_cu);
                ctx.env = env;
                // §三十一:verify v2 臂(8 伪序列 partition 并行;scratch
                // = depth1 倍;bt8 = 8× 页链平铺,engine 每轮 host 写入)
                ctx.attn_v2 = verify_attn_v2.clone();
                ctx.v2_bt8 = Some(&bt8_in);
                // XQA verify 臂(2026-10-13 D3):面 = pool 常量件 + 图输入 seq
                if let Some(x) = xqa_pool_ctx.as_ref() {
                    let seq_in = sc.input("xqa_seq")?;
                    let pt_in = sc.input("xqa_pt")?;
                    let mk_in = sc.input("xqa_mask")?;
                    let sem_in = sc.input("xqa_sem")?;
                    ctx.xqa = Some(owl_models::module::XqaVerifyCtx {
                        page_table: pt_in,
                        seq_lens: seq_in,
                        mask: mk_in,
                        semaphores: sem_in,
                        scratch: x.scratch.clone(),
                        q_seq_len: depth1,
                        max_seq_len: x.max_seq_len,
                        sm_count: x.sm_count,
                    });
                }
                let tap: std::rc::Rc<std::cell::RefCell<Vec<TensorOps>>> =
                    std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
                ctx.gdn_tap = Some(tap.clone());
                let vocab = model_v.logits_vocab();
                let (hidden, taps) = if dflash_tap_decl {
                    let (h, t) = model_v.tapped_hidden(&ids, &ctx, tap_ids.as_slice());
                    if dflash_taps {
                        for (i, t) in t.iter().enumerate() {
                            sc.output(format!("tap{i}"), t)?;
                        }
                    }
                    (h, t)
                } else {
                    let h = model_v.last_hidden(&ids, &ctx);
                    (h.clone(), vec![h])
                };
                let _ = &taps;
                let logits = model_v.embed.lm_head_matmul(&hidden, &ctx.env.gguf);
                // A11(2026-10-14):per-row logits 出槽 —— 投机轮的
                // **采样接受**需要各行目标分布(原先只出 argmax ⇒ spec 臂
                // 完全不采样,请求温度/top_p/seed 失效 = 事实贪心;
                // A7 工具遵循度塌陷 + A8 不可复现的根因,见
                // docs/arch/agent-服务面-设计与施工.md §7.11)。
                // 仅采样档回读(贪心档零额外 D2H)。
                sc.output("logits", &logits)?;
                let rows: Vec<TensorOps> = (0..depth1)
                    .map(|r| owl_models::ops::argmax_f32idx(&logits, vocab, r * vocab))
                    .collect();
                let mut root = TensorOps::call(owl_models::ops::SemanticKernel::Concat);
                for i in 0..owl_models::kernel::CONCAT_ROWS_SLOTS {
                    root = root.arg(rows.get(i).unwrap_or(&rows[0]));
                }
                let root = root
                    .arg_usize(depth1)
                    .arg_usize(1)
                    .arg_usize(1)
                    .with_shape(Dtype::F32, vec![depth1, 1]);
                sc.output("tok", &root)?;
                // **分布式 argmax(2026-10-26)**:每行同时出 (max 值, 索引)
                // ⇒ TP 词表切时两 rank 只需交换 8B/行(而非 4MB 半表 logits)。
                // 只在贪心档消费(`spec.rs`);采样档仍读 `logits`。
                if head_split {
                    let tokv_rows: Vec<TensorOps> = (0..depth1)
                        .map(|r| owl_models::ops::argmax_validx(&logits, vocab, r * vocab))
                        .collect();
                    let mut troot = TensorOps::call(owl_models::ops::SemanticKernel::Concat);
                    for i in 0..owl_models::kernel::CONCAT_ROWS_SLOTS {
                        troot = troot.arg(tokv_rows.get(i).unwrap_or(&tokv_rows[0]));
                    }
                    let troot = troot
                        .arg_usize(depth1)
                        .arg_usize(1)
                        .arg_usize(2)
                        .with_shape(Dtype::F32, vec![depth1, 2]);
                    sc.output("tokv", &troot)?;
                }
                if cand_on {
                    let hist = sc.input("rep_hist")?;
                    let rep = sc.input("rep_penalty")?;
                    let cand = owl_models::ops::topk32_of_logits(
                        &logits,
                        &hist,
                        &rep,
                        depth1,
                        owl_models::layers::dflash2::TOPK_NPARTS,
                        vocab_offset,
                    );
                    sc.output("cand", &cand)?;
                }
                sc.output("hid", &hidden)?;
                for (i, t) in tap.borrow().iter().enumerate() {
                    sc.output(format!("r{i}"), t)?;
                }
                Ok(())
        };
        let (vg, voutcome) = GraphPlan::plan(vface, env, tp, desc, forward).await?;
        owl_shared::log_info!(
            "boot",
            "verify 面事实:FI 节点登记={} | bake={:?} | 段=0 断点=0 | 臂事实(attn.arm.verify.*):{}",
            fi_node_seen.load(std::sync::atomic::Ordering::Relaxed),
            vfi_bake,
            arm_facts("attn.arm.verify.")
        );
        match voutcome {
            PlanOutcome::Captured => {
                owl_shared::log_info!("boot", "verify 桶形图捕获(T={depth1}) +{:.2}s", t0.elapsed().as_secs_f32());
                Ok((Some(vg), fi_node_seen.load(std::sync::atomic::Ordering::Relaxed)))
            }
            other => {
                owl_shared::log_info!("boot", "verify 图降级({other:?})→ eager verify+重放路");
                Ok((None, false))
            }
        }
    }

    /// ③a fold 桶形图族(E5-M5;index = m,状态叶子 = **池 GDN 态**,
    /// `slots` 输入 = 会话行号 —— 与 eager 折叠**同形**(2026-10-19 §9.1(6)a):
    /// 旧形是"就地在 spec 快照缓冲里重放,再 restore 回池",快照缓冲被写成
    /// state@fed ⇒ 轮内**只能折一次**。改写池后:缓冲在轮内保持 pristine
    /// (= state@pos)⇒ 轮内可**多次**回卷(state@b 拍边界快照 → 再回卷
    /// state@fed),而每轮拷贝次数不增(restore 从尾部调到首部)。
    /// 记录 = verify 图输出块前缀视图,指针捕获期烘焙)。96 发射的
    /// eager fold → 单 graph_launch。
    async fn plan_fold_family(
        vg: &GraphPlan<D>,
        extra_faces: &mut Vec<D>,
        boot: &FleetBoot<'_>,
        t0: T,
        depth: usize,
    ) -> Result<Vec<GraphPlan<D>>> {
        let pool = boot.pool;
        let dims = boot.dims;
        let env = boot.env;
        let tp = boot.tp.clone();
        let knobs = boot.knobs;
        let mut fold_graphs = Vec::new();
        let n_rec = pool.gdn_count() * owl_shared::config::SPEC_FOLD_RECORDS_PER_LAYER;
        let rec_blks: Vec<Bytes> = (0..n_rec)
            .map(|i| {
                vg.output_block(&format!("r{i}")).ok_or_else(|| ModelError::EngineInvariant {
                    what: format!("fold 记录槽 r{i} 缺席(图装配与声明不一致)"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let n_roots = pool.gdn_count() * 4;
        // **单图覆盖全部 rows(2026-10-22)**:fold 的 m 只影响 record/输出视图形状
        // 与网格 tile 数(≤32 行恒 1 tile);conv/recurrence 核按设备侧 `cu=[0,rows]`
        // 截断 ⇒ 按**最大 rows + 1** 捕一张,回放时写 `cu` 即可。
        // 7 张桶图实测各 ~66 MiB 驱动侧 exec 内存,单图直接省 ~400 MiB。
        //
        // ⚠️ **必须多捕一行(2026-10-22 实测)**:按 `m1 = depth` 捕获时
        // rows=depth 的折叠状态错(65k AL 2.18 vs 2.96);`m1 = depth+1` 后与
        // 逐 m 桶图/eager 逐位一致(AL 2.955)。verify 记录块本身有 depth+1 行,
        // 多捕一行不越界。动这个数之前先跑 65k AL 恒等门。
        let mmax = depth;
        let m1 = mmax + 1;
        assert!(
            m1 <= owl_shared::config::SPEC_VERIFY_ROWS_MAX,
            "fold 单图: m1={m1} > SPEC_VERIFY_ROWS_MAX={}",
            owl_shared::config::SPEC_VERIFY_ROWS_MAX
        );
        for m in [mmax] {
            let m1 = m + 1;
            let model_f = Arc::clone(&boot.model);
            let gdn_caches = pool.gdn_caches();
            let rec_blks = rec_blks.clone();
            let scalar = env.gdn.scalar;
            let d = dims;
            let mut fouts = Vec::with_capacity(n_roots);
            for i in 0..n_roots {
                let shape = match i % 4 {
                    0 => vec![m1, d.nk * d.hk],
                    1 => vec![m1, d.nk * d.hk],
                    2 => vec![m1, d.nv * d.hv],
                    _ => vec![m1, d.nv, d.hv],
                };
                fouts.push(OutputSlot { name: format!("f{i}"), shape, dtype: Dtype::F16 });
            }
            let Some(fface) = extra_faces.pop() else {
                owl_shared::log_info!("boot", "fold 图 face 耗尽 → eager fold");
                break;
            };
            let (fg, fout) = GraphPlan::plan(
                fface,
                env,
                tp.clone(),
                GraphPlanDesc {
                    inputs: vec![
                        InputSlot::f32("slots", 1).init(vec![0.0]),
                        InputSlot::f32("cu", 2).init(vec![0.0, m1 as f32]),
                    ],
                    outputs: fouts,
                    capture: !knobs.no_graph,
                    slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                    debug_probe: knobs.probes.debug,
                    // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
                    // 锚定活性回收(2026-10-15 归因之修②):本族 plain 口径是**分配总和**
                    // (verify 图实测 133MiB 而真活跃 ~19MB)⇒ 开 scoped 归约,
                    // slab 容量 = 峰值活跃集。机制门 = graph_plan::gpu_capture_reclaim_matches_eager
                    // + 族恒等门(fold/propose/verify 各有对拍)。
                    reclaim_capture: true,
                    warmup: true,
                },
                move |sc: &PlanCtx| -> Result<()> {
                    let slots = sc.input("slots")?;
                    let cu = sc.input("cu")?;
                    let rec_views: Vec<TensorOps> = rec_blks
                        .iter()
                        .enumerate()
                        .map(|(i, b)| {
                            let shape = match i % owl_shared::config::SPEC_FOLD_RECORDS_PER_LAYER {
                                0 | 1 => vec![m1, d.nk * d.hk],
                                2 => vec![m1, d.nv * d.hv],
                                3 | 4 => vec![m1, d.nk, d.hk],
                                5 => vec![m1, d.nv, d.hv],
                                _ => vec![m1, d.nv],
                            };
                            TensorOps::of_block(b.id, d.dtype, shape)
                        })
                        .collect();
                    let gdns: Vec<GdnBuffers> = gdn_caches
                        .iter()
                        .map(|l| GdnBuffers {
                            conv_q: l.conv_q.clone(),
                            conv_k: l.conv_k.clone(),
                            conv_v: l.conv_v.clone(),
                            rec: l.rec.clone(),
                            slots: slots.clone(),
                        })
                        .collect();
                    let roots = owl_models::spec::fold_tree(
                        &model_f, &rec_views, &gdns, &slots, &cu, 0, m, scalar,
                    )?;
                    for (i, r) in roots.iter().enumerate() {
                        sc.output(format!("f{i}"), r)?;
                    }
                    Ok(())
                },
            )
            .await?;
            match fout {
                PlanOutcome::Captured => {
                    owl_shared::log_info!(
                        "boot",
                        "fold 单图捕获(rows≤{};m={m}) +{:.2}s",
                        depth,
                        t0.elapsed().as_secs_f32()
                    );
                    fold_graphs.push(fg);
                }
                other => owl_shared::log_info!("boot", "fold 图降级(m={m},{other:?})→ eager fold"),
            }
        }
        Ok(fold_graphs)
    }

    /// ③b DFlash2 propose **单图**(E5-DF4;encode 全 W=K+1 行 + 噪声块 +
    /// selector 运行时定形;多编行下轮被自块写/重编覆盖,幂等安全)。
    /// 输入槽 = tokens/anchor/enc_pos/enc_slots/prop_pos/prop_slots/
    /// kv_len;烘焙叶 = verify taps 槽 + 草稿池叶。prefix_len 曾为
    /// 宿主烘焙标量,已改核内读 kv_len 张量(契约 5)。
    async fn plan_dflash(
        vg: &GraphPlan<D>,
        extra_faces: &mut Vec<D>,
        boot: &FleetBoot<'_>,
        t0: T,
    ) -> Result<Option<GraphPlan<D>>> {
        let Some(Drafter::DFlash2(d2)) = boot.spec.drafter.as_ref() else {
            return Ok(None);
        };
        let pool = boot.pool;
        // **草稿环境**(非目标环境):草稿的 `attn.naive_nc` 等臂按草稿几何
        // (hd=128 / GQA 比)解析 —— 捕获期就固化在算子拾取里;用目标 env
        // (hd=256)会几何门拒(F16 路)或静默错几何(BF16 旧变体门过宽)。
        let env = boot.spec.draft_env;
        let tp = boot.tp.clone();
        let knobs = boot.knobs;
        let tap_blks: Vec<Bytes> = (0..5)
            .map(|i| {
                vg.output_block(&format!("tap{i}")).ok_or_else(|| ModelError::EngineInvariant {
                    what: format!("tap{i} 输出槽缺席(草稿档位不一致)"),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let kv_leaves = pool.dflash_kv_leaves().ok_or_else(|| ModelError::EngineInvariant { what: "dflash KV 池句柄缺席(档位/池几何不一致)".into() })?;
        let bt_leaf = pool.bt_leaf_flat();
        let model_df = Arc::clone(&boot.model);
        let rp_df = boot.spec.draft_rope.as_ref().ok_or_else(|| ModelError::EngineInvariant { what: "draft rope 缺席(草稿档位不一致)".into() })?.clone();
        let d2g = Arc::clone(d2);
        let block = d2.block();
        let depth = d2.depth();
        // **切档形态**（P1.1 × P2.1）:propose 图只出"半表 top-16 候选 + proj" ——
        // host 合并两 rank 候选后由 rank0 跑 select（全表域）。判据 = 事实
        // （本 rank logits 宽 < 全量），与引擎侧 `head_split_active` 同律。
        let head_split = boot.model.logits_vocab() != boot.model.vocab_size();
        let sel_rank = d2g.selector_rank();
        let hidd = d2.hidden();
        let Some(gface) = extra_faces.pop() else {
            owl_shared::log_info!("boot", "dflash propose 图 face 耗尽 → eager propose");
            return Ok(None);
        };
        let (dg, dout) = GraphPlan::plan(
            gface,
            env,
            tp,
            GraphPlanDesc {
                inputs: vec![
                    InputSlot::f32("tokens", block),
                    InputSlot::f32("anchor", 1),
                    InputSlot::f32("enc_pos", block),
                    InputSlot::f32("enc_slots", block),
                    InputSlot::f32("enc_kv_lens", block),
                    InputSlot::f32("prop_pos", block),
                    InputSlot::f32("prop_slots", block),
                    InputSlot::f32("prop_kv_lens", block),
                    InputSlot::f32("kv_len", 1),
                ],
                outputs: {
                    let mut o = if head_split {
                        vec![OutputSlot::f32("topk", &[2 * depth * owl_models::layers::dflash2::TOP_K])]
                    } else {
                        // 单缓冲：drafts[0..depth] ‖ scores[depth..]；topk 另块。
                        vec![
                            OutputSlot::f32("sel", &[depth * (16 * 16 + 1)]),
                            OutputSlot::f32("topk_all", &[2 * depth * owl_models::layers::dflash2::TOP_K]),
                        ]
                    };
                    if head_split {
                        o.push(OutputSlot {
                            name: "sel_proj".into(),
                            shape: vec![depth, sel_rank],
                            dtype: d2g.dtype(),
                        });
                    }
                    o.push(OutputSlot { name: "e0".into(), shape: vec![1], dtype: Dtype::BF16 });
                    o.push(OutputSlot { name: "e1".into(), shape: vec![1], dtype: Dtype::BF16 });
                    o.push(OutputSlot { name: "e2".into(), shape: vec![1], dtype: Dtype::BF16 });
                    o.push(OutputSlot { name: "e3".into(), shape: vec![1], dtype: Dtype::BF16 });
                    o.push(OutputSlot { name: "e4".into(), shape: vec![1], dtype: Dtype::BF16 });
                    o
                },
                capture: !knobs.no_graph,
                slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                debug_probe: knobs.probes.debug,
                // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
                // 锚定活性回收(2026-10-15 归因之修②;理由同族桶形图):
                // plain = 分配总和(主会话 13.6MiB / 该图实测总量级更高)⇒
                // 开 scoped,容量降到峰值活跃集;恒等门同族。
                reclaim_capture: true,
                warmup: true,
            },
            move |sc: &PlanCtx| -> Result<()> {
                let taps: Vec<TensorOps> = tap_blks
                    .iter()
                    .map(|b| TensorOps::of_block(b.id, Dtype::F16, vec![block, hidd]))
                    .collect();
                let tokens = sc.input("tokens")?;
                let anchor = sc.input("anchor")?;
                let enc_pos = sc.input("enc_pos")?;
                let enc_slots = sc.input("enc_slots")?;
                // 逐行可见长(含自身;eager dflash_encode 同式
                // pos+i+1 —— 原绑 enc_pos 缺自身 = AL 劣化根因)
                let enc_kv_lens = sc.input("enc_kv_lens")?;
                let prop_pos = sc.input("prop_pos")?;
                let prop_slots = sc.input("prop_slots")?;
                // propose 噪声块逐行可见长(fp+1+i;原绑标量
                // kv_len,1 值槽供 W 行读 = 越界垃圾)
                let prop_kv_lens = sc.input("prop_kv_lens")?;
                let kv_len = sc.input("kv_len")?;
                let bt = bt_leaf.clone();
                let enc_kvs: Vec<KvBuffers> = kv_leaves
                    .iter()
                    .map(|(k, v)| KvBuffers {
                        k_cache: k.clone(),
                        v_cache: v.clone(),
                        slots: enc_slots.clone(),
                        kv_lens: enc_kv_lens.clone(),
                        block_tables: bt.clone(),
                    })
                    .collect();
                let prop_kvs: Vec<KvBuffers> = kv_leaves
                    .iter()
                    .map(|(k, v)| KvBuffers {
                        k_cache: k.clone(),
                        v_cache: v.clone(),
                        slots: prop_slots.clone(),
                        kv_lens: prop_kv_lens.clone(),
                        block_tables: bt.clone(),
                    })
                    .collect();
                let mut ctx = ForwardCtx::minimal(block);
                // 草稿 ctx 的环境与图环境同源(层侧读 ctx.env 时零分歧)
                ctx.env = env;
                let mem = d2g.project_memory(&taps, &ctx);
                let enc_roots = d2g.encode_kv(&mem, &enc_pos, &enc_kvs, &rp_df, &ctx);
                for (i, r) in enc_roots.iter().enumerate() {
                    sc.output(format!("e{i}"), r)?;
                }
                // drafts = select 单缓冲前 7 元(offset 0;输出槽
                // 形状 = 整缓冲 —— dtoh 整父块契约)
                if head_split {
                    // **切档形态**:只出候选（整 topk 缓冲）+ proj
                    // （select 由 host 合并后跑）
                    let (_h, _cand, vals, proj, _l) = d2g.propose_block_split(
                        &tokens, &prop_pos, &prop_kvs, &rp_df, &model_df.embed, &kv_len, &ctx, None,
                    );
                    // vals 是 topk 单缓冲的**前段视图** ⇒ 输出整父块
                    // （值区 ‖ 索引区;host 按同序切片）
                    sc.output("topk", &vals)?;
                    sc.output("sel_proj", &proj)?;
                    Ok(())
                } else {
                    let (_h, _drafts, scores, _l, _cand, topk) = d2g.propose_block(
                        &tokens, &prop_pos, &anchor, &prop_kvs, &rp_df,
                        &model_df.embed, &kv_len, &ctx, None,
                    );
                    // `scores` 是 select 单缓冲的尾部视图；声明槽按整父块
                    // 宽度，host 从 sel[0..depth] 取 drafts、其后取 scores。
                    sc.output("sel", &scores)?;
                    sc.output("topk_all", &topk)
                }
            },
        )
        .await?;
        match dout {
            PlanOutcome::Captured => {
                owl_shared::log_info!("boot", "dflash propose 图捕获(encode+propose 单图,W={block} 行) +{:.2}s", t0.elapsed().as_secs_f32());
                Ok(Some(dg))
            }
            other => {
                owl_shared::log_info!("boot", "dflash propose 图降级({other:?})→ eager propose");
                Ok(None)
            }
        }
    }

    /// ③c **切档 propose 收割图**(2026-10-26 propose 刀3)。
    ///
    /// 切档(`head_split`)时 propose 图只出「半表 top-16 候选 + sel_proj」,
    /// 收割要两步:① 两半候选合并成全表 top-K;② rank0 跑 selector。
    /// 原实现两步都在 host(eager):read0/read1 两次 dtoh + from_host 回写 +
    /// `eval_ops_scoped_env` 一次发射 ⇒ 每轮 ~0.8ms 的主机往返与同步。
    ///
    /// 本图把两步**整体下沉**:输入 = peer 的 topk 单缓冲(唯一需要主机同步
    /// 的一件,因为两 rank 的图是并发发射的,不能互相读对方本轮产物)+
    /// anchor;输出 = select 单缓冲(前 `rows` 元 = drafts)。
    /// 本 rank 的 `topk`/`sel_proj` 直接吃 dflash 图的**输出块**(刀2 已验证:
    /// 图输出块指针在捕获态恒定,可作 `Op::Block` 叶子被下游图消费)。
    ///
    /// 只建在 rank0(selector 的全表域码本/合并结果只在 rank0 用),rank1 不建。
    async fn plan_dflash_select(
        dg: &GraphPlan<D>,
        extra_faces: &mut Vec<D>,
        boot: &FleetBoot<'_>,
        t0: T,
    ) -> Result<Option<GraphPlan<D>>> {
        let Some(Drafter::DFlash2(d2)) = boot.spec.drafter.as_ref() else {
            return Ok(None);
        };
        let rows = d2.depth();
        let k = owl_models::layers::dflash2::TOP_K;
        let rank = d2.selector_rank();
        let half = boot.model.logits_vocab();
        let env = boot.spec.draft_env;
        let tp = boot.tp.clone();
        let knobs = boot.knobs;
        let topk_blk = dg.output_block("topk").ok_or_else(|| ModelError::EngineInvariant {
            what: "select 图:propose 图输出槽 topk 缺席(切档声明不一致)".into(),
        })?;
        let proj_blk = dg.output_block("sel_proj").ok_or_else(|| ModelError::EngineInvariant {
            what: "select 图:propose 图输出槽 sel_proj 缺席(切档声明不一致)".into(),
        })?;
        if topk_blk.len != 2 * rows * k {
            return Err(ModelError::EngineInvariant {
                what: format!(
                    "select 图:topk 槽长 {} ≠ 2·{}·{}",
                    topk_blk.len,
                    rows,
                    k
                ),
            });
        }
        let Some(sface) = extra_faces.pop() else {
            owl_shared::log_warn!("boot", "select 图 face 耗尽 → 回落 host 合并 + eager select");
            return Ok(None);
        };
        let d2s = Arc::clone(d2);
        let (sg, sout) = GraphPlan::plan(
            sface,
            env,
            tp,
            GraphPlanDesc {
                inputs: vec![
                    // peer 的 topk 单缓冲(值区 ‖ 索引区;半表域 id)
                    InputSlot::f32("peercand", 2 * rows * k),
                    InputSlot::f32("anchor", 1),
                ],
                outputs: vec![OutputSlot {
                    name: "sel".into(),
                    shape: vec![rows * (k * k + 1)],
                    dtype: Dtype::F32,
                }],
                capture: !knobs.no_graph,
                slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                debug_probe: knobs.probes.debug,
                reclaim_capture: true,
                warmup: true,
            },
            move |sc: &PlanCtx| -> Result<()> {
                // 本 rank 的候选/proj = propose 图的输出块(只读叶子)
                let own = TensorOps::of_block(topk_blk.id, Dtype::F32, vec![2 * rows * k]);
                let peer = sc.input("peercand")?.reshape(vec![2 * rows * k]);
                let anchor = sc.input("anchor")?;
                let own_v = own.slice_view(0, vec![rows, k]);
                let own_i = own.slice_view(rows * k, vec![rows, k]);
                let peer_v = peer.slice_view(0, vec![rows, k]);
                let peer_i = peer.slice_view(rows * k, vec![rows, k]);
                // 两半合并(全序律与 host `merge_topk_two_halves` 逐位同;
                // 单测 = crates/kernels/tests/topk_merge_halves.rs)
                let merged = TensorOps::call(owl_models::ops::SemanticKernel::TopkMergeHalves)
                    .aux(&[rows])
                    .arg(&own_v)
                    .arg(&own_i)
                    .arg(&peer_v)
                    .arg(&peer_i)
                    .arg_i32(half as i32)
                    .arg_i32(rows as i32)
                    .arg_i32(k as i32)
                    .with_shape(Dtype::F32, vec![2 * rows * k]);
                let cand = merged.slice_view(rows * k, vec![rows, k]);
                let vals = merged.slice_view(0, vec![rows, k]);
                let proj = TensorOps::of_block(proj_blk.id, proj_blk.dtype, vec![rows, rank]);
                let (drafts, scores) =
                    d2s.selector().select_from_candidates(&cand, &vals, &proj, &anchor, rows);
                // drafts 是 select 单缓冲的前缀视图 ⇒ 输出整父块
                // (host 从 sel[0..rows] 取 drafts,与未切档档同约定)
                let _ = drafts;
                sc.output("sel", &scores)?;
                Ok(())
            },
        )
        .await?;
        match sout {
            PlanOutcome::Captured => {
                owl_shared::log_info!(
                    "boot",
                    "select 图捕获(两半候选合并 + selector 单图;rows={rows} k={k} rank={rank}) +{:.2}s",
                    t0.elapsed().as_secs_f32()
                );
                Ok(Some(sg))
            }
            other => {
                owl_shared::log_warn!(
                    "boot",
                    "select 图降级({other:?})→ 回落 host 合并 + eager select"
                );
                Ok(None)
            }
        }
    }

    /// ③c MTP propose 桶形图族(E5-M5;index = m ∈ 0..=depth —— 全接受
    /// m=depth 也要出下轮草稿;extend m+1 行 + k-1 链步;
    /// hidden = verify hid 槽前缀视图;lm_head/链全部入图)。
    async fn plan_propose_family(
        vg: &GraphPlan<D>,
        extra_faces: &mut Vec<D>,
        boot: &FleetBoot<'_>,
        _t0: T,
        depth: usize,
    ) -> Result<Vec<GraphPlan<D>>> {
        let Some(Drafter::Mtp(mtp)) = boot.spec.drafter.as_ref() else {
            return Ok(Vec::new());
        };
        let pool = boot.pool;
        let dims = boot.dims;
        // MTP 草稿层/链步池是 F16(StatePool 按目标 dims.dtype 分配),
        // 不能用主 env(fp8 生产档会选到 fp8 读核)。
        let env = boot.spec.draft_env;
        let tp = boot.tp.clone();
        let knobs = boot.knobs;
        let spec_depth = depth;
        let mut propose_graphs = Vec::new();
        let hid_blk = vg.output_block("hid").ok_or_else(|| ModelError::EngineInvariant { what: "verify 图输出槽 hid 缺席(图装配与声明不一致)".into() })?;
        // 链步 decode 读核 scratch(seqs=1;与 draft.rs eager 路同源):
        // kNHD 池落 v1 = 布局错位 → 草稿垃圾(2026-10-13 守卫捕获)
        let attn_v2_pool = pool.attn_v2_scratch();
        let extend_v2_pool = pool.verify_attn_v2_scratch();
        for m in 0..=spec_depth {
            // move 闭包逐个消耗 → 每轮取一份克隆
            let attn_v2_p = attn_v2_pool.clone();
            let extend_v2_p = extend_v2_pool.as_ref().and_then(|s| s.prefix(m + 1));
            let m1 = m + 1;
            let model_p = Arc::clone(&boot.model);
            let mtp = Arc::clone(mtp);
            let rp_p = boot.rope.clone();
            let kv_leaf = pool.mtp_kv_leaf().ok_or_else(|| ModelError::EngineInvariant { what: "MTP KV 池句柄缺席(draft 档位不一致)".into() })?;
            let bt_leaf = pool.bt_mtp_leaf().ok_or_else(|| ModelError::EngineInvariant { what: "MTP 块表槽缺席(草稿档位不一致)".into() })?;
            let bt_nb = *bt_leaf.shape().last().unwrap_or(&1);
            let hid = hid_blk.clone();
            let env_p = env;
            let vocab = boot.model.vocab_size();
            let mut pins = vec![InputSlot::f32("tok_ext", m1), InputSlot::f32("pos_ext", m1), InputSlot::f32("slots_ext", m1), InputSlot::f32("lens_ext", m1)];
            // MTP extend 的 v2 伪序列页表(每行同一条 MTP 页链)
            pins.push(InputSlot::f32("v2_bt8", m1 * bt_nb));
            for j in 0..spec_depth - 1 {
                pins.push(InputSlot::f32(format!("c{j}_pos"), 1));
                pins.push(InputSlot::f32(format!("c{j}_slots"), 1));
                pins.push(InputSlot::f32(format!("c{j}_lens"), 1));
                // Windowed-MTP：chain 每步自己的读表（window=0 时填完整链）
                pins.push(InputSlot::f32(format!("c{j}_bt"), bt_nb));
            }
            pins.push(InputSlot::f32("seq_cu", 2).init(vec![0.0, m1 as f32]));
            let Some(pface) = extra_faces.pop() else {
                return Err(owl_iface::contract::ModelError::Msg(format!(
                    "propose 图 face 耗尽(m={m}/depth={spec_depth}):检查 fleet_faces(depths) 预算（每档 fold+propose + verify）"
                )));
            };
            let (pg, pout) = GraphPlan::plan(
                pface,
                env,
                tp.clone(),
                GraphPlanDesc {
                    inputs: pins,
                    outputs: vec![OutputSlot { name: "drafts".into(), shape: vec![spec_depth, 1], dtype: Dtype::F32 }],
                    capture: !knobs.no_graph,
                    slab_fixed: knobs.cuda.capture_slab_mb.is_some(),
                    debug_probe: knobs.probes.debug,
                    // 本族不加锚定回收(plain 口径:窗内块常驻 —— 历史行为零变化)
                    // 锚定活性回收(2026-10-15 归因之修②):本族 plain 口径是**分配总和**
                    // (verify 图实测 133MiB 而真活跃 ~19MB)⇒ 开 scoped 归约,
                    // slab 容量 = 峰值活跃集。机制门 = graph_plan::gpu_capture_reclaim_matches_eager
                    // + 族恒等门(fold/propose/verify 各有对拍)。
                    reclaim_capture: true,
                    warmup: true,
                },
                move |sc: &PlanCtx| -> Result<()> {
                    // PlanCtx 输入声明形 [1, len] → flatten(行主前缀)
                    let tok_ext = sc.input("tok_ext")?.reshape(vec![m1]);
                    let pos_ext = sc.input("pos_ext")?.reshape(vec![m1]);
                    let slots_ext = sc.input("slots_ext")?.reshape(vec![m1]);
                    let lens_ext = sc.input("lens_ext")?.reshape(vec![m1]);
                    let seq_cu = sc.input("seq_cu")?;
                    let v2_bt8 = sc.input("v2_bt8")?.reshape(vec![m1, bt_nb]);
                    let bt = bt_leaf.clone();
                    let kv_ext = KvBuffers {
                        k_cache: kv_leaf.0.clone(),
                        v_cache: kv_leaf.1.clone(),
                        slots: slots_ext.clone(),
                        kv_lens: lens_ext.clone(),
                        block_tables: bt.clone(),
                    };
                    let hid_view = TensorOps::of_block(hid.id, dims.dtype, vec![m1, dims.hidden]);
                    let mut chain: Vec<(TensorOps, KvBuffers)> = Vec::new();
                    for j in 0..spec_depth - 1 {
                        let pos_c = sc.input(&format!("c{j}_pos"))?;
                        let slots_c = sc.input(&format!("c{j}_slots"))?;
                        let lens_c = sc.input(&format!("c{j}_lens"))?;
                        let bt_c = sc.input(&format!("c{j}_bt"))?;
                        chain.push((
                            pos_c,
                            KvBuffers {
                                k_cache: kv_leaf.0.clone(),
                                v_cache: kv_leaf.1.clone(),
                                slots: slots_c.clone(),
                                kv_lens: lens_c.clone(),
                                block_tables: bt_c,
                            },
                        ));
                    }
                    let chain_refs: Vec<(&TensorOps, &KvBuffers)> =
                        chain.iter().map(|(p, k)| (p, k)).collect();
                    let droot = mtp.propose_ext(
                        &tok_ext,
                        &hid_view,
                        &pos_ext,
                        &kv_ext,
                        &slots_ext,
                        &lens_ext,
                        &seq_cu,
                        &chain_refs,
                        &model_p.embed,
                        &rp_p,
                        vocab,
                        env_p,
                        // 图步链读核也走 v2(同 draft.rs:kNHD 池落 v1 = 错位)
                        attn_v2_p.clone(),
                        extend_v2_p.clone(),
                        Some(&v2_bt8),
                    );
                    sc.output("drafts", &droot)
                },
            )
            .await?;
            match pout {
                PlanOutcome::Captured => {
                    owl_shared::log_info!("boot", "propose 桶形图捕获(m={m})");
                    propose_graphs.push(pg);
                }
                other => owl_shared::log_info!("boot", "propose 图降级(m={m},{other:?})→ eager propose"),
            }
        }
        Ok(propose_graphs)
    }
}

