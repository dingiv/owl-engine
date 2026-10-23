//! **选择层运行态**(B 轴;2026-10-21 S1 收户)。
//!
//! 为什么独立成户:视图 / 覆盖率探针 / 驻留调和账原本摊在 `RunningEngine`
//! 上(9 个字段,写者分布在 coverage / phases / turn_end / scheduler 四个
//! 文件)—— 状态没有守门人,任何文件都能逐字段 poke。本户把「字段私有 +
//! 语义方法」收口:外部只表达**意图**(`note_view_step` / `note_stage_in` /
//! `take_cov_pending`),不直接摸字段。
//!
//! 纪律:
//! - 字段一律私有;读 = `view()` / `stage()` / `prof()`;写 = `note_*` / `set_*`;
//! - 方法自肃:只改本户字段,不碰 bank / dev / store(跨域编排留在引擎方法里);
//! - 报告面(`line()`)与数据同屋,随类型一起迁入。

use owl_store::ChainHash;

use crate::exec::coverage::{CoveragePending, CoverageReport};

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
    /// 选中块的内容身份(与 `pages` 同序;2026-10-23 驻留校验用)。
    pub blocks: Vec<ChainHash>,
    /// 选中块的 token 数(= `pages.len() × page`)。
    pub tokens: usize,
    /// 召回点绝对位置(块边界;尾部从这里起)。
    pub recall_pos: usize,
}
/// **B 轴运行态**(视图 / 覆盖率探针 / 驻留调和账 / 相位账)。
///
/// 生命周期:跨 turn 存活,**视图**是 turn 域资源(turn 收口清空);
/// 其余是引擎级累计账(只在报告/调试面读)。
#[derive(Default)]
pub(crate) struct SelectRuntime {
    /// B-3d 选择层视图态(召回点产出;`select_decode` 档每步展开成视图页表)
    view: Option<SelectView>,
    /// 视图**生效**的 decode 步数(证据面:该臂真的走了视图)
    view_steps: usize,
    /// 视图输入槽**回退为全量链**的步数(视图态缺位;与上者和 = decode 步数)
    view_fallback: usize,
    /// **B 轴覆盖率探针**待分析面(Q 已回读;等块边界事件补齐索引账)
    cov_pending: Option<CoveragePending>,
    /// 覆盖率探针最近一次观测(测试/调试读数面;`None` = 未观测过)
    cov_last: Option<CoverageReport>,
    /// **上次完整召回的选择**(内容身份,时间序)。
    ///
    /// 用途:**廉价边界**的驻留调和用**冻结名单**,而不是重新打分 ——
    /// 也**不能是"什么都不做"**:`ensure_for_len` 驱逐不了会话自己持有的页,
    /// 链每 chunk 要新增 `chunk/page` 页,不腾就必然 `Exhausted`
    /// (实测:池 96 页,链要到 100 页 ⇒ 耗尽)。
    keep: Vec<ChainHash>,
    /// B-3e stage-in 实测账(召回回灌的块数/字节/耗时)
    stage: StageStats,
    /// B-3e-2 部分驻留账(下沉 / 回灌 / 不可沉)
    partial: PartialStats,
    /// B 轴相位账(纯计时,零语义)
    prof: BAxisProfile,
}

/// **stage-in 来源**(块数拆分;两者互斥记账)。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StageSrc {
    /// 召回前置(prefill 段边界)
    Prefill,
    /// 决策单(`verdict.in_`)
    Verdict,
}

impl SelectRuntime {
    // ── 视图(召回点产出;turn 收口清空)──────────────────────────────

    pub(crate) fn view(&self) -> Option<&SelectView> {
        self.view.as_ref()
    }

    pub(crate) fn has_view(&self) -> bool {
        self.view.is_some()
    }

    pub(crate) fn set_view(&mut self, view: SelectView) {
        self.view = Some(view);
    }

    /// 清视图(turn 收口:视图是 turn 域资源,留着 = 用旧视图读新会话)。
    pub(crate) fn clear_view(&mut self) {
        self.view = None;
    }

    pub(crate) fn note_view_step(&mut self) {
        self.view_steps += 1;
    }

    /// 视图缺位回退一步(占位符守卫路径;与 `note_view_step` 互斥)。
    pub(crate) fn note_view_fallback(&mut self) {
        self.view_fallback += 1;
    }

    /// 视图生效/回退步数(证据面;测试与 e2e 门读)。
    #[cfg(any(test, feature = "e2e"))]
    pub(crate) fn view_steps(&self) -> usize {
        self.view_steps
    }

    #[cfg(any(test, feature = "e2e"))]
    pub(crate) fn view_fallbacks(&self) -> usize {
        self.view_fallback
    }

    // ── 覆盖率探针 ─────────────────────────────────────────────────

    pub(crate) fn take_cov_pending(&mut self) -> Option<CoveragePending> {
        self.cov_pending.take()
    }

    pub(crate) fn set_cov_last(&mut self, report: CoverageReport) {
        self.cov_last = Some(report);
    }

    pub(crate) fn cov_last(&self) -> Option<&CoverageReport> {
        self.cov_last.as_ref()
    }

    /// **探针分析对**(prof, cov)一次借出。
    ///
    /// 为什么成对:拆成两次 `&mut self` 调用会在借用检查上打架
    /// (调用方要同时用两件);一次借出 = 两件互不重叠的不变量写在类型上。
    pub(crate) fn analysis_pair(
        &mut self,
    ) -> (&mut BAxisProfile, &mut Option<CoveragePending>) {
        (&mut self.prof, &mut self.cov_pending)
    }

    // ── 召回选中集(内容身份,时间序)────────────────────────────────

    pub(crate) fn keep(&self) -> &[ChainHash] {
        &self.keep
    }

    pub(crate) fn set_keep(&mut self, keep: Vec<ChainHash>) {
        self.keep = keep;
    }

    // ── stage-in 账 ─────────────────────────────────────────────────

    pub(crate) fn note_stage_in(
        &mut self,
        blocks: usize,
        bytes: usize,
        nanos: u64,
        src: StageSrc,
    ) {
        self.stage.blocks += blocks;
        self.stage.bytes += bytes;
        self.stage.nanos += nanos;
        match src {
            StageSrc::Prefill => self.stage.from_prefill += blocks,
            StageSrc::Verdict => self.stage.from_verdict += blocks,
        }
    }

    #[cfg(any(test, feature = "e2e"))]
    pub(crate) fn stage(&self) -> StageStats {
        self.stage
    }

    // ── 部分驻留账 ─────────────────────────────────────────────────

    pub(crate) fn note_prefill_subset(&mut self) {
        self.partial.prefill_subset += 1;
    }

    pub(crate) fn note_lazy_keep(&mut self, blocks: usize) {
        self.partial.lazy_keep += blocks;
    }

    pub(crate) fn note_not_cacheable(&mut self, blocks: usize) {
        self.partial.not_cacheable += blocks;
    }

    pub(crate) fn note_sunk(&mut self, blocks: usize) {
        self.partial.sunk += blocks;
    }

    pub(crate) fn note_staged(&mut self, blocks: usize) {
        self.partial.staged += blocks;
    }

    #[cfg(any(test, feature = "e2e"))]
    pub(crate) fn partial(&self) -> PartialStats {
        self.partial
    }

    // ── B 轴相位账(纯计时;名字 = 相位语义,不是字段名)──────────────

    pub(crate) fn note_boundary(&mut self) {
        self.prof.boundaries += 1;
    }

    /// 走过的块边界数(召回点周期判定用)。
    pub(crate) fn boundaries(&self) -> usize {
        self.prof.boundaries
    }

    /// 召回档:+1(**完整** = 打分 + 按分选择;廉价 = 近期选择)。
    pub(crate) fn note_recall(&mut self, full: bool) {
        if full {
            self.prof.recall_full += 1;
        } else {
            self.prof.recall_cheap += 1;
        }
    }

    pub(crate) fn note_index(&mut self, ms: f64) {
        self.prof.index_ms += ms;
    }

    pub(crate) fn note_q_face(&mut self, ms: f64) {
        self.prof.q_face_ms += ms;
    }

    pub(crate) fn note_q_derope(&mut self, ms: f64) {
        self.prof.q_derope_ms += ms;
    }

    pub(crate) fn note_q_mass(&mut self, ms: f64) {
        self.prof.q_mass_ms += ms;
    }

    pub(crate) fn note_mass(&mut self, ms: f64) {
        self.prof.mass_ms += ms;
    }

    pub(crate) fn note_score(&mut self, ms: f64) {
        self.prof.score_ms += ms;
    }

    pub(crate) fn note_paper(&mut self, ms: f64) {
        self.prof.paper_ms += ms;
    }

    pub(crate) fn note_select(&mut self, ms: f64) {
        self.prof.select_ms += ms;
    }

    pub(crate) fn note_view(&mut self, ms: f64) {
        self.prof.view_ms += ms;
    }

    pub(crate) fn note_reconcile(&mut self, ms: f64) {
        self.prof.reconcile_ms += ms;
    }

    /// 一个召回点的总相位(收口;与 `note_boundary` 成对)。
    pub(crate) fn note_total(&mut self, ms: f64) {
        self.prof.total_ms += ms;
    }

    /// B 轴相位账只读面(生产诊断也读；纯计时，零语义)。
    pub(crate) fn prof(&self) -> &BAxisProfile {
        &self.prof
    }

}
