//! **B 轴 KVMem 选择层:覆盖率探针**(Q-2;B-1 打分器的前置实证件)。
//!
//! 设计出处 = `docs/arch/kvmem-选择层-设计与施工.md` §8.6(施工单)/ §8.8(Q-1
//! 类型面)/ §8.4(约定门),论文 = KVMem(arXiv:2609.04852)。
//!
//! # 它回答什么问题
//!
//! **「索引打分能否复现真实注意力」** —— 唯一的实证判据(§8.6 门):
//!
//! ```text
//! 真实面:post-RoPE Q × 池内 post-RoPE K → 逐 (层,查询头) softmax → 逐块质量
//! 索引面:去 RoPE 的 q̃ × Mean-K k̄ → score_blocks(Eq.10)→ 逐块分
//! 曲线  :top-k 块的质量占比(k = 1,2,4,8,…)vs 论文 top-8≈66.5% / top-16≈77%
//! ```
//!
//! 曲线落在论文量级 ⇒ B-1 打分器可开工;掉到 `k/块数`(随机水平)⇒ 先查
//! §8.2 的配对约定与 §8.3 的布局单源(两条曲线**同时**是该约定的实证:
//! 约定错 ⇒ 打分退化为噪声)。
//!
//! # 三段式(每段独立可回滚)
//!
//! 1. **Q 槽接入 + 单遍归约**(`eval_root_and_q`):探针档把 `ForwardCtx::q_sink`
//!    接到层侧(Q-1 已落),与主根**同一遍** `eval_ops_multi_scoped_env` 归约
//!    —— 前向带 GDN 状态副作用,**绝不双执行**(与 `pf_bisect` 同律)。
//! 2. **Q 回读**(`coverage_take_q`):探针档在**块边界 chunk**结束处把 Q 面
//!    dtoh 成 host f32(`[层][t·hq·hd]`),设备块当场 free。
//! 3. **分析**(`coverage_analyze`):在**块边界事件之后**(索引/快照已落)读
//!    池内 K(会话真页链)+ 索引账 ⇒ 出曲线。分析必须晚于边界事件,否则
//!    刚完成的块还没进索引(打分缺一块 = 覆盖率虚低)。
//!
//! # 纪律
//!
//! - **探针关(生产缺省)= 零改动**:`q_sink = None`,`eval_root_and_q` 走原
//!   `eval_ops_scoped_env` 逐字不变;
//! - 全部回读都是**只读**;唯一的写者仍是模型前向本身(一遍);
//! - 逆 RoPE 只有一份实现(`owl_store::derope_halves_inplace`,与
//!   `MeanKCollector::push_post_rope_halves` 同源)。

use super::*;
use crate::running::RunningEngine;
use std::cell::RefCell;

/// Q 输出槽(探针档;索引 = **全注意力层序号** `fi_kvi`,与池 `kvs` 同序)。
pub(crate) type QSink = RefCell<Vec<Option<TensorOps>>>;

/// **位置方案**(B-2 决策;见设计卷 §7 待拍 1):`A` = 原位置稀疏(零 re-RoPE),
/// `B` = 紧凑窗口 + 有界 delta re-RoPE(需 B-3 的核)。先走 A ⇒ 拿到 dense 对照
/// 数字后再决定要不要为 B 付一枚新核。
pub(crate) const POS_SCHEME: &str = "A(原位置稀疏)";

/// **query 窗宽**(本 chunk 末 W 行一起当 query;论文的 step 级 query 面对应到
/// 我们的 chunk 边界 = 末 W 行)。取 8:末行注意力太局部(常只盯前几个字),
/// 而 W 太大则混入与“当前问题”无关的行。
const W_QUERY_TOKENS: usize = 8;

/// 待分析面(Q 已回读成 host f32;等块边界事件补齐索引账)。
#[derive(Debug)]
pub(crate) struct CoveragePending {
    /// 本 chunk 的绝对起点。
    pub(crate) base: usize,
    /// 本 chunk 的行数。
    pub(crate) t: usize,
    /// 池几何(回读时定格;分析期交叉校验,防档位漂移)
    pub(crate) hq: usize,
    pub(crate) hkv: usize,
    pub(crate) hd: usize,
    pub(crate) layers: usize,
    /// post-RoPE Q:`[层][t × hq × hd]`(层序 = 全注意力层出现序)。
    pub(crate) q: Vec<Vec<f32>>,
}

/// 一次覆盖率观测的 host 结果(供测试/调试断言)。
///
/// 诊断面(#5;#5):部分字段只被探针测试读 ⇒ lib 单独编译时为"无读者",
/// 保留完整读数面(不因告警删观测项)。
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub(crate) struct CoverageReport {
    /// 观测时的总 token 数(= 块边界位置,页对齐)。
    pub(crate) ctx_tokens: usize,
    /// 候选块数(= `ctx_tokens / page`)。
    pub(crate) blocks: usize,
    /// 打分时索引**实际在场**的候选块数(< 块数 = 索引缺口,曲线虚低)
    pub(crate) indexed: usize,
    /// query token 的绝对位置(= `ctx_tokens − 1`)。
    pub(crate) query_pos: usize,
    pub(crate) layers: usize,
    pub(crate) hq: usize,
    pub(crate) hkv: usize,
    pub(crate) hd: usize,
    /// `(k, 覆盖率)` 曲线(覆盖率 = 按索引分排序前 k 块的真实注意力质量之和)。
    pub(crate) curve: Vec<(usize, f64)>,
    /// **上界曲线**(按**真实质量**排序 —— 任何选择器的天花板;实测数据平坦度
    /// 的直接读数:上界也低 ⇒ 是数据平,不是选择器差)。
    pub(crate) oracle: Vec<(usize, f64)>,
    /// 索引 top-8 ∩ 真实质量 top-8 的重合数(0..=8;排序质量的直接读数)。
    pub(crate) overlap8: usize,
    /// 随机基线 `k / 块数`(约定错 ⇒ 曲线贴这条线)。
    pub(crate) uniform: f64,
    /// 逐块真实注意力质量(Σ=1;同序 = 逻辑块序)。
    pub(crate) mass: Vec<f32>,
    /// 逐块索引打分(同序;快路口径 = `ScoreMode::LogitMean`)。
    pub(crate) scores: Vec<f32>,
    /// **B-1 选择器读数**(预算 = `OWL_SELECT_BLOCKS`,缺省 64;sink=2 / recent=4)
    pub(crate) recall: RecallReport,
}

impl CoverageReport {
    /// 指定 k 的覆盖率(None = 该 k 不在曲线里)。
    #[allow(dead_code)] // 诊断读数面:探针测试用
    pub(crate) fn at(&self, k: usize) -> Option<f64> {
        self.curve.iter().find(|(kk, _)| *kk == k).map(|(_, c)| *c)
    }

    /// 人类可读一行(日志/断言失败信息用)。
    pub(crate) fn line(&self) -> String {
        let pts: Vec<String> =
            self.curve.iter().map(|(k, c)| format!("top{k}={:.1}%", c * 100.0)).collect();
        let orc: Vec<String> = self
            .oracle
            .iter()
            .map(|(k, c)| format!("top{k}={:.1}%", c * 100.0))
            .collect();
        let rec = &self.recall;
        format!(
            "覆盖率探针:ctx={} tok / {} 块(索引在场 {})/ {} 层 × {} 头 × hd{} / query@{} —— {} | 上界 {} | top8 重合 {}/8 (随机基线 {:.1}%)\n\
             ‖ 选择器(B-1):预算 {}(sink {}+recent {}) ⇒ 选 {} · 掉 {} · 保底占位 {} | 覆盖 {:.1}%(同预算上界 {:.1}% / 论文口径 {:.1}%) | 打分 {:.1}ms + 选择 {:.1}ms\n\
             ‖ 视图(B-2):{} 页 / {} tok(**物化比 {:.1}%**)· 位置方案 {}\n\
             ‖ 方案对照(与 dense 的注意力输出):A cos={:.4} relL2={:.3} · B cos={:.4} relL2={:.3} · 集中度 dense/A/B = {:.3}/{:.3}/{:.3}",
            self.ctx_tokens,
            self.blocks,
            self.indexed,
            self.layers,
            self.hq,
            self.hd,
            self.query_pos,
            pts.join(" "),
            orc.join(" "),
            self.overlap8,
            self.uniform * 100.0,
            rec.budget,
            rec.sink,
            rec.recent,
            rec.selected,
            rec.dropped,
            rec.reserved,
            rec.coverage * 100.0,
            rec.oracle_coverage * 100.0,
            rec.paper_coverage * 100.0,
            rec.score_ms,
            rec.select_ms,
            rec.view_pages,
            rec.view_tokens,
            rec.materialize_ratio * 100.0,
            rec.pos_scheme,
            rec.scheme.cos_a,
            rec.scheme.rel_a,
            rec.scheme.cos_b,
            rec.scheme.rel_b,
            rec.scheme.topw_dense,
            rec.scheme.topw_a,
            rec.scheme.topw_b
        )
    }
}

/// **方案对照读数**(A = 原位置稀疏 / B = 紧凑 + re-RoPE;均与 dense 比)。
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct SchemeStats {
    /// 与 dense 输出的余弦相似度(对 (层,头) 取均)
    pub(crate) cos_a: f64,
    pub(crate) cos_b: f64,
    /// 相对 L2 误差 `|out−dense| / |dense|`(对 (层,头) 取均)
    pub(crate) rel_a: f64,
    pub(crate) rel_b: f64,
    /// dense / A / B 的输出均方根(量级参考)
    pub(crate) rms_dense: f64,
    /// 三种方案下注意力“集中度”(softmax 最大权重,均值) —— 解释差异用
    pub(crate) topw_dense: f64,
    pub(crate) topw_a: f64,
    pub(crate) topw_b: f64,
}

/// **三方案注意力输出对照**(纯 host;单 query)。
///
/// - `dense` = 全 ctx token(原位置);
/// - `A` = **选中块**的 token(原 RoPE,即池内原样);
/// - `B` = 同一批 token 的 K **重旋到紧凑位置**(`0..W-1`),query 也重旋到 `W`
///   —— 即“选中块当成连续历史”。
///
/// re-RoPE 走**同一份原语**(`owl_store::derope_pair`;正向旋转 = 传 `-sin`),
/// 相对角用 rope 表在 `|Δ|` 处取样(`θ(a)−θ(b) = θ(a−b)`)⇒ 无第二套约定。
pub(crate) fn scheme_compare(
    q_post: &[Vec<f32>],
    k_post: &[Vec<f32>],
    v_post: &[Vec<f32>],
    picked_tok: &[usize],
    ctx_tokens: usize,
    geom: (usize, usize, usize, usize),
    rope: &owl_models::layers::rope::Rope,
) -> SchemeStats {
    let (layers, hq, hkv, hd) = geom;
    let half = rope.rotary_dim() / 2;
    let scale = 1.0 / (hd as f32).sqrt();
    let per_kv = (hq / hkv.max(1)).max(1);
    let (mut ca, mut cb, mut ra, mut rb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut rd, mut td, mut ta, mut tb) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut n = 0usize;
    let mut idx_dense: Vec<usize> = (0..ctx_tokens).collect();
    idx_dense.truncate(ctx_tokens);
    let w = picked_tok.len();
    if w == 0 || layers == 0 {
        return SchemeStats::default();
    }
    // B 的 K/query 重旋(逐 token × 头 × dim;相对角 = θ(|Δ|),Δ = new − old ≤ 0)
    let rotate_into = |dst: &mut [f32], src: &[f32], delta: i64, rope: &owl_models::layers::rope::Rope| {
        dst.copy_from_slice(src);
        if delta == 0 {
            return;
        }
        let a = delta.unsigned_abs() as usize;
        let Some((c, s)) = rope.cos_sin_at(a) else { return };
        // Δ<0 ⇒ 正向旋转量 sin δ = −sin|Δ| ⇒ 传 s_eff = −sin δ = +sin|Δ|
        let sgn = if delta < 0 { 1.0f32 } else { -1.0f32 };
        for i in 0..half {
            let (x, y) = (dst[i], dst[i + half]);
            let (nx, ny) = owl_store::derope_pair(x, y, c[i], s[i] * sgn);
            dst[i] = nx;
            dst[i + half] = ny;
        }
    };
    let mut qbuf = vec![0.0f32; hd];
    let mut kbuf = vec![0.0f32; hd];
    for l in 0..layers {
        for h in 0..hq {
            let g = (h / per_kv).min(hkv.saturating_sub(1));
            let qo = h * hd;
            let dot = |k_row: &[f32], q: &[f32]| -> f32 {
                let mut d = 0.0f32;
                for (x, y) in k_row.iter().zip(q) {
                    d += x * y;
                }
                d * scale
            };
            // ── dense ──
            let mut lg: Vec<f32> = Vec::with_capacity(ctx_tokens);
            for &t in &idx_dense {
                let ko = (t * hkv + g) * hd;
                lg.push(dot(&k_post[l][ko..ko + hd], &q_post[l][qo..qo + hd]));
            }
            let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if !mx.is_finite() {
                continue;
            }
            let mut sw = 0.0f32;
            for x in lg.iter_mut() {
                *x = (*x - mx).exp();
                sw += *x;
            }
            let mut out_d = vec![0.0f32; hd];
            let mut topw = 0.0f32;
            for (i, ww) in lg.iter().enumerate() {
                let p = ww / sw;
                topw = topw.max(p);
                let vo = (idx_dense[i] * hkv + g) * hd;
                for d in 0..hd {
                    out_d[d] += p * v_post[l][vo + d];
                }
            }
            // ── A:视图原位置 ──
            let mut la: Vec<f32> = Vec::with_capacity(w);
            for &t in picked_tok {
                let ko = (t * hkv + g) * hd;
                la.push(dot(&k_post[l][ko..ko + hd], &q_post[l][qo..qo + hd]));
            }
            let mxa = la.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut swa = 0.0f32;
            for x in la.iter_mut() {
                *x = (*x - mxa).exp();
                swa += *x;
            }
            let mut out_a = vec![0.0f32; hd];
            let mut topa = 0.0f32;
            for (i, ww) in la.iter().enumerate() {
                let p = ww / swa;
                topa = topa.max(p);
                let vo = (picked_tok[i] * hkv + g) * hd;
                for d in 0..hd {
                    out_a[d] += p * v_post[l][vo + d];
                }
            }
            // ── B:紧凑位置 + re-RoPE(K 与 query 都搬到紧凑帧)──
            let q_compact = w as i64; // query 落在紧凑历史尾部
            rotate_into(&mut qbuf, &q_post[l][qo..qo + hd], q_compact - (ctx_tokens as i64 - 1), rope);
            let mut lb: Vec<f32> = Vec::with_capacity(w);
            for (i, &t) in picked_tok.iter().enumerate() {
                let ko = (t * hkv + g) * hd;
                rotate_into(&mut kbuf, &k_post[l][ko..ko + hd], i as i64 - t as i64, rope);
                lb.push(dot(&kbuf, &qbuf));
            }
            let mxb = lb.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut swb = 0.0f32;
            for x in lb.iter_mut() {
                *x = (*x - mxb).exp();
                swb += *x;
            }
            let mut out_b = vec![0.0f32; hd];
            let mut topb = 0.0f32;
            for (i, ww) in lb.iter().enumerate() {
                let p = ww / swb;
                topb = topb.max(p);
                let vo = (picked_tok[i] * hkv + g) * hd;
                for d in 0..hd {
                    out_b[d] += p * v_post[l][vo + d];
                }
            }
            // ── 统计 ──
            let (mut dd, mut da, mut db) = (0.0f64, 0.0f64, 0.0f64);
            let (mut na, mut nb) = (0.0f64, 0.0f64);
            for d in 0..hd {
                let (x, a, b) = (out_d[d] as f64, out_a[d] as f64, out_b[d] as f64);
                dd += x * x;
                da += a * a;
                db += b * b;
                na += x * a;
                nb += x * b;
            }
            let (nd, na2, nb2) = (dd.sqrt(), da.sqrt(), db.sqrt());
            if nd <= 1e-12 {
                continue;
            }
            ca += if na2 > 1e-12 { na / (nd * na2) } else { 0.0 };
            cb += if nb2 > 1e-12 { nb / (nd * nb2) } else { 0.0 };
            let ea: f64 = (0..hd).map(|d| (out_a[d] as f64 - out_d[d] as f64).powi(2)).sum::<f64>().sqrt();
            let eb: f64 = (0..hd).map(|d| (out_b[d] as f64 - out_d[d] as f64).powi(2)).sum::<f64>().sqrt();
            ra += ea / nd;
            rb += eb / nd;
            rd += nd;
            td += topw as f64;
            ta += topa as f64;
            tb += topb as f64;
            n += 1;
        }
    }
    if n == 0 {
        return SchemeStats::default();
    }
    let k = n as f64;
    SchemeStats {
        cos_a: ca / k,
        cos_b: cb / k,
        rel_a: ra / k,
        rel_b: rb / k,
        rms_dense: rd / k,
        topw_dense: td / k,
        topw_a: ta / k,
        topw_b: tb / k,
    }
}

/// 按值降序取前 k 的下标(稳定 ⇒ 同分值保序;与 `coverage_curve` 同律)。
pub(crate) fn top_k_idx(vals: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..vals.len()).collect();
    idx.sort_by(|a, b| vals[*b].partial_cmp(&vals[*a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k.min(vals.len()));
    idx
}

/// **真实逐块注意力质量**(论文口径,纯 host):
/// 逐 (层, 查询头) 对 `[0, visible)` 的**已可见**历史 token 做因果 softmax,
/// 按 32-token 块求和;再对 (层,头) 取均值(与 `score_blocks` 的 `mean_{l,h}` 同律
/// —— 两条曲线必须同口径才可比)。
///
/// `q_post[层]` = **单条** query 的 post-RoPE Q(`[hq × hd]`);
/// `k_post[层]` = 池内 post-RoPE K(`[tokens × hkv × hd]`);
/// `visible` = 该 query 的因果前缀长(= pos+1;query = 末 token 时 = 全 ctx);
/// `page` = 块 = 物理页 token 数。
pub(crate) fn true_block_mass(
    q_post: &[Vec<f32>],
    k_post: &[Vec<f32>],
    geom: (usize, usize, usize, usize),
    page: usize,
    visible: usize,
) -> Vec<f32> {
    let (layers, hq, hkv, hd) = geom;
    let tokens = k_post.first().map(|k| k.len() / (hkv * hd)).unwrap_or(0);
    let n_blocks = if page == 0 { 0 } else { tokens / page };
    let vis = visible.min(tokens);
    if layers == 0 || hq == 0 || n_blocks == 0 || vir_empty(q_post.len(), layers, vis) {
        return vec![0.0; n_blocks];
    }
    let scale = 1.0 / (hd as f32).sqrt();
    let per_kv = hq / hkv.max(1); // 核内同律:kv_head = q_head / (hq/hkv)
    let mut acc = vec![0.0f32; n_blocks];
    let mut heads = 0usize;
    let mut scratch = vec![f32::NEG_INFINITY; vis];
    for l in 0..layers {
        let ql = &q_post[l];
        let kl = &k_post[l];
        for h in 0..hq {
            let g = (h / per_kv.max(1)).min(hkv.saturating_sub(1));
            let qo = h * hd;
            // ① logits(可见区 = [0, visible);未掩码区之下的才是因果)
            let mut max = f32::NEG_INFINITY;
            for (i, s) in scratch.iter_mut().enumerate().take(vis) {
                let ko = (i * hkv + g) * hd;
                let mut dot = 0.0f32;
                for d in 0..hd {
                    dot += ql[qo + d] * kl[ko + d];
                }
                let v = dot * scale;
                *s = v;
                if v > max {
                    max = v;
                }
            }
            if !max.is_finite() {
                continue;
            }
            // ② softmax + ③ 逐块求和
            let mut sum = 0.0f32;
            for s in scratch.iter_mut().take(vis) {
                let w = (*s - max).exp();
                *s = w;
                sum += w;
            }
            if sum <= 0.0 {
                continue;
            }
            for (i, w) in scratch.iter().take(vis).enumerate() {
                acc[i / page] += w / sum;
            }
            heads += 1;
        }
    }
    if heads > 0 {
        let denom = heads as f32;
        for a in acc.iter_mut() {
            *a /= denom;
        }
    }
    acc
}

/// 空可见区的早退判据(单独拆出:主函数留一行)。
#[inline]
fn vir_empty(q_layers: usize, layers: usize, vis: usize) -> bool {
    vis == 0 || q_layers < layers
}

/// **B-1 选择结果的真实读数**(召回点的产物;与曲线同一次观测)。
#[derive(Debug, Clone, Default)]
#[allow(dead_code)] // 诊断读数面:探针测试用
pub(crate) struct RecallReport {
    /// 预算(块数)与保底位数(论文:sink + recent 恒保留)
    pub(crate) budget: usize,
    pub(crate) sink: usize,
    pub(crate) recent: usize,
    /// 实际选中 / 因预算挤掉 / 保底占位
    pub(crate) selected: usize,
    pub(crate) dropped: usize,
    pub(crate) reserved: usize,
    /// 选中集的**真实注意力质量覆盖**(与 `curve` 同口径)
    pub(crate) coverage: f64,
    /// 同预算下**上界**覆盖(按真实质量排序取前 `budget` 块)
    pub(crate) oracle_coverage: f64,
    /// 论文口径(softmax 均值)在**同一预算**下的覆盖(对照)
    pub(crate) paper_coverage: f64,
    /// 耗时(ms):打分 / 选择
    pub(crate) score_ms: f64,
    pub(crate) select_ms: f64,
    /// **视图面**(B-2):写入视图页表的页数 / 视图 token 数
    pub(crate) view_pages: usize,
    pub(crate) view_tokens: usize,
    /// **物化比**(G3②)= 视图 token / 全 ctx token(越小越省)
    pub(crate) materialize_ratio: f64,
    /// 位置方案(`A` = 原位置稀疏,零 re-RoPE;`B` = 紧凑 + re-RoPE,待 B-3)
    pub(crate) pos_scheme: &'static str,
    /// **方案 A/B 的注意力输出对照**(`OWL_SCHEME_PROBE=1` 才算;与 dense 比)
    pub(crate) scheme: SchemeStats,
}

/// **同预算上界覆盖**(按真实质量排序取前 k 块的和;纯 host)。
fn oracle_at_budget(mass: &[f32], budget: usize) -> f64 {
    let total: f64 = mass.iter().map(|m| *m as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let mut v: Vec<f32> = mass.to_vec();
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let hit: f64 = v.iter().take(budget).map(|m| *m as f64).sum();
    hit / total
}

/// 选中集的质量占比(与 `oracle_at_budget` 同口径)。
fn coverage_of_set(mass: &[f32], picked: &[usize]) -> f64 {
    let total: f64 = mass.iter().map(|m| *m as f64).sum();
    if total <= 0.0 {
        return 0.0;
    }
    let hit: f64 = picked.iter().filter(|i| **i < mass.len()).map(|i| mass[*i] as f64).sum();
    hit / total
}

/// GQA 映射(查询头 → KV 头;核内同律 `head / (hq/hkv)`,见
/// `cu/attn/pagedattention_f16.cu` 与 `cu/attn/naive_attn_nc.cu`)。
pub(crate) fn gqa_map(hq: usize, hkv: usize) -> Vec<u16> {
    let per = if hkv == 0 { 1 } else { (hq / hkv).max(1) };
    (0..hq).map(|h| (h / per).min(hkv.saturating_sub(1)) as u16).collect()
}

/// f16/e4m3 字节 → f32(回读解码;f32 池直读)。
fn decode_row(dt: Dtype, buf: &[u8]) -> Vec<f32> {
    match dt {
        Dtype::F16 => buf
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F32 => buf.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        _ => Vec::new(),
    }
}

/// 选择预算(环境旋钮;缺省 64 块 + sink 2 / recent 4)。
impl<D: DeviceClient> RunningEngine<D> {
    /// 召回预算(`OWL_SELECT_BLOCKS` / `OWL_SELECT_SINK` / `OWL_SELECT_RECENT`;
    /// 缺省 64/2/4)。选择层未接入物化 ⇒ 现在只是探针参数。
    pub(crate) fn recall_budget(&self) -> (usize, usize, usize) {
        let k = &self.cfg.knobs;
        (k.select_blocks, k.select_sink, k.select_recent)
    }

    /// **B-3e-2 第 2 项:召回点的驻留面调和**(下沉被踢出的 / 回灌选中的)。
    ///
    /// 顺序**先沉后升**:沉的块释放设备槽,升的块正好用这些槽(池不变大)。
    /// 记账一律走 store 的 plan/commit(引擎只编排):批面 = 一次 swap 搬多块。
    async fn reconcile_residency(
        &mut self,
        sid: u64,
        cands: &[owl_store::ChainHash],
        keep: &[owl_store::ChainHash],
        n_blocks: usize,
    ) -> Result<(usize, usize)> {
        use owl_store::ChainHash;
        // ① 分拣:表项 + 身份 + 归属(设备槽 / 占位)
        let mut table: Vec<u32> = self
            .bank
            .sessions
            .get(sid)
            .map(|s| s.block_table.clone())
            .unwrap_or_default();
        table.truncate(n_blocks);
        let mut sink: Vec<(usize, ChainHash, u32)> = Vec::new();
        let mut need: Vec<(usize, ChainHash)> = Vec::new();
        for (i, h) in cands.iter().enumerate().take(n_blocks) {
            let Some(slot) = table.get(i).copied() else { continue };
            let kept = keep.iter().any(|k| k == h);
            if slot == crate::scheduler::PENDING_LOAD_PLACEHOLDER {
                if kept {
                    need.push((i, *h));
                }
            } else if !kept {
                sink.push((i, *h, slot));
            }
        }
        // ①′ **可沉性前置**:store 的宿主层按**缓存树条目**键控 ⇒ 只有
        // "树 + 会话双持"(refs == 2)的块才能:decref(→1 = 缓存独占)→ 下沉。
        // 本轮刚写、**还没进缓存树**的块(refs == 1)若 decref 会直接变自由槽
        // (实测踩:"槽 2 是自由槽,不能加引用")⇒ 这类块**本轮不可沉**,记数上报。
        // (链注册在 turn 收口;要让"本轮内的块也能沉"需增量注册 —— 第 2b 项,见 §8.19)
        let mut sinkable: Vec<(usize, owl_store::ChainHash, u32)> = Vec::new();
        let mut not_in_tree = 0usize;
        for (i, h, slot) in sink.drain(..) {
            if self.bank.rel.leg.slot_refs(slot) == 2 {
                sinkable.push((i, h, slot));
            } else {
                not_in_tree += 1;
            }
        }
        self.partial.not_cacheable += not_in_tree;
        if not_in_tree > 0 {
            owl_shared::log_info!(
                "cov-probe",
                "部分驻留:{not_in_tree} 块本轮未入缓存树 ⇒ 不可下沉(增量注册见 §8.19)"
            );
        }
        let sink = sinkable;
        let n_sink = sink.len();
        let n_need = need.len();
        if n_sink + n_need == 0 {
            return Ok((0, 0));
        }
        // ② 放开会话引用(只对双持块:decref 后 refs == 1 = 缓存独占)
        for (_, _, slot) in &sink {
            if let Err(e) = self.bank.rel.leg.decref(*slot) {
                owl_shared::log_warn!("cov-probe", "下沉前 decref 槽 {slot} 失败:{e}(跳过该块)");
            }
        }
        // ③ 下沉(批:一次 swap 搬 N 块)
        if n_sink > 0 {
            let hashes: Vec<ChainHash> = sink.iter().map(|(_, h, _)| *h).collect();
            let (blocks, pieces, tbl) = {
                let p = owl_store::plan_offload_blocks(&mut self.bank.rel.leg, &hashes)?;
                if p.skipped.len() > 0 {
                    owl_shared::log_warn!(
                        "cov-probe",
                        "逐块下沉票跳过 {} 块(原因见账;LiveRefs = 会话仍持)",
                        p.skipped.len()
                    );
                }
                (p.blocks, p.pieces, p.table)
            };
            if !blocks.is_empty() {
                let conn = {
                    let face = self.graphs.session.face_mut();
                    match face
                        .swap(owl_iface::contract::SwapDir::Out, &pieces, Box::new(tbl))
                        .await
                    {
                        Ok(c) => c,
                        Err(e) => {
                            owl_store::abort_offload(&mut self.bank.rel.leg, &blocks);
                            return Err(owl_iface::contract::ModelError::Msg(format!(
                                "选择层下沉 swap(Out) 失败:{e}"
                            )));
                        }
                    }
                };
                let tbl = owl_store::reclaim_slot_table(conn)?;
                let n = owl_store::commit_offload(&mut self.bank.rel.leg, &blocks, tbl)?;
                // 表写占位(**只对真沉下去的那些**;被跳过的块保留原槽)
                let sunk_set: std::collections::HashSet<ChainHash> =
                    blocks.iter().map(|(h, _)| *h).collect();
                for (i, h, _) in &sink {
                    if sunk_set.contains(h) {
                        table[*i] = crate::scheduler::PENDING_LOAD_PLACEHOLDER;
                    }
                }
                if let Some(s) = self.bank.sessions.get_mut(sid) {
                    s.block_table[..n_blocks].copy_from_slice(&table[..n_blocks]);
                }
                let _ = n;
            }
        }
        // ④ 回灌选中的宿主块(批)
        let mut staged = 0usize;
        if n_need > 0 {
            let hashes: Vec<ChainHash> = need.iter().map(|(_, h)| *h).collect();
            let (cs, slots, pieces, tbl) = {
                let p = owl_store::plan_onload(&mut self.bank.rel.leg, &hashes)?;
                (p.chains, p.slots, p.pieces, p.table)
            };
            let conn = {
                let face = self.graphs.session.face_mut();
                match face
                    .swap(owl_iface::contract::SwapDir::In, &pieces, Box::new(tbl))
                    .await
                {
                    Ok(c) => c,
                    Err(e) => {
                        owl_store::abort_onload(&mut self.bank.rel.leg, &cs, &slots);
                        return Err(owl_iface::contract::ModelError::Msg(format!(
                            "选择层回灌 swap(In) 失败:{e}"
                        )));
                    }
                }
            };
            let tbl = owl_store::reclaim_slot_table(conn)?;
            let got = owl_store::commit_onload(&mut self.bank.rel.leg, &cs, &slots, tbl)?;
            staged = got.len();
            // 槽写回表(同序:need 与 cs 同序)
            for ((i, _), slot) in need.iter().zip(&got) {
                table[*i] = *slot;
            }
            if let Some(s) = self.bank.sessions.get_mut(sid) {
                s.block_table[..n_blocks].copy_from_slice(&table[..n_blocks]);
            }
        }
        Ok((n_sink, staged))
    }

    /// **B-3e-2 第 3 项:prefill 的"过去视图"**(部分驻留档)。
    ///
    /// 返回 `(页链, 视图内 token 数)`;`None` = 无视图(全量链,老路径)。
    ///
    /// - 页链 = `选中块(时间序)` ++ `会话链[recall_pos/page ..= 覆盖到本 chunk 末]`;
    /// - 视图 token 数 = `选中 token + (本 chunk 末位置 + 1 − recall_pos)`;
    /// - 需要写设备表 ⇒ 顺手把 `bt_view` 重写为**本 chunk 的**表(池块,非图槽)。
    ///
    /// 为什么要它:被踢出工作集的块已下沉(会话链上是占位符)⇒ prefill 若不换表,
    /// 读到的是占位符当页号 = 静默错页。
    pub(crate) fn past_view_of(&mut self, base: usize, t: usize) -> Option<(Vec<u32>, usize)> {
        if !self.cfg.knobs.select_partial {
            return None;
        }
        let v = self.sel_view.as_ref()?;
        if base < v.recall_pos {
            return None; // 召回点之前:链还是完整的,不必换表
        }
        let page = self.pool.page.max(1);
        let chain: Vec<u32> = self
            .turns
            .active
            .as_ref()
            .and_then(|a| self.bank.sessions.get(a.session_id))
            .map(|s| s.block_table.clone())
            .unwrap_or_default();
        // **只有在链上真出现洞(已下沉)时才换表** —— 没洞就用全量链(更保真;
        // 换表 = 稀疏化过去,是有代价的语义选择,不该无谓付)。
        let last_lb = (base + t.saturating_sub(1)) / page;
        if !chain[..=last_lb.min(chain.len().saturating_sub(1))]
            .iter()
            .any(|s| *s == crate::scheduler::PENDING_LOAD_PLACEHOLDER)
        {
            return None;
        }
        self.partial.prefill_subset += 1;
        let mut pages = v.pages.clone();
        let last = base + t.saturating_sub(1);
        for lb in (v.recall_pos / page)..=(last / page) {
            if let Some(p) = chain.get(lb) {
                pages.push(*p);
            }
        }
        let tokens = v.tokens + (last + 1).saturating_sub(v.recall_pos);
        Some((pages, tokens))
    }

    /// **B-3d 每步视图展开**:`(sel_pt, sel_len)` 两个输入槽的值。
    ///
    /// - 无视图态(召回点未发生 / 已收口)⇒ **全量链 + `pos+1`**(与不开视图
    ///   逐字等价 —— 单一路径,不用切换图);
    /// - 有视图态 ⇒ `选中页 ++ 尾部页(recall_pos..=pos)` 与
    ///   `选中 token + (pos+1 − recall_pos)`;尾部页取自会话真链(连续);
    /// - 超容量(视图页数 > 池页数)= **响亮拒前的兜底**:截尾并记一次(不静默
    ///   少块 —— 选择层的账必须响)。
    pub(crate) fn select_view_step(&mut self, sid: u64, pos: usize) -> (Vec<f32>, usize) {
        let nb = self.pool.nb;
        let page = self.pool.page.max(1);
        let chain: Vec<u32> = self
            .bank
            .sessions
            .get(sid)
            .map(|s| s.block_table.clone())
            .unwrap_or_default();
        let Some(v) = self.sel_view.as_ref() else {
            // **占位符守卫**(B-3e-2):部分驻留档下"无视图"是致命的 —— 会话链里
            // 有 `PENDING_LOAD_PLACEHOLDER`(u32::MAX),当页号读 = 静默错页。
            // 响亮打出哨兵(调用方看日志即知:召回点没跑 / 视图被清)。
            if chain.iter().any(|s| *s == crate::scheduler::PENDING_LOAD_PLACEHOLDER) {
                owl_shared::log_error!(
                    "cov-probe",
                    "视图缺位但会话链含占位块({} 项)⇒ 本步页表不可信(部分驻留档必须每步有视图)",
                    chain.len()
                );
            }
            self.sel_view_fallback += 1;
            let mut pt = vec![0.0f32; nb];
            for (i, b) in chain.iter().take(nb).enumerate() {
                pt[i] = *b as f32;
            }
            return (pt, pos + 1);
        };
        self.select_view_steps += 1;
        // 尾部:召回点是块边界 ⇒ 从 recall_pos/page 起到 pos/page
        let tail_lo = v.recall_pos / page;
        let tail_hi = pos / page;
        let mut pages = v.pages.clone();
        for lb in tail_lo..=tail_hi {
            if let Some(p) = chain.get(lb) {
                pages.push(*p);
            }
        }
        let len = v.tokens + (pos + 1).saturating_sub(v.recall_pos);
        if pages.len() > nb {
            owl_shared::log_warn!(
                "cov-probe",
                "视图超容量:{}/{} 页 ⇒ 截尾(选块预算应 ≤ 池容量 − 尾部)",
                pages.len(), nb
            );
            pages.truncate(nb);
        }
        let mut pt = vec![0.0f32; nb];
        for (i, p) in pages.iter().enumerate() {
            pt[i] = *p as f32;
        }
        (pt, len)
    }

    /// **主根 (+ Q 槽)单遍归约**(Q-2 第一段)。
    ///
    /// - `sink = None`(生产 / 探针关)⇒ 原 `eval_ops_scoped_env`,**逐字不变**;
    /// - `sink = Some` ⇒ `eval_ops_multi_scoped_env`(多根共享 memo):主根与
    ///   Q 节点一次归约,共享子树只算一遍 —— 前向有 GDN 状态副作用,
    ///   双执行 = 双推进(与 `pf_bisect` 同一条律)。
    ///
    /// 返回 `(主根块, 回收清单, Q 块表)`;`sink = None` 时 Q 块表 = `None`。
    pub(crate) async fn eval_root_and_q(
        &mut self,
        root: &TensorOps,
        sink: Option<&QSink>,
    ) -> Result<(Bytes, Vec<u64>, Option<Vec<Bytes>>)> {
        let Some(sink) = sink else {
            let face = self.graphs.session.face_mut();
            let (b, arena) = owl_models::interpreters::eval_ops_scoped_env(root.step(), face, self.env)
                .await?;
            return Ok((b, arena, None));
        };
        let qs: Vec<TensorOps> = sink.borrow().iter().flatten().cloned().collect();
        if qs.is_empty() {
            return Err(ModelError::Msg(
                "覆盖率探针:Q 槽为空 —— 本 chunk 未经过任何全注意力层(或层侧未接线)".into(),
            ));
        }
        let mut roots: Vec<&TensorOps> = Vec::with_capacity(qs.len() + 1);
        roots.push(root);
        roots.extend(qs.iter());
        let outs = {
            let face = self.graphs.session.face_mut();
            owl_models::interpreters::eval_ops_multi_scoped_env(&roots, face, self.env).await?
        };
        let hfb = outs[0].clone();
        // ⚠️ 根块**不被** multi_scoped 回收 ⇒ 调用方负责 hfb / Q 块的生命周期
        Ok((hfb, Vec::new(), Some(outs[1..].to_vec())))
    }

    /// **Q 回读**(Q-2 第二段):设备 Q 块 → host f32(`[层][t·hq·hd]`),块当场 free。
    ///
    /// 只接**块边界 chunk**(调用方判 `(base+t) % page == 0`):分析期索引账
    /// 才完整(§8.4)。
    pub(crate) async fn coverage_take_q(
        &mut self,
        q_blocks: Vec<Bytes>,
        base: usize,
        t: usize,
    ) -> Result<()> {
        let d = self.pool.dims;
        let (hq, hkv, hd) = (d.hq, d.hkv, d.hd);
        let n = t * hq * hd;
        let elem = d.dtype.size_bytes();
        let mut q = Vec::with_capacity(q_blocks.len());
        {
            let face = self.graphs.session.face_mut();
            for b in &q_blocks {
                let mut buf = vec![0u8; n * elem];
                face.dtoh(b, &mut buf).await?;
                let row = decode_row(d.dtype, &buf);
                if row.len() != n {
                    return Err(ModelError::Msg(format!(
                        "覆盖率探针:Q 面解码长不符 {} ≠ {n}(dtype={:?})",
                        row.len(),
                        d.dtype
                    )));
                }
                q.push(row);
            }
            let ids: Vec<u64> = q_blocks.iter().map(|b| b.id).collect();
            face.free(&ids).await?;
        }
        self.cov_pending = Some(CoveragePending {
            base,
            t,
            hq,
            hkv,
            hd,
            layers: q.len(),
            q,
        });
        Ok(())
    }

    /// **分析**(Q-2 第三段):池内 K(会话真页链)+ 索引账 ⇒ 覆盖率曲线。
    ///
    /// 前置:块边界事件已落(索引/快照已登记)⇒ 由 `execute_prefill` 在
    /// `capture_gdn_snap` 之后调用。`ctx_tokens` = 边界位置(页对齐)。
    pub(crate) async fn coverage_analyze(&mut self, sid: u64, ctx_tokens: usize) -> Result<()> {
        let Some(pid) = self.cov_pending.take() else { return Ok(()) };
        // **方案对照档**(`OWL_SCHEME_PROBE=1`):多读一份 V + host 侧算三种方案的
        // 注意力输出(比"接进 decode 跑两遍"便宜得多,且**同一个口径**可比)
        // `OWL_SCHEME_MIN_CTX`(缺省 0 = 每个边界都算):只在这个 ctx 以上才算
        // ⇒ 规模档只跑最后一次(host 数学与 dense 成正比,小边界算了也白算)
        let scheme_min = owl_shared::env_reader::parse::<usize>("OWL_SCHEME_MIN_CTX").unwrap_or(0);
        let want_scheme =
            owl_shared::env_reader::flag("OWL_SCHEME_PROBE") && ctx_tokens >= scheme_min;
        // **重诊断采样**(`OWL_COV_ANALYZE_EVERY=k`;缺省 1 = 每边界都算)。
        // 为什么:重诊断 = K/V 全量回读 + host 侧逐元素 f16→f32(27B/8k 每边界
        // ~2.7 亿次解码)⇒ debug 构建下单边界秒级;选择/视图/驻留(生产面)**不**受影响,
        // 它们只用索引打分(host 面,零回读)。
        let analyze_every = owl_shared::env_reader::parse::<usize>("OWL_COV_ANALYZE_EVERY").unwrap_or(1);
        let analyze_now =
            analyze_every > 0 && (ctx_tokens / self.pool.page.max(1)) % analyze_every == 0;
        if ctx_tokens != pid.base + pid.t {
            owl_shared::log_debug!(
                "cov-probe",
                "待分析面与边界错配(pend {}..{} vs ctx {ctx_tokens})⇒ 丢弃(下一次边界重采)",
                pid.base,
                pid.base + pid.t
            );
            return Ok(());
        }
        let d = self.pool.dims;
        let (hq, hkv, hd, layers) = (d.hq, d.hkv, d.hd, pid.layers);
        let page = self.pool.page;
        if (hq, hkv, hd) != (pid.hq, pid.hkv, pid.hd) {
            return Err(ModelError::Msg("覆盖率探针:池几何与待分析面不符(档位漂移)".into()));
        }
        if page == 0 || !ctx_tokens.is_multiple_of(page) || self.pool.kvs.len() < layers {
            return Err(ModelError::Msg(format!(
                "覆盖率探针:几何非法(ctx={ctx_tokens} page={page} 池层数={} 探针层数={layers})",
                self.pool.kvs.len()
            )));
        }
        if self.pool.kv_fp8 {
            // 一期只接 f16 池:K 回读是**诊断面**,fp8 解码另开一刀(不静默算错值)
            return Err(ModelError::Msg(
                "覆盖率探针:fp8 池 K 回读解码未接线(一期只支持 f16 池;见设计卷 §8.6 挂账)".into(),
            ));
        }
        let n_blocks = ctx_tokens / page;
        let bt: Vec<u32> = self
            .bank
            .sessions
            .get(sid)
            .map(|s| s.block_table.clone())
            .unwrap_or_default();
        if bt.len() < n_blocks {
            return Err(ModelError::Msg(format!(
                "覆盖率探针:会话块表不足({} < {n_blocks})",
                bt.len()
            )));
        }

        // ── ① 池内 post-RoPE K 回读(逐层 × 逐逻辑块;会话真页链)——
        // **仅采样档执行**(见 `analyze_now`);采样外:mass/curve 为空,选择面照跑
        let mut k_post: Vec<Vec<f32>> = if analyze_now {
            vec![vec![0.0f32; ctx_tokens * hkv * hd]; layers]
        } else {
            Vec::new()
        };
        let mut v_post: Vec<Vec<f32>> = Vec::new();
        if analyze_now {
        let elem = 2usize; // f16 池(上方已拒 fp8)
        let page_bytes = page * hkv * hd * elem;
        // V 面(方案 A/B 的**输出级对照**要用;同段同布局,Layout = [token][head][dim])
        v_post = if want_scheme {
            vec![vec![0.0f32; ctx_tokens * hkv * hd]; layers]
        } else {
            Vec::new()
        };
        // **一条批量命令**读完所有 (层, 页) 的 K(与可选 V)—— 原实现每 (层,页)
        // 一次同步 `dtoh_at`:27B/8192 tok = 16×256 = 4096 次往返/边界 ⇒ 20s+ 边界,
        // 65 个边界就把一次 27B prefill 拖成 20 分钟(实测"像卡死")。
        let host_skipped = bt
            .iter()
            .take(n_blocks)
            .filter(|p| owl_store::is_host_placeholder(**p))
            .count();
        let mut pieces: Vec<(&owl_iface::contract::Bytes, usize, usize)> = Vec::new();
        for kv in self.pool.kvs.iter().take(layers) {
            for (lb, phys) in bt.iter().take(n_blocks).enumerate() {
                let _ = lb;
                if owl_store::is_host_placeholder(*phys) {
                    continue;
                }
                let off = (*phys as usize) * page_bytes;
                pieces.push((&kv.k_cache.0, off, page_bytes));
                if want_scheme {
                    pieces.push((&kv.v_cache.0, off, page_bytes));
                }
            }
        }
        let all = {
            let face = self.graphs.session.face_mut();
            face.dtoh_batch(&pieces).await?
        };
        // 按同一顺序切回(每块 K(+V));`host_skipped` 早已在收集时跳过 ⇒ 顺序自洽
        let per_page = if want_scheme { page_bytes * 2 } else { page_bytes };
        let mut cursor = 0usize;
        for (li, _kv) in self.pool.kvs.iter().enumerate().take(layers) {
            let dst = &mut k_post[li];
            let mut vdst = if want_scheme { v_post.get_mut(li) } else { None };
            for (lb, phys) in bt.iter().take(n_blocks).enumerate() {
                if owl_store::is_host_placeholder(*phys) {
                    continue;
                }
                let kbuf = &all[cursor..cursor + page_bytes];
                cursor += page_bytes;
                let vbuf = if want_scheme {
                    let b = &all[cursor..cursor + page_bytes];
                    cursor += page_bytes;
                    Some(b)
                } else {
                    None
                };
                let _ = per_page;
                // 页内序 = [token][head][dim](knhd_offset 单源)
                for p in 0..page {
                    let tok = lb * page + p;
                    for h in 0..hkv {
                        let src = (p * hkv + h) * hd * elem;
                        let off = (tok * hkv + h) * hd;
                        for dd in 0..hd {
                            let b0 = src + dd * elem;
                            dst[off + dd] =
                                half::f16::from_le_bytes([kbuf[b0], kbuf[b0 + 1]]).to_f32();
                            if let (Some(v), Some(vb)) = (vdst.as_deref_mut(), vbuf) {
                                v[off + dd] =
                                    half::f16::from_le_bytes([vb[b0], vb[b0 + 1]]).to_f32();
                            }
                        }
                    }
                }
            }
        }
        if host_skipped > 0 {
            owl_shared::log_info!(
                "cov-probe",
                "覆盖率读数口径:{host_skipped}/{n_blocks} 块在宿主(部分驻留)⇒ 真实质量/方案对照只覆盖设备驻留子集"
            );
        }
        } // ← 重诊断(采样)段结束
        // ── ② 打分面:q̃(去 RoPE)+ 候选块链哈希 + Mean-K 索引 ──
        // **query 窗**(论文 §4.3:当前 step 的 query;**单 token 的注意力太局
        // 部**,末 token 常只盯前几个字)。取本 chunk **末 W 行**,逐行去 RoPE +
        // 逐行真实质量,再对窗取均值 —— 与论文 Eq.10 的 `Σ_m` 同构。
        let w = W_QUERY_TOKENS.min(pid.t);
        let first = pid.t - w;
        let mut face = owl_store::QueryFace::new(hd, layers, hq);
        let mut mass = vec![0.0f32; n_blocks];
        let mut q_post_last: Vec<Vec<f32>> = Vec::new();
        for qi in first..pid.t {
            let pos = pid.base + qi;
            let (cos, sin) = match self.rope.cos_sin_at(pos) {
                Some((c, s)) => (c.to_vec(), s.to_vec()),
                None => {
                    return Err(ModelError::Msg(format!(
                        "覆盖率探针:rope 表不含位置 {pos}(表长不足?)"
                    )))
                }
            };
            let q_off = qi * hq * hd;
            let mut flat = vec![0.0f32; layers * hq * hd];
            let mut q_post: Vec<Vec<f32>> = Vec::with_capacity(layers);
            let mut row = vec![0.0f32; hq * hd];
            let mut qh = vec![0.0f32; hd];
            for l in 0..layers {
                row.copy_from_slice(&pid.q[l][q_off..q_off + hq * hd]);
                for h in 0..hq {
                    qh.copy_from_slice(&row[h * hd..(h + 1) * hd]);
                    owl_store::derope_halves_inplace(&mut qh, &cos, &sin)?;
                    flat[(l * hq + h) * hd..(l * hq + h + 1) * hd].copy_from_slice(&qh);
                }
                q_post.push(row.clone());
            }
            face.push_token(flat)?;
            let m = true_block_mass(&q_post, &k_post, (layers, hq, hkv, hd), page, pos + 1);
            for (a, b) in mass.iter_mut().zip(m) {
                *a += b;
            }
            q_post_last = q_post;
        }
        let denom = w as f32;
        for a in mass.iter_mut() {
            *a /= denom;
        }
        let pos = ctx_tokens - 1;
        // 候选块身份 = 本会话 token 链的逐块父链哈希(与索引登记同源:
        // `owl_store::hash_block` 是身份的唯一产地)
        let toks = self.tokens_upto(0, n_blocks * page)?;
        let mut parent = owl_store::ChainHash::new(0);
        let mut cands = Vec::with_capacity(n_blocks);
        for b in 0..n_blocks {
            parent = owl_store::hash_block(parent, &toks[b * page..(b + 1) * page]);
            cands.push(parent);
        }
        let Some(index) = self.bank.rel.index.as_ref() else {
            return Err(ModelError::Msg(
                "覆盖率探针:索引账不在场(index_probe 未开?两枚旋钮需同开)".into(),
            ));
        };
        let indexed = cands.iter().filter(|c| index.contains(**c)).count();
        let gqa = gqa_map(hq, hkv);
        // ── **B-1 召回点**:打分(快路口径)+ 选择(预算/sink/recent)+ 实时读数 ──
        // 快路 = `ScoreMode::LogitMean`:两层线性折叠(窗 Σ_m + GQA 头 Σ_h),
        // 与论文口径**仅 FP 重排**(排序相同)但便宜 50×~24×(见 store 台账)。
        let t_sc = std::time::Instant::now();
        let scores = owl_store::score_blocks(
            index,
            &face,
            &cands,
            &gqa,
            owl_store::ScoreTuning { scale: 0.0, mode: owl_store::ScoreMode::LogitMean },
        )?;
        let score_ms = t_sc.elapsed().as_secs_f64() * 1000.0;
        let (budget_blocks, sink, recent) = self.recall_budget();
        let t_sel = std::time::Instant::now();
        let scored: Vec<(owl_store::ChainHash, f32)> = cands.iter().copied().zip(scores.iter().copied()).collect();
        let set = owl_store::select(
            &scored,
            &cands,
            owl_store::SelectBudget::new(budget_blocks.min(n_blocks), sink, recent),
        );
        let select_ms = t_sel.elapsed().as_secs_f64() * 1000.0;
        // **B-4 召回信号**:把"本步被注意力选中"的块告诉策略面
        // (attention-aware LRFU 的 keep-score;LRU 策略 = 无操作 ⇒ 默认路径零影响)
        self.bank.rel.note_recall(&set.blocks);
        let picked: Vec<usize> = set
            .blocks
            .iter()
            .filter_map(|b| cands.iter().position(|c| c == b))
            .collect();
        let coverage = coverage_of_set(&mass, &picked);
        let oracle_coverage = oracle_at_budget(&mass, budget_blocks.min(n_blocks));
        // 论文口径(softmax 均值)在同预算下的覆盖 —— 保真对照(慢,只报数)
        let paper_scores = owl_store::score_blocks(index, &face, &cands, &gqa, owl_store::ScoreTuning::default())?;
        let paper_set = {
            let ps: Vec<(owl_store::ChainHash, f32)> =
                cands.iter().copied().zip(paper_scores.iter().copied()).collect();
            owl_store::select(
                &ps,
                &cands,
                owl_store::SelectBudget::new(budget_blocks.min(n_blocks), sink, recent),
            )
        };
        let paper_picked: Vec<usize> = paper_set
            .blocks
            .iter()
            .filter_map(|b| cands.iter().position(|c| c == b))
            .collect();
        let paper_coverage = coverage_of_set(&mass, &paper_picked);
        // ── B-3e-2 第 2 项:召回点的**下沉 / 回灌**(部分驻留档)──
        // 被踢出工作集的链块 ⇒ `decref`(放开会话引用)+ 逐块下沉 + 表写占位;
        // 选中但内容在宿主的块 ⇒ 回灌 + 写回槽(复用 turn 起点那套 plan/swap/commit)。
        // 先沉后升:沉的这一步**腾出的槽**正是升要用的槽。
        if self.cfg.knobs.select_partial {
            let (sunk, staged) = self.reconcile_residency(sid, &cands, &set.blocks, n_blocks).await?;
            self.partial.sunk += sunk;
            self.partial.staged += staged;
            if sunk + staged > 0 {
                owl_shared::log_info!(
                    "cov-probe",
                    "部分驻留:下沉 {sunk} 块 · 回灌 {staged} 块(表 {} 项)",
                    n_blocks
                );
            }
        }
        // ── B-2 视图装配:选中块(时间序)的**物理页** ⇒ 视图块表 ──
        // 位置方案 **A**(原位置稀疏):K 一字不改(池里就是它在原位置算出的
        // post-RoPE K),只换页表 —— 零新核、零数值漂移。方案 B(紧凑 + re-RoPE)
        // 留 B-3 当对照实验(同一套门跑两遍比数字)。
        // 用**调和后**的会话表(回灌已把新槽写回;先装配再回灌 = 页表带占位符 ⇒ 非法地址)
        let bt_after: Vec<u32> = self
            .bank
            .sessions
            .get(sid)
            .map(|s| s.block_table.clone())
            .unwrap_or_default();
        let mut view_pages: Vec<u32> = set
            .blocks
            .iter()
            .filter_map(|b| cands.iter().position(|c| c == b))
            .filter_map(|i| bt_after.get(i).copied())
            .collect();
        if view_pages.iter().any(|p| owl_store::is_host_placeholder(*p)) {
            return Err(ModelError::Msg(
                "视图装配:选中块仍是宿主占位符(回灌没成功?)⇒ 拒绝写视图页表".into(),
            ));
        }
        let cap = self.pool.select_view_capacity();
        if view_pages.len() > cap {
            owl_shared::log_warn!(
                "cov-probe",
                "视图超容量:{}/{} 页 ⇒ 截到容量(预算应 ≤ 池容量;选择层账不平会响亮记)"
                , view_pages.len(), cap
            );
            view_pages.truncate(cap);
        }
        let view_tokens = view_pages.len() * page;
        {
            let face = self.graphs.session.face_mut();
            self.pool.write_select_view(face, &view_pages).await?;
        }
        // B-3d:视图态落引擎(供 decode 每步展开);位置方案 A ⇒ 位置一律原样。
        // ⚠️ **只在 `select_decode` 档落** —— 否则 dense 对照臂也会用上视图,
        // 那就没有对照了(踩过:直觉写法会两臂同视图)。
        if self.cfg.knobs.select_decode {
            self.sel_view = Some(crate::running::SelectView {
                pages: view_pages.clone(),
                tokens: view_tokens,
                recall_pos: ctx_tokens,
            });
        }
        // ── B-3c-pre 方案对照(A vs B,各自与 dense 比;host 单 query)──
        let scheme = if want_scheme {
            let mut picked_tok: Vec<usize> = Vec::new();
            for b in &set.blocks {
                if let Some(i) = cands.iter().position(|c| c == b) {
                    picked_tok.extend(i * page..(i * page + page).min(ctx_tokens));
                }
            }
            let q_row: Vec<Vec<f32>> = (0..layers)
                .map(|l| {
                    let off = (pid.t - 1) * hq * hd;
                    pid.q[l][off..off + hq * hd].to_vec()
                })
                .collect();
            let st = scheme_compare(
                &q_row,
                &k_post,
                &v_post,
                &picked_tok,
                ctx_tokens,
                (layers, hq, hkv, hd),
                &self.rope,
            );
            owl_shared::log_info!(
                "cov-probe",
                "方案对照(与 dense 的注意力输出):A(原位置) cos={:.4} relL2={:.3} · B(紧凑+re-RoPE) cos={:.4} relL2={:.3} ‖ 集中度 dense/A/B = {:.3}/{:.3}/{:.3}",
                st.cos_a, st.rel_a, st.cos_b, st.rel_b, st.topw_dense, st.topw_a, st.topw_b
            );
            st
        } else {
            SchemeStats::default()
        };
        let recall = RecallReport {
            budget: budget_blocks.min(n_blocks),
            sink,
            recent,
            selected: set.blocks.len(),
            dropped: set.dropped,
            reserved: set.reserved,
            coverage,
            oracle_coverage,
            paper_coverage,
            score_ms,
            select_ms,
            view_pages: view_pages.len(),
            view_tokens,
            materialize_ratio: if ctx_tokens == 0 { 0.0 } else { view_tokens as f64 / ctx_tokens as f64 },
            pos_scheme: POS_SCHEME,
            scheme,
        };
        // 对照:论文口径的逐块分(曲线用;与快路同序 ⇒ 曲线取自 fast)
        let _ = &paper_scores;
        // ── ③ 覆盖率(真实质量已在上方按 query 窗累均)──
        let _ = &q_post_last; // 末行 post-RoPE 面(诊断保留;后续 B-1 打分器用于 tile 化)
        let ks: Vec<usize> =
            [1usize, 2, 4, 8, 16, 32].iter().copied().filter(|k| *k <= n_blocks).collect();
        let curve = owl_store::coverage_curve(&mass, &scores, &ks);
        // 上界(按真实质量自排 = 完美选择器)+ top-8 重合(排序质量的直接读数)
        let oracle = owl_store::coverage_curve(&mass, &mass, &ks);
        let overlap8 = {
            let a = top_k_idx(&scores, 8);
            let b = top_k_idx(&mass, 8);
            a.iter().filter(|i| b.contains(i)).count()
        };
        let rep = CoverageReport {
            ctx_tokens,
            blocks: n_blocks,
            indexed,
            query_pos: pos,
            layers,
            hq,
            hkv,
            hd,
            curve,
            oracle,
            overlap8,
            uniform: 1.0 / n_blocks as f64,
            mass,
            scores,
            recall,
        };
        owl_shared::log_info!("cov-probe", "{}", rep.line());
        if indexed < n_blocks {
            owl_shared::log_warn!(
                "cov-probe",
                "索引缺口:{indexed}/{n_blocks} 块在场 ⇒ 曲线虚低(边界事件与分析的序?)"
            );
        }
        self.cov_last = Some(rep);
        Ok(())
    }
}
