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
    /// **本面存的行数**(= `W_QUERY_TOKENS.min(t)`)。
    ///
    /// 为什么不是整个 chunk:2026-10-19 相位账实测 —— 原实现把**整个 chunk**
    /// 的 Q(0.8B/chunk1024 = 6 层 × 4MB = 25MB)全量 D2H + 逐元素解码,
    /// 而分析面**只用末 8 行** ⇒ 查询面相位 **302ms/边界(占 B 轴 112%)** ——
    /// 128× 白烧。改为只回读末 W 行(仍一条批命令)⇒ 字节数与解码量同降。
    pub(crate) w: usize,
    /// post-RoPE Q:`[层][w × hq × hd]`(层序 = 全注意力层出现序)。
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
        // **A1 取证**(2026-10-19):池被"本轮新写、结构性不可沉"的页填满
        // ⇒ 下一 chunk 的 `ensure_for_len` 必 `Exhausted`。这里把三条数打出来:
        // 可沉 / 待回灌 / 不可沉,以及**候选块的 refs 分布**(1=仅会话持,
        // 2=树+会话双持=可沉)。
        if std::env::var_os("OWL_RECON_DBG").is_some() {
            let mut hist = [0usize; 4];
            for (i, _h) in cands.iter().enumerate().take(n_blocks) {
                if let Some(slot) = table.get(i).copied() {
                    if slot != crate::scheduler::PENDING_LOAD_PLACEHOLDER {
                        let r = self.bank.rel.leg.slot_refs(slot).min(3) as usize;
                        hist[r] += 1;
                    }
                }
            }
            eprintln!(
                "[recon] 候选 {n_blocks} · 表 {} · 可沉 {} · 待回灌 {} · **不可沉 {}** · refs(1/2/3+)={}/{}/{}",
                table.len(),
                sink.len(),
                need.len(),
                not_in_tree,
                hist[1],
                hist[2],
                hist[3]
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
        // **视图等效偏移**(= 本 chunk **起点**处的可见 token 数)。
        //
        // 语义:换表后的链 = [召回点的选中块(时间序)] ++ [recall_pos, base+t) 的尾页 ⇒
        // 第 i 行(本 chunk 内偏移 i)的可见数 = `本值 + i + 1`。
        // ⚠️ 旧实现用的是 `last+1−recall_pos`(= **含本 chunk** 的计数)⇒
        //   `lens` 被抬高 `t`(同一 chunk 的**未来 token 泄漏**)且 choke 了 FI 表的
        //   "等效偏移"契约(见下 `fi_tables_view` 调用)⇒ 注意力窗口 = 全 ctx
        //   ⇒ **每 chunk O(ctx)**(2026-10-19 262k 冲刺实测:前向 517→1718ms/chunk)。
        let tokens = v.tokens + base.saturating_sub(v.recall_pos);
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

    /// ⚠️ **临时取证**(A2;`OWL_COV_DUMP=1`,事后删):**末行 hidden 指纹**。
    ///
    /// 判别“前向真的随 chunk 漂”(同一绝对行的 hidden 不同 ⇒ 算出来的东西不同)
    /// 还是“Q 槽读回面坏了”(hidden 相同而 Q 面不同 ⇒ 读的是别的内存)。
    pub(crate) async fn dump_hidden_row(
        &mut self,
        block: &owl_iface::contract::Bytes,
        base: usize,
        t: usize,
        d_model: usize,
    ) -> Result<()> {
        if std::env::var_os("OWL_COV_DUMP").is_none() || t == 0 {
            return Ok(());
        }
        let esz = self.pool.dims.dtype.size_bytes();
        let row = d_model * esz;
        // **整块回读**(逐行 L2)—— 异常是"尾行"还是"整块",只有整块 profile 能判
        let buf = {
            let face = self.graphs.session.face_mut();
            face.dtoh_batch(&[(block, 0, t * row)]).await?
        };
        let dec = |c: &[u8]| -> f32 {
            match esz {
                2 => half::f16::from_le_bytes([c[0], c[1]]).to_f32(),
                _ => f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
            }
        };
        let mut rows: Vec<f32> = Vec::with_capacity(t);
        for r in 0..t {
            let seg = &buf[r * row..(r + 1) * row];
            rows.push(seg.chunks_exact(esz).map(dec).map(|x| x * x).sum::<f32>().sqrt());
        }
        let tail: Vec<String> = rows[t.saturating_sub(16)..].iter().map(|v| format!("{v:.1}")).collect();
        let stride = (t / 8).max(1);
        let prof: Vec<String> = (0..t)
            .step_by(stride)
            .map(|r| format!("{}:{:.1}", base + r, rows[r]))
            .collect();
        eprintln!(
            "[cov-dump] HID base={base} t={t} 首行={:.1} 末行={:.1} ‖ 尾16=[{}]",
            rows[0],
            rows[t - 1],
            tail.join(" ")
        );
        eprintln!("[cov-dump] PROF base={base} {}", prof.join(" "));
        let l2 = rows[t - 1];
        let last_seg = &buf[(t - 1) * row..t * row];
        let h = {
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            for x in last_seg.chunks_exact(esz).map(dec) {
                for b in x.to_le_bytes() {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x100_0000_01b3);
                }
            }
            h
        };
        let v0 = dec(&last_seg[..esz]);
        let v1 = dec(&last_seg[esz..2 * esz]);
        let v2 = dec(&last_seg[2 * esz..3 * esz]);
        let v3 = dec(&last_seg[3 * esz..4 * esz]);
        eprintln!(
            "[cov-dump] HID2 base={base} t={t} row={} fnv={h:016x} l2={l2:.4} first4=[{v0:.5} {v1:.5} {v2:.5} {v3:.5}]",
            base + t - 1
        );
        Ok(())
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
        // **只留 query 窗**(末 W 行;从尾部按行偏移切)—— W 口径与
        // `coverage_analyze` 同源(`W_QUERY_TOKENS`),此处不得再写第二个数。
        let w = W_QUERY_TOKENS.min(t);
        let elem = d.dtype.size_bytes();
        let row_bytes = hq * hd * elem;
        let off = (t - w) * row_bytes;
        let len = w * row_bytes;
        let t_q = std::time::Instant::now();
        let mut q = Vec::with_capacity(q_blocks.len());
        {
            // **一条批命令**读完所有层的末 W 行(原实现逐层一次 `dtoh`
            // ⇒ 层数个往返;此处与 store 的 `dtoh_batch` 同律)
            for b in &q_blocks {
                if b.len < t * hq * hd {
                    return Err(ModelError::Msg(format!(
                        "覆盖率探针:Q 面元素数不符(块 {} < 应有 {})",
                        b.len,
                        t * hq * hd
                    )));
                }
            }
            let pieces: Vec<(&Bytes, usize, usize)> =
                q_blocks.iter().map(|b| (b, off, len)).collect();
            let face = self.graphs.session.face_mut();
            let t_io = std::time::Instant::now();
            let all = face.dtoh_batch(&pieces).await?;
            let io_ms = t_io.elapsed().as_secs_f64() * 1000.0;
            let t_dec = std::time::Instant::now();
            for chunk in all.chunks_exact(len) {
                let row = decode_row(d.dtype, chunk);
                if row.len() != w * hq * hd {
                    return Err(ModelError::Msg(format!(
                        "覆盖率探针:Q 面解码长不符 {} ≠ {}(dtype={:?})",
                        row.len(),
                        w * hq * hd,
                        d.dtype
                    )));
                }
                q.push(row);
            }
            let ids: Vec<u64> = q_blocks.iter().map(|b| b.id).collect();
            face.free(&ids).await?;
            self.prof.q_toh_ms += io_ms;
            self.prof.q_dec_ms += t_dec.elapsed().as_secs_f64() * 1000.0;
        }
        self.cov_pending = Some(CoveragePending {
            base,
            t,
            hq,
            hkv,
            hd,
            layers: q.len(),
            w,
            q,
        });
        // 计时归入“查询面”(Q 回读 + 解码;逆 RoPE 在 `coverage_analyze`)
        self.prof.q_face_ms += t_q.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }

    /// **召回点分析**(Q-2 第三段)。
    ///
    /// 前置:块边界事件已落(索引/快照已登记)⇒ 由 `execute_prefill` 在
    /// `capture_gdn_snap` 之后调用。`ctx_tokens` = 边界位置(页对齐)。
    ///
    /// # 两扇门(2026-10-19 拆分)
    ///
    /// | 面 | 门 | 内容 |
    /// |---|---|---|
    /// | **生产面** | `select_decode` | 打分 → 选择 → 视图页表 → 驻留调和 |
    /// | **诊断面** | `coverage_probe` | 真实质量(K/V 全量回读 + 逐行扫全 ctx)、覆盖率曲线/上界、论文口径对照、方案 A/B 对照、读数报告 |
    ///
    /// 拆之前两者共一门 ⇒ **低显存档想开功能就必须背诊断的账**
    /// (27B/8k 实测诊断 6.3s = 墙钟 32%,生产面只有 6%)。
    pub(crate) async fn coverage_analyze(
        &mut self,
        sid: u64,
        ctx_tokens: usize,
        last_boundary: bool,
    ) -> Result<()> {
        let t_all = std::time::Instant::now();
        let Some(pid) = self.cov_pending.take() else { return Ok(()) };
        self.prof.boundaries += 1;
        // **诊断面总门**:下面所有“重读数/保真对照/报告”都挂它。
        let diag = self.cfg.knobs.coverage_probe;
        // **方案对照档**(`OWL_SCHEME_PROBE=1`):多读一份 V + host 侧算三种方案的
        // 注意力输出(比"接进 decode 跑两遍"便宜得多,且**同一个口径**可比)
        // `OWL_SCHEME_MIN_CTX`(缺省 0 = 每个边界都算):只在这个 ctx 以上才算
        // ⇒ 规模档只跑最后一次(host 数学与 dense 成正比,小边界算了也白算)
        let scheme_min = owl_shared::env_reader::parse::<usize>("OWL_SCHEME_MIN_CTX").unwrap_or(0);
        let want_scheme = diag
            && owl_shared::env_reader::flag("OWL_SCHEME_PROBE")
            && ctx_tokens >= scheme_min;
        // **重诊断采样**(`OWL_COV_ANALYZE_EVERY`;三档):
        //
        // | 值 | 语义 |
        // |---|---|
        // | 未设(**缺省**) | **只在 prefill 末个边界**(报告读数就取自那一天) |
        // | `=k>0` | 每 k 块一次(曲线/回归需要多采点时用;`=1` = 每边界) |
        // | `=0` | 关了(只用生产面) |
        //
        // 为何缺省不是「每边界」:重诊断是**真重活** —— 单边界要
        // ① K/V 全量回读 + 逐元素解码 ② 逐 query 行 × (层,查询头) 对**全部已缓存
        // token** 做 softmax(`true_block_mass`)⇒ 成本含 `ctx` 因子 ⇒ 全军一遍
        // = **O(ctx²)**(2026-10-19 相位账实测:0.8B/3072 tok 下重诊断 ≈ **250ms/边界**,
        // 占 B 轴 95%)。它产出的 mass/curve/oracle/paper_coverage **全是诊断读数**;
        // 打分/选择/视图/驻留(生产面)一步不省。
        let analyze_every = owl_shared::env_reader::parse::<usize>("OWL_COV_ANALYZE_EVERY");
        // **分析边界目标**(`[kvmem].analyze_at_ctx`;0 = 最后一个边界):
        // 本边界恰好落在该 ctx ⇒ 算诊断(即使它不是最后一个边界)。
        // 为什么:边界位置随 chunk 漂移 ⇒ 钉住 ctx 才能跨 chunk 比读数。
        let at = self.cfg.knobs.coverage_analyze_at;
        let analyze_now = diag
            && if at != 0 {
                // **钉住的边界**:只在这一条 ctx 上算诊断(不再叠加"末边界",
                // 否则末边界会把它冲掉 —— 第一版就是叠加,读数仍随 chunk 漂移)。
                // `OWL_COV_ANALYZE_EVERY=k` 仍可显式要求多采点(调试用)。
                match analyze_every {
                    Some(0) => false,
                    Some(k) => (ctx_tokens / self.pool.page.max(1)) % k == 0,
                    None => ctx_tokens == at,
                }
            } else {
                match analyze_every {
                    Some(0) => false,
                    Some(k) => (ctx_tokens / self.pool.page.max(1)) % k == 0,
                    None => last_boundary,
                }
            };
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
        let t_mass = std::time::Instant::now();
        // **池页格式从池实况取**(单源;f16 / fp8 e4m3 同一寻址式)
        let fmt = crate::exec::kvpage::KvPageFormat::of_pool(self.pool.kv_fp8);
        let _ = fmt.elem_bytes(); // 元素宽已由 KvPageFormat 内部用(此处只留显式读口)
        let page_bytes = fmt.page_bytes(page, hkv, hd);
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
                // 页内序 = [token][head][dim](knhd_offset 单源);
                // 元素→f32 走 `KvPageFormat`(f16 / fp8 e4m3 两支)
                let mut krow = vec![0.0f32; hd];
                let mut vrow = vec![0.0f32; hd];
                for p in 0..page {
                    let tok = lb * page + p;
                    for h in 0..hkv {
                        let off = (tok * hkv + h) * hd;
                        fmt.decode_row(kbuf, p, h, hkv, hd, &mut krow)?;
                        dst[off..off + hd].copy_from_slice(&krow);
                        if let (Some(v), Some(vb)) = (vdst.as_deref_mut(), vbuf) {
                            fmt.decode_row(vb, p, h, hkv, hd, &mut vrow)?;
                            v[off..off + hd].copy_from_slice(&vrow);
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
        self.prof.mass_ms += t_mass.elapsed().as_secs_f64() * 1000.0;
        } // ← 重诊断(采样)段结束
        // ── ② 打分面:q̃(去 RoPE)+ 候选块链哈希 + Mean-K 索引 ──
        // **query 窗**(论文 §4.3:当前 step 的 query;**单 token 的注意力太局
        // 部**,末 token 常只盯前几个字)。取本 chunk **末 W 行**,逐行去 RoPE +
        // 逐行真实质量,再对窗取均值 —— 与论文 Eq.10 的 `Σ_m` 同构。
        let w = W_QUERY_TOKENS.min(pid.t);
        let first = pid.t - w;
        // **口径单源硬断**:待分析面只留了末 `w` 行(`coverage_take_q` 按同一
        // `W_QUERY_TOKENS` 切片)⇒ 两侧不得各写一个数,否则行列错位 = 静默错值。
        if pid.w != w {
            return Err(ModelError::Msg(format!(
                "覆盖率探针:query 窗口径不合(面里 {} 行 ≠ 应有 {w} 行)",
                pid.w
            )));
        }
        let t_qface = std::time::Instant::now();
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
            // 面内行号 = 绝对行号 − 窗口起点(面只存末 w 行)
            let q_off = (qi - first) * hq * hd;
            let mut flat = vec![0.0f32; layers * hq * hd];
            let mut q_post: Vec<Vec<f32>> = Vec::new();
            let mut row = vec![0.0f32; hq * hd];
            let mut qh = vec![0.0f32; hd];
            let t_derope = std::time::Instant::now();
            for l in 0..layers {
                row.copy_from_slice(&pid.q[l][q_off..q_off + hq * hd]);
                for h in 0..hq {
                    qh.copy_from_slice(&row[h * hd..(h + 1) * hd]);
                    owl_store::derope_halves_inplace(&mut qh, &cos, &sin)?;
                    flat[(l * hq + h) * hd..(l * hq + h + 1) * hd].copy_from_slice(&qh);
                }
                // post-RoPE 面**只为真实质量算**(诊断面)⇒ 不在采样档不克隆
                if analyze_now {
                    q_post.push(row.clone());
                }
            }
            self.prof.q_derope_ms += t_derope.elapsed().as_secs_f64() * 1000.0;
            face.push_token(flat)?;
            // **真实质量只在采样档算** —— 这是真重诊断:逐 query 行 ×
            // (层,查询头) 对**全部已缓存 token** 做 softmax ⇒ 单边界
            // `W × 层 × 头 × ctx × hd` 次 MAC(0.8B/3072 tok 实测 ≈ 每边界秒级)。
            // 它服务的 mass/curve/oracle/paper_coverage **全是诊断读数**;
            // 分数面(`scores`/选择/视图)不碰它。⇒ 与 KV 回读同门。
            if analyze_now {
                let t_m = std::time::Instant::now();
                let m = true_block_mass(&q_post, &k_post, (layers, hq, hkv, hd), page, pos + 1);
                // ⚠️ **临时取证**(`OWL_COV_DUMP=1`;A2 定谳用,事后删):逐 query 行的
                // 真实质量峰位 —— 报告里的 `mass` 是 W 行均值,掩盖“哪一行退化”。
                if owl_shared::env_reader::flag("OWL_COV_DUMP") {
                    let mut ord: Vec<usize> = (0..m.len()).collect();
                    ord.sort_by(|a: &usize, b: &usize| {
                        m[*b].partial_cmp(&m[*a]).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let top: Vec<String> =
                        ord.iter().take(5).map(|i| format!("{i}:{:.4}", m[*i])).collect();
                    eprintln!(
                        "[cov-dump] rowmass qi={qi} pos={pos} visible={} sum={:.4} top5={}",
                        pos + 1,
                        m.iter().sum::<f32>(),
                        top.join(" ")
                    );
                    // 最后一行的**逐层**质量 + 峰/均比(平不平一眼看出)
                    if qi == pid.t - 1 {
                        for l in 0..layers {
                            let ml = true_block_mass(
                                &q_post[l..l + 1],
                                &k_post[l..l + 1],
                                (1, hq, hkv, hd),
                                page,
                                pos + 1,
                            );
                            let mut ol: Vec<usize> = (0..ml.len()).collect();
                            ol.sort_by(|a: &usize, b: &usize| {
                                ml[*b].partial_cmp(&ml[*a]).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let tl: Vec<String> = ol
                                .iter()
                                .take(3)
                                .map(|i| format!("{i}:{:.4}", ml[*i]))
                                .collect();
                            let mean = ml.iter().sum::<f32>() / ml.len().max(1) as f32;
                            eprintln!(
                                "[cov-dump] rowlast 层={l} 峰/均={:.1} top3={}",
                                ml[ol[0]] / mean.max(1e-12),
                                tl.join(" ")
                            );
                        }
                    }
                }
                for (a, b) in mass.iter_mut().zip(m) {
                    *a += b;
                }
                self.prof.q_mass_ms += t_m.elapsed().as_secs_f64() * 1000.0;
                q_post_last = std::mem::take(&mut q_post);
            }
        }
        let denom = w as f32;
        for a in mass.iter_mut() {
            *a /= denom;
        }
        self.prof.q_face_ms += t_qface.elapsed().as_secs_f64() * 1000.0;
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
        // ⚠️ **临时取证**(`OWL_COV_DUMP=1`;A2 定谳用,事后删):三面指纹 ——
        // ① Q 面(本边界末 W 行,post-RoPE 原样)**② 池内 K 回读**(定块)
        // **③ 索引账 Mean-K**(同块)—— 三者谁随 chunk 漂,谁就是 A2 的源。
        if owl_shared::env_reader::flag("OWL_COV_DUMP") {
            let hf = |v: &[f32]| -> u64 {
                let mut h = 0xcbf2_9ce4_8422_2325u64;
                for x in v {
                    for b in x.to_le_bytes() {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x100_0000_01b3);
                    }
                }
                h
            };
            let hu = |v: &[u16]| -> u64 {
                let mut h = 0xcbf2_9ce4_8422_2325u64;
                for x in v {
                    for b in x.to_le_bytes() {
                        h ^= b as u64;
                        h = h.wrapping_mul(0x100_0000_01b3);
                    }
                }
                h
            };
            let l2 = |v: &[f32]| -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() };
            eprintln!(
                "[cov-dump] geom base={} t={} w={} ctx={ctx_tokens} page={page} blocks={n_blocks} layers={layers} hq={hq} hkv={hkv} hd={hd} analyze={analyze_now} last={last_boundary}",
                pid.base, pid.t, pid.w
            );
            for qi in 0..pid.w {
                let q_off = qi * hq * hd;
                let allq: Vec<f32> = pid
                    .q
                    .iter()
                    .flat_map(|l| l[q_off..q_off + hq * hd].to_vec())
                    .collect();
                let r0 = &pid.q[0][q_off..q_off + hd];
                let per_l: Vec<String> = pid
                    .q
                    .iter()
                    .map(|l| format!("{:.1}", l2(&l[q_off..q_off + hq * hd])))
                    .collect();
                let per_l_fnv: Vec<String> = pid
                    .q
                    .iter()
                    .map(|l| format!("{:08x}", (hf(&l[q_off..q_off + hq * hd]) & 0xffff_ffff) as u32))
                    .collect();
                eprintln!(
                    "[cov-dump] Q qi={qi} pos={} fnv={:016x} l2={:.4} per层l2=[{}] per层fnv=[{}] l0h0=[{:.5} {:.5} {:.5} {:.5}]",
                    pid.base + (pid.t - pid.w) + qi,
                    hf(&allq),
                    l2(&allq),
                    per_l.join(" "),
                    per_l_fnv.join(" "),
                    r0[0],
                    r0[1],
                    r0[2],
                    r0[3]
                );
            }
            if analyze_now {
                for lb in [0usize, 1, 2, 17, 36, 56, 95] {
                    if lb >= n_blocks {
                        continue;
                    }
                    let lo = lb * page * hkv * hd;
                    let hi = lo + page * hkv * hd;
                    let allk: Vec<f32> =
                        k_post.iter().take(layers).flat_map(|l| l[lo..hi].to_vec()).collect();
                    let k0 = &k_post[0][lo..lo + hd];
                    let (idxn, idxv) = match index.block_k_bar(cands[lb]) {
                        Some(seg) => {
                            let row: Vec<f32> = seg[..hd]
                                .iter()
                                .map(|b| owl_store::f16_to_f32_pub(*b))
                                .collect();
                            (
                                format!("fnv={:016x} l2={:.4}", hu(seg), l2(&row)),
                                format!(
                                    "[{:.4} {:.4} {:.4} {:.4}]",
                                    row[0], row[1], row[2], row[3]
                                ),
                            )
                        }
                        None => ("无条目".to_string(), String::new()),
                    };
                    eprintln!(
                        "[cov-dump] blk={lb} K fnv={:016x} l2={:.4} t0h0=[{:.5} {:.5} {:.5} {:.5}] ‖ IDX {idxn} {idxv}",
                        hf(&allk),
                        l2(&allk),
                        k0[0],
                        k0[1],
                        k0[2],
                        k0[3]
                    );
                }
            }
        }
        let gqa = gqa_map(hq, hkv);
        // ── **召回节奏**(`OWL_RECALL_EVERY`;缺省 1 = 每边界都完整召回)──
        //
        // 完整召回 = 打分(线性) + 按分选择;廉价召回 = **不打分的近期选择**
        // (sink+recent+从尾往前填,与参考实现的 prefill 压力路径同构)。
        // 两者**都要**走后面的驻留调和与视图装配 —— 设备侧不会因为跳过一个
        // 边界就跟视图对不上;变的只是“这一边界的**选择名单**要不要重新算”。
        //
        // **末个边界永远是完整召回** —— decode 消费的视图就取自它。
        let recall_every = self.cfg.knobs.recall_every.max(1);
        // 用**边界序号**而不是块号做步进:块号是页宽的整数倍(边界只在页对齐处),
        // 拿它取模会退化 —— 当页宽本身是 k 的倍数时**每个**边界都会命中
        // (chunk 与页对齐时实测 k=4 仍然 15/15 全完整召回)。
        //
        // `analyze_now` **隐含完整召回**:要出诊断读数(mass/curve/oracle/paper)
        // 的那条边界必须有当场的分数与选中集,否则读数无主。
        let recall_now =
            analyze_now || last_boundary || self.prof.boundaries % recall_every == 0;
        if recall_now {
            self.prof.recall_full += 1;
        } else {
            self.prof.recall_cheap += 1;
        }
        // **廉价边界 = 不重新打分,但仍要调和驻留**(两个反面教材都实测过,已废):
        // ① "近期选择 + 调和":近期名单与上次打分名单**互相覆盖** ⇒ 每次完整召回
        //    都要把上次沉下去的回灌一遍(调和 +92%、下沉/回灌翻倍);
        // ② "什么都不做":`ensure_for_len` **驱逐不了会话自己持有的页**(引用在
        //    会话账上)⇒ 链每 chunk 要新增 `chunk/page` 页,不腾就必然 `Exhausted`
        //    (实测:池 96 页、链要 100 页 ⇒ "KV 块池耗尽")。
        // ⇒ 正解:**冻结上次完整召回的选择 + 收进 sink/新尾部**,下游调和/视图照走。
        // ── **B-1 召回点**:打分(快路口径)+ 选择(预算/sink/recent)+ 实时读数 ──
        // 快路 = `ScoreMode::LogitMean`:两层线性折叠(窗 Σ_m + GQA 头 Σ_h),
        // 与论文口径**仅 FP 重排**(排序相同)但便宜 50×~24×(见 store 台账)。
        let t_sc = std::time::Instant::now();
        let scores = if recall_now {
            owl_store::score_blocks(
                index,
                &face,
                &cands,
                &gqa,
                owl_store::ScoreTuning { scale: 0.0, mode: owl_store::ScoreMode::LogitMean },
            )?
        } else {
            Vec::new()
        };
        let score_ms = t_sc.elapsed().as_secs_f64() * 1000.0;
        self.prof.score_ms += score_ms;
        let (budget_blocks, sink, recent) = self.recall_budget();
        let t_sel = std::time::Instant::now();
        let scored: Vec<(owl_store::ChainHash, f32)> = cands.iter().copied().zip(scores.iter().copied()).collect();
        let budget = owl_store::SelectBudget::new(budget_blocks.min(n_blocks), sink, recent);
        let set = if recall_now {
            let s = owl_store::select(&scored, &cands, budget);
            // 冻结名单(供廉价边界复用)+ 记账
            self.sel_keep = s.blocks.clone();
            s
        } else {
            // **廉价档:冻结上次完整召回的选择 + sink + 新尾部**(不打分)
            let head = self.cfg.knobs.select_sink;
            let tail = self.cfg.knobs.select_recent.max(1);
            let mut keep: Vec<owl_store::ChainHash> = Vec::with_capacity(self.sel_keep.len() + tail + head);
            for c in cands.iter().take(head.min(cands.len())) {
                if !keep.contains(c) {
                    keep.push(*c);
                }
            }
            for c in cands.iter().skip(cands.len().saturating_sub(tail)) {
                if !keep.contains(c) {
                    keep.push(*c);
                }
            }
            for c in &self.sel_keep {
                if !keep.contains(c) {
                    keep.push(*c);
                }
            }
            let all = n_blocks;
            keep.sort_by_key(|c| cands.iter().position(|x| x == c).unwrap_or(usize::MAX));
            owl_store::BlockSet {
                dropped: all.saturating_sub(keep.len()),
                reserved: (head + tail).min(keep.len()),
                blocks: keep,
            }
        };
        let select_ms = t_sel.elapsed().as_secs_f64() * 1000.0;
        self.prof.select_ms += select_ms;
        // **B-4 召回信号**:把"本步被注意力选中"的块告诉策略面
        // (attention-aware LRFU 的 keep-score;LRU 策略 = 无操作 ⇒ 默认路径零影响)。
        // ⚠️ **只在完整召回档发信号**:廉价档的名单是**纯近期序**推出来的,不含任何
        // 注意力信息 ⇒ 拿它喂 LRFU 等于用"最近"冒充"被注意力用过"(自激回路)。
        if recall_now {
            self.bank.rel.note_recall(&set.blocks);
        }
        let picked: Vec<usize> = set
            .blocks
            .iter()
            .filter_map(|b| cands.iter().position(|c| c == b))
            .collect();
        let coverage = coverage_of_set(&mass, &picked);
        let oracle_coverage = oracle_at_budget(&mass, budget_blocks.min(n_blocks));
        // 论文口径(softmax 均值)在同预算下的覆盖 —— **保真对照(诊断面)**。
        //
        // 为什么不每边界都算(2026-10-19 性能修正):这是慢路 —— store 的
        // release 延迟门实测 27B 几何下 **1.45ms/块 = 快路的 53×**(8192 块
        // 11905ms vs 222ms)。而它报的 `paper_coverage` 是**相对于 `mass`(真实
        // 注意力质量)的覆盖** —— `mass` 本身只在 `analyze_now` 采样档才读得到
        // (缺它 => `coverage_of_set` 恒 0)。所以采样外重算它 = **纯白烧**,
        // 且算出来的 0 还会把读数弄脏。⇒ 与 `mass` 同门(这也正是本段上方
        // 「重诊断采样」注释里已经写下的意图)。
        let (paper_coverage, paper_ms) = if analyze_now {
            let t_paper = std::time::Instant::now();
            let paper_scores = owl_store::score_blocks(
                index,
                &face,
                &cands,
                &gqa,
                owl_store::ScoreTuning::default(),
            )?;
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
            (
                coverage_of_set(&mass, &paper_picked),
                t_paper.elapsed().as_secs_f64() * 1000.0,
            )
        } else {
            // 采样外不重算:沿用上一次采样边界的读数(`mass` 不在场时这个数
            // 本来就不可解释;沿用 > 拿 0 冒充)。
            (self.cov_last.as_ref().map(|r| r.recall.paper_coverage).unwrap_or(0.0), 0.0)
        };
        self.prof.paper_ms += paper_ms;
        // ── B-3e-2 第 2 项:召回点的**下沉 / 回灌**(部分驻留档)──
        // 被踢出工作集的链块 ⇒ `decref`(放开会话引用)+ 逐块下沉 + 表写占位;
        // 选中但内容在宿主的块 ⇒ 回灌 + 写回槽(复用 turn 起点那套 plan/swap/commit)。
        // 先沉后升:沉的这一步**腾出的槽**正是升要用的槽。
        if self.cfg.knobs.select_partial {
            let t_rec = std::time::Instant::now();
            let (sunk, staged) = self.reconcile_residency(sid, &cands, &set.blocks, n_blocks).await?;
            self.prof.reconcile_ms += t_rec.elapsed().as_secs_f64() * 1000.0;
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
        let t_view = std::time::Instant::now();
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
        self.prof.view_ms += t_view.elapsed().as_secs_f64() * 1000.0;
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
                    let off = (w - 1) * hq * hd;
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
        // **只有诊断真跑了的边界才更新报告**(2026-10-19)。
        //
        // 原实现每边界都覆盖 ⇒ 后到的边界会把**目标边界**的报告冲掉 ——
        // 即使那条边界没算诊断(mass 为空/取自别的 ctx),`cov_last` 也会被换成
        // 它的读数,于是"钉住分析边界"失效(实测仍随 chunk 漂移:3072/3584/3840)。
        // 现在语义清楚:`cov_last` = **最后一次真诊断**的读数。
        if analyze_now {
            self.cov_last = Some(rep);
        }
        self.prof.total_ms += t_all.elapsed().as_secs_f64() * 1000.0;
        Ok(())
    }
}
