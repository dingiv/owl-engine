//! DFlash2 proposal distribution q（NInfer `proposal_q` 同构）。
//!
//! 选择器码本 walk 的真实分布：每一步在 top-16 候选上做
//! `softmax(score / temperature)`，从 q 抽签得到 draft；拒绝采样再用
//! `min(1, p(d)/q(d))` 接受，拒绝时从 `max(p-q,0)` 归一化抽 correction。
//! greedy 档 q 未使用；lookup 改写草稿后 q 退化为该 token 的 one-hot
//! （lookup 是 host 侧确定性改写，不是选择器分布）。

use owl_models::ModelError;

/// 每步候选数（DFlash2 selector_top_k）。
pub(crate) const CAND: usize = 16;

/// 目标/提案 RNG 的用途分隔（与 sampler 的位置派生态同构）。
const PURPOSE_PROPOSAL: u64 = 0x9E37_79B9_7F4A_7C15;
const PURPOSE_ACCEPT: u64 = 0xD1B5_4A32_D192_ED03;
const PURPOSE_CORRECTION: u64 = 0xA24B_AED4_963E_E407;

/// 一步 q 的 one-hot 覆写（lookup 命中后）。
#[derive(Clone, Copy, Debug)]
struct OneHot {
    token: u32,
    prob: f32,
}

/// 本轮 K 步的 proposal q。
#[derive(Clone, Debug)]
pub(crate) struct DraftQ {
    /// 与 q 同步的实际 draft（lookup reconcile 后更新）。
    drafts: Vec<u32>,
    /// [K, 16] 候选全局 id。
    cand: Vec<u32>,
    /// [K, 16] q（对 cand 的归一化概率）。
    q: Vec<f32>,
    /// 每步 one-hot 覆写（None = 用 q/cand）。
    one_hot: Vec<Option<OneHot>>,
    /// 每步是否被**请求的 top-k/top-p** 截断（C3 观测面）
    trunc: Vec<bool>,
    /// 被截断丢掉的质量（逐轮累加；0 = 请求没截断草稿支撑）
    dropped: f32,
    /// **每步实际使用的那一行原始分数**（`[K,16]`；已乘 α，未过采样变换）。
    /// 只为标定测量（`[spec] q_dump`）保留——生产路径不读。
    raw: Vec<f32>,
}

/// **`DraftQ::from_scores` 的显式输入包**（走格事实 + RNG 定位）。
///
/// 分三个正交的面：①分数表（`scores`/`unary`/`cand`/`depth`）
/// ②走格策略（`walk`）③采样/RNG（`cfg`/`seed`/`base_pos`）。
/// 打包而不是八个位置参数，是为了让调用点写清"哪张表喂哪个模式"。
pub(crate) struct QInput<'a> {
    /// `[depth, 16, 16]` 边分（`scores[e, prev, c]`）。
    pub scores: &'a [f32],
    /// `[depth, 16]` 候选全局 id。
    pub cand: &'a [u32],
    pub depth: usize,
    /// `[depth, 16]` 逐位置边际（选择器 top-k 值半区；`walk = Unary` 时必需）。
    pub unary: Option<&'a [f32]>,
    /// 走格读哪一行（`[spec] dflash_walk`）。
    pub walk: owl_shared::config::DflashWalk,
    pub cfg: &'a crate::sampler::SamplerCfg,
    pub seed: u64,
    pub base_pos: usize,
    /// **草稿 q 的 logit 缩放**（`[spec] q_alpha`;1.0 = 原律）。
    /// 选择器边分不是校准 logits ⇒ 默认 softmax 近乎 one-hot ⇒ 拒绝采样的
    /// 每步接受概率退化为 `p(argmax)`。乘 α 把 q 摊开（α→0 = 候选集上均匀）。
    pub alpha: f32,
    /// **候选集上的均匀地板**（`[spec] q_floor`;0 = 不加）。
    pub floor: f32,
}

/// 选择器格"第 e 步读哪一行"（单一事实源：采样走格与贪心走格同用）。
fn row_of<'a>(
    scores: &'a [f32],
    unary: &'a [f32],
    e: usize,
    prev: usize,
    walk: owl_shared::config::DflashWalk,
) -> &'a [f32] {
    use owl_shared::config::DflashWalk as W;
    let (src, base) = match walk {
        // 逐位置边际：丢弃边缘分（DDTree 的 q_i(v)）
        W::Unary => (unary, e * CAND),
        // 固定参考前驱 ⇒ 表里第 0 行（`cand[e-1][0]`）
        W::FixedPred => (scores, e * CAND * CAND),
        // 条件格：上一步**选中**的前驱
        W::Conditional => (scores, (e * CAND + prev) * CAND),
    };
    &src[base..base + CAND]
}

/// 走格形状门（`from_scores` / [`walk_greedy`] 共用同一判据）。
fn check_walk_shape(
    unary: &[f32],
    depth: usize,
    walk: owl_shared::config::DflashWalk,
) -> Result<(), ModelError> {
    if depth == 0 || depth > CAND - 1 {
        return Err(ModelError::Msg(format!(
            "proposal_q: depth={depth} 越界（1..=15）"
        )));
    }
    if walk == owl_shared::config::DflashWalk::Unary && unary.len() < depth * CAND {
        return Err(ModelError::Msg(format!(
            "proposal_q: walk=unary 需要逐位置边际 unary[{depth},{CAND}]（实得 {}）",
            unary.len()
        )));
    }
    Ok(())
}

/// **贪心走格**（纯函数；设备核 `owl_dflash_select_*` 的 walk 同律）。
///
/// 返回每步选中的**候选下标**（`0..CAND`）。平局取最小下标，与核内
/// `if (row[c] > best)` 的严格大于同律；整行非有限 = 批次坏，响亮拒。
///
/// 用途：①`dflash_walk != conditional` 时在收割面重走格（TP2 词表切分档）
/// ②门：`conditional` 臂必须与设备 `sel[..depth]` 逐位相同（一致性自检）。
pub(crate) fn walk_greedy(
    scores: &[f32],
    unary: &[f32],
    depth: usize,
    walk: owl_shared::config::DflashWalk,
) -> Result<Vec<usize>, ModelError> {
    check_walk_shape(unary, depth, walk)?;
    if scores.len() < depth * CAND * CAND {
        return Err(ModelError::Msg(format!(
            "proposal_q: walk 槽尺寸不符 scores={}（需 {}×{}²）",
            scores.len(),
            depth,
            CAND
        )));
    }
    let mut out = Vec::with_capacity(depth);
    let mut prev = 0usize;
    for e in 0..depth {
        let row = row_of(scores, unary, e, prev, walk);
        if !row.iter().any(|s| s.is_finite()) {
            return Err(ModelError::Msg(format!(
                "proposal_q: 第 {e} 步边分全非有限（selector 输出坏）"
            )));
        }
        let mut best = row[0];
        let mut bi = 0usize;
        for (c, &s) in row.iter().enumerate().skip(1) {
            if s > best {
                best = s;
                bi = c;
            }
        }
        prev = bi;
        out.push(bi);
    }
    Ok(out)
}

/// 收割面重走格：候选**下标** → token id（`[spec] dflash_walk` 非 conditional 时用）。
pub(crate) fn walk_tokens(
    scores: &[f32],
    unary: &[f32],
    cand: &[u32],
    depth: usize,
    walk: owl_shared::config::DflashWalk,
) -> Result<Vec<u32>, ModelError> {
    let idx = walk_greedy(scores, unary, depth, walk)?;
    Ok(idx
        .iter()
        .enumerate()
        .map(|(e, &i)| cand[e * CAND + i])
        .collect())
}

/// **一致性自检**：host `Conditional` 走格是否与设备 `sel[..depth]` 逐位相同。
///
/// 消融臂的所有差异必须只有"读哪一行"这一项；若条件臂本身对不上设备，
/// 那么 `fixed_pred`/`unary` 的数就是两个变量混在一起，不可归因。
pub(crate) fn walk_matches_device(
    scores: &[f32],
    unary: &[f32],
    cand: &[u32],
    dev_tokens: &[u32],
    depth: usize,
) -> Result<bool, ModelError> {
    let idx = walk_greedy(scores, unary, depth, owl_shared::config::DflashWalk::Conditional)?;
    Ok(idx.len() == dev_tokens.len()
        && idx
            .iter()
            .enumerate()
            .all(|(e, &i)| cand.get(e * CAND + i).copied() == dev_tokens.get(e).copied()))
}

impl DraftQ {
    /// 从选择器 `scores [K,16,16]` 与 `cand [K,16]` 构造并抽签。
    ///
    /// 布局与 `owl_dflash_select_*` 一致：`scores[e, prev, c]` 是第 e 步、
    /// 前驱候选 rank=prev 时选 c 的边分。缺省 `dflash_walk = Conditional`
    /// = 用**实际抽中的前驱**（NInfer 的 conditional walk 同义）；
    /// `FixedPred` / `Unary` 是 AL 消融臂（见 [`row_of`]）。
    pub(crate) fn from_scores(inp: QInput<'_>) -> Result<Self, ModelError> {
        let QInput {
            scores,
            cand,
            depth,
            unary,
            walk,
            cfg,
            seed,
            base_pos,
            alpha,
            floor,
        } = inp;
        let unary = unary.unwrap_or(&[]);
        check_walk_shape(unary, depth, walk)?;
        if !(cfg.temp.is_finite() && cfg.temp >= 0.0) {
            return Err(ModelError::Msg(format!(
                "proposal_q: temperature={} 非正/非有限",
                cfg.temp
            )));
        }
        if scores.len() < depth * CAND * CAND || cand.len() < depth * CAND {
            return Err(ModelError::Msg(format!(
                "proposal_q: 槽尺寸不符 scores={} cand={}（需 {}×{}² / {}×{}）",
                scores.len(),
                cand.len(),
                depth,
                CAND,
                depth,
                CAND
            )));
        }
        let mut qv = vec![0f32; depth * CAND];
        let mut raw_rows = vec![0f32; depth * CAND];
        let mut drafts = vec![0u32; depth];
        let mut trunc = vec![false; depth];
        let mut dropped = 0f32;
        let mut prev = 0usize;
        for e in 0..depth {
            let row = row_of(scores, unary, e, prev, walk);
            if !row.iter().any(|s| s.is_finite()) {
                return Err(ModelError::Msg(format!(
                    "proposal_q: 第 {e} 步边分全非有限（selector 输出坏）"
                )));
            }
            // **q 与 p 同一套变换**（temp → top-k → softmax → top-p → 质量重均摊）。
            //
            // 为什么必须同源（2026-11-03 C2）：请求的 top-k/top-p 把目标分布 p
            // 截断到它的支撑上；若 q 仍在被截掉的候选上留质量，抽到那里的草稿
            // p̃=0 ⇒ **必然整轮拒**（vLLM 同款注释："mass outside support =
            // guaranteed rejection"）。截断后 q 的支撑 ⊆ p̃ 的支撑，白付消失。
            // 精确性不受影响：q 可以是任意分布（拒绝采样对 q 无要求）。
            let mut vals: Vec<f32> = row.to_vec();
            if !(alpha == 1.0) {
                for v in vals.iter_mut() {
                    *v *= alpha;
                }
            }
            raw_rows[e * CAND..(e + 1) * CAND].copy_from_slice(&vals);
            let mut dist = crate::sampler::distribution(&mut vals, cfg, &[]);
            if dist.is_empty() {
                return Err(ModelError::Msg(format!(
                    "proposal_q: 第 {e} 步采样变换返回空分布"
                )));
            }
            // 均匀地板:支撑不变（仍 ∈ 请求截断后的候选集）,只是把质量往平坦推。
            // 对高熵负载（创作）这是“把 q 从 one-hot 拉回来”的最便宜形式。
            if floor > 0.0 {
                let u = floor / dist.len() as f32;
                for (_, p) in dist.iter_mut() {
                    *p = *p * (1.0 - floor) + u;
                }
            }
            // C3 记账：请求截断丢掉了多少草稿支撑质量
            let kept: Vec<u32> = dist.iter().map(|(i, _)| *i).collect();
            let d = truncated_mass(&vals, cfg, &kept);
            if d > 1e-6 {
                trunc[e] = true;
                dropped += d;
            }
            for (idx, p) in &dist {
                qv[e * CAND + *idx as usize] = *p;
            }
            // 抽样：与旧实现同一个 RNG 派生态（位置定序，可复现）
            let u = uniform(seed, base_pos as u64 + e as u64, PURPOSE_PROPOSAL);
            let mut acc = 0f32;
            let mut pick = dist[dist.len() - 1].0 as usize;
            for (idx, p) in &dist {
                acc += *p;
                if u < acc {
                    pick = *idx as usize;
                    break;
                }
            }
            prev = pick;
            drafts[e] = cand[e * CAND + pick];
        }
        Ok(Self {
            drafts,
            cand: cand.to_vec(),
            q: qv,
            one_hot: vec![None; depth],
            trunc,
            dropped,
            raw: raw_rows,
        })
    }

    /// 被请求 top-k/top-p 截断的步数（C3 观测面；>0 正常）。
    pub(crate) fn truncated_steps(&self) -> usize {
        self.trunc.iter().filter(|t| **t).count()
    }

    /// 被丢掉的质量（逐轮累加；持续 >1e-3 说明草稿头分布很平）。
    pub(crate) fn dropped_mass(&self) -> f32 {
        self.dropped
    }

    pub(crate) fn drafts(&self) -> &[u32] {
        &self.drafts
    }

    /// **标定用**：每步使用的原始分数行（`[K,16]`，已乘 α）与候选 id 表。
    pub(crate) fn raw_rows(&self) -> &[f32] {
        &self.raw
    }

    pub(crate) fn cands(&self) -> &[u32] {
        &self.cand
    }

    pub(crate) fn drafts_mut(&mut self) -> &mut [u32] {
        &mut self.drafts
    }

    /// lookup 改写后强制某步 one-hot（`token` 不在候选集也合法）。
    pub(crate) fn force_one_hot(&mut self, e: usize, token: u32) {
        if e < self.one_hot.len() {
            self.one_hot[e] = Some(OneHot { token, prob: 1.0 });
            self.drafts[e] = token;
        }
    }

    /// q(v | 第 e 步)。lookup 覆写后 = one-hot；否则扫 16 候选。
    pub(crate) fn q_of(&self, e: usize, token: u32) -> f32 {
        if let Some(Some(oh)) = self.one_hot.get(e) {
            return if oh.token == token { oh.prob } else { 0.0 };
        }
        let base = e * CAND;
        self.cand[base..base + CAND]
            .iter()
            .zip(self.q[base..base + CAND].iter())
            .filter(|(t, _)| **t == token)
            .map(|(_, p)| *p)
            .sum()
    }

    /// lookup 改写后：把 q 退化为实际 draft 的 one-hot（确定性 proposal）。
    /// 只改与 `current` 不一致的步；返回被覆写步数。
    // 结构性豁免(2026-11-02 clippy 清理):循环变量是**维度序**(多数组同索引 / 手工归约要显式序);zip 会藏住对应关系
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn reconcile(&mut self, current: &[u32]) -> usize {
        let n = current.len().min(self.drafts.len());
        let mut hit = 0usize;
        for e in 0..n {
            if current[e] != self.drafts[e] {
                self.one_hot[e] = Some(OneHot { token: current[e], prob: 1.0 });
                self.drafts[e] = current[e];
                hit += 1;
            }
        }
        hit
    }

}


/// 请求的 top-k/top-p 从**这一步的草稿边分**里砍掉的质量（C3 记账）。
///
/// 口径 = 截断前 softmax(边分 / T) 中落在保留集之外的质量和；
/// `temp = 0`（贪心档）没有"质量"概念 ⇒ 恒 0。
fn truncated_mass(row: &[f32], cfg: &crate::sampler::SamplerCfg, kept: &[u32]) -> f32 {
    if !(cfg.temp.is_finite() && cfg.temp > 0.0) {
        return 0.0;
    }
    let mx = row
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !mx.is_finite() {
        return 0.0;
    }
    let mut ps = [0f32; CAND];
    let mut sum = 0f32;
    for (c, &v) in row.iter().enumerate().take(CAND) {
        let p = if v.is_finite() { ((v - mx) / cfg.temp).exp() } else { 0.0 };
        ps[c] = p;
        sum += p;
    }
    if !(sum.is_finite() && sum > 0.0) {
        return 0.0;
    }
    let mut dropped = 0f32;
    for (c, p) in ps.iter().enumerate().take(CAND) {
        if !kept.contains(&(c as u32)) {
            dropped += p / sum;
        }
    }
    dropped
}

/// 拒绝采样接受（NInfer `speculative_accept_sparse_drafts` 的 owl 版）。
///
/// `rows_n = depth + 1`，verify 行 0..depth；返回 `(m, emitted, bonus)`。
/// 目标分布用 `sampler::distribution`（与裸 decode 同一套变换）；q 由
/// selector 边分给出。全接受时 bonus 从最后一行目标分布抽。
#[allow(clippy::too_many_arguments)]
pub(crate) fn accept_rejection(
    logits: &[f32],
    vocab: usize,
    rows_n: usize,
    drafts: &[u32],
    depth: usize,
    q: &DraftQ,
    cfg: &crate::sampler::SamplerCfg,
    hist: &mut Vec<u32>,
    seed_base: u64,
    pos: usize,
) -> Result<(usize, Vec<u32>, u32), ModelError> {
    if rows_n != depth + 1 || drafts.len() < depth {
        return Err(ModelError::Msg(format!(
            "spec rejection: rows_n={rows_n} depth={depth} drafts={} 不闭合",
            drafts.len()
        )));
    }
    if logits.len() < rows_n * vocab {
        return Err(ModelError::Msg(format!(
            "spec rejection: logits {} < {rows_n}×{vocab}",
            logits.len()
        )));
    }
    let mut emitted: Vec<u32> = Vec::with_capacity(rows_n);
    for i in 0..depth {
        let mut scratch = logits[i * vocab..(i + 1) * vocab].to_vec();
        let p = crate::sampler::distribution(&mut scratch, cfg, hist);
        let d = drafts[i];
        let pd = p.iter().find(|(t, _)| *t == d).map(|(_, p)| *p).unwrap_or(0.0);
        let qd = q.q_of(i, d);
        if !(qd.is_finite() && qd > 0.0) {
            return Err(ModelError::Msg(format!(
                "spec rejection: 第 {i} 步 draft={d} 的 q={qd} 非正/非有限（proposal_q 与 drafts 不同源）"
            )));
        }
        let accept = (pd / qd).clamp(0.0, 1.0);
        if accept_uniform(seed_base, pos + i) < accept {
            emitted.push(d);
            hist.push(d);
            continue;
        }
        let corr = residual_sample(&p, q, i, seed_base, pos)?;
        emitted.push(corr);
        hist.push(corr);
        return Ok((i, emitted, corr));
    }
    // 全接受：bonus 从第 depth 行目标分布抽（hist 已含全部 draft）。
    let mut scratch = logits[depth * vocab..(depth + 1) * vocab].to_vec();
    let mut rng = seed_base ^ ((pos + depth) as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let bonus = crate::sampler::sample(&mut scratch, cfg, &mut rng, hist);
    emitted.push(bonus);
    Ok((depth, emitted, bonus))
}

/// 拒绝采样用：`max(p-q, 0)` 归一化抽 correction。
///
/// `p` = [`crate::sampler::distribution`] 的稀疏目标分布。q 的支撑只在
/// 候选集（或 lookup one-hot），因此残差在 p 的支撑上逐项比较即可。
pub(crate) fn residual_sample(
    p: &[(u32, f32)],
    q: &DraftQ,
    e: usize,
    seed: u64,
    base_pos: usize,
) -> Result<u32, ModelError> {
    let mut weights: Vec<(u32, f32)> = Vec::with_capacity(p.len());
    let mut sum = 0f32;
    for &(tok, pp) in p {
        let w = (pp - q.q_of(e, tok)).max(0.0);
        if w > 0.0 {
            sum += w;
            weights.push((tok, w));
        }
    }
    if !(sum.is_finite() && sum > 0.0) {
        // **数值兜底（参考实现同律）**：残差被减尽/下溢（q 与 p 在浮点上重合，
        // 典型出现在 `q(d)/p(d)` 因舍入略小于 1 而被拒的极窄窗口）——直接按**目标
        // 分布 p** 抽一个 correction token，保证任何数值情形下都有合法输出。
        //
        // 为什么不报错：这是可恢复的浮点边界，不是不变量失守；把整个请求 abort
        // 掉代价太大。但也**不静默**：记 `spec.residual.fallback`，一旦非零就说明
        // q 已经贴到 p（或 q 口径出了问题），可观测。
        super::mcnt("spec.residual.fallback", 1);
        let mut rng = seed ^ ((base_pos as u64 + e as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9));
        return Ok(crate::sampler::sample_sparse(p, &mut rng));
    }
    let u = uniform(seed, base_pos as u64 + e as u64, PURPOSE_CORRECTION);
    let mut acc = 0f32;
    for (tok, w) in &weights {
        acc += *w / sum;
        if u < acc {
            return Ok(*tok);
        }
    }
    Ok(weights.last().map(|(t, _)| *t).unwrap_or(0))
}

/// 接受门 RNG（与 proposal/target 抽样分离）。
pub(crate) fn accept_uniform(seed: u64, pos: usize) -> f32 {
    uniform(seed, pos as u64, PURPOSE_ACCEPT)
}

fn uniform(seed: u64, pos: u64, purpose: u64) -> f32 {
    // splitmix64：无状态、跨平台确定；24 位尾数足够采样。
    let mut z = seed
        .wrapping_add(pos.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(purpose);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32) / (1u64 << 24) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 无截断的请求档（temp=1 / top-k 覆盖 16 候选 / top-p 关）
    fn cfg_plain() -> crate::sampler::SamplerCfg {
        crate::sampler::SamplerCfg { temp: 1.0, topk: 32, topp: 1.0, rep_penalty: 1.0 }
    }

    fn scores(depth: usize) -> Vec<f32> {
        (0..depth * CAND * CAND).map(|i| ((i * 17) % 29) as f32 / 4.0).collect()
    }

    /// 采样走格的输入包（缺省 `conditional` = 生产律）
    fn qi<'a>(
        scores: &'a [f32],
        cand: &'a [u32],
        depth: usize,
        cfg: &'a crate::sampler::SamplerCfg,
        seed: u64,
        pos: usize,
    ) -> QInput<'a> {
        QInput {
            scores,
            cand,
            depth,
            unary: None,
            walk: owl_shared::config::DflashWalk::Conditional,
            cfg,
            seed,
            base_pos: pos,
            alpha: 1.0,
            floor: 0.0,
        }
    }

    /// 三臂分数表：`conditional` 与 `fixed_pred` **必然不同**。
    ///
    /// · 第 0 步（row 0）最大 = c=3；
    /// · 第 1 步 row **3**（条件于选中的 3）最大 = c=7；
    /// · 第 1 步 row **0**（固定参考前驱）最大 = c=11。
    ///
    /// ⇒ conditional = [3,7]、fixed_pred = [3,11]。其余位置全 -10（平局取小下标）。
    fn split_table() -> Vec<f32> {
        let e1 = CAND * CAND; // 第 1 步的基址
        let mut sc = vec![-10.0f32; 2 * CAND * CAND];
        sc[3] = 10.0; // e=0、row 0、c=3
        sc[e1 + 3 * CAND + 7] = 10.0; // e=1、row 3（条件于选中 3）、c=7
        sc[e1 + 11] = 10.0; // e=1、row 0（固定参考前驱）、c=11
        sc
    }

    /// **S0 门：条件格 vs 固定前驱 vs 纯边际**（AL 消融臂的语义锁）。
    ///
    /// 三臂必须读不同的行 —— 否则 A/B 量的是同一个东西。
    #[test]
    fn walk_arms_read_different_rows() {
        use owl_shared::config::DflashWalk as W;
        let sc = split_table();
        // 边际：逐位置 argmax（与边缘分无关）
        let mut unary = vec![0f32; 2 * CAND];
        unary[5] = 1.0;
        unary[CAND + 9] = 1.0;

        assert_eq!(walk_greedy(&sc, &unary, 2, W::Conditional).unwrap(), vec![3, 7]);
        assert_eq!(walk_greedy(&sc, &unary, 2, W::FixedPred).unwrap(), vec![3, 11]);
        assert_eq!(walk_greedy(&sc, &unary, 2, W::Unary).unwrap(), vec![5, 9]);
    }

    /// **平局取最小下标**（与设备核 `if (row[c] > best)` 严格大于同律）。
    #[test]
    fn walk_ties_take_min_index() {
        use owl_shared::config::DflashWalk as W;
        let sc = vec![0.0f32; 3 * CAND * CAND];
        assert_eq!(walk_greedy(&sc, &[], 3, W::Conditional).unwrap(), vec![0, 0, 0]);
    }

    /// `unary` 臂缺边际表 ⇒ 响亮拒（不静默退成条件格）。
    #[test]
    fn unary_walk_requires_marginals() {
        use owl_shared::config::DflashWalk as W;
        let e = walk_greedy(&split_table(), &[], 2, W::Unary).unwrap_err();
        assert!(format!("{e}").contains("unary"), "{e}");
    }

    /// **一致性自检门**：`walk_matches_device` 只在 token 逐位相同时为真。
    #[test]
    fn walk_matches_device_detects_divergence() {
        let sc = split_table();
        let cand: Vec<u32> = (0..2 * CAND).map(|i| 500 + i as u32).collect();
        let unary = vec![0f32; 2 * CAND];
        let dev = vec![cand[3], cand[CAND + 7]];
        assert!(walk_matches_device(&sc, &unary, &cand, &dev, 2).unwrap());
        let bad = vec![cand[3], cand[CAND + 11]];
        assert!(!walk_matches_device(&sc, &unary, &cand, &bad, 2).unwrap());
        assert!(!walk_matches_device(&sc, &unary, &cand, &dev[..1], 2).unwrap());
    }

    /// **采样档 `unary` 臂**：temp=0 时草稿 = 逐位置边际 argmax（边缘分不参与）。
    /// **q 校准（α / ε）**：同一张过尖的分数表，α 越小 q(top1) 越低、地板再摊平；
    /// 支撑不变（候选集外 q=0），且 q 归一。
    #[test]
    fn q_alpha_and_floor_shape_the_proposal() {
        let depth = 1usize;
        let mut scores = vec![0f32; depth * CAND * CAND];
        // 第 0 行：强尖峰 c=3，次高 c=7（gap 4 nats = 选择器边分的典型过自信）
        scores[3] = 8.0;
        scores[7] = 4.0;
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 900 + i as u32).collect();
        let cfg = crate::sampler::SamplerCfg {
            temp: 1.0,
            topk: 20,
            topp: 1.0,
            rep_penalty: 1.0,
        };
        let mk = |alpha: f32, floor: f32| {
            DraftQ::from_scores(QInput {
                scores: &scores,
                cand: &cand,
                depth,
                unary: None,
                walk: owl_shared::config::DflashWalk::Conditional,
                cfg: &cfg,
                seed: 5,
                base_pos: 0,
                alpha,
                floor,
            })
            .unwrap()
        };
        let sharp = mk(1.0, 0.0);
        let flat = mk(0.25, 0.0);
        let floored = mk(0.25, 0.5);
        let t1 = cand[3];
        assert!(
            sharp.q_of(0, t1) > 0.95,
            "α=1 应近乎 one-hot：{}",
            sharp.q_of(0, t1)
        );
        assert!(
            flat.q_of(0, t1) < sharp.q_of(0, t1),
            "α 缩小必须摊开 q：{} vs {}",
            flat.q_of(0, t1),
            sharp.q_of(0, t1)
        );
        assert!(
            floored.q_of(0, t1) < flat.q_of(0, t1),
            "地板必须再摊平：{} vs {}",
            floored.q_of(0, t1),
            flat.q_of(0, t1)
        );
        // 支撑不变：候选集外的 token q = 0（不给 p=0 的地方白付拒绝）
        assert_eq!(floored.q_of(0, 12_345), 0.0);
        let sum: f32 = (0..CAND).map(|c| floored.q_of(0, cand[c])).sum();
        assert!((sum - 1.0).abs() < 1e-5, "q 和 {sum}");
    }

    #[test]
    fn unary_walk_sampling_ignores_edges() {
        use owl_shared::config::DflashWalk as W;
        let depth = 2usize;
        let sc = split_table();
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 700 + i as u32).collect();
        let mut unary = vec![0f32; depth * CAND];
        unary[5] = 3.0;
        unary[CAND + 9] = 4.0;
        let cfg = crate::sampler::SamplerCfg { temp: 0.0, topk: 20, topp: 0.95, rep_penalty: 1.0 };
        let q = DraftQ::from_scores(QInput {
            scores: &sc,
            cand: &cand,
            depth,
            unary: Some(&unary),
            walk: W::Unary,
            cfg: &cfg,
            seed: 11,
            base_pos: 0,
            alpha: 1.0,
            floor: 0.0,
        })
        .unwrap();
        assert_eq!(q.drafts(), &[cand[5], cand[CAND + 9]]);
    }

    #[test]
    fn q_normalizes_and_walk_is_consistent() {
        let depth = 7;
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 100 + i as u32).collect();
        let q = DraftQ::from_scores(qi(&scores(depth), &cand, depth, &cfg_plain(), 42, 1000)).unwrap();
        for e in 0..depth {
            let s: f32 = (0..CAND).map(|c| q.q_of(e, cand[e * CAND + c])).sum();
            assert!((s - 1.0).abs() < 1e-5, "e={e} sum={s}");
            assert!(q.drafts().contains(&q.drafts()[e]));
        }
    }

    #[test]
    fn residual_falls_back_to_p_when_exhausted() {
        // **残差兜底**（参考实现同律）：q 与 p 在同一个 token 上重合 ⇒
        // `Σ max(p−q,0) = 0` 的浮点边界。旧实现直接报错（整个请求 abort）；
        // 现在回退到按 p 抽 —— 任何数值情形下都要有合法输出。
        let depth = 1;
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 500 + i as u32).collect();
        let mut q = DraftQ::from_scores(qi(&scores(depth), &cand, depth, &cfg_plain(), 3, 0)).unwrap();
        let t0 = cand[0];
        q.force_one_hot(0, t0); // q = δ(t0)
        let p = vec![(t0, 1.0f32)]; // p 也是 δ(t0) ⇒ 残差全 0
        assert_eq!(q.q_of(0, t0), 1.0);
        let got = residual_sample(&p, &q, 0, 7, 0).expect("残差耗尽必须兜底，不得报错");
        assert_eq!(got, t0, "兜底必须从 p 的支撑里抽");
    }

    #[test]
    fn rejection_preserves_target_distribution() {
        // p = softmax([2,1,0,-1,...]);q = softmax([-1,-1,2,0,...])。
        // accepted+correction 的 token 频率必须回到 p（与 q 无关）。
        let depth = 1;
        let vocab = 16usize;
        let mut logits = vec![0f32; 2 * vocab];
        logits[0] = 2.0;
        logits[1] = 1.0;
        logits[2] = 0.0;
        logits[3] = -1.0;
        let mut qlog = vec![-1000f32; CAND * CAND];
        qlog[0] = -1.0;
        qlog[1] = -1.0;
        qlog[2] = 2.0;
        qlog[3] = 0.0;
        qlog[4] = -1000.0;
        let cand: Vec<u32> = (0..CAND as u32).collect();
        let cfg = crate::sampler::SamplerCfg { temp: 1.0, topk: 16, topp: 1.0, rep_penalty: 1.0 };
        let mut cnt = [0usize; 4];
        let n = 20_000usize;
        for s in 0..n {
            let mut hist = Vec::new();
            // proposal_q 每轮按 seed 重新抽签 ⇒ draft ~ q（不是固定单点）。
            let q = DraftQ::from_scores(qi(&qlog, &cand, depth, &cfg, s as u64, 0)).unwrap();
            let (_, emitted, _) = accept_rejection(
                &logits, vocab, 2, q.drafts(), depth, &q, &cfg, &mut hist, s as u64, 0,
            )
            .unwrap();
            let t = emitted[0];
            if (t as usize) < 4 {
                cnt[t as usize] += 1;
            }
        }
        let z: f32 = (0..vocab).map(|i| (logits[i] - 2.0).exp()).sum();
        for i in 0..4 {
            let want = (logits[i] - 2.0).exp() / z;
            let got = cnt[i] as f32 / n as f32;
            assert!((got - want).abs() < 0.025, "token {i}: got {got:.4} want {want:.4}");
        }
    }

    /// **C2**:q 必须走**请求的**采样变换(top-k 截断 + 支撑重归一)。
    #[test]
    fn q_honours_request_topk() {
        let depth = 3usize;
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 100 + i as u32).collect();
        let cfg = crate::sampler::SamplerCfg { temp: 0.7, topk: 4, topp: 1.0, rep_penalty: 1.0 };
        let q = DraftQ::from_scores(qi(&scores(depth), &cand, depth, &cfg, 5, 0)).unwrap();
        for e in 0..depth {
            let support = (0..CAND).filter(|c| q.q_of(e, cand[e * CAND + c]) > 0.0).count();
            assert!(support <= 4, "e={e} 支撑 {support} > 请求 top_k=4(q 未走请求变换)");
            let sum: f32 = (0..CAND).map(|c| q.q_of(e, cand[e * CAND + c])).sum();
            assert!((sum - 1.0).abs() < 1e-5, "e={e} Σq={sum}");
        }
        assert_eq!(q.truncated_steps(), depth, "每一步都该被 top_k 截断");
        assert!(q.dropped_mass() > 0.0, "被截掉的质量必须记账(C3)");
    }

    /// **C2**:请求关掉截断时 q 不截断、不计账(与 p 同口径)。
    #[test]
    fn q_untruncated_when_request_keeps_everything() {
        let depth = 2usize;
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 100 + i as u32).collect();
        let q = DraftQ::from_scores(qi(&scores(depth), &cand, depth, &cfg_plain(), 5, 0)).unwrap();
        assert_eq!(q.truncated_steps(), 0, "无截断请求不许记截断");
        assert_eq!(q.dropped_mass(), 0.0);
    }

    /// **C2**:temp=0(贪心请求)== argmax one-hot,与裸 decode 同语义。
    #[test]
    fn q_is_one_hot_at_zero_temperature() {
        let depth = 2usize;
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 100 + i as u32).collect();
        let sc = scores(depth);
        let cfg = crate::sampler::SamplerCfg { temp: 0.0, topk: 20, topp: 0.95, rep_penalty: 1.0 };
        let q = DraftQ::from_scores(qi(&sc, &cand, depth, &cfg, 9, 0)).unwrap();
        // 第 0 步 argmax(row0)
        let row0 = &sc[..CAND];
        let best = row0
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal).then(b.0.cmp(&a.0)))
            .map(|(c, _)| c)
            .unwrap();
        assert_eq!(q.q_of(0, cand[best]), 1.0, "temp=0 ⇒ one-hot(argmax)");
        assert_eq!(q.drafts()[0], cand[best]);
        assert_eq!(q.truncated_steps(), 0, "贪心档没有质量可截");
    }

    /// **C2-c**:q 的支撑与概率必须与 p 侧 `distribution_of_cands` **逐项相同**
    /// (同一个 `SamplerCfg`;这就是"消灭两套采样规范"的机器门)。
    #[test]
    fn q_matches_p_transform() {
        use crate::sampler::distribution_of_cands;
        let depth = 1usize; // 单步:prev 恒 0 ⇒ 行 = scores[..CAND]
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 100 + i as u32).collect();
        let sc = scores(depth);
        let cfg = crate::sampler::SamplerCfg { temp: 0.8, topk: 6, topp: 0.9, rep_penalty: 1.0 };
        let q = DraftQ::from_scores(qi(&sc, &cand, depth, &cfg, 3, 0)).unwrap();
        let row_cands: Vec<(f32, u32)> =
            sc[..CAND].iter().enumerate().map(|(c, &v)| (v, cand[c])).collect();
        let p = distribution_of_cands(&row_cands, &cfg);
        let mut want: Vec<(u32, f32)> = p.clone();
        let mut got: Vec<(u32, f32)> = cand[..CAND]
            .iter()
            .copied()
            .map(|t| (t, q.q_of(0, t)))
            .filter(|(_, pr)| *pr > 0.0)
            .collect();
        want.sort_unstable_by_key(|(t, _)| *t);
        got.sort_unstable_by_key(|(t, _)| *t);
        assert_eq!(want.len(), got.len(), "支撑大小不同:q={got:?} p={want:?}");
        for (w, g) in want.iter().zip(got.iter()) {
            assert_eq!(w.0, g.0, "支撑集不同");
            assert!((w.1 - g.1).abs() < 1e-6, "token {} 概率不同:{} vs {}", w.0, w.1, g.1);
        }
    }

    #[test]
    fn one_hot_reconcile() {
        let depth = 3;
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 100 + i as u32).collect();
        let mut q = DraftQ::from_scores(qi(&scores(depth), &cand, depth, &cfg_plain(), 7, 0)).unwrap();
        let mut cur = q.drafts().to_vec();
        cur[1] = 999;
        assert_eq!(q.reconcile(&cur), 1);
        assert_eq!(q.q_of(1, 999), 1.0);
        assert_eq!(q.q_of(1, 0), 0.0);
    }
}

/// **拒绝采样接受(稀疏目标分布版;2026-10-31)**。
///
/// 与 [`accept_rejection`] **同语义**,区别只在 p 的来源:
///
/// - [`accept_rejection`] 吃 `depth+1 × vocab` 的整行 logits;
/// - 本函数吃**每行已算好的稀疏目标分布**(`distribution` 的产物)。
///
/// 为什么要这一版:**单卡采样档**的 verify 只能从设备端拿 top-32 候选
/// (`cand` 槽),而全文 logits 的回读 + host 全表 top-k 实测要 **+10ms/轮**
/// (89 → 67 t/s)。但 `proposal_q` 路径此前被排除在候选臂之外
/// (`use_cand` 里的 `proposal_q.is_none()`),因为本函数原来只会从整行 logits
/// 现算 p。稀疏 p 的支撑 ⊆ top-20 ⊆ 设备 top-32 ⇒ 候选行足以给出**逐项相同**的
/// p(温度是单调变换,不改序),于是两条路可合流。
///
/// 不变量(与 [`accept_rejection`] 逐字相同):
/// - `p_rows.len() == depth+1`(行 0..depth = verify 行;行 depth = bonus);
/// - `hist` 由调用方维护(接受一个就 push 一个);
/// - RNG 派生顺序不变(accept / correction / bonus 三支各自独立)。
#[allow(clippy::too_many_arguments)]
pub(crate) fn accept_rejection_sparse(
    p_rows: &[Vec<(u32, f32)>],
    drafts: &[u32],
    depth: usize,
    q: &DraftQ,
    hist: &mut Vec<u32>,
    seed_base: u64,
    pos: usize,
) -> Result<(usize, Vec<u32>, u32), ModelError> {
    if p_rows.len() != depth + 1 || drafts.len() < depth {
        return Err(ModelError::Msg(format!(
            "spec rejection(稀疏):行数={} depth={depth} drafts={} 不闭合",
            p_rows.len(),
            drafts.len()
        )));
    }
    let mut emitted: Vec<u32> = Vec::with_capacity(depth + 1);
    for (i, p) in p_rows.iter().enumerate().take(depth) {
        let d = drafts[i];
        let pd = p.iter().find(|(t, _)| *t == d).map(|(_, p)| *p).unwrap_or(0.0);
        let qd = q.q_of(i, d);
        if !(qd.is_finite() && qd > 0.0) {
            return Err(ModelError::Msg(format!(
                "spec rejection(稀疏):第 {i} 步 draft={d} 的 q={qd} 非正/非有限（proposal_q 与 drafts 不同源）"
            )));
        }
        let accept = (pd / qd).clamp(0.0, 1.0);
        if accept_uniform(seed_base, pos + i) < accept {
            emitted.push(d);
            hist.push(d);
            continue;
        }
        let corr = residual_sample(p, q, i, seed_base, pos)?;
        emitted.push(corr);
        hist.push(corr);
        return Ok((i, emitted, corr));
    }
    // 全接受:bonus 从第 depth 行目标分布抽(RNG 派生与整行版逐字同)
    let mut rng = seed_base ^ ((pos + depth) as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let bonus = crate::sampler::sample_sparse(&p_rows[depth], &mut rng);
    emitted.push(bonus);
    Ok((depth, emitted, bonus))
}

#[cfg(test)]
mod sparse_equivalence {
    //! **稀疏 p 版 ≡ 整行 logits 版**(2026-10-31 候选臂合流的正确性门)。
    //!
    //! 判据必须落在"数值逐项相同"上 —— 否则单卡采样走候选臂时,投机接受
    //! 会静默偏离参考实现(采样偏置没人看得见)。

    use super::*;
    use crate::sampler::{distribution, distribution_of_cands, SamplerCfg};

    const VOCAB: usize = 256;

    fn cfg() -> SamplerCfg {
        SamplerCfg { temp: 0.7, topk: 20, topp: 0.95, rep_penalty: 1.0 }
    }

    /// 采样走格的输入包（本模块只用生产缺省 `conditional`）
    fn qi<'a>(
        scores: &'a [f32],
        cand: &'a [u32],
        depth: usize,
        cfg: &'a SamplerCfg,
        seed: u64,
        pos: usize,
    ) -> QInput<'a> {
        QInput {
            scores,
            cand,
            depth,
            unary: None,
            walk: owl_shared::config::DflashWalk::Conditional,
            cfg,
            seed,
            base_pos: pos,
            alpha: 1.0,
            floor: 0.0,
        }
    }

    /// 造一行"有尖峰"的 logits(确定性伪随机,不引 random crate)
    fn row(seed: u32) -> Vec<f32> {
        let mut x = seed.wrapping_mul(2654435761);
        (0..VOCAB)
            .map(|i| {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                let base = ((x >> 16) as f32 / 65536.0 - 0.5) * 2.0;
                // 每 16 个位置放一个尖峰,保证 top-k 集非平凡
                base + if i % 16 == 0 { 6.0 } else { 0.0 }
            })
            .collect()
    }

    /// 设备端 top-32 的等效物(host 侧模拟:按值降序取 32,索引升序平局)
    fn top32(logits: &[f32]) -> Vec<(f32, u32)> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx.truncate(32);
        let mut v: Vec<(f32, u32)> =
            idx.into_iter().map(|i| (logits[i], i as u32)).collect();
        v.sort_unstable_by_key(|&(_, id)| id);
        v
    }

    /// 稀疏分布 = 整行分布(逐项、含概率)
    #[test]
    fn sparse_distribution_matches_full_row() {
        for seed in 1..8u32 {
            let lg = row(seed);
            let full = distribution(&mut lg.clone(), &cfg(), &[]);
            let sparse = distribution_of_cands(&top32(&lg), &cfg());
            assert_eq!(full.len(), sparse.len(), "seed={seed} 支撑大小不同");
            for ((tf, pf), (ts, ps)) in full.iter().zip(sparse.iter()) {
                assert_eq!(tf, ts, "seed={seed} 支撑集不同");
                assert!((pf - ps).abs() < 1e-7, "seed={seed} p 不同:{pf} vs {ps}");
            }
        }
    }

    /// 接受/拒绝/bonus 三支逐字相同(含 RNG 派生顺序)
    #[test]
    fn sparse_accept_matches_full_logits_accept() {
        let depth = 2usize;
        // q:每步 16 候选,全零边分 ⇒ 均匀分布(与 prev 无关)
        let scores = vec![0.0f32; depth * CAND * CAND];
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 10 + i as u32).collect();
        for seed in 1..6u32 {
            for pos in [0usize, 7, 33] {
                let q = DraftQ::from_scores(qi(&scores, &cand, depth, &cfg(), seed as u64, pos))
                    .expect("from_scores");
                // depth+1 行:每行不同尖峰
                let rows: Vec<Vec<f32>> = (0..=depth).map(|r| row(seed + r as u32)).collect();
                let cfg = cfg();
                // 整行版
                let flat: Vec<f32> = rows.iter().flatten().copied().collect();
                let mut h_full = Vec::new();
                let a = accept_rejection(
                    &flat, VOCAB, depth + 1, q.drafts(), depth, &q, &cfg, &mut h_full, seed as u64, pos,
                );
                // 稀疏版
                let p_rows: Vec<Vec<(u32, f32)>> =
                    rows.iter().map(|r| distribution_of_cands(&top32(r), &cfg)).collect();
                let mut h_sp = Vec::new();
                let b = accept_rejection_sparse(
                    &p_rows, q.drafts(), depth, &q, &mut h_sp, seed as u64, pos,
                );
                let a = a.expect("整行版");
                let b = b.expect("稀疏版");
                assert_eq!(a, b, "seed={seed} pos={pos} 接受结果不同");
                assert_eq!(h_full, h_sp, "seed={seed} pos={pos} 历史推进不同");
            }
        }
    }

    /// 抽样器:候选式与整行式同 RNG ⇒ 同 token
    #[test]
    fn sparse_sample_matches_candidate_sample() {
        for seed in 1..6u32 {
            let lg = row(seed);
            let t32 = top32(&lg);
            let cfg = cfg();
            for rng_seed in 0..4u64 {
                let mut r1 = rng_seed ^ 0x9E37_79B9_7F4A_7C15;
                let a = crate::sampler::sample_from_candidates(&t32, &cfg, &mut r1);
                let p = distribution_of_cands(&t32, &cfg);
                let mut r2 = rng_seed ^ 0x9E37_79B9_7F4A_7C15;
                let b = crate::sampler::sample_sparse(&p, &mut r2);
                assert_eq!(a, b, "seed={seed} rng={rng_seed}");
            }
        }
    }
}
