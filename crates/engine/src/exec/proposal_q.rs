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
}

impl DraftQ {
    /// 从选择器 `scores [K,16,16]` 与 `cand [K,16]` 构造并抽签。
    ///
    /// 布局与 `owl_dflash_select_*` 一致：`scores[e, prev, c]` 是第 e 步、
    /// 前驱候选 rank=prev 时选 c 的边分。walk 用**实际抽中的前驱**，
    /// 与 NInfer 的 conditional walk 同义。
    pub(crate) fn from_scores(
        scores: &[f32],
        cand: &[u32],
        depth: usize,
        temperature: f32,
        seed: u64,
        base_pos: usize,
    ) -> Result<Self, ModelError> {
        if depth == 0 || depth > CAND - 1 {
            return Err(ModelError::Msg(format!(
                "proposal_q: depth={depth} 越界（1..=15）"
            )));
        }
        if !(temperature.is_finite() && temperature > 0.0) {
            return Err(ModelError::Msg(format!(
                "proposal_q: temperature={temperature} 非正/非有限"
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
        let mut drafts = vec![0u32; depth];
        let mut prev = 0usize;
        for e in 0..depth {
            let row = &scores[(e * CAND + prev) * CAND..(e * CAND + prev + 1) * CAND];
            let mut mx = f32::NEG_INFINITY;
            for &s in row {
                if s.is_finite() && s > mx {
                    mx = s;
                }
            }
            if !mx.is_finite() {
                return Err(ModelError::Msg(format!(
                    "proposal_q: 第 {e} 步边分全非有限（selector 输出坏）"
                )));
            }
            let mut sum = 0f32;
            for (c, &s) in row.iter().enumerate() {
                let p = if s.is_finite() { ((s - mx) / temperature).exp() } else { 0.0 };
                qv[e * CAND + c] = p;
                sum += p;
            }
            if !(sum.is_finite() && sum > 0.0) {
                return Err(ModelError::Msg(format!(
                    "proposal_q: 第 {e} 步 softmax 和={sum} 非正/非有限"
                )));
            }
            for p in &mut qv[e * CAND..(e + 1) * CAND] {
                *p /= sum;
            }
            let u = uniform(seed, base_pos as u64 + e as u64, PURPOSE_PROPOSAL);
            let mut acc = 0f32;
            let mut pick = CAND - 1;
            for (c, &p) in qv[e * CAND..(e + 1) * CAND].iter().enumerate() {
                acc += p;
                if u < acc {
                    pick = c;
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
        })
    }

    pub(crate) fn drafts(&self) -> &[u32] {
        &self.drafts
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
        let accept = (pd / qd).min(1.0).max(0.0);
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
        return Err(ModelError::Msg(format!(
            "spec residual: 第 {e} 步 max(p-q,0) 总质量={sum}（q 不应支配 p）"
        )));
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

    fn scores(depth: usize) -> Vec<f32> {
        (0..depth * CAND * CAND).map(|i| ((i * 17) % 29) as f32 / 4.0).collect()
    }

    #[test]
    fn q_normalizes_and_walk_is_consistent() {
        let depth = 7;
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 100 + i as u32).collect();
        let q = DraftQ::from_scores(&scores(depth), &cand, depth, 1.0, 42, 1000).unwrap();
        for e in 0..depth {
            let s: f32 = (0..CAND).map(|c| q.q_of(e, cand[e * CAND + c])).sum();
            assert!((s - 1.0).abs() < 1e-5, "e={e} sum={s}");
            assert!(q.drafts().contains(&q.drafts()[e]));
        }
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
            let q = DraftQ::from_scores(&qlog, &cand, depth, 1.0, s as u64, 0).unwrap();
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

    #[test]
    fn one_hot_reconcile() {
        let depth = 3;
        let cand: Vec<u32> = (0..depth * CAND).map(|i| 100 + i as u32).collect();
        let mut q = DraftQ::from_scores(&scores(depth), &cand, depth, 1.0, 7, 0).unwrap();
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
        let accept = (pd / qd).min(1.0).max(0.0);
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
                let q = DraftQ::from_scores(&scores, &cand, depth, 1.0, seed as u64, pos)
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
