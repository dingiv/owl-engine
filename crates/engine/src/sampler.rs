//! 采样器(S3-b v1;host 侧)。
//!
//! **为什么存在**:greedy 解码在 instruct 模型上塌进复读吸引子
//! (2026-10-01 实测:27B「长城是长城长城…」96 tok 全程;0.8B 同病),
//! checkpoint 自带 generation_config 即 `do_sample=true, temp=1.0,
//! top_k=20, top_p=0.95` —— 采样是官方推荐的默认姿势,不是可选项。
//!
//! **v1 形态 = host 采样**:decode 图每步已可回读 logits
//! (`read_output_f32("logits")`,OWL_HOST_ARGMAX 先例),host 侧做
//! temperature → top-k → top-p → 多项式采样。代价 = 每步 ~1MB logits
//! dtoh + host softmax(debug 档不可见;release ~10% 步时)。device
//! 采样核(每步零 D2H 律)留 E4 靶面,本模块 API 面按可搬迁设计。
//!
//! **控制面(显式依赖,2026-10-10)**:参数经 [`SamplerCfg`] 随
//! EngineKnobs 传入(入口/测试构造;`OWL_SAMPLER=greedy` 关采样、
//! `OWL_TEMP`/`OWL_TOPK`/`OWL_TOPP` 映射在 `EngineKnobs::from_env`
//! 唯一登记)—— 引擎热路径零 env 读取(原"每步 from_env"已废除)。
//!
//! **RNG 契约**:xorshift64*,seed 由调用方从 (turn id, step) 派生 ——
//! 同 turn 重放确定性(测试可复现),跨 turn 不同。

/// **设备端重复惩罚窗口**(2026-10-26):固定 256 = 图输入槽 `rep_hist` 的长度。
///
/// 为什么要固定:图槽形状是捕获期契约,长度不能随请求/上下文变 —— 变长就
/// 要新建图桶(本刀明确禁止)。语义 = 最近 256 个**已提交** token(去重排序)。
pub const REP_HISTORY_WINDOW: usize = 256;

/// 空槽/尾部填充值。真实词表 id < 2²⁴(见 argmax 契约),不会碰撞。
pub const REP_HISTORY_SENTINEL: u32 = u32::MAX;

/// 构建设备端惩罚集:最近 [`REP_HISTORY_WINDOW`] 个已提交 token,
/// **去重后升序**,尾部用 [`REP_HISTORY_SENTINEL`] 填满。
///
/// 输入拆成 prompt/generated 两段只是为了避免调用点各拼一份 Vec;语义上就是
/// `prompt_ids ++ generated` 的尾部窗口。草稿 token 不进入 v1 的历史(它们是
/// speculative,尚未提交;见交接文档 §3.1)。
#[must_use]
pub fn penalty_history(prompt_ids: &[u32], generated: &[u32]) -> Vec<u32> {
    let total = prompt_ids.len().saturating_add(generated.len());
    let start = total.saturating_sub(REP_HISTORY_WINDOW);
    let mut hist: Vec<u32> = Vec::with_capacity(REP_HISTORY_WINDOW);
    if start < prompt_ids.len() {
        hist.extend_from_slice(&prompt_ids[start..]);
        hist.extend_from_slice(generated);
    } else {
        hist.extend_from_slice(&generated[start - prompt_ids.len()..]);
    }
    hist.sort_unstable();
    hist.dedup();
    debug_assert!(hist.len() <= REP_HISTORY_WINDOW);
    hist.resize(REP_HISTORY_WINDOW, REP_HISTORY_SENTINEL);
    hist
}

/// 采样配置(generation_config 同款三维 + 反循环惩罚)
#[derive(Clone, Copy, Debug)]
pub struct SamplerCfg {
    pub temp: f32,
    pub topk: usize,
    pub topp: f32,
    /// 重复惩罚(llama.cpp 同语义:**每个不同 token 只施加一次**;正 logit 除,
    /// 负 logit 乘;1.0 = 关)。
    ///
    /// ⚠️ 默认 **1.0(关)**(2026-10-14 由 1.15 改回):该旋钮是 owl 自加的
    /// "反循环服务档",而 checkpoint `generation_config`、OpenAI 协议、
    /// vLLM(pi 侧无 `samplingParams`)都不施加 ⇒ 属**对参考实现的单方偏离**。
    /// 实测(pi 真机):1.15 之下长思考/长输出必崩(复读→词沙拉),1.0 恢复正常。
    pub rep_penalty: f32,
}

impl Default for SamplerCfg {
    /// 缺省 = generation_config 同款三维,**不做重复惩罚**(rep_penalty = 1.0)
    fn default() -> Self {
        SamplerCfg { temp: 1.0, topk: 20, topp: 0.95, rep_penalty: 1.0 }
    }
}

impl SamplerCfg {
    /// **入口侧组装器**:自全量配置 sampling module 显式搬运
    /// (引擎内部经 EngineKnobs.sampler 携带,热路径零 env 读取)
    pub fn from_sampling(s: &owl_shared::config::SamplingCfg) -> Self {
        SamplerCfg { temp: s.temp, topk: s.topk, topp: s.topp, rep_penalty: s.rep_penalty }
    }
}



/// xorshift64* 步进(返回 64 位;调用方持有 state)
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

/// [0, 1) 均匀抽样(24 位尾数;f32 精度足够采样)
fn uniform(state: &mut u64) -> f32 {
    ((xorshift(state) >> 40) as f32) / (1u64 << 24) as f32
}

/// logits → 目标分布（**与 [`sample`] 同一套变换**：repeat-penalty →
/// temperature → top-k → softmax → top-p 重均摊）。返回降序 `(token, prob)`。
///
/// 仅保留有正质量的截断集（top-k/top-p 之后 ≤ topk 项）；`temp <= 0` 返回
/// 单点 one-hot。投机拒绝采样需要这个稀疏表示：目标分布 p 的支撑最多
/// topk（缺省 20）项，残差 `max(p-q,0)` 也只在这个支撑上有值。
pub fn distribution(logits: &mut [f32], cfg: &SamplerCfg, history: &[u32]) -> Vec<(u32, f32)> {
    let n = logits.len();
    debug_assert!(n > 0);
    // ── repetition penalty(在 temperature 之前;llama.cpp 同语义:正除负乘)──
    //
    // **去重律**:每个**不同** token 只施加一次（详见 `sample` 历史注记）。
    if cfg.rep_penalty > 1.0 && !history.is_empty() {
        let mut uniq: Vec<u32> = history.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        for &t in &uniq {
            let t = t as usize;
            if t < n {
                let v = logits[t];
                logits[t] = if v > 0.0 { v / cfg.rep_penalty } else { v * cfg.rep_penalty };
            }
        }
    }
    if cfg.temp <= 0.0 {
        // greedy 臂(temp ≤ 0):纯 argmax(平局取小索引,与设备核同语义)
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > bv {
                bv = v;
                best = i;
            }
        }
        return vec![(best as u32, 1.0)];
    }

    // ── top-k 选择(k=20 量级:有界 Vec + 线性 min 追踪;替换稀疏,
    //   均摊每元素 ~O(1);f32 非 Ord 故不走 BinaryHeap)──
    let k = cfg.topk.min(n).max(1);
    let mut cand: Vec<(f32, u32)> = Vec::with_capacity(k);
    let mut min_pos = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        let v = if v.is_finite() { v } else { f32::NEG_INFINITY };
        if cand.len() < k {
            cand.push((v, i as u32));
            if v < cand[min_pos].0 {
                min_pos = cand.len() - 1;
            }
        } else if v > cand[min_pos].0 {
            cand[min_pos] = (v, i as u32);
            min_pos = cand
                .iter()
                .enumerate()
                .min_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
                .unwrap()
                .0;
        }
    }

    // ── 规范化序(2026-10-26;分布式采样等价性的前提)──
    // 选出的 k 个候选按 **(值降, 索引升)** 重排后再做 softmax:
    // 拓扑选择只保证"集合"一致,而**槽位顺序**是扫描历史的产物(被替换的
    // 低值填充位会改变顺序)⇒ 求和顺序随实现漂移。规范化后,全表路径与
    // "两半 top-K 并集"路径的集合相同 ⇒ 采样**逐位相同**。
    cand.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    // ── softmax(temperature;减 max 稳定化)──
    let mx = cand.iter().map(|(v, _)| *v).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for (v, _) in cand.iter_mut() {
        *v = ((*v - mx) / cfg.temp).exp();
        sum += *v;
    }
    for (v, _) in cand.iter_mut() {
        *v /= sum;
    }

    // ── top-p 截断:降序累积,含跨过 topp 的那个 token(vLLM 同语义)──
    cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0f32;
    let mut cut = cand.len();
    for (i, (p, _)) in cand.iter().enumerate() {
        cum += p;
        if cum >= cfg.topp {
            cut = i + 1;
            break;
        }
    }
    let tail: f32 = cand[cut..].iter().map(|(p, _)| p).sum();
    let spread = tail / cut as f32;
    let keep = &mut cand[..cut];
    for (p, _) in keep.iter_mut() {
        *p += spread; // 截断质量重均摊到保留集(和恒 1)
    }
    keep.iter().map(|(p, t)| (*t, *p)).collect()
}

/// **候选集采样**(分布式采样,2026-10-26):TP 词表切下两半表各出 top-K,
/// host 合并后只在这 ≤2K 个候选上采样 —— 不再回读 4MB logits。
///
/// `union` = **(全局索引升序)** 的 `(logit, 全局 token id)`,由调用方排序。
/// 语义与 [`sample`] **逐位等价**:
/// - 全局 top-k ⊆ 两半 top-K 的并集 ⇒ 在并集上按同一"index 升序扫描 +
///   严格 `>` 替换"的算法选 k,其结果**集合与内部顺序**与全域扫描一致;
/// - 之后的 softmax/top-p/多项式抽样直接复用 [`distribution`]。
///
/// 前置门(调用方保证,不满足就走 logits 老路):`cfg.topk > 0`、
/// `cfg.topk <= K`、`union.len() >= cfg.topk`。
///
/// **重复惩罚已由设备端 top-32 核在选候选之前完成**(2026-10-26):这里
/// 传空 history 给 [`distribution`],不会二次惩罚;候选值就是最终比较值。
pub fn sample_from_candidates(union: &[(f32, u32)], cfg: &SamplerCfg, rng: &mut u64) -> u32 {
    debug_assert!(cfg.topk > 0 && union.len() >= cfg.topk);
    let mut vals: Vec<f32> = union.iter().map(|(v, _)| *v).collect();
    let keep = distribution(&mut vals, cfg, &[]);
    let u = uniform(rng);
    let mut acc = 0f32;
    for (pos, p) in keep.iter() {
        acc += *p;
        if u < acc {
            return union[*pos as usize].1;
        }
    }
    keep.last().map(|(pos, _)| union[*pos as usize].1).unwrap_or(0)
}

/// logits → 采样 token（[`distribution`] + 多项式抽样）。
/// repeat-penalty 会原地修改 `logits`（与旧行为一致）。
pub fn sample(logits: &mut [f32], cfg: &SamplerCfg, rng: &mut u64, history: &[u32]) -> u32 {
    let keep = distribution(logits, cfg, history);
    let u = uniform(rng);
    let mut acc = 0f32;
    for (idx, p) in keep.iter() {
        acc += *p;
        if u < acc {
            return *idx;
        }
    }
    keep.last().map(|(idx, _)| *idx).unwrap_or(0)
}

#[cfg(test)]
mod candidate_tests {
    use super::*;

    /// 候选式采样必须与全表采样**逐 token 等价**(同 rng、同 cfg)。
    /// 覆盖:普通分布 / 大量平局 / topk=1 / topp=1.0。
    fn check(logits: &[f32], cfg: &SamplerCfg, k: usize, seeds: u64) {
        let n = logits.len();
        // 全表 top-32(值降,索引升)= 两半并集的上界
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap()
                .then(a.cmp(&b))
        });
        let mut union: Vec<(f32, u32)> = order[..k.min(n)]
            .iter()
            .map(|&i| (logits[i], i as u32))
            .collect();
        union.sort_unstable_by_key(|&(_, i)| i);
        for sd in 0..seeds {
            let state = sd.wrapping_mul(0x9E3779B97F4A7C15) | 1;
            let (mut r1, mut r2) = (state, state);
            let mut full = logits.to_vec();
            let a = sample(&mut full, cfg, &mut r1, &[]);
            let b = sample_from_candidates(&union, cfg, &mut r2);
            assert_eq!(a, b, "seed={sd} cfg={cfg:?}");
            assert_eq!(r1, r2, "RNG 消耗必须一致");
        }
    }

    #[test]
    fn candidates_equal_full_table_random() {
        let n = 4096usize;
        let mut x = vec![0f32; n];
        let mut s = 123456789u64;
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0;
        }
        let cfg = SamplerCfg { temp: 0.9, topk: 20, topp: 0.92, rep_penalty: 1.0 };
        check(&x, &cfg, 32, 500);
    }

    #[test]
    fn candidates_equal_full_table_heavy_ties() {
        // 大量同值:候选边界全是平局 —— 最考"取最小索引"语义
        let n = 512usize;
        let x: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.5).collect();
        let cfg = SamplerCfg { temp: 1.0, topk: 20, topp: 0.95, rep_penalty: 1.0 };
        check(&x, &cfg, 32, 500);
    }

    #[test]
    fn candidates_equal_full_table_k1_and_topp1() {
        let n = 1024usize;
        let x: Vec<f32> = (0..n).map(|i| ((i * 37 % 101) as f32) * 0.1).collect();
        check(&x, &SamplerCfg { temp: 1.3, topk: 1, topp: 0.95, rep_penalty: 1.0 }, 32, 200);
        check(&x, &SamplerCfg { temp: 0.7, topk: 20, topp: 1.0, rep_penalty: 1.0 }, 32, 200);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与 production `distribution` 同式的 host 惩罚(仅测试参考用)。
    fn apply_rep(logits: &mut [f32], rep: f32, history: &[u32]) {
        if rep <= 1.0 || history.is_empty() {
            return;
        }
        let mut uniq = history.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        for &t in &uniq {
            let t = t as usize;
            if t < logits.len() {
                let v = logits[t];
                logits[t] = if v > 0.0 { v / rep } else { v * rep };
            }
        }
    }

    #[test]
    fn penalty_history_is_recent_dedup_sorted_with_sentinel() {
        assert_eq!(penalty_history(&[], &[]), vec![REP_HISTORY_SENTINEL; REP_HISTORY_WINDOW]);

        let h = penalty_history(&[9, 3, 9, 7], &[3, 5]);
        // 最近窗口 = prompt ++ generated,去重升序,尾部哨兵
        assert_eq!(&h[..4], &[3, 5, 7, 9]);
        assert!(h[4..].iter().all(|&x| x == REP_HISTORY_SENTINEL));

        // 超过窗口:只保留最近 256 个,再做去重排序
        let prompt: Vec<u32> = (0..300).collect();
        let gen: Vec<u32> = (300..400).collect();
        let h = penalty_history(&prompt, &gen);
        assert_eq!(h.len(), REP_HISTORY_WINDOW);
        assert_eq!(h[0], 144, "窗口应从 400-256=144 开始");
        assert_eq!(h[REP_HISTORY_WINDOW - 1], 399);
        assert!(h.windows(2).all(|w| w[0] < w[1]), "必须严格升序");

        // 窗口内重复 token 只留一份;空位仍是哨兵
        let h = penalty_history(&[5, 5, 5], &[5, 6, 6]);
        assert_eq!(h[0], 5);
        assert_eq!(h[1], 6);
        assert!(h[2..].iter().all(|&x| x == REP_HISTORY_SENTINEL));
    }

    /// 候选路径的重复惩罚语义:设备端先惩罚再 top-32,host 只在候选上
    /// 做 temperature/top-k/top-p ⇒ 必须与全表 `sample` 逐 token 等价。
    #[test]
    fn candidates_equal_full_table_with_device_prepenalty() {
        let n = 4096usize;
        let mut x = vec![0f32; n];
        let mut s = 0x1234_5678u64;
        for v in x.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0;
        }
        // 让历史命中若干真实高概率 token,并带重复项
        let history: Vec<u32> = (20..70).chain(20..40).collect();
        let cfg = SamplerCfg { temp: 0.8, topk: 20, topp: 0.92, rep_penalty: 1.3 };

        let mut penalized = x.clone();
        apply_rep(&mut penalized, cfg.rep_penalty, &history);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            penalized[b]
                .partial_cmp(&penalized[a])
                .unwrap()
                .then(a.cmp(&b))
        });
        let mut union: Vec<(f32, u32)> = order[..32]
            .iter()
            .map(|&i| (penalized[i], i as u32))
            .collect();
        union.sort_unstable_by_key(|&(_, i)| i);

        for seed in 0..300u64 {
            let state = seed.wrapping_mul(0x9E3779B97F4A7C15) | 1;
            let (mut r1, mut r2) = (state, state);
            let mut full = x.clone();
            let a = sample(&mut full, &cfg, &mut r1, &history);
            let b = sample_from_candidates(&union, &cfg, &mut r2);
            assert_eq!(a, b, "候选路径与全表路径不等价 seed={seed}");
            assert_eq!(r1, r2, "RNG 消耗必须一致 seed={seed}");
        }
    }

    #[test]
    fn greedy_arm_and_temp_zero() {
        let mut logits = vec![-1.0, 3.0, 2.0, 0.5];
        let g = SamplerCfg { temp: 0.0, topk: 20, topp: 0.95, rep_penalty: 1.0 };
        assert_eq!(sample(&mut logits, &g, &mut 1, &[]), 1, "temp=0 → argmax");
    }

    #[test]
    fn dominant_token_wins_but_not_always() {
        // 双峰分布(top-p 保留两个 token):高频非独占 —— 与 greedy 本质区别
        let mut logits = vec![-10f32; 1000];
        logits[7] = 1.0; // P≈0.55
        logits[100] = 0.8; // P≈0.45(其余 exp(-10)≈0)
        let cfg = SamplerCfg { temp: 1.0, topk: 20, topp: 0.95, rep_penalty: 1.0 };
        let mut rng = 42u64;
        let (h7, h100) = (0..400).fold((0, 0), |(a, b), _| {
            match sample(&mut logits, &cfg, &mut rng, &[]) {
                7 => (a + 1, b),
                100 => (a, b + 1),
                _ => (a, b),
            }
        });
        assert!(h7 > 120 && h7 < 280, "主导 token 高频非必然:h7={h7}/400");
        assert!(h100 > 80, "次峰应实际参与采样:h100={h100}/400");
    }

    #[test]
    fn deterministic_same_seed() {
        let mut logits: Vec<f32> = (0..5000).map(|i| ((i * 37) % 97) as f32 / 10.0).collect();
        let cfg = SamplerCfg { temp: 1.0, topk: 20, topp: 0.95, rep_penalty: 1.0 };
        let (mut a, mut b) = (12345u64, 12345u64);
        let seq_a: Vec<u32> = (0..50).map(|_| sample(&mut logits, &cfg, &mut a, &[])).collect();
        let seq_b: Vec<u32> = (0..50).map(|_| sample(&mut logits, &cfg, &mut b, &[])).collect();
        assert_eq!(seq_a, seq_b, "同 seed 必同序列(测试可复现契约)");
    }

    #[test]
    fn nan_poison_ignored() {
        let mut logits = vec![-1f32; 100];
        logits[3] = f32::NAN;
        logits[50] = 2.0; // 真 max
        let cfg = SamplerCfg { temp: 1.0, topk: 20, topp: 0.95, rep_penalty: 1.0 };
        let mut rng = 7u64;
        for _ in 0..100 {
            let t = sample(&mut logits, &cfg, &mut rng, &[]);
            assert_ne!(t, 3, "NaN 位不得进分布");
        }
    }

    #[test]
    fn rep_penalty_breaks_attractor() {
        // 复读吸引子(真实 gap 形态:top 与次峰差 1.5 logit):惩罚后让位
        let mut logits = vec![-3f32; 500];
        logits[5] = 8.0;
        logits[10] = 6.5; // P(5) ≈ 0.82,top-p 后必高概率中
        let cfg = SamplerCfg { temp: 1.0, topk: 20, topp: 0.95, rep_penalty: 1.3 };
        let mut rng = 11u64;
        let no_pen = (0..200)
            .map(|_| sample(&mut logits.clone(), &cfg, &mut rng, &[]))
            .filter(|&t| t == 5)
            .count();
        let mut rng = 11u64;
        let with_pen = (0..200)
            .map(|_| sample(&mut logits.clone(), &cfg, &mut rng, &[5]))
            .filter(|&t| t == 5)
            .count();
        assert!(no_pen > 140, "无惩罚应高频: {no_pen}/200");
        assert!(with_pen < 120, "惩罚后应实质让位: {with_pen}/200");
    }

    /// **去重律**:同一 token 出现 k 次只罚一次(回归门;原实现 = 1.15^k 累积)
    #[test]
    fn rep_penalty_once_per_distinct_token() {
        let cfg = SamplerCfg { temp: 0.0, topk: 20, topp: 0.95, rep_penalty: 1.5 };
        let base = || {
            let mut v = vec![-3f32; 64];
            v[5] = 8.0;
            v[6] = 7.0;
            v
        };
        let once = {
            let mut l = base();
            sample(&mut l, &cfg, &mut 7, &[5]);
            l
        };
        let five = {
            let mut l = base();
            sample(&mut l, &cfg, &mut 7, &[5, 5, 5, 5, 5]);
            l
        };
        assert_eq!(once[5], five[5], "重复出现不得叠加惩罚(1.5^k 雷已焊死)");
        assert!(once[5] < base()[5], "至少罚一次");
        // 两个不同 token 各罚一次:8/1.5=5.333 vs 7/1.5=4.667
        let two = {
            let mut l = base();
            sample(&mut l, &cfg, &mut 7, &[5, 6]);
            l
        };
        assert!((two[5] - 5.3333).abs() < 1e-3 && (two[6] - 4.6667).abs() < 1e-3, "{:?} {:?}", two[5], two[6]);
        // 1.0 = 关
        let off = SamplerCfg { rep_penalty: 1.0, ..cfg };
        let mut l = base();
        sample(&mut l, &off, &mut 7, &[5, 5, 5]);
        assert_eq!(l[5], 8.0, "rep_penalty=1.0 不得改动 logits");
    }

    /// top-p 截断后采样集中于头部(尖峰分布 + 大词表)。
    #[test]
    fn topp_mass_concentrates_head() {
        // 尖峰分布 + 大词表:top-p 截断后采样几乎全落头部
        let mut logits = vec![-20f32; 10000];
        for (i, v) in logits.iter_mut().enumerate().take(5) {
            *v = 3.0 - i as f32 * 0.5;
        }
        let cfg = SamplerCfg { temp: 1.0, topk: 20, topp: 0.95, rep_penalty: 1.0 };
        let mut rng = 99u64;
        let head_hits = (0..300)
            .filter(|_| {
                let t = sample(&mut logits, &cfg, &mut rng, &[]);
                t < 5
            })
            .count();
        assert!(head_hits >= 295, "top-p 后应集中头部:hits={head_hits}/300");
    }
}
