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

/// 采样配置(generation_config 同款三维 + 反循环惩罚)
#[derive(Clone, Copy, Debug)]
pub struct SamplerCfg {
    pub temp: f32,
    pub topk: usize,
    pub topp: f32,
    /// 重复惩罚(llama.cpp 同语义:>1 压制已现 token;正 logit 除,
    /// 负 logit 乘;1.0 = 关。默认 1.15 = 反循环服务档 ——
    /// top-p 逃不出 P>0.95 的复读吸引子,2026-10-01 实测定谳)
    pub rep_penalty: f32,
}

impl Default for SamplerCfg {
    /// 缺省 = 反循环服务档(generation_config 同款)
    fn default() -> Self {
        SamplerCfg { temp: 1.0, topk: 20, topp: 0.95, rep_penalty: 1.15 }
    }
}

impl SamplerCfg {
    /// **入口侧构造器**(server config / 测试用例专用;引擎内部经
    /// EngineKnobs.sampler 显式携带,热路径零 env 读取)
    pub fn from_env() -> Self {
        let f = |k: &str, d: f32| -> f32 {
            owl_shared::env_reader::parse_or(k, d)
        };
        SamplerCfg {
            temp: f("OWL_TEMP", 1.0),
            topk: f("OWL_TOPK", 20.0) as usize,
            topp: f("OWL_TOPP", 0.95),
            rep_penalty: f("OWL_REP_PENALTY", 1.15),
        }
    }
}

/// 采样开关映射(默认开 = 成功推理的默认姿势;OWL_SAMPLER=greedy 关)。
/// **入口侧构造器**:引擎内部经 EngineKnobs.sampler_enabled 显式携带。
pub fn enabled() -> bool {
    owl_shared::env_reader::str("OWL_SAMPLER").map(|v| v != "greedy").unwrap_or(true)
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

/// logits → 采样 token(repeat-penalty → temperature → top-k → softmax →
/// top-p 截断 → 多项式)。`history` = 已生成 token(惩罚集;空 = 无惩罚)。
/// NaN/Inf 视作 -inf(坏位不进分布);k/p 越界按全词表。
pub fn sample(logits: &mut [f32], cfg: &SamplerCfg, rng: &mut u64, history: &[u32]) -> u32 {
    let n = logits.len();
    debug_assert!(n > 0);
    // ── repetition penalty(在 temperature 之前;llama.cpp 同语义:正除负乘)──
    if cfg.rep_penalty > 1.0 && !history.is_empty() {
        for &t in history {
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
        return best as u32;
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

    // ── 多项式抽样 ──
    let u = uniform(rng);
    let mut acc = 0f32;
    for (p, idx) in keep.iter() {
        acc += p;
        if u < acc {
            return *idx;
        }
    }
    keep.last().map(|(_, idx)| *idx).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

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
