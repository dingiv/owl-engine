//! **内存模型**(`context = "max"` 的核心;纯函数,无 GPU / 无权重)。
//!
//! 与引擎治理器(`crates/engine/src/state.rs` 的 `[vram-gov]`)**同一算式**:
//!
//! ```text
//! target_total = total × vram_target
//! fixed        = gdn_row × (gdn_slots + snap_max) + 图 slab 640MiB + warmup 138MiB
//! budget       = target_total − reserve − fixed − live(权重/运行底价)
//! cap          = budget / per_tok                （治理器会夹池到这个上限）
//! usable       = budget − activation − stability_margin
//! pool_tokens  = usable / per_tok                （再按 4k 向下取整）
//! ```
//!
//! 治理器**不**知道 `activation`(chunk 工作集)与长跑泄漏,所以手写档要么
//! 过保守、要么 160k 跑 15 轮后撞 reserve(2026-10-20 现场)。本模块把这两项
//! 显式建模,`--check` 把每一项依据打出来。
//!
//! ## 校准(2026-10-20,单 3090 Ti + 27B AWQ-INT4 / FP8 KV / spec off)
//!
//! | ctx | 池 | 结果 |
//! |---:|---:|---|
//! | 112k | 3.5G | std 15/15 绿 |
//! | 128k | 4.0G | **15/15 绿(档位取此)** |
//! | 144k | 4.5G | 15/15 绿 |
//! | 160k | 5.0G | ~15 轮后撞 reserve(存量 turn 临时块泄漏) |
//!
//! 取 `activation = 0.60 GiB`(512+512 桶档实测 0.4–0.6)、
//! `stability_margin = 1.30 GiB`(覆盖已实测的泄漏曲线 + 激活抖动)⇒ 解出
//! **128k**(与 `profiles/qwen38_27b-3090-w4a16.toml` 同档,便于 A/B 对拍)。
//! P2 的校准探测会把这套数字写回硬件档。

use serde::Serialize;

use super::intent::ModelProfile;

/// 图 slab 预算(保守值;engine `GRAPH_SLAB_BUDGET` 桶退役后 = 320)。
///
/// ⚠️ 这里**故意保留 640**:降到 320 会把 `context=max` 从 131072 抬到 143360,
/// 贴近已知 144k/15 轮 reserve 泄漏带。等 144k 泄漏校准后再两边同步降;
/// 当前 `--check` 对图项偏保守 320MiB(不影响安全,只少报可用池)。
pub const GRAPH_SLAB_BYTES: u64 = 640 << 20;
/// warmup 瞬态 + 杂项(engine `WARMUP_MISC_BUDGET`)
pub const WARMUP_MISC_BYTES: u64 = 138 << 20;
/// 长跑稳定裕度(已知 turn 临时块泄漏 + 激活抖动;P2 校准探测后收紧)
pub const STABILITY_MARGIN_BYTES: u64 = 1_300 << 20;
/// `context = "max"` 的向下取整粒度(token;避免 147,584 这类数字)
pub const CONTEXT_ROUND_TOKENS: usize = 4096;

/// 固定项分解(与 `[vram-gov] fixed_after=[...]` 同口径)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct FixedParts {
    pub gdn_grid: u64,
    pub gdn_snap: u64,
    pub graph_slab: u64,
    pub warmup_misc: u64,
}

impl FixedParts {
    pub fn total(&self) -> u64 {
        self.gdn_grid + self.gdn_snap + self.graph_slab + self.warmup_misc
    }
}

/// 一次估算的完整输入(全部为**最终**配置值,不是意图)
#[derive(Clone, Copy, Debug)]
pub struct MemoryInputs {
    pub total_bytes: u64,
    pub vram_target: f64,
    pub reserve_bytes: u64,
    pub model: &'static ModelProfile,
    pub spec_dflash: bool,
    pub kv_fp8: bool,
    pub gdn_slots: usize,
    pub snap_max: usize,
    /// 并发会话数(非超卖档池 = 上下文 × sessions;每会话一份整上下文预算)
    pub sessions: usize,
    /// TP world(1/2;2 = 权/KV/GDN 按卡摊,视觉塔每 rank 全量)
    pub tp_world: usize,
    /// 本档是否装视觉塔(权重每 rank 全量)
    pub vision: bool,
    pub activation_bytes: u64,
    pub activation_note: &'static str,
    pub stability_margin_bytes: u64,
}

/// 估算结果(`--check` / `--print-config` / 启动摘要共用)
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MemoryEstimate {
    pub total_bytes: u64,
    pub vram_target: String,
    pub target_total_bytes: u64,
    pub reserve_bytes: u64,
    pub weights_bytes: u64,
    pub fixed: FixedParts,
    pub fixed_bytes: u64,
    pub per_tok_main: u64,
    pub per_tok_draft: u64,
    pub per_tok: u64,
    pub activation_bytes: u64,
    pub activation_note: String,
    pub stability_margin_bytes: u64,
    /// 治理器预算(不含 activation/margin)
    pub budget_bytes: u64,
    /// 治理器池上限(= budget / per_tok;手写值超过它会被静默夹)
    pub cap_tokens: usize,
    pub pool_tokens: usize,
    pub pool_bytes: u64,
    pub context_tokens: usize,
    pub non_overcommit: bool,
    /// `budget − activation − margin − pool`
    pub slack_bytes: i64,
    pub calibrated: bool,
    pub notes: Vec<String>,
}

impl MemoryEstimate {
    /// 估算使用的总占用(含 activation/margin)
    pub fn used_bytes(&self) -> u64 {
        self.weights_bytes + self.fixed_bytes + self.pool_bytes + self.activation_bytes
            + self.stability_margin_bytes
    }
}

/// 激活面(按 chunk+tail 的桶档查表;上界超过 1024 = 无桶 eager)
///
/// 数值来自 2026-10-20 单 3090 Ti / 27B 实测(chunk 512 档峰 0.4–0.6G,
/// chunk 1024 档 0.8–1.2G,无桶 2048 块 ~1.6G);取各档上界。
pub fn activation_for(chunk: usize, tail_merge: usize) -> (u64, &'static str) {
    // ① 上界超过桶阶梯顶 ⇒ 该块没有桶图,走 eager(1.6–1.8G);
    // ② 否则按**最大块**分档(512+512 仍按 512 档 —— 尾块合并只在末块发生,
    //    实测峰值由常规 512 块主导,见档位实测 0.4–0.6G)。
    let upper = chunk.saturating_add(tail_merge);
    let block = chunk.max(tail_merge);
    if upper > 1024 {
        (1_700 << 20, "chunk+tail>1024(无对应桶 ⇒ eager;激活约 1.6–1.8G)")
    } else if block > 512 {
        (1_050 << 20, "最大块≤1024(1024 桶档;激活约 0.8–1.2G)")
    } else {
        (600 << 20, "最大块≤512(512 桶档;激活约 0.4–0.6G)")
    }
}

/// 治理器预算与固定项(纯算式;`context` 求解与实际校验共用)
/// 本档每卡权重(TP 摊分;视觉塔每 rank 全量)
pub fn weights_bytes(inp: &MemoryInputs) -> u64 {
    let p = inp.model;
    let world = inp.tp_world.max(1) as u64;
    let base = if world == 2 && p.weights_base_tp2_bytes > 0 {
        p.weights_base_tp2_bytes
    } else {
        p.weights_base_bytes / world
    };
    let draft = if inp.spec_dflash {
        if world == 2 && p.draft_weights_tp2_bytes > 0 {
            p.draft_weights_tp2_bytes
        } else {
            p.draft_weights_bytes / world
        }
    } else {
        0
    };
    base + draft + if inp.vision { p.vision_weights_bytes } else { 0 }
}

/// 本档每卡每 token KV(主池 + 草稿池都按 world 摊)
pub fn per_tok_bytes(inp: &MemoryInputs) -> u64 {
    let world = inp.tp_world.max(1) as u64;
    (inp.model.kv_bytes_per_token(inp.kv_fp8)
        + if inp.spec_dflash { inp.model.draft_bytes_per_token() } else { 0 })
        / world
}

pub fn budget_parts(inp: &MemoryInputs) -> (FixedParts, u64, u64, usize) {
    let world = inp.tp_world.max(1) as u64;
    let gdn_row = inp.model.gdn_row_bytes() / world;
    let fixed = FixedParts {
        gdn_grid: gdn_row * inp.gdn_slots as u64,
        gdn_snap: gdn_row * inp.snap_max as u64,
        graph_slab: GRAPH_SLAB_BYTES,
        warmup_misc: WARMUP_MISC_BYTES,
    };
    let target_total = (inp.total_bytes as f64 * inp.vram_target) as u64;
    let weights = weights_bytes(inp);
    let spend = inp.reserve_bytes + fixed.total() + weights;
    let budget = target_total.saturating_sub(spend);
    let per_tok = per_tok_bytes(inp);
    let cap = (budget / per_tok.max(1)) as usize;
    (fixed, target_total, budget.max(0), cap)
}

/// `context = "max"` 求解(非超卖:ctx = pool;超卖:ctx = model_max,pool = 预算)
///
/// 返回 `(context_tokens, pool_tokens, cap_tokens)`。
/// 求解失败(预算装不下一个 4k 块)⇒ Err 带完整账目。
pub fn solve_max(inp: &MemoryInputs, model_max: usize, overcommit: bool) -> Result<(usize, usize, usize), String> {
    let (fixed, _target_total, budget, cap) = budget_parts(inp);
    let reserve_for_pool = inp.activation_bytes + inp.stability_margin_bytes;
    if budget <= reserve_for_pool {
        return Err(format!(
            "内存模型无解:治理器预算 {:.2}G ≤ 激活 {:.2}G + 稳定裕度 {:.2}G\n\
             (total {:.2}G × {:.0}% − reserve {:.2}G − fixed {:.2}G − weights {:.2}G)\n\
             建议:减小 prefill_chunk / 关闭 reuse / 换更小模型或更大卡",
            budget as f64 / GB,
            inp.activation_bytes as f64 / GB,
            inp.stability_margin_bytes as f64 / GB,
            inp.total_bytes as f64 / GB,
            inp.vram_target * 100.0,
            inp.reserve_bytes as f64 / GB,
            fixed.total() as f64 / GB,
            crate::config::memory::weights_bytes(inp) as f64 / GB,
        ));
    }
    let usable = budget - reserve_for_pool;
    let per_tok = per_tok_bytes(inp);
    let sessions = inp.sessions.max(1) as u64;
    let raw = (usable / (per_tok.max(1) * sessions)) as usize;
    let rounded = raw / CONTEXT_ROUND_TOKENS * CONTEXT_ROUND_TOKENS;
    if rounded == 0 {
        return Err(format!(
            "内存模型无解:可用 {:.2}G / 每 token {}B < {} token(最小粒度)\n\
             建议:减小 prefill_chunk / 关闭 reuse / 降 kv 精度",
            usable as f64 / GB,
            per_tok,
            CONTEXT_ROUND_TOKENS
        ));
    }
    if overcommit {
        // 逻辑上下文可到模型上限;设备池只装工作集
        Ok((model_max, rounded.min(cap), cap))
    } else {
        // 池是**全局**容量:每个会话都可能吃满上下文 ⇒ pool = ctx × sessions
        let ctx = rounded.min(model_max);
        Ok((ctx, ctx.saturating_mul(inp.sessions.max(1)), cap))
    }
}

/// 用**最终**配置值算一份完整估算(显式 ctx/pool 也走这里)
///
/// `context_tokens`/`pool_tokens` 由 resolver 定;本函数只负责对账与告警。
pub fn estimate(
    inp: &MemoryInputs,
    context_tokens: usize,
    pool_tokens: usize,
    non_overcommit: bool,
) -> Result<MemoryEstimate, String> {
    let (fixed, target_total, budget, cap) = budget_parts(inp);
    let world = inp.tp_world.max(1) as u64;
    let per_tok_main = inp.model.kv_bytes_per_token(inp.kv_fp8) / world;
    let per_tok_draft = if inp.spec_dflash {
        inp.model.draft_bytes_per_token() / world
    } else {
        0
    };
    let per_tok = per_tok_main + per_tok_draft;
    let pool_bytes = pool_tokens as u64 * per_tok;
    let mut notes: Vec<String> = Vec::new();

    // ① 上下文超出治理器池上限(硬):非超卖档池会被夹 ⇒ 请求到不了 max_seq
    if non_overcommit && context_tokens > cap {
        return Err(format!(
            "上下文 {} tok 超过治理器池上限 {} tok(预算 {:.2}G / 每 token {}B)\n\
             —— 非超卖档池必须装得下整条上下文;请降 context / 减小 prefill_chunk / 关 reuse,\n\
             或改用 `[intent].kvmem = \"overcommit\"`(池小于上下文,需宿主腿)",
            context_tokens, cap, budget as f64 / GB, per_tok
        ));
    }
    // ② 显式池超过 cap:治理器会**静默夹**,这里显式告警(不拒 —— 高级覆盖可能是有意的)
    if pool_tokens > cap {
        notes.push(format!(
            "pool_tokens {} > 治理器上限 {} ⇒ 启动会被夹到 {} tok(显式覆盖请自行确认)",
            pool_tokens, cap, cap
        ));
    }
    // ③ 稳定裕度不足(144k 实测可跑,但长跑泄漏可能撞 reserve;响亮告警)
    let slack = budget as i64 - inp.activation_bytes as i64 - inp.stability_margin_bytes as i64
        - pool_bytes as i64;
    if slack < 0 {
        notes.push(format!(
            "余量 {:.2}G < 激活+稳定裕度 {:.2}G(超出 {:.2}G)—— \
             短测通常可跑;长跑 turn 临时块泄漏可能撞 reserve(另案修 TurnEnd)",
            (budget as f64 - pool_bytes as f64) / GB,
            (inp.activation_bytes + inp.stability_margin_bytes) as f64 / GB,
            (-slack) as f64 / GB
        ));
    }
    if !inp.model.calibrated {
        notes.push(format!(
            "模型 {} 的权重字节为保守估计(未实测校准),context=max 可能偏小",
            inp.model.key
        ));
    }
    Ok(MemoryEstimate {
        total_bytes: inp.total_bytes,
        vram_target: format!("{:.0}%", inp.vram_target * 100.0),
        target_total_bytes: target_total,
        reserve_bytes: inp.reserve_bytes,
        weights_bytes: weights_bytes(inp),
        fixed,
        fixed_bytes: fixed.total(),
        per_tok_main,
        per_tok_draft,
        per_tok,
        activation_bytes: inp.activation_bytes,
        activation_note: inp.activation_note.to_string(),
        stability_margin_bytes: inp.stability_margin_bytes,
        budget_bytes: budget,
        cap_tokens: cap,
        pool_tokens,
        pool_bytes,
        context_tokens,
        non_overcommit,
        slack_bytes: slack,
        calibrated: inp.model.calibrated,
        notes,
    })
}

const GB: f64 = 1073741824.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::intent::{RTX_3090TI_24G, QWEN38_27B_AWQ};

    fn inputs(kv_fp8: bool, chunk: usize, tail: usize, snap_max: usize) -> MemoryInputs {
        let (act, note) = activation_for(chunk, tail);
        MemoryInputs {
            total_bytes: RTX_3090TI_24G.total_bytes,
            vram_target: 0.99,
            reserve_bytes: 256 << 20,
            model: &QWEN38_27B_AWQ,
            spec_dflash: false,
            kv_fp8,
            gdn_slots: 1,
            snap_max,
            sessions: 1,
            tp_world: 1,
            vision: false,
            activation_bytes: act,
            activation_note: note,
            stability_margin_bytes: STABILITY_MARGIN_BYTES,
        }
    }

    /// 3090 Ti + 27B/fp8/spec off/512+512:`context="max"` 必须解出 128k
    /// (与 `profiles/qwen38_27b-3090-w4a16.toml` 同档,供 A/B 对拍)。
    #[test]
    fn max_context_solves_to_128k_on_3090ti() {
        let inp = inputs(true, 512, 512, 0);
        let (fixed, _target, budget, cap) = budget_parts(&inp);
        // 与 boot 现场对齐(2026-10-20):fixed_after=0.90G / budget=5.95G / cap≈194k
        assert_eq!(fixed.total(), (926 << 20) + (1 << 19), "fixed = 148.5 + 778 = 926.5 MiB");
        assert!(
            (budget as f64 / GB - 5.95).abs() < 0.02,
            "budget={:.3}G 应 ≈5.95G",
            budget as f64 / GB
        );
        assert!((190_000..200_000).contains(&cap), "cap={cap}");
        let (ctx, pool, _) = solve_max(&inp, QWEN38_27B_AWQ.geom.max_pos, false).expect("有解");
        assert_eq!(ctx, 131_072, "128k:激活 0.60G + 裕度 1.30G 是本次定档");
        assert_eq!(pool, ctx, "非超卖档池 = 上下文");
    }

    /// 160k 必须被判为"裕度不足"(实测 15 轮后撞 reserve),但**不拒启**
    /// —— 显式覆盖是高级面,`--check` 只告警。
    #[test]
    fn explicit_160k_warns_but_does_not_reject() {
        let inp = inputs(true, 512, 512, 0);
        let e = estimate(&inp, 163_840, 163_840, true).expect("显式覆盖不拒");
        assert!(e.slack_bytes < 0, "160k 应余量不足:{e:?}");
        assert!(e.notes.iter().any(|n| n.contains("稳定裕度")), "{:?}", e.notes);
        assert_eq!(e.pool_bytes, 163_840 * 32768);
    }

    /// 上下文超过治理器上限 = 硬拒(非超卖档池会被夹 ⇒ 请求到不了 max_seq)
    #[test]
    fn context_over_governor_cap_is_rejected() {
        let inp = inputs(true, 512, 512, 0);
        let cap = budget_parts(&inp).3;
        let e = estimate(&inp, cap + 4096, cap + 4096, true).expect_err("超 cap 必拒");
        assert!(e.contains("超过治理器池上限"), "{e}");
    }

    /// 并发会话:池是全局容量 ⇒ pool = ctx × sessions,单会话上下文按并发切分
    #[test]
    fn concurrency_splits_context_budget() {
        let mut inp = inputs(true, 512, 512, 0);
        inp.sessions = 2;
        let (ctx, pool, cap) = solve_max(&inp, QWEN38_27B_AWQ.geom.max_pos, false).expect("有解");
        assert_eq!(ctx, 65_536, "4.05G / (32KiB × 2) ⇒ 64k 档");
        assert_eq!(pool, ctx * 2, "池必须装得下两条满上下文");
        assert!(pool <= cap, "pool={pool} cap={cap}");
        let e = estimate(&inp, ctx, pool, true).expect("估");
        assert_eq!(e.pool_tokens, 131_072);
        assert!(e.slack_bytes >= 0, "{e:?}");
    }

    /// 激活表:桶上界 >1024 必须走 eager 上界(切尾块 OOM 的回归门)
    #[test]
    fn activation_steps_at_bucket_top() {
        assert_eq!(activation_for(512, 0).0, 600 << 20);
        assert_eq!(activation_for(512, 512).0, 600 << 20);
        assert_eq!(activation_for(1024, 0).0, 1_050 << 20);
        assert_eq!(activation_for(1024, 1024).0, 1_700 << 20);
        assert!(activation_for(1024, 1024).1.contains("无对应桶"));
    }

    /// spec=dflash:草稿权重 + 草稿池都要进账(否则 context=max 会高估)
    #[test]
    fn dflash_accounts_draft_weights_and_pool() {
        let mut inp = inputs(true, 512, 512, 1);
        inp.spec_dflash = true;
        let e = estimate(&inp, 32768, 32768, true).expect("估");
        assert_eq!(e.per_tok_main, 32768);
        assert_eq!(e.per_tok_draft, 20480);
        assert_eq!(e.per_tok, 53248);
        assert_eq!(e.weights_bytes, (16_650 + 1_290) << 20);
        assert_eq!(e.fixed.gdn_snap, 155_713_536, "snap_max=1 ⇒ 一条 148.5 MiB");
    }
}
