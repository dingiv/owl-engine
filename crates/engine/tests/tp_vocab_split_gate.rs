//! **P1.1 词表切门**（27B;TP2 双卡;2026-10-19 立)—— 半表头 + host 拼接的端到端判据。
//!
//! 为什么是 27B:0.8B 是 **tied** 模型（`lm_head` 复用 embed 全量表）⇒ 词表切
//! 无据可依,装载期已**响亮拒**(那条判据在 `tp_g8_gate` 的 `OWL_TP_VOCAB_SPLIT`
//! 档)。本门用 **untied** 的 27B 验正路径。
//!
//! # 两臂(唯一差别 = 词表切开关;同卡对/同 prompt/greedy)
//!
//! - **臂 A**:TP2 不切档(每 rank 全表头)——
//! - **臂 B**:TP2 切档(每 rank 半表头;logits 由引擎 host 拼接)。
//!
//! # 判据(顺序有义)
//!
//! - **L1 结构**:切档两 rank 半表等宽、`半表 × 2 == 臂 A 全表宽`;
//! - **L2 数值**:逐步「拼接全表 vs 臂 A 全表」max|Δ| —— 上屏 + 阈值
//!   (半表 GEMM 与全表 GEMM 只差 kernel 切分,量级应 ≪ 1;> 1 即半表行
//!   取错/装载错位);
//! - **L3 功能**:两臂 token 序列 —— 相同 ⇒ 过;首分歧 ⇒ 必须**互为 top-N**
//!   (vLLM `check_logprobs_close` 同形判据;MTP/半表 GEMM 的 1ULP 差在并列
//!   附近可翻 token)。
//!
//! # 运行
//!
//! ```bash
//! OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 \
//!   OWL_27B_DIR=/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4 \
//!   cargo test -p owl-engine --test tp_vocab_split_gate -- --nocapture
//! ```
//! `A`/`B` = TP2 双 rank 卡（**CUDA ordinal**;需同型号,3080 对即可）。

use std::path::{Path, PathBuf};

use owl_engine::{Engine, EngineConfig, EngineKnobs, RunningEngine, TpEngine};
use owl_iface::contract::ModelError;
use owl_models::module::QuantPlan;

const PROMPT: &str = "The capital of France is";
const MAX_NEW: usize = 8;

fn dev(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn dir_of() -> PathBuf {
    std::env::var("OWL_27B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4")
        })
}

/// 两臂同参:greedy(设备 argmax;确定性)+ 关前缀缓存
fn knobs() -> EngineKnobs {
    let mut k = EngineKnobs::default();
    k.sampler_enabled = false;
    k.prefix_cache = false;
    k
}

fn cfg_of(ordinal: usize, chunk: usize) -> EngineConfig {
    EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: 1024,
        prefill_chunk: chunk,
        knobs: knobs(),
    }
}

/// **门内独立实现的半表拼接**（P1.1）—— 刻意**不与引擎实现同源**:
/// 判据若复用被测实现,同一个拼接 bug 会在两边同时“正确”。
/// 语义:单行、rank0(低半) ++ rank1(高半)。
fn concat_halves(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    out.extend_from_slice(a);
    out.extend_from_slice(b);
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "max_abs: 长度不等 {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

fn topn(v: &[f32], n: usize) -> Vec<u32> {
    let n = n.min(v.len());
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.select_nth_unstable_by(n, |&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    let mut out: Vec<(usize, f32)> = idx[..n].iter().map(|&i| (i, v[i])).collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    out.into_iter().map(|(i, _)| i as u32).collect()
}

/// 一臂的产物
struct Arm {
    /// 生成段 token
    gen: Vec<u32>,
    /// 逐步**全表** logits(切档臂 = 门内拼接;不切臂 = 直读)
    logits: Vec<Vec<f32>>,
    /// 末步 rank0 输出(切档 = 半表)
    l0_last: Vec<f32>,
    /// 末步 rank1 输出(切档 = 半表)
    l1_last: Vec<f32>,
}

/// 跑一臂:两 rank 分片装载(带/不带词表切)→ 装配 → 一个 turn(逐步收全表)。
async fn run_arm(dir: &Path, oa: usize, ob: usize, chunk: usize, vsplit: bool) -> Arm {
    let mut e0 = Engine::new(cfg_of(oa, chunk)).expect("engine(r0)");
    let m0 = e0
        .loader()
        .slice(0, 2)
        .vocab_split(vsplit)
        .load_qwen38_27b_awq(dir, dir, QuantPlan::F16, QuantPlan::W4A16Awq)
        .await
        .expect("rank0 分片装载");
    let mut e1 = Engine::new_with_ledger(cfg_of(ob, chunk), e0.ledger_cell(), 1).expect("engine(r1)");
    let m1 = e1
        .loader()
        .slice(1, 2)
        .vocab_split(vsplit)
        .load_qwen38_27b_awq(dir, dir, QuantPlan::F16, QuantPlan::W4A16Awq)
        .await
        .expect("rank1 分片装载");
    let tp = TpEngine::assemble(vec![e0, e1], vec![m0, m1])
        .await
        .expect("TP2 装配");
    let mut r = tp.run().expect("TP2 进执行态");
    assert!(r.has_peer(), "TP2 执行态必须带 peer 束");

    let (sid, turn) = r
        .submit_session_ex(Some(1), PROMPT, MAX_NEW, true)
        .expect("submit");
    let prompt_len = r.turn_prompt_len(turn).expect("prompt 长度");
    let mut logits: Vec<Vec<f32>> = Vec::new();
    loop {
        let ev = r.pump().await.expect("pump");
        if matches!(&ev, owl_engine::TurnEvent::Token { turn: t, .. } if *t == turn) {
            let a = r
                .read_output_f32("logits")
                .await
                .expect("rank0 logits")
                .expect("logits 槽");
            let full = if vsplit {
                let b = r
                    .read_peer_output_f32("logits")
                    .await
                    .expect("rank1 logits")
                    .expect("logits 槽");
                assert_eq!(a.len(), b.len(), "切档:两 rank 半表长度不等");
                concat_halves(&a, &b)
            } else {
                a
            };
            logits.push(full);
        }
        match ev {
            owl_engine::TurnEvent::Idle => break,
            owl_engine::TurnEvent::Failed { turn: t, err } => {
                panic!("t{t} 失败:{err}")
            }
            _ => {}
        }
    }
    let toks = r.session_token_ids(sid).expect("会话账不在");
    let gen = toks[prompt_len..].to_vec();
    let l0_last = r
        .read_output_f32("logits")
        .await
        .expect("rank0 末步")
        .expect("logits 槽");
    let l1_last = r
        .read_peer_output_f32("logits")
        .await
        .expect("rank1 末步")
        .expect("logits 槽");
    let _ = r.shutdown_tp().await;
    Arm { gen, logits, l0_last, l1_last }
}

#[test]
fn tp_vocab_split_gate() {
    let (Some(oa), Some(ob)) = (dev("OWL_TP_TEST_DEV_A"), dev("OWL_TP_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_TP_TEST_DEV_A / _B(TP2 双 rank;CUDA ordinal)");
        return;
    };
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        eprintln!("skip: 27B 检查点不在 {dir:?}");
        return;
    }
    std::thread::Builder::new()
        .stack_size(128 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(body(oa, ob, dir))
        })
        .expect("spawn")
        .join()
        .expect("join");
}

async fn body(oa: usize, ob: usize, dir: PathBuf) {
    let chunk: usize = std::env::var("OWL_TP_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    eprintln!("[vsplit] 27B TP2 {oa}/{ob};chunk={chunk}");

    // ── 臂 A:不切档(参考)──────────────────────────────────────────
    eprintln!("[vsplit] 臂 A(不切档)装载 + 跑 turn…");
    let a = run_arm(&dir, oa, ob, chunk, false).await;
    assert_eq!(a.l0_last.len(), a.l1_last.len(), "不切档:两 rank 全表应等宽");
    eprintln!(
        "[vsplit] 臂 A:{} tok;全表宽 {};gen={:?}",
        a.gen.len(),
        a.l0_last.len(),
        a.gen
    );

    // ── 臂 B:切档 ────────────────────────────────────────────────
    eprintln!("[vsplit] 臂 B(词表切)装载 + 跑 turn…");
    let b = run_arm(&dir, oa, ob, chunk, true).await;
    eprintln!(
        "[vsplit] 臂 B:{} tok;半表宽 {};gen={:?}",
        b.gen.len(),
        b.l0_last.len(),
        b.gen
    );

    // ── L1 结构 ─────────────────────────────────────────────────
    let half = b.l0_last.len();
    assert_eq!(half, b.l1_last.len(), "切档:两 rank 半表应等宽");
    assert_eq!(
        half * 2,
        a.l0_last.len(),
        "切档:半表 × 2 应 == 臂 A 全表宽({} ≠ {})",
        half * 2,
        a.l0_last.len()
    );
    eprintln!(
        "[vsplit] L1 结构:半表 {half} ×2 = 全表 {} ✓（广于不切档的 {}）",
        a.l0_last.len(),
        a.l0_last.len()
    );

    // ── L2 数值:逐步拼接全表 vs 不切全表 ────────────────────────
    assert_eq!(a.logits.len(), b.logits.len(), "两臂逐步 logits 条数不等");
    let per: Vec<f32> = a
        .logits
        .iter()
        .zip(&b.logits)
        .map(|(x, y)| max_abs(x, y))
        .collect();
    let worst = per.iter().cloned().fold(0f32, f32::max);
    let mean = per.iter().sum::<f32>() / per.len() as f32;
    eprintln!("[vsplit] L2 数值:逐步 max|Δ| = {per:?}");
    eprintln!("[vsplit] L2 数值:全程最大 {worst:.6} / 均值 {mean:.6}");
    assert!(
        worst < 1.0,
        "切档 vs 不切档的 logits 差异过大({worst})—— 半表行取错/装载错位嫌疑"
    );

    // ── L3 功能:token 序列 ──────────────────────────────────────
    let n_top = 5usize;
    assert_eq!(a.gen.len(), b.gen.len(), "两臂生成长度不等");
    let first_div = (0..a.gen.len()).find(|&i| a.gen[i] != b.gen[i]);
    match first_div {
        None => eprintln!("[vsplit] L3 功能:token 序列逐位相同({} tok)✓", a.gen.len()),
        Some(i) => {
            let (t1, t2) = (topn(&a.logits[i], n_top), topn(&b.logits[i], n_top));
            let ok = t1.contains(&b.gen[i]) && t2.contains(&a.gen[i]);
            eprintln!(
                "[vsplit] L3 首分歧步 {i}:A={}(rank {:?}) / B={}(rank {:?})",
                a.gen[i],
                t1.iter().position(|&x| x == a.gen[i]),
                b.gen[i],
                t2.iter().position(|&x| x == b.gen[i])
            );
            eprintln!("[vsplit] L3 A.top{n_top}={t1:?}");
            eprintln!("[vsplit] L3 B.top{n_top}={t2:?}");
            assert!(
                ok,
                "首分歧处不互为 top-{n_top} ⇒ 词表切引入真错(不是 1ULP 并列翻转)"
            );
            eprintln!("[vsplit] L3 功能:首分歧处互为 top-{n_top} ✓（vLLM 判据同形）");
        }
    }
    eprintln!("[vsplit] 全绿(L1 结构 + L2 数值 + L3 功能)");
}
