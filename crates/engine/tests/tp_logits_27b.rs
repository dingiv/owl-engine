//! **27B TP 首步 logits 对拍**(2026-10-17;S6 前哨诊断)—— 量级判别仪。
//!
//! 用途:TP1(单卡)与 TP2(双 rank)在同一 prompt 上的**首步 logits** 对比。
//! 判读:
//! - TP2 的 |logits| 量级远大于 TP1 ⇒ **残差流被逐层放大**(AR 面:如 peer 槽
//!   取到自己的部分和 = 每层 ×2);
//! - 量级同档但 top-N 完全不同 ⇒ **权重/布局面**(分片读错);
//! - 首步 logits 近同而后续发散 ⇒ 解码面(KV 池/采样)。
//!
//! 只跑**首步**(max_new = 1):不需要 KV 复用,也不需要多轮 —— 诊断面最小化。
//!
//! # 运行
//!
//! ```bash
//! OWL_TP_TEST_DEV_A=0 OWL_TP_TEST_DEV_B=2 OWL_TP_TEST_DEV_C=3 \
//!   OWL_27B_DIR=<…/Qwen3.8-27B-AWQ-INT4> \
//!   cargo test -p owl-engine --test tp_logits_27b -- --nocapture
//! ```
//! `A` = TP1 参考卡(单卡);`B`/`C` = TP2 双 rank(必须**同型号**)。

use std::path::PathBuf;

use owl_engine::{Engine, EngineConfig, EngineKnobs, RunningEngine, TpEngine};
use owl_iface::contract::ModelError;

const PROMPT: &str = "The capital of France is";

fn dev(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn dir_of() -> PathBuf {
    std::env::var("OWL_27B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4"))
}

fn knobs() -> EngineKnobs {
    let mut k = EngineKnobs::default();
    k.sampler_enabled = false; // 设备 argmax(逐 token 确定性)
    k.prefix_cache = false;
    k
}

fn cfg_of(ordinal: usize, chunk: usize) -> EngineConfig {
    EngineConfig { device_ordinal: ordinal, max_seq_tokens: 1024, prefill_chunk: chunk, knobs: knobs() }
}

/// 跑一个 turn(只取首步 logits;`max_new = 1`)
async fn first_logits(
    running: &mut RunningEngine<owl_cuda::GpuClient>,
    prompt: &str,
) -> Result<Vec<f32>, ModelError> {
    let (sid, turn) = running.submit_session_ex(Some(1), prompt, 1, true)?;
    let prompt_len = running
        .turn_prompt_len(turn)
        .ok_or_else(|| ModelError::Msg("取不到 prompt 长度".into()))?;
    let mut logits: Option<Vec<f32>> = None;
    let mut text = String::new();
    loop {
        let ev = running.pump().await?;
        if matches!(&ev, owl_engine::TurnEvent::Token { turn: t, .. } if *t == turn) && logits.is_none() {
            logits = running.read_output_f32("logits").await?;
        }
        match ev {
            owl_engine::TurnEvent::Idle => break,
            owl_engine::TurnEvent::Completed { text: out, .. } => text = out,
            owl_engine::TurnEvent::Failed { turn: t, err } => {
                return Err(ModelError::Msg(format!("t{t} 失败:{err}")))
            }
            _ => {}
        }
    }
    let toks = running
        .session_token_ids(sid)
        .ok_or_else(|| ModelError::Msg("会话账不在".into()))?;
    let gen = &toks[prompt_len..];
    eprintln!("  [diag] 生成 {} tok;首步文本 {:?}", gen.len(), text.trim());
    logits.ok_or_else(|| ModelError::Msg("首步读不到 logits 槽".into()))
}

fn topn(v: &[f32], n: usize) -> Vec<u32> {
    let n = n.min(v.len());
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.select_nth_unstable_by(n, |&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    let mut out: Vec<(usize, f32)> = idx[..n].iter().map(|&i| (i, v[i])).collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    out.into_iter().map(|(i, _)| i as u32).collect()
}

fn stats(v: &[f32]) -> (f32, f32, f32) {
    let mx = v.iter().cloned().fold(f32::MIN, f32::max);
    let mn = v.iter().cloned().fold(f32::MAX, f32::min);
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    (mn, mx, mean)
}

#[test]
fn tp_logits_27b() {
    let (Some(oa), Some(ob), Some(oc)) = (
        dev("OWL_TP_TEST_DEV_A"),
        dev("OWL_TP_TEST_DEV_B"),
        dev("OWL_TP_TEST_DEV_C"),
    ) else {
        eprintln!("skip: 需 OWL_TP_TEST_DEV_A(TP1 参考卡)/ _B / _C(TP2 双 rank)");
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
                .block_on(body(oa, ob, oc, dir))
        })
        .expect("spawn")
        .join()
        .expect("join");
}

async fn body(oa: usize, ob: usize, oc: usize, dir: PathBuf) {
    let chunk: usize = std::env::var("OWL_TP_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    eprintln!("[27b-logits] chunk={chunk};TP1=ordinal {oa};TP2={ob}/{oc}");

    // ── ① TP1 参考 ──
    let l1 = {
        let mut e = Engine::new(cfg_of(oa, chunk)).expect("engine(TP1)");
        let loaded = e.loader().load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, owl_models::module::QuantPlan::W4A16Awq)
            .await
            .expect("TP1 装载");
        let mut r = e.run(loaded).await.expect("TP1 装配");
        let v = first_logits(&mut r, PROMPT).await.expect("TP1 首步");
        // 单卡档无 shutdown 口(进程退出即释放);显式 drop 掉引擎
        drop(r);
        v
    };
    let (mn1, mx1, mean1) = stats(&l1);
    eprintln!("[27b-logits] TP1:{} 项 logits;min={mn1:.3} max={mx1:.3} mean={mean1:.4}", l1.len());
    eprintln!("[27b-logits] TP1 top8 = {:?}", topn(&l1, 8));

    // ── ② TP2 ──
    let mut e0 = Engine::new(cfg_of(ob, chunk)).expect("engine(r0)");
    let m0 = e0.loader().slice(0, 2)
        .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, owl_models::module::QuantPlan::W4A16Awq)
        .await.expect("rank0 分片装载");
    let mut e1 = Engine::new(cfg_of(oc, chunk)).expect("engine(r1)");
    let m1 = e1.loader().slice(1, 2)
        .load_qwen38_27b_awq(&dir, &dir, owl_models::module::QuantPlan::F16, owl_models::module::QuantPlan::W4A16Awq)
        .await.expect("rank1 分片装载");
    let tp = TpEngine::assemble(vec![e0, e1], vec![m0, m1]).await.expect("TP2 装配");
    let mut r = tp.run().expect("TP2 进执行态");
    let l2 = first_logits(&mut r, PROMPT).await.expect("TP2 首步");
    let l2r1 = r.read_peer_output_f32("logits").await.ok().flatten();
    // ★ staging 游标自证(turn 之后):两侧必须 produced == consumed 且 err == 0
    match r.tp_staging_state().await {
        Ok(Some((a, b))) => {
            eprintln!("[27b-logits] staging 游标:r0={a:?} r1={b:?}");
            eprintln!(
                "[27b-logits] 判据:r0 平衡={} r1 平衡={} err0={} err1={}",
                a.0 == a.1,
                b.0 == b.1,
                a.2,
                b.2
            );
        }
        Ok(None) => eprintln!("[27b-logits] staging:单卡档(无 peer)"),
        Err(e) => eprintln!("[27b-logits] staging 读失败:{e}"),
    }
    let _ = r.shutdown_tp().await;

    let (mn2, mx2, mean2) = stats(&l2);
    eprintln!("[27b-logits] TP2:{} 项 logits;min={mn2:.3} max={mx2:.3} mean={mean2:.4}", l2.len());
    eprintln!("[27b-logits] TP2 top8 = {:?}", topn(&l2, 8));
    if let Some(p) = &l2r1 {
        let nbit = p.iter().zip(&l2).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("[27b-logits] 两 rank logits:{nbit} 项不同(应 0)");
    }
    let mx = l1.iter().zip(&l2).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("[27b-logits] 逐步 max|Δ|(TP1 vs TP2)= {mx:.4}");
    eprintln!(
        "[27b-logits] **量级比** max(TP2)/max(TP1) = {:.3}(≈1 同档;≫1 = 逐层放大)",
        mx2 / mx1.max(1e-9)
    );
}
