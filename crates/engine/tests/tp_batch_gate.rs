//! **TP2 真批 decode 门**（2026-11-15）。
//!
//! 判据：
//! - 两个会话同时在场时，调度必须走到 `BatchDecode`（`batch_steps > 0`）；
//! - 两 rank 的 batch 图都必须在场（缺 peer 图 = 调度门失守）；
//! - 并发档的输出与各自串行跑**逐字相同**（greedy / ignore_eos）。
//!
//! 运行（两张空闲卡 = CUDA ordinal；未设 = skip）：
//! ```bash
//! OWL_TP_TEST_DEV_A=1 OWL_TP_TEST_DEV_B=2 \
//!   cargo test -p owl-engine --test tp_batch_gate -- --nocapture
//! ```
use std::path::PathBuf;

use owl_engine::{Engine, EngineConfig, EngineKnobs, RunningEngine, TpEngine, TurnEvent};

const MAX_SEQ: usize = 256;
const MAX_NEW: usize = 16;

fn dev(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn cfg_of(ordinal: usize) -> EngineConfig {
    EngineConfig {
        device_ordinal: ordinal,
        max_context_length_tokens: MAX_SEQ,
        prefill_chunk: 64,
        knobs: EngineKnobs {
            sampler_enabled: false,
            max_batch: 2,
            gdn_slots: 2,
            raw_completion: true,
            prefix_cache: false,
            // 0.8B 小档：池 = 2 × ctx（并发自动池的测试形状）。
            pool_tokens: Some(2 * MAX_SEQ),
            ..Default::default()
        },
    }
}

async fn load_tp2(oa: usize, ob: usize) -> TpEngine<owl_cuda::GpuClient> {
    let dir = std::env::var("OWL_0_8B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../models/Qwen/Qwen3.5-0.8B"));
    let mut e0 = Engine::new(cfg_of(oa)).expect("rank0 构造");
    let l0 = e0
        .loader()
        .slice(0, 2)
        .load_qwen35_0_8b(&dir)
        .await
        .expect("rank0 装载");
    let mut e1 = Engine::new_with_ledger(cfg_of(ob), e0.ledger_cell(), 1).expect("rank1 构造");
    let l1 = e1
        .loader()
        .slice(1, 2)
        .load_qwen35_0_8b(&dir)
        .await
        .expect("rank1 装载");
    TpEngine::assemble(vec![e0, e1], vec![l0, l1])
        .await
        .expect("TP2 装配")
}

async fn serial(
    r: &mut RunningEngine<owl_engine::GpuFace>,
    sid: u64,
    prompt: &str,
) -> String {
    let (_, turn) = r
        .submit_session_ex(Some(sid), prompt, MAX_NEW, true)
        .expect("submit");
    loop {
        match r.pump().await.expect("pump") {
            TurnEvent::Completed { turn: t, text, .. } if t == turn => return text,
            TurnEvent::Failed { turn: t, err } => panic!("t{t} 失败: {err}"),
            _ => {}
        }
    }
}

#[test]
fn tp2_batch_decode_matches_serial() {
    let (Some(oa), Some(ob)) = (dev("OWL_TP_TEST_DEV_A"), dev("OWL_TP_TEST_DEV_B")) else {
        eprintln!("[tp-batch] skip: OWL_TP_TEST_DEV_A/B 未设");
        return;
    };
    let p1 = "用一句话介绍长城。";
    let p2 = "What is the capital of France?";

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async move {
        let mut r = load_tp2(oa, ob).await.run().expect("TP2 进执行态");
        assert!(r.has_peer(), "TP2 执行态必须带 peer");
        assert!(r.has_batch_graph(), "rank0 batch 图缺失（捕获失败？）");

        // 串行基线（不同会话 id，避免前缀账互相影响）。
        let a1 = serial(&mut r, 101, p1).await;
        let a2 = serial(&mut r, 102, p2).await;
        assert_eq!(r.batch_steps(), 0, "串行臂不应走批路径");

        // 并发臂。
        let (_, t1) = r
            .submit_session_ex(Some(201), p1, MAX_NEW, true)
            .expect("submit c1");
        let (_, t2) = r
            .submit_session_ex(Some(202), p2, MAX_NEW, true)
            .expect("submit c2");
        let (mut b1, mut b2) = (None, None);
        loop {
            match r.pump().await.expect("pump") {
                TurnEvent::Completed { turn, text, .. } if turn == t1 => b1 = Some(text),
                TurnEvent::Completed { turn, text, .. } if turn == t2 => b2 = Some(text),
                TurnEvent::Failed { turn, err } => panic!("c{turn} 失败: {err}"),
                TurnEvent::Idle => break,
                _ => {}
            }
            if b1.is_some() && b2.is_some() {
                break;
            }
        }
        let (b1, b2) = (b1.expect("c1 完成"), b2.expect("c2 完成"));
        eprintln!("[tp-batch] serial A={a1:?}\n[tp-batch] serial B={a2:?}");
        eprintln!("[tp-batch] batch  A={b1:?}\n[tp-batch] batch  B={b2:?}");
        assert!(r.batch_steps() > 0, "并发臂必须真走到 BatchDecode");
        assert_eq!(a1, b1, "会话 A：批 decode 与串行不一致");
        assert_eq!(a2, b2, "会话 B：批 decode 与串行不一致");
        eprintln!("[tp-batch] ✅ TP2 批 decode 逐字一致；批步数 = {}", r.batch_steps());
        r.shutdown_tp().await.expect("TP2 收尾");
    });
}
