//! **DFlash2 并发会话门（27B / 单卡）**：两个活跃 turn 交错时，
//! 目标输出必须与各自串行跑逐 token 相同。
//!
//! 这是 per-session 草稿 ring（ring0/ring1）与 per-turn GDN 行的正确性门；
//! 不比较 AL（ring1 暂走 eager propose，速度另测）。
//!
//! 运行：
//! ```bash
//! OWL_TEST_DEVICE=41:00.0 \
//!   cargo test -p owl-engine --test dflash2_concurrent_gate -- --nocapture
//! ```
use owl_engine::turn::TurnStats;
use owl_engine::{Engine, EngineConfig, EngineKnobs, RunningEngine, TurnEvent};

const MAX_SEQ: usize = 512;
const MAX_NEW: usize = 16;

fn gpu_ordinal() -> Option<usize> {
    owl_cuda::test_device_ordinal_opt()
}

fn resource(name: &str) -> Option<String> {
    owl_shared::testkit::resource(name)
}

async fn boot(dir: &str, ddir: &str) -> RunningEngine<owl_engine::GpuFace> {
    let mut knobs = EngineKnobs {
        sampler_enabled: false,
        max_batch: 2,
        gdn_slots: 2,
        prefix_cache: false,
        raw_completion: true,
        ..Default::default()
    };
    knobs.draft.depth = 7;
    knobs.draft.dflash2_dir = Some(ddir.to_string());
    knobs.snap_device_slots = 1;
    knobs.pool_tokens = Some(2048);
    let mut engine = Engine::new(EngineConfig {
        device_ordinal: gpu_ordinal().expect("OWL_TEST_DEVICE"),
        max_context_length_tokens: MAX_SEQ,
        prefill_chunk: 32,
        knobs,
    })
    .expect("27B 构造");
    let loaded = engine
        .loader()
        .load_qwen38_27b_awq(
            std::path::Path::new(dir),
            std::path::Path::new(dir),
            owl_models::module::QuantPlan::F16,
            owl_models::module::QuantPlan::W4A16Awq,
        )
        .await
        .expect("27B 装载");
    engine.run(loaded).await.expect("27B 装配")
}

async fn drive_one(
    r: &mut RunningEngine<owl_engine::GpuFace>,
    sid: u64,
    prompt: &str,
) -> (Vec<u32>, TurnStats) {
    let (sid, turn) = r
        .submit_session_ex(Some(sid), prompt, MAX_NEW, true)
        .expect("submit");
    let plen = r.turn_prompt_len(turn).expect("prompt len");
    let stats = loop {
        match r.pump().await.expect("pump") {
            TurnEvent::Completed { turn: t, stats, .. } if t == turn => break stats,
            TurnEvent::Failed { turn: t, err } => panic!("t{t} 失败: {err}"),
            _ => {}
        }
    };
    let toks = r.session_token_ids(sid).expect("session 账");
    (toks[plen..].to_vec(), stats)
}

#[test]
fn dflash2_two_sessions_match_serial() {
    if gpu_ordinal().is_none() {
        eprintln!("[dflash2-conc] skip: OWL_TEST_DEVICE 未设");
        return;
    }
    let Some(dir) = resource("ckpt:27b-awq") else {
        eprintln!("[dflash2-conc] skip: ckpt:27b-awq 未设");
        return;
    };
    let Some(ddir) = resource("ckpt:dflash2") else {
        eprintln!("[dflash2-conc] skip: ckpt:dflash2 未设");
        return;
    };
    let p1 = "The quick brown fox jumps over the lazy dog. Count to five.";
    let p2 = "Name three colors, then stop.";

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async move {
        let mut r = boot(&dir, &ddir).await;
        // 串行臂（同引擎；prefix_cache=false ⇒ 无跨会话污染）。
        let (s1, st1) = drive_one(&mut r, 1, p1).await;
        let (s2, st2) = drive_one(&mut r, 2, p2).await;

        // 并发臂：两个不同会话同时提交。
        let (_, t1) = r
            .submit_session_ex(Some(11), p1, MAX_NEW, true)
            .expect("submit c1");
        let (sid1, _) = (11, t1);
        let (_, t2) = r
            .submit_session_ex(Some(12), p2, MAX_NEW, true)
            .expect("submit c2");
        let (sid2, _) = (12, t2);
        let plen1 = r.turn_prompt_len(t1).expect("p1 len");
        let plen2 = r.turn_prompt_len(t2).expect("p2 len");
        let (mut cst1, mut cst2) = (None, None);
        loop {
            match r.pump().await.expect("pump") {
                TurnEvent::Completed { turn, stats, .. } if turn == t1 => cst1 = Some(stats),
                TurnEvent::Completed { turn, stats, .. } if turn == t2 => cst2 = Some(stats),
                TurnEvent::Failed { turn, err } => panic!("c{turn} 失败: {err}"),
                TurnEvent::Idle => break,
                _ => {}
            }
            if cst1.is_some() && cst2.is_some() {
                break;
            }
        }
        let cst1 = cst1.expect("c1 stats");
        let cst2 = cst2.expect("c2 stats");
        let c1 = r.session_token_ids(sid1).expect("c1 账")[plen1..].to_vec();
        let c2 = r.session_token_ids(sid2).expect("c2 账")[plen2..].to_vec();
        let al = |st: &TurnStats| {
            if st.spec_rounds == 0 {
                0.0
            } else {
                st.spec_accepted as f64 / st.spec_rounds as f64
            }
        };
        eprintln!("[dflash2-conc] serial A={s1:?}\n[dflash2-conc] conc   A={c1:?}");
        eprintln!("[dflash2-conc] serial B={s2:?}\n[dflash2-conc] conc   B={c2:?}");
        eprintln!(
            "[dflash2-conc] serial AL A={:.2} B={:.2}; conc AL A={:.2} B={:.2}",
            al(&st1),
            al(&st2),
            al(&cst1),
            al(&cst2)
        );
        assert_eq!(
            s1, c1,
            "会话 A：并发与串行逐 token 不一致（目标 KV/GDN 串了？）"
        );
        assert_eq!(
            s2, c2,
            "会话 B：并发与串行逐 token 不一致（目标 KV/GDN 串了？）"
        );
        eprintln!("[dflash2-conc] ✅ 两会话并发逐 token 一致（DFlash2 ring0/ring1）");
    });
}
