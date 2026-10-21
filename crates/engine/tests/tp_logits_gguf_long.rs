//! **GGUF × TP2 长上下文对拍**(2026-10-20;立案 `gguf27b-1m` 双卡轮)。
//!
//! 背景:5-token 的 `gpu_gguf_27b_tp2`(src/tests_gguf.rs)全绿,但真机 chat 在
//! **~1.3k tok** 的 prompt 上输出胡言(单卡同 prompt 正常)。本门把判据搬到
//! **可复现的最后一词 logits** 上 —— 不经过 chat 模板,不看文本,只看数值。
//!
//! 判据:
//! ① 同一 prompt(≈1300 tok)上 TP1 与 TP2 的**末位置 logits** 应同档:
//!    `cos > 0.99`,top-1 相同(TP 只改归约序,不能改信息面);
//! ② 两 rank logits 逐位相同(AR 统一)。
//!
//! 运行(**两张空闲的 3090 Ti**;未设 = skip):
//! ```bash
//! OWL_GGUF_TP_DEV_A=41:00.0 OWL_GGUF_TP_DEV_B=42:00.0 \
//!   OWL_GGUF_TP_WORDS=900 OWL_GGUF_TP_CHUNK=512 \
//!   cargo test -p owl-engine --test tp_logits_gguf_long -- --nocapture
//! ```

use owl_engine::{Engine, EngineConfig, EngineKnobs, RunningEngine, TpEngine, TurnEvent};
use owl_iface::contract::ModelError;
use owl_shared::testkit::{resource, resource_ok};

fn cfg_of(ordinal: usize, max_seq: usize, chunk: usize) -> EngineConfig {
    let mut knobs = EngineKnobs { sampler_enabled: false, raw_completion: true, ..Default::default() };
    knobs.pool_tokens = Some(max_seq);
    knobs.gdn_slots = 1;
    knobs.host_cache_mb = 32;
    knobs.snap_host_mb = 0;
    knobs.snap_max = 0;
    knobs.prefix_cache = false;
    EngineConfig { device_ordinal: ordinal, max_seq_tokens: max_seq, prefill_chunk: chunk, knobs }
}

fn dev(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .filter(|s| !s.is_empty())
        .map(|bdf| owl_cuda::resolve_ordinal_by_bdf(&bdf).unwrap_or_else(|e| panic!("{e}")))
}

/// 长 prompt(词级拼装;`words` 约 = 1.28 × tok)
fn long_prompt(words: usize) -> String {
    let pool = [
        "apple", "river", "stone", "cloud", "mountain", "garden", "window", "silver",
        "forest", "island", "letter", "market", "number", "object", "paper", "quiet",
        "reason", "signal", "table", "united", "village", "winter", "yellow", "zebra",
        "fabric", "gentle", "hunter", "ivory", "jacket", "kernel", "ladder", "meadow",
    ];
    let half = words / 2;
    let mut s = String::with_capacity(words * 8 + 160);
    for i in 0..words {
        if i == half {
            s.push_str(" IMPORTANT FACT: the secret passcode is PASSCODE-8842. Remember it exactly. ");
        }
        s.push_str(pool[i % pool.len()]);
        s.push(' ');
    }
    s.push_str(
        "\nQuestion: What is the secret passcode stated in the IMPORTANT FACT sentence? Answer with just the passcode.\nAnswer:",
    );
    s
}

async fn last_logits(
    running: &mut RunningEngine<owl_engine::GpuFace>,
    prompt: &str,
    max_new: usize,
) -> Result<(Vec<f32>, Vec<u32>), ModelError> {
    let (sid, turn) = running.submit_session_ex(Some(1), prompt, max_new, true)?;
    let prompt_len = running.turn_prompt_len(turn).unwrap_or(0);
    let mut logits: Option<Vec<f32>> = None;
    loop {
        let ev = running.pump().await?;
        if matches!(&ev, TurnEvent::Token { turn: t, .. } if *t == turn) && logits.is_none() {
            logits = running.read_output_f32("logits").await?;
        }
        match ev {
            TurnEvent::Idle => break,
            TurnEvent::Completed { turn: t, .. } if t == turn => break,
            TurnEvent::Failed { turn: t, err } => {
                return Err(ModelError::Msg(format!("t{t} 失败:{err}")))
            }
            _ => {}
        }
    }
    let toks = running.session_token_ids(sid).unwrap_or_default();
    let gen = if toks.len() > prompt_len { toks[prompt_len..].to_vec() } else { Vec::new() };
    let l = logits.ok_or_else(|| ModelError::Msg("末步读不到 logits 槽".into()))?;
    Ok((l, gen))
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::MIN), |a, (i, x)| if *x > a.1 { (i, *x) } else { a }).0
}

fn topn(v: &[f32], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_unstable_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    idx.into_iter().take(n).collect()
}

fn stats(a: &[f32], b: &[f32]) -> (f32, f64) {
    let mut dmax = 0f32;
    let (mut dot, mut n1, mut n2) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dmax = dmax.max((x - y).abs());
        dot += (*x as f64) * (*y as f64);
        n1 += (*x as f64).powi(2);
        n2 += (*y as f64).powi(2);
    }
    (dmax, dot / (n1.sqrt() * n2.sqrt()).max(1e-30))
}

/// ⚠️ **诊断仪,不是门**(2026-10-20 立案结论见 `docs/arch/gguf-格式支持-设计与施工.md` §23.8):
/// TP 的归约序与单卡不同 ⇒ 长 ctx 的 logits `cos` 天然低于 1,生成段会 near-tie 分叉。
/// 真正可判的是"**有/无宿主腿**"两臂的对比(见 §23.8 的实测表)。
#[test]
#[ignore = "诊断仪(cos/生成段不是 TP1↔TP2 的判据;见文件头与设计 §23.8)"]
fn tp_logits_gguf_long() {
    let (Some(oa), Some(ob)) = (dev("OWL_GGUF_TP_DEV_A"), dev("OWL_GGUF_TP_DEV_B")) else {
        owl_shared::testkit::report::skip("需 OWL_GGUF_TP_DEV_A / OWL_GGUF_TP_DEV_B(两张空闲卡的 BDF)");
        return;
    };
    assert_ne!(oa, ob, "TP2 需要两张不同的卡");
    if !resource_ok("gguf:27b") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:27b\"");
        return;
    }
    let words: usize = std::env::var("OWL_GGUF_TP_WORDS").ok().and_then(|s| s.parse().ok()).unwrap_or(900);
    let chunk: usize = std::env::var("OWL_GGUF_TP_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(body(oa, ob, words, chunk))
        })
        .expect("起测试线程")
        .join()
        .expect("门失败");
}

async fn body(oa: usize, ob: usize, words: usize, chunk: usize) {
    let path = std::path::PathBuf::from(resource("gguf:27b").expect("列了就有"));
    let prompt = long_prompt(words);
    eprintln!("[gguf-tp-long] chunk={chunk} · 词 {words} · 设备 A={oa} B={ob}");

    let skip_ref = std::env::var_os("OWL_GGUF_TP_SKIP_REF").is_some();
    // ── ① 单卡参考(最后一词 logits)──
    let (ref_logits, ref_gen) = if skip_ref { (Vec::new(), Vec::new()) } else {
        let mut e = Engine::new(cfg_of(oa, 4096, chunk)).expect("engine(单卡参考)");
        let loaded = e.loader().load_gguf(&path).await.expect("单卡装载");
        let mut r = e.run(loaded).await.expect("单卡装配");
        let v = last_logits(&mut r, &prompt, 16).await.expect("单卡末步 logits");
        drop(r);
        v
    };

    // ── ② TP2(同 prompt)──
    let mut e0 = Engine::new(cfg_of(oa, 4096, chunk)).expect("engine(r0)");
    let m0 = e0.loader().slice(0, 2).load_gguf(&path).await.expect("rank0 分片装载");
    let mut e1 = Engine::new_with_ledger(cfg_of(ob, 4096, chunk), e0.ledger_cell(), 1).expect("engine(r1)");
    let m1 = e1.loader().slice(1, 2).load_gguf(&path).await.expect("rank1 分片装载");
    let tp = TpEngine::assemble(vec![e0, e1], vec![m0, m1]).await.expect("TP2 装配");
    let mut r = tp.run().expect("TP2 进执行态");
    let tp_out = last_logits(&mut r, &prompt, 16).await.expect("TP2 末步 logits");
    let peer = r.read_peer_output_f32("logits").await.expect("读 peer logits");
    let _ = r.shutdown_tp().await;

    let (tp_logits, tp_gen) = tp_out;
    let first_div = ref_gen.iter().zip(tp_gen.iter()).position(|(a, b)| a != b);
    eprintln!("[gguf-tp-long] 单卡 gen = {ref_gen:?}");
    eprintln!("[gguf-tp-long] TP2  gen = {tp_gen:?}(首分歧 {:?})", first_div);
    let (dmax, cos) = stats(&ref_logits, &tp_logits);
    let (t1a, t1b) = (argmax(&ref_logits), argmax(&tp_logits));
    let (top_a, top_b) = (topn(&ref_logits, 10), topn(&tp_logits, 10));
    let hit = top_a.iter().filter(|x| top_b.contains(x)).count();
    eprintln!(
        "[gguf-tp-long] max|Δ| = {dmax:.4} · cos = {cos:.8} · top1 {t1a} vs {t1b} · top10 重合 {hit}/10"
    );
    if let Some(p) = peer {
        let nbit = p.iter().zip(tp_logits.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("[gguf-tp-long] 两 rank logits:{nbit} 项不同(应 0)");
    }
    if !skip_ref {
        assert_eq!(first_div, None, "生成段与单卡在超步 {:?} 分叉", first_div);
        assert!(cos > 0.99, "cos = {cos:.6} < 0.99 —— TP2 在 {words} 词(~1.3k tok)上信息面已偏");
        assert_eq!(t1a, t1b, "top1 不同 —— TP2 长上下文分片/AR 有实质错");
    }
}
