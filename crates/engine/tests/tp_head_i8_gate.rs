//! **TP2 × int8 lm_head 门**(P1.5-a 双卡接线;2026-10-20 立)。
//!
//! 判据三件(顺序有义):
//! 1. **结构**:词表切档下每 rank 的 logits 宽 = `vocab/2`(i8 头与 bf16 头
//!    同源 ⇒ 派生键 `lm_head.i8q/i8s` 确实按 `UniformCol` 切了行;宽度不等
//!    = 装载规则/头槽不一致);
//! 2. **可跑**:TP2 装配 + 双 rank 图捕获 + 一个 turn 全链(i8 头走
//!    `owl_gemv_i8w_{f16,bf16}`,prefill 与 decode 同核);
//! 3. **数值**:与 bf16 头的 **token 级**一致率 ≥ 0.95(两进程各跑一臂再比;
//!    同进程连跑两臂 = 两次 27B 装载,第二次被 VRAM 池守卫拒)。
//!
//! # 运行(两进程 + 一次比对)
//!
//! ```bash
//! OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_TP_HEAD_ARM=f16 \
//!   cargo test -p owl-engine --test tp_head_i8_gate -- --nocapture
//! OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_TP_HEAD_ARM=i8 \
//!   cargo test -p owl-engine --test tp_head_i8_gate -- --nocapture
//! OWL_TP_HEAD_ARM=compare cargo test -p owl-engine --test tp_head_i8_gate -- --nocapture
//! ```
//! `A`/`B` = TP2 双 rank 卡(**CUDA ordinal**;本机 3080 对 = `2,3`)。
//! 臂文件 = `$OWL_TP_HEAD_OUT`(缺省 `/tmp/owl-tp-head-<arm>.txt`)。

use std::path::{Path, PathBuf};

use owl_engine::{Engine, EngineConfig, EngineKnobs, TpEngine};
use owl_models::module::QuantPlan;

const PROMPTS: [&str; 3] = [
    "用五十字介绍长城。",
    "一个水池有甲乙两个进水管，甲管单独注满需要 6 小时，乙管单独注满需要 4 小时。两管同时打开，多少小时可以注满？",
    "用 Rust 写一个求中位数的函数。",
];

fn dir_of() -> PathBuf {
    std::env::var("OWL_27B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from("/home/div/Documents/codes/models/cyankiwi/Qwen3.8-27B-AWQ-INT4")
        })
}

/// 两臂同参:greedy(设备 argmax)+ 关前缀缓存(同 `tp_vocab_split_gate`)
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

fn art(a: &str) -> PathBuf {
    std::env::var("OWL_TP_HEAD_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(format!("/tmp/owl-tp-head-{a}.txt")))
}

/// 一臂:两 rank 分片装载(+词表切)→ 装配 → 逐 prompt 一个 turn。
///
/// 返回 `(每 prompt 生成 token, 本 rank logits 宽度)`。
async fn run_arm(
    dir: &Path,
    oa: usize,
    ob: usize,
    plan: QuantPlan,
    vsplit: bool,
    n: usize,
) -> (Vec<Vec<u32>>, usize) {
    let chunk: usize = std::env::var("OWL_TP_HEAD_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let mut e0 = Engine::new(cfg_of(oa, chunk)).expect("engine(r0)");
    let m0 = e0
        .loader()
        .slice(0, 2)
        .vocab_split(vsplit)
        .load_qwen38_27b_awq(dir, dir, plan, QuantPlan::W4A16Awq)
        .await
        .expect("rank0 分片装载");
    // rank1:账本/设备序**装载前**从 rank0 接过(TP 合流纪律)
    let mut e1 = Engine::new_with_ledger(cfg_of(ob, chunk), e0.ledger_cell(), 1).expect("engine(r1)");
    let m1 = e1
        .loader()
        .slice(1, 2)
        .vocab_split(vsplit)
        .load_qwen38_27b_awq(dir, dir, plan, QuantPlan::W4A16Awq)
        .await
        .expect("rank1 分片装载");
    let tp = TpEngine::assemble(vec![e0, e1], vec![m0, m1])
        .await
        .expect("TP2 装配");
    // ★ 词表切的结构判据(装配侧形状单源):两 rank 头等宽、合计 = 全词表;
    // int8 与 bf16 两臂**同形**(派生键按 `UniformCol` 切行 ⇒ 槽宽不变)
    if vsplit {
        let (sp0, sp1) = (tp.spec(0), tp.spec(1));
        assert_eq!(sp0.head_vocab, sp1.head_vocab, "两 rank 半表不等");
        assert_eq!(sp0.head_vocab * 2, sp0.vocab, "半表 × 2 ≠ 全词表");
        eprintln!(
            "[tp-head] 词表切:头每 rank {} 行(全表 {})",
            sp0.head_vocab, sp0.vocab
        );
    }
    let mut r = tp.run().expect("TP2 进执行态");
    assert!(r.has_peer(), "TP2 执行态必须带 peer 束");

    // 诊断:逐步打 top-2 边际(判"首分歧是否 near-tie";OWL_TP_HEAD_TOP2=1)
    let top2 = std::env::var("OWL_TP_HEAD_TOP2").as_deref() == Ok("1");
    let mut out = Vec::new();
    let mut width = 0usize;
    for (i, p) in PROMPTS.iter().enumerate() {
        let sid = 7 + i as u64;
        let (sid, turn) = r
            .submit_session_ex(Some(sid), *p, n, true)
            .expect("submit");
        let prompt_len = r.turn_prompt_len(turn).expect("prompt 长度");
        // 逐步计时(decode 段):首 token 到末 token 的墙钟 ⇒ ms/tok。
        // 24B 的 int8 头收益 ≈ 0.87 ms/tok(1.76 → 0.89),要分辨它需 ≥256 tok。
        let mut t_first: Option<std::time::Instant> = None;
        let mut t_last = std::time::Instant::now();
        let mut n_tok = 0usize;
        loop {
            let ev = r.pump().await.expect("pump");
            if matches!(&ev, owl_engine::TurnEvent::Token { turn: t, .. } if *t == turn) {
                let now = std::time::Instant::now();
                if t_first.is_none() {
                    t_first = Some(now);
                } else {
                    n_tok += 1;
                }
                t_last = now;
                let a = r
                    .read_output_f32("logits")
                    .await
                    .expect("rank0 logits")
                    .expect("logits 槽");
                let mut full = a;
                if vsplit {
                    let b = r
                        .read_peer_output_f32("logits")
                        .await
                        .expect("rank1 logits")
                        .expect("logits 槽");
                    assert_eq!(full.len(), b.len(), "切档:两 rank 半表长度不等");
                    width = full.len();
                    full.extend_from_slice(&b); // rank0 低半 ++ rank1 高半
                } else {
                    width = full.len();
                }
                if top2 {
                    let mut best = (0usize, f32::NEG_INFINITY);
                    let mut second = (0usize, f32::NEG_INFINITY);
                    for (j, &v) in full.iter().enumerate() {
                        if v > best.1 {
                            second = best;
                            best = (j, v);
                        } else if v > second.1 {
                            second = (j, v);
                        }
                    }
                    eprintln!(
                        "[tp-head] p{i} top1={}({:.4}) top2={}({:.4}) margin={:.2e}",
                        best.0, best.1, second.0, second.1, best.1 - second.1
                    );
                }
            }
            match ev {
                owl_engine::TurnEvent::Idle => break,
                owl_engine::TurnEvent::Failed { turn: t, err } => panic!("t{t} 失败:{err}"),
                _ => {}
            }
        }
        let toks = r.session_token_ids(sid).expect("会话账不在");
        out.push(toks[prompt_len..].to_vec());
        let ms = t_first
            .map(|t0| t_last.duration_since(t0).as_secs_f64() * 1e3 / (n_tok.max(1) as f64))
            .unwrap_or(f64::NAN);
        eprintln!(
            "[tp-head] #{} {:?} → {} tok · decode {:.2} ms/tok(含首 token 后的 {} 步)",
            i,
            p.chars().take(14).collect::<String>(),
            out[i].len(),
            ms,
            n_tok
        );
    }
    let _ = r.shutdown_tp().await;
    (out, width)
}

#[test]
fn tp_head_i8_gate() {
    let arm = std::env::var("OWL_TP_HEAD_ARM").unwrap_or_default();
    let n: usize = std::env::var("OWL_TP_HEAD_N").ok().and_then(|s| s.parse().ok()).unwrap_or(48);
    let vsplit = match std::env::var("OWL_TP_HEAD_VSPLIT").as_deref() {
        Ok("0") | Ok("false") => false,
        _ => true,
    };
    // ── 比对臂(零 GPU):读两臂 token → 一致率 ───────────────────────
    if arm == "compare" {
        let read = |a: &str| -> Vec<(String, Vec<u32>)> {
            std::fs::read_to_string(art(a))
                .unwrap_or_else(|e| panic!("读 {a} 臂失败({}):{e} —— 先跑两臂", art(a).display()))
                .lines()
                .filter_map(|l| {
                    let (p, rest) = l.split_once('\t')?;
                    Some((
                        p.to_string(),
                        rest.split_whitespace().filter_map(|t| t.parse().ok()).collect(),
                    ))
                })
                .collect()
        };
        let (a, b) = (read("f16"), read("i8"));
        assert_eq!(a.len(), b.len(), "两臂 prompt 数不等");
        let (mut agree, mut total, mut exact) = (0usize, 0usize, 0usize);
        for ((pa, ta), (_pb, tb)) in a.iter().zip(b.iter()) {
            let same = ta.iter().zip(tb.iter()).filter(|(x, y)| x == y).count();
            let first = ta.iter().zip(tb.iter()).position(|(x, y)| x != y);
            if ta == tb {
                exact += 1;
            }
            eprintln!(
                "[tp-head] {:?} · len {} vs {} · 相同 {same} · 首个分歧 {first:?}",
                pa.chars().take(22).collect::<String>(),
                ta.len(),
                tb.len()
            );
            agree += same;
            total += ta.len().max(tb.len());
        }
        let rate = agree as f64 / total.max(1) as f64;
        eprintln!(
            "[tp-head] **TP2 int8 头 token 一致率 = {rate:.4}**({agree}/{total});逐字全等 {exact}/{} ⇒ {}",
            a.len(),
            if rate >= 0.95 { "过门(≥0.95)" } else { "✗ 未过" }
        );
        assert!(rate >= 0.95, "TP2 int8 头精度门未过({rate:.4} < 0.95)");
        return;
    }
    let (Some(oa), Some(ob)) = (
        std::env::var("OWL_TP_TEST_DEV_A").ok().and_then(|s| s.parse().ok()),
        std::env::var("OWL_TP_TEST_DEV_B").ok().and_then(|s| s.parse().ok()),
    ) else {
        eprintln!("skip: 需 OWL_TP_TEST_DEV_A / _B(TP2 双 rank;CUDA ordinal)");
        return;
    };
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        eprintln!("skip: 27B 检查点不在 {dir:?}");
        return;
    }
    let plan = match arm.as_str() {
        "f16" => QuantPlan::F16,
        "i8" => QuantPlan::I8W,
        other => panic!("OWL_TP_HEAD_ARM 只认 f16|i8|compare(得 {other:?})"),
    };
    std::thread::Builder::new()
        .stack_size(128 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(async move {
                    let (toks, width) = run_arm(&dir, oa, ob, plan, vsplit, n).await;
                    let vocab = 248320usize;
                    if vsplit {
                        assert_eq!(width * 2, vocab, "运行时 logits 宽 ×2 ≠ 词表({width}×2 vs {vocab})");
                        eprintln!("[tp-head] 运行时 logits 宽 = {width}(半表;×2 = {vocab})✓");
                    } else {
                        assert_eq!(width, vocab, "不切档时 logits 宽应 = 全词表({width} vs {vocab})");
                    }
                    let mut out = String::new();
                    for (p, t) in PROMPTS.iter().zip(toks.iter()) {
                        out.push_str(&format!(
                            "{}\t{}\n",
                            p.replace('\t', " "),
                            t.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ")
                        ));
                    }
                    let path = art(&arm);
                    std::fs::write(&path, out).expect("写 token 文件");
                    eprintln!(
                        "[tp-head] {arm} 臂落盘:{}(比对 = OWL_TP_HEAD_ARM=compare)",
                        path.display()
                    );
                })
        })
        .expect("spawn")
        .join()
        .expect("join");
}
