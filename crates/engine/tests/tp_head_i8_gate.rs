//! **TP2 × int8 lm_head 门**(P1.5-a 双卡接线;2026-10-20 立)。
//!
//! 判据三件(顺序有义):
//! 1. **结构**:词表切档下每 rank 的 logits 宽 = `vocab/2`(i8 头与 bf16 头
//!    同源 ⇒ 派生键 `lm_head.i8q/i8s` 确实按 `UniformCol` 切了行;宽度不等
//!    = 装载规则/头槽不一致);
//! 2. **可跑**:TP2 装配 + 双 rank 图捕获 + 一个 turn 全链(i8 头
//!    m=1 走 `owl_gemv_i8w_*`,spec verify m=8 / DFlash2 propose m=7
//!    走 `owl_gemv_i8w_mt_*`;W8A8 臂同路走 `*_a8_*`);
//! 3. **数值**:与 bf16 头的 **token 级**一致率 ≥ 0.95(两进程各跑一臂再比;
//!    同进程连跑两臂 = 两次 27B 装载,第二次被 VRAM 池守卫拒)。
//!
//! # spec 臂(2026-10-22 质量门)
//!
//! `OWL_TP_HEAD_SPEC=1` 时按生产 DFlash2 d7 启动:target verify = 8 行,
//! draft propose 复用 target lm_head = 7 行 ⇒ 这正是 W8A8 的 m=7/8 路径。
//! `OWL_TP_HEAD_DRAFT` 可覆盖草稿目录;缺省取 target 同级的 syvai 档。
//! 结束时打印 `spec.rounds` / `spec.m0..m7`,轮次 > 0 即证明 verify 走了
//! m=8(DFlash2 的 verify 行数 = depth+1,不是 1)。
//!
//! # 运行(两进程 + 一次比对)
//!
//! ```bash
//! OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_TP_HEAD_ARM=f16 \
//!   cargo test -p owl-engine --test tp_head_i8_gate -- --nocapture
//! OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_TP_HEAD_ARM=i8 \
//!   cargo test -p owl-engine --test tp_head_i8_gate -- --nocapture
//! OWL_TP_HEAD_ARM=compare cargo test -p owl-engine --test tp_head_i8_gate -- --nocapture
//! OWL_TP_HEAD_ARM=a8 ...       # W8A8 臂
//! OWL_TP_HEAD_ARM=compare_a8 ... # i8 vs a8 一致率
//! OWL_TP_HEAD_SPEC=1 OWL_TP_HEAD_ARM=i8 ...   # spec m=7/8 质量门
//! OWL_TP_HEAD_SPEC=1 OWL_TP_HEAD_ARM=a8 ...
//! OWL_TP_HEAD_ARM=compare_a8 ...
//!
//! # 32K fixed-tail / 采样档（2026-10-23）
//!
//! ```bash
//! python3 tools/fixedtail_prompt.py --length 32768 --out /tmp/fixedtail32k.txt
//! OWL_TP_HEAD_SPEC=1 OWL_TP_HEAD_PROMPT_FILE=/tmp/fixedtail32k.txt \
//! OWL_TP_HEAD_CTX=35000 OWL_TP_HEAD_N=64 OWL_TP_HEAD_ARM=i8 ... # 再跑 f16 后 compare
//! OWL_TP_HEAD_SAMPLE=1 OWL_TP_HEAD_TEMP=0.1 OWL_TP_HEAD_SEED=12345 ... # 采样门
//! OWL_TP_HEAD_DEPTH=15 ... # K=15 单 block（默认 depth=7）
//! OWL_TP_AR_FUSED=1 OWL_TP_AR2=1 OWL_TP_AR_F8=1 ... # TP AR 融合/AR-2 / 生产同参
//! ```
//! `A`/`B` = TP2 双 rank 卡(**CUDA ordinal**;本机 3080 对 = `2,3`)。
//! 臂文件 = `$OWL_TP_HEAD_OUT`(缺省 `/tmp/owl-tp-head-<arm>.txt`)。

use std::path::{Path, PathBuf};

use owl_engine::{Engine, EngineConfig, EngineKnobs, TpEngine, TpOpts};
use owl_models::module::QuantPlan;
use owl_shared::config::HeadQuant;

const PROMPTS: [&str; 6] = [
    "用五十字介绍长城。",
    "一个水池有甲乙两个进水管，甲管单独注满需要 6 小时，乙管单独注满需要 4 小时。两管同时打开，多少小时可以注满？",
    "用 Rust 写一个求中位数的函数。",
    "解释快速排序的基本思想，并给一个包含 8 个元素的例子。",
    "写一个 Python 脚本读取 CSV 文件，统计其中某一列的平均值，并处理空值。",
    "请用三百字说明为什么长上下文推理中固定尾部比随机前缀更难，并给出一个例子。",
];

fn dir_of() -> PathBuf {
    std::env::var("OWL_27B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/cyankiwi/Qwen3.8-27B-AWQ-INT4"))
        })
}

fn spec_on() -> bool {
    matches!(std::env::var("OWL_TP_HEAD_SPEC").as_deref(), Ok("1") | Ok("true"))
}

fn draft_dir_of(dir: &Path) -> PathBuf {
    std::env::var("OWL_TP_HEAD_DRAFT").map(PathBuf::from).unwrap_or_else(|_| {
        let models = dir
            .parent()
            .and_then(|p| p.parent())
            .unwrap_or_else(|| Path::new("."));
        models.join("syvai/Qwen3.8-27B-DFlash2-W4A16")
    })
}

/// 臂 → 配置头方案。**必须与 loader 的 head_plan 同源**:tag filter 从
/// `EngineKnobs.head_quant` 投影;只传 loader plan 而这里留 Off,会在
/// 装配期被 `owl_gemv_i8w_*` 的 QUANT_INT8 facet 响亮拒。
fn head_quant_of(plan: QuantPlan) -> HeadQuant {
    match plan {
        QuantPlan::F16 => HeadQuant::Off,
        QuantPlan::I8W => HeadQuant::RtnInt8Row,
        other => panic!("tp_head_i8_gate 不支持 head plan {other:?}"),
    }
}

/// 两臂同参:greedy(设备 argmax)+ 关前缀缓存(同 `tp_vocab_split_gate`)。
/// spec 臂按生产 DFlash2 d7:草稿双 rank 放行 + 单快照 + 160MB capture slab。
fn sample_on() -> bool {
    matches!(std::env::var("OWL_TP_HEAD_SAMPLE").as_deref(), Ok("1") | Ok("true"))
}

fn flag(k: &str) -> bool {
    matches!(std::env::var(k).as_deref(), Ok("1") | Ok("true"))
}

/// TP 装配档:默认全关(与旧 `assemble` 等价);`OWL_TP_AR_FUSED=1` /
/// `OWL_TP_AR2=1` 显式开融合/小包 AR-2,供 nsys/round A/B 使用。
fn tp_opts() -> TpOpts {
    TpOpts {
        ar_f8: flag("OWL_TP_AR_F8"),
        ar_fused: flag("OWL_TP_AR_FUSED"),
        ar2: flag("OWL_TP_AR2"),
    }
}

fn knobs_for(plan: QuantPlan, dir: &Path) -> EngineKnobs {
    let mut k = EngineKnobs::default();
    k.sampler_enabled = sample_on();
    if sample_on() {
        k.sampler.temp = std::env::var("OWL_TP_HEAD_TEMP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.7);
        k.sampler.topk = std::env::var("OWL_TP_HEAD_TOPK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(20);
        k.sampler.topp = std::env::var("OWL_TP_HEAD_TOPP")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.95);
    }
    k.prefix_cache = false;
    k.head_quant = head_quant_of(plan);
    if spec_on() {
        let draft = draft_dir_of(dir);
        assert!(draft.join("config.json").exists(), "DFlash2 草稿目录不在: {draft:?}");
        k.draft.depth = std::env::var("OWL_TP_HEAD_DEPTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(7);
        k.draft.dflash2_dir = Some(draft.display().to_string());
        k.draft.tp_draft = true;
        // 质量门隔离:lookup 是 host 侧草稿增强,不改变 head 路径;默认关,
        // 需要跟生产档同参时显式 OWL_TP_HEAD_LOOKUP=1。
        k.draft.lookup = std::env::var("OWL_TP_HEAD_LOOKUP").as_deref() == Ok("1");
        k.snap_max = 1;
        k.cuda.capture_slab_mb = Some(160);
        k.vram_reserve_mb = 256;
    }
    k
}

fn cfg_of(ordinal: usize, chunk: usize, knobs: EngineKnobs) -> EngineConfig {
    let max_seq = std::env::var("OWL_TP_HEAD_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: max_seq,
        prefill_chunk: chunk,
        knobs,
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
    prompts: &[String],
) -> (Vec<Vec<u32>>, usize) {
    let chunk: usize = std::env::var("OWL_TP_HEAD_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let mut e0 = Engine::new(cfg_of(oa, chunk, knobs_for(plan, dir))).expect("engine(r0)");
    let m0 = e0
        .loader()
        .slice(0, 2)
        .vocab_split(vsplit)
        .load_qwen38_27b_awq(dir, dir, plan, QuantPlan::W4A16Awq)
        .await
        .expect("rank0 分片装载");
    // rank1:账本/设备序**装载前**从 rank0 接过(TP 合流纪律)
    let mut e1 = Engine::new_with_ledger(
        cfg_of(ob, chunk, knobs_for(plan, dir)),
        e0.ledger_cell(),
        1,
    )
    .expect("engine(r1)");
    let m1 = e1
        .loader()
        .slice(1, 2)
        .vocab_split(vsplit)
        .load_qwen38_27b_awq(dir, dir, plan, QuantPlan::W4A16Awq)
        .await
        .expect("rank1 分片装载");
    let tp = TpEngine::assemble_with(vec![e0, e1], vec![m0, m1], tp_opts())
        .await
        .expect("TP2 装配");
    // ★ 词表切的结构判据(装配侧形状单源):两 rank 头等宽、合计 = 全词表;
    // int8 与 bf16 两臂**同形**(派生键按 `UniformCol` 切行 ⇒ 槽宽不变)。
    // 宽度**不靠**运行时 logits 槽读回:spec 档槽里可能是 8 行 verify logits。
    let width = {
        let sp0 = tp.spec(0);
        if vsplit {
            let sp1 = tp.spec(1);
            assert_eq!(sp0.head_vocab, sp1.head_vocab, "两 rank 半表不等");
            assert_eq!(sp0.head_vocab * 2, sp0.vocab, "半表 × 2 ≠ 全词表");
            eprintln!(
                "[tp-head] 词表切:头每 rank {} 行(全表 {})",
                sp0.head_vocab, sp0.vocab
            );
        } else {
            assert_eq!(sp0.head_vocab, sp0.vocab, "不切档时头应 = 全词表");
        }
        sp0.head_vocab
    };
    let mut r = tp.run().expect("TP2 进执行态");
    assert!(r.has_peer(), "TP2 执行态必须带 peer 束");
    let opts = tp_opts();
    eprintln!(
        "[tp-head] TpOpts: ar_f8={} ar_fused={} ar2={}",
        opts.ar_f8, opts.ar_fused, opts.ar2
    );

    // metrics 是进程级 OnceLock:必须在 turn 之前 init,否则 spec.* 全 0。
    let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());

    // 诊断:逐步打 top-2 边际(判"首分歧是否 near-tie";OWL_TP_HEAD_TOP2=1)。
    // spec 档 logits 槽可能是 [8, vocab] verify 块,跨行扫 top2 无意义 ⇒ 跳过。
    let top2 = std::env::var("OWL_TP_HEAD_TOP2").as_deref() == Ok("1") && !spec_on();
    let mut out = Vec::new();
    for (i, p) in prompts.iter().enumerate() {
        let sid = 7 + i as u64;
        let opts = if sample_on() {
            owl_engine::turn::TurnOpts {
                ignore_eos: true,
                sampler: Some(owl_engine::turn::TurnSampler {
                    temp: Some(std::env::var("OWL_TP_HEAD_TEMP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.7)),
                    topk: Some(std::env::var("OWL_TP_HEAD_TOPK").ok().and_then(|s| s.parse().ok()).unwrap_or(20)),
                    topp: Some(std::env::var("OWL_TP_HEAD_TOPP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.95)),
                }),
                seed: Some(std::env::var("OWL_TP_HEAD_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(12345)),
                ..Default::default()
            }
        } else {
            owl_engine::turn::TurnOpts { ignore_eos: true, ..Default::default() }
        };
        let (sid, turn) = r.submit_session_opts(Some(sid), p, n, opts).expect("submit");
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
                if top2 {
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
                        full.extend_from_slice(&b); // rank0 低半 ++ rank1 高半
                    }
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
    if spec_on() {
        let cnt = |t: &str| -> u64 {
            owl_shared::metrics::query_metrics(
                &owl_shared::metrics::MetricsFilter::new().tag(t),
            )
            .iter()
            .map(|x| x.counter)
            .sum()
        };
        let rounds = cnt("spec.rounds");
        let hist: Vec<u64> = (0..=owl_shared::config::SPEC_DEPTH_MAX)
            .map(|m| cnt(&format!("spec.m{m}")))
            .collect();
        let n_v1 = cnt("tp.ar.v1");
        let n_fused = cnt("tp.ar.fused");
        let n_ar2 = cnt("tp.ar.ar2");
        eprintln!(
            "[tp-head] spec 账:rounds={rounds} m 直方图 [m0..m{}]={hist:?} q={} none={} reject={}",
            owl_shared::config::SPEC_DEPTH_MAX,
            cnt("spec.q.round"),
            cnt("spec.q.none"),
            cnt("spec.reject")
        );
        eprintln!(
            "[tp-head] AR 路径账:v1={n_v1} fused={n_fused} ar2={n_ar2}"
        );
        if opts.ar2 {
            assert!(n_ar2 > 0, "开了 ar2 但一次都没走到");
            // 预填充/propose 的大包仍走 fused;这里只要求小包路径真的进 ar2。
            // 未开 fused 时,大包/预填充走 v1 两核也是预期。
        } else if opts.ar_fused {
            assert!(n_fused > 0, "开了 ar_fused 但一次都没走到");
            // fp8 大载荷腿仍走两核(融合核只非 fp8)⇒ ar_f8 档允许 v1 存在,
            // 但 decode 小载荷必须已走 fused(上面的 n_fused > 0)。
            if !opts.ar_f8 {
                assert_eq!(n_v1, 0, "融合档下不应走 v1 两核");
            }
        } else {
            assert!(n_v1 > 0, "未开融合时应有 v1 两核路径");
            assert_eq!(n_fused, 0, "未开融合却走了 fused 路径");
        }
        if !opts.ar2 {
            assert_eq!(n_ar2, 0, "未开 ar2 却走了 barrier 路径");
        }
        for tag in ["spec.round", "spec.verify", "spec.propose"] {
            let r = owl_shared::metrics::query_metrics(
                &owl_shared::metrics::MetricsFilter::new().tag(tag),
            );
            if let Some(r) = r.first() {
                eprintln!(
                    "[tp-head] {tag}: avg {:.3} ms · p50 {:.3} ms · n={}",
                    r.avg.as_secs_f64() * 1e3,
                    r.p50.as_secs_f64() * 1e3,
                    r.count,
                );
            }
        }
        assert!(rounds > 0, "spec 臂零轮:没有走到 m=8 verify / m=7 propose");
        assert_eq!(
            hist.iter().sum::<u64>(),
            rounds,
            "spec.m 直方图与 rounds 不闭合:历史轮/降级轮混入"
        );
    }
    (out, width)
}

#[test]
fn tp_head_i8_gate() {
    let arm = std::env::var("OWL_TP_HEAD_ARM").unwrap_or_default();
    let n: usize = std::env::var("OWL_TP_HEAD_N").ok().and_then(|s| s.parse().ok()).unwrap_or(96);
    let vsplit = match std::env::var("OWL_TP_HEAD_VSPLIT").as_deref() {
        Ok("0") | Ok("false") => false,
        _ => true,
    };
    // ── 比对臂(零 GPU):读两臂 token → 一致率 ───────────────────────
    if arm == "compare" || arm == "compare_a8" {
        let (left, right) = if arm == "compare_a8" { ("i8", "a8") } else { ("f16", "i8") };
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
        let (a, b) = (read(left), read(right));
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
        owl_shared::testkit::report::skip("需 OWL_TP_TEST_DEV_A / _B(TP2 双 rank;CUDA ordinal)");
        return;
    };
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        owl_shared::testkit::report::skip(format_args!("27B 检查点不在 {dir:?}"));
        return;
    }
    let prompts: Vec<String> = match std::env::var("OWL_TP_HEAD_PROMPT_FILE") {
        Ok(f) => vec![std::fs::read_to_string(&f).unwrap_or_else(|e| panic!("读 prompt 文件 {f} 失败:{e}"))],
        Err(_) => PROMPTS.iter().map(|s| (*s).to_string()).collect(),
    };
    let plan = match arm.as_str() {
        "f16" => QuantPlan::F16,
        "i8" => QuantPlan::I8W,
        "a8" => QuantPlan::I8W8,
        other => panic!("OWL_TP_HEAD_ARM 只认 f16|i8|a8|compare|compare_a8(得 {other:?})"),
    };
    std::thread::Builder::new()
        .stack_size(128 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(async move {
                    let (toks, width) = run_arm(&dir, oa, ob, plan, vsplit, n, &prompts).await;
                    let vocab = 248320usize;
                    if vsplit {
                        assert_eq!(width * 2, vocab, "运行时 logits 宽 ×2 ≠ 词表({width}×2 vs {vocab})");
                        eprintln!("[tp-head] 运行时 logits 宽 = {width}(半表;×2 = {vocab})✓");
                    } else {
                        assert_eq!(width, vocab, "不切档时 logits 宽应 = 全词表({width} vs {vocab})");
                    }
                    let mut out = String::new();
                    for (p, t) in prompts.iter().zip(toks.iter()) {
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
