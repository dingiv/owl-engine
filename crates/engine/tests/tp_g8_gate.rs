//! **G8-b:vs 单卡功能等价门**(vLLM `check_logprobs_close` 判据)。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §19.1(vLLM 判例)/ §19.2(门表)/ §20.2 T2。
//!
//! # 判据(三档,顺序有义)
//!
//! - **L1(接线)**:两 rank 的末步 logits **逐位相同**(AR 协议/别名/序的实证);
//! - **L2(等价,G8-b 本体)**:TP1 与 TP2 的生成序列逐位置比较 ——
//!   token 相同 ⇒ 继续;不同 ⇒ **必须互为对方 top-N 候选**(logits 自算),
//!   否则**硬失败**;**分歧即止**(首个不满足处停止比较);
//! - **L3(记账)**:首分歧步 / 分歧率 / 逐步 max|Δ| 上屏(攒数据;阈值待定,见 §20.3.1)。
//!
//! # 为什么不是"逐 token 一致"
//!
//! §18.5 已定谳:TP2 与单卡的分歧 = **AR 固有数值语义**(部分和各自 round f16
//! 再相加 vs 整段 f32 累加一次落 f16),1 ULP 经 24 层 GDN 递推混沌放大。
//! 业内(§19.1):vLLM 的 TP CI 就是这个判据(采样 token 不同 ⇒ 互为对方
//! top-N ⇒ 打 warning 并停止比对);SGLang 明确只承诺 **TP2 内部**的归约序固定,
//! **不**承诺 ≡ 单卡。
//!
//! # 运行(两张空闲卡 = **CUDA ordinal**;未设 = skip)
//!
//! ```bash
//! OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_0_8B_DIR=<…/Qwen3.5-0.8B> \
//!   cargo test -p owl-engine --test tp_g8_gate -- --nocapture
//! ```
//! `OWL_G8B_TOPN`(默认 5)= 判据的 N;`OWL_G8_BISECT=1` = 附带逐层 checksum 对拍诊断。

use std::path::PathBuf;

use owl_engine::{Engine, EngineConfig, EngineKnobs, RunningEngine, TpEngine};
use owl_iface::contract::ModelError;
use owl_models::formats::qwen35::qwen3_5_0_8b;
use owl_models::formats::spec::SpecSource;
use owl_models::model::ModelSpec;

const PROMPT: &str = "用一句话介绍长城。";
const MAX_NEW: usize = 16;

fn dev(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn dir_of() -> PathBuf {
    std::env::var("OWL_0_8B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../models/Qwen/Qwen3.5-0.8B"))
}

fn topn_of() -> usize {
    std::env::var("OWL_G8B_TOPN").ok().and_then(|s| s.parse().ok()).unwrap_or(5)
}

/// 同一份档(两臂必须逐字段同参 —— 否则比的是配置不是 TP):greedy +
/// 关前缀缓存(单请求档;快照仍开 ⇒ 快照镜像也被走到)
fn knobs() -> EngineKnobs {
    let mut k = EngineKnobs::default();
    k.sampler_enabled = false; // 设备 argmax(逐 token 确定性)
    k.prefix_cache = false;
    k.pf_bisect = std::env::var("OWL_G8_BISECT").is_ok();
    k
}

/// 层 checksum 取数(boot 前缀剥掉 ⇒ 两臂可比)
fn pf_map(prefix: &str) -> std::collections::BTreeMap<String, u64> {
    let mut m = std::collections::BTreeMap::new();
    for r in owl_shared::metrics::query_metrics(
        &owl_shared::metrics::MetricsFilter::new().tag_prefix(prefix),
    ) {
        let key = r.tag.split_once('.').map(|(_, k)| k.to_string()).unwrap_or_else(|| r.tag.clone());
        m.insert(key, r.counter);
    }
    m
}

fn cfg_of(ordinal: usize) -> EngineConfig {
    EngineConfig { device_ordinal: ordinal, max_seq_tokens: 256, prefill_chunk: 64, knobs: knobs() }
}

/// 一步一采样：一个 turn 的**逐步**产物
struct Steps {
    /// 生成段 token(会话账 `prompt_len..`)
    gen: Vec<u32>,
    /// 逐步 logits(**每个采样步一条**;与 `gen` 同长 —— 判据的对齐前提)
    logits: Vec<Vec<f32>>,
}

/// 跑一个 turn,并在**每个采样步**收一条 logits(采样步 = `Token` 事件;
/// prefill 中间块与终局 `Completed` 都不产 token ⇒ 不记 —— 后者若记会多一条,
/// 其 logits 槽是最后一次采样的旧值)。
async fn run_turn_steps(
    running: &mut RunningEngine<owl_engine::GpuFace>,
    sid: u64,
    prompt: &str,
    max_new: usize,
) -> Result<Steps, ModelError> {
    let (sid, turn) = running.submit_session_ex(Some(sid), prompt, max_new, true)?;
    let prompt_len = running
        .turn_prompt_len(turn)
        .ok_or_else(|| ModelError::Msg(format!("turn {turn} 提交后取不到 prompt 长度")))?;
    let mut logits: Vec<Vec<f32>> = Vec::new();
    let mut text = String::new();
    loop {
        let ev = running.pump().await?;
        let is_sample_step = match &ev {
            // 只有 `Token` 是采样步（最后一个 token 也由 `Token` 携带；
            // 其后的 `Completed` 是终局判定,不产 token —— 算进去会多一条)
            owl_engine::TurnEvent::Token { turn: t, .. } => *t == turn,
            _ => false,
        };
        if is_sample_step {
            match running.read_output_f32("logits").await? {
                Some(l) => logits.push(l),
                None => {
                    return Err(ModelError::Msg(format!(
                        "turn {turn}:采样步读不到 logits 槽(逐步对拍前提不成立)"
                    )))
                }
            }
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
        .ok_or_else(|| ModelError::Msg(format!("会话 {sid} 账不在")))?;
    let gen = toks[prompt_len..].to_vec();
    assert!(!text.trim().is_empty(), "turn 产物为空(门不该在空输出上过)");
    if logits.len() != gen.len() {
        return Err(ModelError::Msg(format!(
            "逐步 logits 条数 {} ≠ 生成 token 数 {}(采样步对齐失败;判据前提见 run_turn_steps 注)",
            logits.len(),
            gen.len()
        )));
    }
    Ok(Steps { gen, logits })
}

/// top-N token id(降序;N ≤ 词表)
fn topn(v: &[f32], n: usize) -> Vec<u32> {
    let n = n.min(v.len());
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.select_nth_unstable_by(n, |&a, &b| v[b].partial_cmp(&v[a]).unwrap());
    let mut out: Vec<(usize, f32)> = idx[..n].iter().map(|&i| (i, v[i])).collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    out.into_iter().map(|(i, _)| i as u32).collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

#[test]
fn tp_g8_gate() {
    let (Some(oa), Some(ob)) = (dev("OWL_TP_TEST_DEV_A"), dev("OWL_TP_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_TP_TEST_DEV_A / OWL_TP_TEST_DEV_B(两张空闲卡的 CUDA ordinal)");
        return;
    };
    assert_ne!(oa, ob, "A/B 必须是两张不同卡");
    let ref_dev = dev("OWL_TP_REF_DEV").unwrap_or(oa);
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        eprintln!("skip: 0.8B 检查点不在 {dir:?}(设 OWL_0_8B_DIR)");
        return;
    }
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(body(ref_dev, oa, ob, dir))
        })
        .expect("起测试线程")
        .join()
        .expect("G8-b 门失败");
}

async fn body(ref_dev: usize, oa: usize, ob: usize, dir: PathBuf) {
    let n_top = topn_of();
    // ★ **词表切档（P1.1）**：本模型（Qwen3.5-0.8B）`tie_word_embeddings = true`
    // ⇒ lm_head 复用 embed 全量表 ⇒ 词表切**无据可依** ⇒ 装载必须**响亮拒**。
    // 本档只验证这条拒（正判据在 27B 门：那里是 untied 检查点）。
    if matches!(std::env::var("OWL_TP_VOCAB_SPLIT").as_deref(), Ok("1") | Ok("true")) {
        let mut e = Engine::new(cfg_of(oa)).expect("engine(tied 拒判据)");
        let r = e
            .loader()
            .slice(0, 2)
            .vocab_split(true)
            .load_qwen35_0_8b(&dir)
            .await;
        match r {
            Ok(_) => panic!(
                "tied 模型（tie_word_embeddings=true）+ 词表切应被**响亮拒** —— \
                 实测装载成功 = 判据失守（词表切静默不生效）"
            ),
            Err(e) => {
                eprintln!("[g8b] 词表切 × tied：装载响亮拒 ✓（{e}）");
                return;
            }
        }
    }
    // ── ① TP1 参考臂:单卡全量(先跑先卸,腾出显存给双 rank)──────────
    let (tp1, pf1) = {
        let mut e = Engine::new(cfg_of(ref_dev)).expect("engine(TP1)");
        let loaded = e.loader().load_qwen35_0_8b(&dir).await.expect("TP1 装载");
        let mut r = e.run(loaded).await.expect("TP1 装配");
        let s = run_turn_steps(&mut r, 1, PROMPT, MAX_NEW).await.expect("TP1 turn");
        let pf = pf_map("b1.pf.");
        let last = r.read_output_f32("logits").await.expect("TP1 logits").expect("logits 槽");
        eprintln!(
            "[g8b] TP1(单卡 ordinal {ref_dev}):{} tok;逐步 logits {} 条;末步 {} 项",
            s.gen.len(),
            s.logits.len(),
            last.len()
        );
        (s, pf)
    };
    let Steps { gen: tp1_gen, logits: tp1_logits } = tp1;

    // ── ② TP2:两 rank 分片装载 + 装配 + 执行 ───────────────────────
    let mut e0 = Engine::new(cfg_of(oa)).expect("engine(TP2 rank0)");
    let l0 = e0.loader().slice(0, 2).load_qwen35_0_8b(&dir).await.expect("rank0 分片装载");
    let mut e1 = Engine::new_with_ledger(cfg_of(ob), e0.ledger_cell(), 1).expect("engine(TP2 rank1)");
    let l1 = e1.loader().slice(1, 2).load_qwen35_0_8b(&dir).await.expect("rank1 分片装载");
    // ★ **push 档开关**(2026-10-18;§36.6/§36.7)—— 测试用例可放 env
    // (owl 纪律:行为开关走配置,测试可例外)。
    let push = matches!(std::env::var("OWL_TP_PUSH").as_deref(), Ok("1") | Ok("true"));
    // ★ **fp8 载荷档**(§47 第二步;测试用例可放 env —— owl 纪律:行为开关走配置,测试例外)
    let ar_f8 = matches!(std::env::var("OWL_TP_AR_F8").as_deref(), Ok("1") | Ok("true"));
    // ★ **v1 两核融合档**(§11.2;测试例外同上)
    let ar_fused = matches!(std::env::var("OWL_TP_AR_FUSED").as_deref(), Ok("1") | Ok("true"));
    let tp = TpEngine::assemble_with(
        vec![e0, e1],
        vec![l0, l1],
        owl_engine::TpOpts { push, ar_f8, ar_fused, one_stage: false },
    )
        .await
        .expect("TP2 装配");
    let mut r = tp.run().expect("TP2 进执行态");
    assert!(r.has_peer(), "TP2 执行态必须带 peer 束");
    // ★ **路径存在性判据**(纪律⑮/§35.5;2026-10-18):门全绿证明不了新路径被执行。
    // ⚠️ metrics store 是**进程级 OnceLock** ⇒ **必须在 turn 之前 init**
    // (turn 期间 counter_add 才有效;事后 init 只能看到 0 —— 本刀踩过)。
    let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
    let cnt0 = |t: &str| -> u64 {
        owl_shared::metrics::query_metrics(&owl_shared::metrics::MetricsFilter::new().tag(t))
            .iter()
            .map(|x| x.counter)
            .sum()
    };
    // **基线快照**(2026-10-20):metrics store 是**进程级 OnceLock**,而本目标还有
    // 另一个测试(`body_probe`,走 `assemble` 默认档 = 两核)⇒ 若它先跑,计数已经
    // 不是 0 ⇒ "融合档下 v1 必须为 0"的绝对判据会**假红**(实测 v1=2016 全是它留的)。
    // 判据改成**本臂增量**(turn 前后的差),与测试顺序解耦。
    let base = ["tp.ar.push", "tp.ar.v1", "tp.ar.fused", "tp.ar.1stage"].map(|t| cnt0(t));
    let Steps { gen: tp2_gen, logits: tp2_logits } =
        run_turn_steps(&mut r, 1, PROMPT, MAX_NEW).await.expect("TP2 turn");
    let cnt = |t: &str| -> u64 {
        let now: u64 = owl_shared::metrics::query_metrics(
            &owl_shared::metrics::MetricsFilter::new().tag(t),
        )
        .iter()
        .map(|x| x.counter)
        .sum();
        let i = match t {
            "tp.ar.push" => 0,
            "tp.ar.v1" => 1,
            "tp.ar.fused" => 2,
            _ => 3,
        };
        now.saturating_sub(base[i])
    };
    let (n_push, n_v1) = (cnt("tp.ar.push"), cnt("tp.ar.v1"));
    let n_fused = cnt("tp.ar.fused");
    // ⚠️ 1-stage 档的路径判据在 `tp_perf_split_probe`(那里断言 `tp.ar.1stage > 0 ∧ v1 = 0`);
    // 本条只管 push/v1 两档 —— **1-stage 开时 v1 本就该是 0**,别在这里拦(2026-10-19)
    let one_stage_on = matches!(std::env::var("OWL_TP_1STAGE").as_deref(), Ok("1") | Ok("true"));
    eprintln!("[g8b] 路径判据:push={n_push} · v1={n_v1} · fused={n_fused}(OWL_TP_PUSH={push} · 融合={ar_fused} · 1-stage={one_stage_on})");
    if one_stage_on {
        /* 走 tp_perf_split_probe 的判据 */
    } else if ar_fused {
        // **融合档的路径判据**(§11.2;2026-10-19):必须走到 fused,且 v1/push 都为 0
        // (否则"开了融合档但没走"= 静默不生效,与 1-stage §60.2 同族坑)
        assert!(n_fused > 0, "开了融合档但一次都没走到(tp.ar.fused = 0)");
        assert_eq!(n_v1, 0, "融合档下不应走 v1 两核(tp.ar.v1 = {n_v1})");
        assert_eq!(n_push, 0, "融合档下不应走 push 两核(tp.ar.push = {n_push})");
    } else if push {
        assert!(n_push > 0, "开了 push 但一次都没走到(tp.ar.push = 0)");
        assert_eq!(n_v1, 0, "开了 push 却仍在走 v1 绑定(tp.ar.v1 = {n_v1})");
    } else {
        assert_eq!(n_push, 0, "没开 push 却走了 push 绑定");
        assert!(n_v1 > 0, "v1 档应有 tp.ar.v1 > 0");
    }
    eprintln!("[g8b] TP2(双 rank {oa}/{ob}):{} tok;逐步 logits {} 条", tp2_gen.len(), tp2_logits.len());

    // ── ③ L1(接线):两 rank **同一个采样步**的 logits 逐位相同 ─────
    // 逐位检查放末步即可(接线错会在任何一步都错;逐步全查 = 16×1MB 读数不划算)
    let l0_last = r.read_output_f32("logits").await.expect("rank0 logits").expect("logits 槽");
    let l1_last = r.read_peer_output_f32("logits").await.expect("rank1 logits").expect("logits 槽");
    assert_eq!(l0_last.len(), l1_last.len(), "两 rank logits 长度不一致");
    let nbit = l0_last.iter().zip(&l1_last).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    let mx01 = max_abs(&l0_last, &l1_last);
    eprintln!("[g8b] L1 接线:rank0 vs rank1 末步 logits:{nbit} 项不同, max|Δ| = {mx01:.6}");
    assert_eq!(nbit, 0, "两 rank logits 应逐位相同(AR 接线/协议错?)");
    r.shutdown_tp().await.expect("TP2 收尾");

    // ── ④ L2 + L3:逐位置比较(vLLM check_logprobs_close 形态)──────
    assert_eq!(
        tp1_gen.len(),
        tp2_gen.len(),
        "生成长度不一致(ignore_eos 定长档应恒等)\nTP1={:?}\nTP2={:?}",
        tp1_gen,
        tp2_gen
    );
    let steps = tp1_gen.len();
    let mut per_step_max: Vec<f32> = Vec::with_capacity(steps);
    for i in 0..steps {
        per_step_max.push(max_abs(&tp1_logits[i], &tp2_logits[i]));
    }
    let worst = per_step_max.iter().cloned().fold(0f32, f32::max);
    let mean = per_step_max.iter().sum::<f32>() / steps as f32;
    eprintln!("[g8b] L3 数值面:逐步 max|Δ| —— 首步 {:.6}, 末步 {:.6}, 全程最大 {:.6}, 均值 {:.6}",
        per_step_max[0], per_step_max[steps - 1], worst, mean);
    eprintln!("[g8b] 逐步 max|Δ| 全序列 = {:?}", per_step_max.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());

    let mut divergences = 0usize;
    let mut first_div: Option<usize> = None;
    // 全序列的 token 不同步(仅参考:分歧后上下文已分岔 ⇒ 不参与判据)
    let mut all_diff: Vec<usize> = Vec::new();
    let mut stop = false;
    // 对齐自证(前 3 步):读回的 logits 是否就是采样所用那份?
    // greedy 档下 `argmax(logits[i])` 必须 == `gen[i]`(否则是读数错位/非 greedy)
    for i in 0..steps.min(3) {
        let am1 = topn(&tp1_logits[i], 1)[0];
        let am2 = topn(&tp2_logits[i], 1)[0];
        let r1 = topn(&tp1_logits[i], 64).iter().position(|&x| x == tp1_gen[i]);
        let nxt = tp1_gen.get(i + 1).copied().unwrap_or(u32::MAX);
        eprintln!(
            "[diag] step {i}: TP1 gen={} argmax(logits)={} rank64(gen)={r1:?} next_gen={nxt};TP2 gen={} argmax={}",
            tp1_gen[i], am1, tp2_gen[i], am2
        );
    }
    for i in 0..steps {
        if tp1_gen[i] == tp2_gen[i] {
            continue;
        }
        all_diff.push(i);
        if stop {
            continue;
        }
        // 分歧即止:只在**首个分歧**处判"互为 top-N"
        stop = true;
        divergences += 1;
        first_div = Some(i);
        let (t1, t2) = (topn(&tp1_logits[i], n_top), topn(&tp2_logits[i], n_top));
        let a = t1.contains(&tp2_gen[i]); // TP2 选的 token 在 TP1 的 top-N 里?
        let b = t2.contains(&tp1_gen[i]); // TP1 选的 token 在 TP2 的 top-N 里?
        eprintln!(
            "[g8b] 首分歧步 {i}:TP1={}(rank {:?}) / TP2={}(rank {:?})",
            tp1_gen[i],
            t1.iter().position(|&x| x == tp1_gen[i]),
            tp2_gen[i],
            t2.iter().position(|&x| x == tp2_gen[i]),
        );
        eprintln!("[g8b] 互含判定(N={n_top}):TP1.topN ∋ TP2 选={a} / TP2.topN ∋ TP1 选={b}");
        eprintln!("[g8b] TP1 topN = {t1:?}");
        eprintln!("[g8b] TP2 topN = {t2:?}");
        assert!(
            a && b,
            "G8-b 违约:第 {i} 个 token 分歧且**不互为 top-{n_top}**\n  TP1 选 {} rank {:?};TP2 选 {} rank {:?}\n\
             (判据 = vLLM check_logprobs_close;§19.2 G8-b。若 N={n_top} 太小可 OWL_G8B_TOPN 加大后复核)",
            tp1_gen[i],
            t1.iter().position(|&x| x == tp1_gen[i]),
            tp2_gen[i],
            t2.iter().position(|&x| x == tp2_gen[i]),
        );
    }
    // 分歧即止 ⇒ 判据窗口 = [0, 首分歧];窗口外不可比(上下文已分岔)
    let cmp_len = first_div.map(|i| i + 1).unwrap_or(steps);
    let rate = divergences as f64 / cmp_len as f64;
    eprintln!(
        "[g8b] L3 记账:判据窗口 {cmp_len}/{steps} 步(分歧即止);窗口内分歧 {divergences} 步;窗口内分歧率 {:.1}%;全序列 token 不同 {}/{} 步 {:?}(参考值,不参与判据)",
        rate * 100.0,
        all_diff.len(),
        steps,
        all_diff
    );

    // ── 附:逐层 checksum 诊断(旧二分法;默认关)──────────────────
    if std::env::var("OWL_G8_BISECT").is_ok() {
        let pf2 = pf_map("b2.pf.");
        let mut diffs = 0;
        let mut first: Option<String> = None;
        for (k, v1) in &pf1 {
            match pf2.get(k) {
                Some(v2) if v2 == v1 => {}
                other => {
                    diffs += 1;
                    if first.is_none() {
                        first = Some(format!("{k}: TP1={v1} TP2={other:?}"));
                    }
                }
            }
        }
        eprintln!("[bisect] 层 checksum 差异 = {diffs}/{};首个 = {first:?}", pf1.len());
    }

    // 附:全量 spec 自证(world = 2 整除前提;CPU 侧读 config,零 GPU)
    let (full, _) = ModelSpec::from_ckpt(&dir, SpecSource::family(&qwen3_5_0_8b())).expect("全量 spec");
    assert_eq!(full.inter % 2, 0, "inter 必须偶(world = 2 整除硬门前提)");
    match first_div {
        None => eprintln!(
            "[g8b] 全绿(逐 token 一致):{} tok(单卡 ordinal {ref_dev} ≡ 双 rank {oa}/{ob};档 = greedy/无前缀缓存)",
            tp1_gen.len()
        ),
        Some(i) => eprintln!(
            "[g8b] 全绿(G8-b:首分歧步 {i} 处互为 top-{n_top};分歧率 {:.1}%;全程 max|Δ| {:.6})—— 与 vLLM TP CI 同形(§19.1)",
            rate * 100.0,
            worst
        ),
    }
}

// ============================================================================
// ★ 分步计时探针(2026-10-18;**非门** —— 无判据,只 dump 分布与总墙钟)
// ============================================================================
//
// 为什么需要它:1-stage(§35)结构更对却性能中性 ⇒ 必须看清**时间花在哪**。
// 三个变量要分开:
//   · `OWL_GPU_PROF=1` ⇒ 每次发射后同步 COMPUTE 流 ⇒ `srv.op.<核名>` 里有**真设备时间**,
//     但**总时间失真**(同步本身很贵)—— 这就是"日志自己的开销"那一档;
//   · `no_graph = true`(eager)⇒ 逐核可归因(图模式整步只发一次,归因不到核);
//   · 两者都关 ⇒ 只剩**总时间**(由 `tools/tp-ab-std.sh` 给的是生产版)。
//
// 跑法(本例 = 分布档;把 `OWL_GPU_PROF` 去掉 = 总时间档):
// ```bash
// OWL_GPU_PROF=1 OWL_PERF_TAG=v1prof OWL_TP_1STAGE=0 \
//   OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_0_8B_DIR=<…0.8B> \
//   cargo test -p owl-engine --test tp_g8_gate tp_perf_split_probe -- --nocapture
// ```
// `OWL_PERF_REPS`(缺省 40)控制 prompt 长度(每遍约 20 字 ≈ 13 tok)。

/// **只推进、不读 logits**的 turn(`run_turn_steps` 要逐步 logits,那是判据面的需求;
/// 计时探针不需要它 —— 而且 dtoh 本身会进总时间,污染"日志自己的开销"那一档)。
async fn run_turn_plain<D: owl_iface::contract::DeviceClient>(
    r: &mut RunningEngine<D>,
    sid: u64,
    prompt: &str,
    max_new: usize,
) -> Result<usize, ModelError> {
    let (_s, turn) = r.submit_session_ex(Some(sid), prompt, max_new, true)?;
    let mut n = 0usize;
    loop {
        match r.pump().await? {
            owl_engine::TurnEvent::Token { turn: t, .. } if t == turn => n += 1,
            owl_engine::TurnEvent::Idle => break,
            owl_engine::TurnEvent::Failed { turn: t, err } => {
                return Err(ModelError::Msg(format!("t{t} 失败:{err}")))
            }
            _ => {}
        }
    }
    Ok(n)
}

fn cfg_eager(ordinal: usize) -> EngineConfig {
    let mut c = cfg_of(ordinal);
    c.knobs.no_graph = true; // eager:逐核可归因
    c.max_seq_tokens = 1024; // 探针用长 prompt(缺省 256 装不下)
    c
}

#[test]
fn tp_perf_split_probe() {
    let (Some(oa), Some(ob)) = (dev("OWL_TP_TEST_DEV_A"), dev("OWL_TP_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_TP_TEST_DEV_A / OWL_TP_TEST_DEV_B");
        return;
    };
    let dir = dir_of();
    if !dir.join("config.json").exists() {
        eprintln!("skip: 0.8B 检查点不在 {dir:?}(设 OWL_0_8B_DIR)");
        return;
    }
    let reps: usize = std::env::var("OWL_PERF_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(40);
    let tag = std::env::var("OWL_PERF_TAG").unwrap_or_else(|_| "perf".to_string());
    let prompt: String = "在深海下面有一座城市它的名字没有人记得。".repeat(reps);
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(body_probe(oa, ob, dir, prompt, tag))
        })
        .expect("起测试线程")
        .join()
        .expect("perf 探针失败");
}

async fn body_probe(oa: usize, ob: usize, dir: PathBuf, prompt: String, tag: String) {
    // metrics store 是**进程级 OnceLock**:生产由 server 起时 init,测试里得自己来
    // (不 init ⇒ `dump` 静默 Ok 且不落文件 —— 2026-10-18 踩过)
    let _ = owl_shared::metrics::init_metrics(owl_shared::metrics::MetricsStore::new());
    let mut e0 = Engine::new(cfg_eager(oa)).expect("engine rank0");
    let l0 = e0.loader().slice(0, 2).load_qwen35_0_8b(&dir).await.expect("rank0 装载");
    let mut e1 = Engine::new_with_ledger(cfg_eager(ob), e0.ledger_cell(), 1).expect("engine rank1");
    let l1 = e1.loader().slice(1, 2).load_qwen35_0_8b(&dir).await.expect("rank1 装载");
    let tp = TpEngine::assemble(vec![e0, e1], vec![l0, l1]).await.expect("装配");
    let mut r = tp.run().expect("进执行态");
    // 预热一次(装载后的首跑含 lazy 初始化;不进计时)
    let _ = run_turn_plain(&mut r, 1, &prompt, 1).await.expect("预热 turn");
    // ── ★ **串行 vs 并发**(2026-10-18;TBO 的最小验证)────────────────
    // sglang 的 TBO 本质 = "两个独立微批的 forward 交错" ⇒ 它能不能拿到重叠,
    // 取决于**调度层是否让两个 forward 同时在飞**。本档量这件事:
    //   serial     : turn → turn(两次,单会话各一次)
    //   concurrent : 同时挂两个会话,一起 pump
    // 判据:**concurrent 明显快于 serial ⇒ 调度层已是并发的**,重叠有地基;
    //       ≈ 相等 ⇒ 引擎串行处理 turn,交错要在引擎里做(本档就是立案证据)。
    let mode = std::env::var("OWL_PERF_MODE").unwrap_or_else(|_| "serial".to_string());
    let t0 = std::time::Instant::now();
    let ngen = if mode == "concurrent" {
        let (_s1, t1) = r.submit_session_ex(Some(11), &prompt, 1, true).expect("提交 A");
        let (_s2, t2) = r.submit_session_ex(Some(12), &prompt, 1, true).expect("提交 B");
        let mut n = 0usize;
        loop {
            match r.pump().await.expect("pump") {
                owl_engine::TurnEvent::Token { .. } => n += 1,
                owl_engine::TurnEvent::Idle => break,
                owl_engine::TurnEvent::Failed { turn, err } => {
                    panic!("t{turn} 失败:{err}")
                }
                _ => {}
            }
        }
        let _ = (t1, t2);
        n
    } else {
        let a = run_turn_plain(&mut r, 21, &prompt, 1).await.expect("turn A");
        let b = run_turn_plain(&mut r, 22, &prompt, 1).await.expect("turn B");
        a + b
    };
    let total = t0.elapsed();
    eprintln!("[perf-split:{tag}] mode={mode}");
    eprintln!(
        "[perf-split:{tag}] prompt {} 字 / gen {} tok ⇒ **总墙钟 {:.3}s** | 1-stage={} · gpu_prof={} · eager=on",
        prompt.chars().count(),
        ngen,
        total.as_secs_f64(),
        std::env::var("OWL_TP_1STAGE").unwrap_or_else(|_| "(缺省 v1)".to_string()),
        std::env::var("OWL_GPU_PROF").is_ok(),
    );
    // ★ **路径存在性判据**(2026-10-18):1-stage 开时,融合核必须真被走到
    // —— 门全绿 ≠ 新路径被执行(本轮实测:融合核 0 次而门全绿)。
    let cnt = |t: &str| -> u64 {
        owl_shared::metrics::query_metrics(&owl_shared::metrics::MetricsFilter::new().tag(t))
            .iter()
            .map(|r| r.counter)
            .sum()
    };
    let one_stage = matches!(std::env::var("OWL_TP_1STAGE").as_deref(), Ok("1") | Ok("true"));
    let (n_new, n_old) = (cnt("tp.ar.1stage"), cnt("tp.ar.v1"));
    eprintln!("[perf-split:{tag}] 路径判据:fusion={n_new} · v1={n_old}(开 1-stage={one_stage})");
    if one_stage {
        assert!(n_new > 0, "开了 1-stage 但融合核一次都没被走到(tp.ar.1stage = 0)");
        assert_eq!(n_old, 0, "开了 1-stage 却仍在走 v1 两段(tp.ar.v1 = {n_old})");
    } else {
        assert_eq!(n_new, 0, "没开 1-stage 却走了融合核");
        assert!(n_old > 0, "v1 档应有 tp.ar.v1 > 0");
    }
    let path = format!("/tmp/tp-perf-{tag}.txt");
    match owl_shared::metrics::dump(&owl_shared::metrics::MetricsFilter::new(), &path) {
        Ok(()) => eprintln!("[perf-split:{tag}] metrics → {path}"),
        Err(e) => eprintln!("[perf-split:{tag}] metrics dump 失败: {e}"),
    }
    r.shutdown_tp().await.expect("收尾");
}
