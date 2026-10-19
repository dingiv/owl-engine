//! **G8-a 配置内不变性门**(TP2;SGLang 判据 —— "Unique samples: 1")。
//!
//! 设计出处:`docs/arch/TP2-施工设计.md` §19.2(门表)/ §19.3(纪律)/ §20.2 T1。
//!
//! # 判据
//!
//! **同 world(world = 2)、同 prompt、同 greedy 档**,只变"形状":
//! - ① 队列形状:目标 turn 前压 3 个干扰 turn(队列 FIFO;GDN 格/块链/事件序都被扰动);
//! - ② prefill 块长:`prefill_chunk` = 32 / 64 / 128(装配档);
//! - ③ 前缀缓存:`prefix_cache` 开/关(装配档);
//! - ④ 会话状态:同 sid 二次提交(热账 + 跨会话块链复用路径);
//! - ⑤ 跨会话同 prompt(跨会话内容复用)。
//!
//! 所有 arm 的**生成段 token 序列逐位唯一** ⇒ 通过。任一 arm 与基线不同 ⇒ **响亮失败**
//! (打印首分歧位置与两臂 token)。
//!
//! # 为什么 owl v1 天生该过(卖点;§19.2)
//!
//! v1 的 AR = 固定核 + 相对游标 + **单 chunk**(载荷 > slot ⇒ 响亮拒,不静默分块)
//! ⇒ **归约序与 shape 无关** —— 正是 SGLang 给 TP=2 的
//! "consistent floating-point addition order"。
//! §S5 一旦放开多块/流水,**本门必须在 S5 之后仍绿**(§19.3 纪律 ①)。
//!
//! 与 [`tp_g8_gate`](G8-b:vs 单卡功能等价)的分工:本门**不碰单卡**,只问
//! "同配置内是否唯一"——混沌放大(§18.5)对本门无影响(不参与跨配置比较)。
//!
//! # 运行(两张空闲卡 = **CUDA ordinal**;未设 = skip)
//!
//! ```bash
//! OWL_TP_TEST_DEV_A=2 OWL_TP_TEST_DEV_B=3 OWL_0_8B_DIR=<…/Qwen3.5-0.8B> \
//!   cargo test -p owl-engine --test tp_determinism_gate -- --nocapture
//! ```
//! 子集跑(调试):`OWL_G8A_GROUPS=chunk64-pc-on`(逗号分隔的装配组标签)。

use std::path::PathBuf;

use owl_engine::{Engine, EngineConfig, EngineKnobs, RunningEngine, TpEngine};
use owl_iface::contract::ModelError;

/// 目标 prompt(所有 arm 逐字相同)与干扰 prompt(只改队列形状)
const PROMPT: &str = "长城是中国著名的古代建筑,横亘于北方的群山之间,历经两千余年。";
const DISTRACT: [&str; 3] = [
    "黄河是中国的第二长河,流经九个省区。",
    "长江发源于青藏高原,最终注入东海。",
    "京杭大运河是世界上最长的人工运河。",
];
/// 目标生成长度(ignore_eos = 定长;逐位比较的前提)
const GEN: usize = 16;
/// 干扰生成长度(短;只求"扰动队列/池/状态")
const DISTRACT_GEN: usize = 3;
/// 单 turn token 预算(KV 直排槽位;prompt + 生成 ≪ 此值)
const MAX_SEQ: usize = 512;

fn dev(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn dir_of() -> PathBuf {
    std::env::var("OWL_0_8B_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("../../models/Qwen/Qwen3.5-0.8B"))
}

/// 两臂必须逐字段同参(差异只允许在"形状"轴上)。
/// `raw_completion` = 绕过 chat 模板 ⇒ prompt token 账 = 原样 encode(门要
/// 直接构造 token 级前缀,greedy + 关采样同 G8-b)。
fn knobs(prefix_cache: bool) -> EngineKnobs {
    let mut k = EngineKnobs::default();
    k.sampler_enabled = false; // 设备 argmax(逐 token 确定性)
    k.prefix_cache = prefix_cache;
    k.raw_completion = true;
    k
}

fn cfg_of(ordinal: usize, chunk: usize, prefix_cache: bool) -> EngineConfig {
    EngineConfig {
        device_ordinal: ordinal,
        max_seq_tokens: MAX_SEQ,
        prefill_chunk: chunk,
        knobs: knobs(prefix_cache),
    }
}

/// 一个形状 arm 的产物
struct Arm {
    label: String,
    /// 生成段 token(会话账 `prompt_len..`)
    gen: Vec<u32>,
    /// 生成文本(回填构造用 —— 前缀复用的 arm 需要它)
    text: String,
    /// 提交期前缀命中量(`turn_cached_len`;guard 命中量 —— 诊断面)
    cached_submit: usize,
    /// 执行期起始命中量(`active_cached_len`;含调度期跨会话复用 —— 诊断面)
    cached_active: Option<usize>,
}

/// 提交一个 turn 并驱动到完结;返回生成段 + 命中量诊断。
async fn run_turn(
    r: &mut RunningEngine<owl_engine::GpuFace>,
    sid: u64,
    prompt: &str,
    max_new: usize,
) -> Result<Arm, ModelError> {
    let (sid, turn) = r.submit_session_ex(Some(sid), prompt, max_new, true)?;
    let prompt_len = r
        .turn_prompt_len(turn)
        .ok_or_else(|| ModelError::Msg(format!("turn {turn} 提交后取不到 prompt 长度")))?;
    let cached_submit = r.turn_cached_len(turn).unwrap_or(0);
    let mut text = String::new();
    let mut cached_active = None;
    loop {
        match r.pump().await? {
            owl_engine::TurnEvent::Idle => break,
            owl_engine::TurnEvent::Completed { turn: t, text: out, .. } if t == turn => {
                text = out;
            }
            owl_engine::TurnEvent::Failed { turn: t, err } => {
                return Err(ModelError::Msg(format!("t{t} 失败:{err}")))
            }
            _ => {}
        }
        // 活跃期内读起始命中量(调度期决策,含跨会话复用;turn 终了即焚)
        if cached_active.is_none() {
            cached_active = r.active_cached_len(turn);
        }
    }
    let toks = r
        .session_token_ids(sid)
        .ok_or_else(|| ModelError::Msg(format!("会话 {sid} 账不在")))?;
    if toks.len() < prompt_len + max_new {
        return Err(ModelError::Msg(format!(
            "会话 {sid} 账短了:{} < {prompt_len} + {max_new}(turn 未跑满?)",
            toks.len()
        )));
    }
    let gen = toks[prompt_len..prompt_len + max_new].to_vec();
    eprintln!(
        "[g8a] sid={sid} turn={turn}:prompt={prompt_len} gen={};cached(submit={cached_submit} active={cached_active:?});text={:?}",
        gen.len(),
        text.chars().take(24).collect::<String>()
    );
    Ok(Arm { label: String::new(), gen, text, cached_submit, cached_active })
}

/// 两 rank 分片装载 + 装配(一次 = 一个装配档)
async fn assemble(
    oa: usize,
    ob: usize,
    dir: &PathBuf,
    chunk: usize,
    prefix_cache: bool,
) -> TpEngine<owl_cuda::GpuClient> {
    let mut e0 = Engine::new(cfg_of(oa, chunk, prefix_cache)).expect("engine(rank0)");
    let l0 = e0
        .loader()
        .slice(0, 2)
        .load_qwen35_0_8b(dir)
        .await
        .expect("rank0 分片装载");
    // rank1 账本/设备序装载前从 rank0 接过(TP 合流纪律;见 `Engine::new_with_ledger`)
    let mut e1 =
        Engine::new_with_ledger(cfg_of(ob, chunk, prefix_cache), e0.ledger_cell(), 1)
            .expect("engine(rank1)");
    let l1 = e1
        .loader()
        .slice(1, 2)
        .load_qwen35_0_8b(dir)
        .await
        .expect("rank1 分片装载");
    TpEngine::assemble(vec![e0, e1], vec![l0, l1])
        .await
        .expect("TP 装配")
}

/// 逐位硬判(打印首分歧上下文)
fn assert_same(label: &str, got: &[u32], want: &[u32]) {
    assert_eq!(
        got.len(),
        want.len(),
        "G8-a 违约[{label}]:生成长度 {} ≠ 基线 {}(token 账形状就不对)",
        got.len(),
        want.len()
    );
    if got != want {
        let i = got.iter().zip(want).position(|(a, b)| a != b).unwrap_or(0);
        panic!(
            "G8-a 违约[{label}]:第 {i} 个 token 起分歧\n  本臂 = {got:?}\n  基线 = {want:?}\n\
             (同 world 同 prompt 变形状必须逐位唯一;§19.2 G8-a)"
        );
    }
}

/// 回填续写尾(前缀复用的 arm 用它当"增量段")
const TAIL: &str = "\n\n请继续。";

#[test]
fn tp_determinism_gate() {
    let (Some(oa), Some(ob)) = (dev("OWL_TP_TEST_DEV_A"), dev("OWL_TP_TEST_DEV_B")) else {
        eprintln!("skip: 需 OWL_TP_TEST_DEV_A / OWL_TP_TEST_DEV_B(两张空闲卡的 CUDA ordinal)");
        return;
    };
    assert_ne!(oa, ob, "A/B 必须是两张不同卡");
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
                .block_on(body(oa, ob, dir))
        })
        .expect("起测试线程")
        .join()
        .expect("G8-a 门失败");
}

async fn body(oa: usize, ob: usize, dir: PathBuf) {
    // 装配档矩阵(轴 ②③):(标签, prefill_chunk, prefix_cache)
    let all_groups: [(&str, usize, bool); 4] = [
        ("chunk64-pc-on", 64, true),
        ("chunk32-pc-on", 32, true),
        ("chunk128-pc-on", 128, true),
        ("chunk64-pc-off", 64, false),
    ];
    let only = std::env::var("OWL_G8A_GROUPS").ok();
    let groups: Vec<(&str, usize, bool)> = all_groups
        .into_iter()
        .filter(|(label, _, _)| match &only {
            Some(s) => s.split(',').any(|x| x.trim() == *label),
            None => true,
        })
        .collect();
    assert!(!groups.is_empty(), "OWL_G8A_GROUPS 过滤后没有装配档:去检查标签拼写");

    let mut baseline: Option<(String, Vec<u32>)> = None;
    let mut total = 0usize;
    for (label, chunk, pc) in groups {
        eprintln!("[g8a] ══ 装配档 {label}(prefill_chunk = {chunk};前缀缓存 = {pc})══");
        let tp = assemble(oa, ob, &dir, chunk, pc).await;
        let mut r = tp.run().expect("TP2 进执行态");
        assert!(r.has_peer(), "TP2 执行态必须带 peer 束");

        let mut arms: Vec<Arm> = Vec::new();
        // ① 基线:单请求冷启动(sid = 1)
        arms.push(labeled(&label, "/base", run_turn(&mut r, 1, PROMPT, GEN).await));
        // ② 队列形状:3 个干扰 turn 在前(队列 FIFO),目标 turn 在后(sid = 2)
        for (i, d) in DISTRACT.iter().enumerate() {
            let _ = run_turn(&mut r, 10 + i as u64, d, DISTRACT_GEN)
                .await
                .expect("干扰 turn 失败");
        }
        arms.push(labeled(&label, "/queue3", run_turn(&mut r, 2, PROMPT, GEN).await));
        // ③ 跨会话同 prompt(新 sid = 3;pc 开时走跨会话前缀匹配)
        arms.push(labeled(
            &label,
            "/cross-session",
            run_turn(&mut r, 3, PROMPT, GEN).await,
        ));
        // ④ 会话热账:同 sid 重发同 prompt(sid = 1;账 = PROMPT+GEN ⇒ guard 失配,
        //    但前缀匹配/paged 复用仍可达 —— "有旧账要归还"的形状)
        arms.push(labeled(
            &label,
            "/session-replay",
            run_turn(&mut r, 1, PROMPT, GEN).await,
        ));
        // ⑤ **前缀复用(热)** —— 本门唯一能真走到复用路径的形状:
        //    sid=5 先跑 PROMPT(得 text1),再交**回填历史** = PROMPT+text1+TAIL
        //    ⇒ guard 命中(账 `[0, cached_len)` = PROMPT+gen1 与重发前缀逐 token 等)
        //    ⇒ 只增量 prefill TAIL。
        //
        //    ⚠️ 判据 = **同形重复唯一**(连羦两个 sid 跑同一 shape),**不是**"热==冷":
        //    2026-10-17 实测定谳 —— 热路(GDN 分段推进:旧账 34 tok + 增量 4 tok)
        //    与冷路(一次全量 38 tok)的 GDN 状态**数值不同** ⇒ 生成不同。
        //    这是前缀缓存的**业内同款语义**(vLLM/SGLang 的 prefix caching 都不
        //    承诺与全量位级一致;SGLang 的 prefix 测项也只判"同状态重复唯一")。
        //    冷臂仍跑(作参考值上屏 + 兼任一次全量形状)。
        let hot1 = run_turn(&mut r, 5, PROMPT, GEN).await.expect("hot-a turn1");
        let full = format!("{PROMPT}{}{TAIL}", hot1.text);
        let hot_a = run_turn(&mut r, 5, &full, GEN).await.expect("hot-a turn2(回填)");
        let cold = run_turn(&mut r, 4, &full, GEN).await.expect("cold(全量参考)");
        let hot1b = run_turn(&mut r, 6, PROMPT, GEN).await.expect("hot-b turn1");
        assert_same(&format!("{label}/prefix-reuse turn1"), &hot1b.gen, &hot1.gen);
        let full_b = format!("{PROMPT}{}{TAIL}", hot1b.text);
        let hot_b = run_turn(&mut r, 6, &full_b, GEN).await.expect("hot-b turn2(回填)");
        assert!(
            hot_a.cached_submit > 0 && hot_b.cached_submit > 0,
            "G8-a arm⑤ 空转:回填后 guard 未命中(cached_submit = {}/{})⇒ 本 arm 没测到复用路径。prompt = {full:?}",
            hot_a.cached_submit,
            hot_b.cached_submit
        );
        assert_same(&format!("{label}/prefix-reuse(同形重复)"), &hot_b.gen, &hot_a.gen);
        // turn1 与基线同形(同 prompt)⇒ 必须与基线逐位一致(兼查"先跑别的 turn 后再跑同形")
        assert_same(&format!("{label}/prefix-reuse turn1 vs base"), &hot1.gen, &arms[0].gen);
        eprintln!(
            "[g8a] 复用臂 vs 冷臂(**参考值,不判** —— GDN 分段推进≠一次推进):hot={:?} cold={:?}",
            &hot_a.gen[..4.min(hot_a.gen.len())],
            &cold.gen[..4.min(cold.gen.len())]
        );
        // ⚠️ **不入 arms**:它的产物是**另一个 prompt**(PROMPT+text1+TAIL)的
        // 生成段,只与自己的同形重复比(横向比会拿两个不同上下文比 —— 语义错)。

        r.shutdown_tp().await.expect("TP 执行态收尾");

        // ── 判据:同 world 同 prompt ⇒ 生成段逐位唯一 ──────────────────
        for a in &arms {
            for other in &arms {
                if a.label != other.label {
                    assert_same(&format!("{} ↔ {}", a.label, other.label), &a.gen, &other.gen);
                }
            }
            if let Some((bl, bg)) = &baseline {
                assert_same(&format!("{} vs 基线[{bl}]", a.label), &a.gen, bg);
            }
        }
        for a in &arms {
            owl_shared::log_info!(
                "g8a",
                "{}:gen={:?} cached(submit={} active={:?})",
                a.label,
                a.gen,
                a.cached_submit,
                a.cached_active
            );
        }
        total += arms.len();
        if baseline.is_none() {
            baseline = Some((arms[0].label.clone(), arms[0].gen.clone()));
        }
        let first = &baseline.as_ref().expect("基线已设").1;
        eprintln!(
            "[g8a] {} 档内 + 跨档全绿({} arm;基线 {first:?}…)",
            label,
            arms.len()
        );
    }

    let (bl, bg) = baseline.expect("至少一个装配档");
    eprintln!(
        "[g8a] 全绿:Unique samples: 1({} arm;基线 [{bl}] = {:?}…)—— 同 world 变形状(队列/块长/缓存/会话态)token 唯一",
        total,
        &bg[..bg.len().min(8)]
    );
}

fn labeled(
    group: &str,
    arm: &str,
    r: Result<Arm, ModelError>,
) -> Arm {
    let mut a = r.unwrap_or_else(|e| panic!("{group}{arm} 失败:{e}"));
    a.label = format!("{group}{arm}");
    a
}
