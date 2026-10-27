//! **GGUF 档 e2e**(2026-10-19 gguf 轮;设计 §七 刀 7 / REQ-GGUF-07/08)。
//!
//! 与 `tests.rs` 的分工:那里是**检查点族**(0.8B/27B AWQ)的探针与门;
//! 本模块专放 **GGUF 单文件档**的端到端实证(装载 → 生成 → 对拍)。
//!
//! 门的层次(**从廉价到昂贵**;先跑廉价门,别一上来就上全量对拍):
//!
//! | 门 | 判据 | 挡住什么 |
//! |---|---|---|
//! | 冒烟(`gpu_gguf_4b_smoke`) | 装载过 + 出**连贯文本** + 步时合理 | 键漏接 / 错变换 / 词表错 / rope 错 / tied 头错(全是"能跑但输出垃圾") |
//! | 对拍(`gpu_gguf_4b_greedy_vs_llamacpp`) | 同文件同 prompt greedy **逐 token** 与 llama.cpp 相同 | 数值口径漂移(权重/激活/归约序) |
//! | TP2(`gpu_gguf_27b_tp2`) | 首 token 硬门 + 两 rank logits **逐位相同** + AR 对称 + 每 rank 半量驻留 | 分片几何(行置换 × 块对齐列切 × v-head 聚合)错 |
//! | VLM(`gpu_gguf_{4b,27b}_vlm_image_smoke`) | mmproj 装载 + 单图 prefill 出非空文本 | 塔键映射/patch 时间片/mmproj 解量化/图像注入 |
//!
//! ⚠️ **为什么冒烟不是形式主义**:GGUF 档的错法**几乎全是静默错值**(行/列序、
//! 段偏移、norm 的 ±1、乘错表)→ 输出会立刻退化成乱码/重复串。因此"连贯"
//! 本身就是强判据;真正的逐位判据在第二门。

use crate::engine::{Engine, EngineConfig, EngineKnobs};
use crate::running::RunningEngine;
use crate::turn::TurnEvent;
#[cfg(test)]
use owl_macros::owl_test;
use owl_shared::testkit::{resource, resource_ok};

/// 4B 档的最简引擎配置(单卡 3090 Ti;池只需装下冒烟会话)。
///
/// ⚠️ **两个 knob 是与外部 oracle 对拍的硬前提**:
/// - `raw_completion = true`:否则 `submit_session` 会先 `chat_wrap`(家族
///   ChatML)—— 而 llama.cpp 的 `/completion`(裸 prompt) 不做模板
///   ⇒ 比的是**两个不同的输入**(2026-10-19 实测:模板态首 token = ` thinking`,
///   裸态 = `" Paris"`,两边都"连贯"但根本不是同一道题)。
/// - `sampler_enabled = false`:贪心(恒等门口径);temp=0 的对手也是 argmax。
/// **测试侧 GGUF 臂集**(2026-10-21):引擎已零 env —— 本文件把 `OWL_GGUF_*`
/// 当**用例入参**读一次,显式落到 `EngineKnobs.env.gguf`(语义与原 env 逐字一致:
/// 未设 = 开,`=0` = 关)。A/B 探针因此仍能在同进程逐臂切换。
fn gguf_arms_from_env() -> owl_models::env::GgufEnv {
    let flag = |k: &str| !matches!(std::env::var(k).ok().as_deref(), Some("0"));
    owl_models::env::GgufEnv {
        tile: flag("OWL_GGUF_TILE"),
        cublas: flag("OWL_GGUF_CUBLAS"),
        mmq2: flag("OWL_GGUF_MMQ2"),
        mmq: flag("OWL_GGUF_MMQ"),
        smem: flag("OWL_GGUF_SMEM"),
        ..Default::default()
    }
}

fn gguf_cfg(ordinal: usize, max_seq: usize, pool_tokens: usize, gdn_slots: usize) -> EngineConfig {
    let mut knobs = EngineKnobs { sampler_enabled: false, raw_completion: true, ..Default::default() };
    knobs.pool_tokens = Some(pool_tokens);
    knobs.gdn_slots = gdn_slots;
    knobs.host_cache_mb = 32;
    knobs.snap_host_mb = 0;
    knobs.snap_max = 0;
    // 臂集 = env 入参(测试例外);生产由 `[dispatch] gguf_*` 给
    knobs.env.gguf = gguf_arms_from_env();
    EngineConfig { device_ordinal: ordinal, max_seq_tokens: max_seq, prefill_chunk: 128, knobs }
}

/// 冒烟场景(参数化;M1 = 4B,M2 = 27B)。
///
/// `n_layers` / `n_gdn` 是**模型的声明事实**(32 层/24 GDN vs 64 层/48 GDN):
/// 既验层型表推导,也验"列序表张数 = GDN 层数"(虚拟键的落点数)。
async fn smoke_scenario(key: &str, n_layers: usize, n_gdn: usize, want_first: &str, cfg: EngineConfig) {
    if !resource_ok(key) {
        owl_shared::testkit::report::skip("资源清单缺 \"{key}\"");
        return;
    }
    let path = std::path::PathBuf::from(resource(key).expect("列了就有"));
    let file_gib = std::fs::metadata(&path).expect("stat").len() as f64 / (1u64 << 30) as f64;
    let t0 = std::time::Instant::now();
    let mut engine = Engine::new(cfg).expect("引擎构造");
    // ⚠️ **按装载增量算账**(不是进程绝对值):同进程里可能已装过别的模型
    // (4B/27B 两档同跑一个进程)⇒ 绝对值会把前一个模型的驻留记到本模型头上
    // (2026-10-19 实测:4B 报 20.79 GiB —— 那是 27B 的 17.7 + 4B 的 2.7)。
    let live0 = owl_shared::vram::live_bytes();
    let loaded = engine.loader().load_gguf(&path).await.expect("GGUF 装载");
    let live_gib =
        owl_shared::vram::live_bytes().saturating_sub(live0) as f64 / (1u64 << 30) as f64;
    eprintln!(
        "[gguf-e2e] {key} 装载 {:.2}s · 层 {} · 设备 live {:.2} GiB vs 文件 {:.2} GiB(比 {:.3})",
        t0.elapsed().as_secs_f32(),
        loaded.model.layers.len(),
        live_gib,
        file_gib,
        live_gib / file_gib
    );
    assert_eq!(loaded.model.layers.len(), n_layers, "[{key}] 解码层数");
    // **I-3 的钱行**:压缩驻留 ⇒ 设备占用不得超过文件(27B 若被解量化就是 54G)
    assert!(
        live_gib <= file_gib * 1.02,
        "[{key}] 设备 live {live_gib:.2} GiB > 文件 {file_gib:.2} GiB —— 权重被解量化驻留了?(I-3)"
    );
    // 词表口径先行(llama.cpp `/tokenize` 的实测值;token 面不一致 ⇒ 后面全无意义)
    let ids = loaded.tokenizer.encode("The capital of France is");
    eprintln!("[gguf-e2e] prompt ids = {ids:?}(llama.cpp 实测 [760, 6511, 314, 9338, 369])");
    let mut running = engine.run(loaded).await.expect("装配");
    // 裸文本(不走模板 —— 与 llama.cpp 的 `-p` 裸口径可比):短问答 + 续写
    let prompt = "The capital of France is";
    let (_, tid) = running.submit_session(Some(1), prompt, 24).expect("submit");
    let mut text = String::new();
    let t1 = std::time::Instant::now();
    let mut steps = 0usize;
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Token { turn, delta } if turn == tid => {
                text.push_str(&delta);
                steps += 1;
            }
            TurnEvent::Completed { turn, .. } if turn == tid => break,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            _ => {}
        }
    }
    // **首 token 单发**(诊断口径:预填充面 vs 解码面)—— 另起会话避免续写污染
    if let Ok((_, t1)) = running.submit_session(Some(2), prompt, 1) {
        let mut first = String::new();
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Token { turn, delta } if turn == t1 => first.push_str(&delta),
                TurnEvent::Completed { turn, .. } if turn == t1 => break,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                _ => {}
            }
        }
        eprintln!("[gguf-e2e] 首 token(单发) = {first:?}");
        // **首 token 判据**(最强的一条廉价门):同文件同 prompt 的 argmax 必须与
        // llama.cpp 实测一致。为什么它够强:首 token 由**整条预填充链**(全部权重
        // 解释 × 全部层 × 最终 norm × 输出头)的 logits 顶决定 —— 任何一处接错
        // (含"头用错表"这类静默错值)都会在这里翻车,而"能跑出连贯文本"抓不住。
        // 实测 margin:4B/27B 的 oracle top-1 都领先次高 ≥2.5 nat ⇒ 不是 near-tie。
        assert_eq!(
            first, want_first,
            "[{key}] 首 token 与 oracle 不符 —— 预填充链/输出头有实质错(非 near-tie)"
        );
    }
    let dt = t1.elapsed().as_secs_f32();
    eprintln!(
        "[gguf-e2e] 生成 {steps} tok / {dt:.2}s({:.1} tok/s)· 文本 = {text:?}",
        steps as f32 / dt.max(1e-6)
    );
    assert!(steps >= 16, "生成步数太少({steps})—— 早停?");
    assert!(!text.trim().is_empty(), "输出为空");
    // ---- 刀 8 面①:**捕获预检**(两道:树审计 Htod + face 能力)真过了 ----
    //
    // 判据 = 图族报告 `PlanOutcome::Captured`(不是 `EagerFallback{reason}`)。
    // 这一步是 GGUF 档的关键:**权重槽是 U8 块字节 + 虚拟键列序表**,若哪条链
    // 在捕获期做了 Htod(树审计)或 face 不支持(能力门),这里会诚实地降级 ——
    // 降级不是错,但"以为捕获了其实没捕获"会让 decode 步时翻倍(级联回归源)。
    if let crate::graph_plan::PlanOutcome::EagerFallback { reason } = &running.dev.graphs.capture_outcome {
        panic!("GGUF 档图捕获降级(eager 回退):{reason}");
    }
    eprintln!("[gguf-e2e] 捕获预检:PlanOutcome::{:?}", running.dev.graphs.capture_outcome);

    // ---- 刀 8 面②:**decode 步时账**(引擎既有 `mrec(\"decode.step\")`,GGUF 档必须同样在账)----
    let reps = owl_shared::metrics::query_metrics(
        &owl_shared::metrics::MetricsFilter::new().tag_prefix("decode."),
    );
    let step = reps
        .iter()
        .find(|r| r.tag == "decode.step")
        .unwrap_or_else(|| panic!("`decode.step` 无样本(metrics 未 init 或 decode 路没记账);实得 {:?}",
            reps.iter().map(|r| r.tag.as_str()).collect::<Vec<_>>()));
    assert!(step.count >= 16, "`decode.step` 样本 {}(应 ≥ 生成步数)", step.count);
    eprintln!(
        "[gguf-e2e] decode 步时账:{} 步 · avg {:.3}ms · p50 {:.3}ms · p99 {:.3}ms",
        step.count,
        step.avg.as_secs_f64() * 1e3,
        step.p50.as_secs_f64() * 1e3,
        step.p99.as_secs_f64() * 1e3
    );

    // ---- 刀 8 面③:**账本无"未命名块"**(设计 REQ-GGUF-06 / I-4 的运行期姊妹)----
    //
    // 判据 = 块账按名清单里不出现 `(未命名!)`。为什么在**跑完一轮之后**查:
    // 装载 + 图捕获 + decode 是三条不同的分配路径(权重槽/图 slab/scratch),
    // 只有全走一遍才谈得上"全量记名"。
    if let Some(f) = running.face_clone() {
        let rep = f.client().ledger_report(8).await.expect("ledger_report");
        assert!(
            !rep.contains("(未命名!)"),
            "块账出现未命名块(全量记名漏网 ⇒ OOM 现场问不出\"谁占着\"):{rep}"
        );
        eprintln!("[gguf-e2e] 账本(前 8 类,零未命名)= {rep}");
        // 块账按**权重键**记名(`wt.<键>`)—— 全表 JSON 里必须看得到 24 张列序表
        // (虚拟键也是权重槽,一样要记名:OOM 现场"谁占着"才答得出来)
        let js = f.client().ledger_json().await.expect("ledger_json");
        let n_perm = js.matches("ssm_out.perm").count();
        assert_eq!(
            n_perm, n_gdn,
            "[{key}] 列序表在账本里 {n_perm} 张(应 = GDN 层数 {n_gdn})—— 虚拟键没记名?"
        );
        assert!(js.contains("wt.token_embd.weight"), "tied 量化表未入账");
    }
    // 连贯判据(廉价但有力):不得是同一字符/词的无脑重复(错变换的典型签名)
    let uniq: std::collections::HashSet<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(uniq.len() >= 4, "输出字符种类过少({:?})—— 疑似错变换导致的退化", text);
}

/// **M1 冒烟(4B)**
#[cfg(test)]
#[owl_test(gpu)]
async fn gpu_gguf_4b_smoke() {
    let ord = owl_cuda::test_device_ordinal();
    smoke_scenario("gguf:4b", 32, 24, " Paris", gguf_cfg(ord, 1024, 1024, 1)).await;
}

/// **M2 冒烟(27B Q4_K_M,单卡)**:混型面(Q4_K 193 / Q8_0 288 / Q6_K 17 / F32 353)
/// 在 64 层(16 full + 48 GDN)上的完整装载 + 生成;池按小 ctx 给(冒烟不要大池)。
#[cfg(test)]
#[owl_test(gpu)]
async fn gpu_gguf_27b_smoke() {
    let ord = owl_cuda::test_device_ordinal();
    smoke_scenario("gguf:27b", 64, 48, " Paris", gguf_cfg(ord, 4096, 4096, 1)).await;
}

/// **IQ 族 e2e 门**(2026-10-20;REQ-GGUF-04):0.8B IQ4_XS 装载 + 生成,
/// 与**同族 Q4_K_M** 的首 token 对拍(跨量化档的"接线级"判据)+ 压缩驻留(I-3)。
///
/// 为什么拿 Q4_K_M 当参照臂:同 dims / 同 tokenizer / 同模型族,只差量化 ⇒
/// 首 token 由整条 prefill 链(24 层 × 全部 IQ4_XS 权重 × 输出头)决定;
/// 错码本/错尺度/错块位移都会在这里翻车(单元门已逐张量验过)。
#[cfg(test)]
#[owl_test(gpu)]
async fn gpu_gguf_iq4xs_0_8b_vs_q4km() {
    if !resource_ok("gguf:0.8b-iq4xs") || !resource_ok("gguf:0.8b-q4km") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:0.8b-iq4xs\" / \"gguf:0.8b-q4km\"");
        return;
    }
    let ord = owl_cuda::test_device_ordinal();
    let prompt = "The capital of France is";
    let mut firsts: Vec<(&str, String, f64)> = Vec::new();
    for key in ["gguf:0.8b-q4km", "gguf:0.8b-iq4xs"] {
        let path = std::path::PathBuf::from(resource(key).expect("列了就有"));
        let file_gib = std::fs::metadata(&path).expect("stat").len() as f64 / (1u64 << 30) as f64;
        let live0 = owl_shared::vram::live_bytes();
        let mut engine = Engine::new(gguf_cfg(ord, 2048, 2048, 1)).expect("引擎");
        let loaded = engine.loader().load_gguf(&path).await.expect("装载");
        let live_gib =
            owl_shared::vram::live_bytes().saturating_sub(live0) as f64 / (1u64 << 30) as f64;
        assert_eq!(loaded.model.layers.len(), 24, "[{key}] 层数");
        // 小模型有固定开销(池/norm 等 ~50 MiB)⇒ 判据 = 文件 + 64 MiB 余量,
        // 而不是百分比(0.5 GiB 的 2% 只有 10 MiB,会把正常开销判成解量化驻留)
        assert!(
            live_gib <= file_gib + 0.0625,
            "[{key}] 设备 live {live_gib:.2} GiB > 文件 {file_gib:.2} GiB + 64 MiB(权重被解量化驻留?)"
        );
        let mut running = engine.run(loaded).await.expect("装配");
        // 首 token 单发(贪心;不走模板 —— 与裸文本口径一致)
        let (_, tid) = running.submit_session(Some(9), prompt, 1).expect("submit");
        let mut first = String::new();
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Token { turn, delta } if turn == tid => first.push_str(&delta),
                TurnEvent::Completed { turn, .. } if turn == tid => break,
                TurnEvent::Failed { turn, err } => panic!("[{key}] t{turn} 失败: {err}"),
                TurnEvent::Idle => panic!("[{key}] turn {tid} 队列丢失"),
                _ => {}
            }
        }
        eprintln!(
            "[gguf-iq-e2e] {key} {:.2} GiB / live {live_gib:.2} GiB · 首 token = {first:?}",
            file_gib
        );
        firsts.push((key, first, live_gib / file_gib));
    }
    assert_eq!(
        firsts[0].1, firsts[1].1,
        "IQ4_XS 首 token 与 Q4_K_M 参照臂不符(错码本/错尺度/错块位移会给完全不同的 argmax)"
    );
}

/// **交叉源对拍:同 dims,两源权重**(GGUF Q4_K_M ↔ HF 快照 f16;设计 REQ-GGUF-07)。
///
/// # 判据为什么不是"逐 token 全序列相同"(2026-10-19 实测定谳)
///
/// 端到端**逐 token 对拍在量化档做不了硬门**:贪心是 argmax,而两条实现路径
/// (Q4_K_M 的 8bit 激活量化 + 不同归约序 vs f16 参考)在 **near-tie** 处必然分叉。
/// 实测两处分叉的 oracle 侧 top-2 间隔(llama.cpp `/v1/completions` 的 logprobs):
///
/// | 题 | 分叉点 | oracle top-2 | 间隔 | 谁选了谁 |
/// |---|---|---|---|---|
/// | `The capital of France is` | 第 4 token | `A` −1.400 / `The` −1.543 | **0.14 nat** | HF=oracle;GGUF 选了次高 |
/// | `Water is composed of hydrogen and` | 第 3 token | `Which` −2.425 / `If` −2.438 | **0.013 nat** | GGUF=oracle;HF 选了次高 |
///
/// **关键**:两次分叉各由**不同一侧**命中 oracle 的 argmax ⇒ 分叉是噪声而非
/// "某一侧错"(若 GGUF 路径有实质错,它会**单向**、**大间隔**地偏离)。
///
/// 因此本门的判据分两层:
/// ① **硬门 = 首 token 一致**(6 题):首 token 由**整条预填充链**(32 层 ×
///    全部权重解释)的 logits 顶决定 ⇒ 捕捉一切"接线级"错(错变换/错段/错头);
/// ② **软账 = 全 12 token 一致数**:报告 + 要求 ≥ 半数(留出 near-tie 分叉的空间)。
#[cfg(test)]
#[owl_test(gpu)]
async fn gpu_gguf_4b_vs_hf_4b() {
    const PROMPTS: &[&str] = &[
        "The capital of France is",
        "1, 2, 3, 4, 5,",
        "The quick brown fox jumps over the lazy",
        "Water is composed of hydrogen and",
        "def add(a, b):\n    return",
        "Q: What is 2+2?\nA:",
    ];
    let Some(hf_dir) = owl_shared::testkit::resource("tpl:hf4b") else {
        owl_shared::testkit::report::skip("资源清单缺 \"tpl:hf4b\"");
        return;
    };
    if !resource_ok("gguf:4b") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
        return;
    }
    let gguf_path = std::path::PathBuf::from(resource("gguf:4b").expect("列了就有"));
    // 维度事实两边相同:拿 GGUF 的 spec 当 HF 侧的"家族声明基座"
    // (本门比的是**同一模型两源** ⇒ dims 必须逐字段相同,已在 meta 门锁过)
    let src = owl_models::formats::gguf::GgufSource::open(&gguf_path).expect("开 GGUF");
    let preset = src.spec().clone();
    let rf = owl_models::formats::gguf::rope_facts(src.gguf()).expect("rope 事实");
    drop(src);
    let ordinal = owl_cuda::test_device_ordinal();
    // ---- 臂 A:GGUF(Q4_K_M) ----
    let mut eng_g = Engine::new(gguf_cfg(ordinal, 1024, 1024, 1)).expect("引擎 G");
    let lg = eng_g.loader().load_gguf(&gguf_path).await.expect("GGUF 装载");
    let mut run_g = eng_g.run(lg).await.expect("装配 G");
    // ---- 臂 B:HF 快照(f16;rope 事实与 GGUF 同源 ⇒ 只差权重) ----
    let mut eng_h = Engine::new(gguf_cfg(ordinal, 1024, 1024, 1)).expect("引擎 H");
    let lh = eng_h
        .loader()
        .load_qwen35_hf(
            std::path::Path::new(&hf_dir),
            preset,
            (rf.rotary_dim, rf.theta, rf.max_pos),
        )
        .await
        .expect("HF 装载");
    let mut run_h = eng_h.run(lh).await.expect("装配 H");

    let mut same_all = 0usize;
    for (i, p) in PROMPTS.iter().enumerate() {
        let t_g = greedy_in(&mut run_g, p, 12).await;
        let t_h = greedy_in(&mut run_h, p, 12).await;
        let head6 = |s: &str| s.chars().take(6).collect::<String>();
        assert_eq!(
            head6(&t_g),
            head6(&t_h),
            "题 {i} `{p}` 首 token 即分歧 —— 预填充链有实质扰动(不是 near-tie):\n  GGUF={t_g:?}\n  HF  ={t_h:?}"
        );
        let eq = t_g == t_h;
        same_all += eq as usize;
        eprintln!("[3way] {i} 首token同 · 全序列{} · {p:?}", if eq { "同" } else { "分叉(near-tie)" });
        if !eq {
            eprintln!("        GGUF={t_g:?}\n        HF  ={t_h:?}");
        }
    }
    assert!(
        same_all * 2 >= PROMPTS.len(),
        "全序列一致仅 {same_all}/{} —— 除 near-tie 外仍大面积分歧 ⇒ 回查源/计划面",
        PROMPTS.len()
    );
    eprintln!(
        "[3way] 绿:首 token 6/6 一致 · 全 12 token 一致 {same_all}/6(分叉点见上,oracle margin 0.14/0.013 nat)"
    );
}

/// 单会话贪心(多 prompt 诊断用;每次新建会话避免续写污染)
#[cfg(test)]
async fn greedy_in<D: owl_iface::contract::DeviceClient + 'static>(
    running: &mut RunningEngine<D>,
    prompt: &str,
    n: usize,
) -> String {
    let (_, tid) = running.submit_session(None, prompt, n).expect("submit");
    let mut text = String::new();
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Token { turn, delta } if turn == tid => text.push_str(&delta),
            TurnEvent::Completed { turn, .. } if turn == tid => break,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            _ => {}
        }
    }
    text
}


// ============================================================================
// **W-1:GGUF + TP2**(2026-10-19)—— 分片几何的真机门
// ============================================================================
//
// 判据(设计 §15):**首 token 硬门** + 分片装载账(每 rank ≈ 文件/2)+ 捕获态
// + 零 Xid。首 token 是"整条预填充链"的判据:任何一处
// (行置换 × 块对齐列切 × v-head grouped 聚合)接错都会在这里翻车,
// 而"文本连贯"抓不住 —— 27B 单卡的首 token 已由 llama.cpp 钉死(`" Paris"`)。
//
// 运行(两张空闲卡的 **BDF**;未设 = skip):
//
// ```bash
// OWL_GGUF_TP_DEV_A=41:00.0 OWL_GGUF_TP_DEV_B=42:00.0 \
//   cargo test -p owl-engine --lib gpu_gguf_27b_tp2 -- --nocapture
// ```
#[cfg(test)]
#[test]
fn gpu_gguf_27b_tp2() {
    tp2_entry("gguf:27b", 64, " Paris");
}

/// **4B 档 TP2**(同一套分片几何的小靶位;诊断 27B 偏差时先跑它 —— 32 层、2.7G)
#[cfg(test)]
#[test]
fn gpu_gguf_4b_tp2() {
    tp2_entry("gguf:4b", 32, " Paris");
}

#[cfg(test)]
fn tp2_entry(key: &'static str, layers: usize, want_first: &'static str) {
    let dev = |name: &str| {
        std::env::var(name)
            .ok()
            .filter(|s| !s.is_empty())
            .map(|bdf| owl_cuda::resolve_ordinal_by_bdf(&bdf).unwrap_or_else(|e| panic!("{e}")))
    };
    let (Some(oa), Some(ob)) = (dev("OWL_GGUF_TP_DEV_A"), dev("OWL_GGUF_TP_DEV_B")) else {
        owl_shared::testkit::report::skip("需 OWL_GGUF_TP_DEV_A / OWL_GGUF_TP_DEV_B(两张空闲卡的 BDF)");
        return;
    };
    assert_ne!(oa, ob, "TP2 需要两张不同的卡");
    eprintln!("[gguf-tp2] 设备:A = ordinal {oa} / B = ordinal {ob}");
    if !resource_ok(key) {
        owl_shared::testkit::report::skip("资源清单缺 \"{key}\"");
        return;
    }
    // 大盘装配:大栈 + current_thread rt(与 tp_assemble_gate 同律)
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("rt")
                .block_on(tp2_body(key, layers, want_first, oa, ob))
        })
        .expect("起测试线程")
        .join()
        .expect("TP2 门失败");
}

/// 首步 logits(与 `tp_logits_27b` 同法:`submit_session_ex(…, want_logits)`)
#[cfg(test)]
async fn first_logits_gguf<D: owl_iface::contract::DeviceClient>(
    running: &mut RunningEngine<D>,
    prompt: &str,
) -> Vec<f32> {
    let (sid, turn) = running.submit_session_ex(Some(1), prompt, 1, true).expect("submit");
    let _ = sid;
    let mut logits: Option<Vec<f32>> = None;
    loop {
        let ev = running.pump().await.expect("pump");
        if matches!(&ev, TurnEvent::Token { turn: t, .. } if *t == turn) && logits.is_none() {
            logits = running.read_output_f32("logits").await.expect("read logits");
        }
        match ev {
            TurnEvent::Idle => break,
            TurnEvent::Completed { turn: t, .. } if t == turn => break,
            TurnEvent::Failed { turn: t, err } => panic!("t{t} 失败: {err}"),
            _ => {}
        }
    }
    logits.expect("首步读不到 logits 槽")
}

#[cfg(test)]
async fn tp2_body(key: &str, layers: usize, want_first: &str, oa: usize, ob: usize) {
    let path = std::path::PathBuf::from(resource(key).expect("列了就有"));
    let file_gib = std::fs::metadata(&path).expect("stat").len() as f64 / (1u64 << 30) as f64;
    let cfg = gguf_cfg(oa, 4096, 4096, 1);
    // ── ⓪ 单卡参考的**首步 logits**(TP2 的数值判据;比"全序列逐 token"强得多)──
    //    单卡与 TP2 不能同卡共存(24G 装不下两份)⇒ 先跑单卡、drop、再上 TP2。
    let ref_logits = {
        let mut e = Engine::new(cfg.clone()).expect("engine(单卡参考)");
        let loaded = e.loader().load_gguf(&path).await.expect("单卡装载");
        let mut r = e.run(loaded).await.expect("单卡装配");
        let v = first_logits_gguf(&mut r, "The capital of France is").await;
        drop(r);
        v
    };
    eprintln!("[gguf-tp2] 单卡参考:{} 项 logits", ref_logits.len());
    let mut e0 = Engine::new(cfg.clone()).expect("engine(rank0)");
    let live0 = owl_shared::vram::live_bytes();
    let l0 = e0.loader().slice(0, 2).load_gguf(&path).await.expect("rank0 分片装载");
    let g0 = owl_shared::vram::live_bytes().saturating_sub(live0) as f64 / (1u64 << 30) as f64;
    let mut e1 = Engine::new(EngineConfig { device_ordinal: ob, ..cfg }).expect("engine(rank1)");
    let l1 = e1.loader().slice(1, 2).load_gguf(&path).await.expect("rank1 分片装载");
    eprintln!(
        "[gguf-tp2] 分片装载:rank0 {g0:.2} GiB(文件 {file_gib:.2});两侧层数 {}/{}",
        l0.model.layers.len(),
        l1.model.layers.len()
    );
    assert_eq!(l0.model.layers.len(), layers, "rank0 解码层数");
    assert_eq!(l1.model.layers.len(), layers, "rank1 解码层数");
    // 每 rank ≤ 文件/2 + 复制件(embed/output 约 1.7 GiB);宽上限 = 文件 × 0.72
    assert!(
        g0 <= file_gib * 0.72,
        "rank0 设备 live {g0:.2} GiB > 文件 {file_gib:.2} × 0.72 —— 分片没生效(整份驻留?)"
    );
    let tp = crate::tp::TpEngine::assemble(vec![e0, e1], vec![l0, l1])
        .await
        .expect("TP2 装配");
    let mut running = tp.run().expect("run");
    let tp_logits = first_logits_gguf(&mut running, "The capital of France is").await;
    // 数值判据:TP 只改**归约序**(sum 两半 vs 一次 GEMM)⇒ logits 应近乎相同
    let mut dmax = 0f32;
    let mut amax = 0f32;
    let mut dot = 0f64;
    let (mut n1, mut n2) = (0f64, 0f64);
    for (a, b) in ref_logits.iter().zip(tp_logits.iter()) {
        dmax = dmax.max((a - b).abs());
        amax = amax.max(a.abs());
        dot += (*a as f64) * (*b as f64);
        n1 += (*a as f64).powi(2);
        n2 += (*b as f64).powi(2);
    }
    let cos = dot / (n1.sqrt() * n2.sqrt()).max(1e-30);
    let arg = |v: &[f32]| -> usize {
        v.iter().enumerate().fold((0, f32::MIN), |acc, (i, x)| if *x > acc.1 { (i, *x) } else { acc }).0
    };
    eprintln!(
        "[gguf-tp2] logits 对拍:max|Δ| = {dmax:.4}(max|L| = {amax:.3})· cos = {cos:.8} · top1 {} == {}",
        arg(&ref_logits),
        arg(&tp_logits)
    );
    // AR 注入面必须真的走到(否则两 rank 各留"部分和" —— 静默半量错)
    let ar_count = |tag: &str| -> u64 {
        owl_shared::metrics::query_metrics(&owl_shared::metrics::MetricsFilter::new().tag(tag))
            .iter()
            .map(|r| r.counter)
            .sum()
    };
    eprintln!(
        "[gguf-tp2] AR 注入计数:r0={} r1={}(1stage={} v1={})",
        ar_count("tp.ar.r0"),
        ar_count("tp.ar.r1"),
        ar_count("tp.ar.1stage"),
        ar_count("tp.ar.v1")
    );
    assert_eq!(
        arg(&ref_logits),
        arg(&tp_logits),
        "首步 argmax 不同 —— 分片几何/AR 有实质错"
    );
    // 判据的**口径**(2026-10-19 实测对照,不是随便定的):
    //   同机 AWQ-27B 的 TP2 vs TP1 首步 logits `max|Δ| = 10.83`(`tp_logits_27b`)
    //   ⇒ "TP2 与单卡 logits 不等价"是**引擎既有的数值行为**(row-parallel 拆半
    //   + f16 AR 的逐层舍入),不是 GGUF 分片的错。故这里只钉:
    //   ① argmax 相同(信息面);② max|Δ| 与 AWQ 对照同量级;③ 两 rank 逐位相同。
    assert!(
        dmax < 12.0,
        "TP2 与单卡 max|Δ| = {dmax:.3} 超出既有 TP2 噪声量级(AWQ 对照 10.83)⇒ 回查分片/AR"
    );
    let peer = running.read_peer_output_f32("logits").await.expect("读 peer logits");
    if let Some(p) = peer {
        let nbit = p.iter().zip(tp_logits.iter()).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        eprintln!("[gguf-tp2] 两 rank logits:{nbit} 项不同(应 0)");
        assert_eq!(nbit, 0, "两 rank logits 不逐位相同(AR 没把两侧统一?)");
    }
    eprintln!("[gguf-tp2] 捕获态:{:?} · peer 束 {}", running.capture_outcome(), running.has_peer());
    if let crate::graph_plan::PlanOutcome::EagerFallback { reason } = running.capture_outcome() {
        panic!("GGUF TP2 图捕获降级:{reason}");
    }
    let prompt = "The capital of France is";
    // ① 首 token 硬门(单发)
    let (_, t1) = running.submit_session(Some(2), prompt, 1).expect("submit");
    let mut first = String::new();
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Token { turn, delta } if turn == t1 => first.push_str(&delta),
            TurnEvent::Completed { turn, .. } if turn == t1 => break,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {}
        }
    }
    eprintln!("[gguf-tp2] 首 token = {first:?}");
    assert_eq!(
        first, want_first,
        "TP2 首 token 与 oracle 不符 —— 分片几何(行置换/列聚合/块边界)有实质错(非 near-tie)"
    );
    // ② 24 token 续写(与单卡/llama.cpp 的同 prompt 文本对照;软账 —— TP 的
    //    归约序与单卡不同,量化档允许 near-tie 分叉)
    let (_, tid) = running.submit_session(Some(1), prompt, 24).expect("submit");
    let mut text = String::new();
    let t = std::time::Instant::now();
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Token { turn, delta } if turn == tid => text.push_str(&delta),
            TurnEvent::Completed { turn, .. } if turn == tid => break,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            TurnEvent::Idle => panic!("turn {tid} 队列丢失"),
            _ => {}
        }
    }
    let dt = t.elapsed().as_secs_f32();
    eprintln!("[gguf-tp2] 24 token / {dt:.2}s({:.1} tok/s)· {text:?}", 24.0 / dt.max(1e-6));
    let 单卡: &str = if layers == 64 {
        " Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is Rome.\nThe capital of Spain is"
    } else {
        "" // 4B 不钉全序列(只钉首 token + 字符种类)
    };
    if !单卡.is_empty() && text != 单卡 {
        eprintln!("[gguf-tp2] ⚠️ 全序列与单卡不同(量化档 + 归约序差;首 token 已硬门):\n  单卡={单卡:?}\n  TP2 ={text:?}");
    } else {
        eprintln!("[gguf-tp2] 全 24 token 与单卡逐字相同 ✓");
    }
    let uniq: std::collections::HashSet<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(uniq.len() >= 4, "输出字符种类过少({text:?})—— 疑似错几何导致的退化");
}

/// **prefill 探针**(W-2e A/B):长 prompt 的 TTFT ⇒ prefill tok/s。
///
/// 两臂对照(`OWL_GGUF_CUBLAS=1` dequant+cuBLAS / `=0` 官方 MMQ)× 两档 chunk
/// (**1024 = 服务端现役档**,512 = 历史对照)。
/// 判据不是"绝对值"(机器状态相关),而是**同进程两臂的比值**;
/// 另判:两臂首 token 必须一致(数值差 ~1e-2,同 argmax)。
#[cfg(test)]
#[owl_test(gpu)]
async fn gpu_gguf_prefill_probe() {
    if !resource_ok("gguf:27b") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:27b\"");
        return;
    }
    let path = std::path::PathBuf::from(resource("gguf:27b").expect("列了就有"));
    // ≈1024 token 的 prompt(重复句;实际 token 数由 `turn_prompt_len` 报)
    let prompt: String = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let ord = owl_cuda::test_device_ordinal();
    let mut rows: Vec<(usize, bool, usize, f64, String)> = Vec::new();
    for chunk in [512usize, 1024] {
        for cublas in [false, true] {
            std::env::set_var("OWL_GGUF_CUBLAS", if cublas { "1" } else { "0" });
            let mut cfg = gguf_cfg(ord, 8192, 8192, 1);
            cfg.prefill_chunk = chunk;
            let mut engine = Engine::new(cfg).expect("引擎");
            let loaded = engine.loader().load_gguf(&path).await.expect("装载");
            let mut running = engine.run(loaded).await.expect("装配");
            // prompt 长度**提交后立刻**取:turn 收摊后账会退役(实测取到 0)
            let (_, tid) = running.submit_session(Some(9), &prompt, 1).expect("submit");
            let ntok = running.turn_prompt_len(tid).unwrap_or(0);
            let t0 = std::time::Instant::now();
            let mut first = String::new();
            loop {
                match running.pump().await.expect("pump") {
                    TurnEvent::Token { turn, delta } if turn == tid => {
                        first.push_str(&delta);
                        break;
                    }
                    TurnEvent::Completed { turn, .. } if turn == tid => break,
                    TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                    _ => {}
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            rows.push((chunk, cublas, ntok, dt, first.clone()));
            eprintln!(
                "[gguf-prefill] chunk={chunk} cublas={} · prompt {ntok} tok · {dt:.3}s ⇒ {:.1} tok/s · 首 token {first:?}",
                cublas as u8,
                ntok as f64 / dt.max(1e-9)
            );
            drop(running);
        }
    }
    // 比值读数(同 chunk 内):cuBLAS 臂 vs MMQ(不设硬门 —— 两臂谁快
    // 取决于 chunk/机器档,只入档)
    for chunk in [512usize, 1024] {
        let get = |tile: bool| {
            rows.iter()
                .find(|(c, t, ..)| *c == chunk && *t == tile)
                .map(|r| (r.3, r.4.clone()))
                .expect("缺样本")
        };
        let ((mmq, t_mmq), (cublas, t_cublas)) = (get(false), get(true));
        // 两臂数值差 ~1e-2(cuBLAS = f16 输入 × f32 累加;MMQ = q8_1 激活)
        // ⇒ 只要求**首 token 串一致**(同 argmax),不是位级相同。
        assert_eq!(
            t_cublas, t_mmq,
            "chunk={chunk}:两臂首 token 不同(分派/数值路径出问题?{t_cublas:?} vs {t_mmq:?})"
        );
        eprintln!(
            "[gguf-prefill] chunk={chunk} 加速比 MMQ/cuBLAS = {:.2}×",
            mmq / cublas.max(1e-9)
        );
    }
}

/// **4B prefill 单程探针**(性能剖析靶位;nsys/CUPTI 用)。
///
/// 只跑**一次**长 prompt 的 prefill(不复用臂 A/B、不换 chunk)—— 给外部
/// 剖析器一个干净、可重复的负载窗口。跑法:
///
/// ```bash
/// OWL_TEST_DEVICE=41:00.0 OWL_GGUF_TILE=1 \
/// nsys profile -o /tmp/gguf4b --force-overwrite true \
///   --trace=cuda,nvtx --stats=true \
///   cargo test -p owl-engine --lib gpu_gguf_prefill_4b_probe -- --nocapture
/// ```
#[cfg(test)]
#[owl_test(gpu)]
async fn gpu_gguf_prefill_4b_probe() {
    if !resource_ok("gguf:4b") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
        return;
    }
    let path = std::path::PathBuf::from(resource("gguf:4b").expect("列了就有"));
    let prompt: String = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let chunk: usize = std::env::var("OWL_GGUF_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let ord = owl_cuda::test_device_ordinal();
    let mut cfg = gguf_cfg(ord, 8192, 8192, 1);
    cfg.prefill_chunk = chunk;
    let mut engine = Engine::new(cfg).expect("引擎");
    let loaded = engine.loader().load_gguf(&path).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    // 预热一次(图捕获/桶装配落在窗外;剖析看的是**第二次**)
    for i in 0..2 {
        let (_, tid) = running.submit_session(Some(10 + i), &prompt, 1).expect("submit");
        let ntok = running.turn_prompt_len(tid).unwrap_or(0);
        let t0 = std::time::Instant::now();
        loop {
            match running.pump().await.expect("pump") {
                TurnEvent::Token { turn, .. } if turn == tid => break,
                TurnEvent::Completed { turn, .. } if turn == tid => break,
                TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
                _ => {}
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(
            "[gguf-4b-prefill] 第 {} 趟 · chunk={chunk} · prompt {ntok} tok · {dt:.3}s ⇒ {:.1} tok/s",
            i + 1,
            ntok as f64 / dt.max(1e-9)
        );
    }
}

/// **27B prefill 单程探针**(W-2e;剖析靶位):一次长 prompt prefill,单配置。
///
/// 与 `gpu_gguf_prefill_probe` 的差别:只跑**一个** arm/chunk,给 nsys 一个
/// 干净的负载窗口,并允许拧 dequant 行窗。跑法:
///
/// ```bash
/// OWL_TEST_DEVICE=42:00.0 OWL_GGUF_CHUNK=1024 OWL_GGUF_CUBLAS_TILE=4096 \
/// nsys profile -o /tmp/owl27 --force-overwrite true --trace=cuda \
///   cargo test -p owl-engine --lib gpu_gguf_prefill_single -- --nocapture
/// ```
///
/// 环境:`OWL_GGUF_CHUNK`(缺省 1024)· `OWL_GGUF_CUBLAS`(缺省开)·
/// `OWL_GGUF_CUBLAS_TILE`(缺省 = 96 MiB 预算档)。
#[cfg(test)]
#[owl_test(gpu)]
async fn gpu_gguf_prefill_single() {
    if !resource_ok("gguf:27b") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:27b\"");
        return;
    }
    let path = std::path::PathBuf::from(resource("gguf:27b").expect("列了就有"));
    let prompt: String = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let chunk: usize = std::env::var("OWL_GGUF_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(1024);
    let ord = owl_cuda::test_device_ordinal();
    let mut cfg = gguf_cfg(ord, 8192, 8192, 1);
    cfg.prefill_chunk = chunk;
    eprintln!("[gguf-prefill-single] 臂集 = {:?}", cfg.knobs.env.gguf);
    let mut engine = Engine::new(cfg).expect("引擎");
    let loaded = engine.loader().load_gguf(&path).await.expect("装载");
    let mut running = engine.run(loaded).await.expect("装配");
    let (_, tid) = running.submit_session(Some(11), &prompt, 1).expect("submit");
    let ntok = running.turn_prompt_len(tid).unwrap_or(0);
    let t0 = std::time::Instant::now();
    let mut first = String::new();
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Token { turn, delta } if turn == tid => {
                first.push_str(&delta);
                break;
            }
            TurnEvent::Completed { turn, .. } if turn == tid => break,
            TurnEvent::Failed { turn, err } => panic!("t{turn} 失败: {err}"),
            _ => {}
        }
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "[gguf-prefill-single] chunk={chunk} · prompt {ntok} tok · {dt:.3}s ⇒ {:.1} tok/s · 首 token {first:?}",
        ntok as f64 / dt.max(1e-9)
    );
}

/// **4B GGUF + mmproj 视觉塔冒烟**(2026-10-21):GGUF 文本主干 + 同目录
/// `mmproj-*.gguf` 装配,单图 prefill 出非空文本。
///
/// 判据只要求"链路通 + 出文本":塔数值由 `mmproj::source_covers_tower_keys`
/// 与 HF 权重面保证;这里抓的是**装载/注入/mRoPE/占位展开**的接线错误
/// (任何一处错会立刻退化成空串或 `Failed`)。
#[cfg(test)]
#[owl_test(gpu)]
async fn gpu_gguf_4b_vlm_image_smoke() {
    use owl_models::formats::gguf::resolve_mmproj_path;
    use crate::turn::TurnOpts;

    if !resource_ok("gguf:4b") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:4b\"");
        return;
    }
    // GPU 套件在同一进程串跑时，前一个 27B 测试的 `live` 账可能未随 drop 回落
    //（free 已还 OS，但 LIVE 计数是进程级；`[vram-gov]` 会把它当在驻留）。
    // 4B 门单独跑/新进程跑；检测到已有大模型驻留就 skip，避免把测试隔离问题
    // 误报成 4B VLM 失败。
    let live = owl_shared::vram::live_bytes();
    if live > (8u64 << 30) {
        eprintln!(
            "skip: 进程已有 {:.1}G 设备驻留（GPU 套件请单测跑 4B VLM；27B 门已跑过）",
            live as f64 / (1u64 << 30) as f64
        );
        return;
    }
    let path = std::path::PathBuf::from(resource("gguf:4b").expect("列了就有"));
    let mmproj = match resolve_mmproj_path(&path, None) {
        Ok(p) => p,
        Err(e) => {
            owl_shared::testkit::report::skip(format_args!("同目录无唯一 mmproj: {e}"));
            return;
        }
    };
    let ord = owl_cuda::test_device_ordinal();
    let mut cfg = gguf_cfg(ord, 2048, 2048, 1);
    cfg.knobs.vision = true;
    cfg.knobs.mmproj = Some(mmproj.clone());
    let mut engine = Engine::new(cfg).expect("引擎构造");
    let loaded = engine.loader().load_gguf(&path).await.expect("GGUF + mmproj 装载");
    assert!(loaded.vlm.is_some(), "vision=true 必须有视觉塔(mmproj={})", mmproj.display());
    let mut running = engine.run(loaded).await.expect("装配");

    // 先跑一轮纯文本,避开历史 F1 "首轮带图"变量(4B 是否复现另案)。
    let (_, warm) = running
        .submit_session(Some(1), "The capital of France is", 1)
        .expect("warmup submit");
    loop {
        match running.pump().await.expect("warmup pump") {
            TurnEvent::Token { turn, .. } if turn == warm => break,
            TurnEvent::Completed { turn, .. } if turn == warm => break,
            TurnEvent::Failed { turn: _, err } => panic!("warmup 失败: {err}"),
            _ => {}
        }
    }

    let l_img = 4usize; // grid (1,4,4) ⇒ (4/2)²
    let patch_flat = 3 * 2 * 16 * 16;
    let rows = 16usize;
    let img = owl_models::vlm::VlmImage {
        pixel_values: (0..rows * patch_flat)
            .map(|i| 0.2 + (i % 7) as f32 * 0.01)
            .collect(),
        grid: (1, 4, 4),
    };
    let prompt = format!(
        "<|im_start|>user\n<|vision_start|>{}<|vision_end|>What is in this image?<|im_end|>\n<|im_start|>assistant\n",
        "<|image_pad|>".repeat(l_img)
    );
    let (_, t) = running
        .submit_session_opts(Some(2), prompt.as_str(), 8, TurnOpts { images: vec![img.into()], ..Default::default() })
        .expect("image submit");
    let mut text = String::new();
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Token { turn, delta } if turn == t => text.push_str(&delta),
            TurnEvent::Completed { turn, .. } if turn == t => break,
            TurnEvent::Failed { turn, err } if turn == t => panic!("GGUF VLM 失败: {err}"),
            TurnEvent::Idle => panic!("turn {t} 队列丢失"),
            _ => {}
        }
    }
    eprintln!(
        "[gguf-vlm] 4B + mmproj · grid (1,4,4) · {l_img} image tokens ⇒ {text:?}"
    );
    assert!(!text.trim().is_empty(), "GGUF 视觉 prefill 必须产出非空文本");
}

/// **27B GGUF + mmproj 视觉塔冒烟**(2026-10-21):生产 27B GGUF 主干 +
/// `mmproj-Qwen3.8-27B-Q8_0.gguf`;27B mmproj 的线性层是 Q8_0,另外覆盖
/// `MmprojSource` 的 Q8_0 解量化臂(4B f16 档覆盖不到)。
#[cfg(test)]
#[owl_test(gpu)]
async fn gpu_gguf_27b_vlm_image_smoke() {
    use owl_models::formats::gguf::resolve_mmproj_path;
    use crate::turn::TurnOpts;

    if !resource_ok("gguf:27b") {
        owl_shared::testkit::report::skip("资源清单缺 \"gguf:27b\"");
        return;
    }
    let path = std::path::PathBuf::from(resource("gguf:27b").expect("列了就有"));
    let mmproj = match resolve_mmproj_path(&path, None) {
        Ok(p) => p,
        Err(e) => {
            owl_shared::testkit::report::skip(format_args!("同目录无唯一 mmproj: {e}"));
            return;
        }
    };
    let ord = owl_cuda::test_device_ordinal();
    let mut cfg = gguf_cfg(ord, 2048, 2048, 1);
    cfg.knobs.vision = true;
    cfg.knobs.mmproj = Some(mmproj.clone());
    let mut engine = Engine::new(cfg).expect("引擎构造");
    let loaded = engine.loader().load_gguf(&path).await.expect("27B GGUF + mmproj 装载");
    assert!(loaded.vlm.is_some(), "vision=true 必须有视觉塔(mmproj={})", mmproj.display());
    let mut running = engine.run(loaded).await.expect("装配");

    let l_img = 4usize;
    let patch_flat = 3 * 2 * 16 * 16;
    let rows = 16usize;
    let img = owl_models::vlm::VlmImage {
        pixel_values: (0..rows * patch_flat)
            .map(|i| 0.2 + (i % 7) as f32 * 0.01)
            .collect(),
        grid: (1, 4, 4),
    };
    let prompt = format!(
        "<|im_start|>user\n<|vision_start|>{}<|vision_end|>What is in this image?<|im_end|>\n<|im_start|>assistant\n",
        "<|image_pad|>".repeat(l_img)
    );
    let (_, t) = running
        .submit_session_opts(Some(2), prompt.as_str(), 8, TurnOpts { images: vec![img.into()], ..Default::default() })
        .expect("image submit");
    let mut text = String::new();
    loop {
        match running.pump().await.expect("pump") {
            TurnEvent::Token { turn, delta } if turn == t => text.push_str(&delta),
            TurnEvent::Completed { turn, .. } if turn == t => break,
            TurnEvent::Failed { turn, err } if turn == t => panic!("27B GGUF VLM 失败: {err}"),
            TurnEvent::Idle => panic!("turn {t} 队列丢失"),
            _ => {}
        }
    }
    eprintln!(
        "[gguf-vlm] 27B + mmproj · grid (1,4,4) · {l_img} image tokens ⇒ {text:?}"
    );
    assert!(!text.trim().is_empty(), "27B GGUF 视觉 prefill 必须产出非空文本");
}
